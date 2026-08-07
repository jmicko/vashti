#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

for command in cargo curl jq openssl podman rsync sha256sum tar; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "$command is required for the managed-update smoke test" >&2
        exit 1
    fi
done

signing_key="${VASHTI_RELEASE_SIGNING_KEY:-$repo_root/.private/keys/vashti-release-signing-key.pem}"
if [[ ! -f "$signing_key" ]]; then
    echo "missing release signing key: $signing_key" >&2
    echo "set VASHTI_RELEASE_SIGNING_KEY to run this local release-path test" >&2
    exit 1
fi

current_version="$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)"
candidate_version="$(awk -F. '{ print $1 "." $2 "." ($3 + 1) }' <<<"$current_version")"
rollback_version="$(awk -F. '{ print $1 "." $2 "." ($3 + 2) }' <<<"$current_version")"
container_name="vashti-update-smoke-$$"
test_root="$(mktemp -d)"
hub_url=""
app_url=""

cleanup() {
    local status=$?
    if (( status != 0 )) && podman container exists "$container_name"; then
        if [[ -f "$test_root/app.log" ]]; then
            echo "Vashti test service log:" >&2
            tail -n 80 "$test_root/app.log" >&2 || true
        fi
        echo "Podman container log:" >&2
        podman logs "$container_name" >&2 || true
    fi
    podman rm -f "$container_name" >/dev/null 2>&1 || true
    rm -rf "$test_root"
    return "$status"
}
trap cleanup EXIT

wait_for_url() {
    local url="$1"
    local attempts="${2:-100}"
    for ((attempt = 1; attempt <= attempts; attempt += 1)); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    echo "timed out waiting for $url" >&2
    return 1
}

package_binary() {
    local binary="$1"
    local version="$2"
    local archive="$3"
    local package_dir="$test_root/package-$version/vashti-v$version-linux-x86_64"
    mkdir -p "$package_dir"
    cp "$binary" "$package_dir/vashti"
    chmod 0755 "$package_dir/vashti"
    tar -C "$(dirname "$package_dir")" -czf "$archive" "$(basename "$package_dir")"
}

sign_artifact() {
    local version="$1"
    local artifact="$2"
    local message_file="$test_root/signing-message.txt"
    local size_bytes sha256
    size_bytes="$(wc -c < "$artifact" | tr -d '[:space:]')"
    sha256="$(sha256sum "$artifact" | awk '{ print $1 }')"
    printf '%s\nversion=v%s\ntarget=linux-x86_64\nfilename=%s\nsha256=%s\nsize_bytes=%s\n' \
        'vashti-update-manifest-v1' \
        "$version" \
        "$(basename "$artifact")" \
        "$sha256" \
        "$size_bytes" > "$message_file"
    openssl pkeyutl -sign -rawin -inkey "$signing_key" -in "$message_file" \
        | openssl base64 -A
}

new_hub_upload_key() {
    curl -fsS -b "$test_root/hub-cookies.txt" -X POST \
        "$hub_url/api/admin/upload-key" | jq -er '.token'
}

upload_release() {
    local version="$1"
    local artifact="$2"
    local signature upload_key
    signature="$(sign_artifact "$version" "$artifact")"
    upload_key="$(new_hub_upload_key)"
    curl -fsS \
        -H "Authorization: Bearer $upload_key" \
        -F "version=v$version" \
        -F "notes=Managed update smoke test v$version" \
        -F "target=linux-x86_64" \
        -F "signature=$signature" \
        -F "artifact=@$artifact" \
        "$hub_url/api/releases" >/dev/null
}

app_post() {
    local endpoint="$1"
    curl -fsS -b "$test_root/app-cookies.txt" -X POST "$app_url$endpoint"
}

run_worker() {
    podman exec \
        -e PATH=/test/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        -e VASHTI_DATA_DIR=/test/app-data \
        -e VASHTI_BIND=0.0.0.0:7771 \
        -e VASHTI_MANAGED_UPDATES=1 \
        -e VASHTI_UPDATE_DATA_DIR=/test/app-data \
        -e VASHTI_UPDATE_WORK_DIR=/test/update-work \
        -e VASHTI_UPDATE_DATABASE_PATH=/test/app-data/app.db \
        -e VASHTI_UPDATE_BINARY_PATH=/test/install/vashti \
        -e VASHTI_UPDATE_SERVICE_NAME=vashti.service \
        -e VASHTI_UPDATE_HEALTH_URL=http://127.0.0.1:7771/api/version \
        -e VASHTI_UPDATE_BASE_URL=http://127.0.0.1:7781 \
        "$container_name" /test/install/vashti --apply-update
}

echo "Building the current v$current_version server and Hub..."
cargo build --release -p vashti -p vashti-hub
mkdir -p "$test_root/bin" "$test_root/install" "$test_root/app-data/update" \
    "$test_root/hub-data" "$test_root/update-work" "$test_root/artifacts"
chmod 0700 "$test_root/update-work"
cp target/release/vashti "$test_root/install/vashti"
cp target/release/vashti-hub "$test_root/bin/vashti-hub"

echo "Building a temporary v$candidate_version update candidate..."
rsync -a \
    --exclude .git \
    --exclude .private \
    --exclude target \
    --exclude node_modules \
    --exclude apps/vashti/dist \
    "$repo_root/" "$test_root/source/"
sed -i "0,/^version = \"$current_version\"$/s//version = \"$candidate_version\"/" \
    "$test_root/source/apps/vashti/Cargo.toml"
CARGO_TARGET_DIR="$repo_root/target/managed-update-smoke" \
    VASHTI_SKIP_WEB_BUILD=1 \
    cargo build \
        --manifest-path "$test_root/source/Cargo.toml" \
        --release \
        -p vashti
candidate_binary="$repo_root/target/managed-update-smoke/release/vashti"
[[ "$($candidate_binary --version)" == "$candidate_version" ]]

current_archive="$test_root/artifacts/vashti-linux-x86_64-v$current_version.tar.gz"
candidate_archive="$test_root/artifacts/vashti-linux-x86_64-v$candidate_version.tar.gz"
rollback_archive="$test_root/artifacts/vashti-linux-x86_64-v$rollback_version.tar.gz"
package_binary "$test_root/install/vashti" "$current_version" "$current_archive"
package_binary "$candidate_binary" "$candidate_version" "$candidate_archive"

cat > "$test_root/bin/systemctl" <<'EOF_SYSTEMCTL'
#!/bin/sh
set -eu
action="$1"
pid_file=/test/app.pid
case "$action" in
    stop)
        if [ -f "$pid_file" ]; then
            pid="$(cat "$pid_file")"
            kill "$pid" 2>/dev/null || true
            for attempt in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.1
            done
            rm -f "$pid_file"
        fi
        ;;
    start)
        /test/install/vashti > /test/app.log 2>&1 &
        pid=$!
        echo "$pid" > "$pid_file"
        sleep 0.2
        kill -0 "$pid" 2>/dev/null
        [ "$(awk '{ print $3 }' "/proc/$pid/stat")" != "Z" ]
        ;;
    *)
        echo "unsupported test systemctl action: $action" >&2
        exit 1
        ;;
esac
EOF_SYSTEMCTL
chmod 0755 "$test_root/bin/systemctl"

cat > "$test_root/broken-update" <<EOF_BROKEN
#!/bin/sh
if [ "\${1:-}" = "--version" ]; then
    echo "$rollback_version"
    exit 0
fi
exit 1
EOF_BROKEN
chmod 0755 "$test_root/broken-update"
package_binary "$test_root/broken-update" "$rollback_version" "$rollback_archive"

echo "Starting an isolated Hub and Vashti in Podman..."
podman run -d \
    --name "$container_name" \
    -p 127.0.0.1::7771 \
    -p 127.0.0.1::7781 \
    -v "$test_root:/test:Z" \
    docker.io/library/ubuntu:24.04 \
    sleep infinity >/dev/null

hub_address="$(podman port "$container_name" 7781/tcp | head -n 1)"
app_address="$(podman port "$container_name" 7771/tcp | head -n 1)"
hub_url="http://$hub_address"
app_url="http://$app_address"

podman exec -d \
    -e VASHTI_HUB_DATA_DIR=/test/hub-data \
    -e VASHTI_HUB_BIND=0.0.0.0:7781 \
    "$container_name" /test/bin/vashti-hub
wait_for_url "$hub_url/api/admin/status"
setup_key="$(tr -d '[:space:]' < "$test_root/hub-data/admin-setup-key.txt")"
curl -fsS -c "$test_root/hub-cookies.txt" \
    -H 'Content-Type: application/json' \
    -d "$(jq -cn --arg key "$setup_key" '{setup_key:$key,password:"smoke-test-password"}')" \
    "$hub_url/api/admin/setup" >/dev/null

upload_release "$current_version" "$current_archive"
curl -fsS -b "$test_root/hub-cookies.txt" \
    -H 'Content-Type: application/json' \
    -d '{"rollup_previous_prereleases":false}' \
    "$hub_url/api/releases/v$current_version/promote" >/dev/null
upload_release "$candidate_version" "$candidate_archive"

podman exec \
    -e PATH=/test/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e VASHTI_DATA_DIR=/test/app-data \
    -e VASHTI_BIND=0.0.0.0:7771 \
    -e VASHTI_MANAGED_UPDATES=1 \
    -e VASHTI_UPDATE_BASE_URL=http://127.0.0.1:7781 \
    "$container_name" systemctl start vashti.service
wait_for_url "$app_url/api/version"
curl -fsS -c "$test_root/app-cookies.txt" \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","email":"admin@example.test","password":"smoke-test-password"}' \
    "$app_url/api/auth/register" >/dev/null
curl -fsS -b "$test_root/app-cookies.txt" -X PATCH \
    -H 'Content-Type: application/json' \
    -d '{"update_channel":"prerelease"}' \
    "$app_url/api/settings" >/dev/null

available_version="$(app_post /api/admin/update/check | jq -er '.available.version')"
[[ "$available_version" == "v$candidate_version" ]]
app_post /api/admin/update/install >/dev/null
run_worker
wait_for_url "$app_url/api/version"
installed_version="$(curl -fsS "$app_url/api/version" | jq -er '.version')"
[[ "$installed_version" == "$candidate_version" ]]

echo "Verified signed update v$current_version -> v$candidate_version."
echo "Testing a failed update and automatic rollback..."
upload_release "$rollback_version" "$rollback_archive"
available_version="$(app_post /api/admin/update/check | jq -er '.available.version')"
[[ "$available_version" == "v$rollback_version" ]]
app_post /api/admin/update/install >/dev/null
if run_worker > "$test_root/rollback-worker.log" 2>&1; then
    echo "the deliberately broken update unexpectedly succeeded" >&2
    exit 1
fi
wait_for_url "$app_url/api/version"
restored_version="$(curl -fsS "$app_url/api/version" | jq -er '.version')"
operation_state="$(curl -fsS -b "$test_root/app-cookies.txt" \
    "$app_url/api/admin/update" | jq -er '.operation.state')"
[[ "$restored_version" == "$candidate_version" ]]
[[ "$operation_state" == "rolled_back" ]]

echo "Verified rollback restored v$candidate_version after a failed v$rollback_version update."
echo "Managed-update Podman smoke test passed."
