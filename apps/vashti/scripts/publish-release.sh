#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

version="${VERSION:-v$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
hub_url="${VASHTI_HUB_URL:-https://vashti.chat}"
signing_key="${VASHTI_RELEASE_SIGNING_KEY:-$repo_root/.private/keys/vashti-release-signing-key.pem}"
public_key_file="$repo_root/crates/vashti-update-manifest/release-public-key.txt"
notes="${NOTES:-}"
notes_file="${RELEASE_NOTES_FILE:-}"
declare -a targets=()
declare -a artifacts=()
declare -a signatures=()

if [[ -n "${ARTIFACT:-}" ]]; then
    targets+=("${RELEASE_TARGET:-linux-x86_64}")
    artifacts+=("$ARTIFACT")
else
    linux_artifact="apps/vashti/dist/release/$version/vashti-linux-x86_64.tar.gz"
    android_artifact="apps/vashti/dist/release/$version/vashti-android.apk"
    if [[ -f "$linux_artifact" ]]; then
        targets+=("linux-x86_64")
        artifacts+=("$linux_artifact")
    fi
    if [[ -f "$android_artifact" ]]; then
        targets+=("android-universal")
        artifacts+=("$android_artifact")
    fi
fi

if [[ -z "$notes" && -z "$notes_file" ]]; then
    versioned_notes_file="apps/vashti/release-notes/$version.md"
    latest_notes_file="apps/vashti/release-notes-latest.md"
    if [[ -f "$versioned_notes_file" ]]; then
        notes_file="$versioned_notes_file"
    elif [[ -f "$latest_notes_file" ]]; then
        notes_file="$latest_notes_file"
    fi
fi

upload_key="${VASHTI_HUB_UPLOAD_KEY:-${VASHTI_HUB_TOKEN:-}}"
if [[ -z "$upload_key" ]]; then
    if [[ -t 0 ]]; then
        read -r -s -p "One-time Hub upload key (hidden; press Enter after paste): " upload_key
        echo
        if [[ -z "$upload_key" ]]; then
            echo "upload key cannot be empty" >&2
            exit 1
        fi
        echo "Upload key received. Publishing..."
    else
        echo "missing one-time hub upload key: set VASHTI_HUB_UPLOAD_KEY" >&2
        exit 1
    fi
else
    echo "Using upload key from environment. Publishing..."
fi

if (( ${#artifacts[@]} == 0 )); then
    echo "no release artifacts found for $version" >&2
    echo "run the Linux and/or Android package scripts first" >&2
    exit 1
fi
for artifact in "${artifacts[@]}"; do
    if [[ ! -f "$artifact" ]]; then
        echo "missing artifact: $artifact" >&2
        exit 1
    fi
done

if [[ -n "$notes_file" && ! -f "$notes_file" ]]; then
    echo "missing release notes file: $notes_file" >&2
    exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to sign release artifacts" >&2
    exit 1
fi
if [[ ! -f "$signing_key" ]]; then
    echo "missing release signing key: $signing_key" >&2
    echo "set VASHTI_RELEASE_SIGNING_KEY to the offline Ed25519 private key" >&2
    exit 1
fi
if [[ ! -f "$public_key_file" ]]; then
    echo "missing compiled release public key: $public_key_file" >&2
    exit 1
fi

expected_public_key="$(tr -d '[:space:]' < "$public_key_file")"
actual_public_key="$(
    openssl pkey -in "$signing_key" -pubout -outform DER 2>/dev/null \
        | tail -c 32 \
        | openssl base64 -A
)"
if [[ "$actual_public_key" != "$expected_public_key" ]]; then
    echo "release signing key does not match Vashti's compiled public key" >&2
    exit 1
fi

signing_message_file="$(mktemp)"
trap 'rm -f "$signing_message_file"' EXIT
for index in "${!artifacts[@]}"; do
    artifact="${artifacts[$index]}"
    filename="$(basename "$artifact")"
    size_bytes="$(wc -c < "$artifact" | tr -d '[:space:]')"
    sha256="$(sha256sum "$artifact" | awk '{ print $1 }')"
    printf '%s\nversion=%s\ntarget=%s\nfilename=%s\nsha256=%s\nsize_bytes=%s\n' \
        'vashti-update-manifest-v1' \
        "$version" \
        "${targets[$index]}" \
        "$filename" \
        "$sha256" \
        "$size_bytes" > "$signing_message_file"
    signatures+=("$(
        openssl pkeyutl \
            -sign \
            -rawin \
            -inkey "$signing_key" \
            -in "$signing_message_file" \
            | openssl base64 -A
    )")
done

curl_args=(
    -fsS
    -H "Authorization: Bearer $upload_key"
    -F "version=$version"
)
if [[ -n "$notes_file" ]]; then
    curl_args+=(-F "notes=<$notes_file")
else
    curl_args+=(-F "notes=$notes")
fi
for index in "${!artifacts[@]}"; do
    curl_args+=(
        -F "target=${targets[$index]}"
        -F "signature=${signatures[$index]}"
        -F "artifact=@${artifacts[$index]}"
    )
done
curl_args+=("$hub_url/api/releases")

curl "${curl_args[@]}" >/dev/null

echo
latest="$(curl -fsS "$hub_url/releases/latest/VERSION" | tr -d '[:space:]')"

echo "Published release"
echo "  Version:  $version"
echo "  Latest:   $latest"
for index in "${!artifacts[@]}"; do
    artifact="${artifacts[$index]}"
    filename="$(basename "$artifact")"
    size_bytes="$(wc -c < "$artifact" | tr -d '[:space:]')"
    sha256="$(sha256sum "$artifact" | awk '{ print $1 }')"
    echo
    echo "  Target:   ${targets[$index]}"
    echo "  File:     $filename"
    echo "  Size:     $size_bytes bytes"
    echo "  SHA256:   $sha256"
    echo "  URL:      $hub_url/releases/$version/$filename"
done
if [[ -n "$notes_file" ]]; then
    echo "  Notes:    $notes_file"
elif [[ -n "$notes" ]]; then
    echo "  Notes:    inline"
else
    echo "  Notes:    none"
fi
echo
echo "Success!"
