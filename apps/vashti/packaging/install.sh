#!/bin/sh
set -eu

version="${VASHTI_VERSION:-latest}"
release_base_url="${VASHTI_RELEASE_BASE_URL:-https://vashti.chat/releases}"
install_dir="${VASHTI_INSTALL_DIR:-/usr/local/bin}"
data_dir="${VASHTI_DATA_DIR:-/var/lib/vashti}"
bind_addr="${VASHTI_BIND:-0.0.0.0:7771}"
service_user="${VASHTI_USER:-vashti}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required command: $1" >&2
        exit 1
    fi
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

need_cmd curl
need_cmd tar
need_cmd sha256sum
need_cmd uname
need_cmd mktemp

os="$(lower "$(uname -s)")"
case "$os" in
    linux) ;;
    *)
        echo "unsupported operating system: $os" >&2
        exit 1
        ;;
esac

case "$(uname -m)" in
    x86_64 | amd64)
        arch="x86_64"
        ;;
    aarch64 | arm64)
        arch="aarch64"
        ;;
    *)
        echo "unsupported CPU architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

if [ "$(id -u)" -eq 0 ]; then
    sudo_cmd=""
else
    need_cmd sudo
    sudo_cmd="sudo"
fi

archive="vashti-${os}-${arch}.tar.gz"
if [ "$version" = "latest" ]; then
    base_url="${release_base_url%/}/latest"
    version="$(curl -fsL "$base_url/VERSION" | tr -d '[:space:]')"
    if [ -z "$version" ]; then
        echo "could not resolve latest Vashti release version" >&2
        exit 1
    fi
else
    base_url="${release_base_url%/}/$version"
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT INT TERM

echo "Downloading Vashti ${version} for ${os}/${arch}..."
curl -fL "$base_url/$archive" -o "$tmp_dir/$archive"
curl -fL "$base_url/SHA256SUMS" -o "$tmp_dir/SHA256SUMS"

(
    cd "$tmp_dir"
    checksum_line="$(grep "[[:space:]]$archive\$" SHA256SUMS || true)"
    if [ -z "$checksum_line" ]; then
        echo "checksum file did not include $archive" >&2
        exit 1
    fi
    printf '%s\n' "$checksum_line" | sha256sum -c -
)

tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
package_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name 'vashti-v*-linux-*' | head -n 1)"
if [ -z "$package_dir" ] || [ ! -x "$package_dir/vashti" ]; then
    echo "downloaded package did not contain a vashti binary" >&2
    exit 1
fi

echo "Installing Vashti binary to $install_dir/vashti..."
$sudo_cmd install -d "$install_dir"
$sudo_cmd install -m 0755 "$package_dir/vashti" "$install_dir/vashti"

if ! id "$service_user" >/dev/null 2>&1; then
    echo "Creating system user $service_user..."
    if ! $sudo_cmd useradd --system --home-dir "$data_dir" --shell /usr/sbin/nologin "$service_user" 2>/dev/null; then
        $sudo_cmd useradd --system --home-dir "$data_dir" --shell /sbin/nologin "$service_user"
    fi
fi

echo "Preparing data directory $data_dir..."
$sudo_cmd install -d "$data_dir"
$sudo_cmd chown "$service_user:$service_user" "$data_dir"

if command -v systemctl >/dev/null 2>&1 && [ "${VASHTI_NO_SYSTEMD:-0}" != "1" ]; then
    echo "Installing systemd service..."
    service_file="$tmp_dir/vashti.service"
    cat > "$service_file" <<EOF_SERVICE
[Unit]
Description=Vashti self-hosted Ollama chat
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
Group=$service_user
StateDirectory=vashti
WorkingDirectory=$data_dir
Environment=VASHTI_DATA_DIR=$data_dir
Environment=VASHTI_BIND=$bind_addr
ExecStart=$install_dir/vashti
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ReadWritePaths=$data_dir

[Install]
WantedBy=multi-user.target
EOF_SERVICE
    $sudo_cmd install -m 0644 "$service_file" /etc/systemd/system/vashti.service
    $sudo_cmd systemctl daemon-reload
    $sudo_cmd systemctl enable vashti
    $sudo_cmd systemctl restart vashti

    echo "Vashti is running. Open http://SERVER_IP:${bind_addr##*:}"
else
    echo "Systemd was not detected or was skipped."
    echo "Run Vashti manually with:"
    echo "  VASHTI_DATA_DIR=$data_dir VASHTI_BIND=$bind_addr $install_dir/vashti"
fi
