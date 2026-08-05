#!/bin/sh
set -eu

version="${VASHTI_VERSION:-latest}"
release_base_url="${VASHTI_RELEASE_BASE_URL:-https://vashti.chat/releases}"
install_dir="${VASHTI_INSTALL_DIR:-/usr/local/bin}"
data_dir="${VASHTI_DATA_DIR:-/var/lib/vashti}"
bind_addr="${VASHTI_BIND:-0.0.0.0:7771}"
service_user="${VASHTI_USER:-vashti}"
update_base_url="${VASHTI_UPDATE_BASE_URL:-https://vashti.chat}"

reject_whitespace() {
    name="$1"
    value="$2"
    case "$value" in
        *[[:space:]]*)
            echo "$name cannot contain whitespace" >&2
            exit 1
            ;;
    esac
}

case "$install_dir" in
    /*) ;;
    *) echo "VASHTI_INSTALL_DIR must be an absolute path" >&2; exit 1 ;;
esac
case "$data_dir" in
    /*) ;;
    *) echo "VASHTI_DATA_DIR must be an absolute path" >&2; exit 1 ;;
esac
reject_whitespace "VASHTI_INSTALL_DIR" "$install_dir"
reject_whitespace "VASHTI_DATA_DIR" "$data_dir"
reject_whitespace "VASHTI_BIND" "$bind_addr"
case "$service_user" in
    [A-Za-z_]* ) ;;
    *) echo "VASHTI_USER must start with a letter or underscore" >&2; exit 1 ;;
esac
case "$service_user" in
    *[!A-Za-z0-9_-]*) echo "VASHTI_USER contains unsupported characters" >&2; exit 1 ;;
esac
update_authority="${update_base_url#https://}"
update_authority="${update_authority%/}"
if [ "$update_authority" = "$update_base_url" ]; then
    echo "VASHTI_UPDATE_BASE_URL must use HTTPS" >&2
    exit 1
fi
case "$update_authority" in
    '' | *[/@?#]*)
        echo "VASHTI_UPDATE_BASE_URL must be an HTTPS origin without credentials or a path" >&2
        exit 1
        ;;
esac

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required command: $1" >&2
        exit 1
    fi
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

version_number() {
    printf '%s' "$1" | sed 's/^v//'
}

compare_versions() {
    left="$(version_number "$1")"
    right="$(version_number "$2")"
    left_major="$(printf '%s' "$left" | cut -d. -f1)"
    left_minor="$(printf '%s' "$left" | cut -d. -f2)"
    left_patch="$(printf '%s' "$left" | cut -d. -f3)"
    right_major="$(printf '%s' "$right" | cut -d. -f1)"
    right_minor="$(printf '%s' "$right" | cut -d. -f2)"
    right_patch="$(printf '%s' "$right" | cut -d. -f3)"

    for value in "$left_major" "$left_minor" "$left_patch" "$right_major" "$right_minor" "$right_patch"; do
        case "$value" in
            '' | *[!0-9]*)
                echo "0"
                return
                ;;
        esac
    done

    if [ "$left_major" -lt "$right_major" ]; then echo "-1"; return; fi
    if [ "$left_major" -gt "$right_major" ]; then echo "1"; return; fi
    if [ "$left_minor" -lt "$right_minor" ]; then echo "-1"; return; fi
    if [ "$left_minor" -gt "$right_minor" ]; then echo "1"; return; fi
    if [ "$left_patch" -lt "$right_patch" ]; then echo "-1"; return; fi
    if [ "$left_patch" -gt "$right_patch" ]; then echo "1"; return; fi
    echo "0"
}

bind_port() {
    case "$1" in
        *:*) printf '%s\n' "${1##*:}" ;;
        *) printf '%s\n' "7771" ;;
    esac
}

bind_host() {
    case "$1" in
        *:*) printf '%s\n' "${1%:*}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

health_check_url() {
    host="$(bind_host "$bind_addr")"
    port="$(bind_port "$bind_addr")"
    case "$host" in
        0.0.0.0 | 127.* | localhost) host="127.0.0.1" ;;
        :: | '[::]') host="[::1]" ;;
    esac
    printf 'http://%s:%s/api/version\n' "$host" "$port"
}

network_label() {
    interface="$1"
    case "$interface" in
        wg* | tun* | tap* | tailscale* | zt* | zerotier*)
            printf '%s\n' "VPN"
            ;;
        eth* | en* | wl* | wifi* | br* | bond*)
            printf '%s\n' "same network"
            ;;
        *)
            printf '%s\n' "network"
            ;;
    esac
}

detected_ipv4_addresses() {
    if command -v ip >/dev/null 2>&1; then
        ip -4 -o addr show scope global 2>/dev/null |
            awk '{ split($4, parts, "/"); print parts[1] " " $2 }'
        return
    fi

    if command -v hostname >/dev/null 2>&1; then
        hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ { print $1 " network" }'
    fi
}

print_access_urls() {
    host="$(bind_host "$bind_addr")"
    port="$(bind_port "$bind_addr")"

    echo
    echo "Vashti is running."
    echo
    echo "Try these URLs:"

    case "$host" in
        127.* | localhost)
            echo "  On this machine: http://127.0.0.1:$port"
            echo
            echo "Vashti is bound to $host, so it is only reachable from this machine."
            echo "Set VASHTI_BIND=0.0.0.0:$port before installing if you want LAN access."
            return
            ;;
    esac

    address_file="$tmp_dir/detected-addresses"
    detected_ipv4_addresses > "$address_file" || true

    if [ "$host" = "0.0.0.0" ] || [ "$host" = "::" ]; then
        echo "  On this machine: http://127.0.0.1:$port"
        detected_any=0
        while IFS=' ' read -r ip_addr interface_name; do
            [ -n "$ip_addr" ] || continue
            [ "$ip_addr" = "127.0.0.1" ] && continue
            detected_any=1
            label="$(network_label "${interface_name:-network}")"
            echo "  On the $label (${interface_name:-unknown}): http://$ip_addr:$port"
        done < "$address_file"

        if [ "$detected_any" = "0" ]; then
            echo "  From another device: use this machine's IP address with port $port"
        fi
        return
    fi

    matched_interface=""
    while IFS=' ' read -r ip_addr interface_name; do
        if [ "$ip_addr" = "$host" ]; then
            matched_interface="${interface_name:-unknown}"
            break
        fi
    done < "$address_file"

    if [ -n "$matched_interface" ]; then
        label="$(network_label "$matched_interface")"
        echo "  On the $label ($matched_interface): http://$host:$port"
    else
        echo "  Configured address: http://$host:$port"
    fi
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

installed_version=""
if command -v "$install_dir/vashti" >/dev/null 2>&1; then
    installed_version="$("$install_dir/vashti" --version 2>/dev/null | tr -d '[:space:]' || true)"
fi
if [ -n "$installed_version" ]; then
    comparison="$(compare_versions "$version" "$installed_version")"
    if [ "$comparison" = "-1" ] && [ "${VASHTI_ALLOW_DOWNGRADE:-0}" != "1" ]; then
        echo "refusing to downgrade Vashti from v$(version_number "$installed_version") to $version" >&2
        echo "downgrades can break SQLite data after migrations have run" >&2
        echo "set VASHTI_ALLOW_DOWNGRADE=1 only if you have backed up or intentionally reset the data directory" >&2
        exit 1
    fi
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
$sudo_cmd install -d -m 0700 "$data_dir"
$sudo_cmd chown "$service_user:$service_user" "$data_dir"
$sudo_cmd chmod 0700 "$data_dir"

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
StateDirectoryMode=0700
WorkingDirectory=$data_dir
Environment=VASHTI_DATA_DIR=$data_dir
Environment=VASHTI_BIND=$bind_addr
Environment=VASHTI_MANAGED_UPDATES=1
Environment=VASHTI_UPDATE_BASE_URL=$update_base_url
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
    update_service_file="$tmp_dir/vashti-update.service"
    cat > "$update_service_file" <<EOF_UPDATE_SERVICE
[Unit]
Description=Install a verified Vashti update
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
Group=$service_user
UMask=0007
StateDirectory=vashti-update
StateDirectoryMode=0700
Environment=VASHTI_UPDATE_DATA_DIR=$data_dir
Environment=VASHTI_UPDATE_WORK_DIR=/var/lib/vashti-update
Environment=VASHTI_UPDATE_DATABASE_PATH=$data_dir/app.db
Environment=VASHTI_UPDATE_BINARY_PATH=$install_dir/vashti
Environment=VASHTI_UPDATE_SERVICE_NAME=vashti.service
Environment=VASHTI_UPDATE_HEALTH_URL=$(health_check_url)
Environment=VASHTI_UPDATE_BASE_URL=$update_base_url
ExecStart=$install_dir/vashti --apply-update
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
LockPersonality=true
MemoryDenyWriteExecute=true
RestrictRealtime=true
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
SystemCallArchitectures=native
ReadWritePaths=$data_dir $install_dir
RestrictSUIDSGID=true

EOF_UPDATE_SERVICE
    update_path_file="$tmp_dir/vashti-update.path"
    cat > "$update_path_file" <<EOF_UPDATE_PATH
[Unit]
Description=Watch for an approved Vashti update request

[Path]
PathExists=$data_dir/update/request.json
PathExists=$data_dir/update/request.in-progress.json
Unit=vashti-update.service

[Install]
WantedBy=multi-user.target
EOF_UPDATE_PATH
    $sudo_cmd install -d -m 0700 -o "$service_user" -g "$service_user" "$data_dir/update"
    $sudo_cmd install -m 0644 "$service_file" /etc/systemd/system/vashti.service
    $sudo_cmd install -m 0644 "$update_service_file" /etc/systemd/system/vashti-update.service
    $sudo_cmd install -m 0644 "$update_path_file" /etc/systemd/system/vashti-update.path
    $sudo_cmd systemctl daemon-reload
    $sudo_cmd systemctl enable vashti
    $sudo_cmd systemctl enable --now vashti-update.path
    $sudo_cmd systemctl restart vashti

    print_access_urls
else
    echo "Systemd was not detected or was skipped."
    echo "Run Vashti manually with:"
    echo "  VASHTI_DATA_DIR=$data_dir VASHTI_BIND=$bind_addr $install_dir/vashti"
fi
