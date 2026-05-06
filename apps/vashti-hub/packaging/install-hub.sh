#!/bin/sh
set -eu

install_dir="${VASHTI_HUB_INSTALL_DIR:-/usr/local/bin}"
data_dir="${VASHTI_HUB_DATA_DIR:-/var/lib/vashti-hub}"
bind_addr="${VASHTI_HUB_BIND:-127.0.0.1:7781}"
service_user="${VASHTI_HUB_USER:-vashti-hub}"
cookie_secure="${VASHTI_HUB_COOKIE_SECURE:-false}"
trust_proxy_headers="${VASHTI_HUB_TRUST_PROXY_HEADERS:-false}"
max_upload_bytes="${VASHTI_HUB_MAX_UPLOAD_BYTES:-536870912}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required command: $1" >&2
        exit 1
    fi
}

read_answer() {
    printf "%s" "$1" >&2
    if IFS= read -r answer; then
        printf "%s" "$answer"
    else
        printf ""
    fi
}

prompt_value() {
    prompt_label="$1"
    current_value="$2"
    answer="$(read_answer "$prompt_label [$current_value]: ")"
    if [ -n "$answer" ]; then
        printf "%s" "$answer"
    else
        printf "%s" "$current_value"
    fi
}

prompt_yes_no() {
    prompt_label="$1"
    default_answer="$2"

    while :; do
        answer="$(read_answer "$prompt_label [$default_answer]: ")"
        if [ -z "$answer" ]; then
            answer="$default_answer"
        fi

        case "$answer" in
            y | Y | yes | YES | Yes)
                return 0
                ;;
            n | N | no | NO | No)
                return 1
                ;;
            *)
                echo "Please answer yes or no."
                ;;
        esac
    done
}

detect_bind_options() {
    detect_port="$1"
    if command -v ip >/dev/null 2>&1; then
        ip -4 -o addr show 2>/dev/null \
            | awk -v port="$detect_port" '
                {
                    split($4, a, "/")
                    if (a[1] != "" && a[1] !~ /^169[.]254[.]/) {
                        print $2 " " a[1] ":" port
                    }
                }
            '
    fi
}

prompt_bind_addr() {
    default_port="${bind_addr##*:}"
    case "$default_port" in
        *[!0-9]* | "")
            default_port="7781"
            ;;
    esac

    options_file="${TMPDIR:-/tmp}/vashti-hub-bind-options.$$"
    detect_bind_options "$default_port" > "$options_file"
    option_count="$(wc -l < "$options_file" | tr -d ' ')"
    other_choice=$((option_count + 1))

    echo
    echo "Bind address"
    echo "This controls which network address Vashti Hub listens on."
    echo "Use a WireGuard address when nginx reaches Hub over WireGuard."
    echo "Press Enter to keep the current safe default: $bind_addr"
    echo

    choice_label=1
    while IFS=" " read -r iface addr; do
        if [ -n "$iface" ] && [ -n "$addr" ]; then
            echo "  $choice_label) $addr ($iface)"
            choice_label=$((choice_label + 1))
        fi
    done < "$options_file"
    echo "  $other_choice) other"
    echo

    while :; do
        answer="$(read_answer "Choose bind address: ")"
        if [ -z "$answer" ]; then
            rm -f "$options_file"
            return
        fi

        if [ "$answer" = "$other_choice" ]; then
            custom_bind="$(prompt_value "Type bind address" "$bind_addr")"
            bind_addr="$custom_bind"
            rm -f "$options_file"
            return
        fi

        if [ "$answer" -ge 1 ] 2>/dev/null && [ "$answer" -le "$option_count" ] 2>/dev/null; then
            selected_addr="$(awk -v n="$answer" 'NR == n { print $2 }' "$options_file")"
            if [ -n "$selected_addr" ]; then
                bind_addr="$selected_addr"
                rm -f "$options_file"
                return
            fi
        fi

        echo "Enter a number from the list, or press Enter to keep $bind_addr."
    done
}

configure_interactive() {
    echo
    echo "Vashti Hub installer"
    echo "Press Enter at any prompt to accept the default shown in brackets."
    echo

    echo "Install directory"
    echo "This is where the vashti-hub binary will be installed."
    install_dir="$(prompt_value "Install directory" "$install_dir")"
    echo

    echo "Data directory"
    echo "This stores the Hub database, release artifacts, admin setup keys, and upload token."
    data_dir="$(prompt_value "Data directory" "$data_dir")"
    echo

    prompt_bind_addr

    echo
    echo "Public HTTPS reverse proxy mode"
    echo "Enable this when browsers reach Hub through HTTPS via nginx, Caddy, or a tunnel."
    echo "It enables Secure cookies and trusts proxy headers from your reverse proxy."
    echo "Leave it disabled for direct local HTTP access."
    if [ "$cookie_secure" = "true" ] || [ "$trust_proxy_headers" = "true" ]; then
        proxy_default="yes"
    else
        proxy_default="no"
    fi
    if prompt_yes_no "Enable public HTTPS reverse proxy mode?" "$proxy_default"; then
        cookie_secure="true"
        trust_proxy_headers="true"
    else
        cookie_secure="false"
        trust_proxy_headers="false"
    fi

    echo
    echo "Maximum upload size"
    echo "This limits release artifact uploads through the admin page."
    upload_mb=$((max_upload_bytes / 1048576))
    upload_mb="$(prompt_value "Maximum upload size in MiB" "$upload_mb")"
    case "$upload_mb" in
        *[!0-9]* | "")
            echo "Invalid upload size: $upload_mb" >&2
            exit 1
            ;;
    esac
    max_upload_bytes=$((upload_mb * 1048576))

    echo
    echo "Install summary"
    echo "  Binary: $install_dir/vashti-hub"
    echo "  Data: $data_dir"
    echo "  Bind: $bind_addr"
    echo "  Public HTTPS reverse proxy mode: $cookie_secure"
    echo "  Max upload: $upload_mb MiB"
    echo
    if ! prompt_yes_no "Continue with installation?" "yes"; then
        echo "Installation cancelled."
        exit 0
    fi
    echo
}

need_cmd install
need_cmd id

if [ -t 0 ] && [ "${VASHTI_HUB_NO_PROMPT:-0}" != "1" ]; then
    configure_interactive
fi

if [ "$(id -u)" -eq 0 ]; then
    sudo_cmd=""
else
    need_cmd sudo
    sudo_cmd="sudo"
fi

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
binary_path="${VASHTI_HUB_BINARY:-$script_dir/vashti-hub}"

if [ ! -x "$binary_path" ]; then
    echo "missing executable vashti-hub binary at $binary_path" >&2
    exit 1
fi

echo "Installing Vashti Hub binary to $install_dir/vashti-hub..."
$sudo_cmd install -d "$install_dir"
$sudo_cmd install -m 0755 "$binary_path" "$install_dir/vashti-hub"

if ! id "$service_user" >/dev/null 2>&1; then
    need_cmd useradd
    echo "Creating system user $service_user..."
    if ! $sudo_cmd useradd --system --home-dir "$data_dir" --shell /usr/sbin/nologin "$service_user" 2>/dev/null; then
        $sudo_cmd useradd --system --home-dir "$data_dir" --shell /sbin/nologin "$service_user"
    fi
fi

echo "Preparing data directory $data_dir..."
$sudo_cmd install -d "$data_dir"
$sudo_cmd chown "$service_user:$service_user" "$data_dir"

if command -v systemctl >/dev/null 2>&1 && [ "${VASHTI_HUB_NO_SYSTEMD:-0}" != "1" ]; then
    service_file="${TMPDIR:-/tmp}/vashti-hub.service.$$"
    cat > "$service_file" <<EOF_SERVICE
[Unit]
Description=Vashti Hub
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
Group=$service_user
StateDirectory=vashti-hub
WorkingDirectory=$data_dir
Environment=VASHTI_HUB_DATA_DIR=$data_dir
Environment=VASHTI_HUB_BIND=$bind_addr
Environment=VASHTI_HUB_COOKIE_SECURE=$cookie_secure
Environment=VASHTI_HUB_TRUST_PROXY_HEADERS=$trust_proxy_headers
Environment=VASHTI_HUB_MAX_UPLOAD_BYTES=$max_upload_bytes
ExecStart=$install_dir/vashti-hub
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ReadWritePaths=$data_dir

[Install]
WantedBy=multi-user.target
EOF_SERVICE
    trap 'rm -f "$service_file"' EXIT INT TERM

    echo "Installing systemd service..."
    $sudo_cmd install -m 0644 "$service_file" /etc/systemd/system/vashti-hub.service
    $sudo_cmd systemctl daemon-reload
    $sudo_cmd systemctl enable vashti-hub
    $sudo_cmd systemctl restart vashti-hub

    echo "Vashti Hub is running on $bind_addr."
    echo "Open /admin through your reverse proxy to claim the Hub."
else
    echo "Systemd was not detected or was skipped."
    echo "Run Vashti Hub manually with:"
    echo "  VASHTI_HUB_DATA_DIR=$data_dir VASHTI_HUB_BIND=$bind_addr VASHTI_HUB_COOKIE_SECURE=$cookie_secure VASHTI_HUB_TRUST_PROXY_HEADERS=$trust_proxy_headers $install_dir/vashti-hub"
fi
