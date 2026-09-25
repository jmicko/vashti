#!/bin/sh
set -eu

version="${KUNO_VERSION:-latest}"
release_base_url="${KUNO_RELEASE_BASE_URL:-https://vashti.chat/releases}"
case "$release_base_url" in
    https://*) ;;
    *) echo 'KUNO_RELEASE_BASE_URL must use HTTPS' >&2; exit 1 ;;
esac
for command in curl tar sha256sum uname mktemp; do
    command -v "$command" >/dev/null 2>&1 || { echo "Missing required command: $command" >&2; exit 1; }
done
case "$(uname -s)/$(uname -m)" in
    Linux/x86_64|Linux/amd64) archive=kuno-linux-x86_64.tar.gz ;;
    *) echo 'This Kuno image-engine package requires Linux x86_64 with systemd.' >&2; exit 1 ;;
esac
if [ "$(id -u)" -eq 0 ]; then
    sudo_cmd=""
else
    command -v sudo >/dev/null 2>&1 || { echo 'sudo is required to install Kuno' >&2; exit 1; }
    sudo_cmd=sudo
fi
if [ "$version" = latest ]; then
    version="$(curl --proto '=https' --proto-redir '=https' -fsSL "${release_base_url%/}/latest/VERSION")"
fi
case "$version" in
    v[0-9]*) ;;
    [0-9]*) version="v$version" ;;
    *) echo 'Invalid Kuno release version' >&2; exit 1 ;;
esac
case "$version" in
    *[!a-zA-Z0-9.-]*|*..*) echo 'Invalid Kuno release version' >&2; exit 1 ;;
esac
base_url="${release_base_url%/}/$version"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' 0
trap 'exit 1' INT TERM

echo "Downloading Kuno from release $version..."
curl --proto '=https' --proto-redir '=https' -fL "$base_url/$archive" -o "$tmp_dir/$archive"
curl --proto '=https' --proto-redir '=https' -fsSL "$base_url/SHA256SUMS" -o "$tmp_dir/SHA256SUMS"
(
    cd "$tmp_dir"
    # Select an exact filename; never pass other downloaded paths to sha256sum.
    awk -v name="$archive" '$2 == name { print $0; found = 1 } END { if (!found) exit 1 }' SHA256SUMS > checksum
    sha256sum -c checksum
) || { echo 'Kuno checksum verification failed; nothing was installed.' >&2; exit 1; }
# Extract only the executable. The release archive has a flat layout.
tar -xzf "$tmp_dir/$archive" -C "$tmp_dir" -- kuno
if [ ! -f "$tmp_dir/kuno" ] || [ -L "$tmp_dir/kuno" ] || [ ! -x "$tmp_dir/kuno" ]; then
    echo 'Package did not contain a regular Kuno executable' >&2; exit 1
fi
operation=install
if [ -f /usr/local/lib/kuno/kuno ]; then operation=upgrade; fi
printf '\nKuno will %s its systemd service and briefly interrupt inference.\n' "$operation"
echo 'Existing apps keep their Ollama URL. The Kuno installer will ask before changing services.'
if [ "${KUNO_YES:-0}" = 1 ]; then
    $sudo_cmd "$tmp_dir/kuno" "$operation" --yes
else
    # curl | sh consumes stdin; read the actual confirmation from the terminal.
    if ! ( : </dev/tty ) 2>/dev/null; then
        echo 'No terminal is available. Re-run interactively, or set KUNO_YES=1 to explicitly accept the interruption.' >&2
        exit 1
    fi
    $sudo_cmd "$tmp_dir/kuno" "$operation" </dev/tty
fi
printf '\nKuno is running. Diagnostic report: sudo /usr/local/lib/kuno/kuno doctor\n'
echo 'Image generation also needs a model collection and GPU configuration; see https://vashti.chat/getting-started#kuno.'
