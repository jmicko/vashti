#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

version="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
target="${TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
dist_dir="${DIST_DIR:-apps/vashti/dist/release}"

case "$target" in
    x86_64-unknown-linux-gnu | x86_64-unknown-linux-musl)
        os="linux"
        arch="x86_64"
        ;;
    aarch64-unknown-linux-gnu | aarch64-unknown-linux-musl)
        os="linux"
        arch="aarch64"
        ;;
    *)
        echo "unsupported release target: $target" >&2
        exit 1
        ;;
esac

cargo_args=(build -p vashti --release)
if [[ -n "${TARGET:-}" ]]; then
    cargo_args+=(--target "$target")
fi

cargo "${cargo_args[@]}"

binary_path="target/release/vashti"
if [[ -n "${TARGET:-}" ]]; then
    binary_path="target/$target/release/vashti"
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required to verify release binary dependencies" >&2
    exit 1
fi

if readelf -d "$binary_path" | grep -Eq '\(NEEDED\).*libsqlite3([.]so)?'; then
    echo "release binary dynamically links SQLite; use the bundled SQLite build instead" >&2
    exit 1
fi

package_name="vashti-v${version}-${os}-${arch}"
work_dir="$dist_dir/work"
package_dir="$work_dir/$package_name"
archive_name="vashti-${os}-${arch}.tar.gz"
version_label="v${version}"
release_dir="$dist_dir/$version_label"

rm -rf "$package_dir" "$release_dir"
mkdir -p "$package_dir"

cp "$binary_path" "$package_dir/vashti"
cp apps/vashti/packaging/README-install.txt "$package_dir/README-install.txt"
cp apps/vashti/packaging/vashti.service "$package_dir/vashti.service"

mkdir -p "$release_dir"
tar -C "$work_dir" -czf "$release_dir/$archive_name" "$package_name"

(
    cd "$release_dir"
    sha256sum "$archive_name" > SHA256SUMS
)

printf '%s\n' "$version_label" > "$release_dir/VERSION"

echo "created $release_dir/$archive_name"
echo "created $release_dir/SHA256SUMS"
