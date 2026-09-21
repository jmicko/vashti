#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
target="$(rustc -vV | sed -n 's/^host: //p')"
case "$target" in
  x86_64-unknown-linux-gnu) arch=x86_64 ;;
  aarch64-unknown-linux-gnu) arch=aarch64 ;;
  *) echo "Kuno currently packages native Linux GNU builds only" >&2; exit 1 ;;
esac

cargo build --locked --release -p kuno
package_dir="$repo_root/dist/kuno-linux-$arch"
mkdir -p "$package_dir"
install -m 755 target/release/kuno "$package_dir/kuno"
install -m 644 apps/kuno/packaging/README-install.txt "$package_dir/README-install.txt"
install -m 644 LICENSE "$package_dir/LICENSE"
tar -C "$package_dir" -czf "$repo_root/dist/kuno-linux-$arch.tar.gz" kuno README-install.txt LICENSE
(
  cd "$repo_root/dist"
  sha256sum "kuno-linux-$arch.tar.gz" "kuno-linux-$arch/kuno" > "kuno-linux-$arch.sha256"
)
printf 'Built %s\n' "$package_dir/kuno"
