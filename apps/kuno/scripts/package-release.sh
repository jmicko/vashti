#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
case "$(rustc -vV | sed -n 's/^host: //p')" in
  x86_64-unknown-linux-gnu) arch=x86_64 ;;
  *) echo "The distributed Kuno image engine currently requires Linux x86_64 GNU" >&2; exit 1 ;;
esac
command -v readelf >/dev/null
# Explicit paths prevent private Cargo target settings from making this script
# package a stale executable.
target=x86_64-unknown-linux-gnu
build_dir="$repo_root/target/kuno-release"
# The default features carry the image engine (Rust; Vulkan and Level Zero are loaded from
# the system at run time).
cargo build --locked --release -p kuno --target "$target" --target-dir "$build_dir"
binary="$build_dir/$target/release/kuno"
dynamic="$(readelf -d "$binary")"
if printf '%s\n' "$dynamic" | grep -Ei 'NEEDED.*\[(lib(sycl|mkl|dnnl|ur_|tcm|umf|imf|svml|intlc|irng|iomp|tbb|OpenCL|cuda|cublas|vulkan|ze_))|\((RPATH|RUNPATH)\)' >/dev/null; then
  echo 'Kuno must not link GPU or private runtime libraries or embed a runtime search path.' >&2
  exit 1
fi
version="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
version="v${version#v}"
release_dir="$repo_root/apps/vashti/dist/release/$version"
package_dir="$(mktemp -d)"
trap 'rm -rf "$package_dir"' EXIT
mkdir -p "$release_dir"
install -m 755 "$binary" "$package_dir/kuno"
install -m 644 apps/kuno/packaging/README-install.txt "$package_dir/README-install.txt"
install -m 644 LICENSE "$package_dir/LICENSE"
printf '%s\n' "$version" > "$package_dir/VERSION"
archive="kuno-linux-$arch.tar.gz"
tar -C "$package_dir" -czf "$release_dir/$archive" kuno README-install.txt LICENSE VERSION
(
  cd "$release_dir"
  find . -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.apk' \) -printf '%f\n' | sort | xargs sha256sum > SHA256SUMS
)
printf '%s\n' "$version" > "$release_dir/VERSION"
printf 'Built %s\n' "$release_dir/$archive"
