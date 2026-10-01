#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
if [[ "${KUNO_DEV_SYCL_BUILD:-0}" != 0 ]]; then
  echo 'Do not publish the development SYCL-linked worker as the standard Kuno executable.' >&2
  exit 1
fi
case "$(rustc -vV | sed -n 's/^host: //p')" in
  x86_64-unknown-linux-gnu) arch=x86_64 ;;
  *) echo "The distributed Kuno image engine currently requires Linux x86_64 GNU" >&2; exit 1 ;;
esac
: "${KUNO_SD_SOURCE_DIR:?Set KUNO_SD_SOURCE_DIR to the pinned native source checkout}"
: "${KUNO_SD_BUILD_DIR:?Set KUNO_SD_BUILD_DIR to its prepared native build directory}"
# Never publish the gateway-only binary under the image-engine artifact name.
cargo build --locked --release -p kuno --features native-image
version="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
version="v${version#v}"
release_dir="$repo_root/apps/vashti/dist/release/$version"
package_dir="$(mktemp -d)"
trap 'rm -rf "$package_dir"' EXIT
mkdir -p "$release_dir" "$package_dir/licenses"
install -m 755 target/release/kuno "$package_dir/kuno"
install -m 644 apps/kuno/packaging/README-install.txt "$package_dir/README-install.txt"
install -m 644 LICENSE "$package_dir/LICENSE"
install -m 644 "$KUNO_SD_SOURCE_DIR/LICENSE" "$package_dir/licenses/stable-diffusion.cpp.txt"
install -m 644 "$KUNO_SD_SOURCE_DIR/ggml/LICENSE" "$package_dir/licenses/ggml.txt"
if grep -q '^SD_CUDA:BOOL=ON$' "$KUNO_SD_BUILD_DIR/CMakeCache.txt"; then
  cuda_library="$(sed -n 's|^CUDA_cudart_static_LIBRARY:[^=]*=||p' "$KUNO_SD_BUILD_DIR/CMakeCache.txt" | head -n 1)"
  cuda_root="${KUNO_CUDA_ROOT:-$(dirname "$(dirname "$cuda_library")")}"
  for notice in LICENSE cuda_cudart-LICENSE cuda_culibos-LICENSE libcublas-LICENSE; do
    if [[ ! -f "$cuda_root/$notice" ]]; then
      echo "Missing CUDA redistribution notice $notice; set KUNO_CUDA_ROOT to the toolkit containing its licenses" >&2
      exit 1
    fi
    install -m 644 "$cuda_root/$notice" "$package_dir/licenses/nvidia-$notice.txt"
  done
fi
printf '%s\n' "$version" > "$package_dir/VERSION"
archive="kuno-linux-$arch.tar.gz"
tar -C "$package_dir" -czf "$release_dir/$archive" kuno README-install.txt LICENSE licenses VERSION
(
  cd "$release_dir"
  find . -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.apk' \) -printf '%f\n' | sort | xargs sha256sum > SHA256SUMS
)
printf '%s\n' "$version" > "$release_dir/VERSION"
printf 'Built %s\n' "$release_dir/$archive"
