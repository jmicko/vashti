#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

target="${TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
dist_dir="${DIST_DIR:-apps/vashti-hub/dist/hub}"

cargo_args=(build -p vashti-hub --release)
if [[ -n "${TARGET:-}" ]]; then
    cargo_args+=(--target "$target")
fi

cargo "${cargo_args[@]}"

binary_path="target/release/vashti-hub"
if [[ -n "${TARGET:-}" ]]; then
    binary_path="target/$target/release/vashti-hub"
fi

mkdir -p "$dist_dir"
cp "$binary_path" "$dist_dir/vashti-hub"
cp apps/vashti-hub/packaging/vashti-hub.service "$dist_dir/vashti-hub.service"
cp apps/vashti-hub/packaging/install-hub.sh "$dist_dir/install-hub.sh"

echo "created $dist_dir/vashti-hub"
