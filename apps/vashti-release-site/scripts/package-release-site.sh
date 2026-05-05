#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

target="${TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
dist_dir="${DIST_DIR:-apps/vashti-release-site/dist/release-site}"

cargo_args=(build -p vashti-release-site --release)
if [[ -n "${TARGET:-}" ]]; then
    cargo_args+=(--target "$target")
fi

cargo "${cargo_args[@]}"

binary_path="target/release/vashti-release-site"
if [[ -n "${TARGET:-}" ]]; then
    binary_path="target/$target/release/vashti-release-site"
fi

mkdir -p "$dist_dir"
cp "$binary_path" "$dist_dir/vashti-release-site"
cp apps/vashti-release-site/packaging/vashti-release-site.service "$dist_dir/vashti-release-site.service"

echo "created $dist_dir/vashti-release-site"
