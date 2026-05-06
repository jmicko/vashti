#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

version="${VERSION:-v$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
target="${RELEASE_TARGET:-linux-x86_64}"
artifact="${ARTIFACT:-apps/vashti/dist/release/$version/vashti-$target.tar.gz}"
hub_url="${VASHTI_HUB_URL:-https://vashti.chat}"
notes="${NOTES:-}"

upload_key="${VASHTI_HUB_UPLOAD_KEY:-${VASHTI_HUB_TOKEN:-}}"
if [[ -z "$upload_key" ]]; then
    if [[ -t 0 ]]; then
        read -r -s -p "One-time Hub upload key: " upload_key
        echo
    else
        echo "missing one-time hub upload key: set VASHTI_HUB_UPLOAD_KEY" >&2
        exit 1
    fi
fi

if [[ ! -f "$artifact" ]]; then
    echo "missing artifact: $artifact" >&2
    echo "run ./apps/vashti/scripts/package-release.sh first" >&2
    exit 1
fi

curl -fsS \
    -H "Authorization: Bearer $upload_key" \
    -F "version=$version" \
    -F "target=$target" \
    -F "notes=$notes" \
    -F "artifact=@$artifact" \
    "$hub_url/api/releases"

echo
echo "published $artifact to $hub_url as $version / $target"
