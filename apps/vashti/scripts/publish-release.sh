#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

version="${VERSION:-v$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
target="${RELEASE_TARGET:-linux-x86_64}"
artifact="${ARTIFACT:-apps/vashti/dist/release/$version/vashti-$target.tar.gz}"
hub_url="${VASHTI_HUB_URL:-https://vashti.chat}"
token_file="${VASHTI_HUB_TOKEN_FILE:-apps/vashti/.secrets/hub-token}"
notes="${NOTES:-}"

token="${VASHTI_HUB_TOKEN:-}"
if [[ -z "$token" ]]; then
    if [[ ! -f "$token_file" ]]; then
        echo "missing hub token: set VASHTI_HUB_TOKEN or create $token_file" >&2
        exit 1
    fi
    token="$(tr -d '[:space:]' < "$token_file")"
fi

if [[ ! -f "$artifact" ]]; then
    echo "missing artifact: $artifact" >&2
    echo "run ./apps/vashti/scripts/package-release.sh first" >&2
    exit 1
fi

curl -fsS \
    -H "Authorization: Bearer $token" \
    -F "version=$version" \
    -F "target=$target" \
    -F "notes=$notes" \
    -F "artifact=@$artifact" \
    "$hub_url/api/releases"

echo
echo "published $artifact to $hub_url as $version / $target"
