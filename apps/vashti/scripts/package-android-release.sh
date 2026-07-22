#!/usr/bin/env bash
set -euo pipefail

aligned_apk=""
cleanup() {
    if [[ -n "$aligned_apk" ]]; then
        rm -f "$aligned_apk"
    fi
}
trap cleanup EXIT

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

version="${VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti/Cargo.toml | head -n 1)}"
android_version="$(sed -n 's/^version = "\(.*\)"/\1/p' apps/vashti-android/Cargo.toml | head -n 1)"
version_label="v${version}"
release_dir="${DIST_DIR:-apps/vashti/dist/release}/$version_label"
artifact="$release_dir/vashti-android.apk"

if [[ "$android_version" != "$version" ]]; then
    echo "version mismatch: vashti is $version but vashti-android is $android_version" >&2
    exit 1
fi

if [[ -f .private/android-signing.env ]]; then
    # shellcheck disable=SC1091
    source .private/android-signing.env
fi

export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
if [[ ! -x "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" ]]; then
    echo "Android SDK not found at $ANDROID_HOME" >&2
    echo "See apps/vashti/docs/android-client.md for setup instructions." >&2
    exit 1
fi

if [[ -z "${NDK_HOME:-}" ]]; then
    NDK_HOME="$(find "$ANDROID_HOME/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi
export NDK_HOME

find_jdk() {
    local javac_path=""

    if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/javac" ]]; then
        printf '%s\n' "$JAVA_HOME"
        return 0
    fi

    if command -v javac >/dev/null 2>&1; then
        javac_path="$(readlink -f "$(command -v javac)")"
    else
        javac_path="$(find \
            "$HOME/.local/share/vashti-build" \
            /usr/lib/jvm \
            /opt/android-studio/jbr \
            "$HOME/.local/share/JetBrains/Toolbox/apps/AndroidStudio" \
            -type f -path '*/bin/javac' -perm -u+x 2>/dev/null | sort -V | tail -n 1)"
    fi

    if [[ -n "$javac_path" ]]; then
        dirname "$(dirname "$javac_path")"
        return 0
    fi

    return 1
}

if ! JAVA_HOME="$(find_jdk)"; then
    echo "A Java 17 or newer JDK is required; a JRE alone is not enough." >&2
    echo "Set JAVA_HOME to a JDK containing bin/javac." >&2
    echo "See apps/vashti/docs/android-client.md for setup instructions." >&2
    exit 1
fi
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

if [[ -z "$NDK_HOME" || ! -d "$NDK_HOME" ]]; then
    echo "Android NDK not found under $ANDROID_HOME/ndk" >&2
    exit 1
fi

pushd apps/vashti-android >/dev/null
npm --prefix ../vashti/web exec tauri -- android build --apk --ci
popd >/dev/null

unsigned_apk="$(find apps/vashti-android/gen/android/app/build/outputs/apk \
    -type f -name '*release*.apk' | sort | tail -n 1)"
if [[ -z "$unsigned_apk" || ! -f "$unsigned_apk" ]]; then
    echo "Android release build did not produce an APK" >&2
    exit 1
fi

build_tools="$(find "$ANDROID_HOME/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
apksigner="$build_tools/apksigner"
zipalign="$build_tools/zipalign"
if [[ ! -x "$apksigner" || ! -x "$zipalign" ]]; then
    echo "Android build tools are incomplete under $build_tools" >&2
    exit 1
fi

mkdir -p "$release_dir"
if "$apksigner" verify "$unsigned_apk" >/dev/null 2>&1; then
    cp "$unsigned_apk" "$artifact"
else
    : "${VASHTI_ANDROID_KEYSTORE:?Set VASHTI_ANDROID_KEYSTORE or create .private/android-signing.env}"
    : "${VASHTI_ANDROID_KEY_ALIAS:?Set VASHTI_ANDROID_KEY_ALIAS}"
    : "${VASHTI_ANDROID_KEYSTORE_PASSWORD:?Set VASHTI_ANDROID_KEYSTORE_PASSWORD}"
    : "${VASHTI_ANDROID_KEY_PASSWORD:=$VASHTI_ANDROID_KEYSTORE_PASSWORD}"
    export VASHTI_ANDROID_KEYSTORE_PASSWORD VASHTI_ANDROID_KEY_PASSWORD

    aligned_apk="$(mktemp --suffix=.apk)"
    "$zipalign" -f -p 4 "$unsigned_apk" "$aligned_apk"
    "$apksigner" sign \
        --ks "$VASHTI_ANDROID_KEYSTORE" \
        --ks-key-alias "$VASHTI_ANDROID_KEY_ALIAS" \
        --ks-pass env:VASHTI_ANDROID_KEYSTORE_PASSWORD \
        --key-pass env:VASHTI_ANDROID_KEY_PASSWORD \
        --out "$artifact" \
        "$aligned_apk"
fi

"$apksigner" verify --verbose "$artifact" >/dev/null
(
    cd "$release_dir"
    sha256sum vashti-* > SHA256SUMS
)
printf '%s\n' "$version_label" > "$release_dir/VERSION"

echo "created signed Android artifact $artifact"
echo "updated $release_dir/SHA256SUMS"
