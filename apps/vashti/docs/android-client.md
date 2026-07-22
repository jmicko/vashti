# Vashti Android Client

Vashti's Android app is a Tauri 2 shell around the same React interface used by
the web app. The APK bundles the frontend at build time. It never loads or
executes JavaScript from a configured Vashti server.

## Architecture

The Android shell owns the network and credential boundary:

* the user saves one or more Vashti server root URLs
* `/api/version` verifies that a new URL is a compatible Vashti installation
* the normal session response re-checks server identity and protocol on startup
* all API, streaming, multipart, and authenticated media requests run through
  the Rust shell
* the `vashti_session` cookie is stored through Android's native keystore and
  is never returned to frontend JavaScript
* private-storage vault keys exist only in the active frontend runtime
* each server exposes a durable installation UUID, used with the signed-in user
  to isolate local IndexedDB data between servers and accounts
* HTTPS certificates are validated normally; there is no certificate bypass
* plain HTTP requires an explicit local-network warning acknowledgement

The generated Android package identifier is `chat.vashti.app`. The shell
currently targets API 36 and supports Android API 24 or newer.

## Prerequisites

Install:

* a Java 17 or newer JDK, including `javac`
* Android SDK command-line tools
* Android platform and build tools
* Android NDK
* Rust Android targets

The scripts use `$ANDROID_HOME`, defaulting to `$HOME/Android/Sdk`, and the
newest installed NDK unless `$NDK_HOME` is set.

Example Rust targets:

```sh
rustup target add \
  aarch64-linux-android \
  armv7-linux-androideabi \
  i686-linux-android \
  x86_64-linux-android
```

Set a JDK explicitly when the system `java` command belongs to a JRE:

```sh
export JAVA_HOME=/path/to/jdk-21
export PATH="$JAVA_HOME/bin:$PATH"
```

Initialize generated Android files only when creating or intentionally
regenerating the project:

```sh
cd apps/vashti-android
npm --prefix ../vashti/web exec tauri -- android init --ci
```

Review generated-file changes before keeping them. Normal builds do not need to
run `android init` again.

## Development Build

Run or build from `apps/vashti-android`:

```sh
cd apps/vashti-android
npm --prefix ../vashti/web exec tauri -- android dev
```

Build a debug APK without launching a device:

```sh
cd apps/vashti-android
npm --prefix ../vashti/web exec tauri -- android build --debug --apk --ci
```

Tauri runs `npm run build:native` in the web project. That build writes
`apps/vashti/web/dist-native` and deliberately omits the PWA service worker.

## Release Signing

Keep signing material outside Git. The package script optionally reads
`.private/android-signing.env`, which is ignored by the repository:

```sh
VASHTI_ANDROID_KEYSTORE=/absolute/path/to/vashti-release.jks
VASHTI_ANDROID_KEY_ALIAS=vashti
VASHTI_ANDROID_KEYSTORE_PASSWORD=replace-me
VASHTI_ANDROID_KEY_PASSWORD=replace-me
```

The same release key must sign every public Android update. Losing or changing
that key prevents Android from installing future APKs over existing installs.

Build the universal signed artifact from the workspace root:

```sh
./apps/vashti/scripts/package-android-release.sh
```

Output:

```txt
apps/vashti/dist/release/vX.Y.Z/vashti-android.apk
apps/vashti/dist/release/vX.Y.Z/SHA256SUMS
apps/vashti/dist/release/vX.Y.Z/VERSION
```

The Android crate, Tauri config, and main Vashti crate must have matching
versions before packaging.

## Publishing

Build both public artifacts before using a one-time Hub upload key:

```sh
./apps/vashti/scripts/package-release.sh
./apps/vashti/scripts/package-android-release.sh
./apps/vashti/scripts/publish-release.sh
```

The publish script uploads the Linux archive and Android APK as one release.
After an admin promotes it, the stable Android download URL is:

```txt
https://vashti.chat/releases/latest/vashti-android.apk
```

## Device Test Checklist

Before promoting an Android prerelease:

1. Install the APK on a physical Android device.
2. Add an HTTPS server and verify login, logout, restart, and reconnect.
3. Add a LAN HTTP server and verify the warning is required.
4. Send, stream, cancel, retry, regenerate, edit, and branch messages.
5. Upload and view image and text attachments.
6. Create a private chat, restart the app, and verify only the same account and
   server can reopen it.
7. Switch between two saved servers and verify chats, sessions, and private
   storage do not cross between them.
8. Reset a test server at the same URL and verify its old native session and
   private-local namespace are not reused.
9. Install the next APK over the old one and verify app data remains available.
