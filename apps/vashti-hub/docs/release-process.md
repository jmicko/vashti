# Vashti Release Process

Vashti Hub keeps promoted releases as the official public history. Uploads remain staged as
prereleases until an admin promotes or deletes them.

## Normal Release Flow

1. Set the same version in `apps/vashti/Cargo.toml`, `apps/vashti-android/Cargo.toml`,
   and `apps/vashti-android/tauri.conf.json`, then refresh Cargo.lock. Beta labels such as
   `0.13.0-beta.1` are supported. Increment `bundle.android.versionCode` for every
   published APK, including beta-to-beta and beta-to-final upgrades. It is an independent
   monotonically increasing integer: 0.12.0 used 12000; 0.13.0-beta.1 uses 13001.
   Keep the explicit override in future versions; never reset it to Tauri's derived default.
2. Write release notes while working in:

   ```txt
   apps/vashti/release-notes-latest.md
   ```

3. When the version is final, copy or move the notes to:

   ```txt
   apps/vashti/release-notes/vX.Y.Z.md
   ```

4. Build the Linux release package:

   ```sh
   ./apps/vashti/scripts/package-release.sh
   ```

5. Build the signed universal Android APK when publishing the Android app:

   ```sh
   ./apps/vashti/scripts/package-android-release.sh
   ```

6. Build the Kuno package with `apps/kuno/scripts/package-release.sh`, pointing
   `KUNO_SD_SOURCE_DIR` and `KUNO_SD_BUILD_DIR` at the prepared native engine build.
   Verify all three packages are present, then run `CHECK_ONLY=1 apps/vashti/scripts/publish-release.sh`.
   Only then open the Hub admin page and create a one-time upload key.
7. Publish all artifacts found for that version:

   ```sh
   ./apps/vashti/scripts/publish-release.sh
   ```

   The script automatically uses the versioned notes file if it exists, otherwise it falls back to
   `release-notes-latest.md`.

   Linux, Android, and Kuno are uploaded in one request and belong to the same staged release. A one-time
   key cannot be reused to add a second artifact later.

8. Test the staged prerelease from the Releases page using the version-specific install command and
   APK download.
9. After physical Android and packaged Linux/Kuno validation, upload a final version without
   the beta suffix and promote it to latest. Suffix-bearing beta versions remain on the
   prerelease channel and cannot be promoted.

## Release Signing Key

Every uploaded artifact is signed before Hub accepts it. By default the publisher reads the
Ed25519 private key from:

```txt
.private/keys/vashti-release-signing-key.pem
```

That directory is ignored by Git. Keep the file mode restricted to the release operator and keep
an encrypted offline backup outside this workstation. Losing the private key prevents existing
Vashti installations from trusting future managed updates; publishing with a replacement key
requires an explicit key-rotation release or a manual recovery rollout.

To use a key stored elsewhere, set `VASHTI_RELEASE_SIGNING_KEY` to its absolute path. The publish
script derives its public key and refuses to continue unless it matches the public key compiled
into Vashti and Hub.

## Staged Prereleases

Hub can retain multiple prereleases so a candidate can be installed and tested before a later
candidate is uploaded. Uploading a new version never removes an earlier staged version. Uploading
the same version twice is rejected; versioned artifacts are immutable after Hub accepts them.

When promoting a release, the admin can optionally incorporate earlier prereleases newer than the
current stable release. Hub deterministically appends their release notes to the promoted release,
then deletes those incorporated release records and artifact files. The confirmation dialog lists
every version that will be removed. Leaving the option unchecked promotes only the selected version
and preserves all other prereleases.

## Installs And Updates

The default install command installs the promoted latest release:

```sh
curl -fsSL https://vashti.chat/install.sh | sh
```

The Releases page shows version-specific install commands:

```sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_VERSION=v0.1.1 sh
```

The installer checks the currently installed Vashti binary when it can. Downgrades are refused by
default because older binaries may not understand newer SQLite migrations.

Emergency downgrade override:

```sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_VERSION=v0.1.1 VASHTI_ALLOW_DOWNGRADE=1 sh
```

Use that only after backing up or intentionally resetting the data directory.
