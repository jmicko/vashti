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

6. Build the Kuno package with `apps/kuno/scripts/package-release.sh`. The Rust
   image and video worker ships in the Kuno executable, including for supported
   Intel GPUs. Do not build or upload a separate runtime. Historical runtime
   artifacts stay downloadable with their older releases.
   APK filenames include the version (`vashti-0.13.0-beta.5.apk`); older releases
   with `vashti-android.apk` remain supported.
   Verify every intended package is present, then run
   `REQUIRE_COMPLETE_RELEASE=1 CHECK_ONLY=1 apps/vashti/scripts/publish-release.sh`.
   Only then open the Hub admin page and create a one-time upload key.
   Check the complete upload against `VASHTI_HUB_MAX_UPLOAD_BYTES` and any reverse-proxy
   body limit using the actual package sizes and allowing for multipart overhead.

   When Kuno is unchanged, skip its build and publish only Vashti instead. Build
   the matching Linux server and Android APK, then run:

   ```sh
   RELEASE_SCOPE=vashti REQUIRE_COMPLETE_RELEASE=1 CHECK_ONLY=1 apps/vashti/scripts/publish-release.sh
   RELEASE_SCOPE=vashti REQUIRE_COMPLETE_RELEASE=1 apps/vashti/scripts/publish-release.sh
   ```

   `RELEASE_SCOPE=vashti` explicitly excludes Kuno,
   even if old files are present in the release directory. It does not rebuild,
   rename, or bump Kuno. `RELEASE_SCOPE=kuno` publishes only the inference host
   package; `auto` (the default) retains the historical behavior
   of uploading all discovered artifacts. `all` has the same discovery behavior.
   Complete-release checks apply to the selected scope. Deploy the updated Hub
   before using partial releases; older Hubs assume every component shares the
   newest release version.
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

The **Roll in earlier prereleases** action performs that consolidation without
promotion, including for suffix-bearing beta candidates. The confirmation lists
the exact releases to remove and their estimated storage size. Release notes
are appended deterministically, and the candidate remains on the prerelease
channel; stable does not change. Notes and release-record changes are atomic.
If the eligible set changes while the dialog is open, it must be reopened before
anything is deleted. Filesystem cleanup failures are reported, not counted as
successful space recovery.

Only fully superseded prereleases are removed. A release containing the newest
build of any target (for example, Kuno or the optional Intel runtime) is retained
and listed in the confirmation. Its files, notes, original version and signatures
stay intact. This avoids re-uploading unchanged large binaries or breaking their
download links. Older redundant Kuno-bearing betas can still be removed; the
newest needed Kuno build cannot be discarded just to remove its beta entry.
No stable release or newer candidate is included in cleanup.

Managed update lookup is per channel **and target**, not one global version:
`/api/updates/prerelease/kuno-linux-x86_64` can return beta.3 while the matching
Vashti targets return beta.4. Signatures always describe the original artifact
version. A staged Kuno build never becomes stable merely because a newer
Vashti-only release is promoted.

Fresh installers resolve a component-specific stable version through
`/releases/latest/{target}/VERSION`, then use that version's artifacts and
checksums. The Hub's served Kuno installer adapts its existing default resolver
to this endpoint without modifying Kuno source. Explicit `KUNO_VERSION` and
`VASHTI_VERSION` selections keep their exact-version behavior. Legacy latest
download filenames resolve to the newest stable release containing that file;
`vashti-android.apk` also supports the versioned APK filename.

## Local Validation

`cargo test -p vashti-hub --locked` covers consolidation, rollback, stale
confirmations, admin/origin checks, component channels and downloads. Publisher
scope checks use `node --test apps/vashti/scripts/release-packaging.test.mjs`.

For browser-only inspection, run
`cargo test -p vashti-hub manual_hub_browser_fixture --locked -- --ignored --nocapture`
and open `http://127.0.0.1:17884/admin`. This uses a new temporary database and
four-byte synthetic artifacts, never a live database or release signing key.
Its test-only login matches the `rollup_test_state` helper. Inspect mobile and
desktop previews, cancellation, retained-component labels and install controls.
Stop the fixture with Ctrl-C. It removes its temporary data on graceful exit.

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
