# Vashti Release Process

Vashti Hub keeps promoted releases as the official public history. Uploads are staged as a single
prerelease until an admin promotes them.

## Normal Release Flow

1. Update `apps/vashti/Cargo.toml` to the next version.
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

6. Open the Hub admin page and create a one-time upload key.
7. Publish all artifacts found for that version:

   ```sh
   ./apps/vashti/scripts/publish-release.sh
   ```

   The script automatically uses the versioned notes file if it exists, otherwise it falls back to
   `release-notes-latest.md`.

   Linux and Android are uploaded in one request and belong to the same staged release. A one-time
   key cannot be reused to add a second artifact later.

8. Test the staged prerelease from the Releases page using the version-specific install command and
   APK download.
9. If both artifacts work, promote the release to latest from the Hub admin page.

## Staged Prereleases

Hub keeps only one prerelease. Uploading another prerelease before promotion replaces the previous
staged prerelease and removes its artifact files from Hub storage.

This keeps the official release history clean. Only promoted releases remain as durable public
history.

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
