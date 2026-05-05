# Vashti Deployment

Vashti is not at `v1.0.0` yet. Until then, `v0.x.y` builds should be treated as prereleases.

The deployment pipeline is owned by the Vashti operator. `vashti.chat` runs a small Rust release server that serves the public website, install script, release artifacts, checksums, and basic download stats.

## Workspace Layout

```txt
apps/
  vashti/
    Cargo.toml
    docs/
    packaging/
    scripts/
    src/
    migrations/
    web/
  vashti-release-site/
    Cargo.toml
    docs/
    packaging/
    scripts/
    src/
    migrations/
    static/
```

The root `Cargo.toml` is a Cargo workspace manifest.

## Release Site

The release site binary is `vashti-release-site`.

Default settings:

* bind: `127.0.0.1:7781`
* dev data from workspace root: `apps/vashti-release-site/data`
* packaged/prod data: `/var/lib/vashti-release-site` through `VASHTI_RELEASE_DATA_DIR`
* database: `release-site.db`
* artifacts: `artifacts/`

For production, run it behind nginx or Caddy. Example nginx location:

```nginx
server {
    listen 443 ssl http2;
    server_name vashti.chat;

    location / {
        proxy_pass http://127.0.0.1:7781;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

On first startup, the release site creates an upload token and writes it to:

```txt
/var/lib/vashti-release-site/upload-token.txt
```

Store that token locally in:

```txt
apps/vashti/.secrets/release-token
```

Do not commit app `.secrets/` directories.

## Public Release Routes

The release site serves:

```txt
https://vashti.chat/
https://vashti.chat/admin
https://vashti.chat/install.sh
https://vashti.chat/releases/latest/VERSION
https://vashti.chat/releases/latest/SHA256SUMS
https://vashti.chat/releases/latest/vashti-linux-x86_64.tar.gz
https://vashti.chat/releases/v0.1.0/VERSION
https://vashti.chat/releases/v0.1.0/SHA256SUMS
https://vashti.chat/releases/v0.1.0/vashti-linux-x86_64.tar.gz
```

The admin page can upload artifacts, rotate the upload token, and inspect basic download counts.

## Versioning

Use the package version from `apps/vashti/Cargo.toml`.

Before `v1.0.0`:

* patch versions are bug fixes and small safety fixes
* minor versions may include migrations and feature changes
* downgrade support is not promised

The release site does not invent versions. It validates uploaded `vMAJOR.MINOR.PATCH` labels and marks the highest uploaded version as `latest`.

## Build Vashti

Package the main app:

```sh
./apps/vashti/scripts/package-release.sh
```

This writes:

```txt
apps/vashti/dist/release/v0.1.0/vashti-linux-x86_64.tar.gz
apps/vashti/dist/release/v0.1.0/SHA256SUMS
apps/vashti/dist/release/v0.1.0/VERSION
```

## Publish a Release

Upload the package to the release site:

```sh
./apps/vashti/scripts/publish-release.sh
```

Useful overrides:

```sh
VASHTI_RELEASE_SITE_URL=https://staging.vashti.chat ./apps/vashti/scripts/publish-release.sh
VASHTI_RELEASE_TOKEN=... ./apps/vashti/scripts/publish-release.sh
NOTES="Fix login flow" ./apps/vashti/scripts/publish-release.sh
```

## Build the Release Site

Package the release site binary for the VM:

```sh
./apps/vashti-release-site/scripts/package-release-site.sh
```

This writes:

```txt
apps/vashti-release-site/dist/release-site/vashti-release-site
apps/vashti-release-site/dist/release-site/vashti-release-site.service
```

Install those on the VM:

```sh
sudo install -m 0755 apps/vashti-release-site/dist/release-site/vashti-release-site /usr/local/bin/vashti-release-site
sudo install -m 0644 apps/vashti-release-site/dist/release-site/vashti-release-site.service /etc/systemd/system/vashti-release-site.service
sudo systemctl daemon-reload
sudo systemctl enable --now vashti-release-site
```

## Install Script

The public install command is:

```sh
curl -fsSL https://vashti.chat/install.sh | sh
```

The installer:

* detects Linux architecture
* downloads the release archive from `https://vashti.chat/releases`
* verifies `SHA256SUMS`
* installs `vashti` to `/usr/local/bin/vashti`
* creates a `vashti` system user when needed
* stores data in `/var/lib/vashti`
* installs and starts a systemd service

Environment overrides:

```sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_VERSION=v0.1.0 sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_RELEASE_BASE_URL=https://updates.example.com/releases sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_BIND=127.0.0.1:7771 sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_DATA_DIR=/srv/vashti sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_NO_SYSTEMD=1 sh
```

## Update Path

Until in-app update checks exist, update by rerunning:

```sh
curl -fsSL https://vashti.chat/install.sh | sh
```

The installer replaces the binary and restarts the systemd service without touching `/var/lib/vashti`.
