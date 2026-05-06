# Vashti Deployment

Vashti is not at `v1.0.0` yet. Until then, `v0.x.y` builds should be treated as prereleases.

The deployment pipeline is owned by the Vashti operator. `vashti.chat` runs Vashti Hub, a small Rust server that serves the public website, install script, release artifacts, checksums, and basic download stats.

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
  vashti-hub/
    Cargo.toml
    docs/
    packaging/
    scripts/
    src/
    migrations/
    static/
```

The root `Cargo.toml` is a Cargo workspace manifest.

## Vashti Hub

The hub binary is `vashti-hub`.

Default settings:

* bind: `127.0.0.1:7781`
* dev data from workspace root: `apps/vashti-hub/data`
* packaged/prod data: `/var/lib/vashti-hub` through `VASHTI_HUB_DATA_DIR`
* database: `hub.db`
* artifacts: `artifacts/`

Hub-specific environment variables:

* `VASHTI_HUB_BIND`: listen address, default `127.0.0.1:7781`
* `VASHTI_HUB_DATA_DIR`: data directory, default `apps/vashti-hub/data` from the workspace root
* `VASHTI_HUB_COOKIE_SECURE`: set to `true` when the browser reaches Hub through HTTPS
* `VASHTI_HUB_TRUST_PROXY_HEADERS`: set to `true` only when Hub is reachable only through a trusted reverse proxy
* `VASHTI_HUB_MAX_UPLOAD_BYTES`: maximum release artifact upload size

For production, run it behind nginx or Caddy. If nginx reaches Hub over WireGuard, bind Hub to
the WireGuard IP, for example `VASHTI_HUB_BIND=10.8.0.2:7781`, and firewall the host so the raw
Hub port is not reachable from the public network.

Example nginx location:

```nginx
server {
    listen 443 ssl http2;
    server_name vashti.chat;

    location / {
        proxy_pass http://10.8.0.2:7781;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

On first startup, Vashti Hub is unclaimed and does not serve release files yet. Open `/admin`.
Hub creates an admin setup key and writes it to:

```txt
/var/lib/vashti-hub/admin-setup-key.txt
```

The admin setup page shows the exact path and a `cat` command. Use that key to create the first
admin password.

After Hub is claimed, the admin page can create one-time upload keys. Upload keys work once, expire
after 10 minutes, and are kept only in Hub memory. Generate one immediately before running the
release publish script.

If the admin password is forgotten, `/admin` can create a reset key at:

```txt
/var/lib/vashti-hub/admin-reset-key.txt
```

Creating a reset key does not disable the current admin password. The reset key plus a new password
must be submitted before the password changes.

## Public Release Routes

Vashti Hub serves:

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

The admin page can create one-time upload keys and inspect basic download counts.

## Versioning

Use the package version from `apps/vashti/Cargo.toml`.

Before `v1.0.0`:

* patch versions are bug fixes and small safety fixes
* minor versions may include migrations and feature changes
* downgrade support is not promised

Vashti Hub does not invent versions. It validates uploaded `vMAJOR.MINOR.PATCH` labels and marks the highest uploaded version as `latest`.

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

Upload the package to Vashti Hub:

```sh
./apps/vashti/scripts/publish-release.sh
```

The script asks for a one-time upload key. Create the key from the Hub admin page and paste it when
prompted.

Useful overrides:

```sh
VASHTI_HUB_URL=https://staging.vashti.chat ./apps/vashti/scripts/publish-release.sh
VASHTI_HUB_UPLOAD_KEY=... ./apps/vashti/scripts/publish-release.sh
NOTES="Fix login flow" ./apps/vashti/scripts/publish-release.sh
```

## Build Vashti Hub

Package the hub binary for the VM:

```sh
./apps/vashti-hub/scripts/package-hub.sh
```

This writes:

```txt
apps/vashti-hub/dist/hub/vashti-hub
apps/vashti-hub/dist/hub/vashti-hub.service
apps/vashti-hub/dist/hub/install-hub.sh
```

Copy `apps/vashti-hub/dist/hub/` to the Hub VM, then run the installer from that directory.
The installer is interactive when run from a terminal:

```sh
sudo ./install-hub.sh
```

It asks where to install the binary, where to store Hub data, which detected network address to
bind, whether to enable public HTTPS reverse proxy mode, and the maximum release upload size. For a
WireGuard-only Hub listener, choose the WireGuard address, for example `10.8.0.2:7781`.

The installer installs `/usr/local/bin/vashti-hub`, writes `/etc/systemd/system/vashti-hub.service`,
creates/reuses the `vashti-hub` system user, prepares `/var/lib/vashti-hub`, and starts the service.

Useful non-interactive installer overrides:

```sh
VASHTI_HUB_NO_PROMPT=1 sudo ./install-hub.sh
VASHTI_HUB_BIND=127.0.0.1:7781 sudo ./install-hub.sh
VASHTI_HUB_DATA_DIR=/srv/vashti-hub sudo ./install-hub.sh
VASHTI_HUB_INSTALL_DIR=/opt/vashti/bin sudo ./install-hub.sh
VASHTI_HUB_USER=vashti-hub sudo ./install-hub.sh
VASHTI_HUB_NO_SYSTEMD=1 ./install-hub.sh
```

For scripted WireGuard/reverse-proxy installs:

```sh
sudo VASHTI_HUB_NO_PROMPT=1 \
  VASHTI_HUB_BIND=10.8.0.2:7781 \
  VASHTI_HUB_COOKIE_SECURE=true \
  VASHTI_HUB_TRUST_PROXY_HEADERS=true \
  ./install-hub.sh
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
