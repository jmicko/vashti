# Vashti

Vashti is a lightweight self-hosted chat UI for Ollama. It is built to be simple to run on a local machine or on a LAN, with only a single binary installed with a simple bash script.

The goal is a smaller, simpler alternative for people who want a chat interface for Ollama without Docker stacks, Python environments, or a large platform to manage.

Check it out at [https://www.vashti.chat](https://www.vashti.chat)

## Status

Vashti is pre-release software. It is usable for testing, but the schema and release flow are still moving.

## Quick Install

On Linux, paste the following into a terminal:

```sh
curl -fsSL https://vashti.chat/install.sh | sh
```

That's it! The installer adds a systemd service and starts Vashti on port `7771`. After install, open the local or LAN URL printed by the script and create the first account. If there are no existing admins, that first account becomes the admin.

To update to the latest version, just run the same command again. The installer will pull the latest release, stop the service, replace the binary, and restart it. Just refresh the page afterwards and you're good to go.

For a specific version:

```sh
curl -fsSL https://vashti.chat/install.sh | VASHTI_VERSION=v0.1.5 sh
```

All releases are available at [https://vashti.chat/releases](https://vashti.chat/releases).

## Running From Source

Install Rust, Node, and npm first. Then from the repo root:

```sh
cargo run -p vashti
```

The frontend assets will be automatically built and embedded into the binary on run.

For frontend-only development:

```sh
cd apps/vashti/web
npm install
npm run dev
```

For a release-style build with embedded frontend assets:

```sh
./apps/vashti/scripts/package-release.sh
```

## Repository Layout

```text
apps/vashti      Main Vashti chat application
apps/vashti-hub  Release hosting site for vashti.chat
```

`vashti` is the app that users install. `vashti-hub` is the small release server used to publish builds and serve install/update metadata.

## Public Deployment

Vashti can run on plain HTTP for trusted local/LAN use. For public internet access, put it behind an HTTPS reverse proxy such as Caddy or nginx, then enable Public HTTPS Reverse Proxy mode in the admin settings.

It is ill-advised to expose Vashti's raw internal port directly to the public internet.

## License

Apache License 2.0. See [LICENSE](LICENSE).
