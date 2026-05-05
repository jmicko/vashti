Vashti release package
======================

This package contains a single self-contained Vashti binary with embedded frontend assets.

Quick start:

  sudo install -m 0755 vashti /usr/local/bin/vashti
  sudo useradd --system --home-dir /var/lib/vashti --shell /usr/sbin/nologin vashti
  sudo mkdir -p /var/lib/vashti
  sudo chown vashti:vashti /var/lib/vashti
  sudo install -m 0644 vashti.service /etc/systemd/system/vashti.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now vashti

Then open:

  http://SERVER_IP:7771

Operational notes:

- Vashti stores SQLite data, uploads, and local app state under /var/lib/vashti by default.
- The systemd unit binds to 0.0.0.0:7771 for LAN access.
- For public internet access, put Vashti behind an HTTPS reverse proxy and enable Public HTTPS Reverse Proxy mode in the admin settings.
- Do not expose the raw 7771 port directly to the public internet.
- If network settings break login, create /var/lib/vashti/recover_network.txt and restart Vashti.
