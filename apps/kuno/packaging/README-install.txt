Kuno — Linux systemd inference gateway and native image engine

This beta release provides the Ollama gateway, usage tracking, and
optional managed GPU workers. Adaptive idle profiling is still pending.
This package includes the experimental native image engine. Model weights and
GPU configuration are separate setup steps; see docs/kuno-images.md. CUDA and
Vulkan need working host GPU drivers. The CUDA-enabled package requires the
NVIDIA driver library libcuda.so.1 even when using another GPU for images.
The optional Intel SYCL worker is a separate verified runtime, installed with
install-image-runtime and selected using configure-image. See
docs/kuno-managed-sycl.md and docs/kuno-release-handoff.md for the exact bundle
checksum, install commands and tested platform limits. No Python image server
is required; Python packaging helpers are only used by release developers.
The VERSION file identifies the coordinated Vashti release containing this build.

Installation
------------
From this extracted directory, install and start Kuno:

  sudo ./kuno install

Installation warns that inference will briefly be interrupted, then waits for
Continue? [y/N] before changing services. Enter or n cancels. You do not need to
close connected applications; clients can reconnect after installation.
The installer verifies service ownership, listener ownership, and health itself.
Install, upgrade, recover, and takeover-restore all prompt. Unattended scripts must pass
--yes explicitly; noninteractive input without that flag is rejected.

For an optional diagnostic report before or after installation:

  sudo ./kuno doctor

Existing clients retain their Ollama URL. The installer discovers the actual
listener (including dual-stack IPv6), moves Ollama to a free loopback port, and
starts Kuno on the original listener. Kuno is enabled at system startup and runs
without root using systemd DynamicUser. Installation alone requires sudo.

The installer preserves OLLAMA_MAX_LOADED_MODELS, including an existing value of 1.
Optional managed workers use their own model limits and GPU selection. It disables
Ollama request logging and raw stdout/stderr in its own removable service drop-in.
No model files are downloaded, moved, or deleted.

Supported: a running native Ollama systemd service, root-controlled executable
and unit files, direct `ollama serve`, one host-network listener, and ordinary
Environment settings. EnvironmentFile-based settings, wrappers with differing
runtime settings, containers, ambiguous listeners, and existing unrelated Kuno
services are rejected with a diagnostic code instead of guessed at.

Paths
-----
/usr/local/lib/kuno/kuno              Installed executable
/etc/kuno/config.json                Gateway configuration
/etc/kuno/control-token              Root-only control credential
/etc/systemd/system/kuno.service     Unprivileged gateway unit
/var/lib/kuno                        Usage and recovery state (systemd-managed)
/var/lib/kuno-setup                  Root-only installation/rollback journal

Cancellation recovery
---------------------
Kuno drains abandoned responses without saving their contents, releasing
admission when upstream completion is confirmed. Inference and drain deadlines
are disabled by default, preserving long Ollama requests. Uncertain upstream
completion can temporarily block admission. The installed recovery timer checks
roughly every 30 seconds, with a renewable attempt budget and failure backoff;
it never treats a long active request alone as a reason to restart.
For explicit recovery:

  sudo /usr/local/lib/kuno/kuno recover

This restarts managed Ollama and Kuno, unloading Ollama's resident models. It
clears the empty recovery marker only after stopping Ollama's runner group.
It retains usage history and does not touch other GPU applications. A confirmed
host reboot also clears uncertainty recorded under a previous Linux boot ID.
Restarting only Kuno does not; legacy markers need this explicit recovery once.

Undo installation or recover an interrupted installation
-------------------------------------------------------
This briefly interrupts inference. Use the original downloaded binary if the
installed copy is missing:

  sudo ./kuno takeover-restore

Restore removes unchanged Kuno-owned files, disables Kuno, and restores Ollama's
original listener and running state. It keeps usage history and the setup journal.
Later administrator edits to owned files cause a refusal instead of being erased.
Restore stops both services before clearing any uncertain in-flight marker.

Diagnostics
-----------
`resources` returns a read-only GPU-memory and loaded-model report:
  ./kuno resources

It automatically uses the installed configuration when present. Intel queries
use DRM render nodes, NVIDIA queries use the installed NVML library. Generated
services join the existing render group when available. Missing or restricted
measurements remain unknown; no model is loaded or unloaded. Ollama's reported
VRAM allocation is not a physical GPU-assignment guarantee.

`doctor` returns content-free JSON. Without sudo, only unprivileged discovery is
verified. Privileged installation and post-install checks verify actual processes,
listeners, effective model limits, and both sides of the proxy.

For a service startup error, these fields are useful without exposing prompts:
  sudo systemctl show kuno.service ollama.service \
    --property=ActiveState,SubState,Result,ExecMainStatus

Kuno's own classified diagnostics:
  sudo journalctl -u kuno.service -n 50 --no-pager

Keep this binary for restoration. Re-running install with the same binary is
idempotent. Upgrade using the newly downloaded binary:

  sudo ./kuno upgrade

This asks once before replacing the installation and restores the previous files
if the new version fails activation. Interrupted upgrade processes use the same
restore journal described above. Normal runtime never automatically rolls back to
direct Ollama access.

Validation: automated proxy and transaction fault tests, staged systemd-unit
validation, and real privileged installation on Linux Mint passed. Post-install
diagnostics and IPv4/IPv6 metadata forwarding passed on 2026-09-22. Live reboot
remains untested. Single-command upgrade, recovery, restore/reinstall, and
remote streamed generation and cancellation passed on 2026-09-23. Host-wide swap/hibernation privacy
and GPU execution placement are not certified by this gateway release.

Experimental managed GPU workers:
  sudo ./kuno configure-managed --workers /absolute/path/to/workers.json
  sudo ./kuno configure-managed --disable

The workers file is a PoolConfig JSON object with reserve_bytes and workers.
See apps/kuno/README.md for its schema and current endpoint coverage. This mode
requires explicit device selection. The installer supplies private worker homes
and a read-only model mount; worker discovery verifies GPU identity before
activation. Failed activation restores the prior configuration/unit/binary.
Like install/upgrade, these commands wait for interruption confirmation unless
--yes is explicitly supplied. Ordinary upgrades preserve the managed setting.
