Kuno 0.1.0 — Linux systemd inference gateway

This development release provides the Ollama gateway and usage tracking.
GPU placement, adaptive profiling, and image generation are not implemented yet.

Installation
------------
1. Close applications making inference requests to Ollama. Installation restarts
   Ollama and briefly interrupts its existing listener. Keep them closed until
   installation finishes. An empty /api/ps response alone does not prove idleness.
2. From this extracted directory, inspect the host:

     sudo ./kuno doctor

3. Install and start Kuno:

     sudo ./kuno install --quiesced

   --quiesced confirms that you closed inference clients for this maintenance
   window. The installer also checks for open connections and listener ownership.
4. Verify:

     sudo /usr/local/lib/kuno/kuno doctor

Existing clients retain their Ollama URL. The installer discovers the actual
listener (including dual-stack IPv6), moves Ollama to a free loopback port, and
starts Kuno on the original listener. Kuno is enabled at system startup and runs
without root using systemd DynamicUser. Installation alone requires sudo.

The installer preserves OLLAMA_MAX_LOADED_MODELS, including an existing value of 1.
Kuno will manage that ceiling when the GPU scheduler is implemented. It disables
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
Until automatic GPU/runtime reconciliation is implemented, an interrupted
inference request blocks further admission. Close inference clients, then run:

  sudo /usr/local/lib/kuno/kuno recover --quiesced

This restarts managed Ollama and Kuno, unloading Ollama's resident models. It
clears the empty recovery marker only after stopping Ollama's runner group.
It retains usage history and does not touch other GPU applications.

Undo installation or recover an interrupted installation
-------------------------------------------------------
Close inference clients, then use the original downloaded binary if the installed
copy is missing:

  sudo ./kuno takeover-restore --quiesced

Restore removes unchanged Kuno-owned files, disables Kuno, and restores Ollama's
original listener and running state. It keeps usage history and the setup journal.
Later administrator edits to owned files cause a refusal instead of being erased.
If ordinary runtime recovery is required, run `recover --quiesced` first.

Diagnostics
-----------
`doctor` returns content-free JSON. Without sudo, only unprivileged discovery is
verified. Privileged installation and post-install checks verify actual processes,
listeners, effective model limits, and both sides of the proxy.

For a service startup error, these fields are useful without exposing prompts:
  sudo systemctl show kuno.service ollama.service \
    --property=ActiveState,SubState,Result,ExecMainStatus

Kuno's own classified diagnostics:
  sudo journalctl -u kuno.service -n 50 --no-pager

Keep this binary for restoration. Re-running install with the same binary is
idempotent; upgrading a changed installation requires restore followed by install
in this first release. Normal runtime never automatically rolls back to direct
Ollama access.

Validation limit: this package has automated proxy and transaction fault tests
and staged systemd-unit validation. Real privileged installation and reboot must
still be confirmed on the target host. Host-wide swap/hibernation privacy and GPU
execution placement are not certified by this gateway release.
