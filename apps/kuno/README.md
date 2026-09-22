# Kuno development gateway

Kuno's first implementation provides a local Ollama reverse proxy, conservative
request admission, an authenticated control listener, persistent usage counts,
and an installable systemd gateway with reversible Ollama port takeover. This
development release does not yet provide GPU placement or image generation.
The [specification](../../docs/kuno-spec.md) defines the full product and milestones.

## Install on this Linux host

Build a native release package:

```sh
./apps/kuno/scripts/package-release.sh
```

Install with:

```sh
sudo ./dist/kuno-linux-x86_64/kuno install
```

Installation warns that inference will briefly be interrupted, then waits for
`Continue? [y/N]` before changing services. Enter or `n` cancels. Connected clients
can reconnect afterward; you do not need to close them first. The installer checks
actual listener ownership, native executable identity, effective environment, and
competing services before changing anything. It stops Ollama's runner group during
cutover rather than trying to infer idleness from open TCP connections.

`install`, `recover`, and `takeover-restore` all require this confirmation.
Unattended scripts must explicitly pass `--yes`; redirected input does not imply
consent. The website installer should display the interruption warning and obtain
confirmation before invoking an unattended command.

Installation includes health checks. For a separate diagnostic report later, run
`sudo /usr/local/lib/kuno/kuno doctor`.

The installer preserves the actual IPv4/IPv6 listener, moves Ollama to a free
loopback port, installs Kuno using an unprivileged systemd `DynamicUser`, and
enables Kuno at boot. Kuno starts paused; only after both listeners and upstream
health are verified does installation activate admission. Subsequent boots wait
for upstream health and use a root-controlled activation marker. The original
loaded-model ceiling remains effective until M2 supplies the GPU scheduler.

Install/restore require sudo; serving inference does not. Supported installations
use a running native `ollama serve` systemd service with root-controlled unit and
executable files. Containers, environment files, differing wrapper environments,
and ambiguous listeners receive diagnostic errors. `--service NAME.service`
selects a different native Ollama service. The current default host has a
dual-stack `[::]:11434` listener even though `OLLAMA_HOST` says `0.0.0.0:11434`.

Cancellation recovery (also briefly interrupts inference):

```sh
sudo /usr/local/lib/kuno/kuno recover
```

This explicitly restarts the managed gateway and Ollama, clears the empty
in-flight marker only after Ollama's runner group stops, and retains usage data.
It unloads Ollama's resident models. It never restarts unrelated GPU applications.

Undo installation, or recover an interrupted installer, using the release binary:

```sh
sudo ./dist/kuno-linux-x86_64/kuno takeover-restore
```

Keep that release binary available in case restoration has already removed the
installed copy. Restore disables Kuno, removes unchanged owned files, and returns
Ollama to its original listener and active state. Usage data and the protected
setup journal remain. Restore stops both services before clearing uncertain runtime
work; clients can reconnect after Ollama returns to its original port. Changed
installed files are not overwritten.
Re-running the same install is idempotent; this first release uses restore/install
for upgrades rather than overwriting a changed deployment.

See [packaged installation instructions](packaging/README-install.txt) for paths
and content-free diagnostic commands. On 2026-09-22, installation and privileged
post-install diagnostics passed on the target Linux Mint host. Both IPv4 and IPv6
forwarding matched upstream metadata, and the service is enabled at boot. Real
reboot, restore, and operator recovery still need live acceptance testing.

## Run alongside Ollama

From the repository root, with Ollama already listening on `127.0.0.1:11434`:

```sh
export KUNO_CONTROL_TOKEN="$(openssl rand -hex 32)"
cargo run -p kuno -- serve --config apps/kuno/kuno.example.json
```

Kuno listens on `127.0.0.1:11433`, with its control API on `127.0.0.1:11432`.
Point a test client's Ollama URL at port 11433. Existing clients on 11434 continue
to reach Ollama directly and therefore do not participate in admission yet.
For example, in another terminal:

```sh
OLLAMA_HOST=http://127.0.0.1:11433 ollama list
curl --fail http://127.0.0.1:11433/api/version
```

The example token command uses OpenSSL only to generate a development credential;
Kuno itself does not invoke OpenSSL. Supply a random 32–256-character token using
letters, digits, `-`, `_`, or `.`. Do not put the token in a tracked configuration.
Only the control listener requires it. The compatibility listener preserves
Ollama's client authentication and origin behavior.
`KUNO_CONTROL_TOKEN_FILE` can supply the credential instead, including through
systemd's `LoadCredential`. Setting both credential sources is rejected.

Configuration is JSON, with unknown keys rejected. `check --config FILE` validates
it without starting a listener. Upstream and control addresses must be literal
loopback socket addresses. All three ports must differ. Set an absolute,
dedicated `state_dir` outside the checkout when running elsewhere; relative paths
are resolved against the process's working directory. Reuse that same directory
on restarts. Each state directory permits one Kuno process at a time.

## What works

- Opaque HTTP forwarding to one configured local upstream, including native and
  compatibility APIs, unknown fields/routes, binary uploads, upstream errors,
  incremental NDJSON/SSE bodies, and backpressure. No request-body buffering or
  payload spooling is added. Redirects and possibly executed requests are never
  followed/replayed by Kuno.
- Exact metadata-route exemptions; other requests share one exclusive admission
  lease through upstream body completion. Catalog changes and unknown routes
  use the same gate. This is host-wide serialization, not GPU-aware scheduling.
- Bounded waiting, queue/header/stream-idle timeouts, and content-free errors.
  Stream idle timeouts apply when polling upstream; downstream backpressure can
  retain a lease longer. HTTP protocol upgrades/CONNECT are explicitly unsupported.
- A separate bearer-authenticated control surface with status and pause/resume.
  The compatibility listener does not expose Kuno control handlers.
- A fixed, empty in-flight marker written durably before dispatch. Normal upstream
  completion clears it; uncertain cancellation or a crash leaves it in place.
  Restarting Kuno does not bypass a retained recovery requirement.
- Read-only systemd discovery that proposes a private upstream port and a
  Kuno-owned drop-in, while preserving the current loaded-model cap.
- Persistent real-use counts and timestamps for catalog-verified model digests.
  The bounded request observer retains only model identifiers and input presence;
  it skips nested prompt/image contents without copying them.

The route inventory was checked against [Ollama v0.34.2](https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go).
Unknown endpoints that start asynchronous work need adapters before their GPU
lifetime can be managed. Completing an HTTP response is not proof that such a
future asynchronous operation has finished.

## Control API

All routes below are available only on the control listener and require
`Authorization: Bearer <KUNO_CONTROL_TOKEN>`:

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/kuno/v1/status` | Version, mode, admission state, and capability flags |
| GET | `/kuno/v1/usage` | Aggregate model calls, last-use times, and persistence health |
| POST | `/kuno/v1/admission/pause` | Reject new managed work; let active work finish |
| POST | `/kuno/v1/admission/resume` | Resume admission unless upstream recovery is required |

Pausing is not an immediate GPU reservation. Wait until `active` is false, and
ensure `recovery_required` is false, before treating the upstream as quiescent.
Loaded models may still occupy GPU memory. No image worker is implemented yet.

Cancellation closes the upstream HTTP connection, but Kuno cannot yet verify
that Ollama has stopped computing. A dropped or failed managed response therefore
sets `recovery_required`. Further managed requests receive HTTP 503; metadata
and control remain available. `resume` deliberately does not clear this state.

For an uninstalled development gateway, recovery is manual: stop Kuno, establish that the upstream has no
active work (for a disposable test upstream, stop/restart it), then remove only
`upstream-in-flight` from the configured state directory and restart Kuno. Do not
reset an actively used Ollama instance merely to clear this marker. Automated
runtime reconciliation and safe recovery belong to M2. Changing state directories
or deleting a marker without establishing upstream completion defeats the guard.

SIGINT/SIGTERM stops new admission and allows up to 30 seconds for draining. A
forced interruption retains the marker. The currently active model may remain
resident in Ollama; resident models are not the same as active requests.

## Usage tracking

Known JSON inference requests with nonempty input and successful upstream response
headers count as real calls. Counts describe accepted attempts, including attempts
later cancelled or failed while streaming. Metadata and empty-input preload/unload
requests do not count. An observer handles arbitrarily large nested input with a
small fixed capture bound; it does not buffer the request to parse its settings.

Only identities found in `/api/tags` can be persisted. The catalog is cached for
up to 60 seconds and invalidated by catalog mutations and opaque operations.
Aliases for the same digest share a counter. Unknown identities increment only
an unattributed counter; the supplied name is discarded. Future profiling calls
must use the internal worker path and must never call the real-use recorder.

Seven-day eligibility requires recent real use and the same digest in the current
catalog. Expiry does not delete model files or usage history. History is bounded
to 4,096 identities, evicting the least recently used entry only at that limit.
Atomic writes preserve the previous file after an interrupted replacement. A
write failure appears in the usage API; corrupt existing history stops startup
instead of silently erasing it.

This first observer covers JSON endpoints, not multipart audio, compressed bodies,
or future unknown formats. Ambiguous duplicate model fields and incomplete bodies
are not attributed. These requests still proxy normally; they do not enroll a
model for idle profiling. Pre-existing use before Kuno started is unknown.

## Takeover discovery

```sh
cargo run -p kuno -- takeover-plan
```

This command inspects `ollama.service`; `--service NAME.service` selects another
systemd service. It prints a JSON proposal and unfinished checks. It does not
write a drop-in, restart services, change model limits, or transfer the port.
The port probe is advisory; the installer verifies actual binds and rolls back failures.
Environment files, unsupported host formats, and ambiguous environment encoding
are rejected for manual discovery. Wrapper/process environment and listener
ownership are verified with root privileges by the install command.

To prepare reviewable installation files without installing them:

```sh
cargo run -p kuno -- takeover-bundle --output /tmp/kuno-review
```

The destination must be new. The private bundle contains the current Kuno binary,
config, a generated control credential, the proposed Ollama drop-in, and a systemd
unit using `DynamicUser`, `StateDirectory`, and `LoadCredential`. Its runtime state
path is `/var/lib/kuno`. The unit syntax has been checked with `systemd-analyze`
against a staged filesystem. A bundle is a review artifact, not an executable
installation plan: `install` repeats discovery and creates fresh protected state.

The transaction core journals intent before stopping services, publishes files
without overwriting occupied paths, verifies readiness before completion, and
rolls back startup failures. Reopening an interrupted journal requires explicit
restore before another apply. Restore checks installed content before touching
services and refuses later administrator edits or symlink replacements. Complete
staging files can be recovered; a torn or edited staging file requires manual
review. Existing Ollama overrides are retained, including the one-model setting.
Tests simulate service failures/crashes and preserve the original active state.

The existing `OLLAMA_MAX_LOADED_MODELS=1` workaround stays effective in this
checkpoint. M2 will replace it with Kuno's managed ceiling once memory admission
and placement verification can safely replace the workaround. Three is the idle
profiling limit, not the future production residency limit.

## Privacy and remaining work

Kuno writes no inference bodies, URLs, tokens, or backend error text to logs. Its
runtime files are the empty instance lock/in-flight marker and aggregate
`usage.json` (plus an atomic-write temporary file). Installation bundles additionally
contain explicitly generated configuration, credentials, and the executable;
transaction journals contain owned-file hashes, never inference-content hashes.
The executable disables core dumping on Linux, including pipe-based collectors.
Managed units set `MemorySwapMax=0` and disable core files. The removable Ollama
drop-in disables request logging and raw stdout/stderr; Kuno supplies classified
diagnostics. Host-wide swap/hibernation privacy is not certified. Running the
uninstalled development gateway does not alter existing Ollama logging settings.

M1's remaining acceptance work includes a privileged target-host installation,
reboot/restore verification, remaining request-format usage coverage, and real
client inference through the adopted listener. GPU placement, CPU-spill recovery,
learned profiles, model-limit overrides, image execution, model downloads, and
Vashti image UI are later milestones. No GPU-placement or image support is
advertised by the current status API.

## Validation

```sh
cargo test -p kuno -p kuno-protocol
cargo clippy -p kuno -p kuno-protocol --all-targets -- -D warnings
cargo fmt -p kuno -p kuno-protocol --check
```

Tests run against disposable loopback HTTP servers and temporary state
directories. They exercise streaming, opaque binary uploads, error/redirect
fidelity, hop headers, admission, authentication, cancellation, timeouts, restart
recovery markers, locking, usage privacy/expiry, and deployment failure recovery
without loading a model or changing systemd.
