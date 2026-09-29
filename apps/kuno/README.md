# Kuno development gateway

Kuno's first implementation provides a local Ollama reverse proxy, conservative
request admission, an authenticated control listener, persistent usage counts,
and an installable systemd gateway with reversible Ollama port takeover. This
default service remains in gateway-only mode. An opt-in native Ollama GPU-worker
path is implemented and hardware-tested. An optional experimental native Qwen image
path is under validation; see [image development](../../docs/kuno-images.md).
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

`install`, `upgrade`, `configure-managed`, `configure-timeouts`, `recover`, and
`takeover-restore` all require this confirmation.
Unattended scripts must explicitly pass `--yes`; redirected input does not imply
consent. The website installer should display the interruption warning and obtain
confirmation before invoking an unattended command.

Installation includes health checks. For a separate diagnostic report later, run
`sudo /usr/local/lib/kuno/kuno doctor`.

New configurations have no inference response deadline. To remove the old
five-minute limits from an existing installation, deploy a current release with:

```sh
sudo ./kuno configure-timeouts --response-headers 0 --stream-idle 0
```

This updates the owned configuration and executable transactionally, with the
same interruption confirmation, health checks, and rollback as an upgrade.
Ordinary upgrades preserve explicit installed values. Either setting can instead
be set to a positive number of seconds (maximum 86,400). These are opt-in hard
limits, not progress-aware stall detection. A missing response alone is not proof
of a stalled GPU. Backend failure handling, finite connection establishment, and
admission queue limits remain active. The managed request-upload allowance is
separate from the inference deadline.

The installer preserves the actual IPv4/IPv6 listener, moves Ollama to a free
loopback port, installs Kuno using an unprivileged systemd `DynamicUser`, and
enables Kuno at boot. Kuno starts paused; only after both listeners and upstream
health are verified does installation activate admission. Subsequent boots wait
for upstream health and use a root-controlled activation marker. The original
loaded-model ceiling remains effective for the original Ollama service. Optional
managed workers use Kuno's separate ceiling and GPU admission checks.

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
It also resets the automatic recovery cooldown or replaces an unreadable budget
record. Normal cooldowns expire automatically.

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
Re-running the same install is idempotent. To update, run the new downloaded
binary with `sudo ./kuno upgrade`. It asks once for the whole replacement, verifies
the existing owned files, preserves configuration and usage history, and restores
the previous artifacts if the new version fails activation. A process crash still
uses the existing recoverable restore journal; keep a release binary available.

See [packaged installation instructions](packaging/README-install.txt) for paths
and content-free diagnostic commands. On 2026-09-22, installation and privileged
post-install diagnostics passed on the target Linux Mint host. Both IPv4 and IPv6
forwarding matched upstream metadata, and the service is enabled at boot. Operator
recovery and restore/reinstall also passed on 2026-09-23. The original IP binding
was verified remotely from 192.168.1.74, including streamed generation after
recovery and successful admission after cancelling a live generation. The installed fix drains cancelled responses and records boot identity;
its reboot handling has automated coverage. Live upgrades preserving managed
GPU configuration have also passed; a full managed-service reboot test remains.

## Resource observations (M2 foundation)

The new binary provides a read-only snapshot:

```sh
./dist/kuno-linux-x86_64/kuno resources
```

It uses the installed `/etc/kuno/config.json` when present, or accepts an explicit
`--config FILE`. The authenticated control API exposes the same report at
`GET /kuno/v1/resources`. Monitoring does not change usage eligibility or load,
unload, or generate with models. The report is not persisted.

PCI addresses identify physical device locations, with a NVIDIA UUID when exposed.
Intel Xe VRAM uses the same read-only DRM memory-region query proven in the local
itop project. Ambiguous zero usage is reported as unavailable, never free memory.
NVIDIA uses its installed NVML driver library directly, preferring the v2 counters
that distinguish used, reserved, and free memory. The v1 fallback explicitly marks
that used memory includes driver reservations. AMD sysfs support is implemented
but has not been validated on AMD hardware. No monitoring CLI or new runtime
package is required on the tested NVIDIA/Intel host.

Generated service units include the existing `render` group when available so an
unprivileged Kuno process can query DRM render nodes. This grants no root or
performance-monitoring capability. Missing permissions, restricted counters,
unsupported drivers, and failed probes remain explicit errors; their capacities
are not guessed. A native probe has a three-second response deadline and a single
in-flight probe limit. A timed-out probe retains its slot until it exits, so
repeated monitoring cannot accumulate blocked driver threads. Probes use a
standalone thread so runtime shutdown does not wait for that worker.

Validated on the development host with the Arc Pro B70 (`xe`) and RTX 3060
(NVML), including the Intel memory query inside a transient service using
`DynamicUser=yes`, `SupplementaryGroups=render`, and the gateway's sandbox
settings. The check does not require restarting the installed gateway or Ollama.

Ollama `/api/ps` reports are classified as GPU-resident, partially offloaded,
CPU-resident, or unknown **according to Ollama's byte counters**. They do not prove
which physical GPU holds a model or where every computation executes. No external
GPU consumer is blamed or blacklisted based on these observations. The default
installation keeps scheduling disabled and preserves the original Ollama limit.

The internal `placement` module now plans across backend-managed workers: exact
warm reuse, another available GPU, then LRU eviction of idle verified loads.
Plans account for peak workspace, headroom, external allocations, runtime/driver
and settings identity, and every GPU/worker touched by a split-model eviction.
Unknown telemetry is not capacity. Eviction always requires a fresh measurement
before loading. The opt-in `managed` configuration now executes single-device
plans through owned Ollama workers and durable GPU/worker reservations.

### Experimental managed inference

A configuration may include `managed: {"reserve_bytes":1073741824,"workers":[...]}`.
Each worker has absolute `executable`, shared `models`, private writable `home`,
a unique loopback `address`, `max_loaded_models`, `context_length`, and a `device`:
`{"backend":"cuda","uuid":"GPU-...","pci_address":"0000:04:00.0"}` or
`{"backend":"vulkan","index":1,"pci_address":"0000:2d:00.0"}`. Device indexes
are host-specific. Startup verifies Ollama's discovered PCI identity, and CUDA
UUID, before loading. A changed enumeration fails closed. This is currently a
development configuration. Installed services can apply it transactionally with
`sudo kuno configure-managed --workers /absolute/path/to/workers.json`. That file
contains the `PoolConfig` object (`reserve_bytes` and `workers`), not the full
gateway configuration. `sudo kuno configure-managed --disable` restores gateway
mode. Both commands warn and wait for confirmation, or accept an explicit
`--yes` for an already-authorized unattended invocation. They preserve the
listener, upstream, credentials, original Ollama settings, and gateway limits.

The installer assigns worker homes within Kuno's private state directory and
generates a scoped read-only model mount. Serving uses that mount instead of
requiring access through a user's private parent directories. Each worker must
verify its physical GPU during startup before the gateway opens its listeners.
Failed configuration activation rolls back the unit, configuration, and binary
together. Later upgrades retain the current managed configuration. Hardware
selection remains explicit until automatic discovery/configuration is ready.

Native `/api/chat`, `/api/generate`, `/api/embed`, and `/api/embeddings`, plus
`/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`, use the workers.
Ollama performs its existing compatibility conversion and SSE formatting. Those
compatibility endpoints use worker/model context defaults, as Ollama does; native
`options` supplied to a compatibility request do not influence admission. Verified
cloud tags use the original Ollama. Unknown models, other compatibility APIs,
and opaque operations use the original gateway path
after stopping owned workers; those fallback paths do **not** guarantee GPU-only
execution. Global admission still serializes inference across both cards.
`/api/ps` aggregates worker residency. Status reports
`managed_experimental`; the global GPU-placement guarantee remains false.

Workers set their own finite loaded-model ceiling (tested with eight), independent
of the catalog owner's original one-model limit. Preloading verifies the artifact
digest, context, and full GPU residency before sending user input. Unsupported
memory-affecting request overrides are rejected. First use isolates a workload;
subsequent requests can use a RAM-only load-footprint observation plus workspace
allowance and headroom. This is not a measured inference-peak profile or a learned
safe-combination database. Unknown/external memory is not credited or blacklisted.

The returned job retains reservations through response completion, including
draining on client disconnect. Uncertain completion stops the owned
worker process group and proves that group empty before the worker and device
scope are released, including on an explicitly enabled response timeout, a transport failure, or a
service shutdown. A group that cannot be proven empty keeps its scope
quarantined, so no device is reused while a runner may survive; reconciliation
repeats the same proof before reopening the scope. Durable uncertainty after
process restart requires `kuno recover`, which stops the service and backend
before clearing both admission and GPU-reservation markers. Systemd owns the
worker process tree. Request bodies and engine stderr are not persisted.

Failed placement now captures memory before cleanup, stops only the affected
owned worker, and waits up to three seconds for a stable window of three unloaded
readings. The lowest observed unloaded memory use is a **provisional** baseline:
it includes desktop usage and may include unknown applications. Memory still
above that baseline is classified as unexplained pressure; missing or unstable
telemetry remains inconclusive. A return to baseline requires a controlled retest
before blaming a combination, since an external allocation could have disappeared.
No outcome creates a blacklist.

For capacity/estimate failures with existing residents, a recovered baseline
allows one isolated retry after cleanup. No useful inference has been dispatched
at that point. Runtime/verification failures, unexplained pressure, and failed
isolated attempts do not trigger that retry. The pool may still try another GPU.
The authenticated `/kuno/v1/learning` endpoint exposes at most 64 failure records
per worker, including artifact digests, context sizes, co-resident workloads,
memory observations, and classified outcomes. This evidence stays in RAM and
does not yet provide persistent profiles, idle retesting, or process attribution.
Hardware tests on both CUDA/RTX 3060 and Vulkan/Arc Pro B70 verify pre-cleanup
evidence, delayed memory release, one isolated retry, and successful loading with
a corrected estimate. A separate test runtime holding GPU memory stays alive and
loaded while Kuno classifies the remaining allocation as unexplained pressure.

Hardware acceptance on this host passed for CUDA/RTX 3060 and Vulkan/Arc Pro B70:
physical-device selection, two GPU-resident models, actual text and embeddings,
eviction followed by memory remeasurement, abandoned-job recovery, and full
native gateway routing with warm reuse and unload. A two-card gateway run also
verified that a cold workload uses the empty alternate GPU while preserving the
first model for warm reuse. Both GPU paths also passed generation, embedding,
multiple-residency, and unload checks inside a restricted `DynamicUser` service.
On this host, `/media/jmicko` grants traversal to the original `ollama` user but
not the dynamic account. The acceptance service exposes only the shared model
directory through `BindReadOnlyPaths` at a service-private path. Parent directory
permissions were not widened. Installed managed deployment provides the same scoped
access; joining the user's personal group is not required. Models were
`lfm2.5-thinking:1.2b-bf16` at context 4096 and `embeddinggemma:latest` at 2048.
A two-card development gateway additionally passed OpenAI-style chat, completions,
embeddings, SSE completion, and draining an abandoned compatibility stream.
The installed service passed generation and embedding across both GPUs and
`ornith-1.5:35b` generation at context 128000 with full GPU residency on the Arc.
An intentionally incorrect Vulkan index failed physical-device verification and
restored the previous binary, unit, and configuration byte-for-byte. A parser
regression test checks the model mount with a real user systemd manager without
starting the temporary unit (`cargo test -p kuno --lib systemd_parses_model_mount
-- --ignored`). The source and destination must be quoted separately so systemd
does not interpret their separator as part of the source path.
Tests emit no inference content. Set `KUNO_TEST_SECOND_WORKER_CONFIG` to a second
worker JSON to exercise both cards in the gateway test. To repeat with an idle
explicitly assigned GPU:

```sh
KUNO_TEST_WORKER_CONFIG=/absolute/path/to/worker.json \
  cargo test -p kuno --test worker_live -- --ignored --test-threads=1
```

Remaining before default activation: compatibility-path enforcement, durable
versioned profiles, inference peak measurement, external-pressure classification,
automatic system discovery, and independent-GPU request
concurrency. Idle profiling remains a later milestone; experimental image execution
is described in [image development](../../docs/kuno-images.md).

[Bonsai 2 and native Rust engine development](../../docs/kuno-bonsai.md) records
the additional engine track and its Ollama model-list/API compatibility contract.

Driver/API references:
[NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html),
[Linux Xe UAPI](https://github.com/torvalds/linux/blob/master/include/uapi/drm/xe_drm.h),
[Ollama running models](https://docs.ollama.com/api/ps).

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
- Bounded queues and connection establishment, with content-free errors.
  Response-header and stream-idle deadlines are disabled by default (`0`), so
  long prompt processing, thinking, and buffered responses are not cut off.
  Administrators may explicitly enable either deadline. Stream idle limits apply
  when polling upstream; downstream backpressure can retain a lease longer.
  HTTP protocol upgrades/CONNECT are explicitly unsupported.
- A separate bearer-authenticated control surface with status and pause/resume.
  The compatibility listener does not expose Kuno control handlers.
- A fixed, empty in-flight marker written durably before dispatch. Normal upstream
  completion clears it. On client disconnect (including before response headers),
  Kuno retains the reservation and discards the remaining response in memory.
  The drain has no default time limit and never replays a request. If configured,
  the stream idle limit resets on each received frame; it is not a total-duration
  limit. Transport failures or explicit timeouts still require recovery when
  backend completion cannot be established.
- A saved Linux boot identity clears an old empty marker after a confirmed host
  reboot, when the local upstream cannot retain prior work. Restarting only Kuno
  preserves uncertainty. Legacy markers without a boot record and corrupted
  state remain subject to explicit recovery.
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
| GET | `/kuno/v1/resources` | Native GPU memory observations and Ollama-reported residency |
| GET | `/kuno/v1/learning` | Bounded placement-failure evidence and provisional unloaded baselines |
| GET | `/kuno/v1/usage` | Aggregate model calls, last-use times, and persistence health |
| POST | `/kuno/v1/admission/pause` | Reject new managed work; let active work finish |
| POST | `/kuno/v1/admission/resume` | Resume admission unless upstream recovery is required |

Pausing is not an immediate GPU reservation. Wait until `active` is false, and
ensure `recovery_required` is false, before treating the upstream as quiescent.
Loaded models may still occupy GPU memory. The optional native image worker has
a separate authenticated generation route and reserves admission through cleanup;
see [image API and lifecycle](../../docs/kuno-images.md#api-and-lifecycle).

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

An installed gateway checks itself with `kuno-recovery.timer` roughly every 30
seconds. It acts on latched upstream uncertainty while idle and unpaused,
repeated control-health failure, or an interrupted recovery attempt. A manual
`systemctl stop` is never undone. Each attempt is charged before services are touched: at most
five starts per renewable 30-minute window, with failure backoff from 30 seconds
to 30 minutes. Cooldowns expire automatically; successful recoveries never
permanently exhaust a boot-wide allowance. Three failed control-health checks
for the same running PID also trigger recovery, while an already pending attempt
continues even if control is unreachable. The records under
`/var/lib/kuno-setup` hold only counters, deadlines, process/boot identity, and
allowlisted failure categories. `sudo kuno doctor` reports the budget and pending
state, and `kuno recover` resets cooldowns on demand.

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

The existing `OLLAMA_MAX_LOADED_MODELS=1` workaround stays effective for the
original catalog service. Managed workers override it independently, allowing
multiple models when admission and residency checks pass. Three is the planned
idle profiling limit, not the production residency limit.

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

Kuno emits `kuno_lifecycle` JSON events to the service journal for managed-worker
startup, profiling/reuse, preload, response headers/body, cancellation draining,
forced shutdown, and recovery-required transitions. The schema allows only fixed
stage/outcome names, monotonic elapsed milliseconds, worker PID, and validated PCI
address. It excludes model names/digests, prompts, images, responses, request IDs,
client addresses, URLs, headers, and backend error text. Metadata polling and
individual generated tokens do not emit these events. Observation itself does
not stop workers or change scheduling.

To correlate a failure with Intel driver events on an installed Linux host:

```sh
sudo journalctl -b -f -o short-monotonic \
  _SYSTEMD_UNIT=kuno.service + _SYSTEMD_UNIT=kuno-recovery.service + _TRANSPORT=kernel \
  --grep='kuno_lifecycle|automatic_recovery|xe.*(reset|wedg|timeout|fault|GuC|guc)'
```

`worker_load_headers` and `worker_load_body` measure the preload request;
`upstream_headers` starts after placement/preloading. A `worker_signal` with
`sigkill` is emitted immediately before the existing signal syscall. Successful
`worker_group_exit` and `worker_reap` events establish cleanup completion;
`worker_drop_signal` alone does not establish completion. Journal timestamps
allow comparison with kernel resets without retaining inference content.

Privileged installation and real client inference through the adopted listener
have been tested on this host. Remaining work includes full reboot/restore
acceptance and request-format usage coverage. Experimental managed workers provide
placement verification, CPU-spill rejection, and model-limit overrides as described
above. Idle profiling, a Vashti generation form/chat tool, and full image workflow
fidelity validation remain outstanding. Curated Qwen downloads, an experimental
native image worker, and Vashti metadata inspection are available as described
in [image development](../../docs/kuno-images.md). The status API does not claim
a global GPU-placement guarantee.

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
