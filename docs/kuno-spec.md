# Kuno inference host specification

Status: M1 gateway development release installed on the target Linux Mint host
(2026-09-22). Privileged installation, post-install diagnostics, IPv4/IPv6
forwarding, and startup enablement passed. Recovery and restore/reinstall passed on 2026-09-23, followed by remote streamed
generation. Experimental managed GPU workers were installed and validated on
2026-09-23: two-card text/embedding routing, a 35B model at context 128000 fully
resident on the Arc B70, and rollback after incorrect physical-device selection.
LAN model listing, OpenAI-compatible generation, and unload passed from Clark's
machine. A subsequent binary upgrade preserved the managed configuration and
both verified workers, with healthy post-upgrade diagnostics.
Reboot reconciliation has automated coverage; a full host reboot of the managed
configuration remains pending.
Planning baseline: 2026-09-20.

See [Kuno development instructions](../apps/kuno/README.md) for the implemented
checkpoint and remaining acceptance work. Supported inference routes now verify
GPU placement through optional managed workers. Experimental native Qwen image
execution is installed and has passed LLM/image handoff and client cancellation
on the RTX 3060. The B70 requires a fresh-boot retest after Vulkan/xe failures;
full reference fidelity and the generation UI/tool remain outstanding. See the
[image checkpoint](kuno-images.md) for evidence and limits. Opaque fallback routes
do not guarantee GPU-only execution.

This specification defines Kuno and its integration with Vashti. It records the
decisions from the planning discussion and the evidence required before claiming
that GPU placement, privacy, or image compatibility works. Runtime performance
and exact image fidelity remain implementation validation tasks.

## 1. Purpose and scope

Kuno runs on each inference machine. It coordinates local inference runtimes so
LLMs, embedding models, and image models can share GPUs without manual unloading
or unintended CPU placement. Vashti may run on a different machine.

The first useful deliverable is an Ollama-compatible gateway with optional port
takeover. Existing Ollama clients should work through it without changing their
URLs. Image generation follows, beginning with the supplied Qwen-Image-2512
workflow and a small, mobile-friendly set of controls in Vashti.

The initial implementation targets Linux with systemd. Hardware validation will
use Linux Mint with an Intel Arc Pro B70 and an NVIDIA RTX 3060. Host/service
adapters should leave room for other operating systems and installation methods.
Cross-platform support is not an MVP acceptance claim.

Kuno targets one installed Rust executable, with as few runtime dependencies as
practical. A linked native image library and host GPU drivers are compatible
with that goal. Dependencies requiring separate installation must be detected
and explained, or installed through an explicit supported setup path.

MVP scope excludes a node editor, arbitrary Comfy workflow execution, a general
plugin platform, and automatic GPU resets. A native Rust inference engine is an
approved follow-on track, beginning with Bonsai 2; it is not a prerequisite for
the image MVP. See [Bonsai and native-engine development](kuno-bonsai.md).

## 2. Responsibilities and workspace

The repository already is a Cargo workspace. Add these members during
implementation, without restructuring the existing applications:

```text
apps/kuno                 Inference gateway, host management, scheduler, CLI
crates/kuno-protocol       Shared versioned types for capabilities and jobs
apps/vashti               Existing application and future image experience
```

Keep backend adapters inside Kuno initially. Split additional crates only when
there is a concrete boundary or reuse requirement.

| Component | Responsibilities |
| --- | --- |
| Vashti | Users, permissions, conversations, image controls, durable standard-chat jobs and artifacts, private-chat storage policy |
| Kuno | Admission, placement, residency, runtime supervision, model collections, downloads, ephemeral execution, resource profiles |
| Ollama adapter | Compatible inference and model APIs, loading/unloading, residency observations, assigned device visibility |
| Kuno LLM adapter | Kuno-owned model catalog entries, compatible LLM APIs, native Rust engine execution; Prism runtime as a Bonsai reference and optional bridge |
| Image adapter | Fixed workflow execution, component memory management, progress, cancellation, in-memory image output |

All Vashti inference on a Kuno-managed host must use its gateway: normal and
private chat, embeddings, retrieval/indexing, titles, and tool follow-up turns.
A separate preflight call followed by a direct Ollama request is insufficient;
Kuno must own admission through completion of the upstream operation.

## 3. Content privacy and persistence

Kuno and its managed workers must not persist inference content. This applies to
normal execution, background profiling, failure recovery, and diagnostics.

Forbidden persistent inference data includes prompts, negative prompts, messages, tool
arguments/results, request bodies, generated text, input/output images, previews,
latents, prompt embeddings, request seeds, raw imported workflows, and hashes of
user content. Logs, temporary files, database journals, crash artifacts, and
backend debugging must not become alternate content stores.

Allowed persistent data is limited to:

- Downloaded model artifacts and non-content runtime/kernel caches.
- Configuration, protected connection credentials, and takeover/rollback state.
- Curated collection and built-in workflow definitions.
- Aggregate operational profiles, model-level last-real-use times/counts, and
  bounded background-test coverage records.

Operational profiles identify hardware, model artifacts, workload size classes,
and runtime versions. They do not identify users or conversations and do not
form a request history. Persist allowlisted numeric measurements and classified
error codes; arbitrary backend stderr or exception text may contain content.

Explicit model-management requests may intentionally persist model definitions,
including their templates, system prompts, or example messages, as requested artifacts.
This preserves model-creation compatibility. It does not authorize saving live
inference messages into artifacts or retaining model-management request logs.
The image adapter has no facility for saving user prompts or image history.

Inputs, request-derived caches, and results may exist transiently in RAM/VRAM for
execution. Image request buffers and derived caches are released after delivery,
cancellation, failure, or a bounded delivery timeout. Ollama's transient prompt
cache is subject to its managed runtime lifecycle and must never be saved to disk.
Request content must not be passed in process command-line arguments.

Use the native image library's in-memory interface; filesystem image output is
not part of the adapter contract. Audit memory-to-disk paths, including swap and
core dumps, in the supported service configuration. A temporary filesystem alone
does not establish this guarantee. Unsupported protection or incomplete telemetry
must be reported honestly, rather than advertised as verified privacy.

Managed Ollama must disable request-body logging. Its
[`OLLAMA_DEBUG_LOG_REQUESTS` feature](https://github.com/ollama/ollama/blob/v0.34.2/server/inference_request_log.go)
can otherwise write replayable requests to temporary storage.

### Job ownership and restart behavior

Kuno's queue, correlation IDs, deduplication state, and result buffers are
ephemeral and bounded. Kuno does not spool requests or completed results to disk.

Vashti owns durable job state for standard chats. Private-local jobs retain their
existing client-owned persistence semantics: neither Vashti's server nor Kuno
may persist private content. Mobile reconnection is handled through the relevant
Vashti job owner while Kuno execution remains live.

Persist a content-free Linux boot identity alongside the in-flight marker. A
confirmed host reboot may clear an empty marker from an earlier boot because the
local runtime cannot retain that work. A Kuno-only restart, a legacy marker without
a boot identity, or invalid state cannot establish that guarantee.

A Kuno restart loses in-flight work. The caller detects interruption and decides
whether to resubmit. Do not promise exactly-once execution across restarts, or
silently replay an operation after it may have executed or emitted output.

## 4. Ollama compatibility and optional port takeover

Kuno offers two setup modes:

- A separately configured gateway in front of an explicitly selected Ollama
  instance, useful for development and manual installations.
- Optional takeover of the existing Ollama listener, preserving client URLs.

For example, Kuno can inherit `0.0.0.0:11434`, with Ollama relocated to
`127.0.0.1:11435`. The actual address and private port are discovered and stored;
neither port is assumed available.

### Takeover transaction

1. Inspect the actual listener and owner, effective `OLLAMA_HOST`, service
   overrides, executable, model store, service identity, and GPU settings.
   Ollama has several installation forms, not one universal config file.
2. Prepare and test Kuno on a nonconflicting endpoint before changing the live
   service. Record the original configuration and service state atomically.
3. Warn that inference will be interrupted during cutover and wait for explicit
   confirmation before changing services. Default to cancellation; unattended
   setup requires `--yes`. Apply the same rule to recovery and restore. Do not require users
   to close clients or reject setup because connections remain open. Stop the
   managed service and its runners; clients may reconnect after health checks
   complete. Existing sockets cannot be transferred, and `/api/ps` does not
   establish whether requests are active.
4. Create a Kuno-owned systemd override that moves Ollama to a private loopback
   port. Preserve unrelated user settings and overrides.
5. Restart and health-check Ollama, verifying the actual bind. A port-availability
   probe has a race; handle bind failure with bounded retries or rollback.
6. Activate Kuno at the original address, including the intended IPv4/IPv6
   behavior, and verify requests through the original endpoint.
7. Upgrade uses a single confirmation for the complete replacement. Verify owned
   files before interruption and restore the previous artifacts on ordinary new
   version activation failure. Interrupted processes retain a recovery journal.
8. On migration failure, undo Kuno's changes and restore the recorded original
   configuration. Provide an idempotent uninstall/restore operation that does
   not overwrite later unrelated user edits.

Only the setup operation needs service-configuration privileges. Kuno's normal
inference service runs unprivileged. Manual processes and containers require
explicit adapters or configuration; an unrelated port owner is never killed.
[Ollama documents systemd overrides](https://docs.ollama.com/linux#customizing).

Rollback is an installation recovery mechanism. A runtime failure must not
silently reopen a direct Ollama route that bypasses scheduling.

### Proxy contract

Use a general HTTP reverse proxy, preserving method, path/query, unknown fields,
status, end-to-end headers, error bodies, multipart data, and incremental
NDJSON/SSE delivery. Handle hop-by-hop headers correctly. Forward only to
configured upstreams, do not follow upstream redirects automatically, and
preserve the deployment's origin/authentication behavior.

Use bounded in-memory buffers and backpressure, with no payload spooling. Client
disconnects request backend cancellation once its completion can be verified.
In the M1 generic HTTP adapter, an abandoned response is instead drained and
discarded, including disconnection before headers. There is no default
inference response-header or stream-idle deadline; a long prefill or silent
thinking phase is not a failure. Optional administrator limits use `0` for
disabled. An enabled stream-idle limit also applies to draining and resets on
every received frame, rather than limiting total generation time.
A successful drain releases admission; a failed or explicitly timed-out drain
retains uncertainty. This avoids treating an ordinary client disconnect as permanent
backend failure without assuming that a closed TCP connection stops GPU work.
Reservations remain held until the upstream has actually stopped, not merely
until headers arrive or a client leaves.
Do not automatically retry a request that may already have executed.

Classification wraps the proxy; it does not reimplement every Ollama response:

| Request class | Scheduling behavior |
| --- | --- |
| Known metadata reads | Serve/proxy without a GPU reservation |
| Chat, generation, embeddings, compatibility inference | Reserve appropriate resources before dispatch |
| Explicit preload/unload or residency changes | Coordinate with running work and update residency state |
| Model create/pull/copy/delete and artifact changes | Coordinate catalog mutations and invalidate changed identities |
| Unknown endpoint | Proxy through conservative exclusive upstream admission; reconcile residency afterward |

Audit the pinned Ollama version's native and compatibility route inventory.
For example, v0.34.2 includes Responses, compaction, audio transcription, and
Anthropic-compatible messages in addition to familiar chat and embedding APIs.
See the [registered routes](https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go).
Opaque future endpoints that launch asynchronous work need explicit support
before claiming full scheduling coverage.

Client keep-alive preferences are residency hints subject to Kuno's resource
policy. Explicit unloading must respect active requests. Requested model,
context, quality, and precision must not be silently reduced to make a job fit.

Start with one upstream. Multiple device-assigned workers later require one
catalog owner, correct aggregation of running-model information, and a defined
default upstream for opaque requests. Do not broadcast mutations to all workers.

Kuno's privileged control API is separate from the compatibility surface and
requires authentication. Preserving an existing Ollama client connection must
not grant access to service installation, runtime updates, or host configuration.

Port takeover covers clients using that endpoint. Separate servers, direct
native inference, games, and local clients deliberately using a private upstream
remain external consumers whose allocations must be observed.

## 5. Admission, placement, and residency

Identify physical GPUs stably and map each runtime's device names to them. An
allocation spread across several GPUs reserves all affected devices. Managed
worker device visibility provides a concrete placement boundary; adoption of an
existing instance must report any weaker control or observability.

Keep model residency separate from active computation. Several small models may
stay warm on a card while active inference is serialized. Begin conservatively
with one active inference operation per GPU reservation; independent devices
can run concurrently. Broader concurrency requires its own measurements.

### Ollama loaded-model limit

Kuno takes ownership of `OLLAMA_MAX_LOADED_MODELS` for managed Ollama instances
once resource admission and residency verification are operational. The existing
value of `1` on the development host is a workaround for CPU spill, not a user
preference to preserve during managed scheduling. Kuno must override that value
when present so its scheduler can retain multiple models that fit in GPU memory.

Discover and record the original effective value and its configuration source,
including whether the setting was absent. Apply the managed value through a
Kuno-owned service override or worker environment, verify the effective value
after restarting, and restore the previous configuration on rollback or removal
without overwriting unrelated later edits. Detect conflicting overrides rather
than reporting successful adoption while the old limit remains effective.

Use an explicit finite per-instance ceiling sufficient for the resident sets
Kuno admits. The ceiling is a backend capacity setting; actual admission depends
on measured memory, workload requirements, and placement verification. The
three-model idle-testing bound is not a production residency limit: four or more
small models may remain loaded when resources permit. Keep this setting separate
from `OLLAMA_NUM_PARALLEL` and context length; raising the loaded-model ceiling
does not authorize increasing either of those memory costs.

M1 discovers and records the existing limit but preserves it while operating as
a gateway without the complete resource scheduler. M2 activates the managed
override alongside the protections that replace the one-model workaround.

### Admission sequence

Admission follows this order:

1. Reconcile managed worker state with actual device availability.
2. Identify the effective workload: model/version, precision, context and
   parallelism, or image workflow/resolution/batch.
3. Prefer a compatible warm model, then another suitable GPU, then eviction of
   enough idle managed models. Wait behind active work when necessary.
4. Reserve resources before loading. Serialize conflicting transitions.
5. Load and verify placement before forwarding useful inference where supported.
6. Hold reservations through upstream completion/cancellation and update profiles.

Model files alone do not establish memory cost. Estimates include resident
weights, KV cache, temporary workspace, backend overhead, external allocations,
and a conservative reserve. Image peaks are evaluated across encoding,
denoising, and VAE phases. Avoid double-counting memory already accounted for
when comparing a proposed resident set with current free memory.

Full GPU model placement is the default. Explicit CPU/offload modes, if exposed,
are separately selected policies. Slow output is not sufficient evidence of CPU
spill. Combine managed device assignments, runtime diagnostics, residency, and
performance observations. Ollama's [`/api/ps`](https://docs.ollama.com/api/ps)
reports model/VRAM sizes; it does not prove where every operation executes.

Verify the strength of each adapter's GPU-only enforcement during implementation.
An API limitation must be surfaced, addressed in the adapter/runtime, or exclude
that mode from a strict placement claim. An unexpected spill is not a successful
profile observation. Recovery explicitly unloads/reloads the wrongly placed
model; do not assume freeing space migrates existing CPU allocations.

Queues and retries are bounded. Use fair admission and reload-cost awareness to
avoid starvation and repeated model swapping. Existing clients can have their
own timeouts; Kuno must preserve cancellation and meaningful resource errors.
Any future automatic stall detector must consider backend progress, GPU/CPU
activity, and loading/I/O evidence. Missing telemetry or a quiet GPU alone must
not authorize stopping a request. The optional response deadlines are explicit
operator policy, not a claim that elapsed time proves a backend is stuck.
On the observed Ollama v0.34.2/llama.cpp b10969 runtime, the owned runner's
`/slots` endpoint exposes increasing prompt-processing counters during long
prefill. Any future adapter must validate listener ownership, bound the response,
and extract only allowlisted numeric/boolean fields in memory: the complete slot
response can contain prompt data. Compare counters only within the same runner
and slot/task; resets and missing samples mean unknown. A flat counter during a
single expensive batch is not conclusive failure. Until such a detector is
implemented and validated, response deadlines remain disabled by default.

### LLM tools and image handoff

The LLM's GPU reservation covers one inference request, not the complete tool
loop. Otherwise an image tool could deadlock waiting for its own LLM reservation.

1. Receive the complete assistant response and terminal completion indication.
   A partial tool-call chunk or truncated stream is insufficient.
2. Preserve structured assistant content, thinking where applicable, and tool
   calls in Vashti's permitted storage/memory.
3. Release that inference reservation and admit the image job.
4. Drain conflicting work, unload conflicting models, and verify actual memory
   recovery before loading the image worker.
5. Return the image and effective recipe/provenance to Vashti in memory.
6. Release image resources and admit the next LLM request with the tool result.

Ollama's tool cycle accepts results in a new request. Unloading preserves the
conversation held by Vashti, but loses runtime caches and adds reload/prompt
processing cost. See [tool calling](https://docs.ollama.com/capabilities/tool-calling).
An unload acknowledgement or disappearance from the model list alone does not
establish that device memory has recovered.

## 6. Learning from measurements

Persist a small, bounded, resettable resource-profile database. Prefer explainable
estimates with measured corrections over exhaustive configuration search.

Profile identity includes device/topology, driver and runtime versions, backend,
model/component digests, quantization, and relevant workload bounds. Distinguish
LLM context/parallelism and image resolution/batch/phase behavior. Aliases that
resolve to the same effective workload should share observations where valid.

Record sample counts/confidence, observed residency and peak memory, timing
aggregates, actual placement, and classified outcomes. Maintain individual-model
baselines plus selected pair/triple observations. Estimate untested combinations
from component costs with reserve; concentrate experiments on useful uncertainty.

Useful statuses are predicted, measured successful, suspect, known unsafe for a
specific configuration, unsupported, and inconclusive due to external pressure
or inadequate telemetry. A successful load is distinct from an exercised workload
and from concurrent peak execution.

Use observed peaks and uncertainty margins, rather than a mean that can conceal
large allocations. Downgrade or invalidate affected observations when models,
drivers, runtimes, topology, precision, or workload definitions change. Historical
success informs admission; current resources must still be checked.

## 7. Failure diagnosis and recovery

### Installed-service automatic recovery

Installation and upgrade enable `kuno-recovery.timer`. Roughly every 30 seconds
after its previous check finishes, its root-owned oneshot inspects the gateway
state and, when it is reachable, its authenticated control status. Latched
upstream uncertainty with no active request and no intentional admission pause
triggers recovery, as does an earlier interrupted attempt that already stopped
the gateway. Healthy inference, manually stopped gateways, and starts still in
progress are left alone. The public gateway still runs as an unprivileged service.

Recovery holds the installation transaction lock, verifies installed artifacts,
charges one attempt against a bounded budget, and records an empty root-owned
pending marker before stopping services. It stops Kuno and its workers, then the
private catalog Ollama, and verifies their cgroups are empty before clearing
uncertainty markers. It restarts and verifies both services. A failed or
interrupted attempt retains the pending marker, so the independent timer retries
even with Kuno down, and is charged again. Attempts are serialized and service
transitions are bounded; failed requests are never replayed. Requests can receive
temporary 503s or connection failures during recovery and should retry.

Automatic attempts are rate-limited to five starts per renewable 30-minute
window. Failed attempts back off from 30 seconds to 30 minutes. Success clears
failure counters but retains the window budget; the boot counter is diagnostic
only. Expired windows renew automatically, including migration of legacy
`manual_required` records. No finite number of successful recoveries permanently
disables supervision. State holds only bounded counters, deadlines, host boot
identity, and allowlisted failure categories, never request content.

Three consecutive failed control-health checks for the same running service PID
also justify recovery. A healthy response clears that evidence; a new process
starts the count over. Pending recovery bypasses the control check, so an
unreachable replacement gateway cannot prevent a scheduled retry. Cleanup still
requires the same process-group proof before any uncertainty is cleared.

A managed request that cannot prove upstream completion stops its owned worker
process group and proves that group empty before releasing the worker and device
scope. A group that cannot be proven empty keeps its scope quarantined, so no
device is reused while a runner may survive; the installed supervisor then stops
the service instead of clearing uncertainty. Ordinary drained completion, by
contrast, releases admission without any restart.

Restore disables the timer and clears the budget. `kuno recover --yes` remains
available for immediate manual recovery, including when a budget record cannot
be interpreted;
automatic recovery requires no interactive confirmation after the administrator
has installed this supervisor. Modified installation artifacts or unproven
process cleanup still prevent unsafe marker clearing. This mechanism does not
reset GPU drivers or control unrelated GPU applications.

Local acceptance, 2026-09-25: the native-image build was upgraded on the dual-GPU
Linux host. An injected empty uncertainty marker survived a Kuno-only restart,
then the timer recovered admission in 20 seconds. An emulated interrupted
recovery (gateway stopped with a pending marker) recovered in 42 seconds.
A subsequent real `lfm2.5-thinking:1.2b-bf16` generation completed, and admission
was idle, unpaused, and free of recovery requirements. The Kuno test suite and
strict Clippy checks passed; hardware-specific ignored tests remain separate.

The episode budget, the proof-before-release rule, and the recovery-decision
table are covered by automated tests. Live acceptance of the bounded supervisor
on the dual-GPU host is still pending: the earlier observation there was a
supervisor without a budget restarting a failing gateway about every twenty
minutes, which is the behavior the bound now prevents.


A failure begins unclassified. Do not blacklist a model combination just because
it failed while a game or another application occupied the GPU.

1. Capture resource evidence before cleanup: managed allocations, available
   memory, external graphics/compute activity where observable, and backend state.
2. Recover pending user work conservatively, using fewer idle models or another
   device. Bound retries and do not silently replay after output delivery.
3. When active managed work on the affected device has drained, unload its managed
   models and wait for actual memory recovery. For workers spread across GPUs,
   include their affected device group. Healthy independent GPUs remain undisturbed.
4. Compare the stable result with a calibrated unloaded baseline that includes
   display/desktop usage and expected runtime overhead.
5. Treat unexpected allocations, unstable readings, missing attribution, or a
   driver fault as an inconclusive/environmental result. Unknown memory is not
   proof of a particular external process.
6. Retest the suspect combination during a clean idle period, monitoring conditions
   throughout. A normal baseline after failure is insufficient by itself: an
   external consumer may have released memory in the meantime.
7. Only reproducible controlled failure establishes an unsafe configuration.
   Prefer correcting estimated memory or reserve when that explains the failure;
   retain a specific exclusion for persistent configuration-specific problems.

Never unload external applications, kill unrelated processes, or reset the GPU
as automatic recovery. Distinguish model-capacity errors from runtime defects.
Telemetry quality is part of confidence: some
[Vulkan configurations provide approximate accounting](https://docs.ollama.com/gpu#vulkan-gpu-support).

## 8. Idle profiling

The agreed defaults are:

| Setting | Default |
| --- | --- |
| Idle delay | 30 minutes after real inference finishes, with no real work queued/running |
| Eligible workloads | Used by a real inference request during the previous 7 days |
| Maximum workloads in one background test | 3 |
| Simultaneous background experiments | 1, which may involve several GPUs |
| Inputs and outputs | Synthetic inputs in memory; discard outputs |

Metadata polling does not reset idle time. Real requests from every proxied
client do. Profiling and Kuno's internal retries/probes do not increment demand
or refresh seven-day eligibility. Keep aggregate real-use counts and last-use
times, not request histories. Prior usage is unknown at first installation;
eligibility starts from observed traffic.

An expired workload leaves the testing pool without deleting its files or useful
measurements. A new real request makes it eligible again. A Qwen workflow and
its transformer/encoder/VAE dependency bundle count as one workload; all component
memory still counts. The three-workload cap limits idle testing, not production
residency when more small models safely fit.

For distinct unordered model sets, the candidate count is
`n + n*(n-1)/2 + n*(n-1)*(n-2)/6`:

| Eligible models | Singles | Pairs | Triples | Total |
| --- | ---: | ---: | ---: | ---: |
| 33 | 33 | 528 | 5,456 | 6,017 |
| 10 | 10 | 45 | 120 | 175 |

GPU assignments and workload presets add variants. Literal exhaustive coverage
is not a goal. Prioritize single-model baselines, actual/common use cases, recent
demand, and uncertain predictions. Include occasional lower-demand useful cases
so popularity does not permanently prevent exploration. Generate candidates
lazily and stop when there is no useful untested or invalidated case.

Exclude proven incompatible cases and configurations exceeding defensible hard
memory bounds. Estimates alone should not permanently exclude uncertain cases.
Use bounded supported context, concurrency, and image presets; a short prompt
does not certify a long context, and a small image does not certify a larger one.
Track what was actually exercised, including incomplete synthetic generations.

Before and during tests, check relevant GPU activity, unexplained residency,
system-memory headroom, and device health. Low utilization alone does not prove
availability. Apply the same reservations and failure-classification policy as
production. Keep profiling disableable and bounded by operational limits.

Foreground arrival immediately stops new tests and requests cancellation of the
current one. After a bounded grace period, only an owned profiling worker may
be terminated. Verify memory release before conflicting foreground execution.
Measure cancellation/recovery latency; a backend that cannot yield acceptably
must not profile in the background. Interrupted tests return to pending and are
neither successful nor failed compatibility evidence.

Do not leave experimental allocations occupying devices indefinitely. Restore
useful warm residency when appropriate, without delaying waiting foreground work.

## 9. First image workflow and compatibility

The supplied `Qwen-Image-2512_01366_.png` at the repository root is the initial
reference. Its execution metadata gives the following recipe:

| Field | Reference value |
| --- | --- |
| Diffusion model | `qwen_image_2512_fp8_e4m3fn.safetensors` |
| Text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | `qwen_image_vae.safetensors` |
| Sampler / scheduler | Euler / simple |
| Steps / CFG | 100 / 5 |
| Flow shift | 3.1 (serialized as `3.1000000000000005`) |
| Denoise | 1 |
| Width / height / batch | 832 / 1248 / 1 |
| Reference seed | `684489560344628` |
| Prompts | Positive and negative text embedded in the supplied PNG |
| Active LoRA | None |

The reference seed is specification fixture information, not permission for Kuno
to retain request seeds. Preserve seed values losslessly through the protocol.

The active chain is diffusion loading, flow-sampling configuration, positive and
negative text encoding, empty latent creation, sampling, and VAE decoding. The
Lightning LoRA mentioned in a note is not an active dependency.

Vashti imports PNG generation metadata and maps recognized execution graphs to a
versioned built-in recipe, initially `qwen-image-2512-text-to-image/v1`. Trace
backward from the selected output; do not infer execution from disconnected
notes or canvas layout. Handle indirect prompt inputs. Ambiguous outputs and
unsupported operations must be explained, not silently dropped.

Imported metadata is data, not executable code or permission to fetch arbitrary
URLs. Resolve dependencies against curated manifests or explicit local mappings.
Bound parsing/decompression and graph traversal. Missing metadata or unknown
nodes should leave the original image usable without claiming reproducibility.

Vashti may preserve original metadata and an effective recipe under the selected
storage mode. Generated images should support reusing settings and metadata
export through Vashti. Kuno returns effective settings/provenance transiently.

Promise faithful settings and recorded provenance. Pixel identity across GPUs,
engines, precision, and runtime versions is a separate measured claim. Filenames
in the PNG do not establish exact original file hashes or runtime configuration.

### Engine boundary and first implementation

Use a pinned `stable-diffusion.cpp` implementation behind a small image adapter
covering capabilities, preparation, generation/progress, cancellation, and release.
Kuno owns the workflow schema and scheduling, keeping a future native Qwen engine
or another backend possible without changing Vashti's product contract.

A linked library called from a private worker mode of the same Rust executable
is the preferred packaging experiment. Worker exit provides a useful cleanup
boundary. Vulkan is the initial cross-vendor path to test; choose other backends
from measured compatibility/performance, not assumed vendor superiority.

Current source supports relevant split files and sampling controls, but execution
of this exact workflow on the B70 remains unverified. See
[native Qwen support](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image.md).
Keep FP8 residency intact: the upstream-GGML build option can expand FP8 weights
to FP16, so pin and verify the chosen build configuration.
[Build documentation](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)

If another implementation becomes necessary, preserve the exact sampling/noise,
scaled-weight, latent, and CFG semantics. For example, a Diffusers compatibility
adapter needs fixed flow shifting and Comfy-compatible guidance instead of simply
accepting that pipeline's defaults. A replacement engine is a response to a
measured limitation; individual bugs may be addressed in the existing backend.

## 10. Collections, installation, and updates

Admin-selected collections declare the model dependency closure: immutable
source revisions, paths, hashes, byte sizes, roles, architecture, precision/scaling
format, license/access requirements, compatible workflow/runtime versions, and
estimated resource requirements. Future LoRAs also declare compatible base
models, application order/strength, and associated workflow settings.

The first collection contains the reference Qwen model, encoder, and VAE. Other
Qwen and accelerated/Lightning recipes follow separate validation; they must not
silently substitute for the imported reference recipe.

Download required assets automatically after collection selection. Support space
checks, progress, cancellation, resume, checksum verification, atomic activation,
deduplication, and clear handling of gated/missing assets. Downloads do not hold
GPU reservations. Shared model storage has one mutation/pruning owner. Model
removal must account for active use and invalidate affected profiles.

Support ordinary boot-time service startup. If the configured model volume is
unavailable, report that condition and wait/fail clearly rather than recreating
the store at the missing mount point. Keep binaries, models, profiles, and
Vashti-owned content separated so updates cannot erase workflows or images.

Provide a diagnostic command reporting hardware/backend versions, permissions,
service/listener ownership, model availability, placement evidence, profiling
status, and classified failures. It must produce useful human-readable and
machine-readable output without request content or credentials.

Pin validated runtime builds and support rollback between jobs. Kuno-owned
runtime downloads may eventually avoid repeated sudo after setup. Updating a
system-installed Ollama or drivers remains a distinct privileged operation;
never store a sudo password or automatically upgrade drivers during inference.

## 11. Vashti experience and integration

The detailed [Vashti Images design](vashti-images-spec.md) defines the chat tools,
dedicated workspace, editing branches, gallery stacks, and job/storage behavior.
It is a authorized integration specification before app implementation.

Provide a direct image-generation form and a `generate_image` tool. A user should
be able to generate an image without an LLM. Start with prompt, size/aspect ratio,
and built-in recipe selection; expose supported advanced controls such as steps,
CFG, seed, and negative prompt. Future examples can become additional built-in
recipes. A node editor is outside scope.

Use existing global tool enablement, permission tags, per-chat preferences,
standard/private execution, and expandable activity patterns. Image artifacts
belong to the originating message/revision or direct generation job. Tool results
use artifact references and structured status, not image bytes in a text prompt.

Expose understandable progress for waiting, freeing memory, loading, encoding,
sampling, decoding, and delivery. Cancellation must reach the owning worker.
Verify touch controls and narrow layouts. Reconnection must not accidentally
submit a duplicate job or weaken private-chat storage rules.

Admin inference settings configure Kuno hosts, capabilities, placement policy,
collections, and profiling controls. Routine users see actionable status; detailed
device/runtime diagnostics belong in expandable or administrative views.

## 12. Delivery milestones and acceptance

### M1: Ollama gateway and optional takeover

Add the Kuno binary, config, separate compatibility/control surfaces, one upstream,
transparent streaming proxy, basic request classification, bounded queues, resource
reservation hooks, and reversible systemd takeover. Record aggregate real-use
counts and last-use times immediately so eligibility data accumulates while later
milestones are built. Provide explicit gateway-only development mode before live
adoption. Discover and preserve existing loaded-model limits for later managed
scheduling and rollback.

Acceptance:

- Ollama CLI and native/compatibility API clients operate through the original URL.
- Unknown fields/routes and upstream errors retain their contract; supported
  streams are incremental, and cancellation releases work/reservations correctly.
- Known GPU work and opaque operations cannot bypass basic admission; an external
  client waits behind a simulated exclusive image reservation.
- Takeover handles occupied ports, startup/health failures, restart, repeat setup,
  and restoration without losing unrelated service settings.
- Kuno/control privileges are separated; no inference content is written to logs
  or files, including failure paths.

Passing proxy tests alone does not certify GPU-only placement or image generation.

### M2: Resource admission, placement, and recovery

Current implementation: native NVIDIA/Intel telemetry, a backend-independent
placement policy, durable GPU/worker reservations, and owned CUDA/Vulkan Ollama
workers. The optional managed gateway supports native inference plus OpenAI-style
chat/completions/embeddings, preloads, verifies digest/context
and GPU-only residency, then forwards inference while retaining its reservation.
Real acceptance tests passed on this host's RTX 3060 and Arc Pro B70, including
multiple residents, eviction/recheck, cancellation recovery, text, embeddings,
and gateway routing. Workers override the old one-model ceiling independently
of the original catalog service.

Default installation remains gateway-only. `configure-managed --workers FILE`
now installs a configured worker pool with scoped model access, GPU discovery
before listener activation, and transactional rollback. `--disable` restores
gateway mode; upgrades preserve the chosen mode. The experimental route has
RAM-only load observations, not measured inference peaks or persistent learned
profiles. Remaining compatibility/opaque fallback is serialized and stops workers first,
but does not enforce GPU-only execution. Independent-device request concurrency,
external-pressure classification, durable fingerprinted learning, and automatic
hardware selection remain before default M2 activation. Missing telemetry stays
unknown and failures do not create combination blacklists.

Initial failure evidence is now implemented: sample before cleanup, stop the
affected owned worker, settle unloaded telemetry, and compare with a provisional
baseline. The control API exposes bounded RAM-only observations and distinguishes
unexplained pressure, missing/unstable telemetry, and recovered memory requiring
controlled retest. Capacity-related failures with other residents can retry once
in isolation before user inference is dispatched. Process attribution, controlled
idle retests, and persistent configuration-specific exclusions remain pending.

Add real telemetry, runtime/device control, residency verification, conservative
memory estimates, and the complete failure-classification sequence. Activate
Kuno's managed `OLLAMA_MAX_LOADED_MODELS` override so an existing one-model cap
does not prevent admitted models from remaining resident together.

Acceptance includes single-GPU contention, independent GPUs, multiple warm LLMs,
embeddings/titles alongside chat, bounded spill recovery, cancellation, delayed
memory release, external GPU pressure, incomplete telemetry, and clean retests.
No active unrelated job is interrupted to classify another failure.

Verify adoption of an existing limit of `1`, effective configuration after
restart, multiple GPU-resident models within budget, and rejection or recovery
of CPU spill despite the higher ceiling. Test idempotent setup, conflicting
overrides, and restoration of both an explicit original limit and an originally
absent setting. Confirm that idle profiling's limit of three does not become a
production residency cap and that context/parallelism settings remain unchanged.

### M3: Learned profiles and idle testing

Add content-free profiles, predictions, seven-day eligibility, usage priority,
and one-to-three-workload idle experiments.

Acceptance includes the 30-minute gate, no refresh from tests or metadata polls,
real demand from non-Vashti clients, immediate foreground preemption, interrupted
test neutrality, bounded search, expiry/re-entry, and version-sensitive invalidation.
Tests must distinguish loading, exercised workload bounds, and concurrent peaks.

### M4: Native Qwen image execution and clean comparison

Development checkpoint (2026-09-23): the exact PNG recipe importer, verified
three-file model downloader, and pinned native Vulkan worker are implemented.
The installed NVIDIA path passes image execution, LLM eviction/reload, and
disconnect cleanup. Vashti can inspect/export imported settings. Full reference
execution, a fresh ComfyUI comparison, B70 reliability, and the remaining M5
generation controls are still required; this milestone is not complete.

Implement the exact first recipe and collection using the native adapter. Use a
fresh, pinned ComfyUI environment and fresh verified model files for comparison.
Share that fresh model store where formats permit. Do not use or repair the old
Stability Matrix installation as a baseline. Comfy is a development reference,
not a product runtime dependency.

The comparison harness captures images in memory; normal Comfy save/preview
nodes write files and must be bypassed. User-content persistence remains solely
under Vashti's applicable policy.

Acceptance:

- Recognize the PNG recipe exactly, including indirect prompts and inactive LoRA
  notes. Use a compact synthetic metadata fixture for committed parser tests;
  the user's currently untracked PNG is a local fidelity reference.
- Verify model/scaling handling, sigma schedule, CFG, noise/seed behavior, latent
  layout, and VAE normalization against the reference.
- Measure cold/warm phase times, peak GPU/system memory, actual execution backend,
  fallback, cancellation, and release on the B70; compare the 3060 where feasible.
- Confirm LLM-to-image-to-LLM handoff and worker-crash cleanup.
- Audit filesystem/log writes during success, failure, profiling, and cancellation.
- Select the production image backend from these results; disclose remaining
  fidelity limitations without promising pixel identity or a speed advantage.

### M5: Vashti integration and installable MVP

Deliver image controls/tool, permissions, progress, import/reuse, correct artifact
ownership, private mode, admin collections, packaging, startup, diagnostics, and
runtime rollback. Validate interrupted downloads, disconnected clients, Kuno
restart, unavailable storage, service startup, and narrow-screen behavior.

## 13. Initial implementation checkpoint

Start with M1 on an isolated implementation branch. Develop against a separate
listener and fake/controlled upstream before adopting the running service.
Complete the proxy, rollback, privacy, and basic admission checks before changing
the production listener. No image-model download or Comfy installation is required
to deliver that checkpoint.

Exact memory reserves, cancellation grace periods, compatible runtime builds,
wire-schema details, and hardware performance thresholds are implementation
decisions to measure and document. They must preserve the product and privacy
requirements above. This specification does not itself install software, move a
port, or change the host's model limits.
