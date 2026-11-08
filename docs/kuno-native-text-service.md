# Native Intel text service preview

## Active extension contract (2026-10-19, installed backend acceptance)

The user requested thinking, tool use, faster cold load and configurable context.
Image-enabled gateway `36626d30` and selected prompt worker `c8db19de` are installed.
The prior toolbox repair `a6b21a38` remains included. Exact staged numerical,
thinking/tool/code-quality/privacy/recovery, official installation and final
installed public-request/deployment metadata gates pass. See the
[current prompt-processing receipt](#native-strata-prompt-optimization-2026-10-20).
The previous worker60c20e36 is the historical
[baseline](#native-prompt-processing-optimization-2026-10-19).
The earlier actual-service/live Vashti all-tools checks are recorded in the
[toolbox repair receipt](#kuno-toolbox-prefill-repair-2026-10-19). Earlier preview
receipts below are historical. Ollama chat behavior follows
the actual embedded Qwen template:
`think` accepts booleans or low/medium/high/xhigh (absent remains false), reasoning streams separately
from content, and XML-style function calls translate to structured Ollama
`message.tool_calls` with typed argument objects. Assistant thinking/tool-call
history and role=tool results round-trip; Kuno never executes those tools.
The installed `/api/show` advertises completion, thinking and tools after actual
model round-trip validation. Vashti UI acceptance is recorded separately below.
Unsupported modalities/formats continue to fail explicitly.

Cold-load integrity work moves full SHA verification to administrator enrollment.
Root-controlled model identities (device/inode/size/mtime/ctime) can authenticate
unchanged artifacts on later starts. A changed identity must undergo full
verification; old settings without an identity continue using full verification.
This cache contains model metadata only, never prompts, outputs or content hashes.

`num_ctx` should select the request budget, bounded by the engine's allocated
capacity and the artifact's advertised262144-token limit. The present compiled
32K target/MTP geometry and16-bit graph keys must be extended together before
accepting larger values. Do not merely remove the API check, truncate tokens or
advertise262144 as live-supported before the real target/draft bounds work.
No Vashti consumer/UI changes are authorized in this backend task.


This is the backend integration of the selected Rust engine, following the
selected-engine audit in `kuno-strata-current-source-audit.md`. The 33.3707 warm
rate is one fixed public engine workload, not a promise for every API request.
The initial service is a deliberately narrow, opt-in text preview. This document
specifies its contract; installation and live acceptance are recorded separately.

## Enrollment and model identity

`Config.native_text` is optional and defaults to absent. When absent, ordinary
Ollama forwarding is unchanged. Enrollment supplies a unique `kuno-` model name,
an absolute frozen worker executable, an absolute settings JSON path, and its
physical PCI address. The first local name is
`kuno-qwen3.8-flash-next:iq3_s`. It refers to the original two DASLab IQ3_S shards
(83,617,662,656 bytes together) and the original Q2 MTP sidecar, both on Intel B70.
It does not replace an Ollama Q4 model with a differently quantized artifact.
Do not enroll a name that exists in Ollama.

The administrator installs and verifies these assets before enrollment. The
worker settings contain `shards: [{path, sha256}, ...]`, `mtp: {path, sha256}` and
`device_id: 57891` (0xe223). Paths refer to existing read-only model artifacts;
there is no second 83GB model copy. A same-filesystem hardlink may put the MTP
sidecar under the existing read-only model-store mount. Enrollment hashes complete files and records their verified filesystem identity
in root-controlled settings. Unchanged enrolled files skip subsequent complete
hashes; missing/untrusted/changed identities require full verification. The worker
validates original inventory and graph and keeps backing alive until Session close.
Reported cold load includes all actual verification and engine setup.

Discovery extends `/api/tags` and `/v1/models`, preserving original entries and
extra JSON fields. `/api/show` describes the owned entry, advertising completion,
thinking and tools. `/api/ps` includes the native entry once resident; unknown actual VRAM size
is omitted rather than reported as CPU residency. The digest identifies the
original first shard, not a hash of user content. No Vashti protocol types change.

## Current request support

- `/api/chat`: text `system`, `developer`, `user`, `assistant` and `tool` messages;
  assistant thinking/tool-call history and tool results round-trip.
- `/api/generate`: ordinary prompt/system rendering or explicit `raw: true`.
- Streaming NDJSON and `stream: false`, UTF-8 across token boundaries, actual
  output limits, EOS/length reasons and token/timing fields.
- Non-thinking rendering matches the accepted Qwen public fixture template.
  Thinking is off by default. `think: false` disables it; true enables high effort;
  strings low/medium/high/xhigh select the embedded template instructions. Reasoning
  is returned separately in `message.thinking`; function calls are structured
  `message.tool_calls[].function.{name,arguments,index}`. At most128 tools/calls;
  Kuno never executes the functions. Raw generation with explicit thinking fails.
- Options: `temperature` (default0), `top_k` (20), `top_p` (.95), `min_p` (0),
  unsigned `seed` (0), `num_predict` (1024), and `num_ctx`512..32768 (default32768).
  Input tokens plus output budget must leave the original eight-token headroom.
  Smaller budgets enforce a request limit without shrinking the loaded32K buffers.
  No prompt truncation; overflow is400 and retains the clean loaded worker.
- `keep_alive` absent or -1 retains the model until competing work or shutdown.
  `keep_alive: 0` unloads after completion; an empty request with0 explicitly
  unloads. Other durations are not implemented in this initial contract.
- An empty preload request with retention loads the real model and performs one
  fixed, discarded one-token warmup, then fully erases its request state before
  returning `done_reason: load` with zero delivered tokens. It grants no shortcut
  around subsequent request rebind/erasure checks.

Unknown options, images, format constraints, embeddings, OpenAI inference routes
and model-management operations for this
owned name fail before inference. They are never silently dropped or forwarded
to Ollama. `/v1/models` discovery does not imply OpenAI inference acceptance.
Ordinary other model names and opaque endpoints continue through Ollama.
A rejected owned request is not retried through another engine.

## Process, scheduling and privacy

The existing global admission lease covers the whole request, including cleanup.
Native admission evicts image residency and stops owned Ollama workers. Competing
Ollama/image inference first closes the native worker. Metadata does not unload it.
The retained Session preserves model weights, original placement and aggregate
adaptive usage, but clears prompt, KV/recurrent state, suffix state, sampling and
other request owners before acknowledging DONE. Mixed prompts must be validated
on this caller independently of the identical fixed-public benchmark.

The worker uses inherited pipes, with a bounded private binary ABI (`KNT2`/`KNR2`),
not command-line content, temp files or a separately exposed HTTP listener. Its
parent clears the environment, disables disk shader caches and discards stderr.
The child enforces no swap, no core dumps, non-dumpability and the existing
Landlock write policy before receiving content. It installs a redacted panic
hook and a default in-memory CubeCL configuration. It never enables laboratory
public-output authorization, receipt writers or operator profiling. Raw backend
errors are not returned to callers. Only fixed error categories cross the pipe.

A client disconnect drains and discards output while the lease remains owned,
matching the existing gateway drain contract. There is no default inference
header/total/idle deadline. Errors before first output return an HTTP error;
after streaming begins they produce a normal NDJSON error frame. Unload waits
for checked close, then uses a bounded process-reap fallback if needed. An
already-exited child is still waited/reaped: a failed kill of an already-dead
child is not a permanent admission latch. Unconfirmed reaping retains uncertainty.
No automatic GPU reset, driver reload, reboot or request replay is performed.

## Vashti handoff and remaining limits

No connector/UI implementation belongs to this task. Consumers use the existing
Ollama fields for thinking, tools and request options; refresh `/api/show` to learn
capabilities. Unsupported penalties/num_gpu/num_thread must be cleared explicitly,
not silently stripped. Context is a512..32768 request budget, not yet a configurable
larger allocation. Default-native bindings of existing names, OpenAI adapters,
contexts above32K, broader soak and general runtime packaging remain open.

Gateway and worker MUST be deployed together: KNT2/KNR2 changes the private frame
from32 to36 control bytes to carry context. A new gateway with the old KNT1 worker
is incompatible. Use the new gateway's `configure-native-text` transaction with
the new frozen worker and trusted settings; do not first install a gateway-only
upgrade. REJECTED frame6 covers safe pre-execution tokenizer/context rejection,
which leaves the loaded worker usable. Fatal/backend errors still close and reap.

Installation must preserve the existing CUDA/Vulkan image-enabled Kuno binary
features and managed SYCL runtime. A gateway-only replacement is not acceptable.
Before replacing live services, verify the frozen native worker with a protected
framed caller, then verify the staged gateway's catalog, real streamed/nonstreamed
mixed prompts, preload, unload, subsequent ordinary Ollama inference, recovery
and absence of content files. Keep an exact binary/config rollback.

## Historical initial integration checkpoint (2026-10-19 UTC)

- Worktree `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
  `codex/strata-rust-port`. Do not edit the canonical Vashti checkout.
- Initial adapter7f54777a, protected Vulkan/pipe correction39c3e8e1,
  transactional enrollment1c4cf520, actual-Ollama-CLI defaults handlingd4bb87fb.
  Empty serialized suffix/template/options defaults are accepted while real
  unsupported features continue to fail explicitly.
- Frozen selected worker `strata-native-worker-scope-20261019`, runtime39c3e8e1,
  SHA25680890bcebd620244b12b80c5b31f3a34460075f061c0fbc10b33f653b7d6354f.
  Its wire/math are unchanged by gateway-only enrollment and parser changes.
- `native-worker-framed-scope-check-20261019` completed real preload,362-output
  public code,64-output capped prose and two identical23-output sampled repeats.
  All five requests returned only after the real engine request-erasure path.
  The harness then FAILED attempting to read the non-dumpable child's `/proc/io`
  as the unprivileged caller. This is not an engine numerical failure, but it is
  NOT final-close/complete-private-caller acceptance: its exception path killed
  the idle child. Root observer confirmed normal admission,50,688,000B Intel,
  NVIDIA compute0 and no driver faults. Do the privileged IO inspection and
  graceful close in the staged gateway test; do not relabel this failed receipt.
- First candidate5d1c238c failed before inference because the private mode had
  not enabled all original HAL gates. Its physical recovery also passed.
- Final Kuno CPU run:226 passed across15 suites,7 hardware tests ignored.
  Root all-target and selected worker strict Clippy pass. The first final parallel
  run failed the existing privacy fixture
  `binary_writes_only_content_free_state_and_diagnostics` (initial pause expected
  503, observed200). Its isolated repeat and the full serial rerun passed. A
  parallel ephemeral-port race is suspected but not established; preserve the
  failed log rather than claiming that cause proven.
- Image-enabled replacement build preserves CUDA/Vulkan and the existing
  managed SYCL bundle. Fresh pinned source/build under this worktree's
  `target/kuno-sd-{source,build}-88411ef-b51e05e8aaf4e67e[-cuda]`.
  Use default `GGML_CUDA_NO_VMM=OFF`, matching the installed binary. An accidental
  ON preparation failed its CUDA-driver link and was not installed; restored OFF
  rebuild succeeds, including embedding, attention and36 loader CPU cases.
  Gateway Cargo target: `/media/jmicko/faststore/kuno-development/native-install-target-20261019`.
  Final frozen gateway is runtime d4bb87fb65d3386a8a8b8314d924c4d428c6684f,
  SHA25681370d88a4572aadee65f2cb621f12c5ff99bbe63b9552c688846e868314be92
  (`native-gateway-frozen-20261019`). No SYCL link dependency was added to it.
- `native-gateway-stage-check-v3-20261019` passes actual Kuno/installed-Ollama
  CLI generation,64-output capped nonstreamed prose, three-frame multi-turn
  streaming, resident metadata, unsupported-format400, empty Go-style preload,
  graceful native unload and subsequent real ordinary Ollama inference. Root
  inspection measured worker `write_bytes=0`; unprivileged inspection was denied.
  Final Intel50,688,000B, NVIDIA compute0, admission normal, all services HTTP200,
  no classified driver-fault delta. v1/v2 failed before inference because the
  systemd test client lacked HOME; v3 supplies an isolated client home. They are
  failed harness attempts, not successful backend acceptance receipts.
- Official `upgrade --yes` and `configure-native-text --settings FILE --yes`
  completed on this machine. The gateway and worker match the frozen hashes;
  existing image, managed-worker and listener settings are unchanged. Exact old
  binary/config backups and the installation receipt are root-only under
  `/var/lib/kuno-native-preview-20261019`. Deployment ownership/hashes remain
  managed by the normal reversible transaction; do not manually replace files.
  Enrollment copies root-controlled source worker/settings into the existing
  reversible deployment and preserves them across later gateway/image upgrades.
  `native-installed-check-20261019` now passes real inference through this
  installed identity and model mount. See the installed acceptance below.
- Model settings for installation must reference the existing read-only model
  mount `/var/lib/kuno/models/.kuno-import/...`. Hardlink the existing889MB MTP
  into that store on the same filesystem, not a second83GB target download.
  Test scripts/settings/frozen artifacts and classified receipts live under
  `/media/jmicko/faststore/kuno-development`; they contain fixed public fixtures
  only. Do not save generated text or private caller bodies in the next tests.

## Profile and next optimization

The protected warm target-window profile `strata-adapted-profile-check-20261019`
uses runtime4771bd05 and frozen SHA256
1ab7c9646ba7595d8d51d340ef9b5490457f51bc28286a9b2f931b0cfdedff0f.
It profiles only window64 of the retained second request, after real adaptation:
362 outputs/EOS,4887 published exchanges, repeat equality, real request erasure,
checked close and physical recovery all pass. Its serialized target sample is
85.754117ms across47 expert layers:

| Target operator | GPU milliseconds | Share |
| --- | ---: | ---: |
| Joint quantized expert gate/up | 25.484 | 29.72% |
| Quantized expert down | 19.064 | 22.23% |
| Packed dense Q8 | 10.189 | 11.88% |
| All other sampled target work | 31.018 | 36.17% |

This diagnostic does not measure overall throughput, draft execution, cold load
or request cleanup. The selected fixed-public warm engine rate remains33.3707
tokens/s from the separate accepted measurement. Do not present target-window
percentages as an end-to-end speedup prediction or universal workload result.

The next candidate is the coupled quantized expert path: inspect memory stalls,
register pressure and subgroup occupancy in joint gate/up and down, then change
their access/dot scheduling together while preserving the original packing,
real adaptive-source addresses, target verification and numerical contract.
Packed dense work is the second candidate. BF16 matrix work is only3.50% of
this sample, so a standalone BF16 rewrite is not the leading measured target.
Do not replace retained-cache placement or draft speculation merely because
this target-only profile does not include their CPU/transfer costs.

## Historical initial installed acceptance (2026-10-19 UTC)

`native-installed-check-20261019` passes through the actual192.168.1.164:11434
listener: ordinary Ollama generation/unload, native362-output code response/EOS,
and a different three-frame multi-turn streamed reply on the same native PID.
All caller prompts are fixed public fixtures; replies and their hashes are not
saved. The live worker is nonroot under the actual DynamicUser service, can read
the actual read-only model mount, has swap/core disabled and denies unprivileged
process inspection. Root inspection measures `write_bytes=0`. Cold reported load
is223.333s including full artifact verification; total first request260.914s.
This cold request is not the warm33.3707-token/s public benchmark.

Both the gateway and worker hashes match the frozen artifacts above. Existing
image settings, managed settings, upstream127.0.0.1:11435 and listen[::]:11434 are
preserved. The MTP sidecar is hardlinked into the existing model store. No target
weights are duplicated. Existing managed SYCL assets are preserved; no new image
visual/fidelity run was performed, so CPU/image-build coverage does not establish
new live image acceptance. No frontend files, release version or published
package changed.

After acceptance, all three service HTTP probes are200; admission is unpaused,
inactive, waiting0 and recovery_required=false. Classified Xe timeout/wedge/
reset/memory-fault and NVIDIA Xid counters remain0. The native weights are
intentionally retained (Intel33,312,804,864 bytes at this check); NVIDIA compute0.
Do not call this an idle-GPU recovery receipt. Graceful native unload and physical
recovery were separately accepted in the staged test above.

Use `kuno run kuno-qwen3.8-flash-next:iq3_s`, or select this entry through the
existing Ollama connector. For Vashti, use a fresh text conversation with tools
and thinking disabled, supported sampling options only, and no context override
(or32768). The connector serializes optional penalties/num_gpu/num_thread when
set, and those are unsupported400 for this entry; clear them rather than silently
stripping a user's request. This task does not change Vashti code or claim an
end-to-end private-chat UI run. Any capability-aware setting controls and native
model presentation belong to the Vashti task. No protocol crate migration is
required for basic text discovery/chat.

`kuno stop kuno-qwen3.8-flash-next:iq3_s` explicitly frees it. Ordinary subsequent
inference automatically closes it first. To remove enrollment while preserving
Ollama/image configuration, use `sudo kuno configure-native-text --disable` and
confirm the restart. Normal setup transactions retain rollback ownership; keep
the root backup under `/var/lib/kuno-native-preview-20261019` until accepted by
the user. This is a local development preview, not a released/downloadable native
runtime. Broader long-context/soak, tool/thinking/OpenAI support, other GPUs/models
and automatic name bindings remain open.

### Retained installed API timing

`native-installed-check-20261019/warm-repeat-receipt.json` passes two identical
362-output/EOS code responses through the installed LAN-address API on the same
retained worker. Temperature0, seed0,35 rendered prompt tokens,384 output cap;
original public merge-sorted-lists fixture as in the selected benchmark. Neither
response nor its hash is persisted. Both report zero load duration.

| Request | Delivered eval tokens/s | Eval seconds | Full request seconds | Full request tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Warm code after mixed prompt | 33.6602 | 10.7546 | 20.6869 | 17.4990 |
| Identical warm repeat | 36.1275 | 10.0201 | 18.3087 | 19.7720 |

The API eval timer excludes prompt/prefix processing and checked private erasure;
total_duration includes them. The second prompt processing duration is5.2690s.
Do not call36.1275 the whole-request rate. This is one fixed public workload,
not general throughput, a new Ollama comparison or a long-context soak. Final
`host-final.json`/`final-ready.json` independently confirm normal admission,
healthy services, no active test units, zero worker write_bytes and driver faults,
NVIDIA compute0 and intentionally retained native weights.

## Extension source and CPU acceptance (2026-10-19)

The following changes from `36626d30` are now installed. The original CPU-only
checkpoint preceded the protected validation and installation recorded below.

Implemented source changes:

- `native_chat.rs`: rendering based on the actual enrolled shard's
  `tokenizer.chat_template` (including original XML tool instructions), separate
  streaming thinking/content, typed function argument objects and complete
  assistant/tool-result history. Boolean/low/medium/high/xhigh reasoning modes;
  absent thinking remains disabled so the established fast non-thinking path
  stays explicit and unchanged. Raw generation with explicit thinking is rejected
  rather than ignoring the requested feature. Kuno returns calls; it does not
  execute them. Tools are limited to128 definitions/calls and safe XML names.
  This does not implement JSON-schema output constraints or OpenAI inference.
- `/api/show` advertises completion/thinking/tools after actual round-trip
  validation in the protected staged and installed checks.
  Existing clients should use the standard Ollama `message.thinking` and
  `message.tool_calls[].function.{name,arguments,index}` fields. A successful
  CPU parser test is not evidence that the model will choose the right tool.
- `native_artifact.rs`: full enrollment hashes with descriptor/path identity
  checks before/after reading. Root-owned installed settings retain only model
  dev/inode/length/mtime/ctime; unchanged files skip repeated full startup hashes.
  New source settings must undergo full verification even if they supply their
  own identity. Only the existing transaction-verified installed settings can
  reuse enrollment identities. Untrusted development settings use full hashes.
  No prompt/response hash or request-derived cache is added. Re-enrollment updates
  changed-but-still-correct artifacts; failed hashes never refresh the cache.
- Private ABI is now KNT2/KNR2,36 control bytes including u32 context budget.
  Gateway and worker MUST be deployed together in one configure-native-text
  transaction using the NEW gateway executable and NEW worker/settings source.
  Do not run an intermediate gateway-only upgrade against the old KNT1 worker.
  Nonfatal REJECTED frame6 acknowledges pre-execution tokenizer/context rejection;
  it preserves the clean loaded worker, preventing a user context error from
  forcing another cold load. Fatal/backend errors still close/reap normally.
- `num_ctx` now accepts512..32768 as an enforced prompt+output+eight-headroom
  request budget, with no prompt truncation. Smaller budgets do not shrink the
  still-fixed32K model allocation. Prompt overflow returns400 without discarding
  loaded weights. Larger contexts are NOT implemented yet: engine32K constants,
  target/MTP graph u16 capacity fields, QSA/draft cache and RoPE sizes, suffix/HOST
  bounds, prompt overlay ownership and cache/VRAM pricing must change together.
  The actual artifact metadata advertises262144; do not report that as the live
  runtime capacity. This is the remaining substantial part of the user's request.

CPU cases cover split-at-every-character thinking/tool delimiters, typed XML
arguments, malformed/incomplete calls, history/tool-result rendering,
stream versus collected response shape, context frame round trips/version
mismatch, artifact modification/replacement and rejection of untrusted cache
identities. Relevant root/worker strict Clippy and selected worker compile checks
pass. Final root suite:234 passed,7 hardware tests ignored (including the
updated capability fixture); logs are `native-chat-final-all-tests-20261019.log`,
`native-chat-clippy-20261019.log` and `native-chat-worker-clippy-20261019.log`
under the development artifact directory. Build artifacts are recorded after
freezing the source.
Existing inherited untracked benchmarks/text/__pycache__ is preserved.

Protected staged and installed fixture acceptance is complete; see the exact
receipt below. Larger runtime capacity remains the next substantial implementation
item. No release or frontend changes. Vashti default-tool UI failure is under
investigation; fixture acceptance does not establish full application acceptance.

### Previous extension artifacts (before the toolbox repair)

Runtime source `36626d306d5e47ff0270fce3917bca456edab1a2`:

- Image-enabled gateway: `native-chat-gateway-frozen-20261019/kuno`, SHA256
  `d3ed07f473b28bbb46596a619349d933edd6c166b7af10dd42f9821e4ba3048c`.
  Native-image features and the same CUDA/Vulkan dependencies are retained;
  readelf shows no new SYCL dependency or RPATH.
- Worker: `strata-native-chat-20261019/kuno-strata-rust-f871238b732d344ca28ebf4adac33a42dbc7d2e2a713675e049dd3466dcce499`,
  SHA256 `f871238b732d344ca28ebf4adac33a42dbc7d2e2a713675e049dd3466dcce499`,
  features strata-rust-port,strata-intel-core,strata-gpu-adaptive-usage.
- These exact artifacts are installed and protected GPU-validated. Source and
  build logs live under `/media/jmicko/faststore/kuno-development`. Do not reuse
  the previous install script's old hashes/worker or KNT1 staged acceptance.

## Protected extension acceptance and installation (2026-10-19 UTC)

Source `36626d306d5e47ff0270fce3917bca456edab1a2`; documentation dependency
checkpoint `a35cb880`. The gateway and worker hashes in the artifact section
match the installed `/usr/local/lib/kuno/{kuno,native-text-worker}` exactly.
The new gateway's official `configure-native-text` transaction installed both
without an intermediate incompatible wire version. Existing complete configuration
is unchanged, including listen[::]:11434, upstream127.0.0.1:11435, managed workers
and CUDA/Vulkan/SYCL image settings. All three model identities are root-enrolled.
Root-only backup and installation receipt: `/var/lib/kuno-native-features-20261019`.
No release/version change, published package or frontend implementation change.

All following development receipts are under
`/media/jmicko/faststore/kuno-development`:

- `native-features-stage-check-20261019/receipt.json`: fixed public code362-output
  completion, low reasoning, streamed true reasoning, reasoning disabled,
  actual structured weather call -> synthetic tool result -> final answer,
  streamed tool call, context-overflow400 then successful smaller-context reuse
  on the same worker PID, checked graceful unload/reload and subsequent ordinary
  Ollama inference all pass. Worker write_bytes0; unprivileged inspection denied.
  Two root-attested cold starts measure102.813357 and102.314585 seconds.
- Staged `host-after.json`: Intel50,688,000 bytes after unload, NVIDIA compute0,
  all three service probes200, admission unpaused/inactive/waiting0/no recovery,
  no change to the zero classified Xe/NVIDIA fault counters. No active staged
  worker/test service remains.
- `native-features-installed-check-20261019/receipt.json`: actual LAN listener,
  DynamicUser identity and installed model mount pass low reasoning, actual
  tool-call/result exchange and real `/usr/local/bin/kuno run --think=low`.
  Cold reported load101.981292 seconds. Root observes write_bytes0, nonroot worker,
  swap/core0 and denied unprivileged inspection. Prompts/replies remain transient;
  receipts contain only booleans, numeric timings and public model/build metadata.
- Installed `host-after.json`: services200, normal admission and no driver faults;
  native model intentionally retained, Intel33,340,067,840 bytes, NVIDIA compute0.
  This is retained residency, not a fully idle-GPU recovery claim.

The three unchanged-artifact cold loads are about54% shorter than the historical
223.333-second installed cold request. Full hashing is still performed at first
trusted enrollment (and after identity changes); it is not omitted. About102
seconds of actual loader/engine setup remains. These are load-time results, not
new generation-throughput measurements or broad soak acceptance.

### Historical Vashti browser check: default tools blocked before the repair

The user explicitly requested this application check. A fresh private chat in live
`https://vashti.crabhonk.com/` discovers the native thinking/tools capabilities
without frontend changes. Fixed public arithmetic only; no existing conversation
is opened. Low thinking with Calculator enabled and Notes/Images disabled completes
reasoning -> actual `calculate` call -> tool result -> final answer. Vashti renders
two reasoning sections, the calculator result and the final numerical answer.
Observed UI stats:73 output tokens,2,006 context tokens,5.1 eval tokens/s, about3m
cold total including1m43s load and53.3s prompt evaluation. This is a different,
two-round workload; the historical short-code33..36 eval rate is NOT universal.
A screenshot of only this public test result is saved by the browser test client
at `/tmp/kuno-vashti-thinking-calculator-20261019.png`, not by Kuno.

**At this historical checkpoint, default configuration was NOT accepted:** with Notes/Calculator/Images all
available, the same public prompt returns503 before visible output. Reproduced
twice on retained workers and once after a fresh worker start. The generic
Vashti error does not expose the upstream static error body. This rules out a
retained-only failure, but does not yet identify the cause. A larger prompt/
prefill or tool-set difference is a hypothesis, not a proven diagnosis. Do not
silently drop tools, route to Ollama, relax fault/erasure checks or claim a fix.
Next reproduce with a compiled public long/tool fixture and inspect bounded
content-free error stages; compare actual prefix geometry with the passing
2,006-token calculator case. Add regression acceptance before claiming the full
application works. Contexts above32K remain separately unimplemented.

The browser leaves the successful response selected (version2/4) in this one
private test chat, with Calculator on and Notes/Images off for that chat only.
The failed variants remain as diagnostic history in Vashti's device storage.
No global setting, consumer source or existing conversation was changed.

After the failed requests, Kuno reaps the worker and recovers automatically:
`native-features-vashti-ui-host-final-20261019.json` confirms active=false,
paused=false, waiting0, recovery_required=false, all three services200,
Intel50,688,000 bytes and NVIDIA compute0. Classified Xe/NVIDIA faults remain0;
the live cgroup records no OOM, OOM kill or memory-limit event. No native worker
or active test client remains. The installed extension remains available for
supported calls; model weights are now unloaded after this application check.
These limits supersede any earlier claim that the weights remain warm at finish.


### Larger-context implementation dependencies

The next capacity change must preserve the exact default32K path while adding
immutable per-worker capacity selected before constructing target/draft owners:

1. Widen `NativeStrataBranch.context_capacity` and all target/commit/MTP constructor
   and geometry signatures from u16 to u32, including checked caller conversions
   in resident_model_batch, resident_native_commit and resident_mtp_window/replay.
   Keep positions/boundaries checked against the actual branch capacity, not only
   the model's262144 ceiling.
2. Replace selected fixed constants across strata_intel_core, qsa_gpu,
   mtp_resident_kernels, strata_sampler and strata_speculation with one consistent
   immutable loaded capacity. Suffix hash slots require power-of-two geometry;
   the request budget may be smaller than the allocated capacity. Growing past
   an existing capacity must close/reprice/reload; smaller requests can retain it.
3. Price and allocate QSA/KV/indexer caches, RoPE tables, draft attention scratch,
   suffix HOST buffers and prompt score/root layouts using that same capacity.
   `strata_prompt_allocation::Plan` currently switches to CacheLoan above32768,
   while `strata_engine::Session` deliberately passes a zero source loan. That
   path cannot simply be enabled: implement the real overlay ownership or use
   correctly priced owned roots and reduce expert residency to make room.
4. Prove32K byte/geometry regression equivalence, bounds near65536 and higher,
   target/draft position correctness and real physical memory recovery before
   advertising a larger tested runtime limit. Model metadata alone is not enough.
   No silent prompt truncation or pretend support by dropping num_ctx.

After removing repeated hashes, instrument only allowlisted content-free startup
phase durations (artifact verification, graph/tokenizer preparation, target,
source cache, shared weights and draft) to find the remaining cold-load cost.
Do not enable disk shader/graph dumps or persist request-dependent programs just
because a development run uses public fixtures. Startup gains are not measured
until the protected candidate actually runs.

## Kuno toolbox prefill repair (2026-10-19)

The observed default-tool failure is a Kuno engine bug, not a Vashti tool-format
problem. Native prompt allocation/controller uses4096-token chunks, but the
checked PLE history helper still limited its token slice to2048. The mismatch
rejects publication after real GPU prompt execution, retires the worker and
returns503; it is not OOM or a wedged GPU. Compiled-public diagnosticse8dbea20
and a3894733 reproduce it without any tool and expose the exact fixed label:
`checked PLE chunk transition: invalid parameter`. These diagnostics accept only
compiled public text, never stdin; the private worker cannot enable diagnostics
through environment or request flags. No private content logging is added.

Fixa6b21a387eb181db24a304f8ff469cb6ba75ee11 introduces one core
`MAX_PROMPT_BATCH=4096` used by both PLE checks and GPU prompt-owner allocation.
Prompt arithmetic, MAX8 speculative bounds, owner identity, cancellation, real
whole-model fences, finite-value checks, erasure and32K context are preserved.
Regression tests compare history against the independent scalar token reference
across2048/2049/4095/4096, repeated chunks and short continuations, and verify
4097-token rejection and abandoned proposals leave original owners unchanged.
Tail-history tests retain signed-zero bits and avoid a chunk-sized host mirror.

Validation/artifacts under `/media/jmicko/faststore/kuno-development`:

- `native-ple-boundary-tests-20261019.log`:10 checked-PLE cases pass.
- `native-ple-full-library-tests-20261019.log`:146 pass,2 ignored.
- `native-ple-library-clippy-20261019.log` and
  `native-tools-worker-clippy-20261019.log`: relevant strict library/tests and
  selected worker Clippy pass. The broader all-targets attempt failed in
  unmodified reference examples with missing `crate::strata_port` and lab
  feature cfgs; preserve `native-ple-boundary-clippy-20261019.log`, do not claim
  every workspace example passes.
- `native-public-long-prompt-{check,device-check}-20261019` preserve the two
  FAILED original-bound controls and physical recovery; do not relabel them.
- `native-public-long-prompt-fixed-check-20261019/receipt.json`: SAME public
  fixture now passes,8,445 prompt tokens/4 delivered outputs/correct answer,
  three actual prompt chunks and checked request erasure/final close. Replies
  remain in protected client RAM. Its host-after confirms Intel50,688,000B,
  NVIDIA compute0, all three services200, unpaused/inactive/no recovery and
  unchanged zero classified Xe/NVIDIA faults. This is correctness, not a speed
  benchmark or blanket long-context acceptance.
- `strata-tool-prefill-fix-20261019`: frozen sourcea6b21a38, worker SHA256
  `7bc1633bc2bc80d2a3ef27ddbc3366f93af5a6d90c13da03a19cf07d313ef30c`,
  unchanged strata-rust-port,strata-intel-core,strata-gpu-adaptive-usage features.

### Staged API, installation and actual Vashti acceptance

`native-toolbox-stage-check-20261019/receipt.json` passes the real private worker
through staged Kuno with10 available function schemas: preload,4,207-token
streamed thinking/tool call, synthetic weather result and final answer on the
same PID, zero worker write_bytes, denied unprivileged inspection, graceful
unload and subsequent real ordinary Ollama inference. Its host-after confirms
Intel50,688,000B, NVIDIA compute0, all services200, normal admission and zero
classified driver-fault changes. The public fixture exposes a mixed notes/image/
calculator/weather toolbox; only the weather function is executed by the test
client. No real notes or images are accessed, generated or persisted.

The official `configure-native-text` transaction installs worker7bc1633b using
the existing image-enabled gatewayd3ed07f4. Private KNT2/KNR2, model identity,
all existing configuration/ports/image/managed settings remain unchanged. Both
installed artifact hashes match the frozen files. Root-only before-state backup
and installation receipt: `/var/lib/kuno-tool-prefill-fix-20261019`. No release
version, distributed package or frontend code changed.

Live browser acceptance uses the SAME public arithmetic request in the existing
test private chat, with Notes, Calculator and Images all enabled and low thinking.
Response version5/5 completes thinking -> actual calculator call -> tool result
-> final correct answer. All three groups remain enabled afterward. This is
actual application acceptance of the formerly failing full toolbox, not just a
mock/parser fixture. It does not execute every individual note/image operation.
No existing user conversation or global setting was changed. Browser proof of
only the public test result: `/tmp/kuno-vashti-all-tools-fixed-20261019.png`; no
private inference content is added to repository notes or Kuno persistence.

`native-toolbox-installed-final-check-20261019/{receipt,host-final}.json` confirms
the actual installed service identity/model mount, nonroot worker, swap/core0,
write_bytes0, denied unprivileged inspection, no OOM, all three service probes200,
admission active=false/paused=false/waiting0/recovery_required=false and zero
classified Xe/NVIDIA faults. The native model is intentionally retained on
Intel33,348,456,448B; NVIDIA compute0. Do not label this a fully idle GPU receipt;
full physical recovery was separately accepted in the staged and public tests.
The bounded sudo refresh loop is stopped at the checkpoint, and no test client
or staged service is left active.

**Vashti needs no code/API changes for this repair.** Keep sending the full tool
definitions and normal assistant/tool result history. No function is silently
dropped, no request is retried through Ollama, and no content is truncated.
The supported context remains512..32768; above32K, broader soak, individual image
fidelity and later performance/loader improvements remain separate work. No new
comparable throughput benchmark is claimed.


## Native Strata prompt optimization (2026-10-20)

This is internal Kuno work. Public Ollama routes/fields, KNT2/KNR2, thinking/tools,
context512..32768, owned model name and content-free persistence are unchanged.
**No Vashti connector/UI/configuration change is required.** The original DASLab
IQ3_S shards and Q2 MTP stay on Intel. No prompt truncation, shortened tools,
server prompt/KV cache, quantization substitution or NVIDIA dependency is added.

Selected source `c8db19de33b0585278be7380db14dae7323ef560`, frozen artifact
`strata-prefill-selected-short-20261019`, worker SHA256
`fb31e1244af754dea9585db25b82bc5dafd6dffdfbe512d47cb2c82e98781699`.
Commits5fbc3ca5 (router/vector dequant/matrix geometry),7e28bc5a (original timeline
submission batching),6dfaf5ef (coherent K64 paired shared buffers/vector loads),
076d0ec3/5aeb11e3 (bounded short input path plus real nonce/retained provenance),
7962fe68 (compiled-public warm check) andc8db19de (accepted matrix selection)
form the connected implementation. Additional register-first variantfaf5c6f6
regresses and is not selected. All original owner/fault/erasure and IEEE/reference
checks remain. [Full investigation](kuno-prefill-strata-gap-investigation.md).

Matched clean compiled-public4245-token warm prefix43.2055 ->24.9658s, **42.2%
shorter**. Exact staged toolbox4207/4293 prefixes29.5055/29.9196s versus previous
46.991/49.630s;35-token code prefix1.2901s versus previous4.639s. The quantized
short path produces a different complete501-token answer, so its full response
time is not comparable to the former362-token answer. This is no claim of faster
token generation, model loading, universal rate or local Strata parity. Startup
remains about106s plus initial GPU warmup. The selected short implementation is
MAX3/64 with bounded wiping host residuals and heads; full MAX8/no-head/device
carry, larger-context work and further large GEMM optimizations remain open.

Exact selected-binary acceptance under
`/media/jmicko/faststore/kuno-development/`:

- `prefill-selected-short-operators-numeric-20261019/receipt.json`:14 actual
  FP16/BF16 original-weight matrices +9 router/30 dequant valid/fault controls,
  unchanged oracles, full owner/tails, actual close and physical recovery.
- `prefill-selected-short-shader-validation-20261019.log`:402 ordinary shader
  and64 matrix configurations pass the pinned BF16-capable validator.
- `prefill-selected-short-clean-20261019/receipt.json`: cold45 then warm4245/4245,
  correct replies, actual erasure/retained rebind/close and physical recovery.
- `native-prefill-selected-short-stage-quality-check-20261019/receipt.json`:
  preload, ten-tool streamed thinking/call/result, total64/65/66 boundaries,
  same-worker mixed/repeated requests, identical complete501-token code answers.
  The generated merge function passes155 deterministic cases in each nonroot,
  resource-bounded function-only-AST child; text/source/hash is not persisted.
  Worker write_bytes0, denied inspection, graceful unload, ordinary Ollama after
  native and actual recovery all pass; no driver faults or reboot.

The initial stage's obsolete362-token assertion failed when the changed answer
hit its384-output cap. Its failed log and recovery remain recorded; it is not
counted as acceptance. The new fixed-public fixture allows1024 outputs, requires
EOS/no truncation, semantic correctness and identical repeats. No GPU arithmetic
tolerance or production guard was relaxed.

Official installation preserves the exact image-enabled gateway SHA256
`d3ed07f473b28bbb46596a619349d933edd6c166b7af10dd42f9821e4ba3048c`, complete
configuration and trusted root-enrolled model identities. Backup/receipt/official
rollback: `/var/lib/kuno-prefill-selected-short-quality-20261019`.
Final installed acceptance also passes:
`native-prefill-selected-short-quality-installed-public-check-20261019/receipt.json`
and `native-prefill-selected-short-quality-installed-final-check-20261019/{receipt,host-final}.json`.
Two complete identical501-token answers each pass155 merge cases. Cold load
105.8716s, cold prefix9.5601s; warm prefix1.4305s,501 outputs/15.2196s (32.92 output
tokens/s), full21.3211s. The shorter prefix is69.2% below the old4.639s, but different
response length prevents a full-request speed comparison. Nonroot/mount, swap/core0,
worker write_bytes0, inspection denial, zero OOM/driver faults, all services200 and
normal admission pass. Intel model weights remain warm (33,314,902,016B), NVIDIA
compute0. No test client/staged service/task sudo-refresh loop remains. No reset
or reboot. No fresh Vashti/browser/image test, package/version change or release.
Preserve this source selection and gateway image features in later packaging.

## Native prompt-processing optimization (2026-10-19)

Historical baseline, superseded by the2026-10-20 selection above.

This is internal Kuno work; the Ollama-compatible public contract, KNT2/KNR2,
thinking/tool support, context512..32768, model naming and privacy policy above
are unchanged. Vashti needs no source/API change. Full tools remain available;
there is no cross-request content cache. This pass does not repeat browser QA or
claim every individual image/note operation works.

Exact code, rejected candidates, timing definitions, frozen hashes, tests and
protected recovery receipts are in
[kuno-prompt-processing-profile.md](kuno-prompt-processing-profile.md).
Changes:342970e2 public-only profiling,4a33422e bounded expert member grids,
f9471ce8 attention subdivisions,ef886df5 same-stream DMA/kernel graph scheduling,
23454d39/60c20e36 direct original-buffer matrix tiles. The scheduling change keeps
original access intervals/dependencies and actual final completion/fault checks;
there is no early erasure or completion credit. The K64 part ofef886df5 was
replaced, and the final blocked experimental variantd06c03d4 failed its small
numeric test; selectiona37db7da restores the matrix source exactly to60c20e36.
Do not install the experimental K64/blocked artifacts or weaken their oracles.

Matched uninstrumented public4,245-token warm prompt55.821 ->43.205s (22.6% shorter),
not a universal chat speedup. Source60c20e36 / worker SHA256
`1db4664d9530df7b079aea0c23c6e26a74fc450598784e6ab68388e62032cc8a` passes eight
actual original-weight FP16/BF16 matrix cases, unchanged IEEE tolerance and all
owner/tail/erasure checks. 27 prompt tests,4 tests of the original vendored
scheduler (including cross-stream submission), strict worker Clippy,401 ordinary
shader variants and32 matrix configurations pass. The pinned matrix-capable
validator is required because the distribution tool cannot parse BF16 capability5116.

`native-prefill-stage-check-20261019` passes preload, ten public tool schemas,
streamed thinking/weather call, tool-result final answer, identical repeated362-token
code replies on the same worker, write_bytes0, inspection denied, graceful unload,
ordinary Ollama afterward and physical recovery. Toolbox prompts4,207/4,293 take
46.991/49.630s: no matched old-toolbox timing in this pass. The shorter code warm
decode is33.70 tokens/s; do not mix it with long-prefix timings.

The official transaction preserves the unchanged image-enabled gateway SHA256
`d3ed07f473b28bbb46596a619349d933edd6c166b7af10dd42f9821e4ba3048c`, ports, complete
configuration and root-enrolled original model identities. Root-only backup and
receipt: `/var/lib/kuno-prompt-processing-20261019`. `native-prefill-installed-public-check-20261019/receipt.json` passes identical
362-token code replies twice through the installed listener. Cold load102.475s;
warm prefix4.639s, decode10.899s (33.21 output tokens/s), full request18.758s.
`native-prefill-installed-final-check-20261019/{receipt,host-final}.json` confirms
nonroot worker/model mount, swap/core0, write_bytes0, inspection denied, zero OOM
and driver faults, all services200 and normal admission. Intel native weights
intentionally remain warm (33,314,902,016B total device usage); NVIDIA compute0.
No test client, staged service or sudo refresh loop remains active. This is
backend API acceptance; no new live Vashti browser test is claimed. No
version/package, frontend or release change.

The proposed Vashti-owned prefix snapshot extension is **not implemented**.
Its approximate raw sizes, remaining format/privacy/validation requirements and
network tradeoffs are in the profile document. A later shared contract and
separate Kuno/Vashti acceptance are required before adding that optional request.

## Native Strata parity continuation installed — 2026-10-20

Selected source0771a726 / worker SHA256
`ccdaf8861819ec3dfb6e92208904dba5f989d79f49d176289f69e265be2e8057`
is installed through unchanged gateway36626d30 / SHAd3ed07f4. The official
transaction preserves the entire configuration and root-enrolled identities;
backup/rollback `/var/lib/kuno-strata-parity-native-20261020` restoresfb31e124.
This is an internal implementation change: API, KNT2/KNR2, settings, model name,
context512..32768, thinking/tool rendering, keep-alive, Ollama compatibility and
private content-free behavior are unchanged. No Vashti integration is required.

The selected complete pipeline combines checked MAX8 headless known-prefix
processing, scalar-state GDN, QSA specialization, original native matrices,
bounded rolling submission and borrowed immutable layer bindings. All97 actual
operator controls pass. The explicit public terminal-prefix failure test proves
real recovery-fence waits on two queues, protected terminal process exit and
physical GPU recovery, then a fresh worker succeeds. No reset/reboot.
Protected staged thinking/ten-tool call/result,64/65/66 boundaries,16 MAX8 tails,
two complete semantic repeats (155 cases each), privacy, ordinary Ollama and
physical recovery pass. See the active coordination guide for exact receipts.

Matched clean4245-token repeated prefix24.9658 ->12.4373s; first long14.1743s.
The fresh matched Strata reference remains5.3515s, so parity is still pending.
Actual installed public repeated code/identity/mount checks pass: cold load
100.3023s and first short prefix9.9475s including JIT; warm35-token prefix1.1636s,
501 outputs/15.2889s (~32.77 tokens/s), total19.7015s. This is no new decode gain.
Actual nonroot worker, swap/core0, write_bytes0, denied inspection, zero OOM/new
GPU faults and healthy services/admission are confirmed. Experimental matrix
variants remain uninstalled; no new Vashti browser/image QA or release is claimed.


## Installed native32 prompt improvement — 2026-10-20

Worker3fffc266 / SHAeac78ce0 replaces0771a726 / SHAccdaf886 through the unchanged
image-enabled gateway36626d30 / SHAd3ed07f4. Backup/official restore transaction:
`/var/lib/kuno-strata-parity-native-wide32-20261020`. Full configuration and
root-enrolled original model identities remain unchanged. Exact artifacts and
acceptance are in [the active backend guide](kuno-strata-parity-goal.md#accepted-native32-installation).

Clean repeated4245-token prompt12.4373 ->11.1474s, first14.1743 ->12.9623s.
All103 actual numerical controls,402/144 offline layouts, terminal recovery, full
tools/MAX8/semantic/privacy and installed-service gates pass. Installed public
semantic repeats remain identical/complete501 outputs and155 cases each. Warm
35-token prefix1.0908s and15.0813s decode (~33.22 output tokens/s) are no new
decode gain. Cold load101.0112s remains slow. Nonroot/model mount, swap/core0,
write_bytes0, denied inspection, zero OOM/driver faults and healthy services pass.

There is no API, wire, setting, model-name or Vashti change. No browser/image
acceptance or release is claimed. The later paired-QK/descriptor-batch candidate
is not installed; it keeps the original numerical, owner and recovery gates.
