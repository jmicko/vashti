# Native Intel text service preview

## Active extension contract (2026-10-19, implementation in progress)

The user requested thinking, tool use, faster cold load and configurable context.
The installed checkpoint below remains the narrow preview until replaced and
validated. New Ollama chat behavior will follow the actual embedded Qwen template:
`think` accepts booleans or low/medium/high/xhigh (absent remains false), reasoning streams separately
from content, and XML-style function calls translate to structured Ollama
`message.tool_calls` with typed argument objects. Assistant thinking/tool-call
history and role=tool results round-trip; Kuno never executes those tools.
Capabilities are advertised only when this adapter is connected and tested.
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
sidecar under the existing read-only model-store mount. The worker hashes the
complete files at process startup and validates the original inventory and graph.
It keeps their backing alive until the Session closes. Startup hashing is included
in the service's first reported load duration and can be substantial.

Discovery extends `/api/tags` and `/v1/models`, preserving original entries and
extra JSON fields. `/api/show` describes the owned entry, advertising completion
only. `/api/ps` includes the native entry once resident; unknown actual VRAM size
is omitted rather than reported as CPU residency. The digest identifies the
original first shard, not a hash of user content. No Vashti protocol types change.

## Initial request support

- `/api/chat`: text `system`, `user`, `assistant` messages only.
- `/api/generate`: ordinary prompt/system rendering or explicit `raw: true`.
- Streaming NDJSON and `stream: false`, UTF-8 across token boundaries, actual
  output limits, EOS/length reasons and token/timing fields.
- Non-thinking rendering matches the accepted Qwen public fixture template.
  `think: false` is allowed; `think: true` is explicitly unsupported.
- Options: `temperature` (default0), `top_k` (20), `top_p` (.95), `min_p` (0),
  unsigned `seed` (0), `num_predict` (1024), and `num_ctx` only when32768.
  Input tokens plus output budget must leave the original eight-token headroom.
- `keep_alive` absent or -1 retains the model until competing work or shutdown.
  `keep_alive: 0` unloads after completion; an empty request with0 explicitly
  unloads. Other durations are not implemented in this initial contract.
- An empty preload request with retention loads the real model and performs one
  fixed, discarded one-token warmup, then fully erases its request state before
  returning `done_reason: load` with zero delivered tokens. It grants no shortcut
  around subsequent request rebind/erasure checks.

Unknown options, tools, tool messages, images, format constraints, thinking,
embeddings, OpenAI inference routes and model-management operations for this
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

The worker uses inherited pipes, with a bounded private binary ABI (`KNT1`/`KNR1`),
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

No connector/UI implementation belongs to this task. A compatible consumer can
use the owned name through its existing Ollama text API with tools and thinking
disabled and omit `num_ctx` or set32768. Do not advertise unsupported capabilities.
The normal Ollama name remains available for richer features. Default-native
bindings of existing names, transparent pre-execution capability fallback,
OpenAI/tool/thinking adapters, configurable contexts and broader soak acceptance
remain later work; this preview does not claim them complete.

Installation must preserve the existing CUDA/Vulkan image-enabled Kuno binary
features and managed SYCL runtime. A gateway-only replacement is not acceptable.
Before replacing live services, verify the frozen native worker with a protected
framed caller, then verify the staged gateway's catalog, real streamed/nonstreamed
mixed prompts, preload, unload, subsequent ordinary Ollama inference, recovery
and absence of content files. Keep an exact binary/config rollback.

## Integration checkpoint (2026-10-19 UTC)

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

## Installed acceptance and usage (2026-10-19 UTC)

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

## Extension CPU checkpoint (2026-10-19, not installed)

The installed binary/worker remain the previous accepted preview. This source
extension is not live acceptance or a new performance result. No GPU inference,
service restart, driver operation or installation was done for this extension.
The shared shell is waiting at sudo-v; the requested root credential is pending.

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
- `/api/show` in the candidate advertises completion/thinking/tools; do not
  install this advertisement without actual model round-trip validation.
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
arguments, malformed/incomplete calls and unknown-name rejection, history/tool-result rendering,
stream versus collected response shape, context frame round trips/version
mismatch, artifact modification/replacement and rejection of untrusted cache
identities. Relevant root/worker strict Clippy and selected worker compile checks
pass. Final root suite:234 passed,7 hardware tests ignored (including the
updated capability fixture); logs are `native-chat-final-all-tests-20261019.log`,
`native-chat-clippy-20261019.log` and `native-chat-worker-clippy-20261019.log`
under the development artifact directory. Build artifacts are recorded after
freezing the source.
Existing inherited untracked benchmarks/text/__pycache__ is preserved.

Next protected validation before installing: thinking true/false/effort on real
mixed prompts; actual tool request -> structured call -> synthetic tool result ->
final answer; stream/nonstream parity;512/4096/32768 budget rejection followed by
successful reuse of the same worker; two cold worker starts through root-attested
settings to measure load savings; zero worker writes, swap/core protection,
graceful close and physical recovery; ordinary Ollama/image coexistence. Then
finish larger runtime capacities with u32 graph keys and real allocation/VRAM
checks. Keep the old accepted install if any of these gates fails. No release.
