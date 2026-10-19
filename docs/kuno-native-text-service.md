# Native Intel text service preview

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

## Integration checkpoint (2026-10-19 UTC, before installation)

- Worktree `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
  `codex/strata-rust-port`. Do not edit the canonical Vashti checkout.
- Initial adapter7f54777a, protected Vulkan/pipe correction39c3e8e1,
  transactional enrollment1c4cf520. Additional actual-Ollama-CLI defaults handling
  is being tested; it accepts empty serialized suffix/template/options defaults
  while continuing to reject real unsupported features.
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
- Initial Kuno CPU run:222 passed across15 suites,7 hardware tests ignored.
  Latest targeted lib/native/installer tests and strict Clippy pass; run the final
  broad Kuno suite again after the final parser/installer edits are frozen.
- Image-enabled replacement build preserves CUDA/Vulkan and the existing
  managed SYCL bundle. Fresh pinned source/build under this worktree's
  `target/kuno-sd-{source,build}-88411ef-b51e05e8aaf4e67e[-cuda]`.
  Use default `GGML_CUDA_NO_VMM=OFF`, matching the installed binary. An accidental
  ON preparation failed its CUDA-driver link and was not installed; restored OFF
  rebuild succeeds, including embedding, attention and36 loader CPU cases.
  Gateway Cargo target: `/media/jmicko/faststore/kuno-development/native-install-target-20261019`.
  Rebuild the final gateway after the last commit before freezing/copying it.
- Nothing is installed yet. Next: protected staged HTTP gateway with disjoint
  listener/control/worker ports, mixed real requests, installed Ollama CLI shape,
  root-only worker IO check, graceful unload and ordinary Ollama routing. Then
  use official `upgrade --yes` and `configure-native-text --settings FILE --yes`.
  Do not manually edit installed config/binary and invalidate deployment hashes.
  Enrollment copies root-controlled source worker/settings into the existing
  reversible deployment and preserves them across later gateway/image upgrades.
- Model settings for installation must reference the existing read-only model
  mount `/var/lib/kuno/models/.kuno-import/...`. Hardlink the existing889MB MTP
  into that store on the same filesystem, not a second83GB target download.
  Test scripts/settings/frozen artifacts and classified receipts live under
  `/media/jmicko/faststore/kuno-development`; they contain fixed public fixtures
  only. Do not save generated text or private caller bodies in the next tests.
