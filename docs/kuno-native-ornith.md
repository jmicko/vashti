# Fully resident Ornith speed comparison

User-authorized direction, 2026-10-07: isolate native execution speed with the
already downloaded `ornith:35b` before returning to Flash-Next offloading. The
immediate target is now approximately80 decode tokens/s, matching the observed
fully resident Ollama result. Operator microbenchmarks do not establish this.

## Scope

The inspected original Ollama blob is
`6540411d9e4ca638983e905959188e1be74db7ff62151c69d71a6f2748584f68`,
21,166,757,664 bytes. It is `qwen35moe`, not `qwen4exp`: 40 layers,
2048 hidden width, 256 experts/top8, 512 expert width, GDN head dimension128
with16 key/32 value heads, and full attention every fourth layer with16
query/two KV heads of dimension256. All packed weights fit the32-GiB B70;
physical residency must still be measured. No model copy or download is needed.

The first adapter is a Rust/WGPU Vulkan laboratory using the existing Rust
GGUF reader, tokenizer and original packed quantization. Construct and retain
all weight buffers and dispatch bindings once. Keep routing, recurrent state,
KV state and activations on GPU; encode a complete token without intermediate
host routing reads. Only final token selection crosses to the host. Ordinary
attention and GDN must both match their architecture; in particular Qwen3.5's
GDN output uses SiLU gating, whereas the current Qwen4exp reference uses its
own sigmoid gate. This is not a foreign inference-engine wrapper.

Formula/layout reference: llama.cpp `b11232`, `src/models/qwen35moe.cpp`
(MIT). Existing Rust packed decoders retain their existing GGML provenance.
Text-only inference is the initial scope: no multimodal encoder, MTP, sampling
API, private worker, registration, installed binary or frontend change.

## Evidence

Use the same fixed public rainwater prompt, raw chat prefix, greedy settings,
seed424242 and128-output cap for Ollama and native. Record cold model-load and
warm fresh-sequence durations separately; warm weights do not mean reused KV.
Inspect actual GPU memory and Ollama placement. No repeated Flash-Next baseline
is needed. The new architecture requires its own initial matched baseline.

The lab runs with swap and core dumps disabled. Private host buffers are
zeroizing; GPU activation, routing, convolution, recurrent, KV and result
buffers are explicitly cleared before release. Public synthetic token IDs and
numeric timings may enter receipts; no real prompts, output text or request
derived expert histogram is persisted. Normal cleanup remains required, while
extended cancellation/API integration work is deferred at the user's request.

Source tests, actual numerical/model acceptance and performance comparisons
remain distinct. Completed fixed-public speed receipts are recorded below.
This laboratory does not expose ordinary inference. Required Vashti changes: none.

## First device outcome and pause

`c174aa9`'s frozen `6917d349…` normal release binary completes the40-layer
fully resident B70 graph. The long fixed public fixture observes13.9408/13.9184
decode tokens/s versus Ollama's83.3857 warm report. Only the first14 of128
generated IDs agree; numerical parity remains unaccepted. Native fresh
sequences are internally deterministic. See the
[current checkpoint](kuno-native-text-checkpoint.md#fully-resident-ornith-checkpoint-2026-10-07)
and its numeric aggregate for exact counters, artifact identities and limits.

This first result is superseded by the speed acceptance below. The previous
pause ended when the user returned both GPUs for speed work.

## Resident speed acceptance, 2026-10-08

The explicit vector packed-word arithmetic plus cached parallel expert selection
reaches85.35/85.38 decode tokens/s on two fresh75/128 sequences. Longer runs
stop at EOS:82.92/83.04 tokens/s for294 following steps; an alternate fixed
public prompt reaches82.63/82.66 for335 steps. The shared-kernel refactor recheck
reaches86.00/85.27 with identical128 output IDs. These complete-model runs
meet the approximately80-token/s resident target. The original scalar reference
remains available and all128 native generated IDs agree with it.

Select explicitly with `KUNO_LAB_ORNITH_VECTOR_DOT=1`,
`KUNO_LAB_ORNITH_PARALLEL_ROUTER=1` and
`KUNO_LAB_ORNITH_TRUSTED_BOUNDS=1`. Defaults remain unchanged. The last flag
removes runtime shader bounds checks only for the validated fixed geometry;
ordinary WGSL validation remains. Keeping bounds checks measures78.20/77.48.
The earlier block-only candidate measures15.44/15.44 and is not selected.

Independent decoded-original-row probes cover54 projection cases, with maximum
absolute error2.421438694e-7 and tolerance5e-4+5e-5*abs(reference). Six router
cases check scores, ties, normalized weights and owning intervals. Instrumented
timestamps identify the original scalar router as the main issue:42.064ms per
token across40 layers becomes0.419ms. Timestamp instrumentation is excluded
from the throughput figures above.

Teacher-forcing all128 public Ollama tokens gives121 top1 matches; the other
seven expected IDs are rank2. Scalar and fast native choices/ranks agree at
every position. Free-running native and Ollama still share only the first14
IDs, so bitwise Ollama parity is not claimed. Rates are approximately matched
throughput, not a reliable2% speed win: continuation IDs, counter definitions
and warm prefix reuse differ. Native4-second model loading follows file warming
and is not a matched cold-load comparison.

Source commits `000bdbe`, `fbddf49`, `3d8973e`, `5a93d1b`, `26829f6` and
`4a06b83` implement probes, profiling, explicit bounds selection, packed vectors,
parallel routing, EOS/teacher diagnostics and shared kernels. The
[aggregate receipt](../apps/kuno/benchmarks/text/receipts/ornith-resident-speed-20261008.json)
pins source/artifact identities and original numeric receipts. Offline shader
tests and strict all-bin Clippy pass; all accepted jobs enforce swap0/core0 and
explicit normal private-buffer clearing. This is not an installed engine,
arbitrary-request, multimodal, quality/soak or cancellation acceptance.

## Transfer to Flash-Next

The user selected approximately10 whole-model Flash-Next tokens/s as the next
milestone before experimental expert-placement strategies. Shared packed
arithmetic and512/top10 routing pass fixed-public original Next slabs, including
Q4 gate/up, original Q8 and Q5 down weights, F32 routing and Q6 vocabulary head.
The head's vector median is4.59 times faster than scalar in this isolated test.
This port is not connected to the complete Next model yet. Its last complete
measured path remains1.691 tokens/s; resident Ornith speed does not establish
Flash-Next speed. No further Ollama baseline, API or frontend work is required.
