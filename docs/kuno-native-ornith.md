# Fully resident Ornith speed comparison

User-authorized direction, 2026-10-07: isolate native execution speed with the
already downloaded `ornith:35b` before returning to Flash-Next offloading. The
working 10 decode tokens/s target remains; neither it nor Ollama parity is
established by an operator microbenchmark.

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
remain distinct. This document is an implementation direction, not a completed
device or throughput receipt. Required Vashti changes: none.

## First device outcome and pause

`c174aa9`'s frozen `6917d349…` normal release binary completes the40-layer
fully resident B70 graph. The long fixed public fixture observes13.9408/13.9184
decode tokens/s versus Ollama's83.3857 warm report. Only the first14 of128
generated IDs agree; numerical parity remains unaccepted. Native fresh
sequences are internally deterministic. See the
[current checkpoint](kuno-native-text-checkpoint.md#fully-resident-ornith-checkpoint-2026-10-07)
and its numeric aggregate for exact counters, artifact identities and limits.

The subsequent `KUNO_LAB_ORNITH_BLOCK_DOT=1` Q4/Q6 grouped packed-dot candidate
is source-tested and disabled by default. It has no device/performance receipt.
The user requested a pause; no additional GPU tests may run until resumed.
GPU memory and ordinary inference admission are restored; Vashti remains
available. No installed engine, routing or API behavior changed.
