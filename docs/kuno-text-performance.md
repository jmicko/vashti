# Native Rust text inference: implementation and benchmark receipts

Target: the borrowed full Qwen3.8-Flash-Next Q4_K_M artifact documented in
[the research plan](kuno-text-engine-research.md#reuse-the-downloaded-ollama-model).
Native text inference is under development; these measurements are **Ollama
reference results, not Kuno native results**.

## Updated Ollama baseline, 2026-10-05

The user updated Ollama and requested a fixed-prompt/seed cold and warm baseline
before Rust implementation and optimization. Exact public synthetic prompts and
options are in [the fixture](../apps/kuno/benchmarks/text/qwen38-baseline.json);
allowlisted raw numeric results are in
[the receipt](../apps/kuno/benchmarks/text/receipts/ollama-0.35.1-20261005.json).
Seed is **424242**, temperature 0, repeat penalty 1, context 4096, 128 requested
output tokens, batch 128 and 16 requested CPU threads. Raw generation uses an
explicit text-only prompt wrapper, without automatic templating or MTP.

The tested daemon reports **Ollama 0.35.1**. The direct managed upstream listener
was used while Kuno admission was paused and inactive; Kuno's already-running
older managed worker was not mistaken for the updated engine. `ollama stop`
unloaded the prior resident before testing. No service restart or driver reset
was required. The first request began with no loaded models. OS disk page cache
was not flushed, so "cold" means model-unloaded, not cold storage.

| Run | Client total | Ollama load | First content | Uncached prompt / total | Prefill | Decode | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 100.346 s | 41.500 s | 75.039 s | 75 / 75 | 2.236 tokens/s | 5.058 tokens/s | 128 tokens |
| Warm, same prompt | 16.690 s | 0.0045 s | 0.635 s | 4 / 75 | 8.981 tokens/s | 7.973 tokens/s | 128 tokens |
| Warm, changed system prefix | 25.935 s | 0.0022 s | 9.096 s | 80 / 80 | 8.799 tokens/s | 7.602 tokens/s | 128 tokens |

Every request ended normally at its output limit. Cold and repeated-prompt
output digests match. The warm repeat reused 71 prompt tokens; the changed-prefix
request reports zero cached prompt tokens. Do not describe the repeated-prompt
time as full fresh-prefill performance or compare it silently to a native worker
that clears all prefix state at idle. Only one cold/warm sequence was measured;
this is a baseline receipt, not a statistical speed or overnight stability claim.

Although the initial command label guessed CUDA, actual automatic placement was
**B70 via Vulkan**. The active runner mapped `libggml-vulkan` and the Intel Vulkan
driver, with DRM descriptors identifying `xe` PCI `0000:2d:00.0`; no GGML CUDA
backend was mapped. B70 fdinfo reported 29,060,212 KiB resident VRAM. Ollama's
own `/api/ps` counters reported 28,646,991,461 VRAM bytes and runtime size
63,879,889,549 bytes. These counters are not the catalog's 120 GB artifact size,
and they do not imply all weights fit on GPU. NVIDIA memory was 551 MiB during
loading, consistent with the display footprint rather than a second model cache.

The operator temporarily capped the upstream cgroup at 52 GiB host memory,
preserving `MemorySwapMax=0` and `LimitCORE=0`. The benchmark client also ran with
swap and core dumps disabled. A late sample measured 53,026,336,768 cgroup bytes
(including reclaimable file pages), not a continuously measured peak. No cgroup
OOM/max events occurred. The original unlimited `MemoryMax` was restored after
unloading. This limit is part of the benchmark configuration and must be matched
or explicitly distinguished in later comparisons.

Only the public synthetic fixture, numeric counters and synthetic output
digests are retained. Output text/thinking, raw backend errors and private
requests are not retained. Production Kuno still forbids prompt/seed/output
logging; this explicit operator benchmark does not introduce a request logger.

## Benchmark client validation

`python3 -m unittest discover -s apps/kuno/scripts -p test_benchmark_text.py -v`
passes three tests: chunk-independent synthetic output hashes, exclusion of
cached prompt tokens from prefill rates, and failure without replay when the
stream lacks a completion record. Generation imposes no wall-clock/idle timeout.
The receipt is tied to the exact manifest digest and refuses a cold start with
already loaded models or overwriting an existing receipt.

## Rust engine milestone

The user authorized implementation after the research checkpoint. Rust must own
the loader, graph, operator/state execution and sampling. A whole C++ graph
wrapper is not the requested engine. Start with safe Rust scalar operations and
correct raw inference; only then optimize measured bottlenecks. Rust-authored
GPU kernels through crates are a candidate; driver/compiler dependencies are
distinct from delegating the model graph to another engine.

The current development branch is `codex/kuno-native-text`, based on research
commit `e9de15a` and integrated beta.3 `ae3e70d`. Work is isolated from Vashti.
Separate loader and operator worktrees share that baseline. No native text model
is advertised or routed in the installed gateway. Future receipts must identify
exact source commits, operator/graph acceptance, cleanup protections, measured
memory and matched performance before default-native routing is enabled.
