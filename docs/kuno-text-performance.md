# Native Rust text inference: implementation and benchmark receipts

Target: the borrowed full Qwen3.8-Flash-Next Q4_K_M artifact documented in
[the research plan](kuno-text-engine-research.md#reuse-the-downloaded-ollama-model).
Native text inference is under development. The initial baseline is updated
Ollama; later receipts cover bounded native correctness experiments. **A native
speedup has not yet been demonstrated.**

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

## Rust correctness and GPU laboratory, 2026-10-05

The experimental `kuno-text` crate now owns bounded GGUF access, all seven
mixed-format decoders, Qwen35 byte BPE, the complete sequential Qwen4exp trunk,
per-sequence GDN/QSA/PLE state, and greedy token selection. It reads the existing
Ollama blob; no weights are downloaded, copied or converted. Exact formulas,
permutations and tensor geometry are in [the graph contract](kuno-native-text-graph.md).
The code is not linked into the installed gateway or its model list.

Two **fixed public synthetic** smoke sequences match updated Ollama 0.35.1's
output exactly: one raw input token followed by four output tokens, and a
15-token wrapped chat followed by eight output tokens. The native CPU result
for the chat is a coherent greeting. This proves a small end-to-end Rust
inference path, not long-context, every sampling option, tool, vision or full
128-token baseline acceptance. Prompt wrappers, seed 424242, greedy IDs, output
digests and numeric receipts are in [the smoke receipt](../apps/kuno/benchmarks/text/receipts/rust-smoke-20261005.json).
The scalar reference remains slower than Ollama; its timings are correctness
probe observations, not a matched benchmark matrix.

The separate [GPU laboratory](../apps/kuno/benchmarks/native-text/README.md)
keeps experimental CubeCL/wgpu dependencies out of the installed Kuno release.
Rust-authored Vulkan kernels passed independent packed decoding/dot checks for
F32, F16, BF16, Q4_K, Q5_0, Q6_K and Q8_0 on both B70 and RTX3060. Maximum
observed error was about 1.2e-7 on these bounded actual-model slices. The complete
Rust graph using those GPU matrix kernels also reproduced the four-token raw
smoke on B70. This is GPU execution under a Rust-owned graph, without a foreign
model engine or C/CUDA/SYCL kernel sources. Other graph operators still run in
Rust on CPU.

The first small 4 GiB GPU cache recorded **zero cache hits**: its cyclic working
set exceeded capacity, causing every matrix to be uploaded again. An 8 GiB
operator cache reduced four-pass uploads from 18.45 GB to 6.82 GB, with 4,539
hits. The measured 27.488-second four-token computation spent 26.825 seconds in
matrix execution; 21.663 seconds was GPU execution/readback wait. This points to
coalesced parallel matrix reductions and reducing synchronization as the next
measured optimization. It is **not yet faster than Ollama**.

Validation at this checkpoint: 70 normal Rust tests, two explicit installed
artifact checks (including exact 75/80 baseline token counts), scalar full-graph
smokes, both-card packed probes, and all-target Clippy/format checks. A bounded
persistent CPU thread pool has bit-identical row results across all seven
formats and tested cancellation/error cleanup. Its full-graph performance is
not yet measured. Test clients ran with swap/core dumps disabled; no production
user content was used. Bounded cache keys/ranks are transient to the public
experiment. Persistent private worker containment, all-failure GPU cleanup,
crash/cancellation/concurrency, native routing and complete matched baseline
remain required before API exposure. No service restart, driver reset, install,
release or Vashti change was performed.

## Cooperative reductions, expert banks and GDN laboratory

The next Rust-only prototype retains the independent scalar implementation and
adds 64-lane packed dot reductions, full ten-expert gate/up/down batching, and a
transactional GPU GDN operator. The CPU still owns committed recurrent state;
old/next GPU state buffers are distinct, validated and cleared before commit.
Both devices pass seven-step tiny/actual-dimension GDN state/output checks, with
maximum state differences below 6e-8. Full selected-expert tests pass for both
Q5_0 and Q8_0 down weights, including repeated hits and protected-slot eviction;
maximum observed output difference is below 1.4e-8. These are bounded fixtures,
not general floating-point equivalence or long-context acceptance.

The fixed public four-token graph result still matches CPU/Ollama on both cards.
The selected B70 experiment records 6.844 s prefill and 5.040 s for the three
following model steps. The 3060 records 9.580 s and 24.339 s. Receipts contain
numeric counters, exact public IDs, source-file/binary hashes and fixture scope:
[B70](../apps/kuno/benchmarks/text/receipts/rust-batched-bang-intel-gdn-20261005.json),
[3060](../apps/kuno/benchmarks/text/receipts/rust-batched-bang-nvidia-20261005.json).
They are **not a speedup over Ollama** or the complete 128-token matched baseline.
The three following steps still incur new expert uploads; "same process" is not
proof of fully warm weights. GPU GDN also does not establish an end-to-end gain.

Measurements explain two remaining problems. On B70, dense execution consumes
2.315 s and routed experts 2.027 s of the following three steps; GDN costs
0.620 s. These top-level categories are separate, but read/upload/wait counters
are nested and must not be added to them. On 3060, the 3 GiB static cache is too
small: the following steps upload 11.332 GB, versus B70's 2.209 GB. The next
placement experiment will reserve fewer expert slots and a larger static cache.
Each expert bank still protects all ten selected experts before eviction; no
expert approximation, pruning or model-file conversion is used.

The laboratory now pins every cache to one inspected GGUF descriptor, validates
rank/layout before dispatch, accounts for power-of-two allocation pages, checks
post-allocation reservation, and clears host activation readbacks. Host planning
is transactional; partial-upload failures discard the entire affected bank.
Unchecked shader launches are narrowly confined to validated packed geometry;
scalar and checked device probes remain the oracle. Actual private-worker
containment, all-failure cleanup, long sampling/context acceptance, native
catalog/routing and a full matched performance matrix remain outstanding.

At this checkpoint 70 ordinary core tests and 31 laboratory host tests pass,
as do all-target Clippy with warnings denied and formatting. Both-card public
probes/full raw smoke use swap/core-disabled bounded operator units. No gateway
API, installed services, Vashti files, public release, reset or reboot changed.


## Transfer reductions and first full public native fixture

Backend branch `codex/kuno-native-text`, following `f8b2d2d` and read-only bound
plan `2467ac7`. The installed gateway and Vashti remain unchanged. The lab now
queues every matrix output tile into one output/readback, provides an enqueue
interface for resident activations, keeps tiny projections in a bounded Rust
AVX2/scalar model-weight cache, and uses packed block decoding where measured.
None of these implementations call a foreign model engine.

The 3060 placement experiment moves from 3 GiB static/32 expert slots to
6 GiB/16 slots: three following raw-fixture steps improve 24.339 to 5.511 s,
with original-weight uploads falling from 11.332 to 2.261 GB. B70's combined
matrix readback improves the corresponding observation from 5.040 to 3.933 s.
Tiny CPU projections reduce one later B70 observation to 3.725 s. These are
independent short runs with expert misses, not statistical whole-model wins.
Exact public IDs still match. The rebalance binary hash was not captured before
rebuilding; its receipt explicitly records that limitation.

Cached ABBA operator tests use ten samples per variant and retain all samples.
The large output matrix block kernel roughly halves median time on both cards
(about 10 to 4.9 ms on B70 and 17.8 to 8.8 ms on 3060), with zero observed output
difference. Smaller HC projections do not consistently benefit, so automatic
selection keeps their generic kernel. The full ten-expert block gate/up kernel
also improves cached tests for both mixed down formats on both cards, with zero
observed difference between variants. These microbenchmarks do not prove the
same gain for a full request, especially when experts must be loaded.

The first full Rust GPU fixture completes **75 prompt tokens and 128 output
tokens**, seed 424242, greedy temperature zero, original borrowed artifact. On
B70, model binding is 0.330 s, sequential prefill 147.900 s and the following
127 decode steps/128 output selections 219.925 s. The unit runs 368.640 s,
with a live observed host peak 8,854,532,096 bytes and zero cgroup swap. The
52 GiB client limit and disabled core dumps match baseline containment. OS page
cache is not flushed. GPU initialization and lazy model uploads are not included
in the binding time; it is not equivalent to Ollama's `load_duration`.

The fixed response is coherent and follows the Ollama response closely; wording
diverges near the end. Its output digest is
`a1f70f78093e51fbd523b71a9a1f824c4e02fdf01688241d062d7dd8ea7dddbb`.
An independent updated-Ollama repeat retains the original response digest
`03095da322ed258d16f7548365a10fe15b5f3a234879d7fd1d01e79577ce1858`.
This is an initial longer-sequence check, not comprehensive quality or bit-exact
logit parity. Native sequencing remains token-at-a-time, with allocated context
203 versus Ollama's configured 4096 (visible tokens are equal). Native has no
cross-request prefix cache. No complete native cold/warm/changed-prefix matrix
is claimed, and concurrent source compilation makes this first long run a
correctness/profile receipt rather than a frozen compiler-idle speed trial.

**Native is still substantially slower than Ollama.** The original warm
Ollama baseline records 7.973 tokens/s and 16.690 s wall for the repeated request
(with 71 cached prompt tokens); its changed-prefix run records 7.602 tokens/s,
no cached prompt tokens, and 25.935 s wall. Do not compare cached prefill with
native's full prefill as if they perform equal work. In native's following
steps, dense matrix time is 103.620 s, routed MoE 80.368 s and GDN 29.264 s.
Read/upload/wait counters are nested. Recurrent-state round trips and hundreds
of intermediate readback fences remain the next measured target.

Receipts are in `apps/kuno/benchmarks/text/receipts/rust-*-20261005.json`;
`rust-rainwater-intel-20261005.json` holds fixed public IDs, counters, options,
binary identity and measured memory. This operator-only executable cannot
accept arbitrary prompts. No production content logger or weight copy is added.

## Resident operator and ownership acceptance

The read-only `BoundGraph` exposes only successfully validated tensors, static
vectors, exact geometry, PLE hash and opaque binding identity. It rejects PLE
hash history incompatible with the fixed sixteen-row embedding. GPU buffers
belong to an epoch-checked private arena that charges and initializes complete
power-of-two owning pages, uses the required custom allocator, synchronizes
full-span erasure, and retires a failed context instead of reporting cleanup
success. CPU core state remains the independent oracle.

Both cards pass residual RMS/down activation/mix/original-residual combination
for tiny and actual dimensions. Full owning-page initialization/close erasure
is inspected before buffer release, including 8,193 logical words in a
16,384-word span, and reset invalidates old IDs. Both cards also pass seven-step
resident convolution, Q/K preparation, controls, recurrence and output gating,
for tiny-head and actual Qwen geometry. Recurrent-state maximum difference is
below 3e-8; maximum gated-output difference is below 2.4e-6. This checks the
resident GDN block, not a complete resident model graph or production worker.

Device testing caught a CubeCL 0.10 WGSL lowering bug in conditional-literal
success-flag assignments: generated branches were empty and flags always zero.
Explicit conditional stores fix it; generated-code regression checks and
both-card device oracle checks verify the correction. A failed status gate
rejected the fixture before state acceptance; no reset/reboot was performed.

The [residency and I/O audit](kuno-native-text-residency.md) identifies remaining
runtime-owned upload/readback/uniform copies and proposes explicit Rust cleanup
hooks. Arena-owned erasure is **not** proof that these copies are sanitized.
Persistent private-worker acceptance, whole-model resident execution, full
sampling/context/tool coverage and default-native/`ollama.` scheduled routing
remain required. No public API, consumer, installed service or release changes.

Current source checks: 74 ordinary core tests, 62 lab host/offline tests,
formatting and all-target Clippy with warnings denied. Operator units disable
swap/core dumps and use only authorized fixed public fixtures. Final transient
unit memory peaks can under-report, so the long run uses live sampled memory.

## Connected resident graph acceptance (public laboratory, 2026-10-05)

The fixed-fixture owner now joins GR, complete GDN, exact QSA candidate/cache
state and all ten routed experts plus the separate shared expert on the device.
Projected PLE/hash/embedding-row selection remains Rust host work. Full-token
fault acceptance precedes state/cursor commits; any partial failure erases and
poisons the entire sequence. Read-only binding and device/stream/arena identities
are checked. Immutable expert allocations cache their physical alias proof,
while epoch/extent/model/context checks still execute per call.

The following three-step decode observations use the same four expected public
IDs, original weights and fixed settings. They are individual short runs,
with lazy uploads/compilation, not paired statistical performance trials:

| B70 variant | Prefill ms | Following three decode steps ms |
| --- | ---: | ---: |
| Helper graph with CPU-owned state | 6917 | 3923 |
| Resident GR/GDN with host QSA/MoE | 6588 | 2676 |
| Resident all-ten MoE, host QSA | 11477 | 2343 |
| Resident QSA, public-weight wipe removed, cached alias proof | 5379 | 1497 |

RTX3060's corresponding final observation is7255/2948ms. B70's full resident
chat matches all eight public reference IDs:12571ms prefill for15 tokens,
3198ms for seven following decode steps. Earlier host-QSA/resident-MoE chat
was22739/6892ms. These comparisons establish progress within the Rust engine,
**not an Ollama performance win**. Exact receipts include binary SHA256; the
final connected fixture binary is
`2b9748b9979d8df9699fe8af138a714175fed1d3fdde8e18c29ef168f2ddb777`.

Public model bytes no longer receive volatile `Vec::zeroize()` after expert
upload; these are original public GGUF bytes and contain no activation, routing
or request content. Private CPU scratch, GPU activation pages, IDs/probabilities
and readbacks retain cleanup requirements. The warm three-step B70 profile
still uploads2,209,484,800 public bytes: expert upload/enqueue is930ms, router
fences97ms. These nested wall profiles are only compiled synthetic-fixture
instrumentation, not infrastructure logging or separate additive GPU timings.
Further bounded transfer batching/cache placement is required.

On both cards, cancellation before a step and after queued attention at layers0
(GDN) and3(QSA) erases all owned pages/host PLE state, rejects poisoned retry,
and permits a fresh sequence to reproduce the first expected token. The fixture
only asserts arena/host ownership: **runtime upload/readback/uniform staging
sanitization remains unaccepted**. Swap/core dumps are disabled in bounded12GiB
operator units; no production model worker, service/API or Vashti change occurs.

B70 QSA operator parity passes contexts1/4/5/2048/2049/2052, including exact
incomplete tails and the first complete-block drop. NVIDIA's long-position
query check fails at context2048/index0 (absolute difference0.00031375885,
allowed0.000115561445), while shorter positions pass. Shader trigonometric
accuracy is being replaced by exact public Rust-generated tables; this is not a
GPU crash or a reason to loosen acceptance. Long-context whole-model acceptance
is still required on both cards. Host/offline checks currently pass89 tests and
all-target Clippy with warnings denied; core remains74 tests.
