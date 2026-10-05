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

### Rotary accuracy correction verified

The NVIDIA large-angle failure is resolved by1MiB of public immutable sin/cos
constants, computed once in Rust for all4096 positions/all32 pairs using the
core's exact f32 angle and `sin_cos`. Each bound public model cache uploads two
bounded arrays once; positions remain runtime data, never shader constants.
Checked table bounds reject positions outside the supported text context.

Both cards pass the same six QSA contexts with unchanged tolerances. At2052,
maximum normalization error is9.54e-7, score error≤3.82e-6, output error≤7.46e-9.
Exact selection, candidate append and immutable committed prefix checks pass.
Receipts `rust-qsa-rust-tables-{intel,nvidia}-20261005.json` record binary
`3e9653dccb4c3c812f7f47d9b38ed95486e3c6dabe0d46a333050140ae7d843a`.
Ninety laboratory host/offline tests and strict all-target Clippy pass. This is
operator acceptance, not long-context whole-model or private-worker acceptance.

### Full resident public fixture

The rotary-corrected resident graph completes the same75-token prompt and128
output selections on B70:64.936s sequential prefill,52.086s for127 following
steps,0.975s bind/owner setup,118.489s transient unit wall. Its binary is the
same `3e9653d...` public-table acceptance build. The first GPU implementation
was147.900/219.925s; these remain exploratory profiles with overlapping host
builds, not an idle paired cold/warm matrix. All original weights are borrowed.
Ollama's warm changed-prefix result remains faster (25.935s wall,
7.602decode tokens/s). No cross-request native prefix cache is used.

Following steps still transfer50,682,982,400 original-weight bytes. Nested
profile time includes31.565s expert upload/enqueue,2.996s router fences,
5.843s attention enqueue/fences,14.008s weight reads and6.174s weight uploads.
The next experiment batches each layer's missing public gate/up/down weights
into one checked upload/scatter. Live observed host peak reaches11,169,583,104
bytes under52GiB limit, with zero cgroup swap/core dumps. Final unit output is
not used as the sole memory proof. The full connected fixture remains public
laboratory correctness/profile acceptance, not complete private-worker,
long-context quality, production API or deployment acceptance.


### Rejected expert-upload batching experiments (2026-10-05)

Legacy separate public weight uploads remain the default. A bounded concatenated
public packed source plus GPU scatter was numerically correct on both GPUs,
including all ten routed experts, Q5/Q8 down banks, hits and eviction. It was
slower for the fixed one-prompt/four-output bang fixture. No routing pruning,
quant conversion, private comptime values or model-file writes were used.

The 64-MiB ABBA comparison used frozen binary
`4407d714119d9ee1f77881fdff1d8c3fd13c4add1923aa5bed833fd234437ca7`.
B70 legacy prefill/decode ranges were 5.250–6.015 / 1.341–1.474 s; batched
ranges were 8.801–9.374 / 5.929–6.276 s. RTX3060 legacy ranges were
7.174–7.773 / 2.884–3.109 s; batched ranges were 10.779–10.848 /
7.588–7.666 s. Decode here means three following graph steps.

Bounded 4/8/16-MiB B70 windows used binary
`c72ccbd547860b066da61a8d11d5ed648bad66c7df170552cf2d96ce9879053c`: legacy
5.865 / 1.528 s, 4-MiB 5.953 / 1.568 s, 8-MiB 6.883 / 2.577 s,
16-MiB 9.386 / 3.579 s, then legacy 5.921 / 1.653 s.

A follow-up removed per-word division with public geometry specialization and
a three-dimensional scatter; all ten slots stay runtime uniforms. Both variants
passed both-card MoE parity but remained slower in the B70 paired fixture.
Binary `87259b75683f1c86824422db9fd76246ea5a834c812140ec6d44e45115a04b8d`,
8-MiB source windows: legacy 5.342 / 1.456 s, specialized1D 6.136 / 2.185 s,
division-free3D 6.090 / 2.198 s, repeated3D 6.683 / 2.870 s, repeated1D
7.569 / 2.397 s, then legacy 5.929 / 1.444 s. All four output IDs matched.
The data do not establish one sole cause; upload timing includes queued GPU
work, host concatenation and staging. Neither candidate is selected.

Exact numeric receipts are `rust-expert-batch-paired-20261005.json`,
`rust-expert-batch-windows-20261005.json`, and
`rust-expert-scatter-paired-20261005.json` under the public receipts directory.
Timing units used MemorySwapMax=0, LimitCORE=0, MemoryMax=12G and an explicit
180-second operator bound. Short unit MemoryPeak summaries are unreliable, so
these do not claim a measured host peak. No ordinary inference deadline,
installed service, API/default route or Vashti consumer changed.


Aligned F32 decoding now loads and bitcasts the original aligned u32 directly,
while preserving the 64-lane sum order and all other quantized formats. Both
GPUs pass original-artifact numeric probes and the four-token graph. The short
ABBA graph measurement does not establish a material speedup: B70 decode times
1.385/1.363/1.473/1.573 s, RTX3060 3.116/3.036/3.122/3.225 s (old/new/new/old).
`rust-f32-aligned-word-paired-20261005.json` records complete settings and both
binary hashes. The new binary also contains inactive private-cleanup guards;
this receipt accepts exact original-word decode, not an isolated speed claim.


### Owned runtime I/O and resident admission acceptance

The standalone laboratory pins/vendors exactly three CubeCL0.10.0 crates at
`7cf203735e095e640a2c03b2400d0faa03196bb4`; production dependencies and global
registry files are unchanged. Twenty-seven runtime ownership host fixtures,
128 native host/offline tests, 74 ordinary core tests (two installed checks
ignored), formatting and strict host/native Clippy pass. The 18-file composed
runtime patch applies to fresh pinned archives with all modified hashes matching.
Independent source reviews accept the limited preowned Wgpu/Arena path and
public-graph admission; they do not establish a production privacy guarantee.

Both GPUs pass fixed `transient-io-probe` using frozen binary
`9136b527e6634fe4394076b8bce297bb2a5a4aee6775385461fc418d5ee057e7`.
It covers full tracked upload/uniform/readback spans, pending future drop,
retained Bytes rejection before mutation, guarded host capacities, runtime
scalar parity, repeated audited cleanup, public cache preservation and complete
65536-byte main-span erasure. Every epoch's inspection counts match all full
owned spans, including audit stages. Swap/core dumps are disabled under 4GiB
host/180-second operator bounds. An initial explicit-zero-offset validation
failure is resolved and tested with the actual contiguous allocator; no GPU
fault occurred. Receipt: `rust-owned-io-both-gpus-20261005.json`.

The connected graph has a sealed, completely charged public-vector inventory,
actual 1286-page/785186816-byte private plan, separate 64MiB main IO capacity,
and full independently rounded outstanding source reservation. Admission charges
before creation, checks before slot/state publication, reconciles only at
completed readbacks and remains poisoned after failure. Eviction/Drop/telemetry
submission alone grant no credits. Cached projections avoid telemetry that
forced extra GPU submissions. Both GPUs reproduce all four bang and eight chat
IDs; cancellation at queued GDN0/QSA3 erases the owner, rejects poisoned reuse
and restores fresh first-token parity. Intel's 72-slot option also matches chat.
Receipt: `rust-main-admission-both-gpus-20261005.json`, frozen binary
`e92dd97bea0465a6fa86f0cf32eb27f5600783b5b1d5b047d866d7f69d7aa29e`.
These runs overlap isolated host builds/root-disk pressure and are acceptance
runs, not matched throughput or reliable MemoryPeak comparisons. The default
expert capacity remains unchanged.

Full connected-graph private staging remains pending; that graph still uses the
public arena policy at this receipt. Opaque driver/firmware padding, failed Drop,
historical swap and abandoned contexts remain excluded. This changes no API,
installed service, ordinary inference deadline or Vashti consumer. Native is
still slower than updated Ollama; no complete performance goal is claimed.

### Connected owned I/O, full-page inspection and multi-RHS operators

The previous paragraph describes the earlier checkpoint. The isolated resident
graph now has an explicit `KUNO_LAB_TRANSIENT_IO=1` path: private source pages
are retained before moved, guarded uploads; all map leases end before checked
sanitization. Separate main and runtime I/O capacities remain 64 MiB each.
There is no fallback to ordinary private staging. Cleanup failure retires the
shared physical stream across fresh logical owners. Ordinary inference polling
and installed services are unchanged.

Both GPUs reproduce the four-token bang and eight-token, 15-input-token chat,
including queued GDN/QSA cancellation, poisoned retry rejection and fresh owner
recovery. `KUNO_LAB_PREFILL_LAST_HEAD=1` runs every intermediate token through
all layers and a completed finite/status fence, omitting only its unused final
mixer/vocabulary projection. On B70 it preserves all 128 prior native output
IDs for the fixed 75-token baseline. This does not make the longer native output
bit-identical to Ollama. Owned-I/O bind/owner setup is 1.244 s, sequential prefill
64.591 s, and the 127 following steps take 75.753 s; transient unit wall is
142.156 s. Lazy weight loading is part of prefill, so this setup duration is not
equivalent to Ollama's reported model-load duration. Host builds overlap this
acceptance run; it is not a matched throughput comparison or a speed win.

The optional `KUNO_LAB_IO_TOKEN_CHECKPOINT=1` keeps intermediate staging under
the same independent hard runtime budget until the final accepted token fence.
Intermediate completed reads grant only main-pool credit, after dropping host
leases; they cannot reclaim runtime I/O spans. Final fault/logit acceptance still
sanitizes before committing recurrent/cache state. Both cards pass chat and
cancellation with this cadence. A later compiler-idle B70 chat ABBA is slower:
median prefill 12.355→13.731 s, following seven steps 3.323→3.512 s. Each process
has a fresh owner/executor; weights load lazily and the OS file cache is not
flushed. All eight IDs and explicit cleanup pass. The option remains unselected;
this short run does not establish a long-run or NVIDIA timing result.
`rust-token-cadence-abba-intel-20261005.json` records the four ordered samples
and frozen `5a7f7c2...` binary. Planning
temporaries, candidate/committed expert mappings, LRU state and duplicate route
vectors now wipe their complete retained host capacities on discarded ownership.
This is a narrow ownership improvement, not complete host privacy acceptance.

With `KUNO_LAB_OWNED_ZERO_AUDIT=1`, both cards inspect all **1,286 main pages /
785,186,816 bytes** after erasure, retaining handles until each full page and its
readback staging are accepted as zero. Each card checks four owners: completed
chat completion, a fresh prompt, and cancellation after GDN0 and QSA3.
Cumulative source-page counts are respectively 330/12,255,232 bytes,
225/8,355,840 bytes, 1/65,536 bytes and 4/196,608 bytes. These are page lifetimes
over resets, not simultaneous memory. Cleanup-only maps use checked native polls;
normal inference reads have no new deadline. Failed opted-in audit ownership is
quarantined until process recovery. Public receipts are written only after
explicit cleanup, cancellation/fresh-owner checks and executor shutdown succeed.

`rust-secure-resident-owned-io-20261005.json` records the exact flags and frozen
binaries: `fdf6d68...` for connected owned I/O/head skipping, `5edf911...` for
token cadence/host route cleanup, and `30f6e44...` for whole-page audits. Units
disable swap/core dumps; short runs use 12 GiB host limits and the full fixture
52 GiB. Unit bounds are operator controls, not production inference deadlines.
Observed full-run cgroup memory reaches at least 13,662,404,608 bytes with zero
swap; unreliable short systemd MemoryPeak summaries are not measurements.

#### Packed multi-RHS acceptance and cached timing

The Rust packed matrix primitive decodes each weight once per lane/column and
reuses it across four/eight token accumulators, retaining each token's original
64-lane sum order and reduction tree. Both GPUs pass all seven original GGUF
types with counts 1/3/5/8 and tiles 4/8: 56 cases per card. Nonzero runtime bases,
irregular strides, inactive tails, padding, physical nonaliasing, guarded I/O
inspection, full main erasure and public-weight preservation pass. Maximum
observed error against the independent Rust decode/tree is 1.20e-7.

A compiler-idle, same-process comparison measures eight legacy cooperative
matvec launches versus one multi-RHS launch, including completed output readback.
Weights/inputs are cached; initialization, numerical checks and audited cleanup
are outside the clock. Three ABBA rounds per candidate preserve sample order.
Tile8 ratios of median legacy time to candidate time are:

| Original matrix | B70 | RTX3060 |
| --- | ---: | ---: |
| Q4_K gate, 640×2560 | 3.71× | 2.96× |
| Q5_0 down, 2560×640 | 3.15× | 2.50× |
| Q8_0 down, 2560×640 | 3.05× | 2.09× |

`rust-packed-matmul-both-gpus-20261005.json` includes all raw numeric cases,
audits and ordered samples. Frozen numeric binary is `5edf911...`; timing binary
is `5a7f7c2...`. This is an operator win. Full layer-major batching still needs
causal GDN/QSA/PLE transactions and bounded expert-union dispatch; no whole-engine
or updated-Ollama speedup follows from this table.

#### Rejected exact-schedule down candidate

The optional Q5_0/Q8_0 kernel uses 16 physical threads for the same 64 virtual
accumulators and original addition tree. Both GPUs match the old all-ten MoE
output exactly in these samples, but five cached ABBA cycles are slower:
B70 Q5/Q8 median legacy→candidate 0.935→1.123 / 1.187→1.287 ms;
RTX3060 1.153→1.320 / 0.985→1.149 ms. The kernel remains disabled.
`rust-word-down-abba-both-gpus-20261005.json` records both formats, remapped
slots, cache-hit invariants and frozen binary `30f6e44...`.

Current checks pass 169 native host/offline tests, 28 imported runtime ownership
tests, strict all-target native/host Clippy, formatting and exact-archive composed
runtime patch reconstruction. Core remains at the separately accepted 74 normal
tests with two installed checks ignored. The runtime patch remains 18 files;
current composed SHA256 is
`a0d985c9b32653e95289cd64af651208f7b4b6142eb91e55e2af08e29e9ec5eb`.
No private request API, persistent worker, full options/quality/long-context
acceptance, native-default/`ollama.` routing, deployment or Vashti integration is
accepted here. Opaque driver/firmware memory and historical swap remain excluded.

#### Ordered public upload: accepted lifecycle, rejected timing

An explicit, default-off `KUNO_LAB_ORDERED_PUBLIC_UPLOAD=1` retains preowned
power-of-two source pages before moved uploads of original expert bytes. It
uses the unchanged copy kernel and legacy nonbatch fill, with no weight
conversion, routing pruning or private compile constants. Ten misses require
40 MiB of main source pages plus 40 MiB of separately tracked runtime uploads;
the pinned complete-router window is conservatively 42.5 MiB under the existing
64 MiB runtime cap. Every completed router, including cache hits, drops read
leases before checked staging cleanup, source release and MAIN fence credit.

The first B70 cancellation/audit run failed closed at first-position QSA3,
writing no receipt. That short QSA path has no selector read, so a preceding
40 MiB upload window remained live when the optional audit tried to duplicate
its stages under a 64 MiB cap. Explicit owner close now drains/erases those
stages and releases public sources before full private-main/source inspection,
without taking the ledger lock or granting credits. Failure retains sources
and retires the context. Drop remains unchanged. Already sanitized runtime
stages have checked erasure/completion proof, not a zero-inspection receipt;
normal router checkpoints also sanitize without inspection.

Both GPUs and both token-cadence settings then pass exact eight-token chat,
queued GDN0/QSA3 cancellation, poisoned retry rejection, fresh recovery and all
four 1,286-page private-main audits. No pending sources remain after cleanup.
Strict Clippy and the 176 host/offline suite pass; the explicit teardown fix
is additionally checked by that device matrix and independent source review.

A compiler-idle ABBA uses one frozen executable with only the upload flag
changed. Each process has a fresh owner/cache, no OS-cache flush, one input
token and three following graph steps; all four IDs and explicit cleanup pass.

| Card | Median prefill, legacy→candidate | Median following steps, legacy→candidate |
| --- | ---: | ---: |
| B70 | 5.232→7.368 s | 1.843→3.613 s |
| RTX3060 | 7.107→12.042 s | 3.350→10.322 s |

The candidate remains disabled. `rust-ordered-public-upload-both-gpus-20261005.json`
includes exact source/dispatch inventory, all accepted device cases, raw ordered
samples and frozen binary `6582901...`; the rejected initial binary is recorded
separately. Units disable swap/core dumps with 12 GiB host and 600/300-second
acceptance/timing bounds. No long-fixture, production worker/privacy or whole
engine gain is inferred. These observations do not establish one sole cause
of the regression. The [causal prefill plan](kuno-native-text-prefill.md) records
the next whole-model batching boundary.

#### Batched GR operator acceptance

The separate `gr-batch-probe` borrows the original attention, feed-forward and
final GR weights and compares count1/3/5/8 at matrix tiles4/8. It retains the
original 64-lane reductions, HC0..3 sum order, SiLU and injection formulas, and
the final mixer's no-injection behavior. Both cards pass all 24 cases against
the existing scalar GPU operators and an independent Rust CPU oracle. Maximum
observed absolute error is 2.0980835e-5 under the unchanged operator tolerance.

Runtime spans include nonzero bases, irregular strides, inactive columns and
gaps. Scalar comparison spans separately respect the actual storage alignment.
Each case owns two complete MAX8 private sets totaling 7,536,640 bytes under a
16 MiB MAIN cap, with an independent 64 MiB runtime I/O cap. Full-page private
erasure and guarded readback checks pass; both receipts report 768 audited
buffers/180,879,360 bytes in aggregate and no retired context. Protected units
disable swap/core dumps with an 8 GiB host and 300-second operator limit.

`rust-gr-batch-both-gpus-20261005.json` records all cases, source identities and
frozen binary `4db6515...`. Root host/offline tests pass 184 cases. This is
numerical and ownership acceptance only: no GR timing, complete causal prefill,
private worker, public API or whole-engine gain is established.

#### GDN recurrence chunk acceptance

The separate `gdn-chunk-probe` computes up to eight consecutive recurrence steps
while retaining each state column locally. It keeps the exact original f32
decay, prediction, delta, update and readout order, modulo head pairing and
immutable committed input state. Actual count and every span base/stride remain
runtime data. Every active output/status and the final candidate state must pass
acceptance together; the operator provides no commit API.

Both cards pass counts1/3/5/8 for odd7/2/6 and actual Qwen128/16/48 shapes from
nonzero initial state. The actual-shape outputs/final state exactly match the
unchanged single-step GPU operator in every case. Maximum error against the
independent Rust core is 5.960465e-8 across all fixtures. Finite invalid controls,
pre-enqueue/post-fence cancellation, old/next aliases, old-state immutability,
inactive gaps and complete owning padding are checked.

The fixed main plans are 524,288/22,249,472 bytes under a 32 MiB MAIN cap; runtime
I/O remains separately capped at 64 MiB. The largest primary I/O/audit-duplicate
windows are 13,665,280/27,330,560 bytes. Both receipts inspect 128 complete main
page lifetimes/91,095,040 bytes before release and report accepted cleanup without
context retirement. Units disable swap/core dumps, with 8 GiB host and 300-second
operator bounds. The current combined root host/offline suite passes 197 tests,
strict all-target Clippy and formatting.

`rust-gdn-chunk-both-gpus-20261005.json` records all cases and frozen executable
`7082b1d...`. No model payload is read by this probe. Convolution, matrix
projections, closing gates, whole-model chunk integration and performance are
outside this acceptance; register/spill and opaque driver memory are outside
the owned API-page erasure proof.

#### Segmented 96-slot expert cache: short acceptance, long failure

An explicit resident-only 96-slot candidate splits each original packed bank
into immutable 64+32 segments. It keeps every token's ten experts/probabilities,
global ranks and original arithmetic. Full 48-layer banks charge 18 GiB of
complete pages; they require an explicit static-cache budget (6 GiB tested)
under the unchanged 26 GiB model limit. Ordinary 64/16-slot defaults and the
8/6 GiB static-cache defaults remain unchanged. Generic/nonresident MoE,
expert-batch, word-down and ordered-upload combinations are rejected before
allocation. The full 96-slot plan is rejected for the 3060's 10 GiB limit.

`bank96-probe` has a separate bounded one-layer admission; it never bypasses
full-model preflight. Ordinary plans/fills/accepted consumers prime actual
slots0..95. Both cards exactly match flat10 activation, down partials and output
for Q5/Q8 with slots0/63/64/95, preserving all ten ranks. Independent Rust
64-lane trees/core references also pass. Post-enqueue cancellation and a partial
copy into a real second-segment LRU victim poison the owner and reject retries;
full private-page cleanup and public weight sentinels pass. An initial envelope
check failed before allocation: flat10 costs48 MiB for Q5 and64 MiB for Q8.
The corrected combined charges are432/448 MiB. Root tests/strict Clippy pass202.

The full B70 96/6 GiB chat fixture preserves all eight native IDs, queued
GDN0/QSA3 cancellation, poisoned retry, fresh recovery and four complete
785,186,816-byte main inspections. A compiler-idle short ABBA is near noise:

| Static cache for 64 slots | Median prefill64→96 | Median seven following steps64→96 |
| --- | ---: | ---: |
| 6 GiB | 12.483→12.618 s | 3.249→3.263 s |
| 8 GiB | 12.471→12.623 s | 3.283→3.255 s |

The longer 75-input/128-output ABBA did **not** complete. Its 64/8 GiB reference
preserves all prior native IDs with accepted cleanup (59.392 s prefill/71.238 s
following steps). The subsequent 96/6 GiB run fails during runtime I/O cleanup
after partial generation, writing no success receipt. The underlying category
was collapsed by the existing error path; remaining samples were not run.
Kernel inspection found no matching Xe/DRM/GPU fault/reset messages in the
checked interval, and a fresh Intel GDN chunk/cleanup test passes immediately.
This does not establish the cause or accept longer 96-slot operation.

`rust-bank96-both-gpus-20261005.json` records accepted cases, all ordered short
samples, the interrupted long comparison and recovery with frozen executable
`32387b0...`. The larger cache remains opt-in, with **no default selection or
whole-model performance claim**. No service, public API or Vashti changes apply.


## Static owned-I/O diagnostic follow-up (2026-10-05)

The default-off96-slot/6GiB cache candidate failed a second protected long
public rainwater run, this time after52 completed prefill tokens. The unchanged
cleanup policy now preserves a whitelist of28 static first-observed categories;
the failure was `upload_map`. It does not expose error descriptions, shader
source, INFO values or request content. This category still conflates mapping
rejection and missing callback delivery, so it does not establish a driver cause.
The stream retires and emits no successful receipt. No cap, timeout, retry,
ordinary inference deadline, installed runtime or public API changed.

The frozen diagnostic binary is
`bcd9bcf2ebcc8b9636fbe8928d1666f7917a79837f34e5c777e7a69bba9ea9b7`.
The run exited1 after42.964s; this failed run is not a performance sample.
A fresh Intel context immediately passed all8 GDN chunk fixtures and full
128-page/91,095,040-byte inspection without retirement. This is fresh-context
acceptance, not a proof that the failed context was erased or that the underlying
mapping issue is solved. No reset or reboot was attempted.

204 native host/offline tests and31 owned-runtime fixtures pass, as do strict
all-target Clippy and formatting. The composed18-file vendor patch was
regenerated from exact pinned cached archives and reconstructed with every
modified hash verified; patch SHA256
`1dcabe09fb8e095c9b780a590a65a188e32ed9681c4a9faa2d022d52ebc2da7d`.
The ordinary64-slot default is retained. See the structured
[diagnostic receipt](../apps/kuno/benchmarks/text/rust-owned-io-diagnostic-20261005.json).


## GDN batched preprocessing and closing acceptance (2026-10-05)

The candidate-only fixedMAX8 path now batches causal RAW convolution+SiLU,
Q/K normalization and scaling, stored-negative-A decay/write controls and
per-head RMS+sigmoid closing gate. Count/bases/strides are runtime; old RAW
history is immutable and no layer/cursor commit is provided. The four projections,
recurrence and final projection still need a connected full-layer receipt.

Both physical GPUs pass24 cases across odd and actual dimensions/counts1/3/5/8,
plus12 invalid raw/alpha/z/stored-A fixtures. Full active columns, statuses,
unchanged inputs, inactive/gap/padding regions and final raw histories pass.
Existing scalar128-wide GPU kernels and independent core operators provide the
references; odd7-wide normalization/closing uses the separate generic count-one
reference and core, without pretending the unchanged128-only kernel supports7.
Maximum absolute CPU/scalar-GPU error is2.3841858e-7 on B70 and1.1920929e-7
on3060, within the existing tolerance. Each card accepts1080 complete main-page
lifetimes/121,634,816 bytes plus owned-I/O inspection with no retired context.

Frozen executable SHA256:
`bd9fe5f9f1916a82148a1c4abb42f6c8656904c099ed2bc0d76ccdf100b9d492`.
218 combined host/offline tests, strict all-target Clippy, release build and
formatting pass. The same binary also contains an unaccepted optional QSA
candidate; this receipt establishes only GDN pointwise acceptance. No timing,
whole-engine speedup, private worker, production default or API change is claimed.
See [the numeric receipt](../apps/kuno/benchmarks/text/receipts/rust-gdn-pointwise-both-gpus-20261005.json).


## Causal QSA chunk acceptance and plane-index correction (2026-10-05)

The after-projection QSA candidate passes24 fixtures on BOTH B70 and3060:
committed starts0/3/4/2047/2049/2051 and active counts1/3/5/8. Each query builds
only its visible complete raw-indexer-key groups before normalization/RoPE;
a later completed group never enters an earlier plane. Original head/12,
stable score selection, complete blocks and exact incomplete tails are retained.
All candidate tokens and gated outputs are bitwise equal to unchanged scalar GPU
execution on both cards. Maximum independent core error is2.861023e-6, within
the existing tolerance. Late nonfinite future-key rejection preserves earlier
planes and rejects the whole chunk. Both cards accept full1728 owning-page
lifetimes/1,379,401,728 bytes and runtime audits, cancellation and immutable
committed cache. No sequence commit hook is provided.

The first Intel attempt failed validation because three kernels used the
flattened `ABSOLUTE_POS` across gridY. Later token planes were skipped. Those
three kernels now use X-only element coordinates; the mandatory validators and
numeric tolerance are unchanged. Checked/unchecked WGSL index coverage plus
the complete both-card device fixtures cover the regression.

Frozen accepted binary:
`91ce1fcea660808eb569f3e881872d358520d35423a5dbd105c4023163923b51`.
220 combined native host/offline tests,35 runtime fixtures, strict all-target
Clippy, release build and formatting pass. This binary also contains the bounded
mapping-callback repair described below. The QSA receipt alone does not establish
whole-model batched prefill, image behavior, a private worker or performance.
See [the complete receipt](../apps/kuno/benchmarks/text/receipts/rust-qsa-chunk-both-gpus-20261005.json).


## Mapping callback completion repair (2026-10-05)

Pinned CubeCL runs a native background polling thread. In pinned wgpu-core29,
queue maintenance removes ready mapping callbacks under its lock, then fires
them after dropping locks. A background poll can claim a callback and be
rescheduled after cleanup's own poll returns. The former immediate `try_recv`
therefore treated a still-undelivered successful map as failure. This is a
source-confirmed reachable race; the failed96 run's ambiguous old `upload_map`
code alone cannot prove that it hit this interleaving.

Eraser upload remapping, audit mapping and the synchronous cleanup-read seam
now require delivered callback completion within ONE mapping deadline. Native
polling and the callback receive share the remaining30-second window; there is
no remap, retry, additional full wait or arbitrary inference deadline. Full
volatile wipe, unmap, destination clear, completion/error acceptance and audit
remain mandatory. Failed/panicking/timed-out mappings retain ownership and retire
the context. Readback leases retain cancellation/unmap behavior and mutation
acceptance follows `into_bytes` success. Ordinary async inference reads are
unchanged. This requested-wait bound cannot preempt a synchronous driver call
or OS descheduling; whole cleanup still comprises several bounded operations.

The whitelist now has37 static codes, separating mapping rejection, disconnected
callback and callback timeout at all three seams. It reports the first observed
category, which can precede a later-consumed typed GPU error; it never formats
backend descriptions or source. Native BufferAsyncError discards its underlying
cause, and device loss is separate from the three scoped GPU error classes.
35 ownership host fixtures pass, including controlled callback-claimed/Empty-
then-delivery, errors, disconnects, a shared deadline and stale-success rejection.
Full device/graph acceptance is recorded separately below.

The repaired frozen binary completes the original public75-token prompt and128
generated tokens with96 expert slots and the explicit6-GiB static cache. All128
native token IDs equal the previous64-slot native result; checked I/O cleanup
succeeds. Prefill is59.480 s and the127 following decode steps take77.039 s.
This is one diagnostic run with concurrent compilation permitted, without full
main-page zero audit. It establishes long-run acceptance, not a matched speed
gain or proof that the earlier failure necessarily hit the callback race. The96
cache remains default-off. See [the exact long receipt](../apps/kuno/benchmarks/text/receipts/rust-mapping-callback-long-acceptance-20261005.json).

## Append-only INFO target slabs (2026-10-05)

The optional runtime target-only slab accepts0,4096 or8192 bytes; default0
preserves individual targets. Full initialized target spans stay owned until
checked erasure, including unused tails. A slab appends within its current I/O
epoch, without rewinding, reusing across epochs, batching mapped upload sources
or silently checkpointing. Before mutation, admission reserves source/target
ownership and future full duplicated audit staging under the same64-MiB I/O
ledger. Records too large for a slab use the checked individually owned path.

Both physical GPUs pass all three sizes with67 distinct runtime scalar records,
two epochs, retained-readback rejection, dropped pending futures, abandoned
queued records, original public-weight preservation and full owning-page/I/O
audits. Oversized individual INFO has host-policy checks but no separate GPU
fixture. The frozen819c9ac binary then passes original chat8 completion on both
cards at sizes0/4096, and separate explicit cancellation runs on all four
combinations pass poisoned retry rejection, fresh first-token match and queued
GDN0/QSA3 cancellation. Each cancellation run verifies four complete main owners,
each1,286 pages/785,186,816 bytes, with independent guarded I/O cleanup.

The normal v1 runs did not select cancellation; only the v2 runs establish it.
Neither sequence is compiler-idle paired timing or production private-worker
acceptance. Slabs remain disabled pending measurement. Root host checks at the
frozen binary pass222 native and39 ownership fixtures plus strict Clippy and
formatting. See [the operator and complete-owner receipts](../apps/kuno/benchmarks/text/receipts/rust-info-slab-both-gpus-20261005.json).

## Guarded contiguous expert-wave planner (2026-10-05)

The host-only MAX8 planner preserves every token's ten original expert IDs and
probability bits, builds maximal contiguous waves under the routed-bank limit,
and exposes the complete stable union before any future victim selection.
Explicit initialization bits reject an omitted rank even if its default zero
expert/probability would otherwise look valid. It never drops, sorts or
renormalizes original routes. Nine host fixtures cover overlap/disjoint unions,
capacity boundaries, exact route order, malformed packets, cancellation and
full guarded storage wiping. Fixed owned API storage peaks at2,528 bytes; there
is no private heap allocation. The module is registered only for host tests.
The future GPU adapter must additionally retain and validate the actual pending
bank plan/generation through completed consumers; a union alone does not pin
slots. Fullgraph integration and private request acceptance are outstanding.

## Full GDN layer and fused matrix fault source checkpoint (2026-10-05)

The new candidate-only full GDN layer composes all five original packed layer0
matrices, accepted causal convolution/normalization/recurrence/closing-gate
kernels and output projection. It preserves immutable committed history/state,
checks all ten status planes and emits no state/cursor commit. The fixed probe
compares every stage and final candidates at counts1/3/5/8 against unchanged
single-token GPU scheduling and the independent decoded public core oracle.
Its exact plans admit24,117,248 private main bytes,67,698,688 original public
weight-page bytes,17,235,968 runtime audit-peak bytes and88,053,848 guarded host
scratch bytes. The decoded public oracle alone is231,669,760 bytes, bounded
below256 MiB; it is a test oracle, not a new inference weight representation.

Independent source reviews found no material order/geometry/ownership or scalar
comparison blocker. Root passes245 native host tests including the new wrapper,
full owning-read and generated checked/unchecked WGSL cases, strict all-target
Clippy, release build, formatting and diff checks. **Physical GPU acceptance of
this complete layer is pending.** See [the exact source receipt](../apps/kuno/benchmarks/text/receipts/rust-gdn-layer-host-20261005.json).

A separate default-off `KUNO_LAB_FUSED_MATRIX_FAULT=1` candidate folds the
existing output-only finite predicate and fault-bit OR into the cooperative
projection, preserving the original64-lane arithmetic and output values.
Block-kernel fallbacks retain the separate finite fence; no graph state commits
early. Both execution modes compile and Naga-validate all seven packed formats.
The fixed device probe borrows small original model rows, compares original GPU
output/faults with normal, NaN and signed-infinity activations, and requires exact
full owning read lengths before decoding, guarded leases, all tails and full
cleanup. It has not run on a GPU yet. Frozen combined source binary SHA256 is
`a1be34e63179b2ba1fdc33ad13c15fe27f8f456362f5cda69482dce3041f59b7`.
The earlier `bf4a1ef7` executable predates full-read hardening and is not accepted.
Privileged device units did not start because cached sudo expired; normal Kuno
admission remains enabled. No device, performance, private-worker or release
claim follows from these source checks.

The shared cache-backed MAX8 projector now validates each complete original
input/output row span before dividing matrices into bounded row tiles, reuses
the existing original packed static-weight cache and preserves its reserved
memory ceiling. It passes actual runtime counts/bases/strides through guarded
packets and keeps the separate full-output finite fault fence. Routed expert
banks remain a separate adapter. This seam accepts neither state nor cursor and
has no connected production consumer or device acceptance yet. Combined source
checks pass247 host tests, strict all-target Clippy and formatting; see the
[source-only composition receipt](../apps/kuno/benchmarks/text/receipts/rust-layer-composition-source-20261005.json).
The frozen `a1be34e6` probe binary predates this shared adapter; its prepared
device fixtures prove only their exact frozen source after they run.

## Grouped all-ten expert operator source checkpoint (2026-10-05)

The test-only grouped MoE candidate reuses each decoded original packed weight
across4/8 RHS members while preserving each route's original64-lane tree. Gate/up
is fused; down scatters back to original token/rank partials. One final reduction
runs after all waves in ranks0..9 order. Runtime maps/counts/slots/probabilities
never specialize pipelines. The separate shared expert remains outside this
routed operator, and no route is discarded or renormalized for residency.

All seven original formats pass78 checked/unchecked offline WGSL variants and
12 focused host/operator fixtures. Combined root acceptance passes259 tests and
strict all-target Clippy. Fixed maps and probability/coverage scratch are guarded
before filling and wipe on ordinary success, error and unwind. Actual D2560/FF640
full owning scratch charges1,409,024 bytes, or1,671,168 bytes including MAX8 input
and output; caps remain unchanged. See [the source-only operator receipt](../apps/kuno/benchmarks/text/receipts/rust-grouped-moe-operator-source-20261005.json).

A borrowed slot tuple is not proof of actual bank protection. The owning adapter
must validate and retain the immutable wave and real pending bank plan through
completed consumers, stage successive wave ledgers privately, and publish only
after the whole-model chunk fence. That wrapper, actual physical bindings and
both-card numeric/cleanup checks remain pending. No device, whole-model, timing,
default-route or production private-worker claim follows from these host checks.
## PLE and complete QSA layer source candidates (2026-10-05)

The standalone lab now composes original layer1 PLE and layer3 QSA candidates.
This is source acceptance, with physical comparisons still pending. Combined
root validation passes284 native host/offline tests, strict all-target Clippy,
locked offline release build, formatting and diff checks. Normal `kuno-text`
passes75 tests with two installed checks ignored. No installed service, public
API, catalog, default routing or Vashti consumer changed.

PLE preserves all16 original u64 hash branches, boundary248044 versus EOS248046,
bounded110-byte Q5 row reads, original projections and normalized-history taps
9/6/3/0. Input and both committed histories remain immutable. Candidate history
is exposed only after layer acceptance, with no whole-model commit API. The
existing resident PLE tail is CPU; the new count1 GPU tail is labeled separately
from scalar GPU projection and independent core comparisons. Eight numeric and
four failure cases on each card remain pending. Paired full MAIN pages total
15,958,016 B under16 MiB; diagnostic owning-I/O bound17,367,040 B under64 MiB;
conservative guarded host inventory42,909,696 B under64 MiB.

QSA includes all six original packed matrices and the accepted causal selector/
attention operators. Prepared geometry, physical buffer ownership and pending
state are sealed before dispatch. Its unchanged scalar comparator resets every
status plane per token and advances only its separate reference cache after
completed acceptance. Twenty-four contexts/counts and four rejection paths on
each card remain pending. Dual capacity2064 full MAIN pages total58,818,560 B
under64 MiB; audited I/O bound35,491,840 B under64 MiB; guarded host bound
169,577,300 B under256 MiB. The dual4096 fixture needs100,925,440 B and is
explicitly rejected before allocation rather than enlarging the cap.

Both fixtures require exact full owning reads, scrub/drop of readback leases,
checked independent I/O cleanup and full MAIN zero inspection before a receipt.
Unexpected run/read/dispatch failure retires the context before best-effort
close. This proves no opaque driver/register/spill or abrupt-loss erasure, and
does not accept a private worker, whole-model prefill, full context4096 or speed.
Receipts are
`text/receipts/rust-ple-chunk-source-20261005.json` and
`text/receipts/rust-qsa-layer-composition-source-20261005.json` under
`apps/kuno/benchmarks`. They identify source hashes and frozen executable
`b7b2e40955d39829f71bf8a6dc09620b38f658ada3725dc311e77e13cf328983`,
which includes the shared cache-backed batch projector and sealed GDN owners.
Earlier prepared `a1be34…` units describe an older frozen binary; do not
substitute identities or infer device acceptance from this build.

## Native GR workspace and owning expert-wave source checkpoint (2026-10-05)

The laboratory now has a cache-backed native GR workspace and an owning grouped
expert-wave fixture. Root acceptance passes300 native host/offline tests,
strict all-target Clippy, locked offline release build, formatting and diff
checks. The unchanged core acceptance is75 normal tests with two installed
checks ignored. The frozen source executable is
`0c1b3f553ae24ee497f2f9637324778582feeeb89b9427e0c451d67d157438f5`,
at `/tmp/kuno-rust-native-batch-owner-benchmark-20261005`. This is source-only
acceptance: there was no GPU run or installed service/API/default change.

GR reuses the original cached packed projections and public norm vectors. Its
ten MAX8 pages charge1,998,848 B, including the131,072 B mixed-input page. A
prepared borrow prevents scratch reuse until the residual combine or final
output head and their active finite folds are queued. The last head consumes
only the last active mixed column; intermediate prompt chunks omit the entire
final mixer/head. Abandoned/error capabilities poison the workspace. Available
means consumers queued, not completed GPU acceptance or allocation credit.
Peer review caught premature final mixed-input release; the explicit original
head enqueue now retains that capability, and the focused recheck passes.

The shared status fold validates complete active extents and checks the active
row before loading storage. A nested shader branch is required because the
Cube DSL eagerly lowers both operands of a logical AND. Checked/unchecked Naga
validation and a source-order regression cover that guard. Existing scalar
projection and status behavior remain unchanged.

The expert-wave fixture binds the actual original graph, layer and bank
generation, validates canonical pending mappings before mutation, and protects
each complete wave union through bounded fill windows and completed consumers.
Each source window has at most ten misses and40 MiB; independent runtime I/O
remains64 MiB. Staged bank ledgers remain private: the routed operator never
publishes the original ledger. Later waves preserve already completed rank
partials. Cancellation, partial mutation, panic or unexpected runtime failure
poison the owner and retire its physical context before best-effort erasure.
Source credit follows owning lease destruction and verified I/O cleanup.

The grouped fixture's private candidate plus scalar comparator charge1,998,848 B
under2 MiB; guarded host inventory is15,872,000 B under32 MiB. Physical bank
upper bounds, complete allocator baseline and the temporary source window fit
its1 GiB MAIN cap. Forty positive, twelve rejection and twelve fresh-owner cases
remain pending on both cards. These are operator fixtures, not a native
full-model expert adapter, separate shared expert, full FFN or whole-model
transaction. Neither source checks nor isolated operator improvements establish
an Ollama speedup, private-worker acceptance or opaque GPU/abrupt-loss erasure.

Exact source hashes, budgets, review boundaries and remaining work are recorded
in `rust-native-gr-batch-source-20261005.json` and
`rust-grouped-moe-chunk-source-20261005.json` under
`apps/kuno/benchmarks/text/receipts`. Native GDN/QSA adapters and the complete
layer-major transaction remain under implementation; no frontend change follows
from this internal laboratory checkpoint.

## Both-card layer and expert fixture acceptance (2026-10-05)

The frozen layer fixture `b7b2e40955d39829f71bf8a6dc09620b38f658ada3725dc311e77e13cf328983`
now passes on both B70 and RTX3060: four GDN counts and four rejection cases,
eight PLE cases including odd geometry and its four rejection paths, and24 QSA
causal-boundary/count cases plus four rejection cases. These receipts supersede
the corresponding device-pending statements above, not their narrow scope.
Across all cases, GDN original scalar-GPU error is zero; independent core error
is at most1.9073486e-5 on B70 and1.5258789e-5 on3060. PLE original projection
and new count1 GPU-tail errors are zero, with original CPU/core tail error at
most1.9073486e-6 on each card. QSA original scalar-GPU error is zero; core error
is at most9.5367432e-7 on B70 and1.4305115e-6 on3060.

The owning routed-expert fixture frozen at
`0c1b3f553ae24ee497f2f9637324778582feeeb89b9427e0c451d67d157438f5`
passes40 primary numeric cases,12 rejection cases and an independent fresh owner
after each rejection on each card. All ten original ranks/probabilities and
staged bank generations are preserved; no original ledger is published. Maximum
core-output error is4.1909516e-9 on B70 and4.6566129e-9 on3060; scalar-GPU
error is2.3283064e-10 and zero respectively. Cumulative owning MAIN zero
inspection covers123,994,112 B/812 pages per card. The separate shared expert,
full FFN and actual ResidentModel adapter are outside this fixture's acceptance.

All eight invocations used user systemd units with `MemorySwapMax=0`,
`LimitCORE=0`,8 GiB `MemoryMax` and a900-second lab-only runtime bound. A wrapper
verified actual cgroup/core limits before executing. This removes the need for
sudo for these lab tests; it adds no production inference deadline. The user
declared the cards free, and both listeners reported no resident models before
the loop. Installed Kuno admission was not paused and independent CPU builds
remained active. Unit runtimes are not performance measurements; inconsistent
systemd MemoryPeak reports are not used as memory evidence.

The complete numeric reports, all-case maxima, exact frozen identities,
protection-wrapper hash, bounds and limitations are retained in
[the layer device receipt](../apps/kuno/benchmarks/text/receipts/rust-gdn-ple-qsa-device-20261005.json)
and [the expert device receipt](../apps/kuno/benchmarks/text/receipts/rust-grouped-moe-device-20261005.json).
Owning MAIN/IO acceptance excludes opaque driver/register/spill and abrupt loss.
These frozen fixtures predate the new native cache-backed adapters. They do not
prove full4096 QSA, whole-model batching, private-worker acceptance, quality/
options, production routing, or an Ollama speedup. Installed services, API,
catalog, defaults and Vashti are unchanged.

## Native cache-backed layer adapters (2026-10-05)

Native GDN and QSA adapters now borrow the real ResidentModel state/cache and
original packed projector. One reusable MAX8 workspace adds3,932,160 B for GDN
and17,825,792 B for QSA, with no new public weights or duplicated committed
cache. Opaque staged receipts bind the actual sequence, chunk, graph-derived
36/12 layer masks and generations. QSA's required intermediate selector maps
owning score/status pages under the actual admission/client and shared poison;
both read leases drop before cleanup or credit. No state advances on enqueue.

The separate shared expert borrows its original three cached matrices and gate
vector, adding262,144 B of five full pages. GR final mixing reads only the last
original residual column; intermediate prompt chunks omit final mixer/head.
Full physical private extents must be offset zero and equal the actual backing
buffer. The common actual admission now carries sticky logical sequence poison;
pending Drop stores an atomic without reentering its held mutex, and allocation/
readback recheck it under the admission guard. Logical failure is distinct from
physical GPU retirement.

Root source acceptance passes336 host/offline tests, strict all-target Clippy,
locked offline release, formatting and diff checks. The
[source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-layer-adapters-source-20261005.json)
records exact files, incremental owning budgets and the external whole-model
fence/all-family preflight requirements. These native adapters are not connected
to a whole-model chunk yet and require separate physical acceptance. Runtime
components are not approval of the future aggregate64 MiB I/O budget. The actual
native MoE and PLE adapters, complete constructor/transaction, matched Ollama
improvement, quality/options and production worker/routing remain pending.

The optional finite-fault projection fixture now passes28 original-format cases
on each card with untouched scalar-GPU output/fault parity and owning cleanup:
[exact receipt](../apps/kuno/benchmarks/text/receipts/rust-fused-matrix-fault-device-20261005.json).
Earlier failures came from fixture initialization policy and intentional NaN
upload rejection. Original public weights now prewarm before owned I/O; fixed
nonfinite inputs use guarded IEEE words while the normal float upload finite
check stays intact. All seven formats preserve the original output and OR bits;
each card inspects4,587,520 cumulative MAIN bytes/140 pages as zero. This remains
a small operator fixture; the optimization is default-off pending full-model
parity, cancellation and paired measurements. No public or installed behavior
changed.

### Full scalar finite-fault lifecycle smoke

Frozen `861b285a…` also passes the original full scalar `resident-bang` fixture
with finite fusion enabled on both GPUs. Both reproduce IDs
`[49276,2349,1074,5928]`. Pre-enqueue cancellation, queued GDN layer0 and QSA
layer3 cancellation erase complete owners; poisoned retries fail and a new
owner reproduces the first output. Each of four owner audits per card inspects
785,186,816 B/1,286 pages, with separate owning source/IO cleanup.
[The exact receipt](../apps/kuno/benchmarks/text/receipts/rust-fused-full-resident-device-20261005.json)
retains fixed public settings, diagnostic timings and observed lifecycle proofs.
The identical protection wrapper is preserved as
[protected-lab-exec.py](../apps/kuno/benchmarks/text/protected-lab-exec.py).
These runs use the original scalar model, not new MAX8 adapters; they establish
neither full quality nor paired performance acceptance. The flag stays default-off.

## Native PLE bridge into actual CPU histories (2026-10-05)

The native layer1 adapter now borrows the real CPU token/convolution histories,
original cached projections and sealed public vectors. Nineteen new complete
MAIN pages add6,258,688 B; no public weights or committed CPU state are duplicated.
Its524,288 B GPU OLD page is an explicit readonly mirror, with NEXT candidate-only.
Seven stage statuses plus history/output folds join the same model fault.

Enqueue does no maps, completion or state commit. After the future whole-model
fence, owning OLD/NORMALIZED/NEXT maps validate complete pages and exact original
history bits. A guarded core factory reconstructs the ordered token/normalized
extension, compares NEXT bitwise and returns an opaque borrowing commit ticket.
The field-split ticket retains only actual PLE histories/anchor, leaving other
families and a second actual fault/IO/admission proof available. All families
must preflight before any state swap; subsequent scalar generation uses the
same accepted CPU history objects. Error/abandonment wipes candidates and poisons
the logical owner; there is no arbitrary imported-state setter.

Root acceptance passes343 native and80 core tests (two installed checks ignored),
strict all-target Clippy for both, locked offline release, formatting/diff and
focused peer source review. The [source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-ple-bridge-source-20261005.json)
pins identities, full-page/guarded-host bounds and the4,587,520 B audited I/O
component, which is not global64 MiB credit. The
[approved actual connection plan](kuno-native-text-model-batching.md) now records
aggregate constructor and acceptance budgets. This adapter is still unconnected;
physical native/full-model acceptance and matched speed remain pending.

## Native actual-bank MoE and guarded router (2026-10-05)

The native MAX8 routed adapter now uses the actual executor's original bank
instances and retained `StagedSlots`/`BankSlots` plans. It protects the complete
selected union before choosing victims, uses bounded original packed fills,
and retains the final pending plan for each of48 layers until external complete
model acceptance and all-family preflight. Scalar bank access is rejected while
the native session is armed. Interior owning dependencies permit safe scratch
and victim reuse; they do not publish the original bank ledger or establish
whole-model completion.

Five reusable full MAIN pages add1,409,024 B. Original scalar owners remain for
subsequent decode. The private pending metadata fits a2 MiB component cap;
original public packed payloads retain their separate40 MiB cap. The524,288 B
runtime component includes the491,520 B conservative audited/slab bound and
does not grant aggregate64 MiB I/O credit. Drop only stores sticky logical poison;
explicit terminal cleanup remains outside admission guards.

The new guarded core router preserves the original f64 softmax/sum and f32
round points, probability ordering/ID ties, denominator clamp and all ten ranks,
including zero probabilities. Its fixed scratch/result total6,256 B on this
64-bit host and allocate no heap storage. Guards clear complete API-visible
fields on error, unwind and drop; opaque compiler/register copies are outside
this proof. The existing scalar router API is unchanged.

Root acceptance passes350 native and86 core tests, with two installed checks
ignored, strict all-target Clippy for both, locked offline release, formatting/
diff and focused peer source review. The initially recorded frozen executable
`27c25fea…` was an unrelated historical probe; its source association is invalid.
[The forward correction](../apps/kuno/benchmarks/text/receipts/rust-native-artifact-binding-correction-20261005.json)
retains the error and selects the current combined executable from actual Cargo
artifact metadata. Source checks remain valid; no device acceptance came from
that incorrect executable.
[The exact source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-moe-router-source-20261005.json)
pins files, delivery identities and component bounds. This actual-bank adapter
has no connected whole-model consumer or new physical acceptance yet; older
grouped fixture and scalar lifecycle receipts do not establish those properties.

[The native decode plan](kuno-native-text-decode.md) records the next measured
optimization path. Stage1 original packed Rust CPU experts and a pinned-file
positional reader are approved for isolated synthetic source implementation.
Actual CPU/GPU calibration, mixed execution and an expert-only secondary GPU
remain later experiments. MAX8 prompt batching alone cannot establish faster
hot decode. No public API, model catalog/default, installed service or Vashti
consumer behavior changes at this checkpoint.

### Owning embedding and output I/O helpers

The connected-model prerequisites now include bounded original embedding-row
gather, router packet readback, final vocabulary readback and complete fault-page
readback. These use the actual client/admission and full disjoint physical owners;
no MAIN page is added. Embedding fills a preguarded full owning source and queues
only. Router output fills an already-zeroed boxed guarded table, avoiding a move
of the filled table on return. Final logits have exactly the admitted guarded
capacity after full finite/fault/padding validation.

Mandatory admission readback drops raw guarded leases inside its callback before
checked audited I/O cleanup and forced normal checkpoint/credit. Failed, partial,
empty or overflowing audit receipts retire the context. All errors/cancel/unwind
poison the logical sequence; no helper commits state or grants model acceptance.
Private host caps are768 KiB embedding,64 KiB router and2 MiB output. Audited I/O
components are1,064,960 B embedding,81,920 B per router boundary and65,536 B for
the second final fault proof, within the separately planned aggregate64 MiB cap.

The composed root passes361 native host/offline tests, strict all-target Clippy,
locked offline release, formatting/diff and peer source review. Core86 tests are
unchanged. [The source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-batch-io-source-20261005.json)
pins exact interfaces/files/bounds. The earlier identical `27c25fea…` identity
was an unrelated old probe, as recorded in the forward correction above; it did
not prove dead-code elimination or the helper source association. Source checks
remain valid, and no device or performance claim follows from them.

## Connected native model laboratory (2026-10-05)

Source `371b9d1` connects the actual48 original layers through native chunks of
up to eight prompt tokens. It uses the original cached dense/GR/GDN/QSA/shared
projections, authoritative CPU PLE histories, actual original staged expert banks
and original guarded all-ten router. Intermediate chunks omit the complete final
mixer/head; only the last original residual column reaches the final head.
Subsequent ordinary scalar generation uses those accepted original model owners.

The constructor charges retained scalar plus90 native pages before allocating:
1,376 private pages/818,249,728 B and the same379 sealed original public vector
entries/25,165,824 B. Explicit nine clear records add18,432 audited runtime bytes;
complete runtime reserves64,000,000 B for this Q6 head and64,090,112 B for the
worst supported F32 head, below64 MiB. Separate original bank-state capacity
checks add512 KiB to controlled private host accounting:11,075,584 B below12 MiB.
Existing native MoE candidate/next reserve stays intact. The
[historical approved plan](kuno-native-text-model-batching.md) now records these
implemented accounting corrections.

Output acceptance completes the actual owning same-fault/output proof, guarded
PLE maps/lease destruction/audit, a second same-fault/I/O/admission proof, and all
family preflights before any state/bank publication. Cancellation/partial commit/
unwind poisons the complete model; explicit model/executor-registry teardown runs
outside admission guards. No healthy cursor/output or fallback follows failure.

Root369 native tests, strict all-target Clippy, locked offline release and
formatting/diff checks pass; core86 checks are unchanged. The
[exact source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-max8-source-20261005.json)
records complete budgets and actual Cargo-selected executable
`fca4b66911251cfcf8a74ce1399f0304e4f3f979e5dcedc688fd9e735d61ca15`.
The default-off fixed-public flag `KUNO_LAB_NATIVE_MAX8=1` requires owned transient
I/O; `KUNO_LAB_NATIVE_BATCH_COUNT=1..8` selects a contiguous public prompt partition.
There is no new public API, catalog/default, installation or Vashti behavior.

Initial protected device runs pass native `resident-bang` (actual partition `[1]`)
and `resident-chat` (`[8,7]`) on both cards. Both produce original scalar public
IDs and continue with ordinary scalar generation. Pre-enqueue cancel, queued
native GDN layer0/QSA layer3 cancel, poisoned retry rejection and fresh native
replay pass; each of four owner audits per run inspects all1,376 pages/
818,249,728 B, with separate source/I/O cleanup and no context retirement. These
are short fixed-public lifecycle smokes. Their complete reports and lifecycle
observations are preserved in the
[initial device receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-initial-device-20261005.json).
The public runner in `apps/kuno/benchmarks/text/run-native-max8-lab.py` verifies
Cargo executable identity, protected limits, complete model receipt and actual
cancellation/cleanup messages; successful exit alone is insufficient. Other
partitions, full75/128 output,
whole-model long-context boundaries, injected failures, quality/options,
compiler-idle matched performance and production worker/routing remain pending.

Both cards additionally pass the complete short chat partition matrix:
count1=`[1;15]`, count3=`[3,3,3,3,3]`, count5=`[5,5,5]`, count8=`[8,7]`.
Every case reproduces the same original eight public IDs, accepts scalar
continuation and passes four full audits plus native GDN/QSA cancel/poison/fresh
recovery. The
[partition receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-partitions-device-20261005.json)
preserves all eight actual reports under the original `fca4b669…` artifact;
typed effective card/cache/bank/runtime/source-audit fields were revalidated.
The final NVIDIA count1 run directly uses that hash-verified immutable binary;
its failed first launch never reached the protection wrapper/model and is excluded.
This establishes short-fixture partition invariance, without full-logit, long
context, options or matched speed acceptance. The newer CPU-composed binary
retains separate provenance; the full75/128 native fixture is the next device gate.

The CPU-composed `b1ba040f…` binary now passes the full Intel75/128 native fixture,
partition nine8s plus3, all128 prior native public IDs and scalar continuation.
All four1,376-page/818,249,728-byte private audits, native GDN/QSA cancellation,
poisoned retry rejection and fresh full75-prompt replay pass. The
[full Intel receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-full-intel-20261005.json)
preserves exact source/build/report/reference identities and the observed
worker-unit exit0. CPU experts remain uncalled by this model.

The first outer runner exited1 after the model completed: its validator wrongly
applied the simultaneous64-MiB source cap to cumulative completed audit counts.
`PrivateArena::reset` erases/inspects/releases each live source batch, then adds
numeric counts; only those counts survive.127 scalar continuations each inspect
15 buffers/17 full32-KiB pages, giving1,905 buffers/70,746,112 B in total. This
does not describe simultaneous residency. The corrected runner requires those
exact cumulative counts for the public fixture; a missing full page rejects.
Root revalidated the original completed report without rerunning the model,
preserving both worker0 and outer-runner1 outcomes. No cap or engine behavior
was changed. NVIDIA's completed counterpart now passes the same 128 original
IDs, full-prefix fresh replay, cancellation and all four MAIN/source audits.
Its worker and corrected outer runner both exit 0. The
[full NVIDIA receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-full-nvidia-20261005.json)
preserves the complete report, Cargo/binary identities and original run record.
Both full runs use the same `b1ba040f…` binary. Their recorded timings include
inspection and diagnostic instrumentation; these are correctness/lifecycle
runs, not paired speed measurements.

## Original packed Rust CPU expert source acceptance (2026-10-05)

Root integrates `7950074`/`d1a0446`, the author's `55dfe343` plus corrected
`d8c08873` metadata ownership. The standalone core helpers preserve all seven
original weight layouts, F32 activations, explicit64-lane multiply/add order,
tree32/16/8/4/2/1 and original sign-stable SiLU. They return unweighted requested
rank partials; no selected rank is pruned and probability/shared/final reduction
remain the caller's responsibility. Portable and AVX2 backends are explicit;
physical plans choose1..16 allowed distinct physical cores without fallback.

The new positional reader clones the already-open original descriptor, pins
actual model/tensor identity, bounds every read, checks unchanged size/mtime
before/after and clears the full destination on I/O/late-consistency failure.
It never reopens by path or mutates the ordinary reader cursor. Same-size/
same-mtime malicious changes remain outside the original guard. No model
download, duplication, conversion, expanded whole expert cache or mmap is added.

Controlled CPU request buffers, <=16x32-KiB scratch and full retained job/worker
metadata are admitted under a1-MiB component cap before private fill. Original
public packed jobs stay below40 MiB and deallocate normally; private slots,
offsets/counts and dead outer headers erase after ordinary payload release and
completed workers. Four focused witnesses cover full retained capacity, partial
fill/read/cancel/unwind and join ordering. Compiler/register/OS-stack/opaque std
thread storage are explicitly outside the controlled erasure claim.

Root debug and release core checks pass105 tests (two installed checks ignored);
the composed native lab passes369. Both strict all-target Clippy, locked offline
release and formatting pass. The
[exact CPU source receipt](../apps/kuno/benchmarks/text/receipts/rust-packed-cpu-experts-source-20261005.json)
pins source/check hashes and the new composed Cargo-selected lab artifact
`b1ba040f…5e8f589`. The helpers are not yet called by model execution. Real original
CPU/GPU numeric comparisons, completed joint timing and whole-model hybrid
acceptance remain pending under the [decode plan](kuno-native-text-decode.md).
New scoped workers per phase are a measured overhead question. Source equality
does not imply device bit equality or faster-than-Ollama performance.

## Original packed Rust CPU expert physical acceptance (2026-10-05)

The fixed-public example `packed_expert_public_probe` now executes the integrated
CPU operator against actual original model slices on the Ryzen 5900XT. It
derives the first Q5 down layer (6) and Q8 down layer (0) from metadata and
verifies their original Q4 gate/up geometry. Experts 0 and 7 use the compiled
public input `((column % 29) - 14) / 512`; no prompt or route is accepted.

An independent original row decoder uses explicit scalar 64-lane multiply/add,
tree 32/16/8/4/2/1, original SiLU and down projection. It holds one guarded
decoded row and never expands a complete expert. Portable single-worker full
partials and AVX2 physical worker plans 1/4/8/16 match that reference bitwise.
All 16 AVX2 cases also match both raw gate/up projections and leave the nine
inactive rank tails exactly zero. The final original-file consistency check and
protected user unit complete successfully with verified swap/core limits.

The [actual CPU receipt](../apps/kuno/benchmarks/text/receipts/rust-packed-cpu-experts-actual-20261005.json)
pins the example/source hashes, successful strict example Clippy and locked
offline release checks, immutable executable `2c89851a…`, Cargo snapshot and
complete public result lines. Systemd's reported 256-KiB memory peak is excluded
as an unreliable measurement. This is limited single-expert CPU numeric
acceptance, not an all-ten CPU/GPU merge, timing, whole-model hybrid or production
private-worker proof. CPU helpers remain uncalled by model execution. Completed
joint calibration and matched updated-Ollama performance remain outstanding.

## Actual model failure and context fixture source (2026-10-05)

Root `19e12af` integrates the isolated lifecycle candidate `c2f120c`; `cdb3524`
registers an explicit fixed-public laboratory entry. It constructs 22 actual
model/executor owners serially, using original scalar sequential references at
context 4/8 and native partitions `[3,1]`, `[3,5]`, `[8]`. The original tokenizer's
single `!` token is repeated through genuine inference; no cursor/history/cache
setter is used. Exhaustion and empty/9-token/out-of-vocabulary chunks check real
terminal cleanup, rejection by both retry APIs and separate fresh recovery.
Original scalar exhaustion still rejects at preflight; the diagnostic explicitly
closes that owner. Native malformed/exhausted chunks must close automatically.

Sealed default-unarmed hooks inject a late error after layer 47 routed/shared
combine, error/panic while the real PLE ticket is alive before the second owning
fault proof, and cancellation after the first genuine GDN state/history commit.
Failure must prevent output/cursor publication and continued use of any partially
changed sequence. Unexpected failures, incomplete audits or uncertain cleanup
retire the physical stream. Independent review of these actual sites found no
material source blocker.

Each owner retains the original native 1,376 pages/818,249,728 bytes or scalar
1,286 pages/785,186,816 bytes. No extra MAIN is admitted. The separate diagnostic
has a 4-MiB host and 64-KiB metadata cap; original model host/source/runtime caps
are unchanged. Full 248,320-logit bit/greedy comparisons report numerical mismatch
only after checked cleanup, without accepting a new tolerance. Source tests do
not establish device equality.

The protected `resident-lifecycle-probe` entry requires both transient I/O and
complete audits before device creation, configures one physical retirement Arc,
and writes a new numeric-only receipt after all owners close. The runner checks
all 22 exact cases, complete audits, contexts/hooks, numerical result and protected
completion; a cleaned numerical rejection is retained with a nonzero result.
Its 1,800-second diagnostic containment does not change production inference
deadlines. Root 376 native tests, strict all-target Clippy, locked offline release,
format/diff and three actual pre-device protection rejections pass. The
[source receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-lifecycle-source-20261005.json)
pins the Cargo-selected `110b9926…` artifact. Both-card actual execution remains
pending. Whole-model long contexts, options/quality, production private workers,
native routing and faster matched Ollama inference remain outstanding.

### Both-card actual lifecycle acceptance and numerical rejection

The frozen `110b9926…` candidate now completes all 22 real owner cases on each
card: context 4/8 exhaustion, malformed chunks, the genuine late-error/panic/
post-first-commit cancellation sites, both retry rejections, full owning cleanup
and fresh recovery. Each scalar audit covers 1,286 pages/785,186,816 bytes and
each native audit 1,376 pages/818,249,728 bytes; native source counters are zero.
Actual scalar source resets inspect exactly 15 buffers/17 full 32-KiB pages per
token. Both physical retirement flags remain false and both worker units exit 0.

All 24 final greedy comparisons match the original scalar reference. The full
248,320-logit comparisons differ: maximum absolute delta is
`1.0371208190917969e-5` on Intel and `8.702278137207031e-6` on NVIDIA. None exceeds
the recorded stage diagnostic bound, but that bound is not an accepted whole-model
tolerance. Both reports explicitly retain `whole_model_numeric_accepted=false`;
the outer runners each exit 1 after successful cleanup. The
[complete device receipt](../apps/kuno/benchmarks/text/receipts/rust-native-model-lifecycle-device-20261005.json)
preserves both outcomes, all cases, reports/run hashes and numerical statistics.
Identical aggregate statistics do not prove native-partition vector identity.
The first differing stage and its arithmetic cause remain to be investigated.

These opposite-card runs were simultaneous and include diagnostic inspections;
their times are not performance evidence. The existing full 128-token native
reference acceptance remains separately valid. There is no new public inference,
model quality, long-context, production-worker, routing or release acceptance.
