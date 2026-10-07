# Native scalar decode: Rust CPU co-execution and mixed-GPU plan

## Foundational optimization pass, 2026-10-07

The user authorized a substantial pass on placement, routing, arithmetic,
weight access and execution overhead, using10 decode tokens/s as the working
target without repeating Ollama baselines. The implementation below remains
laboratory-only until actual connected output, lifecycle and timing acceptance.
The original ordinary engine settings and installed gateway stay unchanged.

The AVX2 CPU expert operator now directly decodes original Q4_K, Q5_0 and Q8_0
weights into vector registers. It retains F32 activations, separate multiply/add,
all64 lane round points and the original reduction tree. Other layouts use the
existing decoder. This is exact CPU arithmetic optimization, not activation
requantization or a foreign engine.

`KUNO_LAB_MAPPED_WEIGHTS=1` is an explicit protected count1 opt-in. One read-only
mapping uses the originally inspected GGUF descriptor, with no path reopen,
model copy, page prefault or pinning. CPU expert jobs borrow bounded original
packed spans, replacing their per-call payload allocation and read/copy. Their
request-derived headers retain the full-capacity erased owner and joined worker
lifetimes. The40-MiB public selected-span bound and1-MiB private host bound still
apply; virtual mapping length and OS-managed public file pages are separate
from private request buffers. Ordinary positional reads remain the default.

Mapping requires preventing in-place writes/truncation of the pinned inode for
the full model lifetime; an unsafe constructor states this obligation. Atomic
path replacement cannot redirect the original map. Stat/ownership/span checks
remain but cannot prevent a concurrent truncate or same-stat malicious rewrite.
This first lab uses the existing immutable content-addressed Ollama blob under
exclusive model-installation access. Production enforcement is still required.

`KUNO_LAB_KERNEL_PROFILE=1` explicitly configures basic kernel timing for the
fixed-public count1 fixture only. No arbitrary request profiler, prompt logging,
compilation dump or persisted route history is introduced. Profiling changes
execution timing and never establishes an unprofiled speed claim. Both options
are0/1, rejected outside the protected fixture, and recorded in its receipt.

Placement and reusable joined workers are already connected. Host CPU expert
execution still needs the per-layer router/input dependency. GPU routing,
subgroup reductions and an additional device cache must be separately connected
and measured; no claim follows merely from source existing.

The next explicit protected option, `KUNO_LAB_FAST_GPU=1`, replaces count1
dense projections with four rows per128-thread workgroup. Each row uses32
physical lanes, each accumulating two of the original64 logical lanes, then
performs the original stride32/16/8/4/2/1 tree using shared memory and subgroup
shuffles. Logical lane numbering comes from subgroup ID/size, so no hidden
mapping from local invocation indices is assumed. The adapter must advertise
enabled subgroups with bounds16..32 and128-thread workgroups. Q4_K shares block
headers and nibble bytes; other original formats use the existing decoder.
Prefill remains the original MAX8 path and private arena/fault/cleanup owners
are unchanged. The completed receipt records actual fast projection dispatches.

`KUNO_LAB_LONG_FIXTURE=1` selects the existing fixed-public75-token rainwater
prompt and128 outputs through the same hybrid model path. No arbitrary prompt,
variable generation limit or private trace is accepted. Validation uses all128
previously accepted token IDs,127 actual decode steps, original inventory and
the same fresh/cancel/poison/erasure gates. Longer timing is needed to separate
count1 shader startup from sustained decode. The default remains short chat.

`KUNO_LAB_CPU_Q8=1` explicitly selects a second CPU arithmetic mode, restricted
to pooled AVX2 workers and the original Q4_K/Q5_0/Q8_0 expert triple. Each32-value
activation block is requantized once with a finite amax/127 scale, ties-to-even
rounding and clipping to[-127,127]. Zero and underflowed blocks become zero.
Packed original weights remain unchanged. AVX2 integer dot products include
Q4 minimum corrections and Q5 centering; Q8 uses unsigned absolute weights with
signed activations, avoiding maddubs saturation even at weight=-128. This
changes activation precision and reduction order, and is NOT original64
numerical equivalence. Original F32 arithmetic remains the default. Unsupported
layouts/ISA/worker policy reject; inference failure never selects this mode.
All quantized inputs (up to11 fixed owners) are charged before private fill and
their complete capacity is erased after joined consumers on every exit. The
1-MiB private CPU cap remains unchanged. Tests compare integer dots with an
independent decoded-weight/dequantized-activation oracle, extreme signed codes,
non-finite scales, cancellation, panic/retry and complete-owner erasure. Greedy
fixture IDs, fresh continuation and cleanup still must pass device validation;
full-vocabulary numerical/quality acceptance is separately outstanding.

`KUNO_LAB_LARGE_WEIGHT_CACHE=1` and runner `--large-weight-cache` explicitly
allow a48-GiB whole-process cgroup for immutable mapped model pages. They require
the fixed mapped count1 fixture. The wrapper independently verifies the actual
51539607552-byte hard limit, swap0 and core0 before native execution; a receipt
cannot grant an unbounded allowance. The ordinary8-GiB protected limit remains
unchanged for all other modes. Private CPU/GPU/IO ledgers are NOT enlarged.
This addresses observed model-page reclaim under the8-GiB process allowance,
not private request caching. It is a local64-GiB-host benchmark option, not
production hardware sizing or an automatic memory policy.

`KUNO_LAB_COMPACT_INFO=1` and `--compact-info` connect the existing bounded
4096-byte runtime INFO slab to this hybrid fixture. The requested runtime
`KUNO_LAB_INFO_SLAB_BYTES` must exactly match4096 (ordinary mode0); neither is
inherited by the runner. Records use separate aligned non-overlapping slices;
full slab gaps/tails and all upload/readback/INFO owning spans still undergo
the same mandatory audits at every model readback. There is no audit skip,
deferred private cleanup or cross-epoch private owner cache. The slab-specific
public multi-record/cancellation/retained-lease probe and whole-model gates
are needed on the actual devices before selecting it.

`KUNO_LAB_BATCH_TASKS=1` and `--batch-tasks` explicitly increase the runtime
submission group from32 to256 compute dispatches. Dependency readbacks still
flush the same stream. Model/arena/fault/cancellation/span/erasure/admission
proofs and all private hard limits remain unchanged. This is bounded scheduling,
not a cached command graph or a claim of device-side routing. Completed receipts
pin both selections and their numerical runtime bounds.

`KUNO_LAB_FAST_GPU_FAULT=1` / `--fast-gpu-fault` requires the explicit fast
subgroup path. Each producer atomically folds its finite result into the same
global model fault. Only the last public packed row tile checks the entire
remaining logical output, excluding the full active matrix interval because
each producer has already checked it. Thus other tiles can never race a tail
check, and large heads do not rescan every inactive MAX8 tail for every tile.
The old complete logical finite check remains for prefill and all non-fused
projections. Full owning padding/erasure checks, original arithmetic/rounding,
fault lifetime and final acceptance remain unchanged. Every format must pass
fixed-public active-overwrite, inactive-NaN, bad-input and sticky-poison tests
before whole-model work; the receipt requires actual fused dispatch counts
equal to the actual fast projection counts. This is a kernel fusion experiment,
not permission to skip numerical faults or change output owner extents.

Source-only proposal, 2026-10-05. Prepared by the resident-MoE child after the native MAX8 adapter handoff. No code, GPU, service, model-payload read, benchmark or default change accompanies this proposal. Parent owns whole-model batching correctness, admission, device acceptance and integration. This is a later decode experiment; MAX8 prompt batching alone cannot establish faster hot decode.

## Evidence and first decision

Start with an exact-schedule Rust packed CPU expert operator, then a simultaneous CPU/actual-upload calibration fixture. If that wins, add an opt-in CPU/B70 mixed expert seam. Add the 3060 as an expert-only cache only after the two-way seam passes. Device-side routing/global caches remain separate later experiments.

The existing long profile in `docs/kuno-text-performance.md:352-367` reports 50,682,982,400 original expert bytes moved across 127 following steps, 31.565 s nested expert-upload/enqueue time and 2.996 s router-fence time. That receipt overlaps builds and is not a current paired throughput baseline. Enqueue timers can include preceding GPU work, allocation, implicit submission and read/copy work; subtracting those nested counters from wall time does not prove a compute floor. The later owned-I/O long receipts are acceptance runs, also not a compiler-idle comparison.

The recorded updated-Ollama changed-prefix baseline is 7.602 decode tokens/s; repeated-prompt decode is 7.973 tokens/s with cached prefill. A current clean native/Ollama matrix is required before any speed claim. Compare generated-token rates separately from prompt work; native has no cross-request prefix reuse. Preserve the fixed public prompt, seed 424242, greedy options, visible token counts and exact artifact identity.

Current actual sources:

- `resident_model.rs::moe`: original F32 router projection, completed owning 513-word router/fault packet, original host router normalization, actual bank selection/fill, routed output and separate shared expert. There is one route dependency per each of 48 layers. The fence cannot simply disappear while the host decides actual misses.
- `expert_banks.rs::fill_source`: each original gate/up/down component is read to public packed bytes, uploaded through the narrowly scoped owned-public-byte path, then copied to its actual slot. The candidate removes redundant host Vec copies while retaining the ordinary GPU upload policy; see the runtime-ownership README for privacy and physical acceptance limits. The packed bytes no longer receive private volatile erasure. Bigger public upload batching, ordered uploads and alternate word-down have measured regressions and remain off.
- `moe_block.rs` and `moe_gpu.rs`: exact 64-lane accumulation order; gate/up SiLU; unweighted down partial per original rank; final rank 0..9 probability multiplication and addition; separate shared path.
- `crates/kuno-text/src/cpu.rs::Matrix::apply`: bounded reference decoder but a left-to-right f32 sum, with row/tile allocations; it is not the GPU expert schedule.
- `small_cpu.rs::dot_avx2`: guarded small matrices, four 8-wide accumulators over 32 columns and a different final tree. Its <=64-output public decoded cache is useful for tiny projections, not a drop-in 640/2560-row expert engine.
- `gguf.rs::TensorView::read_range`: validates artifact/extent but holds one shared file mutex across stat, seek and read. Naive parallel CPU workers therefore serialize with each other and GPU public fills.

FreeToken is pinned at `189c8ad0c3350e30252209d40780274976cb7527`; the inspected local checkout's HEAD matches. Its useful ideas are concurrent miss execution/transfer calibration, bounded integer split, resident-hit protection and recency-based choice of which recurring misses to fetch. It also merges already-reduced GPU and CPU outputs, which would reorder our original ten-rank sum. Its C++ Q4_0 CPU path quantizes activations to Q8_0, and its other CPU paths use different formats, BF16 and/or FMA. We must implement original GGUF/F32 math in Rust instead of importing that compute path. CUDA mapped flags/stream memory operations are not an established Vulkan mixed-vendor synchronization primitive.

Primary pinned sources: [overlap calibration](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/benchbw.py#L538-L596), [fraction selection](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/bench_profile.py#L156-L190), [hybrid operator order](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/layers/moe.py#L298-L351), [capped LRU](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/offload_kernels.py#L313-L411), [CPU packing/math](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp#L1114-L1220).

## Stage 1: independently accepted packed Rust CPU experts

Add a focused CPU packed projection primitive plus a selected-expert operator. Keep dependencies fixed initially: `std::arch` AVX2 intrinsics, runtime feature detection, existing GGUF/quant code, `zeroize` and bounded scoped workers. No whole-model foreign engine, activation requantization, approximate SiLU, changed probability bits or expert pruning.

Use eight 8-lane accumulators representing lanes 0..63. For every row, visit input columns lane + 64*tile in the current GPU order. Explicit separate multiply/add, no FMA, fast-math or reassociation. Store the 64 guarded partial sums and apply strides 32,16,8,4,2,1 in that order. Q4_K gate/up decode can share block headers and each private input load. Apply original sign-stable SiLU after the two complete sums, then original down projection. CPU outputs remain unweighted [rank][2560]; final probability multiplication/sum stays on the primary GPU.

Implement all seven original decoders (F32/F16/BF16/Q5_0/Q8_0/Q4_K/Q6_K), with the first device comparison focused on actual Q4_K gate/up plus Q5_0/Q8_0 down. Reuse exact block field interpretation/round points from quant.rs. A scalar *64-lane* CPU reference must be independent of the vector decoder. Current Matrix::apply remains an independent mathematical oracle within tolerance, not the bitwise accumulation oracle. CPU/GPU exp, denormal and contraction behavior still require measured acceptance; source schedule equality alone does not imply bitwise device equality. Reject non-finite inputs, decoded values, intermediates and outputs; retain all zero-probability ranks as real selections.

Parallelize independent output rows, not a row's reduction or the ten-rank sum. Start with 1/4/8/16 physical-core workers and separately test SMT; do not assume all 32 threads win. Guard private arrays before fill/allocation, keep descriptor/offset identity pinned, wipe full capacities on errors/unwind, join workers before releasing their private input/output owners, and use static content-free errors. Cancellation checks at bounded row/block work points; no hard inference deadline.

Original expert payloads are exactly:

| Component | Original geometry/packing | Bytes per expert |
| --- | --- | ---: |
| Gate | Q4_K 640 x 2560 | 921600 |
| Up | Q4_K 640 x 2560 | 921600 |
| Down | Q5_0 2560 x 640 | 1126400 |
| Down | Q8_0 2560 x 640 | 1740800 |
| Complete Q5/Q8 expert | No conversion | 2969600 / 3584000 |

Each complete expert has 4,915,200 weight/input products plus 640 SiLU operations. Avoid expanding every expert into F32. First use at most ten bounded original packed jobs (35,840,000 public logical bytes maximum, explicit 40 MiB public host cap); workers share those immutable original bytes. Allocate job metadata under a private guard before filling; payload is public but which expert was chosen and its offsets are request-derived transient metadata. Do not retain a host decoded expert cache/history across requests.

A narrow immutable positional reader is a prerequisite for real concurrent I/O. On Linux use FileExt positional reads on the pinned opened descriptor, preserving offset/length/max-read validation, short-read handling, unchanged-file checks and artifact identity. Never reopen by path or mutate a shared file cursor. Add injected short/error/change and parallel offset tests. Mmap is a separate measured alternative, not required for the first operator: don't pin the complete 120 GB file or create another file copy. A public packed job can read its three components once, then workers perform no file I/O while computing.

## Stage 2: measurement before split policy

Use fixed-public data only, numerical acceptance first. Warm the compilation paths, pools, public fixture pages and clocks outside timed samples. Use payload working sets larger than the CPU last-level cache, rotating public experts without saving real request routing. Record device/driver/artifact/binary/CPU worker count, actual page charges, every sample and numeric checks. No heavy concurrent compilation.

Measure for each actual down format and misses 0..10:

1. CPU original packed complete experts, completed outputs; scalar versus AVX2 and worker counts.
2. Actual current legacy GPU read/upload/copy plus completed consumer/fence, not only enqueue duration. Separate same-bytes-hot host payload versus actual GGUF reads.
3. Resident GPU experts with no misses, including current gate/up/down/reduce/status work.
4. CPU execution and real GPU fills/resident work concurrently from one start barrier, plus a deliberately serialized control using identical work and memory/IO checks.
5. Transfer private 2560 input and rank-indexed results through the actual owned I/O path, including completed fences and required cleanup, not raw memcpy estimates.

Search discrete assignments 0..M CPU misses using the *measured joint* times. The heuristic is max(T_cpu(M-F), T_primary_gpu(hits,F)) plus measured bridge/merge cost. A bytes/GB/s formula is only the starting guess; include setup, SiLU, dtype, worker saturation and memory contention. Keep GPU hits on GPU, protect them before victims, and fetch promising recurring misses only under bounded private request-local recency. Never silently switch to CPU on an unexpected GPU failure. Initially use explicit lab counts; any future auto choice needs public hardware/format calibration and stated intentional CPU placement. No persisted prompt, selected-ID histogram or request-derived profile.

## Stage 3: rank-preserving CPU/B70 native seam

Reuse the existing required router read dependency. An opt-in combined owning packet can contain original 512 router logits, original 2560 mixed input and one fault word (3073 words/12292 logical bytes, one 32 KiB owning page). Drop its readback lease and complete checked I/O before launching CPU jobs. Keep host normalization and exact guarded original all-ten router selection. This adds 10 KiB logical input traffic instead of downloading every expert activation.

Each rank has exactly one runtime owner: primary GPU or CPU. GPU bank planning protects every selected resident GPU expert before deciding any victim. Only chosen GPU misses enter a staged actual BankSlots fill plan; CPU-assigned misses remain absent. Bank instance/revision/full handles and plans remain pinned through completed consumers, and all real ledger publication stays behind the whole-model final fault/I/O/causal/shared acceptance. Use the native registry/poison Arc; no scalar bank access or replacement during a hybrid session.

Do not upload CPU results into the same private buffer while queued GPU kernels write it. Primary GPU keeps its existing partials; add a distinct full ten-rank CPU partial owner. Start CPU jobs, enqueue safe resident/fetched GPU ranks and separate shared work, then join CPU jobs, validate finite complete CPU partials and upload that guarded owner. One final kernel selects each rank's one accepted owner and evaluates probability[rank]*partial in rank 0..9 order. No aggregate GPU+CPU output merge. Runtime owner/slot/count values stay in private buffers/scalars, never comptime or cache/debug keys. Predicated GPU routes must guard all bank accesses and participate in unconditional barriers; unused CPU slots are not decoded or executed with fake zero weights.

Initial incremental envelope, to be recomputed against the *current* whole-model aggregate before code:

- MAIN: packet32 KiB + CPU partial128 KiB + runtime rank ownership/readiness32 KiB = **196608 B**. Existing primary GPU ACT/PART/status/probability and shared path remain owned and charged.
- Private host: full input/packet, full ten-rank output, rank/job descriptors, <=16 workers x32 KiB scratch and reduction guards; explicit **1 MiB cap** with actual Vec capacities checked. Public selected packed weights are a separate <=40 MiB host job cap.
- Incremental owned runtime I/O: packet read32 KiB + full CPU partial upload128 KiB + owner/readiness upload32 KiB + conservatively128 INFO records at <=512 B target+source =327680 B. Duplicate audit staging and two8 KiB slab tails produce688128 B; reserve **1 MiB component bound**. This is not aggregate credit or a change to either independent64 MiB cap. Actual alignment/grid/bindings/record count must be validated before allocation; fail unsupported limits.

CPU jobs own their private buffers until all workers stop. Drop only sets shared sticky poison; explicit abort joins outside admission locks, then checked erases/retirement on the original streams. Any worker panic, partial read/fill, enqueue, mapping, non-finite output, cancellation or cleanup error fails the whole token; no accepted output or original ledger retry. No callback or Drop may synchronously reenter ResidentAdmission. Ordinary inference still has no arbitrary deadline.

## Stage 4: 3060 expert-only cache, not tensor parallelism

Use the existing Rust Vulkan backend on each GPU, two independent clients, arenas, owned I/O ledgers and physical admission plans; there is no proof of cross-vendor peer access. Keep dense/GDN/QSA/GR/shared work on B70 initially. Avoid duplicating the entire static matrix cache on the 3060. An explicit16-slot-per-layer secondary original bank set has **48 x64 MiB =3 GiB** pooled pages for either actual down dtype (gate16+up16+down32 MiB), plus its separately admitted small private/source/runtime resources. This is a bounded candidate, not a default budget change; derive/verify every actual layer triple and available device bound first. Start with one-layer operators before whole 48-bank allocation.

Prefer existing B70 hits, then secondary resident hits, then choose CPU/B70 fetch/secondary fetch using measured costs. Avoid intentionally caching the same hot expert twice; retain private ownership/recency metadata only inside the request/session. The NV selected ranks return unweighted rank-indexed partials through owned host readback to a *separate* B70 remote owner, then the same original rank-order merge runs once. Complete and validate the remote model fault/status/I/O before accepting bridge bytes. Both device owners share one sequence abort Arc; failure on either poisons whole-model acceptance and prevents publication on either bank. Admission/final preflights of both clients and all model families precede any publication.

For a nonempty secondary set, logical transfer is one 10240-byte input toward the 3060 and up to 102400-byte rank-partial output back; page/padding/IO charges are larger and must be planned. Compare this against avoided original 2.97/3.58 MB expert transfers, but don't assume it wins: 3060 compute, its chipset PCIe path, two callbacks/cleanup boundaries and shared host DRAM can dominate. Measure CPU+B70+3060 contention together. More expert retention is not automatically faster.

Only later compare layer partitioning (whole layers on3060 with activation bridges at boundaries) and a global mixed-format pool. The current B70 per-layer page/cache structure can strand free slots and duplicate equally reserved capacity across hot/cold layers; a global cache needs new bins, immutable destination geometry and slot lifetime proofs, and must not be conflated with this narrow secondary-cache experiment. Public72-slot same-page option and96/6 GiB candidate remain separate matched placement trials with no silent static-cache shrink.

## Acceptance ladder and blockers

1. Host decoder/64-tree fixtures: all seven layouts, nonzero offsets, signed zero/subnormal/non-finite values, overflow/bounds and worker-panic/cancel/guard checks. CPU scalar-versus-vector exact round-point tests; no private scratch left on any exit.
2. Parent-owned both-GPU one-expert/intermediate comparisons using original actual Q4/Q5/Q8 slices, existing numeric tolerances and full owning gaps/tails/status/zero audits. Vary every rank/owner bit including allCPU/allGPU and zero probabilities. Do not reduce expected experts to improve parity.
3. Real staged-cache hybrid one-layer acceptance, protected hits/eviction/partial fill/source debt/cancel/panic/drop/stale identity and failure-no-publication tests.
4. Whole fixed-public75/128 native fixture, short chat, GDN/QSA/shared/GR fault/cancel/poisoned retry/fresh recovery and all cleanup audits. IDs/logit tolerances relative to the accepted native schedule; quality claims remain separate.
5. Compiler-idle same-binary ABBA legacy versus CPU splits, then NV-resident splits. All samples/completions retained, warm host weights distinguished from model-unloaded state. Rerun updated-Ollama cold/warm/changed-prefix matrix only at this phase. Select a candidate only if actual end-to-end hot decode wins reliably with correctness/containment intact.

An immediate all-GPU router replacement is not this first step. The original normalization uses f64 exp/sums with explicit f32 round points and total_cmp ties; a WGSL f32 softmax/top-k is not identical. GPU f64 capability/emulation and expert-cache bookkeeping need an independent exactness/limit design before dropping the 48 host decisions. Packet reduction can reduce traffic but cannot remove required slot decisions. Likewise speculative expert prefetch/MTP/activation quantization are separate opt-in contracts, not hidden speed fixes.

Expected implementation order: packed CPU primitive + positional reader; joint public operator calibration; explicit CPU/B70 native rank seam; whole-model measured trial; secondary expert-only3060 trial. Parent batching/owner acceptance remains the current priority. No faster-than-Ollama, privacy-erasure, mixed-GPU or production/API claim follows from this plan.
