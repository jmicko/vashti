# Coupled K64 matrix staging candidate — 2026-10-19

Kuno-only candidate after the source-only transfer review. No GPU execution,
sudo, service change, shared Cargo build, API or Vashti source change.

The previous32x64 macro product loads all four subgroup operands directly from
global memory everyK16, duplicating operands between neighboring subgroups.
The new complete macro chain keeps the same four16-lane subgroups, four8x16
accumulators per subgroup, exact orderedK16 products and original result/fault/
status/route stores, while introducing:

- Actual typedvec4 FP16/BF16 source bindings, with authentic8-byte lengths and
  alignment. Each lane copies8 adjacent values using two Vulkan-legalvec4
  loads/stores, preserving bits without F32 conversion.
- Cooperative A[32][64] / B[64][64] operand stages shared between subgroups.
  Two stages occupy24KiB; the unchanged result epilogue adds8KiB. Actual compiled
  shared memory is exactly32KiB, including masked tails, with no old macro
  tail arrays retained alongside the new storage.
- A ping-pong stage loop. First stage loads and synchronizes; each iteration
  loads/stores the next independent stage before four orderedK16 products on
  the current stage, then all lanes synchronize before switching stages.
  This is software staging/prefetch intent, not guaranteed asynchronous DMA.
  Next-stage stores may still wait for global load latency; a register-first
  prefetch variant can be considered after actual measurements, accounting for
  register pressure. No performance gain is claimed from source structure.
- Whole-stage zero masking for token, feature and reduction tails. Reduction
  dimensions retain the original multiple-of16 guard; unused finalK16 products
  are skipped, rather than adding zero products to the original accumulation.

The small product is unchanged. Shapes below32 tokens/features, non-vector-
aligned lengths/offsets or insufficient actual shared-memory capability use the
original scalar-half entry, not a forged vector descriptor or device capability.
Existing complete owner/range admission remains unchanged. Offline validation
now additionally checks the actual macro32KiB compiled envelope.

CPU acceptance uses the actual modified native module and existing real CubeCL,
wgpu, half and kuno-text rlibs, with tiny non-GPU sibling metadata helpers:

- `rustc --test -D warnings`:3 new tests pass for binding/fallback geometry,
  real compiledvec4 half8-byte storage and32KiB shared allocation across all
  eight device-layout masks, plus independent full-vector/tail/ping-pong index
  coverage and exactK16 visit order.
- Strict standalone production-module Clippy passes.
- All64 real small/macro FP16/BF16/dense/expert SPIR-V variants validate with
  the pinned matrix-capable validator and actual SIMD16 selector.

Artifacts: `/tmp/kuno-prefill-submit-review/matrix-{check.rs,tests.log,check.log}`.
The three new tests also live inside the actual native matrix module and run
under the ordinary matrix test filter. Existing GPU tolerances are unchanged.
Production full build, physical numerical matrices, whole-model outputs,
erasure/close/recovery and end-to-end timing remain root-task acceptance gates.

## Register-first successor to ab475149

The ab475149 baseline remains a distinct complete candidate. Its coherent
successor changes the prefetch phase as a whole: four A jobs and eight B jobs
per lane load into fixed local arrays of8 and16vec4 values BEFORE current-stage
MMA. The same vectors publish to the independent SLM stage AFTER current-stage
MMA, then the original uniform barrier switches stages. All load/publish job
loops are statically unrolled. Initial-stage publication, bit-preserving vector
types, zero masking,32KiB shared allocation, orderedK16 math and fault/owner
checks remain unchanged.

This removes the immediate next-load-to-SLM-store dependency before current
math. It still does not guarantee asynchronous execution. The extra per-lane
payload is96 half values (192 bytes), so register pressure/occupancy and possible
driver spills must earn an actual end-to-end gain.

All64 actual variants validate again. The actual compiled-module test now
requires that no Function-storage prefetch array survive: CubeCL's optimizer
scalarizes the statically indexed locals into SSA vectors. This does not prove
the physical driver allocates every value to registers without spilling. The
index/tail/ping-pong test now models fixed per-lane prefetch vectors separately
and publishes them only after current-stage reads. All3 focused CPU tests and
standalone production-module strict Clippy pass. Evidence:
`/tmp/kuno-prefill-submit-review/register-matrix-{tests.log,check.log}`.
Protected combined source7962fe68 / SHA76d5f706 passes all53 actual matrix,
router and dequant GPU controls, cold and repeated short requests, complete
45/4245/4245-token requests, actual erasure/close and physical recovery. However,
its clean long prefixes40.0054/38.1505s regress materially from the coherent
shared-buffer baseline6dfaf5ef's25.0353/24.8817s. The extra registers are a
plausible pressure/occupancy cause, not a measured spill diagnosis. Receipt:
`/media/jmicko/faststore/kuno-development/prefill-short-scope-clean-20261019/receipt.json`.

The forward selection restores this module exactly to6dfaf5ef while retaining
the working bounded short-prefix scope/retained-state fixes. This keeps the full
coupled K64/vector/paired-shared-buffer implementation; it removes only the
additional register-first successor. Numerical references, original owners,
runtime guards, model/context and API remain unchanged. The exact recombined
binary still needs its own complete acceptance before installation.


## Fixed-stage unrolling and full-tile loads — 2026-10-20

Candidate based on the installed selected K64 pipeline at032e2cdd. This keeps
its complete32x64 tile, four ordered F32 accumulators per SIMD16 subgroup,
paired operand stages,32KiB shared envelope, true vec4 FP16/BF16 bindings,
original output layouts and all finite/fault/owner/status checks. It does not
restore the slower register-first successor, alter the small-shape fallback,
retain request content, or change any API or model representation.

Pinned Strata calls the same logical expert products: gate/up
`(ne,1280,2560)` and down `(ne,2560,640)` in
`sycl/src/prefill/prefill.cpp:3978-3983`. Its oneMKL wrapper uses the original
K-contiguous weights as transposed column-major A and K-contiguous input as
nontransposed B (`sycl/src/prefill/gemm.dp.cpp:799-805`), with F32 accumulation.
There is no missing caller-side transpose or different matrix size to copy.
Kuno's down-status write already occurs only for output feature zero per route;
it is not a whole-output duplicated store.

The actual selected Kuno macro SPIR-V still contained seven runtime loops:
initial A/B staging, next A/B staging, ordered K16 math, outer K64, and output.
Only four static cooperative products appeared inside the K16 loop. The new
coherent pipeline:

- Statically expands the four A and eight B per-lane load/store jobs. Each pair
  is published immediately to its original SLM stage; there are no extra local
  arrays holding the next operands across matrix execution.
- Selects full A/B stage bounds once with workgroup-uniform conditions. Full
  stages use authentic unchecked-within-owner vector loads, while partial token,
  feature and K tails retain the same initialized-zero masked fallback. Both
  eight-half vectors are fully in bounds because reduction sizes are multiples
  of16. No guard or binding extent is weakened.
- Statically expands all four K16 groups in increasing order, exposing sixteen
  matrix products together without adding accumulators or rearranging sums.
  The original uniform barrier and ping-pong stage switch remain unchanged.

Actual production-module CPU validation passes all64 small/macro,
FP16/BF16/dense/expert device-layout variants, the pinned Vulkan SPIR-V validator,
and the SIMD16 selector. The three focused tests pass; they additionally require
exactly two remaining macro loops, sixteen static matrix products, exactly three
workgroup barrier instructions, genuine8-byte vec4 storage, exactly32KiB shared
allocation, no Function-storage operand arrays, and independent tail/ping-pong
ownership plus full-vector bounds. Strict production-module Clippy passes.
Scratch evidence: `/tmp/kuno-prefill-matrix-throughput-20261020/`.

For mask0 FP16 dense, actual SPIR-V size grows8656 ->31632 bytes. More static
instructions can increase instruction-cache/register pressure; the absence of
Function arrays does not prove physical spill absence. This is a source-backed
throughput candidate, not a claimed speedup. Root must validate original-weight
numerics, representative expert products, whole prompts, actual erasure/close,
physical recovery and matched performance before selecting or installing it.
No GPU, sudo, service, shared-target build or root-checkout mutation occurred
while preparing this candidate.

## Compiled-public expert matrix microtiming helper

`strata_matrix::public_timing(client, arena, file)` is a separate diagnostic,
not an extension of the existing complete numerical-case count. Root owns its
compiled-public CLI hook. It reads only expert zero from one original gate/up
pair and one original down tensor, decodes with the original GGUF dtype,
rounds to the real FP16 images, and interleaves GU exactly like prompt inference.
It never loads a complete Session or transfers the model. Inputs are compiled
dyadic values; there is no caller text, seed/preset input or private-wire flag.

The helper runs the unchanged `native::run_dense` primitive at actual
GU1280x2560 and DOWN2560x640 shapes with counts8/16/31/32/64/80/160/320. Each
shape warms three launches, checks results, queues64 measured launches and
waits for one genuine completion fence. Reports distinguish host enqueue from
complete duration, include per-call FLOPs and effective GFLOPs/s, and explicitly
state that host dispatch is included. Setup, upload, reading, oracle calculation,
erasure and compilation warmup are outside the timed region. These are repeated
same-buffer kernel measurements, not full prompt or whole expert-pipeline rates.

Before and after timing it scans the entire output owner for finite live values,
zero padding and zero full-owner fault words, then compares token/feature boundary
samples against the unchanged F64-product/F32-result IEEE oracle and original
1e-4+3e-4*abs(reference) tolerance. The reports explicitly label sampled dots;
they cannot replace complete original-weight numerical acceptance. Original
public model weight owners remain live until each private arena reset completes.
Every per-shape error attempts real reset before returning; caller retains
responsibility for final owning close/audit and acceptance before publishing.

CPU-only preparation typechecks the actual helper and actual modified native
module against existing real CubeCL, GGUF, half, JSON and zeroize rlibs. Non-run
arena/readback metadata siblings stand in only for typechecking; this is not
runtime owner/erasure acceptance. Five tests pass (the three actual matrix
compiler/tail checks and two pure sampled-oracle/full-output checks). Strict
standalone production helper Clippy passes. Artifacts:
`/tmp/kuno-prefill-matrix-throughput-20261020/timing-{check.rs,tests.log}`.
No GPU work or shared build was performed for this helper.
