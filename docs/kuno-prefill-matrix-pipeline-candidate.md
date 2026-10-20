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
