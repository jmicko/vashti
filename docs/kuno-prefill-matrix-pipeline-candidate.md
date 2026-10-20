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
GU1280x2560 and DOWN2560x640 shapes with counts1/3/7/8/16/31/32/64/80/160/320. Each
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

## Narrow expert K64 pipelines — 2026-10-20

The protected public4245-token histogram finds26673 active expert groups:
12221 with1–7 members,3883 with8–15,2850 with16–31,2109 with32–63,
1690 with64–127,1576 with128–255 and2344 with256+. Almost46% of calls
therefore always used the old masked scalar/SLM product, with two barriers
perK16. The original full-tile microtiming misses this when it starts at8.
Fixed-public microtiming now includes1/3/7; root's original small baseline
measures GU682/372/415us at1/3/7 against60us at8, and411us at31. These are
primitive timings, not prompt rates; fixed per-group dequant costs remain.

This coherent candidate adds two separately compiled matrix shapes:

- **M8xN64**, selected for1–8 members: four SIMD16 subgroups each own one8x16
  accumulator. Paired A[8][64]/B[64][64] operands plus the original F32 output
  scratch occupy exactly20480B. A single-row expert retains eight padded matrix
  rows; it never runs a32-row/four-accumulator product.
- **M16xN64**, selected for9–31 members: each subgroup owns two vertically
  adjacent8x16 accumulators. Paired operands plus output occupy24576B. Counts17–24
  round to32 rather than the old24 rows; improved staging/reuse must outweigh
  that bounded padding in actual performance acceptance.

Both shapes use the whole vectorized K64 chain: genuine aligned vec4 half/BF16
loads, immediate publication, full-stage uniform fast loads with initialized-zero
masked tails, separate ping-pong stages, four increasing K16 groups, and one
uniform handoff perK64. A is shared across all four feature subgroups; B covers
their disjoint feature ranges. Output stores, original row maps, finite/fault
bits and the full-capacity down-status plane are unchanged. The compile-time
M8 variant emits only one actual F32 accumulator variable; M16 emits two.

Counts>=32 retain the selected M32 stage/product/epilogues/builders verbatim.
The original scalar-half fallback remains for unsupported token counts, feature
geometry, true vector owner alignment/lengths or actual device shared limits.
There are no forged descriptors/capabilities, changed quantization/context,
relaxed numerical tolerances, request retention or API changes.

The actual compiler/offline inventory grows from64 to128 modules (16 definitions
across8 layouts). It checks authentic8-byte storage, SIMD16, exact20/24/32KiB
shared allocation, exactly three barrier instructions, two loops, and4/8/16
ordered native matrix products with1/2/4 F32 accumulator variables. Additional
CPU controls cover admission boundaries, every output exactly once, each full
vector within the real image, all token/feature/K tails, disjoint paired slots
and the original increasing K16 order. The compiled down-status test now covers
both new variants as well as the original scalar/M32 versions.

This is a complete source candidate awaiting root's original-weight GPU controls,
whole prompts, actual erasure/close, physical recovery and matched timings.
Source-only tests cannot establish numerical or performance acceptance. The
existing numeric probe file is untouched; root is extending its independent
full IEEE cases at tiny and medium counts. No GPU, sudo, service or shared build
was used while preparing the narrow kernels.

## Authentic shared-memory lifetime reuse — 2026-10-20

After the final uniform K64 handoff barrier, no A/B slice is read again; native
matrix fragments hold values rather than operand pointers. The next candidate
ends those real typed shared lifetimes with `SharedMemory::free`, then allocates
the unchanged F32 result scratch. It preserves tile selection, every load/store,
ordered matrix operation, accumulator count, finite/status/fault check, and all
three existing barriers. It does not halve K or reinstate register prefetch.

This uses the actual supported compiler path. CubeCL's SharedLiveness analysis
keeps shared variables live until an explicit Free marker. The SPIR-V backend
then assigns disjoint offsets to simultaneously live objects. With the real
`VK_KHR_workgroup_memory_explicit_layout` feature, it emits truthful Aliased
Workgroup blocks and reuses offsets for non-overlapping lifetimes. No typed
pointer/extent, device capability, owner or physical alias metadata is forged.
CubeCL selects this only when the authenticated Vulkan feature is enabled;
implicit-layout devices still declare independent original arrays.

| Shape | Original/implicit allocation | Actual explicit-lifetime allocation |
| --- | ---: | ---: |
| M32xN64xK64 | 32768B | 24576B |
| M16xN64xK64 | 24576B | 20480B |
| M8xN64xK64 | 20480B | 18432B |

Host admission retains the larger conservative shape limits. It therefore
cannot admit an unsupported device by assuming alias support, while the backend
receives the compiler's actual smaller allocation on supported devices. Offline
expectations now depend on the genuine explicit-layout flag; every original
implicit-layout configuration remains accepted at its original allocation.

CPU coverage inspects all three real Workgroup owners in every actual staged
module. In explicit mode it requires Aliased decorations, disjoint live A/B
ranges, and an output interval contained within an old operand interval. It then
builds the actual instruction control-flow graph, including loop backedges:
every operand-access-to-output-store path must cross a real Workgroup
AcquireRelease barrier, and no operand access may be reachable after output
publication. An actual-module negative control strips the handoff barriers and
must fail that check. Existing8-byte vector,2-loop,3-barrier,4/8/16-product,
1/2/4-accumulator, full-vector/tail and unchanged ownership checks remain.

Allocation reduction is not an occupancy or speed result. The Vulkan maximum is
a per-workgroup limit, and hardware allocation buckets, resident pool size and
GRF modes all affect concurrency. Intel's [occupancy guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html)
also warns that allocations can round up and that higher occupancy need not
improve performance. A later K32 pipeline could combine shorter stages with
these real lifetime boundaries, but is not implemented in this candidate.

No GPU/shared build/service operation was performed here. Root still owns full
GPU numerical, prompt, erasure/recovery and matched performance acceptance.

## Opt-in complete CubeK dense pipeline — 2026-10-20

`strata-cubek-gemm` adds a dense-only library candidate. It preserves the
selected original expert kernels and small/unaligned dense fallback. It does
not change the installed build, API, private worker protocol, context,
quantization or prompt retention. Root owns GPU acceptance and selection.

The dependency is the pinned registry `cubek-matmul=0.2.0` / `cubek-std=0.2.0`,
MIT OR Apache-2.0, compatible with the already pinned CubeCL0.10. Their actual
registry sources match the retained v0.2.0 audit sources. Only these two crates
and `cubecl-std=0.10.0` are added to the preserved laboratory lockfile; existing
package versions do not change. Dependencies are optional and this feature is
never enabled by default. There is no vendored CubeK fork.

The selected implementation is **CyclicDoubleBufferingAlgorithm**, not a
minimal Simple/naive fallback. Its complete public batch/global/stage/tile
pipeline includes paired operand stages, partial cyclic global readers,
event-listener scheduling, matrix fragments and the F32 writer. The forced
blueprint is:

* Cooperative tile8x16x16, original FP16 or BF16 inputs/stages, F32 accumulation
  and F32 output. No inferred flex32/TF32 conversion, autotuning or split-K.
* Partition4x1x4 and stage1x4x1: M32xN64xK64, four16-lane planes, four native
  accumulator tiles per plane, at most32768B actual compiled shared memory.
* Increasing naive K16 arithmetic order inside the complete double-buffered
  pipeline. `PartitionBuffering::Single` selects the one intra-partition RHS
  register fragment (partitionN=1); the global operand pipeline still has
  two actual stages. Reduction sizes must be divisible by128 so that this
  even-stage algorithm adds no extra padded K64 products.

The input remains the original row-major token image. The original physical
weight image `[feature][K]` is viewed as a column-major `[K][feature]` tensor;
there is no transpose, allocation or conversion. Output views retain the
original row stride. Views use checked arithmetic, real existing handles and
the authenticated descriptor alignment. Counts below32 or above4096, features
below64 or not divisible by16, unsupported reduction sizes, and unaligned
views use the original primitive. Actual per-axis workgroup, total invocation,
shared-memory and exact CMMA capabilities are checked before launch.

The library entry's expanded CubeCL IR gets a bounded, entry-specific topology
normalization: `UnitPosX` becomes `UnitPosPlane`, and `UnitPosY` becomes
`PlanePos`. Branch operands and nested scopes are visited, without changing
math, types, argument extents or order. Other local/absolute invocation
builtins, unreviewed operations and relaxed arithmetic flags fail compilation.
This retains actual physical LocalSize16x4x1 and does not pretend that a
flattened workgroup still has a useful Y coordinate. No other kernel receives
the normalization.

The Vulkan selector now admits only bounded SPIR-V1.6 true16xN matrix geometry
(N<=16), alongside its existing1D geometry and exact known cooperative matrix
types. Existing real subgroup-size-control device admission and required16
pipeline metadata remain. SPIR-V1.6 and X=required16 provide full-subgroup
completeness; the adapter independently avoids assuming any relationship
between local invocation IDs and subgroup IDs. It does not set the
`REQUIRE_FULL_SUBGROUPS` flag without enabling its separate feature. See the
[Khronos subgroup rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html)
and [pipeline requirements](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageCreateInfo.html).

After the library product, a same-stream finite epilogue examines each original
logical result exactly once, leaves finite values unchanged, replaces invalid
values with zero and sets the original dense fault bit1. Existing full-owner
leases, native admission, IEEE FP16 preservation, request erasure and close/
recovery mechanisms remain. No new temporary output or arbitrary expert-DOWN
scratch offset is introduced. Expert scatter/status processing is unchanged;
a future library expert path needs a genuine routing output implementation.
The finite epilogue uses a balanced true2D grid bounded by actual device group
counts, including large4096x10240 projection results. CubeCL's actual
`ABSOLUTE_POS` flattens these dimensions; excess invocations remain masked.
Grid padding must fit the real U32 index width. Its128-invocation block is
checked separately from the library's64-invocation block before either launch.

CPU preparation was isolated under
`/media/jmicko/faststore/kuno-development/prefill-cubek-cpu-20261020/target`,
with two build jobs and no device/service/sudo operations:

* Exact production adapter typecheck and strict production Clippy pass.
* Five focused controls cover actual compilation, unchanged fallback admission,
  genuine owner views/overflow/alignment, nested IR normalization, rejection of
  unsupported IDs, shader printing and relaxed math, and the actual guard-grid
  planner's large-output coverage, hardware/count/index bounds.
*32 actual library SPIR-V modules (FP16/BF16, complete/tail counts32/33, all
  eight physical layout combinations) pass the pinned validator for Vulkan1.3.
  They retain true16x4x1 geometry and SIMD16 selection, real shared allocation
  <=32768B, native matrix products and no FPFastMathMode. The actual modules
  contain subgroup ID/lane builtins and no local/absolute invocation builtins.
* Nine actual HAL selector controls pass, including original1D guards,
  malformed/conflicting modules, and new2D dimension/version boundaries.

These are compilation/admission results. No numerical GPU result, prompt
duration or speedup is claimed. Root must run original full-owner FP16/BF16
numeric controls, private fault/lifecycle controls, the matched public prompts,
then actual erasure/close and physical recovery before selection. Compare the
whole coupled library pipeline; do not judge an isolated staging subcomponent.
