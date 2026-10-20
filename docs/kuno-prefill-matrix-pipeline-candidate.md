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

### Opt-in routed expert continuation

The same optional CubeK feature now also provides `cubek::try_expert(client,
[input,weight,output,map,status,fault],count,columns,rows,weight_first,group,down)`.
This function returns the **actual enqueue** result, with `false` preserving the
original small/unaligned primitive. The integration owner must call it only
inside the existing `strata_matrix::expert` admission, before the original
native fallback. That entry/admission and its existing leases are unchanged in
this commit; the root task owns that wiring.

GU is the actual2560-to1280 product and writes member-contiguous original GU.
DOWN is the actual640-to2560 product and scatters each member to its original
route in the complete PART owner. Custom public `MatmulArgs` and write-only
CubeCL views integrate these stores into the unchanged complete cyclic GEMM;
there is no temporary result owner, copy, readback or additional runtime
kernel. The output view checks the genuine computed F32vec4 SSA type before
scalar publication. Unsupported accumulator reads, raw linear slices and TMA
access fail compilation rather than pretending a routed image is contiguous.

The original map header/group/member/route bounds precede every actual mapped
write. Finite values are unchanged; nonfinite values become positive zero and
set original fault bit128. Only the unique vector containing feature zero
updates DOWN's original `ROUTES+route` status plane, using the original
feature-zero-finite AND incoming-status-equals-one condition. Other invalid
features still set fault128 without changing this status rule. Real original
six handles, exact complete GU/PART/map/status extents, descriptor offsets,
same-stream identity and exact FP16-toF32 CMMA capability are checked. Existing
native top-level owner/phase/context validation, request erasure and completion
remain authoritative. Kernel mapping comes from the actual prepared library
`cube_count_plan`; it is never defaulted or fabricated.

Separate public acceptance is exported as
`cubek::public_expert_numeric(&client,&mut arena)`. It requires an already
selected **empty original public-audit arena** and the identical client; it
cannot promote a private request arena to public access. Root should budget
at least512MiB for this call because PART itself is400MiB. It returns14
additional static/numeric reports, without replacing the prior97 controls:

* Both actual GU/DOWN at counts32,33,64,257, including complete and partial M.
* Both products with nonfinite feature0, nonfinite feature4, and one deliberate
  invalid-route trip after authentic fixed-public group construction/validation.
* The original grouping builder/validator constructs a full4096-row map. The
  selected group has a nonzero member offset and sorted, noncontiguous routes
  above20479. Weights use a genuine descriptor-aligned128-half-element prefix.
* Every output, status, fault and source owner word is checked, including
  inactive routes, complete owning padding, unchanged source prefixes, valid/
  invalid incoming statuses and feature-zero-only propagation. The independent
  full F64 oracle uses the unchanged `1e-4+3e-4*abs(reference)` threshold and
  original FP16 values, including an intentional positive subnormal row.

PART inspection uses genuine16MiB views inside the original live full owner,
not a monolithic400MiB readback or an enlarged allocation claim. Every mapped
lease is wiped on successful or failed checks, dropped and joined through the
actual runtime I/O sanitizer before the next window. Each case resets/drains/
erases the real arena before its report can be returned. The enclosing
protected client's explicit close/audit and physical recovery remain required.
These diagnostics have no caller-controlled text, prompt persistence or
private-worker logging.

CPU preparation uses the same isolated artifact target, two build jobs, no GPU
or services. The real routed CubeK entry compiled in32 variants (GU/DOWN,
full32/tail33, eight physical layout modes), passed the original FP16 preserve
and SIMD16 selection, and passed Vulkan1.3 SPIR-V validation. Exact six storage
owners, <=32KiB real shared memory, true16x4x1 geometry, native products and no
relaxed-math or physical invocation-ID dependency are checked from the compiled
modules. Actual GPU numerical/status/fault/erasure acceptance and useful
performance are still root-owned gates; this source preparation claims neither.

### Complete Tilewise alternative, after the cyclic artifact

`strata-cubek-tilewise` selects the pinned library's complete
`TilewiseDoubleBufferingAlgorithm` through the shared algorithm alias. Both
dense projections and the routed GU/DOWN adapter therefore use the same new
family together. Without that additional opt-in, the complete cyclic artifact
remains available. The root-owned build selector requires explicit Intel/core
and CubeK selection; no installed/private runtime flag or API is added.

This changes actual per-plane loading **and** operand stage layout, while
retaining M32xN64xK64, original fixed types, ordered K16 reductions, checked
physical layouts/owners, finite/status stores, mapping normalization and the
same32KiB bound. The library uses row-major tilewise LHS and column-major
tilewise RHS loaders. A plane owns a complete tile's loads; RHS K tiles for a
feature tile are grouped in shared memory rather than the cyclic family's
row-major tile ordering. Both families already vectorize contiguous K values
inside each tile; the cyclic version must not be described as simply having
uncoalesced scalar weight reads.

For this fixed blueprint, A and B each have16 tiles per stage, divisible by
four actual loading planes. A has32 vec4s/tile and B64; each plane therefore
has a complete integral8/16-task workload. Global paired operand stages,
partial reader bounds, event-listener scheduling, stage/fragments and the
original F32 plane writer are all library implementations. No naive/minimal
calibration path, larger-M occupancy change, split-K, relaxed arithmetic or
extra result owner is introduced. CPU compiled checks exercise the actual
selected family in32 dense plus32 expert layout/full/tail modules and print
the measured shared-memory sizes only in those fixed-public test binaries.
GPU numerical/lifecycle acceptance and whole-workload performance remain
required before selection; source layout differences are a hypothesis.

The actual Tilewise source passes all64 compiled-module controls above and
six focused source/admission/IR/fixture/window tests in the full worker crate.
Both the dense and expert compiled inventories report **26624B** shared in all
eight layout modes, below the unchanged32768B admission cap. This includes the
library's actual fragment-sized publication buffer; an assumed whole-output
tile should not be added when estimating a future blueprint. The public expert
probe alias is crate-visible so the root matrix module may directly re-export
it; its original public-arena guard and private/request silence are unchanged.
Evidence is under the isolated `prefill-cubek-cpu-20261020` artifact directory:
`tilewise-whole-spirv.log` and `tilewise-controls.log`. No GPU speed result is
claimed by these CPU controls.

### Separate bounded M64 Tilewise candidate

`strata-cubek-m64` is a separate opt-in, depending on the complete Tilewise
feature. One shared host `MatrixPlan`/scheme builder controls both dense and
routed-expert launches. It selects M64xN64xK64 only when **count>=256** and the
**actual client's per-workgroup shared limit is>=34816B**. Otherwise the full
M32 Tilewise path remains. Small/unaligned native fallbacks and all original
owner, descriptor, stream, precision, finite/status, context and erasure checks
remain unchanged. Neither a private API setting nor an installed default is
added by this source candidate.

This blueprint uses partition8x1x4, stage1x4x1 and the same true16x4x1 launch
geometry. It retains cooperative8x16x16 FP16/BF16 products, F32 accumulators and
ordered K16 reductions, without split-K. Each subgroup has eight accumulator
tiles instead of four; the per-lane-equivalent F32 accumulator storage is64
values instead of32. These are semantic tile counts, **not measured Intel GRF
allocation**. For sufficiently large groups it increases operand reuse across
M and the independent work available within a subgroup. Count65 would pad128
rows rather than M32's96, so applying M64 indiscriminately to small groups would
add unnecessary work; the explicit256 threshold avoids that case.

Actual source calibration compiled FP16/BF16 dense and genuine GU/DOWN variants
for both M32/M64, in default and fully explicit layouts (16 modules). Every M64
variant measured **34816B** shared versus M32's26624B and128 static cooperative
product statements versus64 in the complete expanded modules. Those statement
counts include loop/final-path bodies; they are not a per-inference operation
count or throughput measurement. The library's pure resource planner also
confirms four actual planes for each blueprint.

Root's read-only physical snapshot
`/media/jmicko/faststore/kuno-development/native-shared-limit-snapshot-20261020.json`
reports49152B `maxComputeSharedMemorySize` for BMG G31. Vulkan adapter mapping
exposes this as `max_compute_workgroup_storage_size`; `strata_worker` requests
complete admitted adapter limits, and CubeCL's actual shared-memory property
comes from the resulting device. The existing global32912B minimum alone would
not prove M64 admission, which is why the selector separately checks34816B.
**49152B is a per-workgroup limit, not the total SLM pool of a core/subslice.**
It does not establish the number of resident workgroups. Actual SLM allocation
buckets, native registers and occupancy remain unmeasured.

The separate final CPU inventory compiles full256 and tail257 variants for
FP16/BF16 dense and GU/DOWN across all eight physical layouts (64 modules),
checking the real shared envelope, native product count, actual four-plane
resource plan, true geometry, subgroup-only mapping, unchanged native IEEE
controls and Vulkan1.3 validation. Existing M32 compiler controls and original
GPU acceptance are preserved. The source-only artifacts are
`m64-source-measurement.log`, `m64-whole-spirv.log` and `m64-controls.log` under
`prefill-cubek-cpu-20261020`. Root must complete original numerical/lifecycle
controls and compare the whole coupled workload before judging performance.

The M64 final64-module inventory passes all checks above (two CPU shader-test
parts,224.60s total wall time), with34816B and128 static cooperative products in
every actual module. Seven focused original/selector/fixture/window controls
also pass. This is CPU compiler/source acceptance; no M64 GPU numeric or speed
result is claimed. The source keeps the already tested M32 behavior unless the
additional feature, count boundary and actual device capability all permit it.

### Complete balanced Tilewise partition pipeline

`strata-cubek-balanced` is a further explicit Tilewise opt-in. It changes both
the subgroup partition geometry and the library partition buffering together,
for dense projections and genuine routed GU/DOWN. M32 uses partition2x2x4 and
stage2x2x1; M64 uses partition4x2x4 and stage2x2x1. Both still launch the genuine
16x4x1 geometry, four subgroups, and the original M32/M64 by N64 by K64 problem
blocks. The balanced partition keeps two N fragments live with the library's
`PartitionBuffering::Double`; the existing unbalanced alternative retains its
valid single-N-fragment `Single` pipeline. No hand-written partial substitute
for the global/stage/output library pipeline is introduced.

For M32 this matches the accepted native subgroup geometry: two M partitions
by two N partitions, each with two A fragments, two B fragments and four F32
accumulators per K16. The previous 1M by4N library geometry held four A and one
B fragment for those four accumulators. M64 keeps the same balanced N geometry
and doubles its M fragments/accumulators. This changes fragment reuse and
register blocking; it does not claim measured native GRF allocation, occupancy
or a guaranteed speed gain. The selected library already has one workgroup
synchronization per completed K64 stage, not an extra barrier per K16: its
`must_sync_plane_after_execution` setting is false.

One shared production plan/scheme/buffering helper now supplies both actual
launches and their compiled controls. M64 still requires count>=256 and the
actual client shared limit>=34816B; otherwise M32 remains selected. Original
physical layouts, vec4 alignment, scoped owners, finite/fault/status behavior,
FP16/BF16 to F32 arithmetic, ascending K16 reductions, subgroup builtin mapping,
private-request silence and erasure checks are retained. There is no split-K,
new scratch result owner, model/context change, API setting or tolerance change.

The final CPU-only acceptance batch is
`prefill-cubek-cpu-20261020/balanced-double-whole-controls.log`. It compiles the
actual selected dense and expert modules, including full and tail M32/M64
cases across all eight storage-layout modes, with original native IEEE and
Vulkan1.3 validation. It also runs the existing focused source/fixture/window
controls and actual resource-planner checks. Protected GPU numerical, lifecycle
and complete-workload acceptance remains the root task's next gate.

The final coherent balanced+Double batch passes all **128 actual modules and
seven focused controls** (11 tests, 322.93s CPU-only wall time). Every compiled
M32 module measures **26624B** shared; every M64 module measures **34816B**,
with the original64/128 static cooperative-product counts and four genuine
resource-planner planes. The preliminary balanced Single-buffer log was
intentionally stopped when Double buffering was approved; it is incomplete
and is not the acceptance artifact. No GPU numeric or speed result is claimed
by the final CPU acceptance.

### Separate complete wide dense Tilewise candidate

`strata-cubek-wide` is an additional dense-only opt-in. With count>=512 and
the actual client's shared limit>=36864B, at least128 workgroup invocations
and dimension bounds covering16x8x1, it selects a complete M128xN128xK32
Tilewise pipeline. Otherwise the existing M32/M64 dense plan remains. Routed
GU/DOWN always retain their existing genuine four-plane plan. The original
outer admission also retains its128-lane finite-guard requirements; the wide
selection does not substitute smaller/fabricated device limits. Actual library
prepare must return exactly16x8x1 and the original fixed dtypes before launch.

The forced blueprint uses partition4x4x2, stage4x2x1, true SIMD16 and eight
actual library compute planes, together with Double RHS partition buffering.
It includes the library's complete paired global operand stages, tilewise
loaders, stage matmul, original output and separate finite-output guard.
No extra temporary output, quantization/layout conversion, split-K or reduction
reordering is introduced. Two successive ordered K16 products per K32 stage
retain the original FP16/BF16 to F32 arithmetic contract. Descriptor alignment,
full owner bounds, stream identity and fault handling are unchanged.

The input operand stages require32768B and genuine eight-plane publication
requires4096B, for a compiled total36864B. Each subgroup owns16 accumulator
tiles instead of M32's four, or128 F32 lane-equivalent values. This is a logical
fragment count, not measured Intel native GRF allocation, occupancy or speed.
The larger M/N tile reuses operands across more output products, while the
K32 paired stages keep the actual per-workgroup shared footprint bounded.

The existing native selector already admits original SPIR-V1.6 LocalSize
16x8x1 with the known subgroup-scoped8x16x16 matrix types, actual subgroup
lane/ID builtins, and required SIMD16. No selector override or physical feature
is widened for this candidate. Root's separate physical query reports the B70
maxComputeWorkgroupSubgroups=64, maxWorkgroupInvocations=1024 and per-workgroup
shared limit49152B, with required SIMD16 compute support. These real properties
do not imply a specific number of resident workgroups.

The isolated CPU artifact is
`prefill-cubek-cpu-20261020/wide-whole-controls.log`. Dedicated controls compile
the actual selected full512/tail513 entries, both FP16/BF16 and all eight
physical storage-layout modes (32 modules), checking actual eight-plane resource
planning,16x8x1 entry geometry, genuine subgroup mapping, shared size, preserved
native IEEE controls, absence of relaxed arithmetic and Vulkan1.3 validation.
The pure selection control covers count/capability boundaries and verifies
that unavailable wide geometry falls back to the original plan. GPU numeric
acceptance must additionally preserve all prior controls and add the four
actual512/513-by128-output cases, both original dtypes, before whole-workload
performance is judged. No GPU result is claimed by this source candidate.

All32 actual wide modules and the boundary/real-geometry selection control
pass (three tests,32.62s CPU-only wall time). Every module reports exactly
36864B shared and128 static cooperative product statements, with the original
finite guard still a separate unchanged kernel. The actual8-plane resource
planner and original16x8x1 entry/subgroup/IEEE/Vulkan checks pass in every full
and tail layout. Existing M32/M64 compiler controls remain in place; protected
GPU numerical/lifecycle and full-model timing gates are still required.

### Separate native direct-CMMA full-tile wide candidate

`strata-native-wide` is separate from the experimental CubeK family; the build
selector rejects combining their flags. The new module
`strata_matrix_native_wide.rs` implements a complete direct-CMMA dense kernel
using the existing CubeCL Rust frontend and existing native Vulkan selector.
Mapped experts remain on the accepted native paths. A returned `true` from
`try_dense` means this actual kernel was launched; root-owned private/public
wiring must use that returned result rather than infer execution from a feature.

Selection requires count>=512, count and output-feature count divisible by128,
nonzero K divisible by32, aligned genuine vec4 FP16/BF16 operand views, actual
shared memory>=40960B and a real256-invocation256x1x1 workgroup capability.
Actual SIMD16 must lie in the device's subgroup range and the original
8x16x16 FP16/BF16-to-F32 CMMA feature must be present. Existing fixed-subgroup
native pipeline/device admission remains in force. All other geometry, including
count513 and320-feature tails, returns `false` for the original dense fallback.
Logical owner lengths, source/output offsets, stride, shared stream, exact
four-byte fault owner, complete output extent and U32 index/grid bounds are
checked before launch. No request content or new runtime state is retained.

The true256-thread workgroup has sixteen16-lane subgroups, arranged4M by4N.
Each subgroup owns32x32 results, fed by four A and two B fragments into eight
explicit F32 accumulators. This retains64 F32 lane-equivalent accumulator
values, rather than the experimental eight-subgroup library wide kernel's128.
These are semantic fragment counts, not measured Intel GRF use or occupancy.
The complete workgroup computes M128xN128xK32 blocks with successive ordered
K16 products, preserving the original FP16/BF16 and F32 arithmetic contract.

Each real typed operand allocation contains two128x32 half stages (16KiB),
for32KiB of paired operand storage. Every full stage uses256 threads with two
fixed jobs each, publishing pairs of genuine vec4 loads immediately to SLM.
The next stage is issued before current-stage arithmetic, following the
accepted native ordering without a large register-first payload or a claimed
asynchronous DMA guarantee. Uniform workgroup barriers protect initial
publication and every stage switch.

One separate real8KiB scratch allocation holds one8x16 F32 fragment for every
subgroup. Eight explicit publication phases reuse it for the eight accumulators;
each phase is cooperative store, uniform workgroup barrier, unique finite output
writes, then another uniform barrier before reuse. Original finite values are
written unchanged; nonfinite values become zero and set the original fault bit1.
No whole128x128 F32 scratch result, output copy, operand-free alias assumption,
new dtype/layout conversion, split-K or tolerance change is introduced. The
actual expected per-workgroup allocation is40KiB, below the separately observed
B70 per-workgroup49,152B cap; this does not establish hardware occupancy.

The isolated CPU acceptance artifact is
`prefill-cubek-cpu-20261020/native-wide-whole-controls.log`. There are sixteen
distinct compiled modules (FP16/BF16 times eight physical storage-layout modes);
generic runtime512/1024 shape values do not create different shader modules.
Controls check actual256x1x1/required16 metadata, subgroup-ID/lane builtins,
real shared allocations, native IEEE preservation, original finite fault,
cooperative instruction/publication/barrier structure and Vulkan1.3 validation.
Separate pure controls cover full-tile/capability selection and tail fallback,
complete original owner bounds/offsets and the full-stage/output address
bijections. Root must preserve the original GPU numerical controls and add
actual full-tile512 and rejected513-tail cases before any performance judgment.

The full distinct-module and three pure-control batch passes (four tests,7.45s
CPU shader-test wall time). Every actual module reports40,960B shared,16 static
cooperative product statements,12 cooperative loads,8 fragment stores and18
workgroup barrier statements, matching the explicit phase structure. This is
static compiler inventory, not per-inference event counts or throughput. A
subsequent focused host-view control passes with extra genuine owner-offset,
stream-mismatch and scalar-view-alignment cases. Valid scalar-only operand
views return to the original fallback; no invalid vector pointer is fabricated.
No GPU correctness or speed result is claimed here. Root owns the minimal
feature-gated module/actual launch wiring and protected numerical acceptance.
