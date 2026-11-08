# Coalesced original expert dequantization

Optional internal candidate on top of complete de309ebf. No API, model,
context, private persistence, owner, capacity, routing, release, or Vashti
change. This source is not yet built, accepted on hardware, or timed.

The current original32-value decoder prepares a genuine encoded record once
then emits four sequential vector8 stores from one invocation. Adjacent lanes
therefore publish different32-value records during each quarter-store wave.
The selected Strata IQ GU path distributes256 values over32 physical lanes
(`sycl/src/kernels/cuda/iq_kernels.dp.cpp:2725–2773,4390–4417`, pinned MIT
fb58e0dbc8399662c0e47c76578c6e878b14f6cf). This candidate retains Kuno's
original exact decoder and shares preparation across four physical lanes,
each publishing its own adjacent8-value quarter.

`strata-dequant-quad32` must require `strata-dequant-record32` and
`strata-intel-core`; the latter requests and enables the actual SUBGROUP device
feature. No native Quad capability or local-invocation lane ordering is assumed.
The source uses the genuine subgroup ID, invocation ID, and width. Record
number is `wg.x*64 + subgroup*(width/4) + lane/4`; quarter is `lane%4`.
Queried client bounds and a shader width guard require complete four-lane
quads. Shader width must be4..128, divide256, be divisible by4, and its product
with the actual subgroup count must be256. All valid
and inactive final lanes reach every shuffle. Only an in-range record's
quarter0 lane reads its authenticated quantized record; inactive quads return
only after sharing initialized zero metadata.

Original GU/down physical roots, source tables, map/header checks, whole image
and exact per-role view guards, mixed HOST/resident handling, and all image
offsets remain. Metadata uses integer shuffles, including bit-cast FP32 scale
bits. The separate IQ2_S16-value factors and original packed code helpers are
unchanged. Each of four lanes calls the exact original `store_original8`, so
all F32 scale products, original FP16 round points, finite/overflow checks,
per-vector zero/fault behavior, and aligned16-byte stores stay unchanged.
There is no workgroup scratch, barrier, new owner, or new allocation.

Runtime source and every legacy/cohort/slot/numeric-probe grid change together:
8 values per invocation. The existing record32 source, its tests, and the
original429-shader offline population are retained separately and unchanged.
A distinct `experts::validate_quad_sources()` validates24 regular/resident/
slot modules plus6 cohort modules through the same physical-source SPIR-V
factory used by actual native inference, with standalone SPIR-V validation.
Each emits `strata_dequant_quad32_shader_accepted`; these are additional proof
rows, never replacements for original shader counts. Existing complete-owner,
original-decoder six-format GPU oracle controls use the selected runtime source.

Two new CPU controls cover full and partial physical-quad mapping/geometry,
all24 source variants, real builtin mapping, original store-body equality,
original source/owner guards, leader-only preparation, and shuffle-before-tail
exit. These tests have not been run in the agent worktree. Root owns all
compilation, GPU correctness, numerical/recovery gates, and composed timing.
The new public-only counter `expert_dequant_quad32_records` counts the same
logical records as retained `expert_dequant_record32_records`; it adds no
private timer or storage. No performance claim precedes composed acceptance.
