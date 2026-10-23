# Fused quantized small-expert matrices

## Driver loop retention (next source, no acceptance yet)

Complete source131c8fa6 passes all720 GPU controls, both full-model chains and
physical recovery. Diagnostic first/repeat11.849637/9.832439s; host profile
11.845247/9.722708s. The sampled cohort matrix drops47.435354 to38.772952ms
versus K128, but six M32 programs still spill. Actual SPIR-V has an inner
four-trip product loop; final ISA has only staging/outer loops, indicating driver
unrolling. No clean run is claimed and the Strata goal remains unmet.

The authenticated native packing pass now retains atomic-free product loops
with SPIR-V DontUnroll. It traverses actual CFG edges up to each real merge;
loops containing the original fault atomics and all non-product loops keep
their original controls. On genuine modules this selects only M32's inner
loop. Arithmetic, loop bounds, CFG edges, descriptors, source/owner/finite
guards and original K16 order are unchanged. Both HAL and native replay use
this existing shared pass. Public half-edge entry is excluded. Conflicting
controls, missing blocks and unsupported merges fail closed; the pass is
idempotent on cached native modules. No new feature or inference API is added.

Two additive CPU cases compile all18 genuine matrix definitions, check exactly
one retained loop for M32 and zero for M8/M16, and verify only loop-control words
change on already packed modules. Conflicting Unroll and an unknown genuine
CFG edge must fail. All144 real compiler layouts must prove the retained-loop
count while preserving every original operation and compiler population.
Actual driver ISA, exact GPU comparisons and full-model timing remain required.

## Current allocation policy after compiler inspection

Both source55573ec0 and header-uniform sourceddeafd30 compile B and scratch
as separate allocations:26112B in every layout. The latter passes353 CPU,
18/47 runtime and8+9 metadata. Its offline rejection is the attempted24064B
reuse property, not an out-of-bounds or numerical failure. Preserve both failed
expectations and workers. No GPU has run since the accepted K128 build.

Use the actual26112B envelope for all layouts. This remains below the original
48KiB per-workgroup limit; the former inference that this limit also describes
per-core workgroup occupancy was not established. No occupancy claim is made.
Retain the padded producer/reader layout, global A, K64 stages, direct signs,
M32 loop and one final WG handoff as a complete bundle for measurement. Lifetime
release remains present, but actual publication alias must be reported false.
This changes the unachieved optimization expectation, not numerical tolerances,
source/owner bounds, finite/fault checks or any original GPU oracle.

## Complete K64 padded layout (initial reuse proposal, not GPU accepted)

Compiler checkpoint: source55573ec0 passes353 CPU and all runtime/metadata
controls, then fails at the first explicit layout: actual26112B, expected24064B.
The non-explicit72 layouts pass. Current header-continuation repair uses genuine
subgroup votes over the same immutable uniform map-cell guards, matching the
original native kernels, so lifetime analysis can see the accepted continuation.
No guard is removed and no allocation expectation is relaxed. No GPU has run
for either padded source or the unpadded K64 source.

Compose the K64/direct-sign revision below with padded B staging and matching
cooperative readers before live speed tests. The unpadded K64 sourcea97cee6c
passes352 CPU,18/47 runtime,8+9 metadata and every compiler population, but has
not run on the GPU. Do not claim its performance or hardware acceptance.

B uses pitch24 halves for each16-value column. Each K16 tile trims the last
column's unused padding, occupying376 halves; all tile starts and strides stay
16-byte aligned. Pitch20 would violate the cooperative-load alignment rule and
is rejected by design ([Vulkan08986](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986)).
Each tile's15 padding cells is initialized exactly once by the corresponding
real decoder job. Existing all-eight values/fault behavior remains unchanged.

Four tiles, two banks and four subgroups occupy24064 bytes. After the last K64
handoff, one genuine whole-workgroup barrier joins every subgroup before the B
lifetime ends. Only then may publication scratch alias B on explicit-layout
devices. The real compiler envelope is24064 bytes with explicit layout and
26112 bytes without it. The B70 supports the former, as established by original
shared-reuse GPU controls. Two such workgroups fit its48KiB shared limit; this
is a capacity calculation, not a measured occupancy or speed claim.

Native compiler proofs must inspect actual F16vec4 B pointers, stride6 in
vec4 units, original global A storage, chronological products, one full-WG
handoff and the real physical envelope in all144 layouts. Original subgroup
panel barriers remain; the new WG barrier is only for the cross-subgroup
publication alias. Old source guard changes from forbidding every WG barrier
to requiring exactly this single epilogue fence before free/scratch creation.
An additive independent mapping/padding/alignment oracle accompanies it.

## K64 occupancy and direct sign conversion (pending validation)

The accepted K128 version preserves exact results but is slower: diagnostic
repeat10.619382166s versus native-half K32 10.071408282s. Its first4096 chunk
is nearly unchanged, while the148 tail grows2.406616230 to2.884975162s.
Six M32 quant programs spill. Keep direct global input loads and the complete
encoded source/publication pipeline; correct its execution grain together:

- K64 double weight banks occupy16384 bytes plus2048 publication scratch.
  Each half subgroup decodes one contiguous64-column row, with eight rounds
  covering all16 rows. Original chronological K16 products remain unchanged.
- M8/M16 use four static K16 steps. M32 uses one genuine four-iteration loop
  to limit the lifetime of global input fragments and temporary addresses.
  Compiler proofs distinguish actual global A and shared B load counts in
  this loop from the two static smaller variants.
- IQ2/IQ3 retain magnitude bytes and the separate original sign mask until
  conversion, rather than packing signed bytes then unpacking those bytes.
  Signed arithmetic extension retains the original8-bit wrap for every byte,
  including magnitude128. FP32 factor association, native F16 rounding and
  the all-eight finite gate are unchanged. IQ4/Q2 keep their original path.

Original K32/K128 mapping oracles remain, with an additive K64/sign oracle;
all original real-format whole-owner GPU comparisons remain mandatory. No
new API, option, feature, model storage or runtime owner is introduced.

## K128 panels with direct input loads (source proposal, 2026-10-23)

The complete native-half bundle is numerically accepted but its sampled fused
matrix phase remains expensive. Replace its K32 input/weight shared staging
with K128 weight-only double banks and authentic global input matrix loads.
Gather and activation already initialize every padded input row, including the
unused members of the M8/M16/M32 tile. Existing narrow products use this same
global-input cooperative-load contract.

Each SG16 owns sixteen output rows. Its sixteen lanes decode contiguous
eight-value runs spanning K128 of one row, then advance through sixteen rows.
The four subgroup-private weight banks remain disjoint; each coefficient is
decoded once, with the original native packing and all-eight finite gate.
Two banks occupy 32768 bytes; publication scratch adds 2048 bytes, for a real
34816-byte maximum in every tile/layout. No allocation lifetime reuse is needed.
At each K128 stage, eight K16 products consume the existing accumulation order.
Subgroup barriers protect bank handoffs exactly as before. Compact producers,
RAW release, source ownership, padding, activation and ordered output do not
change. Whole-owner comparison against the original decoded-image products
must remain bit-exact across all six formats and all member boundaries.

This is a complete staging/consumer change; no hardware or speed acceptance is
claimed until the new compiled kernel and original complete pipeline run.

This optional internal bundle removes full decoded FP16 weight images from
groups with fewer than32 members. It changes no public API or model format.
The active pinned Strata Intel reference still creates those images; this is
a Kuno optimization after the reference execution port, not a missing Strata
cache. Current repeated public diagnostics produce186.3GB of small-group FP16
images out of262.2GB total for only6.78% of the useful expert members.
Those byte counts describe GPU image production, not PCIe traffic.

## Complete pipeline

Keep the original private80MiB working owner and six logical phases:

1. Original checked gather of expert input rows.
2. A genuine MAIN compute producer copies encoded gate/up roles into a compact
   cohort slab within the working owner.
3. A second genuine MAIN producer copies encoded DOWN. Its actual completion
   releases the RAW source; no future fused product reads that RAW bank.
4. A fused quantized GU matrix decodes weights into subgroup-private FP16 shared
   memory, then performs the original chronological K16 cooperative products.
5. The existing GU finite check and activation produces the original hidden
   FP16 values and status.
6. A fused quantized DOWN matrix publishes expert order for the existing
   pending-status/inverse/finite consumer.

Compact producers retain the original physical prototype/address/RAW/model
source bindings, loan-tier resolution, bounds and fault behavior. They copy
encoded bits without numerical conversion. Slab slots are logical cohort
ordinals; source RAW slots, including resident sentinel8, are separate values.
The new path must not weaken or reuse HOST-only copy admission for GPU sources.
Keeping compact producers on MAIN avoids blocking current work behind future
HOST uploads on COPY. Existing owner erasure and capacity accounting remain.

Products use four SG16 groups in64 threads, output N64 and M8/M16/M32 for
member counts1–8/9–16/17–31. This avoids decoding an expert's weights twice for
17–31 members. Each subgroup stages its own double K32 A/B banks with real
subgroup memory barriers. Original K16 accumulation order, explicit F16
rounding, eight-value nonfinite handling and checked partial publication must
be preserved. The largest planned shared envelope is26KiB including publication
scratch, below this device's48KiB limit without lifetime-reuse assumptions.
GU formats18/21/22/23 and DOWN formats20/42 cover every actual model layer.
All old decoder/image/product definitions remain as independent controls and
as the feature-off path.

## Integration and acceptance

Application feature `strata-small-quantized-matrix` composes existing small
publication, grouped experts, direct GU and ordered consumers. Runtime features
use `kuno-small-quantized-matrix`. Exact new producer identities and fused
matrix identities must form a closed grammar: compact producers cannot feed
old FP16-weight consumers, and old decoded images cannot feed the new kernels.
Retain original source/owner extents, RAW release, status/fault and whole-layer
completion checks. Counters must report actual compact bytes and launches;
do not continue counting removed FP16 image production as executed work.

Acceptance requires exact compact-copy controls from both real resident and
RAW sources, all six decoder formats against the original formulas and FP16
rounding, complete owner/status/fault checks, actual shader/layout validation,
and whole-model correctness plus physical recovery. Performance testing follows
the complete producer/scheduler/runtime composition. Preserve original677 GPU
controls and original CPU/runtime populations; no relaxed tolerance or speed
selection of partial pieces. Root owns integration/builds/GPUs; isolated source
tasks own kernels, cohort producers and native admission. No implementation or
performance acceptance is claimed by this design document.

## Composed source checkpoint — 2026-10-23

Frozen source `888d8b205071e98214cf3075fb7e4efd0ac07ea0`, worker
`a3192db31630a59cfff5713ff1bf303efcd4feec082f2d5840c23fb6d0922df3`,
composes compact producers7ac52735, runtime admissiondec5ef70,
root integration105716f3, original activation metadata repair362f5717,
matrix producersf1ea330a, typed shift repair80382bdf and constant-loop repair888d8b20.
All343 selected AppCPU controls pass, as do18 feature-off/47 full runtime,
8 original metadata and a separate9-test feature-enabled selector population.
The original compiler populations pass unchanged, plus144 fused matrix layouts
and6 compact-copy modules. Actual shared allocation14336/18432/26624 bytes fits
48KiB hardware in all8 compiler layouts. The6 original quant formats retain
explicit bit rounding, eight-value finite checks and original K16 ordering.

Protected GPU validation is running as `kuno-small-quant-20261023` against the
complete bundle, with swap/core dumps disabled. Planned population719 retains
all677 prior controls and adds30 compact-source plus12 original-format matrix
comparisons. This paragraph is not GPU acceptance or a performance result.
The two forward18-script chains preserve original numerical, ownership and
reuse assertions; their counters now explicitly describe packed bytes and
10-launch packed cohorts. Do not describe these counter checks as unchanged.
The original frozen harness and failed attempts remain intact.

Evidence under the development directory: `small-quant-root-build-receipt-20261023.json`,
`small-quant-root-acceptance-freeze-20261023.json`, `small-quant-harness-20261023/`,
`check-small-quant-connected-20261023.py` and `small-quant-connected-20261023.log`.
Attempt1 failed Rust constant visibility/shift types; attempt2 was an external
harness missing its feature declaration; attempt3 reached offline expansion and
caught mutation of an unrolled nibble-shift constant. All are preserved with
source/worker evidence. No GPU ran in those failed attempts.
No installation, API, model, context, Vashti or driver change. Goal remains unmet.

## First hardware result — 2026-10-23

Source888d8b20/worker a3192db3 failed the public IQ4XS finite matrix comparison
(dtype23), after the first six IQ3XXS/IQ3S/IQ2S finite/poison cases completed.
This is a numerical failure, not a speed result; no full-model request ran.
The frozen build and receipts are archived as development
`small-quant-attempt4-gpu-numeric-20261023/`; protected original operator output
remains `strata-parity-small-quant-20261023-operators/`. All original validation
requirements remain. The run exited1 after4m58.821s with zero swap; physical
recovery passed, and a separate live check confirms all driver faults0,
Intel50,688,000B, NVIDIA compute0 and all services/admission healthy.
Continue the complete bundle and fix the format-specific failure before timing.

## Correctness accepted; measured conversion bottleneck

Source4b567eef/worker4abf583d is identified exactly in the coordination handoff. It retains all719
GPU controls and both forward full-model chains with recovery. The IQ4 repair
uses explicit runtime selection and adds48 real SPIR-V selector checks; the
original failed NIR had lost the upper-nibble branch. No tolerance was changed.
Diagnostic repeated prompt12.609879799s (host12.532337098s) is slower than the
previous complete path; no clean measurement of this candidate is claimed.
Same request3/layer24/count4096 cohort matrix12.709929→87.376758ms isolates the
new cost. Dequant19.224670→8.000772ms improved, with3.648422ms compact copies.
All8 old wide ISA programs still have zero spills;6/18 new quant programs spill.

Next complete optimization preserves original encoded copies and publication:
reduce duplicate staged decoder expansion; replace scale normalization loops;
port original GLSL.std.450 PackHalf2x16/UnpackHalf2x16 rather than software-heavy
conversion, only through authenticated typed native shader lowering. Final
modules must preserve the original rounding/denormal and all8 finite behavior,
with independent original software/decoded-image controls. Arbitrary default
OpFConvert remains unacceptable. Public-only native pipeline timing is separate
from clean comparisons and must never activate or retain clocks in private use.
No new hardware/performance acceptance is claimed for those pending changes.


## Native conversion bundle — compiled, hardware validation pending

Sourcecbd8a5b9 uses the original GLSL.std.450 native half packing operations.
Typed CubeCL vec2 F32/F16 casts are only compiler input: the shared HAL/native
replay pass replaces them with PackHalf2x16/UnpackHalf2x16 and bitcasts before
driver pipeline creation. It recognizes only the genuine cooperative WG64/SG16
quantized entry and separately compiled public edge entry, with original complete
descriptor inventory. Unknown types, wrong widths or shapes fail closed. All
original source/owner/geometry/rounding/denormal admission remains. Manual offline
definitions now carry the same names as their generated runtime entries; no
alternate name or unvalidated fallback was admitted.

All finite values, signed zeros, half subnormals and boundary rounding remain
bit-exact. Internal NaN payloads may be quieted by the original native operation;
they are never published as model values: the original all-eight finite gate
zeros the group and records its fault. The new edge proof fixes this contract
up front, compares native and retained software helpers with independent host
half conversion, initializes output sentinels, and checks all original owner
words/padding and physical zero erasure. Its genuine cooperative matrix result
is independently observed. This is one additive GPU case, not a replacement for
any of the12 unchanged original-format matrix controls or original719 total.

The complete source passes350 AppCPU,18/47 runtime,8+9 metadata,144 native half
matrix layouts,8 edge layouts,48 IQ4 selector layouts and every old compiler
population. The protected720-control/both-profile run is pending. No performance,
GPU acceptance or installation claim yet. Timing harness changes retain existing
public chunk events and fixed-schema pipeline clocks, with no new GPU fence.
