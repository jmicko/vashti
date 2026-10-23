# Quantized subgroup producer/consumer pipeline

This internal candidate preserves the complete compact-source/activation/ordered
consumer path. It adds no inference API, configuration field or Vashti change.
The existing scalar/frame contract and original model/owner/fault guards remain.
Source b70ccfa2 is GPU accepted, as recorded below. The loop-retention follow-up
is composed but not yet built or measured.

## Complete execution

Use a distinct genuine `expert_quant_pipeline` entry, WG64/SG16. Subgroups0–1
decode two disjoint N16 weight panels; subgroups2–3 consume the corresponding
panels with logical subgroup index `PLANE_POS-2`. Each workgroup covers N32;
the native grid is `rows/32,1,cohort_count`. Original `expert_quant_cohort`
remains N64 and its original geometry/controls remain independently accepted.

Producers fill bank0 before a common whole-workgroup barrier. In each K64
iteration, producers stage the next bank while consumers execute the current
bank's four chronological K16 products. A single barrier outside both role
branches joins every reader and writer before banks switch. All lanes execute
the same dynamic barrier instances, including the final drain. Separate role
loops with independent barriers are forbidden.

Producer lanes reset their private C fragments after staging, using genuine
matrix fills. Their unused previous accumulators need not remain live through
integer decoding. Consumer lanes preserve their original C accumulation.
M8/M16 retain four static chronological K16 steps. M32 uses a genuine four-trip
inner product loop, retained by the existing authenticated native DontUnroll pass.
This bounds fragment liveness while preserving both subgroup roles and banks.
The initial all-static pipeline and original N64 rolled kernel remain controls.

Keep the existing pitch24, trimmed376-half K16 tiles. Two logical subgroups and
two banks require12032B. Original publication indexes actual SG2–3, so allocate
four128-F32 scratch slots,2048B. Total14080B is a separate-allocation envelope;
do not assume lifetime aliasing. Only consumers publish after the final common
barrier. No new global/private owner or model copy is needed.

## Closed integration

The HAL accepts only the exact new entry, original WG64/SG16 and seven-data-plus-
INFO descriptor inventory. Both native paths use the same exact half packing.
The original entry's wrong-workgroup negative cases must stay rejected. Native
cohort admission derives N32 only from the exact new generated identity; old
identities keep N64. Scalar equations, immutable role/source handles, bounds,
status/fault owners, RAW release and ordered-output rules stay unchanged.

Keep original18 matrix definitions and144 layouts. Add18 pipeline definitions
and144 real compiler layout controls. The original12 fused quant comparisons
must still execute the original kernel; add12 pipeline comparisons against the
same independent original decoded-image products, including all member limits
and poison routes. Report the new comparison population separately. Preserve
all original720 GPU controls; planned total becomes732. Counts/profile classes
must recognize actual new matrix launches, without changing logical phase laws.

Acceptance needs independent role/bank/coverage/chronology CPU oracles, exact
generated identity and grid negatives, genuine shader WG/SG/CMMA/barrier/type/
stride/footprint proofs, full owner/status/fault/source/padding comparison,
whole-model routing/counter/reuse checks, and physical recovery. Speed testing
follows the entire producer/runtime/consumer composition. Original clean
45/4245/4245 fixture remains the only matched Strata acceptance measurement.

## Accepted pipeline and loop follow-up (2026-10-23)

Complete source `b70ccfa22983af053566d07443093f4d8b6a2737`, worker
`b86707e6a1e225543a5306003c8603590a0dd925bf28a25b0ce31c355789166c`,
passes357 AppCPU,18/49 runtime,8+9 separate metadata controls, all original
compiler populations and144 new real pipeline layouts. All732 GPU controls,
both original forward chains, routing/reuse/capacity and physical recovery pass.
The12 new real-format pairs all have zero bit mismatches. Protected unit exited0
in12m39.476s, swap0, peak33.4GiB. Diagnostic first/repeat11.585634904/9.563811616s;
host mode11.694759/9.514985s. No CLEAN run. Latest CLEAN remains the older
12.471177197/8.394039699s against Strata5.5006/5.3515s; parity remains unmet.

Same request3/layer24/count4096 cohort matrix41.076030→34.883995ms; ordered
copy29.110608ms, dequant7.955141ms, macro12.868374ms are essentially unchanged.
Six new M32 programs spill34–35 writes/87–120 reads, using155–173 preallocation
registers; original18 rolled control programs and original wide/QSA checks remain.
Repeated4096 host chunk7.184836538s; tail148+teacher/cleanup2.320561123s.
Repeated native pipeline creation0; first long request23 new pipelines costs
~0.80s preparation+1.18s driver creation, separate from warm execution.

Next source retains M32's real inner loop inside the complete new pipeline,
through the same native CFG-aware hint pass. Only loop spelling and its retained
control change; roles, geometry, K16 order, SLM14080, owners, source guards,
publication and original controls stay. Actual compiler/product/hint checks
must distinguish M32 one static product step from M8/M16 four. This follow-up
has no execution acceptance yet. Do not infer speed from removed spills alone.

Artifacts use `small-quant-pipeline-*` under the development directory, including
root build/freeze receipts, public ISA export and both safe phase receipts.
Independent20:13UTC state: Intel50,688,000B, NVIDIA compute0, all5 faults0,
HTTP200/admission normal, completed139. No installation/API/Vashti changes.
