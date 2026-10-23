# Quantized subgroup producer/consumer pipeline

This internal candidate preserves the complete compact-source/activation/ordered
consumer path. It adds no inference API, configuration field or Vashti change.
The existing scalar/frame contract and original model/owner/fault guards remain.
No implementation, GPU acceptance or speed improvement is claimed by this design.

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
The consumer's four K16 steps are static in this complete role-separated path;
the original M32 rolled path remains an independent control.

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
