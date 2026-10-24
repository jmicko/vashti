# Coupled M128/N64 global register pipeline

Proposed next backend-only complete variant after accepted7f7a1271. No source
implementation or acceptance exists yet. Do not call this Strata parity, a port
of opaque oneMKL machine code, or an established speedup. Keep the preceding
complete four-accumulator/global-image and all original variants/controls.

## Measured reason

Four-accumulator global GU7:96 and DOWN37:204 spills:fills at128GRFs make the
sampled large-expert prefixes roughly twice as slow as their original shared
counterparts. The coherent image producer/cohort consumers do reduce small
matrix time, but this pressure and real image dequantization offset that gain.
Dense global four-C variants also retain1:4/3:7 spills and the whole attention
phase remains above the earlier complete shared-dense/image control.

Use two accumulators per subgroup with both genuine A/B operand banks, preserving
chronological K16 sums. This is a register working-set adjustment, not rollback
of global operands, image producers, consumers or guards. Actual compiler spills
and complete-model time must establish whether it helps. Even eliminating this
regression alone does not prove the remaining5.3515s target is attainable.

## Complete producer geometry

Explicit benchmark feature strata-global-register-n64 composes with the entire
global-image-experts bundle; do not change a default serving configuration.
WG512/SG16, tile M128/N64. Each subgroup owns M16/N16: member offset(sg/4)*16,
column offset(sg%4)*16. Two A8/K16 matrices, one B16/K16 and two F32 C matrices.
Two real banks each contain those two A and one B. Initialize bank0 atK0;
load bank1 atK+16, consume bank0's two products, guarded preload bank0 atK+32,
then consume bank1's two products. Preserve original per-output chronological
K16 order and never load a missing final bank. No operand SLM/copies/new owner.

Dense aligned M128 and original-view prefix/suffix remain; grid X=rows/64,
Y=count/128. Genuine BF16 GR320 now has five complete N64 tiles, without padded
rows or reads. Preserve its authentic-shape special admission and full/prefix
credit laws; do not accidentally admit arbitrary320-row shapes. Original dense
F32 owner/holes/strides/weight offsets and validating finite consumer remain.
Additive partition counter dense_global_n64_products joins original slm40/deep/
asymmetric/global_register products; exact finite-consumer count still matches.

Expert selection keeps the original min/count/128-member prefix and tail views,
source/vector/CMMA/hardware/full-call bounds and preflight before any enqueue.
Change only genuine physical X grid to rows/64 (GU20/DOWN40), Y=prefix/128.
Use the original small_direct::publish primitive with new explicit member/row
coordinates and16KiB subgroup fallback scratch. The old large-wide publication
helper assumes N32/subgroup and MUST NOT be reused with wrong coordinates.
GU uses gu_prefix0 and original validating activation; DOWN uses original
ordered pending2 and original finite/inverse/combine closure. Keep original
small/direct tails and every route/status/fault/owner failure poststate.

## Matching native/runtime components

Dense has actual subgroupID but no explicit lane builtin. Compose an exact new
native entry recognizer for dense_global_n64_store_f_: WG512/SPIRV1.6/SG16,
original three StorageBuffers+INFO,0shared/barriers/atomics,9 global loads,
4 products,2 global stores. Retain existing four-C entry's exact12/8/4 rule and
all original native controls. Do not broaden every no-lane512 shader.

Expert entries have genuine lane/SG builtins through original publication.
Introduce exact generated identities and closed prefix grammar for new X grid;
retain all original role/INFO/owner/phase/tail checks. Clearly distinguish physical
N64 tile counts from any original logical128-prefix credit; do not label a changed
dispatch as an old128-column block. Add actual source/profile entry classification.
Old entries and their exact geometry/numeric probes remain independently runnable.
No global resource/host/private retained-capacity or source-lease relaxation.

## Acceptance before speed

Keep all366 current AppCPU,18/51 runtime,8+9 metadata,10 native controls, every
current compiler layout and all792 GPU controls. Add separate real new layout/
CFG checks, not weakened four-C assertions: three dense types x8 masks, two expert
roles x8 masks, genuine9 loads(A6/B3),3 initial and6 in-loop, ordered two-product
banks,4 products. Dense2 global stores and0SLM; expert2global+2 original fallback
stores/16KiB scratch, no product-loop SLM loads/barriers/atomics. Prove dynamic
original strides, final bounds and actual subgroup selection plus negative native
mutations. Test complete output/source mapping and authentic GR320 boundaries.

Add dense21 original-GPU/F64 cases and expert24 paired full-owner/F64 cases
using the original complete8 direct-tail publication policy and original weight
prefix. Preserve all old poisoning, holes/status/fault/owner/erasure checks and
all real capability reporting. Planned837 GPU cases, subject to exact final
explicit inventory; CPU/runtime additions must be separately enumerated and
must retain exact old-name sets. Never count an unavailable shape as executed.

Then both original whole-model chains with genuine image/cohort selections,
new prefix/dense path counters, original finite consumers, original source/
capacity/reuse/rolling proofs and physical recovery. Inspect actual ISA/spills;
only then use complete normal/diagnostic timings. A promising complete candidate
gets the original matched CLEAN45/4245/4245 client; no changed reference target,
seed/context/chunk policy or instrumented-versus-clean comparison. Keep old
source/evidence regardless of a speed report and diagnose remaining differences.

## Complete source composition (2026-10-24 UTC)

New source now contains both dense and expert M128/N64 producers and both real
operand banks, authentic full GR320 N64 rows, original dense finite consumer,
original small/direct expert publication and tails, exact physical X launch and
closed role/INFO/owner grammar, anonymous path/workgroup counters, exact new
native no-lane dense recognition,24+16 actual layout/CFG proofs and additive
21+24 full-owner GPU/F64 cases. Original four-C kernels/proofs remain unchanged.

Planned exact populations:372 AppCPU (original366 plus3 dense,2 expert and1 real
profile-identity tests),18/53 runtime (original51 plus2 new exact-grid grammar
controls),8+9 metadata,10 original native controls and837 GPU cases. Native
negative controls additionally mutate actual new dense modules in every layout.
Original logical M128 wide_blocks credit remains unchanged; the new anonymous
expert_global_n64_workgroups separately counts real(rows/64)*(prefix/128).
This is source composition only: no build, GPU or performance acceptance yet.
