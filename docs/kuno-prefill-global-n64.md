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

The first CPU build rejected nested-module visibility/imports and a missing GGUF
argument to the new numerical harness (14 diagnostics); no GPU use. Preserve
that build. Helpers now expose only the existing strata_matrix internal scope,
the suffix uses its actual original native module, and the probe receives the
same original GGUF reference. No arithmetic or resource-policy change.

GR320 has no padded N64 columns, so its genuine BF16 host case can legitimately
emit/share the same device program as ordinary BF16.24 layout-case emissions do
not imply24 distinct modules; the native ISA inventory may contain two or three
dense programs. Real21-case GPU coverage must prove GR320's actual rows/stride,
not a distinct shader hash or fabricated branch.

## Complete acceptance and result (2026-10-24 UTC)

Frozen7a54656c620c89316dcb6835a70c4aec504b4cc5 / worker
 a1534e15e60daed264f0132386d2d21c119ce2b810ebbad4fb676c7ec0e2ab6c.
372 AppCPU,18/53 runtime,8+9 metadata and10 original native controls pass;
original compiler populations plus24 dense/16 expert real layout/CFG emissions.
All837 GPU controls pass, retaining792; both complete original18-script model
chains pass, including actual new dense/expert selections, genuine unchanged
image cohorts, original finite/ordered closure, exact physical N64 workgroup
counter, original capacity/reuse/rolling/erasure and physical recovery. Original
8 wide programs remain zero-spill, both QSA hashes unchanged;180 total programs.

Live unitbfa891656bef4aedaf6df703a764e77a exits0 in13m47.677s, zero swap,
29.2GiB peak. No post-hoc audit correction/replay was needed for this run.
Root compile/source freeze and all input artifacts remain under global-n64-*.

Actual all5 new N64 native programs are zero-spill: three dense emissions
(two distinct byte hashes, legitimate BF16/GR reuse)238/241 instructions,
54 sends,128GRF/non-SSA40; GU1285 instructions/245 sends/128GRF/non-SSA78;
DOWN1482/281/128GRF/non-SSA78. Thus reduced register footprint did work, but
is not sufficient for speed. Diagnostic warmup/first/repeated3.756629191/
11.753655147/10.029326771s; normal3.727016157/11.743674263/9.906160308s.
Repeated normal is slower than preceding9.665937322s. No CLEAN was warranted;
target5.3515s and best previous CLEAN8.394039699s remain unchanged. Goal active,
unmet. Retain this entire variant and preceding variants, not isolated rollback.

Sampled4096-token layer: cohort13.358317ms, wideGU10.023800ms, wideDOWN4.277602ms,
macro12.888372ms, dequant19.221638ms, copies29.167116ms. Complete repeated
expert diagnostic5.934264413s/attention2.781501146s; normal5.832274618/
2.771020269s. Prefix sample improves modestly, but dense attention regresses.
Zero spills and fewer static sends alone do not establish efficient memory use.

Independent00:37:52UTC recovery: Intel50,688,000B,NVIDIA compute0/display596MiB,
all5 classified faults0, HTTP200/admission normal/completed163. No service,
installation/release/API/driver changes. Canonical/inherited work preserved.

Next concrete source difference: Strata's pinned active BF16/f16 wrapper calls
column-major GEMM with W first/transposed and X second/nontransposed, producing
Y^T[N,T]. Its BF16 comment and calls are in gemm.dp.cpp:760–769; f16:799–809.
Kuno's current global kernels put X in matrixA, W in matrixB, then store row-major.
Both compute the same algebra, but assign memory access to different hardware
operands. The opaque library can further transform internally, so API orientation
is verified, not a claim about oneMKL's exact machine-code tile.
Compose both operand role changes AND column-major result publication, with
original8-member pending/failure semantics, before any timing. Full proposal:
[kuno-prefill-transpose-matrix.md](kuno-prefill-transpose-matrix.md).
