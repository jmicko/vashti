# Coupled residual register reuse and subgroup pair packing

Backend-only optional source plan2026-10-23,complete prior bundle retained.
PinnedMIT Stratafb58e0db sycl/src/prefill/kernels.dp.cpp gr_write_norm_rs_kernel
keeps v[10] for the original d=tid+256*k order; normalization uses those same
registers. Kuno's GR_NORM writes pending R,then rereads R in a different pair
map and needs a real storageBarrier. Do not merely delete that necessary barrier.

Port TOGETHER: ten explicit F32 scalar residual registers (no dynamically
indexed local array),same FMA/write and chronological native fused square sum,unchanged two-level
SG32 xor reduction and local workgroup barrier. Normalize each scalar in its
original lane; shuffle xor1 supplies odd neighbor to even lane for identical
BF16 pair packing. Original LocalInvocationIndex,verified neighbor adjacency,full256x1 workgroup/fixed32
contract,bijective original256-stride scalar map. Both in-place(one RW R
descriptor) and disjoint previous/R variants. Original kernels retained as
independent GPU comparators/fallback. New path tested on actual B70 only; no
unsupported cross-device bitwise claim. No R read AFTER reduction,thus no
inter-lane global R producer/consumer and no within-kernel storage barrier.
Keep the real local sums barrier and normal inter-kernel visibility.

All original429WGSL remain. Two new source modules validated separately,with
actual shader ABI/workgroup/storage-memory barrier/whole-owner checks. Add paired
original/new GPU full R/normalized/scales/fault/source/padding comparisons for
apply/no-apply,in-place/disjoint,short/long/count148,offset and nonfinite cases.
Preserve273CPU/15runtime/all old shader/GPU433/BOTH original18-script chains;
add new actual-compiler and mapping controls. No precision,capacity,ownership,
erasure,private persistence,API/model/context/config/frontend/install/release
change. Vashti:none. No compile while timing. No speed claim from source.

## Complete source implementation

Optional strata-gr-register-norm / --gr-register-norm requires full prior
wide-register-stage bundle. New separate RegisterNorm task preserves the exact
original globals/Params ABI and production original live-scope requirement.
A dedicated fixed-public raw-operator comparator uses authentic PrivateArena
owners and claims no model/prompt scope. Original GrNorm/InPlace kernels and
429 original shaders unchanged; two new WGSL modules validate separately.
Two new CPU controls retain273 original controls (275 total): full scalar/pair
map and actual Naga/SPIRV ABI,ten scalar F32 locals,no globalR load after rs,
one genuine local barrier and no storage barrier. 20 paired GPU controls added
to433 (453 total): counts1/8/148/512,apply/no-apply,in-place/disjoint,nonzero
first10240,NaN previous/residual and infinite block. Every R/image/scales/fault
word,readonly source and padded owner is checked against the original GPU.
No new production payload allocation. All native zero-spill and BOTH unchanged
original18-script chains,actual recovery before claims; no compile during timing.
No speed/build/GPU/install claim for this new source.

Initial ee76a55f CPU build reports only internal probe visibility errors E0364/
E0603; change probe::cases to pub(crate),matching its internal worker re-export.
No shader/math,API or GPU change. Failed build logs retained.

abe19c73/worker83708a1c passes275CPU/15runtime/all429originalWGSL/320matrix+
16raw+32compact+24padded/48realCFG checks plus2new norm modules. No GPU run.
Next source gives each new kernel an exact static entrypoint name for actual
compiler ISA/spill identification; Params/bindings and mathematical body unchanged.
No inference endpoint or runtime content enters this identity.

## Actual full-model guard failure and stronger fixtures

fd975336/workerbfa336a4 passes275CPU/all shaders/ALL453 GPU and public profile
compute/recovery, but immutable original QSA-uniform chain rejects changed
expert-group counts. Therefore NOT full acceptance; no normal or clean/install.
Profile9.772395357/8.501690357s is not an accepted speed result. Both actual
new native kernels SIMD32/128GRF/0spills,832/843 instructions. Preserve failure.

The dyadic raw fixtures missed real-activation precision/topology behavior.
Retain the ORIGINAL LocalInvocationIndex for every scalar sum and write,not
sg*32+lane. Shuffle pack uses physical-tid parity and explicitly checks actual
neighbor tid==(tid xor1); unsupported adjacency sets the original sticky fault
without divergent early exit/barrier. Optional B70 laboratory feature only;
not advertised portable fallback. Original uninstalled path stays available.
Add four full-mantissa F32 fixture cases (old20 retained,total24,new457GPU)
covering apply/no-apply and in-place/disjoint. Failure diagnostics contain only
public mismatch counts per R/image/scale/fault field. Do NOT relax routing or
bitwise guards. No shader precision policy changes. Source-only correction.

## Physical mapping fixture failure and chronological FMA correction

Source4fa1d067/workere04fa7d2 passes275CPU/15runtime/all original shader
layouts plus the two new modules. Protected GPU operators STOP before model
execution on the stronger full-f32 apply=true/disjoint case: mismatches
[R,image,scale,fault]=[0,0,1,0]. No accepted throughput result. Recovery
2026-10-23T04:39:48UTC: Intel50,688,000B,NVIDIA compute0,faults0,HTTP200,
admission normal. Both failed benchmark logs and receipts are retained.

Actual Mesa NIR from that same public run establishes the arithmetic difference:
original program35 loops FFMA(r,r,sum) chronologically. New program36 computes
r1*r1 first and FFMA(r0,r0,that),then r2 onward. Contracting the unrolled
sum+=r*r permits that exchange and changes rounding. Do not disable contraction
on the new path alone,or relax the original bitwise/routing checks. New source
expresses sum=fma(rk,rk,sum) for all ten terms,matching the actual original
native sequence. Existing actual-module CPU validator now checks all ten real
Naga FMA-to-sum stores and their chronological residual inputs before reduction.
The original kernels remain untouched as independent GPU comparators. Full
original GPU and model acceptance remains required; no speed/install claim.

Failure evidence lives under /media/jmicko/faststore/kuno-development:
gr-register-norm-physical-exact-error-20261023.txt,
gr-register-physical-failed-programs-20261023.json,
gr-register-physical-failed-program-{35,36}-20261023.txt,
gr-register-norm-physical-failure-state-20261023.json.

## Complete 5ae19f5e validation

Frozen source5ae19f5edf2d7b364e8f7c9abf9fd0c5a1f97849,
workerdc00d189266ebf9cd9e4fd799ab545b070782e93e44a480f410d4273e113cb68:
275CPU/15runtime/all429WGSL/320matrix+16raw+32compact+24padded/48actual
pipeline-CFG proofs+2new norm modules/ALL457GPU/BOTH unchanged original
18-script chains and physical recovery PASS. All24 norm cases are bit exact.
Both modes preserve routing[7663,26673,26673],actual fullGRdown counts[0,97,97],
new norm counts[97,194,194]. Both actual new SIMD32 kernels have zero spills.

Public instrumented first/repeated long9.854916837/8.557609991s; normal
9.872980496/8.468479392s. Normal HC attention+router815.766879ms versus prior
891.447830ms; overall repeat gain is small. No clean/staged/install/release.
Do NOT count the initial permission-failed launch as a model test: the root
build's0500 binary initially needed its intended jmicko ownership. Retain that
failed receipt; the owner-corrected run above is the complete actual test.

Evidence under /media/jmicko/faststore/kuno-development:
strata-parity-gr-register-norm-fma-owner-20261023-expert-cohorts-public-summary.json,
gr-register-norm-fma-complete-isa-export-20261023.json,
gr-register-norm-fma-owner-connected-20261023.log.
