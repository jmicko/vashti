# Coupled residual register reuse and subgroup pair packing

Backend-only optional source plan2026-10-23,complete prior bundle retained.
PinnedMIT Stratafb58e0db sycl/src/prefill/kernels.dp.cpp gr_write_norm_rs_kernel
keeps v[10] for the original d=tid+256*k order; normalization uses those same
registers. Kuno's GR_NORM writes pending R,then rereads R in a different pair
map and needs a real storageBarrier. Do not merely delete that necessary barrier.

Port TOGETHER: ten explicit F32 scalar residual registers (no dynamically
indexed local array),same FMA/write and chronological sum r*r,unchanged two-level
SG32 xor reduction and local workgroup barrier. Normalize each scalar in its
original lane; shuffle xor1 supplies odd neighbor to even lane for identical
BF16 pair packing. Explicit logical tid=sg*32+lane,full256x1 workgroup/fixed32
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
