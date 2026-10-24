# Two-accumulator subgroup cells with authenticated padded inputs

Coherent internal candidate after full1305 padded-global acceptance. Keep the
whole preceding variants and original401 AppCPU/1305 GPU controls. No API,
Vashti, installation, model/context/precision, driver or reboot changes.

`strata-balanced-global` / `--balanced-global` requires padded-global and
propagates `kuno-balanced-global`. Separate genuine matrix identities use
ONE chronological K16 loop, two8x16 FP32 accumulator fragments per SG16,
two independent global FP16 weight-A loads and one shared global input-B
matrix. Each subgroup covers16 real/padded input members x16 output rows.
The existing input producers initialize all necessary padding. Private native
macro admission requires their exact Pad32 identities and complete input
extent, including after checked SwiGLU. Packed cohorts retain their original
proven16-row padding/header/slot authority. No unused-capacity assumption.

Macro32/64 retain original M32/M64,N64 dispatch grids and scalar/owner ABI,
with WG128/256 (8/16 independent subgroups) instead of WG32/64. Cohort16 uses
WG64 (four subgroups) instead of WG32. Subgroup token=row-block*tile+(sg/4)*16,
output row=feature-block*64+(sg%4)*16. Complete real cells are covered once;
fully inactive subgroups terminate without workgroup barriers. All original
independent8-half finite/invalid/sentinel/sticky-fault publication, checked GU
and pending ordered DOWN consumers remain. Tile8 and wide tails stay intact.

New subgroup-local publication scratch is4096/8192/2048B for macro32/64/cohort16.
These are explicit NEW shader resources with actual CPU/compiled proofs and
device-limit checks; preceding kernels keep their exact old resources. No
original private GPU owner, input/output/routing/status allocation, command
tape, retention or public-oracle budget grows. This is an occupancy hypothesis
suggested by the remaining small-GEMM profile and earlier higher-parallelism
cohort path, not a claim about oneMKL's opaque internal tile geometry.

Add actual shader CFG/type/SSA/negative proof, exact subgroup-cell coverage
and actual profile-name control =>404 AppCPU, retaining all401. Actual emitted
FP16 A8x16/B16x16, FP32 C8x16 and subgroup scope are authenticated. Genuine
negative mutations include wrong owner/WG/dtype, aliased second A and missing
second product. Distinct A loads must feed their corresponding products;
both products reuse the one actual input-B matrix in the same K iteration.
Require original AtomicOr-only fault descriptor and zero arithmetic-loop atomics
or input barriers. All original staged/independent-B negatives remain in the
retained complete modules. New48 layouts are32 macros plus16 packed cohorts.

Add four exact native full/cohort role and forged-name/owner/grid/phase tests
=>73 full runtime, preserving69 and original18 feature-off,8+9 metadata and10
native controls. Add116 actual previous/new independent F64 and whole-owner
regular/cohort cases, plus four real eight-slot previous/new cohort pairs
with original tile8 retained, one original GPU output owner, complete erasure
between programs and existing bounded zeroizing host reference. Total1425 GPU
controls. All data-dependent publication rules and tolerances remain unchanged.

Compose this entire producer/padding/geometry/math/publication/native/profiling
bundle before speed comparison. Do not roll back individual parts from isolated
timing. No compilation, GPU correctness or performance claim at composition.
