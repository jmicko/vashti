# Matched Strata default attention schedule

Optional complete attention-mode comparison; not installed or speed-accepted.
Retain the complete expert/copy candidate and previous independent attention
implementations. No API/model/config/frontend/private persistence change.

## Actual reference versus selected schedule

Pinned MIT Stratafb58e0 SYCL `qsa_decode_attn.dp.cpp:110,553-578,580` selects
`attn_chunk_kernel<KV_MODE>` by default. Its NC=2 kernel is inside
STRATA_EXPERIMENTAL_SM60; SYCL CMake does not enable it. The exact reference
build script targets SYCL/bmg-g31 without that CUDA experiment. Default uses one
cell per step, original constant XOR reductions and ordinary shared probability
reads. Our selected complete paired/uniform/gather path uses the experimental
NC=2 strategy with indexed shuffles. Packed reads were correct but neutral.

Compare the COMPLETE selected reference schedule together: original load8-style
K dequantization and query vec4 specialization, explicit chronological QK FMA,
constant XOR tree, one-cell loop and ordinary shared chronological PV FMAs.
Use existing `prompt_specialize` + `prompt_fma`, followed by the original local
fault closing for the fault-only entry. Keep the original scalar reference
unchanged and all old generator/inverse/compiler controls. The existing paired
and packed implementations remain available; their source-equality assertions
apply to their selected mode, while new exact composition tests cover this mode.
Decode/MTP/merge, causal selection, full owners/strides, status/fault/output,
barriers, integer geometry, bounds and all original IEEE/F64 references stay.

Source feature `strata-qsa-reference-schedule`, explicit builder
`--qsa-reference-schedule` with existing complete bundle. No new shader enum,
GPU/host allocation, smaller advertised reservation or erased data exemption.
Original shared scores have stride64 (768 F32), matching reference; all complete
owning resources and original synchronization remain. Three new actual CPU
controls prove exact composition, genuine compiled96 FMAs/70 XORs, unchanged
bindings/geometry/barriers and exact cell ownership.96 static FMAs per one-cell
body versus180 per paired body still execute672 QK FMAs per subgroup chunk;
chronological PV work stays identical. Actual native compiler must emit96 MADs
with subgroup32/no spills; original paired mode still MUST emit180. This is an
explicit instruction inventory for the different schedule, not a weaker bound.

Retain all172 CPU controls +three new, runtime13,421 WGSL/216 layouts and all233
GPU controls, original F64/status/poison checks and both whole-model profile/
count/source/owner/erasure/recovery chains. No speed conclusion from source fact.
Original terminal/clean5%-at-EACH/staged quality/privacy gates precede install.


Initial175 suite has174 passes; the new source-inventory control compared local
fault closing against the old shared-atomic closing. Compare the SAME already-
accepted local closing on both sides. All original172 controls pass, and the
new cell inventory/composition controls pass. This corrects only a CPU reference
scope; no engine/shader/number/bound is changed. Preserve initial failure log.
Repeat complete175 exact-source CPU/offline/GPU gates on a new artifact.
