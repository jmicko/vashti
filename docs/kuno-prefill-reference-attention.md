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


## Frozen reference-schedule checkpoint (2026-10-22 00:20UTC)

Source180b91bdd101395a83489e9e4ac954320d33e671; worker SHA256
`e3455e4c95963b7484f957a1674560ac88f71eb86784925551a1ec0c643e8da1`,
build`strata-parity-reference-fixed-20261022`. Final175 App controls PASS;
runtime13 unchanged-source valid fromffa; frozen421 WGSL/216 matrix layouts
PASS with original pinned validator. No source/math/owner/reference bound relaxed.
Earlier release322 was correctly refused after HEAD changed during its build;
no artifact from that incomplete build is accepted. Keep its diagnostic.

Root unit`kuno-reference-attn-20261022.service` now runs original233 GPU controls
and BOTH full profiles/compiler/source/count/owner/erasure/recovery chains.
Exact legacy compact compiler guard was extended ONLY for explicit reference
schedule: native96 MAD instead of180; subgroup32 and spills/fills0 remain exact.
All previous modes still require180. Original helper snapshot preserved at
`check-native-qsa-compact-before-reference-schedule-20261022.py`; new helper SHA
31d3cf54ea49921a18c18d5dc979d66a99e9b445a83a99e505246e4dd0dfb42e
is pinned by the new connected wrapper. No numerical gate/reference changed.

Evidence under `/media/jmicko/faststore/kuno-development`:
`native-reference-fixed-{cpu-selection,cpu-features,cpu-final,offline,release-build,connected}-20261022*`,
`check-native-reference-fixed-{connected,acceptance}-20261022.py`.
GPU/performance completed below. Original terminal/clean/staged/install acceptance
was not started for this neutral/slower candidate.


## Complete outcome and user-requested close

Protected `kuno-reference-attn-20261022.service` completed all233 GPU controls
and BOTH original73/4245/4245 modes, including full compiler/count/source/owner/
erasure/recovery chains. Exact artifact/source/hash above. Aggregate receipt:
`strata-parity-reference-fixed-20261022-reference-attn-public-summary.json`.
No numerical bounds, status/poison controls or owning guards relaxed.

Normal first/repeat11.404259100/9.965325679s. Sampled-profile mode
11.321599225/10.028465246s. Normal warm attention total3.055612906s, QSA
attention/merge1.607492536s. Compared with packed11.3106/9.7327s, this complete
default schedule yields no additional speed. Load times varied117.05s sampled
and81.94s normal; do not mistake load variance for prompt gains. All correct
components retained as optional; installed worker/gateway unchanged. No new
terminal/clean/staged/install selection or decoder speed claim.

User requested close. Fresh00:31:05UTC physical recovery receipt
`native-prefill-final-state-20261022.json` confirms Intel50,688,000B, NVIDIA
compute0/655MiB desktop, reported driver faults0, all services HTTP200 and normal
admission. All owned test/build/worker jobs ended; bounded sudo refresher expired.
Goal remains paused/unmet. See coordination guide for next investigation and
full checkpoint. Required Vashti changes:none.
