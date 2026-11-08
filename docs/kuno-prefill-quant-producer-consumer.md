# Quantized subgroup producer/consumer pipeline

This internal candidate preserves the complete compact-source/activation/ordered
consumer path. It adds no inference API, configuration field or Vashti change.
The existing scalar/frame contract and original model/owner/fault guards remain.
Current K32 source6edb769c is fully GPU accepted below. The original K64 pipeline
and its retained-loop revision remain committed controls. Parity remains unmet.

## Complete execution

Use a distinct genuine `expert_quant_pipeline` entry, WG64/SG16. Subgroups0–1
decode two disjoint N16 weight panels; subgroups2–3 consume the corresponding
panels with logical subgroup index `PLANE_POS-2`. Each workgroup covers N32;
the native grid is `rows/32,1,cohort_count`. Original `expert_quant_cohort`
remains N64 and its original geometry/controls remain independently accepted.

Producers fill bank0 before a common whole-workgroup barrier. In each K32
iteration, producers stage the next bank while consumers execute the current
bank's two chronological K16 products. A single barrier outside both role
branches joins every reader and writer before banks switch. All lanes execute
the same dynamic barrier instances, including the final drain. Separate role
loops with independent barriers are forbidden.

Producer lanes reset their private C fragments after staging, using genuine
matrix fills. Their unused previous accumulators need not remain live through
integer decoding. Consumer lanes preserve their original C accumulation.
M8/M16 retain two static chronological K16 steps. M32 uses a genuine two-trip
inner product loop, retained by the existing authenticated native DontUnroll pass.
This bounds fragment liveness while preserving both subgroup roles and banks.
The initial all-static pipeline and original N64 rolled kernel remain controls.

Keep pitch24, trimmed376-half K16 tiles. Each K32 bank contains two such tiles,
188vec4. Two logical subgroups and two banks require6016B. Publication indexes
actual SG2–3, retaining four128-F32 scratch slots,2048B. Total8064B is separate;
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

## Retained pipeline accepted; coherent K32 follow-up

Source `3f7c48d48a045904b3d40f88c8c7a6acd368da60`, worker
`a83297ae5843a7757c1c37003976f1ece754c7d6f7f5bc3f73833147605dc1f7`,
passes the same357 AppCPU,18/49 runtime,8+9 metadata, all compiler populations,
all732 GPU controls and both full-model chains/recovery. All18 pipeline programs
now have zero spills; M32 has4 native loops, versus3 before. Sampled cohort
34.883995→34.201352ms; diagnostic11.539289877/9.543312487s,
host11.677776/9.511270s, essentially unchanged. No CLEAN run. Protected run
12m50.752s, zero swap, peak32.0GiB. A receipt-only fixed-load regex rejection is
archived; actual compiler and CPU checks passed before its correction. Evidence
uses `small-quant-pipeline-roll-*`. Independent20:36UTC state: Intel50,688,000B,
NVIDIA compute0, all5 faults0, HTTP200/admission normal, completed142.

Next candidate reduces only this pipeline's K grain to32. Producer lane/round
mapping and both consumers change together. Four rounds cover four rows each:
row=round*4+lane/4,column=(lane%4)*8. Two pitch24 trimmed K16 tiles use188vec4
per bank; two SGs×two banks need6016B plus2048B publication scratch=8064B.
All data and padding have unique writers. Same N32/WG64/SG16 roles and common
barriers; M32 retains a now two-trip native inner loop. Old N64 K64 control and
original populations stay unchanged. This footprint is a target, not a measured
occupancy claim. Source is not yet built/GPU accepted. No API/Vashti changes.

## Complete K32 acceptance and next direction (2026-10-23)

Frozen source `6edb769ca9f97123970037315a254d166e348b44`, worker
`f31fa3f6a6c24daa930dbe8152a39985a188241dd087c4a619a192fbe33e4210`:
357 CPU,18/49 runtime,8+9 metadata, all original compiler controls plus144 new
pipeline layouts pass. Actual8064B shared in all layouts, exact native halves,
N32/WG64/SG16, common barriers and48 M32 retained loops. All732 GPU controls,
original18-script chains, full-model routing/counter/reuse/capacity and recovery
pass. Original12 N64 numerical pairs remain; all12 new pipeline pairs have zero
bit mismatches. All18 new pipeline ISA programs have zero spills, as do original
wide programs; both original QSA ISA hashes remain unchanged.

Protected unit `kuno-small-quant-pipeline-k32-20261023`, invocation
`cd58ad4b3ab443ae9f18e3b447d171ab`, exited0 in13m6.216s, zero swap, peak31.1GiB.
Diagnostic first/repeat11.434381486/9.406666034s; host-mode11.435514301/9.306249103s.
Same sampled cohort32.783261ms, versus34.201352ms retained K64 and41.076030ms
original N64 rolled control. Repeated4096 host chunk7.151744100s (normal7.040471967),
148 tail+teacher/cleanup2.193977196s (normal2.208794477). About6% improvement over
the starting diagnostic, without closing parity. No fresh CLEAN run or install.

The complete FP16-image control4cc3f68f remains faster: same diagnostic repeated
8.946789s (normal8.854793s), cohort12.709929ms and dequant19.224670ms. Current
fusion saves image traffic but still costs more end-to-end; do not hide this
regression or select a path from a single kernel. Its last clean12.471177197/
8.394039699s is still the newest clean comparison, against Strata5.5006/5.3515s.
Retain complete coupled candidates; future selection needs whole-model evidence.

Evidence in development uses `small-quant-pipeline-k32-*`, including root build/
freeze receipts, both safe phase exports, complete ISA export, final-state and
`strata-parity-small-quant-pipeline-k32-20261023-expert-cohorts-public-summary.json`.
Independent20:58UTC state: Intel50,688,000B,NVIDIA compute0,all5 kernel faults0,
HTTP200 and normal admission,completed145. GPUs free; installed binaries intact.
Only inherited untracked text/__pycache__ remains in this worktree.

Next review measured remaining dense/GR and expert scheduling gaps against the
existing protected Strata timing receipt, rather than repeatedly changing this
one grain. `strata-matched-timing-reference-20261020/receipt.json` already contains
fixed-public coarse GPU phases; no new profiler install is needed for that.
Its phase intervals include queue idle/event overhead, so they are leads, not
identical boundaries to Kuno's timers. Reference repeated full prompt5.3515s uses
4096+148, identical to ours. Normal Kuno already uses real rolling submission;
the instrumented operator sample deliberately disables it for timestamps.

Read-only GRF research archived in `mesa-large-grf-source-check-20261023/`:
[upstream cb671c51](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/cb671c51db1407f82c366b32642a7271224ceedd)
changes only the executor utility, not automatic ANV compiler policy. Inspected
[brw compiler04cbd47f](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/04cbd47f72d02d3f5a54f5a9dc46439c909d36c9/src/intel/compiler/brw/brw_compiler.c)
still distinguishes128 GRFs on Xe2 from256 on Xe3. Do not assume a news headline
means a driver upgrade fixes Vulkan inference. No driver was built/installed.
No API/config/Vashti change, release, reset or reboot. Goal ACTIVE/UNMET.
