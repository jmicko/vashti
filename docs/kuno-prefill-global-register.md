# Direct global register pipeline contract

Backend-only next complete dense producer/consumer, motivated by the actual
2026-10-23 Strata kernel-properties capture. No inference API, configuration,
Vashti, installation or global driver change. Do not call this an exact port of
opaque oneMKL machine code. Do not claim performance before full acceptance.

## Confirmed difference

Development `strata-kernel-properties-public-20261023.json` contains53 real GEMM
geometry rows for22 entry names: all256 registers/thread, zero private/spill
memory;51 rows use0 shared bytes and2 use24576B. Busy FP16/BF16 32x64/64x40
variants use0 shared bytes. Original fixed45/4245/4245 requests complete with
correct answer, no prompt/cache reuse,32768 context/int8 KV and original binary.
Instrumented first/repeated8.5263/8.2870s are not clean timing references.
The clean target remains5.5006/5.3515s.

Intel documents the register column as register count/thread:
https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md
These are actual queried properties, not tile/register inferences from names.
The trace is2,710,535bytes, SHA256
60c80b3e0c486b0edd5323d7678845783a7d400a33b5148b15fd553a5f4f4fca.

The profiler2.4.0 verified package is a read-only local artifact with just its
matching XPTI/intlc dependencies mounted read-only. Original reference binary,
model, public fixture and exit shim remain unchanged. No system installation.
A CPU-only container-version attempt timed out and its exact no-GPU launcher
was terminated; preserved under development, not called compatibility proof.
Host CPU version check passes2.4.0, embedded commit a56d1d768c1ccf18e70db692914370debc11c503.
That embedded commit could not be read from public upstream (404); do not claim
package/source equivalence. Existing2.3 capture remains independent evidence.
The initial CSV export rejected an empty parsed population because leading
spaces preceded quoted fields; retained rejection and fixed skipinitialspace
for the same immutable trace, without rerunning reference or allowing empties.

Current Vulkan asymmetric M256/N128/K32 WG512 kernels still have128 GRFs and
109:105 spills:fills (FP16/BF16),122:132 (GR320), after source-level A lifetime
reordering. Both complete variants remain numerically accepted. The reordering
did not relieve the real register pressure. Do not keep treating source order
as a proven register schedule or the bigger shared tile as oneMKL's design.

## Next complete implementation

Prefer a no-SLM direct-global A/B register pipeline before more shared-memory
or larger-workgroup tuning. Use WG512/SG16, tile M128/N128, per subgroup M16/N32,
four F32 accumulators. This adapts register blocking to the measured128-GRF
Vulkan constraint; no dependency on hypothetical256-GRF Vulkan support.

Two real matrix-operand banks: each has two M8/K16 A fragments and two K16/N16 B
fragments, loading directly from the original packed FP16/BF16 StorageBuffer
views with their authentic column strides. No source copy/repack/global owner,
no SharedMemory or workgroup publication scratch, no workgroup barriers.
Initialize bank0 at K0. Each K32 loop loads bank1 at K+16 before bank0's four
products, then (only if K+32<columns) preloads bank0 at K+32 before bank1's four
products. Each output retains original chronological K16 accumulation. Matrix
banks are reused only after their genuine prior products; no speculative final
input/weight read or descriptor-copy substitute for actual matrix values.

Keep GR320's missing64 rows unread: padded B fragments remain initialized zero,
with subgroup-uniform row guards, and nonlogical output fragments never store
or set faults. Direct global cooperative stores target the same F32 owner;
append the original finite/fault consumer exactly as before. Same source/dtype,
vector/alignment, bounds/nonalias, output holes/strides and native grid guards.
Retain actual SG16 and WG512 capability gates. Preserve all original resource,
retained-capacity, private erasure and original numerical tolerances.

Compose aligned M128 selection and original-view prefix/suffix handling together:
nonaligned counts use the original input/output subviews, direct-global complete
prefix and unchanged original suffix. No private copy or new payload allocation.
Retain original dense prefix/finite/GR-wide credit laws. New direct-global count
is a separate additive partition alongside slm40/deep/asymmetric, not a relabel
of an unselected kernel. Keep original asymmetric and all old numerical controls.

## Acceptance before speed

Three additive AppCPU controls planned: exact subgroup output/source geometry,
two-bank K16 chronology/final bounds plus capability negatives, actual compiler
layouts. Preserve original360 selected cases; planned363. Runtime18/49 and
metadata8+9 stay unchanged unless actual implementation requires a documented
additive change. Preserve all old compiler populations plus24 real layouts:
FP16/BF16/genuine GR320 across8 layout masks. Require actual0 shared bytes,
WG512/SG16,12 StorageBuffer cooperative loads (initial4 + two loop phases4 each),
eight products per K32, four original global stores, no matrix-phase atomics or
barriers and genuine checked current/next dataflow. New proof is distinct from
old shared/register CFG checks; never weaken their transfer/barrier requirements.

Plan21 paired original-GPU/F64 cases (original15 asymmetric geometry/control
catalog retained as additional new-path cases, plus BF16/F16 M128 finite,
GR320 M148 finite and poisoned final input, GR320 M513 finite, and FP16 M257
nonzero weight/output offsets/stride). Include short K64/K128/K320, genuine
K10240 weights, NaN/Inf sources, full source/output holes/owner/fault checks,
actual prefix+suffix and original final erasure. Original747 controls retained;
planned768 GPU cases. Actual completed evidence must establish populations.

Then run the two original whole-model chains and both diagnostic/normal modes,
check actual path counts, routing/reuse/capacity proofs, original wide/QSA ISA
and physical recovery. Inspect real spills. Retain the complete source regardless
of one measurement; diagnose confirmed remaining differences rather than rolling
back coupled pieces. Run original CLEAN comparison only for a promising complete
candidate. No API/UI/cancellation work is part of this optimization.

## Checkpoint

No direct-global source implementation/build/GPU acceptance exists yet. It is
the next coherent implementation, not a completed change. B70 also reports
max1024 workgroup invocations,64 max compute subgroups, SG16..32; a WG1024
asymmetric adaptation remains a later alternative, not the next first choice
now that direct-global/register differences are measured. Any WG1024 selection
must query/gate real per-device subgroup capacity rather than relying on a
machine-specific capture or tightening the entire backend's existing32 gate.
