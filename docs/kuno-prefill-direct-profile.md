# Direct Strata versus Kuno profiling, 2026-10-26 UTC

The latest accepted layer source8ab19c0e/workf9a33886 has CLEAN first/repeat
12.011342704/7.465637686 seconds, target5.5006/5.3515 remains unmet.
User requests direct profiling before another speculative optimization.

Pinned MIT Stratafb58e0d is C++ with Intel SYCL/DPC++ and oneMKL. The current
Rust/Vulkan implementation preserves model logic but executes different kernels.
Language alone is not a demonstrated cause. Earlier phase/kernel captures exist;
this is a new request3-only device timeline, not the first Strata profile.

External v1/v2 timeline attempts failed before model startup because the Chrome
trace defaulted to a read-only working directory. Both originals and verified
recovery remain. V3 directs only the public profiler trace into its existing
writable trace mount. Original engine source/binary, compiled-public fixture,
45/4245/4245 protocol, context32768 and no-cache checks remain unchanged.
Profiler starts paused, resumes between completed requests2/3 and pauses after3.
All3 requests pass with4outputs, zero reuse/cache/file blobs. Request3 instrumented
prefill8.3308s; this is NOT a replacement for the clean5.3515s target.

Actual trace has208548 device events,96 layer gathers/combines,26473 expert
activations and52946 expert GEMMs/dequantizations. Structural gather→combine
boundaries identify2chunks×48layers. Parent-monotonic versus profiler wall
alignment is not established; do not cut scopes by their apparent timestamps.

For fixed-public chunk4096/layer24:

| Scope | Strata device ms | Kuno serialized device ms |
| --- | ---: | ---: |
| Matrix products |9.694037|27.468950|
| Dequantization |10.314357|16.861090|
| Copies in expert interval |28.964585|28.440772|
| Gather |0.601875|2.069844|
| Activation |0.740589|1.197133|

Strata387 routed groups versus Kuno390, and different instrumentation/fences.
Copies may include concurrent staging; physical ownership is not established
from their time window. These are diagnostic leads, not exact causal savings.
Do not multiply one layer by96 or sum overlapping classes into whole-request time.
Strata entire96expert-window GEMMs total547.472ms, dequant745.717ms, copy2032.180ms;
whole trace also includes4outputs and their verification. QSA batch kernels
678.099ms including request decode, near the older corresponding Kuno QSA family
736.829ms under different timers. Strong next lead is expert matrix execution.

This isolated diagnostic adds four static matrix-tier count/ns buckets to the
EXISTING layer24 request3 timestamp collector: small, regular, large, other.
No source/math/kernel/device/query/fence/selection/capacity/ownership changes.
Original aggregate classes and matrix families remain. Tier totals must exactly
reconstruct the actual original matrix count/ns in the completed public receipt.
All original private/sampling guards remain unchanged. No API/installation claim.

Evidence under `/media/jmicko/faststore/kuno-development`:
strata-repeat-timeline-v3-20261025/trace/strata.1.json,
strata-repeat-timeline-v3-public-analysis-20261025.json,
strata-repeat-timeline-v3-final-safe-20261025.json and versioned harnesses.
V3 unit756b354c is V1, not the V3 invocation: inspect actual journal/unit for V3.
Final03:37UTC Intel42278912B, NVIDIAcompute0, all5faultclasses0,3HTTP200,
admission idle/unpaused/no recovery/wait0. No reset/driver/release/Vashti change.
Diagnostic source is composed but UNBUILT/UNTESTED at this checkpoint.
