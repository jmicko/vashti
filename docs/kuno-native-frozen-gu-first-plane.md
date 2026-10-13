# Count4 compact GU primary-plane experiment

This unused internal leaf probes the cost of the second Q8 activation plane in
frozen DASLab gate/up projection. It is not selected by an environment variable,
feature, API or installed worker. The fixed-public experiment was rejected by
unchanged full CPU13 checks and its compile-time parent selection is now false.
It is an explicit approximation with no quality or speed acceptance.

`frozen_gu_integer::four::launch_compact_first_plane` requires the same original
count4 compact geometry and full owners as `launch_compact`. Both entrypoints
share the complete host validation and dispatch construction. The separate
`ProjectCompactFirstPlane` kernel has a distinct identity/name. Its only shader
differences are removal of the residual packed-dot terms, residual plane loads
and residual scale/add terms for gate and up. It keeps the first Q8 codes and
F32 scales, original packed weights/decoders/codebooks, exact eight-lane tree,
SwiGLU, actual original route/rank addressing, compact group lists, barriers and
full output/status checks. No expert is replaced or forced to match a reference.

Prepare continues to compute BOTH planes in the same full64KiB owner; both
planes retain their existing finite/readiness/padding audits and erasure. No
memory charge, scratch layout, input-copy behavior, INFO metadata or dispatch
count changes. Original compact GU, all count1..3 tails, DOWN and MTP arithmetic
remain unchanged. There is no automatic fallback after numerical failure.

A primary Q8 plane has per32-element quantization error bounded approximately
by `amax/254`, whereas the second residual plane improves this to
`amax/64516`, before floating-point rounding. Nonlinear/routing effects can
amplify that difference. The older one-pass CPU experiment on the Ollama
Q4_K/Q5_0/Q8_0 model failed its original128-ID gate; that is a caution, not a
recorded rejection of this DASLab GPU candidate.

Source tests lower every actual GU dtype with explicit workgroup initialization
policy, verify the unchanged64-lane dimensions/barriers/shared bytes, distinct
kernel identity and removal of exactly half the projection packed-dot calls.
They compare all four member positions, both projection fixture phases and all
four formats against independently dequantized original weights and direct
primary-plane F64 sums. Those tests prove leaf semantics, not model quality.

Any parent experiment must be explicit in its frozen source/artifact receipt.
It must pass the unchanged original full CPU13 vectors, carries and greedy
checks (`0.006 + 0.0003 * abs(reference)`), real repair, all owning/source/router
proofs and cleanup before normal timing credit. Keep failures; do not loosen
those tolerances or force reference routes. If accepted, compare normal
compiler-idle whole-model timing separately from a profiled diagnostic. This
leaf needs no Vashti/API/installed-default change.

## Validation and rejected physical experiment

Root integrated leaf `3bb2c20` as `fd9c319`: focused tests and strict Clippy
passed. Frozen source `fdd2732` selected only count4 compact GU through
`FIRST_PLANE_PUBLIC_PROBE`; both probe receipts expose that compiled setting.
The full direct-device suite passed 1,340 tests with one ignored, and formatting
and strict Clippy passed. Those checks do not establish model-level accuracy.

Physical v195 build SHA256
`28e21d4107b56b200f8dc842fa24c634644a3c9c5e457f5c20245ade599e1ee7`
passed the original cold CPU13 and warm initial prefill but failed the warm
full carry fixture with `fixed public thirteen-position DASLab carry fixture
rejected`. No threshold changed and no normal speed test followed. Failed-owner
and both device cleanups completed; HTTP services and actual GPU baselines
recovered, with no new classified kernel faults.

**`FIRST_PLANE_PUBLIC_PROBE = false` is required at this checkpoint.** Retain
the leaf only as an unselected approximation experiment. Count1..3, DOWN,
original model weights and routes remain unchanged. Evidence is in
`v195_primary_plane_gu_rejected` in the aggregate optimization receipts and
`native-next-v195-first-plane-warm-intel-20261012` under the development base.
Do not infer this candidate meets the existing numerical contract or reactivate
it without a separately agreed precision policy. No installed/API/Vashti change.
