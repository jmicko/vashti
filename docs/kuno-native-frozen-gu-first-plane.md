# Count4 compact GU primary-plane experiment

This unused internal leaf probes the cost of the second Q8 activation plane in
frozen DASLab gate/up projection. It is not selected by an environment variable,
feature, API, parent policy or installed worker. It is an explicit approximation
and has no physical numerical, quality or speed acceptance yet.

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

## Source-only checkpoint

The candidate and its two focused regression tests are implemented in an
isolated worktree based on `62594d0`. Rust formatting and `git diff --check`
pass. **Tests, compilation, Clippy and GPU validation have not been run for
this leaf**: the parent owns the shared build cache and is keeping compilers
idle during the current normal compaction benchmark. The test descriptions
above state intended coverage, not a passing receipt. Parent integration must
run the focused tests and strict direct-device lint before any physical trial.
