# Public single-member grouped MoE experiment

This laboratory source candidate follows the existing native chunk ownership
contract. It changes no production route, model math, expert cache capacity or
private allocation/runtime/host limit. Numerical and lifecycle acceptance on
both GPUs is required before timing or selecting it for a full-model experiment.

`grouped_moe::Geometry` now permits explicit public tile1. It has fixed member
capacity1 and Z=1 rather than tile8's capacity8/Z=1 or tile4's capacity8/Z=2.
Tile1 is not selected from a private token count. Its runtime packet must still
contain count1, a one-token wave, and one member per expert group. Host validation
rejects other values before launch; both grouped shaders additionally reject
them through their runtime map guard. The existing default model allocation
continues to accept tiles4/8 only and defaults to8 through the unchanged runner.

The native routed workspace exposes `projection_tile()` and
`set_lab_single_member_decode(executor, arena, &CompletedNativeProjectionBoundary)`.
The setter requires a healthy matching client/admission, full owning audit,
completed Idle policy with at least one chunk, and no pending grouped session.
Only the model constructs the opaque boundary proof. It replaces the existing
48 public geometries in place; it adds no owner, device allocation, map or fence.
Model/CLI integration is separate and must check the completed all-family cursor
before supplying that proof.

Gate/up shared scratch becomes512 bytes per workgroup instead of4096; down
becomes256 instead of2048. Active routes retain the original packed decoder,
column order,64 lanes, unconditional barriers, sign-stable SiLU and rank0..9
merge. MAX8/MAX80 private pages, route/probability maps, status regions, complete
finite sweeps, fault folds, source windows and original bank retirement stay
unchanged. These source counts do not establish a speed benefit.

The existing `grouped-moe-chunk-probe` now includes46 actual Q4 gate/up with
Q5/Q8 down cases: the previous40 plus count1/tile1 at capacities16/64/96, including
the existing segmented boundary fixture. It also runs14 fixed-public compact
operator cases across F32/F16/BF16/Q4_K/Q5_0/Q6_K/Q8_0 at tiles8/1. Compact public
bank rows repeat two block-aligned original GGUF slices; they are not full model
matrices. Independent `TensorType` decoding, the original64-lane tree and
rank-order reference check every activation/partial/output. Actual immutable
wave/pending bank protection, irregular offsets/strides, inactive/page tails,
read-only/public bank preservation, complete status maps and mandatory owning
erasure/audits remain checked. Receipts contain aggregate numeric results only.

Host tests cover invalid count/wave/member policy, unchanged full-page envelopes,
completed-boundary guards and independent rank arithmetic. Checked/unchecked
offline WGSL cases now include tile1 for every decoder and original segmented
shapes. These tests and the new GPU cases are unrun in this source-only handoff.
