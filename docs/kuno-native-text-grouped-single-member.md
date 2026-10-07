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

The existing `grouped-moe-chunk-probe` now includes52 actual Q4 gate/up with
Q5/Q8 down cases: the previous40 plus generic and Q4Block count1/tile1 at capacities16/64/96, including
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

## Public Q4 block gate/up candidate

`Geometry::new` keeps `GatePolicy::Generic`, including generic tile1 for a
same-binary control. The explicit `with_q4_block()` constructor admits only
tile1 and two Q4_K gate/up projections. Its public `GatePolicy::Q4Block` alone
selects the new compile-time branch; no token count, slot, route or input changes
a shader key. Other formats and generic tiles1/4/8 keep their original decoder.

The candidate copies the existing scalar `moe_block` field reconstruction.
Each lane loads the original Q4_K header and three scale words once per256-value
block, then consumes four quant bytes. Columns remain
`block*256 + pair*64 + lane`; factor/quant/minimum arithmetic, accumulation,
the64-lane tree, sign-stable SiLU, original-rank output and status stores remain
in their original order. All runtime shard/route/address guards, finite sweeps,
MAX8/MAX80 owners, maps, limits and full erasure/audits are unchanged. This
removes repeated explicit header conversion; it does not establish speed.

`Workspace::gate_policy()` reports actual public policy.
`set_lab_single_q4_decode(executor, arena, &CompletedNativeProjectionBoundary)`
has the existing healthy/completed/full-audit owner checks and returns tile1.
It validates all48 replacement geometries in fixed public stack storage before
installing any, so an incompatible late layer leaves every layer unchanged.
The existing single-member setter still selects generic tile1.

Each of the six new original-model operator cases follows its matching generic
tile1 case after full arena reset and bank cleanup. The Q4Block case compares
every active activation, down partial and output with independent decoding/tree
and core references, scalar GPU, and the retained generic GPU snapshot at the
unchanged tolerances. Full status, inactive/page tails, read-only/public banks,
original ledger and owning erasure remain required. Only one full Zeroizing
generic host snapshot survives between cases; no GPU owner overlaps. Its actual
lengths/capacities are checked against1,671,168 bytes and explicitly included in
the17,543,168-byte host scratch bound under the unchanged32MiB cap. Every failure
or unwind wipes the retained snapshot. Receipts add public gate policy and
generic comparison error, and require52 total cases including six Q4Block.

New host tests cover public-policy rejection, all-or-error48-layer replacement,
independent Q4 field/column/lane arithmetic at shard edges and actual retained
snapshot capacity. Checked/unchecked offline shaders retain all existing generic
load/binding/runtime/barrier invariants and add the Q4Block header-reuse case.
No build, tests, GPU execution or timing was run for this source handoff. Parent
must integrate protected model controls and establish both-card numerical and
full-model fresh/cancel/recovery acceptance before timing the candidate.

The separate `apps/kuno/benchmarks/text/validate-native-q4-operator.py` validator
accepts only this current52-case receipt and the exact14 format cases, with no
legacy fallback. Its `validate_receipt(receipt, expected_layers=(q5, q8))` API
requires caller-pinned original layer IDs (currently6,0); card, manifest, binary
and cgroup proof remain separate. The typed ordered inventory independently
selects policy, including six immediately paired generic/Q4Block tile1 cases.
Every declared guard, limit, bank page charge and complete cleanup count is
exact:64 successful combined arena resets including recovery,12 failed grouped
resets and14 compact resets yield153,944,064 bytes across1,078 full MAIN buffers.

Reported error maxima must be finite nonnegative f32-range numbers, with a
generic comparison required only for Q4Block. Zero error is not an acceptance
requirement. The producing operator retains its per-element
`1e-4 + 3e-4*abs(reference)` checks and core
`1e-3 + 3e-3*abs(reference)` checks. Maxima omit reference magnitudes, so the
validator cannot reconstruct those relative comparisons or prove them without
the producer's independently established run identity. Its strict schema and
inventory do not establish full-model, opaque driver erasure or speed acceptance.
The existing numeric-boundary validator and archived formats are unchanged.
