# Native wide routed experts: source-only proposal

This is a proposed next candidate, not an implemented path, GPU result or change
to the currently installed worker. Finish the complete PV/bulk/loading candidate
and its original acceptance first. The proposal adds no API, configuration,
quantization, precision, request cache or new private-content persistence.

## Why this is the next plausible matrix change

The fixed-public `strata-native-fast-cpu-readyfix-20261020/receipt.json` contains
the same histogram for both long requests: 26,338 groups and 2,037,120 members.
The `256_plus` bin has 2,321 groups and 1,357,660 members, or about 8.8% of calls
and 66.6% of members. There is **no separate count for groups of 512 or more**.
This is public-fixture evidence, not a persisted real-user usage profile.

The installed native32 dense kernel already implements a full M128xN128xK32
tile with 512 threads, 32 actual SIMD16 subgroups, four accumulators per subgroup,
two increasing K16 steps per K32, two genuine operand stages and 49,152 bytes of
declared shared memory. Its actual dense numerical/physical acceptance does not
establish routed-expert correctness or expert register behavior.

In contrast, `strata_matrix.rs::native::run_expert` still selects the original
M32xN64 macro product for every group of at least 32 members. Covering a full
128x128 output rectangle with that product uses eight workgroups. Source-level
operand staging for a K64 range is 96 KiB over those eight groups, versus 32 KiB
for one proposed wide group. That is a threefold reduction in the staged operand
bytes for this rectangle, **not a measured memory-traffic or speed claim**.
Wide staging also changes barriers and occupancy, so only the complete candidate
can establish the net effect.

## Connected implementation

Use a new opt-in module, tentatively
`strata_matrix_native_expert_wide.rs`, alongside the accepted dense module.
Keep the accepted dense kernel unchanged. Reuse its actual typed vec4 staging
helper through a narrow visibility change, and retain its exact native32
accumulator geometry, chronological K16 products and publication barriers.
The new publication function consumes the existing six genuine expert owners:
input, FP16 weight image, GU/PART output, route map, full status plane and fault.
There is no contiguous DOWN scratch allocation and no post-scatter copy.

The host selector initially requires count >=256, at least one complete
128-member prefix, original fixed GU/DOWN dimensions, aligned real half-vector
views, the same stream and exact original map/status/fault extents. Check actual
CMMA F16/F16/F32 support, 512 invocations and X dimension, 49,152 shared bytes,
grid limits and the existing physical Graph subgroup admission. An unsupported
device/view returns false before launch and uses the entire original expert
path. Do not approximate a capability, force a pointer cast or weaken a guard.

For an admitted group:

1. Compute `prefix = count /128 *128` with checked bounds. Launch only full
   128-member tiles, with full 128-feature tiles (GU1280, DOWN2560).
2. GU stores to `global_member *1280 +feature`. DOWN resolves the authentic
   route at `map[1540 +map[1028 +group] +global_member]`, checks its original
   bound, and stores to `route *2560 +feature` in the full PART owner.
3. Finite values and zeros for nonfinite values retain fault bit128. Only the
   unique feature-zero publication updates `status[ROUTES +route]` using the
   original feature-zero finite check and `status[route] ==1`. A failure at a
   different feature must not invent a different status policy.
4. The remaining 0..127 members use the **unchanged original** M32 or M8/M16
   product, through small routed tail wrappers. The wrapper has an explicitly
   checked `member_base =prefix`, full original group count, and local remaining
   count. An authentic input subview starts at `prefix *columns *2`; its offsets
   remain relative to the original owning Handle. The epilogue adds member_base
   to both GU and route-map member indices. It never reruns or overwrites member0.

Keep the full original group/header and route validation; local tail dimensions
are arithmetic geometry, not new group ownership. The full map/status/fault and
output owners remain unchanged. Original native product helpers need only narrow
module visibility so the tail wrappers can call their actual implementations.
If a new wrapper cannot reuse those products without duplicating or changing
their arithmetic, report that source constraint before expanding the work.

`native::run_expert` gets one opt-in try-hook before its existing macro/narrow
dispatch. A false selection leaves the original path intact. A selected path
must enqueue the wide prefix **and** its tail as one coherent group result;
there is no partial-success fallback after a GPU launch.

The scoped expert-bulk collector must admit the exact new generated wide/tail
kernel identities in its existing matrix positions (3 and 5 of the six-operator
family). A prefix plus tail also means two real matrix commands for one logical
GU or DOWN operation. Therefore its current strict `launch_index %6` recognizer
needs a bounded, explicit continuation grammar for the two genuine matrix
dispatches, or the matrix wrapper must expose a sealed logical-op boundary that
retains every physical command. Do not simply broaden the name filter or treat
the extra tail command as an activation/dequant operation. Existing command,
metadata, ownership and ordered submission limits still apply.

The current compiled-public acceptance also expects exactly six physical
launches per group. That expectation remains correct for the current bundle and
must not be changed before its measurement. A future wide-plus-tail candidate
needs separate receipt fields for six logical phases, actual GU/DOWN pieces and
tail dispatches. Validate each matrix phase as either one original full-range
dispatch or an exact, nonoverlapping wide-prefix/tail pair covering the original
member count. Preserve Gather, DQ-GU, DQ-DOWN, GU, Activate and DOWN order. Never
report six real launches when there were seven or eight, and never credit a
partial matrix pair as a complete group.

## Numerical and ownership acceptance

The existing 14 routed expert controls are currently reachable only through the
CubeK feature and its direct `try_expert` call. Extract their public fixture,
original route builder/validator, independent CPU oracle and complete owner
scanner into a shared probe module. Keep the existing CubeK14 launch requirement
and reports unchanged when that feature is selected; add a separate native
runner under Intel-core support, without depending on CubeK or pretending a
private PromptScope exists. The audited public arena and same-client/full-owner
guards remain mandatory.

The native runner must exercise the actual new kernel and connected original
tail product, rather than a dense stand-in. Retain all original IEEE acceptance
(`abs1e-4 +rel3e-4`), nonfinite/fault128, high/noncontiguous route, initialized
status, full output/status/fault/source and padding checks. Add bounded full and
split cases at256,257,287,288,383 and384 members: pure full tiles, tail1,
M16-tail31, M32-tail32, tail127 and three full tiles. Keep the original smaller
32/33/64 fallback controls. Include nonfinite feature0/other-feature and invalid
routes in both the wide prefix and tail, not only the old33-member path.
The report must identify actual selected wide blocks and tail family, using only
the compiled fixed-public fixture; do not add real-request usage telemetry.

CPU controls precede live work: real generated modules across the existing
layout variants, native32 topology/shared size/K ordering, complete output-store
bijection, one feature-zero status store per member, checked owner subviews,
wrong member-base/header/route negatives and full/tail boundary geometry. Preserve
all existing original matrix/router/dequant/QSA/GDN controls. Then require actual
new routed numeric cases, real full-request erasure/rebind, staged thinking/tools
and semantic quality, terminal-failure recovery and a matched clean long run.
Do not infer expert register spills or speed from the dense operator ISA dump.

## Scope checkpoint before implementation

The core module and hook are bounded. The coupled bulk command recognizer is a
real dependency: current six-physical-command accounting cannot represent a
split matrix without an explicit continuation. Agree that small grammar change
before implementation. Do not add a new private authority, completion credit,
cache or replacement resource owner to avoid it.

Relevant current sources: `strata_matrix_native_wide.rs::{stage,publish,
dense_wide,selected,try_dense}`, `strata_matrix.rs::native::{macro_product,
narrow_product,run_expert}`, `strata_cubek_expert_probe.rs`,
`strata_prompt_experts.rs::enqueue_original`, and the new scoped collector's
`strata_expert_batch.rs::operator`.
