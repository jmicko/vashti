# Larger per-subgroup matrices with an ordered route guard

NEXT PROPOSAL ONLY after accepted1d5d0adb; no implementation or performance
claim. Backend benchmark feature, retain all complete variants. No API/UI,
installed-service/driver/model/dtype/capacity/owner/erasure change. Goal remains
same fixed public first/repeated Strata comparison, target5.3515s repeated.

## Evidence, inference and motivation

Pinned MIT Stratafb58e0d `sycl/src/prefill/gemm.dp.cpp` active BF16/f16 wrappers
submit a pure W/X/Y GEMM to oneMKL. Kuno's expert programs also contain original
route/finite/status/fallback work. Accepted transposed dense programs are210/213
instructions and54 sends; GU1426/237, DOWN1721/289. These are whole-program
static populations, not measurements proving all fallback code runs normally.
Both programs are now zero-spill; changing orientation alone did not close gap.

Captured Strata AOT names include32x64 and32x32 families with256-register
allocation capacity and mostly0SLM. Names/grid fit larger subgroup tiles, but
opaque oneMKL's exact internal ownership/tile is unproven. Do NOT describe the
proposed tile as an exact reconstruction of its opaque implementation. Current
Kuno native programs allocate128GRF. Installed Mesa package26.2.5~kisak1~n.
[Mesa's documented threads/EU controls](https://docs.mesa3d.org/envvars.html#intel-driver-environment-variables)
apply to Xe3+, not an established B70 override. The Aug2026
[large-GRF commit](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/cb671c51db1407f82c366b32642a7271224ceedd)
changes the Intel executor, not proof that the installed ANV compiler can select
256GRF. No private/global driver change or override is authorized by this note.
Do not substitute an untested environment flag for an engine optimization.

Working hypothesis: increase data reuse per subgroup while separating ordinary
arithmetic from publication/failure live ranges, within the current128GRF path.
Compose all changes below before full speed interpretation; no component rollback.

## Dense and expert arithmetic, one complete candidate

Feature/name proposal `strata-global-wide32` over the full transpose bundle.
Genuine M128/N128/WG256/SG16:16 subgroups, token=(sg/4)*32,
column=(sg%4)*32. Each subgroup computes32 weights by32 genuine members using
8 F32 C fragments of8x16 (4 weight fragments x2 member fragments).
Use original weights in A and inputs in B; store each C column-major with
original output stride and correct member+16 and column+8/16/24 offsets.

Use a chronological K16 loop. Hold two B fragments, load one A fragment just
before its two products, then reuse that A variable for the next weight8.
This deliberately bounds operand liveness while retaining8 C accumulators;
not a claim that an earlier bank optimization is generally wrong. Both earlier
complete two-bank variants remain. No padded input, private fake count, new
weight owner, repacking, precision change or operand SLM. Full row admission
requires128 alignment. Authentic GR320 remains the complete transposed N64
path, preserving its five real64-column tiles and original full-GR credit.
Original count128 prefix plus exact original remaining-member tail stays.

Dense finite consumer, original full-call bounds/capabilities/INFO/strides,
nonalias and genuine storage extents remain. Expert shapes/headers/prefix128
stay exact. New physical expert grid rows/128 x prefix/128 and actual WG256,
not reusing N64's rows/64 or its30*logical-block workgroup formula; new factor15
because GU10 and DOWN20 X tiles per original logical pair. Keep the original
logical M128 block counter separate and unchanged.

## Ordered guard and checked branch are required together

Normal expert branch must not combine eight live C fragments with a full
per-element invalid-route publisher. Before EACH selected new GU or DOWN prefix,
queue a tiny genuine route guard using original6 logical owners and original
7 scalar parameters. It validates header, role/shape, original count/group,
base0 and exact count/128*128 prefix, then checks every real prefix route.
Fault writes OR128 into the ORIGINAL sticky fault owner, never clear it.
For DOWN, write pending2 only for valid prefix routes into the original DOWN
status half; invalid-route slots stay untouched. GU status is unchanged.
Header failure must not access route/status beyond original extents.
Original tail executes as before. No host readback, hidden route cache or new flag.

The raw arithmetic prefix reads that SAME sticky fault after a real queue
WRITE->READ/WRITE dependency from the guard. If zero, all prefix routes were
checked: run the eight-C arithmetic and direct column-major stores with no
per-element route/status publisher. Pending2 is not accepted output; existing
GU check/activation and DOWN inverse/finite consumers remain mandatory and
run only after prefix/tail completion. No metadata-only guard may authorize it.

If nonzero, use a checked fallback preserving the exact original8-member
raw/finite/status semantics. Reuse the accepted transposed two-C publisher and
K16 math, processing four serial64-member x64-column subtiles within the genuine
M128/N128/WG256 region (16 subgroups, not the old32-subgroup mapping). Read only
original input/weights. With16 subgroups,2048 F32 slots suffice for the same
128-F32 per-subgroup publication scratch:8KiB, no operand SLM. This allocation
still exists in the compiled expert shader even when its normal branch avoids
it; do NOT claim an actual0SLM expert program. Every scratch extent/barrier and
handoff must be proved for the new geometry; all old variants stay unchanged.

A prior sticky fault conservatively selects checked fallback, even with valid
routes. Fault never changes from nonzero to zero during this prefix, and the
normal branch must not introduce a new fault writer after the guard. Existing
header-fault behavior remains. This avoids a race in branch selection.

Do NOT implement unconditional raw writes followed by zeroing invalid rows:
the old producer leaves invalid-route output storage untouched, including
previous nonzero contents. Likewise do not sanitize all16 members when only
one8-member half is invalid. Original pending2/finite/activated-GU status rules,
all real source owners, capacities and tails remain exact.

## Closed integration before GPU performance

Use new exact generated guard and prefix identities. Explicitly admit guard
only in the current GU/DOWN matrix phase, before its prefix, with authentic
original6 owners and7 scalar values, group/input/map/status/fault and weight/
output views. Keep a pending guard record; prefix must consume the exact
matching guard once. Missing, duplicate, wrong-role, changed-owner/INFO/grid,
legacy prefix after guard, unfinished guard and feature-off forged namespace
must reject. Pending guard may not escape a group, tail or recording.

Original6 logical phases, matrix-piece counts and M128 block credits remain.
Count real guard enqueues separately; propagate count through runtime receipt,
actual WGPU graph records, selected/retained/reused completion and public-only
counters. Update physical launch equations by actual guards, never fabricate
virtual phases or silently remove resource/launch/retained-capacity checks.
Inspect `native_expert_batch` operator admission as well as grammar; one
classifier passing is not the entire native path. Ordinary source/erasure
ownership and borrowed-wrapper lifetimes remain in force.

Proposed acceptance inventory (finalize exact names before freeze): retain
all378 AppCPU,18/55 runtime,8+9 metadata,10 native, original compiler sets and890
GPU cases. Add8 AppCPU (3 dense/3 expert/1 actual profile identity/1 guard-source
and ordering coverage),4 runtime closed guard/role/owner/order controls; planned
386 CPU,18/59 runtime. Additional actual dense/expert/guard modules across8
layout combinations need real CFG proof, raw/fallback branch partition, same
original INFO, chronological K16, output stores, exact physical grids and
native positive/negative checks. No relaxed original tests or tolerances.

Add21 genuine dense cases and48 paired expert cases: retain the new32 expert
controls and add two preexisting-fault controls at257 members (finite and
nonfinite column0), both roles and old/new launchers,8 cases. Generalize original
fixture only through additive initial_fault() with0 for every old control; OR
initial fault into the unchanged expected sticky result. These force the
checked branch with valid routes as well as route-failure cases. Planned951
GPU controls before the sentinel cases below. Also add invalid-low/high8-half
with nonfinite column0 and a nonzero sentinel ONLY in that invalid member’s
original output rows, both roles and old/new launchers:8 more cases, planned959
GPU controls total. The old producer must preserve that sentinel too. Update
the independent full-owner oracle for precisely that original interval; leave
all inactive padding/source/other-member checks exact. Do not accept all-zero
fixtures as proof of untouched storage. Ordinary32 expert pairs,8 initial-fault
and8 sentinel controls make48; plus21 dense retains890 for959 total.

Only after complete source/compiler/numeric/metadata acceptance run both
original18-script whole-model/recovery chains, new guard/prefix/physical-grid
partitions, original capacities/reuse/rolling/erasure and all ISA exports. Keep
8 old zero-spill wide kernels and2 QSA hashes. Assess full normal/diagnostic
first/repeated times; original CLEAN only when promising. Static send/spill
counts and component times are diagnosis, not a substitute for the whole run.
