# Larger per-subgroup matrices with an ordered route guard

Implemented at d081b38a; final acceptance audit pending as recorded below.
No clean performance/parity claim. Backend benchmark feature, retain all complete variants. No API/UI,
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

## Revised guard composition: preserve the existing tape limit

Source review before integration found the existing ORIGINAL Gather already
validates every real route against p.count*10 and atomically ORs128 into the
same fault owner. Its map binding is read-only. Native grammar binds the
matrix to that original Gather's exact group/input/map/fault views; map has
no writer between Gather, GU and DOWN. Existing queue WRITE-to-READ/WRITE
hazards must order fault validation before the raw prefix's atomic fault read.
The prefix still performs the original header/role/count/piece checks.

The initially proposed two extra production guard dispatches would require
up to512*(10+3) command slots, exceeding ORIGINAL5632. Do NOT expand that limit.
Re-use authentic Gather as the production route guard instead. A standalone
numerical-probe launcher, which has no captured Gather, queues a distinct tiny
route guard with original map/fault and actual count/group; it cannot be
admitted into a native captured expert batch. No hidden cached validation.

Use subgroup plane_all on the original atomic fault-zero read so every CMMA
branch is uniform even if an unrelated later fault arrives. If all lanes see
original sticky fault zero, run eight-C arithmetic, direct column-major
stores, then (after all C fragments die) simple original DOWN pending2 writes
for all32 real members in the single column-zero subgroup. If nonzero, use
accepted two-C publication/math over four serial64x64 subtiles with16 SG and
2048 F32/8KiB scratch. This retains exact8-member raw/finite/status halves and
invalid-route storage untouched, even if previously nonzero. The compiled
expert shader still allocates8KiB fallback SLM; do not call it0SLM. No operand
SLM, new flag, fake count, new owner or new production dispatch.

A prior or unrelated sticky fault conservatively selects the checked branch.
Fault never clears during this scope. A bad route in THIS group must already
have set fault before any raw workgroup; a later unrelated fault only switches
some other workgroup to the equivalent conservative branch. Preserve genuine
header faults and existing GU-check/activation/DOWN-inverse finite consumers.
Pending2 never means accepted output. Do not raw-write invalid rows then clear
them: original invalid-route storage must remain untouched.

Closed exact new namespace for genuine GU/DOWN roles, N128/WG256 physical grid,
original6 owners/7 scalars and exact original Gather/activation/tail phase order.
Refuse a missing Gather, changed map/fault/input/owner/INFO/grid, fake nested
legacy names and a standalone guard inserted into a captured group. Keep
original6 logical phases and all physical launch/command/host/retained bounds
unchanged. Count production Gather-guarded prefixes separately from standalone
probe guards; these are enqueue counters, never execution/completion authority.
Original native operator admission must close the new namespace before loose
legacy suffix checks. Feature-off forgeries must remain rejected.

Proposed acceptance inventory (finalize exact names before freeze): retain
all378 AppCPU,18/55 runtime,8+9 metadata,10 native, original compiler sets and890
GPU cases. Add8 AppCPU (3 dense/3 expert/1 actual profile identity/1 guard-source
and ordering coverage),4 runtime closed original-Gather/role/owner/order controls; planned
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


## Implemented full run, 2026-10-24 UTC

Frozen source d081b38a1c56fc09e6b4978ff6d6e2f89d6b0b5b, worker
b85d46489d1b3a206ec81c8b424b629f1834de8d590dd63987714151c300672e.
Actual387 AppCPU (original386 plus the real GPU-sentinel initializer module),
18/59 runtime,8+9 metadata,10 original native tests passed. Actual matrix and
standalone guard modules validate. Sentinels are initialized by a tiny scalar
GPU shader over the ORIGINAL output owner, never an offset host upload or a
new output allocation. Its eight compiler layouts are also checked.

Protected invocation305dee630d794bb49183ad8b9898349c
(`kuno-global-wide32-seeded2-20261024`) completed both original18-script chains,
including operator controls and physical recovery, then EXITED1 in the final
connected audit at line691. The asserted partition required nonzero counts
for BOTH new-wide32 and old-transpose dense products. Its exact actual counters
still need protected receipt inspection before revising this assertion. Fused
HC/GR may bypass the real-GR320 fallback entirely in the whole-model fixture;
this is a source lead, not yet a verified explanation of the failed assertion.
Do NOT call the unit successful, publish a final959 acceptance proof or relaunch
GPU/model tests merely to fix a receipt assertion. Preserve the frozen failed
wrapper and validate the SAME immutable receipts with a narrowly reviewed
read-only audit revision if their real counters justify it.

Completed diagnostic first/repeat11.502265204/9.424782167s; normal
11.453113618/9.369922453s. Warm requests3.787430749/3.736089751s respectively.
No original CLEAN run: normal9.3699s did not meet the existing8.854793s trigger.
These instrumented full results are a modest improvement from preceding
normal9.811339312s, not the clean Strata5.3515s target. Goal ACTIVE/UNMET.

Unit21m46.878s, CPU22m13.055s,30.6GiB peak, swap0. Independent02:33:05UTC:
Intel50,688,000B, NVIDIA compute0/display596MiB, all5 fault classes0; Kuno,
Ollama and Vashti HTTP200, admission normal/completed172. The test processes
are terminal and GPUs physically recovered. No installs/settings/resets.
Artifacts under `/media/jmicko/faststore/kuno-development`:
`wide32-seeded2-driver-20261024.log`, frozen `global-wide32-root-*`,
`strata-parity-global-wide32-20261024-{operators,profile,normal}` and
`global-wide32-final-state-20261024.json`. Final ISA/phase exports were NOT
written because acceptance stopped first. Do not infer absence means no runs.

### Preserved rejected attempts

All sources/workers/receipts/logs remain in separate development directories:

| Directory prefix | Result |
| --- | --- |
| wide32-uniformity-source-revision | Initial CPU build stopped before GPU to require subgroup-uniform fault branch. |
| wide32-output-borrow-compile-rejection | One original-owner borrow compile error; no GPU. |
| wide32-transposed-fallback-proof-rejection | Wrong static input/weight-load inventory; corrected to5input/10weight for the full mixed raw+checked module. No GPU. |
| wide32-fullspan-fixture-rejection | Original private upload rejected offset sentinel write; guards retained. |
| wide32-systemd-git-scope-rejection | Exact-worktree Git trust absent in systemd; per-process trust only, no global change. |
| wide32-systemd-worker-traversal-rejection | Root0700 artifact directory prevented runuser worker traversal; artifact directory ownership corrected. Subsequent preflight correctly refused overwriting its failed directory. |
| wide32-closed-arena-rejection | Actual private arena closed; reset error masked the original cause. Replaced fixture host seed with original-owner GPU initializer and report original case errors before reset. Do not claim the masked original cause was established. |
| wide32-sentinel-ir-import-rejection | Missing IR ElemType import in new actual compiler control; no GPU. |

Provenance caveat: the build-driver script copied into
`wide32-closed-arena-rejection-20261024` was copied AFTER the driver had been
revised for387 tests and artifact ownership. It is not the exact earlier386
runner. Original failed receipts/source/worker/logs are unchanged; an exact
older386 driver copy remains in `wide32-fullspan-fixture-rejection-20261024`.
The new local-route candidate is separately described in
[kuno-prefill-wide32-local-routes.md](kuno-prefill-wide32-local-routes.md).
