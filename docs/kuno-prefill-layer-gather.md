# Whole-layer gather and exact LARGE tile dispatch

## Resumed focused validation, 2026-10-25

The user returned both GPUs and refreshed sudo. Focused compile attempt3 found
one test-only module path error in the allocation assertion; it now uses the
explicit `crate::strata_prompt::layout` path. Attempt4 builds and all four new
AppCPU controls pass. The actual included-source runtime fixture builds and all
105 controls pass, retaining the previous103 and both new layer controls.
Logs: `layer-gather-focused-checks-20261025/app-v3.log`, `app-v4.log`,
`runtime-v1.log`. These are CPU results, not numerical GPU or speed acceptance.
Full456/452 AppCPU, release/offline shader inventory, selector controls and
all2085 original/new numerical GPU controls remain pending.

The prior frozen LARGE source's saved complete receipts now pass a separately
versioned read-only root wrapper. It retains the original direct-GU lower bound
on unused descriptors while allowing zero when all selected groups are genuine
cohorts and standalone direct-GU/wide counts are zero. Original failed wrapper
and receipts remain. No GPU run was repeated for that metadata-only correction.

Source-only candidate based on `a1d48249`. No benchmark, installation or API
change is implied. The isolated LARGE candidate remains untouched.

Select the additive `strata-layer-gather` feature / `--layer-gather` build flag.
It requires the complete LARGE and dual-small/regular bundle. Historical paths
and their operator fixtures remain available with the new protocol unselected.

Gather the exact original routed mixed-FP16 rows once per expert layer into a
contiguous 277.5 MiB X owner with Pad32 zero gutters. Every selected GU tier reads its original map prefix
from this immutable layer owner. GU and H remain unit-local at 20/5 MiB; DOWN
retains the original ordered PART output and final rank-ordered reduction.
Original arithmetic, expert choices, route order, finite/status checks, raw
bank releases and physical source dependencies remain unchanged. No old
per-unit Gather dispatch is issued, including no placeholder/no-op dispatch.

The new X view extends the existing MoE overlay to approximately 779.002 MiB;
the phase root grows 642→780 MiB. Reserve exactly 138 MiB additional shared MAIN
only for this feature and keep the full 512 MiB driver margin in physical
admission. This matches the increased private root price so the expert cache
budget and 11983-pair baseline remain unchanged. Root0 and both 32 MiB raw banks
are unchanged. This is an explicit peak-memory policy, not hidden budget reuse.

The authentic native grammar binds the actual layer X/map/source/fault owners
once and authenticates each subsequent unit's real first image and complete
producer/consumer sequence. X remains immutable through every unit and command
batch; original owners survive all submissions and failures. Legacy fallback
that could overwrite or repack X is rejected before execution in this mode.
Old ordinary operator fixtures keep their old grammar and do not receive false
layer-gather credit.

LARGE GU and DOWN dispatch only the sum of actual per-group prefix/tail tiles.
Each workgroup resolves its original group and local tile from the same bounded
eight records. Activation likewise visits only actual initialized packed rows.
The layer producer initializes each group's complete Pad32 gutter to zero,
so direct CMMA loads retain the original padded-input semantics. Original route
and inverse offsets stay separate from the new padded-X prefix table.

Compose layout/admission, layer producer, every selected GU consumer, genuine
native ownership/lifetime and exact grid grammar, release/counters/CLI together.
CPU mapping/negative/schema/actual shader controls and GPU paired numerical
entry points must cover all selected families, unaligned map prefixes, suffix
requests and invalid/poisoned data. Preserve every original control. Full model
correctness, source/copy/erasure/recovery and unchanged normal/CLEAN comparison
remain required before any speed claim. Compilation/GPU runs are on parent hold.

The source policy changed before consumer implementation from compact 200 MiB X
(+60 MiB root) to padded 277.5 MiB X (+138 MiB root), following parent review.
This avoids introducing per-K masked CMMA loads. MAIN derives its extra charge
from actual selected-minus-original rounded root sizes; it is not a magic cap.
No live physical admission or numerical/speed claim has been made.

Implementation details and validation entry points (source only, not yet run):

- `MAP[43008..43520]` stores the separate per-group Pad32 X prefixes;
  `MAP[43520]` is the total padded row count. Original grouped members and
  ordered route inverse remain untouched. All actual consumers validate this
  table before global-X loads. INFO word13 selects the new publication; words14
  and15 remain zero. Every historical paired reference stays at word13=0.
- The producer dispatch is exactly `(5, padded_rows, 1)` with256 lanes, covering
 1280 packed words per row. Both dimensions are bounded by65535 even at56832
  padded rows. The genuine native grammar authenticates the same grid, actual
  source precision, map,277.5MiB X,5MiB H and fault owners. Unit first-image
  dispatches enter the original consumers without a unit Gather dispatch.
- Small tile8, dual-small tile16, dual-regular and all LARGE prefix/tail GU
  kernels distinguish the global input prefix from the unit output prefix.
  DOWN reads original local H and publishes through the original route map.
  LARGE flat dispatch keeps eight fixed records and resolves only real family
  tiles; its activation uses the same actual packed-row span.
- Four new application CPU identities contain `layer_gather`: padded mapping /
  inverse and corrupt-prefix negatives; exact physical layout/MAIN charge;
  actual WGSL validation / bounded dispatch; flat family-tile enumeration.
  Two new outer native grammar identities cover complete ten-unit layers,
  partial-layer final rejection, owner/schema/grid/precision/fallback failures.
  One selector control checks actual five-phase units plus one layer producer.
  Existing452 application and103 runtime controls must remain present.
- New GPU entry points preserve the original512MiB numerical arena:8 paired
  whole-layer Gather controls (149/4096 rows, F32/F16, ordinary/poisoned),2 flat
  activation pairs,4 selected small pairs,16 regular/LARGE pairs. Matrix tests
  build global X only for the GU side; the preceding per-expert/old-cohort
  oracle still reads its original local input. Original output/status/fault
  owners and full source/untouched-tail comparisons remain. The producer pair
  reuses one277.5MiB GPU output after full verified erase and retains its public
  oracle in zeroizing host memory within the original400MiB reference limit.
- Public source inventory adds exactly one WGSL producer. Original matrix
  programs still compile under all original layout variants; new scalar/grid
  guards must pass the existing native source proof rather than weakening it.
- New profile fields: `expert_layer_gather_launches`,
  `expert_layer_gather_rows`, `expert_layer_gather_padded_rows`,
  `expert_layer_x_owner_bytes`, `expert_native_layer_gather_launches`, and
  `expert_large_cohort_activation_workgroups`. The existing LARGE matrix
  workgroup counter now records the actual sum grid in the new mode. The
  sampler appends `admitted_layer_gather_launches` as field26 only when built
  with the feature. Actual per-unit launch counts fall by one; one producer is
  added per layer. No logical phase or original group is omitted from receipts.

Pending: focused CPU compilation and controls, complete original/new CPU
inventory, feature-off compatibility build, native selector/source proofs,
protected paired GPU numerical controls, fresh physical admission, whole-model
correctness and source/copy/erasure/recovery, then unchanged normal/CLEAN timing.
No compilation, GPU work, service changes, installation or speed measurement
has been performed for this bundle while the parent GPU supervisor is active.

Checkpoint: stopped immediately at the user's request on2026-10-25.
Source remains uncommitted in `kuno-layer-gather`, based on `a1d48249`.
After the parent released the compiler hold, focused CPU compile attempt1
failed because `MetadataBindingInfo` intentionally has no Clone implementation.
Fixed the new layer adapter to validate the original custom INFO schema and
clone only its scrubbed data into a new original `MetadataBindingInfo::custom`.
Attempt2 was deliberately terminated during dependency compilation before the
application test binary completed. Its process group3645964 exited143; a fresh
process inventory confirmed no cargo or rustc processes remained. No new CPU
control, runtime control, shader validation or GPU probe has completed.

Retained logs and exact resumable command preparation:
`/media/jmicko/faststore/kuno-development/layer-gather-focused-checks-20261025/`
contains `app-v1.log` (real compile failure), `app-v2.log` (interrupted compile),
`features.json` (prior77 features plus layer-gather), `app.py` (focused runner),
and `runtime/` (new actual-source runtime fixture, not run). Use a fresh log
name when resuming `python3 .../app.py layer_gather app-v3.log`; do not overwrite
failed or interrupted receipts. Its distinct target is
`/media/jmicko/faststore/kuno-development/large-cohorts-cpu-target-20261025`, with
2 build jobs. Parent owns subsequent full acceptance/release and GPU work.
A source-only `git diff --check` passed before the compile attempts. No speed,
correctness, completed rewrite, installation or release claim applies to this
unvalidated source checkpoint. Frozen LARGE and installed services were untouched.
