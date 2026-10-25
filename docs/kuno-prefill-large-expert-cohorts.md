# Bounded large-expert cohorts

Isolated source candidate after `6a69550b`. No GPU execution, speed acceptance,
installation, API or Vashti change. Preserve every existing selectable variant.

The current whole-prompt inventory has 2,344 groups of at least 256 members,
covering 1,360,382 of 2,037,120 routed members. Those groups still run individual
wide prefixes and residual tails. This candidate batches that entire producer,
consumer and owner lifecycle rather than changing the old regular-cohort limit.

Add an explicit large-cohort unit to the original two-bank planner. It consumes
consecutive original groups with 256..4096 members, at most eight, and admits
only a total of at most 4096 genuinely initialized padded rows. Existing sorted
group ordering, expert identities, within-expert route ordering, inverse map and
rank-ordered final reduction remain unchanged. Raw slots remain unique within
the real bank. No source tier, resident pair count or host-copy byte changes.

The distinct packed producer and consumers use the original 20 MiB X, 5 MiB H,
75 MiB eight-slot expert image and two 32 MiB raw owners. Prefix matrices retain
the actual large eight-C, 128-member tile arithmetic; residual tails retain the
original tail arithmetic and publication. They run as distinct authenticated
large-cohort roles with their actual offsets and grids. No routing of this tier
through the new small two-C cells, new precision, relaxed finite checks or
changed reduction order. Padding must be initialized by the actual producers
and may not overlap the next group's packed rows.

Compose the planner, packed gather/dequant/activation, prefix/tail matrices,
native INFO/phase/owner grammar, retirement and genuine production counters
together. Old regular32..255 and small1..31 contracts remain closed. Changes to
test populations must retain original membership rather than relabel old cases.

CPU validation covers real planning, capacity, packed offsets, member/inverse
identity and rejected malformed metadata. Actual shader layout and whole-output
GPU pairs against the original per-expert prefix/tail path remain mandatory
before numerical or speed claims. Full model, source/owner/erasure/recovery and
unchanged clean timing follow. Work in this checkout is CPU/source-only; the
parent task controls any eventual protected GPU acceptance.

## Source contract

Select `strata-large-cohorts` (builder `--large-cohorts`, requiring the explicit
regular and wide32 bundles). It composes with `strata-small-dual-cells`; it does
not change either old member range. The packed scalar ABI has the same 48 words
but distinct `StrataExpertLargeCohort*` operators and actual large matrix entry.
Every slot is `(original_group, real_members, Pad32_prefix, raw_slot_or_8)`.
There is no sorting or expert replacement in this stage.

A unit emits Gather, GU image, DOWN image, GU prefix, each present residual
family, activation, DOWN prefix, and the same present residual families.
Prefix tiles contain 128 members. Residual 1..8 uses the original 8 kernel,
9..31 uses 16, and 32..127 uses 32. Uniform surplus Y/Z work returns before loads.
The resulting chain has 6, 8, 10 or 12 real launches. The final actual DOWN owns the
original UnitEnd; the existing DOWN-image raw release stays where it was.
X/H only add packed input offsets. GU adds the same packed row offset; DOWN
continues writing the original map's ordered PART offsets and status routes.

The existing `native_expert_grammar` authenticates all original memory owners,
physical views, stream dependencies, complete INFO and exact grids through a
separate large grammar. Large counts are a subset of aggregate cohort counts;
regular and small counts remain separate. New public counters report large
batches, groups, padded rows, actual matrix/prefix/tail launches and matrix work.
The fixed public sampler appends `admitted_large_cohort_groups`,
`admitted_large_cohort_batches` and `admitted_large_cohort_launches` from the
complete authentic grammar. The old 22 fields keep their exact names and indices;
the total is now 25, and private requests remain outside this public sampler.
The original 5632 dispatch and 32 pipeline caps are not enlarged.

## Focused acceptance entry points

The existing public operator command now includes large producer, dequant and
matrix paired checks when the feature is selected. These reuse the prior
whole-owner comparison and transient-erasure helpers. Matrix fixtures cover
all residual families together, eight 512-member groups and the 4095/4096 row
boundary, both GU/DOWN, including poisoned input/invalid routes. Large image
fixtures check all eight distinct original experts and mixed GPU/HOST GU roles.
The old public controls still run afterwards.

CPU checks use a separate target directory:
`/media/jmicko/faststore/kuno-development/large-cohorts-cpu-target-20261025`.
The focused test filter is `large_cohort`; dependency grammar tests use
`cubecl-runtime` with `kuno-large-cohorts`. Actual matrix SPIR-V compilation
covers all eight original layout settings, original SLM limits, SG16 and
Vulkan validation. Its prefix proof keeps the original 8C dataflow/recurrence,
route, fault and publication checks, while admitting only the new fixed
640/2560-column vec4 strides instead of requiring the old dynamic INFO stride.

No protected GPU pair, full-model run, fresh/carry failure replay, owner/source
closure, erasure/recovery run or performance measurement has been performed
for this candidate. Passing CPU checks does not promote it to installed code.

## Source checkpoint validation (2026-10-25)

Focused AppCPU: **6 passed**, including all **64** actual prefix/tail SPIR-V
program/layout pairs, the eight actual WGSL producers, exhaustive 256..4096
prefix/tail partitioning, capacity/metadata negatives, unchanged map/inverse
ordering, bounded two-bank planning and genuine profile identities. Command:

```sh
CARGO_TARGET_DIR=/media/jmicko/faststore/kuno-development/large-cohorts-cpu-target-20261025 CARGO_BUILD_JOBS=4 cargo test --offline --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml --bin kuno-strata-rust --features strata-large-cohorts,strata-small-dual-cells large_cohort -- --nocapture
```

The first shader check rejected the new compile-time column stride because its
inherited validator required dynamic INFO dependence. The distinct fixed-axis
proof now accepts only the actual 640/2560-column vec4 strides, retaining the
original 8C recurrence and negative controls; all 64 layouts passed afterward.

Actual included runtime source fixture: **103 passed**, preserving the original
100 controls and adding the two LARGE controls plus
`dual_small_outer_grammar_accepts_genuine_scope_and_rejects_forged_or_unscoped_consumers`.
The latter fixes an existing complete-model blocker: the regular-role guard
rejected `dual_small` before the genuine small-cohort handler could consume it.
Only that guard moved after its actual handler. Disabled, forged, nested and
out-of-scope consumers still reject; genuine phase, grid and owner checks remain.
The new regression calls the outer `NativeStrataExpertGrammar`, checks an entire
eight-phase fixture and rejects unscoped, old-grid and forged consumers.

Runtime validation used the parent-exported
`large-cohorts-focused-runtime-cpu-20261025` fixture, with includes and patched
dependencies verified to point at this checkout, its original complete feature
set plus `kuno-large-cohorts`, `--locked --offline` and `--test-threads=1`.
Direct dependency-package tests cannot run from the application workspace;
the complete actual-source fixture supplies their required test context.

Still pending: the parent's exact full AppCPU 452/feature-off 446 population,
feature-off runtime 18, release/offline compiler inventory, sampled selector
metadata 9/10/10/11 controls, native source controls and protected GPU acceptance.
The public operator entry adds six producer, six multi-dequant and sixteen paired
matrix checks while retaining every existing control. Those 28 additions are
entry points, **not yet executed GPU results**. No installed version, memory cap,
API, request-content persistence or frontend behavior changed.
