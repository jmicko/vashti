# Complete shape-aware local prefetch policy

**Current status, 2026-10-24 UTC:** the complete policy passes numerical and
whole-model acceptance. It does not improve prompt speed. All older variants
remain available; nothing was installed. The implementation history below is
followed by the exact completed evidence and next source lead.

The preceding complete balanced-wide bundle passed1534 GPU controls but its
new dense and expert kernels spill registers. Preserve that whole variant.
This new opt-in policy changes ALL four matrix consumers together: large dense,
large expert GU/DOWN, macro32/64, and eight-slot cohort16.

One shared Rust CMMA core alternates TWO weight-A fragments inside each K16
iteration. B inputs and A operands are assigned in that iteration; only C lives
across the loop. Prefetch the next row's A before the current row's products.
Large cells retain32 members x32 outputs/eight C/WG256 and both B fragments.
Small cells use16 members x32 outputs/four C, doubling N64->N128 at the original
macro/cohort WG128/256/64. Small inputs still use the original initialized
padding and exact six owners. Independent8-half publication, finite checker,
pending ordered DOWN, guards, raw-bank staging and erasure remain.

Small N128 kernels have distinct exact runtime role identities and actual N128
grids/counters. Do not widen legacy name matching, pretend old N64 grids are
N128, or reuse stale owner/admission state. Keep every older selectable variant
and its controls. Ordinary inference/API/privacy/Vashti/install behavior is
unchanged. No new dependencies or driver configuration.

Compose the whole policy before live speed testing. Validate actual generated
module geometry, storage operands, source origins, current-iteration lifetimes,
publication and negative mutations, then paired numerical/whole-model/recovery
controls. Preserve first/repeated comparison and original CLEAN helper. Inspect
actual Intel assembly for spills; estimated cycles are not throughput. Acceptance
and performance remain pending until measured on the complete frozen build.

The complete source policy is now composed, including the actual producer
selection and exact N128 runtime/cohort grammar. One shared core serves all
four matrix tiers. No live tests or speed claims yet. Planned421 full/418 off
AppCPU retains every preceding control;18/86 runtime includes nine new exact
role/grid/owner/phase/forgery controls. All88 new generated matrix layouts get
real operand-origin, current-iteration and publication/negative audits. Fresh
1747 GPU controls retain1534 and add93 large +116 small +4 multi-slot pairs.
Both original full-model/compiler/recovery chains and original CLEAN gate remain.

First sourcea8e4e66e compiles. All40 new large module layouts pass, then the
small validator stops at the first cohort-GU module: its stride is a constant
expression in vec4-half pointer units, not a scalar INFO load. All preceding
macro modules passed. Preserve this failed CPU artifact. The forward validator
requires the exact role K/4 (2560/4 for GU,640/4 for DOWN), proves the actual
StorageBuffer pointer is vec4 of16-bit floats, and evaluates only the supported
constant/shift/copy expression. Wrong stride and vector-width mutations are
rejected; dynamic INFO validation remains for other paths. No kernel/math change.
A small CPU harness includes this actual production validator with only visibility
and test cfg adjusted; all45 generated modules available from the stopped first
run pass with their negative mutations. Full421/all88 acceptance still pending.


Forward source86c4994a passes421 full/418 off AppCPU and all88 new layouts;
release worker builds and passes the complete offline shader set and feature-off
18 runtime controls. The full runtime harness then finds two stale helper call
names in the new large-role TEST fixture. Production runtime and shaders are
unchanged. Fix those calls, retain the failed logs, and reseal the complete
same-source build/tests before any GPU work. No live test has run yet.

## Completed acceptance

Source `16054c835bd2125bc66c9a3f98580914aa8b7850`, worker
`e23a3bf3d647ecf00ed8c7dd0b832a733fe7874861f5d013382afe8fdc35f16e`:

- Same-source 421 full / 418 feature-off AppCPU controls, 18 / 86 runtime
  controls, 8 + 9 metadata controls and 10 native-source controls pass. The
  original 429 WGSL / 320 matrix layouts and all 88 new actual layouts pass.
- Fresh GPU execution passes all 1,747 controls: every preceding 1,534 plus
  21 large dense, 72 large expert, 116 small and four multi-slot pairs.
- Both original complete model/compiler/owner/capacity/erasure/recovery chains
  pass after the counter-harness correction described below. All requests
  produce four output tokens. These are prompt tests, not sustained decoding.
- Actual compiler export contains 228 programs. All ten new dense/expert/small
  programs have zero spills/fills. The 22 preceding small programs and both
  original QSA assembly hashes remain intact.

| Mode | Warm 73 | First 4,245 | Repeated 4,245 | Model load |
| --- | ---: | ---: | ---: | ---: |
| Normal scheduling | 3.727178218 s | 11.365037066 s | 9.308575798 s | 80.198236073 s |
| Sampled profile | 3.781607056 s | 11.265344339 s | 9.334742040 s | 81.705158747 s |

Compared with the preceding balanced-wide normal result, first/repeated time
is 2.97% / 1.28% longer. Compared with the earlier row-order normal result,
repeated time is 6.11% longer. These are individual observations, not controlled
variance estimates. The original CLEAN trigger was not met, so no new CLEAN
ran. Best preceding CLEAN 11.817666824 / 8.323282816 s and Strata's
5.5006 / 5.3515 s remain unchanged. No parity or production-selection claim.

## Preserved harness failure and forward acceptance

`kuno-local-prefetch-v1-complete-20261024` completes the fresh 1,747 numerical
cases and all three profiled model requests. Its historical cohort-counter
helper then wrongly expects the preceding balanced producer's counters. The
new real `expert_local_small_cohort_*` counters are separate; crediting the old
kernel would be incorrect. The exception occurs before the helper writes its
model receipt. Preserve that failed unit, its journal and operator artifacts.
Physical recovery passes at 16:22:11 UTC. This is not a GPU numerical failure
or a completed first model-acceptance chain.

The forward harness preserves the original helper byte-for-byte and appends
an explicit local-prefetch branch. It checks the original capacity, six-phase,
image-byte and launch partitions against the genuine new counters, requires
the original tile8 counts, and rejects nonzero displaced producer counts.
Fifteen CPU metadata controls cover three prior public profiles translated
only as test fixtures, plus four corruptions per fixture. Those synthetic
metadata fixtures are not new GPU evidence. All 17 original model-chain
scripts are unchanged except path relocation; the eighteenth operator entry
imports the completed numerical result from the EXACT SAME source and worker.
Its complete original implementation remains alongside it as `.original`.

The imported operator receipt SHA256 is
`a48373e7b4b4c64e37fb5fb2973b191204580f1a4533b527dbc3bad2ed816365`.
Source, binary, passed result, recovery and new populations are pinned before
hard-linking its unchanged artifacts into the retry. The outer wrapper still
checks all original and new numerical assertions. No numerical result from an
older worker is substituted. The retry explicitly records
`fresh_public_numerical_execution=false` and the same-worker provenance; do
not describe it as another fresh numerical execution. The controller's inherited
console line saying "starting fresh1747" is stale in this retry; the receipt
and provenance fields are authoritative.

The forward read-only preflight passes. Unit
`kuno-local-prefetch-v2-complete-20261024` exits zero, including both full model
chains and compiler export. The failed attempt and its artifacts are unchanged.
No Rust source or worker changed between these GPU executions.

## Actual costs and next coherent candidate

The new large dense programs have the same actual 340 / 343 instructions,
90 sends and zero spills as the preceding global-wide32 programs. New expert
GU / DOWN have 3,043 / 2,069 instructions, 553 / 381 sends and zero spills,
also matching their earlier global-wide32 summaries. That establishes removal
of the previous balanced-wide spill regression, not a throughput gain from
source-level prefetch spelling.

The same serialized layer24 sample shows:

| Family | Earlier row-order sample | Complete local-prefetch sample |
| --- | ---: | ---: |
| Large GU | 8.038904 ms | 8.063954 ms |
| Large DOWN | 3.344424 ms | 3.334012 ms |
| Macro | 17.328533 ms | 26.528896 ms |
| Cohort | 6.684164 ms | 7.891718 ms |

Macro/cohort costs are 53.09% / 18.07% higher in these samples despite zero
spills. Copy remains 29.171586 ms and dequant 17.715501 ms. Whole normal expert
time is 5.663845805 s versus row-order 5.139562731 s; attention stays near
2.43 s. Sample intervals overlap and serialize work: do not sum them into
normal prompt wall time or extrapolate one layer across both chunks.

Next inspect and compose temporal K-bank pipelining for BOTH N128/four-C
small consumers together. The current shared core keeps only C across K16;
the earlier small two-C path has genuine K32 loop-backedge operand carry.
The retained large balanced-wide core already implements four-C chronological
K32 carry with guarded refill, but previously spilled in its large expert
wrapper. Reuse/audit that genuine arithmetic for small macro AND cohort
wrappers while retaining this complete large eight-C path, authentic padded
inputs, N128 grids and all original publishers/guards/owners. Register pressure
in the new small context is unproven. This is a complete shape-aware candidate,
not an isolated rollback or an established explanation of the slowdown.
Keep every variant and audit actual emitted lifetimes/spills before claiming
an improvement. No implementation of that next candidate is claimed here.

## Recovery and artifact map

The completed unit's final 16:30:17 UTC snapshot and independent 16:30:55 UTC
snapshot report Intel 42,278,912 B, NVIDIA compute processes zero, desktop
620 / 621 MiB, all five kernel fault classes zero, all three services HTTP200,
and admission inactive/unpaused/no recovery/waiting zero/completed258. No GPU
worker remains. Boot/kernel are unchanged. Goal remains active and unmet.
No installation, release, reset, driver, API, canonical checkout or Vashti change.

Artifacts under `/media/jmicko/faststore/kuno-development`:

- `local-prefetch-root-build-receipt-20261024.json`; AppCPU full-v3/off-v2,
  release/offline worker-v2, runtime/deep-runtime-v2 logs. Build unit v3 passes.
- `strata-parity-local-prefetch-v2-20261024/`: exact 71-feature worker above.
- `strata-parity-local-prefetch-v1-20261024-operators/`: actual fresh 1,747
  numerical execution; v1 profile directory contains failed attempt recovery.
- `local-prefetch-forward-harness-v2-20261024/`: pinned original chain,
  additive counter laws, fifteen metadata controls and exact reuse manifest.
- `strata-parity-local-prefetch-v2-20261024-{operators,profile,normal}` and
  `strata-parity-local-prefetch-v2-20261024-expert-cohorts-public-summary.json`.
- `local-prefetch-v2-{acceptance-freeze,complete-isa-export,normal-safe-phases,
  profile-safe-phases,final-state}-20261024.json` and the independent final-state
  JSON. No v2 original-CLEAN result exists.
