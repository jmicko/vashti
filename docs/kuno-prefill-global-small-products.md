# Complete global small and medium expert products

SOURCE COMPOSED after accepted8a2652ec/full983; compilation and complete
acceptance remain pending. No performance claim. Compose the whole producer/consumer/launch/grammar
bundle before interpreting speed. Keep every prior variant and all390 CPU/
983 GPU controls. No API, Vashti, installation, capacity, precision, ownership,
status, privacy or erasure change. Target stays the same clean Strata comparison.

## Verified source and full-profile motivation

Active Strata oneMKL wrappers take original global weights first/transposed,
inputs second, and submit pure GEMM arithmetic. Opaque exact internal tiles
remain unproven. Kuno's large-prefix/dense path now uses this orientation,
larger subgroup cells and0 operand SLM. Small/medium products still use inputs
in A and weights in B. `small_direct::products::macro64_product` uses32KiB
operand shared memory (36KiB in the non-explicit compiler layout), WG128/SG16,
four accumulators/subgroup. Cohort16 uses WG64/four subgroups, two accumulators
each and2KiB publication scratch. Keep the original actual compiler proofs.

Accepted full-profile sample has cohort13.355ms and macro12.887ms, versus
wideGU8.076ms/wideDOWN3.306ms. This identifies a substantial remaining family,
not a sum of disjoint wall intervals or proof of a promised speedup.
The local-route bundle removes9376 encoded barriers but normal repeated
prefill remains9.360s. Keep that bundle; extend the arithmetic coverage.

Pinned reference `sycl/src/prefill/prefill.cpp:226,1190,3964` actually uses
DQ=2 dequantized-expert slots, as Kuno does. Its much larger raw stream ring
is a DIFFERENT resource. Do not increase the dequant ring on the mistaken
assumption that96/384 raw slots mean96/384 decoded FP16 images. Actual active
raw-ring configuration has not been independently reverified here.

## Coupled matrix and publication change

Add an explicit feature/build selector over the complete local-route bundle,
with new exact generated identities; retain old entrypoints and selectors.

For original full-small macro32 (32..63 members), keep genuine M32/N64 grid,
but use WG32/two SG, each32x32 with eight8x16 accumulators. For macro64
(64..255), keep M64/N64 grid but WG64/four SG with the same32x32 cells. Use
chronological K16, weights in A/input in B, bounded one-A reuse, column-major
stores and original F32 accumulations. Genuine rows remain1280/2560; no padded
row or fictitious member count and no full-prefix scope expansion.

Complete32-member subgroups load original input globally. Partial last32 must
mask every real input load and initialize a bounded per-subgroup K16 input
tile before CMMA. Use original count and original view, not spare arena bytes
as authority to read beyond logical input. Keep all synchronization subgroup
local; raw/partial branches may differ between subgroups in the same WG.
Fully inactive subgroups may exit only if no workgroup barrier is required.
Weights remain global. Exact shared sizes must come from actual compiler
layouts; finalize the bounded input/publication scratch before acceptance.

For existing cohort tile16 (9..31 members), keep genuine M16/N64 grid and packed
Z-slot/scalar ABI, but WG32/two SG, each16x32 with four8x16 accumulators.
Reuse ONLY the authentic already-initialized16-member padded image for that
cohort. Keep tile8 (1..8 members) on its complete existing path: its view has
only eight padded members and cannot support an unchecked16-member load.
No new payload allocation, repack, group-count fiction or half conversion.

The transposed publisher must validate original8-member halves independently.
In particular, a partial upper half MUST NOT cause a complete lower half to
lose its original raw/NaN/pending2 behavior. Direct-store only a whole valid
16-member fragment; otherwise use bounded per-SG scratch and preserve raw
complete8 halves, checked partial/invalid halves, untouched invalid sentinels,
sticky faults and original valid status rules. Original GU checking/activation
and ordered DOWN inverse/finite consumption remain mandatory. Pending2 is
never accepted output. No new global fault read or weaker AtomicOr proof.

Original wide-prefix tails remain their prior complete implementation unless
an explicitly specified tail extension is composed with its genuine shifted
input/member_base/full-count grammar and independent controls. Do not silently
reuse full-small scope for tail pieces. This proposal does not require that
additional extension to accept the small/medium bundle.

## Host/native integration and acceptance

Host selection, exact new GU/DOWN identities, real WG32/WG64 and original
physical X/Y/Z grids must change together with the matrix bodies. Original
group/header/count/role/INFO/source/nonalias/full-owner validation stays.
Native closed regular/cohort grammars must authenticate the genuine new roles,
packed scalar ABI, actual grids, matching Gather/activation/inverse owners
and exact ordering, and reject forged nested legacy names, changed owners,
wrong dimensions/roles, missing phases and feature-off attempts. Do not widen
the original command/host/retained-memory capacities or phase counts.

Keep profile families and semantic original group counts. Add distinct new
producer/physical-launch counters so partition laws cover old tile8 and new
tile16/macro products honestly; never claim an old kernel executed when only
its semantic role remains. Keep six original logical phases, actual dispatch
count limits and all source/release/rolling/erasure guarantees.

Before freeze, finalize exact additive test names/populations. Retain all390
AppCPU,18/59 runtime,8+9 metadata,10 native, compiler layouts and983 GPU cases.
Add independent complete/partial subgroup geometry and masked-input bounds,
both8-half semantics with nonfinite features and nonzero invalid sentinels,
actual SPIR-V operand orientation/chronology/real WG/scratch/CFG and production
AtomicOr classification, plus native positive/negative ownership/role/grid
controls. Use actual old GPU and independent F64 references over complete
owners/status/fault/sources/padding with unchanged tolerances. Cover count32,
33 and partial boundaries around64/128/255, plus cohort9/15/16/17/31 and genuine
multi-slot packed groups. No speed selection before the complete bundle passes.

Then run both original full-model/recovery chains, actual family partitions,
physical launch counts and all original capacity/reuse/rolling/erasure proofs.
Export actual ISA/spills and preserve original eight wide/two QSA controls.
Assess complete first/repeated normal/diagnostic results; run original CLEAN
when promising. Retain components and investigate remaining reference
differences even if this complete candidate alone does not close the gap.


## Implementation checkpoint before first CPU build

New feature `strata-global-small-products` / `--global-small-products` requires
local-route bundle, propagating `kuno-global-small-products` through native
runtime. New `native::global_small` has four exact macro32/64 GU/DOWN entries;
packed `cohorts::global_small::expert_global_cohort` retains the original48-u32
scalar ABI and authentic comptime DOWN role/header validation. Real WG32/WG64
and genuine original grids are coupled to the new code and closed grammars.
Original namespaces cannot alias these entries. GU checking/ordered DOWN and
all old limits/owners remain. Original tile8 and wide tails are unchanged.

Normal macros use eight C fragments/direct global operands. Checked branches
use two C fragments over the SAME subgroup cell, with fully initialized masked
16-member/K16 input staging only when the original input is partial. Cohort16
normal uses four C fragments; its checked branch reads authentic padded original
input. Its publisher independently preserves full8 halves under partial16.
Expected bounded macro scratch<=2048B WG32/4096B WG64, cohort<=2048B; actual
per-layout sizes still need emitted-module verification before freezing.

Five new AppCPU controls (including actual compiler/CFG/provenance/AtomicOr/
negative-module proof), four genuine native metadata controls and116 additive
single-slot GPU/F64 paired controls are written. The original390 AppCPU/983 GPU
remain. Three new mathematical test functions compiled separately and pass;
that is NOT full application or emitted-shader compilation. Current planned
AppCPU395, full-runtime63;116 new GPU controls would total1099, but the required
new genuine multi-slot packed-group GPU coverage is STILL PENDING. Do not freeze
or run live speed before this is complete. No new GPU run has been started.

Source files `strata_matrix_global_small{,_cohort,_validation,_tests,_probe}.rs`,
additive feature/host/profile/native-grammar changes and four native test cases.
Prepared CPU-only driver `build-global-small-app-cpu-20261024.py` retains all390
old exact names, runs the five new tests and actual compiler proof first, then
all395. It is allowed to check source while the multi-slot harness is finished;
no timing may drive direction. Keep failures/artifacts and repair the complete
bundle rather than reverting components.

## Packed multi-slot source completion

Source526ebcd1 passes all395 AppCPU controls, including real48 new compiler
layouts and negative-module proofs. No GPU/performance acceptance yet. The
remaining four multi-slot GPU cases are now composed: real eight-group packed
counts[1,3,7,8,9,16,17,31], both GU/DOWN and finite/compound invalid+NaN.
They compare the actual preceding small-direct raw/ordered producer with the
new global tile16 producer while retaining original tile8. They use one
original full output owner, a bounded zeroizing host reference, original
full-span GPU erasure between programs, complete status/fault/source/tail
checks and nonzero group/image/input prefixes. Original packed-pair cases
remain unchanged. Independent F64 controls remain in the116 single-slot cases.

Final additive population is120 GPU controls (116 single-slot +4 packed),
retaining983 =>1103;395 AppCPU retains390;18 feature-off and63 full runtime
retains18/59; original8+9 metadata and10 native controls remain. Final source
compilation, runtime tests, exact observed shader resources and all GPU cases
remain pending. No speed decision or GPU run has been made for this bundle.

## Complete accepted result (2026-10-24 UTC)

Frozen source `d36a393f418f34a2f0858bbc39b9a82c5fe8e09c`, worker
`50bfab9e6e91944df987a6894013e8b5999eccda7a84e683aeefbf90fa305f59`.
All395 AppCPU plus original390 feature-off,18/63 runtime,8+9 metadata,10 native,
all original compiler layouts and1103 GPU controls pass. The120 added cases
include116 independent full-owner/F64 and4 genuine packed eight-slot cases.
Both original18-script full-model/recovery chains, all original capacity,
logical/physical producer partitions, ownership, rolling/reuse and erasure
checks pass. Exact actual shared sizes across all8 layouts: macro32=2048B,
macro64=4096B, cohort16=1024B. The six new native ISA programs are zero-spill
at128 GRF; full197-program export preserves original wide/QSA controls.

Diagnostic warm/first/repeat3.759350146/13.620531729/11.691707270s; normal
3.761780781/13.525956001/11.620150934s. This is a complete-bundle regression
versus preceding normal11.411931752/9.360428477s. No CLEAN rerun; normal does
not beat the retained8.854793s trigger. Strata CLEAN5.5006/5.3515s and best
prior Kuno CLEAN12.471177197/8.394039699s remain unchanged; goal ACTIVE/UNMET.
Retain the complete bundle and prior variants. Do not revert coupled pieces
based on isolated measurements. Installed services were not changed.

Actual sampled layer24/count4096 matrix families: macro236 calls,12.887171ms
preceding ->55.219568ms; cohort278 calls,13.354772->22.373058ms. Wide GU/DOWN,
tails, dequant19.258208ms and OrderedCopy29.157433ms remain essentially
unchanged. Host capture also remains comparable. Intervals may overlap; do not
sum them as wall cost. Production actually selects4218 macro32 and6532 macro64
products,767640 physical macro WGs and3460 new tile16 launches PER ROLE in
each large request;4463 original tile8 launches per role remain.

The retained fallback computes two C fragments at a time and repeats the K
loop for feature/member subtiles. A partial subgroup takes this path even
when every real route is valid. This is a source-confirmed candidate for the
new arithmetic cost, not yet an isolated causal timing proof. Next compose
one full8-C/4-C chronological arithmetic walk for both complete and partial
tiles, authentic masked input loads and original independent8-half publisher;
keep the full raw fast stores. See the next coupled design before coding.

Protected complete unit `kuno-global-small-complete-20261024`, invocation
`2528d58ded784cd4b9ff6772cadf415d`, terminal exit0.05:00:40UTC: Intel50,688,000B,
NVIDIA compute0/display575MiB, all five kernel fault classes0; Kuno/Ollama/
Vashti HTTP200, admission normal/completed178. Both GPU runs terminal.
Bounded four-hour refresher PID2712660 started about03:40UTC; do not extend
indefinitely. No driver/reset/reboot/API/UI/config/install/release changes.

Preserved CPU wrapper failures: initial final-build invocation supplied a
wrong source SHA and failed before compilation. Next wrapper incorrectly
expected `... ok` on the same line under `--nocapture`; actual shader test
passed. V3 completed all395 invocations, then its resource parser missed the
first marker after the libtest prefix. V4 separately reuses the same completed
395 log and binary hash, parses all48 actual markers, and completes390 off.
No numerical tolerance/guard was changed; original failures and logs remain.

Artifacts under `/media/jmicko/faststore/kuno-development`: full source/build/
freeze `global-small-root-*`, accepted worker `strata-parity-global-small-20261024`,
`global-small-final-v4-app-cpu-safe-20261024.json`, original v3 full395 logs,
`global-small-complete-driver-20261024.log`, full1103 public summary,
`global-small-{profile,normal}-safe-phases-20261024.json`,
`global-small-complete-isa-export-20261024.json`,
`global-small-{completed-operators-isa,diagnostic-comparison}-safe-20261024.json`,
`global-small-final-state-20261024.json`. All original18 scripts are retained
through an exact path-only forward copy; additive helper honestly partitions
old tile8/new tile16 producers while retaining every original image-owner law.

Read-only pinned-reference evidence `global-small-reference-routed-ring-evidence-20261024.json`
checks actual clean ready receipt, explicit client and immutable container
image environment, native pack index and source. Model has48 expert layers
(not97;97 is a large request's repeated layer execution count),512 experts
each. Reference resident slots11608/24576 imply nonresident share>=0.527669,
well above the0.10 default stream-all threshold; no client/image override.
Thus this clean comparison uses raw routed STAGE8 and decoded DQ2. The
conditional96/384 raw ring is not an active reference advantage here.
