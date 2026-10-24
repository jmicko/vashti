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
