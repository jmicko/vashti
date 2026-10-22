# Fixed-role expert macro64 specialization

Complete optional source continuation,2026-10-22. No API/model/context/config/
frontend change; Vashti:none. Keep ALL prior ordered-frontier, expert/cohort,
raw-bank, prefetch and native-prepared work. No installation or speed claim
at source implementation.

## Evidence and limits

Complete a761763e/worker89a27c5b passes253CPU/15runtime/429WGSL/296layouts/
all350GPU and both original18-script chains, zero spills and physical recovery.
Normal first/repeat9.654518545/8.354937695s; profile9.655510196/8.510095030s.
ONE clean45/4245/4245 comparison gives first11.053525683/repeat7.924977071s
versus installed10.897223643/8.712129955s. Warm improves9.03%, but FIRST fails
original5% selection; no repeat to choose a favorable sample and no install.
Full clean receipt strata-parity-head-frontier-gap-fixed-clean-20261022-public-summary.json.
Strata clean5.3515s remains unmet.

A separate explicitly NON-clean CPU/PMU diagnostic used the same public fixture
and original first-long request interval, sourcea761/worker89a. It records only
instruction pointers, no stack/register/branch/payload samples; temporary perf
data stays in /run and is removed. Top40 rows cover only40.21% of rounded CPU
user samples: at least5.08% are explicit CubeCL IR/optimizer/Naga symbols, and
20.30% libc memory/allocation symbols. Do NOT describe those partial rows as
a whole CPU attribution or all libc cost as compiler work. Compute-engine
active51.49%, copy-engine18.25% are engine activity, not FLOP/bandwidth use.
This is diagnostic, never a favorable replacement clean result.

New64-row expert programs cannot execute in the45-token warmup because group
count<=45; long requests use6532 such products. Runtime pipeline creation still
compiles missing generated programs through CubeCL IR/Naga/Mesa. This motivates
reducing their generic work. It does not prove compilation explains the entire
first-use gap or guarantee a gain.

## Complete connected source

`strata-macro64-static-geometry` / `--macro64-static-geometry` explicitly requires
the COMPLETE ordered-frontier bundle. Both authentic roles have full64-wide
N/K tiles: GU1280x2560 and DOWN2560x640. The original map/header/count/group/
role shape checks remain FIRST. A surplus X workgroup returns before any staged
read; the original masked kernel also wrote nothing for it. Valid grids are
unchanged and are still checked by the same closed real runtime grammar.

Only after those guards, stage strides use the role's fixed dimensions. The
weight stage is one complete vector load/store path; the old generic masked
N/K branch remains available when the feature is disabled. The input full/tail
choice remains, and every token>=count is still zero without a read. K is a
multiple of64 and every guarded current/next stage is within the validated
original columns; input per-vector K checks are factored into that proof.
Keep the ORIGINAL dynamic outer loop bound and all four chronological K16
guards/products, preventing accidental constant-loop unrolling. No FP change,
extra allocation, altered shader ABI/owner/route/status/fault/precision guard,
barrier removal or capacity reduction. Original CMMA16products,128threads/SG16
and shared49152 logical/32768 lifetime envelope stay.

Two new CPU controls independently enumerate actual weight vectors/offsets/
surplus-X gating and every legal member/K stage/tail/prefetch boundary; retain
all253 original controls and actual16-layout compiler/Cmma/barrier checks.
Intended255App,15actual runtime,429WGSL/296layouts,ALL350GPU (including all20
original fixed-shape macro64 F64/full-owner/fault cases), BOTH original chains
and unchanged zero-spill/compiler/math/recovery gates precede performance.
Actual6532 production products per long request still prove selection.
No CPU compilation during hardware timing. One NEW-source clean comparison
may follow completed correctness/recovery; prior failed comparison is retained.

Initial source6e6dbb62 CPU compilation rejects mixed constant/runtime expand
types in the stage-stride bindings. Explicit usize::cast_from lifts each fixed
constant into the same CubeCL NativeExpand type as its original runtime arm.
No hardware test ran; preserve initial compiler diagnostics and rerun255 CPU
controls before worker/offline/GPU work.
