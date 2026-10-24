# Balanced large-matrix bundle

This backend-only experiment composes the complete accepted row-order/carry
bundle with a new large-matrix path for BOTH ordinary projections and routed
expert GU/DOWN. Existing kernels and their numerical controls remain available.
There is no API, installed-worker or Vashti change.

The new workgroup has512 threads,32 subgroups of16, and the same128-token by
128-output tile. Each subgroup owns16 tokens by32 outputs, using four genuine
8x16 accumulators rather than eight. Weight-A remains row-major and input-B
column-major. Two actual K16 operand banks are consumed chronologically in each
K32 step. Bank0 is loaded before the loop, bank1 before bank0 products, and bank0
is refilled only if another iteration exists. No speculative out-of-bounds loads,
changed reduction order, relaxed finite checks or synthetic input rows.

Dense output uses the original owner and finite consumer. Expert output keeps
the original map, six owners, guard, prefix/tail partition, GU finite checker,
ordered DOWN pending status and independent8-member invalid-route publication.
The new subgroup count needs16KiB of fallback publication scratch. Original
tail kernels, padded small paths, source staging and privacy erasure remain.
Native role classification must admit exact new names and reject nested legacy
forgeries. Physical capabilities, actual shader geometry and owners remain
independently checked; compiler metadata grants no runtime authority.

Acceptance requires actual generated-module structure/layout and SPIR-V
validation, complete output/source coverage, all original numerical controls
plus new dense/expert pairs, whole-model semantic/counter/recovery checks, and
actual Intel compiler evidence. Do not select or abandon coupled components
from individual timings. First/repeated CLEAN comparison uses the original
45/4245/4245 public fixture and no overlapping compilation or profiling.
Implementation and performance acceptance are pending.

First sourcecae34ff4 compiled; coverage/capability controls pass, but its validator
incorrectly assumed scalar-SSA CMMA handles. Forward0e78291e still rejected the
actual generated module. Its exported SPIR-V establishes that CubeCL uses
Function-storage matrix variables and loads/stores, including loop backedges.
Both failed CPU artifacts remain. The next validator uses the original carried
kernel's genuine reaching-definition analysis to prove exact source-bank sets
and four chronological C chains. No arithmetic/kernel direction change.

The exported actual FP16 module now passes the production validator and its
binding/geometry/INFO/store negatives, plus new actual weight-A and C-storage
crosswire rejections. This is a small CPU harness including the production
function with only visibility changed, linked to the actual current WGPU rlib;
it is not whole-model, full-layout or GPU acceptance. It also found that the
new CFG parser must end a block at OpFunctionEnd to retain the real final
loop-backedge branch. Full418 and all-layout acceptance are still pending.
Artifacts: balanced-wide32-app-cpu-v1/v2-20261024 (failed); actual-dense-fp16-v2
SPIR-V/disassembly; actual-validator-v1..v6-20261024, with every failed attempt
preserved. All under /media/jmicko/faststore/kuno-development. No GPU job yet.


## Completed validation, measured regression, and next coherent policy

Source980f37fc52221aa8c60faa0eba44811dc98694f1 / worker
28e0af67e1d1fd6ca9281bc0c45eaf90d753c59e9d1db1d825973dfa520c3e0f
passes418 full/414 feature-off same-source AppCPU,18/77 runtime,8+9 metadata,
10 native source controls, original429 WGSL/320 matrix controls, and all40 new
actual matrix layouts. Fresh all1534 GPU controls pass: every original1441 plus
21 new dense/F64/whole-owner pairs and72 expert/fault/status/source/tail pairs.
Original tolerances, capacities, finite checks and erasure laws are unchanged.
BOTH original18-script full-model/compiler/recovery chains pass. No numerical
receipt from a preceding worker substitutes for these executions.

|Mode|Warm73|First4245|Repeated4245|Load|
|---|---:|---:|---:|---:|
|Normal|3.346464047s|11.036851219s|9.190642528s|78.855768317s|
|Sampled profile|3.355177501s|11.084851310s|9.220993140s|81.672101530s|

Four outputs each; no sustained-decode claim. Compared with the preceding
normal10.554253471/8.772859676s, these individual first/repeated observations are
4.57%/4.76% LONGER, not variance-controlled. Keep the complete variant; no
isolated rollback, performance acceptance or production selection. The original
CLEAN trigger was not met, so no new CLEAN ran and no old CLEAN receipt is reused.
Best CLEAN11.817666824/8.323282816s remains the preceding worker's observation;
Strata5.5006/5.3515s is still unmet.

Actual218-program compiler export contains22 unchanged small kernels with zero
spills; original two QSA hashes remain equal. New dense modules have3:6 spills/
fills versus old0:0. New GU28:346 and DOWN11:77 versus old0:0. New dense399/402
instructions versus old340/343; GU4533 versus3043, DOWN2424 versus2069. All remain
128GRF. These are actual compiler records, not estimated or measured CPU/GPU time.
Estimated-cycle numbers must NOT be used as throughput measurements.

Matching serialized layer24 sample: wide GU8.038904->12.233487ms, DOWN3.344424->
4.759371ms. Macro17.328533->17.348739ms and cohort6.684164->6.691877ms are stable.
Whole normal expert interval5.139562731->5.429490025s. This evidence points to
register pressure and the changed large tile's reuse; it does not establish a
counterfactual saving or justify adding overlapping GPU intervals together.

Next compose a complete LOW-LIVENESS, SHAPE-AWARE prefetch variant, retaining
all older paths. Both large projections AND large experts should keep their
original32-member x32-output subgroup reuse/eight accumulators, with two weight
fragments alternating INSIDE each K16 iteration, rather than ten long-lived
loop-carried operand matrices. For BOTH macro32/64 and cohort16, audit the actual
current N64/two-C cells and compose N128/four-C cells with the same alternating
weight-fragment core. This doubles the small output tile's input reuse while
preserving chronological K16 arithmetic. Current actual macro/cohort WG128/256/
64, padded input producers, six owners, independent8-half publication, checker,
ordered DOWN, packet guards and erasure remain. New exact role identities and
N128 native grids/counters are required; never relabel old N64 grids as valid.
Keep all complete variants, implement the whole policy BEFORE live speed tests,
and inspect actual compiler spill/liveness evidence. No implementation of that
next policy is claimed here. Cold pipeline preparation remains a separate lead;
no forged inference, completion or resource/admission state is allowed.

The full unit exits0. Final14:45:42UTC and independent14:49:58UTC: Intel42278912B,
NVIDIA compute0/desktop621MiB, all5 fault classes0, Kuno/Ollama/Vashti HTTP200,
admission inactive/unpaused/no recovery/waiting0/completed254. No GPU worker
remains. Existing bounded sudo refresher3062861,parent3556, expires17:23:42UTC;
verify its live identity/cache before reuse. No reset, driver, boot, installation,
release, API, canonical checkout or Vashti change. Goal ACTIVE/UNMET.

Artifacts under /media/jmicko/faststore/kuno-development:
- balanced-wide32-app-cpu-v3/off-app-cpu-v1-20261024 (418/414).
- strata-parity-balanced-wide32-20261024 (70-feature release worker), root build
  receipt, exact8+9 selector and10 native-source controls, fresh18/77 runtime.
- strata-parity-balanced-wide32-v1-20261024-{operators,profile,normal} and the
  complete expert-cohorts-public-summary.json (1534 and both original chains).
- balanced-wide32-v1-{acceptance-freeze,complete-isa-export,normal-safe-phases,
  profile-safe-phases,final-state}-20261024.json; independent-final-state JSON.
- All earlier failed CPU/parser/harness/preflight attempts remain. Initial
  preflight failures were helper construction/expected-population guards, before
  any GPU run; actual completed preflight and first full live attempt pass.
