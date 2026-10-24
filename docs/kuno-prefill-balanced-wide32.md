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
