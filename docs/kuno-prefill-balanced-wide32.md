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

First sourcecae34ff4 compiled. The coverage and capability controls pass; the
actual-module control rejects a validator that assumed physical CFG block
serialization was product execution order. Preserve that failed CPU artifact.
The forward validator follows real SSA C dependencies, resolving only copies
and degenerate phis, and still requires four separate chronological two-product
chains with carried/current operand banks. No kernel arithmetic change.
