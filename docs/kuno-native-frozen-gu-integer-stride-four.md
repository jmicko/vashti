# Frozen integer GU group stride four

This is a separate, default-off scheduling candidate for the existing explicit
two-residual-Q8 integer GU leaf. It changes scheduling only. Original packed
decoding, activation planes, source identities, runtime MAP group/rank/member
addresses, block accumulation, eight-lane tree, SiLU, ready status and all
original DOWN/finite/fold/transaction gates remain mandatory and unchanged.
Source acceptance does not establish physical numerical parity or speed.

The original `launch(...)` and `Kernel::Project` retain one routed group per
workgroup. With the actual public envelope30, each of five source partitions
dispatches80 row tiles times30 groups:12,000 GU workgroups per layer. At most
30 real routed groups are active across all partitions, so at most2,400 are
useful. Each inactive workgroup still incurs scheduling and uniform guards.

`launch_group_stride4(...)` has the same arguments and owners. Its separate
`ProjectStride4` pipeline dispatches exactly `(80,4,1)`,64 threads per group:
1,600 total GU workgroups per layer. Each y group considers original routed
groups `g=y; g<envelope; g+=4`. The envelope remains the caller's checked1..80
value, carried in INFO word6 (`p.g`); default `Project` continues to require
that word to be zero. The actual MAP groups remain bounded by10 times public
count1..3. Neither the public count, routes, source identities, envelope nor
memory addresses enter pipeline keys. The kernel discriminant, dtype and
row bytes identify the distinct scheduling pipeline.

Before any codebook write or barrier, every lane uniformly scans its assigned
groups with all existing validity/readiness/in-shard guards. If none is valid,
the complete workgroup exits and ready status remains unchanged. Otherwise,
the original literal codebook initializes once, followed by one barrier. A
uniform loop rechecks every group and runs the same extracted arithmetic
body for each valid group. Scalar/vector accumulators are freshly initialized
per group; every reduction cell, including inactive member cells, is assigned
before reduction. The body retains the original initialization barrier and
tree4/2/1 barriers. An additional barrier after output ensures every lane has
finished reading its reduction before the next group resets shared cells.
Different valid groups retain distinct original output route addresses.

The prepared64KiB owner, fixed `(1,3,1)` prepare dispatch, complete128-group
audit, storage binding count, workgroup-memory maxima and all MAIN/HOST/INFO/
runtime reservations are unchanged. No new persistent allocation, source
page, CPU route readback, transfer, or hidden cache is introduced.

Both schedules expand one WGSL arithmetic BODY; the alternative main only
substitutes its runtime group variable in frame/output addressing. Decoder
improvements inside that body can be reviewed independently and apply to both
schedules without duplicating arithmetic. This candidate contains no new
weight-decoder math.

Focused source tests validate both pipeline variants with Naga/SPIR-V and
workgroup-zero initialization disabled, identical extracted arithmetic,
all four actual GU dtypes, exact codebook writes/shared bytes, uniform
pre-initialization exit, per-group reset/final fence, and original group/source
coverage exactly once across every public count and envelope. These checks
cannot replace the unchanged original CPU13/reference gate, full real repair,
or matched complete language timings on the physical Intel device.

Parent integration and physical receipts are recorded separately. The default
path remains original scheduling until an explicitly immutable source policy
selects this candidate; no automatic scheduling fallback is permitted.
