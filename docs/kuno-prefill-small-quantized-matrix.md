# Fused quantized small-expert matrices

This optional internal bundle removes full decoded FP16 weight images from
groups with fewer than32 members. It changes no public API or model format.
The active pinned Strata Intel reference still creates those images; this is
a Kuno optimization after the reference execution port, not a missing Strata
cache. Current repeated public diagnostics produce186.3GB of small-group FP16
images out of262.2GB total for only6.78% of the useful expert members.
Those byte counts describe GPU image production, not PCIe traffic.

## Complete pipeline

Keep the original private80MiB working owner and six logical phases:

1. Original checked gather of expert input rows.
2. A genuine MAIN compute producer copies encoded gate/up roles into a compact
   cohort slab within the working owner.
3. A second genuine MAIN producer copies encoded DOWN. Its actual completion
   releases the RAW source; no future fused product reads that RAW bank.
4. A fused quantized GU matrix decodes weights into subgroup-private FP16 shared
   memory, then performs the original chronological K16 cooperative products.
5. The existing GU finite check and activation produces the original hidden
   FP16 values and status.
6. A fused quantized DOWN matrix publishes expert order for the existing
   pending-status/inverse/finite consumer.

Compact producers retain the original physical prototype/address/RAW/model
source bindings, loan-tier resolution, bounds and fault behavior. They copy
encoded bits without numerical conversion. Slab slots are logical cohort
ordinals; source RAW slots, including resident sentinel8, are separate values.
The new path must not weaken or reuse HOST-only copy admission for GPU sources.
Keeping compact producers on MAIN avoids blocking current work behind future
HOST uploads on COPY. Existing owner erasure and capacity accounting remain.

Products use four SG16 groups in64 threads, output N64 and M8/M16/M32 for
member counts1–8/9–16/17–31. This avoids decoding an expert's weights twice for
17–31 members. Each subgroup stages its own double K32 A/B banks with real
subgroup memory barriers. Original K16 accumulation order, explicit F16
rounding, eight-value nonfinite handling and checked partial publication must
be preserved. The largest planned shared envelope is26KiB including publication
scratch, below this device's48KiB limit without lifetime-reuse assumptions.
GU formats18/21/22/23 and DOWN formats20/42 cover every actual model layer.
All old decoder/image/product definitions remain as independent controls and
as the feature-off path.

## Integration and acceptance

Application feature `strata-small-quantized-matrix` composes existing small
publication, grouped experts, direct GU and ordered consumers. Runtime features
use `kuno-small-quantized-matrix`. Exact new producer identities and fused
matrix identities must form a closed grammar: compact producers cannot feed
old FP16-weight consumers, and old decoded images cannot feed the new kernels.
Retain original source/owner extents, RAW release, status/fault and whole-layer
completion checks. Counters must report actual compact bytes and launches;
do not continue counting removed FP16 image production as executed work.

Acceptance requires exact compact-copy controls from both real resident and
RAW sources, all six decoder formats against the original formulas and FP16
rounding, complete owner/status/fault checks, actual shader/layout validation,
and whole-model correctness plus physical recovery. Performance testing follows
the complete producer/scheduler/runtime composition. Preserve original677 GPU
controls and original CPU/runtime populations; no relaxed tolerance or speed
selection of partial pieces. Root owns integration/builds/GPUs; isolated source
tasks own kernels, cohort producers and native admission. No implementation or
performance acceptance is claimed by this design document.
