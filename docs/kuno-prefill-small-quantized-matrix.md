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

## Composed source checkpoint — 2026-10-23

Frozen source `888d8b205071e98214cf3075fb7e4efd0ac07ea0`, worker
`a3192db31630a59cfff5713ff1bf303efcd4feec082f2d5840c23fb6d0922df3`,
composes compact producers7ac52735, runtime admissiondec5ef70,
root integration105716f3, original activation metadata repair362f5717,
matrix producersf1ea330a, typed shift repair80382bdf and constant-loop repair888d8b20.
All343 selected AppCPU controls pass, as do18 feature-off/47 full runtime,
8 original metadata and a separate9-test feature-enabled selector population.
The original compiler populations pass unchanged, plus144 fused matrix layouts
and6 compact-copy modules. Actual shared allocation14336/18432/26624 bytes fits
48KiB hardware in all8 compiler layouts. The6 original quant formats retain
explicit bit rounding, eight-value finite checks and original K16 ordering.

Protected GPU validation is running as `kuno-small-quant-20261023` against the
complete bundle, with swap/core dumps disabled. Planned population719 retains
all677 prior controls and adds30 compact-source plus12 original-format matrix
comparisons. This paragraph is not GPU acceptance or a performance result.
The two forward18-script chains preserve original numerical, ownership and
reuse assertions; their counters now explicitly describe packed bytes and
10-launch packed cohorts. Do not describe these counter checks as unchanged.
The original frozen harness and failed attempts remain intact.

Evidence under the development directory: `small-quant-root-build-receipt-20261023.json`,
`small-quant-root-acceptance-freeze-20261023.json`, `small-quant-harness-20261023/`,
`check-small-quant-connected-20261023.py` and `small-quant-connected-20261023.log`.
Attempt1 failed Rust constant visibility/shift types; attempt2 was an external
harness missing its feature declaration; attempt3 reached offline expansion and
caught mutation of an unrolled nibble-shift constant. All are preserved with
source/worker evidence. No GPU ran in those failed attempts.
No installation, API, model, context, Vashti or driver change. Goal remains unmet.
