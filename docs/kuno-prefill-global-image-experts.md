# Coupled global-register expert and FP16-image contract

Backend-only experimental variant. No inference API/configuration, Vashti,
installation or release change. Keep the complete quantized-cohort variant and
all its controls. Do not judge partial producers against different consumers.

## Confirmed remaining work

The complete accepted direct-global dense variant9638aeb8 still spends about6s
in expert processing. Same sampled public layer's quantized cohort matrix path
is32.921752ms, whereas the earlier accepted complete FP16-image path sample was
12.709929ms plus its real dequantization cost. Whole-model normal FP16-image
repeat8.854793s also beat quantized K32's9.306249s. These are reasons to compose
an image-based global-expert variant, not promises that samples add linearly.
Strata's measured busy GEMMs use global operands and mostly no SLM; its opaque
oneMKL implementation is not claimed to be reproduced exactly by this design.

## Complete variant

Use an explicit benchmark feature strata-global-image-experts. Keep quantized
code/compiler/numerical controls available if also compiled; actual production
small cohorts instead use the authentic original FP16 image producer and its
matching original image matrix consumers. Do not relabel compact bytes or omit
dequantization. Keep exact eight image slots, same80MiB working roots and25MiB
X/H views, original addresses/read leases/raw-ring post-consumption boundaries,
original cohorts' authentic8/16 padded members and closed grammar. Existing
grammar already supports independent old-image/quantized populations; no widened
admission or new private retained owner is necessary.

Compose direct-global two-bank operands for genuine large-expert M128/N128
prefixes with original512/SG16 admission, source views and chronological K16
F32 sums. Four accumulators, original row-major input and col-major weight
strides; both real A/B banks and guarded final loads. No operand SLM or whole-
workgroup handoff. Original subgroup publication scratch is permitted only for
unchanged invalid-route finite/fault fallback. GU raw complete publication must
still run original finite activation before status is accepted. DOWN retains
original expert-ordered pending status2 and mandatory original finite/inverse/
combine closure. Keep all original small/direct tails and their owner views.

Use genuine distinct entry identities, exact producer metadata/closed grammar,
full preflight before prefix dispatch, original host resource/CMMA/vector/grid
checks and anonymous public-only path counters. No inference privacy/fault or
capacity relaxation. Other compiled controls must still select original entries
explicitly, independently of the new production variant.

## Acceptance

Preserve all363 original current CPU controls, original runtime18/49 and
metadata8+9; add actual source/compiler/numerical checks for the new products.
Keep all768 original GPU controls. New operator cases must compare full original
owners, source immutability, route/status/fault poststate and F64 tolerances,
including both GU/DOWN, original prefix+suffix counts, nonzero weights, poisoned
values and invalid routes. All layout masks and genuine native SG16 semantics
must be validated; report unavailable hardware honestly.

Run both original whole-model proof chains and diagnostic/normal modes with
actual image-cohort selections, no production quantized launches, genuine global
prefix use, original ordered finite consumer, rolling/reuse/capacity and physical
recovery. Retain compiler/ISA evidence and zero-spill original wide/QSA controls.
Only then perform a matched CLEAN comparison if promising. No component rollback
based on an isolated timing; preserve complete source and explain any remaining
full-pipeline gap.

## Status

Contract composed; implementation and all new acceptance are pending. The last
accepted complete source remains9638aeb8. This is not performance acceptance.
