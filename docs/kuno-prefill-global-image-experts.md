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

## Composed source checkpoint

Complete feature now includes genuine separately generated global GU and ordered
DOWN entries, authentic two A/B banks, no operand SLM, original16KiB subgroup
fallback publication, original host preflight/prefix/tails, original image-cohort
producer/consumers, exact role/ordered layout grammar and anonymous path counters.
Quantized numerical/compiler cases remain compiled independently. Added three
AppCPU controls (two actual-module/geometry controls and real generated profile
identities), two runtime grammar controls,16 real layout/CFG shader proofs and
24 paired original-GPU/F64 controls; original363/49/768 retained. Planned
366 AppCPU,18/51 runtime,8+9 metadata and792 GPU cases. Complete source is not
built or accepted; no timing claims or partial GPU tests.

The first composed366 AppCPU controls (including actual16 new modules/CFG) and
18 feature-off runtime controls pass. The full metadata fixture initially lacked
its feature-forward declaration; it failed before running any full runtime
control. Preserve that build attempt. The source audit also aligns the original
FP16 byte counter with actual image production even when quantized controls
remain compiled: count all real groups once, not only non-cohort groups. This
is profiling metadata, not an inference arithmetic change. Full acceptance and
all live testing remain pending.

All366 AppCPU,18/51 runtime and8+9 metadata controls now pass for sourceedbc7fb8.
The worker-only build was deliberately interrupted before GPU use for a final
numerical-oracle integration correction: the new reference launchers must use
original small/direct tails, matching the production path, and the existing
independent F64/full-owner oracle must recognize those genuine complete8 raw
fragments. Other oracle families and every tolerance remain unchanged. No GPU
or timing result exists for this new variant yet; full rebuild remains required.

## Complete validation and measured result (2026-10-23)

Frozen source7f7a127111c1f3790e382f7ebc304b68c4dd79a7; worker
b65ff90b60a55f926e4fe7253d0459ed414c483bdf726275c6a9a542b451015b.
366 AppCPU,18/51 runtime,8+9 metadata,10 original native controls and all
original compiler populations pass, plus16 genuine new shader/CFG layouts.
All792 GPU controls pass their numerical/full-owner/source/status/fault checks:
original768 retained plus24 genuine old/new expert cases, including real128-half
weight prefix, noncontiguous routes,383-member direct tails and poisoned values.
Operator cases inspect raw/pending publication; actual validating activation and
ordered finite/inverse closure are exercised by the original separate consumer
controls and both complete model chains, not by an invented oracle consumer.

Both original18-script model chains pass. Image production and matching image
matrix consumers are selected in all three requests; actual production quant
launches/compact bytes are0 while quantized numerical/compiler controls remain
compiled and executed independently. New global GU/DOWN prefix counts match the
actual original prefix population. Original rolling/reencoding/reuse/capacity,
finite-consumer and erasure checks pass. All8 original wide programs remain
zero-spill and both QSA ISA hashes remain unchanged.175 total ISA programs.

Diagnostic warmup/first/repeated3.737803327/11.797574348/9.740969385s;
normal3.774674971/11.823474772/9.665937322s. The short first warmup includes cold
shader work, so its reduction is not a measured warm small-prompt improvement.
The repeated complete prompt is essentially unchanged from preceding9.737173969s;
no parity or meaningful gain, no CLEAN run. Best earlier complete FP16 control
normal8.854793s and latest CLEAN8.394039699s remain faster. Strata clean repeated
reference5.3515s remains the goal; it is ACTIVE and UNMET.

Confirmed offset: fixed public layer24/4096 matrix cohort32.921752->13.364053ms,
but genuine dequant7.961387->19.242787ms (the expected real FP16 image cost).
New global wide GU11.019530ms versus original shared5.886924ms; DOWN4.876562ms
versus2.686351ms. Actual global GU128GRF has7:96 spills:fills/462 sends and
DOWN37:204/527 sends. These are substantial new register-pressure costs, not
proof that global operands are inherently slower. Whole repeated diagnostic
expert5.980507601s and attention2.481291469s; normal5.904979469/2.483092488s.
Per-layer timestamps are samples and must not be multiplied into a claimed
whole-prompt speedup. Keep this complete variant and original variants.

### Explicit audit limitation and preserved failure

Live unitcc6902ce3fb24d78a5a038f57d1d744b exits1 AFTER all original operator,
model and physical-recovery stages. Its outer audit retained one stale quant
expectation: sampled compact_copy must be nonzero. This image path correctly
has compact_copy=[0,0] and genuine dequant/matrix work. Original failed unit,
wrapper/log/freeze are preserved in development global-image-experts-compact-
profile-rejection-20261023; runtime13m50.580s, zero swap,31.0GiB peak. Never
report this live unit as exit0 or silently rerun hardware to hide the error.

Separate read-only auditbd8e5113255e4ef5abe4a436fee3aa94 exits0,228ms. It uses
the SAME completed immutable operator/profile/normal receipts and original
source/worker hashes; requires zero compact class, genuine profiled dequant,
real image/quant partition and all original guards. Its AST contains no
subprocess.run/Popen and cannot launch a worker/GPU run. Only completed-receipt
existence and the explicit class expectation change, with original49/new2
runtime-name and original363 CPU-prefix checks independently pinned. Both
original18-script chains stay preserved and checked. A separate CPU-only helper
check uses6 real prior public profiles and rejects24 corrupted counters.
The accepted792 proof explicitly records the live-unit failure and no hardware
reexecution for the metadata correction.

Public evidence under development:
- global-image-experts-root-{build-receipt,acceptance-freeze}-20261023.json
- strata-parity-global-image-experts-20261023-expert-cohorts-public-summary.json
- global-image-experts-{profile,normal}-safe-phases-20261023.json
- global-image-experts-complete-isa-export-20261023.json
- global-image-experts-completed-audit-{provenance-20261023.json,20261023.log}
- global-image-experts-final-state-20261023.json

Independent23:57:16UTC recovery: Intel50,688,000B,NVIDIA compute0/display593MiB,
all5 classified kernel faults0, HTTP200 Kuno/Ollama/Vashti, normal admission/
completed160. Installed binaries/settings unchanged. Canonical Vashti and
inherited pycache preserved. No installation/release/reset/driver change.
Next complete proposal: [N64 register footprint](kuno-prefill-global-n64.md).
