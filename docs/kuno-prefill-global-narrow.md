# Coupled global-load small-expert matrices

Optional backend-only successor to record32, not installed. Existing complete
paths remain. No API/KNT2/KNR2, model/configuration or Vashti change.

## Why this path

The retained public long fixture routes12221 groups with1..7 members, nearly46%
of26673 groups. Existing narrow8/16 products stage A and B through20/24KiB shared
memory, then synchronize at eachK64 handoff. The matched Strata SYCL reference
dequantizes full FP16 images and calls oneMKL (`prefill.cpp3964–3983`,
`gemm.dp.cpp773–805`). Its internal GEMM is opaque: no claim that the proposed
kernel is its exact implementation or that fused quantized GEMM is selected.

The complete `strata-narrow-global` / `--narrow-global` bundle connects:

- Actual Gather initializes inactive8/16-row GU input padding.
- Actual Activate and ActivateCheckGu initialize inactive8/16-row DOWN input
  padding too. Otherwise old gathered X bits remain there.
- Real routed small-expert CMMA loads complete A/B fragments directly from
  their original global vector owners. Scalar matrix strides remain K, typed
  view offsets divide by4. The original chronological K16 DPAS accumulation and
  output publication/route/fault/status handling remain unchanged.

Padding uses original grids and REAL map/count; no fabricated rows/routes/status
or logical member count. Each extra word is written once, strictly past real
rows, before consumption through original whole-view hazards. Check the genuine
output extent before padding. A host guard additionally requires exact fixed
expert feature geometry, complete padded A and full original B owner extent.
Private images and declared owner/source/erasure/capacity ledgers stay complete.
No new allocation or narrower scheduler declaration. Small groups only (<32);
dense products and wide-prefix tails retain their original staged programs.
Conservative old shared-memory device admission remains, though selected operand
SLM is removed. Only2/4KiB original output publication SLM and its fence remain.

## Validation and selection

Keep every original generator, numerical reference, source/owner/status/fault/
padding and recovery check. Retain staged small-expert definitions separately,
and add four selected global definitions across all8 original layouts. Existing
staged alias/lifetime proof stays on its original modules; a new real compiler
control checks global modules' exact output SLM, one publication barrier and
original perK16 matrix-load/product inventory plus genuine address endpoints.

New public producer checks execute both ORIGINAL and selected GPU programs for
counts1,3,7,8,9,15,16,17,31 in both phases. Compare real output bits exactly;
require every padded word zero, complete remaining owner tails/status/fault/
source unchanged, authentic map and real original grid, reset and erasure.
New34 routed-product checks reuse the original independent F64/IEEE oracle and
full-owner scans at those counts plus finite/nonfinite/invalid first/last route
cases. Their padded A fixture is explicit; they do not invent a request/batch.
All original189 GPU controls remain separate and unchanged. Whole-model actual
global-group counters, both73/4245/4245 schedules and original compiler/FMA/
source/owner/erasure/recovery chains precede any speed claim. Retain original
terminal/clean5%-at-EACH/staged quality/privacy/API gates before installation.

Complete source passes171 selected CPU checks (original168 retained +two actual
padding/grid tests +selected four-module/all8-layout compiler/address control).
Initial compile-only three errors were test enum/error-conversion bookkeeping;
failed logs retained, no hardware was invoked. New feature coherently includes
native-wide diagnostic dependencies, and the builder excludes CubeK alternatives
for this explicit native candidate. No GPU/performance acceptance yet.
Legacy `expert_native_capture_cpu`/`expert_native_wait_cpu` fields are WALL times
from Instant, despite their names. Only explicit CLOCK_THREAD_CPUTIME_ID fields
measure thread CPU. Source comment/docs are corrected, field names preserved.


## Frozen protected checkpoint

Sourcee6d04fbcf9aea760e039139d3d989d3c6bcdf4f2, artifact
`strata-parity-narrow-global-20261022`, worker SHA256
`0cba555ef108ae6943562673537bdb23d005fc59570107edd961aa7df21c9ac0`.
Exact committed-source171 CPU PASS again; frozen421 WGSL/248 matrix layouts PASS
with original pinned2026.1 validator. Original216 layouts retained +32 new global
layouts. Runtime tree byte-unchanged from accepted record32 (empty diff receipt),
so original11 runtime controls remain exact-source valid. New compiler controls
prove selected global modules' exact2/4KiB SLM, one publication barrier, two/three
loads and one/two DPAS operations per chronological K16 loop. Original staged
lifetime/alias controls continue on all original staged modules separately.

Protected root unit`kuno-narrow-global-20261022.service` runs original189 GPU
controls plus18 actual old/new producer and34 original-oracle routed controls,
then BOTH complete73/4245/4245 mode/compiler/count/owner/source/erasure/recovery
chains. Same public fixture/seed/model/context and original numerical thresholds.
Wrapper`check-native-narrow-global-connected-20261022.py` SHA256
`d0ef4b45d93ddb1e5969eacd4887c766a6c79794075d3eb250e2f4e10ce67b90`
pins source/artifact/CPU features/tests/offline inventory; actual global groups
must equal all real1..7 +8..15 +16..31 histogram groups in each request. This
proof is for the explicit native bundle excluding CubeK; no alternate-feature
fallback path credit. No fake runtime request/batch or scheduler owner is added.

Evidence base `/media/jmicko/faststore/kuno-development`:
`native-narrow-global-{cpu-selection,cpu-features,cpu-exact-final,cpu-exact-build,offline,release-build,connected}-20261022*`,
`native-narrow-global-runtime-unchanged-20261022.diff`. Earlier compile-only
errors retained as`native-narrow-global-cpu-build-attempt1-20261022*`.
GPU/performance/terminal/clean/staged/install pending; installed binaries unchanged.


## Complete connected outcome

All241 GPU controls PASS (original189 +18 actual old/new producer +34 routed
small-expert). Both original73/4245/4245 schedules pass every compiler/FMA/group/
source/owner/count/pool/erasure/recovery proof. Actual long global groups18954
exactly equal12221 +3883 +2850 real histogram groups. All old routed groups and
wide-prefix/Direct-GU counts remain unchanged; no arithmetic scope correction.

Normal first/repeat10.653114762/9.164950382s versus record32
10.998012681/9.499076228s (~3.14%/3.52% improvement). Combined since copy cohorts'
11.146964481/9.741252502s is4.43%/5.92% faster in these same host-profile modes.
Sampled matrix work30.887260->25.242100ms (~18.28% shorter) in the instrumented
390-group layer; sampled dequant16.207658ms and copies29.046850ms. These samples
serialize work, so they cannot establish normal compute/copy overlap or idle.
Normal warm experts4.942176667s, attention2.953429977s; actual capture HOST WALL
0.571665955s overlaps rolling GPU work and is not CPU utilization.

Exact full proof `strata-parity-narrow-global-20261022-narrow-global-public-summary.json`;
`native-narrow-global-connected-20261022.log`. Whole connected unit finished and
physical recovery passed. Keep both complete components. No Strata parity,
clean selection, staged quality/API, install or sustained decode gain yet.

Original terminal/fresh45/4245/4245 clean selection is now running under
`kuno-narrow-global-acceptance-20261022.service`; wrapper SHA256
`0facb71267bb545798c6bcfbef0290cd34ec0727564be1fb9ab585b4d0374ea9`
pins this same source/artifact/full proof and preserves existing5%-at-EACH gate
against installed3fffc266. It starts original staged full tools/thinking/MAX8/
semantic/privacy checks ONLY if both clean positions pass. Gateway/installed
worker remain untouched during this staged test. No benchmark chosen by seed
or result, and no new ordinary Ollama baseline is being run.


## Clean selection and normal-schedule first-use diagnostic

The original terminal-negative/fresh-recovery chain passed. Clean45/4245/4245
finished first12.320944936/repeat8.699633273s against installed
12.962279132/11.147390805s. Repeat is21.96% shorter; first is4.948% shorter,
missing the existing5% threshold by6.779761ms. The selection receipt FAILS;
no staged quality/API run or installation was started. Keep the complete change,
without changing the gate or choosing a favorable rerun.

Evidence base remains `/media/jmicko/faststore/kuno-development`:
`strata-parity-narrow-global-acceptance-20261022-clean-selection-public-summary.json`,
`native-narrow-global-acceptance-20261022.log`. Wrapper SHA256
0facb71267bb545798c6bcfbef0290cd34ec0727564be1fb9ab585b4d0374ea9.

A separate clean-mode, fixed-public-only flat CPU+Xe PMU diagnostic keeps the
normal queue schedule and adds no GPU timestamp command or serialization.
Root-only instruction-pointer samples and perf counters lived only in/run tmpfs
and were removed; no stack/register/payload samples. Counters enabled at the
second request's original rebind_begin and disabled at its last_prompt_input.
Three real replies45/4245/4245, control interval, zero lost samples, CLOSED and
physical recovery all pass. It is NOT a clean speed/selection result. The first
helper attempt failed a receipt assertion after completing/recovering because
clean mode intentionally has no profile capacity row; that failure is retained.
The corrected helper requires the row to be absent, not a fabricated proof.

Accepted proof `strata-parity-clean-first-costs-fixed-20261022-cpu-proof.json`,
flat report with >=0.3% symbols only (39.10% of samples, not a full aggregate)
`strata-parity-clean-first-costs-fixed-20261022-cpu-symbol-report.txt`, and
`native-clean-first-costs-fixed-20261022.log`. Helper SHA256
9905fdeb405785c8555ae642a360b72e8b3ac2539a2b536a528abd1f1b990d09.

Actual Xe engine active/total ticks over12.273s: RCS51.4135%, BCS16.1394%,
CCS0%. At least32.45% of the interval has neither counted GPU engine active;
unknown overlap means this is a lower bound. These are engine scheduled-active
counters, NOT EU/FLOP/memory utilization. PMU forcewake may alter behavior;
never treat this instrumented interval as a clean benchmark. Current native
MAIN is physical family0 RCS. A complete compute-only primary candidate must
also correct HAL buffer/texture shader stage masks and preserve actual copy/
draft families, concurrent owners, logical lanes and recovery; see
[kuno-prefill-compute-main.md](kuno-prefill-compute-main.md).
