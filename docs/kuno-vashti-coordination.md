# Kuno and Vashti Coordination

**Active Kuno handoff (2026-10-23): native half bundle accepted; larger fused panels next.**

Frozen source `cbd8a5b98351ca057dba7d7ca5465628ae4ea975`, worker
`df784f1ead35bd92c2172a714c29dabb82cd4635d3db813f4317734a7a411207`,
passes all 720 GPU controls and both full-model diagnostic chains, including
physical recovery. The protected unit exited 0 after 12m39.334s with zero swap.
All 350 AppCPU, 18/47 runtime, 8+9 metadata and compiler controls remain accepted.
Diagnostic first/repeated 4245-token times are 12.090085038/10.071408282s;
host-profile times are 12.132022756/9.947339019s. These are not clean timings.
Compared with the previous software conversion bundle, sampled cohort matrix
time fell 87.376758 to 45.719463ms. Two of eighteen new quant programs still
spill. Original eight wide programs remain spill-free and QSA ISA is unchanged.

Actual pipeline clocks attribute 2.017392017s to first-long-request preparation
and driver creation; repeated long requests create no pipelines. The repeated
4096-token chunk takes 7.606289589s and the 148-token tail plus teacher/cleanup
takes 2.406616230s. Larger fused panels with direct input-matrix loads are the
next coherent source change: remove redundant per-subgroup input staging and
reduce the number of weight-panel handoffs while preserving every K16 product
and original compact-source/publication contracts. No new result is claimed.

Evidence: `small-quant-slim-complete-isa-export-20261023.json`,
`small-quant-slim-{profile,normal}-safe-phases-20261023.json`, and the original
`strata-parity-small-quant-slim-20261023-expert-cohorts-public-summary.json`.
Independent 17:45UTC state: all services HTTP200, admission normal, Intel
50,688,000B, NVIDIA compute0, all five driver fault counts0. Installed binaries
unchanged. Goal ACTIVE/UNMET; latest clean remains 12.471177197/8.394039699s
versus Strata 5.5006/5.3515s. No API/Vashti changes or deployment authorization.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): native half conversion bundle compiled; protected GPU validation running.**

Frozen source `cbd8a5b98351ca057dba7d7ca5465628ae4ea975`, worker
`df784f1ead35bd92c2172a714c29dabb82cd4635d3db813f4317734a7a411207`
composes stage reduction7b0985d8, public pipeline timing7459dadc, native
packingceabf664, edge compilerf732a7c7, edge proof4d896931, integration1434e7f8
and strict offline entry identitiescbd8a5b9. All350 AppCPU,18 feature-off/47
full runtime,8 original+9 separate metadata controls pass. Every old compiler
population passes, plus144 actual native PackHalf2x16/UnpackHalf2x16 layouts,
8 genuine cooperative edge layouts and48 original IQ4 runtime-selector proofs.
Read [the complete contract](kuno-prefill-small-quantized-matrix.md) and
[public pipeline timing](kuno-prefill-native-pipeline-timing.md).

`kuno-small-quant-slim-20261023.service`, invocation
`8e1aca91666e4a8e8eaf8ff6981ee893`, is running the protected planned720 GPU
controls and both forward full-model chains, with swap/core dumps disabled.
This is NOT GPU acceptance or a speed result. The extra one-case proof covers
all65536 half encodings,190458 signed midpoint/adjacent-F32 inputs and exact
all-eight finite behavior, plus an actual cooperative matrix witness. Original
719 controls remain unchanged. Root owns GPU execution; no concurrent builds.

The forward timing harness now retains existing prompt_prefix/prompt_chunk
markers without changing engine work or adding fences. For4245 prompt tokens,
the batched prefix is4096+148; the last single-token decode is separate. Second
chunk wall time includes teacher work and final source/loan cleanup. Native
pipeline timing separately attributes first-request creation cost. Both modes
remain diagnostic; original clean45/4245/4245 comparison is required afterward.

Evidence under development: `small-quant-slim-root-build-receipt-20261023.json`,
`small-quant-slim-root-acceptance-freeze-20261023.json`,
`small-quant-slim-harness-20261023/`, `small-quant-slim-connected-20261023.log`.
Attempt1 CPU failure preserved in `small-quant-slim-attempt1-cpu-20261023/`:
manual definitions used a default name; assigning their genuine generated names
repairs selection without relaxing the native pass. Attempt2 aggregate failure
preserved in `small-quant-slim-attempt2-harness-20261023/`: all350/18/47 tests
passed, but a wrapper incorrectly expected feature-off-only negative cases in
the full-feature set. Exact original18 and47 populations were independently
verified, then unchanged source resumed at metadata/build/offline; no passing
CPU tests were rerun or weakened. Before GPU: all servicesHTTP200, admission
normal, Intel50,688,000B, NVIDIA compute0, all5 driver fault counts0.

Goal ACTIVE/UNMET. Latest CLEAN remains12.471177197/8.394039699s versus
Strata5.5006/5.3515s; no speed claim for the pending candidate. No API or Vashti
change; installed binaries unchanged. No installation/release/reset/reboot.

## Previous checkpoint


**Active Kuno handoff (2026-10-23): fused quantized cohort correctness accepted; conversion overhead isolated.**

Frozen source4b567eef / worker4abf583d65ca8855ecec5eaebe1a1a0447554298d870059404bc3d736ae3b5bd
passes343 AppCPU,18 feature-off/47 full runtime,8 original metadata and9 separate
feature-enabled selector controls. All original compiler populations,144 fused
layouts,6 compact modules and48 actual IQ4 runtime-selector checks pass. All719
GPU controls (677 original+30 compact+12 exact matrix), both forward18-script
chains, routing and physical recovery pass. Original QSA ISA hashes and all8
zero-spill wide programs remain unchanged. Six of18 new quant programs spill.
Read [the fused contract and evidence](kuno-prefill-small-quantized-matrix.md).

The initial IQ4XS mismatch was real: final NIR omitted the runtime upper-nibble
shift from a nested CubeCL if-expression. Explicit runtime select repairs it;
actual compiled SPIR-V now requires the selector in all48 IQ4 layouts. No
mathematical tolerance, owner, status or fault guard changed. Failed attempt4
and all prior evidence remain preserved.

Diagnostic profile first/repeated4245-token times14.682549580/12.609879799s;
host profile14.676952542/12.532337098s. These are not clean comparison results
and are slower. Matched request3/layer24/count4096 isolates cohort matrix
12.709929→87.376758ms, while dequant19.224670→8.000772ms and new compact copy
3.648422ms. Macro12.882906→12.868744ms and COPY29.172595→29.109284ms are nearly
unchanged. Keep the coherent bundle and fix conversion expansion; do not infer
an IO/scheduler regression or revert coupled components from this profile.
The previous CLEAN12.471177197/8.394039699s versus Strata5.5006/5.3515s remains
the latest clean result. Goal ACTIVE/UNMET; no new clean run claimed.

Protected unit `kuno-small-quant-fix-20261023` exited0 in13m8.852s, swap0.
Independent11:04UTC check: Intel50,688,000B, NVIDIA compute0, all5 driver fault
counts0, HTTP200 and admission normal, completed99. Evidence under development:
`small-quant-root-build-receipt-20261023.json`,
`small-quant-fix-root-acceptance-freeze-20261023.json`,
`strata-parity-small-quant-fix-20261023-expert-cohorts-public-summary.json`,
`small-quant-{profile,normal}-safe-phases-20261023.json`,
`small-quant-complete-isa-export-20261023.json`,
`small-quant-fix-final-state-20261023.json`.

Next complete source work retains the same compact/source/scheduler design:
7b0985d8 reduces repeated decoder expansion with an equivalent single stage
schedule and loop-free scale normalization;7459dadc adds public-only pipeline
creation timing to distinguish startup from prompt cost. These newer commits
are NOT built or GPU accepted yet. A source task is preparing exact native
PackHalf2x16/UnpackHalf2x16 lowering matching the original WGSL decoder, retaining
software oracles, all8 finite gates and final explicit rounding/denormal rules.
Do not replace this with arbitrary unrounded FConvert. New forward timing
harness is separate; the accepted4b files remain frozen. Root owns integration,
builds and GPUs. No API/Vashti changes; installed binary unchanged. No install,
release, reset or reboot authorized.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): complete compressed small-expert bundle compiled; IQ4 numerical failure under repair.**

Read [the complete fused-small contract and current checkpoint](kuno-prefill-small-quantized-matrix.md).
Frozen source888d8b20 / workera3192db31630a59cfff5713ff1bf303efcd4feec082f2d5840c23fb6d0922df3
passes343 selected AppCPU,18 feature-off/47 full runtime,8 original metadata and
9 separate feature-enabled selector controls. All original compiler populations
plus144 new quantized matrix layouts and6 compact-copy modules pass. This is a
complete compact-source/matrix/scheduler/runtime composition, with original
activation and ordered-output consumers. Actual new shared allocations fit48KiB.

Protected service `kuno-small-quant-20261023` exited1 during the new public
matrix controls, before full-model execution. First failure: dtype23/IQ4XS,
finite case, exact output mismatch; dtype18/21/22 finite and poison cases had
completed. Do not count the planned719 population as accepted or report speed.
No numerical tolerance was relaxed; preserve the complete producer/consumer
bundle while diagnosing. Frozen worker and all build/harness receipts are
preserved under development `small-quant-attempt4-gpu-numeric-20261023/`;
protected operator evidence remains `strata-parity-small-quant-20261023-operators/`.
The run lasted4m58.821s, swap0. Its physical recovery check passed; independent
10:36UTC live check confirms Intel50,688,000B, NVIDIA compute0, all5 driver fault
counts0, Kuno/Ollama/Vashti HTTP200 and admission normal (completed96).
See `small-quant-failed-final-state-20261023.json`. The original harness remains
unchanged; forward counter adaptations and acceptance freezes are retained.

The latest accepted CLEAN baseline remains small-direct12.471177197/8.394039699s
versus Strata5.5006/5.3515s. Goal ACTIVE/UNMET. Do not install or claim parity.
A separate source note identifies lazy pipeline creation as a first-request
investigation, not a measured attribution or repeat-speed gain:
`small-quant-first-prompt-pipelines-20261023.md` under development. That diagnostic
request used73-token warmup, not clean45, and its pipeline counts include decode.
No public API change; required Vashti changes:none. Installed binary unchanged.
No install/release/reset/reboot authorized by this handoff.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): complete small-expert publication accepted; fused small quantized matrices next.**

Frozen source4cc3f68f / worker19d3606e258037e027563b4f299bfdff5d9cc717d0af54a8e24acfb52cbce7c0
passes334 AppCPU,18 feature-off/41 full runtime,8 original metadata, all original
compiler populations plus176 layouts, all677 GPU controls and BOTH unchanged
18-script full-model chains. Routing[7663,26673,26673], actual selection and
physical recovery pass.21 deep-only64KiB controls remain unavailable on48KiB
hardware. Read [complete small-publication evidence](kuno-prefill-small-direct-publication.md).
No numerical or ownership guard was weakened. Protected run exited0 in11m46.908s,
swap0; Intel50,688,000B, NVIDIA compute0, kernel faults0, HTTP200/admission normal.
All8 wide and all new small ISA programs have zero spills; old two unrelated
spilling hashes and original QSA ISA remain unchanged.

Diagnostic profile first/repeated11.004590487/8.946788843s; host-profile
10.967607646/8.854793404s, versus preceding10.134471518/8.745485045s. This is not a
speed win. Keep the coherent bundle and address the next real difference.
The original clean45/4245/4245 comparison passes: first/repeated long
12.471177197/8.394039699s versus Strata5.5006/5.3515s; goal UNMET at both.
Receipt: `strata-parity-small-direct-20261023-fixed-clean-public-summary.json`. Existing host-profile mode retains coarse
Timer fences and must not be called clean. No installed binary changed.

Read [the complete fused-small contract](kuno-prefill-small-quantized-matrix.md).
Next coherent design targets groups<32: they account for186.3GB of262.2GB
FP16 image production in the repeated diagnostic request but only6.78% of
expert members. The active Strata reference also decodes full images; a fused
small quantized-CMMA producer is a new Kuno optimization, not an unported Strata
cache. Preserve exact format decoding, explicit F16 rounding, chronological
K16 math, original finite/status consumers and all source/lifetime checks.
The intended complete change includes packed source preparation, matrix
producers, scheduler and runtime admission together. No partial-component
performance selection. Root owns integration/builds/GPUs; isolated tasks are
source-only. Backend internal; required Vashti changes:none. Goal ACTIVE/UNMET.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): shared lifetime/direct8 accepted; complete small-expert publication next.**

Frozen source `c501ab882248fbe29b291fa9eb3b1055a853dc8a`, worker
`f9a3725b2eba522e209c9737228c632fc0178e37da3c77c19a8a9f66837af18b` passes
326 selected AppCPU,18 feature-off/34 full runtime and8 separate original
metadata controls. All original compiler populations pass, plus32 shared-reuse
layouts and30 direct8 modules; actual register CFG checks96=64 original+32 reuse.
All573 GPU controls pass (525 original semantic controls+48 reuse); the102
existing dequant cases execute actual Direct8/WG32, not old quad. Both unchanged
18-script full-model proof chains, routing[7663,26673,26673], actual selection
counters and physical recovery pass. The original21 deep-only64KiB cases remain
explicitly unavailable on the actual48KiB hardware. No tolerance was relaxed.
Read [shared lifetime reuse](kuno-prefill-wide-shared-reuse.md) and
[direct8 decoder](kuno-prefill-dequant-direct8.md).

Diagnostic profile first/repeated long10.058809444/8.781795945s; normal
10.134471518/8.745485045s versus preceding10.369469785/8.992040901s. This modest
gain is not clean selection and is still above older8.355123182s and Strata
clean5.3515s. Keep the coherent bundle; do not install or declare parity.
Matched layer24 dequant23.947634->19.167420ms, matrix37.084350->36.353866ms,
wide GU6.497654->5.779060ms. Cohort10.839204ms and macro12.273061ms remain.
All8 actual wide programs have zero spills. The2 other spilling programs and
original QSA ISA hashes remain identical to the preceding complete run.
Actual Kuno expert backing24,414,233,600B/11,983 ranked pairs already exceeds
Strata reference11,608 slots/22,556MiB; the legacy192-slot/8GiB constructor is
not the active global budget. Upstream --stream-experts does not enable its
stream_all branch here: actual nonresident share52.77% exceeds its10% gate.
Normal-mode rolling submission is active; timestamp-mode zero rolling counters
are not production evidence. Existing normal host work has~1.16s total ceiling,
partly overlapped already; do not promise all of it is recoverable.

Evidence under development: `reuse-direct8-root-build-receipt-20261023.json`,
`strata-parity-reuse-direct8-20261023-expert-cohorts-public-summary.json`,
`reuse-direct8-{profile,normal}-safe-phases-20261023.json`,
`reuse-direct8-complete-isa-export-20261023.json` and
`reuse-direct8-final-state-20261023.json`. Preserved attempt1 build failure was
CubeCL relative macro path/private module access, fixed617a9442. Attempt2
passed new controls but2 legacy matrix-registration fixtures used empty DQ
arguments with selected Direct8 identity; c501ab88 correctly keeps those
legacy-only fixtures on original identities. New Direct8 runtime validation
was not weakened. FinalGPU service exited0 in9m56s, swap0, allfaults0;
Intel50,688,000B, NVIDIA compute0, HTTP200 and admission normal. Installed
301f1fd1/36626d30 unchanged. No install/release/reset/reboot or Vashti changes.

Next complete source bundle targets smaller expert publication: real direct
CMMA stores for full valid8-row fragments across macro32/macro64/narrow/cohort
and genuine wide tails, original checked fallback for partial/invalid routes,
mandatory existing raw-GU finite activation and ordered-DOWN pending2/inverse
consumer. Original arithmetic, owners, definitions and tests remain. Root owns
integration/build/GPU; three isolated source tasks basedc501ab88 split regular
producers, cohorts and runtime grammar. Do not benchmark partial components.
Goal remains ACTIVE and UNMET. Required Vashti changes:none.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): preserve ordered bundle; close the measured decoder gap.**

Source54162b57 / worker630c4898c8779b0a6d88d56646b1e2465b72cdd36dd2bfc695c645dae2fddbd2
passes317 selected AppCPU,18 feature-off/24 full runtime and8 exact metadata
controls; all original compiler populations plus120 ordered matrix and1 fused
consumer modules. Actual register CFG count64 is original48+HC8+ordered8.
All525 GPU controls pass (original469+40 paired ordered producers+16 consumers),
as do BOTH unchanged18-script full-model chains and physical recovery. Actual
48KiB hardware correctly records21 deep-only cases unavailable, never passed.
Original medium feature is disabled: the planned48 ordered producers was a
counting mistake; the exact active population is40. No tolerance was relaxed.

Profile first/repeated long10.386195819/9.098446276s; normal diagnostic
10.369469785/8.992040901s. This is slower than prior8.355123182s and still above
Strata clean5.3515s. Do not describe it as a clean selection or install it.
The actual matched layer24 timestamp sample isolates dequant14.336803→23.947634ms;
matrix37.400288→37.084350ms, wide DOWN2.897865→2.654066ms, copy29.176347→29.159880ms.
Keep the coupled changes and address the confirmed difference, not isolated
toggle rollback. All four actual wide programs have zero spills; the two other
spilling public programs are byte-identical to the earlier control. Original
QSA ISA hashes remain unchanged. Evidence in development: build receipt,
connected acceptance, safe phases and complete ISA exports under
`expert-ordered-slot-quad-c-*20261023`; frozen build prefix
`strata-parity-expert-ordered-slot-quad-c-20261023`. Initial fixture type/generic
source-parser failures are preserved. Compiler receipt was completed read-only
after correcting56→64 additive CFG accounting, without rerunning hardware.

Next coherent source: [K32 lifetime reuse](kuno-prefill-wide-shared-reuse.md)
composes genuine GU/direct/ordered-DOWN entries with actual32KiB explicit
allocation and original48KiB fallback, original tails/guards and separate actual
selection counters. Producerd4466c9e/runtimee4d57f2b are composed; unbuilt.
In parallel, the active pinned Strata Intel decoder uses direct eight-value
lane-local quant loads with no leader/broadcast. An additive complete decoder
is in progress, retaining all original scalar/record32/quad controls and exact
source authority. Root owns integration/builds/GPUs; no GPU testing while
compilation runs. The performance goal remains ACTIVE and UNMET.

No API/config/model/context/Vashti changes; required Vashti changes:none.
Installed301f1fd1/36626d30 is unchanged. No install/release/reset/reboot.

## Previous checkpoint

**Active Kuno handoff (2026-10-23): expert-ordered output and vector combine.**

Source327124a1 / workercd4142f97e91466b452896e64ee6a6ce93a6b911419b011bd390d35c18f231ed
passes304 selected AppCPU, original15+separate17 runtime and8 exact metadata
controls; all original compiler populations plus16 deep dense,16 deep macro,
24 deep expert and30 quad-dequant modules. Protected GPU acceptance stopped at
`deep dense actual selected matrix rejected`: actual Intel B70 Vulkan and3060
both advertise49152B shared memory, below the new kernels'65536B requirement.
The authentic gate rejected before any new deep matrix ran. This is NOT a
numerical error, GPU fault, or completed490-control/model/performance result.
Physical recovery passed: Intel50688000B, NVIDIA compute0, faults0, HTTP200 and
admission active/paused/recovery=false, waiting0. Evidence: development files
`deep-panel-slot-quad-root-build-receipt-20261023.json` and
`deep-failure-public-summary-20261023.log`. Preserve the failed receipt.

Keep64KiB variants guarded for capable devices; do not spoof hardware capacity
or claim they executed on this machine. Correct hardware-aware validation is
required. Previous accepted de309ebf remains469 GPU controls/BOTH original18-
script chains; diagnostic repeated8.355123182s versus Strata clean5.3515s,
which is still UNMET. Installed301f1fd1/36626d30 remains unchanged.

Next coherent source uses existing PART in expert order, authenticated route
inverse in the dead router RAW view, direct valid DOWN cooperative stores, and
vector4 inverse gather with exact pending finite/status closure and shared
merge. Read [the coupled layout contract](kuno-prefill-expert-ordered-output.md).
Source is now composed (c92e5a56 producers, fe640d10 consumer, 98dae7df/57b0110e
runtime grammar, d9d38d62 actual capability reporting, 38c267fb source-proof
formatting). Intended additions are13CPU,120 ordered matrix layouts,1 consumer
module,48 raw producer and16 consumer GPU cases. All original469 GPU controls
remain;21 deep-only cases are explicitly unavailable on this48KiB device.
No new build/GPU/performance acceptance yet. [Consumer details](kuno-prefill-expert-ordered-consumer.md).

The four new experimental features remain additive: deep matrices, per-slot
copy endpoints, quad32 dequant and expert-ordered output. No speed selection
before complete source/correctness/recovery acceptance. Root owns integration,
builds and GPUs; agents edit isolated worktrees based327124a1.

No API/model/context/config/frontend/install/release/reset/reboot changes;
required Vashti changes:none. Canonical Vashti checkout remains untouched.
Keep prompts/images/tokens/expert content out of files and receipts.

## Earlier checkpoints

**Earlier Kuno handoff (2026-10-23): compose deep matrices, per-slot uploads and coalesced dequant.**

Completed source de309ebfb6db720c4ba5f9c278a205fb41f6701c / worker
0ccb105762abb62ed1b41d95ce75cf7193b689b184bcb5d5e41f4443b2c1746b passes
286 selected CPU controls,15 runtime controls,429 original WGSL modules,
320 original matrix layouts,16 raw,32 composite,24 padded,48 original+8 HC
register CFG checks,2 normalization and16 HC layouts; ALL469 GPU controls;
BOTH unchanged original18-script full-model chains; routing[7663,26673,26673]
and physical recovery. Coupled HC down+injection and eight-group raw lookahead
are retained. Read their [exact contract/evidence](kuno-prefill-hc-down-inject.md#complete-de309ebf-validation)
and [lookahead evidence](kuno-prefill-raw-lookahead.md#complete-de309ebf-validation).
Profile first/repeated long9.728968924/8.419709565s; normal9.797516084/8.355123182s
versus prior9.872980496/8.468479392s. These are diagnostic, not clean selection.
Actual new HC producer/finite ISA0spills/fills; original QSA hashes unchanged.
No clean or installation for this modest gain. Strata clean5.3515s is UNMET.
Two new outer metadata mistakes (48 vs56 CFG markers,4245 vs4244 batched tokens)
were corrected with read-only receipt completion, preserving failed originals,
ALL original checks and completed timings; no favorable hardware rerun.

Next coupled source is COMPOSED; initial Rust build passes, full acceptance pending: double reduction
stages across dense/medium/large expert matrices with authentic64KiB hardware
and actual explicit-layout gates; preserve original K16 arithmetic/owners/faults
and all original fallback/compiler populations. Per-expert actual copy endpoints
retain existing complete three-role commands and all source/slot/whole-bank
hazards. Coalesced record32 uses four authentic subgroup lanes and original
8-value stores. Root owns integration, builds and GPUs; isolated source worktrees
basedde309ebf only. No compilation during GPU timing. Keep all prior features; no performance testing before complete source acceptance.
Composed commits d0dbf59b/96925d4b/b8f7aca3/218b6f67/97924d00/e88a0370/
7c2dacf7/fd936fb4/7b497709/f45b4fbc. Wide-expert shared-memory lifetime
correction e5bfd4af/2db3b15c now passes all24 actual compiler layouts after
rejecting the initial80KiB layout; no new GPU run yet. Read [deep matrices](kuno-prefill-deep-matrix.md),
[slot overlap](kuno-prefill-raw-slot-overlap.md), and
[coalesced dequant](kuno-prefill-dequant-quad32.md) before changes.
Intended304 AppCPU, original15+separate17 runtime controls, repeated8exact
metadata controls, original compiler populations plus16dense/16macro/24expert/
30quad modules,ALL490 GPU controls and BOTH original18-script chains.
Original selector8 fragments remain byte-identical; new graph endpoint method
is outside those metadata tests and needs its own queue/GPU evidence. New flags --deep-matrix,
--raw-slot-overlap,--dequant-quad32 are explicit optional laboratory selectors.
No API/model/context/config/frontend/install/release/reset/reboot change;
required Vashti changes:none. Canonical Vashti checkout remains untouched.
Keep prompts/images/tokens/expert content out of files and receipts.


**Current Kuno continuation (2026-10-23): coupled wide register-stage source.**

Read [real register staging contract](kuno-prefill-wide-register-stage.md) and [complete prefix validation](kuno-prefill-dense-prefix.md#complete4439c220-validation).
Complete4439c220/workerc7cadabd passes270CPU/15runtime/all shader layouts/
ALL433GPU/BOTH original18-script chains and recovery. Normal9.8233/8.5113s
no gain; all compact/padded/finite/QSA ISA zero spills, QSA hashes unchanged.
No clean/staged/install. Goal UNMET vs Strata5.3515s.
New source keeps everything and couples actual next global vector loads into
registers,original current CMMA products,then disjoint next-SLM writes in BOTH
dense and routed-expert wide producers. Original final-block guard,math,stage
addresses,barriers,owners,capacity and erasure retained. Intended273CPU/
15runtime/all original shaders +48 actual compiler in-loop dataflow checks/
all433GPU/BOTH original chains. Source only; no build/GPU/speed claim.
No API/model/context/config/frontend/install/release/reset/reboot changes;
Vashti:none. Installed301f1fd1/36626d30 unchanged. Root owns builds/GPUs;
no compile during GPU timing. Earlier checkpoints history.

**Current Kuno continuation (2026-10-23): real M128 prefix + original tail source.**

Read [exact prefix contract](kuno-prefill-dense-prefix.md) and [completed padded validation](kuno-prefill-dense-slm40.md#complete821bfd6a-validation).
Complete821bfd6a/worker01188d21 passes267CPU/15runtime/all shaders/ALL417GPU/
BOTH original18-script chains and recovery. Normal9.5559/8.4267s neutral,
actual padded/compact/finite/QSA ISA zero spills. No clean/staged/install.
New connected source handles the genuine148-token chunk with128 wide +20
original fallback,retaining every numerical/capability/owner/erasure guard.
Original general minimum512 and original counter97 retained; explicit real
prefix minimum128,new counters,actual offset borrowed tail with no extra tokens
or allocations. Intended270CPU/15runtime/all original shader controls/433GPU/
BOTH chains. Source only; no build/GPU/speed claim. Goal UNMET vs Strata5.3515s.
No API/model/context/config/frontend/install/release/reset/reboot change;
Vashti:none. Installed301f1fd1/36626d30 unchanged. Root alone owns builds/GPUs;
no compilation during timing. Earlier checkpoints are history.

**Current Kuno continuation (2026-10-23): complete pitch40 dense source.**

Read [pitch40 source contract](kuno-prefill-dense-slm40.md) and [completed compact-direct validation](kuno-prefill-dense-direct-store.md#complete-connected-validation27451ef9).
Complete27451ef9/worker61a8d8f6 passes264CPU/15runtime/429WGSL/320matrix+16raw+
32composite/ALL405GPU/BOTH original18-script chains and physical recovery.
Normal9.6114/8.3706s neutral; new actual producer/finite ISA zero spills/fills,
QSA hashes unchanged. No clean/staged/install. Goal UNMET vs Strata5.3515s.
New source couples real pitch40 A/B staging,explicit pad zero and matching
cooperative readers,keeping compact-direct producer and finite consumer.
Intended267CPU/15runtime/all original shader controls+24 new layouts/417GPU/
BOTH original chains. Source only; no build/GPU/speed claim. No API/model/
context/config/frontend/install/release/reset/reboot change; Vashti:none.
Latest01:27:10UTC physical recovery: Intel50,688,000B,NVIDIA compute0,driver
faults0,HTTP200,admission normal. Installed301f1fd1/36626d30 unchanged. Root
alone owns builds/GPUs; no compile during timing. Earlier checkpoints history.

**Current Kuno continuation (2026-10-23): complete direct dense publication source.**

Read [direct dense source contract](kuno-prefill-dense-direct-store.md).
Preceding144be548/workere331b0fa passes261CPU/15runtime/429WGSL/320matrix+16raw/
ALL393GPU/BOTH unchanged original18-script chains and physical recovery. Tiny
component cuts measured HC~76ms; normal repeated8.352865318s is a small gain,
not Strata parity. New tiny native programs zero spills/fills; QSA hashes and
all original guards retained. No clean/staged/install on that small component.

New connected source keeps everything, couples direct cooperative dense output
with real GPU finite/zero/fault consumer, preserving original owners and math.
No publication scratch is allocated; actual32768B producer with original host
capability guard retained. Old shared writer stays for fallback. Intended264CPU/
15runtime/429WGSL/320matrix+16raw+32composite layouts/405GPU/BOTH full chains.
Source only; no build/GPU/performance/install claim. No API/model/context/config/
frontend/release/reset/reboot change; Vashti:none. Root alone owns builds/GPUs;
no compile during GPU timing. Installed301f1fd1/36626d30 unchanged, goal UNMET.
Refresher1668570 expires06:04:15UTC. Earlier checkpoints below are history.

**Current Kuno continuation (2026-10-23): coupled tiny-output pipeline source.**

Read [tiny-output source contract](kuno-prefill-tiny-output.md).
Full precedingfe5285ae/worker0e79dd6c passes257CPU/15runtime/429WGSL/304layouts/
ALL353GPU/BOTH original18-script chains and physical recovery. Normal repeat
8.450195111s is neutral; no clean/staged/install for that component. New outer
counter96 assumption omitted the genuine MTP teacher GR; actual97 reverified by
read-only completion, with no hardware/timing rerun or changed original guard.
Latest physical recovery00:10:35UTC: Intel50,688,000B,NVIDIA compute0,faults0,
HTTP200,admission normal. Installed301f1fd1/36626d30 unchanged.

Complete new source couples REAL raw-U32 padded weight producer with SAME
chronological original CMMA and logicalN4/N1 writer, on existing80MiB scratch.
Targets masked HC injection/shared gate loops, retains all prior optimizations.
Initial7f3a94cd/workerdbdc14c6 passes260CPU/15runtime/429WGSL/320matrix+16raw
layouts. Final current-model BF16 shared-gate selection adds10 GPU controls;
planned393GPU/BOTH chains. No GPU test on this new bundle yet.
Source only; no build, GPU, speed or installation claim. No API/model/context/
config/frontend/release/reset/reboot change; Vashti:none. Goal UNMET vs Strata
clean5.3515s. Root alone owns builds/GPUs; no compile during GPU timing.
Earlier checkpoints below are history. Refresher1668570 expires06:04:15UTC.

**Current Kuno continuation (2026-10-23 00:00UTC): GR wide-tail full validation.**

Read [the exact source, failures, correction and evidence](kuno-prefill-gr-wide-tail.md).
Full optional bundle retains all prior optimizations and adds bounded BF16
HC down10240x320 wide selection. Sourcefe5285ae/worker0e79dd6c passes final257
AppCPU/15runtime/429WGSL/304layouts and separate protected operators/recovery,
including3 new original-model F64/full-owner/NaN-Infinity cases with0 mismatches.
A NEW probe fault-padding expectation caused prior failure; shader unchanged,
original head flag plus zero padding now correctly checked. Earlier summary
truncated at18 log lines; both new finite cases actually passed before failure.

Unit kuno-gr-wide-tail-flags-full-20261022.service runs ALL353GPU and BOTH
complete original18-script chains, swap/core0; original operator stage passed,
normal/profile/full aggregate pending. Root alone owns hardware; no CPU compile.
Next one new-source clean comparison only after complete acceptance/recovery.
Previous static-geometry clean11.0164/7.9515s fails original5% EACH; no installation.
Installed301f1fd1/36626d30 unchanged. Goal UNMET versus Strata clean5.3515s.
No API/model/context/config/frontend/release/reset/reboot change; Vashti:none.
Earlier checkpoints below are history. Refresher1668570 expires06:04:15UTC.

**Current Kuno continuation (2026-10-22): fixed-role macro64 source.**

Read [the source/clean/CPU-profile handoff](kuno-prefill-macro64-static.md).
Complete frontiera761763e/worker89a27c5b passes253CPU/15runtime/429WGSL/296layouts/
ALL350GPU and BOTH original chains, zero spills and physical recovery.
Normal first/repeat9.654518545/8.354937695s. ONE clean gives
11.053525683/7.924977071s: repeat9.03% faster than installed8.712129955s,
but FIRST fails original5% selection against10.897223643s. No install and
no favorable clean rerun. Separate first-long CPU/PMU diagnostic is explicitly
NOT a clean selection and samples no content/stack/register/branch data.

Complete next source keeps all prior optimizations and specializes only the
authenticated64-row GU/DOWN stage strides/full weight geometry. Preserve
header/map/role checks, add surplus-X early close, keep masked input tails,
original dynamic K loop and chronological products/barriers/owners/faults.
Intended255CPU/15runtime/429WGSL/296layouts/350GPU/BOTH original chains and
actual compiler/recovery before claims. Source not built/tested yet. Installed
worker/gateway unchanged; no API/model/context/config/frontend/release/reset/
reboot change; Vashti:none. Whole-model Strata parity remains UNMET.

**PAUSED for user machine update (2026-10-22 21:32UTC).**

Read [the exact source-only checkpoint and completed run](kuno-prefill-qsa-query-planes.md#coupled-gdn-source-checkpoint-and-user-pause--2026-10-22).
Full ad3a8a53/worker488f34ed passed238 App CPU,15 runtime,429WGSL/296layouts,
all350 GPU and BOTH original18-script chains. Normal4245 first/repeat
10.411413116/9.141123116s (~3% shorter repeat than preceding9.422304546s).
Diagnostic only; clean Strata5.3515s parity remains UNMET, no installation.
Next complete query-plane + GDN local fault-mask source is saved but UNCOMPILED
and UNTESTED (intended245 App controls). Do not resume automatically or start
GPU work during the update. Actual fresh recovery21:31:40UTC: Intel50,688,000B,
NVIDIA compute0, driver faults0, services200, admission normal. No API/config/
model/context/frontend/release/reset/reboot change; required Vashti:none.
Installed accepted worker/gateway unchanged. Earlier snapshots below are history.


**Current source continuation (2026-10-22 21:18UTC): query-vector planes.**

Read [the exact source contract](kuno-prefill-qsa-query-planes.md). It retains
the full medium-expert/vertical-Quad bundle and changes only the physical
12KiB query SLM permutation:768 original vec4 stores,24 query read sites,
all3072 bits/guards/barriers/math/owners unchanged. Intended241CPU/runtime15/
429WGSL/296layouts/all350GPU/BOTH original chains +actual ISA/recovery.
Source only; no build/performance/install claim. Root ad3a8a53/worker488f34ed
passed238CPU/runtime15/offline429/296 and currently runs corrected350GPU
controls and both chains. Earlier c8050a65 operator run correctly stopped on
a missing public65-member fixture before dispatch; failed receipt preserved,
physical recovery verified. New actual fixture CPU control covers all six sizes.
No CPU compilation while GPU run active. No API/config/model/context/frontend/
release/reset/reboot change; Vashti:none. Installed301f1fd1 remains; goal UNMET.


**Active Kuno continuation (2026-10-22 21:00UTC): medium expert + vertical Quad bundle.**

Read [the complete candidate specification](kuno-prefill-expert-macro64.md).
Full scale-finite3285f49c passes228CPU/runtime13/offline429/280/all330GPU
and both original chains with recovery; normal10.600400443/9.422304546s is
neutral. Full nativeQuad b7262ce6/worker9b4f090f passes232CPU/runtime13/offline
429/280 and all330GPU math controls. Its sampled full prompt completes and
recovers, but native35spills/81fills fails the original compiler gate. Failed
chain preserved, no normal/clean/staged/install acceptance. Actual indirect
moves332->212; this is compiler evidence, not a clean speed claim.

Complete new source retains full Quad, adds explicit vertical-only mode and
real64x64 expert products including staging/publication/dispatch/INFO/grammar/
pricing/counters and additional independent controls. Intended237AppCPU,15real
runtimeCPU,429WGSL/296layouts,350GPU and BOTH original chains precede claims.
No original numerical/spill/owner/source/recovery bound relaxed. GPUs recovered;
next compile/check this complete bundle, no CPU compilation during GPU timing.
Installed accepted301f1fd1 remains. No API/model/context/config/frontend/
release/reset/reboot change; Vashti:none. Goal UNMET; user requests continued work.


**Active Kuno continuation (2026-10-22): native four-lane lowering.**

Read [the detailed source handoff](kuno-prefill-expert-gap.md#native-four-lane-lowering-continuation-2026-10-22-source-only).
Complete native-XOR175b1c2f passes225CPU/runtime13/offline429/280, all326GPU
and both original chains with recovery; repeat9.372082925s is neutral. The
scale-finite3285f49c worker646da501 passes228CPU/runtime13/offline429/280;
its protected330GPU/both-chain run is underway, operators already pass. Next
optional nativeQuad source is complete, not built/tested: actual hardware
COMPUTE/QUAD/all-stages properties are true. Mesa source identifies a dedicated
32-bit quad path absent from relative shuffles. Preserve all original guards;
232CPU/330GPU/both chains/actual ISA/recovery remain before performance claims.
No CPU compilation while full GPU chains run. No new installation; Vashti:none.
Goal remains UNMET; user authorizes computer/GPUs for the next few hours.


**Current Kuno work (2026-10-22): complete expert-register + native-XOR continuation.**

User authorizes the computer/GPUs for the next few hours and requests continued
work until matching Strata. Goal remains UNMET (controller usageLimited).
Read [the source comparison and active handoff](kuno-prefill-expert-gap.md).
Expert-register8b7490a5/SHA0d821b0b passes222CPU/runtime13/offline429/280,
all326 GPU and both original INNER chains with actual recovery. Normal
first10.591999060/repeat9.341873441s is neutral. OUTER aggregate fails a new
profiling-only selected-family assumption: real directGU uses a fifth generated
type and was still other. Preserve failure; no clean/staged/install selection.
Do not treat a collected failed unit's default ExecMainStatus0 as proof of success.

Full next bundle keeps expert-register and attention prefetch, adds constant
XOR reduction lowering (a022e933) and real direct-GU classification. Strata's
actual constant butterflies use direct register permutations; Kuno indexed
spelling has332 indirect moves. Source equivalence alone does not prove speed.
No CPU compilation while the previous GPU run was active; it has now recovered.
Next root225 CPU/offline/full326 +both chains +actual ISA precede selection.
No API/KNT2/KNR2/config/model/context/frontend/reset/reboot/release change;
Vashti:none. Installed native-prepared301f1fd1 remains. Earlier checkpoints
below are historical. Root alone owns CPU target and hardware.

**Current Kuno checkpoint (2026-10-22 17:30UTC): attention prefetch fully tested; no speed gain.**

Full coupled QSA prefetch sourcec5a17af5 (merged ae24a145)/workerSHA78e7a60e
passes220 selected App CPU controls (all original213 +new7),13 actual runtime
controls,429 WGSL/280 matrix layouts,all326 GPU controls and BOTH complete
original18-script proof/profile/recovery chains. Read
[the tested attention handoff](kuno-prefill-qsa-prefetch.md#root-validation-checkpoint--2026-10-22)
for exact frozen artifacts, guards, compiler evidence and limits. Public summary
under artifact base SHA554375e5db922cb6c2f85ca0ef10f2508337ba5f3e3d42ec26618dba60bb11e1.

Actual SIMD32/128GRF code retains K/Q/PV lookahead and has zero spills/fills,
but2982 instructions/130 sends versus2654/112 before. Normal first4245-token
prompt10.501310285/repeat9.380901735s versus bank/release10.424577118/
9.283137399s:0.74%/1.05% slower; attention bucket essentially unchanged.
Profile10.594575759/9.449852848s also shows no gain. These are instrumented
diagnostics, not clean selection. Keep the COMPLETE optional implementation;
no clean selection, staged quality or installation. Same-host Strata5.3515s
still leads; goal UNMET/controller usageLimited. Do not falsely mark it achieved.

Protected unit completed exit0/inactive/dead. Final actual recovery17:30:44UTC:
unchanged boot,Intel50,688,000B,NVIDIA compute0,driver faults0,services200,
admission normal. No test worker/compiler remains. Installed accepted301f1fd1/
SHA0e226827 worker and36626d30/SHAd3ed07f4 gateway reverified unchanged.
Bounded sudo refresherPID1456723 expires2026-10-23 01:17:20UTC; do not assume
credentials persist beyond it. No API/KNT2/KNR2/model/context/config/frontend/
release/reset/reboot change; required Vashti:none. Earlier checkpoints below
are historical, including the former unmerged/source-only attention status.

**Current Kuno checkpoint (2026-10-22): complete dual-bank/release validation PASS; no new speed selection.**

Goal UNMET; controller currently usageLimited after the overnight usage limit.
Complete raw-bank/unit-release source3c90b3ee/worker7b07349b passes213 App CPU,
13 actual runtime controls (all original11 retained),429 WGSL/280 matrix layouts,
all326 GPU controls and BOTH full original18-script proof/profile/recovery
chains. Actual2 separately attested32MiB owners and genuine terminal releases
match every original execution unit. Protected unit completed exit0/inactive;
physical GPU recovery passed. Public proof under artifact base:
`strata-parity-expert-raw-banks-release-20261022-expert-cohorts-public-summary.json`,
SHA256 0af7699f616d586a612510bcc7ebc0b57980c15205cd3ffa63ba3d42d9b680e1.
Full details in [raw banks](kuno-prefill-raw-banks.md).

Normal first10.424577118/repeat9.283137399s versus previous complete cohort
11.156398257/10.239178827s:6.55%/9.34% shorter. Installed native-prepared
normal10.116598360/9.222042861s still leads (new repeat essentially tied).
Strata5.3515s remains faster. Keep complete bundle; no original clean selection,
staged quality or installation of this successor. Previously accepted301f1fd1/
0e226827 native-prepared worker and36626d30/d3ed07f4 gateway remain installed.

Next optional QSA software pipeline is COMPLETE SOURCE ONLY on isolated
`codex/qsa-prefetch`, commitae24a145fb86b5775d360be113e83b7ba771f8df,
parent2cf3295a; not merged, compiled or GPU-tested. It combines guarded K/Q
lookahead and four-cell V/probability prefetch; both independent static reviews
find no blocker. Current exact native ISA proves serial sends/consumption;
future native assembly must show real earlier sends without spills. Retain all
original numerical/owner/status/fault gates; feature-specific static opcode
counts change because of unrolling, not arithmetic relaxation. No API/KNT2/
KNR2/config/model/frontend/release/reset/reboot change; required Vashti:none.
Bounded sudo refresher expired15:45:12UTC; do not assume credentials remain.
User requested a cost/status checkpoint before changing thinking mode; no new
GPU test started. Earlier active snapshots below are historical.

**Active Kuno checkpoint (2026-10-22): genuine dual raw-bank bundle under construction.**

Read [the complete raw-bank design](kuno-prefill-raw-banks.md). Actual full-ring
DQ reads cause head-of-line upload waits even with the corrected queue cut.
Preserve those authentic full-root dependencies; add a separately priced actual
second32MiB raw allocation, scope/source attestation and complete alternating
execution-unit prefetch scheduler TOGETHER. <=32K owned only, old larger loan
unchanged. Three isolated agent worktrees based4b48791e; root alone merges,
compiles and uses hardware. No new speed/GPU/install claim. Existing326/both
chains fully pass at42efd634 as recorded below; native-prepared remains installed.
No API/KNT2/KNR2/config/model/frontend/release/reset/reboot change; Vashti:none.
Goal ACTIVE/UNMET; sudo refresher1145657 expires15:45:12UTC.

**Active Kuno checkpoint (2026-10-22 12:07UTC): complete cohort successor correct; scheduling attribution continues.**

Goal ACTIVE/UNMET. Frozen42efd634/workerSHA6ba68800 passes198 CPU,
originalruntime11,429 WGSL/280 matrix layouts, all326 GPU operators and BOTH
complete original18-script proof/profile/recovery chains. Exact public summary
`strata-parity-expert-cohorts-overlap-20261022-expert-cohorts-public-summary.json`
under artifact base `/media/jmicko/faststore/kuno-development`. Unit
`kuno-expert-cohorts-overlap-20261022.service` completed/inactive/dead, exit0;
actual memory recovery passed. Normal first11.156398257/repeat10.239178827s
versus complete prior cohort11.429276201/10.419853034s:2.39%/1.73% shorter,
but installed native-prepared normal10.116598360/9.222042861s remains faster.
Keep the complete coupled implementation; no clean-selection/stage/install run
on this successor. Sampled expert kernels and actual CPU capture improve;
normal expert wait wall grows, so investigate transfer/compute scheduling and
real rolling submission overlap rather than discard the bundle. Native-prepared
301f1fd1/SHA0e226827 remains installed with unchanged36626d30 gateway.
No API/KNT2/KNR2/config/model/frontend/release/reset/reboot change; Vashti:none.
Root alone owns hardware. Bounded sudo refresher1145657 expires15:45:12UTC.
Earlier active snapshots below are historical.

**Active Kuno checkpoint (2026-10-22 11:53UTC): complete cohort overlap + scalar bundle GPU validation.**

Goal ACTIVE/UNMET. Native-prepared301f1fd1/SHA0e226827 remains installed with
unchanged gateway36626d30/SHAd3ed07f4 and accepted clean first10.8972/repeat8.7121s.
Complete [cohorts](kuno-prefill-expert-cohorts.md)/[sealed selection](kuno-prefill-cohort-selection.md)
a5/c01 passes all326 GPU/both full original chains+actualselection+recovery;
normal11.4293/10.4199 slower than installed10.1166/9.2220 despite fasterGPU/CPU
capture. Warm flat CPU+PMU complete/recovered/raw/run erased:6.646B cycles versus
8.45B prior, RCS64.24%/BCS19.70%/CCS0 scheduled-active (notEUutilization). No
clean/staged/install run. Keep complete path. Source found lost overlap: early
raw refill made planner merge matrix(A) with DQ(B), importing the nextCOPYwait.
Root7ae81bb8 moves unchanged consume/refill AFTERallmatrix enqueues, using
original real queue cut and DOWN-image release; no ownership/hazard relaxation.
Root42efd634 adds same-kernel bulk192-byte scalar registration (48 exact-growing
allocations become1), same6 arrays/240-byte INFO/erasure; two real-byte/panic
controls. Independent generated-macro/source/ownership audits find no blocker.
Both changes are composed BEFORE timing. Frozen42efd634/workerSHA6ba68800 PASS
198 CPU +originalruntime11 +429 WGSL/280 matrix layouts; separate8 metadata
source hashes unchanged. Protected unit`kuno-expert-cohorts-overlap-20261022.service`
now runs ALL326 GPU and BOTH original complete proof/profile/recovery chains.
Helper`check-expert-cohorts-overlap-connected-20261022.py` SHA45440f41 under
artifact base, log`expert-cohorts-overlap-connected-20261022.log`. No new GPU/
speed/install success claimed yet. Root alone owns hardware; no CPU compile.
No API/KNT2/KNR2/config/model/frontend/release/reset/reboot change; Vashti:none.
Bounded sudo refresher1145657 expires15:45:12UTC; never kill shared shell3556.
Earlier active snapshots are historical; actual original matrix arena is512MiB.

**Active installed Kuno checkpoint (2026-10-22): native-prepared selected.**

Goal ACTIVE/UNMET versus Strata5.3515s. Full staged quality/privacy/API passes,
clean first10.8972/repeat8.7121s is15.93%/21.85% shorter than installed native32.
Native worker301f1fd1/SHA0e226827 is now installed via official transaction;
unchanged gateway36626d30/SHAd3ed07f4. Full config AND complete model settings/
enrolled identities equality verified; official rollback
`/var/lib/kuno-strata-parity-native-prepared-20261022`. Exact artifacts/gates/
failed-harness preservation in [native-prepared spec](kuno-prefill-native-prepared.md).
Actual installed-service semantic/privacy checks PASS, including two complete
identical501-token answers and155 semantic cases each, DynamicUser/model mount,
swap/core0/write_bytes0/inspection denial/OOM-fault0/services/admission. Warm
35-token prefix1.1746s does not replace long clean comparison or claim a new
decode gain. Fixed-public warm flat CPU+PMU diagnostic currently runs in
`kuno-native-prepared-warm-costs-20261022.service`; raw samples confined to/run
and erased. Exact helper/hash and pending status in linked spec. Root owns GPUs;
no simultaneous compilation.

Warm diagnostic now complete/recovered: source-pinned full public summary,
raw/run removal, unchanged boot, Intel50,688,000B/NVIDIA compute0, services200,
all faults0/admission normal. RCS72.45%/BCS22.71%/CCS0% are scheduled-active,
not EU utilization. Full flat CPU shows planner/descriptor/recording overhead;
exact debug symbols resolve all115 driver rows. See linked spec; no new clean
speed claim. CPU target released EXCLUSIVELY to the cohort port agent, with
explicit handoff to selector agent for builds; root runs no GPU concurrently.

Small-expert cohorts are being implemented completely in separate
`codex/expert-cohorts`, agreedbase31f6e54b, with independent structural/math
audits and no hardware activity. Retain the full prior paths and unchanged
numerical/owner/capacity/erasure controls. Coupled authentic cohort selection
for original rolling/poolreuse/sampling is delegated separately at the same base
to`codex/cohort-selection`; agree the sealed API before merging. No public
contract change or fake logical-as-physical dispatch counts. No API/KNT2/KNR2/frontend/model/release/
reset/reboot change, required Vashti:none. Earlier unchanged-install snapshots
below are historical. Owned refresher1145657 remains bounded to15:45:12UTC.

**Active Kuno checkpoint (2026-10-22): native-prepared complete; clean selection running.**

Goal ACTIVE/UNMET; user enabled Ultra and both GPUs/root remain authorized.
Use ONLY isolated`codex/strata-rust-port`. [Native-prepared](kuno-prefill-native-prepared.md)
301f1fd1/SHA0e226827 passes183 CPU/11 unchanged-runtime/421 WGSL/248 matrix,
all241 original GPU plus11 new actual-device admission controls and BOTH full
original proof/profile/recovery chains. Actual prepared/native Graph counts
261/271/271, ordinary inference counts0 in each mode. Normal first10.116598360s
versus global-narrow10.653114762s; repeat9.222042861s versus9.164950382s is
essentially unchanged. No Strata parity or sustained decode gain claim.

Original terminal/clean selection and conditional staged quality/API now run in
`kuno-native-prepared-acceptance-20261022.service`; exact helper/hash in linked
spec. Existing5% at EACH first/repeat gate retained. Installed3fffc266/gateway
36626d30 unchanged; no installation yet. No API/KNT2/KNR2/config/model/frontend/
release/reset/reboot change; required Vashti:none. Root alone owns hardware;
read-only agents investigate the remaining expert/attention gap. Prior snapshots
below are historical. Refresher1145657 expires15:45:12UTC; preserve shell3556.

Update: clean first10.897223643/repeat8.712129955s PASSES original5%-at-EACH
against installed12.962279132/11.147390805s (15.93%/21.85% shorter). Original
stage startup failed on root-only test directory/config under strict UMask0077,
BEFORE inference; failed receipt/recovery retained. Fixed private ownership
helper retains all original quality checks, no world-readable config. Full stage
now runs`kuno-native-prepared-stage-fixed-20261022.service`; exact helper/hash in
native-prepared spec. No installation yet. Root alone owns GPU/services. A new
isolated`codex/expert-cohorts` worktree at agreed31f6e54b implements the COMPLETE
optional small-group cohort producer/matrix/grammar bundle; no concurrent builds
during measurements. This is a new Vulkan-specific optimization, not a claim
that matched Strata already batches its FP16 experts. All original paths remain.

**Active Kuno checkpoint (2026-10-22 09:03UTC): first-use CPU/queue work.**

Goal ACTIVE/UNMET, user enabled Ultra; both GPUs/root testing authorized. Use
ONLY isolated Kuno worktree `codex/strata-rust-port`. Installed worker/gateway
identities below remain unchanged. Complete global-narrow clean first12.320945s
missed original5% gate by6.78ms; repeat8.699633s is21.96% shorter. Retain FAIL
receipt; no staged API/quality/install run. Full241 GPU and both profile proof
chains already pass for frozene6d04fbc/SHA0cba555e. New normal-schedule cold flat
CPU+PMU diagnostic passes recovery: RCS51.4135%, BCS16.1394%, CCS0%. These engine
active fractions are NOT EU utilization; raw tmpfs samples removed. See exact
receipts/limits in [global narrow](kuno-prefill-global-narrow.md).

[Complete compute-primary candidate](kuno-prefill-compute-main.md)69c045a7 /
SHA306d4618 passes179 CPU/11 runtime/421 WGSL/248 matrix, all241 GPU and BOTH
original proof/profile chains with recovery. Normal first11.9705/repeat10.5335s
is12.37%/14.93% slower than render-primary global-narrow. Keep complete optional
path; do not select for install. Additional source-pinned cold flat CPU+PMU
PASSED: actual RCS0%/CCS56.02%/BCS14.93%, raw tmpfs removed, full flat
CPU report and physical recovery. Unit completed/inactive/dead; no clean-speed
claim. Next concrete first-use target, duplicate ordinary/native
pipeline executable creation, is being implemented in separate managed worktree
`kuno-native-prepared-pipeline` at agreedbase69c045a7. No concurrent builds during
root measurements; root alone owns GPUs. Preserve ALL original core+HAL
validation, actual device/context provenance and native owners when skipping
the unused ordinary driver executable. Render-primary remains next selection.
The [complete native-prepared successor](kuno-prefill-native-prepared.md)
301f1fd1/SHA0e226827 is now frozen on render-primary:183 exact-source CPU,
421 WGSL/248 matrix PASS; original runtime tree unchanged. Root unit
`kuno-native-prepared-20261022.service` runs original241 GPU plus11 actual-device
admission checks and BOTH full original chains with actual no-ordinary-program
counts. Frozen18-script harness preserves original math/owner/native compiler
checks; selected compiler population is intentionally1 per scope instead of2.
Exact manifest/helper hashes in linked spec. No GPU/speed/install claim yet.
No API/KNT2/KNR2/config/model/
frontend/release/reset/reboot change; required Vashti changes:none. Bounded sudo
refresher1145657/shared shell3556 expires15:45:12UTC. Earlier handoffs follow.


**Paused Kuno-only goal: match Strata (2026-10-21; native32 installed).**
See [the exact accepted backend and remaining work](kuno-strata-parity-goal.md#accepted-native32-installation).
Selected worker source3fffc266 / SHAeac78ce0 is installed with unchanged
image-enabled gateway36626d30 / SHAd3ed07f4. Official backup/rollback:
`/var/lib/kuno-strata-parity-native-wide32-20261020` (previousccdaf886).
Full configuration and original root-enrolled identities are preserved.

Matched repeated4245-token prompt12.4373 ->11.1474s (10.37% shorter); firstlong
14.1743 ->12.9623s. The fresh same-host Strata reference5.3515s is still faster,
so this goal is not complete. Native32 removes actual matrix register spills;
original103 GPU controls,402/144 offline layouts, terminal error/fresh recovery,
full tools/MAX8/semantic/privacy tests and actual installed service checks pass.
Warm installed35-token prefix1.0908s,501 outputs/15.0813s; no decode gain claim.
Worker swap/core0, write_bytes0, denied inspection, nonroot/model mounts, zero
OOM/driver faults, healthy services and normal admission confirmed.

**No API, KNT2/KNR2, configuration, model identity or Vashti change.**
No frontend, fresh browser/image QA, package, version bump or release is claimed.
Root alone owns the shared GPUs/services. The original paired/bulk source and
public CPU profiles are retained in the parity guide; no private content
instrumentation or relaxed memory/hazard guards are added.

**Active Kuno goal (2026-10-22): complete small-expert global-load candidate.**

User authorized overnight work, both GPUs and refreshed sudo, then enabled Ultra.
Controller now ACTIVE/UNMET. Same isolated checkout/branch; installed worker/
gateway unchanged. [Shared-record decoder](kuno-prefill-dequant-record32.md)
af4de6f0/SHA96b0c3b0 passes168 CPU/11 runtime/421 WGSL/216 matrices, original189
GPU and BOTH full original proof/recovery chains. Normal first10.9980/repeat9.4991s
versus copy11.1470/9.7413s, modest1.34%/2.49%; no new clean/staged/install claim.

Read [complete global-narrow bundle](kuno-prefill-global-narrow.md) before next
change. Gather AND activation zero original tile padding, real small-expert
matrix products load A/B directly from authentic global owners. Same K16 math,
original grids/count/routes/publication/status/fault/source/cap/erasure guards;
dense and wide-tail staged modules retained. Independent source audit confirms
addresses/strides and both producer coupling; GPU acceptance still pending.
Selected native-only bundle excludes CubeK and coherently requires native-wide.
Sourcee6d04fbc/SHA0cba555e frozen,171 CPU/11 unchanged-runtime/421 WGSL/248
matrix layouts PASS. Root`kuno-narrow-global-20261022.service` runs original189
GPU +18 old/new actual producer +34 original-oracle routed controls and BOTH
complete profile/count/compiler/source/owner/erasure/recovery chains. Exact
wrapper/artifact/logs in linked spec. All241 GPU controls and BOTH full original
proof chains now PASS with actual18954 global groups/long request; normal
first10.653114762/repeat9.164950382s versus record32 10.9980/9.4991s (~3.5% warm
gain), sampled matrix18% faster. Strata5.3515s still leads. Root original
terminal+fresh clean selection now runs`kuno-narrow-global-acceptance-20261022`;
staged full quality/privacy/API follows ONLY if existing5%-at-EACH gate passes.
No install yet. Complete producer+matrix wiring retained, no numerical relaxation.

Two historical *_cpu diagnostic fields actually use Instant wall elapsed time;
source/docs corrected without renaming fields. Actual thread-CPU evidence is
CLOCK_THREAD_CPUTIME_ID in host/boundary collectors. Never label capture/wait
wall as CPU utilization or sampled serialized kernel sums as normal idle.

Owned sudo refresher1145657/shared shell3556 bounded until15:45:12UTC; stop
refresher on close, never shared shell. Initial17.67GB idle image residency was
evicted through original native unload path; physical preflight Intel50,688,000B,
NVIDIA compute0, services healthy, reported driver faults0. Previous protected
unit complete/recovered. No API/KNT2/KNR2/frontend/model/config/release/reset/
reboot change. Required Vashti changes:none. Preserve canonical coordination and
inherited untracked cache. Historical stopped checkpoint follows.

**Authoritative stopped handoff (2026-10-22 00:31UTC): user requested close; GPUs and services ready.**

Goal remains PAUSED/UNMET. No more live work should start without user resuming.
Use ONLY isolated `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Canonical coordination work and inherited cache preserved.
Installed worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 freshly
rehashed unchanged. No new installation, API/KNT2/KNR2/model/config/frontend/
release/reset/reboot change; required Vashti changes: none.

Best gain is [copy cohorts](kuno-prefill-copy-cohorts.md)661ddad3/SHA09f39612:
all original proofs pass; clean first12.791548881/repeat9.244574900s versus installed
12.962279132/11.147390805s. Repeat is17.07% faster, first only1.32%; original5%
required at EACH position still fails. No staged/install acceptance. Same-host
Strata repeated4245-token prompt5.3515s still leads. Keep the complete gain.

Later complete optional components remain, with no cherry-picked speed claim:
[packed cache](kuno-prefill-packed-cache.md)fe579785/SHAeaeb67cd passes193 GPU,
normal first11.3106/repeat9.7327s, neutral; [medium experts](kuno-prefill-native-medium-experts.md)
6720c6d2/SHA34b73687 passes233 GPU, normal11.6871/10.0194s, modest slowdown.
[Complete default Strata attention](kuno-prefill-reference-attention.md)
180b91bd/SHAe3455e4c passes175 App/13 unchanged-runtime/421 WGSL/216 layouts,
233 GPU controls and BOTH full original compiler/count/source/owner/erasure/
recovery chains. Normal first11.404259100/repeat9.965325679s; no additional gain.
Original arithmetic/status/poison/F64 bounds and owning memory guards retained.
Its explicit96-MAD single-cell compiler inventory replaces180 only in this mode;
paired modes still require180, subgroup32/no spills unchanged. Original helper
snapshot preserved; selected helper SHA31d3cf54 pinned. Failed intermediate
fixture/source-inventory controls and refused source-changing build retained.
No new terminal/clean/staged/install acceptance for these later experiments.

Evidence base `/media/jmicko/faststore/kuno-development`:
`strata-parity-reference-fixed-20261022-reference-attn-public-summary.json`,
`native-reference-fixed-connected-20261022.log`; other complete artifact names,
source hashes and tests are in linked specifications. Latest whole-model run is
complete; all owned test units inactive/dead, no owned build/worker remains.
Bounded sudo refresher895579 expired naturally; shared shell3556 preserved.
Fresh `native-prefill-final-state-20261022.json` at00:31:05UTC proves Intel
50,688,000B, NVIDIA compute0 (655MiB desktop), reported driver fault counters0,
all three servicesHTTP200; admission inactive/unpaused/recoveryfalse/waiting0.
Boot unchanged. No GPU reset or reboot attempted.

Next resume: profile normal-schedule expert staging/compute and queue gaps before
more attention variants. Rare groups1..7 are12221/26673 groups in the retained
public packed profile; streamed host-stage bytes27,307,776,000 per long prompt.
This makes rare-expert execution/overlap a candidate, not a proven attribution.
Sampled serialized GPU timings are not normal-queue utilization. No new profiling
monitor was started; no further experiment or installation is pending live.

**Previous stopped checkpoint:**

**Authoritative paused handoff (2026-10-21 01:34UTC): complete unused-descriptor candidate correct, speed neutral; system ready for user.**

User requested a breaking point and end of day. Goal PAUSE requested/UNMET;
no further work should automatically start. Use ONLY isolated Kuno worktree
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Read the [complete exact-source outcome](kuno-native-unused-descriptors.md#complete-outcome-and-user-requested-pause-2026-10-21-0134utc).

Sourcee454c594 / SHAeae4c199 passes154 CPU,11 runtime,421/216 offline,
original189 GPU controls and BOTH complete73/4245/4245 profiles/proof chains.
Actual unused-view counts7663/28996/28996; original FMA populations preserved.
Full original declared admission/owners/recording footprint remains separate
from proven instruction-unused memory accesses. No arithmetic/guard/cap relaxed.
Normal first/repeat11.782711649/10.189230061s versus predecessor
11.728765729/10.125904949s: effectively neutral. Strata5.3515s still leads.
No new terminal/clean/staged/install acceptance; installed worker3fffc266/
SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 freshly rehashed unchanged.
Keep complete correct components; do not cherry-pick a favorable timing.

No build/test/worker remains; protected unit inactive/dead. Actual01:34:11UTC:
Intel50,688,000B, NVIDIA compute0(748MiB desktop), reported driver faults0,
all3 servicesHTTP200; admission inactive/unpaused/recoveryfalse/waiting0.
Boot unchanged. Owned sudo refresh375934 STOPPED; shared shell3556 preserved.
No reset/reboot/service/config change. Canonical uncommitted coordination work
and inherited untracked cache preserved. No API/KNT2/KNR2/model/frontend/release
change; required Vashti changes: none.

Next run should locate larger actual backend enqueue/recording or selected QSA
costs from the retained matched Strata receipt. Current host-wall/queue marks
are not isolated thread-CPU/GPU attribution; tiny duplicate memory visitation
cannot explain the gap. This is a future investigation, not pending live work.
Full artifact names, precise timings, limits and acceptance gaps are in the
linked spec. Resume only after the user requests it.

**Previous checkpoint (superseded):**

**Authoritative active handoff (2026-10-21 00:48UTC): direct GU fully correct, speed neutral; system recovered.**

Goal ACTIVE/UNMET. Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
branch`codex/strata-rust-port`. No build/test/GPU unit remains. Installed worker
3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 freshly rehashed unchanged.
No clean/staged/new-terminal/install run occurred for this arm. No API/KNT2/KNR2/
config/model/Vashti/frontend/release change. Required Vashti changes: none.

Read[the complete direct GU outcome and next hypothesis](kuno-native-direct-gu-publication.md#complete-accepted-numericalprofile-outcome-2026-10-21-0048utc).
Source`bb6c65342751a33b968e6c7b18112934b52b441f` (runtime2babfde3+public-launch
correctionbb6c6534), SHA`59e76f48ffed1f957c760ef248c6d0d439391ce6c046fd389619cf27bb0e9640`,
build`strata-parity-direct-gu-public-launch-20261021`, passes147 selected CPU,
11 runtime grammar/capture,421 WGSL/216 matrix layouts, ORIGINAL171 GPU plus18
paired exact-bit cases, BOTH complete original73/4245/4245 modes and every original
compiler/count/pool/scratch/source/owner/erasure/recovery proof. The first setup
failure2babfde3/SHA6fe832e2 is preserved, not numerical acceptance.

Production counts: short0 direct/0 wide/7663 checkers; EACH long2344 direct/4688
wide/26673 checkers. Original FMA groups7663/26673/26673 remain. Actual direct
stores retain invalid-route checked fallback and49152B; all finite/signed-zero/
hf_sat/status/fault/view/epoch/lifetime/erasure/capacity contracts are preserved.
No fabricated prompt/batch authority; genuine closed collector enforces pairing.
Actual GU read/write hazards include checker writes. All old components remain.

Profile first/repeat11.675534147/10.268863056s; normal11.728765729/10.125904949s.
Same subgroup predecessor normal11.645375742/10.189380043s: effectively neutral.
Normal expert5.902606639s(collection1.492020708s). Do not install or claim a gain
from one favorable repeat. Installed clean12.962279132/11.147390805s and same-host
Strata5.3515s still govern selection/goal. Keep the correct complete component.

NEXT production change is NOT yet implemented. A new CPU-only actual-module
diagnostic now PROVES both checked/direct GU declare status binding4 ReadWrite
but never reference its variable ID in any function; DOWN has64 references.
The real compiled/transformed words are inspected through pinned disassembly,
not guessed from names. Test`actual_gu_status_is_declared_mutable_but_has_no_instruction_reference`
passes; log`native-gu-status-declaration-diagnostic-cpu-20261021.log` (one new
cfg(test)-only control after the147 selected suite; no production source change).
Read the[exact next proof/admission requirements](kuno-native-direct-gu-publication.md#actual-descriptor-diagnostic-completed-2026-10-21-0051utc).
Retained SPIR-V currently derives access from declared Visibility, whereas WGSL
uses actual GlobalUse. Implement a bounded fail-closed actual-module proof and
retain full resource/source/mutation/replay/owner admission independently of the
memory-access mask. Do not simply narrow a scheduler flag or complete view.
No production dependency has been narrowed and no overlap/speed gain is claimed.
Actual false dependencies and any downstream genuine hazard must be measured.

Public artifacts under existing base: `native-direct-gu-public-launch-*20261021*`,
`strata-parity-direct-gu-public-launch-20261021-{direct-gu-public-summary,outcome-public}.json`
and exact source-pinned connected helper. Final independent00:47:37UTC state:
Intel50,688,000B, NVIDIA compute0(720MiB desktop), faults0, all3 HTTP200, admission
inactive/unpaused/recoveryfalse/waiting0, completed103; boot unchanged. Root state
receipt`native-direct-gu-final-state-20261021.json`. Owned cached sudo375934 /
sharedshell3556 remains bounded to02:05:20UTC; never kill sharedshell or reset GPUs.
Canonical checkout's uncommitted coordination work and inherited cache preserved.

**Previous checkpoint (superseded):**

**Authoritative active handoff (2026-10-21 00:37UTC): direct GU GPU diagnostic setup corrected; rebuild pending.**

The complete candidate below remains, with all147 CPU controls passing again
in`native-direct-gu-public-launch-complete-cpu-20261021.log`. First frozen
2babfde3/SHA6fe832e2 passed421 WGSL/216 matrix layouts, then its NEW standalone
pair harness correctly failed because it used the production activation launcher
without a parent prompt scope. No speed run occurred. Root's actual operator
receipt confirms physical recovery and admission resumed; no reset/install.
Preserve`native-direct-gu-connected-20261021.log` and
`native-direct-gu-public-failure-20261021.json` with the failure.

The public-audit diagnostic now checks exact logical owner sizes/grid and launches
the original activation directly. It does not fabricate STAGES/prompt/batch
scope. Production dispatch and closed collector are unchanged. Rebuild a new
frozen artifact, repeat original171+new18 numerical controls, then BOTH whole
profiles/actual pair counts/recovery before considering speed. Original failure
is setup evidence, not numerical acceptance or kernel failure. Same worktree,
goal, installed identities, sudo owner, limits and no frontend/API/release scope
as the detailed checkpoint below.

**Previous detailed implementation checkpoint:**

**Authoritative active handoff (2026-10-21 00:31UTC): coupled direct GU candidate CPU-complete; GPU acceptance pending.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Goal ACTIVE/UNMET; installed worker3fffc266/SHAeac78ce0
and gateway36626d30/SHAd3ed07f4 unchanged. Read the complete
[direct GU paired contract and checkpoint](kuno-native-direct-gu-publication.md).
New opt-in direct global GU + mandatory validating activation + exact closed
collector checks are implemented together, with147 actual selected CPU controls
and11 actual runtime grammar/capture controls passing. Original145 controls and
all math/status/hf_sat/source/owner/epoch/retirement/erasure/capacity checks remain.

A source review caught invalid-route overwrite: a subgroup-uniform eight-route
vote selects direct publication for valid fragments, otherwise unchanged checked
publication preserves valid rows AND untouched invalid GU bytes. Shared49152B
therefore remains; no reduced-memory/occupancy claim. Actual8-layout compilation
proves4 global GU stores,4 original scratch fallback stores,8 products and original
fences. Actual checker use inventory is GU READ|WRITE, so original graph hazards
include the new writes. Direct production selection requires a genuine open
collector; closed/escaped clones cannot fall back to ordinary immediate execution.

Release/offline/GPU/full-model and speed gates are PENDING. New18 genuine paired
public numerical cases cover256/257, gate/up NaN/+Inf/-Inf and first/last invalid
routes, complete owners/padding, and actual erasure. They synthesize no production
authority. Original171 GPU cases remain required. BOTH full profile modes must
prove actual checker count=groups and direct-prefix count*2=wide-prefix count,
original FMA populations and all original pool/count/source/recovery checks.
Only a promising full result proceeds to original terminal, fresh clean-at-EACH5%,
staged tools/thinking/MAX8/semantic/privacy and reversible installation gates.

Current safe physical state00:31UTC: Intel50,688,000B; NVIDIA compute0; all kernel
faults0; admission inactive/unpaused/recoveryfalse/waiting0. Owned bounded sudo
375934/sharedshell3556 ends02:05:20UTC; keep it cached and never kill sharedshell.
No GPU reset/reboot. No API/KNT2/KNR2/config/model/Vashti/frontend/release changes.
Exact CPU artifacts `native-direct-gu-*20261021*` under existing artifact base.
Canonical checkout's coordination work and inherited untracked cache preserved.

**Previous authoritative handoff (superseded):**

**Authoritative active handoff (2026-10-21 00:12UTC): subgroup publication accepted, speed unchanged; next coupled GU proposal.**

Goal ACTIVE/UNMET. Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
branch`codex/strata-rust-port`. No test/build/GPU unit remains. Installed worker
3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 freshly rehashed unchanged.
No clean/staged/install or new terminal-negative run was performed for this arm.
Do not install it on the strength of profile timings alone.

Frozen complete source3228cd9e7596b6c0ba6dd3935a44f834b2dba691 /
SHA0e45b16ae74b01e7ff2d5c9e3aa19d1f4d11451fccffaa967577da1ba2f16deb,
build`strata-parity-subgroup-publish-20261020`, passes145 selected CPU,
420 WGSL/208 matrix layouts, original129+42 GPU numerical controls, BOTH complete
original73/4245/4245 profiles, exact original FMA group counts7663/26673/26673,
compiler/actual pool/scratch/source/owner/erasure/recovery proofs. Scalar QSA
matches the reference selection; the complete optional XMX code remains intact.
SG publication retains all original math and finite/output/status/fault guards;
input barriers remain workgroup scoped and actual ordinary+native compiler paths
supply required AcquireRelease|WorkgroupMemory to subgroup publication barriers.

Full profile first/repeat11.718102135/10.321304380s;
normal11.645375742/10.189380043s. SAME scalar/pool predecessor865a9c16:
profile11.632181168/10.289150836s; normal11.608595581/10.128802681s.
These differences show NO measurable speed gain. Keep this correct component;
do not claim that switching off optional XMX is a publication improvement.
Current normal expert interval5.945952864s(enqueue1.525212104s), QSA attention/
merge1.507134167s. Sampled layer24 GPU class intervals are not isolated kernel
costs and must not be added as causal whole-request totals. Installed clean
12.962279132/11.147390805s and same-host Strata5.3515s still govern selection/goal.

Exact evidence under`/media/jmicko/faststore/kuno-development`:
- `native-subgroup-publish-complete-cpu-20261020.log` and selection JSON;
- `strata-parity-subgroup-publish-offline-20261020.log`;
- `strata-parity-subgroup-publish-20261020-subgroup-publish-public-summary.json`;
- `strata-parity-subgroup-publish-20261020-outcome-public.json` exports fixed-public
  aggregate metrics/profiles only; actual root-only receipts retain protection
  and full before/after recovery evidence;
- `check-native-subgroup-publish-connected-20261020.py` pins exact artifact,
  CPU/release feature equality, original wrapper chain and both final proofs;
- `check-native-subgroup-publish-acceptance-20261020.py` is prepared UNEXECUTED,
  preserves original terminal/FRESH clean-at-EACH5%/staged quality/API/privacy gates.
No repeated unchanged test or fresh baseline is needed.

NEXT: implement the full paired direct GU store + validating activation proposal
in[docs/kuno-native-direct-gu-publication.md](kuno-native-direct-gu-publication.md).
It is NOT implemented yet. GU is contiguous, so an actual cooperative global
store could remove its SLM publication round trip. It MUST be paired with the
original per-value finite sanitization moved into the consuming activation,
including positive-zero stores/fault128 and unchanged activation hf_sat/status
behavior. Keep immediate checked primitives for standalone callers; bind a new
role-bearing direct GU only to a genuine closed original batch requiring its
checker before any GPU submission. No unchecked shortcut or fabricated authority.
The collector, shader write inventory, exact GU read/write view, bounds/capacities,
public genuine pair numerical negatives and actual production path counts must
change together. Preserve the existing SG component for DOWN/dense publication,
all old controls and complete optional XMX source. No speed/occupancy claim from
nominal scratch subtraction. Read that complete contract before editing.

Independent00:11:35UTC: Intel50,688,000B; NVIDIA compute0(738MiB desktop), all
kernel faults0, Kuno/Ollama/Vashti200, admission inactive/unpaused/recoveryfalse/
waiting0; no lab worker remains. Bootunchanged. State receipt:
`strata-native-subgroup-publish-final-state-20261021.json`.
Current bounded cached sudo375934 / sharedshell3556 starts00:05:20UTC and ends
02:05:20UTC Oct21; first refresh succeeded and process verified live. Owner/log:
`sudo-refresh-native-publication-20261021-{owner.json,log}`. Old owned297061 was
verified and stopped ONLY after replacement succeeded; old PID now absent.
Never kill sharedshell3556 or reset/reboot GPUs. No API/KNT2/KNR2/configuration/
model/frontend/release change. Worktree has only inherited`__pycache__` untracked.

**Previous checkpoint (2026-10-20 23:52UTC): coherent native subgroup publication candidate,145 CPU pass.**

Use ONLY the existing Kuno worktree/branch below. Previous full matrix bundle
remains intact and optional; its clean failure is preserved. Installed worker/
gateway unchanged. Goal ACTIVE/UNMET; no GPU job remains at this checkpoint.

New`strata-native-subgroup-publish`/builder`--subgroup-publish` requires BOTH
native wide32 dense and expert kernels. Only per-SG result-publication execution
barriers become subgroup scoped. Actual scratch ranges[sg*128,(sg+1)*128) are
disjoint, producer/consumers all in that exact16-lane subgroup. Original input
stages retain full-workgroup barriers; all FMA/K order, source/buffer/epoch/fault/
finite/status/scatter checks, allocations and retirement remain unchanged.

Plain CubeCL SyncPlane emits RELAXED memory semantics, so BOTH actual ordinary
HAL and retained-native compiler paths now strengthen those recognized subgroup
barriers to AcquireRelease|WorkgroupMemory, retaining subgroup scope. Existing
genuine literal264 is reused; no new IDs, no other word changed, no ownership
completion inferred. Unsupported scope/semantics/missing literal fail closed.
All145 actual selected CPU checks pass, including real dense FP16/BF16 and
routed-expert modules across ALL8 layouts, exact narrow-diff/restoration,
unchanged2 input barriers, expected8 publication fences, idempotence and memory
scope/semantics/missing-constant negatives, and disjoint scratch ownership.
Original tails are verified untouched by the transform. Test selection excludes
only4 XMX-feature-only cases; uses matching release feature set.

Release/offline/original171 GPU/full both-mode/clean/staged/install acceptance
remain pending. Use the original complete scalar-QSA/pool wrapper chain and keep
historical FMA group counts/numerical/source/owner guards; this variant changes
no arithmetic. No API/configuration/model/frontend/release change. Exact memory
contract:[subgroup publication](kuno-native-compute-intervals.md#connected-subgroup-local-native-publication-proposal).
Artifacts: `native-subgroup-publish-{cpu-build,complete-cpu}-20261020.log`,
`native-subgroup-publish-cpu-selection-20261020.json` under the existing artifact
base. Owned cached sudo297061/sharedshell3556 ends00:17:49UTC Oct21. Do not kill
the shared shell or reset/reboot the GPUs. Fresh accepted safe state below remains
the last runtime check; only CPU compilation has run since it.

**Previous handoff (full matrix validation and preserved speed failure):**

**Previous handoff (2026-10-20 23:36UTC): full-chunk math accepted; clean selection FAILS, system recovered.**

Goal ACTIVE/UNMET. Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
branch`codex/strata-rust-port`. Preserve the other checkout's coordination files.
No GPU test remains; both protected units are terminal. No installation occurred.
Installed worker3fffc266/SHAeac78ce0 and image-enabled gateway36626d30/SHAd3ed07f4
remain byte-identical, freshly rehashed. Never reset/reboot the GPUs.

Frozen complete sourcee7c43cb553c3ea7adc8eeeef63b6c5b557083cb7 /
SHA88ce000345a0fc1bc69af379cf51b8efcfbe5199642e7024cea83c1bd450d170,
build`strata-parity-qsa-xmx-full-chunk-aligned-20261020`, retains every preceding
component plus persistent query, explicit RTE, full64-cell group staging, real
vec4 stores and register P.V accumulation with one original partial publication.
The12-word reduction padding preserves exact36144B for ALL8 real memory layouts;
original failed unaligned artifact96842067/SHA6ca665aa stays archived.

PASS:149 actual selected CPU,420 WGSL/232 matrix layouts, exact raw/fixed GPU
stage, original171+7 real GPU numerical controls with unchanged bounds and full
owner/tail/fault/causal audits, BOTH original73/4245/4245 profiles and complete
compiler/actual-dispatch/routing-inventory/scratch/pool/source/owner/recovery proofs.
Original historical FMA group equality is explicitly NOT claimed for new matrix
arithmetic; its failed RTE receipt and original wrappers stay unchanged. XMX-only
inventory proof conserves original48*10*(prompt_tokens-1) routes and launch grammar.
Actual cumulative XMX dispatch24/1620/3216 in BOTH modes, real SIMD16 DPAS confirmed.

Diagnostic profile first/repeat12.876510714/11.409061794s;
normal12.780339774/11.291490952s. Normal QSA attention/merge2.665916524s versus
prior persistent matrix3.481686084s(23.43% shorter). This is a diagnostic gain,
not evidence of beating the installed baseline or Strata5.3515s.

Original terminal-negative recovery also PASS: fired1, actual waited queues2,
failed queues0, protected worker exit1, physical recovery; no false normal-session
or owned-zero claim. FRESH original clean45/4245/4245 computes correctly and
physically recovers, but original at-EACH5% gate FAILS:
- firstlong14.899065744s vs installed12.962279132s(14.94% slower);
- repeat10.891321504s vs installed11.147390805s(2.30% shorter, below5% required).
Preserve this failure; do not select repeat alone, change workload/warmup, exclude
cold compilation or install. Staged API/tool/MAX8/semantic/sustained/privacy and
installer were NOT RUN. Source-specific acceptance exits1 only AFTER recovery.
Prepared gated installer is UNEXECUTED; its preflight additionally requires exact
XMX/stage/both-route proofs and original clean/staged gates.

Artifacts under`/media/jmicko/faststore/kuno-development`:
- `strata-parity-qsa-xmx-full-chunk-aligned-20261020-qsa-xmx-public-summary.json`
- `strata-parity-qsa-xmx-full-chunk-aligned-20261020-outcome-public.json`
- `strata-parity-qsa-xmx-full-chunk-aligned-acceptance-20261020-clean-selection-public-summary.json`
- `strata-parity-qsa-xmx-full-chunk-aligned-final-outcome-public-20261020.json`
- `strata-parity-qsa-xmx-full-chunk-aligned-20261020-phase-arrivals-public.json`
- root-only raw receipts retain actual protection/host-before/host-after evidence.
Helper variants and frozen validator remain pinned by
`qsa-xmx-inventory-wrapper-manifest-20261020.json`; originals unchanged.

Next work: preserve this COMPLETE optional matrix bundle; diagnose the cold
first-long penalty and exact selected scalar/reference costs before another
matrix assumption. No repeated baseline or unchanged acceptance runs. The matched
Strata environment leaves XMX disabled and uses32-query scalar fallback: its680ms
QSA timing is NOT an XMX result. Our normal matrix QSA2.666s and expert interval
about6s still leave a large gap; do not claim parity or blame one queue/PCIe rate.
Existing public markers put ~74-77s of LOAD between`global_sources` and
`shared_target_weights`; these are BEFORE-phase labels, so that is the full
`prepare_global_sources_before_prefill` call, NOT the later shared-target call.
It contains static-weight preparation, expert plan/source publication/queue
attachment and final Q6-head activation. Split those exact subphases with bounded
fixed-public metadata if investigating startup; no private samples/content and
no moving setup outside clean clocks. Rebind~7-9ms/preflight~55-60ms cannot
explain the remaining long prompt. The first-long penalty cause is NOT proven.

Fresh independent23:34:47UTC: Intel50,688,000B, NVIDIA compute0(704MiB desktop),
all driver fault counters0, Kuno/Ollama/Vashti200, admission inactive/unpaused/
recoveryfalse/waiting0; no lab worker remains. Owned bounded cached sudo297061 /
sharedshell3556 ends00:17:49UTC Oct21; never kill sharedshell. All commits are
local on the Kuno branch; no API/KNT2/KNR2/config/model/frontend/release change.
Exact contracts and failed/successful receipts are detailed in
[kuno-native-compute-intervals.md](kuno-native-compute-intervals.md#persistent-query-matrix-successor-and-route-inventory-contract).

**Previous checkpoint (2026-10-20 23:19UTC): full-chunk/vector successor alignment fixed,149 CPU controls pass.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Goal ACTIVE/UNMET. Installed worker3fffc266/SHAeac78ce0
and image-enabled gateway36626d30/SHAd3ed07f4 remain unchanged.

Complete persistent-query sourcead3a542db28ab1d181ec982b63242b5360134494 /
SHAc92fed0f7e236dc69537f481c669aadf991d6ae32110b6abf9f20e838476e611,
build`strata-parity-qsa-xmx-persistent-query-20261020`, passes147 actual selected
CPU controls,420 WGSL/232 matrix layouts, exact GPU stage, original171+7 GPU
numerical controls and BOTH full original73/4245/4245 profile modes. Complete
source/owner/compiler/actual-XMX dispatch/routed-work/scratch/pool proofs PASS.
Both recover physically; independent final state/services normal, faults0.
Profile first/repeat13.6317/12.2584s; normal13.6275/12.1523s. Normal QSA attention/
merge3.4817s. This is no speed win; no clean/staged/install acceptance was run.
Entire bundle remains intact. Source/helper/receipts are frozen; preserve old
RTE historical FMA group-equality failure and its correct applicability account.

New full-chunk successor retains the persistent query and explicit RTE fix, but
stages one64-dimension K/V scale group across ALL64 cells. Each complete P.V
product now stays in registers and initializes its original partial ONCE,
avoiding the optional second microtile's global read/add/write. K/V and P hi/lo
are genuine vec4 shared stores and views; original scalar owner/source offsets
and original merge stay. Shared storage36144B, below actual49152B. This is the
connected staging+matrix+publication change, not an isolated component rollback.
All149 actual selected CPU controls pass, including exact once partial/cell
coverage, packed scalar coverage, cooperative-view bounds and exact36144B
reflection for ALL8 actual compiled memory layouts. Original
numerical references/bounds/171+7 hardware controls and all clean gates remain.
Unaligned source96842067/SHA6ca665aa fails its ORIGINAL offline shared-envelope
check before any GPU launch: implicit36132B, some explicit layouts36136B due to
vector alignment. Preserve that failed artifact/log. Padding only reduction9
->12 words makes every stage size a multiple of16, ALL8 actual compiler layouts
now36144B. No check/threshold is removed or loosened. New freeze/release/offline/
GPU/full profiles/clean/staged/install are pending.

Reference-source audit also confirms matched Strata5.3515s did NOT opt into
XMX. `public-reference-client.py` has an explicit container-env allowlist with
no`STRATA_PROMPT_ATTN_XMX`; pinned SYCL XMX mode defaults0 and caller uses the
original decode-attention batch fallback,32 queries. This matrix candidate is
an opt-in successor, NOT the measured reference's selected attention kernel.
Do not attribute Strata's680ms timing to XMX. Keep original scalar fallback and
its numerical controls; compare exact scalar code/ISA if matrix remains slower.

Artifacts under`/media/jmicko/faststore/kuno-development`:
- `strata-parity-qsa-xmx-persistent-query-20261020-qsa-xmx-public-summary.json`
  proves actual ordinary/native SIMD16 DPAS and cumulative dispatch24/1620/3216
  in BOTH modes, all original and additional numerical controls and recovery.
- `strata-parity-qsa-xmx-persistent-query-20261020-outcome-public.json` exports
  only compiled-public aggregate metrics/profile counters/clock metadata.
- `strata-parity-qsa-xmx-persistent-query-20261020-reencode-public-summary.json`
  proves actual idle-pool reuse and preserves all original owner/budget checks.
- `qsa-xmx-inventory-wrapper-manifest-20261020.json` pins8 variant wrappers;
  original wrappers unchanged; frozen validator SHA
  c17daf2a8da62fb11dad1850042dc4932c872ccf63e1aadb25ab5b0419792bf1.

Independent post-run state: Intel50,688,000B, NVIDIA compute0, driver faults0,
Kuno/Ollama/Vashti200, admission inactive/unpaused/recoveryfalse/waiting0.
No GPU job remains. Owned bounded sudo297061/sharedshell3556 ends00:17:49UTC
Oct21. Never kill sharedshell or reset/reboot GPUs. No API/KNT2/KNR2/config/model/
frontend/release change. Exact source contracts and applicability rationale:
[persistent-query and full-chunk matrix contracts](kuno-native-compute-intervals.md#persistent-query-matrix-successor-and-route-inventory-contract).

**Previous handoff (superseded; the RTE unit below is now terminal):**

**Previous handoff (2026-10-20 22:47UTC): explicit rounding passes exact GPU stage and178 numerical controls.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Complete candidate source78d948aa99d9aced6a0e47d8679720fa01ed7a9a /
SHA7047b38173ed00e18c98372b00412a589297b409a6d9eadfbe47ad427a67bfbf,
build`strata-parity-qsa-xmx-rte-20261020`, retains ALL earlier components.
146 selected CPU,420 WGSL/232 matrix layouts PASS. Same-input actual GPU
comparison proves RAW cast helper yields1024 low zeros(expected1),1023 low and
identity mismatches. FIXED explicit integer RTE gives0 high/low/identity
mismatches, actual low zeros1(expected1), nonfinite0, full padding/fault0,
owned erasure and physical recovery PASS. No bound was lowered, no extra
barrier/buffer/copy/SLM was added, no input restriction changed.
Owned`strata-parity-qsa-xmx-rte-stage-control-20261020-public-summary.json` records
both real paths and recovery. This is direct observed behavior, not a claim
identifying which Mesa optimization caused the raw cast residual to vanish.

The complete protected unit`kuno-native-qsa-xmx-rte-connected-20261020` is ACTIVE.
Original171 GPU controls PLUS all7 new actual large-XMX F64/fault controls PASS,
including16/31/32 rows, offsets0/32/64 and32768-capacity boundary; original
scalar refs, all operation-derived bounds and original full-owner/tail/fault/
causal guards remain. Operator close/owned zero/physical recovery PASS. It is
now loading full original target/MTP for the original whole-model profile,
then normal scheduling profile and all compiler/count/pool/actual-XMX proofs.
Do NOT overlap GPUs/services/heavy CPU, launch another test or install while live.
No whole-model/speed/clean/staged/install claim yet. Prepared acceptance helper
`check-native-qsa-xmx-rte-acceptance-20261020.py` is UNEXECUTED and pins this exact
source/hash plus stage/full-XMX/pool proofs, retaining original at-EACH5% gate.

Artifacts and current log under `/media/jmicko/faststore/kuno-development`:
`strata-parity-qsa-xmx-rte-connected-20261020.log` (fixed-public metadata only),
operator report includes actual7 accepted controls. Full connected wrapper
`check-native-qsa-xmx-connected-20261020.py` requires actual native16 DPAS and
full request XMX counts derived from original batching, not fallback success.
Next after successful/recovered full proof: original terminal-negative, FRESH
clean45/4245/4245 at-EACH5%, staged API/tools/semantic/privacy/sustained, then
exact official install/rollback transaction. Do not skip stages or select repeat
alone. Goal ACTIVE/UNMET; matched Strata5.3515s remains target.

Installed worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 unchanged.
Bounded cached sudo297061/parent sharedshell3556 ends00:17:49UTC Oct21; old
220143 naturally ended22:38UTC. Never kill sharedshell or reset/reboot GPUs.
No API/KNT2/KNR2/config/model identity/frontend/release change.
Exact evidence/contract:
[explicit rounding](kuno-native-compute-intervals.md#observed-dynamic-residual-failure-and-explicit-rounding-correction).

**Previous active handoff:**

**Previous handoff (2026-10-20 22:42UTC): GPU stage failure localized; explicit rounding candidate.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Fixed-public dynamic stage source20630c01/SHA83bdd697
passes145 CPU,420 WGSL/224 matrix layouts and completes its GPU check and physically recovers.
Actual high halves all1024 match host; low1023 and identity1023 mismatch, max_abs
[0,3.96484375,3.96484375], fault/padding0. This directly locates bad low parts
in the shared production stage. Preserve FAILED stage summary
`strata-parity-qsa-xmx-stage-control-20261020-public-summary.json`. No speed run.

New complete source78d948aa uses exact IEEE half RTE in shared CPU/GPU integer
bit code to materialize the rounded F32 high value BEFORE residual subtraction.
High/low stores then receive exact intended halves. No extra barrier, scratch,
buffer, copy, source restriction or lowered bound. All146 selected CPU checks
pass, including independent half-crate boundary/tie/neighbor tests across both
signs. The old cast-based stage remains ONLY a fixed-public RAW comparison; the
CPU cast-chain test now checks that original raw probe (one call/chain), while
all original12 numeric references and7 new large-case F64 bounds remain intact.

Frozen release`strata-parity-qsa-xmx-rte-20261020` is built, source
78d948aa99d9aced6a0e47d8679720fa01ed7a9a / SHA
7047b38173ed00e18c98372b00412a589297b409a6d9eadfbe47ad427a67bfbf. Next
420 WGSL/232 matrix layouts, protected SAME-input raw+fixed stage comparison,
then original171+7/full profiles/actual counts/physical recovery. Only after
all pass may original terminal/clean-at-EACH5%/staged-quality/install gates run.
No performance/install acceptance yet. Entire earlier pool/transfer/matrix/
QK/softmax/PV bundle stays; do not roll back components on isolated timings.
Exact contract/evidence:
[explicit rounding](kuno-native-compute-intervals.md#observed-dynamic-residual-failure-and-explicit-rounding-correction).

Installed3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 unchanged. Stage-only
unit is terminal/recovered. Goal ACTIVE/UNMET; Strata5.3515s still leads correct
pool repeat9.6808s. Bounded sudo297061/parent sharedshell3556 ends00:17:49UTC
Oct21 (never kill sharedshell). No reset/reboot, API/config/model/frontend/release
change. The new laboratory selector is captured before original env scrub only
when public_check AND matrix_check; private worker cannot activate it.

**Previous active handoff:**

**Previous handoff (2026-10-20 22:26UTC): XMX executes; precision failure diagnosed to next probe.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Complete matrix candidate source
585dc471c56879466a9bcbe40f6a8229ff06e6a9 / frozenSHA
fd43c5cb4cece90f4987f50a71d482072a3d2ca21fae6ea80eaff9b762ca03fb,
build`strata-parity-qsa-xmx-logical-20261020`, retains ALL prior components.
144 CPU and420 WGSL/216 matrix offline layouts pass. Actual XMX DOES execute:
ordinary+retained native compilers each emitSIMD16/48 DPAS/5777 instructions,
zero spills/fills. Protected operator run then correctly FAILS first new large
F64 case(start0,count16,offset0) and physically recovers. No whole-model profile,
clean speed, staged API or installation was attempted for XMX. Do not describe
this as an unexercised selector anymore, or as numerical/performance acceptance.

Original scalar has0 F64 mismatches. Candidate partial78071 mismatches,
max_abs0.00382843435/max_bound0.000251863085; stats696 mismatches,
max_abs0.00181380627/max_bound0.000112587659. Existing operation-derived bounds
are UNCHANGED. Half-precision-scale error suggests hi/lo staging/conversion is
worth testing but DOES NOT establish a driver/compiler bug. New CPU test
`actual_compiled_split_retains_half_rounding_before_f32_residual` proves both
F32->F16->F32->FSub chains ARE present in actual generated SPIRV (2 pairs), so
CPU compiler conversion elimination is NOT the established cause. All145 selected
CPU controls pass with that additional test. Production source is unchanged by
this test-only addition; frozen585dc471 is the actual failed GPU artifact.

Next: add a bounded fixed-public GPU numeric control of the ACTUAL shared hi/lo
staging helper, compare staged/reconstructed values to the host half reference,
and inspect DPAS operand/accumulation/scale behavior. Consider exact integer RTE
half-rounding in F32 if actual evidence supports it; do not blindly blame Mesa,
change the quant, lower bounds or abandon half of the connected kernel. Preserve
old12 small bitwise/F64 references, full-owner/tail/fault/causal checks and the
additional7 large actual-path controls. Only after all pass run original complete
profiles/count proof, terminal/clean-at-EACH5% gate and staged quality/install.

Owned artifact base`/media/jmicko/faststore/kuno-development`:
`strata-parity-qsa-xmx-logical-precision-postmortem-public-20261020.json` has exact
actual compiler rows and passed=false/recovered=true;
`strata-parity-qsa-xmx-logical-connected-20261020.log` contains only fixed-public
numeric metadata, including both F64 comparison rows. Protected raw compiler
code remains in its original operator directory, not public/private request data.
CPU log`native-qsa-xmx-precision-complete-cpu-20261020.log`:145 pass. Complete
wrapper`check-native-qsa-xmx-connected-20261020.py` retains original171+7 gates,
actual ordinary/native16 DPAS and derived full-model launch-count requirements.

No GPU job remains. Independent22:25:14UTC: Intel50,688,000B, NVIDIA compute0,
all driver faults0, Kuno/Ollama/Vashti200, admission inactive/unpaused/recoveryfalse/
waiting0. Installed3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 unchanged.
Bounded sudo refresh297061 (parent sharedshell3556; NEVER kill sharedshell) ends
2026-10-21 00:17:49UTC; old220143 naturally expires22:38UTC. Goal ACTIVE/UNMET;
Strata5.3515s still leads prior correct candidate repeat9.6808s. No reset/reboot,
API/KNT2/KNR2/config/model identity/frontend/release change.
Details: [matrix candidate](kuno-native-compute-intervals.md#complete-matrix-attention-successor-under-implementation).

**Previous active handoff:**

**Previous handoff (2026-10-20 22:19UTC): complete48KiB kernel plus authentic fault-view fixture.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Final candidate source585dc471c56879466a9bcbe40f6a8229ff06e6a9,
SHAfd43c5cb4cece90f4987f50a71d482072a3d2ca21fae6ea80eaff9b762ca03fb,
build`strata-parity-qsa-xmx-logical-20261020`. Entire connected QK/softmax/PV
kernel and all previous pool/matrix/transfer components are retained. Actual
scratch48420B fits B70 Vulkan49152B. All144 selected CPU controls pass again;
release is frozen. Exact-source offline420/216 and protected original171+7 new
ACTUAL large-XMX GPU/full-profile/recovery proofs follow; no speed/install claim.

Earlier3ac1f893/SHAdcc438de passes CPU/offline and hardware admission, then stops
BEFORE XMX execution because the new fixture supplies its full padded fault
allocation instead of production's genuine logical4-byte fault view. Corrected
fixture derives that view from its original real BufferId; full parent/tail
still read/audited. The strict production4-byte check is UNCHANGED. Do not weaken
it or treat either unexercised run as accepted numerical/performance evidence.
Failed static/recovered receipt:
`strata-parity-qsa-xmx-slm48-20261020-failure-public.json` under the artifact base.
Earlier58660B admission failure remains preserved separately.

Installed worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 unchanged.
Original at-EACH5% clean gate and first-use compilation timing remain; no API,
config, model identity, frontend or release change. Goal ACTIVE/UNMET.
New bounded sudo refresh297061/parent sharedshell3556 ends2026-10-21 00:17:49UTC;
old220143 ends22:38:06UTC naturally. Never kill the shared shell. No reset/reboot.
Exact contract/evidence:
[matrix attention](kuno-native-compute-intervals.md#complete-matrix-attention-successor-under-implementation).

**Previous active handoff:**

**Previous handoff (2026-10-20 22:11UTC): corrected48KiB matrix attention.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. First XMX sourcee2935521/SHAa38e2801 passes144 CPU and
all420 WGSL/216 matrix layouts but FAILS actual-path exercise: Vulkan on this
B70 exposes49152B SLM, not the assumed>=58660B. Selector correctly refuses before
XMX launches. No whole-model or speed run. Physical recovery passes. Preserve
`strata-parity-qsa-xmx-first-failure-public-20261020.json`; no arithmetic failure
or accepted XMX math claim is established by this unexercised run.

Successor source3ac1f893 stages8 heads/two K scale groups/128 PV dimensions at a
time using all8 subgroups. Full QK/softmax/PV remains connected; complete layout
requires48420B within ACTUAL49152B. No guard, device limit or oracle is weakened.
All144 actual CPU controls pass again. Frozen release build
`strata-parity-qsa-xmx-slm48-20261020` is built,
source3ac1f89318ffa263d030642a205f2034b9e409ae /
SHAdcc438dea11fa6c0c825dcbad2dd668e2184367fc0c53bbf21bd9d64ba8aeb1f; exact-source420/216 offline,
original171+7 actual XMX GPU, both requests/profile modes/actual launch counts and
physical recovery remain required before clean speed acceptance. Do not overlap
shared GPU/services during these tests, or install an unaccepted artifact.

Installed3fffc266/SHAeac78ce0, gateway36626d30/SHAd3ed07f4 unchanged. Previous pool
bundle repeat9.6808s improves13.16%, but FIRST13.2478s fails original at-EACH5%
gate. Retain all components and original first-use timing/gates. Strata5.3515s
still leads; goal ACTIVE/UNMET. No API/config/model/frontend/release change.
Cached sudo refresh220143 ends22:38:06UTC; parent sharedshell3556 is NOT ours to
kill. No GPU/device reset/reboot. Exact source/evidence/test contract:
[matrix attention](kuno-native-compute-intervals.md#complete-matrix-attention-successor-under-implementation).

**Previous handoff (2026-10-20 22:02UTC): complete QSA matrix candidate.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. New sourcee2935521 implements the connected Rust
QK/softmax/PV matrix kernel, using original full owners and58660B device-local
scratch. New explicit`strata-qsa-xmx`/builder`--qsa-xmx` retains ALL previous
components, including real completed-pool reuse. Original short/scalar/decode/MTP
paths and all original numerical/owner/capacity/erasure controls remain.
All144 selected CPU controls PASS. Frozen release`strata-parity-qsa-xmx-20261020`,
sourcee2935521db44eae698bb3214e252920ebb4dd732 /
SHAa38e2801b440a5eeec4ae787e8a4cc871b0a661dab736f4b1e3735eb354164c5,
is built; offline420 WGSL/216 matrix variants, original171 GPU+7 additional
ACTUAL full-batch XMX controls, both public profiles/actual launch-count and
physical recovery follow. No GPU/speed/install acceptance yet. Never accept only
old <=8-row fallback tests as validation of the new16..32-row kernel.
New large F64 oracle entry shares the SAME independent math/derived bounds;
old8-row entry keeps its domain. Future/nonfinite negatives require real sticky
fault and untouched inactive/tail owners, not a failed-input payload math claim.
Exact implementation/test contract:
[matrix-attention successor](kuno-native-compute-intervals.md#complete-matrix-attention-successor-under-implementation).

Installed worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 unchanged.
Prior complete pool bundle's clean repeat9.6808s improves13.16%, but FIRST13.2478s
fails original at-EACH5% selection; retain whole bundle and do not weaken gates.
Strata repeat5.3515s still leads. Goal ACTIVE/UNMET. No API/protocol/configuration,
model identity/frontend/release change. Last physical21:50:53UTC clear/healthy;
no GPU job launched for the new candidate yet. Cached sudo refresh220143 ends
22:38:06UTC, parent sharedshell3556 must never be killed. No reset/reboot.

**Previous handoff (2026-10-20 21:51UTC): actual reuse passes; clean FIRST gate fails.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`; preserve canonical Vashti notes. Complete worker source
`865a9c166acc83203fd82c4052a2f6ee2e2240fd`, frozenSHA
`361978398e3c798a6c59eb09dc091719d0d6862787b5ea72a5ab95175c9c3a28`, build
`strata-parity-idle-pools-coupled-20261020`, retains ALL prior components.
Installed3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 remain unchanged.

Final-source136 selected CPU controls,420/208 offline, original129+42 GPU,
full compiler/producer/count/source/owner/budget proofs, BOTH full public
73/4245/4245 profiles and physical recovery PASS. Unlike earlier unused paths,
actual original pools ARE retired/adopted. Normal long expert intervals reuse
89/96 recordings and36183/40107 commands; process-total adoptions60/178/296
increase across requests. Reuse can also serve small captures, so its recording
count may legitimately exceed LARGE rolling-recording count. Every original
rolling/producer/count assertion remains; no source or numerical gate was removed.
Normal sampled layer24 retained_prepare8.36->0.117ms threadCPU and encoding
9.18->6.072ms; these are diagnostic intervals, not whole-request speed acceptance.
Normal long prompt11.6086/10.1288s is instrumented diagnostic only.

Terminal-negative actual2-queue recovery and subsequent FRESH clean45/4245/4245
requests/physical recovery PASS. Original at-EACH-position5% speed SELECTION FAILS:
FIRST12.962279132->13.247788385s (2.20% slower); repeat11.147390805->9.680801649s
(13.16% shorter). Do NOT hide FIRST, change warmup, move compilation outside the
clock or relax the gate. Staged API/semantic/sustained acceptance and installation
were NOT run. Prepared installation/post-install helpers are UNEXECUTED, not proof.
Same-host Strata repeat5.3515s still leads; goal ACTIVE/UNMET.

No GPU job remains after `kuno-native-idle-pools-coupled-acceptance-20261020`
exits at its intended failed selection assertion AFTER recovery. Independent
21:50:53UTC: Intel50,688,000B, NVIDIA compute0, all driver faults0, allHTTP200,
admission inactive/unpaused/recoveryfalse/waiting0. No reset/reboot. Bounded owned
sudo refresh220143 (parent sharedshell3556; never kill shared shell) ends22:38:06UTC.
Artifacts under `/media/jmicko/faststore/kuno-development`: owned
`strata-parity-idle-pools-coupled-20261020-reencode-public-summary.json`,
`idle-pools-coupled-full-public-counters-20261020.json`,
`strata-parity-idle-pools-coupled-acceptance-20261020-clean-selection-public-summary.json`.

Next high-potential target is complete QSA matrix attention: normal attention
2.9494s, of which QSA family attention/merge1.5059s; expert wait3.5382s also remains.
Pinned MIT Strata's experimental XMX v2 couples QK, hi/lo half staging/scales,
64-cell packing, softmax and PV. Port the connected math/layout, not isolated
halves; retain old short/scalar references and add ACTUAL large-path independent
F64/full-owner/fault/causal validation (old <=8-row fallback tests cannot prove a
new >=16-row kernel). No XMX implementation or performance claim yet. Preserve
all existing components, erasure and gates. Exact contract/evidence:
[pool reuse outcome](kuno-native-compute-intervals.md#completed-pool-reuse-runtime-and-clean-selection-outcome).
No API/KNT2/KNR2/config/model identity/frontend/release change.

**Previous handoff (2026-10-20 21:34UTC): coupled completed-pool reuse.**

Use ONLY `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`; preserve canonical Vashti notes. Installed worker
3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 remain unchanged.
Goal active/unmet; matched Strata repeat5.3515s still leads.

Exact-count cdbb041b and capacity2137a0f1 each pass original numerical,
compiler/producer/ownership, both full public profiles and physical recovery,
but FAIL actual reuse exercise (ZERO idle candidates/re-encodes). Their timings
are fallback diagnostics, not accepted optimization evidence. Preserve failed
exercise receipts. Do not infer that queue capacity alone was the cause.

Complete source865a9c16 retains every prior component and connects completed
pool retirement at ORIGINAL idle eviction/I/O cleanup to later adoption. At
most one typed content-free reset pool inventory survives: device/Graph/family,
MAIN/COPY/SHARED pools, timelines/monotonic epochs and command handles only.
Original plans/signatures/source/resources/private owners are still destroyed
before original erasure. Original real flight/access/handoff joins authorize
retirement; all fresh plan/source/descriptor/owner/barrier checks still apply.
Small unkeyed/unprofiled MAIN captures can also consume that original capacity
and keep unused handles for later retirement. Actual old+new allocations remain
inside the original64MiB metadata cap. No GPU/device reset/reboot.

All136 selected CPU controls pass on final source. Frozen release build
`strata-parity-idle-pools-coupled-20261020`, workerSHA361978398e3c798a6c59eb09dc091719d0d6862787b5ea72a5ab95175c9c3a28,
is built. Offline420/208,
original171 GPU/full proof chain, BOTH public profile modes and actual pool
retirement/adoption must pass before clean selection. Public counters explicitly
separate per-expert intervals from process-total retirements/adoptions between
intervals. Fallback-only success cannot validate this new path. No current
GPU execution, clean/terminal/staged/sustained/install acceptance claimed yet.
Artifacts/helpers under `/media/jmicko/faststore/kuno-development`; exact contract:
[completed resource retirement](kuno-native-compute-intervals.md#coupled-completed-resource-retirement-and-adoption).

Last independent21:33:09UTC: Intel50,688,000B, NVIDIA compute0, all driver faults0,
Kuno/Ollama/Vashti200, admission inactive/unpaused/recoveryfalse/waiting0.
Bounded owned sudo refresh PID220143/parent sharedshell3556 ends22:38:06UTC.
Never kill the shared shell. No API/KNT2/KNR2/config/model identity/frontend/
release change. Backend-only internals require no Vashti consumer change.

**Previous handoff (2026-10-20 20:49UTC): complete command re-encoding candidate.**

Use ONLY`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`; preserve canonical Vashti notes. Installed worker
3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 are unchanged.
The previous complete template bundle passes numerical/recovery checks but fails
clean FIRST13.7806s vs12.9623s, despite repeat10.3836s vs11.1474s. Not installed.
Do not roll back any component or weaken any original reference/selection gate.

New source`cdbb041bc34df1d9dac8fb483e14cef9b1211114`, frozenSHA
`82e9581007365556e44e083b84ef4250976d8b77249a2773fab82f6af9a4d3bb`, retains ALL
components and re-encodes original completed expert recordings using their
original3 pools, command inventory, descriptor pool and timeline epochs. Only
original flight-book-not-live, ready/unnamed/unprofiled/same-family/header and
large expert/full32-push cases qualify. Queue COUNTS must match; queue order can
differ via allocation-free permutation using old node scratch. Full original
new plan/ranges/bindings/source/owner/barrier/64MiB capacity/erasure checks remain.
A source guard is released before retiring old resource owners, and reacquired
with original full validation before encoding. No side cache or cross-erasure
retention. Vulkan command-pool lifecycle reset is never a GPU/device reset.

All132 selected actual CPU controls,420/208 offline and original129+42 GPU
controls pass. Protected`kuno-native-command-reencode-connected-20261020` is
ACTIVE: full original proofs + GPU-sampler and normal-host-only whole-model runs.
New proof requires ACTUAL re-encoded recordings/commands in every long request;
fallback success alone is insufficient. No clean/staged/install claim yet.
Do not overlap shared GPU/services or heavy CPU work while profiles run.
Artifacts/helpers under`/media/jmicko/faststore/kuno-development`, build/prefix
`strata-parity-command-reencode-20261020`, anticipated owned
`strata-parity-command-reencode-20261020-reencode-public-summary.json`.

Last independent20:44:51UTC recovery before the new run: Intel50,688,000B,
NVIDIA compute0,all driver faults0,allHTTP200 and normal admission. The new
bounded sudo refresh PID220143 (parent sharedshell3556) runs until22:38:06UTC;
old128882/128884 naturally exited. Do not kill sharedshell or reset/reboot GPUs.
Goal active/unmet. No API/KNT2/KNR2/config/model identity/frontend/release change.
Exact contract and test limits:
[idle recording re-encoding](kuno-native-compute-intervals.md#complete-idle-recording-re-encoding-candidate).

**Previous checkpoint (2026-10-20 20:31UTC): template complete; clean selection FAILS; safe checkpoint.**

Use ONLY`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`; preserve canonical Vashti checkout/notes. Installed
worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 remain unchanged.
No API/KNT2/KNR2/configuration/model identity/frontend/release change.

Complete template source`03815d0b8f47c547bf4729927dd2f98f9dd8a974`, workerSHA
`f0c926f285eea8f8f7c225ff580564da10a83699a5e353077cc9db5b1663eb82`, passes128
selected CPU,420/208 offline, original129+42 GPU, complete original compiler/
producer/count/ownership proofs, BOTH GPU-sampler and normal-host-only whole
public73/4245/4245 runs. Every46746/166220/166220 expert push dispatch really
uses templates in both modes (fresh sets0). Retains all live descriptor checks,
full writes, shader/queue/epoch/barrier/source/owner/budget/erasure components
and unsupported/creation-failure fallback. No components rolled back.

Terminal-negative passes with2 actual recovery queues joined and no failed
recovery. Subsequent FRESH clean45/4245/4245 requests and actual physical
recovery pass, but original5%-faster-at-EACH-position SELECTION FAILS:
FIRST12.9623->13.7806s (6.31% slower), repeat11.1474->10.3836s (6.85% shorter).
Staged API/semantic acceptance and installation were NOT started. No new
sustained/decode/frontend/release claim; Strata repeat5.3515s still leads.
Owned summaries`strata-parity-push-template-20261020-public-summary.json` and
`strata-parity-push-template-clean-selection-public-summary-20261020.json` under
`/media/jmicko/faststore/kuno-development`. Do not select only the repeat,
change warmup input, move JIT outside the clock or weaken reference/gates.

Actual independent20:30:46UTC: Intel50,688,000B,NVIDIA compute0,all faults0,
Kuno/Ollama/Vashti200, admission inactive/unpaused/recoveryfalse/waiting0.
No GPU job remains. Goal active/unmet. Next target: completed recording resource
preparation and first-use costs. Normal-path CPU encoding+retained dominate
~79% of the sampled third request/layer24; original GPU sampler serializes its
layer. Original Commands ALREADY bulk-allocates per pool. Resource reuse is
NOT implemented; require original completed-flight/queue/header ownership,
old+new actual capacity caps, monotonic epochs, full fresh bindings/barriers,
private reset/erasure and terminal recovery. Retain the whole coupled bundle.
Owned sudo refresh128882/128884 expires20:39:26UTC; sharedshell3556 must NOT be
killed. No GPU/device reset/reboot. Exact contracts, metrics, proof limits, source
pointers and next gates:
[final checkpoint](kuno-native-compute-intervals.md#template-clean-selection-and-safe-checkpoint).
No Vashti consumer change is required.

**Previous handoff (2026-10-20 19:48UTC): host capture diagnosis.**

Use ONLY`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`; preserve canonical Vashti checkout/notes. No API,
KNT2/KNR2/configuration/model identity, frontend, version or release change.
Installed source3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 are unchanged.
The complete candidate retains ALL prior wide/attention/bulk/loading/miss-ring/
barrier/source/push/slot/compact components. Do not partially roll them back.

Compact5ac1e4e/SHA1a789884 completes120 CPU,420/208 offline, original129+42
GPU controls, full real requests/all producer/compiler/routed-count proofs,
terminal-negative recovery and fresh clean45/4245/4245 requests. Native QSA
4320->2654 instructions does NOT improve whole-prompt time by itself. Clean
repeat11.1474->10.2885s (7.71% shorter), FIRST12.9623->13.7110s (5.78% slower).
The existing5%-faster-at-EACH-position selection gate FAILS. Full staged API/
semantic tests and installation were NOT started. No sustained or install claim.
Owned summaries`strata-parity-qsa-compact-{public-summary,clean-selection-public-summary}-20261020.json`.

Actual19:46:19UTC recovery: Intel50,688,000B, NVIDIA compute0, all driver faults0,
Kuno/Ollama/Vashti200, admission inactive/unpaused/no recovery/waiting0. No GPU
job remains from acceptance. The bounded owned sudo refresh128882/128884 lasts
until20:39:26UTC. Never kill shared shell3556 or reset/reboot either GPU.

Next explicit`strata-public-host-clock` / builder`--public-host-clock` diagnostic
retains the complete bundle and adds eight public-only wall/THREAD-CPU intervals
inside capture. No extra GPU command, wait, footprint or private observation.
All123 actual App CPU controls pass. The sole dependency change is one edge to
already-locked libc; no version/download/system change. Do not call elapsed
capture pure CPU cost or wall-minus-CPU exclusively GPU waiting. The original
std launcher already reuses its IR Scope/INFO via thread-local storage; don't
assume repeated IR Scope construction without evidence. Freeze,420/208 offline,
original171 GPU/full request/new host-clock proof and physical recovery follow.
Exact contract, failed first scratch guard/fix, identities, commands and limits:
[CPU-clock diagnosis](kuno-native-compute-intervals.md#clean-selection-outcome-and-fixed-public-cpu-clock-diagnosis).
Goal remains active/unmet: matched Strata repeat5.3515s still leads. Backend-only
work needs no Vashti consumer change. Earlier entries are chronological context.

**Active complete compact-QK successor (2026-10-20).**

Corrected scratchb1fb992b/SHA37cf3a1e completes118 CPU,420/208 offline,
original129+42 new GPU controls, full requests/all producer/compiler/routed
counts and physical recovery. Repeated profile11.1290 ->10.6244s; expert portion
6.9208 ->6.4216s with actual barrier calls164876 ->85049. Real queue segments
stay37359. These are profiles; no clean/staged/terminal/sustained/install acceptance.
Actual19:24:36UTC is clear/healthy. Installed identities unchanged.

The next complete`strata-qsa-compact` retains paired/uniform/FMA/PV/local-fault
and every earlier component, but eliminates duplicate QK code by emitting the
EXACT original guarded body once. Producer verifies that its uniformly true
form is byte-identical to the old full-pair body. All120 actual CPU controls
pass, including complete-source reconstruction and actual selected compiled
status/fault SPIR-V, unchanged barriers/geometry and only removed duplicated
math. Original129+42 GPU/full request/compiler/group/recovery/timing gates follow.
Exact contract/results:
[compact attention](kuno-native-compute-intervals.md#next-complete-attention-code-size-candidate).
No API/KNT2/KNR2/configuration/model identity or Vashti changes. No release.

**Active disjoint expert scratch candidate (2026-10-20).**

Uniform84316ccb/SHA7e6b0991 completes all114 CPU,408/208 offline,129 GPU,
full request/compiler/inherited producer/routed-count and recovery gates.
Repeat11.1290s is unchanged; no clean/staged/terminal/sustained/install acceptance.
Physical18:53:52UTC is healthy/clear. Source review disproves per-dispatch INFO
uploads. The next complete`strata-expert-scratch-ring` couples two real input/GU
slots and exact GU/DOWN image views inside the SAME original priced roots.
No arithmetic, six-phase grammar, full source, owner/hazard/epoch/queue/fence,
budget or erasure exception. Actual array metadata and all original checks stay.
Exact contract and pending gates:
[disjoint expert scratch](kuno-native-compute-intervals.md#completed-uniform-attention-and-disjoint-expert-scratch-candidate).
No API, protocol/configuration/model identity or Vashti change. Installed worker
and gateway are unchanged; this is an explicit unselected backend candidate.
First frozen76836aa8/SHA36efb2eb passes117 CPU,408/208 offline and129 original
GPU controls but FAILS the first whole-model raw route-header guard. No timing
or full producer acceptance; not installed. Actual19:08:55UTC recovery is clear
and healthy. Retain every component/guard and diagnose the exact failing layer
with compiled-public-only phase labels; no private content or GPU fault claim.
Source inspection found the old dequantizer's exact80MiB guard rejecting the
new smaller views. Corrected producer uses a DISTINCT SlotDequant identity
with exact role extent AND zero relative offsets; original shaders/guards stay
byte-identical. All118 selected CPU controls pass. Next includes42 NEW exact-role
GPU controls in addition to the original129, then full requests/recovery/timing.
The phase-only diagnostic4507e153 was frozen but never run. No installed change.



**Active complete uniform-QK candidate (2026-10-20).**

Complete FMA e854a89d/SHA420dc7d0 passes111 CPU,408/208 layouts, all129 GPU
controls, complete requests/all old/new producer proofs, native FMA ISA and
physical recovery18:33:15UTC. All three non-dyadic QSA controls now match the
scalar bits as well as passing unchanged F64 bounds; original nine bitwise
cases remain. Repeat profile11.1741s is unchanged/slower versus11.0518s,
with actual expert groups26338 ->26673. No clean/staged/terminal/sustained or
install acceptance. Keep the full source; static instruction reduction alone
is not a whole-prompt win. GDN ordinary16 proof remains missing as documented.

The next coupled `strata-qsa-uniform` keeps all components and additionally
broadcasts immutable selected IDs, hoists exact shuffle indices, and samples
the exact original lane-zero score before uniform sticky-fault checks outside
the lane-zero store branch. No changed arithmetic or skipped failure/owner/
queue/footprint/budget checks. All114 actual App CPU controls pass, including
complete-source undo, real module topology and adversarial other-lane cases.
Frozen84316ccb/SHA7e6b0991
(`strata-parity-qsa-uniform-20261020`) passes408/208 offline and all129 original
GPU controls with physical recovery. Full requests/actual compiler/group-count/
recovery/performance run in `kuno-native-qsa-uniform-connected-20261020`.
Exact contract: [complete FMA and uniform successor](kuno-native-compute-intervals.md#complete-fma-result-and-coupled-uniform-qk-successor).
No API/protocol/configuration/model identity or Vashti consumer change.

A new bounded credential refresh owns outer128882/loop128884, parent3556,
18:39:26..20:39:26UTC; verified old64142 retired. Shared shell untouched.
No current GPU job; FMA's recovery is complete. Installed binaries unchanged.

**Active complete QK FMA candidate (2026-10-20).**

Actual pinned default INT8 Strata attention disassembly proves seven
chronological QK MADs after the first multiply; the previous universal
contract(off) assumption was wrong for this AOT binary. The new opt-in
`strata-qsa-fma` matches those explicit FMA steps, retaining first/dequantization
rounding and every original reduction, owner, fault and scheduling guard.
The independent scalar reference, original nine bitwise QSA cases and three
operation-bounded F64 cases stay unchanged. All111 actual App CPU controls pass,
including the17 affected complete-source/SPIR-V controls. The complete bundle
retains every preceding component, including native GDN16 with its documented
ordinary32 fallback limitation. Freeze/offline/original129 GPU/full request/
actual compiler/recovery/performance gates remain next; no speed/install claim.
Exact contract: [explicit QK FMA](kuno-native-compute-intervals.md#connected-explicit-qk-fma-candidate).
No API/protocol/configuration/model identity or Vashti consumer change.

**Latest Kuno checkpoint (2026-10-20 18:15UTC; installed native32 unchanged).**

Completed canonical-source5b173ad8/SHA cc0aa6aa passes86 CPU,408/208 layouts,
all129 original GPU controls, full real requests/all producer proofs and recovery.
Capture CPU2.0168 ->1.3818s; clean repeat10.7871s versus installed11.1474s is
a modest3.2% difference, still behind matched Strata5.3515s. Complete enqueue
3b35ea10/SHA23747779 passes90 CPU and all GPU/full-request/producer/recovery
gates, but profile11.0527s is unchanged. Neither is installed.

New GDN source022d799b/SHAe3ba33c0 passes94 CPU,408/208 layouts, all129 original
GPU controls, complete requests and all inherited producer/recovery gates.
Native GDN is SIMD16 with zero spills/fills. Ordinary Wgpu remains SIMD32, so
the NEW both-pipeline ISA assertion FAILED after those original gates passed.
Keep that failure visible; do not claim complete producer acceptance. The
owned public diagnostic explicitly records the limitation. Repeated profile
11.0518s is unchanged; no clean/staged/terminal/sustained or install acceptance.
Complete source remains retained opt-in. Actual correctness is not the blocker;
this did not materially improve whole-model prompt speed.

Next focus is actual selected Strata/Kuno attention compiler instructions. A
CPU-only/device-free extraction of the SAME pinned AOT reference now handles
its1429 concatenated ELF images and records selected-family static metadata.
Default decode-attention INT8/GDN images are SIMD32/128GRFs; the first SIMD16
images belong to opt-in XMX and are not evidence that the reference selected it.
Exact receipts, source identities, limitations and next extraction recipe:
[completed checkpoint](kuno-native-compute-intervals.md#completed-gdn-experiment-and-current-performance-checkpoint).

Independent18:12:57UTC recovery: Intel50,688,000B, NVIDIA compute0, faults0,
services200 and normal inactive/unpaused admission/waiting0. No GPU job remains.
Installed worker/gateway SHAeac78ce0/d3ed07f4 are unchanged. The bounded owned
sudo refresh64142 lasts until18:59UTC. Goal active/unmet. No API, KNT2/KNR2,
configuration, model identity or Vashti consumer change. No frontend, version,
package or release. Earlier checkpoints below retain chronological context.

**Latest completed Kuno development (2026-10-20 16:53UTC; installed native32 unchanged).**

Full capture/source candidate`bfbfc0c3d51d39409e364530fef44ad65d93495d`, worker
SHA`ac01855f2dc9002208f81b88533a364657594d4bd4d6e1e9e7c94b1113baa90d`,
build`strata-parity-capture-inventory-20261020`, passes80 actual App CPU,
408/208 offline layouts, all129 original numerical GPU controls, complete real
requests/all old and new producer proofs, graceful exit and physical recovery.
Repeated profile11.7815s versus preceding11.8305s is unchanged; no clean,
full staged/terminal/sustained acceptance or install claim. Installed clean
11.1474s and matched Strata5.3515s still lead. This bundle stays unselected.

Actual204,463 cached nodes,8,968,280 cached views,21,782 guarded-root checks;
52,676 source-reading nodes share96 full bundle instances within real batch
captures. Next is [factoring full source declarations end to end](kuno-native-compute-intervals.md#next-coherent-source-proposal-factor-complete-source-declarations-end-to-end),
not another isolated lookup change. Keep every full declaration, original
signature/replay equivalence, complete write-proof/queue/owner/epoch/budget/
guard/erasure behavior. Private scratch can alias public weight arenas; never
infer immutability from a weight label. No omission of external footprints or
arbitrary removal of read_dependencies. This next representation is a proposal,
NOT implemented. Build capture indexes after replay misses, not on decode hits.

The complete canonical-declaration successor5b173ad8 / SHA
cc0aa6aa6b2694c3474138090c2437692213658a482e9c77eabb9b418f00b47a
passes86 App CPU,408/208 layouts, all129 original GPU controls, full real
requests/all actual producer proofs, graceful exit and physical recovery.
Repeated profile11.1239s (capture1.3818s versus prior2.0168s); clean
repeat10.7871s versus installed11.1474s is a modest3.2% difference. Actual
96 full canonical lists,4,860 certified-root instances and2,675,866 factored
source declarations preserve complete external footprints. Cached view work
8,968,280 ->3,616,548 with unchanged204,463 nodes/21,782 root revalidations.
No full staged/terminal/sustained acceptance or installation claim.
Receipts `strata-parity-canonical-source-20261020-{operators,profile,clean}`
and owned summary `strata-parity-canonical-source-public-summary-20261020.json`.
Profile physical recovery17:35:39UTC; clean recovery marker is present.

Complete enqueue candidate3b35ea10/SHA23747779 passes90 CPU,408/208 layouts,
all129 original GPU controls, full requests/all producer proofs, graceful exit
and actual recovery17:49:54UTC. Repeated profile11.0527s versus11.1239s is
unchanged. Actual204,463 unscoped admissions keep original full capture;
protected-scope admissions retain every original range/write/flight check.
No clean/staged/terminal/sustained acceptance or install claim.

The next complete Kuno-only candidate selects SIMD16 for the subgroup-independent
prompt GDN recurrence, with identical math/state mapping/causal loop/workgroup
barriers and checks. Both actual ordinary and retained native pipelines inspect
complete emitted SPIR-V and retain physical device limits. Unknown proof or
other kernels retain their original policy. Original scalar numerical references
remain unchanged. All94 actual App CPU controls pass, including real compiled
source equivalence and unsupported metadata/geometry/barrier cases. Corrected frozen source022d799b / SHAe3ba33c0
(`strata-parity-gdn-subgroup16-v2-20261020`) passes408/208 offline layouts,
including both actual named entries through unchanged spirv-val after existing
HAL normalization. Original129/full request/actual compiler ISA/recovery now
run in `kuno-native-gdn-subgroup16-connected-20261020`.
No performance or shader-specific spill attribution is claimed yet.
Exact contract: [subgroup-independent GDN](kuno-native-compute-intervals.md#subgroup-independent-gdn-register-pressure-candidate).
No API, KNT2/KNR2, configuration, model identity or Vashti change.

This candidate retains ALL preceding wide/bulk/miss-ring/loading/attention/
scoped-buffer/interval/OR/sealed-mask/global-barrier/push-descriptor components.
It reuses the SAME complete original signature allocation, revalidates every
actual root under EACH real guard, retains all original full views/access flags/
queue selection/INFO/owners/fences/erasure and64MiB, and shares only identical
ordered complete source lists inside one batch. Optional snapshot evidence
unavailable retains the old lookup path; source-cache overflow materializes the
full original views. Original24MiB host cap and8MiB tape remain. The actual
fixture charges18.92MiB; worst-case tape/node budget plus miss-ring metadata
fits. Actual new counters prove reuse, full root revalidation and real sharing.
Exact contract, CPU controls, limits and next gates:
[connected capture/source reuse](kuno-native-compute-intervals.md#connected-original-signature-inventory-and-full-source-bundles).

The latest COMPLETED preceding build9dc5786a/SHA85b9461b passes73 CPU,
408/208 layouts, all129 original numerical GPU controls, complete real requests,
actual164,182 pushed and zero fresh expert descriptor dispatches, graceful exit
and physical recovery. Repeat profile11.8305s is unchanged from11.9077s for
sealed/global065815af; installed clean11.1474s and same-host Strata5.3515s
remain faster. Neither development candidate is selected or installed; no
clean/staged/terminal/sustained acceptance is claimed for either.

Its protected DSO-aware CPU diagnostic passes, all726,548 raw bytes removed,
actual recovery16:23:02UTC. Repeated prompt samples3.4824s user CPU, distributed
across memset0.4623s, Planner0.2563s, full-view fill0.1407s and allocation/hash/
guard work. Leaf samples do not identify callers or promise equivalent wall
savings. Exact source/artifact/receipt identities, results and retained earlier
connected experiments are in [the synchronization guide](kuno-native-compute-intervals.md).

Installed worker/gateway SHAeac78ce0/d3ed07f4 are unchanged. Completed recovery
16:53:49UTC: Intel50,688,000B, NVIDIA compute0, all faults0, endpoints200 and
normal inactive/unpaused admission/waiting0. Independent17:01:14UTC check
reconfirmed those values and installed hashes. No test job remains active;
verify live state before starting another. A new bounded task-owned sudo refresh
is64142, parent3556, started16:59:33UTC and bounded until18:59UTC
(PID file `/media/jmicko/faststore/kuno-development/sudo-refresh-capture-inventory-20261020.pid`); the verified old6524 loop is retired. Never kill the shared shell.
No API, KNT2/KNR2, configuration, model identity or Vashti change. No frontend,
package, version or release. The parity goal stays active and unmet. Earlier
records below retain chronological context.

The complete bulk/PV/loading source12caac8b / SHAa2ad0c62 now passes402/144
layouts, all103 GPU controls, three full public requests and a separate clean
45/4245/4245 comparison, with physical recovery and healthy services. Three
integration repairs preserve genuine fresh/reuse owners, original release
metadata and original INFO page planning. Actual96 bulk batches,158,028
launches and39,225 HOST copies match the original26,338 six-operation groups.
Load82.18s improves from roughly101s, but clean repeated prefix12.4769s is
slower than installed11.1474s. The complete bundle is kept, **unselected and
uninstalled**; no new full staged/terminal acceptance is claimed for it.

The complete routed-wide/bulk/host-batching worker6e61895b / SHA763e39ec
now passes402/208 offline layouts, all129 actual GPU controls, real six-logical
phase/eight-piece serializer and budget controls, and three complete public
requests with genuine wide/tail batch counts and physical recovery. Profiled
long prefixes13.1426/12.3522s remain slower than installed11.1474s and upstream
5.3515s. This complete connected pipeline stays **unselected and uninstalled**;
no partial component rollback or new staged/terminal acceptance is claimed.

A separate protected leaf-CPU diagnostic passes with raw sampling deleted,
complete requests, healthy services, normal admission and physical recovery.
Repeated4245 prefix12.4560s samples3.7085s user CPU: allocation/zeroing,
Planner, Wgpu raw-buffer guards and temporary collection are distributed costs,
not proof of a single driver bottleneck. The next coherent candidate preserves
all wide/bulk/loading components, removes only fault-reporting reductions in
selected attention fault variants, and reuses capture scratch with deduplicated
Recording lifetime resource references. Its frozenfa2138fc / SHA802e3fa2
passes82 CPU controls,402/208 layouts,129 GPU controls and complete requests with
physical recovery; repeated profile12.0604s is a modest gain over12.3522s but
still behind installed11.1474s. It remains unselected and uninstalled.

The next connected candidate couples exact original COPY-view intervals with
synchronization already provided by genuine original ALL_COMMANDS peer waits.
It retains original queue order, releases, full views and ownership. Frozen
c12a2418 / SHA71847aba passes86 CPU controls,402/208 layouts, all129 GPU controls,
complete public requests/counts and physical recovery. It encodes40,281 COPY
nodes with zero per-copy barriers, but repeat12.0792s is unchanged from12.0604s:
this is not the main bottleneck. No installation or clean/staged/terminal
acceptance is claimed for either candidate. The next coherent source lead is
Strata's miss-only eight-slot staging ring and conditional streamed-slot release;
Kuno currently advances its slots for resident groups too. The exact proposal
preserves full BDA/source/loan guards and couples all required pieces.
The complete opt-in miss-ring source is now implemented: actual effective-role
classification, eight misses ahead, genuine disjoint resident raw view and
role-bound resident GU/DOWN metadata plus matched closed grammar.101 App CPU
and seven actual-runtime-source controls pass. Frozen d95a12da / SHA07415e68
passes408/208 offline layouts, all129 original GPU controls, complete real
requests/counts and physical recovery. Profiled repeat11.5581s improves the full
bundle, but clean repeat11.2306s only equals installed11.1474s. It remains
unselected and uninstalled. DSO-aware CPU sampling completes with raw data
removed and physical recovery; repeated prompt samples3.2513s of distributed
CPU costs. The next coherent candidate is capture-local actual buffer lookup
under a single real Wgpu destruction guard per phase, preserving every original
view, INFO, owner, planner and recording lifetime. See the exact scoped guard
contract in [native buffer capture](kuno-native-buffer-capture.md). Frozen45795611 /
SHA8944b257 passes408/208 offline, all129 original GPU controls and complete
requests/counts/recovery, but repeat11.5435s is unchanged from11.5581s. It stays
unselected and uninstalled. The next connected implementation replaces false
same-queue whole-buffer hazards with COMPLETE original-view intervals; actual
queue Plan/waits/releases and owners stay. See
[the conservative synchronization contract](kuno-native-compute-intervals.md).
No new installation claim. See
[the exact implementation and gates](kuno-native-miss-ring.md).
All actual views/owner bindings, original raw-buffer checks and signatures stay.
See [the exact contract and current gates](kuno-native-capture-attention.md).
No API, protocol, model identity, configuration or Vashti consumer change.
Installed3fffc266 remains selected, and the Strata parity goal stays active.

**Prior Kuno-only checkpoint (2026-10-20; superseded by native32 above).**
See [the source decisions, exact receipts and limitations](kuno-strata-parity-goal.md#latest-combined-operator-checkpoint).
Selected worker source0771a726 / SHAccdaf886 is installed through unchanged
image-enabled gateway36626d30 / SHAd3ed07f4. The official transaction preserves
the entire configuration and root-enrolled identities. Backup/rollback:
`/var/lib/kuno-strata-parity-native-20261020` (previous workerfb31e124).

Matched clean4245-token repeated prefix **24.9658 ->12.4373s (50.2% shorter)**;
first long14.1743s. Fresh same-host unchanged MIT Strata takes5.3515s, so the
parity goal remains active. This selection connects complete QSA/GDN specialization,
genuine MAX8 headless known-prefix processing, original native matrix arithmetic,
rolling original-recording submission and borrowed immutable layer bindings.
No prompt truncation, toolbox reduction, quant substitution or retained prefix
cache. Original nine QSA bitwise gates remain strict; three added non-dyadic
cases independently validate both GPU paths against operation-bounded F64 math.

All97 actual GPU controls, complete clean requests, real injected-prefix terminal
recovery and subsequent fresh success pass. Protected staged thinking/ten-tool
call/result,64/65/66 boundaries, MAX8 terminal counts1..8 both directions, two
identical complete501-token code answers with155 semantic cases each, privacy,
unload/Ollama coexistence and actual physical recovery pass. Staged4207/4293-token
toolbox prefixes15.8214/14.7448s versus previous29.506/29.920s. Installed public
semantic repeats and actual DynamicUser/model-mount acceptance pass too: cold
load100.302s, first short prefix9.948s including JIT; warm35-token prefix1.164s,
501 outputs/15.289s (~32.77 output tokens/s). No decode gain is claimed. Worker
write_bytes0, swap/core0, denied inspection, zero OOM/new GPU faults, services200
and normal admission. Tests intentionally retain no request/response text.

Receipts under `/media/jmicko/faststore/kuno-development`:
`strata-parity-native-rolling-bundle-20261020-operators`,
`strata-parity-native-rolling-20261020-{terminal-negative,clean-long,staged-api}`,
`native-strata-parity-installed-{public,final}-check-20261020`.
Experimental library candidates remain separate: Tilewise881d5af3 passes111
controls but regresses20.4503s; balanced M64/Doublec5932d91 passes111 controls
and complete requests but regresses33.9067s. Neither is installed. A wider
library pipeline43281fd6 passes117 GPU controls but regresses28.7764s and remains
unselected. Original-native wide9cf2236b passes103 controls and complete prefixes
12.8885/12.6279s in profiling; it is uninstalled and not a clean A/B. Actual
compiler ISA revealed hot-loop register spills. Complete32-subgroup successor
3fffc266 / SHAeac78ce0 removes them and passes103 GPU controls, full public
requests/recovery and402/144 offline shader layouts; profiled long prefixes
11.7837/11.8168s. Full clean/private staged acceptance passes too: matched
clean first/repeat12.9623/11.1474s, all tools, MAX8 tails, semantic repeats,
privacy and physical recovery. It remains uninstalled while combined candidate
a3b36795 / SHAd7965806 (native32 + paired-cell QK + descriptor batches) runs
original GPU controls and full profiling. Its402/144 offline layouts pass;
no combined GPU speed claim yet. See the parity guide
for exact artifact receipts and capacity/ISA evidence.

**No API, KNT2/KNR2, configuration, model name or Vashti consumer change.**
Original DASLab IQ3_S plus Q2 MTP, context512..32768, thinking/tools, private
content erasure and ordinary Ollama compatibility remain. No fresh Vashti
browser/image QA, version bump, package or release is claimed. Backend-only
internal work needs no frontend integration handoff. Root alone owns shared
services/GPUs; inference may be temporarily paused for the active protected
benchmarks. The user-authorized bounded sudo refresh remains active during work.

**Prior accepted Kuno-only Strata prompt optimization (2026-10-20; superseded above).**
Selected worker sourcec8db19de / SHAfb31e124 is installed through the unchanged
image-enabled gateway36626d30 / SHAd3ed07f4. The official transaction preserves
the complete configuration and root-enrolled model identities. Backup/rollback:
`/var/lib/kuno-prefill-selected-short-quality-20261019`.

Matched clean4245-token warm prompt **43.2055 ->24.9658s (42.2% shorter)**.
Protected staged ten-tool thinking/call/result passes;4207/4293-token prefixes
29.506/29.920s versus previous46.991/49.630s. Short35-token code prefix1.290s
versus previous4.639s. The new quantized short path gives a different, complete
501-token answer: both repeats are identical and the function passes155 bounded
merge cases in a restricted, unprivileged child. Do not compare its full response
time directly to the previous362-token answer or relax the GPU math references.

All53 actual GPU matrix/router/dequant controls,402 ordinary/64 matrix shader
layouts, cold short plus two long requests, threshold64/65/66 requests, same-worker
mixed/repeated API use, write_bytes0, inspection denial, graceful unload, ordinary
Ollama and actual physical recovery pass. No driver faults or reboot. Final
installed public requests and DynamicUser/model-mount metadata pass too. Cold
load105.872s; warm35-token prefix1.4305s,501 outputs/15.2196s (32.92 output tokens/s,
not a new decode gain), total21.3211s. Swap/core0, write_bytes0, inspection denied,
zero OOM/driver faults, all services200 and normal admission. Intel weights remain
warm (33,314,902,016B); NVIDIA compute0. No test client/staged service or task sudo
refresh loop remains. Exact installed receipts:
`native-prefill-selected-short-quality-installed-public-check-20261019` and
`native-prefill-selected-short-quality-installed-final-check-20261019` under the
artifact base in the investigation. See
[complete source decisions, receipts and limitations](kuno-prefill-strata-gap-investigation.md)
and [service acceptance](kuno-native-text-service.md#native-strata-prompt-optimization-2026-10-20).

Contract/consumer requirements: **no API, KNT2/KNR2, setting, model name or Vashti
change**. Original DASLab IQ3_S plus Q2 MTP stay on Intel, context512..32768,
thinking/tools and ordinary Ollama compatibility unchanged. No shortened toolbox,
retained prompt/KV cache or persisted content. Startup remains about106s plus
initial warmup; no decode speedup or full Strata speed parity is claimed. Short
MAX3/64 still copies bounded wiping residuals and computes heads; full MAX8,
no-head/device carry and further GEMM work remain future optimizations. Failed
register-first successor is retained in history but not selected. No fresh Vashti
browser/image suite, package/version change or release is claimed/authorized.

**Prior CPU preparation checkpoint (superseded by GPU continuation above).**
The user authorized investigation and implementation preparation while away and
without sudo. See [full source comparison, candidate and protected next steps](kuno-prefill-strata-gap-investigation.md).
Source5fbc3ca5 combines genuine missing token-level prompt router parallelism,
eight-value IQ weight decoding/vector stores, four-accumulator macro matrix tiles,
and the required bounded SIMD16 multi-subgroup contract repair. The earlierd06c03d4
failure is not established numerical divergence: native subgroup selection rejected
its128-thread cooperative geometry. New actual-module CPU coverage catches this.

65 selected CPU cases pass (10 matrix,29 prompt,5 router,13 engine,8 HAL selector);
strict selected production worker Clippy passes. Broader all-test Clippy still has
unrelated existing fixture/style warnings and is not claimed passing. Frozen release source5fbc3ca5 / SHA b45446bc builds; full402 ordinary/64 matrix
offline validation passes. Artifact: `strata-prefill-coupled-investigation-20261019`.
Exact hashes, failed broader-lint scope and receipt paths are in the investigation. **No new GPU
numerical or speed result, installation, service interruption or Vashti/API change.**
Installed worker remains60c20e36 / SHA1db4664d with gateway36626d30 / SHAd3ed07f4.
Actual hashes are unchanged; Kuno/Ollama/Vashti read-only probes return200.
The temporary isolated review worktrees have been queued for archive; selected
commits are integrated into this backend checkout.

Next needs available GPUs and refreshed sudo for protected numerical, complete
request/erasure/recovery and matched short/long prompt tests. Do not install merely
because the CPU/offline gates pass. Generic decode uses the existing small path;
these prompt changes do not yet claim faster token generation. CubeKv0.2 is a
researched Rust GEMM alternative, not an added dependency or selected backend.


**Prior Kuno-only prompt-processing optimization (2026-10-19; superseded above).**
Worker source60c20e36 / SHA1db4664d is installed with unchanged image-enabled
gateway36626d30 / SHAd3ed07f4. Forward source selectiona37db7da restores the exact
accepted matrix implementation after an unsuccessful experimental variant.
See [complete measurements, changes and receipt paths](kuno-prompt-processing-profile.md).
The matched compiled-public4,245-token warm prompt falls55.821 ->43.205s (22.6%
shorter); this is not a universal rate or a replay of the user's private chat.
Actual FP16/BF16 original-weight checks,401 shader/32 matrix validation,27 prompt
and4 original-scheduler tests pass; private staged ten-tool call/result, repeated
code, zero disk writes, unload, Ollama coexistence and physical recovery pass.
The official transaction preserves the complete configuration. Backup:
`/var/lib/kuno-prompt-processing-20261019`. Final installed repeated-code and identity/mount checks pass, swap/core0, zero
worker writes/OOM/driver faults, all services200 and normal admission. Intel weights
remain warm (33,314,902,016B total device usage); NVIDIA compute0. Warm installed
code prefix4.639s/decode10.899s for362 outputs (33.21 output tokens/s). Backend API
acceptance only: no fresh Vashti browser/image suite or release/version change.
The old toolbox browser acceptance below remains historical evidence. No test
client/staged service or sudo refresh loop is left active. No release is authorized
by this handoff.

Contract/consumer requirements: **no API, private-wire or Vashti change**. Full
systems/tools/history are still processed independently and request state erased;
model weights remain warm by the existing policy. No shortened tools, truncated
context or retained prompt cache. The user's Vashti-owned portable prefix-cache
idea is recorded as a [future shared proposal](kuno-prompt-processing-profile.md#future-proposal-vashti-owned-portable-prefix-snapshots),
not an implemented API. Compact selected-engine estimates are about133MB for1K,
180MB for4K and243MB for8K tokens before transport overhead; exact export state,
versioning, validation and encrypted Vashti handling need a later agreed contract.
No Vashti connector work is requested in this internal optimization checkpoint.

**Active Kuno-only toolbox repair (2026-10-19, installed and Vashti-validated).**
The user explicitly requested the fix in Kuno, preserving Vashti behavior. The
failure is reproduced by compiled public text without tools: native prompt chunks
support4096 tokens, but `ops_ple_transition::token_pair` still rejected more than
2048 after GPU work. Public-only diagnosis `e8dbea20`/`a3894733` identifies that
exact checked-PLE error. Fix `a6b21a38` shares one4096-token constant between core
history validation and GPU prompt allocation; MAX8 decode, state/fence/erasure
checks and32K context capacity are unchanged. No consumer/API change.

146 library tests pass,2 ignored;10 checked-PLE tests pass; strict library/tests
and selected worker Clippy pass. The unchanged compiled-public8,445-token fixture
now completes across three chunks, returns the correct arithmetic answer, erases
request state, closes, and restores physical memory/admission with zero driver
faults. Frozen worker sourcea6b21a38 / SHA7bc1633b. Protected staged10-tool
API acceptance passes preload ->4,207-token streamed tool call -> tool result ->
final answer on the same worker, write_bytes0, graceful unload, subsequent Ollama
and full physical recovery. The official transaction installs this worker with
the unchanged image-enabled gateway36626d30 / SHAd3ed07f4 and preserves the
complete previous configuration. Root backup: `/var/lib/kuno-tool-prefill-fix-20261019`.

Actual live Vashti now passes the SAME previously failing prompt with Notes,
Calculator and Images ALL enabled, low thinking, actual calculator call/result
and final answer (test response version5/5). No Vashti source or global user
setting was changed; no existing conversation was opened. Screenshot:
`/tmp/kuno-vashti-all-tools-fixed-20261019.png`. Installed nonroot/model-mount
checks pass, swap/core0, worker write_bytes0, denied unprivileged inspection,
no OOM or driver faults; all services200 and admission normal. Native weights
intentionally remain warm on Intel (33,348,456,448 bytes); NVIDIA compute0.

Contract/consumer requirements: **no API, private-wire or Vashti changes needed**.
Tool definitions are not removed or shortened. Context remains512..32768; above32K
and further loader/speed work remain separate. No release/version bump, new
throughput benchmark or fresh image/individual-tool suite is claimed. Tested
backend and application receipts are in the [service specification](kuno-native-text-service.md#kuno-toolbox-prefill-repair-2026-10-19).

**Historical installed feature handoff before the toolbox repair (2026-10-19 UTC).**
The local LAN listener exposes `kuno-qwen3.8-flash-next:iq3_s` with completion,
thinking and tools. Source `36626d30`; image-enabled gateway SHA `d3ed07f4`, native
worker SHA `f871238b`. Protected staged and actual DynamicUser/mount acceptance
pass real thinking, tool-call/result round trips, streamed output, CLI thinking,
context-overflow400 followed by same-worker reuse, zero worker disk writes,
graceful native unload/reload, ordinary Ollama inference and physical recovery.
234 CPU cases pass;7 hardware cases ignored; strict gateway/worker Clippy passes.
Exact hashes, contracts and receipts are in the
[native service specification](kuno-native-text-service.md#protected-extension-acceptance-and-installation-2026-10-19-utc).

Cold load is102.813/102.315s staged and101.981s installed, versus historical223.333s
(about54% shorter). Full hashing moves to trusted administrator enrollment and
is repeated when artifact identities change; no request-content cache is added.
This is startup improvement, not a new throughput claim. Historical warm API
fixture rates remain33.6602/36.1275 eval tokens/s (17.4990/19.7720 full request).

Usage: `kuno run kuno-qwen3.8-flash-next:iq3_s --think=low`. Thinking defaults off;
API `think` accepts false/true or low/medium/high/xhigh. Standard Ollama
`message.thinking` and `message.tool_calls[].function.{name,arguments,index}`
carry results; the caller executes tools and sends role=tool history.
`num_ctx` accepts512..32768 as an enforced request budget, without truncation.
The allocation is STILL32K; above32K remains a substantial open engine change,
with the [connected capacity plan](kuno-native-text-service.md#larger-context-implementation-dependencies).
Unsupported options/formats/images/OpenAI inference still fail explicitly.

Gateway and worker use private KNT2/KNR2 together. Do not install a gateway-only
upgrade against an old KNT1 worker; enroll the new pair in one new-gateway
`configure-native-text` transaction. This installation did so; exact existing
config/ports/image/managed settings are preserved. Root-only backup:
`/var/lib/kuno-native-features-20261019`. Disable via normal confirmed
`sudo kuno configure-native-text --disable` if needed. The native worker retains weights on successful calls; competing inference
evicts them and `kuno stop MODEL` explicitly unloads. After the application
checks below it is unloaded, with both GPUs free for compute.

Vashti handoff: existing capability discovery sees thinking/tools without a code
change. Live browser test in a fresh private chat completed reasoning -> calculator
call -> tool result -> final answer with only calculator enabled. The initial
request with all default tools returned503; two retained attempts and one clean
start reproduce it. Do NOT claim full default-configuration application acceptance.
The cause is unproven; next use a compiled public longer/tool prompt and bounded
static error stages, preserving fault/erasure checks. UI calculator stats show
2,006 context tokens and5.1 eval tokens/s on its cold two-round workload, not the
historical short-code rate. Final services/admission are healthy, both GPUs free
for compute, zero driver faults and no OOM events; native worker is unloaded.
No existing chat was opened. Other connector/UI changes remain in the Vashti task;
clear unsupported penalties/num_gpu/num_thread explicitly and refresh capabilities.
No protocol crate edit, fresh image generation, release/version bump or upload.
Work stays in `codex/strata-rust-port`; canonical frontend checkout is untouched.

Earlier checkpoints below are historical and superseded only within this
installed extension's explicitly accepted scope. The remaining measured kernel
target is coupled quantized expert gate/up+down (51.95% of sampled target GPU time),
after the requested larger-context and loader work; no kernel changes this block.

**Selected Intel engine checkpoint: 33.4 tokens/s warm (2026-10-18).**
The connected Rust rewrite has passed the documented selected-engine audit.
Frozen runtime dfe2e6f7 / SHA4cd6d4f7 (strata-adaptive-four-20261019), built with
--intel-core --gpu-adaptive-usage, measures23.9617 tokens/s first request and
33.3707 on the warm repeat:362 identical outputs each,4-input speculation,
4887 fully published expert swaps. Warm generation10.8478s; full request with
prefix16.0885s (22.5005 tokens/s), before erasure/close. One fixed-public code
benchmark, not universal throughput or a new Ollama comparison.

Validation:130 numerical cases including56 bit-exact expert controls on the
unchanged math;37 global/cache/ownership tests with adaptation enabled; actual
head-accounting boundary test;401 shaders/32 matrices; strict relevant Clippy;
retained code repeat, and a separate512-output/no-EOS prose/cap run with3247
published swaps. Both actual target/draft erasure and physical recovery pass.
Four remains default; eight-input trial passes but is slower (18.9600 warm).
Both GPUs are idle for compute; all services/admission normal; no driver faults.

See [final audit and exact receipts](kuno-strata-current-source-audit.md#final-selected-engine-checkpoint-2026-10-18-local).
Selected rewrite/measurement milestone complete; native service/private-caller,
mixed-context/soak and release acceptance remain separate work. The fast build
is a development harness, NOT the installed backend. No API/frontend/install
change. Current worktree is codex/strata-rust-port; canonical frontend checkout
is untouched. Earlier entries below are historical controls and pending gates
superseded only within the explicitly accepted scope.

**Selected Intel rewrite accepted for fixed-public engine measurements (2026-10-18).**
The connected Rust port is now past the source-wiring, representative numerical,
whole-model completion and physical recovery gates for the selected B70 / original
DASLab IQ3_S / original Q2 MTP configuration. See the
[completion decision](kuno-strata-current-source-audit.md#selected-engine-completeness-and-correctness-decision-2026-10-18)
for exact scope and exclusions. This is not private-caller, service, installation
or release acceptance.

Runtime source `299579b52f9e4b9d8feaaaedb41eec4c193fcb83`, frozen artifact
`strata-fp16-preserve-coupled-20261018`, SHA256
`852a0dbd7dbdd11a3a1eb8e3447a9006dbe7fae2b56d8d3ecd1fb3ccd7c473e1`.
`f0707ddb` requests supported FP16 DenormPreserve16; `299579b5` applies the
same transformation to the actual native replay module as the HAL module.
All74 original-weight operator cases now pass the ORIGINAL IEEE reference and
unchanged tolerances, including FP16/BF16 counts1/8/35. Maximum matrix error is
4.2915344e-6. The old29-mismatch/FTZ diagnosis remains failed historical evidence;
it is not used as the new reference. The aligned embedding fix remains exact.

Protected `strata-fp16-preserve-retained-20261018` passes two requests through
one loaded Session: identical362 outputs/EOS,110 windows and31 real repairs per
reported second request. Protected `strata-fp16-preserve-prose-20261018` passes
512 outputs/no EOS,194 windows and83 repairs. Both display coherent public prose
without saving it; target/draft request erasure, final all-peer close and physical
recovery pass. Final Intel50,688,000B, NVIDIA compute0, all HTTP200, normal admission
and zero classified driver-fault deltas. Swap/core protection was observed active.
633 selected CPU cases/88 suites, core/off strict binary Clippy and pinned offline
397 shader variants/32 matrix configurations pass. Artifacts are under
`/media/jmicko/faststore/kuno-development`.

The speed gate is now open ONLY for the exact accepted artifact and unchanged
fixed-public retained configuration. The committed acceptance manifest is checked
by the outer timing runner. Timing uses trusted static phase lines and writes
only numeric results after successful repeat/erasure/physical recovery; no kernel
profiles, response contents or content hashes. The first accepted trial
`strata-accepted-retained-throughput-20261018` completes successfully:19.7811 and
20.6089 delivered tokens/s for362 outputs on each request, with17.5652s warm
generation and5.2670s warm prefix/preflight. Full warm request before erasure is
22.8322s (15.8548 tokens/s); selected runtime setup is101.1493s. These are
last-prompt-inclusive engine rates, not pure GPU decode. Final physical recovery
again passes with the same clean baseline. The30 tokens/s target is not met.
The original execution-check receipts retain their static `benchmark=false` and
`whole_rewrite_accepted=false` fields; the documented audit and separate timing
receipt provide the later decision without rewriting historical evidence.
No Vashti consumer/API change, installation, reset, reboot or release occurred.

Actual selected-path operator profiling now passes on source`f38629a9` after a
strict rejected first attempt. `strata-selected-operator-coupled-check-20261018`
completes362 outputs/EOS,110 windows/31 repairs, full target/draft erase/close and
normal physical recovery. One serialized target suffix sample totals126.456ms:
Down32.163ms, Gate26.715ms, Up26.589ms (67.59% combined). BF16 matrix work is only
2.996ms. This is operator diagnosis, not a new throughput rate. Baseline stays
19.7811/20.6089 tokens/s on frozen299579b5;30 remains unmet. Next substantive GPU
candidate: remove the selected expert kernels' extra full-window activation SLM
staging and read the original already-quantized input planes directly, as upstream
row-entry dot products do. Preserve 4/2/1 chunks, real source pointers, exact
integer finishing/rounding, reduction order, target verification and all scopes.
See the current source audit's latest operator receipt before changing kernels.

The follow-up counter-only candidate `b7cb8e85`/SHA256
`aa12c4732d21b269beac5745ba876a6df6ca0034607daa9e6a8f93299e00bb9e`
also passes the two-request362-token repeat/erasure/physical recovery check.
It only exposes existing aggregate graph counters at three static public phases;
no inference math, placement, queue work or timestamp-query changes. Warm generation
records15.8413s native fence wait,0.1704s capture,2077 submissions/1964 replays.
This is a diagnostic, not a new throughput rate or replacement for the exact
accepted artifact. Next: profile actual GPU kernels in the selected controller;
CPU command capture is not the leading candidate. Both test units are terminal,
admission/services/memory recovered and shared-shell sudo was refreshed.

**Previous execution and numerical checkpoint (2026-10-18):** isolated
`codex/strata-rust-port`. The human paused the goal and requested a final
checkpoint before updating ChatGPT. This block is finished; no GPU test remains
running. No reset/reboot/install/API/frontend changes.

The complete selected engine now returns through generation, target AND draft
retained request erasure, final all-peer close and physical recovery. Protected
`strata-cleanup-window-whole-20261018` exits0 and writes `intel.json` with both
`*_owners_erased=true`. Runtime source isd271eb49, frozen at snapshot
`7233f75fb9f075541703e19fff94b5b7d23d0017`, SHA256
`9465232e773ea1ed5cb65b0d947ab537151b6c8f4fea45282d91572e66ec4dff`.
102 selected CPU cases, strict core/off binary Clippy and independent frozen
mode/source/features/ordinary-equality checks pass. The original full-owner
bounded scan fixes the target erasure failure; no checks or tails are omitted.
After-state: Intel50,688,000B, NVIDIA compute0, all HTTP200, normal admission,
zero driver fault deltas. Live protection receipt verifies swap0/core0 and the
original48.2GiB cgroup cap.

**Numerical acceptance FAILED:** the public code request emits11 outputs then
EOS, but the unsaved preview loops through special tokens rather than answering.
A successful lifecycle receipt does NOT establish correct inference, retained
repeat, whole rewrite acceptance or speed. Both speed/whole-rewrite acceptance
fields remainfalse. Retained-repeat validation is deferred behind this concrete
numerical failure; do not use repeated nonsense as quality proof.

Diagnostic source `72f0b24a` connects the actual selected `strata_gr` kernels to
the existing original-weight CPU operator oracle via the fixed public
`gr-batch-probe` only. Counts1/4/8 compare normalization/down/low/mix/injection/
combined outputs and full inactive padding with the existing tolerances. No
production math or API changes. Its CPU comparison test and strict core binary
Clippy pass. Both protected diagnostic attempts are terminal and stopped before
numerical comparison: the first rejected its dispatch configuration; `4a712a0e`
fixes only that explicit fixed-probe admission. The second created the actual
B70 device, then rejected the GGUF open. No numerical receipt was written.

Latest frozen diagnostic `strata-gr-native-dispatch-20261018` is source
`4a712a0e719deeb589ad8d4b041b4149785d6d85`, SHA256
`c905e937b3240c527ba8a65d36dc6c247ad8b21ce5ace9203ae217d3d9c269a3`.
Protected output `strata-gr-native-dispatch-check-20261018/host-after.json`
confirms normal admission, all three HTTP200, Intel50,688,000B, NVIDIA compute0
and unchanged zero classified driver faults. The unit is inactive with PID0.
Strict Clippy log `strata-gr-native-dispatch-core-clippy-20261018.log` finished
successfully. Artifact paths are under `/media/jmicko/faststore/kuno-development`.

Next inspect the original typed GGUF-open error and connect the diagnostic to
the engine's original split-aware loading/limits; the probe currently uses the
old single-file `GgufFile::open` path on the first shard. This is a loader
hypothesis, not a demonstrated numerical cause. Do not duplicate or merge the
model merely to run the probe. Resume numerical checks after the human returns,
not with a throughput benchmark. See the
[device execution receipt](kuno-strata-rewrite-completion.md#protected-construction-and-first-prompt-execution-2026-10-18).

## Ownership

| Area | Default owner |
| --- | --- |
| `apps/kuno`, `crates/kuno-image`, `crates/kuno-text` | Kuno task: inference, resource management, runtime, installation |
| `apps/vashti/src`, including `images` and the Ollama connector | Vashti task: authorization, persistence, application API, Kuno client |
| `apps/vashti/web`, `apps/vashti-android` | Vashti task: UI, mobile workflows, native integration |
| `crates/kuno-protocol`, workspace dependencies, cross-app specifications | Coordinate changes and identify affected consumers |

The user can explicitly reassign work. This division should not prevent a small,
agreed integration fix, but Kuno work should not independently redesign Vashti's
UI or change its user-data storage.

## Source of Truth

- [Kuno specification](kuno-spec.md): privacy, compatibility, admission, lifecycle.
- [Kuno development guide](../apps/kuno/README.md): operation and validation.
- [Native text research proposal](kuno-text-engine-research.md): pinned FreeToken
  and Strata findings and proposed Qwen3.8 engine phases; no implementation claim.
- [Native text implementation/benchmarks](kuno-text-performance.md): fixed
  synthetic updated-Ollama baseline and Rust engine development receipts.
- [Native causal prefill plan](kuno-native-text-prefill.md): internal MAX8
  transaction, recurrent/attention boundaries and bounded expert waves;
  isolated operators do not establish whole-model batching.
- [Actual native model connection plan](kuno-native-text-model-batching.md):
  approved complete constructor, original model owners and all-family chunk
  acceptance; source adapters do not establish a connected device consumer.
- [Native decode optimization plan](kuno-native-text-decode.md): original packed
  Rust CPU experts, positional reads and later measured CPU/mixed-GPU trials.
- [CPU/GPU calibration lab](kuno-native-text-cpu-gpu-calibration.md): approved
  bounded real-read operator, original-rank merge and completed joint controls;
  source/device/timing acceptance remain separate.
- [Current Strata source audit](kuno-strata-current-source-audit.md): selected
  call-chain review, genuine connection defects and remaining source/device gates.
- [Native text checkpoint](kuno-native-text-checkpoint.md): current
  integrated source, isolated diagnostic candidate, completed checks and the
  next physical acceptance gates. Resume from this checkpoint first.
- [Native disk-access research](kuno-native-text-disk-access.md): intended lazy
  n-gram lookup, actual source paging and the accepted long public diagnostic;
  aggregate disk reads alone do not establish cache thrashing.
- [Fully resident Ornith speed lab](kuno-native-ornith.md): user-selected
  Qwen3.5 MoE diagnostic before further Flash-Next offload work; no API change.
- [Image engine contract](kuno-images.md): supported execution and image API.
- [Vashti Images specification](vashti-images-spec.md): product behavior and UX.

Specifications describe intended behavior; source and tests establish what is
implemented. Clearly label proposals, implemented behavior, and unverified work.
When an approved change supersedes an older requirement, update the relevant
normative section and mark historical notes as superseded rather than leaving
contradictory instructions. Keep this guide focused on the current handoff, not
an ever-growing copy of the specifications.

## Workflow

1. On resume, read this guide, inspect `git status`, and review relevant commits
   since the last handoff. Preserve unrelated work. Do not start from `main` if
   the agreed integration work lives on another branch.
2. Before a cross-app change, record the agreed behavior and contract below:
   routes, methods, request/response fields, authentication, errors, cancellation,
   capability discovery, and old-client/old-server behavior as applicable.
   Resolve unsettled user-facing or security decisions before implementing them.
3. Implement and test the owning application's part. Update the appropriate
   specification and record the backend receipt below. Internal backend changes
   without consumer impact can proceed without this cross-app process.
4. The Vashti task reviews the actual diff, implements the connector and UI
   changes, and tests the full flow on mobile and desktop. Avoid adding UI for
   states or capabilities the backend does not actually report.
5. Mark integration complete only after both receipts exist. Record exact tests
   run, remaining limitations, and whether installed services were tested. Unit
   tests or a build alone do not establish live GPU or mobile behavior.

For sequential work, the shared checkout is sufficient. For concurrent work,
use separate managed worktrees from the agreed current integration commit and
explicitly exchange commits. Coordinate service restarts, installations, model
loads, and GPU tests even with separate worktrees: they still share the host.
Do not auto-message another task or deploy as part of a handoff unless the user
authorizes it.

## Existing Constraints

- Kuno executes transient requests; Vashti owns authorized user-content storage.
  Do not persist inference prompts, outputs, images, or previews in Kuno.
- Keep credentials out of browser responses and handoffs. Kuno diagnostics must
  remain content-free; do not log raw inference or backend error payloads.
- Preserve Ollama-compatible long inference. Do not add arbitrary generation
  deadlines to fix connection or readiness problems; queue/load limits are
  separate concerns.
- Preserve the approved image behavior: each automatic prompt improvement
  starts from the original submitted prompt, not the previous improvement.
  Retry uses the recorded recipe. See the Images specification for details.
- Never edit a previously deployed SQL migration to change its behavior; add a
  forward migration if the integration needs stored-setting changes.

## Active Handoff: Image Access Without Manual Key Setup

**Latest application acceptance:** see the beta.3 preparation/publication
receipts at the end of this guide. Kuno's handoff is merged into the normal
project checkout; all four beta.3 artifacts are published and
full-download/signature verified. Live Hub
and nginx upload limits are now verified at 1 GiB. Physical Android acceptance
is still outstanding.

**New Kuno backend work:** the latest owner handoff is on
`codex/kuno-image-speed` through `ebd1925`, now merged into this checkout.
See [the release handoff](kuno-release-handoff.md).
Read its **Selected Intel performance update (2026-10-02)** before the next
release. It supersedes the earlier `fb2d1d0` residency artifact selection.
Public contracts are unchanged; package the selected standard gateway and
managed ABI 2 runtime together. Integrate the entire backend range after shared
base `4f9b465`, not only the optimization commit. Preserve/commit pending Vashti
work before merging.

**Model-only image residency:** see the resident-image receipt
below. Source, CUDA and installed Intel/runtime acceptance pass; the new
gateway/runtime is installed on this host. This backend-only change needs no
Vashti connector or UI change. Do not
treat the published ABI 1 beta.2 runtime as the new resident worker.

**Selected image performance follow-up (2026-10-02):** the user authorized
installation of the tested Intel closure and release preparation. Whole-RoPE
plus the residual-first kernel is selected from production `230d15b`; public
package pins/build selection are `5351f63`, strict builder validation is `bde8cd0`,
and the owner source/test integration is `05a95fc`. The exact tested flat runtime
is now installed
on this host through `install-image-runtime`. Its manifest is
`fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa`;
normal archive SHA256 is
`85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3`.

A counter-free, compiler-idle eight-request comparison passes full original
50-step generation/25-step editing on B70/kernel38: warm generation
132.893→106.523 s (19.84% faster), edit 72.769→61.831 s (15.03% faster).
Exact native tests, protected SDK-hidden full requests, previews/PNG/privacy,
resident PID reuse, normal shutdown, cancellation and fresh-worker recovery pass
with no new classified Xe fault. Installed keyless HTTP generation/edit pass,
including streamed previews and reuse of one retained worker. The gateway binary,
Ollama routing, inference deadlines and image API are unchanged; no Vashti
connector/UI changes are required.

See [the performance receipt](kuno-image-performance.md#whole-rope-combined-worker-and-full-workflow-comparison)
and [the release handoff](kuno-release-handoff.md). Historical ComfyUI times are
not a new matched benchmark. Pixel-level equivalence/quality tolerance and
indefinite stability are not established; identical-runtime edit variation is
also measured. The exact frozen runtime has been tested and can be packaged
normally. A future fresh managed CMake rebuild needs a reviewed MKL configuration
and acceptance of the resulting binary. Standard native inputs and a fresh
normal gateway build/check/Clippy pass, as do 40 exact CUDA and 40 exact Vulkan
math cases on the 3060. Four original-resolution two-step
CUDA generation/edit requests pass previews/PNG/privacy, resident PID reuse and
normal shutdown. Live Vulkan image acceptance is not claimed for this update; the shared engine patch affects both backends.

Source selection is `05a95fc`, acceptance receipts are `b48d4fb`, on
`codex/kuno-image-speed`. Integrate the complete Kuno range after shared base
`4f9b465`, including residency/shared-PE prerequisites, preserving this checkout's
newer application/coordination work. Do not cherry-pick only the selection commit. Do not replace the standard
gateway with the SYCL-linked worker or reuse the published ABI1 beta.2 runtime.
No model/content storage, API, frontend, public release or driver/reset/reboot
change is part of this performance handoff.

**Status:** Kuno backend, Vashti connector, and conversational image flows are
implemented and source-tested. Synthetic application flows and installed native
generation, regeneration, editing, previews, and cancellation have been verified
through Vashti in browser tests. Physical Android acceptance remains separate.
Baseline `d117295` on `codex/vashti-images`, recorded
2026-09-30. Preserve the current checkout and newer work.

**Agreed access direction:** Kuno is a local-network inference service. Remove
control-key authentication from public generation and previews on upgrade,
without adding inference bearer modes, image access settings, or a separate
operator migration. Privileged loopback controls retain their existing key.
The user's clarification supersedes the earlier pending authentication proposal.

**Current contract:** see [inference access, discovery, and streaming](kuno-images.md#inference-access-discovery-and-streaming-2026-09-30)
for exact discovery fields, readiness/errors, keyless/control separation,
existing-install behavior, and the default NDJSON generation response.
Streaming delivers previews and final output on the generation connection.
The user explicitly dropped compatibility with the older beta Vashti image
connector: remove PNG-only responses, polling, and preview identifiers. Ordinary
Ollama compatibility is preserved. The backend and consumer receipts are below.

### Baseline Coupling (before this change)

- [`gateway.rs`](../apps/kuno/src/gateway.rs): image generation and preview routes
  on the compatibility listener require the same bearer token used by the
  separate control listener. Ordinary Ollama routes keep their proxy semantics.
- [`handlers.rs`](../apps/vashti/src/images/handlers.rs): image settings expose
  `token_configured`; omitted/null `token` preserves the stored value, while an
  empty string clears it. Token validation currently differs from Kuno: Kuno
  accepts `.` in addition to letters, digits, `-`, and `_`; Vashti does not.
- [`service.rs`](../apps/vashti/src/images/service.rs): `endpoint()` rejects an
  empty token before connecting; generation and previews use bearer auth.
- [`ImagesWorkspace.tsx`](../apps/vashti/web/src/images/ImagesWorkspace.tsx) and
  [`types.ts`](../apps/vashti/web/src/images/types.ts): token presence is treated
  as service readiness and gates Generate. The settings form submits null for a
  blank token, so it cannot currently clear an existing credential.
- [`kuno-protocol`](../crates/kuno-protocol/src/lib.rs): versioned shared types;
  current control status is not a general keyless capability contract.

### Contract and Consumer Requirements

- Keyless image generation streams previews on the inference listener, separate
  from privileged control authentication. Preserve existing Ollama auth/origin
  behavior. Do not solve manual setup by disabling all authentication.
- Specify discovery route, versioned fields, actual availability/readiness, and
  authentication expectations. A plain Ollama host must remain usable for chat;
  absence of a token is not proof that an image service is disconnected.
- Preserve external proxy access errors. Remove the obsolete image-key workflow
  and stored secrets through a Vashti forward migration; new Kuno inference
  needs no credential.
  Never return credentials to the browser or silently downgrade failed auth.
- Specify compatibility with installed configurations and older Vashti/Kuno
  versions. Distinguish unsupported images, missing models, unreachable hosts,
  authentication failures, and actual readiness rather than collapsing them all
  into a connection warning.
- Record representative, secret-free JSON responses and relevant HTTP error
  codes in the authoritative API specification. Keep discovery bounded and
  content-free; do not make it an expensive inference probe.

### Acceptance and Receipts

Backend tests cover keyless inference with missing/obsolete credentials,
unauthenticated control rejection, generation/preview policy consistency,
ordinary Ollama proxy behavior, configuration compatibility, discovery/readiness,
and stream events, backpressure, disconnect cancellation, and cleanup ownership.
Vashti tests should cover plain Ollama, the new Kuno contract, older-Kuno upgrade
guidance, stored-credential removal, incremental NDJSON handling, actionable
errors, and admin/non-admin UI.
Manually verify discovery, Generate, previews, Stop, and settings on mobile and
desktop; record any unavailable live-engine tests rather than claiming success.

### Kuno receipt (2026-09-30)

**Changed commit:** `f6f6c22f6eea3f8e33900b725fbbfe56fda44e01`
(`Make LAN image inference keyless and stream previews with results`), baseline
`d117295`. Backend areas: `apps/kuno/src/gateway.rs`, new `image_access.rs` and
`image_stream.rs`, shared discovery/stream types in `crates/kuno-protocol`, native
CUDA capability constant in `crates/kuno-image`, and focused tests. API/source
specifications and the Kuno development guide are updated. No code under
`apps/vashti` changed.

**Exact checks:**

- `cargo test -p kuno -p kuno-protocol -p kuno-image`: 171 passed, nine ignored,
  zero failures. Ignored checks require GPU/native/systemd environments.
- `cargo clippy -p kuno -p kuno-protocol -p kuno-image --all-targets -- -D warnings`:
  passed.
- `KUNO_SD_SOURCE_DIR=/home/jmicko/dev/vashti/target/kuno-sd-source-88411ef KUNO_SD_BUILD_DIR=/home/jmicko/dev/vashti/target/kuno-sd-build-88411ef-cuda cargo check -p kuno --features native-image --all-targets`:
  passed using the pinned patched engine.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

Tests verify public inference with absent/obsolete keys, protected control
routes, no public administrative proxy path, JSON/origin policy, unchanged
Ollama header forwarding, discovery while admission is paused, conditional
attention support, missing-model degraded startup with a disposable worker,
preview/result/error ordering, actual synthetic HTTP preview delivery,
disconnect propagation, content-slot retention through delivery/cleanup,
recovery after uncertain cleanup, and privacy failure before body polling.
Native worker cancellation/device-loss tests remain passing. No owned GPU was
loaded for this work.

**Installation/configuration:** existing configuration loads without new keys.
Upgrading removes the public image control-token requirement automatically.
Control credentials/listeners, explicit timeouts, model settings, and Ollama
routing retain their roles. No installer/settings migration is needed in Kuno.
This is a breaking beta image transport change: install only with the updated
Vashti connector. No installation, service restart, release, or task message was
performed. Existing uncommitted `AGENTS.md`, this guide, and root README edits
remain uncommitted and preserved.

**Required Vashti work:**

1. Discover `/kuno/v1/capabilities` from the selected existing inference backend,
   with the bounded probe/error rules in the API spec. Chat on plain Ollama stays
   usable; unsupported images/old Kuno receive upgrade guidance.
2. Remove the image-token field, readiness gate, and bearer header. Remove stored
   legacy secrets with a forward migration; never return/repurpose those values.
3. Incrementally consume the single NDJSON generation response; previews and
   final image use the same request. Validate/decode bounded PNGs, distinguish
   pre-stream HTTP errors from terminal error events, and treat premature EOF as
   failure. Remove polling and preview-ID handling. Do not replay dispatched jobs.
4. Propagate Stop by dropping the upstream request/response. Preserve Vashti user
   authorization, private storage behavior, and image/job ownership.
5. Test discovery, settings, generation/edit, preview display, errors, and Stop
   through the Rust connector and mobile/desktop UI before deployment.

**Limitations:** source checks and synthetic transports do not prove real engine
stream performance or full application behavior. No live generation on either
GPU, installed-service check, Android/mobile test, or end-to-end Vashti test was
run. Readiness is a cached prerequisite snapshot, not a memory reservation or
GPU health guarantee; it refreshes on demand after five seconds, allows only one
probe, and returns its prior snapshot after a one-second wait. Slow filesystem
probes can outlast that wait. Existing image-worker deadline and global inference
serialization remain unchanged. Fast attention is advertised conservatively
only for a configured CUDA image device with a CUDA-enabled native build.

### Vashti receipt (2026-09-30)

**Consumed backend:** `f6f6c22`, using the discovery and mandatory NDJSON contract
above. Consumer changes are in the current worktree on `codex/vashti-images`;
no consumer commit, release, or installed Kuno update was made in this pass.

- Added bounded, per-host cached discovery and a reusable server-side streaming
  client. Validate service/API markers, same-host paths, reference limits,
  attention, event order and sizes, and decoded PNG dimensions. Surface access,
  readiness, protocol, and interrupted-delivery errors without raw error content.
- Removed key fields, key-presence readiness, bearer auth, preview identifiers,
  and upstream preview polling. Forward migration clears/removes the legacy
  database column while preserving the selected host, quota, and image assets.
- Image preferences now check the selected existing inference host and show
  actual workflows/readiness. Generate, Retry, and chat tool availability use
  capabilities. Settings changes remain admin-only; image access remains
  session/permission scoped. Unsaved host selections cannot dispatch work.
- Both persistent and private image jobs use streamed, transient previews and
  final delivery. Stop drops the upstream body and clears previews. No automatic
  replay or arbitrary generation response deadline was introduced.

**Exact checks:**

- `cargo test -p vashti`: 134 passed, zero failures.
- `cargo clippy -p vashti --all-targets -- -D warnings`: passed.
- `node --experimental-strip-types --test src/*.test.mjs src/images/*.test.mjs`
  from `apps/vashti/web`: 13 passed, zero failures.
- `npm run build` and `npm run build:native`: passed, including TypeScript checks.
- `cargo fmt -p vashti -- --check` and `git diff --check`: passed.

Tests cover old/plain-Ollama discovery, external 401/403 without fallback,
redirect/malformed/oversized/slow discovery, partial readiness, split stream
chunks, invalid and duplicate results, premature EOF, bounded events, legacy-key
migration, admin/non-admin access, cancellation/disconnect, ownership, private
RAM-only delivery, chat image tools, and original-prompt improvement semantics.

**Manual application checks:** used the development app through the browser and
a temporary loopback-only synthetic Kuno host. Verified discovery and saving a
keyless connection, ready workflow choices and unavailable Fast attention,
Generate, decoded preview/final image display, reference-image Edit, Stop
clearing previews and disconnecting upstream, interrupted delivery preserving
saved output without replay, and proxy denial disabling Generate/Retry. Checked
desktop and 390px mobile flows, plus 320px preferences with no horizontal
overflow. Restored the original connection/quota and removed only test-owned
backend, sessions, and assets after checks; user data was not removed.

**Limitations:** browser resizing is not a physical Android-device test. The
installed local host without discovery correctly shows upgrade guidance. No
installed Kuno restart, native GPU generation, or live-engine performance test
was performed. Upgrade Kuno and Vashti together after quiescing existing jobs;
synthetic delivery does not establish GPU readiness or speed.


### Installed Kuno receipt (2026-09-30)

The user explicitly authorized installation. Built committed Kuno `f6f6c22`
from an isolated worktree with the pinned static CUDA/Vulkan native engine:
`KUNO_SD_SOURCE_DIR=/home/jmicko/dev/vashti/target/kuno-sd-source-88411ef KUNO_SD_BUILD_DIR=/home/jmicko/dev/vashti/target/kuno-sd-build-88411ef-cuda CARGO_TARGET_DIR=/home/jmicko/dev/vashti/target cargo build --locked -p kuno --release --features native-image`.
Ran `sudo -n /home/jmicko/dev/vashti/target/kuno-install-f6f6c22/kuno upgrade --yes`
in the shared terminal after user-entered sudo authentication. The transactional
upgrade reported `kuno_upgraded_and_active`. No Vashti application files or
services were installed or restarted in this pass.

Installed executable SHA-256:
`09f9f5d3b4f56a02ad08605ef441508a9ad1723f4d28a03d55073655f876810c`.
Configuration remained byte-for-byte unchanged, including `[::]:11434`, the
loopback upstream/control ports, both workers/collections, image GPU RTX 3060,
and disabled LLM response/idle deadlines.

Verified on the running installation:

- Root `kuno doctor`: installation complete, unpaused, no active/waiting request
  at the post-test check, no recovery requirement/pending recovery.
- Kuno service active/running with zero restarts; recovery timer active.
- Keyless discovery reports all configured image workflows ready. Ollama version
  and discovery succeed through IPv4 loopback, IPv6 loopback, and the LAN address.
- Clark independently reached LAN discovery and Ollama's version endpoint.
- Ollama catalog endpoint succeeded; unauthenticated local control returned 401.
- Direct keyless native Qwen 2.1 stream: 512x512, two steps, standard attention;
  acceptance, two previews, and one final PNG. First preview at 8.60 seconds,
  total 11.61 seconds. Validated dimensions, PNG chunk checksums, metadata
  absence, and terminal completion. All image bytes stayed in client RAM and
  were discarded; no test images were saved.

This is a native transport smoke test, not a quality/performance benchmark or
proof of physical Android/full application workflow behavior. Live Stop/Edit
and full reference workflows were not tested in this installation pass.
Earlier source/application receipts retain their historical limitations; this
receipt supersedes their statements that installed Kuno was unchanged.
Existing uncommitted coordination and Vashti task work is preserved.

### Vashti conversational-flow receipt (2026-09-30)

The user approved replacing the separate image-generation forms with familiar
conversational flows. Consumer work remains uncommitted on `codex/vashti-images`.
The existing Kuno discovery contract was sufficient; no further Kuno API or
engine changes were made by this task. The Kuno task updated the installed local
service while browser acceptance was in progress.

- Added the Images launcher, host-grouped image-model picker, bottom composer,
  expandable settings, compact prompt bubbles, streamed response bubbles, and
  image-flow sidebar history. Gallery and image-tree inspection reuse these
  persisted flows and exact image ancestry. Existing chat tools remain separate.
- A forward migration adds active-result selection, automatic/manual titles,
  history removal, immutable submission fingerprints, and run pagination.
  Regeneration captures the original host, prompt, settings, and references,
  allocates a new seed, and stacks response variants without copying children.
- Idempotent submission retries recover lost acknowledgements without replaying
  dispatched inference. Drafts autosave without resetting the composer or
  overwriting generated titles. Older selected branches load their ancestry.
  A global active-run indicator keeps Stop available from the launcher/gallery.
- Fixed completion/polling races, preview sizing for native low-resolution frames,
  seed advancement after regeneration, narrow-screen composer spacing, and
  unsaved Preferences cancellation. Image decoding/layout changes no longer
  pause following the latest output; deliberate scrolling does. Preview bytes remain transient; completed
  assets are owner-scoped and use the existing bounded browser asset cache.

**Checks:** `cargo test -p vashti` (136 passed), Clippy with warnings denied,
frontend helper tests (20 passed), web/native frontend builds, Rust formatting,
and whitespace checks. Backend tests cover explicit/automatic host routing,
captured recipes, idempotency, new seeds and overflow, exact edit ancestry,
owner-only titles/selection/history, and removal retaining gallery assets.
Frontend tests cover branch reconstruction, incomplete history, independent
batches, pending retries, cyclic input bounds, and safe incremental seeds.

**Manual acceptance:** tested the development app with a disposable synthetic
loopback host, then the installed native Kuno service. Verified model discovery,
prompt/setting persistence, uploads, Match original, reference editing, native
generation and regeneration with distinct seeds, real previews, automatic
titles, cancellation during sampling, failed-delivery/manual Retry, variant
arrows and viewer swiping, exact child-branch restoration, tree focus/navigation,
and Stop while viewing the gallery. Synthetic backend/session/assets were
removed without changing existing hosts, quotas, preferences, or user histories.
Checked 1440x900 desktop, 390x844 mobile, and 320x568 short/narrow layouts; the
composer remains collapsible without document-level horizontal overflow. The
shared ordinary-chat model picker still lists/searches models. One native test
flow with saved generation/edit variants remains available for user inspection.

**Limitations:** resized browser tests are not physical Android acceptance or a
performance benchmark. Native checks use small temporary recipes rather than
changing user defaults. No commit, release, or deployment was performed.

### Kuno inference optimization installation (2026-09-30)

Internal Kuno work is isolated on `codex/kuno-image-performance`, based on
`f6f6c22`; concurrent Vashti source and the uncommitted coordination documents
are preserved. Standard CUDA/Vulkan improvements through `eee97c8` were
transactionally installed with user authorization. Installed SHA-256:
`bbd27f1f6cc9d35ecdb24ab32f274f46e8add649896753298fa2b9fb80d345cf`.
Configuration is byte-for-byte unchanged, including ports, RTX 3060 image GPU,
collections and disabled LLM response/idle deadlines.

The standard candidate passed independent attention/cache GPU oracles on both
cards, full original CUDA generation/edit and Intel edit, streamed previews,
multiple references and cooperative cancellation/recovery. The installed
service passed keyless native 512×512/two-step NDJSON generation (both previews
and final, 11.994 s), PNG checksums/dimensions/no metadata, discovery/version
through IPv4, IPv6 and LAN, and unauthenticated-control 401. No private content
was saved; content-bearing tests use zero-swap cgroups. No new Xe reset entry
was observed.

Intel Vulkan execution works, but the speed target remains open. A private
SYCL vendor-library experiment is under development and is **not installed**;
neither are experimental dense MLP/tile policies enabled in the public worker.
Exact measurements and limitations are in `docs/kuno-image-performance.md` on
the Kuno branch. No API/discovery/stream changes or Vashti consumer changes are
required. No application install, merge, release or publication occurred.
Admission was resumed after the standard installation tests and again after
the private decode diagnostics; the installed service is unpaused and healthy.
Any later exclusive test window must resume it afterward.

Private SYCL source/runtime support through `cb34430` remains isolated on the
Kuno branch and is not enabled or installed. Synthetic kernels and small text
generation passed. The original edit completed all sampling steps but failed
during VAE decoding; reduced reproductions identify the SYCL `IM2COL` operation.
Two additional Xe reset entries occurred earlier, so their origin remains a
separate stability question. Full reference-workflow acceptance remains open.
The optional managed runtime preserves the same image API and native Qwen 2512
route; no Vashti connector/UI changes are required. Source tests and the exact
installation/manifest contract are recorded in `docs/kuno-managed-sycl.md` on
the Kuno branch. Do not present that experimental route as ready to users.


### Installed B70 SYCL acceptance (2026-10-01 UTC)

This receipt supersedes the earlier uninstalled/private-SYCL and failed-decode
status above. Backend source remains isolated on `codex/kuno-image-performance`,
based on `f6f6c22`; main remains on `codex/vashti-images` with its application
changes and both untracked coordination documents preserved.

**Changed commits:** managed runtime `9a51d46`, deployment/environment review
`7fe6283`, bounded IM2COL `46cbd5c`, independent large-oracle/addressing audits
`35eba2e`/`7fe2403`, wide copy addressing/build watchers/oracles `30cebda`, final
hardware/installed receipts `c5b45be7301209091cdd8387633fd1b837e290bf`. Earlier
CUDA/Vulkan improvements remain part of the Kuno branch. Full specifications
and detailed receipts are `docs/kuno-managed-sycl.md`, `docs/kuno-images.md` and
`docs/kuno-image-performance.md` on that branch.

**Exact backend contract:** the existing keyless image NDJSON API, discovery
schema, public Ollama routes and privileged loopback controls are unchanged.
Omitted image.engine still means native; explicit managed_sycl plus fixed runtime
qwen21-sycl-2026.1-v1 selects the optional worker for Qwen 2.1 text/edit only.
Qwen 2512 uses embedded native on the same configured image PCI. The managed
worker advertises standard attention only; explicit fast returns 400
image_attention_unsupported. Missing/invalid/untrusted/corrupt runtime produces
fixed image readiness reasons and 503 preflight errors while ordinary Ollama
remains usable. Manifest/CLI/configuration details are normative in the runtime
specification; no arbitrary executable/library path or inference credential was
added to the public API.

**Installed artifacts/configuration:** standard CUDA/Vulkan gateway SHA256
`d2a7a293bb021daebb1dda2c8b21a67443881a10ce0787dbd4b7ae2c4946d8dc`;
optional worker `414b86a3c498c0f848321766e99784a8d5d7fd7f3984c1bf35796f6b5b14796b`;
54-member runtime manifest
`94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`.
The owned-file transaction installed the runtime without changing config;
explicit image configuration changed only engine/runtime/PCI to B70
0000:2d:00.0. Collections, model store, listeners, Ollama workers and zero LLM
response/idle deadlines were preserved. Global default remains native. The
standard binary does not link oneAPI. Website runtime publishing/download and
older-Linux portability are not implemented or accepted; no public release.

**Exact checks:** source suite 182 passed/nine environment-dependent ignored,
Clippy warnings denied (nonnative and ordinary native all targets), native release
build, formatting/whitespace checks. Independent CPU/GPU arithmetic and attention
oracles pass. IM2COL modes 1–8 and wide-copy modes 1–3 pass, including actual large
VAE geometry, 2/4/8 GiB and INT32/UINT32 boundaries. BF16 numerical GPU coverage
and arbitrary copies above INT32_MAX work-items remain unverified.

The final packaged worker passes original 1184x1776/50-step/CFG1 generation
153.950 s (50 previews), original 1248x832/25-step/CFG4 edit 115.513 s (25 previews),
and two-reference 512x512/four-step editing 61.282 s. These fresh-worker tests used
only the flat bundle with the entire SDK extraction inaccessible, zero swap and
write denial before vendor initialization. Original outputs were visually
inspected in RAM without decoder seams, then discarded. Full-size cancellation
at step 3 reaped cooperatively in 3.061 s. No further Xe reset occurred during final
candidate acceptance; boot event count remains 7, including the earlier failed
large decode. The fixes address proven range/copy bugs, not every driver fault.

Installed HTTP checks pass keyless generation/previews/result, IPv4/IPv6/LAN,
control 401, standard-only discovery/fast 400, native 2512/two-step route 27.328 s,
ordinary Ollama generation 1.077 s, disconnect cleanup 1.003 s followed immediately
by Ollama generation 0.935 s, and missing-runtime 503 with Ollama 200/2512 ready then
restoration without restart. Final service active/running, NRestarts 0,
MemorySwapMax 0, admission unpaused, no active/waiting job or recovery quarantine,
no image worker left. Known synthetic request text/image fields and the control
credential are absent from the acceptance-window service journal. No generated
content was saved; temporary test configuration/runtime renames are removed.

**Consumer impact/limits:** no Vashti code changed or application/mobile test was
rerun for this internal backend work. Existing consumers must honor discovery's
standard-only attention list; saved CUDA fast recipes need explicit standard
selection. Up to ten references remain allowed by the API, but live acceptance
covers two and does not guarantee all inputs fit. ComfyUI user-supplied B70
cold/warm baselines remain 132.06/100.65 s generation and 78.25/64.48 s editing;
Kuno is substantially faster than its prior engines but parity remains open.
Corrupted journal-owned runtime files require manual repair before configure/
upgrade/recover; rollback snapshots are RAM-only. See the runtime specification.
No merge, push, Vashti deployment, release or message to another Codex task was
performed. Resume from this tested checkpoint with ordinary inference available.


### Kuno beta release handoff (2026-10-01 UTC / 2026-09-30 local)

**Status:** tested backend ready for beta integration; the user explicitly does
not require perfect/indefinite stability. Release preparation is complete here;
publication and final application acceptance belong to the Vashti task.

**Exact source:** Kuno branch `codex/kuno-image-performance`, base `f6f6c22`,
final tip `83ec03f5e75206711f151f72c87cfeab3c592a19`. Integrate the entire 44-commit
range `f6f6c22..83ec03f`, preserving the main checkout's uncommitted application,
migrations/specs and both untracked coordination documents. New commits are
`6ef9380` (verified packaging/notices/public build provenance, eight CPU tests)
and `83ec03f` (release receipt); serving behavior/hardware acceptance remains
`c5b45be`. No Vashti/Hub code changed here, and no merge/push/publication/message
to another task occurred.

**Authoritative release instructions:** `docs/kuno-release-handoff.md` on the
Kuno branch, currently at
`/home/jmicko/.codex/worktrees/kuno-image-performance/vashti/docs/kuno-release-handoff.md`.
It contains exact artifact paths/checksums, install commands, source provenance,
platform limits, receipts and remaining release-owner work. Preserve this
worktree and ignored `target/release-handoff` artifacts until they are copied.

**Contract:** no new inference API or credential requirement. Native remains
the default; explicit managed_sycl/fixed runtime enables B70 Qwen 2.1. Native
2512 uses the same configured PCI. Discovery is Standard-only on SYCL; explicit
Fast is 400. Missing optional runtime is fixed-reason 503 for affected images
while Ollama remains usable. Privacy and disabled default long-LLM deadlines
are preserved. Vashti must respect actual attention capabilities and test saved
Fast recipes alongside Generate/Edit/previews/Stop and storage. Physical Android
acceptance remains the application task's responsibility.

**Prepared optional artifact:** `target/release-handoff/kuno-sycl-linux-x86_64.tar.gz`,
156,166,814 bytes, SHA256
`d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f`;
approved manifest
`bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99`.
Its 64 members add ten native notices only; every old worker/library/member byte
is unchanged from the hardware-tested installed 54-member bundle, whose manifest
remains `94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`.
The installed runtime was not modified by packaging. Archive full read-back and
all eight CPU regressions pass. Standard packaging now fixes output paths and
rejects private Intel dependencies/RPATH instead of risking stale publication.

**Latest live receipt:** installed two-step stream after idle passed two previews
and final in 40.870 s; service active/running, NRestarts 0, MemorySwapMax 0, no new
Xe reset (boot count remains seven). Original full Intel generation/edit passed
153.950/115.513 s. Faster than previous Kuno engines, still behind ComfyUI; no
additional speed optimization or overnight soak is needed for this beta.

**Release limits/work:** GLIBC 2.39/GLIBCXX 3.4.30 and tested CPU/driver scope only;
current standard CUDA-enabled ELF requires libcuda.so.1 even with Intel images.
Other-system portability is unverified. The publisher currently lists three
artifacts and the website installer only extracts standard Kuno. Hosting the
optional runtime/checksum/install instructions and the final coordinated build
remain release-task work. Prepare every intended artifact before consuming the
one-time upload key because published Hub versions are immutable. Source and
packaging checks are not proof of full Vashti/Android release acceptance.

### Vashti beta.2 integration receipt (2026-10-01 UTC / 2026-09-30 local)

**Source and Git:** earlier application changes preserved in `c6109d4`; the full
Kuno handoff through `83ec03f` is merged as `63ce405` on `codex/vashti-images`.
The application defaults/selection fixes and regressions are committed as `20d2c0f`.
The normal checkout contains `docs/kuno-release-handoff.md`. Keep the Kuno
worktree: its ignored native build inputs are still used for release packaging.
No Kuno service configuration or production inference implementation was changed
by this application acceptance pass. No merge to main, push, or publication.

**Account settings contract:** generate/edit preference profiles add optional
host, image count, automatic improvement, prompt-helper configuration, and edit
source mode. Absent legacy fields preserve existing defaults; explicit null count
means endless generation. Settings remain owner-scoped. Flow changes persist in
the flow without rewriting account preferences. Explicit Save updates the selected
account profile; Reset restores all settings while retaining the current prompt
and appropriate edit references. New flow with these settings starts a separate
flow without copying positive prompts, references, or branch ancestry. Existing
flows and model-tool override permissions are unchanged. These are additive JSON
fields; no deployed migration was rewritten and Kuno's API is unchanged.

**Consumer fixes:** unsupported saved Fast attention is shown explicitly and
requires Use Standard rather than silently changing the recipe. Submission
transactionally anchors the selected image branch before output arrives; later
manual navigation still wins. Returning to an active generation resumes following
its completion, and returning to completed output persists the selected result.

**Live acceptance:** installed Standard-only Kuno discovery, 512-square/30-step
generation, 25-step editing with two ordered references, streamed previews,
regeneration with a new seed, Stop, Retry, persisted branch selection after
reopening history, and explicit Fast-to-Standard recovery passed in the browser.
The two references were copies of the same synthetic test image; this verifies
transport, not arbitrary multi-image semantic quality. The earlier two-step image
was intentionally a transport smoke test, not a quality sample. Responsive
390x844 settings checks covered Save, Reset, new-flow copying, subsequent fresh
flow defaults, and original/iterative edit mode. Test account preferences were
restored afterward. No inference job remains active.

**Source/package checks:** full locked workspace tests passed serially; Clippy
all targets with warnings denied, formatting, frontend helper tests, web/native
frontend builds, release packaging regressions, and Kuno's eight CPU packaging
tests passed. A 100 ms Kuno preload-body-timeout fixture failed once under heavy
parallel build/GPU load, then passed individually and in the full serial suite;
no inference deadline or test behavior was changed. Packaged Linux starts against
a fresh data directory and serves beta.2 plus the exact final frontend assets.
The universal signed APK has version code 13002 and four ABIs. All four release
artifacts pass checksum and signing dry-run validation, including the unchanged
approved optional runtime archive.

**Release gate and limits:** complete multipart upload is approximately 644 MiB,
but the live Hub currently sets `VASHTI_HUB_MAX_UPLOAD_BYTES=536870912`. Raise it
and any proxy limit to at least 1 GiB before requesting/using a one-time key.
No live Hub change or upload was made. Physical Android installation/export,
browser download completion, other-system portability, overnight soak, and
ComfyUI parity remain unverified. Native/optional runtime platform restrictions
and installation instructions are in the versioned release notes and backend
handoff; source/browser results are not a substitute for those remaining checks.

### Live upload-size gate cleared (2026-10-01)

At the user's request, the live Hub request limit is now 1073741824 bytes,
set in `/etc/systemd/system/vashti-hub.service.d/90-upload-limit.conf`. The original
unit and installed binary are unchanged (their before/after hashes match). The
Hub restarted successfully and its running process environment contains the new
limit. The separate nginx site configuration now sets `client_max_body_size 1G`;
only that value changed, the prior configuration was backed up on the proxy host,
`nginx -t` passed, and nginx reloaded without a restart. Sudo authorization for
both changes was entered by the user in the shared Demons shell, not chat.

Header-only HTTPS probes with no key or artifact data verified the boundary:
declared Content-Length 1073741824 reaches the Hub and returns authentication 401;
1073741825 is rejected by nginx with 413. The public website returns 200. This
establishes the configured request-size gate, not a completed artifact upload.
No key was requested/consumed, no release was uploaded, and the physical Android
and portability limitations above remain unchanged. The earlier 512/500 MiB
limits in the historical receipt no longer block the beta.2 upload.

### Beta.2 publication receipt (2026-10-01)

Published `v0.13.0-beta.2` using the normal signed, four-artifact multipart path
and the user-supplied one-time key. The credential was entered through the
publisher's hidden prompt and was not saved to a file or handoff. The Hub accepted
the complete request; no subsequent artifact was added to the immutable version.

Public verification covered the release database entry/status, exact release
notes, SHA256SUMS, all four prerelease manifests, their Ed25519 signatures against
the repository's compiled public key, artifact filenames/sizes, and full streamed
downloads whose SHA-256 hashes match the local release files:

| Target | Bytes | SHA-256 |
| --- | ---: | --- |
| linux-x86_64 | 9840548 | `bdc8eeffdadf0ab97efa9acc2c1d1625c50f444de396c00b12b29844c5efb577` |
| android-universal | 52023257 | `e3202deffa2604977961e7df7e1e2d5b94bba69d5eeb85a74bf41e0b69ba3f8d` |
| kuno-linux-x86_64 | 456424687 | `b74938ea3d4f1b44626ed295680db85ebe051d5ac398593cdf77b1889d27d126` |
| kuno-sycl-linux-x86_64 | 156166814 | `d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f` |

The prerelease channel now returns beta.2; stable remains `v0.12.0`, with no
promotion, main-branch merge, Git push, or live Vashti/Kuno installation. The
physical Android and platform/runtime limits in the acceptance receipts remain.

### Installed Kuno resident-image handoff (2026-10-01)

This residency receipt is historical. The active handoff above selects the
newer `codex/kuno-image-speed` closure and replaces this artifact selection;
retain the following evidence, but do not package the earlier runtime.

The user approved retaining image models between requests. Backend work is
isolated on `codex/kuno-image-residency` through installed acceptance commit
`fb2d1d0e367d4d53054bf8de63f49b316046d704`, based on `4f9b465`, in
`/home/jmicko/.codex/worktrees/kuno-image-residency/vashti`.
Read that worktree's updated `docs/kuno-images.md`, `docs/kuno-managed-sycl.md`,
coordination receipt, `docs/kuno-release-handoff.md` Resident image update and
`docs/kuno-image-residency-build.json` for the exact contract/artifacts/tests.
The public API is unchanged; Vashti needs no connector or UI change. The new
private runtime uses ABI 2; the published beta.2 ABI 1 runtime remains one-shot
through the new gateway's upgrade fallback.

Source checks passed 205 tests (nine ignored), default/native Clippy, formatting,
eight packaging regressions, and native CPU cleanup tests. CUDA live smoke
confirmed retained GPU weights and normal idle teardown. Authorized local
transactional installation preserved configuration byte for byte. Installed
Intel generation/edit/CFG/resize, previews, LLM handoff, cancellation/reload,
legacy-family reuse and final service/privacy checks pass. The B70 retained
15.6–20.3 GB; small 512-square/four-step generations took 40.604 seconds including
initial load and 2.241 seconds warm. This is a synthetic residency smoke, not
ComfyUI parity or full-workflow performance acceptance. Intel repeats show tiny
rounding-scale pixel drift; see the detailed limitations and legacy diagnostics.

Kuno is active, unpaused, with the Qwen 2.1 model loaded, zero automatic restarts,
zero swap and no recovery latch or new Xe kernel events. No prompts/images were
saved by the engine or test clients. Preserve this uncommitted coordination edit;
no backend merge, push, release or Vashti/Hub code change was performed here.
Integrate the complete resident branch and package its new standard gateway and
ABI 2 archive together for the next beta; do not reuse beta.2's worker. Application
and Android/release acceptance remain with the Vashti task.

### Vashti compatibility and gallery receipt (2026-10-03)

Reviewed the complete owner diff from `4f9b465` through `ebd1925` and its
selected Intel performance handoff. Shared protocol types and public image
discovery/generation fields are unchanged. Installed keyless discovery reports
all three ready workflows, Standard-only attention, and NDJSON generation.
Vashti's connector already enforces advertised attention/reference limits and
handles bounded incremental previews/results, terminal errors and disconnects;
no connector migration or inference deadline was added.

The gallery now uses small square photo tiles, stack-count overlays, adjustable
density, an original-image viewer with its actions, and append-only page loading.
Gallery PNG thumbnails are derived in RAM with a separate bounded cache and
decode queue; originals and encrypted device storage are unchanged. Owner-scoped
server and device galleries use the same UI. Older stacks are loaded across
session-result pages rather than silently limited to the newest 100 results.

Application tests pass: 140 Rust and 44 frontend tests, including the installed
discovery shape, stream/error handling, thumbnail cache identity/eviction/queue
cleanup and gallery pagination. Production web and native-frontend builds pass.
Browser checks cover phone/desktop density and overflow, original-image and
keyboard version viewing, flow navigation, editing from a chosen variant,
removal confirmation/cancel and the empty device-only gallery. No GPU generation,
Kuno installation/restart, physical Android export, release, merge or commit was
performed in this application pass. Integrating the backend source and matching
release artifacts remains a separate release task.
### Detailed resident image backend receipt (2026-10-01, historical)

The user approved retaining image models after generation. Implementation lives
on `codex/kuno-image-residency`, based on `4f9b465`, in the managed worktree
`/home/jmicko/.codex/worktrees/kuno-image-residency/vashti`. Backend changes through
`4fb4e58` add the persistent supervisor, private worker ABI 2, native request
cleanup, retained GPU parameters, cancellation safety, and managed LLM placement.
See the normative [image residency contract](kuno-images.md#api-and-lifecycle)
and [private worker ABI](kuno-managed-sycl.md#resident-worker-abi-2).

The public image/discovery API and Vashti storage contracts are unchanged.
No code in `apps/vashti` or the Hub changed. Metadata keeps the image worker;
managed inference prefers another GPU and evicts images before unknown or
unsafe same-device loads. There is one compatible image context, with no idle
expiry. Cancellation and failure conservatively unload it; useful inference is
still serialized. This is not an arbitrary workload-sharing guarantee.

**Verified source:** `cargo test --offline --locked -p kuno -p kuno-image -p
kuno-protocol -- --test-threads=1` passed 205 tests, nine ignored.
Clippy all targets with warnings denied passed both default and standard
native-image builds. Formatting/diff checks and eight CPU packaging regressions
passed. The nine new supervisor tests cover repeated PID, CLEAN gating,
cancellation/input/output backpressure, malformed output, replacement, and idle
exit recovery. Both native engine archives rebuild all 57 C++ units; the fresh
standard archive passes the actual CPU cache/request-cleanup regression.

**Verified CUDA smoke:** in a swap-disabled user service, three 512-square,
two-step Qwen 2.1 requests reused one process and retained 7.69–8.23 GB in VRAM.
Times were 11.979, 7.977, and 7.772 seconds. A changed seed changed pixels;
returning to the original seed reproduced them. Idle shutdown exited normally.
No images or request content were saved. This is a process-cold smoke with an
uncontrolled file cache, not a full-workflow/ComfyUI benchmark.

**Prepared artifacts:** exact source/build pins and checksums are in
[`kuno-image-residency-build.json`](kuno-image-residency-build.json).
The standard gateway is `target/standard-cargo/release/kuno`; the new ABI 2 flat
runtime is `target/inference-tests/runtime-flat-resident`, and its verified
archive is `target/inference-tests/kuno-sycl-resident.tar.gz`, all within the
resident worktree. All 63 vendor library/license members are unchanged from the
approved beta.2 closure. Never package the private SYCL worker as the standard
gateway. Existing ABI 1 installs use the one-shot fallback until the optional
runtime is upgraded; ABI 2 requires the new gateway.

**Installed acceptance:** the prepared standard gateway and ABI 2 runtime were
installed transactionally through `install-image-runtime --yes` with the user's
sudo authorization. Configuration was preserved byte for byte: B70 image PCI,
model store/collections, Ollama workers/listeners and disabled inference deadlines
are unchanged. Installed hashes match the build receipt; all 63 vendor
libraries/notices are unchanged from beta.2. Tests used swap-disabled transient
clients and the installed no-swap service, with the development SDK hidden.
Live commands from the resident worktree were `sudo -n /usr/bin/python3 target/inference-tests/install-resident.py`, `run-resident-http.py` and
`run-resident-final.py` (the latter two with the same interpreter/prefix and
`target/inference-tests/` path). The wrappers run private clients in collected
systemd units with `MemorySwapMax=0` and `LimitCORE=0`; final audit is
`resident-audit.py`. Harnesses and safe numeric receipts remain in that ignored
directory for operator reproduction.

- Intel 512-square/four-step Qwen 2.1 generation retained one process and about
  15.62 GB VRAM: 40.604 seconds including initial load, then 2.241 seconds warm.
  Two-reference editing (CFG 4, two steps) reused it in 25.329 seconds; changing
  to 640x384/CFG 4 and back to the original recipe also reused it. Editing
  increased model allocations to about 20.33 GB. Previews and bounded PNGs
  passed integrity/dimensions/no-metadata checks; outputs remained in RAM.
- Metadata and a small LLM on the 3060 preserved the image process. The LLM's
  matching digest was resident on the configured 3060 worker, reporting
  2,427,128,381 VRAM bytes equal to its runtime size. A different LLM that could
  not be prepared on that alternate worker evicted the image before B70 load;
  it reported 2,156,354,599 VRAM bytes equal to runtime size. Both returned
  successful inference. Catalog file size is not proof of loaded footprint.
- Reload after LLM handoff passed. Disconnect after a live preview cancelled and
  reaped the image worker in 1.756 seconds; a new request loaded successfully
  in 40.100 seconds without recovery intervention. Repeating cancellation
  passed. Restarting the service with an idle model proved its old PID gone.
- Switching to native Qwen 2512 and back replaced the old processes normally.
  Two 512-square/two-step/CFG 5/shift 3.1 legacy requests reused one Vulkan
  process in 8.691 and 8.562 seconds, retaining about 23.8–24.2 GB VRAM.
- Final service is active, unpaused, with one idle Qwen 2.1 worker retaining
  15,620,767,744 VRAM bytes, zero automatic restarts and no recovery latch.
  IPv4/IPv6/LAN version/discovery pass; keyless image readiness is `ready`,
  unauthenticated controls remain 401. No new Xe reset/wedge/hang events appeared.
- Content-free final audit verified zero swap, core limit zero, worker
  `NoNewPrivs`, no writable regular-file handles, no test prompts/images,
  control credential or catalog model names/digests in Kuno's journal since the
  install window. The private SDK is restored; tests retain no images/prompts.

**Limitations:** this is small synthetic residency/transport acceptance, not a
new ComfyUI/full-workflow performance comparison, useful edit-quality result,
overnight soak or arbitrary two-model inference-peak guarantee. Intel repeated
pixels differed by at most one 8-bit channel level: mean 0.003344 after mixed
jobs versus 0.003307 for an immediate repeat (about 0.33% of channels). This
supports rounding-scale nondeterminism in these samples, not a universal
fidelity/tolerance guarantee or proof of its specific kernel. CUDA smoke was
bitwise repeatable. Native cleanup removes reusable request state, without
claiming forensic zeroization of every third-party freed allocation.
Four classified backend diagnostics occurred during the initial legacy Vulkan
run even though generation/reuse/family switching passed; their internal cause
is unproven. Harness assertions for the wrong DRM counter name, changing
KiB/MiB units and unitless zero were corrected; they did not require backend
changes. Actual per-client GPU allocations and device totals agree.

**Release handoff:** integrate the complete branch range from `4f9b465`, including
behavior through `4fb4e58`, source/build receipt `418569e` and this installed
receipt. The main checkout retains an uncommitted pointer to the final receipt
commit. The new standard binary and ABI 2 runtime must be packaged together;
see [the release handoff](kuno-release-handoff.md#resident-image-update).
No public contract or Vashti connector/UI changes are required. This receipt
records authorized local installation, not publication or application/Android
acceptance. No merge, push, release or task message was performed. Preserve the
worktree and ignored producer artifacts until next-release integration.

### Vashti beta.3 preparation receipt (2026-10-05)

The pending reference-viewing/comparison feature is committed as `e67dd40`.
The complete Kuno owner range after `4f9b465` through `ebd1925` is merged into
`codex/image-flow-refinements` as `18935a8`, preserving both applications'
coordination records. Vashti and Android are aligned at `0.13.0-beta.3`, with
Android version code 13003. No public image contract or migration changed.

All four release artifacts are prepared in
`apps/vashti/dist/release/v0.13.0-beta.3`. The standard Kuno archive uses the
verified `shared-rope-standard-v2` native source/build closure described in the
active handoff, built with the ordinary native-image package script and no
private Intel linkage. The optional archive is the exact approved ABI 2 bundle;
its manifest checksum is
`fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa`.
Versioned notes include explicit runtime replacement/enablement instructions
and platform limits. Existing ABI 1 installs retain the one-shot fallback.

| Target | Bytes | SHA-256 |
| --- | ---: | --- |
| linux-x86_64 | 9841269 | `8518c965354ffce7c5892c0fdf6e70adc28335bc548c32e2ca6db1be43d56e70` |
| android-universal | 52322265 | `2e2326eafd8e06d148951f1b260fdcc85ad3c9bd08f16e23bb4ed10d166ee32e` |
| kuno-linux-x86_64 | 456471447 | `8ab008c9c36647b7b84b6c5d7eb2436c05ea8a27435db0937a8b53f021ad1cfc` |
| kuno-sycl-linux-x86_64 | 156247147 | `85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3` |

Checks passed: locked Rust suites (345 passed, nine hardware/privilege tests
ignored), frontend helpers (57), release packaging (four), Kuno packaging
(eight), managed builder fixtures (six), formatting, whitespace, default and
standard native-image all-target Clippy with warnings denied, and production
web/native frontend builds. The complete four-artifact publisher signing
dry-run and SHA256SUMS verification pass. APK signatures verify under the same
certificate as beta.2; all four Android ABIs and the new version/code are present.

The extracted Linux package started on an isolated loopback listener with a
fresh data directory, completed migrations, served beta.3 `/api/version`, and
rendered the bundled sign-in UI without browser errors. The smoke server and
temporary browser tab were closed. Installed Kuno keyless discovery still
reports three ready Standard-only workflows. No fresh GPU generation, installed
service change, physical Pixel acceptance, or Android export/gesture test was
performed in this release pass; the backend hardware and earlier application
browser receipts retain their stated scope.

The live Hub still lists beta.2 as its newest prerelease and v0.12.0 as stable.
A read-only disk check found an 8 GiB EBS volume with a 7 GiB root partition,
1,572,651,008 bytes available, and about 677 MiB of journals. Published artifact
metadata accounts for 1,493,187,697 bytes. Beta.3 adds 674,882,128 bytes plus small
metadata overhead; finalization hard-links the streamed upload files rather
than duplicating their contents. Recheck free space immediately before upload.
This does not authorize deleting releases/logs or resizing storage.

**Preparation checkpoint (superseded by publication below):** all local checks
were complete and a fresh one-time upload key was requested. At this checkpoint,
no upload, stable promotion, main-branch merge, Git push, Hub deployment or live
Vashti/Kuno installation had occurred.

### Vashti beta.3 publication receipt (2026-10-05)

Uploaded all four prepared artifacts together using the user's fresh one-time
key. Public verification covered the release listing/status, exact versioned
notes, VERSION, SHA256SUMS entries, all four prerelease manifests and their
Ed25519 signatures against the repository's compiled public key. Full streamed
downloads of all four artifacts match the local sizes and SHA-256 values in the
preparation table, totaling 674,882,128 bytes. The Hub orders checksum entries
by target rather than local filename order; the same complete entries match.

The prerelease channel now returns `v0.13.0-beta.3`; stable remains `v0.12.0`.
Published artifacts occupy 2,168,069,825 bytes according to the public release
metadata. The Hub service remains active; the post-upload root disk is 88% full
with about 856 MiB available. No release or log cleanup, storage resize, Hub
deployment, stable promotion, main-branch merge, Git push or live Vashti/Kuno
installation was performed. Physical Android acceptance and the backend
hardware/platform limits remain as recorded above.

### Kuno native text research proposal (2026-10-05)

The user requested research only on FreeToken and Strata. See
[the research and implementation proposal](kuno-text-engine-research.md), based
on integrated beta.3 commit `ae3e70d`, on `codex/kuno-text-research` in the reused
Kuno worktree. It pins both upstream repositories, the official Qwen3.8 model
configuration and an independent graph reference. The proposed first target is
the full 512-expert model on B70, followed by measured CPU/GPU expert scheduling,
heterogeneous execution and verified MTP if a compatible draft module exists.
The user's follow-up requests read-only reuse of Ollama's downloaded files and
default-native routing for verified models, with `ollama.<full-model-name>` as
the explicit managed-Ollama route. The research records the exact existing
Q4_K_M manifest, mixed tensor formats, approximately 120 GB artifact size and
the need for bounded host caches/disk-backed PLE. No duplicate weight download
is proposed. The Bonsai graph plan is unchanged; its intended compatibility
section now records this explicit alias exception to accidental collision
rejection. No native text architecture or alias is implemented.

Evidence is source inspection, model manifest/tensor metadata and read-only
hardware inventory. No runtime build/test, model-weight download, inference,
GPU workload, installation, service
change or performance verification was performed. No Kuno API, protocol,
configuration, consumer or UI behavior changes; **no Vashti work is required for
this research**. Exact endpoint/errors/capability fields remain to be specified
before implementation changes consumers. Implementation needs a separate
acceptance receipt and shared contract agreement. Keep request-derived router,
PLE and prefix state transient under the existing content-free privacy contract.

### Native Rust text implementation started (2026-10-05)

The user authorized the research plan's implementation and a goal to exceed
updated Ollama throughput. Rust inference correctness comes before performance
optimization; embedding an entire C++ graph does not meet this milestone.
Read [the updated Ollama baseline](kuno-text-performance.md). Ollama 0.35.1
automatically selected B70/Vulkan, producing 128 tokens in 100.346 seconds cold
(41.500 seconds load), 16.690 seconds warm with prompt reuse, and 25.935 seconds
warm with a changed prefix and no reported prompt reuse. Exact public synthetic
prompts, seed 424242, options, manifest and numeric receipts are committed with
the harness. This is one reference sequence, not a native speedup claim.

Development is on `codex/kuno-native-text` from `e9de15a`, with isolated loader
and operator worktrees. Initial scope is read-only borrowed GGUF access, pure
Rust quant decoding and scalar graph/state primitives. No native model catalog,
public API, default routing, Vashti connector/UI or deployed engine changes have
been accepted. Frontend work is not required at this checkpoint. Preserve the
explicit native-default/`ollama.` direction for later tested integration.

### Native Rust text correctness receipt (2026-10-05)

On `codex/kuno-native-text`, isolated from the normal Vashti checkout, Rust now
runs the downloaded Qwen4exp trunk directly through `crates/kuno-text`.
Foundations `c5e05b4`, operators `5cef124`/`d3d716c`, graph `8f6cff1`, tokenizer
`5f95495`, graph hardening/public fixtures `44cb36f`, and CPU pool `98b3112` are
committed. Read [the exact graph contract](kuno-native-text-graph.md) and
[performance/acceptance receipts](kuno-text-performance.md) before continuing.
The separate `apps/kuno/benchmarks/native-text` operator laboratory contains
Rust-authored packed Vulkan kernels, outside installed release dependencies.

Acceptance is deliberately narrow: native scalar raw/chat smokes reproduce
updated Ollama output, Rust tokenizer matches 75/80 baseline prompt counts,
all seven actual artifact types pass both-GPU packed numeric probes, and B70
full-graph GPU matrix execution reproduces the four-token raw smoke. 70 normal
Rust tests plus two explicit installed-artifact checks passed; cancellation,
bounded allocation, malformed file/tokenizer handling and state cleanup are
covered at the library level. This does not establish production worker privacy,
long-context/vision/tool/sampling/API acceptance or a native speedup. Model state
belongs to a sequence, cannot cross bindings, and is discarded on execution
error. No private request content or credentials appear in these receipts.

**Required Vashti changes: none at this milestone.** Native default/explicit
`ollama.` routing is still intended but unimplemented; current Ollama endpoints,
Kuno protocol/configuration, installed service and model list are unchanged.
Keep the existing managed scheduler and long-inference behavior when exposing
native text later. A tested public contract/capability and error handoff must
precede consumer integration. No installation or release is authorized by this
receipt; the user's current goal remains correctness followed by a measured
speedup and backend integration.

### Native text laboratory follow-up (2026-10-05)

On `codex/kuno-native-text`, bounded Rust graph backend hooks now support exact
full routed MoE and transactional GDN overrides. Cooperative packed kernels,
expert-bank planning/dispatch, and GDN parity pass on B70 and RTX3060; the short
four-token graph output still matches the scalar/Ollama oracle on both cards.
See [the detailed numeric/source receipt](kuno-text-performance.md#cooperative-reductions-expert-banks-and-gdn-laboratory).
Performance is still slower than updated Ollama. Full baseline, long context,
private-worker lifecycle and native default/`ollama.` managed routing remain
required. No API or Vashti consumer change is exposed by this backend laboratory;
there is nothing to integrate or release from this receipt yet.


### Native text resident-operator checkpoint (2026-10-05)

`codex/kuno-native-text` integrates read-only bound plan `2467ac7` and measured
transfer/cache/block-decoding experiments after `f8b2d2d`. The first full fixed
75-prompt/128-output Rust fixture completes but is still substantially slower
than updated Ollama. Both cards pass residual/full-span buffer erasure and
seven-step resident GDN operator parity. Read [the performance receipt](kuno-text-performance.md#transfer-reductions-and-first-full-public-native-fixture)
and [the resident ownership audit](kuno-native-text-residency.md).

This is backend-only laboratory work. It exposes no native model/capability,
changes no API/configuration or installed service, and requires **no Vashti
changes**. Native-default and explicit managed `ollama.` routing remain intended
but unimplemented. Runtime staging sanitization, the complete resident graph,
quality/options/lifecycle acceptance and a matched performance win remain
outstanding. Preserve normal Ollama behavior and long inference until the
verified engine is integrated. This checkpoint is not a release handoff.

### Native text connected resident graph follow-up (2026-10-05)

The isolated laboratory now connects GR/GDN/QSA/full all-ten MoE with a single
whole-token owner; PLE retains its exact Rust host oracle. Both cards match
four output tokens and pass cancellation after queued GDN/QSA, whole-arena
erasure, poisoned retry rejection and fresh-sequence recovery. B70 matches the
eight-token chat. Short decode overhead improves, but native remains slower than
Ollama. Long-position NVIDIA QSA numerical accuracy is being corrected without
loosening the oracle. Exact binary receipts and limits are in
[performance acceptance](kuno-text-performance.md#connected-resident-graph-acceptance-public-laboratory-2026-10-05).

Public weight wipe removal does not alter private-content cleanup. Runtime
upload/readback/uniform and scalar/INFO host copies still require checked
sanitization before real private requests. No native route/default/catalog,
public contract, deployment or Vashti code changes: **no Vashti integration is
required yet**. This remains development toward the active performance goal,
not a production/release or complete privacy receipt.

Connected resident laboratory checkpoint is committed as `2eec122`; subsequent
public rotary tables pass both-card QSA contexts1/4/5/2048/2049/2052 without
changing acceptance tolerance. See the updated performance receipt. The earlier
NVIDIA numerical failure is resolved; it was not a driver/GPU crash. This fix
changes no public contract or Vashti consumer and does not authorize deployment.


### Native text owned-I/O and admission checkpoint (2026-10-05)

Backend source/receipts are committed as `a27a7c3` on the isolated
`codex/kuno-native-text` branch, after `c75fdf2`. The standalone lab vendors
three exact CubeCL0.10.0 crates with retained licenses/source provenance; normal
production dependencies and installed services are unchanged. Read the updated
[resident ownership receipt](kuno-native-text-residency.md#owned-runtime-io-implementation-and-both-card-proof-2026-10-05)
and [performance limits](kuno-text-performance.md#owned-runtime-io-and-resident-admission-acceptance).

Both physical GPUs pass the fixed synthetic owned-I/O probe: audited complete
upload/uniform/readback spans, pending/retained lease rejection, runtime scalar
parity, public-cache preservation and full main-span erasure. Independent source
reviews found no blocker within that limited preowned Wgpu/Arena proof. The
connected public graph now has precharged complete pages, sealed public-vector
inventory and accounting that grants credits only after completed readbacks;
both GPUs pass bang/chat plus queued GDN/QSA cancellation, poisoned retry
rejection and fresh recovery. Intel's 72-slot option also passes chat; default
capacity remains unchanged. Tests: 27 runtime ownership, 128 native host/offline,
74 normal core with two installed checks ignored; strict host/native Clippy,
formatting and fresh-archive composed-patch/hash verification pass.

**Remaining:** complete connected private staging/host ownership, long fixture
and options/quality/lifecycle acceptance, true batched prefill and a matched
performance win. Native remains slower than updated Ollama. Opaque driver/
firmware padding, failed Drop and historical swap are outside owned-span proof.
Public receipts are fixed synthetic operator artifacts; no real request logs,
model downloads, copied/converted weights, new inference deadlines, device
reset, reboot or installation are part of this checkpoint.

**Vashti changes: none.** Native catalog/default/explicit managed `ollama.`
routing and public capabilities remain intended but unimplemented. No API,
protocol or consumer contract changed. This is a development checkpoint toward
the active goal, not a release/integration handoff. Keep the ordinary managed
Ollama and long-inference contract intact until native acceptance/integration.

### Native text connected ownership and matrix batching checkpoint (2026-10-05)

Source and public receipts are committed as `3de1bff`, after `9649c86`, on
`codex/kuno-native-text`. The isolated laboratory connects the resident graph to explicit
owned uploads, uniforms and readbacks, with separate 64 MiB main/runtime I/O
budgets and retirement shared across fresh owners of the same physical stream.
Both GPUs pass the fixed bang/chat fixtures, queued GDN/QSA cancellation,
poisoned retry rejection and fresh recovery. An opted-in inspection verifies all
1,286 main pages / 785,186,816 bytes after erasure for each of four owners on
each card. Read [the ownership and performance receipt](kuno-text-performance.md#connected-owned-io-full-page-inspection-and-multi-rhs-operators)
and [runtime ownership limits](../apps/kuno/benchmarks/native-text/runtime-ownership/README.md).

Skipping unused intermediate prompt heads preserves all 128 prior native output
IDs on B70. The long native output is not bit-identical to Ollama, and the
connected owned-I/O run is still slower. Token-level cleanup cadence passes
both-card acceptance but has no matched performance benefit yet. Expert
planning/mapping/LRU host allocations now wipe their full retained capacities.
Receipts are emitted only after explicit owner/executor cleanup succeeds.

The new packed multi-RHS operator passes 56 original-artifact numeric cases on
each GPU. Cached eight-vector ABBA samples are 3.05–3.71 times faster on B70
and 2.09–2.96 times faster on RTX3060 for the three tested matrices. This is
an operator result; causal layer-major prefill is not implemented. A separate
exact-order down kernel is numerically accepted but slower on both GPUs and
remains disabled. All new performance choices remain explicit lab options.

Checks pass: 169 native host/offline tests, 28 imported runtime ownership tests,
strict native/runtime Clippy, formatting, and reconstruction of the complete
18-file vendor patch from exact pinned archives. The separately accepted core
remains at 74 normal tests with two installed checks ignored. Full receipts
record public fixture settings, frozen executable hashes, ordered samples,
cleanup counts and limitations. Production worker/privacy, full quality/options,
long-context graph, native default/managed `ollama.` routing and a matched
whole-engine speedup remain outstanding.

**Vashti changes: none.** No public API, model catalog, configuration, installed
service or consumer changes are exposed by this laboratory. The native route
still requires a separately agreed and tested contract before integration.
No installation, release, GPU reset or reboot is part of this checkpoint.

### Native text GR/GDN batching and cache experiments (2026-10-05)

The next source/receipt checkpoint is `e378d32`, after `ecf3320`, on
`codex/kuno-native-text`. Read [the measured operator and cache limits](kuno-text-performance.md#batched-gr-operator-acceptance)
and [the internal causal prefill plan](kuno-native-text-prefill.md). These are
isolated compiled public fixtures, without real request support.

Both physical GPUs pass 24 original-weight GR cases and eight GDN recurrence
chunk cases with full owning-page and guarded I/O cleanup. Actual Qwen GDN
chunks match scalar GPU state/output exactly at counts1/3/5/8. Projections,
convolution, closing gate and full causal layer-major prefill still require
integration; no whole-engine gain is established by these operator checks.

The default-off segmented96 expert cache passes both-card Q5/Q8 boundary-slot
comparisons, partial-fill/post-enqueue cancellation, poisoning and cleanup. Full
B70 chat/cancellation/main-page audits also pass. Short compiler-idle ABBA is
near noise. A long96 candidate fails runtime cleanup after partial generation;
it writes no success receipt and the remaining ABBA samples are not run. The
64 reference and fresh Intel recurrent recovery pass. Underlying cleanup/GPU
error categories need diagnosis before longer96 acceptance or selection.

Token-level cleanup cadence and ordered public upload pass their specified
short acceptance but are slower in paired measurements and remain disabled.
An explicit teardown-order repair safely drains pending source staging before
duplicated main-page audits. All unchanged default choices remain in force.

Root checks pass202 native host/offline tests, strict all-target Clippy,
formatting and diff checks. Core and the pinned18-file runtime vendor patch are
unchanged from the prior checkpoint; their previously accepted checks are not
claimed as newly repeated here. Public receipts record exact executable/source
identities, ordered samples, failure exclusions and unit bounds with swap/core
dumps disabled. Longer96, whole-prefill, private-worker lifecycle, quality/
options, native catalog/default/managed `ollama.` routing and a matched Ollama
speedup remain outstanding.

**Vashti changes: none.** No production protocol, configuration, installed
service, deployment or release changes apply. Kuno admission is temporarily
paused only for exclusive operator GPU tests and must be resumed at the end of
that test window. No GPU reset or reboot is authorized by this checkpoint.

### Latest native-text owner checkpoint (2026-10-05)

**Current source/acceptance commit: `3e244d5`, after `1064a62`, on
`codex/kuno-native-text`.** This supersedes the outstanding callback diagnosis
and long96 acceptance limitation in the preceding `e378d32` checkpoint.
Read [the measured operator boundaries and callback repair](kuno-text-performance.md#mapping-callback-completion-repair-2026-10-05)
and [the full-prefill transaction plan](kuno-native-text-prefill.md).

Both GPUs pass24 causal GDN pointwise and24 causal QSA chunk cases at
counts1/3/5/8, including original arithmetic/status tolerances, future-group
exclusion, late nonfinite failure, immutable committed history, cancellation,
full owning pages and independent I/O cleanup. A QSA plane-index regression was
fixed before acceptance; checked/unchecked generated WGSL tests cover it.
The complete GDN/QSA projection wrappers, PLE batching and whole-layer-major
prefill remain under implementation. These isolated operators do not establish
an overall Ollama speedup.

Owned upload/audit/cleanup-read mapping now waits for callback delivery within
one shared requested deadline after native polling. This repairs a reachable
background-poller race without remapping, retrying, relaxing erasure or changing
ordinary inference waits. Content-free diagnostics expose only37 static first-
observed categories. The long96/6-GiB public75-input/128-output run now completes
with all128 prior native IDs and accepted I/O cleanup. It is one diagnostic run,
without full main-page audit or compiler-idle paired timing;96 stays disabled.

Default-off append-only INFO targets pass67-record/two-epoch numeric and
lease/cleanup fixtures at0/4096/8192 on both cards. Full chat and separate explicit
cancellation at0/4096 pass on both cards, including poisoned retry rejection,
fresh recovery and four1,286-page/785,186,816-byte complete owner audits per run.
No performance selection is made; oversized individual INFO has no separate GPU
fixture. The bounded expert-wave planner passes nine host fixtures preserving
all explicitly supplied ten routes/token; it is host-test-only, without actual
bank pinning or fullgraph consumption yet.

Root acceptance is231 native host/offline tests,39 pinned runtime ownership
fixtures, strict all-target Clippy, formatting and diff checks. The18-file vendor
patch reconstructs against pinned originals, SHA256
`e3aa774114789d47b6d14bc817a2e1f6da0a849020c0324492dd5c21cb2def4a`.
Exact synthetic receipts live under `apps/kuno/benchmarks/text/receipts`.
Full quality/options, production private-worker lifecycle, native catalog/default
and managed `ollama.` routing, matched cold/warm Ollama improvement and release
remain outstanding. Installed Kuno/Ollama are unchanged; admission was resumed
after testing. **Required Vashti changes: none at this checkpoint.** No GPU reset,
reboot, driver change, installation or publication is part of this handoff.

### Native text layer-composition source checkpoint (2026-10-05)

**Source candidate: `3a6c4cc`, after accepted checkpoint `3e244d5` and its
handoff `46be6b4`, on `codex/kuno-native-text`.** This does not supersede the
device acceptance above. Read [the composition source limits](kuno-text-performance.md#full-gdn-layer-and-fused-matrix-fault-source-checkpoint-2026-10-05)
and [whole-model chunk semantics](kuno-native-text-prefill.md) before connecting
these candidates.

The full candidate GDN layer includes all five original packed projections,
causal pointwise operations and recurrence. It has separate immutable current
and candidate state, all ten status planes and no state/cursor commit. Its fixed
public probe compares counts1/3/5/8 against the original scalar GPU schedule and
independent core oracle; those device comparisons have not run yet. Exact main,
public, host and I/O envelopes are in the source receipt.

The shared MAX8 projector reuses the existing original-weight cache, validates
full row extents before bounded tiles and preserves reserved-memory checks and
the separate whole-output fault fence. It is not yet connected to a whole chunk
owner or device-tested. The default-off cooperative projection candidate folds
only its output finite fault into the original64-lane arithmetic; block paths
keep the separate fence. Its probe rejects incomplete owning reads and wipes
byte leases on every exit. Seven formats pass14 checked/unchecked WGSL variants.

Combined host acceptance passes247 tests, strict all-target Clippy, formatting
and diff checks. Prepared device probes use frozen executable SHA256
`a1be34e63179b2ba1fdc33ad13c15fe27f8f456362f5cda69482dce3041f59b7`,
which predates the shared batch projector. Sudo expired before device units
started; normal Kuno admission remains enabled. No GPU/performance/private-worker
claim follows from these source checks. QSA/PLE layer composition, owned expert
waves, whole-model chunk connection and the full acceptance list above remain
outstanding. **Required Vashti changes: none.** No public API, catalog, default
routing, installed service, release or publication changed.

### Native text PLE/QSA and grouped-expert source checkpoint (2026-10-05)

**Current source candidate: `abf4396`, after `e43ea22`, on
`codex/kuno-native-text`.** Read [the layer source boundaries](kuno-text-performance.md#ple-and-complete-qsa-layer-source-candidates-2026-10-05)
and [the complete transaction plan](kuno-native-text-prefill.md). This does not
supersede the prior both-card device receipts.

Original layer1 PLE and layer3 QSA now have complete candidate wrappers and
fixed-public probes, with immutable committed histories, sealed geometry and
physical ownership, exact full owning readbacks and no state/cursor commit API.
PLE includes original16-branch u64 hashing and bounded110-byte Q5 row reads; its
new count1 GPU tail is labeled separately from the existing CPU tail. QSA
includes all six projections, complete-block-only causal selection and all17
status planes. Both-card numeric/cancel/cleanup runs remain pending. The frozen
combined lab is SHA256
`b7b2e40955d39829f71bf8a6dc09620b38f658ada3725dc311e77e13cf328983`; it
includes the earlier cache-backed batch projector and sealed GDN fields. Do not
substitute this identity into old `a1be34…` prepared-unit receipts.

The grouped all-ten expert operator is host/offline-only, supporting all seven
original packed formats and runtime protected maps with exact probability/rank
bits. It preserves the original64-lane arithmetic and reduces original ranks
only after all waves. The owning adapter to actual bank identity/revision,
protected union, partial-fill poisoning and completed consumer lifetime is still
being developed separately. No original bank ledger may publish before complete
model acceptance; routed-only probe acceptance is insufficient.

Root checks pass284 native host/offline tests, strict all-target Clippy, locked
offline release build, formatting and diff checks. Normal `kuno-text` passes75
tests; two installed checks remain ignored. Exact page/I/O/host budgets and
source hashes are recorded in the source-only PLE/QSA/grouped-operator receipts.
Opaque driver/register/spill and abrupt-loss erasure remain outside page audits.
Whole-model batching, quality/options, production private-worker lifecycle,
native catalog/default and managed `ollama.` routing, updated-Ollama cold/warm
speed improvement and release remain outstanding.

**Required Vashti changes: none.** No public API, configuration, installed
service, model download/copy/conversion or release changed. Kuno/Ollama listeners
responded with no resident models; admission remains resumed. Protected device
validation awaits sudo refresh. No GPU reset, reboot or driver change is part
of this checkpoint.

### Native GR ownership and protected expert waves (2026-10-05)

**Source commit `e17957f`, after layer source handoff `25246bc`, on
`codex/kuno-native-text`.** Read [the exact source checkpoint](kuno-text-performance.md#native-gr-workspace-and-owning-expert-wave-source-checkpoint-2026-10-05)
and the [whole-model prefill plan](kuno-native-text-prefill.md). This does not
supersede accepted device receipts or establish complete prefill.

The native GR workspace borrows original cached projections and public norms.
A sealed prepared capability remains alive through all residual consumers or
last-column output-head enqueue. Cancellation/error/abandonment poison it;
queued completion is explicitly distinct from the future whole-model fence.
Ten full MAX8 pages charge1,998,848 B including the mixed page. The new active
status shader guards before storage reads, with checked/unchecked validation.
Source peer review and the repaired final-head lifetime recheck pass.

The owning grouped MoE fixture validates real pending bank generations and
protects complete unions through at most ten misses/40 MiB per source window
and completed consumers. It retains staged ledgers privately and never publishes
the original routed bank ledger. Independent I/O stays64 MiB. Full-page source
credit requires owning lease destruction and checked cleanup; unexpected runtime
failure retires the context. Its40 positive,12 rejection and12 fresh-owner cases
per card remain physically untested. Shared expert/full FFN and the native adapter
using the full model's actual banks are not implemented by this fixture.

Root checks pass300 native host/offline tests, strict all-target Clippy, locked
offline release build, formatting/diff checks. Unchanged core acceptance is75
normal tests/two installed checks ignored. Frozen source binary is
`0c1b3f553ae24ee497f2f9637324778582feeeb89b9427e0c451d67d157438f5`,
`/tmp/kuno-rust-native-batch-owner-benchmark-20261005`. Receipts are
`rust-native-gr-batch-source-20261005.json` and
`rust-grouped-moe-chunk-source-20261005.json` in the text receipts directory.
Sudo remains expired; no GPU unit, service, configuration, API, catalog, route,
frontend or installed binary changed. Native GDN/QSA adapters, whole-model
transaction, matched Ollama improvement, options/quality and production private
worker/routing acceptance remain outstanding. No Vashti consumer change is needed
for this internal checkpoint; a future public native contract needs a separate
backend and consumer handoff.

### Latest native cached-layer and both-card checkpoint (2026-10-05)

**Source/device receipt commit `49cf147`, after `d918a10`, on
`codex/kuno-native-text`.** Read the
[cached-layer source boundary](kuno-text-performance.md#native-cache-backed-layer-adapters-2026-10-05)
and [both-card fixture acceptance](kuno-text-performance.md#both-card-layer-and-expert-fixture-acceptance-2026-10-05).
This supersedes the device-pending and sudo-blocked statements of the preceding
source checkpoints, while preserving their narrower fixture scope.

Frozen `b7b2e409…` passes GDN4 numeric/4 rejection, PLE8 numeric/its4 rejection
paths, and QSA24 numeric/4 rejection cases on each GPU. Frozen `0c1b3f55…`
passes40 routed-expert numeric/12 rejection cases and a fresh independent owner
after each rejection on each GPU. Complete reports, maxima, exact identities,
owned-memory bounds and erasure evidence are in the new device receipts. Tests
used verified user-unit swap/core limits; root authentication is no longer a
prerequisite for this laboratory path. Admission was not paused, CPU builds were
active, and unit runtimes are not benchmark measurements.

New actual-cache GDN/QSA adapters, shared-expert batch workspace and last-column
GR head are implemented and source-tested. Common sticky logical poison belongs
to the actual admission; pending Drop does not reenter its mutex. All-family
preflight and the completed whole-model fence remain mandatory before any state
or bank publication. These new adapters are not covered by the older fixture
receipts and have no selected whole-model consumer yet. Incremental full-page
and I/O components are recorded; they do not approve an aggregate allocation.

Root acceptance passes336 native host/offline tests, strict all-target Clippy,
locked offline release, formatting and diff checks. Optional finite-fault probe
`861b285a0dc581ec6c03c1d16dd7814269e33f1a103cbece43fc7ebd9416409e`
passes28 original-format cases and owning cleanup on each GPU after harness
initialization/nonfinite-upload fixes. It remains default-off. Exact source and
device receipts are `rust-native-layer-adapters-source-20261005.json`,
`rust-gdn-ple-qsa-device-20261005.json`, `rust-grouped-moe-device-20261005.json`
and `rust-fused-matrix-fault-device-20261005.json` in the text receipts directory.

Native actual-bank MoE/CPU-history PLE adapters, complete constructor/chunk
transaction, full adapter/device acceptance, matched Ollama improvement,
quality/options and production private-worker/native-default/managed `ollama.`
routing remain outstanding. Installed Kuno/Ollama services respond and remain
unchanged; no reset/reboot, API, catalog, route, frontend or installed binary
changed. No Vashti consumer work is required for this internal checkpoint.

### Current native actual-history/bank checkpoint (2026-10-05)

**Source commits `0f31146` and `09a9164`, after `49cf147`/`526f31d`, on
`codex/kuno-native-text`.** Read the
[actual CPU-history bridge](kuno-text-performance.md#native-ple-bridge-into-actual-cpu-histories-2026-10-05)
and [actual-bank/router receipt](kuno-text-performance.md#native-actual-bank-moe-and-guarded-router-2026-10-05).
This supersedes the adapter-pending statements of the preceding checkpoint;
complete model execution and device acceptance remain pending.

Native PLE uses the authoritative original CPU history objects and guarded
OLD/NORMALIZED/NEXT owning maps. Its opaque field-split preparation permits
other families and the required second actual fault/I/O/admission proof before
any commit. Original token/hash/convolution math remains unchanged. Native MoE
uses real original executor banks and retained final staged plans; complete
selected unions stay protected, scalar bank access is blocked during sessions,
and original ledgers publish only after complete-model acceptance and all-family
preflight. The guarded router preserves original rounding/ties/clamp and all
ten ranks, including zeros. Full-page, private-host and independent runtime
component bounds are recorded; no component alone admits the aggregate model.

Current root acceptance passes350 native and86 core host/offline tests, with
two installed checks ignored, strict all-target Clippy for both manifests,
locked offline release, formatting/diff and focused peer source review.
`rust-native-ple-bridge-source-20261005.json` pins the343/80 source checkpoint;
`rust-native-moe-router-source-20261005.json` pins the current350/86 checkpoint
The `27c25fea…` frozen-binary association in those source-only receipts is
invalid: an unrelated historical probe was selected. The forward correction
`rust-native-artifact-binding-correction-20261005.json` preserves that error and
selects the current combined model executable from actual Cargo metadata.
Source checks are unaffected; the wrong artifact established no device acceptance.
The approved complete constructor/chunk plan is linked above. Its actual
coordinator is being developed in an isolated snapshot. Owning embedding/router/
final-output/fault helpers are committed as `99f6e73`; composed root acceptance
passes361 native tests, strict all-target Clippy, locked offline release,
formatting/diff and focused peer review. Core86 tests are unchanged. The exact
helper source/file/runtime/host bounds are recorded in
`rust-native-batch-io-source-20261005.json`. Helpers add no MAIN pages; the
original source-only checkpoint preceded connection to the actual coordinator.
Do not infer source provenance from the identical old-probe artifact identity.

Separately, the original scalar model with optional finite-fault fusion passes
fixed-public four-output inference and cancellation/poison/fresh-owner cleanup
on both GPUs. Each card reproduces `[49276,2349,1074,5928]`; four owner audits
each inspect785,186,816 B/1,286 complete pages plus separately checked source/I/O
cleanup. `rust-fused-full-resident-device-20261005.json` preserves exact reports,
settings and binary identity. These protected user-unit runs are lifecycle
evidence, not matched performance measurements or new native MAX8 acceptance.
The exact protection wrapper is committed as
`apps/kuno/benchmarks/text/protected-lab-exec.py`; swap/core limits are checked
inside the unit before exec. Sudo is not required for this laboratory path.

The decode plan approves Stage1 original packed Rust CPU experts and a pinned-FD
positional reader for isolated synthetic source work. Actual calibration and
CPU/B70/secondary-GPU execution are later measured proposals. Whole-model native
adapter parity/lifecycle, complete prompt/output quality and options, faster
matched updated-Ollama cold/warm runs, production private-worker lifecycle and
native-default/managed `ollama.` routing remain outstanding. **Required Vashti
changes: none at this internal checkpoint.** No public API/configuration,
download/copy/conversion, installed service, frontend, release, driver, reset or
reboot changed. Installed Kuno/Ollama listeners respond with no resident models;
admission was not paused and all completed root GPU laboratory units are stopped.

### Current connected native model checkpoint (2026-10-05)

**Source `371b9d1`, after `bed3f96`/`99f6e73`, on `codex/kuno-native-text`.**
This supersedes the coordinator-pending statements immediately above. Read
[connected model acceptance](kuno-text-performance.md#connected-native-model-laboratory-2026-10-05)
and the linked original constructor/chunk and decode plans before further work.
Concurrent source work now uses managed Git worktrees from the agreed base;
older snapshots remain preserved but are not integration authorities.

The real48-layer native coordinator is connected. It charges all retained
original/new pages and original bank-state metadata before fill, executes
original causal/history/router math, completes two owning fault/I/O proofs and
all-family preflight before publication, and continues scalar decode from the
same accepted original model owners. Any failure terminally poisons that model;
explicit registry cleanup runs outside admission guards. Both independent
64-MiB I/O caps and existing GPU/cache/bank defaults are unchanged.

Root acceptance passes369 native and86 unchanged core tests (two installed
checks ignored), strict all-target Clippy, locked offline release and formatting.
The source receipt is `rust-native-model-max8-source-20261005.json`. Correct
Cargo-selected frozen executable SHA256 is `fca4b669…d61ca15`; the forward
artifact-binding correction preserves the unrelated historical probe mistake.
No device acceptance is attributed to the wrong probe.

Both GPUs pass native fixed-public bang/chat, original scalar output IDs,
pre-enqueue and queued native GDN/QSA cancellation, poisoned retry rejection,
fresh-owner recovery and four complete1,376-page/818,249,728-byte private audits
per run. `rust-native-model-initial-device-20261005.json` preserves the four
initial count8 reports and exact lifecycle observations. Counts/longer contexts,
full75/128 output, injected whole-model failures, options/quality, production
private workers, faster matched Ollama inference and native-default/managed
`ollama.` routing remain unaccepted. These short runs are diagnostic lifecycle
acceptance, not a performance claim.

The fixed-public protected runner binds the actual Cargo target/binary and checks
full receipt plus cancellation/cleanup completion; lab-only containment has no
bearing on production long-inference deadlines. Stage1 corrected Rust CPU expert
operators are separately source-reviewed and await root integration and real
CPU/GPU numeric/joint timing acceptance. **Required Vashti changes: none.** No
public contract, installed service/configuration, frontend, release, driver,
reset or reboot is changed by this internal checkpoint.

### Current packed Rust CPU expert checkpoint (2026-10-05)

**Root source `7950074`/`d1a0446`, after device/runner checkpoint `8280ec1`.**
Both corrected CPU commits are integrated; this supersedes the root-integration-
pending note above. Read
[CPU source acceptance](kuno-text-performance.md#original-packed-rust-cpu-expert-source-acceptance-2026-10-05),
the [crate API/limits](../crates/kuno-text/README.md) and the original decode plan.
`rust-packed-cpu-experts-source-20261005.json` pins root source/checks/budgets.

`GgufFile::positional_reader` borrows the already-inspected model/open descriptor;
`PackedProjection`/`PackedExperts` borrow that reader and original views.
Explicit `ExpertOperator` calls return guarded unweighted requested-rank values,
preserving all selected experts and original64-lane math. They are not called by
the native model or installed gateway. Source-only controlled CPU private storage
has an independent1-MiB cap; this does not admit a future whole-model aggregate.

Root passes105 core tests in both debug/release (two installed checks ignored),
369 composed native tests, strict all-target Clippy for both, formatting and
locked offline release. New composed lab binary SHA256 is `b1ba040f…5e8f589`;
prior physical native fixtures retain their separately verified `fca4b669…`
identity and are not attributed to the new binary.

Next source work is the approved bounded real-read one-layer CPU/GPU numeric and
joint-calibration lab in its own managed worktree based on `d1a0446`. It uses
actual original fills/consumers/completed fences, all-ten rank ownership and
one original-rank merge. Numeric acceptance precedes timing. Prepared-host CPU
timing is deferred: the current CPU API always performs positional reads, even
when OS file pages are warm. Root exclusively owns real model/GPU/service runs.
No CPU/GPU parity, hybrid model, speed win or production privacy claim follows
from these source checks. **Required Vashti changes: none.** Public HTTP,
installed services, drivers, release and frontend remain unchanged.

The short native partition matrix is now complete on both GPUs for counts
1/3/5/8, all original eight output IDs, four complete MAIN/source audits and
native GDN/QSA cancel/poison/fresh recovery. Exact reports remain pinned to the
earlier `fca4b669…` artifact in `rust-native-model-partitions-device-20261005.json`;
this is not newer CPU-binary or speed acceptance. The protected runner now also
accepts the already-compiled public75/128 rainwater fixture and compares against
committed original native IDs; full physical acceptance is still pending.

### Current full native fixture checkpoint (2026-10-05)

The root CPU-composed `b1ba040f…` artifact, actual `d1a0446` build snapshot
`3371e900…`, passes Intel's full75/128 public native fixture: all128 original
native IDs, scalar continuation, fresh full-prefix replay, queued native GDN/QSA
cancel, poisoned retry rejection and four complete1,376-page/818,249,728-byte
private audits. `rust-native-model-full-intel-20261005.json` retains the entire
report and exact identities. NVIDIA's full counterpart is now complete with
the same 128 original IDs, full-prefix fresh replay, native cancellation,
poisoned retry rejection and all four MAIN/source audits. Both its worker and
corrected outer runner exit 0; `rust-native-model-full-nvidia-20261005.json`
preserves that separate completed evidence under the same exact binary.

The first outer validator exited1 after the worker's success0 because it
mistook cumulative audited source bytes for simultaneous live usage. Source
arena reset accepts complete erasure/inspection and drops handles before adding
numeric counters.127 scalar continuations yield1,905 buffers/70,746,112 B
cumulatively, under the unchanged independently checked64-MiB live cap. The
forward-corrected runner requires exact original per-step cumulative counts;
root revalidated the completed original report and preserved both outcomes.
No engine limit, cleanup or numerical behavior changed. Full-logit/options/
quality/longcontext/fault/production-worker and speed acceptance remain pending.

The approved one-layer real-read CPU/GPU calibration plan and exact bounds now
live in `docs/kuno-native-text-cpu-gpu-calibration.md` and
`rust-cpu-gpu-calibration-plan-20261005.json`; implementation is isolated from
root GPU work. No model/private/default/API/installed/Vashti change follows
from that source plan. **Required Vashti changes: none.**

### Current actual CPU operator checkpoint (2026-10-05)

The integrated `d1a0446` CPU source now has actual original-weight numeric
acceptance on the Ryzen 5900XT. The fixed-public
`crates/kuno-text/examples/packed_expert_public_probe.rs` executes original
Q4 gate/up with Q5/Q8 down, experts 0/7, and explicit AVX2 physical plans
1/4/8/16. All 16 cases bit-match an independent scalar 64-lane row decoder for
raw gate/up and complete unweighted partials; inactive rank tails are zero.
Portable single-worker complete partials also match. Final file consistency
and the protected swap/core-limited user unit pass. Strict example Clippy and
locked offline release pass; unchanged libraries retain their 105 core/369
native acceptance. Exact source/build/public result evidence is in
`rust-packed-cpu-experts-actual-20261005.json`.

This supersedes the CPU-device-pending note for these limited public CPU cases;
all-ten CPU/GPU parity, completed joint timing and whole-model hybrid execution
are still pending. The new CPU helpers are not called by the native model or
installed gateway. No faster-than-Ollama or production private-worker claim
follows. Additional actual model context/failure hooks and the one-layer joint
calibration runner are isolated source candidates, awaiting root review and
physical acceptance. **Required Vashti changes: none.** Public contracts,
installed services/configuration, drivers, defaults, frontend and release are
unchanged; no reset/reboot is required for this lab work.

### Current whole-model failure fixture source (2026-10-05)

Root `19e12af`/`cdb3524` integrates the independently reviewed actual lifecycle
candidate and its explicit protected lab entry. Read
[the source checkpoint](kuno-text-performance.md#actual-model-failure-and-context-fixture-source-2026-10-05)
and `rust-native-model-lifecycle-source-20261005.json`. Twenty-two serialized
real owners exercise context4/8 partitions, exhaustion, malformed chunks,
late-combine error, real PLE-ticket error/panic and cancellation after the first
actual GDN commit, both retry rejections and fresh recovery. The fixture preserves
original scalar preflight exhaustion and explicitly closes it; native failure
must itself close. No state import, fake position, extra MAIN or relaxed
whole-model numerical tolerance is introduced.

Root passes 376 native tests, strict all-target Clippy, locked offline release,
format/diff and three missing-protection invocations rejected before device
creation. Frozen executable `110b9926…` retains exact Cargo/source provenance.
Physical both-card execution is pending. Cleaned full-logit mismatch is explicitly
unaccepted; process success alone cannot establish parity. The 1,800-second
protected lab containment has no effect on production long inference. Actual
long-context boundaries, options/quality, private worker/native routing and
matched Ollama speed remain pending. **Required Vashti changes: none.** Installed
services, drivers, public API, defaults, frontend and release remain unchanged.

### Current both-card model lifecycle result (2026-10-05)

The preceding physical-pending note is superseded for the fixed context4/8
fixture only. Frozen `110b9926…` completes all 44 actual owner cases across both
cards, real late error/panic/post-first-GDN-commit cancellation, retry rejection,
full original scalar/native MAIN and source audits, fresh recovery and physical
retirement=false. Both worker units exit 0. Exact reports are in
`rust-native-model-lifecycle-device-20261005.json` and
[the performance guide](kuno-text-performance.md#both-card-actual-lifecycle-acceptance-and-numerical-rejection).

All 24 final greedy comparisons equal the scalar reference, but full logits
have small deltas (Intel maximum about1.04e-5; NVIDIA about8.70e-6). Whole-model
bitwise numerical acceptance remains explicitly false and both outer runners
exit 1 after clean completion. No whole-model tolerance was relaxed; a source
investigation is pending. These simultaneous diagnostic runs make no speed
claim. Existing full128-output acceptance is a separate, narrower receipt.
Long contexts, options/quality, production private workers, native-default/
managed `ollama.` routing and faster matched Ollama inference remain outstanding.
**Required Vashti changes: none.** Installed/public/frontend/release state is
unchanged; no reset or reboot occurred.

### Paused native text source checkpoint (2026-10-05)

**The user paused the goal.** Read [the clean checkpoint](kuno-native-text-checkpoint.md)
first on resume. Root `codex/kuno-native-text` integrates the joint expert source
through `e386c79` (`5f2ed09`, `b2b69cc`, `1ef151f`, `e386c79`). Root passes 391
host/offline tests, strict all-target Clippy, formatting and diff checks. The
focused independent review of the corrected reference ordering found no
material blocker. No root CLI, physical joint numeric/timing acceptance or
whole-model hybrid dispatch is established. Numeric precedes timing; references
in Calibration now follow completed measured consumers/merge/audit while all
numeric/file/immutability gates still precede local plan publication. Required
priming/prior samples remain cache conditioning.

The isolated first-divergence diagnostic is frozen at `2439d73` on
`codex/kuno-native-numeric-boundaries`, based on `8621308`, with five focused
tests and strict Clippy/fmt/diff checks passing. It is not integrated or
registered on root. Source review passes; extra peer review and both-card actual
four-boundary/full-cleanup evidence remain pending. No arithmetic or whole-model
tolerance changed; the cause of the measured full-score deltas is unaccepted.

Resume with the checkpoint's exact candidate/source receipts and physical gates.
Preserve the 8-GiB correctness lane; separately address benchmark containment
and cache accounting before a matched Ollama comparison. No new model/GPU job,
installed service/configuration, default, public contract, driver/reset/reboot,
frontend or release change occurred while finishing this checkpoint.
**Required Vashti changes: none.** All current source workers and checks have
finished; goal completion and a speed win remain outstanding.


### Native text resumed numerical and host-limit diagnostics (2026-10-06)

The user resumed after the `~/dev` NVMe migration, then requested a break with
one Intel comparison already running. The latest checkpoint is
[the native text checkpoint](kuno-native-text-checkpoint.md); it supersedes the
historical source-only candidate status above. Finish only the declared Intel
comparison and pause; that comparison has now completed, with no next device job
started and no native lab job remaining. Resume requires the user's instruction.

On `codex/kuno-native-text`, root integrates `195d8a5` (author candidate
`2439d73`), named protected CLI `962c37d`, exact numeric validator `80aef9d`
(author `36d2121`) and the dedicated host-limit diagnostic `5ffffa4` (author
`13844b5`). Root passes 396 host/offline tests followed by the additional focused
CLI guard test, strict all-target Clippy, formatting/diff checks, and 22 focused
Python tests. The original correctness lane still requires actual 8 GiB;
performance uses a separate actual 8/52 guard with unchanged model/cleanup
validation. No guard output or numerical tolerance is rewritten.

Actual Cargo-selected/frozen binary `73ce13b2...` completes all eight fresh
scalar/native owners on each card, full owning maps and explicit audit/erasure,
with worker and validator exit0. Both cards have exact projected KEY/VALUE;
Intel's combined layer0 input differs by at most 7.45e-9 and NVIDIA's is exact.
Injected residual maxima are 1.49e-8/5.96e-8. KEY/VALUE depend on separate token
embeddings, so equality does not explain residual differences. PLE has definite
scalar CPU-f64/native GPU-f32 differences; source inspection alone cannot
attribute Intel's earlier residual drift. Whole-model bitwise and performance
acceptance remain false. Exact reports/build identities and the pre-GPU initial
launch-path failure are preserved in
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-device-20261006.json`.

The existing Intel public comparison uses one persistent user cgroup, one frozen
binary, native count 8 original 75/128 fixture and MemoryMax 8/52/52/8 GiB,
with swap/core0,
full cleanup/cancellation and numeric memory/refault/pressure/process-I/O evidence.
It does not flush global caches or change production limits. NVIDIA's comparison
is not started. All four Intel samples pass with zero OOM and launcher/supervisor
exit0; the unit finishes inactive after 1,046.203 seconds, swap peak0. Decode times
for 127 following steps are 128.939/80.438/63.189/130.799 seconds, respectively
(0.985/1.579/2.010/0.971 tokens/s). The 8-GiB limit repeatedly reclaims model file
pages; 52-GiB warm weight reads are much shorter while upload times stay similar.
Cache order still varies across samples. Exact public receipts/identities and
telemetry hashes are in
`apps/kuno/benchmarks/text/receipts/rust-native-host-limit-intel-20261006.json`.
This remains a host-containment diagnostic, not true hot-full-request or matched
Ollama performance acceptance. Correctness containment remains 8 GiB.

No production private worker, public API/catalog/native-default or `ollama.`
routing, install/configuration, driver/reset/reboot, frontend or release change
is made. **Required Vashti changes: none.** Long contexts/options/quality,
physical CPU/GPU expert calibration, private-worker/scheduler/catalog integration
and a faster matched Ollama comparison remain required goal work.

### Selected port takeover, CLI separation and intentional offload (2026-10-06)

The user wants `kuno run MODEL` to use Kuno while retaining ordinary
`ollama run MODEL` for direct Ollama inference. The original Ollama executable
has not been replaced: its default listener is currently Kuno on 11434, while
the original Ollama daemon is loopback 11435. A new command name alone does not
separate these routes.

**Selected direction:** keep Kuno on the adopted Ollama API listener (normally
11434) and put original Ollama on private loopback 11433 where available. The
user prioritizes unchanged API URLs. This supersedes the separate-default-port
proposal from `501f49e`; separate-listener manual setup remains an option.
`kuno` inference commands would select their own endpoint (explicit endpoint,
`KUNO_HOST`, installed configuration, then the normal Kuno API default),
independently of `OLLAMA_HOST`.
Convert wildcard bind addresses to usable client addresses. Mirror common
inference/model commands without changing the Ollama executable. Native selection
still requires a verified registered adapter; the paused text prototype is not
production-ready. Inference access remains keyless and separate from privileged
installation/control operations.

Use a Kuno-owned systemd drop-in for the original Ollama daemon's private bind;
do not edit/replace the vendor executable or base service file. Its service
environment does not configure terminal clients. Plain `ollama` still selects
its own `OLLAMA_HOST` or defaults 11434. **The later selected shell-export
contract below supersedes the command-scoped CLI helper proposal.** Applications
inheriting and honoring the export may also connect directly to Ollama; the
user accepts that behavior. Exact installation/activation support across shells
remains pending.

This host still uses original Ollama 11435. Preserve its actual recorded port
until an approved, tested migration; 11433 is not assumed free. Migration must
retain Kuno's listener/settings, image runtime, workers and state with rollback
and existing interruption-confirmation requirements. Clients keep their Kuno
URL. Ollama upgrades may still interrupt requests using that backend; there is
no guarantee against a later override explicitly changing its bind.

Direct Ollama commands bypass Kuno reservations. Treat that work as external GPU
consumption; observation alone cannot guarantee serialization of independent
direct Ollama and Kuno jobs. The requested CLI separation does not silently grant
Kuno control over ordinary Ollama commands.

**Approved direction:** support intentional mixed CPU/GPU offloading. The current
installed `worker_cpu_spill_detected` comes from the fully GPU-reported residency
requirement, not an upstream model error. There is no current per-model offload
opt-in; removing only that one check would leave other strict checks and placement
assumptions unresolved. The exact policy/configuration and resource-accounting
contract needs a reviewed implementation. Preserve protection against unexpected
spill for models admitted under GPU-only policy. The main specification now
records this intended capability, explicitly labeled unimplemented.

**Status:** proposal/requirement checkpoint only. Read-only listener/config/source
inspection passes; no CLI, API, runtime, installation, services, GPUs or client
settings changed and no implementation tests are claimed. The native performance
goal remains paused. Required Vashti endpoint changes: none for the selected
takeover layout. A tested backend CLI/offload/installation handoff must still
precede any application feature integration. Shared request schema and discovery
changes are not yet selected or implemented.

### Clark systemd-only CLI port experiment (2026-10-06)

The user authorized a temporary service-port change on Clark to test whether
ordinary Ollama CLI commands inherit the daemon's systemd environment. Clark
ran original Ollama 0.32.13, with no shell `OLLAMA_HOST`, alias/function wrapper,
service drop-in or Kuno listener. The original listener was 127.0.0.1:11434;
`ollama run lfm2.5-thinking` opened normally before the change.

A temporary runtime systemd drop-in set only the daemon's
`OLLAMA_HOST=127.0.0.1:11439`. After restart, `/api/version` and the actual
listener confirmed 11439; 11434 was closed. In the unchanged SSH shell,
`ollama run lfm2.5-thinking` failed with `could not connect to ollama server`
and exit status 1. The command-scoped positive control
`OLLAMA_HOST=http://127.0.0.1:11439 ollama run lfm2.5-thinking` opened normally
and exited with status 0 after `/bye`. No generation prompt was submitted.

Restoration removed only the temporary drop-in and restarted Ollama. The
original 11434 listener and version returned, 11439 closed, the base unit's
SHA-256 remained unchanged, and the service stayed active/enabled. Both service
and shell again had no `OLLAMA_HOST`. The restored plain CLI opened normally
and exited with status 0 after `/bye`. The independent automatic rollback timer
and temporary experiment state were removed after successful restoration.

This live test confirms separate daemon/client endpoint configuration on this
installation. It does not test model output quality or native performance.
Only this diagnostic and restoration were performed; Marge's installed ports,
application APIs and the paused native inference goal are unchanged.
**Required Vashti changes: none.** The selected CLI/port migration above remains
unimplemented.

### Selected persistent Ollama shell export (2026-10-06)

The user selected the ordinary Ollama configuration approach after the restored
Clark test: export `OLLAMA_HOST` in the selected user's shell startup, using the
actual recorded private Ollama client endpoint. This supersedes the shell
function/CLI-only helper recommendation, while retaining the selected Kuno
takeover layout. Do not replace or wrap the original Ollama executable.

The service's Kuno-owned systemd override sets the daemon's private bind. The
separate shell export sets the CLI target, for example
`export OLLAMA_HOST=http://127.0.0.1:11433` after that listener is verified.
Existing installations keep their actual private port until a tested migration;
this host's last recorded private port is 11435. Record the enrolled non-root
user so root/admin upgrades do not silently choose another account. Publish a
marked installer block only after endpoint verification, retaining prior user
settings while supplying the selected effective host. Update/remove only the
unchanged owned block, preserve unrelated edits, and report conflicts. Give
activation/restoration instructions: already-running shells and apps keep their
environment until explicitly changed or relaunched. Other users and
independently configured services are outside that shell export's scope.

Applications inheriting and honoring `OLLAMA_HOST` may bypass Kuno and use
original Ollama directly. **The user explicitly accepts this.** Apps explicitly
targeting Kuno's adopted listener continue to use Kuno. Direct Ollama work still
bypasses Kuno reservations and is external GPU consumption. Kuno's planned CLI
uses its own endpoint selection (explicit endpoint, `KUNO_HOST`, installed
configuration, default), independent of `OLLAMA_HOST`.
Never use `OLLAMA_HOST` as a Kuno endpoint fallback; normalize wildcard Kuno
listener addresses to usable loopback client addresses.

**Status:** contract/documentation update only; no shell exports, services,
installation, CLI, API or application behavior changed. The Clark connection
receipt above verifies command-scoped targeting, not persistent shell setup.
Supported-shell activation, installer upgrade/uninstall and CLI endpoint
separation still require implementation and tests. Documentation diff checks
pass; the native inference goal remains paused. **Required Vashti changes:
none** when its connector explicitly targets Kuno. No API schema change.

### Installed CLI separation and shell export receipt (2026-10-06)

The selected contract above is now implemented and installed on Marge. Backend
commits are `f1c2848` (independent Kuno compatibility client), `942f965` (persistent
Bash enrollment) and `686bc46` (installer/upgrade/restore integration, owned CLI
launcher, new-port selection and current-upstream version verification). These
are the focused CLI changes after decision `9e04bc1`; this receipt does not
accept or resume the native text prototype/performance work.

**Exact installed routes:** Kuno retains `[::]:11434`, private original Ollama
retains `127.0.0.1:11435`, and privileged control retains `127.0.0.1:11432`.
The enrolled `jmicko` Bash startup exports
`OLLAMA_HOST=http://127.0.0.1:11435`. Existing API URLs are preserved. New
installations prefer private 11433 when available; no existing-port migration
was performed. The original Ollama executable is unchanged.

`kuno` is available through an installer-owned `/usr/local/bin/kuno` launcher.
Its inference/model commands invoke the original Ollama client on `PATH`, using
prefix `--host`, then `KUNO_HOST`, then the listener from prefix `--config` or
the installed configuration, then the 11434 default if no configuration exists.
Wildcard listeners become loopback. `OLLAMA_HOST` never selects the Kuno route.
Streaming, arguments, terminal cancellation and exit status remain the client's
own behavior; no inference deadline is added. History defaults off; explicit
Ollama local save/history/editor actions remain inherited client features.

Install/upgrade preflight the recorded non-root account and startup ownership
before interrupting inference, then publish the managed Bash blocks only after
verified service activation. Explicit root installation can use `--cli-user`;
later upgrades reuse the enrolled account. `sudo kuno configure-cli [--user
USER]` repairs/enrolls without restarting services. Restore preflights and
removes unchanged owned blocks, retaining prior settings and unrelated edits.
Enrollment records contain only identity/owned blocks, never copies of user
startup contents. A late publication failure reports active inference plus
retry guidance; it is not mislabeled a service rollback. Shell file publication
rechecks snapshots immediately before replacement but cannot guarantee atomic
comparison against an uncooperative concurrent writer.

The old setup record contained Ollama 0.34.2 while the owned live server was
0.35.1. Upgrade health now compares the owned live gateway/private endpoints,
retaining executable/listener/environment ownership checks and the historical
record. An ordinary Ollama update no longer causes a stale-version rejection.
New artifact conflicts (including a pre-existing CLI launcher) are rejected
before stopping services.

**Validation:** `cargo test --locked -p kuno -p kuno-protocol` passes 216 tests,
with seven environment-dependent tests ignored. This includes 17 shell tests,
four CLI unit/six CLI integration tests, 13 deployment tests and current-version
health regression coverage. Strict all-target Clippy, formatting and diff
checks pass. The standard native-image release build passes against the
preserved verified CUDA/Vulkan closure; ELF checks reject private Intel
dependencies and RPATH/RUNPATH.

The prepared/installed binary SHA256 is
`6c247d9001619117f1cadd2a37499ccd97974c8806d83d0aac47b2f9b4583469`.
The normal transactional upgrade succeeds with both services active/enabled.
Semantic configuration, both disabled inference deadlines, original Ollama
binary and all 65 image-runtime members are unchanged. Fresh interactive/login
Bash exports pass; installed `list`, `ps` and the public Qwen model's `show`
pass for both commands. Live connection traces verify original CLI port 11435
and Kuno CLI port 11434. Repeated enrollment keeps startup file hashes and
service PIDs unchanged. Clark verifies Kuno's 0.35.1 upstream response through
`192.168.1.164:11434`. The current shared shell is activated too.

**Limits:** automatic shell setup is Linux/Bash only; existing processes retain
their environment until explicitly changed or relaunched. Applications
inheriting and honoring the export intentionally bypass Kuno scheduling.
Installed uninstall was not performed; restore/crash handling is covered by
isolated tests. No live generation, GPU benchmark, new image acceptance,
intentional-offload implementation, native text registration or public release
is claimed. **Required Vashti changes: none** for explicitly configured Kuno
connectors; inference API schemas and access controls are unchanged.

### Native text baseline variability follow-up (2026-10-06)

The user's interactive timing counters show decode rates of 3.58, 10.11 and
3.98 tokens/s for unequal-length responses, with reported pauses in the longest
response and separate warm observations up to 12.68 tokens/s. These are
unreproduced observations; only numeric aggregates, never private prompts,
outputs or topic summaries, are retained in the
[measurement update](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06).
The original 7.6–8.0 tokens/s fixture receipt is not a general Ollama ceiling.

The next comparison must match actual memory allowance and placement, distinguish
weight/file-cache warmth from KV prefix reuse, and include repeated short and
long public synthetic requests plus numeric within-stream gap and cache/I/O
telemetry. The existing client does not yet capture those gaps. Qwen's sparse
PLE lookups and expert routing make cache behavior worth measuring; they do not
prove the cause of these stalls. The benchmark fixture/receipts are unchanged.

**Status:** documentation-only benchmark-plan/checkpoint update; no engine,
API, installation, service or GPU operation. The native goal remains paused.
Diff checks pass; no runtime tests are claimed. **Required Vashti changes: none.**

### Native optimization priority amendment (2026-10-06)

The user superseded the expanded Ollama benchmark proposal above: **use 10
decode tokens/s as a working target and skip further/repeated Ollama baseline
runs**. Concentrate remaining usage on correct Rust-engine optimizations and
native before/after evidence. Historical receipts remain unchanged; 10 tokens/s
is not a newly reproduced Ollama measurement. Numerical, privacy, cleanup and
long-inference constraints remain in force. Native performance observations
must retain exact public fixture/seed, actual binary/source/settings, host
allowance and output counts; an operator win is not a whole-model win.

See the [amended checkpoint](kuno-native-text-checkpoint.md#current-priority-amendment-2026-10-06)
and [measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06).
The goal was still paused when recorded. Only documentation and a read-only
next-candidate source review are authorized preparation here; no GPU jobs,
production behavior, installation, API or application changes are claimed.
Diff checks pass. **Required Vashti changes: none.**

The user subsequently resumed the goal and refreshed sudo. Root now integrates
through `22a2413` on `codex/kuno-native-text`: protected native continuation,
public projection/profile controls, single-token MoE and GDN, actual queued
cancellation/recovery, narrow full-page CPU/GPU lab banks and bounded fixed
calibration and explicit Q4 header reuse. 426 host/offline tests and strict
all-target Clippy pass; 62 Python control tests passed at `3f38840`.
Both-card operators and `single-all` eight-token model/lifecycle
checks pass without a useful short-run speed gain. Both-card warmed original-read
CPU/GPU calibration completes12 numeric gates and24 samples per card with a
narrow operator benefit; see the
[current owner checkpoint](kuno-native-text-checkpoint.md#later-owner-acceptance)
for exact artifacts, failed-run evidence and limitations. No full-model throughput
win, private native worker, default routing or automatic placement is accepted.
The `22a2413` Q4 binary passes52 routed/14 packed-format cases on each card,
including all six paired generic/Q4Block comparisons with observed zero maximum
absolute error. Complete-model Q4 acceptance and throughput remain separate.
The count1 CPU-miss/GPU-hit plan at `dc440d8` is a proposal; a staged one-layer
numeric adapter and batched full I/O inspection are being developed in isolation.
Root owns shared
GPU/service operations; isolated agents perform source work. No further Ollama
baseline, API/Vashti change, installation or release is part of this work.
**Required Vashti changes: none.**

### Native upload-erasure acceptance follow-up (2026-10-06)

The subsequent internal `4d4ecbe` admission seam runs the same mandatory full
audit once after map lease drop, then retains ordered-source release, health,
telemetry and MAIN fence credit. Both-card short model/cancellation/recovery/
full erasure acceptance passes with439 native host tests and strict Clippy.
The [checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance)
pins the normal `3b192df…` binary and exact proofs. The longer Intel75/128
fixture now passes output/lifecycle/full-erasure gates at1.311 decode tokens/s
(127steps/96.865s), below the10-token/s target; it is not a matched performance
comparison. Root source through `97d50bc` passes450 native host/offline tests
and includes the explicitly reserved count1 CPU-miss seam. That seam remains
unexposed and physically unaccepted, with CPU joins before GPU enqueue and no
overlap/speed claim. Ordinary inference, APIs, installed services and Vashti are
unchanged. **Required Vashti changes: none.**

The internal `200b38a` upload-wipe batching candidate passes the same both-card
operator and complete eight-token native model/cancellation/recovery/full
erasure inventories as the audit-only candidate. Runtime fault tests now pass49;
native tests remain436, Python controls81 and strict Clippy/provenance pass.
See the [checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance)
for exact `3ce1dbd…` binary and numeric receipts. Intel short generation
observes4.916s for seven steps; NVIDIA9.418s, with no additional NVIDIA speed
gain established. Matched long-run throughput, single audited admission fence
and CPU-miss model integration remain separate. This internal lab change
requires no API, frontend, installation or release changes.
**Required Vashti changes: none.**

### Native audit/staged acceptance follow-up (2026-10-06)

Root integrates through `793c351` on `codex/kuno-native-text`. Full audit
copy/read-map batching and the bounded original-read staged CPU-miss/GPU-hit
numeric adapter pass both-card operator and eight-token model/lifecycle gates.
The exact `91d4dd9` binary,436 native/45 runtime host tests,81 Python controls,
strict Clippy, numeric inventories, erasure/cancellation/recovery receipts and
short timing observations are recorded in the
[current checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance).
Intel/NVIDIA short seven-step generation observes5.307/8.838s, with unchanged
outputs. This does not establish matched long-run throughput or the10-tokens/s
target. Upload erasure batching and explicit count1 CPU-miss model connection
are still separate source candidates. Shared GPUs/services remain root-owned;
no production, API, install, release or frontend changes are accepted.
**Required Vashti changes: none.**

### Paused native optimization checkpoint (2026-10-07)

The user intentionally paused further work to conserve usage after the active
block. Root `codex/kuno-native-text` source through `d13e95d` passes459 Rust
host/offline tests,99 Python tests and strict Clippy. The exact `0741420` /
`3098e9cb…` dense binary passes both-card91-case operators and complete short
model/cancellation/recovery/full-erasure gates. No additional throughput gain
is established. Grouped count1 grid trimming and the experimental CPU-miss
input-tail fix are source-tested but physically unaccepted; no hybrid CLI or
native-default capability is exposed. See the
[clean owner checkpoint](kuno-native-text-checkpoint.md#clean-paused-checkpoint-2026-10-07)
for exact commits, artifact/proof paths, limitations and preserved next proposals.
No GPU/compiler job is left running; installed inference admission is resumed.
No installed binary/configuration, API, release or application change occurred.
**Required Vashti changes: none.**

### User-authorized paused-goal optimization pass (2026-10-07)

The goal remains paused; the user explicitly authorized this additional turn.
`feb76b3` records both-card grouped count1 finite-grid acceptance. `79ba326`
moves owned original public GGUF weights into the ordinary upload queue without
redundant Vec copies; private data and ordered-source APIs are unchanged.
459 native/49 runtime tests, strict all-target Clippy, formatting and full
18-file patch reconstruction pass. The exact `9d4f4277…` normal binary passes
both-card52 routed/14-format/12-failure operators and complete short model,
fresh continuation, cancellation, poisoned retry/recovery and owning erasure.

The complete compiler-idle Intel52-GiB Q4 ABBA also passes all four75/128 fixture
and lifecycle gates. Generic mean1.4052 and Q4 mean1.4258 decode tokens/s give
only a small1.47% observed rate change; the10-tokens/s target is not met.
This comparison does not isolate the upload copy change, repeat Ollama baselines
or prove hot cross-request reuse. Preserve the dated receipts/artifact hashes,
physical postflight and next CPU-miss model gates in the
[latest owner checkpoint](kuno-native-text-checkpoint.md#latest-owned-public-upload-and-long-comparison-2026-10-07).
This supersedes the previous finite-grid source-only status, not the unaccepted
whole-model CPU-miss/default routing status. All lab jobs are stopped, admission
is resumed and both inference listeners respond. Intel physical VRAM is back
to50,688,000 bytes used; NVIDIA399MiB includes the desktop. No install, driver,
hardware reset, API/routing, application or release change was made.
**Required Vashti changes: none.**


### No-root native CPU-miss registration pass (2026-10-07)

The goal remains paused; the user authorized source/host work without sudo.
`a8cb5e0` on `codex/kuno-native-text` exposes only the protected fixed-public
whole-model `resident-count1-hybrid-chat` lab, with cap0/10, actual accepted
CPU-miss counters, fresh continuation and layer0/3 cancellation policy checks.
Its separate validator requires four1378-owner/818413568-byte audits and exact
original output/lifecycle proofs; ordinary1376-owner validation is preserved.
461 native host/offline tests,106 Python controls, strict all-target Clippy,
formatting, a normal release build and four actual predevice rejection cases
pass. Exact `1c44a37a…` artifact/host receipts and sudo-dependent next numerical,
hybrid late-fault and performance gates are in the
[latest checkpoint](kuno-native-text-checkpoint.md#no-root-cpu-miss-model-registration-checkpoint-2026-10-07)
and [registration contract](kuno-native-text-count1-hybrid-plan.md#protected-whole-model-registration-2026-10-07).
Whole-model hybrid device execution and speed remain unaccepted; CPU joins
before GPU enqueue, so no overlap claim. No GPU test, sudo, installed service,
API/catalog/routing, application or release change was made.
**Required Vashti changes: none.**


### Both-card short CPU-miss acceptance (2026-10-07)

After sudo was restored, `ff3c7d1` corrects a recursive admission mutex call
found in the first unaccepted hybrid constructor trial. Its borrowed original
construction proof preserves locking, binding/poison and all page checks.
463 native host/offline tests, strict all-target Clippy, formatting and a normal
`4382edf2…` release binary pass. Cap0/10 protected short whole-model tests pass
on both GPUs, including actual CPU work, exact eight output IDs, fresh next
output, layer0/3 cancellation, poisoned retry/recovery and all four1378-owner
zero audits/source0. Intel short decode stays5.440/5.448s; NVIDIA observes
9.056/6.849s (24.37% shorter). These single-order seven-step pairs do not
establish repeated/long throughput or the10-token/s target. Full-logit and
hybrid late-finality gates remain separate. Exact commits, failures, receipts,
remaining gates and postflight are in the
[latest checkpoint](kuno-native-text-checkpoint.md#both-card-short-hybrid-acceptance-and-constructor-correction-2026-10-07).
Admission is resumed, both live inference listeners respond and Intel physical
VRAM recovered with no classified Xe fault. No hardware reset/reboot, installed
binary/configuration, API/catalog, Vashti or release change was made.
**Required Vashti changes: none.**

### Pooled CPU-worker short acceptance (2026-10-07)

Source408abb9 adds explicit protected-lab worker reuse with unchanged math and
defaults.110 core,464 native and107 Python tests plus strict Clippy pass. Eight
compiler-idle scoped/pooled ABBA short samples pass on both cards with full
original output/lifecycle/erasure proofs: mean duration improves7.22% B70 and
8.31%3060. Full-logit, late-finality and long acceptance remain outstanding;
no10-token/s or production claim. Exact artifact/source/receipts and postflight
are in the [checkpoint](kuno-native-text-checkpoint.md#pooled-worker-short-abba-acceptance-2026-10-07).
Admission is restored and physical Intel memory recovers. No installed binary,
configuration, API, release or Vashti change. Required Vashti changes:none.

### Joined CPU/GPU overlap short checkpoint (2026-10-07)

Source44fd9bc adds explicit protected-lab pooled CPU/parent-GPU submission
overlap with bounded joined ownership and unchanged final math/publication.111
core,465 native and108 Python controls, strict Clippy and normal build pass.
Eight same-artifact short serial/overlap ABBA samples on both cards pass output,
actual CPU work, fresh/cancel/poison/recovery and owning-erasure gates. Observed
mean duration differences1.96% B70 /3.40%3060 fall within short sample variation;
overlap is not selected as a reliable speed win. Full-logit, late-finality and
long gates remain. Exact commits/artifact/receipts, source-contract details and
postflight are in the [checkpoint](kuno-native-text-checkpoint.md#joined-overlap-short-abba-checkpoint-2026-10-07).
Admission resumed, both inference listeners respond and Intel physical memory
recovers without classified faults. No installed service/configuration, public
API/catalog, application or release change. Required Vashti changes:none.


### Paused fully resident Ornith speed checkpoint (2026-10-07)

The user requested core execution speed work, then selected the already
installed `ornith:35b` to isolate it from Flash-Next offloading. On
`codex/kuno-native-text`, `3047d24` adds the Rust/WGPU resident Qwen3.5 MoE
lab; `c174aa9` completes its optional BPE flag/numeric reference handling.
The exact `6917d349…` frozen release artifact completes two75/128 public
sequences at13.9408/13.9184 decode tokens/s, versus Ollama83.3857 warm.
Only the first14 output IDs agree, so full-model numerical parity and Ollama
parity are not accepted. `c071019` prepares an explicit packed-block Q4/Q6
candidate, disabled by default and physically untested. Core118/native477,
the new offline shader inventory, Python118 and strict Clippy pass.

See [the current checkpoint](kuno-native-text-checkpoint.md#fully-resident-ornith-checkpoint-2026-10-07),
[Ornith contract](kuno-native-ornith.md) and their exact numeric receipts,
source/artifact identities and limitations. Earlier integrated Flash-Next
remains1.691tokens/s; neither target completion nor installed native routing
is claimed. The user requested a pause to use the GPUs and Vashti: all lab
jobs ended, GPU memory recovered, admission resumed and both inference
listeners plus Vashti respond. Goal remains paused. No installed binary,
API/catalog/configuration, application or release change occurred.
Required Vashti changes: none. Resume core arithmetic/performance work before
extended lifecycle/API/application work; do not repeat Flash-Next baselines.

### Resident text speed acceptance and Next transfer (2026-10-08)

The user returned the GPUs and authorized the approximately80-token/s resident
Ornith target. Backend source `000bdbe`, `fbddf49`, `3d8973e`, `5a93d1b`,
`26829f6`, `4a06b83` and `ffe1cf0` adds original-row probes, timestamp profiles,
explicit validated-geometry bounds selection, vector packed arithmetic, cached
parallel expert selection, EOS/teacher diagnostics and a shared Next operator
port. No public API, model registration, installed binary or configuration changed.

Complete B70 Ornith runs reach85.35/85.38 tokens/s on two fresh75/128 sequences,
82.6–83.0 on longer EOS-finished public prompts, and86.00/85.27 after the shared
kernel refactor. Every128-token native continuation matches the original scalar
path. Independent original-row and router checks pass. Teacher forcing128 public
Ollama tokens gives121 top1 and7 rank2 matches; free-running Ollama parity and
quality/indefinite stability are not claimed. The fast modes remain opt-in in
the fixed-public laboratory. Source shader suites and strict all-bin Clippy pass.

The original Flash-Next Q4 gate/up, Q8/Q5 down, F32 router and Q6 head slabs plus
512/top10 routing pass shared-kernel operator checks. These operators are not
connected to complete Next inference at that checkpoint: its last whole-model measured path remains
1.691 tokens/s. The user's next milestone is approximately10 whole-model Next
tokens/s before speculative FreeToken/Strata work. Avoid repeated Ollama
baselines or API/lifecycle/frontend work in this speed phase.

See [the current checkpoint](kuno-native-text-checkpoint.md#resident-speed-acceptance-2026-10-08),
[Ornith acceptance](kuno-native-ornith.md#resident-speed-acceptance-2026-10-08)
and [numeric aggregate](../apps/kuno/benchmarks/text/receipts/ornith-resident-speed-20261008.json)
for exact source/artifacts, timings, numeric tolerances and limitations. All
accepted GPU runs enforce swap0/core0 and explicit normal private-buffer clearing.
These are public fixtures; no real prompts, output text or routing trace was
persisted. Required Vashti changes: none.

At postflight all lab jobs exited, GPU memory recovered and Kuno admission was
resumed without an active request, queue or recovery latch. Both inference
listeners and Vashti respond. The broad goal controller remains paused and
incomplete; this receipt completes the directly authorized resident speed block.

### Complete Next shared execution receipt (2026-10-08)

The user authorized complete Flash-Next transfer after the resident Ornith
speed result. Kuno owner source `775d985`, `fc1e89e`, `d351bd6`, `62a52ba` and
`1a80d5b` connects shared packed dense/router and vector recurrent arithmetic,
batching of distinct privately owned GPU parameter records, exact CPU reduction
in SIMD registers, and optional direct caller device dispatch. All work stays
in the isolated native laboratory on `codex/kuno-native-text`. The current
[owner checkpoint](kuno-native-text-checkpoint.md#complete-next-shared-execution-acceptance-2026-10-08)
and [count1 contract](kuno-native-text-count1-hybrid-plan.md#complete-model-shared-execution-acceptance-2026-10-08)
supersede the preceding operator-not-connected and1.691-current statements.

Complete B70 inference now measures3.3473 tokens/s with standard banks and
3.5006 with96 slots, using original Ollama weights, seed424242/temperature0
and unchanged128 native output IDs. The selected frozen normal Cargo release
artifact has explicit `direct-device` feature, source `1a80d5b`, SHA256
`fe49069954829b2f8514b889b66967564937a67fdaabffe60a58418d437b8ec2`.
Its127 following steps take36280ms; prefill65978ms is slower than the standard
sample's48254ms. The10-token/s target remains unmet. Single ordered samples
do not establish statistical or matched Ollama speed; none was repeated.
The CPU-Q8 trial fails original-ID validation and is excluded from selection.

119 core tests (3 ignored),479 normal/481 direct native tests, two additional
offline shader suites each,50 imported runtime tests,22 hybrid Python controls,
strict all-target Clippy, formatting and provenance pass. Accepted physical
fixtures pass startup arithmetic probes, fresh continuation, existing
cancellation/poison recovery and complete1378-owner/818413568-byte normal MAIN
zero audits/source0, without context retirement. All runs enforce swap0/core0.
No real prompts, outputs or route traces enter persistence. Exact changed
commits, artifacts, all13 samples, tests, rejected/profiling limits and postflight
are in
[the numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-shared-speed-20261008.json).

No installed binary/configuration, API/catalog/default routing, application
or release change occurred. **Required Vashti changes: none.** Production
private workers, arbitrary quality/full-logit/long-context/soak acceptance,
mixed-card execution and the10-token/s milestone remain outstanding. Next
work should focus on CPU expert execution and remaining GPU dispatch/readback
handoffs. Cycle profiling and page-cache counters identify candidates, not
proven wall-time causes; the full notes retain those distinctions.

All lab jobs exited; admission is resumed with no active/queued request or
recovery latch. Kuno11434/direct Ollama11435 and Vashti7771 answer HTTP200,
both inference listeners have no resident model, and B70 physical memory
returns50688000 bytes. Same-boot Xe fault classifications remain zero. Goal
controller remains paused and incomplete; this directly requested trial is done.

### Active overnight Next offload optimization checkpoint (2026-10-08)

The user authorizes the isolated backend speed run to reach at least10 decode
tokens/s, or stop safely if sudo expires and effective GPU work is blocked.
Owner remains `codex/kuno-native-text` in the Kuno worktree. Through5ee51a5,
39 accepted public fixtures include diagnostic samples; current best v30 source
9fbda89 / frozen normal artifact3c7346fb119a96ccd631e2a96eec2c4a199d9fc5f46ffd6250311471ba310044
measures17724ms/127 following steps (7.1654 tokens/s). Target is unmet.
Original128 IDs, fresh continuation, poison recovery and complete owning/lifecycle
gates pass. B70 returns50688000B between runs; current classified Xe faults are
zero on the same kernel38 boot. Single samples are not statistical acceptance.

Current candidate changes only full-capacity heap erasure and guarded readback
copying, with496 native tests/two shaders,57 runtime controls/three allocator
controls/two logger controls and strict Clippy passing. Device acceptance is
pending. See the owner checkpoint and count1 contract for exact constraints.
The numeric aggregate `next-offload-overnight-20261008.json` is pending commit
until the run checkpoint is finalized. Admission is paused only during this
exclusive lab block; restore it and verify memory/listeners at an actual stop.
No installed binary/configuration, model blob, API/catalog/default routing,
Vashti application or release changes. **Required Vashti changes: none.**

### Overnight Next continuation: inline readback accepted

Ownerb345f9c/frozen3e707443f0452d00f9b4bc24608efbad406a2fde1f24bc58acc0eb6d658ef709
passes all original128 generated IDs, independent probes, fresh/poison/full
1378-owner cleanup and protected lifecycle checks.127 following steps17047ms
(7.4499912 tokens/s), prefill48856ms; single sample,10-token/s target still
unmet. B70 returns50688000B; same-boot classified Xe faults zero through
2026-10-08T14:50:40.509316+00:00.45 accepted fixtures (including excluded
profiles) are recorded in the pending numeric aggregate. The all-size GPU
Upload clear experiment is withdrawn after a deliberately stopped slow run;
ordinary volatile Upload erasure stays selected.

The next lab candidate partitions recurrent key reductions within a bounded
workgroup. Independent numeric/invalid-column probes and original unchanged
128 IDs remain required; device acceptance is pending. No installed services,
API, routing, frontend or release changes. Required Vashti changes: none.

### Current overnight optimization continuation

The Kuno owner remains `codex/kuno-native-text`. Following inline readbacks,
sourceaa3681c partitions recurrent reductions (full original acceptance, no
measured overall gain);07308b8 pairs existing expert-bank shards;ceb88d9 adds
a separately compiled, explicitly selected joined work-stealing CPU row mode;
7c40189 shares bounded original Q4 activations. Best accepted artifact
9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
completes127 steps in16247ms (7.8168277 tokens/s), prefill15464ms. All unchanged
128 IDs, independent arithmetic/guard checks and owning/fresh/poison/lifecycle
gates pass. Single samples only;10-token/s goal remains active and unmet.

The pending candidate uses fixed16-lane Q4 row groups and actual64-lane tail
coverage.498 native controls/two shader suites,128 core controls (three ignored)
in ordinary and work-stealing modes,39 Python controls and strict Clippy pass.
Complete candidate device acceptance is pending. The aggregate contains49
accepted fixtures, including excluded profiles. B70 returns50688000B; same-boot
Xe classifications remain zero through2026-10-08T15:25:46.933226+00:00.
No installed services/configuration, original model blob, public API/catalog,
Vashti application or release changes. Required Vashti changes: none.


### Active handoff: overnight native text run paused (2026-10-08)

This checkpoint supersedes pending-candidate/active-goal statements above.
Owner remains `codex/kuno-native-text` in the Kuno worktree. User paused for a
debrief; no further benchmark work is scheduled. Best accepted v44 source
7c40189/frozen9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
measures7.8168277 tokens/s (127 steps16247ms), prefill15464ms. Target10 remains
unmet. Fixed16 Q4 and the final active-GPU-group candidate pass original device
acceptance but are slower; both are withdrawn and native/core source is restored
exactly to v44. Aggregate `next-offload-overnight-20261008.json` records51
accepted public fixtures, including excluded profiles. Source tests, exact
commits, artifacts, acceptance limitations and restoration are in the owner
checkpoint. Single samples do not establish general speed or soak acceptance.

All benchmark jobs exited. B70 physical memory returned50688000B; same-boot
root Xe fault classification is zero through2026-10-08T15:45:04.155894+00:00.
Public Kuno admission is resumed, with no active/waiting request or recovery
latch. Kuno11434, direct Ollama11435 and Vashti7771 return HTTP200. Installed
binaries/settings, model blob, API/catalog/default routing and Vashti application
are unchanged. No release/install performed. **Required Vashti changes: none.**


### Active handoff: Astra native-text pass complete (2026-10-08)

This supersedes the preceding paused overnight checkpoint and pending native
candidates. Owner remains `codex/kuno-native-text` in the Kuno worktree.
Selected engine `79739d174af41990258a9571c1a8e720d77edc8a` measures
9.484–9.550 decode tokens/s across two full fixed-public fixtures, up from
7.817. The approved 10-token/s reference is still unmet; no new Ollama baseline
was run. Request-owned I/O reuse with completed Readback clears (`e9d8032`),
bounded INFO buckets (`406cf80`) and map polling (`a5a0d34`) supply the main gain.
Three completed recycle checkpoints and the original 512-entry INFO limit stay
selected. Scratch/full-copy cleanup (`9b51ace`, `708f5fe`) and byte-equivalent
zero-tail uploads (`fbae6f7`, `79739d1`) are retained; separate throughput gains
from those cleanups are not established. The user specifically selected the
zero-tail change: about 87% fewer encoded upload-copy bytes, after prefill only,
with no startup or measured hardware PCIe-traffic claim.

See [the final checkpoint](kuno-native-text-checkpoint.md#astra-pass-complete-selected-source-and-restored-host-2026-10-08),
[count1 contract](kuno-native-text-count1-hybrid-plan.md#request-scoped-zero-tail-upload-2026-10-08-retained)
and [14-fixture numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-request-io-reuse-20261008.json)
for exact source/artifact hashes, tests, acceptance and exclusions. Original
128 IDs, numerical/guard checks, fresh continuation, mid-token cancellation,
poison recovery, full owned-memory erasure and protected lifecycle checks pass.
Source gates: 498 native plus two shader suites; 131 core (three ignored) in
both feature modes; 90 runtime plus three allocator/two logger fixtures;
42 Python controls and strict Clippy. Broader quality, long-context, soak,
mixed-card execution and production-worker integration remain outstanding.

All lab jobs exited; B70 returned 50688000 bytes. Same-boot root Xe classifications
remain zero through `2026-10-08T18:16:29.880861+00:00`. Public admission is resumed
without active/queued work or a recovery latch. Kuno 11434, direct Ollama 11435
and Vashti 7771 return HTTP 200. Installed binaries/configuration, original
model blob, API/catalog/default routing and Vashti remain unchanged. No release
or installation performed. **Required Vashti changes: none.** Goal remains
paused; no further physical work is scheduled.


### Active handoff: private startup uploads and upstream review complete (2026-10-08)

This supersedes the previous selected-source handoff. Kuno owner source
`3b701d4d9bbeb794351502022b21c30734d501eb` includes runtime`909ab3b` and separate
bind/prefill transfer measurement. Zero-tail planning now covers all configured
private uploads, without extending request reuse or changing any erasure gate.
Public packed model-weight uploads are unchanged. Full original128-output and
owning/lifecycle acceptance passes: bind1365ms, prefill15324ms,127 decode steps
13237ms (9.5943 tokens/s). Prefill private copy payload falls75.12%; bind saves
no copy bytes in this fixture. No additional throughput/startup-time gain is
established, and the10-token/s reference remains unmet.

See [current checkpoint](kuno-native-text-checkpoint.md#private-startup-uploads-accepted-2026-10-08),
[contract and pinned upstream comparison](kuno-native-text-count1-hybrid-plan.md#updated-ollamallamacpp-comparison-2026-10-08-research-only),
and [numeric receipt](../apps/kuno/benchmarks/text/receipts/next-startup-zero-tail-20261008.json)
for exact source/artifacts, tests and limitations. Tests pass498 native plus two
shader suites,92 runtime plus three allocator/two logger fixtures,43 Python
controls and strict Clippy. Next recommended experiment is bounded request-local
reuse of identical GPU binding objects; not implemented or timed. Both wgpu and
llama already pool descriptor storage, so this targets repeated object creation,
validation and set allocation/update rather than missing pool support.

All lab jobs exited; B70 returned50688000B; Xe classifications remain zero
through2026-10-08T18:43:11.889831+00:00. Admission resumed without active/queued
work or recovery latch; Kuno11434/Ollama11435/Vashti7771 answer HTTP200. Installed
services, model blob, public API/catalog/default routing, Vashti and release
artifacts remain unchanged. Goal stays paused. **Required Vashti changes: none.**


### Active handoff: ranked native optimizations complete (2026-10-08)

This supersedes the startup-upload handoff and its pending binding experiment.
Kuno owner stays `codex/kuno-native-text` in the dedicated Kuno worktree.
All three user-approved candidates were tried; bounded request-owned binding
reuse `37fe3ca` is selected, with original GDN/float arithmetic restored in
`89ea1bc03ab22c1e834363d46c2736627d03b8da`. GDN retained-state trial `f73970f`
and existing three-/two-pass residual Q8 paths pass original acceptance but
provide no further timing improvement and are excluded. No new CPU arithmetic
source change, public API/catalog/default routing or production-worker change.

Selected normal locked artifact
`04f01a611209fb1bb526d48613deade7e2ec7413b9d6b69e7f03de1b9ea61d8b`, features
`direct-device,cpu-fma,cpu-work-stealing`, measures 10.6428/10.7300 decode
tokens/s on two full fixed-public Qwen3.8 Next fixtures; same-binary binding-off
control measures 9.5317. The user's 10-token/s reference is met on this fixture.
No new Ollama baseline or general quality/context/soak acceptance is claimed.

See [current checkpoint](kuno-native-text-checkpoint.md#ranked-optimization-pass-complete-2026-10-08),
[exact internal contract](kuno-native-text-count1-hybrid-plan.md#ranked-optimization-selection-2026-10-08-accepted-laboratory-only)
and [six-sample numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-ranked-optimizations-20261008.json)
for exact changed commits/artifacts, controls, tests and limitations. All six
fixtures pass original 128 IDs, numerical/guard/fresh/cancel/poison checks and
full owned-memory/lifecycle acceptance. Source gates: 498 native plus two shader
suites; 97 runtime plus three allocator/two logger fixtures; 44 Python controls;
131 core controls (three ignored) in the two-pass/work-stealing combination;
strict release all-target Clippy, formatting/provenance and whitespace checks.

All lab jobs exited; B70 returned 50688000 bytes. Same-boot Xe classifications
are zero through `2026-10-08T19:21:59.447749+00:00`. Admission resumed with no
active/queued work or recovery latch; Kuno11434/Ollama11435/Vashti7771 answer
HTTP200. No GPU reset, reboot, installed service/configuration change or release
performed. Goal remains paused. **Required Vashti changes: none.** This is an
internal standalone laboratory selection, not a production inference handoff.

### Active handoff: placement trials and secondary bank calibration complete (2026-10-08)

This supersedes the ranked-optimization handoff. Kuno owner remains
`codex/kuno-native-text` in the dedicated worktree; current engine source is
`34cdeaf71d8b29ffe186439a35e77b19be1150af`. Original intel144/cap10, float/FMA
and binding256 remain selected. Fused GPU projections, extra resident CPU work,
packed integer projections and binding512 provide no matched warm gain and
remain excluded lab opt-ins. Latest complete original128-output fixture passes
all numerical/owning/fresh/cancel/poison gates at10.9700 decode tokens/s, with
zero warm disk reads/writes and73.08% of expert calls served by B70 residency.

The user's next priority is complementary stationary experts on RTX3060 before
adaptive swapping. A protected original-expert calibration passes on that card;
residency avoids most upload cost but does not always beat the optimized CPU
path. **Full two-device execution is not implemented.** Proposed48 experts per
layer (disjoint from B70),32+16 shards/9GiB total, will need optimized native
kernels, concurrent scheduling, exact rank/source proofs, both-device lifecycle
and numerical acceptance, full-model timing and physical recovery. No private
expert histories/routes are persisted. Current warm results do not justify
adding16GB RAM. Broader quality/context/soak and production integration remain.

See [current checkpoint](kuno-native-text-checkpoint.md#placement-trials-and-secondary-resident-calibration-complete-2026-10-08),
[internal contract/results](kuno-native-text-count1-hybrid-plan.md#secondary-resident-calibration-results-and-next-placement-2026-10-08),
[ten-model-sample receipt](../apps/kuno/benchmarks/text/receipts/next-placement-kernel-trials-20261008.json)
and [operator receipt](../apps/kuno/benchmarks/text/receipts/next-resident-expert-calibration-20261008.json)
for exact commits/artifacts, failures, timings, tests and limitations. Source
checks pass502 native plus two shader suites,49+3 Python controls,98 unchanged
runtime-host controls, strict release Clippy and formatting/whitespace checks.

All lab jobs exited and physical memory recovered on both cards; no classified
Xe/NVIDIA faults this boot. Public admission is resumed without active/queued
work or recovery latch. Kuno11434/Ollama11435/Vashti7771 answer HTTP200. No GPU
reset/reboot or installed-service/configuration, public API/catalog, frontend,
release or installation change. **Required Vashti changes: none.** This remains
an internal standalone laboratory checkpoint, not a production handoff.

### Active handoff: stationary secondary model accepted, performance excluded (2026-10-08)

This supersedes the preceding pending two-device handoff. Kuno owner remains
`codex/kuno-native-text` in the dedicated worktree. Connected engine source is
`3e9f0e4891c75a09ff40eddd47f167e7ae47bef9`; stricter runner partition/provenance
validation is `bfdd986`. The complete range after `1e16862` implements the
secondary bank,32+16 geometry, status/phase admission corrections and counters.
Public API/configuration/catalog and Vashti consumers are unchanged.

The stationary48-expert/layer NVIDIA bank passes the complete original128-ID,
independent Q5/Q8 boundary-row, owning, fresh, cancel and poisoned-retry checks.
Both accepted secondary fixtures require five exact1835008B/nine-owner closes
and five independent row probes. Primary private budgets are unchanged; enqueue
and post-CPU-worker join are separately bounded phases. Frequency, selections
and partials are transient and erased; no private routing history is persisted.

**Keep secondary off.** Same-artifact v69 decode is11.0406 tokens/s for original
intel144/cap10 float/CPU FMA binding256, versus8.2586 with NVIDIA. Coverage rises
73.0791% to79.4767%, saving3900 CPU expert calls, but completion/readback and
runtime reuse add more cost than that saving. No warm disk reads/writes or
expert weight uploads. Extra RAM is not justified by these warm results.
Next secondary performance work needs cheaper bridges/checkpoints; more
residency alone is not evidence of faster inference. No new Ollama baseline.

See [latest checkpoint](kuno-native-text-checkpoint.md#stationary-secondary-model-acceptance-2026-10-08),
[exact internal contract](kuno-native-text-count1-hybrid-plan.md#stationary-secondary-model-acceptance-2026-10-08)
and [four accepted/six rejected numeric receipts](../apps/kuno/benchmarks/text/receipts/next-secondary-stationary-model-20261008.json)
for commits/artifacts, timings, failures and limitations. Checks pass505 native,
two one-test shader suites,50 Python controls, strict release all-target Clippy,
formatting and whitespace checks; runtime source is unchanged. This is a
standalone laboratory, with broader quality/context/soak and native-text
production integration still outstanding. No installation/release claim.

Normal admission is resumed; lab jobs/models are gone. Final same-boot snapshot
at2026-10-08T22:42:05.444314+00:00 reports B70 physical50688000/34242297856B,
NVIDIA373MiB desktop allocations without compute processes, zero classified
Xe/NVIDIA faults, andHTTP200 from Kuno11434/Ollama11435/Vashti7771. No GPU
reset/unbind/reboot/clock/driver or installed-service/configuration change.
**Required Vashti changes: none.** No shared contract or frontend handoff is
needed for this internal opt-in benchmark path.


### Active handoff: full adaptive cache awaits device benchmark (2026-10-08)

Historical; superseded by the adaptive settling/disk handoff below.

User corrected the 16-slot experiment: unused pinned experts must not prevent a
new working set from settling into GPU memory. Latest Kuno lab source
`3d67dbdd3c331b779779406a0fd177d3f459a4cd` makes all144 slots eligible, learns
prefill and accepted decode use, decays frequency with16-token half-life,
protects current selections and retains bounded grouped future-use transfers.
Earlier source commits: `b0c7652` bounded cache, `6a30bc7` empty slots,
`990a1a7` grouped fills/source hints. Read the exact internal contract in
[kuno-native-text-count1-hybrid-plan.md](kuno-native-text-count1-hybrid-plan.md#user-correction-all-slots-adapt-to-frequency-and-recency-2026-10-08)
and the latest top-of-file [checkpoint](kuno-native-text-checkpoint.md).

Full-cache host checks pass513 native, two one-test shader suites,52 Python
controls, strict release all-target Clippy and formatting. Device performance
and complete model/lifecycle acceptance are **pending**. Earlier16-slot paths
passed original gates but were slower than the matched fixed cache; they remain
historical comparisons, not a selected optimization. Native lab only; no
installation, release or public API/config/catalog changes. Required Vashti
changes: none. Normal admission is resumed; no GPU benchmark remains running.
The user requested a reasoning-mode switch before further GPU trials.

### Active handoff: adaptive settling and disk research (2026-10-08)

Historical; superseded by the exact replay handoff below.

Kuno laboratory integration is `8d25a9d` on `codex/kuno-native-text` in the
dedicated Kuno worktree. Source `cbfa8d1` corrects fractional token-time decay,
retains transient frequency/recency across all144 slots and adds optional
original PUBLIC writes into retained GPU slots. Original acceptance passes in
fixed/copy/direct modes; short matched decode is10.9238/10.4829/10.5202tokens/s,
with no demonstrated direct-write speed gain. Long diagnostic source is
`569c8e2`; runner acceptance is `8d25a9d`.

Both v75 separate1024-step public continuations pass full closes and exactly
matching greedy IDs. Last128 steps reach7.475 fixed versus9.405 adaptive
tokens/s; adaptive expert coverage reaches89.18%. Disk reads taper substantially
but total20.922/20.430GiB. This repeated teacher-forced workload is diagnostic;
the ordered pair has a file-cache warming confound. It does not prove arbitrary
prompt performance or repeated hot-page thrashing. Read the
[current checkpoint](kuno-native-text-checkpoint.md),
[disk research](kuno-native-text-disk-access.md), and
[verified numeric receipt](../apps/kuno/benchmarks/text/receipts/next-adaptive-steady-20261008.json).

Checks pass519 native tests plus two one-test shader suites,69 targeted Python
controls, strict release all-target Clippy and formatting. The accepted frozen
v75 SHA is `ecdb3a5f231d1a81b91398ebcc39378a659d96fce2d3358682da9b78f312a882`.
Exact trace/preload and runner candidates `990c52d`/`0740b6b` remain isolated,
unintegrated and unmeasured; root orchestration/main is still needed. Preserve
their clean review worktrees until integration. Do not package these candidates
as accepted runtime changes.

Normal admission is resumed; all lab jobs exited. B70 physical memory returned
to50688000bytes, NVIDIA has no compute process, classified Xe/NVIDIA faults are
zero and all three services returnHTTP200. No reset, unbind, reboot, install,
release or API/configuration change. **Required Vashti changes: none.**

### Active handoff: exact native-text residency replay and quant research (2026-10-08)

Kuno lab source `aef04f2` integrates `d45dc4b` trace/preload and `366d847` protected
runner; fast-report research is `a5d2827`. v76 frozen SHA:
`7e18edff2d5413b48b834f391f6637497bf5d157acd3507542afa2602887c8a7`.
`cpu-144-public-oracle-v76` passes every original output/lifecycle/owning gate
and both extra full training/replay closes. The correct seven-step replay
reaches11.74249tokens/s with3360 GPU calls and zero CPU work, expert uploads or
observed physical disk I/O. All eight original prefix IDs match. This is a
short conditional current-engine residency ceiling, not a hardware maximum;
loading/prefill/preload are excluded. No router/expert substitution.

Read the [latest checkpoint](kuno-native-text-checkpoint.md),
[internal contract](kuno-native-text-count1-hybrid-plan.md#exact-short-resident-replay-diagnostic-2026-10-08),
[primary-source speed/transfer research](kuno-native-text-disk-access.md#bidirectional-cache-transfer-and-published-fast-reports)
and [verified numeric receipt](../apps/kuno/benchmarks/text/receipts/next-public-oracle-20261008.json).
The user now suggests a smaller quant. Full83.6GB IQ3_S support is assessed in
the disk research: split loading and seven additional mixed-format paths need
implementation plus a separate reference fixture. No payload was downloaded;
no smaller quant or speculative decoding has been benchmarked here. Published
Strata/FreeToken-Kai results use materially different quantization, host
residency and often MTP; they have not been reproduced on this system.

Checks pass525 native tests, two one-test shader suites,181 Python controls,
strict release all-target Clippy, formatting and whitespace. Normal admission
is resumed, all lab jobs exited, B70 returned to50688000physical bytes, NVIDIA
has no compute process, classified faults remain zero and all three services
returnHTTP200. No installed binary/configuration, API/catalog, model-file,
Vashti, release, reset/unbind/reboot or driver/clock changes. Expert histories
remain transient. **Required Vashti changes: none.** Review worktrees are
integrated, reviewed and archived. Accepted receipt and quant assessment are
committed in `3601124`. No shared API change is needed.

### Active handoff: native Unsloth quant and MTP implementation (2026-10-08)

The user approved Unsloth UD-Q3_K_XL with the self-contained Q8 MTP head and our
own Rust implementation. Their backend direction is native Kuno where a
supported backend exists, with ordinary Ollama fallback for unsupported models.
Exact future routing/naming must be coordinated before changing consumers.
No production routing or API change is implemented in this handoff.

Historical foundation at `bb84f38`, superseded by the accepted foundation
handoff below. Its source followed `ae97ac4` (bounded full scalar MTP
reference), `cd0c2a1` (original packed IQ3_XXS/IQ4_NL/IQ4_XS decode; isolated
source `1ffd041`), and `712b605` (strict split owner; isolated `ce964280`). Read
the [exact native MTP plan](kuno-native-text-mtp-plan.md) before further changes.
The original Q4 acceptance and receipts remain unchanged.

The Q8 sidecar's complete checksum passed. Its strict34-tensor binding and
complete graph scheduling pass a synthetic executor check; real CPU weights
run on fixed public token0/zero-wide-hidden input and repeat exactly after KV
rewind. Six math/ownership tests cover per-stream embedding-first conditioning,
Q/gate order, independent dense GQA equations, contiguous KV grouping, all
visible positions and wiping helpers. This does not establish independent
full-head numerical parity, target/draft integration or speculative speedup.
The scalar reference remains public-fixture-only because inherited operators
lack complete private error-path erasure acceptance.

Merged foundation checks pass148 kuno-text tests plus two opt-in downloaded-head
tests,526 native laboratory host tests, strict crate/native Clippy and formatting.
Packed IQ reference fixtures contain8032 exact F32 values from pinned upstream
equations. Original optimized/GPU IQ admission is still rejected at this root
commit. GPU quant operator support and explicit Q3 expert/PLE ownership are being
implemented in separate review worktrees. Larger model shards are downloading;
complete hash verification, local no-copy Ollama import and actual native main
graph admission remain pending. No Q3 generation or MTP GPU performance claim.

Normal admission and all installed services are unchanged. There are no lab GPU
jobs/models at this checkpoint; protected GPU tests will use the restored sudo
credential once operator source is ready. No reset, unbind, reboot, driver/clock
change, installation or release. Model provenance and synthetic receipts only;
private prompts, outputs and expert histories remain unpersisted.
**Required Vashti changes: none for this internal milestone.**

### Active handoff: original-packed Q3 foundation and target/MTP coupling (2026-10-08)

Kuno source `4b29087` integrates strict split loading (`712b605`), scalar/GPU IQ
operators (`cd0c2a1`/`c249ed2`), actual Q3 expert/PLE ownership (`5c231b5`),
fixed4/32-position full-logit diagnostics (`42906b8`/`e602916`), bounded8GiB
ordinary cache (`d682e15`), original packed CPU SIMD (`d34c837`), actual guarded
target carry (`ff028b6`) and real CPU target/MTP coupling (`97772b3`). The IQ4
literal candidate needed corrected CubeCL explicit assignments (`4b29087`);
both physical GPUs accept its generated-shader and independent operator checks.
Read the [exact MTP plan](kuno-native-text-mtp-plan.md),
[latest checkpoint](kuno-native-text-checkpoint.md) and
[numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

All four pinned Unsloth payload hashes pass. Main89,986,353,824bytes is imported
as `qwen3.8-flash-next:125b-a6b-ud-q3_k_xl` through Ollama's local split create,
using CAS hardlinks and no second payload. The4,137,429,120-byte Q8 sidecar is
hardlinked outside Ollama's GC namespace; it is not registered as an unsupported
Ollama draft. This model import does not enable native HTTP routing.

Both cards pass16064 independent decoded C values,117 projection/MAX8 cases,
12 cooperative and12 subgroup fault cases with complete operator owning audits.
The compiler-idle32-position Q3 B70 run passes every full logit, exact greedy,
owning/fresh/recovery gate:31 decode steps in4250ms (7.294tokens/s), no ordinary
uploads and1,323,008 observed physical read bytes. The same-artifact cap0
control fails full logits at position23 after greedy equality; it is rejected,
with no accepted timing and no weakened tolerance. One-token prefill leaves
134/144 bank slots empty per layer; explicit bounded filling is a pending
candidate. Preserve this failure and pending status rather than claiming all
Q3 GPU-placement paths accepted.

The CPU target/MTP test uses actual pre-head h0 at draft position1. Full target
logits and both draft hidden/logit vectors repeat exactly. This is executor
parity/repeatability, not an independent whole-head quality oracle. The CPU MTP
reference and generic nonresident GPU probe lack complete private error-path
owning-erasure acceptance and remain fixed-public-only. No speculative speedup,
native production fallback/routing, private MTP or fully GPU draft claim.

Merged host checks pass157 core tests with10 intentionally ignored opt-ins,
547 native main tests with one ignored, plus two one-test shader suites.
The protected real target/MTP opt-in also passes. Strict featured Clippy and
formatting passed the preceding integrated candidates; final new candidates
must repeat relevant checks. See source-pinned receipts for exact artifacts,
limits and test boundaries. No reset/unbind/reboot, clock/driver changes,
installation, release, public API/config or Vashti changes in this block.
Normal admission and all three services recover, both GPUs return to baseline,
and classified fault counters remain unchanged after every accepted/rejected
trial. **Required Vashti changes: none.** The future native catalog/routing
contract still needs coordination before changing consumers.


### Active handoff: Q3 routing diagnosis and correct-expert replay paused (2026-10-08)

This supersedes the preceding Q3 foundation checkpoint for new backend work.
The user requested a clean pause to use the GPUs. Root Kuno source is `9218638`
on `codex/kuno-native-text`, in its existing isolated worktree. Preserve the
canonical Vashti checkout/coordination work. Public API/config/catalog, installed
services/runtime, Vashti and release selection are unchanged. **Required Vashti
changes: none.** This is standalone fixed-public laboratory work.

After `4b29087`, integrate the complete backend range through `9218638` when
resuming sequential work: actual MTP projections (`8cba745`), empty-slot filling
(`b739c32`), same-input expert/router diagnostics (`37f4edb`, `41db769`,
`b285d68`, `67bdcfa`), independent segmented-IQ probe (`f116d09`/`3571bcb`),
explicit original32-reference prefix23 (`4f41d6c`), skip-unused segment option
(`397999c`/`c6d2088`), and fixture-scoped correct-expert replay (`9218638`). Read
[the exact plan](kuno-native-text-mtp-plan.md), [latest checkpoint](kuno-native-text-checkpoint.md)
and [numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

Both cards pass actual target-conditioned MTP GPU projections and independent
original72+72 mixed-IQ MoE operator cases. MTP other graph math remains CPU;
private full-head ownership, independent whole-head quality and speculative
performance remain pending. The Q3 full32 all-GPU logit failure remains:
position23/layer17 near-tie routing follows tiny arithmetic differences and
changes one expert. Same-input expert outputs and CPU top-k on GPU scores agree.
No tolerance relaxation, forced routing, full32 performance or decoder defect
claim. Explicit prefix23 checks the complete original32 file and verifies only
positions0..22 before the known discontinuity.

One compiler-idle same-artifact prefix ideal pair gives9.516→10.118tokens/s
without/with unused-segment skipping, respectively (22steps each).10,560 GPU
expert calls, zero CPU calls, expert uploads and physical read bytes during
replay. Full23 logit vectors/greedy IDs, exact trace membership and full owning
closes pass. Correct original preloading/training are excluded from timing;
this is conditional short residency, not demand-cache/production throughput.
Demand-fill initial runs are slower and I/O-confounded. Keep new options off
by default. No private prompts/routes/history were persisted; all fixtures are
compiled public token0/seed424242.

Root host checks:591 native tests (one ignored), two one-test shader suites,
strict release all-target Clippy default/all-five features, formatting and
whitespace pass. Four v90 runs have five exact818413568-byte/1378-buffer original
owning closes each, plus two additional ideal training/replay closes where used.
At2026-10-09T04:41:26Z all jobs exited, normal admission resumed with no active/
queued/recovery latch, HTTP200 from Kuno/Ollama/Vashti, B70 memory50688000bytes,
no NVIDIA compute process and zero classified Xe/NVIDIA faults. No reset/unbind/
reboot, driver/clocks, installation or release change. GPUs are available.

Next source candidate `6d12ddd` is isolated on `codex/kuno-q3-subgroup-moe`, base
`9218638`, in the Unsloth quant-support managed worktree. **Not integrated,
compiled, tested or GPU accepted.** It adds a leaf-only four-row original-order
subgroup MoE kernel plus module/helper visibility. Resume with count1 opt-in/
device-feature integration, independent three-profile probe/fault/owning checks,
then host and protected both-card/full-vector performance acceptance. No GPU
work should resume until the user gives them back. DASLab remains an assessed
quality comparison candidate, without a new download or measured comparison.


### Active handoff: DASLab native Rust and Strata optimizations (2026-10-09)

**Goal paused by the user after the overnight pass.** Finish only the current
source checkpoint; do not start further GPU work until resumed. The final
paused receipt below supersedes earlier active-goal wording. Installed services
and the canonical Vashti checkout remain unchanged.

Supersedes the paused `9218638` checkpoint for backend inference work. User has
returned the GPUs and approved full DASLab IQ3_S plus Strata source study, with
an active30 accepted-token/s target. Latest model-tested target/MTP source is
`12f9cc7` (v129) in the same isolated `codex/kuno-native-text` worktree: dedicated
request I/O reuse, direct-only submission, complete finite fusion and actual warm
replay pass full physical gates. Earlier
conditional ideal count1 v109 remains selected for that separate scope. Installed services, public contracts/catalog/
configuration, Vashti and release selection remain unchanged. Preserve the
canonical checkout's uncommitted coordination/application work.

Read [current checkpoint](kuno-native-text-checkpoint.md),
[DASLab notes](kuno-native-daslab.md),
[numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/daslab-foundation-20261009.json)
and the [MTP plan](kuno-native-text-mtp-plan.md). Changes since pause include
original-format scalar/SIMD/GPU support (`d11bd15`/`985ffb5`), exact DAS profile/
shared decoder connection (`67a01bc`/`aae7e3b`), explicit original prefix13
(`07be82e`), dense decoder extension (`7d06efb`), original shared-codebook and
integer expert leaves (`d424727`/`2375412`/`b82eefd`), direct lane initializers
(`6189e69`/`5b976e2`), prepared Q6 head (`36cf286`/`fb6cd2f`), aligned Q6 loads
(`bcb9bfc`), host phase diagnostics (`690f032`) and bounded larger request-cache
metadata (`bfadcd4`/`85711cd`). MTP KV-only append is `2dea140`. These are
laboratory changes, not a new production native model route.

Two original DAS shards are fully verified without merged/duplicate payload.
Both GPUs pass independent mixed-quant operators. Fixed prefix13 full-vector/
greedy/ownership tests pass. Best conditional ideal replay is12.728tokens/s;
30tokens/s is not achieved. Original full32 fails at position13 with finite
logit disagreement despite greedy equality; keep that failure and unchanged
tolerance. v105 identifies binding/enqueue/fence overhead; simply growing the
append-only cache in v106 does not improve hits or speed. FIFO cache replacement (`b4c07ed`) and once-prepared integer inputs
(`f5d5e97`/`95b31cc`, full INFO plan `774f2a9`) now pass both-card operator and
protected prefix13 model acceptance. Same-artifact ordered control/prepared
comparison is10.511→12.728tokens/s,17.4% lower latency. Aligned Q6 passes
operators/model gates but adds no speed. Keep the original head. Next larger
[resident GPU routing candidate](kuno-native-resident-router.md) is isolated work
in progress, with no device/throughput acceptance.
No private content/history is persisted; only compiled public numeric fixtures
and model provenance. Actual memory, admission/services and classified-fault
recovery pass after every physical trial. No resets, reboots, driver/clocks,
installation or release. Integrated663 default/666 all-feature native tests and auxiliary suites pass;
strict default/all-feature all-target Clippy pass. **Required Vashti changes: none.** Future production native
catalog/routing still needs a coordinated consumer contract.


#### Backend-only routing checkpoint (2026-10-09,11:00UTC)

Model-tested source `1742100` retains all original DAS13 vector/greedy/trace and
owning checks. New GPU routing is numerically accepted but **slower** than its
matched both-segment control:10.1277→8.4455tokens/s. Keep prepared-only v109
(12.728) selected. Mapper probe `536c98e` passes both GPUs,49 independent cases,
24 fault cases and complete erasure. Full receipts and exact artifact hashes
are in the linked DASLab notes/numeric receipt. Physical memory/admission/HTTPs
recover, all classified fault counts zero. No production API/configuration,
installation/release or Vashti change. Next isolated candidates are parallel
mapper preflight and then prepared integer composition; no speed acceptance
for either yet. **Required Vashti changes: none.**


#### Backend-only parallel routing checkpoint (2026-10-09,11:45UTC)

Model-tested source `6a32b11` now passes prepared integer + parallel sealed GPU
routing with original DAS13 vectors/greedy/actual trace and owning audits.
Same-artifact ABBA gives prepared-only12.7468/12.8670 and routing14.6064/10.0652
accepted continuations/s: the routing gain is inconsistent, so keep v109
prepared-only selected. Parallel original mapper `416f889` passes both cards.
Root695 default/698 featured tests, auxiliary shader suites and strict
Clippy/format checks pass. Exact scope, per-step clocks, physical read counts,
source/artifact hashes and all failures are in the native checkpoint/DASLab
aggregate. Native MTP target/draft leaves are in progress, not performance
accepted. Public services/admission/memory/fault counts recover after all runs;
no install/reset/driver/API/Vashti/release change. **Required Vashti changes:none.**

#### Backend-only target rows/carry checkpoint (2026-10-09,12:08UTC)

Original all-row target `d12b96b` and public registration `38c0e52` pass the B70:
complete CPU logits/actual pre-head carries for all four rows across five
partitions and continuation, exact greedies/graph/cursor checks and full owning
closes. Frozen default artifact is `fc4c9f346c67b571585749d5164b391bcdd21ec6d101e28d3fa32e151f172ab7`.
CPU oracle `68e2c33` matches existing CPU logits bit-for-bit and rehashes both
original shards. Root717 native plus2 auxiliary tests, strict default Clippy,
fmt/diff and80 launcher controls pass. The narrow protected launcher correction
is `ef2cf9e`; initial pre-model rejection is preserved. See the
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json)
and [implemented target contract](kuno-native-mtp-all-rows-plan.md).
All supplied rows commit; accepted-prefix repair, whole GPU MTP and speculative
speed remain pending. No new throughput claim. Physical memory/admission/HTTPs
recover and classified faults stay zero. No install/API/Vashti/release change.
**Required Vashti changes:none.**

#### Backend-only resident MTP checkpoint (2026-10-09,12:48UTC)

Corrected global input RMS `724ca2f` and publisher-aligned CPU v2 oracle
`99386e6` pass complete fresh CPU vectors. Full resident Q8 MTP `a32f241`
and public consumer `1761b06` pass both physical GPUs: teacher/provisional
vectors, exact greedies, actual target-KV repair/continuation, two fresh
workspaces,42-owner erasure and public-weight release. Frozen default source
`f652d53`, SHA256 `7f6a8e12690b2ad2c68827bc26d00808ebe4657deea28f384a59318a9dc266d4`.
739 default native+2auxiliary tests, strict Clippy/fmt and81 launcher controls
pass. Actual memory/HTTP/admission recover and classified faults remainzero.
Initial provisional CPU agreement remains0/2; neither useful steady draft
acceptance nor speculative throughput is claimed. Historical dummy-cell/
per-stream references are retained but superseded for publisher conditioning.
Target-prefix repair source `f3899b6` and public consumer `211b282` pass
747+2 source tests/strict Clippy; physical target repair is next. See the
[native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
No public contract/install/release change. **Required Vashti changes:none.**

#### Backend-only speculative foundation checkpoint (2026-10-09,13:08UTC)

Actualtarget prefixrepair1/2/3 passes at `54c846b` (v118); minimalQ8draft and
GPUfollow pass bothcards at `241415b`/`da69fe0` (v119), keeping fullchecks.
Separate13-row CPUcarry `d6a7df7` matches oldCPUlogits bitwise. First actual
dualGPUpubliccontroller `66c1078` passes12targetcontinuations,6/10draft
acceptance, originalstate repair/realKVrebuild and fullcloses. FrozenSHA
`de3120b39f366914fecb96515c55154a13d314356d38a7377d3108a4074f1ce5`.
Cold normal-demand run1.1326tokens/s includes2,311expertfills/4.777GB and
allactualdraft/verification/repair/publicchecks; this distinctscope doesnot
establish a speedup over selected12.728conditionalidealpreparedcount1.
762native+2aux tests, strictClippy/fmt and82launchercontrols pass. Actual
memory/HTTP/admission recover, classifiedfaults remainzero. Details, all
source/artifact/hash provenance and preservedfailure are in the
[native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
Packedthree-column dense and preparedbatch expert candidates are inprogress.
No publicAPI/config/catalog/install/release change. **Required Vashti changes:none.**

#### Backend-only direct fill and draft fusion checkpoint (2026-10-09,14:37UTC)

Latest model-tested source `a6fcdf5` passes original14-format operators on both
cards, target4/actual prefix repair and fused MTP full CPU vectors/KV repairs,
with complete owning erasure and actual memory/admission/HTTP recovery. Frozen
default SHA256 `38296f8f909cb6791cc249a5180d2fdd2f833c64c6206cfcbb0636d4c19d1c5b`.
Compiler-idle ordered cold normal-demand pair improves1.50087→1.74852 accepted
tokens/s with checked existing-view public fills+fused MTP, unchanged arithmetic
and actual repair. Both have2,311 first-touch fills/zero evictions and6/10 draft
acceptance. This distinct cold scope does not replace selected12.728 conditional
ideal count1;30 remains unmet. Prepared candidate has different disk paging and
no accepted speed gain.796 native+2 auxiliary source tests and85 launcher
controls now pass for the next one-shot public warm replay; its physical gate
is pending. No private content, GPU resets or public API/config/install/release
change. Exact sources, hashes, numeric receipts, all failures and limitations
are in the [native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
**Required Vashti changes:none.**

#### Backend-only warm and finite acceptance (2026-10-09,15:14UTC)

`4b93c99`/`bafd9c0` real healthy one-shot replay passes full original CPU13,
actual target/draft repairs, complete retained-private erase and final closes.
Weights/banks/IDs/admission/epochs stay; transient binding/INFO caches are
cleared. Warm normal-demand2.34472 withzeroexpertuploads establishes that
execution overhead remains. Prepared2.34773 shows no material gain. Packed
Verify3 warm2.61000 and full finite fusion `fe196fa`/`6f581b4` warm2.96368 pass
all original gates. Corrected inventory `5eb8e82` passes60normal+210invalid/
stickycases percard and full actualprefixrepair. Current model-tested v127
SHA256 `7f3fe8a0a9a24c6543784cfb1937c8bde3db7e6507cf612ee363b59318c2baba`.
These ordered warm trials do not establish matchedABBA/30token/s acceptance or
replace v109's separate12.728 conditionalideal count1 selection.

New pure `direct-device` source `834a299` isolates existing GPU submission type
for only fixedpublic target4/repair4/speculative13. Exact existing
`KUNO_LAB_DEVICE_DISPATCH=inline` and direct-only build are mandatory; all CPU
features/default+inline/directwithoutflag mismatch reject beforedevice access.
ActualDeviceHandleblocking is checked in main/run/receipt. Mathematical kernels,
owners/budgets/cache/defaultthreaded/otherlabmodes unchanged.806 native+2auxiliary,
pure-direct registration/runtime and rejectedallfeatures checks,87launcher
controls and strictdefault/direct/allfeaturesClippy/fmt pass in child source;
physical gate is pending in root. Actual GPU memory/admission/services/fault
recovery remains verified after each trial. Exact sources, all numeric receipts,
failures/limits in linked checkpoint/foundation receipt. No private prompts/
outputs stored, productionAPI/config/catalog/frontend/install/release/reset
change. **Required Vashti changes:none.**


### Kuno native text direct target checkpoint,2026-10-09,15:31UTC

`834a299` pure `direct-device` fixed-public DASLab target modes now pass actual
B70 target4/repair4 and B70-target/RTX-draft fullCPU13 cold/warm controller gates.
Warm3.18530 acceptedtokens/s withzero expert transfers/weight reads/processdisk
bytes;30tokens/s is unmet and no matchedABBA selection follows. Original full
vectors, arithmetic, actual state repairs, private/source erasure, memory/HTTP/
admission recovery andzero new classified faults pass. Exact source/artifact/
helper/numeric evidence is in the foundation receipt and native checkpoint.
A target-client-only charged request metadata reuse scope is next, source work
only. Ordinary modes/API/frontend/install/release remain unchanged; required
Vashti changes:none. Preserve canonical application and coordination work.


### Kuno native text request reuse checkpoint,2026-10-09,15:55UTC

Runtime `0cfdbe0`, bounded native permit `9f3b8cc`, public registration `12f9cc7`
now pass original target4/actual prefix repair and fullCPU13 cold/warm dual-GPU
controller gates. Compiler-idle warm6.63953 acceptedtokens/s,zero expert uploads/
weight reads/4,096 process disk bytes. Request begin+original full closing audit
are inside timing; metadata4096 entries/4MiB are fully charged, target-only
programmatic configuration leaves draft/default/cap10 unchanged. Full private/
source erasure, actual memory/services/admission andzero classified faults pass.
Root814+2auxiliary tests (oneignored), strict default/direct Clippy/fmt,88
launcher controls/source review pass. Exact hashes/numeric receipts and scope
are linked in native checkpoint/foundation receipt.30tokens/s remains unmet;
ordered improvement is not matchedABBA or production/private acceptance.
Opt-in original-F32 unused expert-member scratch reductions are the next
isolated source work. No API/frontend/installation/release/reset changes;
required Vashti changes:none. Preserve canonical application/coordination work.

### Paused Kuno overnight handoff,2026-10-09

The user paused the goal; source-only checkpoint `bbe630a` integrates the
original-F32 live-member kernel `7ab49b8`, immutable model policy `13d9929`,
independent multirow operator `a13ebc3` and tiny lint fix `91dd416`. No new GPU
run is authorized by this handoff. Latest model-tested artifact is still
12f9cc7/v129. Normal warm dual-GPU speculative controller is6.63953 accepted
tokens/s; optional two-residual-Q8 prepared mode passes original full-vector
and actual-repair gates at6.77829, but its2.09% ordered gain is not a material
selection. Separate conditional ideal count1 v109 remains12.728; none establishes
30tokens/s or production/private/general inference acceptance.

Live-member source skips only unused member-plane reductions, preserves active
arithmetic/unconditional barriers, and remains default-off. Physical both-card
operator, target4/repair4 and fullCPU13 matched control/live timing are pending.
Root829native+2auxiliary,89 launcher tests and independent source reviews pass;
exact final lint checks, source hashes, numeric receipts, preserved failures,
limitations and resume steps are in the
[paused native checkpoint](kuno-native-text-checkpoint.md#paused-overnight-checkpoint2026-10-09)
and foundation receipt. Fresh16:21UTC actual memory/HTTP/admission recovery and
zero classified Xe/NVIDIA faults pass. No benchmark GPU process remains.
No API/frontend/config/catalog/install/release/reset change. **Required Vashti
changes:none.** Preserve the canonical checkout and its coordination/application
work; resume source/benchmarks in the isolated native-text worktree only.


### Standalone Strata Intel comparison, 2026-10-09

The user explicitly authorized a separate upstream Strata installation while the
native Rust optimization goal remains paused, and requested leaving Strata running
for hands-on use. MIT source v0.1.41/fb58e0d, AOT bmg-g31/oneAPI 2026.1, original
DASLab IQ3_S hard-linked in place, new pinned Q2_0 MTP conversion. No native engine,
API, Vashti integration, production model catalogue, driver or release change.

Actual B70 OpenAI API checks return the correct integer sanity result and
27.3/36.0 decode tokens/s on two 200-token public fixtures (seed 424242, greedy,
no-thinking, PLE direct). This is upstream Strata application inference; it does
not establish native Rust 30 tokens/s or a matched Ollama comparison. Core/swap
protection and no persistent inference logging are checked; corrected rootless
Podman containment requires service Delegate=yes. Actual shutdown returns B70 to
50,688,000 B, admission unpaused and all three HTTP services 200 with zero classified
GPU faults. Final protected restart/streaming and health/web checks pass; sampled streaming
at temperature 0.7 produces 45 tokens at 16.4 decode tokens/s. Strata is left running
on loopback port 8085 for the user; Kuno inference admission stays intentionally paused
until the documented stop control releases the GPU and restores it. No private
inference output is persisted; upstream browser history uses localStorage.

Exact format notes, source hashes, numeric receipts, local installation/control
paths and limitations: [standalone comparison](kuno-strata-comparison.md).
That document includes the user-requested Q2_0 MTP runtime importer notes for
future Rust work. **Required Vashti changes: none.** Preserve canonical application
work. Native goal and pending live-member kernel gates remain paused.

### Native Strata-inspired pass resumed, 2026-10-09

The user authorized teardown of the comparison installation and resumed native
Rust engine optimization with a new goal. The historical Strata-running state
above is superseded: Strata is stopped and its specifically owned build/image
removed, while numeric receipts and the alternate Q2_0 draft files are retained.
Protected 17:43 UTC host checks confirm actual GPU memory recovery, unpaused Kuno
admission, all three HTTP services 200 and zero classified GPU faults. Exact
retained paths and limitations are in [the comparison receipt](kuno-strata-comparison.md).

Native source starts from `de6269d` in the isolated native-text worktree; parallel
backend source work uses separate worktrees at that commit. Priorities are
device-side routing/group maps and a strict alternate bound-target MTP graph,
then real public language-generation benchmarks. No API/frontend/catalogue/
release change is included. **Required Vashti changes: none.** Existing full
CPU13/actual-repair correctness gates remain controls; conditional all-hit GPU
results must not be presented as ordinary demand inference performance.

### Native Strata follow-up source and physical checkpoint, 2026-10-09

Backend-only active native work now integrates original Q2 GPU draft `9f66226`,
independent probe `bedc06f`, host-placement probe `ceb5315`/`740cd91`, profile
selection `9d99d0d` and main `f868437` in the isolated native-text worktree.
Frozen v135 passes both-card full Q2 draft CPU-reference/repair/owning closes.
v136 passes both-card small direct host-allocation original math, combined
CPU13 and actual public code200 generation. The Q2 public owner plan falls
5,011,668,992→2,539,257,856bytes; its cold-code200 result is3.39096 continuation
tokens/s and short all-hit warm13 is5.86668, so no speed improvement/30 claim.

Explicit internal opt-in is `KUNO_LAB_MTP_PROFILE=strata-q2`; absent preserves
Q8. Verified sidecar path remains `KUNO_TEST_MTP`. Both profiles retain actual
target verification and original source ownership. No production engine, API,
catalogue, installation, frontend or release selection is changed. Every trial
restores physical memory, normal admission, all three HTTP200 and fault counts0.
**Required Vashti changes:none.**

Next source work: separately budgeted immutable host-only expert pages and
nonreuse external CubeCL ownership; bounded GPU chronological attention selection
for <=2048 token positions, exact original CPU fallback above that boundary.
Neither connection has whole-model acceptance yet. Numeric receipts, exact
artifacts/tests/limitations and preserved startup failures are in the
[current checkpoint](kuno-native-text-checkpoint.md#smaller-draft-and-direct-host-source-physical-follow-up).


### Native attention/host-source checkpoint, 2026-10-09

Backend-only native source`cc4b8aa`+`597039d` passes both-card chronological QSA
operator, original full CPU13/exactgreedy/actualrepair and normal code200 gates.
Internal opt-in`KUNO_LAB_TARGET_QSA=chronological-public` uses exact chronological
original IDs for end<=2048tokens, retains original guarded CPU selector above;
extra49,152INFO+4,096host bytes, no newMAIN owners. Absent/default unchanged.
Warm7.26311tokens/s/maps51; normal code200candidate2.08246/control2.01633 with
~49s weight-reading/35GB physical reads. No matchedABBA/30/production acceptance.
Parent897+2aux native tests (one ignored), strict direct Clippy/focusedQSA58 and
96launcher controls pass. Exact sources/artifacts/numeric recovery evidence and
limits are in the [native checkpoint](kuno-native-text-checkpoint.md#chronological-attention-and-immutable-host-registration-follow-up).

Immutable external public host runtime`dd0db6b`+`139075e` has source/ownership
acceptance; actual CubeCL host operator and full mirror consumers remain pending.
The user-visible API/configuration/native catalogue and installed services are
unchanged. All physical trials restore memory/admission/allthree HTTP200 and
fault counts0. Required Vashti changes:none; no release/install/reset authorization
is inferred from this internal performance checkpoint. Preserve canonical work.

### Active handoff: RAM-source native text validation; user requested stop, 2026-10-09

The detailed [next-model handoff](kuno-native-next-model-handoff.md) is the
starting point for another model resuming this backend pass. It includes the
exact source/physical acceptance split, model and reference paths, protected
helper limitation, ordered correctness/timing gates, and recovery instructions.
Its creation changes documentation only; GPU work and the goal remain stopped.

This supersedes the pending-host-materialization state above. v143/`f0e088b`
passes both-card80-case authenticated512 source-router operators and the B70
full original CPU13 target/3060draft controller with actual VRAM/RAM experts.
The importer now authenticates actual Vulkan allocation alignment rather than
conservative planner rounding. All original vectors, greedies and actual repair
pass;432host pages detach and CPU-zero/free36,147,609,600bytes. Short deterministic
warm placement is4.21677tokens/s with0explicit expert uploads; it is not normal
language200 or30tokens/s acceptance. GPU512 full-model routing and the new
inactive-partition kernel candidate still require physical acceptance/timing.

The user requested stopping this pass to conserve usage. All GPU work is stopped;
21:23:38UTC checks confirm B70baseline50,688,000B, no NVIDIA compute owners,
normal/unpaused Kuno, allthree HTTP200 and unchanged classified GPU faults0.
Installed Kuno/Ollama/Vashti and their public APIs are unchanged; no release or
installation is inferred. Required Vashti changes:none. Resume from the
[RAM-source checkpoint](kuno-native-text-checkpoint.md#ram-source-full-model-acceptance-and-stop-checkpoint-2026-10-09)
and [internal target/source contract](kuno-native-mtp-all-rows-plan.md#authenticated-immutable-ram-expert-sources-2026-10-09).
Exact source hashes, tests, artifacts, preserved startup failures and numerical
limits are recorded there and in the content-free native optimization receipt.
Preserve the canonical frontend task's unrelated/uncommitted files.

Final source checkpoint`389fb6b` includes explicit inactive-partition candidate
leaf`612b7f6` and guarded parser/launcher`6413b16`; defaultfalse unchanged.
Integrated45public parser/reference +30frozen/routing/shader +102launcher tests,
strict pure-direct all-target Clippy and formatting pass. Candidate GPU/full-model
and speed acceptance remain pending, as does full-model frozen512 routing.
The physical accepted control remainsv143/`f0e088b`; see checkpoint for exact
source chain, protected helper limitation and the ordered next gates. No more
benchmarks are authorized within this stopped pass.

### Active handoff: native lookup-table speed experiment resumed, 2026-10-10

The user explicitly resumed backend optimization after GLM/free-model work,
requesting progress toward20tokens/s. This supersedes the stopped-pass status
above. Inspection found no blocking defect in the router-activation ledger fix
`8c7d838` or preserve-bank diagnostic `4fad434`. v145/fd39cd1 has full physical
CPU13 gates and GPU-routed code200 ABBA: control7.54493/candidate8.33810tokens/s,
matched exit-kernel gain10.51%. No production/API/release acceptance is implied.

The fresh v146/1d0d7c8 preserve-bank replay passes original CPU13 vectors,
greedies/real repair and full closes:12continuations/2.095918815s=5.72541tokens/s,
9120VRAM/0host rank-member routes,0expert uploads. Memory/admission/services and
fault counts recover. The matched canonical control is4.31182tokens/s (ordered pair,+32.78%);
this diagnostic does not establish a normal-language placement optimization,
because code200 already preserves its post-prefill residents.

Next candidate targets repeated by-value codebook array spills in the actual
CubeCL quantized gate/up kernels. Exact opt-in contract is in the
[all-row plan](kuno-native-mtp-all-rows-plan.md#operative-public-lookup-table-lowering-experiment-2026-10-10).
Private-table leaf`120e94d` is integrated as`e054d70`; the guarded public controller
and negative element-write proof are in the follow-up source commit. Root owns
the shared GPUs. Physical candidate acceptance is pending.
No API, frontend, catalogue, installation or release change is included.
Required Vashti changes:none. Preserve canonical application work.

### Active handoff: native text experiments stopped with system usable, 2026-10-10

This supersedes the resumed/pending status above. The user requested finishing
remaining tests and returning the GPUs. All owned GPU/CPU work has stopped;
no benchmark, installation or release is authorized by this handoff.

Native root7364dc8 (private-table leafe054d70) passes source checks and the full
original CPU13 physical gate. One ordered same-artifact public code200 comparison
is7.78036 control /7.62161 private-table continuation tokens/s; no speed gain.
Default/installed behavior is unchanged. Prior repeated v145 result remains
8.33810 tokens/s; neither20 nor30 is accepted. Two isolated clean source leaves
are ready for future integration/testing: workgroup tables3594b76 and IQ4 shared
down scalesb56af6a, both based7364dc8, with focused CPU tests and strict Clippy.
Their root parser/launcher/receipts, composition and GPU acceptance are pending.

Exact commits, immutable artifact hashes, tests, rates, limitations, worktrees,
next gates and final recovery evidence are in the
[latest checkpoint](kuno-native-text-checkpoint.md#resumed-shader-experiments-and-usable-stop-checkpoint-2026-10-10)
and [current resume guide](kuno-native-next-model-handoff.md).
Internal contract is in the all-row plan; no public API/configuration change.
Required Vashti changes:none. Do not infer full feature, deployment or speed
acceptance from source-only checks.

Final live check2026-10-11 01:27:07UTC: B70baseline50,688,000B, no NVIDIA compute
owners, admission idle/unpaused/no recovery latch, allthree services HTTP200 and
classified GPU faults0. Installed services and canonical frontend are unchanged.
Confirm current GPU availability/sudo again before resuming; this is a historical
host snapshot. The optimization goal remains paused.


### Active native kernel optimization resumption, 2026-10-11

User restored GPU access and sudo and resumed the existing30tokens/s goal.
Native work continues in `codex/kuno-native-text`, isolated from the unchanged
canonical Vashti checkout. GU workgroup and down-scale source leaves are now
integrated as `4f23abd`/`ce78d39`; root guarded selection, actual completed-policy
receipts and composed runtime dispatch are being tested. Exact laboratory
contract is in the final section of the all-row plan. This supersedes the
previous statement that parent selection remains pending; GPU acceptance and
normal throughput measurements still remain pending at this source checkpoint.

Fresh host check2026-10-11 12:10:52UTC: the retained managed image worker was
identified by parent/cgroup and cleanly evicted through Kuno's existing empty
Ollama unload request. B70 returned to50,688,000B; NVIDIA compute owners0;
admission idle/unpaused/no recovery; allthree HTTP services200; GPU faults0.
No reset, driver/service installation or private content inspection was needed.
Installed services and public API are unchanged. Required Vashti changes:none.

### Active internal expert-factor/draft-head experiments, 2026-10-11

v148/76ffd66 passed complete original CPU13 target/carry/greedy/repair gates and
full owner closure. Normal same-artifact ordered code200 is7.73410 control versus
6.90538 workgroup-codebook/shared-down-scale candidate; no speed gain. Candidate
remains default-off. Both trials restored admission, all HTTP services and GPU
memory (B70 baseline50,688,000B, NVIDIA compute0), with no new classified faults.
Numeric evidence is under `v148_cooperative_physical` in the native optimization
receipt; latest physical snapshot12:33:49UTC is historical, not future availability.

Next source candidates: GU factor leafa8e81a1 integrated1ba718e; Q8-head broadcast
leafc4cd6c3 integrated0675594. Parent selectors and actual completed-policy receipts
are integrated with runtime dispatch composition. Exact default-off internal
contract is the final section of the all-row plan. Source validation passes:
9exact-public registration/policy checks,5compound runtime/parser checks,5GU
factor checks,4Q8operative/selection checks and6original projection regressions;
107Python launcher checks,9fixed-trial scope/limit checks, strict pure-direct
all-target Clippy, formatting and diff checks. No GPU numerical or throughput
claim is made for these new candidates until a new frozen artifact passes the
original CPU13 gate and normal language timing. Further four-row down kernel and
cooperative draft-head scale candidates are isolated source work in progress.

This is backend-only lab optimization. Installed services/public API/production
configuration and the canonical Vashti checkout remain unchanged. No release or
installation is authorized by this source handoff. Required Vashti changes:none.

### Active row/draft-scale source checkpoint, 2026-10-11

v149/030a20c original full CPU13, real repair, full closure and physical recovery
pass with GU factor plus broadcast-head candidates. Normal ordered code200 is
8.11398 control /7.68506 GU factors; no accepted gain. Both remain default-off;
broadcast-head-only normal speed was not measured. Reference fixture throughput
is diagnostic, not normal generation. Numeric source/model hashes, all results
and recovery are recorded under `v149_factors_head_physical` in the root receipt.

Next integrated source-only leavesf7bb253 (four-row DOWN, original64virtual
sums/tree) and9dde5da (cooperative Q8-head scales) now have complete guarded
parent selectors and actual-completed-policy receipts. Source tests pass11exact
public policy checks, all48actual composed entrypoint/grid checks,3operative
four-row tests,10Q8head checks,108Python launcher checks,9newfixed-trial memory/
scope checks, strict pure-direct all-target Clippy, fmt/diff checks. Exact internal
contract is in final all-row plan sections. Full original GPU CPU13 and normal
language timing remain pending for v150. Source-only counts are not speed proof.

Backend lab only; installed services/public API and canonical Vashti unchanged.
Required Vashti changes:none. Additional two-row GU and transient host-weight
staging experiments remain separate source work, with no physical acceptance.


### Active paired-GU / bounded transfer experiment, 2026-10-11

v150/3ed26fe passes complete original CPU13, actual repair, full closure and
physical recovery. First normal ordered code200 pair8.11952 control /8.48473
four-DOWN/shared-head candidate (+4.50%) is preliminary; repeated ABBA/prose
remain untested. Exact evidence is v150_rows_shared_head_physical in the native
receipt. Last recovered snapshot13:23:00UTC has B70baseline50,688,000B,
NVIDIA compute0, unpaused idle/no-recovery admission, allthree HTTP200 and faults0.

Two-row GU leaf0353369/parent5d26186 and staging precharge10e3d1c are integrated
source-only; exact default-off selectors and bounded owner/runtime contract are
in the final all-row plan sections. GPU validation for paired GU and full staging
integration/validation remain pending. No30tokens/s, production, install or
release acceptance. Backend lab only; required Vashti changes:none.

### Active exact packed HOST staging source gate, 2026-10-11

v151/e3983c3 passes complete original CPU13, repair/full closure and host recovery.
Its paired GU/four DOWN/shared Q8 head normal code200 is7.08234 continuationt/s;
no matched normal control, so no paired-GU gain is claimed. Latest recovered
snapshot13:39:03UTC: B70baseline50,688,000B, NVIDIA compute0, unpaused idle/no
recovery admission, allthree HTTP200 and classified faults0. Full evidence is
v151_paired_gu_physical in the root native optimization receipt.

Leafd640ab2/c3b5f8d packed staging is integrated with the exact final all-row plan
contract: three32MiB MAIN owners,589,824B extra INFO,256-thread copy capability,
full outer/MoE/source alias/stream proofs, original frozen MAP/expert identity,
same-stream copies and consumers, completed actual policy/accepted logical byte
counters. Controls and all experiment defaults are unchanged. Source gate passes
5leaf checks,3root checks,110Python tests,9strict fixed-label scope/memory checks,
all-target direct Clippy and fmt/diff. Full original GPU CPU13 and normal speed
remain pending. No API/configuration/installed service change or release claim.
Required Vashti changes:none. Work remains in the native isolated checkout.

Initial v152/c3e9403 rejected before target allocation: the executor's independent
exact MAIN check omitted the new staging extent. Recovered snapshot13:48:24UTC
retains baseline GPU memory, idle admission/healthy HTTP and zero classified
faults. Followup adds explicit default-off executor staging reservation, Config
parity and exact precharged byte acceptance;4focused root checks and strict
direct Clippy pass. New frozen artifact/full GPU gate remains required. No
numerical/throughput acceptance or installed-service/API/Vashti change.

v153/e56e466 exposed a second construction issue: staging reserve reacquired MAIN
through healthy()/active inside the already-locked allocation callback. Stopped
only the owned benchmark unit and ran helper recovery; baseline GPUs/HTTP/idle
admission/no faults returned at13:58:14UTC. Full owner erasure is not claimed for
that aborted public-only process. The fix retains borrowed construction origin,
poison and full arena/owner checks without locking MAIN again; held-lock token
test and strict direct Clippy pass. Fresh original GPU acceptance remains pending.


### Active native optimization checkpoint: three-member arithmetic,2026-10-11

Supersedes the pending packed-staging acceptance above. Constructor fixes
3de1529d passed full original CPU13/repair/all-owner gate in v154. Normal code200
packed staging/four-DOWN/shared-Q8-head is7.870304 continuationt/s; no same-artifact
control or speed gain claimed, and the alternative stays default off. Numerical
and physical closure evidence is in v154_packed_staging_physical.

Three-member kernel leaf6473be6 and parent226e3fe plus protected policy1db3d8c
pass6focusedRustchecks, strict all-target directClippy,5new/110existingPython
checks and9strict helperlabels/caps. It keeps tile8/MAX8 owners and every retained
F32 operation, only removing unreachable arithmetic cells3..7. Original-source
construction and attachment share the exact geometry reconstruction. Frozenv155
passed full originalCPU13 and actual completed three-member/four-DOWN/shared-head
flags, full1380MAIN/828866560B and432HOST-page36147609600B closure, and zero draft
reservations. Warm12-token5.471858 diagnostic does not establish normal speed;
normalcode200 completed at8.094793continuationt/s (199/24.583703122s),
130/138drafts/69rounds, full repair/closure. No matched control or speed gain
established; defaultoff retained. Exact source/binary/helper hashes and recovered
snapshot14:28:19UTC are under v155_three_member_arithmetic_physical.

Next comparisons: already-supported original Q2 MTP with current frozen target
(explicit new labels, original shard hashing, no Q8-head policy); separately,
isolated original-arithmetic three-token GDN streaming source to remove a large
private column array. Neither is physically accepted yet. Root owns all shared
GPU/services. Backend-only lab; required Vashti changes:none, installed services
and public API unchanged. No30tokens/s or release/production acceptance.


### Active native backend optimization checkpoint,2026-10-11 14:51UTC

Latest sourcee16c549/v156 three-token GDN passes original CPU13 and normal
code200 full greedy/repair/owning closure:8.489327continuationt/s,199timed,
130/138drafts/69rounds,74actualThreeGdn chunks. Same-artifact control is8.542111t/s but agentClippy may overlap;
no compiler-idle causal pair or gain claimed. Original Q2 with previous three-member/four-DOWN target
also passes at7.949402; its public GPU weight budget falls to2,539,257,856B,
but no faster inference is established. Earlier staging and three-member
physical gates are accepted and their slower preliminary timings remain defaultoff.
Read the updated next-model handoff and exact numeric receipt sections
v155_original_q2_frozen_physical /v156_three_gdn_streaming_physical.
Root owns sharedGPU trials; leaf work proceeds in separate managed worktrees.
Next draft-only experiment is the pinned static English/code vocabulary;
source work is not GPU/throughput acceptance. Last recovery14:51:03UTC restores
baseline GPUs, idle/unpaused/no-recovery admission, allHTTP200, classifiedfaults0.
No30tokens/s/private/production claim; installedservices/API/canonicalVashti
unchanged. Backend-only internal work; required Vashti changes:none.


### Active native kernel checkpoint,2026-10-11 15:44UTC

Latest tested source21d97e2/v157 passes original CPU13/targetgreedy/carries,
real repair, full draft/target/source closure and physical recovery with the
fixed40525-row English/code draft head. Normal compiler-idle ordered code200
is8.293190 shortlist versus8.525761 same-artifact full-head control; no gain,
shortlist stays default-off. Warm4.152881 is diagnostic only. Read receipt
v157_public_english_vocab_physical and the updated next-model handoff. Last
recovery15:42:52UTC returns B70baseline50688000B, NVIDIA compute0, idle/unpaused/
no-recovery admission, allHTTP200 and no classified GPU faults.

Next integrated source-only candidate is the explicit two-residual-Q8 frozen
GU integer leaf and immutable parent, throughe655f32/13c276c/fb7c528/9f80c33.
Read the frozen GU leaf/parent contracts. Ten focused agent checks, strict
all-target directClippy and independent read-only source review pass. Original
F32 equivalence, full original CPU13 and throughput are NOT established by
packed-oracle tests. Root owns public controller/launcher integration and all
GPU/services; agents own isolated encoded-weight/group-scheduling followups.
Do not enable these candidates for ordinary inference, widen tolerances or
represent requested flags as actual execution. No30tokens/s/API/install/release
acceptance; canonical Vashti and installed services unchanged. Required Vashti
changes:none.

### Active native GU scheduling and MTP comparison,2026-10-11 16:12UTC

Latest physical sourceaa4df8c/v158 passes unchanged originalCPU13, targetgreedy,
real repair, full1381MAIN/828932096B and432HOST-page/36147609600B closure,
all draft/source closure and physical recovery. Explicit two-residual-Q8/eight
lane GU normal compiler-idle code200 is8.642210continuationt/s:
199/23.026516602s,130/138drafts/69rounds. Actual3552 preparations/17760GUcalls/
3552audits join full native completion. No same-artifact original-math control
or dramatic gain claimed; defaultoff. See v158_frozen_gu_integer_physical.
Last recovery16:00:31UTC: B70baseline50688000B,NVIDIAcompute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults, same boot.

Later integrated source-only encoded-register preparationb4b20f9 and
GUstride leaf/parent50d423d/febf8df/e5182f0 plus controller853e0e0 pass24
combined focused checks and independent read-only review. Guard124Python
checks and three external fixed stride labels pass. The alternative GU grid
reduces12000→1600workgroups/layer while preserving original runtime routes,
all scratch/owner limits and actual joined five-source receipts. Read the
frozen GU integer/stride leaf, parent and controller specifications. Fresh
artifact original GPU gate and same-artifact language timings remain required;
source success is not physical acceptance. Root owns sharedGPU/services.

The user requested measuring whether slow-slot NVIDIA MTP is a net cost.
An isolated real target-only language controller is being prepared: it must
not open or instantiate the draft nor select NVIDIA, and must execute actual
single-token original target passes and full closure. No target-only speed or
30tokens/s acceptance yet. Backend-only lab: canonicalVashti, installedservices
and publicAPI unchanged, required Vashti changes:none. No release/install.


### Active native MTP placement comparison,2026-10-11

Latest physical sourceac78f00/v159 passes original CPU13 cold/warm, targetgreedy,
real repair, full1381MAIN/828932096B and432HOST-page/36147609600B closure,
draft/source closure and physical recovery. Compiler-idle ordered same-artifact
code200:8.636025 control versus8.973138 encoded-register/GUstride candidate,
199 real continuations and130/138 accepted/proposed drafts each. This3.90%
result is preliminary; repeatedABBA/prose are pending. Full receipt and exact
hashes: `v159_encoded_register_gu_stride_physical`. Last recovery16:30:44UTC:
B70baseline50688000B,NVIDIAcompute0,allHTTP200,idle/unpaused/no recovery,
no new classifiedGPUfaults, same boot. This supersedes earlier pending-v159 notes.

The user requested testing Intel MTP before choosing whether to retain MTP.
Real target-only controllerd12b52f and Intel placement7b790f4 are integrated;
[the exact Intel contract](kuno-native-public-intel-mtp-controller.md) requires
fresh privileged Xe memory data, combined full owning charges, original144
expert-bank capacity, distinct device/runtime and original physical closure.
Target-only does not open or bind a sidecar or select NVIDIA. No implicit profile,
slot or admission fallback. Guard130Python tests pass; Rust source tests and
strict direct-device Clippy pass in the isolated owner trees. Integrated
root validation/fresh frozen binary and three-way physical acceptance are next.
Full Q8 on Intel may reject its tight combined memory budget; record any
rejection rather than widening limits. DOWN stride and grouped PLE are separate,
unselected candidates for later numerical/speed gates.

Existing30tokens/s goal remains unachieved. Root owns GPUs/services. Canonical
Vashti, installed services and public contracts unchanged. Required Vashti
changes:none; no install/release or other-task message is authorized by this note.


v160 Intel physical attempt rejected before model allocation at the independent
Wgpu device identity check. Fresh Xe accounting/full admission passed, and
recovery16:50:47UTC returned baseline/allHTTP200/no new faults. Pinned wgpu29
compares registry-local IDs across independent Instance contexts incorrectly
for this purpose; actual live Vulkan logical-device comparison is being fixed.
No Intel inference/timing/memory-fit claim follows from this failure. See
`v160_intel_mtp_registration_failure`; freeze a new artifact for the retry.


### Completed Intel/Nvidia/target-only placement block,2026-10-11

Source86882f9/v161, binarySHA256
`95ab177a40131f60342b5dff72cb75486c1d89958709def127b0e079ba777632`,
passes original full CPU13 cold/warm with Intel MTP: complete target vectors/
carries/greedy, real repair, all1381target MAIN/42draft private owners and432HOST
pages closed/freed, physical memory recovered. Fix compares live Vulkan device
handles rather than Instance-local Wgpu IDs. Six focused identity tests, strict
all-target direct Clippy and formatting pass; previous integrated98Rust and
130Python checks remain applicable to unchanged parser/guard code.

Compiler-idle ordered same-artifact code200 (199 timed continuations each):
Intel MTP9.908034, Nvidia MTP8.876789, real target-only8.336136tokens/s.
Same public prompt/seed424242/original target math/capacity/cache; both MTP use
full Q8/shared head, accept130/138 proposals over69 rounds and perform5 repairs.
Target-only opens no sidecar/creates no draft device and executes199 actual
count1 target passes, zero drafts/repairs. Full original source/owner closure and
physical recovery pass for each. Preliminary Intel gains11.62% over Nvidia and
18.86% over target-only; no repeatedABBA/prose/independentCPU200/30t/s acceptance.
This does not isolate PCIe bandwidth from compute/queue/driver differences.

Full144banks/8GiB static/fullQ8 draft fit together on B70 under the unchanged
combined admission; actual same physical PCI but distinct Vulkan devices and
runtimes are reported. No profile/slot/cap fallback or installed/default change.
Last recovery17:14:23UTC: B70baseline50688000B,NVIDIAcompute0,allHTTP200,
idle/unpaused/no recovery, no classifiedGPUfaults on the same boot. Exact hashes,
checks, failures and content-free records: v161_mtp_placement_comparison and
[the active next-model guide](kuno-native-next-model-handoff.md).

Next source candidates remain separate: DOWNstride(ec367fb) needs protected
registration and physical gates; grouped PLE leaf33b4de7/controllercce4f66 is
clean in its isolated tree with13grouped/78controller checks and strict Clippy,
but unintegrated/unselected. Preserve86882f9 during its controller merge; keep
defaults off and validate unchanged full CPU13 before timing any new candidate.
Canonical Vashti/API/installedservices unchanged; required Vashti changes:none.
No release/install or other-task messaging is authorized by this handoff.


User steering after this receipt: continue native optimization with target and
full Q8 MTP on Intel B70 only. Defer Nvidia experiments; next physical candidates
must use the accepted Intel placement. This is internal benchmark selection,
not authorization for an untested installed/runtime/API default change.


Following Intel-only steering, grouped PLE sourceleaf9f8462e/controller513fe02
are integrated onto accepted HALfix86882f9. The exact contract is now in
[kuno-native-target-grouped-ple.md](kuno-native-target-grouped-ple.md). Root
integrated source review/CPU checks and protected guard/helper preparation are
underway. Both new policies remain default-off. Fresh full original physical
gates and separate Intel-only timing comparisons are still required; no new
performance, installed-service or consumer change is accepted by this source
integration alone.


### Intel-only grouped-read/source gate,2026-10-11

Grouped PLE source9f8462e/controller513fe02 and protected registrationde59c75
are now integrated above accepted HALfix86882f9. The integrated read-only review
finds no blocker;117 relevant Rust tests and strict locked pure-direct all-target
Clippy with two CPU jobs pass. Both original owning/fault proofs and unchanged
CPU13 reference/tolerances remain mandatory.27 protected-launcher tests and nine
new external fixed Intel-only PLE/DOWN/combined labels pass exact-selector and
original host-cap checks. Fake Xe metadata was used only in those source checks;
physical trials require a fresh privileged memory snapshot after draining.

Next artifact will independently gate grouped PLE on original cold/warm CPU13,
then compare normal compiler-idle Intel Q8 MTP code200 control/candidate. DOWN
stride stays off during that pair and needs its own original numerical/speed
gate before any combination. New policies remain default-off; source success
does not establish physical numerical, throughput or30tokens/s acceptance.
Root owns GPUs/services; canonical Vashti and installed/public behavior remain
unchanged. Required Vashti changes:none.


### Intel-only grouped PLE physical receipt,2026-10-11 17:41UTC

Frozen66bdd0f/v162 (SHA498736a0bacd6a6d31c9610931658fff0c04cdee790fa996f20a9331ae19a9d2)
passes unchanged originalCPU13 cold/warm target vectors/carries/greedy, real
repair, full1381MAIN/42draft/432HOST-page close and physical recovery. Warm
selection reports8actual completed groupedPLE chunks; cold false/zero. Source
gate117Rust/239Python/9fixed-label checks and strict directClippy pass.

Compiler-idle same-artifact ordered normalcode200 is9.886106control versus
10.022470groupedPLE continuations/s (199timed,130/138drafts,69rounds,5repairs).
Candidate reports74actualgroupedchunks and302readbackbatches versus450control:
148 fewer batches. Both still report618GPUwaits; scoped map intervals remain
about2.63s. This small1.38% preliminary timing does not establish a repeated
gain or removed actualGPUwaits. No math/owner/cap/reference change.

Latest recovery17:41:33UTC: B70baseline50688000B,Nvidia compute0, allHTTP200,
idle/unpaused/no recovery and unchanged classifiedfaults on sameboot. Exact
numeric/source/helper/model/admission/close evidence is under
`v162_intel_grouped_reads_and_down_schedule` in the root native receipt.
DOWNstride and combined original numerical/throughput gates remain pending;
keep them off in the PLEpair. Next separate source work prepares immutable
grouped GR/GDN/QSA status folds (defaultoff), preserving all original checks.
CanonicalVashti/API/installedservices unchanged; requiredVashti changes:none.
No30tokens/s, normalprose, independentCPU200, production or release claim.


### Completed Intel-only v162 schedule comparison,2026-10-11 18:00UTC

The remaining DOWNstride and combined groupedPLE+DOWNstride gates pass original
CPU13 cold/warm vectors/carries/greedy, real repair and full source/owner closure
on frozen66bdd0f/v162. Warm DOWN reports432layers/2160DOWN/4320source calls;
combined also reports8actual groupedPLE chunks. Original reference/tolerances
and all admission/owner limits remain unchanged.

Compiler-idle same-artifact normalcode200:9.886106control,10.022470groupedPLE,
9.567741DOWNstride and9.558671combined continuations/s, each199timed and
130/138drafts/69rounds/5repairs. DOWNstride reports3552layers/17760DOWN/35520
source calls; combined also reports74groupedPLE chunks. These are single ordered
measurements, not repeated gain estimates. Keep DOWNstride off for the next
internal speed comparison; retain the preliminarily fastest groupedPLE selection.

Final recovery18:00:12UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on the sameboot. Exact
numeric/source/admission/closure records join v162_intel_grouped_reads_and_down_schedule.
Next separate candidates are grouped status folds and packed-integer DOWN;
neither is physically accepted yet. Root owns shared GPUs and source isolation.
Required Vashti changes:none; installedAPI/services/canonical checkout unchanged.
No30tokens/s, normalprose, independentCPU200, production or release acceptance.


### Grouped-status integrated source gate,2026-10-11

Grouped status leaf/parent0cdb291 (originalb9ebef5) and public controllers453d2b7
(originalff2578e) are integrated after the completedv162 Intel-only schedule
comparison. Read the normative [status folds](kuno-native-public-grouped-status-folds.md)
and [controller contract](kuno-native-public-grouped-status-controller.md).
Policies remain independent/default-off; original equations, both whole fault
proofs, all-family commit and source/owner closure remain mandatory. Completed
counts publish only after accepted commits and join97GR/36GDN/12QSA per chunk.

Source gates:25focused status checks plus280resident regressions in isolated
parent,90controller/helperRust and33protected guard checks in isolated controller,
strict direct all-target Clippy/fmt/diff pass. Root integrated245textPython checks
and six fixed Intel-only status/PLE+status selector/cap labels pass; synthetic Xe
metadata used solely for source checks. Physical launch still requires fresh
privileged Xe accounting after paused/drained admission.

Frozenv163 source453d2b76bb74a03e5c6aacacc586d48f2fc806a5,
binarySHA256bddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a,
pure direct-device, is undergoing originalCPU13 acceptance. Same-artifact
compiler-idle Intel normal comparison remains required; source acceptance is
not physical numerical or speed acceptance. DOWNstride remains off. Separate
integer DOWN source/parent/controller work is isolated and unselected. Required
Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Completed Intel-only grouped-status comparison,2026-10-11 18:23UTC

Frozen453d2b7/v163 (SHAbddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a)
passes unchanged originalCPU13 cold/warm vectors/carries/greedy/realrepair and
complete1381MAIN/42draft/432HOST-page closure. Warm initialprefill stays
false/allzero;8accepted statuschunks join776GR/288GDN/96QSA.

Compiler-idle same-artifact orderedcode200:10.096522groupedPLEcontrol versus
9.951850groupedPLE+status continuations/s, both199timed,130/138drafts,69rounds,
5real repairs. Candidate reports74accepted statuschunks7178GR/2664GDN/888QSA,
removing31228fixed statusdispatches; GPUwaits584versus618control,302readbatches
each. No throughputgain is established; keepstatus off in the next speedcandidate.
Retain source and independent composition for later work, with defaultoff.
Scoped phases/waits/maps overlap and are not additive.

Last recovery18:23:18UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on sameboot. Full numeric
records, hashes/admission/closures: v163_intel_grouped_status_physical. Source
tests and limitations recorded there; no repeatedABBA/prose/CPU200/30t/s claim.

Separate integerDOWN leaf58b7f50(original730e135) and parent28d8ed6(original87a6fbf)
are source-integrated only. Leaf7focused/17existing oracle tests and strict
Clippy pass; independent read-only parent review finds no blocker. Parent
regression/Clippy and controller integration remain pending. Read both integer
DOWN normative contracts. Mandatory original physical accuracy/repair/closure
and matched timing remain required; no accepted new arithmetic or speed yet.
Required Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Intel-only integer DOWN integrated source gate,2026-10-11

Separate default-off integerDOWN is source-integrated: leaf58b7f50(original730e135),
parent28d8ed6(original87a6fbf), tested annotation/deltaee4f200(originaldce5b5b),
controller6d1bb47(original0ed0d59). Read the linked integer DOWN leaf/parent and
[controller](kuno-native-public-frozen-down-integer-controller.md) contracts.
The source keeps original experts/routes/packed weights, full-Q8 speculative
MTP scope and original reference/tolerances, with separate alternate math.
Intel combined admission adds exact65536MAIN+462848runtime while reserving
65536HOST separately;196608INFO is already included in runtime. Original144banks,
8GiBcache, fullQ8 owning inventory and512MiB driver margin remain unchanged.

Leaf7new+17existing quantizer/oracle checks, parent13focused/281resident/20source/
13MoE/4constructor/17existing oracle checks, controller99Rust/38guard checks
and strict direct-device all-targetClippy/fmt/diff pass. Parent counts overlap.
Independent read-only parent review finds no blocker. Root integrated250Python
checks and three fixed Intel-only helper labels/caps pass; synthetic Xe metadata
used only in source checks. Actual physical launches require fresh privileged
Xe accounting after paused/drained admission. No scope or profile fallback.

Frozenv164 source6d1bb47cd54937a2dd08cb05d11062313168c862,
SHAe5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b,
pure direct-device, is undergoing originalCPU13 accuracy/repair/owningclose
and physical recovery. Paired normal speed remains pending. Current trial
retains PLE/GUstride/sharedQ8Intel; original DOWNrows/stride and groupedstatus
stay off to isolate new DOWN arithmetic. Source success does not establish
physical accuracy or speed. Required Vashti changes:none; installedAPI/services/
canonicalcheckout unchanged. No30tokens/s or release acceptance.


### Completed Intel integer DOWN comparison, 2026-10-11 18:50 UTC

Frozen source `6d1bb47cd54937a2dd08cb05d11062313168c862`, v164, binary SHA256
`e5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b`, passes
unchanged original CPU13 cold/warm full vectors, carries, greedy IDs, real
repairs and complete owning closure. Warm first token reports actual integer
DOWN 48/240/48 and 480 source dispatches; warm full pass reports
432/2160/432 with 4320 sources. Original reference and tolerances are unchanged.

One compiler-idle same-artifact ordered code200 pair measured **10.023723 →
10.486196 continuations/s (+4.61%)**, each timing 199 real continuations,
130/138 accepted drafts, 69 rounds and five repairs. Integer DOWN reports
3552 preparations, 17760 calls, 3552 audits and 35520 original source calls.
Keep integer DOWN as a preliminary faster candidate with integer GU/stride,
grouped PLE and original full-Q8 shared head, all on Intel. Grouped status and
original DOWNstride remain off. Nvidia is deferred at the user's direction.
Single ordered timing is not repeated ABBA or prose acceptance.

All 1382 target owners/828997632 bytes, 42 draft owners and 432 host pages close.
Last physical recovery18:50:02UTC: Intel50688000B, Nvidia compute0, allHTTP200,
idle/unpaused/no recovery latch and no classified GPU faults on the same boot.
Full source tests, hashes, actual admission/selection and closures are recorded
in `v164_intel_integer_down_physical`. No independent CPU200/private/production,
installed defaults/API, 30tokens/s or release claim. Required Vashti changes:none.

Next isolated source candidates: transfer the existing exact packed IQ4 lookup
into actual Verify3 dense IQ4_NL/IQ4_XS (distinct immutable pipeline policy,
unchanged original addresses/scales/dots/reductions), and a finite target GPU
submission-backlog experiment retaining original submissions and proofs.
Neither has physical acceptance yet. Root coordinates GPU and CPU timing;
separate worktrees preserve source isolation. See the new normative contracts
once integrated. No frontend or API work is part of these candidates.


### Intel Verify3 IQ4 decoder completed; queueing candidate next, 2026-10-11

Frozen v165 source6266201b44faacac187db8e1f2353d35f7438b5a passes all original
operator controls plus9408 exact packed-register IQ4 values,12normal+4fault
Verify3 cases and12normal+42fault fused-finite cases on Intel. Default binary
SHA996786943fb7dfc6f41e1ae975cb0659833d8cd372b673d474cd7f0f83195ad0;
pure direct binary SHA5f86159edbad0652e77b60a19046d6ef5933e936e8677c65ec8dc3626e988719.
Original CPU13 cold/warm full vectors, carries, greedy IDs and real repairs
pass without changing reference80650af8 or ABS0.006/REL0.0003. Cold decoder
staysOriginal/zero; warm actual89first/801full selected dense calls. Full1382
MAIN/828997632B,42draft owners and432HOST pages close with physical recovery.

Compiler-idle same-artifact code200 ordered pair measures10.440895control vs
10.408871register decoder (-0.31percent), both199real continuations,
130/138accepted drafts/69rounds/five repairs. This shows no speed gain in this
single pair. Leave IQ4 registers off for the next speed candidate; keep its
explicit implementation and numerical acceptance. No ABBA/prose/CPU200 or
production/default selection follows. Latest recovery19:26UTC: sameboot,
Intel50688000B/Nvidia compute0/allHTTP200/no recovery latch/no new GPU faults.
See full source/actual counts/closures in v165_intel_iq4_registers_physical.

Next runtime candidate is [explicit bounded target backlog2048](kuno-native-public-target-backlog-2048.md),
leafc8bccc4(original525ce3b), clarificationad4af97(original45d944d),
controller77e421c(original3d94692), protected guard633ac25. It retains original
kernels, queue submissions and closing proofs; target is selected before model
construction for bothcold/warm while independent Intel full-Q8 draft remains512.
Its opaque driver retention is not an exact byte guarantee; fresh actual Intel
admission and unchanged512MiB margin/caps remain required.111controller/helper
checks (including8new),7exact-source state-harness checks, strict direct all-target
Clippy and259integratedPython checks plus6fixed helper labels/caps pass. The
harness limitation is recorded in the normative contract. Physical CPU13 and
same-artifact normal speed comparison remain pending. No GPU reset, installed
service/default/API/frontend change or30tokens/s/release claim. Vashti changes:none.


### Intel target queue comparison complete; placement next, 2026-10-11

Frozen v166 source0d35d5c7d1f7daf040622be61009f4770ce2616b,
pure-direct binary450e32a18ff3b9f3d985a27cb9550564c6b7c29c83bdfb09ba0c8aa3199da045,
passes unchanged original cold/warm CPU13 full vectors/carries/greedy/real repair,
actual target2048/minimum1 with independent draft512/minimum0, full1382MAIN/
828997632B,42draft and432HOST close and physical recovery19:41:31UTC.
Actual selected runtime was constructed before model allocation for both passes,
and remained immutable through completed close. No kernel or route changed.

Same-artifact compiler-idle ordered code200 pair10.451238original512 vs
10.482525queued2048(+0.30percent), both199 real continuations/130of138 drafts/
69rounds/five repairs, shows no material gain in one pair. Keep original512
for the next placement candidate. Global submission waits618→281 and
8.294789→1.668612seconds, while mapped-read wait2.472111→8.967485seconds;
work largely moved to a different wait boundary. These overlapping timers
are not isolated GPU kernel cost and cannot be added as a removable speed gain.
Full numerical hashes, actual policy/counters/admission and closing evidence
are in v166_intel_target_backlog_physical. Integrated strict release/direct
all-targetClippy passed. No ABBA/prose/CPU200/private/production or30tokens/s claim.

Latest physical snapshot remains sameboot Intel50688000B/Nvidia compute0,
allHTTP200, idle/unpaused/no recovery latch and no new classifiedGPU faults.
Nvidia remains deferred. Installedservices/defaults/API/frontend unchanged.
Required Vashti changes:none.

Next isolated source preparation is setup-only public profile-ranked empty-bank
fill: preserve all existing prompt/cold-pass residents, use the pinned public
Strata ranking only for empty slots, then freeze the original144/layer bank,
all512 actual IDs and sorted368 HOST complement. No private prompt or future
route profile is recorded and no capacity/arithmetic/route change is selected.
Read docs/kuno-native-public-profile-ranked-bank-filler.md after leaf integration;
its exact artifact/derivation, full owner/cap and CPU13coverage limitations are
authoritative. Protected scope263Python checks +three fixed helper labels/caps
pass; leaf/controller numerical/physical/speed acceptance remains pending.
Source CPU13 selected preservation may be all-hit: original diagnostic HOST
coverage remains a separate required control, never fabricated. A separate
borrowed CPU-batch leaf is being prepared to compute existing RAM-resident
expert weights with original Rust AVX2 math; it is not whole-model inference
or a speed claim. Root owns cache scheduling, shared GPU tests and physical
recovery; separate source branches retain ownership boundaries.


### Intel public ranking tested; CPU/GPU splitting next, 2026-10-11

Fixed profile source is integrated through e5f1689, after normative7c9da20
(original192d662) and leaf1d14a14(original53e889a). Read
[kuno-native-public-profile-ranked-bank-filler.md](kuno-native-public-profile-ranked-bank-filler.md)
and its controller contract. Frozen v167 pure-direct source is
`e5f168916cfeccca051f53b4c5ead8958fbe90f2`, binary SHA256
`03379e116acddc20c0bbf73f253913f8156145f592e8fbd5b4fd635edf1c0430`.
Both separate canonical and selected cold/warm CPU13 gates pass unchanged full
vectors/carries/greedy/real repair, original tolerances, actual original512
queue/full-Q8 Intel MTP, full1382 MAIN/828997632B,42 draft and432 HOST closures.
Selected warm publishes48 layers,4121 filler/2791 preserved residents and
actually serves9120 VRAM/zero HOST members; the separate canonical diagnostic
serves2590 VRAM/6530 HOST. Preserving the preceding cold replay's residents can
make that replay all-hit; it cannot replace HOST coverage or credit a profile
benefit on unseen normal generation. The warm12-token reference-check intervals
are6.647453 versus11.553927 accepted tokens/s, not normal200-token throughput.

The compiler-idle same-artifact ordered normal code200 pair is10.482354 original
versus10.478727 profile filler (-0.035percent), both199 real continuations,
130of138 drafts,69 rounds and five actual repairs. Actual VRAM/HOST members
are56344/46376 versus56079/46641; VRAM share54.852percent versus54.594percent.
Selected normal setup preserves5143 residents and fills1769 empty slots;
actual48-layer publication and unchanged original full512 map are accepted.
There is no measured gain. Keep Original bank filling for subsequent trials,
original512 queue, original IQ4 decoder, grouped PLE and integer GU/DOWN/stride
with shared full-Q8 head; Nvidia remains deferred by the user. Counters are
routed members, not unique weight reads or PCIe bytes.

Exact source, original numerical and setup receipts, aggregate tiers, complete
owner/closure and recovery are recorded in
`v167_intel_profile_filler_physical` of the native Strata numeric receipt.
82 distinct leaf/parent checks,120 controller/helper checks,263 protected Python
checks plus three fixed labels/caps, and integrated strict release/direct
all-target Clippy pass. Latest actual recovery is20:09:26UTC, same boot,
B70 used50688000B, Nvidia compute0, allHTTP200, admission idle/unpaused without
recovery latch and no new classified GPU faults. This is one ordered code pair;
no ABBA/prose/independent CPU200/private/production/30tokens/s acceptance.
Installed services, selections, public API, frontend and release are unchanged.
Required Vashti changes:none.

The separate borrowed CPU batch source is integrated as2e51bd5(original3212bc2),
with53 CPU expert checks passing and two unchanged timing probes ignored,
strict CPU/native direct all-target Clippy, formatting and independent review.
Read [kuno-native-borrowed-cpu-batch.md](kuno-native-borrowed-cpu-batch.md):
counts1..3 use original packed F32/AVX2 arithmetic, joined workers and sparse
unweighted original-rank outputs. This leaf is default-unused; arbitrary spans
do not authenticate payload origin. No connected CPU/GPU inference or speed
claim follows. Separate source owners are preparing an Attachment-anchored
read-only borrow of the SAME original Vulkan RAM and an actual one-layer CPU
versus HOST-GPU partial comparison with readback/encoding/upload/join included.
No duplicate model payload, disk read, new arithmetic kernel, cap relaxation or
count3 merge is authorized by an operator result. Initial fixed-layer synthetic
operator measurements may be unusually L3-hot; actual whole-model CPU13 and
normal timing remain later gates. Root controls physical trials and compiler
scheduling; agents use separate worktrees/caches and cannot independently use
GPUs, change services or release.

### Attached CPU operator accepted; Intel MTP repair candidate next, 2026-10-11

Continue Intel-only target/full-Q8 MTP optimization. Native30tokens/s remains
unachieved; latest normal selected control is v167 at10.482354 continuations/s.
The current owner checkpoint is [the native-model handoff](kuno-native-next-model-handoff.md).
Canonical Vashti, installed selections and public protocols are unchanged.
Required Vashti changes:none.

Actual original HOST CPU operator v172 is frozen from
`47b972bf73f79c04acee3c87e922526203c54099`, binary SHA256
`107958180ec3334cc6e1c84b65976600b790c94539f12ca04cbd6c4f6e06175a`.
Read [the operator contract](kuno-native-public-host-cpu-batch-probe.md).
All45 synthetic one-layer case points/180 pairs pass original full
partial/inactive/padding/source/fault checks, maximum absolute error1.071021e-8
under unchanged1e-4/3e-4 bounds. Original coldCPU13 and warm token0 full-vector
checks pass. Actual original432 HOST allocations have HOST_VISIBLE,
HOST_COHERENT and HOST_CACHED mapping flags; no copied CPU model was introduced.
All11 operator owners/2392064B,1380 target MAIN owners/828866560B and432 HOST
pages/36147609600B close. No MTP ran in this operator diagnostic.

Complete15-worker CPU handoff/compute/upload means at count3/fiveHOSTranks are
2.324ms versus1.763ms original-F32 GPU,32percent slower; at tenHOSTranks3.394ms
versus2.852ms,19percent slower. Keep GPU execution for the current batched target
path. Some count1 points favor CPU; a single L3-hot synthetic layer and three
repetitions do not establish48-layer hybrid benefit or performance against the
selected integer GPU kernels. No connected CPU hybrid/whole-model speed claim.

The prior v171 failed private readback setup after original coldCPU13/warm0.
Its repeated full-owner audit stages necessarily exceeded the unchanged128MiB
I/O ceiling. Existing audited cleanup after every completed pair fixes the
lifetime without new runtime policy/cap/vendor code;180 actual checkpoints
inspect919937280B and take432.422ms outside both operator arms. Exact failure
and successful numeric/source/owner/recovery receipts are separate keys
`v171_attached_cpu_operator_readback_setup_failure` and
`v172_attached_cpu_operator_physical` in the native-Strata JSON. Latest physical
recovery at21:23:44UTC is same boot, B70used50688000B, Nvidia compute0,
allHTTP200, admission idle/unpaused/no recovery latch, classified faults0.

Source-only candidates integrated: original-math joined MTP repair leaf
`ed7b345` (original`0fa9cc3`) and guarded controller`e37ae50`
(original`012320c`); resident-bank ownership telemetry`f386606`
(original`859c2a5`). Read
[the repair contract](kuno-native-mtp-batched-repair.md) and
[the telemetry contract](kuno-native-resident-bank-telemetry.md).
49 MTP checks,131 controller/helper checks,9 telemetry/attachment checks and
strict direct all-target Clippy pass on their separate source trees. Root's
271 protected-launcher checks and three fixed helper scope/cap checks pass.
Physical originalCPU13/full closure/recovery and a same-artifact normal speed
comparison remain required for MTP repair. Telemetry distinguishes actual
backing padding from unused conservative admission headroom; neither is yet
measured on-device or credited as additional capacity. A separate four-row
dense leaf is being prepared without changing original count3 behavior.


### Intel MTP repair and exact bank backing accepted, 2026-10-11

Frozen v173 source970f8f4141adc903212fe102ada504b4a42edfe8, pure-direct binary
69bc565a1396d85d9f9fdaa702620d94b44795c15b4541329f7f4d502c8ec0ec, passes
unchanged originalCPU13 cold/warm full vectors/carries/greedy/actual repairs.
Same-artifact compiler-idle ordered code200 pair measures10.533053 original
scalar versus10.603973 batched repair continuations/s (+0.67percent). Both199
real continuations/130of138 accepted drafts/69 rounds/five real target repairs.
Actual new69 joined repair calls/130 cells/69 fault gates reduce MTP reads
337to207. Repair phase1.802to1.590s includes true target recomputation/checks,
not isolated readback cost. The explicit leaf is retained; original scalar
repair remains the separate four-row candidate control. One pair does not
establish repeated/prose/independentCPU200 or30tokens/s acceptance.

Exact48-layer/288-buffer bank telemetry reports14,144,716,800B logical payload,
14,557,240,448B actual buffer backing,412,523,648B padding and4,166,132,608B
unused conservative allowance. Attached/close snapshots match. The latter is
not free physical VRAM or additional admission credit. No capacity or GPU
owner changes. Complete1382MAIN/828997632B,42 draft private owners/11698176B
and432 HOST/36147609600B closure passes. Last normal recovery21:43:44UTC:
same boot, B70used50688000B, Nvidia compute0, allHTTP200, admission idle and
unpaused/no recovery latch, classified faults0. Exact sources/actual counters/
admission/closures are in v173_intel_mtp_batched_repair_and_bank_backing_physical.

Next isolated source candidate is genuine four-row target verification with
three actual Intel full-Q8 MTP proposals, original scalar repair and unchanged
original tails/caps/reference. Read the normative parent/dense/MoE/controller
contracts when integrated. Physical acceptance remains separate. Canonical
Vashti, installed services/defaults, API and release are unchanged. Required
Vashti changes:none. Nvidia remains deferred at the user's direction.

### Intel four-row operator acceptance and constructor rejection, 2026-10-11

Current owner source42df907 reserves genuine four-row lookahead only in the
fixed-public Intel target/full-Q8 MTP lab. v174 independent Verify4 plain/fused
operators pass original scalar-GPU/CPU-F64, all10 formats,20 numeric each,
20/70 fault cases and complete source/owner zero checks. Original CPU13
cold/warm full vectors/greedy/carries/real repair and complete closure pass.
Selected four-row CPU13 fails before initial count1 prefill acceptance:
GR incorrectly proves the constructor-reserved four-row logits owner using
its older three-row logical extent. No third-proposal/count4 or normal speed
acceptance follows. Carry fixed owner capacity into the exact proof, preserve
original arithmetic/reference/caps, freeze and retest before timing.

See the latest owner handoff and
v174_four_rows_operators_and_initial_owner_failure in the numeric receipt
for exact source/binary/guards, original controls, failure and recovery.
Latest physical recovery22:32:25UTC: same boot/kernel, B70 used50688000B,
Nvidia compute0, allHTTP200, admission idle/unpaused/no recovery and no new
classified GPU fault. These correctness runs may overlap CPU source checks;
diagnostic rates do not establish compiler-idle normal throughput.
Native30tokens/s remains unachieved. Installed services, public contract,
canonical Vashti and frontend unchanged; required Vashti changes:none.

### Intel-only native optimization continuation, 2026-10-11

Latest user direction is Intel B70 target and full-Q8 MTP only; defer Nvidia.
Latest accepted normal comparison is v177 on26295f6:10.442972 original
three-row versus10.740576 genuine four-row real continuations/s (+2.85percent),
one compiler-idle same-artifact ordered code200 pair. Both emit200/time199
real continuations, complete all real repairs with zero repaired-vector error,
close every original target/draft/HOST owner, and recover normal services and
same-boot B70baseline50688000B with no new classified faults. Four accepts
147of155 proposals over52 rounds/51 actual count4 chunks; original130of138
over69 rounds. Independent third-draft CPU oracle, repeatedABBA/prose and
independentCPU200 acceptance remain absent. Native30tokens/s is unachieved.

Original and selected cold/warm full CPU13 vectors/carries/greedies and real
repair pass on the same artifact. The unchanged independent v174 Verify4
operators are reused after three arithmetic-free setup repairs: immutable
reserved head capacity (`e6baa7e`), mandatory original direct-fill attachment
(`2dbc897`) and actual frozen-four authority (`26295f6`). Original direct
demand loading stays capped at3. Natural prefixes1..4 occur across CPU13 and
normal fixtures. Failed v174..176 runs retain their distinct failure receipts.
Exact source/owners/recovery and limitations are in
v177_actual_frozen_four_cpu13_acceptance and
v177_original_vs_four_normal_intel_pair.

Read the current [owner handoff](kuno-native-next-model-handoff.md) and
[four-row contract](kuno-native-target-four-rows.md). GDN's retained-metadata correction (`cd638c7`) passes original/four cold/warm
CPU13, full closures/recovery and normal code200. v17810.704157 versus
v177four10.740576 gives no measured gain (about0.34percent slower in one
frozen comparison). Real work/repair/prefix counts match. Its35 source checks
and strict direct Clippy pass, with original arithmetic/shaders/caps unchanged.
See v178_gdn_retained_metadata_cpu13_acceptance and
v178_gdn_retained_metadata_four_normal. A bounded original three-row CPU-HOST/GPU-resident overlap operator is
registered on d587d9b (source/admission checks pass). First v179 physical attempt
fails a sparse negative fault check after earlier case progress; no timing or
completed numeric credit is claimed for that failure. Correction `47f4c8e`
passes v180 original CPU13/warm token0, all five cases/75 full arm calls,
all11 negative controls and complete source/owner/recovery checks. Concurrent
CPU/GPU wins disjoint HOST work against the original-F32 control, but loses
the other actual mixed cases. No whole-model or selected-integer GPU speed
claim/consumer is granted. Same-boot normal services/B70baseline50688000B
recover without a new classified fault, reset or reboot. See the
[operator contract](kuno-native-public-cpu-host-overlap.md) and full receipt
`v180_attached_cpu_overlap_physical`.

The user paused the goal2026-10-12UTC; the current benchmark/source block
is complete and no further GPU run is started. Native30tokens/s remains
unachieved; latest whole-model
result remains10.7tokens/s. Unused fused-four MTP repair leaf `bd46b2b`
(original `7ea302c`) passes56 focused tests/strict direct Clippy. Approved
controller composition `83967bc` (original `fc0948c`) passes225 distinct
focused tests/strict direct Clippy. Only actual count4 uses fused repair,
tails keep original repair, and actual completion counters join both paths.
Read the [exact controller contract](kuno-native-public-four-batched-mtp-repair-controller.md).
Root combined strict direct all-target Clippy, full native formatting and diff
checks pass after integration; compiler/CPU/GPU test jobs are idle. The source-only
receipt is `paused_intel_only_mtp_four_repair_source_checkpoint_20261012`.
Protected activation and
unchanged physical CPU13/repair/full timings remain pending. Old1..3/default
paths, math,42 owners and all caps stay unchanged. Read the
[owner checkpoint](kuno-native-next-model-handoff.md) before resuming.
The unused exact persistent144 bank leaf
(`e16ac7a`) grants no additional physical capacity or slots. No API, installed
service, canonical Vashti, frontend or release change; required Vashti
changes:none.


### Latest native text checkpoint — paused after measured profiling (2026-10-12)

This supersedes the preceding pending fused-four-repair checkpoint. User stopped
the Astra manual pass to conserve usage. Native goal remains paused, no queued
CPU/GPU work. Fused-four repair and optional integer target head pass full
CPU13/closures. Normal v182 control10.882673 versus integer16head10.880799
real continuations/s is neutral, with matching generation/acceptance counts.
Paired32 head passes CPU13 but has no normal-speed result. No30tokens/s claim.

v185 onaa90a95 physically profiles the first warm target round: expert GU/DOWN
consume69.86% of250.572596ms measured kernel time; BF16UP only1.04%. Shelve
source-only BF16pair a9cff9e; next approved design is GPU-only compact Source
group lists, not implemented. Preserve exact routes, original arithmetic,
full audits and scratch-owner limits. Resume from the
[owner handoff](kuno-native-next-model-handoff.md) and
[profile contract](kuno-native-public-target-profile.md), with source/artifact,
checks, design and limitations recorded there and in the aggregate receipts.

Fresh02:12:05UTC recovery confirms Kuno/Ollama/Vashti HTTP200, admission idle
and unpaused, B70baseline50688000B, Nvidia compute0 and all classified faults0.
Same boot/kernel38. All root/agent compiler and GPU jobs are idle. Canonical
Vashti, installed defaults, public API, frontend and release remain unchanged;
required Vashti changes:none. Do not automatically resume work.

### Native text checkpoint — count4 compaction implemented and committed (2026-10-12)

Resumed on user instruction after an explicit pause. Per-Source expert group
compaction is now implemented and committed as c42379b on
codex/kuno-native-text. The GPU prepare leaf builds five disjoint shard
records in the already-owned zeroed tail (words 6144..6383 of the existing
full 64KiB owner): ready flag, count, the original GU/DOWN dispatch triplets,
and the stable ascending original group IDs for the unchanged partition ranges
0/72/144/288/432. GU and DOWN then run one indirect dispatch per shard through
checked 12-byte views of that owner, so each leaf reads only its own shard's
groups instead of scanning the routed-group union.

Arithmetic, decoder, member order, routes, IDs and owners are unchanged: both
compact projections reuse the original body verbatim, count1..3 stride-four and
all other schedules keep their original paths, and the new record audit
reconstructs each list from the unchanged MAP, validates headers, counts,
triplets, identity and inactive zeros, stickies faults, and erases the records
only after every shard validates. Counters are checked: each compact leaf
rejects unauthenticated or short views, and the audit refuses a 16384-word
scratch. Host receipts now require actual compaction credit (five GU and five
DOWN indirect dispatches plus one record audit per layer) and both daslab
probes report those counters as evidence.

Checks: full native-text suite 1334 passed/0 failed/1 ignored, clippy
-D warnings clean, fmt clean. No GPU trial has run yet. Admission is idle and
unpaused, but an unrelated 13.3GB worker process currently holds the B70, so
the numerical and normal gates must wait for that GPU to be free. Frozen v186
artifact build is in progress from c42379b. No performance claim yet.

Canonical Vashti, installed defaults, public API, frontend and release remain
unchanged; required Vashti changes: none.

### Latest native text checkpoint — stopped after completed kernel comparisons (2026-10-13)

This supersedes the preceding compaction source-only/interrupted checkpoints.
User requested finishing the current experiment and stopping until tomorrow;
automated goal remains paused and no work is queued. Canonical Vashti is clean
and unchanged; this work stays in the Kuno native-text laboratory worktree.

Inherited compaction is fixed in `62594d0` and `fb7b420` and passes full original
cold/warm CPU13 plus normal code200. Normal10.906tokens/s is neutral against
prior10.883. Primary-plane GU (`fd9c319`/`fdd2732`) failed the unchanged warm
carry fixture and is disabled. Exact-tree subgroup GU/DOWN
(`7d01db5`/`70fd470`/`b9ee1be`) passes CPU13 but measures10.955tokens/s, neutral.
Exact activation-staged DOWN (`41b9d3d`/`d8b58b6`) passes full CPU13 and measures
10.872tokens/s, also neutral. Both experimental selectors are off in `9dc764d`.
Original numerical thresholds, target routes and two-plane arithmetic stay intact.

Full selected-v197 direct-device suite:1346pass/1ignored; strict release
Clippy/fmt pass. Frozen source/artifact identities, real timings, failed/neutral
evidence, specifications and exact resume directions are in the updated
[native owner handoff](kuno-native-next-model-handoff.md) and aggregate receipts.
The short warm CPU13 fixture improved20% with staging but normal generation did
not; next work should profile the representative normal run and verify actual
count4/tail execution before further kernel changes.30tokens/s is not achieved.

Fresh06:54:29UTC protected cleanup: Kuno/Ollama/Vashti HTTP200, admission idle
and unpaused/no latch, B70used50688000B, Nvidia compute0, all classified GPU
faults0, unchanged boot/kernel. No benchmark or compiler processes remain.
No installed binary/default, public API, frontend or release change. Required
Vashti changes: **none**. Do not resume automatically.

### Native text: broad Rust Strata port checkpoint (2026-10-13)

The user explicitly requested a coherent Rust port first, followed by full-model
execution and bug fixes, superseding incremental original-CPU13-precision gates
for this isolated experiment. Work is now on `codex/strata-rust-port` in the
sequential Kuno worktree. Original accepted engine remains on `codex/kuno-native-text`
at6792453; canonical Vashti is untouched. This is backend laboratory work and
requires no consumer/API/frontend change.

Prototype target math/scheduling and suffix/Q2 MTP controller are implemented;
source `f0ed07a` passes build,20full WGSL/SPIR-V compiler variants, both pure
registration presets, strict port Clippy, and1346original-engine regression tests
(one ignored). It reuses existing Rust model operators. Full Strata feature
coverage is not claimed: important cache/prefill/MTP/context features remain.

**No GPU execution, quality or speed acceptance yet.** Cached sudo expired;
one shared-shell prompt is pending. No services/models/GPUs were changed or
loaded. Fresh unprivileged checks show all three HTTP listeners200, B70baseline
50688000B, Nvidia compute0. Exact scope, immutable v204 artifact, defects fixed,
limitations and first full-model command are in the
[port checkpoint](kuno-strata-rust-port.md) and
[owner handoff](kuno-native-next-model-handoff.md). Do not use earlier port
artifacts with the shader-cache identity bug; retain both dtype/variant and mode
in one KernelId info tuple. No deployment/release or automatic work is queued.

### Latest native text port checkpoint — paused after combined test (2026-10-13)

This supersedes the preceding source-only and early combined-port checkpoints.
User requested pausing after v221 and returning both GPUs. Automated goal stays
paused; no benchmark/compiler job, sudo refresher or queued GPU work remains.
Canonical Vashti, deployed binaries/defaults and API/frontend are unchanged;
required Vashti changes: **none**. No release acceptance.

Runtime source is1caf271 oncodex/strata-rust-port. Original accepted engine remains
6792453 oncodex/kuno-native-text. Broad port now connects parallel routing,
compact real argmax/carry, Rust/RAM Q2 dense conversion,106299-token proposal
vocabulary, suffix policy and request-local adaptive two-way cache. Newest source
has exact persistent192-slot banks and a GPU-driven VRAM/mapped-RAM exchange.
The complete v215144-slot/staged-exchange run measured8.458796 continuations/s
through362 emitted tokens.30tokens/s and the full upstream rewrite remain unmet.

v221 reaches cold prefill/materialization but rejects a four-row DOWN launch
using two old partition tables. d682e04 fixes those;1caf271 passes strict port
Clippy/fmt and original direct-device compilation. **No GPU rerun after the fix**;
new direct GPU exchanges have no hardware/timing acceptance. Current full
original-engine regression is still outstanding (1346/1ignored was f0ed07a).
Do not reuse the failed v221 artifact as working. Read the exact scope, artifact,
failures, next acceptance and remaining verifier/prefill/rollback/KV work in the
[port checkpoint](kuno-strata-rust-port.md) and
[owner handoff](kuno-native-next-model-handoff.md) before explicit resume.

Fresh16:29:23UTC read-only stop receipt: Kuno/Ollama/Vashti200, admission idle and
unpaused/no latch, B70used50688000B, Nvidia compute0, classified faults all0,
unchanged boot/kernel38. Recorded refresher3421970 exited. No prompts, outputs,
private histories, API changes or frontend work were persisted/introduced.

### Latest native text checkpoint — resumed combined optimization (2026-10-13)

User explicitly resumed goal/GPU work; this supersedes the paused/v221 entry.
Accepted v224/0f4bfcc produces12.302385code and12.241920prose continuations/s.
192-slot direct GPU/RAM exchange and retained-input state-only prefix repair
work; code36 and prose95 partial repairs require no full target reruns. Both
fixtures complete private/source closure and actual GPU/service recovery.
30tokens/s and full Strata parity remain unmet. v225 integer head and v226 fused
five-tier projection are unaccepted candidates under combined validation.
See exact sources, frozen hashes, measurements, limits and next work in
[port checkpoint](kuno-strata-rust-port.md). No frontend/API/deployment change;
required Vashti changes:none. Normal source/physical bounds and transient
privacy remain despite approximate port arithmetic. Canonical checkout preserved.

### Native text continuation — queued MTP accepted (2026-10-13)

v231/01e74a7 reaches12.755259code and12.598536prose continuations/s. Consecutive
real MTP proposals now stay queued until one final private fault/read fence;
target verification/cache repair unchanged. One additional admitted/erased32KiB
MTP packet; port closure43owners/11,730,944B. Both actual recovery receipts pass.
Head/five-tier fusion tested neutral/slower and remain optional, off by default.
v232 router/v233 aligned expert reads are candidates under validation. Exact
contracts/measurement limits and incomplete full-rewrite scope are in the port
checkpoint. No consumer changes, installed default or release acceptance.

### Native text active checkpoint —13tokens/s and profile-directed next work

v234 default prose13.000929; v232 normalcode12.796036. Latest3e01731/v236's
optional8promotions/layer gives13.004114code with81.66%resident routes; larger
transfer cost offsets much of faster verification. Cache-eight prose is active.
All finished runs erase43MTP/1562target owners and physically restore services,
GPU memory and unchanged kernel faults. Current strict port Clippy and original
1346pass/1ignored; full49shader variants indefault/16/wide atv234.30 and full
Strata parity unmet. Head/fusion/wide/16variants remain optional/off. The new
profile points to mapped-HOST expert work; port guide gives exact next6/8-row
suffix/staging directions and their ownership/shape limits. No API/UI/installed
or release changes; required Vashti changes:none. Goal active; old stop notes
historical. Do not infer the bounded sudo helper's availability from notes.

### Completed cache-eight checkpoint — active goal, recovered host (2026-10-13)

v236/3e01731 optionalcache-eight:13.004114code,13.189027prose. Both source/private
closure and physical GPU/service/fault recovery pass. Clippy and original1346
pass/1ignored pass. Benchmark/compiler jobs ended; bounded refresher stays for
ongoing authorized goal. Default cache2,optional8; no deployed setting/API/UI
change.30still unmet. Port guide now records concrete6/8-row source/ownership
work needed for the next coherent port step and profile-based transfer options.


### Latest native text port checkpoint — paused, GPUs returned (2026-10-13)

Supersedes the preceding active-goal notes. User requested a breaking point;
goal is paused, no GPU benchmark or sudo refresher remains. Kuno/Ollama/Vashti
HTTP 200, admission idle/unpaused/no recovery latch, B70 50,688,000 bytes,
Nvidia compute zero, all classified GPU faults zero. No installed/API/UI/release
change; required Vashti changes **none**. Canonical checkout remains untouched.

On `codex/strata-rust-port`, the optional eight-row Rust verifier now completes
real whole-model target verification and state-only partial-prefix repair.
Runtime source `2780e7a`, frozen SHA256
`33a568e1b2fa6b492381e14c6488c8f80fac12c4611a2d73ac4c457bf556bff4`:
12.372330 code with adaptive five/eight-row windows; normal four-row MTP on the
same binary 12.894535. Both emit 362 including EOS and fully erase private owners.
Source-only helper cleanup follows at `383248a`; strict Clippy and original
regression 1,346 passed / 1 ignored. Frozen eight-row shaders and prefix tests
also pass. v236/`3e01731` remains the
selected faster candidate: 13.004114 code / 13.189027 prose; **30 is unmet**.
Read the [current paused port handoff](kuno-strata-rust-port.md) and
[aggregate receipt](../apps/kuno/benchmarks/text/receipts/strata-rust-port-20261013.json)
for exact identities, checks, packet-receipt caveat and remaining coverage.
Next candidate is selected HOST expert staging/promotion, not API integration.
Do not infer sudo authorization or free GPUs from these historical notes.


### Latest native text checkpoint — combined residency complete; coherent port next (2026-10-13)

This supersedes preceding native-engine next-step proposals. User directs a
connected rewrite of the pinned MIT Strata Intel core before whole-model speed
judgment, followed by integration debugging. Do not performance-gate individual
ingredients while upstream dependencies remain missing. This remains native
backend laboratory work; no Vashti/API/connector/UI changes are required.

Working fallback v260, runtime source
`86a98a4def023ce4bd5834a0cdf40087dd938f1e`, SHA256
`b34e3b83dc3fa5297199ef22ef1e15f1ee4860a65b1d40f0768f014b96667392`:
14.063997 code / 14.300613 prose timed continuations/s on Intel B70 only, public
seed424242/temp0 fixtures. Exact static allocations, 5 GiB static cap, 240
resident experts/layer, ranked empty-slot fill and eight cache promotions are
combined. Four-row MTP, real target verification, 36/95 state-only repairs,
zero full target recomputations. These are single runs, not matched-output
Ollama controls or private/production acceptance. **30 and full rewrite unmet.**

Later optional full-profile initialization v261 gives13.884123code; single local
negative comparisons do not reject the complete upstream design. Latest checked
source a0bd715 passes strict selected port Clippy, two targeted port tests and
1,346 original direct-device tests /1ignored. Frozen v260/v261 shader sets pass.
Read the [current port checkpoint](kuno-strata-rust-port.md) and
[aggregate receipt](../apps/kuno/benchmarks/text/receipts/strata-rust-port-20261013.json)
for exact commits, options, limits, failed startup trials and remaining gaps.

At23:11:21UTC all three listeners200, admission idle/unpaused/no latch,
B70used50,688,000B, Nvidia compute0, classified GPU faults0. No benchmark/compiler
or sudo refresher remains; goal controller paused. Development branch is
`codex/strata-rust-port` in the separate `kuno-image-speed` worktree; canonical
Vashti and deployed binaries unchanged. One graceful Kuno restart evicted idle
image residency before trials; no device reset/rebind/reboot. No frontend handoff,
release or installed performance claim is implied. Next scope is connected
placement/weight-access/router/group-execution/quantized-kernel/verification/state
commit parity with explicit Vulkan substitutions, keeping v260 as fallback.


## Native Intel coherent core checkpoint (2026-10-14)

Backend-only work stays on `codex/strata-rust-port` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`; canonical source and
installed Kuno/Ollama/Vashti binaries/configuration are unchanged. Read the top
and **Native Intel core receipt** in [the port handoff](kuno-strata-rust-port.md)
for exact reproduction/limitations. No API, protocol, runtime ABI or UI change;
**required Vashti changes: none**. Do not package this fixed-public laboratory
binary as an accepted private text worker.

The connected range after `ca2cadb` includes native Rust/ash Vulkan command
recording, retained buffer ownership/initialization, shared IQ codebooks, native
physical expert reads, stable parameter pages and ordered private copies/fills.
Key commits: `0e6f2b7` initializes Wgpu ownership correctly; `049d493` tracks real
buffer dependencies; `2f53b6c`/`66b198d` lower/check physical pointers;
`1c9c9d8` preallocates pointers and stabilizes parameters; `617537b` orders private
copies; `a68ac87`/`f2d4048` match subgroup32 and record large replay counts;
`55adf7d` adds six packed dense attention formats; `9dc6eeb` stages selected
original RAM experts in three genuinely owned64MiB private GPU roots;
`9be8a65`/`44bb1d1` charge/check those owners at construction, physical admission
and closure; `a0c6c3c` bounds the complete stage union. Retain licenses and the
source/error-order fixes with the optimized path.

Frozen v282 source `a0c6c3c9c21745faed6ee7362983bcd00f5a72e6`, SHA256
`9b41ea8fb312ad1fd1ddb3f5986eb479832c5ab9d173c0ffdf51188c102f38c6`, completes
real DASLab IQ3_S/Q2 MTP code14.625229 /prose13.546277 timed continuations/s.
139 /165 large verification graphs replay. Actual target erasure proves1565
buffers /1,064,534,016 bytes; MTP43 /11,730,944 bytes; all original host source
pages detach, B70 returns to50,688,000 bytes, NVIDIA compute processes0, classified
GPU faults0 and normal HTTP/admission recover. Source checks: strict core Clippy,
fmt/whitespace,56 WGSL/SPIRV sources including13 physical `spirv-val` modules,
and1346 original direct-device tests /1 ignored. Full Strata-feature unit suite,
independent whole-model state oracle, private worker/API acceptance and long soak
are not established.30t/s and complete upstream parity remain unmet. v260 is
retained as a faster prose fallback; generated token sequences/lengths differ,
so these are not same-output speed ratios. The existing goal is active and
continues; repository notes must preserve that unfinished scope.


## Native Intel packed shared/MTP and reuse checkpoint (2026-10-14)

This supersedes the preceding native-engine source/checkpoint details. Backend-only
work remains in the isolated `kuno-image-speed` worktree on `codex/strata-rust-port`.
Canonical/deployed source, API, protocol, UI and services are unchanged;
**required Vashti changes: none**. Read the top of
[the current port handoff](kuno-strata-rust-port.md) and aggregate receipt's
`native_intel_packed_extension` before resuming.

Changes `0dad3af`, `50a112e`, `e5bd05d` connect packed shared gate/up/down, scalar
MTP projections/indexed head, subgroup32 input preparation, and GDN/QSA input
reuse. Frozen v285 source `e5bd05d429c0f37dceb6a54735ad0b9ea39c424e`, SHA256
`1319a190a246ab7adcff7055629021db4672a128ddd5e944de3350189f5a69d2`:
14.512497code /12.781750prose timed continuations/s. Code improves over v283/v284
in this single trial, prose does not. Sequences/windows differ; independent
numerical/quality acceptance is absent. Keep v282/v260 fallbacks.30 and complete
Strata parity remain unmet; goal remains active.

Strict core Clippy,65 shader variants including13 physical SPIRV checks, and
original regression1346passed /1ignored. All trials restore actual B70 idle
50,688,000B, NVIDIA compute0, classified faults0, HTTP200 and normal idle admission;
full private target/MTP owners erase and original host pages detach. No reset,
reboot or install. Full core unit suite, private worker/API acceptance, whole-state
oracle and soak are not established. No benchmark/compiler remains; the bounded
sudo helper may still be live and must be inspected, not assumed. Next scope:
public operator timing, lower-register expert scheduling and connected GR fusions,
with packed-epoch/vector verification for the changed prose path.


## Active native-text direction — complete Rust rewrite before speed testing (2026-10-14)

This supersedes all preceding native-text next-step/performance experiment notes.
The user deleted the previous goal and explicitly requires the complete connected
Strata Intel inference path in Rust **before any live testing for speed**. Read
[the completion gate](kuno-strata-rewrite-completion.md) and the top of
[the port checkpoint](kuno-strata-rust-port.md). Intermediate checks may establish
build/shader validity, numerical/state correctness and successful execution; speed
reports must not change direction or justify dropping coupled components.

New goal is active. Working branch `codex/strata-rust-port`, starting runtime
`7455b6f`, isolated `kuno-image-speed` worktree. No canonical/deployed source,
API/protocol/UI or release changes; required Vashti changes **none**. Completion
is not claimed: global expert allocation, full GR/native schedules, prompt prefill
and general request connection remain among the explicit audit gaps. Preserve
existing source, public comparison evidence, Q2 sidecar and attribution.

First connected block: `d58bce4` selects native separate gate/up/DOWN 4/2/1
chunks, activation and scratch reuse together, with all extra INFO charged. The
protected runner requires execution-only validation and blocks speed/profile
trials. Release build, strict core Clippy, 158 WGSL/SPIR-V variants (105 physical
checks) and original regression 1,346/1 ignored pass. **No live GPU inference or
speed test, complete rewrite, numerical model acceptance or production claim.**
The next implementation continues the connected upstream dependency checklist;
no per-component speed selection. Logs live under `kuno-development` with names
`strata-connected-expert-{build,final-clippy,shaders,original-regression}-20261014.log`.

Connected GR block: `56663a3` ports pending-write normalization, resident BF16
sliced down/injection, fixed reduction and up/mix/commit together. Explicit writes
protect PLE, compact final rows and no-head prompt chunks. The reused private UP
owner grows by 524,288 bytes, charged in both family and complete-model quotes.
See the completion guide's **Connected GR implementation block** for backend
substitutions, source paths and exact logs. Strict core Clippy/release build,
163 shader variants (105 physical checks), 12 core GR owner/layout tests and
original direct-device regression 1,346/1 ignored pass. Full core unit suite is
**not green**: 1,148 pass, 200 fail, one ignored; old layout/budget assumptions and
other failures still need scope/contract triage. No live GPU inference, speed
trial, model numerical acceptance, installation or consumer change is claimed.
Goal remains active and the full rewrite incomplete; continue the dependency
checklist, preserving the connected port instead of selecting an intermediate
speed winner. Required Vashti changes: none.

Global allocation/loading source: `65e6e19` implements pinned Strata's native-pack
ranked byte-prefix plan from actual original graph metadata and an owning bounded
persistent GPU loader under existing admission. The same expert-byte budget gives
11,570 public-profile pairs and 160..342 residents/layer. Host-only prefix/source
coverage tests and strict core Clippy pass. **The GPU loader was not executed and
inference consumers still use the old uniform banks.** Read the completion guide's
**Global byte-prefix and owning-loader block** before continuing: replace banks,
HOST imports, per-expert BDA ownership, routing/grouping and exchange together;
do not allocate both caches or force the new layout into fixed old extents.
No GPU speed/profile, API/UI, installed binary or service change. Required Vashti
changes: none. Goal active, full rewrite incomplete, earlier core-suite failures
still open. The refreshed bounded helper's path/process audit is in that handoff.


### Active Kuno engine handoff extension — global source owner (2026-10-14)

Source commit `47d330f` on `codex/strata-rust-port` in the existing isolated
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti` checkout. Read the completion
guide's **Global HOST, source owner and native read dependencies** block before
continuing. It records exact construction/ownership/closure, validation and the
next connected consumer replacement.

The new global Sources owner combines one ranked GPU cache, only its variable
original HOST complement, actual per-expert physical addresses and native read
hazard/recording dependencies. Exclusive executor admission and legacy-cache
exclusion are implemented. Core build, strict Clippy, all 163 shader validations,
three layout tests and ordinary regression (1,349 passed, one ignored) pass.
**The inference coordinator does not call the new constructor yet, and no GPU
load/execute/recovery has been run for this source owner.** Full rewrite, numerical
state acceptance and the earlier experimental core suite failures remain open.
Next connect construction before prefill, routing/HOST fetch/rebase, separate-role
expert execution and private adaptive remap/publication as one coupled block.
No intermediate speed selection or disabling port components.

API specification/installed binary/services/Vashti behavior are unchanged;
required Vashti changes: none. No GPU testing, reset, rebind, installation or
release in this block. Canonical checkout and inherited untracked bytecode are
preserved. The goal is active and incomplete. Existing bounded sudo refresh remains
an authentication-only helper; it is not authority to bypass unavailable credentials.


### Active Kuno engine handoff extension — connected global core (2026-10-14)

Source `d24c348`, same isolated `codex/strata-rust-port` checkout. Read the completion
guide's **Connected global core dispatch block**, which supersedes the preceding
uncalled-constructor handoff without claiming runtime acceptance.

The core now initializes the global cache/HOST source owner before first prefill
and selects a new48-layer driver with GPU routing, private remap/group planning,
HOST fetch/rebase/stamps, native gate/up/activation/DOWN, weighted reduction and
real completion stamps. Each role uses one dispatch with pinned row_entries4/2/1;
IQ4_XS takes the scalar path, and native integer finish/XOR/IQ4_NL scheduling are
ported. Exact family/stage/table quotes are updated. Core release, strict Clippy
and179 shader validations pass. Final ordinary regression passes:1,349 tests,
one ignored. The earlier200 full-core failures are not resolved by this.

No GPU inference, physical closure/recovery, speed trial, installation or service
change. The old language/speculative controller still has frozen-bank receipt and
activation assumptions that can reject after prefill. Next port adaptive exchange
and private remap publication, then replace those controller assumptions with
actual global contracts. Complete1..8 target/MTP/prefill/state support and the full
checklist remain required. Audit original-GGUF versus upstream native conversions
and actual Intel CPU/transfer/overlap mode; do not claim complete parity yet.

No API or Vashti consumer changes. Goal active; full rewrite incomplete; speed
remains blocked. Preserve canonical work and inherited untracked bytecode.


### Active Kuno engine handoff extension — adaptive global cells (2026-10-14)

Source `e1dc698` on the same isolated `codex/strata-rust-port` checkout. Read the
completion guide's **Connected global adaptive exchange** block for exact policy,
publication/ownership, reservations, validation and remaining substitutions.

The global core now connects native every4/global96/gain1.5/decay0.7 adaptation,
unequal physical GPU/HOST cells, paired original-role exchange and private
logical remap/usage publication after actual joined boundaries. Static address
tables remain unchanged; evictions are retained in HOST without file reads.
Core build/Clippy and179 shaders pass. Three pure policy/exchange simulations and
ordinary regression (1,352 passed, one ignored) pass. No GPU/Vulkan exchange,
whole-model numerical/state/recovery or speed acceptance was run.

Next port global language/speculative controllers and real receipt/activation
contracts; reconcile prompt versus verify adaptation windows, complete1..8
prefill/target/MTP/state paths and the remaining fidelity audit. Current in-place
blocking Vulkan exchange is a documented backend substitution, not async-stream
parity. Earlier full-core failures remain open. No API/Vashti/install/service
change; required Vashti changes: none. Goal active, full rewrite incomplete,
speed gate still blocked. Preserve canonical work and inherited bytecode.


### Active Kuno engine handoff extension — global generation controller (2026-10-14)

Source `9abc0fd`, same isolated `codex/strata-rust-port` checkout. Read the
completion guide's **Connected global request and verification controller**
section; it supersedes the stale statement that core language execution still
requires old frozen-bank receipts.

Core now enters its own connected global-source controller: teacher prefill,
real MTP/suffix proposals, target verification, actual accepted-prefix state
repair/commit, MTP carry repair, cursor checks and transient accepted-token
callback. Real source/admission/graph authority replaces frozen-bank identities;
shared GDN/PLE/status/head selection uses that authority. Private usage excludes
prompt/replay routes and adaptation ticks once per completed verification window.
Internal bounded request geometry is charged and authenticated; the current
CLI still selects public fixtures, and general sampling/input remain open.

Release build, strict Clippy and179 offline shaders pass.14 selected core tests
and a bounded-geometry lifecycle test pass. Ordinary regression:1,354 pass,
one ignored. No live model/GPU exchange/recovery/speed run. Old full-core
failures, native CPU-share/queue/packing audits, coupled1..8 target/MTP/state
support and the remaining completion checklist remain required. Default core
capacity is still four; do not claim full rewrite or runtime acceptance.

API specification/installed services/Vashti are unchanged; required Vashti
changes: none. No install, release, reset or rebind. Canonical checkout and
inherited bytecode preserved. Goal active and incomplete; speed gate closed.


### Active Kuno engine handoff extension — eight-row native consumers (2026-10-14)

Source `ac6169a`, isolated `codex/strata-rust-port` checkout. Read the completion
guide's **Connected eight-row consumers and native Q6 head** section. It
supersedes the default-four structural limitation, not the missing GPU or full
rewrite acceptance.

Core now couples capacity8 to packets/logits/quotes/checkpoints/MTP, real1..8
dense Q8 consumers and GDN streaming. Original Q6 native iqs integer-chain/XOR
head is connected and selected before prompt input. Core Q2 MTP repair handles
all accepted prefixes of eight-row windows and uses full draft vocabulary.
Global expert/cache/controller path stays selected. No decoded weight duplicate,
legacy fabricated receipts or intermediate speed selection.

Release build and strict Clippy pass;195 offline shaders pass. Host tests cover
independent canonical Q6 lane mapping, all RHS scratch bounds,1..8 repair prefixes,
compact packet/inactive rows and target rejection/EOS. Ordinary regression:
1,358 pass, one ignored. No live GPU/model/physical recovery/speed test. Prior
full-core failures and full eight-row model/state acceptance remain open.

Next port/audit CPU expert share, transfer/shared execution and queues, native
packing, GDN/QSA/KV/PLE/prefill and MTP graph/state/sampling/general-input paths.
Do not claim full rewrite or open the speed gate yet. API/Vashti consumer changes:
none. Installed services/canonical checkout unchanged; inherited bytecode
preserved. Goal active and incomplete.


### Active Kuno engine handoff extension — native CPU/GPU split (2026-10-14)

Source `e43e271`, isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected native CPU expert share and shared-work overlap** for actual
source/math/ownership/join contracts and explicitly unresolved substitutions.

Real router groups now split resident-first GPU, last141/256 missing GPU staging,
and remaining HOST-backed CPU jobs. Rust native Q8/integer arithmetic covers all
six routed formats; eight RHS share code loads and persistent row workers.
Authentic HOST readonly maps stay alive through CPU joins. GPU expert/shared work
is submitted while CPU runs; CPU merge/reduction precedes shared finish/GR combine.
Actual source mutation/close requires unique Arc ownership. Paired exchange now
includes explicit Vulkan GPU-to-HOST visibility. Added1 MiB private MAIN and16 MiB
CPU host reservation; no disk source reads, weight duplicates or persisted content.

Release build, strict engine/library Clippy,196 offline shaders, native CPU math /
SIMD / sparse-row tests and exhaustive width/split rank accounting pass. Ordinary
regression:1,359 pass/one ignored. Library:141 pass/two ignored. No GPU model,
actual HOST mapping/exchange/overlap/recovery or performance acceptance. Existing
full-core failures remain open. Per-layer readback/upload, fixed share and Rayon
unPinned topology differ from native mapped sink/physical-core/stream scheduling;
remaining fidelity/math/prefill/KV/MTP/sampling/input/recording checklist required.

Keep assembled core selected; speed gate closed. API/Vashti changes required:
none. Installed services/canonical checkout unchanged; inherited bytecode
preserved. Goal active and incomplete.


### Active Kuno engine handoff extension — connected FP16 QSA (2026-10-14)

Source `d5d4e49`, isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected FP16 QSA family** for ownership, actual math/state dispatches,
allocation changes and outstanding substitutions. This supersedes the core F32
KV limitation; it does not establish complete engine or device acceptance.

Core width1..8 now couples actual packed FP16 committed/current KV, native fused
norm/RoPE, FP16-rounded raw index keys, retained completed pooled blocks, native
Intel index scores,64-cell attention/merge/gate and rejected-tail clearing.
Retained POOLED identity joins checkpoint/physical-root proofs. Actual private
stage/scratch allocations and quotes change together. Existing assembled global
sources, CPU/GPU share and controllers remain selected; no partial speed winner.

Release build, strict Clippy,206 offline shaders,26 family and two host geometry/
chunk-oracle tests pass. Ordinary regression:1,361 pass/one ignored. Wider core
QSA filter:58 pass/two fail (language/cache gate and whole-model quote); older
full-core failures remain open. Token-major/raw-history/dead-tail/multi-query/
queue substitutions and numerical/state/recovery acceptance are still required.
No GPU/model or speed execution. Goal active and full rewrite incomplete; no live
speed tests until the connected completeness and correctness gates pass.

Next port GDN's actual multirow conv/L2, four-group FMA recurrence/norm and
accepted-prefix replay together (reference `verify_kernels.dp.cpp`), then continue
the whole checklist. API/installed services/Vashti changes required: none.
Canonical checkout and inherited bytecode preserved; no install/release/reset.


### Active Kuno engine handoff extension — coupled native GDN (2026-10-14)

Source `351e29b`, same isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected native GDN producers, recurrence and repair**; it supersedes
the previous extension's description of the old GDN streamer arithmetic.

Core width1..8 couples RAW conv/native L2, controls,512-lane four-group FMA state
recurrence, query scaling after readout, fused output norm/gate and actual output
projection. Accepted-prefix state-only replay now uses the same arithmetic and
retained native inputs. Existing state/MAIN/INFO reservations stay connected;
actual device workgroup capacity is checked before launch. Global-source and
native QSA families remain selected together.

Release build, strict Clippy,210 offline shaders,17 family and one independent
host math/prefix test pass. Ordinary regression:1,362 pass/one ignored. No GPU,
model/state/recovery or speed test. Head-major/separate candidate state, shared
plane scheduling, BF16 AB projection, optional producer fusion, prompt/capture
and numerical acceptance remain open. Prior full-core failures remain required.

Full rewrite remains incomplete; goal active, speed gate closed. Continue the
whole completion checklist (AB/packing, prompt/PLE/MTP initialization, complete
recording/sampling/arbitrary input), then reconcile and perform final acceptance.
API/installed services/Vashti changes required: none. Canonical checkout and
inherited bytecode preserved; no install/release/reset/rebind.


### Active Kuno engine handoff extension — native GDN AB/window preload (2026-10-14)

Source `3ae92cf`, same isolated `codex/strata-rust-port` checkout. Completion
guide **Native GDN AB and next-token preload** supersedes the generic AB and
single-plane limitations in the previous GDN receipt.

Core now borrows real original resident BF16 AB matrices and runs pinned
32-lane/eight-value FMA window projection plus gating. Real half-result stamps
join before control status acceptance. No generic alpha/beta fallback. QKV/Z,
controls, native recurrence/norm and prefix replay stay connected. Recurrence
and replay now preload next-token Q/K in alternate shared planes. No new MAIN
owner or model duplicate; unused old scalar scratch stays visibly reserved.

Release build, strict Clippy,211 offline shaders and two independent host dot/
state-prefix tests pass. Ordinary regression:1,363 pass/one ignored. No GPU/model
or speed execution. Earlier full-core failures and complete numerical/state/
queue/prompt/packing/MTP/sampling/input acceptance remain required.

Next actual dependency is grouped MTP front/prompt initialization; read the
completion receipt's pinned call sequence and teacher alignment. Current scalar
teacher/repair loops are not native grouped graphs. Keep the whole assembled
core selected, goal active and speed gate closed. API/Vashti changes: none.
Canonical checkout/services and inherited bytecode preserved; no installation.


### Active Kuno engine handoff extension — grouped MTP front (2026-10-14)

Source `147cd9a`, same isolated `codex/strata-rust-port`. Read completion guide
**Connected native MAX8 MTP front** for real owners, loading formats, caller/state
alignment and explicit remaining substitutions.

Core now connects grouped embedding/wide-carry norm, native Q8 input branches,
BF16 fused GR, shared K/V input planes and key norm/RoPE/append. Prompt teaching
and accepted-prefix repair use real grouped windows. Draft/followup count1 uses
the same front from actual GPU Token/Carry owners, with no duplicate KV append.
Original BF16 GR and Q6 draft head are preserved. New20-owner/4,685,824-byte
front,4096 core context,25,296,896-byte private plan and erasure/close accounting
are coupled; ordinary2048 path remains separate.

Release build, strict Clippy,216 offline shaders, two geometry/canonical-embedding
and nine repair tests pass. Ordinary regression:1,365 pass/one ignored. No live
GPU/model/state/recovery or speed run; earlier full-core failures remain open.
F32 cache backing, legacy MTP rest, prompt MMQ/group fences, recorded graphs,
carry handoff and numerical/device acceptance are still required. Next port MTP
rest and packed cache together, then the remaining whole completion checklist.

Goal active; full rewrite incomplete and speed gate closed. API/Vashti changes
required: none. Canonical/services and inherited bytecode preserved; no release,
install, reset or rebind.

### Active Kuno engine handoff — connected packed MTP rest (2026-10-14)

Source `294cb93`, isolated `codex/strata-rust-port`. Read completion guide
**Connected MTP packed cache and attention/GR rest**. Core append/attention/
rewind/guards now agree on native page4/head-major FP16 K/V; count1 attention
and original BF16 FF/head GR consume the real front and publish actual carry.
Private plan18,644,992 bytes includes all22 front/rest owners and cleanup.

Release build, strict core Clippy,221 offline shaders, three layout/front and
nine repair/ownership tests pass. Ordinary regression:1,366 pass/one ignored.
No GPU/model execution, speed testing, deployment, API or Vashti changes.
Whole-port completion and physical/numerical acceptance are not claimed.

Next port MTP scaled-activation/native integer Q2 experts with router/shared
consumers as one coherent block; then complete remaining prompt/PLE, graph/
queues and arbitrary-input/sampling checklist. Earlier full-core failures remain
open. Do not select partial speed winners or reopen live speed benchmarking.
Goal active; full rewrite incomplete. Required Vashti changes: none.

### Active Kuno engine handoff — connected MTP native MoE (2026-10-14)

Source `a4aa0dc`, isolated `codex/strata-rust-port`. Read completion guide
**Connected native MTP routed/shared MoE** for actual original source roles,
activation/data layout substitutions, validation and remaining queue gaps.

Core now connects BF16 router, native top10, F32-scaled S2 input and SwiGLU
quantization, paired integer gate/up/down, fused shared Q8, BF16 scalar and
native merge to the real GR/head/carry graph. All original512 Q2 experts stay
resident; no per-request weight upload or host route read. Native dense input
quantization now follows half-away/finite-half scale/sum contract and supports
the legal640-column shared projection. Private plan18,776,064 bytes includes
all25 front/rest owners, guard/erasure/close accounting.

Final release build, strict Clippy and229 offline shaders pass. Independent
GGUF/S2 host math,640-column batch bounds and nine repair/owner tests pass.
Ordinary pre-fusion regression:1,367 pass/one ignored; subsequent core-only
shader/binding changes pass final core build/shader checks. No GPU/model/state/
recovery or speed acceptance, root/service/API/UI change or installation.

Next implement actual Intel GEMM/streamed prompt prefill plus PLE/history,
then finish graph/queues and arbitrary input/sampling. Pinned SYCL CMake builds
GEMM/prefill with fused stubs and no CUDA/HIP MMQ; port its real Intel consumers,
not the inactive MMQ stub. MTP side-queue/capture scheduling, dense-format audit,
earlier full-core failures and whole-model acceptance remain open. Full rewrite
incomplete; goal active; speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — native PLE/source history (2026-10-14)

Source `1018a09`, isolated `codex/strata-rust-port`. Read completion guide
**Connected native PLE and target BF16/router math** for exact call paths,
allocation/ownership, I/O/history substitutions and remaining schedule gaps.

Core PLE now connects original BF16 MAX8 projections and native1024/512/256-lane
normalization/gate/conv/history to actual accepted-prefix owners. Target router/
shared-scalar BF16 uses the same native projection. Global router now matches
full512-expert softmax/winning-lane normalization. Original private owners,
remap/group/usage and whole-model commit/fault proofs remain authoritative.
Selected-row I/O is deduplicated, parallel and transient, using cloned pinned
descriptors; the additive tensor-reader Rust API has no Vashti consumer change.

Final core release build, strict Clippy and235 offline shaders pass. Three
independent host math/prefix tests,12 ownership/source tests and actual parallel
descriptor test pass. Ordinary pre-router-extension regression:1,370 pass/one
ignored; later core-only changes pass final core/focused checks. No GPU/model,
speed, root/service/API/UI change, deployment or release. Full rewrite incomplete.

Next implement real Intel prompt GEMM/chunk/staging and PLE next-chunk prefetch,
not larger repeated verification windows. Native history mapped snapshots,
stable full graph/queues, dense-format audit, arbitrary input/sampling, earlier
full-core failures and whole-engine physical acceptance remain open. Goal active;
speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — connected prompt GEMM arithmetic (2026-10-14)

Source `e64d770`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt operand/GEMM producers and consumers**. Initial prompt
calls now select rounded BF16/FP16 operands, original GPU weight images and
Rust/Vulkan tile GEMM; normal/pair/bundle/head, GR and AB consumers use it.
Decode remains separate. Existing three64 MiB stage owners are borrowed and
serialized; no new GPU content root or persistent converted artifact.

Final release build, strict core Clippy and247 offline shaders pass. Two host
canonical weight-image/rounded-product tests pass; later span/saturation/dummy
owner changes pass final build/shader checks. No live GPU/model/speed test,
root/service/API/UI change, installation or release.

**Still not complete prompt prefill:** the controller remains a three-row loop.
Large native chunk allocation/budget, streamed FP16 expert GEMM/ring/cache loan,
prompt causal kernels, double-buffered PLE prefetch, device carry/teacher handoff
and whole-prompt scheduling remain required. oneMKL is replaced with ordinary
FMA tile GEMM; numerical/XMX/performance acceptance is unproven. Final stage/
cache/queue lifetime audit and earlier full-core failures remain open. Full
rewrite incomplete; goal active; speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — original-source FP16 prompt experts (2026-10-14)

Source `1cf7e50`, isolated `codex/strata-rust-port`. Read completion guide
**Connected original-source prompt expert arithmetic**. Initial prompt calls
now connect logical-ID grouping, real source fetch, original six-format FP16
weight images, FP16 gathered activations, tile GEMM, saturated SwiGLU and parent
rank reduction. Prompt bypasses decode's CPU split; decode stays separate.

The actual global stage owners are three128 MiB buffers. Earlier prompt notes
and scope guards incorrectly expected64 MiB; both scope guards now match the
real owner. No new private weight root, model duplicate or persistent artifact.
Serial role-major staging, separate GU layout, portable FMA tile GEMM and INFO
admission remain explicit schedule/storage audit work for the full chunk owner.

Final release build, strict Clippy and256 offline shaders (including original
six-format physical BDA checks) pass. No live model/GPU/numerical/state/recovery
or speed execution; no root/service/API/UI change, installation or release.

Next connect the native **large chunk owner and expert-major staging ring/cache
loan**, prompt causal kernels, PLE prefetch and device target/MTP teacher handoff
together. Do not grow/rename the current three-row verification loop and call it
native prefill. Complete graph/queues, dense-format audit, arbitrary input/
sampling, earlier full-core failures and whole-engine acceptance remain open.
Full rewrite incomplete; goal active; speed gate closed. No intermediate timing
may steer or roll back the incomplete assembled port. Required Vashti changes:
none. Canonical checkout/services and inherited bytecode preserved.

### Active Kuno engine handoff — native prompt ring and paired products (2026-10-14)

Source `9339904`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt staging ring and paired expert order** for source mapping,
real owners, copy-contract boundaries and remaining substitutions.

Prompt experts now run in logical-ID expert order through actual eight-slot raw
and two-slot dequantization rings. Paired interleaved GU, FP16 SwiGLU and original
DOWN consumers are connected together; earlier completed rank outputs cannot
be overwritten by later GU rows. No new private roots or model duplicate.
One bounded routing-only readback matches native host grouping; no CPU activation
readback or CPU expert contribution. Decode's integer/CPU path stays separate.

Original HOST buffer imports allow COPY_SRC. New internal CubeCL copy APIs
validate registered readonly sources, full same-stream destination owners and
physical cell extents, then insert actual transfer nodes into the same owned
native graph. Do not bypass that API with an external raw queue: registry
detach must join every consumer. Independent copy/compute queues remain open.

Final release build, strict operational core Clippy,256 offline shaders and one
host-only copy metadata contract test pass. Test metadata is not physical GPU
or copy execution proof. No live GPU/model/numerical/state/recovery/speed test,
root action, service/HTTP API/UI change, installation or release. Earlier full
core failures remain open; canonical checkout/services and bytecode preserved.

Next: connect the full **large prompt chunk owner/carving/budget/cache loan** to
layer-major GR, causal prompt GDN/QSA, larger real routed-row owners, PLE prefetch
and device target/MTP teaching. The current caller still uses three-row windows;
do not label that as native prefill. Queue overlap, complete capture/admission,
oneMKL/XMX substitution, dense formats, arbitrary input/sampling and whole-engine
acceptance remain required. Full rewrite incomplete; goal active; speed gate
closed. Keep the assembled port selected; no timing-driven partial rollback.
Required Vashti changes: none.

### Active Kuno engine handoff — prompt GDN and direct FP16 output (2026-10-14)

Source `810f89e`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt GDN causal pipeline and FP16 consumer** for pinned Intel
selectors, real operators/owners and remaining layout differences.

Actual prompt calls now connect contiguous H, causal conv/history/L2, pipelined
CB32 recurrence and separate FP16 output norm to the real output GEMM. Native
FP16 is consumed directly; no normalized F32 GATED copy or repacking. Existing
private owners/status/commit/repair/erasure remain authoritative. Decode stays
separate. Internal prepacked projection is checked against the actual admitted
stage identity/stream and geometry; no HTTP or Vashti API change.

Final release build, strict Clippy and262 offline shaders pass. No GPU/model,
numerical/state/recovery or speed execution, root action, installation or release.
Existing full-core failures remain open. State is still head-major and split
replay inputs are retained; do not credit native placement parity. The enclosing
caller still uses three-row/MAX8 windows, despite dynamic prompt kernel counts.

Next: connect prompt QSA and the full large chunk owner/carving/budget/cache loan,
larger routed-row ownership, PLE prefetch and device target/MTP teaching together.
Complete queues/capture/admission, state/dense-layout audit, arbitrary input/
sampling and whole-engine acceptance remain required. Full rewrite incomplete;
goal active; speed gate closed. No timing-driven partial rollback. Required
Vashti changes: none. Canonical/services and inherited bytecode preserved.

### Active Kuno engine handoff — prompt QSA and joined PLE prefetch (2026-10-14)

Source `242ce1c`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt QSA consumers and PLE source prefetch**. Prompt QSA selects
separate RMS/RoPE and directly consumed FP16 gated output. PLE's bounded scoped
producer supplies upcoming source rows, authenticated against real expected
rows; receiver disconnect/join and wiping remain request-local. Extra327,680
host bytes is charged at both component and model boundaries. No new GPU root.

Final release build, strict core Clippy and266 offline shaders pass. No live
GPU/model/source-worker, numerical/state/recovery or speed run; earlier full-core
failures remain open. Goal active, full rewrite incomplete, speed gate closed.
Installed services, canonical checkout, GPU residency and inherited bytecode
were untouched. Required Vashti changes: none.

Next implement the **large prompt owner/carving/budget and full layer-major
chunk schedule**, including larger expert route/member owners, device MTP
teaching and final-only head. Current controller/source prefetch still uses
three-row/MAX8 ownership. Pinned Intel XMX prompt attention is opt-in, not its
default. With this model's large HOST complement, native staging defaults to
eight slots; do not port inactive CUDA branches to satisfy a mistaken checklist.
Queue/capture, storage/state, arbitrary input/sampling and whole-engine
acceptance remain required before speed trials. Sudo refresh in shared shell
succeeded during source work; no protected inference was started.

### Active Kuno engine handoff — chunk-capable routed prompt owners (2026-10-14)

Source `0f89b45`, isolated `codex/strata-rust-port`. Read completion guide
**Connected chunk-capable prompt expert owners**. Prompt calls now use their
own1..256-row router, compressed logical-order host groups, tiled FP16 expert
products and carved rank output reduction. GU `[64,66)` MiB and PART `[66,98)`
MiB borrow the real third128 MiB stage; original transfers stay below64 MiB.
Decode remains MAX8. No additional GPU root;256 KiB host/8 MiB I/O is charged.

Final release build, strict Clippy and268 offline shaders pass. No live GPU,
physical view/copy, numerical/state/recovery or speed acceptance. Earlier full-core
failures remain open. Canonical/services/GPU residency untouched; no API or Vashti
change. Inherited bytecode preserved. Full rewrite incomplete; goal active.

Next connect the full prompt residual/GR/attention/PLE/carry owners and actual
layer-major chunk controller, including final-only head and device MTP teaching.
The current parent still uses three-row/MAX8 packets and receipts. Projection
stage working intervals/alias checks must be changed with the new owner, not
silently overlaid. Cache-loan budget, mapped grouping, queues/capture, layouts,
sampling/general input and whole-engine acceptance remain required. No speed
trials or timing-driven partial rollback. Shared shell now reports sudo expired;
continue source work, request authentication before protected GPU tests.

### Active Kuno engine handoff — carved prompt GR ownership (2026-10-14)

Source `649e923`, isolated `codex/strata-rust-port`. Read completion guide
**Connected carved GR workspace**. Actual prompt GR uses distinct carved
upper-stage0 intermediates with256-row capacity; stage0/1 working images occupy
lower64 MiB. Exact interval/prepacked-view checks preserve ownership. No new
GPU root;4 KiB metadata charged. Parent residual/injection/mixed/state receipts
still use small windows. Full large-chunk ownership/controller remains next.

Release build, final strict Clippy and268 offline shaders pass; see the receipt
for the equivalent post-build Clippy guard cleanup. No GPU/model/physical/state/
recovery/speed acceptance or service/API/UI/install change. Earlier full-core
failures remain open. Canonical checkout and inherited bytecode preserved.
Full rewrite incomplete; goal active; sudo expired. Continue full residual/
attention/PLE/carry/state composition before speed trials. Vashti changes: none.

### Active Kuno engine handoff — carved prompt GDN and real replay source (2026-10-14)

Source `81f1d76`, isolated `codex/strata-rust-port`. Read completion guide
**Connected carved GDN prompt data and replay source**. Actual prompt QKV/Z/H/
readout use stage1 upper-half carved views. Original replay snapshots the real
raw producer, including zero inactive rows. New internal projection writes are
restricted to exact registered views. No extra GPU root;73,728 runtime metadata
bytes charged. Status/control/replay/acceptance and parent packets remain MAX8.

Final release build, strict Clippy and269 offline shaders pass. No live GPU,
copy/model/numerical/state/recovery/speed validation. Full-core failures remain
open; full rewrite incomplete, goal active. Canonical/services/bytecode untouched.
No API, install or Vashti change. Last known sudo expired; source work can continue.

Next QSA: shared pooled index state and bounded32-query attention scratch, with
causal selection and exact cache append/state ownership. Read the receipt's
native pointers; do not enlarge old per-query pool/attention arrays. Full
residual/PLE/carry, prompt acceptance, head/MTP teaching, budgets/queues/layouts
and whole-engine acceptance still precede speed testing.

### Active Kuno engine handoff — shared QSA and native GPU selection (2026-10-14)

Source `6314ea4`, isolated `codex/strata-rust-port`. Read completion guide
**Connected shared prompt QSA and native GPU selection**. Actual prompt QSA now
uses carved projections/Q/K/IQ/candidates, shared completed-block index state
and32-query attention scratch with explicit full-chunk row offsets. Append
precedes scoring/attention. Core selection now uses the pinned GPU weighted radix
path, normalized cell-zero dead key, tail bias and exact `min(visible,2051)` width,
including partial blocks. The model no longer CPU-sorts/uploads QSA selections.
The old ordinary selector is preserved, not an oracle for the native width.

Release build, strict Clippy and281 offline shader variants pass; no physical
GPU/model/numerical/state/recovery/speed run. Real larger POOLED owners,16KiB
scope metadata and417,792 extra INFO bytes are charged;1024 lanes/32,912 shared
bytes are required. Existing full-core failures remain open. Full rewrite is
incomplete; goal active; speed gate closed. No API/Vashti changes are required.
Canonical/services/GPU residency/inherited bytecode are untouched. Last sudo
check reported a password required; no root workaround or protected run.

Next: full residual/mixed/block/carry/PLE owners and the real large layer-major
prompt controller, final-only head, device MTP teaching and prompt state acceptance.
The current enclosing parent/receipts still use three-row/MAX8 windows. Complete
native budgets/cache loans, queue/capture and layout/substitution audits, general
input/sampling and whole-engine acceptance before speed trials. Do not revert
coupled components because of incidental timing or claim the rewrite complete.

### Active Kuno engine handoff — full-chunk GDN and native state layout (2026-10-14)

Source `9a7a23b`, isolated `codex/strata-rust-port`. Read completion guide
**Connected full-chunk GDN math and native recurrent layout**. Actual prompt
math now uses256-capable convolution/L2/recurrence/norm entries with direct model
faults, independent of per-element status/replay buffers. The enclosing MAX8
caller retains explicit split/replay/closing compatibility. AB raw projections
use a real carved region and a handle/count entry. New controls/status capacity
fits the same GDN root. Native recurrent row/value-head layout is selected
coherently in prompt, decode and state-only repair; no transpose or new owner.

Release build, strict Clippy and286 offline shader variants pass. No live
GPU/model/numerical/state/recovery/speed acceptance; earlier full-core failures
remain open. Runtime quotes include147,456 extra INFO bytes. No API or Vashti
change, installation or release. Canonical/services/GPU residency/bytecode were
preserved. Goal active, full rewrite incomplete, speed gate closed; last sudo
check required a password, and no protected operation was attempted.

Next connect the full residual/mixed/block/carry/PLE prompt controller and its
actual state acceptance. Finish compact/direct-fault QSA status ownership rather
than scaling its old50MiB-per256-query attention status. Read the receipt's exact
existing handle entry points. Parent still uses three-row/MAX8 windows; final-only
head, device MTP teaching, native budgets/cache loans, queues/capture, remaining
layout audits, general input/sampling and whole-engine acceptance remain required.

### Active Kuno engine handoff — full prompt QSA ownership (2026-10-14)

Source `101f4e7`, isolated `codex/strata-rust-port`. Read completion guide
**Connected full prompt QSA ownership**. Actual prompt QSA now calls one connected
256-capable family using carved compact flags/scores/selections and direct model
faults. It includes positions through packed gated output and32-query attention
scratch; no per-value attention flags or per-query pooled copies. Legacy MAX8
status adapters remain outside it for the existing parent; later old math and
selector events are skipped. Scope metadata32KiB and conservative QSA INFO835,584
bytes are charged. Twenty QSA views fit59,490,304 bytes of the existing upper64MiB.

Release build, strict Clippy and296 offline shader variants pass. No live GPU,
model/numerical/state/recovery/speed acceptance; full-core failures remain open.
Full rewrite incomplete, goal active, speed gate closed. No API/Vashti, service,
install or release change. Canonical/GPU residency/bytecode preserved; last sudo
check expired, no protected run or workaround.

Next implement the real large controller with persistent R/mixed/block/carry,
handle-based routed/shared/PLE calls, charged larger joined PLE source, prompt
state acceptance, final-only head and device MTP teaching as a coupled block.
GR/AB/GDN/QSA/expert math already has256-row handle entry points; the parent
still uses three-row/MAX8 packets. Keep verifier receipts separate and actual
binding/root checks intact. Complete native budgets/loans, queues/capture,
remaining layout/substitution audits, general input/sampling and whole-engine
acceptance before throughput comparisons. Required Vashti changes: none.

### Active Kuno engine handoff — large prompt controller and full device MTP (2026-10-14)

Source `f3196fc`, isolated `codex/strata-rust-port`. Read completion guide
**Connected large prompt controller and device MTP teaching**, including its
validation and exact next-source pointers. The core parent now runs real256-row
prompt chunks across embedding/GR/GDN/QSA/routed/shared/PLE, with persistent
carved R/mixed/blocks, all-family state preflight/commit, a last-chunk-only head
and full-chunk MTP teaching directly from target VRAM. The interim eight-row
teacher and per-row host carries are gone. Verifier MAX_BATCH stays8.

Persistent GR/outer views use57,918,720 bytes of the existing stage0 upper64MiB;
PLE phase views use57,695,232 bytes of stage1 upper64MiB. No new GPU root/MAIN
allocation. PLE source host charge is12MiB, controller host/runtime add8MiB each.
Exact PLE history acceptance maps only its last nine normalized rows and NEXT;
legacy decode-policy/checkpoint workspaces receive a sealed real prompt boundary.
Prompt GR writes consume activated injection once, correcting its former double
activation at explicit PLE/final boundaries.

Release build, strict Clippy,308 offline shader variants and six core PLE history
tests pass. The complete core unit suite **does not pass**:1,149 passed,228 failed,
1 ignored. The receipt distinguishes the historical200-failure baseline from
this newer suite; do not assume all failures are obsolete or from this commit.
No live GPU/model/numerical/state/recovery/speed run or API/Vashti/service/install/
release change. Canonical checkout and bytecode preserved; goal active, full
rewrite incomplete, speed gate closed. Last Demons shell `sudo -n -v` required
a password; no protected operation or privilege workaround followed.

Next complete native queue/recording/copy dependencies and the real GEMM/data-flow
substitution, then remaining placement/default sizing, arbitrary input/sampling,
verifier repair and unit-failure audits. Exact useful pointers are in the receipt:
Rust Vulkan backend already recognizes FP16/BF16 cooperative matrices, while the
currently used WGSL compiler rejects CoopMma; its optional SPIR-V path is worth
investigating. Pinned Intel source defaults BF16X2 off and uses owned prompt
buffers at context<=32768 (default2048 rows), so audit those actual branches
before declaring remainder GEMMs or cache lending unconditional missing work.
Finish the full source review and protected whole-engine acceptance before any
speed comparisons. Required Vashti changes: none.

Supplemental read-only inventory is recorded at the end of that receipt. The
B70 driver exposes cooperative-matrix and BF16 extensions and FP16/F32 matrix
shapes; no matrix computation was run. The existing device adopts one family0
queue; families1/2 exist but were not requested. These are concrete starting
points for the connected compiler/device/queue block, not performance evidence.


### Active Kuno engine handoff — connected cooperative matrix backend (2026-10-14)

Source `d326f23`, isolated `codex/strata-rust-port`. Read completion guide
**Connected Rust cooperative prompt matrices**, including the required queue
continuation and validation details. Actual dense prompt/MTP and grouped expert
GU/DOWN products now use Rust FP16/BF16 cooperative8x16x16/F32 kernels, coupled to
the existing rounded images, maps, faults, activation and downstream consumers.
Target/MTP device creation enables the actual matrix features and accurately
adopts the full extension list without enlarging admitted limits. Both Wgpu and
recorded native pipelines use the same explicit16-lane geometry; existing
physical integer shaders retain32. Native SPIR-V registration includes real
INFO/descriptors and cached operator identities. External model sources retain
their audited read-only WGSL lowering.

One shared HAL canonicalizer fixes redundant implicit Workgroup strides while
preserving explicit Block and host byte ABIs, rejecting aliases it cannot safely
rewrite. Release build, strict core/feature-off Clippy, five host ABI tests,
308 existing shader variants (122 physical-pointer checks) and32 cooperative
variants (all eight independent optional-layout masks) pass. Use the modern
validation-only Khronos tool revision/path in the receipt; the distro2025 tool
cannot validate BF16. No live model/GPU/numerical/state/recovery/throughput run;
the earlier full-core228 failures remain unaudited and the complete suite is not
green. No new global tensor root, API/Vashti change, install, service restart or
release. Canonical checkout and inherited bytecode preserved.

**Next connect the native copy/compute schedule as one block.** The device still
requests only family0 queue0, and both Wgpu roots and imported host copy sources
are EXCLUSIVE. Merely requesting a transfer queue would race or violate ownership.
The receipt names exact allocation/insertion/recording/flush sites and dependency,
replay and partial-submission lifetime requirements. Keep actual created queue
metadata, buffer sharing/ownership, semaphore edges and final dual-queue completion
joined. Do not claim real ring-slot overlap from conservative full-root hazards.
Then complete native default2048/owned-versus-loaned sizing and placement,
arbitrary input/supported sampling, verifier repair and the full source/failure
audits before protected whole-engine acceptance and only then speed comparisons.

Full rewrite incomplete, goal active, speed gate closed. Last shared-shell
`sudo -n -v` still required a password; no protected operation or workaround was
attempted. There is useful source work before root is needed. Required Vashti
changes: none.


### Active Kuno engine handoff — real compute/copy queues and joined replay (2026-10-14)

Source `ac05da8`, isolated `codex/strata-rust-port`, following matrix `d326f23`.
Read completion guide **Connected native compute/copy queue ownership**. The
Intel device now actually creates/adopts a second transfer-family queue before
allocation. Both Wgpu roots and actual HOST copy sources use concurrent sharing
for the two admitted families. Device-local metadata describes the queue really
created; ordinary feature-off construction retains one queue.

Actual native recordings now split shaders/copies/fills into queue-owned command
segments with RAW/WAR/WAW timeline edges, an upload-release header and a final
fence joining both queues. Cached replay uses fresh timeline epochs. Recording
owns both pools, semaphores and retained roots; partial submissions become
in-flight before the first submit, and unknown work cannot be reset/reused or
freed based only on an unsubmitted final fence. Six host planner/epoch tests,
release build, strict core/feature-off Clippy,308 existing shader variants
(122 physical validations) and32 matrix variants pass. Added host queue metadata
is bounded/charged (80MiB core lifetime reservation,64MiB retained queue-vector
cap). Opaque driver allocation and real hardware admission remain unproved.
No live GPU/model/numerical/state/recovery/speed test. Complete unit suite not
rerun; its earlier228-failure baseline remains unaudited. No API/Vashti, installed
service, model-residency, installation or release change. Canonical checkout and
inherited bytecode preserved.

**Next connect exact ring-slot descriptors and byte-range hazards together.**
Current hazards conservatively key whole buffers, so separate slots within the
same64MiB raw buffer still serialize. This is not complete native overlap or a
performance result. The receipt names the actual group-ahead staging loop,
Dequant whole-root binding, `p.r0` offset and shader's fixed raw length/extent
checks. Do not merely tell the scheduler a narrower range: narrow the real
binding and update shader/source bounds coherently, including unequal GU/DOWN
blob sizes. Add independent partial-overlap/disjoint-slot/reuse oracle coverage
and charge bounded interval metadata. Keep BDA/indirect dependencies, replay
signatures, upload acquisition and final dual-queue join intact.

Then finish whole graph/branch capture fidelity, native default2048/owned/loaned
sizing and placement, arbitrary input/sampling, verifier repair and all source/
unit completeness audits. Protected full-engine acceptance precedes speed tests.
Goal active, full rewrite incomplete, speed gate closed. Root was not needed for
this source/offline block; last prior shared-shell check had required a password.
Required Vashti changes: none.


### Active Kuno engine handoff — exact raw-slot ranges and early release (2026-10-14)

Source `3ca345c`, isolated `codex/strata-rust-port`, following queue `ac05da8` and
matrix `d326f23`. Read completion guide **Connected exact raw-slot ranges and
dequant release**. Actual graph descriptors/copy spans now feed a bounded
byte-interval RAW/WAR/WAW planner, preserving untouched intervals and allowing
disjoint slots within one buffer to overlap. Bounds include retained Vec
capacities after contraction; host queue reservation is now128MiB, cached queue
vectors still capped64MiB. Hardware/driver memory admission remains unproved.

Prompt dequant binds the actual raw slot, uses local zero offset and exact length/
role extent checks, retaining original packed formulas and paired products.
The old generic binder also declared every scratch stage as a hidden BDA read:
the prompt shader now has no stage-address binding, and its dedicated binder
retains only genuine original-model GPU/HOST roots. Decode still uses its real
stage pointer table/dependencies. This is a real interface change, not a
scheduler-only narrowing. The internal cache key carries `prompt-raw-slot-v2`.

Pinned Intel `prefill.cpp:3975` releases the slot after both dequants and before
GEMMs. The actual Down-image metadata now forces that same native recording
boundary; fresh/cached pipelines preserve it, and the copy queue can acquire its
timeline signal while matrix products continue from the separate dequantized
owner. Native release keys wrap/preserve original quantization identity, including
external masks; false leaves ordinary cache keys unchanged. Upload acquisition,
replay epochs, both pools/resources and final dual-queue completion remain joined.
No timing switch or speed-driven fallback was introduced.

Release build, strict core/feature-off Clippy,11 queue tests, three raw-view/cache
tests,308 shader variants (122 physical validations) and32 matrix variants pass.
No live GPU/model/numerical/state/recovery/speed run; full unit suite not rerun and
the prior228-failure baseline remains unaudited. No global tensor root, API/Vashti,
installed service/model residency, installation or release change. Canonical and
bytecode preserved. Goal active, rewrite incomplete, speed gate closed. Root was
not needed for source/offline work; last earlier credential check had expired.

**Next port native prompt allocation/placement and configured size policy as a
connected block.** Capacity256/fixed three128MiB stages do not establish native
owned/default2048 grouping or cache loans. Do not emulate2048 by eight separately
routed256-row prompts. Audit the true Intel owned-at-context<=32768 / otherwise
loaned branch, dependent GR/GDN/QSA/MoE/PLE/MTP owners, INFO/host/runtime charges
and expert-cache placement together. The receipt explains native STAGE8/DQ2 and
conditional stream-all/big-ring policy: do not assume a large ring is always
selected (nonresident share>=10% defaults to eight even for large T).

Then finish complete branch/graph capture, arbitrary input/sampling, verifier/
repair, source and unit-failure audits, followed by protected whole-engine
completion/numerical/state/recovery acceptance and only then throughput. The
passing host schedule oracle does not prove actual GPU overlap or performance.
Required Vashti changes: none.


### Active Kuno engine handoff — owned prompt allocation foundation (2026-10-14)

Source `788e793`, isolated `codex/strata-rust-port`, following range `3ca345c`.
Read completion guide **Connected native-owned prompt allocation foundation**.
**Actual prompt math/controller/grouping is still capped256. This is not
2048-row completion, cache lending or permission to benchmark.**

The actual constructor now prices/registers three separate512MiB private prompt
owners (MAIN delta1,610,612,736 bytes); decode keeps its three128MiB fetch stages.
`core_prompt_stages` supplies the real prompt Scope, and the source binder attests
the actual same-stream whole raw owner. Copies/slot descriptors stay in its
lower64MiB; ordinary decode tables/dependencies are unchanged. All new owners are
in real request/MAIN inventory, normal close and unknown-in-flight retention.
These power-of-two Rust envelopes are conservative, not measured SYCL/VRAM parity.

Native allocator/admission supports512MiB owners with explicit device buffer/
binding limits; ordinary feature-off retains128MiB/pools. Cleanup now inspects
complete roots through32MiB readbacks, wiping/dropping/sanitizing each lease
before the next and publishing root credit only after its entire tail/padding
passes. No enlarged host readback allowance, untracked owner or ordinary-I/O
fallback. Policy recognizes native default2048 and owned-at-context<=32768;
CacheLoan is explicitly unimplemented and cannot silently become an owned
allocation. No configurable chunk/loan implementation is claimed.

Release build, strict core/feature-off Clippy,18 arena tests in each mode,
policy/inventory test, actual256/2048 persistent/phase carving test, three slot/
cache tests,308 shader variants (122 physical) and32 matrix variants pass.
No live GPU/model/allocation/erasure/numerical/state/recovery/speed acceptance;
full core suite not rerun, prior228-failure baseline still unaudited. No API/Vashti,
installed service/residency, install or release change. Canonical and bytecode
preserved. Goal active, rewrite incomplete, speed gate closed.

**Next propagate true2048 rows through all actual consumers as one block.** The
receipt lists exact map/status collisions and suggested owned-view separation:
LOGICAL4100 overlaps MEMBERS1540+2048*10; old MoE roots remain decode-sized.
Carve real prompt-specific raw/grouped maps/status/GU/PART views from the priced
owners and connect them to router/grouping/dequant/matrix/reducer/closing proof.
GU needs16MiB and PART256MiB padded at2048. Keep verifier MAX_BATCH8 separate and
leave workgroup256/CB32/attention32 tiling alone. Audit full working images,
PLE actor rows/charges/history and MTP four-residual rows (potential matrix count
8192) as well as GR/GDN/QSA scope/count/state acceptance and controller chunks.
`Storage::at_capacity(2048)` proves only its listed view bounds, not all consumers.

After full2048 flow, implement real cache loans/restoration/exclusion and
conditional ring/stream-all sizing with actual cache placement; then input/
sampling, branch/recording, verifier repair, source and unit-failure audits before
protected whole-engine acceptance and speed comparisons. No privileged work was
needed; the last earlier sudo check had expired. Required Vashti changes: none.


### Active Kuno engine handoff — connected native2048 owned consumers (2026-10-14)

Source `d75549f`, isolated `codex/strata-rust-port`, following `788e793`.
Read completion guide **Connected native2048 owned prompt consumers** for exact
views, matrix/dispatch geometries, accounting, receipts and remaining gates.
This supersedes the previous256-row consumer limit. **This is source/offline
acceptance; the complete rewrite remains unfinished and speed benchmarks are
still prohibited until its completeness gate passes.**

The actual full-chunk controller, request authority, GR/GDN/QSA/PLE/MTP scopes
now use2048 rows. MoE has real, distinct raw packet/grouped map/probability/
role-status/GU/PART descriptors carved above the existing64MiB raw ring in the
third tracked512MiB prompt owner. The router writes the raw view, readback
preserves its original offsets, CPU grouping writes the separate map, and
expert gather/dequant/cooperative matrices/activation/reducer bind those same
views. The reducer no longer reuses old decode map/status/probability roots.
Logical IDs begin at32768, outside all20480 physical/member routes; role status
planes are separated by20480. Per-layer proof views are cleared, while actual
48-layer count/epoch stamps still join the completed whole-model sticky fault.
Prompt image ABI is `prompt-owned-routes-v3`; original format/release identity
is preserved. Ordinary decode/MAX8 roots and verifier capacity stay unchanged.

Both projection and dense matrix admission cover MTP's8192 FC-hidden rows at
2048 target rows. Full residual/teacher normalization use actual80MiB carved
views; packed working images remain bounded64MiB. True linear copy/embedding/
residual-write/pack/dequant kernels flatten oversized grids using actual
`num_workgroups`; GR/head/attention axes remain independent. QSA attention stays
32-row tiled, its row offsets and whole-prompt guards now cover2048, and INFO
includes64 tile pairs. PLE's exact last-nine-row history transition accepts2048;
prefetch metadata is heap-backed/wiping, unique original reads remain deduplicated,
and reconstruction scatters directly by destination. Group host quote1MiB,
PLE source64MiB and separate outer64MiB host/runtime copies are conservative
actual-lifetime bounds; no extra arena owner beyond the priced prompt roots.

Final core release build, strict core/feature-off Clippy,308 shader variants
(122 physical validations plus32 cooperative variants), all12 prompt tests,
source-plan coverage1..=2048 and the scalar-vs-tail PLE history test through2048
pass. Exact log names are in the completion receipt. No live GPU/model/
numerical/state/recovery/speed acceptance; full core suite not rerun and prior
228-failure baseline remains unaudited. No installed service/residency, public
API/Vashti, installation or release changes. Canonical and inherited bytecode
preserved; no privileged action was needed. Goal stays active.

**Next: native cache loans/restoration/exclusion and conditional ring sizing
against real cache placement/owner bytes.** Owned envelopes remain conservative
fixed512MiB; context>32768 lending is still explicitly unimplemented, and current
request/QSA/MTP context is bounded4096. Configurable prompt/long-context fidelity
is not established. Then finish arbitrary input/sampling, complete branch/
recording and verifier repair, source/unit-failure audits and protected whole-
engine completion/numerical/state/recovery acceptance before throughput.
Do not revert coupled port work because of incidental timing. Required Vashti
changes: none.


### Active Kuno engine handoff — contiguous cache / bounded source windows (2026-10-14)

Source `16e99d8`, following `d75549f`, isolated `codex/strata-rust-port`.
Read completion guide **Cache contiguity and bounded public initialization**.
The previous goal turn made source/verification progress. **Cache lending is
still incomplete; this is a real model-loader prerequisite, not a loan receipt
or permission to run speed benchmarks.**

Actual global plan/GPU allocation now packs ranked experts into roots up to
512MiB while retaining16MiB original-model host uploads. GPU contiguity and
host staging no longer share one constant. `Cache::load` allocates an empty
persistent root, fills every byte/padding using `load_chunk_range` and exact
public upload views, and drains each window before another. Source tables and
consumers publish only after complete full-root attestation. Exact root requests
still feed persistent admission; actual `Cell` addresses and original ranked
prefix/HOST complement remain authoritative. The setup quote stays two16MiB
windows plus the1MiB HOST reader, and the internal receipt distinguishes root
limit from upload window. A smaller final root is not fictitious512MiB authority.

Core release build, strict core/feature-off Clippy and all five global-plan tests
pass. They retain the independently recorded11570-pair prefix, empty/partial/
full complete source coverage and exact HOST quote. New tests reconstruct
original roles/padding with independent byte streams across awkward byte windows
and inspect pinned-model root/window boundaries without a512MiB host allocation.
Exact logs are in the receipt. No shader arithmetic changed; prior308 variants
were not rerun. No live GPU/allocation/model/numerical/state/recovery/speed run,
full unit suite, services/residency, API/Vashti, install or release change.
Prior228-failure baseline remains unaudited. Canonical/bytecode preserved; no root
needed. Goal active, complete rewrite and benchmark gate remain unfinished.

**Next connect actual cache-backed private ownership, exclusion and restoration.**
Pinned `sycl/src/program/generate.cpp:11524-11624` marks lent tail pairs unavailable,
initializes prompt scratch there and restores their actual slots before decode.
Kuno's private arena currently attests exclusive power-of-two private allocations;
public cache roots are actual alignment-sized owners, possibly smaller than512MiB.
A real borrowed registration must attest/charge the complete physical root,
include full erasure/inspection and retain it on unknown completion. No cast of
public handles into ordinary private ownership, fake larger descriptor or mere
host bitmap. Leased resident pairs also need a real original-weight staging
source (HOST complement only contains initially nonresident pairs), patched
physical source tables and retained BDA dependencies excluding borrowed cells.
Restore current placement/remaps after the joined prompt close, accounting for
adaptive exchanges before publishing residency. Then finish conditional ring/
stream-all policy and remaining arbitrary input/sampling, branch/recording,
verifier-repair, source/unit-failure and protected whole-engine acceptance gates.
Required Vashti changes: none.


### Active Kuno engine handoff — exact prompt parent spans (2026-10-14)

Source `c0e9efc`, following `16e99d8`, isolated `codex/strata-rust-port`.
Read completion guide **Exact physical-parent prompt view geometry**.
Previous goal turn made model-loader/verification progress; this turn removes
another actual consumer blocker for lending. **No borrowed private registration,
loan exclusion/staging/restore transaction or complete rewrite is claimed.**

Scope, Storage, MoE views, raw-slot slicing and source attachment now derive
slices from each whole parent's actual size, up to512MiB. This supports real
alignment-sized cache backing rather than subtracting fictitious512MiB tails.
The actual family carve enforces its own minimum. Whole-root structure rejects
slices/misalignment/oversize; Storage rejects duplicate identities and foreign
streams. Source attachment still proves complete backend buffer offset/size,
checks real MoE fit, and requires copy destinations to match that attached root.
Working images stay64MiB, raw ring stays below64MiB, descriptors/math/default
owned allocation/pricing are unchanged. Structural validation is not a borrowed
ownership capability; Scope still receives private-arena BufferIds, whose
exclusive power-of-two allocation and cleanup policy are unchanged.

All15 focused prompt tests and core release build pass. The new cases accept
independent non-power-of-two parents, derive/accept exact family minima from
actual views, reject a256-byte deficit in each parent and reject aliases,
slices, foreign streams and oversize; raw-slot tails match real owning ends.
Final strict-check receipts are recorded in the completion guide. No live GPU/
model/allocation/erasure/state/recovery/speed, full unit suite, services/residency,
public API/Vashti, install or release change. Historical228 unit failures remain
unaudited. Canonical and bytecode preserved; no root needed. Goal stays active.

**Next implement genuine private borrowed-root ownership and the connected
source transaction, not more pretend loan geometry.** The actual `PrivateArena`
ledger/cleanup still only registers newly allocated exclusive power-of-two
private roots. Cache roots require attested complete physical backing and a
source-owner exclusion capability before private registration, full erasure/
inspection, retention/retirement on unknown completion and exact shared physical
accounting without invented MAIN credit or hidden allocation. Candidate carving
can now validate actual cache handles without forging a larger root; this alone
must never publish a Scope. Leased resident pairs require real original-weight
staging, patched source tables and BDA dependencies excluding scratch cells;
restore current placement/remaps only after joined private close. Continue the
remaining conditional ring/configuration, branch/recording, input/sampling,
verifier repair, source/unit-failure and protected whole-engine gates before
speed testing. Required Vashti changes: none.


### Active Kuno engine handoff — registered cache-root borrowing component (2026-10-14)

Source `09351ea` + private audit-purpose correction `03ea215`, isolated
`codex/strata-rust-port`, after `c0e9efc`. Read completion guide **Registered
private cache roots and erasure-before-restoration transaction**. The previous
goal turn made source/bounds-test progress; this turn adds the actual source-
owned private borrowing component. **Prompt MoE/controller/configuration still
do not select it; complete loan-mode inference is not implemented or validated.**

The private arena now has a separate unsafe registration of actual complete
cache buffers. It checks real backend buffer offsets/sizes/identity and exact
aligned physical spans, retaining all roots before zero initialization. It
allocates no new GPU root and its local ledger is private coverage of already-
precharged persistent banks, not new MAIN credit. Ordinary power-of-two reserve
stays unchanged; registered arenas cannot allocate or reset. Close mandates
complete erasure/32MiB-window inspection and exact root/byte counts, retaining
unknown/unaudited roots on failure. Borrowing mandates strict private audit
without enabling public-fixture permission/replay.

`Sources::with_private_cache_roots` validates actual candidates, joins readers,
blocks all model source consumers, registers the arena, runs the protected
same-worker body and closes/inspects before restoring current original model
weights in16MiB windows. Only both complete phases can reopen sources; failed
registration/erasure/restore retires the executor's actual shared context and
keeps exclusion/quarantine. Body errors/panics still clean/restore before being
returned. The unsafe body must not leak IDs/handles/Scopes/replays. It receives
only arena/IDs, deliberately no permitted model Sources yet.

A wiping inverse physical placement now follows actual adaptive role exchanges.
It validates current incoming/outgoing identities, then commits only after
joined private remap/usage and adaptive-owner publication. Restoration uses
current logical experts, not initial profile bytes. Current/pending maps and
metadata get an explicit extra128KiB HOST quote; nothing is logged/persisted.

Core release, strict core/feature-off Clippy,20 core arena tests, two source gate/
placement tests and five plan/HOST quote/window tests pass. Final ordinary arena
receipt is in the completion guide. Exact logs and failure cases are recorded
there. No shader math changed. No live GPU/model/allocation/erasure/restoration/
state/recovery/speed acceptance or full suite; prior228 failures remain unaudited.
No installed services/residency, API/Vashti, install or release changes. Canonical
and inherited bytecode preserved; no root needed. Goal active, speed gate closed.

**Next connect prompt-only leased sources and the controller as a real block.**
Current borrowing intentionally revokes every prototype/bind/HOST reader, so
it cannot run MoE. Leased resident pairs are absent from the existing HOST
complement: give them a real bounded/quoted original-weight staging source,
using the current placement, and preserve native-copy queue/header/acquisition
semantics. Do not substitute untracked ordinary uploads for native transfer
nodes. Patch the actual source table/flags and BDA dependencies, including
prototype bindings; exclude scratch roots. Redirect leased pointers to real
original-weight backing as well as setting a flag, so hidden/speculated BDA
loads cannot alias private scratch. Restore original source tables/read sets/
prompt-stage attachment only after joined private close and model restoration.

Thread the registered arena's IDs through Scope/full prompt/MTP flow and revise
actual constructor/cache pricing and configured loan selection. `Plan::CacheLoan`
still errors, default construction still has three512MiB owned prompt roots and
context remains4096. Keeping unused owned buffers is not loan/cache price parity.
Then finish conditional ring/stream-all/auto chunk policy, branch/recording,
input/sampling, verifier repair, source/unit-failure audits and protected complete
engine acceptance before throughput. Required Vashti changes: none.


### Active Kuno engine handoff — prompt-only cache sources/controller (2026-10-14)

Source `60f6760` + aggregate-GR correction `12b310f` + model rebase `8b503e2`, after `f82d00c`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Connected prompt-only cache sources and one-session controller**. This
supersedes the previous handoff's missing fallback/table/prototype/controller
hookup. **The full rewrite and live validation remain incomplete.**

Real bounded registered HOST pages now hold original weights for lent cells,
refreshed from current placement under joined exclusion. Each page authenticates
its actual single source shard. Typed registry extension preserves all live and
retired identities/counters and quotes the metadata peak. A separately explicit
RAM grant is required; the compiled4096 fixture grants zero and stays owned.

The actual source transaction patches pointer/flag tables, prototype handles and
BDA read dependencies before exposing private scratch; native queued HOST copies
remain in use. One owning session/Scope spans the complete prompt plus MTP
teaching, with decode/adaptation excluded. Full private erase/inspection and
current-placement model restoration precede normal source publication. Unknown
completion/abandonment retires the shared context and retains owners. Loan mode
omits the three owned512MiB buffers from constructor/MAIN pricing. The obsolete
local2GiB arena ceiling is replaced by the actual precharged model plan.

Exact build/Clippy/focused test results and limitations are in the receipt above.
No GPU/model/allocation/state/recovery/speed acceptance or full suite this turn;
prior228 full-core failures remain unaudited. No services/residency, public API,
Vashti, install or release changes. Canonical/bytecode preserved. Goal active.

Next connect runtime context/configuration and pre-device RAM grants/cache
budget, conditional ring/stream-all/auto chunks, arbitrary input/sampling and
remaining captured branch/verifier-repair and source/unit-failure audits. Current
bound graph/request permits still cap4096; compiled conditional loan consumers
are not a supported live long-context invocation. Complete the whole-engine
correctness/state/recovery gates before speed tests. Required Vashti changes:none.

The follow-up `8b503e2` makes original-weight refresh explicit in the typed
registry, rather than mutating a resource still declared immutable. Begin revokes
lookups and joins native queues; its non-cloneable token grants only authenticated
model rebasing. Both fallback refresh and adaptive GPU/HOST exchange consume it.
Successful initialization/flush/raw completion is required to resume. Abandonment
keeps readers revoked and backing retained. This is an internal engine contract,
with no Kuno HTTP or Vashti change. See the final rebase receipts above.

**Current hardware coordination:** the user has reclaimed both GPUs while this
source/build phase continues (2026-10-14). Do not start GPU/model execution,
change admission/residency or stop their workloads without reconfirming that the
cards are available. Source edits, CPU-only unit checks and compilation may
continue. No GPU job or installed-service change was started in this pass.


### Active Kuno engine handoff — native last prompt input (2026-10-14)

Source `97e22f9`, following `8b503e2`, on isolated
`codex/strata-rust-port`. Read completion guide **Native prompt prefix and final
decode boundary**. Previous turn made authoritative cache-loan/registry/arena
progress; this turn corrects another real coupled math/controller difference.

The selected controller now follows pinned generate.cpp: batch N-1 prompt inputs,
round the requested chunk to256 (cap2048), teach every prefix R_i with input_i+1,
then close/join prompt scratch and restore any loan BEFORE normal count1 decode
consumes the last original input. It predicts the first result, including for a
one-token prompt. The former prompt-precision final head/full-logits readback is
removed. Actual positions/committed cursor/carry are checked; MTP remains atN-1.
The PLE worker prefetches only the prefix and joins before ordinary decode.

Exact source/build/host-check receipts are in the linked section. No GPU math,
model completion, numerical/state/recovery or speed acceptance and no full suite.
The full Rust rewrite remains incomplete. No public API/Vashti, service/admission/
residency, install or release change. Canonical and inherited bytecode preserved.

Continue runtime context/pre-device grants/cache budgeting, native conditional
ring/stream-all/auto chunk policy, actual GPU sampling and captured verifier/state
repair/source/failure audits. Sampling source pointers/semantics are recorded;
no sampling implementation is claimed. The user is using BOTH GPUs: reconfirm
availability before any GPU work. Goal active; required Vashti changes:none.


### Active Kuno engine handoff — sampler source component (2026-10-14)

Source `267a71b`, after `97e22f9`, on isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Native sampler source and request-data boundary**, including its final CPU
validation receipt and actual pinned caller references. This turn makes source
progress; it does not complete the engine/sampling gate.

New Rust-owned sampler component follows original penalties,1024-thread greedy,
4096-logit/four-subgroup split top-k, private two-list merge cursors, distributed
exponentials/quotients with ordered F32 scans, Philox and id/probability output.
History helpers match each target row and the coupled draft base/cell+1 counter.
Request parameters are wiped host-side and must live in a private128-byte DEVICE
storage view; kernels/cache identities contain no seed/prompt/history/counter.
No implicit GPU allocations, host-logit sampling or persistent content is added.

**Not connected yet:** the actual controller, target packet and MTP head still
use earlier greedy paths. Do not claim sampled model inference from these helper
checks or call the unsafe launcher without actual parent ownership/staging/price
proofs. Next connect request configuration/consumed history, pre-device MAIN/
HOST/I/O grants and real private owner, target sampler-to-packet consumer,
coupled GPU stage/ring/actual step-record counter/publish chain and actual MTP
probability consumers together. Preserve exact-match TARGET sampled-token
acceptance and repair only accepted input state/history. The completion receipt
identifies the pinned target call outside its recorded graph and the MTP
round/step device-staging dependencies; read them before making the hookups.

All7 focused CPU checks, core release build, strict core/feature-off Clippy and
formatting pass. Full offline shader set is312 variants, with122 existing
physical-pointer and32 cooperative-matrix checks. Use the already-built pinned
2026.1 SPIR-V validator path in the receipt; distro2025 does not understand the
existing BF16 capabilities. No GPU/model/math/state/recovery/live-speed proof or
full-suite pass. Prior228 full-core failures remain to audit. Remaining runtime
context/grants/cache policy, conditional ring/stream-all, captured branches and
verifier-repair dependencies also remain; read the whole completion gate.

Both GPUs remain with the user: request availability before any GPU/model test
or residency/admission/service change. Source/CPU work can proceed without sudo.
No installed services, API, Vashti, deployment or release change. Canonical clean
checkout and inherited bytecode preserved. Goal active; required Vashti changes:
none.


### Active Kuno engine handoff — request sampling connection (2026-10-14)

Source `ab0b15c`, after `267a71b`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Connected request sampling owners and target/MTP hooks**, including the CPU
receipts. This supersedes the preceding missing owner/target/MTP ID hookups;
**full rewrite, sampling-policy parity and GPU execution remain incomplete**.

The selected controller now owns wiping consumed-input history/settings through
a same-thread lexical sampler Scope. Seven real full private owners add557056
MAIN bytes, quoted constructor/host/runtime costs and no implicit GPU allocation.
The constructor's unsafe lifetime contract retains original arenas/models and
pre-device grants through close/retirement. Same-client/binding/full-owner checks,
no reset/close, non-Send/Sync guard and complete private uploads are explicit.
Target sampling runs after the actual head inside its protected packet-I/O
closure, feeding a separate authenticated selected-id/probability/carry packet.
History advances only after actual accepted state/input-prefix commit. The first
normal prompt input follows that same target path.

Sampled full-Q2 MTP hooks read the native-front DEVICE position table for
cell+1, check its real GPU input against the coupled ring, apply the same chain
and draw, publish the selected token to the existing draft/follow consumers and
extend the ring. Probability bits are retained in the private ring. Each param
word has one writer. Greedy retains native uncoupled drafting; subset-head use
is rejected for this owning original-vocabulary request. Normal close erases/
inspects all seven owners before original transient-I/O closure. Abandonment
retires context and retains original arena cleanup authority. None of this
runtime/numerical/erasure behavior has been physically accepted yet.

Final core build, strict core/off Clippy, formatting,10 focused CPU checks and
offline shader validation pass.314 regular variants +122 physical pointer and
32 cooperative-matrix checks; see exact paths in the receipt. No full-suite run
or waiver of the prior228 failures. No GPU/model/device operation, speed test,
installed service/admission/residency change, API/Vashti, deployment or release.
Canonical clean checkout/inherited bytecode preserved. Both GPUs remain with
the user: reconfirm before ANY GPU work; source/CPU work needs no sudo.

Next source block: original confidence/probability/early-stop/chain-launch
consumers and greedy top-probability behavior, then the remaining captured
branch/repair/failure audit, arbitrary request/context/grants/cache policy,
conditional ring/stream-all and whole completeness audit. Current DraftWindow
still returns ids and old cost/suffix policy; probability storage is not a
consumer-parity claim. The harness explicitly constructs greedy Params while
internal Request takes sampling parameters; a general input command is still
pending. Retain the full rewrite goal and all protected numerical/state/recovery
gates before any speed trial. Required Vashti changes:none; goal active.


### Active Kuno engine handoff — native MTP confidence (2026-10-14)

Source `8f1377c`, after `ab0b15c`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Native MTP confidence producers and serial consumers**, its substitutions and
final CPU receipts. This supersedes the earlier missing probability/serial
threshold consumers; **full rewrite and physical acceptance remain incomplete**.

Native full-Q2 head now uses1024-thread no-penalty argmax and8x128 top probability
with original strided F32/XOR32/ordered32-warp sums, last-CTA counter and explicit
Device release/acquire ordering. Atomic u32 bit parts replace native volatile
floats; exp accuracy and physical memory ordering remain to accept. Existing
Candidates32KiB root lends disjoint scratch/word64 probability views. No new MAIN.
Coupled sample Publish feeds its actual final-distribution probability into
that same slot; it does not replace sampled confidence with an argmax mass or1.
Scalar and queued draft packets return bounded wiping probabilities and check
full owner/padding. The native mode keeps its original32/64KiB owner sizes.

The controller takes native maximum length capped by remaining output, with
new internal `draft_min_probability` (public harness explicitly0). Zero queues
all draft steps before one final fence. Positive values gate launching each
later step on the preceding completed probability, and the first low draft is
excluded from target inputs. Drafts complete BEFORE suffix-policy choice. Native
repair permits a longer accepted suffix than a confidence-stopped draft and
replays only actual target-approved token/residual pairs; Q8/frozen plan bounds
remain separate. Source-level CPU state/format coverage is not numerical proof.

Probability0 is valid; no-token/NaN output initialization and sticky invalid-
launch faults preserve missing-writer detection. The additional full private
32KiB target-output initialization is charged: request/MTP runtime4063232,
MAIN557056/host2MiB unchanged. No seed/prompt/logits/probability persistence or
cache-key data. Existing same-thread Scope and full parent owner cleanup remain.

Final8 confidence +10 sampler/request checks, core release build, strict core/
off Clippy, formatting and offline316-body shader validation pass, including122
physical-pointer/32 cooperative-matrix cases. Packet expansions cover checked/
unchecked and native/legacy modes. Exact command/log paths are in the receipt.
No GPU/model/device, live speed or full-suite acceptance; prior228 failures are
not waived. No service/admission/residency, API/Vashti, install or release change.
Canonical/inherited bytecode preserved. Goal active, required Vashti changes:none.

Next: coherent native capture_round/capture_step and accepted-front catch-up/
next-rest grouping (current repair-front then next-cell split is still different),
applicable nonblocking/force-token chain event consumers, stable captured branch/
state/failure audit. Keep suffix/cost/no-MTP lookup and runtime input/context/
cache sizing/conditional-ring/stream-all and remaining completion rows open.
No throughput trials until full source and protected correctness gates pass.
Both GPUs remain with the user: request availability before ANY GPU/model or
residency/admission work. Source/CPU work may continue without sudo.


### Active Kuno engine handoff — native grouped MTP round (2026-10-14)

Source `f671543`, after `8f1377c`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Grouped accepted MTP front and next draft rest**, including remaining device
residual/graph substitutions. Full rewrite/physical acceptance remain incomplete.

The continuing controller moves actual accepted target predictions/R rows into
one bounded wiping request-local prefix. Next MTP round checks original binding/
cursor/token, retains the known p-1 cell, rewinds provisional p onward, and runs
one front for predicted[r]/R[r] at p+r, r<n. This includes the new confirmed
last input. One GPU kernel selects the last row's R/injection/mixed together,
then requantizes mixed before the real rest/head. No eager n-1 accepted repair
followed by separate next-cell front remains between speculative windows.
Native scalar follow-ups, confidence gating/queued zero-threshold completion,
actual sampled counter/penalty ring and unchanged target prefix verification
remain connected. Final/count1-only exits keep the exact old cache end boundary.
No new owner/budget or persistence; complete parent arena authority is retained.

Important remaining gap: Strata binds the verifier's actual device R_ rows to
MTP. Rust still reads the full private verifier packet to wiping host carries
and uploads them. Next connect a lifetime-bound original device-source grant/
copy with safe scratch retention; a view is not erase/free authority. Stable
whole round/step captures, mapped/event publication, applicable force-token/
nonblocking chains, commit overlap and physical state/failure/recovery remain
open. Grouping alone does not prove those dependencies or numerical parity.
Runtime input/context/cache sizing, conditional-ring/stream-all, full-source
and protected correctness gates remain required before any live speed trial.

Final4 new CPU checks +8 confidence +10 sampler/request +1 existing all-width
prefix check pass. Core release build, strict core/off Clippy, formatting and
317 ordinary/122 physical-pointer/32 matrix offline shader checks pass. Exact
logs/limits are in the receipt. No full-suite waiver of prior228 failures.
No GPU/model, speed, sudo, service/admission/residency, API/Vashti, installation
or release operation. Canonical clean checkout/inherited bytecode preserved.
Both GPUs remain with the user; reconfirm availability before any GPU work.
Required Vashti changes:none. Whole rewrite goal remains active.


### Active Kuno engine handoff — original device residual grants (2026-10-14)

Source `2532963`, after `f671543`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Original verifier device residual grant for accepted MTP rounds**. Full rewrite
and physical acceptance remain incomplete; this is backend-only engine work.

The real compact packet producer now issues a private source receipt tied to
its actual outer residual BufferId, opaque graph, write generation, start/count
and wiping target picks. Before native/prompt/scalar/reset writes, old receipts
are invalidated without generation wrap. Actual state-only prefix commit/public
cache adaptation preserve the original R rows. The parent checks committed state/
cursor, source membership/full extent/provenance and original identity before
returning a lifetime-bound immutable arena borrow. A receipt alone is neither a
raw handle nor erase/free permission. Prediction prefix and accepted copy ranges
are checked; same-client/stream/disjoint full owners are authenticated again.

Selected continuing MTP rounds copy the original selected R row into B::Carry
and accepted R prefix into native CARRY, preserving grouped last-row R/injection/
mixed selection, quantization, sampler counter/history and target verification.
No host carry vector is retained as the next round's MTP source. The source
borrow survives the real result/fault fence. An unjoined copy retires the original
source context; original parent cleanup authority remains. No new MAIN/runtime
owner; core host increment adds8192 bytes (now12,591,104). No content persistence.

Remaining: initial/final/count1 host-carry/repair and full residual packet readback
still exist. This is not a claim of zero host residual reads. Connect those device
paths and ids-only finite-checked results, stable native round/step/branch capture,
mapped/event publication, applicable nonblocking/force-token/commit overlap,
runtime sizing and full-source/physical correctness/recovery gates before speed.

Final28 focused CPU checks, core release build, strict core/off Clippy, fmt and
317 ordinary/122 physical-pointer/32 matrix offline shader validations pass.
Exact logs and limits are in the receipt. No full-suite waiver of prior228 failures.
No GPU/model/device, sudo, service/admission/residency, API/Vashti, install,
deployment or release change. Canonical clean checkout and inherited bytecode
preserved. Both GPUs remain with the user; reconfirm availability before GPU work.
Required Vashti changes:none. Goal active; CPU source work may continue.


### Active Kuno engine handoff — native device ends and constructor gap (2026-10-14)

Source `9edc3f6`, after `2532963`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Native target ids-only results and first/request-exit device paths**.
**The current core speculative harness is not runnable through the shared-client
path yet.** CPU/source connection is not whole-model acceptance.

Selected controller explicitly opts into target packet v3:96 logical bytes for
ids, geometry, prior fault and per-row finite status; original R stays in its
full device owner. GPU128-thread row groups inspect all active10240 residual
values. CPU checks every id/status/header/inactive row, then issues the same
original device receipt. Legacy host-carry mode remains separately scoped.
First proposal now uses the final prompt's original device receipt, including
cell0. Count1 target inputs front real KV without a proposal head; distinct
LastFront cannot authorize LastForward/follow. All mutation/prompt/close paths
invalidate old stamps. Native EOS/cap exit commits target state and does not
perform another MTP catch-up; only original parent close owns cache erasure.
No new GPU owner/budget or request persistence. Target host residual extraction
is gone in the selected controller. Positive-threshold draft hidden readback is
still a separate remaining gap.

NEXT, before capture work: `daslab_language_probe::draft_client` still creates
another Vulkan device/client; caller configures independent runtime/backlog.
`ResidentMtp::construct` demands a fresh client and6GiB ledger baseline0. Prompt
device teaching/coupled sampling/direct grants require the actual same client,
so this old constructor cannot satisfy the new path. Unify Intel construction,
combined pre-device/main/runtime/host accounting, transient-I/O policy, retirement
and both-owner close ordering. Do not relax identity checks, clone the client
without resolving budgets, fake peer allocation credit, or pass foreign handles.
Source/CPU work is available without sudo/GPU; goal is not blocked.

Afterwards finish native capture/warm/event/chain/commit dependencies, reconcile
configured native draft count versus remaining-output/count1 policy, compact
positive-threshold observations, runtime/context/cache/ring and full-source/
physical correctness/recovery gates. No speed trials yet.

Final32 focused CPU checks, core release build, strict core/off Clippy, format,
317-body/controller +122 physical-pointer/32 matrix offline validations pass.
New ids packet also passes external Vulkan1.3 SPIR-V validation. Exact final
log prefix `strata-device-ends-accepted-*` is recorded in the receipt; it does
not mean full-rewrite acceptance. Prior228 core failures remain unaudited.
No GPU/model, sudo, service/admission/residency, API/Vashti, install, deployment
or release change. Canonical/inherited bytecode preserved. Both GPUs remain with
the user; reconfirm before GPU work. Required Vashti changes:none. Goal active.


### Active Kuno engine handoff — shared native Intel construction (2026-10-14)

Source `e418929`, after `9edc3f6`, isolated `codex/strata-rust-port`
at `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Shared native Intel construction, accounting and close**. This supersedes the
preceding separate-client constructor gap, not the whole-rewrite gate.

Core construction now consumes a one-shot original-graph permit and uses the
actual target client, MAIN account, request-I/O scope and retirement flag for
original Q2 MTP. Peer capacity is priced before target/expert allocation. Both
owner quotes and49 address-table roots leave the exact global ranked-prefix
budget; HOST complement follows that plan. Core admission now accepts the actual
allocator's512MiB roots. Fresh Xe physical proof uses the combined31GiB envelope
plus512MiB margin; shared transient I/O is256MiB, while non-core target policy
remains128MiB. The real MTP arena closes before target poisons the shared account;
all remaining owners/public IQ tables close before the actual final sync/audit/
zero-allocator observation. Profile close validation includes the25 native-front
roots and split-attention geometry. Independent legacy laboratories retain their
separate clients/accounts. No identity checks are weakened or zero bases invented.

Final46 distinct core checks (7 shared,10 MAIN,32 prior regressions with3
MAIN overlaps),7 non-core shared checks, strict core/non-core Clippy, core release
build and317 ordinary/controller +122 physical-pointer/32 matrix offline shader
checks pass. Exact log prefix `strata-shared-context-accepted-*` and limitations
are in the completion receipt. This is not full-suite or whole-model acceptance.
No GPU/model, speed, sudo, service/admission/residency,
installation, API/Vashti, driver or release operation. Canonical checkout and
inherited bytecode preserved. Both GPUs stay with the user; obtain availability
before any GPU work. Required Vashti changes:none. Whole goal active/incomplete.

Remaining: target embedding/head roots are still duplicated; implement actual
pinned shared ownership without LRU/refund errors. Complete native stable capture/
warm/event/chain/commit and policy/input gates, remove unused positive-threshold
draft hidden readback, and run the full source/physical numerical/state/recovery
audit. Revalidate the protected HOST/cgroup envelope for the actual complement.
Historical228 full-core failures remain unaudited. No speed trials until the
whole rewrite and correctness gates pass.


### Active Kuno engine handoff — shared target dense roots (2026-10-14)

Source `9de7fef`, after `e418929`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. See completion guide
**Shared original dense roots and GPU embedding** for actual ownership,
accounting and the bounded-tile Vulkan substitution. This supersedes the
duplicate-weight blocker, not the incomplete whole-rewrite gate.

Original embedding/head are now pinned once, loaded before MTP/expert cache
and shared by real graph/client identity. Static target LRU cannot evict or
refund those roots; MTP peer admission prices only additional allocations.
Selected target prompt/decode embeddings gather on GPU. Actual private/loan
view membership and disjoint spans are checked without fabricating owner
authority. MTP releases its shared views before target closure; final real
shared-context close/audit remains required. Initial/final core backlog
receipts now query the same actual immutable shared-runtime policy.

Whole source, numerical/state and physical-recovery acceptance remains open;
no GPU/model/speed run, install, release or API/Vashti change. Required Vashti
changes:none. Both cards are now authorized for protected correctness work,
superseding older GPU-reservation notes. Sudo was refreshed in Demons. Continue
compact confidence observations, native capture/branch/warm/event/chain/commit,
policy and arbitrary-input/runtime gates. Preserve inherited bytecode and
canonical checkout. Historical228 core-suite failures still need audit.

Source9de7fef validation:11 core/shared and11 feature-off/shared checks,32 prior
focused native regressions, strict core/off Clippy, core release build and
318 ordinary/controller plus122 physical-pointer/32 matrix offline shaders pass.
Exact logs/limits are in the completion receipt. No GPU work was needed for
this source block.


### Active Kuno engine handoff — native policy and connected owner pricing (2026-10-14)

Source1554d5c (after395f4b2,575971f,a26adc2,9de7fef), isolated
`codex/strata-rust-port`. Completion guide **Native draft observations, lookup
policy and exact commits** contains exact changes, substitutions and evidence.
This is backend source-port work; required Vashti changes:none.

Core positive-threshold drafting now reads token/probability/fault only from
the original32KiB packet owner; real hidden vectors stay on GPU. Zero threshold
joins once, positive threshold joins each required step and preserves native
equality/stopping. Native four-way suffix index and lookup/chain cost policy
are connected, including unmeasured/stale probes and reached-chain accounting.
Policy costs include verify/commit plus the next actual draft. Configured
windows no longer shrink to output count; state commit and EOS/cap delivery
are separate. Temporary lookup tails, history/index and observations are
request-local/wiped. Original full-owner erase/fault/source identity remain.

CPU policy/confidence/grouped/prefix checks and strict core/off Clippy pass at
a26adc2. Exact original target-owner test passes at575971f. Protected runs
a26adc2,575971f,395f4b2 stopped BEFORE payload load at stale exact-parent quotes
for native scalar QSA, batch QSA and GDN INFO. Their connected parent consumers
are now priced in source1554d5c, preserving strict equalities and old feature-off
contracts. A stale scalar-QSA CPU expected value and first GDN INFO assertion
still need updating before final tests. Next protected completion check pending.

All three failed attempts restored Intel50,688,000 bytes, no NVIDIA compute,
zero unchanged classified faults, normal Kuno admission and all three HTTP200.
No service binary, driver, API/UI, install, release or model file was changed.
Both GPUs remain authorized; refresh/check sudo before protected work.

NEXT connected block after constructor completion: native mapped ownership
and CPU doorbell/plan service, stable complete target and MTP branch recording/
warm-up, parameter staging and event/commit overlap. Current CPU plan still
reads route/input per layer and native flush waits a fence;16 exact-node LRU
entries are not the native captured graph. Do not benchmark or revert source
based on incidental durations. Full rewrite, arbitrary-input/runtime and
numerical/state acceptance remain incomplete. Historical228 full-core failures
have NOT been waived by the specific reproduced/fixed constructor issues.


Latest continuation: source172e6df afterab4bf67/bb705ae. Both trace runs
identified a false empty-I/O rejection at shared target-weight loading, AFTER
real target construction; they recovered the same physical baseline and
services. Source172e6df adds a scoped completed-empty observation while
ordinary private-readback proofs remain nonempty and strict. Actual sanitizer,
source joins, completed epoch and MAIN/health checks remain mandatory. See
the completion receipt for exact contract and pending test/build/protected run.
The native QSA/GDN CPU owner fixtures now pass; earlier notes saying their
assertions still need updates are superseded by3fe048e. Whole goal incomplete.


Continuation throughf597084: see the completion guide's **Shared construction
continuation and real CPU/GPU submit boundary** receipt for exact revisions and
negative-test/physical evidence. Constructor fixes are connected to the real
caller, not waived validators. Latest protected851475a check gets past the PLE
zero-cursor gate and fails within original global-source setup; no full execution
or numerical acceptance. New CPU/GPU submit-only path retains original owners
through an actual completion fence; build/test pending. No installation/API/UI
work or other-task message. Historical pending assertions above are superseded
only by the named focused tests; full core regression remains unaccepted.


### Active Kuno engine handoff — shared source initialization and CPU/GPU submission (2026-10-15)

Isolated `codex/strata-rust-port`, sourceea2b53c after8b5f7aa/dbf2312/f597084/
851475a/5b91d9a/560474b/172e6df. No canonical checkout, installed binary, driver,
API, UI, release or model-file change. No speed/profile run or GPU reset.

Real shared constructors/global sources are past their prior blockers:
- Native exact QSA/GDN quotes and actual MTP split-attention device limits.
- 137 original MTP owners and combined GPU+HOST source inventories inspected
  through128-binding batches; full cross-batch physical checks preserved.
- Untouched zero-token setup proven separately from an accepted nonzero commit;
  completed-policy tokens still require real committed execution.
- Actual submit-only expert/shared GPU recording alongside scoped CPU workers;
  all owners/queue plans/fence retained, later mutation/read/sync/cleanup joins,
  failed/partial submissions cannot mint completion. Ordinary flush unchanged.

Frozen8b5f7aa gets through global source setup into request execution, then
fails resident graph device operation. Owning cleanup is NOT accepted; process
exit recovers Intel50,688,000 bytes, no NVIDIA compute, zero unchanged GPU faults,
normal Kuno admission and all three HTTP200. This is neither GPU crash evidence
nor numerical acceptance. Sourceea2b53c adds fixed-public phase/error-class tags;
release and strict core/off Clippy pass, protected follow-up running at writing.
The interim a3cd02a build failed a missing import; use corrected ea2b53c.

CPU checks throughdbf2312:15 native_shared and6 native_speculation pass; their
full-inventory helper covers4096 owners/failed batches. Strict core/off Clippy
also passes8b5f7aa andea2b53c. Full core suite/historical228 failures remain
unaccepted beyond the named repaired cases. See completion guide for exact
logs, immutable binary hashes, scope/substitutions and latest protected result.

NEXT: finish current execution-failure diagnosis, then the remaining connected
mapped observation/CPU doorbell service, stable target/MTP branch recording and
warm-up, parameter staging and commit/event overlap. Do not rank partial engines
by speed or back out connected source solely due duration. Full arbitrary-input,
numerical/state and physical recovery acceptance precedes speed comparisons.
Goal controller remains paused; user authorizes active-turn GPU work. No consumer
handoff beyond documentation is needed: required Vashti changes:none.


**Latest continuation (sourceec37fad):** see completion guide **Full-owner prompt
uploads and ordered carved copies / Final checkpoint**.539f205/f4400a7/eb4ff8e/
ec37fad fix all identified sliced private-upload callers without weakening the
full-owner API: target token packet, PLE/routing-map views and MTP teacher tokens.
Existing complete scratch roots are uploaded/zero-padded, GPU copies preserve
neighboring data, full scratch is restored before original padding guards. No
new GPU allocation or user-facing API. The combined protected run fails FIRST at
stream.rs1366's Wgpu upload completion poll; the later4-owner metadata rejection
is abandoned-mutation retirement, not the root failure. Actual PollError variant/
preceding validation error still need evidence. Do not waive validation, erase
proofs, retired health or raise deadlines. Final release/core/off Clippy pass;
17 focused shared cases pass throughf4400a7. No whole/model numerical/speed
acceptance. Worker exits, exact physical baseline/zero unchanged faults and all
services/admission recover. GPUs idle; no install/reset/reboot. Both canonical
and isolated source scope remain unchanged. Resume the whole-source gate; goal
still paused. Required Vashti changes:none.


**Superseding continuation (source30f5220, GPUs returned to user):**
Read completion guide **First submission validation and native root bookkeeping**
and **Prompt expert logical fault view and overnight stop**.31a2e6f identifies
WrongSubmissionIndex and validation_usage;0eb9b36 initializes each true storage
root separately with a readonly zero-work shader, preserving actual native
range/queue/owner contracts. Protected0eb9b36 and6d7f037 pass that earlier failure.
Latest physical6d7f037 rejects at strata_prompt_experts.rs:235. Actual caller
passed the full fault root (32 KiB), while the consumer correctly requires its
four-byte logical view. da7a74e passes u::handle(arena,fault) instead; the arena
still retains the full owning root for cleanup.30f5220 removes an unused import.
No further GPU run is authorized until the user returns the cards. Final state
strata-native-gpu-release-state-20261015.json: Intel50,688,000 bytes, zero NVIDIA
compute owners, all five classified fault counts zero, normal admission, all
three HTTP200. Owning erasure/numerical acceptance is still NOT established.
No speed comparison, install, model/service/driver change, reset or reboot.
Goal paused; next GPU gate is protected original completion with corrected view,
then continue the full connected rewrite. Required Vashti changes:none.


### Active Kuno engine handoff — selected Intel path and CPU-only continuation (2026-10-15)

This supersedes the older CPU-doorbell/warmup requirement and paused-goal notes.
Read the normative [completion gate](kuno-strata-rewrite-completion.md), including
its **Selected serving-mode audit and resumed completion checks**. The actual
retained Strata server enables DEVICE_PLAN/NO_HOST; GPU directly consumes real
resident or mapped HOST expert bytes. Generic CPU servicing and optional split
windows are not dependencies of that measured B70 path. Selected INT8 KV/CJK
draft head and full lazy branch capture remain required; no speed-driven retreat.

Latest integrated source on `codex/strata-rust-port`:
- `b17bbb0`: exact BF16 shared-gate vectors admitted as one-row projections.
- `36d13d3`: GPU direct HOST/VRAM expert plan, no per-layer CPU readback/join.
- `8c239bc`: authenticated four-byte fault view for the native final-head bridge.
- `19aa565`: request-owned stable branch parameter slabs and native MTP captures.
- `7aa574f`: target verification branch capture with actual immutable context.

Prompt prefix + MTP teaching pass protected execution. Final count1 head still
failed in the latest36d13d3 run; the proven view fix8c239bc is built but not GPU
validated. Whole-model generation/numerical/owning-erasure acceptance remains
open. Process exit recovery is separately verified at06:07:22Z: Intel50,688,000
bytes, no NVIDIA compute, fault counts0unchanged, normal admission and all three
HTTP200. Evidence: `strata-native-no-host-check-20261015/host-after.json`.
Release build through19aa565 and strict core Clippy through7aa574f pass.

**Current user instruction: CPU-only work, goal remains active.** No further
GPU/model/service work until the user returns authorization. Continue isolated
source/compile checks; do not interpret cached sudo as GPU permission.
Current peer worktrees are `strata-branch-parameters` (PLE staging / typed graph
invocation), `strata-int8-kv` (target/MTP INT8), and `strata-resident-first` branch
`codex/strata-native-cjk` (original-order CJK head + coupled sampling). Their
unintegrated work is not validated main behavior; preserve it if interrupted.
No frontend or shared application API change. Required Vashti changes:none.
No release, installation, reset, reboot, model mutation or task message.


**Final superseding stop:** user requested finish-and-stop after usage reset.
Goal paused. Source4bf3ed9 includes INT8 target/MTP cache and actual GDN binding-role
capture keys; locked release strict core Clippy passes. INT8 shader/unit/GPU
validation remains pending; prior317shader receipt predates INT8. PLE staging is
connected but true direct graph invocation/receipt separation is unfinished.
CJK work stays isolated pending review/integration. See final completion-gate
receipt for exact source, tests, substitutions and next steps. Last physical
recovery evidence remains06:07:22Z, no subsequent GPU run or service change.

Isolated CJK source is saved cleanly as`cd703bc` on`codex/strata-native-cjk`,
worktree`/home/jmicko/.codex/worktrees/strata-resident-first/vashti`. Detailed
handoff: `docs/kuno-strata-cjk-checkpoint.md` in that worktree. It connects
original-order106299-row Q6 head, public ownership/budget and mapped coupled
sampling; no compile/tests/GPU acceptance. It adds224800768 public device bytes
and no private roots. **Not cherry-picked into main engine yet.** Next resume
review/cherry-pick onlycd703bc, preserve the distinct INT8 Append changes, then
compile and validate all compact-head/sampler variants. The two other peer
worktrees are clean and their completed source is already integrated.


### Active Kuno engine handoff — combined CJK/INT8 CPU integration (2026-10-16)

Goal resumed CPU-only. Source0bfbea8 integrates the isolated original-order CJK
head componentcd703bc, preserving intervening INT8/cache/capture/PLE source.
Do not reapply the isolated commit. Source83035de adds exact full/tail Q6 shader
coverage and checked sampler scratch sizing; ed6373d is equivalent map-guard
style cleanup. Target vocabulary staysfull248320; only draft head/sampling uses
original106299subset rows and maps back to original IDs.

25focused CPU tests pass.326offline shader variants validate/lower, including
newINT8 kernels and CJK argmax/probability/map/sampler/bridge variants. Release
artifact source83035de and hash are in the completion receipt. These CPU checks
do not prove native startup, full numerical output/state, complete erasure or
performance. Whole rewrite remains incomplete; no speed trials.

Next: GDN/QSA armed transaction preparation separate from math enqueue, typed
invocation positions/epoch controls, then direct graph launch retaining actual
resources/fences and original final proofs. Existing graph reuse still rebuilds
enqueue calls. PLE source/history staging already precedes target capture.
Keep the explicit memory-placement/configuration and commit/event differences
visible during the final source audit. Required Vashti changes:none.

No GPU/service/sudo operations this pass. Last recovered runtime snapshot is
Oct15historical evidence, not a current live check. Renew GPU authorization before
protected completion/numerical tests; only then consider the speed gate after
all source requirements pass.


### Active Kuno engine handoff — attention transactions before graph capture (2026-10-16)

Source`f965f63` connects original GDN/QSA armed-transaction preparation before
capture for all48 core attention stages. At each layer, real math enqueue consumes
its exact prepared ticket and returns the original pending receipt. Existing
anchors/generations, full owner/span/fault identities, cancellation/Drop poison
and whole-model acceptance checks remain. No new GPU roots. Additional fixed
metadata arrays are explicitly priced under the admitted host cap. Old enqueue
entry points compose these phases; no ordinary math schedule is replaced.

38focused CPU tests pass across GDN/QSA state policies, prepared Drop, selected
metadata capacity and native branch layout. Strict locked release core/off
Clippy, formatting and whitespace pass. See completion guide Attention preparation
CPU receipts for exact logs and limits. No GPU/service/sudo work occurred; old
physical recovery snapshot remains historical. Required Vashti changes:none.

Continue typed invocation parameters and direct retained graph launch, preserving
actual GR/shared/MoE/replay metadata and original completion/commit proofs. This
is connected preparation, not complete no-enqueue execution. Numerical/hardware
and speed acceptance remain pending. Goal active, CPU-only.


### Active Kuno engine handoff — typed target invocation parameters (2026-10-16)

Source`e91677a` supplies actual target position/global route epoch outside branch
keys. Native QSA INFO declares its position field; route/reduce declare epoch.
Retained records bind exact schema/static template/original uniform owner+offset.
Dynamic words are zeroed in CPU templates and actual upload bytes render from
fresh controls through write-only mapping. Stale values or changed constants/
schema/binding fail; no value-search heuristic or mapped-read bypass. The QSA
positions stamp now uses this native INFO path. No GPU root added.

28focused CPU tests,327offline shader variants, locked release build and strict
core/off Clippy pass. Full details and immutable artifact hash are in completion
guide **Typed target invocation parameters**. No GPU/service/sudo operations or
private input tests ran. No whole-engine numeric/performance/erasure acceptance.

Current calls still enqueue math. Continue direct target graph submission with
all original logical consumer/replay state and output/fault proofs, then MTP
round/step typed controls and direct replay. MTP remains in earlier untyped mode;
target invocation controls reject MTP keys, preventing false completion claims.
No application API change; required Vashti changes:none. Goal active, CPU-only.

### Active Kuno engine handoff — direct target graph replay (2026-10-16)

Engine `d40442c`, same isolated checkout. This supersedes the earlier statement
that every repeated target window must traverse/enqueue all its math. Source now
bulk-renders the actual typed INFO program and submits the original retained native
recording after real upload and queue-fence completion. The selected target loop
uses this automatically for eligible cache hits, consuming the original prepared
GDN/QSA/PLE transactions and preserving all output/fault/commit/erasure checks.

The private CubeCL runtime contract adds `replay_native_strata_branch`: false only
for an unavailable/unsupported recording before replay mutation; true only after
actual completion; errors retire the request rather than falling back. It is not a
wire API, state acceptance or erasure receipt. Original request nonce, static key,
state role, resource inventory, INFO owner/offset, accounting and full cleanup stay
required. The verification usage flag is refreshed alongside position/route epoch.
No separate image/text authentication, connector or frontend changes.

Limits and exact receipts are in completion guide **Direct retained target graph
submission**.52 focused CPU tests,327 offline shader variants, immutable release
build and strict selected/off Clippy pass. No full-model
or speed claim. MTP is still in its old untyped mode. Its first input staging and
observation boundaries must be separated before adding equivalent direct replay;
see the current native text checkpoint for exact entry points and dependencies.

Work and the active goal are CPU-only. No GPU/service/sudo operations or installation.
Required Vashti changes:none. Existing unrelated files and canonical checkout remain
untouched. Do not run throughput comparisons until the whole rewrite gate passes.

### Active Kuno engine handoff — MTP input staging and observation (2026-10-16)

Engine `862f5fa` connects original first-front TOKEN/borrowed target-residual
staging before capture, original front math through an opaque prepared ticket,
and actual packet/fault readback after closing the producer graph. Positive
confidence thresholds still observe each actual proposal before a followup.
No borrowed target source is released early and no synthetic forward is published.

Coupled draft controls upload once per round into the original32KiB owner;
per-step4KiB aligned views feed the unchanged actual GPU bridge/sampler. There is
no per-draft CONTROL upload inside capture. Original main/runtime/host caps and
complete-owner erase remain; no new GPU root or public API is introduced.

26 focused CPU tests, selected/off strict Clippy, immutable release build and327
offline shader variants pass. Exact receipts/limitations are in **MTP input staging and
actual observation boundaries**. MTP parameters are still untyped and math is still
enqueued. Next work is explicit phase/position schemas and real direct submission;
current checkpoint records the required coupled state/sampler bookkeeping.

Goal remains active for CPU-only work. No GPU/model/service/sudo operations,
installation or live speed trials. Required Vashti changes:none.

### Active Kuno engine handoff — typed MTP phases and model ownership (2026-10-16)

Source `8262367` connects actual MTP math to typed round/step controls, explicit
front/rest/follow-up phases, separate4096-capable cache boundaries and original
model-instance recording keys. Runtime resolves declared phase fields to static
window offsets before capture; it never guesses fields by comparing old values.
The original single shared client/model resources and request nonce stay required.
The existing staging/actual-observation separation in `862f5fa` remains selected.

67 focused CPU cases, strict selected/off Clippy, immutable release build and327
offline shader variants pass. Exact receipts are in completion guide **Typed MTP phases and original
model namespace**. **MTP direct submission and original host-accounting handback
are still missing**; current queued math is preserved. Next source entry points
and required state/sampler bookkeeping are in the current text checkpoint.

Private CubeCL control declarations only; no public API, connector, UI or auth
change. No GPU/model/service/sudo action, install or speed trial. Goal remains
active for CPU-only work. Required Vashti changes:none.


### Active Kuno engine handoff — direct native MTP submission (2026-10-16)

Source `d0670b0` connects retained native MTP replay for whole chains and independent
confidence-stopped steps. Capture-derived bounded accounting and sampler CPU
handback occur only after actual submitted/joined work; original proposal/fault/
padding/source/ledger observation and final forward publication remain required.
Coupled/greedy inventories have separate shape/model keys and original request
nonce enforcement. No tokens, prompts or other inference content enter the
accounting cache. Cold/unsupported recordings retain original enqueue/capture;
failed submission/join is terminal.

73 focused CPU checks, strict selected/off release Clippy, immutable release build
and327 offline shader variants pass. Exact source/artifact hash, limitations and
logs are in completion guide **Direct native MTP submission and original host
handback**. No physical/model/numerical/speed acceptance claim.

Next coupled source block is accepted-prefix asynchronous commit and real independent
compute queues. The current backend joins the previous submission before another;
wait=false alone cannot reproduce upstream target-commit/MTP overlap. Current
checkpoint records the actual source calls, original owner/queue requirements and
remaining embedding/context/prefill/prompt/final-acceptance substitutions. Do not
benchmark or undo connected components based on incidental timings.

Goal is active for CPU-only work. No GPU/model/service/sudo changes, installation,
release or public API changes this pass. Required Vashti changes:none.


### Active Kuno engine handoff — dedicated MTP compute queue (2026-10-16)

Source `b6732c8` requests/adopts an actual independent compute family and routes
native MTP branch capture/replay to it, with family-correct command pools/recording
reuse, concurrent original buffers/raw host aliases and real cross-queue timeline
memory handoffs. Device creation selects by actual capabilities/counts; it never
assumes queue1 exists in the main family. Saved B70 Oct14 properties establish a
compatible topology, not current physical execution acceptance.

40 focused CPU cases, strict selected/off Clippy, immutable release build and327
offline shader variants pass; exact hash/logs are in completion guide **Dedicated native
MTP compute queue and memory handoffs**. No GPU/numerical/speed/recovery claim.

The current graph still has one in-flight fence and joins before switching. Next
coupled dependencies are native INFO copy/owner scheduling plus pending accepted-
prefix commit/events and original consumer/final joins. In particular, fresh
Upload owners may differ from the Uniform's previous source: a retained copy must
not silently point at stale data. Current checkpoint records actual next entry
points and remaining substitutions. Unkeyed MTP preparation/prefill is still main-
queue work. No live speed testing before the whole connected gate passes.

No installed-service/model/sudo action, public API/connector/UI change, deployment
or release. Goal remains active for CPU-only work. Required Vashti changes:none.


### Active Kuno engine handoff — native INFO transfers (2026-10-16)

Source `414e2c3` connects actual current full Upload-to-Uniform copies to the native
transfer timeline for cold execution and retained math replay. Current source
handles are retained and checked, rather than assuming previous Upload allocations.
The original compute header waits on real copy completion; INFO-only drain preserves
later native/Wgpu visibility. Original mapping, ledger, page/schema and terminal
wipe obligations remain. Native-only flush/replay avoids an empty primary-queue
submission/poll; actual encoder mutations/real primary work are still tracked and
joined. Additional bounded copy metadata is charged2MiB through the original host
admission quote. No inference content is persisted.

43 focused CPU cases, final selected/off strict Clippy, immutable release build and
327 offline shader variants pass; exact artifact/hash/logs are in completion guide **Native INFO
transfer with actual current owners**. Helper tests do not prove physical GPU copy,
visibility, numerical or speed acceptance.

Next is the actual pending accepted-prefix commit/family-state and per-flight queue
ownership block. The graph still joins previous math globally; no commit/draft
overlap is claimed. Preserve completed-replay authority, real INFO-copy timeline
edges and exact verified carry ownership while changing that schedule coherently.
Current checkpoint records original source/consumer entry points and other gaps.

Goal remains active for CPU-only work. No GPU/model/service/sudo operation, public
API/connector/UI change, installation or release. Required Vashti changes:none.


### Active Kuno engine handoff — native lane flight ownership (2026-10-16)

Source `86f4c35` connects all actual capture/hit/direct submissions to separately
owned target/draft fences and stable original recording identities. Slots reserve
before reset/submit; partial work never completes. Real matching-fence success is
required before host completion counters clear. Live recordings cannot be replayed,
replaced or evicted; actual resources remain retained. Original output/fault/
family/erasure proof requirements and completed-replay semantics are unchanged.

46 focused CPU cases, final selected/off strict Clippy, immutable release build and
327 offline shader variants pass; exact hash/logs/scope are in completion guide
**Native lane fences and original recording flight ownership**. No numerical/GPU/
parallel execution or throughput acceptance claim.

Global drains still join both lanes. The next coupled block is original accepted-
prefix family-state/permit publication and exact carry/dependency consumers; merely
removing the waits would be incorrect. Current checkpoint identifies the borrowed
GDN/QSA/PLE lifetime constraint, dynamic commit controls and remaining source gaps.

Goal stays active for CPU-only work. No GPU/model/service/sudo operation, public
API/connector/UI change, installation or release. Required Vashti changes:none.


### Active Kuno engine handoff — GPU PLE history (2026-10-16)

Engine source through `91e1f9e` (includes `595393e`) keeps native PLE float history
in the existing original GPU OLD/NEXT owners across prompt, decode and prefix
repair. Default float readbacks/uploads and the prompt diagnostic tail copy are
removed together; host publication advances only the two original ngram token IDs.
Explicit diagnostics still read real data, while the feature-off CPU oracle stays.
Native checkpoints retain token/control metadata; rejection uses original GPU
prewindow/normalized owners and the real joint fault fence. Actual target capture
keys distinguish independent GDN/PLE roles. No public API or private GPU root is
added. Never reupload the now-stale legacy CPU convolution mirror in native decode.

61 focused native CPU checks and9 core transition cases, strict selected/off
Clippy and formatting pass. Immutable artifact/offline shader evidence is in
**Native GPU PLE history and real prefix consumers** in the completion guide.
No GPU/numerical/speed acceptance claim. Installed runtime/services are untouched;
last physical recovery evidence remains historical Oct15.

Next is the actual accepted-prefix commit block: source dynamic n_keep, count1
self-commit and multirow commit together, original verified carry ownership,
pending state events, draft dependencies and final/cleanup joins. Current Rust
still commits the full proposal before prefix repair and joins globally. The PLE
normalized-row recomputation/role swap differs from source indexed snapshots;
that and GDN state roles require explicit resolution in this block/final audit.
Do not remove waits or abandon coupled components based on incidental speed.

Goal remains active for CPU-only work. No GPU/model/service/sudo, installation,
deployment, release or API/connector/UI changes. Required Vashti changes:none.


### Active Kuno engine handoff — direct accepted-prefix (2026-10-16)

Engine source `131d29c` connects the selected native controller to real provisional
target rows while original full-window state transactions remain open. It forms
accepted GDN/QSA/PLE candidate state, accepts the original fault/I/O proof, then
publishes only n. The selected controller no longer commits a full proposal before
rewinding it. Original receipt counts/epochs stay unchanged. Count1 uses its real
candidate; all-accept multirow replays state as upstream does. Exact device carry
is retained, the host row lease drops before state work, and adaptation follows
the completed boundary. Original diagnostic checkpoint/repair paths stay separate.
No new GPU state owner or public API is added; runtime INFO coexistence is charged
544,768 extra bytes through original admission.

102 nonempty focused CPU cases, strict selected/off Clippy and formatting pass.
Exact immutable source/hash, corrected test-filter scope and shader evidence are
in completion guide **Native accepted-prefix before state publication**. These are
not whole-model GPU/numerical acceptance. No live speed trial or recovery claim.

Next is actual retained dynamic commit-graph/pending-event fidelity and independent
draft work. The new path still joins synchronously. Preserve real INFO/resource
owners, verified carry, family proof/poison obligations, target ordering and
session/final/cleanup joins together. A wait=false flag alone is insufficient.
GDN/PLE role swaps, QSA raw-index storage and PLE normalized-row reconstruction
remain explicit backend substitutions for final review.

Goal stays active for CPU-only work. Installed services/runtime/models remain
untouched; last physical recovery evidence is historical Oct15. No sudo, install,
deployment, release, API/connector/UI changes. Required Vashti changes:none.


### Active Kuno engine handoff — retained commit graph (2026-10-16)

Engine source `4083258` connects the actual selected prefix consumer to a retained
multirow commit graph. Every cold/hit path rechecks original armed family/input
receipts and exact proposed extents. Model-instance/proposed-T/context/root-role
keys exclude accepted n and position; those are explicit typed INFO controls.
The fixed133-command body retains all rejection consumers even when captured
with all proposals accepted. Count1 uses its original candidate without a graph.
Actual native finite/zero folds and subsequent original joint fault/I/O/family
proof remain. No new GPU state root, runtime budget or public API is added.

104 focused CPU cases, strict selected/off Clippy, immutable release and330
shader variants pass. Exact source/hash/logs are in completion guide **Retained
native commit graph and explicit accepted count**. These establish source/CPU/
offline integration, not GPU numerical/visibility acceptance or speed.

The current submission APIs still join. Next is the real pending event and draft
consumer together with INFO/queue hazards. Global stream drains, copy completion,
live uniform destinations and queue-switch signals must be handled coherently.
MTP preparation (output clear, TOKEN/carry staging) precedes named-branch begin;
it also needs the independent lane and a verified-row dependency established
before commit, rather than a main signal queued behind commit. Retain original
family/carry/flight owners through dependent proof and final/cleanup joins.
Do not claim overlap from wait=false or a host flag.

Goal stays active for CPU-only work. No GPU/model/sudo/service, installation,
deployment, release or API/connector/UI action. Last physical recovery evidence
remains historical Oct15. Required Vashti changes:none.


### Active Kuno engine handoff — pending commit ownership (2026-10-16)

Engine source `91e1f9e` adds actual cold/replay pending-commit APIs and exact
original graph/scope/submission joins. Unique submission epochs distinguish
replays of the same recording; ordinary drains retain completion until ack.
Partial, stale, duplicate and foreign identities cannot authorize a join. Real
matching-lane fence success remains required; old joined-bool APIs retain their
meaning. Actual INFO wait epochs are credited only by their own math-fence join,
preventing a false INFO-only dependency or credit to a newer unrelated copy.
Original resources remain retained and cleanup cannot forget unknown work.

The selected engine consumes this pending receipt but explicitly joins it before
its original fault/I/O/family acceptance. No independent draft overlap yet. Next
is verified-row readiness before commit plus original MTP staging/math on the
independent lane, coordinated with whole INFO-copy destinations, graph/stream
hazards and final joins. Existing global guards must not simply be removed.
Full-owner INFO copies can overlap even when record offsets do not.

107 focused CPU cases, strict selected/off Clippy, immutable release and330
offline shader variants pass. Exact source/hash/logs, limitations and new internal
runtime method names are in completion guide **Original pending commit submissions
and exact joins**. GPU numerical/visibility/completion/recovery remains unverified.
No public HTTP/connector/UI contract changed; required Vashti changes:none.

Goal remains active for CPU-only work. No GPU/model/sudo/service operation,
installation, deployment or release. Last physical recovery evidence remains
historical Oct15; installed services/runtime/models were untouched.


### Active Kuno engine handoff — scoped verified-row drafting (2026-10-16)

Engine source `1f46f76`, followed by delivery source `0bf0b5f`, connects the next actual MTP producer inside the original
pending-prefix interval. Its limited immutable R grant authenticates original
row generation, graph/arena root and full proposed/accepted extents. It cannot
publish target state or grant erasure. Real copy/consumer completion remains
mandatory. The controller carries the completed original next producer forward,
including actual cursor and confidence/first-candidate metadata. The prompt-only
initial seed keeps its original committed borrow. EOS/cap skips next drafting;
request-local sampling commits accepted inputs exactly once before coupled
next-round controls. Original target join/fault/I/O/family proof still precedes
state publication. Delivery source `0bf0b5f` emits genuine verified predictions
after commit submission and before next drafting, matching pinned serving. A later
draft/commit failure terminates the stream without granting state readiness or
request success; original delivery errors survive terminal cleanup.

This is connected source progress, **not physical overlap or parity**. Global
waits still serialize execution. Verified delivery now precedes next drafting. Next:
establish original R readiness before commit,
move all MTP staging to the independent lane, and resolve whole INFO-copy hazards,
shared MAIN completion/retirement credit and final/failure joins together.
Keep the whole rewrite incomplete until the source and physical gates pass.

113 focused CPU cases, selected/off strict Clippy, immutable final source
`0bf0b5f` and330 offline shader variants pass. See completion guide
**Scoped verified-row draft consumer** for exact hashes/logs and
remaining substitutions. No GPU/model/sudo/service operation or installation/
release. Public API/connector/UI unchanged; required Vashti changes:none.
Goal stays active for CPU-only work; historical Oct15 recovery is not current
physical health evidence.


### Active Kuno engine handoff — pending retirement accounting (2026-10-16)

Source `543bc73` binds each original pending multirow target commit to its shared
MAIN retirement hold. The scoped original MTP consumer cannot reclaim its peer's
pending allocation allowance. Real telemetry still checks limits/unexpected growth.
Only the guard's original runtime join can release the hold, and release alone
cannot refund MAIN: a later real checked completion boundary remains required.
Count1 has no additional commit/hold. The guard is private, not cloneable/sendable,
and grants no completed model state or erasure. Unknown Drop atomically retires
original context/sequence before releasing its retained client, without taking
admission's mutex or releasing debt. Existing close still attempts actual drains.

152 focused CPU cases and selected/off strict Clippy pass; formatting/whitespace
pass. Exact logs/limitations are in completion guide **Original pending commit
retirement**. No new shader/GPU owner and no fresh immutable release artifact;
latest frozen `0bf0b5f` predates this block. Build current source before physical
execution. Source/fence/numerical acceptance and whole rewrite remain incomplete.
Next is original recording access inventory and whole INFO-copy hazards, R
readiness before commit and all MTP preparation on its independent lane. Shared
I/O recycle still globally flushes. Do not remove waits based on this hold alone.

Goal stays active for CPU-only work. No GPU/model/sudo/service/install/release or
public API/connector/UI change. Required Vashti changes:none.


### Active Kuno engine handoff — recording/INFO ownership (2026-10-16)

Source `a8e538f` retains actual original descriptor/copy/physical/indirect access
ranges with each recording. Canonicalization preserves real gaps, shared readers
and historical writes. Capacity remains charged inside the original64MiB retained
metadata bound. Actual whole-owner INFO copy accesses preflight both original
flights and join only conflicting matching-lane fences; partial/missing/foreign
owners fail. INFO command-pool reuse waits on its own copy. Final visibility is
decided after actual math joins, so older math cannot credit a newer INFO epoch.
Runtime exports, pending MAIN retention and original close remain mandatory.

156 focused CPU tests and strict selected/off Clippy pass. See completion guide
**Original recording access and INFO copy ownership** for exact source/logs and
limitations, and its **Next connected scope design** before implementation.
Global stream/graph waits still prevent physical independent execution. No new
shader/layout or immutable artifact; latest frozen `0bf0b5f` is stale for current
source. Build current source before physical acceptance. No new speed result.

Public APIs/connector/UI unchanged; required Vashti changes:none. Goal remains
active for CPU-only work. User requested keeping sudo refreshed; noninteractive
validation succeeded in the shared Demons shell. No GPU/model/service/install/
release or protected inference/driver operation ran.


### Active Kuno engine handoff — original R readiness and draft scope (2026-10-16)

Sources `d2d3228` and terminal-drain follow-up `dcd783b` connect real precommit
MAIN->draft readiness and an original request-owned scope around the entire
Pending observer. The actual current R
resource AND allocator binding stay pinned/read-only through completion, including
partial dependencies. The original graph/request/monotonic generation validates
prepare/entry/end; the existing carry-generation/window/graph receipt remains
mandatory. Sampler controls, scratch/token/R-copy/front and named MTP branches
keep one draft lane, including unkeyed intervals. Original recorded/copy/INFO
ranges and direct replay plans reject writes to the held source. No new data root
or content persistence. Actual completed end restores MAIN visibility, then
releases the source. Unknown guard/Graph destruction retains or retires original
owners, never declares them joined/ready/clean. Original target exported join,
shared MAIN retention and family/fault/PLE/I/O/erasure proofs stay authoritative.

This is NOT physical independent execution: entry and push/execute/replay/flush/
metadata/readback/reuse still globally/primarily serialize, potentially joining
target before verified delivery. Next connect exact current-lane/conflicting-flight
admission, original INFO SOURCE remap authority, confidence packet copy/map and
selective upload/readback reuse together. Preserve queued-versus-submitted work,
fresh Wgpu initialization and positive-confidence per-cell observations. See the
completion guide **Original precommit R readiness and draft lane scope** for
internal method names, exact source/logs and remaining scope/full rewrite gates.

163 focused CPU cases across19 nonempty suites, selected/off strict Clippy and
format/whitespace pass. No shader/layout change or new immutable ordinary
artifact. Frozen `0bf0b5f` is stale for current source; rebuild before protected
execution. No GPU/model/service/protected inference/driver/install/release or
public API/connector/UI change. Required Vashti changes:none. Sudo timestamp
refresh succeeded in Demons. Goal remains active for CPU-only work; historical
Oct15 recovery is not new physical health evidence.


### Active Kuno engine handoff — native packet readback (2026-10-16)

Source `7ad47ca` connects selected native private packet observations to actual
full-owner fill/copy recordings, original selected compute/copy plan and final
fence, explicit HOST_READ availability and guarded original Vulkan HAL mapping.
Full-range non-coherent invalidation, one batch deadline, original I/O ledger/
lease and wiping logical-byte controller remain required. No Wgpu primary packet
copy/map callback/poll is requested; ordinary and terminal cleanup paths remain.
Packet I/O is unnamed and cannot replace a named math shape. Actual cold source
initialization is preserved; no new persistent content or GPU data owner.

165 focused CPU cases, selected/off strict Clippy and format/whitespace pass.
See completion guide **Original native packet readback** for exact source/logs,
internal ownership flow and still-unverified Vulkan map/barrier/failure/erasure.
Current graph/scope/flush global waits and shared I/O reuse still serialize.
Next connect current-lane/conflicting-flight admission and selective original
source reuse/remap authority together, including positive-confidence per-cell
checkpoints. Preserve original readiness, INFO epochs, exports, MAIN debt and
terminal drains. This is not physical overlap, parity or a speed result.

No new shader/layout or immutable ordinary artifact; frozen `0bf0b5f` is stale.
No GPU/model/service/protected inference/driver/install/release or public API/
connector/UI change. Required Vashti changes:none. Sudo refresh alone succeeded
in Demons. Goal remains active for CPU-only work; full rewrite gates remain open.


### Active Kuno engine handoff — native upload and joined reuse (2026-10-16)

Source `7cb1cbc` connects actual data/INFO/replay producers to bounded write-only
uploads, explicit Wgpu/native-ready/writing/staged ownership, guarded actual
native map/full flush/unmap and original copy sealing. Only a real all-owner
stream/graph completion grants checkpoint reuse; partial/unsubmitted writers,
held leases/INFO, wrong requests and GPU errors cannot publish free owners.
Original I/O ledger/pins/erase, exports and MAIN debt stay owned. Selected native
reuse no longer requests Wgpu upload map callbacks or primary transitions/
Readback clears. The original native packet fill/copy/fence/map path still applies.
No new GPU data root, content persistence, public API/connector/UI change.

276 focused CPU cases (including original I/O/cache/parameter owner tests), strict
selected/off Clippy and formatting/whitespace pass. The CPU harness disables
activation and uses lab/generic budget cfg; it is not real GPU/fence/map/selected-
limit/erasure proof. Exact source/logs/failed dependency-test attempts and remaining
requirements are in completion guide **Original native upload and joined I/O reuse**.
No shader/layout or frozen artifact change; frozen `0bf0b5f` is stale.

This remains globally joined. Next couple original-owner selective reuse/exclusion
with current-lane/conflicting-flight graph/stream admission. Never grant all-owner
reuse after a peer fence; inspect actual INFO/source/descriptor and future retained
replay pins. Preserve epochs/visibility, cold initialization, exports/MAIN debt and
terminal drains, including positive-confidence per-cell checkpoints. No physical
independence/parity/speed result. No GPU/model/service/protected inference/driver/
install/release action. Required Vashti changes:none. Sudo refresh succeeded in
Demons; goal stays active for CPU-only work and the full rewrite remains incomplete.


### Active Kuno engine handoff — selective I/O owners (2026-10-16)

Source `7477b66` binds selective native reuse to the same checkpoint's original
full I/O owners/request/epoch/slots. Graph validates original recording/flight
ownership and actual scoped-lane/INFO copy/readiness/handoff completion; named
future replay references AND live anonymous references exclude their actual
buffers. Only admitted slots gain native writers/index entries; excluded unused
grants revoke, partial/mapped/staged excluded uploads reject and INFO pins stay.
No source content, new GPU root, owning release or fake completion/erasure receipt.
Temporary bounded metadata adds4MiB HOST allowance, no VRAM/model data.

281 focused CPU cases, strict selected/off Clippy and format/whitespace pass.
Exact logs, metadata bound, CPU-harness limitations and required next dependencies
are in completion guide **Original selective I/O owners and retained replay pins**
and **Next connected graph admission and lifetime block**. Current stream/graph
waits STILL serialize; no physical independence/parity/speed result. Next couple
current-lane/conflict admission and queued-vs-flight dispatch with real INFO/dirty-
primary visibility and deferred managed allocator cleanup through joint scope end.
Keep original exports/MAIN debt/R loan/family proof and final/retired drains.

No shader/layout/frozen ordinary artifact/GPU/model/service/protected driver/
inference/install/release or public API/connector/UI change. Frozen `0bf0b5f`
is stale. Required Vashti changes:none. Sudo refresh succeeded in Demons; goal
stays active for CPU-only work. Full rewrite and physical gates remain incomplete.


### Active Kuno engine handoff — scoped admission and allocator lifetimes (2026-10-16)

Source `f9d1e06` completes the coupled admission/lifetime block identified in the
previous handoff: queued nodes vs live peers; desired-lane draft entry using the
original precommit readiness; actual node/INFO/retained-plan hazards with matched
flight joins and real queue handoffs; stable recording identity; scoped native
flush/readback/reuse. Dirty primary/cold initialization, ordinary sync, scope end
and terminal cleanup preserve actual global drains and MAIN visibility.

Original managed bindings are pinned from before preparation through actual joint
scope end, covering both new and resolved allocations. Allocator cleanup/uniform
release defers centrally, including explicit server cleanup. Unknown/failure
retains pins. INFO source-copy completion preserves the still-owed math timeline
wait; it is not a visibility/target-ready credit. Original exports, pending MAIN
debt, R source loan/generation, I/O ledger, fault/family proof and state publication
remain distinct. Adds4MiB bounded HOST bookkeeping (queue134 ->138MiB), no GPU root.

Final locked selected build,283 focused CPU cases/24 nonempty suites, strict
selected/off Clippy and format/whitespace pass. Exact logs/limits and next audit:
completion guide **Scoped native admission and original allocation lifetimes**.
No device execution, speed result, numerical/physical/recovery acceptance or
complete-rewrite claim. Ordinary frozen `0bf0b5f` is stale; rebuild current source
before protected work. No shader/layout/public API/connector/UI change.
Required Vashti changes:none. No service/model/GPU/install/release action.
Sudo refresh succeeds in Demons. Goal stays active for CPU-only work.


### Active Kuno engine handoff — mapped HOST embedding (2026-10-16)

Source `89da25f` moves the shared native IQ4_XS embedding into the original HOST
owner/sole readonly registry. All16 full original pages use the existing actual
nonlocal allocation/import/flush and exact detach/zero/free lifecycle. The same
source pages feed target and MTP GPU gathers. Global source construction now
precedes sharing; shared reader loans must release before HOST detach. Unknown/
live readers refuse release. No duplicate model source or request persistence.

Quotes distinguish head-only MAIN521474048 bytes, HOST embedding337715200 bytes
and shared addressable859189248 bytes. MTP excludes BOTH borrowed tensors from
its extra MAIN allocation. HOST quote includes all embedding allocation envelopes
and registry slots. GPU gather tiling is unchanged and still needs final review;
static dense capacity remains5GiB, so no larger expert cache/speed is claimed.

301 focused CPU cases/29 nonempty suites, final selected build, strict selected/
off Clippy and format/whitespace pass. Exact receipts and next source/physical/
substitution gates: completion guide **Shared native embedding in original HOST
backing**. No GPU/model/service/protected inference, install or release action.
No shader/layout or public API/connector/UI change; required Vashti changes:none.
Sudo refresh succeeds. Goal stays active for CPU-only work; rewrite incomplete.

Frozen ordinary selected release builds at source89da25f in
`/media/jmicko/faststore/kuno-development/strata-mapped-embedding-ordinary-20261016/`,
SHA2566016128a10833a9ef2448209e25f43afe2dac4e4cfbb7c895fa7f43f44fb840c.
No binary/device/model execution occurred. Use this current artifact instead of
stale0bf0b5f when protected correctness work is authorized; no speed trials yet.


### Active Kuno engine handoff — 4096-row prompt capacity (2026-10-16)

Source `bcc30dd` connects4096-row prompt capacity to actual full scratch owners,
router/grouped packets and native shader ABI. Three512MiB roots become three1GiB
roots;128MiB working images, larger GU/PART/maps/status and matching HOST/runtime
quotes are priced before allocation. Private allocator cap/pool bins, MAIN page
admission, actual device-limit checks and complete erasure bounds move together.
This conservatively adds1.5GiB MAIN scratch, with exact upstream aliasing/pricing
still open. No observed free-memory credit or speed result. MAX8 is unchanged.

320 focused CPU cases/33 nonempty suites, strict selected/off lints, normal frozen
build and330 offline shader validations pass. An additional legacy budget test
fails at an unchanged16K expected/core32K scratch fixture BEFORE its runtime-cap
assertion. Full regressions and actual runtime-cap/device execution acceptance
remain open. Exact failure/receipts: completion guide **Connected4096-row prompt
capacity**. The unchanged4096 context still prevents a full4096 prefix plus decode;
32768 context and general request entry remain next connected dependencies.

Frozen sourcebcc30dd: `strata-prompt4096-ordinary-20261016/`,
SHA256d702405db8e7d9d6195af21b115a851194176e12a2bb95c1ef692cfa9560b93d.
Supersedes89da25f. Shader source changes are internal; no public API/connector/UI
change, required Vashti changes:none. No GPU/model/service/protected inference/
install/release action. Sudo refresh succeeds; goal active for CPU-only work.

Follow-up source `669437d` fixes the native dispatch inventory: actual prompt/
erasure grids require65535 groups along X, so the old16384 guard was insufficient.
Legacy mode retains its old requirement. Exact-one-below inventory tests, all320
focused cases and selected/off strict lints pass again. Shader sources/ABI are
unchanged from bcc30dd's330 offline validations. Use the latest ordinary artifact
recorded in the completion receipt for future authorized protected work.

Latest normal frozen source669437d builds successfully:
`strata-prompt4096-grid-ordinary-20261016/`,
SHA256b6febebaad8e980ce3dfde92ff0094b5beb4bc53c0d0a1da6d64305667c56b0f.
Supersedes the pre-guard bcc30dd artifact. No binary device/model execution;
330 offline validations refer to bcc30dd's identical unchanged shader sources.


## Active Kuno engine handoff — 32768-cell context capacity (2026-10-16)

Isolated `codex/strata-rust-port`, source `8d301b1c1ec8dea8972dddb05ac62b189bac1ecb`.
Read [connected context receipt](kuno-strata-rewrite-completion.md#connected-32768-cell-context-capacity-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). This supersedes the4096-context
limitation in the earlier prompt handoff; the whole Rust rewrite remains incomplete.

The selected runtime now connects32768 context across actual target/GDN/QSA and
MTP owners, positions/RoPE, pooled selection, prompt chunks, routing/result guards,
request coordinates, typed retained graph controls and all full-page quotes.
Reference geometry stays unchanged; MAX8 and4096-row prompt chunks stay separate.
MTP cache guard/rewind use genuine65535x2 linear dispatches. Suffix history covers
the full selected context within the existing4MiB HOST budget. Private MTP64 ->128MiB,
target/language runtime envelopes256 ->512MiB; actual tracked I/O charges and
constructor equality/ownership gates remain required.

Final CPU receipt:371 distinct cases/40 nonempty suites, both locked strict lint
configurations, Rust formatting/whitespace. Normal frozen artifact and offline
shader receipt are in the completion document. These establish source/CPU checks,
not GPU correctness, physical limits, throughput or installed behavior. Previous
old16K-vs-core32K legacy fixture failure is still an outstanding regression; full
suite/device acceptance is not claimed.

The user authorizes CPU-only continuation and Demons sudo timestamp refresh.
No GPU/model/service/install/release work occurred. Last actual physical recovery
evidence remains historical Oct15. Keep working on arbitrary transient request
entry, exact dense/prompt pricing and the source scheduling/lifetime/substitution
review; GPU/numerical/actual recovery work needs renewed authority. No speed trial
until the whole gate passes. No API/config/connector/UI change; required Vashti
changes:none. Preserve the original shared checkout and its coordination edits.


## Active Kuno engine handoff — transient native request owner (2026-10-16)

Isolated `codex/strata-rust-port`, source `dc7ace7823c860b7237535cfb159f3bca2dab687`.
Read [request-owner receipt](kuno-strata-rewrite-completion.md#connected-transient-native-request-owner-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). The public native diagnostic now
calls the actual `strata_engine::execute`; old Intel-specific branches after that
early return are not its selected path. The controller test namespace moved from
`daslab_language_probe::global_controller` to `strata_engine::controller`; never
count an old filter returning zero cases as a pass.

Owned raw-token requests carry bounded sampler/speculation controls and can use
the remaining32K context. The same original target/MTP constructor, shared source
ownership, request I/O and ordered close/recovery consumers are connected. The
engine does not serialize/log/persist inputs or outputs; the caller receives
output tokens via a callback and content-free completion after checked cleanup.
Public fixture preview/receipt handling stays in the diagnostic, with its existing
memory preflight, protected process/source obligations and512-output budget.

374 distinct CPU cases/41 nonempty suites, both strict lint configurations,
format/whitespace pass; frozen artifact/offline shader evidence is in the receipt.
No GPU/model/service/install/release work. Sudo refresh succeeds. Goal active
CPU-only; no current device-health or performance claim.

The current executable caller is still compiled-public. A protected arbitrary
prompt/tokenization caller and verified public-weight/runtime reuse across requests
remain required before general/private or hot multi-request support is claimed.
The unsafe internal entry relies on real caller authentication/protection/admission;
those are not waived by its tests. Full numerical/physical/failure/recovery checks,
exact pricing, final source audit and legacy regressions remain. No speed tests
until the whole rewrite gate passes. No API/config/connector/UI change; required
Vashti changes:none. Preserve the original shared checkout and coordination edits.

## Active Kuno engine handoff — prompt phase overlay and selection batches (2026-10-16)

Isolated `codex/strata-rust-port`, source `12a60bf0615e303409467bccffc7723ddece1395`.
Read [prompt-overlay receipt](kuno-strata-rewrite-completion.md#connected-prompt-phase-overlay-and-256-query-selection-2026-10-16)
and the [checkpoint](kuno-native-text-checkpoint.md). The connected prompt path
now follows native mutually exclusive attention/MoE scratch lifetimes and256-query
scoring/selection with32-query attention. Persistent residual/GR, paired matrix
images and the raw transfer ring retain their independent lifetimes. Complete
selected rows keep actual aligned parent descriptors and logical row offsets.

Actual prompt MAIN roots `[1GiB,1GiB,32MiB]` total2080MiB,992MiB below the prior
3GiB. Real allocation, source attachment, copy/view bounds, shader ABI and pricing
change together; QSA INFO records are separately increased to7,888,896 bytes.
378 distinct CPU cases/42 nonempty suites and both strict lint configurations
pass. See the receipt for frozen artifact and offline shader checks.

No GPU/model/service/install/API/frontend/release change; required Vashti changes:
none. Noninteractive sudo refresh currently reports password required in Demons;
this does not block CPU-only continuation. HistoricalOct15 recovery is not current
health evidence. Full rewrite incomplete; goal active CPU-only and speed gate
closed. Exact native packed/in-place storage and allocator-granularity/dense
pricing, protected arbitrary-input/tokenization, verified hot request reuse,
final scheduling/substitution/source audit, old regressions and physical/numerical/
failure/recovery acceptance remain open. Preserve canonical application work.

## Active Kuno engine handoff — sealed dense inventory and automatic expert capacity (2026-10-16)

Isolated `codex/strata-rust-port`, source through `910cb58` (`cbf4381` plus
`910cb58`). Read the [dense-inventory receipt](kuno-strata-rewrite-completion.md#connected-sealed-dense-inventory-and-automatic-expert-capacity-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Native construction now prices,
preloads and seals every original dense projection before expert-source attachment;
missing/changed cache entries cannot use the legacy LRU. Actual remaining shared
MAIN sizes the ranked prefix, without the old uniform240-slot request cap.

The original shard receipt covers all784 projections; actual256-byte root pricing
is4,166,315,008 bytes including the one shared head, versus the previous5GiB
allowance. Source/root/peer/private/runtime/HOST budgets stay separately checked.
387 distinct CPU cases/44 nonempty suites, both strict lint configurations and
formatting/whitespace pass. See the receipt for the corrected144-vs-selected test.
No shader changes or new frozen normal artifact;12a60bf is stale for these changes.
A fresh artifact is required before renewed authorized physical acceptance.

No GPU/model/service/install/API/frontend/release change; required Vashti changes:
none. Goal active CPU-only, full rewrite incomplete and speed gate closed. Private
prompt allocator/substitution pricing, arbitrary protected input, hot request reuse,
final source/scheduling audit, old regressions and numerical/physical/recovery gates
remain. Last sudo refresh failed; CPU continuation does not need it. Historical
Oct15 recovery is not current health evidence. Preserve the canonical checkout.

## Active Kuno engine handoff — retained request erasure (2026-10-16)

Isolated `codex/strata-rust-port`, source `4ceb86f104114f9b89293a59641cb5e3cd597e24`.
Read the [retained-erasure receipt and exact next steps](kuno-strata-rewrite-completion.md#connected-retained-targetmtp-request-erasure-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). The actual native entry now clears
completed target/MTP private state while original weights/owners remain loaded,
validates original-owner receipts, then follows its existing terminal cleanup.
The13-token diagnostic permission is preserved separately. Native HOST pricing
adds a dedicated64MiB bounded inspection window; normal MTP packet limits remain.

402 distinct CPU cases/50 nonempty suites, both strict lint configurations and
format/whitespace pass. No new shader, frozen normal artifact, GPU/model/service/
install/API/frontend/release work. Required Vashti changes:none. Old12a60bf artifact
is stale; physical/numerical/recovery acceptance is pending. Goal active CPU-only,
speed gate closed; last sudo refresh failed and root is not needed for CPU work.

This is NOT cross-request model reuse yet. The wrapper still closes after one
request, the language permit remains Closed/single-scope, and erased used owners
must not impersonate generation0/unstarted PLE. Next implement a retained session,
capacity/monotonic request rearming from original receipts, reuse of selected
policies, and private HOST/control clearing; preserve every original owner/epoch/
retirement gate and terminal close order. The linked receipt details these exact
constraints plus remaining whole-rewrite gates. Preserve canonical application work.


## Active Kuno engine handoff — retained loaded sessions (2026-10-16)

Isolated `codex/strata-rust-port`, source `1f4b195560e5b8e7a83361f19c844a7104d11cbf`.
Read the [loaded-session receipt](kuno-strata-rewrite-completion.md#connected-retained-loaded-session-lifecycle-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Actual native entry now uses
`Session::load/run/close`, with original public weights/target/MTP/runtime owners
and checked all-attempt terminal cleanup.32K capacity, bounded request geometry,
monotonic scope rearming from original erasure receipts, distinct retained-ready
PLE and immutable FourGdn/grouped policy reuse are connected in source. The real
controller verifies its issued request scope, preserving owner IDs/epochs/counters
and retirement. Session stays on its native calling thread. No new GPU roots.

404 distinct CPU cases/50 nonempty suites, final locked release test build, both
strict lint configurations and formatting/whitespace pass. Exact logs/limitations
in the linked receipt. Earlier builds in this block are intermediate; final CPU
run follows the checked build's terminal success. No shader/arithmetic change or
new frozen ordinary artifact;12a60bf is stale for current source.

This is NOT device-verified hot reuse. Executable public diagnostics still use
the one-request wrapper; protected arbitrary-input/tokenization and the complete
retained HOST/control audit remain. Existing end-of-request I/O sanitization
clears recordings, parameter templates and binding/header caches while retaining
compiled pipelines/public weights. Full source/scheduling/substitution/pricing,
old regression and numerical/device/failure/recovery gates remain. Whole rewrite
incomplete; no speed trials. Goal stays active CPU-only. Sudo validation still
reports password required in Demons; root is not needed for this work. Historical
Oct15 recovery does not establish current health. No GPU/model/service/install/
release/API/config/frontend change. Required Vashti changes:none. Preserve the
canonical application checkout and inherited untracked files.


## Active Kuno engine handoff — retained controls and sampler (2026-10-16)

Isolated `codex/strata-rust-port`, source `a8bbdf0` plus `f1dd77f6a5a5cf98253488edec1fb761c1e9a0a4`.
Read the [controls/sampler receipt and HOST inventory](kuno-strata-rewrite-completion.md#retained-source-controls-and-full-context-sampler-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). The preceding retained Session
exposed two real correctness dependencies, now connected in source:

- MAIN erasure cleared the route map while actual expert weights stayed swapped.
  Next-request bind now inverts the SAME canonical physical placement, checks the
  existing adaptive map, writes/independently verifies the original full128KiB
  owner with zero padding under real audited I/O and fault before rearming. Never
  reconstruct from the initial profile or reset epochs/rounds. Existing weight
  sources and usage erasure remain; no added model/GPU owner.
- Sampler request positions/consumed history/control/verified-prefix bounds were
  incorrectly tied to the4096 penalty ring. Selected core bookkeeping now reaches
  the actual32K context. The native penalty window, BASE/RING/shader ABI and seven
  root sizes stay unchanged, matching pinned generate.cpp's separate cap.

Live sampler/prompt/PLE source scopes reject retained reset/rearm. The receipt
records actual wiping/local/public/canonical-placement owner classifications,
existing I/O cache/template clearing and exact CPU/source limitations. Source is
not whole-process forensic erasure or device-verified hot inference. Current
executable remains compiled-public/one-request; protected arbitrary input and
physical repeated-request/numerical/failure/recovery checks are next gates.

419 distinct CPU cases/52 nonempty suites, final locked release test build, both
strict lint configurations and formatting/whitespace pass. Final CPU run follows
verified checked-build completion. Exact logs, visibility/lint corrections and
intermediate-vs-final receipts are in the linked completion document. No new
shader/generator/arithmetic or frozen normal artifact;12a60bf remains stale.

No GPU/model/service/install/release/API/config/connector/UI change; required
Vashti changes:none. Goal active CPU-only; full rewrite incomplete and speed gate
closed. Remaining packed/in-place/private allocator/whole-device substitutions,
final pinned source/scheduling audit, old regression and device gates remain.
Last sudo refresh failed; this block needed no root. HistoricalOct15 recovery is
not current health evidence. Preserve canonical application work/untracked files.


## Active Kuno engine handoff — bounded rendered-text input (2026-10-16)

Isolated `codex/strata-rust-port`, source `e1458410b20f956586033b359c57186922d7ad8f`.
Read the [rendered-input receipt](kuno-strata-rewrite-completion.md#connected-bounded-rendered-text-input-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). The actual public native entry
now supplies its compiled rendered fixture through the same bounded wiping UTF-8
reader/Rust-tokenizer consumer into the original Request/engine.1MiB text plus
one overflow sentinel, interrupted/invalid/oversize/I/O bounds and fixed failure
categories; no args/files/logging/receipt/template override. Native tokenizer
capacity now covers general bounded text/context instead of old1KiB/64-token
fixture limits. Ordinary tokenizer limits and compiled public checks remain.

Native original HOST/component/metadata quotes add64MiB transient text/BPE/regex
staging; tokenizer model/cache192MiB and generation/suffix4MiB remain separately
owned. Text/scratch leave scope before generation; no GPU root or weight copy.
421 distinct CPU cases/53 nonempty suites, final locked selected test build,
strict selected/off Clippy and format/whitespace pass. Exact logs and source/CPU
limitations are in the receipt. No new shader/generator/arithmetic or normal
frozen artifact.12a60bf remains stale.

Executable input stays compiled-public/one-request. This is NOT a standalone
private launcher: actual process no-swap/core protection, immutable original
model/tokenizer identity and full exclusive HOST/GPU admission must be verified
BEFORE reading any private bytes. Next connect that independently protected caller
to retained Session runs/output/checked close. Do not widen the public fixture/
receipt allowlist or route private output through public preview/recording.
Remaining full source/scheduling/substitution/pricing/legacy/numerical/device/
recovery gates remain. No GPU/model/service/install/release/API/connector/UI work;
required Vashti changes:none. Goal active CPU-only, full rewrite incomplete and
speed gate closed. Root not needed for this block. Preserve canonical work and
inherited untracked files; last physical recovery stays historicalOct15.


## Active Kuno engine handoff — direct-source memory owners (2026-10-16)

Isolated `codex/strata-rust-port`, source `636c9f3543bbb68a188856719beb0262a033a1aa`. Read the
[full owner/legacy receipt](kuno-strata-rewrite-completion.md#direct-source-memory-owners-and-legacy-regression-reconciliation-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Selected native Intel remains
pinned `DEVICE_PLAN=1,NO_HOST=1`. GU/DOWN directly read actual VRAM/mapped HOST
sources; removed three dormant128MiB fetch roots and unused32KiB address table.
Dense input now owns512KiB in the original arena. Binding schemas, original read
retention, real source/device/queue attachment and pricing change together;
prompt raw-ring/phase owners and actual target/MTP request lifecycle remain.
Private allocation quote reduces383.5MiB; no physical residency/speed claim.

Final selected locked release test build,452 distinct CPU cases/56 nonempty
suites, strict selected/off lint, formatting and whitespace pass. Reproduced nine
legacy failures are stale profile expectations; all31 now pass in selected core,
noncore Strata and original no-feature reference. Production caps/admission remain
unchanged. This is focused coverage, not the full historical broad-core suite.
The linked receipt records intermediate failures and exact logs. Current normal
immutable artifact is `strata-direct-source-ordinary-20261016/`, SHA256
`c0032c4ed9cc065712384deeb624055a7ec1be40af952ee3216986eb6db7b662`; all330 offline
shader variants pass. Clean/stable source build, independent hash/mode0500 and
CPU-only shader validation verified; old12a60bf artifact is superseded.

Remaining gates: packed/in-place scratch/allocator/whole-device pricing, final
pinned source/scheduling/substitution audit and physical/numerical/full-model/
hot-reuse/failure/recovery acceptance. Protected private executable input still
requires independent actual protection/admission before reading content; current
executable remains compiled-public/one-request. No gateway/API/config/frontend
change; required Vashti changes:none. Goal stays active CPU-only, rewrite
incomplete, speed gate closed. No GPU/model/service/install/release action;
last physical recovery stays historicalOct15. Sudo renewal was requested by the
user but actual noninteractive cache validation failed; this block needed no root.
Preserve canonical application work and inherited untracked files.


## Active Kuno engine handoff — native packed prompt GR (2026-10-16)

Isolated `codex/strata-rust-port`, source `37dced95f977428b2be4f616a7e04124bca3953c`. Read the
[exact implementation/validation receipt](kuno-strata-rewrite-completion.md#native-prompt-gr-scales-and-packed-image-consumers-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Pinned Strata's default Intel
prompt GR now uses original four row scales/token and directly emitted BF16
normalized/SiLU images. Actual down/injection/up consumers use these original
views, removing intermediate Pack dispatches. Mix recomputes unrounded F32
values from the actual residual/scales/gamma with original FMA order.

Logical carved payload shrinks82.4375MiB at4096 rows; the fixed2080MiB root
allocation is unchanged. No physical memory/refund/speed claim.453 distinct
focused CPU cases /56 nonempty suites, locked test build, strict core/off lint
and changed format/whitespace pass. Current normal artifact
`strata-packed-gr-ordinary-20261016/` has SHA256
`fc0d1c655302c774427905e36932797c5c45eb3c02b9b2baf79403be8e198d7b`;330 offline shader variants pass (122 physical checks). Source-clean/stable
build, independent hash/mode0500 verified. No device initialized. These are source/CPU checks, not numerical device
or whole-model acceptance.

Remaining mixed packed-input reuse, in-place GR/PLE consumers, allocator and
whole-device pricing/source/scheduling substitutions and numerical/hot-reuse/
failure/recovery gates remain required. Private executable input is still gated
on independent protection/admission; current CLI is compiled-public/one-request.
No API/config/frontend change; required Vashti changes:none. Goal active CPU-only,
whole rewrite incomplete, speed gate closed. No GPU/model/service/install/release
action; last physical recovery is historicalOct15. Root not needed. Preserve
canonical application work and inherited untracked files.


## Active Kuno engine handoff — mixed-input image reuse (2026-10-16)

Isolated `codex/strata-rust-port`, source `40fe97e6989128189c6046b3b73c64c09288c94a`. Read the
[source/validation receipt](kuno-strata-rewrite-completion.md#connected-native-mixed-input-image-reuse-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Actual GR mix now emits F32,
BF16 and FP16 together. Original source/view/count/first/stride and a scoped
actual queued-writer marker select original packed images for GDN/QSA/indexer,
router/shared projections and routed-expert gather. Next GR invalidates the
marker; no external readiness metadata. Native matrix and original route/weight/
source/queue owners remain. No extra root, new API or required Vashti change.

455 distinct CPU cases /56 nonempty suites, locked release test build, strict
core/off lint and format/whitespace pass. Normal artifact
`strata-mixed-images-ordinary-20261016/`, SHA256
`9d1bedf93be55ca5447bfa3ca8d40a29736820b7345335d182c0fac571f3a5d8`;
all330 offline shader variants pass, including122 physical-pointer checks.
Clean/stable source build and independent hash/mode0500 verified. No device
initialization or numerical model acceptance. Both images use40MiB inside the
unchanged2080MiB prompt root inventory; no physical refund/residency/speed claim.

Next native in-place GR/PLE consumers, allocator/whole-device pricing and final
source/scheduling substitutions remain, plus full numerical/state/hot-request/
failure/recovery acceptance under renewed GPU authorization. Protected private
CLI input remains unclosed; existing executable is compiled-public/one-request.
Goal stays active CPU-only, rewrite incomplete, speed gate closed. No GPU/model/
service/install/release/API/config/frontend action. Root not needed. Last physical
recovery historicalOct15; preserve canonical application work/untracked files.


## Active Kuno engine handoff — native PLE region reuse (2026-10-16)

Isolated `codex/strata-rust-port`, source `9d7821a9145be34b190493006e4d6b46c5cc8d03`. Read the
[implementation/validation receipt](kuno-strata-rewrite-completion.md#native-ple-key-normalization-and-query-region-reuse-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Actual selected PLE normalizes
key in place with one writable binding and reuses the query region for conv norm
only after gate queues its last reader. Gated/original residual/history remain
live. Ten unique regions replace twelve; logical phase payload shrinks320MiB
inside unchanged2080MiB owning roots. Actual MTP teacher uses idle views8/9;
removed view10 has no remaining caller. No physical memory or speed claim.

457 distinct CPU cases /57 nonempty suites, locked checked release test build,
strict core/off lint and format/whitespace pass. Linked receipt records exact
normal frozen artifact `strata-ple-reuse-ordinary-20261016/`, SHA256
`dc1628984b186162f56fe926e2d93e2b2f8536c2059acfe447c67d8abbe9d4b4`;
331 offline shader variants pass (122 physical checks). Source-clean/stable
build and independent hash/mode0500 verified; no device initialized. First test closure typing
error is fixed; final coverage explicitly includes the new shader schema case.
These checks do not prove in-place numerical/device execution.

**Next source fix:** upstream requires32-lane PLE norm/gate; native plain-WGSL
registration currently leaves required subgroup unset and allows varying size.
Audit fixed32 families and preserve their exact requirement in ordinary/native
pipeline paths; keep cooperative16 separate. This is an identified source gap,
not an observed hardware fault. Remaining GR/PLE residual in-place, allocator/
whole-device pricing/source/scheduling and full numerical/state/hot-request/
failure/recovery gates remain. Private CLI input remains independently gated.
No API/config/frontend change; required Vashti changes:none. Goal active CPU-only,
rewrite incomplete, speed gate closed. No GPU/model/service/install/release action.
Root not needed; last physical recovery historicalOct15. Preserve canonical work.


## Active Kuno engine handoff — original device subgroup policy (2026-10-16)

Source `5d668c3cdbb669eb5d72bfa93fcccfbb3abe15ad`, isolated `codex/strata-rust-port`.
Read the [exact source/validation receipt](kuno-strata-rewrite-completion.md#original-device-subgroup-policy-and-shared-pipeline-construction-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Before creating any shader,
the SAME admitted original device checks real subgroup properties and establishes
fixed32 WGSL policy. Actual module factories snapshot it, and ordinary/retained
pipeline builders share required-size/flags/pNext construction. Matrix SPIR-V
retains16; generic devices retain unset policy. No shader arithmetic, model/root
allocation or application contract changes. Earlier PLE subgroup source-gap
instructions are superseded; physical width/numerical acceptance is still open.

463 distinct CPU cases /58 nonempty suites, locked selected release test build,
strict core/off lint and formatting pass. Normal frozen artifact
`strata-subgroup-owner-ordinary-20261016/`, SHA256
`0810aa82d7699be35d866ec238010ec2272cb0ea9e27f84d04472e4d4fd27263`;331 offline variants pass
(122 physical checks), source-clean/stable build and independent hash/mode0500
verified. Exact logs and earlier ShaderModule.as_hal API correction are in the
receipt. Null-handle metadata fixtures do not create driver objects or establish
hardware ownership/acceptance. No adapter or GPU was initialized.

Next remaining native GR/PLE residual in-place, allocator/whole-device pricing,
source/scheduling/substitution audit and full numerical/state/full-context/hot-
request/failure/recovery gates. Private executable input remains independently
gated; current CLI compiled-public/one-request. Goal active CPU-only, rewrite
incomplete, live speed gate closed. No GPU/model/service/install/release/API/
frontend action; required Vashti changes:none. Root not needed. Physical recovery
remains historicalOct15. Preserve canonical application work/untracked files.


## Active Kuno engine handoff — sole prompt residual (2026-10-16)

Source `81ff9640afcaf9c520e1915ea668cd49bdef60dc`, isolated `codex/strata-rust-port`.
Read the [exact receipt](kuno-strata-rewrite-completion.md#sole-prompt-residual-and-coupled-teacher-scratch-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Coupled GR/PLE now update the
same original target R through one writable descriptor, with partial aliases
rejected. Actual MTP teacher uses original idle PLE key scratch, preserving target
carry. All outer/dynamic shared/mixed/tail consumers move together; second-R
payload160MiB is removed. Fixed2080MiB roots remain; no physical refund/speed claim.
The packed GR producer gains the required storage-memory ordering before other-
lane reads. Earlier fixed32/cooperative16 policies remain connected.

467 distinct CPU cases /58 nonempty suites, final locked release test build,
strict core/off lint and changed formatting pass. Linked receipt preserves exact
logs and corrected import/null-ID fixture failures. Normal artifact
`strata-single-residual-ordinary-20261016/`, SHA256
`215dc87b810702baa372bb7fd4709a2929695371508c7b0d6ef7728de0616d2c`; source-clean/stable build,
independent hash/mode0500 and334 offline variants pass (122 physical checks). No device initialized; these
checks do not establish numerical/device correctness. Remaining allocator/whole-
device pricing/source/scheduling audit and full numerical/state/hot-request/
failure/recovery acceptance remain required. Private executable input is an
independent gate; current CLI compiled-public/one-request. Goal active CPU-only,
rewrite incomplete, speed gate closed. No GPU/model/service/install/release/API/
frontend action; required Vashti changes:none. Sudo refresh in the shared shell
unconfirmed; no root needed. Physical recovery historicalOct15. Preserve canonical
application work and inherited untracked files.


## Active Kuno engine handoff — exact native pages (2026-10-16)

Source `ba0cc6a815c8f4aa562b7bb386342817e8a0ee30`, isolated `codex/strata-rust-port`.
Read the [complete source/validation receipt](kuno-strata-rewrite-completion.md#exact-exclusive-pages-and-shared-prompt-inventory-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Selected original allocator
owns exact aligned pages, reuses only equal sizes and checks descriptor capacity.
Shared live inventory drives prompt carving and allocation;2MiB private granules,
admission and target/MTP/public-source bounds move together. Prompt requests are
756/642/32MiB; target private plan2,778,038,272B is842MiB lower. Ordinary allocation
policy stays unchanged. No original expert/model/kernel math or API changes.

504 distinct selected CPU cases /61 nonempty suites,54 ordinary cases /4 suites,
final locked release test builds, strict core/off lint and changed format pass.
Real allocator tests use metadata-only counting CPU storage, not a GPU. Linked
receipt records corrected profile fixtures/admission quotes and source limits.
Normal artifact `strata-exact-pages-ordinary-20261016/`, SHA256
`794438e1b0b4f8bbb54752fd46e03f1c0c126b1e1a020e528f14468c6685e88f`; source-clean/stable build,
independent hash/mode0500 and334 offline variants pass (122 physical checks). No device,
model, inference, service, install, release or frontend action. Required Vashti
changes:none. No physical memory/refund/residency or speed claim. Goal active
CPU-only, whole rewrite incomplete, live speed gate closed. Next final source
queue/scheduling/substitution and whole-device physical/runtime/HOST pricing,
numerical/state/full-context/hot-request/failure/recovery gates. Private input
remains independently gated; current CLI compiled-public/one-request. Root not
needed; last sudo refresh unconfirmed. Recovery historicalOct15. Preserve
canonical application work and inherited untracked files.

## Active Kuno engine handoff — original cell transfers (2026-10-16)

Source `8bd6edff399f531565642a66472dfd6afbba0ea1`, isolated `codex/strata-rust-port`.
Read the [source/acceptance receipt](kuno-strata-rewrite-completion.md#original-cell-transfer-copy-path-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Actual global adaptive exchange
now records original HOST-to-scratch, GPU-to-HOST, scratch-to-GPU transfers for
all three roles instead of the global swap shader. Actual original HOST imports
support these copies; the registered source predicate and HAL producer barrier
change together. Scratch is the already-attached complete32MiB dormant prompt
raw owner, retained through the actual join and fully erased on reset/close.
No additional buffer or duplicate model/HOST complement; original role/placement/
rebase and private route-map/usage publication remain coupled.

508 distinct CPU cases /62 suites, selected release build and strict core/off
Clippy pass. Detailed receipt records exact scope and the duplicate-filter runner
correction. Normal source-clean/stable artifact `strata-cell-transfer-ordinary-20261016/`
(`strata-cell-transfer-ordinary-20261016.out`) has SHA256
`b85b758107e6a34db4e97d053ffa5e98c654fca1842a41bfe22892d9b33abc59`.
Independent source/hash/mode0500 checks pass. Offline
`strata-cell-transfer-shaders-20261016.log` exits0 with334 variants and122
physical-pointer checks. This command returns before adapter/device creation.

**Open source dependency:** adaptation still runs after target commit and MTP
next drafting, with original whole-client/rebase joins. Upstream serves with
independent copy submission during those operations, then joins pending copies
before next target residency publication. Implement the connected original
usage/transfer/reader lifetime/publication boundary together; removing waits or
adding a queue alone is insufficient. Audit suspected first count1 usage/tick
accounting, final source substitutions and full physical/pricing/numerical/state/
hot-request/failure/recovery gates. Full rewrite incomplete, speed gate closed.

No API/protocol/config/consumer change; required Vashti changes:none. No GPU,
adapter, model, inference, service, install or release action. No current physical
recovery or speed claim. Goal active CPU-only; root not needed. Latest shared
shell sudo refresh required a password. Canonical application/inherited files
preserved. This handoff does not authorize integration, deployment or release.

## Active Kuno engine handoff — adaptive copy flight (2026-10-16)

Source `53cd14ced31da04c71408e76b266bdec2a2b8c5c`, isolated `codex/strata-rust-port`. Read the
[detailed source/acceptance receipt](kuno-strata-rewrite-completion.md#adaptive-copy-flight-across-commit-and-drafting-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Actual verified windows now
start original expert-copy flights after their row observer releases its lease
and before target commit/next MTP drafting. Actual adopted copy queue, original
buffers/scratch, registered rebase and private-arena cleanup exclusion stay owned
through actual transfer and copy-to-MAIN semaphore/fence completion. Publication
of physical placement and private route-map/usage still gates next target reuse.
The final prompt input/count1 uses this same accounting; no duplicate after-window
tick. Diagnostic checkpoint replay no longer trains the selected native cache.

514 CPU cases /63 suites, selected test build and strict core/off lint pass.
Exact tests/limits/build correction are in the linked receipt. Normal source-clean/stable artifact `strata-adapt-flight-ordinary-20261016/`
(`strata-adapt-flight-ordinary-20261016.out`) has SHA256
`4fe370cae0c05c30453dd1dd66c1f01933e46017b6ea951016223d09f1deea46`; independent original source/hash/mode0500 checks pass.
The build exits0 without initializing an adapter/device.
No changed kernels or new shader acceptance. No GPU/model/inference or physical
residency/overlap/speed claim. No extra GPU/model root or duplicated complement.

Source substitutions remain explicit: CPU proposal selection is serial before
commit; adaptation and MTP INFO uploads share the adopted copy queue; first-window
outer delivery/drafting remains as before. Final source/queue/ownership/pricing
review and full protected numerical/state/context/hot-request/failure/recovery
acceptance remain required. Whole rewrite incomplete; speed gate closed. Private
executable input remains separately gated; current CLI compiled-public/one-request.

No API/protocol/config/consumer change; required Vashti changes:none. No installed
service, release, frontend or GPU action. Goal active CPU-only; root not needed;
latest shared-shell sudo refresh unsuccessful. Canonical/inherited work preserved.
This handoff does not authorize integration/deployment/release.

## Active Kuno engine handoff — mapped embedding gather (2026-10-16)

Source `5b520bb50de8bc39387c42e42166ab24437d6447`, isolated `codex/strata-rust-port`. Read the
[source/test receipt](kuno-strata-rewrite-completion.md#original-mapped-embedding-gather-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md). Actual target and shared Q2 MTP
front/teaching now use one original mapped embedding gather with Strata's32-lane/
256-value work sharing and two float4 writes per lane. Existing table0's tail
carries original16 page addresses; all actual HOST page owners and native read
dependencies remain retained. No new GPU/model root, weight copy, source format,
residency price or API change. Standalone nonshared diagnostics keep old tiling.

522 selected CPU cases /65 suites and strict core/off lint pass. The linked
receipt records the independent IQ4_XS lane oracle, actual address/shader checks,
newly-included front coverage and corrected old context fixture. Normal source-clean/stable artifact `strata-embedding-gather-ordinary-20261016/`
(`strata-embedding-gather-ordinary-20261016.out`) has SHA256
`6f6e8c34069852c1bd7725bae7a417a54d036c38aa4b70d3e623f63dc4578c77`.
Independent original source/hash/mode0500 checks pass. Offline
`strata-embedding-gather-shaders-20261016.log` exits0 with335 variants and123
physical-pointer checks from that exact frozen executable. No adapter/device is
created by this validator.

Whole rewrite incomplete; speed gate closed. Physical numerical/read/residency/
reuse/recovery behavior is unverified. Next final source/head/sampling/queue/
ownership/pricing/substitution audit, then full protected device acceptance under
renewed GPU permission. Preserve earlier adaptive scheduling substitutions and
the independent protected-private-caller boundary; current CLI compiled-public/
one-request. No GPU/model/inference/service/install/release/frontend action or
new physical speed claim. Required Vashti changes:none. Goal active CPU-only;
root not needed. Canonical/inherited files preserved. No deployment authorization.

## Active Kuno engine handoff — native wide Q6 heads (2026-10-16)

Source `ffe7293be448bb62ebc48e0ad9c51e11cc79c7af`, isolated `codex/strata-rust-port`. Read the
[source/test receipt](kuno-strata-rewrite-completion.md#native-aligned-wide-q6-heads-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md). Target and compact-CJK MTP Q6
consumers now select aligned wide A2/RPW1/subgroup32, four blocks per subgroup,
eight lanes per block and actual1–8 RHS variants. Original210-byte weights,
Q8 producer/plane, private/public owners and pricing remain; no weight copy or
new GPU root. Other-format/indexed Q8 consumers are unchanged.

527 CPU cases /66 suites, selected release check/test build and strict core/off
lint pass. Linked receipt distinguishes actual vector/schema/lane coverage,
newly-included dense cases and the corrected old640-column test fixture.
Normal source-clean/stable artifact `strata-wide-q6-ordinary-20261016/`
(`strata-wide-q6-ordinary-20261016.out`) has SHA256
`21a58866c2334c3efe54c6db56b4097783b50727b38abdf357b504468989a2dd`.
Independent original source/hash/mode0500 checks pass. Offline
`strata-wide-q6-shaders-20261016.log` exits0 with340 Naga/SPIR-V variants and123
physical-pointer validation checks from that exact frozen executable. The
validator returns before adapter/device creation.

**Remaining implementation:** whole-head dispatch over original bounded target
and compact-CJK roots, authenticated BDA metadata/read dependencies and exact
ownership/admission. Current controllers still loop pages; upstream uses one
launch. This is not complete because the inner wide kernel matches. Then final
sampling/source/queue/ownership/pricing/substitution audit and full protected
numerical/state/context/reuse/failure/recovery acceptance under renewed GPU
permission. Whole rewrite incomplete; speed gate closed. No physical speed/
residency or model-parity claim. Keep existing adaptive substitutions and the
independent protected-private-caller gate (CLI remains compiled-public/one-request).

No API/protocol/config/consumer change; required Vashti changes:none. No GPU,
model, inference, service/install/release/frontend action. Goal active CPU-only;
root not needed; canonical/inherited files preserved. No deployment authorization.

## Active Kuno engine handoff — whole original heads (2026-10-16)

Source `308de4079bae86efa59c9bbeaba75d93dc481448`, isolated `codex/strata-rust-port`. Read the
[source/test receipt](kuno-strata-rewrite-completion.md#whole-original-target-and-compact-cjk-head-dispatch-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md).
Target and compact-CJK MTP now each prepare Q8 once and project their whole
original Q6 head in one 128-thread/four-row dispatch. Original bounded source
pages remain; actual native recordings retain every page they read. Two owning
32 KiB address tables add 65,536 public GPU bytes in the shared model plan. Exact
original tensor subtraction, private/source windows and model bytes are unchanged.

532 distinct selected CPU cases / 68 suites, strict core/off binary Clippy,
changed formatting/whitespace and normal stable-source build pass. Frozen artifact
`strata-whole-head-ordinary-20261016/` has SHA256
`5766b594472a1debfcf18c07a8ff01459239d60913e3266971a17ba2995a3f0a`. Independent source/hash/mode0500 checks pass;
`strata-whole-head-shaders-20261016.log` accepts 356 offline Naga/SPIR-V
variants and 139 physical-pointer checks. The validator returns before device
creation. These are source/CPU/compiler checks, not GPU numerical/model acceptance.

Actual target output/verifier and original-order CJK MTP consumers both select
this path. The internal metadata helper borrows the original native device and
full same-stream resources; it grants no submission/cleanup/admission credit.
CJK startup readback checks table bytes and owner padding. Legacy verifier tile
counter still counts16 original source pages; actual projection instrumentation
counts one whole dispatch. No new model copies or private scratch. Generic Q8 MTP
public price remains4,203,216,896 B; runtime Q2 public 1,946,607,616 B; target public
521,506,816 B; original two-tensor subtraction
859,189,248 B remains exact. Preserve the complete connected branch, not a leaf commit.

Whole rewrite remains incomplete; live speed gate closed. Next final connected
sampling/source/queue/ownership/pricing/substitution audit, then protected full
numerical/state/context/hot-request/failure/recovery acceptance under renewed GPU
permission. Keep adaptive serial-proposal/shared-copy-queue/first-window
substitutions explicit. Protected private caller remains independently gated;
current CLI compiled-public/one-request. No GPU/model/inference/service/install/
release/frontend/API action or physical performance/residency/refund claim.
Required Vashti changes: none. Goal active CPU-only. Root not needed; latest shared
shell refresh still reports a password required. Physical recovery historicalOct 15.
Preserve canonical application work and inherited files. No deployment authority.

## Active Kuno engine handoff — coupled draft sampling (2026-10-16)

Source `8a4ee5b820ad3508b6d7f60d4d58e5efd81bd408`, isolated `codex/strata-rust-port`. Read the
[source/acceptance receipt](kuno-strata-rewrite-completion.md#coupled-draft-penalty-writes-and-fixed-64-selection-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md). Full and compact-CJK draft callers
now select the complete original coupled sequence, including the logits writer
and fixed 64 list/merge dependency. Candidate/Params/output/history owners and
all admission prices remain unchanged; no weight, private or public root is added.
Target and greedy MTP sampling retain their separate original selection. Existing
GPU counter/ring/ID bridges and target position-based draw semantics were audited;
physical/numerical execution remains unverified.

534 CPU cases / 68 suites, selected release check/test build, strict core/off
binary lint and changed formatting pass. Normal stable-source frozen artifact
`strata-coupled-sampling-ordinary-20261016/` has SHA256
`c8ee12ced38eb658a5ebd8082a221dd0e49ea778b429b78d956417021ea295a6`; independent source/original hash/mode0500 pass. Offline validator
accepts 360 Naga/SPIR-V variants and 139 physical checks before device
creation. Exact logs and CPU oracle limits are in the linked receipt. Do not claim
all-target/test lint passing; inherited findings remain in the previous receipt.

Whole rewrite incomplete; speed gate closed. Remaining final source/queue/
ownership/pricing/substitution audit and protected numerical/state/context/hot-
request/failure/recovery acceptance stay required. Preserve existing adaptive
substitutions and independent protected-private-caller gate (CLI compiled-public/
one-request). Goal active CPU-only. No GPU/model/inference/service/install/
release/frontend/API action. Required Vashti changes: none. Root not needed;
last shared-shell sudo refresh unsuccessful. Physical recovery historical Oct 15.
Canonical/inherited files preserved. No deployment authority.

## Active Kuno engine handoff — compute-local uploads (2026-10-16)

Source `7c35203653d0405e8291bff0307f9438b8376f13`, isolated `codex/strata-rust-port`. Read the
[complete source/validation receipt](kuno-strata-rewrite-completion.md#compute-local-control-uploads-and-mtp-copy-scheduling-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md). Target and MTP INFO uploads now use
separate actual compute-family owners/epochs/fences. Original MTP copy/fill/math
nodes use its compute lane, with their actual transfer access flags/barriers.
Adaptation retains the existing transfer queue; no new hardware queue is required.
Pending peer INFO source/destination roots are excluded from scoped I/O reuse.
INFO-only drains retain actual GPU semaphore/fence visibility even on one queue.

Native HOST metadata grows 2 MiB to 140 MiB; parent prompt/controller quote derives
268 MiB. No model/private/public GPU buffer is added. One extra actual INFO command
pool/buffer/timeline/fence stays owned; physical pricing/runtime acceptance is open.
537 CPU cases / 69 suites, locked test build, strict core/off binary lint
and changed formatting pass. Exact frozen artifact
`strata-draft-local-uploads-ordinary-20261016/` has SHA256
`c2e0c7e52e70ba0d0937eec9e1880f270a96ad366a771e874a4754834f7cc6d0`; independent source/original hash/mode0500 and 360 offline variants/139
physical checks pass before device creation. Kernel arithmetic is unchanged.
These are CPU/source/compiler receipts, not device execution or overlap proof.

Remaining concrete source gaps in the linked receipt: adaptive host ranking/
submission is still serial; the first retained-request draft is outside the scoped
observer and can miss overlap on an adaptive tick; original serve retains usage
and learned policy while current request reset/local controller do not. Resolve
these with content-free model telemetry and original request erasure, never
retained prompts/tokens/suffix/KV or serialized traces. Final source/queue/
ownership/pricing and protected numerical/state/context/hot-request/failure/
recovery acceptance remain required. Full rewrite incomplete; no speed trials.
Protected private caller stays separate; CLI compiled-public/one-request.

No API/protocol/configuration/consumer change; Vashti changes: none. Goal active
CPU-only. No GPU/model/inference/service/install/release/frontend action. No root
used; canonical/inherited files preserved. No deployment authority or new physical
recovery/performance/residency claim.

## Active Kuno engine handoff — retained model telemetry (2026-10-16)

Source `39c59c1fc90dc71b94e7f671677fae9c1626be89`, isolated `codex/strata-rust-port`.
Read the [complete receipt](kuno-strata-rewrite-completion.md#loaded-model-usage-and-draft-policy-across-erased-requests-2026-10-16) and
[checkpoint](kuno-native-text-checkpoint.md). Supersedes the previous handoff's
usage/policy gap only; keep its connected compute-local queues/INFO, coupled
sampler and all preceding owners.

Actual successful closed requests snapshot bounded aggregate expert counters,
then erase/inspect FULL original private arenas. The next request consumes
original erasure receipts before restoring actual physical routing/aggregate
usage, with exact complete bit/padding readback. Loaded Session retains only
fixed scalar learned-policy bins. Tokens/suffix/history/seed/KV remain separate
and wipe. Model close/unload wipes Cache/Policy; no logging/serialization/export
or durable content store is added. Existing HOST allowance covers metadata;
no GPU buffer, public contract or quote changes.

540 CPU cases / 71 suites, locked release test build, strict core/off **binary**
lint and changed formatting pass. Normal frozen artifact
`strata-model-telemetry-ordinary-20261016/` has SHA256
`47eb2abef25a3d65e6501b353b27e4f7dc756ee0dcfe7cb51f347b1d030fb823`; independent source/original hash/
mode0500 checks and offline 360 shader variants/139 physical checks pass. No GPU,
hot-request, numerical, overlap, privacy-forensics or performance acceptance claim.
Initial synthetic erase fixture failure/correction is preserved in the receipt.

Remaining source work: serial adaptive host ranking/submission and first count1
MTP drafting outside the scoped observer on retained requests. Final source/
ownership/placement/queue/substitution audit and protected whole-model numerical/
state/context/hot-request/failure/recovery acceptance remain, then speed trials.
Complete rewrite still incomplete; CLI compiled-public/one-request, private process
caller independently gated. No API/protocol/configuration/consumer/frontend change;
required Vashti changes: none. No GPU/model/inference/service/install/release/root
action. Goal active CPU-only. Sudo refresh was attempted but not confirmed; root
was not needed. Canonical/inherited files preserved. No deployment authority or
new physical recovery/residency/performance claim.

## Active Kuno engine handoff — scoped first draft (2026-10-16)

Source `01291fe0052ba7e71454ae009b86033972211cd4`, isolated `codex/strata-rust-port`.
Read the [complete source/validation/substitution receipt](kuno-strata-rewrite-completion.md#scoped-first-window-mtp-consumer-and-actual-runtime-nonce-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Preserve the full connected
branch, including usage/policy retention and compute-local INFO/MTP queues.

Actual final-prompt count1 now uses the same Verified/Pending transaction as later
windows: real host-row lease release, adaptive flight, original-R readiness/pin,
sampler commit, one prediction delivery and actual MTP producer in the scoped lane,
then original full-family publication. EOS/cap skips drafting; count1 fronts and
confidence windows stay authentic. Outer fallback is removed; every subsequent
MTP target input requires its real completed scoped producer.

The backend scoped graph requires an ACTUAL runtime nonce. The first window
previously preceded the long generation scope, exposing a missing owner in source.
New explicit `InitialWindow` authority opens an admitted short runtime scope,
allows exactly the final prompt input, and requires real model/fault/full I/O
closure before returning. It grants no generated cursor/scope/reset/rearm credit.
Then target policy selection and long generation scope proceed unchanged. This
extra actual first-window audit/drain is an explicit Vulkan ownership substitution;
physical behavior/cost remain unverified. No nonce/completion is fabricated, no
model/KV private erasure happens between first window and continuation, and no
extra GPU/model/root or simultaneous runtime grant is added.

542 CPU cases / 71 suites, locked release test build, strict Intel/core-off binary
lint and changed formatting pass. Normal frozen artifact
`strata-first-scoped-draft-ordinary-20261016/` has SHA256
`8ced9ff4099d1a4f2fc3f47e6880af9e92d03696d3e6e0cc6bd4b50e9eb42c42`; independent source/original/hash/
mode0500 pass. Frozen binary not executed. Shader arithmetic/inventory unchanged;
previous `39c59c1` offline 360 variants/139 physical checks are the last shader
receipt, not rerun. Initial transition test failure and duplicate-filter rejection,
then fixes and final accepted cases, are preserved in the detailed receipt.

Remaining: serial adaptive host ranking/submission, final full connected source/
ownership/placement/queue/pricing/substitution audit, protected whole-model
numerical/state/context/hot-request/failure/recovery acceptance, then speed tests.
Rewrite incomplete; CLI compiled-public/one-request and private caller independently
gated. No API/configuration/consumer/frontend change; Vashti changes: none. Goal
active CPU-only. No GPU/model/inference/service/install/release/root action;
canonical/inherited files preserved. No deployment authority or new physical
recovery/performance claim. No sudo credential was needed or used.

## Active Kuno engine handoff — adaptive host job (2026-10-16)

**Historical extension handoff:** selected counter policy and universal12MiB
allowance are superseded by the selected-policy/router handoff below. Preserve
the real copy worker for explicit GPU-derived usage; do not assume NO_HOST
reference mode promotes experts.

Source `f9c7471f498e877be1a9c733aa1a3db3ccdfb154`, isolated `codex/strata-rust-port`.
Read the [complete source/ownership/bounds receipt](kuno-strata-rewrite-completion.md#adaptive-host-job-and-owned-transfer-submission-authority-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md). Keep the FULL coupled branch,
including original compute-local INFO, scoped first draft and retained usage/policy.

The actual selected adaptive tick moves SAME Cache/wiping usage into a bounded
host worker before commit/MTP. It does original ranking, validates authenticated
physical cells, records the three-role exact exchange and submits actual copies.
Parent retains client/registry token/original scratch exclusion and MAIN queues,
then joins actual worker/copy/visibility before original private control/physical
publication. No thread result or attempt bit grants GPU completion. Unknown
completion retains/quarantines original owners; private arena/TLS never moves.

An immutable48x512x3 physical-cell alias inventory is authenticated once at load;
no weight mirror. All expert HOST pages are revoked before selection; embedding/
shadow sources stay distinct. New HAL owned transfer authority retains same device
and serializes BOTH graph/worker submits, not waits, on the actually adopted
queue. No unsafe Send/Sync or extra GPU queue. HOST allowance 2->12MiB explicitly
covers existing temporaries,2MiB worker stack and8MiB inventory cap; no new GPU
weight/private/scratch root. Real physical/runtime pricing still needs acceptance.

545 CPU cases / 71 suites, locked release test build, strict core/off binary lint,
changed formatting and normal stable build pass. Frozen artifact
`strata-adaptive-host-worker-ordinary-20261016/` has SHA256
`d4cfebbb5a28d645b296c20aaa4dea09239b961f1cea4456ab8299ed2e17e7e5`; independent source/original/hash/
mode0500 pass. Frozen binary NOT executed. Shader math/inventory unchanged;
previous `39c59c1` offline360/139 is still the latest shader receipt, not rerun.
No numerical/GPU/registry/allocator/overlap/performance acceptance is claimed.

All three concrete state/schedule gaps from the compute-local audit are connected
in source. Full rewrite still incomplete: final whole-branch source/placement/
queue/ownership/pricing/substitution audit, protected numerical/state/context/
hot-request/failure/recovery acceptance, then speed trials. CLI remains compiled-
public/one-request; private process caller separate. No API/configuration/consumer/
frontend change; required Vashti changes: none. Goal active CPU-only. No GPU/model/
inference/service/install/release/root action; canonical/inherited files preserved.
No sudo needed/used, deployment authority or new physical recovery claim.

## Active Kuno engine handoff — selected policy and router (2026-10-16)

Runtime source `0d9e3a527041c36b2a395cda5c1070a88288684d`, including policy
`3d6a0390815e0ac116f98d765c398e5b74b1cec9`, isolated `codex/strata-rust-port`.
Read the [exact source, ownership, bounds and validation receipt](kuno-strata-rewrite-completion.md#selected-no_host-policy-and-native-routerplan-schedule-2026-10-16).
Keep the complete coupled branch. This corrects selected pinned behavior, with
no throughput-driven rollback.

Reference DEVICE_PLAN/NO_HOST skips the only expert-usage callback. Default now
keeps zero counters and actual every4-window bounded empty host job beside commit/
MTP. Cold/capture/replay flags share one policy; no empty-tick GPU readback,
copy/revocation or unchanged-control upload. Original full erasure/rebind audits
remain. Explicit `--intel-core --gpu-adaptive-usage` preserves prior promotion
inventory/worker/owned queue path. Adaptive HOST4MiB default/12MiB extension.

Selected decode now uses per-token256-thread router blocks (one active32-lane
subgroup) and original128-thread packed parallel plan scan. SAME original private
map/fault/INFO owners connect native RAW/WAW dependencies to actual VRAM/HOST role
consumers. Existing fixed8 member planes and physical-cell/address-table layout
are recorded Vulkan substitutions. No new GPU root, missing expert or CPU expert
callback. Prompt's large-chunk grouping is unchanged.

549 CPU cases/72 suites, strict core/off binary lint and formatting pass.
Eight focused usage/worker/router cases also pass in the explicit extension build.
Normal stable-source frozen build SHA256
`2d60cec93970727aeea74a1d7ad3f362cd98dc2ea300e56e62f1a79d94c39a8c`, independent source/original hash/mode0500/
features pass. Offline-only361 shader variants/139 physical checks/32 matrix
configurations pass with the pinned matrix-aware validator (initial old-system-
validator failure retained). Exact artifact/logs and substitutions are in the
linked primary receipt.
No actual GPU numerical/driver/overlap/whole-model/throughput acceptance. Full
branch final audit and protected correctness/state/hot-reuse/recovery remain before
speed. CLI compiled-public/one-request; private process caller independently gated.
No API/consumer/frontend change; required Vashti changes: none. Goal active CPU-only.
No GPU/model/inference/service/install/release action, no root operation. Canonical/
inherited files preserved. No deployment authority or new physical recovery claim.

## Active Kuno engine handoff — context scope and shared expert gap (2026-10-16)

Runtime source `c5fcc20bc44e8dce900ac3bc4d1865477854f2fc`, isolated `codex/strata-rust-port`. Read the
[exact boundary correction, receipts and next coupled implementation plan](kuno-strata-rewrite-completion.md#native-context-headroom-and-full-window-scope-audit-2026-10-16).
Keep the FULL branch. Native Request/reader/controller and original Geometry /
Authority constructor/rearm now reserve eight cells as pinned serving does.
Selected active run and original close/reset bounds accept a full committed
prefix beyond clipped output delivery, while no next window may start at or
past the final delivery origin. Actual family/state/fault/I/O/erase proof is still
required; scope shape does not grant completion. Old non-native guards retained.

551 CPU cases/72 suites and strict core/off binary lint pass. Exact initial failures/final receipts are in the linked
primary audit. No shader change or new frozen normal artifact for this audit block;
latest frozen source`0d9e3a52` cannot test the new context/scope behavior.

**Concrete required next block:** source selected Intel shared-expert fork/join
and native scalar-gate/scale/fused SwiGLU->Q8 path. Rust currently serializes shared
prepare/finish on MAIN, uses a generic64-lane scalar gate and has only compute/
copy segments inside a target graph. Independent MTP lane does not complete this.
The linked plan covers selected master-switch defaults, actual source call sites,
independent quantized scratch and fault fold, possible reuse of existing second
compute queue, exact recorded/replay dependencies, INFO/fences/failure retention
and pricing. Implement these together; do not evaluate a half fork for speed.

Whole rewrite incomplete, speed gate closed. Final source review and protected
whole-model numerical/state/context/hot-reuse/failure/resource-recovery remain.
No public API/consumer/frontend change; Vashti changes: none. Goal active CPU-only.
No root/GPU/model/inference/service/install/release action; canonical/inherited
files preserved. No deployment authority or new performance/recovery claim.

## Active Kuno engine handoff — connected shared expert (2026-10-17)

Runtime source `46e695a7c6462eab86171086d04c742b3e4386de`, isolated `codex/strata-rust-port`.
Read the [exact source/ownership/quote and validation receipt](kuno-strata-rewrite-completion.md#connected-shared-expert-and-three-queue-target-graph-2026-10-17)
and [checkpoint](kuno-native-text-checkpoint.md). Keep the FULL coupled branch.

Selected target shared math now forks onto existing compute queue1 after router/
plan while routed roles run on MAIN. Independent shared fault/Q8 owners, fused
SwiGLU pack, authentic rank-one BF16 gate, native vec4 scale and ordered combined
reduction connect together. MAIN release prevents later join from serializing
routed work. Actual graph Nodes/signatures, three-queue interval hazards, family
pools, INFO edges and original final fence carry cold/capture/replay. Final fence
joins all shared/copy work before the queue is reused by MTP. No synthetic
completion or weakened general fault hazards. Additional private fault32KiB and
HOST16MiB are explicitly quoted; actual physical admission/overlap remain unverified.

558 distinct CPU cases/73 suites, strict core/off binary lint and normal stable-
source build pass. Frozen source46e695a7 independently matches its ordinary binary,
mode0500/features; offline365 shader variants/139 physical checks/32 matrix
configurations pass. Exact SHA256/artifact/logs are in the linked primary receipt.
This is CPU/source acceptance, not a live device or inference result. Initial shader reserved-name rejection and
runner duplicate-suite rejection were repaired; initial logs remain evidence of
failure, not acceptance. No GPU/numerical/driver/whole-model/throughput acceptance.

Next: final whole-branch source/ownership/placement/queue/pricing/substitution
audit, protected count1/multirow numerical/family/state/context, retained/hot-request
and all-resource failure/recovery validation, then speed. Full rewrite incomplete;
CLI compiled-public/one-request and private process caller independently gated.
No API/protocol/configuration/consumer/frontend change; required Vashti changes:
none. Goal active CPU-only. No GPU/model/inference/service/install/release action;
sudo cache refreshed, no privileged runtime operation. Canonical and inherited
untracked cache preserved. No deployment authority or new performance/recovery claim.

## Active Kuno engine handoff — cold target scope audit (2026-10-17)

Runtime source `811968c38a272916e0bf41620473eb16185dae81`, isolated `codex/strata-rust-port`.
Read the [actual caller/queue audit and receipts](kuno-strata-rewrite-completion.md#cold-target-scope-and-sharedmtp-source-audit-2026-10-17)
and [checkpoint](kuno-native-text-checkpoint.md). Preserve the FULL coupled branch.

Actual one-token-prompt target starts at0 with a real InitialWindow request nonce
and no named capture key. Longer initial prompts may be captured. The new shared
guard incorrectly required a key; backend preflight now distinguishes original
runtime scope from optional capture. Named keys still require matching nonce and
target family; Graph still requires actual MAIN lane/no active MTP scope. Existing
real descriptors/fault fold/peer waits/final fence/replay/cleanup remain unchanged.
No fabricated scope/readiness/completion. Source audit records actual32768-task
Intel runtime construction and genuine target fence/R pin before shared queue1
is reused for MTP; physical whole-model behavior remains unverified.

559 CPU cases/73 suites, strict core/off binary lint and normal stable-source
build pass. Frozen source811968c3 SHA256
`cd6ee7aca6300e4096bf03bd4433ce03df022a438f8160a99eca289a671217ed`, independent source/ordinary/frozen hash/mode0500/
features verified. Binary not executed. Shader producers unchanged; previous
source46e695a7 offline365 variants/139 physical checks/32 matrix configurations
remain evidence, not rerun. Exact logs/limits are in the primary receipt.

Next: finish the complete branch source/artifact/math/prompt/sampling/state/
placement/queue/pricing/substitution audit, then protected count1/multirow whole-
model correctness, context, retained/hot-request and failure/all-resource recovery;
only then speed. Rewrite incomplete; CLI compiled-public/one-request and private
caller independently gated. No API/configuration/consumer/frontend change;
required Vashti changes none. Goal active CPU-only. No GPU/adapter/model/inference/
service/install/release action; authorized sudo cache refresh only. Canonical and
inherited cache preserved. No deployment authority or new physical/recovery/speed claim.


## Active Kuno engine handoff — original artifact and runtime Q8 (2026-10-17)

Runtime source `71dceafcd6727bf04cfc85cc6bb3797d08351b97`, isolated
`codex/strata-rust-port`. Preserve the full coupled branch and inherited files.
Read the [exact artifact/conversion receipt](kuno-strata-rewrite-completion.md#original-artifact-and-runtime-q8-conversion-audit-2026-10-17)
and [checkpoint](kuno-native-text-checkpoint.md). The active pointer supersedes
older source/checkpoint claims above; those remain historical receipts.

Actual original target and MTP binding/roles/norms/tokenizer pass CPU inspection.
The native startup loader now reproduces pinned NumPy F32 reciprocal/multiply/
add-before-floor Q8 weight conversion. All nine complete used matrices match
independent packed hashes;24,743 weight codes differ from the old division path.
Thirty actual-vocabulary public tokenizer cases in both special-token modes pass.
No converted model file, prompt/image/token history, API/configuration change or
new GPU owner is introduced. This does not establish physical model correctness.

561 distinct CPU cases/74 suites, strict core/off binary lint and normal build
pass. Frozen source71dceafc SHA256
`0193577d5af95f72e213527adc9124f20c28fa309c44ace70bcf0e874679b187`, independently
verified source/original/frozen hash, mode0500/features; not executed. Shader
producers unchanged; previous source46e695a7 offline365/139/32 remains evidence.
Exact logs, initial failures and limits are in the primary receipt.

Next connected source block: authentic MTP shared-expert fork/join, separate
fault, rounded sigmoid scale and native vec4 ordered-FMA combination, including
actual family/commit/queue/owner/pricing/erasure dependencies. Current MTP path is
still serial there. Then complete the remaining full source/math/prompt/sampling/
state/substitution audit and protected whole-model numerical/context/hot request/
failure/recovery acceptance before any speed trial. No partial rollback based
on incidental timing. Whole rewrite incomplete, goal active CPU-only. No GPU/
model/inference/service/install/release action; authorized sudo refresh only.
No consumer/frontend changes; required Vashti changes none.


## Active Kuno engine handoff — MTP shared fork/join (2026-10-17)

Runtime source `35f0990fe3d89635797870d477ab80e321d47a31`, isolated `codex/strata-rust-port`.
Read the [full source, queue substitution, owner and validation receipt](kuno-strata-rewrite-completion.md#connected-mtp-shared-fork-and-native-combination-2026-10-17)
and [checkpoint](kuno-native-text-checkpoint.md). Preserve the full coupled branch,
canonical application checkout and inherited untracked cache. Prior source71dceafc
original artifact/full nine Q8 matrices/tokenizer receipt still applies.

Actual MTP shared projections/scalar/rounded scale now fork beside routed Q2
experts, with independent private fault and native128-thread vec4 ordered-FMA
combination after true row/fault joins. Graph authenticates actual target or SAME
active draft request; shared code creates no request, capture, verified-R scope,
readiness or completion. Genuine buffers/intervals/INFO/recording/all-peer fence
and original full cleanup remain. Additional actual private owner is32KiB and
is included in admission, prepared-front identity, rebind/erasure/padding/close.
Native32K private plan is62,619,648 bytes. Activation half-away rounding follows
SYCL round; the earlier NumPy weight-pack rule remains distinct.

Explicit backend substitution uses opposite existing compute queues: target side
on draft, draft side on MAIN after any already submitted commit there. MTP front/
router/routed work stays independent. Do not claim identical three-way overlap
or physical acceptance. No fourth hardware queue or uncharged scratch/model copy.
Actual Graph family/admission/submission/final-fence paths use the checked peer.

565 selected CPU cases/76 suites, strict core/off binary lint and normal build
pass. Frozen source35f0990f SHA256 `384fcc563a8499e326274555f7fd74a94a290b03bb75cd949a46a9f74a6b8aff`
independently verified source/original/frozen/mode0500/features. Only its pre-
adapter offline shader entry ran:366 variants/139 physical checks/32 matrix
configurations pass. Exact initial failures, final logs and limits are in the
receipt. No GPU/model inference, numerical device or throughput result.

Next: complete remaining connected source/math/prompt/sampling/state/placement/
queue/pricing/substitution audit, then protected whole-model count1/multirow/
context/retained/hot-request and failure/all-resource recovery before speed.
Whole rewrite incomplete, goal active CPU-only. CLI compiled-public/one-request;
private caller independently gated. Authorized sudo refresh only. No service/
install/release/API/configuration/consumer/UI change; Vashti changes none.


## Active Kuno engine handoff — native routed Q8 and fused producers (2026-10-17)

Runtime source `1a034311407a39d06a714090f82e506b5ef5174a`, isolated `codex/strata-rust-port`.
Read the [full source/ABI/substitution/validation receipt](kuno-strata-rewrite-completion.md#native-routed-q8-and-fused-swiglu-producers-2026-10-17)
and [checkpoint](kuno-native-text-checkpoint.md). Preserve the complete connected
branch and inherited cache. Canonical application work and installed services
are untouched. Earlier artifact/Q8 weight/tokenizer/MTP shared receipts still apply.

Actual routed decode now uses subgroup32 Q8_1 GU preparation and fused native
SwiGLU/Q8 DOWN over original PART/status/planes. Old serial global quantization
and separate stabilized ACT transfer are no longer selected. Original role
stamps, readiness, private owner padding/erase, physical source pointers and
real graph fences remain. No new private GPU buffer or model copy. Frame-owned
workgroup grouping is a reviewed backend substitution; physical numeric/launch
behavior remains unverified. Full installed144-tensor expert inventory matches
all four GU/two DOWN formats.

Actual native registration and offline compiler share NoContraction annotation
of fused products, preserving complete header/IDs/instructions/interfaces and
explicit FMA. This also protects target shared and MTP fused Q8 producers.
Source/byte model artifacts are unchanged; no prompt/image/seed/token persistence.

568 CPU cases/77 suites, strict core/off binary lint and normal build pass.
Frozen source1a034311 SHA256 `d3473d8d509d1ed8e92f98192a7a258160a0008c30f8b0bd01b4c0e2d999a25f`
independently source/original/frozen/mode0500/features verified. Only pre-adapter
shader validation ran:365 variants/139 physical checks/4 product modules/32
matrix configurations pass. Exact failure/final receipts and grouping/exp/math
limitations are in the primary receipt. No GPU/model or speed result.

Next finish the complete connected source/math/prompt/sampling/state/placement/
queue/pricing/substitution audit, then protected count1/multirow/context/retained/
hot-request and failure/all-resource recovery acceptance before speed. Rewrite
incomplete, goal active CPU-only. CLI compiled-public/one-request; private caller
independently gated. Sudo refresh only; no service/install/release/API/config/UI
change. Required Vashti changes none. Source readiness does not prove GPU output.


## Active Kuno engine handoff — native tiled GDN commit (2026-10-17)

**Current CPU-only source:** `39c8d9e1b37700c0adeab0556e2d5c2123dbaffd`, branch
`codex/strata-rust-port`. Read the [native tiled GDN commit receipt](kuno-strata-rewrite-completion.md#native-tiled-accepted-prefix-gdn-commit-2026-10-17).
Actual accepted-prefix GDN recurrence now uses pinned Strata's default state-only
commit kernel: four 32-column tiles per head, 128 threads per block, native four-row-group
FMA order. It no longer dispatches the 512-thread output kernel in state-only mode.
Actual retained key/value/control/original-state roots and paired candidate feed
the same authenticated accepted-count, fault, family-acceptance and role protocol.
No new GPU root/model copy, admission or public API change.

572 distinct selected CPU cases/78 nonempty suites and strict core/off binary
lint pass. Normal stable-source frozen artifact SHA256
`3a18ce74c197b0a78c00d2e7e70edefaea5cd1926a07847b87b82da32585641c`, independently verified source/ordinary/frozen hash,
mode 0500 and default Intel features. Only the pre-adapter offline entry ran:
365 shaders/139 physical-pointer checks/4 rounded-product modules/32 matrix
configurations pass. These are CPU/compiler results, not device execution proof.
Earlier routed Q8/product, original artifact/nine packed weight matrices/tokenizer
and MTP/shared receipts still apply.

**Whole rewrite incomplete; speed gate closed.** Next resolve the multirow
verification-only provisional GDN state store versus ordinary prompt/full-row
candidate publication using actual caller/capture authority; do not skip writes
based on count alone. Then finish the complete connected source/math/prompt/
sampling/state/placement/queue/pricing/substitution audit and protected count 1/
multirow numerical/context/retained/hot-request and failure/all-resource recovery
acceptance before speed. CLI compiled-public/one-request; private caller separately
gated. Goal active CPU-only, authorized sudo refresh only. No GPU/model inference,
service/install/release/API/configuration/frontend action. Vashti changes none.
Canonical checkout and inherited untracked cache preserved.

Read the detailed receipt linked above for exact pinned source, original
root/caller/capture/accepted-count dependencies, current substitution, next
concrete gap and CPU-only test/build/compiler evidence. Runtime change is
`39c8d9e1b37700c0adeab0556e2d5c2123dbaffd`; prior full coupled branch remains selected. All evidence files are
under `/media/jmicko/faststore/kuno-development/strata-gdn-commit-*20261017*`.
A handoff does not authorize GPU use, installation, release or another task.


## Active Kuno engine handoff — deferred GDN and history-only commit (2026-10-17)

**Current CPU-only source:** `7a0489d38bdce98961901759cbe65d95a1db33ba`, branch
`codex/strata-rust-port`. Read the [deferred GDN and history-only commit receipt](kuno-strata-rewrite-completion.md#deferred-verification-state-and-history-only-gdn-commit-2026-10-17).
Actual multirow verified-prefix calls now retain recurrent state in registers and
leave history untouched until acceptance, matching pinned Strata's ALL_OUT and
noncommitting convolution. Ordinary full-row/prompt and count-one calls retain
their actual candidate writes. Both full-row and verification recordings remain
connected, with static state mode in their original branch identity and genuine
replay completion. Accepted-prefix commit now runs native history-only 256-thread
work followed by the previously ported 128-thread/four-column-tile recurrence.
It no longer repeats convolution/QKV normalization during commit.

576 distinct selected CPU cases/78 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`8bf3271d74778c8b927a97eae76db28cbe9eb1fe31d8343c1010d15777394556`, independently source/ordinary/frozen/mode 0500/default-features
verified. Only its pre-adapter offline shader branch ran: 368 variants/139 physical
pointer checks/four product modules/32 matrix configurations pass. These remain
CPU/compiler results, not numerical/physical execution or throughput proof.
No new GPU owner, model copy, request content or public API/configuration change.

**Whole rewrite incomplete; speed gate closed.** This closes the previously
listed provisional GDN recurrent/history write and commit-recomputation source
gaps. Existing paired OLD/NEXT, separate cached K/V, Vulkan math and finite-fault
substitutions still require physical acceptance. Next trace QSA KV/indexer/pooled
history and accepted-prefix handling, then finish the full connected source/
math/prompt/sampling/state/placement/queue/pricing/substitution audit. Protected
whole-model count-one/multirow/context/retained/hot-request and failure/all-resource
recovery acceptance must precede speed. CLI compiled-public/one-request; private
caller separately gated. Earlier original artifact/Q8/tokenizer/MTP/shared/routed
receipts still apply. Goal active CPU-only; authorized sudo refresh only. No GPU/
model inference/service/install/release/frontend action. Vashti changes none.
Canonical checkout and inherited untracked cache preserved.

Read the linked receipt for exact pinned sources, actual caller/recording/state
contract, changed source `7a0489d38bdce98961901759cbe65d95a1db33ba`, tests, initial failures, metadata bounds,
remaining substitutions and required next action. All final evidence is under
`/media/jmicko/faststore/kuno-development/strata-gdn-deferred-*20261017*`.
No consumer changes are required. No handoff grants GPU/service/release access.


## Active Kuno engine handoff — native QSA window indexer (2026-10-17)

**Current CPU-only source:** `e5981a138a58a4209448d8091b1f576c3b1968b4`, branch
`codex/strata-rust-port`. Read the [connected native QSA indexer receipt](kuno-strata-rewrite-completion.md#native-qsa-window-indexer-and-direct-pooled-scoring-2026-10-17).
Actual decode now consumes the native 256-thread serial window indexer, completed
pooled keys in the original persistent owner and direct causal scoring. The
per-query retained-prefix copy, separate mean/norm/publication path and two unused
pooling-buffer clears are no longer selected. Original tail/spare semantics and
accepted-prefix cache/pooled suffix handling remain connected. Original ownership,
full-request erase, source-root retention and conservative admission are preserved.
No new GPU root, model copy, CPU content readback or public API/configuration.

581 distinct selected CPU cases/80 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`1254d3ba1cc8fa9107f8c99060dd07c5aa45ac3aab6d4fe892c8ce7f40cceb01`, independently source/ordinary/frozen/mode 0500/default-features
verified. Only its early offline branch ran: 371 shaders/139 physical-pointer
checks/five rounded-product modules/32 matrix configurations pass. These prove
CPU/compiler acceptance, not physical numerical/model/recovery/throughput parity.
Earlier original artifact/Q8/tokenizer/MTP/shared/routed/GDN receipts still apply.

**Whole rewrite incomplete; speed gate closed.** New source substitution keeps
at most three pre-window FP16 keys in Workgroup storage from the existing packed
cache, versus upstream's separate persistent tail/snapshot. The spare is still
in the existing final pooled slot. Native 256-thread order, math and original
buffer dependencies are connected; Vulkan math and physical replay/state handling
remain unverified. Continue QSA projection/norm/KV/selection/attention and source
placement/data-transfer audit, then complete remaining whole-branch ownership/
math/prompt/sampling/queue/pricing/substitution checks. Protected count-one/
multirow/context/retained/hot-request and failure/all-resource recovery acceptance
must precede speed. CLI compiled-public/one-request; private caller separately
gated. Goal active CPU-only; authorized sudo refresh only. No GPU/model inference,
service/install/release/frontend action. Vashti changes none. Canonical checkout
and inherited untracked cache preserved.

Runtime change `e5981a138a58a4209448d8091b1f576c3b1968b4`; exact source/data-flow/bounds/substitution/CPU/compiler
receipt is linked above. Evidence prefix:
`/media/jmicko/faststore/kuno-development/strata-qsa-indexer-*20261017*`.
The whole coupled branch remains selected. No source handoff grants GPU use,
installation, release or another task message. Required Vashti changes:none.


## Active Kuno engine handoff — native multi-query QSA score (2026-10-17)

**Current CPU-only source:** `31c46e4719457b0c236bcd6684af9f3ec8821934`, branch
`codex/strata-rust-port`. Read the [native window-score receipt](kuno-strata-rewrite-completion.md#native-multi-query-qsa-scoring-2026-10-17).
Actual decode scoring now uses Strata's default multi-query kernel: fixed 256
workgroups, eight warps each, early exit before staging, all active queries in 16KiB
Workgroup memory and each pooled/dead key reused across the window. The real
producer, grid guard, runtime source/status dependencies and retained replay are
connected together. Existing four-product/XOR/positive-head score arithmetic and
native indexer/causal spare/accepted-prefix behavior remain selected.

583 distinct selected CPU cases/81 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`83f99da057bc271ceb36970750c033e658467c7617a891c9ae6d11d74483ac7b`, independently source/ordinary/frozen/mode 0500/default-features
verified. Only its pre-adapter offline branch ran: 371 variants/139 physical-pointer
checks/five rounded-product modules/32 matrix configurations pass. These prove
CPU/compiler acceptance, not physical numerical/model/recovery/throughput parity.
No new GPU heap owner/model copy/CPU content readback/API/configuration. Earlier
original artifact/Q8/tokenizer/MTP/shared/routed/GDN/indexer receipts still apply.

**Whole rewrite incomplete; speed gate closed.** Actual selected attention's
warp score/max/exp/value-FMA/chunk merge structure was traced, without claiming
physical/cache-placement parity. Next audit QSA active scratch clearing and
host/device KV placement/resolution, plus prompt/selection/source substitutions.
Do not port the unrelated native_qsa_score scalar routine into this path: actual
Verifier calls qsa_select::qsa_block_scores, whose default MQ8 mode is now ported.
Then finish the whole connected source/ownership/math/sampling/queue/pricing audit
and protected count-one/multirow/context/retained/hot-request/failure/all-resource
recovery acceptance before speed. CLI compiled-public/one-request; private caller
separately gated. Goal active CPU-only; authorized sudo refresh only. No GPU/model
inference/service/install/release/frontend action. Vashti changes none. Canonical
checkout and inherited untracked cache preserved.

Runtime change `31c46e4719457b0c236bcd6684af9f3ec8821934`; exact source, producer/consumer/grid/ownership/substitution
and CPU/compiler evidence is in the linked receipt. Evidence prefix:
`/media/jmicko/faststore/kuno-development/strata-qsa-window-score-*20261017*`.
Full coupled branch remains selected; no consumer implementation is required.
No handoff authorizes GPU use, installation, release or another task message.


## Active Kuno engine handoff — QSA payload clears and KV-mode audit (2026-10-17)

**Current CPU-only source:** `3d7b0752f34d0ecdaee0817a7483d87ecc926a98`, branch
`codex/strata-rust-port`. Read the [native payload-clear and KV audit receipt](kuno-strata-rewrite-completion.md#native-qsa-live-payload-clear-and-kv-mode-audit-2026-10-17).
Actual native decode no longer clears fully overwritten temporary payloads before
their producers. Both individual and grouped schedules retain all 12 readiness/
append clears exactly once. Prompt/feature-off clearing, initial allocation,
complete owner retention/pricing and full-request owning erasure are unchanged.
The saved reference CLI confirms INT8/32K without a KV residency/growth override;
upstream's fully resident default uses identity pages and a no-op resolver.

585 distinct selected CPU cases/81 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`3a0b8975348577bf8dd0db536810ce4702e67d0e722039abd02f678c12f4aa03`, independently source/ordinary/frozen/mode 0500/default-features
verified, not executed. All WGSL and all four CubeCL producer functions are
unchanged from source 31c46e47; its existing 371 shaders/139 physical-pointer checks/
five product modules/32 matrix configurations remain the compiler receipt.
No shader or live GPU benchmark was rerun for this host-schedule-only change.

**Whole rewrite incomplete; speed gate closed.** A concrete storage-fidelity gap
is now quantified: target CACHE still retains 128 F32-expanded raw index words per
context cell (16MiB/layer at 32K) versus native's bounded three-row tail/snapshot
and current-window raw keys. The existing suffix rollback is correct by its
current design, but this full-history storage is not intrinsically required by
Rust. Next port bounded tail/snapshot/window retention across decode, prompt,
commit, quote/identity/rebind and complete erasure together; do not merely shrink
CACHE_WORDS and break one of its real consumers. Keep existing native indexer/
window scoring and the whole coupled branch. Finish remaining source/placement/
selection/math/sampling/queue/pricing/substitution audit and protected whole-model
numerical/context/retained/hot-request/failure/all-resource recovery before speed.
No new GPU heap owner/model copy/content persistence/API/configuration. CLI stays
compiled-public/one-request; private caller separately gated. Goal active CPU-only;
sudo refresh only authorized. No GPU/model inference/service/install/release/UI
action. Vashti changes none; canonical checkout/inherited cache preserved.

Runtime change `3d7b0752f34d0ecdaee0817a7483d87ecc926a98`. The linked receipt contains the exact overwrite and
clear contract, reference KV evidence, next bounded-tail/history task, unchanged
shader provenance and tested artifact. Evidence prefix:
`/media/jmicko/faststore/kuno-development/strata-qsa-live-payload-*20261017*`.
No consumer/API changes required. No handoff authorizes GPU use, installation,
release or another task message.


## Active Kuno engine handoff — bounded QSA state (2026-10-17)

**Current CPU-only source:** `2f928c9e250ae6b6441f6f73ea03d28f356bfea2`, following
`2282982494cc7876bc87b8758ee7f8a576ff793e`, branch `codex/strata-rust-port`.
Read the [bounded QSA state receipt](kuno-strata-rewrite-completion.md#native-bounded-qsa-indexer-state-2026-10-17).
Actual native decode, full prompt, packed KV, append/selection, accepted prefix,
diagnostic restore, owner identity/rebind/erase and admission now share the
bounded tail/snapshot/original F32 MAX8 contract. Full-context raw index history
is removed:at 32K full cache owners50MiB to34MiB/layer,192 MiB less across 12 layers.
All31 original per-layer owners remain fully charged/erased; three scalar-native-
unused owners carry the bounded state. No new GPU owner or uncharged source.
Prompt-tail grouping matches original three 128-thread groups. Original native
indexer/scoring/GDN/MTP and the whole coupled branch remain selected.

588 distinct selected CPU cases/82 nonempty suites, strict core/off binary lint,
changed formatting and whitespace pass. Stable-source frozen artifact SHA256
`23037328db7eaa4cc93eaf108b1dcbd91d73ac9bdb58882e921114e0823b3e44`, independently source/ordinary/frozen/mode0500/default-features verified.
Only its pre-adapter offline branch ran: 372 shaders/139 physical-pointer compiler
checks/five rounded-product modules/32 matrix configurations pass. These are CPU/
compiler receipts; physical numerical/model/context/replay/recovery/throughput
acceptance remains unproven. Initial 22829824 frozen build is superseded by the
native launch alignment; use the final artifact in the linked receipt.

**Whole rewrite incomplete; speed gate closed.** Next consolidate current source
completeness against actual selected prompt/selection/sampling/queue/ownership
call paths and resolve genuine remaining substitution dependencies. Distinguish
old superseded findings from open work. Then protected count1/multirow/prompt/
context/retained/hot-request/accept-reject/all-resource failure and recovery
acceptance must precede speed. CLI compiled-public/one-request; private caller
independently gated. Goal active CPU-only; sudo refresh only authorized. No GPU/
model inference/service/install/release/frontend operation. Required Vashti/API
changes: none; no protocol/configuration behavior changed. Canonical checkout and
inherited untracked cache preserved. No handoff grants GPU use or deployment.

Evidence prefix: `/media/jmicko/faststore/kuno-development/strata-qsa-bounded-tail-*20261017*`.
The linked receipt records exact changed commits, actual owner/consumer contract,
reviewed commit/storage substitutions, byte quotes, tests and remaining gates.


## Active Kuno engine handoff — grouped replay and native completion (2026-10-17)

**Current CPU-only runtime:** `9bd006d68ea5dd98a0d16d9ecf4dc6394fb26337`, branch
`codex/strata-rust-port`. Read the [current source audit](kuno-strata-current-source-audit.md)
and [grouped replay/completion receipt](kuno-strata-rewrite-completion.md#native-grouped-replay-and-fused-gr-completion-inventory-2026-10-17).
The audit found two actual defects: normal policy selected grouped checks while
the direct replay path/adapters rejected them; native GR's direct sticky-fault
writers expected zero separate GR folds, while the join/report still required 97.
Both are corrected together. Immutable target fold mode separates graph programs;
original prepared stage/fault/state proofs account real recorded GDN/QSA folds.
Native completion uses `[0,36,12]`; legacy retains `[97,36,12]`. No GR math or
status checks are removed. Conservative selected shapes 126 <= original 128 bound.
The prior bounded QSA/native math/MTP/ownership/queue branch remains selected.

596 selected CPU cases / 84 nonempty suites, strict core/off binary lint, changed
formatting and whitespace pass. Frozen artifact SHA256
`044f92abe50b5337f309831564bdbaf3979e5ad2ffd5d0d91a442db34f08726b`, independently source/ordinary/frozen/mode 0500/exact requested-and-resolved Cargo features
verified. Only its pre-adapter offline branch ran: 372 shaders /139 physical-
pointer compiler checks /five rounded-product modules /32 matrix configurations
pass. No live model, numerical/physical replay, recovery or speed acceptance.
No new GPU heap owner/model copy/content persistence/API/configuration behavior.

**Whole rewrite incomplete; speed gate closed.** The new indexed audit records
reviewed entry/loading/cleanup/sampling/target flow and the genuine integration
fix. Finish selected dense/expert/operator/prompt/queue/ownership source review
and retained mixed-request shape/lifetime accounting, then protected whole-model
prompt/count1/multirow/accept-reject/EOS/cap/context/retained/hot-request/failure/
all-resource recovery before any speed measurement. CLI compiled-public/one-
request; private caller independently gated. Goal active CPU-only; authorized
sudo refresh only. GPUs/services/install/release/frontend untouched. Required
Vashti/API changes: none. Canonical checkout and inherited untracked cache kept.

Evidence prefix: `/media/jmicko/faststore/kuno-development/strata-grouped-replay-*20261017*`.
The linked receipt names exact source, consumers, proof boundaries, tests, frozen
artifact and limitations. No handoff grants GPU use, installation or release.
