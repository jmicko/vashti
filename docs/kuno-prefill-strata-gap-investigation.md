# Strata prompt-processing gap investigation — 2026-10-19

## Checkpoint and limits

The user authorized a full investigation and implementation preparation without
sudo, also considering general decode gains. Work is in `codex/strata-rust-port`
at `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Three isolated review
worktrees start at1efa8372; their selected commits are integrated here.
Canonical Vashti source and its coordination notes are untouched.

**Current installed worker: sourcec8db19de / SHAfb31e124**, unchanged image-enabled
gateway36626d30 / SHAd3ed07f4. Protected exact-binary numerical, complete public,
staged thinking/tools/threshold/code-quality/privacy and physical recovery all
pass. Official installation preserves the complete configuration and trusted
model identities; final installed public-request and deployment metadata acceptance pass.

Before this pass the installed worker was60c20e36 / SHA1db4664d with a matched
4245-token warm prompt43.2055s. The selected worker takes24.9658s (**42.2% shorter**),
with warm staged35-token prefix1.2901s versus old4.639s. Toolbox4207/4293-token
prefixes29.5055/29.9196s versus previous46.991/49.630s. These are prompt intervals,
not universal throughput or a controlled fresh Strata comparison. Full history
below preserves rejected variants and their gates. No API/Vashti consumer change.

## Protected GPU continuation (2026-10-19 local time)

All paths below are under `/media/jmicko/faststore/kuno-development`.
Original coupled source5fbc3ca5 / worker SHAb45446bc passes the real B70 matrix
mode: fourteen FP16/BF16 original-weight cases, counts/rows1/33,8/33,35/33,
257/33,16/64,64/64,65/65 in each format. Unchanged IEEE oracle/tolerance,
maximum absolute error4.2915344e-6, no mismatches, complete owner/tail audit and
real close. Receipt: `prefill-coupled-matrix-numeric-20261019/receipt.json`.

Same-source complete public profile passes all three45/4245/4245-token requests,
correct answer, actual per-request erasure and Session close. Repeated warm prompt
**43.4647 ->34.9928s (19.5% shorter)** versus the matched installed profiled
baseline. This is fenced diagnostic time, not a clean HTTP benchmark or a local
Strata A/B. Cold load102.722s. Receipt:
`prefill-coupled-profile-20261019/receipt.json`.

Repeated4245-token diagnostic: expert interval21.4586s, expert enqueue1.8476s,
attention about10.13s, QSA core3.0004s, GDN projections3.2864/1.3344s,
QSA projections0.9563/0.4475s, GR input1.2487s/router1.2175s,
GDN recurrence0.8885s. Nested categories overlap. Warm request rebind4.7/7.7ms
and preflight55–60ms do not explain the multi-second prompt gap.

Actual aggregate counters:26,673 expert groups;262,206,259,200 bytes of local
FP16 image writes;39,834 submitted HOST copy segments /27,315,942,400 source
bytes. These are not all PCIe writes. Enqueue timing ends before `client.sync`;
native graph recording/individual `vkQueueSubmit` calls inside sync can therefore
contribute to the completed expert interval. It is not established pure device
math time. A coherent per-physical-queue submission-batching candidate is under
implementation with original waits/signals/final join retained.

Both completed protected runs restore Intel device usage50,688,000 bytes,
NVIDIA compute processes0, normal admission, all services200 and unchanged zero
driver-fault counts. Compiled-public text/replies stay in RAM; swap/core disabled.

Public-only probe commitsfd6ad094/6e82bcbd/9a18f1f5 add actual production router
and vector-eight dequant checks, independent canonical references and reachable
negative controls (9 router,30 dequant cases) to the existing matrix mode.
Production WGSL, matrix and quantization math are unchanged from5fbc3ca5.
Audited probe artifact `strata-prefill-coupled-operators-audited-20261019`, source
9a18f1f5, SHA256
`493bf946a58fdae4c4b382102dfe116c9a313f2b485e3ad73df1c10694e148f3`.
The initial probe run passes fourteen matrices and six valid router sizes, then
fails BEFORE GPU fault injection: `Upload::floats` rejects the intended NaN.
`694e90d9` repairs only that compiled-public fixture using the existing owned raw
word upload. No production finite check is removed. Historical failed receipt:
`prefill-coupled-operators-numeric-20261019`; recovery passes.

`7e28bc5a` adds bounded per-physical-queue submission batching while preserving
every actual timeline edge, command, queue authority and final joined fence;
27 focused original-source CPU tests pass. See
[exact queue contract and limits](kuno-prefill-transfer-runtime-review.md).
Frozen7e28bc5a / SHAfff19863 complete public profile passes, repeated4245-token
35.0018s, experts21.4536s: no material change from34.9928s/21.4586s. This bounds
the benefit of this particular host-call reduction; it does not isolate GPU math
from every remaining recording/planning cost.

Frozen source694e90d9 / worker SHA256
`28004b38bb9e836fa17e85926dcacd60785a5b929828ca6617106fe7056edc82`, artifact
`strata-prefill-batched-submit-checked-20261019`, now passes ALL **14 matrix +
9 router +30 dequant GPU cases**, original references/tolerances, exact dequant
FP16 bits, real full-owner/tail audits and close. Physical recovery and zero
driver-fault deltas pass. Receipt:
`prefill-batched-submit-operators-numeric-20261019/receipt.json`.

Same694e90d9 artifact clean public check passes all45/4245/4245-token requests,
correct replies and actual request erasure/close. Warm4245 prefixes34.5050 and
34.5316s, versus installed matched43.2055s: **20.1% shorter** on the repeat.
Cold load104.265s; loading is not improved. Recovery passes. Receipt:
`prefill-batched-submit-clean-20261019/receipt.json`.

Original coupled5fbc3ca5/b45446bc isolated private staged API passes preload,
ten-tool streamed weather call, role=tool result, two identical35-token code
requests, same-worker reuse, write_bytes0, denied unprivileged inspection,
graceful unload, subsequent ordinary Ollama and actual physical recovery.
Toolbox4207/4293-token prefixes38.807/40.827s (previous46.991/49.630s);
warm35-token code4.141s,362 outputs/10.719s. Receipt:
`native-prefill-coupled-stage-check-20261019/receipt.json`.
Final694e90d9 isolated staged API acceptance also passes all these gates and real
physical recovery, with its exact SHA28004b38. Toolbox prefixes39.001/41.020s;
warm35-token code4.190s,362 outputs/10.751s. Receipt:
`native-prefill-batched-stage-check-20261019/receipt.json`. No installation or new
live Vashti/browser/image acceptance is claimed.

Further candidate under implementation keeps the32x64/four-accumulator macro
but couples K64 shared operand staging, vector loads and two-stage preparation.
It replaces the old tail operand arrays:24KiB operands +8KiB F32 output =32KiB
per workgroup, within the existing native minimum32912-byte device limit.
Original orderedK16 accumulation, rounded input bits and fault/output owners
remain required. Do not claim asynchronous copy from source ordering alone.
`df904195` adds only a compiled-public64-dispatch/one-fence DQ diagnostic to
separate repeated-fence cost from bounded completed batched work. Favorable
same-weight cache reuse is explicitly not a whole-prompt estimate.

### Shared-memory matrix continuation

Integrated source6dfaf5ef (original isolatedab475149), frozen artifact
`strata-prefill-shared-macro-20261019`, SHA256
`a50e28b3d09a50349265f6a263105cd39563dd99f5af1737401f9be8ea1d63c0`, passes
all14 matrix +39 router/dequant actual GPU cases and complete physical recovery.
Receipt: `prefill-shared-macro-operators-numeric-20261019/receipt.json`.
All402 ordinary /64 matrix offline configurations validate on the frozen binary.
The complete same public profile passes all requests, original answer, actual
request erasure and close, then restores memory/admission/services/fault counts.
Repeated4245-token profile **25.3916s**, versus35.0018s before shared staging and
installed43.4647s: **41.6% shorter than installed profiled baseline**. Experts
16.3217s (was21.4536), attention7.0605s (was10.1416). Load105.515s.
Receipt: `prefill-shared-macro-profile-20261019/receipt.json`.

Exact6dfaf5ef clean acceptance now passes all45/4245/4245-token requests, original
arithmetic answer, request erasure and close. Warm4245 prefixes25.0353/24.8817s;
the repeat is **42.4109% shorter** than installed43.2055s. Cold load105.6827s is
not improved. Receipt: `prefill-shared-macro-clean-20261019/receipt.json`.
Exact private staged API also passes ten-tool streamed thinking/call/result,
same-worker mixed and repeated code, write_bytes0, denied inspection, graceful
unload, ordinary Ollama inference and physical recovery. Toolbox4207/4293-token
prefixes28.1395/30.0307s versus installed46.991/49.630s; warm35-token code3.8968s,
362 output tokens/10.8024s. Receipt:
`native-prefill-shared-stage-check-20261019/receipt.json`.

`faf5c6f6` (isolateda9dc9a96) completes register-first next-stage prefetch, keeping
source6dfaf5ef as a distinct coherent baseline. It loads fixed unrolled A/B local
vectors before the current ordered MMA group and publishes the next shared stage
after the group, before the original uniform barrier. Actual compiled macro
modules scalarize the local arrays;32KiB shared allocation remains. This does
not prove driver register pressure/spills or an asynchronous speedup.

`076d0ec3` (isolated695a34fb) adds an existing-authority MAX3 known-prefix
stepping stone for<=64 fresh rows, with actual fenced R, exact next input teacher
tokens, checked target/draft cursors and wiping RAM. See
[upstream selection and complete short-path limits](kuno-strata-short-prompt-path.md).
It does not claim full upstream MAX8/no-head/device-carry parity. This matters
because upstream's historical~0.7s short prompts normally use verification
windows, not the same per-expert FP16 prefill used in our earlier comparison.

Combined076d0ec3 frozen artifact `strata-prefill-pipelined-short-20261019`, SHA256
`6d3ab0b1921b193d9066e6de5f7b5dcf34a5d2aabeb117f4d977609fed0927f1`, passes
all14+39 GPU numerical cases, complete owner/erasure and physical recovery, and
all402/64 offline shaders. Whole-model profile REJECTS at the first
`short_prompt_chunk`, before any reply; static error `resident graph device
operation`, worker1. Actual recovery passes; this is NOT acceptance for ordinary
inference or installation. Receipt:
`prefill-pipelined-short-profile-20261019/receipt.json`.
The original helper omitted the compiled-public `public_device_error` line from
its failure filter; the next helper included the heading but omitted multiline
`Caused by`. The complete fixed-public diagnostic identifies
`native submission requires original request reuse` in
`prefill-pipelined-short-cause-20261019`. All failed runs restore physical memory,
admission and services with no driver-fault delta. No production guard was removed.

`5aeb11e3` (isolatedf8d3fccb) provides a genuine bounded ShortPrefix phase with
the original runtime nonce, original full closing audit and retained erasure
provenance. It does not invent a generation credit or substitute a new Ready
constructor stamp. `7962fe68` adds compiled-public cold and two warm short
requests. Frozen artifact `strata-prefill-short-scope-20261019`, source
`7962fe688946ab2f08891ed53b2c240cc66c0bea`, SHA256
`76d5f706e17f741901652121d193312ceda9955616f912bdf0810f048b5339be` passes all53
actual GPU controls, unchanged oracles, full-owner/tail audit, close and recovery:
`prefill-short-scope-operators-numeric-20261019/receipt.json`.
It also completes all three45-token arithmetic requests correctly in one loaded
Session, with actual per-request erasure, retained rebind and close. Cold load
104.5667s, cold prefix10.0511s, warm prefixes1.8118/1.7496s; each produces4tokens.
Receipt: `prefill-short-scope-warm-20261019/receipt.json`. This is evidence for the
actual short branch and warm provenance, not yet complete API acceptance or a
matched35-token Strata comparison. Original model/quantization/context stay fixed.

### Final shared-buffer and short-prefix selection

The first fixed-scope combined7962fe68 request run passes complete outputs,
per-request erasure and physical recovery, but warm4245 prefixes40.0054/38.1505s
materially regress from6dfaf5ef's25.0353/24.8817s. Receipt:
`prefill-short-scope-clean-20261019/receipt.json`. The additional register-first
payload is a plausible pressure/occupancy cause; spills are not measured.

Forward selectionc8db19de restores `strata_matrix.rs` **exactly** to6dfaf5ef.
Independent Git/source review confirmsfaf5c6f6 is the only intervening change
in this file. The full K64/vector/paired-shared-buffer kernel is retained, as are
all short-prefix nonce/provenance fixes, router/vector8 dequant, native submission
batching and public checks in other files. This is selection between complete
coherent kernels, not removal of the coupled shared-memory strategy.

Frozen source `c8db19de33b0585278be7380db14dae7323ef560`, artifact
`strata-prefill-selected-short-20261019`, SHA256
`fb31e1244af754dea9585db25b82bc5dafd6dffdfbe512d47cb2c82e98781699` passes all53
actual matrix/router/dequant controls and physical recovery; exact offline402
ordinary/64 matrix shader layouts also pass. Receipts:
`prefill-selected-short-operators-numeric-20261019/receipt.json` and
`prefill-selected-short-shader-validation-20261019.log`.
Exact clean45/4245/4245-token acceptance passes correct replies, request erasure,
close and recovery. Cold load106.1662s; warm4245 prefixes26.5540/24.9658s. The
repeat is **42.2160% shorter** than installed43.2055s. Receipt:
`prefill-selected-short-clean-20261019/receipt.json`.

Initial exact staged API passes preload, streamed thinking/ten-tool call and
role=tool result. Original-worker boundary requests64/65/66 tokens (prefixes
63/64/65) each correctly return323 without truncation or reload:2.3602/2.4359/
4.6688s prefixes. The first35-token merge-code request processes in1.3169s but
hits the old384-output cap, so the historical362-token assertion correctly
rejects the acceptance fixture. This is a changed quantized-prefix answer, not
an engine error; full recovery still passes, and no install is made. Failed
receipt/log directory: `native-prefill-selected-short-stage-check-20261019`.
It does NOT count as complete staged acceptance. A distinct fixed-public quality
fixture must allow a complete answer and check the generated function semantically,
with deterministic repeat, rather than silently weakening the old token count.
No model code may execute as root or persist to disk. This historical checkpoint
is superseded by the accepted quality fixture and installation continuation below.

## Comparison evidence

Original upstream source used for comparison is `Niko1221/Strata` tagv0.1.41,
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, read locally at
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
This is the actual selected SYCL path, not inactive CUDA/HIP code.

- Historical direct Strata API on this B70:26-token arithmetic0.7873s,
 35-token merge-code0.7058s,30-token prose0.6373s, all cached_tokens0.
 Receipt: `/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009/receipts/public-api-check.json`.
 Setup and teardown: [standalone comparison](kuno-strata-comparison.md).
- Current Kuno installed same35-token code prompt:4.639s warm prefix. Different
 run settings/timing scopes mean this is a strong lead, not a controlled A/B.
- Upstream reports980–1,002 prompt tokens/s for4,095 tokens on B70/original
 IQ3_S with4,096-token chunks and PLE RAM, about4.1–4.2s. This is NOT our local
4K measurement. Our fixture is4,245 tokens and uses a different workload and
PLE configuration. [Pinned Intel report](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/INTEL.md#arc-pro-b70-with-a-model-that-does-not-fit-flash-next-iq3_s-2026-10-07-branch-f-141-intel).

The accepted Kuno fenced profile has25.761s experts,13.430s attention, with only
2.273s spent enqueueing the expert interval. This points primarily at completed
device/queue work, not hidden repeated full-model hashing or CPU expert execution.
The fenced intervals are diagnostic and overlap in nested categories.

## Concrete gaps and changes

### Per-expert dequantization: fixed preparation cost

`strata_prompt_experts` previously decoded two adjacent FP16 values per invocation,
calling the original record preparation twice, and stored oneu32. Selected
Strata IQ dequantization emits aligned runs of eight values per thread while
sharing metadata. See pinned `sycl/src/kernels/cuda/iq_kernels.dp.cpp` around
`dequant_flat_kernel`/`iq_dequant_gu_f16`, especially the eight-value vector stores.

| Accepted public profile | Active expert groups | Completed expert interval | Approx. routed rows/group |
| --- | ---: | ---: | ---: |
|45-token first request |5,840 |4.988s |3.6 |
|4,245-token warm repeat |26,673 |25.761s |76.4 |

Rows/group rise about21x while time/group changes only13%. First-request
compilation can distort the short sample, so this is a strong hypothesis rather
than a measurement of dequantization alone. The original9,830,400-byte FP16
GU+DOWN images imply57,409,536,000 and262,206,259,200 output bytes respectively.
These are **local GPU writes, not PCIe traffic**; Strata also creates these images.
The target is efficient production of the same bytes.

`5d4134e3` vectorizes this complete consumer chain: one prepared32-code record,
two packed quartets, eight decoded values and one alignedvec4<u32> store. One quarter
as many invocations per original image, without claiming fourfold speed. Original
signed codes, row order, IQ2_S16-value subscales, Q2_064-value block halves,
F32 arithmetic and FP16 rounding remain. Full raw/rounded finite checks, original
owner extents, slot alignment and terminal fault128 remain. On failure all eight
values are zeroed rather than just the failing pair; the terminal fault still
rejects the request. `5fbc3ca5` removes redundant constant-only test assertions.

The same commit adds compiled-public-only aggregate expert groups, FP16 output
bytes and actual submitted HOST copy segment/byte counts. Private worker never
enables the collector. No IDs, route values, offsets or contents enter a receipt.
New `request_rebind_begin/complete` phase markers separate fixed rebind cost in
compiled-public phase timing; public wire metrics and logging remain unchanged.

### Prompt routing lacked token parallelism

Actual SYCL `prefill/kernels.dp.cpp::route` launchesceil(T/8)256-thread workgroups,
eight SG32 tokens each. Rust `strata_prompt_route.wgsl` used one256-thread
workgroup looping through all tokens in groups of eight.

`252f6039` preserves each token's exact subgroup arithmetic and gives each
workgroup eight disjoint tokens. A separate ordered finalize dispatch checks the
original fault word before publishing the valid header. Optional aggregate usage
retains its original serial route order. Readback still observes all producer
writes through the actual same-owner queue dependency. No early header or
cross-workgroup barrier assumption is introduced. One ordinary shader is added.

This may affect long prompts materially; it has not been timed in isolation.
Decode already parallelizes its tokens, so this change does not imply decode gains.

### Matrix primitive: move beyond one tile per workgroup

True Strata SYCL `prefill/gemm.dp.cpp::Gemm::{bf16,f16}` dispatches oneMKL GEMM.
Our accepted primitive has one16-lane subgroup and one8x16 output tile with one
accumulator, without reuse across neighboring tiles. Direct global loads helped
but did not supply a mature GEMM's tiling and instruction overlap.

`4ab474d3` (integrateddf67029b), fixture repair`5148a53a`, adds a32x64 macro tile:
four16-lane subgroups, each computing16x32 outputs with four independent
accumulators. Four operand loads feed four products, twice the operand reuse of
four independent tiles. Each output keeps orderedK16 accumulation, FP16/BF16
inputs, F32 accumulator and denormal requirements. Complete tiles use original
global buffers; partial token/feature tiles are zero-masked in bounded shared
memory with uniform barriers. Original small path remains forcount<32 orrows<32.
Expert route scatter, full status plane and finite/fault checks are preserved.

Physical matrix fixtures expand eight -> fourteen: original cases plus FP16/BF16
(16,64), (64,64), (65,65), covering a complete macro tile and both output tails.
No numerical tolerance is relaxed.

### Why the prior larger matrix variant was rejected

The previousd06c03d4 protected matrix mode returned a generic failure when it
first selected a128-thread kernel. Its earliercount1/8 cases used the old kernel.
Source investigation finds `kuno_cooperative_subgroup` recognized ONLY literal
LocalSize16x1x1. Native replay rejects cooperative modules without this metadata.
The previous result is therefore **not established arithmetic divergence**; the
launch-geometry contract itself was broken. Existing small status-plane tests
missed it; full offline matrix validation would have caught it.

`b64aa80f` (integratedd1b95a5e) accepts recognized FP16/BF16/F32 subgroup-scoped
8x16x16 cooperative types with bounded one-dimensional multiples of16 up to256
threads. It verifies a single compute entry, matching literal LocalSize, known
constant types/scope and no conflicting subgroup mode. Both ordinary HAL and
native replay share the selector; normal SG32 policy/device capability guards
remain. New CPU tests use actual small/macro kernel definitions rather than an
incomplete synthetic matrix marker.

## Things already implemented or outside the prompt number

- Real independent compute/copy queues and interval dependencies already exist.
- Eight raw staging slots/two FP16 image slots match Strata's selected default
 for this model. Its96/384-slot cross-layer streaming ring defaults on only
 when nonresident share is below10%; original Flash-Next IQ3_S does not qualify.
 Do not rewrite this scheduling merely because the inactive path exists.
- Prompt expert staging uses retained HOST/GPU sources. Binding checks perform
 file metadata checks, not repeated complete model hashes. PLE disk reads are
 separate; exact submitted expert PCIe volume is now instrumented for public tests.
- Actual target/draft erasure runs after `request_private_erasure`; it affects
 complete HTTP time, not reported prompt/eval time. Warm rebind and request PLE
 reader initialization are inside prompt time. Rebind restores/checks small maps,
 without reloading the model.
- Kuno reads a larger router map and uploads256KiB per layer using eight32KiB
 private scratch operations, unlike Strata's compact maps. This is real extra
 work but enqueues in the native graph, not eight synchronous host waits. Compact
 map upload remains a secondary candidate after measuring the coupled changes.

## General inference opportunities

The implemented routing, eight-value preparation and macro selection primarily
improve prompt processing. Ordinary small decode/MTP batches keep the accepted
small matrix path; do not advertise a decode improvement from these changes.
The corrected subgroup contract and reusable matrix tiling can support larger
verification/prompt batches and other models. Larger speculation must still earn
an end-to-end gain, since draft acceptance and repairs change the result.

A stronger Rust GEMM option is [CubeKv0.2.0](https://github.com/tracel-ai/cubek/tree/v0.2.0),
MIT/Apache-2.0, compatible with our CubeCL0.10. Inspected source:
`/tmp/kuno-cubek-v0.2.0-audit`, commitf91effcf2cb18726c4a5e393bedd40fea7d1302a.
It offers structured static GEMM over existing client/Handle storage with FP16/
BF16 inputs and F32 accumulation/output. It is NOT added as a dependency here.
Important integration constraints for a later comparison:

1. Its normal API selects hardware maximum subgroup width, commonly32. Use the
 real supported16 width through lower-level routine settings; never forge the
 device's capabilities. Its16xN workgroups need a separate validated geometry
 contract, not silent launch-dimension rewriting.
2. Preserve dense finite/fault handling and expert route/status scatter. Generic
 GEMM alone omits those guards; use authentic custom output views or checked
 member-contiguous scratch/scatter, with all memory priced and erased.
3. Use static launch; no private-input autotuning or persisted tuning content.
 Existing worker no-swap/core/cache/log policies remain mandatory.
4. Validate original IEEE/denormal behavior, actual ownership and request erasure
 before comparing performance. A source-level orderedK16 loop is not GPU proof.

The pinned Strata report's fused dequant/XMX experiment is slower than separate
dequantization+oneMKL on this GPU. It is not selected as a priority. Future cold
load work needs its own source/copy/first-use profile; these changes do not claim
to shorten the current~102-second model load.

## Validation and next protected run

Historical preparation plan, completed by the accepted selection and receipts
above/below. Do not repeat these gates without a new change or unresolved failure.

Artifact base: `/media/jmicko/faststore/kuno-development`.
Current CPU evidence:

- 10 actual matrix/compiler/coverage/status tests pass.
-29 prompt/ownership/vector-packing tests pass, including canonical independent
 quant decoders over all six supported GU/DOWN formats and original WGSL vector
 shape/finite/fault guards. CPU tests cannot prove actual device dequantization.
-5 router/map tests pass, including actual Naga/SPIR-V compilation and1/8/9/35/
4096-token coverage.
-13 engine/controller/request-state tests pass.
-8 standalone original HAL selector tests pass with `rustc --test -D warnings`.
- Strict selected production worker Clippy passes. A broader `--tests` Clippy run
 also checks both aliases ofmain.rs and fails existing test-only warnings such
 asduplicate_mod, old assertions and loop style. It is not reported as passing;
 new constant-only assertions were removed. No unrelated lint refactor is included.

Logs: `prefill-investigation-combined-{matrix,prompt,router,engine}-tests-20261019.log`,
`prefill-investigation-selector-tests-20261019.log`,
`prefill-investigation-worker-clippy-20261019.log`. The initial router filter
`strata_global_moe::tests` selectedzero tests; only the later real
`strata_global_moe::route_plan_tests` run is acceptance evidence.

Frozen release source `5fbc3ca5df9a51e7449ead13860af0f85447721a`, artifact
`strata-prefill-coupled-investigation-20261019`, SHA256
`b45446bcd45af1fd2607938f1fda79ad2944163d307f478f8aefe82ae692b157`.
The existing builder verifies committed unchanged sources and the resolved feature
set `strata-rust-port,strata-intel-core,strata-gpu-adaptive-usage`, then freezes the
executable mode0500. Release build passes. Offline validation of the actual
frozen binary passes **402 ordinary shader variants and64 cooperative matrix
configurations**, using the pinned matrix-capable validator; see
`prefill-investigation-offline-shaders-20261019.log` and
`prefill-investigation-build-20261019.log`. No Vulkan device or GPU inference is
constructed by that mode. No device speed or numerical claim.

`prefill-investigation-cpu-checkpoint-20261019.json` records unchanged installed
worker/gateway hashes and read-only Kuno/Ollama/Vashti HTTP200. It is NOT a new
physical-memory, fault-counter or GPU-recovery audit. No test client or staged
service was launched. The isolated review worktrees are queued for archive;
all selected commits are already integrated. Inherited untracked Python cache
and the canonical checkout are preserved.

Prepared root clients (syntax checked, NOT run):
`check-prefill-coupled-matrix-20261019.py` requires all fourteen exact original
matrix cases; `check-prefill-coupled-profile-20261019.py` saves only numeric public
phase/counter metrics and original answer-correctness booleans. They preserve
no-swap/core, actual erasure/close and physical recovery. Do not reuse the old
8-case assertion or edit historical passed receipts.

When the user is home and sudo/GPU availability is confirmed:

1. Accept actual GPU numerical routing/dequant/matrix results and injected-fault
 header behavior. Full matrix mode alone does NOT validate the new dequantizer.
2. Run complete fixed-public varying/chunked requests and checked erasure/close;
 require actual physical memory recovery and zero driver-fault changes.
3. Run matched warm short and4,245-token clean/profile fixtures with the coupled
 candidate. Use new byte counters and rebind phases to attribute remaining cost.
 Distinguish first-use compilation, load, prompt, decode and final erasure.
4. Run private staged thinking/tool-result/streaming and ordinary Ollama recovery
 before any official installation. Preserve gateway/image/configuration state.
5. Record exact accepted hashes, measurements and limitations in coordination.
 No release, automatic Strata restart or service update is authorized here.

## Accepted API and installation continuation (2026-10-20 local time)

Exact selected workerc8db19de / SHAfb31e124 passes
`native-prefill-selected-short-stage-quality-check-20261019/receipt.json` and its
physical recovery records. Original gateway, ten public tool schemas, streamed
low thinking/weather call, role=tool final answer, threshold total64/65/66 (known
prefix63/64/65), same-worker35-token code repeats, request erasure, write_bytes0,
inspection denial, graceful unload and ordinary Ollama afterward all pass.
Both code answers end normally at501tokens and are byte-identical. Each passes
155 deterministic merge cases plus strict function-only AST and time/space
explanation checks in a resource-bounded nonroot child. Generated source stays
in RAM/pipes only. The checker itself passes3 positive and13 rejection fixtures,
and the real swap0/core0 child acceptance in `public-merge-validator-child-20261019.log`.
This explicitly replaces an obsolete362-token response fixture, not an arithmetic
reference/tolerance or an engine safety guard. Both failed fixtures remain recorded.

Official transaction via `install-native-prefill-selected-short-quality-20261019.py`
requires the exact53 numerical checks, complete clean45/4245/4245 requests, all
API/semantic/boundary gates and their actual physical recovery before modifying
anything. Installed hashes match original frozenfb31e124 worker andd3ed07f4
gateway; full configuration equality and trusted model identities pass. Backup,
installation receipt and official rollback script are root-only under
`/var/lib/kuno-prefill-selected-short-quality-20261019`.

Final installed acceptance passes:
`native-prefill-selected-short-quality-installed-public-check-20261019/receipt.json`
and `native-prefill-selected-short-quality-installed-final-check-20261019/{receipt,host-final}.json`.
Two complete identical501-token code answers each pass155 semantic cases through
the actual installed listener. Cold load105.8716s, cold prefix9.5601s; warm prefix
**1.4305s versus old4.639s (69.2% shorter)**. Warm output501/15.2196s (32.918 output
tokens/s), full request21.3211s. Different response length prevents comparing
full-request time directly to the old362-token answer. No new decode gain claimed.
Actual DynamicUser/model mount, swap/core0, worker write_bytes0, denied unprivileged
inspection, zero OOM/driver faults, healthy Kuno/Ollama/Vashti and normal admission
all pass. Native Intel weights intentionally stay warm (33,314,902,016B device
usage); NVIDIA compute processes0. All test clients/staged services and the
identified task sudo-refresh loop are stopped. No reboot/reset was performed.

No API, protocol, configuration schema or Vashti changes. Retain original
image-enabled gateway/runtime when packaging this worker; select release source
c8db19de or its descendants that preserve this selection, not experimentalfaf5c6f6
or076d0ec3 alone. This is a local installation, not a package/release. No new live
Vashti/browser/image acceptance. Startup, broader soak, above32K context, complete
Strata MAX8/no-head short path and remaining large-prefix GEMM gap remain open.

The main remaining measured long-prefix costs in the same selected K64 matrix
implementation are experts16.32s and attention7.06s (6dfaf5ef diagnostic, nested
slices overlap). Larger prompt GEMM/data reuse remains the strongest long-path
lead; short MAX8/no-head/device carry is a separate bounded follow-up. Do not
select the slower register-first successor merely because its source looks more
asynchronous. No new local controlled Strata comparison was performed.
