# Strata prompt-processing gap investigation — 2026-10-19

## Checkpoint and limits

The user authorized a full investigation and implementation preparation without
sudo, also considering general decode gains. Work is in `codex/strata-rust-port`
at `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Three isolated review
worktrees start at1efa8372; their selected commits are integrated here.
Canonical Vashti source and its coordination notes are untouched.

**Installed backend remains60c20e36 / worker SHA1db4664d** with the unchanged
image-enabled gateway36626d30 / SHAd3ed07f4. Its matched4,245-token warm prompt
is43.205s; API/tool/privacy/recovery acceptance is recorded in
[kuno-prompt-processing-profile.md](kuno-prompt-processing-profile.md).
The new coupled candidate below is NOT installed, hardware-validated or timed.
No GPU inference, unload, sudo, driver/service change or authentication workaround
is performed in this investigation. No API or Vashti consumer change.

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
