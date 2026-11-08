# Native MoE text inference: research and proposed implementation

Research date: 2026-10-05. **Proposal only; no engine or API change is implemented.**
Repository baseline: `ae3e70d` (published beta.3 integration). Research lives on
`codex/kuno-text-research` in an isolated, already-attached Kuno worktree.

The user requested research on FreeToken and Strata, followed by a plan for a
Kuno-owned Rust engine on this machine. No upstream install/build scripts were
executed, model weights downloaded, inference run, services changed, or GPUs
loaded. Read-only hardware inventory and source inspection are the evidence here.
Published throughput figures are upstream measurements, not Kuno benchmarks.

This proposes Qwen3.8-Flash-Next as the next engine target. The earlier
[Bonsai plan](kuno-bonsai.md) remains a separate model implementation; implementing
Qwen does not implement Bonsai's activation transforms or packing. Its shared
naming direction is updated for the user's explicit request to reuse Ollama
artifacts and default verified models to Kuno, with an explicit Ollama alias.

## Conclusion

A focused native engine is feasible. The useful advantages come from controlling
model execution, memory residency, expert scheduling and synchronization. Rust
alone provides no throughput advantage. Rust should own the loader, graph,
request state, scheduler and sampling; narrowly scoped GPU kernels or math-library
FFI remain compatible with that goal. Wrapping an entire Strata/llama.cpp graph is
a reference or temporary adapter, not completion of the native-engine milestone.

The proposed first target is the **full, 512-expert Qwen3.8-Flash-Next**, using
the existing Ollama Q4_K_M GGUF without a second weight copy, initially on B70.
Use Strata as a source of model-specific operators and FreeToken as a source of
measured CPU/GPU scheduling. Add heterogeneous execution and MTP after the
single-device graph and lifecycle are proven.

## Sources and exact revisions

| Source | Revision inspected | Use |
| --- | --- | --- |
| [FreeToken](https://github.com/FlashML-org/FreeToken/tree/189c8ad0c3350e30252209d40780274976cb7527) | `189c8ad0c3350e30252209d40780274976cb7527` | Runtime, expert scheduling, prefill/decode, loader and support limits |
| [Strata](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab) | `6f32ec070f23ced9f50e704d854d775da52591ab` | Qwen graph, PLE, quantized kernels, verification, experimental Intel port |
| [Official model configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) | `de4b8e4d43b917e7706784d8bb445c9af86a3540` | Actual architecture and dimensions |
| [Transformers reference](https://github.com/huggingface/transformers/blob/0a896aa41bba78f92338db21cc468fe043888657/src/transformers/models/qwen4_exp/modular_qwen4_exp.py) | `0a896aa41bba78f92338db21cc468fe043888657` | Independent graph/operator reference; inspect inherited operators too |
| [FreeToken paper](https://arxiv.org/html/2608.16157v1) | v1 | Rationale for bandwidth-adaptive execution; current code differs in calibration |
| [Qwen architecture paper](https://arxiv.org/html/2608.30320v1) | v1 | QSA, gated residual and off-accelerator n-gram design |

FreeToken's top-level license is Apache-2.0; Strata's is MIT. Imported kernels,
runtime libraries, packs and model weights have their own provenance. The
official model uses Qwen Community License 1.0. Record the exact artifact's
license and notices when defining its downloadable collection; a repository
license does not cover every dependency or model.

## What the model actually requires

The official configuration identifies `qwen4_exp`, with 48 layers: 36 Gated
DeltaNet layers and 12 Qwen Sparse Attention (QSA) layers. Hidden width is 2560;
there are four gated residual branches, 512 routed experts per layer, ten
selected routed experts per token, and a shared expert. The n-gram table is
additional to the 125B backbone; MTP is another separate component.

Implement these components together:

- Correct gated residual read/write gates and normalization, rather than an
  ordinary single residual stream.
- GDN recurrence, short convolutions and persistent state; prefill and decode
  must implement the same graph.
- QSA's indexer, block compression, causal selection and attention. The configured
  budget selects up to 512 complete four-token blocks plus the incomplete tail.
  Sparse attention limits work; it does not eliminate storage of old keys/values.
- PLE n-gram hashing, artifact-specific constants, row decoding, token history,
  convolution and placement at the configured layer. Match integer overflow,
  EOS boundaries, missing predecessors and nibble order against the artifact.
- Exact router/top-k normalization, all selected experts and shared-expert gates.
- Tokenizer, chat template, thinking/tool delimiters and sampling.

The Transformers reference supplies an independent check for QSA, gated residual
and PLE, but precision/format differences must be accounted for when comparing a
quantized GGUF. Derive constants from artifact metadata rather than comments or
marketing names. Strata's actual geometry is 36 GDN + 12 QSA; a header comment
elsewhere incorrectly says 48 GDN + 12 QSA.

## FreeToken: mechanisms to learn from

FreeToken combines a host expert pool with a bounded GPU cache of complete
experts. Cache hits execute on GPU. Missing experts can be transferred for GPU
execution or computed in place on CPU, with partial outputs merged. It preserves
the routed expert set; reduced precision and floating-point reduction remain
separate correctness questions.

The paper's bandwidth ratio is a starting model, not a constant to copy into
Kuno. Current source calibrates CPU execution and PCIe transfer together. They
share host memory bandwidth, so independently measured peak rates can predict an
unrealistic split. Measure the actual quantized CPU kernel, transfer path, batch
size and overlap. Our AVX2 CPU and asymmetric PCIe paths make this especially
relevant.

Other useful mechanisms are a global LRU expert cache, device-side routing and
cache bookkeeping to avoid a host synchronization at every layer, and separate
double-buffered full-layer prefill rather than forcing a decode cache policy onto
large prompt batches. Runtime memory reconfiguration belongs at synchronized
safe points and must preserve request state and correct slot ownership.

The decode implementation reads mapped, pinned host aliases through a fused GPU
gather; prefill instead uses full-layer transfers and two buffers. These need
separate backend implementations. Current reconfiguration is idle-only and
single-rank, not an established automatic recovery loop. Its integer fetch
balancing can choose zero fills, unlike the paper's at-least-one-fill description.

Current limits matter:

- Qwen3.8 is explicitly implemented; this is not merely Qwen3.5 compatibility.
- Its Qwen loader pins approximately **47.7 GiB of FP8 PLE** in host memory before
  expert pools. Copying that memory layout is inappropriate for our 64 GB host.
- Its Qwen path rejects tensor parallelism above one. No mixed CUDA/Intel
  implementation was found. CUDA is the mature path; ROCm work does not establish
  Intel support.
- CPU co-execution depends on packing. The block-FP8 expert path is GPU-only;
  NVFP4 has a CPU path. Q4_0 CPU kernels add Q8 activation quantization. These
  cannot be advertised as identical precision to arbitrary GPU/reference paths.
- Cross-request prefix reuse is useful for agents. The proposed native worker
  clears prefix/request state at idle. Existing managed Ollama has its own
  transient prompt-cache lifecycle; this proposal does not change that runtime.

Loader evidence: [Qwen weights](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/models/qwen4_exp/weight.py).
Primary implementation evidence:
[overlapped calibration](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/bench_profile.py#L156-L190),
[cache and transfers](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/offload_cache.py),
[hybrid admission](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/offload_kernels.py#L313-L410),
[MoE phase dispatch](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/layers/moe.py#L298-L396),
[CPU execution](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/moe/cpu_executor.py),
[FP8 limits](https://github.com/FlashML-org/FreeToken/blob/189c8ad0c3350e30252209d40780274976cb7527/python/freetoken/layers/quantization/moe/fp8_block.py#L20-L32).

## Strata: mechanisms to learn from

Strata has a purpose-built Qwen graph and quantized kernels. Its PLE reader
accesses only the required table rows, with bounded caching, asynchronous reads
and batched page deduplication. In its IQ4_NL layout, sixteen 90-byte rows supply
one token's 2560 embedding values; the complete table occupies about 28.8 GB on
disk. Keeping that whole table pinned in RAM is unnecessary. Direct I/O and mmap
are alternatives to measure, not universal winners.

Its expert paths use CPU/GPU co-execution, hot expert residency and separate
prefill scheduling. A low-RAM mode keeps host copies principally for experts
that are absent from GPU, avoiding duplication of every cached expert. That is
valuable if combined VRAM can hold most of a suitable quantization.

Qwen-specific opportunities include fused quantized prefill, GDN recurrence and
convolution, gated residual operations, QSA block selection and sparse KV
gathers. Some CUDA optimizations require newer NVIDIA architectures: thread-block
clusters are not a 3060 optimization. Activation/KV quantization must be explicit
and validated, not hidden inside a reported speedup.

Strata's speculative verification samples the target at each position and accepts
the longest matching draft prefix. Coupled draft sampling increases the match
rate; drafts never replace unchecked target outputs. Its commit logic handles
GDN/convolution, QSA tails and PLE history, not just an ordinary KV truncation.
Porting only the draft generation would be incorrect.

Important qualifications:

- [Intel support](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md)
  exists as an experimental SYCL port, including community B70 measurements.
  The pinned latest port is compile/CPU-kernel tested by maintainers, but its
  latest version has not been validated on Arc hardware. We have not run it.
- The headline Coder results use a model with **256 experts per layer**, removing
  half the original experts. Swift is a fine-tune. Neither is a matched benchmark
  of the full base model against Ollama.
- An earlier same-GGUF B70 comparison reported 20.2–20.8 tokens/s with only a
  suffix drafter, versus 24.2–26 for llama.cpp SYCL; adding MTP reached 42.7.
  These are historical port measurements, not a latest-version comparison.
  They demonstrate why draft acceptance and residency must be separated from
  claims about superior kernels or implementation language.
- Changes in CPU/GPU placement, expert grouping, prefix reuse and speculative
  batching can change floating-point rounding and greedy output. Validate
  numerical tolerances; do not infer bitwise parity from plausible responses.
- Its saved conversation checkpoints and learned expert-use profiles conflict
  with Kuno's content-free persistence. They must not be imported.

Primary implementation evidence:
[geometry](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layout.hpp),
[PLE contract](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/ngram.hpp),
[PLE reader](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/ngram/ple_reader.cpp),
[expert sources](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/expert_source.hpp),
[prefill staging](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp),
[target verification and commit](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/verify.cpp#L1282-L1311),
[draft acceptance](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/generate.cpp#L8035-L8104),
[model distinctions](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md),
[numeric repeatability](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/DETAILS.md),
[Intel measurements and limitations](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL.md#L156-L176).

## Existing Kuno foundation

Kuno currently supervises Ollama for text; it does not already contain a native
Rust text graph or serve Bonsai. Reuse [placement](../apps/kuno/src/placement.rs),
[scoped reservations](../apps/kuno/src/reservations.rs),
[gateway streaming](../apps/kuno/src/gateway.rs) and the protected image-worker
packaging/cleanup patterns. Native model registration, text state/operators,
host RAM admission and measured inference peaks are new work. Managed workers'
current RAM-only load observations are not native MoE peak profiles. The global
gateway admission permit still serializes inference; per-device reservation
support alone does not create independent-device concurrency.

## This machine and proposed placement

Read-only inventory on 2026-10-05:

| Resource | Evidence | Consequence |
| --- | --- | --- |
| CPU | Ryzen 9 5900XT, 16 physical cores / 32 threads; AVX2, no AVX-512 | Use an AVX2 expert implementation and benchmark physical-core thread counts |
| Host RAM | 62.7 GiB physical; about 48 GiB available in this snapshot | Budget host weights, DMA staging and worker state alongside other programs |
| NVIDIA | RTX 3060, 12 GiB; chipset path limited by a PCIe 3.0 x4 bridge | Cold expert transfers may favor CPU computation; copy bandwidth is unmeasured |
| Intel | Arc Pro B70, nominal 32 GB; physical upstream link PCIe 3.0 x16 | First primary target for a larger resident expert cache |

Idle endpoint link speeds vary. The B70's internal endpoint/bridge reports x1,
but the external upstream path reports x16; do not treat the internal report as
a bandwidth measurement. No transfer probe was run. The two cards do not form
one 44 GB allocator, and CUDA-to-SYCL peer transfers/coherence are unverified.

Use the existing artifact described below as the baseline. It is larger than
host RAM and larger than combined VRAM; disk-backed PLE and bounded expert
caches are required. Avoid full host expert mirrors. Host/GPU duplication,
context state, prefill scratch, any draft head, allocator fragmentation and
safety reserves must all appear in the budget. Do not choose a format merely by
multiplying the advertised active parameter count by its bits. A smaller
quantization is a later, explicit storage/quality tradeoff, not a replacement
silently downloaded or requantized during this work.

Default placement should remain eligible to leave the 3060 for images while text
runs on B70. Test dual-card modes against that useful single-card baseline; using
both is an optional cost/capacity decision.

## Reuse the downloaded Ollama model

Read-only service configuration, `/api/tags`, metadata-only `/api/show` and the
local manifest identify this installation's store as
`/media/jmicko/faststore/ollama`, configured through `OLLAMA_MODELS`.
[Ollama documents the override](https://docs.ollama.com/faq#where-are-models-stored);
do not assume its default Linux directory or the interactive user's home.
No weight payload was streamed or hashed during this research.

| Component | Observed identity / footprint |
| --- | --- |
| Public model | `qwen3.8-flash-next:125b-a6b-q4_K_M` |
| Manifest digest | `f79436a1cc83fd1a9b7752cf62247a2e7ad9fd7775c67230da4254fcc1d07f01` |
| Main GGUF blob | `sha256:89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75`; 119,150,722,112 bytes |
| Separate vision projector | `sha256:375f156fdc1232f994c42f43813861fac4fdc791f0440a36c85e87b6907a7eee`; 907,542,784 bytes |
| Main architecture | `qwen4exp`, 48 layers, 512 experts; 1,224 listed tensors |
| Expert tensor payload | 75.0 GiB: Q4_K gate/up; Q5_0 down in 24 layers and Q8_0 down in 24 |
| PLE tensor | Q5_0, shape `[160, 320001536]`; 110 packed bytes/row, 35,200,168,960 bytes (32.783 GiB) |
| Remaining main tensor payload | Approximately 3.175 GiB; the full file adds headers/alignment |
| Other required types | Q6_K, BF16, F16 and F32, in addition to the expert/PLE types |

These are manifest sizes and computed tensor footprints, not measured resident
memory. The separately downloaded `qwen3.8:27b` aliases are a `qwen35` graph and
are not this target. The snapshot has no MTP/nextn/draft tensor names or metadata;
do not promise MTP without validating a compatible draft module separately.
The projector establishes an available vision artifact, not a working native
vision implementation.

Strata's actual [PLE implementation](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ngram.cpp#L190-L219)
supports Q5_0's 110-byte rows, despite stale IQ4_NL/FP8-only comments. Port and
verify that decoder/stride without converting the 35 GB table to IQ4_NL.
Its compatibility packs convert some small projections to BF16; the native
loader must implement the original mixed quantized tensors, or explicitly
account for any in-memory conversion's rounding. A complete cache miss for ten
experts through all 48 layers reads about 1.465 GiB of expert weights per decoded token before reuse,
excluding shared/dense work. This is an arithmetic upper-miss scenario, not a
prediction of normal routing or disk throughput. Measure cache hits and storage
latency before promising an advantage over Ollama. B70-only execution may need
cold expert disk reads; dual-card complement residency may reduce them but must
include host headroom and all execution buffers.

Proposed storage contract:

- Discover the local managed Ollama store, allow an explicit root override, and
  bind a complete manifest snapshot to validated local blob descriptors. A
  remote Ollama catalog alone cannot provide local weight access; report that
  limitation rather than downloading an implicit duplicate.
- Borrow blobs read-only. Keep Ollama as owner of manifests, downloads and
  deletion; Kuno does not rewrite, relocate, rename, hard-link or garbage-collect
  borrowed weights. Grant the protected worker only the required read access.
- Validate architecture, quantization, tensor ranges/alignment, artifact-specific
  PLE constants, tokenizer/template/parameters and any projector/draft modules.
  Preserve distinct tokenizer and PLE boundary IDs. Reject unsupported packing
  or missing required transformations rather than treating ordinary GGUF
  recognition as complete model compatibility.
  Integrity verification must be bounded, explicit and cached by artifact
  identity; never hash a 120 GB model on every inference request.
- Keep open read-only descriptors for one immutable snapshot during a job.
  Refresh catalog bindings when Ollama updates/removes a manifest. Do not mix
  new tokenizer/parameters with old weight descriptors. An ordinary pull/delete
  through Kuno must coordinate active native workers and invalidate affected
  registrations/profiles. Out-of-band changes require detection before new work;
  POSIX unlink survival alone is not a complete update policy.
- Disk-backed model pages are public weight data. Private request buffers,
  token/PLE keys, routing and I/O submission state remain under no-swap/no-core
  protections. Do not persist content-derived read traces or expert hot sets.

## Proposed model names and dispatch

User-requested direction, **not implemented in beta.3**:

| Requested model | Intended route after native acceptance |
| --- | --- |
| `qwen3.8-flash-next:125b-a6b-q4_K_M` | Kuno native engine using the borrowed artifact |
| `ollama.qwen3.8-flash-next:125b-a6b-q4_K_M` | Same artifact through managed Ollama |
| Other ordinary Ollama model names | Existing Ollama route until a native adapter is verified and enabled |

The first name intentionally selects Kuno once its native adapter passes
acceptance; it is not an accidental name collision. The explicit Ollama alias
still enters Kuno admission, placement and leases, then removes the prefix
before upstream dispatch. It is not a scheduling bypass or a new server URL.
Use an exact registered alias map, not arbitrary prefix stripping; a literal
upstream name that collides with a proposed alias prevents that native binding
from activating until the naming conflict is resolved.

Expose both usable routes in `/api/tags` and `/v1/models`, with consistent
`/api/show` and `/api/ps`. Reuse the underlying artifact identity, but report
actual adapter capabilities and each worker's real resident footprint. Preserve
upstream entries/fields for unaffected models. Response model identifiers must
identify the selected public route, including the explicit alias where used.
Determine any additive engine/capability discovery fields before consumer work.
The native route must not advertise Ollama's vision/tools/thinking capabilities
until those native paths pass acceptance.

Preserve the artifact's effective template/default parameters and reject
unsupported native options/modalities clearly. Do not silently fall back to
Ollama after an unsupported request or failed native execution; the explicit
alias gives clients a deliberate comparison/recovery path. Production remains
on Ollama until the native registration is verified and enabled.

Known model-scoped endpoints resolve the route consistently. Ordinary unknown
Ollama endpoints retain proxy behavior; unknown native operations fail clearly.
For borrowed artifacts, pull/create/copy/delete/push remain Ollama-owned storage
operations with coordinated invalidation, rather than native inference calls.
Do not blindly forward a native-only model's management request. Any explicit
alias used in supported management fields must be normalized consistently and
its response restored, with tests for source/destination names and collisions.

The shared [native compatibility contract](kuno-bonsai.md#ollama-compatibility-contract)
records this intentional alias exception to its earlier collision prohibition.
No connector/UI implementation is required during research. Before integration,
publish the exact endpoint/error/capability contract and client parity tests;
Vashti should consume actual discovery rather than infer the engine from a
marketing model name.

## Implementation phases and acceptance

### 1. Freeze the reference and measure the execution primitives

Pin the existing Ollama manifest and complete borrowed artifact set, effective
tokenizer/template/parameters, operator reference and exact Ollama/llama.cpp/Strata
builds. Prepare public synthetic fixtures and a matched benchmark harness.
Confirm the reference supports this graph and packing rather
than silently falling back or changing the model. If Ollama cannot run that
artifact, record that limitation and use a verified lower-level reference.

Measure small bounded AVX2 expert kernels, B70/NVIDIA quantized matvec/matmul,
upload/download, dispatch/events and overlapped CPU/DMA. Compare direct I/O/mmap
for synthetic PLE row reads and bounded cold expert reads. Implement/test Q5_0
PLE and every mixed tensor format present in this artifact. Select backend
bindings from these results.
Proposed initial GPU path: Rust-owned execution with a narrow SYCL kernel ABI
on Intel and CUDA kernel ABI on NVIDIA; retain Vulkan as a measured alternative.
No full oneAPI toolkit or Docker requirement should be imposed on installed users.

Before any complete-model run, establish experimental host RAM/pinned-memory
admission and per-device reservations under ancestor cgroup and host limits.
Reserve a fixed budget for weights, staging, request state, scratch and safety
headroom; prevent other owned workers from consuming the same reservation.

**Exit:** artifact/operator inventory, actual memory plan, reviewed runtime
closure, baseline receipt and measured transfer/compute costs. No speed promise.

### 2. Build the Rust graph and one B70 worker

Create a small experimental text crate/worker. Rust owns a bounded GGUF reader,
quant layout, tensors, loader, complete model graph, events, request state,
sampling and cancellation. Implement the chosen artifact's minimal set of tensor
formats first, including mixed gate/up/down and dense/PLE types; reject others.
Use independently checked primitive kernels. Include PLE, gated residual, GDN,
QSA, all routed/shared experts and the exact tokenizer/template.

Single-device means computation on B70, not a promise that all full-model weights
fit its VRAM. Where necessary, use fixed, budgeted disk-to-host-to-B70 expert
streaming and disk-backed PLE, and label those paths explicitly. Adaptive
placement and CPU expert computation remain phase 3; large whole-model
allocation waits for phase 1's admission plan.

Keep a slow deterministic/reference execution mode to isolate errors. Test
integer PLE semantics, quant blocks, intermediate tensors and logits on synthetic
fixtures; compare full prefill, repeated single-token decode, chunk boundaries
and cache state. Set precision-specific tolerances from the reference. Run MTP
off and fixed placement initially.

**Exit:** complete single-device model correctness, measured peaks, bounded
no-swap execution, streamed output/cancellation, clean model-only idle reuse and
fresh-worker recovery. Keep production routing unchanged while experimental.

### 3. Add measured expert placement and CPU co-execution

Maintain a bounded expert cache with explicit slot versions and completion
events. Never evict or refill a slot while a kernel/DMA operation uses it.
On a miss, select CPU execution versus GPU fill using measured costs and host
bandwidth contention. Preserve all selected experts and their router weights.
Cache predictions may decide when to fetch; a prediction miss cannot skip work.

Separate decode from prefill: group prompt tokens by expert and overlap bounded
next-layer transfers with current-layer computation. Compare LRU, layer-aware
allocations and a synthetic hot-set preset before adding policy complexity.
Account for host mirrors and pinned buffers; pin bounded staging or verified
weight regions, not whatever free memory happens to remain.

This is an explicit streamed-weight/CPU-expert execution profile. Kuno must
report it honestly rather than treating intentional CPU work as verified full
GPU residency or silently enabling CPU fallback on an existing Ollama request.

**Exit:** same graph/packing correctness across hits, misses, evictions and CPU
partitions; reproducible cost-model calibration; demonstrable improvement over
fixed placement without memory-pressure instability.

### 4. Optimize the graph and evaluate both GPUs

Profile before fusing kernels: quantized prefill, GDN, QSA index/top-k/gather,
residual gates, CPU expert dispatch and output sampling. Port only optimizations
supported by the device. On Intel, begin with explicit events and correct memory
fences; do not transplant CUDA's device/host polling protocol unchanged.

Test B70-only, 3060-only, and then two alternatives: contiguous layer splits
(few cross-device boundaries) and secondary-device expert groups (more transfers,
possibly more parallel compute). Use host staging until peer support is proved.
Count transfer and synchronization cost, duplicate common weights, request state
and per-device headroom. Retain single-card execution when dual-card is slower.

**Exit:** a measured placement policy for this hardware; all touched GPUs and host
memory are reserved atomically. No claim that simply summing VRAM improves speed.

### 5. Add verified MTP and optional cache compression

First establish whether the borrowed model contains a usable draft module; the
current metadata does not show one. A separately acquired compatible module is
an explicit additional artifact, not grounds to duplicate the main weights.
Use the correctly packed draft module only after non-speculative parity.
Verify every draft against the target. Commit accepted positions and the target
correction; restore every rejected recurrent, convolution, indexer/KV, PLE and
sampler state. Test rejection at every position, EOS/stops, cancellation and
sampled as well as greedy generation. Emit tool calls only from committed output.

Introduce activation/KV compression as separately identified, opt-in execution
profiles with quality/error measurements. No expert pruning, omitted residual
branches, substituted draft output, or silent context/precision reduction.

**Exit:** reference-compatible target sampling/state behavior and speed measured
by text type, context and draft acceptance. Quantized modes carry honest limits.

### 6. Integrate verified text models into Kuno

Use the existing coarse resource leases outside the worker and the per-layer
expert scheduler inside it. Integrate the earlier proven host RAM/pinned-memory
admission and measured peak profiles into the common scheduler/reporting;
existing load observations are insufficient. Independent-device
HTTP concurrency is a separate unfinished gateway feature, not created by the
new inner scheduler.

Implement borrowed-store ownership, explicit alias registration and accidental
collision checks, `/api/tags`, `/api/show`,
`/api/ps`, `/v1/models`, chat/generate and streaming conversion per the
[existing native-engine compatibility contract](kuno-bonsai.md#ollama-compatibility-contract).
Follow the proposed default-native/`ollama.` routing contract above; borrowed
artifact management remains Ollama-owned. Unknown native-model operations fail
explicitly. Unaffected Ollama routes keep their proxy behavior. Preserve long
inference with no arbitrary generation deadline, no request replay once execution
may have begun (even before the first output), structured thinking/tool calls,
keep-alive, unload and classified failure recovery. Advertise only tested
capabilities; embeddings/vision/structured output are not implied by text chat.

**Exit:** client transcript parity, resource/lifecycle and privacy acceptance,
protected runtime packaging, sustained synthetic soak and a tested backend handoff.
Any changed shared discovery/protocol contract is agreed and documented before
Vashti consumes it. No UI change or release is part of this research.

## Privacy and recovery apply to the new engine

Model weights and kernel caches may persist. Prompts, outputs, token IDs, KV/GDN
state, embeddings, PLE access keys, router scores/expert sequences, draft state
and request fingerprints may not persist in logs, files, journals, swap or core
dumps. For the proposed native worker's model-only idle policy, finish or cancel
and synchronize CPU kernels, GPU compute/DMA and PLE I/O, then clear full-capacity
request buffers and KV/GDN/convolution/PLE/routing/draft state. Acknowledge CLEAN
only after that work is complete; retain only model-weight state. Keep resource
leases until completion is proven, or reap/quarantine the owned worker on
uncertain cleanup and verify memory recovery. Do not import Strata's saved
conversations or FreeToken's idle prefix cache. This proposes the new worker's
boundary; it does not rewrite managed Ollama's existing RAM-only cache lifecycle.

Persist only reviewed, bounded operational aggregates by hardware/model/runtime
and workload size class: peak bytes, transfer/compute costs and classified
outcomes. Expert hot-set profiles may be shipped from public synthetic fixtures;
do not learn and save token/expert/PLE histories from private requests. Any
production-derived cache policy metadata needs an explicit privacy review before
retention. Runtime adapters must apply Kuno's existing no-swap, no-core,
restricted-write and allowlisted-diagnostics protections before reading content.

Reserve workspace, context, verification and prefill peaks before filling an
expert cache. Strata's Intel notes report oversubscription and host stalls; do
not use a successful allocation as proof of safe GPU residency. On capacity
pressure, synchronize and shrink caches or retire the owned worker, classify
external pressure separately, and retry only when dispatch/execution has not
begun and the attempt is proven safe.
Do not add indefinite failure latches, arbitrary inference deadlines, automatic
driver resets, or silent changes to the requested context/model.

## Benchmark receipt required in the next phase

Use identical full model bytes, quantization and execution precision wherever
comparable. Separate Coder/pruned models, Swift fine-tunes, MTP on/off, compressed
KV, prefix reuse and reduced thinking from the matched base-model comparison.

Report cold load/time to first token, prefill tokens/s, decode tokens/s and total
latency separately. Repeat fixed-length synthetic workloads at short, medium
and long contexts, with cold/warm expert caches and fixed output lengths.
Measure RAM/pinned RAM, each GPU's resident and peak memory, copies, CPU/GPU
overlap, cache behavior, draft acceptance and energy/temperature where available.
Persist no real-user trace. Record source/artifact digests, driver/runtime and
topology; an idle PCIe link snapshot is not a transfer result.

Include cancellation during loading/transfers/verification, worker death,
external memory pressure, admission headroom, recovery and model-only idle
privacy. A speed result does not establish numerical fidelity or stability.
This research ran none of those tests and changes no installed behavior.

## Research validation receipt

The findings were cross-checked against both pinned source checkouts, the official
configuration/reference, current Kuno specifications and the downloaded Ollama
model's manifest/tensor metadata. Only model metadata and file sizes were
inspected; no user request content or full weight payload was read or retained.
The proposal received independent source/architecture review, including the distinction between GPU
computation and full weight residency, early host-memory admission, and complete
cleanup/lease ownership. Documentation whitespace, local links and referenced
source-path/revision/range checks pass.
Runtime, numerical, throughput, installation and end-to-end application tests
remain unperformed; no existing acceptance receipt covers the proposed engine.
