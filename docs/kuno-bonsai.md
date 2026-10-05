# Bonsai 2 and a native Rust inference engine

Accepted direction, 2026-09-23. This work follows the GPU scheduler and managed
worker boundary. It must not delay the initial Qwen image path. No Bonsai model
is currently advertised or served by Kuno.

The goal is a Kuno-owned Rust inference engine, initially focused on Bonsai 2,
with ordinary Ollama clients able to discover and use it. Wrapping another
server is a useful reference/bridge, not completion of the native-engine goal.
Rust owns loading, execution, device memory, request lifetime, and sampling;
using GPU driver APIs, compute libraries, or shader kernels is compatible with
that goal. A Rust wrapper around a complete C++ model graph is not the native
engine milestone.

## Delivery sequence

1. Finish backend-independent placement and managed worker lifecycle. Workers
   supply verified physical device assignments, exact workload identities,
   resident allocations, and peak-memory estimates. They expose load, unload,
   cancellation/completion, and classified failures. The same scheduler manages
   Ollama, image workers, and the new LLM engine.
2. Reproduce a pinned Prism reference on each target GPU during an idle window.
   Check correctness before throughput. The old project is reference material;
   do not import its environment or assume a driver upgrade fixes performance.
3. Build a small native-engine crate when implementation begins. Start with a
   bounded, validated GGUF reader and one precisely identified packing. Verify
   dequantization, transforms, tokenization, and graph operations against the
   pinned reference before running the complete model. Reject unimplemented
   architecture/packing combinations instead of producing plausible bad output.
4. Match the complete text graph and greedy token/logit behavior on fixed
   synthetic fixtures, including multi-token prefill and recurrent/cache state.
   Choose numeric tolerances from the reference precision and publish them with
   tests. Compare intermediate tensors to locate divergence; matching a few
   conversational answers is insufficient.
5. Implement and profile the GPU path, beginning with the B70 target. Measure
   prefill and decode separately. Keep an NVIDIA comparison with the same model
   packing and settings. Optimize measured bottlenecks; do not promise a speedup
   from switching languages. Validate transformed/quantized matvec and fused
   operations independently before accepting performance results.
6. Expose the verified engine through Kuno's model registry and compatibility
   adapter. A Prism bridge may make Bonsai usable sooner, but remains a separate
   implementation choice with its own capability/placement checks. Vision,
   tools, and thinking are advertised only as their paths pass acceptance.

The correctness harness uses synthetic test content. Production inference and
background profiling retain only the operational metadata allowed by the main
specification. No prompt cache files, request logs, output files, or content in
process arguments. Runtime diagnostics must be reduced to allowlisted metadata
before persistence. Audit dependency logging and crash behavior too.

## Ollama compatibility contract

Use a stable explicit public model identifier, provisionally `bonsai2:27b`.
Reject accidental collisions with Ollama names/aliases. **Updated intended
direction, 2026-10-05; not implemented:** a verified native adapter may explicitly
bind an existing Ollama model name and borrow its files read-only. That ordinary
name then selects Kuno, while a registered `ollama.<full-model-name>` alias
selects managed Ollama through the same scheduler. This user-requested binding
supersedes the earlier unconditional collision prohibition; it does not allow
undeclared shadowing or arbitrary prefix stripping. See the
[storage/routing research contract](kuno-text-engine-research.md#proposed-model-names-and-dispatch).
Pin artifact digests, quantization, tokenizer/template, runtime build, and
supported capabilities. A native-only model still requires its own unique name.

- Merge installed usable entries into `/api/tags` and `/v1/models`. A warm load
  is not required for discovery. Do not advertise a merely planned download or
  unsupported engine. Preserve upstream entries and their extra fields.
- Resolve the model before dispatch. Route Kuno-owned names and verified explicit
  bindings to their adapter; registered `ollama.` aliases and other ordinary
  names retain Ollama execution. Unknown endpoints for an explicitly
  Kuno-owned model must fail clearly rather than accidentally reaching Ollama.
- Supply `/api/show`, running-model `/api/ps`, and corresponding OpenAI model
  metadata consistently. A Kuno worker must not be double-counted as Ollama.
- Translate chat/generate requests, streaming NDJSON, OpenAI streaming events,
  final markers, timing/token counts, stop reasons, cancellation, and errors.
  Preserve thinking and tool calls as structured data. Do not silently discard
  an unsupported option, requested modality, format constraint, or API feature.
- Define keep-alive/load/unload behavior through the common scheduler. Borrowed
  artifacts retain Ollama ownership of deletion, copy, pull, push and creation;
  coordinate workers and invalidate bindings when those artifacts change.
  For native-only artifacts, management must implement the operation or return
  an explicit unsupported-operation error.
  No management request for a Kuno model is blindly forwarded to Ollama.
- Test the same client transcript against both runtime routes, including
  fragmented streams, tool round trips, cancellation before/after headers,
  structured-output requests, queue timeouts, and memory-pressure recovery.

## Existing local evidence

The previous project is
`/media/jmicko/faststore/Documents/Codex/2026-09-19/wo/outputs/b70-comparison/bonsai2`.
Its `RESULTS.md` and launch scripts were reread on 2026-09-23. These are historical
measurements, not benchmarks of the current Kuno implementation:

| GPU / backend / packing | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| RTX 3060 / CUDA / PQ2_0 | 457.14 | 30.36 |
| RTX 3060 / CUDA / PTQ1_0 | 235.95 | 24.30 |
| B70 / SYCL / experimental Q2_0 | 855.38 | 13.96 |
| B70 / Vulkan / PTQ1_0 | 428.02 | 3.69 |

The reference release was `prism-b10709-9a9394a`. The Vulkan record verified
65/65 layers on GPU; its slow decode was not explained by layer offload. SYCL
used a different packing, so it is not a controlled backend-only comparison.
The report's basic arithmetic/JSON/code checks do not establish full numerical
correctness. Local metadata identifies the serialized graph as `qwen35`; use
artifact metadata and source implementation rather than a marketing model name
to select operators.

The official [Prism backend support matrix](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md)
documents Bonsai-specific transforms as well as packing support. At its pinned
release, SYCL's fast Hadamard path has width/type restrictions and larger
rotations may use dense multiplication. That is a profiling lead, not a proven
cause of the local result.

The [format documentation](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md)
warns that recognizing Bonsai 2 Q2_0 and its base architecture does not establish
compatibility: missing activation transforms can still produce incorrect output.
Implement sign flips, transforms, and graph placement together; a fast standalone
Hadamard kernel is not sufficient.

Reverify upstream support and pin source revisions when engine work starts.
Choose the GPU abstraction after a small B70 operation/transfer/dispatch spike;
do not choose a library merely because it is written in Rust or assume it
already implements this model. Keep the production gateway independent of
experimental engine builds until correctness, placement, privacy, and lifecycle
checks pass.
