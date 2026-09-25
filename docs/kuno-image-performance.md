# Qwen 2.1 performance follow-up

This records the remaining performance work after the seam and dimension fixes.
The installed engine is still slower than the supplied ComfyUI workflows. The
speed goal is not complete. The integration checkpoint describes the current
implementation; the later measurements also retain earlier experimental history.

## Integration checkpoint

The bounded layer-by-layer cache pipeline and redundant modulation-copy removal
are implemented. Cache storage uses the device's pinned host buffer type when
available; execution reserves two bounded GPU staging slots and uses a separate
transfer backend plus events. Ineligible backends, pageable allocation fallback,
unsupported graph layouts, or insufficient staging headroom retain synchronous
execution. Transfers are drained on scope exit, including cancellation.

`ImageRecipe.attention` now accepts `standard` (the backward-compatible default)
or `fast`. Fast selects SageAttention only for Qwen 2.1 on CUDA; unsupported
workflow/backend combinations fail explicitly. It remains an opt-in numerical
tradeoff. The further kernel/layout ideas below are deferred while app integration
is designed.

Synthetic 512×512/four-step edits returned byte-identical output before/after the
pipeline with GPU and RAM FP16 caches. The original 25-step edit with fast
attention and the active pipeline completed in **169.25 s**, 1248×832 RGBA, alpha
246–255, without a visible seam. A short lint build overlapped the beginning of
that run, so the timing is not a strictly isolated benchmark. The original
50-step cabbage with fast attention completed in **262.89 s**, exactly 1184×1776
RGBA, alpha 246–255, without a visible seam; no build or other inference ran
alongside it. Both returned metadata-free PNGs inspected only in RAM. They remain
slower than the user's ComfyUI cold times of 126.14 s and 199.94 s respectively.

Build/native regressions, 122 Rust tests (8 hardware tests ignored), and the
CUDA-enabled Clippy check passed. Multiple-reference fast generation and
cancellation after the transfer pipeline became active passed (4.02 s cleanup,
no forced stop). Installed service validation is recorded in `kuno-images.md`.

## Measured state

The original weasel recipe uses 25 Euler/simple steps, CFG 4, a 1248×832 output,
and the original JPEG reference. The preceding installed CUDA implementation took
241.59 seconds on the RTX 3060. It uses an FP16 prefix cache in system RAM and
fused SwiGLU. The user measured ComfyUI at 126.14 seconds cold and 109.51 seconds
warm on the same card. The supplied B70 ComfyUI results are 78.25/64.48 seconds;
equivalent full native performance on that card is still unresolved.

A separate restricted worker enabling the engine's existing SageAttention took
187.84 seconds for the original weasel recipe. The returned metadata-free RGBA
image was 1248×832, alpha 246–255, and had no visible central seam. It was inspected
in a temporary RAM-only preview and discarded. No compilation or other inference
ran during that test, and no new kernel errors were observed. This is about 22%
less time than the installed build but still 49% above ComfyUI's cold result.
SageAttention changes numerical results and is not the default. The integration
checkpoint above adds the original cabbage, multiple-reference, second-seed and
cancellation checks for the combined implementation.

## What the profile establishes

After two warm-up steps, the installed algorithm's next two synthetic edit steps
took 16.04891 seconds. Summed measured GPU operations took 12.7755 seconds:
attention 5.1170, matrix multiplication 3.6389, contiguous copies 1.0843, and other
operations 2.9353. Instrumentation synchronizes operations, so these are
bottleneck measurements rather than a second end-to-end benchmark.

There were no diffusion-weight reloads between the measured steps. However,
the host prefix cache is copied synchronously into one complete graph before
computation. Its two CFG conditions transfer about 2037.5 + 2035.5 MiB per step.
Warm execution uses one graph segment and 2676.20 MiB of workspace. The remaining
1.64 seconds per step includes transfers and graph overhead; it has not all been
attributed to PCIe. The physical link is shared with the ComfyUI comparison, but
the transfer schedules differ.

The local ComfyUI reference pins cache storage and prefetches the next layer
through two reusable staging buffers while the current layer computes. It also
contains fused modulation, residual, and SwiGLU/quantization paths. Source
availability does not prove which optional kernels ran in the user's timed
ComfyUI installation.

## Implementation notes and deferred work

1. **Remove full-range modulation copies.** In native
   `src/model/diffusion/qwen_image_2_1.hpp`, cached steps have `prefix_length == 0`.
   Use `x` directly in `modulate()` instead of a full-range `ggml_ext_slice`, which
   always materializes a contiguous copy. This removes four copies per block
   without changing arithmetic. The candidate is in the pinned patch and built
   successfully with the native regression checks. Separate 512×512/four-step
   synthetic edit comparisons were byte-identical before/after with both GPU
   and RAM FP16 caches. The release candidate SHA256 is
   `e6d9934a30f2a558104a9003cc85aa5b6c87e03384e4d32220677e22309ae974`.
   This was the pre-pipeline candidate. The combined results are above.
   The short correctness timings are not a speed claim.

2. **Pipeline the host cache by layer.** Work within `src/core/runner_cache.*`,
   `ggml_runner.*`, and the Qwen graph boundaries. Reuse patterns in
   `model_manager_prefetch.cpp`: device-provided pinned host buffers, a separate
   transfer backend, asynchronous tensor sets, and completion events. Force layer
   segmentation only for host-cache reuse. Allocate and account for two reusable
   K/V staging slots; prefetch the next layer while the current layer executes.
   Simply using asynchronous copies on the present monolithic graph will not
   produce the required overlap. Retain the synchronous path where the needed
   backend capabilities or memory are unavailable.

   Keep positive/negative cache identities and reference lengths distinct. Wait
   for pending copies before reusing a staging slot, switching conditions,
   cancelling, evicting, or freeing host storage. Include staging and pinned
   memory in resource budgets. Verify byte-for-byte synthetic output, cancellation
   with a copy pending, and the full original edit. Measure copied bytes, transfer
   wait time, workspace, and whole-request time; do not infer overlap from API
   submission alone.

3. **Evaluate SageAttention as an explicit performance option.** Its output layout
   matches the native attention contract. Its FP16-prefix fast path currently
   rounds live K to FP16 and then expands it to F32 for Sage; a Sage-specific path
   can retain live K in F32 while expanding only the stored prefix. Compare actual
   attention/denoiser errors and full-workflow images in RAM. Keep masked attention
   on its supported fallback. Do not claim bit-identical images or enable it
   indiscriminately on other backends.

## Further exact or mathematically equivalent candidates

- CUDA flash attention accepts strided V. The current implementation materializes
  a whole V transpose before attention. A guarded batch-one FP16 path could pass
  the four-dimensional permuted view directly, avoiding the contiguous copy and
  three-dimensional reshape. Test odd lengths and a dense attention oracle;
  benchmark attention plus copy because a strided kernel may be slower.
- Qwen-specific multi-axis RoPE could use existing `ggml_rope_multi`, with channel
  reordering and frequency factors reproducing the per-axis frequency restart.
  This is distinct from the unsuccessful unfused layout prototype. Validate
  negative image coordinates, long text positions, reference prefixes and cached
  offsets; retain the existing circular-embedding path. Different trigonometric
  implementations may change rounding. Both CUDA and Vulkan need measurements.
- Fused residual/modulation and SwiGLU inside the ConvRot activation quantizer
  could reduce memory traffic further. These are larger backend-kernel changes,
  not simple model-graph edits. Preserve operation/reduction rounding initially.
  Native Q/K/V already reuse quantized activations; do not duplicate that work.
- Moving all activations to BF16 is a broad numerical/backend change. CFG batching
  increases workspace and cache pressure. Neither is the first next change.

## Acceptance boundary

Preserve the seam fix, exact requested dimensions, supported settings, ordered
multiple references, metadata-free output, and the no-disk/no-swap inference
boundary. Compare complete original cabbage and weasel recipes on each claimed
device with idle hardware and matched settings. Short synthetic tests or a fast
attention kernel alone cannot establish that the full speed gap is closed.
