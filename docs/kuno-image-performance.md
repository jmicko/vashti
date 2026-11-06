# Qwen 2.1 performance follow-up

This records the remaining performance work after the seam and dimension fixes.
The optional Intel SYCL engine is now installed on this host and passes the
original full generation/edit workflows. It remains slower than the supplied
ComfyUI workflows; the speed goal is not complete. The latest receipt describes the installed
implementation; earlier measurements remain as experimental history.

## Native optimization candidate (2026-09-30)

Work starts from installed backend `f6f6c22` in a separate Kuno worktree;
the concurrent Vashti application changes are preserved. The standard candidate through `eee97c8` was initially installed on this host;
the optional SYCL runtime was subsequently accepted and installed as recorded
in the final receipt below. Full Intel edit execution and synthetic GPU numerical
checks have passed; performance is still below the supplied ComfyUI baseline. The clean
CUDA edit and subsequent full generation acceptance are recorded below.

- Vulkan Qwen 2.1 can retain FP16 prefix K/V on the GPU, halving cache storage.
  Cache precision and working-memory reserve are now separate: Vulkan still
  reserves 9 GiB for weights/work, while CUDA retains its 3 GiB reserve and
  existing bounded host-cache policy. Vulkan does not use the CUDA host pipeline.
  Insufficient memory still disables prefix caching explicitly.
- Both CUDA and Vulkan concatenate supported resident FP16 caches directly.
  Native flash attention already consumes FP16 K/V. The adapter avoids copies
  whose only operation was casting an already-FP16 tensor to FP16 again.
- Batch-one CUDA flash attention can read a permuted FP16 V view without a full
  contiguous transpose copy. Other backends, batches, scaling, and padded heads
  keep the established layout.
- Opt-in fast attention keeps newly computed K in FP32 for Sage instead of
  rounding it to FP16 and expanding it again. The stored prefix remains FP16.
  This can change fast-attention pixels; standard attention does not take that
  path. Masked attention retains the supported fallback.

Added `attention-layout-test.cpp` with an independent double-precision dense
attention oracle, odd sequence lengths, batch and mask cases, scaling, both
input precisions, and exact prefix/live concatenation checks. Combined CPU
attention and embedding/rotation/dimension/memory-policy regressions passed.
CPU oracle relative L2 error was 0.000525–0.001196. The expanded 67-case GPU
oracle passed on both cards: maximum relative L2 error was 0.00069327 on Intel
and 0.00125529 on CUDA. It exercises partial query/key blocks, head padding,
masks, batching, and FP16/FP32 QK accumulation; both cache concatenation checks
also passed. The development build runs CPU regressions; explicit `--device
CUDA PCI` / `--device Vulkan PCI` acceptance requires an idle GPU.

`cargo test --locked -p kuno -p kuno-image -p kuno-protocol` passed; hardware
checks remain ignored. No image API, discovery, ordinary Ollama routing,
generation deadlines, preview transport, or content-persistence behavior changed.
No Vashti consumer changes are required for these internal optimizations.

The current host reports kernel 7.0.0-34 and Mesa Vulkan 26.2.3 with Intel
cooperative-matrix and integer-dot-product capabilities. This is discovery,
not evidence of the cause of previous GPU resets. The initial pinned SYCL backend lacked this engine's ConvRot operations; the
later private prototype adds them. The development results below supersede
that initial limitation, but do not yet establish full-workflow acceptance.

### Intel rectangular matrix attention

Live cooperative-matrix discovery reports FP16 M8/N16/K16 (FP16 and FP32
accumulation), not the square M16/N16/K16 shape required by the pinned attention
kernel. This sent Intel attention through the scalar fallback. Patch
`002-vulkan-flash-shapes.patch` adds a separate transposed-PV M8 shader, gated by
Xe2, the exact reported shapes, input types, accumulator support, and shared
memory. The legacy square shader remains unchanged. GGML is separately pinned
to `4bf5f6000653b7881d00963cd6ddb665ccd62a8d`; the engine and GGML diffs are checked
independently. Both new shader variants compiled and passed SPIR-V validation.

A synthetic warmed attention graph with 4056 queries, 8192 keys, 128 channels,
and 16 heads took 0.0830–0.0834 s with the scalar kernel and 0.0319–0.0326 s with
M8, about 2.6 times faster. These are representative sequence sizes, not an
end-to-end image result. GPU execution was isolated from other inference and
heavy compilation; upload, allocation, and readback were outside the timing.

The original Intel edit (25 steps, CFG 4, 1248×832, standard attention, previews)
took **733.826 s** with the FP16-cache/layout candidate before M8 and **558.176 s**
with M8. The first diagnostic had a 0.18 s feature-query compilation overlapping;
the M8 run had no heavy compilation or competing inference. Both produced valid
metadata-free RGBA images and all 25 previews. Images were inspected in RAM and
discarded. The M8 image had alpha 249–255 and no visible seam. The kernel reset
count stayed at its pre-test four entries, and memory was released after worker
exit. This is a 24% improvement but remains far above ComfyUI's 78.25 s cold edit;
Intel speed acceptance is **not complete**.

### Remaining Intel matrix bottleneck

A bounded synthetic harness compares ConvRot quantization, packed INT8 GEMM,
combined execution, and dense FP16 GEMM with FP32 accumulation. Dense weights
are dequantized and inverse-rotated from the same synthetic source weights;
its output intentionally differs by activation quantization and FP16 rounding
(relative L2 about 0.009). Upload and final readback are unmeasured. The first
invocation is reported separately; earlier tiny validation has already warmed
some generic pipelines, so it is not a process-cold claim.

| Matrix shape (4056 rows) | Quantization | Packed INT8 GEMM | Dense FP16 GEMM |
| --- | --- | --- | --- |
| 4096 → 4096 | 1.3–1.4 ms | 11.8 ms | 9.2–10.2 ms |
| 4096 → 24576 | 0.9–1.4 ms | 96.1–97.1 ms | 30.8–32.3 ms |
| 12288 → 4096 | 3.7 ms | 55.9–56.4 ms | 16.3–19.1 ms |

The large MLP INT8 kernel is about three times slower than the dense path.
Quantizer barriers are not the dominant cost in these measurements. The controlled tile and selective dense-weight experiments below are implemented
for development, but no new precision policy is accepted for deployment.

### Whole-pipeline timing and CUDA repeat

Private instrumentation closes a fixed numeric window after two warm steps;
no prompts, tensors, images, or raw backend diagnostics are recorded. Intel's
next two edit steps took 40.516 s, of which 40.469 s was GPU execution. No prefix
cache bytes were uploaded or cut back in this resident-cache window. This rules
out host-pipeline setup or repeated weight transfers as the main warm bottleneck.

CUDA's next four fast-attention steps took 20.105 s, with 18.845 s in GPU execution
and 0.881 s in upload submission. The window transferred 17,085,497,344 cache bytes
and cut back 17,543,725,056 bytes. Host resource creation/destruction took 0.019 s;
graph building took 0.013 s. These are inclusive phases, not additive operation
profiles. The instrumented binary is private and must not be installed.

A clean CUDA repeat of the original edit (25 steps, CFG 4, fast attention,
1248×832, previews) took **160.702 s**, versus the earlier 169.25 s. All 25 previews
and a valid metadata-free RGBA result were returned; alpha was 249–255. Visual
inspection found no central seam. Output stayed in RAM and was discarded. Heavy
compilation and other inference were excluded. ComfyUI's user-supplied cold
126.14 s remains faster. This worker uses the previously recorded standard
candidate SHA-256 `2a09a1276b76968a8bdd6185cffe00ea32f6a29738b63548232920db9b917820`.

### Controlled kernel experiments

All tuning flags are internal, opt-in development probes. The public worker
clears its environment, so these experiments cannot be enabled accidentally by
a service environment. Default query/key tiles remain Br16/Bc32 and the legacy
INT8 tile remains unchanged.

- INT8 4096→24576: legacy 96.3–97.0 ms; 4×4/subgroup16 83.8–84.7 ms;
  2×2/subgroup16/32 118.2/118.6 ms; 4×8, 8×4, and 8×8/subgroup16
  343.5–345.6, 147–148.4, and 273.5–274.3 ms respectively. Larger tiles
  regress, despite identical integer math. Seven tiny independent integer
  cases plus the full graph passed each candidate (maximum GEMM relative L2
  0.0000010354). Dense FP16 remained 30–32 ms for that shape.
- Attention query Br32/64: all 67 oracle cases passed; maximum relative L2
  0.00060391/0.00060201. Warm representative graph 36.8/105.5 ms, versus
  33.0–33.4 ms at Br16. Neither is adopted.
- Attention key Bc64/128 with Br16: all 67 oracle cases passed; maximum relative
  L2 0.00062896/0.00062197. Warm graph 30.81–31.03/31.21–31.51 ms, versus
  33.5–33.7 ms at Bc32. These small graph gains are not whole-image acceptance.
  Shared-memory checks retain fallbacks, and query/key experiments cannot combine.

Frozen query and key shader variants passed SPIR-V validation. Strict CPU tile
parser tests reject malformed/unbounded settings. The development script now
content-addresses native source/build directories by the complete two-patch
bundle; old native checkouts and unknown local modifications are preserved.

### Selective dense MLP experiment

`KUNO_EXPERIMENTAL_QWEN_MLP_F16=1` reconstructs only Qwen 2.1 Vulkan diffusion MLP
weights at load time. Other weights stay INT8. It remains disabled by default;
see [memory and precision contract](../crates/kuno-image/native/mlp-f16-experiment.md).
The real ModelLoader/Linear CPU regression covers independent H4 reconstruction,
scale types/shapes, indexed shards, destination bounds, and rejection cases.
All engine C++ translation units were rebuilt to avoid stale inline Linear code.
Loading workers are capped at four; inference still uses eight compute threads.

A four-step original edit took 103.798 s: first step 47.454 s, final sampling
86.318 s, and decode/delivery 17.480 s. Warm sampling is about 12.96 s/step,
versus 20.23 s/step with INT8 MLP. An earlier 133.798 s run incorrectly used four
compute threads; its longer decoding phase is not an accepted speed comparison.

The full original 25-step Intel edit took **376.803 s**, with all 25 previews,
1248×832 metadata-free RGBA output, alpha 247–255, and no visible central seam.
Heavy compilation and competing inference were excluded. Output was inspected
only in RAM and discarded; the zero-swap test cgroup reported zero swap bytes.
Kernel reset entries stayed at the four-entry pre-test baseline. This is 32.5%
less time than the M8/INT8 candidate, and 48.7% less than the pre-M8 candidate,
but still 4.8 times ComfyUI's 78.25 s cold edit. **Speed acceptance remains open.**

Xe reported a 50,688,000-byte idle baseline and 27,356,573,696-byte peak
(27,305,885,696-byte delta). The worker's reported VmHWM was 40,322,760,704 bytes,
including 39,300,001,792 RssFile and 1,190,039,552 RssAnon. The test cgroup peak
was 1,338,941,440 bytes. RSS includes mapped model/driver/device pages and the
cgroup may not own all previously charged mappings; these are separate observed
counters, **not a claim of 37.6 GiB physical host RAM**. A later source review found that Vulkan's host-buffer provider is globally tied
to Vulkan device zero and can select host-visible device-local memory. The
candidate now uses ordinary CPU buffers for CPU-owned Vulkan parameters,
preserving CUDA pinned storage and operation validation. The 16-case fake-backend
regression checks provider calls and requires zero driver initializations or
allocations; reverting the selection fails the regression.

A post-fix four-step dense edit passed all four previews. GPU peak remained
exactly 27,356,573,696 bytes, so this run does **not** establish a GPU memory
reduction or earlier duplicated local-memory allocation. VmHWM was
40,335,425,536 bytes; RssAnon rose to 11,407,409,152 and RssFile fell to
28,919,529,472. Cgroup peak rose to 11,547,480,064 bytes with zero swap. This
confirms anonymous CPU ownership and improves accounting, without interpreting
RSS as physical RAM. The 109.140 s elapsed time overlapped native compilation
and is a correctness/memory test, not a clean speed comparison.

### Private Intel vendor-library probe

A signed-repository-verified private oneAPI bundle was unpacked under ignored
build output, without installing system packages, drivers, repositories, or
services. Compiler 2026.1.1-325 and MKL 2026.1.0-236 ran a physical-BDF-selected
Level Zero synthetic probe. Four tiny signed/padded/zero-control cases compare
every output against independent CPU dot products; the full MLP shape adds 64
CPU spot checks and compares every I8-I32 result against same-number F16-F32.
All checks passed, with zero output differences and checked asynchronous errors.

At rows4056/K4096/N24576, MKL signed I8 GEMM warmed at 2.714–2.736 ms
(median 2.729 ms), and mixed F16→F32 at 4.827–4.862 ms (median 4.862 ms).
Explicit device buffers total 1,149,272,064 bytes, excluding vendor workspace.
This suggests a useful alternate kernel path; it is **not whole-image inference
or deployment acceptance**. The pinned SYCL backend still needs Qwen ConvRot/I8
support and suitable unmasked MHA attention. oneMKL SYCL requires dynamic runtime
libraries; the ordinary CUDA/Vulkan release build does not acquire this dependency.

### Cache-precision compatibility check

A two-reference 512×512/four-step synthetic edit produced byte-identical RGBA
with the installed FP32 Vulkan prefix cache and the FP16/layout candidate before
M8 (zero RMS and maximum pixel delta). This isolates cache precision/layout;
it does not establish byte equivalence of M8 attention or dense MLP math. Both
results were kept in RAM and discarded.

CUDA cancellation during sampling passed with 6.655 s cooperative cleanup,
followed by a two-reference four-step fast-attention edit returning four previews
and metadata-free RGBA. No forced termination was used. These tests overlapped
builds and establish correctness/cleanup, not benchmark speed.

Intel dense-path cancellation after step three completed cooperatively in
14.407 s, without forced termination, and the test cgroup used zero swap.
The next default-precision two-reference 512×512/four-step edit passed, returning
all four previews and metadata-free RGBA. Xe peak was 20,572,209,152 bytes,
cgroup RAM peak 2,015,928,320 bytes and swap zero. Compilation overlapped, so
85.893 s is correctness/cleanup evidence rather than clean speed acceptance.

The fresh, content-addressed Vulkan native build script completed, including CPU
embedding/rotation, 67 attention/cache oracle cases, strict tile parser, 16 buffer
policy cases and real dense-loader regression. The dual CUDA/Vulkan standard
candidate was separately linked from freshly rebuilt engine C++ objects and the
new Vulkan shaders; CUDA libraries were unchanged. Rust checks passed with 171
tests, nine hardware tests ignored, Clippy warnings denied and formatting. An
initial parallel gateway run had two timing-sensitive failures; the complete
serial suite then passed. No arbitrary inference deadlines changed.

The standard dual-backend candidate's original CUDA 50-step generation also
passed: 1184×1776 metadata-free RGBA, all 50 previews, alpha 247–255 and no
visible central seam. The elapsed 248.820 s overlapped private SYCL compilation,
so this is functional acceptance, not a clean speed comparison. Output stayed
in RAM and was discarded. The candidate SHA-256 is
`bbd27f1f6cc9d35ecdb24ab32f274f46e8add649896753298fa2b9fb80d345cf`.

Intel default-precision text-to-image also passed at 512×512/four steps, with
all four previews and metadata-free RGBA (alpha 246–255). Xe peak was
8,803,368,960 bytes, cgroup RAM peak 1,266,565,120 bytes and swap zero.
The 18.595 s elapsed time overlaps compilation and is a smoke test.

Pending acceptance includes installed-service verification. CUDA graphs are disabled by the pinned upstream default; graph
capture is not an explanation for current performance without a separate
controlled experiment.

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

## Installed standard candidate receipt (2026-09-30)

The authorized transactional upgrade installed the tested CUDA/Vulkan binary
from this worktree through `eee97c8`, SHA-256
`bbd27f1f6cc9d35ecdb24ab32f274f46e8add649896753298fa2b9fb80d345cf`.
Configuration remained byte-for-byte unchanged: the public listener and
loopback upstream/control ports, RTX 3060 image selection, model collections,
and disabled LLM response/idle deadlines were preserved. Experimental dense
MLP/tile flags are not forwarded by the public worker; the later private SYCL
prototype is not in this installation.

The running service was active/running with zero restarts. Keyless discovery
and Ollama version passed on IPv4 loopback, IPv6 loopback and LAN; unauthenticated
control returned 401. A native keyless two-step 512×512 Qwen 2.1 generation
returned acceptance, both previews and one final PNG in 11.994 seconds, first
preview at 9.126 seconds. PNG dimensions, chunk checksums, terminal order and
metadata absence passed. Client and worker used zero-swap cgroups; image bytes
stayed in RAM and were discarded. This is installed transport acceptance, not
a clean performance or physical Android test. Admission is temporarily paused
for the remaining exclusive GPU experiments and must be resumed afterward.

### Restricted SYCL kernel acceptance (2026-09-30)

The private ABI-160 SYCL prototype now supplies native ConvRot INT8 operations
and an opt-in unmasked oneDNN SDPA path. It is not installed in the public Kuno
binary. The initial independent GPU arithmetic test found an approximate
reciprocal changing a rounding tie from byte 64 to 63. Explicit nearest-even
FP32 division/multiplication fixed it; all 30 expanded arithmetic GPU cases
passed unchanged tolerance (maximum absolute error 0.0000022885). CPU policy,
arithmetic, malformed-shape and strict verbose-parser regressions also passed.

Both 67-case GPU attention/cache suites passed under inherited Landlock write
denial and zero-swap cgroups. Maximum relative L2 error was 0.00170402 with the
no-mask path off and 0.00125530 with it on. These tests cover masked/unmasked,
partial sequences, padded heads, batches, accumulator precision and cache reuse.

A fresh isolated 4056-query/8192-key/head-128/32-head graph passed finite-output
checks with 200,671,232 input bytes, 66,453,504 output bytes and 133,562,368 graph
workspace bytes. Warm execution was about 9.77 ms. The strict oneDNN 3.11.4
counter classifier found four strict GPU SDP graph executions, one nested
`ocl:micro:reusable` SDPA creation, and zero decomposition/reference/unknown,
MatMul, Softmax, malformed or filtered records. This establishes optimized-path
evidence for that isolated graph, together with numeric/device-wait checks and
Xe peak 687,726,592 bytes (637,038,592 delta), rather than inferring fusion from
a supported partition alone. Cgroup peak was 319,651,840 bytes and swap zero.

The actual GGML ConvRot large-MLP graph (4056 rows, K4096/N24576) passed its
seven tiny independent cases and large-shape checks. Warm quantization was
2.290–2.299 ms, INT8 GEMM 4.247–4.314 ms, combined graph 5.054–5.079 ms. Its
different-precision dense comparison remained separate (relative L2 0.008677).
Xe peak was 1,827,667,968 bytes and cgroup peak 1,264,758,784 bytes, swap zero.
These are synthetic kernel results, not whole-image speed acceptance.

The private runtime environment must leave `SYCL_DUMP_IMAGES` unset: the bundled
2026.1 runtime still attempted to dump SPIR-V when it was set to `0`, despite
newer upstream documentation describing zero as disabled. Landlock denied that
write; omitting the setting fixed initialization without relaxing restrictions.
Dynamic vendor constructors also run before the worker's own privacy setup, so
tests apply restrictions before `exec` and the worker rechecks before input.
The first image launch exposed an omitted MKL dependency under Rust's
`--as-needed`; ELF startup validation is required in addition to a passing link.
The small-image acceptance and full-edit decode failure are recorded below;
no full reference recipe has been accepted on SYCL yet.


### Private SYCL image acceptance and decode diagnosis (2026-10-01 UTC)

The private executable SHA-256 is
`643d56a8271429baa70948c72802c34ff647b39536c8955cb26e8be2715d68d7`.
It uses the exact ABI-160 static engine/GGML and reviewed dynamic vendor closure;
it is never installed as the standard gateway executable. Full tests omit
vendor verbose data, use write restrictions before exec, disable core dumps,
check no swap before input, and retain all content-bearing bytes only in RAM.

A synthetic 512×512/four-step/CFG1 text generation passed in 39.305 s with all
four previews and a metadata-free RGBA result. Sampling step one completed at
36.774 s and step four at 37.512 s. Xe peak was 8,882,982,912 bytes, cgroup peak
1,315,393,536 bytes, swap zero. This includes cold initialization/JIT and is not
a full-workflow performance claim; the generated pixels were discarded.

The original 1248×832/25-step/CFG4 edit completed all sampling steps at
105.482 s, then exited 1 at 107.518 s before final delivery. Xe peak was
24,754,106,368 bytes, cgroup peak 1,921,179,648 bytes, swap zero. Two reduced
same-size/two-step reproductions failed at 52.582/52.603 s after sampling ended
at 50.560/50.564 s. The latter's bounded numeric/static filter identifies
`ggml-sycl.cpp:5276`, the compute-forward SYCL exception handler, and opcode
`IM2COL`. This narrows the failure to VAE decoding without recording raw exception
text. It does not establish OOM or a driver defect. The pinned handler exits
rather than returning a recoverable GGML status, so VAE allocation fallback and
the shim's ordinary error mapping cannot handle it.

A synthetic two-reference test was cooperatively cancelled at sampling step
one while investigating the kernel history; it reported `image_cancelled`,
exited without forced termination and released the worker. Its Xe peak was
31,528,890,368 bytes, cgroup peak 1,962,901,504 bytes, swap zero. It is cancellation
evidence only, not successful multi-reference acceptance.

The kernel has six reset entries this boot: four historical entries on Sep 27
and two at 2026-10-01 01:23:16.258/01:24:11.366 UTC. Those two preceded the accepted
arithmetic/attention kernels and full-edit tests. No further entry appeared
after the reduced decode reproduction. Their originating operation is not yet
established; they are not evidence that the later full edit wedged the device.
The optional managed route remains disabled pending both decoding and stability
acceptance. The installed standard gateway was resumed and remained healthy;
exclusive test windows pause admission temporarily and must always resume it.

Source-only managed-runtime support is in `9a51d46`, with installer/environment
review fixes in `7fe6283`. It preserves default native routing and old collections,
restricts vendor initialization before input, and degrades images without
preventing ordinary Ollama startup if the optional runtime is absent. The full
source suite before the review fix passed 182 tests with nine hardware-dependent
ignores; the review fix passed 19 focused runtime/deployment tests and Clippy.
These do not prove relocated vendor-library, live managed cancellation or
installed-service acceptance. See [the optional runtime contract](kuno-managed-sycl.md).


## SYCL full decode and larger-copy boundary (2026-10-01 UTC)

The bounded IM2COL candidate worker SHA-256 is
`8a6d07e18aefdc77fdbbbfface1e87085132378b82c6d8a06c11a7e7fdfb1a70`.
The full original 1248x832, 25-step, CFG4 edit passed in **109.624 s**, with all
25 previews, metadata-free RGBA final output and alpha range247-255. The final
image was reviewed through a RAM-only, no-store loopback server and discarded;
no image file was saved. Step1 was at48.373 s and step25 at105.771 s, about2.39 s
per subsequent step. Xe peak was 24,754,114,560 bytes, cgroup RAM peak1,930,915,840
bytes and swap 0. This was a clean run without native compilation. The equivalent
accepted CUDA edit is160.702 s; user ComfyUI Intel cold/warm baselines are78.25 /
64.48 s. SYCL is substantially faster than Kuno's earlier paths but does not yet
match the user's ComfyUI timing.

The original 1184x1776, 50-step, CFG1 generation completed sampling at136.939 s
but exited-6 at140.988 s during decode, with the fixed
`UR_RESULT_ERROR_DEVICE_LOST` classification. Xe peak was22,143,942,656 bytes,
cgroup RAM peak1,328,771,072 bytes, swap 0. One new Xe engine-reset entry at
02:52:56.688688 UTC raised the boot count from6 to7. This reset matches the
large-generation test window; the earlier two entries remain separate. No
production configuration or worker was changed. Admission was resumed afterward.

Independent IM2COL modes 5–8 pass on the B70, including the exact large geometry,
10,900,832,256-byte output, samples across2/4/8GiB and INT32/UINT32 element
boundaries, and bounded-launch chunk boundaries. All runs used Landlock and
zero-swap cgroups; no further Xe reset appeared. Exact-case C144/C288 modes
passed249/255 samples; the UINT32-bracketing cases passed252/255 samples.
Embedded worker SPIR-V also retains64-bit offsets/access-chain indices. These
results do not reproduce other asynchronous VAE operations.

The actual CPU-only VAE graph has seven non-contiguous F32 CONT operations
crossing signed-int byte offsets in the larger workflow, versus zero in the
passing edit. At this historical pre-copy-fix checkpoint, the legacy SYCL copy path narrowed
byte strides/offsets to int, and focused copy-boundary validation was pending.
The optional runtime was then uninstalled and unaccepted. The following final
wide-copy and installed receipts supersede that status.


## Wide-copy fix and packaged Intel acceptance (2026-10-01 UTC)

Commit `30cebda` widens SYCL copy dimensions, indices, byte strides and offsets
through every wrapper/kernel. The independent actual-VAE graph has seven
noncontiguous F32 CONT nodes crossing signed 32-bit byte offsets at 1184x1776;
the largest has 605,601,792 scalar elements (within the work-item bound) but
2,422,407,164-byte last offsets. The passing edit has no such wide CONT node.
Copy scalar conversion and quantization math/routing are unchanged. Combined
with bounded IM2COL in `46cbd5c`, this removes both demonstrated decoder limits.
The old device-lost/reset receipt above remains historical evidence; it does
not establish that every possible driver fault has been fixed.

Independent copy acceptance passed 5,785 CPU offset checks and 630 CPU numerical
checks. B70 GPU mode 1 passed all 630 checks. Modes 2/3 passed eight source,
destination and conversion cases with ten checks each around byte strides
2,147,483,712 and 4,294,967,360, including untouched canaries. GPU peaks were
2,299,695,104 /4,448,436,224 bytes and swap 0. BF16 numerical GPU coverage and
arbitrary copies above INT32_MAX work-items remain unverified.

The final worker SHA256 is
`414b86a3c498c0f848321766e99784a8d5d7fd7f3984c1bf35796f6b5b14796b`.
Its flat runtime manifest SHA256 is
`94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`:
54 regular members, 531,417,369 bytes (worker, 23 library/alias names and
30 exact license/notice files). Patch003 SHA256 is
`7f6a75c75544c0e3776d780b0710cab2fabb94ddfe57eeb436dff02891bd4771`.
ABI160 and engine/GGML pins remain unchanged. The standard gateway does not link
the vendor runtime. Its tested SHA256 is
`d2a7a293bb021daebb1dda2c8b21a67443881a10ce0787dbd4b7ae2c4946d8dc`.

All following final-worker tests hid the entire SDK extraction using systemd
InaccessiblePaths, loaded only the flat regular-file runtime, restricted writes
before exec, disabled core dumps and used zero-swap cgroups. All 21 observed
vendor library families loaded one distinct inode each; no SDK fallback or
concurrent heavy compilation/inference occurred. Each request used a fresh
worker, so these are process-fresh timings, not a persistent warm-model claim.
OS file caches and prior generic GPU initialization were not flushed.

| Accepted workload | Total | GPU peak bytes | Cgroup RAM peak bytes |
| --- | ---: | ---: | ---: |
| Original generation, 1184x1776, 50 steps, CFG1 | 153.950 s | 22,144,405,504 | 1,240,580,096 |
| Original edit, 1248x832, 25 steps, CFG4 | 115.513 s | 24,754,569,216 | 1,961,013,248 |
| Two synthetic references, 512x512, 4 steps, CFG4 | 61.282 s | 31,394,676,736 | 2,040,250,368 |

Every result passed PNG framing, dimensions and metadata checks, including
50/25/4 previews respectively. Generation alpha was 251-255, edit 249-255,
two-reference 242-255. The two original final images were visually inspected
through a RAM-only no-store loopback server, with no visible decoder seams,
then discarded. A reduced two-step original generation passed first in 31.471 s.
Full-size edit cancellation at step 3 exited 1 with the fixed `image_cancelled`
label, reaped cooperatively in 3.061 s, and used no forced termination. Its GPU
peak was 24,754,556,928 bytes, cgroup RAM 1,603,768,320 bytes, swap 0.

Kernel event count remained 7 after all final-worker GPU tests: no further reset
beyond the earlier failed decode. This bounds the observed acceptance window;
it does not prove indefinite driver stability. ComfyUI user-supplied Intel
cold/warm baselines remain 132.06/100.65 s generation and 78.25/64.48 s editing.
The final SYCL edit is much faster than Kuno Vulkan 558.176 s and CUDA 160.702 s,
but neither accepted full SYCL workflow reaches ComfyUI parity.


## Installed service acceptance (2026-10-01 UTC)

Release follow-up: after idle, a fresh installed 512x512/two-step keyless stream
passed two previews/final in 40.870 s (first preview 38.327 s), plus IPv4/IPv6/LAN
and control 401. Client ran in a zero-swap user cgroup. Service remains
active/running with NRestarts 0 and MemorySwapMax 0; kernel Xe event count remains
seven, with no new reset. Release packaging adds notices only; executable/library
bytes are unchanged. Exact new archive/manifest hashes and integration steps are
in [kuno-release-handoff.md](kuno-release-handoff.md). Beta readiness does not
claim overnight/indefinite stability or ComfyUI parity.

The standard gateway and final flat runtime above were installed through the
owned-file systemd transaction. Runtime installation preserved the configuration
byte for byte; explicit image configuration changed only engine, runtime and PCI
to B70. Collections, model store, listener addresses/ports, Ollama workers and
zero LLM response/idle deadlines were preserved. The global engine default
remains native; this is an explicit local SYCL selection, not a release.

Installed checks passed:

- Keyless Qwen 2.1 512x512/two-step NDJSON: started, two valid metadata-free PNG
  previews, final result; first preview 38.865 s, total 41.067 s. Discovery/version
  work through IPv4, IPv6 and LAN; unauthenticated control returns 401.
- All three workflows report ready and standard attention only; explicit fast
  receives 400 `image_attention_unsupported` before inference.
- Native Qwen 2512 on the newly selected B70: 512x512, two steps, CFG1, shift 3.1;
  two previews and final PNG passed in 27.328 s. This verifies routing, not the
  original 100-step 2512 quality/performance recipe.
- Ordinary Ollama generation after images passed in 1.077 s. A separate 20-step
  image stream was disconnected after the first preview; child cleanup/admission
  completed in 1.003 s without recovery quarantine. The immediate next Ollama
  generation passed in 0.935 s. These one-token compatibility probes do not
  replace earlier long-context/deadline acceptance.
- Temporarily hiding only the optional runtime made Qwen 2.1 workflows unavailable
  and generation return 503 `image_managed_runtime_unavailable`; ordinary Ollama
  version stayed 200 and native 2512 stayed ready. Restoring the directory restored
  all workflows without service restart. Synthetic source tests cover corruption
  and activation rollback; no live corrupt binary or new driver fault was induced.

The initial cancellation observer expected a nonexistent recovery_pending field
under admission; correcting that harness assumption and repeating cancellation
plus the following LLM request passed. The final health observer also accounts
for managed Ollama workers legitimately stopping while idle. These were test
harness corrections, not production failures.

Final installed checks: active/running, NRestarts 0, MemorySwapMax 0, admission
unpaused, active false, waiting 0, recovery_required false, no image worker left.
The reachable backend has no resident models; inactive managed backends refuse
connections as expected. Xe kernel event count remains 7. Known synthetic request
text, encoded-image field names/data prefixes and the control credential are
absent from the service journal during installed acceptance. Temporary image
settings and test runtime renames are removed. Content-bearing clients/workers
used zero-swap cgroups; no generated image, reference copy or prompt was saved.
This is a bounded privacy check, not a forensic guarantee for unrelated software.

The API/discovery shapes and Vashti connector/UI are unchanged. Existing saved
CUDA fast recipes must select standard for this Intel route. Source remains on
codex/kuno-image-performance; concurrent Vashti/untracked coordination work is
preserved. No merge, push, application deployment or public release occurred.


## Resident full-workflow baseline and experimental speed candidates

This checkpoint measures the accepted installed ABI 2 resident worker from
[`kuno-image-residency-build.json`](kuno-image-residency-build.json), before the
experimental source changes below. Original B70 workloads retained their requested
dimensions, steps and CFG. The benchmark ran without concurrent heavy compilation
or inference; one worker PID served all four requests. Process-cold includes worker
initialization/model load, with OS file caches uncontrolled. The first edit reused
loaded image models but was the first vision/edit request in that process; it is
not a process-cold edit measurement.

| Original workload | First request | Resident repeat | User-reported ComfyUI warm |
| --- | ---: | ---: | ---: |
| Generation, 1184x1776, 50 steps, CFG1 | 157.208 s, process-cold | 135.547 s | 100.65 s |
| Edit, 1248x832, 25 steps, CFG4 | 101.958 s, model-warm/vision-first | 76.952 s | 64.48 s |

| Resident repeat | First-to-last preview span | Last preview through decode/cleanup |
| --- | ---: | ---: |
| Generation | 125.142 s | 5.852 s |
| Edit | 63.003 s | 2.864 s |

The preview span excludes work before the first preview; it is not a complete
sampling-stage profile. All four requests delivered the expected final dimensions
and all 50/25 previews per generation/edit request. Private input/output/comparison bytes stayed in RAM,
no prompt/reference/generated-image files were saved, and swap remained zero.
The kernel event count was seven before and after the full baseline. These
resident timings supersede the earlier small-image smoke as the full-workflow
baseline, without invalidating historical fresh-process receipts. Both warm
workflows remain slower than the user's ComfyUI measurements. The ComfyUI values
are user-reported comparison points, not a matched runtime/precision/cache
experiment or a parity claim.

### Experimental source and CPU validation

The combined candidate tree is `f6aa880`; the isolated source-check tree
`400e1eb59a6c8106f5f8c5b468b8eeb0e51a501d` matches it exactly. Candidate patch
checksums are:

- `001-qwen-quantized-config.patch`:
  `ed75739ccb193099f23d4600cc4a8846f9d17fe6c2cec7591a7c9cb30c2d0134`.
- `003-sycl-convrot.patch`:
  `f97db2f0851df733e202fb7e65f6d98b75eee3d8b1e212299bde8ddf484c7b7b`.

Patch 001 shares the already permuted positional matrix within one Qwen 2.1
compute graph, removing 63 repeated materializations. It retains no positional
tensor across steps or requests; see the
[graph-local sharing test and lifetime notes](../crates/kuno-image/native/rope-sharing.md).
Patch 003 skips F16 K/V packing independently only when the input is contiguous,
non-null and aligned to the backend's 128-byte `SYCL_BUFFER_ALIGNMENT`. Strided,
unaligned and other-type inputs retain their previous packing/conversion paths.
That alignment is a conservative fast-path policy, not a claimed universal
oneDNN requirement. Requested arithmetic, resolution, steps and CFG are unchanged.

Default `cargo test --offline --locked -p kuno -p kuno-image -p kuno-protocol --
--test-threads=1` passed 205 tests, zero failed, nine ignored. Default all-target
Clippy with warnings denied, formatting and diff checks passed. Rust/Cargo are
1.95.0. The expanded attention oracle passed 101 CPU cases plus two exact cache
checks, maximum relative L2 0.00170561. Twelve CPU pointer-policy cases cover
null/aligned/unaligned inputs and independent K/V eligibility; the shape envelope
also passed 199 accepted and 47 rejected cases. Direct attention cases include
34-byte and 128-byte F16 view offsets with independent dense/strided K/V layouts.
These source/CPU results do not establish B70 correctness or a speed improvement.

The private SYCL worker build at `f6aa880` completed without GPU invocation.
Its SHA-256 is
`db8c2368b5f4b4a31f8f0b25a513de2e9291b0302c43a55a41d9a8d33e7dc50d`
(135,504,672 bytes); the rebuilt engine archive SHA-256 is
`56fc04019678ae50f35407d6b0891bdc1882720ac2041d667e6b1001e3be4e6e`.
These identify unaccepted candidate build artifacts, not an installed runtime,
approved release package or successful hardware execution.

### GPU failure boundary and pending acceptance

The initial matched 85-case GPU oracle passed against the unchanged accepted
SYCL archive, maximum relative L2 0.0012553. The original candidate, which allowed
unaligned contiguous F16 aliases, then failed the direct SDPA offset-layout
numerical assertion at test line 269, exit -6. Output buffering prevented exact
identification of the failing offset/layout combination. This supports restricting
the fast path, but does not prove alignment is the sole cause.

The guarded 101-case oracle has line-buffered, flushed numeric results. Its
unchanged baseline aborted before the first reported case during the subsequent
test window. At that checkpoint the alignment guard was unvalidated on hardware;
there was no accepted candidate speed result or valid matched GPU timing.
The post-reboot checks below supersede this pending numerical boundary.

| Kernel observation | `engine_reset` | `job_timeout` | Total |
| --- | ---: | ---: | ---: |
| Before and after the controlled full baseline | 7 | 0 | 7 |
| After the later oracle diagnostic window | 10 | 93 | 103 |

The new events are associated in time with the experimental test window. These
counts do not establish which operation caused them, whether an assertion/context
exit contributed, or the underlying driver/kernel cause. Numerical assertions
abort without normal test teardown, and the existing SYCL backend uses `wait()`
rather than `wait_and_throw()` for synchronization, limiting deferred-error
diagnosis. Rejected and guarded binaries, archives and content-free numerical
receipts are preserved. This diagnostic window used kernel 7.0.0-34-generic;
GPU testing was paused pending B70 recovery.

An accepted-only small compute smoke later timed out after 45 seconds and
added 93 classified job timeouts. A targeted driver recovery helper was prepared,
but the retained visible execution history does not confirm that it ran.
The user subsequently reported a black screen and powered off/rebooted.
The previous boot journal is unavailable, so the cause of the black screen
is unresolved; do not attribute it to a monitor, a particular test or a rebind.

The installed runtime and configuration remain unchanged. Standard/private
candidate build receipts are separate from hardware acceptance. Complete device
recovery, numerical and original-workflow cold/warm comparisons, protected
runtime/privacy checks and cancellation/reuse acceptance before selecting either
source optimization. Do not include these candidates in a release or advance
the packaging helper's current accepted pins yet. Historical build/manifest
receipts retain their original checksums. No Vashti code or API change is involved;
the [release handoff](kuno-release-handoff.md#experimental-performance-candidates)
records this pending boundary.

### Post-reboot guarded attention checks

The reboot selected kernel `7.0.0-38-generic`. The unchanged accepted two-case
compute smoke passed in 6.293 seconds. Both accepted and guarded SYCL archives
passed all 101 attention cases plus two exact cache checks, with maximum relative
L2 0.0012553. No new classified Xe reset/wedge/hang/timeout events appeared.
These are numerical tests, not whole-workflow or stability acceptance.

A matched 32-head attention microbenchmark used 4056 queries, 8192 keys and
128 channels, with one warm-up and three measured synchronized graph evaluations.
Compilers and other inference were idle. Median time was 10.584 ms accepted
versus 8.837 ms guarded (16.5% lower, 1.747 ms saved). This suggests only a few
seconds across a full workflow; it does not establish an end-to-end improvement.
The oneDNN/JIT warm-up is excluded from these measured samples.

Temporary standard build files were erased by the reboot. The standard RoPE
candidate was restored into the persistent producer worktree, rebuilding all
57 C++ units and verifying the 15 C objects and five unchanged GGML archives.
See the [persistent producer receipt](../crates/kuno-image/native/rope-sharing.md).
Private SYCL candidate files remained in persistent `target/`; installed gateway,
runtime and configuration remain byte-identical to the accepted residency build.
The restored CUDA RoPE oracle passed 40 cases in 4.921 seconds. The new safe
SYCL oracle at source `dd60ee2` passed all 40 device cases in 6.576 seconds,
including explicit finite outputs, bitwise legacy/shared equality and input
immutability. Numerical failures unwind through fixed phases; backend exceptions
conservatively skip uncertain release. This does not strengthen the underlying
SYCL `wait()` to `wait_and_throw()` or handle vendor fatal assertions.

The private combined candidate passed a two-step full-resolution generation/edit
preflight under a read-only namespace runtime overlay. The installed runtime was
unchanged; the SDK was inaccessible, swap/core dumps disabled, and all expected
previews and metadata-free final PNGs arrived. One worker reused its loaded
models for all four requests and exited normally, with no classified errors or
new kernel events. Original 50/25-step matched execution subsequently passed; detailed results
follow below. Cancellation acceptance remains pending. Historical producer receipts are retained rather than repinned.

### Matched original-workflow execution after reboot

On kernel `7.0.0-38-generic`, the accepted resident worker and combined `f6aa880`
candidate each served first/warm generation then first/warm editing in one
process. Original dimensions, 50/25 steps, CFG 1/4, seeds and references were
matched between variants. There was no concurrent compilation/inference during
the timed comparison. OS file caches were uncontrolled. The edit first request
was model-warm/vision-first, not process-cold.

| Workload | Accepted first | Candidate first | Accepted warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| Generation, 1184x1776, 50 steps, CFG1 | 153.472 s | 150.465 s | 135.450 s | 134.130 s |
| Edit, 1248x832, 25 steps, CFG4 | 101.150 s | 98.063 s | 76.872 s | 73.673 s |

Warm first-to-last preview spans fell from 125.087 to 121.944 seconds for
generation and from 62.959 to 59.966 seconds for editing. Generation's first
preview instead rose from 4.500 to 6.330 seconds in this pair. The cause of that
startup difference is unresolved. End-to-end gains are therefore just 1.320
seconds generation (0.97%) and 3.199 seconds editing (4.16%). This single pair
does not establish a repeated performance distribution or ComfyUI parity.

All eight requests passed dimensions, all expected previews, metadata-free PNG
integrity, resident PID reuse, zero swap/core limits, NoNewPrivs and absence of
writable regular-file handles. SDK paths were inaccessible. Each variant exited
normally; there were no classified worker errors or new Xe events. The private
candidate overlay affected only its mount namespace; installed binaries/runtime
and configuration remain unchanged. Prompts, references and output/comparison
pixels remained in RAM; the ignored receipt stores numeric measurements only.

Matched-output differences were measured in 8-bit channels: mean absolute delta
0.1455/0.1652 for generation first/warm, and 1.0272/1.0087 for edit first/warm;
maxima were 21/129 and 203/209 respectively. These are **not bitwise equality**,
not a visual-quality judgment, and cannot be attributed to ordinary backend
nondeterminism without an appropriate full-step control. The earlier two-step
rounding-scale repeat does not establish a tolerance for these longer workflows.
Full-step accepted same-seed controls subsequently passed as the final GPU test;
see the control receipt below.

The candidate remains experimental/unselected. Warm sampling is modestly
faster, but the measured whole-workflow gain is small and output equivalence
plus cancellation acceptance are unresolved. No installation, packaging pin
advance, merge, push, publication or Vashti/API change is authorized by this
receipt. The user requested returning the GPUs after the current control run;
no further GPU work will start in this session.

### Prepared next investigations; no further GPU work

A separate guarded F32 residual MUL→ADD fusion is prepared on
`codex/kuno-image-fused-residual`, based on `f6aa880`. It preserves separate
rounded multiply/add and original operand order, rejects unsafe shapes, views,
partial aliases and device/buffer mismatches, and adds no allocation or cache.
CPU policy testing found a view-escape gap in the generic fusion helper; the
prototype now explicitly rejects that case. Source commits are `4d603ab` and its required safety follow-up `ecd532f`.
The CPU policy passes 141 checks, and eight independent materialized-F32
arithmetic cases plus pre-loader argument rejection pass. Exact source/CPU
receipt hashes are in `kuno-image-speed-candidate-build.json`. It has no native SYCL build,
GPU arithmetic result, full-workflow timing or acceptance. Do not cherry-pick
only its original source commit before the safety follow-up.

A diagnostic-only one-step op profiler is preserved at
`/home/jmicko/.codex/worktrees/kuno-image-gap-review/vashti/target/sycl-op-profile-v1`.
Its `diagnostic.patch` SHA-256 is
`f2b6d0ad691c12e9aae7b86ff907dcd8dcbde9299895f46b9b061c4409e8893f`;
`checkpoint.json` SHA-256 is
`5553fd63b0c4494a509b5479570d88962193bb1c6f498e1ae5a5770f4f602c71`.
Nineteen source/policy checks and eight CPU fake-event cases pass. The proposed
SYCL tags preserve normal dispatch/fusion and aggregate fixed op enums in bounded
RAM without per-op waits or tensor names/shapes. Native compilation, device tag
support, math equivalence and overhead remain unverified. The prototype is
ignored and unselected, not a production feature or release artifact.

Resume by confirming the GPUs are available, checking live kernel/service state,
then verifying full-step numerical drift and measuring the dominant bottleneck.
Root remains the sole owner of GPU/service tests. Preserve all persistent
producer directories and source proof; previously temporary artifacts were lost
on reboot. Existing Vashti work and its uncommitted coordination notes are
untouched.

### Final same-seed control and stopped checkpoint

The accepted full-step worker repeated each original recipe with identical seed
in one process: generation cold then warm, followed by edit vision-first then
warm. All requests, previews, dimensions, privacy checks and normal shutdown
passed, with no new kernel events. Other CPU compilation was allowed during
this numerical control; these control elapsed times are not a new matched
performance comparison. Output bytes stayed only in no-swap RAM.

| Accepted same-seed pair | Mean absolute 8-bit channel delta | Maximum | Changed fraction |
| --- | ---: | ---: | ---: |
| Generation, 50 steps | 0.0000083223 | 1 | 0.0000083223 |
| Edit, 25 steps/CFG4 | 1.780833 | 235 | 0.5361793 |

The existing full edit path therefore varies substantially even before these
optimizations. This control does not separate numerical nondeterminism from
cold/warm request state, and cannot by itself excuse the candidate changes.
Generation's candidate deltas are much greater than this accepted repeat;
output equivalence is still unaccepted. Investigate the two source optimizations
separately, and compare same-process warm/warm plus inter-process controls
before attributing a cause. No additional GPU tests ran after the user's stop
request. Native fusion and profiler device builds/tests are deferred.

Final read-only audit: installed accepted gateway/manifest/all members match,
Kuno active with PID 2032, zero automatic restarts, zero service swap, admission
unpaused with no active/waiting requests or recovery latch, no image workers
remaining and benchmark unit inactive. The benchmark GPUs are released.
Source remains on `codex/kuno-image-speed`; the main Vashti checkout and its
uncommitted frontend/coordination work were preserved. The next session needs
a fresh GPU-availability check before any device work.

### Continued bounded Intel investigations (2026-10-02)

The user renewed exclusive access to both cards after the earlier stopped
checkpoint and requested no action requiring a restart. All device work in this
session uses the B70 on kernel `7.0.0-38-generic`; no driver reset, detach/rebind,
package/driver installation, reboot or installed Kuno replacement is performed.
Private clients remain in swap-disabled/core-disabled transient units, and
mount overlays affect only test processes. The installed accepted worker,
public API and concurrent Vashti checkout remain unchanged.

A diagnostic-only step-2 profiler passed its synthetic tag/capability probe and
three-step original-resolution generation/edit runs. These are shortened
profiling runs, **not new full-workflow timing or candidate acceptance**. There
was no overlapping compilation or inference. Each returned every preview and a
valid metadata-free final PNG, reused/cleared content normally and shut down
without new classified Xe events. The profiler records fixed GGML operation
numbers/counts/nanoseconds only; request content remained in no-swap RAM.
Queue tag intervals include submission gaps and fused work, rather than pure
individual kernel time. Profiling overhead and full math equivalence are still
unaccepted; the diagnostic runtime must not be selected for release.

| Operation | Generation step-2 interval share | Edit step-2 interval share |
| --- | ---: | ---: |
| Matrix multiply | 23.23% | 22.80% |
| Attention | 18.05% | 18.19% |
| CONT | 14.71% | 17.10% |
| ConvRot quantization | 12.27% | 12.25% |
| CPY | 6.10% | 5.77% |
| REPEAT | 5.91% | 3.34% |

Generation's summed intervals were 2.435696 seconds; editing's were 2.459109
seconds across both conditioning calls. Source-derived accounting identifies
128 RoPE repeat nodes, 128 RoPE CONT nodes and 128 much smaller modulation CONT
nodes per diffusion graph. That is graph accounting, not measured per-node
attribution; the number of small nodes does not establish their cost.

Paired SDPA tests keep one graph/input allocation and compare five routing
modes: legacy packing, aliasing, packing only K, packing only V, and making both
copies while feeding the eligible aliases. Eleven small/layout/view/canary
cases, each evaluated twice per mode, produced 143 exact comparison rows.
The 4056-query/8192-key/32-head/128-channel case produced another 13 rows.
All output bits matched, with maximum ULP difference zero and no new GPU fault.
These results narrow the aliasing investigation but do not prove whole-pipeline
equality or explain the earlier generation pixel drift.

A row-local ConvRot quantizer prototype preserves the old arithmetic, scale
bits and INT8 packing while caching rotated values in work-group local memory.
The actual B70 reports 131072 local-memory bytes and maximum work-group size
1024. All 164 small cases and four cases at each of 4056/8214 rows passed exact
packed bytes, scale bits, padding, graph/direct outputs and input immutability.
However, a compiler-idle, alternating 3-warm-up/7-measurement test rejects this
candidate on performance:

| Real quantization graph | Legacy median | Local-row median |
| --- | ---: | ---: |
| K4096, 4056 rows | 0.731557 ms | 2.294512 ms |
| K4096, 8214 rows | 1.510144 ms | 4.575682 ms |

The measurements include submission, eligibility queries and synchronization;
they are not end-to-end times. The approximately threefold regression is also
present in the direct route. Small-row gains are minor and do not justify this
prototype for full images. Keep the legacy quantizer; no production worker is
built from this rejected experiment. No new classified Xe events appeared.

The first explicitly rounded residual fusion failed special-value comparisons.
Replacing explicit round-to-nearest intrinsics with the backend's ordinary
operations preserves the residual-first cases, but the corrected strict host
oracle still detects NaN payload/sign differences in reverse-operand cases.
The earlier phase-12 host control failure was separately caused by Intel host
compiler defaults: pre-backend FMA distinction failed with FTZ/DAZ enabled.
A test-only strict host relink fixes that control without changing the device
archive; it passes all ten residual-first device cases, then fails the reverse
special-value case. Neither unrestricted fusion is accepted. Exact checks and
fallback coverage are required before any narrowed implementation is tested.

Numeric receipts, binary/source hashes and status are recorded in
[kuno-image-speed-candidate-build.json](kuno-image-speed-candidate-build.json).
The profiler, paired SDPA and quantizer producer artifacts remain in isolated
persistent worktrees. This session is still in progress; final audit and any
additional accepted or rejected experiments will follow below.

The installed accepted worker also completed a two-step warm-up followed by
two identical full-step requests for each original recipe in one process.
This tests warm/warm variation rather than the prior cold/warm pair. Generation
had maximum channel delta 1, mean delta 0.00255007 and changed fraction
0.00255007. Editing had maximum 255, mean 2.591600 and changed fraction
0.581892. These are individual controls, not a validated tolerance/distribution
or evidence of an identified cause. Candidate generation drift remains much
larger than this control. CPU compilation was allowed, so elapsed times are
not performance evidence. All six requests passed preview/PNG/dimension/privacy
checks, reused one PID and shut down normally; kernel-event delta remained zero.

The narrowed residual-first implementation at `7579f98` subsequently passed
all 20 protected device fixtures in 7.732 seconds. Test-only linker wrappers
prove forward graphs actually use one fused call and no ordinary MUL/ADD,
while reverse graphs make no fused attempt and execute the original two calls.
Supported special values, both full-size geometries, exact activation/residual
aliases, input immutability and unsupported-order fallback remain checked.
There was no kernel event or uncertain cleanup. The production worker contains
no test wrappers/counters; all 63 vendor/library/notice members are unchanged.
The private ABI 2 runtime and build receipt are checkpointed at `f6c917d`.

An eight-request two-step generation/edit preflight compared that runtime only
against the existing `f6aa880` runtime, keeping their inherited engine changes
identical. All previews, PNG dimensions/integrity, content cleanup, PID reuse
and normal shutdown passed. Generation output differences were bounded by one
channel level. Edit differences were much larger, as in accepted controls;
this is not a validated numerical tolerance or final quality acceptance.
The original full 50/25-step performance comparison is still pending below.

The private cancellation harness was separately checked against the installed
accepted engine: SIGTERM after the first original-resolution edit preview
returned `image_cancelled` and exited with status 1 in 3.011 seconds, without
forced termination. A fresh worker then completed a two-step edit and normal
idle shutdown. Privacy checks and zero new Xe events passed. This validates
the harness/control, not cancellation of a new candidate.

### Residual-first fusion original-workflow pair

All eight original 50/25-step requests passed against `f6aa880` with compilers,
package work and other inference idle. The private candidate differs only in
residual-first SYCL fusion; both variants inherit the same shared-RoPE/SDPA
engine. One worker served first/warm generation then first/warm editing per
variant. OS file caches remain uncontrolled, and edit-first is vision-first,
not process-cold.

| Workload | f6 first | Fusion first | f6 warm | Fusion warm |
| --- | ---: | ---: | ---: | ---: |
| Generate, 50 steps/1184x1776/CFG1 | 150.390 s | 148.564 s | 132.279 s | 130.355 s |
| Edit, 25 steps/1248x832/CFG4 | 98.469 s | 97.860 s | 73.638 s | 72.848 s |

Warm reductions are 1.924 seconds (1.45%) and 0.790 seconds (1.07%), respectively.
This single pair is a small improvement, **not ComfyUI parity** or a performance
distribution. Every preview, metadata-free PNG/dimension check, PID reuse,
content cleanup and normal idle shutdown passed; zero swap/core/writable-file
checks and SDK isolation passed. There were no classified errors or Xe events.

Generation first/warm mean absolute channel differences were 0.1447449/0.1670824,
maxima 25/106, and changed fractions 0.1326949/0.1402320. Edit means were
1.7476395/1.2955953, maxima 227/213 and changed fractions 0.5266101/0.4660250.
These are not bitwise equal, and generation differences substantially exceed
same-process accepted controls. Their magnitude resembles the earlier
accepted-versus-f6 pair despite the new comparison sharing its RoPE/SDPA code.
An identical accepted-runtime control across separate worker processes is
therefore required before assigning the drift to either source optimization.
That control is pending; no source/precision cause or tolerance is established.
The candidate stays unselected and uninstalled. Cancellation and additional
non-dyadic finite arithmetic controls also remain pending.

### Identical workers in separate processes and further controls

Two identical installed accepted runtimes, each in its own worker process and
identical protected mount configuration, completed the original first/warm
generation pair. Matching requests across processes had mean channel
differences 0.1420189/0.1406261, maxima 28/51 and changed fractions
0.1307590/0.1251452. This resembles the candidate's inter-process generation
differences and shows the larger variation exists without these source changes.
It does not identify the kernel/cause, establish a tolerance/distribution, or
prove every candidate change harmless. Compilation overlapped this control;
elapsed times are not performance evidence. All four requests and shutdowns
passed, with no new classified Xe event.

The private residual-first fusion passed first-preview edit cancellation in
2.906 seconds (normal positive exit status1, no forced termination) and a fresh
worker completed the two-step edit/reap recovery. Zero swap, core limit, writable
file, SDK-isolation and transient-content checks passed; kernel delta stayed zero.
The expanded non-dyadic finite arithmetic corpus remains pending.

The separately gated REPEAT-to-MUL bypass at `9b563e3` passed all 50 device cases
and 528 all-bit comparisons in 3.493 seconds. Sixteen actual prepared-PE Q/K RoPE
graphs, intermediate output/view/use escapes, repeat execution, signed zeros,
NaN payloads/classes and input immutability passed. Counters prove 74 eligible
operations actually bypassed the copies: 72 general and two large unravel
dispatches, across 25 finite and 25 special-value cases. Existing MUL routing
uses a different zero-stride kernel/grid from the materialized legacy input;
CPU address checks alone were not considered GPU arithmetic proof. There was no
fixed error, cleanup uncertainty or new Xe fault. The feature remains default
off and is not present in an accepted production worker; timings/full workflow
and cancellation remain separate acceptance gates.

The compiler-idle repeat-pair microbenchmark used the same graph and allocation
for both modes, three warm-ups, then five measurements of four evaluations.
Legacy median was 3.347200 ms/four evaluations; bypass median was 2.295120 ms,
a 31.43% reduction for that isolated L4096 pair. It includes host submission and
synchronization, not a pure kernel or whole-image measurement; modes were grouped
rather than alternated. All numerical/routing checks passed again. Independent
review then found unsafe host-vector unwinding if a device call threw or returned
a failure status in the test executable. No such failure occurred in either run;
these are clean-path receipts only. A new test-only fatal-boundary revision is
required before further GPU use. Device resources alone do not protect host DMA
buffers; preserve process lifetime or exit without C++ unwinding on uncertainty.

### Corrected arithmetic boundaries and attention-only full generation

The residual oracle expanded to 48 exact device cases. New finite families use
full mantissas, all finite exponent encodings, rounding ties and cancellation;
large 4056/8214 cases and both operand orders are retained. CPU coverage passes
20 graph cases, 58 known rounding fixtures and 250,000 strict-host cross-checks
against an independently reviewed integer binary32 nearest-even reference.
The corrected v5 device-boundary oracle at `56986ab` passes all 48 in 13.326 s.
Its22CPU failure/generator fixtures include nine no-unwind witnesses: uncertain
backend exceptions/statuses now exit at the call boundary before host DMA
vectors can unwind. The production archive is unchanged. The repeat oracle's
same correction at `f484871` also passes 50/528 in 3.364 s. Earlier clean v3/v4
results remain historical numerical receipts, not failed-path safety claims.
The underlying backend wait() can still omit deferred device errors; kernel
monitoring supplements it without claiming every possible driver fault detected.

The attention-only control restores the accepted engine001 patch and combines
it with the unchanged guarded-SDPA003 backend. Its source/build proof at
`d3814f3` rebuilds all 57 engine C++ units, retains 15 C objects and matches all 63
vendor members. It first passed a four-request two-step original-resolution
preflight (max channel difference1). Full 50-step generation then passed with
all compilers, package work and other inference idle:

| Generation | Accepted engine | Attention-only |
| --- | ---: | ---: |
| First request | 153.423 s | 151.022 s |
| Warm request | 135.547 s | 134.718 s |
| Warm first preview | 4.609 s | 6.551 s |
| Warm span after first preview | 125.090 s | 122.294 s |

The total warm gain is 0.829 s (0.61%). Sampling saves 2.796 s but first-preview
overhead adds 1.942 s in this one pair. No cause or distribution for that startup
difference is established. Generation mean channel differences 0.1448698/
0.1459796, maxima 32/57 and changed fractions 0.1327517/0.1285549 resemble the
identical-runtime inter-process controls. They do not establish pixel equality
or a validated tolerance. All previews, dimensions, PNG/privacy, PID reuse,
normal shutdown and zero new Xe events passed. There is no new full edit
comparison for this attention-only closure, and it remains uninstalled.

Source-only work on a broader, default-off whole-RoPE fusion continues in the
repeat producer worktree. It must preserve two separately rounded products
and ADD order, validate a bounded exact graph window, decline input-owner
overlap/intermediate escapes/capture, and prove produced-input lifetime with
actual allocator/segmented graphs. No speed or production acceptance is claimed
for that unfinished prototype.


### Accepted RoPE with guarded attention and residual fusion

A new closure at `d476407` (documentation `dde33b6`) restores the accepted
engine001 RoPE patch and combines the guarded attention backend with narrowed
residual-first fusion. It does not inherit the shared-RoPE engine experiment.
The normal source verifier checks this closure; no build-verification bypass,
test counters, profiler or linker wrappers are present in the worker. An
independent review verifies all 57 C++ units' 211 public dependencies, the GGML
headers and all six native archive hashes. All 63 vendor members match the
installed accepted bundle. Actual-engine CPU request cleanup passes with the
CPU-only registry, without initializing a GPU.

The isolated ABI 2 bundle is in the Comfy review worktree's
`target/accepted-rope-sdpa-residual-v1/runtime-flat`; the protected test copy is
`/usr/local/lib/kuno-test/qwen21-accepted-rope-fusion-v1`. The installed bundle,
configuration and gateway are unchanged. An eight-request two-step preflight
passes original-resolution generation/edit, previews, PNG validation, privacy,
PID reuse and normal shutdown with zero new Xe events. Generation differs by at
most one channel level in these shortened samples; editing remains variable.
The full 50/25-step compiler-idle pair subsequently passes all eight requests:

| Workload | Accepted first | Candidate first | Accepted warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| Generate, 50 steps/1184x1776/CFG1 | 153.707 s | 149.311 s | 137.345 s | 130.801 s |
| Edit, 25 steps/1248x832/CFG4 | 101.868 s | 97.779 s | 76.870 s | 73.187 s |

Warm reductions are 6.544 s (4.76%) and 3.683 s (4.79%) in this single pair.
The generation span after its first preview saves 4.641 s; first-preview timing
also differs by 1.905 s, without an established cause/distribution. Generation
first/warm mean channel differences are 0.1455907/0.1475765, maxima 59/44 and
changed fractions 0.1328674/0.1294907. Edit means are 1.4228366/2.0253584, maxima
206/250 and changed fractions 0.5066609/0.4954374. These resemble the measured
accepted inter-process generation and edit controls; they are not bitwise
pipeline equality or a validated tolerance. PNG/privacy/PID reuse, request
cleanup, normal idle shutdown and zero new Xe events pass. Cancellation of this
exact closure also passes: first original edit preview followed by SIGTERM
returns `image_cancelled` and normal status1 in 2.905 s, without forced stop.
A fresh worker completes the two-step original edit and normal idle shutdown;
privacy checks and zero new Xe events pass. The installed control passes the
same sequence (3.084 s cancellation). Recovery timings are not performance
evidence. Source/build receipts are recorded in the JSON appendix. The bundle stays unselected and uninstalled; no ComfyUI parity,
performance distribution or indefinite stability is claimed.


### Whole-RoPE exact synthetic execution and isolated timing

The default-off whole-RoPE source at `cc028e5` (receipt notes `a5032ef`)
recognizes a bounded exact seventeen-node subtree and seven-compute window,
then submits one strict F32 kernel at the terminal ADD. It preserves two
separately rounded products and ADD operand order. Input/PE full owners must
remain disjoint from the destination, and intermediate escapes/protected cuts,
unsupported layouts/devices/capture and oversized/overlapping plan sets retain
the original path. The inclusive 64-plan cap fits the observed 32-layer Qwen2.1
Q/K graph; the 65th eligible plan discards the entire set before any launch.
Host plan storage stays below 4 KiB and does not persist across calls.

System and strict Intel CPU validation pass 78 cases/1368 output comparisons,
134 policy graphs/4,227,553 checks, 64 real allocation graphs/2130 checks and nine
failure fixtures. The native test starts from the guarded-attention archive 0887,
with only the whole feature/test flags enabled; repeat/residual bypass flags
remain off so the reference executes all seven original operations. It replaces
one backend member, adds the strict kernel member and preserves 99 others.
Independent source/fatal-boundary reviews are linked in the JSON receipt.

Owner B70 execution passes 84 cases/1392 all-bit output comparisons in 11.100 s.
Wide finite exponents, public multi-axis PE, NaN payload/class, infinities,
signed zeros, subnormals, producer lifetime, changed PE and distinct live PE
pointers pass, including repeat execution. Routing counts prove 125 plans/launches,
32 produced-input cases and three wide cases. The 64-plan graph runs 64 launches;
65/128-plan graphs fall back completely. No fixed error, uncertain cleanup or
new Xe event occurs. Compiler flags alone are not device arithmetic proof;
Intel's strict-host floating-point support warning remains documented.

With all compilers/package/inference work idle, the same binary also passes
its 84/1392 checks and grouped-mode microbenchmark in 11.449 s. Each mode uses
three warm-ups and five trials of four evaluations:

| Public-PE graph | Legacy median/four | Whole median/four | Reduction |
| --- | ---: | ---: | ---: |
| L4096 | 17.952428 ms | 2.649538 ms | 85.24% |
| L8214 | 35.730032 ms | 5.270405 ms | 85.25% |

These intervals include host planning/submission and synchronization. Modes are
grouped rather than alternated; this is an isolated graph improvement, not a
whole-image estimate/distribution or ComfyUI comparison.

**Engine constraint:** this matcher requires the f6 shared, earlier prepared-PE
engine001/56fc. Ordinary accepted001/f900 creates PE inside the closed window
and falls back; no useful whole routing is claimed for that engine. A combined
f6/shared-RoPE + residual + whole source is prepared at `230d15b` (oracle
documentation `12d3687`). Its independently reviewed native test closure passes
86 cases/1400 comparisons in 13.087 s, including two actual in-place RMS/MUL
ownership cases: both select and launch the kernel. It records 127 launches,
34 produced-input cases and two complete capacity fallbacks. The merged
48-case residual arithmetic oracle also passes in 13.267 s. Both exit normally
with no fixed error, uncertain cleanup or new classified Xe event.

The 64-plan limit applies per graph computation; conditional/unconditional edit
sampling uses separate computations. Shared-PE construction does not guarantee
that its first monolithic window qualifies: depth-first scheduling can put PE
preparation inside that window. Later windows or bound segmented-prelude PE may
qualify, while ownership/escape guards still decide each actual allocation.
These checks establish bounded examples, not universal full-model routing.

### Whole-RoPE combined worker and full workflow comparison

The counter-free ABI2 worker at `230d15b` (build notes `0c646fe`, release
checklist `1c3c238`) uses shared-PE engine56fc, guarded attention, the exact
residual-first kernel and whole-RoPE. Independent source, dependency, archive,
worker and streamed-package reviews pass. The new archive replaces only the
backend member and adds the whole kernel, preserving the other 99 members,
including the residual kernels. All 63 vendor/license members are unchanged.
The normal Cargo source verifier passes; no TEST, profiler, repeat bypass,
counter or linker wrapper is compiled into the worker. It has no RPATH/RUNPATH.
Whole activation is an explicit ordinary CMake ON option; the source default
remains OFF. Public package pins and the managed build recipe are now selected
at `5351f63`, with the later builder verification guard recorded below.

The protected runtime is `/usr/local/lib/kuno-test/qwen21-whole-rope-residual-v1`.
An eight-request two-step preflight passes original-resolution generation/edit,
all previews/PNG/privacy checks, resident PID reuse and normal shutdown with no
new classified Xe event. A temporary main-thread-only perf entry counter proves
126 whole-kernel calls in each generation and 254 in each edit. It records no
arguments, samples or event file, and is closed after the diagnostic. Its CPU
fixture validates main-thread counts and excludes other threads; an earlier
inherited-counter fixture prevented thread creation, so inheritance is disabled.
These instrumented preflight durations are not performance evidence.

With all compilers, packaging and review hashing idle, a separate uninstrumented
eight-request full comparison passes against the existing f6/residual
`qwen21-fusion-order-v3` closure:

| Workload | Baseline first | Whole first | Baseline warm | Whole warm |
| --- | ---: | ---: | ---: | ---: |
| Generate, 50 steps/1184x1776/CFG1 | 149.031 s | 125.142 s | 132.893 s | 106.523 s |
| Edit, 25 steps/1248x832/CFG4 | 97.463 s | 86.391 s | 72.769 s | 61.831 s |

Warm reductions are 26.370 s (19.84%) and 10.938 s (15.03%) in this one paired
run. The generation span after its first preview falls from 120.003 to 96.713 s;
editing falls from 59.029 to 49.100 s. First-preview timing also varies, so the
entire gain is not attributed to the isolated kernel microbenchmark. All eight
requests validate every preview, final dimensions/PNG, cleanup, privacy, PID
reuse and normal shutdown with no new classified Xe event. The developer SDK is
inaccessible inside the unit; memory swap and core dumps are disabled.

Your historical ComfyUI warm timings are 100.65/64.48 s, versus 106.523/61.831 s
here. This substantially narrows the reported gap, but is not a fresh matched
ComfyUI benchmark or a performance distribution. Generation first/warm mean
channel differences are 0.1424282/0.1364353 (maxima 30/45), similar to the earlier
identical-runtime generation controls. Edit means are 3.1707658/1.6650222
(maxima 254/234); no bitwise pipeline equality or validated image-quality
tolerance is claimed. A separate four-request identical-f6 edit control passes
normal shutdown/privacy with no new Xe event: first/warm means are
1.1163759/0.9485032, maxima 230/202. This shows variation without whole-RoPE; it
does not fully explain the larger candidate difference or establish a quality
tolerance.

Cancellation after the first edit preview passes for both closures: baseline
2.859 s, whole 2.567 s, normal cancellation status with no forced stop. Fresh
two-step edit recovery passes in 54.176/52.971 s, followed by normal idle SHUTDOWN.
All four requests pass privacy checks with zero new classified Xe events.

The user authorized installation after those tests. The existing transactional
installer backed by a separately verified old-runtime copy installs the frozen
whole closure successfully. Installed runtime manifest is
`fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa`, worker
`4deb9a1bcd7108be8d22b56e8cc8958cb52458a56361c3b19c2b193012e679a3`.
The gateway binary remains `49115713...`; its configured GPU, Ollama listeners,
privileged control and disabled inference deadlines retain their roles.

Installed keyless HTTP acceptance passes original-resolution two-step generation
and editing with two previews each (31.022/39.403 s), the same retained worker PID,
final PNG/privacy checks and zero new classified Xe events. The client has its
SDK path hidden and zero swap/core; the worker uses the already accepted flat
closure and its installed zero-swap service. The earlier protected full tests
hide the SDK from the worker as well. Test inputs/results remain in RAM only.

Normal selected packaging produces archive
`85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3`
(156,247,147 bytes), identical to the tested archive without a pin override.
The release handoff distinguishes this frozen tested closure from a future
fresh oneAPI/CMake rebuild, which requires a reviewed MKL configuration and
renewed acceptance. There is no API or Vashti consumer change.

The original standard gateway/engine under `/tmp` is gone. A persistent v2
standard closure was subsequently found; its complete source/archive verification
are recorded in [the source/build receipt](kuno-image-speed-candidate-build.json)
under `2026_10_02_continued_testing`. Final CUDA workflow acceptance is recorded
separately; live Vulkan image acceptance is not claimed for this update. Generic
patch001 also affects CUDA/Vulkan, so source equality alone is insufficient.


### Selected standard build and final backend release checks

The persistent standard v2 source/cache/native closure is reverified: 57 C++
objects, 15 unchanged C members, five GGML archives, four CUDA static producers,
211 native dependencies and complete pinned source diffs. A fresh normal
native-image Cargo release build, all-target check and all-target Clippy with
warnings denied pass. The new standard gateway is byte-identical to v2,
`9865c0d4f075b3f7eab05c4907f9da0c4c68d49ab06f539c4f05b0814061b557`,
without private Intel dependencies or RPATH. Verified native archives were reused;
this is not a fresh full CMake/GGML compilation.

The current safer RoPE oracle and failure fixture compile against that closure.
CPU arithmetic/topology and failure/cleanup checks pass. The GPU oracle passes
40 exact legacy/prepared comparisons and input-immutability cases each on
CUDA and Vulkan/3060 (5.021/6.635 s).
Four original-resolution two-step CUDA generation/edit requests pass every
preview, PNG/dimensions, per-request privacy, same resident PID and normal idle
SHUTDOWN with no new classified Xe event (36.700/35.021 s generation,
52.498/50.169 s edit). These short durations are functional receipts, not a
full-step NVIDIA performance comparison or ComfyUI parity claim.

The first CUDA workflow attempt stopped at an overbroad test assertion against
a writable regular handle. A synthetic-only follow-up identifies an unlinked
anonymous memfd on tmpfs, not disk storage. Four CPU classification fixtures pass;
the protected harness permits only that in-memory class with zero swap/core and
still rejects writable disk-backed regular handles. Production code is unchanged.
The corrected four-request test passes. No private inputs/results are persisted.

Owner source integration is `05a95fc` (public selection `5351f63` and strict
builder guard `bde8cd0`). Root source checks pass 205 tests, nine ignored,
Clippy with warnings denied, formatting and whitespace checks. Packaging eight,
installer four and managed-builder six CPU checks pass. The normal root packager
reproduces the tested 156,247,147-byte archive `85d7aecb...` without overrides.
The final numeric/source evidence is in `selected_release` and
`2026_10_02_continued_testing` in
[the build receipt](kuno-image-speed-candidate-build.json). Live Vulkan image
workflows, a new full CUDA timing pair, output-quality tolerance and indefinite
stability remain outside this beta acceptance. Application/versioned release
packaging and publication belong to the Vashti release task.

Final installed-service audit: active, no automatic restarts, zero swap, admission
unpaused, zero active/waiting requests, no recovery required, all runtime members
verified, original gateway checksum/installation identity preserved, and both
Ollama inference deadlines remain zero. All benchmark workers are stopped and the
benchmark unit is inactive. Classified Xe fault and NVIDIA Xid counts are zero
for this boot. Both GPUs are released. No new prompts/images/previews, raw errors
or content digests are saved by the test clients or Kuno.

## Claude speed candidate (2026-11-05/06)

Branch `claude/image-speed`; dev source/build tree `/media/jmicko/faststore/dev/kuno-image-dev`
(source = whole-RoPE closure + these changes, `final-build.sh` builds the AOT worker).
Candidate runtime (not installed): `/usr/local/lib/kuno-test/qwen21-claude-speed-v1`,
worker `d4f1521ada834f37dd41cfb76ba507cf679701f7de335b292efeef6e6be11f37`.

Changes (all in patch 003 unless noted):

- ConvRot quantizer: row-parallel sub-group kernel (register radix-4 stages + padded SLM
  transpose, original stage order/expressions). The IMF `fmul_rn` software multiply is
  replaced by the IEEE multiply except for rows whose inverse scale is >= 1e30. ~4x faster;
  bytes and scales identical to the legacy kernel (incl. zero/subnormal/inf/NaN rows).
- Fusions (bit-identical to the unfused ops, verified on 1184x1776, 1024x1024 CFG4,
  640x896 and 1024x768 edit with byte-compare replays): gate_up GEMM + SwiGLU + quantize;
  MUL(row)/SCALE + quantize; out/down GEMM epilogue + SCALE + gate MUL + residual ADD;
  q/k GEMM + per-head RMS norm + whole RoPE (K written as F16 into the prefix concat);
  V CPY+CONCAT+PERMUTE+CONT gather; oneDNN attention F16 output straight into the
  out-projection quantizer and an F16 Q copy from the q kernel. All fusions check for
  allocator buffer reuse between fused inputs and outputs and fall back otherwise.
- Same-shape float copies without 64-bit index division; tiled transpose copies; faster
  attention Q/O conversions.
- Wan VAE on SYCL uses CONV_2D (patch 001 `model_builders.cpp`) implemented with a oneDNN
  convolution (fallback: banded im2col + the same oneDNN GEMM). The full im2col matrix
  (VAE compute buffer 19.6 GB) is gone (9.2 GB), so the text encoder is no longer evicted
  and re-staged every request. Not bit-identical: max 1 level in ~0.3% of channels vs the
  im2col path.
- Weight uploads through two pinned 64 MiB chunks (~5 GB/s instead of ~1.5 GB/s).
- AOT device code for bmg-g31 (`GGML_SYCL_DEVICE_ARCH`, `sycl-runtime-linker.sh` with
  `KUNO_SYCL_AOT_DEVICE`/`KUNO_OCLOC_DIR` at build time; managed builder `--ocloc`).
  ocloc 26.35 was extracted locally (not installed) and works with the installed IGC.
  Removes ~15 s of first-request JIT.

Protected harness (`full-resident-bench.py` copy that checks new Xe events instead of
requiring zero since boot; two pre-existing resets were from a dev bug before this run),
original recipes, installed baseline vs candidate, separate processes:

| Workload | ComfyUI (user) | Installed | Candidate |
| --- | ---: | ---: | ---: |
| Generate 50 steps first | 132.06 s | 124.156 s | 67.132 s |
| Generate warm | 100.65 s | 106.726 s | 61.667 s |
| Edit 25 steps first | 78.25 s | 84.624 s | 40.922 s |
| Edit warm | 64.48 s | 61.963 s | 37.686 s |

Pixel differences vs installed: generate mean 0.143/0.148 (max 28/72), edit mean
1.29/1.41 — the same envelope as the earlier identical-runtime controls (0.14; edit 1.1-1.8).
All previews/PNG/privacy/residency checks passed, normal idle shutdown, zero new kernel
events. Cancellation after first preview: 1.85 s generate / 2.05 s edit, no forced stop,
recovery and shutdown normal. Not installed; the installed runtime is unchanged.
