# Qwen 2.1 performance follow-up

This records the remaining performance work after the seam and dimension fixes.
The installed engine is still slower than the supplied ComfyUI workflows. The
speed goal is not complete. The integration checkpoint describes the current
implementation; the later measurements also retain earlier experimental history.

## Native optimization candidate (2026-09-30)

Work starts from installed backend `f6f6c22` in a separate Kuno worktree;
the concurrent Vashti application changes are preserved. The validated standard candidate through `eee97c8` was installed on this host;
later SYCL development changes remain separate. Full Intel edit execution and synthetic GPU numerical checks
have passed; performance is still below the supplied ComfyUI baseline. The clean
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
per subsequent step. Xe peak was24,754,114,560 bytes, cgroup RAM peak1,930,915,840
bytes and swap0. This was a clean run without native compilation. The equivalent
accepted CUDA edit is160.702 s; user ComfyUI Intel cold/warm baselines are78.25 /
64.48 s. SYCL is substantially faster than Kuno's earlier paths but does not yet
match the user's ComfyUI timing.

The original 1184x1776, 50-step, CFG1 generation completed sampling at136.939 s
but exited-6 at140.988 s during decode, with the fixed
`UR_RESULT_ERROR_DEVICE_LOST` classification. Xe peak was22,143,942,656 bytes,
cgroup RAM peak1,328,771,072 bytes, swap0. One new Xe engine-reset entry at
02:52:56.688688 UTC raised the boot count from6 to7. This reset matches the
large-generation test window; the earlier two entries remain separate. No
production configuration or worker was changed. Admission was resumed afterward.

Independent IM2COL modes5-8 pass on the B70, including the exact large geometry,
10,900,832,256-byte output, samples across2/4/8GiB and INT32/UINT32 element
boundaries, and bounded-launch chunk boundaries. All runs used Landlock and
zero-swap cgroups; no further Xe reset appeared. Exact-case C144/C288 modes
passed249/255 samples; the UINT32-bracketing cases passed252/255 samples.
Embedded worker SPIR-V also retains64-bit offsets/access-chain indices. These
results do not reproduce other asynchronous VAE operations.

The actual CPU-only VAE graph has seven non-contiguous F32 CONT operations
crossing signed-int byte offsets in the larger workflow, versus zero in the
passing edit. The legacy SYCL copy path narrows byte strides/offsets toint; a
separate focused fix and independent copy-boundary tests are in progress. The
optional managed runtime therefore remains uninstalled and unaccepted. The
standard installed CUDA/Vulkan service remains the user-facing image engine.
