# Experimental Qwen image execution

Kuno now has a bounded ComfyUI PNG importer, a curated model downloader, and an
native image worker embedded in the distributed executable. Vashti includes
an Images workspace and generation/editing chat tools, with live previews.
GPU configuration and curated model downloads currently use the CLI; an admin
collection picker remains future work. This is an experimental beta.

## Attention and cache execution

Requests may set `"attention": "fast"` for the Qwen 2.1 CUDA backend. The default
is `"standard"`. Fast attention uses the native SageAttention implementation;
it can change output pixels and is not supported for Qwen 2512 or Vulkan.
Unsupported choices return `image_attention_unsupported`. The embedded
CUDA/Vulkan path needs no Python engine or additional runtime installation.

The [optional managed SYCL worker](kuno-managed-sycl.md) adds Qwen 2.1 execution
on Intel hardware, accepted locally on the B70. Its libraries are
separate from the standard executable; embedded CUDA/Vulkan remains the default,
and Qwen 2512 retains its native route. This runtime does not change the image API.

Large Qwen 2.1 edit caches can use bounded pinned host memory. On CUDA, capable
workers stage the current and next layer through two GPU buffers, overlapping
transfers with computation. Staging counts toward VRAM reservations, and pending
transfers are drained before freeing buffers or leaving a request. If the
required memory or backend capabilities are unavailable, execution retains the
synchronous cache path. Prompts, references, and results still remain transient.

## Exact supported recipe

`qwen-image-2512-text-to-image/v1` uses the Qwen Image 2512 FP8 diffusion model,
Qwen 2.5 VL 7B scaled FP8 text encoder, and Qwen Image VAE. It supports Euler with
the simple scheduler, full denoise, batch size one, and an explicit flow shift.
The provided PNG imports as 832×1248, 100 steps, CFG 5, shift 3.1, and seed
`684489560344628`. Both direct text and linked primitive-string prompts work.
Only active execution-graph nodes affect the import; disconnected LoRA notes
are ignored. An active unsupported LoRA or sampler fails explicitly.

PNG metadata does not establish the original model hashes. Matching settings
also does not imply identical pixels across ComfyUI and the native engine.
Seeds are decimal strings in the API to preserve all 64 bits in JavaScript.
The current native adapter accepts only nonnegative signed-64-bit seeds; larger
imported seeds can be viewed/exported but generation rejects them explicitly.

## Qwen Image 2.1

The additional recipes are `qwen-image-2.1-text-to-image/v1` and
`qwen-image-2.1-edit/v1`. They use the INT8 ConvRot diffusion model and Qwen3-VL
8B encoder, plus the dedicated BF16 2.1 VAE. The collection is
`qwen-image-2.1-int8` (17.28 GB decimal, Qwen Research license). The 2512
collection remains available independently.

The importer follows the active TextEncodeQwenImage21, QwenImage21Cache
(auto/default), resolution selector, and ComfySwitchNode branch. Imported seeds,
steps, CFG, prompts, and dimensions are preserved. The native 2.1 schedule is
automatically resolution-dependent: omit `flow_shift`. Output dimensions use the
model's 16-pixel VAE grid without additional rounding to 32. Output is 8-bit
RGB/RGBA PNG, including alpha; a ComfyUI 16-bit
save setting is reported as an output-format difference.

Editing accepts `reference_images`, an ordered list of one to ten base64-encoded
PNG, JPEG, or WebP images. Filesystem paths and remote URLs are not accepted.
Original references are not embedded in a ComfyUI output PNG: the importer reports
how many must be supplied again. Inputs have bounded dimensions and decoded
pixel counts; animated images are rejected. JPEG orientation is applied. Each
reference is resized with Lanczos to approximately 1024 squared pixels, preserving
aspect ratio and rounding dimensions to multiples of 32, matching the supplied
ComfyUI encoder's resolution rule. Reference order and alpha are retained.

Both Vulkan and CUDA store Qwen 2.1 prefix K/V in FP16, about 512 KiB per prefix
token per condition, matching the precision supplied to flash attention. Cache
precision is separate from the working-memory reserve: Vulkan conservatively
reserves 9 GiB for full weights and workspace, while CUDA keeps its segment
reserve and bounded host-cache policy. Kuno estimates the cache from reference
dimensions and a conservative text-token bound. The CUDA policy reserves working
space for a segment and allows the engine to stream weights; Vulkan budgets full
diffusion weights. It disables caching when the estimate exceeds the available
GPU budget. These estimates are not guarantees against other GPU memory pressure.

The request body limit is 64 MiB, with at most two image bodies admitted into
memory at once. Each encoded reference is limited to 16 MiB and the combined
reference payload to 63 MiB. Native execution remains one image at a time.

## Build and models

On Linux Mint/Ubuntu, development builds need:

```sh
sudo apt install build-essential cmake ninja-build glslc libvulkan-dev spirv-headers glslang-tools
./apps/kuno/build-native-image.sh
./target/release/kuno fetch-models qwen-image-2512-fp8 --directory /absolute/model/store
./target/release/kuno fetch-models qwen-image-2.1-int8 --directory /absolute/model/store
```

Rust is also required for building. Qwen 2.1 INT8 also requires a shader compiler
that supports `GL_EXT_integer_dot_product`; Mint's shaderc 2023.8 does not. The
build probes this feature and fails with setup instructions rather than silently
building an engine that sends those operations to the CPU. A verified local build
uses LunarG SDK 1.4.357.1 (archive SHA-256
`4b41e3b30e8aedaa5dac7c136561ab463eb316a25a54e2c6245f2c299ea1fb85`). Set
`VULKAN_SDK=/absolute/sdk/1.4.357.1/x86_64` when running the build script; only the
compiler and headers are used, without replacing installed drivers. Get the SDK
from [LunarG](https://vulkan.lunarg.com/sdk/home).

An optional NVIDIA build embeds the CUDA runtime and cuBLAS static libraries,
alongside Vulkan. It prefers CUDA when both backends expose the selected physical
GPU. Other GPUs retain the Vulkan path. This produces a substantially larger
single executable and requires the NVIDIA driver (including `libcuda.so.1`) on
the inference host. End users do not need the CUDA development toolkit.

```sh
KUNO_IMAGE_CUDA=1 KUNO_CUDA_ROOT=/absolute/cuda/toolkit \
  KUNO_CUDA_ARCHITECTURES=86 VULKAN_SDK=/absolute/vulkan/sdk/x86_64 \
  ./apps/kuno/build-native-image.sh
```

Architecture 86 targets the RTX 3060; use the architectures required by the
intended host, or omit the setting for a native build. CUDA toolkit 13.0 was
tested. Obtain development components from [NVIDIA's official redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json)
and verify their manifest checksums. The toolkits are build inputs, not driver
upgrades. Both CUDA and Vulkan shader disk caches are disabled in image workers.

Installed binaries need the GPU's working
Vulkan driver and the usual native system libraries; they do not require Python
or a separate image server. The build pins stable-diffusion.cpp to
`88411ef1e0688ff2df1010aeeb5d92b2d8cea2be`, retaining its matching patched GGML.
A checked-in compatibility patch fixes quantization-scale tensor name detection
and adds bounded INT8 embedding row lookup with inverse ConvRot rotation. It
also preserves input scaling while packing INT8 attention projections for the
GPU, avoiding the upstream CPU fallback for Qwen's 1/32 attention scale. The
build checks the exact patch and runs an independent synthetic numerical test;
it refuses unrelated local dependency changes. A runtime kernel-capability
probe rejects unsupported INT8 GPUs before model loading with
`image_gpu_int8_unsupported`; check the build toolchain and Vulkan driver in that
case rather than retrying model downloads.
The build is optional; ordinary Kuno builds do not link the image engine.

The Qwen 2512 collection pins a Hugging Face repository revision, exact sizes,
and SHA-256 hashes for all three artifacts (about 30.1 GB decimal). Downloads
resume from private partial files and activate only after checksum verification.
Completed public model artifacts are read-only-accessible to the service through
its scoped model mount. Existing mismatching files are preserved and rejected.
This downloader never accepts model URLs or filesystem paths from PNG metadata.

## Configure the installed service

First enable the managed Ollama workers as described in the Kuno README. Create
a settings file containing only machine configuration:

```json
{
  "models": "/absolute/model/store",
  "pci_address": "0000:2d:00.0",
  "timeout_seconds": 3600,
  "collections": ["qwen-image-2512-fp8", "qwen-image-2.1-int8"]
}
```

Then run the installed binary:

```sh
sudo /usr/local/lib/kuno/kuno configure-image --settings /absolute/settings.json
```

This asks for interruption confirmation, verifies all model checksums, and
replaces the binary, configuration, and read-only systemd mount transactionally.
The model files must be readable by the service. To disable images while retaining
Ollama management, use `configure-image --disable`. Upgrades must retain the
`native-image` feature while images are configured. The release package builder includes the native engine and its license notices.

## API and lifecycle

`POST /kuno/v1/images/generate` is available on the public compatibility listener
for keyless trusted-LAN inference. Privileged control operations remain separate.
See [discovery and response streaming](#inference-access-discovery-and-streaming-2026-09-30). It accepts:

```json
{
  "workflow": "qwen-image-2512-text-to-image/v1",
  "prompt": "A red wooden boat on a quiet lake.",
  "negative_prompt": "Blurry, low quality.",
  "seed": "42",
  "width": 512,
  "height": 512,
  "steps": 20,
  "cfg": 5.0,
  "flow_shift": 3.1
}
```

The successful response is a single `application/x-ndjson` stream with
`Cache-Control: no-store`: acceptance, previews, then a terminal result or error.
The result contains a base64 PNG with no embedded generation metadata. Vashti
owns any eventual output/provenance storage.
Model downloads and operational state are separate from inference content.

The first scheduler implementation holds the same exclusive admission lease as
LLM inference, stops managed Ollama workers on the image GPU, and releases canonical Ollama's
resident models. Verified managed workers on other GPUs remain resident. The image runs in a short-lived child process selected by the
physical GPU PCI address. Following LLM requests can start workers again. Child
cancellation, output limits, and deadlines retain admission through process
cleanup; uncertain cleanup requires recovery instead of admitting competing work.
The global admission lease still serializes useful LLM/image execution; retaining
a model on the other GPU does not yet imply simultaneous inference.

The worker reads its recipe through stdin and returns PNG bytes through stdout.
It enters its restrictions before any async runtime threads are created.
It refuses to read content unless swap is disabled for its cgroup (or the host).
Core dumps are disabled. Linux Landlock ABI 3 or newer denies filesystem content
writes, file creation, removal, and truncation. Exceptions allow existing GPU device
nodes and writes/truncation to this process's ephemeral `/proc/self/task` controls:
CUDA names its internal threads through their `comm` entries. This grants no file
creation rights; a regression verifies that proc fd aliases cannot bypass disk
write denial. All native threads inherit the policy. Shader disk caches are disabled.
Backend logs are reduced to fixed error
codes and allowlisted engine source locations; prompts and raw backend messages
are never logged.

Systemd configures `MemorySwapMax=0` for the installed service. Standalone worker
development must use equivalent isolation, such as a transient systemd service
with swap disabled and an appropriate RAM limit. Do not run unconstrained GPU
tests alongside unrelated GPU work. The `max_vram` native budget leaves driver
headroom but is not a hard device-wide allocation limit.

## Verification

Automated checks cover active-graph import, duplicate JSON keys, PNG checksums,
compressed metadata bounds, large seeds, worker cancellation/deadlines/output
bounds, inference/control routing separation, and filesystem write denial. Committed tests use
synthetic metadata; the local reference image remains outside the build inputs.

On the target Arc Pro B70, a standalone 512×512, four-step synthetic generation
completed in about 37 seconds, returned a visually coherent image, used no swap,
and released its GPU allocations. This is a smoke result, not a quality or speed
comparison with ComfyUI. Full reference execution and cross-engine fidelity must
be assessed separately.

The RTX 3060 completed the same standalone smoke workload in about 105 seconds.
The installed service subsequently passed LLM → native image → LLM handoff,
client disconnect cleanup (about 4.2 seconds), and another successful LLM request
after cancellation. A separate 100-step request was disconnected after 90 seconds
and Kuno was restarted during cleanup: graceful stop/start took about 18 seconds,
left no recovery marker or image worker, and subsequent LLM inference succeeded.
Final installed-binary verification also passed model listing, generation, and
unload from Clark's machine over the original LAN address. Kuno was left active,
unpaused, with no resident models or recovery requirement.
These are Vulkan results on both cards, not a comparison
against an optimized CUDA image runtime.

Later B70 image runs produced kernel `xe` job-timeout/engine-reset messages and
Vulkan compute failures. The larger 832×1248, 100-step reference run reached
sampling but was intentionally stopped before completion. Subsequent failures
also reproduced outside the service; changing locked-memory limits and using
serialized, smaller Vulkan submissions did not resolve them in that session.
Cancellation now requests native cancellation and allows up to 90 seconds for
GPU work to finish before a last-resort kill. A forced stop or detected Vulkan
device loss requires recovery rather than silently releasing admission.

The B70 is also open in Xorg and the desktop app. No device unbind/reset, driver
replacement, or reboot was performed. A fresh-boot B70 retest is needed before
claiming reliability or attributing the failure to one specific cause. Similar
symptoms appear in an [upstream B70/xe report](https://github.com/ggml-org/llama.cpp/issues/24810);
that is a diagnostic lead, not proof that this host has the same bug. The current
installed experimental image configuration uses the RTX 3060.

### Qwen 2.1 bring-up

The 2.1 collection has been downloaded and checksum-verified. Automated tests
cover both supplied workflow shapes, ordered references, JPEG/WebP decoding,
reference resizing, and INT8 embedding scale/rotation math. A 512×512, four-step generation completed on the RTX 3060 in about 35 seconds.
The installed authenticated API repeated that workload in 36.15 seconds while
`ornith-1.5:35b` stayed fully resident on the B70 (21,360,363,436 bytes reported by
Ollama, context 4096), and the LLM answered again afterward. The output was a
visually recognizable boat. These are correctness smoke tests, not a quality or
cross-engine performance comparison.

The supplied original JPEG also completed the imported 1248×832, 25-step,
CFG 4 edit through the installed API in 794.27 seconds on the RTX 3060. The
output lengthened the animal and preserved the scene, but had a visible central
seam absent from the supplied ComfyUI output. This establishes execution, not
quality parity. NVIDIA allocation/mapping warnings occurred during that run;
the subsequent build adds the conservative prefix-cache budget described above.
That build completed a two-reference request (one 16-bit PNG and one 8-bit PNG,
512×512 output, four steps, CFG 4) in 185.85 seconds with no new kernel messages.
The output was visually poor at four steps; this verifies reference handling and
bounded execution, not useful multi-image editing quality. Both runs returned
metadata-free PNGs with `Cache-Control: no-store`; preview images lived only in RAM.

The supplied PNGs contain workflow and generation settings, but no recorded
generation duration. The user measured these ComfyUI baselines on this host:

| GPU | Cabbage cold | Cabbage warm | Weasel cold | Weasel warm |
| --- | ---: | ---: | ---: | ---: |
| Arc Pro B70 | 132.06 s | 100.65 s | 78.25 s | 64.48 s |
| RTX 3060 | 199.94 s | 193.41 s | 126.14 s | 109.51 s |

Cold runs used the original reference seeds. Warm runs incremented each seed by
one to force sampling while allowing ComfyUI to reuse preparation. The supplied
Intel warm PNGs confirm the seeds and unchanged sampler settings. The user
reports visually identical NVIDIA outputs; those files were not supplied for
comparison. These timings are user-reported, not extracted from image metadata.

Kuno's original Vulkan 794.27-second weasel run was about 6.3 times the ComfyUI
cold time on the same RTX 3060. The subsequent CUDA build with FP16 prefix caching
completed the same original-seed, 1248×832, 25-step, CFG 4 workflow in **283.19 s**
through the restricted Rust image worker. This is about 2.8 times faster than the
Vulkan result, but still 2.2 times ComfyUI's cold time. The visible central seam
remains, so this does not establish quality parity. The returned RGBA PNG had no
metadata and lived only in RAM. No new kernel messages occurred during the run.

Initial CUDA validation found that denying writes to thread `comm` entries caused
CUDA discovery to fail and silently left Vulkan available. The task-control
exception above fixes this without permitting persistent file writes; the final
benchmark explicitly verified CUDA selection. A separate 512×512 four-step
synthetic edit produced byte-identical pixels with FP32 versus FP16 prefix caches.
That narrow numerical check does not establish equivalence for every workflow.

The CUDA/Vulkan build is installed on this host. Installed API smoke tests at
512×512 and four steps completed in 12.49 s for Qwen 2.1 (CFG 1) and 78.87 s for
Qwen 2512 (CFG 5, shift 3.1). A 35B LLM remained fully resident on the B70 during
both 3060 image requests. Disconnecting a subsequent 100-step Qwen 2.1 request
after 20 seconds released admission in 1.0 s without forced termination or
recovery, and the LLM answered again afterward. These small runs verify service
integration, not image quality. Responses were metadata-free PNGs with
`Cache-Control: no-store`; no image bytes were saved. The service was left active
and unpaused with no loaded models, recovery requirement, or new kernel messages.
Clark's machine also reached the public API over the original LAN address.

Validation: 118 Rust tests passed (8 hardware tests ignored), CUDA-enabled Clippy
passed with warnings denied, and native numerical/backend-selection tests passed.
Full ComfyUI performance parity remains unresolved.

### Seam and dimension correction

A VAE-only round trip reproduced the central seam without any diffusion sampling.
The decoder's allocation fallback selected half-image tiles, which met without
overlap at this resolution. Qwen 2.1 now retries with bounded 32-latent-pixel
tiles and overlapping context. The complete original-seed weasel edit subsequently
finished in 286.25 s with no visible central seam (alpha range 247–255, versus
1–255 before the fix). Images were inspected through a temporary RAM-only preview.
Compilation ran during part of this test; its timing is a correctness result,
not a clean new performance baseline.

The native engine also incorrectly imposed an extra factor of two on Qwen 2.1's
16-pixel VAE grid, although its diffusion projection uses one latent per token.
The patch removes that factor, with a native request regression preserving
1184×1776 exactly. The importer and execution validator now accept this size.
Reference-image preprocessing retains its separate 32-pixel grid, matching
[ComfyUI's Qwen 2.1 encoder](https://github.com/Comfy-Org/ComfyUI/blob/9d3bfc393b3415e146efb8434b32784cc2e69521/comfy_extras/nodes_qwen.py).

The complete original-seed, 50-step, CFG 1 cabbage recipe returned exactly
1184×1776 in **354.03 s** on the 3060. The restricted Rust worker returned a
metadata-free RGBA PNG (alpha 248–255), inspected in RAM with no visible seam.
No compilation or other inference ran alongside it. This remains about 1.77
times the user-reported ComfyUI cold time of 199.94 s on the same card.
During a 20-second sampling interval, PCIe receive traffic was only 23–53 MB/s;
link saturation does not explain that interval. The earlier four-step run took
50.08 s and served only as a dimension/execution check.

An experimental FP16 prefix cache in system RAM produced byte-identical output
to the GPU cache in a 512×512, four-step synthetic edit. At 1248×832, four steps,
CFG 4, the RAM-cache probe took 63.86 s, with later steps around 9.1 s. This
option now has an admission policy that checks MemAvailable and cgroup
ancestor memory.max/memory.high headroom, retaining 24 GiB for models and
workspace. Unknown accounting disables the RAM option. The full-workflow
result with this policy is recorded below. Moving this cache alone does not close
the ComfyUI speed gap.

Initial profiling used synthetic references and discarded output pixels. On the
3060, a two-step, CFG 4, 1248×832 edit spent about 33 seconds preparing inputs,
68 seconds sampling, and 8 seconds decoding. Per-operation instrumentation
attributed 43 seconds to matrix multiplication and 6 seconds to attention;
instrumentation synchronizes operations and adds overhead. An uninstrumented
four-step B70 run with the prefix cache enabled took 164 seconds, with later
steps around 27 seconds. Neither is the supplied 25-step benchmark.

A separate 4096×4096 INT8 matrix test took about 13–14 ms through Vulkan on the
3060, versus 1.6–1.9 ms through CUDA cuBLAS (the latter excludes activation
packing). This motivates a full CUDA comparison, not an end-to-end speed claim.

### CUDA profiling after correctness fixes

A two-step synthetic 1248×832 edit with CFG 4 and an experimental RAM prefix
cache attributed 6.47 s to flash attention, 5.64 s to matrix multiplication,
and 2.58 s to contiguous tensor copies during sampling. This instrumentation
synchronizes each operation; these figures are for locating bottlenecks, not
an end-to-end benchmark. Attention at 4,056 query tokens, about 8,130 key tokens,
32 heads and head dimension 128 took about 40 ms per call. Most diffusion weights
remained resident for the second step, so repeated full-model transfers alone
cannot account for the steady sampling cost.

The CUDA FP16 cache now joins current and cached K/V directly in FP16 before
flash attention, eliminating an FP16-to-FP32-to-FP16 round trip. Separate
512×512 four-step synthetic edit comparisons returned byte-identical pixels
before and after this change, with both GPU and experimental RAM caches.
The one-megapixel RAM-cache probe changed from 63.86 s to 63.29 s overall;
this is a small result and does not establish ComfyUI performance parity.

An isolated, uninstalled attention prototype tested larger query tiles using
the existing kernel implementation. A 128-query tile took around 34 ms versus
38–40 ms for the default 64-query tile on a constant-input microbenchmark.
Smaller tiles were slower. This prototype has not passed a general numerical
comparison and does not modify the pinned ggml checkout or installed engine.

The seam/dimension fixes and redundant-conversion removal are installed on this
host. Validation passed: 118 Rust tests (8 hardware tests ignored), native
numerical/dimension tests, CUDA-enabled Clippy with warnings denied, formatting,
and diff checks. Installed API checks took 12.43 s for Qwen 2.1 and 78.32 s for
Qwen 2512 at 512×512/four steps; the 35B LLM stayed resident on the B70 during
both 3060 requests. Cancellation cleaned up in 1.0 s, the LLM answered afterward,
and all models were unloaded. Kuno was active and unpaused, with no recovery
requirement or new kernel messages. The full-resolution speed gap is unresolved.

Further steady-state profiling reset counters after the second of four synthetic
edit steps. The remaining two steps took 16.98 s: attention accounted for 5.09 s,
matrix multiplication 3.69 s, contiguous copies 1.43 s, multiplication 0.92 s,
and other measured operations about 2.2 s. The remaining time includes graph
execution and data movement; it has not been attributed to PCIe specifically.
The same physical link is used by ComfyUI, so link width alone cannot establish
the cause of the performance difference.

A separate SageAttention experiment reduced the isolated attention operation
from about 40 ms to 16 ms and a four-step synthetic edit from 63.29 s to 56.86 s.
Synthetic oracle comparisons passed the probe thresholds, but quantized
attention changes numerical results (including 5.5% relative L2 error in a
small high-magnitude case). It remains a diagnostic-only experiment pending
full-workflow quality checks. Neither its timing nor the larger attention tile
prototype is a claim of end-to-end parity.

The Qwen 2.1 fused gate/up projection now uses the backend-supported SwiGLU
operation, avoiding two intermediate chunk copies and separate activation and
multiply operations. Unsupported backends retain the original operations.
512×512/four-step synthetic edits were byte-identical before and after this
change with both GPU and RAM FP16 caches.

With fused SwiGLU and the bounded RAM-cache policy, the original-seed 25-step,
CFG 4 weasel edit completed in **241.59 s** on the 3060, returning exactly
1248×832 RGBA with no metadata, alpha 247–255 and no visible central seam.
The restricted worker confirmed RAM-cache selection. No compilation or other
inference ran during this test. This improves on the earlier roughly 283–286 s
runs, but remains 1.92 times the user-reported 126.14 s ComfyUI cold result.
The combined test does not isolate each optimization's contribution.
Validation passed 121 Rust tests (8 hardware tests ignored), native regression
tests, CUDA-enabled Clippy with warnings denied, formatting and diff checks.

This candidate is installed (binary SHA256
`741256cd2efb378b1b5674b7bfc7b958c9aba41abe153223296ce4b187a66db3`).
Installed 512×512/four-step API checks passed for Qwen 2.1 (14.13 s) and
Qwen 2512 (79.03 s), retaining a fully GPU-resident 35B LLM on the B70.
Cancellation released the image worker in 0.8 s, the LLM answered afterward,
and all models were unloaded. The service is active with admission resumed.
SageAttention and custom attention tile prototypes remain uninstalled.

See [the performance follow-up](kuno-image-performance.md) for the subsequent
187.84-second experimental SageAttention result, the demonstrated cache-transfer
scheduling difference, and the next exact optimizations. These newer experiments
are not part of the installed binary described above.

### Installed integration checkpoint

The installed build now includes the redundant modulation-copy removal, bounded
pinned-cache transfer pipeline, and opt-in `attention: "fast"` API setting.
SHA256: `b3d073704c093ea88875c070acc6660c54f14237d44746e95ddbfcaa239e6699`.
The original weasel edit took 169.25 s with fast attention and confirmed active
cache pipelining; the original cabbage took 262.89 s. Both preserved dimensions
and passed RAM-only visual seam checks. See the performance follow-up for test
conditions and the remaining ComfyUI gap. Standard attention remains the default.

Validation passed 122 Rust tests (8 hardware tests ignored), native regressions,
CUDA-enabled Clippy, formatting, and diff checks. Multiple-reference fast
generation passed in 62.59 s. Cancellation after the standard-attention cache
pipeline became active exited in 4.02 s without a forced stop. Further kernel
optimizations are deferred. The next work is app integration, pending the user's
UI/design direction; this checkpoint does not claim public release readiness.

Installed API checks passed: Qwen 2.1 fast attention (512×512/four steps,
11.80 s), Qwen 2512 standard attention (80.14 s), and cancellation cleanup
(1.0 s). The 35B LLM stayed fully resident on the B70 during the image requests,
answered after cancellation, and was then unloaded. The service is active and
unpaused with no recovery requirement or new kernel errors.


### Transient previews (historical beta implementation)

The following records the earlier polling implementation and smoke results.
It is superseded by the single response stream specified below; the new API
does not accept preview identifiers or expose polling routes.

The PNG generation endpoint accepts optional `x-kuno-image-preview-id`: 64 random
hex characters supplied by the caller. While that request is
active, keyless `GET /kuno/v1/images/previews/{id}` on the compatibility listener
returns null or the
latest `{step, steps, width, height, png_base64}` projection preview, with
`Cache-Control: no-store`. The identifier is per request and must never be put
in persistent job recipes. The entry is removed after worker cleanup, including
cancellation and failure. There is no listing or replay endpoint.

An opt-in worker environment flag enables bounded `KIP1` stdout pipe frames:
magic, one-byte kind, big-endian u32 payload length, payload. Kind 1 contains
step/steps/width/height as four u32 values followed by a metadata-free PNG;
kind 2 contains the final PNG. Preview dimensions are at most 512 square,
payload at most 1 MiB, final at most 20 MiB. Diagnostics remain on stderr and
never contain image frames. Existing callers without previews still receive
the original PNG protocol. The engine uses `PREVIEW_PROJ`, not a full VAE decode
per sampling step. Native compilation and synthetic transport/UI checks pass. A live RTX 3060
Qwen 2.1 smoke test (512 square, two steps, CFG 1, standard attention) emitted
one 32-square RGBA preview per step and the final PNG in 11.05 seconds. The test
consumed all image bytes in RAM and saved no images. This is a transport smoke
test, not a quality or comparable full-workflow performance benchmark; preview
overhead on the normal 50-step workflow and Intel still needs measurement.

## Inference access, discovery, and streaming (2026-09-30)

**Implemented and source-tested Kuno contract; Vashti integration and synthetic
application checks complete. Installed-service and real-engine acceptance remain
separate; see the coordination receipts.**
Baseline `d117295`.
The user clarified that Kuno serves a trusted local network: all inference,
including image generation and previews, requires no Kuno authentication key.
This supersedes the earlier pending proposal for optional inference bearer
credentials, legacy access modes, and an explicit image-access migration.
See [the active handoff](kuno-vashti-coordination.md).

### Inference versus control

The compatibility listener exposes keyless image inference and discovery to the
same machines using Ollama through Kuno. There is no new authentication setting
or image credential. Existing installed configurations need no migration;
upgrading applies this policy. Old image bearer headers are ignored on these
routes. Normal Ollama proxy headers, upstream authentication/origin behavior,
request/response formats, and long inference semantics remain unchanged.

Privileged status, resource/learning reports, and admission pause/resume remain
on the loopback control listener and require its existing control credential.
Image generation is an inference operation and is not exposed on the control
listener. Public `/kuno/` routes cannot reach privileged controls: unknown
routes return 404 `unknown_inference_route` instead of proxying to Ollama.
Installation and updates remain local administrative commands.

Kuno-owned public inference routes grant no browser CORS access and reject any
`Origin` header with 403 `inference_origin_forbidden`. Vashti connects from Rust
server-to-server without that header. Image POST requires `application/json`
(optional charset accepted), otherwise 415 `image_content_type_required`.
This does not change ordinary Ollama origin handling. An external reverse proxy,
if deliberately configured, may impose its own access policy: a client must not
bypass a 401/403 by retrying anonymously. Kuno does not add a public/cloud hosting
or authenticated inference feature in this change.

### Discovery

`GET /kuno/v1/capabilities` returns HTTP 200 and `Cache-Control: no-store`.
Example with one enabled collection on a configured CUDA image device:

```json
{
  "service": "kuno",
  "discovery_version": 1,
  "version": "0.1.0",
  "inference_auth": "none",
  "images": {
    "api_version": 1,
    "access": "inference",
    "state": "ready",
    "reason": null,
    "generate_path": "/kuno/v1/images/generate",
    "response_formats": ["application/x-ndjson"],
    "workflows": [
      {
        "id": "qwen-image-2.1-text-to-image/v1",
        "state": "ready",
        "reason": null,
        "reference_images_min": 0,
        "reference_images_max": 0,
        "attention": ["standard", "fast"]
      },
      {
        "id": "qwen-image-2.1-edit/v1",
        "state": "ready",
        "reason": null,
        "reference_images_min": 1,
        "reference_images_max": 10,
        "attention": ["standard", "fast"]
      }
    ]
  }
}
```

`version` identifies the running Kuno package, independently of discovery/API
versions. `inference_auth: "none"` describes Kuno itself, not an external proxy.
Image states are `disabled` (not configured), `unsupported` (configured engine
absent from the build), `unavailable` (no enabled workflow usable), `partial`
(some usable), and `ready` (all usable). Workflow states are `ready` and
`unavailable`. Reasons are fixed public codes: `image_generation_not_configured`,
`image_engine_not_in_this_build`, `image_models_unavailable`,
`image_privacy_unavailable`, and `image_runtime_unavailable`. Partial state has a
null overall reason; per-workflow reasons explain unavailable workflows.
No paths, private model inventory, jobs, history, credentials, or GPU processes
are exposed. Only enabled built-in workflow IDs appear. Qwen 2512 supports
standard attention. Qwen 2.1 advertises fast attention only when the native build
includes CUDA and a configured CUDA worker identifies that image PCI device;
otherwise it advertises standard only. Unsupported fast requests are rejected
before dispatch with 400 `image_attention_unsupported`. Reference bounds are 0/0
for generation and 1/10 for edit. Existing recipe versions retain numeric
settings limits. Native seeds are decimal strings in 0..9223372036854775807.

Discovery never enters admission, loads/unloads models, contacts a GPU worker,
hashes weights, or downloads files. Prerequisite snapshots refresh on demand
at most once every five seconds, with one outstanding blocking probe per gateway.
A discovery request waits at most one second for a new snapshot, then returns
the previous snapshot (initially unavailable if configured). A slow probe can
continue in the background; readiness remains advisory. Checks inspect only
curated artifact existence/size and image privacy prerequisites.

Missing image prerequisites do not fail the entire gateway's startup. Generation
checks privacy prerequisites before reading/parsing image content and rechecks
model prerequisites before GPU preparation; cached readiness is neither a reservation
nor a VRAM guarantee. Existing queue/runtime errors stay distinct. Invalid
configuration still fails validation. GPU load does not itself mark the image
service unsupported. Artifact/prerequisite repair is recognized on the next
successful refresh or generation attempt, without restarting Kuno.

Vashti probes server-side at the configured inference base URL, preserving any
path prefix, with a two-second discovery deadline, a 64 KiB response limit, and
no redirects. These limits apply only to discovery. Validate the service marker
and supported discovery version; ignore additive unknown fields. Advertised
paths must remain on the same inference host/base path, never the control port.

- 404: no discovery support. Continue ordinary Ollama chat behavior. Old Kuno and
  plain Ollama cannot be distinguished by this result alone.
- Connection failure/timeout: discovery unknown/unreachable; do not remove chat
  configuration or claim the host is definitely plain Ollama.
- 401/403: an access failure; do not bypass it or downgrade to anonymous access.
- Malformed/unsupported version: discovery unavailable; leave chat configured.
- 200: use actual image availability; partial state permits its ready workflows.

### A single streamed generation response

Send the existing recipe to `POST /kuno/v1/images/generate`. Its successful
response is always `application/x-ndjson`; no Accept negotiation, preview
identifier, polling, or WebSocket upgrade is required. Headers include
`Cache-Control: no-store` and `X-Accel-Buffering: no` (a proxy hint, not a
buffering guarantee). Ordinary Ollama streaming remains unchanged.

Each UTF-8 JSON event is newline-terminated. An illustrative stream:

```json
{"type":"started"}
{"type":"preview","step":1,"steps":50,"width":64,"height":64,"png_base64":"<PNG base64>"}
{"type":"result","mime_type":"image/png","png_base64":"<final PNG base64>"}
```

`started` means admission accepted the request, not that sampling has begun.
Preview fields retain their existing meanings and bounds. Frames may be skipped
for slow clients: only the latest preview is held, not an ever-growing queue.
The terminal event is exactly one `result` or `error`, followed by EOF. Example:

```json
{"type":"error","error":"image_generation_failed"}
```

Validation/prerequisite/admission errors before streaming starts use the existing
non-2xx JSON HTTP error surface. Once streaming starts, HTTP status is 200 and
worker errors are terminal error events with fixed codes. EOF without a terminal
event is interrupted delivery, not success. Final PNG bytes are validated and
worker cleanup completes before a result is emitted. PNG/preview limits remain
20 MiB final PNG and 1 MiB preview pipe payload; base64 adds roughly one third.
Consumers must parse incrementally across arbitrary network chunk boundaries
and bound event/decoded-image sizes, rather than reading unlimited NDJSON.

The body owns the supervisor result receiver: closing it cancels inference.
The supervisor retains admission through worker cleanup; uncertain/forced/device
loss cleanup retains existing recovery behavior. A separate content-capacity
permit is shared by the body and supervisor, bounding slow completed responses
as well as active requests. Preview-channel closure does not terminate the final
result. No reconnection/replay, persisted jobs, or new response deadlines are
introduced. The existing configured image-worker deadline remains unchanged.

There is no separate preview lookup, enumeration, reconnect, or replay endpoint.
Previews belong to the initiating response only. The prior beta PNG-only response
and polling route are removed. A supplied old preview-ID header has no effect.
Previews/recipes/results remain transient and unlogged; closing or completing
the response releases its content. Vashti owns permissions and user-content
storage. Native-worker preview framing itself remains unchanged.

### Required Vashti changes and compatibility

Vashti should discover capabilities on the chosen existing inference backend,
remove the separate image-token readiness check/form/header, and use the
advertised ready workflows. Delete obsolete stored image credentials through a
forward migration; do not expose or repurpose them. No image credential
keep/clear workflow is needed after migration.

Kuno has only shipped in beta; compatibility with the old Vashti image connector
is deliberately not retained. Update that connector alongside this Kuno API.
Older Kuno lacks discovery and expects a control key for images: report that
images require a Kuno upgrade instead of restoring manual key setup. Plain
Ollama chat continues working when image discovery is unsupported.

Vashti's Rust connector should incrementally parse NDJSON, process previews,
validate/decode the final image, surface terminal errors, and cancel by dropping
the upstream response/request. EOF without a terminal event is a failure. Do
not retry/replay a possibly dispatched generation automatically. Rust
authorization/storage and mobile/desktop UI changes remain entirely in the
Vashti task. The browser receives its own authorized Vashti events, never a
Kuno control credential. Existing jobs must be quiescent before coordinated
installation; this handoff does not itself authorize deployment.

### Verification receipt

Backend verification on 2026-09-30:

- `cargo test -p kuno -p kuno-protocol -p kuno-image`: 171 passed, nine ignored
  hardware/native-environment checks; no failures.
- `cargo clippy -p kuno -p kuno-protocol -p kuno-image --all-targets -- -D warnings`:
  passed.
- `KUNO_SD_SOURCE_DIR=/home/jmicko/dev/vashti/target/kuno-sd-source-88411ef KUNO_SD_BUILD_DIR=/home/jmicko/dev/vashti/target/kuno-sd-build-88411ef-cuda cargo check -p kuno --features native-image --all-targets`:
  passed against the pinned CUDA/Vulkan static engine.
- Formatting and `git diff --check`: passed.

Synthetic checks include real HTTP preview delivery before completion,
disconnect propagation, exact final PNG decoding, terminal error/EOF ordering,
latest-only backpressure, retained content/admission permits, uncertain cleanup,
privacy rejection before body polling, keyless discovery/inference, control
separation, preserved Ollama headers/origins, degraded startup, and conditional
attention capabilities. No installed-service, live-GPU, end-to-end Vashti, or
mobile verification is claimed. Exact changed commits and consumer requirements
are recorded in [the coordination guide](kuno-vashti-coordination.md).
