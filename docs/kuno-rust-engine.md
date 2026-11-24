# Kuno Rust model engine

Goal: Kuno's image and video models run on Rust code only. Today the worker's model
execution is stable-diffusion.cpp with ggml's SYCL backend (C++), patched by
`apps/kuno/patches`. The Rust engine in `crates/kuno-image/src/engine` replaces it one model
at a time, behind the unchanged Kuno API and worker protocol.

## Method

1. Port one model's forward pass to Rust.
2. Write a slow double-precision decoder for it from the layer definitions, sharing no code
   with the fast path. This is the arbiter: when Rust and the C++ engine disagree, both are
   measured against it.
3. Capture the C++ engine's input and output for that stage from a real run and compare
   numbers and speed. A model takes over only when it is at least as accurate and at least as
   fast as the C++ stage it replaces.
4. Hand the stage over inside the worker: the C++ pipeline passes the stage's input tensor to
   Rust and skips its own work. Each hand-over removes C++ from the build when its last user
   is gone.

Captured tensors derive from test prompts written for the purpose. They stay outside the
repository (`kuno-video-dev/golden`); the comparison tests are `#[ignore]` and take their
paths from the environment.

## Status

| Stage (MiniMax H3) | C++ today | Rust | State |
|---|---|---|---|
| Audio decoder (BigVGAN, 32 kHz stereo) | 0.90 s on the GPU, 51 dB from exact | 0.47 s on 16 CPU cores, 129 dB from exact | Rust, in the worker |
| Video decoder (36-layer transformer) | 0.59 s per tile on the GPU, 70.9 dB from the CPU reference | 0.135 s per tile as machine code (79.1 dB); 0.26 s as Vulkan shaders (83.7 dB) on other cards | Rust, in the worker (dev build) |
| DiT sampling (INT8, 8 or 4 steps) | 3.35 s per step (66 ms per block); warm 2.3 s clip 47.7 s | whole network and text refiner on the GPU, follows the C++ sampler; 3.40 s per step (66 ms per block); warm clip 48.1 s, first clip 56 to 60 s (C++ 60.9 s) | correct, within 2% per step; kept in video memory between clips; with the machine-code decoder and the Rust text encoder a warm clip is 40.2 to 40.3 s |
| Qwen3-VL text encoder (INT8, 50 layers) | 3.55 s per clip, of which 3.3 s copying 25.5 GB to the GPU | language model on the GPU with weights streamed from host memory: 2.55 s; the vision tower stays C++ | Rust, in the worker (dev build); text tokens 37 dB from the C++ engine |
| Video encoder, single frames (start, end, reference images) | about 1.8 s per frame, in 256-pixel tiles | whole frame on the GPU: 106 to 128 ms for 832x544; 76 to 78 dB from exact | Rust, in the worker (dev build); 64.6 dB from the C++ engine's whole-frame encode; multi-frame (video reference) encodes stay C++ |
| Vision tower (Qwen3-VL, single images) | about 0.39 s per image; 30 to 44 dB from exact | whole image on the GPU: 135 to 150 ms for 832x544; 44 to 58 dB from exact | Rust, in the worker (dev build); video-reference frame pairs stay C++ |
| Tokenizer (Qwen2 BPE) | C++ | Rust, id for id | done; 96 synthetic strings match the engine's own tokenizer |
| Pipeline (latents, noise, schedule, Euler loop, previews, decode glue) | `video.cpp` | `native/pipeline.rs` | Rust, in the worker (dev build): MiniMax H3 video no longer calls the C++ engine |
| Audio encoder, video references | C++ | not started | Kuno does not send audio or video references yet |
| Clip encoding (AV1, FLAC, MP4) | n/a | Rust since the first video release | done |

| Stage (Qwen Image 2.1) | C++ today | Rust | State |
|---|---|---|---|
| Diffusion transformer (32 blocks, INT8) | about 1.14 s per step at 1184x1776 (8214 tokens) | prompt and references once per request (their keys and values kept per block), then target-only steps: 1.23 s per step at 1184x1776; 38 to 43 dB from captured C++ steps (the CPU reference too) | Rust (dev build) |
| Text encoder (Qwen3-VL 8B, 36 layers) | C++ | the MiniMax encoder with its sizes from the file; 0.77 s for 469 tokens; 38.4 dB from the CPU reference | Rust (dev build) |
| Vision tower (references) | C++ | the MiniMax tower with its prefix and width from the file; 146 ms at 832x544 | Rust (dev build) |
| Prompt template and emphasis | C++ | `(text:1.5)`, `[text]`, `BREAK` byte for byte (18 tricky prompts) | Rust (dev build) |
| VAE decoder (RGBA) | 2.40 s at 1184x1776 | 1.58 s; 86 to 87 dB from the engine, worst byte 1 | Rust (dev build) |
| VAE encoder (references, RGBA) | C++ | 20 to 42 ms; 64 to 68 dB from the engine | Rust (dev build) |
| Pipeline (FLUX schedule, Euler, CFG, previews) | `image.cpp` | `engine/qwen_image.rs`, `native/qwen.rs` | Rust (dev build): text-to-image 40.7 dB from the engine's image of the same request; edits the same picture (CFG amplifies the encoders' small differences) |


| Stage (Qwen Image 2512, FP8) | C++ today (Vulkan) | Rust | State |
|---|---|---|---|
| Diffusion transformer (60 dual-stream blocks) | about 17.8 s per pass at 832x1248 (4056 tokens) | INT8 block linears made at load from the FP8 weights (rotated, per row), FP8 modulation evaluated exactly per step: 0.53 s per pass (0.49 s fast); 37 to 48 dB from captured C++ steps | Rust (dev build) |
| Text encoder (Qwen2.5-VL 7B, FP8) | C++ | the streamed GPU encoder with biases, no query/key norms, plain rotary and the final norm; 0.69 s per prompt; 30 to 32 dB from the engine | Rust (dev build) |
| VAE decoder (Wan 2.1, RGB) | 2.8 s at 256x256, tiles at 832x1248 | the 2.1 decoder with the causal kernels' last time slice: 0.32 s at 832x1248; 34.5 dB from the engine's half-float decode | Rust (dev build) |
| Pipeline (simple schedule with shift, Euler, CFG, previews) | `image.cpp` | `engine/qwen2512_image.rs`, `native/qwen2512.rs` | Rust (dev build): 20 steps at 832x1248 with CFG 23.1 s warm (C++ 755 s); the same picture, 3.5 of 255 apart on average |

The audio decoder runs on CPU threads on purpose: the engine hands over the audio latent
before it decodes video frames, so audio is decoded while the GPU is busy with video, and it
is skipped entirely when a request turns audio off. The C++ audio VAE is no longer loaded.

Why the C++ audio decode is only 51 dB from exact has not been isolated (rounding
convolution inputs to half precision, as ggml's `conv_1d` does, accounts for about 70 dB,
not 51). It is audible to no one, but it means the Rust output is not sample-identical to
earlier releases; it is closer to the model.

## Video decoder

The MiniMax H3 video decoder is not a convolutional VAE. It is a 36-layer transformer
(width 2048, 32 heads, gated feed-forward of 8192, about 2.4 billion half-precision weights)
that turns each latent position into a 4x16x16 block of pixels. Only the encoder, used for
start and end frames, is 3D convolutions with group normalization.

So the decoder needs the same kernels as the DiT: a half-precision matrix product, RMS
normalization, rotary positions, attention and the gated activation. Building them for the
decoder builds most of what the DiT needs.

The C++ engine decodes in tiles of 16x16 latent positions by 7 latent frames (1797 tokens
with the 4 register tokens and one zero token). A 832x544 clip is 12 tiles per time chunk
and 3 chunks for 57 frames; each tile takes about 0.59 s on the B70, roughly 15 TFLOPS for
about 9 TFLOP of matrix products. The Rust Vulkan matrix kernels in the text engine reach
about 55 TFLOPS on the same card, so this stage has room for a real speedup.

`engine/video_vae.rs` is the CPU form of one tile (`VideoDecoder::decode_tile`). It streams
one layer's weights at a time and is the numeric reference for the GPU kernels, not a
production path. On a tile captured from the C++ engine (16x16x7 latents, 28 frames of
256x256) it is 130.8 dB from the exact double-precision decode; the C++ engine, which
computes in half precision, is 70.9 dB from exact. The CPU form takes 32 s per tile.

### GPU form

`engine/gpu` is a small Vulkan compute layer on `ash`: one device (chosen by PCI address in
the worker), storage buffers, compute kernels and batched dispatch. Kernels are GLSL in
`engine/gpu/shaders` with their SPIR-V checked in, so a build needs no shader compiler; a
test recompiles them when `glslangValidator` is installed and fails on stale binaries
(`KUNO_UPDATE_SHADERS=1` rewrites them).

| Kernel | Job |
|---|---|
| `gemm16` | Linear layers: f16 operands, f32 accumulation, eight accumulators per subgroup |
| `gemm8` | The same plan for INT8 operands with exact int32 sums (DiT) |
| `gemm` | Batched f16 product in 128 x 128 tiles, used for attention scores |
| `norm` | RMS or layer normalization of the f32 stream, written as f16 |
| `heads` | Query/key/value split with bias, unit RMS and rotary positions |
| `mix` | Attention weights `exp(score / 8)` computed while the mixing product copies its operand, padded keys masked |
| `merge` | Per-head attention output back into token rows, divided by the weight sum |
| `residual` | `x += scale * (update + bias)` on the f32 stream |
| `gate` | Gated activation between the feed-forward projections |

`engine/video_gpu.rs` chains them (`GpuVideoDecoder`): the residual stream stays in f32,
matrix operands are f16, and tokens are padded to a multiple of 128 so the matrix kernel
needs no bounds checks. `engine/video_tiles.rs` is the clip level: the reference engine's
tile plan, cross-fades and time chunks, unit-tested with fake tile decoders.

Measured on the B70 (Mesa 26.2.6):

| | C++ engine | Rust |
|---|---|---|
| Matrix kernel at the decoder's largest shape | about 15 TFLOPS effective | 63.5 TFLOPS |
| One 16x16x7 tile | 0.59 s, 70.9 dB from exact | 0.27 s, 83.7 dB from the CPU reference |
| 832x544, 73-frame decode (48 tiles), standalone | 31.4 s | 13.1 s, 69.7 dB from the C++ frames |
| Weights to the GPU | part of the above | 1.1 s per clip |
| Decode inside the worker, 2.3 s clip (36 tiles) | about 21 s | 1.5 s load + 10.2 s decode |
| Warm 2.3 s clip end to end | 59.1 to 59.5 s | 47.4 to 47.5 s |

The worker's clips are byte-identical between the first GPU form (0.34 s tiles, 50 s clips)
and the current kernels, and three same-seed clips are byte-identical to each other.

What shaped the kernels:

- Cooperative-matrix load strides are in elements of the array being read (vec4s for a
  `f16vec4` array), not in scalars.
- Half-precision subnormals read as zero unless the shader asks for `DenormPreserve` on
  16-bit floats; all kernels do.
- Elementwise kernels must have neighbouring lanes touch neighbouring memory. One invocation
  per row with a long loop was 10 to 40 times slower than one vec4 per invocation.
- The matrix kernel needs a few dispatches to reach full speed (31 TFLOPS over 8 runs, 63
  over 64).
- Queries and keys have unit RMS, so `score / 8` stays within -8..8 and `exp` needs no row
  maximum. A row of ones in the value operand makes the mixing product return each row's
  weight sum for free, so the softmax needs no reduction pass.
- The linear layers moved to the DiT's eight-accumulator plan with 4096 weight rows per
  dispatch (0.335 s to 0.30 s per tile), and the attention weights moved into the mixing
  product's operand copy, which removed a pass over every score (0.30 s to 0.27 s).
  Half-precision scores would save another 3% but cost 6 dB, so scores stay f32.
- Attention is limited by memory traffic, not arithmetic: making the softmax elementwise
  saved little. Per tile, hot: linear layers 151 ms (57 TFLOPS), scores 32 ms, weights
  about 50 ms, mixing 20 ms, operand split 20 ms, everything else about 25 ms.

In the worker the C++ pipeline hands the video latent (VAE scale) to Rust through
`sd_kuno_set_video_decoder` and decodes itself only if Rust reports failure. The decoder is
loaded for the call and dropped after, so it holds no GPU memory while other models run.

Still to do for this stage: a fused attention kernel (scores, weights and mixing are about a
third of a tile and limited by memory traffic), and the encoder.

## Diffusion transformer

The DiT is 50 blocks of width 5376 (56 heads of 128, gated feed-forward of 14336) over one
sequence of text, condition, audio and video tokens: about 8600 tokens for a 2.3 s clip at
832x544. Its four linear layers per block are INT8 with a per-row scale in a rotated basis
("ConvRot"): activations are rotated by a Hadamard transform in groups of 256, rounded to
INT8 with one scale per token, and multiplied as integers. Each token carries one of a few
time/modality rows that shift, scale and gate the block (text tokens tagged as image
placeholders use the video row).

`engine/dit.rs` is the CPU reference for one evaluation (`Dit::forward`): token layout and
positions, the quantizer in the C++ engine's arithmetic order, modulated blocks, partial
rotary attention (96 of 128 head channels, pairs `(i, i + 48)`) and the output heads. It is
only practical for small latents (58 s for a 192x128 clip's step on 16 cores).

There is no separate exact arbiter for this model, because rounding to INT8 in every layer
amplifies any arithmetic difference. On a step captured from the C++ engine the reference is
27.9 dB (video) and 23.6 dB (audio) away; the reference with attention operands rounded to
half precision, as the C++ engine has them, differs from itself by 30 and 23 dB. So whole
network outputs can only confirm structure. The GPU form has to be checked per block against
this reference on identical inputs.

### What the C++ engine spends per block

Profiled on the 2.3 s clip (8611 tokens), per block per step:

| Part | C++ engine | How |
|---|---|---|
| Attention | 24.8 ms | oneDNN scaled dot-product attention, about 83 TFLOPS |
| Query/key/value product (5376 to 21504) | 13.4 ms | oneMKL INT8 product plus fused norm and rotary |
| Feed-forward in (5376 to 28672) | 12.2 ms | oneMKL, about 217 TOPS |
| Attention out and feed-forward out (to 5376) | 8.0 ms | oneMKL |
| Norms, modulation, quantizers, residuals | about 8 ms | fused SYCL kernels |

The two libraries are Intel's own hand-tuned GPU code, so this stage is the hard one: the
Rust kernels have to match them through Mesa's compiler.

### The INT8 matrix kernel

`gemm8` is the integer form of the cooperative-matrix kernel and is bit-exact. Measured at
the four layer shapes with 8448 tokens:

| Kernel | Per block | Rate |
|---|---|---|
| First port of the half-precision kernel's plan | 56 ms | 96 to 122 TOPS |
| Current `gemm8` | 38 ms | 126 to 178 TOPS |
| C++ engine (oneMKL, with its fused epilogues) | about 34 ms | 150 to 260 TOPS |

What the measurements showed, in order:

- The multiply instructions alone (operands kept in registers) run at 268 to 331 TOPS, and
  with shared-memory loads and barriers at about 190. The first port ran at 120 because of
  the copy from global to shared memory: a kernel doing only that copy took as long as the
  whole kernel.
- The weight copy cost twice the token copy for the same bytes. Workgroups run across weight
  tiles first, so a token tile stays cached while every workgroup reads a new weight tile
  from video memory (7.6 GB per product, close to the card's memory bandwidth). Dispatching
  4096 weight rows at a time keeps the weight block cached while all tokens pass over it.
- Reordering each step to request the next operands first, multiply, then store them lets
  the multiply overlap the wait.
- Eight accumulators per subgroup (32 tokens by 32 rows) halve the bytes copied per multiply
  and still fit the 128 registers when each step loads only one 32-column fragment set.
  Staging 64 or more columns per step with eight accumulators spills and halves the speed.
- Loading operands straight from global memory without shared memory was slower (75 TOPS),
  also on Kuno's own Mesa build. Tile shapes with more weight fragments than token fragments
  were slower too. 1024-lane workgroups gained nothing.

### The GPU form

`engine/dit_gpu.rs` runs the 50 blocks on the GPU (`GpuDit`); embedding, the text refiner and
the output heads still use the CPU reference's code. Weights load in 4 s from page cache
(19 GB). One submission per block, because the kernel driver ends jobs that run for several
seconds.

| Kernel | Job |
|---|---|
| `modnorm` | RMS norm with weight, then the token's shift and scale |
| `rotate8` | ConvRot quantizer, bit-identical to the reference |
| `gemm8` | INT8 products, 4096 weight rows per dispatch |
| `qk8`, `value8` | Attention operands from the integer sums: unit RMS, weights, rotary; values transposed |
| `scoremax`, `rowmax`, `mixtop` | Scores as f16 relative to each row's largest score per key tile, the row's largest score overall, true-softmax mixing with weight sums |
| `merge8`, `resid8`, `gate8` | Head outputs to token rows, gated residuals, gated activation |

Scores reach about 30 in most blocks and 300 in the last (large query and key norm weights),
so the decoder's shortcut of exponentiating without a row maximum does not apply here.

Checks:

- Quantizer: bytes and scales equal the reference's at all three widths. The GPU's
  single-precision divide is not correctly rounded, so the scale is divided in double
  precision.
- Single blocks on identical inputs (blocks 0, 20, 49): 54 to 56, 40 and 45 dB from the
  reference's update, the same at 317 and at 8611 tokens. The gap is the half-precision
  attention operands, which the C++ engine also has.
- Whole network against steps captured from the C++ engine: 29.4 / 24.4 dB (video / audio)
  at 317 tokens, 23.9 / 18.1 dB at 8611.
- Sampling: all eight captured steps of a small clip are 27 to 32 dB from the C++ outputs.
  Running the Euler loop on this DiT's own outputs from the same noise ends 24.6 dB from the
  C++ final latent; decoded, the two clips are 27.7 dB PSNR apart and show the same picture.

Time per block at 8611 tokens:

| Part | First GPU form | Now | C++ engine |
|---|---|---|---|
| INT8 products | 40 ms | 41 ms | about 34 ms |
| Scores, row maximum, mixing | 33 + 30 + 33 ms | 27 + 5 + 25 ms | 24.8 ms |
| Operands, norms, quantizers, residuals, gate, merge | 93 ms, then 15 ms | 15 ms | about 8 ms |
| Whole block | 234 ms | 107 ms | 66 ms |

What moved attention:

- The first form wrote every score to video memory in f32 and read it twice (17 GB per pass
  per block); a pass costs about 30 ms at the card's memory bandwidth whatever the kernel.
- The row maximum now comes out of the scores kernel: after the product its staging memory is
  free, the f32 accumulators are stored there 64 token rows at a time, and eight lanes per row
  reduce them. `rowmax` scans 68 values per row instead of 8704.
- The same lanes subtract that maximum and write half precision. Scores near a row's maximum,
  the ones that carry weight, keep nearly full precision; the block's agreement with the
  reference did not change. Converting the accumulators to f16 with the matrix store instead
  kept the accuracy too but made the kernel 50% slower.
- Mixing on the eight-accumulator plan (256 queries by 128 value rows) gained nothing while it
  read f32 scores, because it was bound by reading them, and 8 ms once they were f16.
- No gain: the eight-accumulator plan for the scores product (its loop is 8 steps; the cost
  is in staging and the epilogue), and reading back stored scores to pre-reduce the maximum
  (lanes read values other lanes had not finished storing).
- bfloat16 weights would need no row maximum for all but the last block, but the value
  operand would have to be bfloat16 too (8 bits of mantissa), and the installed shader
  compiler (glslang 15.1) does not have the type.

Even with scores never leaving the workgroup, the two products alone take about 43 ms at the
rate these kernels reach; the C++ engine's whole attention is 25 ms.

### In the worker

Built with the `rust-dit` feature (`cargo build -p kuno --features native-image,rust-dit`),
the worker samples MiniMax H3 with the Rust DiT. The feature is off by default, because a
step still takes longer than the C++ engine's; a default build is unchanged.

The engine patch adds `sd_kuno_set_dit`: every network evaluation is offered to the
evaluator with its inputs (packed latent, timestep, text features and tags, condition
latents, key-frame and reference layout, schedule shifts). Before sampling the engine
releases the text encoder's and the VAE's device memory, which it does not need while the
evaluator holds the device, and since its own DiT is never evaluated its weights are never
staged. The worker loads the Rust DiT at a clip's first evaluation (4.6 s) and drops it
before decoding, so it holds no GPU memory while other models run. An error hands the
remaining steps back to the engine.

Checked with the feature on (dev worker, 2.3 s clips at 832x544):

- All seven modes (image to video, first frame, text only, first and last frame, last frame
  only, two references, one reference) produce clips, and a Qwen image after a video still
  works. The reference modes use the second DiT file and the reference token layout, which
  no captured step had covered.
- Side by side with clips the C++ sampler made from the same requests, each mode shows the
  same scene at the same quality. Frames are not identical: 32 to 34 dB PSNR for the
  image-conditioned modes, 23 dB for text only, 16 dB for the 4-step reference modes, where
  the composition drifts in details.
- Two same-seed clips are byte-identical to each other.
- A step takes 6.2 s (C++ engine 3.4 s): 5.4 s in the blocks, 0.3 s reading the stream back,
  0.2 s embedding, 0.3 s for modulation vectors, uploads and output heads on the CPU. The
  first clip of a worker takes 80 s where the C++ sampler takes 63 s.

### Reaching vendor-library speed without SYCL (probe, 2026-11-13)

The C++ engine's speed on the two hot operations does not come from SYCL or
from Intel's compiler. A synthetic int8 product (8192 x 5376 x 14336, exact
against CPU spot checks) was run three ways:

| Path | Throughput |
|---|---|
| Our cooperative-matrix kernel through Mesa (`gemm8`) | 126-178 TOPS |
| Same kind of kernel through Intel's compiler, direct loads, packed weights | 36 TOPS (128 registers), 55 TOPS (256 registers, 4x4 tile) |
| oneMKL int8 product (earlier vendor probe) | about 299 TOPS |

Intel's compiler accepts `SPV_KHR_cooperative_matrix` in OpenCL-flavoured
SPIR-V, but a kernel of ours compiled by it is slower than the Mesa kernels,
and register spills are far more costly there (3-6 TOPS when a tile does not
fit). oneMKL and oneDNN reach their speed with kernels emitted as machine
code by their own generator, not with a compiler.

A Rust program with no SYCL runtime in the process drove the card through
Level Zero's C API (`libze_loader.so.1`, about 20 functions): context, queue,
device memory, module from SPIR-V, kernel launch, read-back. The driver's
native module is a relocatable ELF: raw kernel code in `.text.<kernel>` and a
YAML `.ze_info` section naming register count, SIMD width and the argument
payload layout. Reloading that file as a native module ran at the same speed.
So Rust can hand the driver machine code it generated itself, which is the
route to vendor-library speed with no C++ library in the engine: a small Xe2
instruction encoder in Rust (encodings are in Mesa's `brw_eu` sources), a
fixed-shape int8 product and attention kernel written against it, and Level
Zero plumbing. Mesa (Vulkan) and Intel's runtime (Level Zero) already share
the card in today's worker.

Probe sources live outside the repository in
`kuno-video-dev/igc-probe` (`jm8.cpp`, `ze/`).

### Hand-written machine code (probe, 2026-11-14)

A hand-written int8 product kernel, loaded as a native module through Level
Zero from Rust, was timed on the DiT's largest layer shape (8192 x 5376 x
14336, 4000 random outputs checked against the CPU, all exact):

| Kernel | Time | Throughput |
|---|---|---|
| First version, loads and math serialized | 7.07 ms | 179 TOPS |
| Next step's loads overlapped with the math | 5.98 ms | 211 TOPS |
| Same, dispatched in blocks of 4096 tokens | 4.96 ms | 255 TOPS |
| Math only (no loads in the loop) | 3.53 ms | 358 TOPS |

The best Mesa kernel reaches 126-178 TOPS and the oneMKL probe about 299.
Times include the launch and wait.

How the kernel is built:

- One hardware thread per tile of 32 tokens x 64 outputs, 256 registers:
  16 accumulators of 8 registers, two operand sets of 48 registers.
- `dpas.8x8 (16)` multiplies an 8 x 32 token fragment (4 registers, one
  row per 8 dwords) by a 32 x 16 weight fragment (8 registers, one k-group
  of 4 bytes per lane per register) into 8 rows x 16 dwords.
- Operands are stored as streams in exactly that order, so one 256- or
  512-byte transposed load fills a fragment; address offsets are immediates.
- `dpas` runs out of order like a send: it sets a scoreboard token, later
  readers and writers of its registers wait on that token. Leaving this out
  gives wrong sums.
- The kernel is bound by bytes loaded, not by the multiplier and not by the
  number of load messages. Loads are served from the card's cache when the
  token operand of a dispatch block fits (22 MB does, 44 MB does not).
- Thread payload: group ids arrive in `r0.1`, `r0.6`, `r0.7`; the first 32
  bytes of arguments arrive in `r1` when `.ze_info` declares inline data.

The prototype assembles text with Mesa's `gentool` (built from
`mesa-26.2.6` with `-Dtools=intel`) and wraps the code with a small script.
Sources are outside the repository in `kuno-video-dev/igc-probe/hand`
(`gen.py`, `mkzebin.py`) and `igc-probe/ze` (Rust runner `hand`).

### Machine-code linear layers in the engine (2026-11-14)

The prototype is now Rust in the engine (`engine/xe2`), with no outside
assembler or SYCL involved:

- `xe2/mod.rs` encodes the instruction forms used (mov, add, mul, cmp,
  jmpi, sync, send, dpas) with their dependency annotations. Its tests hold
  Mesa's assembler output for the same instructions; the whole kernel was
  byte-identical to the assembled prototype on five shapes before the
  prototype was retired.
- `xe2/zebin.rs` writes the native module container. The compatibility
  note is Battlemage's, copied from a driver-compiled module; another card
  rejects the module and the engine falls back to the shader.
- `xe2/level_zero.rs` binds the dozen Level Zero calls needed, opening
  `libze_loader.so.1` at run time.
- `xe2/int8.rs` is the kernel and its operand layouts.

The DiT's Vulkan kernels and the Level Zero kernel work on the same memory:
Vulkan allocates the buffer with dma-buf export and Level Zero imports the
descriptor. The memory must be `DEVICE_LOCAL`; a host-visible type that is
not device-local is system RAM behind PCIe and ran the product 240 times
slower. Each block is now four Vulkan submissions with a Level Zero launch
between them, each waited for.

What changed against the prototype:

- Sums are stored row-major (`[token][row]`), one 64-byte store per
  accumulator row, so the shaders that consume them are unchanged. This
  costs about 6% against storing whole accumulators.
- The quantizer shader writes token bytes directly in the kernel's stream
  layout; weights are rearranged once at load (load time is unchanged at
  about 9 s).
- Threads dispatched past the last token tile exit at once, so the token
  block size need not divide the token count.
- The token block is 100 tiles (3200 tokens) for every layer. Measured at
  8704 tokens, in TOPS:

| Layer (inputs x outputs) | 8 tiles | 32 | 64 | 100 | 134 | 272 |
|---|---|---|---|---|---|---|
| qkv (5376 x 21504) | 186 | 212 | 225 | 235 | 226 | 114 |
| out (7168 x 5376) | 208 | 218 | 237 | 241 | 235 | 164 |
| fc1 (5376 x 28672) | 186 | 209 | 222 | 232 | 222 | 110 |
| fc2 (14336 x 5376) | 203 | 208 | 221 | 226 | 225 | 193 |

Result at 8611 tokens: the four linear layers of a block take 30.5 ms
(41 ms with the shader; the C++ engine spends 33.6 ms), a block 98 ms
(105), a step's blocks and heads 5.4 s (6.0). The integer sums equal the
shader's on every output (`machine_code_product_matches_shader`), and the
block and whole-step comparisons pass unchanged. What remains against the
C++ engine's 66 ms per block is attention (52 ms against 25) and the small
kernels around it (19 ms against about 8).

### Attention as machine code (first kernel, 2026-11-15)

`xe2/attend.rs` is a fused attention kernel: one thread owns 16 queries of a
head and walks the keys 16 at a time, computing scores (half-float `dpas`,
exact at half the int8 rate: 120 TFLOPS in the product probe), turning them
into half-float weights and mixing the values, with nothing stored in
between. It is checked against a CPU softmax on random operands (73 to 83 dB)
and is **not wired into the DiT yet**: at clip size it takes 90 ms against
the shaders' 54 ms.

What it established:

- Dependency distances count instructions of the producer's pipe. Integer
  and float instructions run in separate in-order pipes, so where a group of
  instructions reads what the other pipe's group wrote, "four back" is wrong
  for all but the first; the symptom was wrong weights in three rows of four.
- Float sources must sit at their lane position, so a reduction across lanes
  is done on integers (non-negative floats order like their bits), and a
  16-lane conversion to packed half floats takes two instructions.
- A thread's loads are served about 0.65 microseconds after they are issued.
  The kernel issues one 512-byte load at a time (its registers are spent on
  16 queries x 128 channels of f32 accumulators), so 17 loads per step make
  it latency-bound. The card's load throughput (about 88 G cache lines per
  second, from the int8 kernel) would allow about 25 ms.

### Attention in the engine (2026-11-15)

The kernel was rebuilt with the products transposed: queries are the 16 lanes
of every register, keys and channels are register rows. Per query, the
reference and the weight sum are then one lane of one register, a key's
weights convert to half floats in place, and 24 registers are left for three
512-byte load buffers that are refilled as soon as their products have read
them (the next chunk's first keys are requested during mixing). The last,
partial key chunk gets its own copy of the step that skips dead keys, so no
masking or marker channel is needed.

| At 8611 tokens, 56 heads | Time |
|---|---|
| Shaders: scores, row maxima, mixing | 54 ms |
| First kernel, one load in flight | 90 ms |
| Transposed kernel, three load buffers | 39 ms |
| Same without computing the weights | 36 ms |
| Exponents in place, up to five loads in flight | 32 ms |
| Same without computing the weights | 30 ms |

The three-buffer kernel was bound by operand loading (about 60 G cache lines
per second against the 88 G the int8 kernel reaches). The current one writes
the exponents over the scores, checks for overflow before any weight is
computed (the slow path lowers the exponents instead of starting over), and
borrows idle registers as load buffers: the weights' registers while scores
are computed, the scores' registers while values are mixed. Four loads are in
flight during scoring, two during the weights, five during mixing. A step of
16 keys now takes about 3.7 microseconds per thread, of which the 32 `dpas`
alone are about 2.2 at the rate the half-float probe reached, so what is left
is mostly arithmetic.

In the DiT (`Kernels::block`), three shader variants write the kernel's
layouts (`qk8s`, `value8s`) and gather its transposed output (`merge8s`).
Work buffers and the kernel are built for an exact token count. One block at
8611 tokens: 78.6 ms (attention 32.5, linear layers 29.4, the rest 17); a
step's blocks and heads 4.45 s (C++ about 3.4 s). `block_matches_reference`
gives 39 to 47 dB and the whole step passes against the C++ engine (video
23.0 dB, audio 22.4 dB).

### Fewer passes around the kernels (2026-11-15)

With the two hot kernels as machine code, what was left of a step was memory traffic
around them and work still on the CPU. At 8611 tokens:

| | Before | Now | C++ engine |
|---|---|---|---|
| Attention per block | 39.8 ms | 32.5 ms | 24.8 ms |
| Linear layers per block | 29.1 ms | 29.6 ms | 33.6 ms |
| Small kernels per block | 17.4 ms | 12.6 ms | about 8 ms |
| One block | 84.8 ms | 74.9 ms | 66 ms |
| One step: inputs, blocks, heads | 4.94 s | 3.83 s | 3.35 s |
| Text refiner, once per clip | 3.85 s (CPU) | 0.19 s | |

- The small kernels are bound by video memory bandwidth (about 450 GB/s over 7.9 GB per
  block), so the gain is in passes not made. The quantizer used to read a finished f32 row
  twice (largest magnitude, then rounding). `normq8`, `mergeq8` and `gateq8` compute the
  row, rotate it into the workgroup's shared memory and round from there, so the row is
  never written out. Shared memory is 49152 bytes per workgroup: a 14336-wide row does not
  fit, so `gateq8` keeps half there and passes half through a buffer.
- Output heads (`final8`), input projections (`embed8`) and the blocks' modulation vectors
  (`parts8`) run on the GPU. A step now uploads 3 MB and reads back 3 MB where it moved
  450 MB; the CPU and transfer time of a step went from 0.66 s to about 0.03 s.
- The text refiner (`dit_gpu/refiner.rs`) is two blocks of the DiT's shape with
  half-precision weights on the text tokens: the half-precision product shader, the DiT's
  attention kernel built for the text's token count (its operand writers `qk8s` and
  `value8s` gained an f32 input mode), and four small shaders. 56.6 dB from the CPU
  reference on a captured text of 469 tokens. Its weights add 1.5 GB of video memory. It
  needs Level Zero; without it the CPU refiner runs.
- `projections_and_heads_match_reference` holds the projections and heads to the CPU
  reference (119 dB with no block between them, 54 dB with one);
  `refiner_matches_reference` holds the refiner.

Whole-network checks after these changes: a captured 8611-token step is 22.3 / 21.2 dB
(video / audio) from the C++ engine; the eight steps of the small captured clip are 27 to 32
dB from the C++ outputs, the Euler loop on this DiT's own outputs ends 25.7 dB from the C++
final latent, and the decoded clips are 28.8 dB PSNR apart (27.7 before).

In the dev worker (`rust-dit` build, 2.3 s clip at 832x544, 8 steps, same request):

| | Rust DiT before | Rust DiT now | C++ sampler |
|---|---|---|---|
| One step | 6.2 s | 3.86 s | 3.35 s |
| First clip of a worker | 80 s | 71.5 s | 60.9 s |
| Later clips | | 52.5 s | 47.7 s |

The worker used to read all 21 GB from the file for every clip, because the DiT has to
leave video memory before the decoder and the next clip's text encoder load (7.5 s per
clip). The weights now stay in host staging memory for the worker's life (`Host`, one model
at a time, 21 GB of RAM); a clip copies them into video memory in 2.1 s (`GpuDit::place`)
and frees that before decoding as before. The text features kept for a clip are dropped
with it. The C++ engine prepares weights lazily, so in this build it never reads its own
copy of the DiT (the worker's file-backed memory stays at the text encoder's 27 GB).

All seven modes produce clips in this build and a Qwen image after a video still works;
the same request gives the same bytes across worker builds and across clips. Contact sheets
of the text-only and two-reference clips show the expected scenes.

### Weights read in the background (2026-11-15)

| Same clip, dev worker | Before | Now | C++ sampler |
|---|---|---|---|
| Read of the DiT's weights (page-cached file) | 7.7 s after conditioning | 4.4 s alone, 5.5 s beside the text encoder | |
| Video decoder weights per clip | 1.5 s from the file | 0.53 s from host memory | 1.5 s (same Rust decoder, older build) |
| First clip of a worker | 65.0 s (71.5 s on a cold page cache) | 56.1 s | 60.9 s |
| Later clips | 52.8 s | 51.3 to 51.5 s | 47.7 s |

- The read starts with the request, on its own thread, and the first step waits for it
  (`dit_host` in `native.rs`). The text encoder's load slows by about 2 s beside it, still a
  clear net gain.
- Each matrix used to be read whole into a temporary and then rearranged into the kernel's
  stream layout by one thread per layer. Tiles of 64 rows are now read and rearranged in
  parallel, so the bytes cross memory once (`LayerHost::read`, timed by `reads_weights`).
  More than four reader threads gain nothing.
- The video decoder keeps its weights and compiled kernels in host memory the same way
  (`video_gpu::Host`, 5 GB of RAM) and they are read on the same background thread. Decoder
  checks are unchanged: 69.7 dB from the C++ frames, 83.7 dB from the CPU reference.
- Output is unchanged: the same requests give the same MP4 bytes as before.

What is left of the gap to the C++ sampler on a later clip (3.6 s): 0.5 s per step, of
which attention is 0.38 s. A C++-sampler build of the same tree would also gain the
decoder's second.

### Attention was waiting for memory, not arithmetic (2026-11-16)

The earlier reading of the attention kernel ("what is left is mostly arithmetic") was
wrong. Timing switches in `attention_matches_cpu` (they drop the weights, the key and value
loads, or either product) gave, at 8611 tokens and 56 heads:

| Kernel variant | Time |
|---|---|
| Whole kernel | 31.7 ms |
| Loads only, no arithmetic | 28.3 ms |
| Arithmetic only, loads removed | 14.1 ms |
| Loads only, every step from the same chunk | 7.2 ms |
| Loads only, every head reading the first head's stream | 11.8 ms |
| One head launched alone (544 threads, half the thread slots) | 1.08 ms per head |
| Two heads launched together (all slots) | 1.03 ms per head |

So the loads were cheap when the data was already cached and expensive otherwise, and
filling the second half of the thread slots bought nothing: the card was saturated with
cache misses. All 544 threads of a head start together and walked the keys in step, so
each chunk was demanded by every thread at the same moment, before any of them had
brought it in.

Now tile `i` starts at key chunk `i` and wraps around (the softmax sum does not care about
order; the partial last chunk keeps its own copy of the step, and the three loads issued
past the last chunk are reissued after the wrap). A thread then mostly reads what another
thread read a step earlier.

| At 8611 tokens | Before | Now | C++ engine |
|---|---|---|---|
| Attention kernel alone, 56 heads | 31.7 ms | 19.9 ms | |
| Attention per block | 32.5 ms | 21.0 ms | 24.8 ms |
| One block | 74.9 ms | 66.0 ms | 66 ms |
| One step: inputs, blocks, heads | 3.83 s | 3.40 s | 3.35 s |

Checks: 73.4 dB from the CPU softmax (unchanged); blocks 39.2 to 47.3 dB; a captured step
21.3 / 22.5 dB (video / audio) from the C++ engine; refiner 60.0 dB; the eight-step loop
ends 25.1 dB from the C++ final latent and the decoded clips are 27.8 dB apart. The order
of the softmax sum changed, so these moved by up to a decibel either way.

The same stagger was tried on the int8 linear kernel (start step `(a*x + b*y) mod ks/2`,
wrapping) and removed: results stayed exact but every pattern was slower than threads in
step (227 to 242 TOPS in step, about 200 staggered). That kernel was already tuned by its
dispatch block size and is near its arithmetic rate.

Other start patterns (chunk `k*i mod chunks` for k from 2 to 537) were all slower than
`k = 1`; the reverse order gave 25.5 ms. The attention kernel is still 5 ms above its
arithmetic (17.2 ms with loads only against 12.7 ms with the loads removed).

In the dev worker (same 2.3 s clip): steps 3.40 to 3.43 s (C++ 3.35 s), later clips 48.0
to 48.2 s (51.3 s before; C++ sampler 47.7 s in a build that still read the decoder's
weights per clip). All seven modes produce clips and a Qwen image afterwards works (20.9
s); contact sheets of the text-only and two-reference clips show the expected scenes. The
clips' bytes differ from earlier builds because the order of the softmax sum changed.

### The video decoder as machine code (2026-11-16)

The decoder's tile was 0.26 s of Vulkan shaders: linear layers 151 ms at 57 TFLOPS,
attention about 120 ms, the small steps about 25 ms. Its linear layers are half-float
products, which the card's matrix instruction runs at half the int8 rate (120 TFLOPS in the
probe), so the same move as for the DiT applied.

**The half-float product is the int8 kernel.** A half-float step is 16 inputs in the same 32
bytes, so `xe2/int8.rs` only changes the multiply's operand types (`Shape::half`). Sums are
bit-identical to the `gemm16` shader's. Subnormal weights read as zero until the kernel
sets the float control register (`cr0` bits 7 and 10, then a `sync`), which the shaders get
from `DenormPreserve`.

| Product (1920 tokens) | Shader | Machine code |
|---|---|---|
| 16384 x 2048 | 42 to 78 TFLOPS | 83 to 108 |
| 2048 x 8192 | 40 to 55 | 58 to 94 |
| 6144 x 2048 | 47 to 67 | 85 to 117 |

One dispatch block for all 60 token tiles is fastest here; smaller blocks (8 to 40 tiles)
were 2 to 40% slower.

**A Vulkan submission costs more than the small kernels.** With only the products as machine
code the tile stayed at 0.263 s: linear time fell from 151 to 88 ms, and the shaders between
the products, now 290 separate submissions per tile, lost as much. Measured alone, one
submission and wait is 0.14 to 0.24 ms and one machine-code launch 0.015 ms; polling the
fence instead of sleeping on it changed nothing, so the time is in the driver's submission
path. Everything between two products therefore had to become machine code as well.

- `xe2/norm.rs`: one thread per token row, the row's 2048 values held in registers. It
  applies the residual update (`x += scale * update + shift`, the shift being scale times
  bias, precomputed), stores the row, sums its squares, and writes the normalized row as
  half floats straight into the next product's token layout. One kernel replaces the
  residual and norm shaders on both sides of a product: 0.10 ms against 0.28 ms, 147 dB
  (stream) and 103 dB (halves) from the shaders.
- `xe2/gate.rs`: the feed-forward's gated activation, one thread per token walking 8192
  outputs with two alternating sets of load buffers. `silu` uses the hardware's
  power-of-two and reciprocal, which makes it bit-identical to the shader: 0.37 ms against
  0.49 ms.
- Encoder additions: `math` (reciprocal, reciprocal square root, power of two; Xe2 has no
  division), `or`, the control register. Math instructions run in a pipe of their own, so
  a group that reads the other pipe's results waits on it as a whole (`A@1` on its first
  instruction); counting back within the group reads garbage.

**Attention reuses the DiT's kernel.** The decoder's heads are 64 channels and the kernel's
are 128, so `heads64s` writes queries, keys and values into the lower half and leaves the
rest zero, and `merge64s` reads the lower half back into token rows. The kernel takes its
reference score from the first key it meets, which for the decoder cost accuracy (69.7 dB
from the CPU reference, below the C++ engine's 70.9): weights far below the first key's
lost their precision as half floats. The decoder's scores are bounded (unit-length queries
and keys, so a scaled score stays within eight), so `Shape::bounded` starts the reference
at zero: 79.1 dB. The shader path is 83.7 dB; the rest of the gap is probably the kernel's
polynomial for the fraction of the exponent (3e-5).

**Host side.** Each tile allocated and filled about 60 MB around the GPU work (30 ms). It
now reuses mapped staging memory for the stream and the pixels, uploads the rotary table
once per tile shape, and adds the output bias while unpatchifying.

| One 16x16x7 tile | Shaders | Machine code |
|---|---|---|
| Linear layers | 151 ms | 89 ms (104 TFLOPS) |
| Attention | about 120 ms | 62 ms: operands 26 (shader), kernel 27, merge 9 (shader) |
| Residual, norm, gate | about 25 ms | 21 ms |
| Whole tile | 0.256 s, 83.7 dB from the CPU reference | 0.188 s, 79.1 dB |
| 832x544, 73 frames (48 tiles) | 13.1 s | 9.46 s, 69.3 dB from the C++ frames |

The shader path remains for cards without Level Zero and is tested beside the machine-code
path (`gpu_tile_matches_cpu_reference_and_reports_speed` runs both). With machine code the
weights are stored in the product's stream layout, which takes about a second longer to
read (2.3 s against 1.4 s, once per worker).

In the dev worker (2.3 s clip, 36 tiles, Rust DiT build): decode 10.2 s to 7.1 s; later
clips 48.1 s to 44.8 to 45.1 s (C++ sampler with the shader decoder 47.7 s); the first clip
of a worker 50.2 s. All seven modes produce clips and an image job afterwards works (21.9
s); the text-only and two-reference contact sheets show the expected scenes. Clip bytes
differ from earlier builds (attention weights and the residual's rounding changed); the
same request still gives the same bytes. The DiT's own checks are unchanged to the last
digit, since its kernels only gained options it does not use.

### The rest of the decoder as machine code (2026-11-17)

Three more steps took the tile from 0.188 s to 0.135 s:

- **Attention operands and merge.** `xe2/heads.rs` splits the query/key/value projection
  for the attention kernel: one thread per 16 tokens of a head, one token per lane, so
  gather loads (one address per lane, four channels each) turn token rows into registers
  that each hold one channel of 16 tokens, which is the kernel's query and value layout.
  Bias, unit RMS and the rotary turn happen in registers; the rotary table sits behind the
  key/value stream. Keys need each token's channels side by side, so they leave through
  scatter stores. `xe2/merge.rs` divides the kernel's output by the weight sums and
  scatters it into the next product's token layout. Both are 103 to 111 dB from a CPU
  reference. Scattering one channel (16 bits) per lane per store was slow (merge 0.28 ms);
  packing four channels into a 64-bit lane first cut the stores to a quarter (merge 0.055
  ms, split 0.45 to 0.23 ms). A tile is now one Vulkan submission, for the final norm.
- **A 64-channel attention.** `attend::Shape::narrow` skips the empty upper half of the
  kernel's 128-channel heads in scores, mixing, loads and stores.
- **The gate inside the product.** `int8::Shape::gated` ends the first feed-forward
  product in the activation instead of storing its sums: weight rows (and bias) are
  stored so that each thread's 64 rows are 32 gates and the 32 values they multiply, and
  the thread's 32 tokens are one tile of the next product's layout, so the result leaves as
  whole kilobytes. Bit-identical to the product shader followed by the gate shader, 0.97
  ms against 2 to 2.8 ms, and the 126 MB of sums per layer are never written.

Encoder additions: per-lane sends (`send16`), `shr`, `and`, stride-four destinations and
`lane_addresses`. A 64-bit instruction's source must sit where its lane is, so the 32-bit
lane offsets are built in every second dword before widening; built packed, they produced
garbage addresses (the reference disassembler flags the form).

| One 16x16x7 tile | 2026-11-16 | Now |
|---|---|---|
| Linear layers | 89 ms | 86 ms, gate included (110 to 125 TFLOPS for QKV and the gated product) |
| Attention with split and merge | 62 ms (two shaders, 72 submissions) | 26 ms |
| Residual and norm | 21 ms with the gate | 8 ms |
| Whole tile | 0.188 s | 0.135 s, 79.1 dB from the CPU reference |
| 832x544, 73 frames (48 tiles) | 9.46 s | 6.90 s, 69.3 dB from the C++ frames |

Reordering the gated product's weight rows adds about half a second to reading the
decoder's weights (2.9 s, once per worker, on the background thread).

In the dev worker (2.3 s clip, 36 tiles, Rust DiT build): decode 7.1 s to 5.24 s; later
clips 44.8 to 45.1 s to 42.9 to 43.1 s; the first clip of a worker 49.9 s (its weight read
varies with the page cache). All seven modes produce clips, same-seed requests repeat their
bytes, and an image job afterwards works (21.1 s); the text-only and two-reference contact
sheets show the expected scenes.

What is left: the products are near the instruction's rate. The hardware's power of two in
place of the attention kernel's polynomial may close the accuracy gap to the shader path
(79.1 against 83.7 dB).

## Text encoder

MiniMax H3 conditions on the hidden states after the 50th (last kept) layer of Qwen3-VL
32B's language model, without its final norm: width 5120, 64 query and 8 key/value heads
of 128, a gated feed-forward of 25600, INT8 ConvRot weights like the DiT's (24.4 GB). A
prompt with a first frame is about 470 tokens: 442 come from the vision tower's view of the
image, which also adds "deep-stack" rows to those tokens after the first three layers.
Rotary positions are interleaved over time, height and width (ggml's IMROPE): text tokens
have one position on all three axes, image tokens their row and column.

### Where the C++ engine spends its time (2026-11-18)

Sampling the worker's GPU counters every 50 ms (`drm-total-vram0` and engine cycles in
`/proc/<pid>/fdinfo`) during a warm clip showed conditioning as 1.8 s of VAE encoding of
the first frame, then 3.3 s of copy-engine traffic while video memory grew by 25.5 GB (the
text encoder's weights, 7.7 GB/s), then about 0.25 s of compute. The DiT is copied in after
the text encoder leaves (2.07 s), because the two do not fit together.

The B70 sits behind a PCIe 3.0 x16 root port (`00:03.1`, 8 GT/s): copies to the card top
out at 10.5 to 11.2 GB/s. So the text encoder costs at least 2.2 s per clip however it is
computed, and holding it in video memory would push out the DiT.

### The GPU form

`text_gpu.rs` keeps every layer's matrices in host staging memory (read once per worker, on
its own thread) and per prompt copies each layer into one 488 MB device window and runs
it: about 44 ms of copying and 6 ms of compute per layer. The kernels are the DiT's
quantizer and integer product shaders (`rotate8`, `gemm8`, any width) and six small
shaders: the norm, query/key/value preparation (dequantize, per-head RMS, rotary), causal
grouped attention with an online softmax, the residual, the gate and the deep-stack rows.
The device holds the window, the prompt's activations (about 0.25 GB at 470 tokens) and
every layer's scales and norm weights (16 MB). `text_encoder.rs` is the CPU reference.

The engine patch adds `sd_kuno_set_text_encoder`: a MiniMax H3 prompt's token ids, each
image's embeddings, patch grid and deep-stack rows are offered to Rust before the engine
runs its language model; since its text layers are then never evaluated, their weights
are never copied to the card.

Tried and dropped:

- Reading each layer from the model file per prompt into two landing areas while the
  previous layer runs. Alone, a read takes 49 ms and a copy 44 ms, but together a layer
  took 69 ms: the page-cache read and the bus copy share the host's memory bandwidth
  (dual-channel DDR4). More reader threads did not help (4 and 8 the same).
- Importing the model file's mapped pages as Vulkan host memory, so the card reads the
  page cache in place: the driver rejects a read-only file mapping.
- The CPU for short prompts: an AVX2 INT8 product reached 1.6 TOPS on 16 cores, so a
  470-token prompt would take about 14 s.

### Accuracy

Text tokens are 37 to 38 dB from the C++ engine's output (the first token, which carries
most of the energy, 66 dB), as close as the GPU form is to the CPU reference (37 dB; 46 dB
overall on a made-up prompt). Image tokens are not reproducible by any implementation:
the GPU form is 9.5 dB from the engine and 9.6 dB from the CPU reference, and the CPU
reference is 4.4 dB from the engine. Their final states amplify rounding differences
across the 50 INT8 layers. Variants that would reveal a semantic error all moved further
from the engine: sequential positions for image tokens 1.3 dB, non-interleaved axes
-3.1 dB, no deep-stack rows -4.5 dB, half-precision attention operands 4.4 dB.

What matters is the clip: with the same seed, the clip made with the Rust text encoder is
29.7 dB PSNR from the one made with the engine's (the Rust and C++ samplers on the same
request are 28 to 34 dB apart), and contact sheets show the same scene and motion frame
for frame.

### In the worker

| 2.3 s clip, dev worker | Before | Now |
|---|---|---|
| Text encoder | 3.55 s (C++) | 2.51 to 2.57 s |
| DiT copied to the card | 2.07 s every clip | first clip only |
| Later clips | 42.9 to 43.1 s | 40.2 to 40.3 s (C++ sampler 47.7 s) |
| First clip of a worker | 49.9 s | 56.4 s |

- With the text encoder streamed, the DiT (22.5 GB with the refiner) stays in video memory
  between clips of the same model (`RESIDENT_DIT` in `native.rs`). The decoder still fits
  beside it; if it does not, the DiT is freed and the decoder placed again. Ending a clip
  drops the DiT's kept text features and its work buffers, so nothing request-derived stays.
  An image job, or a clip with the other DiT, frees it first.
- The first clip is slower: the text encoder's 24.4 GB read and the DiT's 21 GB read share
  the host's memory bandwidth (the DiT's read went from about 10 s to 16.5 s), and
  conditioning waits for the text encoder's.
- The DiT's host copy goes once the DiT is in video memory (`Host::release`, 2026-11-19), to
  leave RAM for language models: the worker holds about 29 GiB between clips (text encoder
  24.4 GB, decoder 5 GB; system MemAvailable 103 GiB stopped, 74 GiB idle) where it held about
  50 GB. Later clips of the same model need no read. A clip after an image job or the other
  model reads the DiT from the file again: 54.1 s against 40.3 s (16 s read, partly hidden
  behind conditioning). Before this, the worker held 26 GB plus 27 GB of the text encoder's
  file pages (evictable) under the C++ encoder.
- Same-seed clips repeat their bytes; the DiT kept between clips does not change them.

## Video encoder (2026-11-21)

Start, end and reference frames are encoded by the VAE's encoder: a stack of causal 3D
convolutions with group norms (128 to 1024 channels, 16x spatial downsampling). For one frame
the causal padding puts two zero frames first, so every 3x3x3 kernel reduces to its last time
slice, a 3x3 convolution. `video_encoder.rs` holds the layout and the exact form (f64);
`encoder_gpu.rs` runs it on the GPU: activations channel-major f32, each convolution unfolded
into half-float rows (`ecol`, with the group norm and SiLU of a block fused in) and multiplied
in `gemm16` with output channels in the token slot, so the product lands channel-major;
`estats`/`estatsend` compute the group statistics (two passes per slice, Chan's merge) and
`ebias` adds bias and residual. A frame is one submission.

The C++ engine encodes in 256-pixel tiles, and each tile normalizes with its own statistics,
which is not the model: on a real 832x544 start frame its tiled result is 6.8 dB from its own
whole-frame encode. The Rust encoder is 64.6 dB from the C++ whole-frame encode and 76 to
78 dB from the exact form on synthetic frames, so it is the faithful one. The clips barely
notice: the first frame of an image-to-video clip is 27.43 dB from the input image with the
Rust encoder against 27.67 dB with the C++ one (within noise; the decoder and AV1 bound it),
and the clips show the same scene and motion (24.6 dB apart on average).

| 2.3 s image-to-video clip, dev worker | Before | Now |
|---|---|---|
| Frame encode | about 1.8 s (C++) | 106 to 128 ms; 1 to 2 s once per worker to place 0.4 GB of weights |
| Warm clip | 40.2 to 40.3 s | 38.6 s |

The encoder's weights stay in video memory (0.4 GB) once a frame has needed them; text-only
clips never place them. Multi-frame encodes (video references) and sizes not a multiple of 16
fall back to the C++ engine.

## Vision tower (2026-11-22)

Images in a MiniMax H3 prompt (the start frame of image-to-video, reference pictures) are
read by Qwen3-VL's vision tower: a ViT of 27 pre-norm blocks (width 1152, 16 heads of 72,
LayerNorm, tanh GELU, 4304-wide feed-forward) over 16x16 patches, about 0.6 billion BF16
weights. Positions enter twice, as a learned 48x48 table interpolated bilinearly to the patch
grid and as a 2D rotary turn (rows on a head's first 18 channel pairs, columns on the next
18; pair i is channels i and i + 36). Patches are ordered in 2x2 blocks, so the final merger
(LayerNorm per patch, then two linear layers around an erf GELU) reads four neighbouring rows
as one 4608-wide row and emits one 5120-wide row per 32x32 pixels; three deep-stack mergers
do the same after blocks 8, 16 and 24, with the LayerNorm over the merged row, and the text
encoder adds their rows after its first three layers. An 832x544 frame is 1768 patches and
442 rows.

`vision.rs` holds the layout, the preprocessing (patch rows, interpolated positions, rotary
angles) and the exact form (f64). `vision_gpu.rs` keeps the patch stream f32 (patches padded
to 512, so merged rows fill whole 128-row tiles) and runs every linear layer in `gemm16` on
half-float rows. Five small kernels do the rest: `vnorm` (LayerNorm to half floats), `vbias`
(bias, then the residual, a GELU or the output), `vheads` (query/key/value split with the
rotary turn), `vsoft` (softmax over one head's scores) and `vgather` (attention output laid
out for the projection). Attention runs head by head as two products, scores and then
weighted values, so its scratch grows with the square of the patch count; images above 8192
patches fall back to the C++ engine. An image is one submission.

The engine patch adds `sd_kuno_set_vision_encoder`: the image as the engine resized it for
the tower (values in -1..1) is offered to Rust, which returns the four sets of rows.

Accuracy, from an in-worker comparison on a real 832x544 start frame (main rows, then the
three deep-stack sets):

| | Main | Deep-stack 1 | 2 | 3 |
|---|---|---|---|---|
| Rust, from exact | 44.1 dB | 57.7 dB | 52.2 dB | 49.1 dB |
| C++ engine, from exact | 30.4 dB | 43.8 dB | 40.1 dB | 30.4 dB |

The Rust tower is 14 dB closer to the model on every set; the two differ by exactly the
engine's own error. On synthetic images the GPU form is 53 to 68 dB from exact.

| 2.3 s image-to-video clip, dev worker | C++ tower | Rust tower |
|---|---|---|
| Tower | about 0.39 s | 150 ms (237 ms on the first clip) |
| Warm clip, first step at | 6.77 s | 6.55 s |
| Warm clip | 38.3 to 38.6 s | 38.4 s |
| First clip | 48.2 s | 48.6 s (within run-to-run noise of the weight reads) |

The tower's weights (1.2 GB as half floats) are placed on a thread of their own when a
request has images, beside the text encoder's and the DiT's reads, and stay in video memory.
Placing them goes straight from the file into upload memory (0.67 s). Run first and the text
encoder's 24 GB read waited for it: that cost the first clip 3 s.

## The pipeline in Rust (2026-11-22)

`native/pipeline.rs` is the engine's `generate_video` for the requests Kuno makes, so MiniMax
H3 video no longer calls the C++ engine (the worker frees the engine's context before a
clip, so an earlier image job's models leave video memory):

- Start and end frames, or reference pictures (resized nearest-neighbour to at most the
  clip's area, multiples of 32), go through the Rust frame encoder, the latent mean and
  deviation, and the engine's conditioning noise (`latent * 0.999 + philox * 0.001`, a
  fresh Philox generator on the clip's seed for each).
- The prompt is `<Picture n>: <|vision_start|>`, one `<|image_pad|>` per vision row,
  `<|vision_end|>` for each picture, then the text, tokenized by `engine/tokenizer.rs`
  (Qwen2's BPE, id for id with the engine) and encoded by the Rust vision tower and text
  encoder. Text tokens are tagged 1, picture rows and their two markers 0.
- The packed latent (24 video channels, the audio latent in further channels) starts as
  PyTorch-compatible MT19937 noise on the seed (`engine/rng.rs`; the first sigma is 1).
  Sigmas follow the discrete flow schedule with shift 12, and each step is the engine's
  Euler update through its denoised estimate. Previews project the denoised middle frame
  to RGB with the engine's fixed matrix; they also drive the sampling progress.
- After the loop the audio channels leave the video schedule's scale (times 3/12), the
  audio latent goes to the Rust audio decoder on CPU threads, and the video latent to the
  Rust decoder; frames are converted to 8 bits as the engine does.

Step inputs match the engine's exactly (noise, schedule, text rows, tags, keyframes) and
its conditioning latents to 160 dB. The clips are not byte-identical to the hybrid path's:
given identical inputs, the DiT's output differs (25 dB at the first step) while the C++
engine's video context is live in the process. Each path is deterministic on its own (the
same seed gives the same clip, also after an image job in the same worker), and the
outputs are equally faithful (an image-to-video clip's first frame is 27.44 dB from the
input image against 27.43 dB). The cause was not isolated; the engine's video context no
longer exists in this path.

Checked: all seven video modes, same-seed clips, an image job after video, cancellation
(SIGTERM during sampling ends the request after the current step with `image_cancelled`).
A warm image-to-video clip takes 38.3 s, the first 48.0 to 49.3 s, as before.

## Qwen Image 2.1 in Rust (2026-11-23)

Qwen Image 2.1 is a single-stream DiT (32 blocks of width 4096, 32 heads, SwiGLU 12288,
INT8 ConvRot linears like MiniMax H3's) over the prompt's tokens, each reference's latent
in place of its vision tokens, and the target latent. Its LayerNorms have no weights; a
block scales them by `1 + m` and gates its residual adds by `tanh(m)`. The prefix (prompt
and references) is modulated with the time-zero row and never attends to the target, and
text runs attend causally, so the prefix's keys and values do not depend on the step: the
GPU form runs the prefix once per request, run by run through all blocks (text runs with a
small causal shader, images with the machine-code attention), keeps each block's keys and
values, and runs only the target's tokens per step. Their attention reads the target's own
keys first and the kept prefix after them (the softmax does not depend on the keys' order,
and the target's keys stay aligned for the stream writers); the attention kernel takes a
query count separate from its key stride for this.

The VAE is a Wan 2.2-style RGBA image VAE (latent 64 channels at 1/16). For one image its
temporal paths reduce to their first-chunk forms: the decoder's up blocks add a channel-
repeating depth-to-space shortcut, the encoder's down blocks an average-pooling shortcut
whose grouping includes the zero frame padded in front of a single image (half its channels
read zeros). Both run as unfolded convolutions in `gemm16` with the RMS norm, SiLU, nearest
upsampling and strided reads folded into the unfold.

The text encoder and vision tower are MiniMax H3's code with their sizes read from the
file; the 8B file's INT8 token embeddings are scaled and un-rotated in the engine's order.
The engine parses A1111-style emphasis in the user's text (and tokenizes each piece
separately), so `prompt.rs` ports that byte for byte.

Measured (B70, 1184x1776, 50 steps unless stated):

| | C++ engine | Rust |
|---|---|---|
| DiT step | about 1.14 s | 1.23 s |
| VAE decode | 2.40 s | 1.58 s |
| 512x512, 6 steps, end to end | | 1.98 s, 40.7 dB from the engine's image |
| 512x512 edit, CFG 4, 8 steps | | 5.3 s, same picture (23.4 dB) |
| Worker, generate 1184x1776, 50 steps (warm) | 61.5 s, 1.165 s per step | 65.4 s, 1.247 s per step; 36.4 dB from the engine's image |
| Worker, edit 1248x832, 25 steps, CFG 4 (warm) | 37.8 s, 1.20 s per step | 40.6 s, 1.39 s per step; 30.1 dB |

In the worker (`native/qwen.rs`, `rust-dit` builds) Qwen Image 2.1 requests no longer call
the C++ engine: it releases its own context first, the models stay loaded between image
requests, and video requests or the other image model free them. Previews are the engine's
latent projection of each step's denoised latent; SIGTERM cancels after the current step.
These first numbers had the Rust DiT 7% (generate) to 16% (edit) per step behind the engine;
the next section closes that.

### Qwen DiT speed work (2026-11-23)

Step at 1184x1776 (`step_speed_at_benchmark_size`, best of four, includes the CPU's time
layers) after each change, all on the same drivers:

| Change | Step |
|---|---|
| Starting point (Vulkan glue, one submission per block) | 1.23 s |
| Time layers for all steps at once, residual + next norm fused | 1.19 s |
| Whole step as one Level Zero command list: glue in OpenCL C (`xe2/qwen.cl`, built by `xe2/qwen-glue.sh` into the native `xe2/qwen.bin`), `LevelZero::Batch` | 1.15 s |
| Feed-forward activation in the INT8 product's epilogue (`int8::Shape::gated8`) | 1.133 s |
| Product dispatch blocks balanced (about 128 token tiles each) | 1.127 s |
| Attention query tiles in groups of 8 threads per core (`attend::Shape::group`) | 1.086 s |
| Hidden quantizer keeps its row in local memory (no spill) | 1.078 s |
| Feed-forward activation stored as bf16 | 1.063 s |
| Query/key/value product scales its sums and stores bf16 (`int8::Shape::scaled8`) | 1.048 s |

The C++ engine's step is about 1.165 s in the worker. Captured steps stay 36.9 to 44.6 dB
from the engine (the mean moved by less than half a decibel over the series).

In the worker (`image-bench.py`, same requests as the table above; first run includes
opening the models):

| | C++ engine | Rust (d0b26dbc) |
|---|---|---|
| Generate 1184x1776, 50 steps, warm | 61.5 s, 1.165 s per step | 55.4 s, 1.045 s per step; 38.5 dB from the engine's image |
| Generate, first request | 67.4 s | 60.9 s |
| Edit 1248x832 (1248x832 reference), 25 steps, CFG 4, warm | 37.8 s, 1.20 s per step | 33.7 s, 1.11 s per step; 28.9 dB, same picture |

The edit's distance from the engine's image moves by a few decibels with any change of
rounding (CFG 4 amplifies it over 25 steps); side by side the pictures match. The final PNG
is now written with fast compression (lossless; 0.6 s -> 0.02 s, file about 13% larger), not
yet in these numbers.

What did not help:
- Chaining Vulkan and Level Zero with timeline semaphores (correct only with
  `zeCommandListAppendBarrier`, then no faster); a native OpenCL module removed the
  cross-API hand-offs instead. Loading the glue as SPIR-V at run time failed (`gpu_kernel`);
  the ocloc-built native binary loads.
- Attention start chunks spread further than one chunk per tile (`tile >> s`), a group's
  threads starting on the same chunk, or packed start chunks: all slower. The kernel is
  bound by loads (10.5 ms loads only against 6.8 ms without loads at 32 heads x 8239 keys);
  groups keep neighbouring tiles on one core so they reuse each other's lines.
- bf16 outputs for the attention-out and feed-forward-out products as well: 0.8% faster,
  but the residual increments lost about 3 dB on one captured step; kept as int32 sums.

Installed 2026-11-23 (gateway bff4388d, runtime worker 61064827). After it, over HTTP:
1024x1024 8-step image 5.0 s warm, same-seed outputs byte-identical.

Where a step's time goes (`step_profile`: the real step's command list with the GPU clock
stamped after every launch; 1184x1776, 8214 tokens, 32 blocks, 0.994 s):

| Kernel | Share | Per block | Rate |
|---|---|---|---|
| Attention | 35% | 10.9 ms | ~102 TFLOPS half float (~56% of peak) |
| Products FC1 / FC2 / QKV / OUT | 51% | 7.2 / 4.0 / 3.6 / 1.2 ms | 228-237 TOPS (B70 peak 367) |
| Residual + norm + quantize (x2) | 5.7% | 0.90 ms | ~485 GB/s |
| Hidden quantize, merge, queries/keys, values | ~7% | 0.75 / 0.59 / 0.30 x2 / 0.31 ms | 280-450 GB/s |

The products alone with zero inputs run 5-9% faster than in the step (epilogues, and real data
costs clock); best blockings reach 233-263 TOPS. At the edit size (4056 target tokens) the
products cost per token what they cost at 8214, so running both CFG branches as one batch would
not speed them up.

Glue at bandwidth (bit-identical; `*_matches_reference` tests run the kept previous kernel
`<name>_ref` on random input, compare bytes and time both):
- Hidden quantizer: rotated values stay in registers, local memory only for each sub-group's
  (bank-padded) transposes: 0.97 -> 0.76 ms, 48 KB -> 17 KB local memory per group.
- Queries/keys: a group per head and 16-token tile; results meet in local memory and leave
  as one contiguous 4 KB block instead of scattered 4-byte stores: queries 0.96 -> 0.31 ms,
  keys 0.52 -> 0.30 ms.
- Step 1.008 -> 0.994 s in `step_profile`.

### Fast attention: int8 queries and keys (2026-11-23)

Requests with `"attention": "fast"` (worker ABI 5) score attention with signed-byte
queries and keys, SageAttention-style; values and mixing stay half floats. `q_qk8`
writes each token's 128 channels scaled to its own largest magnitude, with the scale
beside the tile (queries) or chunk (keys); `q_prefix8` does the same to the stored
prompt/reference keys as it copies them, so a chunk shared by target and prefix keys
needs no common scale. The kernel (`attend::Shape::int8`) converts the integer scores and
multiplies by the key's scale (the query's is folded into the softmax constant).

The f16 kernel is limited by its loads (loads alone take 92% of its time; without the
score products it is only 4% faster). In order:

| Change | Attention, 24 heads x 8214 tokens (alone) |
|---|---|
| Half floats (standard) | 7.3 ms |
| int8 queries/keys (key bytes halved) | 6.66 ms |
| + three more load buffers (values requested a phase ahead) | 6.13 ms |
| + conversions four at a time | 5.69 ms |
| + four running maxima (bit-identical in both forms) | 5.50 ms |
| + four running sums | 5.40 ms |
| + math pipe `2^x` instead of the polynomial | 4.61 ms |

In the step (`step_profile`, 1184x1776): attention 10.7 -> 6.8 ms per block, step 0.991 ->
0.867 s (-12.5%). Against the C++ captures fast steps score 35.9-43.8 dB (standard
37.7-44.2), 34.9-44.7 dB from standard. Random operands: 47.7 dB (spread 1.5) and 39.5 dB
(spread 3) against an exact softmax, where half floats reach 70-73 dB.

Through the worker (benchmark recipes, warm): generate 53.8 -> 47.6 s, edit 32.8 -> 29.0 s
(both -11.6%); pixels differ from standard by 1.0-1.2 on average (of 255). A neutral
1184x1776 prompt: mean 0.45, 0.1% of pixels more than 8 apart, no visible difference.

Opening the models from page cache (`open_costs`): DiT 2.3 s (layers load on three threads,
so copies overlap the rearranging; 3.3 s before), VAE about 1.1 s (parallel rearranging;
2.0 s), text encoder 1.3 s, vision tower 0.7 s. The worker disables Mesa's shader cache,
which adds about 1 s.

## Qwen Image 2512 in Rust (2026-11-23)

Qwen Image 2512 is not a variant of 2.1: 60 blocks of two streams (text and image) with
their own weights, 3072 wide, plain GELU MLPs, a modulation layer per block and stream, and
the text attending to the image in every block (so nothing is computed once per prompt).
Its file is FP8 (e4m3) without scales: 13.6 G parameters of block linears and 6.8 G of
modulation.

- `qwen2512_dit.rs`: the CPU reference from the engine's graph (`qwen_image.hpp`), 34 to 47
  dB from captured C++ calls (the engine computes in half floats); with the block linears
  through INT8 as on the GPU, 34 to 44 dB.
- `qwen2512_gpu.rs`, `xe2/qwen2512.cl`: one command list per pass. Block linears become
  the machine-code INT8 products at load (widened, rotated in groups of 256, one scale per
  row; 21 s for the whole model); the FP8 modulation rows stay on the device and are
  evaluated exactly once per step (`k_mod`) for both CFG passes. Queries sit in one slot
  stream (image, then text after the image's padded rows); keys and values in one key
  stream with the text right after the image's live keys, so `attend.rs` runs unchanged.
- Fast attention: block 0's query and key norm weights have a 57x outlier channel (blocks
  52 and 58 about 8x). One channel then dominates the scores and per-token int8 queries lose
  the rest (25 dB; smoothing the keys does not help, 22 dB). Blocks whose norm weights peak
  above three times their median keep half-float attention; the rest use int8: 38.6 to
  46.6 dB from the captures, as standard does.
- Text encoder: Qwen2.5-VL 7B through the streamed GPU encoder (`TextEncoder::qwen25`):
  FP8 matrices times their scale, rotated and rounded to INT8 at load; query, key and value
  biases (`tqkv.comp` flag 2), no query/key norms (flag 1 off), plain rotary with theta 1e6,
  and the final norm (the engine takes the normed last layer: without it -2 dB). The
  template's first 34 tokens are dropped; captured row counts (22, 10) match.
- VAE: the Wan 2.1 decoder through the 2.1 GPU decoder (`QwenVae::open_wan21`): 16
  latent channels at 1/8, 384/192/96 channels, three upsamplings without the depth-to-space
  shortcut, RGB out; the causal 3x3x3 kernels see a single image only through their last
  time slice.
- Schedule: the engine's simple scheduler, `index = 999 - trunc(i * 1000 / n)`,
  `sigma = shift t / (1 + (shift - 1) t)` with `t = (index + 1) / 1000`; Euler on the flow
  output, CFG on the outputs.

The worker runs 2512 itself from ABI 6 (`QWEN2512_WORKER_ABI`): the gateway sends 2512 to
the managed runtime when its verified worker has that ABI, and to its own engine otherwise.

## Qwen Image 2512 from the publisher's weights (2026-11-24)

Kuno no longer depends on ComfyUI's repackaging for 2512. The `qwen-image-2512-int8`
collection downloads `Qwen/Qwen-Image-2512` at a pinned commit (the publisher's BF16
files, 57.7 GB in 14 pieces) and converts each piece as it arrives (`engine/convert.rs`):
block linears and modulation layers to INT8 in the rotated basis with a scale per row,
everything else as published, the VAE's diffusers names renamed to the Wan layout, the
unused `lm_head` dropped. One input file gives one output file, the original is removed
once its output matches its pinned checksum, and the result is 28.6 GB. The conversion is
deterministic, so its outputs are pinned like downloads: anyone converting the official
files gets the same bytes.

The installer (`collections::fetch`) overlaps downloads and conversions: a download starts
only when the disk holds it, its output and every pending output (plus 512 MiB); otherwise
it waits for a conversion to free its input. With ample space downloads run ahead; with
little it alternates. Converting all 14 pieces takes about two minutes (6 to 13 s each,
about 1.3 GB of memory); the download dominates.

Every loader takes all three forms: the INT8 conversion directly (DiT 11.8 s, text encoder
1.1 s), ComfyUI's FP8 files, and the publisher's BF16 directories (rounded at load). A
directory of `*.safetensors` reads as one model; ComfyUI's name prefix is optional.

Against the exact model (the BF16 weights computed in f32 on the CPU), the FP8 file itself
is the largest error:

| Path | DiT (4 captured calls) | Text encoder (2 prompts) |
|---|---|---|
| C++ engine, FP8 file | 23.2 to 26.2 dB | 24.9 to 27.4 dB |
| Rust GPU, FP8 file | 23.0 to 26.1 dB | 23.7 to 25.8 dB |
| Rust GPU, INT8 conversion | 34.6 to 39.1 dB (fast 34.7 to 38.5) | 30.6 to 32.4 dB |

Through the worker at 832x1248, 20 steps with CFG: the first request 39.2 s (81 s when the
FP8 file is converted at load), then 23.0 s (21.1 s fast). The image differs from the
FP8 paths' by about 10 of 255 on average (a slightly different composition of the same
scene), as the FP8 model differs from the model it was made from.

## The worker without C++ (2026-11-23)

With 2512 in Rust, the worker's C++ engine only held a session object and the SIGTERM flag.
`--features rust-image` builds the worker without it: `native/rust_shim.rs` provides the six
`kuno_*` entry points (an empty session, the cancellation flag set by a Rust SIGTERM/SIGINT
handler, and the C++ generate entries answering "unsupported", which the Rust paths never
call). The binary is 22 MB (the C++ worker 333 MB) and links only libc, libm and libgcc;
Vulkan and Level Zero come from the system loaders at run time, so the runtime directory
needs no oneAPI libraries (`make-image-candidate-v7.py`: worker, license, manifest). It
builds in seconds instead of twelve minutes.

Qwen 2.1 images from this worker differ slightly from the C++-linked worker's for the same
request (0.8 of 255 on average, the same picture; still identical run to run). The Rust code
is the same: in the C++-linked worker Intel's `libimf` is loaded ahead of glibc's `libm`, so
Rust's `expf`, `sinf`, `cosf`, `powf` and `tanhf` (rotary tables, time embeddings, small CPU
layers) bind to Intel's versions, which round differently in the last bits. Preloading
`libimf` into the Rust-only worker reproduces the old images bit for bit.

## Any Vulkan GPU: the RTX 3060 (2026-11-24)

Every image model now also runs on GPUs other than the B70 through portable Vulkan kernels,
chosen per device (`Gpu::shapes`, `narrow`, `subgroup`, `gemm8`, `gemm16`, `matrix2`). On
Intel the kernels and numbers are unchanged; elsewhere:

- **Products.** `gemm8nv`/`gemm16nv`: 32-lane subgroups, 16 x 16 matrices, 64-column steps
  through shared rows padded against bank conflicts, the next step fetched into registers.
  INT8 59-62 TOPS on the 3060 (exact).
- **Attention.** `fattend16nv` on `VK_NV_cooperative_matrix2` workgroup-scope matrices:
  scores stay in registers (row reductions, per-element exponentials and masks, a
  conversion to the value product's half-precision operand), 18.3 TFLOPS at 24 heads x 4078
  tokens, 84 dB from an exact softmax. `fattend16` (subgroup scope, shared memory) is the
  fallback for GPUs without the extension (5.7 TFLOPS). Both take causal prompt runs.
- **Glue.** The 16-lane shaders work on groups of 16 invocations with clustered reductions
  (bit-identical on 16-lane subgroups); `tattend32` for the text encoder.
- **Qwen 2.1 DiT** (`qwen_vk.rs`): 38.3-45.8 dB from the captured steps (B70: 37.7-44.2).
- **Qwen 2512 DiT** (`qwen2512_vk.rs`): 44-50 dB from the Xe2 form on the same INT8 file.
  The modulation layers (a third of the model) run on the CPU from host memory; blocks stay
  on the device while the budget leaves room for a CFG step (35 of 60 on 12 GB), the rest are
  copied into one device slot per step, both CFG passes running through each block while it
  is there.
- Text encoder 38.6 dB, vision tower 37-68 dB, VAEs as on the B70.

On a 12 GB card the pipeline also: keeps a prefix's keys and values in host memory when
the steps' buffers would not fit beside them (each block then copies them over the bus);
shares one set of step buffers between CFG's prompts; decodes in strips of latent rows when
the whole image's VAE buffers do not fit (each strip with a halo of every convolution's
reach after the attention, so the pixels are identical: no seam); and frees the DiT for the
decode only when strips would be mostly overlap, loading it again on the next request.

Through the gateway's own worker (default build) on the RTX 3060 (PCIe x4 slot):

| Request | Time (warm) | Per step | Reference |
| --- | ---: | ---: | --- |
| Qwen 2.1 1184x1776, 20 steps, CFG 4 | 209 s | 9.0 s | ComfyUI 193 s |
| Qwen 2512 832x1248, 20 steps, CFG 5 | 137 s | 6.5 s | B70 23 s |
| Qwen 2.1 1248x832 edit, same-size reference, CFG 4 | about 170 s (25 steps, estimated) | 5.85 s | ComfyUI 109 s, C++ CUDA 283 s |

Fast attention (int8 queries and keys) is not implemented on Vulkan yet; the embedded engine
advertises standard attention only. Known speed work: copying streamed 2512 blocks and
host-memory prefixes while the previous block computes (a transfer queue), and the
products' efficiency at large token counts (44-56 TOPS in the step).

### The gateway's engine

`apps/kuno` builds with `rust-image` by default: the gateway executable is its own image
worker with the Rust engine on any GPU (22.6 MB), and `native-image` (the C++ engine) is
exclusive with it. The engine picks the Xe2 DiTs on Intel GPUs that run them and the Vulkan
ones elsewhere; the Vulkan DiTs refuse devices without 16 x 16 x 16 half-precision matrices.

## Running the comparisons

GPU tests (matrix kernel, tile, whole clip; they need the Intel GPU with enough free memory):

```
KUNO_VIDEO_VAE=<models>/minimax_h3_video_vae_fp16.safetensors \
KUNO_VIDEO_GOLDEN=<tile dir> KUNO_VIDEO_CLIP=<dir with clip-latent.f32, clip-pixels.f32, clip-shape.txt> \
cargo test --release -p kuno-image gpu -- --include-ignored --nocapture
```

Video decoder tile on CPU (the exact decode takes about 4 minutes on 16 cores):

```
KUNO_VIDEO_VAE=<models>/minimax_h3_video_vae_fp16.safetensors \
KUNO_VIDEO_GOLDEN=<dir with tile-latent.f32, tile-pixels.f32, tile-shape.txt> \
cargo test --release -p kuno-image engine::video_vae -- --include-ignored --nocapture
```

DiT reference against a captured step, and the integer kernel:

```
KUNO_DIT=<models>/minimax_h3_fl2va_turbo8_int8_convrot.safetensors \
KUNO_DIT_GOLDEN=<dir>/step0 \
cargo test --release -p kuno-image engine::dit -- --include-ignored --nocapture
cargo test --release -p kuno-image gemm8 -- --include-ignored --nocapture
```

GPU DiT (the whole-network tests need about 25 GB of free video memory):

```
KUNO_DIT=<model> KUNO_DIT_GOLDEN=<dir>/step0 \
cargo test --release -p kuno-image engine::dit_gpu -- --include-ignored --nocapture --test-threads 1
KUNO_DIT=<model> KUNO_DIT_STEPS=<dir with step0.. of one clip> KUNO_VIDEO_VAE=<decoder> \
cargo test --release -p kuno-image sampling_follows -- --include-ignored --nocapture
```

Vision tower against the exact form (two small images, then the time of an 832x544 frame):

```
KUNO_TEXT_ENCODER=<models>/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
cargo test --release -p kuno-image engine::vision -- --include-ignored --nocapture
```

Text encoder (the reference test reads 24 GB and takes about two minutes on the CPU):

```
KUNO_TEXT_ENCODER=<models>/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
cargo test --release -p kuno-image text_ -- --include-ignored --nocapture --test-threads 1
```

Video encoder (exact form on two small synthetic frames, then the GPU's time on 832x544):

```
KUNO_VIDEO_VAE=<models>/minimax_h3_video_vae_fp16.safetensors \
cargo test --release -p kuno-image gpu_encoder -- --include-ignored --nocapture
```

Audio decoder:

```
KUNO_AUDIO_VAE=<models>/minimax_h3_audio_vae_fp32.safetensors \
KUNO_AUDIO_GOLDEN=<dir with audio-latent.f32 and audio-wave.f32> \
cargo test --release -p kuno-image engine -- --include-ignored --nocapture
```

Qwen Image 2.1 (captures from the dev CLI with temporary dumps, `kuno-video-dev/qwen-run.sh`):

```
KUNO_QWEN_DIT=<models>/qwen_image_2.1_int8_convrot.safetensors KUNO_QWEN_GOLDEN=<step dir> \
cargo test --release -p kuno-image qwen_dit -- --include-ignored --nocapture
KUNO_QWEN_DIT=<model> KUNO_QWEN_GOLDEN=<step dir> \
cargo test --release -p kuno-image qwen_gpu -- --include-ignored --nocapture
KUNO_QWEN_VAE=<models>/qwen_image_2.1_vae_bf16.safetensors KUNO_QWEN_VAE_GOLDEN=<decode dir> \
KUNO_QWEN_ENCODE_GOLDEN=<encode dir> cargo test --release -p kuno-image qwen_vae -- --include-ignored --nocapture
KUNO_QWEN_MODELS=<models> KUNO_QWEN_REFERENCE=<engine PNG> KUNO_QWEN_EDIT=<dir> \
cargo test --release -p kuno-image qwen_image -- --include-ignored --nocapture --test-threads 1
```
