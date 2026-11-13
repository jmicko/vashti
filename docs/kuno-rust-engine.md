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
| Video decoder (36-layer transformer) | 0.59 s per tile on the GPU | 0.27 s per tile on the GPU (Vulkan), closer to exact | Rust, in the worker (dev build) |
| DiT sampling (INT8, 8 or 4 steps) | about 3.4 s per step (66 ms per block) | whole network on the GPU, follows the C++ sampler; 107 ms per block | correct, 1.6 times too slow; C++ still runs it |
| Qwen3-VL text encoder (INT8) | conditioning phase | not started | |
| Pipeline glue (sampler, latents, keyframes) | `video.cpp` | not started | moves with the DiT |
| Clip encoding (AV1, FLAC, MP4) | n/a | Rust since the first video release | done |

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

Audio decoder:

```
KUNO_AUDIO_VAE=<models>/minimax_h3_audio_vae_fp32.safetensors \
KUNO_AUDIO_GOLDEN=<dir with audio-latent.f32 and audio-wave.f32> \
cargo test --release -p kuno-image engine -- --include-ignored --nocapture
```
