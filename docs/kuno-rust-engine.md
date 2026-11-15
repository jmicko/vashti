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
| DiT sampling (INT8, 8 or 4 steps) | about 3.4 s per step (66 ms per block) | whole network on the GPU, follows the C++ sampler; 85 ms per block with machine-code linear layers and attention | correct, 1.3 times too slow; C++ still runs it |
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

The kernel is bound by operand loading (about 60 G cache lines per second
against the 88 G the int8 kernel reaches); the weights cost 3 ms.

In the DiT (`Kernels::block`), three shader variants write the kernel's
layouts (`qk8s`, `value8s`) and gather its transposed output (`merge8s`).
Work buffers and the kernel are built for an exact token count. One block at
8611 tokens: 84.8 ms (attention 39.8, linear layers 29.1, the rest 16); a
step's blocks and heads 4.7 s (C++ about 3.4 s). `block_matches_reference`
gives 39 to 47 dB and the whole step still passes against the C++ engine.

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
