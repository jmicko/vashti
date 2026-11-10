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
| Video decoder (36-layer transformer) | 0.59 s per tile on the GPU, 36 tiles for a 2.3 s clip | one tile on CPU, checked | GPU kernels next |
| DiT sampling (INT8, 8 or 4 steps) | 2.7 to 3.9 s per step | not started | |
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
production path. On a tile captured from the C++ engine
(16x16x7 latents, 28 frames of 256x256) it is 130.8 dB from the exact double-precision
decode; the C++ engine, which computes in half precision, is 70.9 dB from exact, and the two
agree to 70.9 dB. The CPU form takes 32 s per tile, which is why this stage needs the GPU.

Still to port for this stage: the GPU kernels, the spatial tile plan and blending, the time
chunking with its 5-frame cross-fade, and the encoder.

## Running the comparisons

Video decoder tile (the exact decode takes about 4 minutes on 16 cores):

```
KUNO_VIDEO_VAE=<models>/minimax_h3_video_vae_fp16.safetensors \
KUNO_VIDEO_GOLDEN=<dir with tile-latent.f32, tile-pixels.f32, tile-shape.txt> \
cargo test --release -p kuno-image engine::video_vae -- --include-ignored --nocapture
```

Audio decoder:

```
KUNO_AUDIO_VAE=<models>/minimax_h3_audio_vae_fp32.safetensors \
KUNO_AUDIO_GOLDEN=<dir with audio-latent.f32 and audio-wave.f32> \
cargo test --release -p kuno-image engine -- --include-ignored --nocapture
```
