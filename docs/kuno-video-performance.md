# MiniMax H3 video on the B70 (development, 2026-11-06)

Status: development measurements with `sd-cli` built from the Kuno stable-diffusion.cpp
source and patches. Video is not yet exposed through the Kuno API or Vashti; no runtime
is installed for it. The SYCL changes also touch the shared image runtime (quantizer
coverage, head-64 oneDNN SDPA, concat, conv_transpose_1d), so the protected Qwen image
acceptance must be rerun before any runtime built from these patches is installed.

## Setup

- Weights (Comfy-Org/MiniMax-H3): `minimax_h3_fl2va_pruned_int8_convrot` DiT,
  `qwen3vl_32b_minimax_h3_int8_convrot` text encoder, `minimax_h3_video_vae_fp16`,
  `minimax_h3_audio_vae_fp32`, and the `minimax_h3_fl2v_turbo_8step_v1.0` LoRA.
- The turbo LoRA is merged offline into the INT8 ConvRot DiT with
  `apps/kuno/scripts/merge-lora-int8-convrot.py` (dequantize in the rotated basis, add the
  rotated delta, per-row absmax requantize with seeded stochastic rounding, as ComfyUI's
  `requantize_from_float` does; BF16 token-refiner weights use stochastic BF16 rounding).
  Round-to-nearest would discard most of the LoRA: its deltas are ~0.07% of the weights,
  far below one INT8 step. Runtime (`at_runtime`) LoRA cost ~2 s per step.
- Workload: image-to-video (first frame), 832x544, 8 steps, CFG 1, audio on, 24 fps.
- `GGML_SYCL_FA_ONEDNN_NO_MASK=1` (already set by Kuno's managed SYCL launcher) and
  flash attention for every model (`--fa`).

## Results (end to end, fresh process)

| Clip | First run | After all fixes |
|---|---:|---:|
| 2.3 s (56 frames) | 5:01 | 1:04 |
| 5.2 s (124 frames) | 4:19 (after the SDPA fix) | 2:07 |

2.3 s breakdown now: conditioning 11 s (mostly reading the 25 GB text encoder from disk in
a fresh process), sampling 34 s (8 x ~3.4 s plus DiT load), audio decode 0.9 s, video
decode 12.9 s. Previous 3060 baseline (user report): 10+ minutes.

## Changes (all verified bit-exact against the unfused graph unless noted)

- Unmasked oneDNN SDPA: generic FA ran at ~450 ms/call (~4 TFLOPS); oneDNN is 24 ms
  (~89 TFLOPS). Head dim 64 (video VAE) now also uses it; its difference to the native
  kernel (0.25-0.5% mean) is in the same range as the accepted head-128 path.
- ConvRot quantizer: the fast row kernel now covers any 256-multiple K up to 16384
  (MiniMax uses 5376/7168/14336); byte-identical to the legacy kernel.
- `conv_transpose_1d` visits only contributing inputs (audio VAE 17.7 s -> 1.6 s, 5 s clip).
- DiT fusions: partial RoPE; fc1 SwiGLU + quantize from the INT8 product; q/k/v attention
  inputs from the INT8 product (RMS norm, partial RoPE, K/V scale + F16, Q F16 shadow);
  segmented AdaLN modulation into the quantizer and segmented gated residuals (planned
  once per graph run, with overlap checks against nodes still executed in the window).
  DiT step 8.8 s -> 3.4 s (excluding LoRA merge gains).
- Video VAE: single-launch concat (one kernel per ne3 slice before), bias + SwiGLU fusion,
  and the VAE runner kept across temporal chunks (weights, workspace, graph cache).

Run-to-run output is not deterministic (two runs of the same binary differ by mean ~1.5
levels in early frames, ~9 in late frames), so end-to-end frame comparisons cannot
validate a change; every fusion is checked with the dev replay checks instead
(`KUNO_DEV_FUSE_CHECK`, `KUNO_DEV_QUANT_CHECK`, `KUNO_DEV_FA_CHECK`, `KUNO_DEV_CONVT_CHECK`
in the dev-check source).

## Remaining costs and ideas

- Attention: ~37% of a 2.3 s-clip step, ~55% for 5 s clips (quadratic). oneDNN micro-SDPA
  reaches ~89 of ~176 TFLOPS FP16; a custom XMX flash-attention kernel is the main lever.
- Video VAE steady tile ~0.24 s: remaining glue (qkv bias/split/RMS/K-V casts, output
  bias + layer scale + residual) and IM2COL_3D convolutions (oneDNN conv3d candidate).
- First VAE tile per process ~3 s and DiT/text-encoder loads are one-time costs that a
  warm Kuno worker would not repeat.
