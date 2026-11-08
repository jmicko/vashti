# Kuno video generation (proposal, 2026-11-07)

Status: **proposed contract, backend in progress** on branch `claude/video-api`. Nothing
here is implemented in an installed Kuno yet. Vashti must not depend on it until the
backend receipt in the [coordination guide](kuno-vashti-coordination.md) says so.

The first workflow is MiniMax H3 image-to-video with generated audio, running on the
managed SYCL image runtime (stable-diffusion.cpp). The engine will later be rewritten in
Rust piece by piece behind this same API, checked against the stable-diffusion.cpp
output and speed. Development timings: [video performance](kuno-video-performance.md).

## Principles (unchanged from images)

- Requests, first frames, previews, audio and results are transient content. Kuno never
  logs or persists them; Vashti owns storage. Diagnostics stay content-free with fixed
  error codes.
- Video uses the image GPU, the same exclusive admission lease, the same worker privacy
  restrictions (no swap, Landlock, no core dumps), and the existing image-worker
  deadline. No new response deadline is added.
- One retained model context per worker. Switching between image and video workflows
  replaces it.

## Discovery

`GET /kuno/v1/capabilities` gains an additive `videos` object. Clients must ignore
unknown fields; older Kuno simply has no `videos` key (treat as unsupported).

```json
"videos": {
  "api_version": 1,
  "access": "inference",
  "state": "ready",
  "reason": null,
  "generate_path": "/kuno/v1/videos/generate",
  "response_formats": ["application/x-ndjson"],
  "workflows": [
    {
      "id": "minimax-h3-image-to-video/v1",
      "state": "ready",
      "reason": null,
      "first_frame": "required",
      "fps": 24,
      "frames": [22, 39, 56, 73, 90, 107, 124],
      "max_pixels": 452608,
      "dimension_multiple": 32,
      "min_dimension": 256,
      "max_dimension": 1280,
      "audio": true,
      "output_mime_type": "video/mp4"
    }
  ]
}
```

`state`/`reason` use the same values and codes as `images` (`disabled`, `unsupported`,
`unavailable`, `partial`, `ready`; `image_generation_not_configured`,
`image_models_unavailable`, `image_privacy_unavailable`, `image_runtime_unavailable`,
...). `frames` lists the accepted frame counts (MiniMax needs 17k+5 frames). The limits
are what has been tested on the B70; they may grow additively. `max_pixels` bounds
`width * height` (832x544 = 452608).

## Request

`POST /kuno/v1/videos/generate` on the public inference listener, keyless like images.

```json
{
  "workflow": "minimax-h3-image-to-video/v1",
  "prompt": "The scene comes to life with gentle natural motion.",
  "seed": "42",
  "width": 832,
  "height": 544,
  "frames": 124,
  "audio": true,
  "first_frame": "<base64 PNG, JPEG or WebP>"
}
```

- `seed`: decimal string, 0..9223372036854775807 (as images).
- `prompt`: 1..32768 bytes, no NUL. There is no negative prompt: the 8-step turbo recipe
  runs at CFG 1, where a negative prompt has no effect.
- `width`/`height`: multiples of 32 in 256..1280, `width * height <= max_pixels`.
- `frames`: one of the advertised counts; duration is `frames / 24` seconds.
- `audio` (default `true`): include the generated audio track. Audio is always generated
  jointly by the model; `false` only omits the track.
- `first_frame`: required; at most 16 MiB of base64, decoded at most 4096x4096. Kuno
  scales it to cover `width x height` and centre-crops (aspect preserved). Clients should
  pick dimensions close to the image's aspect ratio.
- Unknown fields are rejected (`deny_unknown_fields`). Total request body is bounded like
  images (64 MiB).

Fixed recipe (not request fields): 8 steps, CFG 1, the turbo 8-step LoRA merged into the
INT8 ConvRot DiT, CPU RNG, flash attention.

Validation errors before streaming use the existing non-2xx JSON error surface, with
codes `unsupported_video_workflow`, `video_prompt_invalid`, `image_seed_invalid`,
`video_dimensions_unsupported`, `video_frames_unsupported`, `video_first_frame_invalid`.

## Streamed response

HTTP 200, `application/x-ndjson`, `Cache-Control: no-store`, `X-Accel-Buffering: no`, as
for images. Events:

```json
{"type":"started"}
{"type":"progress","phase":"conditioning"}
{"type":"progress","phase":"sampling","step":1,"steps":8}
{"type":"preview","step":1,"steps":8,"width":52,"height":34,"png_base64":"<PNG>"}
{"type":"progress","phase":"decoding"}
{"type":"progress","phase":"encoding"}
{"type":"result","mime_type":"video/mp4","width":832,"height":544,"frames":124,"fps":24,"audio":true,"video_base64":"<MP4>"}
```

- `progress` phases are `conditioning`, `sampling` (with `step`/`steps`), `decoding` and
  `encoding`, in that order. They carry no content. Like previews, progress is latest-only
  for slow clients (intermediate steps may be skipped), but the final sampling step
  (`step == steps`) is always reported before `decoding`.
- `preview` is a small, coarse PNG of the middle latent frame after a sampling step (its
  size follows the latent grid, at most 512x512). Only the latest is held for slow clients.
- The terminal event is exactly one `result` or `error`, then EOF. EOF without one is an
  interrupted delivery, not success. Closing the response cancels generation.
- Result: MP4 (ISO BMFF) with an AV1 video track and, when `audio` is true, a FLAC audio
  track (32 kHz). No metadata (no prompt, seed or encoder tags). At most 64 MiB before
  base64. Kuno validates the container before emitting it.

Codec note: AV1 + FLAC in MP4 plays in Chrome/Edge/Firefox and Android Chrome. Safari
plays AV1 only on devices with hardware AV1 decoding. Both encoders are pure Rust
(rav1e, flacenc), keeping the output path in Rust.

## Models

Collection `minimax-h3-int8` (Comfy-Org/MiniMax-H3, pinned revision and SHA-256):
the INT8 ConvRot `fl2va` DiT, the Qwen3-VL 32B INT8 ConvRot text encoder, the FP16 video
VAE, the FP32 audio VAE and the turbo 8-step LoRA (~77 GB total with the derived file).
After download, `kuno fetch-models minimax-h3-int8` derives
`minimax_h3_fl2va_turbo8_int8_convrot.safetensors` by merging the LoRA in the rotated INT8
basis with seeded stochastic rounding (`apps/kuno/src/lora_merge.rs`, ~90 s). The merge is
plain f32 in a fixed order with counter-based noise, so its output is bit-identical across
machines and thread counts and its SHA-256 is pinned in the collection. Statistically it
matches the earlier Python tool (same changed fraction, noise/delta ratio and correlation
with the LoRA delta). Runtime LoRA application would cost ~2 s per step. Model license:
MiniMax H3 Community License Agreement.

Enable it by adding `minimax-h3-int8` to `image.collections` with the managed SYCL engine.

## Old/new compatibility

- Old Kuno: no `videos` in discovery, `/kuno/v1/videos/generate` is 404. Vashti hides
  video.
- Old Vashti: ignores `videos`; nothing else changes. The image API is unchanged.
- The video workflow requires a managed SYCL runtime whose manifest declares worker ABI 3
  (the resident ABI 2 plus video jobs and progress frames). An older installed runtime
  reports the workflow `unavailable` with `image_runtime_unavailable` and keeps serving
  images. Gateways older than this one reject ABI 3 manifests, so upgrade the gateway first.

## Not in this version

Text-only video, last-frame conditioning (the model supports it; an additive optional
field later), reference audio/video, longer clips or higher resolutions than tested,
resumable or persisted jobs.
