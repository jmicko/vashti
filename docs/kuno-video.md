# Kuno video generation (2026-11-07, modes added 2026-11-08)

Status: **backend installed and tested (2026-11-07)** from branch `claude/video-api`; see the
backend receipt in the [coordination guide](kuno-vashti-coordination.md). The Vashti
connector and UI are not built yet.

MiniMax H3 video with generated audio, running on the managed SYCL image runtime
(stable-diffusion.cpp). Three workflows:

| Workflow | Inputs | Model (collection) | Steps |
|---|---|---|---|
| `minimax-h3-image-to-video/v1` | first frame (required) | `minimax-h3-int8` | 8 |
| `minimax-h3-video/v1` | text only, first frame, last frame, or both | `minimax-h3-int8` | 8 |
| `minimax-h3-reference-video/v1` | 1 to 4 reference images | `minimax-h3-ref2va-int8` | 4 |

`minimax-h3-image-to-video/v1` is unchanged for existing clients; new clients can use
`minimax-h3-video/v1` for all keyframe cases. Reference images cannot be combined with a
first or last frame (a model limit). The two models are separate DiTs: switching between
the reference workflow and the others reloads the model (the first clip after a switch is
slower). The engine will later be rewritten in
Rust piece by piece behind this same API, checked against the stable-diffusion.cpp
output and speed. Development timings: [video performance](kuno-video-performance.md).

## Principles (unchanged from images)

- Requests, input images, previews, audio and results are transient content. Kuno never
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
      "last_frame": "unsupported",
      "max_reference_images": 0,
      "steps": 8,
      "fps": 24,
      "frames": [22, 39, 56, 73, 90, 107, 124],
      "max_pixels": 452608,
      "dimension_multiple": 32,
      "min_dimension": 256,
      "max_dimension": 1280,
      "audio": true,
      "output_mime_type": "video/mp4"
    },
    {
      "id": "minimax-h3-video/v1",
      "first_frame": "optional",
      "last_frame": "optional",
      "max_reference_images": 0,
      "steps": 8,
      "...": "same limits as above"
    },
    {
      "id": "minimax-h3-reference-video/v1",
      "first_frame": "unsupported",
      "last_frame": "unsupported",
      "max_reference_images": 4,
      "steps": 4,
      "...": "same limits as above"
    }
  ]
}
```

Each workflow appears only when its collection is in `image.collections`, with its own
`state`/`reason` (for example the reference workflow is `unavailable` with
`image_models_unavailable` until its model is downloaded). The top-level `state` is
`ready` when any workflow is. `last_frame`, `max_reference_images` and `steps` were added
2026-11-08; older Kuno omits them (treat as `"unsupported"`, `0`, and 8).

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
- `first_frame`: required for `minimax-h3-image-to-video/v1`, optional for
  `minimax-h3-video/v1`, rejected for the reference workflow. At most 16 MiB of base64,
  decoded at most 4096x4096. Kuno scales it to cover `width x height` and centre-crops
  (aspect preserved). Clients should pick dimensions close to the image's aspect ratio.
- `last_frame`: optional for `minimax-h3-video/v1` only; same format and scaling. The
  clip ends on it. With neither frame the video is generated from the prompt alone.
- `reference_images`: 1 to 4 for `minimax-h3-reference-video/v1` only, same format. They
  keep their aspect ratio (no crop). The prompt refers to them by position as
  `<Picture 1>`, `<Picture 2>`, ..., e.g. "Use the cat from <Picture 1> as the main
  character, riding the bicycle from <Picture 2>." Clients should explain this naming.
- Unknown fields are rejected (`deny_unknown_fields`). Total request body is bounded like
  images (64 MiB).

Fixed recipe (not request fields): CFG 1, CPU RNG, flash attention, and the turbo LoRA
merged into the INT8 ConvRot DiT (8 steps for the keyframe model, 4 for the reference
model; discovery reports `steps`).

Validation errors before streaming use the existing non-2xx JSON error surface, with
codes `unsupported_video_workflow`, `video_prompt_invalid`, `image_seed_invalid`,
`video_dimensions_unsupported`, `video_frames_unsupported`, `video_first_frame_invalid`,
`video_last_frame_invalid`, `video_reference_images_invalid` (an image that is missing
when required, present when not accepted, too large, or undecodable).

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

Collection `minimax-h3-ref2va-int8` (same revision) holds the reference model: the INT8
ConvRot `ref2va` DiT and the 4-step turbo LoRA (v0.1), plus the same text encoder and VAEs
(shared files, stored once). It derives `minimax_h3_ref2va_turbo4_int8_convrot.safetensors`
the same way (~44 GB more on disk with the derived file).

Enable them by adding `minimax-h3-int8` and/or `minimax-h3-ref2va-int8` to
`image.collections` with the managed SYCL engine.

## Old/new compatibility

- Old Kuno: no `videos` in discovery, `/kuno/v1/videos/generate` is 404. Vashti hides
  video.
- Old Vashti: ignores `videos`; nothing else changes. The image API is unchanged.
- Video requires a managed SYCL runtime whose manifest declares worker ABI 4 (ABI 3 added
  video jobs and progress frames; ABI 4 adds text-only, last-frame and reference inputs).
  An older installed runtime reports video `unavailable` with `image_runtime_unavailable`
  and keeps serving images. Older gateways reject ABI 4 manifests, so upgrade the gateway
  first.
- Requests valid before 2026-11-08 are unchanged. In `kuno_protocol::VideoRecipe`,
  `first_frame` is now `Option<String>` and `last_frame`/`reference_images` were added, so
  Rust clients that build the struct need a one-line update (JSON is compatible).

## Not in this version

Reference videos and reference audio (the reference model supports them), keyframes in
the middle of the clip, longer clips or higher resolutions than tested, resumable or
persisted jobs.
