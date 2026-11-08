# Vashti Video Flows

Status (2026-11-08): Vashti integration implemented on
`codex/personal-assistant`, after merging `main` at `2fb4ad0d`.
The [Kuno video contract](kuno-video.md) is unchanged. No Kuno source,
configuration, installation, GPU use, or service restart is part of this work.

## Workflow

- Images keeps its apps screen. Video appears only when a configured backend
  advertises ready, compatible image-to-video generation. Old Ollama/Kuno hosts
  continue to work without a Video tile.
- The video model picker groups workflows by host. MiniMax H3 currently requires
  one first frame; the composer accepts common browser image formats, converting
  unsupported formats to lossless PNG. The original accepted frame is retained.
- The composer preserves the prompt and source after sending. Blank prompts use
  the fixed gentle-motion prompt. Settings are accessible beside attachment/send
  controls, with aspect ratio (including original-image detection), advertised
  clip lengths, audio inclusion, and exact decimal seeds.
- First-frame upload chooses the nearest supported aspect ratio, always within
  the advertised size grid and pixel budget. Kuno performs cover/center-crop.
- Account defaults contain inference settings, not the prompt or frame. Changes
  belong to the flow unless explicitly saved as defaults. Reset keeps the current
  prompt and source. Server and device defaults are separate.
- Prompt summaries expand to the immutable submitted recipe/source. Progress
  shows conditioning, sampling, decoding and encoding; previews are coarse latent
  frames, not final-quality output. Completed responses have playback/download.
- Retry retains the original prompt, source, model and inference recipe, with
  numbered versions and the selected fixed/increment/random seed policy.
  Sending a changed prompt makes another turn in the same flow.
- Video history replaces image history while in video mode. Titles start with
  the submitted prompt; users can search, rename or explicitly delete flows.
  Deletion is confirmed and permanent, including clips. Shared frames remain
  while referenced by another flow.

## Storage and Failure Handling

Authenticated `/api/videos` routes enforce existing Images permission tags and
owner scoping. A forward migration adds preferences, frames, sessions and runs.
Saved clips use owner-checked UUID records and protected `data/videos/*.mp4`
files (0700 directory, 0600 files). Original frames and bounded PNG thumbnails
are stored in SQLite. Image/video storage shares the account media quota.

Device-only flows use the existing account/server-scoped encrypted IndexedDB
vault for drafts, frames, recipes and clips. Only transient private job content
is sent through Vashti to Kuno; it is not written to server rows or files.
Private jobs cancel after 45 seconds without polling; terminal output is held
in RAM for at most 120 seconds. Leaving a device-only flow may therefore
interrupt it, unlike a saved server job. Reopening makes interruption explicit.

Request UUIDs prevent duplicate submission after a lost acknowledgement. Saved
jobs survive navigation but interrupted server processes become failed, never
automatically replayed. Cancellation drops the inference response and releases
the shared image/video media slot. Restart cleanup removes only UUID-named
partial/orphan video files; completed records remain intact.

The consumer requires one started event, exactly one terminal result, and EOF.
Progress is advisory and latest-only: when delivered, phases cannot regress and
decoding follows sampling 8/8. Independent previews can precede the first sampling
progress (including a step-zero preview); late previews do not replace the
encoding display. Final success does not depend on seeing intermediate progress.
Result metadata and the actual bounded MP4 container are validated before saving.
Backend diagnostics/content are not exposed through errors.
No short generation deadline is added.

Only one response player retains a full clip in RAM. Sixteen first-frame
thumbnails are cached per open flow; originals are loaded on demand. Active
polling fetches lightweight status, not the whole transcript. Native video
downloads accept 64 MiB MP4s through the existing Android document saver;
non-video response/export bounds remain unchanged.

## Verification

- `VASHTI_SKIP_WEB_BUILD=1 cargo test -p vashti -p vashti-android --locked -- --test-threads=2`:
  185 server tests and 9 native-wrapper tests passed; four explicit live/Podman
  tests remain ignored. Disposable SQLite databases and local fake inference
  servers test retries, identity, quota, cancellation, owner isolation, private
  non-persistence, deletion, recovery and MP4 validation.
- `node --test src/videos/videos.test.mjs` from the web directory checks aspect
  grids, exact seeds, encrypted private flows, concurrent draft/title writes,
  account/server changes and native export routing.
- All 96 frontend tests passed. ARM64 Android cross-compilation also passed;
  this includes the Android-only transport, unlike host tests alone.
- Web/native frontend builds and strict Clippy passed. Browser fixture:
  `/tests/videos.html`, with every API request intercepted. No live Kuno request
  can escape the fixture.
- `src/videos/fixtures/clip*.mp4` are synthetic CPU-generated AV1/FLAC test
  clips, not model output or user content. Reproduce using
  `web/tests/video-fixture-generator` (isolated Cargo workspace); software FFmpeg
  decoding and browser AV1 playback were checked.

The user released the GPU reservation later in the turn. Live Vashti-to-installed-
Kuno acceptance then passed: discovery, original upload, conditioning/sampling/
preview/decoding, a retry in the same flow (2/2), browser playback and download.
The downloaded MP4 was independently decoded with software FFmpeg: AV1 672x672,
22 frames at 24 fps, FLAC 32 kHz stereo, 35,768 bytes. The source/prompt were
synthetic. The first live attempt exposed overly strict preview/progress ordering;
regression tests now exercise independent/late previews and latest-only progress.
The live upload also caught a missing explicit POST in the new caller; that has
its own regression test.

Physical Android playback/export and release acceptance remain pending. No
release, deployment, Kuno installation/configuration or service changes are
authorized by this receipt.
