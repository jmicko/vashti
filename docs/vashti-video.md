# Vashti Video Flows

Status (2026-11-23): `codex/video-workflow-modes` integrates the installed
[Kuno video contract](kuno-video.md) from `main` at `a2ab6ac1`. Newer engine
experiments are not included. The Vashti integration does not install or configure Kuno.

## Workflow

- Video has its own sidebar workspace at `/app/videos`, with its own history
  and New Flow action. Images keeps its separate apps screen. The Video entry
  appears only when a configured backend
  advertises a ready, compatible video workflow. Old Ollama/Kuno hosts
  continue to work without a Video entry. Discovery runs from the app shell,
  without requiring a visit to Images, and refreshes on focus and periodically
  while visible. Existing `/app/images/video` links still open the video flow.
- The video model picker groups workflows by host with distinct model labels.
  The lower composer offers Text only, Keyframes (start and/or end frame), and
  References (one to four ordered images). Availability follows discovery;
  first-frame-only hosts remain usable. References are named `<Picture 1>`,
  `<Picture 2>`, etc. in the prompt, with touch-sized reorder/remove controls.
  Reference mode uses a separate model and the first clip after switching can
  take longer. Source thumbnails open the shared full-size image viewer.
  Common browser formats are accepted, converting unsupported formats to
  lossless PNG. Original accepted images are retained.
- The composer preserves the prompt and source after sending. Blank prompts use
  the fixed gentle-motion prompt when input images exist; text-only generation
  requires a prompt. A chevron expands inference settings inline
  in the lower composer, matching Images. The controls scroll while the chevron
  and Send/Stop remain reachable; Escape collapses without losing changes.
  Settings include clip length, visual aspect-ratio presets and Match original,
  audio inclusion, and exact decimal seeds. Memory-capable hosts also show
  Standard and High resolution independently of aspect ratio.
- For memory-capable hosts, the composer and server enforce the advertised frame
  grid and dimension alignment. Standard allows up to 452,608 pixels and 243
  frames (10.13 s); High allows up to 901,120 pixels and 124 frames (5.17 s).
  Both use aligned sides between 256 and 1280 pixels. These are product limits,
  not guarantees about the host's memory. The default stays 832x544 and 56 frames.
  Older hosts keep their advertised sizes and clip lengths. Switching to a host
  or resolution with lower limits fits the editable draft without changing any
  submitted recipe.
- The first input image chooses a supported aspect ratio. Match original uses the
  first reference, or the start/end frame present in a keyframe-only flow. Vashti
  sends uncropped images. A memory-capable Kuno keeps the keyframe aspect ratio
  and composes it into the output; older hosts retain their previous framing.
- Account defaults contain inference settings, not the prompt or frame. Changes
  belong to the flow unless explicitly saved as defaults. Reset keeps the current
  prompt, source and selected input workflow when resetting inference values.
  Server and device defaults are separate.
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
decoding follows the final sampling step (8/8 for keyframes/text, 4/4 for references).
Independent previews can precede the first sampling
progress (including a step-zero preview); late previews do not replace the
encoding display. Final success does not depend on seeing intermediate progress.
Result metadata and the actual bounded MP4 container are validated before saving.
Backend diagnostics/content are not exposed through errors.
No short generation deadline is added.
GPU and host RAM failures explain how to recover without deleting inputs or
automatically resubmitting. The result and Android export bounds remain 64 MiB.

Only one response player retains a full clip in RAM. Sixteen input-image
thumbnails are cached per open flow; originals are loaded on demand. Active
polling fetches lightweight status, not the whole transcript. Native video
downloads accept 64 MiB MP4s through the existing Android document saver;
non-video response/export bounds remain unchanged.

## Verification

### Expanded Clip Acceptance (2026-11-23)

- The isolated server advertised all three workflows with `limits: "memory"`.
  `VASHTI_DEV_VIDEO_SET=memory` with `scripts/video-dev.mjs` submitted a 1280x704,
  22-frame clip and an 832x544, 243-frame clip. They completed in about 70 and
  271 seconds respectively. Neither is a throughput benchmark. A third job
  was canceled during generation and returned `cancelled`.
- FFprobe and complete software decoding verified both results as AV1 MP4 with
  FLAC audio. Chromium played both to the end against the embedded frontend
  and downloaded byte-identical files. The embedded page allows blob URLs in
  `media-src`, not in `script-src`.
- Playwright checked the new controls at 390x844, 844x390, and 1440x960,
  including reducing the duration when High is selected and preserving the
  seven options for older hosts. The locally signed APK installed on the
  connected Pixel 6; the screen remained locked, so native playback and
  export were not verified.

### Current Modes Acceptance (2026-11-15)

- 203 ordinary server tests and 117 frontend tests passed, including all input
  combinations, four-step reference streaming, immutable retries, ownership,
  shared start/end/reference cleanup and private non-persistence. Four existing
  opt-in Assistant/Podman tests remain ignored. Strict server Clippy passed.
- Real installed-Kuno requests through Vashti passed for text-only, first+last
  keyframes and two references. All outputs are 512x512, 22-frame AV1 MP4 with
  32 kHz FLAC audio. Independent FFprobe inspection and complete software FFmpeg
  decoding passed. The live reference clip played to its end in the browser
  (readyState 4, no media error); submitted sources remained inspectable.
  These short synthetic clips are functional checks, not quality or speed benchmarks.
- Browser checks covered reference upload (including multi-select directly from
  text mode), numbering/reorder, full-size inspection/return, original aspect
  matching, duration options, retained prompts and touch controls. Real layouts
  fit 320px/390px portrait, 844x390 landscape and 1440x960 desktop without document
  horizontal overflow; Collapse and Send remain inside the short viewport.
- `scripts/video-dev.mjs` is an explicitly invoked localhost-only smoke test.
  It requires a synthetic account and environment-provided password. Its three
  clearly titled flows and synthetic inputs are retained for development. No
  production/private account content or Kuno diagnostics are used as fixtures.
- A signed universal APK was built locally without changing the version or Hub.
  The factory-reset Pixel 6 (Android 17) is authorized, the app installs and opens
  to its server-connection screen. One launch immediately after replacement exited
  with status zero and a native mutex teardown warning; an explicit relaunch
  rendered correctly and remained running. No AndroidRuntime exception was found,
  but repeat cold-start acceptance remains part of physical-device testing.
  USB forwarding exposes the development server at `http://127.0.0.1:7771` on the
  connected phone. Native generation, playback and document export still need
  hands-on acceptance after connecting/signing in; startup is not full mobile QA.

The November 15 host advertised only frames 22/39/56/73/90/107/124 at 24 fps,
approximately 0.92 through 5.17 seconds. Legacy first-frame requests
retain their old wire shape for older Vashti/Kuno installations. New modes use
optional start/end inputs, ordered references and transient private payloads;
new private-mode requests bind the expected user/server identity before dispatch.

### Initial Integration (2026-11-08)

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

Sidebar follow-up (2026-11-08): all 101 frontend tests, web/native frontend
builds and fixture typechecking passed. Browser checks covered discovery from
Chat, independent Video/Image selection and history, New Flow, legacy links,
320px/390px portrait and short landscape navigation. The isolated fixture also
checked old-server hiding, transient-offline retention and discovery recovery.
No new live inference or physical Android test was performed for this change.

Inline-settings restoration (2026-11-08, user requested): Images and Video
again use a composer chevron with bounded inline scrolling rather than an
inference-settings dialog. Browser checks covered collapse/reopen persistence,
Escape focus restoration, exact seeds, Enter preventing accidental submission,
and simulated Send/Stop with controls expanded. Web/native frontend builds,
all 101 frontend tests and fixture typechecking passed. No Kuno request or
physical Android verification was part of this UI follow-up.
