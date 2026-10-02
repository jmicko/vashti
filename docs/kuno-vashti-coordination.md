# Kuno and Vashti Coordination

This is the shared handoff between the **Build Kuno inference helper** task
(Kuno backend) and the **Vashti** task (frontend and application integration).
Repository contracts and decisions should survive switching tasks; neither task
should depend on having the other's entire conversation history.

## Ownership

| Area | Default owner |
| --- | --- |
| `apps/kuno`, `crates/kuno-image` | Kuno task: inference, resource management, runtime, installation |
| `apps/vashti/src`, including `images` and the Ollama connector | Vashti task: authorization, persistence, application API, Kuno client |
| `apps/vashti/web`, `apps/vashti-android` | Vashti task: UI, mobile workflows, native integration |
| `crates/kuno-protocol`, workspace dependencies, cross-app specifications | Coordinate changes and identify affected consumers |

The user can explicitly reassign work. This division should not prevent a small,
agreed integration fix, but Kuno work should not independently redesign Vashti's
UI or change its user-data storage.

## Source of Truth

- [Kuno specification](kuno-spec.md): privacy, compatibility, admission, lifecycle.
- [Kuno development guide](../apps/kuno/README.md): operation and validation.
- [Image engine contract](kuno-images.md): supported execution and image API.
- [Vashti Images specification](vashti-images-spec.md): product behavior and UX.

Specifications describe intended behavior; source and tests establish what is
implemented. Clearly label proposals, implemented behavior, and unverified work.
When an approved change supersedes an older requirement, update the relevant
normative section and mark historical notes as superseded rather than leaving
contradictory instructions. Keep this guide focused on the current handoff, not
an ever-growing copy of the specifications.

## Workflow

1. On resume, read this guide, inspect `git status`, and review relevant commits
   since the last handoff. Preserve unrelated work. Do not start from `main` if
   the agreed integration work lives on another branch.
2. Before a cross-app change, record the agreed behavior and contract below:
   routes, methods, request/response fields, authentication, errors, cancellation,
   capability discovery, and old-client/old-server behavior as applicable.
   Resolve unsettled user-facing or security decisions before implementing them.
3. Implement and test the owning application's part. Update the appropriate
   specification and record the backend receipt below. Internal backend changes
   without consumer impact can proceed without this cross-app process.
4. The Vashti task reviews the actual diff, implements the connector and UI
   changes, and tests the full flow on mobile and desktop. Avoid adding UI for
   states or capabilities the backend does not actually report.
5. Mark integration complete only after both receipts exist. Record exact tests
   run, remaining limitations, and whether installed services were tested. Unit
   tests or a build alone do not establish live GPU or mobile behavior.

For sequential work, the shared checkout is sufficient. For concurrent work,
use separate managed worktrees from the agreed current integration commit and
explicitly exchange commits. Coordinate service restarts, installations, model
loads, and GPU tests even with separate worktrees: they still share the host.
Do not auto-message another task or deploy as part of a handoff unless the user
authorizes it.

## Existing Constraints

- Kuno executes transient requests; Vashti owns authorized user-content storage.
  Do not persist inference prompts, outputs, images, or previews in Kuno.
- Keep credentials out of browser responses and handoffs. Kuno diagnostics must
  remain content-free; do not log raw inference or backend error payloads.
- Preserve Ollama-compatible long inference. Do not add arbitrary generation
  deadlines to fix connection or readiness problems; queue/load limits are
  separate concerns.
- Preserve the approved image behavior: each automatic prompt improvement
  starts from the original submitted prompt, not the previous improvement.
  Retry uses the recorded recipe. See the Images specification for details.
- Never edit a previously deployed SQL migration to change its behavior; add a
  forward migration if the integration needs stored-setting changes.

## Active Handoff: Image Access Without Manual Key Setup

**Latest application acceptance:** see the beta.2 integration receipt at the end
of this guide. Kuno's handoff is merged into the normal project checkout; all four
beta.2 artifacts are published and full-download/signature verified. Live Hub
and nginx upload limits are now verified at 1 GiB. Physical Android acceptance
is still outstanding.

**New Kuno backend work:** model-only image residency is implemented on
`codex/kuno-image-residency`, based on `4f9b465`. See the resident-image receipt
below. Source, CUDA and installed Intel/runtime acceptance pass; the new
gateway/runtime is installed on this host. This backend-only change needs no
Vashti connector or UI change. Do not
treat the published ABI 1 beta.2 runtime as the new resident worker.

**Selected image performance follow-up (2026-10-02):** the user authorized
installation of the tested Intel closure and release preparation. Whole-RoPE
plus the residual-first kernel is selected from production `230d15b`; public
package pins/build selection are `5351f63`, strict builder validation is `bde8cd0`,
and the owner source/test integration is `05a95fc`. The exact tested flat runtime
is now installed
on this host through `install-image-runtime`. Its manifest is
`fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa`;
normal archive SHA256 is
`85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3`.

A counter-free, compiler-idle eight-request comparison passes full original
50-step generation/25-step editing on B70/kernel38: warm generation
132.893→106.523 s (19.84% faster), edit 72.769→61.831 s (15.03% faster).
Exact native tests, protected SDK-hidden full requests, previews/PNG/privacy,
resident PID reuse, normal shutdown, cancellation and fresh-worker recovery pass
with no new classified Xe fault. Installed keyless HTTP generation/edit pass,
including streamed previews and reuse of one retained worker. The gateway binary,
Ollama routing, inference deadlines and image API are unchanged; no Vashti
connector/UI changes are required.

See [the performance receipt](kuno-image-performance.md#whole-rope-combined-worker-and-full-workflow-comparison)
and [the release handoff](kuno-release-handoff.md). Historical ComfyUI times are
not a new matched benchmark. Pixel-level equivalence/quality tolerance and
indefinite stability are not established; identical-runtime edit variation is
also measured. The exact frozen runtime has been tested and can be packaged
normally. A future fresh managed CMake rebuild needs a reviewed MKL configuration
and acceptance of the resulting binary. Standard native inputs and a fresh
normal gateway build/check/Clippy pass, as do 40 exact CUDA and 40 exact Vulkan
math cases on the 3060. Four original-resolution two-step
CUDA generation/edit requests pass previews/PNG/privacy, resident PID reuse and
normal shutdown. Live Vulkan image acceptance is not claimed for this update; the shared engine patch affects both backends.

Source selection is `05a95fc`, acceptance receipts are `b48d4fb`, on
`codex/kuno-image-speed`. Integrate the complete Kuno range after shared base
`4f9b465`, including residency/shared-PE prerequisites, preserving this checkout's
newer application/coordination work. Do not cherry-pick only the selection commit. Do not replace the standard
gateway with the SYCL-linked worker or reuse the published ABI1 beta.2 runtime.
No model/content storage, API, frontend, public release or driver/reset/reboot
change is part of this performance handoff.

**Status:** Kuno backend, Vashti connector, and conversational image flows are
implemented and source-tested. Synthetic application flows and installed native
generation, regeneration, editing, previews, and cancellation have been verified
through Vashti in browser tests. Physical Android acceptance remains separate.
Baseline `d117295` on `codex/vashti-images`, recorded
2026-09-30. Preserve the current checkout and newer work.

**Agreed access direction:** Kuno is a local-network inference service. Remove
control-key authentication from public generation and previews on upgrade,
without adding inference bearer modes, image access settings, or a separate
operator migration. Privileged loopback controls retain their existing key.
The user's clarification supersedes the earlier pending authentication proposal.

**Current contract:** see [inference access, discovery, and streaming](kuno-images.md#inference-access-discovery-and-streaming-2026-09-30)
for exact discovery fields, readiness/errors, keyless/control separation,
existing-install behavior, and the default NDJSON generation response.
Streaming delivers previews and final output on the generation connection.
The user explicitly dropped compatibility with the older beta Vashti image
connector: remove PNG-only responses, polling, and preview identifiers. Ordinary
Ollama compatibility is preserved. The backend and consumer receipts are below.

### Baseline Coupling (before this change)

- [`gateway.rs`](../apps/kuno/src/gateway.rs): image generation and preview routes
  on the compatibility listener require the same bearer token used by the
  separate control listener. Ordinary Ollama routes keep their proxy semantics.
- [`handlers.rs`](../apps/vashti/src/images/handlers.rs): image settings expose
  `token_configured`; omitted/null `token` preserves the stored value, while an
  empty string clears it. Token validation currently differs from Kuno: Kuno
  accepts `.` in addition to letters, digits, `-`, and `_`; Vashti does not.
- [`service.rs`](../apps/vashti/src/images/service.rs): `endpoint()` rejects an
  empty token before connecting; generation and previews use bearer auth.
- [`ImagesWorkspace.tsx`](../apps/vashti/web/src/images/ImagesWorkspace.tsx) and
  [`types.ts`](../apps/vashti/web/src/images/types.ts): token presence is treated
  as service readiness and gates Generate. The settings form submits null for a
  blank token, so it cannot currently clear an existing credential.
- [`kuno-protocol`](../crates/kuno-protocol/src/lib.rs): versioned shared types;
  current control status is not a general keyless capability contract.

### Contract and Consumer Requirements

- Keyless image generation streams previews on the inference listener, separate
  from privileged control authentication. Preserve existing Ollama auth/origin
  behavior. Do not solve manual setup by disabling all authentication.
- Specify discovery route, versioned fields, actual availability/readiness, and
  authentication expectations. A plain Ollama host must remain usable for chat;
  absence of a token is not proof that an image service is disconnected.
- Preserve external proxy access errors. Remove the obsolete image-key workflow
  and stored secrets through a Vashti forward migration; new Kuno inference
  needs no credential.
  Never return credentials to the browser or silently downgrade failed auth.
- Specify compatibility with installed configurations and older Vashti/Kuno
  versions. Distinguish unsupported images, missing models, unreachable hosts,
  authentication failures, and actual readiness rather than collapsing them all
  into a connection warning.
- Record representative, secret-free JSON responses and relevant HTTP error
  codes in the authoritative API specification. Keep discovery bounded and
  content-free; do not make it an expensive inference probe.

### Acceptance and Receipts

Backend tests cover keyless inference with missing/obsolete credentials,
unauthenticated control rejection, generation/preview policy consistency,
ordinary Ollama proxy behavior, configuration compatibility, discovery/readiness,
and stream events, backpressure, disconnect cancellation, and cleanup ownership.
Vashti tests should cover plain Ollama, the new Kuno contract, older-Kuno upgrade
guidance, stored-credential removal, incremental NDJSON handling, actionable
errors, and admin/non-admin UI.
Manually verify discovery, Generate, previews, Stop, and settings on mobile and
desktop; record any unavailable live-engine tests rather than claiming success.

### Kuno receipt (2026-09-30)

**Changed commit:** `f6f6c22f6eea3f8e33900b725fbbfe56fda44e01`
(`Make LAN image inference keyless and stream previews with results`), baseline
`d117295`. Backend areas: `apps/kuno/src/gateway.rs`, new `image_access.rs` and
`image_stream.rs`, shared discovery/stream types in `crates/kuno-protocol`, native
CUDA capability constant in `crates/kuno-image`, and focused tests. API/source
specifications and the Kuno development guide are updated. No code under
`apps/vashti` changed.

**Exact checks:**

- `cargo test -p kuno -p kuno-protocol -p kuno-image`: 171 passed, nine ignored,
  zero failures. Ignored checks require GPU/native/systemd environments.
- `cargo clippy -p kuno -p kuno-protocol -p kuno-image --all-targets -- -D warnings`:
  passed.
- `KUNO_SD_SOURCE_DIR=/home/jmicko/dev/vashti/target/kuno-sd-source-88411ef KUNO_SD_BUILD_DIR=/home/jmicko/dev/vashti/target/kuno-sd-build-88411ef-cuda cargo check -p kuno --features native-image --all-targets`:
  passed using the pinned patched engine.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

Tests verify public inference with absent/obsolete keys, protected control
routes, no public administrative proxy path, JSON/origin policy, unchanged
Ollama header forwarding, discovery while admission is paused, conditional
attention support, missing-model degraded startup with a disposable worker,
preview/result/error ordering, actual synthetic HTTP preview delivery,
disconnect propagation, content-slot retention through delivery/cleanup,
recovery after uncertain cleanup, and privacy failure before body polling.
Native worker cancellation/device-loss tests remain passing. No owned GPU was
loaded for this work.

**Installation/configuration:** existing configuration loads without new keys.
Upgrading removes the public image control-token requirement automatically.
Control credentials/listeners, explicit timeouts, model settings, and Ollama
routing retain their roles. No installer/settings migration is needed in Kuno.
This is a breaking beta image transport change: install only with the updated
Vashti connector. No installation, service restart, release, or task message was
performed. Existing uncommitted `AGENTS.md`, this guide, and root README edits
remain uncommitted and preserved.

**Required Vashti work:**

1. Discover `/kuno/v1/capabilities` from the selected existing inference backend,
   with the bounded probe/error rules in the API spec. Chat on plain Ollama stays
   usable; unsupported images/old Kuno receive upgrade guidance.
2. Remove the image-token field, readiness gate, and bearer header. Remove stored
   legacy secrets with a forward migration; never return/repurpose those values.
3. Incrementally consume the single NDJSON generation response; previews and
   final image use the same request. Validate/decode bounded PNGs, distinguish
   pre-stream HTTP errors from terminal error events, and treat premature EOF as
   failure. Remove polling and preview-ID handling. Do not replay dispatched jobs.
4. Propagate Stop by dropping the upstream request/response. Preserve Vashti user
   authorization, private storage behavior, and image/job ownership.
5. Test discovery, settings, generation/edit, preview display, errors, and Stop
   through the Rust connector and mobile/desktop UI before deployment.

**Limitations:** source checks and synthetic transports do not prove real engine
stream performance or full application behavior. No live generation on either
GPU, installed-service check, Android/mobile test, or end-to-end Vashti test was
run. Readiness is a cached prerequisite snapshot, not a memory reservation or
GPU health guarantee; it refreshes on demand after five seconds, allows only one
probe, and returns its prior snapshot after a one-second wait. Slow filesystem
probes can outlast that wait. Existing image-worker deadline and global inference
serialization remain unchanged. Fast attention is advertised conservatively
only for a configured CUDA image device with a CUDA-enabled native build.

### Vashti receipt (2026-09-30)

**Consumed backend:** `f6f6c22`, using the discovery and mandatory NDJSON contract
above. Consumer changes are in the current worktree on `codex/vashti-images`;
no consumer commit, release, or installed Kuno update was made in this pass.

- Added bounded, per-host cached discovery and a reusable server-side streaming
  client. Validate service/API markers, same-host paths, reference limits,
  attention, event order and sizes, and decoded PNG dimensions. Surface access,
  readiness, protocol, and interrupted-delivery errors without raw error content.
- Removed key fields, key-presence readiness, bearer auth, preview identifiers,
  and upstream preview polling. Forward migration clears/removes the legacy
  database column while preserving the selected host, quota, and image assets.
- Image preferences now check the selected existing inference host and show
  actual workflows/readiness. Generate, Retry, and chat tool availability use
  capabilities. Settings changes remain admin-only; image access remains
  session/permission scoped. Unsaved host selections cannot dispatch work.
- Both persistent and private image jobs use streamed, transient previews and
  final delivery. Stop drops the upstream body and clears previews. No automatic
  replay or arbitrary generation response deadline was introduced.

**Exact checks:**

- `cargo test -p vashti`: 134 passed, zero failures.
- `cargo clippy -p vashti --all-targets -- -D warnings`: passed.
- `node --experimental-strip-types --test src/*.test.mjs src/images/*.test.mjs`
  from `apps/vashti/web`: 13 passed, zero failures.
- `npm run build` and `npm run build:native`: passed, including TypeScript checks.
- `cargo fmt -p vashti -- --check` and `git diff --check`: passed.

Tests cover old/plain-Ollama discovery, external 401/403 without fallback,
redirect/malformed/oversized/slow discovery, partial readiness, split stream
chunks, invalid and duplicate results, premature EOF, bounded events, legacy-key
migration, admin/non-admin access, cancellation/disconnect, ownership, private
RAM-only delivery, chat image tools, and original-prompt improvement semantics.

**Manual application checks:** used the development app through the browser and
a temporary loopback-only synthetic Kuno host. Verified discovery and saving a
keyless connection, ready workflow choices and unavailable Fast attention,
Generate, decoded preview/final image display, reference-image Edit, Stop
clearing previews and disconnecting upstream, interrupted delivery preserving
saved output without replay, and proxy denial disabling Generate/Retry. Checked
desktop and 390px mobile flows, plus 320px preferences with no horizontal
overflow. Restored the original connection/quota and removed only test-owned
backend, sessions, and assets after checks; user data was not removed.

**Limitations:** browser resizing is not a physical Android-device test. The
installed local host without discovery correctly shows upgrade guidance. No
installed Kuno restart, native GPU generation, or live-engine performance test
was performed. Upgrade Kuno and Vashti together after quiescing existing jobs;
synthetic delivery does not establish GPU readiness or speed.


### Installed Kuno receipt (2026-09-30)

The user explicitly authorized installation. Built committed Kuno `f6f6c22`
from an isolated worktree with the pinned static CUDA/Vulkan native engine:
`KUNO_SD_SOURCE_DIR=/home/jmicko/dev/vashti/target/kuno-sd-source-88411ef KUNO_SD_BUILD_DIR=/home/jmicko/dev/vashti/target/kuno-sd-build-88411ef-cuda CARGO_TARGET_DIR=/home/jmicko/dev/vashti/target cargo build --locked -p kuno --release --features native-image`.
Ran `sudo -n /home/jmicko/dev/vashti/target/kuno-install-f6f6c22/kuno upgrade --yes`
in the shared terminal after user-entered sudo authentication. The transactional
upgrade reported `kuno_upgraded_and_active`. No Vashti application files or
services were installed or restarted in this pass.

Installed executable SHA-256:
`09f9f5d3b4f56a02ad08605ef441508a9ad1723f4d28a03d55073655f876810c`.
Configuration remained byte-for-byte unchanged, including `[::]:11434`, the
loopback upstream/control ports, both workers/collections, image GPU RTX 3060,
and disabled LLM response/idle deadlines.

Verified on the running installation:

- Root `kuno doctor`: installation complete, unpaused, no active/waiting request
  at the post-test check, no recovery requirement/pending recovery.
- Kuno service active/running with zero restarts; recovery timer active.
- Keyless discovery reports all configured image workflows ready. Ollama version
  and discovery succeed through IPv4 loopback, IPv6 loopback, and the LAN address.
- Clark independently reached LAN discovery and Ollama's version endpoint.
- Ollama catalog endpoint succeeded; unauthenticated local control returned 401.
- Direct keyless native Qwen 2.1 stream: 512x512, two steps, standard attention;
  acceptance, two previews, and one final PNG. First preview at 8.60 seconds,
  total 11.61 seconds. Validated dimensions, PNG chunk checksums, metadata
  absence, and terminal completion. All image bytes stayed in client RAM and
  were discarded; no test images were saved.

This is a native transport smoke test, not a quality/performance benchmark or
proof of physical Android/full application workflow behavior. Live Stop/Edit
and full reference workflows were not tested in this installation pass.
Earlier source/application receipts retain their historical limitations; this
receipt supersedes their statements that installed Kuno was unchanged.
Existing uncommitted coordination and Vashti task work is preserved.

### Vashti conversational-flow receipt (2026-09-30)

The user approved replacing the separate image-generation forms with familiar
conversational flows. Consumer work remains uncommitted on `codex/vashti-images`.
The existing Kuno discovery contract was sufficient; no further Kuno API or
engine changes were made by this task. The Kuno task updated the installed local
service while browser acceptance was in progress.

- Added the Images launcher, host-grouped image-model picker, bottom composer,
  expandable settings, compact prompt bubbles, streamed response bubbles, and
  image-flow sidebar history. Gallery and image-tree inspection reuse these
  persisted flows and exact image ancestry. Existing chat tools remain separate.
- A forward migration adds active-result selection, automatic/manual titles,
  history removal, immutable submission fingerprints, and run pagination.
  Regeneration captures the original host, prompt, settings, and references,
  allocates a new seed, and stacks response variants without copying children.
- Idempotent submission retries recover lost acknowledgements without replaying
  dispatched inference. Drafts autosave without resetting the composer or
  overwriting generated titles. Older selected branches load their ancestry.
  A global active-run indicator keeps Stop available from the launcher/gallery.
- Fixed completion/polling races, preview sizing for native low-resolution frames,
  seed advancement after regeneration, narrow-screen composer spacing, and
  unsaved Preferences cancellation. Image decoding/layout changes no longer
  pause following the latest output; deliberate scrolling does. Preview bytes remain transient; completed
  assets are owner-scoped and use the existing bounded browser asset cache.

**Checks:** `cargo test -p vashti` (136 passed), Clippy with warnings denied,
frontend helper tests (20 passed), web/native frontend builds, Rust formatting,
and whitespace checks. Backend tests cover explicit/automatic host routing,
captured recipes, idempotency, new seeds and overflow, exact edit ancestry,
owner-only titles/selection/history, and removal retaining gallery assets.
Frontend tests cover branch reconstruction, incomplete history, independent
batches, pending retries, cyclic input bounds, and safe incremental seeds.

**Manual acceptance:** tested the development app with a disposable synthetic
loopback host, then the installed native Kuno service. Verified model discovery,
prompt/setting persistence, uploads, Match original, reference editing, native
generation and regeneration with distinct seeds, real previews, automatic
titles, cancellation during sampling, failed-delivery/manual Retry, variant
arrows and viewer swiping, exact child-branch restoration, tree focus/navigation,
and Stop while viewing the gallery. Synthetic backend/session/assets were
removed without changing existing hosts, quotas, preferences, or user histories.
Checked 1440x900 desktop, 390x844 mobile, and 320x568 short/narrow layouts; the
composer remains collapsible without document-level horizontal overflow. The
shared ordinary-chat model picker still lists/searches models. One native test
flow with saved generation/edit variants remains available for user inspection.

**Limitations:** resized browser tests are not physical Android acceptance or a
performance benchmark. Native checks use small temporary recipes rather than
changing user defaults. No commit, release, or deployment was performed.

### Kuno inference optimization installation (2026-09-30)

Internal Kuno work is isolated on `codex/kuno-image-performance`, based on
`f6f6c22`; concurrent Vashti source and the uncommitted coordination documents
are preserved. Standard CUDA/Vulkan improvements through `eee97c8` were
transactionally installed with user authorization. Installed SHA-256:
`bbd27f1f6cc9d35ecdb24ab32f274f46e8add649896753298fa2b9fb80d345cf`.
Configuration is byte-for-byte unchanged, including ports, RTX 3060 image GPU,
collections and disabled LLM response/idle deadlines.

The standard candidate passed independent attention/cache GPU oracles on both
cards, full original CUDA generation/edit and Intel edit, streamed previews,
multiple references and cooperative cancellation/recovery. The installed
service passed keyless native 512×512/two-step NDJSON generation (both previews
and final, 11.994 s), PNG checksums/dimensions/no metadata, discovery/version
through IPv4, IPv6 and LAN, and unauthenticated-control 401. No private content
was saved; content-bearing tests use zero-swap cgroups. No new Xe reset entry
was observed.

Intel Vulkan execution works, but the speed target remains open. A private
SYCL vendor-library experiment is under development and is **not installed**;
neither are experimental dense MLP/tile policies enabled in the public worker.
Exact measurements and limitations are in `docs/kuno-image-performance.md` on
the Kuno branch. No API/discovery/stream changes or Vashti consumer changes are
required. No application install, merge, release or publication occurred.
Admission was resumed after the standard installation tests and again after
the private decode diagnostics; the installed service is unpaused and healthy.
Any later exclusive test window must resume it afterward.

Private SYCL source/runtime support through `cb34430` remains isolated on the
Kuno branch and is not enabled or installed. Synthetic kernels and small text
generation passed. The original edit completed all sampling steps but failed
during VAE decoding; reduced reproductions identify the SYCL `IM2COL` operation.
Two additional Xe reset entries occurred earlier, so their origin remains a
separate stability question. Full reference-workflow acceptance remains open.
The optional managed runtime preserves the same image API and native Qwen 2512
route; no Vashti connector/UI changes are required. Source tests and the exact
installation/manifest contract are recorded in `docs/kuno-managed-sycl.md` on
the Kuno branch. Do not present that experimental route as ready to users.


### Installed B70 SYCL acceptance (2026-10-01 UTC)

This receipt supersedes the earlier uninstalled/private-SYCL and failed-decode
status above. Backend source remains isolated on `codex/kuno-image-performance`,
based on `f6f6c22`; main remains on `codex/vashti-images` with its application
changes and both untracked coordination documents preserved.

**Changed commits:** managed runtime `9a51d46`, deployment/environment review
`7fe6283`, bounded IM2COL `46cbd5c`, independent large-oracle/addressing audits
`35eba2e`/`7fe2403`, wide copy addressing/build watchers/oracles `30cebda`, final
hardware/installed receipts `c5b45be7301209091cdd8387633fd1b837e290bf`. Earlier
CUDA/Vulkan improvements remain part of the Kuno branch. Full specifications
and detailed receipts are `docs/kuno-managed-sycl.md`, `docs/kuno-images.md` and
`docs/kuno-image-performance.md` on that branch.

**Exact backend contract:** the existing keyless image NDJSON API, discovery
schema, public Ollama routes and privileged loopback controls are unchanged.
Omitted image.engine still means native; explicit managed_sycl plus fixed runtime
qwen21-sycl-2026.1-v1 selects the optional worker for Qwen 2.1 text/edit only.
Qwen 2512 uses embedded native on the same configured image PCI. The managed
worker advertises standard attention only; explicit fast returns 400
image_attention_unsupported. Missing/invalid/untrusted/corrupt runtime produces
fixed image readiness reasons and 503 preflight errors while ordinary Ollama
remains usable. Manifest/CLI/configuration details are normative in the runtime
specification; no arbitrary executable/library path or inference credential was
added to the public API.

**Installed artifacts/configuration:** standard CUDA/Vulkan gateway SHA256
`d2a7a293bb021daebb1dda2c8b21a67443881a10ce0787dbd4b7ae2c4946d8dc`;
optional worker `414b86a3c498c0f848321766e99784a8d5d7fd7f3984c1bf35796f6b5b14796b`;
54-member runtime manifest
`94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`.
The owned-file transaction installed the runtime without changing config;
explicit image configuration changed only engine/runtime/PCI to B70
0000:2d:00.0. Collections, model store, listeners, Ollama workers and zero LLM
response/idle deadlines were preserved. Global default remains native. The
standard binary does not link oneAPI. Website runtime publishing/download and
older-Linux portability are not implemented or accepted; no public release.

**Exact checks:** source suite 182 passed/nine environment-dependent ignored,
Clippy warnings denied (nonnative and ordinary native all targets), native release
build, formatting/whitespace checks. Independent CPU/GPU arithmetic and attention
oracles pass. IM2COL modes 1–8 and wide-copy modes 1–3 pass, including actual large
VAE geometry, 2/4/8 GiB and INT32/UINT32 boundaries. BF16 numerical GPU coverage
and arbitrary copies above INT32_MAX work-items remain unverified.

The final packaged worker passes original 1184x1776/50-step/CFG1 generation
153.950 s (50 previews), original 1248x832/25-step/CFG4 edit 115.513 s (25 previews),
and two-reference 512x512/four-step editing 61.282 s. These fresh-worker tests used
only the flat bundle with the entire SDK extraction inaccessible, zero swap and
write denial before vendor initialization. Original outputs were visually
inspected in RAM without decoder seams, then discarded. Full-size cancellation
at step 3 reaped cooperatively in 3.061 s. No further Xe reset occurred during final
candidate acceptance; boot event count remains 7, including the earlier failed
large decode. The fixes address proven range/copy bugs, not every driver fault.

Installed HTTP checks pass keyless generation/previews/result, IPv4/IPv6/LAN,
control 401, standard-only discovery/fast 400, native 2512/two-step route 27.328 s,
ordinary Ollama generation 1.077 s, disconnect cleanup 1.003 s followed immediately
by Ollama generation 0.935 s, and missing-runtime 503 with Ollama 200/2512 ready then
restoration without restart. Final service active/running, NRestarts 0,
MemorySwapMax 0, admission unpaused, no active/waiting job or recovery quarantine,
no image worker left. Known synthetic request text/image fields and the control
credential are absent from the acceptance-window service journal. No generated
content was saved; temporary test configuration/runtime renames are removed.

**Consumer impact/limits:** no Vashti code changed or application/mobile test was
rerun for this internal backend work. Existing consumers must honor discovery's
standard-only attention list; saved CUDA fast recipes need explicit standard
selection. Up to ten references remain allowed by the API, but live acceptance
covers two and does not guarantee all inputs fit. ComfyUI user-supplied B70
cold/warm baselines remain 132.06/100.65 s generation and 78.25/64.48 s editing;
Kuno is substantially faster than its prior engines but parity remains open.
Corrupted journal-owned runtime files require manual repair before configure/
upgrade/recover; rollback snapshots are RAM-only. See the runtime specification.
No merge, push, Vashti deployment, release or message to another Codex task was
performed. Resume from this tested checkpoint with ordinary inference available.


### Kuno beta release handoff (2026-10-01 UTC / 2026-09-30 local)

**Status:** tested backend ready for beta integration; the user explicitly does
not require perfect/indefinite stability. Release preparation is complete here;
publication and final application acceptance belong to the Vashti task.

**Exact source:** Kuno branch `codex/kuno-image-performance`, base `f6f6c22`,
final tip `83ec03f5e75206711f151f72c87cfeab3c592a19`. Integrate the entire 44-commit
range `f6f6c22..83ec03f`, preserving the main checkout's uncommitted application,
migrations/specs and both untracked coordination documents. New commits are
`6ef9380` (verified packaging/notices/public build provenance, eight CPU tests)
and `83ec03f` (release receipt); serving behavior/hardware acceptance remains
`c5b45be`. No Vashti/Hub code changed here, and no merge/push/publication/message
to another task occurred.

**Authoritative release instructions:** `docs/kuno-release-handoff.md` on the
Kuno branch, currently at
`/home/jmicko/.codex/worktrees/kuno-image-performance/vashti/docs/kuno-release-handoff.md`.
It contains exact artifact paths/checksums, install commands, source provenance,
platform limits, receipts and remaining release-owner work. Preserve this
worktree and ignored `target/release-handoff` artifacts until they are copied.

**Contract:** no new inference API or credential requirement. Native remains
the default; explicit managed_sycl/fixed runtime enables B70 Qwen 2.1. Native
2512 uses the same configured PCI. Discovery is Standard-only on SYCL; explicit
Fast is 400. Missing optional runtime is fixed-reason 503 for affected images
while Ollama remains usable. Privacy and disabled default long-LLM deadlines
are preserved. Vashti must respect actual attention capabilities and test saved
Fast recipes alongside Generate/Edit/previews/Stop and storage. Physical Android
acceptance remains the application task's responsibility.

**Prepared optional artifact:** `target/release-handoff/kuno-sycl-linux-x86_64.tar.gz`,
156,166,814 bytes, SHA256
`d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f`;
approved manifest
`bf31edfe7c9630b6b6b367c98831aa7d4e931a797fcf29d519b00d7843413a99`.
Its 64 members add ten native notices only; every old worker/library/member byte
is unchanged from the hardware-tested installed 54-member bundle, whose manifest
remains `94daf0ed8e7cd57ae861a0213c1ef5751a0261b7d2ee9d04bcd47d682941b59a`.
The installed runtime was not modified by packaging. Archive full read-back and
all eight CPU regressions pass. Standard packaging now fixes output paths and
rejects private Intel dependencies/RPATH instead of risking stale publication.

**Latest live receipt:** installed two-step stream after idle passed two previews
and final in 40.870 s; service active/running, NRestarts 0, MemorySwapMax 0, no new
Xe reset (boot count remains seven). Original full Intel generation/edit passed
153.950/115.513 s. Faster than previous Kuno engines, still behind ComfyUI; no
additional speed optimization or overnight soak is needed for this beta.

**Release limits/work:** GLIBC 2.39/GLIBCXX 3.4.30 and tested CPU/driver scope only;
current standard CUDA-enabled ELF requires libcuda.so.1 even with Intel images.
Other-system portability is unverified. The publisher currently lists three
artifacts and the website installer only extracts standard Kuno. Hosting the
optional runtime/checksum/install instructions and the final coordinated build
remain release-task work. Prepare every intended artifact before consuming the
one-time upload key because published Hub versions are immutable. Source and
packaging checks are not proof of full Vashti/Android release acceptance.

### Vashti beta.2 integration receipt (2026-10-01 UTC / 2026-09-30 local)

**Source and Git:** earlier application changes preserved in `c6109d4`; the full
Kuno handoff through `83ec03f` is merged as `63ce405` on `codex/vashti-images`.
The application defaults/selection fixes and regressions are committed as `20d2c0f`.
The normal checkout contains `docs/kuno-release-handoff.md`. Keep the Kuno
worktree: its ignored native build inputs are still used for release packaging.
No Kuno service configuration or production inference implementation was changed
by this application acceptance pass. No merge to main, push, or publication.

**Account settings contract:** generate/edit preference profiles add optional
host, image count, automatic improvement, prompt-helper configuration, and edit
source mode. Absent legacy fields preserve existing defaults; explicit null count
means endless generation. Settings remain owner-scoped. Flow changes persist in
the flow without rewriting account preferences. Explicit Save updates the selected
account profile; Reset restores all settings while retaining the current prompt
and appropriate edit references. New flow with these settings starts a separate
flow without copying positive prompts, references, or branch ancestry. Existing
flows and model-tool override permissions are unchanged. These are additive JSON
fields; no deployed migration was rewritten and Kuno's API is unchanged.

**Consumer fixes:** unsupported saved Fast attention is shown explicitly and
requires Use Standard rather than silently changing the recipe. Submission
transactionally anchors the selected image branch before output arrives; later
manual navigation still wins. Returning to an active generation resumes following
its completion, and returning to completed output persists the selected result.

**Live acceptance:** installed Standard-only Kuno discovery, 512-square/30-step
generation, 25-step editing with two ordered references, streamed previews,
regeneration with a new seed, Stop, Retry, persisted branch selection after
reopening history, and explicit Fast-to-Standard recovery passed in the browser.
The two references were copies of the same synthetic test image; this verifies
transport, not arbitrary multi-image semantic quality. The earlier two-step image
was intentionally a transport smoke test, not a quality sample. Responsive
390x844 settings checks covered Save, Reset, new-flow copying, subsequent fresh
flow defaults, and original/iterative edit mode. Test account preferences were
restored afterward. No inference job remains active.

**Source/package checks:** full locked workspace tests passed serially; Clippy
all targets with warnings denied, formatting, frontend helper tests, web/native
frontend builds, release packaging regressions, and Kuno's eight CPU packaging
tests passed. A 100 ms Kuno preload-body-timeout fixture failed once under heavy
parallel build/GPU load, then passed individually and in the full serial suite;
no inference deadline or test behavior was changed. Packaged Linux starts against
a fresh data directory and serves beta.2 plus the exact final frontend assets.
The universal signed APK has version code 13002 and four ABIs. All four release
artifacts pass checksum and signing dry-run validation, including the unchanged
approved optional runtime archive.

**Release gate and limits:** complete multipart upload is approximately 644 MiB,
but the live Hub currently sets `VASHTI_HUB_MAX_UPLOAD_BYTES=536870912`. Raise it
and any proxy limit to at least 1 GiB before requesting/using a one-time key.
No live Hub change or upload was made. Physical Android installation/export,
browser download completion, other-system portability, overnight soak, and
ComfyUI parity remain unverified. Native/optional runtime platform restrictions
and installation instructions are in the versioned release notes and backend
handoff; source/browser results are not a substitute for those remaining checks.

### Live upload-size gate cleared (2026-10-01)

At the user's request, the live Hub request limit is now 1073741824 bytes,
set in `/etc/systemd/system/vashti-hub.service.d/90-upload-limit.conf`. The original
unit and installed binary are unchanged (their before/after hashes match). The
Hub restarted successfully and its running process environment contains the new
limit. The separate nginx site configuration now sets `client_max_body_size 1G`;
only that value changed, the prior configuration was backed up on the proxy host,
`nginx -t` passed, and nginx reloaded without a restart. Sudo authorization for
both changes was entered by the user in the shared Demons shell, not chat.

Header-only HTTPS probes with no key or artifact data verified the boundary:
declared Content-Length 1073741824 reaches the Hub and returns authentication 401;
1073741825 is rejected by nginx with 413. The public website returns 200. This
establishes the configured request-size gate, not a completed artifact upload.
No key was requested/consumed, no release was uploaded, and the physical Android
and portability limitations above remain unchanged. The earlier 512/500 MiB
limits in the historical receipt no longer block the beta.2 upload.

### Beta.2 publication receipt (2026-10-01)

Published `v0.13.0-beta.2` using the normal signed, four-artifact multipart path
and the user-supplied one-time key. The credential was entered through the
publisher's hidden prompt and was not saved to a file or handoff. The Hub accepted
the complete request; no subsequent artifact was added to the immutable version.

Public verification covered the release database entry/status, exact release
notes, SHA256SUMS, all four prerelease manifests, their Ed25519 signatures against
the repository's compiled public key, artifact filenames/sizes, and full streamed
downloads whose SHA-256 hashes match the local release files:

| Target | Bytes | SHA-256 |
| --- | ---: | --- |
| linux-x86_64 | 9840548 | `bdc8eeffdadf0ab97efa9acc2c1d1625c50f444de396c00b12b29844c5efb577` |
| android-universal | 52023257 | `e3202deffa2604977961e7df7e1e2d5b94bba69d5eeb85a74bf41e0b69ba3f8d` |
| kuno-linux-x86_64 | 456424687 | `b74938ea3d4f1b44626ed295680db85ebe051d5ac398593cdf77b1889d27d126` |
| kuno-sycl-linux-x86_64 | 156166814 | `d242d7628ccb6c6048e7bdb7edca04d1c2e7453404e68667e829aff61b5ddd2f` |

The prerelease channel now returns beta.2; stable remains `v0.12.0`, with no
promotion, main-branch merge, Git push, or live Vashti/Kuno installation. The
physical Android and platform/runtime limits in the acceptance receipts remain.

### Resident image backend receipt (2026-10-01)

The user approved retaining image models after generation. Implementation lives
on `codex/kuno-image-residency`, based on `4f9b465`, in the managed worktree
`/home/jmicko/.codex/worktrees/kuno-image-residency/vashti`. Backend changes through
`4fb4e58` add the persistent supervisor, private worker ABI 2, native request
cleanup, retained GPU parameters, cancellation safety, and managed LLM placement.
See the normative [image residency contract](kuno-images.md#api-and-lifecycle)
and [private worker ABI](kuno-managed-sycl.md#resident-worker-abi-2).

The public image/discovery API and Vashti storage contracts are unchanged.
No code in `apps/vashti` or the Hub changed. Metadata keeps the image worker;
managed inference prefers another GPU and evicts images before unknown or
unsafe same-device loads. There is one compatible image context, with no idle
expiry. Cancellation and failure conservatively unload it; useful inference is
still serialized. This is not an arbitrary workload-sharing guarantee.

**Verified source:** `cargo test --offline --locked -p kuno -p kuno-image -p
kuno-protocol -- --test-threads=1` passed 205 tests, nine ignored.
Clippy all targets with warnings denied passed both default and standard
native-image builds. Formatting/diff checks and eight CPU packaging regressions
passed. The nine new supervisor tests cover repeated PID, CLEAN gating,
cancellation/input/output backpressure, malformed output, replacement, and idle
exit recovery. Both native engine archives rebuild all 57 C++ units; the fresh
standard archive passes the actual CPU cache/request-cleanup regression.

**Verified CUDA smoke:** in a swap-disabled user service, three 512-square,
two-step Qwen 2.1 requests reused one process and retained 7.69–8.23 GB in VRAM.
Times were 11.979, 7.977, and 7.772 seconds. A changed seed changed pixels;
returning to the original seed reproduced them. Idle shutdown exited normally.
No images or request content were saved. This is a process-cold smoke with an
uncontrolled file cache, not a full-workflow/ComfyUI benchmark.

**Prepared artifacts:** exact source/build pins and checksums are in
[`kuno-image-residency-build.json`](kuno-image-residency-build.json).
The standard gateway is `target/standard-cargo/release/kuno`; the new ABI 2 flat
runtime is `target/inference-tests/runtime-flat-resident`, and its verified
archive is `target/inference-tests/kuno-sycl-resident.tar.gz`, all within the
resident worktree. All 63 vendor library/license members are unchanged from the
approved beta.2 closure. Never package the private SYCL worker as the standard
gateway. Existing ABI 1 installs use the one-shot fallback until the optional
runtime is upgraded; ABI 2 requires the new gateway.

**Installed acceptance:** the prepared standard gateway and ABI 2 runtime were
installed transactionally through `install-image-runtime --yes` with the user's
sudo authorization. Configuration was preserved byte for byte: B70 image PCI,
model store/collections, Ollama workers/listeners and disabled inference deadlines
are unchanged. Installed hashes match the build receipt; all 63 vendor
libraries/notices are unchanged from beta.2. Tests used swap-disabled transient
clients and the installed no-swap service, with the development SDK hidden.
Live commands from the resident worktree were `sudo -n /usr/bin/python3 target/inference-tests/install-resident.py`, `run-resident-http.py` and
`run-resident-final.py` (the latter two with the same interpreter/prefix and
`target/inference-tests/` path). The wrappers run private clients in collected
systemd units with `MemorySwapMax=0` and `LimitCORE=0`; final audit is
`resident-audit.py`. Harnesses and safe numeric receipts remain in that ignored
directory for operator reproduction.

- Intel 512-square/four-step Qwen 2.1 generation retained one process and about
  15.62 GB VRAM: 40.604 seconds including initial load, then 2.241 seconds warm.
  Two-reference editing (CFG 4, two steps) reused it in 25.329 seconds; changing
  to 640x384/CFG 4 and back to the original recipe also reused it. Editing
  increased model allocations to about 20.33 GB. Previews and bounded PNGs
  passed integrity/dimensions/no-metadata checks; outputs remained in RAM.
- Metadata and a small LLM on the 3060 preserved the image process. The LLM's
  matching digest was resident on the configured 3060 worker, reporting
  2,427,128,381 VRAM bytes equal to its runtime size. A different LLM that could
  not be prepared on that alternate worker evicted the image before B70 load;
  it reported 2,156,354,599 VRAM bytes equal to runtime size. Both returned
  successful inference. Catalog file size is not proof of loaded footprint.
- Reload after LLM handoff passed. Disconnect after a live preview cancelled and
  reaped the image worker in 1.756 seconds; a new request loaded successfully
  in 40.100 seconds without recovery intervention. Repeating cancellation
  passed. Restarting the service with an idle model proved its old PID gone.
- Switching to native Qwen 2512 and back replaced the old processes normally.
  Two 512-square/two-step/CFG 5/shift 3.1 legacy requests reused one Vulkan
  process in 8.691 and 8.562 seconds, retaining about 23.8–24.2 GB VRAM.
- Final service is active, unpaused, with one idle Qwen 2.1 worker retaining
  15,620,767,744 VRAM bytes, zero automatic restarts and no recovery latch.
  IPv4/IPv6/LAN version/discovery pass; keyless image readiness is `ready`,
  unauthenticated controls remain 401. No new Xe reset/wedge/hang events appeared.
- Content-free final audit verified zero swap, core limit zero, worker
  `NoNewPrivs`, no writable regular-file handles, no test prompts/images,
  control credential or catalog model names/digests in Kuno's journal since the
  install window. The private SDK is restored; tests retain no images/prompts.

**Limitations:** this is small synthetic residency/transport acceptance, not a
new ComfyUI/full-workflow performance comparison, useful edit-quality result,
overnight soak or arbitrary two-model inference-peak guarantee. Intel repeated
pixels differed by at most one 8-bit channel level: mean 0.003344 after mixed
jobs versus 0.003307 for an immediate repeat (about 0.33% of channels). This
supports rounding-scale nondeterminism in these samples, not a universal
fidelity/tolerance guarantee or proof of its specific kernel. CUDA smoke was
bitwise repeatable. Native cleanup removes reusable request state, without
claiming forensic zeroization of every third-party freed allocation.
Four classified backend diagnostics occurred during the initial legacy Vulkan
run even though generation/reuse/family switching passed; their internal cause
is unproven. Harness assertions for the wrong DRM counter name, changing
KiB/MiB units and unitless zero were corrected; they did not require backend
changes. Actual per-client GPU allocations and device totals agree.

**Release handoff:** integrate the complete branch range from `4f9b465`, including
behavior through `4fb4e58`, source/build receipt `418569e` and this installed
receipt. The main checkout retains an uncommitted pointer to the final receipt
commit. The new standard binary and ABI 2 runtime must be packaged together;
see [the release handoff](kuno-release-handoff.md#resident-image-update).
No public contract or Vashti connector/UI changes are required. This receipt
records authorized local installation, not publication or application/Android
acceptance. No merge, push, release or task message was performed. Preserve the
worktree and ignored producer artifacts until next-release integration.
