# Kuno and Vashti Coordination

This is the shared handoff between the **Build Kuno inference helper** task
(Kuno backend) and the **Vashti** task (frontend and application integration).
Repository contracts and decisions should survive switching tasks; neither task
should depend on having the other's entire conversation history.

## Ownership

| Area | Default owner |
| --- | --- |
| `apps/kuno`, `crates/kuno-image`, `crates/kuno-text` | Kuno task: inference, resource management, runtime, installation |
| `apps/vashti/src`, including `images` and the Ollama connector | Vashti task: authorization, persistence, application API, Kuno client |
| `apps/vashti/web`, `apps/vashti-android` | Vashti task: UI, mobile workflows, native integration |
| `crates/kuno-protocol`, workspace dependencies, cross-app specifications | Coordinate changes and identify affected consumers |

The user can explicitly reassign work. This division should not prevent a small,
agreed integration fix, but Kuno work should not independently redesign Vashti's
UI or change its user-data storage.

## Source of Truth

- [Kuno specification](kuno-spec.md): privacy, compatibility, admission, lifecycle.
- [Kuno development guide](../apps/kuno/README.md): operation and validation.
- [Native text research proposal](kuno-text-engine-research.md): pinned FreeToken
  and Strata findings and proposed Qwen3.8 engine phases; no implementation claim.
- [Native text implementation/benchmarks](kuno-text-performance.md): fixed
  synthetic updated-Ollama baseline and Rust engine development receipts.
- [Native causal prefill plan](kuno-native-text-prefill.md): internal MAX8
  transaction, recurrent/attention boundaries and bounded expert waves;
  isolated operators do not establish whole-model batching.
- [Actual native model connection plan](kuno-native-text-model-batching.md):
  approved complete constructor, original model owners and all-family chunk
  acceptance; source adapters do not establish a connected device consumer.
- [Native decode optimization plan](kuno-native-text-decode.md): original packed
  Rust CPU experts, positional reads and later measured CPU/mixed-GPU trials.
- [CPU/GPU calibration lab](kuno-native-text-cpu-gpu-calibration.md): approved
  bounded real-read operator, original-rank merge and completed joint controls;
  source/device/timing acceptance remain separate.
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

**Latest application acceptance:** see the beta.3 preparation/publication
receipts at the end of this guide. Kuno's handoff is merged into the normal
project checkout; all four beta.3 artifacts are published and
full-download/signature verified. Live Hub
and nginx upload limits are now verified at 1 GiB. Physical Android acceptance
is still outstanding.

**New Kuno backend work:** the latest owner handoff is on
`codex/kuno-image-speed` through `ebd1925`, now merged into this checkout.
See [the release handoff](kuno-release-handoff.md).
Read its **Selected Intel performance update (2026-10-02)** before the next
release. It supersedes the earlier `fb2d1d0` residency artifact selection.
Public contracts are unchanged; package the selected standard gateway and
managed ABI 2 runtime together. Integrate the entire backend range after shared
base `4f9b465`, not only the optimization commit. Preserve/commit pending Vashti
work before merging.

**Model-only image residency:** see the resident-image receipt
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

### Installed Kuno resident-image handoff (2026-10-01)

This residency receipt is historical. The active handoff above selects the
newer `codex/kuno-image-speed` closure and replaces this artifact selection;
retain the following evidence, but do not package the earlier runtime.

The user approved retaining image models between requests. Backend work is
isolated on `codex/kuno-image-residency` through installed acceptance commit
`fb2d1d0e367d4d53054bf8de63f49b316046d704`, based on `4f9b465`, in
`/home/jmicko/.codex/worktrees/kuno-image-residency/vashti`.
Read that worktree's updated `docs/kuno-images.md`, `docs/kuno-managed-sycl.md`,
coordination receipt, `docs/kuno-release-handoff.md` Resident image update and
`docs/kuno-image-residency-build.json` for the exact contract/artifacts/tests.
The public API is unchanged; Vashti needs no connector or UI change. The new
private runtime uses ABI 2; the published beta.2 ABI 1 runtime remains one-shot
through the new gateway's upgrade fallback.

Source checks passed 205 tests (nine ignored), default/native Clippy, formatting,
eight packaging regressions, and native CPU cleanup tests. CUDA live smoke
confirmed retained GPU weights and normal idle teardown. Authorized local
transactional installation preserved configuration byte for byte. Installed
Intel generation/edit/CFG/resize, previews, LLM handoff, cancellation/reload,
legacy-family reuse and final service/privacy checks pass. The B70 retained
15.6–20.3 GB; small 512-square/four-step generations took 40.604 seconds including
initial load and 2.241 seconds warm. This is a synthetic residency smoke, not
ComfyUI parity or full-workflow performance acceptance. Intel repeats show tiny
rounding-scale pixel drift; see the detailed limitations and legacy diagnostics.

Kuno is active, unpaused, with the Qwen 2.1 model loaded, zero automatic restarts,
zero swap and no recovery latch or new Xe kernel events. No prompts/images were
saved by the engine or test clients. Preserve this uncommitted coordination edit;
no backend merge, push, release or Vashti/Hub code change was performed here.
Integrate the complete resident branch and package its new standard gateway and
ABI 2 archive together for the next beta; do not reuse beta.2's worker. Application
and Android/release acceptance remain with the Vashti task.

### Vashti compatibility and gallery receipt (2026-10-03)

Reviewed the complete owner diff from `4f9b465` through `ebd1925` and its
selected Intel performance handoff. Shared protocol types and public image
discovery/generation fields are unchanged. Installed keyless discovery reports
all three ready workflows, Standard-only attention, and NDJSON generation.
Vashti's connector already enforces advertised attention/reference limits and
handles bounded incremental previews/results, terminal errors and disconnects;
no connector migration or inference deadline was added.

The gallery now uses small square photo tiles, stack-count overlays, adjustable
density, an original-image viewer with its actions, and append-only page loading.
Gallery PNG thumbnails are derived in RAM with a separate bounded cache and
decode queue; originals and encrypted device storage are unchanged. Owner-scoped
server and device galleries use the same UI. Older stacks are loaded across
session-result pages rather than silently limited to the newest 100 results.

Application tests pass: 140 Rust and 44 frontend tests, including the installed
discovery shape, stream/error handling, thumbnail cache identity/eviction/queue
cleanup and gallery pagination. Production web and native-frontend builds pass.
Browser checks cover phone/desktop density and overflow, original-image and
keyboard version viewing, flow navigation, editing from a chosen variant,
removal confirmation/cancel and the empty device-only gallery. No GPU generation,
Kuno installation/restart, physical Android export, release, merge or commit was
performed in this application pass. Integrating the backend source and matching
release artifacts remains a separate release task.
### Detailed resident image backend receipt (2026-10-01, historical)

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

### Vashti beta.3 preparation receipt (2026-10-05)

The pending reference-viewing/comparison feature is committed as `e67dd40`.
The complete Kuno owner range after `4f9b465` through `ebd1925` is merged into
`codex/image-flow-refinements` as `18935a8`, preserving both applications'
coordination records. Vashti and Android are aligned at `0.13.0-beta.3`, with
Android version code 13003. No public image contract or migration changed.

All four release artifacts are prepared in
`apps/vashti/dist/release/v0.13.0-beta.3`. The standard Kuno archive uses the
verified `shared-rope-standard-v2` native source/build closure described in the
active handoff, built with the ordinary native-image package script and no
private Intel linkage. The optional archive is the exact approved ABI 2 bundle;
its manifest checksum is
`fc169dd2bf4418c2770940c85ec570f5b0a8c5ddaa21e32176cf839b36c88baa`.
Versioned notes include explicit runtime replacement/enablement instructions
and platform limits. Existing ABI 1 installs retain the one-shot fallback.

| Target | Bytes | SHA-256 |
| --- | ---: | --- |
| linux-x86_64 | 9841269 | `8518c965354ffce7c5892c0fdf6e70adc28335bc548c32e2ca6db1be43d56e70` |
| android-universal | 52322265 | `2e2326eafd8e06d148951f1b260fdcc85ad3c9bd08f16e23bb4ed10d166ee32e` |
| kuno-linux-x86_64 | 456471447 | `8ab008c9c36647b7b84b6c5d7eb2436c05ea8a27435db0937a8b53f021ad1cfc` |
| kuno-sycl-linux-x86_64 | 156247147 | `85d7aecb92e9e27af85f56bf5569d31feffd63d37955e7549ac081ab3e3e54a3` |

Checks passed: locked Rust suites (345 passed, nine hardware/privilege tests
ignored), frontend helpers (57), release packaging (four), Kuno packaging
(eight), managed builder fixtures (six), formatting, whitespace, default and
standard native-image all-target Clippy with warnings denied, and production
web/native frontend builds. The complete four-artifact publisher signing
dry-run and SHA256SUMS verification pass. APK signatures verify under the same
certificate as beta.2; all four Android ABIs and the new version/code are present.

The extracted Linux package started on an isolated loopback listener with a
fresh data directory, completed migrations, served beta.3 `/api/version`, and
rendered the bundled sign-in UI without browser errors. The smoke server and
temporary browser tab were closed. Installed Kuno keyless discovery still
reports three ready Standard-only workflows. No fresh GPU generation, installed
service change, physical Pixel acceptance, or Android export/gesture test was
performed in this release pass; the backend hardware and earlier application
browser receipts retain their stated scope.

The live Hub still lists beta.2 as its newest prerelease and v0.12.0 as stable.
A read-only disk check found an 8 GiB EBS volume with a 7 GiB root partition,
1,572,651,008 bytes available, and about 677 MiB of journals. Published artifact
metadata accounts for 1,493,187,697 bytes. Beta.3 adds 674,882,128 bytes plus small
metadata overhead; finalization hard-links the streamed upload files rather
than duplicating their contents. Recheck free space immediately before upload.
This does not authorize deleting releases/logs or resizing storage.

**Preparation checkpoint (superseded by publication below):** all local checks
were complete and a fresh one-time upload key was requested. At this checkpoint,
no upload, stable promotion, main-branch merge, Git push, Hub deployment or live
Vashti/Kuno installation had occurred.

### Vashti beta.3 publication receipt (2026-10-05)

Uploaded all four prepared artifacts together using the user's fresh one-time
key. Public verification covered the release listing/status, exact versioned
notes, VERSION, SHA256SUMS entries, all four prerelease manifests and their
Ed25519 signatures against the repository's compiled public key. Full streamed
downloads of all four artifacts match the local sizes and SHA-256 values in the
preparation table, totaling 674,882,128 bytes. The Hub orders checksum entries
by target rather than local filename order; the same complete entries match.

The prerelease channel now returns `v0.13.0-beta.3`; stable remains `v0.12.0`.
Published artifacts occupy 2,168,069,825 bytes according to the public release
metadata. The Hub service remains active; the post-upload root disk is 88% full
with about 856 MiB available. No release or log cleanup, storage resize, Hub
deployment, stable promotion, main-branch merge, Git push or live Vashti/Kuno
installation was performed. Physical Android acceptance and the backend
hardware/platform limits remain as recorded above.

### Kuno native text research proposal (2026-10-05)

The user requested research only on FreeToken and Strata. See
[the research and implementation proposal](kuno-text-engine-research.md), based
on integrated beta.3 commit `ae3e70d`, on `codex/kuno-text-research` in the reused
Kuno worktree. It pins both upstream repositories, the official Qwen3.8 model
configuration and an independent graph reference. The proposed first target is
the full 512-expert model on B70, followed by measured CPU/GPU expert scheduling,
heterogeneous execution and verified MTP if a compatible draft module exists.
The user's follow-up requests read-only reuse of Ollama's downloaded files and
default-native routing for verified models, with `ollama.<full-model-name>` as
the explicit managed-Ollama route. The research records the exact existing
Q4_K_M manifest, mixed tensor formats, approximately 120 GB artifact size and
the need for bounded host caches/disk-backed PLE. No duplicate weight download
is proposed. The Bonsai graph plan is unchanged; its intended compatibility
section now records this explicit alias exception to accidental collision
rejection. No native text architecture or alias is implemented.

Evidence is source inspection, model manifest/tensor metadata and read-only
hardware inventory. No runtime build/test, model-weight download, inference,
GPU workload, installation, service
change or performance verification was performed. No Kuno API, protocol,
configuration, consumer or UI behavior changes; **no Vashti work is required for
this research**. Exact endpoint/errors/capability fields remain to be specified
before implementation changes consumers. Implementation needs a separate
acceptance receipt and shared contract agreement. Keep request-derived router,
PLE and prefix state transient under the existing content-free privacy contract.

### Native Rust text implementation started (2026-10-05)

The user authorized the research plan's implementation and a goal to exceed
updated Ollama throughput. Rust inference correctness comes before performance
optimization; embedding an entire C++ graph does not meet this milestone.
Read [the updated Ollama baseline](kuno-text-performance.md). Ollama 0.35.1
automatically selected B70/Vulkan, producing 128 tokens in 100.346 seconds cold
(41.500 seconds load), 16.690 seconds warm with prompt reuse, and 25.935 seconds
warm with a changed prefix and no reported prompt reuse. Exact public synthetic
prompts, seed 424242, options, manifest and numeric receipts are committed with
the harness. This is one reference sequence, not a native speedup claim.

Development is on `codex/kuno-native-text` from `e9de15a`, with isolated loader
and operator worktrees. Initial scope is read-only borrowed GGUF access, pure
Rust quant decoding and scalar graph/state primitives. No native model catalog,
public API, default routing, Vashti connector/UI or deployed engine changes have
been accepted. Frontend work is not required at this checkpoint. Preserve the
explicit native-default/`ollama.` direction for later tested integration.

### Native Rust text correctness receipt (2026-10-05)

On `codex/kuno-native-text`, isolated from the normal Vashti checkout, Rust now
runs the downloaded Qwen4exp trunk directly through `crates/kuno-text`.
Foundations `c5e05b4`, operators `5cef124`/`d3d716c`, graph `8f6cff1`, tokenizer
`5f95495`, graph hardening/public fixtures `44cb36f`, and CPU pool `98b3112` are
committed. Read [the exact graph contract](kuno-native-text-graph.md) and
[performance/acceptance receipts](kuno-text-performance.md) before continuing.
The separate `apps/kuno/benchmarks/native-text` operator laboratory contains
Rust-authored packed Vulkan kernels, outside installed release dependencies.

Acceptance is deliberately narrow: native scalar raw/chat smokes reproduce
updated Ollama output, Rust tokenizer matches 75/80 baseline prompt counts,
all seven actual artifact types pass both-GPU packed numeric probes, and B70
full-graph GPU matrix execution reproduces the four-token raw smoke. 70 normal
Rust tests plus two explicit installed-artifact checks passed; cancellation,
bounded allocation, malformed file/tokenizer handling and state cleanup are
covered at the library level. This does not establish production worker privacy,
long-context/vision/tool/sampling/API acceptance or a native speedup. Model state
belongs to a sequence, cannot cross bindings, and is discarded on execution
error. No private request content or credentials appear in these receipts.

**Required Vashti changes: none at this milestone.** Native default/explicit
`ollama.` routing is still intended but unimplemented; current Ollama endpoints,
Kuno protocol/configuration, installed service and model list are unchanged.
Keep the existing managed scheduler and long-inference behavior when exposing
native text later. A tested public contract/capability and error handoff must
precede consumer integration. No installation or release is authorized by this
receipt; the user's current goal remains correctness followed by a measured
speedup and backend integration.

### Native text laboratory follow-up (2026-10-05)

On `codex/kuno-native-text`, bounded Rust graph backend hooks now support exact
full routed MoE and transactional GDN overrides. Cooperative packed kernels,
expert-bank planning/dispatch, and GDN parity pass on B70 and RTX3060; the short
four-token graph output still matches the scalar/Ollama oracle on both cards.
See [the detailed numeric/source receipt](kuno-text-performance.md#cooperative-reductions-expert-banks-and-gdn-laboratory).
Performance is still slower than updated Ollama. Full baseline, long context,
private-worker lifecycle and native default/`ollama.` managed routing remain
required. No API or Vashti consumer change is exposed by this backend laboratory;
there is nothing to integrate or release from this receipt yet.


### Native text resident-operator checkpoint (2026-10-05)

`codex/kuno-native-text` integrates read-only bound plan `2467ac7` and measured
transfer/cache/block-decoding experiments after `f8b2d2d`. The first full fixed
75-prompt/128-output Rust fixture completes but is still substantially slower
than updated Ollama. Both cards pass residual/full-span buffer erasure and
seven-step resident GDN operator parity. Read [the performance receipt](kuno-text-performance.md#transfer-reductions-and-first-full-public-native-fixture)
and [the resident ownership audit](kuno-native-text-residency.md).

This is backend-only laboratory work. It exposes no native model/capability,
changes no API/configuration or installed service, and requires **no Vashti
changes**. Native-default and explicit managed `ollama.` routing remain intended
but unimplemented. Runtime staging sanitization, the complete resident graph,
quality/options/lifecycle acceptance and a matched performance win remain
outstanding. Preserve normal Ollama behavior and long inference until the
verified engine is integrated. This checkpoint is not a release handoff.

### Native text connected resident graph follow-up (2026-10-05)

The isolated laboratory now connects GR/GDN/QSA/full all-ten MoE with a single
whole-token owner; PLE retains its exact Rust host oracle. Both cards match
four output tokens and pass cancellation after queued GDN/QSA, whole-arena
erasure, poisoned retry rejection and fresh-sequence recovery. B70 matches the
eight-token chat. Short decode overhead improves, but native remains slower than
Ollama. Long-position NVIDIA QSA numerical accuracy is being corrected without
loosening the oracle. Exact binary receipts and limits are in
[performance acceptance](kuno-text-performance.md#connected-resident-graph-acceptance-public-laboratory-2026-10-05).

Public weight wipe removal does not alter private-content cleanup. Runtime
upload/readback/uniform and scalar/INFO host copies still require checked
sanitization before real private requests. No native route/default/catalog,
public contract, deployment or Vashti code changes: **no Vashti integration is
required yet**. This remains development toward the active performance goal,
not a production/release or complete privacy receipt.

Connected resident laboratory checkpoint is committed as `2eec122`; subsequent
public rotary tables pass both-card QSA contexts1/4/5/2048/2049/2052 without
changing acceptance tolerance. See the updated performance receipt. The earlier
NVIDIA numerical failure is resolved; it was not a driver/GPU crash. This fix
changes no public contract or Vashti consumer and does not authorize deployment.


### Native text owned-I/O and admission checkpoint (2026-10-05)

Backend source/receipts are committed as `a27a7c3` on the isolated
`codex/kuno-native-text` branch, after `c75fdf2`. The standalone lab vendors
three exact CubeCL0.10.0 crates with retained licenses/source provenance; normal
production dependencies and installed services are unchanged. Read the updated
[resident ownership receipt](kuno-native-text-residency.md#owned-runtime-io-implementation-and-both-card-proof-2026-10-05)
and [performance limits](kuno-text-performance.md#owned-runtime-io-and-resident-admission-acceptance).

Both physical GPUs pass the fixed synthetic owned-I/O probe: audited complete
upload/uniform/readback spans, pending/retained lease rejection, runtime scalar
parity, public-cache preservation and full main-span erasure. Independent source
reviews found no blocker within that limited preowned Wgpu/Arena proof. The
connected public graph now has precharged complete pages, sealed public-vector
inventory and accounting that grants credits only after completed readbacks;
both GPUs pass bang/chat plus queued GDN/QSA cancellation, poisoned retry
rejection and fresh recovery. Intel's 72-slot option also passes chat; default
capacity remains unchanged. Tests: 27 runtime ownership, 128 native host/offline,
74 normal core with two installed checks ignored; strict host/native Clippy,
formatting and fresh-archive composed-patch/hash verification pass.

**Remaining:** complete connected private staging/host ownership, long fixture
and options/quality/lifecycle acceptance, true batched prefill and a matched
performance win. Native remains slower than updated Ollama. Opaque driver/
firmware padding, failed Drop and historical swap are outside owned-span proof.
Public receipts are fixed synthetic operator artifacts; no real request logs,
model downloads, copied/converted weights, new inference deadlines, device
reset, reboot or installation are part of this checkpoint.

**Vashti changes: none.** Native catalog/default/explicit managed `ollama.`
routing and public capabilities remain intended but unimplemented. No API,
protocol or consumer contract changed. This is a development checkpoint toward
the active goal, not a release/integration handoff. Keep the ordinary managed
Ollama and long-inference contract intact until native acceptance/integration.

### Native text connected ownership and matrix batching checkpoint (2026-10-05)

Source and public receipts are committed as `3de1bff`, after `9649c86`, on
`codex/kuno-native-text`. The isolated laboratory connects the resident graph to explicit
owned uploads, uniforms and readbacks, with separate 64 MiB main/runtime I/O
budgets and retirement shared across fresh owners of the same physical stream.
Both GPUs pass the fixed bang/chat fixtures, queued GDN/QSA cancellation,
poisoned retry rejection and fresh recovery. An opted-in inspection verifies all
1,286 main pages / 785,186,816 bytes after erasure for each of four owners on
each card. Read [the ownership and performance receipt](kuno-text-performance.md#connected-owned-io-full-page-inspection-and-multi-rhs-operators)
and [runtime ownership limits](../apps/kuno/benchmarks/native-text/runtime-ownership/README.md).

Skipping unused intermediate prompt heads preserves all 128 prior native output
IDs on B70. The long native output is not bit-identical to Ollama, and the
connected owned-I/O run is still slower. Token-level cleanup cadence passes
both-card acceptance but has no matched performance benefit yet. Expert
planning/mapping/LRU host allocations now wipe their full retained capacities.
Receipts are emitted only after explicit owner/executor cleanup succeeds.

The new packed multi-RHS operator passes 56 original-artifact numeric cases on
each GPU. Cached eight-vector ABBA samples are 3.05–3.71 times faster on B70
and 2.09–2.96 times faster on RTX3060 for the three tested matrices. This is
an operator result; causal layer-major prefill is not implemented. A separate
exact-order down kernel is numerically accepted but slower on both GPUs and
remains disabled. All new performance choices remain explicit lab options.

Checks pass: 169 native host/offline tests, 28 imported runtime ownership tests,
strict native/runtime Clippy, formatting, and reconstruction of the complete
18-file vendor patch from exact pinned archives. The separately accepted core
remains at 74 normal tests with two installed checks ignored. Full receipts
record public fixture settings, frozen executable hashes, ordered samples,
cleanup counts and limitations. Production worker/privacy, full quality/options,
long-context graph, native default/managed `ollama.` routing and a matched
whole-engine speedup remain outstanding.

**Vashti changes: none.** No public API, model catalog, configuration, installed
service or consumer changes are exposed by this laboratory. The native route
still requires a separately agreed and tested contract before integration.
No installation, release, GPU reset or reboot is part of this checkpoint.

### Native text GR/GDN batching and cache experiments (2026-10-05)

The next source/receipt checkpoint is `e378d32`, after `ecf3320`, on
`codex/kuno-native-text`. Read [the measured operator and cache limits](kuno-text-performance.md#batched-gr-operator-acceptance)
and [the internal causal prefill plan](kuno-native-text-prefill.md). These are
isolated compiled public fixtures, without real request support.

Both physical GPUs pass 24 original-weight GR cases and eight GDN recurrence
chunk cases with full owning-page and guarded I/O cleanup. Actual Qwen GDN
chunks match scalar GPU state/output exactly at counts1/3/5/8. Projections,
convolution, closing gate and full causal layer-major prefill still require
integration; no whole-engine gain is established by these operator checks.

The default-off segmented96 expert cache passes both-card Q5/Q8 boundary-slot
comparisons, partial-fill/post-enqueue cancellation, poisoning and cleanup. Full
B70 chat/cancellation/main-page audits also pass. Short compiler-idle ABBA is
near noise. A long96 candidate fails runtime cleanup after partial generation;
it writes no success receipt and the remaining ABBA samples are not run. The
64 reference and fresh Intel recurrent recovery pass. Underlying cleanup/GPU
error categories need diagnosis before longer96 acceptance or selection.

Token-level cleanup cadence and ordered public upload pass their specified
short acceptance but are slower in paired measurements and remain disabled.
An explicit teardown-order repair safely drains pending source staging before
duplicated main-page audits. All unchanged default choices remain in force.

Root checks pass202 native host/offline tests, strict all-target Clippy,
formatting and diff checks. Core and the pinned18-file runtime vendor patch are
unchanged from the prior checkpoint; their previously accepted checks are not
claimed as newly repeated here. Public receipts record exact executable/source
identities, ordered samples, failure exclusions and unit bounds with swap/core
dumps disabled. Longer96, whole-prefill, private-worker lifecycle, quality/
options, native catalog/default/managed `ollama.` routing and a matched Ollama
speedup remain outstanding.

**Vashti changes: none.** No production protocol, configuration, installed
service, deployment or release changes apply. Kuno admission is temporarily
paused only for exclusive operator GPU tests and must be resumed at the end of
that test window. No GPU reset or reboot is authorized by this checkpoint.

### Latest native-text owner checkpoint (2026-10-05)

**Current source/acceptance commit: `3e244d5`, after `1064a62`, on
`codex/kuno-native-text`.** This supersedes the outstanding callback diagnosis
and long96 acceptance limitation in the preceding `e378d32` checkpoint.
Read [the measured operator boundaries and callback repair](kuno-text-performance.md#mapping-callback-completion-repair-2026-10-05)
and [the full-prefill transaction plan](kuno-native-text-prefill.md).

Both GPUs pass24 causal GDN pointwise and24 causal QSA chunk cases at
counts1/3/5/8, including original arithmetic/status tolerances, future-group
exclusion, late nonfinite failure, immutable committed history, cancellation,
full owning pages and independent I/O cleanup. A QSA plane-index regression was
fixed before acceptance; checked/unchecked generated WGSL tests cover it.
The complete GDN/QSA projection wrappers, PLE batching and whole-layer-major
prefill remain under implementation. These isolated operators do not establish
an overall Ollama speedup.

Owned upload/audit/cleanup-read mapping now waits for callback delivery within
one shared requested deadline after native polling. This repairs a reachable
background-poller race without remapping, retrying, relaxing erasure or changing
ordinary inference waits. Content-free diagnostics expose only37 static first-
observed categories. The long96/6-GiB public75-input/128-output run now completes
with all128 prior native IDs and accepted I/O cleanup. It is one diagnostic run,
without full main-page audit or compiler-idle paired timing;96 stays disabled.

Default-off append-only INFO targets pass67-record/two-epoch numeric and
lease/cleanup fixtures at0/4096/8192 on both cards. Full chat and separate explicit
cancellation at0/4096 pass on both cards, including poisoned retry rejection,
fresh recovery and four1,286-page/785,186,816-byte complete owner audits per run.
No performance selection is made; oversized individual INFO has no separate GPU
fixture. The bounded expert-wave planner passes nine host fixtures preserving
all explicitly supplied ten routes/token; it is host-test-only, without actual
bank pinning or fullgraph consumption yet.

Root acceptance is231 native host/offline tests,39 pinned runtime ownership
fixtures, strict all-target Clippy, formatting and diff checks. The18-file vendor
patch reconstructs against pinned originals, SHA256
`e3aa774114789d47b6d14bc817a2e1f6da0a849020c0324492dd5c21cb2def4a`.
Exact synthetic receipts live under `apps/kuno/benchmarks/text/receipts`.
Full quality/options, production private-worker lifecycle, native catalog/default
and managed `ollama.` routing, matched cold/warm Ollama improvement and release
remain outstanding. Installed Kuno/Ollama are unchanged; admission was resumed
after testing. **Required Vashti changes: none at this checkpoint.** No GPU reset,
reboot, driver change, installation or publication is part of this handoff.

### Native text layer-composition source checkpoint (2026-10-05)

**Source candidate: `3a6c4cc`, after accepted checkpoint `3e244d5` and its
handoff `46be6b4`, on `codex/kuno-native-text`.** This does not supersede the
device acceptance above. Read [the composition source limits](kuno-text-performance.md#full-gdn-layer-and-fused-matrix-fault-source-checkpoint-2026-10-05)
and [whole-model chunk semantics](kuno-native-text-prefill.md) before connecting
these candidates.

The full candidate GDN layer includes all five original packed projections,
causal pointwise operations and recurrence. It has separate immutable current
and candidate state, all ten status planes and no state/cursor commit. Its fixed
public probe compares counts1/3/5/8 against the original scalar GPU schedule and
independent core oracle; those device comparisons have not run yet. Exact main,
public, host and I/O envelopes are in the source receipt.

The shared MAX8 projector reuses the existing original-weight cache, validates
full row extents before bounded tiles and preserves reserved-memory checks and
the separate whole-output fault fence. It is not yet connected to a whole chunk
owner or device-tested. The default-off cooperative projection candidate folds
only its output finite fault into the original64-lane arithmetic; block paths
keep the separate fence. Its probe rejects incomplete owning reads and wipes
byte leases on every exit. Seven formats pass14 checked/unchecked WGSL variants.

Combined host acceptance passes247 tests, strict all-target Clippy, formatting
and diff checks. Prepared device probes use frozen executable SHA256
`a1be34e63179b2ba1fdc33ad13c15fe27f8f456362f5cda69482dce3041f59b7`,
which predates the shared batch projector. Sudo expired before device units
started; normal Kuno admission remains enabled. No GPU/performance/private-worker
claim follows from these source checks. QSA/PLE layer composition, owned expert
waves, whole-model chunk connection and the full acceptance list above remain
outstanding. **Required Vashti changes: none.** No public API, catalog, default
routing, installed service, release or publication changed.

### Native text PLE/QSA and grouped-expert source checkpoint (2026-10-05)

**Current source candidate: `abf4396`, after `e43ea22`, on
`codex/kuno-native-text`.** Read [the layer source boundaries](kuno-text-performance.md#ple-and-complete-qsa-layer-source-candidates-2026-10-05)
and [the complete transaction plan](kuno-native-text-prefill.md). This does not
supersede the prior both-card device receipts.

Original layer1 PLE and layer3 QSA now have complete candidate wrappers and
fixed-public probes, with immutable committed histories, sealed geometry and
physical ownership, exact full owning readbacks and no state/cursor commit API.
PLE includes original16-branch u64 hashing and bounded110-byte Q5 row reads; its
new count1 GPU tail is labeled separately from the existing CPU tail. QSA
includes all six projections, complete-block-only causal selection and all17
status planes. Both-card numeric/cancel/cleanup runs remain pending. The frozen
combined lab is SHA256
`b7b2e40955d39829f71bf8a6dc09620b38f658ada3725dc311e77e13cf328983`; it
includes the earlier cache-backed batch projector and sealed GDN fields. Do not
substitute this identity into old `a1be34…` prepared-unit receipts.

The grouped all-ten expert operator is host/offline-only, supporting all seven
original packed formats and runtime protected maps with exact probability/rank
bits. It preserves the original64-lane arithmetic and reduces original ranks
only after all waves. The owning adapter to actual bank identity/revision,
protected union, partial-fill poisoning and completed consumer lifetime is still
being developed separately. No original bank ledger may publish before complete
model acceptance; routed-only probe acceptance is insufficient.

Root checks pass284 native host/offline tests, strict all-target Clippy, locked
offline release build, formatting and diff checks. Normal `kuno-text` passes75
tests; two installed checks remain ignored. Exact page/I/O/host budgets and
source hashes are recorded in the source-only PLE/QSA/grouped-operator receipts.
Opaque driver/register/spill and abrupt-loss erasure remain outside page audits.
Whole-model batching, quality/options, production private-worker lifecycle,
native catalog/default and managed `ollama.` routing, updated-Ollama cold/warm
speed improvement and release remain outstanding.

**Required Vashti changes: none.** No public API, configuration, installed
service, model download/copy/conversion or release changed. Kuno/Ollama listeners
responded with no resident models; admission remains resumed. Protected device
validation awaits sudo refresh. No GPU reset, reboot or driver change is part
of this checkpoint.

### Native GR ownership and protected expert waves (2026-10-05)

**Source commit `e17957f`, after layer source handoff `25246bc`, on
`codex/kuno-native-text`.** Read [the exact source checkpoint](kuno-text-performance.md#native-gr-workspace-and-owning-expert-wave-source-checkpoint-2026-10-05)
and the [whole-model prefill plan](kuno-native-text-prefill.md). This does not
supersede accepted device receipts or establish complete prefill.

The native GR workspace borrows original cached projections and public norms.
A sealed prepared capability remains alive through all residual consumers or
last-column output-head enqueue. Cancellation/error/abandonment poison it;
queued completion is explicitly distinct from the future whole-model fence.
Ten full MAX8 pages charge1,998,848 B including the mixed page. The new active
status shader guards before storage reads, with checked/unchecked validation.
Source peer review and the repaired final-head lifetime recheck pass.

The owning grouped MoE fixture validates real pending bank generations and
protects complete unions through at most ten misses/40 MiB per source window
and completed consumers. It retains staged ledgers privately and never publishes
the original routed bank ledger. Independent I/O stays64 MiB. Full-page source
credit requires owning lease destruction and checked cleanup; unexpected runtime
failure retires the context. Its40 positive,12 rejection and12 fresh-owner cases
per card remain physically untested. Shared expert/full FFN and the native adapter
using the full model's actual banks are not implemented by this fixture.

Root checks pass300 native host/offline tests, strict all-target Clippy, locked
offline release build, formatting/diff checks. Unchanged core acceptance is75
normal tests/two installed checks ignored. Frozen source binary is
`0c1b3f553ae24ee497f2f9637324778582feeeb89b9427e0c451d67d157438f5`,
`/tmp/kuno-rust-native-batch-owner-benchmark-20261005`. Receipts are
`rust-native-gr-batch-source-20261005.json` and
`rust-grouped-moe-chunk-source-20261005.json` in the text receipts directory.
Sudo remains expired; no GPU unit, service, configuration, API, catalog, route,
frontend or installed binary changed. Native GDN/QSA adapters, whole-model
transaction, matched Ollama improvement, options/quality and production private
worker/routing acceptance remain outstanding. No Vashti consumer change is needed
for this internal checkpoint; a future public native contract needs a separate
backend and consumer handoff.

### Latest native cached-layer and both-card checkpoint (2026-10-05)

**Source/device receipt commit `49cf147`, after `d918a10`, on
`codex/kuno-native-text`.** Read the
[cached-layer source boundary](kuno-text-performance.md#native-cache-backed-layer-adapters-2026-10-05)
and [both-card fixture acceptance](kuno-text-performance.md#both-card-layer-and-expert-fixture-acceptance-2026-10-05).
This supersedes the device-pending and sudo-blocked statements of the preceding
source checkpoints, while preserving their narrower fixture scope.

Frozen `b7b2e409…` passes GDN4 numeric/4 rejection, PLE8 numeric/its4 rejection
paths, and QSA24 numeric/4 rejection cases on each GPU. Frozen `0c1b3f55…`
passes40 routed-expert numeric/12 rejection cases and a fresh independent owner
after each rejection on each GPU. Complete reports, maxima, exact identities,
owned-memory bounds and erasure evidence are in the new device receipts. Tests
used verified user-unit swap/core limits; root authentication is no longer a
prerequisite for this laboratory path. Admission was not paused, CPU builds were
active, and unit runtimes are not benchmark measurements.

New actual-cache GDN/QSA adapters, shared-expert batch workspace and last-column
GR head are implemented and source-tested. Common sticky logical poison belongs
to the actual admission; pending Drop does not reenter its mutex. All-family
preflight and the completed whole-model fence remain mandatory before any state
or bank publication. These new adapters are not covered by the older fixture
receipts and have no selected whole-model consumer yet. Incremental full-page
and I/O components are recorded; they do not approve an aggregate allocation.

Root acceptance passes336 native host/offline tests, strict all-target Clippy,
locked offline release, formatting and diff checks. Optional finite-fault probe
`861b285a0dc581ec6c03c1d16dd7814269e33f1a103cbece43fc7ebd9416409e`
passes28 original-format cases and owning cleanup on each GPU after harness
initialization/nonfinite-upload fixes. It remains default-off. Exact source and
device receipts are `rust-native-layer-adapters-source-20261005.json`,
`rust-gdn-ple-qsa-device-20261005.json`, `rust-grouped-moe-device-20261005.json`
and `rust-fused-matrix-fault-device-20261005.json` in the text receipts directory.

Native actual-bank MoE/CPU-history PLE adapters, complete constructor/chunk
transaction, full adapter/device acceptance, matched Ollama improvement,
quality/options and production private-worker/native-default/managed `ollama.`
routing remain outstanding. Installed Kuno/Ollama services respond and remain
unchanged; no reset/reboot, API, catalog, route, frontend or installed binary
changed. No Vashti consumer work is required for this internal checkpoint.

### Current native actual-history/bank checkpoint (2026-10-05)

**Source commits `0f31146` and `09a9164`, after `49cf147`/`526f31d`, on
`codex/kuno-native-text`.** Read the
[actual CPU-history bridge](kuno-text-performance.md#native-ple-bridge-into-actual-cpu-histories-2026-10-05)
and [actual-bank/router receipt](kuno-text-performance.md#native-actual-bank-moe-and-guarded-router-2026-10-05).
This supersedes the adapter-pending statements of the preceding checkpoint;
complete model execution and device acceptance remain pending.

Native PLE uses the authoritative original CPU history objects and guarded
OLD/NORMALIZED/NEXT owning maps. Its opaque field-split preparation permits
other families and the required second actual fault/I/O/admission proof before
any commit. Original token/hash/convolution math remains unchanged. Native MoE
uses real original executor banks and retained final staged plans; complete
selected unions stay protected, scalar bank access is blocked during sessions,
and original ledgers publish only after complete-model acceptance and all-family
preflight. The guarded router preserves original rounding/ties/clamp and all
ten ranks, including zeros. Full-page, private-host and independent runtime
component bounds are recorded; no component alone admits the aggregate model.

Current root acceptance passes350 native and86 core host/offline tests, with
two installed checks ignored, strict all-target Clippy for both manifests,
locked offline release, formatting/diff and focused peer source review.
`rust-native-ple-bridge-source-20261005.json` pins the343/80 source checkpoint;
`rust-native-moe-router-source-20261005.json` pins the current350/86 checkpoint
The `27c25fea…` frozen-binary association in those source-only receipts is
invalid: an unrelated historical probe was selected. The forward correction
`rust-native-artifact-binding-correction-20261005.json` preserves that error and
selects the current combined model executable from actual Cargo metadata.
Source checks are unaffected; the wrong artifact established no device acceptance.
The approved complete constructor/chunk plan is linked above. Its actual
coordinator is being developed in an isolated snapshot. Owning embedding/router/
final-output/fault helpers are committed as `99f6e73`; composed root acceptance
passes361 native tests, strict all-target Clippy, locked offline release,
formatting/diff and focused peer review. Core86 tests are unchanged. The exact
helper source/file/runtime/host bounds are recorded in
`rust-native-batch-io-source-20261005.json`. Helpers add no MAIN pages; the
original source-only checkpoint preceded connection to the actual coordinator.
Do not infer source provenance from the identical old-probe artifact identity.

Separately, the original scalar model with optional finite-fault fusion passes
fixed-public four-output inference and cancellation/poison/fresh-owner cleanup
on both GPUs. Each card reproduces `[49276,2349,1074,5928]`; four owner audits
each inspect785,186,816 B/1,286 complete pages plus separately checked source/I/O
cleanup. `rust-fused-full-resident-device-20261005.json` preserves exact reports,
settings and binary identity. These protected user-unit runs are lifecycle
evidence, not matched performance measurements or new native MAX8 acceptance.
The exact protection wrapper is committed as
`apps/kuno/benchmarks/text/protected-lab-exec.py`; swap/core limits are checked
inside the unit before exec. Sudo is not required for this laboratory path.

The decode plan approves Stage1 original packed Rust CPU experts and a pinned-FD
positional reader for isolated synthetic source work. Actual calibration and
CPU/B70/secondary-GPU execution are later measured proposals. Whole-model native
adapter parity/lifecycle, complete prompt/output quality and options, faster
matched updated-Ollama cold/warm runs, production private-worker lifecycle and
native-default/managed `ollama.` routing remain outstanding. **Required Vashti
changes: none at this internal checkpoint.** No public API/configuration,
download/copy/conversion, installed service, frontend, release, driver, reset or
reboot changed. Installed Kuno/Ollama listeners respond with no resident models;
admission was not paused and all completed root GPU laboratory units are stopped.

### Current connected native model checkpoint (2026-10-05)

**Source `371b9d1`, after `bed3f96`/`99f6e73`, on `codex/kuno-native-text`.**
This supersedes the coordinator-pending statements immediately above. Read
[connected model acceptance](kuno-text-performance.md#connected-native-model-laboratory-2026-10-05)
and the linked original constructor/chunk and decode plans before further work.
Concurrent source work now uses managed Git worktrees from the agreed base;
older snapshots remain preserved but are not integration authorities.

The real48-layer native coordinator is connected. It charges all retained
original/new pages and original bank-state metadata before fill, executes
original causal/history/router math, completes two owning fault/I/O proofs and
all-family preflight before publication, and continues scalar decode from the
same accepted original model owners. Any failure terminally poisons that model;
explicit registry cleanup runs outside admission guards. Both independent
64-MiB I/O caps and existing GPU/cache/bank defaults are unchanged.

Root acceptance passes369 native and86 unchanged core tests (two installed
checks ignored), strict all-target Clippy, locked offline release and formatting.
The source receipt is `rust-native-model-max8-source-20261005.json`. Correct
Cargo-selected frozen executable SHA256 is `fca4b669…d61ca15`; the forward
artifact-binding correction preserves the unrelated historical probe mistake.
No device acceptance is attributed to the wrong probe.

Both GPUs pass native fixed-public bang/chat, original scalar output IDs,
pre-enqueue and queued native GDN/QSA cancellation, poisoned retry rejection,
fresh-owner recovery and four complete1,376-page/818,249,728-byte private audits
per run. `rust-native-model-initial-device-20261005.json` preserves the four
initial count8 reports and exact lifecycle observations. Counts/longer contexts,
full75/128 output, injected whole-model failures, options/quality, production
private workers, faster matched Ollama inference and native-default/managed
`ollama.` routing remain unaccepted. These short runs are diagnostic lifecycle
acceptance, not a performance claim.

The fixed-public protected runner binds the actual Cargo target/binary and checks
full receipt plus cancellation/cleanup completion; lab-only containment has no
bearing on production long-inference deadlines. Stage1 corrected Rust CPU expert
operators are separately source-reviewed and await root integration and real
CPU/GPU numeric/joint timing acceptance. **Required Vashti changes: none.** No
public contract, installed service/configuration, frontend, release, driver,
reset or reboot is changed by this internal checkpoint.

### Current packed Rust CPU expert checkpoint (2026-10-05)

**Root source `7950074`/`d1a0446`, after device/runner checkpoint `8280ec1`.**
Both corrected CPU commits are integrated; this supersedes the root-integration-
pending note above. Read
[CPU source acceptance](kuno-text-performance.md#original-packed-rust-cpu-expert-source-acceptance-2026-10-05),
the [crate API/limits](../crates/kuno-text/README.md) and the original decode plan.
`rust-packed-cpu-experts-source-20261005.json` pins root source/checks/budgets.

`GgufFile::positional_reader` borrows the already-inspected model/open descriptor;
`PackedProjection`/`PackedExperts` borrow that reader and original views.
Explicit `ExpertOperator` calls return guarded unweighted requested-rank values,
preserving all selected experts and original64-lane math. They are not called by
the native model or installed gateway. Source-only controlled CPU private storage
has an independent1-MiB cap; this does not admit a future whole-model aggregate.

Root passes105 core tests in both debug/release (two installed checks ignored),
369 composed native tests, strict all-target Clippy for both, formatting and
locked offline release. New composed lab binary SHA256 is `b1ba040f…5e8f589`;
prior physical native fixtures retain their separately verified `fca4b669…`
identity and are not attributed to the new binary.

Next source work is the approved bounded real-read one-layer CPU/GPU numeric and
joint-calibration lab in its own managed worktree based on `d1a0446`. It uses
actual original fills/consumers/completed fences, all-ten rank ownership and
one original-rank merge. Numeric acceptance precedes timing. Prepared-host CPU
timing is deferred: the current CPU API always performs positional reads, even
when OS file pages are warm. Root exclusively owns real model/GPU/service runs.
No CPU/GPU parity, hybrid model, speed win or production privacy claim follows
from these source checks. **Required Vashti changes: none.** Public HTTP,
installed services, drivers, release and frontend remain unchanged.

The short native partition matrix is now complete on both GPUs for counts
1/3/5/8, all original eight output IDs, four complete MAIN/source audits and
native GDN/QSA cancel/poison/fresh recovery. Exact reports remain pinned to the
earlier `fca4b669…` artifact in `rust-native-model-partitions-device-20261005.json`;
this is not newer CPU-binary or speed acceptance. The protected runner now also
accepts the already-compiled public75/128 rainwater fixture and compares against
committed original native IDs; full physical acceptance is still pending.

### Current full native fixture checkpoint (2026-10-05)

The root CPU-composed `b1ba040f…` artifact, actual `d1a0446` build snapshot
`3371e900…`, passes Intel's full75/128 public native fixture: all128 original
native IDs, scalar continuation, fresh full-prefix replay, queued native GDN/QSA
cancel, poisoned retry rejection and four complete1,376-page/818,249,728-byte
private audits. `rust-native-model-full-intel-20261005.json` retains the entire
report and exact identities. NVIDIA's full counterpart is now complete with
the same 128 original IDs, full-prefix fresh replay, native cancellation,
poisoned retry rejection and all four MAIN/source audits. Both its worker and
corrected outer runner exit 0; `rust-native-model-full-nvidia-20261005.json`
preserves that separate completed evidence under the same exact binary.

The first outer validator exited1 after the worker's success0 because it
mistook cumulative audited source bytes for simultaneous live usage. Source
arena reset accepts complete erasure/inspection and drops handles before adding
numeric counters.127 scalar continuations yield1,905 buffers/70,746,112 B
cumulatively, under the unchanged independently checked64-MiB live cap. The
forward-corrected runner requires exact original per-step cumulative counts;
root revalidated the completed original report and preserved both outcomes.
No engine limit, cleanup or numerical behavior changed. Full-logit/options/
quality/longcontext/fault/production-worker and speed acceptance remain pending.

The approved one-layer real-read CPU/GPU calibration plan and exact bounds now
live in `docs/kuno-native-text-cpu-gpu-calibration.md` and
`rust-cpu-gpu-calibration-plan-20261005.json`; implementation is isolated from
root GPU work. No model/private/default/API/installed/Vashti change follows
from that source plan. **Required Vashti changes: none.**

### Current actual CPU operator checkpoint (2026-10-05)

The integrated `d1a0446` CPU source now has actual original-weight numeric
acceptance on the Ryzen 5900XT. The fixed-public
`crates/kuno-text/examples/packed_expert_public_probe.rs` executes original
Q4 gate/up with Q5/Q8 down, experts 0/7, and explicit AVX2 physical plans
1/4/8/16. All 16 cases bit-match an independent scalar 64-lane row decoder for
raw gate/up and complete unweighted partials; inactive rank tails are zero.
Portable single-worker complete partials also match. Final file consistency
and the protected swap/core-limited user unit pass. Strict example Clippy and
locked offline release pass; unchanged libraries retain their 105 core/369
native acceptance. Exact source/build/public result evidence is in
`rust-packed-cpu-experts-actual-20261005.json`.

This supersedes the CPU-device-pending note for these limited public CPU cases;
all-ten CPU/GPU parity, completed joint timing and whole-model hybrid execution
are still pending. The new CPU helpers are not called by the native model or
installed gateway. No faster-than-Ollama or production private-worker claim
follows. Additional actual model context/failure hooks and the one-layer joint
calibration runner are isolated source candidates, awaiting root review and
physical acceptance. **Required Vashti changes: none.** Public contracts,
installed services/configuration, drivers, defaults, frontend and release are
unchanged; no reset/reboot is required for this lab work.

### Current whole-model failure fixture source (2026-10-05)

Root `19e12af`/`cdb3524` integrates the independently reviewed actual lifecycle
candidate and its explicit protected lab entry. Read
[the source checkpoint](kuno-text-performance.md#actual-model-failure-and-context-fixture-source-2026-10-05)
and `rust-native-model-lifecycle-source-20261005.json`. Twenty-two serialized
real owners exercise context4/8 partitions, exhaustion, malformed chunks,
late-combine error, real PLE-ticket error/panic and cancellation after the first
actual GDN commit, both retry rejections and fresh recovery. The fixture preserves
original scalar preflight exhaustion and explicitly closes it; native failure
must itself close. No state import, fake position, extra MAIN or relaxed
whole-model numerical tolerance is introduced.

Root passes 376 native tests, strict all-target Clippy, locked offline release,
format/diff and three missing-protection invocations rejected before device
creation. Frozen executable `110b9926…` retains exact Cargo/source provenance.
Physical both-card execution is pending. Cleaned full-logit mismatch is explicitly
unaccepted; process success alone cannot establish parity. The 1,800-second
protected lab containment has no effect on production long inference. Actual
long-context boundaries, options/quality, private worker/native routing and
matched Ollama speed remain pending. **Required Vashti changes: none.** Installed
services, drivers, public API, defaults, frontend and release remain unchanged.
