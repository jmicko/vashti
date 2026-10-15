# Kuno and Vashti Coordination

This is the shared handoff between the **Build Kuno inference helper** task
(Kuno backend) and the **Vashti** task (frontend and application integration).
Repository contracts and decisions should survive switching tasks; neither task
should depend on having the other's entire conversation history.

**Current native-engine resume pointer (2026-10-14):** the isolated Rust rewrite
is through `8b503e2` on `codex/strata-rust-port`; read the
[prompt-only cache sources/controller handoff](#active-kuno-engine-handoff--prompt-only-cache-sourcescontroller-2026-10-14)
and [whole-rewrite gate](kuno-strata-rewrite-completion.md). It is incomplete;
no live speed benchmark before the connected source and correctness gates.
This backend work does not change the published image/application contract.

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
- [Native text checkpoint](kuno-native-text-checkpoint.md): current
  integrated source, isolated diagnostic candidate, completed checks and the
  next physical acceptance gates. Resume from this checkpoint first.
- [Native disk-access research](kuno-native-text-disk-access.md): intended lazy
  n-gram lookup, actual source paging and the accepted long public diagnostic;
  aggregate disk reads alone do not establish cache thrashing.
- [Fully resident Ornith speed lab](kuno-native-ornith.md): user-selected
  Qwen3.5 MoE diagnostic before further Flash-Next offload work; no API change.
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

### Current both-card model lifecycle result (2026-10-05)

The preceding physical-pending note is superseded for the fixed context4/8
fixture only. Frozen `110b9926…` completes all 44 actual owner cases across both
cards, real late error/panic/post-first-GDN-commit cancellation, retry rejection,
full original scalar/native MAIN and source audits, fresh recovery and physical
retirement=false. Both worker units exit 0. Exact reports are in
`rust-native-model-lifecycle-device-20261005.json` and
[the performance guide](kuno-text-performance.md#both-card-actual-lifecycle-acceptance-and-numerical-rejection).

All 24 final greedy comparisons equal the scalar reference, but full logits
have small deltas (Intel maximum about1.04e-5; NVIDIA about8.70e-6). Whole-model
bitwise numerical acceptance remains explicitly false and both outer runners
exit 1 after clean completion. No whole-model tolerance was relaxed; a source
investigation is pending. These simultaneous diagnostic runs make no speed
claim. Existing full128-output acceptance is a separate, narrower receipt.
Long contexts, options/quality, production private workers, native-default/
managed `ollama.` routing and faster matched Ollama inference remain outstanding.
**Required Vashti changes: none.** Installed/public/frontend/release state is
unchanged; no reset or reboot occurred.

### Paused native text source checkpoint (2026-10-05)

**The user paused the goal.** Read [the clean checkpoint](kuno-native-text-checkpoint.md)
first on resume. Root `codex/kuno-native-text` integrates the joint expert source
through `e386c79` (`5f2ed09`, `b2b69cc`, `1ef151f`, `e386c79`). Root passes 391
host/offline tests, strict all-target Clippy, formatting and diff checks. The
focused independent review of the corrected reference ordering found no
material blocker. No root CLI, physical joint numeric/timing acceptance or
whole-model hybrid dispatch is established. Numeric precedes timing; references
in Calibration now follow completed measured consumers/merge/audit while all
numeric/file/immutability gates still precede local plan publication. Required
priming/prior samples remain cache conditioning.

The isolated first-divergence diagnostic is frozen at `2439d73` on
`codex/kuno-native-numeric-boundaries`, based on `8621308`, with five focused
tests and strict Clippy/fmt/diff checks passing. It is not integrated or
registered on root. Source review passes; extra peer review and both-card actual
four-boundary/full-cleanup evidence remain pending. No arithmetic or whole-model
tolerance changed; the cause of the measured full-score deltas is unaccepted.

Resume with the checkpoint's exact candidate/source receipts and physical gates.
Preserve the 8-GiB correctness lane; separately address benchmark containment
and cache accounting before a matched Ollama comparison. No new model/GPU job,
installed service/configuration, default, public contract, driver/reset/reboot,
frontend or release change occurred while finishing this checkpoint.
**Required Vashti changes: none.** All current source workers and checks have
finished; goal completion and a speed win remain outstanding.


### Native text resumed numerical and host-limit diagnostics (2026-10-06)

The user resumed after the `~/dev` NVMe migration, then requested a break with
one Intel comparison already running. The latest checkpoint is
[the native text checkpoint](kuno-native-text-checkpoint.md); it supersedes the
historical source-only candidate status above. Finish only the declared Intel
comparison and pause; that comparison has now completed, with no next device job
started and no native lab job remaining. Resume requires the user's instruction.

On `codex/kuno-native-text`, root integrates `195d8a5` (author candidate
`2439d73`), named protected CLI `962c37d`, exact numeric validator `80aef9d`
(author `36d2121`) and the dedicated host-limit diagnostic `5ffffa4` (author
`13844b5`). Root passes 396 host/offline tests followed by the additional focused
CLI guard test, strict all-target Clippy, formatting/diff checks, and 22 focused
Python tests. The original correctness lane still requires actual 8 GiB;
performance uses a separate actual 8/52 guard with unchanged model/cleanup
validation. No guard output or numerical tolerance is rewritten.

Actual Cargo-selected/frozen binary `73ce13b2...` completes all eight fresh
scalar/native owners on each card, full owning maps and explicit audit/erasure,
with worker and validator exit0. Both cards have exact projected KEY/VALUE;
Intel's combined layer0 input differs by at most 7.45e-9 and NVIDIA's is exact.
Injected residual maxima are 1.49e-8/5.96e-8. KEY/VALUE depend on separate token
embeddings, so equality does not explain residual differences. PLE has definite
scalar CPU-f64/native GPU-f32 differences; source inspection alone cannot
attribute Intel's earlier residual drift. Whole-model bitwise and performance
acceptance remain false. Exact reports/build identities and the pre-GPU initial
launch-path failure are preserved in
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-device-20261006.json`.

The existing Intel public comparison uses one persistent user cgroup, one frozen
binary, native count 8 original 75/128 fixture and MemoryMax 8/52/52/8 GiB,
with swap/core0,
full cleanup/cancellation and numeric memory/refault/pressure/process-I/O evidence.
It does not flush global caches or change production limits. NVIDIA's comparison
is not started. All four Intel samples pass with zero OOM and launcher/supervisor
exit0; the unit finishes inactive after 1,046.203 seconds, swap peak0. Decode times
for 127 following steps are 128.939/80.438/63.189/130.799 seconds, respectively
(0.985/1.579/2.010/0.971 tokens/s). The 8-GiB limit repeatedly reclaims model file
pages; 52-GiB warm weight reads are much shorter while upload times stay similar.
Cache order still varies across samples. Exact public receipts/identities and
telemetry hashes are in
`apps/kuno/benchmarks/text/receipts/rust-native-host-limit-intel-20261006.json`.
This remains a host-containment diagnostic, not true hot-full-request or matched
Ollama performance acceptance. Correctness containment remains 8 GiB.

No production private worker, public API/catalog/native-default or `ollama.`
routing, install/configuration, driver/reset/reboot, frontend or release change
is made. **Required Vashti changes: none.** Long contexts/options/quality,
physical CPU/GPU expert calibration, private-worker/scheduler/catalog integration
and a faster matched Ollama comparison remain required goal work.

### Selected port takeover, CLI separation and intentional offload (2026-10-06)

The user wants `kuno run MODEL` to use Kuno while retaining ordinary
`ollama run MODEL` for direct Ollama inference. The original Ollama executable
has not been replaced: its default listener is currently Kuno on 11434, while
the original Ollama daemon is loopback 11435. A new command name alone does not
separate these routes.

**Selected direction:** keep Kuno on the adopted Ollama API listener (normally
11434) and put original Ollama on private loopback 11433 where available. The
user prioritizes unchanged API URLs. This supersedes the separate-default-port
proposal from `501f49e`; separate-listener manual setup remains an option.
`kuno` inference commands would select their own endpoint (explicit endpoint,
`KUNO_HOST`, installed configuration, then the normal Kuno API default),
independently of `OLLAMA_HOST`.
Convert wildcard bind addresses to usable client addresses. Mirror common
inference/model commands without changing the Ollama executable. Native selection
still requires a verified registered adapter; the paused text prototype is not
production-ready. Inference access remains keyless and separate from privileged
installation/control operations.

Use a Kuno-owned systemd drop-in for the original Ollama daemon's private bind;
do not edit/replace the vendor executable or base service file. Its service
environment does not configure terminal clients. Plain `ollama` still selects
its own `OLLAMA_HOST` or defaults 11434. **The later selected shell-export
contract below supersedes the command-scoped CLI helper proposal.** Applications
inheriting and honoring the export may also connect directly to Ollama; the
user accepts that behavior. Exact installation/activation support across shells
remains pending.

This host still uses original Ollama 11435. Preserve its actual recorded port
until an approved, tested migration; 11433 is not assumed free. Migration must
retain Kuno's listener/settings, image runtime, workers and state with rollback
and existing interruption-confirmation requirements. Clients keep their Kuno
URL. Ollama upgrades may still interrupt requests using that backend; there is
no guarantee against a later override explicitly changing its bind.

Direct Ollama commands bypass Kuno reservations. Treat that work as external GPU
consumption; observation alone cannot guarantee serialization of independent
direct Ollama and Kuno jobs. The requested CLI separation does not silently grant
Kuno control over ordinary Ollama commands.

**Approved direction:** support intentional mixed CPU/GPU offloading. The current
installed `worker_cpu_spill_detected` comes from the fully GPU-reported residency
requirement, not an upstream model error. There is no current per-model offload
opt-in; removing only that one check would leave other strict checks and placement
assumptions unresolved. The exact policy/configuration and resource-accounting
contract needs a reviewed implementation. Preserve protection against unexpected
spill for models admitted under GPU-only policy. The main specification now
records this intended capability, explicitly labeled unimplemented.

**Status:** proposal/requirement checkpoint only. Read-only listener/config/source
inspection passes; no CLI, API, runtime, installation, services, GPUs or client
settings changed and no implementation tests are claimed. The native performance
goal remains paused. Required Vashti endpoint changes: none for the selected
takeover layout. A tested backend CLI/offload/installation handoff must still
precede any application feature integration. Shared request schema and discovery
changes are not yet selected or implemented.

### Clark systemd-only CLI port experiment (2026-10-06)

The user authorized a temporary service-port change on Clark to test whether
ordinary Ollama CLI commands inherit the daemon's systemd environment. Clark
ran original Ollama 0.32.13, with no shell `OLLAMA_HOST`, alias/function wrapper,
service drop-in or Kuno listener. The original listener was 127.0.0.1:11434;
`ollama run lfm2.5-thinking` opened normally before the change.

A temporary runtime systemd drop-in set only the daemon's
`OLLAMA_HOST=127.0.0.1:11439`. After restart, `/api/version` and the actual
listener confirmed 11439; 11434 was closed. In the unchanged SSH shell,
`ollama run lfm2.5-thinking` failed with `could not connect to ollama server`
and exit status 1. The command-scoped positive control
`OLLAMA_HOST=http://127.0.0.1:11439 ollama run lfm2.5-thinking` opened normally
and exited with status 0 after `/bye`. No generation prompt was submitted.

Restoration removed only the temporary drop-in and restarted Ollama. The
original 11434 listener and version returned, 11439 closed, the base unit's
SHA-256 remained unchanged, and the service stayed active/enabled. Both service
and shell again had no `OLLAMA_HOST`. The restored plain CLI opened normally
and exited with status 0 after `/bye`. The independent automatic rollback timer
and temporary experiment state were removed after successful restoration.

This live test confirms separate daemon/client endpoint configuration on this
installation. It does not test model output quality or native performance.
Only this diagnostic and restoration were performed; Marge's installed ports,
application APIs and the paused native inference goal are unchanged.
**Required Vashti changes: none.** The selected CLI/port migration above remains
unimplemented.

### Selected persistent Ollama shell export (2026-10-06)

The user selected the ordinary Ollama configuration approach after the restored
Clark test: export `OLLAMA_HOST` in the selected user's shell startup, using the
actual recorded private Ollama client endpoint. This supersedes the shell
function/CLI-only helper recommendation, while retaining the selected Kuno
takeover layout. Do not replace or wrap the original Ollama executable.

The service's Kuno-owned systemd override sets the daemon's private bind. The
separate shell export sets the CLI target, for example
`export OLLAMA_HOST=http://127.0.0.1:11433` after that listener is verified.
Existing installations keep their actual private port until a tested migration;
this host's last recorded private port is 11435. Record the enrolled non-root
user so root/admin upgrades do not silently choose another account. Publish a
marked installer block only after endpoint verification, retaining prior user
settings while supplying the selected effective host. Update/remove only the
unchanged owned block, preserve unrelated edits, and report conflicts. Give
activation/restoration instructions: already-running shells and apps keep their
environment until explicitly changed or relaunched. Other users and
independently configured services are outside that shell export's scope.

Applications inheriting and honoring `OLLAMA_HOST` may bypass Kuno and use
original Ollama directly. **The user explicitly accepts this.** Apps explicitly
targeting Kuno's adopted listener continue to use Kuno. Direct Ollama work still
bypasses Kuno reservations and is external GPU consumption. Kuno's planned CLI
uses its own endpoint selection (explicit endpoint, `KUNO_HOST`, installed
configuration, default), independent of `OLLAMA_HOST`.
Never use `OLLAMA_HOST` as a Kuno endpoint fallback; normalize wildcard Kuno
listener addresses to usable loopback client addresses.

**Status:** contract/documentation update only; no shell exports, services,
installation, CLI, API or application behavior changed. The Clark connection
receipt above verifies command-scoped targeting, not persistent shell setup.
Supported-shell activation, installer upgrade/uninstall and CLI endpoint
separation still require implementation and tests. Documentation diff checks
pass; the native inference goal remains paused. **Required Vashti changes:
none** when its connector explicitly targets Kuno. No API schema change.

### Installed CLI separation and shell export receipt (2026-10-06)

The selected contract above is now implemented and installed on Marge. Backend
commits are `f1c2848` (independent Kuno compatibility client), `942f965` (persistent
Bash enrollment) and `686bc46` (installer/upgrade/restore integration, owned CLI
launcher, new-port selection and current-upstream version verification). These
are the focused CLI changes after decision `9e04bc1`; this receipt does not
accept or resume the native text prototype/performance work.

**Exact installed routes:** Kuno retains `[::]:11434`, private original Ollama
retains `127.0.0.1:11435`, and privileged control retains `127.0.0.1:11432`.
The enrolled `jmicko` Bash startup exports
`OLLAMA_HOST=http://127.0.0.1:11435`. Existing API URLs are preserved. New
installations prefer private 11433 when available; no existing-port migration
was performed. The original Ollama executable is unchanged.

`kuno` is available through an installer-owned `/usr/local/bin/kuno` launcher.
Its inference/model commands invoke the original Ollama client on `PATH`, using
prefix `--host`, then `KUNO_HOST`, then the listener from prefix `--config` or
the installed configuration, then the 11434 default if no configuration exists.
Wildcard listeners become loopback. `OLLAMA_HOST` never selects the Kuno route.
Streaming, arguments, terminal cancellation and exit status remain the client's
own behavior; no inference deadline is added. History defaults off; explicit
Ollama local save/history/editor actions remain inherited client features.

Install/upgrade preflight the recorded non-root account and startup ownership
before interrupting inference, then publish the managed Bash blocks only after
verified service activation. Explicit root installation can use `--cli-user`;
later upgrades reuse the enrolled account. `sudo kuno configure-cli [--user
USER]` repairs/enrolls without restarting services. Restore preflights and
removes unchanged owned blocks, retaining prior settings and unrelated edits.
Enrollment records contain only identity/owned blocks, never copies of user
startup contents. A late publication failure reports active inference plus
retry guidance; it is not mislabeled a service rollback. Shell file publication
rechecks snapshots immediately before replacement but cannot guarantee atomic
comparison against an uncooperative concurrent writer.

The old setup record contained Ollama 0.34.2 while the owned live server was
0.35.1. Upgrade health now compares the owned live gateway/private endpoints,
retaining executable/listener/environment ownership checks and the historical
record. An ordinary Ollama update no longer causes a stale-version rejection.
New artifact conflicts (including a pre-existing CLI launcher) are rejected
before stopping services.

**Validation:** `cargo test --locked -p kuno -p kuno-protocol` passes 216 tests,
with seven environment-dependent tests ignored. This includes 17 shell tests,
four CLI unit/six CLI integration tests, 13 deployment tests and current-version
health regression coverage. Strict all-target Clippy, formatting and diff
checks pass. The standard native-image release build passes against the
preserved verified CUDA/Vulkan closure; ELF checks reject private Intel
dependencies and RPATH/RUNPATH.

The prepared/installed binary SHA256 is
`6c247d9001619117f1cadd2a37499ccd97974c8806d83d0aac47b2f9b4583469`.
The normal transactional upgrade succeeds with both services active/enabled.
Semantic configuration, both disabled inference deadlines, original Ollama
binary and all 65 image-runtime members are unchanged. Fresh interactive/login
Bash exports pass; installed `list`, `ps` and the public Qwen model's `show`
pass for both commands. Live connection traces verify original CLI port 11435
and Kuno CLI port 11434. Repeated enrollment keeps startup file hashes and
service PIDs unchanged. Clark verifies Kuno's 0.35.1 upstream response through
`192.168.1.164:11434`. The current shared shell is activated too.

**Limits:** automatic shell setup is Linux/Bash only; existing processes retain
their environment until explicitly changed or relaunched. Applications
inheriting and honoring the export intentionally bypass Kuno scheduling.
Installed uninstall was not performed; restore/crash handling is covered by
isolated tests. No live generation, GPU benchmark, new image acceptance,
intentional-offload implementation, native text registration or public release
is claimed. **Required Vashti changes: none** for explicitly configured Kuno
connectors; inference API schemas and access controls are unchanged.

### Native text baseline variability follow-up (2026-10-06)

The user's interactive timing counters show decode rates of 3.58, 10.11 and
3.98 tokens/s for unequal-length responses, with reported pauses in the longest
response and separate warm observations up to 12.68 tokens/s. These are
unreproduced observations; only numeric aggregates, never private prompts,
outputs or topic summaries, are retained in the
[measurement update](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06).
The original 7.6–8.0 tokens/s fixture receipt is not a general Ollama ceiling.

The next comparison must match actual memory allowance and placement, distinguish
weight/file-cache warmth from KV prefix reuse, and include repeated short and
long public synthetic requests plus numeric within-stream gap and cache/I/O
telemetry. The existing client does not yet capture those gaps. Qwen's sparse
PLE lookups and expert routing make cache behavior worth measuring; they do not
prove the cause of these stalls. The benchmark fixture/receipts are unchanged.

**Status:** documentation-only benchmark-plan/checkpoint update; no engine,
API, installation, service or GPU operation. The native goal remains paused.
Diff checks pass; no runtime tests are claimed. **Required Vashti changes: none.**

### Native optimization priority amendment (2026-10-06)

The user superseded the expanded Ollama benchmark proposal above: **use 10
decode tokens/s as a working target and skip further/repeated Ollama baseline
runs**. Concentrate remaining usage on correct Rust-engine optimizations and
native before/after evidence. Historical receipts remain unchanged; 10 tokens/s
is not a newly reproduced Ollama measurement. Numerical, privacy, cleanup and
long-inference constraints remain in force. Native performance observations
must retain exact public fixture/seed, actual binary/source/settings, host
allowance and output counts; an operator win is not a whole-model win.

See the [amended checkpoint](kuno-native-text-checkpoint.md#current-priority-amendment-2026-10-06)
and [measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06).
The goal was still paused when recorded. Only documentation and a read-only
next-candidate source review are authorized preparation here; no GPU jobs,
production behavior, installation, API or application changes are claimed.
Diff checks pass. **Required Vashti changes: none.**

The user subsequently resumed the goal and refreshed sudo. Root now integrates
through `22a2413` on `codex/kuno-native-text`: protected native continuation,
public projection/profile controls, single-token MoE and GDN, actual queued
cancellation/recovery, narrow full-page CPU/GPU lab banks and bounded fixed
calibration and explicit Q4 header reuse. 426 host/offline tests and strict
all-target Clippy pass; 62 Python control tests passed at `3f38840`.
Both-card operators and `single-all` eight-token model/lifecycle
checks pass without a useful short-run speed gain. Both-card warmed original-read
CPU/GPU calibration completes12 numeric gates and24 samples per card with a
narrow operator benefit; see the
[current owner checkpoint](kuno-native-text-checkpoint.md#later-owner-acceptance)
for exact artifacts, failed-run evidence and limitations. No full-model throughput
win, private native worker, default routing or automatic placement is accepted.
The `22a2413` Q4 binary passes52 routed/14 packed-format cases on each card,
including all six paired generic/Q4Block comparisons with observed zero maximum
absolute error. Complete-model Q4 acceptance and throughput remain separate.
The count1 CPU-miss/GPU-hit plan at `dc440d8` is a proposal; a staged one-layer
numeric adapter and batched full I/O inspection are being developed in isolation.
Root owns shared
GPU/service operations; isolated agents perform source work. No further Ollama
baseline, API/Vashti change, installation or release is part of this work.
**Required Vashti changes: none.**

### Native upload-erasure acceptance follow-up (2026-10-06)

The subsequent internal `4d4ecbe` admission seam runs the same mandatory full
audit once after map lease drop, then retains ordered-source release, health,
telemetry and MAIN fence credit. Both-card short model/cancellation/recovery/
full erasure acceptance passes with439 native host tests and strict Clippy.
The [checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance)
pins the normal `3b192df…` binary and exact proofs. The longer Intel75/128
fixture now passes output/lifecycle/full-erasure gates at1.311 decode tokens/s
(127steps/96.865s), below the10-token/s target; it is not a matched performance
comparison. Root source through `97d50bc` passes450 native host/offline tests
and includes the explicitly reserved count1 CPU-miss seam. That seam remains
unexposed and physically unaccepted, with CPU joins before GPU enqueue and no
overlap/speed claim. Ordinary inference, APIs, installed services and Vashti are
unchanged. **Required Vashti changes: none.**

The internal `200b38a` upload-wipe batching candidate passes the same both-card
operator and complete eight-token native model/cancellation/recovery/full
erasure inventories as the audit-only candidate. Runtime fault tests now pass49;
native tests remain436, Python controls81 and strict Clippy/provenance pass.
See the [checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance)
for exact `3ce1dbd…` binary and numeric receipts. Intel short generation
observes4.916s for seven steps; NVIDIA9.418s, with no additional NVIDIA speed
gain established. Matched long-run throughput, single audited admission fence
and CPU-miss model integration remain separate. This internal lab change
requires no API, frontend, installation or release changes.
**Required Vashti changes: none.**

### Native audit/staged acceptance follow-up (2026-10-06)

Root integrates through `793c351` on `codex/kuno-native-text`. Full audit
copy/read-map batching and the bounded original-read staged CPU-miss/GPU-hit
numeric adapter pass both-card operator and eight-token model/lifecycle gates.
The exact `91d4dd9` binary,436 native/45 runtime host tests,81 Python controls,
strict Clippy, numeric inventories, erasure/cancellation/recovery receipts and
short timing observations are recorded in the
[current checkpoint](kuno-native-text-checkpoint.md#batched-audit-and-staged-cpu-miss-acceptance).
Intel/NVIDIA short seven-step generation observes5.307/8.838s, with unchanged
outputs. This does not establish matched long-run throughput or the10-tokens/s
target. Upload erasure batching and explicit count1 CPU-miss model connection
are still separate source candidates. Shared GPUs/services remain root-owned;
no production, API, install, release or frontend changes are accepted.
**Required Vashti changes: none.**

### Paused native optimization checkpoint (2026-10-07)

The user intentionally paused further work to conserve usage after the active
block. Root `codex/kuno-native-text` source through `d13e95d` passes459 Rust
host/offline tests,99 Python tests and strict Clippy. The exact `0741420` /
`3098e9cb…` dense binary passes both-card91-case operators and complete short
model/cancellation/recovery/full-erasure gates. No additional throughput gain
is established. Grouped count1 grid trimming and the experimental CPU-miss
input-tail fix are source-tested but physically unaccepted; no hybrid CLI or
native-default capability is exposed. See the
[clean owner checkpoint](kuno-native-text-checkpoint.md#clean-paused-checkpoint-2026-10-07)
for exact commits, artifact/proof paths, limitations and preserved next proposals.
No GPU/compiler job is left running; installed inference admission is resumed.
No installed binary/configuration, API, release or application change occurred.
**Required Vashti changes: none.**

### User-authorized paused-goal optimization pass (2026-10-07)

The goal remains paused; the user explicitly authorized this additional turn.
`feb76b3` records both-card grouped count1 finite-grid acceptance. `79ba326`
moves owned original public GGUF weights into the ordinary upload queue without
redundant Vec copies; private data and ordered-source APIs are unchanged.
459 native/49 runtime tests, strict all-target Clippy, formatting and full
18-file patch reconstruction pass. The exact `9d4f4277…` normal binary passes
both-card52 routed/14-format/12-failure operators and complete short model,
fresh continuation, cancellation, poisoned retry/recovery and owning erasure.

The complete compiler-idle Intel52-GiB Q4 ABBA also passes all four75/128 fixture
and lifecycle gates. Generic mean1.4052 and Q4 mean1.4258 decode tokens/s give
only a small1.47% observed rate change; the10-tokens/s target is not met.
This comparison does not isolate the upload copy change, repeat Ollama baselines
or prove hot cross-request reuse. Preserve the dated receipts/artifact hashes,
physical postflight and next CPU-miss model gates in the
[latest owner checkpoint](kuno-native-text-checkpoint.md#latest-owned-public-upload-and-long-comparison-2026-10-07).
This supersedes the previous finite-grid source-only status, not the unaccepted
whole-model CPU-miss/default routing status. All lab jobs are stopped, admission
is resumed and both inference listeners respond. Intel physical VRAM is back
to50,688,000 bytes used; NVIDIA399MiB includes the desktop. No install, driver,
hardware reset, API/routing, application or release change was made.
**Required Vashti changes: none.**


### No-root native CPU-miss registration pass (2026-10-07)

The goal remains paused; the user authorized source/host work without sudo.
`a8cb5e0` on `codex/kuno-native-text` exposes only the protected fixed-public
whole-model `resident-count1-hybrid-chat` lab, with cap0/10, actual accepted
CPU-miss counters, fresh continuation and layer0/3 cancellation policy checks.
Its separate validator requires four1378-owner/818413568-byte audits and exact
original output/lifecycle proofs; ordinary1376-owner validation is preserved.
461 native host/offline tests,106 Python controls, strict all-target Clippy,
formatting, a normal release build and four actual predevice rejection cases
pass. Exact `1c44a37a…` artifact/host receipts and sudo-dependent next numerical,
hybrid late-fault and performance gates are in the
[latest checkpoint](kuno-native-text-checkpoint.md#no-root-cpu-miss-model-registration-checkpoint-2026-10-07)
and [registration contract](kuno-native-text-count1-hybrid-plan.md#protected-whole-model-registration-2026-10-07).
Whole-model hybrid device execution and speed remain unaccepted; CPU joins
before GPU enqueue, so no overlap claim. No GPU test, sudo, installed service,
API/catalog/routing, application or release change was made.
**Required Vashti changes: none.**


### Both-card short CPU-miss acceptance (2026-10-07)

After sudo was restored, `ff3c7d1` corrects a recursive admission mutex call
found in the first unaccepted hybrid constructor trial. Its borrowed original
construction proof preserves locking, binding/poison and all page checks.
463 native host/offline tests, strict all-target Clippy, formatting and a normal
`4382edf2…` release binary pass. Cap0/10 protected short whole-model tests pass
on both GPUs, including actual CPU work, exact eight output IDs, fresh next
output, layer0/3 cancellation, poisoned retry/recovery and all four1378-owner
zero audits/source0. Intel short decode stays5.440/5.448s; NVIDIA observes
9.056/6.849s (24.37% shorter). These single-order seven-step pairs do not
establish repeated/long throughput or the10-token/s target. Full-logit and
hybrid late-finality gates remain separate. Exact commits, failures, receipts,
remaining gates and postflight are in the
[latest checkpoint](kuno-native-text-checkpoint.md#both-card-short-hybrid-acceptance-and-constructor-correction-2026-10-07).
Admission is resumed, both live inference listeners respond and Intel physical
VRAM recovered with no classified Xe fault. No hardware reset/reboot, installed
binary/configuration, API/catalog, Vashti or release change was made.
**Required Vashti changes: none.**

### Pooled CPU-worker short acceptance (2026-10-07)

Source408abb9 adds explicit protected-lab worker reuse with unchanged math and
defaults.110 core,464 native and107 Python tests plus strict Clippy pass. Eight
compiler-idle scoped/pooled ABBA short samples pass on both cards with full
original output/lifecycle/erasure proofs: mean duration improves7.22% B70 and
8.31%3060. Full-logit, late-finality and long acceptance remain outstanding;
no10-token/s or production claim. Exact artifact/source/receipts and postflight
are in the [checkpoint](kuno-native-text-checkpoint.md#pooled-worker-short-abba-acceptance-2026-10-07).
Admission is restored and physical Intel memory recovers. No installed binary,
configuration, API, release or Vashti change. Required Vashti changes:none.

### Joined CPU/GPU overlap short checkpoint (2026-10-07)

Source44fd9bc adds explicit protected-lab pooled CPU/parent-GPU submission
overlap with bounded joined ownership and unchanged final math/publication.111
core,465 native and108 Python controls, strict Clippy and normal build pass.
Eight same-artifact short serial/overlap ABBA samples on both cards pass output,
actual CPU work, fresh/cancel/poison/recovery and owning-erasure gates. Observed
mean duration differences1.96% B70 /3.40%3060 fall within short sample variation;
overlap is not selected as a reliable speed win. Full-logit, late-finality and
long gates remain. Exact commits/artifact/receipts, source-contract details and
postflight are in the [checkpoint](kuno-native-text-checkpoint.md#joined-overlap-short-abba-checkpoint-2026-10-07).
Admission resumed, both inference listeners respond and Intel physical memory
recovers without classified faults. No installed service/configuration, public
API/catalog, application or release change. Required Vashti changes:none.


### Paused fully resident Ornith speed checkpoint (2026-10-07)

The user requested core execution speed work, then selected the already
installed `ornith:35b` to isolate it from Flash-Next offloading. On
`codex/kuno-native-text`, `3047d24` adds the Rust/WGPU resident Qwen3.5 MoE
lab; `c174aa9` completes its optional BPE flag/numeric reference handling.
The exact `6917d349…` frozen release artifact completes two75/128 public
sequences at13.9408/13.9184 decode tokens/s, versus Ollama83.3857 warm.
Only the first14 output IDs agree, so full-model numerical parity and Ollama
parity are not accepted. `c071019` prepares an explicit packed-block Q4/Q6
candidate, disabled by default and physically untested. Core118/native477,
the new offline shader inventory, Python118 and strict Clippy pass.

See [the current checkpoint](kuno-native-text-checkpoint.md#fully-resident-ornith-checkpoint-2026-10-07),
[Ornith contract](kuno-native-ornith.md) and their exact numeric receipts,
source/artifact identities and limitations. Earlier integrated Flash-Next
remains1.691tokens/s; neither target completion nor installed native routing
is claimed. The user requested a pause to use the GPUs and Vashti: all lab
jobs ended, GPU memory recovered, admission resumed and both inference
listeners plus Vashti respond. Goal remains paused. No installed binary,
API/catalog/configuration, application or release change occurred.
Required Vashti changes: none. Resume core arithmetic/performance work before
extended lifecycle/API/application work; do not repeat Flash-Next baselines.

### Resident text speed acceptance and Next transfer (2026-10-08)

The user returned the GPUs and authorized the approximately80-token/s resident
Ornith target. Backend source `000bdbe`, `fbddf49`, `3d8973e`, `5a93d1b`,
`26829f6`, `4a06b83` and `ffe1cf0` adds original-row probes, timestamp profiles,
explicit validated-geometry bounds selection, vector packed arithmetic, cached
parallel expert selection, EOS/teacher diagnostics and a shared Next operator
port. No public API, model registration, installed binary or configuration changed.

Complete B70 Ornith runs reach85.35/85.38 tokens/s on two fresh75/128 sequences,
82.6–83.0 on longer EOS-finished public prompts, and86.00/85.27 after the shared
kernel refactor. Every128-token native continuation matches the original scalar
path. Independent original-row and router checks pass. Teacher forcing128 public
Ollama tokens gives121 top1 and7 rank2 matches; free-running Ollama parity and
quality/indefinite stability are not claimed. The fast modes remain opt-in in
the fixed-public laboratory. Source shader suites and strict all-bin Clippy pass.

The original Flash-Next Q4 gate/up, Q8/Q5 down, F32 router and Q6 head slabs plus
512/top10 routing pass shared-kernel operator checks. These operators are not
connected to complete Next inference at that checkpoint: its last whole-model measured path remains
1.691 tokens/s. The user's next milestone is approximately10 whole-model Next
tokens/s before speculative FreeToken/Strata work. Avoid repeated Ollama
baselines or API/lifecycle/frontend work in this speed phase.

See [the current checkpoint](kuno-native-text-checkpoint.md#resident-speed-acceptance-2026-10-08),
[Ornith acceptance](kuno-native-ornith.md#resident-speed-acceptance-2026-10-08)
and [numeric aggregate](../apps/kuno/benchmarks/text/receipts/ornith-resident-speed-20261008.json)
for exact source/artifacts, timings, numeric tolerances and limitations. All
accepted GPU runs enforce swap0/core0 and explicit normal private-buffer clearing.
These are public fixtures; no real prompts, output text or routing trace was
persisted. Required Vashti changes: none.

At postflight all lab jobs exited, GPU memory recovered and Kuno admission was
resumed without an active request, queue or recovery latch. Both inference
listeners and Vashti respond. The broad goal controller remains paused and
incomplete; this receipt completes the directly authorized resident speed block.

### Complete Next shared execution receipt (2026-10-08)

The user authorized complete Flash-Next transfer after the resident Ornith
speed result. Kuno owner source `775d985`, `fc1e89e`, `d351bd6`, `62a52ba` and
`1a80d5b` connects shared packed dense/router and vector recurrent arithmetic,
batching of distinct privately owned GPU parameter records, exact CPU reduction
in SIMD registers, and optional direct caller device dispatch. All work stays
in the isolated native laboratory on `codex/kuno-native-text`. The current
[owner checkpoint](kuno-native-text-checkpoint.md#complete-next-shared-execution-acceptance-2026-10-08)
and [count1 contract](kuno-native-text-count1-hybrid-plan.md#complete-model-shared-execution-acceptance-2026-10-08)
supersede the preceding operator-not-connected and1.691-current statements.

Complete B70 inference now measures3.3473 tokens/s with standard banks and
3.5006 with96 slots, using original Ollama weights, seed424242/temperature0
and unchanged128 native output IDs. The selected frozen normal Cargo release
artifact has explicit `direct-device` feature, source `1a80d5b`, SHA256
`fe49069954829b2f8514b889b66967564937a67fdaabffe60a58418d437b8ec2`.
Its127 following steps take36280ms; prefill65978ms is slower than the standard
sample's48254ms. The10-token/s target remains unmet. Single ordered samples
do not establish statistical or matched Ollama speed; none was repeated.
The CPU-Q8 trial fails original-ID validation and is excluded from selection.

119 core tests (3 ignored),479 normal/481 direct native tests, two additional
offline shader suites each,50 imported runtime tests,22 hybrid Python controls,
strict all-target Clippy, formatting and provenance pass. Accepted physical
fixtures pass startup arithmetic probes, fresh continuation, existing
cancellation/poison recovery and complete1378-owner/818413568-byte normal MAIN
zero audits/source0, without context retirement. All runs enforce swap0/core0.
No real prompts, outputs or route traces enter persistence. Exact changed
commits, artifacts, all13 samples, tests, rejected/profiling limits and postflight
are in
[the numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-shared-speed-20261008.json).

No installed binary/configuration, API/catalog/default routing, application
or release change occurred. **Required Vashti changes: none.** Production
private workers, arbitrary quality/full-logit/long-context/soak acceptance,
mixed-card execution and the10-token/s milestone remain outstanding. Next
work should focus on CPU expert execution and remaining GPU dispatch/readback
handoffs. Cycle profiling and page-cache counters identify candidates, not
proven wall-time causes; the full notes retain those distinctions.

All lab jobs exited; admission is resumed with no active/queued request or
recovery latch. Kuno11434/direct Ollama11435 and Vashti7771 answer HTTP200,
both inference listeners have no resident model, and B70 physical memory
returns50688000 bytes. Same-boot Xe fault classifications remain zero. Goal
controller remains paused and incomplete; this directly requested trial is done.

### Active overnight Next offload optimization checkpoint (2026-10-08)

The user authorizes the isolated backend speed run to reach at least10 decode
tokens/s, or stop safely if sudo expires and effective GPU work is blocked.
Owner remains `codex/kuno-native-text` in the Kuno worktree. Through5ee51a5,
39 accepted public fixtures include diagnostic samples; current best v30 source
9fbda89 / frozen normal artifact3c7346fb119a96ccd631e2a96eec2c4a199d9fc5f46ffd6250311471ba310044
measures17724ms/127 following steps (7.1654 tokens/s). Target is unmet.
Original128 IDs, fresh continuation, poison recovery and complete owning/lifecycle
gates pass. B70 returns50688000B between runs; current classified Xe faults are
zero on the same kernel38 boot. Single samples are not statistical acceptance.

Current candidate changes only full-capacity heap erasure and guarded readback
copying, with496 native tests/two shaders,57 runtime controls/three allocator
controls/two logger controls and strict Clippy passing. Device acceptance is
pending. See the owner checkpoint and count1 contract for exact constraints.
The numeric aggregate `next-offload-overnight-20261008.json` is pending commit
until the run checkpoint is finalized. Admission is paused only during this
exclusive lab block; restore it and verify memory/listeners at an actual stop.
No installed binary/configuration, model blob, API/catalog/default routing,
Vashti application or release changes. **Required Vashti changes: none.**

### Overnight Next continuation: inline readback accepted

Ownerb345f9c/frozen3e707443f0452d00f9b4bc24608efbad406a2fde1f24bc58acc0eb6d658ef709
passes all original128 generated IDs, independent probes, fresh/poison/full
1378-owner cleanup and protected lifecycle checks.127 following steps17047ms
(7.4499912 tokens/s), prefill48856ms; single sample,10-token/s target still
unmet. B70 returns50688000B; same-boot classified Xe faults zero through
2026-10-08T14:50:40.509316+00:00.45 accepted fixtures (including excluded
profiles) are recorded in the pending numeric aggregate. The all-size GPU
Upload clear experiment is withdrawn after a deliberately stopped slow run;
ordinary volatile Upload erasure stays selected.

The next lab candidate partitions recurrent key reductions within a bounded
workgroup. Independent numeric/invalid-column probes and original unchanged
128 IDs remain required; device acceptance is pending. No installed services,
API, routing, frontend or release changes. Required Vashti changes: none.

### Current overnight optimization continuation

The Kuno owner remains `codex/kuno-native-text`. Following inline readbacks,
sourceaa3681c partitions recurrent reductions (full original acceptance, no
measured overall gain);07308b8 pairs existing expert-bank shards;ceb88d9 adds
a separately compiled, explicitly selected joined work-stealing CPU row mode;
7c40189 shares bounded original Q4 activations. Best accepted artifact
9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
completes127 steps in16247ms (7.8168277 tokens/s), prefill15464ms. All unchanged
128 IDs, independent arithmetic/guard checks and owning/fresh/poison/lifecycle
gates pass. Single samples only;10-token/s goal remains active and unmet.

The pending candidate uses fixed16-lane Q4 row groups and actual64-lane tail
coverage.498 native controls/two shader suites,128 core controls (three ignored)
in ordinary and work-stealing modes,39 Python controls and strict Clippy pass.
Complete candidate device acceptance is pending. The aggregate contains49
accepted fixtures, including excluded profiles. B70 returns50688000B; same-boot
Xe classifications remain zero through2026-10-08T15:25:46.933226+00:00.
No installed services/configuration, original model blob, public API/catalog,
Vashti application or release changes. Required Vashti changes: none.


### Active handoff: overnight native text run paused (2026-10-08)

This checkpoint supersedes pending-candidate/active-goal statements above.
Owner remains `codex/kuno-native-text` in the Kuno worktree. User paused for a
debrief; no further benchmark work is scheduled. Best accepted v44 source
7c40189/frozen9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
measures7.8168277 tokens/s (127 steps16247ms), prefill15464ms. Target10 remains
unmet. Fixed16 Q4 and the final active-GPU-group candidate pass original device
acceptance but are slower; both are withdrawn and native/core source is restored
exactly to v44. Aggregate `next-offload-overnight-20261008.json` records51
accepted public fixtures, including excluded profiles. Source tests, exact
commits, artifacts, acceptance limitations and restoration are in the owner
checkpoint. Single samples do not establish general speed or soak acceptance.

All benchmark jobs exited. B70 physical memory returned50688000B; same-boot
root Xe fault classification is zero through2026-10-08T15:45:04.155894+00:00.
Public Kuno admission is resumed, with no active/waiting request or recovery
latch. Kuno11434, direct Ollama11435 and Vashti7771 return HTTP200. Installed
binaries/settings, model blob, API/catalog/default routing and Vashti application
are unchanged. No release/install performed. **Required Vashti changes: none.**


### Active handoff: Astra native-text pass complete (2026-10-08)

This supersedes the preceding paused overnight checkpoint and pending native
candidates. Owner remains `codex/kuno-native-text` in the Kuno worktree.
Selected engine `79739d174af41990258a9571c1a8e720d77edc8a` measures
9.484–9.550 decode tokens/s across two full fixed-public fixtures, up from
7.817. The approved 10-token/s reference is still unmet; no new Ollama baseline
was run. Request-owned I/O reuse with completed Readback clears (`e9d8032`),
bounded INFO buckets (`406cf80`) and map polling (`a5a0d34`) supply the main gain.
Three completed recycle checkpoints and the original 512-entry INFO limit stay
selected. Scratch/full-copy cleanup (`9b51ace`, `708f5fe`) and byte-equivalent
zero-tail uploads (`fbae6f7`, `79739d1`) are retained; separate throughput gains
from those cleanups are not established. The user specifically selected the
zero-tail change: about 87% fewer encoded upload-copy bytes, after prefill only,
with no startup or measured hardware PCIe-traffic claim.

See [the final checkpoint](kuno-native-text-checkpoint.md#astra-pass-complete-selected-source-and-restored-host-2026-10-08),
[count1 contract](kuno-native-text-count1-hybrid-plan.md#request-scoped-zero-tail-upload-2026-10-08-retained)
and [14-fixture numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-request-io-reuse-20261008.json)
for exact source/artifact hashes, tests, acceptance and exclusions. Original
128 IDs, numerical/guard checks, fresh continuation, mid-token cancellation,
poison recovery, full owned-memory erasure and protected lifecycle checks pass.
Source gates: 498 native plus two shader suites; 131 core (three ignored) in
both feature modes; 90 runtime plus three allocator/two logger fixtures;
42 Python controls and strict Clippy. Broader quality, long-context, soak,
mixed-card execution and production-worker integration remain outstanding.

All lab jobs exited; B70 returned 50688000 bytes. Same-boot root Xe classifications
remain zero through `2026-10-08T18:16:29.880861+00:00`. Public admission is resumed
without active/queued work or a recovery latch. Kuno 11434, direct Ollama 11435
and Vashti 7771 return HTTP 200. Installed binaries/configuration, original
model blob, API/catalog/default routing and Vashti remain unchanged. No release
or installation performed. **Required Vashti changes: none.** Goal remains
paused; no further physical work is scheduled.


### Active handoff: private startup uploads and upstream review complete (2026-10-08)

This supersedes the previous selected-source handoff. Kuno owner source
`3b701d4d9bbeb794351502022b21c30734d501eb` includes runtime`909ab3b` and separate
bind/prefill transfer measurement. Zero-tail planning now covers all configured
private uploads, without extending request reuse or changing any erasure gate.
Public packed model-weight uploads are unchanged. Full original128-output and
owning/lifecycle acceptance passes: bind1365ms, prefill15324ms,127 decode steps
13237ms (9.5943 tokens/s). Prefill private copy payload falls75.12%; bind saves
no copy bytes in this fixture. No additional throughput/startup-time gain is
established, and the10-token/s reference remains unmet.

See [current checkpoint](kuno-native-text-checkpoint.md#private-startup-uploads-accepted-2026-10-08),
[contract and pinned upstream comparison](kuno-native-text-count1-hybrid-plan.md#updated-ollamallamacpp-comparison-2026-10-08-research-only),
and [numeric receipt](../apps/kuno/benchmarks/text/receipts/next-startup-zero-tail-20261008.json)
for exact source/artifacts, tests and limitations. Tests pass498 native plus two
shader suites,92 runtime plus three allocator/two logger fixtures,43 Python
controls and strict Clippy. Next recommended experiment is bounded request-local
reuse of identical GPU binding objects; not implemented or timed. Both wgpu and
llama already pool descriptor storage, so this targets repeated object creation,
validation and set allocation/update rather than missing pool support.

All lab jobs exited; B70 returned50688000B; Xe classifications remain zero
through2026-10-08T18:43:11.889831+00:00. Admission resumed without active/queued
work or recovery latch; Kuno11434/Ollama11435/Vashti7771 answer HTTP200. Installed
services, model blob, public API/catalog/default routing, Vashti and release
artifacts remain unchanged. Goal stays paused. **Required Vashti changes: none.**


### Active handoff: ranked native optimizations complete (2026-10-08)

This supersedes the startup-upload handoff and its pending binding experiment.
Kuno owner stays `codex/kuno-native-text` in the dedicated Kuno worktree.
All three user-approved candidates were tried; bounded request-owned binding
reuse `37fe3ca` is selected, with original GDN/float arithmetic restored in
`89ea1bc03ab22c1e834363d46c2736627d03b8da`. GDN retained-state trial `f73970f`
and existing three-/two-pass residual Q8 paths pass original acceptance but
provide no further timing improvement and are excluded. No new CPU arithmetic
source change, public API/catalog/default routing or production-worker change.

Selected normal locked artifact
`04f01a611209fb1bb526d48613deade7e2ec7413b9d6b69e7f03de1b9ea61d8b`, features
`direct-device,cpu-fma,cpu-work-stealing`, measures 10.6428/10.7300 decode
tokens/s on two full fixed-public Qwen3.8 Next fixtures; same-binary binding-off
control measures 9.5317. The user's 10-token/s reference is met on this fixture.
No new Ollama baseline or general quality/context/soak acceptance is claimed.

See [current checkpoint](kuno-native-text-checkpoint.md#ranked-optimization-pass-complete-2026-10-08),
[exact internal contract](kuno-native-text-count1-hybrid-plan.md#ranked-optimization-selection-2026-10-08-accepted-laboratory-only)
and [six-sample numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-ranked-optimizations-20261008.json)
for exact changed commits/artifacts, controls, tests and limitations. All six
fixtures pass original 128 IDs, numerical/guard/fresh/cancel/poison checks and
full owned-memory/lifecycle acceptance. Source gates: 498 native plus two shader
suites; 97 runtime plus three allocator/two logger fixtures; 44 Python controls;
131 core controls (three ignored) in the two-pass/work-stealing combination;
strict release all-target Clippy, formatting/provenance and whitespace checks.

All lab jobs exited; B70 returned 50688000 bytes. Same-boot Xe classifications
are zero through `2026-10-08T19:21:59.447749+00:00`. Admission resumed with no
active/queued work or recovery latch; Kuno11434/Ollama11435/Vashti7771 answer
HTTP200. No GPU reset, reboot, installed service/configuration change or release
performed. Goal remains paused. **Required Vashti changes: none.** This is an
internal standalone laboratory selection, not a production inference handoff.

### Active handoff: placement trials and secondary bank calibration complete (2026-10-08)

This supersedes the ranked-optimization handoff. Kuno owner remains
`codex/kuno-native-text` in the dedicated worktree; current engine source is
`34cdeaf71d8b29ffe186439a35e77b19be1150af`. Original intel144/cap10, float/FMA
and binding256 remain selected. Fused GPU projections, extra resident CPU work,
packed integer projections and binding512 provide no matched warm gain and
remain excluded lab opt-ins. Latest complete original128-output fixture passes
all numerical/owning/fresh/cancel/poison gates at10.9700 decode tokens/s, with
zero warm disk reads/writes and73.08% of expert calls served by B70 residency.

The user's next priority is complementary stationary experts on RTX3060 before
adaptive swapping. A protected original-expert calibration passes on that card;
residency avoids most upload cost but does not always beat the optimized CPU
path. **Full two-device execution is not implemented.** Proposed48 experts per
layer (disjoint from B70),32+16 shards/9GiB total, will need optimized native
kernels, concurrent scheduling, exact rank/source proofs, both-device lifecycle
and numerical acceptance, full-model timing and physical recovery. No private
expert histories/routes are persisted. Current warm results do not justify
adding16GB RAM. Broader quality/context/soak and production integration remain.

See [current checkpoint](kuno-native-text-checkpoint.md#placement-trials-and-secondary-resident-calibration-complete-2026-10-08),
[internal contract/results](kuno-native-text-count1-hybrid-plan.md#secondary-resident-calibration-results-and-next-placement-2026-10-08),
[ten-model-sample receipt](../apps/kuno/benchmarks/text/receipts/next-placement-kernel-trials-20261008.json)
and [operator receipt](../apps/kuno/benchmarks/text/receipts/next-resident-expert-calibration-20261008.json)
for exact commits/artifacts, failures, timings, tests and limitations. Source
checks pass502 native plus two shader suites,49+3 Python controls,98 unchanged
runtime-host controls, strict release Clippy and formatting/whitespace checks.

All lab jobs exited and physical memory recovered on both cards; no classified
Xe/NVIDIA faults this boot. Public admission is resumed without active/queued
work or recovery latch. Kuno11434/Ollama11435/Vashti7771 answer HTTP200. No GPU
reset/reboot or installed-service/configuration, public API/catalog, frontend,
release or installation change. **Required Vashti changes: none.** This remains
an internal standalone laboratory checkpoint, not a production handoff.

### Active handoff: stationary secondary model accepted, performance excluded (2026-10-08)

This supersedes the preceding pending two-device handoff. Kuno owner remains
`codex/kuno-native-text` in the dedicated worktree. Connected engine source is
`3e9f0e4891c75a09ff40eddd47f167e7ae47bef9`; stricter runner partition/provenance
validation is `bfdd986`. The complete range after `1e16862` implements the
secondary bank,32+16 geometry, status/phase admission corrections and counters.
Public API/configuration/catalog and Vashti consumers are unchanged.

The stationary48-expert/layer NVIDIA bank passes the complete original128-ID,
independent Q5/Q8 boundary-row, owning, fresh, cancel and poisoned-retry checks.
Both accepted secondary fixtures require five exact1835008B/nine-owner closes
and five independent row probes. Primary private budgets are unchanged; enqueue
and post-CPU-worker join are separately bounded phases. Frequency, selections
and partials are transient and erased; no private routing history is persisted.

**Keep secondary off.** Same-artifact v69 decode is11.0406 tokens/s for original
intel144/cap10 float/CPU FMA binding256, versus8.2586 with NVIDIA. Coverage rises
73.0791% to79.4767%, saving3900 CPU expert calls, but completion/readback and
runtime reuse add more cost than that saving. No warm disk reads/writes or
expert weight uploads. Extra RAM is not justified by these warm results.
Next secondary performance work needs cheaper bridges/checkpoints; more
residency alone is not evidence of faster inference. No new Ollama baseline.

See [latest checkpoint](kuno-native-text-checkpoint.md#stationary-secondary-model-acceptance-2026-10-08),
[exact internal contract](kuno-native-text-count1-hybrid-plan.md#stationary-secondary-model-acceptance-2026-10-08)
and [four accepted/six rejected numeric receipts](../apps/kuno/benchmarks/text/receipts/next-secondary-stationary-model-20261008.json)
for commits/artifacts, timings, failures and limitations. Checks pass505 native,
two one-test shader suites,50 Python controls, strict release all-target Clippy,
formatting and whitespace checks; runtime source is unchanged. This is a
standalone laboratory, with broader quality/context/soak and native-text
production integration still outstanding. No installation/release claim.

Normal admission is resumed; lab jobs/models are gone. Final same-boot snapshot
at2026-10-08T22:42:05.444314+00:00 reports B70 physical50688000/34242297856B,
NVIDIA373MiB desktop allocations without compute processes, zero classified
Xe/NVIDIA faults, andHTTP200 from Kuno11434/Ollama11435/Vashti7771. No GPU
reset/unbind/reboot/clock/driver or installed-service/configuration change.
**Required Vashti changes: none.** No shared contract or frontend handoff is
needed for this internal opt-in benchmark path.


### Active handoff: full adaptive cache awaits device benchmark (2026-10-08)

Historical; superseded by the adaptive settling/disk handoff below.

User corrected the 16-slot experiment: unused pinned experts must not prevent a
new working set from settling into GPU memory. Latest Kuno lab source
`3d67dbdd3c331b779779406a0fd177d3f459a4cd` makes all144 slots eligible, learns
prefill and accepted decode use, decays frequency with16-token half-life,
protects current selections and retains bounded grouped future-use transfers.
Earlier source commits: `b0c7652` bounded cache, `6a30bc7` empty slots,
`990a1a7` grouped fills/source hints. Read the exact internal contract in
[kuno-native-text-count1-hybrid-plan.md](kuno-native-text-count1-hybrid-plan.md#user-correction-all-slots-adapt-to-frequency-and-recency-2026-10-08)
and the latest top-of-file [checkpoint](kuno-native-text-checkpoint.md).

Full-cache host checks pass513 native, two one-test shader suites,52 Python
controls, strict release all-target Clippy and formatting. Device performance
and complete model/lifecycle acceptance are **pending**. Earlier16-slot paths
passed original gates but were slower than the matched fixed cache; they remain
historical comparisons, not a selected optimization. Native lab only; no
installation, release or public API/config/catalog changes. Required Vashti
changes: none. Normal admission is resumed; no GPU benchmark remains running.
The user requested a reasoning-mode switch before further GPU trials.

### Active handoff: adaptive settling and disk research (2026-10-08)

Historical; superseded by the exact replay handoff below.

Kuno laboratory integration is `8d25a9d` on `codex/kuno-native-text` in the
dedicated Kuno worktree. Source `cbfa8d1` corrects fractional token-time decay,
retains transient frequency/recency across all144 slots and adds optional
original PUBLIC writes into retained GPU slots. Original acceptance passes in
fixed/copy/direct modes; short matched decode is10.9238/10.4829/10.5202tokens/s,
with no demonstrated direct-write speed gain. Long diagnostic source is
`569c8e2`; runner acceptance is `8d25a9d`.

Both v75 separate1024-step public continuations pass full closes and exactly
matching greedy IDs. Last128 steps reach7.475 fixed versus9.405 adaptive
tokens/s; adaptive expert coverage reaches89.18%. Disk reads taper substantially
but total20.922/20.430GiB. This repeated teacher-forced workload is diagnostic;
the ordered pair has a file-cache warming confound. It does not prove arbitrary
prompt performance or repeated hot-page thrashing. Read the
[current checkpoint](kuno-native-text-checkpoint.md),
[disk research](kuno-native-text-disk-access.md), and
[verified numeric receipt](../apps/kuno/benchmarks/text/receipts/next-adaptive-steady-20261008.json).

Checks pass519 native tests plus two one-test shader suites,69 targeted Python
controls, strict release all-target Clippy and formatting. The accepted frozen
v75 SHA is `ecdb3a5f231d1a81b91398ebcc39378a659d96fce2d3358682da9b78f312a882`.
Exact trace/preload and runner candidates `990c52d`/`0740b6b` remain isolated,
unintegrated and unmeasured; root orchestration/main is still needed. Preserve
their clean review worktrees until integration. Do not package these candidates
as accepted runtime changes.

Normal admission is resumed; all lab jobs exited. B70 physical memory returned
to50688000bytes, NVIDIA has no compute process, classified Xe/NVIDIA faults are
zero and all three services returnHTTP200. No reset, unbind, reboot, install,
release or API/configuration change. **Required Vashti changes: none.**

### Active handoff: exact native-text residency replay and quant research (2026-10-08)

Kuno lab source `aef04f2` integrates `d45dc4b` trace/preload and `366d847` protected
runner; fast-report research is `a5d2827`. v76 frozen SHA:
`7e18edff2d5413b48b834f391f6637497bf5d157acd3507542afa2602887c8a7`.
`cpu-144-public-oracle-v76` passes every original output/lifecycle/owning gate
and both extra full training/replay closes. The correct seven-step replay
reaches11.74249tokens/s with3360 GPU calls and zero CPU work, expert uploads or
observed physical disk I/O. All eight original prefix IDs match. This is a
short conditional current-engine residency ceiling, not a hardware maximum;
loading/prefill/preload are excluded. No router/expert substitution.

Read the [latest checkpoint](kuno-native-text-checkpoint.md),
[internal contract](kuno-native-text-count1-hybrid-plan.md#exact-short-resident-replay-diagnostic-2026-10-08),
[primary-source speed/transfer research](kuno-native-text-disk-access.md#bidirectional-cache-transfer-and-published-fast-reports)
and [verified numeric receipt](../apps/kuno/benchmarks/text/receipts/next-public-oracle-20261008.json).
The user now suggests a smaller quant. Full83.6GB IQ3_S support is assessed in
the disk research: split loading and seven additional mixed-format paths need
implementation plus a separate reference fixture. No payload was downloaded;
no smaller quant or speculative decoding has been benchmarked here. Published
Strata/FreeToken-Kai results use materially different quantization, host
residency and often MTP; they have not been reproduced on this system.

Checks pass525 native tests, two one-test shader suites,181 Python controls,
strict release all-target Clippy, formatting and whitespace. Normal admission
is resumed, all lab jobs exited, B70 returned to50688000physical bytes, NVIDIA
has no compute process, classified faults remain zero and all three services
returnHTTP200. No installed binary/configuration, API/catalog, model-file,
Vashti, release, reset/unbind/reboot or driver/clock changes. Expert histories
remain transient. **Required Vashti changes: none.** Review worktrees are
integrated, reviewed and archived. Accepted receipt and quant assessment are
committed in `3601124`. No shared API change is needed.

### Active handoff: native Unsloth quant and MTP implementation (2026-10-08)

The user approved Unsloth UD-Q3_K_XL with the self-contained Q8 MTP head and our
own Rust implementation. Their backend direction is native Kuno where a
supported backend exists, with ordinary Ollama fallback for unsupported models.
Exact future routing/naming must be coordinated before changing consumers.
No production routing or API change is implemented in this handoff.

Historical foundation at `bb84f38`, superseded by the accepted foundation
handoff below. Its source followed `ae97ac4` (bounded full scalar MTP
reference), `cd0c2a1` (original packed IQ3_XXS/IQ4_NL/IQ4_XS decode; isolated
source `1ffd041`), and `712b605` (strict split owner; isolated `ce964280`). Read
the [exact native MTP plan](kuno-native-text-mtp-plan.md) before further changes.
The original Q4 acceptance and receipts remain unchanged.

The Q8 sidecar's complete checksum passed. Its strict34-tensor binding and
complete graph scheduling pass a synthetic executor check; real CPU weights
run on fixed public token0/zero-wide-hidden input and repeat exactly after KV
rewind. Six math/ownership tests cover per-stream embedding-first conditioning,
Q/gate order, independent dense GQA equations, contiguous KV grouping, all
visible positions and wiping helpers. This does not establish independent
full-head numerical parity, target/draft integration or speculative speedup.
The scalar reference remains public-fixture-only because inherited operators
lack complete private error-path erasure acceptance.

Merged foundation checks pass148 kuno-text tests plus two opt-in downloaded-head
tests,526 native laboratory host tests, strict crate/native Clippy and formatting.
Packed IQ reference fixtures contain8032 exact F32 values from pinned upstream
equations. Original optimized/GPU IQ admission is still rejected at this root
commit. GPU quant operator support and explicit Q3 expert/PLE ownership are being
implemented in separate review worktrees. Larger model shards are downloading;
complete hash verification, local no-copy Ollama import and actual native main
graph admission remain pending. No Q3 generation or MTP GPU performance claim.

Normal admission and all installed services are unchanged. There are no lab GPU
jobs/models at this checkpoint; protected GPU tests will use the restored sudo
credential once operator source is ready. No reset, unbind, reboot, driver/clock
change, installation or release. Model provenance and synthetic receipts only;
private prompts, outputs and expert histories remain unpersisted.
**Required Vashti changes: none for this internal milestone.**

### Active handoff: original-packed Q3 foundation and target/MTP coupling (2026-10-08)

Kuno source `4b29087` integrates strict split loading (`712b605`), scalar/GPU IQ
operators (`cd0c2a1`/`c249ed2`), actual Q3 expert/PLE ownership (`5c231b5`),
fixed4/32-position full-logit diagnostics (`42906b8`/`e602916`), bounded8GiB
ordinary cache (`d682e15`), original packed CPU SIMD (`d34c837`), actual guarded
target carry (`ff028b6`) and real CPU target/MTP coupling (`97772b3`). The IQ4
literal candidate needed corrected CubeCL explicit assignments (`4b29087`);
both physical GPUs accept its generated-shader and independent operator checks.
Read the [exact MTP plan](kuno-native-text-mtp-plan.md),
[latest checkpoint](kuno-native-text-checkpoint.md) and
[numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

All four pinned Unsloth payload hashes pass. Main89,986,353,824bytes is imported
as `qwen3.8-flash-next:125b-a6b-ud-q3_k_xl` through Ollama's local split create,
using CAS hardlinks and no second payload. The4,137,429,120-byte Q8 sidecar is
hardlinked outside Ollama's GC namespace; it is not registered as an unsupported
Ollama draft. This model import does not enable native HTTP routing.

Both cards pass16064 independent decoded C values,117 projection/MAX8 cases,
12 cooperative and12 subgroup fault cases with complete operator owning audits.
The compiler-idle32-position Q3 B70 run passes every full logit, exact greedy,
owning/fresh/recovery gate:31 decode steps in4250ms (7.294tokens/s), no ordinary
uploads and1,323,008 observed physical read bytes. The same-artifact cap0
control fails full logits at position23 after greedy equality; it is rejected,
with no accepted timing and no weakened tolerance. One-token prefill leaves
134/144 bank slots empty per layer; explicit bounded filling is a pending
candidate. Preserve this failure and pending status rather than claiming all
Q3 GPU-placement paths accepted.

The CPU target/MTP test uses actual pre-head h0 at draft position1. Full target
logits and both draft hidden/logit vectors repeat exactly. This is executor
parity/repeatability, not an independent whole-head quality oracle. The CPU MTP
reference and generic nonresident GPU probe lack complete private error-path
owning-erasure acceptance and remain fixed-public-only. No speculative speedup,
native production fallback/routing, private MTP or fully GPU draft claim.

Merged host checks pass157 core tests with10 intentionally ignored opt-ins,
547 native main tests with one ignored, plus two one-test shader suites.
The protected real target/MTP opt-in also passes. Strict featured Clippy and
formatting passed the preceding integrated candidates; final new candidates
must repeat relevant checks. See source-pinned receipts for exact artifacts,
limits and test boundaries. No reset/unbind/reboot, clock/driver changes,
installation, release, public API/config or Vashti changes in this block.
Normal admission and all three services recover, both GPUs return to baseline,
and classified fault counters remain unchanged after every accepted/rejected
trial. **Required Vashti changes: none.** The future native catalog/routing
contract still needs coordination before changing consumers.


### Active handoff: Q3 routing diagnosis and correct-expert replay paused (2026-10-08)

This supersedes the preceding Q3 foundation checkpoint for new backend work.
The user requested a clean pause to use the GPUs. Root Kuno source is `9218638`
on `codex/kuno-native-text`, in its existing isolated worktree. Preserve the
canonical Vashti checkout/coordination work. Public API/config/catalog, installed
services/runtime, Vashti and release selection are unchanged. **Required Vashti
changes: none.** This is standalone fixed-public laboratory work.

After `4b29087`, integrate the complete backend range through `9218638` when
resuming sequential work: actual MTP projections (`8cba745`), empty-slot filling
(`b739c32`), same-input expert/router diagnostics (`37f4edb`, `41db769`,
`b285d68`, `67bdcfa`), independent segmented-IQ probe (`f116d09`/`3571bcb`),
explicit original32-reference prefix23 (`4f41d6c`), skip-unused segment option
(`397999c`/`c6d2088`), and fixture-scoped correct-expert replay (`9218638`). Read
[the exact plan](kuno-native-text-mtp-plan.md), [latest checkpoint](kuno-native-text-checkpoint.md)
and [numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

Both cards pass actual target-conditioned MTP GPU projections and independent
original72+72 mixed-IQ MoE operator cases. MTP other graph math remains CPU;
private full-head ownership, independent whole-head quality and speculative
performance remain pending. The Q3 full32 all-GPU logit failure remains:
position23/layer17 near-tie routing follows tiny arithmetic differences and
changes one expert. Same-input expert outputs and CPU top-k on GPU scores agree.
No tolerance relaxation, forced routing, full32 performance or decoder defect
claim. Explicit prefix23 checks the complete original32 file and verifies only
positions0..22 before the known discontinuity.

One compiler-idle same-artifact prefix ideal pair gives9.516→10.118tokens/s
without/with unused-segment skipping, respectively (22steps each).10,560 GPU
expert calls, zero CPU calls, expert uploads and physical read bytes during
replay. Full23 logit vectors/greedy IDs, exact trace membership and full owning
closes pass. Correct original preloading/training are excluded from timing;
this is conditional short residency, not demand-cache/production throughput.
Demand-fill initial runs are slower and I/O-confounded. Keep new options off
by default. No private prompts/routes/history were persisted; all fixtures are
compiled public token0/seed424242.

Root host checks:591 native tests (one ignored), two one-test shader suites,
strict release all-target Clippy default/all-five features, formatting and
whitespace pass. Four v90 runs have five exact818413568-byte/1378-buffer original
owning closes each, plus two additional ideal training/replay closes where used.
At2026-10-09T04:41:26Z all jobs exited, normal admission resumed with no active/
queued/recovery latch, HTTP200 from Kuno/Ollama/Vashti, B70 memory50688000bytes,
no NVIDIA compute process and zero classified Xe/NVIDIA faults. No reset/unbind/
reboot, driver/clocks, installation or release change. GPUs are available.

Next source candidate `6d12ddd` is isolated on `codex/kuno-q3-subgroup-moe`, base
`9218638`, in the Unsloth quant-support managed worktree. **Not integrated,
compiled, tested or GPU accepted.** It adds a leaf-only four-row original-order
subgroup MoE kernel plus module/helper visibility. Resume with count1 opt-in/
device-feature integration, independent three-profile probe/fault/owning checks,
then host and protected both-card/full-vector performance acceptance. No GPU
work should resume until the user gives them back. DASLab remains an assessed
quality comparison candidate, without a new download or measured comparison.


### Active handoff: DASLab native Rust and Strata optimizations (2026-10-09)

**Goal paused by the user after the overnight pass.** Finish only the current
source checkpoint; do not start further GPU work until resumed. The final
paused receipt below supersedes earlier active-goal wording. Installed services
and the canonical Vashti checkout remain unchanged.

Supersedes the paused `9218638` checkpoint for backend inference work. User has
returned the GPUs and approved full DASLab IQ3_S plus Strata source study, with
an active30 accepted-token/s target. Latest model-tested target/MTP source is
`12f9cc7` (v129) in the same isolated `codex/kuno-native-text` worktree: dedicated
request I/O reuse, direct-only submission, complete finite fusion and actual warm
replay pass full physical gates. Earlier
conditional ideal count1 v109 remains selected for that separate scope. Installed services, public contracts/catalog/
configuration, Vashti and release selection remain unchanged. Preserve the
canonical checkout's uncommitted coordination/application work.

Read [current checkpoint](kuno-native-text-checkpoint.md),
[DASLab notes](kuno-native-daslab.md),
[numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/daslab-foundation-20261009.json)
and the [MTP plan](kuno-native-text-mtp-plan.md). Changes since pause include
original-format scalar/SIMD/GPU support (`d11bd15`/`985ffb5`), exact DAS profile/
shared decoder connection (`67a01bc`/`aae7e3b`), explicit original prefix13
(`07be82e`), dense decoder extension (`7d06efb`), original shared-codebook and
integer expert leaves (`d424727`/`2375412`/`b82eefd`), direct lane initializers
(`6189e69`/`5b976e2`), prepared Q6 head (`36cf286`/`fb6cd2f`), aligned Q6 loads
(`bcb9bfc`), host phase diagnostics (`690f032`) and bounded larger request-cache
metadata (`bfadcd4`/`85711cd`). MTP KV-only append is `2dea140`. These are
laboratory changes, not a new production native model route.

Two original DAS shards are fully verified without merged/duplicate payload.
Both GPUs pass independent mixed-quant operators. Fixed prefix13 full-vector/
greedy/ownership tests pass. Best conditional ideal replay is12.728tokens/s;
30tokens/s is not achieved. Original full32 fails at position13 with finite
logit disagreement despite greedy equality; keep that failure and unchanged
tolerance. v105 identifies binding/enqueue/fence overhead; simply growing the
append-only cache in v106 does not improve hits or speed. FIFO cache replacement (`b4c07ed`) and once-prepared integer inputs
(`f5d5e97`/`95b31cc`, full INFO plan `774f2a9`) now pass both-card operator and
protected prefix13 model acceptance. Same-artifact ordered control/prepared
comparison is10.511→12.728tokens/s,17.4% lower latency. Aligned Q6 passes
operators/model gates but adds no speed. Keep the original head. Next larger
[resident GPU routing candidate](kuno-native-resident-router.md) is isolated work
in progress, with no device/throughput acceptance.
No private content/history is persisted; only compiled public numeric fixtures
and model provenance. Actual memory, admission/services and classified-fault
recovery pass after every physical trial. No resets, reboots, driver/clocks,
installation or release. Integrated663 default/666 all-feature native tests and auxiliary suites pass;
strict default/all-feature all-target Clippy pass. **Required Vashti changes: none.** Future production native
catalog/routing still needs a coordinated consumer contract.


#### Backend-only routing checkpoint (2026-10-09,11:00UTC)

Model-tested source `1742100` retains all original DAS13 vector/greedy/trace and
owning checks. New GPU routing is numerically accepted but **slower** than its
matched both-segment control:10.1277→8.4455tokens/s. Keep prepared-only v109
(12.728) selected. Mapper probe `536c98e` passes both GPUs,49 independent cases,
24 fault cases and complete erasure. Full receipts and exact artifact hashes
are in the linked DASLab notes/numeric receipt. Physical memory/admission/HTTPs
recover, all classified fault counts zero. No production API/configuration,
installation/release or Vashti change. Next isolated candidates are parallel
mapper preflight and then prepared integer composition; no speed acceptance
for either yet. **Required Vashti changes: none.**


#### Backend-only parallel routing checkpoint (2026-10-09,11:45UTC)

Model-tested source `6a32b11` now passes prepared integer + parallel sealed GPU
routing with original DAS13 vectors/greedy/actual trace and owning audits.
Same-artifact ABBA gives prepared-only12.7468/12.8670 and routing14.6064/10.0652
accepted continuations/s: the routing gain is inconsistent, so keep v109
prepared-only selected. Parallel original mapper `416f889` passes both cards.
Root695 default/698 featured tests, auxiliary shader suites and strict
Clippy/format checks pass. Exact scope, per-step clocks, physical read counts,
source/artifact hashes and all failures are in the native checkpoint/DASLab
aggregate. Native MTP target/draft leaves are in progress, not performance
accepted. Public services/admission/memory/fault counts recover after all runs;
no install/reset/driver/API/Vashti/release change. **Required Vashti changes:none.**

#### Backend-only target rows/carry checkpoint (2026-10-09,12:08UTC)

Original all-row target `d12b96b` and public registration `38c0e52` pass the B70:
complete CPU logits/actual pre-head carries for all four rows across five
partitions and continuation, exact greedies/graph/cursor checks and full owning
closes. Frozen default artifact is `fc4c9f346c67b571585749d5164b391bcdd21ec6d101e28d3fa32e151f172ab7`.
CPU oracle `68e2c33` matches existing CPU logits bit-for-bit and rehashes both
original shards. Root717 native plus2 auxiliary tests, strict default Clippy,
fmt/diff and80 launcher controls pass. The narrow protected launcher correction
is `ef2cf9e`; initial pre-model rejection is preserved. See the
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json)
and [implemented target contract](kuno-native-mtp-all-rows-plan.md).
All supplied rows commit; accepted-prefix repair, whole GPU MTP and speculative
speed remain pending. No new throughput claim. Physical memory/admission/HTTPs
recover and classified faults stay zero. No install/API/Vashti/release change.
**Required Vashti changes:none.**

#### Backend-only resident MTP checkpoint (2026-10-09,12:48UTC)

Corrected global input RMS `724ca2f` and publisher-aligned CPU v2 oracle
`99386e6` pass complete fresh CPU vectors. Full resident Q8 MTP `a32f241`
and public consumer `1761b06` pass both physical GPUs: teacher/provisional
vectors, exact greedies, actual target-KV repair/continuation, two fresh
workspaces,42-owner erasure and public-weight release. Frozen default source
`f652d53`, SHA256 `7f6a8e12690b2ad2c68827bc26d00808ebe4657deea28f384a59318a9dc266d4`.
739 default native+2auxiliary tests, strict Clippy/fmt and81 launcher controls
pass. Actual memory/HTTP/admission recover and classified faults remainzero.
Initial provisional CPU agreement remains0/2; neither useful steady draft
acceptance nor speculative throughput is claimed. Historical dummy-cell/
per-stream references are retained but superseded for publisher conditioning.
Target-prefix repair source `f3899b6` and public consumer `211b282` pass
747+2 source tests/strict Clippy; physical target repair is next. See the
[native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
No public contract/install/release change. **Required Vashti changes:none.**

#### Backend-only speculative foundation checkpoint (2026-10-09,13:08UTC)

Actualtarget prefixrepair1/2/3 passes at `54c846b` (v118); minimalQ8draft and
GPUfollow pass bothcards at `241415b`/`da69fe0` (v119), keeping fullchecks.
Separate13-row CPUcarry `d6a7df7` matches oldCPUlogits bitwise. First actual
dualGPUpubliccontroller `66c1078` passes12targetcontinuations,6/10draft
acceptance, originalstate repair/realKVrebuild and fullcloses. FrozenSHA
`de3120b39f366914fecb96515c55154a13d314356d38a7377d3108a4074f1ce5`.
Cold normal-demand run1.1326tokens/s includes2,311expertfills/4.777GB and
allactualdraft/verification/repair/publicchecks; this distinctscope doesnot
establish a speedup over selected12.728conditionalidealpreparedcount1.
762native+2aux tests, strictClippy/fmt and82launchercontrols pass. Actual
memory/HTTP/admission recover, classifiedfaults remainzero. Details, all
source/artifact/hash provenance and preservedfailure are in the
[native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
Packedthree-column dense and preparedbatch expert candidates are inprogress.
No publicAPI/config/catalog/install/release change. **Required Vashti changes:none.**

#### Backend-only direct fill and draft fusion checkpoint (2026-10-09,14:37UTC)

Latest model-tested source `a6fcdf5` passes original14-format operators on both
cards, target4/actual prefix repair and fused MTP full CPU vectors/KV repairs,
with complete owning erasure and actual memory/admission/HTTP recovery. Frozen
default SHA256 `38296f8f909cb6791cc249a5180d2fdd2f833c64c6206cfcbb0636d4c19d1c5b`.
Compiler-idle ordered cold normal-demand pair improves1.50087→1.74852 accepted
tokens/s with checked existing-view public fills+fused MTP, unchanged arithmetic
and actual repair. Both have2,311 first-touch fills/zero evictions and6/10 draft
acceptance. This distinct cold scope does not replace selected12.728 conditional
ideal count1;30 remains unmet. Prepared candidate has different disk paging and
no accepted speed gain.796 native+2 auxiliary source tests and85 launcher
controls now pass for the next one-shot public warm replay; its physical gate
is pending. No private content, GPU resets or public API/config/install/release
change. Exact sources, hashes, numeric receipts, all failures and limitations
are in the [native checkpoint](kuno-native-text-checkpoint.md) and
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
**Required Vashti changes:none.**

#### Backend-only warm and finite acceptance (2026-10-09,15:14UTC)

`4b93c99`/`bafd9c0` real healthy one-shot replay passes full original CPU13,
actual target/draft repairs, complete retained-private erase and final closes.
Weights/banks/IDs/admission/epochs stay; transient binding/INFO caches are
cleared. Warm normal-demand2.34472 withzeroexpertuploads establishes that
execution overhead remains. Prepared2.34773 shows no material gain. Packed
Verify3 warm2.61000 and full finite fusion `fe196fa`/`6f581b4` warm2.96368 pass
all original gates. Corrected inventory `5eb8e82` passes60normal+210invalid/
stickycases percard and full actualprefixrepair. Current model-tested v127
SHA256 `7f3fe8a0a9a24c6543784cfb1937c8bde3db7e6507cf612ee363b59318c2baba`.
These ordered warm trials do not establish matchedABBA/30token/s acceptance or
replace v109's separate12.728 conditionalideal count1 selection.

New pure `direct-device` source `834a299` isolates existing GPU submission type
for only fixedpublic target4/repair4/speculative13. Exact existing
`KUNO_LAB_DEVICE_DISPATCH=inline` and direct-only build are mandatory; all CPU
features/default+inline/directwithoutflag mismatch reject beforedevice access.
ActualDeviceHandleblocking is checked in main/run/receipt. Mathematical kernels,
owners/budgets/cache/defaultthreaded/otherlabmodes unchanged.806 native+2auxiliary,
pure-direct registration/runtime and rejectedallfeatures checks,87launcher
controls and strictdefault/direct/allfeaturesClippy/fmt pass in child source;
physical gate is pending in root. Actual GPU memory/admission/services/fault
recovery remains verified after each trial. Exact sources, all numeric receipts,
failures/limits in linked checkpoint/foundation receipt. No private prompts/
outputs stored, productionAPI/config/catalog/frontend/install/release/reset
change. **Required Vashti changes:none.**


### Kuno native text direct target checkpoint,2026-10-09,15:31UTC

`834a299` pure `direct-device` fixed-public DASLab target modes now pass actual
B70 target4/repair4 and B70-target/RTX-draft fullCPU13 cold/warm controller gates.
Warm3.18530 acceptedtokens/s withzero expert transfers/weight reads/processdisk
bytes;30tokens/s is unmet and no matchedABBA selection follows. Original full
vectors, arithmetic, actual state repairs, private/source erasure, memory/HTTP/
admission recovery andzero new classified faults pass. Exact source/artifact/
helper/numeric evidence is in the foundation receipt and native checkpoint.
A target-client-only charged request metadata reuse scope is next, source work
only. Ordinary modes/API/frontend/install/release remain unchanged; required
Vashti changes:none. Preserve canonical application and coordination work.


### Kuno native text request reuse checkpoint,2026-10-09,15:55UTC

Runtime `0cfdbe0`, bounded native permit `9f3b8cc`, public registration `12f9cc7`
now pass original target4/actual prefix repair and fullCPU13 cold/warm dual-GPU
controller gates. Compiler-idle warm6.63953 acceptedtokens/s,zero expert uploads/
weight reads/4,096 process disk bytes. Request begin+original full closing audit
are inside timing; metadata4096 entries/4MiB are fully charged, target-only
programmatic configuration leaves draft/default/cap10 unchanged. Full private/
source erasure, actual memory/services/admission andzero classified faults pass.
Root814+2auxiliary tests (oneignored), strict default/direct Clippy/fmt,88
launcher controls/source review pass. Exact hashes/numeric receipts and scope
are linked in native checkpoint/foundation receipt.30tokens/s remains unmet;
ordered improvement is not matchedABBA or production/private acceptance.
Opt-in original-F32 unused expert-member scratch reductions are the next
isolated source work. No API/frontend/installation/release/reset changes;
required Vashti changes:none. Preserve canonical application/coordination work.

### Paused Kuno overnight handoff,2026-10-09

The user paused the goal; source-only checkpoint `bbe630a` integrates the
original-F32 live-member kernel `7ab49b8`, immutable model policy `13d9929`,
independent multirow operator `a13ebc3` and tiny lint fix `91dd416`. No new GPU
run is authorized by this handoff. Latest model-tested artifact is still
12f9cc7/v129. Normal warm dual-GPU speculative controller is6.63953 accepted
tokens/s; optional two-residual-Q8 prepared mode passes original full-vector
and actual-repair gates at6.77829, but its2.09% ordered gain is not a material
selection. Separate conditional ideal count1 v109 remains12.728; none establishes
30tokens/s or production/private/general inference acceptance.

Live-member source skips only unused member-plane reductions, preserves active
arithmetic/unconditional barriers, and remains default-off. Physical both-card
operator, target4/repair4 and fullCPU13 matched control/live timing are pending.
Root829native+2auxiliary,89 launcher tests and independent source reviews pass;
exact final lint checks, source hashes, numeric receipts, preserved failures,
limitations and resume steps are in the
[paused native checkpoint](kuno-native-text-checkpoint.md#paused-overnight-checkpoint2026-10-09)
and foundation receipt. Fresh16:21UTC actual memory/HTTP/admission recovery and
zero classified Xe/NVIDIA faults pass. No benchmark GPU process remains.
No API/frontend/config/catalog/install/release/reset change. **Required Vashti
changes:none.** Preserve the canonical checkout and its coordination/application
work; resume source/benchmarks in the isolated native-text worktree only.


### Standalone Strata Intel comparison, 2026-10-09

The user explicitly authorized a separate upstream Strata installation while the
native Rust optimization goal remains paused, and requested leaving Strata running
for hands-on use. MIT source v0.1.41/fb58e0d, AOT bmg-g31/oneAPI 2026.1, original
DASLab IQ3_S hard-linked in place, new pinned Q2_0 MTP conversion. No native engine,
API, Vashti integration, production model catalogue, driver or release change.

Actual B70 OpenAI API checks return the correct integer sanity result and
27.3/36.0 decode tokens/s on two 200-token public fixtures (seed 424242, greedy,
no-thinking, PLE direct). This is upstream Strata application inference; it does
not establish native Rust 30 tokens/s or a matched Ollama comparison. Core/swap
protection and no persistent inference logging are checked; corrected rootless
Podman containment requires service Delegate=yes. Actual shutdown returns B70 to
50,688,000 B, admission unpaused and all three HTTP services 200 with zero classified
GPU faults. Final protected restart/streaming and health/web checks pass; sampled streaming
at temperature 0.7 produces 45 tokens at 16.4 decode tokens/s. Strata is left running
on loopback port 8085 for the user; Kuno inference admission stays intentionally paused
until the documented stop control releases the GPU and restores it. No private
inference output is persisted; upstream browser history uses localStorage.

Exact format notes, source hashes, numeric receipts, local installation/control
paths and limitations: [standalone comparison](kuno-strata-comparison.md).
That document includes the user-requested Q2_0 MTP runtime importer notes for
future Rust work. **Required Vashti changes: none.** Preserve canonical application
work. Native goal and pending live-member kernel gates remain paused.

### Native Strata-inspired pass resumed, 2026-10-09

The user authorized teardown of the comparison installation and resumed native
Rust engine optimization with a new goal. The historical Strata-running state
above is superseded: Strata is stopped and its specifically owned build/image
removed, while numeric receipts and the alternate Q2_0 draft files are retained.
Protected 17:43 UTC host checks confirm actual GPU memory recovery, unpaused Kuno
admission, all three HTTP services 200 and zero classified GPU faults. Exact
retained paths and limitations are in [the comparison receipt](kuno-strata-comparison.md).

Native source starts from `de6269d` in the isolated native-text worktree; parallel
backend source work uses separate worktrees at that commit. Priorities are
device-side routing/group maps and a strict alternate bound-target MTP graph,
then real public language-generation benchmarks. No API/frontend/catalogue/
release change is included. **Required Vashti changes: none.** Existing full
CPU13/actual-repair correctness gates remain controls; conditional all-hit GPU
results must not be presented as ordinary demand inference performance.

### Native Strata follow-up source and physical checkpoint, 2026-10-09

Backend-only active native work now integrates original Q2 GPU draft `9f66226`,
independent probe `bedc06f`, host-placement probe `ceb5315`/`740cd91`, profile
selection `9d99d0d` and main `f868437` in the isolated native-text worktree.
Frozen v135 passes both-card full Q2 draft CPU-reference/repair/owning closes.
v136 passes both-card small direct host-allocation original math, combined
CPU13 and actual public code200 generation. The Q2 public owner plan falls
5,011,668,992→2,539,257,856bytes; its cold-code200 result is3.39096 continuation
tokens/s and short all-hit warm13 is5.86668, so no speed improvement/30 claim.

Explicit internal opt-in is `KUNO_LAB_MTP_PROFILE=strata-q2`; absent preserves
Q8. Verified sidecar path remains `KUNO_TEST_MTP`. Both profiles retain actual
target verification and original source ownership. No production engine, API,
catalogue, installation, frontend or release selection is changed. Every trial
restores physical memory, normal admission, all three HTTP200 and fault counts0.
**Required Vashti changes:none.**

Next source work: separately budgeted immutable host-only expert pages and
nonreuse external CubeCL ownership; bounded GPU chronological attention selection
for <=2048 token positions, exact original CPU fallback above that boundary.
Neither connection has whole-model acceptance yet. Numeric receipts, exact
artifacts/tests/limitations and preserved startup failures are in the
[current checkpoint](kuno-native-text-checkpoint.md#smaller-draft-and-direct-host-source-physical-follow-up).


### Native attention/host-source checkpoint, 2026-10-09

Backend-only native source`cc4b8aa`+`597039d` passes both-card chronological QSA
operator, original full CPU13/exactgreedy/actualrepair and normal code200 gates.
Internal opt-in`KUNO_LAB_TARGET_QSA=chronological-public` uses exact chronological
original IDs for end<=2048tokens, retains original guarded CPU selector above;
extra49,152INFO+4,096host bytes, no newMAIN owners. Absent/default unchanged.
Warm7.26311tokens/s/maps51; normal code200candidate2.08246/control2.01633 with
~49s weight-reading/35GB physical reads. No matchedABBA/30/production acceptance.
Parent897+2aux native tests (one ignored), strict direct Clippy/focusedQSA58 and
96launcher controls pass. Exact sources/artifacts/numeric recovery evidence and
limits are in the [native checkpoint](kuno-native-text-checkpoint.md#chronological-attention-and-immutable-host-registration-follow-up).

Immutable external public host runtime`dd0db6b`+`139075e` has source/ownership
acceptance; actual CubeCL host operator and full mirror consumers remain pending.
The user-visible API/configuration/native catalogue and installed services are
unchanged. All physical trials restore memory/admission/allthree HTTP200 and
fault counts0. Required Vashti changes:none; no release/install/reset authorization
is inferred from this internal performance checkpoint. Preserve canonical work.

### Active handoff: RAM-source native text validation; user requested stop, 2026-10-09

The detailed [next-model handoff](kuno-native-next-model-handoff.md) is the
starting point for another model resuming this backend pass. It includes the
exact source/physical acceptance split, model and reference paths, protected
helper limitation, ordered correctness/timing gates, and recovery instructions.
Its creation changes documentation only; GPU work and the goal remain stopped.

This supersedes the pending-host-materialization state above. v143/`f0e088b`
passes both-card80-case authenticated512 source-router operators and the B70
full original CPU13 target/3060draft controller with actual VRAM/RAM experts.
The importer now authenticates actual Vulkan allocation alignment rather than
conservative planner rounding. All original vectors, greedies and actual repair
pass;432host pages detach and CPU-zero/free36,147,609,600bytes. Short deterministic
warm placement is4.21677tokens/s with0explicit expert uploads; it is not normal
language200 or30tokens/s acceptance. GPU512 full-model routing and the new
inactive-partition kernel candidate still require physical acceptance/timing.

The user requested stopping this pass to conserve usage. All GPU work is stopped;
21:23:38UTC checks confirm B70baseline50,688,000B, no NVIDIA compute owners,
normal/unpaused Kuno, allthree HTTP200 and unchanged classified GPU faults0.
Installed Kuno/Ollama/Vashti and their public APIs are unchanged; no release or
installation is inferred. Required Vashti changes:none. Resume from the
[RAM-source checkpoint](kuno-native-text-checkpoint.md#ram-source-full-model-acceptance-and-stop-checkpoint-2026-10-09)
and [internal target/source contract](kuno-native-mtp-all-rows-plan.md#authenticated-immutable-ram-expert-sources-2026-10-09).
Exact source hashes, tests, artifacts, preserved startup failures and numerical
limits are recorded there and in the content-free native optimization receipt.
Preserve the canonical frontend task's unrelated/uncommitted files.

Final source checkpoint`389fb6b` includes explicit inactive-partition candidate
leaf`612b7f6` and guarded parser/launcher`6413b16`; defaultfalse unchanged.
Integrated45public parser/reference +30frozen/routing/shader +102launcher tests,
strict pure-direct all-target Clippy and formatting pass. Candidate GPU/full-model
and speed acceptance remain pending, as does full-model frozen512 routing.
The physical accepted control remainsv143/`f0e088b`; see checkpoint for exact
source chain, protected helper limitation and the ordered next gates. No more
benchmarks are authorized within this stopped pass.

### Active handoff: native lookup-table speed experiment resumed, 2026-10-10

The user explicitly resumed backend optimization after GLM/free-model work,
requesting progress toward20tokens/s. This supersedes the stopped-pass status
above. Inspection found no blocking defect in the router-activation ledger fix
`8c7d838` or preserve-bank diagnostic `4fad434`. v145/fd39cd1 has full physical
CPU13 gates and GPU-routed code200 ABBA: control7.54493/candidate8.33810tokens/s,
matched exit-kernel gain10.51%. No production/API/release acceptance is implied.

The fresh v146/1d0d7c8 preserve-bank replay passes original CPU13 vectors,
greedies/real repair and full closes:12continuations/2.095918815s=5.72541tokens/s,
9120VRAM/0host rank-member routes,0expert uploads. Memory/admission/services and
fault counts recover. The matched canonical control is4.31182tokens/s (ordered pair,+32.78%);
this diagnostic does not establish a normal-language placement optimization,
because code200 already preserves its post-prefill residents.

Next candidate targets repeated by-value codebook array spills in the actual
CubeCL quantized gate/up kernels. Exact opt-in contract is in the
[all-row plan](kuno-native-mtp-all-rows-plan.md#operative-public-lookup-table-lowering-experiment-2026-10-10).
Private-table leaf`120e94d` is integrated as`e054d70`; the guarded public controller
and negative element-write proof are in the follow-up source commit. Root owns
the shared GPUs. Physical candidate acceptance is pending.
No API, frontend, catalogue, installation or release change is included.
Required Vashti changes:none. Preserve canonical application work.

### Active handoff: native text experiments stopped with system usable, 2026-10-10

This supersedes the resumed/pending status above. The user requested finishing
remaining tests and returning the GPUs. All owned GPU/CPU work has stopped;
no benchmark, installation or release is authorized by this handoff.

Native root7364dc8 (private-table leafe054d70) passes source checks and the full
original CPU13 physical gate. One ordered same-artifact public code200 comparison
is7.78036 control /7.62161 private-table continuation tokens/s; no speed gain.
Default/installed behavior is unchanged. Prior repeated v145 result remains
8.33810 tokens/s; neither20 nor30 is accepted. Two isolated clean source leaves
are ready for future integration/testing: workgroup tables3594b76 and IQ4 shared
down scalesb56af6a, both based7364dc8, with focused CPU tests and strict Clippy.
Their root parser/launcher/receipts, composition and GPU acceptance are pending.

Exact commits, immutable artifact hashes, tests, rates, limitations, worktrees,
next gates and final recovery evidence are in the
[latest checkpoint](kuno-native-text-checkpoint.md#resumed-shader-experiments-and-usable-stop-checkpoint-2026-10-10)
and [current resume guide](kuno-native-next-model-handoff.md).
Internal contract is in the all-row plan; no public API/configuration change.
Required Vashti changes:none. Do not infer full feature, deployment or speed
acceptance from source-only checks.

Final live check2026-10-11 01:27:07UTC: B70baseline50,688,000B, no NVIDIA compute
owners, admission idle/unpaused/no recovery latch, allthree services HTTP200 and
classified GPU faults0. Installed services and canonical frontend are unchanged.
Confirm current GPU availability/sudo again before resuming; this is a historical
host snapshot. The optimization goal remains paused.


### Active native kernel optimization resumption, 2026-10-11

User restored GPU access and sudo and resumed the existing30tokens/s goal.
Native work continues in `codex/kuno-native-text`, isolated from the unchanged
canonical Vashti checkout. GU workgroup and down-scale source leaves are now
integrated as `4f23abd`/`ce78d39`; root guarded selection, actual completed-policy
receipts and composed runtime dispatch are being tested. Exact laboratory
contract is in the final section of the all-row plan. This supersedes the
previous statement that parent selection remains pending; GPU acceptance and
normal throughput measurements still remain pending at this source checkpoint.

Fresh host check2026-10-11 12:10:52UTC: the retained managed image worker was
identified by parent/cgroup and cleanly evicted through Kuno's existing empty
Ollama unload request. B70 returned to50,688,000B; NVIDIA compute owners0;
admission idle/unpaused/no recovery; allthree HTTP services200; GPU faults0.
No reset, driver/service installation or private content inspection was needed.
Installed services and public API are unchanged. Required Vashti changes:none.

### Active internal expert-factor/draft-head experiments, 2026-10-11

v148/76ffd66 passed complete original CPU13 target/carry/greedy/repair gates and
full owner closure. Normal same-artifact ordered code200 is7.73410 control versus
6.90538 workgroup-codebook/shared-down-scale candidate; no speed gain. Candidate
remains default-off. Both trials restored admission, all HTTP services and GPU
memory (B70 baseline50,688,000B, NVIDIA compute0), with no new classified faults.
Numeric evidence is under `v148_cooperative_physical` in the native optimization
receipt; latest physical snapshot12:33:49UTC is historical, not future availability.

Next source candidates: GU factor leafa8e81a1 integrated1ba718e; Q8-head broadcast
leafc4cd6c3 integrated0675594. Parent selectors and actual completed-policy receipts
are integrated with runtime dispatch composition. Exact default-off internal
contract is the final section of the all-row plan. Source validation passes:
9exact-public registration/policy checks,5compound runtime/parser checks,5GU
factor checks,4Q8operative/selection checks and6original projection regressions;
107Python launcher checks,9fixed-trial scope/limit checks, strict pure-direct
all-target Clippy, formatting and diff checks. No GPU numerical or throughput
claim is made for these new candidates until a new frozen artifact passes the
original CPU13 gate and normal language timing. Further four-row down kernel and
cooperative draft-head scale candidates are isolated source work in progress.

This is backend-only lab optimization. Installed services/public API/production
configuration and the canonical Vashti checkout remain unchanged. No release or
installation is authorized by this source handoff. Required Vashti changes:none.

### Active row/draft-scale source checkpoint, 2026-10-11

v149/030a20c original full CPU13, real repair, full closure and physical recovery
pass with GU factor plus broadcast-head candidates. Normal ordered code200 is
8.11398 control /7.68506 GU factors; no accepted gain. Both remain default-off;
broadcast-head-only normal speed was not measured. Reference fixture throughput
is diagnostic, not normal generation. Numeric source/model hashes, all results
and recovery are recorded under `v149_factors_head_physical` in the root receipt.

Next integrated source-only leavesf7bb253 (four-row DOWN, original64virtual
sums/tree) and9dde5da (cooperative Q8-head scales) now have complete guarded
parent selectors and actual-completed-policy receipts. Source tests pass11exact
public policy checks, all48actual composed entrypoint/grid checks,3operative
four-row tests,10Q8head checks,108Python launcher checks,9newfixed-trial memory/
scope checks, strict pure-direct all-target Clippy, fmt/diff checks. Exact internal
contract is in final all-row plan sections. Full original GPU CPU13 and normal
language timing remain pending for v150. Source-only counts are not speed proof.

Backend lab only; installed services/public API and canonical Vashti unchanged.
Required Vashti changes:none. Additional two-row GU and transient host-weight
staging experiments remain separate source work, with no physical acceptance.


### Active paired-GU / bounded transfer experiment, 2026-10-11

v150/3ed26fe passes complete original CPU13, actual repair, full closure and
physical recovery. First normal ordered code200 pair8.11952 control /8.48473
four-DOWN/shared-head candidate (+4.50%) is preliminary; repeated ABBA/prose
remain untested. Exact evidence is v150_rows_shared_head_physical in the native
receipt. Last recovered snapshot13:23:00UTC has B70baseline50,688,000B,
NVIDIA compute0, unpaused idle/no-recovery admission, allthree HTTP200 and faults0.

Two-row GU leaf0353369/parent5d26186 and staging precharge10e3d1c are integrated
source-only; exact default-off selectors and bounded owner/runtime contract are
in the final all-row plan sections. GPU validation for paired GU and full staging
integration/validation remain pending. No30tokens/s, production, install or
release acceptance. Backend lab only; required Vashti changes:none.

### Active exact packed HOST staging source gate, 2026-10-11

v151/e3983c3 passes complete original CPU13, repair/full closure and host recovery.
Its paired GU/four DOWN/shared Q8 head normal code200 is7.08234 continuationt/s;
no matched normal control, so no paired-GU gain is claimed. Latest recovered
snapshot13:39:03UTC: B70baseline50,688,000B, NVIDIA compute0, unpaused idle/no
recovery admission, allthree HTTP200 and classified faults0. Full evidence is
v151_paired_gu_physical in the root native optimization receipt.

Leafd640ab2/c3b5f8d packed staging is integrated with the exact final all-row plan
contract: three32MiB MAIN owners,589,824B extra INFO,256-thread copy capability,
full outer/MoE/source alias/stream proofs, original frozen MAP/expert identity,
same-stream copies and consumers, completed actual policy/accepted logical byte
counters. Controls and all experiment defaults are unchanged. Source gate passes
5leaf checks,3root checks,110Python tests,9strict fixed-label scope/memory checks,
all-target direct Clippy and fmt/diff. Full original GPU CPU13 and normal speed
remain pending. No API/configuration/installed service change or release claim.
Required Vashti changes:none. Work remains in the native isolated checkout.

Initial v152/c3e9403 rejected before target allocation: the executor's independent
exact MAIN check omitted the new staging extent. Recovered snapshot13:48:24UTC
retains baseline GPU memory, idle admission/healthy HTTP and zero classified
faults. Followup adds explicit default-off executor staging reservation, Config
parity and exact precharged byte acceptance;4focused root checks and strict
direct Clippy pass. New frozen artifact/full GPU gate remains required. No
numerical/throughput acceptance or installed-service/API/Vashti change.

v153/e56e466 exposed a second construction issue: staging reserve reacquired MAIN
through healthy()/active inside the already-locked allocation callback. Stopped
only the owned benchmark unit and ran helper recovery; baseline GPUs/HTTP/idle
admission/no faults returned at13:58:14UTC. Full owner erasure is not claimed for
that aborted public-only process. The fix retains borrowed construction origin,
poison and full arena/owner checks without locking MAIN again; held-lock token
test and strict direct Clippy pass. Fresh original GPU acceptance remains pending.


### Active native optimization checkpoint: three-member arithmetic,2026-10-11

Supersedes the pending packed-staging acceptance above. Constructor fixes
3de1529d passed full original CPU13/repair/all-owner gate in v154. Normal code200
packed staging/four-DOWN/shared-Q8-head is7.870304 continuationt/s; no same-artifact
control or speed gain claimed, and the alternative stays default off. Numerical
and physical closure evidence is in v154_packed_staging_physical.

Three-member kernel leaf6473be6 and parent226e3fe plus protected policy1db3d8c
pass6focusedRustchecks, strict all-target directClippy,5new/110existingPython
checks and9strict helperlabels/caps. It keeps tile8/MAX8 owners and every retained
F32 operation, only removing unreachable arithmetic cells3..7. Original-source
construction and attachment share the exact geometry reconstruction. Frozenv155
passed full originalCPU13 and actual completed three-member/four-DOWN/shared-head
flags, full1380MAIN/828866560B and432HOST-page36147609600B closure, and zero draft
reservations. Warm12-token5.471858 diagnostic does not establish normal speed;
normalcode200 completed at8.094793continuationt/s (199/24.583703122s),
130/138drafts/69rounds, full repair/closure. No matched control or speed gain
established; defaultoff retained. Exact source/binary/helper hashes and recovered
snapshot14:28:19UTC are under v155_three_member_arithmetic_physical.

Next comparisons: already-supported original Q2 MTP with current frozen target
(explicit new labels, original shard hashing, no Q8-head policy); separately,
isolated original-arithmetic three-token GDN streaming source to remove a large
private column array. Neither is physically accepted yet. Root owns all shared
GPU/services. Backend-only lab; required Vashti changes:none, installed services
and public API unchanged. No30tokens/s or release/production acceptance.


### Active native backend optimization checkpoint,2026-10-11 14:51UTC

Latest sourcee16c549/v156 three-token GDN passes original CPU13 and normal
code200 full greedy/repair/owning closure:8.489327continuationt/s,199timed,
130/138drafts/69rounds,74actualThreeGdn chunks. Same-artifact control is8.542111t/s but agentClippy may overlap;
no compiler-idle causal pair or gain claimed. Original Q2 with previous three-member/four-DOWN target
also passes at7.949402; its public GPU weight budget falls to2,539,257,856B,
but no faster inference is established. Earlier staging and three-member
physical gates are accepted and their slower preliminary timings remain defaultoff.
Read the updated next-model handoff and exact numeric receipt sections
v155_original_q2_frozen_physical /v156_three_gdn_streaming_physical.
Root owns sharedGPU trials; leaf work proceeds in separate managed worktrees.
Next draft-only experiment is the pinned static English/code vocabulary;
source work is not GPU/throughput acceptance. Last recovery14:51:03UTC restores
baseline GPUs, idle/unpaused/no-recovery admission, allHTTP200, classifiedfaults0.
No30tokens/s/private/production claim; installedservices/API/canonicalVashti
unchanged. Backend-only internal work; required Vashti changes:none.


### Active native kernel checkpoint,2026-10-11 15:44UTC

Latest tested source21d97e2/v157 passes original CPU13/targetgreedy/carries,
real repair, full draft/target/source closure and physical recovery with the
fixed40525-row English/code draft head. Normal compiler-idle ordered code200
is8.293190 shortlist versus8.525761 same-artifact full-head control; no gain,
shortlist stays default-off. Warm4.152881 is diagnostic only. Read receipt
v157_public_english_vocab_physical and the updated next-model handoff. Last
recovery15:42:52UTC returns B70baseline50688000B, NVIDIA compute0, idle/unpaused/
no-recovery admission, allHTTP200 and no classified GPU faults.

Next integrated source-only candidate is the explicit two-residual-Q8 frozen
GU integer leaf and immutable parent, throughe655f32/13c276c/fb7c528/9f80c33.
Read the frozen GU leaf/parent contracts. Ten focused agent checks, strict
all-target directClippy and independent read-only source review pass. Original
F32 equivalence, full original CPU13 and throughput are NOT established by
packed-oracle tests. Root owns public controller/launcher integration and all
GPU/services; agents own isolated encoded-weight/group-scheduling followups.
Do not enable these candidates for ordinary inference, widen tolerances or
represent requested flags as actual execution. No30tokens/s/API/install/release
acceptance; canonical Vashti and installed services unchanged. Required Vashti
changes:none.

### Active native GU scheduling and MTP comparison,2026-10-11 16:12UTC

Latest physical sourceaa4df8c/v158 passes unchanged originalCPU13, targetgreedy,
real repair, full1381MAIN/828932096B and432HOST-page/36147609600B closure,
all draft/source closure and physical recovery. Explicit two-residual-Q8/eight
lane GU normal compiler-idle code200 is8.642210continuationt/s:
199/23.026516602s,130/138drafts/69rounds. Actual3552 preparations/17760GUcalls/
3552audits join full native completion. No same-artifact original-math control
or dramatic gain claimed; defaultoff. See v158_frozen_gu_integer_physical.
Last recovery16:00:31UTC: B70baseline50688000B,NVIDIAcompute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults, same boot.

Later integrated source-only encoded-register preparationb4b20f9 and
GUstride leaf/parent50d423d/febf8df/e5182f0 plus controller853e0e0 pass24
combined focused checks and independent read-only review. Guard124Python
checks and three external fixed stride labels pass. The alternative GU grid
reduces12000→1600workgroups/layer while preserving original runtime routes,
all scratch/owner limits and actual joined five-source receipts. Read the
frozen GU integer/stride leaf, parent and controller specifications. Fresh
artifact original GPU gate and same-artifact language timings remain required;
source success is not physical acceptance. Root owns sharedGPU/services.

The user requested measuring whether slow-slot NVIDIA MTP is a net cost.
An isolated real target-only language controller is being prepared: it must
not open or instantiate the draft nor select NVIDIA, and must execute actual
single-token original target passes and full closure. No target-only speed or
30tokens/s acceptance yet. Backend-only lab: canonicalVashti, installedservices
and publicAPI unchanged, required Vashti changes:none. No release/install.


### Active native MTP placement comparison,2026-10-11

Latest physical sourceac78f00/v159 passes original CPU13 cold/warm, targetgreedy,
real repair, full1381MAIN/828932096B and432HOST-page/36147609600B closure,
draft/source closure and physical recovery. Compiler-idle ordered same-artifact
code200:8.636025 control versus8.973138 encoded-register/GUstride candidate,
199 real continuations and130/138 accepted/proposed drafts each. This3.90%
result is preliminary; repeatedABBA/prose are pending. Full receipt and exact
hashes: `v159_encoded_register_gu_stride_physical`. Last recovery16:30:44UTC:
B70baseline50688000B,NVIDIAcompute0,allHTTP200,idle/unpaused/no recovery,
no new classifiedGPUfaults, same boot. This supersedes earlier pending-v159 notes.

The user requested testing Intel MTP before choosing whether to retain MTP.
Real target-only controllerd12b52f and Intel placement7b790f4 are integrated;
[the exact Intel contract](kuno-native-public-intel-mtp-controller.md) requires
fresh privileged Xe memory data, combined full owning charges, original144
expert-bank capacity, distinct device/runtime and original physical closure.
Target-only does not open or bind a sidecar or select NVIDIA. No implicit profile,
slot or admission fallback. Guard130Python tests pass; Rust source tests and
strict direct-device Clippy pass in the isolated owner trees. Integrated
root validation/fresh frozen binary and three-way physical acceptance are next.
Full Q8 on Intel may reject its tight combined memory budget; record any
rejection rather than widening limits. DOWN stride and grouped PLE are separate,
unselected candidates for later numerical/speed gates.

Existing30tokens/s goal remains unachieved. Root owns GPUs/services. Canonical
Vashti, installed services and public contracts unchanged. Required Vashti
changes:none; no install/release or other-task message is authorized by this note.


v160 Intel physical attempt rejected before model allocation at the independent
Wgpu device identity check. Fresh Xe accounting/full admission passed, and
recovery16:50:47UTC returned baseline/allHTTP200/no new faults. Pinned wgpu29
compares registry-local IDs across independent Instance contexts incorrectly
for this purpose; actual live Vulkan logical-device comparison is being fixed.
No Intel inference/timing/memory-fit claim follows from this failure. See
`v160_intel_mtp_registration_failure`; freeze a new artifact for the retry.


### Completed Intel/Nvidia/target-only placement block,2026-10-11

Source86882f9/v161, binarySHA256
`95ab177a40131f60342b5dff72cb75486c1d89958709def127b0e079ba777632`,
passes original full CPU13 cold/warm with Intel MTP: complete target vectors/
carries/greedy, real repair, all1381target MAIN/42draft private owners and432HOST
pages closed/freed, physical memory recovered. Fix compares live Vulkan device
handles rather than Instance-local Wgpu IDs. Six focused identity tests, strict
all-target direct Clippy and formatting pass; previous integrated98Rust and
130Python checks remain applicable to unchanged parser/guard code.

Compiler-idle ordered same-artifact code200 (199 timed continuations each):
Intel MTP9.908034, Nvidia MTP8.876789, real target-only8.336136tokens/s.
Same public prompt/seed424242/original target math/capacity/cache; both MTP use
full Q8/shared head, accept130/138 proposals over69 rounds and perform5 repairs.
Target-only opens no sidecar/creates no draft device and executes199 actual
count1 target passes, zero drafts/repairs. Full original source/owner closure and
physical recovery pass for each. Preliminary Intel gains11.62% over Nvidia and
18.86% over target-only; no repeatedABBA/prose/independentCPU200/30t/s acceptance.
This does not isolate PCIe bandwidth from compute/queue/driver differences.

Full144banks/8GiB static/fullQ8 draft fit together on B70 under the unchanged
combined admission; actual same physical PCI but distinct Vulkan devices and
runtimes are reported. No profile/slot/cap fallback or installed/default change.
Last recovery17:14:23UTC: B70baseline50688000B,NVIDIAcompute0,allHTTP200,
idle/unpaused/no recovery, no classifiedGPUfaults on the same boot. Exact hashes,
checks, failures and content-free records: v161_mtp_placement_comparison and
[the active next-model guide](kuno-native-next-model-handoff.md).

Next source candidates remain separate: DOWNstride(ec367fb) needs protected
registration and physical gates; grouped PLE leaf33b4de7/controllercce4f66 is
clean in its isolated tree with13grouped/78controller checks and strict Clippy,
but unintegrated/unselected. Preserve86882f9 during its controller merge; keep
defaults off and validate unchanged full CPU13 before timing any new candidate.
Canonical Vashti/API/installedservices unchanged; required Vashti changes:none.
No release/install or other-task messaging is authorized by this handoff.


User steering after this receipt: continue native optimization with target and
full Q8 MTP on Intel B70 only. Defer Nvidia experiments; next physical candidates
must use the accepted Intel placement. This is internal benchmark selection,
not authorization for an untested installed/runtime/API default change.


Following Intel-only steering, grouped PLE sourceleaf9f8462e/controller513fe02
are integrated onto accepted HALfix86882f9. The exact contract is now in
[kuno-native-target-grouped-ple.md](kuno-native-target-grouped-ple.md). Root
integrated source review/CPU checks and protected guard/helper preparation are
underway. Both new policies remain default-off. Fresh full original physical
gates and separate Intel-only timing comparisons are still required; no new
performance, installed-service or consumer change is accepted by this source
integration alone.


### Intel-only grouped-read/source gate,2026-10-11

Grouped PLE source9f8462e/controller513fe02 and protected registrationde59c75
are now integrated above accepted HALfix86882f9. The integrated read-only review
finds no blocker;117 relevant Rust tests and strict locked pure-direct all-target
Clippy with two CPU jobs pass. Both original owning/fault proofs and unchanged
CPU13 reference/tolerances remain mandatory.27 protected-launcher tests and nine
new external fixed Intel-only PLE/DOWN/combined labels pass exact-selector and
original host-cap checks. Fake Xe metadata was used only in those source checks;
physical trials require a fresh privileged memory snapshot after draining.

Next artifact will independently gate grouped PLE on original cold/warm CPU13,
then compare normal compiler-idle Intel Q8 MTP code200 control/candidate. DOWN
stride stays off during that pair and needs its own original numerical/speed
gate before any combination. New policies remain default-off; source success
does not establish physical numerical, throughput or30tokens/s acceptance.
Root owns GPUs/services; canonical Vashti and installed/public behavior remain
unchanged. Required Vashti changes:none.


### Intel-only grouped PLE physical receipt,2026-10-11 17:41UTC

Frozen66bdd0f/v162 (SHA498736a0bacd6a6d31c9610931658fff0c04cdee790fa996f20a9331ae19a9d2)
passes unchanged originalCPU13 cold/warm target vectors/carries/greedy, real
repair, full1381MAIN/42draft/432HOST-page close and physical recovery. Warm
selection reports8actual completed groupedPLE chunks; cold false/zero. Source
gate117Rust/239Python/9fixed-label checks and strict directClippy pass.

Compiler-idle same-artifact ordered normalcode200 is9.886106control versus
10.022470groupedPLE continuations/s (199timed,130/138drafts,69rounds,5repairs).
Candidate reports74actualgroupedchunks and302readbackbatches versus450control:
148 fewer batches. Both still report618GPUwaits; scoped map intervals remain
about2.63s. This small1.38% preliminary timing does not establish a repeated
gain or removed actualGPUwaits. No math/owner/cap/reference change.

Latest recovery17:41:33UTC: B70baseline50688000B,Nvidia compute0, allHTTP200,
idle/unpaused/no recovery and unchanged classifiedfaults on sameboot. Exact
numeric/source/helper/model/admission/close evidence is under
`v162_intel_grouped_reads_and_down_schedule` in the root native receipt.
DOWNstride and combined original numerical/throughput gates remain pending;
keep them off in the PLEpair. Next separate source work prepares immutable
grouped GR/GDN/QSA status folds (defaultoff), preserving all original checks.
CanonicalVashti/API/installedservices unchanged; requiredVashti changes:none.
No30tokens/s, normalprose, independentCPU200, production or release claim.


### Completed Intel-only v162 schedule comparison,2026-10-11 18:00UTC

The remaining DOWNstride and combined groupedPLE+DOWNstride gates pass original
CPU13 cold/warm vectors/carries/greedy, real repair and full source/owner closure
on frozen66bdd0f/v162. Warm DOWN reports432layers/2160DOWN/4320source calls;
combined also reports8actual groupedPLE chunks. Original reference/tolerances
and all admission/owner limits remain unchanged.

Compiler-idle same-artifact normalcode200:9.886106control,10.022470groupedPLE,
9.567741DOWNstride and9.558671combined continuations/s, each199timed and
130/138drafts/69rounds/5repairs. DOWNstride reports3552layers/17760DOWN/35520
source calls; combined also reports74groupedPLE chunks. These are single ordered
measurements, not repeated gain estimates. Keep DOWNstride off for the next
internal speed comparison; retain the preliminarily fastest groupedPLE selection.

Final recovery18:00:12UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on the sameboot. Exact
numeric/source/admission/closure records join v162_intel_grouped_reads_and_down_schedule.
Next separate candidates are grouped status folds and packed-integer DOWN;
neither is physically accepted yet. Root owns shared GPUs and source isolation.
Required Vashti changes:none; installedAPI/services/canonical checkout unchanged.
No30tokens/s, normalprose, independentCPU200, production or release acceptance.


### Grouped-status integrated source gate,2026-10-11

Grouped status leaf/parent0cdb291 (originalb9ebef5) and public controllers453d2b7
(originalff2578e) are integrated after the completedv162 Intel-only schedule
comparison. Read the normative [status folds](kuno-native-public-grouped-status-folds.md)
and [controller contract](kuno-native-public-grouped-status-controller.md).
Policies remain independent/default-off; original equations, both whole fault
proofs, all-family commit and source/owner closure remain mandatory. Completed
counts publish only after accepted commits and join97GR/36GDN/12QSA per chunk.

Source gates:25focused status checks plus280resident regressions in isolated
parent,90controller/helperRust and33protected guard checks in isolated controller,
strict direct all-target Clippy/fmt/diff pass. Root integrated245textPython checks
and six fixed Intel-only status/PLE+status selector/cap labels pass; synthetic Xe
metadata used solely for source checks. Physical launch still requires fresh
privileged Xe accounting after paused/drained admission.

Frozenv163 source453d2b76bb74a03e5c6aacacc586d48f2fc806a5,
binarySHA256bddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a,
pure direct-device, is undergoing originalCPU13 acceptance. Same-artifact
compiler-idle Intel normal comparison remains required; source acceptance is
not physical numerical or speed acceptance. DOWNstride remains off. Separate
integer DOWN source/parent/controller work is isolated and unselected. Required
Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Completed Intel-only grouped-status comparison,2026-10-11 18:23UTC

Frozen453d2b7/v163 (SHAbddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a)
passes unchanged originalCPU13 cold/warm vectors/carries/greedy/realrepair and
complete1381MAIN/42draft/432HOST-page closure. Warm initialprefill stays
false/allzero;8accepted statuschunks join776GR/288GDN/96QSA.

Compiler-idle same-artifact orderedcode200:10.096522groupedPLEcontrol versus
9.951850groupedPLE+status continuations/s, both199timed,130/138drafts,69rounds,
5real repairs. Candidate reports74accepted statuschunks7178GR/2664GDN/888QSA,
removing31228fixed statusdispatches; GPUwaits584versus618control,302readbatches
each. No throughputgain is established; keepstatus off in the next speedcandidate.
Retain source and independent composition for later work, with defaultoff.
Scoped phases/waits/maps overlap and are not additive.

Last recovery18:23:18UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on sameboot. Full numeric
records, hashes/admission/closures: v163_intel_grouped_status_physical. Source
tests and limitations recorded there; no repeatedABBA/prose/CPU200/30t/s claim.

Separate integerDOWN leaf58b7f50(original730e135) and parent28d8ed6(original87a6fbf)
are source-integrated only. Leaf7focused/17existing oracle tests and strict
Clippy pass; independent read-only parent review finds no blocker. Parent
regression/Clippy and controller integration remain pending. Read both integer
DOWN normative contracts. Mandatory original physical accuracy/repair/closure
and matched timing remain required; no accepted new arithmetic or speed yet.
Required Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Intel-only integer DOWN integrated source gate,2026-10-11

Separate default-off integerDOWN is source-integrated: leaf58b7f50(original730e135),
parent28d8ed6(original87a6fbf), tested annotation/deltaee4f200(originaldce5b5b),
controller6d1bb47(original0ed0d59). Read the linked integer DOWN leaf/parent and
[controller](kuno-native-public-frozen-down-integer-controller.md) contracts.
The source keeps original experts/routes/packed weights, full-Q8 speculative
MTP scope and original reference/tolerances, with separate alternate math.
Intel combined admission adds exact65536MAIN+462848runtime while reserving
65536HOST separately;196608INFO is already included in runtime. Original144banks,
8GiBcache, fullQ8 owning inventory and512MiB driver margin remain unchanged.

Leaf7new+17existing quantizer/oracle checks, parent13focused/281resident/20source/
13MoE/4constructor/17existing oracle checks, controller99Rust/38guard checks
and strict direct-device all-targetClippy/fmt/diff pass. Parent counts overlap.
Independent read-only parent review finds no blocker. Root integrated250Python
checks and three fixed Intel-only helper labels/caps pass; synthetic Xe metadata
used only in source checks. Actual physical launches require fresh privileged
Xe accounting after paused/drained admission. No scope or profile fallback.

Frozenv164 source6d1bb47cd54937a2dd08cb05d11062313168c862,
SHAe5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b,
pure direct-device, is undergoing originalCPU13 accuracy/repair/owningclose
and physical recovery. Paired normal speed remains pending. Current trial
retains PLE/GUstride/sharedQ8Intel; original DOWNrows/stride and groupedstatus
stay off to isolate new DOWN arithmetic. Source success does not establish
physical accuracy or speed. Required Vashti changes:none; installedAPI/services/
canonicalcheckout unchanged. No30tokens/s or release acceptance.


### Completed Intel integer DOWN comparison, 2026-10-11 18:50 UTC

Frozen source `6d1bb47cd54937a2dd08cb05d11062313168c862`, v164, binary SHA256
`e5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b`, passes
unchanged original CPU13 cold/warm full vectors, carries, greedy IDs, real
repairs and complete owning closure. Warm first token reports actual integer
DOWN 48/240/48 and 480 source dispatches; warm full pass reports
432/2160/432 with 4320 sources. Original reference and tolerances are unchanged.

One compiler-idle same-artifact ordered code200 pair measured **10.023723 →
10.486196 continuations/s (+4.61%)**, each timing 199 real continuations,
130/138 accepted drafts, 69 rounds and five repairs. Integer DOWN reports
3552 preparations, 17760 calls, 3552 audits and 35520 original source calls.
Keep integer DOWN as a preliminary faster candidate with integer GU/stride,
grouped PLE and original full-Q8 shared head, all on Intel. Grouped status and
original DOWNstride remain off. Nvidia is deferred at the user's direction.
Single ordered timing is not repeated ABBA or prose acceptance.

All 1382 target owners/828997632 bytes, 42 draft owners and 432 host pages close.
Last physical recovery18:50:02UTC: Intel50688000B, Nvidia compute0, allHTTP200,
idle/unpaused/no recovery latch and no classified GPU faults on the same boot.
Full source tests, hashes, actual admission/selection and closures are recorded
in `v164_intel_integer_down_physical`. No independent CPU200/private/production,
installed defaults/API, 30tokens/s or release claim. Required Vashti changes:none.

Next isolated source candidates: transfer the existing exact packed IQ4 lookup
into actual Verify3 dense IQ4_NL/IQ4_XS (distinct immutable pipeline policy,
unchanged original addresses/scales/dots/reductions), and a finite target GPU
submission-backlog experiment retaining original submissions and proofs.
Neither has physical acceptance yet. Root coordinates GPU and CPU timing;
separate worktrees preserve source isolation. See the new normative contracts
once integrated. No frontend or API work is part of these candidates.


### Intel Verify3 IQ4 decoder completed; queueing candidate next, 2026-10-11

Frozen v165 source6266201b44faacac187db8e1f2353d35f7438b5a passes all original
operator controls plus9408 exact packed-register IQ4 values,12normal+4fault
Verify3 cases and12normal+42fault fused-finite cases on Intel. Default binary
SHA996786943fb7dfc6f41e1ae975cb0659833d8cd372b673d474cd7f0f83195ad0;
pure direct binary SHA5f86159edbad0652e77b60a19046d6ef5933e936e8677c65ec8dc3626e988719.
Original CPU13 cold/warm full vectors, carries, greedy IDs and real repairs
pass without changing reference80650af8 or ABS0.006/REL0.0003. Cold decoder
staysOriginal/zero; warm actual89first/801full selected dense calls. Full1382
MAIN/828997632B,42draft owners and432HOST pages close with physical recovery.

Compiler-idle same-artifact code200 ordered pair measures10.440895control vs
10.408871register decoder (-0.31percent), both199real continuations,
130/138accepted drafts/69rounds/five repairs. This shows no speed gain in this
single pair. Leave IQ4 registers off for the next speed candidate; keep its
explicit implementation and numerical acceptance. No ABBA/prose/CPU200 or
production/default selection follows. Latest recovery19:26UTC: sameboot,
Intel50688000B/Nvidia compute0/allHTTP200/no recovery latch/no new GPU faults.
See full source/actual counts/closures in v165_intel_iq4_registers_physical.

Next runtime candidate is [explicit bounded target backlog2048](kuno-native-public-target-backlog-2048.md),
leafc8bccc4(original525ce3b), clarificationad4af97(original45d944d),
controller77e421c(original3d94692), protected guard633ac25. It retains original
kernels, queue submissions and closing proofs; target is selected before model
construction for bothcold/warm while independent Intel full-Q8 draft remains512.
Its opaque driver retention is not an exact byte guarantee; fresh actual Intel
admission and unchanged512MiB margin/caps remain required.111controller/helper
checks (including8new),7exact-source state-harness checks, strict direct all-target
Clippy and259integratedPython checks plus6fixed helper labels/caps pass. The
harness limitation is recorded in the normative contract. Physical CPU13 and
same-artifact normal speed comparison remain pending. No GPU reset, installed
service/default/API/frontend change or30tokens/s/release claim. Vashti changes:none.


### Intel target queue comparison complete; placement next, 2026-10-11

Frozen v166 source0d35d5c7d1f7daf040622be61009f4770ce2616b,
pure-direct binary450e32a18ff3b9f3d985a27cb9550564c6b7c29c83bdfb09ba0c8aa3199da045,
passes unchanged original cold/warm CPU13 full vectors/carries/greedy/real repair,
actual target2048/minimum1 with independent draft512/minimum0, full1382MAIN/
828997632B,42draft and432HOST close and physical recovery19:41:31UTC.
Actual selected runtime was constructed before model allocation for both passes,
and remained immutable through completed close. No kernel or route changed.

Same-artifact compiler-idle ordered code200 pair10.451238original512 vs
10.482525queued2048(+0.30percent), both199 real continuations/130of138 drafts/
69rounds/five repairs, shows no material gain in one pair. Keep original512
for the next placement candidate. Global submission waits618→281 and
8.294789→1.668612seconds, while mapped-read wait2.472111→8.967485seconds;
work largely moved to a different wait boundary. These overlapping timers
are not isolated GPU kernel cost and cannot be added as a removable speed gain.
Full numerical hashes, actual policy/counters/admission and closing evidence
are in v166_intel_target_backlog_physical. Integrated strict release/direct
all-targetClippy passed. No ABBA/prose/CPU200/private/production or30tokens/s claim.

Latest physical snapshot remains sameboot Intel50688000B/Nvidia compute0,
allHTTP200, idle/unpaused/no recovery latch and no new classifiedGPU faults.
Nvidia remains deferred. Installedservices/defaults/API/frontend unchanged.
Required Vashti changes:none.

Next isolated source preparation is setup-only public profile-ranked empty-bank
fill: preserve all existing prompt/cold-pass residents, use the pinned public
Strata ranking only for empty slots, then freeze the original144/layer bank,
all512 actual IDs and sorted368 HOST complement. No private prompt or future
route profile is recorded and no capacity/arithmetic/route change is selected.
Read docs/kuno-native-public-profile-ranked-bank-filler.md after leaf integration;
its exact artifact/derivation, full owner/cap and CPU13coverage limitations are
authoritative. Protected scope263Python checks +three fixed helper labels/caps
pass; leaf/controller numerical/physical/speed acceptance remains pending.
Source CPU13 selected preservation may be all-hit: original diagnostic HOST
coverage remains a separate required control, never fabricated. A separate
borrowed CPU-batch leaf is being prepared to compute existing RAM-resident
expert weights with original Rust AVX2 math; it is not whole-model inference
or a speed claim. Root owns cache scheduling, shared GPU tests and physical
recovery; separate source branches retain ownership boundaries.


### Intel public ranking tested; CPU/GPU splitting next, 2026-10-11

Fixed profile source is integrated through e5f1689, after normative7c9da20
(original192d662) and leaf1d14a14(original53e889a). Read
[kuno-native-public-profile-ranked-bank-filler.md](kuno-native-public-profile-ranked-bank-filler.md)
and its controller contract. Frozen v167 pure-direct source is
`e5f168916cfeccca051f53b4c5ead8958fbe90f2`, binary SHA256
`03379e116acddc20c0bbf73f253913f8156145f592e8fbd5b4fd635edf1c0430`.
Both separate canonical and selected cold/warm CPU13 gates pass unchanged full
vectors/carries/greedy/real repair, original tolerances, actual original512
queue/full-Q8 Intel MTP, full1382 MAIN/828997632B,42 draft and432 HOST closures.
Selected warm publishes48 layers,4121 filler/2791 preserved residents and
actually serves9120 VRAM/zero HOST members; the separate canonical diagnostic
serves2590 VRAM/6530 HOST. Preserving the preceding cold replay's residents can
make that replay all-hit; it cannot replace HOST coverage or credit a profile
benefit on unseen normal generation. The warm12-token reference-check intervals
are6.647453 versus11.553927 accepted tokens/s, not normal200-token throughput.

The compiler-idle same-artifact ordered normal code200 pair is10.482354 original
versus10.478727 profile filler (-0.035percent), both199 real continuations,
130of138 drafts,69 rounds and five actual repairs. Actual VRAM/HOST members
are56344/46376 versus56079/46641; VRAM share54.852percent versus54.594percent.
Selected normal setup preserves5143 residents and fills1769 empty slots;
actual48-layer publication and unchanged original full512 map are accepted.
There is no measured gain. Keep Original bank filling for subsequent trials,
original512 queue, original IQ4 decoder, grouped PLE and integer GU/DOWN/stride
with shared full-Q8 head; Nvidia remains deferred by the user. Counters are
routed members, not unique weight reads or PCIe bytes.

Exact source, original numerical and setup receipts, aggregate tiers, complete
owner/closure and recovery are recorded in
`v167_intel_profile_filler_physical` of the native Strata numeric receipt.
82 distinct leaf/parent checks,120 controller/helper checks,263 protected Python
checks plus three fixed labels/caps, and integrated strict release/direct
all-target Clippy pass. Latest actual recovery is20:09:26UTC, same boot,
B70 used50688000B, Nvidia compute0, allHTTP200, admission idle/unpaused without
recovery latch and no new classified GPU faults. This is one ordered code pair;
no ABBA/prose/independent CPU200/private/production/30tokens/s acceptance.
Installed services, selections, public API, frontend and release are unchanged.
Required Vashti changes:none.

The separate borrowed CPU batch source is integrated as2e51bd5(original3212bc2),
with53 CPU expert checks passing and two unchanged timing probes ignored,
strict CPU/native direct all-target Clippy, formatting and independent review.
Read [kuno-native-borrowed-cpu-batch.md](kuno-native-borrowed-cpu-batch.md):
counts1..3 use original packed F32/AVX2 arithmetic, joined workers and sparse
unweighted original-rank outputs. This leaf is default-unused; arbitrary spans
do not authenticate payload origin. No connected CPU/GPU inference or speed
claim follows. Separate source owners are preparing an Attachment-anchored
read-only borrow of the SAME original Vulkan RAM and an actual one-layer CPU
versus HOST-GPU partial comparison with readback/encoding/upload/join included.
No duplicate model payload, disk read, new arithmetic kernel, cap relaxation or
count3 merge is authorized by an operator result. Initial fixed-layer synthetic
operator measurements may be unusually L3-hot; actual whole-model CPU13 and
normal timing remain later gates. Root controls physical trials and compiler
scheduling; agents use separate worktrees/caches and cannot independently use
GPUs, change services or release.

### Attached CPU operator accepted; Intel MTP repair candidate next, 2026-10-11

Continue Intel-only target/full-Q8 MTP optimization. Native30tokens/s remains
unachieved; latest normal selected control is v167 at10.482354 continuations/s.
The current owner checkpoint is [the native-model handoff](kuno-native-next-model-handoff.md).
Canonical Vashti, installed selections and public protocols are unchanged.
Required Vashti changes:none.

Actual original HOST CPU operator v172 is frozen from
`47b972bf73f79c04acee3c87e922526203c54099`, binary SHA256
`107958180ec3334cc6e1c84b65976600b790c94539f12ca04cbd6c4f6e06175a`.
Read [the operator contract](kuno-native-public-host-cpu-batch-probe.md).
All45 synthetic one-layer case points/180 pairs pass original full
partial/inactive/padding/source/fault checks, maximum absolute error1.071021e-8
under unchanged1e-4/3e-4 bounds. Original coldCPU13 and warm token0 full-vector
checks pass. Actual original432 HOST allocations have HOST_VISIBLE,
HOST_COHERENT and HOST_CACHED mapping flags; no copied CPU model was introduced.
All11 operator owners/2392064B,1380 target MAIN owners/828866560B and432 HOST
pages/36147609600B close. No MTP ran in this operator diagnostic.

Complete15-worker CPU handoff/compute/upload means at count3/fiveHOSTranks are
2.324ms versus1.763ms original-F32 GPU,32percent slower; at tenHOSTranks3.394ms
versus2.852ms,19percent slower. Keep GPU execution for the current batched target
path. Some count1 points favor CPU; a single L3-hot synthetic layer and three
repetitions do not establish48-layer hybrid benefit or performance against the
selected integer GPU kernels. No connected CPU hybrid/whole-model speed claim.

The prior v171 failed private readback setup after original coldCPU13/warm0.
Its repeated full-owner audit stages necessarily exceeded the unchanged128MiB
I/O ceiling. Existing audited cleanup after every completed pair fixes the
lifetime without new runtime policy/cap/vendor code;180 actual checkpoints
inspect919937280B and take432.422ms outside both operator arms. Exact failure
and successful numeric/source/owner/recovery receipts are separate keys
`v171_attached_cpu_operator_readback_setup_failure` and
`v172_attached_cpu_operator_physical` in the native-Strata JSON. Latest physical
recovery at21:23:44UTC is same boot, B70used50688000B, Nvidia compute0,
allHTTP200, admission idle/unpaused/no recovery latch, classified faults0.

Source-only candidates integrated: original-math joined MTP repair leaf
`ed7b345` (original`0fa9cc3`) and guarded controller`e37ae50`
(original`012320c`); resident-bank ownership telemetry`f386606`
(original`859c2a5`). Read
[the repair contract](kuno-native-mtp-batched-repair.md) and
[the telemetry contract](kuno-native-resident-bank-telemetry.md).
49 MTP checks,131 controller/helper checks,9 telemetry/attachment checks and
strict direct all-target Clippy pass on their separate source trees. Root's
271 protected-launcher checks and three fixed helper scope/cap checks pass.
Physical originalCPU13/full closure/recovery and a same-artifact normal speed
comparison remain required for MTP repair. Telemetry distinguishes actual
backing padding from unused conservative admission headroom; neither is yet
measured on-device or credited as additional capacity. A separate four-row
dense leaf is being prepared without changing original count3 behavior.


### Intel MTP repair and exact bank backing accepted, 2026-10-11

Frozen v173 source970f8f4141adc903212fe102ada504b4a42edfe8, pure-direct binary
69bc565a1396d85d9f9fdaa702620d94b44795c15b4541329f7f4d502c8ec0ec, passes
unchanged originalCPU13 cold/warm full vectors/carries/greedy/actual repairs.
Same-artifact compiler-idle ordered code200 pair measures10.533053 original
scalar versus10.603973 batched repair continuations/s (+0.67percent). Both199
real continuations/130of138 accepted drafts/69 rounds/five real target repairs.
Actual new69 joined repair calls/130 cells/69 fault gates reduce MTP reads
337to207. Repair phase1.802to1.590s includes true target recomputation/checks,
not isolated readback cost. The explicit leaf is retained; original scalar
repair remains the separate four-row candidate control. One pair does not
establish repeated/prose/independentCPU200 or30tokens/s acceptance.

Exact48-layer/288-buffer bank telemetry reports14,144,716,800B logical payload,
14,557,240,448B actual buffer backing,412,523,648B padding and4,166,132,608B
unused conservative allowance. Attached/close snapshots match. The latter is
not free physical VRAM or additional admission credit. No capacity or GPU
owner changes. Complete1382MAIN/828997632B,42 draft private owners/11698176B
and432 HOST/36147609600B closure passes. Last normal recovery21:43:44UTC:
same boot, B70used50688000B, Nvidia compute0, allHTTP200, admission idle and
unpaused/no recovery latch, classified faults0. Exact sources/actual counters/
admission/closures are in v173_intel_mtp_batched_repair_and_bank_backing_physical.

Next isolated source candidate is genuine four-row target verification with
three actual Intel full-Q8 MTP proposals, original scalar repair and unchanged
original tails/caps/reference. Read the normative parent/dense/MoE/controller
contracts when integrated. Physical acceptance remains separate. Canonical
Vashti, installed services/defaults, API and release are unchanged. Required
Vashti changes:none. Nvidia remains deferred at the user's direction.

### Intel four-row operator acceptance and constructor rejection, 2026-10-11

Current owner source42df907 reserves genuine four-row lookahead only in the
fixed-public Intel target/full-Q8 MTP lab. v174 independent Verify4 plain/fused
operators pass original scalar-GPU/CPU-F64, all10 formats,20 numeric each,
20/70 fault cases and complete source/owner zero checks. Original CPU13
cold/warm full vectors/greedy/carries/real repair and complete closure pass.
Selected four-row CPU13 fails before initial count1 prefill acceptance:
GR incorrectly proves the constructor-reserved four-row logits owner using
its older three-row logical extent. No third-proposal/count4 or normal speed
acceptance follows. Carry fixed owner capacity into the exact proof, preserve
original arithmetic/reference/caps, freeze and retest before timing.

See the latest owner handoff and
v174_four_rows_operators_and_initial_owner_failure in the numeric receipt
for exact source/binary/guards, original controls, failure and recovery.
Latest physical recovery22:32:25UTC: same boot/kernel, B70 used50688000B,
Nvidia compute0, allHTTP200, admission idle/unpaused/no recovery and no new
classified GPU fault. These correctness runs may overlap CPU source checks;
diagnostic rates do not establish compiler-idle normal throughput.
Native30tokens/s remains unachieved. Installed services, public contract,
canonical Vashti and frontend unchanged; required Vashti changes:none.

### Intel-only native optimization continuation, 2026-10-11

Latest user direction is Intel B70 target and full-Q8 MTP only; defer Nvidia.
Latest accepted normal comparison is v177 on26295f6:10.442972 original
three-row versus10.740576 genuine four-row real continuations/s (+2.85percent),
one compiler-idle same-artifact ordered code200 pair. Both emit200/time199
real continuations, complete all real repairs with zero repaired-vector error,
close every original target/draft/HOST owner, and recover normal services and
same-boot B70baseline50688000B with no new classified faults. Four accepts
147of155 proposals over52 rounds/51 actual count4 chunks; original130of138
over69 rounds. Independent third-draft CPU oracle, repeatedABBA/prose and
independentCPU200 acceptance remain absent. Native30tokens/s is unachieved.

Original and selected cold/warm full CPU13 vectors/carries/greedies and real
repair pass on the same artifact. The unchanged independent v174 Verify4
operators are reused after three arithmetic-free setup repairs: immutable
reserved head capacity (`e6baa7e`), mandatory original direct-fill attachment
(`2dbc897`) and actual frozen-four authority (`26295f6`). Original direct
demand loading stays capped at3. Natural prefixes1..4 occur across CPU13 and
normal fixtures. Failed v174..176 runs retain their distinct failure receipts.
Exact source/owners/recovery and limitations are in
v177_actual_frozen_four_cpu13_acceptance and
v177_original_vs_four_normal_intel_pair.

Read the current [owner handoff](kuno-native-next-model-handoff.md) and
[four-row contract](kuno-native-target-four-rows.md). GDN's retained-metadata correction (`cd638c7`) passes original/four cold/warm
CPU13, full closures/recovery and normal code200. v17810.704157 versus
v177four10.740576 gives no measured gain (about0.34percent slower in one
frozen comparison). Real work/repair/prefix counts match. Its35 source checks
and strict direct Clippy pass, with original arithmetic/shaders/caps unchanged.
See v178_gdn_retained_metadata_cpu13_acceptance and
v178_gdn_retained_metadata_four_normal. A bounded original three-row CPU-HOST/GPU-resident overlap operator is
registered on d587d9b (source/admission checks pass). First v179 physical attempt
fails a sparse negative fault check after earlier case progress; no timing or
completed numeric credit is claimed for that failure. Correction `47f4c8e`
passes v180 original CPU13/warm token0, all five cases/75 full arm calls,
all11 negative controls and complete source/owner/recovery checks. Concurrent
CPU/GPU wins disjoint HOST work against the original-F32 control, but loses
the other actual mixed cases. No whole-model or selected-integer GPU speed
claim/consumer is granted. Same-boot normal services/B70baseline50688000B
recover without a new classified fault, reset or reboot. See the
[operator contract](kuno-native-public-cpu-host-overlap.md) and full receipt
`v180_attached_cpu_overlap_physical`.

The user paused the goal2026-10-12UTC; the current benchmark/source block
is complete and no further GPU run is started. Native30tokens/s remains
unachieved; latest whole-model
result remains10.7tokens/s. Unused fused-four MTP repair leaf `bd46b2b`
(original `7ea302c`) passes56 focused tests/strict direct Clippy. Approved
controller composition `83967bc` (original `fc0948c`) passes225 distinct
focused tests/strict direct Clippy. Only actual count4 uses fused repair,
tails keep original repair, and actual completion counters join both paths.
Read the [exact controller contract](kuno-native-public-four-batched-mtp-repair-controller.md).
Root combined strict direct all-target Clippy, full native formatting and diff
checks pass after integration; compiler/CPU/GPU test jobs are idle. The source-only
receipt is `paused_intel_only_mtp_four_repair_source_checkpoint_20261012`.
Protected activation and
unchanged physical CPU13/repair/full timings remain pending. Old1..3/default
paths, math,42 owners and all caps stay unchanged. Read the
[owner checkpoint](kuno-native-next-model-handoff.md) before resuming.
The unused exact persistent144 bank leaf
(`e16ac7a`) grants no additional physical capacity or slots. No API, installed
service, canonical Vashti, frontend or release change; required Vashti
changes:none.


### Latest native text checkpoint — paused after measured profiling (2026-10-12)

This supersedes the preceding pending fused-four-repair checkpoint. User stopped
the Astra manual pass to conserve usage. Native goal remains paused, no queued
CPU/GPU work. Fused-four repair and optional integer target head pass full
CPU13/closures. Normal v182 control10.882673 versus integer16head10.880799
real continuations/s is neutral, with matching generation/acceptance counts.
Paired32 head passes CPU13 but has no normal-speed result. No30tokens/s claim.

v185 onaa90a95 physically profiles the first warm target round: expert GU/DOWN
consume69.86% of250.572596ms measured kernel time; BF16UP only1.04%. Shelve
source-only BF16pair a9cff9e; next approved design is GPU-only compact Source
group lists, not implemented. Preserve exact routes, original arithmetic,
full audits and scratch-owner limits. Resume from the
[owner handoff](kuno-native-next-model-handoff.md) and
[profile contract](kuno-native-public-target-profile.md), with source/artifact,
checks, design and limitations recorded there and in the aggregate receipts.

Fresh02:12:05UTC recovery confirms Kuno/Ollama/Vashti HTTP200, admission idle
and unpaused, B70baseline50688000B, Nvidia compute0 and all classified faults0.
Same boot/kernel38. All root/agent compiler and GPU jobs are idle. Canonical
Vashti, installed defaults, public API, frontend and release remain unchanged;
required Vashti changes:none. Do not automatically resume work.

### Native text checkpoint — count4 compaction implemented and committed (2026-10-12)

Resumed on user instruction after an explicit pause. Per-Source expert group
compaction is now implemented and committed as c42379b on
codex/kuno-native-text. The GPU prepare leaf builds five disjoint shard
records in the already-owned zeroed tail (words 6144..6383 of the existing
full 64KiB owner): ready flag, count, the original GU/DOWN dispatch triplets,
and the stable ascending original group IDs for the unchanged partition ranges
0/72/144/288/432. GU and DOWN then run one indirect dispatch per shard through
checked 12-byte views of that owner, so each leaf reads only its own shard's
groups instead of scanning the routed-group union.

Arithmetic, decoder, member order, routes, IDs and owners are unchanged: both
compact projections reuse the original body verbatim, count1..3 stride-four and
all other schedules keep their original paths, and the new record audit
reconstructs each list from the unchanged MAP, validates headers, counts,
triplets, identity and inactive zeros, stickies faults, and erases the records
only after every shard validates. Counters are checked: each compact leaf
rejects unauthenticated or short views, and the audit refuses a 16384-word
scratch. Host receipts now require actual compaction credit (five GU and five
DOWN indirect dispatches plus one record audit per layer) and both daslab
probes report those counters as evidence.

Checks: full native-text suite 1334 passed/0 failed/1 ignored, clippy
-D warnings clean, fmt clean. No GPU trial has run yet. Admission is idle and
unpaused, but an unrelated 13.3GB worker process currently holds the B70, so
the numerical and normal gates must wait for that GPU to be free. Frozen v186
artifact build is in progress from c42379b. No performance claim yet.

Canonical Vashti, installed defaults, public API, frontend and release remain
unchanged; required Vashti changes: none.

### Latest native text checkpoint — stopped after completed kernel comparisons (2026-10-13)

This supersedes the preceding compaction source-only/interrupted checkpoints.
User requested finishing the current experiment and stopping until tomorrow;
automated goal remains paused and no work is queued. Canonical Vashti is clean
and unchanged; this work stays in the Kuno native-text laboratory worktree.

Inherited compaction is fixed in `62594d0` and `fb7b420` and passes full original
cold/warm CPU13 plus normal code200. Normal10.906tokens/s is neutral against
prior10.883. Primary-plane GU (`fd9c319`/`fdd2732`) failed the unchanged warm
carry fixture and is disabled. Exact-tree subgroup GU/DOWN
(`7d01db5`/`70fd470`/`b9ee1be`) passes CPU13 but measures10.955tokens/s, neutral.
Exact activation-staged DOWN (`41b9d3d`/`d8b58b6`) passes full CPU13 and measures
10.872tokens/s, also neutral. Both experimental selectors are off in `9dc764d`.
Original numerical thresholds, target routes and two-plane arithmetic stay intact.

Full selected-v197 direct-device suite:1346pass/1ignored; strict release
Clippy/fmt pass. Frozen source/artifact identities, real timings, failed/neutral
evidence, specifications and exact resume directions are in the updated
[native owner handoff](kuno-native-next-model-handoff.md) and aggregate receipts.
The short warm CPU13 fixture improved20% with staging but normal generation did
not; next work should profile the representative normal run and verify actual
count4/tail execution before further kernel changes.30tokens/s is not achieved.

Fresh06:54:29UTC protected cleanup: Kuno/Ollama/Vashti HTTP200, admission idle
and unpaused/no latch, B70used50688000B, Nvidia compute0, all classified GPU
faults0, unchanged boot/kernel. No benchmark or compiler processes remain.
No installed binary/default, public API, frontend or release change. Required
Vashti changes: **none**. Do not resume automatically.

### Native text: broad Rust Strata port checkpoint (2026-10-13)

The user explicitly requested a coherent Rust port first, followed by full-model
execution and bug fixes, superseding incremental original-CPU13-precision gates
for this isolated experiment. Work is now on `codex/strata-rust-port` in the
sequential Kuno worktree. Original accepted engine remains on `codex/kuno-native-text`
at6792453; canonical Vashti is untouched. This is backend laboratory work and
requires no consumer/API/frontend change.

Prototype target math/scheduling and suffix/Q2 MTP controller are implemented;
source `f0ed07a` passes build,20full WGSL/SPIR-V compiler variants, both pure
registration presets, strict port Clippy, and1346original-engine regression tests
(one ignored). It reuses existing Rust model operators. Full Strata feature
coverage is not claimed: important cache/prefill/MTP/context features remain.

**No GPU execution, quality or speed acceptance yet.** Cached sudo expired;
one shared-shell prompt is pending. No services/models/GPUs were changed or
loaded. Fresh unprivileged checks show all three HTTP listeners200, B70baseline
50688000B, Nvidia compute0. Exact scope, immutable v204 artifact, defects fixed,
limitations and first full-model command are in the
[port checkpoint](kuno-strata-rust-port.md) and
[owner handoff](kuno-native-next-model-handoff.md). Do not use earlier port
artifacts with the shader-cache identity bug; retain both dtype/variant and mode
in one KernelId info tuple. No deployment/release or automatic work is queued.

### Latest native text port checkpoint — paused after combined test (2026-10-13)

This supersedes the preceding source-only and early combined-port checkpoints.
User requested pausing after v221 and returning both GPUs. Automated goal stays
paused; no benchmark/compiler job, sudo refresher or queued GPU work remains.
Canonical Vashti, deployed binaries/defaults and API/frontend are unchanged;
required Vashti changes: **none**. No release acceptance.

Runtime source is1caf271 oncodex/strata-rust-port. Original accepted engine remains
6792453 oncodex/kuno-native-text. Broad port now connects parallel routing,
compact real argmax/carry, Rust/RAM Q2 dense conversion,106299-token proposal
vocabulary, suffix policy and request-local adaptive two-way cache. Newest source
has exact persistent192-slot banks and a GPU-driven VRAM/mapped-RAM exchange.
The complete v215144-slot/staged-exchange run measured8.458796 continuations/s
through362 emitted tokens.30tokens/s and the full upstream rewrite remain unmet.

v221 reaches cold prefill/materialization but rejects a four-row DOWN launch
using two old partition tables. d682e04 fixes those;1caf271 passes strict port
Clippy/fmt and original direct-device compilation. **No GPU rerun after the fix**;
new direct GPU exchanges have no hardware/timing acceptance. Current full
original-engine regression is still outstanding (1346/1ignored was f0ed07a).
Do not reuse the failed v221 artifact as working. Read the exact scope, artifact,
failures, next acceptance and remaining verifier/prefill/rollback/KV work in the
[port checkpoint](kuno-strata-rust-port.md) and
[owner handoff](kuno-native-next-model-handoff.md) before explicit resume.

Fresh16:29:23UTC read-only stop receipt: Kuno/Ollama/Vashti200, admission idle and
unpaused/no latch, B70used50688000B, Nvidia compute0, classified faults all0,
unchanged boot/kernel38. Recorded refresher3421970 exited. No prompts, outputs,
private histories, API changes or frontend work were persisted/introduced.

### Latest native text checkpoint — resumed combined optimization (2026-10-13)

User explicitly resumed goal/GPU work; this supersedes the paused/v221 entry.
Accepted v224/0f4bfcc produces12.302385code and12.241920prose continuations/s.
192-slot direct GPU/RAM exchange and retained-input state-only prefix repair
work; code36 and prose95 partial repairs require no full target reruns. Both
fixtures complete private/source closure and actual GPU/service recovery.
30tokens/s and full Strata parity remain unmet. v225 integer head and v226 fused
five-tier projection are unaccepted candidates under combined validation.
See exact sources, frozen hashes, measurements, limits and next work in
[port checkpoint](kuno-strata-rust-port.md). No frontend/API/deployment change;
required Vashti changes:none. Normal source/physical bounds and transient
privacy remain despite approximate port arithmetic. Canonical checkout preserved.

### Native text continuation — queued MTP accepted (2026-10-13)

v231/01e74a7 reaches12.755259code and12.598536prose continuations/s. Consecutive
real MTP proposals now stay queued until one final private fault/read fence;
target verification/cache repair unchanged. One additional admitted/erased32KiB
MTP packet; port closure43owners/11,730,944B. Both actual recovery receipts pass.
Head/five-tier fusion tested neutral/slower and remain optional, off by default.
v232 router/v233 aligned expert reads are candidates under validation. Exact
contracts/measurement limits and incomplete full-rewrite scope are in the port
checkpoint. No consumer changes, installed default or release acceptance.

### Native text active checkpoint —13tokens/s and profile-directed next work

v234 default prose13.000929; v232 normalcode12.796036. Latest3e01731/v236's
optional8promotions/layer gives13.004114code with81.66%resident routes; larger
transfer cost offsets much of faster verification. Cache-eight prose is active.
All finished runs erase43MTP/1562target owners and physically restore services,
GPU memory and unchanged kernel faults. Current strict port Clippy and original
1346pass/1ignored; full49shader variants indefault/16/wide atv234.30 and full
Strata parity unmet. Head/fusion/wide/16variants remain optional/off. The new
profile points to mapped-HOST expert work; port guide gives exact next6/8-row
suffix/staging directions and their ownership/shape limits. No API/UI/installed
or release changes; required Vashti changes:none. Goal active; old stop notes
historical. Do not infer the bounded sudo helper's availability from notes.

### Completed cache-eight checkpoint — active goal, recovered host (2026-10-13)

v236/3e01731 optionalcache-eight:13.004114code,13.189027prose. Both source/private
closure and physical GPU/service/fault recovery pass. Clippy and original1346
pass/1ignored pass. Benchmark/compiler jobs ended; bounded refresher stays for
ongoing authorized goal. Default cache2,optional8; no deployed setting/API/UI
change.30still unmet. Port guide now records concrete6/8-row source/ownership
work needed for the next coherent port step and profile-based transfer options.


### Latest native text port checkpoint — paused, GPUs returned (2026-10-13)

Supersedes the preceding active-goal notes. User requested a breaking point;
goal is paused, no GPU benchmark or sudo refresher remains. Kuno/Ollama/Vashti
HTTP 200, admission idle/unpaused/no recovery latch, B70 50,688,000 bytes,
Nvidia compute zero, all classified GPU faults zero. No installed/API/UI/release
change; required Vashti changes **none**. Canonical checkout remains untouched.

On `codex/strata-rust-port`, the optional eight-row Rust verifier now completes
real whole-model target verification and state-only partial-prefix repair.
Runtime source `2780e7a`, frozen SHA256
`33a568e1b2fa6b492381e14c6488c8f80fac12c4611a2d73ac4c457bf556bff4`:
12.372330 code with adaptive five/eight-row windows; normal four-row MTP on the
same binary 12.894535. Both emit 362 including EOS and fully erase private owners.
Source-only helper cleanup follows at `383248a`; strict Clippy and original
regression 1,346 passed / 1 ignored. Frozen eight-row shaders and prefix tests
also pass. v236/`3e01731` remains the
selected faster candidate: 13.004114 code / 13.189027 prose; **30 is unmet**.
Read the [current paused port handoff](kuno-strata-rust-port.md) and
[aggregate receipt](../apps/kuno/benchmarks/text/receipts/strata-rust-port-20261013.json)
for exact identities, checks, packet-receipt caveat and remaining coverage.
Next candidate is selected HOST expert staging/promotion, not API integration.
Do not infer sudo authorization or free GPUs from these historical notes.


### Latest native text checkpoint — combined residency complete; coherent port next (2026-10-13)

This supersedes preceding native-engine next-step proposals. User directs a
connected rewrite of the pinned MIT Strata Intel core before whole-model speed
judgment, followed by integration debugging. Do not performance-gate individual
ingredients while upstream dependencies remain missing. This remains native
backend laboratory work; no Vashti/API/connector/UI changes are required.

Working fallback v260, runtime source
`86a98a4def023ce4bd5834a0cdf40087dd938f1e`, SHA256
`b34e3b83dc3fa5297199ef22ef1e15f1ee4860a65b1d40f0768f014b96667392`:
14.063997 code / 14.300613 prose timed continuations/s on Intel B70 only, public
seed424242/temp0 fixtures. Exact static allocations, 5 GiB static cap, 240
resident experts/layer, ranked empty-slot fill and eight cache promotions are
combined. Four-row MTP, real target verification, 36/95 state-only repairs,
zero full target recomputations. These are single runs, not matched-output
Ollama controls or private/production acceptance. **30 and full rewrite unmet.**

Later optional full-profile initialization v261 gives13.884123code; single local
negative comparisons do not reject the complete upstream design. Latest checked
source a0bd715 passes strict selected port Clippy, two targeted port tests and
1,346 original direct-device tests /1ignored. Frozen v260/v261 shader sets pass.
Read the [current port checkpoint](kuno-strata-rust-port.md) and
[aggregate receipt](../apps/kuno/benchmarks/text/receipts/strata-rust-port-20261013.json)
for exact commits, options, limits, failed startup trials and remaining gaps.

At23:11:21UTC all three listeners200, admission idle/unpaused/no latch,
B70used50,688,000B, Nvidia compute0, classified GPU faults0. No benchmark/compiler
or sudo refresher remains; goal controller paused. Development branch is
`codex/strata-rust-port` in the separate `kuno-image-speed` worktree; canonical
Vashti and deployed binaries unchanged. One graceful Kuno restart evicted idle
image residency before trials; no device reset/rebind/reboot. No frontend handoff,
release or installed performance claim is implied. Next scope is connected
placement/weight-access/router/group-execution/quantized-kernel/verification/state
commit parity with explicit Vulkan substitutions, keeping v260 as fallback.


## Native Intel coherent core checkpoint (2026-10-14)

Backend-only work stays on `codex/strata-rust-port` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`; canonical source and
installed Kuno/Ollama/Vashti binaries/configuration are unchanged. Read the top
and **Native Intel core receipt** in [the port handoff](kuno-strata-rust-port.md)
for exact reproduction/limitations. No API, protocol, runtime ABI or UI change;
**required Vashti changes: none**. Do not package this fixed-public laboratory
binary as an accepted private text worker.

The connected range after `ca2cadb` includes native Rust/ash Vulkan command
recording, retained buffer ownership/initialization, shared IQ codebooks, native
physical expert reads, stable parameter pages and ordered private copies/fills.
Key commits: `0e6f2b7` initializes Wgpu ownership correctly; `049d493` tracks real
buffer dependencies; `2f53b6c`/`66b198d` lower/check physical pointers;
`1c9c9d8` preallocates pointers and stabilizes parameters; `617537b` orders private
copies; `a68ac87`/`f2d4048` match subgroup32 and record large replay counts;
`55adf7d` adds six packed dense attention formats; `9dc6eeb` stages selected
original RAM experts in three genuinely owned64MiB private GPU roots;
`9be8a65`/`44bb1d1` charge/check those owners at construction, physical admission
and closure; `a0c6c3c` bounds the complete stage union. Retain licenses and the
source/error-order fixes with the optimized path.

Frozen v282 source `a0c6c3c9c21745faed6ee7362983bcd00f5a72e6`, SHA256
`9b41ea8fb312ad1fd1ddb3f5986eb479832c5ab9d173c0ffdf51188c102f38c6`, completes
real DASLab IQ3_S/Q2 MTP code14.625229 /prose13.546277 timed continuations/s.
139 /165 large verification graphs replay. Actual target erasure proves1565
buffers /1,064,534,016 bytes; MTP43 /11,730,944 bytes; all original host source
pages detach, B70 returns to50,688,000 bytes, NVIDIA compute processes0, classified
GPU faults0 and normal HTTP/admission recover. Source checks: strict core Clippy,
fmt/whitespace,56 WGSL/SPIRV sources including13 physical `spirv-val` modules,
and1346 original direct-device tests /1 ignored. Full Strata-feature unit suite,
independent whole-model state oracle, private worker/API acceptance and long soak
are not established.30t/s and complete upstream parity remain unmet. v260 is
retained as a faster prose fallback; generated token sequences/lengths differ,
so these are not same-output speed ratios. The existing goal is active and
continues; repository notes must preserve that unfinished scope.


## Native Intel packed shared/MTP and reuse checkpoint (2026-10-14)

This supersedes the preceding native-engine source/checkpoint details. Backend-only
work remains in the isolated `kuno-image-speed` worktree on `codex/strata-rust-port`.
Canonical/deployed source, API, protocol, UI and services are unchanged;
**required Vashti changes: none**. Read the top of
[the current port handoff](kuno-strata-rust-port.md) and aggregate receipt's
`native_intel_packed_extension` before resuming.

Changes `0dad3af`, `50a112e`, `e5bd05d` connect packed shared gate/up/down, scalar
MTP projections/indexed head, subgroup32 input preparation, and GDN/QSA input
reuse. Frozen v285 source `e5bd05d429c0f37dceb6a54735ad0b9ea39c424e`, SHA256
`1319a190a246ab7adcff7055629021db4672a128ddd5e944de3350189f5a69d2`:
14.512497code /12.781750prose timed continuations/s. Code improves over v283/v284
in this single trial, prose does not. Sequences/windows differ; independent
numerical/quality acceptance is absent. Keep v282/v260 fallbacks.30 and complete
Strata parity remain unmet; goal remains active.

Strict core Clippy,65 shader variants including13 physical SPIRV checks, and
original regression1346passed /1ignored. All trials restore actual B70 idle
50,688,000B, NVIDIA compute0, classified faults0, HTTP200 and normal idle admission;
full private target/MTP owners erase and original host pages detach. No reset,
reboot or install. Full core unit suite, private worker/API acceptance, whole-state
oracle and soak are not established. No benchmark/compiler remains; the bounded
sudo helper may still be live and must be inspected, not assumed. Next scope:
public operator timing, lower-register expert scheduling and connected GR fusions,
with packed-epoch/vector verification for the changed prose path.


## Active native-text direction — complete Rust rewrite before speed testing (2026-10-14)

This supersedes all preceding native-text next-step/performance experiment notes.
The user deleted the previous goal and explicitly requires the complete connected
Strata Intel inference path in Rust **before any live testing for speed**. Read
[the completion gate](kuno-strata-rewrite-completion.md) and the top of
[the port checkpoint](kuno-strata-rust-port.md). Intermediate checks may establish
build/shader validity, numerical/state correctness and successful execution; speed
reports must not change direction or justify dropping coupled components.

New goal is active. Working branch `codex/strata-rust-port`, starting runtime
`7455b6f`, isolated `kuno-image-speed` worktree. No canonical/deployed source,
API/protocol/UI or release changes; required Vashti changes **none**. Completion
is not claimed: global expert allocation, full GR/native schedules, prompt prefill
and general request connection remain among the explicit audit gaps. Preserve
existing source, public comparison evidence, Q2 sidecar and attribution.

First connected block: `d58bce4` selects native separate gate/up/DOWN 4/2/1
chunks, activation and scratch reuse together, with all extra INFO charged. The
protected runner requires execution-only validation and blocks speed/profile
trials. Release build, strict core Clippy, 158 WGSL/SPIR-V variants (105 physical
checks) and original regression 1,346/1 ignored pass. **No live GPU inference or
speed test, complete rewrite, numerical model acceptance or production claim.**
The next implementation continues the connected upstream dependency checklist;
no per-component speed selection. Logs live under `kuno-development` with names
`strata-connected-expert-{build,final-clippy,shaders,original-regression}-20261014.log`.

Connected GR block: `56663a3` ports pending-write normalization, resident BF16
sliced down/injection, fixed reduction and up/mix/commit together. Explicit writes
protect PLE, compact final rows and no-head prompt chunks. The reused private UP
owner grows by 524,288 bytes, charged in both family and complete-model quotes.
See the completion guide's **Connected GR implementation block** for backend
substitutions, source paths and exact logs. Strict core Clippy/release build,
163 shader variants (105 physical checks), 12 core GR owner/layout tests and
original direct-device regression 1,346/1 ignored pass. Full core unit suite is
**not green**: 1,148 pass, 200 fail, one ignored; old layout/budget assumptions and
other failures still need scope/contract triage. No live GPU inference, speed
trial, model numerical acceptance, installation or consumer change is claimed.
Goal remains active and the full rewrite incomplete; continue the dependency
checklist, preserving the connected port instead of selecting an intermediate
speed winner. Required Vashti changes: none.

Global allocation/loading source: `65e6e19` implements pinned Strata's native-pack
ranked byte-prefix plan from actual original graph metadata and an owning bounded
persistent GPU loader under existing admission. The same expert-byte budget gives
11,570 public-profile pairs and 160..342 residents/layer. Host-only prefix/source
coverage tests and strict core Clippy pass. **The GPU loader was not executed and
inference consumers still use the old uniform banks.** Read the completion guide's
**Global byte-prefix and owning-loader block** before continuing: replace banks,
HOST imports, per-expert BDA ownership, routing/grouping and exchange together;
do not allocate both caches or force the new layout into fixed old extents.
No GPU speed/profile, API/UI, installed binary or service change. Required Vashti
changes: none. Goal active, full rewrite incomplete, earlier core-suite failures
still open. The refreshed bounded helper's path/process audit is in that handoff.


### Active Kuno engine handoff extension — global source owner (2026-10-14)

Source commit `47d330f` on `codex/strata-rust-port` in the existing isolated
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti` checkout. Read the completion
guide's **Global HOST, source owner and native read dependencies** block before
continuing. It records exact construction/ownership/closure, validation and the
next connected consumer replacement.

The new global Sources owner combines one ranked GPU cache, only its variable
original HOST complement, actual per-expert physical addresses and native read
hazard/recording dependencies. Exclusive executor admission and legacy-cache
exclusion are implemented. Core build, strict Clippy, all 163 shader validations,
three layout tests and ordinary regression (1,349 passed, one ignored) pass.
**The inference coordinator does not call the new constructor yet, and no GPU
load/execute/recovery has been run for this source owner.** Full rewrite, numerical
state acceptance and the earlier experimental core suite failures remain open.
Next connect construction before prefill, routing/HOST fetch/rebase, separate-role
expert execution and private adaptive remap/publication as one coupled block.
No intermediate speed selection or disabling port components.

API specification/installed binary/services/Vashti behavior are unchanged;
required Vashti changes: none. No GPU testing, reset, rebind, installation or
release in this block. Canonical checkout and inherited untracked bytecode are
preserved. The goal is active and incomplete. Existing bounded sudo refresh remains
an authentication-only helper; it is not authority to bypass unavailable credentials.


### Active Kuno engine handoff extension — connected global core (2026-10-14)

Source `d24c348`, same isolated `codex/strata-rust-port` checkout. Read the completion
guide's **Connected global core dispatch block**, which supersedes the preceding
uncalled-constructor handoff without claiming runtime acceptance.

The core now initializes the global cache/HOST source owner before first prefill
and selects a new48-layer driver with GPU routing, private remap/group planning,
HOST fetch/rebase/stamps, native gate/up/activation/DOWN, weighted reduction and
real completion stamps. Each role uses one dispatch with pinned row_entries4/2/1;
IQ4_XS takes the scalar path, and native integer finish/XOR/IQ4_NL scheduling are
ported. Exact family/stage/table quotes are updated. Core release, strict Clippy
and179 shader validations pass. Final ordinary regression passes:1,349 tests,
one ignored. The earlier200 full-core failures are not resolved by this.

No GPU inference, physical closure/recovery, speed trial, installation or service
change. The old language/speculative controller still has frozen-bank receipt and
activation assumptions that can reject after prefill. Next port adaptive exchange
and private remap publication, then replace those controller assumptions with
actual global contracts. Complete1..8 target/MTP/prefill/state support and the full
checklist remain required. Audit original-GGUF versus upstream native conversions
and actual Intel CPU/transfer/overlap mode; do not claim complete parity yet.

No API or Vashti consumer changes. Goal active; full rewrite incomplete; speed
remains blocked. Preserve canonical work and inherited untracked bytecode.


### Active Kuno engine handoff extension — adaptive global cells (2026-10-14)

Source `e1dc698` on the same isolated `codex/strata-rust-port` checkout. Read the
completion guide's **Connected global adaptive exchange** block for exact policy,
publication/ownership, reservations, validation and remaining substitutions.

The global core now connects native every4/global96/gain1.5/decay0.7 adaptation,
unequal physical GPU/HOST cells, paired original-role exchange and private
logical remap/usage publication after actual joined boundaries. Static address
tables remain unchanged; evictions are retained in HOST without file reads.
Core build/Clippy and179 shaders pass. Three pure policy/exchange simulations and
ordinary regression (1,352 passed, one ignored) pass. No GPU/Vulkan exchange,
whole-model numerical/state/recovery or speed acceptance was run.

Next port global language/speculative controllers and real receipt/activation
contracts; reconcile prompt versus verify adaptation windows, complete1..8
prefill/target/MTP/state paths and the remaining fidelity audit. Current in-place
blocking Vulkan exchange is a documented backend substitution, not async-stream
parity. Earlier full-core failures remain open. No API/Vashti/install/service
change; required Vashti changes: none. Goal active, full rewrite incomplete,
speed gate still blocked. Preserve canonical work and inherited bytecode.


### Active Kuno engine handoff extension — global generation controller (2026-10-14)

Source `9abc0fd`, same isolated `codex/strata-rust-port` checkout. Read the
completion guide's **Connected global request and verification controller**
section; it supersedes the stale statement that core language execution still
requires old frozen-bank receipts.

Core now enters its own connected global-source controller: teacher prefill,
real MTP/suffix proposals, target verification, actual accepted-prefix state
repair/commit, MTP carry repair, cursor checks and transient accepted-token
callback. Real source/admission/graph authority replaces frozen-bank identities;
shared GDN/PLE/status/head selection uses that authority. Private usage excludes
prompt/replay routes and adaptation ticks once per completed verification window.
Internal bounded request geometry is charged and authenticated; the current
CLI still selects public fixtures, and general sampling/input remain open.

Release build, strict Clippy and179 offline shaders pass.14 selected core tests
and a bounded-geometry lifecycle test pass. Ordinary regression:1,354 pass,
one ignored. No live model/GPU exchange/recovery/speed run. Old full-core
failures, native CPU-share/queue/packing audits, coupled1..8 target/MTP/state
support and the remaining completion checklist remain required. Default core
capacity is still four; do not claim full rewrite or runtime acceptance.

API specification/installed services/Vashti are unchanged; required Vashti
changes: none. No install, release, reset or rebind. Canonical checkout and
inherited bytecode preserved. Goal active and incomplete; speed gate closed.


### Active Kuno engine handoff extension — eight-row native consumers (2026-10-14)

Source `ac6169a`, isolated `codex/strata-rust-port` checkout. Read the completion
guide's **Connected eight-row consumers and native Q6 head** section. It
supersedes the default-four structural limitation, not the missing GPU or full
rewrite acceptance.

Core now couples capacity8 to packets/logits/quotes/checkpoints/MTP, real1..8
dense Q8 consumers and GDN streaming. Original Q6 native iqs integer-chain/XOR
head is connected and selected before prompt input. Core Q2 MTP repair handles
all accepted prefixes of eight-row windows and uses full draft vocabulary.
Global expert/cache/controller path stays selected. No decoded weight duplicate,
legacy fabricated receipts or intermediate speed selection.

Release build and strict Clippy pass;195 offline shaders pass. Host tests cover
independent canonical Q6 lane mapping, all RHS scratch bounds,1..8 repair prefixes,
compact packet/inactive rows and target rejection/EOS. Ordinary regression:
1,358 pass, one ignored. No live GPU/model/physical recovery/speed test. Prior
full-core failures and full eight-row model/state acceptance remain open.

Next port/audit CPU expert share, transfer/shared execution and queues, native
packing, GDN/QSA/KV/PLE/prefill and MTP graph/state/sampling/general-input paths.
Do not claim full rewrite or open the speed gate yet. API/Vashti consumer changes:
none. Installed services/canonical checkout unchanged; inherited bytecode
preserved. Goal active and incomplete.


### Active Kuno engine handoff extension — native CPU/GPU split (2026-10-14)

Source `e43e271`, isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected native CPU expert share and shared-work overlap** for actual
source/math/ownership/join contracts and explicitly unresolved substitutions.

Real router groups now split resident-first GPU, last141/256 missing GPU staging,
and remaining HOST-backed CPU jobs. Rust native Q8/integer arithmetic covers all
six routed formats; eight RHS share code loads and persistent row workers.
Authentic HOST readonly maps stay alive through CPU joins. GPU expert/shared work
is submitted while CPU runs; CPU merge/reduction precedes shared finish/GR combine.
Actual source mutation/close requires unique Arc ownership. Paired exchange now
includes explicit Vulkan GPU-to-HOST visibility. Added1 MiB private MAIN and16 MiB
CPU host reservation; no disk source reads, weight duplicates or persisted content.

Release build, strict engine/library Clippy,196 offline shaders, native CPU math /
SIMD / sparse-row tests and exhaustive width/split rank accounting pass. Ordinary
regression:1,359 pass/one ignored. Library:141 pass/two ignored. No GPU model,
actual HOST mapping/exchange/overlap/recovery or performance acceptance. Existing
full-core failures remain open. Per-layer readback/upload, fixed share and Rayon
unPinned topology differ from native mapped sink/physical-core/stream scheduling;
remaining fidelity/math/prefill/KV/MTP/sampling/input/recording checklist required.

Keep assembled core selected; speed gate closed. API/Vashti changes required:
none. Installed services/canonical checkout unchanged; inherited bytecode
preserved. Goal active and incomplete.


### Active Kuno engine handoff extension — connected FP16 QSA (2026-10-14)

Source `d5d4e49`, isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected FP16 QSA family** for ownership, actual math/state dispatches,
allocation changes and outstanding substitutions. This supersedes the core F32
KV limitation; it does not establish complete engine or device acceptance.

Core width1..8 now couples actual packed FP16 committed/current KV, native fused
norm/RoPE, FP16-rounded raw index keys, retained completed pooled blocks, native
Intel index scores,64-cell attention/merge/gate and rejected-tail clearing.
Retained POOLED identity joins checkpoint/physical-root proofs. Actual private
stage/scratch allocations and quotes change together. Existing assembled global
sources, CPU/GPU share and controllers remain selected; no partial speed winner.

Release build, strict Clippy,206 offline shaders,26 family and two host geometry/
chunk-oracle tests pass. Ordinary regression:1,361 pass/one ignored. Wider core
QSA filter:58 pass/two fail (language/cache gate and whole-model quote); older
full-core failures remain open. Token-major/raw-history/dead-tail/multi-query/
queue substitutions and numerical/state/recovery acceptance are still required.
No GPU/model or speed execution. Goal active and full rewrite incomplete; no live
speed tests until the connected completeness and correctness gates pass.

Next port GDN's actual multirow conv/L2, four-group FMA recurrence/norm and
accepted-prefix replay together (reference `verify_kernels.dp.cpp`), then continue
the whole checklist. API/installed services/Vashti changes required: none.
Canonical checkout and inherited bytecode preserved; no install/release/reset.


### Active Kuno engine handoff extension — coupled native GDN (2026-10-14)

Source `351e29b`, same isolated `codex/strata-rust-port` checkout. Read completion
guide **Connected native GDN producers, recurrence and repair**; it supersedes
the previous extension's description of the old GDN streamer arithmetic.

Core width1..8 couples RAW conv/native L2, controls,512-lane four-group FMA state
recurrence, query scaling after readout, fused output norm/gate and actual output
projection. Accepted-prefix state-only replay now uses the same arithmetic and
retained native inputs. Existing state/MAIN/INFO reservations stay connected;
actual device workgroup capacity is checked before launch. Global-source and
native QSA families remain selected together.

Release build, strict Clippy,210 offline shaders,17 family and one independent
host math/prefix test pass. Ordinary regression:1,362 pass/one ignored. No GPU,
model/state/recovery or speed test. Head-major/separate candidate state, shared
plane scheduling, BF16 AB projection, optional producer fusion, prompt/capture
and numerical acceptance remain open. Prior full-core failures remain required.

Full rewrite remains incomplete; goal active, speed gate closed. Continue the
whole completion checklist (AB/packing, prompt/PLE/MTP initialization, complete
recording/sampling/arbitrary input), then reconcile and perform final acceptance.
API/installed services/Vashti changes required: none. Canonical checkout and
inherited bytecode preserved; no install/release/reset/rebind.


### Active Kuno engine handoff extension — native GDN AB/window preload (2026-10-14)

Source `3ae92cf`, same isolated `codex/strata-rust-port` checkout. Completion
guide **Native GDN AB and next-token preload** supersedes the generic AB and
single-plane limitations in the previous GDN receipt.

Core now borrows real original resident BF16 AB matrices and runs pinned
32-lane/eight-value FMA window projection plus gating. Real half-result stamps
join before control status acceptance. No generic alpha/beta fallback. QKV/Z,
controls, native recurrence/norm and prefix replay stay connected. Recurrence
and replay now preload next-token Q/K in alternate shared planes. No new MAIN
owner or model duplicate; unused old scalar scratch stays visibly reserved.

Release build, strict Clippy,211 offline shaders and two independent host dot/
state-prefix tests pass. Ordinary regression:1,363 pass/one ignored. No GPU/model
or speed execution. Earlier full-core failures and complete numerical/state/
queue/prompt/packing/MTP/sampling/input acceptance remain required.

Next actual dependency is grouped MTP front/prompt initialization; read the
completion receipt's pinned call sequence and teacher alignment. Current scalar
teacher/repair loops are not native grouped graphs. Keep the whole assembled
core selected, goal active and speed gate closed. API/Vashti changes: none.
Canonical checkout/services and inherited bytecode preserved; no installation.


### Active Kuno engine handoff extension — grouped MTP front (2026-10-14)

Source `147cd9a`, same isolated `codex/strata-rust-port`. Read completion guide
**Connected native MAX8 MTP front** for real owners, loading formats, caller/state
alignment and explicit remaining substitutions.

Core now connects grouped embedding/wide-carry norm, native Q8 input branches,
BF16 fused GR, shared K/V input planes and key norm/RoPE/append. Prompt teaching
and accepted-prefix repair use real grouped windows. Draft/followup count1 uses
the same front from actual GPU Token/Carry owners, with no duplicate KV append.
Original BF16 GR and Q6 draft head are preserved. New20-owner/4,685,824-byte
front,4096 core context,25,296,896-byte private plan and erasure/close accounting
are coupled; ordinary2048 path remains separate.

Release build, strict Clippy,216 offline shaders, two geometry/canonical-embedding
and nine repair tests pass. Ordinary regression:1,365 pass/one ignored. No live
GPU/model/state/recovery or speed run; earlier full-core failures remain open.
F32 cache backing, legacy MTP rest, prompt MMQ/group fences, recorded graphs,
carry handoff and numerical/device acceptance are still required. Next port MTP
rest and packed cache together, then the remaining whole completion checklist.

Goal active; full rewrite incomplete and speed gate closed. API/Vashti changes
required: none. Canonical/services and inherited bytecode preserved; no release,
install, reset or rebind.

### Active Kuno engine handoff — connected packed MTP rest (2026-10-14)

Source `294cb93`, isolated `codex/strata-rust-port`. Read completion guide
**Connected MTP packed cache and attention/GR rest**. Core append/attention/
rewind/guards now agree on native page4/head-major FP16 K/V; count1 attention
and original BF16 FF/head GR consume the real front and publish actual carry.
Private plan18,644,992 bytes includes all22 front/rest owners and cleanup.

Release build, strict core Clippy,221 offline shaders, three layout/front and
nine repair/ownership tests pass. Ordinary regression:1,366 pass/one ignored.
No GPU/model execution, speed testing, deployment, API or Vashti changes.
Whole-port completion and physical/numerical acceptance are not claimed.

Next port MTP scaled-activation/native integer Q2 experts with router/shared
consumers as one coherent block; then complete remaining prompt/PLE, graph/
queues and arbitrary-input/sampling checklist. Earlier full-core failures remain
open. Do not select partial speed winners or reopen live speed benchmarking.
Goal active; full rewrite incomplete. Required Vashti changes: none.

### Active Kuno engine handoff — connected MTP native MoE (2026-10-14)

Source `a4aa0dc`, isolated `codex/strata-rust-port`. Read completion guide
**Connected native MTP routed/shared MoE** for actual original source roles,
activation/data layout substitutions, validation and remaining queue gaps.

Core now connects BF16 router, native top10, F32-scaled S2 input and SwiGLU
quantization, paired integer gate/up/down, fused shared Q8, BF16 scalar and
native merge to the real GR/head/carry graph. All original512 Q2 experts stay
resident; no per-request weight upload or host route read. Native dense input
quantization now follows half-away/finite-half scale/sum contract and supports
the legal640-column shared projection. Private plan18,776,064 bytes includes
all25 front/rest owners, guard/erasure/close accounting.

Final release build, strict Clippy and229 offline shaders pass. Independent
GGUF/S2 host math,640-column batch bounds and nine repair/owner tests pass.
Ordinary pre-fusion regression:1,367 pass/one ignored; subsequent core-only
shader/binding changes pass final core build/shader checks. No GPU/model/state/
recovery or speed acceptance, root/service/API/UI change or installation.

Next implement actual Intel GEMM/streamed prompt prefill plus PLE/history,
then finish graph/queues and arbitrary input/sampling. Pinned SYCL CMake builds
GEMM/prefill with fused stubs and no CUDA/HIP MMQ; port its real Intel consumers,
not the inactive MMQ stub. MTP side-queue/capture scheduling, dense-format audit,
earlier full-core failures and whole-model acceptance remain open. Full rewrite
incomplete; goal active; speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — native PLE/source history (2026-10-14)

Source `1018a09`, isolated `codex/strata-rust-port`. Read completion guide
**Connected native PLE and target BF16/router math** for exact call paths,
allocation/ownership, I/O/history substitutions and remaining schedule gaps.

Core PLE now connects original BF16 MAX8 projections and native1024/512/256-lane
normalization/gate/conv/history to actual accepted-prefix owners. Target router/
shared-scalar BF16 uses the same native projection. Global router now matches
full512-expert softmax/winning-lane normalization. Original private owners,
remap/group/usage and whole-model commit/fault proofs remain authoritative.
Selected-row I/O is deduplicated, parallel and transient, using cloned pinned
descriptors; the additive tensor-reader Rust API has no Vashti consumer change.

Final core release build, strict Clippy and235 offline shaders pass. Three
independent host math/prefix tests,12 ownership/source tests and actual parallel
descriptor test pass. Ordinary pre-router-extension regression:1,370 pass/one
ignored; later core-only changes pass final core/focused checks. No GPU/model,
speed, root/service/API/UI change, deployment or release. Full rewrite incomplete.

Next implement real Intel prompt GEMM/chunk/staging and PLE next-chunk prefetch,
not larger repeated verification windows. Native history mapped snapshots,
stable full graph/queues, dense-format audit, arbitrary input/sampling, earlier
full-core failures and whole-engine physical acceptance remain open. Goal active;
speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — connected prompt GEMM arithmetic (2026-10-14)

Source `e64d770`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt operand/GEMM producers and consumers**. Initial prompt
calls now select rounded BF16/FP16 operands, original GPU weight images and
Rust/Vulkan tile GEMM; normal/pair/bundle/head, GR and AB consumers use it.
Decode remains separate. Existing three64 MiB stage owners are borrowed and
serialized; no new GPU content root or persistent converted artifact.

Final release build, strict core Clippy and247 offline shaders pass. Two host
canonical weight-image/rounded-product tests pass; later span/saturation/dummy
owner changes pass final build/shader checks. No live GPU/model/speed test,
root/service/API/UI change, installation or release.

**Still not complete prompt prefill:** the controller remains a three-row loop.
Large native chunk allocation/budget, streamed FP16 expert GEMM/ring/cache loan,
prompt causal kernels, double-buffered PLE prefetch, device carry/teacher handoff
and whole-prompt scheduling remain required. oneMKL is replaced with ordinary
FMA tile GEMM; numerical/XMX/performance acceptance is unproven. Final stage/
cache/queue lifetime audit and earlier full-core failures remain open. Full
rewrite incomplete; goal active; speed gate closed. Required Vashti changes: none.

### Active Kuno engine handoff — original-source FP16 prompt experts (2026-10-14)

Source `1cf7e50`, isolated `codex/strata-rust-port`. Read completion guide
**Connected original-source prompt expert arithmetic**. Initial prompt calls
now connect logical-ID grouping, real source fetch, original six-format FP16
weight images, FP16 gathered activations, tile GEMM, saturated SwiGLU and parent
rank reduction. Prompt bypasses decode's CPU split; decode stays separate.

The actual global stage owners are three128 MiB buffers. Earlier prompt notes
and scope guards incorrectly expected64 MiB; both scope guards now match the
real owner. No new private weight root, model duplicate or persistent artifact.
Serial role-major staging, separate GU layout, portable FMA tile GEMM and INFO
admission remain explicit schedule/storage audit work for the full chunk owner.

Final release build, strict Clippy and256 offline shaders (including original
six-format physical BDA checks) pass. No live model/GPU/numerical/state/recovery
or speed execution; no root/service/API/UI change, installation or release.

Next connect the native **large chunk owner and expert-major staging ring/cache
loan**, prompt causal kernels, PLE prefetch and device target/MTP teacher handoff
together. Do not grow/rename the current three-row verification loop and call it
native prefill. Complete graph/queues, dense-format audit, arbitrary input/
sampling, earlier full-core failures and whole-engine acceptance remain open.
Full rewrite incomplete; goal active; speed gate closed. No intermediate timing
may steer or roll back the incomplete assembled port. Required Vashti changes:
none. Canonical checkout/services and inherited bytecode preserved.

### Active Kuno engine handoff — native prompt ring and paired products (2026-10-14)

Source `9339904`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt staging ring and paired expert order** for source mapping,
real owners, copy-contract boundaries and remaining substitutions.

Prompt experts now run in logical-ID expert order through actual eight-slot raw
and two-slot dequantization rings. Paired interleaved GU, FP16 SwiGLU and original
DOWN consumers are connected together; earlier completed rank outputs cannot
be overwritten by later GU rows. No new private roots or model duplicate.
One bounded routing-only readback matches native host grouping; no CPU activation
readback or CPU expert contribution. Decode's integer/CPU path stays separate.

Original HOST buffer imports allow COPY_SRC. New internal CubeCL copy APIs
validate registered readonly sources, full same-stream destination owners and
physical cell extents, then insert actual transfer nodes into the same owned
native graph. Do not bypass that API with an external raw queue: registry
detach must join every consumer. Independent copy/compute queues remain open.

Final release build, strict operational core Clippy,256 offline shaders and one
host-only copy metadata contract test pass. Test metadata is not physical GPU
or copy execution proof. No live GPU/model/numerical/state/recovery/speed test,
root action, service/HTTP API/UI change, installation or release. Earlier full
core failures remain open; canonical checkout/services and bytecode preserved.

Next: connect the full **large prompt chunk owner/carving/budget/cache loan** to
layer-major GR, causal prompt GDN/QSA, larger real routed-row owners, PLE prefetch
and device target/MTP teaching. The current caller still uses three-row windows;
do not label that as native prefill. Queue overlap, complete capture/admission,
oneMKL/XMX substitution, dense formats, arbitrary input/sampling and whole-engine
acceptance remain required. Full rewrite incomplete; goal active; speed gate
closed. Keep the assembled port selected; no timing-driven partial rollback.
Required Vashti changes: none.

### Active Kuno engine handoff — prompt GDN and direct FP16 output (2026-10-14)

Source `810f89e`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt GDN causal pipeline and FP16 consumer** for pinned Intel
selectors, real operators/owners and remaining layout differences.

Actual prompt calls now connect contiguous H, causal conv/history/L2, pipelined
CB32 recurrence and separate FP16 output norm to the real output GEMM. Native
FP16 is consumed directly; no normalized F32 GATED copy or repacking. Existing
private owners/status/commit/repair/erasure remain authoritative. Decode stays
separate. Internal prepacked projection is checked against the actual admitted
stage identity/stream and geometry; no HTTP or Vashti API change.

Final release build, strict Clippy and262 offline shaders pass. No GPU/model,
numerical/state/recovery or speed execution, root action, installation or release.
Existing full-core failures remain open. State is still head-major and split
replay inputs are retained; do not credit native placement parity. The enclosing
caller still uses three-row/MAX8 windows, despite dynamic prompt kernel counts.

Next: connect prompt QSA and the full large chunk owner/carving/budget/cache loan,
larger routed-row ownership, PLE prefetch and device target/MTP teaching together.
Complete queues/capture/admission, state/dense-layout audit, arbitrary input/
sampling and whole-engine acceptance remain required. Full rewrite incomplete;
goal active; speed gate closed. No timing-driven partial rollback. Required
Vashti changes: none. Canonical/services and inherited bytecode preserved.

### Active Kuno engine handoff — prompt QSA and joined PLE prefetch (2026-10-14)

Source `242ce1c`, isolated `codex/strata-rust-port`. Read completion guide
**Connected prompt QSA consumers and PLE source prefetch**. Prompt QSA selects
separate RMS/RoPE and directly consumed FP16 gated output. PLE's bounded scoped
producer supplies upcoming source rows, authenticated against real expected
rows; receiver disconnect/join and wiping remain request-local. Extra327,680
host bytes is charged at both component and model boundaries. No new GPU root.

Final release build, strict core Clippy and266 offline shaders pass. No live
GPU/model/source-worker, numerical/state/recovery or speed run; earlier full-core
failures remain open. Goal active, full rewrite incomplete, speed gate closed.
Installed services, canonical checkout, GPU residency and inherited bytecode
were untouched. Required Vashti changes: none.

Next implement the **large prompt owner/carving/budget and full layer-major
chunk schedule**, including larger expert route/member owners, device MTP
teaching and final-only head. Current controller/source prefetch still uses
three-row/MAX8 ownership. Pinned Intel XMX prompt attention is opt-in, not its
default. With this model's large HOST complement, native staging defaults to
eight slots; do not port inactive CUDA branches to satisfy a mistaken checklist.
Queue/capture, storage/state, arbitrary input/sampling and whole-engine
acceptance remain required before speed trials. Sudo refresh in shared shell
succeeded during source work; no protected inference was started.

### Active Kuno engine handoff — chunk-capable routed prompt owners (2026-10-14)

Source `0f89b45`, isolated `codex/strata-rust-port`. Read completion guide
**Connected chunk-capable prompt expert owners**. Prompt calls now use their
own1..256-row router, compressed logical-order host groups, tiled FP16 expert
products and carved rank output reduction. GU `[64,66)` MiB and PART `[66,98)`
MiB borrow the real third128 MiB stage; original transfers stay below64 MiB.
Decode remains MAX8. No additional GPU root;256 KiB host/8 MiB I/O is charged.

Final release build, strict Clippy and268 offline shaders pass. No live GPU,
physical view/copy, numerical/state/recovery or speed acceptance. Earlier full-core
failures remain open. Canonical/services/GPU residency untouched; no API or Vashti
change. Inherited bytecode preserved. Full rewrite incomplete; goal active.

Next connect the full prompt residual/GR/attention/PLE/carry owners and actual
layer-major chunk controller, including final-only head and device MTP teaching.
The current parent still uses three-row/MAX8 packets and receipts. Projection
stage working intervals/alias checks must be changed with the new owner, not
silently overlaid. Cache-loan budget, mapped grouping, queues/capture, layouts,
sampling/general input and whole-engine acceptance remain required. No speed
trials or timing-driven partial rollback. Shared shell now reports sudo expired;
continue source work, request authentication before protected GPU tests.

### Active Kuno engine handoff — carved prompt GR ownership (2026-10-14)

Source `649e923`, isolated `codex/strata-rust-port`. Read completion guide
**Connected carved GR workspace**. Actual prompt GR uses distinct carved
upper-stage0 intermediates with256-row capacity; stage0/1 working images occupy
lower64 MiB. Exact interval/prepacked-view checks preserve ownership. No new
GPU root;4 KiB metadata charged. Parent residual/injection/mixed/state receipts
still use small windows. Full large-chunk ownership/controller remains next.

Release build, final strict Clippy and268 offline shaders pass; see the receipt
for the equivalent post-build Clippy guard cleanup. No GPU/model/physical/state/
recovery/speed acceptance or service/API/UI/install change. Earlier full-core
failures remain open. Canonical checkout and inherited bytecode preserved.
Full rewrite incomplete; goal active; sudo expired. Continue full residual/
attention/PLE/carry/state composition before speed trials. Vashti changes: none.

### Active Kuno engine handoff — carved prompt GDN and real replay source (2026-10-14)

Source `81f1d76`, isolated `codex/strata-rust-port`. Read completion guide
**Connected carved GDN prompt data and replay source**. Actual prompt QKV/Z/H/
readout use stage1 upper-half carved views. Original replay snapshots the real
raw producer, including zero inactive rows. New internal projection writes are
restricted to exact registered views. No extra GPU root;73,728 runtime metadata
bytes charged. Status/control/replay/acceptance and parent packets remain MAX8.

Final release build, strict Clippy and269 offline shaders pass. No live GPU,
copy/model/numerical/state/recovery/speed validation. Full-core failures remain
open; full rewrite incomplete, goal active. Canonical/services/bytecode untouched.
No API, install or Vashti change. Last known sudo expired; source work can continue.

Next QSA: shared pooled index state and bounded32-query attention scratch, with
causal selection and exact cache append/state ownership. Read the receipt's
native pointers; do not enlarge old per-query pool/attention arrays. Full
residual/PLE/carry, prompt acceptance, head/MTP teaching, budgets/queues/layouts
and whole-engine acceptance still precede speed testing.

### Active Kuno engine handoff — shared QSA and native GPU selection (2026-10-14)

Source `6314ea4`, isolated `codex/strata-rust-port`. Read completion guide
**Connected shared prompt QSA and native GPU selection**. Actual prompt QSA now
uses carved projections/Q/K/IQ/candidates, shared completed-block index state
and32-query attention scratch with explicit full-chunk row offsets. Append
precedes scoring/attention. Core selection now uses the pinned GPU weighted radix
path, normalized cell-zero dead key, tail bias and exact `min(visible,2051)` width,
including partial blocks. The model no longer CPU-sorts/uploads QSA selections.
The old ordinary selector is preserved, not an oracle for the native width.

Release build, strict Clippy and281 offline shader variants pass; no physical
GPU/model/numerical/state/recovery/speed run. Real larger POOLED owners,16KiB
scope metadata and417,792 extra INFO bytes are charged;1024 lanes/32,912 shared
bytes are required. Existing full-core failures remain open. Full rewrite is
incomplete; goal active; speed gate closed. No API/Vashti changes are required.
Canonical/services/GPU residency/inherited bytecode are untouched. Last sudo
check reported a password required; no root workaround or protected run.

Next: full residual/mixed/block/carry/PLE owners and the real large layer-major
prompt controller, final-only head, device MTP teaching and prompt state acceptance.
The current enclosing parent/receipts still use three-row/MAX8 windows. Complete
native budgets/cache loans, queue/capture and layout/substitution audits, general
input/sampling and whole-engine acceptance before speed trials. Do not revert
coupled components because of incidental timing or claim the rewrite complete.

### Active Kuno engine handoff — full-chunk GDN and native state layout (2026-10-14)

Source `9a7a23b`, isolated `codex/strata-rust-port`. Read completion guide
**Connected full-chunk GDN math and native recurrent layout**. Actual prompt
math now uses256-capable convolution/L2/recurrence/norm entries with direct model
faults, independent of per-element status/replay buffers. The enclosing MAX8
caller retains explicit split/replay/closing compatibility. AB raw projections
use a real carved region and a handle/count entry. New controls/status capacity
fits the same GDN root. Native recurrent row/value-head layout is selected
coherently in prompt, decode and state-only repair; no transpose or new owner.

Release build, strict Clippy and286 offline shader variants pass. No live
GPU/model/numerical/state/recovery/speed acceptance; earlier full-core failures
remain open. Runtime quotes include147,456 extra INFO bytes. No API or Vashti
change, installation or release. Canonical/services/GPU residency/bytecode were
preserved. Goal active, full rewrite incomplete, speed gate closed; last sudo
check required a password, and no protected operation was attempted.

Next connect the full residual/mixed/block/carry/PLE prompt controller and its
actual state acceptance. Finish compact/direct-fault QSA status ownership rather
than scaling its old50MiB-per256-query attention status. Read the receipt's exact
existing handle entry points. Parent still uses three-row/MAX8 windows; final-only
head, device MTP teaching, native budgets/cache loans, queues/capture, remaining
layout audits, general input/sampling and whole-engine acceptance remain required.

### Active Kuno engine handoff — full prompt QSA ownership (2026-10-14)

Source `101f4e7`, isolated `codex/strata-rust-port`. Read completion guide
**Connected full prompt QSA ownership**. Actual prompt QSA now calls one connected
256-capable family using carved compact flags/scores/selections and direct model
faults. It includes positions through packed gated output and32-query attention
scratch; no per-value attention flags or per-query pooled copies. Legacy MAX8
status adapters remain outside it for the existing parent; later old math and
selector events are skipped. Scope metadata32KiB and conservative QSA INFO835,584
bytes are charged. Twenty QSA views fit59,490,304 bytes of the existing upper64MiB.

Release build, strict Clippy and296 offline shader variants pass. No live GPU,
model/numerical/state/recovery/speed acceptance; full-core failures remain open.
Full rewrite incomplete, goal active, speed gate closed. No API/Vashti, service,
install or release change. Canonical/GPU residency/bytecode preserved; last sudo
check expired, no protected run or workaround.

Next implement the real large controller with persistent R/mixed/block/carry,
handle-based routed/shared/PLE calls, charged larger joined PLE source, prompt
state acceptance, final-only head and device MTP teaching as a coupled block.
GR/AB/GDN/QSA/expert math already has256-row handle entry points; the parent
still uses three-row/MAX8 packets. Keep verifier receipts separate and actual
binding/root checks intact. Complete native budgets/loans, queues/capture,
remaining layout/substitution audits, general input/sampling and whole-engine
acceptance before throughput comparisons. Required Vashti changes: none.

### Active Kuno engine handoff — large prompt controller and full device MTP (2026-10-14)

Source `f3196fc`, isolated `codex/strata-rust-port`. Read completion guide
**Connected large prompt controller and device MTP teaching**, including its
validation and exact next-source pointers. The core parent now runs real256-row
prompt chunks across embedding/GR/GDN/QSA/routed/shared/PLE, with persistent
carved R/mixed/blocks, all-family state preflight/commit, a last-chunk-only head
and full-chunk MTP teaching directly from target VRAM. The interim eight-row
teacher and per-row host carries are gone. Verifier MAX_BATCH stays8.

Persistent GR/outer views use57,918,720 bytes of the existing stage0 upper64MiB;
PLE phase views use57,695,232 bytes of stage1 upper64MiB. No new GPU root/MAIN
allocation. PLE source host charge is12MiB, controller host/runtime add8MiB each.
Exact PLE history acceptance maps only its last nine normalized rows and NEXT;
legacy decode-policy/checkpoint workspaces receive a sealed real prompt boundary.
Prompt GR writes consume activated injection once, correcting its former double
activation at explicit PLE/final boundaries.

Release build, strict Clippy,308 offline shader variants and six core PLE history
tests pass. The complete core unit suite **does not pass**:1,149 passed,228 failed,
1 ignored. The receipt distinguishes the historical200-failure baseline from
this newer suite; do not assume all failures are obsolete or from this commit.
No live GPU/model/numerical/state/recovery/speed run or API/Vashti/service/install/
release change. Canonical checkout and bytecode preserved; goal active, full
rewrite incomplete, speed gate closed. Last Demons shell `sudo -n -v` required
a password; no protected operation or privilege workaround followed.

Next complete native queue/recording/copy dependencies and the real GEMM/data-flow
substitution, then remaining placement/default sizing, arbitrary input/sampling,
verifier repair and unit-failure audits. Exact useful pointers are in the receipt:
Rust Vulkan backend already recognizes FP16/BF16 cooperative matrices, while the
currently used WGSL compiler rejects CoopMma; its optional SPIR-V path is worth
investigating. Pinned Intel source defaults BF16X2 off and uses owned prompt
buffers at context<=32768 (default2048 rows), so audit those actual branches
before declaring remainder GEMMs or cache lending unconditional missing work.
Finish the full source review and protected whole-engine acceptance before any
speed comparisons. Required Vashti changes: none.

Supplemental read-only inventory is recorded at the end of that receipt. The
B70 driver exposes cooperative-matrix and BF16 extensions and FP16/F32 matrix
shapes; no matrix computation was run. The existing device adopts one family0
queue; families1/2 exist but were not requested. These are concrete starting
points for the connected compiler/device/queue block, not performance evidence.


### Active Kuno engine handoff — connected cooperative matrix backend (2026-10-14)

Source `d326f23`, isolated `codex/strata-rust-port`. Read completion guide
**Connected Rust cooperative prompt matrices**, including the required queue
continuation and validation details. Actual dense prompt/MTP and grouped expert
GU/DOWN products now use Rust FP16/BF16 cooperative8x16x16/F32 kernels, coupled to
the existing rounded images, maps, faults, activation and downstream consumers.
Target/MTP device creation enables the actual matrix features and accurately
adopts the full extension list without enlarging admitted limits. Both Wgpu and
recorded native pipelines use the same explicit16-lane geometry; existing
physical integer shaders retain32. Native SPIR-V registration includes real
INFO/descriptors and cached operator identities. External model sources retain
their audited read-only WGSL lowering.

One shared HAL canonicalizer fixes redundant implicit Workgroup strides while
preserving explicit Block and host byte ABIs, rejecting aliases it cannot safely
rewrite. Release build, strict core/feature-off Clippy, five host ABI tests,
308 existing shader variants (122 physical-pointer checks) and32 cooperative
variants (all eight independent optional-layout masks) pass. Use the modern
validation-only Khronos tool revision/path in the receipt; the distro2025 tool
cannot validate BF16. No live model/GPU/numerical/state/recovery/throughput run;
the earlier full-core228 failures remain unaudited and the complete suite is not
green. No new global tensor root, API/Vashti change, install, service restart or
release. Canonical checkout and inherited bytecode preserved.

**Next connect the native copy/compute schedule as one block.** The device still
requests only family0 queue0, and both Wgpu roots and imported host copy sources
are EXCLUSIVE. Merely requesting a transfer queue would race or violate ownership.
The receipt names exact allocation/insertion/recording/flush sites and dependency,
replay and partial-submission lifetime requirements. Keep actual created queue
metadata, buffer sharing/ownership, semaphore edges and final dual-queue completion
joined. Do not claim real ring-slot overlap from conservative full-root hazards.
Then complete native default2048/owned-versus-loaned sizing and placement,
arbitrary input/supported sampling, verifier repair and the full source/failure
audits before protected whole-engine acceptance and only then speed comparisons.

Full rewrite incomplete, goal active, speed gate closed. Last shared-shell
`sudo -n -v` still required a password; no protected operation or workaround was
attempted. There is useful source work before root is needed. Required Vashti
changes: none.


### Active Kuno engine handoff — real compute/copy queues and joined replay (2026-10-14)

Source `ac05da8`, isolated `codex/strata-rust-port`, following matrix `d326f23`.
Read completion guide **Connected native compute/copy queue ownership**. The
Intel device now actually creates/adopts a second transfer-family queue before
allocation. Both Wgpu roots and actual HOST copy sources use concurrent sharing
for the two admitted families. Device-local metadata describes the queue really
created; ordinary feature-off construction retains one queue.

Actual native recordings now split shaders/copies/fills into queue-owned command
segments with RAW/WAR/WAW timeline edges, an upload-release header and a final
fence joining both queues. Cached replay uses fresh timeline epochs. Recording
owns both pools, semaphores and retained roots; partial submissions become
in-flight before the first submit, and unknown work cannot be reset/reused or
freed based only on an unsubmitted final fence. Six host planner/epoch tests,
release build, strict core/feature-off Clippy,308 existing shader variants
(122 physical validations) and32 matrix variants pass. Added host queue metadata
is bounded/charged (80MiB core lifetime reservation,64MiB retained queue-vector
cap). Opaque driver allocation and real hardware admission remain unproved.
No live GPU/model/numerical/state/recovery/speed test. Complete unit suite not
rerun; its earlier228-failure baseline remains unaudited. No API/Vashti, installed
service, model-residency, installation or release change. Canonical checkout and
inherited bytecode preserved.

**Next connect exact ring-slot descriptors and byte-range hazards together.**
Current hazards conservatively key whole buffers, so separate slots within the
same64MiB raw buffer still serialize. This is not complete native overlap or a
performance result. The receipt names the actual group-ahead staging loop,
Dequant whole-root binding, `p.r0` offset and shader's fixed raw length/extent
checks. Do not merely tell the scheduler a narrower range: narrow the real
binding and update shader/source bounds coherently, including unequal GU/DOWN
blob sizes. Add independent partial-overlap/disjoint-slot/reuse oracle coverage
and charge bounded interval metadata. Keep BDA/indirect dependencies, replay
signatures, upload acquisition and final dual-queue join intact.

Then finish whole graph/branch capture fidelity, native default2048/owned/loaned
sizing and placement, arbitrary input/sampling, verifier repair and all source/
unit completeness audits. Protected full-engine acceptance precedes speed tests.
Goal active, full rewrite incomplete, speed gate closed. Root was not needed for
this source/offline block; last prior shared-shell check had required a password.
Required Vashti changes: none.


### Active Kuno engine handoff — exact raw-slot ranges and early release (2026-10-14)

Source `3ca345c`, isolated `codex/strata-rust-port`, following queue `ac05da8` and
matrix `d326f23`. Read completion guide **Connected exact raw-slot ranges and
dequant release**. Actual graph descriptors/copy spans now feed a bounded
byte-interval RAW/WAR/WAW planner, preserving untouched intervals and allowing
disjoint slots within one buffer to overlap. Bounds include retained Vec
capacities after contraction; host queue reservation is now128MiB, cached queue
vectors still capped64MiB. Hardware/driver memory admission remains unproved.

Prompt dequant binds the actual raw slot, uses local zero offset and exact length/
role extent checks, retaining original packed formulas and paired products.
The old generic binder also declared every scratch stage as a hidden BDA read:
the prompt shader now has no stage-address binding, and its dedicated binder
retains only genuine original-model GPU/HOST roots. Decode still uses its real
stage pointer table/dependencies. This is a real interface change, not a
scheduler-only narrowing. The internal cache key carries `prompt-raw-slot-v2`.

Pinned Intel `prefill.cpp:3975` releases the slot after both dequants and before
GEMMs. The actual Down-image metadata now forces that same native recording
boundary; fresh/cached pipelines preserve it, and the copy queue can acquire its
timeline signal while matrix products continue from the separate dequantized
owner. Native release keys wrap/preserve original quantization identity, including
external masks; false leaves ordinary cache keys unchanged. Upload acquisition,
replay epochs, both pools/resources and final dual-queue completion remain joined.
No timing switch or speed-driven fallback was introduced.

Release build, strict core/feature-off Clippy,11 queue tests, three raw-view/cache
tests,308 shader variants (122 physical validations) and32 matrix variants pass.
No live GPU/model/numerical/state/recovery/speed run; full unit suite not rerun and
the prior228-failure baseline remains unaudited. No global tensor root, API/Vashti,
installed service/model residency, installation or release change. Canonical and
bytecode preserved. Goal active, rewrite incomplete, speed gate closed. Root was
not needed for source/offline work; last earlier credential check had expired.

**Next port native prompt allocation/placement and configured size policy as a
connected block.** Capacity256/fixed three128MiB stages do not establish native
owned/default2048 grouping or cache loans. Do not emulate2048 by eight separately
routed256-row prompts. Audit the true Intel owned-at-context<=32768 / otherwise
loaned branch, dependent GR/GDN/QSA/MoE/PLE/MTP owners, INFO/host/runtime charges
and expert-cache placement together. The receipt explains native STAGE8/DQ2 and
conditional stream-all/big-ring policy: do not assume a large ring is always
selected (nonresident share>=10% defaults to eight even for large T).

Then finish complete branch/graph capture, arbitrary input/sampling, verifier/
repair, source and unit-failure audits, followed by protected whole-engine
completion/numerical/state/recovery acceptance and only then throughput. The
passing host schedule oracle does not prove actual GPU overlap or performance.
Required Vashti changes: none.


### Active Kuno engine handoff — owned prompt allocation foundation (2026-10-14)

Source `788e793`, isolated `codex/strata-rust-port`, following range `3ca345c`.
Read completion guide **Connected native-owned prompt allocation foundation**.
**Actual prompt math/controller/grouping is still capped256. This is not
2048-row completion, cache lending or permission to benchmark.**

The actual constructor now prices/registers three separate512MiB private prompt
owners (MAIN delta1,610,612,736 bytes); decode keeps its three128MiB fetch stages.
`core_prompt_stages` supplies the real prompt Scope, and the source binder attests
the actual same-stream whole raw owner. Copies/slot descriptors stay in its
lower64MiB; ordinary decode tables/dependencies are unchanged. All new owners are
in real request/MAIN inventory, normal close and unknown-in-flight retention.
These power-of-two Rust envelopes are conservative, not measured SYCL/VRAM parity.

Native allocator/admission supports512MiB owners with explicit device buffer/
binding limits; ordinary feature-off retains128MiB/pools. Cleanup now inspects
complete roots through32MiB readbacks, wiping/dropping/sanitizing each lease
before the next and publishing root credit only after its entire tail/padding
passes. No enlarged host readback allowance, untracked owner or ordinary-I/O
fallback. Policy recognizes native default2048 and owned-at-context<=32768;
CacheLoan is explicitly unimplemented and cannot silently become an owned
allocation. No configurable chunk/loan implementation is claimed.

Release build, strict core/feature-off Clippy,18 arena tests in each mode,
policy/inventory test, actual256/2048 persistent/phase carving test, three slot/
cache tests,308 shader variants (122 physical) and32 matrix variants pass.
No live GPU/model/allocation/erasure/numerical/state/recovery/speed acceptance;
full core suite not rerun, prior228-failure baseline still unaudited. No API/Vashti,
installed service/residency, install or release change. Canonical and bytecode
preserved. Goal active, rewrite incomplete, speed gate closed.

**Next propagate true2048 rows through all actual consumers as one block.** The
receipt lists exact map/status collisions and suggested owned-view separation:
LOGICAL4100 overlaps MEMBERS1540+2048*10; old MoE roots remain decode-sized.
Carve real prompt-specific raw/grouped maps/status/GU/PART views from the priced
owners and connect them to router/grouping/dequant/matrix/reducer/closing proof.
GU needs16MiB and PART256MiB padded at2048. Keep verifier MAX_BATCH8 separate and
leave workgroup256/CB32/attention32 tiling alone. Audit full working images,
PLE actor rows/charges/history and MTP four-residual rows (potential matrix count
8192) as well as GR/GDN/QSA scope/count/state acceptance and controller chunks.
`Storage::at_capacity(2048)` proves only its listed view bounds, not all consumers.

After full2048 flow, implement real cache loans/restoration/exclusion and
conditional ring/stream-all sizing with actual cache placement; then input/
sampling, branch/recording, verifier repair, source and unit-failure audits before
protected whole-engine acceptance and speed comparisons. No privileged work was
needed; the last earlier sudo check had expired. Required Vashti changes: none.


### Active Kuno engine handoff — connected native2048 owned consumers (2026-10-14)

Source `d75549f`, isolated `codex/strata-rust-port`, following `788e793`.
Read completion guide **Connected native2048 owned prompt consumers** for exact
views, matrix/dispatch geometries, accounting, receipts and remaining gates.
This supersedes the previous256-row consumer limit. **This is source/offline
acceptance; the complete rewrite remains unfinished and speed benchmarks are
still prohibited until its completeness gate passes.**

The actual full-chunk controller, request authority, GR/GDN/QSA/PLE/MTP scopes
now use2048 rows. MoE has real, distinct raw packet/grouped map/probability/
role-status/GU/PART descriptors carved above the existing64MiB raw ring in the
third tracked512MiB prompt owner. The router writes the raw view, readback
preserves its original offsets, CPU grouping writes the separate map, and
expert gather/dequant/cooperative matrices/activation/reducer bind those same
views. The reducer no longer reuses old decode map/status/probability roots.
Logical IDs begin at32768, outside all20480 physical/member routes; role status
planes are separated by20480. Per-layer proof views are cleared, while actual
48-layer count/epoch stamps still join the completed whole-model sticky fault.
Prompt image ABI is `prompt-owned-routes-v3`; original format/release identity
is preserved. Ordinary decode/MAX8 roots and verifier capacity stay unchanged.

Both projection and dense matrix admission cover MTP's8192 FC-hidden rows at
2048 target rows. Full residual/teacher normalization use actual80MiB carved
views; packed working images remain bounded64MiB. True linear copy/embedding/
residual-write/pack/dequant kernels flatten oversized grids using actual
`num_workgroups`; GR/head/attention axes remain independent. QSA attention stays
32-row tiled, its row offsets and whole-prompt guards now cover2048, and INFO
includes64 tile pairs. PLE's exact last-nine-row history transition accepts2048;
prefetch metadata is heap-backed/wiping, unique original reads remain deduplicated,
and reconstruction scatters directly by destination. Group host quote1MiB,
PLE source64MiB and separate outer64MiB host/runtime copies are conservative
actual-lifetime bounds; no extra arena owner beyond the priced prompt roots.

Final core release build, strict core/feature-off Clippy,308 shader variants
(122 physical validations plus32 cooperative variants), all12 prompt tests,
source-plan coverage1..=2048 and the scalar-vs-tail PLE history test through2048
pass. Exact log names are in the completion receipt. No live GPU/model/
numerical/state/recovery/speed acceptance; full core suite not rerun and prior
228-failure baseline remains unaudited. No installed service/residency, public
API/Vashti, installation or release changes. Canonical and inherited bytecode
preserved; no privileged action was needed. Goal stays active.

**Next: native cache loans/restoration/exclusion and conditional ring sizing
against real cache placement/owner bytes.** Owned envelopes remain conservative
fixed512MiB; context>32768 lending is still explicitly unimplemented, and current
request/QSA/MTP context is bounded4096. Configurable prompt/long-context fidelity
is not established. Then finish arbitrary input/sampling, complete branch/
recording and verifier repair, source/unit-failure audits and protected whole-
engine completion/numerical/state/recovery acceptance before throughput.
Do not revert coupled port work because of incidental timing. Required Vashti
changes: none.


### Active Kuno engine handoff — contiguous cache / bounded source windows (2026-10-14)

Source `16e99d8`, following `d75549f`, isolated `codex/strata-rust-port`.
Read completion guide **Cache contiguity and bounded public initialization**.
The previous goal turn made source/verification progress. **Cache lending is
still incomplete; this is a real model-loader prerequisite, not a loan receipt
or permission to run speed benchmarks.**

Actual global plan/GPU allocation now packs ranked experts into roots up to
512MiB while retaining16MiB original-model host uploads. GPU contiguity and
host staging no longer share one constant. `Cache::load` allocates an empty
persistent root, fills every byte/padding using `load_chunk_range` and exact
public upload views, and drains each window before another. Source tables and
consumers publish only after complete full-root attestation. Exact root requests
still feed persistent admission; actual `Cell` addresses and original ranked
prefix/HOST complement remain authoritative. The setup quote stays two16MiB
windows plus the1MiB HOST reader, and the internal receipt distinguishes root
limit from upload window. A smaller final root is not fictitious512MiB authority.

Core release build, strict core/feature-off Clippy and all five global-plan tests
pass. They retain the independently recorded11570-pair prefix, empty/partial/
full complete source coverage and exact HOST quote. New tests reconstruct
original roles/padding with independent byte streams across awkward byte windows
and inspect pinned-model root/window boundaries without a512MiB host allocation.
Exact logs are in the receipt. No shader arithmetic changed; prior308 variants
were not rerun. No live GPU/allocation/model/numerical/state/recovery/speed run,
full unit suite, services/residency, API/Vashti, install or release change.
Prior228-failure baseline remains unaudited. Canonical/bytecode preserved; no root
needed. Goal active, complete rewrite and benchmark gate remain unfinished.

**Next connect actual cache-backed private ownership, exclusion and restoration.**
Pinned `sycl/src/program/generate.cpp:11524-11624` marks lent tail pairs unavailable,
initializes prompt scratch there and restores their actual slots before decode.
Kuno's private arena currently attests exclusive power-of-two private allocations;
public cache roots are actual alignment-sized owners, possibly smaller than512MiB.
A real borrowed registration must attest/charge the complete physical root,
include full erasure/inspection and retain it on unknown completion. No cast of
public handles into ordinary private ownership, fake larger descriptor or mere
host bitmap. Leased resident pairs also need a real original-weight staging
source (HOST complement only contains initially nonresident pairs), patched
physical source tables and retained BDA dependencies excluding borrowed cells.
Restore current placement/remaps after the joined prompt close, accounting for
adaptive exchanges before publishing residency. Then finish conditional ring/
stream-all policy and remaining arbitrary input/sampling, branch/recording,
verifier-repair, source/unit-failure and protected whole-engine acceptance gates.
Required Vashti changes: none.


### Active Kuno engine handoff — exact prompt parent spans (2026-10-14)

Source `c0e9efc`, following `16e99d8`, isolated `codex/strata-rust-port`.
Read completion guide **Exact physical-parent prompt view geometry**.
Previous goal turn made model-loader/verification progress; this turn removes
another actual consumer blocker for lending. **No borrowed private registration,
loan exclusion/staging/restore transaction or complete rewrite is claimed.**

Scope, Storage, MoE views, raw-slot slicing and source attachment now derive
slices from each whole parent's actual size, up to512MiB. This supports real
alignment-sized cache backing rather than subtracting fictitious512MiB tails.
The actual family carve enforces its own minimum. Whole-root structure rejects
slices/misalignment/oversize; Storage rejects duplicate identities and foreign
streams. Source attachment still proves complete backend buffer offset/size,
checks real MoE fit, and requires copy destinations to match that attached root.
Working images stay64MiB, raw ring stays below64MiB, descriptors/math/default
owned allocation/pricing are unchanged. Structural validation is not a borrowed
ownership capability; Scope still receives private-arena BufferIds, whose
exclusive power-of-two allocation and cleanup policy are unchanged.

All15 focused prompt tests and core release build pass. The new cases accept
independent non-power-of-two parents, derive/accept exact family minima from
actual views, reject a256-byte deficit in each parent and reject aliases,
slices, foreign streams and oversize; raw-slot tails match real owning ends.
Final strict-check receipts are recorded in the completion guide. No live GPU/
model/allocation/erasure/state/recovery/speed, full unit suite, services/residency,
public API/Vashti, install or release change. Historical228 unit failures remain
unaudited. Canonical and bytecode preserved; no root needed. Goal stays active.

**Next implement genuine private borrowed-root ownership and the connected
source transaction, not more pretend loan geometry.** The actual `PrivateArena`
ledger/cleanup still only registers newly allocated exclusive power-of-two
private roots. Cache roots require attested complete physical backing and a
source-owner exclusion capability before private registration, full erasure/
inspection, retention/retirement on unknown completion and exact shared physical
accounting without invented MAIN credit or hidden allocation. Candidate carving
can now validate actual cache handles without forging a larger root; this alone
must never publish a Scope. Leased resident pairs require real original-weight
staging, patched source tables and BDA dependencies excluding scratch cells;
restore current placement/remaps only after joined private close. Continue the
remaining conditional ring/configuration, branch/recording, input/sampling,
verifier repair, source/unit-failure and protected whole-engine gates before
speed testing. Required Vashti changes: none.


### Active Kuno engine handoff — registered cache-root borrowing component (2026-10-14)

Source `09351ea` + private audit-purpose correction `03ea215`, isolated
`codex/strata-rust-port`, after `c0e9efc`. Read completion guide **Registered
private cache roots and erasure-before-restoration transaction**. The previous
goal turn made source/bounds-test progress; this turn adds the actual source-
owned private borrowing component. **Prompt MoE/controller/configuration still
do not select it; complete loan-mode inference is not implemented or validated.**

The private arena now has a separate unsafe registration of actual complete
cache buffers. It checks real backend buffer offsets/sizes/identity and exact
aligned physical spans, retaining all roots before zero initialization. It
allocates no new GPU root and its local ledger is private coverage of already-
precharged persistent banks, not new MAIN credit. Ordinary power-of-two reserve
stays unchanged; registered arenas cannot allocate or reset. Close mandates
complete erasure/32MiB-window inspection and exact root/byte counts, retaining
unknown/unaudited roots on failure. Borrowing mandates strict private audit
without enabling public-fixture permission/replay.

`Sources::with_private_cache_roots` validates actual candidates, joins readers,
blocks all model source consumers, registers the arena, runs the protected
same-worker body and closes/inspects before restoring current original model
weights in16MiB windows. Only both complete phases can reopen sources; failed
registration/erasure/restore retires the executor's actual shared context and
keeps exclusion/quarantine. Body errors/panics still clean/restore before being
returned. The unsafe body must not leak IDs/handles/Scopes/replays. It receives
only arena/IDs, deliberately no permitted model Sources yet.

A wiping inverse physical placement now follows actual adaptive role exchanges.
It validates current incoming/outgoing identities, then commits only after
joined private remap/usage and adaptive-owner publication. Restoration uses
current logical experts, not initial profile bytes. Current/pending maps and
metadata get an explicit extra128KiB HOST quote; nothing is logged/persisted.

Core release, strict core/feature-off Clippy,20 core arena tests, two source gate/
placement tests and five plan/HOST quote/window tests pass. Final ordinary arena
receipt is in the completion guide. Exact logs and failure cases are recorded
there. No shader math changed. No live GPU/model/allocation/erasure/restoration/
state/recovery/speed acceptance or full suite; prior228 failures remain unaudited.
No installed services/residency, API/Vashti, install or release changes. Canonical
and inherited bytecode preserved; no root needed. Goal active, speed gate closed.

**Next connect prompt-only leased sources and the controller as a real block.**
Current borrowing intentionally revokes every prototype/bind/HOST reader, so
it cannot run MoE. Leased resident pairs are absent from the existing HOST
complement: give them a real bounded/quoted original-weight staging source,
using the current placement, and preserve native-copy queue/header/acquisition
semantics. Do not substitute untracked ordinary uploads for native transfer
nodes. Patch the actual source table/flags and BDA dependencies, including
prototype bindings; exclude scratch roots. Redirect leased pointers to real
original-weight backing as well as setting a flag, so hidden/speculated BDA
loads cannot alias private scratch. Restore original source tables/read sets/
prompt-stage attachment only after joined private close and model restoration.

Thread the registered arena's IDs through Scope/full prompt/MTP flow and revise
actual constructor/cache pricing and configured loan selection. `Plan::CacheLoan`
still errors, default construction still has three512MiB owned prompt roots and
context remains4096. Keeping unused owned buffers is not loan/cache price parity.
Then finish conditional ring/stream-all/auto chunk policy, branch/recording,
input/sampling, verifier repair, source/unit-failure audits and protected complete
engine acceptance before throughput. Required Vashti changes: none.


### Active Kuno engine handoff — prompt-only cache sources/controller (2026-10-14)

Source `60f6760` + aggregate-GR correction `12b310f` + model rebase `8b503e2`, after `f82d00c`, isolated `codex/strata-rust-port` at
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Read completion guide
**Connected prompt-only cache sources and one-session controller**. This
supersedes the previous handoff's missing fallback/table/prototype/controller
hookup. **The full rewrite and live validation remain incomplete.**

Real bounded registered HOST pages now hold original weights for lent cells,
refreshed from current placement under joined exclusion. Each page authenticates
its actual single source shard. Typed registry extension preserves all live and
retired identities/counters and quotes the metadata peak. A separately explicit
RAM grant is required; the compiled4096 fixture grants zero and stays owned.

The actual source transaction patches pointer/flag tables, prototype handles and
BDA read dependencies before exposing private scratch; native queued HOST copies
remain in use. One owning session/Scope spans the complete prompt plus MTP
teaching, with decode/adaptation excluded. Full private erase/inspection and
current-placement model restoration precede normal source publication. Unknown
completion/abandonment retires the shared context and retains owners. Loan mode
omits the three owned512MiB buffers from constructor/MAIN pricing. The obsolete
local2GiB arena ceiling is replaced by the actual precharged model plan.

Exact build/Clippy/focused test results and limitations are in the receipt above.
No GPU/model/allocation/state/recovery/speed acceptance or full suite this turn;
prior228 full-core failures remain unaudited. No services/residency, public API,
Vashti, install or release changes. Canonical/bytecode preserved. Goal active.

Next connect runtime context/configuration and pre-device RAM grants/cache
budget, conditional ring/stream-all/auto chunks, arbitrary input/sampling and
remaining captured branch/verifier-repair and source/unit-failure audits. Current
bound graph/request permits still cap4096; compiled conditional loan consumers
are not a supported live long-context invocation. Complete the whole-engine
correctness/state/recovery gates before speed tests. Required Vashti changes:none.

The follow-up `8b503e2` makes original-weight refresh explicit in the typed
registry, rather than mutating a resource still declared immutable. Begin revokes
lookups and joins native queues; its non-cloneable token grants only authenticated
model rebasing. Both fallback refresh and adaptive GPU/HOST exchange consume it.
Successful initialization/flush/raw completion is required to resume. Abandonment
keeps readers revoked and backing retained. This is an internal engine contract,
with no Kuno HTTP or Vashti change. See the final rebase receipts above.

**Current hardware coordination:** the user has reclaimed both GPUs while this
source/build phase continues (2026-10-14). Do not start GPU/model execution,
change admission/residency or stop their workloads without reconfirming that the
cards are available. Source edits, CPU-only unit checks and compilation may
continue. No GPU job or installed-service change was started in this pass.
