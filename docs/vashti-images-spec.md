# Vashti Images integration

Status: implementation authorized, 2026-09-23. The user approved the design and
added automatic prompt improvement, saved prompt history, and optional guidance.
This extends section 11 of [the Kuno specification](kuno-spec.md). Native workflow
support and current limits are recorded in [kuno-images.md](kuno-images.md).

## Product boundary

Provide two ways to use the same image service:

1. An **Images** tool group in ordinary chats, containing `generate_image` and
   `edit_image`. The LLM chooses prompts and which operation to use.
2. An **Images** workspace alongside Chats and Notes, with **Gallery**, **Generate**,
   and **Edit** views. Direct generation/editing does not require an LLM.

Place Images below New Chat and above Notes in the sidebar. Support deep links
to sessions and gallery items, browser Back, and reopening previous sessions.
Generate and Edit are persistent working histories, not disposable forms.

Qwen Image 2.1 generation and multi-image editing are the initial workflows.
Keep the older supported Qwen workflow available. Built-in workflows determine
which settings exist; do not present unsupported samplers, arbitrary node graphs,
or LoRA support that the selected runtime does not implement.

Vashti owns users, image history, settings, jobs, and storage. Kuno performs
transient inference and GPU scheduling; it still saves no inference content.
No new inference-engine optimization is part of this integration milestone.

## Chat tools and control over settings

Both tools use the existing admin tool catalog, permission tags, per-chat tool
toggles, activity display, and standard/private execution paths. Image-service
access and allowing an LLM to invoke image tools are distinct permissions: a
person can generate directly with the tools disabled in their chat.

`generate_image` takes a positive prompt and permitted setting overrides.
`edit_image` additionally takes an ordered list of one to ten image references.
References are opaque IDs exposed to that conversation, including its uploaded
and previously generated images. Vashti resolves and authorizes them; the model
does not supply filesystem paths, arbitrary URLs, or base64 in its tool JSON.
Do not grant a model blanket access to the user's gallery. Explicitly attaching
or selecting a gallery image makes it available to that conversation.

Maintain separate user defaults for generation and editing. Every applicable
setting has a default and an **Allow model to change** control. This applies to
steps, CFG, size/aspect ratio, seed policy, negative prompt, and supported advanced
options. A model sees effective defaults, override permissions, and supported
ranges. Omitting an override uses the default; omitting every override is valid.
Positive prompting remains part of the requested operation.

Resolve overrides on the server using the current user's policy. Ignore attempts
to change a locked value, run with its configured value, and report the effective
value and ignored override in the structured tool result/activity. Reject invalid
permitted values with a useful error. Never silently reduce locked quality
settings to make a request fit. Validate coupled size controls together so an
explicit width/height cannot bypass a locked resolution or aspect-ratio policy.
Direct user controls are independent of the model's override permissions.

For example, a person can fix 50 steps while allowing the model to choose an
aspect ratio. The model may expand "generate a weasel" into a detailed prompt,
then use the resulting image ID for "make it white," or generate afresh.
Prompt assistance in the dedicated workspace does not gain the chat tools'
authority to change settings automatically.

Each tool invocation produces one image initially. The model can make further
calls within existing tool-loop limits. Endless generation is a direct-user
control, not a model tool option.

Commit a successful image to its exact assistant revision and tool call before
continuing the LLM. Emit an attachment-ready event and a compact structured tool
result containing image ID, effective settings, and status. Rendering must not
depend on the model writing a Markdown link or even producing subsequent text.
Preserve the image if the LLM follow-up fails. Do not put image bytes in text
context; vision-capable models may receive appropriately bounded image inputs
through the provider's existing image channel. A text-only model still receives
the image ID and metadata but is not told that it has inspected the pixels.

Finish and release the originating LLM request before dispatching image
inference. Its in-memory conversation/tool-call state survives model unloading;
the next LLM request resumes with the tool result after image generation.
Route both requests through Kuno so placement and unloading remain coordinated.

Regenerating an ordinary assistant message produces independent image gallery
entries. **Never stack images across ordinary chat regenerations.**

## Generate and Edit workspace

Use one vertical scrolling surface: existing results, then the prompt, then its
settings. Results appear immediately above the composer. Preserve the prompt and
settings after Send so Send again produces another image without retyping.

Show positive and negative prompts, steps, CFG, aspect ratio/dimensions, seed
mode, image count, and applicable advanced controls. A visible **Customize
controls** action lets the user show/hide each setting, including negative
prompt. Hidden values remain effective and are still visible in result details;
hiding never resets or disables them. Persist visibility choices per user and
workflow. With extras hidden the workspace resembles a prompt-and-image chat.
Start from curated workflow defaults, rather than sharing one set between all
models. Using imported settings preserves their supported values exactly.

The positive prompt remains reachable and editable even in the minimal view.
Explicit **Save as default** updates defaults; adjusting the current session does
not unexpectedly change chat tool defaults or other sessions. Provide a clear
reset to workflow defaults. Capture effective settings at submission, including
hidden fields, so changes made while a job runs only affect a future submission.

Seed modes are **Random**, **Increment**, and **Fixed**. Resolve and record the
actual seed for each output. Use decimal strings end to end; never JavaScript
floating-point numbers for seeds. Increment within the supported range and stop
with a clear message at exhaustion instead of silently wrapping. New sessions
default to changing seeds. A Retry explicitly requests a different seed without
silently changing the session's saved seed policy; details show the seed used.

Offer a finite image count and **Endless**. Generate serially, committing each
result as it arrives. While active, Send becomes **Stop**; Stop cancels the active
image and all remaining images in that run. Already completed images remain.
Endless plus Fixed requires confirmation explaining that repeated identical
settings will generally reproduce the same image. One confirmation applies to
that run; changing relevant settings requires a new confirmation next time.

Each output offers Edit, Retry with a new seed, reuse settings, download, and
generation details. Edit starts an editing session using the selected image.
Normal Send and finite-count outputs are separate gallery items. Explicit Retry
creates a variant in the original image's stack; identical prompts alone do not
make a stack.

## Editing, variants, and branches

Show an ordered input-image strip with add, remove, reorder, and replace actions.
Make the distinction between the two editing modes explicit and always visible:

| Mode | Input used for a new instruction |
| --- | --- |
| Iterative | Only the selected preceding result |
| Original images | The original ordered image set for this editing session |

In iterative mode, "cabbages to carrots" followed by "shirt blue" uses the carrot
image for the second edit. In Original images mode the second request starts
from the original cabbage image, so retaining both changes requires both in the
prompt. Show a thumbnail of the actual base before submission.

For the first edit use every selected original input. After that, iterative
editing uses only the selected result; earlier references are not carried forward
automatically. Preserve those original inputs in history. Users can explicitly
add references to a particular new edit, but its result again becomes the sole
default input for the following edit. Switching mode or changing sources affects
future requests, never old results.

Each step stores an immutable prompt, setting snapshot, editing mode, exact
ordered source-image IDs, and exact parent result. Retry uses that step's original
inputs and parameters with a new seed; it does not edit its own output again.
Editing a previous prompt creates a sibling branch from the same prior result.
Do not overwrite the old instruction, images, or descendants.

Swiping/selecting a variant inside the workflow selects that version for future
edits and updates its gallery stack cover. Display that variant's existing
continuation, if it has one. Other continuations remain accessible on their own
branches. An existing "blue shirt" result continues to point to the exact carrot
version it used, even after another carrot version is selected.

Multiple outputs requested for one instruction are candidates from the same
input snapshot, not automatic successive edits. Each has its own gallery entry
and Retry stack; choose one to continue editing. The first successful candidate
is selected initially. Later batch arrivals must not steal the user's selection.
Disable conflicting submissions in the same session while a run is active;
users can inspect history and prepare the next draft without changing that run.

Reuse the existing conversation tree's navigation and interaction conventions,
with image-specific records. Do not force image operations into artificial
user/assistant message pairs or an LLM-specific chat schema.

## Gallery and fullscreen viewing

The gallery includes the user's standard-chat image outputs and Images-workspace
outputs. It can filter by workflow/source and open the originating session/chat.
New edit steps and separately requested generations each appear separately;
only explicit retries of the same image form a stack. Show its variant count
and the version selected in the workflow as the cover.

Fullscreen supports swiping through a stack, previous/next buttons, keyboard
navigation, and a variant counter. **Browsing** a gallery variant does not change
an editing branch. **Use this version** explicitly changes the corresponding
workflow selection and cover. Inside the editing workflow, selecting a version
already expresses that intent. Distinguish retries from separate edit steps in
labels and navigation.

**Copy to gallery** creates an independent gallery entry for that variant while
retaining the stack. It references the same immutable image asset; it need not
duplicate its file or disappear when the stack's selected version changes.

Use owner-scoped image assets with separate gallery, workflow, and chat links.
Deleting one link must not accidentally delete an image still used elsewhere.
Distinguish removing a gallery entry from deleting an image everywhere; explain
any remaining uses. A confirmed deletion everywhere removes the stored image and
leaves an unavailable-source marker in dependent history instead of silently
substituting another version. Deleting a chat removes its automatically listed
gallery entries along with its chat links, preserving existing chat-deletion
expectations. A separate gallery copy or editing session can retain an image;
make those remaining uses visible before deletion. Remove underlying bytes when
no live references remain. Account deletion removes all of that user's assets.

## Optional LLM help

**Improve prompt**, beside Send, uses a selected LLM and presents a suggestion alongside the
current prompt. An optional **Improvement guidance** field describes what to
change or fix, such as unwanted blue coloring. The user can edit or dismiss the
suggestion, or explicitly confirm **Use this
prompt** before it replaces the current text. Manual improvement does not generate
an image. The accepted wording becomes the working prompt for future sends and
further improvements. Preserve the
original user-written text and accepted suggestions in recoverable draft history,
including before any image has been generated. Repeated improvements must not
erase the original wording; provide Restore and retain the prompt actually used
with each result. A late suggestion cannot overwrite a newer draft without a
fresh acceptance. For edits, assistance should preserve the requested change
rather than rewrite the whole scene unnecessarily.

An explicit, off-by-default **Improve each image's prompt** setting opts into
automatic improvement. Every output, including each candidate in a finite batch
or endless run, independently improves the original submitted prompt using the
submitted guidance. Never feed an improvement back into the next improvement.
Preserve both input and improved wording in prompt history and retain the exact
effective prompt with each result. Leave the working draft unchanged; only
explicitly accepting a manual suggestion or restoring history replaces it.
Stop cancels the improvement stage as well as image generation. If improvement
fails, report it and do not silently generate with a different prompt. Manual
Retry repeats the recorded recipe without another improvement.

The selected LLM must be available through the configured backend; direct image
generation remains usable when no LLM is selected or available. This action does
not expose the entire gallery. Sharing chosen reference images for vision
assistance is explicit; a non-vision model only receives the textual request.

## Execution, storage, and recovery

Introduce a shared Vashti image service used by both entry points. Conceptually:

- **Asset:** immutable image bytes, owner, dimensions, MIME type, provenance.
- **Session:** workspace mode, initial source set, draft, selected branch.
- **Step:** immutable instruction, effective options, ordered inputs, parent result.
- **Result group:** one independently requested image and its Retry variants.
- **Attempt/result:** resolved seed, output asset, status, timings, runtime identity.
- **Run:** one Send, its finite count or endless policy, cancellation and progress.
- **Gallery entry:** standalone asset or result-group reference and cover selection.

Persist provenance in Vashti: prompts, effective recipe, ordered input references,
actual seed, workflow/model identity, attention mode, known runtime version, and
timings. Keep requested versus effective differences visible. Do not invent model
hashes or timing details unavailable from an imported image or the runtime.
Store actual wall time separately from queue/inference stage timings when those
are measurable. Matching imported parameters does not promise identical pixels.

Standard runs are server-owned and survive browser navigation/disconnection.
Client submission IDs prevent double taps/reconnects from enqueueing duplicates.
Return a job ID promptly; stream or poll state separately from its execution.
Keep a visible active-run indicator and Stop action across Images views. A server
or Kuno restart marks interrupted work as interrupted; do not silently restart
an endless run or replay work whose outcome is unknown. Already saved outputs
remain available. Explicit Resume uses a new submission and preserves history.

Dispatch one native image per request. Maintain a bounded Vashti queue with fair
opportunities between images, so endless work cannot reserve Kuno indefinitely.
Do not enqueue infinitely many requests or hold whole batches of images in RAM.
Enforce administrative storage/concurrency limits before starting and before
each subsequent image. Stop on an error or exhausted storage with a useful
message and retained completed outputs; avoid unbounded automatic retry loops.

Cancellation must abort the active Vashti-to-Kuno request and await worker
cleanup before declaring GPU work stopped. Handle cancellation versus successful
completion atomically: an image committed before cancellation remains; no new
attempt starts after Stop. A disconnected progress subscriber is not a Stop for
a server-owned run. Permissions, ownership, and job idempotency are checked on
every endpoint, including status, cancellation, assets, and variant selection.

Private-local chat tool output stays in the existing encrypted device vault.
Neither Vashti's server nor Kuno persists private prompts/images/jobs. Use
ephemeral execution and transient result delivery to the authenticated client;
do not automatically add those results to the server gallery. Moving a private
image to the server Images workspace is an explicit copy with a clear storage
notice. A disconnected private owner follows private-chat cancellation/recovery
semantics rather than being converted into a durable server job. Reuse the client
tool broker for device-owned image resolution and persistence; image bytes travel
through the transient image response, not through its bounded text-result field.
Propagate Stop through tool execution as well as through the LLM request, and
do not start a follow-up LLM request after cancellation.

The current Kuno API returns one PNG and has a basic enabled flag, not complete
capability/progress discovery. Add bounded, versioned capability discovery for
supported workflows, readiness, settings/ranges, reference limits, and optional
attention modes. Vashti connects server-to-server with configured credentials;
do not expose privileged Kuno configuration/control tokens to browsers. Missing
models should lead admins to the curated collection download/status controls.
Readiness errors should be actionable for ordinary users.

Initially display only states actually known: queued, generating, saving,
complete, stopping, stopped, interrupted, or failed. More detailed Kuno phases
and step progress require a real event contract; do not fabricate percentages,
live previews, stage timing, or loading progress from elapsed time.

Live preview requirement: expose the engine's actual denoised latent projection
as an approximate image while sampling, with its reported step number. Request
one preview per step, but keep only the latest frame; slow clients may skip
frames and must never hold up generation. Display previews in the Images flow
and chat image tools, replacing them with the final decoded image. Preview
frames are transient on Kuno, Vashti, and the device: no files, database rows,
gallery entries, history versions, analytics, or log payloads. Stop, failure,
disconnect expiry, and completion clear them. Older engines without preview
support continue to return final images without fabricated progress.


## Layout and compatibility

On mobile, image cards nearly fill the content width. Desktop cards respect both
available width and viewport height, preserving the entire image without crop.
Settings share the page's vertical scroll; avoid a second tall scrolling panel.
Retain scroll position when results arrive while the user inspects old images;
provide a new-result indicator instead of forcing them to the bottom. When
following generation at the bottom, keep the new result/input naturally visible.

Use horizontal gesture discrimination so vertical scrolling does not select a
different variant accidentally. All swipe operations have buttons/keyboard
alternatives. Fullscreen and the mobile keyboard respect safe areas and Back.
Ordinary chats keep their existing image presentation.
Paginate long histories and gallery queries, load thumbnails lazily, and bound
decoded-image memory so an endless run does not make the browser unusable.

Reuse the ComfyUI metadata importer for **Use settings**, including explicit
missing-reference and unsupported-workflow messages. Imported edit images do
not contain their original reference images: request those separately. Maintain
source alpha, orientation handling, and reference order. Enforce the runtime's
current limits, including the 16-pixel output grid and nonnegative signed-64-bit
native seed range. Keep Standard attention default; Fast remains an explicitly
supported option whose numerical output can differ.

## Implementation order and acceptance

1. Shared assets/jobs, Kuno connection/capability handling, cancellation, ownership,
   recipe resolution, and private transient execution. Extend attachment linking
   without breaking existing chat uploads or deletion behavior.
2. Generate/Edit chat tools, typed artifact results, activity/attachment streaming,
   personal defaults and enforced override permissions.
3. Images navigation, direct generation/settings, finite/endless runs, gallery,
   and fullscreen viewing, using the shared execution path.
4. Multi-image editing, original/iterative modes, retry selection, prompt branches,
   stack copies, prompt assistance, and ComfyUI settings reuse.
5. Mobile/desktop and lifecycle verification, documentation, packaging/setup
   checks, then a release-readiness checkpoint. No public release is implied by
   this design approval.

All requested flows belong to the integration target; this order is not a reason
to omit branching, multi-image edits, or gallery stacks from the finished work.
Reuse current `toolPreferences.ts`, `tools/service.rs`, chat/private handlers,
`chatMessages.ts`, version gesture logic, `attachments.tsx`, routing, and the lazy
Notes workspace pattern. Current tool results are text-only and attachments
require a chat ID: both are explicit implementation boundaries, not assumptions
that gallery persistence and binary tool output already work.

Required evidence includes:

- Locked steps with model-selected aspect ratio; correct effective-settings UI;
  owner isolation and no references to inaccessible gallery/private assets.
- Generated images displayed even if follow-up text fails, tool cancellation,
  LLM unload/reload handoff, and private output absent from server persistence.
- Finite count, changing seeds, fixed-seed endless confirmation, Stop while
  queued/running/saving, refresh/reconnect without duplicates, restart interruption.
- Carrots then blue shirt in both editing modes; retry from original step inputs;
  alternate retry selection and earlier-prompt branching preserve old descendants.
- Ordered multiple references; batch candidates do not become successive edits;
  late batch results do not override user selection.
- Gallery cover selection, preview without branch mutation, independent copies,
  ordinary chat regeneration without stacking, and deletion with shared references.
- Narrow mobile touch/keyboard/scroll behavior, desktop image sizing, alpha,
  imported settings, missing models, storage exhaustion, and unavailable inputs.

## Confirmed follow-up decisions

- Prompt assistance is optional, requires explicit acceptance before replacing
  user text, and preserves the original wording for recovery.
- Iterative editing uses only the latest selected result on the next edit;
  earlier reference images are not retained as automatic inputs.


## Implementation checkpoint (2026-09-23)

The development branch now contains the Images workspace, standard-chat tools,
server-owned image runs and gallery, prompt improvement/history, batch/Stop,
iterative/original edit modes, and a separate private-chat image execution path.
Private image jobs, inputs, output bytes, and previews exist only in bounded RAM;
owner polling expires abandoned jobs, result retrieval removes the entry, and
completed entries expire after 60 seconds. The device encrypts final private
attachments and recipe metadata in its existing vault. Private calls never use
standard image sessions, runs, assets, attachments, history, or gallery storage.

Verified: Rust unit/service tests, strict Clippy, frontend build; browser checks
for desktop and 390px mobile preview layout, Stop clearing previews, manual
prompt improvement acceptance, regular-chat preview/final attachment, and
private-chat preview/final attachment retained after reload. The private browser
run left server content-table counts unchanged. Native CUDA smoke test emitted
two real Qwen 2.1 RGBA previews and a final PNG without saving images. Installed
services have not been replaced by this development build.

Remaining release work includes image lifecycle/quota reclamation, paged stack
and ancestor lookup, tool-call replay handling and incremental tool seeds,
curated collection readiness/download UX, native mobile download verification,
full edit/branch/endless regression testing, and packaged deployment validation.
Vashti Stop aborts the upstream request; Kuno keeps its admission lease until
worker cleanup. Vashti does not yet expose a separate acknowledgement that GPU
cleanup has finished. Retain the original `image-ui-notes.txt` until this checklist
and the approved behavior above have been reconciled.


## Beta packaging checkpoint (2026-09-23)

The testing release is `0.13.0-beta.1`; it is not eligible for stable promotion.
Linux, signed Android (versionCode 13001), and native CUDA/Vulkan Kuno artifacts
are packaged together. Android signing identity matches 0.12.0. The release notes
explicitly retain the unfinished feature and device-test items above. The packaged
Linux server passed fresh-database startup and embedded-frontend checks. All 270
workspace tests passed (8 intentionally ignored); subsequent Hub streaming-upload
changes passed 14 Hub tests and a full signed three-artifact local upload test.
Hub now supports SemVer beta labels, Kuno downloads/install/setup routes, and
constant-memory streamed release uploads. Stable remains 0.12.0.

Publication completed: `v0.13.0-beta.1` is available at https://vashti.chat/releases
with Linux, Android, and Kuno artifacts. All three public downloads were read back
and matched their SHA-256 checksums and Ed25519 manifests. Hub's low-memory upload
fix is deployed and healthy; stable remains `v0.12.0`. No stable promotion or local
Vashti/Kuno service upgrade was performed as part of this release publication.

## User testing notes (2026-09-24; record only, no implementation yet)

The user wants to finish testing beta.1 before any behavior changes or the next
release. Collect feedback without implementing it in the meantime.

- Vashti should probe a dedicated Kuno discovery/capability endpoint on each
  inference host. If it receives no Kuno response, treat the host as regular Ollama.
- A host already connected for inference should support image generation without
  a separate user-facing key/token setup. Remove the current manual Kuno access
  token workflow in the next design; the user does not want this extra step.
- Include the release version in the Android download filename, for example
  `vashti-0.13.0-beta.1.apk`. The app already has an internal version/versionCode;
  the shared `vashti-android.apk` filename makes the browser ask to download it
  again even when it is a new release. Update download links and manifests together.

For beta.1 testing, the installed inference host's current access token is stored
at `/etc/kuno/control-token` (root-readable). Retrieve it locally with
`sudo cat /etc/kuno/control-token`, then paste it into Images → Settings →
Kuno access token and save the connection. Do not record the token in these notes.

Additional beta.1 testing feedback:
- Offer an explicit unlimited-storage option alongside a finite per-user quota.
- Numeric inputs must allow an empty editing state when all digits are deleted;
  do not immediately replace the empty field with zero. Validate on commit/blur/send.
- The user successfully generated images and requested a read-only privacy audit
  for evidence of generation content on the inference host.

Read-only inference-host audit after the user's beta generations:
- Kuno journal contained no entries in the preceding two hours.
- Service-private /tmp and /var/tmp contained no regular files.
- Kuno state inventory contained usage.json (model name/digest/call count/last-use
  fields), boot/lock state, Ollama identity files, and GPU driver caches. No saved
  image or prompt file was identified. No model-store files changed in that window.
- Managed Ollama worker caches included 150 Mesa shader-cache files, 21 NVIDIA
  ComputeCache files, and four other NVIDIA cache files; 26 Mesa entries changed
  within the audit window. These reveal activity but did not identify image content
  in this inspection. Cache payloads were not exhaustively decoded.
- MemorySwapMax and LimitCORE were zero for the running Kuno service.
This establishes the result of a scoped file/log audit, not proof that a privileged
operator could never inspect transient process/GPU memory. No behavior was changed.
