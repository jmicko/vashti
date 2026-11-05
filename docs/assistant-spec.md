# Personal Assistant

Status: conversational revision implemented and locally validated, 2026-11-04.
Branch: `codex/personal-assistant`, based on `4bdd9dfc`.

## Goals and Checkpoints

- [x] Research persistent assistants and inspect Vashti's existing capabilities.
- [x] Confirm the dedicated Assistant workspace with the user.
- [x] Implement owner-scoped durable tasks, runs, events, approvals, and recovery.
- [x] Integrate existing permission-checked tools and rootless Podman workspaces.
- [x] Add the responsive workspace, routines, progress, results, and approvals.
- [x] Test failure/security boundaries and the real installed Kuno Qwen model.
- [x] Refine and document measured limits. No deployment is authorized.
- [x] Replace the task-first entry screen with the approved conversational flow.
- [x] Persist real conversation turns and expose scoped task-management tools.
- [x] Validate foreground/background queue separation and live Qwen coordination.

## Research and Decisions

Reviewed official documentation on 2026-11-03. These are design inputs, not
claims of feature parity with the referenced products.

- [OpenAI dots](https://learn.chatgpt.com/docs/dots) separates an ongoing
  responsibility from a single chat response. Work remains inspectable and can
  continue while the user is away. Use a task inbox and durable run records.
- [Dots controls](https://learn.chatgpt.com/docs/dots/controls) distinguishes
  tool access, action approval, and instructions. Preserve that distinction;
  text in retrieved material cannot grant permissions.
- [Grok routines](https://docs.x.ai/grok-bot/skills-routines-and-automations)
  separates reusable procedures from their triggers. Start with a tested task,
  then schedule it; preserve run history and avoid overlapping occurrences.
- [Hermes architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)
  separates the model loop, tool dispatcher, persistent sessions, and gateway.
  Reuse Vashti's Rust tool dispatcher rather than adding another application or
  Python orchestration framework.
- [Hermes scheduling](https://hermes-agent.nousresearch.com/docs/developer-guide/cron-internals)
  makes scheduled runs self-contained and guards against recursive scheduling.
  Schedules are user-owned; models cannot create unbounded recurring work.
- [Hermes security](https://hermes-agent.nousresearch.com/docs/user-guide/security)
  treats authorization, isolation, and approvals as separate layers. Do not
  mistake a container or a safety prompt for complete authorization.
- [Anthropic's agent patterns](https://www.anthropic.com/research/building-effective-agents)
  favor a simple tool loop with explicit stopping conditions before multi-agent
  complexity. Bound foreground and background concurrency separately.
- [Tool design](https://www.anthropic.com/research/writing-tools-for-agents)
  emphasizes concise schemas and outcome-based evaluations. Test observable
  artifacts and mutations, not just fluent model narration.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
  and [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
  motivate persisted checkpoints around human decisions. Crash recovery must
  not blindly replay a possibly completed side effect.
- [OWASP agent security](https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html)
  requires least privilege, action-specific review, tenant isolation, and
  execution-time enforcement. Approval binds the exact persisted tool arguments.
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
  supplies the existing inference protocol. Kuno is an unchanged consumer-facing
  backend; no new inference endpoint or model installation is needed.
- [Podman run](https://docs.podman.io/en/latest/markdown/podman-run.1.html)
  provides rootless containers, resource controls, and network/filesystem
  isolation. This is not a full VM or a guarantee against kernel escapes.

## Product Flow

Assistant is a separate destination alongside Chat, Images, and Notes. Ordinary
chats and custom-model behavior do not change. The user chooses an allowed
backend/model; prefer the installed `kuno-qwen3.8-flash-next:iq3_s` when offered,
but do not silently select a different model if it is unavailable.

Assistant opens into an ongoing conversation, not a task inbox or task form.
Persist user/assistant turns across runs. Tasks and schedules are a secondary
inspector. The coordinator has task creation, editing, inspection, and cancellation
tools; background workers cannot delegate recursively. Task capabilities inherit
the coordinator's selected permissions and cannot be elevated through arguments.
Creating or changing a task requires exact-action approval. Results, questions,
and approvals are surfaced back in the originating conversation. Foreground
conversation and background work have separate bounded worker lanes. A backend
may still serialize inference; this does not promise additional model capacity.
Existing standalone tasks and their provenance links remain accessible.

A task contains instructions, model selection, capability choices, and limits.
Each execution is a separate run. Follow-ups can resume a paused checkpoint or
start a new run with explicit new instructions. A saved recurring task is a
routine, not a second hidden agent. Results remain in Vashti; no email, social
posting, payments, or production/host administration tools are introduced.

The first implementation is explicitly server-stored. Its work can continue
while the browser/phone is closed, provided the Vashti server is running.
It cannot read encrypted device-only notes, memories, chats, or files. A future
device runner needs separate foreground/resume semantics; never upload a local
vault to simulate background support.

## Execution and Safety Contract

- Persist task/run ownership and activity in SQLite. Every route checks the
  authenticated owner, including polling, approval, and artifact downloads.
- Re-check account, backend/model, tool tags, and record-level permissions at
  execution, including after waiting for approval. Revocation must take effect.
- Existing note/memory mutations retain version checks, provenance, history,
  and recoverable deletion. Approval cannot bypass those restrictions.
- Reads/calculation may execute automatically within the selected capabilities.
  Writes and external web requests require exact-action approval by default.
  Explicit task policy may narrow or pre-authorize supported action classes.
- User-input requests are persisted pauses, not fabricated answers. Stop/Pause
  cancels active inference/tool work; resuming is explicit and owner-controlled.
- Bound steps, generated output, context, pending tasks, concurrency, tool
  output, and container resources. Show a limit/interruption rather than claiming
  success when the model or worker stops early. Do not impose a short arbitrary
  inference HTTP timeout or alter Kuno's long-request behavior.
- Checkpoint before tool execution. Record executing/completed separately.
  On restart, expose uncertain side effects for review; never silently replay
  them. Waiting approvals survive restart. No automatic retry of mutations.
- Each recurring occurrence gets a fresh run; do not overlap a routine's runs
  or catch up every missed interval after downtime. Pausing a routine and
  stopping its active run are separate, visible operations.
- Keep credentials and prompt/output content out of operational logs. Agent
  activity is owner-authorized application content, not Kuno telemetry.

## Sandbox Contract

Podman is optional and rootless-only. Availability is reported honestly when it
is missing, misconfigured, or its approved image is not installed. Do not run
privileged containers or auto-download arbitrary images supplied by a model.

- A dedicated container/workspace per run, with installation-specific labels.
- No host directory, home, socket, device/GPU, or credential mounts.
- Network disabled. Web requests use Vashti's separate controlled tools.
- Read-only root, dedicated writable workspace, bounded temporary storage,
  dropped capabilities, no-new-privileges, memory/CPU/PID/output/time limits.
- Commands run only inside that container via fixed argument-array Podman
  invocations, never interpolated into a host shell.
- Pause/cancel/finish stops sandbox processes. Retain only the managed
  workspace/artifacts needed for explicit continuation; cleanup never touches
  another application's containers or volumes.
- Exported files are bounded, owner-scoped downloads, not inline executable
  HTML. Validate paths and do not follow host symlinks or expose server paths.

## Validation Gates

Use disposable users/databases/containers and synthetic notes. Cover ownership,
permission revocation, duplicate decisions, rejection, stale note versions,
restart during a mutation, cancellation, step limits, schedule overlap/missed
ticks, malformed streams/tool calls, and hostile retrieved instructions.

Exercise actual multi-step tool use and a created file with the installed
Kuno-prefixed Qwen model. Keep tool/network/approval boundaries enforced even
when the model asks otherwise. Browser-test mobile and desktop flows, build the
web/native frontend, and distinguish those receipts from physical Android QA.

## Conversational Checkpoint

This supersedes the original task-first interface described in the historical
checkpoints below. `/app/assistant` opens the most recent conversation, or a
normal composer when none exists. New Chat starts a separate conversation.
Model selection and secondary tasks/settings controls share the existing top
bar. Conversation history and recoverable archiving are available in its menu.
Ordinary Chat is unchanged.

Conversation turns are durable runs with full owner-visible activity. New turns
reload recent user/assistant text, not merely the last result or a duplicated
checkpoint. The inference history is bounded to 160 events/24,000 bytes and
further constrained by the selected model's advertised context, current message,
instructions, tool schemas, and an output reserve. Older history remains stored
and pageable; it is not unlimited model memory. Prior private reasoning and raw
tool bodies are not copied into a new turn. Protected records must be read again
through current permission checks. Task status/results are inspected with tools,
not treated as new user authorization.

Only conversation runs receive `create_task`, `update_task`, `list_tasks`,
`read_task`, and `cancel_task`. Creation/editing requires exact-action approval,
including scheduling. A task has self-contained instructions and inherits the
coordinator's model, capabilities, policies, and budgets. No capability overrides
are accepted in tool arguments; subsequent edits can only narrow task access.
Task tools cannot access a different owner's or a different conversation's jobs.
Background workers cannot delegate or schedule recursively. Missing time zones
require clarification; scheduling uses explicit UTC Unix seconds.

One foreground and one background worker may run per owner, with two workers
globally in each lane. Cancel-cleanup exclusion is also lane-specific. Pending
foreground/background runs are separately bounded to eight each. There are at
most 100 active conversations, 50 delegated tasks per conversation, and the
existing 200-active-task bound. Backend/model capacity may still queue inference;
Vashti does not change Kuno scheduling or promise simultaneous GPU execution.

Results, exact approvals, questions, files, and task controls appear in the
originating conversation. Each occurrence retains its own result, rather than
being replaced by the next routine run. The latest 50 background occurrences
are shown with access to older task history. Conversation polling transfers
small turn/status summaries; completed turn activity is loaded once and cached
in mounted components. Reasoning expands inside the single conversation scroll
area. Viewing a completed result clears its attention marker. Archiving cannot
hide a conversation's active jobs/schedules; stop them first.

The ordinary composer is reused with an optional controlled draft. The owner of
that draft controls clearing after acknowledgement, preserving request and
creation identities across failed requests, navigation, and retries. Drafts stay
encrypted and account/server scoped. Account-epoch checks guard chained saves
and sends, so an account change cannot submit the previous account's draft.
Interrupted/uncertain effects still require explicit review, not automatic replay.

Current validation receipts:

- 174 deterministic Rust tests pass; five new conversation cases cover history,
  ownership/retries, independent worker capacity, task approval/inherited access,
  recursion prevention, schedules, durable results, and archival safety.
- The explicit installed `kuno-qwen3.8-flash-next:iq3_s` coordinator test passes:
  task approval, actual calculator work in the delegated run, tool-verified result
  retrieval, and an earlier conversation codeword recalled in a later turn.
- Browser fixtures cover failed sends, draft restoration after navigation, new
  conversation identities, inline approvals while a separate response runs,
  stop, task navigation/return, and long reasoning without outer/double scrolling.
  Mobile checks include a 390 x 844 viewport with no horizontal overflow.
- The signed-in development UI also completes a harmless delegated calculator
  task through an actual approval. No real notes or memories are changed.
- Frontend tests, type checking, web/PWA/native builds, and warnings-denied Clippy
  pass. Physical Android acceptance is still outstanding. No Kuno source, API,
  service configuration, installation, release, or deployment changed.

This remains server-stored work with polling while Vashti is open, not a device
push-notification system. A later turn can inspect completed results; the server
does not launch an unsolicited foreground model turn on every completion.

## Initial Backend Checkpoint

Implemented the task/run/action/event/artifact store, polling and control APIs,
owner-only downloads, versioned note/memory attribution to assistant runs,
per-action approvals, durable questions, step/output bounds, one worker per
account/two globally, routine dispatch, explicit interruption review, and rootless
offline workspaces. Sandbox permission defaults to administrators; existing tool,
model, and record access rules still apply.

Workspace storage is a bounded tmpfs inside a disposable container. Commands
checkpoint a compressed archive (maximum 64 MiB); restoration happens inside the
next isolated container, never by extracting archive members on the host.
Containers are removed after commands. Published artifacts are limited to 20 MiB
each, 32 files/128 MiB per run, and 512 MiB per account; owners can delete them.

Current receipts (2026-11-03):

- Existing 145 Rust tests remain passing; 24 additional deterministic assistant
  tests cover ownership, stale edits, approvals/rejection/revocation, questions,
  crash uncertainty, queue exclusion, cancellation, budgets, malformed/incomplete
  streams, schedule overlap/failures, permission defaults, and HTTP downloads.
- Additional cases cover exact question/reply retries, batched questions,
  immutable creation-request identities after task edits, context compaction,
  protected-record retrieval, full-library filters, and idle SQLite contention.
- Explicit rootless Podman test passes: offline networking, read-only root,
  workspace persistence, export, traversal/symlink rejection, timeout, and no
  leftover test containers. It caught and fixed an unclosed archive input pipe.
  Actual cgroup values also enforce 1 GiB memory, 128 PIDs, and one CPU.
- Installed `kuno-qwen3.8-flash-next:iq3_s` passes the real multi-tool file test:
  calculation, workspace execution, file verification/export, exact CSV bytes,
  and a final link to the actual artifact. Disposable data only.
- An initial fixed 64K context request was rejected by the installed engine.
  Vashti now reads the ordinary `/api/show` profile and honors its configured
  context (currently 32768) instead. No Kuno source/config/service changes.
- The installed Qwen also passes a real note-creation approval test: no note
  exists before approval; afterward the exact run is retained in version history.

## Initial Interface and Retry Checkpoint

`/app/assistant` contains the inbox, server-side search/filtering, task settings,
account defaults, schedules, history, progress, questions, approvals, and files.
The canonical bootstrap API is `GET /api/assistant` without a trailing slash;
the real-app check caught a client mismatch not represented by the initial fixture.
Sandbox availability is checked separately so Podman discovery cannot block the
inbox. Initial task-list responses contain previews, not full prompt/config text.

Task/run request IDs and follow-up drafts are persisted in the existing
account-and-server-scoped encrypted device vault. Late storage operations cannot
cross an account switch. A response lost after a start can be retried with the
same request ID without creating a second run. Questions bind replies to the
exact action; identical retries are accepted, changed replies are not replayed.
Run creation and replies include their activity event in the same transaction.

Polling fetches new events and the unfinished suffix of tool actions, preserving
component identity and avoiding repeat downloads of old result bodies. Explicit
action events preserve order even when multiple steps occur in one second.
Older model reasoning can leave the inference context without leaving the audit.
Long tool output is compacted into a bounded preview and a retrievable reference.
Protected notes, memories, and chats must instead be re-read using their original
permission-checked read tools; the activity cache cannot bypass that boundary.
Generated Markdown images are links, not automatic outbound image requests.

Browser acceptance used isolated fixtures for save/run, failed submissions,
pause/resume/stop, previous-run navigation, questions/choices, approve/reject,
note references, history, archive filtering, schedules, and an actual CSV
download. Long reasoning expands within the one main scroll area. The running
development app also completed a synthetic arithmetic task through the UI using
the preferred Kuno model, with unrelated tools disabled. Its task/run survived
the development watcher's subsequent restarts.
An additional follow-up produced the verified value 2331 for 63*37 in a separate
run, while preserving the earlier run in history. Desktop, 390px mobile, 320px
narrow, and 844x390 landscape geometry were checked. The administrator's real
Tools page shows the workspace permission with the existing admin tag; no
permission settings were changed during that acceptance check.
Saving account defaults was also checked through opening a new task without
reloading; the authoritative saved config now replaces the old in-memory defaults
immediately. This caught and fixed a stale-config handoff in the first UI pass.

Validation receipts:

- `cargo test -p vashti --locked`: 169 passed, three explicitly ignored live tests.
  Both installed-Qwen tests and the rootless-Podman test were separately run and
  passed. No leftover assistant containers remained afterward.
- Frontend Node tests: 89 passed; production/PWA and native frontend builds pass.
- Browser fixture TypeScript and warnings-denied Rust Clippy pass.
- Android host tests: eight passed. All four Android ABIs compile and the signed
  universal APK builds. Physical Android installation/export remains untested.

## Setup and Deliberate Limits

Use the Assistant workspace in the normal Vashti app. New tasks use account
defaults; editing a task affects only its next run. A routine uses one-shot or
fixed-interval timing, not a calendar/DST-aware cron expression. Dispatch failures
stop the schedule and remain visible for human review rather than looping.

For the optional workspace, install rootless Podman on the **Vashti server**, not
necessarily the inference machine. As the OS user running Vashti, prepare:

```sh
podman pull docker.io/library/python:3.12-slim-bookworm
```

Vashti never pulls a model-supplied image. Each run pins the already-installed
image ID. The sandbox permission defaults to administrators and can be assigned
to existing permission tags under Settings > Tools. Each task separately opts in.
No new Rust or frontend orchestration framework/dependency was added.

This is a server-side assistant, not a phone background service or a full VM.
The server must stay running. Device-only vaults remain unavailable. Rootless
containers share the host kernel; isolation is not a guarantee against kernel
exploits. There is no host shell, arbitrary network/package installation, general
browser/computer control, email, payments, recursive scheduling, or delegation.

Defaults are 24 steps and 24,576 output tokens; tasks can choose up to 64 steps
and 131,072 tokens. The context guard is a conservative estimate, not an exact
Qwen tokenizer. A provider may still reject a prompt. Context/output limits stop
visibly and retain activity, rather than claiming successful completion. Model
results still need human judgment. Multi-day soak and physical-phone acceptance
have not been claimed. Tasks can be archived; permanent task-history purge is
not implemented in this first slice. Artifact deletion is explicit and owner-only.
The run picker currently lists the 50 most recent runs; exact older run links
remain owner-accessible. This is not an unbounded history browser.

These receipts are not a release, deployment, or physical Android acceptance.
