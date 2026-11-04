# Personal Assistant

Status: approved workspace direction; implementation in progress, 2026-11-03.
Branch: `codex/personal-assistant`, based on `4bdd9dfc`.

## Goals and Checkpoints

- [x] Research persistent assistants and inspect Vashti's existing capabilities.
- [x] Confirm the dedicated Assistant workspace with the user.
- [x] Implement owner-scoped durable tasks, runs, events, approvals, and recovery.
- [x] Integrate existing permission-checked tools and rootless Podman workspaces.
- [ ] Add the responsive workspace, routines, progress, results, and approvals.
- [ ] Test failure/security boundaries and the real installed Kuno Qwen model.
- [ ] Refine, document measured limits, and commit. No deployment is authorized.

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
  complexity. Use one worker per account and a bounded global concurrency.
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

The inbox lists tasks needing attention, active work, and completed work.
Opening a task gives a full-width conversation-like activity view on mobile,
with progress, results, tool records, files, and precise approval requests.
Frequent controls remain direct; occasional settings live in a sheet/menu.

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

## Backend Checkpoint

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

- Existing 145 Rust tests remain passing; 16 additional deterministic assistant
  tests cover ownership, stale edits, approvals/rejection/revocation, questions,
  crash uncertainty, queue exclusion, cancellation, budgets, malformed/incomplete
  streams, schedule overlap/failures, permission defaults, and HTTP downloads.
- Explicit rootless Podman test passes: offline networking, read-only root,
  workspace persistence, export, traversal/symlink rejection, timeout, and no
  leftover test containers. It caught and fixed an unclosed archive input pipe.
- Installed `kuno-qwen3.8-flash-next:iq3_s` passes the real multi-tool file test:
  calculation, workspace execution, file verification/export, exact CSV bytes,
  and a final link to the actual artifact. Disposable data only.
- An initial fixed 64K context request was rejected by the installed engine.
  Vashti now reads the ordinary `/api/show` profile and honors its configured
  context (currently 32768) instead. No Kuno source/config/service changes.

The interactive frontend and full refinement/validation pass are still in
progress. These receipts are not release or physical Android acceptance.
