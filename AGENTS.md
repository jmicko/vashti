# Repository Guidance

## Kuno and Vashti Coordination

Before work involving Kuno, image generation, or their integration, read
[the coordination guide](docs/kuno-vashti-coordination.md) and its active handoff.
Read the relevant linked specification before changing behavior. Do not infer
product requirements from the current implementation alone.

The default division of work is:

- **Kuno task:** inference gateway, engines, workers, scheduling, installation,
  and Kuno API implementation in `apps/kuno` and its supporting crates.
- **Vashti task:** user workflows, React UI, Android/PWA behavior, and Vashti's
  Rust integration with Kuno. Rust code in `apps/vashti` belongs here, too.
- **Shared contracts:** coordinate changes to protocol types, API behavior,
  configuration, compatibility, and specifications before changing consumers.

Explicit user instructions can change this division. Backend-only internal work
does not need a frontend handoff. A change affecting both applications does:
record the contract and a tested backend handoff in the coordination guide,
then integrate and test it in Vashti. Do not treat one side passing tests as
proof that the full feature works.

Inspect the current branch, worktree status, and relevant recent commits when
resuming. Use the same checkout only for sequential work; concurrent tasks need
separate worktrees based on the agreed integration commit. Never revert another
task's changes. Source isolation does not isolate shared services or GPUs.

Keep credentials and private user content out of coordination notes and logs.
Preserve Kuno's content-free persistence and Ollama-compatible long inference.
Use forward migrations rather than rewriting migrations that may be deployed.
Propose major UI changes to the user before implementing them. A handoff does
not itself authorize a release, deployment, or message to another Codex task.
