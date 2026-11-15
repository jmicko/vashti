# Automatic Personalization

## Contract

Automatic personalization is separate from model-invoked memory tools. A selected
processor extracts durable, user-stated facts from completed chats during idle
time. Chat requests receive a bounded, source-checked profile without a tool call.
The feature is off until the user chooses it; no processor is selected silently.

Approved defaults: ordinary local models can contribute and receive context;
cloud models and character custom models cannot, unless individually allowed or
their group is enabled. A per-model exception overrides the group rules. Unknown
model locality is treated as cloud. Rules apply to both collection and delivery.
The processor is selected separately, and cloud processing requires explicit
consent. Backend/user model permissions still apply.

Server and device profiles never merge. Server processing can run while the user
is away. Device processing runs only with the app open; its profile, settings,
history, source IDs and processed markers are encrypted in the existing scoped
device vault. Private inference payloads transit Vashti like other private chats,
but are not saved to its database, queue, profile or logs.

## Data And Lifecycle

- Extract only explicit useful facts/preferences from user messages, not assistant
  claims, reasoning, tools, attached files, fictional dialogue or system prompts.
- Structured candidates cite an exact source revision and verbatim evidence.
  Invalid evidence is rejected. Model output remains untrusted data, not policy.
- Facts have a stable topic key, current text, provenance and revision history.
  Newer evidence can supersede an automatic fact; human corrections are locked
  against automatic replacement. Human-added facts need no conversation source.
  Repeated corroboration updates evidence without creating another content version.
- Model identity is captured per user-message revision. Switching a conversation's
  model later does not reclassify old statements. Historical edits without reliable
  provenance are skipped rather than classified using the current model.
- Forgetting leaves a content-free topic/source suppression marker, so old chats
  cannot immediately recreate a forgotten fact. Forget-all suppresses existing
  sources, while permitting newly written conversations. History is removed for
  forgotten facts; it must not become a second hidden memory store.
  Suppression stores topic digests rather than the topic's text. Forgotten topics
  remain excluded from automatic relearning; users can still add facts manually.
  Forgetting a profile fact does not delete its original conversation.
- Before publishing a batch, recheck settings revision, profile revision, ownership,
  source revisions/deletions and model eligibility. Stale jobs are discarded.
- Before injection, recheck ownership, settings, recipient and source eligibility.
  Deleting or editing a source removes its derived context immediately. Archived
  chats remain history; deleted messages and changed revisions do not.
- Processing is serial, bounded and retry-backed-off. Foreground generation takes
  priority. Failure never prevents ordinary chat. No inference engine changes,
  tools, external actions or production data are required for extraction.

The installed Kuno native route does not support Ollama's `format` parameter.
The model catalog's existing `kuno_backend` marker selects prompt-only JSON on
that route; ordinary Ollama receives a constrained schema. Both paths use the
same strict response parser, size bounds and exact-evidence validator. No Kuno
contract, source, configuration or service is changed.

## Context

Supply a bounded collection of facts as explicitly untrusted quoted context, not
as a replacement system prompt. Human corrections and current conversation win.
Rank with query relevance and recency; do not perform foreground LLM calls to
decide which facts to use. Source records remain inspectable in Memories.

The initial profile holds at most 200 short facts, with 20 history entries per
fact. Extraction handles up to 16 source revisions / 24,000 characters per batch,
with at most 24 candidate facts. Each source contributes at most its first 6,000
characters. These bounds are deliberately independent of the vector-search cache:
this is a compact user profile, not an exhaustive conversation index. Existing
saved-memory and past-chat search tools remain separate.

The default context budget is 6,000 characters, configurable from 1,000 to 16,000.
Injection also fits the remaining conversation window and never exceeds a quarter
of it. Small windows can omit personalization instead of dropping chat content.
The processor waits five idle minutes by default; the setting allows 1-120.

## Validation

Use separate clearly named synthetic development accounts, with an idempotent
seed script. Keep their realistic history for subsequent development. Cover
contradictions, corrections, deleted/edited sources, fictional characters, cloud
exclusion, user isolation, malformed extraction, injection attempts, stale jobs,
private-vault account changes and bounded context. Live inference uses synthetic
content only and records model names and validation limits, never credentials.

### Development Acceptance (2026-11-15)

- 200 server tests passed; four existing opt-in Assistant/Podman integration tests
  remain ignored. Strict server Clippy passed. All 114 frontend tests and 10 native
  host tests passed. Web/PWA and native frontend production builds passed; the
  Android-only transport passed an ARM64 Android cross-check.
- Real extraction and automatic cross-chat recall passed with
  `kuno-qwen3.8-flash-next:iq3_s`, `gemma4:12b-it-qat` and `lfm3:1.7b-q8_0`.
  Six fresh server chats (two synthetic users across three models) recalled the
  correct individual preferences/project with model tools disabled.
- Qwen and Gemma passed the synthetic adversarial extraction batch, excluding
  quoted fiction, third-party preferences, transient secret-like text and an
  injected policy. LFM produced an invalid quotation in that harder batch; the
  validator rejected the complete batch. This is a guarded rejection, not a
  claim that the smaller model extracted that batch correctly.
- Browser checks covered profile creation/edit/history/restore, source navigation,
  storage switching, group rules and an individual exception reverted without
  saving, dirty-state guards, keyboard tabs, and mobile/desktop layouts at
  320x740, 390x844 and 1440x960. Forget confirmation was inspected and canceled;
  actual forgetting/history removal is covered by server and encrypted-store tests.
- Two encrypted device chats demonstrated extraction of a synthetic instrument
  preference and recall in a different private chat. Read-only database checks
  found zero server facts/messages containing that private test preference.
  Offline device-profile inspection and recovery after server restart passed.
- Native request tests reject a changed server before transmitting private input;
  server device-extraction tests reject a changed signed-in user before inference.
  Settings/source revision races, human corrections, deletion, cloud/character
  exclusion, edited-revision model provenance and account isolation are covered.

The retained fixture accounts are `demo-profile-alex` and `demo-profile-riley`.
They have 22 and 11 server conversations respectively, plus Alex's two encrypted
device conversations in the testing browser. Their current server profiles hold
11 and 5 facts; the device profile holds two. The seed/evaluation harness is
`apps/vashti/scripts/personalization-dev.mjs`, with modes `seed`, `evaluate`,
`models` and `safety`. Supply `VASHTI_DEV_ADMIN_PASSWORD` and
`VASHTI_DEMO_PASSWORD` through the environment; the harness refuses non-local
servers and mismatched development databases. Do not commit credentials or
private fixture payloads. The real admin account has not opted in or been processed.

Installed Kuno Qwen is the recommended tested processor for this setup. Exact
quotes verify provenance, not truth; extraction can still miss a correction,
misunderstand a statement or assign a poor topic. Human review/locking and strict
rejection remain necessary. This pass does not qualify Kuno engine erasure,
cloud behavior, physical Android execution, an APK release, or deployment.

## Research

This is a Vashti design, not a claim about proprietary ChatGPT internals.
The [Generative Agents paper](https://arxiv.org/abs/2304.03442) motivates
source-linked synthesis and relevance/recency retrieval. Its reported memory
embellishments are a reason to require explicit evidence here. OpenAI's
[agent safety guidance](https://developers.openai.com/api/docs/guides/agent-builder-safety)
motivates structured boundaries and keeping untrusted text out of privileged
instructions. Neither structured output nor exact quotation proves semantic
truth; the user must be able to inspect, correct and forget the result.
