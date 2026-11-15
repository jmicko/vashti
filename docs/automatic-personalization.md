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
- Forgetting leaves a content-free topic/source suppression marker, so old chats
  cannot immediately recreate a forgotten fact. Forget-all suppresses existing
  sources, while permitting newly written conversations. History is removed for
  forgotten facts; it must not become a second hidden memory store.
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

## Validation

Use separate clearly named synthetic development accounts, with an idempotent
seed script. Keep their realistic history for subsequent development. Cover
contradictions, corrections, deleted/edited sources, fictional characters, cloud
exclusion, user isolation, malformed extraction, injection attempts, stale jobs,
private-vault account changes and bounded context. Live inference uses synthetic
content only and records model names and validation limits, never credentials.

## Research

This is a Vashti design, not a claim about proprietary ChatGPT internals.
The [Generative Agents paper](https://arxiv.org/abs/2304.03442) motivates
source-linked synthesis and relevance/recency retrieval. Its reported memory
embellishments are a reason to require explicit evidence here. OpenAI's
[agent safety guidance](https://developers.openai.com/api/docs/guides/agent-builder-safety)
motivates structured boundaries and keeping untrusted text out of privileged
instructions. Neither structured output nor exact quotation proves semantic
truth; the user must be able to inspect, correct and forget the result.
