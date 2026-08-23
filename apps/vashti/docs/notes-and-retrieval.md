# Notes and Retrieval

## 1. Goal

Vashti provides a personal Markdown notes workspace and an optional Notes tool
for models. Notes remain useful when tools or semantic search are unavailable.
Model access is explicit, inspectable, scoped, versioned, and reversible.

The notes workspace is not a memory system. Future memories may reuse the
retrieval infrastructure, but memories and notes have separate ownership,
retention, permission, and presentation rules.

## 2. Product Rules

### 2.1 Notes workspace

Authenticated users can:

* create, read, edit, pin, tag, search, trash, restore, and permanently delete
  their own notes
* write Markdown and switch between editing and rendered preview
* use a compact formatting toolbar with additional actions in an overflow menu
* inspect immutable version history, including the actor and source of a change
* restore an older version by creating a new current version
* distinguish saving, saved, offline, and failed-save states

The desktop workspace uses a note list beside the editor. Narrow layouts show
the list and editor as separate views. The primary navigation exposes Notes as
a top-level destination while the Vashti mark continues to navigate home.

Initial organization includes tags, pinning, search, and sorting. Nested
folders, collaborative editing, backlinks, note sharing, and note attachments
are outside the first release.

### 2.2 Storage boundaries

Server notes:

* are private to their owner
* are stored in SQLite
* sync across that user's devices
* may be used by standard server-backed chats

Device notes:

* are stored in the encrypted IndexedDB private-storage boundary
* are available only on that device
* may be explicitly attached to private-local messages
* may be used by enabled Notes tools in private-local chats through the
  authenticated client-tool bridge
* must not leave names, IDs, content, embeddings, or usage metadata in server
  persistence

Standard chats must never use device notes. Private-local tool calls execute in
the authenticated browser session, enforce the same global, per-note, and
per-model permissions as server notes, and return bounded results to the active
generation without persisting note data on the server. Section 8 specifies the
bridge boundary and replay protection.

## 3. Versioning, Trash, and Concurrency

Autosave preserves work frequently without turning every typing pause into a
visible history entry. Human title/content changes made during one editing
session are coalesced into a single checkpoint. Switching or closing the note
ends that session, and ten minutes without an edit starts a new session when
typing resumes. Model changes always create their own version.

A session checkpoint can be replaced only while it remains private to that
note's history. Once a version is pinned to a conversation or attached to a
message, it is immutable and the next human save creates a new checkpoint.

Each retained history version records:

* immutable title and Markdown content
* monotonically increasing version number
* actor type: `human` or `model`
* actor user ID
* model identity and display-name snapshot when applicable
* source chat, message, and tool-call IDs when applicable
* creation timestamp

Restoring history creates a new version copied from the selected version. It
does not move the current pointer backward or delete later history.

Deleting a note moves it to trash. Models can only move notes to trash. Only
the human owner can permanently delete a trashed note. Restoring from trash
keeps all versions.

Updates include the expected current version number and, for editor saves, the
expected version ID. A mismatch returns a conflict instead of overwriting a
newer human or model edit, including when a coalesced checkpoint still has the
same version number.

## 4. Model Access

The user-facing controls follow four clear layers:

1. an administrator makes the Notes tool available on the Vashti server
2. each person chooses what models may do in **Settings → Notes**
3. the Notes toggle turns the tool on or off for an individual chat
4. an individual note may reduce its access level or limit which models can use it

The server continues to enforce global and tag-based policy underneath these
controls. No personal or per-chat setting can bypass administrator policy.

The chat tool menu always lists the Notes tool family. If the selected model or
current permissions leave it with no usable operations, the enabled toggle
shows that limitation instead of making the tool disappear.

Per-note model access levels are presented as:

* **No access** (`none`): omitted from model search and inaccessible through tools
* **Read only** (`read`): searchable and readable
* **Read and edit** (`edit`): read access plus versioned edits
* **Full access** (`manage`): edit access plus moving the note to Trash

Note creation is a library-wide permission because no note exists yet to carry
a per-note rule. Untouched accounts allow every Notes operation, and new notes
default to **Full access** for **Every model**. These defaults apply only to notes
created from then on. Existing notes keep their per-note access, so notes set to
**No access** remain unavailable. The upgrade initializes library-wide access for
accounts that never saved a Notes preference, while preserving explicitly saved
personal settings.

A model-created note always starts with **Full access**. If the user's configured
model list is restricted, the creating model is added to that note's scope so it
can read and revise its own note in later turns. Personal operation permissions
can still block those actions, and the owner can reduce access at any time.

Model selection is either **Every model** or an explicit list. Custom models
appear separately in that list, even when they use the same base model.

Human owners retain full control regardless of AI permissions.

## 5. Model Tools

The UI exposes one persistent Notes tool-family toggle. The model receives
narrow functions according to the effective permissions:

* `search_notes(query, limit)`
* `read_note(note_id)`
* `create_note(title, content, tags)`
* `update_note(note_id, expected_version, title, content)`
* `trash_note(note_id, expected_version)`

There is no permanent-delete function. Search never returns notes that the
current model cannot read. Mutation results identify the created version and
are rendered through the normal visible tool-use cards.

Tool responses are bounded. Search returns IDs, titles, matching excerpts, and
current version numbers rather than complete note libraries. Models call
`read_note` when full content is needed.

## 6. Explicit Chat Context

Typing `/notes` in the composer opens a searchable note picker. Selecting a
note adds a removable note chip rather than literal Markdown to the draft. On
send, Vashti snapshots the exact selected note version and records it with the
message.

An explicitly attached note applies to that message only. Conversation
settings may separately contain pinned notes, which apply to future messages
until removed or explicitly updated. Note tool permission and prompt inclusion
remain separate concepts: allowing a model to read a note never injects it
automatically.

Model settings may link to a Notes view filtered for the selected model. The
picker prioritizes notes scoped to and recently used with the active model.

## 7. Search and Embeddings

SQLite remains the authoritative store. Search has two independent paths:

* FTS5 keyword search over current note titles, tags, and Markdown content
* optional semantic search over note chunks and query embeddings

Ranked results are combined with reciprocal-rank fusion so literal matches
remain strong while semantic matches can recover different wording. Keyword
search remains available while semantic indexing is disabled, unavailable, or
catching up.

Markdown-aware chunking preserves headings, paragraphs, lists, and fenced code
where practical. Only current note versions are indexed for ordinary search.
History remains directly browsable but does not pollute results.

Embedding records include:

* source type, note ID, and note-version ID
* chunk index and chunk text hash
* embedding model identity and vector dimensions
* chunker version
* vector bytes

Notes become keyword-searchable immediately. Embeddings are generated in the
background. Startup and periodic repair jobs enqueue missing or stale chunks.
Changing the configured embedding model or chunker version schedules a rebuild.

The initial semantic index stores vectors in SQLite and computes exact cosine
similarity in Rust, with a bounded per-user memory cache. The index is derived
and rebuildable. A future approximate index such as embedded LanceDB may replace
the exact implementation behind a retrieval interface if corpus size requires
it.

The admin selects an enabled Ollama backend and embedding model. Semantic
search is optional and must fail back to FTS5 without making notes unavailable.

## 8. Device Tool Bridge

The server cannot directly access encrypted IndexedDB. Private generation uses
a generic client-tool protocol for operations that must execute on the active
device:

1. the server emits a client tool-call event during private generation
2. the frontend validates the chat, per-chat tool selection, local defaults,
   per-note access level, and exact model scope
3. the frontend executes the operation against encrypted IndexedDB
4. the frontend stores an encrypted idempotency receipt and returns a bounded
   result associated with the tool-call ID
5. generation resumes with that result

The server broker binds each pending call to the authenticated user, session,
generation, opaque call ID, and one-time resume token. Results from another
user, session, or generation are rejected. Completed calls remain briefly
idempotent so a client retry cannot run a model round twice. Calls time out and
generation resumes with a bounded error result rather than waiting forever.

The bridge is transport only and contains no Notes-specific storage logic.
Known client tool schemas are allowlisted on the server and checked against the
same global and tag-based tool-family permissions as standard chats. The
browser owns the handler registry and encrypted local permission state.

Device note identities, libraries, permissions, and receipts are never written
to server persistence. Tool arguments and results necessarily pass transiently
through Vashti and the selected Ollama backend while that generation is active.
This is the same disclosure boundary as explicitly attaching a device note to
a private prompt; it is not device-local inference.

## 9. Server Data Model

Recommended server tables:

* `notes`: identity, owner, current version, pin state, AI access, trash and
  timestamps
* `note_versions`: immutable title/content snapshots and actor/source metadata
* `note_tags` and `note_tag_links`: owner-scoped organization
* `note_model_scopes`: exact base-model or custom-model identities
* `note_message_attachments`: exact versions explicitly attached to messages
* `chat_pinned_notes`: exact current selections for future messages
* `note_search_chunks`: current-version chunks and embedding metadata/vectors
* an external-content FTS5 table kept consistent transactionally with current
  note content

All IDs use UUID text values. Ownership checks occur in every query; accepting
a note ID from a tool call never substitutes for an ownership check.

## 10. Delivery Slices

1. Server schema, CRUD, version history, trash, concurrency, and keyword search.
2. Responsive Markdown notes workspace and compact navigation.
3. AI access controls, model scopes, and safe server Notes tools.
4. `/notes`, message-version attachments, and conversation-pinned notes.
5. Optional Ollama embeddings and hybrid retrieval.
6. Encrypted device-note workspace and explicit private-message attachment.
7. Generic authenticated client-tool bridge and autonomous device-note tools.

Each slice must include ownership and permission tests, migration tests, Rust
tests, TypeScript checks, and production web builds before it is committed.
