# Vashti Technical Design

## 1. Purpose

This document translates the MVP spec into an implementation-oriented design.

It defines:

* database tables and relationships
* API routes and request/response shapes
* startup/account creation flow
* authentication/session behavior
* standard vs private-local chat behavior
* frontend screens and their data dependencies
* initial repo/module direction

This is the document the implementation should follow unless a later engineering issue forces a conscious change.

---

## 2. Core Architecture

### 2.1 Backend responsibilities

The Rust backend is responsible for:

* serving embedded frontend assets
* serving the app shell
* providing JSON API endpoints under `/api`
* authenticating users with cookie-based sessions
* storing persistent app data in SQLite
* managing standard chats and server-backed attachments
* managing server-stored model personas and immutable persona versions
* proxying generation requests to Ollama backends
* streaming generation output back to the client
* running DB migrations on startup
* creating runtime directories on first run

### 2.2 Frontend responsibilities

The frontend is responsible for:

* login flow
* SPA navigation and app shell
* chat list and chat view rendering
* local UI state and optimistic interaction where appropriate
* IndexedDB persistence for private-local chats
* IndexedDB persistence for private-local model personas
* upload selection UX
* PWA installation and asset caching

#### PWA asset lifecycle

The production Vite build generates the web manifest and a Workbox service
worker. The worker precaches the complete revisioned frontend shell, including
lazy-loaded chat and settings chunks, but the browser only parses those chunks
when their route is opened. The worker never caches `/api` responses or user
data.
Private-local and optional structured-data caches remain explicit IndexedDB
features rather than service-worker runtime caches.

The app registers the worker without delaying the initial render. It checks for
updates after registration, hourly while open, and when a visible tab has not
checked recently. A waiting worker produces a reload prompt; Vashti does not
silently replace the running frontend while the user may be editing or
generating content.

HTTP cache policy supports the same lifecycle:

* `index.html`, `sw.js`, the web manifest, and stable-name public assets use
  `Cache-Control: no-cache`
* revisioned Vite assets and the fingerprinted Workbox runtime use a one-year
  immutable cache policy
* API middleware continues to set `Cache-Control: no-store`
* the Rust server compresses static text assets with Brotli or gzip when the
  client supports it; JSON and streaming NDJSON responses remain uncompressed

Service-worker registration is disabled in Vite development. Installation and
offline shell caching require a browser secure context (HTTPS or localhost),
while normal LAN HTTP access continues without those PWA features.

### 2.3 Chat storage split

There are two storage modes.

#### Standard chats

* authoritative on the server
* persisted in SQLite
* synced across the user’s devices through normal API reads

#### Private-local chats

* authoritative in the browser on that device
* stored in IndexedDB
* no server persistence of prompt/response content
* server only receives transient content for live generation requests

### 2.4 Model personas

Vashti treats personas as reusable model profiles, not as separate provider types.

A persona resolves to:

* base Ollama backend
* base Ollama model name
* immutable persona version
* system prompt
* display name and optional avatar
* future tool policy metadata

Storage modes:

* server personas are stored in SQLite and may be private-to-owner or public
* private personas are stored only in IndexedDB and are usable only in private-local chats
* standard chats may use base Ollama models or server personas, but never private personas
* private-local chats should use local persona records; selecting a public server persona for private use should create or use a local copy so private use does not create server-side persona membership

Generation uses the resolved base backend/model and prepends or includes the persona version's system prompt in the prompt sent to Ollama.

### 2.5 Context blocks

Context blocks are reusable, versioned prompt fragments that are independent of
personas and base models. Users organize blocks into categories with either
`single` or `multiple` selection behavior, then attach an ordered set of block
versions to a conversation.

Standard-chat blocks are owner-scoped SQLite records. Private-local blocks are
encrypted IndexedDB records and never become server metadata. In both modes,
the conversation pins immutable versions and every generated assistant message
records the versions used for that response.

Prompt assembly is deterministic:

1. resolve the chat's system-prompt override or selected persona prompt
2. append selected context blocks in the chat's stored order
3. label each fragment as `[Context: <block name>]`
4. reject the request if the compiled prompt exceeds the configured hard limit

### 2.6 Tools

Tools are optional backend capabilities that can be exposed to Ollama models through Ollama's function-calling API.

MVP tool behavior:

* tools are disabled by default
* admins enable tools globally from settings
* provider API keys are entered in the admin Tools settings page
* API keys are write-only in the UI; API responses only report whether a key is configured
* admins can view and edit the prompt text sent to Ollama for tool behavior and individual tool schemas
* editable prompts should have reset-to-default controls in the UI
* the default tool behavior prompt includes the current UTC date so models do not treat fresh search results as future-dated information
* web search may use Brave Search or Ollama's hosted web search API
* web fetch may use Ollama's hosted web fetch API or Vashti's guarded direct fetcher
* tool schemas are sent only when the selected Ollama model reports the `tools` capability
* tool calls and tool results are relayed through the server during generation; final chat content remains normal assistant output
* direct page fetch must block localhost, private networks, link-local addresses, multicast addresses, and embedded URL credentials

---

## 3. Database Design

## 3.1 ID strategy

Use UUIDs as text IDs for MVP.

Reasoning:

* simple to generate in Rust and frontend
* avoids coordination issues
* easier for attachments and private-local client-generated IDs later

Server-managed tables should still use UUID text IDs even if integer rowids exist internally.

---

## 3.2 Tables

### 3.2.1 `users`

Purpose:

* stores local account records

Columns:

* `id` TEXT PRIMARY KEY
* `username` TEXT NOT NULL UNIQUE
* `email` TEXT UNIQUE
* `password_hash` TEXT NOT NULL
* `role` TEXT NOT NULL DEFAULT 'user'
  Allowed values: `admin`, `user`
* `is_disabled` INTEGER NOT NULL DEFAULT 0
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL
* `last_login_at` INTEGER

Notes:

* first created account becomes admin
* `email` may be nullable if username-only creation is allowed

### 3.2.2 `sessions`

Purpose:

* server-side session storage for cookie auth

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `created_at` INTEGER NOT NULL
* `expires_at` INTEGER NOT NULL
* `last_seen_at` INTEGER NOT NULL
* `ip_address` TEXT
* `user_agent` TEXT

Notes:

* cookie contains only opaque session ID
* expired sessions should be ignored and can be cleaned periodically

### 3.2.3 `ollama_backends`

Purpose:

* admin-configured Ollama servers

Columns:

* `id` TEXT PRIMARY KEY
* `name` TEXT NOT NULL UNIQUE
* `base_url` TEXT NOT NULL
* `is_enabled` INTEGER NOT NULL DEFAULT 1
* `is_localhost_detected` INTEGER NOT NULL DEFAULT 0
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL
* `last_healthcheck_at` INTEGER
* `last_health_status` TEXT
* `last_error` TEXT

Notes:

* `name` is human-readable, like `mac-mini`
* `base_url` should be normalized without trailing slash if possible

### 3.2.4 `user_settings`

Purpose:

* stores per-user preferences

Columns:

* `user_id` TEXT PRIMARY KEY REFERENCES `users`(`id`) ON DELETE CASCADE
* `default_backend_id` TEXT REFERENCES `ollama_backends`(`id`) ON DELETE SET NULL
* `default_model_name` TEXT
* `theme` TEXT
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Notes:

* theme can remain minimal in MVP

### 3.2.5 `app_settings`

Purpose:

* singleton-style app configuration

Columns:

* `id` INTEGER PRIMARY KEY CHECK (`id` = 1)
* `allow_signup` INTEGER NOT NULL DEFAULT 1
* `signup_limit` INTEGER NOT NULL DEFAULT 25
* `signup_count` INTEGER NOT NULL DEFAULT 0
* `max_upload_bytes` INTEGER NOT NULL DEFAULT 10485760
* `request_timeout_ms` INTEGER NOT NULL DEFAULT 120000
* `network_mode` TEXT NOT NULL DEFAULT `lan_http`
  Allowed values: `lan_http`, `public_https_proxy`
* `public_base_url` TEXT
* `trust_proxy_headers` INTEGER NOT NULL DEFAULT 0
* `network_recovery_notice` TEXT
* `tools_enabled` INTEGER NOT NULL DEFAULT 0
* `ollama_web_search_enabled` INTEGER NOT NULL DEFAULT 0
* `ollama_web_fetch_enabled` INTEGER NOT NULL DEFAULT 0
* `ollama_api_key` TEXT
* `brave_search_enabled` INTEGER NOT NULL DEFAULT 0
* `brave_search_api_key` TEXT
* `direct_web_fetch_enabled` INTEGER NOT NULL DEFAULT 0
* `tool_system_prompt` TEXT
* `web_search_tool_prompt` TEXT
* `web_fetch_tool_prompt` TEXT
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Notes:

* one row only
* admin edits this through settings UI
* `allow_signup` and `signup_limit` apply only when an enabled admin already exists
* adminless account creation remains allowed so a system with no admins can recover
* `signup_count` tracks successful public self-registrations after an admin exists; when it reaches `signup_limit`, the app sets `allow_signup = 0`
* `network_mode = lan_http` is the default easy LAN mode and leaves session cookies usable over plain HTTP
* `network_mode = public_https_proxy` is for nginx, Caddy, Cloudflare Tunnel, or similar HTTPS reverse-proxy deployments and enables Secure session cookies
* `trust_proxy_headers` is only meaningful in public reverse-proxy mode and must not be trusted when Vashti is directly exposed to arbitrary clients
* `public_base_url` is used for future share links and reverse-proxy config generation; it does not prove HTTPS is working
* tool provider API keys are stored in `app_settings` and should never be returned to the frontend after save
* `direct_web_fetch_enabled` exposes no key, but still must be treated as a network-risk setting because it lets the server request public URLs
* null or blank tool prompt fields fall back to the built-in defaults
* `{current_date}` inside tool prompts is replaced by the server's current UTC date before sending prompts to Ollama

### 3.2.6 `chats`

Purpose:

* server-backed standard chats only
* owns chat-level defaults and the active root message

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `default_backend_id` TEXT NOT NULL REFERENCES `ollama_backends`(`id`) ON DELETE RESTRICT
* `default_model_name` TEXT NOT NULL
* `persona_id` TEXT REFERENCES `personas`(`id`) ON DELETE SET NULL
* `persona_version_id` TEXT REFERENCES `persona_versions`(`id`) ON DELETE SET NULL
* `title` TEXT NOT NULL
* `chat_mode` TEXT NOT NULL DEFAULT 'standard'
  Allowed values for MVP table usage: `standard`
* `active_root_message_id` TEXT
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL
* `last_message_at` INTEGER NOT NULL
* `archived_at` INTEGER

Notes:

* private-local chats are not stored here in MVP
* `title` can be user-set or auto-generated later
* `active_root_message_id` points at the selected first message in the current visible branch
* `persona_version_id`, when present, binds the chat to an immutable server persona version
* chats keep using the bound persona version until the user explicitly changes or updates it
* this FK may be enforced after `chat_messages` exists, or left application-enforced to avoid circular migration friction

### 3.2.7 `chat_messages`

Purpose:

* logical message nodes in a branchable message tree
* stores tree position, current active revision, deletion state, and generation metadata

Columns:

* `id` TEXT PRIMARY KEY
* `chat_id` TEXT NOT NULL REFERENCES `chats`(`id`) ON DELETE CASCADE
* `parent_message_id` TEXT REFERENCES `chat_messages`(`id`) ON DELETE SET NULL
* `active_child_message_id` TEXT REFERENCES `chat_messages`(`id`) ON DELETE SET NULL
* `active_revision_id` TEXT
* `role` TEXT NOT NULL
  Allowed values: `system`, `user`, `assistant`
* `status` TEXT NOT NULL DEFAULT 'complete'
  Allowed values: `complete`, `streaming`, `stopped`, `error`
* `is_deleted` INTEGER NOT NULL DEFAULT 0
* `backend_id` TEXT REFERENCES `ollama_backends`(`id`) ON DELETE SET NULL
* `model_name` TEXT
* `persona_id` TEXT REFERENCES `personas`(`id`) ON DELETE SET NULL
* `persona_version_id` TEXT REFERENCES `persona_versions`(`id`) ON DELETE SET NULL
* `persona_name_snapshot` TEXT
* `think_mode` TEXT
  Examples: `off`, `on`, `low`, `medium`, `high`
* `done_reason` TEXT
* `error_text` TEXT
* `started_at` INTEGER
* `completed_at` INTEGER
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Notes:

* this is a single-parent tree, not a multi-parent graph
* branches are represented as sibling messages sharing the same `parent_message_id`
* active branch traversal starts at `chats.active_root_message_id`, then follows each node's `active_child_message_id`
* `active_revision_id` points to the visible text/thinking revision for the logical message
* user message edit + save creates a new revision on the same logical message
* user message edit + send creates a sibling message under the same parent, then generates a new assistant child
* assistant regenerate creates a sibling assistant message under the same parent
* deleted messages stay in the tree with `is_deleted = 1`; prompt construction skips deleted messages
* when a message is deleted, revision content should be scrubbed so deleted text is not retained
* generation metadata applies primarily to assistant messages, but nullable columns keep the table simple
* assistant messages should store the resolved persona/model snapshot used at generation time so old chats remain understandable if a persona changes or is deleted

### 3.2.8 `chat_message_revisions`

Purpose:

* immutable text snapshots for edits and regenerations
* stores assistant thinking separately from final visible content

Columns:

* `id` TEXT PRIMARY KEY
* `message_id` TEXT NOT NULL REFERENCES `chat_messages`(`id`) ON DELETE CASCADE
* `content_text` TEXT NOT NULL DEFAULT ''
* `thinking_text` TEXT NOT NULL DEFAULT ''
* `source` TEXT NOT NULL
  Allowed values: `original`, `edit`, `regeneration`
* `created_at` INTEGER NOT NULL

Notes:

* copy-to-clipboard uses `content_text` only, not `thinking_text`
* Ollama thinking-capable models may stream thinking separately from final content
* saving an edit appends a revision and updates `chat_messages.active_revision_id`
* revisions make prior edits available without duplicating child subtrees
* deleting a message should blank `content_text` and `thinking_text` for its revisions

### 3.2.9 `attachments`

Purpose:

* server-backed uploaded files for standard chats
* attach files/images to message revisions for prompt-authoritative attachment state
* pending composer uploads may exist briefly with no message/revision until generation claims them

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `chat_id` TEXT NOT NULL REFERENCES `chats`(`id`) ON DELETE CASCADE
* `message_id` TEXT REFERENCES `chat_messages`(`id`) ON DELETE SET NULL
* `revision_id` TEXT REFERENCES `chat_message_revisions`(`id`) ON DELETE CASCADE
* `storage_path` TEXT NOT NULL UNIQUE
* `original_filename` TEXT NOT NULL
* `mime_type` TEXT NOT NULL
* `size_bytes` INTEGER NOT NULL
* `attachment_kind` TEXT NOT NULL
  Allowed values: `image`, `text`
* `created_at` INTEGER NOT NULL

Notes:

* standard-chat attachments only in server DB
* private-local attachments are not persisted server-side in MVP
* attachments are intentionally deferred from the first text-chat slice
* `revision_id` is the authoritative link for prompt content
* `message_id` is retained for ownership checks, listing, and cleanup convenience
* text attachments are accepted only when uploaded bytes decode as UTF-8 text without embedded NUL bytes
* image attachments are stored and sent to Ollama as base64 `images` entries
* PDFs are intentionally unsupported in the first attachment pass; text PDFs still require a parser, and image-heavy PDFs imply OCR/document-processing scope
* edit + save can replace text and attachments together by creating a new revision with its own attachment set
* deleting a message should delete or detach attachment records for all of its revisions according to the later file-deletion policy

### 3.2.10 `personas`

Purpose:

* server-stored custom model/persona identities
* owner and visibility lifecycle
* points to the current immutable version

Columns:

* `id` TEXT PRIMARY KEY
* `owner_user_id` TEXT REFERENCES `users`(`id`) ON DELETE SET NULL
* `current_version_id` TEXT
* `visibility` TEXT NOT NULL DEFAULT 'private'
  Allowed values: `private`, `public`
* `lifecycle_state` TEXT NOT NULL DEFAULT 'active'
  Allowed values: `active`, `disowned`, `deleted`
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Notes:

* a server persona with `visibility = private` is stored on the server but visible only to its owner
* a public persona is visible to all users and can be used without admin approval
* deleting a public persona with other members should disown it for the requesting user rather than breaking existing usage
* if the creator disowns a public persona, existing versions remain usable for remaining members, but no new versions can be published until ownership transfer exists
* when the last member disowns a public persona, the server may delete the persona and its versions
* private-local personas are not stored in this table

### 3.2.11 `persona_versions`

Purpose:

* immutable snapshots of persona behavior and display metadata
* protects existing chats from later persona edits

Columns:

* `id` TEXT PRIMARY KEY
* `persona_id` TEXT NOT NULL REFERENCES `personas`(`id`) ON DELETE CASCADE
* `version_number` INTEGER NOT NULL
* `display_name` TEXT NOT NULL
* `avatar_asset_id` TEXT REFERENCES `persona_avatar_assets`(`id`) ON DELETE SET NULL
* `avatar_crop_x` REAL NOT NULL DEFAULT 50
* `avatar_crop_y` REAL NOT NULL DEFAULT 50
* `avatar_crop_size` REAL NOT NULL DEFAULT 100
* `base_backend_id` TEXT NOT NULL REFERENCES `ollama_backends`(`id`) ON DELETE RESTRICT
* `base_model_name` TEXT NOT NULL
* `system_prompt` TEXT NOT NULL DEFAULT ''
* `tool_policy_json` TEXT
* `created_by_user_id` TEXT REFERENCES `users`(`id`) ON DELETE SET NULL
* `created_at` INTEGER NOT NULL

Notes:

* behavioral edits and image replacement create a new row and update `personas.current_version_id`
* crop-only edits update the current version's presentation rectangle without recompressing the original or creating a new version
* public persona prompts are visible to users who can use that persona
* `tool_policy_json` is reserved for future tools and should be opaque to MVP generation
* chat generation should use the version bound to the chat, not whatever version is currently latest
* copying a persona copies only the selected version's current fields into a new persona/version 1
* copied personas receive their own avatar asset owned by the copying user

### 3.2.12 `persona_avatar_assets`

Purpose:

* stores metadata for immutable original custom-model profile image files
* keeps image bytes outside SQLite and separate from chat attachments/messages

Columns:

* `id` TEXT PRIMARY KEY
* `owner_user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `original_filename` TEXT NOT NULL
* `storage_path` TEXT NOT NULL UNIQUE
* `mime_type` TEXT NOT NULL
  Allowed values: `image/jpeg`, `image/png`, `image/gif`
* `size_bytes` INTEGER NOT NULL
* `created_at` INTEGER NOT NULL

JPEG, PNG, and GIF bytes live unchanged under
`data/persona-avatars/<owner-user-id>/<asset-id>`. Other browser-readable
formats are converted once to lossless PNG before upload.
Vashti does not convert the original to WebP or repeatedly recompress it.

### 3.2.13 `persona_members`

Purpose:

* tracks users who have adopted or used public personas
* supports disown/delete lifecycle without breaking other users

Columns:

* `persona_id` TEXT NOT NULL REFERENCES `personas`(`id`) ON DELETE CASCADE
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `membership_role` TEXT NOT NULL DEFAULT 'member'
  Allowed values for MVP: `creator`, `member`
* `created_at` INTEGER NOT NULL
* PRIMARY KEY (`persona_id`, `user_id`)

Notes:

* the creator is inserted as a member when the persona becomes public
* a user is inserted as a member when they start a chat with a public persona
* removing a public persona from the user's picker removes membership
* deleting the final membership allows server cleanup

### 3.2.14 `context_categories`

Owner-scoped organizational categories for server context blocks.

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `name` TEXT NOT NULL
* `selection_mode` TEXT NOT NULL DEFAULT `single`
  Allowed values: `single`, `multiple`
* `sort_order` INTEGER NOT NULL DEFAULT 0
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Category deletion sets matching block category IDs to null; it does not delete
the blocks.

### 3.2.15 `context_blocks`

Stable identities for owner-scoped reusable prompt fragments.

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `category_id` TEXT REFERENCES `context_categories`(`id`) ON DELETE SET NULL
* `current_version_id` TEXT NOT NULL
* `sort_order` INTEGER NOT NULL DEFAULT 0
* `deleted_at` INTEGER
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Deleting a block is a soft delete so versions already pinned to chats remain
available.

### 3.2.16 `context_block_versions`

Immutable context-block content snapshots.

Columns:

* `id` TEXT PRIMARY KEY
* `block_id` TEXT NOT NULL REFERENCES `context_blocks`(`id`) ON DELETE CASCADE
* `version_number` INTEGER NOT NULL
* `name` TEXT NOT NULL
* `content` TEXT NOT NULL
* `created_at` INTEGER NOT NULL
* UNIQUE (`block_id`, `version_number`)

### 3.2.17 chat context selections

`chat_context_blocks` stores the ordered block versions currently selected for
a standard chat. `chat_message_context_blocks` stores the ordered immutable
snapshot used for each generated assistant message. Both tables reference
`context_block_versions`; chat/message deletion cascades, while referenced
versions are protected from deletion.

### 3.2.13 `model_availability`

Global admin on/off switch for base Ollama models returned by configured backends.

Columns:

* `backend_id` TEXT NOT NULL REFERENCES `ollama_backends`(`id`) ON DELETE CASCADE
* `model_name` TEXT NOT NULL
* `is_enabled` INTEGER NOT NULL DEFAULT 1
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL
* PRIMARY KEY (`backend_id`, `model_name`)

Notes:

* model rows are created when Vashti first sees a model from an Ollama backend
* newly discovered models receive the current default model permission tags
* admins can disable individual models or bulk enable/disable all currently returned models for a backend
* `GET /api/models` returns only enabled models whose permission tags intersect the current user's effective tags and are visible in that user's picker preferences
* generation endpoints must check this table server-side; frontend filtering is not sufficient
* disabled models are unavailable to everyone, including admins, until re-enabled; admins may still edit their tags while disabled

### 3.2.14 `user_model_preferences`

Per-user visibility preferences for the model picker.

Columns:

* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `backend_id` TEXT NOT NULL
* `model_name` TEXT NOT NULL
* `is_visible` INTEGER NOT NULL DEFAULT 1
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL
* PRIMARY KEY (`user_id`, `backend_id`, `model_name`)
* FOREIGN KEY (`backend_id`, `model_name`) REFERENCES `model_availability`(`backend_id`, `model_name`) ON DELETE CASCADE

Notes:

* missing rows mean visible by default
* users may only set visibility for models they can access through global availability and permission tags
* this table controls picker clutter only; generation endpoints still enforce global availability and permission tags

### 3.2.15 Permission tags

Permission tags are lightweight labels used to grant access to models, tools, and later quotas without a separate group-management system.

Effective user tags:

* every enabled user has `system:everyone`
* every enabled admin also has `system:admin`
* every enabled user has an implicit stable user tag, `user:<user_id>`, displayed as `@username`
* admins may add explicit group tags to users, such as `group:power-users`, displayed as `power-users`

Resource access rule:

```text
resource is enabled
AND resource permission tag list is non-empty
AND resource tags intersect current user's effective tags
```

An empty resource tag list means nobody can use it. Use `everyone` for broad access.

Tag UI behavior:

* UI tag pickers should search existing tags and allow creating a new tag when no result matches
* user tags should display as `@username` but store the stable `user:<user_id>` ID
* unused group tags disappear from suggestions when they are no longer applied anywhere
* default model/tool tags are applied to newly discovered models/tools, but individual resources may be customized afterward
* model default tags and admin-set model tags are separate layers
* removing a default tag from an individual model should leave the tag visible in a muted/removed state so the admin can restore it easily
* applying default model tags to existing models only replaces the default-tag layer; manually added model tags remain intact

Tables:

* `user_permission_tags`
  * `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
  * `tag_id` TEXT NOT NULL
  * explicit user group tags only; implicit system/user tags are computed
* `model_permission_tags`
  * `backend_id` TEXT NOT NULL
  * `model_name` TEXT NOT NULL
  * `tag_id` TEXT NOT NULL
  * references `model_availability` and stores manually added model access tags
* `model_default_permission_tags`
  * `backend_id` TEXT NOT NULL
  * `model_name` TEXT NOT NULL
  * `tag_id` TEXT NOT NULL
  * references `model_availability` and stores the default-tag layer applied to each model
* `tool_permission_state`
  * `tool_id` TEXT PRIMARY KEY
  * records that a tool has been initialized for default-tag application
* `tool_permission_tags`
  * `tool_id` TEXT NOT NULL REFERENCES `tool_permission_state`(`tool_id`) ON DELETE CASCADE
  * `tag_id` TEXT NOT NULL
  * controls tool access
* `app_settings.default_model_permission_tags_json`
* `app_settings.default_tool_permission_tags_json`

Access rules may apply to:

* entire Ollama backends
* individual base model names
* hosted persona count
* public persona count
* tool access

Persona access must perform a double check:

* user must have access to the persona/custom model through ownership, public visibility, or existing public membership
* user must also have access to the persona's underlying base model
* server-private personas remain owner-only

Quota rules:

* admins can set global default hosted-persona and public-persona limits
* admins can override limits by tag
* `unlimited` should be representable, preferably with nullable limit columns
* server-side checks must enforce quotas and access, even when frontend filtering is present

### 3.2.15 `schema_migrations`

If `sqlx` migration tracking is used directly, this table may be tool-managed instead of hand-managed.

Do not create a custom migration table unless needed.

---

## 3.3 Indexes

Recommended indexes:

* `users(username)` unique
* `users(email)` unique when not null
* `sessions(user_id)`
* `sessions(expires_at)`
* `ollama_backends(is_enabled)`
* `chats(user_id, updated_at DESC)`
* `chat_messages(chat_id, created_at)`
* `chat_messages(chat_id, parent_message_id)`
* `chat_messages(active_revision_id)`
* `chat_message_revisions(message_id, created_at)`
* `attachments(chat_id)`
* `attachments(message_id)`
* `attachments(revision_id)`
* `personas(owner_user_id)`
* `personas(visibility, lifecycle_state)`
* `persona_versions(persona_id, version_number)` unique
* `persona_members(user_id)`
* `persona_members(persona_id)`

---

## 4. Authentication and Session Flow

## 4.1 Session model

Use opaque server-side sessions.

Flow:

1. user logs in with username-or-email plus password
2. server verifies password hash
3. server creates a `sessions` row with expiry
4. server sets an HTTP-only cookie containing session ID
5. subsequent requests use cookie automatically

Cookie properties:

* HttpOnly
* SameSite=Lax by default
* Secure when `network_mode = public_https_proxy`
* Path=/

Network modes:

* `lan_http` is the default and supports local/LAN HTTP access such as `http://192.168.x.x:7771`
* `public_https_proxy` is for deployments accessed through an HTTPS reverse proxy and should not be used when accessing Vashti directly over HTTP
* admins should not be shown a raw Secure-cookie toggle; the UI should expose the higher-level network access mode instead

Recovery:

* on startup, if `recover_network.txt` exists in the working directory next to the running app, Vashti resets only DB-backed network settings to LAN defaults
* recovery does not reset users, passwords, chats, uploads, model settings, or `VASHTI_BIND`
* after recovery, Vashti renames the file to `recover_network_success.txt` or a timestamped variant and stores a notice for the admin UI

Request safety:

* mutating API requests (`POST`, `PATCH`, `PUT`, `DELETE`) should reject browser requests whose `Origin` does not match Vashti's own origin or configured public HTTPS base URL
* missing `Origin` is allowed for non-browser clients unless browser fetch metadata says the request is `cross-site`
* trusted proxy headers may only affect request-origin reconstruction when `trust_proxy_headers` is enabled
* login/register, uploads, and generation should have conservative in-memory rate limits to slow brute force attempts and accidental abuse

### 4.2 Password handling

Use a modern password hashing function such as Argon2.

Requirements:

* never store plaintext passwords
* never log passwords
* password reset can be postponed or admin-only initially

### 4.3 Account registration and admin recovery

Unauthenticated users should always see the normal login screen. That screen can include a create-account path, but there is no special first-run screen or `bootstrap_required` state.

When a user account is created through self-registration:

* backend checks whether any enabled admin user exists
* if no enabled admin exists, the new account becomes an enabled `admin`
* if an enabled admin exists, the new account becomes a regular `user` with `is_disabled = 1`
* if an enabled admin exists, self-registration is rejected when `allow_signup = 0`
* if an enabled admin exists, each successful self-registration increments `signup_count`; once `signup_count >= signup_limit`, the app automatically sets `allow_signup = 0`
* disabled self-registered users cannot log in until an admin enables them
* this rule prevents permanent lockout if the only admin account is deleted or disabled

Startup must not create or auto-generate an admin account or password.

---

## 5. Startup Flow

## 5.1 Backend startup sequence

On startup, the server should:

1. resolve data directory path
2. create required directories if missing
3. open SQLite database
4. run migrations
5. ensure singleton app settings row exists
6. optionally attempt localhost Ollama detection if no backend exists
7. start HTTP server

Default bind behavior:

* default to `0.0.0.0:7771` so the app can be reached from the local network
* allow override with `VASHTI_BIND`, for example `VASHTI_BIND=127.0.0.1:7771` when running only behind a local reverse proxy
* external exposure should still be handled deliberately through a reverse proxy, firewall rules, TLS, and normal account/session controls

### 5.2 Localhost Ollama detection

If no `ollama_backends` rows exist:

* try `http://127.0.0.1:11434`
* try `http://localhost:11434`
* if reachable, insert default backend row, ideally named something like `localhost`

This should be conservative and fast.

---

## 6. API Design

All API responses should use JSON except streaming generation endpoints.

Authenticated endpoints require a valid session cookie.

### 6.1 Common response envelope

Do not force every successful response into a wrapper if it adds noise.

Recommended pattern:

* success: return resource-shaped JSON directly
* errors: return JSON like

```json
{
  "error": {
    "code": "not_found",
    "message": "Chat not found"
  }
}
```

---

## 6.2 System endpoints

### `GET /api/version`

Returns the running app name and version from the Rust package metadata.

Response:

```json
{
  "name": "vashti",
  "version": "0.1.0"
}
```

---

## 6.3 Auth endpoints

### `POST /api/auth/register`

Creates a user account from the login screen.

If there is no enabled admin user, the created account is an enabled admin and the response sets a session cookie.
If an enabled admin already exists, the created account is a disabled regular user and the response does not set a session cookie.

Request:

```json
{
  "username": "john",
  "email": "john@example.com",
  "password": "secret"
}
```

Response:

```json
{
  "requires_approval": false,
  "user": {
    "id": "uuid",
    "username": "john",
    "email": "john@example.com",
    "role": "admin",
    "is_disabled": false
  }
}
```

Pending approval response:

```json
{
  "requires_approval": true,
  "user": {
    "id": "uuid",
    "username": "jane",
    "email": "jane@example.com",
    "role": "user",
    "is_disabled": true
  }
}
```

### `POST /api/auth/login`

Request:

```json
{
  "identifier": "john",
  "password": "secret"
}
```

Response:

```json
{
  "user": {
    "id": "uuid",
    "username": "john",
    "email": "john@example.com",
    "role": "admin"
  },
  "private_vault_key": {
    "user_id": "uuid",
    "key_material": "base64-key-material"
  }
}
```

### `POST /api/auth/logout`

Response:

```json
{
  "ok": true
}
```

### `GET /api/auth/session`

Response when logged in:

```json
{
  "is_authenticated": true,
  "can_create_account": true,
  "user": {
    "id": "uuid",
    "username": "john",
    "email": "john@example.com",
    "role": "admin"
  }
}
```

Response when not logged in:

```json
{
  "is_authenticated": false,
  "can_create_account": true,
  "user": null,
  "private_vault_key": null
}
```

`can_create_account` is true when no enabled admin exists, or when an enabled admin exists and public signup is currently enabled and under the configured signup limit. The frontend uses this to show or hide the create-account action on the login screen.

For an authenticated session, `private_vault_key` is the same per-user key
available from `GET /api/private/vault-key`. Returning it with the no-store
session response lets the frontend unlock its encrypted per-user caches without
a second startup request. The dedicated endpoint remains available as a
fallback for older clients and storage recovery.

---

## 6.4 Admin/user endpoints

### `GET /api/admin/users`

Response:

```json
{
  "users": [
    {
      "id": "uuid",
      "username": "john",
      "email": "john@example.com",
      "role": "admin",
      "is_disabled": false,
      "created_at": 1710000000,
      "updated_at": 1710000000,
      "last_login_at": 1710001000
    }
  ]
}
```

### `POST /api/admin/users`

Request:

```json
{
  "username": "friend",
  "email": "friend@example.com",
  "password": "secret",
  "role": "user",
  "is_disabled": false
}
```

Response:

```json
{
  "user": {
    "id": "uuid",
    "username": "friend",
    "email": "friend@example.com",
    "role": "user",
    "is_disabled": false,
    "created_at": 1710000000,
    "updated_at": 1710000000,
    "last_login_at": null
  }
}
```

### `PATCH /api/admin/users/:user_id`

Supports:

* disable/enable
* role change
* optional password reset later

Behavior:

* requires an enabled admin session
* enabling a pending self-registered user makes that account usable
* admins cannot disable or demote their own account through this endpoint

Example request:

```json
{
  "is_disabled": true
}
```

### `DELETE /api/admin/users/:user_id`

Behavior:

* requires an enabled admin session
* deletes the user row and cascades related sessions/settings
* admins cannot delete their own account through this endpoint

Response:

```json
{
  "ok": true
}
```

---

## 6.5 Settings endpoints

### `GET /api/settings`

Admin-oriented app settings.

Response:

```json
{
  "allow_signup": true,
  "signup_limit": 25,
  "signup_count": 3,
  "max_upload_bytes": 10485760,
  "request_timeout_ms": 120000,
  "network_mode": "lan_http",
  "public_base_url": null,
  "trust_proxy_headers": false,
  "network_recovery_notice": null
}
```

### `PATCH /api/settings`

Request:

```json
{
  "allow_signup": false,
  "signup_limit": 25,
  "max_upload_bytes": 20971520,
  "request_timeout_ms": 180000
}
```

### `PATCH /api/settings/network`

Admin-only. Updates advanced network/cookie mode and requires the current admin password plus explicit risk acknowledgment.

Request:

```json
{
  "network_mode": "public_https_proxy",
  "public_base_url": "https://chat.example.com",
  "trust_proxy_headers": true,
  "admin_password": "current-password",
  "acknowledge_risk": true
}
```

Behavior:

* `lan_http` leaves session cookies usable over plain local/LAN HTTP
* `public_https_proxy` sets future session cookies with the Secure flag
* when network settings are saved, the current session cookie is re-issued with the updated cookie flags when possible
* `public_base_url`, if provided, must be an HTTPS URL
* `trust_proxy_headers` should only be enabled when Vashti is reachable by clients only through the trusted reverse proxy

### `POST /api/settings/network-recovery-notice/dismiss`

Admin-only. Clears the one-time network recovery notice after an admin has reviewed it.

### `GET /api/settings/tools`

Admin-only. Returns tool provider settings without exposing stored API key values.

Response:

```json
{
  "tools_enabled": true,
  "ollama_web_search_enabled": true,
  "ollama_web_fetch_enabled": true,
  "ollama_api_key_configured": true,
  "brave_search_enabled": true,
  "brave_search_api_key_configured": true,
  "direct_web_fetch_enabled": false,
  "tool_system_prompt": "Tool behavior guidance...",
  "default_tool_system_prompt": "Tool behavior guidance...",
  "web_search_tool_prompt": "Search the web...",
  "default_web_search_tool_prompt": "Search the web...",
  "web_fetch_tool_prompt": "Fetch a public HTTP or HTTPS page...",
  "default_web_fetch_tool_prompt": "Fetch a public HTTP or HTTPS page..."
}
```

### `PATCH /api/settings/tools`

Admin-only. Updates enabled tool providers and write-only provider API keys.

Request:

```json
{
  "tools_enabled": true,
  "ollama_web_search_enabled": true,
  "ollama_web_fetch_enabled": true,
  "ollama_api_key": "new-or-replacement-key",
  "clear_ollama_api_key": false,
  "brave_search_enabled": true,
  "brave_search_api_key": "new-or-replacement-key",
  "clear_brave_search_api_key": false,
  "direct_web_fetch_enabled": false,
  "tool_system_prompt": "Tool behavior guidance...",
  "web_search_tool_prompt": "Search the web...",
  "web_fetch_tool_prompt": "Fetch a public HTTP or HTTPS page..."
}
```

Behavior:

* omitted booleans leave existing values unchanged
* blank API key fields leave existing keys unchanged unless the matching clear flag is true
* prompt fields are editable admin settings; the UI may reset them by submitting the default prompt text
* prompt fields may use `{current_date}` to inject the current UTC date during generation
* response uses the same shape as `GET /api/settings/tools` and must not return key material

### `GET /api/user-settings`

Response:

```json
{
  "default_backend_id": "uuid",
  "default_model_name": "gemma4",
  "theme": "neon"
}
```

### `PATCH /api/user-settings`

Request:

```json
{
  "default_backend_id": "uuid",
  "default_model_name": "gemma4",
  "theme": "neon"
}
```

Nullable fields may be sent as `null` to clear them.

---

## 6.6 Ollama backend endpoints

### `GET /api/backends`

Response:

```json
{
  "backends": [
    {
      "id": "uuid",
      "name": "rtx-3060",
      "base_url": "http://192.168.1.50:11434",
      "is_enabled": true,
      "last_health_status": "ok",
      "last_error": null
    }
  ]
}
```

### `POST /api/backends`

Request:

```json
{
  "name": "mac-mini",
  "base_url": "http://192.168.1.60:11434"
}
```

### `PATCH /api/backends/:backend_id`

Request example:

```json
{
  "name": "office-mini",
  "base_url": "http://192.168.1.61:11434",
  "is_enabled": true
}
```

### `DELETE /api/backends/:backend_id`

Behavior:

* should fail if backend is still referenced by standard chats, unless later migration behavior is defined

### `POST /api/backends/detect-localhost`

Admin action.

Response:

```json
{
  "detected": [
    {
      "name": "localhost",
      "base_url": "http://127.0.0.1:11434"
    }
  ]
}
```

### `POST /api/backends/scan-local-network`

Admin action.

Scans the server's local IPv4 `/24` network for reachable Ollama servers on port `11434`.

Response shape matches `POST /api/backends/detect-localhost`.

### `GET /api/models`

Returns enabled models grouped by backend.

Model listings are served from Vashti's in-memory Ollama model cache. Vashti refreshes that cache at startup, refreshes it periodically, and lets admins request an immediate hard refresh. Ollama remains the source of truth; the cache only keeps the UI responsive while capability metadata is being refreshed.

Model capability support should prefer Ollama `POST /api/show` metadata when available. A model with a `capabilities` entry of `vision` should be treated as image-capable, and a `thinking` entry should be treated as thinking-capable. `/api/tags` details can still be used as a fallback heuristic for older Ollama responses.

Response:

```json
{
  "is_refreshing": false,
  "cache_updated_at": 1778101847,
  "backends": [
    {
      "backend": {
        "id": "b1",
        "name": "mac-mini"
      },
      "models": [
        {
          "name": "gemma4",
          "supports_images": true,
          "supports_thinking": true,
          "capabilities": ["completion", "vision", "audio", "tools", "thinking"]
        },
        {
          "name": "llama3:70b",
          "supports_images": false,
          "supports_thinking": false,
          "capabilities": ["completion"]
        }
      ]
    }
  ]
}
```

### `GET /api/admin/models`

Admin-only. Returns all cached models currently returned by enabled Ollama backends, including disabled models, grouped by backend.

Response:

```json
{
  "is_refreshing": false,
  "cache_updated_at": 1778101847,
  "available_tags": [
    {
      "id": "system:everyone",
      "label": "everyone",
      "kind": "system"
    }
  ],
  "default_permission_tags": [
    {
      "id": "system:everyone",
      "label": "everyone",
      "kind": "system"
    }
  ],
  "backends": [
    {
      "backend": {
        "id": "b1",
        "name": "mac-mini"
      },
      "models": [
        {
          "name": "gemma4",
          "supports_images": true,
          "supports_thinking": true,
          "capabilities": ["completion", "vision", "thinking"],
          "is_enabled": true
        }
      ]
    }
  ]
}
```

### `POST /api/admin/models/refresh`

Admin-only. Performs a hard refresh of the in-memory model cache by querying all enabled Ollama backends and refreshing model capability metadata. The response shape matches `GET /api/admin/models`.

### `PATCH /api/admin/models`

Admin-only. Enables or disables one model globally.

Request:

```json
{
  "backend_id": "b1",
  "model_name": "gemma4",
  "is_enabled": false
}
```

### `PATCH /api/admin/models/backend`

Admin-only. Bulk enables or disables the currently listed models for a backend.

Request:

```json
{
  "backend_id": "b1",
  "model_names": ["gemma4", "llama3:70b"],
  "is_enabled": false
}
```

### `GET /api/models/status`

Response:

```json
{
  "backends": [
    {
      "backend_id": "b1",
      "name": "mac-mini",
      "status": "ok",
      "latency_ms": 42,
      "error": null
    }
  ]
}
```

---

## 6.7 Persona endpoints

Persona endpoints manage server-stored personas only. Private-local personas live in IndexedDB and are never returned by these endpoints.

### `GET /api/personas`

Returns personas visible to the current user.

Visible personas include:

* personas owned by the current user
* public personas
* public personas where the user has a membership row, even if later lifecycle rules hide them from global discovery

Response:

```json
{
  "personas": [
    {
      "id": "p1",
      "owner_user_id": "u1",
      "owner_username": "dane",
      "visibility": "public",
      "lifecycle_state": "active",
      "current_version": {
        "id": "pv1",
        "version_number": 3,
        "display_name": "Careful Researcher",
        "avatar_asset_id": "pa1",
        "avatar_crop_x": 50,
        "avatar_crop_y": 40,
        "avatar_crop_size": 75,
        "base_backend_id": "b1",
        "base_model_name": "gemma4",
        "system_prompt": "You are careful, concise, and cite uncertainty.",
        "created_at": 1710000000
      },
      "is_owner": false,
      "is_member": true
    }
  ]
}
```

### `POST /api/personas`

Creates a server-stored persona and initial version.

Request:

```json
{
  "visibility": "private",
  "display_name": "Careful Researcher",
  "avatar_asset_id": "pa1",
  "avatar_crop_x": 50,
  "avatar_crop_y": 40,
  "avatar_crop_size": 75,
  "base_backend_id": "b1",
  "base_model_name": "gemma4",
  "system_prompt": "You are careful, concise, and cite uncertainty."
}
```

Behavior:

* creates `personas`
* creates `persona_versions` version `1`
* sets `personas.current_version_id`
* if `visibility = public`, inserts creator into `persona_members`

### `PATCH /api/personas/:persona_id`

Edits persona visibility/lifecycle or creates a new immutable version.

Request:

```json
{
  "visibility": "public",
  "display_name": "Careful Researcher",
  "avatar_asset_id": "pa2",
  "avatar_asset_changed": true,
  "avatar_crop_x": 45,
  "avatar_crop_y": 55,
  "avatar_crop_size": 80,
  "base_backend_id": "b1",
  "base_model_name": "gemma4",
  "system_prompt": "You are careful, concise, and cite uncertainty."
}
```

Behavior:

* creator/owner can publish new versions
* changing behavior or display metadata creates a new `persona_versions` row
* existing chats remain bound to their previous `persona_version_id`
* moving from private server storage to public inserts creator membership
* moving a public persona back to private is not allowed after another user has membership; the owner may disown instead

### `POST /api/personas/:persona_id/copy`

Creates a new persona owned by the current user from a selected visible version.

Request:

```json
{
  "persona_version_id": "pv1",
  "visibility": "private"
}
```

Behavior:

* copies only the selected version's current fields
* does not copy version history
* creates a new persona with version `1`

### `POST /api/personas/:persona_id/disown`

Removes the persona from the current user's available public personas.

Behavior:

* removes the `persona_members` row for the current user
* if this was the final member, the server may delete the persona and versions
* if the creator disowns while other members remain, `owner_user_id` may become `NULL` and `lifecycle_state` may become `disowned`
* existing chats using a bound version must continue to work

### `GET /api/personas/:persona_id/versions`

Returns visible immutable versions for a persona.

Use cases:

* show prompt/version history
* update a chat to the latest version
* copy an older version into a new persona

---

## 6.8 Standard chat endpoints

### `GET /api/chats`

Returns chat summaries only.

Query params:

* `limit` optional
* `cursor` optional later

Response:

```json
{
  "chats": [
    {
      "id": "c1",
      "title": "Roofing questions",
      "default_backend_id": "b1",
      "backend_name": "rtx-3060",
      "default_model_name": "gpt-oss:20b",
      "persona_version_id": null,
      "updated_at": 1710000000,
      "last_message_at": 1710000000,
      "message_count": 18
    }
  ]
}
```

### `POST /api/chats`

Creates a standard chat. New chats may use `New Chat` as a placeholder title until the first assistant response completes.

Request:

```json
{
  "title": "New chat",
  "default_backend_id": "b1",
  "default_model_name": "gemma4",
  "persona_version_id": null
}
```

Response:

```json
{
  "chat": {
    "id": "c1",
    "title": "New chat",
    "default_backend_id": "b1",
    "default_model_name": "gemma4",
    "persona_version_id": null,
    "active_root_message_id": null,
    "created_at": 1710000000,
    "updated_at": 1710000000
  }
}
```

### `GET /api/chats/:chat_id`

Returns chat metadata only.

Response:

```json
{
  "chat": {
    "id": "c1",
    "title": "Roofing questions",
    "default_backend_id": "b1",
    "backend_name": "rtx-3060",
    "default_model_name": "gpt-oss:20b",
    "persona_version_id": "pv1",
    "active_root_message_id": "m1",
    "created_at": 1710000000,
    "updated_at": 1710000500
  }
}
```

### `PATCH /api/chats/:chat_id`

Supports rename and model/backend changes. Chat rename is exposed from each chat row's sidebar overflow menu.

If `persona_version_id` is provided, the backend resolves backend/model from that persona version and binds the chat to it. Standard chats may only use server persona versions visible to the user.

Request example:

```json
{
  "title": "Better title",
  "default_backend_id": "b2",
  "default_model_name": "qwen3.5",
  "persona_version_id": null
}
```

### `DELETE /api/chats/:chat_id`

Deletes a chat after UI confirmation.

Response:

```json
{
  "ok": true
}
```

### Auto-generated chat titles

After the first user message and assistant response complete for a chat still titled `New Chat`, the backend asks the same Ollama model for a concise title.

Rules:

* use the first user message and first assistant response as title context
* instruct the model to return only a short title, ideally 2 to 5 words
* validate the returned title before saving it
* if validation or generation fails, fall back to the first few words of the user's first message
* emit a stream event so the frontend can update the title without a page refresh

### `GET /api/chats/:chat_id/messages`

Returns messages for one standard chat.

Response:

```json
{
  "active_root_message_id": "m1",
  "messages": [
    {
      "id": "m1",
      "parent_message_id": null,
      "active_child_message_id": "m2",
      "active_revision_id": "r1",
      "role": "user",
      "status": "complete",
      "is_deleted": false,
      "created_at": 1710000000,
      "updated_at": 1710000000,
      "active_revision": {
        "id": "r1",
        "content_text": "How do I patch drywall?",
        "thinking_text": "",
        "source": "original",
        "created_at": 1710000000
      },
      "revision_count": 1,
      "attachments": []
    },
    {
      "id": "m2",
      "parent_message_id": "m1",
      "active_child_message_id": null,
      "active_revision_id": "r2",
      "role": "assistant",
      "status": "complete",
      "is_deleted": false,
      "backend_id": "b1",
      "model_name": "gpt-oss:20b",
      "persona_id": "p1",
      "persona_version_id": "pv1",
      "persona_name_snapshot": "Careful Researcher",
      "think_mode": "medium",
      "done_reason": "stop",
      "created_at": 1710000001,
      "updated_at": 1710000018,
      "completed_at": 1710000018,
      "active_revision": {
        "id": "r2",
        "content_text": "Start by...",
        "thinking_text": "I need to answer with steps...",
        "source": "original",
        "created_at": 1710000001
      },
      "revision_count": 1,
      "attachments": []
    }
  ]
}
```

### `POST /api/chats/:chat_id/messages`

This endpoint can be omitted in the first slice if generation endpoint creates both user and assistant messages.

Preferred MVP simplification:

* create user message and start generation through one endpoint instead of splitting them unnecessarily

### `PATCH /api/chats/:chat_id/messages/:message_id`

Edits a message in place by appending a new revision and updating `active_revision_id`.

Request:

```json
{
  "content_text": "Updated message text",
  "attachments": [
    { "id": "attachment-id" }
  ]
}
```

Response returns the updated message shape.

Attachment IDs may reference newly uploaded pending attachments or attachments from the current revision. Existing attachment records/files should be copied to the new revision instead of moved, so older revisions keep their original attachment set.

### `POST /api/chats/:chat_id/messages/:message_id/branch`

Creates a sibling branch from an edited user message and starts a new assistant generation.

Request:

```json
{
  "content_text": "Alternative user message",
  "backend_id": "b1",
  "model_name": "gemma4",
  "persona_version_id": null,
  "think_mode": "medium",
  "attachments": [
    { "id": "attachment-id" }
  ]
}
```

### `POST /api/chats/:chat_id/messages/:message_id/regenerate`

Creates a sibling assistant message under the same parent and starts generation from the active prompt path.

Request:

```json
{
  "backend_id": "b1",
  "model_name": "gemma4",
  "persona_version_id": null,
  "think_mode": "medium"
}
```

### `PATCH /api/chats/:chat_id/active-root`

Sets the selected root message for branch navigation when the first user message has sibling branches.

Request:

```json
{
  "active_root_message_id": "m1b"
}
```

### `PATCH /api/chats/:chat_id/messages/:message_id/active-child`

Sets the selected child for branch navigation.

Request:

```json
{
  "active_child_message_id": "m3"
}
```

### `PATCH /api/chats/:chat_id/messages/:message_id/active-revision`

Sets the selected revision for edit navigation.

Request:

```json
{
  "active_revision_id": "r4"
}
```

### `DELETE /api/chats/:chat_id/messages/:message_id`

Soft-deletes a message, scrubs revision text, leaves the tree node in place, and skips the message during prompt construction.

---

## 6.9 Standard generation endpoint

### `POST /api/chats/:chat_id/generate`

Purpose:

* append a user message to a standard chat
* create placeholder assistant message
* call Ollama backend
* stream assistant output back to client
* persist assistant thinking and final content when complete
* create branchable message nodes and revisions

Request:

```json
{
  "user_message": {
    "content_text": "What kind of paint works on galvanized steel?"
  },
  "backend_id": "b1",
  "model_name": "gemma4",
  "persona_version_id": null,
  "think_mode": "medium",
  "attachments": [
    {
      "id": "a1"
    }
  ]
}
```

Streaming response direction:

* use authenticated chunked HTTP with newline-delimited JSON
* do not use WebSockets
* prefer NDJSON over SSE for this app because generation is started by authenticated `POST`, browser `fetch` handles streamed response bodies cleanly, and the framing closely matches Ollama's native streaming API
* keep format simple and documented

Suggested event stream:

```jsonl
{"type":"message_start","user_message":{"id":"m1"},"assistant_message":{"id":"m2"}}
{"type":"thinking_delta","assistant_message_id":"m2","delta":"We need to explain primer..."}
{"type":"content_delta","assistant_message_id":"m2","delta":"Use a ..."}
{"type":"content_delta","assistant_message_id":"m2","delta":"primer first"}
{"type":"message_done","assistant_message_id":"m2","done_reason":"stop"}
```

On backend side:

* persist user message before generation begins
* persist assistant placeholder with `status=streaming`
* if a persona version is selected, validate access and resolve its base backend/model before generation
* when a persona version is used, include its system prompt in the prompt sent to Ollama
* persona system prompt should be placed before user/assistant chat history, preferably as the first `system` role message
* claim pending attachment IDs onto the new user-message revision before prompt construction
* keep an in-memory snapshot of in-flight assistant thinking/content while streaming
* overlay that in-memory snapshot onto `GET /api/chats/:chat_id/messages` so route changes can recover the whole partial response
* write assistant revision thinking/content to SQLite when generation completes, stops, or errors
* mark assistant message `complete` on success
* mark assistant message `stopped` if the user stops generation
* mark `error` on failure
* prompt construction walks the active path and excludes deleted messages
* prompt construction appends UTF-8 text attachments to message content and passes image attachments through Ollama's `images` array
* assistant messages should store the resolved model/persona snapshot used for display

Notes:

* do not write every streamed token to SQLite in MVP
* do not require token sequence numbers for normal UI reconnects; reconnect reads the full accumulated snapshot, not a delta replay
* a backend process crash during active generation can still lose the in-memory partial response; avoiding that would require periodic or append-only persistence and is not part of the lightweight MVP behavior

### `POST /api/chats/:chat_id/messages/:message_id/stop`

Stops an in-flight assistant generation.

Behavior:

* no WebSocket is required
* backend keeps an in-memory cancellation handle per active assistant message
* stop marks the current assistant message `stopped`
* already-generated thinking/content is persisted when the stop is handled
* the stopped assistant message is treated as a completed node for future branch navigation

---

## 6.10 Standard attachment endpoints

### `POST /api/chats/:chat_id/attachments`

Multipart upload.

Request metadata:

* `file`
* optional `message_id`
* optional `revision_id`

If `message_id` and `revision_id` are omitted, the upload is a pending composer attachment. Generation can later claim it by attachment ID and bind it to the newly created user-message revision.

If `message_id` and `revision_id` are present, both must be present and must belong to the chat owner. `revision_id` determines which version of a message includes the attachment in prompts.

Supported files:

* common images: `jpg`, `jpeg`, `png`, `gif`, `webp`
* UTF-8 text files, including source-code files where bytes decode cleanly as text
* unsupported binary files are rejected

Response:

```json
{
  "attachment": {
    "id": "a1",
    "message_id": "m1",
    "revision_id": "r1",
    "original_filename": "photo.jpg",
    "mime_type": "image/jpeg",
    "size_bytes": 12345,
    "attachment_kind": "image"
  }
}
```

### `GET /api/attachments/:attachment_id`

Behavior:

* returns file content if user owns it

### `DELETE /api/attachments/:attachment_id`

Response:

```json
{
  "ok": true
}
```

---

## 6.11 Private-local generation endpoint

### `POST /api/private/generate`

Purpose:

* accept transient private-local history from the client
* accept transient private-local persona system prompts from the client
* forward to chosen Ollama backend
* stream assistant output back
* avoid persisting content to DB or file storage

Request:

```json
{
  "backend_id": "b1",
  "model_name": "gemma4",
  "system_prompt": "You are careful, concise, and cite uncertainty.",
  "messages": [
    {
      "role": "user",
      "content_text": "hello"
    },
    {
      "role": "assistant",
      "content_text": "hi"
    },
    {
      "role": "user",
      "content_text": "help me fix a door",
      "images": []
    }
  ]
}
```

Behavior:

* no DB writes of prompt/response content
* no DB writes of private persona content or metadata
* private personas are resolved in the frontend from IndexedDB and sent only as transient prompt data
* private-local attachments are persisted only in browser IndexedDB, never in server DB or upload storage
* text attachments are appended to transient `content_text`
* image attachments are sent as transient Ollama `images` entries on the relevant message
* response uses the same stream format as standard generation

### Debug-only private stream test endpoint

In debug builds, the backend may expose:

* `POST /api/dev/private-stream-test`

Purpose:

* emit deterministic private-generation stream chunks without calling Ollama
* stress-test frontend stream handling and IndexedDB persistence
* verify that no thinking/content chunks are dropped, duplicated, reordered, or overwritten during fast local streaming

This endpoint must remain debug-only and should not be exposed as a production API.

---

## 7. Frontend Screen Design

Visual direction:

* use the Vashti logo as the primary brand asset
* keep the app theme anchored in black surfaces with neon green accents
* preserve readability by using the neon treatment for brand, focus, and primary actions rather than every text element

## 7.1 Login and account creation screen

Shown when:

* not authenticated

Needs:

* identifier field
* password field
* login button
* create-account path with username, optional email, and password fields

Calls:

* `POST /api/auth/login`
* `POST /api/auth/register`
* `GET /api/auth/session`

Behavior:

* login signs in existing enabled users
* create account signs in the user only when the created account becomes admin
* regular self-created users see a pending-approval state

## 7.2 Main app shell

Loads after auth.

Startup fetches:

* `GET /api/auth/session`
* `GET /api/chats`
* `GET /api/models`
* `GET /api/personas`
* `GET /api/user-settings`
* optional `GET /api/models/status` later or lazily

The authenticated session response also carries the current user's private
storage key. The shell can therefore render encrypted cached chat summaries and
the cached model picker immediately, while the authoritative `/api/chats` and
model refreshes continue in the background. The chat-summary cache is hydrated
only once per app session so later refreshes cannot momentarily restore stale
rows.

Layout:

* left sidebar for chat list
* center chat view
* header with sidebar toggle, model picker, new chat, overflow menu
* bottom composer with upload, conversation settings, and send
* desktop keeps chat history visible in the left sidebar
* mobile collapses chat history into the top-left menu
* standard and private-local chats share one chat history list
* private-local chats are marked with a lock icon rather than shown in a separate section
* the New Chat button returns to the shared start composer and must not create an empty persisted chat
* the start composer includes a persistent standard/private mode toggle; the mode should default to the last mode the user actively used
* the top-right gear opens the settings/user menu, including logout
* settings open as a full page in the main content area rather than a modal
* client routes should preserve major app state across refreshes; settings use `/app/settings/:section`, and chat routes should use a stable chat URL once chats exist

Model picker behavior:

* base Ollama models remain grouped by backend
* server personas appear near their underlying backend/model with a `Custom` tag
* public personas may show owner attribution, such as `by dane`
* private personas appear only while starting or viewing private-local chats and should use a lock icon
* private personas must never appear as selectable options for standard chats

## 7.3 Standard chat view

When a standard chat is opened:

* fetch `GET /api/chats/:chat_id`
* fetch `GET /api/chats/:chat_id/messages`

On send:

* call `POST /api/chats/:chat_id/generate`
* append stream tokens into in-progress assistant message
* if the chat is bound to a persona version, send that version ID and render the assistant label using the persona snapshot
* the conversation settings menu should allow updating a persona-bound chat to a newer version or switching to a visible older version

## 7.4 Private-local chat view

State source:

* IndexedDB

When opened:

* load local chat from IndexedDB
* no fetch for messages from server

On send:

* call `POST /api/private/generate`
* append assistant output locally
* save resulting chat back to IndexedDB
* if a private persona is selected, resolve its system prompt locally and send it transiently with the generation request
* private persona metadata should remain local and should not be written to server-backed chat records

## 7.5 Settings/admin views

Sections:

* profile/basic preferences
* personas / custom models
* personal model picker visibility
* app settings (admin only)
* users (admin only)
* Ollama backends (admin only)
* global model availability and model permission tags (admin only, shown with backend settings)
* tools/search settings (admin only)
* advanced network access settings (admin only)

Layout:

* settings are a full app page with section navigation
* personal settings appear before admin settings
* admin user management lives inside settings
* user lists should separate pending/disabled users from enabled users so approval movement is clear
* advanced network settings require re-entering the admin password and acknowledging lockout risk before saving
* network settings should include a Caddy/nginx reverse-proxy config generator
* all users can manage personal model picker visibility for models already available to their account

Persona management view needs:

* list own server personas
* list private-local personas stored on this device
* create/edit/delete or disown personas
* visibility/storage mode selector: private-local, server private, server public
* confirmation when moving a persona across the private/server boundary
* prompt viewer for public personas
* copy visible persona version into a new owned persona
* show version history and current version

Backend management view needs:

* backend list
* add backend form
* edit backend form
* enable/disable toggle
* localhost detect action
* expandable global model controls grouped by backend
* per-model global enable/disable switches and permission tags

---

## 8. Frontend Local Data Model

## 8.1 IndexedDB for private-local chats

Store objects like:

* `private_chats`
* `private_messages`
* `private_personas`
* `private_persona_versions`
* `private_context_categories`
* `private_context_blocks`
* `private_context_block_versions`
* `hosted_chat_cache`
* `hosted_chat_list_cache`
* `model_cache`
* local attachment metadata and payloads embedded on private messages

The hosted-chat and model stores are per-user encrypted responsiveness caches;
the server remains authoritative. Listing private chats reads chat metadata and
uses the `private_messages.chat_id` index to count messages without loading or
decrypting every message payload at startup.

Suggested private chat shape:

```json
{
  "id": "client-uuid",
  "title": "Private chat",
  "backend_id": "b1",
  "model_name": "gemma4",
  "private_persona_id": "client-persona-uuid",
  "private_persona_version_id": "client-persona-version-uuid",
  "created_at": 1710000000,
  "updated_at": 1710000100
}
```

Suggested private message shape:

```json
{
  "id": "client-msg-uuid",
  "chat_id": "client-uuid",
  "parent_message_id": null,
  "active_child_message_id": null,
  "active_revision_id": "client-revision-uuid",
  "role": "user",
  "status": "complete",
  "is_deleted": false,
  "active_revision": {
    "id": "client-revision-uuid",
    "content_text": "hello",
    "thinking_text": "",
    "source": "original",
    "created_at": 1710000000
  },
  "attachments": [
    {
      "id": "client-attachment-uuid",
      "message_id": "client-msg-uuid",
      "revision_id": "client-revision-uuid",
      "original_filename": "photo.png",
      "mime_type": "image/png",
      "size_bytes": 12345,
      "attachment_kind": "image",
      "data_url": "data:image/png;base64,..."
    }
  ],
  "created_at": 1710000000
}
```

Private-local storage should mirror the standard message tree shape where practical, while remaining client-owned and IndexedDB-backed.

For MVP private-local attachments:

* image payloads may be stored as data URLs in IndexedDB
* text payloads may be stored as UTF-8 strings in IndexedDB
* unsupported binary files should be rejected client-side
* branch/send operations should preserve existing attachments unless the UI explicitly supports replacing them

Private-local chats should use the same branch and revision navigation model as standard chats:

* edit + save appends a revision on the same local message
* user edit + send creates a sibling branch and generates a new local assistant child
* assistant regenerate creates a sibling assistant message under the same parent
* the UI presents revisions and sibling branches as one chronological version list

## 8.2 IndexedDB for private personas

Private persona storage should mirror server persona/version concepts without syncing to the server.

Suggested private persona shape:

```json
{
  "id": "client-persona-uuid",
  "current_version_id": "client-persona-version-uuid",
  "created_at": 1710000000,
  "updated_at": 1710000100
}
```

Suggested private persona version shape:

```json
{
  "id": "client-persona-version-uuid",
  "persona_id": "client-persona-uuid",
  "version_number": 1,
  "display_name": "Careful Researcher",
  "avatar_data_url": null,
  "base_backend_id": "b1",
  "base_model_name": "gemma4",
  "system_prompt": "You are careful, concise, and cite uncertainty.",
  "tool_policy_json": null,
  "created_at": 1710000000
}
```

Rules:

* private personas are selectable only for private-local chats
* server personas used privately should be copied into local persona storage before generation
* moving a private persona to the server creates a server persona and version from the current local version
* moving a server persona to private storage creates a local copy from the selected server version and then follows server lifecycle rules for the original
* either direction across the private/server boundary requires a confirmation dialog
* private persona deletion removes only local IndexedDB records

## 8.3 IndexedDB for private context blocks

Device-only context categories, block identities, and immutable block versions
mirror the server model but are stored as encrypted private records. A private
chat stores its ordered selected version snapshots, and each generated
assistant message stores the snapshots used for that response. This preserves
old conversations if a local block is edited or removed.

Private context rules:

* private chats may select device blocks only
* category `single`/`multiple` behavior is enforced before generation
* block content is compiled into the transient system prompt in the frontend
* neither block IDs, names, category names, nor content are persisted by the server
* clearing browser storage can permanently remove the device library

---

## 9. Repo Layout Direction

Suggested repo shape:

```text
vashti/
  Cargo.toml              # workspace manifest
  apps/
    vashti/
      Cargo.toml
      build.rs
      docs/
      packaging/
      scripts/
      src/
      migrations/
      web/
    vashti-hub/
      Cargo.toml
      docs/
      packaging/
      scripts/
      src/
      migrations/
      static/
```

Notes:

* `apps/vashti` is the self-hosted chat app
* `apps/vashti-hub` is the release/download/admin site for `vashti.chat`
* app-specific docs, packaging files, release scripts, and dev data live under the app that owns them
* `cargo build -p vashti` runs the Vite production build when needed, then embeds `apps/vashti/web/dist` into the Rust binary
* `apps/vashti/web/dist` is generated output and should not be tracked in git
* local development data defaults to `apps/vashti/data` and `apps/vashti-hub/data` when commands are run from the workspace root
* hub storage lives outside the repo in production, normally under `/var/lib/vashti-hub`

---

## 10. Release and Deployment

Vashti should support prerelease distribution before `v1.0.0`.

Versioning:

* use SemVer tags such as `v0.1.0`, `v0.1.1`, and eventually `v1.0.0`
* `Cargo.toml` package version is the app version
* `GET /api/version` returns the running app name and version
* `v0.x.y` releases should be treated as prereleases

Release artifacts:

* `vashti.chat/releases` is the canonical binary source
* `vashti.chat` is served by the `vashti-hub` Rust app, not by static hosting
* initial target is Linux x86_64
* release assets should include a self-contained binary archive, `SHA256SUMS`, and `VERSION`
* the release archive should include the Vashti binary, a systemd service example, and install notes

Install/update path:

* `install.sh` should detect OS/architecture, download the matching archive from `vashti.chat/releases`, verify checksums, install the binary, and configure systemd on Linux
* the default packaged install should store data in `/var/lib/vashti`
* packaged installs should use `WorkingDirectory=/var/lib/vashti` so `recover_network.txt` recovery lives there
* before in-app self-update exists, updates are performed by rerunning the installer
* publishing should upload artifacts to the hub API using short-lived one-time bearer keys created from the Hub admin page

Vashti Hub:

* serves the public website and `install.sh`
* accepts release artifact uploads through `POST /api/releases`
* validates `vMAJOR.MINOR.PATCH` version labels supplied by the packaged app
* computes checksums server-side
* marks the highest uploaded SemVer version as `latest`
* records basic download counts without storing raw IP addresses
* exposes an admin page for one-time upload key creation and stats

---

## 11. First Implementation Slice

The first real coding slice should aim for this outcome:

1. Axum server starts
2. SQLite connects
3. migrations run
4. session/register/login check works
5. embedded or placeholder frontend shell is served
6. localhost Ollama detect endpoint works
7. `GET /api/models` can return grouped model info from configured backends

This is a better first milestone than jumping straight into full chat UI.

### 11.1 Persona implementation slices

Recommended persona sequence:

1. add server persona/persona-version/persona-member tables and service module
2. add private IndexedDB persona stores and local CRUD helpers
3. add settings UI for creating/editing own personas
4. add persona options to the model picker, with private personas restricted to private-local chats
5. resolve selected persona versions during generation and include system prompts
6. bind chats/messages to immutable persona version snapshots
7. add public sharing, prompt visibility, copy-persona, and disown lifecycle
8. add conversation settings for updating or switching persona versions
9. add tag-based access rules and persona quotas as a later admin-management slice

---

## 12. Codex Handoff Guidance

When handing this to Codex, do not ask it to generate the whole app at once.

Preferred sequence:

1. scaffold backend modules and config
2. add DB and migrations
3. implement register/login/auth/session endpoints
4. add backend management endpoints
5. add Ollama client and model listing
6. scaffold frontend shell
7. add standard chat CRUD
8. add streaming generation
9. add private-local IndexedDB mode
10. add uploads
11. add model personas
12. add PWA polish

Codex should be instructed to:

* modify one slice at a time
* avoid inventing extra frameworks
* keep API contracts aligned with this document
* avoid adding WebSockets in MVP
* avoid adding RAG or provider abstraction
