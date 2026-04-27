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
* upload selection UX
* PWA installation and asset caching

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
* `created_at` INTEGER NOT NULL
* `updated_at` INTEGER NOT NULL

Notes:

* one row only
* admin edits this through settings UI
* `allow_signup` and `signup_limit` apply only when an enabled admin already exists
* adminless account creation remains allowed so a system with no admins can recover
* `signup_count` tracks successful public self-registrations after an admin exists; when it reaches `signup_limit`, the app sets `allow_signup = 0`

### 3.2.6 `chats`

Purpose:

* server-backed standard chats only
* owns chat-level defaults and the active root message

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `default_backend_id` TEXT NOT NULL REFERENCES `ollama_backends`(`id`) ON DELETE RESTRICT
* `default_model_name` TEXT NOT NULL
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

Columns:

* `id` TEXT PRIMARY KEY
* `user_id` TEXT NOT NULL REFERENCES `users`(`id`) ON DELETE CASCADE
* `chat_id` TEXT REFERENCES `chats`(`id`) ON DELETE CASCADE
* `message_id` TEXT REFERENCES `chat_messages`(`id`) ON DELETE SET NULL
* `revision_id` TEXT REFERENCES `chat_message_revisions`(`id`) ON DELETE CASCADE
* `storage_key` TEXT NOT NULL UNIQUE
* `original_filename` TEXT NOT NULL
* `mime_type` TEXT
* `size_bytes` INTEGER NOT NULL
* `attachment_kind` TEXT NOT NULL
  Allowed values: `image`, `file`
* `created_at` INTEGER NOT NULL

Notes:

* standard-chat attachments only in server DB
* private-local attachments are not persisted server-side in MVP
* attachments are intentionally deferred from the first text-chat slice
* `revision_id` is the authoritative link for prompt content
* `message_id` is retained for ownership checks, listing, and cleanup convenience
* edit + save can replace text and attachments together by creating a new revision with its own attachment set
* deleting a message should delete or detach attachment records for all of its revisions according to the later file-deletion policy

### 3.2.10 `schema_migrations`

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
* Secure when running over HTTPS
* Path=/

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

## 6.2 Auth endpoints

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
  "user": null
}
```

`can_create_account` is true when no enabled admin exists, or when an enabled admin exists and public signup is currently enabled and under the configured signup limit. The frontend uses this to show or hide the create-account action on the login screen.

---

## 6.3 Admin/user endpoints

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
  "role": "user"
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

## 6.4 Settings endpoints

### `GET /api/settings`

Admin-oriented app settings.

Response:

```json
{
  "allow_signup": true,
  "signup_limit": 25,
  "signup_count": 3,
  "max_upload_bytes": 10485760,
  "request_timeout_ms": 120000
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

### `GET /api/user-settings`

Response:

```json
{
  "default_backend_id": "uuid",
  "default_model_name": "gemma4"
}
```

### `PATCH /api/user-settings`

Request:

```json
{
  "default_backend_id": "uuid",
  "default_model_name": "gemma4"
}
```

---

## 6.5 Ollama backend endpoints

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

Returns models grouped by backend.

Response:

```json
{
  "backends": [
    {
      "backend": {
        "id": "b1",
        "name": "mac-mini"
      },
      "models": [
        {
          "name": "gemma4",
          "supports_images": true
        },
        {
          "name": "llama3:70b",
          "supports_images": false
        }
      ]
    }
  ]
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

## 6.6 Standard chat endpoints

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
  "default_model_name": "gemma4"
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
    "active_root_message_id": "m1",
    "created_at": 1710000000,
    "updated_at": 1710000500
  }
}
```

### `PATCH /api/chats/:chat_id`

Supports rename and model/backend changes. Chat rename is exposed from each chat row's sidebar overflow menu.

Request example:

```json
{
  "title": "Better title",
  "default_backend_id": "b2",
  "default_model_name": "qwen3.5"
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
  "content_text": "Updated message text"
}
```

Response returns the updated message shape.

### `POST /api/chats/:chat_id/messages/:message_id/branch`

Creates a sibling branch from an edited user message and starts a new assistant generation.

Request:

```json
{
  "content_text": "Alternative user message",
  "backend_id": "b1",
  "model_name": "gemma4",
  "think_mode": "medium"
}
```

### `POST /api/chats/:chat_id/messages/:message_id/regenerate`

Creates a sibling assistant message under the same parent and starts generation from the active prompt path.

Request:

```json
{
  "backend_id": "b1",
  "model_name": "gemma4",
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

## 6.7 Standard generation endpoint

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
  "think_mode": "medium",
  "attachments": []
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
* update assistant revision thinking/content as stream proceeds or write final text at end
* mark assistant message `complete` on success
* mark assistant message `stopped` if the user stops generation
* mark `error` on failure
* prompt construction walks the active path and excludes deleted messages

### `POST /api/chats/:chat_id/messages/:message_id/stop`

Stops an in-flight assistant generation.

Behavior:

* no WebSocket is required
* backend keeps an in-memory cancellation handle per active assistant message
* stop marks the current assistant message `stopped`
* already-streamed thinking/content remains persisted
* the stopped assistant message is treated as a completed node for future branch navigation

---

## 6.8 Standard attachment endpoints

### `POST /api/chats/:chat_id/attachments`

Multipart upload.

Request metadata:

* `message_id`
* `revision_id`

`revision_id` determines which version of a message includes the attachment in prompts.

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

## 6.9 Private-local generation endpoint

### `POST /api/private/generate`

Purpose:

* accept transient private-local history from the client
* forward to chosen Ollama backend
* stream assistant output back
* avoid persisting content to DB or file storage

Request:

```json
{
  "backend_id": "b1",
  "model_name": "gemma4",
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
      "content_text": "help me fix a door"
    }
  ],
  "attachments": []
}
```

Behavior:

* no DB writes of prompt/response content
* no attachment persistence for private-local mode in MVP
* response uses the same stream format as standard generation

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
* `GET /api/user-settings`
* optional `GET /api/models/status` later or lazily

Layout:

* left sidebar for chat list
* center chat view
* header with sidebar toggle, model picker, new chat, overflow menu
* bottom composer with upload and send
* desktop keeps chat history visible in the left sidebar
* mobile collapses chat history into the top-left menu
* the top-right gear opens the settings/user menu, including logout
* settings open as a full page in the main content area rather than a modal
* client routes should preserve major app state across refreshes; settings use `/app/settings/:section`, and chat routes should use a stable chat URL once chats exist

## 7.3 Standard chat view

When a standard chat is opened:

* fetch `GET /api/chats/:chat_id`
* fetch `GET /api/chats/:chat_id/messages`

On send:

* call `POST /api/chats/:chat_id/generate`
* append stream tokens into in-progress assistant message

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

## 7.5 Settings/admin views

Sections:

* profile/basic preferences
* app settings (admin only)
* users (admin only)
* Ollama backends (admin only)

Layout:

* settings are a full app page with section navigation
* admin user management lives inside settings
* user lists should separate pending/disabled users from enabled users so approval movement is clear

Backend management view needs:

* backend list
* add backend form
* edit backend form
* enable/disable toggle
* localhost detect action

---

## 8. Frontend Local Data Model

## 8.1 IndexedDB for private-local chats

Store objects like:

* `private_chats`
* `private_messages`
* optional local attachment metadata

Suggested private chat shape:

```json
{
  "id": "client-uuid",
  "title": "Private chat",
  "backend_id": "b1",
  "model_name": "gemma4",
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
  "created_at": 1710000000
}
```

Private-local storage should mirror the standard message tree shape where practical, while remaining client-owned and IndexedDB-backed.

---

## 9. Repo Layout Direction

Suggested repo shape:

```text
vashti/
  Cargo.toml
  src/
    main.rs
    app_state.rs
    config.rs
    db.rs
    error.rs
    auth/
      mod.rs
      handlers.rs
      service.rs
      middleware.rs
    admin/
      mod.rs
      handlers.rs
    backends/
      mod.rs
      handlers.rs
      service.rs
    chats/
      mod.rs
      handlers.rs
      service.rs
      models.rs
    private/
      mod.rs
      handlers.rs
      service.rs
    uploads/
      mod.rs
      handlers.rs
      service.rs
    frontend/
      mod.rs
    ollama/
      mod.rs
      client.rs
      models.rs
    startup/
      mod.rs
      bootstrap.rs
      migrations.rs
  migrations/
  web/
    package.json
    vite.config.ts
    src/
    public/
```

Notes:

* backend and frontend live in one repo
* `cargo build` runs the Vite production build when needed, then embeds `web/dist` into the Rust binary
* `web/dist` is generated output and should not be tracked in git
* exact module names can change slightly, but ownership boundaries should stay similar

---

## 10. First Implementation Slice

The first real coding slice should aim for this outcome:

1. Axum server starts
2. SQLite connects
3. migrations run
4. session/register/login check works
5. embedded or placeholder frontend shell is served
6. localhost Ollama detect endpoint works
7. `GET /api/models` can return grouped model info from configured backends

This is a better first milestone than jumping straight into full chat UI.

---

## 11. Codex Handoff Guidance

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
10. add uploads and PWA polish

Codex should be instructed to:

* modify one slice at a time
* avoid inventing extra frameworks
* keep API contracts aligned with this document
* avoid adding WebSockets in MVP
* avoid adding RAG or provider abstraction
