# Vashti MVP Spec

## 1. Product Summary

**Vashti** is a lightweight, self-hosted web app for chatting with local LLMs through an Ollama server.

Its priorities are:

* fast startup and fast page load
* simple installation and updates
* minimal dependencies
* multi-user support without enterprise complexity
* a React frontend that feels app-like on desktop and mobile
* a single Rust binary plus a data directory

Vashti is **not** intended to be a plugin framework, agent platform, workflow engine, or feature-for-feature clone of Open WebUI.

---

## 2. Core Product Goals

### 2.1 Main goals

* Ship as a **single Rust binary**.
* Use **SQLite** for persistent app data.
* Serve a **React frontend** from the Rust backend.
* Embed frontend build assets into the binary.
* Support **multi-user accounts**.
* Connect to **one or more Ollama servers** as the only model backend type in v1.
* Be installable as a **PWA**.
* Keep the interface lightweight, fast, and simple.

### 2.2 User experience goals

* App should feel closer to ChatGPT than to an admin dashboard.
* Sidebar should load quickly even with many chats.
* Full chat bodies should **not** be loaded until the user opens a chat.
* The app should work well from phone and desktop.

---

## 3. Non-Goals for MVP

These are explicitly out of scope for the first release:

* RAG and document retrieval pipelines
* vector databases
* bundled embedding models
* plugins, tools marketplace, or pipeline framework
* OpenAI-compatible provider abstraction for many backends
* end-to-end encryption or Proton-style privacy claims
* multi-device sync for local-only private chats
* enterprise SSO, RBAC, teams, org workspaces
* heavy UI frameworks like Bootstrap or Material UI
* desktop app packaging

---

## 4. Core Chat Modes

Vashti supports two chat modes in the product spec.

### 4.1 Standard chat

Standard chats are stored on the server and synced across the user’s devices.

Properties:

* persisted in SQLite
* visible on all devices for that user
* suitable for normal everyday usage
* can support future features like share/export more easily

### 4.2 Private local chat

Private local chats are stored only on the client device.

Properties:

* content is stored in browser storage on that device
* content is **not persisted** on the server
* not synced across devices in v1
* if browser data is cleared, the chat may be lost
* intended as a privacy-oriented mode for users who do not want server-side history storage

Important caveat:

* messages still pass through the Vashti server during live use so the server can relay them to Ollama
* Vashti should avoid storing or logging private chat content on the server

---

## 5. MVP Feature Set

### 5.1 Authentication and accounts

* local account system
* unauthenticated users can create an account from the login screen
* if no enabled admin account exists when a user account is created, that new account becomes an enabled admin
* if an enabled admin already exists, self-created accounts are regular users and start disabled pending admin approval
* admin can disable public signups
* admin can set a public-signup limit; once that limit is reached, signups are disabled automatically
* Vashti must not auto-generate an admin account or password on startup
* login with username or email plus password
* secure session cookie auth
* admin can create users
* optional invite-only signup flow later, but adminless recovery must remain possible

### 5.2 Chat features

* create a new chat
* choose chat mode: standard or private local
* list chats in sidebar
* rename chat
* delete chat
* auto-generate a placeholder chat title after the first assistant response when possible
* stream assistant responses
* select model per chat
* create a new chat quickly from the header

### 5.3 Model integration

* connect to one or more configured Ollama servers
* support multiple Ollama backends in the admin configuration
* auto-detect Ollama on `localhost` / `127.0.0.1` during first-run setup when available
* fetch available models from configured Ollama servers
* present models grouped by server in the model picker
* choose a default model per user
* override model per chat
* support image input for models that accept images

Model picker direction:

* each Ollama server has a human-readable name, such as `mac-mini` or `rtx-3060`
* the picker should group models under their server name
* users choose a server+model combination rather than a global flat model list

Admin backend management direction:

* add Ollama backends manually
* edit backend name and URL
* enable or disable individual backends
* optional local-network scan can be triggered manually by an admin later

### 5.4 Model personas

Vashti should support user-created model personas, exposed in the UI as custom models.

A persona is not a separate Ollama model. It is a reusable profile that resolves to:

* an underlying Ollama backend
* an underlying Ollama model name
* an immutable persona version
* a system prompt
* display metadata such as name and optional avatar
* future tool policy metadata

Persona storage modes:

* server personas are stored in SQLite
* private personas are stored only in browser IndexedDB
* public personas are server personas visible to other users

Privacy and usage rules:

* private personas may be used only in private-local chats
* standard chats may use base Ollama models or server personas only
* private-local chats may use base Ollama models or private-local personas
* using a public server persona in private-local mode should copy the selected version to local storage before use, avoiding server-side membership or usage tracking as a side effect of a private chat
* public personas do not require admin approval in MVP
* users must be able to inspect the system prompt for public personas they can use
* using a persona binds the chat to a specific immutable persona version
* editing a persona creates a new version rather than mutating the version used by existing chats
* deleting a persona must not break old chats

Public persona lifecycle direction:

* when a user uses a public persona, they become a member/user of that persona
* deleting a public persona after others have used it should behave like disowning it
* the persona can be deleted from the server only after the last member disowns it
* the creator remains the only user who can publish new versions in MVP
* users may copy a visible persona version into a new persona they own

### 5.5 Context blocks

Vashti should provide a reusable context library that is independent of custom
models. A context block is a named prompt fragment that can be attached to any
chat whose selected model is otherwise available to the user.

Context blocks are organized into user-created categories. Categories are for
organization and selection behavior, not permissions:

* a `single` category allows at most one selected block in a chat
* a `multiple` category allows any number of selected blocks
* users may select blocks from several categories and reorder the final prompt order
* blocks remain usable with base models and custom models alike

Version and generation rules:

* block content edits create immutable versions
* a chat pins the exact selected block versions until the user changes them
* deleting a block removes it from the library without breaking existing chats
* each assistant generation snapshots the block versions used for that response
* generation prepends the effective model/custom-model system prompt first, then selected blocks in the user's explicit order
* Vashti must reject an oversized compiled system prompt rather than silently truncating it

Storage modes follow the same privacy boundary as chats:

* server blocks are owner-only in MVP and may be used in standard chats
* device blocks are stored only in encrypted browser IndexedDB and may be used in private-local chats
* standard chats must never reference device-only blocks
* private chats must not disclose device-only block names or content to server storage
* copying blocks across the server/device boundary may be added later as an explicit confirmed action

### 5.6 Attachments

* upload images
* upload files
* for MVP, attachments do **not** imply RAG
* image attachments should work with multimodal models when supported
* generic file attachments may be stored and surfaced in UI without semantic retrieval

### 5.7 PWA support

* installable web app
* manifest and icons
* service worker for static asset caching
* fast reload of app shell
* responsive layout for mobile and desktop

### 5.8 Settings

* default Ollama backend
* default model
* persona management
* server and device context-block libraries
* request timeout
* upload size limit
* signup/admin settings, including public signup enablement and limit
* personal model picker visibility
* admin global model availability
* advanced network access mode for LAN HTTP vs public HTTPS reverse-proxy deployment

### 5.9 Admin basics

* manage users

* manage Ollama backends

* manage global model availability with a simple on/off switch and tag-based access rules

* configure public HTTPS reverse-proxy mode without exposing a raw Secure-cookie toggle

* allow users to hide accessible models from their own model picker

* disable or delete users

* see app version

* see basic storage/config health later if needed

* manage users

* disable or delete users

* see app version

* see basic storage/config health later if needed

### 5.9 Release and deployment

* `vashti.chat/releases` is the canonical source for binary downloads
* `v0.x.y` builds are prereleases until Vashti is ready for `v1.0.0`
* first release target is Linux x86_64
* release archives include the embedded frontend binary, systemd service example, install notes, and checksums
* `vashti.chat` is served by Vashti Hub, a small Rust app with an upload API, install script route, admin page, and download stats
* release uploads use short-lived one-time bearer keys created from the Hub admin page, plus server-computed checksums
* the highest uploaded SemVer version becomes `latest`
* in-app self-update is deferred until the install/release path is stable

---

## 6. Privacy Model

### 6.1 Standard chat privacy

* standard chats are stored on the server
* server operator can theoretically access them but there will not be anything in the admin UI to allow this
* users should not be promised zero-access encryption

### 6.2 Private local chat privacy

* content stored only in browser storage on the device
* no server-side persistence of chat content
* no sync in MVP
* user should receive clear warning that local browser storage is the source of truth
* if device storage is cleared, the chat may be unrecoverable

### 6.3 Server behavior for private local mode

For private mode, the backend should aim to:

* avoid writing prompt/response content to the database
* avoid request/response body logging
* avoid persisting private content to disk
* keep only minimal transient data in memory for live relay to Ollama

### 6.4 UI disclosure for private local mode

Suggested product wording:

> Private chats are stored only on this device and are not saved on the server. They are not synced between devices. If you clear your browser data or lose this device, these chats may be unrecoverable.

### 6.5 Persona privacy

Private personas are local-only data and follow the same privacy boundary as private-local chats.

Rules:

* private personas are stored only in browser storage on that device
* private personas must not be selectable in standard server-backed chats
* private-local chats should prefer local copies of server personas rather than directly using public persona membership flows
* if a private persona is moved to server storage, the user must confirm that its name, prompt, avatar, and base model metadata will be uploaded
* if a server persona is moved to private storage, the user must confirm that it will stop being available from other devices and may be removed from the server depending on lifecycle rules
* public persona prompts are intentionally visible to users who can use them

Rationale:

* a private persona name or prompt may itself contain sensitive information
* standard chats are server-backed, so attaching a private persona to them would weaken the privacy promise
* prompt visibility helps users evaluate safety before using public personas, especially once tools exist

### 6.6 Context-block privacy

Context blocks may contain project details, roleplay scenarios, personal notes,
or other sensitive prompt material. Their storage mode is therefore a hard
privacy boundary rather than a display preference.

Rules:

* server blocks are stored in SQLite and are available only to their owner in MVP
* device blocks and categories are stored in the same encrypted IndexedDB boundary as private chats
* standard chats may select server blocks only
* private-local chats may select device blocks only
* private generation sends compiled device-block content transiently but never stores block metadata or content on the server
* moving or copying a block across the boundary requires an explicit future confirmation flow

---

## 7. Performance Strategy

### 7.1 App startup

On initial app load, the frontend should request only:

* current session/user info
* chat summaries (id, title, updated time, mode)
* model list
* minimal settings needed for the UI

It should **not** request every message from every chat.

### 7.2 Lazy chat loading

* full chat history loads only when a chat is opened
* long chats can paginate older messages later if needed
* recent messages load first

### 7.3 Browser caching

* static assets cached with service worker
* client-side structured data may be stored locally for responsiveness
* IndexedDB is the preferred storage for local private chats and optional cached chat data

### 7.4 UI principles

* no heavy animation
* no large design system dependency
* plain CSS
* prioritize responsiveness over visual flourishes

---

## 8. Technical Architecture

### 8.1 Backend

Rust backend responsibilities:

* serve the frontend
* provide JSON API endpoints
* manage auth/session handling
* persist app data in SQLite
* communicate with Ollama
* manage uploads
* run migrations on startup

Likely stack:

* Rust
* Axum
* Tokio
* embedded SQLite
* migration support

### 8.2 Frontend

Frontend responsibilities:

* app shell and navigation
* chat UI
* local caching/storage
* PWA behavior
* upload interactions
* settings screens

Likely stack:

* React
* Vite build process
* plain CSS

Rationale:

* React remains the chosen frontend framework because it is already familiar to the developer
* familiarity is more valuable here than chasing marginal bundle-size savings from switching frameworks
* the main performance wins will come from architecture, caching, and lazy loading rather than from replacing React alone

### 8.3 Embedded assets

Frontend production build should be embedded into the Rust binary so deployment is:

* one executable
* one data directory created at runtime

---

## 9. Runtime Data Layout

On first run, Vashti should create a data directory.

Proposed structure:

```text
apps/vashti/data/
  app.db
  uploads/
  tmp/
```

Notes:

* uploads contains user-provided files that must persist
* tmp is for transient file handling
* structure migrations should be handled automatically by the app when versions change

---

## 10. Database Entities

Initial entities likely needed:

* users
* sessions
* chats
* messages
* attachments
* personas
* persona_versions
* persona_members
* user_settings
* app_settings
* schema_migrations

Notes:

* standard chats use chats/messages tables normally
* private local chats may have little or no server persistence beyond optional minimal metadata, depending on final implementation

---

## 11. UI Structure

### 11.1 Main layout

Top left:

* sidebar/menu toggle
* model picker

Top right:

* new chat button
* overflow menu for chat/user actions

Left panel:

* previous chats list
* likely grouped or sorted by recency

Main area:

* current conversation
* assistant and user messages

Bottom:

* text input
* upload/add button
* conversation settings button for less-used per-chat settings such as persona version
* send button

### 11.2 Design direction

* lightweight visual design
* clean spacing
* black/neon green visual identity anchored by the Vashti logo
* no unnecessary transitions
* mobile-friendly
* feels app-like rather than document-like

---

## 12. Attachment Rules for MVP

### 12.1 Images

* upload from desktop or phone
* mobile file picker should allow photo library selection where browser/platform supports it
* image attachments can be included in prompts sent to multimodal models

### 12.2 Files

* generic file upload support exists in MVP
* files can be attached and stored
* files do not automatically trigger embedding, chunking, or vector indexing
* future versions may add optional text extraction or RAG features

---

## 13. API and Transport Direction

### 13.1 API style

* JSON HTTP API for ordinary CRUD and app state
* streaming endpoint for model output

### 13.2 Streaming direction

Preferred default:

* keep streaming implementation as simple as possible
* avoid introducing WebSockets unless they are clearly needed

Rationale:

* WebSockets may still become useful later for presence or sync, but are not required for the MVP private-local model

---

## 14. Updates and Installation

### 14.1 Installation promise

Target installation model:

* user downloads one precompiled binary, or uses a one-line installer script
* installer should be able to place the binary, create runtime directories, and set up a system service on supported Linux systems
* app creates data directory on first run if needed
* no Python runtime
* no Docker requirement
* no extra frontend files to deploy separately

### 14.2 Updating

Long-term goal:

* app-managed self-update may exist later

MVP stance:

* manual binary replacement is acceptable
* data directory and database survive app updates
* migrations run automatically on startup

Self-update should not block the MVP.

---

## 15. Risks and Tradeoffs

### 15.1 Private local chats

Pros:

* credible privacy story
* no false claims about impossible security guarantees
* simpler than end-to-end encrypted multi-user sync

Cons:

* private chats are not recoverable if local storage is lost
* no sync between devices in MVP
* server still sees live traffic during relay unless architecture changes dramatically

### 15.2 Attachments without RAG

Pros:

* much smaller scope
* lighter install and runtime
* preserves future flexibility

Cons:

* users may assume uploaded files can be “talked to” semantically when they cannot yet
* UI must make the capability boundary clear

---

## 16. Suggested MVP Milestones

### Milestone 1: Foundations

* backend crate layout
* frontend scaffold
* static asset serving
* SQLite setup
* migration system
* config and account creation

### Milestone 2: Auth and sessions

* registration, login, and logout
* login/logout
* session handling
* basic user model

### Milestone 3: Standard chats

* chat CRUD
* branchable message tree
* message revisions for edits/regenerations
* thinking/content separation for assistant messages
* soft delete that scrubs message text but preserves tree continuity
* Ollama integration
* streaming responses
* stop generation
* model selection
* no upload UI or attachment persistence yet; keep the data model compatible with later revision-level attachments

### Milestone 4: Frontend shell

* sidebar
* chat pane
* composer
* model picker
* settings shell

### Milestone 5: Private local chats

* private local mode creation
* IndexedDB storage for private chats
* UI warning and mode handling
* no server persistence for content

### Milestone 6: Model personas and model management

* create/edit/delete own personas
* server-stored personas
* private IndexedDB personas
* immutable persona versions
* persona selection in the model picker
* private personas limited to private-local chats
* public sharing, prompt visibility, copy-persona flow, and disown lifecycle
* basic admin model on/off controls
* tag-based access rules and quotas can follow in a later slice if needed

### Milestone 7: Attachments and PWA

* image uploads
* file uploads
* service worker
* manifest/icons
* mobile installability

### Milestone 8: Hardening

* polish
* logs and error handling
* upload limits
* migration reliability
* cleanup behavior

---

## 17. Resolved Technical Decisions

The following decisions are now locked for the MVP unless a later implementation problem forces reconsideration.

### 17.1 Backend stack

* server framework: **Axum**
* async runtime: **Tokio**
* database access: **sqlx** with SQLite

Rationale:

* async-first architecture fits Axum naturally
* sqlx provides a practical migration and query workflow for this project size
* the expected scale is small enough that the overhead versus lower-level SQLite access is not a meaningful concern

### 17.2 Session/auth direction

* use **secure cookie-based sessions**
* choose the simplest maintainable server-side session approach that integrates cleanly with Axum
* no token-heavy SPA auth design for MVP

Rationale:

* browser app plus same-origin backend is a natural fit for cookie auth
* simpler than bearer-token storage in the client
* easier to reason about for a small self-hosted app

### 17.3 Private local attachments

* attachments for **private local chats are local-only in MVP**
* they should not be persisted on the server
* image attachments may be stored in IndexedDB and sent transiently to Ollama as message `images`
* UTF-8 text attachments may be stored in IndexedDB and appended transiently to prompt text
* unsupported binary files should be rejected client-side

### 17.4 Streaming direction

* use **HTTP streaming** for chat response generation in MVP
* do **not** use WebSockets for the core chat request/response path
* WebSockets remain a possible future addition for presence or sync-related events

Rationale:

* the chat interaction is fundamentally request/response with streaming output
* HTTP streaming preserves this mental model better than WebSockets
* avoids introducing connection state complexity before it is needed
* future sync or presence features can still add WebSockets later without invalidating the MVP design

### 17.5 Route ownership

* backend route layout will be designed around a clean JSON API under `/api`
* frontend app shell and static assets are served from the same binary

### 17.6 Multi-backend Ollama direction

* Vashti supports multiple Ollama backends in MVP
* backends are all Ollama servers; this is not a generic multi-provider abstraction
* each backend has a human-readable name and base URL
* users select models from a server-grouped model picker
* users can hide accessible models from their own picker without changing server permissions
* admin model defaults and admin-added model tags are separate; applying defaults updates only the default layer and preserves manual tags
* admin can enable or disable configured backends
* localhost detection on first run is desirable
* local-network discovery should be an explicit admin action rather than a background behavior

### 17.7 Private-local sync stance

* private-local chats remain **unsynced in MVP**
* future sync can be explored later, potentially with WebSockets or another event channel
* current design should avoid locking the project into assumptions that make future sync impossible

### 17.8 Model persona direction

* personas are custom model profiles layered on top of Ollama models
* personas are versioned; edits create immutable new versions
* chats bind to a specific persona version until the user explicitly updates or switches versions
* private personas are local-only and can be used only in private-local chats
* server personas may be private-to-owner or public
* public personas do not require admin approval in MVP
* users can inspect prompts for public personas they can use
* public persona deletion should use disown/membership semantics once other users have used it
* users can copy a visible persona version into a new persona they own
* tag-based base-model and tool access is part of admin management; persona quotas can follow in a later slice if needed
* a user must have access to a persona's underlying base model before they can use that persona

## 18. Immediate Next Step

Before writing implementation code, the next deliverable should be:

* a concrete technical design covering routes, tables, API payloads, and frontend screens for the MVP

That design should stay aligned with the rules above and resist scope creep.

---

## 19. Technical Direction Notes

### 19.1 Why HTTP streaming for MVP

For Vashti, model generation is naturally shaped like:

1. client sends one request to generate a response for a specific chat turn
2. server forwards that request to Ollama
3. server streams response chunks back to the client
4. client appends those chunks to the in-progress assistant message

This fits HTTP streaming cleanly.

Advantages for MVP:

* easier mental model than a general WebSocket message bus
* simpler debugging in browser dev tools and server logs
* cleaner separation between normal API calls and generation streaming
* less protocol design work early on

Possible future use of WebSockets:

* presence indicators
* sync notifications
* device wake/sync signaling for future private-chat sync experiments

### 19.2 MVP route layout direction

Proposed high-level route groups:

#### Frontend and app shell

* `GET /` → app shell
* `GET /app/*path` → app shell fallback for client routing
* `GET /assets/*path` → embedded frontend assets
* `GET /manifest.webmanifest` → PWA manifest
* `GET /sw.js` → service worker
* `GET /api/version` → running app name/version

#### Auth/session

* `POST /api/auth/login`
* `POST /api/auth/logout`
* `GET /api/auth/session`

Mutating API requests should enforce same-origin browser writes with an `Origin` check, while still allowing ordinary same-origin frontend calls and reasonable non-browser local clients. Login/register, uploads, and generation should be rate-limited in memory.

#### Account/admin

* `POST /api/auth/register`
* `GET /api/admin/users`
* `POST /api/admin/users`
* `PATCH /api/admin/users/:user_id`
* `DELETE /api/admin/users/:user_id`

#### Settings

* `GET /api/settings`
* `PATCH /api/settings`
* `PATCH /api/settings/network`
* `POST /api/settings/network-recovery-notice/dismiss`
* `GET /api/user-settings`
* `PATCH /api/user-settings`

#### Ollama backends and models

* `GET /api/backends`
* `POST /api/backends`
* `PATCH /api/backends/:backend_id`
* `DELETE /api/backends/:backend_id`
* `POST /api/backends/detect-localhost`
* `POST /api/backends/scan-local-network`
* `GET /api/models`
* `GET /api/user-models`
* `POST /api/user-models/refresh`
* `PATCH /api/user-models`
* `GET /api/admin/models`
* `POST /api/admin/models/refresh`
* `PATCH /api/admin/models`
* `PATCH /api/admin/models/backend`
* `PATCH /api/admin/models/tags`
* `PATCH /api/admin/models/default-tags`
* `GET /api/models/status`

#### Standard chats

* `GET /api/chats`
* `POST /api/chats`
* `GET /api/chats/:chat_id`
* `PATCH /api/chats/:chat_id`
* `DELETE /api/chats/:chat_id`
* `GET /api/chats/:chat_id/messages`
* `POST /api/chats/:chat_id/messages`

#### Standard chat streaming generation

* `POST /api/chats/:chat_id/generate`

#### Standard chat attachments

* `POST /api/chats/:chat_id/attachments`
* `GET /api/attachments/:attachment_id`
* `DELETE /api/attachments/:attachment_id`

#### Private-local support

Private-local chats should avoid server persistence of content. The server API for this mode should be narrower.

Possible initial direction:

* no CRUD persistence endpoints for private-local content
* use a direct generation endpoint for transient requests, such as:
* `POST /api/private/generate`

This endpoint would:

* accept transient client-supplied message history
* forward it to Ollama
* stream the response back
* avoid storing prompt/response content in the database

This lets the frontend keep private-local chat history in IndexedDB while still using the same backend for live generation.

### 19.3 Recommended simplification for private-local mode

Instead of trying to make private-local chats look exactly like server-backed chats at the API level, treat them as a separate frontend storage mode that reuses generation and model APIs.

That means:

* standard chats use server IDs and server persistence
* private-local chats use client-generated IDs in IndexedDB
* private-local generation requests include the relevant local history payload when needed
* server remains stateless with respect to private-local content

This is simpler and more honest than pretending the server has authoritative records for private-local chats.

### 19.4 Likely next design artifact

The next document should define:

* exact table schemas
* exact request/response payloads
* exact frontend screens and their data dependencies
* startup flow and auth flow
* how standard and private-local chats differ in the UI and API
* how multiple Ollama backends are represented in the database and model picker

### 19.5 Multi-backend UI note

The model picker should not present one flat list of model names. It should group models by backend, for example:

```text
mac-mini
- gemma4
- llama3:70b

rtx-3060
- gemma4
- qwen3.5
- gpt-oss:20b
```

This keeps server choice and model choice unified in one interaction while still making it obvious which machine will handle the request.
