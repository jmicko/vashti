# Plan: Profile pictures for custom models

## Goals
- Let users upload, crop, and assign a profile picture to a custom model (persona) from the edit view.
- Show the picture next to the model name in chat bubbles, the model picker, and the sidebar/chat list.
- Cover both server personas and device (private) personas in this pass.
- Stay out of the regular-model edit flow for now — no schema or UI work for built-in Ollama models.

## Decisions
- **Backend re-encodes uploads to WebP**, max 512×512, max 2 MB input. One consistent render path, predictable storage.
- **Avatars live in the existing `attachments` table** with two new nullable columns (`persona_version_id`, plus make `chat_id` nullable) and a new `attachment_kind = 'persona_avatar'`.
- **`react-easy-crop` for the cropper.**
- **Device personas get avatars too** — store the blob in the existing IndexedDB-backed `privateChatStore`, upload on save (or lazily), and reuse the same `<ModelAvatar>` component.
- **Cache avatar bytes in IndexedDB** keyed by `attachment_id`. First render fetches and seeds the cache; every subsequent render is instant. The persona collection itself is already cached client-side, so list loads are instant. We're optimizing for "no image pop-in on every chat render."
- **Carry forward avatar on edit**, allow explicit `null` to remove. Editing shouldn't drop the picture.

## Why custom models only
The `persona_versions.avatar_attachment_id` column, the create/update payload, and `PersonaVersionResponse` are already plumbed end-to-end server-side. The frontend just hardcodes `null` and never lets the user pick anything. So the work is "fill in the missing UI + an upload endpoint" — small, contained, with no risk of leaking another user's image. Regular models have no per-model edit form yet; building that whole flow just to host an avatar would be a detour.

## Backend

### Migration (`apps/vashti/migrations/20260609000000_persona_avatars.sql`)
- `ALTER TABLE attachments ADD COLUMN persona_version_id TEXT REFERENCES persona_versions(id) ON DELETE SET NULL;`
- `ALTER TABLE attachments ADD COLUMN attachment_kind TEXT NOT NULL DEFAULT 'chat';` (CHECK allowing `chat` and `persona_avatar`)
- `CREATE INDEX idx_attachments_persona_version ON attachments(persona_version_id);`
- Make `attachments.chat_id` nullable (avatars aren't tied to a chat).
- The `current_version_id` machinery already updates on create/edit, so the new `avatar_attachment_id` on the new version flows naturally.

### Image processing
- Add a new dep: `image = { version = "0.25", default-features = false, features = ["webp", "png", "jpeg"] }`.
- New `apps/vashti/src/uploads/image.rs`:
  - `decode_and_normalize(bytes: &[u8]) -> Result<Vec<u8>, ApiError>`
  - Decode → resize so the longest side is ≤ 512, preserving aspect ratio → re-encode as WebP, quality 85.
  - Cap input at 2 MB; reject non-images by sniffing magic bytes.
  - Returns the encoded WebP bytes (we'll set `mime_type` to `image/webp`).

### Endpoints
- `POST /api/personas/{persona_id}/avatar` — multipart, single `file` field. Auth + owner-only. Re-encodes, stores a new `attachment_kind = 'persona_avatar'` row linked to the persona's current version, updates `current_version.avatar_attachment_id`, and deletes the old avatar attachment (if any) in the same transaction. Returns `{ attachment_id }`.
- `DELETE /api/personas/{persona_id}/avatar` — auth + owner-only. Clears `current_version.avatar_attachment_id` and deletes the attachment row.
- `GET /api/attachments/{id}` — extend the auth check to allow access when the attachment's `attachment_kind = 'persona_avatar'` and the requester is the persona owner or a member. Today the handler is keyed on chat ownership; we extend that check.

### Persona copy / version flow
- `POST /api/personas/{id}/copy`: when copying a version to a new persona, the new version gets its own attachment row pointing at the same on-disk bytes (cheap, and disowning one persona doesn't dangle another's avatar).
- When the user edits and we insert a new `persona_version` row, carry forward `avatar_attachment_id` from the prior version unless the payload explicitly sets it to `null` (the user removed it).

### Service layer
- `apps/vashti/src/personas/service.rs`:
  - `update_persona`: insert new version carrying forward `avatar_attachment_id` by default; respect explicit `null` from the payload.
  - `create_persona`: accept the optional `avatar_attachment_id` from the payload (already in the request shape — we just stop hardcoding `null`).

## Frontend

### Dep
- Add `react-easy-crop` to `apps/vashti/web/package.json`.

### New components
- `web/src/avatarCache.ts`:
  - Tiny IndexedDB wrapper: a single `avatars` object store keyed by `attachment_id`, holding the `Blob`.
  - `get(id) → Promise<Blob | null>`, `put(id, blob)`, `delete(id)`.
  - `getObjectUrl(id) → Promise<string | null>`: returns a fresh `URL.createObjectURL(blob)`; the caller is responsible for `URL.revokeObjectURL` on unmount.
- `web/src/ModelAvatar.tsx`:
  - Props: `attachmentId?: string | null`, `displayName: string`, `size?: number` (default 32), `blobUrl?: string | null` (for device-persona blobs already on the client).
  - Reads from `avatarCache` first. On miss, fetches `/api/attachments/{id}`, stores the blob, and uses its object URL.
  - If no `attachmentId`, renders a colored initial circle. Color is derived from a stable hash of `displayName` so it's consistent across renders.
- `web/src/PersonaAvatarField.tsx`:
  - File input + dropzone styled to fit the existing form.
  - Shows current avatar (from props) with "Replace" and "Remove" buttons.
  - On file pick, opens `<PersonaAvatarCropModal>` with the picked file.
  - On crop confirm, returns a `Blob` via callback.
  - "Remove" sets the field's local state to "cleared," which the parent form turns into `avatar_attachment_id: null` on save.
- `web/src/PersonaAvatarCropModal.tsx`:
  - Wraps `react-easy-crop`. Square 1:1 aspect. Zoom slider. Outputs a WebP `Blob` at 512×512 via canvas.
  - Confirm/cancel buttons.

### Wiring into `settingsPersonas.tsx`
- `CustomModelsSection` adds `avatarBlob: Blob | null`, `removeAvatar: boolean` to its draft state.
- `startEditingPersona` / `startEditingPrivatePersona` seed initial avatar from `version.avatar_attachment_id`.
- `savePersona` flow:
  - If `editingPersona` (server):
    1. If `removeAvatar`: `DELETE /api/personas/{id}/avatar`.
    2. Else if `avatarBlob`: `POST /api/personas/{id}/avatar` and stash the returned `attachment_id`.
    3. PATCH the persona with the new `avatar_attachment_id` (or `null` for the cleared case).
  - If `editingPrivatePersona` (device): store the `avatarBlob` directly in `privateChatStore` and skip the upload.
  - On create: upload (server) or store (device) first, then POST with the id.

### `privateChatStore` (device personas)
- Add an `avatar_blob_id?: string` to stored private personas and a small `avatar_blobs` object store in IndexedDB keyed by blob id, holding the raw `Blob`. (Reuses the same IndexedDB the store already opens.)
- `createPrivatePersona` / `updatePrivatePersona` accept an optional `avatarBlob`. The component handles upload-vs-store.
- `<ModelAvatar>` learns a `blobUrl` prop so it can render directly from a stored blob without an HTTP fetch — and that path also seeds `avatarCache` so subsequent loads of the same persona (if it's ever synced) hit the cache.

### Rendering in chat
- `MessageBubble.tsx`: when `message.role === "assistant"` and the chat is bound to a persona, show `<ModelAvatar>` next to the role label. Look up the persona by `message.persona_id` (already on the message per the migration). If the persona is missing/disowned, fall back to text only.
- `ModelPicker.tsx`: custom-model rows show a small `<ModelAvatar>` next to the name.
- `Sidebar.tsx` / `ChatHome.tsx`: chat list rows that have a persona show a tiny avatar next to the chat title.

## Edge cases / non-goals
- Avatars are purely cosmetic. Old chat history without a stored persona id is unaffected.
- Rate limiting on the upload endpoint: 30 / hour per user (same `user_action_key("persona_avatar_upload", ...)` pattern as elsewhere).
- No public CDN caching. Avatars go through the authenticated attachment route like chat attachments.
- The `dev/private-stream-test` route is unrelated; left alone.
- Stale avatar references in `chat_messages.persona_name_snapshot`: snapshots stay text-only. We do not need to backfill avatars in old messages.
- When a persona is disowned/deleted, `get_attachment` returns 404 for the avatar. `<ModelAvatar>` falls back to the initial circle. No special handling needed at the chat level.

## Order of work
1. Backend migration.
2. Image-processing helper + tests.
3. Avatar upload/delete endpoints.
4. Update `get_attachment` auth to allow persona-avatar access.
5. Service-layer: carry avatar through version copy / edit.
6. Frontend dep + `avatarCache`.
7. `ModelAvatar` component.
8. `PersonaAvatarField` + crop modal.
9. Hook into `settingsPersonas.tsx`.
10. `privateChatStore` blob storage.
11. Render avatars in chat, picker, sidebar.
12. Release notes.
