ALTER TABLE chat_messages ADD COLUMN import_source TEXT;

CREATE TABLE chat_import_records (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL UNIQUE REFERENCES chats(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    original_json TEXT NOT NULL,
    allow_history_search INTEGER NOT NULL DEFAULT 0 CHECK (allow_history_search IN (0,1)),
    created_at INTEGER NOT NULL,
    UNIQUE(user_id,source_id)
);

-- Preserve the legacy avatar foreign keys while widening the attachment kinds.
CREATE TEMP TABLE import_avatar_links AS
    SELECT id,avatar_attachment_id FROM persona_versions WHERE avatar_attachment_id IS NOT NULL;
CREATE TABLE attachments_import_upgrade (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
    revision_id TEXT REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    attachment_kind TEXT NOT NULL CHECK (attachment_kind IN ('image','text','file')),
    created_at INTEGER NOT NULL,
    image_asset_id TEXT REFERENCES image_assets(id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED
);
INSERT INTO attachments_import_upgrade SELECT * FROM attachments;
DROP TABLE attachments;
ALTER TABLE attachments_import_upgrade RENAME TO attachments;
CREATE INDEX idx_attachments_chat ON attachments(chat_id);
CREATE INDEX idx_attachments_message ON attachments(message_id);
CREATE INDEX idx_attachments_revision ON attachments(revision_id);
CREATE INDEX idx_attachments_user_pending ON attachments(user_id,chat_id,message_id,revision_id);
CREATE INDEX attachment_image_asset ON attachments(image_asset_id);
UPDATE persona_versions SET avatar_attachment_id = (
    SELECT avatar_attachment_id FROM import_avatar_links WHERE import_avatar_links.id=persona_versions.id
) WHERE id IN (SELECT id FROM import_avatar_links);
DROP TABLE import_avatar_links;

CREATE TABLE chat_import_assets (
    id TEXT PRIMARY KEY,
    import_id TEXT NOT NULL REFERENCES chat_import_records(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes BETWEEN 1 AND 134217728),
    uploaded_bytes INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0,1)),
    UNIQUE(import_id,source_id)
);
