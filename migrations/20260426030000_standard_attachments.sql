CREATE TABLE attachments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
    revision_id TEXT REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    attachment_kind TEXT NOT NULL CHECK (attachment_kind IN ('image', 'text')),
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_attachments_chat ON attachments(chat_id);
CREATE INDEX idx_attachments_message ON attachments(message_id);
CREATE INDEX idx_attachments_revision ON attachments(revision_id);
CREATE INDEX idx_attachments_user_pending ON attachments(user_id, chat_id, message_id, revision_id);
