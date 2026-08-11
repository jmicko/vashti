CREATE TABLE chat_pinned_notes (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    note_version_id TEXT NOT NULL REFERENCES note_versions(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    PRIMARY KEY (chat_id, note_id),
    UNIQUE (chat_id, position)
);

CREATE INDEX idx_chat_pinned_notes_chat_position
    ON chat_pinned_notes(chat_id, position);

CREATE TABLE note_message_attachments (
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL,
    note_version_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    title_snapshot TEXT NOT NULL,
    content_snapshot TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('explicit', 'pinned')),
    position INTEGER NOT NULL,
    PRIMARY KEY (message_id, note_id),
    UNIQUE (message_id, position)
);

CREATE INDEX idx_note_message_attachments_message_position
    ON note_message_attachments(message_id, position);
