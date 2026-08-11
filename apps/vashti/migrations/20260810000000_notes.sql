CREATE TABLE notes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    current_version_id TEXT NOT NULL,
    ai_access TEXT NOT NULL DEFAULT 'none'
        CHECK (ai_access IN ('none', 'read', 'edit', 'manage')),
    all_models INTEGER NOT NULL DEFAULT 1 CHECK (all_models IN (0, 1)),
    is_pinned INTEGER NOT NULL DEFAULT 0 CHECK (is_pinned IN (0, 1)),
    deleted_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX idx_notes_user_state_updated
    ON notes(user_id, deleted_at, is_pinned DESC, updated_at DESC);

CREATE TABLE note_versions (
    id TEXT PRIMARY KEY,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    actor_type TEXT NOT NULL CHECK (actor_type IN ('human', 'model')),
    actor_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    actor_model_key TEXT,
    actor_model_name TEXT,
    source_chat_id TEXT,
    source_message_id TEXT,
    source_tool_call_id TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE(note_id, version_number)
);

CREATE INDEX idx_note_versions_note_number
    ON note_versions(note_id, version_number DESC);

CREATE TABLE note_tags (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX idx_note_tags_user_name
    ON note_tags(user_id, name COLLATE NOCASE);

CREATE TABLE note_tag_links (
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES note_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, tag_id)
);

CREATE INDEX idx_note_tag_links_tag
    ON note_tag_links(tag_id, note_id);

CREATE TABLE note_model_scopes (
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    PRIMARY KEY (note_id, model_key)
);

CREATE INDEX idx_note_model_scopes_model
    ON note_model_scopes(model_key, note_id);

CREATE TABLE user_note_settings (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    allow_model_read INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_read IN (0, 1)),
    allow_model_create INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_create IN (0, 1)),
    allow_model_edit INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_edit IN (0, 1)),
    allow_model_trash INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_trash IN (0, 1)),
    default_ai_access TEXT NOT NULL DEFAULT 'none'
        CHECK (default_ai_access IN ('none', 'read', 'edit', 'manage')),
    default_all_models INTEGER NOT NULL DEFAULT 1 CHECK (default_all_models IN (0, 1)),
    updated_at INTEGER NOT NULL
);

CREATE TABLE user_note_default_model_scopes (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    PRIMARY KEY (user_id, model_key)
);

CREATE VIRTUAL TABLE notes_fts USING fts5(
    note_id UNINDEXED,
    user_id UNINDEXED,
    title,
    content,
    tags,
    tokenize = 'unicode61 remove_diacritics 2'
);
