CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    default_backend_id TEXT NOT NULL REFERENCES ollama_backends(id) ON DELETE RESTRICT,
    default_model_name TEXT NOT NULL,
    title TEXT NOT NULL,
    chat_mode TEXT NOT NULL DEFAULT 'standard' CHECK (chat_mode IN ('standard')),
    active_root_message_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_message_at INTEGER NOT NULL,
    archived_at INTEGER
);

CREATE INDEX idx_chats_user_updated ON chats(user_id, updated_at DESC);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    parent_message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
    active_child_message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
    active_revision_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    status TEXT NOT NULL DEFAULT 'complete' CHECK (status IN ('complete', 'streaming', 'stopped', 'error')),
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    backend_id TEXT REFERENCES ollama_backends(id) ON DELETE SET NULL,
    model_name TEXT,
    think_mode TEXT,
    done_reason TEXT,
    error_text TEXT,
    started_at INTEGER,
    completed_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX idx_chat_messages_chat_created ON chat_messages(chat_id, created_at);
CREATE INDEX idx_chat_messages_parent ON chat_messages(chat_id, parent_message_id);
CREATE INDEX idx_chat_messages_active_revision ON chat_messages(active_revision_id);

CREATE TABLE chat_message_revisions (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    content_text TEXT NOT NULL DEFAULT '',
    thinking_text TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL CHECK (source IN ('original', 'edit', 'regeneration')),
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_chat_message_revisions_message_created ON chat_message_revisions(message_id, created_at);
