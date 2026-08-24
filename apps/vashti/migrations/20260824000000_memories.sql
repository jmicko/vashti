CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    current_version_id TEXT NOT NULL,
    all_models INTEGER NOT NULL DEFAULT 1 CHECK (all_models IN (0, 1)),
    deleted_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX idx_memories_user_state_updated
    ON memories(user_id, deleted_at, updated_at DESC);

CREATE TABLE memory_versions (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    actor_type TEXT NOT NULL CHECK (actor_type IN ('human', 'model')),
    actor_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    actor_model_key TEXT,
    actor_model_name TEXT,
    source_chat_id TEXT,
    source_message_id TEXT,
    source_tool_call_id TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE(memory_id, version_number)
);

CREATE INDEX idx_memory_versions_memory_number
    ON memory_versions(memory_id, version_number DESC);

CREATE TABLE memory_model_scopes (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    PRIMARY KEY (memory_id, model_key)
);

CREATE INDEX idx_memory_model_scopes_model
    ON memory_model_scopes(model_key, memory_id);

CREATE TABLE user_memory_settings (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    allow_model_read INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_read IN (0, 1)),
    allow_model_create INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_create IN (0, 1)),
    allow_model_edit INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_edit IN (0, 1)),
    allow_model_forget INTEGER NOT NULL DEFAULT 0 CHECK (allow_model_forget IN (0, 1)),
    updated_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    memory_id UNINDEXED,
    user_id UNINDEXED,
    content,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE memory_embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_version_id TEXT NOT NULL REFERENCES memory_versions(id) ON DELETE CASCADE,
    embedding_backend_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    vector BLOB NOT NULL,
    indexed_at INTEGER NOT NULL
);

CREATE INDEX idx_memory_embeddings_user_version
    ON memory_embeddings(user_id, memory_version_id);

CREATE TABLE memory_embedding_jobs (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_version_id TEXT NOT NULL REFERENCES memory_versions(id) ON DELETE CASCADE,
    requested_at INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    last_error TEXT
);

CREATE INDEX idx_memory_embedding_jobs_requested
    ON memory_embedding_jobs(requested_at, attempts);

CREATE TABLE memory_embedding_revisions (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL DEFAULT 0
);

CREATE TRIGGER memory_version_enqueue_embedding
AFTER INSERT ON memory_versions
WHEN EXISTS (
    SELECT 1
    FROM memories
    WHERE memories.id = NEW.memory_id
      AND memories.current_version_id = NEW.id
      AND memories.deleted_at IS NULL
)
BEGIN
    INSERT INTO memory_embedding_jobs (
        memory_id, user_id, memory_version_id, requested_at,
        attempts, last_attempt_at, last_error
    )
    SELECT id, user_id, current_version_id, updated_at, 0, NULL, NULL
    FROM memories
    WHERE id = NEW.memory_id
    ON CONFLICT(memory_id) DO UPDATE SET
        user_id = excluded.user_id,
        memory_version_id = excluded.memory_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO memory_embedding_revisions (user_id, revision)
    SELECT user_id, 1 FROM memories WHERE id = NEW.memory_id
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER memory_current_version_enqueue_embedding
AFTER UPDATE OF current_version_id ON memories
WHEN NEW.deleted_at IS NULL AND NEW.current_version_id != OLD.current_version_id
BEGIN
    INSERT INTO memory_embedding_jobs (
        memory_id, user_id, memory_version_id, requested_at,
        attempts, last_attempt_at, last_error
    ) VALUES (NEW.id, NEW.user_id, NEW.current_version_id, NEW.updated_at, 0, NULL, NULL)
    ON CONFLICT(memory_id) DO UPDATE SET
        user_id = excluded.user_id,
        memory_version_id = excluded.memory_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO memory_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER memory_restore_enqueue_embedding
AFTER UPDATE OF deleted_at ON memories
WHEN OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS NULL
BEGIN
    INSERT INTO memory_embedding_jobs (
        memory_id, user_id, memory_version_id, requested_at,
        attempts, last_attempt_at, last_error
    ) VALUES (NEW.id, NEW.user_id, NEW.current_version_id, NEW.updated_at, 0, NULL, NULL)
    ON CONFLICT(memory_id) DO UPDATE SET
        user_id = excluded.user_id,
        memory_version_id = excluded.memory_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO memory_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER memory_forget_remove_embedding
AFTER UPDATE OF deleted_at ON memories
WHEN OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
BEGIN
    DELETE FROM memory_embedding_jobs WHERE memory_id = NEW.id;
    DELETE FROM memory_embeddings WHERE memory_id = NEW.id;

    INSERT INTO memory_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;
