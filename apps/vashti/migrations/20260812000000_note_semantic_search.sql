ALTER TABLE app_settings
    ADD COLUMN notes_semantic_search_enabled INTEGER NOT NULL DEFAULT 0
        CHECK (notes_semantic_search_enabled IN (0, 1));

ALTER TABLE app_settings
    ADD COLUMN notes_embedding_backend_id TEXT REFERENCES ollama_backends(id) ON DELETE SET NULL;

ALTER TABLE app_settings
    ADD COLUMN notes_embedding_model TEXT;

ALTER TABLE app_settings
    ADD COLUMN notes_embedding_last_error TEXT;

CREATE TABLE note_search_chunks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    note_version_id TEXT NOT NULL REFERENCES note_versions(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_text_hash TEXT NOT NULL,
    embedding_backend_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    chunker_version INTEGER NOT NULL,
    vector BLOB NOT NULL,
    indexed_at INTEGER NOT NULL,
    UNIQUE(note_id, chunk_index)
);

CREATE INDEX idx_note_search_chunks_user_version
    ON note_search_chunks(user_id, note_version_id, chunk_index);

CREATE TABLE note_embedding_jobs (
    note_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    note_version_id TEXT NOT NULL REFERENCES note_versions(id) ON DELETE CASCADE,
    requested_at INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    last_error TEXT
);

CREATE INDEX idx_note_embedding_jobs_requested
    ON note_embedding_jobs(requested_at, attempts);

CREATE TABLE note_embedding_revisions (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL DEFAULT 0
);

CREATE TRIGGER note_version_enqueue_embedding
AFTER INSERT ON note_versions
WHEN EXISTS (
    SELECT 1
    FROM notes
    WHERE notes.id = NEW.note_id
      AND notes.current_version_id = NEW.id
      AND notes.deleted_at IS NULL
)
BEGIN
    INSERT INTO note_embedding_jobs (
        note_id, user_id, note_version_id, requested_at, attempts, last_attempt_at, last_error
    )
    SELECT id, user_id, current_version_id, updated_at, 0, NULL, NULL
    FROM notes
    WHERE id = NEW.note_id
    ON CONFLICT(note_id) DO UPDATE SET
        user_id = excluded.user_id,
        note_version_id = excluded.note_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO note_embedding_revisions (user_id, revision)
    SELECT user_id, 1 FROM notes WHERE id = NEW.note_id
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER note_current_version_enqueue_embedding
AFTER UPDATE OF current_version_id ON notes
WHEN NEW.deleted_at IS NULL AND NEW.current_version_id != OLD.current_version_id
BEGIN
    INSERT INTO note_embedding_jobs (
        note_id, user_id, note_version_id, requested_at, attempts, last_attempt_at, last_error
    ) VALUES (NEW.id, NEW.user_id, NEW.current_version_id, NEW.updated_at, 0, NULL, NULL)
    ON CONFLICT(note_id) DO UPDATE SET
        user_id = excluded.user_id,
        note_version_id = excluded.note_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO note_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER note_restore_enqueue_embedding
AFTER UPDATE OF deleted_at ON notes
WHEN OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS NULL
BEGIN
    INSERT INTO note_embedding_jobs (
        note_id, user_id, note_version_id, requested_at, attempts, last_attempt_at, last_error
    ) VALUES (NEW.id, NEW.user_id, NEW.current_version_id, NEW.updated_at, 0, NULL, NULL)
    ON CONFLICT(note_id) DO UPDATE SET
        user_id = excluded.user_id,
        note_version_id = excluded.note_version_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO note_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER note_trash_remove_embeddings
AFTER UPDATE OF deleted_at ON notes
WHEN OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
BEGIN
    DELETE FROM note_embedding_jobs WHERE note_id = NEW.id;
    DELETE FROM note_search_chunks WHERE note_id = NEW.id;

    INSERT INTO note_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

INSERT INTO note_embedding_jobs (
    note_id, user_id, note_version_id, requested_at, attempts, last_attempt_at, last_error
)
SELECT id, user_id, current_version_id, updated_at, 0, NULL, NULL
FROM notes
WHERE deleted_at IS NULL;

INSERT INTO note_embedding_revisions (user_id, revision)
SELECT DISTINCT user_id, 1
FROM notes
WHERE 1 = 1
ON CONFLICT(user_id) DO NOTHING;
