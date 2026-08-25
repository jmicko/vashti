ALTER TABLE user_memory_settings
    ADD COLUMN allow_model_chat_history INTEGER NOT NULL DEFAULT 0
        CHECK (allow_model_chat_history IN (0, 1));

CREATE TABLE conversation_search_documents (
    message_id TEXT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    chat_title TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    source_updated_at INTEGER NOT NULL,
    indexed_at INTEGER NOT NULL
);

CREATE INDEX idx_conversation_search_documents_user_updated
    ON conversation_search_documents(user_id, source_updated_at DESC);

CREATE INDEX idx_conversation_search_documents_chat
    ON conversation_search_documents(chat_id, source_updated_at);

CREATE VIRTUAL TABLE conversation_search_fts USING fts5(
    chat_title,
    content,
    content = 'conversation_search_documents',
    content_rowid = 'rowid',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER conversation_search_documents_fts_insert
AFTER INSERT ON conversation_search_documents
BEGIN
    INSERT INTO conversation_search_fts(rowid, chat_title, content)
    VALUES (NEW.rowid, NEW.chat_title, NEW.content);
END;

CREATE TRIGGER conversation_search_documents_fts_delete
AFTER DELETE ON conversation_search_documents
BEGIN
    INSERT INTO conversation_search_fts(
        conversation_search_fts, rowid, chat_title, content
    ) VALUES ('delete', OLD.rowid, OLD.chat_title, OLD.content);
END;

CREATE TRIGGER conversation_search_documents_fts_update
AFTER UPDATE ON conversation_search_documents
BEGIN
    INSERT INTO conversation_search_fts(
        conversation_search_fts, rowid, chat_title, content
    ) VALUES ('delete', OLD.rowid, OLD.chat_title, OLD.content);
    INSERT INTO conversation_search_fts(rowid, chat_title, content)
    VALUES (NEW.rowid, NEW.chat_title, NEW.content);
END;

CREATE TABLE conversation_embeddings (
    message_id TEXT PRIMARY KEY
        REFERENCES conversation_search_documents(message_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    embedding_backend_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    vector BLOB NOT NULL,
    indexed_at INTEGER NOT NULL
);

CREATE INDEX idx_conversation_embeddings_user_revision
    ON conversation_embeddings(user_id, revision_id);

CREATE TABLE conversation_embedding_jobs (
    message_id TEXT PRIMARY KEY
        REFERENCES conversation_search_documents(message_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    requested_at INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    last_error TEXT
);

CREATE INDEX idx_conversation_embedding_jobs_requested
    ON conversation_embedding_jobs(requested_at, attempts);

CREATE TABLE conversation_embedding_revisions (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL DEFAULT 0
);

CREATE TRIGGER conversation_document_enqueue_embedding
AFTER INSERT ON conversation_search_documents
BEGIN
    INSERT INTO conversation_embedding_jobs (
        message_id, user_id, revision_id, requested_at,
        attempts, last_attempt_at, last_error
    ) VALUES (
        NEW.message_id, NEW.user_id, NEW.revision_id, NEW.source_updated_at,
        0, NULL, NULL
    )
    ON CONFLICT(message_id) DO UPDATE SET
        user_id = excluded.user_id,
        revision_id = excluded.revision_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    INSERT INTO conversation_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER conversation_document_refresh_embedding
AFTER UPDATE OF revision_id, chat_title, role, content ON conversation_search_documents
BEGIN
    INSERT INTO conversation_embedding_jobs (
        message_id, user_id, revision_id, requested_at,
        attempts, last_attempt_at, last_error
    ) VALUES (
        NEW.message_id, NEW.user_id, NEW.revision_id, NEW.source_updated_at,
        0, NULL, NULL
    )
    ON CONFLICT(message_id) DO UPDATE SET
        user_id = excluded.user_id,
        revision_id = excluded.revision_id,
        requested_at = excluded.requested_at,
        attempts = 0,
        last_attempt_at = NULL,
        last_error = NULL;

    DELETE FROM conversation_embeddings WHERE message_id = NEW.message_id;

    INSERT INTO conversation_embedding_revisions (user_id, revision)
    VALUES (NEW.user_id, 1)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER conversation_document_remove_embedding
AFTER DELETE ON conversation_search_documents
BEGIN
    INSERT INTO conversation_embedding_revisions (user_id, revision)
    SELECT OLD.user_id, 1
    WHERE EXISTS (SELECT 1 FROM users WHERE id = OLD.user_id)
    ON CONFLICT(user_id) DO UPDATE SET revision = revision + 1;
END;

CREATE TRIGGER conversation_message_insert_search_document
AFTER INSERT ON chat_messages
BEGIN
    INSERT INTO conversation_search_documents (
        message_id, chat_id, user_id, revision_id, chat_title,
        role, content, source_updated_at, indexed_at
    )
    SELECT NEW.id, NEW.chat_id, chats.user_id, revisions.id, chats.title,
           NEW.role, revisions.content_text, NEW.updated_at, unixepoch()
    FROM chats
    JOIN chat_message_revisions revisions ON revisions.id = NEW.active_revision_id
    WHERE chats.id = NEW.chat_id
      AND NEW.role IN ('user', 'assistant')
      AND NEW.status = 'complete'
      AND NEW.is_deleted = 0
      AND trim(revisions.content_text) != ''
    ON CONFLICT(message_id) DO UPDATE SET
        chat_id = excluded.chat_id,
        user_id = excluded.user_id,
        revision_id = excluded.revision_id,
        chat_title = excluded.chat_title,
        role = excluded.role,
        content = excluded.content,
        source_updated_at = excluded.source_updated_at,
        indexed_at = excluded.indexed_at;
END;

CREATE TRIGGER conversation_revision_insert_search_document
AFTER INSERT ON chat_message_revisions
BEGIN
    INSERT INTO conversation_search_documents (
        message_id, chat_id, user_id, revision_id, chat_title,
        role, content, source_updated_at, indexed_at
    )
    SELECT messages.id, messages.chat_id, chats.user_id, NEW.id, chats.title,
           messages.role, NEW.content_text, messages.updated_at, unixepoch()
    FROM chat_messages messages
    JOIN chats ON chats.id = messages.chat_id
    WHERE messages.id = NEW.message_id
      AND messages.active_revision_id = NEW.id
      AND messages.role IN ('user', 'assistant')
      AND messages.status = 'complete'
      AND messages.is_deleted = 0
      AND trim(NEW.content_text) != ''
    ON CONFLICT(message_id) DO UPDATE SET
        chat_id = excluded.chat_id,
        user_id = excluded.user_id,
        revision_id = excluded.revision_id,
        chat_title = excluded.chat_title,
        role = excluded.role,
        content = excluded.content,
        source_updated_at = excluded.source_updated_at,
        indexed_at = excluded.indexed_at;
END;

CREATE TRIGGER conversation_revision_update_search_document
AFTER UPDATE OF content_text ON chat_message_revisions
WHEN NEW.content_text != OLD.content_text
BEGIN
    UPDATE conversation_search_documents
    SET content = NEW.content_text,
        source_updated_at = (
            SELECT updated_at FROM chat_messages WHERE id = NEW.message_id
        ),
        indexed_at = unixepoch()
    WHERE message_id = NEW.message_id
      AND revision_id = NEW.id;
END;

CREATE TRIGGER conversation_message_update_search_document
AFTER UPDATE OF active_revision_id, status, is_deleted, updated_at ON chat_messages
BEGIN
    DELETE FROM conversation_search_documents
    WHERE message_id = NEW.id
      AND NOT EXISTS (
          SELECT 1
          FROM chat_message_revisions revisions
          WHERE revisions.id = NEW.active_revision_id
            AND NEW.role IN ('user', 'assistant')
            AND NEW.status = 'complete'
            AND NEW.is_deleted = 0
            AND trim(revisions.content_text) != ''
      );

    INSERT INTO conversation_search_documents (
        message_id, chat_id, user_id, revision_id, chat_title,
        role, content, source_updated_at, indexed_at
    )
    SELECT NEW.id, NEW.chat_id, chats.user_id, revisions.id, chats.title,
           NEW.role, revisions.content_text, NEW.updated_at, unixepoch()
    FROM chats
    JOIN chat_message_revisions revisions ON revisions.id = NEW.active_revision_id
    WHERE chats.id = NEW.chat_id
      AND NEW.role IN ('user', 'assistant')
      AND NEW.status = 'complete'
      AND NEW.is_deleted = 0
      AND trim(revisions.content_text) != ''
    ON CONFLICT(message_id) DO UPDATE SET
        chat_id = excluded.chat_id,
        user_id = excluded.user_id,
        revision_id = excluded.revision_id,
        chat_title = excluded.chat_title,
        role = excluded.role,
        content = excluded.content,
        source_updated_at = excluded.source_updated_at,
        indexed_at = excluded.indexed_at
    WHERE conversation_search_documents.revision_id != excluded.revision_id
       OR conversation_search_documents.content != excluded.content
       OR conversation_search_documents.source_updated_at != excluded.source_updated_at;
END;

CREATE TRIGGER conversation_chat_title_search_document
AFTER UPDATE OF title ON chats
WHEN NEW.title != OLD.title
BEGIN
    UPDATE conversation_search_documents
    SET chat_title = NEW.title,
        indexed_at = unixepoch()
    WHERE chat_id = NEW.id;
END;

INSERT INTO conversation_search_documents (
    message_id, chat_id, user_id, revision_id, chat_title,
    role, content, source_updated_at, indexed_at
)
SELECT messages.id, messages.chat_id, chats.user_id, revisions.id, chats.title,
       messages.role, revisions.content_text, messages.updated_at, unixepoch()
FROM chat_messages messages
JOIN chats ON chats.id = messages.chat_id
JOIN chat_message_revisions revisions ON revisions.id = messages.active_revision_id
WHERE messages.role IN ('user', 'assistant')
  AND messages.status = 'complete'
  AND messages.is_deleted = 0
  AND trim(revisions.content_text) != '';
