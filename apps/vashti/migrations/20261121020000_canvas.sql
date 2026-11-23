CREATE TABLE canvas_documents (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL
);
CREATE INDEX canvas_documents_chat ON canvas_documents(chat_id);

CREATE TABLE canvas_versions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES canvas_documents(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES canvas_versions(id) ON DELETE SET NULL,
    version_number INTEGER NOT NULL,
    title TEXT NOT NULL,
    language TEXT NOT NULL,
    content TEXT NOT NULL,
    actor TEXT NOT NULL CHECK(actor IN ('human','model')),
    model_name TEXT,
    edit_session_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE(document_id,version_number)
);
CREATE INDEX canvas_versions_document ON canvas_versions(document_id,version_number);

CREATE TABLE canvas_scopes (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    revision_id TEXT UNIQUE REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    heads_json TEXT NOT NULL DEFAULT '[]',
    updated_at INTEGER NOT NULL
);
CREATE INDEX canvas_scopes_chat ON canvas_scopes(chat_id);

CREATE TABLE canvas_generation_inputs (
    revision_id TEXT PRIMARY KEY REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    heads_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE canvas_comments (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES canvas_scopes(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES canvas_documents(id) ON DELETE CASCADE,
    version_id TEXT NOT NULL REFERENCES canvas_versions(id) ON DELETE CASCADE,
    quote TEXT NOT NULL,
    content TEXT NOT NULL,
    actor TEXT NOT NULL CHECK(actor IN ('human','model')),
    model_name TEXT,
    resolved INTEGER NOT NULL DEFAULT 0 CHECK(resolved IN (0,1)),
    created_at INTEGER NOT NULL
);
CREATE INDEX canvas_comments_scope ON canvas_comments(scope_id,document_id);
