CREATE TABLE image_service_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    backend_id TEXT REFERENCES ollama_backends(id) ON DELETE SET NULL,
    token TEXT NOT NULL DEFAULT '',
    max_user_bytes INTEGER NOT NULL DEFAULT 1073741824 CHECK (max_user_bytes > 0)
);
INSERT INTO image_service_settings (id) VALUES (1);

CREATE TABLE image_preferences (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    value_json TEXT NOT NULL
);

CREATE TABLE image_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('generate', 'iterative', 'original')),
    draft_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX image_sessions_owner ON image_sessions(user_id, updated_at DESC);

CREATE TABLE image_prompt_versions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES image_sessions(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES image_prompt_versions(id) ON DELETE SET NULL,
    prompt TEXT NOT NULL,
    guidance TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL CHECK (source IN ('written', 'improved', 'restored')),
    backend_id TEXT,
    model_name TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX image_prompt_history ON image_prompt_versions(session_id, created_at, id);

-- Image bodies are kept separately from metadata queries. SQLite transactions
-- make saving a result, its provenance and quota accounting atomic with Stop.
CREATE TABLE image_assets (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    body BLOB,
    created_at INTEGER NOT NULL
);
CREATE INDEX image_assets_owner ON image_assets(user_id);

CREATE TABLE image_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES image_sessions(id) ON DELETE CASCADE,
    client_request_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'improving', 'generating', 'saving', 'complete', 'stopping', 'stopped', 'interrupted', 'failed')),
    completed_count INTEGER NOT NULL DEFAULT 0,
    effective_prompt TEXT,
    prompt_version_id TEXT REFERENCES image_prompt_versions(id) ON DELETE SET NULL,
    error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE(user_id, client_request_id)
);
CREATE UNIQUE INDEX image_one_active_run_per_session ON image_runs(session_id)
    WHERE status IN ('queued', 'improving', 'generating', 'saving', 'stopping');
CREATE INDEX image_runs_owner ON image_runs(user_id, created_at DESC);

CREATE TABLE image_result_groups (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES image_sessions(id) ON DELETE CASCADE,
    selected_result_id TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE image_results (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES image_sessions(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES image_runs(id) ON DELETE CASCADE,
    group_id TEXT NOT NULL REFERENCES image_result_groups(id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL REFERENCES image_assets(id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    parent_result_id TEXT REFERENCES image_results(id) ON DELETE SET NULL,
    recipe_json TEXT NOT NULL,
    input_ids_json TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX image_results_session ON image_results(session_id, created_at, id);
CREATE INDEX image_results_group ON image_results(group_id, created_at, id);

CREATE TABLE image_gallery (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id TEXT REFERENCES image_result_groups(id) ON DELETE CASCADE,
    asset_id TEXT REFERENCES image_assets(id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    created_at INTEGER NOT NULL,
    CHECK ((group_id IS NULL) != (asset_id IS NULL))
);
CREATE INDEX image_gallery_owner ON image_gallery(user_id, created_at DESC);

ALTER TABLE image_sessions ADD COLUMN chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE;
ALTER TABLE attachments ADD COLUMN image_asset_id TEXT REFERENCES image_assets(id) ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX attachment_image_asset ON attachments(image_asset_id);
