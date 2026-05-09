ALTER TABLE app_settings
ADD COLUMN default_model_permission_tags_json TEXT NOT NULL DEFAULT '["system:everyone"]';

ALTER TABLE app_settings
ADD COLUMN default_tool_permission_tags_json TEXT NOT NULL DEFAULT '["system:everyone"]';

CREATE TABLE user_permission_tags (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, tag_id)
);

CREATE INDEX idx_user_permission_tags_tag_id
    ON user_permission_tags(tag_id);

CREATE TABLE model_permission_tags (
    backend_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    tag_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (backend_id, model_name, tag_id),
    FOREIGN KEY (backend_id, model_name)
        REFERENCES model_availability(backend_id, model_name)
        ON DELETE CASCADE
);

CREATE INDEX idx_model_permission_tags_tag_id
    ON model_permission_tags(tag_id);

CREATE TABLE tool_permission_state (
    tool_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE tool_permission_tags (
    tool_id TEXT NOT NULL REFERENCES tool_permission_state(tool_id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (tool_id, tag_id)
);

CREATE INDEX idx_tool_permission_tags_tag_id
    ON tool_permission_tags(tag_id);

INSERT INTO model_permission_tags (backend_id, model_name, tag_id, created_at)
SELECT backend_id, model_name, 'system:everyone', CAST(strftime('%s', 'now') AS INTEGER)
FROM model_availability;

INSERT INTO tool_permission_state (tool_id, created_at, updated_at)
VALUES
    ('brave_web_search', CAST(strftime('%s', 'now') AS INTEGER), CAST(strftime('%s', 'now') AS INTEGER)),
    ('ollama_web_search', CAST(strftime('%s', 'now') AS INTEGER), CAST(strftime('%s', 'now') AS INTEGER)),
    ('ollama_web_fetch', CAST(strftime('%s', 'now') AS INTEGER), CAST(strftime('%s', 'now') AS INTEGER)),
    ('direct_web_fetch', CAST(strftime('%s', 'now') AS INTEGER), CAST(strftime('%s', 'now') AS INTEGER));

INSERT INTO tool_permission_tags (tool_id, tag_id, created_at)
SELECT tool_id, 'system:everyone', CAST(strftime('%s', 'now') AS INTEGER)
FROM tool_permission_state;
