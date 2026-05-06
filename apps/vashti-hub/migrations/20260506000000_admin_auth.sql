CREATE TABLE IF NOT EXISTS hub_admins (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hub_admin_sessions (
    id TEXT PRIMARY KEY,
    expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hub_admin_sessions_expires_at
ON hub_admin_sessions(expires_at);

ALTER TABLE release_settings ADD COLUMN admin_setup_key_hash TEXT;
ALTER TABLE release_settings ADD COLUMN reset_key_hash TEXT;
ALTER TABLE release_settings ADD COLUMN reset_key_generated_at INTEGER;
