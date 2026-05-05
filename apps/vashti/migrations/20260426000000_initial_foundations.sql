CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    is_disabled INTEGER NOT NULL DEFAULT 0 CHECK (is_disabled IN (0, 1)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_login_at INTEGER
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);

CREATE TABLE ollama_backends (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1 CHECK (is_enabled IN (0, 1)),
    is_localhost_detected INTEGER NOT NULL DEFAULT 0 CHECK (is_localhost_detected IN (0, 1)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_healthcheck_at INTEGER,
    last_health_status TEXT,
    last_error TEXT
);

CREATE INDEX idx_ollama_backends_is_enabled ON ollama_backends(is_enabled);

CREATE TABLE user_settings (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    default_backend_id TEXT REFERENCES ollama_backends(id) ON DELETE SET NULL,
    default_model_name TEXT,
    theme TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE app_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    allow_signup INTEGER NOT NULL DEFAULT 0 CHECK (allow_signup IN (0, 1)),
    max_upload_bytes INTEGER NOT NULL DEFAULT 10485760,
    request_timeout_ms INTEGER NOT NULL DEFAULT 120000,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
