CREATE TABLE video_preferences(user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, value_json TEXT NOT NULL);
CREATE TABLE video_frames(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, mime_type TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, body BLOB NOT NULL, thumbnail BLOB NOT NULL, size_bytes INTEGER NOT NULL);
CREATE TABLE video_sessions(id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, title TEXT NOT NULL, draft_json TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, archived_at INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE INDEX video_sessions_owner ON video_sessions(user_id,updated_at);
CREATE TABLE video_runs(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES video_sessions(id) ON DELETE CASCADE, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, group_id TEXT NOT NULL, request_json TEXT NOT NULL, draft_json TEXT NOT NULL, status TEXT NOT NULL, error TEXT, size_bytes INTEGER NOT NULL DEFAULT 0, duration_ms INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE UNIQUE INDEX video_active_session ON video_runs(session_id) WHERE status IN ('queued','generating','saving');
CREATE INDEX video_runs_owner ON video_runs(user_id,session_id,created_at);
