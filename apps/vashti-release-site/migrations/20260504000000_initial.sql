CREATE TABLE IF NOT EXISTS release_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    upload_token_hash TEXT,
    latest_version TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS releases (
    version TEXT PRIMARY KEY,
    version_major INTEGER NOT NULL,
    version_minor INTEGER NOT NULL,
    version_patch INTEGER NOT NULL,
    notes TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS release_artifacts (
    id TEXT PRIMARY KEY,
    release_version TEXT NOT NULL REFERENCES releases(version) ON DELETE CASCADE,
    target TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(release_version, target),
    UNIQUE(release_version, filename)
);

CREATE TABLE IF NOT EXISTS download_events (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES release_artifacts(id) ON DELETE CASCADE,
    release_version TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    user_agent_hash TEXT,
    downloaded_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_release_artifacts_release_version
ON release_artifacts(release_version);

CREATE INDEX IF NOT EXISTS idx_download_events_artifact_id
ON download_events(artifact_id);

CREATE INDEX IF NOT EXISTS idx_download_events_downloaded_at
ON download_events(downloaded_at);
