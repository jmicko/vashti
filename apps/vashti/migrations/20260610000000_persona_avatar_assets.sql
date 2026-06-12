CREATE TABLE persona_avatar_assets (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL CHECK (mime_type IN ('image/jpeg', 'image/png')),
    size_bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_persona_avatar_assets_owner
    ON persona_avatar_assets(owner_user_id, created_at);

ALTER TABLE persona_versions
    ADD COLUMN avatar_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE persona_versions
    ADD COLUMN avatar_crop_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE persona_versions
    ADD COLUMN avatar_crop_y REAL NOT NULL DEFAULT 50.0;

CREATE INDEX idx_persona_versions_avatar_asset
    ON persona_versions(avatar_asset_id);
