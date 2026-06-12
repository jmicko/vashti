ALTER TABLE persona_versions
    ADD COLUMN avatar_crop_size REAL NOT NULL DEFAULT 100.0;

PRAGMA legacy_alter_table = ON;

ALTER TABLE persona_avatar_assets
    RENAME TO persona_avatar_assets_legacy;

CREATE TABLE persona_avatar_assets (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL CHECK (mime_type IN ('image/jpeg', 'image/png', 'image/gif')),
    size_bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

INSERT INTO persona_avatar_assets (
    id,
    owner_user_id,
    original_filename,
    storage_path,
    mime_type,
    size_bytes,
    created_at
)
SELECT id,
       owner_user_id,
       original_filename,
       storage_path,
       mime_type,
       size_bytes,
       created_at
FROM persona_avatar_assets_legacy;

DROP TABLE persona_avatar_assets_legacy;

CREATE INDEX idx_persona_avatar_assets_owner
    ON persona_avatar_assets(owner_user_id, created_at);

PRAGMA legacy_alter_table = OFF;
