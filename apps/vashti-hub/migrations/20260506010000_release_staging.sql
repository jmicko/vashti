ALTER TABLE releases ADD COLUMN release_status TEXT NOT NULL DEFAULT 'released';
ALTER TABLE release_settings ADD COLUMN prerelease_version TEXT;

UPDATE releases
SET release_status = 'released';
