ALTER TABLE model_availability
    ADD COLUMN background_message_dim REAL NOT NULL DEFAULT 0.82;

ALTER TABLE model_availability
    ADD COLUMN background_landscape_scale REAL NOT NULL DEFAULT 35.0;

ALTER TABLE model_availability
    ADD COLUMN background_portrait_scale REAL NOT NULL DEFAULT 35.0;

ALTER TABLE user_model_preferences
    ADD COLUMN background_message_dim REAL NOT NULL DEFAULT 0.82;

ALTER TABLE user_model_preferences
    ADD COLUMN background_landscape_scale REAL NOT NULL DEFAULT 35.0;

ALTER TABLE user_model_preferences
    ADD COLUMN background_portrait_scale REAL NOT NULL DEFAULT 35.0;

ALTER TABLE persona_versions
    ADD COLUMN background_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE persona_versions
    ADD COLUMN background_dim REAL NOT NULL DEFAULT 0.72;

ALTER TABLE persona_versions
    ADD COLUMN background_message_dim REAL NOT NULL DEFAULT 0.82;

ALTER TABLE persona_versions
    ADD COLUMN background_landscape_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE persona_versions
    ADD COLUMN background_landscape_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE persona_versions
    ADD COLUMN background_landscape_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE persona_versions
    ADD COLUMN background_landscape_scale REAL NOT NULL DEFAULT 35.0;

ALTER TABLE persona_versions
    ADD COLUMN background_portrait_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE persona_versions
    ADD COLUMN background_portrait_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE persona_versions
    ADD COLUMN background_portrait_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE persona_versions
    ADD COLUMN background_portrait_scale REAL NOT NULL DEFAULT 35.0;

CREATE INDEX idx_persona_versions_background_asset
    ON persona_versions(background_asset_id);
