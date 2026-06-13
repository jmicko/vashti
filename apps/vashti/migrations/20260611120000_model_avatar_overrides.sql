ALTER TABLE model_availability
    ADD COLUMN avatar_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE model_availability
    ADD COLUMN avatar_crop_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE model_availability
    ADD COLUMN avatar_crop_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE model_availability
    ADD COLUMN avatar_crop_size REAL NOT NULL DEFAULT 100.0;

CREATE INDEX idx_model_availability_avatar_asset
    ON model_availability(avatar_asset_id);

ALTER TABLE user_model_preferences
    ADD COLUMN avatar_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE user_model_preferences
    ADD COLUMN avatar_crop_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE user_model_preferences
    ADD COLUMN avatar_crop_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE user_model_preferences
    ADD COLUMN avatar_crop_size REAL NOT NULL DEFAULT 100.0;

CREATE INDEX idx_user_model_preferences_avatar_asset
    ON user_model_preferences(avatar_asset_id);
