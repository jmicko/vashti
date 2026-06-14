ALTER TABLE model_availability
    ADD COLUMN background_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE model_availability
    ADD COLUMN background_dim REAL NOT NULL DEFAULT 0.72;

ALTER TABLE model_availability
    ADD COLUMN background_landscape_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE model_availability
    ADD COLUMN background_landscape_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE model_availability
    ADD COLUMN background_landscape_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE model_availability
    ADD COLUMN background_portrait_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE model_availability
    ADD COLUMN background_portrait_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE model_availability
    ADD COLUMN background_portrait_y REAL NOT NULL DEFAULT 50.0;

CREATE INDEX idx_model_availability_background_asset
    ON model_availability(background_asset_id);

ALTER TABLE user_model_preferences
    ADD COLUMN background_asset_id TEXT REFERENCES persona_avatar_assets(id) ON DELETE SET NULL;

ALTER TABLE user_model_preferences
    ADD COLUMN background_dim REAL NOT NULL DEFAULT 0.72;

ALTER TABLE user_model_preferences
    ADD COLUMN background_landscape_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE user_model_preferences
    ADD COLUMN background_landscape_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE user_model_preferences
    ADD COLUMN background_landscape_y REAL NOT NULL DEFAULT 50.0;

ALTER TABLE user_model_preferences
    ADD COLUMN background_portrait_mode TEXT NOT NULL DEFAULT 'fill';

ALTER TABLE user_model_preferences
    ADD COLUMN background_portrait_x REAL NOT NULL DEFAULT 50.0;

ALTER TABLE user_model_preferences
    ADD COLUMN background_portrait_y REAL NOT NULL DEFAULT 50.0;

CREATE INDEX idx_user_model_preferences_background_asset
    ON user_model_preferences(background_asset_id);
