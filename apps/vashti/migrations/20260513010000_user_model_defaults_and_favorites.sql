ALTER TABLE user_model_preferences
ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1));

ALTER TABLE user_model_preferences
ADD COLUMN is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1));

CREATE UNIQUE INDEX idx_user_model_preferences_user_default
    ON user_model_preferences(user_id)
    WHERE is_default = 1;
