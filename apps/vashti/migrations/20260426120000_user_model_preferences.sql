CREATE TABLE user_model_preferences (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    backend_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    is_visible INTEGER NOT NULL DEFAULT 1 CHECK (is_visible IN (0, 1)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, backend_id, model_name),
    FOREIGN KEY (backend_id, model_name)
        REFERENCES model_availability(backend_id, model_name)
        ON DELETE CASCADE
);

CREATE INDEX idx_user_model_preferences_user_visible
    ON user_model_preferences(user_id, is_visible);
