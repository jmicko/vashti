CREATE TABLE user_memory_settings_new (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    allow_model_read INTEGER NOT NULL DEFAULT 1 CHECK (allow_model_read IN (0, 1)),
    allow_model_create INTEGER NOT NULL DEFAULT 1 CHECK (allow_model_create IN (0, 1)),
    allow_model_edit INTEGER NOT NULL DEFAULT 1 CHECK (allow_model_edit IN (0, 1)),
    allow_model_forget INTEGER NOT NULL DEFAULT 1 CHECK (allow_model_forget IN (0, 1)),
    updated_at INTEGER NOT NULL
);

INSERT INTO user_memory_settings_new (
    user_id,
    allow_model_read,
    allow_model_create,
    allow_model_edit,
    allow_model_forget,
    updated_at
)
SELECT
    users.id,
    1,
    1,
    1,
    1,
    CAST(strftime('%s', 'now') AS INTEGER)
FROM users;

DROP TABLE user_memory_settings;
ALTER TABLE user_memory_settings_new RENAME TO user_memory_settings;
