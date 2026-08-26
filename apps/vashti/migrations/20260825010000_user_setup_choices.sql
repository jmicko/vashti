CREATE TABLE user_setting_decisions (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    setting_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('migration', 'setup', 'settings')),
    decided_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, setting_key)
);

CREATE INDEX idx_user_setting_decisions_key
    ON user_setting_decisions(setting_key, user_id);

-- Existing users have already lived with these Notes and Memories controls.
-- Record their current effective values so this migration only introduces the
-- newly added past-chat choice. Accounts created after this migration receive
-- no decision rows and therefore see every registered setup choice.
INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'notes.allow_model_read',
       CASE WHEN COALESCE(settings.allow_model_read, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_note_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'notes.allow_model_create',
       CASE WHEN COALESCE(settings.allow_model_create, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_note_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'notes.allow_model_edit',
       CASE WHEN COALESCE(settings.allow_model_edit, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_note_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'notes.allow_model_trash',
       CASE WHEN COALESCE(settings.allow_model_trash, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_note_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'memories.allow_model_read',
       CASE WHEN COALESCE(settings.allow_model_read, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_memory_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'memories.allow_model_create',
       CASE WHEN COALESCE(settings.allow_model_create, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_memory_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'memories.allow_model_edit',
       CASE WHEN COALESCE(settings.allow_model_edit, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_memory_settings settings ON settings.user_id = users.id;

INSERT INTO user_setting_decisions (
    user_id, setting_key, value_json, source, decided_at, updated_at
)
SELECT users.id, 'memories.allow_model_forget',
       CASE WHEN COALESCE(settings.allow_model_forget, 1) = 1 THEN 'true' ELSE 'false' END,
       'migration', unixepoch(), unixepoch()
FROM users
LEFT JOIN user_memory_settings settings ON settings.user_id = users.id;
