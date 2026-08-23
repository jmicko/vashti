INSERT INTO user_note_settings (
    user_id,
    allow_model_read,
    allow_model_create,
    allow_model_edit,
    allow_model_trash,
    default_ai_access,
    default_all_models,
    updated_at
)
SELECT
    users.id,
    1,
    1,
    1,
    1,
    'manage',
    1,
    CAST(strftime('%s', 'now') AS INTEGER)
FROM users
WHERE NOT EXISTS (
    SELECT 1
    FROM user_note_settings
    WHERE user_note_settings.user_id = users.id
);
