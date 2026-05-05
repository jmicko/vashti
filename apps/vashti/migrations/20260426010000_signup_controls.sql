ALTER TABLE app_settings
ADD COLUMN signup_limit INTEGER NOT NULL DEFAULT 25;

ALTER TABLE app_settings
ADD COLUMN signup_count INTEGER NOT NULL DEFAULT 0;

UPDATE app_settings
SET allow_signup = 1,
    updated_at = CAST(strftime('%s', 'now') AS INTEGER)
WHERE id = 1;
