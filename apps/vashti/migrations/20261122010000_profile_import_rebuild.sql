ALTER TABLE chat_import_records ADD COLUMN allow_profile_learning INTEGER NOT NULL DEFAULT 1
    CHECK (allow_profile_learning IN (0,1));
ALTER TABLE user_personalization ADD COLUMN rebuild_id TEXT;
ALTER TABLE personalization_processed ADD COLUMN explicitly_forgotten INTEGER NOT NULL DEFAULT 0
    CHECK (explicitly_forgotten IN (0,1));

CREATE TABLE assistant_notices (
    id TEXT PRIMARY KEY NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    read_at INTEGER
);
CREATE INDEX assistant_notices_owner ON assistant_notices(user_id, read_at, created_at);
