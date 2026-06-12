-- The preceding table rebuild left persona_versions pointing at the temporary
-- table name on SQLite connections with foreign keys enabled. Repair only that
-- generated reference without rebuilding persona_versions and cascading the
-- temporary name into chats and messages.
PRAGMA writable_schema = ON;

UPDATE sqlite_schema
SET sql = replace(
    replace(
        sql,
        'REFERENCES "persona_avatar_assets_legacy"',
        'REFERENCES persona_avatar_assets'
    ),
    'REFERENCES persona_avatar_assets_legacy',
    'REFERENCES persona_avatar_assets'
)
WHERE type = 'table'
  AND name = 'persona_versions'
  AND sql LIKE '%persona_avatar_assets_legacy%';

PRAGMA writable_schema = OFF;
PRAGMA schema_version = 20260611;
