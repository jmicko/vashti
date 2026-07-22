CREATE TABLE server_identity (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    instance_id TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL
);
