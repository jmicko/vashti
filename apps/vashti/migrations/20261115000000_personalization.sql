CREATE TABLE user_personalization (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    settings_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    last_processed_at INTEGER,
    last_error TEXT,
    requested INTEGER NOT NULL DEFAULT 0 CHECK (requested IN (0, 1))
);

CREATE TABLE personalization_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0, 1)),
    version INTEGER NOT NULL DEFAULT 1,
    sources_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (user_id, topic)
);
CREATE INDEX personalization_facts_owner ON personalization_facts(user_id, updated_at DESC);

CREATE TABLE personalization_fact_versions (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES personalization_facts(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    actor TEXT NOT NULL CHECK (actor IN ('human', 'processor')),
    created_at INTEGER NOT NULL,
    UNIQUE(fact_id, version)
);

CREATE TABLE personalization_processed (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    PRIMARY KEY(user_id, revision_id)
);

-- No forgotten content is retained in suppression records.
CREATE TABLE personalization_suppressed (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    PRIMARY KEY(user_id, topic)
);
