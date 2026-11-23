CREATE TABLE personalization_replay_facts (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    fact_json TEXT NOT NULL,
    history_json TEXT NOT NULL,
    PRIMARY KEY(user_id, topic)
);

CREATE TABLE personalization_replay_processed (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    PRIMARY KEY(user_id, revision_id)
);
