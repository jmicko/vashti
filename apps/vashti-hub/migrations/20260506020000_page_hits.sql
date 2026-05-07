CREATE TABLE IF NOT EXISTS page_hits (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    visitor_hash TEXT NOT NULL,
    user_agent_hash TEXT,
    visited_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_hits_path
ON page_hits(path);

CREATE INDEX IF NOT EXISTS idx_page_hits_visited_at
ON page_hits(visited_at);
