ALTER TABLE image_sessions ADD COLUMN active_result_id TEXT
    REFERENCES image_results(id) ON DELETE SET NULL;
ALTER TABLE image_sessions ADD COLUMN title_generated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE image_sessions ADD COLUMN is_archived INTEGER NOT NULL DEFAULT 0;
ALTER TABLE image_runs ADD COLUMN submission_json TEXT;
UPDATE image_runs SET submission_json = request_json;
CREATE INDEX image_runs_session ON image_runs(session_id, created_at DESC);
