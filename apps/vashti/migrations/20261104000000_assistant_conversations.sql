ALTER TABLE assistant_tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'task' CHECK(kind IN ('task','conversation'));
ALTER TABLE assistant_tasks ADD COLUMN conversation_id TEXT REFERENCES assistant_tasks(id);
ALTER TABLE assistant_runs ADD COLUMN lane TEXT NOT NULL DEFAULT 'background' CHECK(lane IN ('foreground','background'));
CREATE INDEX assistant_conversation_tasks ON assistant_tasks(user_id,conversation_id,updated_at);
CREATE INDEX assistant_conversations ON assistant_tasks(user_id,kind,updated_at);
CREATE UNIQUE INDEX assistant_running_lane ON assistant_runs(user_id,lane) WHERE status='running';
