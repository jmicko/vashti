CREATE TABLE assistant_preferences (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    config_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE assistant_tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    config_json TEXT NOT NULL,
    interval_seconds INTEGER CHECK (interval_seconds IS NULL OR interval_seconds BETWEEN 300 AND 2592000),
    next_run_at INTEGER,
    schedule_enabled INTEGER NOT NULL DEFAULT 0 CHECK (schedule_enabled IN (0, 1)),
    archived_at INTEGER,
    last_run_id TEXT,
    viewed_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX assistant_tasks_owner ON assistant_tasks(user_id, archived_at, updated_at DESC);
CREATE INDEX assistant_tasks_due ON assistant_tasks(next_run_at) WHERE schedule_enabled = 1 AND archived_at IS NULL;

CREATE TABLE assistant_runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES assistant_tasks(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('queued','running','waiting_approval','waiting_input','paused','completed','failed','cancelled','interrupted','limit_reached')),
    instructions TEXT NOT NULL,
    config_json TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    progress_text TEXT NOT NULL DEFAULT '',
    progress_thinking TEXT NOT NULL DEFAULT '',
    result_text TEXT NOT NULL DEFAULT '',
    error_text TEXT NOT NULL DEFAULT '',
    steps INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    scheduled INTEGER NOT NULL DEFAULT 0 CHECK (scheduled IN (0, 1)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    finished_at INTEGER
);
CREATE INDEX assistant_runs_task ON assistant_runs(task_id, created_at DESC);
CREATE INDEX assistant_runs_queue ON assistant_runs(status, created_at);
CREATE UNIQUE INDEX assistant_one_active_run ON assistant_runs(task_id)
    WHERE status IN ('queued','running','waiting_approval','waiting_input','paused');

CREATE TABLE assistant_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES assistant_runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ready','pending','approved','rejected','executing','completed','uncertain')),
    result_text TEXT,
    requested_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    decided_at INTEGER,
    finished_at INTEGER,
    UNIQUE(run_id, ordinal)
);
CREATE INDEX assistant_actions_run ON assistant_actions(run_id, ordinal);

CREATE TABLE assistant_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES assistant_runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX assistant_events_run ON assistant_events(run_id, id);

CREATE TABLE assistant_artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES assistant_runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size INTEGER NOT NULL CHECK (size BETWEEN 0 AND 20971520),
    created_at INTEGER NOT NULL
);
