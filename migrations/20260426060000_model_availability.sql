CREATE TABLE model_availability (
    backend_id TEXT NOT NULL REFERENCES ollama_backends(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    is_enabled INTEGER NOT NULL DEFAULT 1 CHECK (is_enabled IN (0, 1)),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (backend_id, model_name)
);

CREATE INDEX idx_model_availability_backend_enabled
    ON model_availability(backend_id, is_enabled);
