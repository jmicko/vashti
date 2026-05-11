CREATE TABLE model_default_permission_tags (
    backend_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    tag_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (backend_id, model_name, tag_id),
    FOREIGN KEY (backend_id, model_name)
        REFERENCES model_availability(backend_id, model_name)
        ON DELETE CASCADE
);

CREATE INDEX idx_model_default_permission_tags_tag_id
    ON model_default_permission_tags(tag_id);

INSERT INTO model_default_permission_tags (backend_id, model_name, tag_id, created_at)
SELECT mpt.backend_id, mpt.model_name, mpt.tag_id, mpt.created_at
FROM model_permission_tags mpt
WHERE EXISTS (
    SELECT 1
    FROM app_settings, json_each(app_settings.default_model_permission_tags_json)
    WHERE app_settings.id = 1
      AND json_each.value = mpt.tag_id
);

DELETE FROM model_permission_tags
WHERE EXISTS (
    SELECT 1
    FROM app_settings, json_each(app_settings.default_model_permission_tags_json)
    WHERE app_settings.id = 1
      AND json_each.value = model_permission_tags.tag_id
);
