ALTER TABLE persona_versions
    ADD COLUMN inference_settings_json TEXT NOT NULL DEFAULT '{}';
