ALTER TABLE persona_versions
ADD COLUMN model_type TEXT NOT NULL DEFAULT 'general'
CHECK (model_type IN ('general', 'character'));
