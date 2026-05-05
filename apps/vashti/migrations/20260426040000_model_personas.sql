CREATE TABLE personas (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    current_version_id TEXT,
    visibility TEXT NOT NULL DEFAULT 'private' CHECK (visibility IN ('private', 'public')),
    lifecycle_state TEXT NOT NULL DEFAULT 'active' CHECK (lifecycle_state IN ('active', 'disowned', 'deleted')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE persona_versions (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    avatar_attachment_id TEXT REFERENCES attachments(id) ON DELETE SET NULL,
    base_backend_id TEXT NOT NULL REFERENCES ollama_backends(id) ON DELETE RESTRICT,
    base_model_name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    tool_policy_json TEXT,
    created_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(persona_id, version_number)
);

CREATE TABLE persona_members (
    persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    membership_role TEXT NOT NULL DEFAULT 'member' CHECK (membership_role IN ('creator', 'member')),
    created_at INTEGER NOT NULL,
    PRIMARY KEY (persona_id, user_id)
);

CREATE INDEX idx_personas_owner ON personas(owner_user_id);
CREATE INDEX idx_personas_visibility_state ON personas(visibility, lifecycle_state);
CREATE INDEX idx_persona_versions_persona_version ON persona_versions(persona_id, version_number);
CREATE INDEX idx_persona_members_user ON persona_members(user_id);
CREATE INDEX idx_persona_members_persona ON persona_members(persona_id);

ALTER TABLE chats ADD COLUMN persona_id TEXT REFERENCES personas(id) ON DELETE SET NULL;
ALTER TABLE chats ADD COLUMN persona_version_id TEXT REFERENCES persona_versions(id) ON DELETE SET NULL;

ALTER TABLE chat_messages ADD COLUMN persona_id TEXT REFERENCES personas(id) ON DELETE SET NULL;
ALTER TABLE chat_messages ADD COLUMN persona_version_id TEXT REFERENCES persona_versions(id) ON DELETE SET NULL;
ALTER TABLE chat_messages ADD COLUMN persona_name_snapshot TEXT;
