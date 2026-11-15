CREATE TABLE personalization_chat_sources (
    message_id TEXT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    backend_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    persona_id TEXT,
    persona_version_id TEXT
);

-- Prefer the original response's model over a conversation's current defaults.
INSERT INTO personalization_chat_sources
SELECT m.id,
       COALESCE((SELECT a.backend_id FROM chat_messages a WHERE a.parent_message_id=m.id AND a.role='assistant' ORDER BY a.created_at,a.rowid LIMIT 1),c.default_backend_id),
       COALESCE((SELECT a.model_name FROM chat_messages a WHERE a.parent_message_id=m.id AND a.role='assistant' ORDER BY a.created_at,a.rowid LIMIT 1),c.default_model_name),
       COALESCE((SELECT a.persona_id FROM chat_messages a WHERE a.parent_message_id=m.id AND a.role='assistant' ORDER BY a.created_at,a.rowid LIMIT 1),c.persona_id),
       COALESCE((SELECT a.persona_version_id FROM chat_messages a WHERE a.parent_message_id=m.id AND a.role='assistant' ORDER BY a.created_at,a.rowid LIMIT 1),c.persona_version_id)
FROM chat_messages m JOIN chats c ON c.id=m.chat_id WHERE m.role='user';

CREATE TRIGGER personalization_capture_source
AFTER INSERT ON chat_messages
WHEN NEW.role='assistant' AND NEW.parent_message_id IS NOT NULL
BEGIN
    INSERT OR IGNORE INTO personalization_chat_sources
    SELECT m.id,NEW.backend_id,NEW.model_name,NEW.persona_id,NEW.persona_version_id
    FROM chat_messages m WHERE m.id=NEW.parent_message_id AND m.role='user';
END;
