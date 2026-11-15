DROP TRIGGER personalization_capture_source;
ALTER TABLE personalization_chat_sources RENAME TO personalization_message_sources;

CREATE TABLE personalization_chat_sources (
    revision_id TEXT PRIMARY KEY REFERENCES chat_message_revisions(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    backend_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    persona_id TEXT,
    persona_version_id TEXT,
    confirmed INTEGER NOT NULL DEFAULT 0
);

-- Historical edits have no reliable model provenance. Do not guess from today's chat.
INSERT INTO personalization_chat_sources
SELECT r.id,s.message_id,s.backend_id,s.model_name,s.persona_id,s.persona_version_id,1
FROM personalization_message_sources s
JOIN chat_message_revisions r ON r.message_id=s.message_id
WHERE r.source='original' AND r.rowid=(
    SELECT first.rowid FROM chat_message_revisions first
    WHERE first.message_id=s.message_id ORDER BY first.created_at,first.rowid LIMIT 1
);
DROP TABLE personalization_message_sources;

CREATE TRIGGER personalization_capture_revision
AFTER INSERT ON chat_message_revisions
BEGIN
    INSERT INTO personalization_chat_sources
    SELECT NEW.id,m.id,c.default_backend_id,c.default_model_name,c.persona_id,c.persona_version_id,
           CASE WHEN NEW.source='original' THEN 0 ELSE 1 END
    FROM chat_messages m JOIN chats c ON c.id=m.chat_id
    WHERE m.id=NEW.message_id AND m.role='user';
END;

-- Initial sends may override the chat defaults in the same transaction.
CREATE TRIGGER personalization_capture_source
AFTER INSERT ON chat_messages
WHEN NEW.role='assistant' AND NEW.parent_message_id IS NOT NULL
BEGIN
    UPDATE personalization_chat_sources
    SET backend_id=NEW.backend_id,model_name=NEW.model_name,
        persona_id=NEW.persona_id,persona_version_id=NEW.persona_version_id,confirmed=1
    WHERE confirmed=0 AND NEW.backend_id IS NOT NULL AND NEW.model_name IS NOT NULL
      AND revision_id=(SELECT active_revision_id FROM chat_messages WHERE id=NEW.parent_message_id);
END;
