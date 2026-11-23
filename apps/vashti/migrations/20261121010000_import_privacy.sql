ALTER TABLE chat_import_records ADD COLUMN do_not_remember INTEGER NOT NULL DEFAULT 0 CHECK (do_not_remember IN (0,1));
UPDATE chat_import_records SET do_not_remember=1 WHERE json_extract(original_json,'$.is_do_not_remember')=1;
DELETE FROM personalization_chat_sources
WHERE message_id IN (SELECT id FROM chat_messages WHERE import_source IS NOT NULL);

DROP TRIGGER personalization_capture_revision;
CREATE TRIGGER personalization_capture_revision
AFTER INSERT ON chat_message_revisions
BEGIN
    INSERT INTO personalization_chat_sources
    SELECT NEW.id,m.id,c.default_backend_id,c.default_model_name,c.persona_id,c.persona_version_id,
           CASE WHEN NEW.source='original' THEN 0 ELSE 1 END
    FROM chat_messages m JOIN chats c ON c.id=m.chat_id
    WHERE m.id=NEW.message_id AND m.role='user' AND m.import_source IS NULL
      AND NOT EXISTS (SELECT 1 FROM chat_import_records imported WHERE imported.chat_id=c.id AND imported.do_not_remember=1);
END;
