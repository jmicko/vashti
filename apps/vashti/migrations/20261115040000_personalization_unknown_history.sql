-- A historical unanswered message has no immutable model identity to recover.
DELETE FROM personalization_chat_sources
WHERE confirmed=1
  AND revision_id IN (SELECT id FROM chat_message_revisions WHERE source='original')
  AND NOT EXISTS (
      SELECT 1 FROM chat_messages a
      WHERE a.parent_message_id=personalization_chat_sources.message_id AND a.role='assistant'
  );
