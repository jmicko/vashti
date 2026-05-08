ALTER TABLE chats
ADD COLUMN tool_use_enabled INTEGER NOT NULL DEFAULT 1 CHECK (tool_use_enabled IN (0, 1));

ALTER TABLE chats
ADD COLUMN web_search_tool_enabled INTEGER NOT NULL DEFAULT 1 CHECK (web_search_tool_enabled IN (0, 1));

ALTER TABLE chats
ADD COLUMN web_fetch_tool_enabled INTEGER NOT NULL DEFAULT 1 CHECK (web_fetch_tool_enabled IN (0, 1));
