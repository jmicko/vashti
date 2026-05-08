ALTER TABLE app_settings
ADD COLUMN tool_system_prompt TEXT;

ALTER TABLE app_settings
ADD COLUMN web_search_tool_prompt TEXT;

ALTER TABLE app_settings
ADD COLUMN web_fetch_tool_prompt TEXT;
