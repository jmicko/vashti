ALTER TABLE app_settings
ADD COLUMN tools_enabled INTEGER NOT NULL DEFAULT 0 CHECK (tools_enabled IN (0, 1));

ALTER TABLE app_settings
ADD COLUMN ollama_web_search_enabled INTEGER NOT NULL DEFAULT 0 CHECK (ollama_web_search_enabled IN (0, 1));

ALTER TABLE app_settings
ADD COLUMN ollama_web_fetch_enabled INTEGER NOT NULL DEFAULT 0 CHECK (ollama_web_fetch_enabled IN (0, 1));

ALTER TABLE app_settings
ADD COLUMN ollama_api_key TEXT;

ALTER TABLE app_settings
ADD COLUMN brave_search_enabled INTEGER NOT NULL DEFAULT 0 CHECK (brave_search_enabled IN (0, 1));

ALTER TABLE app_settings
ADD COLUMN brave_search_api_key TEXT;

ALTER TABLE app_settings
ADD COLUMN direct_web_fetch_enabled INTEGER NOT NULL DEFAULT 0 CHECK (direct_web_fetch_enabled IN (0, 1));
