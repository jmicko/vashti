ALTER TABLE app_settings
ADD COLUMN network_mode TEXT NOT NULL DEFAULT 'lan_http'
    CHECK (network_mode IN ('lan_http', 'public_https_proxy'));

ALTER TABLE app_settings
ADD COLUMN public_base_url TEXT;

ALTER TABLE app_settings
ADD COLUMN trust_proxy_headers INTEGER NOT NULL DEFAULT 0
    CHECK (trust_proxy_headers IN (0, 1));

ALTER TABLE app_settings
ADD COLUMN network_recovery_notice TEXT;
