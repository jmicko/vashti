use std::collections::{HashMap, HashSet};

use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{
    auth::service::{self as auth_service, unix_timestamp},
    error::ApiError,
    permissions::service::{self as permissions, PermissionTagResponse},
    settings::handlers::{
        UpdateAppSettingsRequest, UpdateNetworkSettingsRequest, UpdateToolSettingsRequest,
        UpdateUserSettingsRequest,
    },
};

pub const DEFAULT_TOOL_SYSTEM_PROMPT: &str = "Tool behavior guidance:\n- Current date: {current_date} UTC.\n- Use an available web search tool proactively when the user asks for current, recent, latest, news, prices, schedules, releases, versions, or anything likely to have changed.\n- Use an available web fetch tool when the user provides a URL or when a search result needs more detail.\n- Treat tool results as current external data even when their dates are newer than your training cutoff.\n- Answer from the tool results and include source URLs when they are available.\n- Only call tools by the exact names listed as available in this chat.";
pub const DEFAULT_WEB_SEARCH_TOOL_PROMPT: &str = "Search the web for current public information. Returns compact result titles, URLs, and snippets. Use this for current or time-sensitive questions, latest news, recent releases, prices, schedules, or facts that may have changed.";
pub const DEFAULT_WEB_FETCH_TOOL_PROMPT: &str = "Fetch a public HTTP or HTTPS page by URL and return readable text plus discovered links. Use this after web search when a result needs more detail, or when the user asks about a specific URL.";

#[derive(Debug, Clone, Serialize)]
pub struct AppSettingsResponse {
    pub allow_signup: bool,
    pub signup_limit: i64,
    pub signup_count: i64,
    pub max_upload_bytes: i64,
    pub request_timeout_ms: i64,
    pub network_mode: String,
    pub public_base_url: Option<String>,
    pub trust_proxy_headers: bool,
    pub network_recovery_notice: Option<String>,
    pub update_channel: String,
}

impl AppSettingsResponse {
    pub fn secure_session_cookies(&self) -> bool {
        self.network_mode == "public_https_proxy"
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UserSettingsResponse {
    pub default_backend_id: Option<String>,
    pub default_model_name: Option<String>,
    pub theme: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSettingsResponse {
    pub tools_enabled: bool,
    pub ollama_web_search_enabled: bool,
    pub ollama_web_fetch_enabled: bool,
    pub ollama_api_key_configured: bool,
    pub brave_search_enabled: bool,
    pub brave_search_api_key_configured: bool,
    pub direct_web_fetch_enabled: bool,
    pub tool_system_prompt: String,
    pub default_tool_system_prompt: &'static str,
    pub web_search_tool_prompt: String,
    pub default_web_search_tool_prompt: &'static str,
    pub web_fetch_tool_prompt: String,
    pub default_web_fetch_tool_prompt: &'static str,
    pub available_tags: Vec<PermissionTagResponse>,
    pub default_tool_permission_tags: Vec<PermissionTagResponse>,
    pub tool_permissions: Vec<ToolPermissionResponse>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AvailableToolsResponse {
    pub tools_enabled: bool,
    pub tools: Vec<AvailableToolResponse>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AvailableToolResponse {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolPermissionResponse {
    pub tool_id: &'static str,
    pub permission_tags: Vec<PermissionTagResponse>,
}

#[derive(Debug, Clone)]
pub struct ToolSettingsPrivate {
    pub tools_enabled: bool,
    pub ollama_web_search_enabled: bool,
    pub ollama_web_fetch_enabled: bool,
    pub ollama_api_key: Option<String>,
    pub brave_search_enabled: bool,
    pub brave_search_api_key: Option<String>,
    pub direct_web_fetch_enabled: bool,
    pub tool_system_prompt: String,
    pub web_search_tool_prompt: String,
    pub web_fetch_tool_prompt: String,
}

pub async fn get_available_tools(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<AvailableToolsResponse, sqlx::Error> {
    permissions::ensure_tool_records(pool).await?;
    let settings = get_tool_settings_private(pool).await?;
    let user_tags = permissions::effective_user_tag_ids(pool, user_id).await?;
    let tool_tags = permissions::tool_tags_by_tool(pool).await?;
    let mut tools = Vec::new();

    if settings.tools_enabled {
        if settings.brave_search_enabled
            && settings.has_brave_key()
            && tool_allowed(
                crate::tools::service::TOOL_BRAVE_WEB_SEARCH,
                &user_tags,
                &tool_tags,
            )
        {
            tools.push(AvailableToolResponse {
                id: "brave_web_search",
                label: "Brave web search",
                description: "Search the web with Brave Search.",
            });
        }

        if settings.ollama_web_search_enabled
            && settings.has_ollama_key()
            && tool_allowed(
                crate::tools::service::TOOL_OLLAMA_WEB_SEARCH,
                &user_tags,
                &tool_tags,
            )
        {
            tools.push(AvailableToolResponse {
                id: "ollama_web_search",
                label: "Ollama web search",
                description: "Search the web through Ollama's hosted search API.",
            });
        }

        if settings.ollama_web_fetch_enabled
            && settings.has_ollama_key()
            && tool_allowed(
                crate::tools::service::TOOL_OLLAMA_WEB_FETCH,
                &user_tags,
                &tool_tags,
            )
        {
            tools.push(AvailableToolResponse {
                id: "ollama_web_fetch",
                label: "Ollama web fetch",
                description: "Fetch pages through Ollama's hosted fetch API.",
            });
        }

        if settings.direct_web_fetch_enabled
            && tool_allowed(
                crate::tools::service::TOOL_DIRECT_WEB_FETCH,
                &user_tags,
                &tool_tags,
            )
        {
            tools.push(AvailableToolResponse {
                id: "direct_web_fetch",
                label: "Direct page fetch",
                description: "Fetch public HTTP/HTTPS pages from the Vashti server.",
            });
        }
    }

    Ok(AvailableToolsResponse {
        tools_enabled: settings.tools_enabled,
        tools,
    })
}

pub async fn get_app_settings(pool: &SqlitePool) -> Result<AppSettingsResponse, sqlx::Error> {
    let row = sqlx::query(
        r#"
        SELECT allow_signup,
               signup_limit,
               signup_count,
               max_upload_bytes,
               request_timeout_ms,
               network_mode,
               public_base_url,
               trust_proxy_headers,
               network_recovery_notice,
               update_channel
        FROM app_settings
        WHERE id = 1
        "#,
    )
    .fetch_one(pool)
    .await?;

    row_to_app_settings(row)
}

fn tool_allowed(
    tool_id: &str,
    user_tags: &HashSet<String>,
    tool_tags: &HashMap<String, Vec<String>>,
) -> bool {
    tool_tags
        .get(tool_id)
        .is_some_and(|tags| permissions::has_matching_tag(user_tags, tags))
}

pub async fn get_tool_settings(pool: &SqlitePool) -> Result<ToolSettingsResponse, sqlx::Error> {
    permissions::ensure_tool_records(pool).await?;
    let private = get_tool_settings_private(pool).await?;
    private.to_response(pool).await
}

pub async fn get_tool_settings_private(
    pool: &SqlitePool,
) -> Result<ToolSettingsPrivate, sqlx::Error> {
    let row = sqlx::query(
        r#"
        SELECT tools_enabled,
               ollama_web_search_enabled,
               ollama_web_fetch_enabled,
               ollama_api_key,
               brave_search_enabled,
               brave_search_api_key,
               direct_web_fetch_enabled,
               tool_system_prompt,
               web_search_tool_prompt,
               web_fetch_tool_prompt
        FROM app_settings
        WHERE id = 1
        "#,
    )
    .fetch_one(pool)
    .await?;

    row_to_tool_settings_private(row)
}

pub async fn update_tool_settings(
    pool: &SqlitePool,
    payload: UpdateToolSettingsRequest,
) -> Result<ToolSettingsResponse, ApiError> {
    let current = get_tool_settings_private(pool).await?;
    let ollama_api_key = updated_secret(
        current.ollama_api_key,
        payload.ollama_api_key,
        payload.clear_ollama_api_key,
    );
    let brave_search_api_key = updated_secret(
        current.brave_search_api_key,
        payload.brave_search_api_key,
        payload.clear_brave_search_api_key,
    );
    let tool_system_prompt = updated_prompt(
        &current.tool_system_prompt,
        payload.tool_system_prompt,
        DEFAULT_TOOL_SYSTEM_PROMPT,
    )?;
    let web_search_tool_prompt = updated_prompt(
        &current.web_search_tool_prompt,
        payload.web_search_tool_prompt,
        DEFAULT_WEB_SEARCH_TOOL_PROMPT,
    )?;
    let web_fetch_tool_prompt = updated_prompt(
        &current.web_fetch_tool_prompt,
        payload.web_fetch_tool_prompt,
        DEFAULT_WEB_FETCH_TOOL_PROMPT,
    )?;

    let row = sqlx::query(
        r#"
        UPDATE app_settings
        SET tools_enabled = COALESCE(?, tools_enabled),
            ollama_web_search_enabled = COALESCE(?, ollama_web_search_enabled),
            ollama_web_fetch_enabled = COALESCE(?, ollama_web_fetch_enabled),
            ollama_api_key = ?,
            brave_search_enabled = COALESCE(?, brave_search_enabled),
            brave_search_api_key = ?,
            direct_web_fetch_enabled = COALESCE(?, direct_web_fetch_enabled),
            tool_system_prompt = ?,
            web_search_tool_prompt = ?,
            web_fetch_tool_prompt = ?,
            updated_at = ?
        WHERE id = 1
        RETURNING tools_enabled,
                  ollama_web_search_enabled,
                  ollama_web_fetch_enabled,
                  ollama_api_key,
                  brave_search_enabled,
                  brave_search_api_key,
                  direct_web_fetch_enabled,
                  tool_system_prompt,
                  web_search_tool_prompt,
                  web_fetch_tool_prompt
        "#,
    )
    .bind(payload.tools_enabled.map(i64::from))
    .bind(payload.ollama_web_search_enabled.map(i64::from))
    .bind(payload.ollama_web_fetch_enabled.map(i64::from))
    .bind(ollama_api_key)
    .bind(payload.brave_search_enabled.map(i64::from))
    .bind(brave_search_api_key)
    .bind(payload.direct_web_fetch_enabled.map(i64::from))
    .bind(tool_system_prompt)
    .bind(web_search_tool_prompt)
    .bind(web_fetch_tool_prompt)
    .bind(unix_timestamp())
    .fetch_one(pool)
    .await?;

    if let Some(tags) = payload.default_tool_permission_tags {
        permissions::update_default_tool_tags(pool, tags).await?;
    }
    if let Some(tool_permissions) = payload.tool_permissions {
        for update in tool_permissions {
            permissions::replace_tool_tags(pool, &update.tool_id, update.permission_tags).await?;
        }
    }

    let settings = row_to_tool_settings_private(row).map_err(ApiError::from)?;
    settings.to_response(pool).await.map_err(ApiError::from)
}

pub async fn update_app_settings(
    pool: &SqlitePool,
    payload: UpdateAppSettingsRequest,
) -> Result<AppSettingsResponse, ApiError> {
    if payload.signup_limit.is_some_and(|value| value < 0) {
        return Err(ApiError::bad_request(
            "invalid_signup_limit",
            "Signup limit cannot be negative",
        ));
    }

    if payload.max_upload_bytes.is_some_and(|value| value < 1) {
        return Err(ApiError::bad_request(
            "invalid_upload_limit",
            "Upload limit must be positive",
        ));
    }

    if payload.request_timeout_ms.is_some_and(|value| value < 1) {
        return Err(ApiError::bad_request(
            "invalid_timeout",
            "Request timeout must be positive",
        ));
    }

    let update_channel = payload
        .update_channel
        .as_deref()
        .map(validate_update_channel)
        .transpose()?;

    let row = sqlx::query(
        r#"
        UPDATE app_settings
        SET allow_signup = COALESCE(?, allow_signup),
            signup_limit = COALESCE(?, signup_limit),
            max_upload_bytes = COALESCE(?, max_upload_bytes),
            request_timeout_ms = COALESCE(?, request_timeout_ms),
            update_channel = COALESCE(?, update_channel),
            updated_at = ?
        WHERE id = 1
        RETURNING allow_signup,
                  signup_limit,
                  signup_count,
                  max_upload_bytes,
                  request_timeout_ms,
                  network_mode,
                  public_base_url,
                  trust_proxy_headers,
                  network_recovery_notice,
                  update_channel
        "#,
    )
    .bind(payload.allow_signup.map(i64::from))
    .bind(payload.signup_limit)
    .bind(payload.max_upload_bytes)
    .bind(payload.request_timeout_ms)
    .bind(update_channel)
    .bind(unix_timestamp())
    .fetch_one(pool)
    .await?;

    row_to_app_settings(row).map_err(ApiError::from)
}

pub async fn update_network_settings(
    pool: &SqlitePool,
    admin_user_id: &str,
    payload: UpdateNetworkSettingsRequest,
) -> Result<AppSettingsResponse, ApiError> {
    if !payload.acknowledge_risk {
        return Err(ApiError::bad_request(
            "network_risk_not_acknowledged",
            "Network setting changes require confirmation",
        ));
    }

    if !auth_service::verify_user_password(pool, admin_user_id, &payload.admin_password).await? {
        return Err(ApiError::invalid_credentials());
    }

    let network_mode = validate_network_mode(&payload.network_mode)?;
    let public_base_url = normalize_public_base_url(payload.public_base_url)?;
    let trust_proxy_headers = network_mode == "public_https_proxy" && payload.trust_proxy_headers;

    let row = sqlx::query(
        r#"
        UPDATE app_settings
        SET network_mode = ?,
            public_base_url = ?,
            trust_proxy_headers = ?,
            updated_at = ?
        WHERE id = 1
        RETURNING allow_signup,
                  signup_limit,
                  signup_count,
                  max_upload_bytes,
                  request_timeout_ms,
                  network_mode,
                  public_base_url,
                  trust_proxy_headers,
                  network_recovery_notice,
                  update_channel
        "#,
    )
    .bind(network_mode)
    .bind(public_base_url)
    .bind(i64::from(trust_proxy_headers))
    .bind(unix_timestamp())
    .fetch_one(pool)
    .await?;

    row_to_app_settings(row).map_err(ApiError::from)
}

pub async fn reset_network_settings_for_recovery(
    pool: &SqlitePool,
    notice: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE app_settings
        SET network_mode = 'lan_http',
            public_base_url = NULL,
            trust_proxy_headers = 0,
            network_recovery_notice = ?,
            updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(notice)
    .bind(unix_timestamp())
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn dismiss_network_recovery_notice(
    pool: &SqlitePool,
) -> Result<AppSettingsResponse, ApiError> {
    let row = sqlx::query(
        r#"
        UPDATE app_settings
        SET network_recovery_notice = NULL,
            updated_at = ?
        WHERE id = 1
        RETURNING allow_signup,
                  signup_limit,
                  signup_count,
                  max_upload_bytes,
                  request_timeout_ms,
                  network_mode,
                  public_base_url,
                  trust_proxy_headers,
                  network_recovery_notice,
                  update_channel
        "#,
    )
    .bind(unix_timestamp())
    .fetch_one(pool)
    .await?;

    row_to_app_settings(row).map_err(ApiError::from)
}

pub async fn get_user_settings(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<UserSettingsResponse, ApiError> {
    ensure_user_settings(pool, user_id).await?;

    let row = sqlx::query(
        r#"
        SELECT default_backend_id, default_model_name, theme
        FROM user_settings
        WHERE user_id = ?
        "#,
    )
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    row_to_user_settings(row).map_err(ApiError::from)
}

pub async fn update_user_settings(
    pool: &SqlitePool,
    user_id: &str,
    payload: UpdateUserSettingsRequest,
) -> Result<UserSettingsResponse, ApiError> {
    ensure_user_settings(pool, user_id).await?;

    if payload.default_backend_id.is_none()
        && payload.default_model_name.is_none()
        && payload.theme.is_none()
    {
        return Err(ApiError::bad_request(
            "empty_update",
            "No user settings changes were provided",
        ));
    }

    let current = get_user_settings(pool, user_id).await?;
    let default_backend_id =
        normalize_optional_string_update(payload.default_backend_id, current.default_backend_id)?;
    let default_model_name =
        normalize_optional_string_update(payload.default_model_name, current.default_model_name)?;
    let theme = normalize_theme_update(payload.theme, current.theme)?;

    if let Some(backend_id) = &default_backend_id {
        ensure_enabled_backend(pool, backend_id).await?;
    }

    let row = sqlx::query(
        r#"
        UPDATE user_settings
        SET default_backend_id = ?,
            default_model_name = ?,
            theme = ?,
            updated_at = ?
        WHERE user_id = ?
        RETURNING default_backend_id, default_model_name, theme
        "#,
    )
    .bind(default_backend_id)
    .bind(default_model_name)
    .bind(theme)
    .bind(unix_timestamp())
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    row_to_user_settings(row).map_err(ApiError::from)
}

fn row_to_app_settings(row: sqlx::sqlite::SqliteRow) -> Result<AppSettingsResponse, sqlx::Error> {
    Ok(AppSettingsResponse {
        allow_signup: row.try_get::<i64, _>("allow_signup")? != 0,
        signup_limit: row.try_get("signup_limit")?,
        signup_count: row.try_get("signup_count")?,
        max_upload_bytes: row.try_get("max_upload_bytes")?,
        request_timeout_ms: row.try_get("request_timeout_ms")?,
        network_mode: row.try_get("network_mode")?,
        public_base_url: row.try_get("public_base_url")?,
        trust_proxy_headers: row.try_get::<i64, _>("trust_proxy_headers")? != 0,
        network_recovery_notice: row.try_get("network_recovery_notice")?,
        update_channel: row.try_get("update_channel")?,
    })
}

pub async fn get_update_channel(pool: &SqlitePool) -> Result<String, sqlx::Error> {
    sqlx::query_scalar("SELECT update_channel FROM app_settings WHERE id = 1")
        .fetch_one(pool)
        .await
}

fn validate_update_channel(value: &str) -> Result<&str, ApiError> {
    match value {
        "stable" | "prerelease" => Ok(value),
        _ => Err(ApiError::bad_request(
            "invalid_update_channel",
            "Update channel must be stable or prerelease",
        )),
    }
}

fn row_to_tool_settings_private(
    row: sqlx::sqlite::SqliteRow,
) -> Result<ToolSettingsPrivate, sqlx::Error> {
    Ok(ToolSettingsPrivate {
        tools_enabled: row.try_get::<i64, _>("tools_enabled")? != 0,
        ollama_web_search_enabled: row.try_get::<i64, _>("ollama_web_search_enabled")? != 0,
        ollama_web_fetch_enabled: row.try_get::<i64, _>("ollama_web_fetch_enabled")? != 0,
        ollama_api_key: row.try_get("ollama_api_key")?,
        brave_search_enabled: row.try_get::<i64, _>("brave_search_enabled")? != 0,
        brave_search_api_key: row.try_get("brave_search_api_key")?,
        direct_web_fetch_enabled: row.try_get::<i64, _>("direct_web_fetch_enabled")? != 0,
        tool_system_prompt: prompt_or_default(
            row.try_get("tool_system_prompt")?,
            DEFAULT_TOOL_SYSTEM_PROMPT,
        ),
        web_search_tool_prompt: prompt_or_default(
            row.try_get("web_search_tool_prompt")?,
            DEFAULT_WEB_SEARCH_TOOL_PROMPT,
        ),
        web_fetch_tool_prompt: prompt_or_default(
            row.try_get("web_fetch_tool_prompt")?,
            DEFAULT_WEB_FETCH_TOOL_PROMPT,
        ),
    })
}

impl ToolSettingsPrivate {
    pub async fn to_response(
        &self,
        pool: &SqlitePool,
    ) -> Result<ToolSettingsResponse, sqlx::Error> {
        let available_tags = permissions::known_tags(pool).await?;
        let default_tag_ids = permissions::default_tool_tag_ids(pool).await?;
        let default_tool_permission_tags =
            permissions::tag_responses(pool, &default_tag_ids).await?;
        let tags_by_tool = permissions::tool_tags_by_tool(pool).await?;
        let mut tool_permissions = Vec::new();
        for tool_id in permissions::tool_ids() {
            let tags = tags_by_tool.get(tool_id).cloned().unwrap_or_default();
            tool_permissions.push(ToolPermissionResponse {
                tool_id,
                permission_tags: permissions::tag_responses(pool, &tags).await?,
            });
        }

        Ok(ToolSettingsResponse {
            tools_enabled: self.tools_enabled,
            ollama_web_search_enabled: self.ollama_web_search_enabled,
            ollama_web_fetch_enabled: self.ollama_web_fetch_enabled,
            ollama_api_key_configured: self.ollama_api_key.is_some(),
            brave_search_enabled: self.brave_search_enabled,
            brave_search_api_key_configured: self.brave_search_api_key.is_some(),
            direct_web_fetch_enabled: self.direct_web_fetch_enabled,
            tool_system_prompt: self.tool_system_prompt.clone(),
            default_tool_system_prompt: DEFAULT_TOOL_SYSTEM_PROMPT,
            web_search_tool_prompt: self.web_search_tool_prompt.clone(),
            default_web_search_tool_prompt: DEFAULT_WEB_SEARCH_TOOL_PROMPT,
            web_fetch_tool_prompt: self.web_fetch_tool_prompt.clone(),
            default_web_fetch_tool_prompt: DEFAULT_WEB_FETCH_TOOL_PROMPT,
            available_tags,
            default_tool_permission_tags,
            tool_permissions,
        })
    }

    pub fn has_ollama_key(&self) -> bool {
        self.ollama_api_key.is_some()
    }

    pub fn has_brave_key(&self) -> bool {
        self.brave_search_api_key.is_some()
    }
}

fn prompt_or_default(value: Option<String>, default: &'static str) -> String {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn updated_prompt(
    current: &str,
    next: Option<String>,
    default: &'static str,
) -> Result<String, ApiError> {
    let Some(next) = next else {
        return Ok(current.to_string());
    };
    let next = next.trim().to_string();
    if next.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_tool_prompt",
            "Tool prompts cannot be empty",
        ));
    }
    if next == default {
        Ok(default.to_string())
    } else {
        Ok(next)
    }
}

fn updated_secret(
    current: Option<String>,
    next: Option<String>,
    clear: Option<bool>,
) -> Option<String> {
    if clear.unwrap_or(false) {
        return None;
    }

    next.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or(current)
}

fn validate_network_mode(network_mode: &str) -> Result<&'static str, ApiError> {
    match network_mode {
        "lan_http" => Ok("lan_http"),
        "public_https_proxy" => Ok("public_https_proxy"),
        _ => Err(ApiError::bad_request(
            "invalid_network_mode",
            "Network mode is invalid",
        )),
    }
}

fn normalize_public_base_url(public_base_url: Option<String>) -> Result<Option<String>, ApiError> {
    let Some(public_base_url) = public_base_url else {
        return Ok(None);
    };
    let public_base_url = public_base_url.trim();
    if public_base_url.is_empty() {
        return Ok(None);
    }

    let mut parsed = reqwest::Url::parse(public_base_url).map_err(|_| {
        ApiError::bad_request(
            "invalid_public_base_url",
            "Public base URL must be a valid HTTPS URL",
        )
    })?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(ApiError::bad_request(
            "invalid_public_base_url",
            "Public base URL must be an HTTPS origin without a path, query, or credentials",
        ));
    }

    parsed.set_path("");
    Ok(Some(parsed.as_str().trim_end_matches('/').to_string()))
}

async fn ensure_user_settings(pool: &SqlitePool, user_id: &str) -> Result<(), sqlx::Error> {
    let now = unix_timestamp();

    sqlx::query(
        r#"
        INSERT OR IGNORE INTO user_settings (user_id, created_at, updated_at)
        VALUES (?, ?, ?)
        "#,
    )
    .bind(user_id)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(())
}

async fn ensure_enabled_backend(pool: &SqlitePool, backend_id: &str) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
            SELECT 1
            FROM ollama_backends
            WHERE id = ?
              AND is_enabled = 1
        )
        "#,
    )
    .bind(backend_id)
    .fetch_one(pool)
    .await?;

    if exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_backend",
            "Default backend must be enabled",
        ));
    }

    Ok(())
}

fn normalize_optional_string_update(
    update: Option<serde_json::Value>,
    current: Option<String>,
) -> Result<Option<String>, ApiError> {
    match update {
        Some(serde_json::Value::String(value)) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                Ok(Some(value.to_string()))
            }
        }
        Some(serde_json::Value::Null) => Ok(None),
        Some(_) => Err(ApiError::bad_request(
            "invalid_user_setting",
            "User setting value must be a string or null",
        )),
        None => Ok(current),
    }
}

fn normalize_theme_update(
    update: Option<serde_json::Value>,
    current: Option<String>,
) -> Result<Option<String>, ApiError> {
    let Some(update) = update else {
        return Ok(current);
    };

    match update {
        serde_json::Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Ok(None);
            }

            match value {
                "system" | "vashti" | "light" => Ok(Some(value.to_string())),
                _ => Err(ApiError::bad_request(
                    "invalid_theme",
                    "Theme is not available",
                )),
            }
        }
        serde_json::Value::Null => Ok(None),
        _ => Err(ApiError::bad_request(
            "invalid_theme",
            "Theme must be text or null",
        )),
    }
}

fn row_to_user_settings(row: sqlx::sqlite::SqliteRow) -> Result<UserSettingsResponse, sqlx::Error> {
    Ok(UserSettingsResponse {
        default_backend_id: row.try_get("default_backend_id")?,
        default_model_name: row.try_get("default_model_name")?,
        theme: row.try_get("theme")?,
    })
}

#[cfg(test)]
mod tests {
    use super::normalize_public_base_url;

    #[test]
    fn public_base_url_is_normalized_to_an_https_origin() {
        assert_eq!(
            normalize_public_base_url(Some(" https://chat.example.com/ ".to_string()))
                .expect("valid public URL")
                .as_deref(),
            Some("https://chat.example.com")
        );
        assert_eq!(
            normalize_public_base_url(Some("https://chat.example.com:8443".to_string()))
                .expect("valid public URL with port")
                .as_deref(),
            Some("https://chat.example.com:8443")
        );
    }

    #[test]
    fn public_base_url_rejects_non_origin_values() {
        for value in [
            "http://chat.example.com",
            "https://user:password@chat.example.com",
            "https://chat.example.com/app",
            "https://chat.example.com?mode=public",
            "https://chat.example.com/#section",
        ] {
            assert!(
                normalize_public_base_url(Some(value.to_string())).is_err(),
                "{value} should be rejected"
            );
        }
    }
}
