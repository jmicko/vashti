use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{
    auth::service::{self as auth_service, unix_timestamp},
    error::ApiError,
    settings::handlers::{
        UpdateAppSettingsRequest, UpdateNetworkSettingsRequest, UpdateUserSettingsRequest,
    },
};

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
               network_recovery_notice
        FROM app_settings
        WHERE id = 1
        "#,
    )
    .fetch_one(pool)
    .await?;

    row_to_app_settings(row)
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

    let row = sqlx::query(
        r#"
        UPDATE app_settings
        SET allow_signup = COALESCE(?, allow_signup),
            signup_limit = COALESCE(?, signup_limit),
            max_upload_bytes = COALESCE(?, max_upload_bytes),
            request_timeout_ms = COALESCE(?, request_timeout_ms),
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
                  network_recovery_notice
        "#,
    )
    .bind(payload.allow_signup.map(i64::from))
    .bind(payload.signup_limit)
    .bind(payload.max_upload_bytes)
    .bind(payload.request_timeout_ms)
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
                  network_recovery_notice
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
                  network_recovery_notice
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
    let theme = normalize_optional_string_update(payload.theme, current.theme)?;

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
    })
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
    let public_base_url = public_base_url.trim().trim_end_matches('/').to_string();
    if public_base_url.is_empty() {
        return Ok(None);
    }

    let parsed = reqwest::Url::parse(&public_base_url).map_err(|_| {
        ApiError::bad_request(
            "invalid_public_base_url",
            "Public base URL must be a valid HTTPS URL",
        )
    })?;
    if parsed.scheme() != "https" || parsed.host_str().is_none() {
        return Err(ApiError::bad_request(
            "invalid_public_base_url",
            "Public base URL must be a valid HTTPS URL",
        ));
    }

    Ok(Some(public_base_url))
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

fn row_to_user_settings(row: sqlx::sqlite::SqliteRow) -> Result<UserSettingsResponse, sqlx::Error> {
    Ok(UserSettingsResponse {
        default_backend_id: row.try_get("default_backend_id")?,
        default_model_name: row.try_get("default_model_name")?,
        theme: row.try_get("theme")?,
    })
}
