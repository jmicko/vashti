use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{
    auth::service::unix_timestamp,
    error::ApiError,
    settings::handlers::{UpdateAppSettingsRequest, UpdateUserSettingsRequest},
};

#[derive(Debug, Clone, Serialize)]
pub struct AppSettingsResponse {
    pub allow_signup: bool,
    pub signup_limit: i64,
    pub signup_count: i64,
    pub max_upload_bytes: i64,
    pub request_timeout_ms: i64,
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
        SELECT allow_signup, signup_limit, signup_count, max_upload_bytes, request_timeout_ms
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
        RETURNING allow_signup, signup_limit, signup_count, max_upload_bytes, request_timeout_ms
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
    })
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
