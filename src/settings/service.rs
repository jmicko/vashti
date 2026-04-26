use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{
    auth::service::unix_timestamp, error::ApiError, settings::handlers::UpdateAppSettingsRequest,
};

#[derive(Debug, Clone, Serialize)]
pub struct AppSettingsResponse {
    pub allow_signup: bool,
    pub signup_limit: i64,
    pub signup_count: i64,
    pub max_upload_bytes: i64,
    pub request_timeout_ms: i64,
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

fn row_to_app_settings(row: sqlx::sqlite::SqliteRow) -> Result<AppSettingsResponse, sqlx::Error> {
    Ok(AppSettingsResponse {
        allow_signup: row.try_get::<i64, _>("allow_signup")? != 0,
        signup_limit: row.try_get("signup_limit")?,
        signup_count: row.try_get("signup_count")?,
        max_upload_bytes: row.try_get("max_upload_bytes")?,
        request_timeout_ms: row.try_get("request_timeout_ms")?,
    })
}
