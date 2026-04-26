use serde::Serialize;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::auth::service::unix_timestamp;

#[derive(Debug, Clone)]
pub struct OllamaBackend {
    pub id: String,
    pub name: String,
    pub base_url: String,
}

#[derive(Debug, Serialize)]
pub struct BackendResponse {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub is_enabled: bool,
    pub last_health_status: Option<String>,
    pub last_error: Option<String>,
}

pub async fn list_backends(pool: &SqlitePool) -> Result<Vec<BackendResponse>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT id, name, base_url, is_enabled, last_health_status, last_error
        FROM ollama_backends
        ORDER BY name ASC
        "#,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(BackendResponse {
                id: row.try_get("id")?,
                name: row.try_get("name")?,
                base_url: row.try_get("base_url")?,
                is_enabled: row.try_get::<i64, _>("is_enabled")? != 0,
                last_health_status: row.try_get("last_health_status")?,
                last_error: row.try_get("last_error")?,
            })
        })
        .collect()
}

pub async fn list_enabled_backends(pool: &SqlitePool) -> Result<Vec<OllamaBackend>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT id, name, base_url
        FROM ollama_backends
        WHERE is_enabled = 1
        ORDER BY name ASC
        "#,
    )
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(OllamaBackend {
                id: row.try_get("id")?,
                name: row.try_get("name")?,
                base_url: row.try_get("base_url")?,
            })
        })
        .collect()
}

pub async fn insert_detected_localhost_backend(
    pool: &SqlitePool,
    base_url: &str,
) -> Result<(), sqlx::Error> {
    let now = unix_timestamp();
    let base_url = normalize_base_url(base_url);

    sqlx::query(
        r#"
        INSERT INTO ollama_backends (
            id,
            name,
            base_url,
            is_enabled,
            is_localhost_detected,
            created_at,
            updated_at,
            last_healthcheck_at,
            last_health_status
        )
        VALUES (?, 'localhost', ?, 1, 1, ?, ?, ?, 'ok')
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(base_url)
    .bind(now)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn record_backend_health(
    pool: &SqlitePool,
    backend_id: &str,
    status: &str,
    error: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE ollama_backends
        SET last_healthcheck_at = ?,
            last_health_status = ?,
            last_error = ?
        WHERE id = ?
        "#,
    )
    .bind(unix_timestamp())
    .bind(status)
    .bind(error)
    .bind(backend_id)
    .execute(pool)
    .await?;

    Ok(())
}

fn normalize_base_url(base_url: &str) -> String {
    base_url.trim().trim_end_matches('/').to_string()
}
