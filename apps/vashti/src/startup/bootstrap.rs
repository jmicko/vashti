use sqlx::{Row, SqlitePool};

use crate::{backends, ollama};

pub async fn ensure_app_settings(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let now = crate::auth::service::unix_timestamp();

    sqlx::query(
        r#"
        INSERT INTO app_settings (id, allow_signup, signup_limit, signup_count, created_at, updated_at)
        VALUES (1, 1, 25, 0, ?, ?)
        ON CONFLICT(id) DO NOTHING
        "#,
    )
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn detect_localhost_ollama_if_empty(
    pool: &SqlitePool,
    client: &reqwest::Client,
) -> Result<(), sqlx::Error> {
    let count: i64 = sqlx::query("SELECT COUNT(*) AS count FROM ollama_backends")
        .fetch_one(pool)
        .await?
        .try_get("count")?;

    if count > 0 {
        return Ok(());
    }

    for base_url in ["http://127.0.0.1:11434", "http://localhost:11434"] {
        match ollama::client::is_reachable(client, base_url).await {
            Ok(true) => {
                backends::service::insert_detected_localhost_backend(pool, base_url).await?;
                tracing::info!(base_url, "detected local Ollama backend");
                return Ok(());
            }
            Ok(false) => {}
            Err(error) => {
                tracing::debug!(base_url, ?error, "local Ollama detection failed");
            }
        }
    }

    Ok(())
}
