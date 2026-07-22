use sqlx::{Row, SqlitePool};
use uuid::Uuid;

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

pub async fn ensure_server_identity(pool: &SqlitePool) -> Result<String, sqlx::Error> {
    let instance_id = Uuid::new_v4().to_string();
    let now = crate::auth::service::unix_timestamp();

    sqlx::query(
        r#"
        INSERT INTO server_identity (id, instance_id, created_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO NOTHING
        "#,
    )
    .bind(instance_id)
    .bind(now)
    .execute(pool)
    .await?;

    sqlx::query_scalar("SELECT instance_id FROM server_identity WHERE id = 1")
        .fetch_one(pool)
        .await
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

#[cfg(test)]
mod tests {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    use super::*;
    use crate::startup;

    #[tokio::test]
    async fn server_identity_is_created_once_and_remains_stable() {
        let options = SqliteConnectOptions::new()
            .filename(":memory:")
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .expect("connect test database");
        startup::migrations::run(&pool)
            .await
            .expect("run migrations");

        let first = ensure_server_identity(&pool)
            .await
            .expect("create server identity");
        let second = ensure_server_identity(&pool)
            .await
            .expect("load server identity");

        assert_eq!(first, second);
        assert!(Uuid::parse_str(&first).is_ok());
    }
}
