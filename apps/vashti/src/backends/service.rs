use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, Ipv4Addr, UdpSocket},
};

use serde::Serialize;
use sqlx::{Row, SqlitePool};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::{auth::service::unix_timestamp, error::ApiError, ollama};

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

#[derive(Debug, Serialize)]
pub struct DetectedBackendResponse {
    pub name: String,
    pub base_url: String,
}

#[derive(Debug, Serialize)]
pub struct ModelAvailabilityResponse {
    pub backend_id: String,
    pub model_name: String,
    pub is_enabled: bool,
}

#[derive(Debug)]
pub struct UpdateBackendParams {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub is_enabled: Option<bool>,
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

    let mut seen = HashSet::new();
    let mut backends = Vec::new();

    for row in rows {
        let backend = row_to_backend(row)?;
        if dedupe_key(&backend.base_url).is_some_and(|key| !seen.insert(key)) {
            continue;
        }
        backends.push(backend);
    }

    Ok(backends)
}

pub async fn create_backend(
    pool: &SqlitePool,
    name: String,
    base_url: String,
) -> Result<BackendResponse, ApiError> {
    let name = validate_name(&name)?;
    let base_url = validate_base_url(&base_url)?;
    ensure_unique_local_backend(pool, &base_url, None).await?;
    let now = unix_timestamp();

    let insert = sqlx::query(
        r#"
        INSERT INTO ollama_backends (
            id,
            name,
            base_url,
            is_enabled,
            is_localhost_detected,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, 1, 0, ?, ?)
        RETURNING id, name, base_url, is_enabled, last_health_status, last_error
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(name)
    .bind(base_url)
    .bind(now)
    .bind(now)
    .fetch_one(pool)
    .await;

    match insert {
        Ok(row) => row_to_backend(row).map_err(ApiError::from),
        Err(error) => handle_backend_write_error(error),
    }
}

pub async fn update_backend(
    pool: &SqlitePool,
    backend_id: &str,
    params: UpdateBackendParams,
) -> Result<BackendResponse, ApiError> {
    let Some(current) = sqlx::query(
        r#"
        SELECT id, name, base_url, is_enabled, last_health_status, last_error
        FROM ollama_backends
        WHERE id = ?
        "#,
    )
    .bind(backend_id)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::not_found(
            "backend_not_found",
            "Backend not found",
        ));
    };

    let name = match params.name {
        Some(name) => validate_name(&name)?,
        None => current.try_get("name")?,
    };
    let base_url = match params.base_url {
        Some(base_url) => validate_base_url(&base_url)?,
        None => current.try_get("base_url")?,
    };
    ensure_unique_local_backend(pool, &base_url, Some(backend_id)).await?;
    let is_enabled = params
        .is_enabled
        .map(i64::from)
        .unwrap_or(current.try_get("is_enabled")?);

    let update = sqlx::query(
        r#"
        UPDATE ollama_backends
        SET name = ?,
            base_url = ?,
            is_enabled = ?,
            updated_at = ?
        WHERE id = ?
        RETURNING id, name, base_url, is_enabled, last_health_status, last_error
        "#,
    )
    .bind(name)
    .bind(base_url)
    .bind(is_enabled)
    .bind(unix_timestamp())
    .bind(backend_id)
    .fetch_one(pool)
    .await;

    match update {
        Ok(row) => row_to_backend(row).map_err(ApiError::from),
        Err(error) => handle_backend_write_error(error),
    }
}

pub async fn delete_backend(pool: &SqlitePool, backend_id: &str) -> Result<(), ApiError> {
    let delete = sqlx::query("DELETE FROM ollama_backends WHERE id = ?")
        .bind(backend_id)
        .execute(pool)
        .await;

    let result = match delete {
        Ok(result) => result,
        Err(error) => {
            if let sqlx::Error::Database(_) = error {
                return Err(ApiError::conflict(
                    "backend_in_use",
                    "Backend is still in use",
                ));
            }
            return Err(error.into());
        }
    };

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found(
            "backend_not_found",
            "Backend not found",
        ));
    }

    Ok(())
}

pub async fn detect_localhost_backends(
    pool: &SqlitePool,
    client: &reqwest::Client,
) -> Result<Vec<DetectedBackendResponse>, ApiError> {
    for base_url in ["http://127.0.0.1:11434", "http://localhost:11434"] {
        match ollama::client::is_reachable(client, base_url).await {
            Ok(true) => {
                let backend = ensure_detected_backend(pool, base_url, "localhost").await?;
                return Ok(vec![backend]);
            }
            Ok(false) => {}
            Err(error) => {
                tracing::debug!(base_url, ?error, "manual local Ollama detection failed");
            }
        }
    }

    Ok(Vec::new())
}

pub async fn scan_local_network_backends(
    pool: &SqlitePool,
    client: &reqwest::Client,
) -> Result<Vec<DetectedBackendResponse>, ApiError> {
    let Some(local_ip) = local_ipv4_address() else {
        return Ok(Vec::new());
    };

    if local_ip.is_loopback() {
        return Ok(Vec::new());
    }

    let [a, b, c, own] = local_ip.octets();
    let mut tasks = JoinSet::new();

    for host in 1..=254u8 {
        if host == own {
            continue;
        }

        let client = client.clone();
        let base_url = format!("http://{a}.{b}.{c}.{host}:11434");
        tasks.spawn(async move {
            match ollama::client::is_reachable(&client, &base_url).await {
                Ok(true) => Some(base_url),
                Ok(false) | Err(_) => None,
            }
        });
    }

    let mut detected = Vec::new();
    while let Some(result) = tasks.join_next().await {
        let Ok(Some(base_url)) = result else {
            continue;
        };
        let name = scan_backend_name(&base_url);
        detected.push(ensure_detected_backend(pool, &base_url, &name).await?);
    }

    detected.sort_by(|left, right| left.base_url.cmp(&right.base_url));

    Ok(detected)
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

    let mut seen = HashSet::new();
    let mut backends = Vec::new();

    for row in rows {
        let backend = OllamaBackend {
            id: row.try_get("id")?,
            name: row.try_get("name")?,
            base_url: row.try_get("base_url")?,
        };
        if dedupe_key(&backend.base_url).is_some_and(|key| !seen.insert(key)) {
            continue;
        }
        backends.push(backend);
    }

    Ok(backends)
}

pub async fn model_availability_by_backend(
    pool: &SqlitePool,
    backend_id: &str,
) -> Result<HashMap<String, bool>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT model_name, is_enabled
        FROM model_availability
        WHERE backend_id = ?
        "#,
    )
    .bind(backend_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok((
                row.try_get("model_name")?,
                row.try_get::<i64, _>("is_enabled")? != 0,
            ))
        })
        .collect()
}

pub async fn set_model_availability(
    pool: &SqlitePool,
    backend_id: &str,
    model_name: &str,
    is_enabled: bool,
) -> Result<ModelAvailabilityResponse, ApiError> {
    ensure_backend_exists(pool, backend_id).await?;
    let model_name = validate_model_name(model_name)?;
    let now = unix_timestamp();

    sqlx::query(
        r#"
        INSERT INTO model_availability (
            backend_id,
            model_name,
            is_enabled,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(backend_id, model_name)
        DO UPDATE SET
            is_enabled = excluded.is_enabled,
            updated_at = excluded.updated_at
        "#,
    )
    .bind(backend_id)
    .bind(&model_name)
    .bind(i64::from(is_enabled))
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(ModelAvailabilityResponse {
        backend_id: backend_id.to_string(),
        model_name,
        is_enabled,
    })
}

pub async fn set_model_availability_many(
    pool: &SqlitePool,
    backend_id: &str,
    model_names: &[String],
    is_enabled: bool,
) -> Result<(), ApiError> {
    ensure_backend_exists(pool, backend_id).await?;
    let model_names = model_names
        .iter()
        .map(|model_name| validate_model_name(model_name))
        .collect::<Result<Vec<_>, _>>()?;
    if model_names.is_empty() {
        return Ok(());
    }

    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    for model_name in model_names {
        sqlx::query(
            r#"
            INSERT INTO model_availability (
                backend_id,
                model_name,
                is_enabled,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(backend_id, model_name)
            DO UPDATE SET
                is_enabled = excluded.is_enabled,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(backend_id)
        .bind(model_name)
        .bind(i64::from(is_enabled))
        .bind(now)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;

    Ok(())
}

pub async fn ensure_model_enabled(
    pool: &SqlitePool,
    backend_id: &str,
    model_name: &str,
) -> Result<(), ApiError> {
    let model_name = validate_model_name(model_name)?;
    let is_enabled = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT is_enabled
        FROM model_availability
        WHERE backend_id = ?
          AND model_name = ?
        "#,
    )
    .bind(backend_id)
    .bind(model_name)
    .fetch_optional(pool)
    .await?;

    if is_enabled.unwrap_or(1) == 0 {
        return Err(ApiError::forbidden(
            "model_disabled",
            "This model is disabled by the server admin",
        ));
    }

    Ok(())
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

async fn ensure_detected_backend(
    pool: &SqlitePool,
    base_url: &str,
    base_name: &str,
) -> Result<DetectedBackendResponse, ApiError> {
    let base_url = normalize_base_url(base_url);

    if is_localhost_base_url(&base_url) {
        if let Some(row) = find_equivalent_local_backend(pool, &base_url, None).await? {
            let backend_id: String = row.try_get("id")?;
            record_backend_health(pool, &backend_id, "ok", None).await?;
            return Ok(row_to_detected_backend(row)?);
        }
    }

    if let Some(row) = sqlx::query(
        r#"
        SELECT id, name, base_url
        FROM ollama_backends
        WHERE base_url = ?
        LIMIT 1
        "#,
    )
    .bind(&base_url)
    .fetch_optional(pool)
    .await?
    {
        let backend_id: String = row.try_get("id")?;
        record_backend_health(pool, &backend_id, "ok", None).await?;
        return Ok(DetectedBackendResponse {
            name: row.try_get("name")?,
            base_url: row.try_get("base_url")?,
        });
    }

    let now = unix_timestamp();
    let name = next_available_name(pool, base_name).await?;
    let row = sqlx::query(
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
        VALUES (?, ?, ?, 1, 1, ?, ?, ?, 'ok')
        RETURNING name, base_url
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(name)
    .bind(base_url)
    .bind(now)
    .bind(now)
    .bind(now)
    .fetch_one(pool)
    .await?;

    Ok(DetectedBackendResponse {
        name: row.try_get("name")?,
        base_url: row.try_get("base_url")?,
    })
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

fn local_ipv4_address() -> Option<Ipv4Addr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;

    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(address) => Some(address),
        IpAddr::V6(_) => None,
    }
}

fn scan_backend_name(base_url: &str) -> String {
    reqwest::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .map(|host| format!("host-{}", host.replace('.', "-")))
        .unwrap_or_else(|| "host".to_string())
}

async fn ensure_unique_local_backend(
    pool: &SqlitePool,
    base_url: &str,
    exclude_backend_id: Option<&str>,
) -> Result<(), ApiError> {
    if !is_localhost_base_url(base_url) {
        return Ok(());
    }

    if find_equivalent_local_backend(pool, base_url, exclude_backend_id)
        .await?
        .is_some()
    {
        return Err(ApiError::conflict(
            "backend_exists",
            "A localhost backend is already configured",
        ));
    }

    Ok(())
}

async fn find_equivalent_local_backend(
    pool: &SqlitePool,
    base_url: &str,
    exclude_backend_id: Option<&str>,
) -> Result<Option<sqlx::sqlite::SqliteRow>, sqlx::Error> {
    let Some(key) = localhost_key(base_url) else {
        return Ok(None);
    };

    let rows = sqlx::query(
        r#"
        SELECT id, name, base_url
        FROM ollama_backends
        ORDER BY created_at ASC, name ASC
        "#,
    )
    .fetch_all(pool)
    .await?;

    for row in rows {
        let id: String = row.try_get("id")?;
        if exclude_backend_id.is_some_and(|exclude| exclude == id) {
            continue;
        }

        let existing_base_url: String = row.try_get("base_url")?;
        if localhost_key(&existing_base_url).as_deref() == Some(key.as_str()) {
            return Ok(Some(row));
        }
    }

    Ok(None)
}

fn dedupe_key(base_url: &str) -> Option<String> {
    localhost_key(base_url)
}

fn is_localhost_base_url(base_url: &str) -> bool {
    localhost_key(base_url).is_some()
}

fn localhost_key(base_url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(base_url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    if host != "localhost" && host != "127.0.0.1" {
        return None;
    }

    Some(format!(
        "{}://localhost:{}",
        parsed.scheme(),
        parsed.port_or_known_default()?
    ))
}

fn validate_name(name: &str) -> Result<String, ApiError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_backend_name",
            "Name is required",
        ));
    }

    Ok(name.to_string())
}

fn validate_model_name(model_name: &str) -> Result<String, ApiError> {
    let model_name = model_name.trim();
    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "Model name is required",
        ));
    }

    Ok(model_name.to_string())
}

async fn ensure_backend_exists(pool: &SqlitePool, backend_id: &str) -> Result<(), ApiError> {
    let exists: i64 =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM ollama_backends WHERE id = ?)")
            .bind(backend_id)
            .fetch_one(pool)
            .await?;

    if exists == 0 {
        return Err(ApiError::not_found(
            "backend_not_found",
            "Backend not found",
        ));
    }

    Ok(())
}

fn validate_base_url(base_url: &str) -> Result<String, ApiError> {
    let base_url = normalize_base_url(base_url);
    let parsed = reqwest::Url::parse(&base_url).map_err(|_| {
        ApiError::bad_request("invalid_backend_url", "Base URL must be a valid URL")
    })?;

    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(ApiError::bad_request(
            "invalid_backend_url",
            "Base URL must include http(s) scheme and host",
        ));
    }

    Ok(base_url)
}

async fn next_available_name(pool: &SqlitePool, base_name: &str) -> Result<String, sqlx::Error> {
    for suffix in 0..100 {
        let candidate = if suffix == 0 {
            base_name.to_string()
        } else {
            format!("{base_name}-{suffix}")
        };

        let exists: i64 =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM ollama_backends WHERE name = ?)")
                .bind(&candidate)
                .fetch_one(pool)
                .await?;

        if exists == 0 {
            return Ok(candidate);
        }
    }

    Ok(format!("{}-{}", base_name, Uuid::new_v4()))
}

fn row_to_backend(row: sqlx::sqlite::SqliteRow) -> Result<BackendResponse, sqlx::Error> {
    Ok(BackendResponse {
        id: row.try_get("id")?,
        name: row.try_get("name")?,
        base_url: row.try_get("base_url")?,
        is_enabled: row.try_get::<i64, _>("is_enabled")? != 0,
        last_health_status: row.try_get("last_health_status")?,
        last_error: row.try_get("last_error")?,
    })
}

fn row_to_detected_backend(
    row: sqlx::sqlite::SqliteRow,
) -> Result<DetectedBackendResponse, sqlx::Error> {
    Ok(DetectedBackendResponse {
        name: row.try_get("name")?,
        base_url: row.try_get("base_url")?,
    })
}

fn handle_backend_write_error(error: sqlx::Error) -> Result<BackendResponse, ApiError> {
    if let sqlx::Error::Database(database_error) = &error {
        if database_error.is_unique_violation() {
            return Err(ApiError::conflict(
                "backend_exists",
                "A backend with that name already exists",
            ));
        }
    }

    Err(error.into())
}
