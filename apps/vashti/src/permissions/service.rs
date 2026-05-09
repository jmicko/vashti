use std::collections::{HashMap, HashSet};

use serde::Serialize;
use sqlx::{Row, SqlitePool};

use crate::{auth::service::unix_timestamp, error::ApiError};

pub const TAG_EVERYONE: &str = "system:everyone";
pub const TAG_ADMIN: &str = "system:admin";

#[derive(Debug, Clone, Serialize)]
pub struct PermissionTagResponse {
    pub id: String,
    pub label: String,
    pub kind: String,
}

pub fn user_tag_id(user_id: &str) -> String {
    format!("user:{user_id}")
}

pub fn tool_ids() -> [&'static str; 4] {
    [
        crate::tools::service::TOOL_BRAVE_WEB_SEARCH,
        crate::tools::service::TOOL_OLLAMA_WEB_SEARCH,
        crate::tools::service::TOOL_OLLAMA_WEB_FETCH,
        crate::tools::service::TOOL_DIRECT_WEB_FETCH,
    ]
}

pub async fn effective_user_tag_ids(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<HashSet<String>, sqlx::Error> {
    let Some(row) = sqlx::query("SELECT id, role, is_disabled FROM users WHERE id = ?")
        .bind(user_id)
        .fetch_optional(pool)
        .await?
    else {
        return Ok(HashSet::new());
    };

    if row.try_get::<i64, _>("is_disabled")? != 0 {
        return Ok(HashSet::new());
    }

    let mut tags = HashSet::from([TAG_EVERYONE.to_string(), user_tag_id(user_id)]);
    if row.try_get::<String, _>("role")? == "admin" {
        tags.insert(TAG_ADMIN.to_string());
    }

    let rows = sqlx::query("SELECT tag_id FROM user_permission_tags WHERE user_id = ?")
        .bind(user_id)
        .fetch_all(pool)
        .await?;
    for row in rows {
        tags.insert(row.try_get("tag_id")?);
    }

    Ok(tags)
}

pub fn has_matching_tag(user_tags: &HashSet<String>, resource_tags: &[String]) -> bool {
    !resource_tags.is_empty() && resource_tags.iter().any(|tag| user_tags.contains(tag))
}

pub async fn known_tags(pool: &SqlitePool) -> Result<Vec<PermissionTagResponse>, sqlx::Error> {
    let mut tag_ids = HashSet::from([TAG_EVERYONE.to_string(), TAG_ADMIN.to_string()]);

    let users = sqlx::query("SELECT id, username FROM users ORDER BY username ASC")
        .fetch_all(pool)
        .await?;
    for row in &users {
        let id: String = row.try_get("id")?;
        tag_ids.insert(user_tag_id(&id));
    }

    for query in [
        "SELECT tag_id FROM user_permission_tags",
        "SELECT tag_id FROM model_permission_tags",
        "SELECT tag_id FROM tool_permission_tags",
    ] {
        let rows = sqlx::query(query).fetch_all(pool).await?;
        for row in rows {
            tag_ids.insert(row.try_get("tag_id")?);
        }
    }

    for column in [
        "default_model_permission_tags_json",
        "default_tool_permission_tags_json",
    ] {
        let value: String =
            sqlx::query_scalar(&format!("SELECT {column} FROM app_settings WHERE id = 1"))
                .fetch_one(pool)
                .await?;
        if let Ok(defaults) = serde_json::from_str::<Vec<String>>(&value) {
            tag_ids.extend(defaults);
        }
    }

    let user_labels = user_label_map(pool).await?;
    let mut tags = tag_ids
        .into_iter()
        .map(|tag_id| tag_response(&tag_id, &user_labels))
        .collect::<Vec<_>>();
    tags.sort_by(|left, right| {
        tag_sort_key(left)
            .cmp(&tag_sort_key(right))
            .then_with(|| left.label.cmp(&right.label))
    });

    Ok(tags)
}

pub async fn tag_responses(
    pool: &SqlitePool,
    tag_ids: &[String],
) -> Result<Vec<PermissionTagResponse>, sqlx::Error> {
    let user_labels = user_label_map(pool).await?;
    let mut tags = tag_ids
        .iter()
        .map(|tag_id| tag_response(tag_id, &user_labels))
        .collect::<Vec<_>>();
    tags.sort_by(|left, right| tag_sort_key(left).cmp(&tag_sort_key(right)));
    Ok(tags)
}

pub async fn normalize_tag_ids(
    pool: &SqlitePool,
    values: Vec<String>,
) -> Result<Vec<String>, ApiError> {
    let mut seen = HashSet::new();
    let mut tags = Vec::new();

    for value in values {
        let tag = normalize_tag_id(pool, &value).await?;
        if seen.insert(tag.clone()) {
            tags.push(tag);
        }
    }

    Ok(tags)
}

pub async fn replace_user_group_tags(
    pool: &SqlitePool,
    user_id: &str,
    values: Vec<String>,
) -> Result<(), ApiError> {
    ensure_user_exists(pool, user_id).await?;
    let tags = normalize_tag_ids(pool, values)
        .await?
        .into_iter()
        .filter(|tag| tag.starts_with("group:"))
        .collect::<Vec<_>>();
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM user_permission_tags WHERE user_id = ?")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;

    for tag in tags {
        sqlx::query(
            r#"
            INSERT INTO user_permission_tags (user_id, tag_id, created_at)
            VALUES (?, ?, ?)
            "#,
        )
        .bind(user_id)
        .bind(tag)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

pub async fn user_group_tags(pool: &SqlitePool, user_id: &str) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT tag_id
        FROM user_permission_tags
        WHERE user_id = ?
        ORDER BY tag_id ASC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(|row| row.try_get("tag_id")).collect()
}

pub async fn default_model_tag_ids(pool: &SqlitePool) -> Result<Vec<String>, sqlx::Error> {
    default_tag_ids(pool, "default_model_permission_tags_json").await
}

pub async fn default_tool_tag_ids(pool: &SqlitePool) -> Result<Vec<String>, sqlx::Error> {
    default_tag_ids(pool, "default_tool_permission_tags_json").await
}

pub async fn update_default_model_tags(
    pool: &SqlitePool,
    values: Vec<String>,
    apply_to_existing: bool,
) -> Result<Vec<String>, ApiError> {
    let tags = normalize_tag_ids(pool, values).await?;
    update_default_tags(
        pool,
        "default_model_permission_tags_json",
        &tags,
        apply_to_existing,
    )
    .await?;
    Ok(tags)
}

pub async fn update_default_tool_tags(
    pool: &SqlitePool,
    values: Vec<String>,
) -> Result<Vec<String>, ApiError> {
    let tags = normalize_tag_ids(pool, values).await?;
    let json = serde_json::to_string(&tags)
        .map_err(|_| ApiError::internal("Failed to serialize permission tags"))?;
    sqlx::query(
        r#"
        UPDATE app_settings
        SET default_tool_permission_tags_json = ?,
            updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(json)
    .bind(unix_timestamp())
    .execute(pool)
    .await?;
    Ok(tags)
}

pub async fn ensure_model_record(
    pool: &SqlitePool,
    backend_id: &str,
    model_name: &str,
) -> Result<(), sqlx::Error> {
    let now = unix_timestamp();
    let result = sqlx::query(
        r#"
        INSERT INTO model_availability (
            backend_id,
            model_name,
            is_enabled,
            created_at,
            updated_at
        )
        VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(backend_id, model_name) DO NOTHING
        "#,
    )
    .bind(backend_id)
    .bind(model_name)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    if result.rows_affected() > 0 {
        let defaults = default_model_tag_ids(pool).await?;
        replace_model_tags(pool, backend_id, model_name, defaults)
            .await
            .map_err(api_error_to_sqlx)?;
    }

    Ok(())
}

pub async fn ensure_tool_records(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let now = unix_timestamp();
    let defaults = default_tool_tag_ids(pool).await?;
    for tool_id in tool_ids() {
        let result = sqlx::query(
            r#"
            INSERT INTO tool_permission_state (tool_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(tool_id) DO NOTHING
            "#,
        )
        .bind(tool_id)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await?;

        if result.rows_affected() > 0 {
            replace_tool_tags(pool, tool_id, defaults.clone())
                .await
                .map_err(api_error_to_sqlx)?;
        }
    }

    Ok(())
}

pub async fn model_tags_by_backend(
    pool: &SqlitePool,
    backend_id: &str,
) -> Result<HashMap<String, Vec<String>>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT model_name, tag_id
        FROM model_permission_tags
        WHERE backend_id = ?
        ORDER BY model_name ASC, tag_id ASC
        "#,
    )
    .bind(backend_id)
    .fetch_all(pool)
    .await?;

    let mut tags = HashMap::<String, Vec<String>>::new();
    for row in rows {
        tags.entry(row.try_get("model_name")?)
            .or_default()
            .push(row.try_get("tag_id")?);
    }

    Ok(tags)
}

pub async fn model_tags(
    pool: &SqlitePool,
    backend_id: &str,
    model_name: &str,
) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT tag_id
        FROM model_permission_tags
        WHERE backend_id = ?
          AND model_name = ?
        ORDER BY tag_id ASC
        "#,
    )
    .bind(backend_id)
    .bind(model_name)
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(|row| row.try_get("tag_id")).collect()
}

pub async fn replace_model_tags(
    pool: &SqlitePool,
    backend_id: &str,
    model_name: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ApiError> {
    let tags = normalize_tag_ids(pool, values).await?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        DELETE FROM model_permission_tags
        WHERE backend_id = ?
          AND model_name = ?
        "#,
    )
    .bind(backend_id)
    .bind(model_name)
    .execute(&mut *tx)
    .await?;

    for tag in &tags {
        sqlx::query(
            r#"
            INSERT INTO model_permission_tags (backend_id, model_name, tag_id, created_at)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(backend_id)
        .bind(model_name)
        .bind(tag)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(tags)
}

pub async fn tool_tags_by_tool(
    pool: &SqlitePool,
) -> Result<HashMap<String, Vec<String>>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT tool_id, tag_id
        FROM tool_permission_tags
        ORDER BY tool_id ASC, tag_id ASC
        "#,
    )
    .fetch_all(pool)
    .await?;

    let mut tags = HashMap::<String, Vec<String>>::new();
    for row in rows {
        tags.entry(row.try_get("tool_id")?)
            .or_default()
            .push(row.try_get("tag_id")?);
    }

    Ok(tags)
}

pub async fn replace_tool_tags(
    pool: &SqlitePool,
    tool_id: &str,
    values: Vec<String>,
) -> Result<Vec<String>, ApiError> {
    if !tool_ids().contains(&tool_id) {
        return Err(ApiError::bad_request("invalid_tool", "Unknown tool"));
    }
    let tags = normalize_tag_ids(pool, values).await?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO tool_permission_state (tool_id, created_at, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(tool_id) DO UPDATE SET updated_at = excluded.updated_at
        "#,
    )
    .bind(tool_id)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    sqlx::query("DELETE FROM tool_permission_tags WHERE tool_id = ?")
        .bind(tool_id)
        .execute(&mut *tx)
        .await?;

    for tag in &tags {
        sqlx::query(
            r#"
            INSERT INTO tool_permission_tags (tool_id, tag_id, created_at)
            VALUES (?, ?, ?)
            "#,
        )
        .bind(tool_id)
        .bind(tag)
        .bind(now)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(tags)
}

async fn default_tag_ids(pool: &SqlitePool, column: &str) -> Result<Vec<String>, sqlx::Error> {
    let value: String =
        sqlx::query_scalar(&format!("SELECT {column} FROM app_settings WHERE id = 1"))
            .fetch_one(pool)
            .await?;
    Ok(serde_json::from_str(&value).unwrap_or_else(|_| vec![TAG_EVERYONE.to_string()]))
}

async fn update_default_tags(
    pool: &SqlitePool,
    column: &str,
    tags: &[String],
    apply_to_existing: bool,
) -> Result<(), ApiError> {
    let json = serde_json::to_string(tags)
        .map_err(|_| ApiError::internal("Failed to serialize permission tags"))?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(&format!(
        "UPDATE app_settings SET {column} = ?, updated_at = ? WHERE id = 1"
    ))
    .bind(json)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    if apply_to_existing {
        sqlx::query("DELETE FROM model_permission_tags")
            .execute(&mut *tx)
            .await?;
        let rows = sqlx::query("SELECT backend_id, model_name FROM model_availability")
            .fetch_all(&mut *tx)
            .await?;
        for row in rows {
            let backend_id: String = row.try_get("backend_id")?;
            let model_name: String = row.try_get("model_name")?;
            for tag in tags {
                sqlx::query(
                    r#"
                    INSERT INTO model_permission_tags (backend_id, model_name, tag_id, created_at)
                    VALUES (?, ?, ?, ?)
                    "#,
                )
                .bind(&backend_id)
                .bind(&model_name)
                .bind(tag)
                .bind(now)
                .execute(&mut *tx)
                .await?;
            }
        }
    }

    tx.commit().await?;
    Ok(())
}

async fn normalize_tag_id(pool: &SqlitePool, value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_permission_tag",
            "Permission tag cannot be empty",
        ));
    }

    if matches!(value, "everyone" | TAG_EVERYONE) {
        return Ok(TAG_EVERYONE.to_string());
    }
    if matches!(value, "admin" | TAG_ADMIN) {
        return Ok(TAG_ADMIN.to_string());
    }
    if let Some(username) = value.strip_prefix('@') {
        let user_id: Option<String> = sqlx::query_scalar("SELECT id FROM users WHERE username = ?")
            .bind(username)
            .fetch_optional(pool)
            .await?;
        return user_id
            .map(|id| user_tag_id(&id))
            .ok_or_else(|| ApiError::bad_request("unknown_user_tag", "User tag was not found"));
    }
    if let Some(user_id) = value.strip_prefix("user:") {
        ensure_user_exists(pool, user_id).await?;
        return Ok(user_tag_id(user_id));
    }
    if let Some(group) = value.strip_prefix("group:") {
        return Ok(format!("group:{}", normalize_group_name(group)?));
    }

    Ok(format!("group:{}", normalize_group_name(value)?))
}

fn normalize_group_name(value: &str) -> Result<String, ApiError> {
    let mut normalized = String::new();
    let mut previous_dash = false;

    for character in value.trim().chars().flat_map(char::to_lowercase) {
        if character.is_ascii_alphanumeric() || character == '_' {
            normalized.push(character);
            previous_dash = false;
        } else if (character.is_whitespace() || character == '-') && !previous_dash {
            normalized.push('-');
            previous_dash = true;
        }
    }

    let normalized = normalized.trim_matches('-').to_string();
    if normalized.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_permission_tag",
            "Permission tag must include letters or numbers",
        ));
    }

    Ok(normalized)
}

async fn ensure_user_exists(pool: &SqlitePool, user_id: &str) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM users WHERE id = ?)")
        .bind(user_id)
        .fetch_one(pool)
        .await?;
    if exists == 0 {
        return Err(ApiError::not_found("user_not_found", "User not found"));
    }
    Ok(())
}

async fn user_label_map(pool: &SqlitePool) -> Result<HashMap<String, String>, sqlx::Error> {
    let rows = sqlx::query("SELECT id, username FROM users")
        .fetch_all(pool)
        .await?;
    let mut labels = HashMap::new();
    for row in rows {
        let id: String = row.try_get("id")?;
        let username: String = row.try_get("username")?;
        labels.insert(user_tag_id(&id), format!("@{username}"));
    }
    Ok(labels)
}

fn tag_response(tag_id: &str, user_labels: &HashMap<String, String>) -> PermissionTagResponse {
    if tag_id == TAG_EVERYONE {
        return PermissionTagResponse {
            id: tag_id.to_string(),
            label: "everyone".to_string(),
            kind: "system".to_string(),
        };
    }
    if tag_id == TAG_ADMIN {
        return PermissionTagResponse {
            id: tag_id.to_string(),
            label: "admin".to_string(),
            kind: "system".to_string(),
        };
    }
    if let Some(label) = user_labels.get(tag_id) {
        return PermissionTagResponse {
            id: tag_id.to_string(),
            label: label.clone(),
            kind: "user".to_string(),
        };
    }
    PermissionTagResponse {
        id: tag_id.to_string(),
        label: tag_id.strip_prefix("group:").unwrap_or(tag_id).to_string(),
        kind: "group".to_string(),
    }
}

fn tag_sort_key(tag: &PermissionTagResponse) -> (u8, String) {
    let rank = match tag.kind.as_str() {
        "system" => 0,
        "user" => 1,
        _ => 2,
    };
    (rank, tag.label.clone())
}

fn api_error_to_sqlx(error: ApiError) -> sqlx::Error {
    sqlx::Error::Protocol(format!("{error:?}"))
}
