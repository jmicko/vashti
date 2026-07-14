use std::collections::HashSet;

use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    context_blocks::{
        handlers::{
            CreateContextBlockRequest, CreateContextCategoryRequest, UpdateContextBlockRequest,
            UpdateContextCategoryRequest,
        },
        models::{
            ContextBlockResponse, ContextBlockSelection, ContextBlockVersionResponse,
            ContextCategoryResponse,
        },
    },
    error::ApiError,
};

pub const MAX_CONTEXT_BLOCKS_PER_CHAT: usize = 32;
pub const MAX_CONTEXT_BLOCK_CONTENT_LEN: usize = 60_000;
pub const MAX_COMPILED_SYSTEM_PROMPT_LEN: usize = 240_000;
const MAX_CATEGORY_NAME_LEN: usize = 80;
const MAX_BLOCK_NAME_LEN: usize = 120;

pub async fn list_library(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<(Vec<ContextCategoryResponse>, Vec<ContextBlockResponse>), ApiError> {
    let category_rows = sqlx::query(
        r#"
        SELECT id, name, selection_mode, sort_order, created_at, updated_at
        FROM context_categories
        WHERE user_id = ?
        ORDER BY sort_order ASC, name COLLATE NOCASE ASC, created_at ASC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let block_rows = sqlx::query(
        r#"
        SELECT b.id,
               b.category_id,
               b.sort_order,
               b.created_at,
               b.updated_at,
               v.id AS version_id,
               v.block_id AS version_block_id,
               v.version_number,
               v.name,
               v.content,
               v.created_at AS version_created_at
        FROM context_blocks b
        JOIN context_block_versions v ON v.id = b.current_version_id
        WHERE b.user_id = ?
          AND b.deleted_at IS NULL
        ORDER BY b.sort_order ASC, v.name COLLATE NOCASE ASC, b.created_at ASC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let categories = category_rows
        .into_iter()
        .map(row_to_category)
        .collect::<Result<Vec<_>, _>>()?;
    let blocks = block_rows
        .into_iter()
        .map(row_to_block)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((categories, blocks))
}

pub async fn create_category(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreateContextCategoryRequest,
) -> Result<ContextCategoryResponse, ApiError> {
    let name = validate_category_name(&payload.name)?;
    let selection_mode =
        validate_selection_mode(payload.selection_mode.as_deref().unwrap_or("single"))?;
    ensure_category_name_available(pool, user_id, &name, None).await?;
    let id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    sqlx::query(
        r#"
        INSERT INTO context_categories (
            id, user_id, name, selection_mode, sort_order, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 0, ?, ?)
        "#,
    )
    .bind(&id)
    .bind(user_id)
    .bind(name)
    .bind(selection_mode)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;
    get_category(pool, user_id, &id).await
}

pub async fn update_category(
    pool: &SqlitePool,
    user_id: &str,
    category_id: &str,
    payload: UpdateContextCategoryRequest,
) -> Result<ContextCategoryResponse, ApiError> {
    let current = get_category(pool, user_id, category_id).await?;
    let name = match payload.name {
        Some(name) => validate_category_name(&name)?,
        None => current.name,
    };
    ensure_category_name_available(pool, user_id, &name, Some(category_id)).await?;
    let selection_mode = match payload.selection_mode {
        Some(mode) => validate_selection_mode(&mode)?,
        None => current.selection_mode,
    };
    let sort_order = payload.sort_order.unwrap_or(current.sort_order);
    sqlx::query(
        r#"
        UPDATE context_categories
        SET name = ?, selection_mode = ?, sort_order = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
        "#,
    )
    .bind(name)
    .bind(selection_mode)
    .bind(sort_order)
    .bind(unix_timestamp())
    .bind(category_id)
    .bind(user_id)
    .execute(pool)
    .await?;
    get_category(pool, user_id, category_id).await
}

pub async fn delete_category(
    pool: &SqlitePool,
    user_id: &str,
    category_id: &str,
) -> Result<(), ApiError> {
    let result = sqlx::query("DELETE FROM context_categories WHERE id = ? AND user_id = ?")
        .bind(category_id)
        .bind(user_id)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(ApiError::not_found(
            "context_category_not_found",
            "Context category not found",
        ));
    }
    Ok(())
}

pub async fn create_block(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreateContextBlockRequest,
) -> Result<ContextBlockResponse, ApiError> {
    let category_id = normalize_optional(payload.category_id);
    ensure_category_owned(pool, user_id, category_id.as_deref()).await?;
    let name = validate_block_name(&payload.name)?;
    let content = validate_block_content(&payload.content)?;
    let block_id = Uuid::new_v4().to_string();
    let version_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"
        INSERT INTO context_blocks (
            id, user_id, category_id, current_version_id, sort_order,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, 0, ?, ?)
        "#,
    )
    .bind(&block_id)
    .bind(user_id)
    .bind(category_id)
    .bind(&version_id)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    insert_version(&mut tx, &version_id, &block_id, 1, &name, &content, now).await?;
    tx.commit().await?;
    get_block(pool, user_id, &block_id, false).await
}

pub async fn update_block(
    pool: &SqlitePool,
    user_id: &str,
    block_id: &str,
    payload: UpdateContextBlockRequest,
) -> Result<ContextBlockResponse, ApiError> {
    let current = get_block(pool, user_id, block_id, false).await?;
    let category_id = match payload.category_id {
        Some(value) => normalize_optional(value),
        None => current.category_id.clone(),
    };
    ensure_category_owned(pool, user_id, category_id.as_deref()).await?;
    let name = match payload.name {
        Some(name) => validate_block_name(&name)?,
        None => current.current_version.name.clone(),
    };
    let content = match payload.content {
        Some(content) => validate_block_content(&content)?,
        None => current.current_version.content.clone(),
    };
    let sort_order = payload.sort_order.unwrap_or(current.sort_order);
    let content_changed =
        name != current.current_version.name || content != current.current_version.content;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    let next_version_id = if content_changed {
        let version_id = Uuid::new_v4().to_string();
        insert_version(
            &mut tx,
            &version_id,
            block_id,
            current.current_version.version_number + 1,
            &name,
            &content,
            now,
        )
        .await?;
        version_id
    } else {
        current.current_version.id
    };
    sqlx::query(
        r#"
        UPDATE context_blocks
        SET category_id = ?, current_version_id = ?, sort_order = ?, updated_at = ?
        WHERE id = ? AND user_id = ? AND deleted_at IS NULL
        "#,
    )
    .bind(category_id)
    .bind(next_version_id)
    .bind(sort_order)
    .bind(now)
    .bind(block_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    get_block(pool, user_id, block_id, false).await
}

pub async fn delete_block(
    pool: &SqlitePool,
    user_id: &str,
    block_id: &str,
) -> Result<(), ApiError> {
    let result = sqlx::query(
        "UPDATE context_blocks SET deleted_at = ?, updated_at = ? WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
    )
    .bind(unix_timestamp())
    .bind(unix_timestamp())
    .bind(block_id)
    .bind(user_id)
    .execute(pool)
    .await?;
    if result.rows_affected() == 0 {
        return Err(ApiError::not_found(
            "context_block_not_found",
            "Context block not found",
        ));
    }
    Ok(())
}

pub async fn list_versions(
    pool: &SqlitePool,
    user_id: &str,
    block_id: &str,
) -> Result<Vec<ContextBlockVersionResponse>, ApiError> {
    ensure_block_owned(pool, user_id, block_id).await?;
    let rows = sqlx::query(
        r#"
        SELECT id, block_id, version_number, name, content, created_at
        FROM context_block_versions
        WHERE block_id = ?
        ORDER BY version_number DESC
        "#,
    )
    .bind(block_id)
    .fetch_all(pool)
    .await?;
    rows.into_iter()
        .map(row_to_version)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::from)
}

pub async fn resolve_selections_for_user(
    pool: &SqlitePool,
    user_id: &str,
    version_ids: &[String],
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    resolve_selections(pool, user_id, version_ids, &HashSet::new()).await
}

pub async fn resolve_selections_for_chat_update(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    version_ids: &[String],
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    let existing_rows =
        sqlx::query("SELECT block_version_id FROM chat_context_blocks WHERE chat_id = ?")
            .bind(chat_id)
            .fetch_all(pool)
            .await?;
    let existing_version_ids = existing_rows
        .into_iter()
        .map(|row| row.try_get::<String, _>("block_version_id"))
        .collect::<Result<HashSet<_>, _>>()?;
    resolve_selections(pool, user_id, version_ids, &existing_version_ids).await
}

async fn resolve_selections(
    pool: &SqlitePool,
    user_id: &str,
    version_ids: &[String],
    allowed_deleted_version_ids: &HashSet<String>,
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    if version_ids.len() > MAX_CONTEXT_BLOCKS_PER_CHAT {
        return Err(ApiError::bad_request(
            "too_many_context_blocks",
            format!("Select at most {MAX_CONTEXT_BLOCKS_PER_CHAT} context blocks"),
        ));
    }
    let mut seen = HashSet::new();
    let mut selections = Vec::with_capacity(version_ids.len());
    for (position, version_id) in version_ids.iter().enumerate() {
        if !seen.insert(version_id.as_str()) {
            return Err(ApiError::bad_request(
                "duplicate_context_block",
                "A context block version can be selected only once",
            ));
        }
        let Some(row) = sqlx::query(
            r#"
            SELECT b.id AS block_id,
                   v.id AS block_version_id,
                   b.category_id,
                   c.name AS category_name,
                   c.selection_mode AS category_selection_mode,
                   v.version_number,
                   v.name,
                   v.content,
                   b.deleted_at
            FROM context_block_versions v
            JOIN context_blocks b ON b.id = v.block_id
            LEFT JOIN context_categories c ON c.id = b.category_id
            WHERE v.id = ? AND b.user_id = ?
            "#,
        )
        .bind(version_id)
        .bind(user_id)
        .fetch_optional(pool)
        .await?
        else {
            return Err(ApiError::bad_request(
                "invalid_context_block",
                "A selected context block is unavailable",
            ));
        };
        if row.try_get::<Option<i64>, _>("deleted_at")?.is_some()
            && !allowed_deleted_version_ids.contains(version_id)
        {
            return Err(ApiError::bad_request(
                "invalid_context_block",
                "A selected context block is unavailable",
            ));
        }
        selections.push(selection_from_row(row, position as i64)?);
    }
    validate_single_category_selections(&selections)?;
    Ok(selections)
}

pub async fn load_chat_selections(
    pool: &SqlitePool,
    chat_id: &str,
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    load_selections(pool, "chat_context_blocks", "chat_id", chat_id).await
}

pub async fn load_message_selections(
    pool: &SqlitePool,
    message_id: &str,
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    load_selections(
        pool,
        "chat_message_context_blocks",
        "message_id",
        message_id,
    )
    .await
}

async fn load_selections(
    pool: &SqlitePool,
    table: &str,
    owner_column: &str,
    owner_id: &str,
) -> Result<Vec<ContextBlockSelection>, ApiError> {
    let sql = format!(
        r#"
        SELECT b.id AS block_id,
               v.id AS block_version_id,
               b.category_id,
               c.name AS category_name,
               c.selection_mode AS category_selection_mode,
               v.version_number,
               v.name,
               v.content,
               selected.position
        FROM {table} selected
        JOIN context_block_versions v ON v.id = selected.block_version_id
        JOIN context_blocks b ON b.id = v.block_id
        LEFT JOIN context_categories c ON c.id = b.category_id
        WHERE selected.{owner_column} = ?
        ORDER BY selected.position ASC
        "#,
    );
    let rows = sqlx::query(&sql).bind(owner_id).fetch_all(pool).await?;
    rows.into_iter()
        .map(|row| {
            let position = row.try_get("position")?;
            selection_from_row(row, position)
        })
        .collect()
}

pub async fn replace_chat_selections(
    tx: &mut Transaction<'_, Sqlite>,
    chat_id: &str,
    selections: &[ContextBlockSelection],
) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM chat_context_blocks WHERE chat_id = ?")
        .bind(chat_id)
        .execute(&mut **tx)
        .await?;
    for selection in selections {
        sqlx::query(
            "INSERT INTO chat_context_blocks (chat_id, block_version_id, position) VALUES (?, ?, ?)",
        )
        .bind(chat_id)
        .bind(&selection.block_version_id)
        .bind(selection.position)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

pub async fn snapshot_message_selections(
    tx: &mut Transaction<'_, Sqlite>,
    message_id: &str,
    selections: &[ContextBlockSelection],
) -> Result<(), ApiError> {
    for selection in selections {
        sqlx::query(
            "INSERT INTO chat_message_context_blocks (message_id, block_version_id, position) VALUES (?, ?, ?)",
        )
        .bind(message_id)
        .bind(&selection.block_version_id)
        .bind(selection.position)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

pub fn compile_system_prompt(
    base_prompt: Option<&str>,
    selections: &[ContextBlockSelection],
) -> Result<Option<String>, ApiError> {
    let mut parts = Vec::with_capacity(selections.len() + usize::from(base_prompt.is_some()));
    if let Some(base) = base_prompt.map(str::trim).filter(|value| !value.is_empty()) {
        parts.push(base.to_string());
    }
    for selection in selections {
        parts.push(format!(
            "[Context: {}]\n{}",
            selection.name.trim(),
            selection.content.trim()
        ));
    }
    if parts.is_empty() {
        return Ok(None);
    }
    let compiled = parts.join("\n\n");
    if compiled.len() > MAX_COMPILED_SYSTEM_PROMPT_LEN {
        return Err(ApiError::bad_request(
            "context_prompt_too_large",
            format!("The combined system prompt exceeds {MAX_COMPILED_SYSTEM_PROMPT_LEN} bytes"),
        ));
    }
    Ok(Some(compiled))
}

async fn get_category(
    pool: &SqlitePool,
    user_id: &str,
    category_id: &str,
) -> Result<ContextCategoryResponse, ApiError> {
    let row = sqlx::query(
        "SELECT id, name, selection_mode, sort_order, created_at, updated_at FROM context_categories WHERE id = ? AND user_id = ?",
    )
    .bind(category_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| {
        ApiError::not_found("context_category_not_found", "Context category not found")
    })?;
    row_to_category(row).map_err(ApiError::from)
}

async fn get_block(
    pool: &SqlitePool,
    user_id: &str,
    block_id: &str,
    include_deleted: bool,
) -> Result<ContextBlockResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT b.id,
               b.category_id,
               b.sort_order,
               b.created_at,
               b.updated_at,
               v.id AS version_id,
               v.block_id AS version_block_id,
               v.version_number,
               v.name,
               v.content,
               v.created_at AS version_created_at,
               b.deleted_at
        FROM context_blocks b
        JOIN context_block_versions v ON v.id = b.current_version_id
        WHERE b.id = ? AND b.user_id = ?
        "#,
    )
    .bind(block_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("context_block_not_found", "Context block not found"))?;
    if !include_deleted && row.try_get::<Option<i64>, _>("deleted_at")?.is_some() {
        return Err(ApiError::not_found(
            "context_block_not_found",
            "Context block not found",
        ));
    }
    row_to_block(row).map_err(ApiError::from)
}

async fn ensure_category_name_available(
    pool: &SqlitePool,
    user_id: &str,
    name: &str,
    except_id: Option<&str>,
) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM context_categories WHERE user_id = ? AND name = ? COLLATE NOCASE AND (? IS NULL OR id != ?))",
    )
    .bind(user_id)
    .bind(name)
    .bind(except_id)
    .bind(except_id)
    .fetch_one(pool)
    .await?;
    if exists != 0 {
        return Err(ApiError::conflict(
            "context_category_exists",
            "A context category with that name already exists",
        ));
    }
    Ok(())
}

async fn ensure_category_owned(
    pool: &SqlitePool,
    user_id: &str,
    category_id: Option<&str>,
) -> Result<(), ApiError> {
    let Some(category_id) = category_id else {
        return Ok(());
    };
    let exists: i64 = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM context_categories WHERE id = ? AND user_id = ?)",
    )
    .bind(category_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    if exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_context_category",
            "Context category not found",
        ));
    }
    Ok(())
}

async fn ensure_block_owned(
    pool: &SqlitePool,
    user_id: &str,
    block_id: &str,
) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM context_blocks WHERE id = ? AND user_id = ?)",
    )
    .bind(block_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    if exists == 0 {
        return Err(ApiError::not_found(
            "context_block_not_found",
            "Context block not found",
        ));
    }
    Ok(())
}

async fn insert_version(
    tx: &mut Transaction<'_, Sqlite>,
    version_id: &str,
    block_id: &str,
    version_number: i64,
    name: &str,
    content: &str,
    now: i64,
) -> Result<(), ApiError> {
    sqlx::query(
        r#"
        INSERT INTO context_block_versions (
            id, block_id, version_number, name, content, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(version_id)
    .bind(block_id)
    .bind(version_number)
    .bind(name)
    .bind(content)
    .bind(now)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

fn row_to_category(row: sqlx::sqlite::SqliteRow) -> Result<ContextCategoryResponse, sqlx::Error> {
    Ok(ContextCategoryResponse {
        id: row.try_get("id")?,
        name: row.try_get("name")?,
        selection_mode: row.try_get("selection_mode")?,
        sort_order: row.try_get("sort_order")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_block(row: sqlx::sqlite::SqliteRow) -> Result<ContextBlockResponse, sqlx::Error> {
    Ok(ContextBlockResponse {
        id: row.try_get("id")?,
        category_id: row.try_get("category_id")?,
        sort_order: row.try_get("sort_order")?,
        current_version: ContextBlockVersionResponse {
            id: row.try_get("version_id")?,
            block_id: row.try_get("version_block_id")?,
            version_number: row.try_get("version_number")?,
            name: row.try_get("name")?,
            content: row.try_get("content")?,
            created_at: row.try_get("version_created_at")?,
        },
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_version(
    row: sqlx::sqlite::SqliteRow,
) -> Result<ContextBlockVersionResponse, sqlx::Error> {
    Ok(ContextBlockVersionResponse {
        id: row.try_get("id")?,
        block_id: row.try_get("block_id")?,
        version_number: row.try_get("version_number")?,
        name: row.try_get("name")?,
        content: row.try_get("content")?,
        created_at: row.try_get("created_at")?,
    })
}

fn selection_from_row(
    row: sqlx::sqlite::SqliteRow,
    position: i64,
) -> Result<ContextBlockSelection, ApiError> {
    Ok(ContextBlockSelection {
        block_id: row.try_get("block_id")?,
        block_version_id: row.try_get("block_version_id")?,
        category_id: row.try_get("category_id")?,
        category_name: row.try_get("category_name")?,
        category_selection_mode: row.try_get("category_selection_mode")?,
        version_number: row.try_get("version_number")?,
        name: row.try_get("name")?,
        content: row.try_get("content")?,
        position,
    })
}

fn validate_single_category_selections(
    selections: &[ContextBlockSelection],
) -> Result<(), ApiError> {
    let mut single_categories = HashSet::new();
    for selection in selections {
        if selection.category_selection_mode.as_deref() != Some("single") {
            continue;
        }
        let Some(category_id) = selection.category_id.as_deref() else {
            continue;
        };
        if !single_categories.insert(category_id) {
            return Err(ApiError::bad_request(
                "context_category_single_selection",
                "Only one block may be selected from a single-selection category",
            ));
        }
    }
    Ok(())
}

fn validate_category_name(value: &str) -> Result<String, ApiError> {
    validate_text(value, MAX_CATEGORY_NAME_LEN, "category name")
}

fn validate_block_name(value: &str) -> Result<String, ApiError> {
    validate_text(value, MAX_BLOCK_NAME_LEN, "block name")
}

fn validate_text(value: &str, max_len: usize, field: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_context_block",
            format!("A {field} is required"),
        ));
    }
    if value.chars().count() > max_len {
        return Err(ApiError::bad_request(
            "invalid_context_block",
            format!("The {field} must be {max_len} characters or fewer"),
        ));
    }
    Ok(value.to_string())
}

fn validate_block_content(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_context_block",
            "Context block content is required",
        ));
    }
    if value.len() > MAX_CONTEXT_BLOCK_CONTENT_LEN {
        return Err(ApiError::bad_request(
            "invalid_context_block",
            format!("Context block content must be {MAX_CONTEXT_BLOCK_CONTENT_LEN} bytes or fewer"),
        ));
    }
    Ok(value.to_string())
}

fn validate_selection_mode(value: &str) -> Result<String, ApiError> {
    match value.trim() {
        "single" => Ok("single".to_string()),
        "multiple" => Ok("multiple".to_string()),
        _ => Err(ApiError::bad_request(
            "invalid_context_category",
            "Selection mode must be single or multiple",
        )),
    }
}

fn normalize_optional(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use sqlx::{
        SqlitePool,
        sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    };

    use super::*;
    use crate::{auth::service::register_user, startup};

    async fn test_pool() -> SqlitePool {
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
        startup::bootstrap::ensure_app_settings(&pool)
            .await
            .expect("ensure app settings");
        pool
    }

    async fn create_test_user(pool: &SqlitePool) -> String {
        register_user(
            pool,
            "context-admin".to_string(),
            None,
            "secret-pass".to_string(),
        )
        .await
        .expect("register test user")
        .user
        .id
    }

    async fn create_test_chat(pool: &SqlitePool, user_id: &str, chat_id: &str) {
        let now = unix_timestamp();
        sqlx::query(
            "INSERT INTO ollama_backends (id, name, base_url, created_at, updated_at) VALUES ('context-backend', 'Context Backend', 'http://127.0.0.1:11434', ?, ?)",
        )
        .bind(now)
        .bind(now)
        .execute(pool)
        .await
        .expect("create test backend");
        sqlx::query(
            "INSERT INTO chats (id, user_id, default_backend_id, default_model_name, title, created_at, updated_at, last_message_at) VALUES (?, ?, 'context-backend', 'test-model', 'Context test', ?, ?, ?)",
        )
        .bind(chat_id)
        .bind(user_id)
        .bind(now)
        .bind(now)
        .bind(now)
        .execute(pool)
        .await
        .expect("create test chat");
    }

    fn selection(
        category_id: Option<&str>,
        mode: Option<&str>,
        name: &str,
    ) -> ContextBlockSelection {
        ContextBlockSelection {
            block_id: format!("block-{name}"),
            block_version_id: format!("version-{name}"),
            category_id: category_id.map(str::to_string),
            category_name: None,
            category_selection_mode: mode.map(str::to_string),
            version_number: 1,
            name: name.to_string(),
            content: format!("content for {name}"),
            position: 0,
        }
    }

    #[test]
    fn compiled_prompt_preserves_base_and_selection_order() {
        let result = compile_system_prompt(
            Some("Base instructions"),
            &[
                selection(Some("mood"), Some("single"), "Calm"),
                selection(Some("project"), Some("multiple"), "Vashti"),
            ],
        )
        .expect("prompt compiles")
        .expect("prompt exists");
        assert_eq!(
            result,
            "Base instructions\n\n[Context: Calm]\ncontent for Calm\n\n[Context: Vashti]\ncontent for Vashti"
        );
    }

    #[test]
    fn single_category_rejects_multiple_selections() {
        let result = validate_single_category_selections(&[
            selection(Some("mood"), Some("single"), "Calm"),
            selection(Some("mood"), Some("single"), "Playful"),
        ]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn deleted_block_stays_pinned_to_an_existing_chat() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool).await;
        let chat_id = "context-chat";
        create_test_chat(&pool, &user_id, chat_id).await;

        let block = create_block(
            &pool,
            &user_id,
            CreateContextBlockRequest {
                category_id: None,
                name: "Project rules".to_string(),
                content: "Never use unwrap in Rust.".to_string(),
            },
        )
        .await
        .expect("create context block");
        let selections =
            resolve_selections_for_user(&pool, &user_id, &[block.current_version.id.clone()])
                .await
                .expect("resolve selection");
        let mut tx = pool.begin().await.expect("begin selection transaction");
        replace_chat_selections(&mut tx, chat_id, &selections)
            .await
            .expect("pin chat selection");
        tx.commit().await.expect("commit chat selection");

        delete_block(&pool, &user_id, &block.id)
            .await
            .expect("soft delete block");

        let pinned = load_chat_selections(&pool, chat_id)
            .await
            .expect("load pinned selection");
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].content, "Never use unwrap in Rust.");
        resolve_selections_for_chat_update(
            &pool,
            &user_id,
            chat_id,
            &[block.current_version.id.clone()],
        )
        .await
        .expect("existing chat can retain deleted selection");
        assert!(
            resolve_selections_for_user(&pool, &user_id, &[block.current_version.id.clone()])
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn deleting_user_cascades_context_library_and_snapshots() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool).await;
        let chat_id = "context-delete-chat";
        create_test_chat(&pool, &user_id, chat_id).await;
        let block = create_block(
            &pool,
            &user_id,
            CreateContextBlockRequest {
                category_id: None,
                name: "Temporary context".to_string(),
                content: "Removed with the account.".to_string(),
            },
        )
        .await
        .expect("create context block");
        let selections =
            resolve_selections_for_user(&pool, &user_id, &[block.current_version.id.clone()])
                .await
                .expect("resolve selection");
        let now = unix_timestamp();
        sqlx::query(
            "INSERT INTO chat_messages (id, chat_id, role, status, created_at, updated_at) VALUES ('context-message', ?, 'assistant', 'complete', ?, ?)",
        )
        .bind(chat_id)
        .bind(now)
        .bind(now)
        .execute(&pool)
        .await
        .expect("create test message");
        let mut tx = pool.begin().await.expect("begin snapshot transaction");
        replace_chat_selections(&mut tx, chat_id, &selections)
            .await
            .expect("pin chat selection");
        snapshot_message_selections(&mut tx, "context-message", &selections)
            .await
            .expect("snapshot message selection");
        tx.commit().await.expect("commit snapshots");

        sqlx::query("DELETE FROM users WHERE id = ?")
            .bind(&user_id)
            .execute(&pool)
            .await
            .expect("delete user with context data");

        for table in [
            "context_blocks",
            "context_block_versions",
            "chat_context_blocks",
            "chat_message_context_blocks",
        ] {
            let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
                .fetch_one(&pool)
                .await
                .expect("count context rows");
            assert_eq!(count, 0, "{table} should be empty");
        }
    }
}
