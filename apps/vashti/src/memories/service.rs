use std::collections::{HashMap, HashSet};

use sqlx::{Row, Sqlite, SqlitePool, Transaction, sqlite::SqliteRow};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    error::ApiError,
    memories::models::{
        CreateMemoryRequest, MemoryModelScope, MemoryResponse, MemorySettingsResponse,
        MemorySummaryResponse, MemoryVersionResponse, UpdateMemoryRequest,
        UpdateMemorySettingsRequest,
    },
};

pub const MAX_MEMORY_CONTENT_BYTES: usize = 32 * 1024;
pub const MAX_MEMORY_MODEL_SCOPES: usize = 64;
const MAX_MODEL_KEY_CHARS: usize = 512;
const DEFAULT_LIST_LIMIT: i64 = 50;
const MAX_LIST_LIMIT: i64 = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryListStatus {
    Active,
    Forgotten,
    All,
}

#[derive(Clone, Debug)]
pub struct ListMemoriesOptions {
    pub query: Option<String>,
    pub status: MemoryListStatus,
    pub limit: i64,
    pub offset: i64,
}

impl Default for ListMemoriesOptions {
    fn default() -> Self {
        Self {
            query: None,
            status: MemoryListStatus::Active,
            limit: DEFAULT_LIST_LIMIT,
            offset: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ListMemoriesResult {
    pub memories: Vec<MemorySummaryResponse>,
    pub total: i64,
}

#[derive(Clone, Debug)]
pub struct MemoryMutationActor {
    pub actor_type: &'static str,
    pub actor_user_id: String,
    pub actor_model_key: Option<String>,
    pub actor_model_name: Option<String>,
    pub source_chat_id: Option<String>,
    pub source_message_id: Option<String>,
    pub source_tool_call_id: Option<String>,
}

impl MemoryMutationActor {
    pub fn human(user_id: &str) -> Self {
        Self {
            actor_type: "human",
            actor_user_id: user_id.to_string(),
            actor_model_key: None,
            actor_model_name: None,
            source_chat_id: None,
            source_message_id: None,
            source_tool_call_id: None,
        }
    }

    pub fn model(
        user_id: &str,
        model_key: &str,
        model_name: &str,
        chat_id: &str,
        message_id: &str,
        tool_call_id: &str,
    ) -> Self {
        Self {
            actor_type: "model",
            actor_user_id: user_id.to_string(),
            actor_model_key: Some(model_key.to_string()),
            actor_model_name: Some(model_name.to_string()),
            source_chat_id: Some(chat_id.to_string()),
            source_message_id: Some(message_id.to_string()),
            source_tool_call_id: Some(tool_call_id.to_string()),
        }
    }

    fn is_model(&self) -> bool {
        self.actor_type == "model"
    }
}

#[derive(Debug)]
struct CurrentMemory {
    current_version_id: String,
    version_number: i64,
    content: String,
    all_models: bool,
    deleted_at: Option<i64>,
}

pub async fn list_memories(
    pool: &SqlitePool,
    user_id: &str,
    mut options: ListMemoriesOptions,
) -> Result<ListMemoriesResult, ApiError> {
    options.limit = options.limit.clamp(1, MAX_LIST_LIMIT);
    options.offset = options.offset.max(0);
    options.query = normalize_search_query(options.query);

    let rows = if let Some(query) = options
        .query
        .as_deref()
        .filter(|_| options.status == MemoryListStatus::Active)
    {
        let fts_query = build_fts_query(query);
        if fts_query.is_empty() {
            list_without_fts(pool, user_id, &options).await?
        } else {
            sqlx::query(
                r#"
                SELECT m.id, m.current_version_id,
                       v.version_number AS current_version_number,
                       substr(v.content, 1, 420) AS excerpt,
                       m.all_models, m.deleted_at, m.created_at, m.updated_at
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.memory_id
                JOIN memory_versions v ON v.id = m.current_version_id
                WHERE memories_fts MATCH ?
                  AND memories_fts.user_id = ?
                  AND m.deleted_at IS NULL
                ORDER BY bm25(memories_fts, 0.0, 0.0, 1.0) ASC, m.updated_at DESC
                LIMIT ? OFFSET ?
                "#,
            )
            .bind(fts_query)
            .bind(user_id)
            .bind(options.limit)
            .bind(options.offset)
            .fetch_all(pool)
            .await?
        }
    } else {
        list_without_fts(pool, user_id, &options).await?
    };
    let total = count_memories(pool, user_id, &options).await?;
    let scopes = load_user_scopes(pool, user_id).await?;
    let memories = rows
        .into_iter()
        .map(|row| row_to_summary(row, &scopes))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ListMemoriesResult { memories, total })
}

pub async fn search_memories_for_model(
    pool: &SqlitePool,
    user_id: &str,
    model_key: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<MemorySummaryResponse>, ApiError> {
    if !get_memory_settings(pool, user_id).await?.allow_model_read {
        return Ok(Vec::new());
    }
    let fts_query = build_fts_query(query.trim());
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let rows = sqlx::query(
        r#"
        SELECT m.id, m.current_version_id,
               v.version_number AS current_version_number,
               substr(v.content, 1, 420) AS excerpt,
               m.all_models, m.deleted_at, m.created_at, m.updated_at
        FROM memories_fts
        JOIN memories m ON m.id = memories_fts.memory_id
        JOIN memory_versions v ON v.id = m.current_version_id
        WHERE memories_fts MATCH ?
          AND memories_fts.user_id = ?
          AND m.deleted_at IS NULL
          AND (
              m.all_models = 1
              OR EXISTS (
                  SELECT 1 FROM memory_model_scopes scopes
                  WHERE scopes.memory_id = m.id AND scopes.model_key = ?
              )
          )
        ORDER BY bm25(memories_fts, 0.0, 0.0, 1.0) ASC, m.updated_at DESC
        LIMIT ?
        "#,
    )
    .bind(fts_query)
    .bind(user_id)
    .bind(model_key)
    .bind(limit.clamp(1, 10))
    .fetch_all(pool)
    .await?;
    let scopes = load_user_scopes(pool, user_id).await?;
    rows.into_iter()
        .map(|row| row_to_summary(row, &scopes))
        .collect()
}

pub async fn get_memory(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
) -> Result<MemoryResponse, ApiError> {
    let row = memory_row(pool, user_id, memory_id, None).await?;
    let scope = load_memory_scope(pool, user_id, memory_id).await?;
    row_to_memory(row, scope)
}

pub async fn get_memory_for_model(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    model_key: &str,
) -> Result<MemoryResponse, ApiError> {
    if !get_memory_settings(pool, user_id).await?.allow_model_read {
        return Err(model_memory_unavailable());
    }
    let row = memory_row(pool, user_id, memory_id, Some(model_key)).await?;
    let scope = load_memory_scope(pool, user_id, memory_id).await?;
    row_to_memory(row, scope)
}

pub async fn create_memory(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreateMemoryRequest,
    actor: &MemoryMutationActor,
) -> Result<MemoryResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    let content = validate_content(&payload.content)?;
    let settings = get_memory_settings(pool, user_id).await?;
    if actor.is_model() && !settings.allow_model_create {
        return Err(ApiError::forbidden(
            "memory_model_create_forbidden",
            "Creating memories is disabled for models",
        ));
    }
    let scope = if actor.is_model() {
        MemoryModelScope {
            all_models: false,
            model_keys: vec![actor.actor_model_key.clone().ok_or_else(|| {
                ApiError::internal("Model memory changes require a model identity")
            })?],
        }
    } else {
        validate_model_scope(payload.model_scope.unwrap_or_default())?
    };
    let memory_id = Uuid::new_v4().to_string();
    let version_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"
        INSERT INTO memories (
            id, user_id, current_version_id, all_models, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&memory_id)
    .bind(user_id)
    .bind(&version_id)
    .bind(scope.all_models)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    insert_version(&mut tx, &version_id, &memory_id, 1, &content, actor, now).await?;
    replace_model_scope(&mut tx, &memory_id, &scope).await?;
    rebuild_fts(&mut tx, user_id, &memory_id, &content, true).await?;
    tx.commit().await?;
    get_memory(pool, user_id, &memory_id).await
}

pub async fn update_memory(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    payload: UpdateMemoryRequest,
    actor: &MemoryMutationActor,
) -> Result<MemoryResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    if actor.is_model() && payload.model_scope.is_some() {
        return Err(ApiError::forbidden(
            "memory_model_scope_forbidden",
            "Models cannot change who can access a memory",
        ));
    }
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, memory_id, payload.expected_version).await?;
    let current = get_current_in_tx(&mut tx, user_id, memory_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "memory_forgotten",
            "Restore this memory before editing it",
        ));
    }
    ensure_model_mutation_allowed(&mut tx, user_id, memory_id, actor, ModelOperation::Edit).await?;
    let content = match payload.content {
        Some(content) => validate_content(&content)?,
        None => current.content.clone(),
    };
    let scope = match payload.model_scope {
        Some(scope) => Some(validate_model_scope(scope)?),
        None => None,
    };
    let now = unix_timestamp();
    let next_version_id = if content != current.content {
        let id = Uuid::new_v4().to_string();
        insert_version(
            &mut tx,
            &id,
            memory_id,
            current.version_number + 1,
            &content,
            actor,
            now,
        )
        .await?;
        id
    } else {
        current.current_version_id
    };
    sqlx::query(
        "UPDATE memories SET current_version_id = ?, all_models = ?, updated_at = ? WHERE id = ? AND user_id = ?",
    )
    .bind(next_version_id)
    .bind(scope.as_ref().map_or(current.all_models, |value| value.all_models))
    .bind(now)
    .bind(memory_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    if let Some(scope) = &scope {
        replace_model_scope(&mut tx, memory_id, scope).await?;
    }
    rebuild_fts(&mut tx, user_id, memory_id, &content, true).await?;
    tx.commit().await?;
    get_memory(pool, user_id, memory_id).await
}

pub async fn forget_memory(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    expected_version: i64,
    actor: &MemoryMutationActor,
) -> Result<MemoryResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, memory_id, expected_version).await?;
    let current = get_current_in_tx(&mut tx, user_id, memory_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "memory_already_forgotten",
            "Memory is already forgotten",
        ));
    }
    ensure_model_mutation_allowed(&mut tx, user_id, memory_id, actor, ModelOperation::Forget)
        .await?;
    let now = unix_timestamp();
    sqlx::query("UPDATE memories SET deleted_at = ?, updated_at = ? WHERE id = ? AND user_id = ?")
        .bind(now)
        .bind(now)
        .bind(memory_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    delete_fts(&mut tx, memory_id).await?;
    tx.commit().await?;
    get_memory(pool, user_id, memory_id).await
}

pub async fn restore_memory(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    expected_version: i64,
) -> Result<MemoryResponse, ApiError> {
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, memory_id, expected_version).await?;
    let current = get_current_in_tx(&mut tx, user_id, memory_id).await?;
    if current.deleted_at.is_none() {
        return Err(ApiError::conflict(
            "memory_not_forgotten",
            "Memory is not forgotten",
        ));
    }
    let now = unix_timestamp();
    sqlx::query(
        "UPDATE memories SET deleted_at = NULL, updated_at = ? WHERE id = ? AND user_id = ?",
    )
    .bind(now)
    .bind(memory_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    rebuild_fts(&mut tx, user_id, memory_id, &current.content, true).await?;
    tx.commit().await?;
    get_memory(pool, user_id, memory_id).await
}

pub async fn permanently_delete_memory(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
) -> Result<(), ApiError> {
    let mut tx = pool.begin().await?;
    let deleted_at: Option<Option<i64>> =
        sqlx::query_scalar("SELECT deleted_at FROM memories WHERE id = ? AND user_id = ?")
            .bind(memory_id)
            .bind(user_id)
            .fetch_optional(&mut *tx)
            .await?;
    match deleted_at {
        None => return Err(memory_not_found()),
        Some(None) => {
            return Err(ApiError::conflict(
                "memory_not_forgotten",
                "Forget this memory before deleting it permanently",
            ));
        }
        Some(Some(_)) => {}
    }
    delete_fts(&mut tx, memory_id).await?;
    sqlx::query("DELETE FROM memories WHERE id = ? AND user_id = ?")
        .bind(memory_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

pub async fn list_versions(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
) -> Result<Vec<MemoryVersionResponse>, ApiError> {
    ensure_owned(pool, user_id, memory_id).await?;
    let rows = sqlx::query(
        r#"
        SELECT id, memory_id, version_number, content, actor_type, actor_user_id,
               actor_model_key, actor_model_name, source_chat_id, source_message_id,
               source_tool_call_id, created_at
        FROM memory_versions
        WHERE memory_id = ?
        ORDER BY version_number DESC
        "#,
    )
    .bind(memory_id)
    .fetch_all(pool)
    .await?;
    rows.into_iter()
        .map(row_to_version)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ApiError::from)
}

pub async fn restore_version(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    version_id: &str,
    expected_version: i64,
) -> Result<MemoryResponse, ApiError> {
    let actor = MemoryMutationActor::human(user_id);
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, memory_id, expected_version).await?;
    let current = get_current_in_tx(&mut tx, user_id, memory_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "memory_forgotten",
            "Restore this memory before restoring its history",
        ));
    }
    let content: String =
        sqlx::query_scalar("SELECT content FROM memory_versions WHERE id = ? AND memory_id = ?")
            .bind(version_id)
            .bind(memory_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| {
                ApiError::not_found("memory_version_not_found", "Memory version not found")
            })?;
    let next_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    insert_version(
        &mut tx,
        &next_id,
        memory_id,
        current.version_number + 1,
        &content,
        &actor,
        now,
    )
    .await?;
    sqlx::query(
        "UPDATE memories SET current_version_id = ?, updated_at = ? WHERE id = ? AND user_id = ?",
    )
    .bind(next_id)
    .bind(now)
    .bind(memory_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    rebuild_fts(&mut tx, user_id, memory_id, &content, true).await?;
    tx.commit().await?;
    get_memory(pool, user_id, memory_id).await
}

pub async fn get_memory_settings(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<MemorySettingsResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT allow_model_read, allow_model_create, allow_model_edit, allow_model_forget
        FROM user_memory_settings WHERE user_id = ?
        "#,
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;
    let Some(row) = row else {
        return Ok(MemorySettingsResponse {
            allow_model_read: false,
            allow_model_create: false,
            allow_model_edit: false,
            allow_model_forget: false,
        });
    };
    Ok(MemorySettingsResponse {
        allow_model_read: row.try_get("allow_model_read")?,
        allow_model_create: row.try_get("allow_model_create")?,
        allow_model_edit: row.try_get("allow_model_edit")?,
        allow_model_forget: row.try_get("allow_model_forget")?,
    })
}

pub async fn update_memory_settings(
    pool: &SqlitePool,
    user_id: &str,
    payload: UpdateMemorySettingsRequest,
) -> Result<MemorySettingsResponse, ApiError> {
    let now = unix_timestamp();
    sqlx::query(
        r#"
        INSERT INTO user_memory_settings (
            user_id, allow_model_read, allow_model_create, allow_model_edit,
            allow_model_forget, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            allow_model_read = excluded.allow_model_read,
            allow_model_create = excluded.allow_model_create,
            allow_model_edit = excluded.allow_model_edit,
            allow_model_forget = excluded.allow_model_forget,
            updated_at = excluded.updated_at
        "#,
    )
    .bind(user_id)
    .bind(payload.allow_model_read)
    .bind(payload.allow_model_create)
    .bind(payload.allow_model_edit)
    .bind(payload.allow_model_forget)
    .bind(now)
    .execute(pool)
    .await?;
    get_memory_settings(pool, user_id).await
}

async fn memory_row(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
    model_key: Option<&str>,
) -> Result<SqliteRow, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT m.id, m.deleted_at, m.created_at, m.updated_at,
               v.id AS version_id, v.memory_id AS version_memory_id,
               v.version_number, v.content, v.actor_type, v.actor_user_id,
               v.actor_model_key, v.actor_model_name, v.source_chat_id,
               v.source_message_id, v.source_tool_call_id,
               v.created_at AS version_created_at
        FROM memories m
        JOIN memory_versions v ON v.id = m.current_version_id
        WHERE m.id = ? AND m.user_id = ?
          AND (? IS NULL OR (
              m.deleted_at IS NULL
              AND (
                  m.all_models = 1
                  OR EXISTS (
                      SELECT 1 FROM memory_model_scopes scopes
                      WHERE scopes.memory_id = m.id AND scopes.model_key = ?
                  )
              )
          ))
        "#,
    )
    .bind(memory_id)
    .bind(user_id)
    .bind(model_key)
    .bind(model_key)
    .fetch_optional(pool)
    .await?;
    row.ok_or_else(|| {
        if model_key.is_some() {
            model_memory_unavailable()
        } else {
            memory_not_found()
        }
    })
}

async fn list_without_fts(
    pool: &SqlitePool,
    user_id: &str,
    options: &ListMemoriesOptions,
) -> Result<Vec<SqliteRow>, ApiError> {
    let status = status_clause(options.status);
    let query = options.query.as_ref().map(|value| format!("%{value}%"));
    let sql = format!(
        r#"
        SELECT m.id, m.current_version_id,
               v.version_number AS current_version_number,
               substr(v.content, 1, 420) AS excerpt,
               m.all_models, m.deleted_at, m.created_at, m.updated_at
        FROM memories m
        JOIN memory_versions v ON v.id = m.current_version_id
        WHERE m.user_id = ? AND {status}
          AND (? IS NULL OR v.content LIKE ? COLLATE NOCASE)
        ORDER BY m.updated_at DESC
        LIMIT ? OFFSET ?
        "#,
    );
    Ok(sqlx::query(&sql)
        .bind(user_id)
        .bind(query.as_deref())
        .bind(query.as_deref())
        .bind(options.limit)
        .bind(options.offset)
        .fetch_all(pool)
        .await?)
}

async fn count_memories(
    pool: &SqlitePool,
    user_id: &str,
    options: &ListMemoriesOptions,
) -> Result<i64, ApiError> {
    let query = options.query.as_ref().map(|value| format!("%{value}%"));
    let sql = format!(
        r#"
        SELECT COUNT(*) FROM memories m
        JOIN memory_versions v ON v.id = m.current_version_id
        WHERE m.user_id = ? AND {}
          AND (? IS NULL OR v.content LIKE ? COLLATE NOCASE)
        "#,
        status_clause(options.status)
    );
    Ok(sqlx::query_scalar(&sql)
        .bind(user_id)
        .bind(query.as_deref())
        .bind(query.as_deref())
        .fetch_one(pool)
        .await?)
}

fn status_clause(status: MemoryListStatus) -> &'static str {
    match status {
        MemoryListStatus::Active => "m.deleted_at IS NULL",
        MemoryListStatus::Forgotten => "m.deleted_at IS NOT NULL",
        MemoryListStatus::All => "1 = 1",
    }
}

async fn acquire_expected_version(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    memory_id: &str,
    expected_version: i64,
) -> Result<(), ApiError> {
    let result = sqlx::query(
        r#"
        UPDATE memories SET updated_at = updated_at
        WHERE id = ? AND user_id = ?
          AND current_version_id IN (
              SELECT id FROM memory_versions
              WHERE memory_id = ? AND version_number = ?
          )
        "#,
    )
    .bind(memory_id)
    .bind(user_id)
    .bind(memory_id)
    .bind(expected_version)
    .execute(&mut **tx)
    .await?;
    if result.rows_affected() != 0 {
        return Ok(());
    }
    let exists: i64 =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM memories WHERE id = ? AND user_id = ?)")
            .bind(memory_id)
            .bind(user_id)
            .fetch_one(&mut **tx)
            .await?;
    if exists == 0 {
        return Err(memory_not_found());
    }
    Err(ApiError::conflict(
        "memory_version_conflict",
        "This memory changed elsewhere. Reload it before saving",
    ))
}

async fn get_current_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    memory_id: &str,
) -> Result<CurrentMemory, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT m.current_version_id, m.all_models, m.deleted_at,
               v.version_number, v.content
        FROM memories m
        JOIN memory_versions v ON v.id = m.current_version_id
        WHERE m.id = ? AND m.user_id = ?
        "#,
    )
    .bind(memory_id)
    .bind(user_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(memory_not_found)?;
    Ok(CurrentMemory {
        current_version_id: row.try_get("current_version_id")?,
        version_number: row.try_get("version_number")?,
        content: row.try_get("content")?,
        all_models: row.try_get("all_models")?,
        deleted_at: row.try_get("deleted_at")?,
    })
}

async fn insert_version(
    tx: &mut Transaction<'_, Sqlite>,
    version_id: &str,
    memory_id: &str,
    version_number: i64,
    content: &str,
    actor: &MemoryMutationActor,
    created_at: i64,
) -> Result<(), ApiError> {
    sqlx::query(
        r#"
        INSERT INTO memory_versions (
            id, memory_id, version_number, content, actor_type, actor_user_id,
            actor_model_key, actor_model_name, source_chat_id, source_message_id,
            source_tool_call_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(version_id)
    .bind(memory_id)
    .bind(version_number)
    .bind(content)
    .bind(actor.actor_type)
    .bind(&actor.actor_user_id)
    .bind(&actor.actor_model_key)
    .bind(&actor.actor_model_name)
    .bind(&actor.source_chat_id)
    .bind(&actor.source_message_id)
    .bind(&actor.source_tool_call_id)
    .bind(created_at)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn replace_model_scope(
    tx: &mut Transaction<'_, Sqlite>,
    memory_id: &str,
    scope: &MemoryModelScope,
) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM memory_model_scopes WHERE memory_id = ?")
        .bind(memory_id)
        .execute(&mut **tx)
        .await?;
    if !scope.all_models {
        for model_key in &scope.model_keys {
            sqlx::query("INSERT INTO memory_model_scopes (memory_id, model_key) VALUES (?, ?)")
                .bind(memory_id)
                .bind(model_key)
                .execute(&mut **tx)
                .await?;
        }
    }
    Ok(())
}

async fn rebuild_fts(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    memory_id: &str,
    content: &str,
    active: bool,
) -> Result<(), ApiError> {
    delete_fts(tx, memory_id).await?;
    if active {
        sqlx::query("INSERT INTO memories_fts (memory_id, user_id, content) VALUES (?, ?, ?)")
            .bind(memory_id)
            .bind(user_id)
            .bind(content)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

async fn delete_fts(tx: &mut Transaction<'_, Sqlite>, memory_id: &str) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM memories_fts WHERE memory_id = ?")
        .bind(memory_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

async fn load_user_scopes(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<HashMap<String, Vec<String>>, ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT scopes.memory_id, scopes.model_key
        FROM memory_model_scopes scopes
        JOIN memories m ON m.id = scopes.memory_id
        WHERE m.user_id = ?
        ORDER BY scopes.model_key
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let mut scopes = HashMap::<String, Vec<String>>::new();
    for row in rows {
        scopes
            .entry(row.try_get("memory_id")?)
            .or_default()
            .push(row.try_get("model_key")?);
    }
    Ok(scopes)
}

async fn load_memory_scope(
    pool: &SqlitePool,
    user_id: &str,
    memory_id: &str,
) -> Result<MemoryModelScope, ApiError> {
    let all_models: bool =
        sqlx::query_scalar("SELECT all_models FROM memories WHERE id = ? AND user_id = ?")
            .bind(memory_id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?
            .ok_or_else(memory_not_found)?;
    let model_keys = sqlx::query_scalar(
        "SELECT model_key FROM memory_model_scopes WHERE memory_id = ? ORDER BY model_key",
    )
    .bind(memory_id)
    .fetch_all(pool)
    .await?;
    Ok(MemoryModelScope {
        all_models,
        model_keys,
    })
}

fn row_to_summary(
    row: SqliteRow,
    scopes: &HashMap<String, Vec<String>>,
) -> Result<MemorySummaryResponse, ApiError> {
    let id: String = row.try_get("id")?;
    let all_models: bool = row.try_get("all_models")?;
    Ok(MemorySummaryResponse {
        excerpt: row.try_get::<String, _>("excerpt")?.trim().to_string(),
        current_version_id: row.try_get("current_version_id")?,
        current_version_number: row.try_get("current_version_number")?,
        model_scope: MemoryModelScope {
            all_models,
            model_keys: scopes.get(&id).cloned().unwrap_or_default(),
        },
        deleted_at: row.try_get("deleted_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        id,
    })
}

fn row_to_memory(
    row: SqliteRow,
    model_scope: MemoryModelScope,
) -> Result<MemoryResponse, ApiError> {
    Ok(MemoryResponse {
        id: row.try_get("id")?,
        current_version: MemoryVersionResponse {
            id: row.try_get("version_id")?,
            memory_id: row.try_get("version_memory_id")?,
            version_number: row.try_get("version_number")?,
            content: row.try_get("content")?,
            actor_type: row.try_get("actor_type")?,
            actor_user_id: row.try_get("actor_user_id")?,
            actor_model_key: row.try_get("actor_model_key")?,
            actor_model_name: row.try_get("actor_model_name")?,
            source_chat_id: row.try_get("source_chat_id")?,
            source_message_id: row.try_get("source_message_id")?,
            source_tool_call_id: row.try_get("source_tool_call_id")?,
            created_at: row.try_get("version_created_at")?,
        },
        model_scope,
        deleted_at: row.try_get("deleted_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_version(row: SqliteRow) -> Result<MemoryVersionResponse, sqlx::Error> {
    Ok(MemoryVersionResponse {
        id: row.try_get("id")?,
        memory_id: row.try_get("memory_id")?,
        version_number: row.try_get("version_number")?,
        content: row.try_get("content")?,
        actor_type: row.try_get("actor_type")?,
        actor_user_id: row.try_get("actor_user_id")?,
        actor_model_key: row.try_get("actor_model_key")?,
        actor_model_name: row.try_get("actor_model_name")?,
        source_chat_id: row.try_get("source_chat_id")?,
        source_message_id: row.try_get("source_message_id")?,
        source_tool_call_id: row.try_get("source_tool_call_id")?,
        created_at: row.try_get("created_at")?,
    })
}

fn validate_content(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_memory_content",
            "Memory content is required",
        ));
    }
    if value.len() > MAX_MEMORY_CONTENT_BYTES {
        return Err(ApiError::bad_request(
            "invalid_memory_content",
            format!("Memory content must be {MAX_MEMORY_CONTENT_BYTES} bytes or fewer"),
        ));
    }
    Ok(value.to_string())
}

fn validate_model_scope(mut scope: MemoryModelScope) -> Result<MemoryModelScope, ApiError> {
    if scope.model_keys.len() > MAX_MEMORY_MODEL_SCOPES {
        return Err(ApiError::bad_request(
            "too_many_memory_models",
            format!("A memory can allow at most {MAX_MEMORY_MODEL_SCOPES} models"),
        ));
    }
    if scope.all_models {
        scope.model_keys.clear();
        return Ok(scope);
    }
    let mut seen = HashSet::new();
    let mut model_keys = Vec::with_capacity(scope.model_keys.len());
    for value in scope.model_keys {
        let value = value.trim();
        if value.is_empty()
            || value.chars().count() > MAX_MODEL_KEY_CHARS
            || !(value.starts_with("base:") || value.starts_with("persona:"))
        {
            return Err(ApiError::bad_request(
                "invalid_memory_model",
                "A memory scope contains an invalid model identity",
            ));
        }
        if seen.insert(value.to_string()) {
            model_keys.push(value.to_string());
        }
    }
    scope.model_keys = model_keys;
    Ok(scope)
}

fn normalize_search_query(query: Option<String>) -> Option<String> {
    query
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn build_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .take(12)
        .filter_map(|token| {
            let token = token.trim_matches(|character: char| character.is_ascii_punctuation());
            (!token.is_empty()).then(|| format!("\"{}\"*", token.replace('"', "\"\"")))
        })
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn ensure_actor_owner(user_id: &str, actor: &MemoryMutationActor) -> Result<(), ApiError> {
    if actor.actor_user_id != user_id {
        return Err(ApiError::forbidden(
            "memory_actor_mismatch",
            "A memory change cannot be attributed to another user",
        ));
    }
    match actor.actor_type {
        "human" => Ok(()),
        "model"
            if actor
                .actor_model_key
                .as_deref()
                .is_some_and(|v| !v.is_empty())
                && actor
                    .actor_model_name
                    .as_deref()
                    .is_some_and(|v| !v.is_empty())
                && actor
                    .source_chat_id
                    .as_deref()
                    .is_some_and(|v| !v.is_empty())
                && actor
                    .source_message_id
                    .as_deref()
                    .is_some_and(|v| !v.is_empty())
                && actor
                    .source_tool_call_id
                    .as_deref()
                    .is_some_and(|v| !v.is_empty()) =>
        {
            Ok(())
        }
        "model" => Err(ApiError::internal(
            "Model memory changes require complete attribution",
        )),
        _ => Err(ApiError::internal("Unknown memory mutation actor")),
    }
}

#[derive(Clone, Copy)]
enum ModelOperation {
    Edit,
    Forget,
}

async fn ensure_model_mutation_allowed(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    memory_id: &str,
    actor: &MemoryMutationActor,
    operation: ModelOperation,
) -> Result<(), ApiError> {
    if !actor.is_model() {
        return Ok(());
    }
    let model_key = actor
        .actor_model_key
        .as_deref()
        .ok_or_else(|| ApiError::internal("Model memory changes require a model identity"))?;
    let row = sqlx::query(
        r#"
        SELECT m.all_models, m.deleted_at,
               COALESCE(settings.allow_model_read, 0) AS allow_model_read,
               COALESCE(settings.allow_model_edit, 0) AS allow_model_edit,
               COALESCE(settings.allow_model_forget, 0) AS allow_model_forget,
               EXISTS (
                   SELECT 1 FROM memory_model_scopes scopes
                   WHERE scopes.memory_id = m.id AND scopes.model_key = ?
               ) AS model_matches
        FROM memories m
        LEFT JOIN user_memory_settings settings ON settings.user_id = m.user_id
        WHERE m.id = ? AND m.user_id = ?
        "#,
    )
    .bind(model_key)
    .bind(memory_id)
    .bind(user_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(model_memory_unavailable)?;
    let operation_allowed = match operation {
        ModelOperation::Edit => row.try_get::<bool, _>("allow_model_edit")?,
        ModelOperation::Forget => row.try_get::<bool, _>("allow_model_forget")?,
    };
    let in_scope =
        row.try_get::<bool, _>("all_models")? || row.try_get::<bool, _>("model_matches")?;
    if row.try_get::<bool, _>("allow_model_read")?
        && operation_allowed
        && in_scope
        && row.try_get::<Option<i64>, _>("deleted_at")?.is_none()
    {
        Ok(())
    } else {
        Err(model_memory_unavailable())
    }
}

async fn ensure_owned(pool: &SqlitePool, user_id: &str, memory_id: &str) -> Result<(), ApiError> {
    let exists: i64 =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM memories WHERE id = ? AND user_id = ?)")
            .bind(memory_id)
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    if exists == 0 {
        return Err(memory_not_found());
    }
    Ok(())
}

fn memory_not_found() -> ApiError {
    ApiError::not_found("memory_not_found", "Memory not found")
}

fn model_memory_unavailable() -> ApiError {
    ApiError::forbidden(
        "memory_unavailable_to_model",
        "This memory is not available to the current model",
    )
}

#[cfg(test)]
mod tests {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

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

    async fn user(pool: &SqlitePool) -> String {
        register_user(
            pool,
            "memory-user".to_string(),
            None,
            "secret-pass".to_string(),
        )
        .await
        .expect("register user")
        .user
        .id
    }

    fn model_actor(user_id: &str, model_key: &str) -> MemoryMutationActor {
        MemoryMutationActor::model(user_id, model_key, "Test Model", "chat", "message", "call")
    }

    #[tokio::test]
    async fn model_access_defaults_off_and_created_memory_is_model_scoped() {
        let pool = test_pool().await;
        let user_id = user(&pool).await;
        let actor = model_actor(&user_id, "base:test:model");
        let denied = create_memory(
            &pool,
            &user_id,
            CreateMemoryRequest {
                content: "Likes concise answers".to_string(),
                model_scope: None,
            },
            &actor,
        )
        .await;
        assert_eq!(denied.unwrap_err().code(), "memory_model_create_forbidden");

        update_memory_settings(
            &pool,
            &user_id,
            UpdateMemorySettingsRequest {
                allow_model_read: true,
                allow_model_create: true,
                allow_model_edit: true,
                allow_model_forget: false,
            },
        )
        .await
        .expect("enable memory tools");
        let created = create_memory(
            &pool,
            &user_id,
            CreateMemoryRequest {
                content: "Likes concise answers".to_string(),
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect("create scoped memory");
        assert!(!created.model_scope.all_models);
        assert_eq!(created.model_scope.model_keys, vec!["base:test:model"]);
        assert!(
            get_memory_for_model(&pool, &user_id, &created.id, "base:other:model")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn versions_conflict_and_forgetting_is_reversible_by_the_owner() {
        let pool = test_pool().await;
        let user_id = user(&pool).await;
        let actor = MemoryMutationActor::human(&user_id);
        let created = create_memory(
            &pool,
            &user_id,
            CreateMemoryRequest {
                content: "Uses Linux".to_string(),
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect("create memory");
        let updated = update_memory(
            &pool,
            &user_id,
            &created.id,
            UpdateMemoryRequest {
                expected_version: 1,
                content: Some("Uses Linux and Rust".to_string()),
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect("update memory");
        assert_eq!(updated.current_version.version_number, 2);
        let conflict = update_memory(
            &pool,
            &user_id,
            &created.id,
            UpdateMemoryRequest {
                expected_version: 1,
                content: Some("stale".to_string()),
                model_scope: None,
            },
            &actor,
        )
        .await
        .unwrap_err();
        assert_eq!(conflict.code(), "memory_version_conflict");

        forget_memory(&pool, &user_id, &created.id, 2, &actor)
            .await
            .expect("forget memory");
        let restored = restore_memory(&pool, &user_id, &created.id, 2)
            .await
            .expect("restore memory");
        assert!(restored.deleted_at.is_none());
        assert_eq!(restored.current_version.content, "Uses Linux and Rust");
    }
}
