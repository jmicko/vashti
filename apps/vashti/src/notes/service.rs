use std::collections::{HashMap, HashSet};

use sqlx::{Row, Sqlite, SqlitePool, Transaction, sqlite::SqliteRow};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    error::ApiError,
    notes::models::{
        CreateNoteRequest, NoteAiAccess, NoteModelScope, NoteResponse, NoteSettingsResponse,
        NoteSummaryResponse, NoteVersionResponse, UpdateNoteRequest, UpdateNoteSettingsRequest,
    },
};

pub const MAX_NOTE_TITLE_CHARS: usize = 200;
pub const MAX_NOTE_CONTENT_BYTES: usize = 900_000;
pub const MAX_NOTE_TAGS: usize = 32;
pub const MAX_NOTE_TAG_CHARS: usize = 64;
pub const MAX_NOTE_MODEL_SCOPES: usize = 64;
const MAX_MODEL_KEY_CHARS: usize = 512;
const DEFAULT_LIST_LIMIT: i64 = 50;
const MAX_LIST_LIMIT: i64 = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoteListStatus {
    Active,
    Trashed,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoteSort {
    Updated,
    Created,
    Title,
}

#[derive(Clone, Debug)]
pub struct ListNotesOptions {
    pub query: Option<String>,
    pub status: NoteListStatus,
    pub sort: NoteSort,
    pub limit: i64,
    pub offset: i64,
}

impl Default for ListNotesOptions {
    fn default() -> Self {
        Self {
            query: None,
            status: NoteListStatus::Active,
            sort: NoteSort::Updated,
            limit: DEFAULT_LIST_LIMIT,
            offset: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ListNotesResult {
    pub notes: Vec<NoteSummaryResponse>,
    pub total: i64,
}

#[derive(Clone, Debug)]
pub struct NoteMutationActor {
    pub actor_type: &'static str,
    pub actor_user_id: String,
    pub actor_model_key: Option<String>,
    pub actor_model_name: Option<String>,
    pub source_chat_id: Option<String>,
    pub source_message_id: Option<String>,
    pub source_tool_call_id: Option<String>,
}

impl NoteMutationActor {
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

struct NewNoteVersion<'a> {
    id: &'a str,
    note_id: &'a str,
    number: i64,
    title: &'a str,
    content: &'a str,
    created_at: i64,
}

#[derive(Debug)]
struct CurrentNote {
    current_version_id: String,
    version_number: i64,
    title: String,
    content: String,
    ai_access: NoteAiAccess,
    all_models: bool,
    is_pinned: bool,
    deleted_at: Option<i64>,
}

pub async fn list_notes(
    pool: &SqlitePool,
    user_id: &str,
    mut options: ListNotesOptions,
) -> Result<ListNotesResult, ApiError> {
    options.limit = options.limit.clamp(1, MAX_LIST_LIMIT);
    options.offset = options.offset.max(0);
    options.query = normalize_search_query(options.query);

    let rows = if options.query.is_some() && options.status == NoteListStatus::Active {
        search_active_notes(pool, user_id, &options).await?
    } else {
        list_notes_without_fts(pool, user_id, &options).await?
    };
    let total = count_notes(pool, user_id, &options).await?;
    let (tags, scopes) = load_library_metadata(pool, user_id).await?;
    let notes = rows
        .into_iter()
        .map(|row| row_to_summary(row, &tags, &scopes))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ListNotesResult { notes, total })
}

pub async fn search_notes_for_model(
    pool: &SqlitePool,
    user_id: &str,
    model_key: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<NoteSummaryResponse>, ApiError> {
    let query = normalize_search_query(Some(query.to_string())).ok_or_else(|| {
        ApiError::bad_request("invalid_note_search", "A note search query is required")
    })?;
    let search = build_fts_query(&query);
    if search.is_empty() {
        return Ok(Vec::new());
    }
    let rows = sqlx::query(
        r#"
        SELECT n.id,
               v.title,
               snippet(notes_fts, 3, '', '', ' ... ', 28) AS excerpt,
               n.current_version_id,
               v.version_number AS current_version_number,
               n.ai_access,
               n.all_models,
               n.is_pinned,
               n.deleted_at,
               n.created_at,
               n.updated_at
        FROM notes_fts
        JOIN notes n ON n.id = notes_fts.note_id
        JOIN note_versions v ON v.id = n.current_version_id
        JOIN user_note_settings settings ON settings.user_id = n.user_id
        WHERE notes_fts MATCH ?
          AND notes_fts.user_id = ?
          AND n.deleted_at IS NULL
          AND settings.allow_model_read = 1
          AND n.ai_access IN ('read', 'edit', 'manage')
          AND (
              n.all_models = 1
              OR EXISTS (
                  SELECT 1
                  FROM note_model_scopes scopes
                  WHERE scopes.note_id = n.id AND scopes.model_key = ?
              )
          )
        ORDER BY bm25(notes_fts, 0.0, 0.0, 5.0, 1.0, 2.0) ASC,
                 n.is_pinned DESC,
                 n.updated_at DESC
        LIMIT ?
        "#,
    )
    .bind(search)
    .bind(user_id)
    .bind(model_key)
    .bind(limit.clamp(1, 10))
    .fetch_all(pool)
    .await?;
    let (tags, scopes) = load_library_metadata(pool, user_id).await?;
    rows.into_iter()
        .map(|row| row_to_summary(row, &tags, &scopes))
        .collect()
}

pub async fn get_note_for_model(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
    model_key: &str,
) -> Result<NoteResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT n.id,
               n.ai_access,
               n.all_models,
               n.is_pinned,
               n.deleted_at,
               n.created_at,
               n.updated_at,
               v.id AS version_id,
               v.note_id AS version_note_id,
               v.version_number,
               v.title,
               v.content,
               v.actor_type,
               v.actor_user_id,
               v.actor_model_key,
               v.actor_model_name,
               v.source_chat_id,
               v.source_message_id,
               v.source_tool_call_id,
               v.created_at AS version_created_at
        FROM notes n
        JOIN note_versions v ON v.id = n.current_version_id
        JOIN user_note_settings settings ON settings.user_id = n.user_id
        WHERE n.id = ?
          AND n.user_id = ?
          AND n.deleted_at IS NULL
          AND settings.allow_model_read = 1
          AND n.ai_access IN ('read', 'edit', 'manage')
          AND (
              n.all_models = 1
              OR EXISTS (
                  SELECT 1
                  FROM note_model_scopes scopes
                  WHERE scopes.note_id = n.id AND scopes.model_key = ?
              )
          )
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .bind(model_key)
    .fetch_optional(pool)
    .await?
    .ok_or_else(model_note_unavailable)?;
    let tags = load_note_tags(pool, user_id, note_id).await?;
    let model_scope = load_note_scope(pool, user_id, note_id).await?;
    row_to_note(row, tags, model_scope)
}

pub async fn get_note(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<NoteResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT n.id,
               n.ai_access,
               n.all_models,
               n.is_pinned,
               n.deleted_at,
               n.created_at,
               n.updated_at,
               v.id AS version_id,
               v.note_id AS version_note_id,
               v.version_number,
               v.title,
               v.content,
               v.actor_type,
               v.actor_user_id,
               v.actor_model_key,
               v.actor_model_name,
               v.source_chat_id,
               v.source_message_id,
               v.source_tool_call_id,
               v.created_at AS version_created_at
        FROM notes n
        JOIN note_versions v ON v.id = n.current_version_id
        WHERE n.id = ? AND n.user_id = ?
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(note_not_found)?;
    let tags = load_note_tags(pool, user_id, note_id).await?;
    let model_scope = load_note_scope(pool, user_id, note_id).await?;
    row_to_note(row, tags, model_scope)
}

pub async fn create_note(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreateNoteRequest,
    actor: &NoteMutationActor,
) -> Result<NoteResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    let title = validate_title(&payload.title)?;
    let content = validate_content(&payload.content)?;
    let tags = validate_tags(payload.tags)?;
    let settings = get_note_settings(pool, user_id).await?;
    if actor.is_model() && !settings.allow_model_create {
        return Err(ApiError::forbidden(
            "note_model_create_forbidden",
            "Note creation is disabled for models",
        ));
    }
    let ai_access = if actor.is_model() {
        settings.default_ai_access
    } else {
        payload.ai_access.unwrap_or(settings.default_ai_access)
    };
    let scope = validate_model_scope(if actor.is_model() {
        settings.default_model_scope
    } else {
        payload.model_scope.unwrap_or(settings.default_model_scope)
    })?;
    let is_pinned = !actor.is_model() && payload.is_pinned;
    let note_id = Uuid::new_v4().to_string();
    let version_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO notes (
            id, user_id, current_version_id, ai_access, all_models, is_pinned,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&note_id)
    .bind(user_id)
    .bind(&version_id)
    .bind(ai_access.as_str())
    .bind(scope.all_models)
    .bind(is_pinned)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    insert_version(
        &mut tx,
        NewNoteVersion {
            id: &version_id,
            note_id: &note_id,
            number: 1,
            title: &title,
            content: &content,
            created_at: now,
        },
        actor,
    )
    .await?;
    replace_tags(&mut tx, user_id, &note_id, &tags, now).await?;
    replace_model_scope(&mut tx, &note_id, &scope).await?;
    rebuild_note_fts(&mut tx, user_id, &note_id, &title, &content, &tags, true).await?;
    tx.commit().await?;

    get_note(pool, user_id, &note_id).await
}

pub async fn update_note(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
    payload: UpdateNoteRequest,
    actor: &NoteMutationActor,
) -> Result<NoteResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    if actor.is_model()
        && (payload.tags.is_some()
            || payload.is_pinned.is_some()
            || payload.ai_access.is_some()
            || payload.model_scope.is_some())
    {
        return Err(ApiError::forbidden(
            "note_model_metadata_forbidden",
            "Models cannot change note tags, pin state, access, or model scope",
        ));
    }
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, note_id, payload.expected_version).await?;
    let current = get_current_note_in_tx(&mut tx, user_id, note_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "note_trashed",
            "Restore this note before editing it",
        ));
    }
    ensure_model_note_mutation_allowed(&mut tx, user_id, note_id, actor, NoteAiAccess::Edit)
        .await?;

    let title = match payload.title {
        Some(title) => validate_title(&title)?,
        None => current.title.clone(),
    };
    let content = match payload.content {
        Some(content) => validate_content(&content)?,
        None => current.content.clone(),
    };
    let tags = match payload.tags {
        Some(tags) => Some(validate_tags(tags)?),
        None => None,
    };
    let scope = match payload.model_scope {
        Some(scope) => Some(validate_model_scope(scope)?),
        None => None,
    };
    let ai_access = payload.ai_access.unwrap_or(current.ai_access);
    let is_pinned = payload.is_pinned.unwrap_or(current.is_pinned);
    let content_changed = title != current.title || content != current.content;
    let now = unix_timestamp();
    let next_version_id = if content_changed {
        let version_id = Uuid::new_v4().to_string();
        insert_version(
            &mut tx,
            NewNoteVersion {
                id: &version_id,
                note_id,
                number: current.version_number + 1,
                title: &title,
                content: &content,
                created_at: now,
            },
            actor,
        )
        .await?;
        version_id
    } else {
        current.current_version_id
    };

    sqlx::query(
        r#"
        UPDATE notes
        SET current_version_id = ?, ai_access = ?, all_models = ?,
            is_pinned = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
        "#,
    )
    .bind(next_version_id)
    .bind(ai_access.as_str())
    .bind(
        scope
            .as_ref()
            .map_or(current.all_models, |value| value.all_models),
    )
    .bind(is_pinned)
    .bind(now)
    .bind(note_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    if let Some(tags) = &tags {
        replace_tags(&mut tx, user_id, note_id, tags, now).await?;
    }
    if let Some(scope) = &scope {
        replace_model_scope(&mut tx, note_id, scope).await?;
    }
    let current_tags = match tags {
        Some(tags) => tags,
        None => load_note_tags_in_tx(&mut tx, user_id, note_id).await?,
    };
    rebuild_note_fts(
        &mut tx,
        user_id,
        note_id,
        &title,
        &content,
        &current_tags,
        true,
    )
    .await?;
    cleanup_unused_tags(&mut tx, user_id).await?;
    tx.commit().await?;

    get_note(pool, user_id, note_id).await
}

pub async fn trash_note(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
    expected_version: i64,
    actor: &NoteMutationActor,
) -> Result<NoteResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, note_id, expected_version).await?;
    let current = get_current_note_in_tx(&mut tx, user_id, note_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "note_trashed",
            "Note is already in trash",
        ));
    }
    ensure_model_note_mutation_allowed(&mut tx, user_id, note_id, actor, NoteAiAccess::Manage)
        .await?;
    let now = unix_timestamp();
    sqlx::query("UPDATE notes SET deleted_at = ?, updated_at = ? WHERE id = ? AND user_id = ?")
        .bind(now)
        .bind(now)
        .bind(note_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    delete_note_fts(&mut tx, note_id).await?;
    tx.commit().await?;
    get_note(pool, user_id, note_id).await
}

pub async fn restore_note(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
    expected_version: i64,
) -> Result<NoteResponse, ApiError> {
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, note_id, expected_version).await?;
    let current = get_current_note_in_tx(&mut tx, user_id, note_id).await?;
    if current.deleted_at.is_none() {
        return Err(ApiError::conflict(
            "note_not_trashed",
            "Note is not in trash",
        ));
    }
    let now = unix_timestamp();
    sqlx::query("UPDATE notes SET deleted_at = NULL, updated_at = ? WHERE id = ? AND user_id = ?")
        .bind(now)
        .bind(note_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    let tags = load_note_tags_in_tx(&mut tx, user_id, note_id).await?;
    rebuild_note_fts(
        &mut tx,
        user_id,
        note_id,
        &current.title,
        &current.content,
        &tags,
        true,
    )
    .await?;
    tx.commit().await?;
    get_note(pool, user_id, note_id).await
}

pub async fn permanently_delete_note(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<(), ApiError> {
    let mut tx = pool.begin().await?;
    let deleted_at: Option<Option<i64>> =
        sqlx::query_scalar("SELECT deleted_at FROM notes WHERE id = ? AND user_id = ?")
            .bind(note_id)
            .bind(user_id)
            .fetch_optional(&mut *tx)
            .await?;
    match deleted_at {
        None => return Err(note_not_found()),
        Some(None) => {
            return Err(ApiError::conflict(
                "note_not_trashed",
                "Move this note to trash before deleting it permanently",
            ));
        }
        Some(Some(_)) => {}
    }
    delete_note_fts(&mut tx, note_id).await?;
    sqlx::query("DELETE FROM notes WHERE id = ? AND user_id = ?")
        .bind(note_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    cleanup_unused_tags(&mut tx, user_id).await?;
    tx.commit().await?;
    Ok(())
}

pub async fn list_versions(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<Vec<NoteVersionResponse>, ApiError> {
    ensure_note_owned(pool, user_id, note_id).await?;
    let rows = sqlx::query(
        r#"
        SELECT id, note_id, version_number, title, content, actor_type,
               actor_user_id, actor_model_key, actor_model_name, source_chat_id,
               source_message_id, source_tool_call_id, created_at
        FROM note_versions
        WHERE note_id = ?
        ORDER BY version_number DESC
        "#,
    )
    .bind(note_id)
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
    note_id: &str,
    version_id: &str,
    expected_version: i64,
    actor: &NoteMutationActor,
) -> Result<NoteResponse, ApiError> {
    ensure_actor_owner(user_id, actor)?;
    let mut tx = pool.begin().await?;
    acquire_expected_version(&mut tx, user_id, note_id, expected_version).await?;
    let current = get_current_note_in_tx(&mut tx, user_id, note_id).await?;
    if current.deleted_at.is_some() {
        return Err(ApiError::conflict(
            "note_trashed",
            "Restore this note before restoring its history",
        ));
    }
    let source =
        sqlx::query("SELECT title, content FROM note_versions WHERE id = ? AND note_id = ?")
            .bind(version_id)
            .bind(note_id)
            .fetch_optional(&mut *tx)
            .await?
            .ok_or_else(|| {
                ApiError::not_found("note_version_not_found", "Note version not found")
            })?;
    let title: String = source.try_get("title")?;
    let content: String = source.try_get("content")?;
    let next_version_id = Uuid::new_v4().to_string();
    let now = unix_timestamp();
    insert_version(
        &mut tx,
        NewNoteVersion {
            id: &next_version_id,
            note_id,
            number: current.version_number + 1,
            title: &title,
            content: &content,
            created_at: now,
        },
        actor,
    )
    .await?;
    sqlx::query("UPDATE notes SET current_version_id = ?, updated_at = ? WHERE id = ?")
        .bind(next_version_id)
        .bind(now)
        .bind(note_id)
        .execute(&mut *tx)
        .await?;
    let tags = load_note_tags_in_tx(&mut tx, user_id, note_id).await?;
    rebuild_note_fts(&mut tx, user_id, note_id, &title, &content, &tags, true).await?;
    tx.commit().await?;
    get_note(pool, user_id, note_id).await
}

pub async fn get_note_settings(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<NoteSettingsResponse, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT allow_model_read, allow_model_create, allow_model_edit,
               allow_model_trash, default_ai_access, default_all_models
        FROM user_note_settings
        WHERE user_id = ?
        "#,
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;
    let model_keys = sqlx::query_scalar(
        "SELECT model_key FROM user_note_default_model_scopes WHERE user_id = ? ORDER BY model_key",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let Some(row) = row else {
        return Ok(NoteSettingsResponse {
            allow_model_read: false,
            allow_model_create: false,
            allow_model_edit: false,
            allow_model_trash: false,
            default_ai_access: NoteAiAccess::None,
            default_model_scope: NoteModelScope {
                all_models: true,
                model_keys,
            },
        });
    };
    Ok(NoteSettingsResponse {
        allow_model_read: row.try_get("allow_model_read")?,
        allow_model_create: row.try_get("allow_model_create")?,
        allow_model_edit: row.try_get("allow_model_edit")?,
        allow_model_trash: row.try_get("allow_model_trash")?,
        default_ai_access: parse_ai_access(row.try_get::<String, _>("default_ai_access")?)?,
        default_model_scope: NoteModelScope {
            all_models: row.try_get("default_all_models")?,
            model_keys,
        },
    })
}

pub async fn update_note_settings(
    pool: &SqlitePool,
    user_id: &str,
    payload: UpdateNoteSettingsRequest,
) -> Result<NoteSettingsResponse, ApiError> {
    let scope = validate_model_scope(payload.default_model_scope)?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"
        INSERT INTO user_note_settings (
            user_id, allow_model_read, allow_model_create, allow_model_edit,
            allow_model_trash, default_ai_access, default_all_models, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            allow_model_read = excluded.allow_model_read,
            allow_model_create = excluded.allow_model_create,
            allow_model_edit = excluded.allow_model_edit,
            allow_model_trash = excluded.allow_model_trash,
            default_ai_access = excluded.default_ai_access,
            default_all_models = excluded.default_all_models,
            updated_at = excluded.updated_at
        "#,
    )
    .bind(user_id)
    .bind(payload.allow_model_read)
    .bind(payload.allow_model_create)
    .bind(payload.allow_model_edit)
    .bind(payload.allow_model_trash)
    .bind(payload.default_ai_access.as_str())
    .bind(scope.all_models)
    .bind(now)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM user_note_default_model_scopes WHERE user_id = ?")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    if !scope.all_models {
        for model_key in &scope.model_keys {
            sqlx::query(
                "INSERT INTO user_note_default_model_scopes (user_id, model_key) VALUES (?, ?)",
            )
            .bind(user_id)
            .bind(model_key)
            .execute(&mut *tx)
            .await?;
        }
    }
    tx.commit().await?;
    get_note_settings(pool, user_id).await
}

async fn list_notes_without_fts(
    pool: &SqlitePool,
    user_id: &str,
    options: &ListNotesOptions,
) -> Result<Vec<SqliteRow>, ApiError> {
    let status_clause = status_clause(options.status);
    let order_clause = sort_clause(options.sort);
    let query_filter = options.query.as_ref().map(|query| format!("%{}%", query));
    let sql = format!(
        r#"
        SELECT n.id,
               v.title,
               substr(v.content, 1, 260) AS excerpt,
               n.current_version_id,
               v.version_number AS current_version_number,
               n.ai_access,
               n.all_models,
               n.is_pinned,
               n.deleted_at,
               n.created_at,
               n.updated_at
        FROM notes n
        JOIN note_versions v ON v.id = n.current_version_id
        WHERE n.user_id = ?
          AND {status_clause}
          AND (? IS NULL OR v.title LIKE ? COLLATE NOCASE OR v.content LIKE ? COLLATE NOCASE)
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?
        "#,
    );
    Ok(sqlx::query(&sql)
        .bind(user_id)
        .bind(query_filter.as_deref())
        .bind(query_filter.as_deref())
        .bind(query_filter.as_deref())
        .bind(options.limit)
        .bind(options.offset)
        .fetch_all(pool)
        .await?)
}

async fn search_active_notes(
    pool: &SqlitePool,
    user_id: &str,
    options: &ListNotesOptions,
) -> Result<Vec<SqliteRow>, ApiError> {
    let search = build_fts_query(options.query.as_deref().unwrap_or_default());
    if search.is_empty() {
        return list_notes_without_fts(pool, user_id, options).await;
    }
    let order_clause = match options.sort {
        NoteSort::Updated => {
            "bm25(notes_fts, 0.0, 0.0, 5.0, 1.0, 2.0) ASC, n.is_pinned DESC, n.updated_at DESC"
        }
        NoteSort::Created => "n.is_pinned DESC, n.created_at DESC",
        NoteSort::Title => "n.is_pinned DESC, v.title COLLATE NOCASE ASC",
    };
    let sql = format!(
        r#"
        SELECT n.id,
               v.title,
               snippet(notes_fts, 3, '', '', ' ... ', 28) AS excerpt,
               n.current_version_id,
               v.version_number AS current_version_number,
               n.ai_access,
               n.all_models,
               n.is_pinned,
               n.deleted_at,
               n.created_at,
               n.updated_at
        FROM notes_fts
        JOIN notes n ON n.id = notes_fts.note_id
        JOIN note_versions v ON v.id = n.current_version_id
        WHERE notes_fts MATCH ?
          AND notes_fts.user_id = ?
          AND n.deleted_at IS NULL
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?
        "#,
    );
    Ok(sqlx::query(&sql)
        .bind(search)
        .bind(user_id)
        .bind(options.limit)
        .bind(options.offset)
        .fetch_all(pool)
        .await?)
}

async fn count_notes(
    pool: &SqlitePool,
    user_id: &str,
    options: &ListNotesOptions,
) -> Result<i64, ApiError> {
    if options.query.is_some() && options.status == NoteListStatus::Active {
        let search = build_fts_query(options.query.as_deref().unwrap_or_default());
        if !search.is_empty() {
            return Ok(sqlx::query_scalar(
                r#"
                SELECT COUNT(*)
                FROM notes_fts
                JOIN notes n ON n.id = notes_fts.note_id
                WHERE notes_fts MATCH ?
                  AND notes_fts.user_id = ?
                  AND n.deleted_at IS NULL
                "#,
            )
            .bind(search)
            .bind(user_id)
            .fetch_one(pool)
            .await?);
        }
    }
    let query_filter = options.query.as_ref().map(|query| format!("%{}%", query));
    let sql = format!(
        r#"
        SELECT COUNT(*)
        FROM notes n
        JOIN note_versions v ON v.id = n.current_version_id
        WHERE n.user_id = ?
          AND {}
          AND (? IS NULL OR v.title LIKE ? COLLATE NOCASE OR v.content LIKE ? COLLATE NOCASE)
        "#,
        status_clause(options.status),
    );
    Ok(sqlx::query_scalar(&sql)
        .bind(user_id)
        .bind(query_filter.as_deref())
        .bind(query_filter.as_deref())
        .bind(query_filter.as_deref())
        .fetch_one(pool)
        .await?)
}

fn status_clause(status: NoteListStatus) -> &'static str {
    match status {
        NoteListStatus::Active => "n.deleted_at IS NULL",
        NoteListStatus::Trashed => "n.deleted_at IS NOT NULL",
        NoteListStatus::All => "1 = 1",
    }
}

fn sort_clause(sort: NoteSort) -> &'static str {
    match sort {
        NoteSort::Updated => "n.is_pinned DESC, n.updated_at DESC",
        NoteSort::Created => "n.is_pinned DESC, n.created_at DESC",
        NoteSort::Title => "n.is_pinned DESC, v.title COLLATE NOCASE ASC",
    }
}

async fn acquire_expected_version(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
    expected_version: i64,
) -> Result<(), ApiError> {
    let result = sqlx::query(
        r#"
        UPDATE notes
        SET updated_at = updated_at
        WHERE id = ?
          AND user_id = ?
          AND current_version_id IN (
              SELECT id FROM note_versions
              WHERE note_id = ? AND version_number = ?
          )
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .bind(note_id)
    .bind(expected_version)
    .execute(&mut **tx)
    .await?;
    if result.rows_affected() != 0 {
        return Ok(());
    }
    let exists: i64 =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM notes WHERE id = ? AND user_id = ?)")
            .bind(note_id)
            .bind(user_id)
            .fetch_one(&mut **tx)
            .await?;
    if exists == 0 {
        return Err(note_not_found());
    }
    Err(ApiError::conflict(
        "note_version_conflict",
        "This note changed elsewhere. Reload it before saving",
    ))
}

async fn get_current_note_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
) -> Result<CurrentNote, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT n.current_version_id, n.ai_access, n.all_models, n.is_pinned,
               n.deleted_at, v.version_number, v.title, v.content
        FROM notes n
        JOIN note_versions v ON v.id = n.current_version_id
        WHERE n.id = ? AND n.user_id = ?
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(note_not_found)?;
    Ok(CurrentNote {
        current_version_id: row.try_get("current_version_id")?,
        version_number: row.try_get("version_number")?,
        title: row.try_get("title")?,
        content: row.try_get("content")?,
        ai_access: parse_ai_access(row.try_get::<String, _>("ai_access")?)?,
        all_models: row.try_get("all_models")?,
        is_pinned: row.try_get("is_pinned")?,
        deleted_at: row.try_get("deleted_at")?,
    })
}

async fn insert_version(
    tx: &mut Transaction<'_, Sqlite>,
    version: NewNoteVersion<'_>,
    actor: &NoteMutationActor,
) -> Result<(), ApiError> {
    sqlx::query(
        r#"
        INSERT INTO note_versions (
            id, note_id, version_number, title, content, actor_type,
            actor_user_id, actor_model_key, actor_model_name, source_chat_id,
            source_message_id, source_tool_call_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(version.id)
    .bind(version.note_id)
    .bind(version.number)
    .bind(version.title)
    .bind(version.content)
    .bind(actor.actor_type)
    .bind(&actor.actor_user_id)
    .bind(&actor.actor_model_key)
    .bind(&actor.actor_model_name)
    .bind(&actor.source_chat_id)
    .bind(&actor.source_message_id)
    .bind(&actor.source_tool_call_id)
    .bind(version.created_at)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn replace_tags(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
    tags: &[String],
    now: i64,
) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM note_tag_links WHERE note_id = ?")
        .bind(note_id)
        .execute(&mut **tx)
        .await?;
    for tag in tags {
        sqlx::query(
            "INSERT INTO note_tags (id, user_id, name, created_at) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        )
        .bind(Uuid::new_v4().to_string())
        .bind(user_id)
        .bind(tag)
        .bind(now)
        .execute(&mut **tx)
        .await?;
        let tag_id: String = sqlx::query_scalar(
            "SELECT id FROM note_tags WHERE user_id = ? AND name = ? COLLATE NOCASE",
        )
        .bind(user_id)
        .bind(tag)
        .fetch_one(&mut **tx)
        .await?;
        sqlx::query("INSERT INTO note_tag_links (note_id, tag_id) VALUES (?, ?)")
            .bind(note_id)
            .bind(tag_id)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

async fn replace_model_scope(
    tx: &mut Transaction<'_, Sqlite>,
    note_id: &str,
    scope: &NoteModelScope,
) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM note_model_scopes WHERE note_id = ?")
        .bind(note_id)
        .execute(&mut **tx)
        .await?;
    if !scope.all_models {
        for model_key in &scope.model_keys {
            sqlx::query("INSERT INTO note_model_scopes (note_id, model_key) VALUES (?, ?)")
                .bind(note_id)
                .bind(model_key)
                .execute(&mut **tx)
                .await?;
        }
    }
    Ok(())
}

async fn rebuild_note_fts(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
    title: &str,
    content: &str,
    tags: &[String],
    active: bool,
) -> Result<(), ApiError> {
    delete_note_fts(tx, note_id).await?;
    if active {
        sqlx::query(
            "INSERT INTO notes_fts (note_id, user_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(note_id)
        .bind(user_id)
        .bind(title)
        .bind(content)
        .bind(tags.join(" "))
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

async fn delete_note_fts(tx: &mut Transaction<'_, Sqlite>, note_id: &str) -> Result<(), ApiError> {
    sqlx::query("DELETE FROM notes_fts WHERE note_id = ?")
        .bind(note_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

async fn cleanup_unused_tags(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
) -> Result<(), ApiError> {
    sqlx::query(
        r#"
        DELETE FROM note_tags
        WHERE user_id = ?
          AND NOT EXISTS (
              SELECT 1 FROM note_tag_links WHERE tag_id = note_tags.id
          )
        "#,
    )
    .bind(user_id)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn load_note_tags(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<Vec<String>, ApiError> {
    Ok(sqlx::query_scalar(
        r#"
        SELECT t.name
        FROM note_tag_links links
        JOIN note_tags t ON t.id = links.tag_id
        JOIN notes n ON n.id = links.note_id
        WHERE links.note_id = ? AND n.user_id = ?
        ORDER BY t.name COLLATE NOCASE
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .fetch_all(pool)
    .await?)
}

async fn load_note_tags_in_tx(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
) -> Result<Vec<String>, ApiError> {
    Ok(sqlx::query_scalar(
        r#"
        SELECT t.name
        FROM note_tag_links links
        JOIN note_tags t ON t.id = links.tag_id
        JOIN notes n ON n.id = links.note_id
        WHERE links.note_id = ? AND n.user_id = ?
        ORDER BY t.name COLLATE NOCASE
        "#,
    )
    .bind(note_id)
    .bind(user_id)
    .fetch_all(&mut **tx)
    .await?)
}

async fn load_note_scope(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<NoteModelScope, ApiError> {
    let all_models: bool =
        sqlx::query_scalar("SELECT all_models FROM notes WHERE id = ? AND user_id = ?")
            .bind(note_id)
            .bind(user_id)
            .fetch_optional(pool)
            .await?
            .ok_or_else(note_not_found)?;
    let model_keys = sqlx::query_scalar(
        "SELECT model_key FROM note_model_scopes WHERE note_id = ? ORDER BY model_key",
    )
    .bind(note_id)
    .fetch_all(pool)
    .await?;
    Ok(NoteModelScope {
        all_models,
        model_keys,
    })
}

type LibraryMetadata = (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>);

async fn load_library_metadata(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<LibraryMetadata, ApiError> {
    let tag_rows = sqlx::query(
        r#"
        SELECT links.note_id, t.name
        FROM note_tag_links links
        JOIN note_tags t ON t.id = links.tag_id
        JOIN notes n ON n.id = links.note_id
        WHERE n.user_id = ?
        ORDER BY t.name COLLATE NOCASE
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let scope_rows = sqlx::query(
        r#"
        SELECT scopes.note_id, scopes.model_key
        FROM note_model_scopes scopes
        JOIN notes n ON n.id = scopes.note_id
        WHERE n.user_id = ?
        ORDER BY scopes.model_key
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let mut tags: HashMap<String, Vec<String>> = HashMap::new();
    for row in tag_rows {
        tags.entry(row.try_get("note_id")?)
            .or_default()
            .push(row.try_get("name")?);
    }
    let mut scopes: HashMap<String, Vec<String>> = HashMap::new();
    for row in scope_rows {
        scopes
            .entry(row.try_get("note_id")?)
            .or_default()
            .push(row.try_get("model_key")?);
    }
    Ok((tags, scopes))
}

fn row_to_summary(
    row: SqliteRow,
    tags: &HashMap<String, Vec<String>>,
    scopes: &HashMap<String, Vec<String>>,
) -> Result<NoteSummaryResponse, ApiError> {
    let id: String = row.try_get("id")?;
    let all_models: bool = row.try_get("all_models")?;
    Ok(NoteSummaryResponse {
        title: row.try_get("title")?,
        excerpt: row.try_get::<String, _>("excerpt")?.trim().to_string(),
        current_version_id: row.try_get("current_version_id")?,
        current_version_number: row.try_get("current_version_number")?,
        tags: tags.get(&id).cloned().unwrap_or_default(),
        is_pinned: row.try_get("is_pinned")?,
        ai_access: parse_ai_access(row.try_get::<String, _>("ai_access")?)?,
        model_scope: NoteModelScope {
            all_models,
            model_keys: scopes.get(&id).cloned().unwrap_or_default(),
        },
        deleted_at: row.try_get("deleted_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        id,
    })
}

fn row_to_note(
    row: SqliteRow,
    tags: Vec<String>,
    model_scope: NoteModelScope,
) -> Result<NoteResponse, ApiError> {
    Ok(NoteResponse {
        id: row.try_get("id")?,
        current_version: NoteVersionResponse {
            id: row.try_get("version_id")?,
            note_id: row.try_get("version_note_id")?,
            version_number: row.try_get("version_number")?,
            title: row.try_get("title")?,
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
        tags,
        is_pinned: row.try_get("is_pinned")?,
        ai_access: parse_ai_access(row.try_get::<String, _>("ai_access")?)?,
        model_scope,
        deleted_at: row.try_get("deleted_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_version(row: SqliteRow) -> Result<NoteVersionResponse, sqlx::Error> {
    Ok(NoteVersionResponse {
        id: row.try_get("id")?,
        note_id: row.try_get("note_id")?,
        version_number: row.try_get("version_number")?,
        title: row.try_get("title")?,
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

fn validate_title(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_note_title",
            "A note title is required",
        ));
    }
    if value.chars().count() > MAX_NOTE_TITLE_CHARS {
        return Err(ApiError::bad_request(
            "invalid_note_title",
            format!("Note titles must be {MAX_NOTE_TITLE_CHARS} characters or fewer"),
        ));
    }
    Ok(value.to_string())
}

fn validate_content(value: &str) -> Result<String, ApiError> {
    if value.len() > MAX_NOTE_CONTENT_BYTES {
        return Err(ApiError::bad_request(
            "invalid_note_content",
            format!("Note content must be {MAX_NOTE_CONTENT_BYTES} bytes or fewer"),
        ));
    }
    Ok(value.to_string())
}

fn validate_tags(values: Vec<String>) -> Result<Vec<String>, ApiError> {
    if values.len() > MAX_NOTE_TAGS {
        return Err(ApiError::bad_request(
            "too_many_note_tags",
            format!("A note can have at most {MAX_NOTE_TAGS} tags"),
        ));
    }
    let mut seen = HashSet::new();
    let mut tags = Vec::with_capacity(values.len());
    for value in values {
        let tag = value.trim();
        if tag.is_empty() {
            continue;
        }
        if tag.chars().count() > MAX_NOTE_TAG_CHARS {
            return Err(ApiError::bad_request(
                "invalid_note_tag",
                format!("Note tags must be {MAX_NOTE_TAG_CHARS} characters or fewer"),
            ));
        }
        let normalized = tag.to_lowercase();
        if seen.insert(normalized) {
            tags.push(tag.to_string());
        }
    }
    Ok(tags)
}

fn validate_model_scope(mut scope: NoteModelScope) -> Result<NoteModelScope, ApiError> {
    if scope.model_keys.len() > MAX_NOTE_MODEL_SCOPES {
        return Err(ApiError::bad_request(
            "too_many_note_models",
            format!("A note can allow at most {MAX_NOTE_MODEL_SCOPES} models"),
        ));
    }
    if scope.all_models {
        scope.model_keys.clear();
        return Ok(scope);
    }
    let mut seen = HashSet::new();
    let mut keys = Vec::with_capacity(scope.model_keys.len());
    for value in scope.model_keys {
        let key = value.trim();
        if key.is_empty()
            || key.chars().count() > MAX_MODEL_KEY_CHARS
            || !(key.starts_with("base:") || key.starts_with("persona:"))
        {
            return Err(ApiError::bad_request(
                "invalid_note_model",
                "A note model scope contains an invalid model identity",
            ));
        }
        if seen.insert(key.to_string()) {
            keys.push(key.to_string());
        }
    }
    scope.model_keys = keys;
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
            if token.is_empty() {
                None
            } else {
                Some(format!("\"{}\"*", token.replace('"', "\"\"")))
            }
        })
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn parse_ai_access(value: String) -> Result<NoteAiAccess, ApiError> {
    NoteAiAccess::try_from(value.as_str()).map_err(|error| {
        tracing::error!(%error, "invalid note AI access stored in database");
        ApiError::internal("Stored note permissions are invalid")
    })
}

fn ensure_actor_owner(user_id: &str, actor: &NoteMutationActor) -> Result<(), ApiError> {
    if actor.actor_user_id != user_id {
        return Err(ApiError::forbidden(
            "note_actor_mismatch",
            "A note change cannot be attributed to another user",
        ));
    }
    match actor.actor_type {
        "human" => Ok(()),
        "model"
            if actor
                .actor_model_key
                .as_deref()
                .is_some_and(|value| !value.is_empty())
                && actor
                    .actor_model_name
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
                && actor
                    .source_chat_id
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
                && actor
                    .source_message_id
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
                && actor
                    .source_tool_call_id
                    .as_deref()
                    .is_some_and(|value| !value.is_empty()) =>
        {
            Ok(())
        }
        "model" => Err(ApiError::internal(
            "Model note changes require complete attribution",
        )),
        _ => Err(ApiError::internal("Unknown note mutation actor")),
    }
}

async fn ensure_model_note_mutation_allowed(
    tx: &mut Transaction<'_, Sqlite>,
    user_id: &str,
    note_id: &str,
    actor: &NoteMutationActor,
    required_access: NoteAiAccess,
) -> Result<(), ApiError> {
    if !actor.is_model() {
        return Ok(());
    }
    let model_key = actor
        .actor_model_key
        .as_deref()
        .ok_or_else(|| ApiError::internal("Model note changes require a model identity"))?;
    let row = sqlx::query(
        r#"
        SELECT n.ai_access,
               n.deleted_at,
               settings.allow_model_read,
               settings.allow_model_edit,
               settings.allow_model_trash,
               n.all_models,
               EXISTS (
                   SELECT 1
                   FROM note_model_scopes scopes
                   WHERE scopes.note_id = n.id AND scopes.model_key = ?
               ) AS model_matches
        FROM notes n
        JOIN user_note_settings settings ON settings.user_id = n.user_id
        WHERE n.id = ? AND n.user_id = ?
        "#,
    )
    .bind(model_key)
    .bind(note_id)
    .bind(user_id)
    .fetch_optional(&mut **tx)
    .await?
    .ok_or_else(model_note_unavailable)?;
    let access = parse_ai_access(row.try_get::<String, _>("ai_access")?)?;
    let operation_allowed = match required_access {
        NoteAiAccess::Edit => row.try_get::<bool, _>("allow_model_edit")?,
        NoteAiAccess::Manage => row.try_get::<bool, _>("allow_model_trash")?,
        NoteAiAccess::Read => true,
        NoteAiAccess::None => false,
    };
    let can_read = row.try_get::<bool, _>("allow_model_read")?;
    let in_scope =
        row.try_get::<bool, _>("all_models")? || row.try_get::<bool, _>("model_matches")?;
    let active = row.try_get::<Option<i64>, _>("deleted_at")?.is_none();
    if can_read && operation_allowed && access >= required_access && in_scope && active {
        Ok(())
    } else {
        Err(model_note_unavailable())
    }
}

async fn ensure_note_owned(
    pool: &SqlitePool,
    user_id: &str,
    note_id: &str,
) -> Result<(), ApiError> {
    let exists: i64 =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM notes WHERE id = ? AND user_id = ?)")
            .bind(note_id)
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    if exists == 0 {
        return Err(note_not_found());
    }
    Ok(())
}

fn note_not_found() -> ApiError {
    ApiError::not_found("note_not_found", "Note not found")
}

fn model_note_unavailable() -> ApiError {
    ApiError::forbidden(
        "note_unavailable_to_model",
        "This note is not available to the current model",
    )
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

    async fn create_test_user(pool: &SqlitePool, username: &str) -> String {
        register_user(pool, username.to_string(), None, "secret-pass".to_string())
            .await
            .expect("register test user")
            .user
            .id
    }

    fn request(title: &str, content: &str) -> CreateNoteRequest {
        CreateNoteRequest {
            title: title.to_string(),
            content: content.to_string(),
            tags: vec!["Project".to_string()],
            is_pinned: false,
            ai_access: Some(NoteAiAccess::Read),
            model_scope: Some(NoteModelScope::default()),
        }
    }

    #[tokio::test]
    async fn notes_are_owner_scoped_and_searchable() {
        let pool = test_pool().await;
        let owner = create_test_user(&pool, "note-owner").await;
        let other = create_test_user(&pool, "note-other").await;
        let note = create_note(
            &pool,
            &owner,
            request("Rust rules", "Never use unwrap in production Rust."),
            &NoteMutationActor::human(&owner),
        )
        .await
        .expect("create note");

        let results = list_notes(
            &pool,
            &owner,
            ListNotesOptions {
                query: Some("unwrap".to_string()),
                ..ListNotesOptions::default()
            },
        )
        .await
        .expect("search notes");
        assert_eq!(results.total, 1);
        assert_eq!(results.notes[0].id, note.id);
        assert!(get_note(&pool, &other, &note.id).await.is_err());
        let other_results = list_notes(
            &pool,
            &other,
            ListNotesOptions {
                query: Some("unwrap".to_string()),
                ..ListNotesOptions::default()
            },
        )
        .await
        .expect("search other notes");
        assert_eq!(other_results.total, 0);
    }

    #[tokio::test]
    async fn stale_edits_conflict_instead_of_overwriting() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool, "note-conflict").await;
        let note = create_note(
            &pool,
            &user_id,
            request("Draft", "One"),
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create note");
        let updated = update_note(
            &pool,
            &user_id,
            &note.id,
            UpdateNoteRequest {
                expected_version: 1,
                title: None,
                content: Some("Two".to_string()),
                tags: None,
                is_pinned: None,
                ai_access: None,
                model_scope: None,
            },
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("update note");
        assert_eq!(updated.current_version.version_number, 2);
        assert!(
            update_note(
                &pool,
                &user_id,
                &note.id,
                UpdateNoteRequest {
                    expected_version: 1,
                    title: None,
                    content: Some("Stale".to_string()),
                    tags: None,
                    is_pinned: None,
                    ai_access: None,
                    model_scope: None,
                },
                &NoteMutationActor::human(&user_id),
            )
            .await
            .is_err()
        );
        assert_eq!(
            get_note(&pool, &user_id, &note.id)
                .await
                .expect("reload note")
                .current_version
                .content,
            "Two"
        );
    }

    #[tokio::test]
    async fn trash_removes_search_result_and_only_trash_can_be_purged() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool, "note-trash").await;
        let note = create_note(
            &pool,
            &user_id,
            request("Temporary", "Delete me later"),
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create note");
        assert!(
            permanently_delete_note(&pool, &user_id, &note.id)
                .await
                .is_err()
        );
        trash_note(
            &pool,
            &user_id,
            &note.id,
            1,
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("trash note");
        let results = list_notes(
            &pool,
            &user_id,
            ListNotesOptions {
                query: Some("temporary".to_string()),
                ..ListNotesOptions::default()
            },
        )
        .await
        .expect("search active notes");
        assert_eq!(results.total, 0);
        permanently_delete_note(&pool, &user_id, &note.id)
            .await
            .expect("purge note");
        assert!(get_note(&pool, &user_id, &note.id).await.is_err());
    }

    #[tokio::test]
    async fn history_restore_creates_a_new_version() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool, "note-history").await;
        let note = create_note(
            &pool,
            &user_id,
            request("History", "First"),
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create note");
        update_note(
            &pool,
            &user_id,
            &note.id,
            UpdateNoteRequest {
                expected_version: 1,
                title: None,
                content: Some("Second".to_string()),
                tags: None,
                is_pinned: None,
                ai_access: None,
                model_scope: None,
            },
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("update note");
        let versions = list_versions(&pool, &user_id, &note.id)
            .await
            .expect("list history");
        let first = versions
            .iter()
            .find(|version| version.version_number == 1)
            .expect("first version");
        let restored = restore_version(
            &pool,
            &user_id,
            &note.id,
            &first.id,
            2,
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("restore version");
        assert_eq!(restored.current_version.version_number, 3);
        assert_eq!(restored.current_version.content, "First");
    }

    #[tokio::test]
    async fn model_scope_is_exact_and_model_changes_are_attributed() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool, "note-model-scope").await;
        let model_key = "persona:test-persona";
        update_note_settings(
            &pool,
            &user_id,
            UpdateNoteSettingsRequest {
                allow_model_read: true,
                allow_model_create: true,
                allow_model_edit: true,
                allow_model_trash: false,
                default_ai_access: NoteAiAccess::Edit,
                default_model_scope: NoteModelScope {
                    all_models: false,
                    model_keys: vec![model_key.to_string()],
                },
            },
        )
        .await
        .expect("enable model notes");
        let actor = NoteMutationActor::model(
            &user_id,
            model_key,
            "Test custom model",
            "chat-id",
            "message-id",
            "tool-call-id",
        );
        let note = create_note(
            &pool,
            &user_id,
            CreateNoteRequest {
                title: "Scoped".to_string(),
                content: "Only one custom model can read this.".to_string(),
                tags: Vec::new(),
                is_pinned: true,
                ai_access: Some(NoteAiAccess::Manage),
                model_scope: Some(NoteModelScope::default()),
            },
            &actor,
        )
        .await
        .expect("create model-authored note");

        assert!(!note.model_scope.all_models);
        assert_eq!(note.model_scope.model_keys, vec![model_key]);
        assert_eq!(note.ai_access, NoteAiAccess::Edit);
        assert!(!note.is_pinned);
        assert_eq!(note.current_version.actor_type, "model");
        assert_eq!(
            note.current_version.actor_model_key.as_deref(),
            Some(model_key)
        );

        assert_eq!(
            search_notes_for_model(&pool, &user_id, model_key, "custom", 5)
                .await
                .expect("search in-scope notes")
                .len(),
            1
        );
        assert!(
            search_notes_for_model(&pool, &user_id, "persona:other", "custom", 5)
                .await
                .expect("search out-of-scope notes")
                .is_empty()
        );

        let updated = update_note(
            &pool,
            &user_id,
            &note.id,
            UpdateNoteRequest {
                expected_version: 1,
                title: None,
                content: Some("The permitted custom model changed this.".to_string()),
                tags: None,
                is_pinned: None,
                ai_access: None,
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect("model update note");
        assert_eq!(updated.current_version.version_number, 2);
        assert_eq!(
            updated.current_version.source_message_id.as_deref(),
            Some("message-id")
        );
        assert_eq!(
            updated.current_version.source_tool_call_id.as_deref(),
            Some("tool-call-id")
        );

        let metadata_error = update_note(
            &pool,
            &user_id,
            &note.id,
            UpdateNoteRequest {
                expected_version: 2,
                title: None,
                content: None,
                tags: Some(vec!["model-added".to_string()]),
                is_pinned: None,
                ai_access: None,
                model_scope: None,
            },
            &actor,
        )
        .await
        .expect_err("models cannot change note metadata");
        assert_eq!(metadata_error.code(), "note_model_metadata_forbidden");
    }

    #[tokio::test]
    async fn model_trash_requires_global_and_per_note_manage_access() {
        let pool = test_pool().await;
        let user_id = create_test_user(&pool, "note-model-trash").await;
        let model_key = "base:backend:model";
        update_note_settings(
            &pool,
            &user_id,
            UpdateNoteSettingsRequest {
                allow_model_read: true,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_trash: true,
                default_ai_access: NoteAiAccess::None,
                default_model_scope: NoteModelScope::default(),
            },
        )
        .await
        .expect("enable model trash");
        let editable = create_note(
            &pool,
            &user_id,
            CreateNoteRequest {
                ai_access: Some(NoteAiAccess::Edit),
                ..request("Editable", "This note cannot be trashed by a model.")
            },
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create editable note");
        let manageable = create_note(
            &pool,
            &user_id,
            CreateNoteRequest {
                ai_access: Some(NoteAiAccess::Manage),
                ..request("Manageable", "This note can be moved to trash.")
            },
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create manageable note");
        let actor = NoteMutationActor::model(
            &user_id,
            model_key,
            "Test model",
            "chat-id",
            "message-id",
            "trash-call-id",
        );

        assert!(
            trash_note(&pool, &user_id, &editable.id, 1, &actor)
                .await
                .is_err()
        );
        let trashed = trash_note(&pool, &user_id, &manageable.id, 1, &actor)
            .await
            .expect("trash manageable note");
        assert!(trashed.deleted_at.is_some());

        update_note_settings(
            &pool,
            &user_id,
            UpdateNoteSettingsRequest {
                allow_model_read: true,
                allow_model_create: false,
                allow_model_edit: false,
                allow_model_trash: false,
                default_ai_access: NoteAiAccess::None,
                default_model_scope: NoteModelScope::default(),
            },
        )
        .await
        .expect("disable model trash");
        let second = create_note(
            &pool,
            &user_id,
            CreateNoteRequest {
                ai_access: Some(NoteAiAccess::Manage),
                ..request("Second", "Global trash is now disabled.")
            },
            &NoteMutationActor::human(&user_id),
        )
        .await
        .expect("create second note");
        assert!(
            trash_note(&pool, &user_id, &second.id, 1, &actor)
                .await
                .is_err()
        );
    }
}
