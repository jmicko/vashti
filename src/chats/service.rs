use std::collections::HashSet;

use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    chats::{
        handlers::{
            BranchMessageRequest, CreateChatRequest, CreateMessageRequest, EditMessageRequest,
            GenerateChatRequest, RegenerateMessageRequest, SetActiveChildRequest,
            SetActiveRootRequest, UpdateChatRequest,
        },
        models::{ChatDetail, ChatMessage, ChatMessageRevision, ChatSummary},
    },
    error::ApiError,
    ollama::models::OllamaChatMessage,
};

#[derive(Debug)]
pub struct PreparedGeneration {
    pub backend_base_url: String,
    pub model_name: String,
    pub think_mode: Option<String>,
    pub prompt_messages: Vec<OllamaChatMessage>,
    pub user_message: Option<ChatMessage>,
    pub assistant_message: ChatMessage,
}

#[derive(Debug)]
struct EnabledBackend {
    id: String,
    base_url: String,
}

struct MessageModel {
    backend_id: String,
    model_name: String,
}

struct InsertMessage<'a> {
    id: &'a str,
    chat_id: &'a str,
    parent_message_id: Option<&'a str>,
    active_revision_id: &'a str,
    role: &'a str,
    status: &'a str,
    backend_id: Option<&'a str>,
    model_name: Option<&'a str>,
    think_mode: Option<&'a str>,
    started_at: Option<i64>,
    now: i64,
}

pub async fn list_chats(pool: &SqlitePool, user_id: &str) -> Result<Vec<ChatSummary>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT c.id,
               c.title,
               c.default_backend_id,
               b.name AS backend_name,
               c.default_model_name,
               c.updated_at,
               c.last_message_at,
               COUNT(m.id) AS message_count
        FROM chats c
        JOIN ollama_backends b ON b.id = c.default_backend_id
        LEFT JOIN chat_messages m ON m.chat_id = c.id
        WHERE c.user_id = ?
          AND c.archived_at IS NULL
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(ChatSummary {
                id: row.try_get("id")?,
                title: row.try_get("title")?,
                default_backend_id: row.try_get("default_backend_id")?,
                backend_name: row.try_get("backend_name")?,
                default_model_name: row.try_get("default_model_name")?,
                updated_at: row.try_get("updated_at")?,
                last_message_at: row.try_get("last_message_at")?,
                message_count: row.try_get("message_count")?,
            })
        })
        .collect()
}

pub async fn create_chat(
    pool: &SqlitePool,
    user_id: &str,
    payload: CreateChatRequest,
) -> Result<ChatDetail, ApiError> {
    let title = normalized_title(payload.title);
    let backend_id = payload.default_backend_id.trim().to_string();
    let model_name = payload.default_model_name.trim().to_string();

    if backend_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_backend",
            "A backend is required",
        ));
    }

    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "A model is required",
        ));
    }

    ensure_enabled_backend(pool, &backend_id).await?;

    let now = unix_timestamp();
    let chat_id = Uuid::new_v4().to_string();

    sqlx::query(
        r#"
        INSERT INTO chats (
            id,
            user_id,
            default_backend_id,
            default_model_name,
            title,
            chat_mode,
            created_at,
            updated_at,
            last_message_at
        )
        VALUES (?, ?, ?, ?, ?, 'standard', ?, ?, ?)
        "#,
    )
    .bind(&chat_id)
    .bind(user_id)
    .bind(backend_id)
    .bind(model_name)
    .bind(title)
    .bind(now)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    get_chat(pool, user_id, &chat_id).await
}

pub async fn get_chat(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
) -> Result<ChatDetail, ApiError> {
    let Some(row) = sqlx::query(
        r#"
        SELECT c.id,
               c.title,
               c.default_backend_id,
               b.name AS backend_name,
               c.default_model_name,
               c.active_root_message_id,
               c.created_at,
               c.updated_at
        FROM chats c
        JOIN ollama_backends b ON b.id = c.default_backend_id
        WHERE c.id = ?
          AND c.user_id = ?
          AND c.archived_at IS NULL
        "#,
    )
    .bind(chat_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::not_found("chat_not_found", "Chat not found"));
    };

    row_to_chat_detail(row).map_err(ApiError::from)
}

pub async fn update_chat(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    payload: UpdateChatRequest,
) -> Result<ChatDetail, ApiError> {
    let current = get_chat(pool, user_id, chat_id).await?;
    let title = payload.title.map(normalized_title).unwrap_or(current.title);
    let backend_id = match payload.default_backend_id {
        Some(backend_id) => {
            let backend_id = backend_id.trim().to_string();
            ensure_enabled_backend(pool, &backend_id).await?;
            backend_id
        }
        None => current.default_backend_id,
    };
    let model_name = match payload.default_model_name {
        Some(model_name) => {
            let model_name = model_name.trim().to_string();
            if model_name.is_empty() {
                return Err(ApiError::bad_request(
                    "invalid_model",
                    "A model is required",
                ));
            }
            model_name
        }
        None => current.default_model_name,
    };

    sqlx::query(
        r#"
        UPDATE chats
        SET title = ?,
            default_backend_id = ?,
            default_model_name = ?,
            updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(title)
    .bind(backend_id)
    .bind(model_name)
    .bind(unix_timestamp())
    .bind(chat_id)
    .bind(user_id)
    .execute(pool)
    .await?;

    get_chat(pool, user_id, chat_id).await
}

pub async fn delete_chat(pool: &SqlitePool, user_id: &str, chat_id: &str) -> Result<(), ApiError> {
    let result = sqlx::query(
        r#"
        DELETE FROM chats
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(chat_id)
    .bind(user_id)
    .execute(pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("chat_not_found", "Chat not found"));
    }

    Ok(())
}

pub async fn list_messages(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
) -> Result<(Option<String>, Vec<ChatMessage>), ApiError> {
    let chat = get_chat(pool, user_id, chat_id).await?;
    let rows = sqlx::query(
        r#"
        SELECT m.id,
               m.parent_message_id,
               m.active_child_message_id,
               m.active_revision_id,
               m.role,
               m.status,
               m.is_deleted,
               m.backend_id,
               m.model_name,
               m.think_mode,
               m.done_reason,
               m.error_text,
               m.started_at,
               m.completed_at,
               m.created_at,
               m.updated_at,
               r.id AS revision_id,
               r.content_text,
               r.thinking_text,
               r.source,
               r.created_at AS revision_created_at,
               (
                   SELECT COUNT(*)
                   FROM chat_message_revisions cr
                   WHERE cr.message_id = m.id
               ) AS revision_count
        FROM chat_messages m
        LEFT JOIN chat_message_revisions r ON r.id = m.active_revision_id
        WHERE m.chat_id = ?
        ORDER BY m.created_at ASC
        "#,
    )
    .bind(chat_id)
    .fetch_all(pool)
    .await?;

    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
        messages.push(row_to_message(row)?);
    }

    Ok((chat.active_root_message_id, messages))
}

pub async fn create_user_message(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    payload: CreateMessageRequest,
) -> Result<ChatMessage, ApiError> {
    let _chat = get_chat(pool, user_id, chat_id).await?;
    let content_text = payload.content_text.trim().to_string();
    if content_text.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Message text is required",
        ));
    }

    let parent_message_id = payload
        .parent_message_id
        .map(|parent_id| parent_id.trim().to_string())
        .filter(|parent_id| !parent_id.is_empty());

    if let Some(parent_id) = &parent_message_id {
        let exists: i64 = sqlx::query_scalar(
            r#"
            SELECT EXISTS(
                SELECT 1 FROM chat_messages
                WHERE id = ?
                  AND chat_id = ?
            )
            "#,
        )
        .bind(parent_id)
        .bind(chat_id)
        .fetch_one(pool)
        .await?;

        if exists == 0 {
            return Err(ApiError::bad_request(
                "invalid_parent",
                "Parent message not found",
            ));
        }
    }

    let now = unix_timestamp();
    let message_id = Uuid::new_v4().to_string();
    let revision_id = Uuid::new_v4().to_string();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO chat_messages (
            id,
            chat_id,
            parent_message_id,
            active_revision_id,
            role,
            status,
            is_deleted,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, 'user', 'complete', 0, ?, ?)
        "#,
    )
    .bind(&message_id)
    .bind(chat_id)
    .bind(&parent_message_id)
    .bind(&revision_id)
    .bind(now)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO chat_message_revisions (
            id,
            message_id,
            content_text,
            thinking_text,
            source,
            created_at
        )
        VALUES (?, ?, ?, '', 'original', ?)
        "#,
    )
    .bind(&revision_id)
    .bind(&message_id)
    .bind(content_text)
    .bind(now)
    .execute(&mut *tx)
    .await?;

    if let Some(parent_id) = &parent_message_id {
        sqlx::query(
            r#"
            UPDATE chat_messages
            SET active_child_message_id = ?,
                updated_at = ?
            WHERE id = ?
              AND chat_id = ?
            "#,
        )
        .bind(&message_id)
        .bind(now)
        .bind(parent_id)
        .bind(chat_id)
        .execute(&mut *tx)
        .await?;
    } else {
        sqlx::query(
            r#"
            UPDATE chats
            SET active_root_message_id = ?
            WHERE id = ?
              AND user_id = ?
            "#,
        )
        .bind(&message_id)
        .bind(chat_id)
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    }

    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    get_message(pool, user_id, chat_id, &message_id).await
}

pub async fn prepare_generation(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    payload: GenerateChatRequest,
) -> Result<PreparedGeneration, ApiError> {
    let chat = get_chat(pool, user_id, chat_id).await?;
    let content_text = payload.user_message.content_text.trim().to_string();
    if content_text.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Message text is required",
        ));
    }

    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let backend_id = payload
        .backend_id
        .map(|backend_id| backend_id.trim().to_string())
        .filter(|backend_id| !backend_id.is_empty())
        .or_else(|| latest_model.as_ref().map(|model| model.backend_id.clone()))
        .unwrap_or(chat.default_backend_id);
    let model_name = payload
        .model_name
        .map(|model_name| model_name.trim().to_string())
        .filter(|model_name| !model_name.is_empty())
        .or_else(|| latest_model.as_ref().map(|model| model.model_name.clone()))
        .unwrap_or(chat.default_model_name);

    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "A model is required",
        ));
    }

    let backend = enabled_backend(pool, &backend_id).await?;
    let parent_message_id =
        active_tail_message_id(pool, chat_id, chat.active_root_message_id).await?;
    let now = unix_timestamp();
    let user_message_id = Uuid::new_v4().to_string();
    let user_revision_id = Uuid::new_v4().to_string();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(payload.think_mode);
    let mut tx = pool.begin().await?;

    insert_message(
        &mut tx,
        InsertMessage {
            id: &user_message_id,
            chat_id,
            parent_message_id: parent_message_id.as_deref(),
            active_revision_id: &user_revision_id,
            role: "user",
            status: "complete",
            backend_id: None,
            model_name: None,
            think_mode: None,
            started_at: None,
            now,
        },
    )
    .await?;
    insert_revision(
        &mut tx,
        &user_revision_id,
        &user_message_id,
        &content_text,
        "",
        "original",
        now,
    )
    .await?;

    if let Some(parent_id) = &parent_message_id {
        set_active_child(&mut tx, chat_id, parent_id, &user_message_id, now).await?;
    } else {
        set_active_root(&mut tx, user_id, chat_id, &user_message_id).await?;
    }

    insert_message(
        &mut tx,
        InsertMessage {
            id: &assistant_message_id,
            chat_id,
            parent_message_id: Some(&user_message_id),
            active_revision_id: &assistant_revision_id,
            role: "assistant",
            status: "streaming",
            backend_id: Some(&backend.id),
            model_name: Some(&model_name),
            think_mode: think_mode.as_deref(),
            started_at: Some(now),
            now,
        },
    )
    .await?;
    insert_revision(
        &mut tx,
        &assistant_revision_id,
        &assistant_message_id,
        "",
        "",
        "original",
        now,
    )
    .await?;
    set_active_child(
        &mut tx,
        chat_id,
        &user_message_id,
        &assistant_message_id,
        now,
    )
    .await?;

    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let prompt_messages =
        active_prompt_messages(pool, user_id, chat_id, &assistant_message_id).await?;
    let user_message = get_message(pool, user_id, chat_id, &user_message_id).await?;
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        prompt_messages,
        user_message: Some(user_message),
        assistant_message,
    })
}

pub async fn prepare_regeneration(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    payload: RegenerateMessageRequest,
) -> Result<PreparedGeneration, ApiError> {
    let target = get_message(pool, user_id, chat_id, message_id).await?;
    if target.role != "assistant" {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Only assistant messages can be regenerated",
        ));
    }

    let parent_message_id = target.parent_message_id.clone().ok_or_else(|| {
        ApiError::bad_request(
            "invalid_message",
            "Assistant message does not have a parent prompt",
        )
    })?;
    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let backend_id = payload
        .backend_id
        .map(|backend_id| backend_id.trim().to_string())
        .filter(|backend_id| !backend_id.is_empty())
        .or_else(|| target.backend_id.clone())
        .or_else(|| latest_model.as_ref().map(|model| model.backend_id.clone()))
        .ok_or_else(|| ApiError::bad_request("invalid_backend", "A backend is required"))?;
    let model_name = payload
        .model_name
        .map(|model_name| model_name.trim().to_string())
        .filter(|model_name| !model_name.is_empty())
        .or_else(|| target.model_name.clone())
        .or_else(|| latest_model.as_ref().map(|model| model.model_name.clone()))
        .ok_or_else(|| ApiError::bad_request("invalid_model", "A model is required"))?;
    let backend = enabled_backend(pool, &backend_id).await?;
    let now = unix_timestamp();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(payload.think_mode).or(target.think_mode);
    let mut tx = pool.begin().await?;

    insert_message(
        &mut tx,
        InsertMessage {
            id: &assistant_message_id,
            chat_id,
            parent_message_id: Some(&parent_message_id),
            active_revision_id: &assistant_revision_id,
            role: "assistant",
            status: "streaming",
            backend_id: Some(&backend.id),
            model_name: Some(&model_name),
            think_mode: think_mode.as_deref(),
            started_at: Some(now),
            now,
        },
    )
    .await?;
    insert_revision(
        &mut tx,
        &assistant_revision_id,
        &assistant_message_id,
        "",
        "",
        "regeneration",
        now,
    )
    .await?;
    set_active_child(
        &mut tx,
        chat_id,
        &parent_message_id,
        &assistant_message_id,
        now,
    )
    .await?;
    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let prompt_messages =
        active_prompt_messages(pool, user_id, chat_id, &assistant_message_id).await?;
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        prompt_messages,
        user_message: None,
        assistant_message,
    })
}

pub async fn prepare_branch_generation(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    payload: BranchMessageRequest,
) -> Result<PreparedGeneration, ApiError> {
    let chat = get_chat(pool, user_id, chat_id).await?;
    let target = get_message(pool, user_id, chat_id, message_id).await?;
    if target.role != "user" {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Only user messages can be sent as edited branches",
        ));
    }
    if target.is_deleted {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Deleted messages cannot be branched",
        ));
    }

    let content_text = payload.content_text.trim().to_string();
    if content_text.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Message text is required",
        ));
    }

    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let backend_id = payload
        .backend_id
        .map(|backend_id| backend_id.trim().to_string())
        .filter(|backend_id| !backend_id.is_empty())
        .or_else(|| latest_model.as_ref().map(|model| model.backend_id.clone()))
        .unwrap_or(chat.default_backend_id);
    let model_name = payload
        .model_name
        .map(|model_name| model_name.trim().to_string())
        .filter(|model_name| !model_name.is_empty())
        .or_else(|| latest_model.as_ref().map(|model| model.model_name.clone()))
        .unwrap_or(chat.default_model_name);
    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "A model is required",
        ));
    }

    let backend = enabled_backend(pool, &backend_id).await?;
    let parent_message_id = target.parent_message_id.clone();
    let now = unix_timestamp();
    let user_message_id = Uuid::new_v4().to_string();
    let user_revision_id = Uuid::new_v4().to_string();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(payload.think_mode);
    let mut tx = pool.begin().await?;

    insert_message(
        &mut tx,
        InsertMessage {
            id: &user_message_id,
            chat_id,
            parent_message_id: parent_message_id.as_deref(),
            active_revision_id: &user_revision_id,
            role: "user",
            status: "complete",
            backend_id: None,
            model_name: None,
            think_mode: None,
            started_at: None,
            now,
        },
    )
    .await?;
    insert_revision(
        &mut tx,
        &user_revision_id,
        &user_message_id,
        &content_text,
        "",
        "edit",
        now,
    )
    .await?;

    if let Some(parent_id) = &parent_message_id {
        set_active_child(&mut tx, chat_id, parent_id, &user_message_id, now).await?;
    } else {
        set_active_root(&mut tx, user_id, chat_id, &user_message_id).await?;
    }

    insert_message(
        &mut tx,
        InsertMessage {
            id: &assistant_message_id,
            chat_id,
            parent_message_id: Some(&user_message_id),
            active_revision_id: &assistant_revision_id,
            role: "assistant",
            status: "streaming",
            backend_id: Some(&backend.id),
            model_name: Some(&model_name),
            think_mode: think_mode.as_deref(),
            started_at: Some(now),
            now,
        },
    )
    .await?;
    insert_revision(
        &mut tx,
        &assistant_revision_id,
        &assistant_message_id,
        "",
        "",
        "original",
        now,
    )
    .await?;
    set_active_child(
        &mut tx,
        chat_id,
        &user_message_id,
        &assistant_message_id,
        now,
    )
    .await?;

    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let prompt_messages =
        active_prompt_messages(pool, user_id, chat_id, &assistant_message_id).await?;
    let user_message = get_message(pool, user_id, chat_id, &user_message_id).await?;
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        prompt_messages,
        user_message: Some(user_message),
        assistant_message,
    })
}

pub async fn edit_message(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    payload: EditMessageRequest,
) -> Result<ChatMessage, ApiError> {
    ensure_message_owner(pool, user_id, chat_id, message_id).await?;
    let content_text = payload.content_text.trim().to_string();
    if content_text.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Message text is required",
        ));
    }

    let now = unix_timestamp();
    let revision_id = Uuid::new_v4().to_string();
    let mut tx = pool.begin().await?;

    insert_revision(
        &mut tx,
        &revision_id,
        message_id,
        &content_text,
        "",
        "edit",
        now,
    )
    .await?;
    sqlx::query(
        r#"
        UPDATE chat_messages
        SET active_revision_id = ?,
            is_deleted = 0,
            updated_at = ?
        WHERE id = ?
          AND chat_id = ?
        "#,
    )
    .bind(&revision_id)
    .bind(now)
    .bind(message_id)
    .bind(chat_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    get_message(pool, user_id, chat_id, message_id).await
}

pub async fn delete_message(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
) -> Result<ChatMessage, ApiError> {
    ensure_message_owner(pool, user_id, chat_id, message_id).await?;
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        UPDATE chat_message_revisions
        SET content_text = '',
            thinking_text = ''
        WHERE message_id = ?
        "#,
    )
    .bind(message_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        r#"
        UPDATE chat_messages
        SET is_deleted = 1,
            status = CASE WHEN status = 'streaming' THEN 'stopped' ELSE status END,
            done_reason = CASE WHEN status = 'streaming' THEN 'deleted' ELSE done_reason END,
            completed_at = CASE WHEN status = 'streaming' THEN ? ELSE completed_at END,
            updated_at = ?
        WHERE id = ?
          AND chat_id = ?
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(message_id)
    .bind(chat_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    get_message(pool, user_id, chat_id, message_id).await
}

pub async fn select_active_child(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    parent_message_id: &str,
    payload: SetActiveChildRequest,
) -> Result<ChatMessage, ApiError> {
    ensure_message_owner(pool, user_id, chat_id, parent_message_id).await?;
    let active_child_message_id = payload.active_child_message_id.trim().to_string();
    if active_child_message_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_child",
            "Active child message is required",
        ));
    }

    let child_exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM chat_messages
            WHERE id = ?
              AND chat_id = ?
              AND parent_message_id = ?
        )
        "#,
    )
    .bind(&active_child_message_id)
    .bind(chat_id)
    .bind(parent_message_id)
    .fetch_one(pool)
    .await?;

    if child_exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_child",
            "Child message not found for this parent",
        ));
    }

    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    set_active_child(
        &mut tx,
        chat_id,
        parent_message_id,
        &active_child_message_id,
        now,
    )
    .await?;
    sqlx::query(
        r#"
        UPDATE chats
        SET updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;

    get_message(pool, user_id, chat_id, parent_message_id).await
}

pub async fn select_active_root(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    payload: SetActiveRootRequest,
) -> Result<ChatDetail, ApiError> {
    let active_root_message_id = payload.active_root_message_id.trim().to_string();
    if active_root_message_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_root",
            "Active root message is required",
        ));
    }

    let root_exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM chat_messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE m.id = ?
              AND m.chat_id = ?
              AND c.user_id = ?
              AND c.archived_at IS NULL
              AND m.parent_message_id IS NULL
        )
        "#,
    )
    .bind(&active_root_message_id)
    .bind(chat_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    if root_exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_root",
            "Root message not found for this chat",
        ));
    }

    let now = unix_timestamp();
    sqlx::query(
        r#"
        UPDATE chats
        SET active_root_message_id = ?,
            updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(&active_root_message_id)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(pool)
    .await?;

    get_chat(pool, user_id, chat_id).await
}

pub async fn finish_generation(
    pool: &SqlitePool,
    assistant_message_id: &str,
    content_text: &str,
    thinking_text: &str,
    done_reason: Option<&str>,
) -> Result<(), ApiError> {
    update_generation_message(
        pool,
        assistant_message_id,
        content_text,
        thinking_text,
        "complete",
        done_reason,
        None,
    )
    .await
}

pub async fn stop_generation(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    assistant_message_id: &str,
    content_text: &str,
    thinking_text: &str,
) -> Result<(), ApiError> {
    ensure_message_owner(pool, user_id, chat_id, assistant_message_id).await?;
    update_generation_message(
        pool,
        assistant_message_id,
        content_text,
        thinking_text,
        "stopped",
        Some("stopped"),
        None,
    )
    .await
}

pub async fn stop_generation_by_id(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    assistant_message_id: &str,
) -> Result<(), ApiError> {
    ensure_message_owner(pool, user_id, chat_id, assistant_message_id).await?;
    let now = unix_timestamp();
    sqlx::query(
        r#"
        UPDATE chat_messages
        SET status = 'stopped',
            done_reason = 'stopped',
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?
        WHERE id = ?
          AND chat_id = ?
          AND status = 'streaming'
        "#,
    )
    .bind(now)
    .bind(now)
    .bind(assistant_message_id)
    .bind(chat_id)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn fail_generation(
    pool: &SqlitePool,
    assistant_message_id: &str,
    content_text: &str,
    thinking_text: &str,
    error_text: &str,
) -> Result<(), ApiError> {
    update_generation_message(
        pool,
        assistant_message_id,
        content_text,
        thinking_text,
        "error",
        Some("error"),
        Some(error_text),
    )
    .await
}

async fn ensure_enabled_backend(pool: &SqlitePool, backend_id: &str) -> Result<(), ApiError> {
    enabled_backend(pool, backend_id).await?;

    Ok(())
}

async fn enabled_backend(pool: &SqlitePool, backend_id: &str) -> Result<EnabledBackend, ApiError> {
    let Some(row) = sqlx::query(
        r#"
        SELECT id, base_url
        FROM ollama_backends
        WHERE id = ?
          AND is_enabled = 1
        "#,
    )
    .bind(backend_id)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::bad_request(
            "invalid_backend",
            "Enabled backend not found",
        ));
    };

    Ok(EnabledBackend {
        id: row.try_get("id")?,
        base_url: row.try_get("base_url")?,
    })
}

async fn latest_assistant_model(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
) -> Result<Option<MessageModel>, ApiError> {
    let Some(row) = sqlx::query(
        r#"
        SELECT m.backend_id, m.model_name
        FROM chat_messages m
        JOIN chats c ON c.id = m.chat_id
        WHERE m.chat_id = ?
          AND c.user_id = ?
          AND c.archived_at IS NULL
          AND m.role = 'assistant'
          AND m.backend_id IS NOT NULL
          AND m.model_name IS NOT NULL
        ORDER BY m.created_at DESC
        LIMIT 1
        "#,
    )
    .bind(chat_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    else {
        return Ok(None);
    };

    Ok(Some(MessageModel {
        backend_id: row.try_get("backend_id")?,
        model_name: row.try_get("model_name")?,
    }))
}

fn normalized_title(title: String) -> String {
    let title = title.trim();
    if title.is_empty() {
        "New Chat".to_string()
    } else {
        title.to_string()
    }
}

fn normalized_think_mode(think_mode: Option<String>) -> Option<String> {
    let mode = think_mode?.trim().to_ascii_lowercase();
    match mode.as_str() {
        "true" | "false" | "low" | "medium" | "high" => Some(mode),
        _ => None,
    }
}

async fn insert_message(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    params: InsertMessage<'_>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO chat_messages (
            id,
            chat_id,
            parent_message_id,
            active_revision_id,
            role,
            status,
            is_deleted,
            backend_id,
            model_name,
            think_mode,
            started_at,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(params.id)
    .bind(params.chat_id)
    .bind(params.parent_message_id)
    .bind(params.active_revision_id)
    .bind(params.role)
    .bind(params.status)
    .bind(params.backend_id)
    .bind(params.model_name)
    .bind(params.think_mode)
    .bind(params.started_at)
    .bind(params.now)
    .bind(params.now)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn insert_revision(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    revision_id: &str,
    message_id: &str,
    content_text: &str,
    thinking_text: &str,
    source: &str,
    now: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT INTO chat_message_revisions (
            id,
            message_id,
            content_text,
            thinking_text,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(revision_id)
    .bind(message_id)
    .bind(content_text)
    .bind(thinking_text)
    .bind(source)
    .bind(now)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn set_active_child(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    chat_id: &str,
    parent_message_id: &str,
    child_message_id: &str,
    now: i64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE chat_messages
        SET active_child_message_id = ?,
            updated_at = ?
        WHERE id = ?
          AND chat_id = ?
        "#,
    )
    .bind(child_message_id)
    .bind(now)
    .bind(parent_message_id)
    .bind(chat_id)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn set_active_root(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    user_id: &str,
    chat_id: &str,
    root_message_id: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        UPDATE chats
        SET active_root_message_id = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(root_message_id)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut **tx)
    .await?;

    Ok(())
}

async fn active_tail_message_id(
    pool: &SqlitePool,
    chat_id: &str,
    root_message_id: Option<String>,
) -> Result<Option<String>, ApiError> {
    let mut current = root_message_id;
    let mut tail = None;
    let mut seen = HashSet::new();

    while let Some(message_id) = current {
        if !seen.insert(message_id.clone()) {
            break;
        }

        let row = sqlx::query(
            r#"
            SELECT active_child_message_id
            FROM chat_messages
            WHERE id = ?
              AND chat_id = ?
            "#,
        )
        .bind(&message_id)
        .bind(chat_id)
        .fetch_optional(pool)
        .await?;

        let Some(row) = row else {
            break;
        };

        tail = Some(message_id);
        current = row.try_get("active_child_message_id")?;
    }

    Ok(tail)
}

async fn active_prompt_messages(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    stop_before_message_id: &str,
) -> Result<Vec<OllamaChatMessage>, ApiError> {
    let chat = get_chat(pool, user_id, chat_id).await?;
    let mut current = chat.active_root_message_id;
    let mut seen = HashSet::new();
    let mut messages = Vec::new();

    while let Some(message_id) = current {
        if message_id == stop_before_message_id || !seen.insert(message_id.clone()) {
            break;
        }

        let Some(row) = sqlx::query(
            r#"
            SELECT m.role,
                   m.is_deleted,
                   m.active_child_message_id,
                   r.content_text,
                   r.thinking_text
            FROM chat_messages m
            LEFT JOIN chat_message_revisions r ON r.id = m.active_revision_id
            WHERE m.id = ?
              AND m.chat_id = ?
            "#,
        )
        .bind(&message_id)
        .bind(chat_id)
        .fetch_optional(pool)
        .await?
        else {
            break;
        };

        let role: String = row.try_get("role")?;
        let is_deleted = row.try_get::<i64, _>("is_deleted")? != 0;
        let content: Option<String> = row.try_get("content_text")?;
        let thinking: Option<String> = row.try_get("thinking_text")?;

        if !is_deleted {
            if let Some(content) = content.filter(|content| !content.trim().is_empty()) {
                messages.push(OllamaChatMessage {
                    role,
                    content,
                    thinking: thinking.filter(|thinking| !thinking.trim().is_empty()),
                });
            }
        }

        current = row.try_get("active_child_message_id")?;
    }

    Ok(messages)
}

async fn update_generation_message(
    pool: &SqlitePool,
    assistant_message_id: &str,
    content_text: &str,
    thinking_text: &str,
    status: &str,
    done_reason: Option<&str>,
    error_text: Option<&str>,
) -> Result<(), ApiError> {
    let now = unix_timestamp();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        UPDATE chat_message_revisions
        SET content_text = ?,
            thinking_text = ?
        WHERE id = (
            SELECT active_revision_id
            FROM chat_messages
            WHERE id = ?
        )
        "#,
    )
    .bind(content_text)
    .bind(thinking_text)
    .bind(assistant_message_id)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        UPDATE chat_messages
        SET status = ?,
            done_reason = ?,
            error_text = ?,
            completed_at = ?,
            updated_at = ?
        WHERE id = ?
        "#,
    )
    .bind(status)
    .bind(done_reason)
    .bind(error_text)
    .bind(now)
    .bind(now)
    .bind(assistant_message_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    Ok(())
}

async fn ensure_message_owner(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM chat_messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE m.id = ?
              AND m.chat_id = ?
              AND c.user_id = ?
              AND c.archived_at IS NULL
        )
        "#,
    )
    .bind(message_id)
    .bind(chat_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    if exists == 0 {
        return Err(ApiError::not_found(
            "message_not_found",
            "Message not found",
        ));
    }

    Ok(())
}

fn row_to_chat_detail(row: sqlx::sqlite::SqliteRow) -> Result<ChatDetail, sqlx::Error> {
    Ok(ChatDetail {
        id: row.try_get("id")?,
        title: row.try_get("title")?,
        default_backend_id: row.try_get("default_backend_id")?,
        backend_name: row.try_get("backend_name")?,
        default_model_name: row.try_get("default_model_name")?,
        active_root_message_id: row.try_get("active_root_message_id")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn row_to_message(row: sqlx::sqlite::SqliteRow) -> Result<ChatMessage, sqlx::Error> {
    let active_revision = match row.try_get::<Option<String>, _>("revision_id")? {
        Some(id) => Some(ChatMessageRevision {
            id,
            content_text: row.try_get("content_text")?,
            thinking_text: row.try_get("thinking_text")?,
            source: row.try_get("source")?,
            created_at: row.try_get("revision_created_at")?,
        }),
        None => None,
    };

    Ok(ChatMessage {
        id: row.try_get("id")?,
        parent_message_id: row.try_get("parent_message_id")?,
        active_child_message_id: row.try_get("active_child_message_id")?,
        active_revision_id: row.try_get("active_revision_id")?,
        role: row.try_get("role")?,
        status: row.try_get("status")?,
        is_deleted: row.try_get::<i64, _>("is_deleted")? != 0,
        backend_id: row.try_get("backend_id")?,
        model_name: row.try_get("model_name")?,
        think_mode: row.try_get("think_mode")?,
        done_reason: row.try_get("done_reason")?,
        error_text: row.try_get("error_text")?,
        started_at: row.try_get("started_at")?,
        completed_at: row.try_get("completed_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        active_revision,
        revision_count: row.try_get("revision_count")?,
        attachments: Vec::new(),
    })
}

async fn get_message(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
) -> Result<ChatMessage, ApiError> {
    let Some(row) = sqlx::query(
        r#"
        SELECT m.id,
               m.parent_message_id,
               m.active_child_message_id,
               m.active_revision_id,
               m.role,
               m.status,
               m.is_deleted,
               m.backend_id,
               m.model_name,
               m.think_mode,
               m.done_reason,
               m.error_text,
               m.started_at,
               m.completed_at,
               m.created_at,
               m.updated_at,
               r.id AS revision_id,
               r.content_text,
               r.thinking_text,
               r.source,
               r.created_at AS revision_created_at,
               (
                   SELECT COUNT(*)
                   FROM chat_message_revisions cr
                   WHERE cr.message_id = m.id
               ) AS revision_count
        FROM chat_messages m
        JOIN chats c ON c.id = m.chat_id
        LEFT JOIN chat_message_revisions r ON r.id = m.active_revision_id
        WHERE m.id = ?
          AND m.chat_id = ?
          AND c.user_id = ?
          AND c.archived_at IS NULL
        "#,
    )
    .bind(message_id)
    .bind(chat_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::not_found(
            "message_not_found",
            "Message not found",
        ));
    };

    row_to_message(row).map_err(ApiError::from)
}
