use std::{collections::HashSet, path::Path};

use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp,
    backends::service as backends_service,
    chats::{
        handlers::{
            AttachmentReference, BranchMessageRequest, CreateChatRequest, CreateMessageRequest,
            EditMessageRequest, GenerateChatRequest, RegenerateMessageRequest,
            SetActiveChildRequest, SetActiveRevisionRequest, SetActiveRootRequest,
            UpdateChatRequest,
        },
        models::{
            ChatDetail, ChatInferenceSettings, ChatMessage, ChatMessageRevision, ChatSummary,
            ChatToolPreferences,
        },
    },
    error::ApiError,
    ollama::models::{OllamaChatMessage, OllamaChatOptions, OllamaUsageStats},
    personas::service::{self as persona_service, ResolvedPersonaVersion},
    tools::service::{self as tools_service, ToolSelection},
    uploads,
};

#[derive(Debug)]
pub struct PreparedGeneration {
    pub backend_base_url: String,
    pub model_name: String,
    pub think_mode: Option<String>,
    pub inference_options: Option<OllamaChatOptions>,
    pub prompt_messages: Vec<OllamaChatMessage>,
    pub user_message: Option<ChatMessage>,
    pub assistant_message: ChatMessage,
    pub tool_selection: ToolSelection,
}

#[derive(Debug)]
struct EnabledBackend {
    id: String,
    base_url: String,
}

struct MessageModel {
    backend_id: String,
    model_name: String,
    persona_version_id: Option<String>,
}

struct ResolvedGenerationModel {
    backend: EnabledBackend,
    model_name: String,
    persona: Option<ResolvedPersonaVersion>,
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
    persona_id: Option<&'a str>,
    persona_version_id: Option<&'a str>,
    persona_name_snapshot: Option<&'a str>,
    think_mode: Option<&'a str>,
    started_at: Option<i64>,
    now: i64,
}

impl From<ChatToolPreferences> for ToolSelection {
    fn from(preferences: ChatToolPreferences) -> Self {
        Self {
            tool_use_enabled: preferences.tool_use_enabled,
            brave_web_search_enabled: preferences
                .tool_enabled(tools_service::TOOL_BRAVE_WEB_SEARCH),
            ollama_web_search_enabled: preferences
                .tool_enabled(tools_service::TOOL_OLLAMA_WEB_SEARCH),
            ollama_web_fetch_enabled: preferences
                .tool_enabled(tools_service::TOOL_OLLAMA_WEB_FETCH),
            direct_web_fetch_enabled: preferences
                .tool_enabled(tools_service::TOOL_DIRECT_WEB_FETCH),
        }
    }
}

pub async fn list_chats(pool: &SqlitePool, user_id: &str) -> Result<Vec<ChatSummary>, sqlx::Error> {
    let rows = sqlx::query(
        r#"
        SELECT c.id,
               c.title,
               c.default_backend_id,
               b.name AS backend_name,
               c.default_model_name,
               c.persona_id,
               c.persona_version_id,
               pv.display_name AS persona_name,
               c.updated_at,
               c.last_message_at,
               COUNT(m.id) AS message_count
        FROM chats c
        JOIN ollama_backends b ON b.id = c.default_backend_id
        LEFT JOIN persona_versions pv ON pv.id = c.persona_version_id
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
                persona_id: row.try_get("persona_id")?,
                persona_version_id: row.try_get("persona_version_id")?,
                persona_name: row.try_get("persona_name")?,
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
    let tool_preferences = payload.tool_preferences.unwrap_or_default();
    let persona = match normalize_optional_string(payload.persona_version_id) {
        Some(persona_version_id) => Some(
            persona_service::resolve_persona_version_for_use(pool, user_id, &persona_version_id)
                .await?,
        ),
        None => None,
    };
    let backend_id = persona
        .as_ref()
        .map(|persona| persona.base_backend_id.clone())
        .unwrap_or_else(|| payload.default_backend_id.trim().to_string());
    let model_name = persona
        .as_ref()
        .map(|persona| persona.base_model_name.clone())
        .unwrap_or_else(|| payload.default_model_name.trim().to_string());

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
    backends_service::ensure_model_enabled_for_user(pool, user_id, &backend_id, &model_name)
        .await?;

    let now = unix_timestamp();
    let chat_id = Uuid::new_v4().to_string();
    let tool_preferences_json = serialize_tool_preferences(&tool_preferences)?;
    let inference_settings =
        normalized_inference_settings(payload.inference_settings.unwrap_or_default());
    let inference_settings_json = serialize_inference_settings(&inference_settings)?;
    let system_prompt_override = payload.system_prompt_override.map(normalized_system_prompt);

    sqlx::query(
        r#"
        INSERT INTO chats (
            id,
            user_id,
            default_backend_id,
            default_model_name,
            persona_id,
            persona_version_id,
            system_prompt_override,
            tool_use_enabled,
            web_search_tool_enabled,
            web_fetch_tool_enabled,
            tool_preferences_json,
            inference_settings_json,
            title,
            chat_mode,
            created_at,
            updated_at,
            last_message_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'standard', ?, ?, ?)
        "#,
    )
    .bind(&chat_id)
    .bind(user_id)
    .bind(&backend_id)
    .bind(&model_name)
    .bind(persona.as_ref().map(|persona| persona.persona_id.as_str()))
    .bind(
        persona
            .as_ref()
            .map(|persona| persona.persona_version_id.as_str()),
    )
    .bind(system_prompt_override)
    .bind(i64::from(tool_preferences.tool_use_enabled))
    .bind(i64::from(legacy_web_search_enabled(&tool_preferences)))
    .bind(i64::from(legacy_web_fetch_enabled(&tool_preferences)))
    .bind(tool_preferences_json)
    .bind(inference_settings_json)
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
               c.persona_id,
               c.persona_version_id,
               pv.display_name AS persona_name,
               c.system_prompt_override,
               c.tool_use_enabled,
               c.web_search_tool_enabled,
               c.web_fetch_tool_enabled,
               c.tool_preferences_json,
               c.inference_settings_json,
               c.active_root_message_id,
               c.created_at,
               c.updated_at
        FROM chats c
        JOIN ollama_backends b ON b.id = c.default_backend_id
        LEFT JOIN persona_versions pv ON pv.id = c.persona_version_id
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
    let tool_preferences = payload.tool_preferences.unwrap_or(current.tool_preferences);
    let inference_settings = payload
        .inference_settings
        .map(normalized_inference_settings)
        .unwrap_or(current.inference_settings);
    let system_prompt_override = payload
        .system_prompt_override
        .map(|value| value.map(normalized_system_prompt))
        .unwrap_or(current.system_prompt_override);
    let has_base_model_update =
        payload.default_backend_id.is_some() || payload.default_model_name.is_some();
    let title = payload.title.map(normalized_title).unwrap_or(current.title);
    let persona = match normalize_optional_string(payload.persona_version_id) {
        Some(persona_version_id) => Some(
            persona_service::resolve_persona_version_for_use(pool, user_id, &persona_version_id)
                .await?,
        ),
        None => None,
    };
    let backend_id = if let Some(persona) = &persona {
        persona.base_backend_id.clone()
    } else {
        match payload.default_backend_id {
            Some(backend_id) => {
                let backend_id = backend_id.trim().to_string();
                ensure_enabled_backend(pool, &backend_id).await?;
                backend_id
            }
            None => current.default_backend_id,
        }
    };
    let model_name = if let Some(persona) = &persona {
        persona.base_model_name.clone()
    } else {
        match payload.default_model_name {
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
        }
    };
    let (persona_id, persona_version_id) = if let Some(persona) = &persona {
        (
            Some(persona.persona_id.clone()),
            Some(persona.persona_version_id.clone()),
        )
    } else if has_base_model_update {
        (None, None)
    } else {
        (current.persona_id, current.persona_version_id)
    };
    if persona.is_some() || has_base_model_update {
        ensure_enabled_backend(pool, &backend_id).await?;
        backends_service::ensure_model_enabled_for_user(pool, user_id, &backend_id, &model_name)
            .await?;
    }
    let tool_preferences_json = serialize_tool_preferences(&tool_preferences)?;
    let inference_settings_json = serialize_inference_settings(&inference_settings)?;

    sqlx::query(
        r#"
        UPDATE chats
        SET title = ?,
            default_backend_id = ?,
            default_model_name = ?,
            persona_id = ?,
            persona_version_id = ?,
            tool_use_enabled = ?,
            web_search_tool_enabled = ?,
            web_fetch_tool_enabled = ?,
            tool_preferences_json = ?,
            inference_settings_json = ?,
            system_prompt_override = ?,
            updated_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(title)
    .bind(backend_id)
    .bind(model_name)
    .bind(persona_id)
    .bind(persona_version_id)
    .bind(i64::from(tool_preferences.tool_use_enabled))
    .bind(i64::from(legacy_web_search_enabled(&tool_preferences)))
    .bind(i64::from(legacy_web_fetch_enabled(&tool_preferences)))
    .bind(tool_preferences_json)
    .bind(inference_settings_json)
    .bind(system_prompt_override)
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
               m.persona_id,
               m.persona_version_id,
               m.persona_name_snapshot,
               m.think_mode,
               m.done_reason,
               m.error_text,
               m.stats_json,
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
        let mut message = row_to_message(row)?;
        hydrate_message_revisions(pool, &mut message).await?;
        hydrate_message_attachments(pool, user_id, chat_id, &mut message).await?;
        messages.push(message);
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
    uploads_dir: &Path,
    user_id: &str,
    chat_id: &str,
    payload: GenerateChatRequest,
) -> Result<PreparedGeneration, ApiError> {
    let chat = get_chat(pool, user_id, chat_id).await?;
    let GenerateChatRequest {
        user_message,
        backend_id,
        model_name,
        persona_version_id,
        think_mode,
        inference_settings,
        tool_preferences,
        attachments,
    } = payload;
    let tool_preferences = tool_preferences.unwrap_or_else(|| chat.tool_preferences.clone());
    let inference_settings = inference_settings
        .map(normalized_inference_settings)
        .unwrap_or_else(|| chat.inference_settings.clone());
    let content_text = user_message.content_text.trim().to_string();
    if content_text.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_message",
            "Message text is required",
        ));
    }

    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let resolved = resolve_generation_model(
        pool,
        user_id,
        &chat,
        latest_model.as_ref(),
        backend_id,
        model_name,
        persona_version_id,
    )
    .await?;
    let backend = resolved.backend;
    let model_name = resolved.model_name;
    let persona = resolved.persona;
    let parent_message_id =
        active_tail_message_id(pool, chat_id, chat.active_root_message_id).await?;
    let now = unix_timestamp();
    let attachment_ids = attachment_ids(&attachments)?;
    let user_message_id = Uuid::new_v4().to_string();
    let user_revision_id = Uuid::new_v4().to_string();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(think_mode);
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
            persona_id: None,
            persona_version_id: None,
            persona_name_snapshot: None,
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
    uploads::service::attach_referenced_attachments(
        &mut tx,
        uploads_dir,
        user_id,
        chat_id,
        &user_message_id,
        &user_revision_id,
        &attachment_ids,
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
            persona_id: persona.as_ref().map(|persona| persona.persona_id.as_str()),
            persona_version_id: persona
                .as_ref()
                .map(|persona| persona.persona_version_id.as_str()),
            persona_name_snapshot: persona
                .as_ref()
                .map(|persona| persona.display_name.as_str()),
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
        SET default_backend_id = ?,
            default_model_name = ?,
            persona_id = ?,
            persona_version_id = ?,
            updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(&backend.id)
    .bind(&model_name)
    .bind(persona.as_ref().map(|persona| persona.persona_id.as_str()))
    .bind(
        persona
            .as_ref()
            .map(|persona| persona.persona_version_id.as_str()),
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let mut prompt_messages =
        active_prompt_messages(pool, uploads_dir, user_id, chat_id, &assistant_message_id).await?;
    prepend_system_prompt(
        &mut prompt_messages,
        chat.system_prompt_override.as_deref(),
        persona.as_ref(),
    );
    let user_message = get_message(pool, user_id, chat_id, &user_message_id).await?;
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        inference_options: inference_settings_to_options(&inference_settings),
        prompt_messages,
        user_message: Some(user_message),
        assistant_message,
        tool_selection: tool_preferences.into(),
    })
}

pub async fn prepare_regeneration(
    pool: &SqlitePool,
    uploads_dir: &Path,
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
    let RegenerateMessageRequest {
        backend_id,
        model_name,
        persona_version_id,
        think_mode,
        inference_settings,
        tool_preferences,
        attachments: _,
    } = payload;
    let chat = get_chat(pool, user_id, chat_id).await?;
    let tool_preferences = tool_preferences.unwrap_or_else(|| chat.tool_preferences.clone());
    let inference_settings = inference_settings
        .map(normalized_inference_settings)
        .unwrap_or_else(|| chat.inference_settings.clone());
    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let fallback_model = MessageModel {
        backend_id: target.backend_id.clone().unwrap_or_default(),
        model_name: target.model_name.clone().unwrap_or_default(),
        persona_version_id: target.persona_version_id.clone(),
    };
    let latest_model =
        if fallback_model.backend_id.is_empty() || fallback_model.model_name.is_empty() {
            latest_model.as_ref()
        } else {
            Some(&fallback_model)
        };
    let resolved = resolve_generation_model(
        pool,
        user_id,
        &chat,
        latest_model,
        backend_id,
        model_name,
        persona_version_id,
    )
    .await?;
    let backend = resolved.backend;
    let model_name = resolved.model_name;
    let persona = resolved.persona;
    let now = unix_timestamp();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(think_mode).or(target.think_mode);
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
            persona_id: persona.as_ref().map(|persona| persona.persona_id.as_str()),
            persona_version_id: persona
                .as_ref()
                .map(|persona| persona.persona_version_id.as_str()),
            persona_name_snapshot: persona
                .as_ref()
                .map(|persona| persona.display_name.as_str()),
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
        SET default_backend_id = ?,
            default_model_name = ?,
            persona_id = ?,
            persona_version_id = ?,
            updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(&backend.id)
    .bind(&model_name)
    .bind(persona.as_ref().map(|persona| persona.persona_id.as_str()))
    .bind(
        persona
            .as_ref()
            .map(|persona| persona.persona_version_id.as_str()),
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let mut prompt_messages =
        active_prompt_messages(pool, uploads_dir, user_id, chat_id, &assistant_message_id).await?;
    prepend_system_prompt(
        &mut prompt_messages,
        chat.system_prompt_override.as_deref(),
        persona.as_ref(),
    );
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        inference_options: inference_settings_to_options(&inference_settings),
        prompt_messages,
        user_message: None,
        assistant_message,
        tool_selection: tool_preferences.into(),
    })
}

pub async fn prepare_branch_generation(
    pool: &SqlitePool,
    uploads_dir: &Path,
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

    let BranchMessageRequest {
        content_text: _,
        backend_id,
        model_name,
        persona_version_id,
        think_mode,
        inference_settings,
        tool_preferences,
        attachments,
    } = payload;
    let tool_preferences = tool_preferences.unwrap_or_else(|| chat.tool_preferences.clone());
    let inference_settings = inference_settings
        .map(normalized_inference_settings)
        .unwrap_or_else(|| chat.inference_settings.clone());
    let latest_model = latest_assistant_model(pool, user_id, chat_id).await?;
    let resolved = resolve_generation_model(
        pool,
        user_id,
        &chat,
        latest_model.as_ref(),
        backend_id,
        model_name,
        persona_version_id,
    )
    .await?;
    let backend = resolved.backend;
    let model_name = resolved.model_name;
    let persona = resolved.persona;
    let parent_message_id = target.parent_message_id.clone();
    let now = unix_timestamp();
    let attachment_ids = attachment_ids(&attachments)?;
    let user_message_id = Uuid::new_v4().to_string();
    let user_revision_id = Uuid::new_v4().to_string();
    let assistant_message_id = Uuid::new_v4().to_string();
    let assistant_revision_id = Uuid::new_v4().to_string();
    let think_mode = normalized_think_mode(think_mode);
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
            persona_id: None,
            persona_version_id: None,
            persona_name_snapshot: None,
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
    uploads::service::attach_referenced_attachments(
        &mut tx,
        uploads_dir,
        user_id,
        chat_id,
        &user_message_id,
        &user_revision_id,
        &attachment_ids,
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
            persona_id: persona.as_ref().map(|persona| persona.persona_id.as_str()),
            persona_version_id: persona
                .as_ref()
                .map(|persona| persona.persona_version_id.as_str()),
            persona_name_snapshot: persona
                .as_ref()
                .map(|persona| persona.display_name.as_str()),
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
        SET default_backend_id = ?,
            default_model_name = ?,
            persona_id = ?,
            persona_version_id = ?,
            updated_at = ?,
            last_message_at = ?
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(&backend.id)
    .bind(&model_name)
    .bind(persona.as_ref().map(|persona| persona.persona_id.as_str()))
    .bind(
        persona
            .as_ref()
            .map(|persona| persona.persona_version_id.as_str()),
    )
    .bind(now)
    .bind(now)
    .bind(chat_id)
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;

    let mut prompt_messages =
        active_prompt_messages(pool, uploads_dir, user_id, chat_id, &assistant_message_id).await?;
    prepend_system_prompt(
        &mut prompt_messages,
        chat.system_prompt_override.as_deref(),
        persona.as_ref(),
    );
    let user_message = get_message(pool, user_id, chat_id, &user_message_id).await?;
    let assistant_message = get_message(pool, user_id, chat_id, &assistant_message_id).await?;

    Ok(PreparedGeneration {
        backend_base_url: backend.base_url,
        model_name,
        think_mode,
        inference_options: inference_settings_to_options(&inference_settings),
        prompt_messages,
        user_message: Some(user_message),
        assistant_message,
        tool_selection: tool_preferences.into(),
    })
}

pub async fn edit_message(
    pool: &SqlitePool,
    uploads_dir: &Path,
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
    let attachment_ids = attachment_ids(&payload.attachments)?;
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
    uploads::service::attach_referenced_attachments(
        &mut tx,
        uploads_dir,
        user_id,
        chat_id,
        message_id,
        &revision_id,
        &attachment_ids,
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

pub async fn select_active_revision(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    payload: SetActiveRevisionRequest,
) -> Result<ChatMessage, ApiError> {
    ensure_message_owner(pool, user_id, chat_id, message_id).await?;
    let active_revision_id = payload.active_revision_id.trim().to_string();
    if active_revision_id.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_revision",
            "Active revision is required",
        ));
    }

    let revision_exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM chat_message_revisions
            WHERE id = ?
              AND message_id = ?
        )
        "#,
    )
    .bind(&active_revision_id)
    .bind(message_id)
    .fetch_one(pool)
    .await?;

    if revision_exists == 0 {
        return Err(ApiError::bad_request(
            "invalid_revision",
            "Revision not found for this message",
        ));
    }

    let now = unix_timestamp();
    let mut tx = pool.begin().await?;
    sqlx::query(
        r#"
        UPDATE chat_messages
        SET active_revision_id = ?,
            updated_at = ?
        WHERE id = ?
          AND chat_id = ?
        "#,
    )
    .bind(&active_revision_id)
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

pub async fn finish_generation(
    pool: &SqlitePool,
    assistant_message_id: &str,
    content_text: &str,
    thinking_text: &str,
    done_reason: Option<&str>,
    stats: Option<&OllamaUsageStats>,
) -> Result<(), ApiError> {
    update_generation_message(
        pool,
        assistant_message_id,
        content_text,
        thinking_text,
        "complete",
        done_reason,
        None,
        stats,
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
        None,
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
        SELECT m.backend_id,
               m.model_name,
               m.persona_version_id
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
        persona_version_id: row.try_get("persona_version_id")?,
    }))
}

async fn resolve_generation_model(
    pool: &SqlitePool,
    user_id: &str,
    chat: &ChatDetail,
    latest_model: Option<&MessageModel>,
    backend_id: Option<String>,
    model_name: Option<String>,
    persona_version_id: Option<String>,
) -> Result<ResolvedGenerationModel, ApiError> {
    let explicit_backend_id = normalize_optional_string(backend_id);
    let explicit_model_name = normalize_optional_string(model_name);
    let explicit_persona_version_id = normalize_optional_string(persona_version_id);

    if let Some(persona_version_id) = explicit_persona_version_id {
        return resolve_persona_generation_model(pool, user_id, &persona_version_id).await;
    }

    if explicit_backend_id.is_none() && explicit_model_name.is_none() {
        if let Some(persona_version_id) = latest_model
            .and_then(|model| model.persona_version_id.clone())
            .or_else(|| chat.persona_version_id.clone())
        {
            return resolve_persona_generation_model(pool, user_id, &persona_version_id).await;
        }
    }

    let backend_id = explicit_backend_id
        .or_else(|| latest_model.map(|model| model.backend_id.clone()))
        .unwrap_or_else(|| chat.default_backend_id.clone());
    let model_name = explicit_model_name
        .or_else(|| latest_model.map(|model| model.model_name.clone()))
        .unwrap_or_else(|| chat.default_model_name.clone());

    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "A model is required",
        ));
    }

    let backend = enabled_backend(pool, &backend_id).await?;
    backends_service::ensure_model_enabled_for_user(pool, user_id, &backend.id, &model_name)
        .await?;

    Ok(ResolvedGenerationModel {
        backend,
        model_name,
        persona: None,
    })
}

async fn resolve_persona_generation_model(
    pool: &SqlitePool,
    user_id: &str,
    persona_version_id: &str,
) -> Result<ResolvedGenerationModel, ApiError> {
    let persona =
        persona_service::resolve_persona_version_for_use(pool, user_id, persona_version_id).await?;
    let backend = enabled_backend(pool, &persona.base_backend_id).await?;
    let model_name = persona.base_model_name.clone();
    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "Persona base model is required",
        ));
    }
    backends_service::ensure_model_enabled_for_user(pool, user_id, &backend.id, &model_name)
        .await?;

    Ok(ResolvedGenerationModel {
        backend,
        model_name,
        persona: Some(persona),
    })
}

fn prepend_system_prompt(
    messages: &mut Vec<OllamaChatMessage>,
    system_prompt_override: Option<&str>,
    persona: Option<&ResolvedPersonaVersion>,
) {
    let system_prompt = system_prompt_override
        .or_else(|| persona.map(|persona| persona.system_prompt.as_str()))
        .unwrap_or("")
        .trim();
    if system_prompt.is_empty() {
        return;
    }

    messages.insert(
        0,
        OllamaChatMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
            thinking: None,
            images: None,
            tool_name: None,
            tool_calls: None,
        },
    );
}

fn normalized_title(title: String) -> String {
    let title = title.trim();
    if title.is_empty() {
        "New Chat".to_string()
    } else {
        title.to_string()
    }
}

fn normalized_system_prompt(value: String) -> String {
    value.trim().to_string()
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn attachment_ids(attachments: &[AttachmentReference]) -> Result<Vec<String>, ApiError> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();

    for attachment in attachments {
        let id = attachment.id.trim();
        if id.is_empty() {
            return Err(ApiError::bad_request(
                "invalid_attachment",
                "Attachment id is required",
            ));
        }

        if seen.insert(id.to_string()) {
            ids.push(id.to_string());
        }
    }

    Ok(ids)
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
            persona_id,
            persona_version_id,
            persona_name_snapshot,
            think_mode,
            started_at,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    .bind(params.persona_id)
    .bind(params.persona_version_id)
    .bind(params.persona_name_snapshot)
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
    uploads_dir: &Path,
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
                   r.id AS revision_id,
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
        let revision_id: Option<String> = row.try_get("revision_id")?;
        let mut content: String = row
            .try_get::<Option<String>, _>("content_text")?
            .unwrap_or_default();
        let thinking: Option<String> = row.try_get("thinking_text")?;

        if !is_deleted {
            let mut images = Vec::new();
            if let Some(revision_id) = &revision_id {
                let (attachment_text, attachment_images) =
                    uploads::service::prompt_attachment_payload(
                        pool,
                        uploads_dir,
                        user_id,
                        chat_id,
                        &message_id,
                        revision_id,
                    )
                    .await?;
                content.push_str(&attachment_text);
                images = attachment_images;
            }

            if !content.trim().is_empty() || !images.is_empty() {
                messages.push(OllamaChatMessage {
                    role,
                    content,
                    thinking: thinking.filter(|thinking| !thinking.trim().is_empty()),
                    images: (!images.is_empty()).then_some(images),
                    tool_name: None,
                    tool_calls: None,
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
    stats: Option<&OllamaUsageStats>,
) -> Result<(), ApiError> {
    let now = unix_timestamp();
    let stats_json = stats
        .and_then(|stats| serde_json::to_string(stats).ok())
        .filter(|stats| !stats.is_empty());
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
            stats_json = ?,
            completed_at = ?,
            updated_at = ?
        WHERE id = ?
        "#,
    )
    .bind(status)
    .bind(done_reason)
    .bind(error_text)
    .bind(stats_json)
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
    let tool_preferences = row_to_tool_preferences(&row)?;
    let inference_settings = row_to_inference_settings(&row)?;
    Ok(ChatDetail {
        id: row.try_get("id")?,
        title: row.try_get("title")?,
        default_backend_id: row.try_get("default_backend_id")?,
        backend_name: row.try_get("backend_name")?,
        default_model_name: row.try_get("default_model_name")?,
        persona_id: row.try_get("persona_id")?,
        persona_version_id: row.try_get("persona_version_id")?,
        persona_name: row.try_get("persona_name")?,
        system_prompt_override: row.try_get("system_prompt_override")?,
        tool_preferences,
        inference_settings,
        active_root_message_id: row.try_get("active_root_message_id")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn serialize_tool_preferences(preferences: &ChatToolPreferences) -> Result<String, ApiError> {
    serde_json::to_string(preferences)
        .map_err(|_| ApiError::internal("Failed to serialize chat tool preferences"))
}

fn serialize_inference_settings(settings: &ChatInferenceSettings) -> Result<String, ApiError> {
    serde_json::to_string(settings)
        .map_err(|_| ApiError::internal("Failed to serialize chat inference settings"))
}

fn row_to_inference_settings(
    row: &sqlx::sqlite::SqliteRow,
) -> Result<ChatInferenceSettings, sqlx::Error> {
    if let Some(json) = row.try_get::<Option<String>, _>("inference_settings_json")? {
        if let Ok(settings) = serde_json::from_str::<ChatInferenceSettings>(&json) {
            return Ok(normalized_inference_settings(settings));
        }
    }

    Ok(ChatInferenceSettings::default())
}

fn row_to_tool_preferences(
    row: &sqlx::sqlite::SqliteRow,
) -> Result<ChatToolPreferences, sqlx::Error> {
    if let Some(json) = row.try_get::<Option<String>, _>("tool_preferences_json")? {
        if let Ok(preferences) = serde_json::from_str::<ChatToolPreferences>(&json) {
            return Ok(preferences);
        }
    }

    let mut preferences = ChatToolPreferences {
        tool_use_enabled: row.try_get::<i64, _>("tool_use_enabled")? != 0,
        ..ChatToolPreferences::default()
    };
    let search_enabled = row.try_get::<i64, _>("web_search_tool_enabled")? != 0;
    let fetch_enabled = row.try_get::<i64, _>("web_fetch_tool_enabled")? != 0;
    preferences.tools.insert(
        tools_service::TOOL_BRAVE_WEB_SEARCH.to_string(),
        search_enabled,
    );
    preferences.tools.insert(
        tools_service::TOOL_OLLAMA_WEB_SEARCH.to_string(),
        search_enabled,
    );
    preferences.tools.insert(
        tools_service::TOOL_OLLAMA_WEB_FETCH.to_string(),
        fetch_enabled,
    );
    preferences.tools.insert(
        tools_service::TOOL_DIRECT_WEB_FETCH.to_string(),
        fetch_enabled,
    );

    Ok(preferences)
}

fn normalized_inference_settings(settings: ChatInferenceSettings) -> ChatInferenceSettings {
    ChatInferenceSettings {
        temperature: settings
            .temperature
            .and_then(|value| clamp_f64(value, 0.0, 2.0)),
        top_k: settings.top_k.map(|value| value.clamp(1, 1_000)),
        top_p: settings.top_p.and_then(|value| clamp_f64(value, 0.01, 1.0)),
        min_p: settings.min_p.and_then(|value| clamp_f64(value, 0.0, 1.0)),
        repeat_penalty: settings
            .repeat_penalty
            .and_then(|value| clamp_f64(value, 0.5, 2.0)),
        repeat_last_n: settings.repeat_last_n.map(|value| value.clamp(-1, 262_144)),
        presence_penalty: settings
            .presence_penalty
            .and_then(|value| clamp_f64(value, -2.0, 2.0)),
        frequency_penalty: settings
            .frequency_penalty
            .and_then(|value| clamp_f64(value, -2.0, 2.0)),
        num_ctx: settings.num_ctx.map(|value| value.clamp(512, 262_144)),
        num_predict: settings.num_predict.map(|value| value.clamp(1, 131_072)),
        num_gpu: settings.num_gpu.map(|value| value.clamp(0, 10_000)),
        num_thread: settings.num_thread.map(|value| value.clamp(1, 1_024)),
        seed: settings.seed,
    }
}

fn inference_settings_to_options(settings: &ChatInferenceSettings) -> Option<OllamaChatOptions> {
    let options = OllamaChatOptions {
        temperature: settings.temperature,
        top_k: settings.top_k,
        top_p: settings.top_p,
        min_p: settings.min_p,
        repeat_penalty: settings.repeat_penalty,
        repeat_last_n: settings.repeat_last_n,
        presence_penalty: settings.presence_penalty,
        frequency_penalty: settings.frequency_penalty,
        num_ctx: settings.num_ctx,
        num_predict: settings.num_predict,
        num_gpu: settings.num_gpu,
        num_thread: settings.num_thread,
        seed: settings.seed,
    };

    options.has_any().then_some(options)
}

fn clamp_f64(value: f64, min: f64, max: f64) -> Option<f64> {
    value.is_finite().then_some(value.clamp(min, max))
}

fn legacy_web_search_enabled(preferences: &ChatToolPreferences) -> bool {
    preferences.tool_enabled(tools_service::TOOL_BRAVE_WEB_SEARCH)
        || preferences.tool_enabled(tools_service::TOOL_OLLAMA_WEB_SEARCH)
}

fn legacy_web_fetch_enabled(preferences: &ChatToolPreferences) -> bool {
    preferences.tool_enabled(tools_service::TOOL_OLLAMA_WEB_FETCH)
        || preferences.tool_enabled(tools_service::TOOL_DIRECT_WEB_FETCH)
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

    let stats_json: Option<String> = row.try_get("stats_json")?;
    let stats = stats_json
        .as_deref()
        .and_then(|stats| serde_json::from_str::<OllamaUsageStats>(stats).ok());

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
        persona_id: row.try_get("persona_id")?,
        persona_version_id: row.try_get("persona_version_id")?,
        persona_name_snapshot: row.try_get("persona_name_snapshot")?,
        think_mode: row.try_get("think_mode")?,
        done_reason: row.try_get("done_reason")?,
        error_text: row.try_get("error_text")?,
        stats,
        started_at: row.try_get("started_at")?,
        completed_at: row.try_get("completed_at")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        revisions: active_revision.iter().cloned().collect(),
        revision_count: row.try_get("revision_count")?,
        active_revision,
        attachments: Vec::new(),
    })
}

async fn hydrate_message_revisions(
    pool: &SqlitePool,
    message: &mut ChatMessage,
) -> Result<(), ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT id,
               content_text,
               thinking_text,
               source,
               created_at
        FROM chat_message_revisions
        WHERE message_id = ?
        ORDER BY created_at ASC, id ASC
        "#,
    )
    .bind(&message.id)
    .fetch_all(pool)
    .await?;

    let revisions = rows
        .into_iter()
        .map(|row| {
            Ok(ChatMessageRevision {
                id: row.try_get("id")?,
                content_text: row.try_get("content_text")?,
                thinking_text: row.try_get("thinking_text")?,
                source: row.try_get("source")?,
                created_at: row.try_get("created_at")?,
            })
        })
        .collect::<Result<Vec<_>, sqlx::Error>>()?;

    if !revisions.is_empty() {
        message.revision_count = revisions.len() as i64;
        if let Some(active_revision_id) = &message.active_revision_id {
            message.active_revision = revisions
                .iter()
                .find(|revision| revision.id == *active_revision_id)
                .cloned();
        }
        message.revisions = revisions;
    }

    Ok(())
}

async fn hydrate_message_attachments(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message: &mut ChatMessage,
) -> Result<(), ApiError> {
    let Some(active_revision_id) = &message.active_revision_id else {
        message.attachments = Vec::new();
        return Ok(());
    };

    message.attachments = uploads::service::list_revision_attachments(
        pool,
        user_id,
        chat_id,
        &message.id,
        active_revision_id,
    )
    .await?;

    Ok(())
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
               m.persona_id,
               m.persona_version_id,
               m.persona_name_snapshot,
               m.think_mode,
               m.done_reason,
               m.error_text,
               m.stats_json,
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

    let mut message = row_to_message(row)?;
    hydrate_message_revisions(pool, &mut message).await?;
    hydrate_message_attachments(pool, user_id, chat_id, &mut message).await?;

    Ok(message)
}
