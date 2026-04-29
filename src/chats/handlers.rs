use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Path, State},
    http::{HeaderValue, StatusCode, header},
    response::Response,
};
use axum_extra::extract::CookieJar;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, convert::Infallible, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    app_state::{AppState, GenerationProgress},
    auth,
    chats::{
        models::{ChatDetail, ChatMessage, ChatMessageRevision, ChatSummary},
        service,
    },
    error::ApiError,
    ollama::{
        self,
        models::{OllamaChatChunk, OllamaChatMessage, OllamaChatRequest, OllamaThink},
    },
};

type GenerationProgressMap = Arc<tokio::sync::Mutex<HashMap<String, GenerationProgress>>>;

#[derive(Debug, Serialize)]
pub struct ListChatsResponse {
    pub chats: Vec<ChatSummary>,
}

#[derive(Debug, Deserialize)]
pub struct CreateChatRequest {
    pub title: String,
    pub default_backend_id: String,
    pub default_model_name: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateChatRequest {
    pub title: Option<String>,
    pub default_backend_id: Option<String>,
    pub default_model_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub chat: ChatDetail,
}

#[derive(Debug, Serialize)]
pub struct DeleteChatResponse {
    pub ok: bool,
}

#[derive(Debug, Serialize)]
pub struct ListMessagesResponse {
    pub active_root_message_id: Option<String>,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
pub struct CreateMessageRequest {
    pub content_text: String,
    pub parent_message_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: ChatMessage,
}

#[derive(Debug, Deserialize)]
pub struct EditMessageRequest {
    pub content_text: String,
    #[serde(default)]
    pub attachments: Vec<AttachmentReference>,
}

#[derive(Debug, Deserialize)]
pub struct SetActiveChildRequest {
    pub active_child_message_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SetActiveRootRequest {
    pub active_root_message_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SetActiveRevisionRequest {
    pub active_revision_id: String,
}

#[derive(Debug, Deserialize)]
pub struct GenerateChatRequest {
    pub user_message: GenerateUserMessageRequest,
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub think_mode: Option<String>,
    #[serde(default)]
    pub attachments: Vec<AttachmentReference>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateUserMessageRequest {
    pub content_text: String,
}

#[derive(Debug, Deserialize)]
pub struct AttachmentReference {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct StopGenerationResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct RegenerateMessageRequest {
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub think_mode: Option<String>,
    #[serde(default)]
    pub attachments: Vec<AttachmentReference>,
}

#[derive(Debug, Deserialize)]
pub struct BranchMessageRequest {
    pub content_text: String,
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub think_mode: Option<String>,
    #[serde(default)]
    pub attachments: Vec<AttachmentReference>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GenerateEvent {
    MessageStart {
        user_message: Option<ChatMessage>,
        assistant_message: ChatMessage,
    },
    ThinkingDelta {
        assistant_message_id: String,
        delta: String,
    },
    ContentDelta {
        assistant_message_id: String,
        delta: String,
    },
    MessageDone {
        assistant_message_id: String,
        done_reason: Option<String>,
    },
    ChatTitle {
        chat_id: String,
        title: String,
    },
    MessageStopped {
        assistant_message_id: String,
    },
    Error {
        assistant_message_id: Option<String>,
        message: String,
    },
}

pub async fn list_chats(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<ListChatsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let chats = service::list_chats(&state.db, &user.id).await?;

    Ok(Json(ListChatsResponse { chats }))
}

pub async fn create_chat(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let chat = service::create_chat(&state.db, &user.id, payload).await?;

    Ok(Json(ChatResponse { chat }))
}

pub async fn get_chat(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
) -> Result<Json<ChatResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let chat = service::get_chat(&state.db, &user.id, &chat_id).await?;

    Ok(Json(ChatResponse { chat }))
}

pub async fn update_chat(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
    Json(payload): Json<UpdateChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let chat = service::update_chat(&state.db, &user.id, &chat_id, payload).await?;

    Ok(Json(ChatResponse { chat }))
}

pub async fn delete_chat(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
) -> Result<Json<DeleteChatResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_chat(&state.db, &user.id, &chat_id).await?;

    Ok(Json(DeleteChatResponse { ok: true }))
}

pub async fn list_messages(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
) -> Result<Json<ListMessagesResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let (active_root_message_id, mut messages) =
        service::list_messages(&state.db, &user.id, &chat_id).await?;
    let progress = state.generation_progress.lock().await;
    overlay_generation_progress(&mut messages, &user.id, &chat_id, &progress);

    Ok(Json(ListMessagesResponse {
        active_root_message_id,
        messages,
    }))
}

pub async fn create_message(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
    Json(payload): Json<CreateMessageRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let message = service::create_user_message(&state.db, &user.id, &chat_id, payload).await?;

    Ok(Json(MessageResponse { message }))
}

pub async fn edit_message(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
    Json(payload): Json<EditMessageRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let message = service::edit_message(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &chat_id,
        &message_id,
        payload,
    )
    .await?;

    Ok(Json(MessageResponse { message }))
}

pub async fn delete_message(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
) -> Result<Json<MessageResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let message = service::delete_message(&state.db, &user.id, &chat_id, &message_id).await?;

    Ok(Json(MessageResponse { message }))
}

pub async fn set_active_child(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
    Json(payload): Json<SetActiveChildRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let message =
        service::select_active_child(&state.db, &user.id, &chat_id, &message_id, payload).await?;

    Ok(Json(MessageResponse { message }))
}

pub async fn set_active_revision(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
    Json(payload): Json<SetActiveRevisionRequest>,
) -> Result<Json<MessageResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let message =
        service::select_active_revision(&state.db, &user.id, &chat_id, &message_id, payload)
            .await?;

    Ok(Json(MessageResponse { message }))
}

pub async fn set_active_root(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
    Json(payload): Json<SetActiveRootRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let chat = service::select_active_root(&state.db, &user.id, &chat_id, payload).await?;

    Ok(Json(ChatResponse { chat }))
}

pub async fn generate_chat(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
    Json(payload): Json<GenerateChatRequest>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;

    let prepared = service::prepare_generation(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &chat_id,
        payload,
    )
    .await?;
    Ok(start_generation_stream(state, user.id, chat_id, prepared).await)
}

pub async fn branch_message(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
    Json(payload): Json<BranchMessageRequest>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;

    let prepared = service::prepare_branch_generation(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &chat_id,
        &message_id,
        payload,
    )
    .await?;
    Ok(start_generation_stream(state, user.id, chat_id, prepared).await)
}

pub async fn regenerate_message(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
    Json(payload): Json<RegenerateMessageRequest>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    if !payload.attachments.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_attachments",
            "Attachments can only be added to user messages",
        ));
    }

    let prepared = service::prepare_regeneration(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &chat_id,
        &message_id,
        payload,
    )
    .await?;
    Ok(start_generation_stream(state, user.id, chat_id, prepared).await)
}

pub async fn stop_generation(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat_id, message_id)): Path<(String, String)>,
) -> Result<Json<StopGenerationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    if let Some(cancellation) = state
        .generation_cancellations
        .lock()
        .await
        .remove(&message_id)
    {
        cancellation.cancel();
    }

    service::stop_generation_by_id(&state.db, &user.id, &chat_id, &message_id).await?;

    Ok(Json(StopGenerationResponse { ok: true }))
}

async fn stream_generation(
    tx: mpsc::Sender<Result<Bytes, Infallible>>,
    db: sqlx::SqlitePool,
    client: reqwest::Client,
    progress: GenerationProgressMap,
    user_id: String,
    chat_id: String,
    prepared: service::PreparedGeneration,
    cancellation: tokio_util::sync::CancellationToken,
) {
    let assistant_message_id = prepared.assistant_message.id.clone();
    let title_backend_base_url = prepared.backend_base_url.clone();
    let title_model_name = prepared.model_name.clone();
    let title_prompt_messages = prepared.prompt_messages.clone();
    let mut content_text = String::new();
    let mut thinking_text = String::new();
    let mut done_reason = None;
    set_generation_progress(
        &progress,
        &assistant_message_id,
        &user_id,
        &chat_id,
        &content_text,
        &thinking_text,
    )
    .await;

    if !send_event(
        &tx,
        &GenerateEvent::MessageStart {
            user_message: prepared.user_message,
            assistant_message: prepared.assistant_message,
        },
    )
    .await
    {
        let _ = service::stop_generation(
            &db,
            &user_id,
            &chat_id,
            &assistant_message_id,
            &content_text,
            &thinking_text,
        )
        .await;
        clear_generation_progress(&progress, &assistant_message_id).await;
        return;
    }

    let request = OllamaChatRequest {
        model: prepared.model_name,
        messages: prepared.prompt_messages,
        stream: true,
        think: prepared.think_mode.as_deref().and_then(think_from_mode),
    };

    let response =
        match ollama::client::chat_stream(&client, &prepared.backend_base_url, &request).await {
            Ok(response) => response,
            Err(error) => {
                let message = format!("Ollama request failed: {error}");
                let _ = service::fail_generation(
                    &db,
                    &assistant_message_id,
                    &content_text,
                    &thinking_text,
                    &message,
                )
                .await;
                let _ = send_event(
                    &tx,
                    &GenerateEvent::Error {
                        assistant_message_id: Some(assistant_message_id.clone()),
                        message,
                    },
                )
                .await;
                clear_generation_progress(&progress, &assistant_message_id).await;
                return;
            }
        };

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    loop {
        tokio::select! {
            _ = cancellation.cancelled() => {
                let _ = service::stop_generation(
                    &db,
                    &user_id,
                    &chat_id,
                    &assistant_message_id,
                    &content_text,
                    &thinking_text,
                )
                .await;
                let _ = send_event(
                    &tx,
                    &GenerateEvent::MessageStopped {
                        assistant_message_id: assistant_message_id.clone(),
                    },
                )
                .await;
                clear_generation_progress(&progress, &assistant_message_id).await;
                return;
            }
            next = stream.next() => {
                match next {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(line_end) = buffer.find('\n') {
                            let line = buffer[..line_end].trim().to_string();
                            buffer.drain(..=line_end);
                            if line.is_empty() {
                                continue;
                            }
                            match handle_ollama_line(
                                &tx,
                                &progress,
                                &assistant_message_id,
                                &user_id,
                                &chat_id,
                                &line,
                                &mut content_text,
                                &mut thinking_text,
                                &mut done_reason,
                            )
                            .await
                            {
                                Ok(()) => {}
                                Err(message) => {
                                    let _ = service::fail_generation(
                                        &db,
                                        &assistant_message_id,
                                        &content_text,
                                        &thinking_text,
                                        &message,
                                    )
                                    .await;
                                    clear_generation_progress(&progress, &assistant_message_id)
                                        .await;
                                    return;
                                }
                            }
                        }
                    }
                    Some(Err(error)) => {
                        let message = format!("Ollama stream failed: {error}");
                        let _ = service::fail_generation(
                            &db,
                            &assistant_message_id,
                            &content_text,
                            &thinking_text,
                            &message,
                        )
                        .await;
                        let _ = send_event(
                            &tx,
                            &GenerateEvent::Error {
                                assistant_message_id: Some(assistant_message_id.clone()),
                                message,
                            },
                        )
                        .await;
                        clear_generation_progress(&progress, &assistant_message_id).await;
                        return;
                    }
                    None => break,
                }
            }
        }
    }

    let line = buffer.trim().to_string();
    if !line.is_empty() {
        if let Err(message) = handle_ollama_line(
            &tx,
            &progress,
            &assistant_message_id,
            &user_id,
            &chat_id,
            &line,
            &mut content_text,
            &mut thinking_text,
            &mut done_reason,
        )
        .await
        {
            let _ = service::fail_generation(
                &db,
                &assistant_message_id,
                &content_text,
                &thinking_text,
                &message,
            )
            .await;
            clear_generation_progress(&progress, &assistant_message_id).await;
            return;
        }
    }

    let _ = service::finish_generation(
        &db,
        &assistant_message_id,
        &content_text,
        &thinking_text,
        done_reason.as_deref(),
    )
    .await;
    let _ = send_event(
        &tx,
        &GenerateEvent::MessageDone {
            assistant_message_id: assistant_message_id.clone(),
            done_reason,
        },
    )
    .await;
    clear_generation_progress(&progress, &assistant_message_id).await;

    if let Some(title) = maybe_generate_chat_title(
        &db,
        &client,
        &user_id,
        &chat_id,
        &title_backend_base_url,
        &title_model_name,
        &title_prompt_messages,
        &content_text,
    )
    .await
    {
        let _ = send_event(&tx, &GenerateEvent::ChatTitle { chat_id, title }).await;
    }
}

fn overlay_generation_progress(
    messages: &mut [ChatMessage],
    user_id: &str,
    chat_id: &str,
    progress: &HashMap<String, GenerationProgress>,
) {
    for message in messages {
        let Some(progress) = progress.get(&message.id) else {
            continue;
        };

        if progress.user_id != user_id || progress.chat_id != chat_id {
            continue;
        }

        message.status = "streaming".to_string();
        message.done_reason = None;
        message.error_text = None;
        message.completed_at = None;

        let active_revision_id = message
            .active_revision_id
            .clone()
            .unwrap_or_else(|| format!("{}-active", message.id));
        let mut revision = message
            .active_revision
            .clone()
            .unwrap_or_else(|| ChatMessageRevision {
                id: active_revision_id.clone(),
                content_text: String::new(),
                thinking_text: String::new(),
                source: "original".to_string(),
                created_at: message.created_at,
            });
        revision.content_text = progress.content_text.clone();
        revision.thinking_text = progress.thinking_text.clone();

        message.active_revision = Some(revision.clone());
        if let Some(existing) = message
            .revisions
            .iter_mut()
            .find(|candidate| candidate.id == revision.id)
        {
            *existing = revision;
        } else {
            message.revisions.push(revision);
            message.revision_count += 1;
        }
    }
}

async fn set_generation_progress(
    progress: &GenerationProgressMap,
    assistant_message_id: &str,
    user_id: &str,
    chat_id: &str,
    content_text: &str,
    thinking_text: &str,
) {
    progress.lock().await.insert(
        assistant_message_id.to_string(),
        GenerationProgress {
            user_id: user_id.to_string(),
            chat_id: chat_id.to_string(),
            content_text: content_text.to_string(),
            thinking_text: thinking_text.to_string(),
        },
    );
}

async fn clear_generation_progress(progress: &GenerationProgressMap, assistant_message_id: &str) {
    progress.lock().await.remove(assistant_message_id);
}

async fn start_generation_stream(
    state: AppState,
    user_id: String,
    chat_id: String,
    prepared: service::PreparedGeneration,
) -> Response {
    let assistant_message_id = prepared.assistant_message.id.clone();
    let cancellation = tokio_util::sync::CancellationToken::new();

    state
        .generation_cancellations
        .lock()
        .await
        .insert(assistant_message_id.clone(), cancellation.clone());

    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(32);
    let db = state.db.clone();
    let client = state.http_client.clone();
    let cancellations = state.generation_cancellations.clone();
    let progress = state.generation_progress.clone();

    tokio::spawn(async move {
        stream_generation(
            tx,
            db,
            client,
            progress,
            user_id,
            chat_id,
            prepared,
            cancellation,
        )
        .await;
        cancellations.lock().await.remove(&assistant_message_id);
    });

    let stream = ReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));

    response
}

async fn maybe_generate_chat_title(
    db: &sqlx::SqlitePool,
    client: &reqwest::Client,
    user_id: &str,
    chat_id: &str,
    backend_base_url: &str,
    model_name: &str,
    prompt_messages: &[OllamaChatMessage],
    assistant_text: &str,
) -> Option<String> {
    let chat = service::get_chat(db, user_id, chat_id).await.ok()?;
    if chat.title != "New Chat" || prompt_messages.len() != 1 {
        return None;
    }

    let user_message = prompt_messages.first()?;
    if user_message.role != "user" || user_message.content.trim().is_empty() {
        return None;
    }

    let fallback = fallback_title_from_prompt(&user_message.content);
    let generated_title = request_generated_title(
        client,
        backend_base_url,
        model_name,
        &user_message.content,
        assistant_text,
    )
    .await
    .and_then(|title| validated_title(&title))
    .unwrap_or(fallback);

    let updated = service::update_chat(
        db,
        user_id,
        chat_id,
        UpdateChatRequest {
            title: Some(generated_title),
            default_backend_id: None,
            default_model_name: None,
        },
    )
    .await
    .ok()?;

    Some(updated.title)
}

async fn request_generated_title(
    client: &reqwest::Client,
    backend_base_url: &str,
    model_name: &str,
    user_message: &str,
    assistant_message: &str,
) -> Option<String> {
    let transcript = format!(
        "User message:\n{}\n\nAssistant response:\n{}",
        user_message.trim(),
        assistant_message.trim()
    );
    let request = OllamaChatRequest {
        model: model_name.to_string(),
        stream: false,
        think: Some(OllamaThink::Bool(false)),
        messages: vec![
            OllamaChatMessage {
                role: "system".to_string(),
                content: "Create a concise chat title. Return only the title, no preface, no quotes, no explanation. Use 2 to 5 words when possible. Emojis are allowed if useful.".to_string(),
                thinking: None,
                images: None,
            },
            OllamaChatMessage {
                role: "user".to_string(),
                content: transcript,
                thinking: None,
                images: None,
            },
        ],
    };

    let response = ollama::client::chat_once(client, backend_base_url, &request)
        .await
        .ok()?;
    response.message.map(|message| message.content)
}

fn validated_title(raw_title: &str) -> Option<String> {
    let mut title = raw_title
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(raw_title)
        .trim()
        .trim_matches(['"', '\'', '`', '*', '#', ' '])
        .to_string();

    if let Some((_, after_colon)) = title.rsplit_once(':') {
        title = after_colon.trim().to_string();
    }

    if let Some((before_sentence, _)) = title.split_once(['.', '?']) {
        title = before_sentence.trim().to_string();
    }

    title = title
        .trim()
        .trim_matches(['"', '\'', '`', '*', '#', '.', ':', ';', ' '])
        .to_string();

    if title.is_empty()
        || title.len() > 64
        || title.split_whitespace().count() > 7
        || looks_like_explanation(&title)
    {
        return None;
    }

    Some(title)
}

fn looks_like_explanation(title: &str) -> bool {
    let lower = title.to_ascii_lowercase();
    ["okay", "sure", "here", "title", "this chat"]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn fallback_title_from_prompt(prompt: &str) -> String {
    let title = prompt
        .split_whitespace()
        .take(5)
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(['"', '\'', '`', '*', '#', '.', ':', ';', ',', ' '])
        .to_string();

    if title.is_empty() {
        "New Chat".to_string()
    } else if title.len() > 64 {
        title.chars().take(64).collect()
    } else {
        title
    }
}

async fn handle_ollama_line(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    progress: &GenerationProgressMap,
    assistant_message_id: &str,
    user_id: &str,
    chat_id: &str,
    line: &str,
    content_text: &mut String,
    thinking_text: &mut String,
    done_reason: &mut Option<String>,
) -> Result<(), String> {
    let chunk = match serde_json::from_str::<OllamaChatChunk>(line) {
        Ok(chunk) => chunk,
        Err(error) => {
            let message = format!("Invalid Ollama stream chunk: {error}");
            let _ = send_event(
                tx,
                &GenerateEvent::Error {
                    assistant_message_id: Some(assistant_message_id.to_string()),
                    message: message.clone(),
                },
            )
            .await;
            return Err(message);
        }
    };

    if let Some(message) = chunk.message {
        if !message.thinking.is_empty() {
            thinking_text.push_str(&message.thinking);
            set_generation_progress(
                progress,
                assistant_message_id,
                user_id,
                chat_id,
                content_text,
                thinking_text,
            )
            .await;
            if !send_event(
                tx,
                &GenerateEvent::ThinkingDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.thinking,
                },
            )
            .await
            {
                return Ok(());
            }
        }

        if !message.content.is_empty() {
            content_text.push_str(&message.content);
            set_generation_progress(
                progress,
                assistant_message_id,
                user_id,
                chat_id,
                content_text,
                thinking_text,
            )
            .await;
            if !send_event(
                tx,
                &GenerateEvent::ContentDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.content,
                },
            )
            .await
            {
                return Ok(());
            }
        }
    }

    if chunk.done {
        *done_reason = chunk.done_reason;
    }

    Ok(())
}

async fn send_event(tx: &mpsc::Sender<Result<Bytes, Infallible>>, event: &GenerateEvent) -> bool {
    let Ok(mut payload) = serde_json::to_vec(event) else {
        return false;
    };
    payload.push(b'\n');
    let _ = tx.send(Ok(Bytes::from(payload))).await;
    true
}

fn think_from_mode(mode: &str) -> Option<OllamaThink> {
    match mode {
        "true" => Some(OllamaThink::Bool(true)),
        "false" => Some(OllamaThink::Bool(false)),
        "low" | "medium" | "high" => Some(OllamaThink::Level(mode.to_string())),
        _ => None,
    }
}
