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
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    app_state::AppState,
    auth,
    chats::{
        models::{ChatDetail, ChatMessage, ChatSummary},
        service,
    },
    error::ApiError,
    ollama::{
        self,
        models::{OllamaChatChunk, OllamaChatRequest, OllamaThink},
    },
};

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
pub struct GenerateChatRequest {
    pub user_message: GenerateUserMessageRequest,
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub think_mode: Option<String>,
    #[serde(default)]
    pub attachments: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateUserMessageRequest {
    pub content_text: String,
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
    pub attachments: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct BranchMessageRequest {
    pub content_text: String,
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub think_mode: Option<String>,
    #[serde(default)]
    pub attachments: Vec<serde_json::Value>,
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
    let (active_root_message_id, messages) =
        service::list_messages(&state.db, &user.id, &chat_id).await?;

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
    let message =
        service::edit_message(&state.db, &user.id, &chat_id, &message_id, payload).await?;

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
    if !payload.attachments.is_empty() {
        return Err(ApiError::bad_request(
            "attachments_not_supported",
            "Attachments are not implemented yet",
        ));
    }

    let prepared = service::prepare_generation(&state.db, &user.id, &chat_id, payload).await?;
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
    if !payload.attachments.is_empty() {
        return Err(ApiError::bad_request(
            "attachments_not_supported",
            "Attachments are not implemented yet",
        ));
    }

    let prepared =
        service::prepare_branch_generation(&state.db, &user.id, &chat_id, &message_id, payload)
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
            "attachments_not_supported",
            "Attachments are not implemented yet",
        ));
    }

    let prepared =
        service::prepare_regeneration(&state.db, &user.id, &chat_id, &message_id, payload).await?;
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
    user_id: String,
    chat_id: String,
    prepared: service::PreparedGeneration,
    cancellation: tokio_util::sync::CancellationToken,
) {
    let assistant_message_id = prepared.assistant_message.id.clone();
    let mut content_text = String::new();
    let mut thinking_text = String::new();
    let mut done_reason = None;

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
                        assistant_message_id: Some(assistant_message_id),
                        message,
                    },
                )
                .await;
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
                        assistant_message_id,
                    },
                )
                .await;
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
                                &assistant_message_id,
                                &line,
                                &mut content_text,
                                &mut thinking_text,
                                &mut done_reason,
                            )
                            .await
                            {
                                Ok(true) => {}
                                Ok(false) => {
                                    let _ = service::stop_generation(
                                        &db,
                                        &user_id,
                                        &chat_id,
                                        &assistant_message_id,
                                        &content_text,
                                        &thinking_text,
                                    )
                                    .await;
                                    return;
                                }
                                Err(message) => {
                                    let _ = service::fail_generation(
                                        &db,
                                        &assistant_message_id,
                                        &content_text,
                                        &thinking_text,
                                        &message,
                                    )
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
                                assistant_message_id: Some(assistant_message_id),
                                message,
                            },
                        )
                        .await;
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
            &assistant_message_id,
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
            assistant_message_id,
            done_reason,
        },
    )
    .await;
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

    tokio::spawn(async move {
        stream_generation(tx, db, client, user_id, chat_id, prepared, cancellation).await;
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

async fn handle_ollama_line(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    assistant_message_id: &str,
    line: &str,
    content_text: &mut String,
    thinking_text: &mut String,
    done_reason: &mut Option<String>,
) -> Result<bool, String> {
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
            if !send_event(
                tx,
                &GenerateEvent::ThinkingDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.thinking,
                },
            )
            .await
            {
                return Ok(false);
            }
        }

        if !message.content.is_empty() {
            content_text.push_str(&message.content);
            if !send_event(
                tx,
                &GenerateEvent::ContentDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.content,
                },
            )
            .await
            {
                return Ok(false);
            }
        }
    }

    if chunk.done {
        *done_reason = chunk.done_reason;
    }

    Ok(true)
}

async fn send_event(tx: &mpsc::Sender<Result<Bytes, Infallible>>, event: &GenerateEvent) -> bool {
    let Ok(mut payload) = serde_json::to_vec(event) else {
        return false;
    };
    payload.push(b'\n');
    tx.send(Ok(Bytes::from(payload))).await.is_ok()
}

fn think_from_mode(mode: &str) -> Option<OllamaThink> {
    match mode {
        "true" => Some(OllamaThink::Bool(true)),
        "false" => Some(OllamaThink::Bool(false)),
        "low" | "medium" | "high" => Some(OllamaThink::Level(mode.to_string())),
        _ => None,
    }
}
