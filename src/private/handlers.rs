use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
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
    auth, backends,
    error::ApiError,
    ollama::{
        self,
        models::{OllamaChatChunk, OllamaChatRequest, OllamaThink},
    },
    private::service,
    rate_limit,
};

#[derive(Debug, Deserialize)]
pub struct PrivateGenerateRequest {
    pub assistant_message_id: String,
    pub backend_id: String,
    pub model_name: String,
    pub think_mode: Option<String>,
    pub messages: Vec<service::PrivateMessageInput>,
}

#[cfg(debug_assertions)]
#[derive(Debug, Deserialize)]
pub struct PrivateStreamTestRequest {
    pub assistant_message_id: String,
    pub content_tokens: Option<u32>,
    pub thinking_tokens: Option<u32>,
    pub delay_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PrivateGenerateEvent {
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
    Error {
        assistant_message_id: Option<String>,
        message: String,
    },
}

#[derive(Debug, Serialize)]
pub struct PrivateVaultKeyResponse {
    pub user_id: String,
    pub key_material: String,
}

pub async fn vault_key(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<PrivateVaultKeyResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let key = service::get_or_create_private_vault_key(&state.db, &user.id).await?;

    Ok(Json(PrivateVaultKeyResponse {
        user_id: key.user_id,
        key_material: key.key_material,
    }))
}

pub async fn generate(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<PrivateGenerateRequest>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("generate", &user.id), 60, 60)
        .await?;

    let assistant_message_id = validate_assistant_message_id(&payload.assistant_message_id)?;
    let backend = service::get_enabled_backend(&state.db, &payload.backend_id).await?;
    let model_name = service::validate_model_name(&payload.model_name)?;
    backends::service::ensure_model_enabled(&state.db, &payload.backend_id, &model_name).await?;
    let messages = service::private_prompt_messages(payload.messages)?;

    Ok(start_private_stream(
        state.http_client,
        backend.base_url,
        model_name,
        assistant_message_id,
        payload.think_mode,
        messages,
    )
    .await)
}

#[cfg(debug_assertions)]
pub async fn generate_stream_test(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<PrivateStreamTestRequest>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(
            rate_limit::user_action_key("dev-stream-test", &user.id),
            30,
            60,
        )
        .await?;
    let assistant_message_id = validate_assistant_message_id(&payload.assistant_message_id)?;
    let content_tokens = payload.content_tokens.unwrap_or(1000).clamp(1, 10_000);
    let thinking_tokens = payload.thinking_tokens.unwrap_or(120).clamp(0, 10_000);
    let delay_ms = payload.delay_ms.unwrap_or(0).min(100);

    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(32);
    tokio::spawn(async move {
        stream_synthetic_private_generation(
            tx,
            assistant_message_id,
            content_tokens,
            thinking_tokens,
            delay_ms,
        )
        .await;
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

    Ok(response)
}

async fn start_private_stream(
    client: reqwest::Client,
    backend_base_url: String,
    model_name: String,
    assistant_message_id: String,
    think_mode: Option<String>,
    messages: Vec<ollama::models::OllamaChatMessage>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(32);

    tokio::spawn(async move {
        stream_private_generation(
            tx,
            client,
            backend_base_url,
            model_name,
            assistant_message_id,
            think_mode,
            messages,
        )
        .await;
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

async fn stream_private_generation(
    tx: mpsc::Sender<Result<Bytes, Infallible>>,
    client: reqwest::Client,
    backend_base_url: String,
    model_name: String,
    assistant_message_id: String,
    think_mode: Option<String>,
    messages: Vec<ollama::models::OllamaChatMessage>,
) {
    let request = OllamaChatRequest {
        model: model_name,
        messages,
        stream: true,
        think: think_mode.as_deref().and_then(think_from_mode),
    };

    let response = match ollama::client::chat_stream(&client, &backend_base_url, &request).await {
        Ok(response) => response,
        Err(error) => {
            let _ = send_event(
                &tx,
                &PrivateGenerateEvent::Error {
                    assistant_message_id: Some(assistant_message_id),
                    message: format!("Ollama request failed: {error}"),
                },
            )
            .await;
            return;
        }
    };

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut done_reason = None;

    while let Some(next) = stream.next().await {
        let bytes = match next {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = send_event(
                    &tx,
                    &PrivateGenerateEvent::Error {
                        assistant_message_id: Some(assistant_message_id),
                        message: format!("Ollama stream failed: {error}"),
                    },
                )
                .await;
                return;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer.drain(..=line_end);
            if line.is_empty() {
                continue;
            }

            match handle_ollama_line(&tx, &assistant_message_id, &line, &mut done_reason).await {
                Ok(true) => {}
                Ok(false) => return,
                Err(message) => {
                    let _ = send_event(
                        &tx,
                        &PrivateGenerateEvent::Error {
                            assistant_message_id: Some(assistant_message_id),
                            message,
                        },
                    )
                    .await;
                    return;
                }
            }
        }
    }

    let line = buffer.trim().to_string();
    if !line.is_empty() {
        match handle_ollama_line(&tx, &assistant_message_id, &line, &mut done_reason).await {
            Ok(true) => {}
            Ok(false) => return,
            Err(message) => {
                let _ = send_event(
                    &tx,
                    &PrivateGenerateEvent::Error {
                        assistant_message_id: Some(assistant_message_id),
                        message,
                    },
                )
                .await;
                return;
            }
        }
    }

    let _ = send_event(
        &tx,
        &PrivateGenerateEvent::MessageDone {
            assistant_message_id,
            done_reason,
        },
    )
    .await;
}

async fn handle_ollama_line(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    assistant_message_id: &str,
    line: &str,
    done_reason: &mut Option<String>,
) -> Result<bool, String> {
    let chunk = serde_json::from_str::<OllamaChatChunk>(line)
        .map_err(|error| format!("Invalid Ollama stream chunk: {error}"))?;

    if let Some(message) = chunk.message {
        if !message.thinking.is_empty()
            && !send_event(
                tx,
                &PrivateGenerateEvent::ThinkingDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.thinking,
                },
            )
            .await
        {
            return Ok(false);
        }

        if !message.content.is_empty()
            && !send_event(
                tx,
                &PrivateGenerateEvent::ContentDelta {
                    assistant_message_id: assistant_message_id.to_string(),
                    delta: message.content,
                },
            )
            .await
        {
            return Ok(false);
        }
    }

    if chunk.done {
        *done_reason = chunk.done_reason;
    }

    Ok(true)
}

#[cfg(debug_assertions)]
async fn stream_synthetic_private_generation(
    tx: mpsc::Sender<Result<Bytes, Infallible>>,
    assistant_message_id: String,
    content_tokens: u32,
    thinking_tokens: u32,
    delay_ms: u64,
) {
    for index in 1..=thinking_tokens {
        if !send_event(
            &tx,
            &PrivateGenerateEvent::ThinkingDelta {
                assistant_message_id: assistant_message_id.clone(),
                delta: synthetic_thinking_delta(index),
            },
        )
        .await
        {
            return;
        }
        sleep_between_test_tokens(delay_ms).await;
    }

    for index in 1..=content_tokens {
        if !send_event(
            &tx,
            &PrivateGenerateEvent::ContentDelta {
                assistant_message_id: assistant_message_id.clone(),
                delta: synthetic_content_delta(index),
            },
        )
        .await
        {
            return;
        }
        sleep_between_test_tokens(delay_ms).await;
    }

    let _ = send_event(
        &tx,
        &PrivateGenerateEvent::MessageDone {
            assistant_message_id,
            done_reason: Some("synthetic_test".to_string()),
        },
    )
    .await;
}

#[cfg(debug_assertions)]
async fn sleep_between_test_tokens(delay_ms: u64) {
    if delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
}

#[cfg(debug_assertions)]
fn synthetic_thinking_delta(index: u32) -> String {
    format!("think-{index:05} ")
}

#[cfg(debug_assertions)]
fn synthetic_content_delta(index: u32) -> String {
    if index % 17 == 0 {
        format!("\nchunk-{index:05};")
    } else {
        format!("tok-{index:05} ")
    }
}

async fn send_event(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    event: &PrivateGenerateEvent,
) -> bool {
    let Ok(mut payload) = serde_json::to_vec(event) else {
        return false;
    };
    payload.push(b'\n');
    tx.send(Ok(Bytes::from(payload))).await.is_ok()
}

fn validate_assistant_message_id(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_assistant_message",
            "Assistant message id is required",
        ));
    }

    Ok(value.to_string())
}

fn think_from_mode(mode: &str) -> Option<OllamaThink> {
    match mode {
        "true" => Some(OllamaThink::Bool(true)),
        "false" => Some(OllamaThink::Bool(false)),
        "low" | "medium" | "high" => Some(OllamaThink::Level(mode.to_string())),
        _ => None,
    }
}
