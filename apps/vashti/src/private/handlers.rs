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
use std::{collections::HashSet, convert::Infallible};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::{
    app_state::AppState,
    auth, backends,
    chats::models::ChatInferenceSettings,
    client_tools::{
        CLIENT_TOOL_TIMEOUT, ClientToolBroker, ClientToolCallIdentity, ResolveClientToolCall,
        session_binding,
    },
    error::ApiError,
    ollama::{
        self,
        models::{
            OllamaChatChunk, OllamaChatMessage, OllamaChatOptions, OllamaChatRequest, OllamaThink,
            OllamaTool, OllamaToolCall, OllamaUsageStats,
        },
    },
    private::service,
    rate_limit, settings, tools,
};

const MAX_CLIENT_TOOLS: usize = 8;
const MAX_CLIENT_TOOL_RESULT_CHARS: usize = 24_000;

#[derive(Debug, Deserialize)]
pub struct PrivateGenerateRequest {
    pub assistant_message_id: String,
    pub backend_id: String,
    pub model_name: String,
    pub think_mode: Option<String>,
    pub inference_settings: Option<ChatInferenceSettings>,
    pub messages: Vec<service::PrivateMessageInput>,
    #[serde(default)]
    pub client_tools: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ClientToolResultRequest {
    pub generation_id: String,
    pub call_id: String,
    pub resume_token: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ClientToolResultResponse {
    pub accepted: bool,
    pub duplicate: bool,
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
    ClientToolCall {
        assistant_message_id: String,
        generation_id: String,
        call_id: String,
        resume_token: String,
        name: String,
        arguments: serde_json::Value,
    },
    MessageDone {
        assistant_message_id: String,
        done_reason: Option<String>,
        stats: Option<OllamaUsageStats>,
    },
    Error {
        assistant_message_id: Option<String>,
        message: String,
    },
}

struct PrivateGenerationTask {
    tx: mpsc::Sender<Result<Bytes, Infallible>>,
    client: reqwest::Client,
    backend_base_url: String,
    model_name: String,
    assistant_message_id: String,
    think_mode: Option<String>,
    inference_options: Option<OllamaChatOptions>,
    messages: Vec<ollama::models::OllamaChatMessage>,
    tools: Vec<OllamaTool>,
    broker: ClientToolBroker,
    generation_id: String,
    user_id: String,
    session_binding: String,
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
    backends::service::ensure_model_enabled_for_user(
        &state.db,
        &user.id,
        &payload.backend_id,
        &model_name,
    )
    .await?;
    let mut messages = service::private_prompt_messages(payload.messages)?;
    let available_tools = allowed_client_tools(&state, &user.id, payload.client_tools).await?;
    let available_tools = if available_tools.is_empty()
        || ollama::client::model_supports_tools(&state.http_client, &backend.base_url, &model_name)
            .await
    {
        available_tools
    } else {
        Vec::new()
    };
    if !available_tools.is_empty() {
        let tool_settings = settings::service::get_tool_settings_private(&state.db).await?;
        messages.insert(
            0,
            OllamaChatMessage {
                role: "system".to_string(),
                content: tools::service::tool_system_prompt(&tool_settings, &available_tools),
                thinking: None,
                images: None,
                tool_name: None,
                tool_calls: None,
            },
        );
    }
    let session_binding = request_session_binding(&state, &jar)?;

    Ok(start_private_stream(
        state.http_client,
        backend.base_url,
        model_name,
        assistant_message_id,
        payload.think_mode,
        inference_settings_to_options(&normalized_inference_settings(
            payload.inference_settings.unwrap_or_default(),
        )),
        messages,
        available_tools,
        state.client_tools,
        user.id,
        session_binding,
    )
    .await)
}

pub async fn client_tool_result(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<ClientToolResultRequest>,
) -> Result<Json<ClientToolResultResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(
            rate_limit::user_action_key("client-tool-result", &user.id),
            240,
            60,
        )
        .await?;
    let result = bounded_client_tool_result(payload.result, payload.error)?;
    let identity = ClientToolCallIdentity {
        generation_id: payload.generation_id.trim(),
        call_id: payload.call_id.trim(),
        user_id: &user.id,
        session_binding: &request_session_binding(&state, &jar)?,
        resume_token: payload.resume_token.trim(),
    };
    let outcome = state.client_tools.resolve(&identity, result).await?;

    Ok(Json(ClientToolResultResponse {
        accepted: true,
        duplicate: outcome == ResolveClientToolCall::AlreadyCompleted,
    }))
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
    inference_options: Option<OllamaChatOptions>,
    messages: Vec<ollama::models::OllamaChatMessage>,
    tools: Vec<OllamaTool>,
    broker: ClientToolBroker,
    user_id: String,
    session_binding: String,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(32);

    tokio::spawn(async move {
        stream_private_generation(PrivateGenerationTask {
            tx,
            client,
            backend_base_url,
            model_name,
            assistant_message_id,
            think_mode,
            inference_options,
            messages,
            tools,
            broker,
            generation_id: Uuid::new_v4().to_string(),
            user_id,
            session_binding,
        })
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

async fn stream_private_generation(task: PrivateGenerationTask) {
    let PrivateGenerationTask {
        tx,
        client,
        backend_base_url,
        model_name,
        assistant_message_id,
        think_mode,
        inference_options,
        messages: mut prompt_messages,
        tools: available_tools,
        broker,
        generation_id,
        user_id,
        session_binding,
    } = task;
    let mut done_reason = None;
    let mut usage_stats: Option<OllamaUsageStats> = None;
    let available_tool_names = available_tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect::<HashSet<_>>();

    loop {
        let request = OllamaChatRequest {
            model: model_name.clone(),
            messages: prompt_messages.clone(),
            stream: true,
            options: inference_options.clone(),
            think: think_mode.as_deref().and_then(think_from_mode),
            tools: (!available_tools.is_empty()).then_some(available_tools.clone()),
        };
        let response = match ollama::client::chat_stream(&client, &backend_base_url, &request).await
        {
            Ok(response) => response,
            Err(error) => {
                send_private_error(
                    &tx,
                    &assistant_message_id,
                    format!("Ollama request failed: {error}"),
                )
                .await;
                return;
            }
        };
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut round_content = String::new();
        let mut round_thinking = String::new();
        let mut round_tool_calls = Vec::new();

        while let Some(next) = stream.next().await {
            let bytes = match next {
                Ok(bytes) => bytes,
                Err(error) => {
                    send_private_error(
                        &tx,
                        &assistant_message_id,
                        format!("Ollama stream failed: {error}"),
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
                if let Err(message) = handle_private_tool_line(
                    &tx,
                    &assistant_message_id,
                    &line,
                    &mut done_reason,
                    &mut usage_stats,
                    &mut round_content,
                    &mut round_thinking,
                    &mut round_tool_calls,
                )
                .await
                {
                    send_private_error(&tx, &assistant_message_id, message).await;
                    return;
                }
            }
        }

        let trailing = buffer.trim();
        if !trailing.is_empty()
            && let Err(message) = handle_private_tool_line(
                &tx,
                &assistant_message_id,
                trailing,
                &mut done_reason,
                &mut usage_stats,
                &mut round_content,
                &mut round_thinking,
                &mut round_tool_calls,
            )
            .await
        {
            send_private_error(&tx, &assistant_message_id, message).await;
            return;
        }
        if round_tool_calls.is_empty() {
            break;
        }

        prompt_messages.push(OllamaChatMessage {
            role: "assistant".to_string(),
            content: round_content,
            thinking: (!round_thinking.trim().is_empty()).then_some(round_thinking),
            images: None,
            tool_name: None,
            tool_calls: Some(normalized_tool_calls(&round_tool_calls)),
        });

        for call in &round_tool_calls {
            let result = if available_tool_names.contains(&call.function.name) {
                let registered = broker
                    .register(&generation_id, &user_id, &session_binding)
                    .await;
                let call_id = registered.call_id.clone();
                if !send_event(
                    &tx,
                    &PrivateGenerateEvent::ClientToolCall {
                        assistant_message_id: assistant_message_id.clone(),
                        generation_id: generation_id.clone(),
                        call_id: call_id.clone(),
                        resume_token: registered.resume_token,
                        name: call.function.name.clone(),
                        arguments: call.function.arguments.clone(),
                    },
                )
                .await
                {
                    broker.cancel(&call_id).await;
                    return;
                }
                let result = tokio::select! {
                    received = registered.receiver => received.unwrap_or_else(|_| {
                        serde_json::json!({ "error": "The device tool call was cancelled." }).to_string()
                    }),
                    _ = tx.closed() => {
                        broker.cancel(&call_id).await;
                        return;
                    },
                    _ = tokio::time::sleep(CLIENT_TOOL_TIMEOUT) => {
                        broker.cancel(&call_id).await;
                        serde_json::json!({ "error": "The device tool did not respond before the timeout." }).to_string()
                    },
                };
                if !send_event(
                    &tx,
                    &PrivateGenerateEvent::ThinkingDelta {
                        assistant_message_id: assistant_message_id.clone(),
                        delta: tools::service::tool_usage_block(call, &result),
                    },
                )
                .await
                {
                    return;
                }
                result
            } else {
                serde_json::json!({
                    "error": format!("{} is not available in this private chat.", call.function.name)
                })
                .to_string()
            };
            prompt_messages.push(OllamaChatMessage {
                role: "tool".to_string(),
                content: result,
                thinking: None,
                images: None,
                tool_name: Some(call.function.name.clone()),
                tool_calls: None,
            });
        }
    }

    let _ = send_event(
        &tx,
        &PrivateGenerateEvent::MessageDone {
            assistant_message_id,
            done_reason,
            stats: usage_stats,
        },
    )
    .await;
}

fn normalized_tool_calls(calls: &[OllamaToolCall]) -> Vec<OllamaToolCall> {
    calls
        .iter()
        .cloned()
        .map(|mut call| {
            call.kind = Some("function".to_string());
            call
        })
        .collect()
}

async fn send_private_error(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    assistant_message_id: &str,
    message: String,
) {
    let _ = send_event(
        tx,
        &PrivateGenerateEvent::Error {
            assistant_message_id: Some(assistant_message_id.to_string()),
            message,
        },
    )
    .await;
}

async fn handle_private_tool_line(
    tx: &mpsc::Sender<Result<Bytes, Infallible>>,
    assistant_message_id: &str,
    line: &str,
    done_reason: &mut Option<String>,
    usage_stats: &mut Option<OllamaUsageStats>,
    round_content: &mut String,
    round_thinking: &mut String,
    round_tool_calls: &mut Vec<OllamaToolCall>,
) -> Result<(), String> {
    let chunk = serde_json::from_str::<OllamaChatChunk>(line)
        .map_err(|error| format!("Invalid Ollama stream chunk: {error}"))?;
    if let Some(chunk_stats) = chunk.usage_stats() {
        match usage_stats {
            Some(stats) => stats.add_assign(chunk_stats),
            None => *usage_stats = Some(chunk_stats),
        }
    }
    if let Some(message) = chunk.message {
        round_tool_calls.extend(message.tool_calls);
        if !message.thinking.is_empty() {
            round_thinking.push_str(&message.thinking);
            if !send_event(
                tx,
                &PrivateGenerateEvent::ThinkingDelta {
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
            round_content.push_str(&message.content);
            if !send_event(
                tx,
                &PrivateGenerateEvent::ContentDelta {
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

async fn allowed_client_tools(
    state: &AppState,
    user_id: &str,
    requested_names: Vec<String>,
) -> Result<Vec<OllamaTool>, ApiError> {
    if requested_names.len() > MAX_CLIENT_TOOLS {
        return Err(ApiError::bad_request(
            "too_many_client_tools",
            "Too many device tools were requested",
        ));
    }
    let notes_allowed = settings::service::tool_is_available_for_user(
        &state.db,
        user_id,
        tools::service::TOOL_NOTES,
    )
    .await?;
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for requested in requested_names {
        let name = requested.trim();
        let Some(tool) = tools::service::client_tool_schema(name) else {
            return Err(ApiError::bad_request(
                "unknown_client_tool",
                format!("Unknown device tool: {name}"),
            ));
        };
        if notes_allowed && seen.insert(name.to_string()) {
            selected.push(tool);
        }
    }
    Ok(selected)
}

fn bounded_client_tool_result(
    result: Option<serde_json::Value>,
    error: Option<String>,
) -> Result<String, ApiError> {
    let value = match (result, error) {
        (Some(result), None) => result,
        (None, Some(error)) => serde_json::json!({ "error": error.trim() }),
        _ => {
            return Err(ApiError::bad_request(
                "invalid_client_tool_result",
                "Provide exactly one of result or error",
            ));
        }
    };
    let serialized = serde_json::to_string(&value)
        .map_err(|_| ApiError::bad_request("invalid_client_tool_result", "Invalid tool result"))?;
    if serialized.chars().count() > MAX_CLIENT_TOOL_RESULT_CHARS {
        return Err(ApiError::bad_request(
            "client_tool_result_too_large",
            "Device tool result is too large",
        ));
    }
    Ok(serialized)
}

fn request_session_binding(state: &AppState, jar: &CookieJar) -> Result<String, ApiError> {
    let session = jar
        .get(&state.config.session_cookie_name)
        .ok_or_else(ApiError::unauthorized)?;
    Ok(session_binding(session.value()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_tool_result_requires_exactly_one_payload() {
        assert!(bounded_client_tool_result(Some(serde_json::json!({ "ok": true })), None).is_ok());
        assert!(bounded_client_tool_result(None, Some("failed".to_string())).is_ok());
        assert!(bounded_client_tool_result(None, None).is_err());
        assert!(
            bounded_client_tool_result(
                Some(serde_json::json!({ "ok": true })),
                Some("failed".to_string())
            )
            .is_err()
        );
    }

    #[test]
    fn client_tool_result_is_bounded() {
        let oversized = "x".repeat(MAX_CLIENT_TOOL_RESULT_CHARS + 1);
        let error = bounded_client_tool_result(Some(serde_json::json!(oversized)), None)
            .expect_err("oversized device tool results must be rejected");
        assert_eq!(error.code(), "client_tool_result_too_large");
    }
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
            stats: None,
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
    if index.is_multiple_of(17) {
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
