use std::{collections::HashMap, time::Duration};

use base64::{
    Engine,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use futures_util::StreamExt;
use reqwest::{Method, Response, header};
use serde::{Deserialize, Serialize};
use tauri::{State, ipc::Channel};
use tokio_util::sync::CancellationToken;

use crate::{
    connections::{Connection, NativeState},
    validation::api_url,
};

const MAX_NATIVE_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct HttpRequest {
    method: String,
    path: String,
    #[serde(default)]
    headers: HashMap<String, String>,
    body_text: Option<String>,
    response_type: ResponseType,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResponseType {
    Text,
    Base64,
}

#[derive(Debug, Deserialize)]
pub struct MultipartRequest {
    method: String,
    path: String,
    #[serde(default)]
    headers: HashMap<String, String>,
    parts: Vec<MultipartPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum MultipartPart {
    Text {
        name: String,
        value: String,
    },
    File {
        name: String,
        filename: String,
        mime_type: String,
        data_base64: String,
    },
}

#[derive(Debug, Deserialize)]
pub struct StreamRequest {
    request_id: String,
    method: String,
    path: String,
    #[serde(default)]
    headers: HashMap<String, String>,
    body_text: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HttpResponse {
    status: u16,
    headers: HashMap<String, String>,
    body_text: Option<String>,
    body_base64: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    chunk_base64: String,
}

#[tauri::command]
pub async fn native_http_request(
    state: State<'_, NativeState>,
    request: HttpRequest,
) -> Result<HttpResponse, String> {
    let connection = state.active_connection().await?;
    let method = parse_method(&request.method)?;
    let mut builder =
        request_builder(&state, &connection, method, &request.path, &request.headers).await?;
    if let Some(body) = request.body_text {
        builder = builder.body(body);
    }
    let response = builder
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(request_error)?;
    let response = capture_session(&state, &connection, response).await?;
    response_payload(response, request.response_type).await
}

#[tauri::command]
pub async fn native_http_multipart(
    state: State<'_, NativeState>,
    request: MultipartRequest,
) -> Result<HttpResponse, String> {
    let connection = state.active_connection().await?;
    let method = parse_method(&request.method)?;
    let mut form = reqwest::multipart::Form::new();
    for part in request.parts {
        form = match part {
            MultipartPart::Text { name, value } => form.text(name, value),
            MultipartPart::File {
                name,
                filename,
                mime_type,
                data_base64,
            } => {
                let bytes = STANDARD
                    .decode(data_base64)
                    .map_err(|_| "Uploaded file data was invalid".to_string())?;
                let part = reqwest::multipart::Part::bytes(bytes)
                    .file_name(filename)
                    .mime_str(&mime_type)
                    .map_err(|_| "Uploaded file type was invalid".to_string())?;
                form.part(name, part)
            }
        };
    }
    let response = request_builder(&state, &connection, method, &request.path, &request.headers)
        .await?
        .multipart(form)
        .timeout(Duration::from_secs(180))
        .send()
        .await
        .map_err(request_error)?;
    let response = capture_session(&state, &connection, response).await?;
    response_payload(response, ResponseType::Text).await
}

#[tauri::command]
pub async fn native_http_stream(
    state: State<'_, NativeState>,
    request: StreamRequest,
    on_event: Channel<StreamChunk>,
) -> Result<(), String> {
    if request.request_id.len() > 100 {
        return Err("Native request identifier is invalid".to_string());
    }
    let connection = state.active_connection().await?;
    let token = CancellationToken::new();
    state
        .active_requests
        .lock()
        .await
        .insert(request.request_id.clone(), token.clone());

    let result = async {
        let method = parse_method(&request.method)?;
        let mut builder =
            request_builder(&state, &connection, method, &request.path, &request.headers).await?;
        if let Some(body) = request.body_text {
            builder = builder.body(body);
        }
        let response = tokio::select! {
            _ = token.cancelled() => return Err("Generation cancelled".to_string()),
            response = builder.send() => response.map_err(request_error)?,
        };
        let response = capture_session(&state, &connection, response).await?;
        if !response.status().is_success() {
            let status = response.status();
            let body =
                String::from_utf8_lossy(&response_bytes_limited(response).await?).into_owned();
            return Err(server_error(status.as_u16(), &body));
        }
        let mut stream = response.bytes_stream();
        loop {
            let next = tokio::select! {
                _ = token.cancelled() => return Err("Generation cancelled".to_string()),
                next = stream.next() => next,
            };
            let Some(chunk) = next else {
                break;
            };
            let chunk = chunk.map_err(request_error)?;
            on_event
                .send(StreamChunk {
                    chunk_base64: STANDARD.encode(chunk),
                })
                .map_err(|_| "The app stopped receiving the generation stream".to_string())?;
        }
        Ok(())
    }
    .await;

    state
        .active_requests
        .lock()
        .await
        .remove(&request.request_id);
    result
}

#[tauri::command]
pub async fn native_cancel_request(
    state: State<'_, NativeState>,
    request_id: String,
) -> Result<(), String> {
    if let Some(token) = state.active_requests.lock().await.get(&request_id) {
        token.cancel();
    }
    Ok(())
}

pub async fn authenticated_media(
    state: &NativeState,
    path: &str,
) -> Result<(u16, HashMap<String, String>, Vec<u8>), String> {
    let connection = state.active_connection().await?;
    let response = request_builder(state, &connection, Method::GET, path, &HashMap::new())
        .await?
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .map_err(request_error)?;
    let response = capture_session(state, &connection, response).await?;
    let status = response.status().as_u16();
    let headers = response_headers(&response);
    let bytes = response_bytes_limited(response).await?;
    Ok((status, headers, bytes))
}

pub async fn resolve_authenticated_media(
    state: &NativeState,
    namespace: &str,
    encoded_path: &str,
) -> Result<(u16, HashMap<String, String>, Vec<u8>), String> {
    let connection = state.active_connection().await?;
    let expected_namespace = format!("{}-{}", connection.instance_id, connection.id);
    if namespace != expected_namespace {
        return Err("Media URL does not belong to the selected server".to_string());
    }

    let path = String::from_utf8(
        URL_SAFE_NO_PAD
            .decode(encoded_path)
            .map_err(|_| "Media URL is invalid".to_string())?,
    )
    .map_err(|_| "Media URL is invalid".to_string())?;
    if !path.starts_with("/api/attachments/") && !path.starts_with("/api/persona-avatars/") {
        return Err("Media URL is not allowed".to_string());
    }

    authenticated_media(state, &path).await
}

async fn request_builder(
    state: &NativeState,
    connection: &Connection,
    method: Method,
    path: &str,
    headers: &HashMap<String, String>,
) -> Result<reqwest::RequestBuilder, String> {
    let url = api_url(&connection.base_url, path)?;
    let mut builder = state
        .client
        .request(method, url)
        .header(header::ORIGIN, &connection.base_url)
        .header(header::ACCEPT, "application/json");
    if let Some(cookie) = state.sessions.get(&connection.id).await? {
        builder = builder.header(header::COOKIE, cookie);
    }
    for (name, value) in headers {
        let normalized = name.to_ascii_lowercase();
        if !matches!(normalized.as_str(), "accept" | "content-type" | "range") {
            return Err(format!("Native request header {name} is not allowed"));
        }
        builder = builder.header(name, value);
    }
    Ok(builder)
}

fn parse_method(value: &str) -> Result<Method, String> {
    let method =
        Method::from_bytes(value.as_bytes()).map_err(|_| "Invalid HTTP method".to_string())?;
    if matches!(
        method,
        Method::GET | Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    ) {
        Ok(method)
    } else {
        Err("Native request method is not allowed".to_string())
    }
}

async fn capture_session(
    state: &NativeState,
    connection: &Connection,
    response: Response,
) -> Result<Response, String> {
    let mut session_update = None;
    for value in response.headers().get_all(header::SET_COOKIE) {
        let Ok(value) = value.to_str() else {
            continue;
        };
        let Some(pair) = value.split(';').next().map(str::trim) else {
            continue;
        };
        if pair.starts_with("vashti_session=") {
            if pair == "vashti_session=" {
                session_update = Some(None);
            } else {
                session_update = Some(Some(pair.to_string()));
            }
        }
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED || session_update == Some(None) {
        state.sessions.delete(&connection.id).await?;
    } else if let Some(Some(cookie)) = session_update {
        state.sessions.set(&connection.id, &cookie).await?;
    }
    Ok(response)
}

async fn response_payload(
    response: Response,
    response_type: ResponseType,
) -> Result<HttpResponse, String> {
    let status = response.status().as_u16();
    let headers = response_headers(&response);
    let bytes = response_bytes_limited(response).await?;
    let (body_text, body_base64) = match response_type {
        ResponseType::Text => (
            Some(String::from_utf8(bytes).map_err(|_| "Server returned invalid text".to_string())?),
            None,
        ),
        ResponseType::Base64 => (None, Some(STANDARD.encode(bytes))),
    };
    Ok(HttpResponse {
        status,
        headers,
        body_text,
        body_base64,
    })
}

async fn response_bytes_limited(response: Response) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_NATIVE_RESPONSE_BYTES as u64)
    {
        return Err("Server response was too large".to_string());
    }

    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default()
        .min(MAX_NATIVE_RESPONSE_BYTES);
    let mut bytes = Vec::with_capacity(capacity);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(request_error)?;
        if bytes.len().saturating_add(chunk.len()) > MAX_NATIVE_RESPONSE_BYTES {
            return Err("Server response was too large".to_string());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

fn response_headers(response: &Response) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_LENGTH,
        header::CONTENT_RANGE,
        header::CACHE_CONTROL,
        header::CONTENT_DISPOSITION,
    ] {
        if let Some(value) = response
            .headers()
            .get(&name)
            .and_then(|value| value.to_str().ok())
        {
            headers.insert(name.as_str().to_string(), value.to_string());
        }
    }
    headers
}

fn request_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "The Vashti server request timed out".to_string()
    } else if error.is_connect() {
        "Could not connect to the Vashti server".to_string()
    } else {
        format!("Vashti server request failed: {error}")
    }
}

fn server_error(status: u16, body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|payload| {
            payload
                .pointer("/error/message")?
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("Request failed with {status}"))
}
