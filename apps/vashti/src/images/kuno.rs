//! Server-to-server discovery and transient image streaming. Never replay jobs.
use crate::error::ApiError;
use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use kuno_image::preview::{MAX_PREVIEW_BYTES, Preview};
use kuno_protocol::{
    Capabilities, IMAGE_STREAM_MEDIA_TYPE, ImageRecipe, ImageStreamEvent, ImageWorkflowCapability,
};
use reqwest::{Client, Response, StatusCode, Url};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
};

const MAX_DISCOVERY_BYTES: usize = 64 * 1024;
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4 + 1024;
const GENERATE_PATH: &str = "/kuno/v1/images/generate";
const CACHE_TTL: Duration = Duration::from_secs(5);
type CachedHost = Arc<Mutex<Option<(Instant, Discovery)>>>;

#[derive(Clone, Serialize)]
pub(super) struct Discovery {
    pub state: &'static str,
    pub message: String,
    pub capabilities: Option<Capabilities>,
}
impl Discovery {
    fn unavailable(state: &'static str, message: &str) -> Self {
        Self {
            state,
            message: message.into(),
            capabilities: None,
        }
    }
    pub fn not_configured() -> Self {
        Self::unavailable(
            "not_configured",
            "Choose an inference host in Image preferences.",
        )
    }
    pub fn disabled_host() -> Self {
        Self::unavailable(
            "not_configured",
            "The image inference host is disabled. Ask an administrator to choose an enabled host.",
        )
    }
    pub fn workflow(&self, id: &str) -> Option<&ImageWorkflowCapability> {
        self.capabilities
            .as_ref()
            .filter(|_| matches!(self.state, "ready" | "partial"))?
            .images
            .workflows
            .iter()
            .find(|w| w.id == id && w.state == "ready")
    }
    fn require_recipe(&self, recipe: &ImageRecipe) -> Result<(), ApiError> {
        let Some(workflow) = self.workflow(&recipe.workflow) else {
            return Err(ApiError::conflict(
                "image_service_unavailable",
                if matches!(self.state, "ready" | "partial") {
                    "This image workflow is unavailable on the selected host. Choose a ready workflow in Generation settings.".into()
                } else {
                    self.message.clone()
                },
            ));
        };
        let attention = match recipe.attention {
            kuno_protocol::ImageAttention::Standard => "standard",
            kuno_protocol::ImageAttention::Fast => "fast",
        };
        if !workflow.attention.iter().any(|a| a == attention) {
            return Err(ApiError::bad_request(
                "image_attention_unsupported",
                "The selected host does not support this attention mode. Choose Standard attention.",
            ));
        }
        if !(workflow.reference_images_min as usize..=workflow.reference_images_max as usize)
            .contains(&recipe.reference_images.len())
        {
            return Err(ApiError::bad_request(
                "image_references_invalid",
                "The reference image count does not match the selected workflow.",
            ));
        }
        Ok(())
    }
}

pub(super) struct KunoClient {
    http: Client,
    cache: Mutex<HashMap<String, CachedHost>>,
}
impl Default for KunoClient {
    fn default() -> Self {
        Self {
            http: Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("valid Kuno HTTP client configuration"),
            cache: Mutex::new(HashMap::new()),
        }
    }
}
impl KunoClient {
    pub async fn discover(&self, base: &str) -> Discovery {
        let host = {
            let mut cache = self.cache.lock().await;
            if !cache.contains_key(base)
                && cache.len() >= 32
                && let Some(key) = cache.keys().next().cloned()
            {
                cache.remove(&key);
            }
            cache.entry(base.into()).or_default().clone()
        };
        // Only requests to this host wait for its bounded probe; other hosts
        // remain independent. Cache bounded discovery, never inference content.
        let mut cached = host.lock().await;
        if let Some((checked, value)) = &*cached
            && checked.elapsed() < CACHE_TTL
        {
            return value.clone();
        }
        let value = self.probe(base).await;
        *cached = Some((Instant::now(), value.clone()));
        value
    }

    async fn probe(&self, base: &str) -> Discovery {
        let Ok(url) = inference_url(base, "/kuno/v1/capabilities") else {
            return Discovery::unavailable(
                "invalid",
                "The inference host URL is invalid. Ask an administrator to check it.",
            );
        };
        let response = self
            .http
            .get(url)
            .timeout(Duration::from_secs(2))
            .send()
            .await;
        let Ok(response) = response else {
            return Discovery::unavailable(
                "unreachable",
                "Image discovery could not reach the inference host. Check the host and try again.",
            );
        };
        match response.status() {
            StatusCode::NOT_FOUND => {
                return Discovery::unavailable(
                    "unsupported",
                    "This host has no supported Kuno image discovery. Install or upgrade Kuno to use images; ordinary Ollama chat is unaffected.",
                );
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Discovery::unavailable(
                    "access_denied",
                    "The inference host denied image discovery. Ask an administrator to check its proxy access policy.",
                );
            }
            status if !status.is_success() => {
                return Discovery::unavailable(
                    "unreachable",
                    "The inference host could not report image capabilities. Check the host and try again.",
                );
            }
            _ => (),
        }
        let bytes = match bounded(response, MAX_DISCOVERY_BYTES).await {
            Ok(bytes) => bytes,
            Err(error) if error.code() == "image_host_unreachable" => {
                return Discovery::unavailable(
                    "unreachable",
                    "Image discovery was interrupted or timed out. Check the host and try again.",
                );
            }
            Err(_) => {
                return Discovery::unavailable(
                    "invalid",
                    "Image discovery returned an incomplete or oversized response. Check or upgrade Kuno.",
                );
            }
        };
        let Ok(capabilities) = serde_json::from_slice::<Capabilities>(&bytes) else {
            return Discovery::unavailable(
                "invalid",
                "This host returned invalid image discovery. Check or upgrade Kuno.",
            );
        };
        if !valid_capabilities(&capabilities) {
            return Discovery::unavailable(
                "unsupported",
                "This host uses an unsupported Kuno image API. Upgrade Kuno and Vashti together.",
            );
        }
        let (state, message) = match capabilities.images.state.as_str() {
            "ready" => ("ready", "Image service ready."),
            "partial" => (
                "partial",
                "Some image workflows are unavailable. Choose a ready workflow.",
            ),
            "disabled" => (
                "disabled",
                "Image generation is not enabled on this Kuno host. Ask an administrator to configure images.",
            ),
            "unsupported" => (
                "unsupported",
                "This Kuno build has no image engine. Install a build with image support.",
            ),
            _ => (
                "unavailable",
                reason_message(
                    capabilities
                        .images
                        .reason
                        .as_deref()
                        .unwrap_or("image_runtime_unavailable"),
                ),
            ),
        };
        Discovery {
            state,
            message: message.into(),
            capabilities: Some(capabilities),
        }
    }

    pub async fn generate(
        &self,
        base: &str,
        recipe: &ImageRecipe,
        mut publish: impl FnMut(Preview),
    ) -> Result<Vec<u8>, ApiError> {
        let discovery = self.discover(base).await;
        discovery.require_recipe(recipe)?;
        let path = &discovery
            .capabilities
            .as_ref()
            .unwrap()
            .images
            .generate_path;
        let response = self.http.post(inference_url(base, path)?).json(recipe).send().await
            .map_err(|_| ApiError::conflict("image_host_unreachable", "Kuno disconnected before image delivery. Check the inference host; this request was not retried."))?;
        if !response.status().is_success() {
            let status = response.status();
            if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
                return Err(ApiError::conflict(
                    "image_access_denied",
                    "The inference host denied image generation. Ask an administrator to check its proxy access policy.",
                ));
            }
            let bytes = bounded(response, 8192).await.unwrap_or_default();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
            return Err(inference_error(
                value
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("image_generation_failed"),
            ));
        }
        if response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_none_or(|v| v.split(';').next().unwrap_or("").trim() != IMAGE_STREAM_MEDIA_TYPE)
        {
            return Err(ApiError::conflict(
                "image_stream_unsupported",
                "Kuno did not return the supported image stream. Upgrade Kuno and Vashti together.",
            ));
        }
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(std::io::Error::other));
        let mut lines = FramedRead::new(
            StreamReader::new(stream),
            LinesCodec::new_with_max_length(MAX_EVENT_BYTES),
        );
        let mut started = false;
        let mut result = None;
        let mut count = 0u32;
        while let Some(line) = lines.next().await {
            let line = line.map_err(|_| stream_invalid())?;
            count += 1;
            if count > recipe.steps + 4 {
                return Err(stream_invalid());
            }
            if line.trim().is_empty() {
                continue;
            }
            if result.is_some() {
                return Err(stream_invalid());
            }
            let event =
                serde_json::from_str::<ImageStreamEvent>(&line).map_err(|_| stream_invalid())?;
            match event {
                ImageStreamEvent::Started if !started => started = true,
                ImageStreamEvent::Preview {
                    step,
                    steps,
                    width,
                    height,
                    png_base64,
                } if started => {
                    if steps != recipe.steps
                        || step > steps
                        || width == 0
                        || height == 0
                        || width > 512
                        || height > 512
                    {
                        return Err(stream_invalid());
                    }
                    let max = MAX_PREVIEW_BYTES - 16;
                    let png = decode_png(png_base64, max, width, height).await?;
                    publish(Preview {
                        step,
                        steps,
                        width,
                        height,
                        png_base64: STANDARD.encode(png),
                    });
                }
                ImageStreamEvent::Result {
                    mime_type,
                    png_base64,
                } if started && mime_type == "image/png" => {
                    result = Some(
                        decode_png(png_base64, MAX_IMAGE_BYTES, recipe.width, recipe.height)
                            .await?,
                    );
                }
                ImageStreamEvent::Error { error } if started => {
                    return Err(inference_error(&error));
                }
                _ => return Err(stream_invalid()),
            }
        }
        result.ok_or_else(|| ApiError::conflict("image_stream_interrupted", "Image delivery ended before the final image arrived. No incomplete image was saved; you can retry manually."))
    }
}

fn inference_url(base: &str, path: &str) -> Result<Url, ApiError> {
    if !matches!(path, "/kuno/v1/capabilities" | GENERATE_PATH) {
        return Err(stream_invalid());
    }
    let mut url = Url::parse(base).map_err(|_| stream_invalid())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(stream_invalid());
    }
    url.set_path(&format!("{}{}", url.path().trim_end_matches('/'), path));
    Ok(url)
}
fn valid_capabilities(value: &Capabilities) -> bool {
    let images = &value.images;
    value.service == "kuno"
        && value.discovery_version == 1
        && value.inference_auth == "none"
        && value.version.len() <= 128
        && images.api_version == 1
        && images.access == "inference"
        && images.generate_path == GENERATE_PATH
        && images
            .response_formats
            .iter()
            .any(|v| v == IMAGE_STREAM_MEDIA_TYPE)
        && matches!(
            images.state.as_str(),
            "disabled" | "unsupported" | "unavailable" | "partial" | "ready"
        )
        && images.workflows.len() <= 16
        && images.workflows.iter().all(|w| {
            w.id.len() <= 128
                && matches!(w.state.as_str(), "ready" | "unavailable")
                && w.reference_images_min <= w.reference_images_max
                && w.reference_images_max <= 10
                && !w.attention.is_empty()
                && w.attention
                    .iter()
                    .all(|a| matches!(a.as_str(), "standard" | "fast"))
        })
}
async fn bounded(mut response: Response, limit: usize) -> Result<Vec<u8>, ApiError> {
    if response.content_length().is_some_and(|n| n > limit as u64) {
        return Err(stream_invalid());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        ApiError::conflict(
            "image_host_unreachable",
            "Image host response was interrupted.",
        )
    })? {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(stream_invalid());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
async fn decode_png(
    encoded: String,
    limit: usize,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ApiError> {
    if encoded.len() > limit.div_ceil(3) * 4 {
        return Err(stream_invalid());
    }
    tokio::task::spawn_blocking(move || {
        let bytes = STANDARD.decode(encoded).map_err(|_| stream_invalid())?;
        if bytes.len() > limit {
            return Err(stream_invalid());
        }
        kuno_image::validate_output(&bytes, width, height).map_err(|_| stream_invalid())?;
        Ok(bytes)
    })
    .await
    .map_err(|_| stream_invalid())?
}
fn stream_invalid() -> ApiError {
    ApiError::conflict(
        "image_stream_invalid",
        "Kuno returned an invalid or interrupted image stream. No incomplete image was saved.",
    )
}
fn reason_message(code: &str) -> &'static str {
    match code {
        "image_models_unavailable" | "image_collection_not_enabled" => {
            "Image models are missing or unavailable on Kuno. Ask an administrator to install or enable the required collection."
        }
        "image_privacy_unavailable" => {
            "Kuno's image privacy protections are unavailable. Ask an administrator to check the inference host."
        }
        "image_generation_not_configured" => {
            "Image generation is not configured on Kuno. Ask an administrator to enable it."
        }
        "image_engine_not_in_this_build" => {
            "This Kuno build has no image engine. Install a build with image support."
        }
        "image_attention_unsupported" => {
            "This attention mode is not supported by the image host. Choose Standard attention."
        }
        "image_queue_full" | "queue_full" | "queue_timeout" | "admission_paused"
        | "recovery_required" => {
            "Kuno is busy, paused, or awaiting recovery. Check the inference host before trying again."
        }
        _ => {
            "Kuno could not generate the image. Check model readiness and the inference host; this request was not retried."
        }
    }
}
fn inference_error(code: &str) -> ApiError {
    // Never expose an arbitrary backend error string or its embedded content.
    ApiError::conflict("image_inference_failed", reason_message(code))
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use axum::{
        Json, Router,
        body::Body,
        http::HeaderMap,
        response::IntoResponse,
        routing::{get, post},
    };
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub(crate) fn capabilities() -> Value {
        json!({"service":"kuno","discovery_version":1,"version":"0.1.0","inference_auth":"none",
        "future_field":true,"images":{"api_version":1,"access":"inference","state":"ready","reason":null,
        "generate_path":GENERATE_PATH,"response_formats":[IMAGE_STREAM_MEDIA_TYPE],"workflows":[
            {"id":kuno_protocol::QWEN21_RECIPE,"state":"ready","reason":null,"reference_images_min":0,"reference_images_max":0,"attention":["standard"]},
            {"id":kuno_protocol::QWEN21_EDIT_RECIPE,"state":"ready","reason":null,"reference_images_min":1,"reference_images_max":10,"attention":["standard"]},
            {"id":kuno_protocol::QWEN_RECIPE,"state":"ready","reason":null,"reference_images_min":0,"reference_images_max":0,"attention":["standard"]}
        ]}})
    }
    fn recipe() -> ImageRecipe {
        let mut recipe = crate::images::models::Preferences::default()
            .generate
            .recipe;
        recipe.width = 64;
        recipe.height = 64;
        recipe.steps = 2;
        recipe.prompt = "Synthetic transport test".into();
        recipe
    }
    fn png() -> String {
        STANDARD.encode(include_bytes!("fixtures/64.png"))
    }
    fn events() -> String {
        format!(
            "{{\"type\":\"started\"}}\n{}\n{}\n",
            json!({"type":"preview","step":1,"steps":2,"width":64,"height":64,"png_base64":png()}),
            json!({"type":"result","mime_type":"image/png","png_base64":png()})
        )
    }
    fn discovery(app: Router, value: Value) -> Router {
        app.route(
            "/kuno/v1/capabilities",
            get(move |headers: HeaderMap| {
                let value = value.clone();
                async move {
                    assert!(!headers.contains_key("authorization"));
                    assert!(!headers.contains_key("origin"));
                    Json(value)
                }
            }),
        )
    }
    async fn host(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), server)
    }
    fn stream_app(value: String) -> Router {
        discovery(
            Router::new().route(
                GENERATE_PATH,
                post(move |headers: HeaderMap| {
                    let value = value.clone();
                    async move {
                        assert!(!headers.contains_key("authorization"));
                        assert!(!headers.contains_key("x-kuno-image-preview-id"));
                        assert!(!headers.contains_key("origin"));
                        assert_eq!(headers["content-type"], "application/json");
                        let chunk_size = if value.len() > 1024 * 1024 {
                            64 * 1024
                        } else {
                            13
                        };
                        let chunks = value
                            .as_bytes()
                            .chunks(chunk_size)
                            .map(|b| Ok::<_, std::convert::Infallible>(b.to_vec()))
                            .collect::<Vec<_>>();
                        (
                            [("content-type", IMAGE_STREAM_MEDIA_TYPE)],
                            Body::from_stream(futures_util::stream::iter(chunks)),
                        )
                            .into_response()
                    }
                }),
            ),
            capabilities(),
        )
    }

    #[tokio::test]
    async fn keyless_stream_handles_chunk_boundaries_previews_and_base_prefixes() {
        let (url, server) = host(Router::new().nest("/prefix", stream_app(events()))).await;
        let client = KunoClient::default();
        let mut frames = Vec::new();
        let output = client
            .generate(&format!("{url}/prefix/"), &recipe(), |p| frames.push(p))
            .await
            .unwrap();
        assert_eq!(output, include_bytes!("fixtures/64.png"));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].step, 1);
        assert_eq!(STANDARD.decode(&frames[0].png_base64).unwrap(), output);
        server.abort();
    }

    #[tokio::test]
    async fn discovery_is_cached_and_plain_ollama_access_errors_stay_distinct() {
        let calls = Arc::new(AtomicUsize::new(0));
        let called = calls.clone();
        let app = Router::new().route(
            "/kuno/v1/capabilities",
            get(move || {
                called.fetch_add(1, Ordering::SeqCst);
                async { Json(capabilities()) }
            }),
        );
        let (url, server) = host(app).await;
        let client = KunoClient::default();
        assert_eq!(client.discover(&url).await.state, "ready");
        assert_eq!(client.discover(&url).await.state, "ready");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        server.abort();
        for (status, expected) in [
            (StatusCode::NOT_FOUND, "unsupported"),
            (StatusCode::UNAUTHORIZED, "access_denied"),
            (StatusCode::FORBIDDEN, "access_denied"),
            (StatusCode::SERVICE_UNAVAILABLE, "unreachable"),
        ] {
            let (url, server) = host(
                Router::new().route("/kuno/v1/capabilities", get(move || async move { status })),
            )
            .await;
            assert_eq!(client.discover(&url).await.state, expected);
            server.abort();
        }
        let (url, server) = host(Router::new()).await;
        server.abort();
        server.await.unwrap_err();
        assert_eq!(client.discover(&url).await.state, "unreachable");
    }

    #[tokio::test]
    async fn discovery_rejects_redirects_oversized_invalid_and_incompatible_responses() {
        let redirect = axum::response::Redirect::temporary("/destination");
        let (url, server) = host(Router::new().route(
            "/kuno/v1/capabilities",
            get(move || async move { redirect }),
        ))
        .await;
        assert_eq!(
            KunoClient::default().discover(&url).await.state,
            "unreachable"
        );
        server.abort();
        for body in ["x".repeat(MAX_DISCOVERY_BYTES + 1), "not-json".into()] {
            let (url, server) = host(Router::new().route(
                "/kuno/v1/capabilities",
                get(move || {
                    let body = body.clone();
                    async move { body }
                }),
            ))
            .await;
            assert_eq!(KunoClient::default().discover(&url).await.state, "invalid");
            server.abort();
        }
        for (key, value) in [
            ("discovery_version", json!(2)),
            ("service", json!("not-kuno")),
            ("inference_auth", json!("bearer")),
        ] {
            let mut cap = capabilities();
            cap[key] = value;
            let (url, server) = host(discovery(Router::new(), cap)).await;
            assert_eq!(
                KunoClient::default().discover(&url).await.state,
                "unsupported"
            );
            server.abort();
        }
        let mut cap = capabilities();
        cap["images"]["generate_path"] = json!("http://127.0.0.1:11436/kuno/v1/admission/pause");
        assert!(!valid_capabilities(&serde_json::from_value(cap).unwrap()));
    }

    #[tokio::test]
    async fn partial_and_disabled_readiness_use_the_advertised_workflows() {
        let mut cap = capabilities();
        cap["images"]["state"] = json!("partial");
        cap["images"]["workflows"][0]["state"] = json!("unavailable");
        cap["images"]["workflows"][0]["reason"] = json!("image_models_unavailable");
        let (url, server) = host(discovery(Router::new(), cap)).await;
        let state = KunoClient::default().discover(&url).await;
        assert_eq!(state.state, "partial");
        assert!(state.workflow(kuno_protocol::QWEN21_RECIPE).is_none());
        assert!(state.workflow(kuno_protocol::QWEN21_EDIT_RECIPE).is_some());
        assert!(state.require_recipe(&recipe()).is_err());
        server.abort();
        for (state, reason) in [
            ("disabled", "image_generation_not_configured"),
            ("unsupported", "image_engine_not_in_this_build"),
            ("unavailable", "image_models_unavailable"),
        ] {
            let mut cap = capabilities();
            cap["images"]["state"] = json!(state);
            cap["images"]["reason"] = json!(reason);
            let (url, server) = host(discovery(Router::new(), cap)).await;
            let found = KunoClient::default().discover(&url).await;
            assert_eq!(found.state, state);
            assert!(found.workflow(kuno_protocol::QWEN21_RECIPE).is_none());
            server.abort();
        }
    }

    #[tokio::test]
    async fn incomplete_out_of_order_invalid_and_duplicate_results_are_never_accepted() {
        let result = json!({"type":"result","mime_type":"image/png","png_base64":png()});
        let invalid = json!({"type":"result","mime_type":"image/png","png_base64":"not-a-png"});
        for text in [
            "{\"type\":\"started\"}\n".into(),
            format!("{result}\n"),
            format!("{{\"type\":\"started\"}}\n{invalid}\n"),
            format!("{{\"type\":\"started\"}}\n{result}\n{result}\n"),
            "{\"type\":\"started\"}\n{\"type\":\"result\"".into(),
        ] {
            let (url, server) = host(stream_app(text)).await;
            assert!(
                KunoClient::default()
                    .generate(&url, &recipe(), |_| ())
                    .await
                    .is_err()
            );
            server.abort();
        }
        let mut bad_preview: Value =
            serde_json::from_str(events().lines().nth(1).unwrap()).unwrap();
        bad_preview["width"] = json!(512);
        let (url, server) = host(stream_app(format!(
            "{{\"type\":\"started\"}}\n{bad_preview}\n{result}\n"
        )))
        .await;
        assert!(
            KunoClient::default()
                .generate(&url, &recipe(), |_| panic!("invalid preview published"))
                .await
                .is_err()
        );
        server.abort();
    }

    #[tokio::test]
    async fn terminal_and_http_errors_are_actionable_without_leaking_backend_content() {
        let (url, server) = host(stream_app(
            "{\"type\":\"started\"}\n{\"type\":\"error\",\"error\":\"PRIVATE BACKEND CONTENT\"}\n"
                .into(),
        ))
        .await;
        let error = KunoClient::default()
            .generate(&url, &recipe(), |_| ())
            .await
            .unwrap_err();
        assert_eq!(error.code(), "image_inference_failed");
        assert!(!error.message().contains("PRIVATE"));
        server.abort();
        for (status, body, code, fragment) in [
            (
                StatusCode::UNAUTHORIZED,
                json!({"error":"private"}),
                "image_access_denied",
                "proxy access",
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"error":"image_models_unavailable"}),
                "image_inference_failed",
                "models",
            ),
        ] {
            let app = discovery(
                Router::new().route(
                    GENERATE_PATH,
                    post(move || {
                        let body = body.clone();
                        async move { (status, Json(body)) }
                    }),
                ),
                capabilities(),
            );
            let (url, server) = host(app).await;
            let error = KunoClient::default()
                .generate(&url, &recipe(), |_| ())
                .await
                .unwrap_err();
            assert_eq!(error.code(), code);
            assert!(error.message().contains(fragment));
            server.abort();
        }
    }

    #[tokio::test]
    async fn old_png_responses_and_unadvertised_attention_require_action_not_fallback() {
        let app = discovery(
            Router::new().route(
                GENERATE_PATH,
                post(|| async {
                    (
                        [("content-type", "image/png")],
                        include_bytes!("fixtures/64.png").as_slice(),
                    )
                }),
            ),
            capabilities(),
        );
        let (url, server) = host(app).await;
        let client = KunoClient::default();
        assert_eq!(
            client
                .generate(&url, &recipe(), |_| ())
                .await
                .unwrap_err()
                .code(),
            "image_stream_unsupported"
        );
        let mut recipe = recipe();
        recipe.attention = kuno_protocol::ImageAttention::Fast;
        assert_eq!(
            client
                .generate(&url, &recipe, |_| ())
                .await
                .unwrap_err()
                .code(),
            "image_attention_unsupported"
        );
        server.abort();
    }

    #[tokio::test]
    async fn discovery_deadline_covers_slow_responses_and_stream_event_size_is_bounded() {
        let app = Router::new().route(
            "/kuno/v1/capabilities",
            get(|| async {
                Body::from_stream(futures_util::stream::pending::<
                    Result<Vec<u8>, std::convert::Infallible>,
                >())
            }),
        );
        let (url, server) = host(app).await;
        let started = Instant::now();
        assert_eq!(
            KunoClient::default().discover(&url).await.state,
            "unreachable"
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        server.abort();
        let (url, server) = host(stream_app(" ".repeat(MAX_EVENT_BYTES + 1))).await;
        assert!(
            KunoClient::default()
                .generate(&url, &recipe(), |_| ())
                .await
                .is_err()
        );
        server.abort();
    }
}
