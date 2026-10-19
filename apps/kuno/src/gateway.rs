use crate::{
    admission::{Admission, Lease},
    config::Config,
    lifecycle::{Outcome, Span, Stage},
    observe::Observer,
    usage::{Catalog, CatalogModel, Usage, UsageSnapshot},
};
use axum::{
    Json, Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use bytes::Bytes;
use http_body::{Body as HttpBody, Frame, SizeHint};
use hyper::body::Incoming;
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use kuno_protocol::{API_VERSION, Status};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};
use tokio::time::Sleep;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestClass {
    Metadata,
    Inference,
    Catalog,
    Opaque,
}

/// Exact method/path allowlist. Unknown methods and paths acquire the same
/// exclusive lease as inference, including future endpoints and mutations.
pub fn classify(method: &Method, path: &str) -> RequestClass {
    if (matches!(*method, Method::GET | Method::HEAD)
        && matches!(
            path,
            "/" | "/api/version" | "/api/tags" | "/api/ps" | "/api/status" | "/v1/models"
        ))
        || (*method == Method::POST && path == "/api/show")
    {
        return RequestClass::Metadata;
    }
    if *method == Method::POST
        && matches!(
            path,
            "/api/chat"
                | "/api/generate"
                | "/api/embed"
                | "/api/embeddings"
                | "/v1/chat/completions"
                | "/v1/completions"
                | "/v1/embeddings"
                | "/v1/responses"
                | "/v1/responses/compact"
                | "/v1/audio/transcriptions"
                | "/v1/messages"
        )
    {
        return RequestClass::Inference;
    }
    if (*method == Method::POST
        && matches!(
            path,
            "/api/pull" | "/api/push" | "/api/create" | "/api/copy"
        ))
        || (*method == Method::DELETE && path == "/api/delete")
    {
        return RequestClass::Catalog;
    }
    RequestClass::Opaque
}

pub struct Gateway {
    config: Config,
    client: Client<HttpConnector, Body>,
    pub admission: Arc<Admission>,
    token: Vec<u8>,
    pub usage: Usage,
    catalog: tokio::sync::Mutex<Option<(std::time::Instant, Catalog)>>,
    managed: Option<crate::pool::Pool>,
    image_shutdown: tokio::sync::watch::Sender<bool>,
    image_requests: Arc<tokio::sync::Semaphore>,
    image_readiness: Arc<crate::image_access::ImageReadiness>,
    image_fast_attention: bool,
    native_runtime: crate::native_text::Runtime,
    image_runtime: crate::image_residency::ImageRuntime,
}

impl Gateway {
    pub fn new(config: Config, token: &str) -> Result<Arc<Self>, &'static str> {
        let admission = Admission::new(
            config.max_waiting,
            Duration::from_secs(config.queue_timeout_seconds),
        );
        Self::with_admission(config, token, admission, false)
    }

    pub fn persistent(config: Config, token: &str) -> Result<Arc<Self>, &'static str> {
        config.validate()?;
        let admission = Admission::persistent(
            config.max_waiting,
            Duration::from_secs(config.queue_timeout_seconds),
            &config.state_dir,
        )?;
        Self::with_admission(config, token, admission, true)
    }

    fn with_admission(
        config: Config,
        token: &str,
        admission: Arc<Admission>,
        persistent: bool,
    ) -> Result<Arc<Self>, &'static str> {
        config.validate()?;
        if token.len() < 32
            || token.len() > 256
            || !token
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err("control_token_must_be_32_to_256_url_safe_characters");
        }
        let mut connector = HttpConnector::new();
        connector.set_connect_timeout(Some(Duration::from_secs(10)));
        let client = Client::builder(TokioExecutor::new())
            // A request may mutate state or start inference. Never replay it.
            .retry_canceled_requests(false)
            // Each request owns its connection: dropping an unfinished body
            // closes that upstream connection instead of recycling it.
            .pool_max_idle_per_host(0)
            .build(connector);
        let usage = Usage::open(persistent.then_some(config.state_dir.as_path()))?;
        let managed = config
            .managed
            .clone()
            .map(|managed| {
                crate::pool::Pool::new(
                    managed,
                    config.upstream,
                    persistent.then_some(config.state_dir.as_path()),
                    config.max_waiting,
                    Duration::from_secs(config.queue_timeout_seconds),
                )
            })
            .transpose()?;
        let image_fast_attention = kuno_image::NATIVE_CUDA_ENABLED
            && config.image.as_ref().is_some_and(|image| {
                config.managed.as_ref().is_some_and(|pool| {
                    pool.workers.iter().any(|worker| {
                        matches!(worker.device, crate::worker::DeviceSelection::Cuda { .. })
                            && worker
                                .device
                                .pci_address()
                                .eq_ignore_ascii_case(&image.pci_address)
                    })
                })
            });
        let image_readiness =
            crate::image_access::ImageReadiness::new(config.image.as_ref(), image_fast_attention);
        Ok(Arc::new(Self {
            config,
            client,
            admission,
            token: format!("Bearer {token}").into_bytes(),
            usage,
            catalog: tokio::sync::Mutex::new(None),
            managed,
            image_shutdown: tokio::sync::watch::channel(false).0,
            image_requests: Arc::new(tokio::sync::Semaphore::new(2)),
            image_readiness,
            image_fast_attention,
            native_runtime: Default::default(),
            image_runtime: Default::default(),
        }))
    }

    pub async fn initialize_workers(&self) -> Result<(), &'static str> {
        // Image prerequisites affect image availability, never healthy Ollama
        // forwarding. Generation checks them again before touching inference.
        self.image_readiness
            .snapshot(self.config.image.clone())
            .await;
        if let Some(pool) = &self.managed {
            pool.initialize().await?;
        }
        Ok(())
    }
    pub fn stop_images(&self) {
        self.image_shutdown.send_replace(true);
    }
    pub async fn evict_native_text(&self) -> Result<(), &'static str> {
        self.native_runtime.evict().await
    }
    pub async fn evict_images(&self) -> Result<(), &'static str> {
        self.image_runtime.evict().await
    }

    pub fn compatibility(self: &Arc<Self>) -> Router {
        // Trusted-LAN inference is keyless, including images and discovery.
        // Local privileged controls retain their separate authentication.
        let images = Router::new()
            .route("/kuno/v1/capabilities", get(capabilities))
            .route("/kuno/v1/images/generate", post(generate_image))
            .layer(middleware::from_fn(inference_guard));
        images
            .fallback(compatibility_fallback)
            .with_state(self.clone())
    }

    pub fn control(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/kuno/v1/status", get(status))
            .route("/kuno/v1/usage", get(usage))
            .route("/kuno/v1/resources", get(resources))
            .route("/kuno/v1/learning", get(learning))
            .route("/kuno/v1/admission/pause", post(pause))
            .route("/kuno/v1/admission/resume", post(resume))
            .fallback(|| async { error(StatusCode::NOT_FOUND, "unknown_control_route") })
            .layer(middleware::from_fn_with_state(self.clone(), authenticate))
            .with_state(self.clone())
    }

    async fn catalog_model(&self, name: &str) -> Option<CatalogModel> {
        let mut cache = self.catalog.lock().await;
        if cache
            .as_ref()
            .is_none_or(|(when, _)| when.elapsed() > Duration::from_secs(60))
        {
            let request = Request::builder()
                .uri(format!("http://{}/api/tags", self.config.upstream))
                .body(Body::empty())
                .ok()?;
            let fetched = tokio::time::timeout(Duration::from_secs(2), async {
                let response = self.client.request(request).await.ok()?;
                if !response.status().is_success() {
                    return None;
                }
                let bytes = axum::body::to_bytes(Body::new(response.into_body()), 2 * 1024 * 1024)
                    .await
                    .ok()?;
                let catalog: Catalog = serde_json::from_slice(&bytes).ok()?;
                Some(catalog)
            })
            .await
            .ok()
            .flatten()?;
            *cache = Some((std::time::Instant::now(), fetched));
        }
        cache.as_ref()?.1.resolve(name).cloned()
    }
}

async fn resources(
    State(gateway): State<Arc<Gateway>>,
) -> Json<crate::resources::ResourceSnapshot> {
    Json(crate::resources::snapshot(gateway.config.upstream).await)
}

async fn learning(State(gateway): State<Arc<Gateway>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "api_version": API_VERSION,
        "persistent": false,
        "blacklisting_enabled": false,
        "workers": gateway.managed.as_ref().map(|p| p.learning()).unwrap_or_default(),
    }))
}

async fn usage(State(gateway): State<Arc<Gateway>>) -> Json<UsageSnapshot> {
    Json(gateway.usage.snapshot())
}

async fn authenticate(
    State(gateway): State<Arc<Gateway>>,
    request: Request,
    next: Next,
) -> Response {
    let supplied = request
        .headers()
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    let expected = &gateway.token;
    // Compare all bytes of a valid-length token without prefix-dependent exits.
    let valid = supplied.len() == expected.len()
        && supplied
            .iter()
            .zip(expected)
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0;
    if !valid {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    next.run(request).await
}

async fn status(State(gateway): State<Arc<Gateway>>) -> Json<Status> {
    let mut admission = gateway.admission.status();
    admission.recovery_required |= gateway
        .managed
        .as_ref()
        .is_some_and(|p| p.recovery_required());
    Json(Status {
        api_version: API_VERSION,
        mode: if gateway.managed.is_some() {
            "managed_experimental"
        } else {
            "gateway_only"
        }
        .into(),
        admission,
        gpu_placement_verified: false,
        image_generation: gateway.config.image.as_ref().is_some_and(|image| {
            cfg!(feature = "native-image")
                || image.engine == crate::image_engine::ImageEngine::ManagedSycl
        }),
    })
}

async fn capabilities(State(gateway): State<Arc<Gateway>>) -> Response {
    let images = gateway
        .image_readiness
        .snapshot(gateway.config.image.clone())
        .await;
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(kuno_protocol::Capabilities {
            service: "kuno".into(),
            discovery_version: 1,
            version: env!("CARGO_PKG_VERSION").into(),
            inference_auth: "none".into(),
            images,
        }),
    )
        .into_response()
}

async fn inference_guard(request: Request, next: Next) -> Response {
    // Vashti connects server-to-server. Do not grant cross-origin browser access
    // to keyless Kuno-owned routes; leave Ollama's origin policy untouched.
    if request.headers().contains_key(header::ORIGIN) {
        return error(StatusCode::FORBIDDEN, "inference_origin_forbidden");
    }
    if request.method() == Method::POST {
        let json = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(';').next())
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"));
        if !json {
            return error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "image_content_type_required",
            );
        }
    }
    next.run(request).await
}

async fn compatibility_fallback(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    if request.uri().path() == "/kuno" || request.uri().path().starts_with("/kuno/") {
        return error(StatusCode::NOT_FOUND, "unknown_inference_route");
    }
    proxy(State(gateway), request).await
}

async fn image_recipe(
    request: Request,
    privacy: impl std::future::Future<Output = Result<(), &'static str>>,
) -> Result<kuno_protocol::ImageRecipe, (StatusCode, &'static str)> {
    privacy
        .await
        .map_err(|code| (StatusCode::SERVICE_UNAVAILABLE, code))?;
    let bytes = match tokio::time::timeout(
        Duration::from_secs(60),
        axum::body::to_bytes(request.into_body(), kuno_protocol::MAX_IMAGE_REQUEST_BYTES),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => return Err((StatusCode::PAYLOAD_TOO_LARGE, "image_request_too_large")),
        Err(_) => return Err((StatusCode::REQUEST_TIMEOUT, "image_request_timeout")),
    };
    serde_json::from_slice(&bytes).map_err(|_| (StatusCode::BAD_REQUEST, "image_request_invalid"))
}

async fn generate_image(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    let Some(config) = gateway.config.image.clone() else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "image_generation_not_configured",
        );
    };
    if !cfg!(feature = "native-image") && config.engine == crate::image_engine::ImageEngine::Native
    {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "image_engine_not_in_this_build",
        );
    }
    // Bound content-bearing memory before reading large multi-image requests.
    let slot = match gateway.image_requests.clone().try_acquire_owned() {
        Ok(slot) => Arc::new(slot),
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image_queue_full"),
    };
    let recipe = match image_recipe(request, crate::image_access::require_privacy()).await {
        Ok(recipe) => recipe,
        Err((status, code)) => return error(status, code),
    };
    if let Err(code) = recipe.validate_execution() {
        return error(StatusCode::BAD_REQUEST, code);
    }
    if recipe.attention == kuno_protocol::ImageAttention::Fast
        && (!gateway.image_fast_attention || config.uses_sycl(&recipe.workflow))
    {
        return error(StatusCode::BAD_REQUEST, "image_attention_unsupported");
    }
    if !config
        .collections
        .iter()
        .any(|id| id == recipe.collection_id())
    {
        return error(StatusCode::BAD_REQUEST, "image_collection_not_enabled");
    }
    if recipe.seed.parse::<i64>().is_err() {
        return error(
            StatusCode::BAD_REQUEST,
            "native_image_seed_range_unsupported",
        );
    }
    if let Err(code) = crate::image_access::require_workflow(
        config.clone(),
        &recipe.workflow,
        gateway.image_fast_attention,
    )
    .await
    {
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    let mut lease = match gateway.admission.acquire().await {
        Ok(lease) => lease,
        Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
    };
    let (previews, stream_previews) = tokio::sync::watch::channel(None);
    let (mut send, receive) = tokio::sync::oneshot::channel();
    let task_slot = slot.clone();
    // The task owns cleanup even when HTTP drops the waiting handler. No next
    // model may enter until the native process is reaped, or recovery is set.
    tokio::spawn(async move {
        let _request_slot = task_slot;
        let result = async {
            if send.is_closed() {
                return Err("image_cancelled");
            }
            lease.dispatched()?;
            let pool = gateway
                .managed
                .as_ref()
                .ok_or("image_requires_managed_pool")?;
            gateway.native_runtime.evict().await?;
            pool.prepare_image(&config.pci_address).await?;
            let executable = std::env::current_exe().map_err(|_| "executable_unavailable")?;
            let finished = gateway
                .image_runtime
                .run(
                    &executable,
                    &config,
                    recipe,
                    &mut send,
                    gateway.image_shutdown.subscribe(),
                    previews,
                )
                .await;
            if finished.settled
                && !finished.forced
                && !matches!(finished.output, Err("image_vulkan_device_lost"))
            {
                lease.complete();
            }
            finished.output
        }
        .await;
        drop(lease);
        let _ = send.send(result);
    });
    crate::image_stream::response(receive, stream_previews, slot)
}

async fn pause(State(gateway): State<Arc<Gateway>>) -> Json<Status> {
    gateway.admission.pause(true);
    status(State(gateway)).await
}

async fn resume(State(gateway): State<Arc<Gateway>>) -> Json<Status> {
    gateway.admission.pause(false);
    status(State(gateway)).await
}

pub fn error(status: StatusCode, code: &'static str) -> Response {
    (status, Json(serde_json::json!({"error": code}))).into_response()
}

/// Releases admission only when a managed job proved its worker group is gone.
/// An unproven group keeps the reservation quarantined and the upstream
/// uncertain, so the installed supervisor stops the service instead.
async fn finish_uncertain(lease: Option<Lease>, job: Option<crate::managed::Job>) {
    if let Some(job) = job
        && job.abort().await.is_ok()
        && let Some(mut lease) = lease
    {
        lease.complete();
    }
}

fn response_deadline(seconds: u64) -> Option<Duration> {
    (seconds != 0).then(|| Duration::from_secs(seconds))
}

async fn optional_timeout<F: std::future::Future>(
    limit: Option<Duration>,
    future: F,
) -> Result<F::Output, tokio::time::error::Elapsed> {
    match limit {
        Some(limit) => tokio::time::timeout(limit, future).await,
        None => Ok(future.await),
    }
}

fn strip_hop_headers(headers: &mut HeaderMap) {
    let nominated: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|v| HeaderName::from_bytes(v.trim().as_bytes()).ok())
        .collect();
    for name in nominated {
        headers.remove(name);
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    ] {
        headers.remove(name);
    }
}

async fn proxy(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    if request.method() == Method::CONNECT || request.headers().contains_key(header::UPGRADE) {
        return error(StatusCode::NOT_IMPLEMENTED, "protocol_upgrade_unsupported");
    }
    if gateway.config.native_text.is_some()
        && request.method() == Method::GET
        && matches!(request.uri().path(), "/api/tags" | "/v1/models")
    {
        return native_catalog(gateway, request).await;
    }
    let class = classify(request.method(), request.uri().path());
    if request.method() == Method::GET
        && request.uri().path() == "/api/ps"
        && let Some(pool) = &gateway.managed
    {
        return match pool.running().await {
            Ok(mut report) => {
                if gateway.native_runtime.resident()
                    && let Some(config) = &gateway.config.native_text
                    && let Some(models) = report["models"].as_array_mut()
                {
                    models.push(config.metadata());
                }
                Json(report).into_response()
            }
            Err(code) => error(StatusCode::SERVICE_UNAVAILABLE, code),
        };
    }
    let lease = if class != RequestClass::Metadata {
        match gateway.admission.acquire().await {
            Ok(lease) => Some(lease),
            Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
        }
    } else {
        None
    };

    // Once admitted, keep ownership until the upstream response completes even
    // if the downstream disconnects before receiving headers. The body wrapper
    // drains abandoned responses without retaining their contents.
    tokio::spawn(forward(gateway, request, class, lease))
        .await
        .unwrap_or_else(|_| error(StatusCode::BAD_GATEWAY, "upstream_task_failed"))
}

async fn forward(
    gateway: Arc<Gateway>,
    request: Request,
    class: RequestClass,
    mut lease: Option<Lease>,
) -> Response {
    let (mut parts, mut body) = request.into_parts();
    if let Some(config) = &gateway.config.native_text {
        if matches!(
            parts.uri.path(),
            "/api/chat"
                | "/api/generate"
                | "/api/embed"
                | "/api/embeddings"
                | "/api/show"
                | "/api/create"
                | "/api/delete"
                | "/api/pull"
                | "/api/push"
                | "/api/copy"
                | "/v1/chat/completions"
                | "/v1/completions"
                | "/v1/embeddings"
                | "/v1/responses"
                | "/v1/messages"
        ) {
            let bytes = match tokio::time::timeout(
                Duration::from_secs(300),
                axum::body::to_bytes(body, 64 * 1024 * 1024),
            )
            .await
            {
                Ok(Ok(b)) => b,
                _ => return error(StatusCode::BAD_REQUEST, "native_request_body_unavailable"),
            };
            if let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                if ["model", "name", "source", "destination"]
                    .iter()
                    .any(|key| {
                        value.get(key).and_then(serde_json::Value::as_str)
                            == Some(config.model.as_str())
                    })
                {
                    if parts.uri.path() == "/api/show" {
                        return Json(config.show()).into_response();
                    }
                    return native_inference(gateway, value, parts.uri.path(), lease).await;
                }
                crate::native_text::scrub_json(&mut value);
            }
            body = Body::from(bytes);
        }
        if class != RequestClass::Metadata
            && let Err(code) = gateway.native_runtime.evict().await
        {
            return error(StatusCode::SERVICE_UNAVAILABLE, code);
        }
    }
    let mut managed_job = None;
    let mut address = gateway.config.upstream;
    if let Some(pool) = &gateway.managed {
        if class == RequestClass::Inference && crate::pool::Pool::supports(parts.uri.path()) {
            let bytes = match tokio::time::timeout(
                // Bound receiving the request separately from inference. A
                // disabled response deadline must not mean an immediate
                // upload timeout, or change the existing upload allowance.
                Duration::from_secs(300),
                axum::body::to_bytes(body, 64 * 1024 * 1024),
            )
            .await
            {
                Ok(Ok(bytes)) => bytes,
                _ => {
                    return error(
                        StatusCode::BAD_REQUEST,
                        "managed_request_body_unavailable_or_too_large",
                    );
                }
            };
            let mut value: serde_json::Value = match serde_json::from_slice(&bytes) {
                Ok(value) => value,
                Err(_) => return error(StatusCode::BAD_REQUEST, "managed_request_json_invalid"),
            };
            let model = value
                .get("model")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let unload = value.get("keep_alive").and_then(serde_json::Value::as_i64) == Some(0)
                && value.get("prompt").is_none_or(|v| v.as_str() == Some(""))
                && value
                    .get("messages")
                    .is_none_or(|v| v.as_array().is_some_and(|a| a.is_empty()))
                && value.get("input").is_none();
            if !unload
                && let Some(name) = model
                && let Some(model) = gateway.catalog_model(&name).await
                && !model.is_remote()
            {
                let resident_device = gateway.image_runtime.resident_device().await;
                let prepared = if let Some(pci) = resident_device {
                    match pool
                        .prepare_with_resident_image(
                            parts.uri.path(),
                            &mut value,
                            &model.name,
                            &model.digest,
                            &pci,
                        )
                        .await
                    {
                        Err("image_resident_needs_eviction") => {
                            if let Err(code) = gateway.evict_images().await {
                                if let Some(lease) = &mut lease {
                                    let _ = lease.dispatched();
                                }
                                return error(StatusCode::SERVICE_UNAVAILABLE, code);
                            }
                            pool.prepare(parts.uri.path(), &mut value, &model.name, &model.digest)
                                .await
                        }
                        result => result,
                    }
                } else {
                    pool.prepare(parts.uri.path(), &mut value, &model.name, &model.digest)
                        .await
                };
                match prepared {
                    Ok(job) => {
                        address = job.address();
                        managed_job = Some(job);
                    }
                    Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
                }
            } else {
                if let Err(code) = gateway.evict_images().await {
                    if let Some(lease) = &mut lease {
                        let _ = lease.dispatched();
                    }
                    return error(StatusCode::SERVICE_UNAVAILABLE, code);
                }
                if let Err(code) = pool.stop_all().await {
                    return error(StatusCode::SERVICE_UNAVAILABLE, code);
                }
            }
            let encoded = match serde_json::to_vec(&value) {
                Ok(bytes) => bytes,
                Err(_) => return error(StatusCode::BAD_REQUEST, "managed_request_json_invalid"),
            };
            parts.headers.remove(header::CONTENT_LENGTH);
            body = Body::from(encoded);
        } else if class != RequestClass::Metadata {
            if let Err(code) = gateway.evict_images().await {
                if let Some(lease) = &mut lease {
                    let _ = lease.dispatched();
                }
                return error(StatusCode::SERVICE_UNAVAILABLE, code);
            }
            if let Err(code) = pool.stop_all().await {
                return error(StatusCode::SERVICE_UNAVAILABLE, code);
            }
        }
    } else if class != RequestClass::Metadata
        && let Err(code) = gateway.evict_images().await
    {
        if let Some(lease) = &mut lease {
            let _ = lease.dispatched();
        }
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    let observation =
        (class == RequestClass::Inference).then(|| Arc::new(Mutex::new(Observer::default())));
    let body = if let Some(observation) = &observation {
        Body::new(ObservedBody {
            inner: body,
            observation: observation.clone(),
        })
    } else {
        body
    };
    let path = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    parts.uri = match format!("http://{address}{path}").parse() {
        Ok(uri) => uri,
        Err(_) => return error(StatusCode::BAD_REQUEST, "invalid_request_target"),
    };
    strip_hop_headers(&mut parts.headers);
    parts.headers.insert(
        header::HOST,
        HeaderValue::from_str(&address.to_string()).unwrap(),
    );
    if let Some(lease) = &mut lease
        && let Err(code) = lease.dispatched()
    {
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    let headers =
        (class != RequestClass::Metadata).then(|| Span::new(Stage::UpstreamHeaders, None, None));
    let upstream = match optional_timeout(
        response_deadline(gateway.config.response_header_timeout_seconds),
        gateway.client.request(Request::from_parts(parts, body)),
    )
    .await
    {
        Ok(Ok(response)) => {
            if let Some(span) = headers {
                span.finish(if response.status().is_success() {
                    Outcome::Success
                } else {
                    Outcome::Rejected
                });
            }
            response
        }
        // The request may already have reached the upstream. A managed worker
        // can prove that it stopped; an opaque upstream stays uncertain.
        Ok(Err(_)) => {
            if let Some(span) = headers {
                span.finish(Outcome::TransportError);
            }
            finish_uncertain(lease, managed_job).await;
            return error(StatusCode::BAD_GATEWAY, "upstream_transport_failed");
        }
        Err(_) => {
            if let Some(span) = headers {
                span.finish(Outcome::Timeout);
            }
            finish_uncertain(lease, managed_job).await;
            return error(StatusCode::GATEWAY_TIMEOUT, "upstream_headers_timeout");
        }
    };
    let (mut parts, body) = upstream.into_parts();
    if parts.status.is_success()
        && let Some(observation) = observation
    {
        let model = observation.lock().unwrap().model().map(str::to_owned);
        if let Some(name) = model {
            let model = gateway.catalog_model(&name).await;
            gateway.usage.record(model.as_ref(), crate::usage::now());
        }
    }
    // Mutations and opaque endpoints can change model artifacts. Never reuse a
    // pre-mutation identity for the next user request.
    if matches!(class, RequestClass::Catalog | RequestClass::Opaque) {
        *gateway.catalog.lock().await = None;
    }
    strip_hop_headers(&mut parts.headers);
    let idle = response_deadline(gateway.config.stream_idle_timeout_seconds);
    let body = LeasedBody {
        inner: Some(body),
        lease,
        managed_job,
        idle,
        timer: idle.map(|limit| Box::pin(tokio::time::sleep(limit))),
        settled: false,
        uncertain: false,
        lifecycle: (class != RequestClass::Metadata)
            .then(|| Span::new(Stage::UpstreamBody, None, None)),
    };
    Response::from_parts(parts, Body::new(body))
}

struct ObservedBody {
    inner: Body,
    observation: Arc<Mutex<Observer>>,
}

impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let frame = Pin::new(&mut self.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &frame
            && let Some(bytes) = frame.data_ref()
        {
            self.observation.lock().unwrap().feed(bytes);
        }
        frame
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

/// Ties admission to the upstream body's lifetime, not the response headers.
/// Interrupted bodies mark the upstream uncertain before releasing the permit.
struct LeasedBody {
    inner: Option<Incoming>,
    lease: Option<Lease>,
    managed_job: Option<crate::managed::Job>,
    idle: Option<Duration>,
    timer: Option<Pin<Box<Sleep>>>,
    settled: bool,
    uncertain: bool,
    lifecycle: Option<Span>,
}

impl LeasedBody {
    fn observed(&mut self, outcome: Outcome) {
        if let Some(span) = self.lifecycle.take() {
            span.finish(outcome);
        }
    }

    fn finish(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        self.observed(Outcome::Success);
        if let Some(job) = self.managed_job.take() {
            job.complete();
        }
        if let Some(mut lease) = self.lease.take() {
            lease.complete();
        }
    }
}

impl HttpBody for LeasedBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        match Pin::new(self.inner.as_mut().expect("response body present")).poll_frame(cx) {
            Poll::Ready(None) => {
                if !self.uncertain {
                    self.finish();
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(idle) = self.idle
                    && let Some(timer) = self.timer.as_mut()
                {
                    timer.as_mut().reset(tokio::time::Instant::now() + idle);
                }
                if !self.uncertain && self.inner.as_ref().is_none_or(HttpBody::is_end_stream) {
                    self.finish();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(_))) => {
                self.uncertain = true;
                self.observed(Outcome::BodyError);
                Poll::Ready(Some(Err(std::io::Error::other("upstream_body_failed"))))
            }
            Poll::Pending => {
                if self
                    .timer
                    .as_mut()
                    .is_some_and(|timer| timer.as_mut().poll(cx).is_ready())
                {
                    self.uncertain = true;
                    self.observed(Outcome::Timeout);
                    Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "upstream_body_timeout",
                    ))))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.as_ref().is_none_or(HttpBody::is_end_stream)
    }
    fn size_hint(&self) -> SizeHint {
        self.inner
            .as_ref()
            .map(HttpBody::size_hint)
            .unwrap_or_default()
    }
}

impl Drop for LeasedBody {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        if !self.uncertain && self.inner.as_ref().is_none_or(HttpBody::is_end_stream) {
            self.finish();
            return;
        }
        self.settled = true;
        self.observed(Outcome::Cancelled);
        let Some(mut lease) = self.lease.take() else {
            return;
        };
        let job = self.managed_job.take();
        let body = if self.uncertain {
            None
        } else {
            self.inner.take()
        };
        let uncertain = self.uncertain;
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let limit = self.idle;
        runtime.spawn(async move {
            let completed = if uncertain {
                false
            } else if let Some(mut body) = body {
                let drain = Span::new(Stage::UpstreamDrain, None, None);
                // Discarding until EOF proves completion without retaining or
                // replaying content. Only an explicitly configured idle limit
                // applies; continuing response progress resets that limit.
                let result = async {
                    while let Some(frame) = optional_timeout(
                        limit,
                        std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)),
                    )
                    .await?
                    {
                        if frame.is_err() {
                            return Ok::<_, tokio::time::error::Elapsed>(false);
                        }
                    }
                    Ok(true)
                }
                .await;
                drain.finish(match result {
                    Ok(true) => Outcome::Success,
                    Ok(false) => Outcome::BodyError,
                    Err(_) => Outcome::Timeout,
                });
                result.unwrap_or(false)
            } else {
                false
            };
            if completed {
                lease.complete();
                if let Some(job) = job {
                    job.complete();
                }
            } else if let Some(job) = job
                && job.abort().await.is_ok()
            {
                lease.complete();
            }
        });
    }
}

async fn native_catalog(gateway: Arc<Gateway>, request: Request) -> Response {
    let path = request.uri().path().to_owned();
    let (mut parts, body) = request.into_parts();
    let target = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    parts.uri = match format!("http://{}{}", gateway.config.upstream, target).parse() {
        Ok(uri) => uri,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "upstream_uri_invalid"),
    };
    strip_hop_headers(&mut parts.headers);
    parts.headers.insert(
        header::HOST,
        HeaderValue::from_str(&gateway.config.upstream.to_string()).unwrap(),
    );
    let response = match gateway
        .client
        .request(Request::from_parts(parts, body))
        .await
    {
        Ok(r) => r,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "upstream_connection_failed"),
    };
    if !response.status().is_success() {
        return response.map(Body::new);
    }
    let bytes = match axum::body::to_bytes(Body::new(response.into_body()), 2 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "native_catalog_unavailable"),
    };
    let mut value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "native_catalog_invalid"),
    };
    let config = gateway.config.native_text.as_ref().unwrap();
    let key = if path == "/api/tags" {
        "models"
    } else {
        "data"
    };
    let Some(models) = value.get_mut(key).and_then(serde_json::Value::as_array_mut) else {
        return error(StatusCode::BAD_GATEWAY, "native_catalog_invalid");
    };
    if !models
        .iter()
        .any(|m| m["name"] == config.model || m["id"] == config.model)
    {
        models.push(if key=="models" {config.metadata()} else {serde_json::json!({"id":config.model,"object":"model","owned_by":"kuno","created":1792368000})});
    }
    Json(value).into_response()
}

async fn native_inference(
    gateway: Arc<Gateway>,
    value: serde_json::Value,
    path: &str,
    mut lease: Option<Lease>,
) -> Response {
    let mut value = value;
    let prepared = crate::native_text::prepare(path, &value);
    crate::native_text::scrub_json(&mut value);
    drop(value);
    let prepared = match prepared {
        Ok(p) => p,
        Err(code) => return error(StatusCode::BAD_REQUEST, code),
    };
    let config = gateway.config.native_text.clone().unwrap();
    let chat = prepared.chat;
    let stream = prepared.stream;
    if let Err(code) = gateway.evict_images().await {
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    if let Some(pool) = &gateway.managed
        && let Err(code) = pool.prepare_native_text().await
    {
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    if let Some(lease) = &mut lease
        && let Err(code) = lease.dispatched()
    {
        return error(StatusCode::SERVICE_UNAVAILABLE, code);
    }
    let Some(request) = prepared.request else {
        let result = gateway.native_runtime.evict().await;
        // evict only returns after wait/reap; a live process retains uncertainty.
        if result != Err("native_worker_reap_failed")
            && let Some(lease) = &mut lease
        {
            lease.complete();
        }
        return match result {
            Ok(()) => {
                let mut v = crate::native_text::chunk(&config.model, chat, "", None);
                v["done"] = true.into();
                v["done_reason"] = "unload".into();
                Json(v).into_response()
            }
            Err(code) => error(StatusCode::SERVICE_UNAVAILABLE, code),
        };
    };
    let preload = request.outputs == 0;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
    let task_gateway = gateway.clone();
    let model = config.model.clone();
    tokio::spawn(async move {
        let settled = task_gateway
            .native_runtime
            .run(&config, request, sender)
            .await;
        let closed = if !prepared.keep_alive {
            task_gateway.native_runtime.evict().await
        } else {
            Ok(())
        };
        if settled
            && closed != Err("native_worker_reap_failed")
            && let Some(lease) = &mut lease
        {
            lease.complete();
        }
    });
    if !stream {
        let mut text = zeroize::Zeroizing::new(String::with_capacity(8 * 1024 * 1024));
        while let Some(event) = receiver.recv().await {
            match event {
                crate::native_text::Event::Text(bytes) => {
                    let part = match std::str::from_utf8(&bytes) {
                        Ok(v) => v,
                        Err(_) => {
                            return error(StatusCode::BAD_GATEWAY, "native_worker_utf8_invalid");
                        }
                    };
                    if text.len() + part.len() > 8 * 1024 * 1024 {
                        return error(StatusCode::BAD_GATEWAY, "native_response_too_large");
                    }
                    text.push_str(part);
                }
                crate::native_text::Event::Done(metrics) => {
                    let mut v = crate::native_text::chunk(&model, chat, &text, Some(&metrics));
                    if preload {
                        v["done_reason"] = "load".into();
                    }
                    return Json(v).into_response();
                }
                crate::native_text::Event::Failed(code) => {
                    return error(crate::native_text::error_status(code), code);
                }
            }
        }
        return error(StatusCode::BAD_GATEWAY, "native_worker_response_incomplete");
    }
    // Hold headers through startup/load: an OOM before the first token is a
    // real HTTP503. Later stream failures use the ordinary NDJSON error form.
    let first = match receiver.recv().await {
        Some(crate::native_text::Event::Failed(code)) => {
            return error(crate::native_text::error_status(code), code);
        }
        Some(event) => event,
        None => return error(StatusCode::BAD_GATEWAY, "native_worker_response_incomplete"),
    };
    let events = futures_util::stream::unfold(
        (receiver, model, Some(first)),
        move |(mut receiver, model, mut first)| async move {
            let event = match first.take() {
                Some(event) => event,
                None => receiver.recv().await?,
            };
            let mut value = match event {
                crate::native_text::Event::Text(bytes) => match std::str::from_utf8(&bytes) {
                    Ok(text) => crate::native_text::chunk(&model, chat, text, None),
                    Err(_) => serde_json::json!({"error":"native_worker_utf8_invalid"}),
                },
                crate::native_text::Event::Done(metrics) => {
                    let mut v = crate::native_text::chunk(&model, chat, "", Some(&metrics));
                    if preload {
                        v["done_reason"] = "load".into();
                    }
                    v
                }
                crate::native_text::Event::Failed(code) => serde_json::json!({"error":code}),
            };
            let mut encoded = serde_json::to_vec(&value).unwrap();
            crate::native_text::scrub_json(&mut value);
            encoded.push(b'\n');
            Some((
                Ok::<_, std::convert::Infallible>(encoded),
                (receiver, model, first),
            ))
        },
    );
    (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Body::from_stream(events),
    )
        .into_response()
}

#[cfg(test)]
mod response_deadline_tests {
    use super::*;

    #[tokio::test]
    async fn privacy_failure_never_polls_the_content_body() {
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = polls.clone();
        let body = Body::from_stream(futures_util::stream::poll_fn(move |_| {
            seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Poll::Ready(Some(Ok::<_, std::convert::Infallible>(Bytes::from_static(
                b"{}",
            ))))
        }));
        let request = Request::builder().body(body).unwrap();
        let result = image_recipe(request, async { Err("image_privacy_unavailable") }).await;
        assert_eq!(
            result.err(),
            Some((StatusCode::SERVICE_UNAVAILABLE, "image_privacy_unavailable"))
        );
        assert_eq!(polls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn disabled_deadline_allows_long_work_and_preserves_backend_errors() {
        let future = async {
            tokio::time::sleep(Duration::from_secs(3_600)).await;
            Err::<(), _>("backend_failed")
        };
        assert_eq!(
            optional_timeout(response_deadline(0), future)
                .await
                .unwrap(),
            Err("backend_failed")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn explicit_deadline_still_expires() {
        assert!(
            optional_timeout(response_deadline(300), std::future::pending::<()>())
                .await
                .is_err()
        );
    }
}
