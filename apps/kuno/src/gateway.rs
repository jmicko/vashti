use crate::{
    admission::{Admission, Lease},
    config::Config,
    observe::Observer,
    usage::{Catalog, CatalogModel, Usage, UsageSnapshot},
};
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Request, State},
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
    image_previews: Mutex<
        std::collections::HashMap<
            String,
            tokio::sync::watch::Sender<Option<kuno_image::preview::Preview>>,
        >,
    >,
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
            image_previews: Mutex::new(Default::default()),
        }))
    }

    pub async fn initialize_workers(&self) -> Result<(), &'static str> {
        if let Some(image) = &self.config.image {
            if !cfg!(feature = "native-image") {
                return Err("image_engine_not_in_this_build");
            }
            crate::image_privacy::require_no_swap()?;
            crate::image_privacy::require_landlock()?;
            for id in &image.collections {
                crate::collections::ready(&image.models, id)?;
            }
        }
        if let Some(pool) = &self.managed {
            pool.initialize().await?;
        }
        Ok(())
    }
    pub fn stop_images(&self) {
        self.image_shutdown.send_replace(true);
    }

    pub fn compatibility(self: &Arc<Self>) -> Router {
        // Kuno's content endpoint is authenticated even on the compatibility
        // listener. Existing Ollama routes retain their original semantics.
        let images = Router::new()
            .route("/kuno/v1/images/generate", post(generate_image))
            .route("/kuno/v1/images/previews/{id}", get(image_preview))
            .layer(middleware::from_fn_with_state(self.clone(), authenticate));
        images.fallback(proxy).with_state(self.clone())
    }

    pub fn control(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/kuno/v1/status", get(status))
            .route("/kuno/v1/usage", get(usage))
            .route("/kuno/v1/resources", get(resources))
            .route("/kuno/v1/learning", get(learning))
            .route("/kuno/v1/images/generate", post(generate_image))
            .route("/kuno/v1/images/previews/{id}", get(image_preview))
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
        image_generation: cfg!(feature = "native-image") && gateway.config.image.is_some(),
    })
}

// Keys are random, per-call capabilities supplied over authenticated transport.
// Entries and their sole latest frame disappear when the request ends.
struct PreviewScope {
    gateway: Arc<Gateway>,
    id: String,
}
impl Drop for PreviewScope {
    fn drop(&mut self) {
        if let Ok(mut entries) = self.gateway.image_previews.lock() {
            entries.remove(&self.id);
        }
    }
}
async fn image_preview(State(gateway): State<Arc<Gateway>>, Path(id): Path<String>) -> Response {
    let frame = gateway
        .image_previews
        .lock()
        .ok()
        .and_then(|entries| entries.get(&id).map(|sender| sender.borrow().clone()));
    match frame {
        Some(frame) => ([(header::CACHE_CONTROL, "no-store")], Json(frame)).into_response(),
        None => error(StatusCode::NOT_FOUND, "image_preview_unavailable"),
    }
}

async fn generate_image(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    let Some(config) = gateway
        .config
        .image
        .clone()
        .filter(|_| cfg!(feature = "native-image"))
    else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "image_generation_not_configured",
        );
    };
    // Bound content-bearing memory before reading large multi-image requests.
    let slot = match gateway.image_requests.clone().try_acquire_owned() {
        Ok(slot) => slot,
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image_queue_full"),
    };
    let preview_id = match request.headers().get("x-kuno-image-preview-id") {
        None => None,
        Some(value) => match value.to_str() {
            Ok(id) if id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()) => {
                Some(id.to_owned())
            }
            _ => return error(StatusCode::BAD_REQUEST, "image_preview_id_invalid"),
        },
    };
    let bytes = match tokio::time::timeout(
        Duration::from_secs(60),
        axum::body::to_bytes(request.into_body(), kuno_protocol::MAX_IMAGE_REQUEST_BYTES),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => return error(StatusCode::PAYLOAD_TOO_LARGE, "image_request_too_large"),
        Err(_) => return error(StatusCode::REQUEST_TIMEOUT, "image_request_timeout"),
    };
    let recipe: kuno_protocol::ImageRecipe = match serde_json::from_slice(&bytes) {
        Ok(recipe) => recipe,
        Err(_) => return error(StatusCode::BAD_REQUEST, "image_request_invalid"),
    };
    drop(bytes);
    if let Err(code) = recipe.validate_execution() {
        return error(StatusCode::BAD_REQUEST, code);
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
    let mut lease = match gateway.admission.acquire().await {
        Ok(lease) => lease,
        Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
    };
    let (preview_scope, previews) = if let Some(id) = preview_id {
        let mut entries = gateway.image_previews.lock().unwrap();
        if entries.contains_key(&id) {
            return error(StatusCode::CONFLICT, "image_preview_id_in_use");
        }
        let sender = tokio::sync::watch::channel(None).0;
        entries.insert(id.clone(), sender.clone());
        (
            Some(PreviewScope {
                gateway: gateway.clone(),
                id,
            }),
            Some(sender),
        )
    } else {
        (None, None)
    };
    let (mut send, receive) = tokio::sync::oneshot::channel();
    // The task owns cleanup even when HTTP drops the waiting handler. No next
    // model may enter until the native process is reaped, or recovery is set.
    tokio::spawn(async move {
        let _request_slot = slot;
        let _preview_scope = preview_scope;
        let result = async {
            if send.is_closed() {
                return Err("image_cancelled");
            }
            lease.dispatched()?;
            let pool = gateway
                .managed
                .as_ref()
                .ok_or("image_requires_managed_pool")?;
            pool.prepare_image(&config.pci_address).await?;
            let executable = std::env::current_exe().map_err(|_| "executable_unavailable")?;
            let finished = crate::image_runtime::run_with_previews(
                &executable,
                &config,
                recipe,
                &mut send,
                Some(gateway.image_shutdown.subscribe()),
                previews,
            )
            .await;
            if finished.reaped
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
    match receive.await {
        Ok(Ok(png)) => (
            [
                (header::CONTENT_TYPE, "image/png"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            png,
        )
            .into_response(),
        Ok(Err(code)) => error(
            if code.starts_with("image_reference_") {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            },
            code,
        ),
        Err(_) => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "image_worker_supervisor_failed",
        ),
    }
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
    let class = classify(request.method(), request.uri().path());
    if request.method() == Method::GET
        && request.uri().path() == "/api/ps"
        && let Some(pool) = &gateway.managed
    {
        return match pool.running().await {
            Ok(report) => Json(report).into_response(),
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
    let mut managed_job = None;
    let mut address = gateway.config.upstream;
    if let Some(pool) = &gateway.managed {
        if class == RequestClass::Inference && crate::pool::Pool::supports(parts.uri.path()) {
            let bytes = match tokio::time::timeout(
                Duration::from_secs(gateway.config.response_header_timeout_seconds),
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
                match pool
                    .prepare(parts.uri.path(), &mut value, &model.name, &model.digest)
                    .await
                {
                    Ok(job) => {
                        address = job.address();
                        managed_job = Some(job);
                    }
                    Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
                }
            } else if let Err(code) = pool.stop_all().await {
                return error(StatusCode::SERVICE_UNAVAILABLE, code);
            }
            let encoded = match serde_json::to_vec(&value) {
                Ok(bytes) => bytes,
                Err(_) => return error(StatusCode::BAD_REQUEST, "managed_request_json_invalid"),
            };
            parts.headers.remove(header::CONTENT_LENGTH);
            body = Body::from(encoded);
        } else if class != RequestClass::Metadata
            && let Err(code) = pool.stop_all().await
        {
            return error(StatusCode::SERVICE_UNAVAILABLE, code);
        }
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
    let upstream = match tokio::time::timeout(
        Duration::from_secs(gateway.config.response_header_timeout_seconds),
        gateway.client.request(Request::from_parts(parts, body)),
    )
    .await
    {
        Ok(Ok(response)) => response,
        // The request may already have reached the upstream. A managed worker
        // can prove that it stopped; an opaque upstream stays uncertain.
        Ok(Err(_)) => {
            finish_uncertain(lease, managed_job).await;
            return error(StatusCode::BAD_GATEWAY, "upstream_transport_failed");
        }
        Err(_) => {
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
    let idle = Duration::from_secs(gateway.config.stream_idle_timeout_seconds);
    let body = LeasedBody {
        inner: Some(body),
        lease,
        managed_job,
        idle,
        timer: Box::pin(tokio::time::sleep(idle)),
        settled: false,
        uncertain: false,
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
    idle: Duration,
    timer: Pin<Box<Sleep>>,
    settled: bool,
    uncertain: bool,
}

impl LeasedBody {
    fn finish(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
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
                let next = tokio::time::Instant::now() + self.idle;
                self.timer.as_mut().reset(next);
                if !self.uncertain && self.inner.as_ref().is_none_or(HttpBody::is_end_stream) {
                    self.finish();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(_))) => {
                self.uncertain = true;
                Poll::Ready(Some(Err(std::io::Error::other("upstream_body_failed"))))
            }
            Poll::Pending => {
                if self.timer.as_mut().poll(cx).is_ready() {
                    self.uncertain = true;
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
                // A bounded discard proves ordinary generation finished without
                // persisting or replaying content. Errors, stalls, and process
                // exit retain the recovery requirement because completion is
                // uncertain.
                tokio::time::timeout(limit, async {
                    while let Some(frame) =
                        std::future::poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await
                    {
                        if frame.is_err() {
                            return false;
                        }
                    }
                    true
                })
                .await
                .unwrap_or(false)
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
