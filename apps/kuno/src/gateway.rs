use crate::{
    admission::{Admission, Lease},
    config::Config,
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
        Ok(Arc::new(Self {
            config,
            client,
            admission,
            token: format!("Bearer {token}").into_bytes(),
            usage,
            catalog: tokio::sync::Mutex::new(None),
        }))
    }

    pub fn compatibility(self: &Arc<Self>) -> Router {
        Router::new().fallback(proxy).with_state(self.clone())
    }

    pub fn control(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/kuno/v1/status", get(status))
            .route("/kuno/v1/usage", get(usage))
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
    Json(Status {
        api_version: API_VERSION,
        mode: "gateway_only".into(),
        admission: gateway.admission.status(),
        gpu_placement_verified: false,
        image_generation: false,
    })
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
    let mut lease = if class != RequestClass::Metadata {
        match gateway.admission.acquire().await {
            Ok(lease) => Some(lease),
            Err(code) => return error(StatusCode::SERVICE_UNAVAILABLE, code),
        }
    } else {
        None
    };

    let (mut parts, body) = request.into_parts();
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
    parts.uri = match format!("http://{}{path}", gateway.config.upstream).parse() {
        Ok(uri) => uri,
        Err(_) => return error(StatusCode::BAD_REQUEST, "invalid_request_target"),
    };
    strip_hop_headers(&mut parts.headers);
    parts.headers.insert(
        header::HOST,
        HeaderValue::from_str(&gateway.config.upstream.to_string()).unwrap(),
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
        Ok(Err(_)) => return error(StatusCode::BAD_GATEWAY, "upstream_transport_failed"),
        Err(_) => return error(StatusCode::GATEWAY_TIMEOUT, "upstream_headers_timeout"),
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
        inner: body,
        lease,
        idle,
        timer: Box::pin(tokio::time::sleep(idle)),
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
    inner: Incoming,
    lease: Option<Lease>,
    idle: Duration,
    timer: Pin<Box<Sleep>>,
}

impl LeasedBody {
    fn finish(&mut self) {
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
        match Pin::new(&mut self.inner).poll_frame(cx) {
            Poll::Ready(None) => {
                self.finish();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(frame))) => {
                let next = tokio::time::Instant::now() + self.idle;
                self.timer.as_mut().reset(next);
                if self.inner.is_end_stream() {
                    self.finish();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(_))) => {
                self.lease.take();
                Poll::Ready(Some(Err(std::io::Error::other("upstream_body_failed"))))
            }
            Poll::Pending => {
                if self.timer.as_mut().poll(cx).is_ready() {
                    self.lease.take();
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
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for LeasedBody {
    fn drop(&mut self) {
        if self.inner.is_end_stream() {
            self.finish();
        }
    }
}
