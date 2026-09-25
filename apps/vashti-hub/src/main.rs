use std::{
    collections::{HashMap, VecDeque},
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use axum::{
    Json, Router,
    body::Body,
    extract::{ConnectInfo, DefaultBodyLimit, Multipart, Path as AxumPath, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use axum_extra::extract::{
    CookieJar,
    cookie::{Cookie, SameSite},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio::{fs, signal};
use tokio_util::io::ReaderStream;
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;
use vashti_update_manifest::{SCHEMA_VERSION, SignedArtifact, verify_release_signature};

const INDEX_HTML: &str = include_str!("../static/index.html");
const ADMIN_HTML: &str = include_str!("../static/admin.html");
const GETTING_STARTED_HTML: &str = include_str!("../static/getting-started.html");
const RELEASES_HTML: &str = include_str!("../static/releases.html");
const STYLES_CSS: &str = include_str!("../static/styles.css");
const NAV_JS: &str = include_str!("../static/nav.js");
const RELEASE_UI_JS: &str = include_str!("../static/release-ui.js");
const ROBOTS_TXT: &str = include_str!("../static/robots.txt");
const SITEMAP_XML: &str = include_str!("../static/sitemap.xml");
const LLMS_TXT: &str = include_str!("../static/llms.txt");
const FAVICON_PNG: &[u8] = include_bytes!("../static/favicon.png");
const LOGO_PNG: &[u8] = include_bytes!("../static/logo.png");
const KUNO_GUIDE: &str = concat!(
    include_str!("../../kuno/README.md"),
    "\n\n",
    include_str!("../../../docs/kuno-images.md")
);
const KUNO_INSTALL_SH: &str = include_str!("../../kuno/packaging/install.sh");
const INSTALL_SH: &str = include_str!("../../vashti/packaging/install.sh");
const AUTHORIZATION: HeaderName = HeaderName::from_static("authorization");
const X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");
const X_REAL_IP: HeaderName = HeaderName::from_static("x-real-ip");
const X_FORWARDED_PROTO: HeaderName = HeaderName::from_static("x-forwarded-proto");
const ADMIN_SESSION_COOKIE: &str = "vashti_hub_admin";
const ADMIN_SETUP_KEY_FILE: &str = "admin-setup-key.txt";
const ADMIN_RESET_KEY_FILE: &str = "admin-reset-key.txt";
const ADMIN_SESSION_TTL_SECONDS: i64 = 60 * 60 * 12;
const UPLOAD_KEY_TTL_SECONDS: i64 = 60 * 10;

#[derive(Clone)]
struct AppState {
    config: Config,
    db: SqlitePool,
    rate_limiter: Arc<RateLimiter>,
    upload_keys: Arc<UploadKeyStore>,
}

#[derive(Clone)]
struct Config {
    data_dir: PathBuf,
    artifact_dir: PathBuf,
    database_path: PathBuf,
    bind_addr: SocketAddr,
    max_upload_bytes: usize,
    trust_proxy_headers: bool,
    secure_session_cookies: bool,
}

impl Config {
    fn from_env() -> Result<Self, AppError> {
        let app_root = env::current_dir().map_err(AppError::from)?;
        let data_dir = env::var_os("VASHTI_HUB_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_data_dir(&app_root));
        let data_dir = if data_dir.is_absolute() {
            data_dir
        } else {
            app_root.join(data_dir)
        };

        let bind_addr = env::var("VASHTI_HUB_BIND")
            .unwrap_or_else(|_| "127.0.0.1:7781".to_string())
            .parse()
            .map_err(|_| AppError::bad_request("invalid_bind", "VASHTI_HUB_BIND is invalid"))?;
        let max_upload_bytes = env::var("VASHTI_HUB_MAX_UPLOAD_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(512 * 1024 * 1024);
        let trust_proxy_headers = env_flag("VASHTI_HUB_TRUST_PROXY_HEADERS", false);
        let secure_session_cookies = env_flag("VASHTI_HUB_COOKIE_SECURE", false);

        Ok(Self {
            artifact_dir: data_dir.join("artifacts"),
            database_path: data_dir.join("hub.db"),
            data_dir,
            bind_addr,
            max_upload_bytes,
            trust_proxy_headers,
            secure_session_cookies,
        })
    }
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn default_data_dir(app_root: &Path) -> PathBuf {
    if app_root.join("apps/vashti-hub").is_dir() {
        PathBuf::from("apps/vashti-hub/data")
    } else {
        PathBuf::from("data")
    }
}

#[derive(Default)]
struct RateLimiter {
    buckets: Mutex<HashMap<String, VecDeque<i64>>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self::default()
    }

    async fn check(
        &self,
        key: impl Into<String>,
        limit: usize,
        window_seconds: i64,
    ) -> Result<(), AppError> {
        let key = key.into();
        let now = unix_timestamp();
        let cutoff = now.saturating_sub(window_seconds);
        let mut buckets = self.buckets.lock().await;
        let bucket = buckets.entry(key).or_default();

        while bucket.front().is_some_and(|timestamp| *timestamp <= cutoff) {
            bucket.pop_front();
        }

        if bucket.len() >= limit {
            return Err(AppError::too_many_requests(
                "rate_limited",
                "Too many requests. Try again later.",
            ));
        }

        bucket.push_back(now);

        if buckets.len() > 10_000 {
            buckets.retain(|_, bucket| bucket.back().is_some_and(|timestamp| *timestamp > cutoff));
        }

        Ok(())
    }
}

#[derive(Default)]
struct UploadKeyStore {
    keys: Mutex<HashMap<String, i64>>,
}

impl UploadKeyStore {
    fn new() -> Self {
        Self::default()
    }

    async fn create(&self) -> UploadKey {
        let token = generate_token();
        let expires_at = unix_timestamp() + UPLOAD_KEY_TTL_SECONDS;
        let mut keys = self.keys.lock().await;
        keys.retain(|_, expiry| *expiry > unix_timestamp());
        keys.insert(token_hash(&token), expires_at);
        UploadKey { token, expires_at }
    }

    async fn consume(&self, token: &str) -> bool {
        let now = unix_timestamp();
        let mut keys = self.keys.lock().await;
        keys.retain(|_, expiry| *expiry > now);
        keys.remove(&token_hash(token))
            .is_some_and(|expiry| expiry > now)
    }
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    code: String,
    message: String,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: String,
    message: String,
}

impl AppError {
    fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
        }
    }

    fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Admin authentication is required",
        )
    }

    fn not_found(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, code, message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    fn conflict(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, code, message)
    }

    fn forbidden(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, code, message)
    }

    fn too_many_requests(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, code, message)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response()
    }
}

impl From<sqlx::Error> for AppError {
    fn from(error: sqlx::Error) -> Self {
        tracing::error!(?error, "database error");
        Self::internal("Database operation failed")
    }
}

impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        tracing::error!(?error, "filesystem error");
        Self::internal("Filesystem operation failed")
    }
}

#[derive(Debug, Serialize)]
struct ReleaseResponse {
    version: String,
    release_status: String,
    is_latest: bool,
    is_prerelease: bool,
    notes: Option<String>,
    created_at: i64,
    artifacts: Vec<ArtifactResponse>,
}

#[derive(Debug, Serialize)]
struct ArtifactResponse {
    id: String,
    version: String,
    target: String,
    filename: String,
    sha256: String,
    size_bytes: i64,
    downloads: i64,
    download_url: String,
}

#[derive(Debug, Serialize)]
struct UpdateManifestResponse {
    schema_version: u32,
    channel: String,
    release_status: String,
    notes: Option<String>,
    artifact: SignedArtifact,
    signature: String,
    download_url: String,
}

#[derive(Debug, Serialize)]
struct UploadResponse {
    release: ReleaseResponse,
    latest_version: Option<String>,
}

#[derive(Debug, Serialize)]
struct PromoteReleaseResponse {
    release: ReleaseResponse,
    latest_version: Option<String>,
    incorporated_versions: Vec<String>,
}

#[derive(Debug, Serialize)]
struct UploadKeyResponse {
    token: String,
    expires_at: i64,
    ttl_seconds: i64,
}

struct UploadKey {
    token: String,
    expires_at: i64,
}

#[derive(Debug, Serialize)]
struct AdminStatusResponse {
    claimed: bool,
    setup_key_path: Option<String>,
    setup_key_command: Option<String>,
    reset_key_path: Option<String>,
    reset_key_command: Option<String>,
    authenticated: bool,
}

#[derive(Debug, Serialize)]
struct MessageResponse {
    message: String,
}

#[derive(Debug, Serialize)]
struct ResetRequestResponse {
    message: String,
    reset_key_path: String,
    reset_key_command: String,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    artifacts: Vec<ArtifactResponse>,
    page_hits: Vec<PageHitSummary>,
    total_page_hits: i64,
    unique_visitors: i64,
}

#[derive(Debug, Serialize)]
struct PageHitSummary {
    path: String,
    hits: i64,
    visitors: i64,
}

#[derive(Debug, Deserialize)]
struct SetupAdminRequest {
    setup_key: String,
    password: String,
}

#[derive(Debug, Deserialize)]
struct LoginRequest {
    password: String,
}

#[derive(Debug, Deserialize)]
struct ResetPasswordRequest {
    reset_key: String,
    password: String,
}

#[derive(Debug, Default, Deserialize)]
struct PromoteReleaseRequest {
    #[serde(default)]
    rollup_previous_prereleases: bool,
}

struct ParsedUpload {
    version: String,
    major: i64,
    minor: i64,
    patch: i64,
    notes: Option<String>,
    artifacts: Vec<ParsedArtifactUpload>,
}

#[derive(Debug)]
struct ParsedArtifactUpload {
    target: String,
    filename: String,
    content_type: String,
    file: tempfile::NamedTempFile,
    size_bytes: i64,
    sha256: String,
    signature: String,
}

const DEFAULT_REQUEST_BODY_LIMIT: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<(), AppError> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vashti_hub=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env()?;
    prepare_storage(&config).await?;

    let database_url = format!("sqlite://{}", config.database_path.display());
    let database_options = SqliteConnectOptions::from_str(&database_url)
        .map_err(AppError::from)?
        .create_if_missing(true)
        .foreign_keys(true);
    let db = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(database_options)
        .await
        .map_err(AppError::from)?;
    secure_file_if_present(&config.database_path).await?;
    sqlx::migrate!("./migrations")
        .run(&db)
        .await
        .map_err(|error| {
            tracing::error!(?error, "migration error");
            AppError::internal("Database migration failed")
        })?;

    cleanup_upload_files(&config.artifact_dir).await?;
    ensure_settings(&db).await?;
    ensure_admin_setup_key(&db, &config).await?;
    clear_legacy_upload_token(&db, &config).await?;

    let bind_addr = config.bind_addr;
    let state = AppState {
        config,
        db,
        rate_limiter: Arc::new(RateLimiter::new()),
        upload_keys: Arc::new(UploadKeyStore::new()),
    };
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    tracing::info!("vashti hub listening on http://{}", listener.local_addr()?);

    axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(AppError::from)?;

    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/admin", get(admin))
        .route("/getting-started", get(getting_started))
        .route("/releases", get(releases_page))
        .route("/styles.css", get(styles))
        .route("/nav.js", get(nav_js))
        .route("/release-ui.js", get(release_ui_js))
        .route("/favicon.png", get(favicon))
        .route("/logo.png", get(logo))
        .route("/robots.txt", get(robots_txt))
        .route("/sitemap.xml", get(sitemap_xml))
        .route("/llms.txt", get(llms_txt))
        .route("/install.sh", get(install_script))
        .route("/install-kuno.sh", get(kuno_install_script))
        .route("/kuno-setup.txt", get(kuno_setup_guide))
        .route("/api/admin/status", get(admin_status))
        .route("/api/admin/setup", post(setup_admin))
        .route("/api/admin/login", post(login_admin))
        .route("/api/admin/logout", post(logout_admin))
        .route("/api/admin/upload-key", post(create_upload_key))
        .route(
            "/api/admin/password-reset/request",
            post(request_password_reset),
        )
        .route(
            "/api/admin/password-reset/confirm",
            post(confirm_password_reset),
        )
        .route(
            "/api/releases",
            get(list_releases)
                .post(upload_release)
                .layer(DefaultBodyLimit::max(state.config.max_upload_bytes)),
        )
        .route("/api/releases/{version}", delete(delete_release))
        .route("/api/releases/{version}/promote", post(promote_release))
        .route("/api/releases/latest", get(latest_release))
        .route("/api/updates/{channel}/{target}", get(update_manifest))
        .route("/api/stats", get(stats))
        .route("/releases/latest/VERSION", get(latest_version_file))
        .route("/releases/latest/SHA256SUMS", get(latest_checksums))
        .route("/releases/latest/{filename}", get(download_latest))
        .route("/releases/{version}/VERSION", get(version_file))
        .route("/releases/{version}/SHA256SUMS", get(version_checksums))
        .route("/releases/{version}/{filename}", get(download_version))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(TraceLayer::new_for_http())
}

async fn prepare_storage(config: &Config) -> Result<(), std::io::Error> {
    fs::create_dir_all(&config.data_dir).await?;
    fs::create_dir_all(&config.artifact_dir).await?;
    secure_directory(&config.data_dir).await?;
    secure_directory(&config.artifact_dir).await?;
    Ok(())
}

#[cfg(unix)]
async fn secure_directory(path: &Path) -> Result<(), std::io::Error> {
    set_mode(path, 0o700).await
}

#[cfg(not(unix))]
async fn secure_directory(_path: &Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
async fn secure_file_if_present(path: &Path) -> Result<(), std::io::Error> {
    if fs::try_exists(path).await? {
        set_mode(path, 0o600).await?;
    }
    Ok(())
}

#[cfg(not(unix))]
async fn secure_file_if_present(_path: &Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
async fn set_mode(path: &Path, mode: u32) -> Result<(), std::io::Error> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = fs::metadata(path).await?;
    let mut permissions = metadata.permissions();
    if permissions.mode() & 0o777 != mode {
        permissions.set_mode(mode);
        fs::set_permissions(path, permissions).await?;
    }
    Ok(())
}

async fn index(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
) -> Result<impl IntoResponse, AppError> {
    if admin_exists(&state.db).await? {
        record_page_hit_best_effort(&state, &headers, remote_addr, &jar, "/").await;
        Ok(html(INDEX_HTML))
    } else {
        Ok(html(ADMIN_HTML))
    }
}

async fn admin() -> impl IntoResponse {
    html(ADMIN_HTML)
}

async fn getting_started(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
) -> impl IntoResponse {
    record_page_hit_best_effort(&state, &headers, remote_addr, &jar, "/getting-started").await;
    html(GETTING_STARTED_HTML)
}

async fn releases_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
) -> impl IntoResponse {
    record_page_hit_best_effort(&state, &headers, remote_addr, &jar, "/releases").await;
    html(RELEASES_HTML)
}

async fn styles() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLES_CSS,
    )
}

async fn nav_js() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=604800"),
        ],
        NAV_JS,
    )
}

async fn release_ui_js() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        RELEASE_UI_JS,
    )
}

async fn favicon() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=604800"),
        ],
        FAVICON_PNG,
    )
}

async fn logo() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=604800"),
        ],
        LOGO_PNG,
    )
}

async fn robots_txt() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        ROBOTS_TXT,
    )
}

async fn sitemap_xml() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/xml; charset=utf-8")],
        SITEMAP_XML,
    )
}

async fn llms_txt() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        LLMS_TXT,
    )
}

async fn install_script(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
) -> Result<impl IntoResponse, AppError> {
    require_claimed(&state.db).await?;
    record_page_hit_best_effort(&state, &headers, remote_addr, &jar, "/install.sh").await;
    Ok((
        [
            (header::CONTENT_TYPE, "text/x-shellscript; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"install.sh\"",
            ),
        ],
        INSTALL_SH,
    ))
}

async fn kuno_setup_guide() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        KUNO_GUIDE,
    )
}

async fn kuno_install_script(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
) -> Result<impl IntoResponse, AppError> {
    require_claimed(&state.db).await?;
    record_page_hit_best_effort(&state, &headers, remote_addr, &jar, "/install-kuno.sh").await;
    Ok((
        [
            (header::CONTENT_TYPE, "text/x-shellscript; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"install-kuno.sh\"",
            ),
        ],
        KUNO_INSTALL_SH,
    ))
}

async fn admin_status(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<AdminStatusResponse>, AppError> {
    let claimed = admin_exists(&state.db).await?;
    let authenticated = admin_from_session(&state.db, &jar).await?.is_some();
    Ok(Json(AdminStatusResponse {
        claimed,
        setup_key_path: (!claimed).then(|| setup_key_path(&state.config).display().to_string()),
        setup_key_command: (!claimed).then(|| cat_command(&setup_key_path(&state.config))),
        reset_key_path: claimed.then(|| reset_key_path(&state.config).display().to_string()),
        reset_key_command: claimed.then(|| cat_command(&reset_key_path(&state.config))),
        authenticated,
    }))
}

async fn setup_admin(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
    Json(payload): Json<SetupAdminRequest>,
) -> Result<(CookieJar, Json<MessageResponse>), AppError> {
    require_same_origin(&headers)?;
    state
        .rate_limiter
        .check(
            format!(
                "setup:{}",
                client_key(&headers, remote_addr, state.config.trust_proxy_headers)
            ),
            5,
            300,
        )
        .await?;

    if admin_exists(&state.db).await? {
        return Err(AppError::conflict(
            "admin_exists",
            "The hub already has an admin account.",
        ));
    }

    let expected_hash: Option<String> =
        sqlx::query_scalar("SELECT admin_setup_key_hash FROM release_settings WHERE id = 1")
            .fetch_one(&state.db)
            .await?;
    if expected_hash.as_deref() != Some(&token_hash(payload.setup_key.trim())) {
        return Err(AppError::unauthorized());
    }

    create_admin(&state.db, &payload.password).await?;
    clear_setup_key(&state.db, &state.config).await?;
    let session_id = create_admin_session(&state.db).await?;
    let secure = should_secure_cookie(&state.config, &headers);

    Ok((
        jar.add(admin_session_cookie(&session_id, secure)),
        Json(MessageResponse {
            message: "Admin account created.".to_string(),
        }),
    ))
}

async fn login_admin(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    jar: CookieJar,
    Json(payload): Json<LoginRequest>,
) -> Result<(CookieJar, Json<MessageResponse>), AppError> {
    require_same_origin(&headers)?;
    state
        .rate_limiter
        .check(
            format!(
                "login:{}",
                client_key(&headers, remote_addr, state.config.trust_proxy_headers)
            ),
            3,
            300,
        )
        .await?;

    require_claimed(&state.db).await?;
    if !verify_admin_password(&state.db, &payload.password).await? {
        return Err(AppError::unauthorized());
    }

    let session_id = create_admin_session(&state.db).await?;
    let secure = should_secure_cookie(&state.config, &headers);
    Ok((
        jar.add(admin_session_cookie(&session_id, secure)),
        Json(MessageResponse {
            message: "Logged in.".to_string(),
        }),
    ))
}

async fn logout_admin(
    State(state): State<AppState>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Result<(CookieJar, Json<MessageResponse>), AppError> {
    require_same_origin(&headers)?;
    if let Some(cookie) = jar.get(ADMIN_SESSION_COOKIE) {
        delete_admin_session(&state.db, cookie.value()).await?;
    }
    let secure = should_secure_cookie(&state.config, &headers);
    Ok((
        jar.remove(expired_admin_session_cookie(secure)),
        Json(MessageResponse {
            message: "Logged out.".to_string(),
        }),
    ))
}

async fn request_password_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
) -> Result<Json<ResetRequestResponse>, AppError> {
    require_same_origin(&headers)?;
    require_claimed(&state.db).await?;
    state
        .rate_limiter
        .check("reset:global", 1, 60 * 60 * 24)
        .await?;
    state
        .rate_limiter
        .check(
            format!(
                "reset:{}",
                client_key(&headers, remote_addr, state.config.trust_proxy_headers)
            ),
            1,
            60 * 60 * 24,
        )
        .await?;

    let key = generate_token();
    store_reset_key(&state.db, &state.config, &key).await?;
    let path = reset_key_path(&state.config);
    Ok(Json(ResetRequestResponse {
        message: "Password reset key generated on the server.".to_string(),
        reset_key_path: path.display().to_string(),
        reset_key_command: cat_command(&path),
    }))
}

async fn confirm_password_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(remote_addr): ConnectInfo<SocketAddr>,
    Json(payload): Json<ResetPasswordRequest>,
) -> Result<Json<MessageResponse>, AppError> {
    require_same_origin(&headers)?;
    require_claimed(&state.db).await?;
    state
        .rate_limiter
        .check(
            format!(
                "reset-confirm:{}",
                client_key(&headers, remote_addr, state.config.trust_proxy_headers)
            ),
            5,
            300,
        )
        .await?;

    let expected_hash: Option<String> =
        sqlx::query_scalar("SELECT reset_key_hash FROM release_settings WHERE id = 1")
            .fetch_one(&state.db)
            .await?;
    if expected_hash.as_deref() != Some(&token_hash(payload.reset_key.trim())) {
        return Err(AppError::unauthorized());
    }

    update_admin_password(&state.db, &payload.password).await?;
    clear_reset_key(&state.db, &state.config).await?;
    Ok(Json(MessageResponse {
        message: "Admin password updated.".to_string(),
    }))
}

async fn list_releases(
    State(state): State<AppState>,
) -> Result<Json<Vec<ReleaseResponse>>, AppError> {
    require_claimed(&state.db).await?;
    Ok(Json(load_releases(&state.db).await?))
}

async fn latest_release(State(state): State<AppState>) -> Result<Json<ReleaseResponse>, AppError> {
    require_claimed(&state.db).await?;
    let version = latest_version(&state.db).await?;
    Ok(Json(load_release(&state.db, &version).await?))
}

async fn update_manifest(
    State(state): State<AppState>,
    AxumPath((channel, target)): AxumPath<(String, String)>,
) -> Result<impl IntoResponse, AppError> {
    require_claimed(&state.db).await?;
    let channel = validate_update_channel(&channel)?;
    let target = validate_target(&target)?;
    let version = update_version_for_channel(&state.db, channel).await?;
    let row = sqlx::query(
        r#"
        SELECT r.release_status,
               r.notes,
               a.filename,
               a.sha256,
               a.size_bytes,
               a.update_signature
        FROM releases r
        JOIN release_artifacts a ON a.release_version = r.version
        WHERE r.version = ? AND a.target = ?
        "#,
    )
    .bind(&version)
    .bind(&target)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| {
        AppError::not_found(
            "update_not_found",
            "No signed update is available for this target",
        )
    })?;
    let signature: Option<String> = row.try_get("update_signature")?;
    let signature = signature.ok_or_else(|| {
        AppError::not_found(
            "update_not_signed",
            "This release is not available for managed updates",
        )
    })?;
    let artifact = SignedArtifact {
        version: version.clone(),
        target,
        filename: row.try_get("filename")?,
        sha256: row.try_get("sha256")?,
        size_bytes: row.try_get("size_bytes")?,
    };
    verify_release_signature(&artifact, &signature).map_err(|error| {
        tracing::error!(?error, version, "stored update signature is invalid");
        AppError::internal("Stored update manifest is invalid")
    })?;

    let manifest = UpdateManifestResponse {
        schema_version: SCHEMA_VERSION,
        channel: channel.to_string(),
        release_status: row.try_get("release_status")?,
        notes: row.try_get("notes")?,
        download_url: format!("/releases/{version}/{}", artifact.filename),
        artifact,
        signature,
    };
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(manifest)))
}

async fn promote_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    jar: CookieJar,
    AxumPath(version): AxumPath<String>,
    request: Option<Json<PromoteReleaseRequest>>,
) -> Result<Json<PromoteReleaseResponse>, AppError> {
    require_same_origin(&headers)?;
    require_admin_session(&state, &jar).await?;
    let version = normalize_version_label(&version)?.0;
    let request = request.map(|Json(request)| request).unwrap_or_default();
    let target = sqlx::query(
        r#"
        SELECT release_status, notes, version_major, version_minor, version_patch
        FROM releases
        WHERE version = ?
        "#,
    )
    .bind(&version)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::not_found("release_not_found", "Release not found"))?;
    let release_status: String = target.try_get("release_status")?;
    if release_status != "prerelease" {
        return Err(AppError::conflict(
            "release_already_promoted",
            "Only a prerelease can be promoted",
        ));
    }

    require_stable_version(&version)?;
    let target_version = release_version(&version)?;
    if newest_version(&state.db, "stable")
        .await?
        .map(|latest| release_version(&latest))
        .transpose()?
        .is_some_and(|latest| target_version <= latest)
    {
        return Err(AppError::conflict(
            "release_not_newer",
            "A promoted release must be newer than the current stable release",
        ));
    }
    let target_notes: Option<String> = target.try_get("notes")?;
    let incorporated = if request.rollup_previous_prereleases {
        previous_prereleases_for_rollup(&state.db, &version).await?
    } else {
        Vec::new()
    };
    let incorporated_versions = incorporated
        .iter()
        .map(|release| release.version.clone())
        .collect::<Vec<_>>();
    let merged_notes = merge_release_notes(&version, target_notes.as_deref(), &incorporated);

    let mut transaction = state.db.begin().await?;
    sqlx::query("UPDATE releases SET release_status = 'released', notes = ? WHERE version = ?")
        .bind(&merged_notes)
        .bind(&version)
        .execute(&mut *transaction)
        .await?;

    for incorporated_version in &incorporated_versions {
        sqlx::query("DELETE FROM download_events WHERE release_version = ?")
            .bind(incorporated_version)
            .execute(&mut *transaction)
            .await?;
        sqlx::query("DELETE FROM release_artifacts WHERE release_version = ?")
            .bind(incorporated_version)
            .execute(&mut *transaction)
            .await?;
        sqlx::query("DELETE FROM releases WHERE version = ? AND release_status = 'prerelease'")
            .bind(incorporated_version)
            .execute(&mut *transaction)
            .await?;
    }

    transaction.commit().await?;
    update_latest_version(&state.db).await?;

    for incorporated_version in &incorporated_versions {
        let artifact_dir = state.config.artifact_dir.join(incorporated_version);
        if let Err(error) = fs::remove_dir_all(&artifact_dir).await
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                ?error,
                version = incorporated_version,
                "failed to remove incorporated prerelease artifacts"
            );
        }
    }

    Ok(Json(PromoteReleaseResponse {
        release: load_release(&state.db, &version).await?,
        latest_version: latest_version_optional(&state.db).await?,
        incorporated_versions,
    }))
}

async fn delete_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    jar: CookieJar,
    AxumPath(version): AxumPath<String>,
) -> Result<Json<MessageResponse>, AppError> {
    require_same_origin(&headers)?;
    require_admin_session(&state, &jar).await?;
    let version = normalize_version_label(&version)?.0;

    let rows_deleted = sqlx::query("DELETE FROM releases WHERE version = ?")
        .bind(&version)
        .execute(&state.db)
        .await?
        .rows_affected();
    if rows_deleted == 0 {
        return Err(AppError::not_found(
            "release_not_found",
            "Release not found",
        ));
    }

    sqlx::query("DELETE FROM download_events WHERE release_version = ?")
        .bind(&version)
        .execute(&state.db)
        .await?;
    sqlx::query("DELETE FROM release_artifacts WHERE release_version = ?")
        .bind(&version)
        .execute(&state.db)
        .await?;
    let artifact_dir = state.config.artifact_dir.join(&version);
    match fs::remove_dir_all(&artifact_dir).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    update_latest_version(&state.db).await?;

    Ok(Json(MessageResponse {
        message: format!("Deleted release {version}."),
    }))
}

async fn stats(
    State(state): State<AppState>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Result<Json<StatsResponse>, AppError> {
    require_same_origin(&headers)?;
    require_admin_session(&state, &jar).await?;
    let (total_page_hits, unique_visitors) = page_hit_totals(&state.db).await?;
    Ok(Json(StatsResponse {
        artifacts: load_artifacts_with_downloads(&state.db, None).await?,
        page_hits: load_page_hit_summaries(&state.db).await?,
        total_page_hits,
        unique_visitors,
    }))
}

async fn upload_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Json<UploadResponse>, AppError> {
    require_claimed(&state.db).await?;
    consume_upload_key(&state, &headers).await?;
    let upload = parse_upload(multipart, &state.config.artifact_dir).await?;
    let release_exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM releases WHERE version = ?)")
            .bind(&upload.version)
            .fetch_one(&state.db)
            .await?;
    if release_exists {
        return Err(AppError::conflict(
            "release_already_exists",
            "That release version has already been uploaded",
        ));
    }
    let artifact_dir = state.config.artifact_dir.join(&upload.version);
    // Never overwrite an existing version directory, including an interrupted upload.
    fs::create_dir(&artifact_dir).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            AppError::conflict(
                "release_directory_exists",
                "Release directory already exists; inspect the interrupted upload before retrying",
            )
        } else {
            error.into()
        }
    })?;
    let mut transaction = state.db.begin().await?;
    let now = unix_timestamp();
    sqlx::query(
        r#"
        INSERT INTO releases (version, version_major, version_minor, version_patch, release_status, notes, created_at)
        VALUES (?, ?, ?, ?, 'prerelease', ?, ?)
        "#,
    )
    .bind(&upload.version)
    .bind(upload.major)
    .bind(upload.minor)
    .bind(upload.patch)
    .bind(&upload.notes)
    .bind(now)
    .execute(&mut *transaction)
    .await?;

    for artifact in &upload.artifacts {
        let storage_path = artifact_dir.join(&artifact.filename);
        // The verified temporary file is on the same filesystem. Linking is
        // constant-memory and cannot overwrite an existing file.
        fs::hard_link(artifact.file.path(), &storage_path).await?;
        sqlx::query(
            r#"
            INSERT INTO release_artifacts
                (id, release_version, target, filename, content_type, size_bytes, sha256, storage_path, update_signature, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(&upload.version)
        .bind(&artifact.target)
        .bind(&artifact.filename)
        .bind(&artifact.content_type)
        .bind(artifact.size_bytes)
        .bind(&artifact.sha256)
        .bind(storage_path.to_string_lossy().to_string())
        .bind(&artifact.signature)
        .bind(now)
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;

    Ok(Json(UploadResponse {
        release: load_release(&state.db, &upload.version).await?,
        latest_version: latest_version_optional(&state.db).await?,
    }))
}

async fn create_upload_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    jar: CookieJar,
) -> Result<Json<UploadKeyResponse>, AppError> {
    require_same_origin(&headers)?;
    require_admin_session(&state, &jar).await?;
    let key = state.upload_keys.create().await;

    Ok(Json(UploadKeyResponse {
        token: key.token,
        expires_at: key.expires_at,
        ttl_seconds: UPLOAD_KEY_TTL_SECONDS,
    }))
}

async fn latest_version_file(State(state): State<AppState>) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    Ok(text_file("VERSION", latest_version(&state.db).await?))
}

async fn version_file(
    State(state): State<AppState>,
    AxumPath(version): AxumPath<String>,
) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    Ok(text_file("VERSION", normalize_version_label(&version)?.0))
}

async fn latest_checksums(State(state): State<AppState>) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    let version = latest_version(&state.db).await?;
    checksums_for_version(&state, &version).await
}

async fn version_checksums(
    State(state): State<AppState>,
    AxumPath(version): AxumPath<String>,
) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    let version = normalize_version_label(&version)?.0;
    checksums_for_version(&state, &version).await
}

async fn download_latest(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    AxumPath(filename): AxumPath<String>,
) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    let version = latest_version(&state.db).await?;
    download_artifact(&state, &method, &headers, &version, &filename, "latest").await
}

async fn download_version(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    AxumPath((version, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    require_claimed(&state.db).await?;
    let version = normalize_version_label(&version)?.0;
    download_artifact(&state, &method, &headers, &version, &filename, "version").await
}

async fn checksums_for_version(state: &AppState, version: &str) -> Result<Response, AppError> {
    let artifacts = load_artifacts_with_downloads(&state.db, Some(version)).await?;
    if artifacts.is_empty() {
        return Err(AppError::not_found(
            "release_not_found",
            "Release not found",
        ));
    }

    let body = artifacts
        .iter()
        .map(|artifact| format!("{}  {}", artifact.sha256, artifact.filename))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    Ok(text_file("SHA256SUMS", body))
}

async fn download_artifact(
    state: &AppState,
    method: &Method,
    headers: &HeaderMap,
    version: &str,
    filename: &str,
    kind: &str,
) -> Result<Response, AppError> {
    let row = sqlx::query(
        r#"
        SELECT id, release_version, target, filename, content_type, size_bytes, storage_path
        FROM release_artifacts
        WHERE release_version = ? AND filename = ?
        "#,
    )
    .bind(version)
    .bind(filename)
    .fetch_optional(&state.db)
    .await?
    .ok_or_else(|| AppError::not_found("artifact_not_found", "Artifact not found"))?;

    let artifact_id: String = row.try_get("id")?;
    let release_version: String = row.try_get("release_version")?;
    let target: String = row.try_get("target")?;
    let content_type: String = row.try_get("content_type")?;
    let size_bytes: i64 = row.try_get("size_bytes")?;
    let storage_path: String = row.try_get("storage_path")?;
    let file = fs::File::open(&storage_path).await?;

    if counts_as_download(method) {
        record_download(
            &state.db,
            headers,
            &artifact_id,
            &release_version,
            &target,
            kind,
        )
        .await?;
    }

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, size_bytes.to_string())
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", filename.replace('"', "")),
        )
        .body(Body::from_stream(ReaderStream::new(file)))
        .map_err(|_| AppError::internal("Failed to build response"))
}

fn counts_as_download(method: &Method) -> bool {
    method == Method::GET
}

// Called before accepting requests, so only files from interrupted processes remain.
async fn cleanup_upload_files(directory: &Path) -> Result<(), AppError> {
    let mut entries = fs::read_dir(directory).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_name().to_string_lossy().starts_with(".upload-")
            && entry.file_type().await?.is_file()
        {
            fs::remove_file(entry.path()).await?;
        }
    }
    Ok(())
}

async fn parse_upload(
    mut multipart: Multipart,
    directory: &Path,
) -> Result<ParsedUpload, AppError> {
    let mut version = None;
    let mut targets = Vec::new();
    let mut signatures = Vec::new();
    let mut notes = None;
    let mut uploaded_artifacts = Vec::new();
    let mut fields = 0;
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|_| AppError::bad_request("invalid_multipart", "Invalid multipart upload"))?
    {
        fields += 1;
        if fields > 50 {
            return Err(AppError::bad_request(
                "too_many_fields",
                "At most 16 artifacts may be uploaded together",
            ));
        }
        let name = field.name().unwrap_or_default().to_string();
        if name == "artifact" {
            let filename = field.file_name().map(safe_filename).ok_or_else(|| {
                AppError::bad_request("missing_filename", "Artifact filename is required")
            })?;
            let content_type = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_string();
            let file = tempfile::Builder::new()
                .prefix(".upload-")
                .tempfile_in(directory)?;
            let mut writer = fs::File::from_std(file.reopen()?);
            let mut hasher = Sha256::new();
            let mut size_bytes = 0_i64;
            while let Some(chunk) = field.chunk().await.map_err(|_| {
                AppError::bad_request("invalid_multipart", "Incomplete artifact upload")
            })? {
                hasher.update(&chunk);
                size_bytes += chunk.len() as i64;
                writer.write_all(&chunk).await?;
            }
            writer.sync_all().await?;
            drop(writer);
            uploaded_artifacts.push(UploadedArtifact {
                filename,
                content_type,
                file,
                size_bytes,
                sha256: hex::encode(hasher.finalize()),
            });
            continue;
        }
        if !matches!(name.as_str(), "version" | "target" | "signature" | "notes") {
            return Err(AppError::bad_request(
                "unknown_field",
                "Unsupported upload field",
            ));
        }
        let limit = if name == "notes" { 1024 * 1024 } else { 4096 };
        let mut bytes = Vec::new();
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|_| AppError::bad_request("invalid_multipart", "Invalid multipart field"))?
        {
            if bytes.len() + chunk.len() > limit {
                return Err(AppError::bad_request(
                    "field_too_large",
                    "Upload metadata field is too large",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let value = text_field(&bytes)?;
        match name.as_str() {
            "version" => version = Some(value),
            "target" => targets.push(validate_target(&value)?),
            "signature" => signatures.push(value),
            "notes" => notes = (!value.trim().is_empty()).then_some(value),
            _ => unreachable!(),
        }
    }

    let (version, major, minor, patch) = normalize_version_label(
        &version.ok_or_else(|| AppError::bad_request("missing_version", "Version is required"))?,
    )?;
    let artifacts = assemble_artifact_uploads(targets, uploaded_artifacts, signatures)?;
    validate_upload_signatures(&version, &artifacts)?;

    Ok(ParsedUpload {
        version,
        major,
        minor,
        patch,
        notes,
        artifacts,
    })
}

struct UploadedArtifact {
    filename: String,
    content_type: String,
    file: tempfile::NamedTempFile,
    size_bytes: i64,
    sha256: String,
}

fn assemble_artifact_uploads(
    targets: Vec<String>,
    uploaded_artifacts: Vec<UploadedArtifact>,
    signatures: Vec<String>,
) -> Result<Vec<ParsedArtifactUpload>, AppError> {
    if targets.is_empty() || uploaded_artifacts.is_empty() {
        return Err(AppError::bad_request(
            "missing_artifact",
            "At least one target and artifact are required",
        ));
    }
    if targets.len() != uploaded_artifacts.len() || targets.len() != signatures.len() {
        return Err(AppError::bad_request(
            "artifact_metadata_mismatch",
            "Each artifact must have one matching target and signature",
        ));
    }

    let mut seen_targets = std::collections::HashSet::new();
    let mut seen_filenames = std::collections::HashSet::new();
    let mut artifacts = Vec::with_capacity(targets.len());
    for ((target, uploaded), signature) in
        targets.into_iter().zip(uploaded_artifacts).zip(signatures)
    {
        let UploadedArtifact {
            filename,
            content_type,
            file,
            size_bytes,
            sha256,
        } = uploaded;
        if size_bytes == 0 {
            return Err(AppError::bad_request(
                "empty_artifact",
                "Artifact cannot be empty",
            ));
        }
        if !seen_targets.insert(target.clone()) {
            return Err(AppError::bad_request(
                "duplicate_target",
                "Each release target may be uploaded only once",
            ));
        }
        if !seen_filenames.insert(filename.clone()) {
            return Err(AppError::bad_request(
                "duplicate_filename",
                "Each release artifact filename must be unique",
            ));
        }
        artifacts.push(ParsedArtifactUpload {
            target,
            filename,
            content_type,
            file,
            size_bytes,
            sha256,
            signature,
        });
    }

    Ok(artifacts)
}

fn validate_upload_signatures(
    version: &str,
    artifacts: &[ParsedArtifactUpload],
) -> Result<(), AppError> {
    for artifact in artifacts {
        let signed_artifact = SignedArtifact {
            version: version.to_string(),
            target: artifact.target.clone(),
            filename: artifact.filename.clone(),
            sha256: artifact.sha256.clone(),
            size_bytes: artifact.size_bytes,
        };
        verify_release_signature(&signed_artifact, &artifact.signature).map_err(|_| {
            AppError::bad_request(
                "invalid_update_signature",
                "Artifact release signature is invalid",
            )
        })?;
    }
    Ok(())
}

fn validate_update_channel(channel: &str) -> Result<&str, AppError> {
    match channel {
        "stable" | "prerelease" => Ok(channel),
        _ => Err(AppError::bad_request(
            "invalid_update_channel",
            "Update channel must be stable or prerelease",
        )),
    }
}

async fn newest_version(db: &SqlitePool, channel: &str) -> Result<Option<String>, AppError> {
    let versions = sqlx::query_scalar::<_, String>(
        "SELECT version FROM releases WHERE release_status = 'released' OR (? = 'prerelease' AND release_status = 'prerelease')",
    ).bind(channel).fetch_all(db).await?;
    let mut candidates = Vec::with_capacity(versions.len());
    for label in versions {
        let version = release_version(&label)?;
        if channel == "prerelease" || version.pre.is_empty() {
            candidates.push((version, label));
        }
    }
    Ok(candidates
        .into_iter()
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, label)| label))
}

async fn update_version_for_channel(db: &SqlitePool, channel: &str) -> Result<String, AppError> {
    newest_version(db, channel).await?.ok_or_else(|| {
        AppError::not_found(
            "update_not_found",
            "No release is available for this update channel",
        )
    })
}

#[derive(Debug)]
struct PrereleaseRollup {
    version: String,
    notes: Option<String>,
}

async fn previous_prereleases_for_rollup(
    db: &SqlitePool,
    target_version: &str,
) -> Result<Vec<PrereleaseRollup>, AppError> {
    let latest = newest_version(db, "stable")
        .await?
        .map(|v| release_version(&v))
        .transpose()?;
    let target = release_version(target_version)?;
    let rows = sqlx::query(
        "SELECT version, notes FROM releases WHERE release_status = 'prerelease' AND version != ?",
    )
    .bind(target_version)
    .fetch_all(db)
    .await?;
    let mut releases = Vec::new();
    for row in rows {
        let version: String = row.try_get("version")?;
        let parsed = release_version(&version)?;
        if parsed < target && latest.as_ref().is_none_or(|latest| &parsed > latest) {
            releases.push((
                parsed,
                PrereleaseRollup {
                    version,
                    notes: row.try_get("notes")?,
                },
            ));
        }
    }
    releases.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(releases.into_iter().map(|(_, release)| release).collect())
}

fn merge_release_notes(
    target_version: &str,
    target_notes: Option<&str>,
    incorporated: &[PrereleaseRollup],
) -> Option<String> {
    let target_body = release_notes_body(target_notes.unwrap_or_default());
    let mut sections = Vec::new();
    if !target_body.is_empty() {
        sections.push(target_body.to_string());
    }
    for release in incorporated {
        let body = release_notes_body(release.notes.as_deref().unwrap_or_default());
        if !body.is_empty() {
            sections.push(format!("## Included from {}\n\n{}", release.version, body));
        }
    }

    if sections.is_empty() {
        None
    } else {
        Some(format!("# {target_version}\n\n{}", sections.join("\n\n")))
    }
}

fn release_notes_body(notes: &str) -> &str {
    let notes = notes.trim();
    let Some((first_line, remainder)) = notes.split_once('\n') else {
        return if notes.starts_with("# ") { "" } else { notes };
    };
    if first_line.starts_with("# ") {
        remainder.trim_start()
    } else {
        notes
    }
}

async fn load_releases(db: &SqlitePool) -> Result<Vec<ReleaseResponse>, AppError> {
    let rows = sqlx::query(
        r#"
        SELECT version, release_status, notes, created_at
        FROM releases
        ORDER BY
            CASE WHEN release_status = 'prerelease' THEN 0 ELSE 1 END,
            version_major DESC,
            version_minor DESC,
            version_patch DESC
        "#,
    )
    .fetch_all(db)
    .await?;

    let latest = latest_version_optional(db).await?;
    let mut releases = Vec::with_capacity(rows.len());
    for row in rows {
        let version: String = row.try_get("version")?;
        let release_status: String = row.try_get("release_status")?;
        releases.push(ReleaseResponse {
            artifacts: load_artifacts_with_downloads(db, Some(&version)).await?,
            is_latest: latest.as_deref() == Some(version.as_str()),
            is_prerelease: release_status == "prerelease",
            version,
            release_status,
            notes: row.try_get("notes")?,
            created_at: row.try_get("created_at")?,
        });
    }

    // All stored labels are validated on upload. SemVer orders beta.10 after beta.2.
    releases.sort_by_cached_key(|release| {
        (
            !release.is_prerelease,
            std::cmp::Reverse(release_version(&release.version).ok()),
        )
    });
    Ok(releases)
}

async fn load_release(db: &SqlitePool, version: &str) -> Result<ReleaseResponse, AppError> {
    let row = sqlx::query(
        "SELECT version, release_status, notes, created_at FROM releases WHERE version = ?",
    )
    .bind(version)
    .fetch_optional(db)
    .await?
    .ok_or_else(|| AppError::not_found("release_not_found", "Release not found"))?;
    let latest = latest_version_optional(db).await?;
    let version: String = row.try_get("version")?;
    let release_status: String = row.try_get("release_status")?;

    Ok(ReleaseResponse {
        artifacts: load_artifacts_with_downloads(db, Some(&version)).await?,
        is_latest: latest.as_deref() == Some(version.as_str()),
        is_prerelease: release_status == "prerelease",
        version,
        release_status,
        notes: row.try_get("notes")?,
        created_at: row.try_get("created_at")?,
    })
}

async fn load_artifacts_with_downloads(
    db: &SqlitePool,
    version: Option<&str>,
) -> Result<Vec<ArtifactResponse>, AppError> {
    let rows = if let Some(version) = version {
        sqlx::query(
            r#"
            SELECT a.id,
                   a.release_version,
                   a.target,
                   a.filename,
                   a.sha256,
                   a.size_bytes,
                   COUNT(d.id) AS downloads
            FROM release_artifacts a
            LEFT JOIN download_events d ON d.artifact_id = a.id
            WHERE a.release_version = ?
            GROUP BY a.id
            ORDER BY a.target
            "#,
        )
        .bind(version)
        .fetch_all(db)
        .await?
    } else {
        sqlx::query(
            r#"
            SELECT a.id,
                   a.release_version,
                   a.target,
                   a.filename,
                   a.sha256,
                   a.size_bytes,
                   COUNT(d.id) AS downloads
            FROM release_artifacts a
            LEFT JOIN download_events d ON d.artifact_id = a.id
            GROUP BY a.id
            ORDER BY a.release_version DESC, a.target
            "#,
        )
        .fetch_all(db)
        .await?
    };

    rows.into_iter()
        .map(|row| {
            let version: String = row.try_get("release_version")?;
            let filename: String = row.try_get("filename")?;
            Ok(ArtifactResponse {
                id: row.try_get("id")?,
                version: version.clone(),
                target: row.try_get("target")?,
                filename: filename.clone(),
                sha256: row.try_get("sha256")?,
                size_bytes: row.try_get("size_bytes")?,
                downloads: row.try_get("downloads")?,
                download_url: format!("/releases/{}/{}", version, filename),
            })
        })
        .collect()
}

async fn ensure_settings(db: &SqlitePool) -> Result<(), AppError> {
    sqlx::query(
        r#"
        INSERT INTO release_settings (id, created_at, updated_at)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO NOTHING
        "#,
    )
    .bind(unix_timestamp())
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn admin_exists(db: &SqlitePool) -> Result<bool, AppError> {
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM hub_admins")
        .fetch_one(db)
        .await?;
    Ok(count > 0)
}

async fn require_claimed(db: &SqlitePool) -> Result<(), AppError> {
    if admin_exists(db).await? {
        Ok(())
    } else {
        Err(AppError::forbidden(
            "hub_unclaimed",
            "Vashti Hub must be claimed by an admin before releases are served.",
        ))
    }
}

async fn ensure_admin_setup_key(db: &SqlitePool, config: &Config) -> Result<(), AppError> {
    if admin_exists(db).await? {
        clear_setup_key(db, config).await?;
        return Ok(());
    }

    let path = setup_key_path(config);
    let key = match fs::read_to_string(&path).await {
        Ok(existing) if !existing.trim().is_empty() => existing.trim().to_string(),
        _ => {
            let key = generate_token();
            fs::write(&path, format!("{}\n", key)).await?;
            set_owner_only_permissions(&path)?;
            tracing::warn!(
                path = %path.display(),
                "generated initial vashti hub admin setup key"
            );
            key
        }
    };

    sqlx::query(
        r#"
        UPDATE release_settings
        SET admin_setup_key_hash = ?, updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(token_hash(&key))
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn require_admin_session(state: &AppState, jar: &CookieJar) -> Result<(), AppError> {
    require_claimed(&state.db).await?;

    if admin_from_session(&state.db, jar).await?.is_some() {
        return Ok(());
    }

    Err(AppError::unauthorized())
}

async fn consume_upload_key(state: &AppState, headers: &HeaderMap) -> Result<(), AppError> {
    let Some(token) = bearer_token(headers) else {
        return Err(AppError::unauthorized());
    };
    state
        .upload_keys
        .consume(token)
        .await
        .then_some(())
        .ok_or_else(AppError::unauthorized)
}

async fn create_admin(db: &SqlitePool, password: &str) -> Result<(), AppError> {
    let password_hash = hash_password(password)?;
    let now = unix_timestamp();
    sqlx::query(
        r#"
        INSERT INTO hub_admins (id, password_hash, created_at, updated_at)
        VALUES (1, ?, ?, ?)
        "#,
    )
    .bind(password_hash)
    .bind(now)
    .bind(now)
    .execute(db)
    .await?;

    Ok(())
}

async fn update_admin_password(db: &SqlitePool, password: &str) -> Result<(), AppError> {
    let password_hash = hash_password(password)?;
    sqlx::query("UPDATE hub_admins SET password_hash = ?, updated_at = ? WHERE id = 1")
        .bind(password_hash)
        .bind(unix_timestamp())
        .execute(db)
        .await?;

    Ok(())
}

async fn verify_admin_password(db: &SqlitePool, password: &str) -> Result<bool, AppError> {
    let Some(password_hash): Option<String> =
        sqlx::query_scalar("SELECT password_hash FROM hub_admins WHERE id = 1")
            .fetch_optional(db)
            .await?
    else {
        return Ok(false);
    };

    verify_password(password, &password_hash)
}

async fn create_admin_session(db: &SqlitePool) -> Result<String, AppError> {
    cleanup_expired_sessions(db).await?;
    let now = unix_timestamp();
    let session_id = generate_token();
    sqlx::query(
        r#"
        INSERT INTO hub_admin_sessions (id, expires_at, created_at)
        VALUES (?, ?, ?)
        "#,
    )
    .bind(&session_id)
    .bind(now + ADMIN_SESSION_TTL_SECONDS)
    .bind(now)
    .execute(db)
    .await?;

    Ok(session_id)
}

async fn admin_from_session(db: &SqlitePool, jar: &CookieJar) -> Result<Option<String>, AppError> {
    let Some(cookie) = jar.get(ADMIN_SESSION_COOKIE) else {
        return Ok(None);
    };

    let now = unix_timestamp();
    let row = sqlx::query("SELECT id, expires_at FROM hub_admin_sessions WHERE id = ?")
        .bind(cookie.value())
        .fetch_optional(db)
        .await?;

    let Some(row) = row else {
        return Ok(None);
    };
    let expires_at: i64 = row.try_get("expires_at")?;
    if expires_at <= now {
        delete_admin_session(db, cookie.value()).await?;
        return Ok(None);
    }

    Ok(Some(row.try_get("id")?))
}

async fn delete_admin_session(db: &SqlitePool, session_id: &str) -> Result<(), AppError> {
    sqlx::query("DELETE FROM hub_admin_sessions WHERE id = ?")
        .bind(session_id)
        .execute(db)
        .await?;

    Ok(())
}

async fn cleanup_expired_sessions(db: &SqlitePool) -> Result<(), AppError> {
    sqlx::query("DELETE FROM hub_admin_sessions WHERE expires_at <= ?")
        .bind(unix_timestamp())
        .execute(db)
        .await?;

    Ok(())
}

async fn clear_setup_key(db: &SqlitePool, config: &Config) -> Result<(), AppError> {
    let path = setup_key_path(config);
    match fs::remove_file(&path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    sqlx::query(
        "UPDATE release_settings SET admin_setup_key_hash = NULL, updated_at = ? WHERE id = 1",
    )
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn store_reset_key(db: &SqlitePool, config: &Config, key: &str) -> Result<(), AppError> {
    let path = reset_key_path(config);
    fs::write(&path, format!("{}\n", key)).await?;
    set_owner_only_permissions(&path)?;

    sqlx::query(
        r#"
        UPDATE release_settings
        SET reset_key_hash = ?, reset_key_generated_at = ?, updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(token_hash(key))
    .bind(unix_timestamp())
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn clear_reset_key(db: &SqlitePool, config: &Config) -> Result<(), AppError> {
    let path = reset_key_path(config);
    match fs::remove_file(&path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    sqlx::query(
        r#"
        UPDATE release_settings
        SET reset_key_hash = NULL, reset_key_generated_at = NULL, updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn clear_legacy_upload_token(db: &SqlitePool, config: &Config) -> Result<(), AppError> {
    let token_path = config.data_dir.join("upload-token.txt");
    match fs::remove_file(&token_path).await {
        Ok(()) => {
            tracing::info!(
                path = %token_path.display(),
                "removed legacy persistent vashti hub upload token"
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    sqlx::query(
        "UPDATE release_settings SET upload_token_hash = NULL, updated_at = ? WHERE id = 1",
    )
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn update_latest_version(db: &SqlitePool) -> Result<(), AppError> {
    let latest = newest_version(db, "stable").await?;

    sqlx::query("UPDATE release_settings SET latest_version = ?, updated_at = ? WHERE id = 1")
        .bind(latest)
        .bind(unix_timestamp())
        .execute(db)
        .await?;

    Ok(())
}

async fn latest_version(db: &SqlitePool) -> Result<String, AppError> {
    latest_version_optional(db)
        .await?
        .ok_or_else(|| AppError::not_found("release_not_found", "No releases have been uploaded"))
}

async fn latest_version_optional(db: &SqlitePool) -> Result<Option<String>, AppError> {
    Ok(
        sqlx::query_scalar("SELECT latest_version FROM release_settings WHERE id = 1")
            .fetch_one(db)
            .await?,
    )
}

async fn record_download(
    db: &SqlitePool,
    headers: &HeaderMap,
    artifact_id: &str,
    version: &str,
    target: &str,
    kind: &str,
) -> Result<(), AppError> {
    let user_agent_hash = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(token_hash);

    sqlx::query(
        r#"
        INSERT INTO download_events
            (id, artifact_id, release_version, target, kind, user_agent_hash, downloaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(artifact_id)
    .bind(version)
    .bind(target)
    .bind(kind)
    .bind(user_agent_hash)
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn record_page_hit_best_effort(
    state: &AppState,
    headers: &HeaderMap,
    remote_addr: SocketAddr,
    jar: &CookieJar,
    path: &str,
) {
    if matches!(admin_from_session(&state.db, jar).await, Ok(Some(_))) {
        return;
    }

    if let Err(error) = record_page_hit(state, headers, remote_addr, path).await {
        tracing::warn!(?error, path, "failed to record page hit");
    }
}

async fn record_page_hit(
    state: &AppState,
    headers: &HeaderMap,
    remote_addr: SocketAddr,
    path: &str,
) -> Result<(), AppError> {
    let visitor_hash = token_hash(&client_key(
        headers,
        remote_addr,
        state.config.trust_proxy_headers,
    ));
    let user_agent_hash = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(token_hash);

    sqlx::query(
        r#"
        INSERT INTO page_hits
            (id, path, visitor_hash, user_agent_hash, visited_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(path)
    .bind(visitor_hash)
    .bind(user_agent_hash)
    .bind(unix_timestamp())
    .execute(&state.db)
    .await?;

    Ok(())
}

async fn load_page_hit_summaries(db: &SqlitePool) -> Result<Vec<PageHitSummary>, AppError> {
    let rows = sqlx::query(
        r#"
        SELECT path,
               COUNT(*) AS hits,
               COUNT(DISTINCT visitor_hash) AS visitors
        FROM page_hits
        GROUP BY path
        ORDER BY path
        "#,
    )
    .fetch_all(db)
    .await?;

    rows.into_iter()
        .map(|row| {
            Ok(PageHitSummary {
                path: row.try_get("path")?,
                hits: row.try_get("hits")?,
                visitors: row.try_get("visitors")?,
            })
        })
        .collect()
}

async fn page_hit_totals(db: &SqlitePool) -> Result<(i64, i64), AppError> {
    let row = sqlx::query(
        r#"
        SELECT COUNT(*) AS hits,
               COUNT(DISTINCT visitor_hash) AS visitors
        FROM page_hits
        "#,
    )
    .fetch_one(db)
    .await?;

    Ok((row.try_get("hits")?, row.try_get("visitors")?))
}

fn html(body: &'static str) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], body)
}

fn text_file(filename: &str, body: String) -> Response {
    (
        [
            (
                header::CONTENT_TYPE,
                "text/plain; charset=utf-8".to_string(),
            ),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}\"", filename),
            ),
        ],
        body,
    )
        .into_response()
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(&AUTHORIZATION)?.to_str().ok()?.trim();
    value.strip_prefix("Bearer ").map(str::trim)
}

fn admin_session_cookie(session_id: &str, secure: bool) -> Cookie<'static> {
    Cookie::build((ADMIN_SESSION_COOKIE, session_id.to_string()))
        .path("/")
        .http_only(true)
        .secure(secure)
        .same_site(SameSite::Lax)
        .max_age(time::Duration::seconds(ADMIN_SESSION_TTL_SECONDS))
        .build()
}

fn expired_admin_session_cookie(secure: bool) -> Cookie<'static> {
    Cookie::build((ADMIN_SESSION_COOKIE, ""))
        .path("/")
        .http_only(true)
        .secure(secure)
        .same_site(SameSite::Lax)
        .max_age(time::Duration::seconds(0))
        .build()
}

fn setup_key_path(config: &Config) -> PathBuf {
    config.data_dir.join(ADMIN_SETUP_KEY_FILE)
}

fn reset_key_path(config: &Config) -> PathBuf {
    config.data_dir.join(ADMIN_RESET_KEY_FILE)
}

fn cat_command(path: &Path) -> String {
    format!("cat {}", shell_quote_path(path))
}

fn shell_quote_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    if value.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '/' | '.' | '_' | '-')
    }) {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn should_secure_cookie(config: &Config, headers: &HeaderMap) -> bool {
    config.secure_session_cookies
        || (config.trust_proxy_headers
            && header_text(headers, &X_FORWARDED_PROTO)
                .is_some_and(|value| value.eq_ignore_ascii_case("https")))
}

fn require_same_origin(headers: &HeaderMap) -> Result<(), AppError> {
    let Some(host) = header_text(headers, &header::HOST).map(str::trim) else {
        return Ok(());
    };

    if let Some(origin) = header_text(headers, &header::ORIGIN)
        && !url_matches_host(origin, host)
    {
        return Err(AppError::forbidden(
            "cross_origin_request",
            "Cross-origin admin requests are not allowed.",
        ));
    }

    if header_text(headers, &header::ORIGIN).is_none()
        && let Some(referer) = header_text(headers, &header::REFERER)
        && !url_matches_host(referer, host)
    {
        return Err(AppError::forbidden(
            "cross_origin_request",
            "Cross-origin admin requests are not allowed.",
        ));
    }

    Ok(())
}

fn url_matches_host(value: &str, host: &str) -> bool {
    let value = value.trim();
    let Some(rest) = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
    else {
        return false;
    };

    rest == host
        || rest
            .strip_prefix(host)
            .is_some_and(|tail| tail.starts_with('/'))
}

fn client_key(headers: &HeaderMap, remote_addr: SocketAddr, trust_proxy_headers: bool) -> String {
    if trust_proxy_headers {
        if let Some(forwarded_for) = header_text(headers, &X_FORWARDED_FOR)
            && let Some(first_ip) = forwarded_for.split(',').next().map(str::trim)
            && !first_ip.is_empty()
        {
            return format!("ip:{}", compact_key_part(first_ip, 96));
        }

        if let Some(real_ip) = header_text(headers, &X_REAL_IP)
            && !real_ip.trim().is_empty()
        {
            return format!("ip:{}", compact_key_part(real_ip.trim(), 96));
        }
    }

    format!("ip:{}", remote_addr.ip())
}

fn compact_key_part(value: &str, max_chars: usize) -> String {
    value
        .trim()
        .chars()
        .take(max_chars)
        .collect::<String>()
        .to_ascii_lowercase()
}

fn header_text<'a>(headers: &'a HeaderMap, name: &HeaderName) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn release_version(value: &str) -> Result<semver::Version, AppError> {
    let bare = value.trim().strip_prefix('v').unwrap_or(value.trim());
    let version = semver::Version::parse(bare).map_err(|_| {
        AppError::bad_request(
            "invalid_version",
            "Use a version such as v0.13.0 or v0.13.0-beta.1",
        )
    })?;
    if bare.len() > 96 || !version.build.is_empty() {
        return Err(AppError::bad_request(
            "invalid_version",
            "Version must be at most 96 characters, without build metadata",
        ));
    }
    Ok(version)
}

fn normalize_version_label(value: &str) -> Result<(String, i64, i64, i64), AppError> {
    let version = release_version(value)?;
    let part = |value| {
        i64::try_from(value).map_err(|_| {
            AppError::bad_request("invalid_version", "Version number is too large to store")
        })
    };
    Ok((
        format!("v{version}"),
        part(version.major)?,
        part(version.minor)?,
        part(version.patch)?,
    ))
}

fn require_stable_version(version: &str) -> Result<(), AppError> {
    if !release_version(version)?.pre.is_empty() {
        return Err(AppError::bad_request(
            "prerelease_version",
            "Upload a version without a prerelease suffix before promoting to stable",
        ));
    }
    Ok(())
}

fn validate_target(value: &str) -> Result<String, AppError> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 96
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(AppError::bad_request(
            "invalid_target",
            "Target must contain only letters, numbers, hyphens, and underscores",
        ));
    }

    Ok(value.to_string())
}

fn text_field(bytes: &[u8]) -> Result<String, AppError> {
    String::from_utf8(bytes.to_vec())
        .map(|value| value.trim().to_string())
        .map_err(|_| AppError::bad_request("invalid_text", "Multipart text field is invalid"))
}

fn safe_filename(value: &str) -> String {
    let filename = Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact.bin")
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        })
        .collect::<String>();

    if filename.is_empty() {
        "artifact.bin".to_string()
    } else {
        filename
    }
}

fn hash_password(password: &str) -> Result<String, AppError> {
    let password = password.trim();
    if password.len() < 8 || password.len() > 1024 {
        return Err(AppError::bad_request(
            "invalid_password",
            "Password must be between 8 and 1024 bytes.",
        ));
    }

    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|error| {
            tracing::error!(?error, "password hashing failed");
            AppError::internal("Password hashing failed")
        })
}

fn verify_password(password: &str, password_hash: &str) -> Result<bool, AppError> {
    if password.len() > 1024 {
        return Ok(false);
    }
    let parsed_hash = PasswordHash::new(password_hash).map_err(|error| {
        tracing::error!(?error, "stored password hash is invalid");
        AppError::internal("Stored password hash is invalid")
    })?;

    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok())
}

fn generate_token() -> String {
    let mut bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn token_hash(value: &str) -> String {
    sha256_hex(value.as_bytes())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs() as i64
}

fn set_owner_only_permissions(path: &Path) -> Result<(), AppError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path)?.permissions();
        permissions.set_mode(0o600);
        std::fs::set_permissions(path, permissions)?;
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install terminate signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn release_test_db() -> SqlitePool {
        let db = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("in-memory database should open");
        sqlx::migrate!("./migrations")
            .run(&db)
            .await
            .expect("Hub migrations should run");
        ensure_settings(&db)
            .await
            .expect("settings should initialize");
        db
    }

    async fn insert_test_release(db: &SqlitePool, version: &str, status: &str) {
        let (_, major, minor, patch) =
            normalize_version_label(version).expect("test version should be valid");
        sqlx::query(
            r#"
            INSERT INTO releases
                (version, version_major, version_minor, version_patch, release_status, created_at)
            VALUES (?, ?, ?, ?, ?, 1)
            "#,
        )
        .bind(version)
        .bind(major)
        .bind(minor)
        .bind(patch)
        .bind(status)
        .execute(db)
        .await
        .expect("test release should insert");
    }

    fn artifact(filename: &str, bytes: &[u8]) -> UploadedArtifact {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), bytes).unwrap();
        UploadedArtifact {
            filename: filename.into(),
            content_type: "application/octet-stream".into(),
            file,
            size_bytes: bytes.len() as i64,
            sha256: sha256_hex(bytes),
        }
    }

    #[test]
    fn only_get_requests_count_as_artifact_downloads() {
        assert!(counts_as_download(&Method::GET));
        assert!(!counts_as_download(&Method::HEAD));
        assert!(!counts_as_download(&Method::OPTIONS));
    }

    #[test]
    fn assembles_multiple_release_artifacts_in_field_order() {
        let artifacts = assemble_artifact_uploads(
            vec!["linux-x86_64".to_string(), "android-universal".to_string()],
            vec![
                artifact("vashti-linux-x86_64.tar.gz", b"linux"),
                artifact("vashti-android.apk", b"android"),
            ],
            vec![
                "linux-signature".to_string(),
                "android-signature".to_string(),
            ],
        )
        .expect("valid artifacts should be accepted");

        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].target, "linux-x86_64");
        assert_eq!(artifacts[0].filename, "vashti-linux-x86_64.tar.gz");
        assert_eq!(artifacts[1].target, "android-universal");
        assert_eq!(artifacts[1].filename, "vashti-android.apk");
    }

    #[test]
    fn rejects_mismatched_artifact_and_target_counts() {
        let error = assemble_artifact_uploads(
            vec!["linux-x86_64".to_string(), "android-universal".to_string()],
            vec![artifact("vashti-linux-x86_64.tar.gz", b"linux")],
            vec!["linux-signature".to_string()],
        )
        .expect_err("unpaired fields must be rejected");

        assert_eq!(error.code, "artifact_metadata_mismatch");
    }

    #[test]
    fn rejects_duplicate_targets_and_filenames() {
        let duplicate_target = assemble_artifact_uploads(
            vec!["linux-x86_64".to_string(), "linux-x86_64".to_string()],
            vec![
                artifact("one.tar.gz", b"one"),
                artifact("two.tar.gz", b"two"),
            ],
            vec!["one".to_string(), "two".to_string()],
        )
        .expect_err("duplicate targets must be rejected");
        assert_eq!(duplicate_target.code, "duplicate_target");

        let duplicate_filename = assemble_artifact_uploads(
            vec!["linux-x86_64".to_string(), "android-universal".to_string()],
            vec![
                artifact("vashti.bin", b"one"),
                artifact("vashti.bin", b"two"),
            ],
            vec!["one".to_string(), "two".to_string()],
        )
        .expect_err("duplicate filenames must be rejected");
        assert_eq!(duplicate_filename.code, "duplicate_filename");
    }

    #[test]
    fn rejects_empty_artifacts() {
        let error = assemble_artifact_uploads(
            vec!["android-universal".to_string()],
            vec![artifact("vashti-android.apk", b"")],
            vec!["android-signature".to_string()],
        )
        .expect_err("empty artifacts must be rejected");

        assert_eq!(error.code, "empty_artifact");
    }

    #[test]
    fn rejects_artifacts_without_a_valid_release_signature() {
        let artifacts = assemble_artifact_uploads(
            vec!["linux-x86_64".to_string()],
            vec![artifact("vashti-linux-x86_64.tar.gz", b"linux")],
            vec!["not-a-signature".to_string()],
        )
        .expect("artifact metadata should assemble before verification");

        let error = validate_upload_signatures("v1.2.3", &artifacts)
            .expect_err("invalid signatures must be rejected before release storage changes");
        assert_eq!(error.code, "invalid_update_signature");
    }

    #[test]
    fn merges_prerelease_notes_without_duplicate_top_level_headings() {
        let incorporated = vec![
            PrereleaseRollup {
                version: "v0.9.1".to_string(),
                notes: Some("# v0.9.1\n\nFirst staged change.".to_string()),
            },
            PrereleaseRollup {
                version: "v0.9.0".to_string(),
                notes: Some("Second staged change.".to_string()),
            },
        ];

        let notes = merge_release_notes(
            "v0.9.2",
            Some("# v0.9.2\n\nCurrent release."),
            &incorporated,
        )
        .expect("nonempty notes should merge");

        assert_eq!(
            notes,
            "# v0.9.2\n\nCurrent release.\n\n## Included from v0.9.1\n\nFirst staged change.\n\n## Included from v0.9.0\n\nSecond staged change."
        );
    }

    #[test]
    fn empty_release_notes_stay_empty() {
        assert_eq!(merge_release_notes("v1.0.0", None, &[]), None);
    }

    #[tokio::test]
    async fn update_channels_and_rollup_support_multiple_prereleases() {
        let db = release_test_db().await;
        insert_test_release(&db, "v0.9.0", "released").await;
        insert_test_release(&db, "v0.9.1", "prerelease").await;
        insert_test_release(&db, "v0.9.2", "prerelease").await;
        insert_test_release(&db, "v0.9.3", "prerelease").await;

        assert_eq!(
            update_version_for_channel(&db, "stable")
                .await
                .expect("stable release should resolve"),
            "v0.9.0"
        );
        assert_eq!(
            update_version_for_channel(&db, "prerelease")
                .await
                .expect("highest prerelease should resolve"),
            "v0.9.3"
        );

        let candidates = previous_prereleases_for_rollup(&db, "v0.9.2")
            .await
            .expect("rollup candidates should load");
        assert_eq!(
            candidates
                .iter()
                .map(|release| release.version.as_str())
                .collect::<Vec<_>>(),
            vec!["v0.9.1"]
        );
    }
    #[test]
    fn prerelease_labels_are_canonical_and_cannot_be_promoted() {
        assert_eq!(
            normalize_version_label("0.13.0-beta.1").unwrap(),
            ("v0.13.0-beta.1".into(), 0, 13, 0)
        );
        for invalid in [
            "0.13.0-beta.01",
            "0.13.0+build.1",
            "0.13",
            "../0.13.0",
            "00.13.0",
        ] {
            assert!(normalize_version_label(invalid).is_err(), "{invalid}");
        }
        assert!(require_stable_version("v0.13.0-beta.1").is_err());
        assert!(require_stable_version("v0.13.0").is_ok());
    }

    #[tokio::test]
    async fn semver_channels_listing_and_rollup_order_beta_versions() {
        let db = release_test_db().await;
        insert_test_release(&db, "v0.12.0", "released").await;
        for version in ["v0.13.0-beta.10", "v0.13.0-beta.1", "v0.13.0-beta.2"] {
            insert_test_release(&db, version, "prerelease").await;
        }
        update_latest_version(&db).await.unwrap();
        assert_eq!(latest_version(&db).await.unwrap(), "v0.12.0");
        assert_eq!(
            update_version_for_channel(&db, "prerelease").await.unwrap(),
            "v0.13.0-beta.10"
        );
        let expected = ["v0.13.0-beta.10", "v0.13.0-beta.2", "v0.13.0-beta.1"];
        let candidates = previous_prereleases_for_rollup(&db, "v0.13.0")
            .await
            .unwrap();
        assert_eq!(
            candidates
                .iter()
                .map(|r| r.version.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        let releases = load_releases(&db).await.unwrap();
        assert_eq!(
            releases[..3]
                .iter()
                .map(|r| r.version.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        insert_test_release(&db, "v0.13.0", "prerelease").await;
        assert_eq!(
            update_version_for_channel(&db, "prerelease").await.unwrap(),
            "v0.13.0"
        );
        // Defense in depth: even an incorrectly marked beta cannot enter stable.
        insert_test_release(&db, "v0.14.0-beta.1", "released").await;
        assert_eq!(
            update_version_for_channel(&db, "stable").await.unwrap(),
            "v0.12.0"
        );
    }

    #[test]
    fn accepts_signed_kuno_artifact_metadata_alongside_vashti() {
        let artifacts = assemble_artifact_uploads(
            vec!["linux-x86_64".into(), "kuno-linux-x86_64".into()],
            vec![
                artifact("vashti-linux-x86_64.tar.gz", b"server"),
                artifact("kuno-linux-x86_64.tar.gz", b"inference"),
            ],
            vec!["server-signature".into(), "kuno-signature".into()],
        )
        .unwrap();
        assert_eq!(artifacts[1].target, "kuno-linux-x86_64");
    }
    #[tokio::test]
    async fn streamed_upload_failures_remove_temporary_files() {
        use axum::{
            body::{Bytes, to_bytes},
            http::Request,
        };
        use tower::ServiceExt;
        for (name, suffix, expected) in [
            (
                "artifact",
                "\r\n--boundary--\r\n",
                "invalid_update_signature",
            ),
            ("artifact", "", "invalid_multipart"),
            ("notes", "\r\n--boundary--\r\n", "field_too_large"),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().to_path_buf();
            let app = Router::new()
                .route(
                    "/",
                    post(move |multipart: Multipart| async move {
                        parse_upload(multipart, &path).await.map(|_| StatusCode::OK)
                    }),
                )
                .layer(DefaultBodyLimit::max(16 * 1024 * 1024));
            let header = format!(
                "--boundary\r\nContent-Disposition: form-data; name=\"version\"\r\n\r\nv0.13.0-beta.1\r\n--boundary\r\nContent-Disposition: form-data; name=\"target\"\r\n\r\nkuno-linux-x86_64\r\n--boundary\r\nContent-Disposition: form-data; name=\"signature\"\r\n\r\ninvalid\r\n--boundary\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"kuno.tar.gz\"\r\n\r\n"
            );
            let chunk = Bytes::from(vec![42; 64 * 1024]);
            let chunks = std::iter::once(Ok::<_, std::io::Error>(Bytes::from(header)))
                .chain((0..128).map(move |_| Ok(chunk.clone())))
                .chain(std::iter::once(Ok(Bytes::from_static(suffix.as_bytes()))));
            let request = Request::builder()
                .method("POST")
                .uri("/")
                .header(
                    header::CONTENT_TYPE,
                    "multipart/form-data; boundary=boundary",
                )
                .body(Body::from_stream(futures_util::stream::iter(chunks)))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(String::from_utf8_lossy(&body).contains(expected));
            assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        }
    }

    #[tokio::test]
    async fn startup_only_removes_owned_interrupted_upload_files() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join(".upload-aborted"), b"temporary")
            .await
            .unwrap();
        fs::write(directory.path().join("keep.txt"), b"keep")
            .await
            .unwrap();
        fs::create_dir(directory.path().join("v0.13.0-beta.1"))
            .await
            .unwrap();
        cleanup_upload_files(directory.path()).await.unwrap();
        assert!(!directory.path().join(".upload-aborted").exists());
        assert!(directory.path().join("keep.txt").exists());
        assert!(directory.path().join("v0.13.0-beta.1").exists());
    }
}
