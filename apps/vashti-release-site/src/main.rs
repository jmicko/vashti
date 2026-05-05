use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand_core::{OsRng, RngCore};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use tokio::{fs, signal};
use tower_http::{set_header::SetResponseHeaderLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

const INDEX_HTML: &str = include_str!("../static/index.html");
const ADMIN_HTML: &str = include_str!("../static/admin.html");
const STYLES_CSS: &str = include_str!("../static/styles.css");
const INSTALL_SH: &str = include_str!("../../vashti/packaging/install.sh");
const AUTHORIZATION: HeaderName = HeaderName::from_static("authorization");

#[derive(Clone)]
struct AppState {
    config: Config,
    db: SqlitePool,
}

#[derive(Clone)]
struct Config {
    data_dir: PathBuf,
    artifact_dir: PathBuf,
    database_path: PathBuf,
    bind_addr: SocketAddr,
    max_upload_bytes: usize,
}

impl Config {
    fn from_env() -> Result<Self, AppError> {
        let app_root = env::current_dir().map_err(AppError::from)?;
        let data_dir = env::var_os("VASHTI_RELEASE_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_data_dir(&app_root));
        let data_dir = if data_dir.is_absolute() {
            data_dir
        } else {
            app_root.join(data_dir)
        };

        let bind_addr = env::var("VASHTI_RELEASE_BIND")
            .unwrap_or_else(|_| "127.0.0.1:7781".to_string())
            .parse()
            .map_err(|_| AppError::bad_request("invalid_bind", "VASHTI_RELEASE_BIND is invalid"))?;
        let max_upload_bytes = env::var("VASHTI_RELEASE_MAX_UPLOAD_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(512 * 1024 * 1024);

        Ok(Self {
            artifact_dir: data_dir.join("artifacts"),
            database_path: data_dir.join("release-site.db"),
            data_dir,
            bind_addr,
            max_upload_bytes,
        })
    }
}

fn default_data_dir(app_root: &Path) -> PathBuf {
    if app_root.join("apps/vashti-release-site").is_dir() {
        PathBuf::from("apps/vashti-release-site/data")
    } else {
        PathBuf::from("data")
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
            "A valid admin token is required",
        )
    }

    fn not_found(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, code, message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
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
struct UploadResponse {
    release: ReleaseResponse,
    latest_version: Option<String>,
}

#[derive(Debug, Serialize)]
struct TokenResponse {
    token: String,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    artifacts: Vec<ArtifactResponse>,
}

struct ParsedUpload {
    version: String,
    major: i64,
    minor: i64,
    patch: i64,
    target: String,
    notes: Option<String>,
    filename: String,
    content_type: String,
    bytes: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vashti_release_site=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env()?;
    fs::create_dir_all(&config.data_dir).await?;
    fs::create_dir_all(&config.artifact_dir).await?;

    let database_url = format!("sqlite://{}", config.database_path.display());
    let database_options = SqliteConnectOptions::from_str(&database_url)
        .map_err(AppError::from)?
        .create_if_missing(true);
    let db = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(database_options)
        .await
        .map_err(AppError::from)?;
    sqlx::migrate!("./migrations")
        .run(&db)
        .await
        .map_err(|error| {
            tracing::error!(?error, "migration error");
            AppError::internal("Database migration failed")
        })?;

    ensure_settings(&db).await?;
    ensure_upload_token(&db, &config).await?;

    let bind_addr = config.bind_addr;
    let state = AppState { config, db };
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    tracing::info!(
        "release site listening on http://{}",
        listener.local_addr()?
    );

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(AppError::from)?;

    Ok(())
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/admin", get(admin))
        .route("/styles.css", get(styles))
        .route("/install.sh", get(install_script))
        .route("/api/releases", get(list_releases).post(upload_release))
        .route("/api/releases/latest", get(latest_release))
        .route("/api/stats", get(stats))
        .route("/api/admin/token/rotate", post(rotate_token))
        .route("/releases/latest/VERSION", get(latest_version_file))
        .route("/releases/latest/SHA256SUMS", get(latest_checksums))
        .route("/releases/latest/{filename}", get(download_latest))
        .route("/releases/{version}/VERSION", get(version_file))
        .route("/releases/{version}/SHA256SUMS", get(version_checksums))
        .route("/releases/{version}/{filename}", get(download_version))
        .with_state(state.clone())
        .layer(DefaultBodyLimit::max(state.config.max_upload_bytes))
        .layer(SetResponseHeaderLayer::if_not_present(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(TraceLayer::new_for_http())
}

async fn index() -> impl IntoResponse {
    html(INDEX_HTML)
}

async fn admin() -> impl IntoResponse {
    html(ADMIN_HTML)
}

async fn styles() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLES_CSS,
    )
}

async fn install_script() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "text/x-shellscript; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"install.sh\"",
            ),
        ],
        INSTALL_SH,
    )
}

async fn list_releases(
    State(state): State<AppState>,
) -> Result<Json<Vec<ReleaseResponse>>, AppError> {
    Ok(Json(load_releases(&state.db).await?))
}

async fn latest_release(State(state): State<AppState>) -> Result<Json<ReleaseResponse>, AppError> {
    let version = latest_version(&state.db).await?;
    Ok(Json(load_release(&state.db, &version).await?))
}

async fn stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<StatsResponse>, AppError> {
    require_admin(&state.db, &headers).await?;
    Ok(Json(StatsResponse {
        artifacts: load_artifacts_with_downloads(&state.db, None).await?,
    }))
}

async fn upload_release(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<Json<UploadResponse>, AppError> {
    require_admin(&state.db, &headers).await?;
    let upload = parse_upload(multipart).await?;
    let artifact_dir = state.config.artifact_dir.join(&upload.version);
    fs::create_dir_all(&artifact_dir).await?;
    let storage_path = artifact_dir.join(&upload.filename);
    fs::write(&storage_path, &upload.bytes).await?;

    let sha256 = sha256_hex(&upload.bytes);
    let now = unix_timestamp();
    sqlx::query(
        r#"
        INSERT INTO releases (version, version_major, version_minor, version_patch, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(version) DO UPDATE SET notes = COALESCE(excluded.notes, releases.notes)
        "#,
    )
    .bind(&upload.version)
    .bind(upload.major)
    .bind(upload.minor)
    .bind(upload.patch)
    .bind(&upload.notes)
    .bind(now)
    .execute(&state.db)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO release_artifacts
            (id, release_version, target, filename, content_type, size_bytes, sha256, storage_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(release_version, target) DO UPDATE SET
            filename = excluded.filename,
            content_type = excluded.content_type,
            size_bytes = excluded.size_bytes,
            sha256 = excluded.sha256,
            storage_path = excluded.storage_path,
            created_at = excluded.created_at
        "#,
    )
    .bind(Uuid::new_v4().to_string())
    .bind(&upload.version)
    .bind(&upload.target)
    .bind(&upload.filename)
    .bind(&upload.content_type)
    .bind(upload.bytes.len() as i64)
    .bind(&sha256)
    .bind(storage_path.to_string_lossy().to_string())
    .bind(now)
    .execute(&state.db)
    .await?;

    update_latest_version(&state.db).await?;

    Ok(Json(UploadResponse {
        release: load_release(&state.db, &upload.version).await?,
        latest_version: latest_version_optional(&state.db).await?,
    }))
}

async fn rotate_token(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<TokenResponse>, AppError> {
    require_admin(&state.db, &headers).await?;
    let token = generate_token();
    store_token(&state.db, &state.config, &token).await?;

    Ok(Json(TokenResponse { token }))
}

async fn latest_version_file(State(state): State<AppState>) -> Result<Response, AppError> {
    Ok(text_file("VERSION", latest_version(&state.db).await?))
}

async fn version_file(AxumPath(version): AxumPath<String>) -> Result<Response, AppError> {
    Ok(text_file("VERSION", normalize_version_label(&version)?.0))
}

async fn latest_checksums(State(state): State<AppState>) -> Result<Response, AppError> {
    let version = latest_version(&state.db).await?;
    checksums_for_version(&state, &version).await
}

async fn version_checksums(
    State(state): State<AppState>,
    AxumPath(version): AxumPath<String>,
) -> Result<Response, AppError> {
    let version = normalize_version_label(&version)?.0;
    checksums_for_version(&state, &version).await
}

async fn download_latest(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(filename): AxumPath<String>,
) -> Result<Response, AppError> {
    let version = latest_version(&state.db).await?;
    download_artifact(&state, &headers, &version, &filename, "latest").await
}

async fn download_version(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath((version, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    let version = normalize_version_label(&version)?.0;
    download_artifact(&state, &headers, &version, &filename, "version").await
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
    headers: &HeaderMap,
    version: &str,
    filename: &str,
    kind: &str,
) -> Result<Response, AppError> {
    let row = sqlx::query(
        r#"
        SELECT id, release_version, target, filename, content_type, storage_path
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
    let storage_path: String = row.try_get("storage_path")?;
    let bytes = fs::read(&storage_path).await?;

    record_download(
        &state.db,
        headers,
        &artifact_id,
        &release_version,
        &target,
        kind,
    )
    .await?;

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", filename.replace('"', "")),
        )
        .body(Body::from(bytes))
        .map_err(|_| AppError::internal("Failed to build response"))
}

async fn parse_upload(mut multipart: Multipart) -> Result<ParsedUpload, AppError> {
    let mut version = None;
    let mut target = None;
    let mut notes = None;
    let mut artifact = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| AppError::bad_request("invalid_multipart", "Invalid multipart upload"))?
    {
        let name = field.name().unwrap_or_default().to_string();
        let filename = field.file_name().map(safe_filename);
        let content_type = field
            .content_type()
            .map(str::to_string)
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let bytes = field
            .bytes()
            .await
            .map_err(|_| AppError::bad_request("invalid_multipart", "Invalid multipart field"))?;

        match name.as_str() {
            "version" => {
                version = Some(text_field(bytes.as_ref())?);
            }
            "target" => {
                target = Some(validate_target(&text_field(bytes.as_ref())?)?);
            }
            "notes" => {
                let value = text_field(bytes.as_ref())?;
                notes = (!value.trim().is_empty()).then_some(value);
            }
            "artifact" => {
                artifact = Some((
                    filename.ok_or_else(|| {
                        AppError::bad_request("missing_filename", "Artifact filename is required")
                    })?,
                    content_type,
                    bytes.to_vec(),
                ));
            }
            _ => {}
        }
    }

    let (version, major, minor, patch) = normalize_version_label(
        &version.ok_or_else(|| AppError::bad_request("missing_version", "Version is required"))?,
    )?;
    let target =
        target.ok_or_else(|| AppError::bad_request("missing_target", "Target is required"))?;
    let (filename, content_type, bytes) = artifact
        .ok_or_else(|| AppError::bad_request("missing_artifact", "Artifact is required"))?;
    if bytes.is_empty() {
        return Err(AppError::bad_request(
            "empty_artifact",
            "Artifact cannot be empty",
        ));
    }

    Ok(ParsedUpload {
        version,
        major,
        minor,
        patch,
        target,
        notes,
        filename,
        content_type,
        bytes,
    })
}

async fn load_releases(db: &SqlitePool) -> Result<Vec<ReleaseResponse>, AppError> {
    let rows = sqlx::query(
        r#"
        SELECT version, notes, created_at
        FROM releases
        ORDER BY version_major DESC, version_minor DESC, version_patch DESC
        "#,
    )
    .fetch_all(db)
    .await?;

    let mut releases = Vec::with_capacity(rows.len());
    for row in rows {
        let version: String = row.try_get("version")?;
        releases.push(ReleaseResponse {
            artifacts: load_artifacts_with_downloads(db, Some(&version)).await?,
            version,
            notes: row.try_get("notes")?,
            created_at: row.try_get("created_at")?,
        });
    }

    Ok(releases)
}

async fn load_release(db: &SqlitePool, version: &str) -> Result<ReleaseResponse, AppError> {
    let row = sqlx::query("SELECT version, notes, created_at FROM releases WHERE version = ?")
        .bind(version)
        .fetch_optional(db)
        .await?
        .ok_or_else(|| AppError::not_found("release_not_found", "Release not found"))?;

    Ok(ReleaseResponse {
        artifacts: load_artifacts_with_downloads(db, Some(version)).await?,
        version: row.try_get("version")?,
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

async fn ensure_upload_token(db: &SqlitePool, config: &Config) -> Result<(), AppError> {
    let existing: Option<String> =
        sqlx::query_scalar("SELECT upload_token_hash FROM release_settings WHERE id = 1")
            .fetch_one(db)
            .await?;
    if existing.is_some() {
        return Ok(());
    }

    let token = generate_token();
    store_token(db, config, &token).await?;
    tracing::warn!(
        path = %config.data_dir.join("upload-token.txt").display(),
        "generated initial release-site upload token"
    );
    Ok(())
}

async fn store_token(db: &SqlitePool, config: &Config, token: &str) -> Result<(), AppError> {
    let token_path = config.data_dir.join("upload-token.txt");
    fs::write(&token_path, format!("{}\n", token)).await?;
    set_owner_only_permissions(&token_path)?;

    sqlx::query(
        r#"
        UPDATE release_settings
        SET upload_token_hash = ?, updated_at = ?
        WHERE id = 1
        "#,
    )
    .bind(token_hash(token))
    .bind(unix_timestamp())
    .execute(db)
    .await?;

    Ok(())
}

async fn require_admin(db: &SqlitePool, headers: &HeaderMap) -> Result<(), AppError> {
    let Some(token) = bearer_token(headers) else {
        return Err(AppError::unauthorized());
    };
    let Some(expected_hash): Option<String> =
        sqlx::query_scalar("SELECT upload_token_hash FROM release_settings WHERE id = 1")
            .fetch_one(db)
            .await?
    else {
        return Err(AppError::unauthorized());
    };

    if token_hash(token) != expected_hash {
        return Err(AppError::unauthorized());
    }

    Ok(())
}

async fn update_latest_version(db: &SqlitePool) -> Result<(), AppError> {
    let latest = sqlx::query_scalar::<_, String>(
        r#"
        SELECT version
        FROM releases
        ORDER BY version_major DESC, version_minor DESC, version_patch DESC
        LIMIT 1
        "#,
    )
    .fetch_one(db)
    .await?;

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

fn normalize_version_label(value: &str) -> Result<(String, i64, i64, i64), AppError> {
    let trimmed = value.trim();
    let bare = trimmed.strip_prefix('v').unwrap_or(trimmed);
    let parts = bare.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(AppError::bad_request(
            "invalid_version",
            "Version must look like v0.1.0",
        ));
    }

    let major = parse_version_part(parts[0])?;
    let minor = parse_version_part(parts[1])?;
    let patch = parse_version_part(parts[2])?;

    Ok((
        format!("v{}.{}.{}", major, minor, patch),
        major,
        minor,
        patch,
    ))
}

fn parse_version_part(value: &str) -> Result<i64, AppError> {
    if value.is_empty() || !value.chars().all(|character| character.is_ascii_digit()) {
        return Err(AppError::bad_request(
            "invalid_version",
            "Version must look like v0.1.0",
        ));
    }

    value.parse().map_err(|_| {
        AppError::bad_request("invalid_version", "Version number is too large to store")
    })
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
