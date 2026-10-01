use super::{models::*, service};
use crate::{
    app_state::AppState, auth, backends, error::ApiError, permissions::service as permissions,
    rate_limit,
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::{HeaderMap, HeaderValue, header},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::Row;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(super::transient::router())
        .route("/settings", get(get_settings).put(put_settings))
        .route("/capabilities", get(capabilities))
        .route("/hosts", get(hosts))
        .route("/preferences", get(get_preferences).put(put_preferences))
        .route("/sessions", get(list_sessions).post(create_session))
        .route(
            "/sessions/{id}",
            get(get_session).put(update_session).delete(archive_session),
        )
        .route("/sessions/{id}/title", axum::routing::patch(rename_session))
        .route("/sessions/{id}/history", get(history))
        .route("/sessions/{id}/results", get(results))
        .route("/sessions/{id}/runs", get(runs).post(submit))
        .route("/runs/{id}", get(run))
        .route("/runs/active", get(active_runs))
        .route("/runs/{id}/stop", post(stop))
        .route("/results/{id}/select", post(select))
        .route("/gallery", get(gallery))
        .route("/gallery/copy/{id}", post(copy))
        .route("/gallery/{id}", axum::routing::delete(remove_gallery))
        .route(
            "/assets",
            post(upload).layer(DefaultBodyLimit::max(13 * 1024 * 1024)),
        )
        .route("/assets/{id}", get(asset))
        .route("/assets/{id}/settings", get(import_settings))
}
pub(super) async fn user(state: &AppState, jar: &CookieJar) -> Result<String, ApiError> {
    let user =
        auth::service::require_user(&state.db, jar, &state.config.session_cookie_name).await?;
    permissions::ensure_tool_records(&state.db).await?;
    let tags = permissions::effective_user_tag_ids(&state.db, &user.id).await?;
    let tools = permissions::tool_tags_by_tool(&state.db).await?;
    if !tools
        .get("images")
        .is_some_and(|allowed| permissions::has_matching_tag(&tags, allowed))
    {
        return Err(ApiError::forbidden(
            "images_not_allowed",
            "Images are unavailable for this account. Ask an administrator.",
        ));
    }
    Ok(user.id)
}
#[derive(Deserialize, Default)]
struct Page {
    #[serde(default)]
    offset: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionInput {
    title: Option<String>,
    mode: String,
    draft: Draft,
    revision: Option<i64>,
}
async fn list_sessions(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(page): Query<Page>,
) -> Result<Json<Vec<Session>>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::sessions(&state.db, &u, page.offset).await?))
}
async fn create_session(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<SessionInput>,
) -> Result<Json<Session>, ApiError> {
    let u = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("image-session", &u), 60, 60)
        .await?;
    Ok(Json(
        service::create_session(
            &state.db,
            &u,
            input.title.as_deref().unwrap_or("New images"),
            &input.mode,
            input.draft,
        )
        .await?,
    ))
}
async fn get_session(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Session>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::get_session(&state.db, &u, &id).await?))
}
async fn update_session(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<SessionInput>,
) -> Result<Json<Session>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(
        service::update_session(
            &state.db,
            &u,
            &id,
            input.revision.ok_or_else(|| {
                ApiError::bad_request("revision_required", "A draft version is required")
            })?,
            input.title.as_deref(),
            &input.mode,
            input.draft,
        )
        .await?,
    ))
}
async fn history(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Vec<PromptVersion>>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::history(&state.db, &u, &id).await?))
}
async fn results(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Query(page): Query<Page>,
) -> Result<Json<Vec<ImageResult>>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(
        service::results(&state.db, &u, &id, page.offset).await?,
    ))
}
async fn runs(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Query(page): Query<Page>,
) -> Result<Json<Vec<Run>>, ApiError> {
    let u = user(&state, &jar).await?;
    let mut runs = service::runs(&state.db, &u, &id, page.offset).await?;
    for run in &mut runs {
        run.preview = service::preview(&state, &run.id);
    }
    Ok(Json(runs))
}
async fn submit(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<RunRequest>,
) -> Result<Json<Run>, ApiError> {
    let u = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("image-run", &u), 60, 60)
        .await?;
    Ok(Json(service::submit(&state, &u, &id, input).await?))
}
async fn run(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    let u = user(&state, &jar).await?;
    let mut run = service::get_run(&state.db, &u, &id).await?;
    run.preview = service::preview(&state, &id);
    Ok(Json(run))
}
async fn active_runs(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Vec<Run>>, ApiError> {
    let u = user(&state, &jar).await?;
    let ids: Vec<String> = sqlx::query_scalar("SELECT r.id FROM image_runs r JOIN image_sessions s ON s.id=r.session_id WHERE r.user_id=? AND s.chat_id IS NULL AND r.status IN ('queued','improving','generating','saving','stopping') ORDER BY r.created_at LIMIT 2")
        .bind(&u).fetch_all(&state.db).await?;
    let mut runs = Vec::with_capacity(ids.len());
    for id in ids {
        runs.push(service::get_run(&state.db, &u, &id).await?);
    }
    Ok(Json(runs))
}
async fn stop(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Run>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::stop(&state, &u, &id).await?))
}
async fn select(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let u = user(&state, &jar).await?;
    service::select_result(&state.db, &u, &id).await?;
    Ok(Json(json!({"ok":true})))
}
async fn gallery(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(page): Query<Page>,
) -> Result<Json<Vec<GalleryItem>>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::gallery(&state.db, &u, page.offset).await?))
}
async fn copy(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let u = user(&state, &jar).await?;
    service::copy_to_gallery(&state.db, &u, &id).await?;
    Ok(Json(json!({"ok":true})))
}
async fn remove_gallery(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let u = user(&state, &jar).await?;
    sqlx::query("DELETE FROM image_gallery WHERE id=? AND user_id=?")
        .bind(id)
        .bind(u)
        .execute(&state.db)
        .await?;
    Ok(Json(json!({"ok":true})))
}
async fn upload(
    State(state): State<AppState>,
    jar: CookieJar,
    mut multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let u = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("image-upload", &u), 30, 60)
        .await?;
    let field = multipart
        .next_field()
        .await
        .map_err(|_| ApiError::bad_request("image_upload_invalid", "Could not read image"))?
        .ok_or_else(|| ApiError::bad_request("image_upload_missing", "Choose an image"))?;
    let bytes = field.bytes().await.map_err(|_| {
        ApiError::bad_request(
            "image_upload_invalid",
            "Image upload is too large or invalid",
        )
    })?;
    Ok(Json(service::upload(&state.db, &u, bytes.to_vec()).await?))
}
async fn asset(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<(HeaderMap, Vec<u8>), ApiError> {
    let u = user(&state, &jar).await?;
    let (mime, bytes) = service::asset(&state.db, &u, &id).await?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&mime).map_err(|_| ApiError::internal("Invalid image type"))?,
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    Ok((headers, bytes))
}
async fn import_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<kuno_protocol::ImportedImageRecipe>, ApiError> {
    let u = user(&state, &jar).await?;
    let (_, bytes) = service::asset(&state.db, &u, &id).await?;
    let imported = tokio::task::spawn_blocking(move || kuno_image::import_png(&bytes, None))
        .await
        .map_err(|_| ApiError::internal("Could not read image metadata"))?
        .map_err(|code| {
            ApiError::bad_request(
                code,
                "No supported ComfyUI workflow was found in this image",
            )
        })?;
    Ok(Json(imported))
}
async fn get_preferences(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Preferences>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::preferences(&state.db, &u).await?))
}
async fn put_preferences(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(value): Json<Preferences>,
) -> Result<Json<Preferences>, ApiError> {
    let u = user(&state, &jar).await?;
    Ok(Json(service::save_preferences(&state.db, &u, value).await?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceSettings {
    backend_id: Option<String>,
    max_user_bytes: i64,
}
async fn settings(state: &AppState) -> Result<Json<Value>, ApiError> {
    let r = sqlx::query("SELECT backend_id,max_user_bytes FROM image_service_settings WHERE id=1")
        .fetch_one(&state.db)
        .await?;
    let discovery = service::discovery(state, None).await?;
    Ok(Json(
        json!({"backend_id":r.try_get::<Option<String>,_>("backend_id")?,"max_user_bytes":r.try_get::<i64,_>("max_user_bytes")?,"service":discovery}),
    ))
}
#[derive(Deserialize)]
struct CapabilityQuery {
    backend_id: Option<String>,
}
async fn capabilities(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(query): Query<CapabilityQuery>,
) -> Result<Json<super::kuno::Discovery>, ApiError> {
    let u = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("image-discovery", &u), 60, 60)
        .await?;
    Ok(Json(
        service::discovery(&state, query.backend_id.as_deref()).await?,
    ))
}
async fn hosts(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Vec<service::ImageHost>>, ApiError> {
    let u = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("image-discovery", &u), 60, 60)
        .await?;
    Ok(Json(service::image_hosts(&state).await?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TitleInput {
    title: String,
}
async fn rename_session(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<TitleInput>,
) -> Result<Json<Session>, ApiError> {
    let u = user(&state, &jar).await?;
    service::get_session(&state.db, &u, &id).await?;
    if input.title.trim().is_empty() || input.title.len() > 200 {
        return Err(ApiError::bad_request(
            "image_title_invalid",
            "Enter a title of at most 200 characters",
        ));
    }
    sqlx::query(
        "UPDATE image_sessions SET title=?,title_generated=2,updated_at=? WHERE id=? AND user_id=?",
    )
    .bind(input.title.trim())
    .bind(auth::service::unix_timestamp())
    .bind(&id)
    .bind(&u)
    .execute(&state.db)
    .await?;
    Ok(Json(service::get_session(&state.db, &u, &id).await?))
}
async fn archive_session(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let u = user(&state, &jar).await?;
    service::get_session(&state.db, &u, &id).await?;
    let changed = sqlx::query("UPDATE image_sessions SET is_archived=1 WHERE id=? AND user_id=? AND NOT EXISTS (SELECT 1 FROM image_runs WHERE session_id=? AND status IN ('queued','improving','generating','saving','stopping'))")
        .bind(&id).bind(u).bind(&id).execute(&state.db).await?;
    if changed.rows_affected() == 0 {
        return Err(ApiError::conflict(
            "image_session_busy",
            "Stop generation before removing this flow",
        ));
    }
    Ok(Json(json!({"ok":true})))
}
async fn get_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Value>, ApiError> {
    user(&state, &jar).await?;
    settings(&state).await
}
async fn put_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<ServiceSettings>,
) -> Result<Json<Value>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    if !(20 * 1024 * 1024..=1024_i64.pow(4)).contains(&input.max_user_bytes) {
        return Err(ApiError::bad_request(
            "image_quota_invalid",
            "Choose an image limit between 20 MiB and 1 TiB",
        ));
    }
    if let Some(backend) = &input.backend_id
        && backends::service::get_enabled_backend(&state.db, backend)
            .await?
            .is_none()
    {
        return Err(ApiError::bad_request(
            "image_backend_invalid",
            "Choose an enabled backend",
        ));
    }
    sqlx::query("UPDATE image_service_settings SET backend_id=?,max_user_bytes=? WHERE id=1")
        .bind(input.backend_id)
        .bind(input.max_user_bytes)
        .execute(&state.db)
        .await?;
    settings(&state).await
}
