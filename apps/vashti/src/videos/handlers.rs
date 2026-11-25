use super::{models::*, service};
use crate::{app_state::AppState, error::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::{HeaderValue, header},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;
use serde_json::{Value, json};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/hosts", get(hosts))
        .route("/preferences", get(preferences).put(save_preferences))
        .route("/sessions", get(sessions).post(create))
        .route("/sessions/{id}", get(session).put(update).delete(remove))
        .route("/sessions/{id}/title", axum::routing::patch(rename))
        .route("/sessions/{id}/runs", post(submit))
        .route("/runs/{id}", get(status))
        .route("/runs/{id}/stop", post(stop))
        .route("/runs/{id}/result", get(result))
        .route(
            "/frames",
            post(frame).layer(DefaultBodyLimit::max(13 * 1024 * 1024)),
        )
        .route("/frames/{id}", get(read_frame))
        .route(
            "/private",
            post(private).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
        )
        .layer(axum::middleware::map_response(
            |mut response: axum::response::Response| async move {
                response
                    .headers_mut()
                    .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
                response
            },
        ))
}
async fn user(state: &AppState, jar: &CookieJar) -> Result<String, ApiError> {
    crate::images::handlers::user(state, jar).await
}
#[derive(Deserialize, Default)]
struct Page {
    #[serde(default)]
    offset: i64,
    #[serde(default)]
    q: String,
    #[serde(default)]
    thumbnail: bool,
}
async fn hosts(State(state): State<AppState>, jar: CookieJar) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(format!("video-discovery:{owner}"), 60, 60)
        .await?;
    Ok(Json(service::hosts(&state).await?))
}
async fn preferences(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Draft>, ApiError> {
    let owner = user(&state, &jar).await?;
    Ok(Json(service::preferences(&state.db, &owner).await?))
}
async fn save_preferences(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(draft): Json<Draft>,
) -> Result<Json<Draft>, ApiError> {
    let owner = user(&state, &jar).await?;
    Ok(Json(
        service::save_preferences(&state.db, &owner, draft).await?,
    ))
}
async fn sessions(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(page): Query<Page>,
) -> Result<Json<Vec<Session>>, ApiError> {
    let owner = user(&state, &jar).await?;
    if page.q.len() > 800 {
        return Err(service::invalid("Search is too long"));
    }
    Ok(Json(
        service::sessions(&state.db, &owner, &page.q, page.offset).await?,
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveSession {
    draft: Draft,
    revision: Option<i64>,
}
async fn create(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<SaveSession>,
) -> Result<Json<Session>, ApiError> {
    let owner = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(format!("video-session:{owner}"), 60, 60)
        .await?;
    Ok(Json(
        service::create_session(&state.db, &owner, input.draft).await?,
    ))
}
async fn session(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    Ok(Json(
        json!({"session":service::session(&state.db,&owner,&id).await?,"runs":service::runs(&state.db,&owner,&id).await?}),
    ))
}
async fn update(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<SaveSession>,
) -> Result<Json<Session>, ApiError> {
    let owner = user(&state, &jar).await?;
    service::validate(&input.draft)?;
    let changed=sqlx::query("UPDATE video_sessions SET draft_json=?,revision=revision+1,updated_at=? WHERE id=? AND user_id=? AND revision=? AND archived_at IS NULL").bind(service::encode(&input.draft)?).bind(crate::auth::service::unix_timestamp()).bind(&id).bind(&owner).bind(input.revision.unwrap_or(-1)).execute(&state.db).await?;
    if changed.rows_affected() != 1 {
        return Err(ApiError::conflict(
            "video_flow_changed",
            "This flow changed elsewhere. Reload before editing.",
        ));
    }
    Ok(Json(service::session(&state.db, &owner, &id).await?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rename {
    title: String,
}
async fn rename(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<Rename>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    service::session(&state.db, &owner, &id).await?;
    if input.title.trim().is_empty() || input.title.chars().count() > 200 {
        return Err(service::invalid("Use a title from 1 to 200 characters"));
    }
    sqlx::query("UPDATE video_sessions SET title=? WHERE id=? AND user_id=?")
        .bind(input.title.trim())
        .bind(&id)
        .bind(owner)
        .execute(&state.db)
        .await?;
    Ok(Json(json!({"updated":true})))
}
async fn remove(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    service::delete_session(&state, &owner, &id).await?;
    Ok(Json(json!({"deleted":true})))
}
async fn submit(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<Request>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(format!("video:{owner}"), 30, 60)
        .await?;
    Ok(Json(
        json!({"id":service::submit(&state,&owner,Some(&id),input,None).await?}),
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateRequest {
    request: Request,
    #[serde(default)]
    expected_user_id: Option<String>,
    #[serde(default)]
    first_frame: Option<String>,
    #[serde(default)]
    last_frame: Option<String>,
    #[serde(default)]
    reference_images: Vec<String>,
}
async fn private(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<PrivateRequest>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    if input
        .expected_user_id
        .as_deref()
        .is_some_and(|id| id != owner)
    {
        return Err(ApiError::conflict(
            "private_identity_changed",
            "The signed-in account changed. Private video generation was not started.",
        ));
    }
    state
        .rate_limiter
        .check(format!("video:{owner}"), 30, 60)
        .await?;
    Ok(Json(
        json!({"id":service::submit(&state,&owner,None,input.request,Some(Inputs { first_frame: input.first_frame, last_frame: input.last_frame, reference_images: input.reference_images })).await?}),
    ))
}
async fn status(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    Ok(Json(service::status(&state, &owner, &id).await?))
}
async fn stop(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    service::stop(&state, &owner, &id).await?;
    Ok(Json(json!({"stopped":true})))
}
async fn result(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let owner = user(&state, &jar).await?;
    let bytes = service::result(&state, &owner, &id).await?;
    Ok((
        [
            (header::CONTENT_TYPE, "video/mp4"),
            (
                header::CONTENT_DISPOSITION,
                "inline; filename=vashti-video.mp4",
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        bytes,
    ))
}
async fn frame(
    State(state): State<AppState>,
    jar: CookieJar,
    mut multipart: Multipart,
) -> Result<Json<Value>, ApiError> {
    let owner = user(&state, &jar).await?;
    state
        .rate_limiter
        .check(format!("video-frame:{owner}"), 30, 60)
        .await?;
    let field = multipart
        .next_field()
        .await
        .map_err(|_| service::invalid("Could not read image upload"))?
        .ok_or_else(|| service::invalid("Choose an input image"))?;
    let bytes = field
        .bytes()
        .await
        .map_err(|_| service::invalid("Could not read image upload"))?;
    Ok(Json(service::frame(&state, &owner, bytes.to_vec()).await?))
}
async fn read_frame(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Query(page): Query<Page>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let owner = user(&state, &jar).await?;
    let (mime, bytes) = service::read_frame(&state.db, &owner, &id, page.thumbnail).await?;
    Ok((
        [
            (header::CONTENT_TYPE, mime),
            (
                header::CACHE_CONTROL,
                "private, max-age=31536000, immutable".into(),
            ),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".into()),
        ],
        bytes,
    ))
}
