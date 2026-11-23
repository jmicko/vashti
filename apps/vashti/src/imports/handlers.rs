use super::service;
use crate::{app_state::AppState, auth, error::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/chatgpt",
            post(import).layer(DefaultBodyLimit::max(32 * 1024 * 1024)),
        )
        .route("/chatgpt/documents", get(documents))
        .route("/chatgpt/assets/{id}", post(chunk))
}
async fn documents(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Vec<service::ImportedDocument>>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(
        service::imported_documents(&state.db, &user.id).await?,
    ))
}
async fn import(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<service::ImportRequest>,
) -> Result<Json<service::ImportResult>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    if input.expected_user_id.as_deref() != Some(user.id.as_str()) {
        return Err(ApiError::forbidden(
            "account_changed",
            "Your account changed. Sign in again before importing.",
        ));
    }
    let result = service::import_conversation(&state.db, &user.id, input).await?;
    state.conversation_retrieval.wake();
    Ok(Json(result))
}
async fn chunk(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<service::Chunk>,
) -> Result<Json<service::AssetStatus>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    if input.expected_user_id.as_deref() != Some(user.id.as_str()) {
        return Err(ApiError::forbidden(
            "account_changed",
            "Your account changed. Sign in again before importing.",
        ));
    }
    Ok(Json(
        service::upload_chunk(
            &state.db,
            &state.config.tmp_dir(),
            &state.config.uploads_dir(),
            &user.id,
            &id,
            input,
        )
        .await?,
    ))
}
