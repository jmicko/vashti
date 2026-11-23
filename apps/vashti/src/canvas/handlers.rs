use super::{models::*, service};
use crate::{app_state::AppState, auth, error::ApiError};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/chats/{chat}", get(workspace))
        .route("/chats/{chat}/mutations", post(mutate))
        .route("/chats/{chat}/documents/{document}/versions", get(history))
        .route(
            "/chats/{chat}/documents/{document}/versions/{version}",
            get(version),
        )
}

async fn workspace(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat): Path<String>,
    Query(scope): Query<Scope>,
) -> Result<Json<Workspace>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(
        service::workspace(&state.db, &user.id, &chat, &scope).await?,
    ))
}
async fn mutate(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat): Path<String>,
    Json(input): Json<MutateRequest>,
) -> Result<Json<Workspace>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(
        service::mutate(
            &state.db,
            &user.id,
            &chat,
            &input.scope,
            input.mutation,
            Actor { model_name: None },
        )
        .await?,
    ))
}
#[derive(Deserialize)]
struct HistoryQuery {
    before: Option<i64>,
}
async fn history(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat, document)): Path<(String, String)>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(
        service::history(&state.db, &user.id, &chat, &document, query.before).await?,
    ))
}
async fn version(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((chat, document, version)): Path<(String, String, String)>,
) -> Result<Json<Version>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(
        service::historical_version(&state.db, &user.id, &chat, &document, &version).await?,
    ))
}
