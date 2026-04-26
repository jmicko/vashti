use axum::{Json, extract::State};
use axum_extra::extract::CookieJar;
use serde::Deserialize;

use crate::{app_state::AppState, auth, error::ApiError, settings::service};

#[derive(Debug, Deserialize)]
pub struct UpdateAppSettingsRequest {
    pub allow_signup: Option<bool>,
    pub signup_limit: Option<i64>,
    pub max_upload_bytes: Option<i64>,
    pub request_timeout_ms: Option<i64>,
}

pub async fn get_app_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::AppSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::get_app_settings(&state.db).await?;

    Ok(Json(settings))
}

pub async fn update_app_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateAppSettingsRequest>,
) -> Result<Json<service::AppSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::update_app_settings(&state.db, payload).await?;

    Ok(Json(settings))
}
