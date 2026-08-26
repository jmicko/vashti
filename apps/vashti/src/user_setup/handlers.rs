use axum::{Json, extract::State};
use axum_extra::extract::CookieJar;
use serde::Deserialize;

use crate::{app_state::AppState, auth, error::ApiError};

use super::service::{self, SetupStatusResponse};

#[derive(Debug, Deserialize)]
pub struct SetupChoiceInput {
    pub key: String,
    pub value: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSetupChoicesRequest {
    pub choices: Vec<SetupChoiceInput>,
}

pub async fn get_setup_status(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<SetupStatusResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    Ok(Json(service::get_setup_status(&state.db, &user.id).await?))
}

pub async fn update_setup_choices(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateSetupChoicesRequest>,
) -> Result<Json<SetupStatusResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::apply_setup_choices(&state.db, &user.id, payload.choices).await?;
    Ok(Json(service::get_setup_status(&state.db, &user.id).await?))
}
