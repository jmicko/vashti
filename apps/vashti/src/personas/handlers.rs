use axum::{
    Json,
    extract::{Path, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth,
    error::ApiError,
    personas::{
        models::{PersonaResponse, PersonaVersionResponse},
        service,
    },
};

#[derive(Debug, Serialize)]
pub struct ListPersonasResponse {
    pub personas: Vec<PersonaResponse>,
}

#[derive(Debug, Serialize)]
pub struct PersonaMutationResponse {
    pub persona: PersonaResponse,
}

#[derive(Debug, Serialize)]
pub struct PersonaVersionsResponse {
    pub versions: Vec<PersonaVersionResponse>,
}

#[derive(Debug, Serialize)]
pub struct PersonaDisownResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreatePersonaRequest {
    pub visibility: String,
    pub display_name: String,
    pub model_type: Option<String>,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: Option<f64>,
    pub avatar_crop_y: Option<f64>,
    pub avatar_crop_size: Option<f64>,
    pub background: Option<PersonaBackgroundRequest>,
    pub base_backend_id: String,
    pub base_model_name: String,
    pub system_prompt: String,
    pub tool_policy_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdatePersonaRequest {
    pub visibility: Option<String>,
    pub display_name: Option<String>,
    pub model_type: Option<String>,
    pub avatar_asset_id: Option<String>,
    pub avatar_asset_changed: Option<bool>,
    pub avatar_crop_x: Option<f64>,
    pub avatar_crop_y: Option<f64>,
    pub avatar_crop_size: Option<f64>,
    pub background: Option<PersonaBackgroundRequest>,
    pub base_backend_id: Option<String>,
    pub base_model_name: Option<String>,
    pub system_prompt: Option<String>,
    pub tool_policy_json: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PersonaBackgroundRequest {
    pub asset_id: Option<String>,
    pub asset_changed: Option<bool>,
    pub dim: f64,
    pub message_dim: f64,
    pub landscape_mode: String,
    pub landscape_x: f64,
    pub landscape_y: f64,
    pub landscape_scale: f64,
    pub portrait_mode: String,
    pub portrait_x: f64,
    pub portrait_y: f64,
    pub portrait_scale: f64,
}

#[derive(Debug, Deserialize)]
pub struct CopyPersonaRequest {
    pub persona_version_id: String,
    pub visibility: Option<String>,
}

pub async fn list_personas(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<ListPersonasResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let personas = service::list_personas(&state.db, &user.id).await?;

    Ok(Json(ListPersonasResponse { personas }))
}

pub async fn create_persona(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreatePersonaRequest>,
) -> Result<Json<PersonaMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let persona = service::create_persona(&state.db, &user.id, payload).await?;

    Ok(Json(PersonaMutationResponse { persona }))
}

pub async fn update_persona(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(persona_id): Path<String>,
    Json(payload): Json<UpdatePersonaRequest>,
) -> Result<Json<PersonaMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let persona = service::update_persona(&state.db, &user.id, &persona_id, payload).await?;

    Ok(Json(PersonaMutationResponse { persona }))
}

pub async fn copy_persona(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(persona_id): Path<String>,
    Json(payload): Json<CopyPersonaRequest>,
) -> Result<Json<PersonaMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let persona = service::copy_persona(
        &state.db,
        &state.config.persona_avatars_dir(),
        &user.id,
        &persona_id,
        payload,
    )
    .await?;

    Ok(Json(PersonaMutationResponse { persona }))
}

pub async fn disown_persona(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(persona_id): Path<String>,
) -> Result<Json<PersonaDisownResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::disown_persona(&state.db, &user.id, &persona_id).await?;

    Ok(Json(PersonaDisownResponse { ok: true }))
}

pub async fn list_versions(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(persona_id): Path<String>,
) -> Result<Json<PersonaVersionsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let versions = service::list_versions(&state.db, &user.id, &persona_id).await?;

    Ok(Json(PersonaVersionsResponse { versions }))
}
