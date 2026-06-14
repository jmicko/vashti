use axum::{
    Json,
    extract::{Path, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use crate::{
    app_state::AppState,
    auth,
    backends::service::{self, BackendResponse},
    error::ApiError,
    model_cache::ModelCacheSnapshot,
    ollama::models::OllamaModel,
    permissions::service::{self as permissions, PermissionTagResponse},
};

#[derive(Debug, Serialize)]
pub struct BackendsResponse {
    pub backends: Vec<BackendResponse>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBackendRequest {
    pub name: String,
    pub base_url: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateBackendRequest {
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub is_enabled: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct BackendMutationResponse {
    pub backend: BackendResponse,
}

#[derive(Debug, Serialize)]
pub struct DeleteBackendResponse {
    pub ok: bool,
}

#[derive(Debug, Serialize)]
pub struct DetectLocalhostResponse {
    pub detected: Vec<service::DetectedBackendResponse>,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub backends: Vec<BackendModelsResponse>,
    pub is_refreshing: bool,
    pub cache_updated_at: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct BackendModelsResponse {
    pub backend: BackendSummaryResponse,
    pub models: Vec<ModelResponse>,
}

#[derive(Debug, Serialize)]
pub struct BackendSummaryResponse {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct ModelResponse {
    pub name: String,
    pub supports_images: bool,
    pub supports_thinking: bool,
    pub capabilities: Vec<String>,
    pub is_favorite: bool,
    pub is_default: bool,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
}

#[derive(Debug, Serialize)]
pub struct AdminModelsResponse {
    pub backends: Vec<AdminBackendModelsResponse>,
    pub available_tags: Vec<PermissionTagResponse>,
    pub default_permission_tags: Vec<PermissionTagResponse>,
    pub is_refreshing: bool,
    pub cache_updated_at: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct UserModelsResponse {
    pub backends: Vec<UserBackendModelsResponse>,
    pub is_refreshing: bool,
    pub cache_updated_at: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct UserBackendModelsResponse {
    pub backend: BackendSummaryResponse,
    pub models: Vec<UserModelResponse>,
}

#[derive(Debug, Serialize)]
pub struct UserModelResponse {
    pub name: String,
    pub supports_images: bool,
    pub supports_thinking: bool,
    pub capabilities: Vec<String>,
    pub is_visible: bool,
    pub is_favorite: bool,
    pub is_default: bool,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
    pub personal_avatar_asset_id: Option<String>,
    pub personal_avatar_crop_x: f64,
    pub personal_avatar_crop_y: f64,
    pub personal_avatar_crop_size: f64,
    pub default_avatar_asset_id: Option<String>,
    pub default_avatar_crop_x: f64,
    pub default_avatar_crop_y: f64,
    pub default_avatar_crop_size: f64,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
    pub personal_background_asset_id: Option<String>,
    pub personal_background_dim: f64,
    pub personal_background_message_dim: f64,
    pub personal_background_landscape_mode: String,
    pub personal_background_landscape_x: f64,
    pub personal_background_landscape_y: f64,
    pub personal_background_landscape_scale: f64,
    pub personal_background_portrait_mode: String,
    pub personal_background_portrait_x: f64,
    pub personal_background_portrait_y: f64,
    pub personal_background_portrait_scale: f64,
    pub default_background_asset_id: Option<String>,
    pub default_background_dim: f64,
    pub default_background_message_dim: f64,
    pub default_background_landscape_mode: String,
    pub default_background_landscape_x: f64,
    pub default_background_landscape_y: f64,
    pub default_background_landscape_scale: f64,
    pub default_background_portrait_mode: String,
    pub default_background_portrait_x: f64,
    pub default_background_portrait_y: f64,
    pub default_background_portrait_scale: f64,
}

#[derive(Debug, Serialize)]
pub struct AdminBackendModelsResponse {
    pub backend: BackendSummaryResponse,
    pub models: Vec<AdminModelResponse>,
}

#[derive(Debug, Serialize)]
pub struct AdminModelResponse {
    pub name: String,
    pub supports_images: bool,
    pub supports_thinking: bool,
    pub capabilities: Vec<String>,
    pub is_enabled: bool,
    pub permission_tags: Vec<PermissionTagResponse>,
    pub default_permission_tags: Vec<PermissionTagResponse>,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateModelAvailabilityRequest {
    pub backend_id: String,
    pub model_name: String,
    pub is_enabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct BulkUpdateModelAvailabilityRequest {
    pub backend_id: String,
    pub model_names: Vec<String>,
    pub is_enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct BulkModelAvailabilityResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateModelTagsRequest {
    pub backend_id: String,
    pub model_name: String,
    pub permission_tags: Option<Vec<String>>,
    pub default_permission_tags: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct UpdateModelTagsResponse {
    pub permission_tags: Vec<PermissionTagResponse>,
    pub default_permission_tags: Vec<PermissionTagResponse>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateDefaultModelTagsRequest {
    pub permission_tags: Vec<String>,
    pub apply_to_existing: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct UpdateDefaultModelTagsResponse {
    pub default_permission_tags: Vec<PermissionTagResponse>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserModelPreferenceRequest {
    pub backend_id: String,
    pub model_name: String,
    pub is_visible: Option<bool>,
    pub is_favorite: Option<bool>,
    pub is_default: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateModelAvatarRequest {
    pub backend_id: String,
    pub model_name: String,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
}

#[derive(Debug, Serialize)]
pub struct UserModelAvatarResponse {
    pub backend_id: String,
    pub model_name: String,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
    pub personal_avatar_asset_id: Option<String>,
    pub personal_avatar_crop_x: f64,
    pub personal_avatar_crop_y: f64,
    pub personal_avatar_crop_size: f64,
    pub default_avatar_asset_id: Option<String>,
    pub default_avatar_crop_x: f64,
    pub default_avatar_crop_y: f64,
    pub default_avatar_crop_size: f64,
}

#[derive(Debug, Serialize)]
pub struct AdminModelAvatarResponse {
    pub backend_id: String,
    pub model_name: String,
    pub avatar_asset_id: Option<String>,
    pub avatar_crop_x: f64,
    pub avatar_crop_y: f64,
    pub avatar_crop_size: f64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateModelBackgroundRequest {
    pub backend_id: String,
    pub model_name: String,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
}

#[derive(Debug, Serialize)]
pub struct UserModelBackgroundResponse {
    pub backend_id: String,
    pub model_name: String,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
    pub personal_background_asset_id: Option<String>,
    pub personal_background_dim: f64,
    pub personal_background_message_dim: f64,
    pub personal_background_landscape_mode: String,
    pub personal_background_landscape_x: f64,
    pub personal_background_landscape_y: f64,
    pub personal_background_landscape_scale: f64,
    pub personal_background_portrait_mode: String,
    pub personal_background_portrait_x: f64,
    pub personal_background_portrait_y: f64,
    pub personal_background_portrait_scale: f64,
    pub default_background_asset_id: Option<String>,
    pub default_background_dim: f64,
    pub default_background_message_dim: f64,
    pub default_background_landscape_mode: String,
    pub default_background_landscape_x: f64,
    pub default_background_landscape_y: f64,
    pub default_background_landscape_scale: f64,
    pub default_background_portrait_mode: String,
    pub default_background_portrait_x: f64,
    pub default_background_portrait_y: f64,
    pub default_background_portrait_scale: f64,
}

#[derive(Debug, Serialize)]
pub struct AdminModelBackgroundResponse {
    pub backend_id: String,
    pub model_name: String,
    pub background_asset_id: Option<String>,
    pub background_dim: f64,
    pub background_message_dim: f64,
    pub background_landscape_mode: String,
    pub background_landscape_x: f64,
    pub background_landscape_y: f64,
    pub background_landscape_scale: f64,
    pub background_portrait_mode: String,
    pub background_portrait_x: f64,
    pub background_portrait_y: f64,
    pub background_portrait_scale: f64,
}

pub async fn list_backends(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<BackendsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let backends = service::list_backends(&state.db).await?;

    Ok(Json(BackendsResponse { backends }))
}

pub async fn create_backend(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateBackendRequest>,
) -> Result<Json<BackendMutationResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let backend = service::create_backend(&state.db, payload.name, payload.base_url).await?;

    Ok(Json(BackendMutationResponse { backend }))
}

pub async fn update_backend(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(backend_id): Path<String>,
    Json(payload): Json<UpdateBackendRequest>,
) -> Result<Json<BackendMutationResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let backend = service::update_backend(
        &state.db,
        &backend_id,
        service::UpdateBackendParams {
            name: payload.name,
            base_url: payload.base_url,
            is_enabled: payload.is_enabled,
        },
    )
    .await?;

    Ok(Json(BackendMutationResponse { backend }))
}

pub async fn delete_backend(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(backend_id): Path<String>,
) -> Result<Json<DeleteBackendResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_backend(&state.db, &backend_id).await?;

    Ok(Json(DeleteBackendResponse { ok: true }))
}

pub async fn detect_localhost(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<DetectLocalhostResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let detected = service::detect_localhost_backends(&state.db, &state.http_client).await?;

    Ok(Json(DetectLocalhostResponse { detected }))
}

pub async fn scan_local_network(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<DetectLocalhostResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let detected = service::scan_local_network_backends(&state.db, &state.http_client).await?;

    Ok(Json(DetectLocalhostResponse { detected }))
}

pub async fn list_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<ModelsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let snapshot = state.model_cache.snapshot().await;

    Ok(Json(models_response(&state, &user.id, snapshot).await?))
}

pub async fn list_admin_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<AdminModelsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let snapshot = state.model_cache.snapshot().await;

    Ok(Json(admin_models_response(&state, snapshot).await?))
}

pub async fn list_user_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<UserModelsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let snapshot = state.model_cache.snapshot().await;

    Ok(Json(
        user_models_response(&state, &user.id, snapshot).await?,
    ))
}

pub async fn refresh_user_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<UserModelsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let snapshot = state
        .model_cache
        .refresh_all(&state.db, &state.http_client)
        .await?;

    Ok(Json(
        user_models_response(&state, &user.id, snapshot).await?,
    ))
}

pub async fn refresh_admin_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<AdminModelsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let snapshot = state
        .model_cache
        .refresh_all(&state.db, &state.http_client)
        .await?;

    Ok(Json(admin_models_response(&state, snapshot).await?))
}

async fn models_response(
    state: &AppState,
    user_id: &str,
    snapshot: ModelCacheSnapshot,
) -> Result<ModelsResponse, ApiError> {
    let user_tags = permissions::effective_user_tag_ids(&state.db, user_id).await?;
    let mut response_backends = Vec::new();

    for backend in service::list_enabled_backends(&state.db).await? {
        let availability = service::model_availability_by_backend(&state.db, &backend.id).await?;
        let default_avatars =
            service::model_avatar_defaults_by_backend(&state.db, &backend.id).await?;
        let default_backgrounds =
            service::model_background_defaults_by_backend(&state.db, &backend.id).await?;
        let preferences =
            service::user_model_preferences_by_backend(&state.db, user_id, &backend.id).await?;
        let model_tags = permissions::model_tags_by_backend(&state.db, &backend.id).await?;
        let models = snapshot
            .backends
            .get(&backend.id)
            .map(|cached| cached.models.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|model| availability.get(&model.name).copied().unwrap_or(true))
            .filter(|model| {
                model_tags
                    .get(&model.name)
                    .is_some_and(|tags| permissions::has_matching_tag(&user_tags, tags))
            })
            .filter(|model| {
                preferences
                    .get(&model.name)
                    .map(|preference| preference.is_visible)
                    .unwrap_or(true)
            })
            .map(|model| {
                let preference = preferences.get(&model.name);
                let default_avatar = default_avatars.get(&model.name);
                let default_background = default_backgrounds.get(&model.name);
                model_response(model, preference, default_avatar, default_background)
            })
            .collect();

        response_backends.push(BackendModelsResponse {
            backend: BackendSummaryResponse {
                id: backend.id,
                name: backend.name,
            },
            models,
        });
    }

    Ok(ModelsResponse {
        backends: response_backends,
        is_refreshing: snapshot.is_refreshing,
        cache_updated_at: snapshot.updated_at,
    })
}

async fn user_models_response(
    state: &AppState,
    user_id: &str,
    snapshot: ModelCacheSnapshot,
) -> Result<UserModelsResponse, ApiError> {
    let user_tags = permissions::effective_user_tag_ids(&state.db, user_id).await?;
    let mut response_backends = Vec::new();

    for backend in service::list_enabled_backends(&state.db).await? {
        let availability = service::model_availability_by_backend(&state.db, &backend.id).await?;
        let default_avatars =
            service::model_avatar_defaults_by_backend(&state.db, &backend.id).await?;
        let default_backgrounds =
            service::model_background_defaults_by_backend(&state.db, &backend.id).await?;
        let preferences =
            service::user_model_preferences_by_backend(&state.db, user_id, &backend.id).await?;
        let model_tags = permissions::model_tags_by_backend(&state.db, &backend.id).await?;
        let models = snapshot
            .backends
            .get(&backend.id)
            .map(|cached| cached.models.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|model| availability.get(&model.name).copied().unwrap_or(true))
            .filter(|model| {
                model_tags
                    .get(&model.name)
                    .is_some_and(|tags| permissions::has_matching_tag(&user_tags, tags))
            })
            .map(|model| {
                let preference = preferences.get(&model.name);
                let personal_avatar = preference.map(|preference| &preference.avatar);
                let default_avatar = default_avatars.get(&model.name);
                let effective_avatar = effective_avatar(personal_avatar, default_avatar);
                let personal_background = preference.map(|preference| &preference.background);
                let default_background = default_backgrounds.get(&model.name);
                let effective_background =
                    effective_background(personal_background, default_background);
                UserModelResponse {
                    is_visible: preference
                        .map(|preference| preference.is_visible)
                        .unwrap_or(true),
                    is_favorite: preference
                        .map(|preference| preference.is_favorite)
                        .unwrap_or(false),
                    is_default: preference
                        .map(|preference| preference.is_default)
                        .unwrap_or(false),
                    avatar_asset_id: effective_avatar.avatar_asset_id.clone(),
                    avatar_crop_x: effective_avatar.avatar_crop_x,
                    avatar_crop_y: effective_avatar.avatar_crop_y,
                    avatar_crop_size: effective_avatar.avatar_crop_size,
                    personal_avatar_asset_id: personal_avatar
                        .and_then(|avatar| avatar.avatar_asset_id.clone()),
                    personal_avatar_crop_x: personal_avatar
                        .map(|avatar| avatar.avatar_crop_x)
                        .unwrap_or(50.0),
                    personal_avatar_crop_y: personal_avatar
                        .map(|avatar| avatar.avatar_crop_y)
                        .unwrap_or(50.0),
                    personal_avatar_crop_size: personal_avatar
                        .map(|avatar| avatar.avatar_crop_size)
                        .unwrap_or(100.0),
                    default_avatar_asset_id: default_avatar
                        .and_then(|avatar| avatar.avatar_asset_id.clone()),
                    default_avatar_crop_x: default_avatar
                        .map(|avatar| avatar.avatar_crop_x)
                        .unwrap_or(50.0),
                    default_avatar_crop_y: default_avatar
                        .map(|avatar| avatar.avatar_crop_y)
                        .unwrap_or(50.0),
                    default_avatar_crop_size: default_avatar
                        .map(|avatar| avatar.avatar_crop_size)
                        .unwrap_or(100.0),
                    background_asset_id: effective_background.background_asset_id.clone(),
                    background_dim: effective_background.background_dim,
                    background_message_dim: effective_background.background_message_dim,
                    background_landscape_mode: effective_background
                        .background_landscape_mode
                        .clone(),
                    background_landscape_x: effective_background.background_landscape_x,
                    background_landscape_y: effective_background.background_landscape_y,
                    background_landscape_scale: effective_background.background_landscape_scale,
                    background_portrait_mode: effective_background.background_portrait_mode.clone(),
                    background_portrait_x: effective_background.background_portrait_x,
                    background_portrait_y: effective_background.background_portrait_y,
                    background_portrait_scale: effective_background.background_portrait_scale,
                    personal_background_asset_id: personal_background
                        .and_then(|background| background.background_asset_id.clone()),
                    personal_background_dim: personal_background
                        .map(|background| background.background_dim)
                        .unwrap_or(0.72),
                    personal_background_message_dim: personal_background
                        .map(|background| background.background_message_dim)
                        .unwrap_or(0.82),
                    personal_background_landscape_mode: personal_background
                        .map(|background| background.background_landscape_mode.clone())
                        .unwrap_or_else(|| "fill".to_string()),
                    personal_background_landscape_x: personal_background
                        .map(|background| background.background_landscape_x)
                        .unwrap_or(50.0),
                    personal_background_landscape_y: personal_background
                        .map(|background| background.background_landscape_y)
                        .unwrap_or(50.0),
                    personal_background_landscape_scale: personal_background
                        .map(|background| background.background_landscape_scale)
                        .unwrap_or(35.0),
                    personal_background_portrait_mode: personal_background
                        .map(|background| background.background_portrait_mode.clone())
                        .unwrap_or_else(|| "fill".to_string()),
                    personal_background_portrait_x: personal_background
                        .map(|background| background.background_portrait_x)
                        .unwrap_or(50.0),
                    personal_background_portrait_y: personal_background
                        .map(|background| background.background_portrait_y)
                        .unwrap_or(50.0),
                    personal_background_portrait_scale: personal_background
                        .map(|background| background.background_portrait_scale)
                        .unwrap_or(35.0),
                    default_background_asset_id: default_background
                        .and_then(|background| background.background_asset_id.clone()),
                    default_background_dim: default_background
                        .map(|background| background.background_dim)
                        .unwrap_or(0.72),
                    default_background_message_dim: default_background
                        .map(|background| background.background_message_dim)
                        .unwrap_or(0.82),
                    default_background_landscape_mode: default_background
                        .map(|background| background.background_landscape_mode.clone())
                        .unwrap_or_else(|| "fill".to_string()),
                    default_background_landscape_x: default_background
                        .map(|background| background.background_landscape_x)
                        .unwrap_or(50.0),
                    default_background_landscape_y: default_background
                        .map(|background| background.background_landscape_y)
                        .unwrap_or(50.0),
                    default_background_landscape_scale: default_background
                        .map(|background| background.background_landscape_scale)
                        .unwrap_or(35.0),
                    default_background_portrait_mode: default_background
                        .map(|background| background.background_portrait_mode.clone())
                        .unwrap_or_else(|| "fill".to_string()),
                    default_background_portrait_x: default_background
                        .map(|background| background.background_portrait_x)
                        .unwrap_or(50.0),
                    default_background_portrait_y: default_background
                        .map(|background| background.background_portrait_y)
                        .unwrap_or(50.0),
                    default_background_portrait_scale: default_background
                        .map(|background| background.background_portrait_scale)
                        .unwrap_or(35.0),
                    name: model.name,
                    supports_images: model.supports_images,
                    supports_thinking: model.supports_thinking,
                    capabilities: model.capabilities,
                }
            })
            .collect();

        response_backends.push(UserBackendModelsResponse {
            backend: BackendSummaryResponse {
                id: backend.id,
                name: backend.name,
            },
            models,
        });
    }

    Ok(UserModelsResponse {
        backends: response_backends,
        is_refreshing: snapshot.is_refreshing,
        cache_updated_at: snapshot.updated_at,
    })
}

async fn admin_models_response(
    state: &AppState,
    snapshot: ModelCacheSnapshot,
) -> Result<AdminModelsResponse, ApiError> {
    let available_tags = permissions::known_tags(&state.db).await?;
    let default_tag_ids = permissions::default_model_tag_ids(&state.db).await?;
    let default_permission_tags = permissions::tag_responses(&state.db, &default_tag_ids).await?;
    let mut response_backends = Vec::new();

    for backend in service::list_enabled_backends(&state.db).await? {
        let availability = service::model_availability_by_backend(&state.db, &backend.id).await?;
        let default_avatars =
            service::model_avatar_defaults_by_backend(&state.db, &backend.id).await?;
        let default_backgrounds =
            service::model_background_defaults_by_backend(&state.db, &backend.id).await?;
        let manual_model_tags =
            permissions::manual_model_tags_by_backend(&state.db, &backend.id).await?;
        let default_model_tags =
            permissions::default_model_tags_by_backend(&state.db, &backend.id).await?;
        let mut models = Vec::new();

        for model in snapshot
            .backends
            .get(&backend.id)
            .map(|cached| cached.models.clone())
            .unwrap_or_default()
        {
            let permission_tags = match manual_model_tags.get(&model.name) {
                Some(tags) => permissions::tag_responses(&state.db, tags).await?,
                None => Vec::new(),
            };
            let default_permission_tags = match default_model_tags.get(&model.name) {
                Some(tags) => permissions::tag_responses(&state.db, tags).await?,
                None => Vec::new(),
            };
            let avatar = default_avatars
                .get(&model.name)
                .cloned()
                .unwrap_or_else(empty_avatar);
            let background = default_backgrounds
                .get(&model.name)
                .cloned()
                .unwrap_or_else(empty_background);
            models.push(AdminModelResponse {
                permission_tags,
                default_permission_tags,
                is_enabled: availability.get(&model.name).copied().unwrap_or(true),
                name: model.name,
                supports_images: model.supports_images,
                supports_thinking: model.supports_thinking,
                capabilities: model.capabilities,
                avatar_asset_id: avatar.avatar_asset_id,
                avatar_crop_x: avatar.avatar_crop_x,
                avatar_crop_y: avatar.avatar_crop_y,
                avatar_crop_size: avatar.avatar_crop_size,
                background_asset_id: background.background_asset_id,
                background_dim: background.background_dim,
                background_message_dim: background.background_message_dim,
                background_landscape_mode: background.background_landscape_mode,
                background_landscape_x: background.background_landscape_x,
                background_landscape_y: background.background_landscape_y,
                background_landscape_scale: background.background_landscape_scale,
                background_portrait_mode: background.background_portrait_mode,
                background_portrait_x: background.background_portrait_x,
                background_portrait_y: background.background_portrait_y,
                background_portrait_scale: background.background_portrait_scale,
            });
        }

        response_backends.push(AdminBackendModelsResponse {
            backend: BackendSummaryResponse {
                id: backend.id,
                name: backend.name,
            },
            models,
        });
    }

    Ok(AdminModelsResponse {
        backends: response_backends,
        available_tags,
        default_permission_tags,
        is_refreshing: snapshot.is_refreshing,
        cache_updated_at: snapshot.updated_at,
    })
}

fn model_response(
    model: OllamaModel,
    preference: Option<&service::UserModelPreference>,
    default_avatar: Option<&service::ModelAvatarReference>,
    default_background: Option<&service::ModelBackgroundReference>,
) -> ModelResponse {
    let effective_avatar = effective_avatar(
        preference.map(|preference| &preference.avatar),
        default_avatar,
    );
    let effective_background = effective_background(
        preference.map(|preference| &preference.background),
        default_background,
    );
    ModelResponse {
        name: model.name,
        supports_images: model.supports_images,
        supports_thinking: model.supports_thinking,
        capabilities: model.capabilities,
        is_favorite: preference.is_some_and(|preference| preference.is_favorite),
        is_default: preference.is_some_and(|preference| preference.is_default),
        avatar_asset_id: effective_avatar.avatar_asset_id.clone(),
        avatar_crop_x: effective_avatar.avatar_crop_x,
        avatar_crop_y: effective_avatar.avatar_crop_y,
        avatar_crop_size: effective_avatar.avatar_crop_size,
        background_asset_id: effective_background.background_asset_id.clone(),
        background_dim: effective_background.background_dim,
        background_message_dim: effective_background.background_message_dim,
        background_landscape_mode: effective_background.background_landscape_mode.clone(),
        background_landscape_x: effective_background.background_landscape_x,
        background_landscape_y: effective_background.background_landscape_y,
        background_landscape_scale: effective_background.background_landscape_scale,
        background_portrait_mode: effective_background.background_portrait_mode.clone(),
        background_portrait_x: effective_background.background_portrait_x,
        background_portrait_y: effective_background.background_portrait_y,
        background_portrait_scale: effective_background.background_portrait_scale,
    }
}

fn empty_avatar() -> service::ModelAvatarReference {
    service::ModelAvatarReference {
        avatar_asset_id: None,
        avatar_crop_x: 50.0,
        avatar_crop_y: 50.0,
        avatar_crop_size: 100.0,
    }
}

fn effective_avatar<'a>(
    personal: Option<&'a service::ModelAvatarReference>,
    default: Option<&'a service::ModelAvatarReference>,
) -> &'a service::ModelAvatarReference {
    personal
        .filter(|avatar| avatar.avatar_asset_id.is_some())
        .or_else(|| default.filter(|avatar| avatar.avatar_asset_id.is_some()))
        .unwrap_or(&EMPTY_AVATAR)
}

fn empty_background() -> service::ModelBackgroundReference {
    EMPTY_BACKGROUND.clone()
}

fn effective_background<'a>(
    personal: Option<&'a service::ModelBackgroundReference>,
    default: Option<&'a service::ModelBackgroundReference>,
) -> &'a service::ModelBackgroundReference {
    personal
        .filter(|background| background.background_asset_id.is_some())
        .or_else(|| default.filter(|background| background.background_asset_id.is_some()))
        .unwrap_or(&EMPTY_BACKGROUND)
}

static EMPTY_AVATAR: service::ModelAvatarReference = service::ModelAvatarReference {
    avatar_asset_id: None,
    avatar_crop_x: 50.0,
    avatar_crop_y: 50.0,
    avatar_crop_size: 100.0,
};

static EMPTY_BACKGROUND: LazyLock<service::ModelBackgroundReference> =
    LazyLock::new(|| service::ModelBackgroundReference {
        background_asset_id: None,
        background_dim: 0.72,
        background_message_dim: 0.82,
        background_landscape_mode: "fill".to_string(),
        background_landscape_x: 50.0,
        background_landscape_y: 50.0,
        background_landscape_scale: 35.0,
        background_portrait_mode: "fill".to_string(),
        background_portrait_x: 50.0,
        background_portrait_y: 50.0,
        background_portrait_scale: 35.0,
    });

pub async fn update_model_availability(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelAvailabilityRequest>,
) -> Result<Json<service::ModelAvailabilityResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let availability = service::set_model_availability(
        &state.db,
        &payload.backend_id,
        &payload.model_name,
        payload.is_enabled,
    )
    .await?;

    Ok(Json(availability))
}

pub async fn update_user_model_preference(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateUserModelPreferenceRequest>,
) -> Result<Json<service::UserModelPreferenceResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let preference = service::update_user_model_preference(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        service::UpdateUserModelPreferenceParams {
            is_visible: payload.is_visible,
            is_favorite: payload.is_favorite,
            is_default: payload.is_default,
        },
    )
    .await?;

    Ok(Json(preference))
}

pub async fn update_user_model_avatar(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelAvatarRequest>,
) -> Result<Json<UserModelAvatarResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let personal = service::set_user_model_avatar(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        service::UpdateModelAvatarParams {
            avatar_asset_id: payload.avatar_asset_id,
            avatar_crop_x: payload.avatar_crop_x,
            avatar_crop_y: payload.avatar_crop_y,
            avatar_crop_size: payload.avatar_crop_size,
        },
    )
    .await?;
    let default = service::model_avatar_defaults_by_backend(&state.db, &payload.backend_id)
        .await?
        .remove(&payload.model_name)
        .unwrap_or_else(empty_avatar);
    let effective = effective_avatar(Some(&personal), Some(&default));

    Ok(Json(UserModelAvatarResponse {
        backend_id: payload.backend_id,
        model_name: payload.model_name,
        avatar_asset_id: effective.avatar_asset_id.clone(),
        avatar_crop_x: effective.avatar_crop_x,
        avatar_crop_y: effective.avatar_crop_y,
        avatar_crop_size: effective.avatar_crop_size,
        personal_avatar_asset_id: personal.avatar_asset_id,
        personal_avatar_crop_x: personal.avatar_crop_x,
        personal_avatar_crop_y: personal.avatar_crop_y,
        personal_avatar_crop_size: personal.avatar_crop_size,
        default_avatar_asset_id: default.avatar_asset_id,
        default_avatar_crop_x: default.avatar_crop_x,
        default_avatar_crop_y: default.avatar_crop_y,
        default_avatar_crop_size: default.avatar_crop_size,
    }))
}

pub async fn update_admin_model_avatar(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelAvatarRequest>,
) -> Result<Json<AdminModelAvatarResponse>, ApiError> {
    let user =
        auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let avatar = service::set_default_model_avatar(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        service::UpdateModelAvatarParams {
            avatar_asset_id: payload.avatar_asset_id,
            avatar_crop_x: payload.avatar_crop_x,
            avatar_crop_y: payload.avatar_crop_y,
            avatar_crop_size: payload.avatar_crop_size,
        },
    )
    .await?;

    Ok(Json(AdminModelAvatarResponse {
        backend_id: payload.backend_id,
        model_name: payload.model_name,
        avatar_asset_id: avatar.avatar_asset_id,
        avatar_crop_x: avatar.avatar_crop_x,
        avatar_crop_y: avatar.avatar_crop_y,
        avatar_crop_size: avatar.avatar_crop_size,
    }))
}

pub async fn update_user_model_background(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelBackgroundRequest>,
) -> Result<Json<UserModelBackgroundResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let personal = service::set_user_model_background(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        service::UpdateModelBackgroundParams {
            background_asset_id: payload.background_asset_id,
            background_dim: payload.background_dim,
            background_message_dim: payload.background_message_dim,
            background_landscape_mode: payload.background_landscape_mode,
            background_landscape_x: payload.background_landscape_x,
            background_landscape_y: payload.background_landscape_y,
            background_landscape_scale: payload.background_landscape_scale,
            background_portrait_mode: payload.background_portrait_mode,
            background_portrait_x: payload.background_portrait_x,
            background_portrait_y: payload.background_portrait_y,
            background_portrait_scale: payload.background_portrait_scale,
        },
    )
    .await?;
    let default = service::model_background_defaults_by_backend(&state.db, &payload.backend_id)
        .await?
        .remove(&payload.model_name)
        .unwrap_or_else(empty_background);
    let effective = effective_background(Some(&personal), Some(&default));

    Ok(Json(UserModelBackgroundResponse {
        backend_id: payload.backend_id,
        model_name: payload.model_name,
        background_asset_id: effective.background_asset_id.clone(),
        background_dim: effective.background_dim,
        background_message_dim: effective.background_message_dim,
        background_landscape_mode: effective.background_landscape_mode.clone(),
        background_landscape_x: effective.background_landscape_x,
        background_landscape_y: effective.background_landscape_y,
        background_landscape_scale: effective.background_landscape_scale,
        background_portrait_mode: effective.background_portrait_mode.clone(),
        background_portrait_x: effective.background_portrait_x,
        background_portrait_y: effective.background_portrait_y,
        background_portrait_scale: effective.background_portrait_scale,
        personal_background_asset_id: personal.background_asset_id,
        personal_background_dim: personal.background_dim,
        personal_background_message_dim: personal.background_message_dim,
        personal_background_landscape_mode: personal.background_landscape_mode,
        personal_background_landscape_x: personal.background_landscape_x,
        personal_background_landscape_y: personal.background_landscape_y,
        personal_background_landscape_scale: personal.background_landscape_scale,
        personal_background_portrait_mode: personal.background_portrait_mode,
        personal_background_portrait_x: personal.background_portrait_x,
        personal_background_portrait_y: personal.background_portrait_y,
        personal_background_portrait_scale: personal.background_portrait_scale,
        default_background_asset_id: default.background_asset_id,
        default_background_dim: default.background_dim,
        default_background_message_dim: default.background_message_dim,
        default_background_landscape_mode: default.background_landscape_mode,
        default_background_landscape_x: default.background_landscape_x,
        default_background_landscape_y: default.background_landscape_y,
        default_background_landscape_scale: default.background_landscape_scale,
        default_background_portrait_mode: default.background_portrait_mode,
        default_background_portrait_x: default.background_portrait_x,
        default_background_portrait_y: default.background_portrait_y,
        default_background_portrait_scale: default.background_portrait_scale,
    }))
}

pub async fn update_admin_model_background(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelBackgroundRequest>,
) -> Result<Json<AdminModelBackgroundResponse>, ApiError> {
    let user =
        auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let background = service::set_default_model_background(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        service::UpdateModelBackgroundParams {
            background_asset_id: payload.background_asset_id,
            background_dim: payload.background_dim,
            background_message_dim: payload.background_message_dim,
            background_landscape_mode: payload.background_landscape_mode,
            background_landscape_x: payload.background_landscape_x,
            background_landscape_y: payload.background_landscape_y,
            background_landscape_scale: payload.background_landscape_scale,
            background_portrait_mode: payload.background_portrait_mode,
            background_portrait_x: payload.background_portrait_x,
            background_portrait_y: payload.background_portrait_y,
            background_portrait_scale: payload.background_portrait_scale,
        },
    )
    .await?;

    Ok(Json(AdminModelBackgroundResponse {
        backend_id: payload.backend_id,
        model_name: payload.model_name,
        background_asset_id: background.background_asset_id,
        background_dim: background.background_dim,
        background_message_dim: background.background_message_dim,
        background_landscape_mode: background.background_landscape_mode,
        background_landscape_x: background.background_landscape_x,
        background_landscape_y: background.background_landscape_y,
        background_landscape_scale: background.background_landscape_scale,
        background_portrait_mode: background.background_portrait_mode,
        background_portrait_x: background.background_portrait_x,
        background_portrait_y: background.background_portrait_y,
        background_portrait_scale: background.background_portrait_scale,
    }))
}

pub async fn update_backend_model_availability(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<BulkUpdateModelAvailabilityRequest>,
) -> Result<Json<BulkModelAvailabilityResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::set_model_availability_many(
        &state.db,
        &payload.backend_id,
        &payload.model_names,
        payload.is_enabled,
    )
    .await?;

    Ok(Json(BulkModelAvailabilityResponse { ok: true }))
}

pub async fn update_model_tags(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateModelTagsRequest>,
) -> Result<Json<UpdateModelTagsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::ensure_model_record(&state.db, &payload.backend_id, &payload.model_name).await?;
    if payload.permission_tags.is_none() && payload.default_permission_tags.is_none() {
        return Err(ApiError::bad_request(
            "empty_model_tag_update",
            "No model tags were provided",
        ));
    }

    if let Some(tags) = payload.permission_tags {
        permissions::replace_model_tags(&state.db, &payload.backend_id, &payload.model_name, tags)
            .await?;
    }
    if let Some(tags) = payload.default_permission_tags {
        permissions::replace_model_default_tags(
            &state.db,
            &payload.backend_id,
            &payload.model_name,
            tags,
        )
        .await?;
    }

    let manual_tags =
        permissions::manual_model_tags_by_backend(&state.db, &payload.backend_id).await?;
    let default_tags =
        permissions::default_model_tags_by_backend(&state.db, &payload.backend_id).await?;
    let permission_tags = permissions::tag_responses(
        &state.db,
        manual_tags
            .get(&payload.model_name)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
    )
    .await?;
    let default_permission_tags = permissions::tag_responses(
        &state.db,
        default_tags
            .get(&payload.model_name)
            .map(Vec::as_slice)
            .unwrap_or(&[]),
    )
    .await?;

    Ok(Json(UpdateModelTagsResponse {
        permission_tags,
        default_permission_tags,
    }))
}

pub async fn update_default_model_tags(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateDefaultModelTagsRequest>,
) -> Result<Json<UpdateDefaultModelTagsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let tags = permissions::update_default_model_tags(
        &state.db,
        payload.permission_tags,
        payload.apply_to_existing.unwrap_or(false),
    )
    .await?;
    let default_permission_tags = permissions::tag_responses(&state.db, &tags).await?;

    Ok(Json(UpdateDefaultModelTagsResponse {
        default_permission_tags,
    }))
}
