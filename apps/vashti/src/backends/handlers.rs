use axum::{
    Json,
    extract::{Path, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

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
    pub permission_tags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct UpdateModelTagsResponse {
    pub permission_tags: Vec<PermissionTagResponse>,
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
pub struct UpdateUserModelVisibilityRequest {
    pub backend_id: String,
    pub model_name: String,
    pub is_visible: bool,
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
            .filter(|model| preferences.get(&model.name).copied().unwrap_or(true))
            .map(model_response)
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
            .map(|model| UserModelResponse {
                is_visible: preferences.get(&model.name).copied().unwrap_or(true),
                name: model.name,
                supports_images: model.supports_images,
                supports_thinking: model.supports_thinking,
                capabilities: model.capabilities,
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
        let model_tags = permissions::model_tags_by_backend(&state.db, &backend.id).await?;
        let mut models = Vec::new();

        for model in snapshot
            .backends
            .get(&backend.id)
            .map(|cached| cached.models.clone())
            .unwrap_or_default()
        {
            let permission_tags = match model_tags.get(&model.name) {
                Some(tags) => permissions::tag_responses(&state.db, tags).await?,
                None => Vec::new(),
            };
            models.push(AdminModelResponse {
                permission_tags,
                is_enabled: availability.get(&model.name).copied().unwrap_or(true),
                name: model.name,
                supports_images: model.supports_images,
                supports_thinking: model.supports_thinking,
                capabilities: model.capabilities,
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

fn model_response(model: OllamaModel) -> ModelResponse {
    ModelResponse {
        name: model.name,
        supports_images: model.supports_images,
        supports_thinking: model.supports_thinking,
        capabilities: model.capabilities,
    }
}

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

pub async fn update_user_model_visibility(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateUserModelVisibilityRequest>,
) -> Result<Json<service::UserModelPreferenceResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let preference = service::set_user_model_visibility(
        &state.db,
        &user.id,
        &payload.backend_id,
        &payload.model_name,
        payload.is_visible,
    )
    .await?;

    Ok(Json(preference))
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
    let tags = permissions::replace_model_tags(
        &state.db,
        &payload.backend_id,
        &payload.model_name,
        payload.permission_tags,
    )
    .await?;
    let permission_tags = permissions::tag_responses(&state.db, &tags).await?;

    Ok(Json(UpdateModelTagsResponse { permission_tags }))
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
