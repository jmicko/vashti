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
    ollama,
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

pub async fn list_backends(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<BackendsResponse>, ApiError> {
    auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
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
    auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;

    let mut response_backends = Vec::new();

    for backend in service::list_enabled_backends(&state.db).await? {
        let models = match ollama::client::fetch_models(&state.http_client, &backend.base_url).await
        {
            Ok(models) => {
                service::record_backend_health(&state.db, &backend.id, "ok", None).await?;
                let availability =
                    service::model_availability_by_backend(&state.db, &backend.id).await?;
                models
                    .into_iter()
                    .filter(|model| availability.get(&model.name).copied().unwrap_or(true))
                    .map(|model| ModelResponse {
                        name: model.name,
                        supports_images: model.supports_images,
                        supports_thinking: model.supports_thinking,
                        capabilities: model.capabilities,
                    })
                    .collect()
            }
            Err(error) => {
                let message = error.to_string();
                service::record_backend_health(&state.db, &backend.id, "error", Some(&message))
                    .await?;
                tracing::warn!(backend_id = %backend.id, base_url = %backend.base_url, error = %message, "failed to fetch Ollama models");
                Vec::new()
            }
        };

        response_backends.push(BackendModelsResponse {
            backend: BackendSummaryResponse {
                id: backend.id,
                name: backend.name,
            },
            models,
        });
    }

    Ok(Json(ModelsResponse {
        backends: response_backends,
    }))
}

pub async fn list_admin_models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<AdminModelsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;

    let mut response_backends = Vec::new();

    for backend in service::list_enabled_backends(&state.db).await? {
        let models = match ollama::client::fetch_models(&state.http_client, &backend.base_url).await
        {
            Ok(models) => {
                service::record_backend_health(&state.db, &backend.id, "ok", None).await?;
                let availability =
                    service::model_availability_by_backend(&state.db, &backend.id).await?;
                models
                    .into_iter()
                    .map(|model| AdminModelResponse {
                        is_enabled: availability.get(&model.name).copied().unwrap_or(true),
                        name: model.name,
                        supports_images: model.supports_images,
                        supports_thinking: model.supports_thinking,
                        capabilities: model.capabilities,
                    })
                    .collect()
            }
            Err(error) => {
                let message = error.to_string();
                service::record_backend_health(&state.db, &backend.id, "error", Some(&message))
                    .await?;
                tracing::warn!(backend_id = %backend.id, base_url = %backend.base_url, error = %message, "failed to fetch Ollama models for admin model settings");
                Vec::new()
            }
        };

        response_backends.push(AdminBackendModelsResponse {
            backend: BackendSummaryResponse {
                id: backend.id,
                name: backend.name,
            },
            models,
        });
    }

    Ok(Json(AdminModelsResponse {
        backends: response_backends,
    }))
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
