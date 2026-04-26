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
                models
                    .into_iter()
                    .map(|model| ModelResponse {
                        name: model.name,
                        supports_images: model.supports_images,
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
