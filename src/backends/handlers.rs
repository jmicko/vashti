use axum::{Json, extract::State};
use axum_extra::extract::CookieJar;
use serde::Serialize;

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
