use axum::{Json, extract::State};
use axum_extra::extract::CookieJar;

use crate::{app_state::AppState, auth, error::ApiError, updates::service};

pub async fn get_update_status(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::UpdateStatusResponse>, ApiError> {
    require_admin(&state, &jar).await?;
    state
        .updates
        .status(&state.db, &state.config)
        .await
        .map(Json)
        .map_err(map_update_error)
}

pub async fn check_for_update(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::UpdateStatusResponse>, ApiError> {
    require_admin(&state, &jar).await?;
    state
        .updates
        .check_for_update(&state.db, &state.http_client, &state.config)
        .await
        .map(Json)
        .map_err(map_update_error)
}

pub async fn install_update(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::UpdateStatusResponse>, ApiError> {
    require_admin(&state, &jar).await?;
    state
        .updates
        .check_for_update(&state.db, &state.http_client, &state.config)
        .await
        .map_err(map_update_error)?;
    state
        .updates
        .request_install(&state.db, &state.config)
        .await
        .map(Json)
        .map_err(map_update_error)
}

async fn require_admin(state: &AppState, jar: &CookieJar) -> Result<(), ApiError> {
    auth::service::require_admin(&state.db, jar, &state.config.session_cookie_name)
        .await
        .map(|_| ())
}

fn map_update_error(error: service::UpdateError) -> ApiError {
    match error {
        service::UpdateError::ManagedUpdatesUnavailable => ApiError::conflict(
            "managed_updates_unavailable",
            "Managed updates are not installed for this Vashti instance",
        ),
        service::UpdateError::NoUpdateAvailable => {
            ApiError::conflict("no_update_available", "No newer update is available")
        }
        service::UpdateError::RequestPending => ApiError::conflict(
            "update_request_pending",
            "An update request is already pending",
        ),
        other => {
            tracing::warn!(?other, "managed update operation failed");
            ApiError::internal("Managed update operation failed")
        }
    }
}
