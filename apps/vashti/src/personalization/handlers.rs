use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use serde_json::{Value, json};

use super::{models::*, service, worker};
use crate::{app_state::AppState, auth, error::ApiError};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_profile))
        .route("/settings", post(save_settings))
        .route("/models", get(models))
        .route("/process", post(process))
        .route("/device-extract", post(device_extract))
        .route("/facts", post(create_fact))
        .route("/facts/{id}", post(edit_fact))
        .route("/facts/{id}/history", get(history))
        .route("/facts/{id}/forget", post(forget_fact))
        .route("/forget", post(forget_all))
}

async fn owner(state: &AppState, jar: &CookieJar) -> Result<String, ApiError> {
    Ok(
        auth::service::require_user(&state.db, jar, &state.config.session_cookie_name)
            .await?
            .id,
    )
}

async fn get_profile(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::profile(&state.db, &owner(&state, &jar).await?).await?,
    ))
}

async fn models(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Vec<Model>>, ApiError> {
    Ok(Json(
        service::catalog(&state, &owner(&state, &jar).await?).await?,
    ))
}

async fn save_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<SaveSettings>,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::save_settings(&state, &owner(&state, &jar).await?, input).await?,
    ))
}

async fn process(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<ExpectedRevision>,
) -> Result<Json<Profile>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let profile = service::profile(&state.db, &owner).await?;
    if !profile.settings.enabled
        || profile.settings.processor_backend_id.is_none()
        || profile.settings.processor_model.is_none()
    {
        return Err(service::invalid(
            "Enable personalization and choose a processing model first",
        ));
    }
    service::ensure(&state.db, &owner).await?;
    let mut tx = state.db.begin().await?;
    service::bump(&mut tx, &owner, input.expected_revision).await?;
    sqlx::query("UPDATE user_personalization SET requested=1,last_attempt_at=NULL,last_error=NULL,failures=0 WHERE user_id=?").bind(&owner).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(Json(service::profile(&state.db, &owner).await?))
}

async fn create_fact(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<EditFact>,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::edit_fact(&state.db, &owner(&state, &jar).await?, None, input).await?,
    ))
}

async fn edit_fact(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<EditFact>,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::edit_fact(&state.db, &owner(&state, &jar).await?, Some(&id), input).await?,
    ))
}

async fn forget_fact(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(input): Json<ExpectedRevision>,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::forget(
            &state.db,
            &owner(&state, &jar).await?,
            Some(&id),
            input.expected_revision,
        )
        .await?,
    ))
}

async fn forget_all(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<ExpectedRevision>,
) -> Result<Json<Profile>, ApiError> {
    Ok(Json(
        service::forget(
            &state.db,
            &owner(&state, &jar).await?,
            None,
            input.expected_revision,
        )
        .await?,
    ))
}

async fn history(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        service::fact_history(&state.db, &owner(&state, &jar).await?, &id).await?,
    ))
}

async fn device_extract(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(input): Json<DeviceExtract>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    if input.expected_user_id != owner {
        return Err(ApiError::conflict(
            "private_identity_changed",
            "The signed-in account changed. Private processing was not started.",
        ));
    }
    if worker::foreground_busy(&state).await {
        return Err(ApiError::conflict(
            "processor_busy",
            "Background processing waits for active generations to finish",
        ));
    }
    state
        .rate_limiter
        .check(format!("personalization:{owner}"), 6, 60)
        .await?;
    // Device input is transient: never enter the server profile or processing ledger.
    let _lane = state
        .personalization_lane
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::conflict(
                "processor_busy",
                "Another profile batch is running. Try again shortly.",
            )
        })?;
    let inference = worker::extract(
        &state,
        &owner,
        &input.backend_id,
        &input.model_name,
        input.allow_cloud_processor,
        &input.sources,
        &input.topics,
    );
    tokio::pin!(inference);
    let result = loop {
        tokio::select! {
            result = &mut inference => break result?,
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                if worker::foreground_busy(&state).await {
                    return Err(ApiError::conflict("processor_busy","Processing yielded to a new generation. Try again later."));
                }
            }
        }
    };
    Ok(Json(
        json!({"facts": result.into_iter().map(|(fact,sources)|json!({"topic":fact.topic,"content":fact.content,"sources":sources})).collect::<Vec<_>>()}),
    ))
}
