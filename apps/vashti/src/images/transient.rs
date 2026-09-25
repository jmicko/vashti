//! Private image jobs exist only in RAM. Never route these through saved runs,
//! assets, prompt history, attachment storage, or diagnostic payload logging.
use super::{handlers, models::SeedMode, service};
use crate::{app_state::AppState, error::ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use kuno_protocol::ImageRecipe;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Default)]
pub struct Jobs(Mutex<HashMap<String, Entry>>);
struct Entry {
    owner: String,
    touched: Instant,
    cancel: CancellationToken,
    status: &'static str,
    recipe: ImageRecipe,
    ignored: Vec<String>,
    output: Option<Vec<u8>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub operation: String,
    pub prompt: String,
    #[serde(default = "empty_settings")]
    pub settings: Value,
    #[serde(default)]
    pub reference_images: Vec<String>,
}
fn empty_settings() -> Value {
    json!({})
}
fn missing() -> ApiError {
    ApiError::not_found(
        "private_image_missing",
        "Private image expired or was not found",
    )
}
pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/private",
            post(start).layer(DefaultBodyLimit::max(64 * 1024 * 1024)),
        )
        .route("/private/{id}", get(status))
        .route("/private/{id}/result", get(result))
        .route("/private/{id}/stop", post(stop))
        .layer(axum::middleware::map_response(
            |mut response: axum::response::Response| async move {
                response.headers_mut().insert(
                    axum::http::header::CACHE_CONTROL,
                    axum::http::HeaderValue::from_static("no-store"),
                );
                response
            },
        ))
}
async fn start(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(request): Json<Request>,
) -> Result<Json<Value>, ApiError> {
    let user = handlers::user(&state, &jar).await?;
    state
        .rate_limiter
        .check(
            crate::rate_limit::user_action_key("private-image", &user),
            30,
            60,
        )
        .await?;
    Ok(Json(json!({"id":submit(&state, &user, request).await?})))
}
pub(super) async fn submit(
    state: &AppState,
    user: &str,
    request: Request,
) -> Result<String, ApiError> {
    let edit = match request.operation.as_str() {
        "generate" => false,
        "edit" => true,
        _ => {
            return Err(ApiError::bad_request(
                "image_operation_invalid",
                "Choose generate or edit",
            ));
        }
    };
    if (edit && request.reference_images.is_empty())
        || (!edit && !request.reference_images.is_empty())
    {
        return Err(ApiError::bad_request(
            "image_references_invalid",
            "Editing requires source images; generation does not accept them",
        ));
    }
    let preferences = service::preferences(&state.db, user).await?;
    let profile = if edit {
        &preferences.edit
    } else {
        &preferences.generate
    };
    let (mut recipe, ignored) =
        service::resolve_tool_recipe(profile, request.prompt, &request.settings)?;
    // An explicit permitted seed is authoritative. Otherwise use the profile.
    if !(profile.model_may_change.contains("seed") && request.settings.get("seed").is_some()) {
        recipe.seed = service::seed(
            &recipe.seed,
            if profile.seed_mode == SeedMode::Fixed {
                SeedMode::Fixed
            } else {
                SeedMode::Random
            },
            0,
        )?;
    }
    recipe.reference_images = request.reference_images;
    recipe.validate_execution().map_err(|_| {
        ApiError::bad_request(
            "image_request_invalid",
            "Unsupported private image settings",
        )
    })?;
    let id = Uuid::new_v4().to_string();
    let cancel = CancellationToken::new();
    let mut jobs = state.image_jobs.private.0.lock().await;
    if jobs.len() >= 8 || jobs.values().filter(|e| e.owner == user).count() >= 2 {
        return Err(ApiError::conflict(
            "private_image_busy",
            "Wait for or stop an existing image request",
        ));
    }
    let mut public_recipe = recipe.clone();
    public_recipe.reference_images.clear();
    jobs.insert(
        id.clone(),
        Entry {
            owner: user.into(),
            touched: Instant::now(),
            cancel: cancel.clone(),
            status: "generating",
            recipe: public_recipe,
            ignored,
            output: None,
        },
    );
    drop(jobs);
    let state = state.clone();
    let job = id.clone();
    tokio::spawn(async move {
        let preview_key = job.clone();
        let outcome = {
            let worker_state = state.clone();
            let work = async move {
                let _permit = worker_state
                    .image_jobs
                    .slots
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| ApiError::internal("Image service unavailable"))?;
                service::generate(&worker_state, &recipe, &preview_key).await
            };
            tokio::pin!(work);
            let deadline = tokio::time::sleep(Duration::from_secs(3600));
            tokio::pin!(deadline);
            let mut tick = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break None,
                    _ = &mut deadline => break None,
                    _ = tick.tick() => {
                        let jobs = state.image_jobs.private.0.lock().await;
                        if jobs.get(&job).is_none_or(|e| e.touched.elapsed() > Duration::from_secs(30)) { break None; }
                    },
                    result = &mut work => break Some(result),
                }
            }
        }; // Drop the request and its input buffers before waiting for collection.
        let mut jobs = state.image_jobs.private.0.lock().await;
        match (jobs.get_mut(&job), outcome) {
            (Some(entry), Some(Ok(bytes))) if !cancel.is_cancelled() => {
                entry.output = Some(bytes);
                entry.status = "complete";
            }
            (Some(entry), Some(Err(_))) if !cancel.is_cancelled() => {
                entry.status = "failed";
            }
            _ => {
                jobs.remove(&job);
                return;
            }
        }
        drop(jobs);
        // Completed bytes are never retained indefinitely, even if polled.
        tokio::select! { _ = cancel.cancelled() => (), _ = tokio::time::sleep(Duration::from_secs(60)) => () }
        state.image_jobs.private.0.lock().await.remove(&job);
    });
    Ok(id)
}
async fn status(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let user = handlers::user(&state, &jar).await?;
    let mut value = read_status(&state.image_jobs.private, &user, &id).await?;
    value["preview"] = serde_json::to_value(service::preview(&state, &id)).unwrap_or(Value::Null);
    Ok(Json(value))
}
pub(super) async fn read_status(jobs: &Jobs, user: &str, id: &str) -> Result<Value, ApiError> {
    let mut jobs = jobs.0.lock().await;
    let entry = jobs
        .get_mut(id)
        .filter(|e| e.owner == user)
        .ok_or_else(missing)?;
    entry.touched = Instant::now();
    Ok(json!({"status":entry.status,"recipe":entry.recipe,"ignored_settings":entry.ignored}))
}
async fn result(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let user = handlers::user(&state, &jar).await?;
    Ok((
        [("content-type", "image/png"), ("cache-control", "no-store")],
        take_result(&state.image_jobs.private, &user, &id).await?,
    ))
}
pub(super) async fn take_result(jobs: &Jobs, user: &str, id: &str) -> Result<Vec<u8>, ApiError> {
    let mut jobs = jobs.0.lock().await;
    let entry = jobs
        .get(id)
        .filter(|e| e.owner == user)
        .ok_or_else(missing)?;
    if entry.output.is_none() {
        return Err(ApiError::conflict(
            "private_image_pending",
            "The private image is not ready",
        ));
    }
    let entry = jobs.remove(id).unwrap();
    entry.cancel.cancel();
    Ok(entry.output.unwrap())
}
async fn stop(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let user = handlers::user(&state, &jar).await?;
    let mut jobs = state.image_jobs.private.0.lock().await;
    if jobs.get(&id).is_some_and(|e| e.owner == user) {
        jobs.remove(&id).unwrap().cancel.cancel();
    }
    Ok(Json(json!({"stopped":true})))
}
