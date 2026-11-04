use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderValue, header},
    response::Response,
    routing::{get, post},
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{models::*, sandbox, service};
use crate::{app_state::AppState, auth, error::ApiError};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(bootstrap))
        .route("/sandbox-status", get(sandbox_status))
        .route("/preferences", axum::routing::put(preferences))
        .route("/summary", get(summary))
        .route("/tasks", get(list).post(create))
        .route("/tasks/{id}", get(task).put(update))
        .route("/tasks/{id}/archive", post(archive))
        .route("/tasks/{id}/viewed", post(viewed))
        .route("/tasks/{id}/runs", post(start))
        .route("/runs/{id}", get(run))
        .route("/runs/{id}/control", post(control))
        .route("/runs/{id}/reply", post(reply))
        .route("/runs/{id}/actions/{action}/decision", post(decide))
        .route("/artifacts/{id}", get(artifact).delete(delete_artifact))
}

async fn owner(state: &AppState, jar: &CookieJar) -> Result<String, ApiError> {
    Ok(
        auth::service::require_user(&state.db, jar, &state.config.session_cookie_name)
            .await?
            .id,
    )
}

async fn bootstrap(State(state): State<AppState>, jar: CookieJar) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let config = service::preferences(&state.db, &owner).await?;
    Ok(Json(
        json!({"config":config,"storage":"server","preferred_model":PREFERRED_MODEL}),
    ))
}

async fn sandbox_status(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let allowed =
        crate::settings::service::tool_is_available_for_user(&state.db, &owner, SANDBOX_TOOL_ID)
            .await?;
    let sandbox = if allowed {
        sandbox::availability().await
    } else {
        sandbox::Availability {
            available: false,
            image: sandbox::IMAGE.into(),
            reason: Some(
                "Sandbox access is controlled by the server administrator in Tools permissions."
                    .into(),
            ),
            image_id: None,
        }
    };
    Ok(Json(json!({"sandbox":sandbox})))
}

async fn preferences(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(config): Json<TaskConfig>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    Ok(Json(
        json!({"config":service::save_preferences(&state.db,&owner,config).await?}),
    ))
}

async fn summary(State(state): State<AppState>, jar: CookieJar) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let rows=sqlx::query_as::<_,(Option<String>,i64,Option<i64>,Option<String>)>("SELECT r.status,t.viewed_at,r.finished_at,t.schedule_error FROM assistant_tasks t LEFT JOIN assistant_runs r ON r.id=t.last_run_id WHERE t.user_id=? AND t.archived_at IS NULL")
        .bind(owner).fetch_all(&state.db).await?;
    let attention = rows
        .iter()
        .filter(|(status, viewed, finished, error)| {
            error.is_some()
                || matches!(
                    status.as_deref(),
                    Some("waiting_approval" | "waiting_input" | "interrupted" | "limit_reached")
                )
                || finished.is_some_and(|at| at > *viewed)
        })
        .count();
    let active = rows
        .iter()
        .filter(|(status, _, _, _)| matches!(status.as_deref(), Some("queued" | "running")))
        .count();
    Ok(Json(json!({"attention":attention,"active":active})))
}

#[derive(Deserialize, Default)]
struct Page {
    offset: Option<i64>,
    after: Option<i64>,
    actions_after: Option<i64>,
    archived: Option<bool>,
    attention: Option<bool>,
    scheduled: Option<bool>,
    search: Option<String>,
}
async fn list(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(page): Query<Page>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let tasks = if page.attention.unwrap_or(false)
        || page.scheduled.unwrap_or(false)
        || page.search.as_ref().is_some_and(|s| !s.is_empty())
    {
        service::list_filtered_tasks(
            &state.db,
            &owner,
            page.offset.unwrap_or(0),
            page.archived.unwrap_or(false),
            page.attention.unwrap_or(false),
            page.scheduled.unwrap_or(false),
            page.search.as_deref().unwrap_or_default(),
        )
        .await?
    } else if page.archived.unwrap_or(false) {
        service::list_archived_tasks(&state.db, &owner, page.offset.unwrap_or(0)).await?
    } else {
        service::list_tasks(&state.db, &owner, page.offset.unwrap_or(0)).await?
    };
    let summaries: Vec<Value> = tasks.into_iter().map(|t| json!({"id":t.id,"title":t.title,"instructions":t.instructions.chars().take(180).collect::<String>(),"interval_seconds":t.interval_seconds,"next_run_at":t.next_run_at,"schedule_enabled":t.schedule_enabled,"schedule_error":t.schedule_error,"archived_at":t.archived_at,"last_run_id":t.last_run_id,"last_status":t.last_status,"needs_attention":t.needs_attention,"created_at":t.created_at,"updated_at":t.updated_at,"revision":t.revision})).collect();
    Ok(Json(json!({"tasks":summaries})))
}
async fn create(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<SaveTask>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let task = service::save_task(&state.db, &owner, None, payload).await?;
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"task":task})))
}
async fn update(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(payload): Json<SaveTask>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let task = service::save_task(&state.db, &owner, Some(&id), payload).await?;
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"task":task})))
}
async fn task(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    Ok(Json(
        json!({"task":service::get_task(&state.db,&owner,&id).await?,"runs":service::list_runs(&state.db,&owner,&id).await?}),
    ))
}
async fn start(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(payload): Json<StartRun>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let run = service::start_run(&state.db, &owner, &id, payload, false).await?;
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"run":run})))
}
async fn run(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Query(page): Query<Page>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let run = service::get_run(&state.db, &owner, &id).await?;
    let events = service::events(&state.db, &owner, &id, page.after.unwrap_or(0)).await?;
    let actions =
        service::actions_since(&state.db, &owner, &id, page.actions_after.unwrap_or(-1)).await?;
    let artifacts = service::artifacts(&state.db, &owner, &id).await?;
    Ok(Json(
        json!({"run":run,"more_events":events.len()==500,"events":events,"actions":actions,"artifacts":artifacts}),
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    command: String,
}
async fn control(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(payload): Json<Control>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    service::control(&state.db, &owner, &id, &payload.command).await?;
    if payload.command != "resume" {
        state.assistant_jobs.cancel(&id).await;
    }
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"ok":true})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    text: String,
    question_id: Option<String>,
}
async fn reply(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(payload): Json<Reply>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    service::reply_question(
        &state.db,
        &owner,
        &id,
        &payload.text,
        payload.question_id.as_deref(),
    )
    .await?;
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"ok":true})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    approve: bool,
}
async fn decide(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((id, action)): Path<(String, String)>,
    Json(payload): Json<Decision>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    service::decide(&state.db, &owner, &id, &action, payload.approve).await?;
    state.assistant_jobs.wake.notify_one();
    Ok(Json(json!({"ok":true})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Archive {
    archived: bool,
}
async fn archive(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Json(payload): Json<Archive>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    service::archive(&state.db, &owner, &id, payload.archived).await?;
    Ok(Json(json!({"ok":true})))
}
async fn viewed(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    service::viewed(&state.db, &owner, &id).await?;
    Ok(Json(json!({"ok":true})))
}
async fn artifact(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let owner = owner(&state, &jar).await?;
    let value=sqlx::query_as::<_,Artifact>("SELECT a.* FROM assistant_artifacts a JOIN assistant_runs r ON r.id=a.run_id WHERE a.id=? AND r.user_id=?")
        .bind(&id).bind(owner).fetch_optional(&state.db).await?
        .ok_or_else(||ApiError::not_found("artifact_not_found","Artifact not found"))?;
    let bytes = tokio::fs::read(sandbox::artifact_path(&state, &id)?)
        .await
        .map_err(|_| ApiError::not_found("artifact_not_found", "Artifact bytes are unavailable"))?;
    if bytes.len() != value.size as usize || bytes.len() > sandbox::ARTIFACT_LIMIT {
        return Err(ApiError::internal(
            "Artifact size does not match its record",
        ));
    }
    let disposition = HeaderValue::from_str(&format!("attachment; filename=\"{}\"", value.name))
        .map_err(|_| ApiError::internal("Invalid artifact filename"))?;
    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_DISPOSITION, disposition)
        .header(header::CACHE_CONTROL, "private, no-store")
        .header("X-Content-Type-Options", "nosniff")
        .body(Body::from(bytes))
        .map_err(|_| ApiError::internal("Could not serve artifact"))
}

async fn delete_artifact(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let owner = owner(&state, &jar).await?;
    let path = sandbox::artifact_path(&state, &id)?;
    let deleted=sqlx::query("DELETE FROM assistant_artifacts WHERE id=? AND run_id IN (SELECT id FROM assistant_runs WHERE user_id=?)")
        .bind(&id).bind(owner).execute(&state.db).await?;
    if deleted.rows_affected() != 1 {
        return Err(ApiError::not_found(
            "artifact_not_found",
            "Artifact not found",
        ));
    }
    if let Err(error) = tokio::fs::remove_file(path).await
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!("assistant artifact cleanup deferred until restart");
    }
    Ok(Json(json!({"ok":true})))
}
