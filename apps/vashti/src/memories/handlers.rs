use axum::{
    Json,
    extract::{Path, Query, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth,
    error::ApiError,
    memories::{
        models::{
            CreateMemoryRequest, ExpectedMemoryVersionRequest, MemoryResponse,
            MemorySettingsResponse, MemorySummaryResponse, MemoryVersionResponse,
            UpdateMemoryRequest, UpdateMemorySettingsRequest,
        },
        service::{self, ListMemoriesOptions, MemoryListStatus, MemoryMutationActor},
    },
};

#[derive(Debug, Deserialize)]
pub struct MemoryListQuery {
    pub query: Option<String>,
    pub status: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct MemoryListResponse {
    pub memories: Vec<MemorySummaryResponse>,
    pub total: i64,
}

#[derive(Debug, Serialize)]
pub struct MemoryMutationResponse {
    pub memory: MemoryResponse,
}

#[derive(Debug, Serialize)]
pub struct MemoryVersionsResponse {
    pub versions: Vec<MemoryVersionResponse>,
}

#[derive(Debug, Serialize)]
pub struct MemoryDeleteResponse {
    pub ok: bool,
}

pub async fn list_memories(
    State(state): State<AppState>,
    jar: CookieJar,
    Query(query): Query<MemoryListQuery>,
) -> Result<Json<MemoryListResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let result = service::list_memories(
        &state.db,
        &user.id,
        ListMemoriesOptions {
            query: query.query,
            status: parse_status(query.status.as_deref())?,
            limit: query.limit.unwrap_or(50),
            offset: query.offset.unwrap_or(0),
        },
    )
    .await?;
    Ok(Json(MemoryListResponse {
        memories: result.memories,
        total: result.total,
    }))
}

pub async fn get_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(MemoryMutationResponse {
        memory: service::get_memory(&state.db, &user.id, &memory_id).await?,
    }))
}

pub async fn create_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateMemoryRequest>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let memory = service::create_memory(
        &state.db,
        &user.id,
        payload,
        &MemoryMutationActor::human(&user.id),
    )
    .await?;
    state.memory_retrieval.wake();
    Ok(Json(MemoryMutationResponse { memory }))
}

pub async fn update_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
    Json(payload): Json<UpdateMemoryRequest>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let memory = service::update_memory(
        &state.db,
        &user.id,
        &memory_id,
        payload,
        &MemoryMutationActor::human(&user.id),
    )
    .await?;
    state.memory_retrieval.wake();
    Ok(Json(MemoryMutationResponse { memory }))
}

pub async fn forget_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
    Json(payload): Json<ExpectedMemoryVersionRequest>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(MemoryMutationResponse {
        memory: service::forget_memory(
            &state.db,
            &user.id,
            &memory_id,
            payload.expected_version,
            &MemoryMutationActor::human(&user.id),
        )
        .await?,
    }))
}

pub async fn restore_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
    Json(payload): Json<ExpectedMemoryVersionRequest>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let memory =
        service::restore_memory(&state.db, &user.id, &memory_id, payload.expected_version).await?;
    state.memory_retrieval.wake();
    Ok(Json(MemoryMutationResponse { memory }))
}

pub async fn permanently_delete_memory(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
) -> Result<Json<MemoryDeleteResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    service::permanently_delete_memory(&state.db, &user.id, &memory_id).await?;
    Ok(Json(MemoryDeleteResponse { ok: true }))
}

pub async fn list_versions(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(memory_id): Path<String>,
) -> Result<Json<MemoryVersionsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(MemoryVersionsResponse {
        versions: service::list_versions(&state.db, &user.id, &memory_id).await?,
    }))
}

pub async fn restore_version(
    State(state): State<AppState>,
    jar: CookieJar,
    Path((memory_id, version_id)): Path<(String, String)>,
    Json(payload): Json<ExpectedMemoryVersionRequest>,
) -> Result<Json<MemoryMutationResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    let memory = service::restore_version(
        &state.db,
        &user.id,
        &memory_id,
        &version_id,
        payload.expected_version,
    )
    .await?;
    state.memory_retrieval.wake();
    Ok(Json(MemoryMutationResponse { memory }))
}

pub async fn get_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<MemorySettingsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(
        service::get_memory_settings(&state.db, &user.id).await?,
    ))
}

pub async fn update_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateMemorySettingsRequest>,
) -> Result<Json<MemorySettingsResponse>, ApiError> {
    let user = require_user(&state, &jar).await?;
    Ok(Json(
        service::update_memory_settings(&state.db, &user.id, payload).await?,
    ))
}

async fn require_user(
    state: &AppState,
    jar: &CookieJar,
) -> Result<crate::auth::service::UserPublic, ApiError> {
    auth::service::require_user(&state.db, jar, &state.config.session_cookie_name).await
}

fn parse_status(value: Option<&str>) -> Result<MemoryListStatus, ApiError> {
    match value.unwrap_or("active") {
        "active" => Ok(MemoryListStatus::Active),
        "forgotten" => Ok(MemoryListStatus::Forgotten),
        "all" => Ok(MemoryListStatus::All),
        _ => Err(ApiError::bad_request(
            "invalid_memory_status",
            "Memory status must be active, forgotten, or all",
        )),
    }
}
