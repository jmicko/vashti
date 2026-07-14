use axum::{
    Json,
    extract::{Path, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth,
    context_blocks::{
        models::{ContextBlockResponse, ContextBlockVersionResponse, ContextCategoryResponse},
        service,
    },
    error::ApiError,
};

#[derive(Debug, Serialize)]
pub struct ContextLibraryResponse {
    pub categories: Vec<ContextCategoryResponse>,
    pub blocks: Vec<ContextBlockResponse>,
}

#[derive(Debug, Serialize)]
pub struct ContextCategoryMutationResponse {
    pub category: ContextCategoryResponse,
}

#[derive(Debug, Serialize)]
pub struct ContextBlockMutationResponse {
    pub block: ContextBlockResponse,
}

#[derive(Debug, Serialize)]
pub struct ContextBlockVersionsResponse {
    pub versions: Vec<ContextBlockVersionResponse>,
}

#[derive(Debug, Serialize)]
pub struct ContextDeleteResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct CreateContextCategoryRequest {
    pub name: String,
    pub selection_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateContextCategoryRequest {
    pub name: Option<String>,
    pub selection_mode: Option<String>,
    pub sort_order: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct CreateContextBlockRequest {
    pub category_id: Option<String>,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateContextBlockRequest {
    pub category_id: Option<Option<String>>,
    pub name: Option<String>,
    pub content: Option<String>,
    pub sort_order: Option<i64>,
}

pub async fn get_library(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<ContextLibraryResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let (categories, blocks) = service::list_library(&state.db, &user.id).await?;
    Ok(Json(ContextLibraryResponse { categories, blocks }))
}

pub async fn create_category(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateContextCategoryRequest>,
) -> Result<Json<ContextCategoryMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let category = service::create_category(&state.db, &user.id, payload).await?;
    Ok(Json(ContextCategoryMutationResponse { category }))
}

pub async fn update_category(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(category_id): Path<String>,
    Json(payload): Json<UpdateContextCategoryRequest>,
) -> Result<Json<ContextCategoryMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let category = service::update_category(&state.db, &user.id, &category_id, payload).await?;
    Ok(Json(ContextCategoryMutationResponse { category }))
}

pub async fn delete_category(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(category_id): Path<String>,
) -> Result<Json<ContextDeleteResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_category(&state.db, &user.id, &category_id).await?;
    Ok(Json(ContextDeleteResponse { ok: true }))
}

pub async fn create_block(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateContextBlockRequest>,
) -> Result<Json<ContextBlockMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let block = service::create_block(&state.db, &user.id, payload).await?;
    Ok(Json(ContextBlockMutationResponse { block }))
}

pub async fn update_block(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(block_id): Path<String>,
    Json(payload): Json<UpdateContextBlockRequest>,
) -> Result<Json<ContextBlockMutationResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let block = service::update_block(&state.db, &user.id, &block_id, payload).await?;
    Ok(Json(ContextBlockMutationResponse { block }))
}

pub async fn delete_block(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(block_id): Path<String>,
) -> Result<Json<ContextDeleteResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_block(&state.db, &user.id, &block_id).await?;
    Ok(Json(ContextDeleteResponse { ok: true }))
}

pub async fn list_block_versions(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(block_id): Path<String>,
) -> Result<Json<ContextBlockVersionsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let versions = service::list_versions(&state.db, &user.id, &block_id).await?;
    Ok(Json(ContextBlockVersionsResponse { versions }))
}
