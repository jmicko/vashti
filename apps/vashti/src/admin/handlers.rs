use axum::{
    Json,
    extract::{Path, State},
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    admin::service::{self, AdminUserResponse},
    app_state::AppState,
    auth,
    error::ApiError,
};

#[derive(Debug, Serialize)]
pub struct ListUsersResponse {
    pub users: Vec<AdminUserResponse>,
}

#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: Option<String>,
    pub password: String,
    pub role: Option<String>,
    pub is_disabled: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub role: Option<String>,
    pub is_disabled: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct UserMutationResponse {
    pub user: AdminUserResponse,
}

#[derive(Debug, Serialize)]
pub struct DeleteUserResponse {
    pub ok: bool,
}

pub async fn list_users(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<ListUsersResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let users = service::list_users(&state.db).await?;

    Ok(Json(ListUsersResponse { users }))
}

pub async fn create_user(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<CreateUserRequest>,
) -> Result<Json<UserMutationResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let user = service::create_user(&state.db, payload).await?;

    Ok(Json(UserMutationResponse { user }))
}

pub async fn update_user(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(user_id): Path<String>,
    Json(payload): Json<UpdateUserRequest>,
) -> Result<Json<UserMutationResponse>, ApiError> {
    let admin =
        auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let user = service::update_user(&state.db, &admin.id, &user_id, payload).await?;

    Ok(Json(UserMutationResponse { user }))
}

pub async fn delete_user(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(user_id): Path<String>,
) -> Result<Json<DeleteUserResponse>, ApiError> {
    let admin =
        auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_user(&state.db, &admin.id, &user_id).await?;

    Ok(Json(DeleteUserResponse { ok: true }))
}
