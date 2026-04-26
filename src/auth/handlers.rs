use axum::{Json, extract::State, http::HeaderMap, response::IntoResponse};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth::service::{self, UserPublic},
    error::ApiError,
};

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub email: Option<String>,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub requires_approval: bool,
    pub user: service::RegisteredUser,
}

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub identifier: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub user: UserPublic,
}

#[derive(Debug, Serialize)]
pub struct SessionResponse {
    pub is_authenticated: bool,
    pub user: Option<UserPublic>,
    pub can_create_account: bool,
}

#[derive(Debug, Serialize)]
pub struct LogoutResponse {
    pub ok: bool,
}

pub async fn session(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<SessionResponse>, ApiError> {
    let user =
        service::current_user_from_cookie(&state.db, &jar, &state.config.session_cookie_name)
            .await?;
    let can_create_account = service::can_create_account(&state.db).await?;

    Ok(Json(SessionResponse {
        is_authenticated: user.is_some(),
        user,
        can_create_account,
    }))
}

pub async fn register(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Json(payload): Json<RegisterRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let registration =
        service::register_user(&state.db, payload.username, payload.email, payload.password)
            .await?;

    if registration.requires_approval {
        return Ok((
            jar,
            Json(RegisterResponse {
                requires_approval: true,
                user: registration.user,
            }),
        ));
    }

    let session = service::create_session(
        &state.db,
        &registration.user.id,
        state.config.session_ttl_seconds,
        None,
        user_agent(&headers),
    )
    .await?;

    let cookie = service::session_cookie(&state.config, &session.id);

    Ok((
        jar.add(cookie),
        Json(RegisterResponse {
            requires_approval: false,
            user: registration.user,
        }),
    ))
}

pub async fn login(
    State(state): State<AppState>,
    jar: CookieJar,
    headers: HeaderMap,
    Json(payload): Json<LoginRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let user = service::authenticate_user(&state.db, payload.identifier, payload.password).await?;
    let session = service::create_session(
        &state.db,
        &user.id,
        state.config.session_ttl_seconds,
        None,
        user_agent(&headers),
    )
    .await?;

    let cookie = service::session_cookie(&state.config, &session.id);

    Ok((jar.add(cookie), Json(LoginResponse { user })))
}

pub async fn logout(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<impl IntoResponse, ApiError> {
    if let Some(cookie) = jar.get(&state.config.session_cookie_name) {
        service::delete_session(&state.db, cookie.value()).await?;
    }

    Ok((
        jar.remove(service::expired_session_cookie(&state.config)),
        Json(LogoutResponse { ok: true }),
    ))
}

fn user_agent(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}
