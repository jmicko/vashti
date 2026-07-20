use std::net::SocketAddr;

use axum::{
    Json,
    extract::{ConnectInfo, State},
    http::HeaderMap,
    response::IntoResponse,
};
use axum_extra::extract::CookieJar;
use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    auth::service::{self, UserPublic},
    error::ApiError,
    private, rate_limit, settings,
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
    pub private_vault_key: Option<private::handlers::PrivateVaultKeyResponse>,
}

#[derive(Debug, Serialize)]
pub struct LogoutResponse {
    pub ok: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProfileRequest {
    pub display_name: Option<serde_json::Value>,
    pub email: Option<serde_json::Value>,
}

pub async fn session(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<SessionResponse>, ApiError> {
    let user =
        service::current_user_from_cookie(&state.db, &jar, &state.config.session_cookie_name)
            .await?;
    let (can_create_account, private_vault_key) = tokio::try_join!(
        async { Ok::<_, ApiError>(service::can_create_account(&state.db).await?) },
        async {
            let Some(user) = user.as_ref() else {
                return Ok::<_, ApiError>(None);
            };
            let key =
                private::service::get_or_create_private_vault_key(&state.db, &user.id).await?;
            Ok(Some(private::handlers::PrivateVaultKeyResponse {
                user_id: user.id.clone(),
                key_material: key.key_material,
            }))
        }
    )?;

    Ok(Json(SessionResponse {
        is_authenticated: user.is_some(),
        user,
        can_create_account,
        private_vault_key,
    }))
}

pub async fn register(
    State(state): State<AppState>,
    jar: CookieJar,
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(payload): Json<RegisterRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let app_settings = settings::service::get_app_settings(&state.db).await?;
    let client_ip =
        rate_limit::client_ip(&headers, app_settings.trust_proxy_headers, peer_addr.ip());
    let client_key = rate_limit::client_key(client_ip);
    state
        .rate_limiter
        .check(format!("auth:register:{}", client_key), 5, 60 * 60)
        .await?;

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
        Some(client_ip.to_string()),
        user_agent(&headers),
    )
    .await?;

    let cookie = service::session_cookie(
        &state.config,
        &session.id,
        app_settings.secure_session_cookies(),
    );

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
    ConnectInfo(peer_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(payload): Json<LoginRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let app_settings = settings::service::get_app_settings(&state.db).await?;
    let client_ip =
        rate_limit::client_ip(&headers, app_settings.trust_proxy_headers, peer_addr.ip());
    let client_key = rate_limit::client_key(client_ip);
    let identifier_key = rate_limit::compact_key_part(&payload.identifier, 128);
    state
        .rate_limiter
        .check(format!("auth:login-client:{}", client_key), 60, 5 * 60)
        .await?;
    state
        .rate_limiter
        .check(
            format!("auth:login:{}:{}", client_key, identifier_key),
            10,
            5 * 60,
        )
        .await?;

    let user = service::authenticate_user(&state.db, payload.identifier, payload.password).await?;
    let session = service::create_session(
        &state.db,
        &user.id,
        state.config.session_ttl_seconds,
        Some(client_ip.to_string()),
        user_agent(&headers),
    )
    .await?;

    let cookie = service::session_cookie(
        &state.config,
        &session.id,
        app_settings.secure_session_cookies(),
    );

    Ok((jar.add(cookie), Json(LoginResponse { user })))
}

pub async fn update_profile(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateProfileRequest>,
) -> Result<Json<LoginResponse>, ApiError> {
    let current = service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let user = service::update_profile(&state.db, &current.id, payload.display_name, payload.email)
        .await?;

    Ok(Json(LoginResponse { user }))
}

pub async fn logout(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<impl IntoResponse, ApiError> {
    if let Some(cookie) = jar.get(&state.config.session_cookie_name) {
        service::delete_session(&state.db, cookie.value()).await?;
    }

    let settings = settings::service::get_app_settings(&state.db).await?;
    Ok((
        jar.remove(service::expired_session_cookie(
            &state.config,
            settings.secure_session_cookies(),
        )),
        Json(LogoutResponse { ok: true }),
    ))
}

fn user_agent(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}
