use axum::{Json, extract::State};
use axum_extra::extract::CookieJar;
use serde::Deserialize;

use crate::{app_state::AppState, auth, error::ApiError, settings::service};

#[derive(Debug, Deserialize)]
pub struct UpdateAppSettingsRequest {
    pub allow_signup: Option<bool>,
    pub signup_limit: Option<i64>,
    pub max_upload_bytes: Option<i64>,
    pub request_timeout_ms: Option<i64>,
    pub update_channel: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateNetworkSettingsRequest {
    pub network_mode: String,
    pub public_base_url: Option<String>,
    pub trust_proxy_headers: bool,
    pub admin_password: String,
    pub acknowledge_risk: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateToolSettingsRequest {
    pub tools_enabled: Option<bool>,
    pub ollama_web_search_enabled: Option<bool>,
    pub ollama_web_fetch_enabled: Option<bool>,
    pub ollama_api_key: Option<String>,
    pub clear_ollama_api_key: Option<bool>,
    pub brave_search_enabled: Option<bool>,
    pub brave_search_api_key: Option<String>,
    pub clear_brave_search_api_key: Option<bool>,
    pub direct_web_fetch_enabled: Option<bool>,
    pub tool_system_prompt: Option<String>,
    pub web_search_tool_prompt: Option<String>,
    pub web_fetch_tool_prompt: Option<String>,
    pub default_tool_permission_tags: Option<Vec<String>>,
    pub tool_permissions: Option<Vec<UpdateToolPermissionRequest>>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateToolPermissionRequest {
    pub tool_id: String,
    pub permission_tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserSettingsRequest {
    pub default_backend_id: Option<serde_json::Value>,
    pub default_model_name: Option<serde_json::Value>,
    pub theme: Option<serde_json::Value>,
}

pub async fn get_app_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::AppSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::get_app_settings(&state.db).await?;

    Ok(Json(settings))
}

pub async fn get_user_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::UserSettingsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::get_user_settings(&state.db, &user.id).await?;

    Ok(Json(settings))
}

pub async fn update_user_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateUserSettingsRequest>,
) -> Result<Json<service::UserSettingsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::update_user_settings(&state.db, &user.id, payload).await?;

    Ok(Json(settings))
}

pub async fn update_app_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateAppSettingsRequest>,
) -> Result<Json<service::AppSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let update_channel_changed = payload.update_channel.is_some();
    let settings = service::update_app_settings(&state.db, payload).await?;

    if update_channel_changed {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(error) = state
                .updates
                .check_for_update(&state.db, &state.http_client, &state.config)
                .await
            {
                tracing::warn!(
                    ?error,
                    "failed to refresh update status after channel change"
                );
            }
        });
    }

    Ok(Json(settings))
}

pub async fn update_network_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateNetworkSettingsRequest>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let user =
        auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::update_network_settings(&state.db, &user.id, payload).await?;

    let current_session_id = jar
        .get(&state.config.session_cookie_name)
        .map(|cookie| cookie.value().to_string());
    let jar = match current_session_id {
        Some(session_id) => jar.add(auth::service::session_cookie(
            &state.config,
            &session_id,
            settings.secure_session_cookies(),
        )),
        None => jar,
    };

    Ok((jar, Json(settings)))
}

pub async fn get_tool_settings(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::ToolSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::get_tool_settings(&state.db).await?;

    Ok(Json(settings))
}

pub async fn get_available_tools(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::AvailableToolsResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let tools = service::get_available_tools(&state.db, &user.id).await?;

    Ok(Json(tools))
}

pub async fn update_tool_settings(
    State(state): State<AppState>,
    jar: CookieJar,
    Json(payload): Json<UpdateToolSettingsRequest>,
) -> Result<Json<service::ToolSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::update_tool_settings(&state.db, payload).await?;

    Ok(Json(settings))
}

pub async fn dismiss_network_recovery_notice(
    State(state): State<AppState>,
    jar: CookieJar,
) -> Result<Json<service::AppSettingsResponse>, ApiError> {
    auth::service::require_admin(&state.db, &jar, &state.config.session_cookie_name).await?;
    let settings = service::dismiss_network_recovery_notice(&state.db).await?;

    Ok(Json(settings))
}
