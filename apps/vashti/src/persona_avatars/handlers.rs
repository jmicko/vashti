use axum::{
    Json,
    body::Body,
    extract::{Multipart, Path, State},
    http::{HeaderValue, header},
    response::Response,
};
use axum_extra::extract::CookieJar;
use serde::Serialize;

use crate::{
    app_state::AppState,
    auth,
    error::ApiError,
    persona_avatars::{
        models::PersonaAvatarAsset,
        service::{self, AvatarUploadInput},
    },
    rate_limit, settings,
};

#[derive(Debug, Serialize)]
pub struct PersonaAvatarResponse {
    pub asset: PersonaAvatarAsset,
}

#[derive(Debug, Serialize)]
pub struct DeletePersonaAvatarResponse {
    pub ok: bool,
}

pub async fn upload_avatar(
    State(state): State<AppState>,
    jar: CookieJar,
    mut multipart: Multipart,
) -> Result<Json<PersonaAvatarResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(
            rate_limit::user_action_key("persona_avatar_upload", &user.id),
            30,
            60 * 60,
        )
        .await?;
    let (original_filename, bytes) = parse_upload(&mut multipart).await?;
    let app_settings = settings::service::get_app_settings(&state.db).await?;
    let asset = service::create_asset(
        &state.db,
        &state.config.persona_avatars_dir(),
        &user.id,
        AvatarUploadInput {
            original_filename,
            bytes,
        },
        app_settings.max_upload_bytes,
    )
    .await?;

    Ok(Json(PersonaAvatarResponse { asset }))
}

pub async fn get_avatar(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(asset_id): Path<String>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let (asset, bytes) = service::get_asset_file(
        &state.db,
        &state.config.persona_avatars_dir(),
        &user.id,
        &asset_id,
    )
    .await?;

    let mut response = Response::new(Body::from(bytes));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&asset.mime_type)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=31536000, immutable"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("inline"),
    );

    Ok(response)
}

pub async fn delete_avatar(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(asset_id): Path<String>,
) -> Result<Json<DeletePersonaAvatarResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_unused_asset(
        &state.db,
        &state.config.persona_avatars_dir(),
        &user.id,
        &asset_id,
    )
    .await?;

    Ok(Json(DeletePersonaAvatarResponse { ok: true }))
}

async fn parse_upload(multipart: &mut Multipart) -> Result<(String, Vec<u8>), ApiError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| ApiError::bad_request("invalid_multipart", "Invalid multipart upload"))?
    {
        if field.name() != Some("file") {
            continue;
        }

        let original_filename = field.file_name().unwrap_or("profile-image").to_string();
        let bytes = field
            .bytes()
            .await
            .map_err(|_| ApiError::bad_request("invalid_upload", "Could not read profile image"))?;
        return Ok((original_filename, bytes.to_vec()));
    }

    Err(ApiError::bad_request(
        "missing_file",
        "Profile image file is required",
    ))
}
