use crate::{app_state::AppState, auth, error::ApiError, rate_limit, uploads::service};
use axum::{
    Json,
    extract::{Path, Query, State},
};
use axum_extra::extract::CookieJar;
use serde::Deserialize;
#[derive(Deserialize)]
pub struct Selection {
    output: Option<String>,
}
/// Reading settings does not create an attachment or persist another copy of
/// the image/workflow. The existing attachment ownership check always applies.
pub async fn get(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(id): Path<String>,
    Query(selection): Query<Selection>,
) -> Result<Json<kuno_protocol::ImportedImageRecipe>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(
            rate_limit::user_action_key("image-settings", &user.id),
            60,
            60,
        )
        .await?;
    let (_, bytes) =
        service::get_attachment_file(&state.db, &state.config.uploads_dir(), &user.id, &id).await?;
    let imported = tokio::task::spawn_blocking(move || {
        kuno_image::import_png(&bytes, selection.output.as_deref())
    })
    .await
    .map_err(|_| ApiError::internal("Image settings could not be read"))?
    .map_err(import_error)?;
    Ok(Json(imported))
}

fn import_error(code: &'static str) -> ApiError {
    let message = match code {
        "comfy_metadata_missing" => "This image has no ComfyUI generation settings.",
        "comfy_outputs_ambiguous" => {
            "This image contains multiple output workflows. Choosing an output is not available in this viewer yet."
        }
        "image_not_png" => "Generation settings import currently supports PNG images.",
        _ => {
            "These image settings could not be mapped to a supported workflow. The original image is still available."
        }
    };
    ApiError::bad_request(code, message)
}
