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
    rate_limit, settings,
    uploads::{
        models::Attachment,
        service::{self, UploadInput},
    },
};

#[derive(Debug, Serialize)]
pub struct AttachmentResponse {
    pub attachment: Attachment,
}

#[derive(Debug, Serialize)]
pub struct DeleteAttachmentResponse {
    pub ok: bool,
}

struct ParsedUpload {
    message_id: Option<String>,
    revision_id: Option<String>,
    original_filename: String,
    bytes: Vec<u8>,
}

pub async fn upload_attachment(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(chat_id): Path<String>,
    mut multipart: Multipart,
) -> Result<Json<AttachmentResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    state
        .rate_limiter
        .check(rate_limit::user_action_key("upload", &user.id), 60, 60)
        .await?;
    let upload = parse_upload(&mut multipart).await?;
    let app_settings = settings::service::get_app_settings(&state.db).await?;

    let attachment = service::create_attachment(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &chat_id,
        UploadInput {
            message_id: upload.message_id,
            revision_id: upload.revision_id,
            original_filename: upload.original_filename,
            bytes: upload.bytes,
        },
        app_settings.max_upload_bytes,
    )
    .await?;

    Ok(Json(AttachmentResponse { attachment }))
}

pub async fn get_attachment(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(attachment_id): Path<String>,
) -> Result<Response, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    let (attachment, bytes) = service::get_attachment_file(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &attachment_id,
    )
    .await?;

    let mut response = Response::new(Body::from(bytes));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&attachment.mime_type)
            .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream")),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&service::safe_content_disposition(
            &attachment.original_filename,
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );

    Ok(response)
}

pub async fn delete_attachment(
    State(state): State<AppState>,
    jar: CookieJar,
    Path(attachment_id): Path<String>,
) -> Result<Json<DeleteAttachmentResponse>, ApiError> {
    let user =
        auth::service::require_user(&state.db, &jar, &state.config.session_cookie_name).await?;
    service::delete_attachment(
        &state.db,
        &state.config.uploads_dir(),
        &user.id,
        &attachment_id,
    )
    .await?;

    Ok(Json(DeleteAttachmentResponse { ok: true }))
}

async fn parse_upload(multipart: &mut Multipart) -> Result<ParsedUpload, ApiError> {
    let mut message_id = None;
    let mut revision_id = None;
    let mut original_filename = None;
    let mut bytes = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| ApiError::bad_request("invalid_multipart", "Invalid multipart upload"))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "message_id" => {
                message_id = optional_field_text(field).await?;
            }
            "revision_id" => {
                revision_id = optional_field_text(field).await?;
            }
            "file" => {
                let filename = field.file_name().unwrap_or("upload").to_string();
                let file_bytes = field.bytes().await.map_err(|_| {
                    ApiError::bad_request("invalid_upload", "Could not read uploaded file")
                })?;
                original_filename = Some(filename);
                bytes = Some(file_bytes.to_vec());
            }
            _ => {}
        }
    }

    Ok(ParsedUpload {
        message_id,
        revision_id,
        original_filename: original_filename
            .ok_or_else(|| ApiError::bad_request("missing_file", "Upload file is required"))?,
        bytes: bytes
            .ok_or_else(|| ApiError::bad_request("missing_file", "Upload file is required"))?,
    })
}

async fn optional_field_text(
    field: axum::extract::multipart::Field<'_>,
) -> Result<Option<String>, ApiError> {
    let value = field
        .text()
        .await
        .map_err(|_| ApiError::bad_request("invalid_multipart", "Invalid multipart field"))?
        .trim()
        .to_string();

    Ok((!value.is_empty()).then_some(value))
}
