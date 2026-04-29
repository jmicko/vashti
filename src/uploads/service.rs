use std::path::{Path, PathBuf};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use sqlx::{Row, SqlitePool, Transaction};
use tokio::fs;
use uuid::Uuid;

use crate::{auth::service::unix_timestamp, error::ApiError, uploads::models::Attachment};

const MAX_ATTACHMENTS_PER_MESSAGE: usize = 10;
const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["gif", "jpeg", "jpg", "png", "webp"];

pub struct UploadInput {
    pub message_id: Option<String>,
    pub revision_id: Option<String>,
    pub original_filename: String,
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

pub async fn create_attachment(
    pool: &SqlitePool,
    uploads_dir: &Path,
    user_id: &str,
    chat_id: &str,
    input: UploadInput,
    max_upload_bytes: i64,
) -> Result<Attachment, ApiError> {
    ensure_chat_owner(pool, user_id, chat_id).await?;
    validate_revision_target(
        pool,
        user_id,
        chat_id,
        input.message_id.as_deref(),
        input.revision_id.as_deref(),
    )
    .await?;

    if input.bytes.is_empty() {
        return Err(ApiError::bad_request(
            "empty_upload",
            "Uploaded file is empty",
        ));
    }

    if input.bytes.len() as i64 > max_upload_bytes {
        return Err(ApiError::bad_request(
            "upload_too_large",
            "Uploaded file exceeds the configured size limit",
        ));
    }

    let original_filename = normalize_filename(&input.original_filename);
    let attachment_kind = classify_attachment(&original_filename, &input.mime_type, &input.bytes)?;
    let attachment_id = Uuid::new_v4().to_string();
    let storage_path = format!("{chat_id}/{attachment_id}");
    let full_path = uploads_dir.join(&storage_path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            tracing::error!(?error, "failed to create upload directory");
            ApiError::internal("Failed to store upload")
        })?;
    }

    fs::write(&full_path, &input.bytes).await.map_err(|error| {
        tracing::error!(?error, "failed to write upload");
        ApiError::internal("Failed to store upload")
    })?;

    let now = unix_timestamp();
    let insert_result = sqlx::query(
        r#"
        INSERT INTO attachments (
            id,
            user_id,
            chat_id,
            message_id,
            revision_id,
            original_filename,
            storage_path,
            mime_type,
            size_bytes,
            attachment_kind,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&attachment_id)
    .bind(user_id)
    .bind(chat_id)
    .bind(&input.message_id)
    .bind(&input.revision_id)
    .bind(&original_filename)
    .bind(&storage_path)
    .bind(&input.mime_type)
    .bind(input.bytes.len() as i64)
    .bind(&attachment_kind)
    .bind(now)
    .execute(pool)
    .await;

    if let Err(error) = insert_result {
        let _ = fs::remove_file(&full_path).await;
        return Err(ApiError::from(error));
    }

    get_attachment(pool, user_id, &attachment_id).await
}

pub async fn get_attachment(
    pool: &SqlitePool,
    user_id: &str,
    attachment_id: &str,
) -> Result<Attachment, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT id,
               chat_id,
               message_id,
               revision_id,
               original_filename,
               mime_type,
               size_bytes,
               attachment_kind,
               created_at
        FROM attachments
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(attachment_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("attachment_not_found", "Attachment not found"))?;

    row_to_attachment(row)
}

pub async fn get_attachment_file(
    pool: &SqlitePool,
    uploads_dir: &Path,
    user_id: &str,
    attachment_id: &str,
) -> Result<(Attachment, Vec<u8>), ApiError> {
    let row = sqlx::query(
        r#"
        SELECT id,
               chat_id,
               message_id,
               revision_id,
               original_filename,
               storage_path,
               mime_type,
               size_bytes,
               attachment_kind,
               created_at
        FROM attachments
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(attachment_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("attachment_not_found", "Attachment not found"))?;

    let storage_path: String = row.try_get("storage_path")?;
    let attachment = row_to_attachment(row)?;
    let bytes = fs::read(uploads_dir.join(storage_path))
        .await
        .map_err(|error| {
            tracing::error!(?error, attachment_id, "failed to read attachment file");
            ApiError::internal("Failed to read attachment")
        })?;

    Ok((attachment, bytes))
}

pub async fn delete_attachment(
    pool: &SqlitePool,
    uploads_dir: &Path,
    user_id: &str,
    attachment_id: &str,
) -> Result<(), ApiError> {
    let row = sqlx::query(
        r#"
        SELECT storage_path
        FROM attachments
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(attachment_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("attachment_not_found", "Attachment not found"))?;
    let storage_path: String = row.try_get("storage_path")?;

    sqlx::query(
        r#"
        DELETE FROM attachments
        WHERE id = ?
          AND user_id = ?
        "#,
    )
    .bind(attachment_id)
    .bind(user_id)
    .execute(pool)
    .await?;

    let full_path = uploads_dir.join(storage_path);
    match fs::remove_file(&full_path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(?error, attachment_id, "failed to remove attachment file");
        }
    }

    Ok(())
}

pub async fn claim_pending_attachments(
    tx: &mut Transaction<'_, sqlx::Sqlite>,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    revision_id: &str,
    attachment_ids: &[String],
) -> Result<(), ApiError> {
    if attachment_ids.len() > MAX_ATTACHMENTS_PER_MESSAGE {
        return Err(ApiError::bad_request(
            "too_many_attachments",
            "Too many attachments were provided",
        ));
    }

    for attachment_id in attachment_ids {
        let attachment_id = attachment_id.trim();
        if attachment_id.is_empty() {
            return Err(ApiError::bad_request(
                "invalid_attachment",
                "Attachment id is required",
            ));
        }

        let result = sqlx::query(
            r#"
            UPDATE attachments
            SET message_id = ?,
                revision_id = ?
            WHERE id = ?
              AND user_id = ?
              AND chat_id = ?
              AND message_id IS NULL
              AND revision_id IS NULL
            "#,
        )
        .bind(message_id)
        .bind(revision_id)
        .bind(attachment_id)
        .bind(user_id)
        .bind(chat_id)
        .execute(&mut **tx)
        .await?;

        if result.rows_affected() == 0 {
            return Err(ApiError::bad_request(
                "invalid_attachment",
                "Attachment is not available for this message",
            ));
        }
    }

    Ok(())
}

pub async fn list_revision_attachments(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    revision_id: &str,
) -> Result<Vec<Attachment>, ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT id,
               chat_id,
               message_id,
               revision_id,
               original_filename,
               mime_type,
               size_bytes,
               attachment_kind,
               created_at
        FROM attachments
        WHERE user_id = ?
          AND chat_id = ?
          AND message_id = ?
          AND revision_id = ?
        ORDER BY created_at ASC, id ASC
        "#,
    )
    .bind(user_id)
    .bind(chat_id)
    .bind(message_id)
    .bind(revision_id)
    .fetch_all(pool)
    .await?;

    rows.into_iter().map(row_to_attachment).collect()
}

pub async fn prompt_attachment_payload(
    pool: &SqlitePool,
    uploads_dir: &Path,
    user_id: &str,
    chat_id: &str,
    message_id: &str,
    revision_id: &str,
) -> Result<(String, Vec<String>), ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT original_filename,
               storage_path,
               attachment_kind
        FROM attachments
        WHERE user_id = ?
          AND chat_id = ?
          AND message_id = ?
          AND revision_id = ?
        ORDER BY created_at ASC, id ASC
        "#,
    )
    .bind(user_id)
    .bind(chat_id)
    .bind(message_id)
    .bind(revision_id)
    .fetch_all(pool)
    .await?;

    let mut text_payload = String::new();
    let mut images = Vec::new();

    for row in rows {
        let original_filename: String = row.try_get("original_filename")?;
        let storage_path: String = row.try_get("storage_path")?;
        let attachment_kind: String = row.try_get("attachment_kind")?;
        let bytes = fs::read(uploads_dir.join(storage_path))
            .await
            .map_err(|error| {
                tracing::error!(?error, "failed to read prompt attachment");
                ApiError::internal("Failed to read attachment")
            })?;

        if attachment_kind == "image" {
            images.push(STANDARD.encode(bytes));
            continue;
        }

        let text = String::from_utf8(bytes).map_err(|error| {
            tracing::error!(?error, "stored text attachment is not valid UTF-8");
            ApiError::internal("Failed to read text attachment")
        })?;

        text_payload.push_str("\n\nAttached file: ");
        text_payload.push_str(&original_filename);
        text_payload.push_str("\n```text\n");
        text_payload.push_str(&text);
        if !text.ends_with('\n') {
            text_payload.push('\n');
        }
        text_payload.push_str("```");
    }

    Ok((text_payload, images))
}

pub fn safe_content_disposition(filename: &str) -> String {
    let escaped = filename.replace('\\', "\\\\").replace('"', "\\\"");
    format!("attachment; filename=\"{escaped}\"")
}

fn classify_attachment(filename: &str, mime_type: &str, bytes: &[u8]) -> Result<String, ApiError> {
    let extension = Path::new(filename)
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);

    if mime_type.to_ascii_lowercase().starts_with("image/")
        && extension
            .as_deref()
            .is_some_and(|extension| SUPPORTED_IMAGE_EXTENSIONS.contains(&extension))
    {
        return Ok("image".to_string());
    }

    if let Ok(text) = std::str::from_utf8(bytes) {
        if !text.contains('\0') {
            return Ok("text".to_string());
        }
    }

    Err(ApiError::bad_request(
        "unsupported_attachment_type",
        "Only image files and UTF-8 text files are supported",
    ))
}

fn normalize_filename(filename: &str) -> String {
    let filename = PathBuf::from(filename)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("upload")
        .trim()
        .replace(['/', '\\'], "_");

    if filename.is_empty() {
        "upload".to_string()
    } else {
        filename
    }
}

async fn ensure_chat_owner(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
) -> Result<(), ApiError> {
    let exists: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM chats
            WHERE id = ?
              AND user_id = ?
              AND archived_at IS NULL
        )
        "#,
    )
    .bind(chat_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    if exists == 0 {
        return Err(ApiError::not_found("chat_not_found", "Chat not found"));
    }

    Ok(())
}

async fn validate_revision_target(
    pool: &SqlitePool,
    user_id: &str,
    chat_id: &str,
    message_id: Option<&str>,
    revision_id: Option<&str>,
) -> Result<(), ApiError> {
    match (message_id, revision_id) {
        (None, None) => Ok(()),
        (Some(message_id), Some(revision_id)) => {
            let exists: i64 = sqlx::query_scalar(
                r#"
                SELECT EXISTS(
                    SELECT 1
                    FROM chat_message_revisions r
                    JOIN chat_messages m ON m.id = r.message_id
                    JOIN chats c ON c.id = m.chat_id
                    WHERE m.id = ?
                      AND r.id = ?
                      AND m.chat_id = ?
                      AND c.user_id = ?
                      AND c.archived_at IS NULL
                )
                "#,
            )
            .bind(message_id)
            .bind(revision_id)
            .bind(chat_id)
            .bind(user_id)
            .fetch_one(pool)
            .await?;

            if exists == 0 {
                return Err(ApiError::bad_request(
                    "invalid_revision",
                    "Message revision not found for this chat",
                ));
            }

            Ok(())
        }
        _ => Err(ApiError::bad_request(
            "invalid_attachment_target",
            "message_id and revision_id must be provided together",
        )),
    }
}

fn row_to_attachment(row: sqlx::sqlite::SqliteRow) -> Result<Attachment, ApiError> {
    Ok(Attachment {
        id: row.try_get("id")?,
        chat_id: row.try_get("chat_id")?,
        message_id: row.try_get("message_id")?,
        revision_id: row.try_get("revision_id")?,
        original_filename: row.try_get("original_filename")?,
        mime_type: row.try_get("mime_type")?,
        size_bytes: row.try_get("size_bytes")?,
        attachment_kind: row.try_get("attachment_kind")?,
        created_at: row.try_get("created_at")?,
    })
}
