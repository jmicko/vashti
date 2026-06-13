use std::path::Path;

use sqlx::{Row, SqlitePool};
use tokio::fs;
use uuid::Uuid;

use crate::{
    auth::service::unix_timestamp, error::ApiError, persona_avatars::models::PersonaAvatarAsset,
};

pub struct AvatarUploadInput {
    pub original_filename: String,
    pub bytes: Vec<u8>,
}

pub async fn create_asset(
    pool: &SqlitePool,
    avatars_dir: &Path,
    user_id: &str,
    input: AvatarUploadInput,
    max_upload_bytes: i64,
) -> Result<PersonaAvatarAsset, ApiError> {
    if input.bytes.is_empty() {
        return Err(ApiError::bad_request(
            "empty_avatar",
            "Profile image is empty",
        ));
    }
    if input.bytes.len() as i64 > max_upload_bytes {
        return Err(ApiError::bad_request(
            "avatar_too_large",
            "Profile image exceeds the configured upload limit",
        ));
    }

    let mime_type = detect_image_type(&input.bytes)?;
    let asset_id = Uuid::new_v4().to_string();
    let storage_path = format!("{user_id}/{asset_id}");
    let full_path = avatars_dir.join(&storage_path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent).await.map_err(|error| {
            tracing::error!(?error, "failed to create profile image directory");
            ApiError::internal("Failed to store profile image")
        })?;
    }
    fs::write(&full_path, &input.bytes).await.map_err(|error| {
        tracing::error!(?error, "failed to write profile image");
        ApiError::internal("Failed to store profile image")
    })?;

    let original_filename = normalize_filename(&input.original_filename);
    let now = unix_timestamp();
    let insert_result = sqlx::query(
        r#"
        INSERT INTO persona_avatar_assets (
            id,
            owner_user_id,
            original_filename,
            storage_path,
            mime_type,
            size_bytes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&asset_id)
    .bind(user_id)
    .bind(&original_filename)
    .bind(&storage_path)
    .bind(mime_type)
    .bind(input.bytes.len() as i64)
    .bind(now)
    .execute(pool)
    .await;

    if let Err(error) = insert_result {
        let _ = fs::remove_file(&full_path).await;
        return Err(ApiError::from(error));
    }

    Ok(PersonaAvatarAsset {
        id: asset_id,
        original_filename,
        mime_type: mime_type.to_string(),
        size_bytes: input.bytes.len() as i64,
        created_at: now,
    })
}

pub async fn get_asset_file(
    pool: &SqlitePool,
    avatars_dir: &Path,
    user_id: &str,
    asset_id: &str,
) -> Result<(PersonaAvatarAsset, Vec<u8>), ApiError> {
    let row = sqlx::query(
        r#"
        SELECT a.id,
               a.original_filename,
               a.storage_path,
               a.mime_type,
               a.size_bytes,
               a.created_at
        FROM persona_avatar_assets a
        WHERE a.id = ?
          AND (
            a.owner_user_id = ?
            OR EXISTS (
                SELECT 1
                FROM persona_versions v
                JOIN personas p ON p.id = v.persona_id
                LEFT JOIN persona_members pm
                  ON pm.persona_id = p.id
                 AND pm.user_id = ?
                WHERE v.avatar_asset_id = a.id
                  AND (
                    p.owner_user_id = ?
                    OR (p.visibility = 'public' AND p.lifecycle_state = 'active')
                    OR pm.user_id IS NOT NULL
                  )
            )
            OR EXISTS (
                SELECT 1
                FROM chats c
                JOIN persona_versions v ON v.id = c.persona_version_id
                WHERE c.user_id = ?
                  AND v.avatar_asset_id = a.id
            )
            OR EXISTS (
                SELECT 1
                FROM chat_messages m
                JOIN chats c ON c.id = m.chat_id
                JOIN persona_versions v ON v.id = m.persona_version_id
                WHERE c.user_id = ?
                  AND v.avatar_asset_id = a.id
            )
            OR EXISTS (
                SELECT 1
                FROM model_availability ma
                WHERE ma.avatar_asset_id = a.id
            )
            OR EXISTS (
                SELECT 1
                FROM user_model_preferences ump
                WHERE ump.user_id = ?
                  AND ump.avatar_asset_id = a.id
            )
          )
        "#,
    )
    .bind(asset_id)
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("avatar_not_found", "Profile image not found"))?;

    let storage_path: String = row.try_get("storage_path")?;
    let asset = row_to_asset(&row)?;
    let bytes = fs::read(avatars_dir.join(storage_path))
        .await
        .map_err(|error| {
            tracing::error!(?error, asset_id, "failed to read profile image");
            ApiError::internal("Failed to read profile image")
        })?;

    Ok((asset, bytes))
}

pub async fn ensure_asset_assignable(
    pool: &SqlitePool,
    user_id: &str,
    asset_id: Option<&str>,
) -> Result<(), ApiError> {
    let Some(asset_id) = asset_id else {
        return Ok(());
    };

    let accessible: i64 = sqlx::query_scalar(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM persona_avatar_assets a
            WHERE a.id = ?
              AND a.owner_user_id = ?
        )
        "#,
    )
    .bind(asset_id)
    .bind(user_id)
    .fetch_one(pool)
    .await?;

    if accessible == 0 {
        return Err(ApiError::bad_request(
            "invalid_avatar",
            "Profile image is not available",
        ));
    }

    Ok(())
}

pub async fn clone_asset_for_owner(
    pool: &SqlitePool,
    avatars_dir: &Path,
    asset_id: &str,
    owner_user_id: &str,
) -> Result<PersonaAvatarAsset, ApiError> {
    let row = sqlx::query(
        r#"
        SELECT original_filename,
               storage_path
        FROM persona_avatar_assets
        WHERE id = ?
        "#,
    )
    .bind(asset_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("avatar_not_found", "Profile image not found"))?;

    let original_filename: String = row.try_get("original_filename")?;
    let storage_path: String = row.try_get("storage_path")?;
    let bytes = fs::read(avatars_dir.join(storage_path))
        .await
        .map_err(|error| {
            tracing::error!(?error, asset_id, "failed to read profile image for copy");
            ApiError::internal("Failed to copy profile image")
        })?;

    create_asset(
        pool,
        avatars_dir,
        owner_user_id,
        AvatarUploadInput {
            original_filename,
            bytes,
        },
        i64::MAX,
    )
    .await
}

pub async fn delete_unused_asset(
    pool: &SqlitePool,
    avatars_dir: &Path,
    user_id: &str,
    asset_id: &str,
) -> Result<(), ApiError> {
    let row = sqlx::query(
        r#"
        SELECT storage_path
        FROM persona_avatar_assets
        WHERE id = ?
          AND owner_user_id = ?
        "#,
    )
    .bind(asset_id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| ApiError::not_found("avatar_not_found", "Profile image not found"))?;

    let reference_count: i64 = sqlx::query_scalar(
        r#"
        SELECT
            (SELECT COUNT(*) FROM persona_versions WHERE avatar_asset_id = ?)
          + (SELECT COUNT(*) FROM model_availability WHERE avatar_asset_id = ?)
          + (SELECT COUNT(*) FROM user_model_preferences WHERE avatar_asset_id = ?)
        "#,
    )
    .bind(asset_id)
    .bind(asset_id)
    .bind(asset_id)
    .fetch_one(pool)
    .await?;
    if reference_count > 0 {
        return Err(ApiError::conflict(
            "avatar_in_use",
            "Profile image is still in use",
        ));
    }

    sqlx::query("DELETE FROM persona_avatar_assets WHERE id = ? AND owner_user_id = ?")
        .bind(asset_id)
        .bind(user_id)
        .execute(pool)
        .await?;

    let storage_path: String = row.try_get("storage_path")?;
    match fs::remove_file(avatars_dir.join(storage_path)).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => tracing::warn!(?error, asset_id, "failed to remove profile image"),
    }

    Ok(())
}

fn detect_image_type(bytes: &[u8]) -> Result<&'static str, ApiError> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok("image/png");
    }
    if bytes.len() >= 3 && bytes[0..3] == [0xff, 0xd8, 0xff] {
        return Ok("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Ok("image/gif");
    }

    Err(ApiError::bad_request(
        "unsupported_avatar",
        "Profile images must be JPEG, PNG, or GIF files",
    ))
}

fn normalize_filename(filename: &str) -> String {
    let normalized = Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("profile-image")
        .trim();

    if normalized.is_empty() {
        "profile-image".to_string()
    } else {
        normalized.chars().take(240).collect()
    }
}

fn row_to_asset(row: &sqlx::sqlite::SqliteRow) -> Result<PersonaAvatarAsset, sqlx::Error> {
    Ok(PersonaAvatarAsset {
        id: row.try_get("id")?,
        original_filename: row.try_get("original_filename")?,
        mime_type: row.try_get("mime_type")?,
        size_bytes: row.try_get("size_bytes")?,
        created_at: row.try_get("created_at")?,
    })
}

#[cfg(test)]
mod tests {
    use super::detect_image_type;

    #[test]
    fn detects_supported_original_formats() {
        assert_eq!(
            detect_image_type(b"\x89PNG\r\n\x1a\nrest").unwrap(),
            "image/png"
        );
        assert_eq!(
            detect_image_type(b"\xff\xd8\xff\xe0rest").unwrap(),
            "image/jpeg"
        );
        assert_eq!(detect_image_type(b"GIF89arest").unwrap(), "image/gif");
    }

    #[test]
    fn rejects_unnormalized_webp_and_unknown_files() {
        assert!(detect_image_type(b"RIFF1234WEBP").is_err());
        assert!(detect_image_type(b"not an image").is_err());
    }
}
