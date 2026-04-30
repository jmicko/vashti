use base64::{Engine as _, engine::general_purpose};
use rand_core::{OsRng, RngCore};
use sqlx::{Row, SqlitePool};

use crate::{auth, error::ApiError, ollama::models::OllamaChatMessage};

#[derive(Clone, Debug)]
pub struct PrivateGenerationBackend {
    pub base_url: String,
}

#[derive(Debug)]
pub struct PrivateVaultKey {
    pub user_id: String,
    pub key_material: String,
}

pub async fn get_or_create_private_vault_key(
    pool: &SqlitePool,
    user_id: &str,
) -> Result<PrivateVaultKey, ApiError> {
    if let Some(row) = sqlx::query(
        r#"
        SELECT key_material
        FROM user_private_vault_keys
        WHERE user_id = ?
        "#,
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?
    {
        return Ok(PrivateVaultKey {
            user_id: user_id.to_string(),
            key_material: row.try_get("key_material")?,
        });
    }

    let mut key_bytes = [0_u8; 32];
    OsRng.fill_bytes(&mut key_bytes);
    let key_material = general_purpose::STANDARD.encode(key_bytes);
    let now = auth::service::unix_timestamp();

    sqlx::query(
        r#"
        INSERT INTO user_private_vault_keys (user_id, key_material, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        "#,
    )
    .bind(user_id)
    .bind(&key_material)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    Ok(PrivateVaultKey {
        user_id: user_id.to_string(),
        key_material,
    })
}

pub async fn get_enabled_backend(
    pool: &SqlitePool,
    backend_id: &str,
) -> Result<PrivateGenerationBackend, ApiError> {
    let Some(row) = sqlx::query(
        r#"
        SELECT base_url
        FROM ollama_backends
        WHERE id = ?
          AND is_enabled = 1
        "#,
    )
    .bind(backend_id)
    .fetch_optional(pool)
    .await?
    else {
        return Err(ApiError::not_found(
            "backend_not_found",
            "Backend not found",
        ));
    };

    Ok(PrivateGenerationBackend {
        base_url: row.try_get("base_url")?,
    })
}

pub fn validate_model_name(model_name: &str) -> Result<String, ApiError> {
    let model_name = model_name.trim();
    if model_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_model",
            "Model name is required",
        ));
    }

    Ok(model_name.to_string())
}

pub fn private_prompt_messages(
    messages: Vec<PrivateMessageInput>,
) -> Result<Vec<OllamaChatMessage>, ApiError> {
    let mut prompt_messages = Vec::new();

    for message in messages {
        let PrivateMessageInput {
            role,
            content_text,
            thinking_text,
            images,
        } = message;
        let role = validate_role(&role)?;
        let content = content_text.trim();
        let images: Vec<String> = images
            .into_iter()
            .map(|image| image.trim().to_string())
            .filter(|image| !image.is_empty())
            .collect();
        if content.is_empty() && images.is_empty() {
            continue;
        }

        prompt_messages.push(OllamaChatMessage {
            role,
            content: content.to_string(),
            thinking: thinking_text
                .map(|thinking| thinking.trim().to_string())
                .filter(|thinking| !thinking.is_empty()),
            images: if images.is_empty() {
                None
            } else {
                Some(images)
            },
        });
    }

    if prompt_messages.is_empty() {
        return Err(ApiError::bad_request(
            "empty_prompt",
            "At least one private message is required",
        ));
    }

    Ok(prompt_messages)
}

#[derive(Debug, serde::Deserialize)]
pub struct PrivateMessageInput {
    pub role: String,
    pub content_text: String,
    pub thinking_text: Option<String>,
    #[serde(default)]
    pub images: Vec<String>,
}

fn validate_role(role: &str) -> Result<String, ApiError> {
    let role = role.trim();
    if !matches!(role, "system" | "user" | "assistant") {
        return Err(ApiError::bad_request(
            "invalid_role",
            "Private message role must be system, user, or assistant",
        ));
    }

    Ok(role.to_string())
}
