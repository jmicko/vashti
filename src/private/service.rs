use sqlx::{Row, SqlitePool};

use crate::{error::ApiError, ollama::models::OllamaChatMessage};

#[derive(Clone, Debug)]
pub struct PrivateGenerationBackend {
    pub base_url: String,
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
        let role = validate_role(&message.role)?;
        let content = message.content_text.trim();
        if content.is_empty() {
            continue;
        }

        prompt_messages.push(OllamaChatMessage {
            role,
            content: content.to_string(),
            thinking: message
                .thinking_text
                .map(|thinking| thinking.trim().to_string())
                .filter(|thinking| !thinking.is_empty()),
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
