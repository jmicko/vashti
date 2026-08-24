use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MemoryModelScope {
    pub all_models: bool,
    #[serde(default)]
    pub model_keys: Vec<String>,
}

impl Default for MemoryModelScope {
    fn default() -> Self {
        Self {
            all_models: true,
            model_keys: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryVersionResponse {
    pub id: String,
    pub memory_id: String,
    pub version_number: i64,
    pub content: String,
    pub actor_type: String,
    pub actor_user_id: String,
    pub actor_model_key: Option<String>,
    pub actor_model_name: Option<String>,
    pub source_chat_id: Option<String>,
    pub source_message_id: Option<String>,
    pub source_tool_call_id: Option<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemorySummaryResponse {
    pub id: String,
    pub excerpt: String,
    pub current_version_id: String,
    pub current_version_number: i64,
    pub model_scope: MemoryModelScope,
    pub deleted_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryResponse {
    pub id: String,
    pub current_version: MemoryVersionResponse,
    pub model_scope: MemoryModelScope,
    pub deleted_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CreateMemoryRequest {
    pub content: String,
    pub model_scope: Option<MemoryModelScope>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub expected_version: i64,
    pub content: Option<String>,
    pub model_scope: Option<MemoryModelScope>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExpectedMemoryVersionRequest {
    pub expected_version: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UpdateMemorySettingsRequest {
    pub allow_model_read: bool,
    pub allow_model_create: bool,
    pub allow_model_edit: bool,
    pub allow_model_forget: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemorySettingsResponse {
    pub allow_model_read: bool,
    pub allow_model_create: bool,
    pub allow_model_edit: bool,
    pub allow_model_forget: bool,
}
