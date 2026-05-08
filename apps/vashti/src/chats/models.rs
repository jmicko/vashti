use serde::{Deserialize, Serialize};

use crate::uploads::models::Attachment;

#[derive(Debug, Serialize)]
pub struct ChatSummary {
    pub id: String,
    pub title: String,
    pub default_backend_id: String,
    pub backend_name: String,
    pub default_model_name: String,
    pub persona_id: Option<String>,
    pub persona_version_id: Option<String>,
    pub persona_name: Option<String>,
    pub updated_at: i64,
    pub last_message_at: i64,
    pub message_count: i64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct ChatToolPreferences {
    pub tool_use_enabled: bool,
    pub web_search_enabled: bool,
    pub web_fetch_enabled: bool,
}

impl Default for ChatToolPreferences {
    fn default() -> Self {
        Self {
            tool_use_enabled: true,
            web_search_enabled: true,
            web_fetch_enabled: true,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatDetail {
    pub id: String,
    pub title: String,
    pub default_backend_id: String,
    pub backend_name: String,
    pub default_model_name: String,
    pub persona_id: Option<String>,
    pub persona_version_id: Option<String>,
    pub persona_name: Option<String>,
    pub tool_preferences: ChatToolPreferences,
    pub active_root_message_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize)]
pub struct ChatMessage {
    pub id: String,
    pub parent_message_id: Option<String>,
    pub active_child_message_id: Option<String>,
    pub active_revision_id: Option<String>,
    pub role: String,
    pub status: String,
    pub is_deleted: bool,
    pub backend_id: Option<String>,
    pub model_name: Option<String>,
    pub persona_id: Option<String>,
    pub persona_version_id: Option<String>,
    pub persona_name_snapshot: Option<String>,
    pub think_mode: Option<String>,
    pub done_reason: Option<String>,
    pub error_text: Option<String>,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
    pub active_revision: Option<ChatMessageRevision>,
    pub revisions: Vec<ChatMessageRevision>,
    pub revision_count: i64,
    pub attachments: Vec<Attachment>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessageRevision {
    pub id: String,
    pub content_text: String,
    pub thinking_text: String,
    pub source: String,
    pub created_at: i64,
}
