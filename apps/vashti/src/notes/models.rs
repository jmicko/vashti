use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize)]
pub struct NoteContextSelection {
    pub note_id: String,
    pub note_version_id: String,
    pub version_number: i64,
    pub title: String,
    pub source: String,
    pub position: i64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NoteAiAccess {
    None,
    Read,
    Edit,
    Manage,
}

impl NoteAiAccess {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Read => "read",
            Self::Edit => "edit",
            Self::Manage => "manage",
        }
    }
}

impl TryFrom<&str> for NoteAiAccess {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "none" => Ok(Self::None),
            "read" => Ok(Self::Read),
            "edit" => Ok(Self::Edit),
            "manage" => Ok(Self::Manage),
            _ => Err(format!("invalid note AI access level: {value}")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NoteModelScope {
    pub all_models: bool,
    #[serde(default)]
    pub model_keys: Vec<String>,
}

impl Default for NoteModelScope {
    fn default() -> Self {
        Self {
            all_models: true,
            model_keys: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NoteVersionResponse {
    pub id: String,
    pub note_id: String,
    pub version_number: i64,
    pub title: String,
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
pub struct NoteSummaryResponse {
    pub id: String,
    pub title: String,
    pub excerpt: String,
    pub current_version_id: String,
    pub current_version_number: i64,
    pub tags: Vec<String>,
    pub is_pinned: bool,
    pub ai_access: NoteAiAccess,
    pub model_scope: NoteModelScope,
    pub deleted_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NoteResponse {
    pub id: String,
    pub current_version: NoteVersionResponse,
    pub tags: Vec<String>,
    pub is_pinned: bool,
    pub ai_access: NoteAiAccess,
    pub model_scope: NoteModelScope,
    pub deleted_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CreateNoteRequest {
    pub title: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub is_pinned: bool,
    pub ai_access: Option<NoteAiAccess>,
    pub model_scope: Option<NoteModelScope>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UpdateNoteRequest {
    pub expected_version: i64,
    pub title: Option<String>,
    pub content: Option<String>,
    pub tags: Option<Vec<String>>,
    pub is_pinned: Option<bool>,
    pub ai_access: Option<NoteAiAccess>,
    pub model_scope: Option<NoteModelScope>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExpectedNoteVersionRequest {
    pub expected_version: i64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UpdateNoteSettingsRequest {
    pub allow_model_read: bool,
    pub allow_model_create: bool,
    pub allow_model_edit: bool,
    pub allow_model_trash: bool,
    pub default_ai_access: NoteAiAccess,
    pub default_model_scope: NoteModelScope,
}

#[derive(Clone, Debug, Serialize)]
pub struct NoteSettingsResponse {
    pub allow_model_read: bool,
    pub allow_model_create: bool,
    pub allow_model_edit: bool,
    pub allow_model_trash: bool,
    pub default_ai_access: NoteAiAccess,
    pub default_model_scope: NoteModelScope,
}
