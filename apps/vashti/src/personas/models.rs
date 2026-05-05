use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct PersonaResponse {
    pub id: String,
    pub owner_user_id: Option<String>,
    pub owner_username: Option<String>,
    pub visibility: String,
    pub lifecycle_state: String,
    pub current_version: PersonaVersionResponse,
    pub is_owner: bool,
    pub is_member: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize)]
pub struct PersonaVersionResponse {
    pub id: String,
    pub persona_id: String,
    pub version_number: i64,
    pub display_name: String,
    pub avatar_attachment_id: Option<String>,
    pub base_backend_id: String,
    pub base_model_name: String,
    pub system_prompt: String,
    pub tool_policy_json: Option<String>,
    pub created_by_user_id: Option<String>,
    pub created_at: i64,
}
