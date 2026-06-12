use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct PersonaAvatarAsset {
    pub id: String,
    pub original_filename: String,
    pub mime_type: String,
    pub size_bytes: i64,
    pub created_at: i64,
}
