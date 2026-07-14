use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ContextCategoryResponse {
    pub id: String,
    pub name: String,
    pub selection_mode: String,
    pub sort_order: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextBlockVersionResponse {
    pub id: String,
    pub block_id: String,
    pub version_number: i64,
    pub name: String,
    pub content: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextBlockResponse {
    pub id: String,
    pub category_id: Option<String>,
    pub sort_order: i64,
    pub current_version: ContextBlockVersionResponse,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextBlockSelection {
    pub block_id: String,
    pub block_version_id: String,
    pub category_id: Option<String>,
    pub category_name: Option<String>,
    pub category_selection_mode: Option<String>,
    pub version_number: i64,
    pub name: String,
    pub content: String,
    pub position: i64,
}
