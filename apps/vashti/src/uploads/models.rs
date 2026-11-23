use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Attachment {
    pub id: String,
    pub chat_id: String,
    pub message_id: Option<String>,
    pub revision_id: Option<String>,
    pub original_filename: String,
    pub mime_type: String,
    pub size_bytes: i64,
    pub attachment_kind: String,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdf_has_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdf_truncated: Option<bool>,
}

pub struct PdfPromptContext {
    pub remaining: usize,
    pub question: Option<String>,
}

impl PdfPromptContext {
    pub fn new(remaining: usize) -> Self {
        Self { remaining, question: None }
    }
}
