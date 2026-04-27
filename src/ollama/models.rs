use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct OllamaModel {
    pub name: String,
    pub supports_images: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<OllamaThink>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum OllamaThink {
    Bool(bool),
    Level(String),
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatChunk {
    pub message: Option<OllamaChatChunkMessage>,
    #[serde(default)]
    pub done: bool,
    pub done_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub message: Option<OllamaChatChunkMessage>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatChunkMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: String,
}

#[derive(Debug, Deserialize)]
pub struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: Option<String>,
    model: Option<String>,
    details: Option<ModelDetails>,
}

#[derive(Debug, Deserialize)]
struct ModelDetails {
    family: Option<String>,
    families: Option<Vec<String>>,
}

impl TagsResponse {
    pub fn into_models(self) -> Vec<OllamaModel> {
        self.models
            .into_iter()
            .filter_map(|model| {
                let name = model.name.or(model.model)?;
                let supports_images = supports_images(&name, model.details.as_ref());

                Some(OllamaModel {
                    name,
                    supports_images,
                })
            })
            .collect()
    }
}

fn supports_images(name: &str, details: Option<&ModelDetails>) -> bool {
    let name = name.to_ascii_lowercase();
    if ["llava", "bakllava", "moondream", "minicpm-v"]
        .iter()
        .any(|needle| name.contains(needle))
    {
        return true;
    }

    let Some(details) = details else {
        return false;
    };

    details
        .family
        .iter()
        .chain(details.families.as_deref().unwrap_or_default().iter())
        .map(|family| family.to_ascii_lowercase())
        .any(|family| family.contains("vision") || family.contains("clip"))
}
