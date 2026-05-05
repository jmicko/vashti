use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct OllamaModel {
    pub name: String,
    pub supports_images: bool,
    pub supports_thinking: bool,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
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

#[derive(Debug, Serialize)]
pub struct ShowModelRequest<'a> {
    pub model: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct ShowModelResponse {
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl TagsResponse {
    pub fn into_models(self) -> Vec<OllamaModel> {
        self.models
            .into_iter()
            .filter_map(|model| {
                let name = model.name.or(model.model)?;
                let supports_images = supports_images(&name, model.details.as_ref());
                let supports_thinking = supports_thinking(&name, model.details.as_ref());
                let capabilities = fallback_capabilities(supports_images, supports_thinking);

                Some(OllamaModel {
                    name,
                    supports_images,
                    supports_thinking,
                    capabilities,
                })
            })
            .collect()
    }
}

impl OllamaModel {
    pub fn apply_capabilities(&mut self, capabilities: Vec<String>) {
        if capabilities.is_empty() {
            return;
        }

        self.supports_images = has_capability(&capabilities, "vision");
        self.supports_thinking = has_capability(&capabilities, "thinking");
        self.capabilities = normalized_capabilities(capabilities);
    }
}

fn has_capability(capabilities: &[String], expected: &str) -> bool {
    capabilities
        .iter()
        .any(|capability| capability.eq_ignore_ascii_case(expected))
}

fn normalized_capabilities(capabilities: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::new();
    for capability in capabilities {
        let capability = capability.trim().to_ascii_lowercase();
        if !capability.is_empty() && !normalized.contains(&capability) {
            normalized.push(capability);
        }
    }

    normalized
}

fn fallback_capabilities(supports_images: bool, supports_thinking: bool) -> Vec<String> {
    let mut capabilities = vec!["completion".to_string()];
    if supports_images {
        capabilities.push("vision".to_string());
    }
    if supports_thinking {
        capabilities.push("thinking".to_string());
    }

    capabilities
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

fn supports_thinking(name: &str, details: Option<&ModelDetails>) -> bool {
    let name = name.to_ascii_lowercase();
    if ["qwen3", "gpt-oss", "deepseek-r1", "deepseek-v3.1"]
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
        .any(|family| {
            ["qwen3", "gpt-oss", "deepseek-r1", "deepseek-v3.1"]
                .iter()
                .any(|needle| family.contains(needle))
        })
}
