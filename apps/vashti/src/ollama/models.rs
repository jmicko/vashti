use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaChatOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<OllamaThink>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl OllamaChatOptions {
    pub fn has_any(&self) -> bool {
        self.temperature.is_some()
            || self.top_p.is_some()
            || self.repeat_penalty.is_some()
            || self.num_ctx.is_some()
            || self.num_predict.is_some()
            || self.seed.is_some()
    }
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
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i64>,
    pub prompt_eval_duration: Option<i64>,
    pub eval_count: Option<i64>,
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub message: Option<OllamaChatChunkMessage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaUsageStats {
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i64>,
    pub prompt_eval_duration: Option<i64>,
    pub eval_count: Option<i64>,
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatChunkMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: String,
    #[serde(default)]
    pub tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaToolCall {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: OllamaToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OllamaToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
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

impl OllamaChatChunk {
    pub fn usage_stats(&self) -> Option<OllamaUsageStats> {
        let stats = OllamaUsageStats {
            total_duration: self.total_duration,
            load_duration: self.load_duration,
            prompt_eval_count: self.prompt_eval_count,
            prompt_eval_duration: self.prompt_eval_duration,
            eval_count: self.eval_count,
            eval_duration: self.eval_duration,
        };

        stats.has_any().then_some(stats)
    }
}

impl OllamaUsageStats {
    pub fn has_any(&self) -> bool {
        self.total_duration.is_some()
            || self.load_duration.is_some()
            || self.prompt_eval_count.is_some()
            || self.prompt_eval_duration.is_some()
            || self.eval_count.is_some()
            || self.eval_duration.is_some()
    }

    pub fn add_assign(&mut self, other: OllamaUsageStats) {
        self.total_duration = sum_optional(self.total_duration, other.total_duration);
        self.load_duration = sum_optional(self.load_duration, other.load_duration);
        self.prompt_eval_count = sum_optional(self.prompt_eval_count, other.prompt_eval_count);
        self.prompt_eval_duration =
            sum_optional(self.prompt_eval_duration, other.prompt_eval_duration);
        self.eval_count = sum_optional(self.eval_count, other.eval_count);
        self.eval_duration = sum_optional(self.eval_duration, other.eval_duration);
    }
}

fn sum_optional(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
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
