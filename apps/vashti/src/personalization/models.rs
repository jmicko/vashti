use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub enabled: bool,
    pub processor_backend_id: Option<String>,
    pub processor_model: Option<String>,
    pub allow_cloud_processor: bool,
    pub learn_from_imports: bool,
    pub allow_local: bool,
    pub allow_cloud: bool,
    pub allow_characters: bool,
    pub model_overrides: BTreeMap<String, bool>,
    pub idle_minutes: i64,
    pub context_chars: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: false,
            processor_backend_id: None,
            processor_model: None,
            allow_cloud_processor: false,
            learn_from_imports: true,
            allow_local: true,
            allow_cloud: false,
            allow_characters: false,
            model_overrides: BTreeMap::new(),
            idle_minutes: 5,
            context_chars: 6000,
        }
    }
}

impl Settings {
    pub fn allows(&self, model: &Model) -> bool {
        self.enabled
            && self.model_overrides.get(&model.key).copied().unwrap_or({
                (!model.character || self.allow_characters)
                    && if model.cloud {
                        self.allow_cloud
                    } else {
                        self.allow_local
                    }
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub key: String,
    pub name: String,
    pub backend_id: String,
    pub model_name: String,
    pub cloud: bool,
    pub character: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub message_id: String,
    pub revision_id: String,
    pub chat_id: String,
    pub model_key: String,
    pub quote: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Fact {
    pub id: String,
    pub topic: String,
    pub content: String,
    pub locked: bool,
    pub version: i64,
    pub sources_json: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize)]
pub struct Profile {
    pub settings: Settings,
    pub revision: i64,
    pub facts: Vec<Fact>,
    pub last_attempt_at: Option<i64>,
    pub last_processed_at: Option<i64>,
    pub last_error: Option<String>,
    pub requested: bool,
    pub rebuild_id: Option<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RebuildPhase {
    Queued,
    Replaying,
    WaitingForIdle,
    WaitingForGeneration,
    Paused,
    NeedsAttention,
}

#[derive(Debug, Serialize)]
pub struct RebuildProgress {
    pub id: String,
    pub processed: i64,
    pub phase: RebuildPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub source: usize,
    pub quote: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub topic: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaces_topic: Option<String>,
    pub content: String,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extraction {
    pub facts: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSource {
    pub message_id: String,
    pub revision_id: String,
    pub chat_id: String,
    pub model_key: String,
    pub text: String,
    pub created_at: i64,
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayHints {
    pub facts: Vec<FactHint>,
    pub earlier_turns: Vec<BranchHint>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactHint {
    pub topic: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchHint {
    pub source: usize,
    pub excerpts: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveSettings {
    pub expected_revision: i64,
    pub settings: Settings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditFact {
    pub expected_revision: i64,
    pub topic: String,
    pub content: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedRevision {
    pub expected_revision: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceExtract {
    pub expected_user_id: String,
    pub backend_id: String,
    pub model_name: String,
    pub allow_cloud_processor: bool,
    pub sources: Vec<InputSource>,
    pub topics: Vec<String>,
    #[serde(default)]
    pub replay: ReplayHints,
}
