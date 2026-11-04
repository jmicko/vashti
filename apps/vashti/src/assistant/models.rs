use serde::{Deserialize, Serialize};

use crate::ollama::models::OllamaChatMessage;

pub const PREFERRED_MODEL: &str = "kuno-qwen3.8-flash-next:iq3_s";
pub const SANDBOX_TOOL_ID: &str = "assistant_sandbox";

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionPolicy {
    Block,
    #[default]
    Ask,
    Allow,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TaskConfig {
    pub backend_id: String,
    pub model_name: String,
    pub instructions: String,
    pub notes: bool,
    pub memories: bool,
    pub chat_history: bool,
    pub writes: ActionPolicy,
    pub web: ActionPolicy,
    pub sandbox: bool,
    pub reasoning: bool,
    pub max_steps: usize,
    pub max_output_tokens: i64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            backend_id: String::new(),
            model_name: PREFERRED_MODEL.into(),
            instructions: String::new(),
            notes: true,
            memories: true,
            chat_history: false,
            writes: ActionPolicy::Ask,
            web: ActionPolicy::Ask,
            sandbox: false,
            reasoning: true,
            max_steps: 24,
            max_output_tokens: 24_576,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub instructions: String,
    pub config: TaskConfig,
    pub interval_seconds: Option<i64>,
    pub next_run_at: Option<i64>,
    pub schedule_enabled: bool,
    pub schedule_error: Option<String>,
    pub archived_at: Option<i64>,
    pub last_run_id: Option<String>,
    pub last_status: Option<String>,
    pub needs_attention: bool,
    pub created_at: i64,
    pub updated_at: i64,
    pub revision: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, sqlx::FromRow)]
pub struct Run {
    pub id: String,
    pub task_id: String,
    pub status: String,
    pub instructions: String,
    pub progress_text: String,
    pub progress_thinking: String,
    pub result_text: String,
    pub error_text: String,
    pub steps: i64,
    pub output_tokens: i64,
    pub scheduled: bool,
    pub created_at: i64,
    pub updated_at: i64,
    pub finished_at: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, sqlx::FromRow)]
pub struct Action {
    pub id: String,
    pub run_id: String,
    pub ordinal: i64,
    pub tool_name: String,
    pub arguments_json: String,
    pub summary: String,
    pub status: String,
    pub result_text: Option<String>,
    pub requested_at: i64,
    pub expires_at: i64,
    pub decided_at: Option<i64>,
    pub finished_at: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, sqlx::FromRow)]
pub struct Event {
    pub id: i64,
    pub kind: String,
    pub text: String,
    pub data_json: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, sqlx::FromRow)]
pub struct Artifact {
    pub id: String,
    pub run_id: String,
    pub name: String,
    pub mime_type: String,
    pub size: i64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Checkpoint {
    pub messages: Vec<OllamaChatMessage>,
    pub pending: Vec<String>,
    pub steps: usize,
    pub output_tokens: i64,
    pub sandbox_image: Option<String>,
    pub context_window: i64,
    pub supports_thinking: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaveTask {
    pub request_id: Option<String>,
    pub title: String,
    pub instructions: String,
    pub config: TaskConfig,
    pub interval_seconds: Option<i64>,
    pub next_run_at: Option<i64>,
    #[serde(default)]
    pub schedule_enabled: bool,
    pub expected_revision: Option<i64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StartRun {
    pub request_id: Option<String>,
    pub message: String,
    pub acknowledge_uncertain: bool,
}

pub fn active(status: &str) -> bool {
    matches!(
        status,
        "queued" | "running" | "waiting_approval" | "waiting_input" | "paused"
    )
}
