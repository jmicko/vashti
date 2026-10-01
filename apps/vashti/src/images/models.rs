use kuno_protocol::{ImageAttention, ImageRecipe, QWEN21_EDIT_RECIPE, QWEN21_RECIPE};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub recipe: ImageRecipe,
    #[serde(default)]
    pub model_may_change: BTreeSet<String>,
    #[serde(default)]
    pub seed_mode: SeedMode,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preferences {
    pub generate: Profile,
    pub edit: Profile,
    #[serde(default)]
    pub hidden_controls: BTreeSet<String>,
}

impl Default for Preferences {
    fn default() -> Self {
        let recipe = ImageRecipe {
            workflow: QWEN21_RECIPE.into(),
            prompt: String::new(),
            negative_prompt: String::new(),
            seed: "0".into(),
            width: 1024,
            height: 1024,
            steps: 50,
            cfg: 1.,
            attention: ImageAttention::Standard,
            flow_shift: None,
            reference_images: vec![],
        };
        let mut edit = recipe.clone();
        edit.workflow = QWEN21_EDIT_RECIPE.into();
        edit.steps = 25;
        edit.cfg = 4.;
        Self {
            generate: Profile {
                recipe,
                seed_mode: SeedMode::Increment,
                model_may_change: BTreeSet::from(["size".into()]),
            },
            edit: Profile {
                recipe: edit,
                seed_mode: SeedMode::Increment,
                model_may_change: BTreeSet::new(),
            },
            hidden_controls: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SeedMode {
    Random,
    #[default]
    Increment,
    Fixed,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptHelp {
    pub backend_id: String,
    pub model_name: String,
    #[serde(default)]
    pub guidance: String,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    #[serde(default)]
    pub backend_id: Option<String>,
    pub recipe: ImageRecipe,
    #[serde(default)]
    pub input_ids: Vec<String>,
    #[serde(default)]
    pub original_input_ids: Vec<String>,
    pub parent_result_id: Option<String>,
    pub prompt_version_id: Option<String>,
    pub seed_mode: SeedMode,
    pub count: Option<u32>,
    #[serde(default)]
    pub auto_improve: bool,
    pub prompt_help: Option<PromptHelp>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    #[serde(skip)]
    pub chat_target: Option<ChatTarget>,
    pub client_request_id: String,
    pub draft: Draft,
    #[serde(default)]
    pub improve_only: bool,
    pub retry_result_id: Option<String>,
    #[serde(default)]
    pub fixed_endless_confirmed: bool,
}

#[derive(Serialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub mode: String,
    pub draft: Draft,
    pub revision: i64,
    pub updated_at: i64,
    pub active_result_id: Option<String>,
}

#[derive(Serialize)]
pub struct Run {
    pub preview: Option<kuno_image::preview::Preview>,
    pub id: String,
    pub session_id: String,
    pub status: String,
    pub completed_count: i64,
    pub effective_prompt: Option<String>,
    pub prompt_version_id: Option<String>,
    pub error: Option<String>,
    pub request: RunRequest,
    pub created_at: i64,
}

#[derive(Serialize)]
pub struct ImageResult {
    pub origin_run_id: String,
    pub id: String,
    pub run_id: String,
    pub session_id: String,
    pub group_id: String,
    pub asset_id: String,
    pub parent_result_id: Option<String>,
    pub recipe: ImageRecipe,
    pub input_ids: Vec<String>,
    pub duration_ms: i64,
    pub created_at: i64,
    pub selected: bool,
}

#[derive(Serialize)]
pub struct PromptVersion {
    pub id: String,
    pub parent_id: Option<String>,
    pub prompt: String,
    pub guidance: String,
    pub source: String,
    pub model_name: Option<String>,
    pub created_at: i64,
}

#[derive(Serialize)]
pub struct GalleryItem {
    pub id: String,
    pub asset_id: String,
    pub group_id: Option<String>,
    pub session_id: Option<String>,
    pub variants: i64,
    pub created_at: i64,
}

#[derive(Clone)]
pub struct ChatTarget {
    pub chat_id: String,
    pub message_id: String,
    pub revision_id: String,
}
