use kuno_protocol::{MINIMAX_H3_I2V_RECIPE, VideoRecipe};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Draft {
    pub backend_id: String,
    pub workflow: String,
    pub prompt: String,
    pub seed: String,
    pub seed_mode: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub audio: bool,
    pub first_frame_id: Option<String>,
}
impl Default for Draft {
    fn default() -> Self {
        Self {
            backend_id: String::new(),
            workflow: MINIMAX_H3_I2V_RECIPE.into(),
            prompt: String::new(),
            seed: "0".into(),
            seed_mode: "increment".into(),
            width: 832,
            height: 544,
            frames: 56,
            audio: true,
            first_frame_id: None,
        }
    }
}
impl Draft {
    pub fn recipe(&self, first_frame: String) -> VideoRecipe {
        VideoRecipe {
            workflow: self.workflow.clone(),
            prompt: if self.prompt.trim().is_empty() {
                "Animate this image with gentle natural motion.".into()
            } else {
                self.prompt.clone()
            },
            seed: self.seed.clone(),
            width: self.width,
            height: self.height,
            frames: self.frames,
            audio: self.audio,
            first_frame,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: String,
    pub draft: Draft,
    pub retry_of: Option<String>,
}
#[derive(Serialize, sqlx::FromRow)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub draft_json: String,
    pub revision: i64,
    pub updated_at: i64,
}
#[derive(Clone, Serialize, sqlx::FromRow)]
pub struct Run {
    pub id: String,
    pub session_id: String,
    pub group_id: String,
    pub draft_json: String,
    pub status: String,
    pub error: Option<String>,
    pub size_bytes: i64,
    pub duration_ms: i64,
    pub created_at: i64,
}
