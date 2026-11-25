use kuno_protocol::{MINIMAX_H3_VIDEO_RECIPE, VideoRecipe};
use serde::{Deserialize, Serialize};

pub const MIN_SIDE: u32 = 256;
pub const MAX_SIDE: u32 = 1280;
pub const STANDARD_PIXELS: u64 = 832 * 544;
pub const HIGH_PIXELS: u64 = 1280 * 704;
pub const MIN_FRAMES: u32 = 22;
pub const STANDARD_MAX_FRAMES: u32 = 243;
pub const HIGH_MAX_FRAMES: u32 = 124;

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
    pub last_frame_id: Option<String>,
    pub reference_image_ids: Vec<String>,
    pub input_mode: Option<String>,
}
impl Default for Draft {
    fn default() -> Self {
        Self {
            backend_id: String::new(),
            workflow: MINIMAX_H3_VIDEO_RECIPE.into(),
            prompt: String::new(),
            seed: "0".into(),
            seed_mode: "increment".into(),
            width: 832,
            height: 544,
            frames: 56,
            audio: true,
            first_frame_id: None,
            last_frame_id: None,
            reference_image_ids: Vec::new(),
            input_mode: None,
        }
    }
}
impl Draft {
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub fn within_clip_limits(&self) -> bool {
        let pixels = self.pixels();
        let max_frames = if pixels > STANDARD_PIXELS {
            HIGH_MAX_FRAMES
        } else {
            STANDARD_MAX_FRAMES
        };
        (MIN_FRAMES..=max_frames).contains(&self.frames)
            && pixels <= HIGH_PIXELS
            && [self.width, self.height]
                .iter()
                .all(|side| (MIN_SIDE..=MAX_SIDE).contains(side))
    }

    pub fn image_ids(&self) -> impl Iterator<Item = &String> {
        self.first_frame_id
            .iter()
            .chain(self.last_frame_id.iter())
            .chain(self.reference_image_ids.iter())
    }

    pub fn recipe(&self, inputs: Inputs) -> VideoRecipe {
        VideoRecipe {
            workflow: self.workflow.clone(),
            prompt: if self.prompt.trim().is_empty()
                && (inputs.first_frame.is_some()
                    || inputs.last_frame.is_some()
                    || !inputs.reference_images.is_empty())
            {
                "Animate this image with gentle natural motion.".into()
            } else {
                self.prompt.clone()
            },
            seed: self.seed.clone(),
            width: self.width,
            height: self.height,
            frames: self.frames,
            audio: self.audio,
            first_frame: inputs.first_frame,
            last_frame: inputs.last_frame,
            reference_images: inputs.reference_images,
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Inputs {
    pub first_frame: Option<String>,
    pub last_frame: Option<String>,
    pub reference_images: Vec<String>,
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
