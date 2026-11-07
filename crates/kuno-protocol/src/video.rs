//! Video requests are transient content, like image requests. Never log or persist them.
use serde::{Deserialize, Serialize};

pub const MINIMAX_H3_I2V_RECIPE: &str = "minimax-h3-image-to-video/v1";
pub const MINIMAX_H3_COLLECTION: &str = "minimax-h3-int8";
pub const VIDEO_FPS: u32 = 24;
/// MiniMax H3 needs 17k + 5 frames. These are the counts tested on the B70.
pub const VIDEO_FRAMES: [u32; 7] = [22, 39, 56, 73, 90, 107, 124];
pub const VIDEO_MAX_PIXELS: u32 = 832 * 544;
pub const VIDEO_DIMENSION_MULTIPLE: u32 = 32;
pub const VIDEO_MIN_DIMENSION: u32 = 256;
pub const VIDEO_MAX_DIMENSION: u32 = 1280;
pub const VIDEO_MIME_TYPE: &str = "video/mp4";
/// Largest encoded MP4 Kuno returns, before base64.
pub const MAX_VIDEO_BYTES: usize = 64 * 1024 * 1024;
const MAX_FIRST_FRAME_BASE64: usize = 16 * 1024 * 1024;

fn default_audio() -> bool {
    true
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoRecipe {
    pub workflow: String,
    pub prompt: String,
    /// Decimal string avoids losing 64-bit seeds in JavaScript clients.
    pub seed: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    /// The model always generates audio; false only omits the track.
    #[serde(default = "default_audio")]
    pub audio: bool,
    /// PNG/JPEG/WebP bytes encoded as base64. Never a URL or filesystem path.
    pub first_frame: String,
}

impl VideoRecipe {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.workflow != MINIMAX_H3_I2V_RECIPE {
            return Err("unsupported_video_workflow");
        }
        if self.prompt.is_empty() || self.prompt.len() > 32768 || self.prompt.contains('\0') {
            return Err("video_prompt_invalid");
        }
        if self.seed.is_empty()
            || self.seed.len() > 20
            || !self.seed.bytes().all(|b| b.is_ascii_digit())
            || self.seed.parse::<i64>().is_err()
        {
            return Err("image_seed_invalid");
        }
        let dimension = |d: u32| {
            (VIDEO_MIN_DIMENSION..=VIDEO_MAX_DIMENSION).contains(&d)
                && d.is_multiple_of(VIDEO_DIMENSION_MULTIPLE)
        };
        if !dimension(self.width)
            || !dimension(self.height)
            || u64::from(self.width) * u64::from(self.height) > u64::from(VIDEO_MAX_PIXELS)
        {
            return Err("video_dimensions_unsupported");
        }
        if !VIDEO_FRAMES.contains(&self.frames) {
            return Err("video_frames_unsupported");
        }
        if self.first_frame.is_empty() || self.first_frame.len() > MAX_FIRST_FRAME_BASE64 {
            return Err("video_first_frame_invalid");
        }
        Ok(())
    }
}

/// Content-free progress phases, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoPhase {
    Conditioning,
    Sampling,
    Decoding,
    Encoding,
}

/// Transient HTTP response content. Never log or persist these events in Kuno.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoStreamEvent {
    Started,
    Progress {
        phase: VideoPhase,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        steps: Option<u32>,
    },
    Preview {
        step: u32,
        steps: u32,
        width: u32,
        height: u32,
        png_base64: String,
    },
    Result {
        mime_type: String,
        width: u32,
        height: u32,
        frames: u32,
        fps: u32,
        audio: bool,
        video_base64: String,
    },
    Error {
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    fn recipe() -> VideoRecipe {
        VideoRecipe {
            workflow: MINIMAX_H3_I2V_RECIPE.into(),
            prompt: "synthetic".into(),
            seed: "42".into(),
            width: 832,
            height: 544,
            frames: 124,
            audio: true,
            first_frame: "AAAA".into(),
        }
    }
    #[test]
    fn accepts_the_tested_recipe_and_rejects_out_of_range_settings() {
        recipe().validate().unwrap();
        type Case = (fn(&mut VideoRecipe), &'static str);
        let cases: [Case; 9] = [
            (
                |r| r.workflow = "other".into(),
                "unsupported_video_workflow",
            ),
            (|r| r.prompt.clear(), "video_prompt_invalid"),
            (|r| r.prompt = "a\0b".into(), "video_prompt_invalid"),
            (
                |r| r.seed = "9223372036854775808".into(),
                "image_seed_invalid",
            ),
            (|r| r.width = 840, "video_dimensions_unsupported"),
            (|r| r.width = 1280, "video_dimensions_unsupported"),
            (|r| r.frames = 57, "video_frames_unsupported"),
            (|r| r.frames = 141, "video_frames_unsupported"),
            (|r| r.first_frame.clear(), "video_first_frame_invalid"),
        ];
        for (change, code) in cases {
            let mut r = recipe();
            change(&mut r);
            assert_eq!(r.validate(), Err(code));
        }
        let mut square = recipe();
        (square.width, square.height) = (672, 672);
        square.validate().unwrap();
    }
    #[test]
    fn audio_defaults_on_and_unknown_fields_are_rejected() {
        let json = r#"{"workflow":"minimax-h3-image-to-video/v1","prompt":"p","seed":"1","width":832,"height":544,"frames":56,"first_frame":"AAAA"}"#;
        let r: VideoRecipe = serde_json::from_str(json).unwrap();
        assert!(r.audio);
        let extra = json.replace("}", r#","negative_prompt":"x"}"#);
        assert!(serde_json::from_str::<VideoRecipe>(&extra).is_err());
    }
    #[test]
    fn frame_counts_follow_the_model_rule() {
        assert!(VIDEO_FRAMES.iter().all(|f| f % 17 == 5));
    }
}
