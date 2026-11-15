//! Video requests are transient content, like image requests. Never log or persist them.
use serde::{Deserialize, Serialize};

/// First frame required. Kept unchanged for existing clients.
pub const MINIMAX_H3_I2V_RECIPE: &str = "minimax-h3-image-to-video/v1";
/// Text only, first frame, last frame, or both. Same model as image-to-video.
pub const MINIMAX_H3_VIDEO_RECIPE: &str = "minimax-h3-video/v1";
/// One to four reference images named in the prompt as <Picture 1>, <Picture 2>, ...
pub const MINIMAX_H3_REFERENCE_RECIPE: &str = "minimax-h3-reference-video/v1";
/// First/last-frame model (image-to-video and video workflows).
pub const MINIMAX_H3_COLLECTION: &str = "minimax-h3-int8";
/// Reference model (reference-video workflow).
pub const MINIMAX_H3_REFERENCE_COLLECTION: &str = "minimax-h3-ref2va-int8";
pub const VIDEO_MAX_REFERENCE_IMAGES: usize = 4;
/// Sampling steps of the merged turbo LoRAs.
pub const VIDEO_KEYFRAME_STEPS: u32 = 8;
pub const VIDEO_REFERENCE_STEPS: u32 = 4;
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
/// Per image; the whole request is also bounded by MAX_IMAGE_REQUEST_BYTES.
const MAX_FRAME_BASE64: usize = 16 * 1024 * 1024;

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
    /// Images are PNG/JPEG/WebP bytes encoded as base64, never a URL or filesystem path.
    /// Required for image-to-video, optional for video, absent for reference video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_frame: Option<String>,
    /// Optional for video; absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_frame: Option<String>,
    /// One to four for reference video; absent otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_images: Vec<String>,
}

impl VideoRecipe {
    /// Whether this request runs on the reference model instead of the first/last-frame model.
    pub fn uses_reference_model(&self) -> bool {
        self.workflow == MINIMAX_H3_REFERENCE_RECIPE
    }
    pub fn steps(&self) -> u32 {
        if self.uses_reference_model() {
            VIDEO_REFERENCE_STEPS
        } else {
            VIDEO_KEYFRAME_STEPS
        }
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        let (first, last, references) = match self.workflow.as_str() {
            MINIMAX_H3_I2V_RECIPE => (Need::Required, Need::Absent, false),
            MINIMAX_H3_VIDEO_RECIPE => (Need::Optional, Need::Optional, false),
            MINIMAX_H3_REFERENCE_RECIPE => (Need::Absent, Need::Absent, true),
            _ => return Err("unsupported_video_workflow"),
        };
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
        first.check(&self.first_frame, "video_first_frame_invalid")?;
        last.check(&self.last_frame, "video_last_frame_invalid")?;
        let count = self.reference_images.len();
        if references != (count > 0)
            || count > VIDEO_MAX_REFERENCE_IMAGES
            || self.reference_images.iter().any(|i| !frame_size(i))
        {
            return Err("video_reference_images_invalid");
        }
        Ok(())
    }
}

fn frame_size(image: &str) -> bool {
    !image.is_empty() && image.len() <= MAX_FRAME_BASE64
}

#[derive(Clone, Copy)]
enum Need {
    Required,
    Optional,
    Absent,
}
impl Need {
    fn check(self, image: &Option<String>, code: &'static str) -> Result<(), &'static str> {
        match (self, image) {
            (Need::Required, None) | (Need::Absent, Some(_)) => Err(code),
            (_, Some(i)) if !frame_size(i) => Err(code),
            _ => Ok(()),
        }
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
            first_frame: Some("AAAA".into()),
            last_frame: None,
            reference_images: Vec::new(),
        }
    }
    #[test]
    fn accepts_the_tested_recipe_and_rejects_out_of_range_settings() {
        recipe().validate().unwrap();
        type Case = (fn(&mut VideoRecipe), &'static str);
        let cases: [Case; 10] = [
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
            (|r| r.first_frame = None, "video_first_frame_invalid"),
            (
                |r| r.first_frame = Some(String::new()),
                "video_first_frame_invalid",
            ),
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
    fn each_workflow_accepts_only_its_inputs() {
        let ok = |r: &VideoRecipe| r.validate();
        // image-to-video: first frame only.
        let mut r = recipe();
        r.last_frame = Some("AAAA".into());
        assert_eq!(ok(&r), Err("video_last_frame_invalid"));
        let mut r = recipe();
        r.reference_images = vec!["AAAA".into()];
        assert_eq!(ok(&r), Err("video_reference_images_invalid"));
        // video: text only, first, last, or both; never references.
        for (first, last) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut r = recipe();
            r.workflow = MINIMAX_H3_VIDEO_RECIPE.into();
            r.first_frame = first.then(|| "AAAA".into());
            r.last_frame = last.then(|| "AAAA".into());
            ok(&r).unwrap();
            assert_eq!(r.steps(), VIDEO_KEYFRAME_STEPS);
            r.reference_images = vec!["AAAA".into()];
            assert_eq!(ok(&r), Err("video_reference_images_invalid"));
        }
        let mut r = recipe();
        r.workflow = MINIMAX_H3_VIDEO_RECIPE.into();
        r.last_frame = Some(String::new());
        assert_eq!(ok(&r), Err("video_last_frame_invalid"));
        // reference video: one to four references, no keyframes.
        let mut r = recipe();
        r.workflow = MINIMAX_H3_REFERENCE_RECIPE.into();
        assert_eq!(ok(&r), Err("video_first_frame_invalid"));
        r.first_frame = None;
        assert_eq!(ok(&r), Err("video_reference_images_invalid"));
        for n in 1..=VIDEO_MAX_REFERENCE_IMAGES {
            r.reference_images = vec!["AAAA".into(); n];
            ok(&r).unwrap();
        }
        assert!(r.uses_reference_model());
        assert_eq!(r.steps(), VIDEO_REFERENCE_STEPS);
        r.last_frame = Some("AAAA".into());
        assert_eq!(ok(&r), Err("video_last_frame_invalid"));
        r.last_frame = None;
        r.reference_images = vec!["AAAA".into(); VIDEO_MAX_REFERENCE_IMAGES + 1];
        assert_eq!(ok(&r), Err("video_reference_images_invalid"));
        r.reference_images = vec![String::new()];
        assert_eq!(ok(&r), Err("video_reference_images_invalid"));
    }
    #[test]
    fn optional_inputs_are_omitted_from_json() {
        let mut r = recipe();
        r.workflow = MINIMAX_H3_VIDEO_RECIPE.into();
        r.first_frame = None;
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            !json.contains("first_frame")
                && !json.contains("last_frame")
                && !json.contains("reference_images")
        );
        let back: VideoRecipe = serde_json::from_str(&json).unwrap();
        back.validate().unwrap();
    }
    #[test]
    fn frame_counts_follow_the_model_rule() {
        assert!(VIDEO_FRAMES.iter().all(|f| f % 17 == 5));
    }
}
