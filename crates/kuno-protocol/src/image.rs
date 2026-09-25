//! Image requests are transient content, unlike resource/diagnostic messages.
use serde::{Deserialize, Serialize};
pub const QWEN21_RECIPE: &str = "qwen-image-2.1-text-to-image/v1";
pub const QWEN21_EDIT_RECIPE: &str = "qwen-image-2.1-edit/v1";
pub const MAX_IMAGE_REQUEST_BYTES: usize = 64 * 1024 * 1024;
pub const QWEN_RECIPE: &str = "qwen-image-2512-text-to-image/v1";
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageAttention {
    #[default]
    Standard,
    Fast,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageRecipe {
    pub workflow: String,
    pub prompt: String,
    pub negative_prompt: String,
    /// Decimal string avoids losing 64-bit seeds in JavaScript clients.
    pub seed: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f64,
    /// Fast attention is opt-in and may change generated pixels.
    #[serde(default)]
    pub attention: ImageAttention,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow_shift: Option<f64>,
    /// Ordered PNG/JPEG/WebP bytes encoded as base64. Never a URL or filesystem path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reference_images: Vec<String>,
}
impl ImageRecipe {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !matches!(
            self.workflow.as_str(),
            QWEN_RECIPE | QWEN21_RECIPE | QWEN21_EDIT_RECIPE
        ) {
            return Err("unsupported_image_workflow");
        }
        if self.attention == ImageAttention::Fast && self.workflow == QWEN_RECIPE {
            return Err("image_attention_unsupported");
        }
        if self.prompt.len() > 32768
            || self.negative_prompt.len() > 32768
            || self.prompt.contains('\0')
            || self.negative_prompt.contains('\0')
        {
            return Err("image_prompt_invalid");
        }
        if self.seed.is_empty()
            || self.seed.len() > 20
            || !self.seed.bytes().all(|b| b.is_ascii_digit())
            || self.seed.parse::<u64>().is_err()
        {
            return Err("image_seed_invalid");
        }
        if !(64..=2048).contains(&self.width)
            || !(64..=2048).contains(&self.height)
            || !self.width.is_multiple_of(16)
            || !self.height.is_multiple_of(16)
        {
            return Err("image_dimensions_unsupported");
        }
        if !(1..=200).contains(&self.steps)
            || !self.cfg.is_finite()
            || !(0.0..=30.0).contains(&self.cfg)
            || (self.workflow == QWEN_RECIPE
                && !self
                    .flow_shift
                    .is_some_and(|s| s.is_finite() && (0.1..=10.0).contains(&s)))
            || (self.workflow != QWEN_RECIPE && self.flow_shift.is_some())
        {
            return Err("image_sampling_settings_invalid");
        }
        if self.reference_images.len() > 10
            || (self.workflow != QWEN21_EDIT_RECIPE && !self.reference_images.is_empty())
            || self
                .reference_images
                .iter()
                .any(|s| s.is_empty() || s.len() > 16 * 1024 * 1024)
            || self.reference_images.iter().map(String::len).sum::<usize>()
                > MAX_IMAGE_REQUEST_BYTES - 1024 * 1024
        {
            return Err("image_references_invalid");
        }
        Ok(())
    }
    pub fn collection_id(&self) -> &'static str {
        if self.workflow == QWEN_RECIPE {
            "qwen-image-2512-fp8"
        } else {
            "qwen-image-2.1-int8"
        }
    }
    pub fn validate_execution(&self) -> Result<(), &'static str> {
        self.validate()?;
        if self.workflow == QWEN21_EDIT_RECIPE && self.reference_images.is_empty() {
            return Err("image_edit_requires_references");
        }
        Ok(())
    }
}
#[derive(Serialize)]
pub struct ImportedImageRecipe {
    pub recipe: ImageRecipe,
    pub selected_output: String,
    pub required_reference_images: usize,
    pub warnings: Vec<&'static str>,
}
