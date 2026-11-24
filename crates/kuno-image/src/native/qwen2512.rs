//! Qwen Image 2512 requests through the Rust engine (`engine::qwen2512_image`). The three
//! models stay loaded between requests (the DiT and the VAE in video memory, the text
//! encoder's weights in host memory) until a video request or another image model needs the
//! memory. Only weights outlive a request: prompts, latents and activations are dropped
//! before this returns.
use std::{path::PathBuf, sync::Mutex};

use crate::engine::{
    gpu::{Gpu, Select},
    qwen2512_image::{Models, Request, SCALE, preview_rgba},
};

/// The loaded models and the directory they came from.
static MODELS: Mutex<Option<(PathBuf, Models)>> = Mutex::new(None);

/// Frees the kept models (for a video request or the other image model).
pub(crate) fn evict() {
    let kept = MODELS.lock().ok().and_then(|mut kept| kept.take());
    drop(kept);
}

pub(super) fn generate(
    directory: &std::path::Path,
    pci: &str,
    recipe: &kuno_protocol::ImageRecipe,
    seed: i64,
    previews: bool,
    cancelled: impl Fn() -> bool,
) -> Result<Vec<u8>, &'static str> {
    let mut kept = MODELS.lock().map_err(|_| "image_engine_unavailable")?;
    if kept.as_ref().is_none_or(|(path, _)| path != directory) {
        *kept = None;
        let gpu = Gpu::open(Select::Pci(pci.to_owned())).map_err(super::qwen::failure)?;
        let models = Models::open(
            &gpu,
            &directory.join("qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            &directory.join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &directory.join("qwen_image_vae.safetensors"),
        )
        .map_err(super::qwen::failure)?;
        *kept = Some((directory.to_owned(), models));
    }
    let (_, models) = kept.as_ref().ok_or("image_engine_unavailable")?;
    let request = Request {
        prompt: &recipe.prompt,
        negative: &recipe.negative_prompt,
        width: recipe.width as usize,
        height: recipe.height as usize,
        steps: recipe.steps as usize,
        cfg: recipe.cfg as f32,
        shift: recipe
            .flow_shift
            .ok_or("image_sampling_settings_invalid")? as f32,
        seed,
        fast_attention: recipe.attention == kuno_protocol::ImageAttention::Fast,
    };
    let (h, w) = (
        recipe.height as usize / SCALE,
        recipe.width as usize / SCALE,
    );
    let result = models.generate(&request, |step, denoised| {
        if cancelled() {
            return false;
        }
        if previews {
            let rgba = preview_rgba(denoised, h, w);
            if let Some(frame) = super::preview_frame(
                step as i32,
                recipe.steps,
                rgba.as_ptr(),
                w as u32,
                h as u32,
                4,
            ) {
                let _ = crate::preview::write_frame(1, &frame);
            }
        }
        true
    });
    match result {
        Ok(rgba) => Ok(rgba),
        Err(error) => {
            // Uncertain state (a failed device or allocation) is not kept.
            if error != "cancelled" {
                *kept = None;
            }
            Err(super::qwen::failure(error))
        }
    }
}
