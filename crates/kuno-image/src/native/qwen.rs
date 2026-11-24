//! Qwen Image 2.1 requests through the Rust engine (`engine::qwen_image`) instead of the C++
//! engine. The three models stay loaded between requests (the DiT, the VAE and the vision
//! tower in video memory, the text encoder's weights in host memory) until a video request
//! or another image model needs the memory. Only weights outlive a request: prompts,
//! references, latents and the prefix caches are dropped before this returns.
use std::{ffi::c_void, path::PathBuf, sync::Mutex};

use crate::engine::{
    gpu::{Gpu, Select},
    qwen_image::{Models, Reference, Request, preview_rgba},
};

unsafe extern "C" {
    fn kuno_qwen_session_release(session: *mut c_void);
    fn kuno_image_cancel_requested() -> bool;
}

/// The loaded models and the directory they came from.
static MODELS: Mutex<Option<(PathBuf, Models)>> = Mutex::new(None);

/// Frees the kept models (for a video request or the other image model).
pub(crate) fn evict() {
    let kept = MODELS.lock().ok().and_then(|mut kept| kept.take());
    drop(kept);
}

/// Engine error names to the worker's image errors.
pub(super) fn failure(error: &'static str) -> &'static str {
    match error {
        "cancelled" => "image_cancelled",
        "gpu_memory" => "image_gpu_memory_unavailable",
        "gpu_lost" | "gpu_failed" => "image_vulkan_device_lost",
        "gpu_unsupported" | "gpu_not_found" => "image_physical_gpu_not_found",
        "model_file_invalid" | "model_tensor_missing" | "model_tensor_invalid" => {
            "image_model_load_failed"
        }
        _ => "image_generation_failed",
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn generate(
    native: *mut c_void,
    directory: &std::path::Path,
    pci: &str,
    recipe: &kuno_protocol::ImageRecipe,
    seed: i64,
    references: &[crate::references::Pixels],
    previews: bool,
) -> Result<Vec<u8>, &'static str> {
    // The C++ engine's own Qwen context holds its copy of the weights in video memory.
    // SAFETY: the session pointer is live for the request; release only resets its context.
    unsafe { kuno_qwen_session_release(native) };
    let mut kept = MODELS.lock().map_err(|_| "image_engine_unavailable")?;
    if kept.as_ref().is_none_or(|(path, _)| path != directory) {
        *kept = None;
        let gpu = Gpu::open(Select::Pci(pci.to_owned())).map_err(failure)?;
        let models = Models::open(
            &gpu,
            &directory.join("qwen3vl_8b_int8_convrot.safetensors"),
            &directory.join("qwen_image_2.1_int8_convrot.safetensors"),
            &directory.join("qwen_image_2.1_vae_bf16.safetensors"),
        )
        .map_err(failure)?;
        *kept = Some((directory.to_owned(), models));
    }
    let (_, models) = kept.as_ref().ok_or("image_engine_unavailable")?;
    let references: Vec<Reference> = references
        .iter()
        .map(|p| Reference {
            rgba: &p.rgba,
            width: p.width as usize,
            height: p.height as usize,
        })
        .collect();
    let request = Request {
        prompt: &recipe.prompt,
        negative: &recipe.negative_prompt,
        width: recipe.width as usize,
        height: recipe.height as usize,
        steps: recipe.steps as usize,
        cfg: recipe.cfg as f32,
        seed,
        references: &references,
        fast_attention: recipe.attention == kuno_protocol::ImageAttention::Fast,
    };
    let (h, w) = (recipe.height as usize / 16, recipe.width as usize / 16);
    let result = models.generate(&request, |step, denoised| {
        // SAFETY: a plain flag read set by the worker's SIGTERM handler.
        if unsafe { kuno_image_cancel_requested() } {
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
            Err(failure(error))
        }
    }
}
