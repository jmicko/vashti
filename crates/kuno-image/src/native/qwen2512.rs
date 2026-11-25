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

/// The text encoder, DiT and VAE: the better form installed, Kuno's INT8 conversion of the
/// publisher's files (`qwen-image-2512-int8`, closer to the original model and loaded without
/// conversion) when it is complete, else ComfyUI's FP8 files.
fn files(directory: &std::path::Path) -> [PathBuf; 3] {
    let int8 = directory.join(INT8);
    if ["text_encoder", "transformer", "vae"]
        .iter()
        .all(|part| complete(&int8.join(part)))
    {
        ["text_encoder", "transformer", "vae"].map(|part| int8.join(part))
    } else {
        [
            "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "qwen_image_2512_fp8_e4m3fn.safetensors",
            "qwen_image_vae.safetensors",
        ]
        .map(|file| directory.join(file))
    }
}
/// Whether a part's directory holds a whole model: at least one file, and for files split as
/// `<name>-NNNNN-of-MMMMM.safetensors`, every piece. The installer only activates files that
/// match their pins, so present means verified; an install still in progress is incomplete.
fn complete(directory: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return false;
    };
    let names: Vec<String> = entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    !names.is_empty()
        && names.iter().all(|name| {
            let stem = name.trim_end_matches(".safetensors");
            let Some((head, total)) = stem.rsplit_once("-of-") else {
                return true;
            };
            let Some((base, _)) = head.rsplit_once('-') else {
                return true;
            };
            let Ok(count) = total.parse::<usize>() else {
                return true;
            };
            (1..=count).all(|i| {
                names.contains(&format!(
                    "{base}-{i:0width$}-of-{total}.safetensors",
                    width = total.len()
                ))
            })
        })
}
/// The directory the `qwen-image-2512-int8` collection's conversions are written to.
pub const INT8: &str = "qwen-image-2512-int8";

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
        let [encoder, dit, vae] = files(directory);
        let models = Models::open(&gpu, &encoder, &dit, &vae).map_err(super::qwen::failure)?;
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
        shift: recipe.flow_shift.ok_or("image_sampling_settings_invalid")? as f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_is_chosen_only_when_every_piece_is_installed() {
        let root = tempfile_dir();
        let put = |path: &str| {
            let p = root.join(path);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"").unwrap();
        };
        let int8 = |part: &str, file: &str| format!("{INT8}/{part}/{file}");
        put(&int8("text_encoder", "model-00001-of-00002.safetensors"));
        put(&int8(
            "transformer",
            "diffusion_pytorch_model-00001-of-00002.safetensors",
        ));
        put(&int8(
            "transformer",
            "diffusion_pytorch_model-00002-of-00002.safetensors",
        ));
        put(&int8("vae", "diffusion_pytorch_model.safetensors"));
        // A text encoder piece is still missing: ComfyUI's FP8 files.
        assert_eq!(
            files(&root)[1],
            root.join("qwen_image_2512_fp8_e4m3fn.safetensors")
        );
        put(&int8("text_encoder", "model-00002-of-00002.safetensors"));
        assert_eq!(files(&root)[1], root.join(INT8).join("transformer"));
        std::fs::remove_dir_all(&root).unwrap();
    }
    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("kuno-2512-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
