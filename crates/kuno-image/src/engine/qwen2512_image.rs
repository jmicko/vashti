//! Qwen Image 2512 text-to-image in Rust: the prompt through Qwen2.5-VL (`text_gpu`, weights
//! streamed per layer), the diffusion transformer (`qwen2512_gpu`), Euler sampling on the
//! engine's simple schedule with an explicit flow shift, and the Wan 2.1 VAE decoder
//! (`qwen_vae_gpu`). Follows the C++ engine's `qwen_image` pipeline: the same prompt template,
//! emphasis weights, noise, schedule and CFG.
use std::path::Path;

use super::{
    gpu::Gpu,
    prompt::{apply_weights, parse_attention},
    qwen_vae::{QwenVae, WAN21_LATENT, WAN21_SCALE, to_rgb},
    qwen_vae_gpu::GpuVae,
    qwen2512_dit::{CONTEXT, START, TEMPLATE},
    qwen2512_gpu::{Prompt as DitPrompt, Qwen2512Gpu},
    rng::Mt19937,
    text_encoder::{Prompt, TextEncoder},
    text_gpu::GpuTextEncoder,
    tokenizer::Tokenizer,
};

const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// Pixels per latent cell, and per DiT token side.
pub const SCALE: usize = WAN21_SCALE;
pub const MULTIPLE: usize = 2 * WAN21_SCALE;
/// The engine's latent preview for the Wan 2.1 VAE: RGB per latent channel and a bias.
const PREVIEW_RGB: [[f32; 3]; WAN21_LATENT] = [
    [0.015123, -0.148418, 0.479828],
    [0.003652, -0.010680, -0.037142],
    [0.212264, 0.063033, 0.016779],
    [0.232999, 0.406476, 0.220125],
    [-0.051864, -0.082384, -0.069396],
    [0.085005, -0.161492, 0.010689],
    [-0.245369, -0.506846, -0.117010],
    [-0.151145, 0.017721, 0.007207],
    [-0.293239, -0.207936, -0.421135],
    [-0.187721, 0.050783, 0.177649],
    [-0.013067, 0.265964, 0.166578],
    [0.028327, 0.109329, 0.108642],
    [-0.205343, 0.043991, 0.148914],
    [0.014307, -0.048647, -0.007219],
    [0.217150, 0.053074, 0.319923],
    [0.155357, 0.083156, 0.064780],
];
const PREVIEW_BIAS: [f32; 3] = [-0.270270, -0.234976, -0.456853];

/// RGBA bytes of a denoised latent `[16][h][w]` (one pixel per latent cell).
pub fn preview_rgba(denoised: &[f32], height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = vec![0u8; plane * 4];
    for (cell, pixel) in out.chunks_exact_mut(4).enumerate() {
        let mut rgb = [0f32; 3];
        for (d, row) in PREVIEW_RGB.iter().enumerate() {
            let value = denoised[d * plane + cell];
            for (c, sum) in rgb.iter_mut().enumerate() {
                *sum += value * row[c];
            }
        }
        for c in 0..3 {
            let v = ((rgb[c] + PREVIEW_BIAS[c]) * 0.5 + 0.5).clamp(0.0, 1.0);
            pixel[c] = (v * 255.0) as u8;
        }
        pixel[3] = 255;
    }
    out
}

/// The engine's simple schedule for a flow model with `shift`: `steps` sigmas from the
/// 1000-entry table, then 0.
pub fn sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let factor = 1000.0f32 / steps as f32;
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            let index = (999 - (i as f32 * factor) as i32).max(0);
            let t = (index + 1) as f32 / 1000.0;
            if shift == 1.0 {
                t
            } else {
                shift * t / (1.0 + (shift - 1.0) * t)
            }
        })
        .collect();
    out.push(0.0);
    out
}

/// The models of one generation, kept between requests.
pub struct Models {
    tokenizer: Tokenizer,
    encoder: GpuTextEncoder,
    dit: Qwen2512Gpu,
    vae: GpuVae,
}

pub struct Request<'a> {
    pub prompt: &'a str,
    pub negative: &'a str,
    /// Image size in pixels, multiples of 16.
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg: f32,
    /// The sampler's flow shift.
    pub shift: f32,
    pub seed: i64,
    /// Attention scored with int8 queries and keys: quicker, slightly different pixels.
    pub fast_attention: bool,
}

impl Models {
    /// Opens the three files on one GPU: the text encoder's weights stay in host memory and
    /// stream through the device per prompt; the DiT and the VAE are placed.
    pub fn open(gpu: &Gpu, encoder: &Path, dit: &Path, vae: &Path) -> Result<Self, &'static str> {
        let file = TextEncoder::open(encoder)?;
        if !file.qwen25 || file.config.width != CONTEXT {
            return Err("model_file_invalid");
        }
        Ok(Self {
            tokenizer: Tokenizer::new()?,
            encoder: GpuTextEncoder::new(gpu, file)?,
            dit: Qwen2512Gpu::load(gpu.clone(), dit)?,
            vae: GpuVae::new(gpu, &QwenVae::open_wan21(vae)?)?,
        })
    }

    /// A prompt's text features in the DiT's text width, as the engine conditions Qwen Image.
    fn condition(&self, text: &str) -> Result<DitPrompt, &'static str> {
        let user = if text.is_empty() { " " } else { text };
        let mut pieces = vec![(TEMPLATE.to_owned(), 1.0f32)];
        pieces.extend(parse_attention(user));
        pieces.push((ASSISTANT.to_owned(), 1.0));
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for (piece, weight) in &pieces {
            let tokens = self.tokenizer.encode(piece)?;
            weights.extend(std::iter::repeat_n(*weight, tokens.len()));
            ids.extend(tokens.into_iter().map(|t| t as i32));
        }
        if ids.len() <= START {
            return Err("text_invalid");
        }
        let mut hidden = self.encoder.forward(&Prompt {
            ids: &ids,
            images: vec![],
        })?;
        apply_weights(&mut hidden, &weights);
        self.dit.prompt(&hidden[START * CONTEXT..])
    }

    /// RGBA bytes of the image, `height * width * 4`. `progress(step, denoised)` is called
    /// after each step (from 1) with that step's denoised latent `[16][height / 8][width / 8]`;
    /// returning false cancels.
    pub fn generate(
        &self,
        request: &Request,
        mut progress: impl FnMut(usize, &[f32]) -> bool,
    ) -> Result<Vec<u8>, &'static str> {
        let Request {
            width,
            height,
            steps,
            cfg,
            shift,
            fast_attention,
            ..
        } = *request;
        if width == 0
            || height == 0
            || !width.is_multiple_of(MULTIPLE)
            || !height.is_multiple_of(MULTIPLE)
            || steps == 0
            || shift.is_nan()
            || shift <= 0.0
        {
            return Err("invalid_input");
        }
        let (h, w) = (height / SCALE, width / SCALE);
        let guided = cfg != 1.0;
        let cond = self.condition(request.prompt)?;
        let uncond = guided
            .then(|| self.condition(request.negative))
            .transpose()?;
        let mut x = Mt19937::new(request.seed as u64).randn(WAN21_LATENT * h * w);
        let sigmas = sigmas(steps, shift);
        let timesteps: Vec<f32> = sigmas[..steps].iter().map(|s| s * 1000.0).collect();
        let times = self.dit.times(&timesteps);
        let mut prompts = vec![&cond];
        if let Some(uncond) = &uncond {
            prompts.push(uncond);
        }
        for step in 0..steps {
            let sigma = sigmas[step];
            let mut outs =
                self.dit
                    .step(&prompts, &x, h / 2, w / 2, &times[step], fast_attention)?;
            let mut out = outs.remove(0);
            if let Some(negative) = outs.first() {
                for (o, n) in out.iter_mut().zip(negative) {
                    *o = n + cfg * (*o - n);
                }
            }
            let next = sigmas[step + 1];
            // The engine's denoised prediction, then its Euler step from it.
            for (o, x) in out.iter_mut().zip(&x) {
                *o = *o * -sigma + x;
            }
            for (x, denoised) in x.iter_mut().zip(&out) {
                let d = (*x - denoised) / sigma;
                *x += d * (next - sigma);
            }
            if !progress(step + 1, &out) {
                return Err("cancelled");
            }
        }
        let image = self.vae.decode(&x, h, w)?;
        Ok(to_rgb(&image, height, width))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{gpu::Select, qwen_dit::tests::read},
        *,
    };

    #[test]
    fn schedule_follows_the_engine() {
        // Two steps with shift 3.1: the captured timesteps 1000 and 756.09753.
        let s = sigmas(2, 3.1);
        assert_eq!(s[0], 1.0);
        assert!((s[1] * 1000.0 - 756.097_53).abs() < 1e-3 && s[2] == 0.0);
        // A count that does not divide the table: the engine truncates each offset.
        let s = sigmas(30, 1.0);
        assert!((s[1] - 0.967).abs() < 1e-6);
    }

    /// `KUNO_QWEN2512_MODELS=<dir> KUNO_QWEN2512_GOLDEN=<dir>`: the capture script's request
    /// (fox prompt, negative, seed 7, its size and steps, CFG from `KUNO_QWEN2512_CFG`) end to
    /// end against the engine's image; the PNG goes to `KUNO_QWEN2512_PNG` if set.
    #[test]
    #[ignore = "needs the Intel GPU and the models"]
    fn generation_follows_the_engine() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_MODELS").unwrap());
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let models = Models::open(
            &gpu,
            &dir.join("qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            &dir.join("qwen_image_2512_fp8_e4m3fn.safetensors"),
            &dir.join("qwen_image_vae.safetensors"),
        )
        .unwrap();
        println!("open {:.1} s", started.elapsed().as_secs_f64());
        let golden = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        let dims: Vec<usize> = std::fs::read_to_string(golden.join("vae.pixels0.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height) = (dims[0], dims[1]);
        let steps = std::fs::read_dir(&golden)
            .unwrap()
            .filter(|e| {
                let name = e.as_ref().unwrap().file_name();
                let name = name.to_string_lossy();
                name.starts_with("step") && name.ends_with(".x.f32")
            })
            .count()
            / 2;
        let cfg: f32 = std::env::var("KUNO_QWEN2512_CFG").map_or(2.5, |v| v.parse().unwrap());
        for fast in [false, true] {
            let started = std::time::Instant::now();
            let rgba = models
                .generate(
                    &Request {
                        prompt: "A red fox sitting on a mossy rock in a misty forest, morning light",
                        negative: "blurry, low quality",
                        width,
                        height,
                        steps,
                        cfg,
                        shift: 3.1,
                        seed: 7,
                        fast_attention: fast,
                    },
                    |_, _| true,
                )
                .unwrap();
            let want = read(&golden.join("vae.pixels0.f32"));
            let plane = width * height;
            let mut diff = 0f64;
            for (cell, pixel) in rgba.chunks_exact(4).enumerate() {
                for c in 0..3 {
                    let w = (want[c * plane + cell] * 255.0).round();
                    diff += (f64::from(pixel[c]) - f64::from(w)).abs();
                }
            }
            println!(
                "{width}x{height}, {steps} steps{}: {:.1} s, mean difference {:.2} of 255",
                if fast { " fast" } else { "" },
                started.elapsed().as_secs_f64(),
                diff / (plane * 3) as f64
            );
            if let Ok(path) = std::env::var("KUNO_QWEN2512_PNG") {
                let path = path.replace(".png", if fast { "-fast.png" } else { ".png" });
                let mut encoder = png::Encoder::new(
                    std::fs::File::create(&path).unwrap(),
                    width as u32,
                    height as u32,
                );
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                encoder
                    .write_header()
                    .unwrap()
                    .write_image_data(&rgba)
                    .unwrap();
            }
        }
    }
}
