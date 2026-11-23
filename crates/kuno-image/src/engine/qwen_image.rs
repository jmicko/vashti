//! Qwen Image 2.1 generation in Rust: the C++ engine's prompt template and emphasis, the
//! Qwen3-VL 8B text encoder, the diffusion transformer's prefix and steps with the FLUX
//! schedule and Euler, classifier-free guidance, and the VAE decoder. Everything runs on the
//! GPU except the tokenizer, the small time layers and the sampler's arithmetic.
use std::path::Path;

use super::{
    gpu::Gpu,
    prompt::{apply_weights, parse_attention},
    qwen_dit::{CHANNELS, CONTEXT},
    qwen_gpu::{Prefix, QwenGpu},
    qwen_vae::{QwenVae, SCALE, to_rgba},
    qwen_vae_gpu::GpuVae,
    rng::Mt19937,
    text_encoder::{Config, Prompt, TextEncoder},
    text_gpu::GpuTextEncoder,
    tokenizer::Tokenizer,
};

const SYSTEM: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const USER: &str = "<|im_start|>user\n";
const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The FLUX schedule for `tokens` latent cells: `steps + 1` sigmas from 1 down to 0.
pub fn sigmas(steps: usize, tokens: usize) -> Vec<f32> {
    let (base, max) = (0.5f32, 1.15f32);
    let m = (max - base) / (4096.0 - 256.0);
    let b = base - m * 256.0;
    let mu = tokens as f32 * m + b;
    let mut out: Vec<f32> = (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            if t <= 0.0 {
                0.0
            } else {
                mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(1.0))
            }
        })
        .collect();
    out[steps] = 0.0;
    out
}

/// The models of one generation, kept between requests.
pub struct Models {
    tokenizer: Tokenizer,
    encoder: GpuTextEncoder,
    dit: QwenGpu,
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
    pub seed: i64,
}

impl Models {
    /// Opens the three files on one GPU: the text encoder's weights stay in host memory and
    /// stream through the device per prompt; the DiT and the VAE are placed.
    pub fn open(gpu: &Gpu, encoder: &Path, dit: &Path, vae: &Path) -> Result<Self, &'static str> {
        let file = TextEncoder::open(encoder)?;
        if file.config != Config::QWEN3_VL_8B {
            return Err("model_file_invalid");
        }
        Ok(Self {
            tokenizer: Tokenizer::new()?,
            encoder: GpuTextEncoder::new(gpu, file)?,
            dit: QwenGpu::load(gpu.clone(), dit)?,
            vae: GpuVae::new(gpu, &QwenVae::open(vae)?)?,
        })
    }

    /// The text features of a prompt `[tokens][CONTEXT]` and each token's image slot (empty:
    /// all text), as the engine conditions Qwen Image 2.1.
    pub fn condition(&self, text: &str) -> Result<(Vec<f32>, Vec<i32>), &'static str> {
        let start = self.tokenizer.encode(SYSTEM)?.len();
        let user = if text.is_empty() { " " } else { text };
        let mut pieces = vec![(format!("{SYSTEM}{USER}"), 1.0f32)];
        pieces.extend(parse_attention(user));
        pieces.push((ASSISTANT.to_owned(), 1.0));
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for (piece, weight) in &pieces {
            let tokens = self.tokenizer.encode(piece)?;
            weights.extend(std::iter::repeat_n(*weight, tokens.len()));
            ids.extend(tokens.into_iter().map(|t| t as i32));
        }
        let mut hidden = self.encoder.forward(&Prompt {
            ids: &ids,
            images: Vec::new(),
        })?;
        apply_weights(&mut hidden, &weights);
        if ids.len() <= start {
            return Err("text_invalid");
        }
        Ok((hidden[start * CONTEXT..].to_vec(), Vec::new()))
    }

    /// RGBA bytes of the image, `height * width * 4`. `progress(step)` is called after each
    /// step; returning false cancels.
    pub fn generate(
        &self,
        request: &Request,
        mut progress: impl FnMut(usize) -> bool,
    ) -> Result<Vec<u8>, &'static str> {
        let Request {
            width,
            height,
            steps,
            cfg,
            ..
        } = *request;
        if width == 0
            || height == 0
            || !width.is_multiple_of(SCALE)
            || !height.is_multiple_of(SCALE)
            || steps == 0
        {
            return Err("invalid_input");
        }
        let (h, w) = (height / SCALE, width / SCALE);
        let guided = cfg != 1.0;
        let prefix = |text: &str| -> Result<Prefix, &'static str> {
            let (context, slots) = self.condition(text)?;
            self.dit.prefix(&context, &slots, &[], (h, w))
        };
        let cond = prefix(request.prompt)?;
        let uncond = guided.then(|| prefix(request.negative)).transpose()?;
        let mut x = Mt19937::new(request.seed as u64).randn(CHANNELS * h * w);
        let sigmas = sigmas(steps, h * w);
        for step in 0..steps {
            let sigma = sigmas[step];
            let timestep = sigma * 1000.0;
            let mut out = self.dit.step(&cond, &x, timestep)?;
            if let Some(uncond) = &uncond {
                let negative = self.dit.step(uncond, &x, timestep)?;
                for (o, n) in out.iter_mut().zip(&negative) {
                    *o = n + cfg * (*o - n);
                }
            }
            let next = sigmas[step + 1];
            for (x, o) in x.iter_mut().zip(&out) {
                let denoised = o * -sigma + *x;
                let d = (*x - denoised) / sigma;
                *x += d * (next - sigma);
            }
            if !progress(step) {
                return Err("cancelled");
            }
        }
        let image = self.vae.decode(&x, h, w)?;
        Ok(to_rgba(&image, height, width))
    }
}

#[cfg(test)]
mod tests {
    use super::{super::gpu::Select, *};

    #[test]
    fn schedule_follows_the_engine() {
        // 16 x 16 latent, 4 steps: the captured timesteps 1000, 831.8243, 622.45935, 354.66122.
        let s = sigmas(4, 256);
        let t: Vec<f32> = s.iter().map(|s| s * 1000.0).collect();
        assert_eq!(t[0], 1000.0);
        assert!((t[1] - 831.8243).abs() < 1e-3 && (t[2] - 622.45935).abs() < 1e-3);
        assert!((t[3] - 354.66122).abs() < 1e-3 && t[4] == 0.0);
    }

    /// `KUNO_QWEN_MODELS=<dir with the three files> KUNO_QWEN_REFERENCE=<engine's PNG>`:
    /// the fox prompt at 512 x 512, 6 steps, seed 7, against the engine's image of the same
    /// request; writes the Rust image beside the reference.
    #[test]
    #[ignore]
    fn generation_matches_cpp_engine() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_MODELS").unwrap());
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let models = Models::open(
            &gpu,
            &dir.join("qwen3vl_8b_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_vae_bf16.safetensors"),
        )
        .unwrap();
        println!("open {:.1} s", started.elapsed().as_secs_f64());
        let request = Request {
            prompt: "A red fox sitting on a mossy rock in a misty forest, morning light",
            negative: "",
            width: 512,
            height: 512,
            steps: 6,
            cfg: 1.0,
            seed: 7,
        };
        let started = std::time::Instant::now();
        let rgba = models.generate(&request, |_| true).unwrap();
        println!("generate {:.2} s", started.elapsed().as_secs_f64());
        let reference = std::path::PathBuf::from(std::env::var("KUNO_QWEN_REFERENCE").unwrap());
        let mut decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&reference).unwrap(),
        ));
        decoder.set_transformations(png::Transformations::EXPAND);
        let mut reader = decoder.read_info().unwrap();
        let mut want = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut want).unwrap();
        assert_eq!((info.width, info.height), (512, 512));
        let channels = info.line_size / 512;
        let (mut error, mut count) = (0f64, 0usize);
        for (a, b) in rgba.chunks_exact(4).zip(want.chunks_exact(channels)) {
            for c in 0..3 {
                error += (f64::from(a[c]) - f64::from(b[c])).powi(2);
                count += 1;
            }
        }
        let psnr = 10.0 * (255.0f64 * 255.0 / (error / count as f64)).log10();
        println!("RGB PSNR against the engine: {psnr:.1} dB");
        let out = reference.with_file_name("rust-fox.png");
        let mut encoder = png::Encoder::new(std::fs::File::create(&out).unwrap(), 512, 512);
        encoder.set_color(png::ColorType::Rgba);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&rgba)
            .unwrap();
        assert!(psnr > 20.0);
    }
}
