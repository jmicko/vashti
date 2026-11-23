//! Qwen Image 2.1 generation and editing in Rust: the C++ engine's prompt template and
//! emphasis, the Qwen3-VL 8B text encoder and vision tower, reference images' VAE latents,
//! the diffusion transformer's prefix and steps with the FLUX schedule and Euler,
//! classifier-free guidance, and the VAE decoder. Everything runs on the GPU except the
//! tokenizer, the small time layers and the sampler's arithmetic.
use std::path::Path;

use super::{
    gpu::Gpu,
    prompt::{apply_weights, parse_attention},
    qwen_dit::{CHANNELS, CONTEXT},
    qwen_gpu::{Prefix, QwenGpu},
    qwen_vae::{QwenVae, SCALE, to_rgba},
    qwen_vae_gpu::GpuVae,
    rng::Mt19937,
    text_encoder::{Config, Image, Prompt, TextEncoder},
    text_gpu::GpuTextEncoder,
    tokenizer::Tokenizer,
    vision::{self, Vision},
    vision_gpu::GpuVision,
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
    vision: GpuVision,
    dit: QwenGpu,
    vae: GpuVae,
}

/// A reference image: interleaved RGBA bytes, sides multiples of 32.
pub struct Reference<'a> {
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
}
/// What the vision tower and the VAE make of a reference, shared by both prompts.
struct Encoded {
    /// The text encoder's rows, then the three deep-stack sets.
    rows: [Vec<f32>; 4],
    /// The patch grid.
    grid: (usize, usize),
    latent: Vec<f32>,
    /// Latent size.
    size: (usize, usize),
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
    pub references: &'a [Reference<'a>],
}

impl Models {
    /// Opens the three files on one GPU: the text encoder's weights stay in host memory and
    /// stream through the device per prompt; the DiT and the VAE are placed.
    pub fn open(gpu: &Gpu, encoder: &Path, dit: &Path, vae: &Path) -> Result<Self, &'static str> {
        let file = TextEncoder::open(encoder)?;
        if file.config != Config::QWEN3_VL_8B {
            return Err("model_file_invalid");
        }
        let tower = Vision::open(encoder)?;
        if tower.out != file.config.width {
            return Err("model_file_invalid");
        }
        Ok(Self {
            tokenizer: Tokenizer::new()?,
            vision: GpuVision::new(gpu, &tower)?,
            encoder: GpuTextEncoder::new(gpu, file)?,
            dit: QwenGpu::load(gpu.clone(), dit)?,
            vae: GpuVae::new(gpu, &QwenVae::open(vae)?)?,
        })
    }

    /// A reference through the vision tower (composited over white, scaled to -1..1) and the
    /// VAE encoder.
    fn encode_reference(&self, reference: &Reference) -> Result<Encoded, &'static str> {
        let (w, h) = (reference.width, reference.height);
        if !w.is_multiple_of(vision::FACTOR)
            || !h.is_multiple_of(vision::FACTOR)
            || reference.rgba.len() != w * h * 4
        {
            return Err("invalid_input");
        }
        let plane = w * h;
        let mut rgb = vec![0f32; 3 * plane];
        for (cell, pixel) in reference.rgba.chunks_exact(4).enumerate() {
            let alpha = f32::from(pixel[3]) / 255.0;
            for c in 0..3 {
                let value = f32::from(pixel[c]) / 255.0;
                rgb[c * plane + cell] = 2.0 * (value * alpha + 1.0 - alpha) - 1.0;
            }
        }
        Ok(Encoded {
            rows: self.vision.encode(&rgb, w, h)?,
            grid: vision::grid(w, h),
            latent: self.vae.encode(reference.rgba, h, w)?,
            size: (h / SCALE, w / SCALE),
        })
    }

    /// The text features of a prompt `[tokens][CONTEXT]` with the references' vision rows,
    /// and each token's image slot (empty: all text), as the engine conditions Qwen Image 2.1.
    fn condition(
        &self,
        text: &str,
        references: &[Encoded],
    ) -> Result<(Vec<f32>, Vec<i32>), &'static str> {
        let start = self.tokenizer.encode(SYSTEM)?.len();
        let mut prefix = format!("{SYSTEM}{USER}");
        // Each image's first token and row count.
        let mut places = Vec::new();
        for (i, reference) in references.iter().enumerate() {
            prefix += if i == 0 { "" } else { " " };
            prefix += &format!("<image{}><|vision_start|>", i + 1);
            let index = self.tokenizer.encode(&prefix)?.len();
            let count = reference.rows[0].len() / CONTEXT;
            places.push((index, count));
            prefix += &"<|image_pad|>".repeat(count);
            prefix += "<|vision_end|>";
        }
        let user = if text.is_empty() { " " } else { text };
        let mut pieces = vec![(prefix, 1.0f32)];
        pieces.extend(parse_attention(user));
        pieces.push((ASSISTANT.to_owned(), 1.0));
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for (piece, weight) in &pieces {
            let tokens = self.tokenizer.encode(piece)?;
            weights.extend(std::iter::repeat_n(*weight, tokens.len()));
            ids.extend(tokens.into_iter().map(|t| t as i32));
        }
        let images = references
            .iter()
            .zip(&places)
            .map(|(r, &(index, size))| Image {
                index,
                size,
                grid_h: r.grid.0,
                grid_w: r.grid.1,
                embeds: &r.rows[0],
                deepstack: r.rows[1..].iter().map(Vec::as_slice).collect(),
            })
            .collect();
        let mut hidden = self.encoder.forward(&Prompt { ids: &ids, images })?;
        apply_weights(&mut hidden, &weights);
        if ids.len() <= start {
            return Err("text_invalid");
        }
        let tokens = ids.len() - start;
        let mut slots = Vec::new();
        if !references.is_empty() {
            slots = vec![0i32; tokens];
            for (i, &(index, count)) in places.iter().enumerate() {
                let begin = index.checked_sub(start).ok_or("text_invalid")?;
                slots
                    .get_mut(begin..begin + count)
                    .ok_or("text_invalid")?
                    .fill(i as i32 + 1);
            }
        }
        Ok((hidden[start * CONTEXT..].to_vec(), slots))
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
        let encoded = request
            .references
            .iter()
            .map(|r| self.encode_reference(r))
            .collect::<Result<Vec<_>, _>>()?;
        let latents: Vec<(&[f32], usize, usize)> = encoded
            .iter()
            .map(|e| (&e.latent[..], e.size.0, e.size.1))
            .collect();
        let prefix = |text: &str| -> Result<Prefix, &'static str> {
            let (context, slots) = self.condition(text, &encoded)?;
            self.dit.prefix(&context, &slots, &latents, (h, w))
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
            references: &[],
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

    fn read_png(path: &std::path::Path) -> (Vec<u8>, usize, usize) {
        let mut decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        decoder.set_transformations(png::Transformations::EXPAND);
        let mut reader = decoder.read_info().unwrap();
        let mut data = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut data).unwrap();
        let (w, h) = (info.width as usize, info.height as usize);
        let channels = info.line_size / w;
        let rgba = data[..w * h * channels]
            .chunks_exact(channels)
            .flat_map(|p| [p[0], p[1], p[2], if channels == 4 { p[3] } else { 255 }])
            .collect();
        (rgba, w, h)
    }

    /// `KUNO_QWEN_MODELS=<dir> KUNO_QWEN_EDIT=<dir with ref-cat-512x384.png, qedit512.png>`:
    /// an edit with one reference (CFG 4, 8 steps, 512 x 512, seed 7) against the engine's.
    #[test]
    #[ignore]
    fn edit_matches_cpp_engine() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_MODELS").unwrap());
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let models = Models::open(
            &gpu,
            &dir.join("qwen3vl_8b_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_vae_bf16.safetensors"),
        )
        .unwrap();
        let shots = std::path::PathBuf::from(std::env::var("KUNO_QWEN_EDIT").unwrap());
        let (rgba, width, height) = read_png(&shots.join("ref-cat-512x384.png"));
        let references = [Reference {
            rgba: &rgba,
            width,
            height,
        }];
        let request = Request {
            prompt: "Make the cat wear a small red hat",
            negative: "",
            width: 512,
            height: 512,
            steps: 8,
            cfg: 4.0,
            seed: 7,
            references: &references,
        };
        let started = std::time::Instant::now();
        let got = models.generate(&request, |_| true).unwrap();
        println!("edit {:.2} s", started.elapsed().as_secs_f64());
        let (want, ..) = read_png(&shots.join("qedit512.png"));
        let mut error = 0f64;
        for (a, b) in got.chunks_exact(4).zip(want.chunks_exact(4)) {
            for c in 0..3 {
                error += (f64::from(a[c]) - f64::from(b[c])).powi(2);
            }
        }
        let psnr = 10.0 * (255.0f64 * 255.0 / (error / (512.0 * 512.0 * 3.0))).log10();
        println!("RGB PSNR against the engine: {psnr:.1} dB");
        let mut encoder = png::Encoder::new(
            std::fs::File::create(shots.join("rust-edit.png")).unwrap(),
            512,
            512,
        );
        encoder.set_color(png::ColorType::Rgba);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&got)
            .unwrap();
        assert!(psnr > 20.0);
    }

    /// `KUNO_QWEN_MODELS=<dir> KUNO_QWEN_EDIT=<dir with ref-cat-256.png>
    /// KUNO_QWEN_GOLDEN=<the qwen-edit capture>`: the conditioning of the captured edit (prompt
    /// and empty negative, one 256 x 256 reference) against the engine's DiT inputs.
    #[test]
    #[ignore]
    fn edit_conditioning_matches_cpp_engine() {
        use super::super::qwen_dit::tests::{Golden, db};
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_MODELS").unwrap());
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let models = Models::open(
            &gpu,
            &dir.join("qwen3vl_8b_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_int8_convrot.safetensors"),
            &dir.join("qwen_image_2.1_vae_bf16.safetensors"),
        )
        .unwrap();
        let shots = std::path::PathBuf::from(std::env::var("KUNO_QWEN_EDIT").unwrap());
        let (rgba, width, height) = read_png(&shots.join("ref-cat-256.png"));
        let encoded = models
            .encode_reference(&Reference {
                rgba: &rgba,
                width,
                height,
            })
            .unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_QWEN_GOLDEN").unwrap());
        for (step, text) in [(0, "Make the cat wear a small red hat"), (1, "")] {
            let want = Golden::load(&golden, step);
            let (context, slots) = models
                .condition(text, std::slice::from_ref(&encoded))
                .unwrap();
            assert_eq!(slots, want.slots, "slots of step {step}");
            println!(
                "step {step}: context {:.1} dB, reference latent {:.1} dB",
                db(&want.context, &context),
                db(&want.references[0].0, &encoded.latent)
            );
            let n = context.len() / CONTEXT;
            let text_rows = |c: &[f32]| -> Vec<f32> {
                (0..n)
                    .filter(|t| slots[*t] == 0)
                    .flat_map(|t| c[t * CONTEXT..(t + 1) * CONTEXT].to_vec())
                    .collect()
            };
            println!(
                "  text rows only: {:.1} dB",
                db(&text_rows(&want.context), &text_rows(&context))
            );
            let first = slots.iter().position(|s| *s != 0).unwrap();
            println!(
                "  the {first} rows before the image: {:.1} dB",
                db(
                    &want.context[..first * CONTEXT],
                    &context[..first * CONTEXT]
                )
            );
        }
        // A text-only prompt: the plain generation capture.
        let small = golden.parent().unwrap().join("qwen-small");
        let want = Golden::load(&small, 0);
        let (context, _) = models
            .condition(
                "A red fox sitting on a mossy rock in a misty forest, morning light",
                &[],
            )
            .unwrap();
        println!("text only: context {:.1} dB", db(&want.context, &context));
        let mut rows: Vec<f64> = want
            .context
            .chunks_exact(CONTEXT)
            .zip(context.chunks_exact(CONTEXT))
            .map(|(a, b)| db(a, b))
            .collect();
        println!(
            "  per row: {:?}",
            rows.iter()
                .map(|v| (v * 10.0).round() / 10.0)
                .collect::<Vec<_>>()
        );
        rows.sort_by(f64::total_cmp);
        println!(
            "  per-row median {:.1} dB, worst {:.1} dB",
            rows[rows.len() / 2],
            rows[0]
        );
    }
}
