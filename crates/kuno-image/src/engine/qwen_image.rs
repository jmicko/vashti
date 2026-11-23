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

/// The engine's preview projection of a latent cell to RGB and alpha (`latent-preview.h`).
#[allow(clippy::excessive_precision)]
const PREVIEW_RGB: [[f32; 3]; 64] = [
    [0.00860495522, 0.01219501462, -0.00321337196],
    [0.01889233090, 0.01246581216, 0.01074959482],
    [0.1255941446, 0.1176879344, -0.0332212352],
    [0.0418238528, 0.1043427754, 0.0121666316],
    [0.02025338, 0.01453670296, -0.000224336044],
    [-0.01896720702, -0.0206099030, -0.0322728584],
    [0.00438984796, -0.01374969766, 0.02849196],
    [-0.0374495856, -0.0286777126, -0.0693192810],
    [0.01511914734, 0.0242979386, 0.0553878870],
    [-0.1138629518, -0.020391466, 0.001550520522],
    [-0.0233650696, -0.0417292018, -0.0362361182],
    [-0.0351603342, -0.0243595924, -0.00216261038],
    [0.01093355288, -0.0373466924, 0.00241315350],
    [0.01778704744, -0.00401984678, -0.0343259192],
    [0.0486059334, 0.0253144, 0.0672564966],
    [0.0309463558, 0.0277963166, 0.0520869622],
    [0.0374485008, 0.0551753676, 0.0225853902],
    [-0.0090809962, -0.004756176, 0.00636443612],
    [-0.0270455652, -0.0384966954, -0.00905908082],
    [-0.00553493756, 0.01484553684, -0.0211502468],
    [0.01319502562, 0.00948005666, 0.0483789212],
    [-0.00931847104, -0.00276452734, -0.01011985302],
    [0.0180478258, 0.01614954356, -0.0209424690],
    [-0.0214530434, -0.00272961176, 0.0217887476],
    [-0.0636772304, -0.0208893548, 0.0479167742],
    [-0.0250321236, -0.0286715676, 0.0530110146],
    [-0.01853078078, 0.01647272818, -0.00207747588],
    [0.0023101082, 0.01228800748, 0.01303505006],
    [-0.01243671408, -0.0258638728, -0.0379116264],
    [0.00598934710, 0.00642563550, -0.01234514304],
    [-0.0296733996, -0.0234698050, 0.00060018212],
    [-0.0322019498, -0.0529200462, -0.00344987414],
    [-0.00205026458, -0.00846599446, 0.00455971038],
    [-0.01082227064, 0.0315661948, -0.0677753362],
    [0.0645553474, 0.1109666998, 0.0674744864],
    [0.01036801108, -0.00484841210, -0.001529168474],
    [0.01264353566, 0.01548126338, -0.00966374324],
    [-0.0223892408, -0.00871751526, -0.000306421670],
    [0.0271322742, 0.03496524, -0.0089692858],
    [0.0512178672, 0.0173080034, 0.00804227746],
    [0.01210987192, 0.00758025926, -0.00281712586],
    [0.1897278390, 0.1210261828, 0.062603892],
    [0.0208058822, 0.00547548182, 0.01262955638],
    [0.00813332858, 0.01015930914, 0.01301771290],
    [-0.000927236014, -0.00152540594, -0.00599213302],
    [0.01663314616, -0.00582789626, 0.0163958132],
    [-0.0252546342, -0.0604193732, -0.1606919922],
    [-0.091722686, -0.0409201224, -0.0959576198],
    [0.0282963112, -0.01387223872, -0.01648814464],
    [0.0552316818, 0.0967547788, 0.0413586632],
    [0.00922849292, 0.00451467542, -0.0529172378],
    [0.0558600768, 0.0122988308, -0.01445942422],
    [0.000210660902, -0.01295958782, -0.01804761764],
    [0.0358136250, -0.0472505970, -0.1156405142],
    [-0.0506390696, -0.0471914842, 0.0349791468],
    [-0.0480143168, 0.00628389868, -0.0545163826],
    [0.0315499582, 0.0564846606, -0.0430850488],
    [-0.0362330316, -0.01267788554, 0.0061024772],
    [0.0038627542, 0.00911055916, -0.00758526008],
    [-0.0447103298, -0.00835411408, 0.01545872328],
    [-0.015006738, 0.00270612302, -0.00784361356],
    [-0.0221755048, -0.0513344748, -0.0475317424],
    [-0.01036656294, -0.00422146068, -0.0213499052],
    [0.01788952706, 0.01191944190, 0.0397205238],
];
#[allow(clippy::excessive_precision)]
const PREVIEW_RGB_BIAS: [f32; 3] = [-0.043293118, -0.02695978, -0.11986706];
#[allow(clippy::excessive_precision)]
const PREVIEW_ALPHA: [f32; 64] = [
    -0.0416241114,
    -0.00678954612,
    -0.0169095515,
    -0.0230551401,
    0.0100882595,
    0.00655586802,
    0.0401166874,
    -0.0055510216,
    0.0224234441,
    -0.0389640963,
    -0.0114492163,
    -0.00721128977,
    -0.0029064082,
    0.0150300547,
    -0.00321615308,
    -0.0498856338,
    -0.0215251401,
    0.0240220482,
    0.0117338008,
    -0.0460420624,
    0.0387872889,
    0.0131517207,
    0.0147100836,
    0.0266985286,
    0.0153097324,
    -0.0418119757,
    0.0421013917,
    0.0401724499,
    0.00972515915,
    0.011718495,
    0.0117622291,
    0.0136505134,
    -0.0350017363,
    -0.0100692606,
    -0.0131582529,
    -0.00660639315,
    0.00253611396,
    -0.0195736368,
    -0.04240184,
    0.0321299262,
    0.0106089414,
    -0.0179845306,
    -0.00806212708,
    0.0135889057,
    -0.0157393098,
    -0.0267791344,
    0.0109068534,
    0.0283931966,
    -0.0435370078,
    0.00187883536,
    -0.0108995378,
    -0.0450757676,
    -0.0699481501,
    0.0123562106,
    -0.0222592249,
    0.0216155907,
    0.0563755424,
    -0.0073379912,
    0.0160012921,
    0.0411637742,
    0.0189607258,
    -0.024025029,
    -0.0161487905,
    -0.016913203,
];
#[allow(clippy::excessive_precision)]
const PREVIEW_ALPHA_BIAS: f32 = 0.871268134;

/// A step's preview as the engine draws it: each latent cell of `denoised` (`[CHANNELS][h][w]`)
/// projected to RGBA bytes, `h * w * 4`.
pub fn preview_rgba(denoised: &[f32], height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = vec![0u8; plane * 4];
    for (cell, pixel) in out.chunks_exact_mut(4).enumerate() {
        let (mut rgb, mut a) = ([0f32; 3], 0f32);
        for d in 0..CHANNELS {
            let value = denoised[d * plane + cell];
            for (c, sum) in rgb.iter_mut().enumerate() {
                *sum += value * PREVIEW_RGB[d][c];
            }
            a += value * PREVIEW_ALPHA[d];
        }
        for c in 0..3 {
            let v = ((rgb[c] + PREVIEW_RGB_BIAS[c]) * 0.5 + 0.5).clamp(0.0, 1.0);
            pixel[c] = (v * 255.0) as u8;
        }
        pixel[3] = ((a + PREVIEW_ALPHA_BIAS).clamp(0.0, 1.0) * 255.0) as u8;
    }
    out
}

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

    /// RGBA bytes of the image, `height * width * 4`. `progress(step, denoised)` is called
    /// after each step (from 1) with that step's denoised latent `[CHANNELS][h / 16][w / 16]`;
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
        let timesteps: Vec<f32> = sigmas[..steps].iter().map(|s| s * 1000.0).collect();
        let times = self.dit.times(&timesteps);
        for step in 0..steps {
            let sigma = sigmas[step];
            let mut out = self.dit.step_at(&cond, &x, &times[step])?;
            if let Some(uncond) = &uncond {
                let negative = self.dit.step_at(uncond, &x, &times[step])?;
                for (o, n) in out.iter_mut().zip(&negative) {
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

    /// `KUNO_QWEN_MODELS=<dir>`: how long each part of opening the models takes, then a
    /// first and a second 1184 x 1776 request of two steps (the first pays for building the
    /// kernels for its size).
    #[test]
    #[ignore]
    fn open_costs() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_MODELS").unwrap());
        let clock = std::time::Instant::now();
        let lap = |what: &str, since: &mut f64| {
            let now = clock.elapsed().as_secs_f64();
            println!("{what}: {:.2} s", now - *since);
            *since = now;
        };
        let mut at = 0.0;
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        lap("gpu", &mut at);
        let encoder = dir.join("qwen3vl_8b_int8_convrot.safetensors");
        let file = TextEncoder::open(&encoder).unwrap();
        let tower = Vision::open(&encoder).unwrap();
        lap("text encoder and vision files", &mut at);
        let tokenizer = Tokenizer::new().unwrap();
        lap("tokenizer", &mut at);
        let vision = GpuVision::new(&gpu, &tower).unwrap();
        lap("vision tower", &mut at);
        let encoder = GpuTextEncoder::new(&gpu, file).unwrap();
        lap("text encoder", &mut at);
        let dit = QwenGpu::load(
            gpu.clone(),
            &dir.join("qwen_image_2.1_int8_convrot.safetensors"),
        )
        .unwrap();
        lap("dit", &mut at);
        let vae = GpuVae::new(
            &gpu,
            &QwenVae::open(&dir.join("qwen_image_2.1_vae_bf16.safetensors")).unwrap(),
        )
        .unwrap();
        lap("vae", &mut at);
        let models = Models {
            tokenizer,
            vision,
            encoder,
            dit,
            vae,
        };
        for run in 0..2 {
            let request = Request {
                prompt: "A lighthouse on a cliff at dusk",
                negative: "",
                width: 1184,
                height: 1776,
                steps: 2,
                cfg: 1.0,
                seed: 7,
                references: &[],
            };
            let mut first = true;
            models
                .generate(&request, |_, _| {
                    if first {
                        lap(&format!("run {run}: to the first step's end"), &mut at);
                        first = false;
                    }
                    true
                })
                .unwrap();
            lap(&format!("run {run}: rest"), &mut at);
        }
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
        let rgba = models.generate(&request, |_, _| true).unwrap();
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
        let got = models.generate(&request, |_, _| true).unwrap();
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
