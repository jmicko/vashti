//! MiniMax H3 video generation in Rust, without the C++ engine.
//!
//! This is the engine's `generate_video` for the requests Kuno makes, step for step: the
//! start, end and reference images become noised latents and pictures in the prompt; the
//! prompt goes through the tokenizer, the vision tower and the text encoder; an Euler loop
//! on the discrete flow schedule (shift 12) runs the DiT; the audio latent and the video
//! frames are decoded. Every model is the worker's existing Rust one; the arithmetic of the
//! glue (noise, schedule, Euler, latent scaling, previews, frame conversion) follows the
//! engine's so that both produce the same clip.
use std::ffi::c_void;

use super::{
    VideoOutput, decode_video_latent, frame_encoder, placed_dit, receive_audio_latent,
    text_encoder, video_phase, video_preview, vision,
};
use crate::engine::{
    dit::{self, AUDIO_CHANNELS, Reference, ReferenceKind, TEXT_WIDTH, VIDEO_CHANNELS, Video},
    rng::{Mt19937, Philox},
    text_encoder::{Image, Prompt},
    tokenizer::Tokenizer,
    video_tiles,
    video_vae::LATENT_CHANNELS,
};

unsafe extern "C" {
    fn kuno_qwen_session_release(session: *mut c_void);
    fn kuno_image_cancel_requested() -> bool;
}

/// The engine's flow shift for MiniMax H3 video, and the audio stream's.
const VIDEO_SHIFT: f32 = 12.0;
const AUDIO_SHIFT: f32 = 3.0;
/// The share of a conditioning latent kept when it is noised.
const VISUAL_COND_TIMESTEP: f32 = 0.999;
/// Pixels per vision-tower row on each side, and its area limits (Qwen3-VL's processor).
const VISION_FACTOR: usize = 32;
const VISION_MIN_AREA: usize = 3136;
const VISION_MAX_AREA: usize = 12_845_056;
// The engine's table, digit for digit.
#[allow(clippy::excessive_precision)]
const PREVIEW_PROJECTION: [[f32; 3]; 24] = [
    [0.198_198_57, 0.115_849_99, 0.079_297_77],
    [-0.160_472_24, -0.106_011_7, -0.159_963_24],
    [0.473_919_51, 0.376_024_75, 0.202_678_26],
    [-0.098_574_41, -0.274_354_49, -0.516_817_5],
    [-0.189_306_05, -0.105_122_78, -0.285_714_78],
    [-0.156_395_69, -0.180_009_29, -0.254_328_52],
    [-0.071_769_21, -0.109_015_98, -0.066_542_53],
    [-0.050_140_77, -0.058_398_26, -0.055_160_87],
    [-0.052_014_24, -0.043_519_13, -0.015_075_79],
    [0.247_504_38, 0.133_074_22, 0.176_841_2],
    [0.073_774_46, 0.102_358_58, 0.117_078_27],
    [0.029_083_04, 0.065_870_22, 0.106_436_9],
    [-0.006_705_31, -0.038_578_79, 0.017_501_51],
    [-0.071_191_07, -0.030_833_23, -0.019_954_5],
    [-0.086_126_27, -0.072_538_41, -0.014_428_9],
    [0.087_932_02, 0.086_817_5, 0.029_946_47],
    [0.008_768_93, 0.027_218_68, 0.000_911_78],
    [-0.034_844_12, -0.027_112_62, -0.001_101_01],
    [-0.006_797_72, -0.018_442_75, -0.016_833_59],
    [0.042_870_28, 0.016_010_68, 0.040_373_97],
    [-0.004_934_32, -0.002_305_28, 0.003_539_11],
    [0.014_950_88, 0.002_923_06, 0.004_166_71],
    [0.004_953_07, 0.050_665_42, 0.052_105_43],
    [-0.021_548_42, -0.015_185_24, 0.004_424_02],
];
const PREVIEW_BIAS: [f32; 3] = [0.077_769_64, -0.015_809_54, -0.065_614_34];

/// One request as Kuno makes it; images are RGB, row by row.
pub(super) struct Request<'a> {
    pub prompt: &'a str,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: usize,
    pub seed: i64,
    /// Start and end frames with their sizes (empty when not given): any shape, each side
    /// a multiple of 16.
    pub first: (&'a [u8], (u32, u32)),
    pub last: (&'a [u8], (u32, u32)),
    /// Reference pictures at their own sizes.
    pub references: &'a [(u32, u32, Vec<u8>)],
}

#[derive(Debug)]
pub(super) enum Failure {
    Cancelled,
    Failed(&'static str),
}
impl From<&'static str> for Failure {
    fn from(code: &'static str) -> Self {
        Self::Failed(code)
    }
}

/// An image as the engine holds it: `[3][height][width]`, values 0..1.
struct Picture {
    width: usize,
    height: usize,
    values: Vec<f32>,
}
impl Picture {
    fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Result<Self, &'static str> {
        if rgb.len() != width * height * 3 || width == 0 || height == 0 {
            return Err("video_input_invalid");
        }
        let plane = width * height;
        let mut values = vec![0f32; 3 * plane];
        for (p, pixel) in rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                values[c * plane + p] = f32::from(pixel[c]) / 255.0;
            }
        }
        Ok(Self {
            width,
            height,
            values,
        })
    }
    /// The engine's default (nearest) resize: source index `i * in / out` on each axis.
    fn resized(&self, width: usize, height: usize) -> Self {
        if (width, height) == (self.width, self.height) {
            return Self {
                width,
                height,
                values: self.values.clone(),
            };
        }
        let mut values = Vec::with_capacity(3 * width * height);
        for c in 0..3 {
            for y in 0..height {
                let sy = y * self.height / height;
                for x in 0..width {
                    let sx = x * self.width / width;
                    values.push(self.values[(c * self.height + sy) * self.width + sx]);
                }
            }
        }
        Self {
            width,
            height,
            values,
        }
    }
}

/// The host's available memory in bytes, when the worker may read it.
fn available_memory() -> Option<usize> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

fn cancelled() -> bool {
    // SAFETY: reads the shim's signal flag.
    unsafe { kuno_image_cancel_requested() }
}

/// Latent frames for a clip's frame count.
pub(super) fn latent_frames(frames: usize) -> usize {
    if frames <= 5 {
        2
    } else {
        (frames - 5) / 17 * 5 + 2
    }
}
/// The discrete flow schedule: `steps` sigmas from 1 down, then 0.
pub(super) fn sigmas(steps: usize) -> Vec<f32> {
    let shifted = |t: f32| {
        let u = (t + 1.0) / 1000.0;
        VIDEO_SHIFT * u / (1.0 + (VIDEO_SHIFT - 1.0) * u)
    };
    let mut out = Vec::with_capacity(steps + 1);
    match steps {
        0 => return out,
        1 => out.push(shifted(999.0)),
        n => {
            let step = 999.0f32 / (n - 1) as f32;
            for i in 0..n {
                out.push(shifted(999.0 - step * i as f32));
            }
        }
    }
    out.push(0.0);
    out
}
/// The vision tower's input size for an image (the engine's resize_for_vision with
/// Qwen3-VL's area limits).
fn vision_size(width: usize, height: usize) -> (usize, usize) {
    let f = VISION_FACTOR as f64;
    let (w, h) = (width as f64, height as f64);
    let mut h_bar = ((h / f).round() as usize * VISION_FACTOR).max(VISION_FACTOR);
    let mut w_bar = ((w / f).round() as usize * VISION_FACTOR).max(VISION_FACTOR);
    let area = (h_bar * w_bar) as f64;
    if area > VISION_MAX_AREA as f64 {
        let beta = (h * w / VISION_MAX_AREA as f64).sqrt();
        h_bar = ((h / beta / f).floor() as usize * VISION_FACTOR).max(VISION_FACTOR);
        w_bar = ((w / beta / f).floor() as usize * VISION_FACTOR).max(VISION_FACTOR);
    } else if area < VISION_MIN_AREA as f64 {
        let beta = (VISION_MIN_AREA as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() as usize * VISION_FACTOR;
        w_bar = (w * beta / f).ceil() as usize * VISION_FACTOR;
    }
    (w_bar, h_bar)
}
/// A reference picture's size for the video encoder: at most the clip's area, multiples
/// of 32.
fn reference_size(width: usize, height: usize, clip: (usize, usize)) -> (usize, usize) {
    let source = (width * height) as f64;
    let target = (clip.0 * clip.1) as f64;
    let scale = (target / source).sqrt().min(1.0);
    let side = |n: usize| ((n as f64 * scale / 32.0).round() as usize * 32).max(32);
    (side(width), side(height))
}

/// A conditioning latent: the video encoder's means at the diffusion model's scale, mostly
/// kept and slightly noised (the engine seeds a fresh generator for each).
fn condition_latent(
    out: &VideoOutput,
    picture: &Picture,
    seed: i64,
) -> Result<Vec<f32>, &'static str> {
    let encoder = frame_encoder(&out.pci, &out.video_model)?;
    let means = encoder.encode(&picture.values, picture.width, picture.height)?;
    let plane = means.len() / LATENT_CHANNELS;
    let noise = Philox::new(seed as u64).randn(means.len());
    let keep = 1.0f32 - VISUAL_COND_TIMESTEP;
    Ok(means
        .iter()
        .enumerate()
        .zip(&noise)
        .map(|((i, &m), &n)| {
            let c = i / plane;
            let normalized = (m - dit::LATENT_MEAN[c]) / dit::LATENT_STD[c];
            normalized * VISUAL_COND_TIMESTEP + n * keep
        })
        .collect())
}

/// The prompt's pictures, in the order they are presented.
struct Seen {
    rows: [Vec<f32>; 4],
    index: usize,
    size: usize,
    grid_h: usize,
    grid_w: usize,
}

pub(super) fn generate(
    native: *mut c_void,
    request: &Request,
    out: &mut VideoOutput,
) -> Result<(), Failure> {
    // The engine's models (an earlier image job's, say) leave video memory first.
    // SAFETY: the session is live and the caller holds the engine lock.
    unsafe { kuno_qwen_session_release(native) };
    let (width, height, frames, steps) =
        (request.width, request.height, request.frames, request.steps);
    if width % 16 != 0 || height % 16 != 0 || frames < 5 || steps == 0 {
        return Err(Failure::Failed("video_input_invalid"));
    }
    let (w, h, t) = (width / 16, height / 16, latent_frames(frames));
    // Kuno sets no size or duration limit; a clip whose decoded frames do not fit in host
    // memory is refused now, before any GPU work (the decoder's float frames and ours,
    // with room to spare).
    let decoded_bytes = 3 * video_tiles::output_frames(t) * width * height * 4;
    if let Some(available) = available_memory()
        && 2 * decoded_bytes + (1 << 30) > available
    {
        return Err(Failure::Failed("image_host_memory_unavailable"));
    }
    let audio_length =
        ((frames as f64 * 40.0 / f64::from(kuno_protocol::VIDEO_FPS)).round() as usize).max(1);

    // Conditioning images: start and end frames, or reference pictures.
    let mut pictures = Vec::new();
    let mut latents = Vec::new();
    let mut keyframes = Vec::new();
    let mut references = Vec::new();
    if request.references.is_empty() {
        for ((rgb, (fw, fh)), index) in [(request.first, 0), (request.last, frames as i32 - 1)] {
            if rgb.is_empty() {
                continue;
            }
            let (fw, fh) = (fw as usize, fh as usize);
            if fw % 16 != 0 || fh % 16 != 0 {
                return Err(Failure::Failed("video_input_invalid"));
            }
            // A frame of another shape enters on its own grid; the model composes it into
            // the clip.
            let picture = Picture::from_rgb(rgb, fw, fh)?;
            latents.push((
                condition_latent(out, &picture, request.seed)?,
                fw / 16,
                fh / 16,
            ));
            keyframes.push(index);
            pictures.push(picture);
        }
    } else {
        for (i, (rw, rh, rgb)) in request.references.iter().enumerate() {
            let source = Picture::from_rgb(rgb, *rw as usize, *rh as usize)?;
            let (pw, ph) = reference_size(source.width, source.height, (width, height));
            let picture = source.resized(pw, ph);
            latents.push((
                condition_latent(out, &picture, request.seed)?,
                pw / 16,
                ph / 16,
            ));
            references.push(Reference {
                kind: ReferenceKind::Image,
                video: Some(i),
                audio: None,
            });
            pictures.push(picture);
        }
    }
    if cancelled() {
        return Err(Failure::Cancelled);
    }

    // The prompt: each picture as the vision tower sees it, then the text.
    static TOKENIZER: std::sync::OnceLock<Result<Tokenizer, &'static str>> =
        std::sync::OnceLock::new();
    let tokenizer = TOKENIZER
        .get_or_init(Tokenizer::new)
        .as_ref()
        .map_err(|e| *e)?;
    let mut text = String::new();
    let mut seen = Vec::new();
    for (i, picture) in pictures.iter().enumerate() {
        text.push_str(&format!("<Picture {}>: ", i + 1));
        let (vw, vh) = vision_size(picture.width, picture.height);
        let mut input = picture.resized(vw, vh);
        input
            .values
            .iter_mut()
            .for_each(|v| *v = v.clamp(0.0, 1.0) * 2.0 - 1.0);
        let tower = vision(&out.pci, &out.text_model)?;
        let rows = tower.encode(&input.values, vw, vh)?;
        text.push_str("<|vision_start|>");
        let index = tokenizer.encode(&text)?.len();
        let size = (vw / VISION_FACTOR) * (vh / VISION_FACTOR);
        for _ in 0..size {
            text.push_str("<|image_pad|>");
        }
        text.push_str("<|vision_end|>");
        seen.push(Seen {
            rows,
            index,
            size,
            grid_h: vh / 16,
            grid_w: vw / 16,
        });
    }
    text.push_str(request.prompt);
    let ids: Vec<i32> = tokenizer
        .encode(&text)?
        .into_iter()
        .map(|id| id as i32)
        .collect();
    crate::worker_protocol::clear_bytes(&mut text.into_bytes());
    let context = {
        let images = seen
            .iter()
            .map(|s| Image {
                index: s.index,
                size: s.size,
                grid_h: s.grid_h,
                grid_w: s.grid_w,
                embeds: &s.rows[0],
                deepstack: s.rows[1..].iter().map(Vec::as_slice).collect(),
            })
            .collect();
        let prompt = Prompt { ids: &ids, images };
        prompt.check()?;
        let started = std::time::Instant::now();
        let encoder = text_encoder(&out.pci, &out.text_model)?;
        let loaded = started.elapsed();
        let hidden = encoder.forward(&prompt)?;
        eprintln!(
            "kuno-image: rust text: setup {} ms, encode {} ms",
            loaded.as_millis(),
            (started.elapsed() - loaded).as_millis()
        );
        hidden
    };
    // Text tokens are tagged 1; each picture's rows and its two markers 0.
    let tokens = context.len() / TEXT_WIDTH;
    let mut tags = vec![1i32; tokens];
    for s in &seen {
        let begin = s.index.saturating_sub(1);
        let end = (s.index + s.size + 1).min(tokens);
        tags[begin..end].iter_mut().for_each(|t| *t = 0);
    }
    drop(seen);
    if cancelled() {
        return Err(Failure::Cancelled);
    }

    // The packed latent: 24 video channels, then the audio latent in further channels.
    let spatial = w * h * t;
    let video_values = spatial * VIDEO_CHANNELS;
    let audio_values = audio_length * 2 * AUDIO_CHANNELS;
    let channels = VIDEO_CHANNELS + audio_values.div_ceil(spatial);
    // The first sigma is 1, so the starting latent is exactly the noise.
    let mut x = Mt19937::new(request.seed as u64).randn(spatial * channels);
    let schedule = sigmas(steps);
    let videos: Vec<Video> = latents
        .iter()
        .map(|(data, lw, lh)| Video {
            data,
            width: *lw,
            height: *lh,
            frames: 1,
        })
        .collect();
    let mut stage = 0;
    for i in 0..steps {
        if cancelled() {
            return Err(Failure::Cancelled);
        }
        let sigma = schedule[i];
        let started = std::time::Instant::now();
        let network = placed_dit(out)?;
        let velocity = network.forward(&dit::Step {
            packed: &x,
            width: w,
            height: h,
            frames: t,
            audio_length,
            timestep: sigma * 1000.0,
            context: &context,
            tags: &tags,
            videos: &videos,
            audios: &[],
            keyframes: &keyframes,
            references: &references,
            video_shift: VIDEO_SHIFT,
            audio_shift: AUDIO_SHIFT,
        })?;
        eprintln!(
            "kuno-image: rust dit: step {} ms",
            started.elapsed().as_millis()
        );
        // The engine's Euler step through its denoised estimate.
        let change = schedule[i + 1] - sigma;
        let mut denoised = vec![0f32; x.len()];
        for ((x, d), v) in x.iter_mut().zip(&mut denoised).zip(&velocity) {
            *d = v * -sigma + *x;
            let slope = (*x - *d) / sigma;
            *x += slope * change;
        }
        // Progress, then the preview of the denoised middle frame.
        let step = i as i32 + 1;
        if step > stage {
            stage = step;
            video_phase(1, step, steps as i32, out as *mut _ as *mut c_void);
            if i + 1 == steps {
                video_phase(2, 0, 0, out as *mut _ as *mut c_void);
            }
        }
        let preview = preview_rgb(&denoised[..video_values], w, h, t, t / 2);
        video_preview(
            step,
            preview.as_ptr(),
            w as u32,
            h as u32,
            3,
            out as *mut _ as *mut c_void,
        );
    }
    drop(videos);
    drop((context, tags, latents));
    // The audio stream leaves the video schedule's scale.
    let scale = AUDIO_SHIFT / VIDEO_SHIFT;
    x[video_values..].iter_mut().for_each(|v| *v *= scale);
    // Audio first, so it decodes on the CPU while the GPU decodes video.
    let data = out as *mut _ as *mut c_void;
    receive_audio_latent(
        x[video_values..].as_ptr(),
        audio_length as i64,
        2,
        AUDIO_CHANNELS as i64,
        data,
    );
    if cancelled() {
        return Err(Failure::Cancelled);
    }
    let latent = dit::decoder_latent(&x[..video_values]);
    x.iter_mut().for_each(|v| *v = 0.0);
    let pixel_frames = video_tiles::output_frames(t);
    let (pw, ph) = (w * 16, h * 16);
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(3 * pixel_frames * pw * ph)
        .map_err(|_| Failure::Failed("image_host_memory_unavailable"))?;
    pixels.resize(3 * pixel_frames * pw * ph, 0f32);
    if !decode_video_latent(
        latent.as_ptr(),
        w as i64,
        h as i64,
        t as i64,
        pixels.as_mut_ptr(),
        pixel_frames as i64,
        data,
    ) {
        return Err(Failure::Failed("video_decode_failed"));
    }
    if pixel_frames < frames || (pw, ph) != (width, height) {
        return Err(Failure::Failed("video_output_invalid"));
    }
    // Frames as the engine converts them: clamped, rounded to 8 bits.
    let plane = pw * ph;
    let to_u8 = |v: f32| -> u8 {
        if v <= 0.0 {
            0
        } else if v >= 1.0 {
            255
        } else {
            (v * 255.0 + 0.5) as u8
        }
    };
    out.rgb.clear();
    for f in 0..frames {
        for p in 0..plane {
            for c in 0..3 {
                out.rgb
                    .push(to_u8(pixels[(c * pixel_frames + f) * plane + p]));
            }
        }
    }
    pixels.iter_mut().for_each(|v| *v = 0.0);
    Ok(())
}

/// The engine's latent preview of one frame: a fixed projection of the 24 channels to RGB,
/// row by row.
fn preview_rgb(latent: &[f32], w: usize, h: usize, t: usize, frame: usize) -> Vec<u8> {
    let plane = w * h;
    let mut rgb = vec![0u8; plane * 3];
    for p in 0..plane {
        let mut sum = [0f32; 3];
        for (c, row) in PREVIEW_PROJECTION.iter().enumerate() {
            let value = latent[(c * t + frame) * plane + p];
            for (s, weight) in sum.iter_mut().zip(row) {
                *s += value * weight;
            }
        }
        for (k, s) in sum.iter().enumerate() {
            let v = ((s + PREVIEW_BIAS[k]) * 0.5 + 0.5).clamp(0.0, 1.0);
            rgb[p * 3 + k] = (v * 255.0) as u8;
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_and_sizes_follow_the_engine() {
        let s = sigmas(8);
        assert_eq!(s.len(), 9);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[8], 0.0);
        assert!(s.windows(2).all(|p| p[0] > p[1]));
        assert_eq!(latent_frames(56), 17);
        assert_eq!(video_tiles::output_frames(17), 56);
        assert_eq!(vision_size(832, 544), (832, 544));
        assert_eq!(vision_size(830, 540), (832, 544));
        assert_eq!(reference_size(1024, 1024, (832, 544)), (672, 672));
        assert_eq!(reference_size(400, 300, (832, 544)), (416, 288));
    }
}
