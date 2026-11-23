//! Private subprocess entry only. Native callbacks are process-global; never
//! run concurrent generations or invoke this inside the long-lived gateway.
use kuno_protocol::ImageRecipe;

mod pipeline;
mod qwen;
use std::{
    ffi::{CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
};
#[repr(C)]
struct NativeImage {
    width: u32,
    height: u32,
    channel: u32,
    data: *mut u8,
}
unsafe extern "C" {
    fn kuno_qwen_session_new(
        directory: *const c_char,
        pci: *const c_char,
        session: *mut *mut c_void,
    ) -> i32;
    fn kuno_qwen_session_free(session: *mut c_void);
    fn kuno_qwen_session_generate(
        session: *mut c_void,
        prompt: *const c_char,
        negative: *const c_char,
        width: i32,
        height: i32,
        steps: i32,
        cfg: f32,
        shift: f32,
        seed: i64,
        version: i32,
        references: *mut NativeImage,
        reference_count: i32,
        host_available: u64,
        fast_attention: i32,
        receive: extern "C" fn(*const u8, usize, u32, *mut c_void),
        preview: Option<extern "C" fn(i32, *const u8, u32, u32, u32, *mut c_void)>,
        userdata: *mut c_void,
    ) -> i32;
}
extern "C" fn receive(bytes: *const u8, len: usize, channels: u32, data: *mut c_void) {
    // SAFETY: shim calls synchronously with a validated RGB allocation; data is
    // the live vector owned by generate. No pointer escapes the callback.
    unsafe {
        let out = &mut *(data as *mut (Vec<u8>, u32, u32));
        out.1 = channels;
        out.0
            .extend_from_slice(std::slice::from_raw_parts(bytes, len));
    }
}
extern "C" fn preview(
    step: i32,
    bytes: *const u8,
    width: u32,
    height: u32,
    channels: u32,
    data: *mut c_void,
) {
    // The synchronous shim owns the RGB/RGBA buffer until this callback returns.
    let steps = unsafe { (*(data as *const (Vec<u8>, u32, u32))).2 };
    if let Some(frame) = preview_frame(step, steps, bytes, width, height, channels) {
        let _ = crate::preview::write_frame(1, &frame);
    }
}
/// A bounded PNG preview frame: step, steps, width, height, then the PNG.
fn preview_frame(
    step: i32,
    steps: u32,
    bytes: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> Option<Vec<u8>> {
    if !matches!(channels, 3 | 4)
        || step < 0
        || width == 0
        || height == 0
        || width > 512
        || height > 512
        || bytes.is_null()
    {
        return None;
    }
    // SAFETY: the shim keeps the buffer alive for the synchronous callback.
    let rgb = unsafe {
        std::slice::from_raw_parts(bytes, width as usize * height as usize * channels as usize)
    };
    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, width, height);
    encoder.set_color(if channels == 4 {
        png::ColorType::Rgba
    } else {
        png::ColorType::Rgb
    });
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().ok()?;
    writer.write_image_data(rgb).ok()?;
    drop(writer);
    if png.len() + 16 > crate::preview::MAX_PREVIEW_BYTES {
        return None;
    }
    let mut frame = Vec::with_capacity(png.len() + 16);
    for number in [step as u32, steps, width, height] {
        frame.extend_from_slice(&number.to_be_bytes());
    }
    frame.extend_from_slice(&png);
    Some(frame)
}
/// An RGB image passed to the shim, borrowed for the duration of one call.
#[repr(C)]
struct RgbImage {
    width: u32,
    height: u32,
    data: *const u8,
}
unsafe extern "C" {
    fn kuno_minimax_session_generate(
        session: *mut c_void,
        reference_model: i32,
        prompt: *const c_char,
        width: i32,
        height: i32,
        frames: i32,
        fps: i32,
        steps: i32,
        seed: i64,
        first_frame: *const u8,
        last_frame: *const u8,
        references: *const RgbImage,
        reference_count: i32,
        receive_frame: extern "C" fn(*const u8, usize, *mut c_void),
        receive_audio_latent: extern "C" fn(*const f32, i64, i64, i64, *mut c_void),
        decode_video: Option<
            extern "C" fn(*const f32, i64, i64, i64, *mut f32, i64, *mut c_void) -> bool,
        >,
        evaluate_dit: Option<extern "C" fn(*const DitStep, *mut f32, *mut c_void) -> bool>,
        encode_text: Option<extern "C" fn(*const TextInput, *mut f32, *mut c_void) -> bool>,
        encode_frame: Option<extern "C" fn(*const f32, i64, i64, *mut f32, *mut c_void) -> bool>,
        encode_vision: Option<extern "C" fn(*const f32, i64, i64, *mut f32, *mut c_void) -> bool>,
        preview: Option<extern "C" fn(i32, *const u8, u32, u32, u32, *mut c_void)>,
        phase: extern "C" fn(i32, i32, i32, *mut c_void),
        userdata: *mut c_void,
    ) -> i32;
}
/// `sd_kuno_dit_step_t`: one evaluation of the diffusion transformer.
#[repr(C)]
struct DitStep {
    x: *const f32,
    width: i64,
    height: i64,
    frames: i64,
    channels: i64,
    timestep: f32,
    context: *const f32,
    text_tokens: i64,
    text_width: i64,
    tags: *const i32,
    tag_count: i64,
    videos: *const *const f32,
    video_shapes: *const i64,
    video_count: i64,
    audios: *const *const f32,
    audio_lengths: *const i64,
    audio_count: i64,
    keyframes: *const i32,
    keyframe_count: i64,
    references: *const i32,
    reference_count: i64,
    audio_length: i32,
    video_shift: f32,
    audio_shift: f32,
}

/// `sd_kuno_text_t`: a prompt for the text encoder's language model.
#[repr(C)]
struct TextInput {
    ids: *const i32,
    count: i64,
    width: i64,
    image_count: i64,
    image_grids: *const i32,
    images: *const *const f32,
    deepstack_layers: i64,
    deepstack: *const *const f32,
}

/// Request-owned video output, filled synchronously by the shim callbacks.
struct VideoOutput {
    frame_bytes: usize,
    frames: u32,
    steps: u32,
    rgb: Vec<u8>,
    /// The loaded audio decoder when the request wants audio.
    decoder: Option<std::sync::Arc<crate::engine::audio_vae::AudioDecoder>>,
    audio: Option<std::thread::JoinHandle<Result<crate::video::Audio, &'static str>>>,
    /// Where the Rust video decoder finds its GPU and model.
    pci: String,
    video_model: std::path::PathBuf,
    /// The Rust DiT: taken from `RESIDENT_DIT` or placed in video memory at the clip's first
    /// evaluation, and kept there for the next clip when sampling ends.
    dit_model: std::path::PathBuf,
    dit: Option<crate::engine::dit_gpu::GpuDit>,
    dit_failed: bool,
    /// The text encoder's model file.
    text_model: std::path::PathBuf,
    invalid: bool,
    previews: bool,
}
impl Drop for VideoOutput {
    fn drop(&mut self) {
        crate::worker_protocol::clear_bytes(&mut self.rgb);
    }
}
extern "C" fn receive_video_frame(bytes: *const u8, len: usize, data: *mut c_void) {
    // SAFETY: the shim calls synchronously with a live frame of `len` bytes; data is the
    // VideoOutput owned by generate_video. No pointer escapes the callback.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    if bytes.is_null()
        || len != out.frame_bytes
        || out.rgb.len() >= out.frame_bytes * out.frames as usize
    {
        out.invalid = true;
        return;
    }
    out.rgb
        .extend_from_slice(unsafe { std::slice::from_raw_parts(bytes, len) });
}
/// The engine hands over the audio latent before it decodes the video frames, so the Rust
/// audio decoder runs on CPU threads while the GPU decodes video.
extern "C" fn receive_audio_latent(
    latent: *const f32,
    frames: i64,
    streams: i64,
    channels: i64,
    data: *mut c_void,
) {
    use crate::engine::audio_vae::{LATENT_CHANNELS, STREAMS};
    // SAFETY: as receive_video_frame; the latent holds frames * streams * channels values.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    if latent.is_null()
        || out.audio.is_some()
        || !(1..=4096).contains(&frames)
        || streams != STREAMS as i64
        || channels != LATENT_CHANNELS as i64
    {
        out.invalid = true;
        return;
    }
    let Some(decoder) = out.decoder.clone() else {
        // Audio was not requested: nothing to decode.
        return;
    };
    let frames = frames as usize;
    let mut latent =
        unsafe { std::slice::from_raw_parts(latent, frames * STREAMS * LATENT_CHANNELS) }.to_vec();
    out.audio = Some(std::thread::spawn(move || {
        let samples = decoder.decode(&latent, frames);
        latent.iter_mut().for_each(|v| *v = 0.0);
        Ok(crate::video::Audio {
            sample_rate: crate::engine::audio_vae::SAMPLE_RATE,
            channels: STREAMS as u32,
            samples: samples?,
        })
    }));
}
/// Decodes the video latent with the Rust GPU decoder. The decoder lives only for this
/// call, so its weights do not hold GPU memory while other models run. Returning false
/// hands the latent back to the C++ engine.
extern "C" fn decode_video_latent(
    latent: *const f32,
    width: i64,
    height: i64,
    frames: i64,
    pixels: *mut f32,
    pixel_frames: i64,
    data: *mut c_void,
) -> bool {
    use crate::engine::{video_gpu::GpuVideoDecoder, video_tiles, video_vae::LATENT_CHANNELS};
    // SAFETY: as receive_video_frame; the latent holds 24 * frames * height * width values
    // and `pixels` 3 * pixel_frames * height * 16 * width * 16.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    if latent.is_null()
        || pixels.is_null()
        || !(1..=4096).contains(&width)
        || !(1..=4096).contains(&height)
        || !(1..=4096).contains(&frames)
    {
        return false;
    }
    // Sampling is over. The DiT stays in video memory for the next clip unless the decoder
    // does not fit beside it.
    keep_dit(&out.dit_model, out.dit.take());
    let (width, height, frames) = (width as usize, height as usize, frames as usize);
    if video_tiles::output_frames(frames) as i64 != pixel_frames {
        return false;
    }
    let latent =
        unsafe { std::slice::from_raw_parts(latent, LATENT_CHANNELS * frames * width * height) };
    let started = std::time::Instant::now();
    let mut loaded = started;
    let host = video_host(&out.pci, &out.video_model);
    let decoded = host
        .clone()
        .and_then(|host| {
            GpuVideoDecoder::place(host.clone()).or_else(|_| {
                evict_dit();
                GpuVideoDecoder::place(host)
            })
        })
        .and_then(|decoder| {
            loaded = std::time::Instant::now();
            decoder.decode(latent, width, height, frames)
        });
    match decoded {
        Ok(mut decoded) => {
            eprintln!(
                "kuno-image: rust video decode: load {} ms, decode {} ms",
                (loaded - started).as_millis(),
                loaded.elapsed().as_millis()
            );
            // SAFETY: lengths match by the frame-count check above.
            unsafe { std::ptr::copy_nonoverlapping(decoded.as_ptr(), pixels, decoded.len()) };
            decoded.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        Err(code) => {
            eprintln!("kuno-image: rust video decode unavailable ({code}); engine decodes");
            false
        }
    }
}
/// The Rust video decoder's weights in host memory and the file they came from.
static VIDEO_HOST: std::sync::Mutex<
    Option<(
        std::path::PathBuf,
        std::sync::Arc<crate::engine::video_gpu::Host>,
    )>,
> = std::sync::Mutex::new(None);
/// The video decoder's weights for `model`, read into host memory unless they are there
/// already. They stay between clips; a clip only copies them into video memory.
fn video_host(
    pci: &str,
    model: &std::path::Path,
) -> Result<std::sync::Arc<crate::engine::video_gpu::Host>, &'static str> {
    use crate::engine::{gpu, video_gpu::Host};
    let mut kept = VIDEO_HOST.lock().map_err(|_| "model_state_lost")?;
    if let Some((path, host)) = kept.as_ref()
        && path == model
    {
        return Ok(host.clone());
    }
    *kept = None;
    let gpu = gpu::Gpu::open(gpu::Select::Pci(pci.to_owned()))?;
    let host = std::sync::Arc::new(Host::read(&gpu, model)?);
    *kept = Some((model.to_owned(), host.clone()));
    Ok(host)
}
/// The Rust DiT kept in video memory after a clip, for the next clip of the same model,
/// and the file it came from. It holds no request-derived state (`GpuDit::finish_clip`).
/// Its host memory copy is released once it is placed, so a DiT that has to be placed again
/// (after an image job or the other model) is read from the file again.
static RESIDENT_DIT: std::sync::Mutex<
    Option<(std::path::PathBuf, crate::engine::dit_gpu::GpuDit)>,
> = std::sync::Mutex::new(None);
/// Keeps `dit`, placed from `model`, for the next clip.
fn keep_dit(model: &std::path::Path, dit: Option<crate::engine::dit_gpu::GpuDit>) {
    let Some(dit) = dit else { return };
    dit.finish_clip();
    if let Ok(mut kept) = RESIDENT_DIT.lock() {
        *kept = Some((model.to_owned(), dit));
    }
}
/// Whether the kept DiT came from `model`.
fn dit_resident(model: &std::path::Path) -> bool {
    RESIDENT_DIT
        .lock()
        .is_ok_and(|kept| kept.as_ref().is_some_and(|(path, _)| path == model))
}
/// Frees the kept DiT's video memory.
pub(crate) fn evict_dit() {
    let kept = RESIDENT_DIT.lock().ok().and_then(|mut kept| kept.take());
    drop(kept);
}
/// Frees the Rust Qwen Image 2.1 models.
pub(crate) fn evict_qwen() {
    qwen::evict();
}
/// The kept DiT if it was placed from `model`; any other kept DiT is freed.
fn resident_dit(model: &std::path::Path) -> Option<crate::engine::dit_gpu::GpuDit> {
    let (path, dit) = RESIDENT_DIT.lock().ok()?.take()?;
    (path == model).then_some(dit)
}
/// The Rust DiT's weights in host memory and the file they came from.
static DIT_HOST: std::sync::Mutex<
    Option<(
        std::path::PathBuf,
        std::sync::Arc<crate::engine::dit_gpu::Host>,
    )>,
> = std::sync::Mutex::new(None);
/// The Rust DiT's weights for `model`, read into host memory unless they are there already.
/// They stay only until the DiT is placed (`release_dit_host`). Concurrent callers wait for
/// the one read.
fn dit_host(
    pci: &str,
    model: &std::path::Path,
) -> Result<std::sync::Arc<crate::engine::dit_gpu::Host>, &'static str> {
    use crate::engine::{dit_gpu::Host, gpu};
    let mut kept = DIT_HOST.lock().map_err(|_| "model_state_lost")?;
    if let Some((path, host)) = kept.as_ref()
        && path == model
        && host.has_weights()
    {
        return Ok(host.clone());
    }
    *kept = None;
    let started = std::time::Instant::now();
    let gpu = gpu::Gpu::open(gpu::Select::Pci(pci.to_owned()))?;
    let host = std::sync::Arc::new(Host::read(&gpu, model)?);
    *kept = Some((model.to_owned(), host.clone()));
    eprintln!(
        "kuno-image: rust dit: read {} ms",
        started.elapsed().as_millis()
    );
    Ok(host)
}
/// Frees the DiT weights' host memory once they are in video memory.
fn release_dit_host() {
    let kept = DIT_HOST.lock().ok().and_then(|mut kept| kept.take());
    if let Some((_, host)) = kept {
        host.release();
    }
}
/// The Rust text encoder and the file it reads.
static TEXT_ENCODER: std::sync::Mutex<
    Option<(
        std::path::PathBuf,
        std::sync::Arc<crate::engine::text_gpu::GpuTextEncoder>,
    )>,
> = std::sync::Mutex::new(None);
fn text_encoder(
    pci: &str,
    model: &std::path::Path,
) -> Result<std::sync::Arc<crate::engine::text_gpu::GpuTextEncoder>, &'static str> {
    use crate::engine::{gpu, text_encoder::TextEncoder, text_gpu::GpuTextEncoder};
    let mut kept = TEXT_ENCODER.lock().map_err(|_| "model_state_lost")?;
    if let Some((path, encoder)) = kept.as_ref()
        && path == model
    {
        return Ok(encoder.clone());
    }
    *kept = None;
    let gpu = gpu::Gpu::open(gpu::Select::Pci(pci.to_owned()))?;
    let file = TextEncoder::open(model)?;
    // The video pipeline's vision tower and hooks assume MiniMax H3's encoder.
    if file.config != crate::engine::text_encoder::Config::MINIMAX_H3 {
        return Err("model_file_invalid");
    }
    let encoder = std::sync::Arc::new(GpuTextEncoder::new(&gpu, file)?);
    *kept = Some((model.to_owned(), encoder.clone()));
    Ok(encoder)
}
/// The Rust single-frame video encoder and the file it reads, kept in video memory (0.4 GB).
static FRAME_ENCODER: std::sync::Mutex<
    Option<(
        std::path::PathBuf,
        std::sync::Arc<crate::engine::encoder_gpu::GpuEncoder>,
    )>,
> = std::sync::Mutex::new(None);
fn frame_encoder(
    pci: &str,
    model: &std::path::Path,
) -> Result<std::sync::Arc<crate::engine::encoder_gpu::GpuEncoder>, &'static str> {
    use crate::engine::{encoder_gpu::GpuEncoder, gpu, video_encoder::VideoEncoder};
    let mut kept = FRAME_ENCODER.lock().map_err(|_| "model_state_lost")?;
    if let Some((path, encoder)) = kept.as_ref()
        && path == model
    {
        return Ok(encoder.clone());
    }
    *kept = None;
    let gpu = gpu::Gpu::open(gpu::Select::Pci(pci.to_owned()))?;
    let encoder = std::sync::Arc::new(GpuEncoder::new(&gpu, &VideoEncoder::open(model)?)?);
    *kept = Some((model.to_owned(), encoder.clone()));
    Ok(encoder)
}
/// Encodes one start, end or reference frame with the Rust GPU encoder. Returning false
/// hands the frame back to the C++ engine.
extern "C" fn encode_frame(
    pixels: *const f32,
    width: i64,
    height: i64,
    latent: *mut f32,
    data: *mut c_void,
) -> bool {
    use crate::engine::video_encoder::{SCALE, supported};
    use crate::engine::video_vae::LATENT_CHANNELS;
    // SAFETY: as receive_video_frame.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    let encoded = (|| -> Result<Vec<f32>, &'static str> {
        let (w, h) = (
            usize::try_from(width).map_err(|_| "invalid_input")?,
            usize::try_from(height).map_err(|_| "invalid_input")?,
        );
        if pixels.is_null() || latent.is_null() || !supported(w, h) || w * h > 4096 * 4096 {
            return Err("invalid_input");
        }
        // SAFETY: the shim passes one frame of 3 * width * height values.
        let pixels = unsafe { std::slice::from_raw_parts(pixels, 3 * w * h) };
        let started = std::time::Instant::now();
        let encoder = frame_encoder(&out.pci, &out.video_model)?;
        let loaded = started.elapsed();
        let values = encoder.encode(pixels, w, h)?;
        eprintln!(
            "kuno-image: rust frame encode: setup {} ms, encode {} ms",
            loaded.as_millis(),
            (started.elapsed() - loaded).as_millis()
        );
        debug_assert_eq!(values.len(), LATENT_CHANNELS * (w / SCALE) * (h / SCALE));
        Ok(values)
    })();
    match encoded {
        Ok(mut values) => {
            // SAFETY: `latent` holds 24 * (width / 16) * (height / 16) values, as `encode` returns.
            unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), latent, values.len()) };
            values.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        Err(code) => {
            eprintln!("kuno-image: rust frame encoder unavailable ({code}); engine encodes");
            false
        }
    }
}
/// The Rust vision tower and the file it reads, kept in video memory (1.2 GB).
static VISION: std::sync::Mutex<
    Option<(
        std::path::PathBuf,
        std::sync::Arc<crate::engine::vision_gpu::GpuVision>,
    )>,
> = std::sync::Mutex::new(None);
fn vision(
    pci: &str,
    model: &std::path::Path,
) -> Result<std::sync::Arc<crate::engine::vision_gpu::GpuVision>, &'static str> {
    use crate::engine::{gpu, vision::Vision, vision_gpu::GpuVision};
    let mut kept = VISION.lock().map_err(|_| "model_state_lost")?;
    if let Some((path, tower)) = kept.as_ref()
        && path == model
    {
        return Ok(tower.clone());
    }
    *kept = None;
    let gpu = gpu::Gpu::open(gpu::Select::Pci(pci.to_owned()))?;
    let tower = std::sync::Arc::new(GpuVision::new(&gpu, &Vision::open(model)?)?);
    *kept = Some((model.to_owned(), tower.clone()));
    Ok(tower)
}
/// Runs the vision tower on one image the engine resized for it. Returning false hands the
/// image back to the C++ engine.
extern "C" fn encode_vision(
    pixels: *const f32,
    width: i64,
    height: i64,
    out_rows: *mut f32,
    data: *mut c_void,
) -> bool {
    use crate::engine::vision::{OUT, PATCH, supported};
    use crate::engine::vision_gpu::MAX_PATCHES;
    // SAFETY: as receive_video_frame.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    let encoded = (|| -> Result<[Vec<f32>; 4], &'static str> {
        let (w, h) = (
            usize::try_from(width).map_err(|_| "invalid_input")?,
            usize::try_from(height).map_err(|_| "invalid_input")?,
        );
        if pixels.is_null()
            || out_rows.is_null()
            || !supported(w, h)
            || (w / PATCH) * (h / PATCH) > MAX_PATCHES
        {
            return Err("invalid_input");
        }
        // SAFETY: the shim passes one image of 3 * width * height values.
        let pixels = unsafe { std::slice::from_raw_parts(pixels, 3 * w * h) };
        let started = std::time::Instant::now();
        let tower = vision(&out.pci, &out.text_model)?;
        let loaded = started.elapsed();
        let sets = tower.encode(pixels, w, h)?;
        eprintln!(
            "kuno-image: rust vision: setup {} ms, encode {} ms",
            loaded.as_millis(),
            (started.elapsed() - loaded).as_millis()
        );
        debug_assert!(sets.iter().all(|s| s.len() == (w / 32) * (h / 32) * OUT));
        Ok(sets)
    })();
    match encoded {
        Ok(mut sets) => {
            let mut at = 0;
            for set in &mut sets {
                // SAFETY: `out_rows` holds four sets of (width / 32) * (height / 32) * 5120
                // values, the lengths `encode` returns.
                unsafe { std::ptr::copy_nonoverlapping(set.as_ptr(), out_rows.add(at), set.len()) };
                at += set.len();
                set.iter_mut().for_each(|v| *v = 0.0);
            }
            true
        }
        Err(code) => {
            eprintln!("kuno-image: rust vision unavailable ({code}); engine encodes");
            false
        }
    }
}
/// Runs the text encoder's language model with the Rust GPU encoder. Returning false hands
/// the prompt back to the C++ engine.
extern "C" fn encode_text(text: *const TextInput, hidden: *mut f32, data: *mut c_void) -> bool {
    use crate::engine::text_encoder::{Config, Image, Prompt};
    const WIDTH: usize = Config::MINIMAX_H3.width;
    // SAFETY: as receive_video_frame; the shim passes a live prompt whose pointers hold the
    // counts it states, and `hidden` has `count * width` values.
    let (out, text) = unsafe { (&mut *(data as *mut VideoOutput), &*text) };
    let encoded = (|| -> Result<Vec<f32>, &'static str> {
        let invalid = "text_invalid";
        if text.ids.is_null()
            || hidden.is_null()
            || text.width != WIDTH as i64
            || !(1..=1 << 16).contains(&text.count)
            || !(0..=64).contains(&text.image_count)
            || !(0..=8).contains(&text.deepstack_layers)
        {
            return Err(invalid);
        }
        let images = text.image_count as usize;
        if images > 0 && (text.image_grids.is_null() || text.images.is_null()) {
            return Err(invalid);
        }
        if images > 0 && text.deepstack_layers > 0 && text.deepstack.is_null() {
            return Err(invalid);
        }
        // SAFETY: lengths as stated by the shim and checked above.
        let ids = unsafe { std::slice::from_raw_parts(text.ids, text.count as usize) };
        let grids = if images == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(text.image_grids, images * 4) }
        };
        let mut list = Vec::with_capacity(images);
        for (i, grid) in grids.chunks_exact(4).enumerate() {
            let size = usize::try_from(grid[1]).map_err(|_| invalid)?;
            if size == 0 || size > ids.len() || grid[0] < 0 || grid[2] < 0 || grid[3] < 0 {
                return Err(invalid);
            }
            let rows = |at: *const f32| -> Result<&[f32], &'static str> {
                if at.is_null() {
                    return Err(invalid);
                }
                // SAFETY: each image's rows hold `size * width` values.
                Ok(unsafe { std::slice::from_raw_parts(at, size * WIDTH) })
            };
            let embeds = rows(unsafe { *text.images.add(i) })?;
            let deepstack = (0..text.deepstack_layers as usize)
                .map(|layer| rows(unsafe { *text.deepstack.add(layer * images + i) }))
                .collect::<Result<Vec<_>, _>>()?;
            list.push(Image {
                index: grid[0] as usize,
                size,
                grid_h: grid[2] as usize,
                grid_w: grid[3] as usize,
                embeds,
                deepstack,
            });
        }
        let prompt = Prompt { ids, images: list };
        prompt.check(&Config::MINIMAX_H3)?;
        let started = std::time::Instant::now();
        let encoder = text_encoder(&out.pci, &out.text_model)?;
        let loaded = started.elapsed();
        let hidden = encoder.forward(&prompt)?;
        eprintln!(
            "kuno-image: rust text: setup {} ms, encode {} ms",
            loaded.as_millis(),
            (started.elapsed() - loaded).as_millis()
        );
        Ok(hidden)
    })();
    match encoded {
        Ok(mut values) => {
            // SAFETY: `hidden` holds `count * width` values, the length `forward` returns.
            unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), hidden, values.len()) };
            values.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        Err(code) => {
            eprintln!("kuno-image: rust text unavailable ({code}); engine encodes");
            false
        }
    }
}
/// Evaluates the diffusion transformer with the Rust GPU DiT. Returning false hands the
/// step back to the C++ engine, for this and every later step of the clip.
/// The clip's DiT: the kept one, or this request's read placed in video memory.
fn placed_dit(out: &mut VideoOutput) -> Result<&crate::engine::dit_gpu::GpuDit, &'static str> {
    if out.dit.is_none() {
        out.dit = resident_dit(&out.dit_model);
    }
    if out.dit.is_none() {
        let host = dit_host(&out.pci, &out.dit_model)?;
        let started = std::time::Instant::now();
        out.dit = Some(crate::engine::dit_gpu::GpuDit::place(host)?);
        release_dit_host();
        eprintln!(
            "kuno-image: rust dit: place {} ms",
            started.elapsed().as_millis()
        );
    }
    out.dit.as_ref().ok_or("gpu_failed")
}
extern "C" fn evaluate_dit(step: *const DitStep, result: *mut f32, data: *mut c_void) -> bool {
    use crate::engine::dit;
    // SAFETY: as receive_video_frame; the shim passes a live step whose pointers hold the
    // counts and shapes it states, and `result` has the packed latent's size.
    let (out, step) = unsafe { (&mut *(data as *mut VideoOutput), &*step) };
    if out.dit_failed || step.x.is_null() || step.context.is_null() || result.is_null() {
        return false;
    }
    let size = |v: i64, most: i64| (1..=most).contains(&v).then_some(v as usize);
    let count = |v: i64, most: i64| (0..=most).contains(&v).then_some(v as usize);
    let evaluated = (|| -> Result<(), &'static str> {
        let invalid = "video_input_invalid";
        let (width, height, frames, channels) = (
            size(step.width, 4096).ok_or(invalid)?,
            size(step.height, 4096).ok_or(invalid)?,
            size(step.frames, 4096).ok_or(invalid)?,
            size(step.channels, 4096).ok_or(invalid)?,
        );
        let tokens = size(step.text_tokens, 1 << 16).ok_or(invalid)?;
        if step.text_width != dit::TEXT_WIDTH as i64 {
            return Err(invalid);
        }
        let slice = |from: *const f32, len: usize| -> Result<&[f32], &'static str> {
            if from.is_null() {
                return Err(invalid);
            }
            // SAFETY: the engine's tensor holds `len` values (shape checked by the caller).
            Ok(unsafe { std::slice::from_raw_parts(from, len) })
        };
        let ints = |from: *const i32, len: usize| -> &[i32] {
            if from.is_null() || len == 0 {
                return &[];
            }
            // SAFETY: as above.
            unsafe { std::slice::from_raw_parts(from, len) }
        };
        let packed = slice(step.x, width * height * frames * channels)?;
        let (video_count, audio_count, reference_count) = (
            count(step.video_count, 64).ok_or(invalid)?,
            count(step.audio_count, 64).ok_or(invalid)?,
            count(step.reference_count, 64).ok_or(invalid)?,
        );
        let mut videos = Vec::with_capacity(video_count);
        for i in 0..video_count {
            // SAFETY: `videos` holds `video_count` pointers and `video_shapes` three values each.
            let (data, shape) = unsafe {
                (
                    *step.videos.add(i),
                    std::slice::from_raw_parts(step.video_shapes.add(i * 3), 3),
                )
            };
            let (w, h, t) = (
                size(shape[0], 4096).ok_or(invalid)?,
                size(shape[1], 4096).ok_or(invalid)?,
                size(shape[2], 4096).ok_or(invalid)?,
            );
            videos.push(dit::Video {
                data: slice(data, dit::VIDEO_CHANNELS * w * h * t)?,
                width: w,
                height: h,
                frames: t,
            });
        }
        let mut audios = Vec::with_capacity(audio_count);
        for i in 0..audio_count {
            // SAFETY: as for the videos.
            let (data, length) = unsafe { (*step.audios.add(i), *step.audio_lengths.add(i)) };
            let length = size(length, 1 << 20).ok_or(invalid)?;
            audios.push(dit::Audio {
                data: slice(data, dit::AUDIO_CHANNELS * 2 * length)?,
                length,
            });
        }
        let mut references = Vec::with_capacity(reference_count);
        for triple in ints(step.references, reference_count * 3).chunks_exact(3) {
            let index = |v: i32, len: usize| match v {
                v if v < 0 => Ok(None),
                v if (v as usize) < len => Ok(Some(v as usize)),
                _ => Err(invalid),
            };
            references.push(dit::Reference {
                kind: match triple[0] {
                    0 => dit::ReferenceKind::Image,
                    1 => dit::ReferenceKind::Video,
                    2 => dit::ReferenceKind::Audio,
                    3 => dit::ReferenceKind::VideoAudio,
                    _ => return Err(invalid),
                },
                video: index(triple[1], videos.len())?,
                audio: index(triple[2], audios.len())?,
            });
        }
        let network = placed_dit(out)?;
        let started = std::time::Instant::now();
        let mut values = network.forward(&dit::Step {
            packed,
            width,
            height,
            frames,
            audio_length: count(i64::from(step.audio_length), 1 << 20).ok_or(invalid)?,
            timestep: step.timestep,
            context: slice(step.context, tokens * dit::TEXT_WIDTH)?,
            tags: ints(step.tags, count(step.tag_count, 1 << 16).ok_or(invalid)?),
            videos: &videos,
            audios: &audios,
            keyframes: ints(
                step.keyframes,
                count(step.keyframe_count, 64).ok_or(invalid)?,
            ),
            references: &references,
            video_shift: step.video_shift,
            audio_shift: step.audio_shift,
        })?;
        eprintln!(
            "kuno-image: rust dit: step {} ms",
            started.elapsed().as_millis()
        );
        // SAFETY: `result` has the packed latent's size, which `forward` returns.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), result, values.len()) };
        values.iter_mut().for_each(|v| *v = 0.0);
        Ok(())
    })();
    match evaluated {
        Ok(()) => true,
        Err(code) => {
            eprintln!("kuno-image: rust dit unavailable ({code}); engine samples");
            out.dit = None;
            out.dit_failed = true;
            false
        }
    }
}
extern "C" fn video_preview(
    step: i32,
    bytes: *const u8,
    width: u32,
    height: u32,
    channels: u32,
    data: *mut c_void,
) {
    let (steps, wanted) = unsafe {
        (
            (*(data as *const VideoOutput)).steps,
            (*(data as *const VideoOutput)).previews,
        )
    };
    if !wanted {
        return;
    }
    if let Some(frame) = preview_frame(step, steps, bytes, width, height, channels) {
        let _ = crate::preview::write_frame(crate::worker_protocol::PREVIEW, &frame);
    }
}
extern "C" fn video_phase(phase: i32, step: i32, steps: i32, _data: *mut c_void) {
    let phase = match phase {
        1 => kuno_protocol::VideoPhase::Sampling,
        2 => kuno_protocol::VideoPhase::Decoding,
        _ => return,
    };
    let _ = crate::worker_protocol::write_progress(phase, step.max(0) as u32, steps.max(0) as u32);
}

static ENGINE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Public model context owned by one private worker. Calls are serialized and
/// request-derived native state is synchronously cleared before returning.
/// Creation is lazy: the first recipe selects the model version and attention.
/// Keep the session on its worker thread; callbacks are process-global.
pub struct Session {
    native: NonNull<c_void>,
    directory: std::path::PathBuf,
    /// Rust audio decoder for MiniMax H3, loaded on first use and kept with the session.
    audio: Option<std::sync::Arc<crate::engine::audio_vae::AudioDecoder>>,
    pci: String,
}

impl Session {
    pub fn new(directory: &Path, pci: &str) -> Result<Self, &'static str> {
        let models = directory.to_path_buf();
        let directory = CString::new(directory.to_str().ok_or("image_model_directory_invalid")?)
            .map_err(|_| "native_image_string_invalid")?;
        let address = pci.to_owned();
        let pci = CString::new(pci).map_err(|_| "native_image_string_invalid")?;
        let mut native = std::ptr::null_mut();
        // SAFETY: the shim copies both strings and only returns an owned pointer.
        let code = unsafe { kuno_qwen_session_new(directory.as_ptr(), pci.as_ptr(), &mut native) };
        if code != 0 {
            return Err("image_engine_unavailable");
        }
        Ok(Self {
            native: NonNull::new(native).ok_or("image_engine_unavailable")?,
            directory: models,
            audio: None,
            pci: address,
        })
    }

    /// Generate in a worker with core dumps/swap disabled. Only public weights
    /// remain resident after success; prompts, references, latents and caches
    /// are freed before the in-memory PNG is returned.
    pub fn generate(
        &mut self,
        recipe: &ImageRecipe,
        previews: bool,
    ) -> Result<Vec<u8>, &'static str> {
        let _exclusive = ENGINE.lock().map_err(|_| "image_engine_unavailable")?;
        // Other models need the video memory a kept DiT holds.
        evict_dit();
        recipe.validate_execution()?;
        let seed = recipe
            .seed
            .parse::<i64>()
            .map_err(|_| "native_image_seed_range_unsupported")?;
        let c = |s: &str| CString::new(s).map_err(|_| "native_image_string_invalid");
        let prompt = c(&recipe.prompt)?;
        let negative = c(&recipe.negative_prompt)?;
        let mut decoded = crate::references::decode(&recipe.reference_images)?;
        crate::references::resize_for_qwen21(&mut decoded)?;
        let mut references: Vec<NativeImage> = decoded
            .iter_mut()
            .map(|p| NativeImage {
                width: p.width,
                height: p.height,
                channel: 4,
                data: p.rgba.as_mut_ptr(),
            })
            .collect();
        let mut pixels = (Vec::<u8>::new(), 0u32, recipe.steps);
        let rust_qwen = cfg!(feature = "rust-dit") && recipe.workflow != kuno_protocol::QWEN_RECIPE;
        if !rust_qwen {
            // The other image model needs the video memory the Rust models hold.
            evict_qwen();
        }
        let code = if rust_qwen {
            drop(references);
            let rgba = qwen::generate(
                self.native.as_ptr(),
                &self.directory,
                &self.pci,
                recipe,
                seed,
                &decoded,
                previews,
            )?;
            pixels.0 = rgba;
            pixels.1 = 4;
            0
        } else {
            unsafe {
                kuno_qwen_session_generate(
                    self.native.as_ptr(),
                    prompt.as_ptr(),
                    negative.as_ptr(),
                    recipe.width as i32,
                    recipe.height as i32,
                    recipe.steps as i32,
                    recipe.cfg as f32,
                    recipe.flow_shift.unwrap_or(0.0) as f32,
                    seed,
                    i32::from(recipe.workflow != kuno_protocol::QWEN_RECIPE),
                    references.as_mut_ptr(),
                    references.len() as i32,
                    crate::host_memory::available(),
                    i32::from(recipe.attention == kuno_protocol::ImageAttention::Fast),
                    receive,
                    previews.then_some(preview),
                    &mut pixels as *mut _ as *mut c_void,
                )
            }
        };
        match code {
            0 => (),
            2 => return Err("image_physical_gpu_not_found"),
            3 => return Err("image_model_load_failed"),
            4 => return Err("image_generation_failed"),
            6 => return Err("image_gpu_memory_unavailable"),
            7 => return Err("image_cancelled"),
            8 => return Err("image_vulkan_device_lost"),
            9 => return Err("image_gpu_int8_unsupported"),
            10 => return Err("image_attention_unsupported"),
            11 => return Err("image_request_cleanup_failed"),
            _ => return Err("image_native_runtime_failed"),
        }
        if !matches!(pixels.1, 3 | 4)
            || pixels.0.len() != recipe.width as usize * recipe.height as usize * pixels.1 as usize
        {
            return Err("image_native_output_invalid");
        }
        let mut out = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut out, recipe.width, recipe.height);
            encoder.set_color(if pixels.1 == 4 {
                png::ColorType::Rgba
            } else {
                png::ColorType::Rgb
            });
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder
                .write_header()
                .map_err(|_| "image_encoding_failed")?;
            writer
                .write_image_data(&pixels.0)
                .map_err(|_| "image_encoding_failed")?;
        }
        Ok(out)
    }
}

impl Session {
    /// MiniMax H3 video with audio: text only, first and/or last frame, or reference images.
    /// Returns a validated, metadata-free MP4. As with images, request-derived native state
    /// is cleared before this returns; only weights remain.
    pub fn generate_video(
        &mut self,
        recipe: &kuno_protocol::VideoRecipe,
        previews: bool,
    ) -> Result<Vec<u8>, &'static str> {
        use kuno_protocol::VideoPhase;
        // The video models need the video memory the Rust image models hold.
        evict_qwen();
        /// Decoded request pixels, cleared on every exit path.
        #[derive(Default)]
        struct Inputs {
            first: Vec<u8>,
            last: Vec<u8>,
            /// The start and end frames' sizes (see `video::keyframe_rgb`).
            first_size: (u32, u32),
            last_size: (u32, u32),
            references: Vec<(u32, u32, Vec<u8>)>,
        }
        impl Drop for Inputs {
            fn drop(&mut self) {
                crate::worker_protocol::clear_bytes(&mut self.first);
                crate::worker_protocol::clear_bytes(&mut self.last);
                for (_, _, rgb) in &mut self.references {
                    crate::worker_protocol::clear_bytes(rgb);
                }
            }
        }
        let steps = recipe.steps();
        let _exclusive = ENGINE.lock().map_err(|_| "image_engine_unavailable")?;
        recipe.validate()?;
        let seed = recipe
            .seed
            .parse::<i64>()
            .map_err(|_| "native_image_seed_range_unsupported")?;
        let prompt =
            CString::new(recipe.prompt.as_str()).map_err(|_| "native_image_string_invalid")?;
        let keyframe = |image: &Option<String>,
                        code: &'static str|
         -> Result<(Vec<u8>, (u32, u32)), &'static str> {
            let Some(image) = image else {
                return Ok((Vec::new(), (0, 0)));
            };
            let decoded = crate::references::decode(&[image.as_str()])
                .map_err(|_| code)?
                .pop()
                .ok_or(code)?;
            let (rgb, w, h) =
                crate::video::keyframe_rgb(decoded, recipe.width, recipe.height, code)?;
            // The engine path takes frames at the clip's size only.
            if !cfg!(feature = "rust-dit") && (w, h) != (recipe.width, recipe.height) {
                return Err(code);
            }
            Ok((rgb, (w, h)))
        };
        let mut inputs = Inputs::default();
        (inputs.first, inputs.first_size) =
            keyframe(&recipe.first_frame, "video_first_frame_invalid")?;
        (inputs.last, inputs.last_size) = keyframe(&recipe.last_frame, "video_last_frame_invalid")?;
        if !recipe.reference_images.is_empty() {
            for pixels in crate::references::decode(&recipe.reference_images)
                .map_err(|_| "video_reference_images_invalid")?
            {
                let (width, height) = (pixels.width, pixels.height);
                inputs
                    .references
                    .push((width, height, crate::video::rgb(pixels)));
            }
        }
        let references: Vec<RgbImage> = inputs
            .references
            .iter()
            .map(|(width, height, rgb)| RgbImage {
                width: *width,
                height: *height,
                data: rgb.as_ptr(),
            })
            .collect();
        let optional = |rgb: &Vec<u8>| {
            if rgb.is_empty() {
                std::ptr::null()
            } else {
                rgb.as_ptr()
            }
        };
        let decoder = if recipe.audio {
            if self.audio.is_none() {
                let path = self.directory.join("minimax_h3_audio_vae_fp32.safetensors");
                self.audio = Some(std::sync::Arc::new(
                    crate::engine::audio_vae::AudioDecoder::load(&path)
                        .map_err(|_| "image_model_load_failed")?,
                ));
            }
            self.audio.clone()
        } else {
            None
        };
        crate::worker_protocol::write_progress(VideoPhase::Conditioning, 0, 0)?;
        let mut output = VideoOutput {
            frame_bytes: recipe.width as usize * recipe.height as usize * 3,
            frames: recipe.frames,
            steps,
            rgb: Vec::new(),
            decoder,
            audio: None,
            pci: self.pci.clone(),
            video_model: self.directory.join("minimax_h3_video_vae_fp16.safetensors"),
            dit_model: self.directory.join(if recipe.uses_reference_model() {
                "minimax_h3_ref2va_turbo4_int8_convrot.safetensors"
            } else {
                "minimax_h3_fl2va_turbo8_int8_convrot.safetensors"
            }),
            dit: None,
            dit_failed: false,
            text_model: self
                .directory
                .join("qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
            invalid: false,
            previews,
        };
        // The clip's frames are the largest host buffer; a clip that does not fit fails
        // here rather than aborting the worker.
        output
            .rgb
            .try_reserve_exact(output.frame_bytes * recipe.frames as usize)
            .map_err(|_| "image_host_memory_unavailable")?;
        // A kept DiT of another model is freed before the engine needs video memory.
        if let Some(kept) = resident_dit(&output.dit_model) {
            keep_dit(&output.dit_model, Some(kept));
        }
        // A kept DiT needs no read.
        let read_dit = cfg!(feature = "rust-dit") && !dit_resident(&output.dit_model);
        // Weights are read into host memory in the background: the text encoder's (which
        // conditioning waits for) beside the DiT's (the first step waits) and the decoder's.
        let reading = {
            let (pci, text, dit, video) = (
                output.pci.clone(),
                output.text_model.clone(),
                output.dit_model.clone(),
                output.video_model.clone(),
            );
            // The vision tower (when the request has images) on a thread of its own: it is
            // needed first, and waiting for it would delay the text encoder's longer read.
            let sees =
                !inputs.first.is_empty() || !inputs.last.is_empty() || !references.is_empty();
            std::thread::spawn(move || {
                let tower = sees.then(|| {
                    let (pci, text) = (pci.clone(), text.clone());
                    std::thread::spawn(move || drop(vision(&pci, &text)))
                });
                let texts = cfg!(feature = "rust-dit").then(|| {
                    let pci = pci.clone();
                    std::thread::spawn(move || drop(text_encoder(&pci, &text)))
                });
                if read_dit {
                    drop(dit_host(&pci, &dit));
                }
                drop(video_host(&pci, &video));
                for thread in [tower, texts].into_iter().flatten() {
                    let _ = thread.join();
                }
            })
        };
        let code = if cfg!(feature = "rust-dit") {
            let request = pipeline::Request {
                prompt: &recipe.prompt,
                width: recipe.width as usize,
                height: recipe.height as usize,
                frames: recipe.frames as usize,
                steps: steps as usize,
                seed,
                first: (&inputs.first, inputs.first_size),
                last: (&inputs.last, inputs.last_size),
                references: &inputs.references,
            };
            match pipeline::generate(self.native.as_ptr(), &request, &mut output) {
                Ok(()) => 0,
                Err(pipeline::Failure::Cancelled) => 7,
                Err(pipeline::Failure::Failed(code)) => {
                    eprintln!("kuno-image: video failed ({code})");
                    match code {
                        "gpu_memory" | "image_gpu_memory_unavailable" => 6,
                        "image_host_memory_unavailable" => 12,
                        // A failed submission may have reset the device: the gateway
                        // replaces the worker.
                        "gpu_failed" | "gpu_lost" => 8,
                        _ => 4,
                    }
                }
            }
        } else {
            unsafe {
                kuno_minimax_session_generate(
                    self.native.as_ptr(),
                    i32::from(recipe.uses_reference_model()),
                    prompt.as_ptr(),
                    recipe.width as i32,
                    recipe.height as i32,
                    recipe.frames as i32,
                    kuno_protocol::VIDEO_FPS as i32,
                    steps as i32,
                    seed,
                    optional(&inputs.first),
                    optional(&inputs.last),
                    references.as_ptr(),
                    references.len() as i32,
                    receive_video_frame,
                    receive_audio_latent,
                    Some(decode_video_latent),
                    cfg!(feature = "rust-dit").then_some(evaluate_dit as _),
                    cfg!(feature = "rust-dit").then_some(encode_text as _),
                    Some(encode_frame),
                    Some(encode_vision),
                    Some(video_preview),
                    video_phase,
                    &mut output as *mut _ as *mut c_void,
                )
            }
        };
        drop(references);
        drop(inputs);
        output.dit = None;
        let _ = reading.join();
        // Always join, so no request-derived work outlives the call.
        let audio = output.audio.take().map(|decode| decode.join());
        match code {
            0 => (),
            2 => return Err("image_physical_gpu_not_found"),
            3 => return Err("image_model_load_failed"),
            4 => return Err("image_generation_failed"),
            6 => return Err("image_gpu_memory_unavailable"),
            7 => return Err("image_cancelled"),
            8 => return Err("image_vulkan_device_lost"),
            9 => return Err("image_gpu_int8_unsupported"),
            11 => return Err("image_request_cleanup_failed"),
            12 => return Err("image_host_memory_unavailable"),
            _ => return Err("image_native_runtime_failed"),
        }
        if output.invalid
            || audio.is_some() != recipe.audio
            || output.rgb.len() != output.frame_bytes * recipe.frames as usize
        {
            return Err("image_native_output_invalid");
        }
        let audio = match audio {
            Some(decoded) => Some(decoded.map_err(|_| "image_native_runtime_failed")??),
            None => None,
        };
        crate::worker_protocol::write_progress(VideoPhase::Encoding, 0, 0)?;
        let clip = crate::video::Clip {
            width: recipe.width,
            height: recipe.height,
            fps: kuno_protocol::VIDEO_FPS,
            frames: recipe.frames,
            rgb: std::mem::take(&mut output.rgb),
            audio,
        };
        drop(output);
        crate::video::encode_mp4(&clip)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _exclusive = ENGINE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: this is the sole owner and no synchronous generate is active.
        unsafe { kuno_qwen_session_free(self.native.as_ptr()) };
    }
}

/// Compatibility entry for direct one-shot callers. The context is destroyed
/// immediately afterward; the gateway's resident worker owns a `Session`.
pub fn generate(
    directory: &Path,
    pci: &str,
    recipe: &ImageRecipe,
    previews: bool,
) -> Result<Vec<u8>, &'static str> {
    Session::new(directory, pci)?.generate(recipe, previews)
}
