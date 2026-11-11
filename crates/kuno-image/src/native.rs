//! Private subprocess entry only. Native callbacks are process-global; never
//! run concurrent generations or invoke this inside the long-lived gateway.
use kuno_protocol::ImageRecipe;
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
        preview: Option<extern "C" fn(i32, *const u8, u32, u32, u32, *mut c_void)>,
        phase: extern "C" fn(i32, i32, i32, *mut c_void),
        userdata: *mut c_void,
    ) -> i32;
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
    use crate::engine::{gpu, video_gpu::GpuVideoDecoder, video_tiles, video_vae::LATENT_CHANNELS};
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
    let (width, height, frames) = (width as usize, height as usize, frames as usize);
    if video_tiles::output_frames(frames) as i64 != pixel_frames {
        return false;
    }
    let latent =
        unsafe { std::slice::from_raw_parts(latent, LATENT_CHANNELS * frames * width * height) };
    let started = std::time::Instant::now();
    let mut loaded = started;
    let decoded = gpu::Gpu::open(gpu::Select::Pci(out.pci.clone()))
        .and_then(|gpu| GpuVideoDecoder::load(&gpu, &out.video_model))
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
        let code = unsafe {
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
        /// Decoded request pixels, cleared on every exit path.
        #[derive(Default)]
        struct Inputs {
            first: Vec<u8>,
            last: Vec<u8>,
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
        let keyframe =
            |image: &Option<String>, code: &'static str| -> Result<Vec<u8>, &'static str> {
                let Some(image) = image else {
                    return Ok(Vec::new());
                };
                let decoded = crate::references::decode(&[image.as_str()])
                    .map_err(|_| code)?
                    .pop()
                    .ok_or(code)?;
                crate::video::cover_rgb(decoded, recipe.width, recipe.height, code)
            };
        let mut inputs = Inputs::default();
        inputs.first = keyframe(&recipe.first_frame, "video_first_frame_invalid")?;
        inputs.last = keyframe(&recipe.last_frame, "video_last_frame_invalid")?;
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
            invalid: false,
            previews,
        };
        output
            .rgb
            .reserve_exact(output.frame_bytes * recipe.frames as usize);
        let code = unsafe {
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
                Some(video_preview),
                video_phase,
                &mut output as *mut _ as *mut c_void,
            )
        };
        drop(references);
        drop(inputs);
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
