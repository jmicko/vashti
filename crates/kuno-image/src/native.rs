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
unsafe extern "C" {
    fn kuno_minimax_session_generate(
        session: *mut c_void,
        prompt: *const c_char,
        width: i32,
        height: i32,
        frames: i32,
        fps: i32,
        steps: i32,
        seed: i64,
        first_frame: *const u8,
        receive_frame: extern "C" fn(*const u8, usize, *mut c_void),
        receive_audio: extern "C" fn(*const f32, u64, u32, u32, *mut c_void),
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
    audio: Option<crate::video::Audio>,
    invalid: bool,
    previews: bool,
}
impl Drop for VideoOutput {
    fn drop(&mut self) {
        crate::worker_protocol::clear_bytes(&mut self.rgb);
    }
}
/// At most a minute of stereo audio at 48 kHz.
const MAX_AUDIO_VALUES: u64 = 48_000 * 60 * 2;
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
extern "C" fn receive_video_audio(
    samples: *const f32,
    count: u64,
    channels: u32,
    rate: u32,
    data: *mut c_void,
) {
    // SAFETY: as receive_video_frame; `count` samples per channel, interleaved.
    let out = unsafe { &mut *(data as *mut VideoOutput) };
    let values = count.saturating_mul(u64::from(channels));
    if samples.is_null()
        || out.audio.is_some()
        || !(1..=2).contains(&channels)
        || values == 0
        || values > MAX_AUDIO_VALUES
    {
        out.invalid = true;
        return;
    }
    out.audio = Some(crate::video::Audio {
        sample_rate: rate,
        channels,
        samples: unsafe { std::slice::from_raw_parts(samples, values as usize) }.to_vec(),
    });
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
}

impl Session {
    pub fn new(directory: &Path, pci: &str) -> Result<Self, &'static str> {
        let directory = CString::new(directory.to_str().ok_or("image_model_directory_invalid")?)
            .map_err(|_| "native_image_string_invalid")?;
        let pci = CString::new(pci).map_err(|_| "native_image_string_invalid")?;
        let mut native = std::ptr::null_mut();
        // SAFETY: the shim copies both strings and only returns an owned pointer.
        let code = unsafe { kuno_qwen_session_new(directory.as_ptr(), pci.as_ptr(), &mut native) };
        if code != 0 {
            return Err("image_engine_unavailable");
        }
        Ok(Self {
            native: NonNull::new(native).ok_or("image_engine_unavailable")?,
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
    /// MiniMax H3 image-to-video: returns a validated, metadata-free MP4. As with images,
    /// request-derived native state is cleared before this returns; only weights remain.
    pub fn generate_video(
        &mut self,
        recipe: &kuno_protocol::VideoRecipe,
        previews: bool,
    ) -> Result<Vec<u8>, &'static str> {
        use kuno_protocol::VideoPhase;
        const STEPS: u32 = 8;
        let _exclusive = ENGINE.lock().map_err(|_| "image_engine_unavailable")?;
        recipe.validate()?;
        let seed = recipe
            .seed
            .parse::<i64>()
            .map_err(|_| "native_image_seed_range_unsupported")?;
        let prompt =
            CString::new(recipe.prompt.as_str()).map_err(|_| "native_image_string_invalid")?;
        let decoded = crate::references::decode(&[recipe.first_frame.as_str()])
            .map_err(|_| "video_first_frame_invalid")?
            .pop()
            .ok_or("video_first_frame_invalid")?;
        let mut first = crate::video::cover_rgb(decoded, recipe.width, recipe.height)?;
        crate::worker_protocol::write_progress(VideoPhase::Conditioning, 0, 0)?;
        let mut output = VideoOutput {
            frame_bytes: recipe.width as usize * recipe.height as usize * 3,
            frames: recipe.frames,
            steps: STEPS,
            rgb: Vec::new(),
            audio: None,
            invalid: false,
            previews,
        };
        output
            .rgb
            .reserve_exact(output.frame_bytes * recipe.frames as usize);
        let code = unsafe {
            kuno_minimax_session_generate(
                self.native.as_ptr(),
                prompt.as_ptr(),
                recipe.width as i32,
                recipe.height as i32,
                recipe.frames as i32,
                kuno_protocol::VIDEO_FPS as i32,
                STEPS as i32,
                seed,
                first.as_ptr(),
                receive_video_frame,
                receive_video_audio,
                Some(video_preview),
                video_phase,
                &mut output as *mut _ as *mut c_void,
            )
        };
        crate::worker_protocol::clear_bytes(&mut first);
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
            || output.audio.is_none()
            || output.rgb.len() != output.frame_bytes * recipe.frames as usize
        {
            return Err("image_native_output_invalid");
        }
        crate::worker_protocol::write_progress(VideoPhase::Encoding, 0, 0)?;
        let clip = crate::video::Clip {
            width: recipe.width,
            height: recipe.height,
            fps: kuno_protocol::VIDEO_FPS,
            frames: recipe.frames,
            rgb: std::mem::take(&mut output.rgb),
            audio: if recipe.audio {
                output.audio.take()
            } else {
                None
            },
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
