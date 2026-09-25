//! Private subprocess entry only. Native callbacks are process-global; never
//! run concurrent generations or invoke this inside the long-lived gateway.
use kuno_protocol::ImageRecipe;
use std::{
    ffi::{CString, c_char, c_void},
    path::Path,
};
#[repr(C)]
struct NativeImage {
    width: u32,
    height: u32,
    channel: u32,
    data: *mut u8,
}
unsafe extern "C" {
    fn kuno_qwen_generate(
        directory: *const c_char,
        pci: *const c_char,
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
    if !matches!(channels, 3 | 4)
        || step < 0
        || width == 0
        || height == 0
        || width > 512
        || height > 512
        || bytes.is_null()
    {
        return;
    }
    // The synchronous shim owns the RGB/RGBA buffer until this callback returns.
    let (steps, rgb) = unsafe {
        (
            (*(data as *const (Vec<u8>, u32, u32))).2,
            std::slice::from_raw_parts(bytes, width as usize * height as usize * channels as usize),
        )
    };
    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, width, height);
    encoder.set_color(if channels == 4 {
        png::ColorType::Rgba
    } else {
        png::ColorType::Rgb
    });
    encoder.set_depth(png::BitDepth::Eight);
    let Ok(mut writer) = encoder.write_header() else {
        return;
    };
    if writer.write_image_data(rgb).is_err() {
        return;
    }
    drop(writer);
    if png.len() + 16 > crate::preview::MAX_PREVIEW_BYTES {
        return;
    }
    let mut frame = Vec::with_capacity(png.len() + 16);
    for number in [step as u32, steps, width, height] {
        frame.extend_from_slice(&number.to_be_bytes());
    }
    frame.extend_from_slice(&png);
    let _ = crate::preview::write_frame(1, &frame);
}
/// Call once in an isolated worker with core dumps/swap disabled. No file or
/// logging output contains inference content. The result is an in-memory PNG.
pub fn generate(
    directory: &Path,
    pci: &str,
    recipe: &ImageRecipe,
    previews: bool,
) -> Result<Vec<u8>, &'static str> {
    static ENGINE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _exclusive = ENGINE.lock().map_err(|_| "image_engine_unavailable")?;
    recipe.validate_execution()?;
    let seed = recipe
        .seed
        .parse::<i64>()
        .map_err(|_| "native_image_seed_range_unsupported")?;
    let c = |s: &str| CString::new(s).map_err(|_| "native_image_string_invalid");
    let directory = c(directory.to_str().ok_or("image_model_directory_invalid")?)?;
    let pci = c(pci)?;
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
        kuno_qwen_generate(
            directory.as_ptr(),
            pci.as_ptr(),
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
