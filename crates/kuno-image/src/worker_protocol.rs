//! Private, bounded resident-worker pipes. Frames never enter diagnostics.
use kuno_protocol::{ImageRecipe, MAX_IMAGE_REQUEST_BYTES, VideoPhase, VideoRecipe};
use std::io::{Read, Write};

pub const REQUEST_MAGIC: &[u8; 4] = b"KIR2";
pub const RESPONSE_MAGIC: &[u8; 4] = b"KIP1";
pub const GENERATE: u8 = 1;
pub const SHUTDOWN: u8 = 2;
pub const PREVIEW: u8 = 1;
pub const FINAL: u8 = 2;
pub const CLEAN: u8 = 3;
pub const READY: u8 = 4;
pub const ERROR: u8 = 5;
/// Content-free video progress: phase byte, step and steps (big-endian u32).
pub const PROGRESS: u8 = 6;
pub const GENERATE_VIDEO: u8 = 3;
pub const MAX_FINAL_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_VIDEO_FINAL_BYTES: usize = kuno_protocol::MAX_VIDEO_BYTES;
pub const MAX_ERROR_BYTES: usize = 128;

/// This acknowledgement protocol permits retaining weights only after every
/// request-owned buffer and the native derived state have been cleared.
pub enum Request {
    Generate(TransientRecipe),
    GenerateVideo(TransientVideoRecipe),
    Shutdown,
}

/// Request-owned video strings must not outlive the job's CLEAN frame.
pub struct TransientVideoRecipe(VideoRecipe);
impl From<VideoRecipe> for TransientVideoRecipe {
    fn from(recipe: VideoRecipe) -> Self {
        Self(recipe)
    }
}
impl std::ops::Deref for TransientVideoRecipe {
    type Target = VideoRecipe;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for TransientVideoRecipe {
    fn drop(&mut self) {
        clear_string(&mut self.0.workflow);
        clear_string(&mut self.0.prompt);
        clear_string(&mut self.0.seed);
        clear_string(&mut self.0.first_frame);
    }
}

/// Request-owned strings must not outlive the successful job's CLEAN frame.
/// Avoid Debug/Clone: content is neither diagnostic nor reusable worker state.
pub struct TransientRecipe(ImageRecipe);
impl From<ImageRecipe> for TransientRecipe {
    fn from(recipe: ImageRecipe) -> Self {
        Self(recipe)
    }
}
impl std::ops::Deref for TransientRecipe {
    type Target = ImageRecipe;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for TransientRecipe {
    fn drop(&mut self) {
        clear_string(&mut self.0.workflow);
        clear_string(&mut self.0.prompt);
        clear_string(&mut self.0.negative_prompt);
        clear_string(&mut self.0.seed);
        for reference in &mut self.0.reference_images {
            clear_string(reference);
        }
    }
}

/// Clear the complete allocation, including spare capacity from earlier writes.
/// Volatile stores prevent the optimizer removing writes immediately before free.
pub fn clear_bytes(bytes: &mut Vec<u8>) {
    let pointer = bytes.as_mut_ptr();
    for offset in 0..bytes.capacity() {
        // SAFETY: the allocated capacity is writable, even beyond initialized
        // length. No reads of uninitialized storage occur and no reference escapes.
        unsafe { pointer.add(offset).write_volatile(0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    bytes.clear();
}
fn clear_string(string: &mut String) {
    // SAFETY: only zeros are written; empty and zero-filled strings are UTF-8.
    unsafe { clear_bytes(string.as_mut_vec()) };
}

/// Drops with allocation clearing on every return path, including pipe failures.
pub struct TransientBytes(pub Vec<u8>);
impl Drop for TransientBytes {
    fn drop(&mut self) {
        clear_bytes(&mut self.0);
    }
}

fn read_header(reader: &mut impl Read) -> Result<Option<[u8; 9]>, &'static str> {
    let mut header = [0u8; 9];
    loop {
        match reader.read(&mut header[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err("image_request_read_failed"),
        }
    }
    reader
        .read_exact(&mut header[1..])
        .map_err(|_| "image_worker_request_invalid")?;
    Ok(Some(header))
}

/// EOF is clean only between jobs. A partial header/payload is never a job.
pub fn read_request(reader: &mut impl Read) -> Result<Option<Request>, &'static str> {
    let Some(header) = read_header(reader)? else {
        return Ok(None);
    };
    if &header[..4] != REQUEST_MAGIC {
        return Err("image_worker_request_invalid");
    }
    let size = u32::from_be_bytes(header[5..9].try_into().unwrap()) as usize;
    match header[4] {
        GENERATE if size > 0 && size <= MAX_IMAGE_REQUEST_BYTES => {
            let mut bytes = TransientBytes(vec![0; size]);
            reader
                .read_exact(&mut bytes.0)
                .map_err(|_| "image_worker_request_invalid")?;
            // ImageRecipe rejects unknown/duplicate fields; from_slice rejects
            // trailing JSON rather than allowing a second unframed request.
            let recipe = TransientRecipe(
                serde_json::from_slice(&bytes.0).map_err(|_| "image_request_invalid")?,
            );
            recipe.validate_execution()?;
            Ok(Some(Request::Generate(recipe)))
        }
        GENERATE_VIDEO if size > 0 && size <= MAX_IMAGE_REQUEST_BYTES => {
            let mut bytes = TransientBytes(vec![0; size]);
            reader
                .read_exact(&mut bytes.0)
                .map_err(|_| "image_worker_request_invalid")?;
            let recipe = TransientVideoRecipe(
                serde_json::from_slice(&bytes.0).map_err(|_| "image_request_invalid")?,
            );
            recipe.validate()?;
            Ok(Some(Request::GenerateVideo(recipe)))
        }
        GENERATE | GENERATE_VIDEO if size > MAX_IMAGE_REQUEST_BYTES => {
            Err("image_request_too_large")
        }
        SHUTDOWN if size == 0 => Ok(Some(Request::Shutdown)),
        _ => Err("image_worker_request_invalid"),
    }
}

fn write_frame(
    writer: &mut impl Write,
    magic: &[u8; 4],
    kind: u8,
    payload: &[u8],
) -> Result<(), &'static str> {
    let size = u32::try_from(payload.len()).map_err(|_| "image_worker_frame_invalid")?;
    writer
        .write_all(magic)
        .and_then(|_| writer.write_all(&[kind]))
        .and_then(|_| writer.write_all(&size.to_be_bytes()))
        .and_then(|_| writer.write_all(payload))
        .and_then(|_| writer.flush())
        .map_err(|_| "image_output_failed")
}

pub fn write_generate(writer: &mut impl Write, recipe: &ImageRecipe) -> Result<(), &'static str> {
    recipe.validate_execution()?;
    let bytes = TransientBytes(serde_json::to_vec(recipe).map_err(|_| "image_request_invalid")?);
    if bytes.0.len() > MAX_IMAGE_REQUEST_BYTES {
        return Err("image_request_too_large");
    }
    write_frame(writer, REQUEST_MAGIC, GENERATE, &bytes.0)
}

pub fn write_generate_video(
    writer: &mut impl Write,
    recipe: &VideoRecipe,
) -> Result<(), &'static str> {
    recipe.validate()?;
    let bytes = TransientBytes(serde_json::to_vec(recipe).map_err(|_| "image_request_invalid")?);
    if bytes.0.len() > MAX_IMAGE_REQUEST_BYTES {
        return Err("image_request_too_large");
    }
    write_frame(writer, REQUEST_MAGIC, GENERATE_VIDEO, &bytes.0)
}

pub fn progress_payload(phase: VideoPhase, step: u32, steps: u32) -> [u8; 9] {
    let mut payload = [0u8; 9];
    payload[0] = phase as u8;
    payload[1..5].copy_from_slice(&step.to_be_bytes());
    payload[5..9].copy_from_slice(&steps.to_be_bytes());
    payload
}

/// Decodes a PROGRESS payload; sampling carries 1..=steps, other phases none.
pub fn parse_progress(payload: &[u8]) -> Result<(VideoPhase, u32, u32), &'static str> {
    let payload: &[u8; 9] = payload
        .try_into()
        .map_err(|_| "image_worker_frame_invalid")?;
    let n = |i: usize| u32::from_be_bytes(payload[i..i + 4].try_into().unwrap());
    let (step, steps) = (n(1), n(5));
    let phase = match payload[0] {
        0 => VideoPhase::Conditioning,
        1 => VideoPhase::Sampling,
        2 => VideoPhase::Decoding,
        3 => VideoPhase::Encoding,
        _ => return Err("image_worker_frame_invalid"),
    };
    let valid = match phase {
        VideoPhase::Sampling => (1..=200).contains(&steps) && (1..=steps).contains(&step),
        _ => step == 0 && steps == 0,
    };
    if !valid {
        return Err("image_worker_frame_invalid");
    }
    Ok((phase, step, steps))
}

/// Worker-side progress through the same stdout pipe as previews.
pub fn write_progress(phase: VideoPhase, step: u32, steps: u32) -> Result<(), &'static str> {
    let payload = progress_payload(phase, step, steps);
    parse_progress(&payload)?;
    crate::preview::write_frame(PROGRESS, &payload).map_err(|_| "image_output_failed")
}

pub fn write_shutdown(writer: &mut impl Write) -> Result<(), &'static str> {
    write_frame(writer, REQUEST_MAGIC, SHUTDOWN, &[])
}

pub fn write_response(
    writer: &mut impl Write,
    kind: u8,
    payload: &[u8],
) -> Result<(), &'static str> {
    let valid = match kind {
        PREVIEW => (16..=crate::preview::MAX_PREVIEW_BYTES).contains(&payload.len()),
        // The supervisor applies the per-request bound (image PNG or video MP4).
        FINAL => (1..=MAX_FINAL_BYTES.max(MAX_VIDEO_FINAL_BYTES)).contains(&payload.len()),
        PROGRESS => parse_progress(payload).is_ok(),
        CLEAN | READY => payload.is_empty(),
        ERROR => {
            !payload.is_empty()
                && payload.len() <= MAX_ERROR_BYTES
                && payload
                    .iter()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
        }
        _ => false,
    };
    if !valid {
        return Err("image_worker_frame_invalid");
    }
    write_frame(writer, RESPONSE_MAGIC, kind, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn recipe() -> ImageRecipe {
        ImageRecipe {
            workflow: kuno_protocol::QWEN21_RECIPE.into(),
            prompt: "synthetic private prompt".into(),
            negative_prompt: String::new(),
            seed: "1".into(),
            width: 64,
            height: 64,
            steps: 1,
            cfg: 1.0,
            attention: Default::default(),
            flow_shift: None,
            reference_images: vec![],
        }
    }
    #[test]
    fn consecutive_requests_do_not_depend_on_eof() {
        let mut pipe = Vec::new();
        write_generate(&mut pipe, &recipe()).unwrap();
        write_generate(&mut pipe, &recipe()).unwrap();
        write_shutdown(&mut pipe).unwrap();
        let mut reader = pipe.as_slice();
        for _ in 0..2 {
            let Some(Request::Generate(recipe)) = read_request(&mut reader).unwrap() else {
                panic!("missing request")
            };
            assert_eq!(recipe.width, 64);
        }
        assert!(matches!(
            read_request(&mut reader).unwrap(),
            Some(Request::Shutdown)
        ));
        assert!(read_request(&mut reader).unwrap().is_none());
    }
    #[test]
    fn truncated_oversized_and_unknown_frames_are_rejected() {
        let mut encoded = Vec::new();
        write_generate(&mut encoded, &recipe()).unwrap();
        for length in 1..encoded.len() {
            assert!(read_request(&mut &encoded[..length]).is_err());
        }
        for (kind, size) in [(0, 0), (GENERATE, 0), (SHUTDOWN, 1), (GENERATE, u32::MAX)] {
            let mut bytes = REQUEST_MAGIC.to_vec();
            bytes.push(kind);
            bytes.extend_from_slice(&size.to_be_bytes());
            assert!(read_request(&mut bytes.as_slice()).is_err());
        }
        encoded[0] = b'X';
        assert!(read_request(&mut encoded.as_slice()).is_err());
    }
    #[test]
    fn unknown_duplicate_and_trailing_json_are_rejected() {
        let original = serde_json::to_vec(&recipe()).unwrap();
        for extra in [b",\"unknown\":0}".as_slice(), b",\"prompt\":\"other\"}"] {
            let mut payload = original[..original.len() - 1].to_vec();
            payload.extend_from_slice(extra);
            let mut pipe = Vec::new();
            write_frame(&mut pipe, REQUEST_MAGIC, GENERATE, &payload).unwrap();
            assert!(read_request(&mut pipe.as_slice()).is_err());
        }
        let mut payload = original;
        payload.extend_from_slice(b" {}");
        let mut pipe = Vec::new();
        write_frame(&mut pipe, REQUEST_MAGIC, GENERATE, &payload).unwrap();
        assert!(read_request(&mut pipe.as_slice()).is_err());
    }
    #[test]
    fn response_frames_enforce_control_and_content_bounds() {
        let mut pipe = Vec::new();
        write_response(&mut pipe, READY, &[]).unwrap();
        write_response(&mut pipe, FINAL, b"pixels").unwrap();
        write_response(&mut pipe, CLEAN, &[]).unwrap();
        write_response(&mut pipe, ERROR, b"image_generation_failed").unwrap();
        for (kind, payload) in [
            (READY, b"private".as_slice()),
            (CLEAN, b"x"),
            (ERROR, b"RAW private"),
            (FINAL, b""),
            (0, b""),
        ] {
            assert!(write_response(&mut Vec::new(), kind, payload).is_err());
        }
    }
    fn video() -> VideoRecipe {
        VideoRecipe {
            workflow: kuno_protocol::MINIMAX_H3_I2V_RECIPE.into(),
            prompt: "synthetic private prompt".into(),
            seed: "7".into(),
            width: 832,
            height: 544,
            frames: 56,
            audio: true,
            first_frame: "AAAA".into(),
        }
    }
    #[test]
    fn video_requests_round_trip_and_reject_invalid_recipes() {
        let mut pipe = Vec::new();
        write_generate_video(&mut pipe, &video()).unwrap();
        write_generate(&mut pipe, &recipe()).unwrap();
        let mut reader = pipe.as_slice();
        let Some(Request::GenerateVideo(v)) = read_request(&mut reader).unwrap() else {
            panic!("missing video request")
        };
        assert_eq!((v.width, v.frames), (832, 56));
        assert!(matches!(
            read_request(&mut reader).unwrap(),
            Some(Request::Generate(_))
        ));
        let mut bad = video();
        bad.frames = 57;
        let payload = serde_json::to_vec(&bad).unwrap();
        let mut pipe = Vec::new();
        write_frame(&mut pipe, REQUEST_MAGIC, GENERATE_VIDEO, &payload).unwrap();
        assert!(read_request(&mut pipe.as_slice()).is_err());
    }
    #[test]
    fn progress_frames_are_content_free_and_bounded() {
        for (phase, step, steps) in [
            (VideoPhase::Conditioning, 0, 0),
            (VideoPhase::Sampling, 3, 8),
            (VideoPhase::Decoding, 0, 0),
            (VideoPhase::Encoding, 0, 0),
        ] {
            let payload = progress_payload(phase, step, steps);
            assert_eq!(parse_progress(&payload).unwrap(), (phase, step, steps));
            write_response(&mut Vec::new(), PROGRESS, &payload).unwrap();
        }
        for payload in [
            progress_payload(VideoPhase::Sampling, 0, 8),
            progress_payload(VideoPhase::Sampling, 9, 8),
            progress_payload(VideoPhase::Decoding, 1, 8),
        ] {
            assert!(parse_progress(&payload).is_err());
        }
        assert!(parse_progress(&[1, 2, 3]).is_err());
        let mut unknown = progress_payload(VideoPhase::Encoding, 0, 0);
        unknown[0] = 4;
        assert!(parse_progress(&unknown).is_err());
    }
    #[test]
    fn clearing_covers_capacity_as_well_as_current_length() {
        let mut bytes = vec![9u8; 32];
        bytes.truncate(1);
        let capacity = bytes.capacity();
        let pointer = bytes.as_ptr();
        clear_bytes(&mut bytes);
        assert!(bytes.is_empty());
        // SAFETY: clear_bytes initialized the entire unchanged allocation.
        assert!(
            unsafe { std::slice::from_raw_parts(pointer, capacity) }
                .iter()
                .all(|b| *b == 0)
        );
        let mut recipe = TransientRecipe(recipe());
        let pointer = recipe.0.prompt.as_ptr();
        let capacity = recipe.0.prompt.capacity();
        clear_string(&mut recipe.0.prompt);
        assert!(
            unsafe { std::slice::from_raw_parts(pointer, capacity) }
                .iter()
                .all(|b| *b == 0)
        );
    }
}
