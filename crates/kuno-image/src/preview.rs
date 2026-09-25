//! Versioned, bounded worker pipe frames. Never write content to diagnostics.
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
pub const MAX_PREVIEW_BYTES: usize = 1024 * 1024;
#[derive(Clone, Serialize, Deserialize)]
pub struct Preview {
    pub step: u32,
    pub steps: u32,
    pub width: u32,
    pub height: u32,
    pub png_base64: String,
}
impl Preview {
    pub fn from_frame(frame: &[u8]) -> Result<Self, &'static str> {
        if frame.len() < 16 || frame.len() > MAX_PREVIEW_BYTES {
            return Err("image_preview_invalid");
        }
        let n = |i| u32::from_be_bytes(frame[i..i + 4].try_into().unwrap());
        let (step, steps, width, height) = (n(0), n(4), n(8), n(12));
        if steps == 0
            || steps > 200
            || step > steps
            || width == 0
            || height == 0
            || width > 512
            || height > 512
        {
            return Err("image_preview_invalid");
        }
        crate::validate_output(&frame[16..], width, height)?;
        Ok(Self {
            step,
            steps,
            width,
            height,
            png_base64: STANDARD.encode(&frame[16..]),
        })
    }
}
/// Header magic identifies protocol v1, then kind (1 preview, 2 final), length.
pub fn write_frame(kind: u8, payload: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut pipe = std::io::stdout().lock();
    pipe.write_all(b"KIP1")?;
    pipe.write_all(&[kind])?;
    pipe.write_all(&(payload.len() as u32).to_be_bytes())?;
    pipe.write_all(payload)?;
    pipe.flush()
}

impl std::fmt::Debug for Preview {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preview")
            .field("step", &self.step)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_frames_bound_content_and_validate_pixels() {
        let mut frame = Vec::new();
        for n in [1u32, 20, 64, 64] {
            frame.extend_from_slice(&n.to_be_bytes());
        }
        frame.extend_from_slice(include_bytes!(
            "../../../apps/vashti/src/images/fixtures/64.png"
        ));
        let preview = Preview::from_frame(&frame).unwrap();
        assert_eq!(preview.step, 1);
        assert_eq!(preview.width, 64);
        frame[8..12].copy_from_slice(&513u32.to_be_bytes());
        assert!(Preview::from_frame(&frame).is_err());
        assert!(Preview::from_frame(&[0; 15]).is_err());
        assert!(Preview::from_frame(&vec![0; MAX_PREVIEW_BYTES + 1]).is_err());
    }
}
