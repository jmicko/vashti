//! Private worker pipe ABI. Content never appears in argv or diagnostics.
//! A generate frame has fixed controls followed by bounded rendered UTF-8.
use std::io::{self, Read, Write};
use zeroize::{Zeroize, Zeroizing};
pub const REQUEST_MAGIC: &[u8; 4] = b"KNT2";
pub const RESPONSE_MAGIC: &[u8; 4] = b"KNR2";
pub const MAX_PROMPT_BYTES: usize = 1024 * 1024;
pub const MAX_FRAME_BYTES: usize = 64 * 1024;
pub const TEXT: u8 = 1;
pub const DONE: u8 = 2;
pub const ERROR: u8 = 3;
pub const CLOSED: u8 = 4;
pub const READY: u8 = 5;
/// Rejected before model execution; the retained worker remains clean and usable.
pub const REJECTED: u8 = 6;
pub const CONTROL_BYTES: usize = 36;
pub const DEFAULT_CONTEXT: u32 = 32768;
pub const MAX_CONTEXT: u32 = 32768;

pub struct Generate {
    pub prompt: Zeroizing<Vec<u8>>,
    pub outputs: u32,
    pub context: u32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
}
impl Drop for Generate {
    fn drop(&mut self) {
        self.outputs.zeroize();
        self.context.zeroize();
        self.seed.zeroize();
        self.temperature.zeroize();
        self.top_p.zeroize();
        self.min_p.zeroize();
        self.top_k.zeroize();
    }
}
fn bad() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "native_worker_frame_invalid")
}
impl Generate {
    pub fn validate(&self) -> io::Result<()> {
        if !(512..=MAX_CONTEXT).contains(&self.context) {
            return Err(bad());
        }
        if self.prompt.is_empty()
            && self.outputs == 0
            && self.seed == 0
            && self.temperature == 0.
            && self.top_p == 1.
            && self.min_p == 0.
            && self.top_k == 1
        {
            return Ok(());
        }
        if self.prompt.is_empty()
            || self.prompt.len() > MAX_PROMPT_BYTES
            || std::str::from_utf8(&self.prompt).is_err()
            || self.outputs == 0
            || self.outputs > self.context.saturating_sub(9)
            || !self.temperature.is_finite()
            || !(0.0..=10.0).contains(&self.temperature)
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || !self.min_p.is_finite()
            || !(0.0..=1.0).contains(&self.min_p)
            || !(0..=64).contains(&self.top_k)
        {
            return Err(bad());
        }
        Ok(())
    }
    pub fn controls(&self) -> io::Result<Zeroizing<[u8; CONTROL_BYTES]>> {
        self.validate()?;
        let mut raw = Zeroizing::new([0u8; CONTROL_BYTES]);
        raw[0..4].copy_from_slice(&(self.prompt.len() as u32).to_le_bytes());
        raw[4..8].copy_from_slice(&self.outputs.to_le_bytes());
        raw[8..16].copy_from_slice(&self.seed.to_le_bytes());
        raw[16..20].copy_from_slice(&self.temperature.to_le_bytes());
        raw[20..24].copy_from_slice(&self.top_p.to_le_bytes());
        raw[24..28].copy_from_slice(&self.min_p.to_le_bytes());
        raw[28..32].copy_from_slice(&self.top_k.to_le_bytes());
        raw[32..36].copy_from_slice(&self.context.to_le_bytes());
        Ok(raw)
    }
}
/// EOF at a frame boundary is graceful shutdown. Truncation is a failure.
pub fn read_request(reader: &mut impl Read) -> io::Result<Option<Generate>> {
    let mut magic = [0; 4];
    loop {
        match reader.read(&mut magic[..1]) {
            Ok(0) => return Ok(None),
            Ok(1) => break,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            _ => return Err(bad()),
        }
    }
    reader.read_exact(&mut magic[1..])?;
    if &magic != REQUEST_MAGIC {
        return Err(bad());
    }
    let mut raw = Zeroizing::new([0u8; CONTROL_BYTES]);
    reader.read_exact(&mut *raw)?;
    let word = |at| u32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
    let length = word(0) as usize;
    if length > MAX_PROMPT_BYTES {
        return Err(bad());
    }
    let mut request = Generate {
        prompt: Zeroizing::new(vec![0; length]),
        outputs: word(4),
        context: word(32),
        seed: u64::from_le_bytes(raw[8..16].try_into().unwrap()),
        temperature: f32::from_bits(word(16)),
        top_p: f32::from_bits(word(20)),
        min_p: f32::from_bits(word(24)),
        top_k: i32::from_le_bytes(raw[28..32].try_into().unwrap()),
    };
    reader.read_exact(&mut request.prompt)?;
    request.validate()?;
    Ok(Some(request))
}
pub fn write_request(writer: &mut impl Write, request: &Generate) -> io::Result<()> {
    let raw = request.controls()?;
    writer.write_all(REQUEST_MAGIC)?;
    writer.write_all(&*raw)?;
    writer.write_all(&request.prompt)?;
    writer.flush()
}
pub fn write_response(writer: &mut impl Write, kind: u8, bytes: &[u8]) -> io::Result<()> {
    if !matches!(kind, TEXT | DONE | ERROR | CLOSED | READY | REJECTED)
        || bytes.len() > MAX_FRAME_BYTES
    {
        return Err(bad());
    }
    writer.write_all(RESPONSE_MAGIC)?;
    writer.write_all(&[kind])?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Generate {
        Generate {
            prompt: Zeroizing::new("hello 🦦\n".as_bytes().to_vec()),
            outputs: 64,
            context: DEFAULT_CONTEXT,
            seed: 123,
            temperature: 0.,
            top_p: 0.95,
            min_p: 0.,
            top_k: 20,
        }
    }
    #[test]
    fn framed_requests_preserve_bytes_and_boundary_eof() {
        let a = request();
        let mut bytes = Vec::new();
        write_request(&mut bytes, &a).unwrap();
        write_request(&mut bytes, &a).unwrap();
        let mut input = bytes.as_slice();
        for _ in 0..2 {
            let b = read_request(&mut input).unwrap().unwrap();
            assert_eq!(*b.prompt, *a.prompt);
            assert_eq!(b.seed, 123);
        }
        assert!(read_request(&mut input).unwrap().is_none());
        for end in 1..(4 + CONTROL_BYTES + a.prompt.len()) {
            assert!(read_request(&mut &bytes[..end]).is_err());
        }
    }
    #[test]
    fn preload_is_a_canonical_control_frame_and_bad_preload_never_passes() {
        let mut r = request();
        r.prompt.clear();
        r.outputs = 0;
        r.seed = 0;
        r.top_k = 1;
        r.top_p = 1.;
        let mut raw = Vec::new();
        write_request(&mut raw, &r).unwrap();
        assert_eq!(
            read_request(&mut raw.as_slice()).unwrap().unwrap().outputs,
            0
        );
        r.temperature = f32::NAN;
        assert!(r.controls().is_err());
    }
    #[test]
    fn invalid_controls_and_oversize_reject_before_output() {
        let mut a = request();
        a.temperature = f32::NAN;
        assert!(a.controls().is_err());
        a.temperature = 0.;
        a.top_k = 65;
        assert!(a.controls().is_err());
        let mut raw = Vec::from(*REQUEST_MAGIC);
        raw.extend_from_slice(&((MAX_PROMPT_BYTES + 1) as u32).to_le_bytes());
        raw.extend_from_slice(&[0; CONTROL_BYTES - 4]);
        assert!(read_request(&mut raw.as_slice()).is_err());
        assert!(write_response(&mut Vec::new(), 9, b"x").is_err());
    }
    #[test]
    fn context_budget_roundtrips_and_protocol_versions_do_not_mix() {
        let mut r = request();
        r.context = 4096;
        let mut bytes = Vec::new();
        write_request(&mut bytes, &r).unwrap();
        assert_eq!(
            read_request(&mut bytes.as_slice())
                .unwrap()
                .unwrap()
                .context,
            4096
        );
        bytes[..4].copy_from_slice(b"KNT1");
        assert!(read_request(&mut bytes.as_slice()).is_err());
        r.context = 511;
        assert!(r.validate().is_err());
        r.context = 32769;
        assert!(r.validate().is_err());
        assert!(write_response(&mut Vec::new(), REJECTED, b"native_context_exceeded").is_ok());
    }
}
