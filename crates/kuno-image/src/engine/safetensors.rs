//! Minimal safetensors reader: header index plus positioned reads of single tensors.
use std::{collections::HashMap, fs::File, os::unix::fs::FileExt, path::Path};

struct Entry {
    f32: bool,
    f16: bool,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}
pub struct Safetensors {
    file: File,
    data: u64,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
}
impl Safetensors {
    pub fn open(path: &Path) -> Result<Self, &'static str> {
        let file = File::open(path).map_err(|_| "model_file_unavailable")?;
        let size = file.metadata().map_err(|_| "model_file_unavailable")?.len();
        let mut n = [0; 8];
        file.read_exact_at(&mut n, 0)
            .map_err(|_| "model_file_invalid")?;
        let n = u64::from_le_bytes(n);
        if n > 100 << 20 || 8 + n > size {
            return Err("model_file_invalid");
        }
        let mut header = vec![0; n as usize];
        file.read_exact_at(&mut header, 8)
            .map_err(|_| "model_file_invalid")?;
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|_| "model_file_invalid")?;
        let data = 8 + n;
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                if let Some(map) = value.as_object() {
                    for (k, v) in map {
                        if let Some(v) = v.as_str() {
                            metadata.insert(k.clone(), v.to_owned());
                        }
                    }
                }
                continue;
            }
            let entry = (|| {
                let shape = value["shape"]
                    .as_array()?
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize))
                    .collect::<Option<Vec<_>>>()?;
                let offsets = value["data_offsets"].as_array()?;
                let (begin, end) = (offsets.first()?.as_u64()?, offsets.get(1)?.as_u64()?);
                (begin <= end && end <= size - data).then_some(Entry {
                    f32: value["dtype"].as_str()? == "F32",
                    f16: value["dtype"].as_str()? == "F16",
                    shape,
                    begin,
                    end,
                })
            })()
            .ok_or("model_file_invalid")?;
            entries.insert(name, entry);
        }
        Ok(Self {
            file,
            data,
            entries,
            metadata,
        })
    }
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
    /// Reads one little-endian F32 tensor, checking its exact shape.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if !entry.f32 || entry.shape != shape || entry.end - entry.begin != count as u64 * 4 {
            return Err("model_tensor_invalid");
        }
        let mut out = vec![0f32; count];
        // SAFETY: `out` owns `count * 4` initialized bytes; any bit pattern is a valid f32.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * 4) };
        self.file
            .read_exact_at(bytes, self.data + entry.begin)
            .map_err(|_| "model_file_invalid")?;
        if cfg!(target_endian = "big") {
            for v in &mut out {
                *v = f32::from_bits(u32::from_le(v.to_bits()));
            }
        }
        Ok(out)
    }
    /// Reads one little-endian F16 tensor widened to f32 (exact), checking its exact shape.
    pub fn f16(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if !entry.f16 || entry.shape != shape || entry.end - entry.begin != count as u64 * 2 {
            return Err("model_tensor_invalid");
        }
        let mut bytes = vec![0u8; count * 2];
        self.file
            .read_exact_at(&mut bytes, self.data + entry.begin)
            .map_err(|_| "model_file_invalid")?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| widen(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    }
}
/// IEEE 754 binary16 to binary32, exact for every input.
pub fn widen(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exponent = u32::from(half >> 10) & 31;
    let fraction = u32::from(half) & 1023;
    let bits = match (exponent, fraction) {
        (0, 0) => 0,
        (0, _) => {
            // Subnormal: shift the leading one up to the implicit position.
            let shift = fraction.leading_zeros() - 21;
            ((113 - shift) << 23) | ((fraction << shift) & 1023) << 13
        }
        (31, _) => (255 << 23) | fraction << 13,
        _ => ((exponent + 112) << 23) | fraction << 13,
    };
    f32::from_bits(sign | bits)
}
#[cfg(test)]
mod tests {
    use super::widen;
    #[test]
    fn widens_half_precision_exactly() {
        assert_eq!(widen(0x3c00), 1.0);
        assert_eq!(widen(0xc000), -2.0);
        assert_eq!(widen(0x7bff), 65504.0);
        assert_eq!(widen(0x0001), 2f32.powi(-24));
        assert_eq!(widen(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(widen(0x0400), 2f32.powi(-14));
        assert_eq!(widen(0x3555), 0.333_251_95);
        assert!(widen(0x8000).is_sign_negative() && widen(0x8000) == 0.0);
        assert_eq!(widen(0x7c00), f32::INFINITY);
        assert!(widen(0x7e00).is_nan());
    }
}
