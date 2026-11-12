//! Minimal safetensors reader: header index plus positioned reads of single tensors.
use std::{collections::HashMap, fs::File, os::unix::fs::FileExt, path::Path};

#[derive(PartialEq)]
enum Kind {
    F32,
    F16,
    Bf16,
    I8,
    Other,
}
struct Entry {
    kind: Kind,
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
                    kind: match value["dtype"].as_str()? {
                        "F32" => Kind::F32,
                        "F16" => Kind::F16,
                        "BF16" => Kind::Bf16,
                        "I8" => Kind::I8,
                        _ => Kind::Other,
                    },
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
        if entry.kind != Kind::F32
            || entry.shape != shape
            || entry.end - entry.begin != count as u64 * 4
        {
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
    /// Reads one F16 tensor's little-endian bytes into `to` (two bytes per value), checking
    /// its exact shape.
    pub fn f16_into(&self, name: &str, shape: &[usize], to: &mut [u8]) -> Result<(), &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if entry.kind != Kind::F16
            || entry.shape != shape
            || entry.end - entry.begin != count as u64 * 2
            || to.len() != count * 2
        {
            return Err("model_tensor_invalid");
        }
        self.file
            .read_exact_at(to, self.data + entry.begin)
            .map_err(|_| "model_file_invalid")
    }
    /// Reads one little-endian F16 tensor widened to f32 (exact), checking its exact shape.
    pub fn f16(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if entry.kind != Kind::F16
            || entry.shape != shape
            || entry.end - entry.begin != count as u64 * 2
        {
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
    /// Reads one INT8 tensor's bytes into `to`, checking its exact shape.
    pub fn i8_into(&self, name: &str, shape: &[usize], to: &mut [u8]) -> Result<(), &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if entry.kind != Kind::I8
            || entry.shape != shape
            || entry.end - entry.begin != count as u64
            || to.len() != count
        {
            return Err("model_tensor_invalid");
        }
        self.file
            .read_exact_at(to, self.data + entry.begin)
            .map_err(|_| "model_file_invalid")
    }
    /// Reads one INT8 tensor's bytes, checking its exact shape.
    pub fn i8(&self, name: &str, shape: &[usize]) -> Result<Vec<i8>, &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        let count: usize = shape.iter().product();
        if entry.kind != Kind::I8 || entry.shape != shape || entry.end - entry.begin != count as u64
        {
            return Err("model_tensor_invalid");
        }
        let mut out = vec![0i8; count];
        // SAFETY: `out` owns `count` initialized bytes; any bit pattern is a valid i8.
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count) };
        self.file
            .read_exact_at(bytes, self.data + entry.begin)
            .map_err(|_| "model_file_invalid")?;
        Ok(out)
    }
    /// Reads one F32, F16 or BF16 tensor as f32 (all exact), checking its exact shape.
    pub fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, &'static str> {
        let entry = self.entries.get(name).ok_or("model_tensor_missing")?;
        match entry.kind {
            Kind::F32 => self.f32(name, shape),
            Kind::F16 => self.f16(name, shape),
            Kind::Bf16 => {
                let count: usize = shape.iter().product();
                if entry.shape != shape || entry.end - entry.begin != count as u64 * 2 {
                    return Err("model_tensor_invalid");
                }
                let mut bytes = vec![0u8; count * 2];
                self.file
                    .read_exact_at(&mut bytes, self.data + entry.begin)
                    .map_err(|_| "model_file_invalid")?;
                // BF16 is the upper half of the f32 bit pattern.
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                    .collect())
            }
            _ => Err("model_tensor_invalid"),
        }
    }
    /// Number of distinct `<prefix><index>.` groups, for counting layers.
    pub fn count(&self, prefix: &str) -> usize {
        let mut seen = std::collections::HashSet::new();
        for name in self.entries.keys() {
            if let Some((index, _)) = name
                .strip_prefix(prefix)
                .and_then(|rest| rest.split_once('.'))
            {
                seen.insert(index);
            }
        }
        seen.len()
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
/// binary32 to binary16, rounding to nearest even; overflow becomes infinity.
pub fn narrow(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 255) as i32 - 112;
    let fraction = bits & 0x7f_ffff;
    if exponent == 143 {
        return sign | 0x7c00 | if fraction != 0 { 0x200 } else { 0 };
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    // Shift out the bits that do not fit, then round on what was shifted out.
    let (kept, shift) = if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        (fraction | 0x80_0000, (14 - exponent) as u32)
    } else {
        ((exponent as u32) << 23 | fraction, 13)
    };
    let (half, rest, middle) = (kept >> shift, kept & ((1 << shift) - 1), 1 << (shift - 1));
    let up = rest > middle || (rest == middle && half & 1 == 1);
    sign | (half + u32::from(up)) as u16
}
#[cfg(test)]
mod tests {
    use super::{narrow, widen};
    #[test]
    fn narrows_to_nearest_even_and_round_trips() {
        for half in 0..=u16::MAX {
            let wide = widen(half);
            if wide.is_nan() {
                assert!(widen(narrow(wide)).is_nan());
            } else {
                assert_eq!(narrow(wide), half);
            }
        }
        assert_eq!(narrow(1.0 + 2f32.powi(-11)), 0x3c00); // tie to even, down
        assert_eq!(narrow(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02); // tie to even, up
        assert_eq!(narrow(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
        assert_eq!(narrow(65520.0), 0x7c00);
        assert_eq!(narrow(65519.0), 0x7bff);
        assert_eq!(narrow(1e9), 0x7c00);
        assert_eq!(narrow(-1e-9), 0x8000);
        assert_eq!(narrow(2f32.powi(-25)), 0);
        assert_eq!(narrow(2f32.powi(-25) * 1.5), 1);
        assert_eq!(narrow(2f32.powi(-14) - 2f32.powi(-26)), 0x0400); // rounds up into normals
    }
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
