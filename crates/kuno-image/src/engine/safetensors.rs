//! Minimal safetensors reader: header index plus positioned reads of single tensors.
use std::{collections::HashMap, fs::File, os::unix::fs::FileExt, path::Path};

struct Entry {
    f32: bool,
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
}
