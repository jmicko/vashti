//! Offline LoRA merge into an INT8 ConvRot (int8_tensorwise, per-row scale) safetensors model.
//!
//! Stored weights are q = round(T(W) / s_row), where T is the per-group product of symmetric,
//! normalized H4 stages (an involution). The merge works in the rotated basis:
//! M = q * s_row + T(dW), then each row is requantized with s = max|M_row| / 127 and seeded
//! stochastic rounding, as ComfyUI's recalculated-scale requantize does. Round-to-nearest would
//! discard most of the LoRA: its deltas are far below one INT8 step.
//! dW = multiplier * alpha / rank * B @ A. BF16 linears get W + dW with stochastic BF16 rounding.
//! Every other tensor, and the header, is copied byte for byte.
//!
//! The arithmetic is plain IEEE f32 in a fixed order per element (Rust never contracts to FMA)
//! and the noise is a counter-based hash of (module, element), so the output is bit-identical on
//! every x86-64 machine and thread count. That makes the result pinnable by SHA-256.
use rayon::prelude::*;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Write, os::unix::fs::FileExt, path::Path};

type Result<T> = std::result::Result<T, &'static str>;
const MAX_HEADER: u64 = 64 * 1024 * 1024;

struct Tensors {
    file: File,
    base: u64,
    header: Map<String, Value>,
    header_bytes: Vec<u8>,
}

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

impl Tensors {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|_| "model_file_unavailable")?;
        let mut size = [0u8; 8];
        file.read_exact_at(&mut size, 0)
            .map_err(|_| "model_read_failed")?;
        let n = u64::from_le_bytes(size);
        if n > MAX_HEADER {
            return Err("model_header_invalid");
        }
        let mut header_bytes = vec![0u8; n as usize];
        file.read_exact_at(&mut header_bytes, 8)
            .map_err(|_| "model_read_failed")?;
        let header = serde_json::from_slice::<Value>(&header_bytes)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .ok_or("model_header_invalid")?;
        Ok(Self {
            file,
            base: 8 + n,
            header,
            header_bytes,
        })
    }
    fn entry(&self, name: &str) -> Result<Entry> {
        let e = self.header.get(name).ok_or("lora_merge_tensor_missing")?;
        let offsets = e["data_offsets"].as_array().ok_or("model_header_invalid")?;
        let (start, end) = match offsets.as_slice() {
            [a, b] => (
                a.as_u64().ok_or("model_header_invalid")?,
                b.as_u64().ok_or("model_header_invalid")?,
            ),
            _ => return Err("model_header_invalid"),
        };
        let shape = e["shape"]
            .as_array()
            .ok_or("model_header_invalid")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or("model_header_invalid")?;
        let dtype = e["dtype"]
            .as_str()
            .ok_or("model_header_invalid")?
            .to_owned();
        if end < start {
            return Err("model_header_invalid");
        }
        Ok(Entry {
            dtype,
            shape,
            start,
            end,
        })
    }
    fn bytes(&self, entry: &Entry) -> Result<Vec<u8>> {
        let mut out = vec![0u8; (entry.end - entry.start) as usize];
        self.file
            .read_exact_at(&mut out, self.base + entry.start)
            .map_err(|_| "model_read_failed")?;
        Ok(out)
    }
    fn f32s(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let e = self.entry(name)?;
        let raw = self.bytes(&e)?;
        let values = match e.dtype.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            _ => return Err("lora_merge_dtype_unsupported"),
        };
        Ok((values, e.shape))
    }
}

/// SplitMix64 of a per-module seed and element index.
fn noise(seed: u64, index: u64) -> u64 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn uniform(seed: u64, index: u64) -> f32 {
    (noise(seed, index) >> 40) as f32 * (1.0 / 16_777_216.0)
}
fn module_seed(module: &str) -> u64 {
    u64::from_le_bytes(Sha256::digest(module.as_bytes())[..8].try_into().unwrap())
}

/// T along one row: H4 stages of stride 1, 4, 16, ... within each group.
fn rotate(row: &mut [f32], group: usize) {
    for chunk in row.chunks_exact_mut(group) {
        let mut stride = 1;
        while stride < group {
            for block in chunk.chunks_exact_mut(4 * stride) {
                for t in 0..stride {
                    let (a, b, c, d) = (
                        block[t],
                        block[stride + t],
                        block[2 * stride + t],
                        block[3 * stride + t],
                    );
                    block[t] = (a + b + c - d) * 0.5;
                    block[stride + t] = (a + b - c + d) * 0.5;
                    block[2 * stride + t] = (a - b + c + d) * 0.5;
                    block[3 * stride + t] = (-a + b + c + d) * 0.5;
                }
            }
            stride *= 4;
        }
    }
}

/// dW row i = scale * sum_k B[i,k] * A[k,:], accumulated in k order.
fn delta_row(
    b: &[f32],
    a: &[f32],
    rank: usize,
    cols: usize,
    i: usize,
    scale: f32,
    out: &mut [f32],
) {
    out.fill(0.0);
    for k in 0..rank {
        let bk = b[i * rank + k];
        for (o, a) in out.iter_mut().zip(&a[k * cols..(k + 1) * cols]) {
            *o += bk * *a;
        }
    }
    out.iter_mut().for_each(|v| *v *= scale);
}

struct Lora {
    a: Vec<f32>,
    b: Vec<f32>,
    rank: usize,
    cols: usize,
    rows: usize,
    scale: f32,
}

fn lora_for(
    lora: &Tensors,
    module: &str,
    multiplier: f64,
    rows: usize,
    cols: usize,
) -> Result<Lora> {
    let prefix = format!("diffusion_model.{module}");
    let (a, a_shape) = lora.f32s(&format!("{prefix}.lora_A.weight"))?;
    let (b, b_shape) = lora.f32s(&format!("{prefix}.lora_B.weight"))?;
    let (alpha, _) = lora.f32s(&format!("{prefix}.alpha"))?;
    let rank = *a_shape.first().ok_or("lora_merge_shape_invalid")?;
    if a_shape != [rank, cols] || b_shape != [rows, rank] || alpha.len() != 1 || rank == 0 {
        return Err("lora_merge_shape_invalid");
    }
    // Computed in f64 then rounded, as numpy's float32(multiplier * alpha / rank).
    let scale = (multiplier * f64::from(alpha[0]) / rank as f64) as f32;
    Ok(Lora {
        a,
        b,
        rank,
        cols,
        rows,
        scale,
    })
}

/// Returns (weight bytes, scale bytes) for an INT8 ConvRot linear.
fn merge_int8(
    base: &Tensors,
    lora: &Tensors,
    module: &str,
    multiplier: f64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let conf_entry = base.entry(&format!("{module}.comfy_quant"))?;
    let conf: Value = serde_json::from_slice(&base.bytes(&conf_entry)?)
        .map_err(|_| "lora_merge_quant_unsupported")?;
    if conf["format"] != "int8_tensorwise" || conf["convrot"] != true {
        return Err("lora_merge_quant_unsupported");
    }
    let group = conf["convrot_groupsize"]
        .as_u64()
        .ok_or("lora_merge_quant_unsupported")? as usize;
    let weight = base.entry(&format!("{module}.weight"))?;
    let (rows, cols) = match weight.shape.as_slice() {
        [r, c] if weight.dtype == "I8" => (*r, *c),
        _ => return Err("lora_merge_shape_invalid"),
    };
    if group < 4
        || !group.is_power_of_two()
        || !group.trailing_zeros().is_multiple_of(2)
        || !cols.is_multiple_of(group)
    {
        return Err("lora_merge_quant_unsupported");
    }
    let q = base.bytes(&weight)?;
    let (s, s_shape) = base.f32s(&format!("{module}.weight_scale"))?;
    if s.len() != rows || s_shape.iter().product::<usize>() != rows || q.len() != rows * cols {
        return Err("lora_merge_shape_invalid");
    }
    let lora = lora_for(lora, module, multiplier, rows, cols)?;
    let seed = module_seed(module);
    let mut q_new = vec![0u8; rows * cols];
    let mut s_new = vec![0f32; rows];
    q_new
        .par_chunks_mut(cols)
        .zip(s_new.par_iter_mut())
        .enumerate()
        .for_each_init(
            || vec![0f32; cols],
            |m, (i, (q_row, s_out))| {
                delta_row(&lora.b, &lora.a, lora.rank, lora.cols, i, lora.scale, m);
                rotate(m, group);
                for (j, v) in m.iter_mut().enumerate() {
                    *v += f32::from(q[i * cols + j] as i8) * s[i];
                }
                let s_row = m.iter().fold(0f32, |acc, v| acc.max(v.abs())) / 127.0;
                let safe = if s_row > 0.0 { s_row } else { 1.0 };
                for (j, (out, v)) in q_row.iter_mut().zip(m.iter()).enumerate() {
                    let u = uniform(seed, (i * cols + j) as u64);
                    *out = (*v / safe + u).floor().clamp(-127.0, 127.0) as i8 as u8;
                }
                *s_out = s_row;
            },
        );
    let _ = lora.rows;
    Ok((q_new, s_new.iter().flat_map(|v| v.to_le_bytes()).collect()))
}

/// W + dW with stochastic rounding to BF16 for unquantized linears.
fn merge_bf16(base: &Tensors, lora: &Tensors, module: &str, multiplier: f64) -> Result<Vec<u8>> {
    let (w, shape) = base.f32s(&format!("{module}.weight"))?;
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return Err("lora_merge_shape_invalid"),
    };
    let lora = lora_for(lora, module, multiplier, rows, cols)?;
    let seed = module_seed(module);
    let mut out = vec![0u8; rows * cols * 2];
    out.par_chunks_mut(cols * 2).enumerate().for_each_init(
        || vec![0f32; cols],
        |m, (i, row)| {
            delta_row(&lora.b, &lora.a, lora.rank, lora.cols, i, lora.scale, m);
            for (j, (bytes, d)) in row.chunks_exact_mut(2).zip(m.iter()).enumerate() {
                let v = w[i * cols + j] + *d;
                let bits = u64::from(v.to_bits());
                let rounded = if v.is_finite() {
                    ((bits + (noise(seed, (i * cols + j) as u64) >> 48)) >> 16) as u16
                } else {
                    (bits >> 16) as u16
                };
                bytes.copy_from_slice(&rounded.to_le_bytes());
            }
        },
    );
    Ok(out)
}

/// Writes `out` (which must not exist) and returns its SHA-256. The header is unchanged:
/// every merged tensor keeps its dtype, shape and offsets.
pub fn merge(
    base_path: &Path,
    lora_path: &Path,
    out: &Path,
    multiplier: f64,
    progress: impl Fn(usize, usize),
) -> Result<String> {
    let base = Tensors::open(base_path)?;
    let lora = Tensors::open(lora_path)?;
    let modules: Vec<String> = lora
        .header
        .keys()
        .filter_map(|k| {
            k.strip_suffix(".lora_A.weight")?
                .strip_prefix("diffusion_model.")
                .map(str::to_owned)
        })
        .collect();
    if modules.is_empty() {
        return Err("lora_merge_empty");
    }
    // Tensor name -> owning module.
    let mut owner = BTreeMap::new();
    for m in &modules {
        owner.insert(format!("{m}.weight"), m.clone());
        if base.header.contains_key(&format!("{m}.weight_scale")) {
            owner.insert(format!("{m}.weight_scale"), m.clone());
        }
    }
    let mut order = Vec::new();
    for name in base.header.keys().filter(|k| *k != "__metadata__") {
        let e = base.entry(name)?;
        order.push((e.start, e.end, name.clone()));
    }
    order.sort();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .map_err(|_| "lora_merge_output_unavailable")?;
    let mut hash = Sha256::new();
    let mut write = |bytes: &[u8]| -> Result<()> {
        hash.update(bytes);
        file.write_all(bytes).map_err(|_| "lora_merge_write_failed")
    };
    write(&(base.header_bytes.len() as u64).to_le_bytes())?;
    write(&base.header_bytes)?;
    let mut pending: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let (mut position, mut done) = (0u64, 0usize);
    let mut chunk = vec![0u8; 64 << 20];
    for (start, end, name) in order {
        if start != position {
            return Err("model_header_invalid");
        }
        if let Some(module) = owner.get(&name) {
            if !pending.contains_key(&name) {
                if base.entry(&format!("{module}.weight"))?.dtype == "BF16" {
                    pending.insert(
                        format!("{module}.weight"),
                        merge_bf16(&base, &lora, module, multiplier)?,
                    );
                } else {
                    let (q, s) = merge_int8(&base, &lora, module, multiplier)?;
                    pending.insert(format!("{module}.weight"), q);
                    pending.insert(format!("{module}.weight_scale"), s);
                }
                done += 1;
                progress(done, modules.len());
            }
            let data = pending.remove(&name).ok_or("lora_merge_tensor_missing")?;
            if data.len() as u64 != end - start {
                return Err("lora_merge_shape_invalid");
            }
            write(&data)?;
        } else {
            let mut at = start;
            while at < end {
                let n = ((end - at) as usize).min(chunk.len());
                base.file
                    .read_exact_at(&mut chunk[..n], base.base + at)
                    .map_err(|_| "model_read_failed")?;
                write(&chunk[..n])?;
                at += n as u64;
            }
        }
        position = end;
    }
    if !pending.is_empty() || done != modules.len() {
        return Err("lora_merge_tensor_missing");
    }
    file.sync_all().map_err(|_| "lora_merge_write_failed")?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_is_an_involution_and_orthonormal() {
        let original: Vec<f32> = (0..512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 8.0)
            .collect();
        let mut row = original.clone();
        rotate(&mut row, 256);
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((norm(&row) / norm(&original) - 1.0).abs() < 1e-5);
        rotate(&mut row, 256);
        for (a, b) in row.iter().zip(&original) {
            assert!((a - b).abs() < 1e-4);
        }
    }
    #[test]
    fn noise_is_uniform_and_stable() {
        let mean = (0..100_000).map(|i| uniform(7, i)).sum::<f32>() / 100_000.0;
        assert!((mean - 0.5).abs() < 0.01);
        assert!((0..1000).all(|i| (0.0..1.0).contains(&uniform(3, i))));
        // Pinned: the derived model's checksum depends on this exact sequence.
        assert_eq!(noise(0, 0), 0xe220_a839_7b1d_cdaf);
    }
    #[test]
    fn delta_row_matches_a_direct_sum() {
        let (rank, cols) = (3, 4);
        let a: Vec<f32> = (0..rank * cols).map(|v| v as f32 * 0.5).collect();
        let b: Vec<f32> = (0..2 * rank).map(|v| v as f32 - 1.0).collect();
        let mut out = vec![0f32; cols];
        delta_row(&b, &a, rank, cols, 1, 2.0, &mut out);
        for j in 0..cols {
            let expect: f32 = (0..rank)
                .map(|k| b[rank + k] * a[k * cols + j])
                .sum::<f32>()
                * 2.0;
            assert_eq!(out[j], expect);
        }
    }
}

/// Development run over real weights:
/// KUNO_MERGE_BASE=... KUNO_MERGE_LORA=... KUNO_MERGE_OUT=... cargo test --release -- --ignored real_merge
#[cfg(test)]
#[test]
#[ignore]
fn real_merge() {
    let var = |k| std::path::PathBuf::from(std::env::var_os(k).expect(k));
    let start = std::time::Instant::now();
    let sha = merge(
        &var("KUNO_MERGE_BASE"),
        &var("KUNO_MERGE_LORA"),
        &var("KUNO_MERGE_OUT"),
        1.0,
        |d, n| {
            if d % 20 == 0 || d == n {
                eprintln!("merged {d}/{n} at {:.1} s", start.elapsed().as_secs_f64());
            }
        },
    )
    .unwrap();
    eprintln!("sha256 {sha} in {:.1} s", start.elapsed().as_secs_f64());
}
