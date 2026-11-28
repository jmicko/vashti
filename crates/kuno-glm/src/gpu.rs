//! The GPU engine (Vulkan, through kuno-image's device layer): the attention, mixers, norms,
//! router and shared expert of every block, and the experts kept on the device; the CPU
//! (`cpu`) runs the experts in host memory.
use kuno_image::engine::gpu::{Buffer, Gpu, Kernel, Readback, Step, bytes_of};
use kuno_text::TensorType;

use crate::{Matrix, Result};

/// The device's kernels, at one subgroup size (`lanes`).
pub struct Kernels {
    pub lanes: u32,
    pub mv: Kernel,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let lanes = gpu.subgroup().unwrap_or(32).clamp(8, 128);
        Ok(Self {
            lanes,
            mv: gpu.kernel(include_bytes!("../shaders/mv.spv"), 4, 9, Some(lanes))?,
        })
    }
}

/// A device buffer holding `bytes`, read back and compared before use: on the RTX 3060
/// uploads occasionally arrive with whole 64 KiB pages wrong (see kuno-audio). A mismatch is
/// uploaded again, up to three times. Intel's devices take uploads unchecked (the fault is
/// NVIDIA's; checking doubles the bus traffic of a load).
pub fn filled(gpu: &Gpu, bytes: &[u8]) -> Result<Buffer> {
    let size = bytes.len().max(4).next_multiple_of(4);
    if gpu.vendor() == 0x8086 {
        return gpu.filled(size as u64, |out| {
            out[..bytes.len()].copy_from_slice(bytes);
            out[bytes.len()..].fill(0);
            Ok(())
        });
    }
    for _ in 0..3 {
        let buffer = gpu.filled(size as u64, |out| {
            out[..bytes.len()].copy_from_slice(bytes);
            out[bytes.len()..].fill(0);
            Ok(())
        })?;
        if verified(gpu, &buffer, bytes, size)? {
            return Ok(buffer);
        }
    }
    Err("glm_gpu_upload_corrupt")
}

const PIECE: usize = 16 << 20;
thread_local! {
    static CHECK: std::cell::RefCell<Option<(usize, Readback)>> = const { std::cell::RefCell::new(None) };
}

fn verified(gpu: &Gpu, buffer: &Buffer, bytes: &[u8], size: usize) -> Result<bool> {
    CHECK.with(|cell| {
        let mut cell = cell.borrow_mut();
        let key = gpu as *const Gpu as usize;
        if cell.as_ref().is_none_or(|(k, _)| *k != key) {
            *cell = Some((key, gpu.readback(PIECE as u64)?));
        }
        let stage = &cell.as_ref().unwrap().1;
        for start in (0..size).step_by(PIECE) {
            let count = PIECE.min(size - start) / 4;
            let same = gpu.floats_with(buffer, start as u64, stage, count, |got| {
                let got = bytes_of(got);
                let want = &bytes[start.min(bytes.len())..(start + got.len()).min(bytes.len())];
                got[..want.len()] == *want && got[want.len()..].iter().all(|b| *b == 0)
            })?;
            if !same {
                return Ok(false);
            }
        }
        Ok(true)
    })
}

/// A weight on the device, laid out for `mv`.
pub struct Mat {
    pub kind: u32,
    pub rows: usize,
    pub cols: usize,
    pub w: Buffer,
    pub s: Buffer,
    /// Few rows, each given a whole workgroup (as `step` runs it: mv.comp's flag 4).
    pub wide: bool,
}

/// Q6_K blocks (210 bytes per 256 values) as `mv`'s kind 3: per block 13 uvec4 (low nibbles,
/// 2-bit highs, int8 scales) and the block's scale as f32.
pub fn repack_q6k(bytes: &[u8], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    use rayon::prelude::*;
    let blocks = rows * cols / 256;
    let mut w = vec![0u8; blocks * 208];
    let mut s = vec![0f32; blocks];
    bytes
        .par_chunks_exact(210)
        .zip(w.par_chunks_exact_mut(208).zip(s.par_iter_mut()))
        .for_each(|(src, (dst, d))| {
            let (ql, qh, sc) = (&src[..128], &src[128..192], &src[192..208]);
            *d = kuno_text::quant::f16_to_f32(u16::from_le_bytes([src[208], src[209]]));
            let mut q = [0u8; 256];
            for half in 0..2 {
                let (ql, qh) = (&ql[half * 64..], &qh[half * 32..]);
                for l in 0..32 {
                    let y = &mut q[half * 128..];
                    y[l] = (ql[l] & 15) | ((qh[l] & 3) << 4);
                    y[l + 32] = (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4);
                    y[l + 64] = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                    y[l + 96] = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                }
            }
            let (lo, rest) = dst.split_at_mut(128);
            let (hi, scales) = rest.split_at_mut(64);
            for (v, &value) in q.iter().enumerate() {
                lo[v / 2] |= (value & 15) << (4 * (v % 2));
                hi[v / 4] |= (value >> 4) << (2 * (v % 4));
            }
            scales.copy_from_slice(sc);
        });
    (w, s)
}

/// Q8_0 blocks (34 bytes per 32 values) as `mv`'s kind 0: the int8 values, then one f16
/// scale per 32 values.
pub fn repack_q8_0(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut w = Vec::with_capacity(bytes.len() / 34 * 32);
    let mut s = Vec::with_capacity(bytes.len() / 34 * 2);
    for block in bytes.chunks_exact(34) {
        s.extend_from_slice(&block[..2]);
        w.extend_from_slice(&block[2..]);
    }
    (w, s)
}

impl Mat {
    pub fn upload(gpu: &Gpu, m: &Matrix) -> Result<Self> {
        let (kind, w, s): (u32, Vec<u8>, Vec<u8>) = match m.dtype {
            TensorType::Q6K => {
                let (w, s) = repack_q6k(&m.bytes, m.rows, m.cols);
                (3, w, bytes_of(&s).to_vec())
            }
            TensorType::Q8_0 => {
                let (w, s) = repack_q8_0(&m.bytes);
                (0, w, s)
            }
            TensorType::F32 => (2, m.bytes.clone(), vec![0u8; 4]),
            _ => return Err("gpu_weight_type_unsupported"),
        };
        Ok(Self {
            kind,
            rows: m.rows,
            cols: m.cols,
            w: filled(gpu, &w)?,
            s: filled(gpu, &s)?,
            wide: false,
        })
    }

    /// `y[y_off..][..rows] (+)= self x[x_off..]` (offsets in floats).
    pub fn step<'a>(
        &'a self,
        k: &'a Kernels,
        buffers: &'a [&'a Buffer; 4],
        constants: &'a [u32; 9],
    ) -> Step<'a> {
        let rows_per_group = 128 / k.lanes;
        // The rows named in the constants (a leading part of the matrix may be used).
        let groups = constants[0].div_ceil(rows_per_group);
        Step {
            kernel: &k.mv,
            buffers,
            constants,
            groups: [groups.min(65535), groups.div_ceil(65535), 1],
        }
    }
    /// Offsets in floats (`x_off` a multiple of 4).
    pub fn constants(&self, x_off: usize, y_off: usize, add: bool) -> [u32; 9] {
        assert!(x_off.is_multiple_of(4));
        [
            self.rows as u32,
            self.cols as u32,
            self.kind,
            x_off as u32,
            y_off as u32,
            0,
            if add { 2 } else { 0 },
            0,
            0,
        ]
    }
    /// Per-head products: row r reads x from `x_off + (r / group) * gstride`.
    pub fn grouped(&self, x_off: usize, y_off: usize, group: usize, gstride: usize) -> [u32; 9] {
        let mut c = self.constants(x_off, y_off, false);
        assert!(gstride.is_multiple_of(4));
        c[7] = group as u32;
        c[8] = gstride as u32;
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    fn model() -> Option<Model> {
        let dir = std::env::var_os("KUNO_GLM_MODEL")?;
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        shards.sort();
        Model::open(&shards).ok()
    }
    fn gpu() -> Gpu {
        let pci = std::env::var("KUNO_GLM_GPU").unwrap_or_else(|_| "0000:2d:00.0".into());
        Gpu::open(kuno_image::engine::gpu::Select::Pci(pci)).unwrap()
    }

    #[test]
    fn q6k_repack_keeps_every_value() {
        // One block with distinct codes: decode it both ways.
        let mut block = [0u8; 210];
        for (i, b) in block.iter_mut().enumerate().take(208) {
            *b = (i * 37 + 11) as u8;
        }
        block[208..].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        let mut want = vec![0f32; 256];
        TensorType::Q6K.decode(&block, &mut want).unwrap();
        let (w, s) = repack_q6k(&block, 1, 256);
        for v in 0..256 {
            let lo = (w[v / 2] >> (4 * (v % 2))) & 15;
            let hi = (w[128 + v / 4] >> (2 * (v % 4))) & 3;
            let q = i32::from(lo | (hi << 4)) - 32;
            let got = s[0] * f32::from(w[192 + v / 16] as i8) * q as f32;
            assert_eq!(got, want[v], "value {v}");
        }
    }

    /// Real tensors of each GPU type against the CPU reference's products; prints the time
    /// and bandwidth of each. KUNO_GLM_MODEL names the shard directory.
    #[test]
    #[ignore = "needs a GPU and the model"]
    fn products_match_cpu() {
        let Some(model) = model() else {
            eprintln!("KUNO_GLM_MODEL unset");
            return;
        };
        let gpu = gpu();
        let k = Kernels::new(&gpu).unwrap();
        // The long head first: the device's clocks rise during it (the first short passes
        // otherwise run at idle clocks).
        for name in [
            "output.weight",
            "blk.5.attn_q.weight",
            "blk.5.attn_output.weight",
            "blk.3.attn_q_b.weight",
            "blk.3.ffn_gate_inp.weight",
            "blk.5.hc_attn_fn.weight",
            "blk.5.attn_q.weight",
        ] {
            let m = model.matrix(name).unwrap();
            let x: Vec<f32> = (0..m.cols)
                .map(|i| ((i * 7919) % 251) as f32 / 61.0 - 2.0)
                .collect();
            let want = m.matvec(&x);
            let g = Mat::upload(&gpu, &m).unwrap();
            let xb = filled(&gpu, bytes_of(&x)).unwrap();
            let yb = gpu.buffer((m.rows * 4) as u64).unwrap();
            let buffers = [&g.w, &g.s, &xb, &yb];
            let constants = g.constants(0, 0, false);
            let steps: Vec<Step> = (0..20).map(|_| g.step(&k, &buffers, &constants)).collect();
            let times = gpu
                .pass_profiled(&steps, &vec![false; steps.len()])
                .unwrap();
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ns = sorted[sorted.len() / 2];
            let stage = gpu.readback((m.rows * 4) as u64).unwrap();
            let mut got = vec![0f32; m.rows];
            gpu.read_floats(&yb, 0, &stage, &mut got).unwrap();
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-6);
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let bytes = m.bytes.len() as f64;
            eprintln!(
                "{name} {:?} {}x{}: {:.1} us, {:.0} GB/s, worst {:.2e} of {:.2e}",
                m.dtype,
                m.rows,
                m.cols,
                ns / 1e3,
                bytes / ns,
                worst,
                scale
            );
            assert!(worst <= 1e-4 * scale, "{name}");
            if name == "output.weight" {
                // The first `rows` rows: time against size.
                for rows in [1024usize, 4096, 8192, 16384, 32768, 65536] {
                    let mut c = constants;
                    c[0] = rows as u32;
                    let steps: Vec<Step> = (0..20).map(|_| g.step(&k, &buffers, &c)).collect();
                    let mut t = gpu
                        .pass_profiled(&steps, &vec![false; steps.len()])
                        .unwrap();
                    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let bytes = bytes * rows as f64 / m.rows as f64;
                    eprintln!(
                        "  {rows} rows: {:.1} us, {:.0} GB/s",
                        t[10] / 1e3,
                        bytes / t[10]
                    );
                }
            }
        }
    }
}
