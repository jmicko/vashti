//! Qwen Image 2.1's diffusion transformer on any Vulkan GPU with cooperative matrices (the
//! RTX 3060 among them): `qwen_gpu.rs`'s plan with every kernel a Vulkan shader. The INT8
//! products are `gemm8` in the device's form, attention is the fused `fattend16`, and the
//! glue is `qwen_gpu.rs`'s shaders writing plain row layouts, plus `qkv16` (queries, keys and
//! values from the product's sums) and `qgate16` (the feed-forward activation). Checked
//! against the same captured steps as `qwen_gpu.rs`.
//!
//! As there, the prompt and reference images form a prefix run once per request, whose keys
//! and values are kept per block; a step then runs only the target's tokens, its own keys
//! first and the prefix's after them.
use std::{collections::HashMap, path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit_gpu::{Call, run},
    gpu::{Buffer, Gpu, Kernel, Readback, bytes_of, shaders},
    qwen_dit::{self, CHANNELS, CONTEXT, DIM, HEAD, HEADS, HIDDEN, Layout, PAIRS, QwenDit},
    qwen_gpu::{
        KEY_NORM, MODS, QUERY_NORM, SCALE_FC1, SCALE_FC2, SCALE_OUT, SCALE_QKV, SPILLED, Small,
        Times, VECTORS,
    },
    safetensors::Safetensors,
};

/// Tokens are padded to the products' token block (and `fattend16`'s query block).
const PAD: usize = 128;
/// Keys are padded to `fattend16`'s key block.
const KEY_PAD: usize = 64;
/// The four integer products of a block: weight rows and columns.
const PRODUCTS: [(usize, usize); 4] =
    [(3 * DIM, DIM), (DIM, DIM), (2 * HIDDEN, DIM), (DIM, HIDDEN)];
const QKV: usize = 0;
const OUT: usize = 1;
const FC1: usize = 2;
const FC2: usize = 3;

struct Kernels {
    gemm8: Kernel,
    normq8: Kernel,
    resnorm8: Kernel,
    resid8: Kernel,
    rotate8: Kernel,
    hidq8: Kernel,
    qkv: Kernel,
    gate: Kernel,
    copy: Kernel,
    attend: Kernel,
    /// Queries per `attend` workgroup.
    attend_block: u32,
    embed: Kernel,
    final_: Kernel,
}
impl Kernels {
    fn new(gpu: &Gpu) -> Result<Self, &'static str> {
        // Attention needs 16 x 16 x 16 half-precision matrices.
        if !gpu.matrix2() && !gpu.has_half16() {
            return Err("gpu_unsupported");
        }
        let any = gpu.subgroup();
        Ok(Self {
            gemm8: gpu.gemm8()?,
            normq8: gpu.kernel(shaders::QNORMQ8, 4, 3, any)?,
            resnorm8: gpu.kernel(shaders::QRESNORM8, 6, 5, any)?,
            resid8: gpu.kernel(shaders::QRESID8, 5, 3, None)?,
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 4, any)?,
            hidq8: gpu.kernel(shaders::QHIDQ8, 4, 2, any)?,
            qkv: gpu.kernel(shaders::QKV16, 9, 9, any)?,
            gate: gpu.kernel(shaders::QGATE16, 4, 2, None)?,
            copy: gpu.kernel(shaders::QCOPY16, 4, 4, None)?,
            attend: if gpu.matrix2() {
                gpu.kernel(shaders::FATTEND16_NV, 4, 7, None)?
            } else {
                gpu.kernel(shaders::FATTEND16, 4, 7, Some(32))?
            },
            attend_block: if gpu.matrix2() { 64 } else { 32 },
            embed: gpu.kernel(shaders::QEMBED, 3, 1, None)?,
            final_: gpu.kernel(shaders::QFINAL, 4, 1, any)?,
        })
    }
}

/// One block's weights: the four INT8 matrices as plain rows (query, key and value rows as
/// one matrix, then gates and values of the feed-forward input), and the vectors at
/// `qwen_gpu.rs`'s offsets.
struct Layer {
    matrices: [Buffer; 4],
    vectors: Buffer,
}
impl Layer {
    fn load(gpu: &Gpu, file: &Safetensors, layer: usize) -> Result<Self, &'static str> {
        let name = |part: &str| format!("transformer_blocks.{layer}.{part}");
        let matrix = |parts: &[&str], which: usize| -> Result<Buffer, &'static str> {
            let (rows, columns) = PRODUCTS[which];
            let per_part = rows / parts.len();
            let staged = gpu.staged((rows * columns) as u64, |to| {
                to.par_chunks_mut(per_part * columns)
                    .zip(parts)
                    .try_for_each(|(to, part)| {
                        let tensor = name(&format!("{part}.weight"));
                        file.i8_part_into(&tensor, &[per_part, columns], 0, to)
                    })
            })?;
            gpu.place(&staged, false)
        };
        let mut vectors = Vec::with_capacity(VECTORS);
        vectors.extend(file.floats(&name("attn.norm_q.weight"), &[HEAD])?);
        vectors.extend(file.floats(&name("attn.norm_k.weight"), &[HEAD])?);
        for part in ["attn.to_q", "attn.to_k", "attn.to_v"] {
            vectors.extend(file.f32(&name(&format!("{part}.weight_scale")), &[DIM, 1])?);
        }
        vectors.extend(file.f32(&name("attn.to_out.0.weight_scale"), &[DIM, 1])?);
        vectors.extend(file.f32(&name("img_mlp.gate_up.weight_scale"), &[2 * HIDDEN, 1])?);
        vectors.extend(file.f32(&name("img_mlp.out.weight_scale"), &[DIM, 1])?);
        Ok(Self {
            matrices: [
                matrix(&["attn.to_q", "attn.to_k", "attn.to_v"], QKV)?,
                matrix(&["attn.to_out.0"], OUT)?,
                matrix(&["img_mlp.gate_up"], FC1)?,
                matrix(&["img_mlp.out"], FC2)?,
            ],
            vectors: floats(gpu, &vectors)?,
        })
    }
}

fn floats(gpu: &Gpu, values: &[f32]) -> Result<Buffer, &'static str> {
    gpu.filled(values.len() as u64 * 4, |to| {
        to.copy_from_slice(bytes_of(values));
        Ok(())
    })
}

/// Buffers of one pass size: `tokens` own tokens after `earlier` cached keys (set per pass;
/// the key and value rows have room for up to `stride` keys per head).
struct Work {
    /// The prefix whose target turns `turns` holds (0: none).
    turns_of: std::cell::Cell<u64>,
    tokens: usize,
    earlier: std::cell::Cell<usize>,
    padded: usize,
    /// Key and value rows per head: own then earlier, padded.
    stride: usize,
    stream: Buffer,
    turns: Buffer,
    rows: Buffer,
    mods: Buffer,
    final_scale: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    hidden: Buffer,
    spill: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    mixed: Buffer,
    out: Buffer,
    readback: Readback,
}
impl Work {
    /// The device memory `new` takes.
    fn bytes(tokens: usize, earlier: usize) -> u64 {
        let padded = tokens.next_multiple_of(PAD);
        let keys = (tokens + earlier).next_multiple_of(KEY_PAD);
        let f32s = padded * (2 * DIM + PAIRS * 2 + 2 * CHANNELS + 1 + 2 * HIDDEN + SPILLED)
            + 2 * MODS
            + DIM;
        let halves = padded * HIDDEN + HEADS * padded * HEAD + 2 * HEADS * keys * HEAD;
        (f32s * 4 + padded * HIDDEN + halves * 2) as u64
    }
    fn new(gpu: &Gpu, tokens: usize, earlier: usize) -> Result<Self, &'static str> {
        let padded = tokens.next_multiple_of(PAD);
        let stride = (tokens + earlier).next_multiple_of(KEY_PAD);
        let f32s = |count: usize| gpu.buffer(count as u64 * 4);
        let halves = |count: usize| gpu.buffer(count as u64 * 2);
        Ok(Self {
            turns_of: std::cell::Cell::new(0),
            tokens,
            earlier: std::cell::Cell::new(earlier),
            padded,
            stride,
            stream: f32s(padded * DIM)?,
            turns: f32s(padded * PAIRS * 2)?,
            rows: f32s(padded * CHANNELS)?,
            mods: f32s(2 * MODS)?,
            final_scale: f32s(DIM)?,
            bytes: gpu.buffer((padded * HIDDEN) as u64)?,
            scales: f32s(padded)?,
            sums: f32s(padded * 2 * HIDDEN)?,
            hidden: halves(padded * HIDDEN)?,
            spill: f32s(padded * SPILLED)?,
            query: halves(HEADS * padded * HEAD)?,
            key: halves(HEADS * stride * HEAD)?,
            value: halves(HEADS * stride * HEAD)?,
            mixed: f32s(padded * DIM)?,
            out: f32s(padded * CHANNELS)?,
            readback: gpu.readback((padded * CHANNELS) as u64 * 4)?,
        })
    }
}

/// The prompt's (and reference images') keys and values for every block, for one request.
pub struct Prefix {
    /// Unique per prefix, so a pass knows whether its turns are this prefix's.
    id: u64,
    /// Per block: keys and values `[head][capacity][128]` f16, in device memory or, when
    /// it is short, in host memory (each step then copies them over the bus).
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    pub length: usize,
    /// The target's latent size and its tokens' rotary turns.
    height: usize,
    width: usize,
    turns: Vec<f32>,
}

pub struct QwenVk {
    gpu: Gpu,
    kernels: Kernels,
    layers: Vec<Layer>,
    small: Small,
    img_in: Buffer,
    proj_out: Buffer,
    /// Pass buffers by own tokens; a request's target passes (CFG's both prompts) reuse one
    /// every step.
    work: Mutex<HashMap<usize, Work>>,
    /// Where prefixes go: `None` by the device's free memory, else in host memory or not.
    #[cfg(test)]
    host_prefix: Option<bool>,
}
impl QwenVk {
    pub fn load(gpu: Gpu, path: &Path) -> Result<Self, &'static str> {
        let model = QwenDit::open(path)?;
        let kernels = Kernels::new(&gpu)?;
        let layers = (0..model.layers)
            .map(|layer| Layer::load(&gpu, &model.file, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let small = Small::read(&model.file)?;
        let img_in = floats(&gpu, &model.file.floats("img_in.weight", &[DIM, CHANNELS])?)?;
        let proj_out = floats(&gpu, &model.file.floats("proj_out.weight", &[CHANNELS, DIM])?)?;
        Ok(Self {
            gpu,
            kernels,
            layers,
            small,
            img_in,
            proj_out,
            work: Mutex::new(HashMap::new()),
            #[cfg(test)]
            host_prefix: None,
        })
    }

    /// As `QwenGpu::prefix`. The keys and values stay in device memory when the steps'
    /// buffers still fit beside them, else in host memory.
    pub fn prefix(
        &self,
        context: &[f32],
        slots: &[i32],
        references: &[(&[f32], usize, usize)],
        target: (usize, usize),
    ) -> Result<Prefix, &'static str> {
        let gpu = &self.gpu;
        let text_len = context.len() / CONTEXT;
        let mut images: Vec<(usize, usize)> = references.iter().map(|r| (r.1, r.2)).collect();
        images.push(target);
        let layout = Layout::new(text_len, slots, &images)?;
        let rotary = qwen_dit::rotary(&layout.positions);
        let length = layout.prefix;
        let rows_bytes = (HEADS * length.max(1) * HEAD) as u64 * 2;
        let need = 2 * self.layers.len() as u64 * rows_bytes
            + Work::bytes(target.0 * target.1, length);
        let host = gpu.free_memory().is_some_and(|free| free < need);
        #[cfg(test)]
        let host = self.host_prefix.unwrap_or(host);
        let half_rows = || {
            if host {
                gpu.host_buffer(rows_bytes)
            } else {
                gpu.buffer(rows_bytes)
            }
        };
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for _ in 0..self.layers.len() {
            keys.push(half_rows()?);
            values.push(half_rows()?);
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let prefix = Prefix {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            keys,
            values,
            length,
            height: target.0,
            width: target.1,
            turns: rotary[length * PAIRS..].iter().flatten().copied().collect(),
        };
        let text = self.small.text(context);
        let mods = self.small.times(&[0.0]).remove(0).mods;
        for segment in layout.segments.iter().filter(|s| s.start < length) {
            let tokens = segment.end - segment.start;
            let mut works = self.work.lock().map_err(|_| "gpu_failed")?;
            let work = Self::work_for(&mut works, gpu, tokens, segment.start)?;
            let mut turns: Vec<f32> = rotary[segment.start * PAIRS..segment.end * PAIRS]
                .iter()
                .flatten()
                .copied()
                .collect();
            turns.resize(work.padded * PAIRS * 2, 0.0);
            gpu.upload(&work.turns, 0, bytes_of(&turns))?;
            work.turns_of.set(0);
            gpu.upload(&work.mods, 0, bytes_of(&mods))?;
            let causal = match segment.image {
                None => {
                    let rows = &text[segment.context_start * DIM..][..tokens * DIM];
                    gpu.upload(&work.stream, 0, bytes_of(rows))?;
                    true
                }
                Some(index) => {
                    let (latent, h, w) = references[index];
                    gpu.upload(&work.rows, 0, bytes_of(&qwen_dit::patchify(latent, h, w)))?;
                    run(gpu, &[self.embed(work)])?;
                    false
                }
            };
            let mut calls = Vec::new();
            for index in 0..self.layers.len() {
                let last = index + 1 == self.layers.len();
                let store = Some((&prefix.keys[index], &prefix.values[index], length));
                calls.extend(self.block(work, 1, causal, store, &prefix, index, last));
            }
            run(gpu, &calls)?;
            // Only this prefix uses these buffers.
            works.remove(&tokens);
        }
        Ok(prefix)
    }

    /// Frees the pass buffers (made again when next needed), for the VAE to have the
    /// memory on small cards.
    pub fn release_work(&self) {
        if let Ok(mut works) = self.work.lock() {
            works.clear();
        }
    }

    /// The pass buffers for `tokens` own tokens after `earlier` keys, made (or made again
    /// with more key rows) when needed.
    fn work_for<'a>(
        works: &'a mut HashMap<usize, Work>,
        gpu: &Gpu,
        tokens: usize,
        earlier: usize,
    ) -> Result<&'a Work, &'static str> {
        let need = (tokens + earlier).next_multiple_of(KEY_PAD);
        if works.get(&tokens).is_some_and(|w| w.stride < need) {
            // Freed before the larger one is made.
            works.remove(&tokens);
        }
        let work = match works.entry(tokens) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Work::new(gpu, tokens, earlier)?)
            }
        };
        work.earlier.set(earlier);
        Ok(work)
    }

    /// Latent rows (`work.rows`) into the stream.
    fn embed<'a>(&'a self, work: &'a Work) -> Call<'a> {
        Call::Shader {
            kernel: &self.kernels.embed,
            buffers: vec![&work.rows, &self.img_in, &work.stream],
            constants: vec![work.tokens as u32],
            groups: [work.tokens as u32, 1, 1],
        }
    }

    /// The time-dependent vectors of each of `timesteps`, for `step_at`.
    pub fn times(&self, timesteps: &[f32]) -> Vec<Times> {
        self.small.times(timesteps)
    }

    /// As `QwenGpu::step`.
    pub fn step(
        &self,
        prefix: &Prefix,
        x: &[f32],
        timestep: f32,
    ) -> Result<Vec<f32>, &'static str> {
        self.step_at(prefix, x, &self.small.times(&[timestep])[0])
    }

    /// As `QwenGpu::step_at`: the whole step is one submission.
    pub fn step_at(
        &self,
        prefix: &Prefix,
        x: &[f32],
        times: &Times,
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let (height, width) = (prefix.height, prefix.width);
        let tokens = height * width;
        let mut works = self.work.lock().map_err(|_| "gpu_failed")?;
        let w = Self::work_for(&mut works, gpu, tokens, prefix.length)?;
        if w.turns_of.get() != prefix.id {
            let mut turns = prefix.turns.clone();
            turns.resize(w.padded * PAIRS * 2, 0.0);
            gpu.upload(&w.turns, 0, bytes_of(&turns))?;
            w.turns_of.set(prefix.id);
        }
        gpu.upload(&w.mods, 0, bytes_of(&times.mods))?;
        gpu.upload(&w.final_scale, 0, bytes_of(&times.scale))?;
        gpu.upload(&w.rows, 0, bytes_of(&qwen_dit::patchify(x, height, width)))?;
        let mut calls = vec![self.embed(w)];
        for index in 0..self.layers.len() {
            calls.extend(self.block(w, 0, false, None, prefix, index, false));
        }
        calls.push(Call::Shader {
            kernel: &self.kernels.final_,
            buffers: vec![&w.stream, &w.final_scale, &self.proj_out, &w.out],
            constants: vec![tokens as u32],
            groups: [tokens as u32, 1, 1],
        });
        run(gpu, &calls)?;
        let mut result = vec![0f32; tokens * CHANNELS];
        gpu.read_floats(&w.out, 0, &w.readback, &mut result)?;
        Ok(qwen_dit::unpatchify(&result, height, width))
    }

    /// The calls of one block on `work.stream` with modulation row `row` (0 the step's, 1
    /// time zero). With `store`, the pass's keys and values are kept in the prefix cache at
    /// rows `work.earlier..`; with `last`, only they are needed, so the block stops after
    /// them.
    #[allow(clippy::too_many_arguments)]
    fn block<'a>(
        &'a self,
        w: &'a Work,
        row: usize,
        causal: bool,
        store: Option<(&'a Buffer, &'a Buffer, usize)>,
        prefix: &'a Prefix,
        index: usize,
        last: bool,
    ) -> Vec<Call<'a>> {
        let k = &self.kernels;
        let layer = &self.layers[index];
        let (n, np) = (w.tokens as u32, w.padded as u32);
        let v = &layer.vectors;
        let mods = |part: usize| ((row * MODS) + part * DIM) as u32;
        let product = |which: usize| {
            let (rows, columns) = PRODUCTS[which];
            Call::Shader {
                kernel: &k.gemm8,
                buffers: vec![&w.bytes, &layer.matrices[which], &w.sums],
                constants: vec![columns as u32, 0, 0, 0, rows as u32],
                groups: [(rows / 256) as u32, np / PAD as u32, 1],
            }
        };
        let norm = |part: usize| Call::Shader {
            kernel: &k.normq8,
            buffers: vec![&w.stream, &w.mods, &w.bytes, &w.scales],
            constants: vec![n, mods(part), 0],
            groups: [np, 1, 1],
        };
        // The residual with gate `part`, then the norm with modulation `next`.
        let residual_norm = |scale: usize, part: usize, next: usize| Call::Shader {
            kernel: &k.resnorm8,
            buffers: vec![&w.stream, &w.mods, &w.bytes, &w.scales, &w.sums, v],
            constants: vec![n, mods(next), 0, scale as u32, mods(part)],
            groups: [np, 1, 1],
        };
        let (cache_keys, cache_values, capacity) = store.unwrap_or((&w.key, &w.value, 0));
        let mut calls = Vec::new();
        if index == 0 {
            calls.push(norm(0));
        }
        calls.push(product(QKV));
        calls.push(Call::Shader {
            kernel: &k.qkv,
            buffers: vec![
                &w.sums,
                &w.scales,
                v,
                &w.turns,
                &w.query,
                &w.key,
                &w.value,
                cache_keys,
                cache_values,
            ],
            constants: vec![
                n,
                np,
                w.stride as u32,
                QUERY_NORM as u32,
                KEY_NORM as u32,
                SCALE_QKV as u32,
                u32::from(store.is_some()),
                w.earlier.get() as u32,
                capacity as u32,
            ],
            groups: [(3 * HEADS / 4) as u32, np, 1],
        });
        if last {
            return calls;
        }
        if w.earlier.get() > 0 {
            calls.push(Call::Shader {
                kernel: &k.copy,
                buffers: vec![&prefix.keys[index], &prefix.values[index], &w.key, &w.value],
                constants: vec![
                    w.earlier.get() as u32,
                    n,
                    w.stride as u32,
                    prefix.length.max(1) as u32,
                ],
                groups: [(w.earlier.get() * 16).div_ceil(64) as u32, HEADS as u32, 1],
            });
        }
        calls.push(Call::Shader {
            kernel: &k.attend,
            buffers: vec![&w.query, &w.key, &w.value, &w.mixed],
            constants: vec![
                n,
                (w.tokens + w.earlier.get()) as u32,
                np,
                w.stride as u32,
                HEADS as u32,
                (1.0 / (HEAD as f32).sqrt()).to_bits(),
                u32::from(causal),
            ],
            groups: [np / k.attend_block, HEADS as u32, 1],
        });
        calls.push(Call::Shader {
            kernel: &k.rotate8,
            buffers: vec![&w.mixed, &w.bytes, &w.scales],
            constants: vec![DIM as u32, (DIM / 256).div_ceil(16) as u32, 0, 0],
            groups: [np, 1, 1],
        });
        calls.push(product(OUT));
        calls.push(residual_norm(SCALE_OUT, 1, 2));
        calls.push(product(FC1));
        calls.push(Call::Shader {
            kernel: &k.gate,
            buffers: vec![&w.sums, &w.scales, v, &w.hidden],
            constants: vec![n, SCALE_FC1 as u32],
            groups: [(HIDDEN / 4 / 64) as u32, np, 1],
        });
        calls.push(Call::Shader {
            kernel: &k.hidq8,
            buffers: vec![&w.hidden, &w.scales, &w.spill, &w.bytes],
            constants: vec![n, 0],
            groups: [np, 1, 1],
        });
        calls.push(product(FC2));
        calls.push(if index + 1 < self.layers.len() {
            residual_norm(SCALE_FC2, 3, 0)
        } else {
            Call::Shader {
                kernel: &k.resid8,
                buffers: vec![&w.stream, &w.sums, &w.scales, v, &w.mods],
                constants: vec![n, SCALE_FC2 as u32, mods(3)],
                groups: [(DIM / 4 / 64) as u32, np, 1],
            }
        });
        calls
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            gpu::Select,
            qwen_dit::tests::{Golden, db},
        },
        *,
    };

    /// `KUNO_QWEN_DIT=<model> [KUNO_GPU_VENDOR=<hex>] [KUNO_PREFIX_HOST=1]`: the time of each
    /// kind of kernel in a step of a 1248 x 832 edit with a reference of the same size (4056
    /// own tokens after 4356 prefix keys), each call run and waited for alone.
    #[test]
    #[ignore = "needs a GPU and the model"]
    fn block_costs_of_a_large_edit() {
        let vendor = std::env::var("KUNO_GPU_VENDOR")
            .map_or(0x10de, |v| u32::from_str_radix(&v, 16).unwrap());
        let gpu = Gpu::open(Select::Vendor(vendor)).unwrap();
        let model =
            QwenVk::load(gpu, Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
        let (height, width, length) = (52, 78, 4356);
        let host = std::env::var("KUNO_PREFIX_HOST").is_ok_and(|v| v == "1");
        let rows = (HEADS * length * HEAD) as u64 * 2;
        let half = || {
            if host {
                model.gpu.host_buffer(rows).unwrap()
            } else {
                model.gpu.buffer(rows).unwrap()
            }
        };
        let prefix = Prefix {
            id: 1,
            keys: (0..model.layers.len()).map(|_| half()).collect(),
            values: (0..model.layers.len()).map(|_| half()).collect(),
            length,
            height,
            width,
            turns: vec![0.0; height * width * PAIRS * 2],
        };
        let x = vec![0.1f32; CHANNELS * height * width];
        let times = model.times(&[500.0]).remove(0);
        for _ in 0..2 {
            let started = std::time::Instant::now();
            model.step_at(&prefix, &x, &times).unwrap();
            println!("step {:.3} s", started.elapsed().as_secs_f64());
        }
        let works = model.work.lock().unwrap();
        let w = &works[&(height * width)];
        let calls = model.block(w, 0, false, None, &prefix, 3, false);
        let names = [
            "qkv product", "qkv16", "prefix copy", "attention", "quantize", "out product",
            "residual+norm", "fc1 product", "gate", "hidden quantize", "fc2 product",
            "residual+norm",
        ];
        let mut total = 0.0;
        for (call, name) in calls.iter().zip(names) {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                run(&model.gpu, std::slice::from_ref(call)).unwrap();
                best = best.min(started.elapsed().as_secs_f64() * 1e3);
            }
            total += best;
            println!("{name}: {best:.2} ms");
        }
        println!("block {total:.1} ms, x {} blocks = {:.2} s", model.layers.len(), total * model.layers.len() as f64 / 1e3);
    }

    /// `KUNO_QWEN_DIT=<model> KUNO_QWEN_GOLDEN=<dir> [KUNO_GPU_VENDOR=<hex>]`: steps against
    /// the captured C++ engine's, as `qwen_gpu::tests::steps_match_cpp_engine`.
    #[test]
    #[ignore = "needs a GPU, the model and captured steps"]
    fn steps_match_cpp_engine() {
        let vendor = std::env::var("KUNO_GPU_VENDOR")
            .map_or(0x10de, |v| u32::from_str_radix(&v, 16).unwrap());
        let gpu = Gpu::open(Select::Vendor(vendor)).unwrap();
        println!("{}", gpu.name());
        let started = std::time::Instant::now();
        let mut model =
            QwenVk::load(gpu, Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
        // KUNO_PREFIX_HOST=1: prefixes in host memory, read over the bus.
        model.host_prefix = std::env::var("KUNO_PREFIX_HOST").ok().map(|v| v == "1");
        println!("load {:.1} s", started.elapsed().as_secs_f64());
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_GOLDEN").unwrap());
        for step in 0.. {
            if !dir.join(format!("step{step}.txt")).exists() {
                assert!(step > 0);
                break;
            }
            let golden = Golden::load(&dir, step);
            let references: Vec<(&[f32], usize, usize)> = golden
                .references
                .iter()
                .map(|(r, h, w)| (&r[..], *h, *w))
                .collect();
            let started = std::time::Instant::now();
            let prefix = model
                .prefix(
                    &golden.context,
                    &golden.slots,
                    &references,
                    (golden.height, golden.width),
                )
                .unwrap();
            let prefix_time = started.elapsed().as_secs_f64();
            let mut out = Vec::new();
            let mut step_time = f64::MAX;
            for _ in 0..2 {
                let started = std::time::Instant::now();
                out = model.step(&prefix, &golden.x, golden.timestep).unwrap();
                step_time = step_time.min(started.elapsed().as_secs_f64());
            }
            let db = db(&golden.out, &out);
            println!(
                "step {step}: t {} {}x{} prefix {} tokens {prefix_time:.2} s, step {step_time:.3} s -> {db:.1} dB",
                golden.timestep, golden.height, golden.width, prefix.length
            );
            assert!(db > 25.0, "step {step}: {db:.1} dB");
        }
    }
}
