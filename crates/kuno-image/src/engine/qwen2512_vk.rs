//! Qwen Image 2512's diffusion transformer on any Vulkan GPU with cooperative matrices,
//! including cards smaller than the model (the RTX 3060's 12 GB): `qwen2512_gpu.rs`'s plan
//! with Vulkan kernels. The block linears are INT8 rows for `gemm8` (Kuno's conversion, or
//! rounded at load from the FP8 file), attention is the fused `fattend16`, and the glue is
//! `q25*.comp` with `rotate8` and `qhidq8`. Checked against the same captures.
//!
//! Memory: the modulation layers (a third of the model, one matrix-vector product per step)
//! are evaluated on the CPU from host memory. Blocks stay in device memory while the budget
//! leaves room for a CFG step's buffers; the rest live in host memory and each is copied into
//! one device slot when its turn comes, every CFG pass running through it while it is there.
use std::{collections::HashMap, path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit::{GROUP, quantize, rotate},
    dit_gpu::{Call, run},
    gpu::{Buffer, Copying, Gpu, Kernel, Readback, Staged, bytes_of, shaders},
    qwen_dit::rotary,
    qwen2512_dit::{
        CHANNELS, DIM, HEAD, HEADS, HIDDEN, PAIRS, Qwen2512Dit, patchify, positions, unpatchify,
    },
    qwen2512_gpu::{Prompt, Small, Times, float_matrix},
    safetensors::{Safetensors, fp8_e4m3},
};

/// Weight row tiles (of 256) per product dispatch.
const ROW_TILES: usize = 8;
/// Rows of each stream are padded to the products' token block.
const PAD: usize = 128;
/// Queries and keys are padded to `fattend16`'s blocks.
const ATTEND_PAD: usize = 64;
/// A stream's four products: weight rows and columns.
const PRODUCTS: [(usize, usize); 4] = [(3 * DIM, DIM), (DIM, DIM), (HIDDEN, DIM), (DIM, HIDDEN)];
const QKV: usize = 0;
const OUT: usize = 1;
const FC1: usize = 2;
const FC2: usize = 3;
/// Bytes of one stream's matrices, and of a block's (image, then text).
const STREAM_BYTES: usize = 3 * DIM * DIM + DIM * DIM + 2 * HIDDEN * DIM;
const BLOCK_BYTES: usize = 2 * STREAM_BYTES;
/// Offsets (in values) into a stream's vector buffer.
const QUERY_NORM: usize = 0;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const QKV_BIAS: usize = KEY_NORM + HEAD;
const QKV_SCALE: usize = QKV_BIAS + 3 * DIM;
const OUT_BIAS: usize = QKV_SCALE + 3 * DIM;
const OUT_SCALE: usize = OUT_BIAS + DIM;
const FC1_BIAS: usize = OUT_SCALE + DIM;
const FC1_SCALE: usize = FC1_BIAS + HIDDEN;
const FC2_BIAS: usize = FC1_SCALE + HIDDEN;
const FC2_SCALE: usize = FC2_BIAS + DIM;
const VECTORS: usize = FC2_SCALE + DIM;
/// Modulation rows of one stream in one block (shift, scale, gate, twice), and of a block.
const MOD_ROWS: usize = 6 * DIM;
const BLOCK_MODS: usize = 2 * MOD_ROWS;
const SHIFT1: usize = 0;
const GATE1: usize = 2 * DIM;
const SHIFT2: usize = 3 * DIM;
const GATE2: usize = 5 * DIM;
const TEXT: usize = 1;
/// The feed-forward activation's rotated groups that `qhidq8` spills.
const SPILLED: usize = 20 * 256;
/// Device memory left free when choosing resident blocks: two CFG passes' buffers at about
/// 1328 x 1328, and the text encoder's and VAE's working memory.
const RESERVE: u64 = 3584 << 20;

/// The byte offset of a stream's product in a block's matrices.
fn product_at(stream: usize, which: usize) -> usize {
    let within = PRODUCTS[..which].iter().map(|(r, c)| r * c).sum::<usize>();
    stream * STREAM_BYTES + within
}

struct Kernels {
    gemm8: Kernel,
    norm: Kernel,
    resid: Kernel,
    qkv: Kernel,
    gelu: Kernel,
    hidq8: Kernel,
    rotate8: Kernel,
    attend: Kernel,
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
            norm: gpu.kernel(shaders::Q25NORM, 6, 6, any)?,
            resid: gpu.kernel(shaders::Q25RESID, 5, 4, None)?,
            qkv: gpu.kernel(shaders::Q25QKV, 7, 7, any)?,
            gelu: gpu.kernel(shaders::Q25GELU, 4, 3, None)?,
            hidq8: gpu.kernel(shaders::QHIDQ8, 4, 2, any)?,
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 4, any)?,
            attend: if gpu.matrix2() {
                gpu.kernel(shaders::FATTEND16_NV, 4, 7, None)?
            } else {
                gpu.kernel(shaders::FATTEND16, 4, 7, Some(32))?
            },
            attend_block: if gpu.matrix2() { 64 } else { 32 },
            embed: gpu.kernel(shaders::Q25EMBED, 4, 1, None)?,
            final_: gpu.kernel(shaders::Q25FINAL, 5, 1, any)?,
        })
    }
}

/// Rows `[rows][columns]` of INT8 for `gemm8` into `to`, and one scale per row: the file's
/// INT8 rows (Kuno's conversion), or float rows rounded here in the rotated basis.
fn plain_int8(
    file: &Safetensors,
    tensors: &[String],
    rows: usize,
    columns: usize,
    to: &mut [u8],
) -> Result<Vec<f32>, &'static str> {
    let per_part = rows / tensors.len();
    let mut scales = Vec::with_capacity(rows);
    for (tensor, to) in tensors.iter().zip(to.chunks_exact_mut(per_part * columns)) {
        if file.is_i8(tensor) {
            let base = tensor
                .strip_suffix(".weight")
                .ok_or("model_tensor_invalid")?;
            scales.extend(file.f32(&format!("{base}.weight_scale"), &[per_part, 1])?);
            file.i8_part_into(tensor, &[per_part, columns], 0, to)?;
        } else {
            let (bytes, row_scales) =
                quantize(&float_matrix(file, tensor, per_part, columns)?, columns);
            for (to, b) in to.iter_mut().zip(&bytes) {
                *to = *b as u8;
            }
            scales.extend(row_scales);
        }
    }
    Ok(scales)
}

/// Modulation rows in host memory, evaluated on the CPU: exact FP8 bytes (ComfyUI's file),
/// or INT8 in the rotated basis with a scale per row.
enum Modulation {
    Fp8 {
        weight: Vec<u8>,
        bias: Vec<f32>,
    },
    Int8 {
        weight: Vec<i8>,
        scales: Vec<f32>,
        bias: Vec<f32>,
    },
}
impl Modulation {
    fn load(file: &Safetensors, names: &[String], rows: usize) -> Result<Self, &'static str> {
        let mut bias = Vec::with_capacity(rows * names.len());
        for name in names {
            let base = name.strip_suffix(".weight").ok_or("model_tensor_invalid")?;
            bias.extend(file.floats(&format!("{base}.bias"), &[rows])?);
        }
        if names.iter().all(|n| file.is_fp8(n))
            && names.iter().all(|n| {
                let base = n.strip_suffix(".weight").unwrap_or(n);
                file.shape(&format!("{base}.scale_weight")).is_none()
            })
        {
            let mut weight = Vec::with_capacity(rows * names.len() * DIM);
            for name in names {
                weight.extend(file.fp8_bytes(name, &[rows, DIM])?);
            }
            return Ok(Self::Fp8 { weight, bias });
        }
        let (mut weight, mut scales) = (Vec::with_capacity(rows * names.len() * DIM), Vec::new());
        for name in names {
            if file.is_i8(name) {
                let base = name.strip_suffix(".weight").ok_or("model_tensor_invalid")?;
                weight.extend(file.i8(name, &[rows, DIM])?);
                scales.extend(file.f32(&format!("{base}.weight_scale"), &[rows, 1])?);
            } else {
                let (bytes, row_scales) = quantize(&float_matrix(file, name, rows, DIM)?, DIM);
                weight.extend(bytes);
                scales.extend(row_scales);
            }
        }
        Ok(Self::Int8 {
            weight,
            scales,
            bias,
        })
    }
    /// `out = weight . t + bias` (`rotated`: `t` rotated in groups of 256, for INT8 rows).
    fn apply(&self, t: &[f32], rotated: &[f32], out: &mut [f32]) {
        match self {
            Self::Fp8 { weight, bias } => {
                let table: Vec<f32> = (0..=255u8).map(fp8_e4m3).collect();
                out.par_iter_mut()
                    .zip(weight.par_chunks_exact(DIM))
                    .zip(bias.par_iter())
                    .for_each(|((o, row), b)| {
                        let sum: f32 = row
                            .iter()
                            .zip(t)
                            .map(|(w, t)| table[usize::from(*w)] * t)
                            .sum();
                        *o = sum + b;
                    });
            }
            Self::Int8 {
                weight,
                scales,
                bias,
            } => {
                out.par_iter_mut()
                    .zip(weight.par_chunks_exact(DIM))
                    .zip(scales.par_iter().zip(bias.par_iter()))
                    .for_each(|((o, row), (s, b))| {
                        let sum: f32 = row
                            .chunks_exact(16)
                            .zip(rotated.chunks_exact(16))
                            .map(|(w, t)| {
                                let mut acc = 0f32;
                                for i in 0..16 {
                                    acc += f32::from(w[i]) * t[i];
                                }
                                acc
                            })
                            .sum();
                        *o = sum * s + b;
                    });
            }
        }
    }
}

/// One block's matrices: in device memory, or in host memory to be copied in when used.
enum Matrices {
    Device(Buffer),
    Host(Staged),
}
/// One block: its matrices (`product_at` layout), each stream's vectors, and the modulation
/// rows of both streams (`[stream][6 * DIM][DIM]`).
struct Layer {
    matrices: Matrices,
    vectors: [Buffer; 2],
    modulation: Modulation,
}
impl Layer {
    fn load(
        gpu: &Gpu,
        model: &Qwen2512Dit,
        layer: usize,
        resident: bool,
    ) -> Result<Self, &'static str> {
        let file = &model.file;
        let name = |part: &str| model.name(&format!("transformer_blocks.{layer}.{part}"));
        let streams: [([&str; 3], &str, &str, [&str; 2]); 2] = [
            (
                ["to_q", "to_k", "to_v"],
                "to_out.0",
                "img_mlp",
                ["norm_q", "norm_k"],
            ),
            (
                ["add_q_proj", "add_k_proj", "add_v_proj"],
                "to_add_out",
                "txt_mlp",
                ["norm_added_q", "norm_added_k"],
            ),
        ];
        let mut all_scales = Vec::new();
        let staged = gpu.staged(BLOCK_BYTES as u64, |to| {
            for (s, (qkv, out, mlp, _)) in streams.iter().enumerate() {
                let a = |part: &str| name(&format!("attn.{part}.weight"));
                let tensors: [Vec<String>; 4] = [
                    qkv.map(a).to_vec(),
                    vec![a(out)],
                    vec![name(&format!("{mlp}.net.0.proj.weight"))],
                    vec![name(&format!("{mlp}.net.2.weight"))],
                ];
                let mut scales = Vec::with_capacity(4);
                for (which, tensors) in tensors.iter().enumerate() {
                    let (rows, columns) = PRODUCTS[which];
                    let at = product_at(s, which);
                    scales.push(plain_int8(
                        file,
                        tensors,
                        rows,
                        columns,
                        &mut to[at..at + rows * columns],
                    )?);
                }
                all_scales.push(scales);
            }
            Ok(())
        })?;
        let matrices = if resident {
            Matrices::Device(gpu.place(&staged, false)?)
        } else {
            Matrices::Host(staged)
        };
        let mut vectors = Vec::with_capacity(2);
        for ((qkv, out, mlp, norms), scales) in streams.iter().zip(&all_scales) {
            let mut v = Vec::with_capacity(VECTORS);
            for norm in norms {
                v.extend(file.floats(&name(&format!("attn.{norm}.weight")), &[HEAD])?);
            }
            for part in qkv {
                v.extend(file.floats(&name(&format!("attn.{part}.bias")), &[DIM])?);
            }
            v.extend(&scales[QKV]);
            v.extend(file.floats(&name(&format!("attn.{out}.bias")), &[DIM])?);
            v.extend(&scales[OUT]);
            v.extend(file.floats(&name(&format!("{mlp}.net.0.proj.bias")), &[HIDDEN])?);
            v.extend(&scales[FC1]);
            v.extend(file.floats(&name(&format!("{mlp}.net.2.bias")), &[DIM])?);
            v.extend(&scales[FC2]);
            debug_assert_eq!(v.len(), VECTORS);
            vectors.push(floats(gpu, &v)?);
        }
        let [image, text]: [Buffer; 2] = vectors.try_into().map_err(|_| "gpu_failed")?;
        Ok(Self {
            matrices,
            vectors: [image, text],
            modulation: Modulation::load(
                file,
                &[name("img_mod.1.weight"), name("txt_mod.1.weight")],
                MOD_ROWS,
            )?,
        })
    }
}

fn floats(gpu: &Gpu, values: &[f32]) -> Result<Buffer, &'static str> {
    gpu.filled(values.len() as u64 * 4, |to| {
        to.copy_from_slice(bytes_of(values));
        Ok(())
    })
}

/// Buffers of one stream in a pass shape.
struct StreamWork {
    tokens: usize,
    padded: usize,
    /// Query slot and key index of the stream's first token.
    first: usize,
    x: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    hidden: Buffer,
    spill: Buffer,
    turns: Buffer,
}

/// Buffers of one pass shape: an image of `height x width` tokens and a prompt of `text`.
struct Work {
    streams: [StreamWork; 2],
    /// Queries, keys and values `[head][padded][128]` f16 (image rows, then text), the
    /// attention output `[padded][DIM]` f32.
    query: Buffer,
    key: Buffer,
    value: Buffer,
    mixed: Buffer,
    padded: usize,
    rows: Buffer,
    out: Buffer,
    readback: Readback,
}
impl Work {
    fn new(gpu: &Gpu, height: usize, width: usize, text: usize) -> Result<Self, &'static str> {
        let image = height * width;
        let rotation = rotary(&positions(text, height, width));
        let turns = |range: std::ops::Range<usize>| -> Vec<f32> {
            rotation[range.start * PAIRS..range.end * PAIRS]
                .iter()
                .flat_map(|[c, s]| [*c, *s])
                .collect()
        };
        let stream = |tokens: usize, first: usize, turns: Vec<f32>| {
            let padded = tokens.next_multiple_of(PAD);
            Ok::<_, &'static str>(StreamWork {
                tokens,
                padded,
                first,
                x: gpu.buffer((padded * DIM * 4) as u64)?,
                bytes: gpu.buffer((padded * HIDDEN) as u64)?,
                scales: gpu.buffer((padded * 4) as u64)?,
                sums: gpu.buffer((padded * HIDDEN * 4) as u64)?,
                hidden: gpu.buffer((padded * HIDDEN * 2) as u64)?,
                spill: gpu.buffer((padded * SPILLED * 4) as u64)?,
                turns: floats(gpu, &turns)?,
            })
        };
        let padded = (image + text).next_multiple_of(ATTEND_PAD);
        let halves = (HEADS * padded * HEAD * 2) as u64;
        Ok(Self {
            streams: [
                stream(image, 0, turns(text..text + image))?,
                stream(text, image, turns(0..text))?,
            ],
            query: gpu.buffer(halves)?,
            key: gpu.buffer(halves)?,
            value: gpu.buffer(halves)?,
            mixed: gpu.buffer((padded * DIM * 4) as u64)?,
            padded,
            rows: gpu.buffer((image.next_multiple_of(PAD) * CHANNELS * 4) as u64)?,
            out: gpu.buffer((image * CHANNELS * 4) as u64)?,
            readback: gpu.readback((image * CHANNELS * 4) as u64)?,
        })
    }
}

/// A step's modulation rows arriving from the CPU thread, uploaded in order: blocks
/// `0..blocks`, then the head's.
struct Mods<'a> {
    receive: std::sync::mpsc::Receiver<Vec<f32>>,
    uploaded: usize,
    blocks: usize,
    gpu: &'a Gpu,
    mods: &'a Buffer,
    final_mods: &'a Buffer,
}
impl Mods<'_> {
    /// Waits until block `upto`'s rows (`blocks`: the head's) are on the device.
    fn ready(&mut self, upto: usize) -> Result<(), &'static str> {
        while self.uploaded <= upto.min(self.blocks) {
            let rows = self.receive.recv().map_err(|_| "gpu_failed")?;
            if self.uploaded == self.blocks {
                self.gpu.upload(self.final_mods, 0, bytes_of(&rows))?;
            } else {
                let at = (self.uploaded * BLOCK_MODS * 4) as u64;
                self.gpu.upload(self.mods, at, bytes_of(&rows))?;
            }
            self.uploaded += 1;
        }
        Ok(())
    }
}

pub struct Qwen2512Vk {
    gpu: Gpu,
    kernels: Kernels,
    layers: Vec<Layer>,
    small: Small,
    img_in: (Buffer, Buffer),
    proj_out: (Buffer, Buffer),
    final_modulation: Modulation,
    /// The two device slots host-memory blocks are copied into: the next one's copy runs
    /// while a block computes.
    slots: Vec<Buffer>,
    /// This step's modulation rows `[block][stream][6 * DIM]`, and the head's (scale, shift).
    mods: Buffer,
    final_mods: Buffer,
    work: Mutex<HashMap<(usize, usize, usize), Work>>,
}
impl Qwen2512Vk {
    pub fn load(gpu: Gpu, path: &Path) -> Result<Self, &'static str> {
        let model = Qwen2512Dit::open(path)?;
        let kernels = Kernels::new(&gpu)?;
        // Blocks are resident while the budget leaves room for two block slots and the
        // reserve.
        let free = gpu.free_memory().unwrap_or(u64::MAX);
        let spare = free.saturating_sub(RESERVE + 2 * BLOCK_BYTES as u64);
        let resident = (spare / BLOCK_BYTES as u64) as usize;
        let layers = (0..model.layers)
            .map(|layer| Layer::load(&gpu, &model, layer, layer < resident))
            .collect::<Result<Vec<_>, _>>()?;
        let slots = if resident < model.layers {
            vec![
                gpu.buffer(BLOCK_BYTES as u64)?,
                gpu.buffer(BLOCK_BYTES as u64)?,
            ]
        } else {
            Vec::new()
        };
        let small = Small::read(&model)?;
        let f = |name: &str, shape: &[usize]| model.floats(name, shape);
        let pair = |w: &str, ws: &[usize], b: &str, bs: &[usize]| -> Result<_, &'static str> {
            Ok((floats(&gpu, &f(w, ws)?)?, floats(&gpu, &f(b, bs)?)?))
        };
        let img_in = pair("img_in.weight", &[DIM, CHANNELS], "img_in.bias", &[DIM])?;
        let proj_out = pair(
            "proj_out.weight",
            &[CHANNELS, DIM],
            "proj_out.bias",
            &[CHANNELS],
        )?;
        let final_modulation = Modulation::load(
            &model.file,
            &[model.name("norm_out.linear.weight")],
            2 * DIM,
        )?;
        let mods = gpu.buffer((layers.len() * BLOCK_MODS * 4) as u64)?;
        let final_mods = gpu.buffer((2 * DIM * 4) as u64)?;
        Ok(Self {
            gpu,
            kernels,
            layers,
            small,
            img_in,
            proj_out,
            final_modulation,
            slots,
            mods,
            final_mods,
            work: Mutex::new(HashMap::new()),
        })
    }

    /// How many blocks are in device memory, of how many.
    pub fn resident(&self) -> (usize, usize) {
        let device = self
            .layers
            .iter()
            .filter(|l| matches!(l.matrices, Matrices::Device(_)))
            .count();
        (device, self.layers.len())
    }

    /// As `Qwen2512Gpu::times`.
    pub fn times(&self, timesteps: &[f32]) -> Vec<Times> {
        self.small.times(timesteps)
    }
    /// As `Qwen2512Gpu::prompt`.
    pub fn prompt(&self, context: &[f32]) -> Result<Prompt, &'static str> {
        self.small.prompt(context)
    }
    /// Frees the pass buffers (made again when next needed).
    pub fn release_work(&self) {
        if let Ok(mut works) = self.work.lock() {
            works.clear();
        }
    }

    /// As `Qwen2512Gpu::step` with standard attention: the model's output for the latent `x`
    /// under each prompt (a CFG pass each). Each block runs for every pass in turn.
    pub fn step(
        &self,
        prompts: &[&Prompt],
        x: &[f32],
        height: usize,
        width: usize,
        times: &Times,
    ) -> Result<Vec<Vec<f32>>, &'static str> {
        let gpu = &self.gpu;
        let mut works = self.work.lock().map_err(|_| "gpu_failed")?;
        for prompt in prompts {
            let key = (height, width, prompt.tokens);
            if let std::collections::hash_map::Entry::Vacant(entry) = works.entry(key) {
                entry.insert(Work::new(gpu, height, width, prompt.tokens)?);
            }
        }
        // This step's modulation rows on the CPU, shared by the passes.
        let t = &times.silu;
        let mut rotated = t.clone();
        for group in rotated.chunks_exact_mut(GROUP) {
            rotate(group.try_into().map_err(|_| "gpu_failed")?);
        }
        // This step's modulation rows on the CPU, shared by the passes, block by block on
        // another thread while the GPU runs the earlier blocks (the head's last).
        std::thread::scope(|scope| -> Result<Vec<Vec<f32>>, &'static str> {
            let (send, receive) = std::sync::mpsc::channel::<Vec<f32>>();
            let layers = &self.layers;
            let final_modulation = &self.final_modulation;
            let rotated = &rotated;
            scope.spawn(move || {
                for layer in layers {
                    let mut out = vec![0f32; BLOCK_MODS];
                    layer.modulation.apply(t, rotated, &mut out);
                    if send.send(out).is_err() {
                        return;
                    }
                }
                let mut out = vec![0f32; 2 * DIM];
                final_modulation.apply(t, rotated, &mut out);
                let _ = send.send(out);
            });
            let mut mods = Mods {
                receive,
                uploaded: 0,
                blocks: layers.len(),
                gpu,
                mods: &self.mods,
                final_mods: &self.final_mods,
            };
            let rows = patchify(x, height, width);
            let passes: Vec<&Work> = prompts
                .iter()
                .map(|p| &works[&(height, width, p.tokens)])
                .collect();
            // Host-memory blocks in order, each copied into the slot after the previous one's.
            let streamed: Vec<(usize, &Staged)> = self
                .layers
                .iter()
                .enumerate()
                .filter_map(|(i, l)| match &l.matrices {
                    Matrices::Host(staged) => Some((i, staged)),
                    Matrices::Device(_) => None,
                })
                .collect();
            let begin = |n: usize| -> Result<Option<Copying>, &'static str> {
                match streamed.get(n) {
                    Some((_, staged)) => {
                        let slot = self.slots.get(n % 2).ok_or("gpu_failed")?;
                        gpu.copy_staged_begin(staged, slot, 0).map(Some)
                    }
                    None => Ok(None),
                }
            };
            let mut copying = begin(0)?;
            let mut next = 0;
            let mut calls = Vec::new();
            for (w, prompt) in passes.iter().zip(prompts) {
                gpu.upload(&w.rows, 0, bytes_of(&rows))?;
                gpu.upload(&w.streams[TEXT].x, 0, bytes_of(&prompt.rows))?;
                self.start(&mut calls, w);
            }
            mods.ready(0)?;
            run(gpu, &calls)?;
            for (index, layer) in self.layers.iter().enumerate() {
                // A block's end norms with the next block's modulation.
                mods.ready(index + 1)?;
                let matrices = match &layer.matrices {
                    Matrices::Device(buffer) => buffer,
                    Matrices::Host(_) => {
                        if let Some(done) = copying.take() {
                            done.wait()?;
                        }
                        let slot = self.slots.get(next % 2).ok_or("gpu_failed")?;
                        // The other slot's last block has finished computing: fill it next.
                        next += 1;
                        copying = begin(next)?;
                        slot
                    }
                };
                let mut calls = Vec::new();
                for w in &passes {
                    self.block(&mut calls, w, index, matrices);
                }
                run(gpu, &calls)?;
            }
            mods.ready(self.layers.len())?;
            let mut calls = Vec::new();
            for w in &passes {
                let image = &w.streams[0];
                calls.push(Call::Shader {
                    kernel: &self.kernels.final_,
                    buffers: vec![
                        &image.x,
                        &self.final_mods,
                        &self.proj_out.0,
                        &self.proj_out.1,
                        &w.out,
                    ],
                    constants: vec![image.tokens as u32],
                    groups: [image.tokens as u32, 1, 1],
                });
            }
            run(gpu, &calls)?;
            let mut outputs = Vec::with_capacity(passes.len());
            for w in &passes {
                let mut result = vec![0f32; height * width * CHANNELS];
                gpu.read_floats(&w.out, 0, &w.readback, &mut result)?;
                outputs.push(unpatchify(&result, height, width));
            }
            Ok(outputs)
        })
    }

    /// The modulation offset of a stream in a block.
    fn base(layer: usize, stream: usize) -> usize {
        layer * BLOCK_MODS + stream * MOD_ROWS
    }

    /// A pass's start: the image rows embedded, then both streams normed for block 0.
    fn start<'a>(&'a self, calls: &mut Vec<Call<'a>>, w: &'a Work) {
        let k = &self.kernels;
        let image = &w.streams[0];
        calls.push(Call::Shader {
            kernel: &k.embed,
            buffers: vec![&w.rows, &self.img_in.0, &self.img_in.1, &image.x],
            constants: vec![image.tokens as u32],
            groups: [image.padded as u32, 1, 1],
        });
        for (s, work) in w.streams.iter().enumerate() {
            calls.push(Call::Shader {
                kernel: &k.norm,
                buffers: vec![
                    &work.x,
                    &self.mods,
                    &work.bytes,
                    &work.scales,
                    &work.sums,
                    &self.layers[0].vectors[s],
                ],
                constants: vec![
                    work.tokens as u32,
                    (Self::base(0, s) + SHIFT1) as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                groups: [work.padded as u32, 1, 1],
            });
        }
    }

    /// One block of one pass, its matrices in `matrices`.
    fn block<'a>(
        &'a self,
        calls: &mut Vec<Call<'a>>,
        w: &'a Work,
        index: usize,
        matrices: &'a Buffer,
    ) {
        let k = &self.kernels;
        let layer = &self.layers[index];
        let last = index + 1 == self.layers.len();
        // Weight rows in dispatches of ROW_TILES tiles (measured faster on the RTX 3060).
        let product = |work: &'a StreamWork, s: usize, which: usize| {
            let (rows, columns) = PRODUCTS[which];
            (0..rows / 256)
                .step_by(ROW_TILES)
                .map(move |first| Call::Shader {
                    kernel: &k.gemm8,
                    buffers: vec![&work.bytes, matrices, &work.sums],
                    constants: vec![
                        columns as u32,
                        0,
                        (product_at(s, which) + first * 256 * columns) as u32,
                        (first * 256) as u32,
                        rows as u32,
                    ],
                    groups: [
                        ROW_TILES.min(rows / 256 - first) as u32,
                        (work.padded / PAD) as u32,
                        1,
                    ],
                })
                .collect::<Vec<_>>()
        };
        for (s, work) in w.streams.iter().enumerate() {
            calls.extend(product(work, s, QKV));
            calls.push(Call::Shader {
                kernel: &k.qkv,
                buffers: vec![
                    &work.sums,
                    &work.scales,
                    &layer.vectors[s],
                    &work.turns,
                    &w.query,
                    &w.key,
                    &w.value,
                ],
                constants: vec![
                    work.first as u32,
                    w.padded as u32,
                    w.padded as u32,
                    QUERY_NORM as u32,
                    KEY_NORM as u32,
                    QKV_SCALE as u32,
                    QKV_BIAS as u32,
                ],
                groups: [(3 * HEADS / 4) as u32, work.tokens as u32, 1],
            });
        }
        let total = (w.streams[0].tokens + w.streams[1].tokens) as u32;
        calls.push(Call::Shader {
            kernel: &k.attend,
            buffers: vec![&w.query, &w.key, &w.value, &w.mixed],
            constants: vec![
                total,
                total,
                w.padded as u32,
                w.padded as u32,
                HEADS as u32,
                (1.0 / (HEAD as f32).sqrt()).to_bits(),
                0,
            ],
            groups: [w.padded as u32 / k.attend_block, HEADS as u32, 1],
        });
        for (s, work) in w.streams.iter().enumerate() {
            // After the last block only the image stream is read.
            if last && s == TEXT {
                continue;
            }
            let v = &layer.vectors[s];
            let b = Self::base(index, s);
            let norm = |add_scale: usize, add_bias: usize, gate: usize, next: usize| Call::Shader {
                kernel: &k.norm,
                buffers: vec![
                    &work.x,
                    &self.mods,
                    &work.bytes,
                    &work.scales,
                    &work.sums,
                    v,
                ],
                constants: vec![
                    work.tokens as u32,
                    next as u32,
                    1,
                    add_scale as u32,
                    add_bias as u32,
                    gate as u32,
                ],
                groups: [work.padded as u32, 1, 1],
            };
            calls.push(Call::Shader {
                kernel: &k.rotate8,
                buffers: vec![&w.mixed, &work.bytes, &work.scales],
                constants: vec![
                    DIM as u32,
                    (DIM / 256).div_ceil(16) as u32,
                    0,
                    work.first as u32,
                ],
                groups: [work.tokens as u32, 1, 1],
            });
            calls.extend(product(work, s, OUT));
            calls.push(norm(OUT_SCALE, OUT_BIAS, b + GATE1, b + SHIFT2));
            calls.extend(product(work, s, FC1));
            calls.push(Call::Shader {
                kernel: &k.gelu,
                buffers: vec![&work.sums, &work.scales, v, &work.hidden],
                constants: vec![work.tokens as u32, FC1_SCALE as u32, FC1_BIAS as u32],
                groups: [(HIDDEN / 4 / 64) as u32, work.padded as u32, 1],
            });
            calls.push(Call::Shader {
                kernel: &k.hidq8,
                buffers: vec![&work.hidden, &work.scales, &work.spill, &work.bytes],
                constants: vec![work.tokens as u32, 0],
                groups: [work.padded as u32, 1, 1],
            });
            calls.extend(product(work, s, FC2));
            calls.push(if last {
                Call::Shader {
                    kernel: &k.resid,
                    buffers: vec![&work.x, &work.sums, &work.scales, v, &self.mods],
                    constants: vec![
                        work.tokens as u32,
                        FC2_SCALE as u32,
                        FC2_BIAS as u32,
                        (b + GATE2) as u32,
                    ],
                    groups: [(DIM / 4 / 64) as u32, work.padded as u32, 1],
                }
            } else {
                norm(
                    FC2_SCALE,
                    FC2_BIAS,
                    b + GATE2,
                    Self::base(index + 1, s) + SHIFT1,
                )
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{gpu::Select, qwen_dit::tests::db, qwen2512_dit::tests::Golden},
        *,
    };

    /// `KUNO_QWEN2512_DIT=<model> KUNO_QWEN2512_GOLDEN=<dir> [KUNO_GPU_VENDOR=<hex>]`: the
    /// captured C++ calls, as `qwen2512_gpu::tests::steps_match_cpp_engine`.
    #[test]
    #[ignore = "needs a GPU, the model and captures"]
    fn steps_match_cpp_engine() {
        let vendor = std::env::var("KUNO_GPU_VENDOR")
            .map_or(0x10de, |v| u32::from_str_radix(&v, 16).unwrap());
        let gpu = Gpu::open(Select::Vendor(vendor)).unwrap();
        println!("{}", gpu.name());
        let started = std::time::Instant::now();
        let model =
            Qwen2512Vk::load(gpu, Path::new(&std::env::var("KUNO_QWEN2512_DIT").unwrap())).unwrap();
        let (device, all) = model.resident();
        println!(
            "load {:.1} s, {device} of {all} blocks in device memory",
            started.elapsed().as_secs_f64()
        );
        // KUNO_QWEN2512_XE2=1: also the Xe2 form on the Intel GPU, from the same file.
        let intel = std::env::var("KUNO_QWEN2512_XE2").ok().map(|_| {
            let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
            let path = std::env::var("KUNO_QWEN2512_DIT").unwrap();
            (super::super::qwen2512_gpu::Qwen2512Gpu::load(gpu, Path::new(&path)).unwrap(),)
        });
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        for step in 0.. {
            if !dir.join(format!("step{step}.x.f32")).exists() {
                assert!(step > 0);
                break;
            }
            let golden = Golden::load(&dir, step);
            let prompt = model.prompt(&golden.context).unwrap();
            let times = model.times(&[golden.timestep]).remove(0);
            let mut out = Vec::new();
            let mut best = f64::MAX;
            for _ in 0..2 {
                let started = std::time::Instant::now();
                out = model
                    .step(&[&prompt], &golden.x, golden.height, golden.width, &times)
                    .unwrap()
                    .remove(0);
                best = best.min(started.elapsed().as_secs_f64());
            }
            let db = db(&golden.out, &out);
            let against = intel.as_ref().map(|(model, ..)| {
                let prompt = model.prompt(&golden.context).unwrap();
                let times = model.times(&[golden.timestep]).remove(0);
                let theirs = model
                    .step(
                        &[&prompt],
                        &golden.x,
                        golden.height,
                        golden.width,
                        &times,
                        false,
                    )
                    .unwrap()
                    .remove(0);
                super::super::qwen_dit::tests::db(&theirs, &out)
            });
            println!(
                "step {step}: t {} {}x{} tokens, {} text, {best:.3} s -> {db:.1} dB from the \
                 capture (C++, FP8){}",
                golden.timestep,
                golden.height,
                golden.width,
                prompt.tokens,
                against.map_or(String::new(), |a| format!(", {a:.1} dB from the Xe2 form"))
            );
            assert!(db > 22.0, "step {step}: {db:.1} dB");
            if let Some(a) = against {
                assert!(a > 30.0, "step {step}: {a:.1} dB from the Xe2 form");
            }
        }
    }
}
