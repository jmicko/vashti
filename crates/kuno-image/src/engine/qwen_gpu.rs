//! Qwen Image 2.1's diffusion transformer on the GPU: INT8 products and attention as Xe2
//! machine code (xe2/int8.rs, xe2/attend.rs), the glue as Vulkan shaders (`q*.comp`). Checked
//! against the CPU reference in `qwen_dit.rs`.
//!
//! The prompt and reference images form a prefix whose keys and values do not depend on the
//! step (they are modulated with the time-zero row and never attend to the target), so they
//! are computed once per request: run by run through all blocks, each run's keys and values
//! stored per block (`Prefix`). A step then runs only the target's tokens, whose attention
//! reads their own keys followed by the stored prefix (attention does not depend on the
//! keys' order, and the target's own keys are then aligned for the stream writers).
use std::{collections::HashMap, path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit::linear,
    dit_gpu::{Call, fast_kernel, run},
    gpu::{Buffer, Gpu, Kernel, Readback, bytes_of, shaders},
    qwen_dit::{self, CHANNELS, CONTEXT, DIM, HEAD, HEADS, HIDDEN, Layout, PAIRS, QwenDit},
    safetensors::Safetensors,
    xe2::{
        attend, int8,
        level_zero::{self, LevelZero},
        zebin,
    },
};

/// Tokens are padded to the products' token block.
const PAD: usize = 256;
/// The four integer products of a block: weight rows and columns.
const PRODUCTS: [(usize, usize); 4] =
    [(3 * DIM, DIM), (DIM, DIM), (2 * HIDDEN, DIM), (DIM, HIDDEN)];
const QKV: usize = 0;
const OUT: usize = 1;
const FC1: usize = 2;
const FC2: usize = 3;
/// Offsets (in values) into a layer's vector buffer.
const QUERY_NORM: usize = 0;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const SCALE_QKV: usize = KEY_NORM + HEAD;
const SCALE_OUT: usize = SCALE_QKV + 3 * DIM;
const SCALE_FC1: usize = SCALE_OUT + DIM;
const SCALE_FC2: usize = SCALE_FC1 + 2 * HIDDEN;
const VECTORS: usize = SCALE_FC2 + DIM;
/// The feed-forward activation's rotated groups that do not fit in shared memory.
const SPILLED: usize = 20 * 256;
/// Modulation vectors of one row: `1 + m0`, `tanh(m1)`, `1 + m2`, `tanh(m3)`.
const MODS: usize = 4 * DIM;

pub struct Kernels {
    normq8: Kernel,
    /// A residual update and the next norm and quantization in one pass.
    resnorm8: Kernel,
    mergeq8: Kernel,
    gateq8: Kernel,
    resid8: Kernel,
    qk: Kernel,
    value: Kernel,
    prefix: Kernel,
    mask: Kernel,
    embed: Kernel,
    final_: Kernel,
    fast: LevelZero,
}
impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self, &'static str> {
        if !gpu.shares() {
            return Err("gpu_unsupported");
        }
        Ok(Self {
            normq8: gpu.kernel(shaders::QNORMQ8, 4, 3, Some(16))?,
            resnorm8: gpu.kernel(shaders::QRESNORM8, 6, 5, Some(16))?,
            mergeq8: gpu.kernel(shaders::QMERGEQ8, 4, 3, Some(16))?,
            gateq8: gpu.kernel(shaders::QGATEQ8, 5, 3, Some(16))?,
            resid8: gpu.kernel(shaders::QRESID8, 5, 3, None)?,
            qk: gpu.kernel(shaders::QQK, 6, 8, Some(16))?,
            value: gpu.kernel(shaders::QVALUE, 5, 6, None)?,
            prefix: gpu.kernel(shaders::QPREFIX, 4, 4, None)?,
            mask: gpu.kernel(shaders::QMASK, 4, 5, Some(16))?,
            embed: gpu.kernel(shaders::QEMBED, 3, 1, None)?,
            final_: gpu.kernel(shaders::QFINAL, 4, 1, Some(16))?,
            fast: LevelZero::open()?,
        })
    }
}

/// One block's weights on the GPU: the four INT8 matrices in the machine-code product's
/// stream layout (query, key and value rows as one matrix), and the vectors at the offsets
/// above.
pub struct Layer {
    matrices: [(Buffer, level_zero::Memory); 4],
    vectors: Buffer,
}
impl Layer {
    pub fn load(
        gpu: &Gpu,
        kernels: &Kernels,
        file: &Safetensors,
        layer: usize,
    ) -> Result<Self, &'static str> {
        let name = |part: &str| format!("transformer_blocks.{layer}.{part}");
        let matrix =
            |parts: &[&str], which: usize| -> Result<(Buffer, level_zero::Memory), &'static str> {
                let (rows, columns) = PRODUCTS[which];
                let per_part = rows / parts.len();
                let tile = int8::OUTPUTS * columns;
                let staged = gpu.staged((rows * columns + int8::SLACK) as u64, |to| {
                    let (stream, slack) = to.split_at_mut(rows * columns);
                    slack.fill(0);
                    stream.par_chunks_mut(tile).enumerate().try_for_each_init(
                        || vec![0u8; tile],
                        |plain, (n, to)| {
                            let first = n * int8::OUTPUTS;
                            let part = first / per_part;
                            let tensor = name(&format!("{}.weight", parts[part]));
                            file.i8_part_into(
                                &tensor,
                                &[per_part, columns],
                                (first - part * per_part) * columns,
                                plain,
                            )?;
                            int8::weight_stream(plain, int8::OUTPUTS, columns, to);
                            Ok(())
                        },
                    )
                })?;
                let buffer = gpu.place(&staged, true)?;
                let stream = kernels.fast.import(&gpu.export(&buffer)?, staged.bytes())?;
                Ok((buffer, stream))
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
            vectors: floats_buffer(gpu, &vectors)?,
        })
    }
}

/// `values` in video memory.
fn floats_buffer(gpu: &Gpu, values: &[f32]) -> Result<Buffer, &'static str> {
    gpu.filled(values.len() as u64 * 4, |to| {
        to.copy_from_slice(bytes_of(values));
        Ok(())
    })
}

/// The small float layers, run on the CPU.
struct Small {
    linear_1: Vec<f32>,
    linear_2: Vec<f32>,
    modulation: Vec<f32>,
    norm_out: Vec<f32>,
    text_norm: Vec<f32>,
    text_in: Vec<f32>,
    text_out: Vec<f32>,
}
impl Small {
    fn read(file: &Safetensors) -> Result<Self, &'static str> {
        let t = "time_text_embed.timestep_embedder";
        Ok(Self {
            linear_1: file.floats(&format!("{t}.linear_1.weight"), &[DIM, 256])?,
            linear_2: file.floats(&format!("{t}.linear_2.weight"), &[DIM, DIM])?,
            modulation: file.floats("modulation.1.weight", &[4 * DIM, DIM])?,
            norm_out: file.floats("norm_out.linear.weight", &[DIM, DIM])?,
            text_norm: file
                .floats("txt_in.text_norm.weight", &[CONTEXT])?
                .iter()
                .map(|w| w + 1.0)
                .collect(),
            text_in: file.floats("txt_in.in_layer.weight", &[DIM, CONTEXT])?,
            text_out: file.floats("txt_in.out_layer.weight", &[DIM, DIM])?,
        })
    }
    /// For each timestep, the modulation vectors of its row and the time-zero row
    /// `[2][MODS]` and the output head's scale; one pass over the weights for all of them.
    fn times(&self, timesteps: &[f32]) -> Vec<Times> {
        let silu = |v: &mut f32| *v /= 1.0 + (-*v).exp();
        let mut rows = Vec::with_capacity(timesteps.len() * 2 * 256);
        for &t in timesteps {
            rows.extend(qwen_dit::time_embedding(t));
            rows.extend(qwen_dit::time_embedding(0.0));
        }
        let mut h = linear(&rows, 256, &self.linear_1, &[]);
        h.iter_mut().for_each(silu);
        let mut h = linear(&h, DIM, &self.linear_2, &[]);
        h.iter_mut().for_each(silu);
        let mut mods = linear(&h, DIM, &self.modulation, &[]);
        for row in mods.chunks_exact_mut(MODS) {
            for (part, values) in row.chunks_exact_mut(DIM).enumerate() {
                for v in values {
                    *v = if part % 2 == 0 { 1.0 + *v } else { v.tanh() };
                }
            }
        }
        // The head's scale comes from each step's own row.
        let own: Vec<f32> = h
            .chunks_exact(2 * DIM)
            .flat_map(|pair| pair[..DIM].to_vec())
            .collect();
        let scales = linear(&own, DIM, &self.norm_out, &[]);
        mods.chunks_exact(2 * MODS)
            .zip(scales.chunks_exact(DIM))
            .map(|(mods, scale)| Times {
                mods: mods.to_vec(),
                scale: scale.to_vec(),
            })
            .collect()
    }
    fn text(&self, context: &[f32]) -> Vec<f32> {
        let mut x = context.to_vec();
        x.par_chunks_exact_mut(CONTEXT).for_each(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / CONTEXT as f32;
            let scale = 1.0 / (mean + 1e-6).sqrt();
            for (v, w) in row.iter_mut().zip(&self.text_norm) {
                *v *= scale * w;
            }
        });
        let mut h = linear(&x, CONTEXT, &self.text_in, &[]);
        h.iter_mut().for_each(|v| *v = qwen_dit::gelu(*v));
        linear(&h, DIM, &self.text_out, &[])
    }
}

/// A step's time-dependent vectors (`QwenGpu::times`).
pub struct Times {
    mods: Vec<f32>,
    scale: Vec<f32>,
}

/// Buffers and kernels of one pass shape: `tokens` own tokens after `earlier` cached keys.
struct Work {
    /// The prefix whose target turns `turns` holds (0: none).
    turns_of: std::cell::Cell<u64>,
    tokens: usize,
    earlier: usize,
    padded: usize,
    /// Keys per head in the key/value stream: own then earlier, padded.
    keys: usize,
    stream: Buffer,
    turns: Buffer,
    rows: Buffer,
    mods: Buffer,
    final_scale: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    spill: Buffer,
    query: Buffer,
    kv: Buffer,
    mixed: Buffer,
    weights: Buffer,
    out: Buffer,
    readback: Readback,
    shared: [level_zero::Memory; 6],
    products: [(level_zero::Kernel, [u32; 3]); 4],
    attend: (level_zero::Kernel, [u32; 3]),
}
impl Work {
    fn new(
        gpu: &Gpu,
        kernels: &Kernels,
        tokens: usize,
        earlier: usize,
    ) -> Result<Self, &'static str> {
        let fast = &kernels.fast;
        let padded = tokens.next_multiple_of(PAD);
        let keys = (tokens + earlier).next_multiple_of(PAD);
        let f32s = |count: usize| gpu.buffer(count as u64 * 4);
        let sizes = [
            (padded * HIDDEN + int8::SLACK) as u64,
            (padded * 2 * HIDDEN) as u64 * 4,
            (HEADS * padded * HEAD) as u64 * 2,
            (HEADS * keys * HEAD * 2) as u64 * 2 + attend::SLACK as u64,
            (HEADS * padded * HEAD) as u64 * 4,
            (HEADS * padded) as u64 * 4,
        ];
        let buffers = [
            gpu.shared(sizes[0])?,
            gpu.shared(sizes[1])?,
            gpu.shared(sizes[2])?,
            gpu.shared(sizes[3])?,
            gpu.shared(sizes[4])?,
            gpu.shared(sizes[5])?,
        ];
        let import = |i: usize| fast.import(&gpu.export(&buffers[i])?, sizes[i]);
        let shared = [
            import(0)?,
            import(1)?,
            import(2)?,
            import(3)?,
            import(4)?,
            import(5)?,
        ];
        let product = |which: usize| -> Result<(level_zero::Kernel, [u32; 3]), &'static str> {
            let (rows, columns) = PRODUCTS[which];
            let shape = int8::Shape::linear(padded, rows, columns).ok_or("gpu_unsupported")?;
            Ok((fast_kernel(fast, &shape)?, shape.groups(rows)))
        };
        let shape = attend::Shape::new(keys, tokens + earlier, HEADS, 1.0 / (HEAD as f32).sqrt())
            .and_then(|s| s.with_queries(padded, HEADS))
            .ok_or("gpu_unsupported")?;
        let native = zebin::write(&zebin::Kernel {
            name: "attend",
            code: &attend::kernel(&shape),
            grf: 256,
            pointers: 4,
        });
        let attend = (fast.kernel(&native, "attend", 4)?, shape.groups(HEADS));
        let [bytes, sums, query, kv, mixed, weights] = buffers;
        Ok(Self {
            turns_of: std::cell::Cell::new(0),
            tokens,
            earlier,
            padded,
            keys,
            stream: f32s(padded * DIM)?,
            turns: f32s(padded * PAIRS * 2)?,
            rows: f32s(padded * CHANNELS)?,
            mods: f32s(2 * MODS)?,
            final_scale: f32s(DIM)?,
            bytes,
            scales: f32s(padded)?,
            sums,
            spill: f32s(padded * SPILLED)?,
            query,
            kv,
            mixed,
            weights,
            out: f32s(padded * CHANNELS)?,
            readback: gpu.readback((padded * CHANNELS) as u64 * 4)?,
            shared,
            products: [product(QKV)?, product(OUT)?, product(FC1)?, product(FC2)?],
            attend,
        })
    }
}

/// The prompt's (and reference images') keys and values for every block, for one request.
pub struct Prefix {
    /// Unique per prefix, so a pass knows whether its turns are this prefix's.
    id: u64,
    /// Per block: keys and values `[head][capacity][128]` f16.
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    length: usize,
    /// The target's latent size and its tokens' rotary turns.
    height: usize,
    width: usize,
    turns: Vec<f32>,
}

/// Which attention a pass uses.
#[derive(Clone, Copy, PartialEq)]
enum Attention {
    /// Causal over the own tokens, plus all earlier keys (a prompt run).
    Causal,
    /// Every own and earlier key (an image).
    Full,
}

pub struct QwenGpu {
    gpu: Gpu,
    kernels: Kernels,
    layers: Vec<Layer>,
    small: Small,
    img_in: Buffer,
    proj_out: Buffer,
    /// Pass buffers by shape (own tokens, earlier keys); a request's target pass is reused
    /// every step.
    work: Mutex<HashMap<(usize, usize), Work>>,
}
impl QwenGpu {
    pub fn load(gpu: Gpu, path: &Path) -> Result<Self, &'static str> {
        let model = QwenDit::open(path)?;
        let kernels = Kernels::new(&gpu)?;
        let layers = (0..model.layers)
            .map(|layer| Layer::load(&gpu, &kernels, &model.file, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let small = Small::read(&model.file)?;
        let img_in = floats_buffer(&gpu, &model.file.floats("img_in.weight", &[DIM, CHANNELS])?)?;
        let proj_out = floats_buffer(
            &gpu,
            &model.file.floats("proj_out.weight", &[CHANNELS, DIM])?,
        )?;
        Ok(Self {
            gpu,
            kernels,
            layers,
            small,
            img_in,
            proj_out,
            work: Mutex::new(HashMap::new()),
        })
    }

    /// Runs the prompt and reference images through every block and keeps their keys and
    /// values. `context`, `slots` and `references` as in `qwen_dit::Step`; `target` is the
    /// target latent's `(height, width)`.
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
        let half_rows = |count: usize| gpu.buffer((HEADS * count.max(1) * HEAD) as u64 * 2);
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let prefix = Prefix {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            keys: (0..self.layers.len())
                .map(|_| half_rows(length))
                .collect::<Result<_, _>>()?,
            values: (0..self.layers.len())
                .map(|_| half_rows(length))
                .collect::<Result<_, _>>()?,
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
            let work = Self::work_for(&mut works, gpu, &self.kernels, tokens, segment.start)?;
            let mut turns: Vec<f32> = rotary[segment.start * PAIRS..segment.end * PAIRS]
                .iter()
                .flatten()
                .copied()
                .collect();
            turns.resize(work.padded * PAIRS * 2, 0.0);
            gpu.upload(&work.turns, 0, bytes_of(&turns))?;
            work.turns_of.set(0);
            gpu.upload(&work.mods, 0, bytes_of(&mods))?;
            let attention = match segment.image {
                None => {
                    let rows = &text[segment.context_start * DIM..][..tokens * DIM];
                    gpu.upload(&work.stream, 0, bytes_of(rows))?;
                    Attention::Causal
                }
                Some(index) => {
                    let (latent, h, w) = references[index];
                    self.embed(work, &qwen_dit::patchify(latent, h, w))?;
                    Attention::Full
                }
            };
            for (index, layer) in self.layers.iter().enumerate() {
                let last = index + 1 == self.layers.len();
                let store = (&prefix.keys[index], &prefix.values[index], length);
                run(
                    gpu,
                    &self.block(layer, work, 1, attention, Some(store), &prefix, index, last),
                )?;
            }
        }
        Ok(prefix)
    }

    fn work_for<'a>(
        works: &'a mut HashMap<(usize, usize), Work>,
        gpu: &Gpu,
        kernels: &Kernels,
        tokens: usize,
        earlier: usize,
    ) -> Result<&'a Work, &'static str> {
        match works.entry((tokens, earlier)) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                Ok(entry.insert(Work::new(gpu, kernels, tokens, earlier)?))
            }
        }
    }

    /// Latent rows `[tokens][CHANNELS]` into the stream.
    fn embed(&self, work: &Work, rows: &[f32]) -> Result<(), &'static str> {
        self.gpu.upload(&work.rows, 0, bytes_of(rows))?;
        run(
            &self.gpu,
            &[Call::Shader {
                kernel: &self.kernels.embed,
                buffers: vec![&work.rows, &self.img_in, &work.stream],
                constants: vec![work.tokens as u32],
                groups: [work.tokens as u32, 1, 1],
            }],
        )
    }

    /// The time-dependent vectors of each of `timesteps`, for `step_at`.
    pub fn times(&self, timesteps: &[f32]) -> Vec<Times> {
        self.small.times(timesteps)
    }

    /// One evaluation of the target `x` (`[CHANNELS][height][width]`) at `timestep` after a
    /// prefix: the model's output in the same shape.
    pub fn step(
        &self,
        prefix: &Prefix,
        x: &[f32],
        timestep: f32,
    ) -> Result<Vec<f32>, &'static str> {
        self.step_at(prefix, x, &self.small.times(&[timestep])[0])
    }

    /// As `step`, with the timestep's vectors computed beforehand.
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
        let work = Self::work_for(&mut works, gpu, &self.kernels, tokens, prefix.length)?;
        if work.turns_of.get() != prefix.id {
            let mut turns = prefix.turns.clone();
            turns.resize(work.padded * PAIRS * 2, 0.0);
            gpu.upload(&work.turns, 0, bytes_of(&turns))?;
            work.turns_of.set(prefix.id);
        }
        gpu.upload(&work.mods, 0, bytes_of(&times.mods))?;
        gpu.upload(&work.final_scale, 0, bytes_of(&times.scale))?;
        gpu.upload(
            &work.rows,
            0,
            bytes_of(&qwen_dit::patchify(x, height, width)),
        )?;
        for (index, layer) in self.layers.iter().enumerate() {
            let mut calls = Vec::new();
            if index == 0 {
                calls.push(Call::Shader {
                    kernel: &self.kernels.embed,
                    buffers: vec![&work.rows, &self.img_in, &work.stream],
                    constants: vec![tokens as u32],
                    groups: [tokens as u32, 1, 1],
                });
            }
            calls.extend(self.block(layer, work, 0, Attention::Full, None, prefix, index, false));
            if index + 1 == self.layers.len() {
                calls.push(Call::Shader {
                    kernel: &self.kernels.final_,
                    buffers: vec![&work.stream, &work.final_scale, &self.proj_out, &work.out],
                    constants: vec![tokens as u32],
                    groups: [tokens as u32, 1, 1],
                });
            }
            run(gpu, &calls)?;
        }
        let mut out = vec![0f32; tokens * CHANNELS];
        gpu.read_floats(&work.out, 0, &work.readback, &mut out)?;
        Ok(qwen_dit::unpatchify(&out, height, width))
    }

    /// The calls of one block on `work.stream` with modulation row `row` (0 the step's, 1
    /// time zero). With `store`, the pass's keys and values are kept in the prefix cache at
    /// rows `work.earlier..`; with `last`, only they are needed, so the block stops after
    /// them.
    #[allow(clippy::too_many_arguments)]
    fn block<'a>(
        &'a self,
        layer: &'a Layer,
        w: &'a Work,
        row: usize,
        attention: Attention,
        store: Option<(&'a Buffer, &'a Buffer, usize)>,
        prefix: &'a Prefix,
        index: usize,
        last: bool,
    ) -> Vec<Call<'a>> {
        let k = &self.kernels;
        let (n, np) = (w.tokens as u32, w.padded as u32);
        let v = &layer.vectors;
        let mods = |part: usize| ((row * MODS) + part * DIM) as u32;
        let product = |which: usize| {
            let (kernel, groups) = &w.products[which];
            Call::Product {
                fast: &k.fast,
                kernel,
                memory: vec![&w.shared[0], &layer.matrices[which].1, &w.shared[1]],
                groups: *groups,
            }
        };
        let norm = |part: usize| Call::Shader {
            kernel: &k.normq8,
            buffers: vec![&w.stream, &w.mods, &w.bytes, &w.scales],
            constants: vec![n, mods(part), 1],
            groups: [np, 1, 1],
        };
        let residual = |scale: usize, part: usize| Call::Shader {
            kernel: &k.resid8,
            buffers: vec![&w.stream, &w.sums, &w.scales, v, &w.mods],
            constants: vec![n, scale as u32, mods(part)],
            groups: [(DIM / 4 / 64) as u32, np, 1],
        };
        // The residual with gate `part`, then the norm with modulation `next`.
        let residual_norm = |scale: usize, part: usize, next: usize| Call::Shader {
            kernel: &k.resnorm8,
            buffers: vec![&w.stream, &w.mods, &w.bytes, &w.scales, &w.sums, v],
            constants: vec![n, mods(next), 1, scale as u32, mods(part)],
            groups: [np, 1, 1],
        };
        // Blocks after the first find their input quantized by the previous block's end.
        let first = index == 0;
        let fuse_out = index + 1 < self.layers.len();
        let (cache_keys, cache_values, capacity) = store.unwrap_or((&w.kv, &w.kv, 0));
        let stored = u32::from(store.is_some());
        let mut calls = Vec::new();
        if first {
            calls.push(norm(0));
        }
        calls.push(product(QKV));
        for (side, to, weight) in [(0u32, &w.query, QUERY_NORM), (1, &w.kv, KEY_NORM)] {
            calls.push(Call::Shader {
                kernel: &k.qk,
                buffers: vec![&w.sums, &w.scales, v, &w.turns, to, cache_keys],
                constants: vec![
                    n,
                    if side == 0 { np } else { w.keys as u32 },
                    side,
                    SCALE_QKV as u32,
                    weight as u32,
                    stored * side,
                    w.earlier as u32,
                    capacity as u32,
                ],
                groups: [(HEADS / 4) as u32, np, 1],
            });
        }
        calls.push(Call::Shader {
            kernel: &k.value,
            buffers: vec![&w.sums, &w.scales, v, &w.kv, cache_values],
            constants: vec![
                n,
                w.keys as u32,
                SCALE_QKV as u32,
                stored,
                w.earlier as u32,
                capacity as u32,
            ],
            groups: [HEADS as u32, np / 32, 1],
        });
        if w.earlier > 0 {
            calls.push(Call::Shader {
                kernel: &k.prefix,
                buffers: vec![&prefix.keys[index], &prefix.values[index], &w.kv, &w.kv],
                constants: vec![
                    w.earlier as u32,
                    n,
                    w.keys as u32,
                    prefix.length.max(1) as u32,
                ],
                groups: [(w.earlier * 16).div_ceil(64) as u32, HEADS as u32, 1],
            });
        }
        if last {
            return calls;
        }
        match attention {
            Attention::Full => {
                let (kernel, groups) = &w.attend;
                calls.push(Call::Product {
                    fast: &k.fast,
                    kernel,
                    memory: vec![&w.shared[2], &w.shared[3], &w.shared[4], &w.shared[5]],
                    groups: *groups,
                });
            }
            Attention::Causal => calls.push(Call::Shader {
                kernel: &k.mask,
                buffers: vec![&w.query, &w.kv, &w.mixed, &w.weights],
                constants: vec![
                    np,
                    w.keys as u32,
                    n,
                    w.earlier as u32,
                    (1.0 / (HEAD as f32).sqrt()).to_bits(),
                ],
                groups: [n, HEADS as u32, 1],
            }),
        }
        calls.push(Call::Shader {
            kernel: &k.mergeq8,
            buffers: vec![&w.mixed, &w.weights, &w.bytes, &w.scales],
            constants: vec![n, np, 1],
            groups: [np, 1, 1],
        });
        calls.push(product(OUT));
        calls.push(residual_norm(SCALE_OUT, 1, 2));
        calls.push(product(FC1));
        calls.push(Call::Shader {
            kernel: &k.gateq8,
            buffers: vec![&w.sums, &w.scales, v, &w.spill, &w.bytes],
            constants: vec![n, SCALE_FC1 as u32, 1],
            groups: [np, 1, 1],
        });
        calls.push(product(FC2));
        calls.push(if fuse_out {
            residual_norm(SCALE_FC2, 3, 0)
        } else {
            residual(SCALE_FC2, 3)
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

    /// `KUNO_QWEN_DIT=<model> KUNO_QWEN_GOLDEN=<dir>`: every captured step through the GPU
    /// form against the C++ engine's output, with one prefix per distinct prompt.
    #[test]
    #[ignore]
    fn steps_match_cpp_engine() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let model =
            QwenGpu::load(gpu, Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
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
            let started = std::time::Instant::now();
            let out = model.step(&prefix, &golden.x, golden.timestep).unwrap();
            let step_time = started.elapsed().as_secs_f64();
            let db = db(&golden.out, &out);
            println!(
                "step {step}: t {} {}x{} prefix {} tokens {prefix_time:.2} s, step {step_time:.3} s -> {db:.1} dB",
                golden.timestep, golden.height, golden.width, prefix.length
            );
            assert!(db > 25.0, "step {step}: {db:.1} dB");
        }
    }

    /// `KUNO_QWEN_DIT=<model> KUNO_QWEN_GOLDEN=<dir>`: step time at the image benchmark's
    /// size (1184 x 1776, a 74 x 111 latent) with a captured prompt.
    #[test]
    #[ignore]
    fn step_speed_at_benchmark_size() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let model =
            QwenGpu::load(gpu, Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_GOLDEN").unwrap());
        let golden = Golden::load(&dir, 0);
        let (height, width) = (111, 74);
        let prefix = model
            .prefix(&golden.context, &golden.slots, &[], (height, width))
            .unwrap();
        let mut state = 1u64;
        let x: Vec<f32> = (0..CHANNELS * height * width)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        let mut best = f64::MAX;
        for _ in 0..4 {
            let started = std::time::Instant::now();
            let out = model.step(&prefix, &x, 500.0).unwrap();
            best = best.min(started.elapsed().as_secs_f64());
            assert!(out.iter().all(|v| v.is_finite()));
        }
        println!("{} tokens: best step {best:.3} s", height * width);
    }

    /// `KUNO_QWEN_DIT=<model> KUNO_QWEN_GOLDEN=<dir>`: each kernel of one block at the
    /// benchmark's size alone, and the block as one call list.
    #[test]
    #[ignore]
    fn block_costs_at_benchmark_size() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let model =
            QwenGpu::load(gpu, Path::new(&std::env::var("KUNO_QWEN_DIT").unwrap())).unwrap();
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN_GOLDEN").unwrap());
        let golden = Golden::load(&dir, 0);
        let (height, width) = (111, 74);
        let prefix = model
            .prefix(&golden.context, &golden.slots, &[], (height, width))
            .unwrap();
        let x = vec![0.5f32; CHANNELS * height * width];
        model.step(&prefix, &x, 500.0).unwrap();
        let mut works = model.work.lock().unwrap();
        let work = QwenGpu::work_for(
            &mut works,
            &model.gpu,
            &model.kernels,
            height * width,
            prefix.length,
        )
        .unwrap();
        let calls = model.block(
            &model.layers[3],
            work,
            0,
            Attention::Full,
            None,
            &prefix,
            3,
            false,
        );
        for _ in 0..2 {
            let started = std::time::Instant::now();
            run(&model.gpu, &calls).unwrap();
            println!(
                "block as one list: {:.2} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        drop(works);
        let started = std::time::Instant::now();
        let _ = model.small.times(&[500.0]);
        println!(
            "time layers on the CPU: {:.2} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        let started = std::time::Instant::now();
        model.step(&prefix, &x, 500.0).unwrap();
        println!(
            "whole step: {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        let mut works = model.work.lock().unwrap();
        let work = QwenGpu::work_for(
            &mut works,
            &model.gpu,
            &model.kernels,
            height * width,
            prefix.length,
        )
        .unwrap();
        let calls = model.block(
            &model.layers[3],
            work,
            0,
            Attention::Full,
            None,
            &prefix,
            3,
            false,
        );
        let k = &model.kernels;
        let mut totals: Vec<(String, f64)> = Vec::new();
        for (i, call) in calls.iter().enumerate() {
            let name = match call {
                Call::Shader { kernel, .. } => {
                    let names = [
                        (&k.normq8, "norm+quantize"),
                        (&k.resnorm8, "residual+norm+quantize"),
                        (&k.mergeq8, "merge+quantize"),
                        (&k.gateq8, "gate+quantize"),
                        (&k.resid8, "residual"),
                        (&k.qk, "queries/keys"),
                        (&k.value, "values"),
                        (&k.prefix, "prefix copy"),
                        (&k.mask, "masked attention"),
                    ];
                    names
                        .iter()
                        .find(|n| std::ptr::eq(n.0, *kernel))
                        .map_or("?", |n| n.1)
                        .to_string()
                }
                Call::Product { .. } => format!("machine code #{i}"),
            };
            let mut best = f64::MAX;
            for _ in 0..3 {
                let started = std::time::Instant::now();
                run(&model.gpu, std::slice::from_ref(call)).unwrap();
                best = best.min(started.elapsed().as_secs_f64());
            }
            totals.push((name, best * 1e3));
        }
        for (name, ms) in &totals {
            println!("{ms:8.2} ms  {name}");
        }
        println!("{:8.2} ms  sum", totals.iter().map(|t| t.1).sum::<f64>());
    }
}
