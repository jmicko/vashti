//! Qwen Image 2512's diffusion transformer on the Intel GPU: the block linears as INT8
//! machine-code products (xe2/int8.rs) with ConvRot inputs, one joint attention per block
//! (xe2/attend.rs), and the glue as OpenCL kernels (`xe2/qwen2512.cl`), a whole pass in one
//! Level Zero command list. Checked against the CPU reference in `qwen2512_dit.rs`.
//!
//! The file's weights are FP8 (e4m3) without scales. At load the block linears are widened,
//! rotated in groups of 256 along their inputs and rounded to INT8 per output row (13.6 GB);
//! the modulation layers, a third of the model, stay FP8 on the device (6.8 GB) and are
//! evaluated exactly once per step for both CFG passes.
//!
//! The image and text streams keep their own rows and products. Attention sees them as one
//! sequence: queries in one stream of slots (image `0..`, text after the image's padded
//! rows), keys and values in one key stream (image `0..n`, text right after the image's live
//! keys, so no dead key sits between them). Attention does not depend on the keys' order.
use std::{collections::HashMap, path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit::{GROUP, linear, quantize, rotate},
    dit_gpu::fast_kernel,
    gpu::{Buffer, Gpu, Readback, bytes_of},
    qwen_dit::{rotary, time_embedding},
    qwen2512_dit::{
        CHANNELS, CONTEXT, DIM, HEAD, HEADS, HIDDEN, PAIRS, Qwen2512Dit, outlier_norms, patchify,
        positions, silu, unpatchify,
    },
    safetensors::Safetensors,
    xe2::{
        attend, int8,
        level_zero::{self, LevelZero},
        zebin,
    },
};

/// Image rows are padded to the products' token block, text rows to their token tile.
const PAD_IMAGE: usize = 256;
const PAD_TEXT: usize = int8::TOKENS;
/// A stream's four products: weight rows and columns.
const PRODUCTS: [(usize, usize); 4] = [(3 * DIM, DIM), (DIM, DIM), (HIDDEN, DIM), (DIM, HIDDEN)];
const QKV: usize = 0;
const OUT: usize = 1;
const FC1: usize = 2;
const FC2: usize = 3;
/// Byte offsets of the scaled products' weight scales in their weight buffers.
const QKV_SCALES: usize = 3 * DIM * DIM + int8::SLACK;
const FC1_SCALES: usize = HIDDEN * DIM + int8::SLACK;
/// Offsets (in values) into a stream's vector buffer.
const QUERY_NORM: usize = 0;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const QKV_BIAS: usize = KEY_NORM + HEAD;
const OUT_BIAS: usize = QKV_BIAS + 3 * DIM;
const OUT_SCALE: usize = OUT_BIAS + DIM;
const FC1_BIAS: usize = OUT_SCALE + DIM;
const FC2_BIAS: usize = FC1_BIAS + HIDDEN;
const FC2_SCALE: usize = FC2_BIAS + DIM;
const VECTORS: usize = FC2_SCALE + DIM;
/// Modulation rows of one stream in one block (shift, scale, gate, twice), and of a block.
const MOD_ROWS: usize = 6 * DIM;
const BLOCK_MODS: usize = 2 * MOD_ROWS;
const SHIFT1: usize = 0;
const GATE1: usize = 2 * DIM;
const SHIFT2: usize = 3 * DIM;
const GATE2: usize = 5 * DIM;
const ATTEND_GROUP: u32 = 8;
const TEXT: usize = 1;

/// The glue (`xe2/qwen2512.cl`, compiled for the B70 by `ocloc` into `xe2/qwen2512.bin`).
const GLUE: &[u8] = include_bytes!("xe2/qwen2512.bin");
struct Glue {
    norm: level_zero::Kernel,
    resnorm: level_zero::Kernel,
    resid: level_zero::Kernel,
    qk: level_zero::Kernel,
    value: level_zero::Kernel,
    merge: level_zero::Kernel,
    hidden: level_zero::Kernel,
    embed: level_zero::Kernel,
    copy: level_zero::Kernel,
    final_: level_zero::Kernel,
    modulate: level_zero::Kernel,
    modulate8: level_zero::Kernel,
}
impl Glue {
    fn new(fast: &LevelZero) -> Result<Self, &'static str> {
        let k = |name: &str, group: u32| fast.compiled_kernel(GLUE, name, [group, 1, 1]);
        Ok(Self {
            norm: k("k_norm", 256)?,
            resnorm: k("k_resnorm", 256)?,
            resid: k("k_resid", 64)?,
            qk: k("k_qk", 64)?,
            value: k("k_value", 256)?,
            merge: k("k_merge", 256)?,
            hidden: k("k_hidden", 256)?,
            embed: k("k_embed", 256)?,
            copy: k("k_copy", 256)?,
            final_: k("k_final", 256)?,
            modulate: k("k_mod", 256)?,
            modulate8: k("k_mod8", 256)?,
        })
    }
}

/// A buffer in video memory with its Level Zero view.
struct Shared {
    buffer: Buffer,
    memory: level_zero::Memory,
}
impl Shared {
    fn new(gpu: &Gpu, fast: &LevelZero, bytes: u64) -> Result<Self, &'static str> {
        let buffer = gpu.shared(bytes)?;
        let memory = fast.import(&gpu.export(&buffer)?, bytes)?;
        Ok(Self { buffer, memory })
    }
    fn zeroed(gpu: &Gpu, fast: &LevelZero, bytes: u64) -> Result<Self, &'static str> {
        let buffer = gpu.filled_shared(bytes, |to| {
            to.fill(0);
            Ok(())
        })?;
        let memory = fast.import(&gpu.export(&buffer)?, bytes)?;
        Ok(Self { buffer, memory })
    }
    fn floats(gpu: &Gpu, fast: &LevelZero, values: &[f32]) -> Result<Self, &'static str> {
        Self::bytes(gpu, fast, bytes_of(values))
    }
    fn bytes(gpu: &Gpu, fast: &LevelZero, values: &[u8]) -> Result<Self, &'static str> {
        let bytes = values.len() as u64;
        let buffer = gpu.filled_shared(bytes, |to| {
            to.copy_from_slice(values);
            Ok(())
        })?;
        let memory = fast.import(&gpu.export(&buffer)?, bytes)?;
        Ok(Self { buffer, memory })
    }
}

/// A matrix (its rows split over `tensors`) in the product's weight stream layout, with each
/// row's scale: copied from an INT8 file (rows already rotated, `<base>.weight_scale` beside
/// them), else widened from its stored float type (FP8 times its `scale_weight` if any) and
/// rotated and rounded here.
fn int8_stream(
    file: &Safetensors,
    tensors: &[String],
    rows: usize,
    columns: usize,
    to: &mut [u8],
) -> Result<Vec<f32>, &'static str> {
    let per_part = rows / tensors.len();
    let tile = int8::OUTPUTS * columns;
    if tensors.iter().all(|t| file.is_i8(t)) {
        let mut scales = Vec::with_capacity(rows);
        for t in tensors {
            let base = t.strip_suffix(".weight").ok_or("model_tensor_invalid")?;
            scales.extend(file.f32(&format!("{base}.weight_scale"), &[per_part, 1])?);
        }
        to[..rows * columns]
            .par_chunks_mut(tile)
            .enumerate()
            .try_for_each_init(
                || vec![0u8; tile],
                |plain, (n, to)| {
                    for (i, row) in plain.chunks_exact_mut(columns).enumerate() {
                        let index = n * int8::OUTPUTS + i;
                        let (part, within) = (index / per_part, index % per_part);
                        file.i8_part_into(
                            &tensors[part],
                            &[per_part, columns],
                            within * columns,
                            row,
                        )?;
                    }
                    int8::weight_stream(plain, int8::OUTPUTS, columns, to);
                    Ok::<_, &'static str>(())
                },
            )?;
        return Ok(scales);
    }
    let parts = tensors
        .iter()
        .map(|t| float_matrix(file, t, per_part, columns))
        .collect::<Result<Vec<_>, _>>()?;
    let scales: Vec<Vec<f32>> = to[..rows * columns]
        .par_chunks_mut(tile)
        .enumerate()
        .map(|(n, to)| {
            let mut widened = vec![0f32; tile];
            for (i, row) in widened.chunks_exact_mut(columns).enumerate() {
                let index = n * int8::OUTPUTS + i;
                let (part, within) = (index / per_part, index % per_part);
                row.copy_from_slice(&parts[part][within * columns..][..columns]);
            }
            let (bytes, scales) = quantize(&widened, columns);
            // SAFETY: i8 and u8 have the same size and alignment.
            let plain: &[u8] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len()) };
            int8::weight_stream(plain, int8::OUTPUTS, columns, to);
            scales
        })
        .collect();
    Ok(scales.concat())
}

/// A float matrix widened to f32 (FP8 times its `scale_weight` if the file has one).
fn float_matrix(
    file: &Safetensors,
    name: &str,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, &'static str> {
    let mut weight = file.floats(name, &[rows, columns])?;
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let scale = format!("{base}.scale_weight");
    if file.shape(&scale).is_some() {
        let scale = file.f32(&scale, &[])?[0];
        weight.par_iter_mut().for_each(|v| *v *= scale);
    }
    Ok(weight)
}

/// Modulation rows on the device: exact FP8 bytes (ComfyUI's file), or INT8 in the rotated
/// basis with a scale per row (converted files, or rounded at load from BF16).
enum Modulation {
    Fp8 {
        weight: Shared,
        bias: Shared,
    },
    Int8 {
        weight: Shared,
        scales: Shared,
        bias: Shared,
    },
}
impl Modulation {
    fn load(
        gpu: &Gpu,
        fast: &LevelZero,
        file: &Safetensors,
        names: &[String],
        rows: usize,
    ) -> Result<Self, &'static str> {
        let mut bias = Vec::with_capacity(rows * names.len());
        for name in names {
            let base = name.strip_suffix(".weight").ok_or("model_tensor_invalid")?;
            bias.extend(file.floats(&format!("{base}.bias"), &[rows])?);
        }
        let bias = Shared::floats(gpu, fast, &bias)?;
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
            return Ok(Self::Fp8 {
                weight: Shared::bytes(gpu, fast, &weight)?,
                bias,
            });
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
        // SAFETY: i8 and u8 have the same size and alignment.
        let raw = unsafe { std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), weight.len()) };
        Ok(Self::Int8 {
            weight: Shared::bytes(gpu, fast, raw)?,
            scales: Shared::floats(gpu, fast, &scales)?,
            bias,
        })
    }
}

/// One stream's weights in one block.
struct StreamWeights {
    matrices: [Shared; 4],
    vectors: Shared,
}
/// One block: the image and text streams' weights and their modulation rows
/// (`[stream][6 * DIM][DIM]`) with biases.
struct Layer {
    streams: [StreamWeights; 2],
    modulation: Modulation,
    /// Its query/key norms have an outlier channel: fast mode keeps half-float attention here.
    outlier: bool,
}
impl Layer {
    fn load(
        gpu: &Gpu,
        fast: &LevelZero,
        model: &Qwen2512Dit,
        layer: usize,
    ) -> Result<Self, &'static str> {
        let file = &model.file;
        let name = |part: &str| model.name(&format!("transformer_blocks.{layer}.{part}"));
        let stream = |qkv: [&str; 3],
                      out: &str,
                      mlp: &str,
                      norms: [&str; 2]|
         -> Result<StreamWeights, &'static str> {
            let mut vectors = Vec::with_capacity(VECTORS);
            for norm in norms {
                vectors.extend(file.floats(&name(&format!("attn.{norm}.weight")), &[HEAD])?);
            }
            for part in qkv {
                vectors.extend(file.floats(&name(&format!("attn.{part}.bias")), &[DIM])?);
            }
            vectors.extend(file.floats(&name(&format!("attn.{out}.bias")), &[DIM])?);
            let out_scale_at = vectors.len();
            vectors.resize(vectors.len() + DIM, 0.0);
            vectors.extend(file.floats(&name(&format!("{mlp}.net.0.proj.bias")), &[HIDDEN])?);
            vectors.extend(file.floats(&name(&format!("{mlp}.net.2.bias")), &[DIM])?);
            let fc2_scale_at = vectors.len();
            vectors.resize(vectors.len() + DIM, 0.0);
            debug_assert_eq!(vectors.len(), VECTORS);
            let matrix = |tensors: &[String], which: usize, scaled: bool| {
                let (rows, columns) = PRODUCTS[which];
                let total = rows * columns + int8::SLACK + if scaled { rows * 4 } else { 0 };
                let mut scales = Vec::new();
                let staged = gpu.staged(total as u64, |to| {
                    scales = int8_stream(file, tensors, rows, columns, to)?;
                    let (_, rest) = to.split_at_mut(rows * columns);
                    let (slack, tail) = rest.split_at_mut(int8::SLACK);
                    slack.fill(0);
                    if scaled {
                        tail.copy_from_slice(bytes_of(&scales));
                    }
                    Ok(())
                })?;
                let buffer = gpu.place(&staged, true)?;
                let memory = fast.import(&gpu.export(&buffer)?, staged.bytes())?;
                Ok::<_, &'static str>((Shared { buffer, memory }, scales))
            };
            let a = |part: &str| name(&format!("attn.{part}.weight"));
            let (qkv_matrix, _) = matrix(&qkv.map(a), QKV, true)?;
            let (out_matrix, out_scales) = matrix(&[a(out)], OUT, false)?;
            let (fc1, _) = matrix(&[name(&format!("{mlp}.net.0.proj.weight"))], FC1, true)?;
            let (fc2, fc2_scales) = matrix(&[name(&format!("{mlp}.net.2.weight"))], FC2, false)?;
            vectors[out_scale_at..out_scale_at + DIM].copy_from_slice(&out_scales);
            vectors[fc2_scale_at..fc2_scale_at + DIM].copy_from_slice(&fc2_scales);
            Ok(StreamWeights {
                matrices: [qkv_matrix, out_matrix, fc1, fc2],
                vectors: Shared::floats(gpu, fast, &vectors)?,
            })
        };
        let image = stream(
            ["to_q", "to_k", "to_v"],
            "to_out.0",
            "img_mlp",
            ["norm_q", "norm_k"],
        )?;
        let text = stream(
            ["add_q_proj", "add_k_proj", "add_v_proj"],
            "to_add_out",
            "txt_mlp",
            ["norm_added_q", "norm_added_k"],
        )?;
        let norms = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]
            .iter()
            .map(|n| file.floats(&name(&format!("attn.{n}.weight")), &[HEAD]))
            .collect::<Result<Vec<_>, _>>()?;
        let modulation = Modulation::load(
            gpu,
            fast,
            file,
            &[name("img_mod.1.weight"), name("txt_mod.1.weight")],
            MOD_ROWS,
        )?;
        Ok(Self {
            streams: [image, text],
            modulation,
            outlier: outlier_norms(&norms),
        })
    }
}

/// The small float layers, run on the CPU.
struct Small {
    linear_1: (Vec<f32>, Vec<f32>),
    linear_2: (Vec<f32>, Vec<f32>),
    text_norm: Vec<f32>,
    text_in: (Vec<f32>, Vec<f32>),
}

/// A step's time-dependent input: `silu(time embedding)`.
pub struct Times {
    silu: Vec<f32>,
}

/// A prompt's rows in the text stream's width (`txt_in` of the normed encoder rows).
pub struct Prompt {
    rows: Vec<f32>,
    tokens: usize,
}
impl Prompt {
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// Buffers of one stream in a pass shape.
struct StreamWork {
    tokens: usize,
    padded: usize,
    /// Query slot and key index of the stream's first token.
    slot: usize,
    key: usize,
    x: Shared,
    bytes: Shared,
    scales: Shared,
    sums: Shared,
    turns: Shared,
    /// Text only: the prompt's rows, copied into `x` at the start of a pass.
    start: Option<Shared>,
    products: [(level_zero::Kernel, [u32; 3]); 4],
}

/// Buffers and kernels of one pass shape: an image of `height x width` tokens and a prompt
/// of `text` tokens.
struct Work {
    streams: [StreamWork; 2],
    query: Shared,
    kv: Shared,
    mixed: Shared,
    weights: Shared,
    rows: Shared,
    out: Shared,
    readback: Readback,
    query_tiles: u32,
    key_tiles: u32,
    attend: (level_zero::Kernel, [u32; 3]),
    attend8: (level_zero::Kernel, [u32; 3]),
}
impl Work {
    fn new(
        gpu: &Gpu,
        fast: &LevelZero,
        height: usize,
        width: usize,
        text: usize,
    ) -> Result<Self, &'static str> {
        let image = height * width;
        let (pi, pt) = (
            image.next_multiple_of(PAD_IMAGE),
            text.next_multiple_of(PAD_TEXT),
        );
        let queries = pi + pt;
        let keys = (image + text).next_multiple_of(attend::TILE);
        let rotation = rotary(&positions(text, height, width));
        let turns = |range: std::ops::Range<usize>| -> Vec<f32> {
            rotation[range.start * PAIRS..range.end * PAIRS]
                .iter()
                .flat_map(|[c, s]| [*c, *s])
                .collect()
        };
        let stream = |tokens: usize,
                      padded: usize,
                      slot: usize,
                      key: usize,
                      turns: Vec<f32>,
                      text: bool|
         -> Result<StreamWork, &'static str> {
            let product = |which: usize| -> Result<(level_zero::Kernel, [u32; 3]), &'static str> {
                let (rows, columns) = PRODUCTS[which];
                let shape = match which {
                    QKV => int8::Shape::scaled8(padded, rows, columns, QKV_SCALES as u32),
                    FC1 => int8::Shape::scaled8(padded, rows, columns, FC1_SCALES as u32),
                    _ => int8::Shape::linear(padded, rows, columns),
                }
                .ok_or("gpu_unsupported")?;
                let blocks = ((shape.tx + 64) / 128).max(1);
                let shape = int8::Shape {
                    xc: shape.tx.div_ceil(blocks),
                    ..shape
                };
                let kernel = if which == QKV || which == FC1 {
                    super::xe2::load(fast, &int8::kernel(&shape), 4)?
                } else {
                    fast_kernel(fast, &shape)?
                };
                Ok((kernel, shape.groups(rows)))
            };
            let sums = (padded * (2 * HIDDEN).max(4 * DIM).max(6 * DIM)) as u64;
            Ok(StreamWork {
                tokens,
                padded,
                slot,
                key,
                x: Shared::zeroed(gpu, fast, (padded * DIM * 4) as u64)?,
                bytes: Shared::zeroed(gpu, fast, (padded * HIDDEN + int8::SLACK) as u64)?,
                scales: Shared::zeroed(gpu, fast, ((padded + 64) * 4) as u64)?,
                sums: Shared::zeroed(gpu, fast, sums)?,
                turns: Shared::floats(gpu, fast, &turns)?,
                start: if text {
                    Some(Shared::zeroed(gpu, fast, (padded * DIM * 4) as u64)?)
                } else {
                    None
                },
                products: [product(QKV)?, product(OUT)?, product(FC1)?, product(FC2)?],
            })
        };
        let image_turns = turns(text..text + image);
        let text_turns = turns(0..text);
        let streams = [
            stream(image, pi, 0, 0, image_turns, false)?,
            stream(text, pt, pi, image, text_turns, true)?,
        ];
        let shape = attend::Shape::new(keys, image + text, HEADS, 1.0 / (HEAD as f32).sqrt())
            .and_then(|s| s.with_queries(queries, HEADS))
            .map(|s| attend::Shape {
                group: ATTEND_GROUP,
                ..s
            })
            .ok_or("gpu_unsupported")?;
        let attention = |shape: &attend::Shape| -> Result<_, &'static str> {
            let native = zebin::write(&zebin::Kernel {
                name: "attend",
                code: &attend::kernel(shape),
                grf: 256,
                pointers: 4,
            });
            let kernel = fast.kernel(&native, "attend", 4)?;
            kernel.set_group([16 * ATTEND_GROUP, 1, 1])?;
            Ok((kernel, shape.groups(HEADS)))
        };
        Ok(Self {
            streams,
            // Slots no pass writes stay zero (finite scores for the padding queries).
            query: Shared::zeroed(gpu, fast, (HEADS * queries * HEAD * 2) as u64)?,
            kv: Shared::zeroed(
                gpu,
                fast,
                (HEADS * keys / attend::TILE * attend::CHUNK_BYTES + attend::SLACK) as u64,
            )?,
            mixed: Shared::zeroed(gpu, fast, (HEADS * queries * HEAD * 4) as u64)?,
            weights: Shared::zeroed(gpu, fast, (HEADS * queries * 4) as u64)?,
            rows: Shared::zeroed(gpu, fast, (pi * CHANNELS * 4) as u64)?,
            out: Shared::zeroed(gpu, fast, (pi * CHANNELS * 4) as u64)?,
            readback: gpu.readback((image * CHANNELS * 4) as u64)?,
            query_tiles: (queries / attend::TILE) as u32,
            key_tiles: (keys / attend::TILE) as u32,
            attend: attention(&shape)?,
            attend8: attention(&attend::Shape {
                int8: true,
                ..shape
            })?,
        })
    }
}

pub struct Qwen2512Gpu {
    gpu: Gpu,
    fast: LevelZero,
    glue: Glue,
    layers: Vec<Layer>,
    small: Small,
    img_in: (Shared, Shared),
    proj_out: (Shared, Shared),
    final_modulation: Modulation,
    /// This step's modulation rows `[block][stream][6 * DIM]`, and the output head's (scale,
    /// shift).
    mods: Shared,
    final_mods: Shared,
    /// This step's `silu(time embedding)`, and rotated in groups of 256 for INT8 modulation.
    time_rotated: Shared,
    time: Shared,
    work: Mutex<HashMap<(usize, usize, usize), Work>>,
}
impl Qwen2512Gpu {
    pub fn load(gpu: Gpu, path: &Path) -> Result<Self, &'static str> {
        if !gpu.shares() {
            return Err("gpu_unsupported");
        }
        let model = Qwen2512Dit::open(path)?;
        let fast = LevelZero::open()?;
        let next = std::sync::atomic::AtomicUsize::new(0);
        let loaded: Vec<Vec<(usize, Result<Layer, &'static str>)>> = std::thread::scope(|scope| {
            let loaders: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let layer = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if layer >= model.layers {
                                return done;
                            }
                            let result = Layer::load(&gpu, &fast, &model, layer);
                            let failed = result.is_err();
                            done.push((layer, result));
                            if failed {
                                next.store(model.layers, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    })
                })
                .collect();
            loaders
                .into_iter()
                .map(|l| l.join().unwrap_or_default())
                .collect()
        });
        let mut loaded: Vec<_> = loaded.into_iter().flatten().collect();
        loaded.sort_by_key(|(layer, _)| *layer);
        if loaded.len() != model.layers {
            return Err("gpu_failed");
        }
        let layers = loaded
            .into_iter()
            .map(|(_, layer)| layer)
            .collect::<Result<Vec<_>, _>>()?;
        let f = |name: &str, shape: &[usize]| model.floats(name, shape);
        let t = "time_text_embed.timestep_embedder";
        let small = Small {
            linear_1: (
                f(&format!("{t}.linear_1.weight"), &[DIM, 256])?,
                f(&format!("{t}.linear_1.bias"), &[DIM])?,
            ),
            linear_2: (
                f(&format!("{t}.linear_2.weight"), &[DIM, DIM])?,
                f(&format!("{t}.linear_2.bias"), &[DIM])?,
            ),
            text_norm: f("txt_norm.weight", &[CONTEXT])?,
            text_in: (
                f("txt_in.weight", &[DIM, CONTEXT])?,
                f("txt_in.bias", &[DIM])?,
            ),
        };
        let pair = |w: &str, ws: &[usize], b: &str, bs: &[usize]| -> Result<_, &'static str> {
            Ok((
                Shared::floats(&gpu, &fast, &f(w, ws)?)?,
                Shared::floats(&gpu, &fast, &f(b, bs)?)?,
            ))
        };
        let img_in = pair("img_in.weight", &[DIM, CHANNELS], "img_in.bias", &[DIM])?;
        let proj_out = pair(
            "proj_out.weight",
            &[CHANNELS, DIM],
            "proj_out.bias",
            &[CHANNELS],
        )?;
        let final_modulation = Modulation::load(
            &gpu,
            &fast,
            &model.file,
            &[model.name("norm_out.linear.weight")],
            2 * DIM,
        )?;
        let mods = Shared::new(&gpu, &fast, ((layers.len() * BLOCK_MODS) * 4) as u64)?;
        let final_mods = Shared::new(&gpu, &fast, (2 * DIM * 4) as u64)?;
        let time = Shared::new(&gpu, &fast, (DIM * 4) as u64)?;
        let time_rotated = Shared::new(&gpu, &fast, (DIM * 4) as u64)?;
        let glue = Glue::new(&fast)?;
        Ok(Self {
            gpu,
            fast,
            glue,
            layers,
            small,
            img_in,
            proj_out,
            final_modulation,
            mods,
            final_mods,
            time,
            time_rotated,
            work: Mutex::new(HashMap::new()),
        })
    }

    /// Each timestep's `silu(time embedding)`.
    pub fn times(&self, timesteps: &[f32]) -> Vec<Times> {
        timesteps
            .iter()
            .map(|&t| {
                let (w1, b1) = &self.small.linear_1;
                let (w2, b2) = &self.small.linear_2;
                let mut h = linear(&time_embedding(t), 256, w1, b1);
                h.iter_mut().for_each(|v| *v = silu(*v));
                let mut h = linear(&h, DIM, w2, b2);
                h.iter_mut().for_each(|v| *v = silu(*v));
                Times { silu: h }
            })
            .collect()
    }

    /// The text encoder's rows `[tokens][CONTEXT]` of one prompt into the text stream's width.
    pub fn prompt(&self, context: &[f32]) -> Result<Prompt, &'static str> {
        if context.is_empty() || !context.len().is_multiple_of(CONTEXT) {
            return Err("invalid_input");
        }
        let mut x = context.to_vec();
        x.par_chunks_exact_mut(CONTEXT).for_each(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / CONTEXT as f32;
            let scale = 1.0 / (mean + 1e-6).sqrt();
            for (v, w) in row.iter_mut().zip(&self.small.text_norm) {
                *v *= scale * w;
            }
        });
        let (w, b) = &self.small.text_in;
        Ok(Prompt {
            rows: linear(&x, CONTEXT, w, b),
            tokens: context.len() / CONTEXT,
        })
    }

    /// The model's output for the latent `x` (`[16][2 * height][2 * width]`, `height` and
    /// `width` in tokens) under each prompt (a CFG pass each), all in one command list.
    /// `fast` scores attention with int8 queries and keys.
    pub fn step(
        &self,
        prompts: &[&Prompt],
        x: &[f32],
        height: usize,
        width: usize,
        times: &Times,
        fast: bool,
    ) -> Result<Vec<Vec<f32>>, &'static str> {
        use level_zero::Arg::{Memory as M, U32 as U};
        let gpu = &self.gpu;
        let g = &self.glue;
        let mut works = self.work.lock().map_err(|_| "gpu_failed")?;
        for prompt in prompts {
            let key = (height, width, prompt.tokens);
            if let std::collections::hash_map::Entry::Vacant(entry) = works.entry(key) {
                entry.insert(Work::new(gpu, &self.fast, height, width, prompt.tokens)?);
            }
        }
        let rows = patchify(x, height, width);
        gpu.upload(&self.time.buffer, 0, bytes_of(&times.silu))?;
        let mut rotated = times.silu.clone();
        for group in rotated.chunks_exact_mut(GROUP) {
            rotate(group.try_into().map_err(|_| "gpu_failed")?);
        }
        gpu.upload(&self.time_rotated.buffer, 0, bytes_of(&rotated))?;
        let mut batch = self.fast.batch()?;
        // This step's modulation rows, shared by the passes.
        let modulate = |batch: &mut level_zero::Batch,
                        m: &Modulation,
                        out: &Shared,
                        rows: usize,
                        first: usize| {
            let groups = [(rows / 16) as u32, 1, 1];
            match m {
                Modulation::Fp8 { weight, bias } => batch.launch(
                    &g.modulate,
                    &[
                        M(&weight.memory),
                        M(&bias.memory),
                        M(&self.time.memory),
                        M(&out.memory),
                        U(rows as u32),
                        U(first as u32),
                    ],
                    groups,
                ),
                Modulation::Int8 {
                    weight,
                    scales,
                    bias,
                } => batch.launch(
                    &g.modulate8,
                    &[
                        M(&weight.memory),
                        M(&scales.memory),
                        M(&bias.memory),
                        M(&self.time_rotated.memory),
                        M(&out.memory),
                        U(rows as u32),
                        U(first as u32),
                    ],
                    groups,
                ),
            }
        };
        for (index, layer) in self.layers.iter().enumerate() {
            modulate(
                &mut batch,
                &layer.modulation,
                &self.mods,
                BLOCK_MODS,
                index * BLOCK_MODS,
            )?;
        }
        modulate(
            &mut batch,
            &self.final_modulation,
            &self.final_mods,
            2 * DIM,
            0,
        )?;
        for prompt in prompts {
            let w = &works[&(height, width, prompt.tokens)];
            gpu.upload(&w.rows.buffer, 0, bytes_of(&rows))?;
            let text = w.streams[TEXT].start.as_ref().ok_or("gpu_failed")?;
            gpu.upload(&text.buffer, 0, bytes_of(&prompt.rows))?;
            self.pass(&mut batch, w, fast)?;
        }
        batch.run()?;
        let mut outputs = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let w = &works[&(height, width, prompt.tokens)];
            let mut result = vec![0f32; height * width * CHANNELS];
            gpu.read_floats(&w.out.buffer, 0, &w.readback, &mut result)?;
            outputs.push(unpatchify(&result, height, width));
        }
        Ok(outputs)
    }

    /// One pass's launches: embed, every block, the output head.
    fn pass(
        &self,
        batch: &mut level_zero::Batch,
        w: &Work,
        int8: bool,
    ) -> Result<(), &'static str> {
        use level_zero::Arg::{Memory as M, U32 as U};
        let g = &self.glue;
        let mods = &self.mods.memory;
        let [image, text] = &w.streams;
        batch.launch(
            &g.embed,
            &[
                M(&w.rows.memory),
                M(&self.img_in.0.memory),
                M(&self.img_in.1.memory),
                M(&image.x.memory),
                U(image.tokens as u32),
            ],
            [image.padded as u32, 1, 1],
        )?;
        let count = (text.padded * DIM / 4) as u32;
        let start = text.start.as_ref().ok_or("gpu_failed")?;
        batch.launch(
            &g.copy,
            &[M(&start.memory), M(&text.x.memory), U(count)],
            [count.div_ceil(256), 1, 1],
        )?;
        let base = |layer: usize, stream: usize| layer * BLOCK_MODS + stream * MOD_ROWS;
        for (s, work) in w.streams.iter().enumerate() {
            batch.launch(
                &g.norm,
                &[
                    M(&work.x.memory),
                    M(mods),
                    M(&work.bytes.memory),
                    M(&work.scales.memory),
                    U(work.tokens as u32),
                    U((base(0, s) + SHIFT1) as u32),
                ],
                [work.padded as u32, 1, 1],
            )?;
        }
        let layers = self.layers.len();
        for (index, layer) in self.layers.iter().enumerate() {
            let last = index + 1 == layers;
            let int8 = int8 && !layer.outlier;
            for (s, work) in w.streams.iter().enumerate() {
                let weights = &layer.streams[s];
                let v = &weights.vectors.memory;
                let (kernel, groups) = &work.products[QKV];
                batch.launch(
                    kernel,
                    &[
                        M(&work.bytes.memory),
                        M(&weights.matrices[QKV].memory),
                        M(&work.sums.memory),
                        M(&work.scales.memory),
                    ],
                    *groups,
                )?;
                for (side, first, tiles, to) in [
                    (0u32, work.slot, w.query_tiles, &w.query),
                    (1, work.key, w.key_tiles, &w.kv),
                ] {
                    batch.launch(
                        &g.qk,
                        &[
                            M(&work.sums.memory),
                            M(v),
                            M(&work.turns.memory),
                            M(&to.memory),
                            U(work.tokens as u32),
                            U(first as u32),
                            U(tiles),
                            U(side),
                            U(if side == 0 { QUERY_NORM } else { KEY_NORM } as u32),
                            U(QKV_BIAS as u32),
                            U(u32::from(int8)),
                        ],
                        [(HEADS / 4) as u32, work.tokens as u32, 1],
                    )?;
                }
                batch.launch(
                    &g.value,
                    &[
                        M(&work.sums.memory),
                        M(v),
                        M(&w.kv.memory),
                        U(work.tokens as u32),
                        U(work.key as u32),
                        U(w.key_tiles),
                        U(QKV_BIAS as u32),
                    ],
                    [(DIM / 4 / 256) as u32, work.tokens as u32, 1],
                )?;
            }
            let (attend, groups) = if int8 { &w.attend8 } else { &w.attend };
            batch.launch(
                attend,
                &[
                    M(&w.query.memory),
                    M(&w.kv.memory),
                    M(&w.mixed.memory),
                    M(&w.weights.memory),
                ],
                *groups,
            )?;
            for (s, work) in w.streams.iter().enumerate() {
                // After the last block only the image stream is read.
                if last && s == TEXT {
                    continue;
                }
                let weights = &layer.streams[s];
                let v = &weights.vectors.memory;
                let b = base(index, s);
                let product = |batch: &mut level_zero::Batch, which: usize, scaled: bool| {
                    let (kernel, groups) = &work.products[which];
                    if scaled {
                        batch.launch(
                            kernel,
                            &[
                                M(&work.bytes.memory),
                                M(&weights.matrices[which].memory),
                                M(&work.sums.memory),
                                M(&work.scales.memory),
                            ],
                            *groups,
                        )
                    } else {
                        batch.launch(
                            kernel,
                            &[
                                M(&work.bytes.memory),
                                M(&weights.matrices[which].memory),
                                M(&work.sums.memory),
                            ],
                            *groups,
                        )
                    }
                };
                batch.launch(
                    &g.merge,
                    &[
                        M(&w.mixed.memory),
                        M(&w.weights.memory),
                        M(&work.bytes.memory),
                        M(&work.scales.memory),
                        U(work.tokens as u32),
                        U(w.query_tiles),
                        U(work.slot as u32),
                    ],
                    [work.padded as u32, 1, 1],
                )?;
                product(batch, OUT, false)?;
                batch.launch(
                    &g.resnorm,
                    &[
                        M(&work.x.memory),
                        M(mods),
                        M(&work.bytes.memory),
                        M(&work.scales.memory),
                        M(&work.sums.memory),
                        M(v),
                        U(work.tokens as u32),
                        U((b + SHIFT2) as u32),
                        U(OUT_SCALE as u32),
                        U(OUT_BIAS as u32),
                        U((b + GATE1) as u32),
                    ],
                    [work.padded as u32, 1, 1],
                )?;
                product(batch, FC1, true)?;
                batch.launch(
                    &g.hidden,
                    &[
                        M(&work.sums.memory),
                        M(v),
                        M(&work.scales.memory),
                        M(&work.bytes.memory),
                        U(work.tokens as u32),
                        U(FC1_BIAS as u32),
                    ],
                    [work.padded as u32, 1, 1],
                )?;
                product(batch, FC2, false)?;
                if last {
                    batch.launch(
                        &g.resid,
                        &[
                            M(&work.x.memory),
                            M(&work.sums.memory),
                            M(&work.scales.memory),
                            M(v),
                            M(mods),
                            U(work.tokens as u32),
                            U(FC2_SCALE as u32),
                            U(FC2_BIAS as u32),
                            U((b + GATE2) as u32),
                        ],
                        [(DIM / 4 / 64) as u32, work.padded as u32, 1],
                    )?;
                } else {
                    batch.launch(
                        &g.resnorm,
                        &[
                            M(&work.x.memory),
                            M(mods),
                            M(&work.bytes.memory),
                            M(&work.scales.memory),
                            M(&work.sums.memory),
                            M(v),
                            U(work.tokens as u32),
                            U((base(index + 1, s) + SHIFT1) as u32),
                            U(FC2_SCALE as u32),
                            U(FC2_BIAS as u32),
                            U((b + GATE2) as u32),
                        ],
                        [work.padded as u32, 1, 1],
                    )?;
                }
            }
        }
        batch.launch(
            &g.final_,
            &[
                M(&image.x.memory),
                M(&self.final_mods.memory),
                M(&self.proj_out.0.memory),
                M(&self.proj_out.1.memory),
                M(&w.out.memory),
                U(image.tokens as u32),
            ],
            [image.tokens as u32, 1, 1],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{gpu::Select, qwen_dit::tests::db, qwen2512_dit::tests::Golden},
        *,
    };

    /// `KUNO_QWEN2512_BF16=<publisher's transformer dir> KUNO_QWEN2512_DIT=<ComfyUI FP8 file>
    /// KUNO_QWEN2512_DIT_INT8=<converted transformer dir> KUNO_QWEN2512_GOLDEN=<dir>`: the
    /// exact model (the publisher's BF16 weights in f32 on the CPU) on each captured call, and
    /// how far from it the C++ FP8 engine, the GPU from the FP8 file and the GPU from the INT8
    /// conversion land.
    #[test]
    #[ignore = "needs the Intel GPU, the models and captures"]
    fn paths_against_exact_model() {
        use super::super::qwen2512_dit::{Qwen2512Dit, Step};
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        let goldens: Vec<Golden> = (0..)
            .take_while(|s| dir.join(format!("step{s}.x.f32")).exists())
            .map(|s| Golden::load(&dir, s))
            .collect();
        let exact = Qwen2512Dit::open(Path::new(&std::env::var("KUNO_QWEN2512_BF16").unwrap())).unwrap();
        let truth: Vec<Vec<f32>> = goldens
            .iter()
            .map(|g| {
                exact
                    .forward(&Step {
                        x: &g.x,
                        height: g.height,
                        width: g.width,
                        timestep: g.timestep,
                        context: &g.context,
                    })
                    .unwrap()
            })
            .collect();
        drop(exact);
        for (step, (g, t)) in goldens.iter().zip(&truth).enumerate() {
            println!("step {step}: C++ FP8 engine {:.1} dB from exact", db(t, &g.out));
        }
        for (label, var) in [("FP8 file", "KUNO_QWEN2512_DIT"), ("INT8 conversion", "KUNO_QWEN2512_DIT_INT8")] {
            let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
            let model = Qwen2512Gpu::load(gpu, Path::new(&std::env::var(var).unwrap())).unwrap();
            for (step, (g, t)) in goldens.iter().zip(&truth).enumerate() {
                let prompt = model.prompt(&g.context).unwrap();
                let times = model.times(&[g.timestep]).remove(0);
                for fast in [false, true] {
                    let out = model
                        .step(&[&prompt], &g.x, g.height, g.width, &times, fast)
                        .unwrap()
                        .remove(0);
                    println!(
                        "step {step}: GPU from {label}{}: {:.1} dB from exact",
                        if fast { " (fast)" } else { "" },
                        db(t, &out)
                    );
                }
            }
        }
    }

    /// `KUNO_QWEN2512_DIT=<model> KUNO_QWEN2512_GOLDEN=<dir>`: the captured C++ calls (cond
    /// and uncond of each step), standard and fast attention.
    #[test]
    #[ignore = "needs the Intel GPU, the model and captures"]
    fn steps_match_cpp_engine() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let model = Qwen2512Gpu::load(gpu, Path::new(&std::env::var("KUNO_QWEN2512_DIT").unwrap()))
            .unwrap();
        println!("load {:.1} s", started.elapsed().as_secs_f64());
        let dir = std::path::PathBuf::from(std::env::var("KUNO_QWEN2512_GOLDEN").unwrap());
        for step in 0.. {
            if !dir.join(format!("step{step}.x.f32")).exists() {
                assert!(step > 0);
                break;
            }
            let golden = Golden::load(&dir, step);
            let prompt = model.prompt(&golden.context).unwrap();
            let times = model.times(&[golden.timestep]).remove(0);
            for fast in [false, true] {
                let started = std::time::Instant::now();
                let out = model
                    .step(
                        &[&prompt],
                        &golden.x,
                        golden.height,
                        golden.width,
                        &times,
                        fast,
                    )
                    .unwrap()
                    .remove(0);
                let db = db(&golden.out, &out);
                println!(
                    "step {step}{}: t {} {}x{} tokens, {} text, {:.3} s -> {db:.1} dB",
                    if fast { " fast" } else { "" },
                    golden.timestep,
                    golden.height,
                    golden.width,
                    prompt.tokens,
                    started.elapsed().as_secs_f64()
                );
                assert!(db > 25.0, "step {step}: {db:.1} dB");
            }
        }
    }
}
