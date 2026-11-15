//! MiniMax H3 diffusion transformer on the GPU (Vulkan). Built kernel by kernel; every piece
//! is checked against the CPU reference in `dit.rs` on identical inputs, because whole-network
//! outputs only agree loosely (see the engine document).
use std::{path::Path, sync::Mutex};

use rayon::prelude::*;

use super::{
    dit,
    gpu::{Buffer, Gpu, Kernel, Readback, Step, bytes_of, shaders},
    xe2::{
        attend, int8,
        level_zero::{self, LevelZero},
        zebin,
    },
};

mod refiner;

/// One unit of GPU work with its arguments owned, so a pass can be assembled as a list.
pub enum Call<'a> {
    /// A compute shader dispatch.
    Shader {
        kernel: &'a Kernel,
        buffers: Vec<&'a Buffer>,
        constants: Vec<u32>,
        groups: [u32; 3],
    },
    /// A machine-code kernel: an integer product (tokens, weights, sums) or attention
    /// (queries, keys and values, mixed values, weight sums).
    Product {
        fast: &'a LevelZero,
        kernel: &'a level_zero::Kernel,
        memory: Vec<&'a level_zero::Memory>,
        groups: [u32; 3],
    },
}
/// Runs the calls in order and waits for them. Consecutive shader dispatches share one
/// submission; a machine-code product runs between submissions.
pub fn run(gpu: &Gpu, calls: &[Call]) -> Result<(), &'static str> {
    let mut steps: Vec<Step> = Vec::new();
    for call in calls {
        match call {
            Call::Shader {
                kernel,
                buffers,
                constants,
                groups,
            } => steps.push(Step {
                kernel,
                buffers,
                constants,
                groups: *groups,
            }),
            Call::Product {
                fast,
                kernel,
                memory,
                groups,
            } => {
                if !steps.is_empty() {
                    gpu.pass(&steps)?;
                    steps.clear();
                }
                fast.launch(kernel, memory, *groups)?;
            }
        }
    }
    if steps.is_empty() {
        Ok(())
    } else {
        gpu.pass(&steps)
    }
}

const DIM: usize = dit::DIM;
const INNER: usize = dit::INNER;
const HIDDEN: usize = dit::HIDDEN;
const HEADS: usize = dit::HEADS;
const HEAD: usize = dit::HEAD;
/// Offsets (in values) into a layer's vector buffer.
const NORM1: usize = 0;
const NORM2: usize = NORM1 + DIM;
const QUERY_NORM: usize = NORM2 + DIM;
const KEY_NORM: usize = QUERY_NORM + HEAD;
const SCALE_QKV: usize = KEY_NORM + HEAD;
const SCALE_OUT: usize = SCALE_QKV + INNER * 3;
const SCALE_FC1: usize = SCALE_OUT + DIM;
const SCALE_FC2: usize = SCALE_FC1 + HIDDEN * 2;
/// Tokens are padded to the mixing kernel's query tile.
const PAD: usize = 256;
/// Token tile of the other matrix kernels.
const TILE: usize = 128;
/// Weight rows per dispatch of the integer product (see gemm8.comp).
const ROW_BLOCK: usize = 4096;
/// Heads per attention dispatch; bounds the score buffer.
const GROUP: usize = 8;

/// Kernels of one DiT evaluation.
pub struct Kernels {
    rotate8: Kernel,
    gemm8: Kernel,
    scoremax: Kernel,
    /// Block input norm and its quantization in one pass.
    normq8: Kernel,
    qk8: Kernel,
    value8: Kernel,
    rowmax: Kernel,
    mixtop: Kernel,
    merge8: Kernel,
    resid8: Kernel,
    /// The feed-forward activation and its quantization in one pass.
    gateq8: Kernel,
    /// Writers and reader of the machine-code attention kernel's layouts.
    qk8s: Kernel,
    value8s: Kernel,
    mergeq8: Kernel,
    /// The input projections, the blocks' modulation vectors and the output heads.
    embed8: Kernel,
    parts8: Kernel,
    final8: Kernel,
    /// Present when the integer products can run as our own machine code.
    fast: Option<LevelZero>,
}
/// The four integer products of a block: weight rows and columns.
const PRODUCTS: [(usize, usize); 4] = [
    (INNER * 3, DIM),
    (DIM, INNER),
    (HIDDEN * 2, DIM),
    (DIM, HIDDEN),
];
const QKV: usize = 0;
const OUT: usize = 1;
const FC1: usize = 2;
const FC2: usize = 3;
/// Builds the machine-code product for `tokens` padded tokens, with its thread groups.
fn fast_product(
    fast: &LevelZero,
    tokens: usize,
    rows: usize,
    columns: usize,
) -> Result<(level_zero::Kernel, [u32; 3]), &'static str> {
    let shape = int8::Shape::linear(tokens, rows, columns).ok_or("gpu_unsupported")?;
    Ok((fast_kernel(fast, &shape)?, shape.groups(rows)))
}
fn fast_kernel(fast: &LevelZero, shape: &int8::Shape) -> Result<level_zero::Kernel, &'static str> {
    let code = int8::kernel(shape);
    let native = zebin::write(&zebin::Kernel {
        name: "product",
        code: &code,
        grf: 256,
        pointers: 3,
    });
    fast.kernel(&native, "product", 3)
}
impl Kernels {
    /// Uses the machine-code integer product when this machine can run it (an Intel Xe2
    /// card with Level Zero installed), the shader otherwise.
    pub fn new(gpu: &Gpu) -> Result<Self, &'static str> {
        let fast = gpu
            .shares()
            .then(|| LevelZero::open().ok())
            .flatten()
            .filter(|fast| fast_product(fast, PAD, PRODUCTS[OUT].0, PRODUCTS[OUT].1).is_ok());
        Self::with(gpu, fast)
    }
    /// Shader kernels only.
    pub fn shaders_only(gpu: &Gpu) -> Result<Self, &'static str> {
        Self::with(gpu, None)
    }
    fn with(gpu: &Gpu, fast: Option<LevelZero>) -> Result<Self, &'static str> {
        Ok(Self {
            fast,
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 3, Some(16))?,
            gemm8: gpu.kernel(shaders::GEMM8, 3, 5, Some(16))?,
            scoremax: gpu.kernel(shaders::SCOREMAX, 4, 11, Some(16))?,
            normq8: gpu.kernel(shaders::NORMQ8, 6, 6, Some(16))?,
            qk8: gpu.kernel(shaders::QK8, 5, 5, Some(16))?,
            value8: gpu.kernel(shaders::VALUE8, 4, 3, None)?,
            rowmax: gpu.kernel(shaders::ROWMAX, 2, 4, None)?,
            mixtop: gpu.kernel(shaders::MIXTOP, 6, 13, Some(16))?,
            merge8: gpu.kernel(shaders::MERGE8, 3, 2, None)?,
            resid8: gpu.kernel(shaders::RESID8, 6, 4, None)?,
            gateq8: gpu.kernel(shaders::GATEQ8, 5, 3, Some(16))?,
            qk8s: gpu.kernel(shaders::QK8S, 5, 6, Some(16))?,
            value8s: gpu.kernel(shaders::VALUE8S, 4, 4, None)?,
            mergeq8: gpu.kernel(shaders::MERGEQ8, 4, 3, Some(16))?,
            embed8: gpu.kernel(shaders::EMBED8, 4, 1, None)?,
            parts8: gpu.kernel(shaders::PARTS8, 3, 1, None)?,
            final8: gpu.kernel(shaders::FINAL8, 5, 1, Some(16))?,
        })
    }
    /// Whether the integer products run as machine code.
    pub fn fast(&self) -> bool {
        self.fast.is_some()
    }
    /// Quantizes `rows` f32 rows of `width` values in `input` to INT8 rows in `bytes` and one
    /// scale per row in `scales`. `stream` selects the machine-code product's byte layout
    /// (`int8::token_word`) instead of plain rows.
    pub fn quantize<'a>(
        &'a self,
        input: &'a Buffer,
        bytes: &'a Buffer,
        scales: &'a Buffer,
        rows: usize,
        width: usize,
        stream: bool,
    ) -> Call<'a> {
        Call::Shader {
            kernel: &self.rotate8,
            buffers: vec![input, bytes, scales],
            constants: vec![
                width as u32,
                (width / 256).div_ceil(16) as u32,
                u32::from(stream),
            ],
            groups: [rows as u32, 1, 1],
        }
    }
    /// `sums[token][row] = bytes[token][..columns] . weight[row][..columns]` for the weight
    /// rows of product `which`: one machine-code launch, or the shader a block of rows per
    /// dispatch.
    fn product<'a>(&'a self, work: &'a Work, weight: &'a Matrix, which: usize) -> Vec<Call<'a>> {
        let (rows, columns) = PRODUCTS[which];
        if let (Some(fast), Some(shared), Some(stream)) = (&self.fast, &work.fast, &weight.stream) {
            let (kernel, groups) = &shared.products[which];
            return vec![Call::Product {
                fast,
                kernel,
                memory: vec![&shared.bytes, stream, &shared.sums],
                groups: *groups,
            }];
        }
        (0..rows)
            .step_by(ROW_BLOCK)
            .map(move |first| Call::Shader {
                kernel: &self.gemm8,
                buffers: vec![&work.bytes, &weight.buffer, &work.sums],
                constants: vec![
                    columns as u32,
                    0,
                    (first * columns) as u32,
                    first as u32,
                    rows as u32,
                ],
                groups: [
                    (ROW_BLOCK.min(rows - first) / 256) as u32,
                    (work.padded / TILE) as u32,
                    1,
                ],
            })
            .collect()
    }
}

/// An INT8 weight matrix on the GPU: plain rows for the shader, or the machine-code
/// product's stream layout together with its Level Zero mapping.
pub struct Matrix {
    buffer: Buffer,
    stream: Option<level_zero::Memory>,
}

/// One block's weights on the GPU.
pub struct Layer {
    qkv: Matrix,
    out: Matrix,
    fc1: Matrix,
    fc2: Matrix,
    /// Norm weights and the linear layers' row scales, at the offsets above.
    vectors: Buffer,
}
impl Layer {
    pub fn load(
        gpu: &Gpu,
        kernels: &Kernels,
        model: &dit::Dit,
        layer: usize,
    ) -> Result<Self, &'static str> {
        let f = &model.file;
        let name = |part: &str| format!("blocks.{layer}.{part}");
        let matrix = |part: &str, which: usize| -> Result<Matrix, &'static str> {
            let (rows, columns) = PRODUCTS[which];
            let name = name(&format!("{part}.weight"));
            let Some(fast) = &kernels.fast else {
                // Straight from the file into mapped GPU staging memory.
                let buffer = gpu.filled((rows * columns) as u64, |to| {
                    f.i8_into(&name, &[rows, columns], to)
                })?;
                return Ok(Matrix {
                    buffer,
                    stream: None,
                });
            };
            let bytes = (rows * columns + int8::SLACK) as u64;
            let buffer = gpu.filled_shared(bytes, |to| {
                let mut plain = vec![0u8; rows * columns];
                f.i8_into(&name, &[rows, columns], &mut plain)?;
                int8::weight_stream(&plain, rows, columns, to);
                Ok(())
            })?;
            let stream = fast.import(&gpu.export(&buffer)?, bytes)?;
            Ok(Matrix {
                buffer,
                stream: Some(stream),
            })
        };
        let mut vectors = Vec::with_capacity(SCALE_FC2 + DIM);
        vectors.extend(f.floats(&name("norm1.weight"), &[DIM])?);
        vectors.extend(f.floats(&name("norm2.weight"), &[DIM])?);
        vectors.extend(f.floats(&name("attn.q_norm.weight"), &[HEAD])?);
        vectors.extend(f.floats(&name("attn.k_norm.weight"), &[HEAD])?);
        vectors.extend(f.f32(&name("attn.qkv_proj.weight_scale"), &[INNER * 3, 1])?);
        vectors.extend(f.f32(&name("attn.out_proj.weight_scale"), &[DIM, 1])?);
        vectors.extend(f.f32(&name("mlp.fc1.weight_scale"), &[HIDDEN * 2, 1])?);
        vectors.extend(f.f32(&name("mlp.fc2.weight_scale"), &[DIM, 1])?);
        Ok(Self {
            qkv: matrix("attn.qkv_proj", QKV)?,
            out: matrix("attn.out_proj", OUT)?,
            fc1: matrix("mlp.fc1", FC1)?,
            fc2: matrix("mlp.fc2", FC2)?,
            vectors: gpu.filled(vectors.len() as u64 * 4, |to| {
                to.copy_from_slice(bytes_of(&vectors));
                Ok(())
            })?,
        })
    }
}

/// Buffers of one evaluation for a padded token count.
pub struct Work {
    pub tokens: usize,
    pub padded: usize,
    /// `[padded][DIM]` f32; padded rows are zero.
    pub stream: Buffer,
    /// Modulation row of each token.
    pub rows: Buffer,
    /// `[padded][48]` cosine and sine.
    pub turns: Buffer,
    /// `[timesteps][TIME_WIDTH]`.
    times: Buffer,
    /// Every block's modulation vectors `[blocks][rows][6][DIM]`.
    pub parts: Buffer,
    rows_count: usize,
    /// `[padded][INNER]` f32: the shader attention's merged rows, and the second half of the
    /// feed-forward activation's rotated rows.
    wide: Buffer,
    bytes: Buffer,
    scales: Buffer,
    sums: Buffer,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    scores: Buffer,
    /// Largest score per key tile, `[GROUP][padded][tiles rounded up to 4]`.
    best: Buffer,
    top: Buffer,
    weights: Buffer,
    mixed: Buffer,
    /// The input projections' rows `[padded][PATCH_VALUES]` and each token's projection.
    patches: Buffer,
    embeds: Buffer,
    /// The output heads' modulation `[timesteps][2][DIM]`, each token's head, and their
    /// outputs `[padded][PATCH_VALUES]`.
    final_parts: Buffer,
    kinds: Buffer,
    outputs: Buffer,
    /// For reading the outputs back.
    readback: Readback,
    /// The machine-code products' view of `bytes` and `sums`, and their kernels.
    fast: Option<Shared>,
}
struct Shared {
    bytes: level_zero::Memory,
    sums: level_zero::Memory,
    /// Kernel and thread groups of each of `PRODUCTS` at this token count.
    products: [(level_zero::Kernel, [u32; 3]); 4],
    /// Attention for this token count, and its view of `query`, `key` (keys and values),
    /// `mixed` and `weights`.
    attend: (level_zero::Kernel, [u32; 3]),
    streams: [level_zero::Memory; 4],
}
impl Work {
    /// Buffers (and, with machine code, kernels) for exactly `tokens` tokens.
    pub fn new(
        gpu: &Gpu,
        kernels: &Kernels,
        tokens: usize,
        rows: usize,
        blocks: usize,
    ) -> Result<Self, &'static str> {
        let padded = tokens.next_multiple_of(PAD);
        let f32s = |count: usize| gpu.buffer(count as u64 * 4);
        let bytes_size = (padded * HIDDEN + int8::SLACK) as u64;
        let sums_size = (padded * HIDDEN * 2) as u64 * 4;
        let half_rows = (HEADS * padded * HEAD) as u64 * 2;
        let plain = |bytes: u64| gpu.buffer(bytes);
        let (bytes, sums, attention, fast) = match &kernels.fast {
            None => (
                gpu.buffer(bytes_size)?,
                gpu.buffer(sums_size)?,
                [
                    plain(half_rows)?,
                    plain(half_rows)?,
                    plain(half_rows * 2)?,
                    plain((HEADS * padded * 8) as u64 * 4)?,
                ],
                None,
            ),
            Some(fast) => {
                let (bytes, sums) = (gpu.shared(bytes_size)?, gpu.shared(sums_size)?);
                let product =
                    |which: usize| fast_product(fast, padded, PRODUCTS[which].0, PRODUCTS[which].1);
                // Queries, keys with values, mixed values, weight sums in the attention
                // kernel's layouts.
                let sizes = [
                    half_rows,
                    half_rows * 2 + attend::SLACK as u64,
                    half_rows * 2,
                    (HEADS * padded) as u64 * 4,
                ];
                let attention = [
                    gpu.shared(sizes[0])?,
                    gpu.shared(sizes[1])?,
                    gpu.shared(sizes[2])?,
                    gpu.shared(sizes[3])?,
                ];
                let import = |i: usize| fast.import(&gpu.export(&attention[i])?, sizes[i]);
                let shape = attend::Shape::new(padded, tokens, HEADS, 1.0 / (HEAD as f32).sqrt())
                    .ok_or("gpu_unsupported")?;
                let native = zebin::write(&zebin::Kernel {
                    name: "attend",
                    code: &attend::kernel(&shape),
                    grf: 256,
                    pointers: 4,
                });
                let shared = Shared {
                    attend: (fast.kernel(&native, "attend", 4)?, shape.groups(HEADS)),
                    streams: [import(0)?, import(1)?, import(2)?, import(3)?],
                    bytes: fast.import(&gpu.export(&bytes)?, bytes_size)?,
                    sums: fast.import(&gpu.export(&sums)?, sums_size)?,
                    products: [product(QKV)?, product(OUT)?, product(FC1)?, product(FC2)?],
                };
                (bytes, sums, attention, Some(shared))
            }
        };
        // The score buffers are only used by the shader form of attention.
        let scores = if fast.is_some() {
            4
        } else {
            (GROUP * padded * padded) as u64 * 2
        };
        let [query, key, mixed, weights] = attention;
        Ok(Self {
            tokens,
            padded,
            stream: f32s(padded * DIM)?,
            rows: f32s(padded)?,
            turns: f32s(padded * dit::PAIRS * 2)?,
            times: f32s(rows / 3 * dit::TIME_WIDTH)?,
            parts: f32s(blocks * rows * 6 * DIM)?,
            rows_count: rows,
            wide: f32s(padded * INNER)?,
            bytes,
            scales: f32s(padded)?,
            sums,
            query,
            key,
            value: gpu.buffer((HEADS * HEAD * padded) as u64 * 2)?,
            scores: gpu.buffer(scores)?,
            best: f32s(GROUP * padded * (padded / TILE).next_multiple_of(4))?,
            top: f32s(HEADS * padded)?,
            weights,
            mixed,
            patches: f32s(padded * dit::PATCH_VALUES)?,
            embeds: f32s(padded)?,
            final_parts: f32s(rows / 3 * 2 * DIM)?,
            kinds: f32s(padded)?,
            outputs: f32s(padded * dit::PATCH_VALUES)?,
            readback: gpu.readback((padded * dit::PATCH_VALUES) as u64 * 4)?,
            fast,
        })
    }
}

impl Kernels {
    /// The dispatches of one block on `work.stream`; the block's modulation vectors are slot
    /// `slot` of `work.parts`.
    pub fn block<'a>(&'a self, layer: &'a Layer, work: &'a Work, slot: usize) -> Vec<Call<'a>> {
        let parts_first = (slot * work.rows_count * 6 * DIM) as u32;
        let (n, np) = (work.tokens as u32, work.padded as u32);
        let w = work;
        let v = &layer.vectors;
        let mut calls = Vec::new();
        let stream = u32::from(w.fast.is_some());
        // Normalized, modulated rows of the stream, quantized.
        let norm = |weight: usize, shift: u32, scale: u32| Call::Shader {
            kernel: &self.normq8,
            buffers: vec![&w.stream, v, &w.parts, &w.rows, &w.bytes, &w.scales],
            constants: vec![n, weight as u32, shift, scale, parts_first, stream],
            groups: [np, 1, 1],
        };
        let residual = |scale: usize, gate: u32| Call::Shader {
            kernel: &self.resid8,
            buffers: vec![&w.stream, &w.sums, &w.scales, v, &w.parts, &w.rows],
            constants: vec![n, scale as u32, gate, parts_first],
            groups: [(DIM / 4 / 64) as u32, np, 1],
        };

        calls.push(norm(NORM1, 0, 1));
        calls.extend(self.product(w, &layer.qkv, QKV));
        if let (Some(fast), Some(shared)) = (&self.fast, &w.fast) {
            // Attention as one machine-code kernel between shaders that write its
            // operands and gather and quantize its output.
            for (side, to, weight) in [(0, &w.query, QUERY_NORM), (1, &w.key, KEY_NORM)] {
                calls.push(Call::Shader {
                    kernel: &self.qk8s,
                    buffers: vec![&w.sums, &w.scales, v, &w.turns, to],
                    constants: vec![n, np, side, SCALE_QKV as u32, weight as u32, 0],
                    groups: [(HEADS / 4) as u32, np, 1],
                });
            }
            calls.push(Call::Shader {
                kernel: &self.value8s,
                buffers: vec![&w.sums, &w.scales, v, &w.key],
                constants: vec![n, np, SCALE_QKV as u32, 0],
                groups: [HEADS as u32, np / 32, 1],
            });
            let (kernel, groups) = &shared.attend;
            calls.push(Call::Product {
                fast,
                kernel,
                memory: shared.streams.iter().collect(),
                groups: *groups,
            });
            calls.push(Call::Shader {
                kernel: &self.mergeq8,
                buffers: vec![&w.mixed, &w.weights, &w.bytes, &w.scales],
                constants: vec![n, np, stream],
                groups: [np, 1, 1],
            });
        } else {
            self.attention(&mut calls, w, v);
            calls.push(self.quantize(&w.wide, &w.bytes, &w.scales, w.padded, INNER, false));
        }
        calls.extend(self.product(w, &layer.out, OUT));
        calls.push(residual(SCALE_OUT, 2));

        calls.push(norm(NORM2, 3, 4));
        calls.extend(self.product(w, &layer.fc1, FC1));
        calls.push(Call::Shader {
            kernel: &self.gateq8,
            buffers: vec![&w.sums, &w.scales, v, &w.wide, &w.bytes],
            constants: vec![n, SCALE_FC1 as u32, stream],
            groups: [np, 1, 1],
        });
        calls.extend(self.product(w, &layer.fc2, FC2));
        calls.push(residual(SCALE_FC2, 5));
        calls
    }
    /// Attention as shaders: scores, row maxima and mixing, a group of heads at a time.
    fn attention<'a>(&'a self, calls: &mut Vec<Call<'a>>, w: &'a Work, v: &'a Buffer) {
        let (n, np) = (w.tokens as u32, w.padded as u32);
        for (side, to, weight) in [(0, &w.query, QUERY_NORM), (1, &w.key, KEY_NORM)] {
            calls.push(Call::Shader {
                kernel: &self.qk8,
                buffers: vec![&w.sums, &w.scales, v, &w.turns, to],
                constants: vec![n, np, side, SCALE_QKV as u32, weight as u32],
                groups: [(HEADS / 4) as u32, np, 1],
            });
        }
        calls.push(Call::Shader {
            kernel: &self.value8,
            buffers: vec![&w.sums, &w.scales, v, &w.value],
            constants: vec![n, np, SCALE_QKV as u32],
            groups: [HEADS as u32, np / 32, 1],
        });
        let tiles = np / TILE as u32;
        let (head_width, group) = (HEAD as u32, GROUP as u32);
        let best_stride = tiles.next_multiple_of(4);
        for head in (0..HEADS as u32).step_by(GROUP) {
            calls.push(Call::Shader {
                kernel: &self.scoremax,
                buffers: vec![&w.query, &w.key, &w.scores, &w.best],
                constants: vec![
                    head_width,
                    head * np * head_width,
                    np * head_width,
                    head * np * head_width,
                    np * head_width,
                    0,
                    np * np,
                    np,
                    n,
                    best_stride,
                    np * best_stride,
                ],
                groups: [tiles, tiles, group],
            });
            calls.push(Call::Shader {
                kernel: &self.rowmax,
                buffers: vec![&w.best, &w.top],
                constants: vec![tiles, best_stride, np, head * np],
                groups: [np / 64, 1, group],
            });
            calls.push(Call::Shader {
                kernel: &self.mixtop,
                buffers: vec![&w.scores, &w.value, &w.mixed, &w.top, &w.weights, &w.best],
                constants: vec![
                    np,
                    0,
                    np * np,
                    head * head_width * np,
                    head_width * np,
                    head * np * head_width,
                    np * head_width,
                    head_width,
                    n,
                    head * np,
                    np,
                    (1.0 / (HEAD as f32).sqrt()).to_bits(),
                    best_stride,
                ],
                groups: [1, np / PAD as u32, group],
            });
        }
        calls.push(Call::Shader {
            kernel: &self.merge8,
            buffers: vec![&w.mixed, &w.weights, &w.wide],
            constants: vec![n, np],
            groups: [HEADS as u32, np, 1],
        });
    }
}

/// The whole DiT with its weights on the GPU.
pub struct GpuDit {
    gpu: Gpu,
    model: dit::Dit,
    kernels: Kernels,
    layers: Vec<Layer>,
    /// Every block's modulation projection: weight `[18 * DIM][TIME_WIDTH]`, then bias.
    modulation: Buffer,
    /// The text refiner, when its attention can run as machine code; the CPU's otherwise.
    refiner: Option<refiner::Refiner>,
    /// The input projections' weights and biases (`dit::Dit::embed_vectors`).
    embed: Buffer,
    /// The output heads' norm weight, weights and biases (`dit::Dit::head_vectors`).
    head: Buffer,
    work: Mutex<Option<Work>>,
}
impl GpuDit {
    /// Reads all blocks (about 19 GB) into video memory.
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        let model = dit::Dit::open(path)?;
        let kernels = Kernels::new(gpu)?;
        // A few layers at a time: reading is the slow part, and each matrix passes through
        // host staging memory of its own size.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .map_err(|_| "gpu_failed")?;
        // The text refiner's weights load beside the blocks'.
        let (layers, refiner) = std::thread::scope(|scope| {
            let refiner = scope.spawn(|| {
                kernels
                    .fast()
                    .then(|| refiner::Refiner::load(gpu, &model))
                    .transpose()
            });
            let layers: Result<Vec<Layer>, &'static str> = pool.install(|| {
                (0..model.layers)
                    .into_par_iter()
                    .map(|layer| Layer::load(gpu, &kernels, &model, layer))
                    .collect()
            });
            (layers, refiner.join().unwrap_or(Err("gpu_failed")))
        });
        let refiner = refiner?;
        let each = 18 * DIM * (dit::TIME_WIDTH + 1) * 4;
        let modulation = gpu.filled((model.layers * each) as u64, |to| {
            for (layer, to) in to.chunks_exact_mut(each).enumerate() {
                let (weight, bias) = model.modulation(layer)?;
                let (weights, biases) = to.split_at_mut(weight.len() * 4);
                weights.copy_from_slice(bytes_of(&weight));
                biases.copy_from_slice(bytes_of(&bias));
            }
            Ok(())
        })?;
        let head = model.head_vectors()?;
        let head = gpu.filled(head.len() as u64 * 4, |to| {
            to.copy_from_slice(bytes_of(&head));
            Ok(())
        })?;
        let embed = model.embed_vectors()?;
        let embed = gpu.filled(embed.len() as u64 * 4, |to| {
            to.copy_from_slice(bytes_of(&embed));
            Ok(())
        })?;
        Ok(Self {
            gpu: gpu.clone(),
            refiner,
            embed,
            head,
            model,
            kernels,
            layers: layers?,
            modulation,
            work: Mutex::new(None),
        })
    }
    /// Evaluates the network once; returns the packed velocity with `step.packed`'s layout.
    pub fn forward(&self, step: &dit::Step) -> Result<Vec<f32>, &'static str> {
        let inputs = self.inputs(step)?;
        self.blocks(step, inputs)
    }
    /// The model's inputs; a new text is refined on the GPU when that is possible.
    fn inputs(&self, step: &dit::Step) -> Result<dit::Inputs, &'static str> {
        self.model
            .inputs_with(step, &|context| match &self.refiner {
                Some(refiner) => refiner.refine(&self.gpu, &self.kernels, &self.model, context),
                None => self.model.refine(context),
            })
    }
    fn blocks(&self, step: &dit::Step, inputs: dit::Inputs) -> Result<Vec<f32>, &'static str> {
        self.through(step, inputs, &self.layers)
    }
    /// Input projections, `layers` (the first blocks) and the output heads.
    fn through(
        &self,
        step: &dit::Step,
        inputs: dit::Inputs,
        layers: &[Layer],
    ) -> Result<Vec<f32>, &'static str> {
        let gpu = &self.gpu;
        let dit::Inputs {
            rows: sources,
            layout,
            times,
            rotary,
        } = inputs;
        let tokens = layout.positions.len();
        let rows = layout.timesteps.len() * 3;
        let mut cached = self.work.lock().map_err(|_| "gpu_failed")?;
        if cached
            .as_ref()
            .is_none_or(|w| w.tokens != tokens || w.rows_count != rows)
        {
            // Drop the old set before allocating the new one.
            *cached = None;
            *cached = Some(Work::new(
                gpu,
                &self.kernels,
                tokens,
                rows,
                self.layers.len(),
            )?);
        }
        let work = cached.as_mut().ok_or("gpu_failed")?;
        work.tokens = tokens;
        let mut token_rows = vec![0u32; work.padded];
        for span in &layout.spans {
            token_rows[span.start..span.end].fill(span.row as u32);
        }
        let mut turns: Vec<f32> = rotary.iter().flatten().copied().collect();
        turns.resize(work.padded * dit::PAIRS * 2, 0.0);
        // The stream: text rows as they are; the projection kernel writes every other live
        // row and zeros the padded ones.
        let width = dit::PATCH_VALUES;
        let mut patches = vec![0f32; work.padded * width];
        let mut embeds = vec![0u32; work.padded];
        for (&(first, end, _), source) in layout.order.iter().zip(&sources) {
            match source {
                dit::Rows::Text(text) => {
                    gpu.upload(&work.stream, (first * DIM) as u64 * 4, bytes_of(text))?
                }
                dit::Rows::Video(values) => {
                    patches[first * width..end * width].copy_from_slice(values);
                    embeds[first..end].fill(1);
                }
                dit::Rows::Audio(values) => {
                    for (to, from) in patches[first * width..end * width]
                        .chunks_exact_mut(width)
                        .zip(values.chunks_exact(dit::AUDIO_CHANNELS))
                    {
                        to[..dit::AUDIO_CHANNELS].copy_from_slice(from);
                    }
                    embeds[first..end].fill(2);
                }
            }
        }
        drop(sources);
        gpu.upload(&work.patches, 0, bytes_of(&patches))?;
        gpu.upload(&work.embeds, 0, bytes_of(&embeds))?;
        gpu.upload(&work.rows, 0, bytes_of(&token_rows))?;
        gpu.upload(&work.turns, 0, bytes_of(&turns))?;
        gpu.upload(&work.times, 0, bytes_of(&times))?;
        let mut kinds = vec![0u32; work.padded];
        for (span, kind) in [(layout.video, 1), (layout.audio, 2)] {
            kinds[span.start..span.end].fill(span.row as u32 * 4 + kind);
        }
        gpu.upload(&work.kinds, 0, bytes_of(&kinds))?;
        let final_parts = self.model.final_parts(&times)?;
        gpu.upload(&work.final_parts, 0, bytes_of(&final_parts))?;
        let work = &*work;
        let timesteps = layout.timesteps.len() as u32;
        run(
            gpu,
            &[
                Call::Shader {
                    kernel: &self.kernels.embed8,
                    buffers: vec![&work.patches, &self.embed, &work.embeds, &work.stream],
                    constants: vec![tokens as u32],
                    groups: [(DIM / 4 / 64) as u32, work.padded as u32, 1],
                },
                Call::Shader {
                    kernel: &self.kernels.parts8,
                    buffers: vec![&work.times, &self.modulation, &work.parts],
                    constants: vec![timesteps],
                    groups: [
                        (18 * DIM / 4 / 64) as u32,
                        timesteps,
                        self.layers.len() as u32,
                    ],
                },
            ],
        )?;
        // At most one submission per block: the kernel driver ends jobs that run for several
        // seconds.
        for (slot, layer) in layers.iter().enumerate() {
            run(gpu, &self.kernels.block(layer, work, slot))?;
        }
        run(
            gpu,
            &[Call::Shader {
                kernel: &self.kernels.final8,
                buffers: vec![
                    &work.stream,
                    &self.head,
                    &work.final_parts,
                    &work.kinds,
                    &work.outputs,
                ],
                constants: vec![tokens as u32],
                groups: [work.padded as u32, 1, 1],
            }],
        )?;
        let mut outputs = vec![0f32; tokens * dit::PATCH_VALUES];
        gpu.read_floats(&work.outputs, 0, &work.readback, &mut outputs)?;
        let head = |span: dit::Span, width: usize| -> Vec<f32> {
            outputs[span.start * dit::PATCH_VALUES..span.end * dit::PATCH_VALUES]
                .chunks_exact(dit::PATCH_VALUES)
                .flat_map(|row| &row[..width])
                .copied()
                .collect()
        };
        Ok(dit::Dit::pack(
            step,
            &head(layout.video, dit::PATCH_VALUES),
            &head(layout.audio, dit::AUDIO_CHANNELS),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::gpu::{Select, floats},
        *,
    };

    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "needs the Intel GPU"]
    fn quantizer_matches_reference_bit_for_bit() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::shaders_only(&gpu).unwrap();
        for width in [5376usize, 7168, 14336] {
            let rows = 37;
            let mut x = noise(rows * width, width as u64);
            // Uneven magnitudes, a zero row and exact rounding ties.
            for (i, v) in x.iter_mut().enumerate() {
                *v *= 1.0 + (i % 97) as f32;
            }
            x[3 * width..4 * width].fill(0.0);
            for (i, v) in x[5 * width..6 * width].iter_mut().enumerate() {
                *v = (i % 5) as f32 * 0.5;
            }
            let (want, want_scales) = dit::quantize(&x, width);
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            gpu.upload(&input, 0, bytes_of(&x)).unwrap();
            run(
                &gpu,
                &[kernels.quantize(&input, &bytes, &scales, rows, width, false)],
            )
            .unwrap();
            let got = gpu.download(&bytes, 0, rows * width).unwrap();
            let got_scales = floats(&gpu.download(&scales, 0, rows * 4).unwrap());
            assert_eq!(got_scales, want_scales, "width {width}");
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| **a as i8 != **b)
                .count();
            assert_eq!(differ, 0, "width {width}");
            // The stream layout holds the same bytes, 16 at a time, at other positions.
            let streamed = gpu
                .buffer((rows.next_multiple_of(32) * width) as u64)
                .unwrap();
            run(
                &gpu,
                &[kernels.quantize(&input, &streamed, &scales, rows, width, true)],
            )
            .unwrap();
            let stream = gpu
                .download(&streamed, 0, rows.next_multiple_of(32) * width)
                .unwrap();
            for row in 0..rows {
                for word in 0..width / 16 {
                    let at = int8::token_word(row, word, width) * 16;
                    let from = row * width + word * 16;
                    assert_eq!(
                        stream[at..at + 16],
                        got[from..from + 16],
                        "row {row} word {word}"
                    );
                }
            }

            let rows = 8704;
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            let call = [kernels.quantize(&input, &bytes, &scales, rows, width, false)];
            run(&gpu, &call).unwrap();
            let started = std::time::Instant::now();
            for _ in 0..4 {
                run(&gpu, &call).unwrap();
            }
            println!(
                "{rows} rows of {width}: {:.2} ms",
                started.elapsed().as_secs_f64() * 250.0
            );
        }
    }

    /// The machine-code product against the shader on the same operands: integer sums, so
    /// every one must be equal. Also run in small dispatch blocks with a partial last block.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn machine_code_product_matches_shader() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let kernels = Kernels::shaders_only(&gpu).unwrap();
        for (tokens, rows, columns) in [(768usize, 1024usize, DIM), (8704, 4096, HIDDEN)] {
            let mut state = 77u32;
            let mut random = |count: usize| -> Vec<u8> {
                (0..count)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (state >> 24) as u8
                    })
                    .collect()
            };
            let (bytes, weight) = (random(tokens * columns), random(rows * columns));
            let sums_size = (tokens * rows * 4) as u64;
            let (plain_bytes, plain_weight, plain_sums) = (
                gpu.buffer(bytes.len() as u64).unwrap(),
                gpu.buffer(weight.len() as u64).unwrap(),
                gpu.buffer(sums_size).unwrap(),
            );
            gpu.upload(&plain_bytes, 0, &bytes).unwrap();
            gpu.upload(&plain_weight, 0, &weight).unwrap();
            let shader = [Call::Shader {
                kernel: &kernels.gemm8,
                buffers: vec![&plain_bytes, &plain_weight, &plain_sums],
                constants: vec![columns as u32, 0, 0, 0, rows as u32],
                groups: [(rows / 256) as u32, (tokens / TILE) as u32, 1],
            }];
            run(&gpu, &shader).unwrap();
            let started = std::time::Instant::now();
            run(&gpu, &shader).unwrap();
            let shader_ms = started.elapsed().as_secs_f64() * 1e3;
            let want = gpu.download(&plain_sums, 0, sums_size as usize).unwrap();

            let mut streamed = vec![0u8; bytes.len() + int8::SLACK];
            for token in 0..tokens {
                for word in 0..columns / 16 {
                    let at = int8::token_word(token, word, columns) * 16;
                    let from = token * columns + word * 16;
                    streamed[at..at + 16].copy_from_slice(&bytes[from..from + 16]);
                }
            }
            let mut weights = vec![0u8; weight.len() + int8::SLACK];
            int8::weight_stream(&weight, rows, columns, &mut weights);
            let (stream_bytes, stream_weight, sums) = (
                gpu.shared(streamed.len() as u64).unwrap(),
                gpu.shared(weights.len() as u64).unwrap(),
                gpu.shared(sums_size).unwrap(),
            );
            gpu.upload(&stream_bytes, 0, &streamed).unwrap();
            gpu.upload(&stream_weight, 0, &weights).unwrap();
            let import = |buffer: &Buffer| {
                fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                    .unwrap()
            };
            let memory = (import(&stream_bytes), import(&stream_weight), import(&sums));
            let whole = int8::Shape::linear(tokens, rows, columns).unwrap();
            for shape in [whole, int8::Shape { xc: 5, ..whole }] {
                gpu.upload(&sums, 0, &vec![0u8; sums_size as usize])
                    .unwrap();
                let kernel = fast_kernel(&fast, &shape).unwrap();
                let call = [Call::Product {
                    fast: &fast,
                    kernel: &kernel,
                    memory: vec![&memory.0, &memory.1, &memory.2],
                    groups: shape.groups(rows),
                }];
                run(&gpu, &call).unwrap();
                let started = std::time::Instant::now();
                run(&gpu, &call).unwrap();
                let ms = started.elapsed().as_secs_f64() * 1e3;
                let got = gpu.download(&sums, 0, sums_size as usize).unwrap();
                let differ = got
                    .chunks_exact(4)
                    .zip(want.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                let tops = 2.0 * (tokens * rows * columns) as f64 / 1e9;
                println!(
                    "{tokens} x {rows} x {columns}, blocks of {} tiles: {ms:.2} ms ({:.0} TOPS), shader {shader_ms:.2} ms ({:.0} TOPS)",
                    shape.xc,
                    tops / ms,
                    tops / shader_ms
                );
                assert_eq!(differ, 0);
            }
        }
    }

    /// One block on the GPU against the reference on the same input, from a captured step.
    #[test]
    #[ignore = "needs the Intel GPU, the model and a captured step"]
    fn block_matches_reference() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let model = dit::Dit::open(std::path::Path::new(&model)).unwrap();
        let prepared = model.prepare(&captured.step()).unwrap();
        let tokens = prepared.layout.positions.len();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        let rows = prepared.layout.timesteps.len() * 3;
        let work = Work::new(&gpu, &kernels, tokens, rows, 1).unwrap();
        let mut token_rows = vec![0u32; work.padded];
        for span in &prepared.layout.spans {
            token_rows[span.start..span.end].fill(span.row as u32);
        }
        let mut turns: Vec<f32> = prepared.rotary.iter().flatten().copied().collect();
        turns.resize(work.padded * dit::PAIRS * 2, 0.0);
        gpu.upload(&work.rows, 0, bytes_of(&token_rows)).unwrap();
        gpu.upload(&work.turns, 0, bytes_of(&turns)).unwrap();
        for index in [0usize, 20, 49] {
            let layer = Layer::load(&gpu, &kernels, &model, index).unwrap();
            let parts = model.parts(index, &prepared.times).unwrap();
            let mut want = prepared.x.clone();
            model
                .block(
                    index,
                    &mut want,
                    &parts,
                    &prepared.layout.spans,
                    &prepared.rotary,
                )
                .unwrap();
            let mut start = prepared.x.clone();
            start.resize(work.padded * DIM, 0.0);
            gpu.upload(&work.stream, 0, bytes_of(&start)).unwrap();
            gpu.upload(&work.parts, 0, bytes_of(&parts)).unwrap();
            let calls = kernels.block(&layer, &work, 0);
            let started = std::time::Instant::now();
            run(&gpu, &calls).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let got = floats(&gpu.download(&work.stream, 0, tokens * DIM * 4).unwrap());
            let (mut error, mut change) = (0f64, 0f64);
            for ((got, want), before) in got.iter().zip(&want).zip(&prepared.x) {
                error += f64::from(got - want).powi(2);
                change += f64::from(want - before).powi(2);
            }
            let db = 10.0 * (change / error.max(1e-30)).log10();
            println!(
                "block {index}: {db:.1} dB from the reference's update, {tokens} tokens, {:.1} ms",
                seconds * 1e3
            );
            assert!(db > 30.0);
        }
    }

    /// Time of one block at a real clip's size (2.3 s at 832x544 is 8611 tokens).
    #[test]
    #[ignore = "needs the Intel GPU and the model"]
    fn block_speed_at_clip_size() {
        let model =
            dit::Dit::open(std::path::Path::new(&std::env::var("KUNO_DIT").unwrap())).unwrap();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        let (tokens, rows) = (8611, 12);
        let work = Work::new(&gpu, &kernels, tokens, rows, 1).unwrap();
        let mut start = noise(tokens * DIM, 5);
        start.resize(work.padded * DIM, 0.0);
        gpu.upload(&work.stream, 0, bytes_of(&start)).unwrap();
        gpu.upload(&work.rows, 0, bytes_of(&vec![0u32; work.padded]))
            .unwrap();
        let mut turns = vec![[1f32, 0.0]; work.padded * dit::PAIRS];
        for (i, turn) in turns.iter_mut().enumerate() {
            let angle = i as f32 * 0.01;
            *turn = [angle.cos(), angle.sin()];
        }
        gpu.upload(&work.turns, 0, bytes_of(&turns)).unwrap();
        gpu.upload(&work.parts, 0, bytes_of(&noise(rows * 6 * DIM, 6)))
            .unwrap();
        let layer = Layer::load(&gpu, &kernels, &model, 10).unwrap();
        let calls = kernels.block(&layer, &work, 0);
        run(&gpu, &calls).unwrap();
        let names = [
            (&kernels.rotate8, "quantize"),
            (&kernels.gemm8, "linear"),
            (&kernels.scoremax, "scores"),
            (&kernels.normq8, "norm and quantize"),
            (&kernels.qk8, "queries and keys"),
            (&kernels.value8, "values"),
            (&kernels.rowmax, "row maximum"),
            (&kernels.mixtop, "mixing"),
            (&kernels.merge8, "merge"),
            (&kernels.resid8, "residual"),
            (&kernels.gateq8, "gate and quantize"),
            (&kernels.qk8s, "queries and keys"),
            (&kernels.value8s, "values"),
            (&kernels.mergeq8, "merge and quantize"),
        ];
        let mut totals = std::collections::BTreeMap::new();
        for call in &calls {
            let started = std::time::Instant::now();
            run(&gpu, std::slice::from_ref(call)).unwrap();
            let name = match call {
                Call::Shader { kernel, .. } => {
                    names.iter().find(|n| std::ptr::eq(n.0, *kernel)).unwrap().1
                }
                Call::Product { memory, .. } if memory.len() == 3 => "linear (machine code)",
                Call::Product { .. } => "attention (machine code)",
            };
            *totals.entry(name).or_insert(0.0) += started.elapsed().as_secs_f64();
        }
        for (name, seconds) in &totals {
            println!("{:8.1} ms  {name}", seconds * 1e3);
        }
        let started = std::time::Instant::now();
        run(&gpu, &calls).unwrap();
        println!(
            "one block, {tokens} tokens: {:.1} ms in one submission",
            started.elapsed().as_secs_f64() * 1e3
        );
    }

    /// The whole network on the GPU against a step captured from the C++ engine.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, the model and a captured step"]
    fn step_matches_cpp_engine() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let network = GpuDit::load(&gpu, std::path::Path::new(&model)).unwrap();
        println!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        let step = captured.step();
        let mut got = Vec::new();
        for _ in 0..2 {
            let started = std::time::Instant::now();
            let inputs = network.inputs(&step).unwrap();
            let ready = started.elapsed().as_secs_f64();
            got = network.blocks(&step, inputs).unwrap();
            println!(
                "inputs and refiner {ready:.2} s, projections, blocks and heads {:.2} s",
                started.elapsed().as_secs_f64() - ready
            );
        }
        let plane = captured.shape[0] * captured.shape[1] * captured.shape[2];
        let report = |name: &str, range: std::ops::Range<usize>| {
            let (mut error, mut power) = (0f64, 0f64);
            for i in range {
                error += f64::from(got[i] - captured.out[i]).powi(2);
                power += f64::from(captured.out[i]).powi(2);
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!("{name}: {db:.1} dB from the C++ engine");
            db
        };
        let video = report("video", 0..plane * dit::VIDEO_CHANNELS);
        let audio = report(
            "audio",
            plane * dit::VIDEO_CHANNELS
                ..plane * dit::VIDEO_CHANNELS + captured.audio_length * 2 * dit::AUDIO_CHANNELS,
        );
        // Whole-network agreement is loose by nature (see dit.rs) and looser with more tokens:
        // 29 and 24 dB on a 317-token step, 24 and 18 dB on an 8611-token one whose single
        // blocks agree with the reference to 40 dB and better. These bounds catch structure.
        assert!(video > 20.0 && audio > 15.0);
    }

    /// The text refiner on the GPU against the CPU reference's on a captured step's text.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, Level Zero, the model and a captured step"]
    fn refiner_matches_reference() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let model = dit::Dit::open(std::path::Path::new(&model)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        let started = std::time::Instant::now();
        let refiner = refiner::Refiner::load(&gpu, &model).unwrap();
        println!("loaded in {:.2} s", started.elapsed().as_secs_f64());
        let context = captured.step().context;
        let mut got = Vec::new();
        for _ in 0..2 {
            let started = std::time::Instant::now();
            got = refiner.refine(&gpu, &kernels, &model, context).unwrap();
            println!(
                "{} tokens refined in {:.3} s",
                got.len() / DIM,
                started.elapsed().as_secs_f64()
            );
        }
        let started = std::time::Instant::now();
        let want = model.refine(context).unwrap();
        println!("on the CPU: {:.2} s", started.elapsed().as_secs_f64());
        let (mut error, mut power) = (0f64, 0f64);
        for (got, want) in got.iter().zip(&want) {
            error += f64::from(got - want).powi(2);
            power += f64::from(*want).powi(2);
        }
        let db = 10.0 * (power / error.max(1e-30)).log10();
        println!("{db:.1} dB from the reference");
        assert!(got.len() == want.len() && db > 40.0);
    }

    /// The input projections and output heads alone (no blocks) against the CPU reference's
    /// on a captured step, then with the first block in between.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, the model and a captured step"]
    fn projections_and_heads_match_reference() {
        let (Ok(model), Ok(golden)) = (std::env::var("KUNO_DIT"), std::env::var("KUNO_DIT_GOLDEN"))
        else {
            panic!("set KUNO_DIT and KUNO_DIT_GOLDEN (a captured step's path prefix)");
        };
        let captured = dit::tests::Captured::read(&golden);
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let network = GpuDit::load(&gpu, std::path::Path::new(&model)).unwrap();
        let step = captured.step();
        for blocks in [0, 1] {
            let mut prepared = network.model.prepare(&step).unwrap();
            if blocks == 1 {
                let parts = network.model.parts(0, &prepared.times).unwrap();
                network
                    .model
                    .block(
                        0,
                        &mut prepared.x,
                        &parts,
                        &prepared.layout.spans,
                        &prepared.rotary,
                    )
                    .unwrap();
            }
            let want = network
                .model
                .finish(&step, &prepared.x, &prepared.layout, &prepared.times)
                .unwrap();
            let inputs = network.model.inputs(&step).unwrap();
            let got = network
                .through(&step, inputs, &network.layers[..blocks])
                .unwrap();
            let (mut error, mut power) = (0f64, 0f64);
            for (got, want) in got.iter().zip(&want) {
                error += f64::from(got - want).powi(2);
                power += f64::from(*want).powi(2);
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!("{blocks} blocks: {db:.1} dB from the reference");
            assert!(got.len() == want.len() && db > if blocks == 0 { 90.0 } else { 35.0 });
        }
    }

    /// Replays a captured clip's sampler steps: every step from the C++ engine's input, and
    /// the whole Euler loop from the first noise on this DiT's own outputs. Optionally
    /// decodes both final latents and compares frames.
    #[test]
    #[ignore = "needs the Intel GPU with 25 GB free, the model and a captured clip"]
    fn sampling_follows_cpp_engine() {
        let model = std::env::var("KUNO_DIT").unwrap();
        let dir = std::env::var("KUNO_DIT_STEPS").unwrap();
        let steps: Vec<_> = (0..)
            .map(|i| format!("{dir}/step{i}"))
            .take_while(|prefix| std::path::Path::new(&format!("{prefix}.txt")).exists())
            .map(|prefix| dit::tests::Captured::read(&prefix))
            .collect();
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let network = GpuDit::load(&gpu, std::path::Path::new(&model)).unwrap();
        let decibels = |got: &[f32], want: &[f32]| {
            let (mut error, mut power) = (0f64, 0f64);
            for (got, want) in got.iter().zip(want) {
                error += f64::from(got - want).powi(2);
                power += f64::from(*want).powi(2);
            }
            10.0 * (power / error.max(1e-30)).log10()
        };
        let sigma = |i: usize| steps.get(i).map_or(0.0, |s| s.timestep / 1000.0);
        let euler = |x: &[f32], out: &[f32], i: usize| -> Vec<f32> {
            let change = sigma(i + 1) - sigma(i);
            x.iter().zip(out).map(|(x, v)| x + change * v).collect()
        };
        let mut own = steps[0].x.clone();
        for (i, captured) in steps.iter().enumerate() {
            let out = network.forward(&captured.step()).unwrap();
            let mut step = captured.step();
            step.packed = &own;
            let next = euler(&own, &network.forward(&step).unwrap(), i);
            println!(
                "step {i} (sigma {:.3}): output {:.1} dB from the C++ engine; own loop's latent {:.1} dB",
                sigma(i),
                decibels(&out, &captured.out),
                decibels(&own, &captured.x),
            );
            own = next;
        }
        let last = steps.len() - 1;
        let theirs = euler(&steps[last].x, &steps[last].out, last);
        let shape = &steps[0].shape;
        let video = shape[0] * shape[1] * shape[2] * dit::VIDEO_CHANNELS;
        println!(
            "final video latent: {:.1} dB from the C++ engine's",
            decibels(&own[..video], &theirs[..video])
        );
        let Ok(vae) = std::env::var("KUNO_VIDEO_VAE") else {
            return;
        };
        drop(network);
        let decoder =
            super::super::video_gpu::GpuVideoDecoder::load(&gpu, std::path::Path::new(&vae))
                .unwrap();
        let decode = |latent: &[f32]| {
            decoder
                .decode(
                    &dit::decoder_latent(&latent[..video]),
                    shape[0],
                    shape[1],
                    shape[2],
                )
                .unwrap()
        };
        let (mine, theirs) = (decode(&own), decode(&theirs));
        let mut error = 0f64;
        for (a, b) in mine.iter().zip(&theirs) {
            error += f64::from(a.clamp(0.0, 1.0) - b.clamp(0.0, 1.0)).powi(2);
        }
        println!(
            "decoded frames: {:.1} dB PSNR between the two",
            10.0 * (mine.len() as f64 / error.max(1e-30)).log10()
        );
        if let Ok(out) = std::env::var("KUNO_DIT_FRAMES_OUT") {
            let (width, height) = (shape[0] * 16, shape[1] * 16);
            let frames = mine.len() / (3 * width * height);
            for (name, pixels) in [("rust", &mine), ("cpp", &theirs)] {
                for frame in [0, frames / 2, frames - 1] {
                    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
                    for at in 0..width * height {
                        for c in 0..3 {
                            let v = pixels[(c * frames + frame) * width * height + at];
                            ppm.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                        }
                    }
                    std::fs::write(format!("{out}/{name}-{frame}.ppm"), ppm).unwrap();
                }
            }
        }
    }
}
