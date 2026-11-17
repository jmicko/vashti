//! MiniMax H3 video decoder on the GPU. Same model as `video_vae` (the CPU reference), run
//! with the Vulkan kernels in `gpu`: half-precision operands, f32 accumulation, and the
//! residual stream kept in f32. On an Intel Xe2 card with Level Zero the heavy kernels run
//! as the engine's own machine code instead: the linear layers (`xe2::int8`, half-float
//! form), attention (`xe2::attend`, using half of its 128-channel heads) with its operand
//! split and merge (`xe2::heads`, `xe2::merge`), the residual updates with the norms
//! (`xe2::norm`) and the gated activation (`xe2::gate`). A tile is then one Vulkan
//! submission, for the final norm: a machine-code launch costs about 15 microseconds
//! against 200 for a submission.
use std::path::Path;

use rayon::prelude::*;

use super::{
    dit_gpu::{Call, fast_kernel, run},
    gpu::{Buffer, Gpu, Kernel, Readback, Staged, bytes_of, shaders},
    safetensors::Safetensors,
    video_vae::{
        DIM, EXTRA, HEAD, HEADS, INNER, LATENT_CHANNELS, LAYERS, PAIRS, PIXELS, REGISTERS, linear,
        rotary, unpatchify_biased,
    },
    xe2::{
        self, attend, gate, heads, int8,
        level_zero::{self, LevelZero},
        merge, norm,
    },
};

/// Offsets of one layer's small vectors inside its f32 vector buffer.
const NORM1: usize = 0;
const QKV_BIAS: usize = NORM1 + DIM;
const OUT_BIAS: usize = QKV_BIAS + DIM * 3;
const SCALE1: usize = OUT_BIAS + DIM;
const NORM2: usize = SCALE1 + DIM;
const W1_BIAS: usize = NORM2 + DIM;
const W2_BIAS: usize = W1_BIAS + INNER * 2;
const SCALE2: usize = W2_BIAS + DIM;
const VECTORS: usize = SCALE2 + DIM;
/// For the machine-code kernels: scale times bias of both residual updates, and the next
/// layer's first norm weight.
const SHIFT1: usize = VECTORS;
const SHIFT2: usize = SHIFT1 + DIM;
const NEXT: usize = SHIFT2 + DIM;
const ALL: usize = NEXT + DIM;
/// The matrix kernel works on 128-token tiles.
const TILE: usize = 128;
/// Attention heads processed per dispatch.
const GROUP: usize = 8;

/// The linear layers: weight rows and columns.
const PRODUCTS: [(usize, usize); 5] = [
    (DIM * 3, DIM),
    (DIM, DIM),
    (INNER * 2, DIM),
    (DIM, INNER),
    (PIXELS, DIM),
];
const QKV: usize = 0;
const OUT: usize = 1;
const W1: usize = 2;
const W2: usize = 3;
const PROJECT: usize = 4;

/// A half-float weight matrix on the GPU: plain rows for the shader, or the machine-code
/// product's stream layout together with its Level Zero mapping.
struct Matrix {
    buffer: Buffer,
    stream: Option<level_zero::Memory>,
}
struct Layer {
    qkv: Matrix,
    out: Matrix,
    w1: Matrix,
    w2: Matrix,
    vectors: Matrix,
}
struct Kernels {
    /// Present when the linear layers can run as our own machine code.
    fast: Option<LevelZero>,
    gemm: Kernel,
    gemm16: Kernel,
    mix: Kernel,
    norm: Kernel,
    heads: Kernel,
    merge: Kernel,
    residual: Kernel,
    gate: Kernel,
}
/// Scratch buffers for one tile shape, kept between tiles.
struct Work {
    /// Latent width, height and frames.
    shape: (usize, usize, usize),
    /// The machine-code kernels for this token count and the buffers as they see them.
    fast: Option<Fast>,
    stream: Buffer,
    /// Rotary turns of the shape's positions.
    turns: Buffer,
    half: Buffer,
    wide: Buffer,
    /// Attention operands of the shader path.
    scratch: Option<Scratch>,
    /// Host-side landing areas for the stream going in and the pixels coming out.
    staging: Readback,
    readback: Readback,
}
struct Scratch {
    query: Buffer,
    key: Buffer,
    value: Buffer,
    scores: Buffer,
    mixed: Buffer,
}
struct Fast {
    stream: level_zero::Memory,
    half: level_zero::Memory,
    wide: level_zero::Memory,
    products: [(level_zero::Kernel, [u32; 3]); 5],
    /// The first layer's input norm.
    enter: level_zero::Kernel,
    /// The residual update after attention with the feed-forward's input norm.
    middle: level_zero::Kernel,
    /// The update after the feed-forward with the next layer's input norm, and without a
    /// norm for the last layer.
    next: level_zero::Kernel,
    leave: level_zero::Kernel,
    gate: level_zero::Kernel,
    /// Attention: queries, keys and values (with the rotary table behind them), mixed
    /// values and weight sums; the kernels that write the first two and read the last two.
    split: (level_zero::Kernel, [u32; 3]),
    merge: level_zero::Kernel,
    attend: (level_zero::Kernel, [u32; 3]),
    streams: [level_zero::Memory; 4],
    _attention: [Buffer; 4],
}
/// A residual-and-norm kernel over every token row.
fn rows<'a>(
    machine: &'a LevelZero,
    shared: &'a Fast,
    kernel: &'a level_zero::Kernel,
    vectors: &'a level_zero::Memory,
    padded: usize,
) -> Call<'a> {
    Call::Product {
        fast: machine,
        kernel,
        memory: vec![&shared.stream, &shared.wide, vectors, &shared.half],
        groups: norm::Shape::groups(padded),
    }
}
fn product(
    fast: &LevelZero,
    tokens: usize,
    which: usize,
) -> Result<(level_zero::Kernel, [u32; 3]), &'static str> {
    let (rows, columns) = PRODUCTS[which];
    let shape = int8::Shape::half(tokens, rows, columns).ok_or("gpu_unsupported")?;
    Ok((fast_kernel(fast, &shape)?, shape.groups(rows)))
}
/// One layer's weights in host memory.
struct LayerHost {
    qkv: Staged,
    out: Staged,
    w1: Staged,
    w2: Staged,
    vectors: Staged,
}
/// The decoder's weights in host memory (about 5 GB) with its compiled kernels: everything
/// that can outlive a clip without holding video memory.
pub struct Host {
    gpu: Gpu,
    kernels: Kernels,
    layers: Vec<LayerHost>,
    /// Final norm weight and bias.
    tail: Staged,
    project: Staged,
    /// CPU-side ends of the model: latent to stream, and the last bias.
    post_quant: (Vec<f32>, Vec<f32>),
    embed: (Vec<f32>, Vec<f32>),
    registers: Vec<f32>,
    project_bias: Vec<f32>,
}
impl Host {
    pub fn read(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        Self::read_with(gpu, path, true)
    }
    /// `machine`: use machine-code kernels when this card can run them.
    fn read_with(gpu: &Gpu, path: &Path, machine: bool) -> Result<Self, &'static str> {
        let file = Safetensors::open(path)?;
        file.metadata("minimax_h3_video_vae")
            .ok_or("model_file_invalid")?;
        let fast = (machine && gpu.shares())
            .then(|| LevelZero::open().ok())
            .flatten()
            .filter(|fast| product(fast, TILE, OUT).is_ok());
        let streamed = fast.is_some();
        let matrix = |name: &str, which: usize| {
            let (rows, columns) = PRODUCTS[which];
            if !streamed {
                // Straight from the file into mapped staging memory.
                return gpu.staged((rows * columns) as u64 * 2, |to| {
                    file.f16_into(name, &[rows, columns], to)
                });
            }
            let mut plain = vec![0u8; rows * columns * 2];
            file.f16_into(name, &[rows, columns], &mut plain)?;
            gpu.staged((plain.len() + int8::SLACK) as u64, |to| {
                int8::weight_stream(&plain, rows, columns * 2, to);
                Ok(())
            })
        };
        let vectors = |values: &[f32]| {
            gpu.staged(values.len() as u64 * 4, |to| {
                to.copy_from_slice(bytes_of(values));
                Ok(())
            })
        };
        // Layers load in parallel: reading is the slow part and the file is in page cache.
        let layers: Result<Vec<LayerHost>, &'static str> = (0..LAYERS)
            .into_par_iter()
            .map(|layer| {
                let name = |part: &str| format!("decoder.transformer_blocks.{layer}.{part}");
                let mut small = Vec::with_capacity(ALL);
                for (part, len) in [
                    ("norm1.weight", DIM),
                    ("attn.to_qkv.bias", DIM * 3),
                    ("attn.to_out.bias", DIM),
                    ("scale1", DIM),
                    ("norm2.weight", DIM),
                    ("ff.w1.bias", INNER * 2),
                    ("ff.w2.bias", DIM),
                    ("scale2", DIM),
                ] {
                    small.extend(file.f16(&name(part), &[len])?);
                }
                for (scale, bias) in [(SCALE1, OUT_BIAS), (SCALE2, W2_BIAS)] {
                    let shift: Vec<f32> = (0..DIM)
                        .map(|i| small[scale + i] * small[bias + i])
                        .collect();
                    small.extend(shift);
                }
                if layer + 1 < LAYERS {
                    let next = format!("decoder.transformer_blocks.{}.norm1.weight", layer + 1);
                    small.extend(file.f16(&next, &[DIM])?);
                }
                small.resize(ALL, 0.0);
                Ok(LayerHost {
                    qkv: matrix(&name("attn.to_qkv.weight"), QKV)?,
                    out: matrix(&name("attn.to_out.weight"), OUT)?,
                    w1: matrix(&name("ff.w1.weight"), W1)?,
                    w2: matrix(&name("ff.w2.weight"), W2)?,
                    vectors: vectors(&small)?,
                })
            })
            .collect();
        let layers = layers?;
        let mut tail = file.f16("decoder.norm_out.weight", &[DIM])?;
        tail.extend(file.f16("decoder.norm_out.bias", &[DIM])?);
        let kernel = |spirv, bindings, constants| gpu.kernel(spirv, bindings, constants, None);
        Ok(Self {
            kernels: Kernels {
                gemm: gpu.kernel(shaders::GEMM, 3, 8, Some(16))?,
                gemm16: gpu.kernel(shaders::GEMM16, 3, 5, Some(16))?,
                mix: gpu.kernel(shaders::MIX, 3, 9, Some(16))?,
                norm: kernel(shaders::NORM, 3, 5)?,
                heads: kernel(shaders::HEADS, 6, 3)?,
                merge: kernel(shaders::MERGE, 2, 3)?,
                residual: kernel(shaders::RESIDUAL, 3, 3)?,
                gate: kernel(shaders::GATE, 3, 3)?,
                fast,
            },
            layers,
            tail: vectors(&tail)?,
            project: matrix("decoder.proj_out.weight", PROJECT)?,
            post_quant: (
                file.f16("post_quant_conv.weight", &[24, 24, 1, 1, 1])?,
                file.f16("post_quant_conv.bias", &[24])?,
            ),
            embed: (
                file.f16("decoder.x_embedder.weight", &[DIM, 24])?,
                file.f16("decoder.x_embedder.bias", &[DIM])?,
            ),
            registers: file.f16("decoder.register_tokens", &[1, REGISTERS, DIM])?,
            project_bias: file.f16("decoder.proj_out.bias", &[PIXELS])?,
            gpu: gpu.clone(),
        })
    }
}
pub struct GpuVideoDecoder {
    work: std::sync::Mutex<Option<Work>>,
    host: std::sync::Arc<Host>,
    layers: Vec<Layer>,
    tail: Buffer,
    project: Matrix,
}
impl GpuVideoDecoder {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        Self::place(std::sync::Arc::new(Host::read(gpu, path)?))
    }
    /// `load` with every kernel a shader, as on a card without the machine-code path.
    #[cfg(test)]
    fn load_shaders(gpu: &Gpu, path: &Path) -> Result<Self, &'static str> {
        Self::place(std::sync::Arc::new(Host::read_with(gpu, path, false)?))
    }
    /// Copies the weights into video memory.
    pub fn place(host: std::sync::Arc<Host>) -> Result<Self, &'static str> {
        let gpu = &host.gpu;
        let matrix = |staged: &Staged| -> Result<Matrix, &'static str> {
            let Some(fast) = &host.kernels.fast else {
                return Ok(Matrix {
                    buffer: gpu.place(staged, false)?,
                    stream: None,
                });
            };
            let buffer = gpu.place(staged, true)?;
            let stream = fast.import(&gpu.export(&buffer)?, staged.bytes())?;
            Ok(Matrix {
                buffer,
                stream: Some(stream),
            })
        };
        let layers: Result<Vec<Layer>, &'static str> = host
            .layers
            .iter()
            .map(|layer| {
                Ok(Layer {
                    qkv: matrix(&layer.qkv)?,
                    out: matrix(&layer.out)?,
                    w1: matrix(&layer.w1)?,
                    w2: matrix(&layer.w2)?,
                    vectors: matrix(&layer.vectors)?,
                })
            })
            .collect();
        Ok(Self {
            layers: layers?,
            tail: gpu.place(&host.tail, false)?,
            project: matrix(&host.project)?,
            work: std::sync::Mutex::new(None),
            host,
        })
    }
    /// Decodes a whole clip; see `video_tiles::decode_clip` for the layouts.
    pub fn decode(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
        frames: usize,
    ) -> Result<Vec<f32>, &'static str> {
        super::video_tiles::decode_clip(latent, width, height, frames, |tile, w, h, t| {
            self.decode_tile(tile, w, h, t)
        })
    }
    /// Same contract as `VideoDecoder::decode_tile`.
    pub fn decode_tile(
        &self,
        latent: &[f32],
        width: usize,
        height: usize,
        frames: usize,
    ) -> Result<Vec<f32>, &'static str> {
        let positions = width * height * frames;
        if positions == 0 || latent.len() != positions * LATENT_CHANNELS {
            return Err("video_latent_invalid");
        }
        let tokens = positions + EXTRA;
        let padded = tokens.div_ceil(TILE) * TILE;
        let mut start = vec![0f32; positions * LATENT_CHANNELS];
        for (channel, plane) in latent.chunks_exact(positions).enumerate() {
            for (position, value) in plane.iter().enumerate() {
                start[position * LATENT_CHANNELS + channel] = *value;
            }
        }
        let start = linear(
            &start,
            LATENT_CHANNELS,
            &self.host.post_quant.0,
            &self.host.post_quant.1,
        );
        let mut start = linear(
            &start,
            LATENT_CHANNELS,
            &self.host.embed.0,
            &self.host.embed.1,
        );
        start.extend(&self.host.registers);

        let gpu = &self.host.gpu;
        let k = &self.host.kernels;
        let (n, np) = (tokens as u32, padded as u32);
        let mut cached = self.work.lock().map_err(|_| "gpu_failed")?;
        let shape = (width, height, frames);
        if cached.as_ref().is_none_or(|work| work.shape != shape) {
            // Drop the old set before allocating the new one.
            *cached = None;
            let stream_bytes = (padded * DIM) as u64 * 4;
            let half_bytes = (padded * INNER * 2 + int8::SLACK) as u64;
            let wide_bytes = ((padded * INNER * 2) * 4 + gate::SLACK) as u64;
            let (stream, half, wide) = match &k.fast {
                Some(_) => (
                    gpu.shared(stream_bytes)?,
                    gpu.shared(half_bytes)?,
                    gpu.shared(wide_bytes)?,
                ),
                None => (
                    gpu.buffer(stream_bytes)?,
                    gpu.buffer(half_bytes)?,
                    gpu.buffer(wide_bytes)?,
                ),
            };
            let fast = match &k.fast {
                Some(fast) => {
                    let glue = |update: Option<(usize, usize)>, weight: Option<usize>| {
                        let shape = norm::Shape {
                            tokens: n,
                            update: update.map(|(scale, shift)| (scale as u32, shift as u32)),
                            norm: weight.map(|weight| weight as u32),
                        };
                        xe2::load(fast, &norm::kernel(&shape), 4)
                    };
                    let tiles = HEADS * padded / attend::TILE;
                    let table = heads::rotary_table(&rotary(width, height, frames), padded);
                    let table_first = tiles * attend::CHUNK_BYTES + attend::SLACK;
                    let sizes = [
                        tiles * attend::TILE_BYTES,
                        table_first + table.len() * 4,
                        tiles * attend::OUT_BYTES,
                        tiles * attend::SUM_BYTES,
                    ]
                    .map(|bytes| bytes as u64);
                    let attention = [
                        gpu.shared(sizes[0])?,
                        gpu.shared(sizes[1])?,
                        gpu.shared(sizes[2])?,
                        gpu.shared(sizes[3])?,
                    ];
                    gpu.upload(&attention[1], table_first as u64, bytes_of(&table))?;
                    let import = |i: usize| fast.import(&gpu.export(&attention[i])?, sizes[i]);
                    let split = heads::Shape {
                        padded: np,
                        bias_first: QKV_BIAS as u32,
                        rotary_first: u32::try_from(table_first).map_err(|_| "gpu_unsupported")?,
                    };
                    // Queries and keys have unit RMS, so scores over eight stay within eight.
                    let shape = attend::Shape {
                        bounded: true,
                        narrow: true,
                        ..attend::Shape::new(padded, tokens, HEADS, (HEAD as f32).sqrt().recip())
                            .ok_or("gpu_unsupported")?
                    };
                    Some(Fast {
                        split: (
                            xe2::load(fast, &heads::kernel(&split), 4)?,
                            split.groups(HEADS),
                        ),
                        merge: xe2::load(fast, &merge::kernel(np, HEADS as u32), 3)?,
                        attend: (
                            xe2::load(fast, &attend::kernel(&shape), 4)?,
                            shape.groups(HEADS),
                        ),
                        streams: [import(0)?, import(1)?, import(2)?, import(3)?],
                        _attention: attention,
                        stream: fast.import(&gpu.export(&stream)?, stream_bytes)?,
                        half: fast.import(&gpu.export(&half)?, half_bytes)?,
                        wide: fast.import(&gpu.export(&wide)?, wide_bytes)?,
                        products: [
                            product(fast, padded, QKV)?,
                            product(fast, padded, OUT)?,
                            product(fast, padded, W1)?,
                            product(fast, padded, W2)?,
                            product(fast, padded, PROJECT)?,
                        ],
                        enter: glue(None, Some(NORM1))?,
                        middle: glue(Some((SCALE1, SHIFT1)), Some(NORM2))?,
                        next: glue(Some((SCALE2, SHIFT2)), Some(NEXT))?,
                        leave: glue(Some((SCALE2, SHIFT2)), None)?,
                        gate: xe2::load(fast, &gate::kernel(W1_BIAS as u32), 3)?,
                    })
                }
                None => None,
            };
            let scratch = match &fast {
                Some(_) => None,
                None => Some(Scratch {
                    query: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
                    key: gpu.buffer((HEADS * padded * HEAD) as u64 * 2)?,
                    // Rows 65..128 of each head are never written and stay zero.
                    value: gpu.buffer((HEADS * TILE * padded) as u64 * 2)?,
                    scores: gpu.buffer((GROUP * padded * padded) as u64 * 4)?,
                    mixed: gpu.buffer((HEADS * padded * TILE) as u64 * 4)?,
                }),
            };
            let turns = gpu.buffer((padded * PAIRS) as u64 * 8)?;
            gpu.upload(&turns, 0, bytes_of(&rotary(width, height, frames)))?;
            *cached = Some(Work {
                fast,
                shape,
                stream,
                turns,
                half,
                wide,
                scratch,
                staging: gpu.readback(stream_bytes)?,
                readback: gpu.readback((padded * PIXELS) as u64 * 4)?,
            });
        }
        let Some(Work {
            fast,
            stream,
            turns,
            half,
            wide,
            scratch,
            staging,
            readback,
            ..
        }) = cached.as_ref()
        else {
            return Err("gpu_failed");
        };
        // The last live token starts as zeros. Padded rows of the stream are zero from the
        // start and no kernel writes them; every other buffer is rewritten in full.
        start.resize(tokens * DIM, 0.0);
        gpu.write_floats(stream, 0, staging, &start)?;
        let token_tiles = np / TILE as u32;
        let stream_layout = u32::from(fast.is_some());
        let shader = |kernel, buffers, constants, groups| Call::Shader {
            kernel,
            buffers,
            constants,
            groups,
        };
        // wide[token][row] = half[token][..columns] . weight[row][..columns]: one
        // machine-code launch, or the shader a few thousand weight rows per dispatch (the
        // weight block then stays cached while every token tile passes over it).
        let block = 4096;
        fn tied<'a, F: Fn(&'a Matrix, usize) -> Vec<Call<'a>>>(f: F) -> F {
            f
        }
        let gemm = tied(|weight, which| {
            let (rows, columns) = PRODUCTS[which];
            if let (Some(machine), Some(shared), Some(weights)) = (&k.fast, fast, &weight.stream) {
                let (kernel, groups) = &shared.products[which];
                return vec![Call::Product {
                    fast: machine,
                    kernel,
                    memory: vec![&shared.half, weights, &shared.wide],
                    groups: *groups,
                }];
            }
            (0..rows)
                .step_by(block)
                .map(|first| {
                    shader(
                        &k.gemm16,
                        vec![half, &weight.buffer, wide],
                        vec![
                            columns as u32,
                            0,
                            (first * columns) as u32,
                            first as u32,
                            rows as u32,
                        ],
                        [(block.min(rows - first) / 256) as u32, token_tiles, 1],
                    )
                })
                .collect()
        });
        let mut steps: Vec<Call<'_>> = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let v = &layer.vectors.buffer;
            // With machine code, the previous layer's last kernel has written this input.
            let machine = match (&k.fast, fast, &layer.vectors.stream) {
                (Some(machine), Some(shared), Some(vectors)) => Some((machine, shared, vectors)),
                _ => None,
            };
            match machine {
                Some((machine, shared, vectors)) if index == 0 => {
                    steps.push(rows(machine, shared, &shared.enter, vectors, padded));
                }
                Some(_) => {}
                None => steps.push(shader(
                    &k.norm,
                    vec![stream, v, half],
                    vec![n, NORM1 as u32, 0, 0, stream_layout],
                    [np, 1, 1],
                )),
            }
            steps.extend(gemm(&layer.qkv, QKV));
            match (machine, scratch) {
                (Some((machine, shared, vectors)), _) => {
                    let [query, key, mixed, sums] = &shared.streams;
                    steps.push(Call::Product {
                        fast: machine,
                        kernel: &shared.split.0,
                        memory: vec![&shared.wide, vectors, query, key],
                        groups: shared.split.1,
                    });
                    steps.push(Call::Product {
                        fast: machine,
                        kernel: &shared.attend.0,
                        memory: shared.streams.iter().collect(),
                        groups: shared.attend.1,
                    });
                    steps.push(Call::Product {
                        fast: machine,
                        kernel: &shared.merge,
                        memory: vec![mixed, sums, &shared.half],
                        groups: merge::groups(padded, HEADS),
                    });
                }
                (
                    None,
                    Some(Scratch {
                        query,
                        key,
                        value,
                        scores,
                        mixed,
                    }),
                ) => {
                    steps.push(shader(
                        &k.heads,
                        vec![wide, v, turns, query, key, value],
                        vec![n, np, QKV_BIAS as u32],
                        [1, np, 1],
                    ));
                    // A few heads at a time keeps the scores small (118 MB instead of 472 MB for all
                    // heads) at the same speed. The weights exist only inside `mix`.
                    for head in (0..HEADS as u32).step_by(GROUP) {
                        steps.push(shader(
                            &k.gemm,
                            vec![query, key, scores],
                            vec![
                                HEAD as u32,
                                head * np * HEAD as u32,
                                np * HEAD as u32,
                                head * np * HEAD as u32,
                                np * HEAD as u32,
                                0,
                                np * np,
                                np,
                            ],
                            [token_tiles, token_tiles, GROUP as u32],
                        ));
                        steps.push(shader(
                            &k.mix,
                            vec![scores, value, mixed],
                            vec![
                                np,
                                0,
                                np * np,
                                head * TILE as u32 * np,
                                TILE as u32 * np,
                                head * np * TILE as u32,
                                np * TILE as u32,
                                TILE as u32,
                                n,
                            ],
                            [1, token_tiles, GROUP as u32],
                        ));
                    }
                    steps.push(shader(
                        &k.merge,
                        vec![mixed, half],
                        vec![n, np, stream_layout],
                        [1, np, 1],
                    ));
                }
                (None, None) => return Err("gpu_failed"),
            }
            steps.extend(gemm(&layer.out, OUT));
            let Some((machine, shared, vectors)) = machine else {
                steps.push(shader(
                    &k.residual,
                    vec![stream, wide, v],
                    vec![n, OUT_BIAS as u32, SCALE1 as u32],
                    [(DIM / 4 / 64) as u32, np, 1],
                ));
                steps.push(shader(
                    &k.norm,
                    vec![stream, v, half],
                    vec![n, NORM2 as u32, 0, 0, stream_layout],
                    [np, 1, 1],
                ));
                steps.extend(gemm(&layer.w1, W1));
                steps.push(shader(
                    &k.gate,
                    vec![wide, v, half],
                    vec![n, W1_BIAS as u32, stream_layout],
                    [(INNER / 4 / 64) as u32, np, 1],
                ));
                steps.extend(gemm(&layer.w2, W2));
                steps.push(shader(
                    &k.residual,
                    vec![stream, wide, v],
                    vec![n, W2_BIAS as u32, SCALE2 as u32],
                    [(DIM / 4 / 64) as u32, np, 1],
                ));
                continue;
            };
            steps.push(rows(machine, shared, &shared.middle, vectors, padded));
            steps.extend(gemm(&layer.w1, W1));
            steps.push(Call::Product {
                fast: machine,
                kernel: &shared.gate,
                memory: vec![&shared.wide, vectors, &shared.half],
                groups: gate::groups(padded),
            });
            steps.extend(gemm(&layer.w2, W2));
            let after = if index + 1 < self.layers.len() {
                &shared.next
            } else {
                &shared.leave
            };
            steps.push(rows(machine, shared, after, vectors, padded));
        }
        steps.push(shader(
            &k.norm,
            vec![stream, &self.tail, half],
            vec![n, 0, DIM as u32, 1, stream_layout],
            [np, 1, 1],
        ));
        steps.extend(gemm(&self.project, PROJECT));
        if std::env::var_os("KUNO_GPU_PROFILE").is_some() {
            // Development aid: one submission per call, summed per kernel.
            let names = [
                (&k.gemm, "gemm"),
                (&k.gemm16, "gemm16"),
                (&k.norm, "norm"),
                (&k.heads, "heads"),
                (&k.mix, "mix"),
                (&k.merge, "merge"),
                (&k.residual, "residual"),
                (&k.gate, "gate"),
            ];
            let mut totals = std::collections::BTreeMap::new();
            for step in &steps {
                let label = match step {
                    Call::Shader {
                        kernel,
                        constants,
                        groups,
                        ..
                    } => {
                        let name = names
                            .iter()
                            .find(|n| std::ptr::eq(n.0, *kernel))
                            .map_or("?", |n| n.1);
                        format!("{name} x{groups:?} c{}", constants[0])
                    }
                    Call::Product { groups, .. } => format!("machine code x{groups:?}"),
                };
                let started = std::time::Instant::now();
                run(gpu, std::slice::from_ref(step))?;
                *totals.entry(label).or_insert(0.0) += started.elapsed().as_secs_f64();
            }
            for (label, seconds) in totals {
                eprintln!("{:8.1} ms  {label}", seconds * 1e3);
            }
        } else {
            run(gpu, &steps)?;
        }
        let pixels = positions * PIXELS;
        gpu.floats_with(wide, 0, readback, pixels, |patches| {
            unpatchify_biased(
                patches,
                Some(&self.host.project_bias),
                width,
                height,
                frames,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::gpu::{Select, floats},
        *,
    };

    fn read(path: &Path) -> Vec<f32> {
        floats(&std::fs::read(path).unwrap())
    }
    fn decibels(want: &[f32], got: &[f32]) -> f64 {
        assert_eq!(want.len(), got.len());
        let (mut signal, mut error) = (0f64, 0f64);
        for (w, g) in want.iter().zip(got) {
            signal += f64::from(*w).powi(2);
            error += (f64::from(*w) - f64::from(*g)).powi(2);
        }
        10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10()
    }
    /// The machine-code half-float product against the shader on the same random operands,
    /// at the decoder's two largest shapes; the second round makes every weight subnormal.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn machine_code_product_matches_shader() {
        use super::super::gpu::Step;
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let gemm16 = gpu.kernel(shaders::GEMM16, 3, 5, Some(16)).unwrap();
        let tokens = 1920;
        for (rows, columns, tiny) in [
            (INNER * 2, DIM, false),
            (DIM, INNER, false),
            (DIM * 3, DIM, true),
        ] {
            let mut state = 77u32;
            // Half floats between 2^-5 and 2, either sign; subnormals when `tiny`.
            let mut random = |count: usize, tiny: bool| -> Vec<u8> {
                (0..count)
                    .flat_map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let bits = (state >> 16) as u16;
                        let exponent = if tiny { 0 } else { 10 + (bits >> 10 & 31) % 6 };
                        (bits & 0x83ff | exponent << 10).to_le_bytes()
                    })
                    .collect()
            };
            let (input, weight) = (
                random(tokens * columns, false),
                random(rows * columns, tiny),
            );
            let sums_size = (tokens * rows * 4) as u64;
            let (plain_input, plain_weight, plain_sums) = (
                gpu.buffer(input.len() as u64).unwrap(),
                gpu.buffer(weight.len() as u64).unwrap(),
                gpu.buffer(sums_size).unwrap(),
            );
            gpu.upload(&plain_input, 0, &input).unwrap();
            gpu.upload(&plain_weight, 0, &weight).unwrap();
            let steps: Vec<(Vec<u32>, [u32; 3])> = (0..rows)
                .step_by(4096)
                .map(|first| {
                    (
                        vec![
                            columns as u32,
                            0,
                            (first * columns) as u32,
                            first as u32,
                            rows as u32,
                        ],
                        [
                            (4096.min(rows - first) / 256) as u32,
                            (tokens / TILE) as u32,
                            1,
                        ],
                    )
                })
                .collect();
            let buffers = vec![&plain_input, &plain_weight, &plain_sums];
            let steps: Vec<Step> = steps
                .iter()
                .map(|(constants, groups)| Step {
                    kernel: &gemm16,
                    buffers: &buffers,
                    constants,
                    groups: *groups,
                })
                .collect();
            let mut shader_ms = f64::MAX;
            for _ in 0..8 {
                let started = std::time::Instant::now();
                gpu.pass(&steps).unwrap();
                shader_ms = shader_ms.min(started.elapsed().as_secs_f64() * 1e3);
            }
            let want = floats(&gpu.download(&plain_sums, 0, sums_size as usize).unwrap());

            let bytes = columns * 2;
            let mut streamed = vec![0u8; input.len() + int8::SLACK];
            for token in 0..tokens {
                for word in 0..bytes / 16 {
                    let at = int8::token_word(token, word, bytes) * 16;
                    let from = token * bytes + word * 16;
                    streamed[at..at + 16].copy_from_slice(&input[from..from + 16]);
                }
            }
            let mut weights = vec![0u8; weight.len() + int8::SLACK];
            int8::weight_stream(&weight, rows, bytes, &mut weights);
            let (stream_input, stream_weight, sums) = (
                gpu.shared(streamed.len() as u64).unwrap(),
                gpu.shared(weights.len() as u64).unwrap(),
                gpu.shared(sums_size).unwrap(),
            );
            gpu.upload(&stream_input, 0, &streamed).unwrap();
            gpu.upload(&stream_weight, 0, &weights).unwrap();
            let import = |buffer: &Buffer| {
                fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                    .unwrap()
            };
            let memory = [import(&stream_input), import(&stream_weight), import(&sums)];
            let memory: Vec<_> = memory.iter().collect();
            let shape = int8::Shape::half(tokens, rows, columns).unwrap();
            let kernel = fast_kernel(&fast, &shape).unwrap();
            let mut ms = f64::MAX;
            for _ in 0..8 {
                let started = std::time::Instant::now();
                fast.launch(&kernel, &memory, shape.groups(rows)).unwrap();
                ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
            }
            let got = floats(&gpu.download(&sums, 0, sums_size as usize).unwrap());
            let db = decibels(&want, &got);
            let largest = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let flops = 2.0 * (tokens * rows * columns) as f64 / 1e9;
            println!(
                "{tokens} x {rows} x {columns}{}: {ms:.2} ms ({:.0} TFLOPS), shader {shader_ms:.2} ms ({:.0} TFLOPS), {db:.1} dB apart, largest sum {largest:e}",
                if tiny { " subnormal weights" } else { "" },
                flops / ms,
                flops / shader_ms
            );
            assert!(db > 100.0 && largest > 0.0);
        }
    }
    /// Needs the Intel GPU, the model and a captured tile (see `video_vae` tests), plus
    /// `tile-rust-cpu.f32` in the golden directory: the CPU reference's pixels for that tile,
    /// which this test writes on its first run (about half a minute).
    #[test]
    #[ignore]
    fn gpu_tile_matches_cpu_reference_and_reports_speed() {
        let model = std::env::var("KUNO_VIDEO_VAE").unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_VIDEO_GOLDEN").unwrap());
        let shape: Vec<usize> = std::fs::read_to_string(golden.join("tile-shape.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height, frames) = (shape[0], shape[1], shape[2]);
        let latent = read(&golden.join("tile-latent.f32"));
        let reference = golden.join("tile-rust-cpu.f32");
        if !reference.exists() {
            let cpu = super::super::video_vae::VideoDecoder::open(Path::new(&model))
                .unwrap()
                .decode_tile(&latent, width, height, frames)
                .unwrap();
            std::fs::write(&reference, bytes_of(&cpu)).unwrap();
        }
        let (cpu, cpp) = (read(&reference), read(&golden.join("tile-pixels.f32")));
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let cpp_db = decibels(&cpu, &cpp);
        for shaders in [false, true] {
            let started = std::time::Instant::now();
            let decoder = if shaders {
                GpuVideoDecoder::load_shaders(&gpu, Path::new(&model))
            } else {
                GpuVideoDecoder::load(&gpu, Path::new(&model))
            }
            .unwrap();
            let load = started.elapsed().as_secs_f64();
            let mut seconds = Vec::new();
            let mut pixels = Vec::new();
            for _ in 0..4 {
                let started = std::time::Instant::now();
                pixels = decoder.decode_tile(&latent, width, height, frames).unwrap();
                seconds.push(started.elapsed().as_secs_f64());
            }
            let gpu_db = decibels(&cpu, &pixels);
            println!(
                "{}: load {load:.1} s; tile {seconds:.3?} s; gpu vs reference {gpu_db:.1} dB, c++ vs reference {cpp_db:.1} dB",
                if shaders {
                    "shaders only"
                } else {
                    "machine code"
                }
            );
            assert!(gpu_db >= cpp_db);
        }
    }
    /// Needs the Intel GPU, the model and a whole decode captured from the C++ engine:
    /// `KUNO_VIDEO_CLIP=<dir>` with `clip-latent.f32`, `clip-pixels.f32` and
    /// `clip-shape.txt` ("width height latent-frames decoded-frames").
    #[test]
    #[ignore]
    fn gpu_clip_matches_cpp_engine_and_reports_speed() {
        let model = std::env::var("KUNO_VIDEO_VAE").unwrap();
        let golden = std::path::PathBuf::from(std::env::var("KUNO_VIDEO_CLIP").unwrap());
        let shape: Vec<usize> = std::fs::read_to_string(golden.join("clip-shape.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let (width, height, frames) = (shape[0], shape[1], shape[2]);
        assert_eq!(super::super::video_tiles::output_frames(frames), shape[3]);
        let latent = read(&golden.join("clip-latent.f32"));
        let cpp = read(&golden.join("clip-pixels.f32"));
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let started = std::time::Instant::now();
        let decoder = GpuVideoDecoder::load(&gpu, Path::new(&model)).unwrap();
        let load = started.elapsed().as_secs_f64();
        let mut seconds = Vec::new();
        let mut pixels = Vec::new();
        for _ in 0..2 {
            let started = std::time::Instant::now();
            pixels = decoder.decode(&latent, width, height, frames).unwrap();
            seconds.push(started.elapsed().as_secs_f64());
        }
        let db = decibels(&cpp, &pixels);
        let worst = cpp
            .iter()
            .zip(&pixels)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "{width}x{height}x{frames} -> {} frames: load {load:.1} s, decode {seconds:.2?} s; vs c++ {db:.1} dB, largest pixel difference {worst:.4}",
            shape[3]
        );
        assert!(db > 55.0);
    }
}
