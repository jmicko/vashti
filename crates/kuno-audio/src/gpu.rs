//! The backbone and the depth decoder on a GPU (Vulkan, through `kuno-image`'s device layer).
//!
//! Generating speech is a long chain of small steps: per 80 ms frame, the backbone reads all
//! of its 1.85 GB once for one token, and the depth decoder runs three passes over twelve
//! positions. Both are bound by memory bandwidth on the CPU (about 50 and 18 ms a frame); a
//! GPU reads its memory ten times faster. Each forward is recorded as one batch of
//! dispatches (one submission), so the many small steps do not each wait for the host.
//!
//! Kernels (`shaders/`): `mv` (Q8_0, f16 or f32 rows times a few tokens), `norm` (RMS norms
//! and residual adds), `rope`, `attend` (grouped-query attention over cached keys),
//! `conv` (the short convolution) and `ew` (SwiGLU, per-head gates, adds). Activations are
//! f32. The codec and the audio encoder stay on the CPU (the codec overlaps the GPU).
use kuno_image::engine::gpu::{Buffer, Gpu, Kernel, Mapped, Readback, Select, Step, bytes_of};

use crate::{
    Result,
    block::{Block, Op},
    tensor::Linear,
};

const MV: &[u8] = include_bytes!("../shaders/mv.spv");
const NORM: &[u8] = include_bytes!("../shaders/norm.spv");
const ROPE: &[u8] = include_bytes!("../shaders/rope.spv");
const ATTEND: &[u8] = include_bytes!("../shaders/attend.spv");
const CONV: &[u8] = include_bytes!("../shaders/conv.spv");
const EW: &[u8] = include_bytes!("../shaders/ew.spv");

/// Tokens per prompt chunk (bounds the activation and score buffers).
pub const CHUNK: usize = 64;

struct Kernels {
    mv: Kernel,
    norm: Kernel,
    rope: Kernel,
    attend: Kernel,
    conv: Kernel,
    ew: Kernel,
}

/// A matrix on the device: kind 0 Q8_0 (int8 values and f16 scales), 1 f16, 2 f32.
pub struct Mat {
    w: Buffer,
    scales: Option<Buffer>,
    kind: u32,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    fn upload(gpu: &Gpu, m: &Linear) -> Result<Self> {
        let (rows, cols) = (m.rows(), m.cols());
        match m {
            Linear::Q8 { w, .. } => {
                // Blocks of 34 bytes (f16 scale, 32 int8) split into two aligned arrays.
                let blocks = rows * cols / 32;
                let mut values = vec![0u8; rows * cols];
                let mut scales = vec![0u16; blocks.next_multiple_of(2)];
                for (b, block) in w.chunks_exact(34).enumerate() {
                    scales[b] = u16::from_le_bytes([block[0], block[1]]);
                    values[b * 32..][..32].copy_from_slice(&block[2..]);
                }
                Ok(Self {
                    w: filled(gpu, &values)?,
                    scales: Some(filled(gpu, bytes_of(&scales))?),
                    kind: 0,
                    rows,
                    cols,
                })
            }
            Linear::F16 { w, .. } => Ok(Self {
                w: filled(gpu, bytes_of(w))?,
                scales: None,
                kind: 1,
                rows,
                cols,
            }),
            Linear::F32 { w, .. } => Ok(Self {
                w: filled(gpu, bytes_of(w))?,
                scales: None,
                kind: 2,
                rows,
                cols,
            }),
        }
    }
}

fn filled(gpu: &Gpu, bytes: &[u8]) -> Result<Buffer> {
    // Storage buffers must not be empty; round up to whole words.
    let size = bytes.len().max(4).next_multiple_of(4);
    gpu.filled(size as u64, |out| {
        out[..bytes.len()].copy_from_slice(bytes);
        out[bytes.len()..].fill(0);
        Ok(())
    })
}
fn floats(gpu: &Gpu, values: &[f32]) -> Result<Buffer> {
    filled(gpu, bytes_of(values))
}

#[allow(clippy::large_enum_variant)]
enum GpuOp {
    Attention {
        q: Mat,
        k: Mat,
        v: Mat,
        o: Mat,
        q_norm: Buffer,
        k_norm: Buffer,
        gate: Option<Mat>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        factors: Option<Buffer>,
        window: usize,
        /// Cached keys and values, `[position][kv_heads * head_dim]`.
        keys: Buffer,
        values: Buffer,
    },
    Conv {
        in_proj: Mat,
        taps: Buffer,
        kernel: usize,
        out_proj: Mat,
        /// The last `kernel - 1` inputs per channel, `[channel][kernel - 1]`.
        history: Buffer,
    },
}

/// One LFM block on the device, with its state.
pub struct GpuBlock {
    width: usize,
    eps: f32,
    ln1: Buffer,
    ln2: Buffer,
    post_op: Option<Buffer>,
    post_ffn: Option<Buffer>,
    op: GpuOp,
    gate: Mat,
    up: Mat,
    down: Mat,
}

impl GpuBlock {
    /// `positions`: the most positions its attention ever holds (cache rows).
    fn upload(gpu: &Gpu, b: &Block, positions: usize) -> Result<Self> {
        let op = match &b.op {
            Op::Attention(a) => {
                let kvw = a.kv_heads * a.head_dim;
                GpuOp::Attention {
                    q: Mat::upload(gpu, &a.q)?,
                    k: Mat::upload(gpu, &a.k)?,
                    v: Mat::upload(gpu, &a.v)?,
                    o: Mat::upload(gpu, &a.o)?,
                    q_norm: floats(gpu, &a.q_norm)?,
                    k_norm: floats(gpu, &a.k_norm)?,
                    gate: a.gate.as_ref().map(|g| Mat::upload(gpu, g)).transpose()?,
                    heads: a.heads,
                    kv_heads: a.kv_heads,
                    head_dim: a.head_dim,
                    rope_base: a.rope_base,
                    factors: a.factors.as_deref().map(|f| floats(gpu, f)).transpose()?,
                    window: a.window.unwrap_or(0),
                    keys: gpu.buffer((positions * kvw * 4) as u64)?,
                    values: gpu.buffer((positions * kvw * 4) as u64)?,
                }
            }
            Op::Conv(c) => GpuOp::Conv {
                in_proj: Mat::upload(gpu, &c.in_proj)?,
                taps: floats(gpu, &c.taps)?,
                kernel: c.kernel,
                out_proj: Mat::upload(gpu, &c.out_proj)?,
                history: gpu.buffer((b.width * (c.kernel - 1)).max(1) as u64 * 4)?,
            },
        };
        Ok(Self {
            width: b.width,
            eps: b.eps,
            ln1: floats(gpu, &b.ln1)?,
            ln2: floats(gpu, &b.ln2)?,
            post_op: b.post_op.as_deref().map(|w| floats(gpu, w)).transpose()?,
            post_ffn: b.post_ffn.as_deref().map(|w| floats(gpu, w)).transpose()?,
            op,
            gate: Mat::upload(gpu, &b.gate)?,
            up: Mat::upload(gpu, &b.up)?,
            down: Mat::upload(gpu, &b.down)?,
        })
    }
    fn inner(&self) -> usize {
        self.gate.rows
    }
    fn q_width(&self) -> usize {
        match &self.op {
            GpuOp::Attention {
                heads, head_dim, ..
            } => heads * head_dim,
            GpuOp::Conv { .. } => 0,
        }
    }
    fn heads(&self) -> usize {
        match &self.op {
            GpuOp::Attention { heads, .. } => *heads,
            GpuOp::Conv { .. } => 0,
        }
    }
    /// Clears the state (a new request).
    fn reset(&self, gpu: &Gpu, zero: &Readback, positions: usize) -> Result<()> {
        let clear = |b: &Buffer, count: usize| -> Result<()> {
            for start in (0..count).step_by(ZERO_FLOATS) {
                let n = ZERO_FLOATS.min(count - start);
                gpu.write_floats(b, (start * 4) as u64, zero, &ZEROS[..n])?;
            }
            Ok(())
        };
        match &self.op {
            GpuOp::Conv {
                history, kernel, ..
            } => clear(history, self.width * (kernel - 1)),
            // Keys are written before they are read; nothing to clear.
            GpuOp::Attention { .. } => {
                let _ = positions;
                Ok(())
            }
        }
    }
}

const ZERO_FLOATS: usize = 4096;
static ZEROS: [f32; ZERO_FLOATS] = [0.0; ZERO_FLOATS];

/// Scratch buffers shared by a network's blocks, for up to `rows` tokens.
struct Scratch {
    h: Buffer,
    op: Buffer,
    q: Buffer,
    att: Buffer,
    bcx: Buffer,
    gates: Buffer,
    ga: Buffer,
    ua: Buffer,
    scores: Buffer,
    score_cap: usize,
}

impl Scratch {
    fn new(gpu: &Gpu, blocks: &[GpuBlock], rows: usize, keys: usize) -> Result<Self> {
        let w = blocks.iter().map(|b| b.width).max().unwrap_or(1);
        let qw = blocks
            .iter()
            .map(GpuBlock::q_width)
            .max()
            .unwrap_or(1)
            .max(1);
        let heads = blocks.iter().map(GpuBlock::heads).max().unwrap_or(1).max(1);
        let inner = blocks.iter().map(GpuBlock::inner).max().unwrap_or(1);
        let f = |count: usize| gpu.buffer((count.max(1) * 4) as u64);
        Ok(Self {
            h: f(rows * w)?,
            op: f(rows * w)?,
            // Also the convolution's output, `[rows][width]`.
            q: f(rows * qw.max(w))?,
            att: f(rows * qw)?,
            bcx: f(rows * 3 * w)?,
            gates: f(rows * heads)?,
            ga: f(rows * inner)?,
            ua: f(rows * inner)?,
            scores: f(rows * heads * keys)?,
            score_cap: keys,
        })
    }
}

/// A kernel, its buffers, constants and grid.
type Recorded<'a> = (&'a Kernel, Vec<&'a Buffer>, Vec<u32>, [u32; 3]);

/// Dispatches recorded for one submission. Buffers and constants are owned per step.
struct Recording<'a> {
    steps: Vec<Recorded<'a>>,
}

impl<'a> Recording<'a> {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }
    fn push(&mut self, k: &'a Kernel, b: Vec<&'a Buffer>, c: Vec<u32>, groups: u32) {
        // Grids wider than 65535 workgroups (the portable limit) wrap onto a second row.
        let groups = groups.max(1);
        let grid = if groups > 65_535 {
            [65_535, groups.div_ceil(65_535), 1]
        } else {
            [groups, 1, 1]
        };
        self.steps.push((k, b, c, grid));
    }
    fn run(&self, gpu: &Gpu) -> Result<()> {
        let steps: Vec<Step> = self
            .steps
            .iter()
            .map(|(kernel, buffers, constants, groups)| Step {
                kernel,
                buffers,
                constants,
                groups: *groups,
            })
            .collect();
        gpu.pass(&steps)
    }
}

/// Places for `mv`'s input and output.
#[derive(Clone, Copy)]
struct At<'a> {
    buffer: &'a Buffer,
    off: usize,
    stride: usize,
}
fn at(buffer: &Buffer, off: usize, stride: usize) -> At<'_> {
    At {
        buffer,
        off,
        stride,
    }
}

pub struct Device {
    pub gpu: Gpu,
    k: Kernels,
    /// A one-word buffer bound where a kernel ignores a binding.
    dummy: Buffer,
    staging: Readback,
}

impl Device {
    pub fn open(pci: &str) -> Result<Self> {
        let gpu = Gpu::open(Select::Pci(pci.to_owned()))?;
        let kernel =
            |spirv: &[u8], bindings, constants| gpu.kernel(spirv, bindings, constants, None);
        let k = Kernels {
            mv: kernel(MV, 5, 11)?,
            norm: kernel(NORM, 3, 10)?,
            rope: kernel(ROPE, 2, 8)?,
            attend: kernel(ATTEND, 5, 14)?,
            conv: kernel(CONV, 4, 4)?,
            ew: kernel(EW, 3, 6)?,
        };
        let dummy = gpu.buffer(16)?;
        // Staging for clearing state between requests.
        let staging = gpu.readback((ZERO_FLOATS * 4) as u64)?;
        Ok(Self {
            gpu,
            k,
            dummy,
            staging,
        })
    }

    fn mv<'a>(&'a self, r: &mut Recording<'a>, m: &'a Mat, x: At<'a>, y: At<'a>, n: usize) {
        self.mv_full(r, m, x, y, n, 0, m.rows, None, false);
    }
    #[allow(clippy::too_many_arguments)]
    fn mv_full<'a>(
        &'a self,
        r: &mut Recording<'a>,
        m: &'a Mat,
        x: At<'a>,
        y: At<'a>,
        n: usize,
        row0: usize,
        rows: usize,
        bias: Option<(&'a Buffer, usize)>,
        accumulate: bool,
    ) {
        let flags = u32::from(bias.is_some()) | (u32::from(accumulate) << 1);
        r.push(
            &self.k.mv,
            vec![
                &m.w,
                m.scales.as_ref().unwrap_or(&self.dummy),
                x.buffer,
                y.buffer,
                bias.map_or(&self.dummy, |b| b.0),
            ],
            vec![
                rows as u32,
                m.cols as u32,
                n as u32,
                m.kind,
                x.off as u32,
                x.stride as u32,
                y.off as u32,
                y.stride as u32,
                row0 as u32,
                bias.map_or(0, |b| b.1) as u32,
                flags,
            ],
            rows as u32,
        );
    }
    /// `mode` as in `norm.comp`: 0 write, 1 add the normed rows, 2 add without a norm.
    #[allow(clippy::too_many_arguments)]
    fn norm<'a>(
        &'a self,
        r: &mut Recording<'a>,
        x: At<'a>,
        weight: Option<(&'a Buffer, usize, usize)>,
        out: At<'a>,
        rows: usize,
        width: usize,
        eps: f32,
        mode: u32,
    ) {
        let (w, w_off, w_stride) = weight.unwrap_or((&self.dummy, 0, 0));
        r.push(
            &self.k.norm,
            vec![x.buffer, w, out.buffer],
            vec![
                rows as u32,
                width as u32,
                x.off as u32,
                x.stride as u32,
                out.off as u32,
                out.stride as u32,
                w_off as u32,
                w_stride as u32,
                eps.to_bits(),
                mode,
            ],
            rows as u32,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn ew<'a>(
        &'a self,
        r: &mut Recording<'a>,
        mode: u32,
        a: At<'a>,
        b: At<'a>,
        out: At<'a>,
        count: usize,
        dim: usize,
    ) {
        r.push(
            &self.k.ew,
            vec![a.buffer, b.buffer, out.buffer],
            vec![
                count as u32,
                mode,
                a.off as u32,
                b.off as u32,
                out.off as u32,
                dim as u32,
            ],
            count.div_ceil(64) as u32,
        );
    }

    /// Records `blocks` over `n` rows of `x` (the residual stream, `[n][width]`, updated in
    /// place). Causal blocks see their cached positions `start..start + n` and earlier;
    /// otherwise the rows attend to each other only.
    #[allow(clippy::too_many_arguments)]
    fn record<'a>(
        &'a self,
        r: &mut Recording<'a>,
        blocks: &'a [GpuBlock],
        s: &'a Scratch,
        x: &'a Buffer,
        n: usize,
        start: usize,
        causal: bool,
    ) {
        for b in blocks {
            let w = b.width;
            self.norm(
                r,
                at(x, 0, w),
                Some((&b.ln1, 0, 0)),
                at(&s.h, 0, w),
                n,
                w,
                b.eps,
                0,
            );
            match &b.op {
                GpuOp::Conv {
                    in_proj,
                    taps,
                    kernel,
                    out_proj,
                    history,
                } => {
                    self.mv(r, in_proj, at(&s.h, 0, w), at(&s.bcx, 0, 3 * w), n);
                    r.push(
                        &self.k.conv,
                        vec![&s.bcx, taps, history, &s.q],
                        vec![n as u32, w as u32, *kernel as u32, 0],
                        w.div_ceil(64) as u32,
                    );
                    self.mv(r, out_proj, at(&s.q, 0, w), at(&s.op, 0, w), n);
                }
                GpuOp::Attention {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gate,
                    heads,
                    kv_heads,
                    head_dim,
                    rope_base,
                    factors,
                    window,
                    keys,
                    values,
                } => {
                    let (hd, qw, kvw) = (*head_dim, heads * head_dim, kv_heads * head_dim);
                    let first = if causal { start } else { 0 };
                    self.mv(r, q, at(&s.h, 0, w), at(&s.q, 0, qw), n);
                    self.mv(r, k, at(&s.h, 0, w), at(keys, first * kvw, kvw), n);
                    self.mv(r, v, at(&s.h, 0, w), at(values, first * kvw, kvw), n);
                    self.norm(
                        r,
                        at(&s.q, 0, hd),
                        Some((q_norm, 0, 0)),
                        at(&s.q, 0, hd),
                        n * heads,
                        hd,
                        b.eps,
                        0,
                    );
                    self.norm(
                        r,
                        at(keys, first * kvw, hd),
                        Some((k_norm, 0, 0)),
                        at(keys, first * kvw, hd),
                        n * kv_heads,
                        hd,
                        b.eps,
                        0,
                    );
                    let pos0 = if causal { start } else { 0 };
                    for (buffer, off, count, stride) in
                        [(&s.q, 0, *heads, qw), (keys, first * kvw, *kv_heads, kvw)]
                    {
                        r.push(
                            &self.k.rope,
                            vec![buffer, factors.as_ref().unwrap_or(&self.dummy)],
                            vec![
                                n as u32,
                                count as u32,
                                hd as u32,
                                off as u32,
                                stride as u32,
                                pos0 as u32,
                                rope_base.to_bits(),
                                u32::from(factors.is_some()),
                            ],
                            (n * count * hd / 2).div_ceil(64) as u32,
                        );
                    }
                    let total = if causal { start + n } else { n };
                    r.push(
                        &self.k.attend,
                        vec![&s.q, keys, values, &s.scores, &s.att],
                        vec![
                            n as u32,
                            *heads as u32,
                            *kv_heads as u32,
                            hd as u32,
                            0,
                            0,
                            0,
                            0,
                            pos0 as u32,
                            total as u32,
                            u32::from(causal),
                            *window as u32,
                            (1.0 / (hd as f32).sqrt()).to_bits(),
                            s.score_cap as u32,
                        ],
                        (n * heads) as u32,
                    );
                    if let Some(g) = gate {
                        self.mv(r, g, at(&s.h, 0, w), at(&s.gates, 0, *heads), n);
                        self.ew(
                            r,
                            1,
                            at(&s.gates, 0, 0),
                            at(&self.dummy, 0, 0),
                            at(&s.att, 0, 0),
                            n * qw,
                            hd,
                        );
                    }
                    self.mv(r, o, at(&s.att, 0, qw), at(&s.op, 0, w), n);
                }
            }
            match &b.post_op {
                Some(pw) => self.norm(
                    r,
                    at(&s.op, 0, w),
                    Some((pw, 0, 0)),
                    at(x, 0, w),
                    n,
                    w,
                    b.eps,
                    1,
                ),
                None => self.norm(r, at(&s.op, 0, w), None, at(x, 0, w), n, w, b.eps, 2),
            }
            self.norm(
                r,
                at(x, 0, w),
                Some((&b.ln2, 0, 0)),
                at(&s.h, 0, w),
                n,
                w,
                b.eps,
                0,
            );
            let inner = b.inner();
            self.mv(r, &b.gate, at(&s.h, 0, w), at(&s.ga, 0, inner), n);
            self.mv(r, &b.up, at(&s.h, 0, w), at(&s.ua, 0, inner), n);
            self.ew(
                r,
                0,
                at(&s.ga, 0, 0),
                at(&s.ua, 0, 0),
                at(&s.ga, 0, 0),
                n * inner,
                1,
            );
            self.mv(r, &b.down, at(&s.ga, 0, inner), at(&s.op, 0, w), n);
            match &b.post_ffn {
                Some(pw) => self.norm(
                    r,
                    at(&s.op, 0, w),
                    Some((pw, 0, 0)),
                    at(x, 0, w),
                    n,
                    w,
                    b.eps,
                    1,
                ),
                None => self.norm(r, at(&s.op, 0, w), None, at(x, 0, w), n, w, b.eps, 2),
            }
        }
    }
}

/// The backbone on the device: blocks, state, the final norm.
pub struct GpuBackbone {
    blocks: Vec<GpuBlock>,
    scratch: Scratch,
    out_norm: Buffer,
    /// The residual stream for a chunk, `[CHUNK][2048]`, and the host's rows for it.
    x: Buffer,
    input: Mapped,
    /// The last row's normed state (the depth decoder's input), `[2048]`.
    pub state: Buffer,
    positions: usize,
    next: usize,
    eps: f32,
}

impl GpuBackbone {
    pub fn upload(
        d: &Device,
        blocks: &[Block],
        out_norm: &[f32],
        positions: usize,
    ) -> Result<Self> {
        let blocks = blocks
            .iter()
            .map(|b| GpuBlock::upload(&d.gpu, b, positions))
            .collect::<Result<Vec<_>>>()?;
        let scratch = Scratch::new(&d.gpu, &blocks, CHUNK, positions)?;
        Ok(Self {
            eps: blocks.first().map_or(1e-5, |b| b.eps),
            blocks,
            scratch,
            out_norm: floats(&d.gpu, out_norm)?,
            x: d.gpu.buffer((CHUNK * 2048 * 4) as u64)?,
            input: d.gpu.mapped((CHUNK * 2048 * 4) as u64)?,
            state: d.gpu.buffer(2048 * 4)?,
            positions,
            next: 0,
        })
    }
    /// A new request: positions from zero, convolution histories cleared.
    pub fn reset(&mut self, d: &Device) -> Result<()> {
        for b in &self.blocks {
            b.reset(&d.gpu, &d.staging, self.positions)?;
        }
        self.next = 0;
        Ok(())
    }
    pub fn position(&self) -> usize {
        self.next
    }
    /// Runs `rows` (`[n][2048]`) through the backbone; the last row's normed state ends up in
    /// `state`.
    pub fn forward(&mut self, d: &Device, rows: &[f32]) -> Result<()> {
        let n_all = rows.len() / 2048;
        if self.next + n_all > self.positions {
            return Err("audio_context_exhausted");
        }
        for chunk in rows.chunks(CHUNK * 2048) {
            let n = chunk.len() / 2048;
            d.gpu.write_mapped(&self.input, 0, chunk)?;
            let mut r = Recording::new();
            let input = self.input.buffer();
            d.ew(
                &mut r,
                4,
                at(input, 0, 0),
                at(input, 0, 0),
                at(&self.x, 0, 0),
                chunk.len(),
                1,
            );
            d.record(
                &mut r,
                &self.blocks,
                &self.scratch,
                &self.x,
                n,
                self.next,
                true,
            );
            d.norm(
                &mut r,
                at(&self.x, (n - 1) * 2048, 2048),
                Some((&self.out_norm, 0, 0)),
                at(&self.state, 0, 2048),
                1,
                2048,
                self.eps,
                0,
            );
            r.run(&d.gpu)?;
            self.next += n;
        }
        Ok(())
    }
    /// The text head over `state`: logits through the tied embeddings.
    pub fn logits(&self, d: &Device, head: &Mat, out: &mut [f32], logits: &Mapped) -> Result<()> {
        let mut r = Recording::new();
        d.mv(
            &mut r,
            head,
            at(&self.state, 0, 2048),
            at(logits.buffer(), 0, head.rows),
            1,
        );
        r.run(&d.gpu)?;
        d.gpu.read_mapped(logits, 0, out)
    }
}

/// The depth decoder on the device.
pub struct GpuDepth {
    blocks: Vec<GpuBlock>,
    scratch: Scratch,
    proj_in: Mat,
    proj_bias: Buffer,
    code_embd: Mat,
    norm: Buffer,
    head_norm: Buffer,
    /// The projected state `[12][1024]`, the pass input and the logits `[4][2017]`.
    base: Buffer,
    x: Buffer,
    add: Mapped,
    logits: Mapped,
    eps: f32,
}

pub const POSITIONS: usize = 12;
const WIDTH: usize = 1024;

impl GpuDepth {
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        d: &Device,
        blocks: &[Block],
        proj_in: &Linear,
        proj_bias: &[f32],
        code_embd: &Linear,
        norm: &[f32],
        head_norm: &[f32],
        codes: usize,
    ) -> Result<Self> {
        let g = &d.gpu;
        let blocks = blocks
            .iter()
            .map(|b| GpuBlock::upload(g, b, POSITIONS))
            .collect::<Result<Vec<_>>>()?;
        let scratch = Scratch::new(g, &blocks, POSITIONS, POSITIONS)?;
        Ok(Self {
            eps: blocks.first().map_or(1e-5, |b| b.eps),
            blocks,
            scratch,
            proj_in: Mat::upload(g, proj_in)?,
            proj_bias: floats(g, proj_bias)?,
            code_embd: Mat::upload(g, code_embd)?,
            norm: floats(g, norm)?,
            head_norm: floats(g, head_norm)?,
            base: g.buffer((POSITIONS * WIDTH * 4) as u64)?,
            x: g.buffer((POSITIONS * WIDTH * 4) as u64)?,
            add: g.mapped((POSITIONS * WIDTH * 4) as u64)?,
            logits: g.mapped((POSITIONS * codes * 4) as u64)?,
        })
    }
    /// One pass: with `project` (the frame's first pass), the backbone state projected into
    /// the twelve positions first; then the positions plus `add` (`[12][1024]`, each position's code embedding or
    /// the mask), three bidirectional blocks, the norms, and the logits of codebooks
    /// `wanted` (`codes` each) into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn pass(
        &self,
        d: &Device,
        project: Option<&Buffer>,
        add: &[f32],
        wanted: &[usize],
        codes: usize,
        out: &mut [f32],
    ) -> Result<()> {
        d.gpu.write_mapped(&self.add, 0, add)?;
        let mut r = Recording::new();
        if let Some(state) = project {
            d.mv_full(
                &mut r,
                &self.proj_in,
                at(state, 0, 2048),
                at(&self.base, 0, self.proj_in.rows),
                1,
                0,
                self.proj_in.rows,
                Some((&self.proj_bias, 0)),
                false,
            );
        }
        let count = POSITIONS * WIDTH;
        let add = self.add.buffer();
        d.ew(
            &mut r,
            2,
            at(&self.base, 0, 0),
            at(add, 0, 0),
            at(&self.x, 0, 0),
            count,
            1,
        );
        d.record(
            &mut r,
            &self.blocks,
            &self.scratch,
            &self.x,
            POSITIONS,
            0,
            false,
        );
        let (x, eps) = (&self.x, self.eps);
        d.norm(
            &mut r,
            at(x, 0, WIDTH),
            Some((&self.norm, 0, 0)),
            at(x, 0, WIDTH),
            POSITIONS,
            WIDTH,
            eps,
            0,
        );
        d.norm(
            &mut r,
            at(x, 0, WIDTH),
            Some((&self.head_norm, 0, WIDTH)),
            at(x, 0, WIDTH),
            POSITIONS,
            WIDTH,
            eps,
            0,
        );
        for (i, &cb) in wanted.iter().enumerate() {
            d.mv_full(
                &mut r,
                &self.code_embd,
                at(x, cb * WIDTH, WIDTH),
                at(self.logits.buffer(), i * codes, codes),
                1,
                cb * codes,
                codes,
                None,
                false,
            );
        }
        r.run(&d.gpu)?;
        d.gpu
            .read_mapped(&self.logits, 0, &mut out[..wanted.len() * codes])
    }
}

/// The text head (tied embeddings) on the device, for recognition.
pub fn upload_head(d: &Device, embed: &Linear) -> Result<(Mat, Mapped)> {
    let m = Mat::upload(&d.gpu, embed)?;
    let logits = d.gpu.mapped((m.rows * 4) as u64)?;
    Ok((m, logits))
}

#[cfg(test)]
mod tests {
    /// The checked-in SPIR-V matches its source (recompiled with `glslangValidator` when
    /// installed; the binaries are checked in so builds need no shader compiler).
    #[test]
    fn checked_in_spirv_matches_sources() {
        use std::process::Command;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
        if Command::new("glslangValidator")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("glslangValidator not installed; skipping");
            return;
        }
        let out = std::env::temp_dir().join(format!("kuno-audio-spv-{}", std::process::id()));
        std::fs::create_dir_all(&out).unwrap();
        for name in ["mv", "norm", "rope", "attend", "conv", "ew"] {
            let built = out.join(format!("{name}.spv"));
            let status = Command::new("glslangValidator")
                .args(["-V", "--target-env", "vulkan1.3", "--quiet", "-o"])
                .arg(&built)
                .arg(dir.join(format!("{name}.comp")))
                .status()
                .unwrap();
            assert!(status.success(), "{name} does not compile");
            let checked_in = std::fs::read(dir.join(format!("{name}.spv"))).unwrap();
            assert_eq!(
                std::fs::read(&built).unwrap(),
                checked_in,
                "{name}.spv is stale"
            );
        }
        let _ = std::fs::remove_dir_all(&out);
    }
}
