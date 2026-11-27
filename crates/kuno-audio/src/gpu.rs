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
use kuno_image::engine::gpu::{
    Buffer, Gpu, Kernel, Mapped, Passing, Readback, Select, Step, bytes_of,
};
use std::cell::Cell;

use crate::{
    Result,
    block::{Block, Op},
    tensor::Linear,
};

const MV: &[u8] = include_bytes!("../shaders/mv.spv");
const MVB: &[u8] = include_bytes!("../shaders/mvb.spv");
const MV1: &[u8] = include_bytes!("../shaders/mv1.spv");
const MVC: &[u8] = include_bytes!("../shaders/mvc.spv");
const MVC8: &[u8] = include_bytes!("../shaders/mvc8.spv");
const MV1G: &[u8] = include_bytes!("../shaders/mv1g.spv");
const MVCG: &[u8] = include_bytes!("../shaders/mvcg.spv");
const MVC8G: &[u8] = include_bytes!("../shaders/mvc8g.spv");
const MV1S: &[u8] = include_bytes!("../shaders/mv1s.spv");
const SAMPLE: &[u8] = include_bytes!("../shaders/sample.spv");
const GATHER: &[u8] = include_bytes!("../shaders/gather.spv");
const NORM: &[u8] = include_bytes!("../shaders/norm.spv");
const ADDNORM: &[u8] = include_bytes!("../shaders/addnorm.spv");
const NORMROPE: &[u8] = include_bytes!("../shaders/normrope.spv");
const ARGMAX: &[u8] = include_bytes!("../shaders/argmax.spv");
const ROW: &[u8] = include_bytes!("../shaders/row.spv");
const MMQ: &[u8] = include_bytes!("../shaders/mmq.spv");
const MMQ8: &[u8] = include_bytes!("../shaders/mmq8.spv");
const ROPE: &[u8] = include_bytes!("../shaders/rope.spv");
const ATTEND: &[u8] = include_bytes!("../shaders/attend.spv");
const CONV: &[u8] = include_bytes!("../shaders/conv.spv");
const EW: &[u8] = include_bytes!("../shaders/ew.spv");

/// Tokens per prompt chunk (bounds the activation and score buffers).
pub const CHUNK: usize = 64;

struct Kernels {
    mv: Kernel,
    /// `mv` for several tokens: 4 rows x 4 tokens per lane (f16 and Q8_0).
    mvb: Kernel,
    /// `mv` for one token: a subgroup per row, `mv1_rows` rows per workgroup.
    mv1: Kernel,
    mv1_rows: usize,
    /// `mv` for 2 to 16 tokens of f16 rows on half-precision matrix units, where the
    /// device has 16 x 16 x 16 ones for 32-lane subgroups (mvc) or 8 x 16 x 16 ones for
    /// 16-lane subgroups (mvc8); both take the same tiled weights, 16 rows a workgroup.
    mvc: Option<Kernel>,
    /// SwiGLU's gate and up products in one (`mv1g` for one token, and the `mvc` kind's).
    mv1g: Kernel,
    mvcg: Option<Kernel>,
    /// `mv1` for long rows: a workgroup per row, its subgroups splitting the row.
    mv1s: Kernel,
    /// `mv` for more than 16 tokens (prompt chunks) of Q8_0 or f16 rows on half-precision
    /// matrix units, where `mvc` is.
    mmq: Option<Kernel>,
    /// Rows a workgroup of `mmq` takes (32; Intel's mmq8 64).
    mmq_rows: usize,
    sample: Kernel,
    gather: Kernel,
    norm: Kernel,
    /// A residual step and the next norm in one (rows of up to 2048).
    addnorm: Kernel,
    /// Per-head norm and rotary embedding in one (heads of up to 128).
    normrope: Kernel,
    /// The first largest value's index, into an id buffer.
    argmax: Kernel,
    /// A matrix row by an id from a buffer, as floats.
    row: Kernel,
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
    /// f16 values in 16 x 16 tiles for `mvc` (per row tile, per column step, the tile's
    /// rows of 16 values in turn), used only for 2 to 16 tokens.
    tiled: bool,
}

impl Mat {
    fn upload(gpu: &Gpu, m: &Linear) -> Result<Self> {
        Self::upload_as(gpu, m, false)
    }
    /// `upload`, f16 matrices of whole 16 x 16 tiles tiled when `tile` (the device has
    /// `mvc` and the matrix only ever multiplies 2 to 16 tokens).
    fn upload_as(gpu: &Gpu, m: &Linear, tile: bool) -> Result<Self> {
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
                    tiled: false,
                })
            }
            Linear::F16 { w, .. } if tile && rows.is_multiple_of(16) && cols.is_multiple_of(16) => {
                let mut tiled = Vec::with_capacity(w.len());
                for tile_row in (0..rows).step_by(16) {
                    for step in (0..cols).step_by(16) {
                        for r in tile_row..tile_row + 16 {
                            tiled.extend_from_slice(&w[r * cols + step..][..16]);
                        }
                    }
                }
                Ok(Self {
                    w: filled(gpu, bytes_of(&tiled))?,
                    scales: None,
                    kind: 1,
                    rows,
                    cols,
                    tiled: true,
                })
            }
            Linear::F16 { w, .. } => Ok(Self {
                w: filled(gpu, bytes_of(w))?,
                scales: None,
                kind: 1,
                rows,
                cols,
                tiled: false,
            }),
            Linear::F32 { w, .. } => Ok(Self {
                w: filled(gpu, bytes_of(w))?,
                scales: None,
                kind: 2,
                rows,
                cols,
                tiled: false,
            }),
        }
    }
}

/// A device buffer holding `bytes`, read back and compared before use. On the RTX 3060
/// test host (driver 595.99), uploads occasionally arrived with whole 64 KiB pages wrong
/// (no kernel errors), and every result from those weights was off; a mismatch is
/// uploaded again, and after three tries the model stays on the CPU.
fn filled(gpu: &Gpu, bytes: &[u8]) -> Result<Buffer> {
    // Storage buffers must not be empty; round up to whole words.
    let size = bytes.len().max(4).next_multiple_of(4);
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
    Err("audio_gpu_upload_corrupt")
}

const PIECE: usize = 16 << 20;
thread_local! {
    /// One readback area per thread and device, reused for every check.
    static CHECK: std::cell::RefCell<Option<(usize, Readback)>> = const { std::cell::RefCell::new(None) };
}

/// Whether `buffer` (`size` bytes, a multiple of 4) starts with exactly `bytes`, read back
/// through a reused staging area in 16 MiB pieces.
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
    /// `small`: tile what `mvc` can take (the block only ever runs 2 to 16 positions at a
    /// time, on a device with `mvc`).
    fn upload(gpu: &Gpu, b: &Block, positions: usize, small: bool) -> Result<Self> {
        let mat = |m: &Linear| Mat::upload_as(gpu, m, small);
        let op = match &b.op {
            Op::Attention(a) => {
                let kvw = a.kv_heads * a.head_dim;
                GpuOp::Attention {
                    q: mat(&a.q)?,
                    k: mat(&a.k)?,
                    v: mat(&a.v)?,
                    o: mat(&a.o)?,
                    q_norm: floats(gpu, &a.q_norm)?,
                    k_norm: floats(gpu, &a.k_norm)?,
                    gate: a.gate.as_ref().map(mat).transpose()?,
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
                in_proj: mat(&c.in_proj)?,
                taps: floats(gpu, &c.taps)?,
                kernel: c.kernel,
                out_proj: mat(&c.out_proj)?,
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
            gate: mat(&b.gate)?,
            up: mat(&b.up)?,
            down: mat(&b.down)?,
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
    /// Per step: whether the next step may start without waiting for it.
    independent: Vec<bool>,
}

impl<'a> Recording<'a> {
    fn new() -> Self {
        Self {
            steps: Vec::new(),
            independent: Vec::new(),
        }
    }
    /// The next step does not depend on the last one (no barrier between them).
    fn together(&mut self) {
        if let Some(last) = self.independent.last_mut() {
            *last = true;
        }
    }
    /// `push` with a grid of `rows` x `columns` workgroups (rows at most 65535).
    fn push_grid(
        &mut self,
        k: &'a Kernel,
        b: Vec<&'a Buffer>,
        c: Vec<u32>,
        rows: u32,
        columns: u32,
    ) {
        self.steps.push((k, b, c, [rows.max(1), columns.max(1), 1]));
        self.independent.push(false);
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
        self.independent.push(false);
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
        gpu.pass_with(&steps, &self.independent)
    }
    /// Appends two recordings that touch none of the same buffers (but read-only ones) side
    /// by side: their steps between barriers are grouped, the groups of both run together and
    /// share each barrier, so the shorter chain's latency hides in the longer one's.
    fn interleave(&mut self, a: Recording<'a>, b: Recording<'a>) {
        fn groups(r: Recording) -> Vec<Vec<Recorded>> {
            let mut out = vec![Vec::new()];
            for (step, independent) in r.steps.into_iter().zip(r.independent) {
                out.last_mut().unwrap().push(step);
                if !independent {
                    out.push(Vec::new());
                }
            }
            out.retain(|g| !g.is_empty());
            out
        }
        let (mut a, mut b) = (groups(a).into_iter(), groups(b).into_iter());
        loop {
            let group: Vec<_> = a.next().into_iter().chain(b.next()).flatten().collect();
            if group.is_empty() {
                break;
            }
            let last = group.len() - 1;
            for (i, step) in group.into_iter().enumerate() {
                self.steps.push(step);
                self.independent.push(i < last);
            }
        }
    }
    /// `run`, timed on the device per step (nanoseconds; for benchmarks).
    #[cfg(test)]
    fn profile(&self, gpu: &Gpu) -> Result<Vec<f64>> {
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
        gpu.pass_profiled(&steps, &self.independent)
    }
    /// `run` without waiting for the device.
    ///
    /// # Safety
    ///
    /// The recording's buffers must outlive the returned `Passing`.
    unsafe fn begin(&self, gpu: &Gpu) -> Result<Passing> {
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
        // SAFETY: the caller's contract.
        unsafe { gpu.pass_begin(&steps, &self.independent) }
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
        // mv1's subgroup size is pinned so the host knows its rows per workgroup.
        let lanes = gpu.subgroup().unwrap_or(32).clamp(8, 128);
        let k = Kernels {
            mv: kernel(MV, 5, 11)?,
            mvb: kernel(MVB, 5, 11)?,
            mv1: gpu.kernel(MV1, 5, 11, Some(lanes))?,
            mv1_rows: 128 / lanes as usize,
            mv1s: kernel(MV1S, 5, 11)?,
            mmq: if gpu.has_half16() && lanes == 32 {
                Some(gpu.kernel(MMQ, 5, 11, Some(32))?)
            } else if gpu.has_half8() {
                Some(gpu.kernel(MMQ8, 5, 11, Some(16))?)
            } else {
                None
            },
            mmq_rows: if gpu.has_half16() && lanes == 32 {
                32
            } else {
                64
            },
            mvc: if gpu.has_half16() && lanes == 32 {
                Some(gpu.kernel(MVC, 5, 11, Some(32))?)
            } else if gpu.has_half8() {
                // Intel's 8 x 16 x 16 units: the transposed kernel at 16-lane subgroups.
                Some(gpu.kernel(MVC8, 5, 11, Some(16))?)
            } else {
                None
            },
            mv1g: gpu.kernel(MV1G, 6, 11, Some(lanes))?,
            mvcg: if gpu.has_half16() && lanes == 32 {
                Some(gpu.kernel(MVCG, 5, 11, Some(32))?)
            } else if gpu.has_half8() {
                Some(gpu.kernel(MVC8G, 5, 11, Some(16))?)
            } else {
                None
            },
            sample: kernel(SAMPLE, 4, 7)?,
            gather: kernel(GATHER, 5, 3)?,
            norm: kernel(NORM, 3, 10)?,
            addnorm: kernel(ADDNORM, 5, 4)?,
            normrope: kernel(NORMROPE, 5, 12)?,
            argmax: kernel(ARGMAX, 2, 2)?,
            row: kernel(ROW, 4, 3)?,
            rope: kernel(ROPE, 2, 8)?,
            attend: kernel(ATTEND, 6, 15)?,
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
        // Several tokens: the register-blocked kernel (1.4 to 2.9 times faster on the RTX
        // 3060 for 12 to 64 tokens); one token, or f32 rows: mv.
        // Tiled matrices (only ever 2 to 16 tokens) go to mvc.
        debug_assert!(!m.tiled || (2..=16).contains(&n) && rows.is_multiple_of(16));
        let matrix = self.k.mvc.as_ref().filter(|_| m.tiled);
        // One token: a subgroup per row, or for long rows (few of them: the FFN's down
        // projection) a workgroup per row (2048 x 7680: 59 to 51 us on the RTX 3060).
        let constants = vec![
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
        ];
        let buffers = |m: &'a Mat| {
            vec![
                &m.w,
                m.scales.as_ref().unwrap_or(&self.dummy),
                x.buffer,
                y.buffer,
                bias.map_or(&self.dummy, |b| b.0),
            ]
        };
        // Prompt chunks: 32 (Intel 64) rows x 64 tokens a workgroup on the matrix units (where there are
        // enough rows to fill the device; fewer stay on mvb).
        if let Some(k) = self.k.mmq.as_ref().filter(|_| {
            n > 16
                && rows >= 1024
                && m.kind <= 1
                && !m.tiled
                && m.cols.is_multiple_of(32)
                && rows <= 65535 * self.k.mmq_rows
        }) {
            r.push_grid(
                k,
                buffers(m),
                constants,
                rows.div_ceil(self.k.mmq_rows) as u32,
                n.div_ceil(64) as u32,
            );
            return;
        }
        let (kernel, groups) = if n == 1 && m.cols >= 4096 {
            (&self.k.mv1s, rows)
        } else if n == 1 {
            (&self.k.mv1, rows.div_ceil(self.k.mv1_rows))
        } else if let Some(k) = matrix {
            (k, rows / 16)
        } else if m.kind != 2 {
            (&self.k.mvb, rows.div_ceil(4))
        } else {
            (&self.k.mv, rows)
        };
        r.push(kernel, buffers(m), constants, groups as u32);
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
    /// SwiGLU's first half: `out = silu(gate x) * (up x)` for `n` tokens of `x` (`inner`
    /// values a token in `out`, `ua` the up products' scratch): one step where a fused
    /// kernel takes the shapes, else the two products side by side and the product.
    #[allow(clippy::too_many_arguments)]
    fn glu<'a>(
        &'a self,
        r: &mut Recording<'a>,
        gate: &'a Mat,
        up: &'a Mat,
        x: At<'a>,
        out: &'a Buffer,
        ua: &'a Buffer,
        n: usize,
    ) {
        let inner = gate.rows;
        let constants = vec![
            inner as u32,
            gate.cols as u32,
            n as u32,
            gate.kind,
            x.off as u32,
            x.stride as u32,
            0,
            inner as u32,
            0,
            0,
            0,
        ];
        let same = gate.kind == up.kind && gate.tiled == up.tiled && gate.cols == up.cols;
        if same && n == 1 && gate.cols < 4096 && !gate.tiled {
            r.push(
                &self.k.mv1g,
                vec![
                    &gate.w,
                    gate.scales.as_ref().unwrap_or(&self.dummy),
                    x.buffer,
                    out,
                    &up.w,
                    up.scales.as_ref().unwrap_or(&self.dummy),
                ],
                constants,
                inner.div_ceil(self.k.mv1_rows) as u32,
            );
            return;
        }
        if let Some(k) = self.k.mvcg.as_ref().filter(|_| same && gate.tiled) {
            r.push(
                k,
                vec![&gate.w, &self.dummy, x.buffer, out, &up.w],
                constants,
                (inner / 16) as u32,
            );
            return;
        }
        self.mv(r, gate, x, at(out, 0, inner), n);
        r.together();
        self.mv(r, up, x, at(ua, 0, inner), n);
        self.ew(
            r,
            0,
            at(out, 0, 0),
            at(ua, 0, 0),
            at(out, 0, 0),
            n * inner,
            1,
        );
    }
    /// `x += op` (op first normed with `post`), then `h` the new `x` normed with `next`:
    /// one step for rows of up to 2048, else separate ones.
    #[allow(clippy::too_many_arguments)]
    fn residual<'a>(
        &'a self,
        r: &mut Recording<'a>,
        s: &'a Scratch,
        x: &'a Buffer,
        b: &'a GpuBlock,
        post: Option<&'a Buffer>,
        next: Option<&'a Buffer>,
        n: usize,
    ) {
        let w = b.width;
        if w <= 2048 {
            let flags = u32::from(post.is_some()) | (u32::from(next.is_some()) << 1);
            r.push(
                &self.k.addnorm,
                vec![
                    &s.op,
                    post.unwrap_or(&self.dummy),
                    x,
                    next.unwrap_or(&self.dummy),
                    &s.h,
                ],
                vec![n as u32, w as u32, b.eps.to_bits(), flags],
                n as u32,
            );
            return;
        }
        match post {
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
        if let Some(nw) = next {
            self.norm(
                r,
                at(x, 0, w),
                Some((nw, 0, 0)),
                at(&s.h, 0, w),
                n,
                w,
                b.eps,
                0,
            );
        }
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
        // Whether `h` already holds this block's normed input (from the last block's step).
        let mut normed = false;
        for (index, b) in blocks.iter().enumerate() {
            let w = b.width;
            if !normed {
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
            }
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
                    // The projections all read `h` and write separate places.
                    self.mv(r, q, at(&s.h, 0, w), at(&s.q, 0, qw), n);
                    r.together();
                    self.mv(r, k, at(&s.h, 0, w), at(keys, first * kvw, kvw), n);
                    r.together();
                    self.mv(r, v, at(&s.h, 0, w), at(values, first * kvw, kvw), n);
                    if let Some(g) = gate {
                        r.together();
                        self.mv(r, g, at(&s.h, 0, w), at(&s.gates, 0, *heads), n);
                    }
                    let pos0 = if causal { start } else { 0 };
                    if hd <= 128 {
                        // Queries and keys in one step.
                        r.push(
                            &self.k.normrope,
                            vec![
                                &s.q,
                                q_norm,
                                factors.as_ref().unwrap_or(&self.dummy),
                                keys,
                                k_norm,
                            ],
                            vec![
                                n as u32,
                                *heads as u32,
                                hd as u32,
                                0,
                                qw as u32,
                                pos0 as u32,
                                rope_base.to_bits(),
                                u32::from(factors.is_some()),
                                b.eps.to_bits(),
                                *kv_heads as u32,
                                (first * kvw) as u32,
                                kvw as u32,
                            ],
                            (n * (heads + kv_heads)) as u32,
                        );
                    } else {
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
                        r.together();
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
                        for (i, (buffer, off, count, stride)) in
                            [(&s.q, 0, *heads, qw), (keys, first * kvw, *kv_heads, kvw)]
                                .into_iter()
                                .enumerate()
                        {
                            if i > 0 {
                                r.together();
                            }
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
                    }
                    let total = if causal { start + n } else { n };
                    r.push(
                        &self.k.attend,
                        vec![
                            &s.q,
                            keys,
                            values,
                            &s.scores,
                            &s.att,
                            if gate.is_some() {
                                &s.gates
                            } else {
                                &self.dummy
                            },
                        ],
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
                            u32::from(gate.is_some()),
                        ],
                        (n * heads) as u32,
                    );
                    self.mv(r, o, at(&s.att, 0, qw), at(&s.op, 0, w), n);
                }
            }
            self.residual(r, s, x, b, b.post_op.as_ref(), Some(&b.ln2), n);
            let inner = b.inner();
            self.glu(r, &b.gate, &b.up, at(&s.h, 0, w), &s.ga, &s.ua, n);
            self.mv(r, &b.down, at(&s.ga, 0, inner), at(&s.op, 0, w), n);
            let next = blocks.get(index + 1).filter(|next| next.width == w);
            self.residual(r, s, x, b, b.post_ffn.as_ref(), next.map(|nb| &nb.ln1), n);
            normed = next.is_some();
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
    next: Cell<usize>,
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
            .map(|b| GpuBlock::upload(&d.gpu, b, positions, false))
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
            next: Cell::new(0),
        })
    }
    /// A new request: positions from zero, convolution histories cleared.
    pub fn reset(&mut self, d: &Device) -> Result<()> {
        for b in &self.blocks {
            b.reset(&d.gpu, &d.staging, self.positions)?;
        }
        self.next.set(0);
        Ok(())
    }
    pub fn position(&self) -> usize {
        self.next.get()
    }
    /// Runs `rows` (`[n][2048]`) through the backbone; the last row's normed state ends up in
    /// `state`.
    pub fn forward(&mut self, d: &Device, rows: &[f32]) -> Result<()> {
        let n_all = rows.len() / 2048;
        if self.next.get() + n_all > self.positions {
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
                self.next.get(),
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
            self.next.set(self.next.get() + n);
        }
        Ok(())
    }
    /// Records one step whose input row is in `src` (on the device).
    fn record_step<'a>(&'a self, d: &'a Device, r: &mut Recording<'a>, src: &'a Buffer) {
        d.ew(
            r,
            4,
            at(src, 0, 0),
            at(src, 0, 0),
            at(&self.x, 0, 0),
            2048,
            1,
        );
        d.record(
            r,
            &self.blocks,
            &self.scratch,
            &self.x,
            1,
            self.next.get(),
            true,
        );
        d.norm(
            r,
            at(&self.x, 0, 2048),
            Some((&self.out_norm, 0, 0)),
            at(&self.state, 0, 2048),
            1,
            2048,
            self.eps,
            0,
        );
    }
}

/// The codec's transformer on the device: 8 blocks over six positions a frame (causal,
/// 30-position windows, state continuing across frames), the output norm and projection to
/// 1282 values per position (log magnitudes and phases). The input projection and the
/// inverse STFT stay on the host. Recorded into the backbone step's submission.
pub struct GpuCodec {
    blocks: Vec<GpuBlock>,
    scratch: Scratch,
    norm: Buffer,
    out: Mat,
    out_bias: Buffer,
    x: Buffer,
    /// Per frame slot (frames alternate, so one is read while the next runs).
    output: [Mapped; 2],
    positions: usize,
    next: Cell<usize>,
    eps: f32,
}

/// The audio encoder's 17 blocks on the device (width 768, causal), a whole piece of audio
/// in one submission, `CHUNK` rows at a time.
pub struct GpuEncoder {
    blocks: Vec<GpuBlock>,
    scratch: Scratch,
    x: Buffer,
    /// The piece's rows in and out, `[positions][768]`.
    input: Mapped,
    output: Mapped,
    positions: usize,
}

const ENCODER_WIDTH: usize = 768;

impl GpuEncoder {
    pub fn upload(d: &Device, blocks: &[Block], positions: usize) -> Result<Self> {
        let g = &d.gpu;
        let blocks = blocks
            .iter()
            .map(|b| GpuBlock::upload(g, b, positions, false))
            .collect::<Result<Vec<_>>>()?;
        if blocks.iter().any(|b| b.width != ENCODER_WIDTH) {
            return Err("model_file_invalid");
        }
        let bytes = (positions * ENCODER_WIDTH * 4) as u64;
        Ok(Self {
            scratch: Scratch::new(g, &blocks, CHUNK, positions)?,
            blocks,
            x: g.buffer((CHUNK * ENCODER_WIDTH * 4) as u64)?,
            input: g.mapped(bytes)?,
            output: g.mapped(bytes)?,
            positions,
        })
    }
    /// The most rows a piece may have.
    pub fn positions(&self) -> usize {
        self.positions
    }
    /// Runs `rows` (`[n][768]`, n at most `positions`) through the blocks in place.
    pub fn forward(&self, d: &Device, rows: &mut [f32]) -> Result<()> {
        let n_all = rows.len() / ENCODER_WIDTH;
        if n_all > self.positions {
            return Err("audio_context_exhausted");
        }
        for b in &self.blocks {
            b.reset(&d.gpu, &d.staging, self.positions)?;
        }
        d.gpu.write_mapped(&self.input, 0, rows)?;
        let (input, output) = (self.input.buffer(), self.output.buffer());
        let mut r = Recording::new();
        let w = ENCODER_WIDTH;
        for start in (0..n_all).step_by(CHUNK) {
            let n = CHUNK.min(n_all - start);
            d.ew(
                &mut r,
                4,
                at(input, start * w, 0),
                at(input, start * w, 0),
                at(&self.x, 0, 0),
                n * w,
                1,
            );
            d.record(&mut r, &self.blocks, &self.scratch, &self.x, n, start, true);
            d.ew(
                &mut r,
                4,
                at(&self.x, 0, 0),
                at(&self.x, 0, 0),
                at(output, start * w, 0),
                n * w,
                1,
            );
        }
        r.run(&d.gpu)?;
        d.gpu.read_mapped(&self.output, 0, rows)
    }
}

/// Positions per codec frame, and its width.
pub const CODEC_POSITIONS: usize = 6;
const CODEC_WIDTH: usize = 512;

impl GpuCodec {
    pub fn upload(
        d: &Device,
        blocks: &[Block],
        norm: &[f32],
        out: &Linear,
        out_bias: &[f32],
        positions: usize,
    ) -> Result<Self> {
        let g = &d.gpu;
        let blocks = blocks
            .iter()
            .map(|b| GpuBlock::upload(g, b, positions, d.k.mvc.is_some()))
            .collect::<Result<Vec<_>>>()?;
        let scratch = Scratch::new(g, &blocks, CODEC_POSITIONS, positions)?;
        let out = Mat::upload(g, out)?;
        Ok(Self {
            eps: blocks.first().map_or(1e-5, |b| b.eps),
            blocks,
            scratch,
            norm: floats(g, norm)?,
            out_bias: floats(g, out_bias)?,
            x: g.buffer((CODEC_POSITIONS * CODEC_WIDTH * 4) as u64)?,
            output: [
                g.mapped((CODEC_POSITIONS * out.rows * 4) as u64)?,
                g.mapped((CODEC_POSITIONS * out.rows * 4) as u64)?,
            ],
            out,
            positions,
            next: Cell::new(0),
        })
    }
    /// A new request: positions from zero, convolution histories cleared.
    pub fn reset(&mut self, d: &Device) -> Result<()> {
        for b in &self.blocks {
            b.reset(&d.gpu, &d.staging, self.positions)?;
        }
        self.next.set(0);
        Ok(())
    }
    /// Values per position of the output (1282).
    pub fn output_width(&self) -> usize {
        self.out.rows
    }
    /// Records a frame whose input (one row, `[512]`) is in `src` on the device, repeated
    /// for the six positions, its output into `slot`.
    fn record_from<'a>(
        &'a self,
        d: &'a Device,
        r: &mut Recording<'a>,
        src: &'a Buffer,
        slot: usize,
    ) {
        let w = CODEC_WIDTH;
        d.ew(
            r,
            5,
            at(src, 0, 0),
            at(src, 0, 0),
            at(&self.x, 0, 0),
            CODEC_POSITIONS * w,
            w,
        );
        self.record_blocks(d, r, slot);
    }
    fn record_blocks<'a>(&'a self, d: &'a Device, r: &mut Recording<'a>, slot: usize) {
        let (n, w) = (CODEC_POSITIONS, CODEC_WIDTH);
        d.record(
            r,
            &self.blocks,
            &self.scratch,
            &self.x,
            n,
            self.next.get(),
            true,
        );
        let x = &self.x;
        d.norm(
            r,
            at(x, 0, w),
            Some((&self.norm, 0, 0)),
            at(x, 0, w),
            n,
            w,
            self.eps,
            0,
        );
        let rows = self.out.rows;
        d.mv_full(
            r,
            &self.out,
            at(x, 0, w),
            at(self.output[slot].buffer(), 0, rows),
            n,
            0,
            rows,
            Some((&self.out_bias, 0)),
            false,
        );
    }
    /// The output of the last frame recorded into `slot`, `[6][1282]`.
    pub fn read(&self, d: &Device, slot: usize, out: &mut [f32]) -> Result<()> {
        d.gpu.read_mapped(&self.output[slot], 0, out)
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
            .map(|b| GpuBlock::upload(g, b, POSITIONS, d.k.mvc.is_some()))
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
            // The codebooks' logits read the same rows and write separate places.
            if i > 0 {
                r.together();
            }
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

impl GpuDepth {
    /// Records a pass whose positions take the frame's codes from `codes` on the device
    /// (undecided ones the mask), with its logits for `wanted` into `logits`.
    #[allow(clippy::too_many_arguments)]
    fn record_pass<'a>(
        &'a self,
        d: &'a Device,
        r: &mut Recording<'a>,
        project: Option<&'a Buffer>,
        codes: &'a Buffer,
        mask: &'a Buffer,
        wanted: &[usize],
        codes_per: usize,
        logits: &'a Buffer,
    ) {
        if let Some(state) = project {
            d.mv_full(
                r,
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
        r.push(
            &d.k.gather,
            vec![&self.code_embd.w, codes, &self.base, mask, &self.x],
            vec![0, WIDTH as u32, codes_per as u32],
            (POSITIONS * WIDTH).div_ceil(64) as u32,
        );
        d.record(r, &self.blocks, &self.scratch, &self.x, POSITIONS, 0, false);
        let (x, eps) = (&self.x, self.eps);
        d.norm(
            r,
            at(x, 0, WIDTH),
            Some((&self.norm, 0, 0)),
            at(x, 0, WIDTH),
            POSITIONS,
            WIDTH,
            eps,
            0,
        );
        d.norm(
            r,
            at(x, 0, WIDTH),
            Some((&self.head_norm, 0, WIDTH)),
            at(x, 0, WIDTH),
            POSITIONS,
            WIDTH,
            eps,
            0,
        );
        for (i, &cb) in wanted.iter().enumerate() {
            if i > 0 {
                r.together();
            }
            d.mv_full(
                r,
                &self.code_embd,
                at(x, cb * WIDTH, WIDTH),
                at(logits, i * codes_per, codes_per),
                1,
                cb * codes_per,
                codes_per,
                None,
                false,
            );
        }
    }
}

/// A whole frame on the device in one submission: three depth passes, each followed by
/// sampling on the device (the host draws the codebook order and the uniforms beforehand,
/// in the order its own sampler uses them), the backbone's feedback and next step, and the
/// frame's codec. The host reads the twelve codes and the codec's output.
pub struct FrameGpu {
    /// Per frame slot, the frame's codes (u32 bits; undecided: all ones) and the uniforms
    /// per codebook (frames alternate slots, so one is read while the next runs).
    codes: [Mapped; 2],
    draws: [Mapped; 2],
    /// Codes chosen earlier in the request, `[12][2017]`.
    seen: Buffer,
    logits: Buffer,
    mask: Buffer,
    feedback: Mat,
    fsq: Buffer,
    fsq_out: Mat,
    fsq_bias: Buffer,
    codec_in: Mat,
    codec_bias: Buffer,
    fb: Buffer,
    q: Buffer,
    z: Buffer,
    zz: Buffer,
    codes_per: usize,
}

impl FrameGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        d: &Device,
        mask: &[f32],
        feedback: &Linear,
        fsq: &[f32],
        fsq_out: (&Linear, &[f32]),
        codec_in: (&Linear, &[f32]),
        codes_per: usize,
    ) -> Result<Self> {
        let g = &d.gpu;
        Ok(Self {
            codes: [g.mapped(12 * 4)?, g.mapped(12 * 4)?],
            draws: [g.mapped(12 * 4)?, g.mapped(12 * 4)?],
            seen: g.buffer((12 * codes_per * 4) as u64)?,
            logits: g.buffer((4 * codes_per * 4) as u64)?,
            mask: floats(g, mask)?,
            feedback: Mat::upload(g, feedback)?,
            fsq: floats(g, fsq)?,
            fsq_out: Mat::upload(g, fsq_out.0)?,
            fsq_bias: floats(g, fsq_out.1)?,
            codec_in: Mat::upload(g, codec_in.0)?,
            codec_bias: floats(g, codec_in.1)?,
            fb: g.buffer(2048 * 4)?,
            q: g.buffer(48 * 4)?,
            z: g.buffer((fsq_out.0.rows() * 4) as u64)?,
            zz: g.buffer((codec_in.0.rows() * 4) as u64)?,
            codes_per,
        })
    }
    /// A new request: no codes seen.
    pub fn reset(&self, d: &Device) -> Result<()> {
        let count = 12 * self.codes_per;
        for start in (0..count).step_by(ZERO_FLOATS) {
            let n = ZERO_FLOATS.min(count - start);
            d.gpu
                .write_floats(&self.seen, (start * 4) as u64, &d.staging, &ZEROS[..n])?;
        }
        Ok(())
    }
    /// Starts a frame on the device without waiting for it: `order` the codebooks in
    /// sampling order (four per pass), `uniforms` per codebook, `slot` alternating between
    /// frames (the previous frame's slot may still be in use). It runs after the frames
    /// started before it; `finish` waits for it and reads its codes and codec output.
    #[allow(clippy::too_many_arguments)]
    pub fn begin<'a>(
        &'a self,
        d: &'a Device,
        depth: &'a GpuDepth,
        backbone: &'a GpuBackbone,
        codec: &'a GpuCodec,
        order: &[usize; 12],
        uniforms: &[f32; 12],
        slot: usize,
    ) -> Result<InFlight<'a>> {
        if backbone.next.get() + 1 > backbone.positions
            || codec.next.get() + CODEC_POSITIONS > codec.positions
        {
            return Err("audio_context_exhausted");
        }
        let (codes, draws) = (&self.codes[slot], &self.draws[slot]);
        d.gpu
            .write_mapped(codes, 0, &[f32::from_bits(u32::MAX); 12])?;
        d.gpu.write_mapped(draws, 0, uniforms)?;
        let r = self.record(d, depth, backbone, codec, order, slot);
        // SAFETY: every buffer recorded belongs to `self`, `d`, `depth`, `backbone` or
        // `codec`, which the returned `InFlight` borrows; it waits for the pass when dropped.
        let passing = unsafe { r.begin(&d.gpu)? };
        backbone.next.set(backbone.next.get() + 1);
        codec.next.set(codec.next.get() + CODEC_POSITIONS);
        Ok(InFlight {
            passing,
            slot,
            _uses: std::marker::PhantomData,
        })
    }
    /// The frame's steps (`begin`).
    fn record<'a>(
        &'a self,
        d: &'a Device,
        depth: &'a GpuDepth,
        backbone: &'a GpuBackbone,
        codec: &'a GpuCodec,
        order: &[usize; 12],
        slot: usize,
    ) -> Recording<'a> {
        let (codes, draws) = (&self.codes[slot], &self.draws[slot]);
        let per = self.codes_per;
        let mut r = Recording::new();
        let codes_buf = codes.buffer();
        for pass in 0..3 {
            let wanted = &order[pass * 4..pass * 4 + 4];
            let project = (pass == 0).then_some(&backbone.state);
            depth.record_pass(
                d,
                &mut r,
                project,
                codes_buf,
                &self.mask,
                wanted,
                per,
                &self.logits,
            );
            let mut constants = vec![per as u32, 128];
            constants.extend(wanted.iter().map(|&cb| cb as u32));
            constants.push(0);
            r.push(
                &d.k.sample,
                vec![&self.logits, &self.seen, codes_buf, draws.buffer()],
                constants,
                4,
            );
        }
        // The backbone's input (the sum of the codes' feedback embeddings) and the codec's
        // (the codes' FSQ values), both from the codes.
        r.push(
            &d.k.gather,
            vec![&self.feedback.w, codes_buf, &d.dummy, &d.dummy, &self.fb],
            vec![1, 2048, per as u32],
            2048u32.div_ceil(64),
        );
        r.together();
        r.push(
            &d.k.gather,
            vec![&self.feedback.w, codes_buf, &self.fsq, &d.dummy, &self.q],
            vec![2, 0, per as u32],
            1,
        );
        // The backbone step and the codec read the codes' values and nothing of each other's:
        // they run side by side.
        let mut step = Recording::new();
        backbone.record_step(d, &mut step, &self.fb);
        let mut c = Recording::new();
        let rows = self.fsq_out.rows;
        d.mv_full(
            &mut c,
            &self.fsq_out,
            at(&self.q, 0, 48),
            at(&self.z, 0, rows),
            1,
            0,
            rows,
            Some((&self.fsq_bias, 0)),
            false,
        );
        let rows2 = self.codec_in.rows;
        d.mv_full(
            &mut c,
            &self.codec_in,
            at(&self.z, 0, rows),
            at(&self.zz, 0, rows2),
            1,
            0,
            rows2,
            Some((&self.codec_bias, 0)),
            false,
        );
        codec.record_from(d, &mut c, &self.zz, slot);
        r.interleave(step, c);
        r
    }
    /// Waits for a frame from `begin` and reads its codes and codec output (`[6][1282]`).
    pub fn finish(
        &self,
        d: &Device,
        codec: &GpuCodec,
        frame: InFlight,
        codes: &mut [usize; 12],
        polar: &mut [f32],
    ) -> Result<()> {
        let slot = frame.slot;
        frame.passing.wait()?;
        let mut bits = [0f32; 12];
        d.gpu.read_mapped(&self.codes[slot], 0, &mut bits)?;
        for (c, b) in codes.iter_mut().zip(bits) {
            *c = b.to_bits() as usize;
        }
        codec.read(d, slot, polar)
    }
}

/// A frame started by `FrameGpu::begin`, running on the device until `finish` (dropping
/// it waits too).
pub struct InFlight<'a> {
    passing: Passing,
    slot: usize,
    _uses: std::marker::PhantomData<&'a ()>,
}

/// The text head (tied embeddings) on the device, for recognition.
/// Greedy text decoding on the device: each step takes the head's logits of the backbone
/// state, their argmax as the token (into `ids`), its embedding (the head's row) and a
/// backbone step on it, with no host work between steps; steps are pipelined.
pub struct GpuDecoder {
    /// The tied embeddings and output head, `[vocabulary][2048]`.
    pub head: Mat,
    logits: Buffer,
    /// The tokens chosen, one per step.
    ids: Mapped,
    row: Buffer,
    pub capacity: usize,
}

impl GpuDecoder {
    pub fn upload(d: &Device, embed: &Linear, capacity: usize) -> Result<Self> {
        let head = Mat::upload(&d.gpu, embed)?;
        Ok(Self {
            logits: d.gpu.buffer((head.rows * 4) as u64)?,
            ids: d.gpu.mapped((capacity * 4) as u64)?,
            row: d.gpu.buffer(2048 * 4)?,
            head,
            capacity,
        })
    }
    /// Starts step `k`: token `k` from the backbone's state, then the backbone on it.
    pub fn begin<'a>(
        &'a self,
        d: &'a Device,
        backbone: &'a GpuBackbone,
        k: usize,
    ) -> Result<Decoding<'a>> {
        if k >= self.capacity || backbone.next.get() + 1 > backbone.positions {
            return Err("audio_context_exhausted");
        }
        let mut r = Recording::new();
        let head = &self.head;
        d.mv(
            &mut r,
            head,
            at(&backbone.state, 0, 2048),
            at(&self.logits, 0, head.rows),
            1,
        );
        let ids = self.ids.buffer();
        r.push(
            &d.k.argmax,
            vec![&self.logits, ids],
            vec![head.rows as u32, k as u32],
            1,
        );
        r.push(
            &d.k.row,
            vec![
                &head.w,
                head.scales.as_ref().unwrap_or(&d.dummy),
                ids,
                &self.row,
            ],
            vec![head.cols as u32, head.kind, k as u32],
            head.cols.div_ceil(64) as u32,
        );
        backbone.record_step(d, &mut r, &self.row);
        // SAFETY: every buffer recorded belongs to `self`, `d` or `backbone`, which the
        // returned `Decoding` borrows; it waits for the pass when dropped.
        let passing = unsafe { r.begin(&d.gpu)? };
        backbone.next.set(backbone.next.get() + 1);
        Ok(Decoding {
            passing,
            k,
            _uses: std::marker::PhantomData,
        })
    }
    /// Waits for a step and returns its token.
    pub fn finish(&self, d: &Device, step: Decoding) -> Result<u32> {
        let k = step.k;
        step.passing.wait()?;
        let mut bits = [0f32; 1];
        d.gpu.read_mapped(&self.ids, (k * 4) as u64, &mut bits)?;
        Ok(bits[0].to_bits())
    }
}

/// A decoding step started by `GpuDecoder::begin` (dropping it waits for it).
pub struct Decoding<'a> {
    passing: Passing,
    k: usize,
    _uses: std::marker::PhantomData<&'a ()>,
}

#[cfg(test)]
mod tests {
    use super::*;

    impl GpuBackbone {
        fn read_state(&self, d: &Device) -> Vec<f32> {
            let mut out = vec![0f32; 2048];
            let stage = d.gpu.readback(2048 * 4).unwrap();
            d.gpu.read_floats(&self.state, 0, &stage, &mut out).unwrap();
            out
        }
    }

    /// The same prompt twenty times: the backbone state and the depth logits must repeat
    /// bit for bit (run by hand with the model: `KUNO_AUDIO_MODELS=dir`).
    #[test]
    #[ignore]
    fn repeats_exactly() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut bb = GpuBackbone::upload(&d, &m.backbone, &m.out_norm, 4096).unwrap();
        let depth = GpuDepth::upload(
            &d,
            &m.depth,
            &m.proj_in.0,
            &m.proj_in.1,
            &m.code_embd,
            &m.depth_norm,
            &m.head_norm,
            2017,
        )
        .unwrap();
        let rows = m.embed_tokens(&(0..65u32).map(|i| 1000 + i * 37).collect::<Vec<_>>());
        let add = m.depth_inputs(&[None; 12]);
        let (mut first_state, mut first_logits) = (None, None);
        let (mut state_diff, mut logit_diff) = (0, 0);
        for _ in 0..20 {
            bb.reset(&d).unwrap();
            bb.forward(&d, &rows).unwrap();
            bb.forward(&d, &rows[..2048]).unwrap();
            let state = bb.read_state(&d);
            let mut logits = vec![0f32; 4 * 2017];
            depth
                .pass(&d, Some(&bb.state), &add, &[0, 1, 2, 3], 2017, &mut logits)
                .unwrap();
            let fs = first_state.get_or_insert_with(|| state.clone());
            state_diff += usize::from(*fs != state);
            let fl = first_logits.get_or_insert_with(|| logits.clone());
            logit_diff += usize::from(*fl != logits);
        }
        eprintln!("runs with a different state: {state_diff}, different logits: {logit_diff}");
        assert_eq!((state_diff, logit_diff), (0, 0));
    }

    /// The frame loop's GPU work (three depth passes and a backbone step per frame, inputs
    /// derived from the logits as sampling would) repeats exactly over 150 frames.
    #[test]
    #[ignore]
    fn frame_loop_repeats_exactly() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut bb = GpuBackbone::upload(&d, &m.backbone, &m.out_norm, 4096).unwrap();
        let depth = GpuDepth::upload(
            &d,
            &m.depth,
            &m.proj_in.0,
            &m.proj_in.1,
            &m.code_embd,
            &m.depth_norm,
            &m.head_norm,
            2017,
        )
        .unwrap();
        let rows = m.embed_tokens(&(0..65u32).map(|i| 1000 + i * 37).collect::<Vec<_>>());
        let run = |bb: &mut GpuBackbone| -> Vec<(usize, Vec<u32>)> {
            bb.reset(&d).unwrap();
            bb.forward(&d, &rows).unwrap();
            let mut trace = Vec::new();
            for frame in 0..150 {
                let mut codes = [None; 12];
                let mut logits = vec![0f32; 4 * 2017];
                for pass in 0..3 {
                    let wanted: Vec<usize> = (pass * 4..pass * 4 + 4).collect();
                    let project = (pass == 0).then_some(&bb.state);
                    depth
                        .pass(
                            &d,
                            project,
                            &m.depth_inputs(&codes),
                            &wanted,
                            2017,
                            &mut logits,
                        )
                        .unwrap();
                    trace.push((frame, logits.iter().map(|v| v.to_bits()).collect()));
                    for (k, &cb) in wanted.iter().enumerate() {
                        let l = &logits[k * 2017..][..2016];
                        codes[cb] = Some(crate::argmax(l));
                    }
                }
                let fb = m.feedback_row(&codes.map(|c| c.unwrap()));
                bb.forward(&d, &fb).unwrap();
            }
            trace
        };
        let first = run(&mut bb);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&first, &mut h);
        eprintln!("TRACE HASH {:016x}", std::hash::Hasher::finish(&h));
        for attempt in 1..2 {
            let again = run(&mut bb);
            if let Some(i) = first.iter().zip(&again).position(|(a, b)| a != b) {
                panic!(
                    "attempt {attempt}: first difference at frame {} (pass {})",
                    first[i].0,
                    i % 3
                );
            }
        }
    }

    /// Every uploaded matrix reads back byte for byte (run in several processes by hand).
    #[test]
    #[ignore]
    fn uploads_read_back_exactly() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut bad = 0;
        let mut checked = 0usize;
        for (bi, b) in m.backbone.iter().enumerate() {
            let mut mats = vec![&b.gate, &b.up, &b.down];
            match &b.op {
                crate::block::Op::Attention(a) => mats.extend([&a.q, &a.k, &a.v, &a.o]),
                crate::block::Op::Conv(c) => mats.extend([&c.in_proj, &c.out_proj]),
            }
            for lin in mats {
                let up = Mat::upload(&d.gpu, lin).unwrap();
                let crate::tensor::Linear::Q8 { w, .. } = lin else {
                    continue;
                };
                let values = d.gpu.download(&up.w, 0, up.w.bytes() as usize).unwrap();
                let again = d.gpu.download(&up.w, 0, up.w.bytes() as usize).unwrap();
                if again != values {
                    eprintln!("block {bi}: two readbacks of the same buffer differ");
                }
                let scales = d
                    .gpu
                    .download(up.scales.as_ref().unwrap(), 0, (w.len() / 34) * 2)
                    .unwrap();
                for (k, block) in w.chunks_exact(34).enumerate() {
                    if block[2..] != values[k * 32..][..32] || block[..2] != scales[k * 2..][..2] {
                        bad += 1;
                        if bad < 5 {
                            eprintln!("block {bi}: mismatch at block {k} of {}", w.len() / 34);
                        }
                    }
                }
                checked += w.len();
            }
        }
        eprintln!("UPLOAD checked {} MB, {bad} bad blocks", checked >> 20);
        assert_eq!(bad, 0);
    }

    /// The GPU prompt state against the CPU's (run in several processes by hand).
    /// `bench_mmq` checks the prompt products themselves against an f64 reference.
    #[test]
    #[ignore]
    fn prompt_state_matches_cpu() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut bb = GpuBackbone::upload(&d, &m.backbone, &m.out_norm, 4096).unwrap();
        let ids: Vec<u32> = (0..64u32).map(|i| 1000 + i * 37).collect();
        bb.reset(&d).unwrap();
        bb.forward(&d, &m.embed_tokens(&ids)).unwrap();
        let gpu = bb.read_state(&d);
        let mut rows = m.embed_tokens(&ids);
        let mut states = vec![crate::block::State::default(); m.backbone.len()];
        let cpu = m.backbone_cpu(&mut rows, &mut states);
        let worst = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&gpu.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &mut h);
        eprintln!(
            "STATE {:016x} worst {worst:e} of {scale:e}",
            std::hash::Hasher::finish(&h)
        );
        // Where the device has matrix units the prompt's products round both sides to
        // halves (mmq), about 2e-4 of the scale here; f32 products stay under 1e-4.
        let bound = if d.k.mmq.is_some() { 1e-3 } else { 1e-4 };
        assert!(
            worst < bound * scale.max(1.0),
            "the GPU state is off by {worst}"
        );
    }

    /// The GPU encoder against the CPU's on 150 rows (three chunks), twice (run by hand).
    #[test]
    #[ignore]
    fn encoder_matches_cpu() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let enc = GpuEncoder::upload(&d, &m.encoder, 1024).unwrap();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let input: Vec<f32> = (0..150 * ENCODER_WIDTH)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        let mut gpu = input.clone();
        enc.forward(&d, &mut gpu).unwrap();
        let mut again = input.clone();
        enc.forward(&d, &mut again).unwrap();
        assert!(gpu == again, "the GPU encoder does not repeat");
        let mut cpu = input;
        let mut states = vec![crate::block::State::default(); m.encoder.len()];
        for (b, s) in m.encoder.iter().zip(&mut states) {
            b.forward(&mut cpu, s, true);
        }
        let worst = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(0f32, f32::max);
        eprintln!("ENCODER worst {worst:e} of {scale:e}");
        assert!(
            worst < 1e-3 * scale.max(1.0),
            "the GPU encoder is off by {worst}"
        );
    }

    /// Poisoning each scratch buffer between runs must not change the result (a kernel
    /// that reads what no step wrote would pick the poison up).
    #[test]
    #[ignore]
    fn scratch_is_written_before_read() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut bb = GpuBackbone::upload(&d, &m.backbone, &m.out_norm, 4096).unwrap();
        let rows = m.embed_tokens(&(0..100u32).map(|i| 1000 + i * 37).collect::<Vec<_>>());
        let run = |bb: &mut GpuBackbone| {
            bb.reset(&d).unwrap();
            bb.forward(&d, &rows).unwrap();
            bb.read_state(&d)
        };
        let clean = run(&mut bb);
        let poison = |b: &Buffer| {
            let n = (b.bytes() / 4) as usize;
            let stage = d.gpu.readback(4 * 65536).unwrap();
            let junk: Vec<f32> = (0..65536).map(|i| (i as f32 * 0.37).sin() * 1e3).collect();
            for start in (0..n).step_by(65536) {
                let k = 65536.min(n - start);
                d.gpu
                    .write_floats(b, (start * 4) as u64, &stage, &junk[..k])
                    .unwrap();
            }
        };
        let names = [
            "h", "op", "q", "att", "bcx", "gates", "ga", "ua", "scores", "x",
        ];
        let mut bad = Vec::new();
        for (k, name) in names.iter().enumerate() {
            {
                let s = &bb.scratch;
                let b = [
                    &s.h, &s.op, &s.q, &s.att, &s.bcx, &s.gates, &s.ga, &s.ua, &s.scores, &bb.x,
                ][k];
                poison(b);
            }
            if run(&mut bb) != clean {
                bad.push(name.to_string());
            }
        }
        for i in 0..bb.blocks.len() {
            let attention = if let GpuOp::Attention { keys, values, .. } = &bb.blocks[i].op {
                poison(keys);
                poison(values);
                true
            } else {
                false
            };
            if attention && run(&mut bb) != clean {
                bad.push(format!("kv cache of block {i}"));
            }
        }
        eprintln!("POISONED AND CHANGED: {bad:?}");
        assert!(bad.is_empty());
    }

    /// Speech through the public path repeats exactly for a seed.
    #[test]
    #[ignore]
    fn speech_repeats_exactly() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::open_on(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
            Some(&pci),
        )
        .unwrap();
        assert_ne!(m.device(), "cpu");
        let text = "Kuno now speaks on the graphics card. The backbone and the depth decoder run \
                    on the GPU, while the codec turns each frame into sound on the processor.";
        for _ in 0..3 {
            let pcm = m
                .speak(text, "American English", "female", 5, 2000, |_| true)
                .unwrap();
            let bits: Vec<u32> = pcm.iter().map(|v| v.to_bits()).collect();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&bits, &mut h);
            eprintln!(
                "SPEECH HASH {:016x} ({} samples)",
                std::hash::Hasher::finish(&h),
                pcm.len()
            );
        }
    }

    /// The codec's GPU step on its own and the backbone step on its own (run by hand).
    #[test]
    #[ignore]
    fn bench_codec() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::load(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
        )
        .unwrap();
        let d = Device::open(&pci).unwrap();
        let mut codec = GpuCodec::upload(
            &d,
            &m.codec,
            &m.codec_norm,
            &m.codec_out.0,
            &m.codec_out.1,
            12288,
        )
        .unwrap();
        for (i, b) in m.codec.iter().enumerate() {
            let kind = match &b.op {
                crate::block::Op::Attention(a) => format!(
                    "attention {}x{} heads {} window {:?}",
                    a.q.rows(),
                    a.q.cols(),
                    a.heads,
                    a.window
                ),
                crate::block::Op::Conv(c) => {
                    format!("conv {}x{}", c.in_proj.rows(), c.in_proj.cols())
                }
            };
            eprintln!(
                "codec block {i}: {kind}, ffn {}x{}",
                b.gate.rows(),
                b.gate.cols()
            );
        }
        codec.reset(&d).unwrap();
        let input = floats(&d.gpu, &[0.1f32; 512]).unwrap();
        let mut times = Vec::new();
        for _ in 0..60 {
            let mut r = Recording::new();
            codec.record_from(&d, &mut r, &input, 0);
            let t0 = std::time::Instant::now();
            r.run(&d.gpu).unwrap();
            times.push(t0.elapsed().as_secs_f64());
            codec.next.set(codec.next.get() + 6);
        }
        let mut bb = GpuBackbone::upload(&d, &m.backbone, &m.out_norm, 4096).unwrap();
        bb.reset(&d).unwrap();
        let row = floats(&d.gpu, &vec![0.01f32; 2048]).unwrap();
        let mut steps = Vec::new();
        for _ in 0..40 {
            let mut r = Recording::new();
            bb.record_step(&d, &mut r, &row);
            let t0 = std::time::Instant::now();
            r.run(&d.gpu).unwrap();
            steps.push(t0.elapsed().as_secs_f64());
            bb.next.set(bb.next.get() + 1);
        }
        steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let weights: usize = m.backbone.iter().map(crate::block::Block::bytes).sum();
        let count = {
            let mut r = Recording::new();
            bb.record_step(&d, &mut r, &row);
            (r.steps.len(), r.independent.iter().filter(|i| !**i).count())
        };
        eprintln!(
            "BACKBONE step median {:.2} ms for {} MB of weights ({} steps, {} barriers)",
            steps[20] * 1e3,
            weights >> 20,
            count.0,
            count.1
        );
        let depth = GpuDepth::upload(
            &d,
            &m.depth,
            &m.proj_in.0,
            &m.proj_in.1,
            &m.code_embd,
            &m.depth_norm,
            &m.head_norm,
            2017,
        )
        .unwrap();
        let inputs = m.depth_inputs(&[None; 12]);
        let mut out = vec![0f32; 4 * 2017];
        let mut passes = Vec::new();
        for _ in 0..30 {
            let t0 = std::time::Instant::now();
            depth
                .pass(&d, None, &inputs, &[0, 1, 2, 3], 2017, &mut out)
                .unwrap();
            passes.push(t0.elapsed().as_secs_f64());
        }
        passes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let depth_bytes: usize = m.depth.iter().map(crate::block::Block::bytes).sum();
        eprintln!(
            "DEPTH pass median {:.2} ms for {} MB of block weights",
            passes[15] * 1e3,
            depth_bytes >> 20
        );
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("CODEC pass median {:.2} ms ({} steps)", times[30] * 1e3, {
            let mut r = Recording::new();
            codec.record_from(&d, &mut r, &input, 0);
            r.steps.len()
        });
    }

    /// Where a frame's device time goes: each step timed, summed per kernel and shape (run by
    /// hand, `KUNO_AUDIO_MODELS` the model directory).
    #[test]
    #[ignore]
    fn profile_frame() {
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MODELS").unwrap());
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let m = crate::Lfm3Audio::open_on(
            &dir.join("LFM3-Audio-2B-Q8_0.gguf"),
            &dir.join("mmproj-LFM3-Audio-2B-F16.gguf"),
            Some(&pci),
        )
        .unwrap();
        let g = m.gpu.as_ref().expect("on the GPU");
        let d = &g.device;
        let mut backbone = g.backbone.lock().unwrap();
        let mut codec = g.codec.lock().unwrap();
        backbone.reset(d).unwrap();
        codec.reset(d).unwrap();
        backbone.forward(d, &m.embed_tokens(&[1, 2, 3, 4])).unwrap();
        g.frame.reset(d).unwrap();
        let order = [3, 7, 1, 0, 11, 5, 9, 2, 4, 8, 6, 10];
        let k = &d.k;
        let mut names: Vec<(&Kernel, &str)> = vec![
            (&k.mv, "mv"),
            (&k.mvb, "mvb"),
            (&k.mv1, "mv1"),
            (&k.mv1s, "mv1s"),
            (&k.mv1g, "mv1g"),
            (&k.sample, "sample"),
            (&k.gather, "gather"),
            (&k.norm, "norm"),
            (&k.addnorm, "addnorm"),
            (&k.normrope, "normrope"),
            (&k.rope, "rope"),
            (&k.attend, "attend"),
            (&k.conv, "conv"),
            (&k.ew, "ew"),
        ];
        if let Some(c) = &k.mvc {
            names.push((c, "mvc"));
        }
        if let Some(c) = &k.mvcg {
            names.push((c, "mvcg"));
        }
        if let Some(c) = &k.mmq {
            names.push((c, "mmq"));
        }
        let mut totals: std::collections::BTreeMap<String, (f64, usize)> = Default::default();
        let mut frame_times = Vec::new();
        for frame in 0..20 {
            let r = g
                .frame
                .record(d, &g.depth, &backbone, &codec, &order, frame % 2);
            let times = r.profile(&d.gpu).unwrap();
            backbone.next.set(backbone.next.get() + 1);
            codec.next.set(codec.next.get() + CODEC_POSITIONS);
            frame_times.push(times.iter().sum::<f64>());
            if frame < 10 {
                continue;
            }
            for ((kernel, _, constants, _), t) in r.steps.iter().zip(&times) {
                let name = names
                    .iter()
                    .find(|(k, _)| std::ptr::eq(*k, *kernel))
                    .map_or("?", |n| n.1);
                let label = if name.starts_with("mv") {
                    format!("{name} {}x{} n{}", constants[0], constants[1], constants[2])
                } else {
                    name.to_string()
                };
                let e = totals.entry(label).or_default();
                e.0 += t / 10.0;
                e.1 += 1;
            }
        }
        let mut rows: Vec<_> = totals.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "PROFILE frame {:.2} ms on the device (median of 20), {} steps",
            frame_times[10] / 1e6,
            g.frame
                .record(d, &g.depth, &backbone, &codec, &order, 0)
                .steps
                .len()
        );
        for (label, (ns, count)) in rows {
            eprintln!(
                "PROFILE {label:28} {:7.1} us a frame, {:4} steps, {:5.1} us each",
                ns / 1e3,
                count / 10,
                ns / 1e3 / (count / 10) as f64
            );
        }
    }

    /// The device sampler picks what the host sampler picks: random logits, some codes
    /// seen, random uniforms (run by hand; f32 against f64 may differ in rare ties).
    #[test]
    #[ignore]
    fn sampler_matches_the_host() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let d = Device::open(&pci).unwrap();
        let per = 2017;
        let logits = d.gpu.buffer((4 * per * 4) as u64).unwrap();
        let seen = d.gpu.buffer((12 * per * 4) as u64).unwrap();
        let codes = d.gpu.mapped(12 * 4).unwrap();
        let draws = d.gpu.mapped(12 * 4).unwrap();
        let stage = d.gpu.readback((12 * per * 4) as u64).unwrap();
        let mut rng = crate::Rng::new(7);
        let (mut same, mut total) = (0, 0);
        for case in 0..100 {
            let l: Vec<f32> = (0..4 * per)
                .map(|_| {
                    (rng.uniform() as f32 - 0.5) * 12.0 + if case % 3 == 0 { 3.0 } else { 0.0 }
                })
                .collect();
            let s: Vec<f32> = (0..12 * per)
                .map(|_| f32::from_bits(u32::from(rng.uniform() < 0.05)))
                .collect();
            let u: Vec<f32> = (0..12).map(|_| rng.uniform() as f32).collect();
            d.gpu.write_floats(&logits, 0, &stage, &l).unwrap();
            d.gpu.write_floats(&seen, 0, &stage, &s).unwrap();
            d.gpu.write_mapped(&draws, 0, &u).unwrap();
            let wanted = [3usize, 7, 0, 11];
            let mut r = Recording::new();
            let mut c = vec![per as u32, 128];
            c.extend(wanted.iter().map(|&w| w as u32));
            c.push(0);
            r.push(
                &d.k.sample,
                vec![&logits, &seen, codes.buffer(), draws.buffer()],
                c,
                4,
            );
            r.run(&d.gpu).unwrap();
            let mut got = [0f32; 12];
            d.gpu.read_mapped(&codes, 0, &mut got).unwrap();
            for (i, &cb) in wanted.iter().enumerate() {
                let mut lv = l[i * per..][..per].to_vec();
                for (j, v) in lv.iter_mut().enumerate() {
                    if s[cb * per + j].to_bits() != 0 {
                        *v = if *v > 0.0 { *v / 1.1 } else { *v * 1.1 };
                    }
                }
                let mut host = crate::Rng::fixed(f64::from(u[cb]));
                let want = host.sample_top_k(&lv, 128, 1.0);
                total += 1;
                same += usize::from(got[cb].to_bits() as usize == want);
            }
        }
        eprintln!("SAMPLER {same} of {total} picks match the host");
        assert!(same * 100 >= total * 98);
    }

    /// Matrix kernels on the model's shapes (run by hand on a GPU):
    /// `KUNO_AUDIO_BENCH_GPU=0000:04:00.0 cargo test --release -p kuno-audio --features gpu
    /// bench_mv -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_mv() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let d = Device::open(&pci).unwrap();
        for (kind, rows, cols, n) in [
            (0u32, 7680usize, 2048usize, 1usize),
            (0, 2048, 2048, 1),
            (0, 6144, 2048, 1),
            (0, 512, 2048, 1),
            (0, 2048, 7680, 1),
            (1, 2816, 1024, 12),
            (1, 1024, 1024, 12),
            (1, 1024, 2816, 12),
            (0, 7680, 2048, 64),
        ] {
            let bytes = rows * cols * if kind == 0 { 1 } else { 2 };
            let m = Mat {
                w: d.gpu.buffer(bytes as u64).unwrap(),
                scales: Some(d.gpu.buffer((rows * cols / 16).max(16) as u64).unwrap()),
                kind,
                rows,
                cols,
                tiled: false,
            };
            let x = d.gpu.buffer((n * cols * 4) as u64).unwrap();
            let y = d.gpu.buffer((n * rows * 4) as u64).unwrap();
            for (name, kernel, per) in [
                ("mv", &d.k.mv, 1),
                ("mvb", &d.k.mvb, 4),
                ("mv1", &d.k.mv1, d.k.mv1_rows),
                ("mv1s", &d.k.mv1s, 1),
            ] {
                if name.starts_with("mv1") && n > 1 {
                    continue;
                }
                for together in [false, true] {
                    let mut r = Recording::new();
                    for _ in 0..50 {
                        r.push(
                            kernel,
                            vec![&m.w, m.scales.as_ref().unwrap(), &x, &y, &d.dummy],
                            vec![
                                rows as u32,
                                cols as u32,
                                n as u32,
                                kind,
                                0,
                                cols as u32,
                                0,
                                rows as u32,
                                0,
                                0,
                                0,
                            ],
                            rows.div_ceil(per) as u32,
                        );
                        if together {
                            r.together();
                        }
                    }
                    r.run(&d.gpu).unwrap();
                    // Device time (submission excluded).
                    let each = r.profile(&d.gpu).unwrap().iter().sum::<f64>() / 50.0 / 1e9;
                    eprintln!(
                        "{name:4} {} kind {kind} {rows}x{cols} n {n:2}: {:6.1} us, {:4.0} GB/s",
                        if together {
                            "overlapped"
                        } else {
                            "in turn    "
                        },
                        each * 1e6,
                        bytes as f64 / each / 1e9
                    );
                }
            }
        }
    }

    /// The cost of a dependent step: a trivial product many times in one submission, each
    /// waiting for the last or not (run by hand).
    #[test]
    #[ignore]
    fn bench_floor() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let d = Device::open(&pci).unwrap();
        let w = d.gpu.buffer(4096).unwrap();
        let x = d.gpu.buffer(4096).unwrap();
        let y = d.gpu.buffer(4096).unwrap();
        for count in [10usize, 100, 1000] {
            for together in [false, true] {
                let mut r = Recording::new();
                for _ in 0..count {
                    r.push(
                        &d.k.mv1,
                        vec![&w, &d.dummy, &x, &y, &d.dummy],
                        vec![16, 32, 1, 2, 0, 32, 0, 16, 0, 0, 0],
                        4,
                    );
                    if together {
                        r.together();
                    }
                }
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = std::time::Instant::now();
                    r.run(&d.gpu).unwrap();
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                eprintln!(
                    "FLOOR {count:4} steps {}: {:7.1} us in all, {:5.2} us each",
                    if together { "overlapped" } else { "in turn   " },
                    best * 1e6,
                    best * 1e6 / count as f64
                );
            }
        }
    }

    /// The tensor-core product (on tiled weights) against mvb on the depth decoder's shapes:
    /// the same sums (up to the inputs' rounding to halves) and their times (run by hand).
    #[test]
    #[ignore]
    fn bench_mvc() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let d = Device::open(&pci).unwrap();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (rows, cols, n) in [
            (2816usize, 1024usize, 12usize),
            (1024, 1024, 12),
            (256, 1024, 12),
            (1024, 2816, 12),
            (1024, 1024, 5),
            (16, 32, 12),
            (16, 32, 1),
        ] {
            let bits: Vec<u16> = (0..rows * cols)
                .map(|_| {
                    let v = next();
                    ((v & 0x8000) as u16)
                        | (((10 + (v >> 16) % 6) as u16) << 10)
                        | ((v >> 32) as u16 & 0x3ff)
                })
                .collect();
            let xs: Vec<f32> = (0..n * cols)
                .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
                .collect();
            let mut want = vec![0f64; n * rows];
            for t in 0..n {
                for r in 0..rows {
                    want[t * rows + r] = (0..cols)
                        .map(|c| {
                            f64::from(crate::tensor::f16(bits[r * cols + c]))
                                * f64::from(xs[t * cols + c])
                        })
                        .sum();
                }
            }
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            let m = Mat {
                w: filled(&d.gpu, &bytes).unwrap(),
                scales: None,
                kind: 1,
                rows,
                cols,
                tiled: false,
            };
            let x = floats(&d.gpu, &xs).unwrap();
            let y = d.gpu.buffer((n * rows * 4) as u64).unwrap();
            let back = d.gpu.readback((n * rows * 4) as u64).unwrap();
            let mut tiled_bits = Vec::with_capacity(bits.len());
            for rt in 0..rows / 16 {
                for ks in 0..cols / 16 {
                    for nn in 0..16 {
                        for kk in 0..16 {
                            tiled_bits.push(bits[(rt * 16 + nn) * cols + ks * 16 + kk]);
                        }
                    }
                }
            }
            let tiled_bytes: Vec<u8> = tiled_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            let tiled_w = filled(&d.gpu, &tiled_bytes).unwrap();
            let mut runs = vec![("mvb", &d.k.mvb, 4, &m.w)];
            if let Some(k) = &d.k.mvc {
                runs.push(("mvc", k, 16, &tiled_w));
            }
            for (name, kernel, per, weights) in runs {
                let mut r = Recording::new();
                for _ in 0..50 {
                    r.push(
                        kernel,
                        vec![weights, &d.dummy, &x, &y, &d.dummy],
                        vec![
                            rows as u32,
                            cols as u32,
                            n as u32,
                            1,
                            0,
                            cols as u32,
                            0,
                            rows as u32,
                            0,
                            0,
                            0,
                        ],
                        rows.div_ceil(per) as u32,
                    );
                }
                r.run(&d.gpu).unwrap();
                // Device time (submission excluded).
                let each = r.profile(&d.gpu).unwrap().iter().sum::<f64>() / 50.0 / 1e9;
                let mut got = vec![0f32; n * rows];
                d.gpu.read_floats(&y, 0, &back, &mut got).unwrap();
                let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(&want)
                    .fold(0f64, |a, (g, w)| a.max((f64::from(*g) - w).abs()));
                eprintln!(
                    "{name:8} {rows}x{cols} n {n:2}: {:6.1} us, {:4.0} GB/s, worst {:.1e} of {scale:.1}",
                    each * 1e6,
                    (rows * cols * 2) as f64 / each / 1e9,
                    worst
                );
                assert!(worst < 1e-3 * scale, "{name} differs");
            }
        }
    }

    /// The prompt-chunk product against an f64 reference and mvb on the backbone's and the
    /// encoder's shapes, Q8_0 and f16 (run by hand).
    #[test]
    #[ignore]
    fn bench_mmq() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let d = Device::open(&pci).unwrap();
        let Some(mmq) = &d.k.mmq else {
            eprintln!("no matrix units for mmq");
            return;
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Variants to compare: `mmq_<row tiles>_<columns>.spv` in KUNO_AUDIO_MMQ_DIR.
        let mut variants: Vec<(String, Kernel, usize)> = vec![];
        if let Ok(dir) = std::env::var("KUNO_AUDIO_MMQ_DIR") {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_stem().unwrap().to_string_lossy().into_owned();
                if !name.starts_with("mmq_") || path.extension().is_none_or(|e| e != "spv") {
                    continue;
                }
                let tiles: usize = name.split('_').nth(1).unwrap().parse().unwrap();
                let k = d
                    .gpu
                    .kernel(&std::fs::read(&path).unwrap(), 5, 11, Some(32))
                    .unwrap();
                variants.push((name, k, tiles * 16));
            }
        }
        variants.sort_by(|a, b| a.0.cmp(&b.0));
        for (q8, rows, cols, n) in [
            (true, 7680usize, 2048usize, 64usize),
            (true, 2048, 7680, 64),
            (true, 512, 2048, 37),
            (true, 2048, 2048, 100),
            (false, 2048, 768, 64),
            (false, 768, 2048, 50),
        ] {
            let linear = if q8 {
                let mut w = Vec::with_capacity(rows * cols / 32 * 34);
                for _ in 0..rows * cols / 32 {
                    let scale = 0x2000u16 | (next() as u16 & 0x3ff);
                    w.extend_from_slice(&scale.to_le_bytes());
                    w.extend((0..32).map(|_| next() as u8));
                }
                Linear::Q8 { w, rows, cols }
            } else {
                let w = (0..rows * cols)
                    .map(|_| {
                        let v = next();
                        ((v & 0x8000) as u16)
                            | (((10 + (v >> 16) % 6) as u16) << 10)
                            | ((v >> 32) as u16 & 0x3ff)
                    })
                    .collect();
                Linear::F16 { w, rows, cols }
            };
            let xs: Vec<f32> = (0..n * cols)
                .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
                .collect();
            let mut want = vec![0f64; n * rows];
            for r in 0..rows {
                let row = linear.row(r);
                for t in 0..n {
                    want[t * rows + r] = row
                        .iter()
                        .zip(&xs[t * cols..][..cols])
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum();
                }
            }
            let m = Mat::upload(&d.gpu, &linear).unwrap();
            let x = floats(&d.gpu, &xs).unwrap();
            let y = d.gpu.buffer((n * rows * 4) as u64).unwrap();
            let back = d.gpu.readback((n * rows * 4) as u64).unwrap();
            let constants = vec![
                rows as u32,
                cols as u32,
                n as u32,
                m.kind,
                0,
                cols as u32,
                0,
                rows as u32,
                0,
                0,
                0,
            ];
            let scales = m.scales.as_ref().unwrap_or(&d.dummy);
            let mut runs: Vec<(&str, &Kernel, usize)> =
                vec![("mvb", &d.k.mvb, 4), ("mmq", mmq, d.k.mmq_rows)];
            for (name, k, per) in &variants {
                runs.push((name, k, *per));
            }
            for (name, kernel, per) in runs {
                // A clean output, so a kernel that misses rows cannot pass on another's.
                d.gpu
                    .write_floats(&y, 0, &back, &vec![f32::NAN; n * rows])
                    .unwrap();
                let mut r = Recording::new();
                for _ in 0..20 {
                    if name != "mvb" {
                        r.push_grid(
                            kernel,
                            vec![&m.w, scales, &x, &y, &d.dummy],
                            constants.clone(),
                            rows.div_ceil(per) as u32,
                            n.div_ceil(64) as u32,
                        );
                    } else {
                        r.push(
                            &d.k.mvb,
                            vec![&m.w, scales, &x, &y, &d.dummy],
                            constants.clone(),
                            rows.div_ceil(4) as u32,
                        );
                    }
                }
                r.run(&d.gpu).unwrap();
                let each = r.profile(&d.gpu).unwrap().iter().sum::<f64>() / 20.0 / 1e9;
                let mut got = vec![0f32; n * rows];
                d.gpu.read_floats(&y, 0, &back, &mut got).unwrap();
                let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (f64::from(*g) - w).abs())
                    .fold(
                        0f64,
                        |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) },
                    );
                eprintln!(
                    "{name:9} {} {rows}x{cols} n {n:3}: {:7.1} us, {:5.2} TFLOPS, worst {:.1e} of {scale:.1}",
                    if q8 { "q8 " } else { "f16" },
                    each * 1e6,
                    2.0 * (rows * cols * n) as f64 / each / 1e12,
                    worst
                );
                assert!(worst < 2e-3 * scale, "{name} differs");
            }
        }
    }

    /// One-token product variants (`mv1v_<loads in flight>.spv` in KUNO_AUDIO_MV1_DIR) at
    /// 16- and 32-lane subgroups on the backbone's shapes, device time (run by hand).
    #[test]
    #[ignore]
    fn bench_mv1_variants() {
        let pci = std::env::var("KUNO_AUDIO_BENCH_GPU").unwrap_or("0000:04:00.0".into());
        let dir = std::path::PathBuf::from(std::env::var("KUNO_AUDIO_MV1_DIR").unwrap());
        let d = Device::open(&pci).unwrap();
        let mut kernels = vec![("mv1".to_string(), &d.k.mv1, d.k.mv1_rows)];
        let mut owned = Vec::new();
        for u in [1, 2, 4, 8] {
            let spirv = std::fs::read(dir.join(format!("mv1v_{u}.spv"))).unwrap();
            for lanes in [16u32, 32] {
                // Sizes the device lacks fail to build and are left out.
                if let Ok(k) = d.gpu.kernel(&spirv, 5, 11, Some(lanes)) {
                    owned.push((format!("u{u} x{lanes}"), k, 128 / lanes as usize));
                }
            }
        }
        for (name, k, rows) in &owned {
            kernels.push((name.clone(), k, *rows));
        }
        for (rows, cols) in [
            (7680usize, 2048usize),
            (6144, 2048),
            (2048, 2048),
            (512, 2048),
        ] {
            let m = Mat {
                w: d.gpu.buffer((rows * cols) as u64).unwrap(),
                scales: Some(d.gpu.buffer((rows * cols / 16) as u64).unwrap()),
                kind: 0,
                rows,
                cols,
                tiled: false,
            };
            let x = d.gpu.buffer((cols * 4) as u64).unwrap();
            let y = d.gpu.buffer((rows * 4) as u64).unwrap();
            for (name, kernel, per) in &kernels {
                let mut r = Recording::new();
                for _ in 0..100 {
                    r.push(
                        kernel,
                        vec![&m.w, m.scales.as_ref().unwrap(), &x, &y, &d.dummy],
                        vec![
                            rows as u32,
                            cols as u32,
                            1,
                            0,
                            0,
                            cols as u32,
                            0,
                            rows as u32,
                            0,
                            0,
                            0,
                        ],
                        rows.div_ceil(*per) as u32,
                    );
                }
                r.run(&d.gpu).unwrap();
                let mut t: Vec<f64> = r.profile(&d.gpu).unwrap();
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let each = t[50] / 1e9;
                eprintln!(
                    "MV1 {name:8} {rows}x{cols}: {:6.1} us, {:4.0} GB/s",
                    each * 1e6,
                    (rows * cols) as f64 * 1.0625 / each / 1e9
                );
            }
        }
    }

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
        for name in [
            "mv", "mvb", "mv1", "mv1s", "mv1g", "mvc", "mvc8", "mvcg", "mvc8g", "sample", "gather",
            "norm", "addnorm", "normrope", "argmax", "row", "mmq", "mmq8", "rope", "attend",
            "conv", "ew",
        ] {
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
