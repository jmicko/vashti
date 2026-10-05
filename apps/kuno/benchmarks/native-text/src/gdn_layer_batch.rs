//! Optional candidate-only GDN layer operator, starting at the GR mixed input.
//! Original packed model pages are immutable; private current/candidate owners
//! remain disjoint. This module has no graph hook, state swap or cursor commit.
use crate::{
    gdn_batch_pointwise as p, gdn_chunk as c,
    gdn_gpu::GdnLayout,
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::matmul::{BatchSpans, MatrixGeometry, packed_matmul},
    residual_batch_gpu::finite_span,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{
    TensorView,
    model::{AttentionKind, BoundGraph, GraphBindingId},
    ops::GdnShape,
};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::Zeroizing;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const MAX_BATCH: usize = 8;
pub const D: usize = 2560;
pub const TILE: usize = 8;
pub const MAIN_CAP: usize = 32 * 1024 * 1024;
pub const IO_CAP: u64 = 64 * 1024 * 1024;
pub const PUBLIC_CAP: usize = 128 * 1024 * 1024;
pub const DECODED_CAP: usize = 256 * 1024 * 1024;
pub const MAX_CONTEXT: usize = 4096;
pub const N: usize = 27;
pub const INPUT: usize = 0;
pub const RAW: usize = 1;
pub const Z: usize = 2;
pub const ALPHA: usize = 3;
pub const BETA: usize = 4;
pub const OLD_HISTORY: usize = 5;
pub const NEXT_HISTORY: usize = 6;
pub const ACT: usize = 7;
pub const Q: usize = 8;
pub const K: usize = 9;
pub const V: usize = 10;
pub const CONTROL: usize = 11;
pub const OLD_STATE: usize = 12;
pub const NEXT_STATE: usize = 13;
pub const RECURRENT: usize = 14;
pub const GATED: usize = 15;
pub const OUT: usize = 16;
pub const STATUS: usize = 17;
pub const STATUS_FIELDS: [usize; 10] = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26];
pub const PROJECTION_FIELDS: [(usize, usize, usize); 5] = [
    (0, RAW, 22),
    (1, Z, 23),
    (2, ALPHA, 24),
    (3, BETA, 25),
    (4, OUT, 26),
];
pub const SUFFIXES: [&str; 5] = [
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_out.weight",
];
#[derive(Clone, Copy)]
pub struct Geometry {
    pub pointwise: p::Layout,
    pub span: [p::Span; N],
    pub old_history: p::History,
    pub next_history: p::History,
    pub old_state: c::StateSpan,
    pub next_state: c::StateSpan,
    pub scalar: bool,
}
impl Geometry {
    pub fn new(scalar: bool) -> Result<Self> {
        let pointwise = p::Layout::new(GdnShape::qwen38())?;
        let l = pointwise;
        let h = l.shape.value_heads;
        let widths = [
            D,
            l.channels,
            l.values,
            h,
            h,
            3 * l.channels,
            3 * l.channels,
            l.channels,
            l.qk,
            l.qk,
            l.values,
            2 * h,
            GdnLayout::new(l.shape)?.state_elements,
            GdnLayout::new(l.shape)?.state_elements,
            l.values,
            l.values,
            D,
            l.channels,
            l.channels,
            h,
            l.values,
            l.values,
            l.channels,
            l.values,
            h,
            h,
            D,
        ];
        let span = std::array::from_fn(|i| {
            if scalar {
                p::Span {
                    base: 0,
                    stride: widths[i],
                    width: widths[i],
                    words: widths[i],
                }
            } else {
                p::Span::new(widths[i], 3 + i % 11, 3 + i % 7).expect("bounded public geometry")
            }
        });
        let history = |next| -> Result<p::History> {
            if scalar {
                Ok(p::History {
                    base: 0,
                    stride: 3,
                    channels: l.channels,
                    words: 3 * l.channels,
                })
            } else {
                Ok(p::History::new(
                    l.channels,
                    if next { 11 } else { 7 },
                    if next { 7 } else { 5 },
                )?)
            }
        };
        let state = |next| -> c::StateSpan {
            let s = l.shape.head_dim;
            if scalar {
                c::StateSpan {
                    base: 0,
                    head_stride: s * s,
                    key_stride: s,
                    column_stride: 1,
                    words: h * s * s,
                }
            } else {
                let key_stride = s + if next { 7 } else { 5 };
                let head_stride = s * key_stride + 7;
                let base = if next { 13 } else { 5 };
                c::StateSpan {
                    base,
                    head_stride,
                    key_stride,
                    column_stride: 1,
                    words: base + h * head_stride + 19,
                }
            }
        };
        let g = Self {
            pointwise,
            span,
            old_history: history(false)?,
            next_history: history(true)?,
            old_state: state(false),
            next_state: state(true),
            scalar,
        };
        g.validate(if scalar { 1 } else { MAX_BATCH })?;
        Ok(g)
    }
    pub fn words(self, i: usize) -> usize {
        match i {
            OLD_HISTORY => self.old_history.words,
            NEXT_HISTORY => self.next_history.words,
            OLD_STATE => self.old_state.words,
            NEXT_STATE => self.next_state.words,
            _ => self.span[i].words,
        }
    }
    pub fn validate(self, count: usize) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&count)
            || (self.scalar && count != 1)
            || self.pointwise != p::Layout::new(GdnShape::qwen38())?
        {
            return Err("GDN layer geometry/count".into());
        }
        let l = self.pointwise;
        let h = l.shape.value_heads;
        let widths = [
            D,
            l.channels,
            l.values,
            h,
            h,
            3 * l.channels,
            3 * l.channels,
            l.channels,
            l.qk,
            l.qk,
            l.values,
            2 * h,
            GdnLayout::new(l.shape)?.state_elements,
            GdnLayout::new(l.shape)?.state_elements,
            l.values,
            l.values,
            D,
            l.channels,
            l.channels,
            h,
            l.values,
            l.values,
            l.channels,
            l.values,
            h,
            h,
            D,
        ];
        for (i, w) in widths.into_iter().enumerate() {
            if matches!(i, OLD_HISTORY | NEXT_HISTORY | OLD_STATE | NEXT_STATE) {
                continue;
            }
            let s = self.span[i];
            if s.width != w {
                return Err("GDN layer changed role width".into());
            }
            if self.scalar {
                if s.base != 0 || s.stride != w || s.words != w {
                    return Err("GDN scalar prefix geometry".into());
                }
            } else {
                s.validate(count, s.words)?;
            }
            BufferLayout::new(self.words(i))?;
        }
        for h in [self.old_history, self.next_history] {
            h.validate(h.words)?;
            if h.channels != l.channels {
                return Err("GDN history width".into());
            }
            if self.scalar && (h.base != 0 || h.stride != 3 || h.words != 3 * l.channels) {
                return Err("GDN scalar compact history".into());
            }
        }
        self.old_state.validate(l.shape)?;
        self.next_state.validate(l.shape)?;
        if self.scalar {
            let s = l.shape.head_dim;
            for state in [self.old_state, self.next_state] {
                if state.base != 0
                    || state.head_stride != s * s
                    || state.key_stride != s
                    || state.column_stride != 1
                    || state.words != h * s * s
                {
                    return Err("GDN scalar compact state".into());
                }
            }
        }
        Ok(())
    }
    pub fn footprint(self) -> Result<usize> {
        (0..N).try_fold(0usize, |sum, i| {
            sum.checked_add(BufferLayout::new(self.words(i))?.owned_bytes)
                .ok_or_else(|| "GDN layer page overflow".into())
        })
    }
    pub fn active(self, i: usize, count: usize, index: usize) -> bool {
        match i {
            OLD_HISTORY | NEXT_HISTORY => if i == OLD_HISTORY {
                self.old_history
            } else {
                self.next_history
            }
            .active(index),
            OLD_STATE | NEXT_STATE => {
                let s = if i == OLD_STATE {
                    self.old_state
                } else {
                    self.next_state
                };
                let shape = self.pointwise.shape;
                if index < s.base {
                    return false;
                }
                let r = index - s.base;
                let head = r / s.head_stride;
                let rr = r % s.head_stride;
                let key = rr / s.key_stride;
                let rr = rr % s.key_stride;
                head < shape.value_heads
                    && key < shape.head_dim
                    && rr.is_multiple_of(s.column_stride)
                    && rr / s.column_stride < shape.head_dim
            }
            _ => self.span[i].active(count, index),
        }
    }
    pub fn chunk(self, count: usize) -> Result<c::ChunkSpans> {
        self.validate(count)?;
        let t = |i: usize| c::TokenSpan {
            base: self.span[i].base,
            stride: self.span[i].stride,
            words: self.span[i].words,
        };
        Ok(c::ChunkSpans {
            count,
            query: t(Q),
            key: t(K),
            value: t(V),
            controls: t(CONTROL),
            old: self.old_state,
            next: self.next_state,
            output: t(RECURRENT),
            status: t(20),
        })
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub main: usize,
    pub io: usize,
}
impl Admission {
    pub fn new() -> Result<Self> {
        let b = Geometry::new(false)?;
        let s = Geometry::new(true)?;
        let main = b
            .footprint()?
            .checked_add(s.footprint()?)
            .and_then(|n| n.checked_add(BufferLayout::new(b.pointwise.channels).ok()?.owned_bytes))
            .ok_or("GDN page overflow")?;
        // Initialize ordinary arena owners one at a time, then dispatch <=64
        // kernels between a completed fault/status read and checked sanitizer.
        // Scalar private sources are input+old raw history+old state; reserve
        // their complete upload pages plus the largest full readback and full
        // Upload/Uniform audit duplicates (factor2), never telemetry credit.
        let sources =
            [INPUT, OLD_HISTORY, OLD_STATE]
                .into_iter()
                .try_fold(0usize, |n, i| -> Result<_> {
                    n.checked_add(BufferLayout::new(s.words(i))?.owned_bytes)
                        .ok_or_else(|| "GDN source overflow".into())
                })?;
        let largest = (0..N).try_fold(0usize, |n, i| {
            Ok::<_, Box<dyn std::error::Error>>(
                n.max(BufferLayout::new(b.words(i))?.owned_bytes)
                    .max(BufferLayout::new(s.words(i))?.owned_bytes),
            )
        })?;
        let io = sources
            .checked_add(largest)
            .and_then(|n| n.checked_add(64 * 2 * 512))
            .and_then(|n| n.checked_mul(2))
            .ok_or("GDN IO overflow")?;
        if main > MAIN_CAP || io as u64 > IO_CAP {
            return Err("GDN aggregate admission".into());
        }
        Ok(Self { main, io })
    }
}
pub struct PackedMatrix {
    pub geometry: MatrixGeometry,
    pub handle: Handle,
    pub words: usize,
    pub decoded: Zeroizing<Vec<f32>>,
}
impl PackedMatrix {
    // Shared fixture-only original packed cache; production uses the admitted
    // BatchedResidentProjector and never needs this decoded CPU oracle.
    pub(crate) fn load(
        client: &ComputeClient<WgpuRuntime>,
        tensor: TensorView<'_>,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        let geometry = matrix_geometry(tensor, rows, columns)?;
        let words = geometry.total_bytes().div_ceil(4);
        let full_words = BufferLayout::new(words)?.owned_words;
        let mut packed = Zeroizing::new(Vec::new());
        packed.try_reserve_exact(full_words * 4)?;
        packed.resize(full_words * 4, 0);
        tensor
            .read_range(0, &mut packed[..geometry.total_bytes()])
            .map_err(|_| "GDN public original read")?;
        let mut decoded = Zeroizing::new(Vec::new());
        decoded.try_reserve_exact(rows * columns)?;
        decoded.resize(rows * columns, 0.0);
        tensor
            .info()
            .dtype()
            .decode(&packed[..geometry.total_bytes()], &mut decoded)
            .map_err(|_| "GDN public original decode")?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err("GDN public matrix nonfinite".into());
        }
        let handle = client.create_from_slice(&packed); // PUBLIC original bytes only, before private configuration.
        Ok(Self {
            geometry,
            handle,
            words: full_words,
            decoded,
        })
    }
    pub fn cpu(&self, input: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
        if input.len() != self.geometry.columns() || input.iter().any(|v| !v.is_finite()) {
            return Err("GDN oracle matrix input".into());
        }
        let mut result = Zeroizing::new(Vec::new());
        result.try_reserve_exact(self.geometry.rows())?;
        for row in self.decoded.chunks_exact(input.len()) {
            let y = row
                .iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32;
            if !y.is_finite() {
                return Err("GDN oracle matrix nonfinite".into());
            }
            result.push(y);
        }
        Ok(result)
    }
}
fn matrix_geometry(tensor: TensorView<'_>, rows: usize, columns: usize) -> Result<MatrixGeometry> {
    if tensor.info().dimensions() != [columns as u64, rows as u64] {
        return Err("GDN public matrix shape".into());
    }
    Ok(MatrixGeometry::new(
        tensor.info().dtype().gguf_id(),
        rows,
        columns,
        TILE,
    )?)
}
pub struct PublicLayer {
    pub binding: GraphBindingId,
    pub matrices: [PackedMatrix; 5],
    pub conv: Handle,
    pub params: Handle,
    pub gamma: Handle,
    pub conv_cpu: Zeroizing<Vec<f32>>,
    pub params_cpu: Zeroizing<Vec<f32>>,
    pub gamma_cpu: Zeroizing<Vec<f32>>,
    pub public_bytes: usize,
}
fn matrix_shapes() -> [(usize, usize); 5] {
    [(10240, D), (6144, D), (48, D), (48, D), (D, 6144)]
}
fn public_budget(dtypes: [u32; 5]) -> Result<(usize, usize)> {
    let mut public = 0usize;
    let mut decoded = 0usize;
    for (dtype, (rows, cols)) in dtypes.into_iter().zip(matrix_shapes()) {
        let g = MatrixGeometry::new(dtype, rows, cols, TILE)?;
        public = public
            .checked_add(BufferLayout::new(g.total_bytes().div_ceil(4))?.owned_bytes)
            .ok_or("GDN public overflow")?;
        decoded = decoded
            .checked_add(
                rows.checked_mul(cols)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("GDN decoded overflow")?,
            )
            .ok_or("GDN decoded overflow")?;
    }
    for words in [40960, 96, 128] {
        public = public
            .checked_add(BufferLayout::new(words)?.owned_bytes)
            .ok_or("GDN public overflow")?;
    }
    if public > PUBLIC_CAP || decoded > DECODED_CAP {
        return Err("GDN public admission".into());
    }
    Ok((public, decoded))
}
pub fn public_preflight(graph: BoundGraph<'_, '_>) -> Result<(usize, usize)> {
    if graph.layer_kind(0)? != AttentionKind::Gdn || graph.geometry().residual.width != D {
        return Err("GDN original graph geometry".into());
    }
    let mut dtypes = [0u32; 5];
    for (i, (suffix, (rows, cols))) in SUFFIXES.into_iter().zip(matrix_shapes()).enumerate() {
        let g = matrix_geometry(graph.tensor(&format!("blk.0.{suffix}"))?, rows, cols)?;
        dtypes[i] = g.dtype();
    }
    for (name, words) in [
        ("ssm_conv1d.weight", 40960),
        ("ssm_dt.bias", 48),
        ("ssm_a", 48),
        ("ssm_norm.weight", 128),
    ] {
        let v = graph.static_vector(&format!("blk.0.{name}"))?;
        if v.len() != words || v.iter().any(|v| !v.is_finite()) {
            return Err("GDN public vector geometry/nonfinite".into());
        }
        if name == "ssm_a" && v.iter().any(|&v| v >= 0.0) {
            return Err("GDN stored A".into());
        }
    }
    public_budget(dtypes)
}
impl PublicLayer {
    pub fn load(client: &ComputeClient<WgpuRuntime>, graph: BoundGraph<'_, '_>) -> Result<Self> {
        let (public_bytes, _) = public_preflight(graph)?;
        Admission::new()?;
        let mut matrices = Vec::new();
        matrices.try_reserve_exact(5)?;
        for (suffix, (rows, columns)) in SUFFIXES.into_iter().zip(matrix_shapes()) {
            matrices.push(PackedMatrix::load(
                client,
                graph.tensor(&format!("blk.0.{suffix}"))?,
                rows,
                columns,
            )?);
        }
        let copy = |suffix: &str| -> Result<Zeroizing<Vec<f32>>> {
            let source = graph.static_vector(&format!("blk.0.{suffix}"))?;
            let mut v = Zeroizing::new(Vec::new());
            v.try_reserve_exact(source.len())?;
            v.extend_from_slice(source);
            Ok(v)
        };
        let conv_cpu = copy("ssm_conv1d.weight")?;
        let gamma_cpu = copy("ssm_norm.weight")?;
        let mut params_cpu = copy("ssm_dt.bias")?;
        params_cpu.extend_from_slice(&copy("ssm_a")?);
        let upload = |v: &[f32]| -> Result<Handle> {
            let n = BufferLayout::new(v.len())?.owned_words;
            let mut padded = Zeroizing::new(Vec::new());
            padded.try_reserve_exact(n)?;
            padded.resize(n, 0.0);
            padded[..v.len()].copy_from_slice(v);
            Ok(client.create_from_slice(f32::as_bytes(&padded)))
        };
        Ok(Self {
            binding: graph.binding_id(),
            matrices: matrices.try_into().map_err(|_| "GDN matrix count")?,
            conv: upload(&conv_cpu)?,
            params: upload(&params_cpu)?,
            gamma: upload(&gamma_cpu)?,
            conv_cpu,
            params_cpu,
            gamma_cpu,
            public_bytes,
        })
    }
}
pub struct Buffers(pub [BufferId; N]);
pub fn arg(arena: &PrivateArena, b: &Buffers, i: usize) -> Result<ArrayArg<WgpuRuntime>> {
    Ok(unsafe {
        ArrayArg::from_raw_parts(
            arena.handle(b.0[i])?.clone(),
            arena.layout(b.0[i])?.owned_words,
        )
    })
}
fn public_arg(h: &Handle, words: usize) -> ArrayArg<WgpuRuntime> {
    unsafe { ArrayArg::from_raw_parts(h.clone(), words) }
}
pub fn binding_proof(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    public: &PublicLayer,
) -> Result<()> {
    g.validate(if g.scalar { 1 } else { 8 })?;
    let client = arena.client()?;
    let mut handles = Vec::new();
    handles.try_reserve_exact(N + 8)?;
    for (i, id) in b.0.iter().enumerate() {
        let h = arena.handle(*id)?;
        if arena.layout(*id)?.logical_words != g.words(i)
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
        {
            return Err("GDN layer full owner mismatch".into());
        }
        handles.push(h.clone());
    }
    handles.extend(public.matrices.iter().map(|m| m.handle.clone()));
    handles.extend([
        public.conv.clone(),
        public.params.clone(),
        public.gamma.clone(),
    ]);
    let resources = handles
        .iter()
        .map(|h| client.get_resource(h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, r) in resources.iter().enumerate() {
        let v = r.resource();
        if handles[i].stream != handles[0].stream
            || v.offset != 0
            || v.size != handles[i].size()
            || v.buffer.size() != v.size
            || resources[..i]
                .iter()
                .any(|r| r.resource().buffer == v.buffer)
        {
            return Err("GDN layer same-stream full physical disjoint proof".into());
        }
    }
    Ok(())
}
pub fn project(
    arena: &PrivateArena,
    public: &PublicLayer,
    g: Geometry,
    b: &Buffers,
    which: usize,
    count: usize,
) -> Result<()> {
    g.validate(count)?;
    let (_, output, status) = PROJECTION_FIELDS[which];
    let input = if which == 4 { GATED } else { INPUT };
    let m = &public.matrices[which];
    let spans = BatchSpans::new(
        count,
        0,
        g.span[input].base,
        g.span[input].stride,
        g.span[output].base,
        g.span[output].stride,
    );
    m.geometry.validate_spans(
        &spans,
        m.words,
        arena.layout(b.0[input])?.owned_words,
        arena.layout(b.0[output])?.owned_words,
    )?;
    let client = arena.client()?;
    if g.scalar {
        crate::quant_matvec::cooperative::cooperative_matvec::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(m.geometry.rows() as u32, 1, 1),
            CubeDim::new_1d(64),
            public_arg(&m.handle, m.words),
            arg(arena, b, input)?,
            arg(arena, b, output)?,
            m.geometry.rows(),
            m.geometry.columns(),
            m.geometry.dtype(),
            0,
        );
    } else {
        packed_matmul::launch::<WgpuRuntime>(
            client,
            m.geometry.grid(),
            CubeDim::new_1d(64),
            public_arg(&m.handle, m.words),
            arg(arena, b, input)?,
            arg(arena, b, output)?,
            count,
            0,
            g.span[input].base,
            g.span[input].stride,
            g.span[output].base,
            g.span[output].stride,
            m.geometry.rows(),
            m.geometry.columns(),
            m.geometry.dtype(),
            TILE,
        );
    }
    finite_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(m.geometry.rows().div_ceil(64) as u32, MAX_BATCH as u32, 1),
        CubeDim::new_1d(64),
        arg(arena, b, output)?,
        arg(arena, b, status)?,
        count,
        g.span[output].base,
        g.span[output].stride,
        g.span[status].base,
        g.span[status].stride,
        m.geometry.rows(),
    );
    Ok(())
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Ready,
    Queued,
    Accepted,
    Poisoned,
}
pub fn validate_context(start: usize, count: usize) -> Result<()> {
    if !(1..=MAX_BATCH).contains(&count) || start.checked_add(count).is_none_or(|n| n > MAX_CONTEXT)
    {
        return Err("GDN layer context/count".into());
    }
    Ok(())
}
fn transition(
    phase: &mut Phase,
    expected: Phase,
    success: Phase,
    admitted: bool,
    work: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if *phase != expected || !admitted {
        *phase = Phase::Poisoned;
        return Err("GDN layer lifecycle/binding/context/cancel".into());
    }
    // Error and panic retirement is established BEFORE error-prone work.
    *phase = Phase::Poisoned;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(Ok(())) => {
            *phase = success;
            Ok(())
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("GDN layer operation panicked; candidate must erase".into()),
    }
}
pub struct LayerBatch {
    // Sealed after construction: queue and acceptance must refer to the same
    // full arena owners and geometry throughout the candidate lifecycle.
    geometry: Geometry,
    buffers: Buffers,
    phase: Phase,
    binding: GraphBindingId,
    start: usize,
    count: usize,
}
impl LayerBatch {
    pub fn new(
        graph: BoundGraph<'_, '_>,
        g: Geometry,
        b: Buffers,
        start: usize,
        count: usize,
    ) -> Result<Self> {
        g.validate(count)?;
        validate_context(start, count)?;
        if g.scalar {
            return Err("GDN layer context".into());
        }
        Ok(Self {
            geometry: g,
            buffers: b,
            phase: Phase::Ready,
            binding: graph.binding_id(),
            start,
            count,
        })
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicLayer,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let admitted = !cancel.load(Ordering::Acquire)
            && self.binding == graph.binding_id()
            && self.binding == public.binding
            && self
                .start
                .checked_add(self.count)
                .is_some_and(|n| n <= MAX_CONTEXT);
        let mut phase = self.phase;
        let result = transition(&mut phase, Phase::Ready, Phase::Queued, admitted, || {
            graph.verify_unchanged()?;
            binding_proof(arena, self.geometry, &self.buffers, public)?;
            self.enqueue_inner(arena, public, cancel)
        });
        self.phase = phase;
        result
    }
    fn enqueue_inner(
        &self,
        arena: &PrivateArena,
        public: &PublicLayer,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let g = self.geometry;
        let b = &self.buffers;
        let l = g.pointwise;
        let count = self.count;
        let client = arena.client()?;
        let dim = CubeDim::new_1d(64);
        let check = || {
            if cancel.load(Ordering::Acquire) {
                Err::<(), Box<dyn std::error::Error>>("GDN layer cancelled".into())
            } else {
                Ok(())
            }
        };
        for which in 0..4 {
            check()?;
            project(arena, public, g, b, which, count)?;
        }
        check()?;
        p::conv_silu::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.channels.div_ceil(64) as u32, 1, 1),
            dim,
            arg(arena, b, RAW)?,
            public_arg(&public.conv, public.conv_cpu.len()),
            arg(arena, b, OLD_HISTORY)?,
            arg(arena, b, NEXT_HISTORY)?,
            arg(arena, b, ACT)?,
            arg(arena, b, 17)?,
            count,
            g.span[RAW].base,
            g.span[RAW].stride,
            g.old_history.base,
            g.old_history.stride,
            g.next_history.base,
            g.next_history.stride,
            g.span[ACT].base,
            g.span[ACT].stride,
            g.span[17].base,
            g.span[17].stride,
            l.channels,
        );
        check()?;
        p::prepare_qkv::launch::<WgpuRuntime>(
            client,
            CubeCount::Static((2 * l.shape.key_heads + l.shape.value_heads) as u32, 8, 1),
            dim,
            arg(arena, b, ACT)?,
            arg(arena, b, Q)?,
            arg(arena, b, K)?,
            arg(arena, b, V)?,
            arg(arena, b, 18)?,
            p::EPSILON,
            count,
            g.span[ACT].base,
            g.span[ACT].stride,
            g.span[Q].base,
            g.span[Q].stride,
            g.span[K].base,
            g.span[K].stride,
            g.span[V].base,
            g.span[V].stride,
            g.span[18].base,
            g.span[18].stride,
            l.shape.head_dim,
            l.shape.key_heads,
            l.shape.value_heads,
        );
        check()?;
        p::prepare_controls::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.shape.value_heads.div_ceil(64) as u32, 8, 1),
            dim,
            arg(arena, b, ALPHA)?,
            arg(arena, b, BETA)?,
            public_arg(&public.params, public.params_cpu.len()),
            arg(arena, b, CONTROL)?,
            arg(arena, b, 19)?,
            count,
            g.span[ALPHA].base,
            g.span[ALPHA].stride,
            g.span[BETA].base,
            g.span[BETA].stride,
            g.span[CONTROL].base,
            g.span[CONTROL].stride,
            g.span[19].base,
            g.span[19].stride,
            l.shape.value_heads,
        );
        check()?;
        c::enqueue(
            arena,
            l.shape,
            g.chunk(count)?,
            &c::ChunkBuffers([
                b.0[Q],
                b.0[K],
                b.0[V],
                b.0[CONTROL],
                b.0[OLD_STATE],
                b.0[NEXT_STATE],
                b.0[RECURRENT],
                b.0[20],
            ]),
            cancel,
        )?;
        check()?;
        p::output_gate::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.shape.value_heads as u32, 8, 1),
            dim,
            arg(arena, b, RECURRENT)?,
            arg(arena, b, Z)?,
            public_arg(&public.gamma, public.gamma_cpu.len()),
            arg(arena, b, GATED)?,
            arg(arena, b, 21)?,
            p::EPSILON,
            count,
            g.span[RECURRENT].base,
            g.span[RECURRENT].stride,
            g.span[Z].base,
            g.span[Z].stride,
            g.span[GATED].base,
            g.span[GATED].stride,
            g.span[21].base,
            g.span[21].stride,
            l.shape.head_dim,
            l.shape.value_heads,
            true,
        );
        check()?;
        project(arena, public, g, b, 4, count)?;
        check()
    }
    /// PRECONDITION: caller just completed an owning readback, dropped all Bytes
    /// leases and accepted the checked I/O sanitizer. No state swap is provided.
    pub fn accept(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        snapshot: &[Zeroizing<Vec<u32>>; N],
        before: &[Zeroizing<Vec<u32>>; N],
        cancel: &AtomicBool,
        retired: bool,
    ) -> Result<()> {
        let mut phase = self.phase;
        let result = transition(
            &mut phase,
            Phase::Queued,
            Phase::Accepted,
            !retired && !cancel.load(Ordering::Acquire),
            || {
                if graph.binding_id() != self.binding {
                    return Err("GDN candidate graph binding changed".into());
                }
                graph.verify_unchanged()?;
                for (i, id) in self.buffers.0.iter().enumerate() {
                    if arena.layout(*id)?.logical_words != self.geometry.words(i) {
                        return Err("GDN candidate stale owner".into());
                    }
                }
                validate_snapshot(self.geometry, self.count, snapshot, before, cancel)
            },
        );
        self.phase = phase;
        result
    }
}
impl Drop for LayerBatch {
    fn drop(&mut self) {
        // Metadata is request-derived even though this lab supplies public
        // positions/counts. No physical state is committed by Drop.
        use zeroize::Zeroize;
        self.start.zeroize();
        self.count.zeroize();
        self.phase = Phase::Poisoned;
    }
}
pub fn validate_snapshot(
    g: Geometry,
    count: usize,
    data: &[Zeroizing<Vec<u32>>; N],
    before: &[Zeroizing<Vec<u32>>; N],
    cancel: &AtomicBool,
) -> Result<()> {
    g.validate(count)?;
    if cancel.load(Ordering::Acquire) {
        return Err("GDN layer cancelled result".into());
    }
    for (i, words) in data.iter().enumerate() {
        if words.len() != BufferLayout::new(g.words(i))?.owned_words
            || before[i].len() != g.words(i)
        {
            return Err("GDN layer exact owning read extent".into());
        }
        for (j, &bits) in words.iter().enumerate() {
            let writable = !matches!(i, INPUT | OLD_HISTORY | OLD_STATE) && g.active(i, count, j);
            if !writable && bits != before[i].get(j).copied().unwrap_or(0) {
                return Err("GDN unchanged source/gap/tail/page changed".into());
            }
            if g.active(i, count, j) {
                if i >= STATUS {
                    if bits != 1 {
                        return Err("GDN layer incomplete/nonfinite stage".into());
                    }
                } else if !f32::from_bits(bits).is_finite() {
                    return Err("GDN layer nonfinite active candidate".into());
                }
                if i == CONTROL {
                    let s = g.span[i];
                    let head = (j - s.base) % s.stride;
                    let x = f32::from_bits(bits);
                    if (head < g.pointwise.shape.value_heads && x > 0.0)
                        || (head >= g.pointwise.shape.value_heads && !(0.0..=1.0).contains(&x))
                    {
                        return Err("GDN layer invalid decay/beta candidate".into());
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SENTINEL: u32 = (-173.25f32).to_bits();
    type Images = [Zeroizing<Vec<u32>>; N];
    fn fixture(g: Geometry, count: usize) -> (Images, Images) {
        let before = std::array::from_fn(|i| Zeroizing::new(vec![SENTINEL; g.words(i)]));
        let data = std::array::from_fn(|i| {
            let page = BufferLayout::new(g.words(i)).unwrap();
            let mut x = Zeroizing::new(vec![0u32; page.owned_words]);
            x[..g.words(i)].copy_from_slice(&before[i]);
            for (j, v) in x.iter_mut().enumerate() {
                if !matches!(i, INPUT | OLD_HISTORY | OLD_STATE) && g.active(i, count, j) {
                    *v = if i >= STATUS { 1 } else { 0f32.to_bits() };
                }
            }
            x
        });
        (before, data)
    }
    fn accepted(g: Geometry, n: usize, before: &Images, data: &Images) -> bool {
        validate_snapshot(g, n, data, before, &AtomicBool::new(false)).is_ok()
    }
    #[test]
    fn exact_geometry_admission_and_scalar_original_layout() {
        let b = Geometry::new(false).unwrap();
        let s = Geometry::new(true).unwrap();
        let a = Admission::new().unwrap();
        assert_eq!(
            a.main,
            b.footprint().unwrap()
                + s.footprint().unwrap()
                + BufferLayout::new(b.pointwise.channels).unwrap().owned_bytes
        );
        assert_eq!(a.main, 24_117_248);
        assert_eq!(a.io, 17_235_968);
        assert!(a.main < MAIN_CAP && (a.io as u64) < IO_CAP);
        assert_eq!(GdnLayout::new(b.pointwise.shape).unwrap().workgroups_x, 96);
        for n in [1, 3, 5, 8] {
            b.validate(n).unwrap();
            b.chunk(n)
                .unwrap()
                .validate(GdnLayout::new(b.pointwise.shape).unwrap())
                .unwrap();
        }
        for n in [0, 9, usize::MAX] {
            assert!(b.validate(n).is_err());
        }
        assert!(s.validate(2).is_err());
        let mut x = b;
        x.span[RAW].width -= 1;
        assert!(x.validate(8).is_err());
        let mut x = b;
        x.span[INPUT].stride = usize::MAX;
        assert!(x.validate(8).is_err());
        let mut x = b;
        x.span[OUT].words -= x.span[OUT].stride;
        assert!(x.validate(1).is_err());
        let mut x = s;
        x.old_history.base = 1;
        x.old_history.words += 1;
        assert!(x.validate(1).is_err());
        let mut x = s;
        x.old_state.key_stride += 1;
        x.old_state.words *= 2;
        assert!(x.validate(1).is_err());
    }
    #[test]
    fn original_layer_metadata_budget_and_context_fail_closed() {
        use crate::quant_matvec::{F32, Q4_K, Q6_K};
        assert_eq!(
            public_budget([Q6_K, Q4_K, Q4_K, Q4_K, Q4_K]).unwrap(),
            (67_698_688, 231_669_760)
        );
        assert!(public_budget([u32::MAX; 5]).is_err());
        assert!(public_budget([F32; 5]).is_err());
        for count in [1, 3, 5, 8] {
            validate_context(0, count).unwrap();
            validate_context(MAX_CONTEXT - count, count).unwrap();
            assert!(validate_context(MAX_CONTEXT - count + 1, count).is_err());
        }
        for (start, count) in [
            (0, 0),
            (0, 9),
            (usize::MAX, 1),
            (usize::MAX, usize::MAX),
            (MAX_CONTEXT, 1),
        ] {
            assert!(validate_context(start, count).is_err());
        }
    }
    #[test]
    fn full_owning_extent_sources_gaps_tails_and_all_ten_statuses() {
        let g = Geometry::new(false).unwrap();
        let (before, mut data) = fixture(g, 3);
        assert!(accepted(g, 3, &before, &data));
        for i in STATUS_FIELDS {
            let j = g.span[i].at(2) + g.span[i].width - 1;
            for bad in [0, 2, u32::MAX] {
                data[i][j] = bad;
                assert!(!accepted(g, 3, &before, &data));
            }
            data[i][j] = 1;
        }
        for i in [INPUT, OLD_HISTORY, OLD_STATE] {
            let j = (0..g.words(i)).find(|&j| g.active(i, 3, j)).unwrap();
            data[i][j] ^= 1;
            assert!(!accepted(g, 3, &before, &data));
            data[i][j] ^= 1;
        }
        for (i, j) in [
            (OUT, 0),
            (OUT, g.span[OUT].at(7)),
            (OUT, data[OUT].len() - 1),
            (NEXT_STATE, 0),
            (NEXT_HISTORY, 0),
        ] {
            data[i][j] ^= 1;
            assert!(!accepted(g, 3, &before, &data));
            data[i][j] ^= 1;
        }
        let tail = data[OUT].pop().unwrap();
        assert!(!accepted(g, 3, &before, &data));
        data[OUT].push(tail);
        data[OUT].push(0);
        assert!(!accepted(g, 3, &before, &data));
        data[OUT].pop();
        assert!(accepted(g, 3, &before, &data));
    }
    #[test]
    fn whole_chunk_rejects_late_nonfinite_and_invalid_controls() {
        let g = Geometry::new(false).unwrap();
        let (before, mut data) = fixture(g, 8);
        for i in [
            RAW,
            Z,
            ALPHA,
            BETA,
            ACT,
            Q,
            K,
            V,
            CONTROL,
            RECURRENT,
            GATED,
            OUT,
            NEXT_HISTORY,
            NEXT_STATE,
        ] {
            let j = (0..g.words(i)).rev().find(|&j| g.active(i, 8, j)).unwrap();
            let old = data[i][j];
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                data[i][j] = bad.to_bits();
                assert!(!accepted(g, 8, &before, &data));
            }
            data[i][j] = old;
        }
        let p = g.span[CONTROL].at(7);
        let h = g.pointwise.shape.value_heads;
        data[CONTROL][p] = 0.125f32.to_bits();
        assert!(!accepted(g, 8, &before, &data));
        data[CONTROL][p] = (-0.0f32).to_bits();
        for beta in [-0.01f32, 1.01] {
            data[CONTROL][p + h] = beta.to_bits();
            assert!(!accepted(g, 8, &before, &data));
        }
        data[CONTROL][p + h] = 1f32.to_bits();
        assert!(accepted(g, 8, &before, &data));
        assert!(validate_snapshot(g, 8, &data, &before, &AtomicBool::new(true)).is_err());
    }
    #[test]
    fn scalar_prefix_acceptance_checks_unchanged_whole_pages() {
        let g = Geometry::new(true).unwrap();
        let (before, mut data) = fixture(g, 1);
        assert!(accepted(g, 1, &before, &data));
        let j = g.words(ALPHA);
        data[ALPHA][j] = 1;
        assert!(!accepted(g, 1, &before, &data));
    }
    #[test]
    fn preflight_dispatch_fence_validation_errors_and_panics_are_terminal() {
        use std::cell::RefCell;
        let order = RefCell::new(Vec::new());
        let mut phase = Phase::Ready;
        transition(&mut phase, Phase::Ready, Phase::Queued, true, || {
            order.borrow_mut().extend(["preflight", "enqueue"]);
            Ok(())
        })
        .unwrap();
        assert_eq!(phase, Phase::Queued);
        transition(&mut phase, Phase::Queued, Phase::Accepted, true, || {
            order
                .borrow_mut()
                .extend(["completed-owning-read", "drop-lease", "audit", "validate"]);
            Ok(())
        })
        .unwrap();
        assert_eq!(phase, Phase::Accepted);
        assert_eq!(
            *order.borrow(),
            [
                "preflight",
                "enqueue",
                "completed-owning-read",
                "drop-lease",
                "audit",
                "validate"
            ]
        );
        for expected in [Phase::Ready, Phase::Queued] {
            for kind in 0..3 {
                let mut p = expected;
                let result = transition(&mut p, expected, Phase::Accepted, kind != 0, || {
                    if kind == 2 {
                        panic!("fixed public injected failure")
                    };
                    Err("fixed public injected error".into())
                });
                assert!(result.is_err());
                assert_eq!(p, Phase::Poisoned);
                assert!(transition(&mut p, expected, Phase::Accepted, true, || Ok(())).is_err());
            }
        }
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    fn shader(g: MatrixGeometry, mode: cubecl::server::ExecutionMode) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope.register_type::<f32>(float);
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        let w = b.input_array(Type::new(uint)).into();
        let x = b.input_array(Type::new(float)).into();
        let y = b.output_array(Type::new(float)).into();
        let n = b.scalar(uint).into();
        let wf = b.scalar(uint).into();
        let xf = b.scalar(uint).into();
        let xs = b.scalar(uint).into();
        let yf = b.scalar(uint).into();
        let ys = b.scalar(uint).into();
        packed_matmul::expand(
            &mut b.scope,
            w,
            x,
            y,
            n,
            wf,
            xf,
            xs,
            yf,
            ys,
            g.rows(),
            g.columns(),
            g.dtype(),
            TILE,
        );
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("original_gdn_projection"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn original_layer_five_projection_shapes_validate_checked_unchecked_wgsl() {
        use crate::quant_matvec::{Q4_K, Q6_K};
        use cubecl::server::ExecutionMode;
        for (dtype, (rows, columns)) in [Q6_K, Q4_K, Q4_K, Q4_K, Q4_K]
            .into_iter()
            .zip(matrix_shapes())
        {
            let g = MatrixGeometry::new(dtype, rows, columns, TILE).unwrap();
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = shader(g, mode);
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::empty(),
                )
                .validate(&module)
                .unwrap();
                // The generated compiler output must retain every request
                // scalar as runtime INFO, including actual count and offsets.
                for scalar in 0..6 {
                    assert!(source.contains(&format!("scalars_u32[{scalar}]")));
                }
                assert!(source.contains("workgroupBarrier"));
            }
        }
    }
}
