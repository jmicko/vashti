//! Fixed MAX8 candidate-only original layer1 PLE. No graph/default/state hook.
//! Original lexical Q5 rows are read in caller-guarded110-byte slices, never
//! cached by private row ID. Token/normalized histories have distinct owners;
//! neither committed history is mutated or committed by this operator.
use crate::{
    ple_batch_gpu as p,
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::matmul::{BatchSpans, MatrixGeometry, packed_matmul},
    residual_batch_gpu::finite_span,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{
    TensorType,
    model::{BoundGraph, GraphBindingId},
    ops::{GatedResidualShape, PleHashMetadata},
};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const MAX_BATCH: usize = 8;
pub const HEADS: usize = 16;
pub const ROW_WIDTH: usize = 160;
pub const EMB_WIDTH: usize = HEADS * ROW_WIDTH;
pub const HISTORY: usize = 9;
pub const MAIN_CAP: usize = 16 * 1024 * 1024;
pub const PUBLIC_CAP: usize = 32 * 1024 * 1024;
pub const IO_CAP: u64 = 64 * 1024 * 1024;
pub const MAX_CONTEXT: usize = 4096;
pub const N: usize = 23;
pub const EMB: usize = 0;
pub const RESIDUAL: usize = 1;
pub const KEY: usize = 2;
pub const VALUE: usize = 3;
pub const KEY_NORM: usize = 4;
pub const QUERY_NORM: usize = 5;
pub const GATES: usize = 6;
pub const GATED: usize = 7;
pub const NORMALIZED: usize = 8;
pub const CONV: usize = 9;
pub const OUTPUT: usize = 10;
pub const OLD: usize = 11;
pub const NEXT: usize = 12;
pub const STATUS: usize = 13;
#[cfg(test)]
pub const STATUS_FIELDS: [usize; 10] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22];

#[derive(Clone, Copy)]
pub struct Span {
    pub base: usize,
    pub stride: usize,
    pub width: usize,
    pub words: usize,
}
impl Span {
    fn fixture(width: usize, aligned: bool) -> Result<Self> {
        let (base, stride, tail) = if aligned {
            (
                64,
                width.checked_add(63).ok_or("PLE span overflow")? / 64 * 64 + 64,
                128,
            )
        } else {
            (5, width.checked_add(3).ok_or("PLE span overflow")?, 13)
        };
        let words = MAX_BATCH
            .checked_mul(stride)
            .and_then(|n| n.checked_add(base))
            .and_then(|n| n.checked_add(tail))
            .ok_or("PLE span overflow")?;
        let s = Self {
            base,
            stride,
            width,
            words,
        };
        s.validate(width)?;
        Ok(s)
    }
    pub fn validate(self, width: usize) -> Result<()> {
        let end = (MAX_BATCH - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.base))
            .and_then(|n| n.checked_add(width))
            .ok_or("PLE span overflow")?;
        if width == 0
            || self.width != width
            || self.stride < width
            || end > self.words
            || [self.base, self.stride, self.width, self.words, end]
                .iter()
                .any(|&n| n > u32::MAX as usize)
        {
            return Err("PLE full MAX8 span".into());
        }
        BufferLayout::new(self.words)?;
        Ok(())
    }
    pub fn at(self, t: usize) -> usize {
        self.base + t * self.stride
    }
    pub fn active(self, count: usize, i: usize) -> bool {
        i >= self.base
            && (i - self.base) / self.stride < count
            && (i - self.base) % self.stride < self.width
    }
}
#[derive(Clone, Copy)]
pub struct HistorySpan {
    pub base: usize,
    pub stride: usize,
    pub channels: usize,
    pub words: usize,
}
impl HistorySpan {
    fn fixture(channels: usize, next: bool) -> Result<Self> {
        let base = if next { 11 } else { 7 };
        let stride = if next { 12 } else { 11 };
        let words = channels
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(base + HISTORY + 13))
            .ok_or("PLE history overflow")?;
        let h = Self {
            base,
            stride,
            channels,
            words,
        };
        h.validate(channels)?;
        Ok(h)
    }
    pub fn validate(self, channels: usize) -> Result<()> {
        let end = channels
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.stride))
            .and_then(|n| n.checked_add(self.base))
            .and_then(|n| n.checked_add(HISTORY))
            .ok_or("PLE history overflow")?;
        if channels == 0
            || self.channels != channels
            || self.stride < HISTORY
            || end > self.words
            || [self.base, self.stride, self.channels, self.words, end]
                .iter()
                .any(|&n| n > u32::MAX as usize)
        {
            return Err("PLE full history span".into());
        }
        BufferLayout::new(self.words)?;
        Ok(())
    }
    pub fn at(self, c: usize) -> usize {
        self.base + c * self.stride
    }
    pub fn active(self, i: usize) -> bool {
        i >= self.base
            && (i - self.base) / self.stride < self.channels
            && (i - self.base) % self.stride < HISTORY
    }
}
#[derive(Clone, Copy)]
pub struct Geometry {
    pub shape: GatedResidualShape,
    pub spans: [Span; N],
    pub old: HistorySpan,
    pub next: HistorySpan,
}
impl Geometry {
    pub fn new(shape: GatedResidualShape, aligned: bool) -> Result<Self> {
        GatedResidualShape::new(shape.streams, shape.width)?;
        if shape.streams > 4 || shape.width > 2560 {
            return Err("PLE public geometry bound".into());
        }
        let channels = shape
            .streams
            .checked_mul(shape.width)
            .ok_or("PLE geometry overflow")?;
        let widths = [
            EMB_WIDTH,
            channels,
            channels,
            shape.width,
            channels,
            channels,
            shape.streams,
            channels,
            channels,
            channels,
            channels,
            channels * HISTORY,
            channels * HISTORY,
            channels,
            shape.width,
            shape.streams,
            shape.streams,
            shape.streams,
            channels,
            shape.streams,
            channels,
            channels,
            channels,
        ];
        let mut spans = [Span {
            base: 0,
            stride: 0,
            width: 0,
            words: 0,
        }; N];
        for (s, w) in spans.iter_mut().zip(widths) {
            *s = Span::fixture(w, aligned)?;
        }
        let old = HistorySpan::fixture(channels, false)?;
        let next = HistorySpan::fixture(channels, true)?;
        // History/status are single full owners, not eight copies.
        spans[OLD].words = old.words;
        spans[NEXT].words = next.words;
        spans[22] = Span {
            base: 5,
            stride: channels,
            width: channels,
            words: channels
                .checked_add(18)
                .ok_or("PLE history status overflow")?,
        };
        let g = Self {
            shape,
            spans,
            old,
            next,
        };
        g.validate(8)?;
        Ok(g)
    }
    pub fn channels(self) -> usize {
        self.shape.streams * self.shape.width
    }
    pub fn words(self, i: usize) -> usize {
        if i == OLD {
            self.old.words
        } else if i == NEXT {
            self.next.words
        } else {
            self.spans[i].words
        }
    }
    pub fn validate(self, count: usize) -> Result<()> {
        if !(1..=8).contains(&count)
            || self.shape.streams == 0
            || self.shape.streams > 4
            || self.shape.width == 0
            || self.shape.width > 2560
        {
            return Err("PLE geometry/count".into());
        }
        let c = self.channels();
        let widths = [
            EMB_WIDTH,
            c,
            c,
            self.shape.width,
            c,
            c,
            self.shape.streams,
            c,
            c,
            c,
            c,
            c * 9,
            c * 9,
            c,
            self.shape.width,
            self.shape.streams,
            self.shape.streams,
            self.shape.streams,
            c,
            self.shape.streams,
            c,
            c,
            c,
        ];
        for (i, &w) in widths.iter().enumerate() {
            if i == OLD {
                self.old.validate(c)?;
            } else if i == NEXT {
                self.next.validate(c)?;
            } else if i == 22 {
                let s = self.spans[i];
                let end = s.base.checked_add(c).ok_or("PLE status overflow")?;
                if s.width != c
                    || end > s.words
                    || [s.base, s.words, end]
                        .iter()
                        .any(|&n| n > u32::MAX as usize)
                {
                    return Err("PLE history status span".into());
                }
                BufferLayout::new(s.words)?;
            } else {
                self.spans[i].validate(w)?;
            }
        }
        Ok(())
    }
    pub fn footprint(self) -> Result<usize> {
        self.validate(8)?;
        (0..N).try_fold(0usize, |n, i| {
            n.checked_add(BufferLayout::new(self.words(i))?.owned_bytes)
                .ok_or_else(|| "PLE main page sum overflow".into())
        })
    }
    pub fn active(self, i: usize, count: usize, j: usize) -> bool {
        if i == OLD {
            self.old.active(j)
        } else if i == NEXT {
            self.next.active(j)
        } else if i == 22 {
            j >= self.spans[i].base && j < self.spans[i].base + self.channels()
        } else {
            self.spans[i].active(count, j)
        }
    }
    pub fn changed(self, i: usize, count: usize, j: usize) -> bool {
        !matches!(i, EMB | RESIDUAL | OLD) && self.active(i, count, j)
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
pub struct Matrix {
    pub geometry: MatrixGeometry,
    pub handle: Handle,
    pub words: usize,
}
pub struct PublicLayer {
    pub binding: GraphBindingId,
    pub shape: GatedResidualShape,
    pub matrices: [Matrix; 2],
    pub gamma: [Handle; 3],
    pub gamma_cpu: [Zeroizing<Vec<f32>>; 3],
    pub conv: Handle,
    pub conv_cpu: Zeroizing<Vec<f32>>,
    pub physical_bytes: usize,
}
pub fn public_preflight(graph: BoundGraph<'_, '_>) -> Result<usize> {
    let g = graph.geometry();
    if g.residual != GatedResidualShape::new(4, 2560)?
        || g.ple.layer != 1
        || g.ple.heads != HEADS
        || g.ple.head_width != ROW_WIDTH
        || g.ple.predecessor_tokens != 2
        || g.ple.boundary_token != 248044
        || g.generation_eos != 248046
    {
        return Err("PLE original graph boundary".into());
    }
    let table = graph.tensor("per_layer_token_embd.weight")?;
    if table.info().dtype() != TensorType::Q5_0
        || table.info().dimensions() != [160, g.ple.table_rows]
    {
        return Err("PLE original Q5 row layout".into());
    }
    let mut total = 0usize;
    for (name, rows) in [
        ("blk.1.ple_key.weight", 10240),
        ("blk.1.ple_value.weight", 2560),
    ] {
        let tensor = graph.tensor(name)?;
        if tensor.info().dimensions() != [2560, rows] || tensor.info().dtype() != TensorType::Q4K {
            return Err("PLE original packed projection layout".into());
        }
        let m = MatrixGeometry::new(tensor.info().dtype().gguf_id(), rows as usize, EMB_WIDTH, 8)?;
        total = total
            .checked_add(BufferLayout::new(m.total_bytes().div_ceil(4))?.owned_bytes)
            .ok_or("PLE public page sum overflow")?;
    }
    for name in [
        "blk.1.ple_norm_key.weight",
        "blk.1.ple_norm_query.weight",
        "blk.1.ple_norm_conv.weight",
        "blk.1.ple_conv1d.weight",
    ] {
        let dtype = graph.tensor(name)?.info().dtype();
        if dtype
            != if name.ends_with("conv1d.weight") {
                TensorType::F16
            } else {
                TensorType::F32
            }
        {
            return Err("PLE original public vector encoding".into());
        }
        let v = graph.static_vector(name)?;
        let expected = if name.ends_with("conv1d.weight") {
            40960
        } else {
            10240
        };
        if v.len() != expected || v.iter().any(|x| !x.is_finite()) {
            return Err("PLE original public vector layout".into());
        }
        total = total
            .checked_add(BufferLayout::new(v.len())?.owned_bytes)
            .ok_or("PLE public page sum overflow")?;
    }
    if total > PUBLIC_CAP {
        return Err("PLE full public budget".into());
    }
    Ok(total)
}
impl PublicLayer {
    pub fn load(client: &ComputeClient<WgpuRuntime>, graph: BoundGraph<'_, '_>) -> Result<Self> {
        let physical_bytes = public_preflight(graph)?;
        let mut matrices = Vec::new();
        matrices.try_reserve_exact(2)?;
        for (name, rows) in [
            ("blk.1.ple_key.weight", 10240),
            ("blk.1.ple_value.weight", 2560),
        ] {
            let tensor = graph.tensor(name)?;
            let geometry =
                MatrixGeometry::new(tensor.info().dtype().gguf_id(), rows, EMB_WIDTH, 8)?;
            let layout = BufferLayout::new(geometry.total_bytes().div_ceil(4))?;
            let mut packed = Zeroizing::new(Vec::new());
            packed.try_reserve_exact(layout.owned_bytes)?;
            packed.resize(layout.owned_bytes, 0);
            tensor
                .read_range(0, &mut packed[..geometry.total_bytes()])
                .map_err(|_| "PLE original public weight read")?;
            matrices.push(Matrix {
                geometry,
                handle: client.create_from_slice(&packed),
                words: layout.owned_words,
            });
        }
        let mut gamma_cpu: [Zeroizing<Vec<f32>>; 3] =
            std::array::from_fn(|_| Zeroizing::new(Vec::new()));
        let names = [
            "blk.1.ple_norm_key.weight",
            "blk.1.ple_norm_query.weight",
            "blk.1.ple_norm_conv.weight",
        ];
        let mut handles = Vec::new();
        handles.try_reserve_exact(3)?;
        for (v, name) in gamma_cpu.iter_mut().zip(names) {
            let data = graph.static_vector(name)?;
            let words = BufferLayout::new(data.len())?.owned_words;
            v.try_reserve_exact(words)?;
            v.resize(words, 0.0);
            v[..data.len()].copy_from_slice(data);
            handles.push(client.create_from_slice(f32::as_bytes(v)));
        }
        let data = graph.static_vector("blk.1.ple_conv1d.weight")?;
        let mut conv_cpu = Zeroizing::new(Vec::new());
        conv_cpu.try_reserve_exact(BufferLayout::new(data.len())?.owned_words)?;
        conv_cpu.resize(BufferLayout::new(data.len())?.owned_words, 0.0);
        conv_cpu[..data.len()].copy_from_slice(data);
        Ok(Self {
            binding: graph.binding_id(),
            shape: graph.geometry().residual,
            matrices: matrices.try_into().map_err(|_| "PLE matrix count")?,
            gamma: handles.try_into().map_err(|_| "PLE gamma count")?,
            gamma_cpu,
            conv: client.create_from_slice(f32::as_bytes(&conv_cpu)),
            conv_cpu,
            physical_bytes,
        })
    }
}
pub fn binding_proof(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    public: &PublicLayer,
) -> Result<()> {
    g.validate(8)?;
    if g.shape != public.shape {
        return Err("PLE immutable weight geometry".into());
    }
    let client = arena.client()?;
    let mut handles = Vec::new();
    handles.try_reserve_exact(N + 6)?;
    for (i, &id) in b.0.iter().enumerate() {
        let h = arena.handle(id)?;
        if arena.layout(id)?.logical_words != g.words(i)
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
        {
            return Err("PLE stale or partial main owner".into());
        }
        handles.push(h.clone());
    }
    handles.extend(public.matrices.iter().map(|m| m.handle.clone()));
    handles.extend(public.gamma.iter().cloned());
    handles.push(public.conv.clone());
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
                .any(|x| x.resource().buffer == v.buffer)
        {
            return Err("PLE same-stream full physical disjoint owners".into());
        }
    }
    let physical = resources.iter().try_fold(0u64, |n, r| {
        n.checked_add(r.resource().size)
            .ok_or("PLE physical sum overflow")
    })?;
    let expected = g
        .footprint()?
        .checked_add(public.physical_bytes)
        .ok_or("PLE combined physical budget overflow")?;
    if physical != expected as u64 || expected > MAIN_CAP + PUBLIC_CAP {
        return Err("PLE actual combined MAIN/PUBLIC pages".into());
    }
    Ok(())
}

/// Caller owns full guarded token/position arrays before filling. No default
/// builder can fabricate omitted ranks/tokens; all active slots are supplied.
pub fn rows_into(
    meta: PleHashMetadata<'_>,
    current: u32,
    history: &[i32; 2],
    output: &mut [u32; HEADS],
) -> Result<()> {
    if meta.multipliers.len() != 3
        || meta.heads_per_ngram != 8
        || meta.vocab_sizes.len() != HEADS
        || meta.offsets.len() != HEADS
        || meta.multipliers.contains(&0)
        || meta.boundary_token != 248044
        || meta.token_vocab_size != 248320
        || current >= meta.token_vocab_size
        || history
            .iter()
            .any(|&x| x >= 0 && x as u32 >= meta.token_vocab_size)
    {
        return Err("PLE exact integer hash metadata/input".into());
    }
    let mut context = Zeroizing::new([0u64; 3]);
    context[0] = u64::from(current);
    let mut cut = Zeroizing::new(false);
    for distance in 1..3 {
        let old = history[2 - distance];
        *cut |= old < 0 || old == meta.boundary_token;
        context[distance] = if *cut {
            meta.boundary_token as u64
        } else {
            old as u64
        };
    }
    let mut mixed = Zeroizing::new(context[0].wrapping_mul(meta.multipliers[0]));
    // Validate public integer row ranges before private output mutation.
    for h in 0..HEADS {
        if meta.vocab_sizes[h] == 0
            || meta.offsets[h]
                .checked_add(meta.vocab_sizes[h])
                .is_none_or(|n| n > u32::MAX as u64 + 1)
        {
            return Err("PLE integer row bounds".into());
        }
    }
    for distance in 1..3 {
        *mixed ^= context[distance].wrapping_mul(meta.multipliers[distance]);
        for (h, out) in output
            .iter_mut()
            .enumerate()
            .take(distance * 8)
            .skip((distance - 1) * 8)
        {
            *out = (*mixed % meta.vocab_sizes[h] + meta.offsets[h]) as u32;
        }
    }
    Ok(())
}
fn decode_row(row: &[u8; 110], decoded: &mut [f32; ROW_WIDTH]) -> Result<()> {
    TensorType::Q5_0
        .decode(row, decoded)
        .map_err(|_| "PLE exact Q5 row decode")?;
    if decoded.iter().any(|x| !x.is_finite()) {
        return Err("PLE original lexical row nonfinite".into());
    }
    Ok(())
}
pub struct RowBatch {
    rows: Zeroizing<[[u32; HEADS]; MAX_BATCH]>,
    next_tokens: Zeroizing<[i32; 2]>,
    upload: Zeroizing<Vec<u8>>,
}
impl RowBatch {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        graph: BoundGraph<'_, '_>,
        g: Geometry,
        count: usize,
        start: usize,
        tokens: &Zeroizing<[u32; MAX_BATCH]>,
        positions: &Zeroizing<[u32; MAX_BATCH]>,
        old: &Zeroizing<[i32; 2]>,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        g.validate(count)?;
        let public = public_preflight(graph)?;
        let main = g.footprint()?;
        if main > MAIN_CAP
            || main
                .checked_add(public)
                .is_none_or(|n| n > MAIN_CAP + PUBLIC_CAP)
        {
            return Err("PLE original row full-page admission".into());
        }
        if g.shape != graph.geometry().residual
            || start.checked_add(count).is_none_or(|n| n > MAX_CONTEXT)
            || (0..count).any(|t| positions[t] as usize != start + t)
            || cancel.load(Ordering::Acquire)
        {
            return Err("PLE runtime causal token positions/count/cancel".into());
        }
        // Metadata and all token sources preflight before any private heap fill.
        let meta = graph.ple_hash().metadata();
        let mut rows = Zeroizing::new([[0u32; HEADS]; MAX_BATCH]);
        let mut next_tokens = Zeroizing::new([old[0], old[1]]);
        for t in 0..count {
            if cancel.load(Ordering::Acquire) {
                return Err("PLE cancelled row preparation".into());
            }
            rows_into(meta, tokens[t], &next_tokens, &mut rows[t])?;
            next_tokens[0] = next_tokens[1];
            next_tokens[1] = tokens[t] as i32;
        }
        let table = graph.tensor("per_layer_token_embd.weight")?;
        let mut upload = Zeroizing::new(Vec::new());
        let layout = BufferLayout::new(g.words(EMB))?;
        upload.try_reserve_exact(layout.owned_bytes)?;
        if upload.capacity() != layout.owned_bytes {
            return Err("PLE exact guarded source capacity".into());
        }
        upload.resize(layout.owned_bytes, 0);
        let mut row = Zeroizing::new([0u8; 110]);
        let mut decoded = Zeroizing::new([0f32; ROW_WIDTH]);
        for t in 0..count {
            for head in 0..HEADS {
                if cancel.load(Ordering::Acquire) {
                    return Err("PLE cancelled original row gather".into());
                }
                let offset = u64::from(rows[t][head])
                    .checked_mul(110)
                    .ok_or("PLE original row offset overflow")?;
                table
                    .read_range(offset, &mut row[..])
                    .map_err(|_| "PLE bounded original Q5 row read")?;
                decode_row(&row, &mut decoded)?;
                for (d, &x) in decoded.iter().enumerate() {
                    let at = (g.spans[EMB].at(t) + head * ROW_WIDTH + d) * 4;
                    upload[at..at + 4].copy_from_slice(&x.to_bits().to_le_bytes());
                }
            }
        }
        graph.verify_unchanged()?;
        Ok(Self {
            rows,
            next_tokens,
            upload,
        })
    }
    pub fn rows(&self) -> &[[u32; HEADS]; MAX_BATCH] {
        &self.rows
    }
    pub fn next_tokens(&self) -> &[i32; 2] {
        &self.next_tokens
    }
    pub fn bytes(&self) -> &[u8] {
        &self.upload
    }
    pub fn enqueue(&mut self, arena: &PrivateArena, target: BufferId) -> Result<()> {
        let layout = arena.layout(target)?;
        if self.upload.len() != layout.owned_bytes {
            return Err("PLE exact full owned embedding upload".into());
        }
        arena
            .client()?
            .write_private(
                arena.handle(target)?.clone(),
                std::mem::take(&mut *self.upload),
            )
            .map_err(|_| "PLE private embedding write")?;
        Ok(())
    }
}

pub fn project(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    public: &PublicLayer,
    which: usize,
    count: usize,
) -> Result<()> {
    g.validate(count)?;
    let (output, status) = if which == 0 {
        (KEY, 13)
    } else if which == 1 {
        (VALUE, 14)
    } else {
        return Err("PLE projection index".into());
    };
    let m = &public.matrices[which];
    let x = g.spans[EMB];
    let y = g.spans[output];
    let flags = g.spans[status];
    let spans = BatchSpans::new(count, 0, x.base, x.stride, y.base, y.stride);
    m.geometry.validate_spans(
        &spans,
        m.words,
        arena.layout(b.0[EMB])?.owned_words,
        arena.layout(b.0[output])?.owned_words,
    )?;
    let client = arena.client()?;
    let dim = CubeDim::new_1d(64);
    packed_matmul::launch::<WgpuRuntime>(
        client,
        m.geometry.grid(),
        dim,
        unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) },
        arg(arena, b, EMB)?,
        arg(arena, b, output)?,
        count,
        0,
        x.base,
        x.stride,
        y.base,
        y.stride,
        m.geometry.rows(),
        m.geometry.columns(),
        m.geometry.dtype(),
        8,
    );
    finite_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(m.geometry.rows().div_ceil(64) as u32, 8, 1),
        dim,
        arg(arena, b, output)?,
        arg(arena, b, status)?,
        count,
        y.base,
        y.stride,
        flags.base,
        flags.stride,
        m.geometry.rows(),
    );
    Ok(())
}
/// Pointwise entry for both original full layer and separately labeled odd
/// projected-tail fixture. Caller proves ALL full owners/public parameters.
#[allow(clippy::too_many_arguments)]
pub fn enqueue_tail(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    gamma: &[Handle; 3],
    gamma_words: [usize; 3],
    conv: &Handle,
    conv_words: usize,
    count: usize,
    cancel: &AtomicBool,
) -> Result<()> {
    g.validate(count)?;
    let c = g.channels();
    if gamma.iter().zip(gamma_words).any(|(h, n)| {
        n < c
            || n.checked_mul(4) != Some(h.size() as usize)
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
    }) || conv_words < c * 4
        || conv_words.checked_mul(4) != Some(conv.size() as usize)
        || conv.offset_start.unwrap_or(0) != 0
        || conv.offset_end.unwrap_or(0) != 0
    {
        return Err("PLE full immutable parameter span".into());
    }
    let client = arena.client()?;
    let dim = CubeDim::new_1d(64);
    let check = || {
        if cancel.load(Ordering::Acquire) {
            Err::<(), Box<dyn std::error::Error>>("PLE cancelled tail".into())
        } else {
            Ok(())
        }
    };
    let pubarg = |h: &Handle, n: usize| unsafe { ArrayArg::from_raw_parts(h.clone(), n) };
    for (input, output, status, which) in [(KEY, KEY_NORM, 15, 0), (RESIDUAL, QUERY_NORM, 16, 1)] {
        check()?;
        let x = g.spans[input];
        let y = g.spans[output];
        let f = g.spans[status];
        p::rms::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(g.shape.streams as u32, 8, 1),
            dim,
            arg(arena, b, input)?,
            pubarg(&gamma[which], gamma_words[which]),
            arg(arena, b, output)?,
            arg(arena, b, status)?,
            1e-6,
            count,
            x.base,
            x.stride,
            0,
            y.base,
            y.stride,
            f.base,
            f.stride,
            g.shape.streams,
            g.shape.width,
        );
    }
    check()?;
    let k = g.spans[KEY_NORM];
    let q = g.spans[QUERY_NORM];
    let gates = g.spans[GATES];
    let f = g.spans[17];
    p::gate::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(g.shape.streams as u32, 8, 1),
        dim,
        arg(arena, b, KEY_NORM)?,
        arg(arena, b, QUERY_NORM)?,
        arg(arena, b, GATES)?,
        arg(arena, b, 17)?,
        count,
        k.base,
        k.stride,
        q.base,
        q.stride,
        gates.base,
        gates.stride,
        f.base,
        f.stride,
        g.shape.streams,
        g.shape.width,
    );
    check()?;
    let v = g.spans[VALUE];
    let y = g.spans[GATED];
    let f = g.spans[18];
    p::gated::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(c.div_ceil(64) as u32, 8, 1),
        dim,
        arg(arena, b, VALUE)?,
        arg(arena, b, GATES)?,
        arg(arena, b, GATED)?,
        arg(arena, b, 18)?,
        count,
        v.base,
        v.stride,
        gates.base,
        gates.stride,
        y.base,
        y.stride,
        f.base,
        f.stride,
        g.shape.streams,
        g.shape.width,
    );
    check()?;
    let n = g.spans[NORMALIZED];
    let f = g.spans[19];
    p::rms::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(g.shape.streams as u32, 8, 1),
        dim,
        arg(arena, b, GATED)?,
        pubarg(&gamma[2], gamma_words[2]),
        arg(arena, b, NORMALIZED)?,
        arg(arena, b, 19)?,
        1e-6,
        count,
        y.base,
        y.stride,
        0,
        n.base,
        n.stride,
        f.base,
        f.stride,
        g.shape.streams,
        g.shape.width,
    );
    check()?;
    let r = g.spans[RESIDUAL];
    let cv = g.spans[CONV];
    let out = g.spans[OUTPUT];
    let sf = g.spans[20];
    let tf = g.spans[21];
    p::conv_tail::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(c.div_ceil(64) as u32, 1, 1),
        dim,
        arg(arena, b, NORMALIZED)?,
        pubarg(conv, conv_words),
        arg(arena, b, OLD)?,
        arg(arena, b, NEXT)?,
        arg(arena, b, RESIDUAL)?,
        arg(arena, b, GATED)?,
        arg(arena, b, CONV)?,
        arg(arena, b, OUTPUT)?,
        arg(arena, b, 20)?,
        arg(arena, b, 21)?,
        arg(arena, b, 22)?,
        count,
        n.base,
        n.stride,
        0,
        g.old.base,
        g.old.stride,
        g.next.base,
        g.next.stride,
        r.base,
        r.stride,
        y.base,
        y.stride,
        cv.base,
        cv.stride,
        out.base,
        out.stride,
        sf.base,
        sf.stride,
        tf.base,
        tf.stride,
        g.spans[22].base,
        c,
    );
    check()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Queued,
    Accepted,
    Poisoned,
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
        return Err("PLE lifecycle/context/cancel".into());
    }
    *phase = Phase::Poisoned;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(Ok(())) => {
            *phase = success;
            Ok(())
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("PLE candidate operation panic; explicit erase required".into()),
    }
}
pub struct Chunk {
    geometry: Geometry,
    buffers: Buffers,
    binding: GraphBindingId,
    phase: Phase,
    start: usize,
    count: usize,
    next_tokens: Zeroizing<[i32; 2]>,
}
impl Drop for Chunk {
    fn drop(&mut self) {
        self.start.zeroize();
        self.count.zeroize();
    }
}
impl Chunk {
    pub fn new(
        graph: BoundGraph<'_, '_>,
        g: Geometry,
        b: Buffers,
        start: usize,
        count: usize,
    ) -> Result<Self> {
        g.validate(count)?;
        if g.shape != graph.geometry().residual
            || start.checked_add(count).is_none_or(|n| n > MAX_CONTEXT)
        {
            return Err("PLE original layer1 input/count/cursor".into());
        }
        Ok(Self {
            geometry: g,
            buffers: b,
            binding: graph.binding_id(),
            phase: Phase::Ready,
            start,
            count,
            next_tokens: Zeroizing::new([0; 2]),
        })
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicLayer,
        tokens: &Zeroizing<[u32; 8]>,
        positions: &Zeroizing<[u32; 8]>,
        old_tokens: &Zeroizing<[i32; 2]>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut phase = self.phase;
        let result = transition(
            &mut phase,
            Phase::Ready,
            Phase::Queued,
            !cancel.load(Ordering::Acquire)
                && self.binding == graph.binding_id()
                && self.binding == public.binding,
            || {
                graph.verify_unchanged()?;
                binding_proof(arena, self.geometry, &self.buffers, public)?;
                let mut rows = RowBatch::prepare(
                    graph,
                    self.geometry,
                    self.count,
                    self.start,
                    tokens,
                    positions,
                    old_tokens,
                    cancel,
                )?;
                self.next_tokens.copy_from_slice(rows.next_tokens());
                rows.enqueue(arena, self.buffers.0[EMB])?;
                if cancel.load(Ordering::Acquire) {
                    return Err("PLE cancelled after owned gather upload".into());
                }
                project(arena, self.geometry, &self.buffers, public, 0, self.count)?;
                project(arena, self.geometry, &self.buffers, public, 1, self.count)?;
                enqueue_tail(
                    arena,
                    self.geometry,
                    &self.buffers,
                    &public.gamma,
                    public.gamma_cpu.each_ref().map(|x| x.len()),
                    &public.conv,
                    public.conv_cpu.len(),
                    self.count,
                    cancel,
                )
            },
        );
        self.phase = phase;
        result
    }
    /// PRECONDITION: completed final owning readback and dropped every Bytes
    /// lease, then accepted checked runtime audit. This is layer-only numeric
    /// acceptance, never a current-history/cursor commit for the whole graph.
    pub fn accept(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        actual: &[Zeroizing<Vec<u32>>; N],
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
                    return Err("PLE candidate graph binding changed".into());
                }
                graph.verify_unchanged()?;
                for (i, &id) in self.buffers.0.iter().enumerate() {
                    if arena.layout(id)?.logical_words != self.geometry.words(i) {
                        return Err("PLE candidate stale epoch".into());
                    }
                }
                validate_snapshot(self.geometry, self.count, actual, before, cancel)
            },
        );
        self.phase = phase;
        result
    }
    pub fn candidate_tokens(&self) -> Result<&[i32; 2]> {
        if self.phase != Phase::Accepted {
            return Err("PLE unaccepted candidate history".into());
        }
        Ok(&self.next_tokens)
    }
}

pub fn validate_snapshot(
    g: Geometry,
    count: usize,
    actual: &[Zeroizing<Vec<u32>>; N],
    before: &[Zeroizing<Vec<u32>>; N],
    cancel: &AtomicBool,
) -> Result<()> {
    g.validate(count)?;
    if cancel.load(Ordering::Acquire) {
        return Err("PLE cancelled completed candidate".into());
    }
    for i in 0..N {
        let full = BufferLayout::new(g.words(i))?.owned_words;
        if actual[i].len() != full || before[i].len() != full {
            return Err("PLE exact full owning snapshot".into());
        }
        for (j, (&a, &old)) in actual[i].iter().zip(before[i].iter()).enumerate() {
            if g.active(i, count, j) {
                if i >= STATUS {
                    if a != 1 {
                        return Err("PLE whole chunk finite/status rejection".into());
                    }
                } else if !f32::from_bits(a).is_finite() {
                    return Err("PLE whole chunk nonfinite rejection".into());
                }
                if i == GATES && !(0.0..=1.0).contains(&f32::from_bits(a)) {
                    return Err("PLE invalid sigmoid gate".into());
                }
            }
            if !g.changed(i, count, j) && a != old {
                return Err("PLE committed/input/inactive/padding changed".into());
            }
        }
    }
    Ok(())
}
pub fn admission() -> Result<(usize, usize)> {
    let g = Geometry::new(GatedResidualShape::new(4, 2560)?, false)?;
    let s = Geometry::new(GatedResidualShape::new(4, 2560)?, true)?;
    let main = g
        .footprint()?
        .checked_add(s.footprint()?)
        .and_then(|n| n.checked_add(32768))
        .ok_or("PLE main sum overflow")?;
    let source = [EMB, RESIDUAL, OLD].into_iter().try_fold(0usize, |n, i| {
        n.checked_add(BufferLayout::new(g.words(i))?.owned_bytes)
            .ok_or_else(|| Box::<dyn std::error::Error>::from("PLE source sum overflow"))
    })?;
    let largest = (0..N).try_fold(0usize, |n, i| {
        Ok::<_, Box<dyn std::error::Error>>(
            n.max(BufferLayout::new(g.words(i))?.owned_bytes)
                .max(BufferLayout::new(s.words(i))?.owned_bytes),
        )
    })?;
    let io = source
        .checked_add(largest)
        .and_then(|n| n.checked_add(64 * 2 * 512))
        .and_then(|n| n.checked_mul(2))
        .ok_or("PLE runtime duplicated audit sum overflow")?;
    if main > MAIN_CAP || io as u64 > IO_CAP {
        return Err("PLE aggregate full page admission".into());
    }
    Ok((main, io))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::ops::{PleHashConfig, PleTokenHistory};
    fn sample(g: Geometry, count: usize) -> [Zeroizing<Vec<u32>>; N] {
        let mut v = std::array::from_fn(|_| Zeroizing::new(Vec::new()));
        for (i, x) in v.iter_mut().enumerate() {
            let n = BufferLayout::new(g.words(i)).unwrap().owned_words;
            x.reserve_exact(n);
            x.resize(n, (-17.25f32).to_bits());
            for (j, y) in x.iter_mut().enumerate() {
                if g.active(i, count, j) {
                    *y = if i >= STATUS {
                        1
                    } else if i == GATES {
                        0.5f32.to_bits()
                    } else {
                        0.0f32.to_bits()
                    };
                }
            }
        }
        v
    }
    #[test]
    fn bounded_original_q5_row_decoder_rejects_last_nonfinite_block() {
        let mut row = Zeroizing::new([0u8; 110]);
        let mut decoded = Zeroizing::new([0f32; ROW_WIDTH]);
        decode_row(&row, &mut decoded).unwrap();
        assert!(decoded.iter().all(|x| *x == 0.));
        // Last of five original22-byte blocks contains a NaN F16 scale.
        row[88] = 0;
        row[89] = 0x7e;
        assert!(decode_row(&row, &mut decoded).is_err());
        // The gather copies a decoded row only AFTER this whole-row check.
        let source = include_str!("ple_chunk.rs");
        let gather = source
            .split("decode_row(&row, &mut decoded)?;")
            .nth(1)
            .unwrap();
        assert!(gather.starts_with("\n                for (d, &x)"));
    }
    #[test]
    fn exact_hash_matches_original_boundary_missing_eos_and_wrapping() {
        let cfg = PleHashConfig::qwen38();
        let meta = cfg.metadata();
        for old in [
            [-1, -1],
            [3, 7],
            [0, 7],
            [248046, 248044],
            [248044, 7],
            [248046, 7],
        ] {
            let mut h = PleTokenHistory::new(&cfg);
            for &v in &old {
                if v >= 0 {
                    cfg.rows_and_advance(v, &mut h).unwrap();
                }
            }
            for t in [0, 42, 248044, 248046, 248319] {
                let mut actual = Zeroizing::new([0; HEADS]);
                rows_into(meta, t, &old, &mut actual).unwrap();
                assert_eq!(&actual[..], cfg.rows(t as i32, &h).unwrap());
            }
        }
        let mut out = [0u32; HEADS];
        let mut bad = meta;
        let multipliers = [u64::MAX, u64::MAX - 2, u64::MAX - 4];
        bad.multipliers = &multipliers;
        rows_into(bad, 248319, &[248318, 248317], &mut out).unwrap();
        let mixed0 = 248319u64.wrapping_mul(multipliers[0]);
        let mixed1 = mixed0 ^ 248317u64.wrapping_mul(multipliers[1]);
        let mixed2 = mixed1 ^ 248318u64.wrapping_mul(multipliers[2]);
        for (h, &v) in out.iter().enumerate() {
            assert_eq!(
                u64::from(v),
                (if h < 8 { mixed1 } else { mixed2 }) % meta.vocab_sizes[h] + meta.offsets[h]
            );
        }
    }
    #[test]
    fn malformed_hash_rejects_before_mutating_any_private_output() {
        let cfg = PleHashConfig::qwen38();
        let meta = cfg.metadata();
        let mut out = [0xabcdef01u32; HEADS];
        for (t, h) in [(248320, [3, 7]), (3, [3, 248320])] {
            assert!(rows_into(meta, t, &h, &mut out).is_err());
            assert_eq!(out, [0xabcdef01; HEADS]);
        }
        let zero = [0u64; HEADS];
        let mut bad = meta;
        bad.vocab_sizes = &zero;
        assert!(rows_into(bad, 3, &[3, 7], &mut out).is_err());
        assert_eq!(out, [0xabcdef01; HEADS]);
    }
    #[test]
    fn exact_full_page_main_and_independent_runtime_caps() {
        assert_eq!(admission().unwrap(), (15_958_016, 3_538_944));
        for aligned in [false, true] {
            let g = Geometry::new(GatedResidualShape::new(4, 2560).unwrap(), aligned).unwrap();
            assert_eq!(g.footprint().unwrap(), 7_962_624);
            for i in 0..N {
                assert!(BufferLayout::new(g.words(i)).unwrap().owned_bytes >= 32768);
            }
        }
    }
    #[test]
    fn spans_reject_overflow_bad_count_and_short_full_max8_before_dispatch() {
        let g = Geometry::new(GatedResidualShape::new(3, 97).unwrap(), false).unwrap();
        for count in [0, 9, usize::MAX] {
            assert!(g.validate(count).is_err());
        }
        for field in [RESIDUAL, KEY, OUTPUT, 21] {
            let mut bad = g;
            bad.spans[field].words = bad.spans[field].at(7) + bad.spans[field].width - 1;
            assert!(bad.validate(1).is_err());
            let mut bad = g;
            bad.spans[field].base = usize::MAX;
            assert!(bad.validate(1).is_err());
        }
        let mut bad = g;
        bad.old.stride = 8;
        assert!(bad.validate(1).is_err());
        let mut bad = g;
        bad.next.words = 9;
        assert!(bad.validate(1).is_err());
    }
    #[test]
    fn all_statuses_last_token_readonly_full_padding_and_surplus_reject() {
        let g = Geometry::new(GatedResidualShape::new(3, 97).unwrap(), false).unwrap();
        let no = AtomicBool::new(false);
        for count in [1, 3, 5, 8] {
            let before = sample(g, count);
            let mut actual = sample(g, count);
            validate_snapshot(g, count, &actual, &before, &no).unwrap();
            for &i in &STATUS_FIELDS {
                let at = if i == 22 {
                    g.spans[i].base + g.channels() - 1
                } else {
                    g.spans[i].at(count - 1) + g.spans[i].width - 1
                };
                let old = actual[i][at];
                actual[i][at] = 0;
                assert!(validate_snapshot(g, count, &actual, &before, &no).is_err());
                actual[i][at] = old;
            }
            for i in [EMB, RESIDUAL, OLD, OUTPUT, NEXT, 21] {
                for at in [0, actual[i].len() - 1] {
                    let old = actual[i][at];
                    actual[i][at] ^= 1;
                    assert!(validate_snapshot(g, count, &actual, &before, &no).is_err());
                    actual[i][at] = old;
                }
            }
            let at = g.spans[OUTPUT].at(count - 1) + g.channels() - 1;
            actual[OUTPUT][at] = f32::NAN.to_bits();
            assert!(validate_snapshot(g, count, &actual, &before, &no).is_err());
            actual[OUTPUT][at] = 0;
            actual[OUTPUT].push(0);
            assert!(validate_snapshot(g, count, &actual, &before, &no).is_err());
            actual[OUTPUT].pop();
            assert!(validate_snapshot(g, count, &actual, &before, &AtomicBool::new(true)).is_err());
        }
    }
    #[test]
    fn failed_or_panicked_transaction_never_exposes_candidate() {
        let mut phase = Phase::Ready;
        assert!(
            transition(&mut phase, Phase::Ready, Phase::Queued, true, || Err(
                "fixed injected queue failure".into()
            ))
            .is_err()
        );
        assert!(phase == Phase::Poisoned);
        let mut phase = Phase::Ready;
        assert!(
            transition(&mut phase, Phase::Ready, Phase::Queued, true, || panic!(
                "fixed injected panic"
            ))
            .is_err()
        );
        assert!(phase == Phase::Poisoned);
        let mut phase = Phase::Queued;
        let called = std::cell::Cell::new(false);
        assert!(
            transition(&mut phase, Phase::Queued, Phase::Accepted, false, || {
                called.set(true);
                Ok(())
            })
            .is_err()
        );
        assert!(!called.get());
        assert!(phase == Phase::Poisoned);
    }
    #[test]
    fn tap_schedule_and_candidate_history_match_independent_core_for_partitions() {
        let channels = 3;
        let weights = [
            0.125, -0.25, 0.375, 0.5, 0.25, 0.125, -0.5, 0.75, 0.1, 0.2, 0.3, 0.4,
        ];
        for count in [1, 3, 5, 8] {
            let mut core = crate::ple_chunk_probe::test_core_history(channels).unwrap();
            let old = core.history().to_vec();
            let mut normalized = [[0f32; 3]; 8];
            for (t, row) in normalized.iter_mut().enumerate() {
                for (d, x) in row.iter_mut().enumerate() {
                    *x = ((t * 7 + d * 3) % 17) as f32 / 19. - 8. / 19.;
                }
            }
            for (t, input) in normalized.iter().enumerate().take(count) {
                let actual = core.step(input, &weights).unwrap();
                for c in 0..channels {
                    let mut expected = 0f64;
                    for tap in 0..4 {
                        let lag = (3 - tap) * 3;
                        let value = if t >= lag {
                            normalized[t - lag][c]
                        } else {
                            old[c * 9 + 9 + t - lag]
                        };
                        expected += f64::from(value) * f64::from(weights[c * 4 + tap]);
                    }
                    assert_eq!(actual[c], expected as f32);
                }
            }
            for c in 0..channels {
                for j in 0..9 {
                    let expected = if j + count < 9 {
                        old[c * 9 + j + count]
                    } else {
                        normalized[j + count - 9][c]
                    };
                    assert_eq!(core.history()[c * 9 + j], expected);
                }
            }
        }
    }
}
