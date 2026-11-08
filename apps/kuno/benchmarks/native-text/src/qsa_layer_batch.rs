//! Fixed-public full QSA layer laboratory, beginning at mixed GR columns.
//! Original six packed matrices, accepted causal chunk, candidate cache only.
//! Enqueue never reads an oracle, commits a cursor, or accesses a model hook.
use crate::{
    gdn_layer_batch::PackedMatrix,
    private_arena::{BufferId, BufferLayout, PrivateArena},
    qsa_chunk::{self as c, Span, Spans},
    qsa_gpu as qsa,
    quant_matvec::matmul::{BatchSpans, MatrixGeometry, packed_matmul},
    residual_batch_gpu::finite_span,
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::model::{AttentionKind, BoundGraph, GraphBindingId};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const N: usize = c::N + 8;
pub const INPUT: usize = c::N;
pub const OUTPUT: usize = INPUT + 1;
pub const PROJECTION_STATUS: usize = OUTPUT + 1;
pub const D: usize = 2560;
pub const MAX_BATCH: usize = 8;
pub const CAPACITY: usize = 2064;
pub const MAIN_CAP: usize = 64 * 1024 * 1024;
pub const IO_CAP: u64 = 64 * 1024 * 1024;
pub const PUBLIC_CAP: usize = 128 * 1024 * 1024;
pub const DECODED_CAP: usize = 256 * 1024 * 1024;
const TILE: usize = 8;
const INFO_BYTES: usize = 512;
// 5 matrix/finite pairs +8 preparation; postselector:3 attention +matrix/finite.
const PREP_DISPATCHES: usize = 18;
const ATTENTION_DISPATCHES: usize = 5;
pub const SUFFIXES: [&str; 6] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "attn_output.weight",
];
pub const PROJECTED: [usize; 6] = [c::RAWQ, c::RAWK, c::VALUE, c::RAWIQ, c::RAWIK, OUTPUT];
pub const NORMS: [&str; 4] = [
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
];
pub fn matrix_shapes() -> [(usize, usize); 6] {
    [
        (qsa::QUERY * 2, D),
        (qsa::KV, D),
        (qsa::KV, D),
        (qsa::INDEX_QUERY, D),
        (qsa::ID, D),
        (D, qsa::QUERY),
    ]
}
#[derive(Clone, Copy)]
pub struct Geometry {
    pub qsa: Spans,
    pub p: [Span; N],
    pub scalar: bool,
}
impl Geometry {
    pub fn new(previous: usize, count: usize, capacity: usize, scalar: bool) -> Result<Self> {
        let mut s = Spans {
            previous,
            count,
            capacity,
            p: [Span {
                base: 0,
                stride: 1,
                words: 1,
            }; c::N],
        };
        s.validate_context()?;
        let mut p = [Span {
            base: 0,
            stride: 1,
            words: 1,
        }; N];
        let widths = s.widths();
        let rows = s.rows();
        for i in 0..c::N {
            p[i] = if scalar {
                let words = match i {
                    c::OLD => capacity
                        .checked_mul(qsa::TOKEN)
                        .ok_or("QSA cache overflow")?,
                    c::POS => capacity,
                    c::GQ => qsa::QUERY,
                    c::GK => qsa::KV,
                    c::GI => qsa::INDEX_QUERY,
                    c::GP => qsa::ID,
                    c::SIN | c::COS => qsa::ROPE_TABLE_WORDS,
                    _ => widths[i],
                };
                Span {
                    base: 0,
                    stride: widths[i],
                    words,
                }
            } else {
                let base = 3 + i % 11;
                let stride = if i == c::POS {
                    2
                } else {
                    widths[i] + 5 + i % 7
                };
                Span {
                    base,
                    stride,
                    words: base + rows[i] * stride + 17,
                }
            };
            s.p[i] = p[i];
        }
        for (i, item) in p.iter_mut().enumerate().skip(INPUT) {
            let width = if i < PROJECTION_STATUS {
                D
            } else {
                matrix_shapes()[i - PROJECTION_STATUS].0
            };
            let base = if scalar { 0 } else { 3 + i % 11 };
            let stride = if scalar { width } else { width + 5 + i % 7 };
            *item = Span {
                base,
                stride,
                words: if scalar {
                    width
                } else {
                    base + MAX_BATCH * stride + 17
                },
            };
        }
        let g = Self { qsa: s, p, scalar };
        g.validate()?;
        Ok(g)
    }
    pub fn validate(self) -> Result<()> {
        self.qsa.validate_context()?;
        if self.scalar && self.qsa.count != 1 {
            return Err("QSA scalar count".into());
        }
        for i in 0..c::N {
            if self.p[i].base != self.qsa.p[i].base
                || self.p[i].stride != self.qsa.p[i].stride
                || self.p[i].words != self.qsa.p[i].words
            {
                return Err("QSA geometry mismatch".into());
            }
        }
        let widths = self.qsa.widths();
        let rows = self.qsa.rows();
        for i in 0..N {
            let width = if i < c::N {
                widths[i]
            } else if i < PROJECTION_STATUS {
                D
            } else {
                matrix_shapes()[i - PROJECTION_STATUS].0
            };
            let row_count = if i < c::N { rows[i] } else { MAX_BATCH };
            let row_count =
                if self.scalar && i != c::OLD && i != c::POS && !(c::GQ..=c::COS).contains(&i) {
                    1
                } else {
                    row_count
                };
            self.p[i].validate(row_count, width)?;
            if self.scalar && self.p[i].base != 0 {
                return Err("QSA scalar binding starts".into());
            }
        }
        Ok(())
    }
    pub fn footprint(self) -> Result<usize> {
        self.validate()?;
        self.p.iter().try_fold(0usize, |n, p| {
            Ok(n.checked_add(BufferLayout::new(p.words)?.owned_bytes)
                .ok_or("QSA main sum overflow")?)
        })
    }
    pub fn active_width(self, i: usize, token: usize) -> Result<usize> {
        let l = self.qsa.layout(token)?;
        Ok(match i {
            c::MEANS | c::POOL => l.complete * qsa::ID,
            c::SCORE | c::POOL_S | c::PN_S | c::SCORE_S => l.complete,
            c::LOGIT | c::PROB | c::LOGIT_S => l.logits_elements,
            c::SELECT => l.selected,
            INPUT | OUTPUT => D,
            PROJECTION_STATUS..N => matrix_shapes()[i - PROJECTION_STATUS].0,
            _ => self.qsa.widths()[i],
        })
    }
    pub fn active(self, i: usize, index: usize) -> bool {
        let p = self.p[i];
        if index < p.base {
            return false;
        }
        let t = (index - p.base) / p.stride;
        t < self.qsa.count && (index - p.base) % p.stride < self.active_width(i, t).unwrap_or(0)
    }
}
// Only the context check is shared with scalar geometry; the accepted operator
// still checks every MAX8 span in its original public validate method.
trait Context {
    fn validate_context(self) -> Result<()>;
}
impl Context for Spans {
    fn validate_context(self) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&self.count)
            || !(4..=qsa::MAX_CONTEXT).contains(&self.capacity)
            || self
                .previous
                .checked_add(self.count)
                .is_none_or(|n| n > self.capacity)
        {
            return Err("QSA layer context/count/capacity".into());
        }
        Ok(())
    }
}
pub struct Admission {
    pub main: usize,
    pub io: usize,
}
impl Admission {
    pub fn new(capacity: usize) -> Result<Self> {
        let b = Geometry::new(0, 8, capacity, false)?;
        let s = Geometry::new(0, 1, capacity, true)?;
        let main = b
            .footprint()?
            .checked_add(s.footprint()?)
            .ok_or("QSA aggregate main overflow")?;
        // Arena initialization drains one complete source at a time. Dispatch
        // phases keep at most one current scalar/batch input, original readonly
        // cache source, selector source and the largest readback. This is a
        // conservative sum even though checkpoints separate those phases.
        let largest =
            b.p.iter()
                .chain(s.p.iter())
                .try_fold(0usize, |n, p| -> Result<_> {
                    Ok(n.max(BufferLayout::new(p.words)?.owned_bytes))
                })?;
        let sources = [INPUT, c::OLD]
            .into_iter()
            .chain(c::SPLIT_S..c::N)
            .chain(PROJECTION_STATUS..N)
            .try_fold(0usize, |n, i| -> Result<_> {
                Ok(n.checked_add(BufferLayout::new(s.p[i].words)?.owned_bytes)
                    .ok_or("QSA source overflow")?)
            })?;
        let score_reads =
            [c::SCORE, c::SCORE_S]
                .into_iter()
                .try_fold(0usize, |n, i| -> Result<_> {
                    Ok(n.checked_add(BufferLayout::new(b.p[i].words)?.owned_bytes)
                        .ok_or("QSA reads overflow")?)
                })?;
        let prep = sources
            .checked_add(score_reads)
            .and_then(|n| n.checked_add(PREP_DISPATCHES * 2 * INFO_BYTES))
            .ok_or("QSA IO overflow")?;
        let attention = BufferLayout::new(b.p[c::SELECT].words)?
            .owned_bytes
            .checked_add(BufferLayout::new(b.p[PROJECTION_STATUS + 5].words)?.owned_bytes)
            .and_then(|n| n.checked_add(ATTENTION_DISPATCHES * 2 * INFO_BYTES))
            .ok_or("QSA IO overflow")?;
        // The completed-score audit drains all prior sources before selection;
        // final status/audit drains inference before one-page diagnostic reads.
        // Reset/close similarly inspect and erase one page at a time.
        let diagnostic = largest
            .checked_add(N * 2 * INFO_BYTES)
            .ok_or("QSA IO overflow")?;
        let io = prep
            .max(attention)
            .max(diagnostic)
            .checked_mul(2)
            .ok_or("QSA IO overflow")?;
        if main > MAIN_CAP || io as u64 > IO_CAP {
            return Err("QSA layer aggregate admission".into());
        }
        Ok(Self { main, io })
    }
}
pub struct PublicLayer {
    pub binding: GraphBindingId,
    pub matrices: [PackedMatrix; 6],
    pub gamma: [Zeroizing<Vec<f32>>; 4],
    pub public_bytes: usize,
}
fn public_budget(dtypes: [u32; 6]) -> Result<(usize, usize)> {
    let mut packed = 0usize;
    let mut decoded = 0usize;
    let mut largest_source = 0usize;
    for (dtype, (rows, cols)) in dtypes.into_iter().zip(matrix_shapes()) {
        let g = MatrixGeometry::new(dtype, rows, cols, TILE)?;
        let full_source = BufferLayout::new(g.total_bytes().div_ceil(4))?.owned_bytes;
        largest_source = largest_source.max(full_source);
        packed = packed.checked_add(full_source).ok_or("QSA public sum")?;
        decoded = decoded
            .checked_add(
                rows.checked_mul(cols)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or("QSA decoded sum")?,
            )
            .ok_or("QSA decoded sum")?;
    }
    // These original public vectors/tables have no separate GPU allocation in
    // this probe: owning duplicate fixture pages already belong to MAIN. The
    // conservative public-cache plan keeps their full pages explicit as well.
    for w in [
        qsa::HD,
        qsa::HD,
        qsa::ID,
        qsa::ID,
        qsa::ROPE_TABLE_WORDS,
        qsa::ROPE_TABLE_WORDS,
    ] {
        packed = packed
            .checked_add(BufferLayout::new(w)?.owned_bytes)
            .ok_or("QSA public sum")?;
    }
    let host_peak = decoded
        .checked_add(largest_source)
        .and_then(|n| n.checked_add(4 * qsa::ROPE_TABLE_WORDS * 4))
        .ok_or("QSA public host sum")?;
    if packed > PUBLIC_CAP || host_peak > DECODED_CAP {
        return Err("QSA public admission".into());
    }
    Ok((packed, decoded))
}
pub fn public_preflight(graph: BoundGraph<'_, '_>) -> Result<(usize, usize)> {
    if graph.layer_kind(3)? != AttentionKind::Qsa || graph.geometry().residual.width != D {
        return Err("QSA original graph geometry".into());
    }
    let mut dtypes = [0; 6];
    for (i, (suffix, (rows, cols))) in SUFFIXES.into_iter().zip(matrix_shapes()).enumerate() {
        let t = graph.tensor(&format!("blk.3.{suffix}"))?;
        if t.info().dimensions() != [cols as u64, rows as u64] {
            return Err("QSA original matrix geometry".into());
        }
        dtypes[i] = t.info().dtype().gguf_id();
    }
    for (name, w) in NORMS.into_iter().zip([qsa::HD, qsa::HD, qsa::ID, qsa::ID]) {
        let gamma = graph.static_vector(&format!("blk.3.{name}"))?;
        if gamma.len() != w || gamma.iter().any(|x| !x.is_finite()) {
            return Err("QSA original gamma".into());
        }
    }
    public_budget(dtypes)
}
impl PublicLayer {
    pub fn load(client: &ComputeClient<WgpuRuntime>, graph: BoundGraph<'_, '_>) -> Result<Self> {
        let (public_bytes, _) = public_preflight(graph)?;
        Admission::new(CAPACITY)?;
        let mut matrices = Vec::new();
        matrices.try_reserve_exact(6)?;
        for (suffix, (rows, cols)) in SUFFIXES.into_iter().zip(matrix_shapes()) {
            matrices.push(PackedMatrix::load(
                client,
                graph.tensor(&format!("blk.3.{suffix}"))?,
                rows,
                cols,
            )?);
        }
        let mut gamma: [Zeroizing<Vec<f32>>; 4] =
            std::array::from_fn(|_| Zeroizing::new(Vec::new()));
        for (v, suffix) in gamma.iter_mut().zip(NORMS) {
            let src = graph.static_vector(&format!("blk.3.{suffix}"))?;
            v.try_reserve_exact(src.len())?;
            v.extend_from_slice(src);
        }
        graph.verify_unchanged()?;
        Ok(Self {
            binding: graph.binding_id(),
            matrices: matrices.try_into().map_err(|_| "QSA matrix count")?,
            gamma,
            public_bytes,
        })
    }
}
pub struct Buffers(pub [BufferId; N]);
impl Buffers {
    pub fn chunk(&self) -> c::Buffers {
        c::Buffers(std::array::from_fn(|i| self.0[i]))
    }
}
pub fn binding_proof(
    arena: &PrivateArena,
    g: Geometry,
    b: &Buffers,
    public: &PublicLayer,
) -> Result<()> {
    g.validate()?;
    c::validate_pages(arena, &b.0, &g.p.map(|p| p.words))?;
    let client = arena.client()?;
    let handles =
        b.0.iter()
            .map(|&id| arena.handle(id).cloned())
            .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut all = handles;
    all.extend(public.matrices.iter().map(|m| m.handle.clone()));
    let resources = all
        .iter()
        .map(|h| client.get_resource(h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (i, r) in resources.iter().enumerate() {
        let v = r.resource();
        if all[i].stream != all[0].stream
            || all[i].offset_start.unwrap_or(0) != 0
            || all[i].offset_end.unwrap_or(0) != 0
            || v.offset != 0
            || v.size != all[i].size()
            || v.buffer.size() != v.size
            || resources[..i]
                .iter()
                .any(|x| x.resource().buffer == v.buffer)
        {
            return Err("QSA full physical public/private nonalias proof".into());
        }
    }
    Ok(())
}
fn arg(arena: &PrivateArena, b: &Buffers, i: usize) -> Result<ArrayArg<WgpuRuntime>> {
    Ok(unsafe {
        ArrayArg::from_raw_parts(
            arena.handle(b.0[i])?.clone(),
            arena.layout(b.0[i])?.owned_words,
        )
    })
}
/// Fixture projection accepts the same runtime BatchSpans as the separate
/// production adapter; CPU oracles, allocation, fences and commit stay outside.
pub fn project(
    arena: &PrivateArena,
    public: &PublicLayer,
    g: Geometry,
    b: &Buffers,
    which: usize,
) -> Result<()> {
    g.validate()?;
    let m = public.matrices.get(which).ok_or("QSA projection index")?;
    let input = if which == 5 { c::OUT } else { INPUT };
    let output = PROJECTED[which];
    let status = PROJECTION_STATUS + which;
    let spans = BatchSpans::new(
        g.qsa.count,
        0,
        g.p[input].base,
        g.p[input].stride,
        g.p[output].base,
        g.p[output].stride,
    );
    m.geometry.validate_spans(
        &spans,
        m.words,
        arena.layout(b.0[input])?.owned_words,
        arena.layout(b.0[output])?.owned_words,
    )?;
    let c = arena.client()?;
    let packed = unsafe { ArrayArg::from_raw_parts(m.handle.clone(), m.words) };
    if g.scalar {
        crate::quant_matvec::cooperative::cooperative_matvec::launch::<WgpuRuntime>(
            c,
            CubeCount::Static(m.geometry.rows() as u32, 1, 1),
            CubeDim::new_1d(64),
            packed,
            arg(arena, b, input)?,
            arg(arena, b, output)?,
            m.geometry.rows(),
            m.geometry.columns(),
            m.geometry.dtype(),
            0,
        );
    } else {
        packed_matmul::launch::<WgpuRuntime>(
            c,
            m.geometry.grid(),
            CubeDim::new_1d(64),
            packed,
            arg(arena, b, input)?,
            arg(arena, b, output)?,
            spans.count(),
            spans.packed_first(),
            spans.input_first(),
            spans.input_stride(),
            spans.output_first(),
            spans.output_stride(),
            m.geometry.rows(),
            m.geometry.columns(),
            m.geometry.dtype(),
            TILE,
        );
    }
    finite_span::launch::<WgpuRuntime>(
        c,
        CubeCount::Static(m.geometry.rows().div_ceil(64) as u32, MAX_BATCH as u32, 1),
        CubeDim::new_1d(64),
        arg(arena, b, output)?,
        arg(arena, b, status)?,
        g.qsa.count,
        g.p[output].base,
        g.p[output].stride,
        g.p[status].base,
        g.p[status].stride,
        m.geometry.rows(),
    );
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Scored,
    Queued,
    Accepted,
    Poisoned,
}
fn transition<T>(
    phase: &mut Phase,
    expected: Phase,
    next: Phase,
    admitted: bool,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if *phase != expected || !admitted {
        *phase = Phase::Poisoned;
        return Err("QSA lifecycle/binding/cancel".into());
    }
    *phase = Phase::Poisoned;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(Ok(v)) => {
            *phase = next;
            Ok(v)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err("QSA layer panicked; erase candidate".into()),
    }
}
pub struct LayerBatch {
    // Sealed together with ready: no caller can replace the packet/owners
    // between preparation, attention, output projection and acceptance.
    geometry: Geometry,
    buffers: Buffers,
    phase: Phase,
    binding: GraphBindingId,
    ready: Option<c::Prepared>,
}
impl LayerBatch {
    pub fn new(graph: BoundGraph<'_, '_>, g: Geometry, b: Buffers) -> Result<Self> {
        g.validate()?;
        if g.scalar {
            return Err("QSA batch scalar geometry".into());
        }
        Ok(Self {
            geometry: g,
            buffers: b,
            phase: Phase::Ready,
            binding: graph.binding_id(),
            ready: None,
        })
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn prepare(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicLayer,
        positions: &[u32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        let admitted = !cancel.load(Ordering::Acquire)
            && graph.binding_id() == self.binding
            && public.binding == self.binding;
        let mut phase = self.phase;
        let result = transition(&mut phase, Phase::Ready, Phase::Scored, admitted, || {
            graph.verify_unchanged()?;
            self.geometry.qsa.validate_positions(positions)?;
            binding_proof(arena, self.geometry, &self.buffers, public)?;
            for which in 0..5 {
                if cancel.load(Ordering::Acquire) {
                    return Err("QSA cancelled projection".into());
                }
                project(arena, public, self.geometry, &self.buffers, which)?;
            }
            Ok(c::enqueue_prepare(
                arena,
                self.geometry.qsa,
                self.buffers.chunk(),
                positions,
                cancel,
            )?)
        });
        self.phase = phase;
        self.ready = Some(result?);
        Ok(())
    }
    pub fn attention(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicLayer,
        ids: Zeroizing<Vec<u32>>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut phase = self.phase;
        let admitted = !cancel.load(Ordering::Acquire)
            && graph.binding_id() == self.binding
            && public.binding == self.binding;
        let result = transition(&mut phase, Phase::Scored, Phase::Queued, admitted, || {
            graph.verify_unchanged()?;
            binding_proof(arena, self.geometry, &self.buffers, public)?;
            let ready = self.ready.as_ref().ok_or("QSA prepared owner missing")?;
            c::enqueue_attention(arena, ready, ids, cancel)?;
            project(arena, public, self.geometry, &self.buffers, 5)?;
            if cancel.load(Ordering::Acquire) {
                return Err("QSA cancelled output projection".into());
            }
            Ok(())
        });
        self.phase = phase;
        result
    }
    /// Caller first completes owning reads, drops Bytes leases and accepts the
    /// checked cleanup audit. Accepted only describes this fixture candidate.
    pub fn accept(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        data: &[Zeroizing<Vec<u32>>; N],
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
                    return Err("QSA changed graph".into());
                }
                graph.verify_unchanged()?;
                c::validate_pages(arena, &self.buffers.0, &self.geometry.p.map(|p| p.words))?;
                validate_snapshot(self.geometry, data, before, cancel)
            },
        );
        self.phase = phase;
        result
    }
}
impl Drop for LayerBatch {
    fn drop(&mut self) {
        self.geometry.qsa.count.zeroize();
        self.geometry.qsa.previous.zeroize();
        self.phase = Phase::Poisoned;
    }
}
pub fn validate_snapshot(
    g: Geometry,
    data: &[Zeroizing<Vec<u32>>; N],
    before: &[Zeroizing<Vec<u32>>; N],
    cancel: &AtomicBool,
) -> Result<()> {
    g.validate()?;
    if cancel.load(Ordering::Acquire) {
        return Err("QSA cancelled completion".into());
    }
    for (i, words) in data.iter().enumerate() {
        if words.len() != BufferLayout::new(g.p[i].words)?.owned_words
            || before[i].len() != g.p[i].words
        {
            return Err("QSA exact owning snapshot".into());
        }
        for (j, &bits) in words.iter().enumerate() {
            let status = (c::SPLIT_S..c::N).contains(&i) || i >= PROJECTION_STATUS;
            let readonly = (c::OLD..=c::COS).contains(&i) || i == INPUT;
            let active = g.active(i, j);
            if (readonly || !active) && bits != before[i].get(j).copied().unwrap_or(0) {
                return Err("QSA readonly/gap/inactive/padding mutation".into());
            }
            if active && !readonly {
                if status {
                    if bits != 1 {
                        return Err("QSA incomplete/nonfinite stage".into());
                    }
                } else if i == c::SELECT { /* index validation below */
                } else if !f32::from_bits(bits).is_finite() {
                    return Err("QSA nonfinite active candidate".into());
                }
            }
        }
    }
    let ids = &data[c::SELECT][..g.p[c::SELECT].words];
    if g.scalar {
        let layout = g.qsa.layout(0)?;
        qsa::validate_selected(layout, &ids[..layout.selected])?;
    } else {
        c::validate_selected(g.qsa, ids)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_pages_are_fixed_and_4096_dual_fixture_fails_closed() {
        let a = Admission::new(CAPACITY).unwrap();
        assert_eq!(a.main, 58_818_560);
        assert!(a.io <= IO_CAP as usize);
        let large = Geometry::new(0, 8, 4096, false)
            .unwrap()
            .footprint()
            .unwrap()
            + Geometry::new(0, 1, 4096, true)
                .unwrap()
                .footprint()
                .unwrap();
        assert_eq!(large, 100_925_440);
        assert!(Admission::new(4096).is_err());
        for p in [0, 3, 4, 2047, 2049, 2051] {
            for n in [1, 3, 5, 8] {
                let g = Geometry::new(p, n, CAPACITY, false).unwrap();
                assert_eq!(
                    g.footprint().unwrap()
                        + Geometry::new(p, 1, CAPACITY, true)
                            .unwrap()
                            .footprint()
                            .unwrap(),
                    a.main
                );
                g.qsa.validate().unwrap();
            }
        }
        assert_eq!(
            public_budget([12, 12, 14, 30, 30, 12]).unwrap(),
            (59_899_904, 205_783_040)
        );
    }
    #[test]
    fn context_geometry_and_overflow_reject_before_mutation() {
        for (p, n, cap) in [
            (0, 0, 2064),
            (0, 9, 2064),
            (usize::MAX, 8, 2064),
            (2060, 8, 2064),
            (0, 1, 4097),
        ] {
            assert!(Geometry::new(p, n, cap, false).is_err());
        }
        let mut g = Geometry::new(4, 8, CAPACITY, false).unwrap();
        g.p[INPUT].stride = usize::MAX;
        assert!(g.validate().is_err());
        let mut g = Geometry::new(4, 8, CAPACITY, false).unwrap();
        g.p[c::RAWQ].words -= 1;
        assert!(g.validate().is_err());
    }
    #[test]
    fn every_fallible_or_panicking_stage_poison_is_terminal() {
        let mut p = Phase::Ready;
        transition(&mut p, Phase::Ready, Phase::Scored, true, || Ok(())).unwrap();
        assert_eq!(p, Phase::Scored);
        assert!(
            transition::<()>(&mut p, Phase::Scored, Phase::Queued, true, || Err(
                "injected failure".into()
            ))
            .is_err()
        );
        assert_eq!(p, Phase::Poisoned);
        assert!(transition(&mut p, Phase::Ready, Phase::Queued, true, || Ok(())).is_err());
        let mut p = Phase::Queued;
        assert!(
            transition::<()>(&mut p, Phase::Queued, Phase::Accepted, true, || panic!(
                "injected public panic"
            ))
            .is_err()
        );
        assert_eq!(p, Phase::Poisoned);
        let mut p = Phase::Queued;
        assert!(transition(&mut p, Phase::Queued, Phase::Accepted, false, || Ok(())).is_err());
        assert_eq!(p, Phase::Poisoned);
    }
    type Images = [Zeroizing<Vec<u32>>; N];
    fn accepted_images(g: Geometry) -> (Images, Images) {
        let before: Images = std::array::from_fn(|i| {
            Zeroizing::new(vec![
                if i == c::SELECT {
                    0
                } else {
                    (-173.25f32).to_bits()
                };
                g.p[i].words
            ])
        });
        let mut data: Images = std::array::from_fn(|i| {
            let page = BufferLayout::new(g.p[i].words).unwrap();
            let mut v = Zeroizing::new(vec![0; page.owned_words]);
            v[..g.p[i].words].copy_from_slice(&before[i]);
            for (j, x) in v.iter_mut().enumerate() {
                if g.active(i, j) && !(c::OLD..=c::COS).contains(&i) && i != INPUT {
                    *x = if (c::SPLIT_S..c::N).contains(&i) || i >= PROJECTION_STATUS {
                        1
                    } else {
                        0f32.to_bits()
                    };
                }
            }
            v
        });
        for t in 0..g.qsa.count {
            let layout = g.qsa.layout(t).unwrap();
            for a in 0..layout.selected {
                data[c::SELECT][g.p[c::SELECT].at(t) + a] = a as u32;
            }
        }
        (before, data)
    }
    #[test]
    fn every_seventeen_stage_flag_and_late_nonfinite_reject_whole_chunk() {
        let g = Geometry::new(4, 8, 12, false).unwrap();
        let (before, mut data) = accepted_images(g);
        let cancel = AtomicBool::new(false);
        validate_snapshot(g, &data, &before, &cancel).unwrap();
        for i in (c::SPLIT_S..c::N).chain(PROJECTION_STATUS..N) {
            let at = g.p[i].at(7);
            let old = data[i][at];
            data[i][at] = 0;
            assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
            data[i][at] = old;
        }
        for i in [c::RAWIK, c::CAND, c::OUT, OUTPUT] {
            let at = g.p[i].at(7);
            let old = data[i][at];
            data[i][at] = f32::NAN.to_bits();
            assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
            data[i][at] = old;
        }
        cancel.store(true, Ordering::Release);
        assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
    }
    #[test]
    fn owning_lengths_immutable_gaps_padding_and_future_selected_ids_reject() {
        let g = Geometry::new(4, 3, 12, false).unwrap();
        let (before, mut data) = accepted_images(g);
        let cancel = AtomicBool::new(false);
        validate_snapshot(g, &data, &before, &cancel).unwrap();
        for (i, at) in [
            (c::OLD, g.p[c::OLD].at(0)),
            (INPUT, g.p[INPUT].at(0)),
            (c::SIN, 0),
            (OUTPUT, 0),
            (OUTPUT, g.p[OUTPUT].at(7)),
            (OUTPUT, g.p[OUTPUT].words),
        ] {
            let old = data[i][at];
            data[i][at] ^= 1;
            assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
            data[i][at] = old;
        }
        let at = g.p[c::SELECT].at(2);
        let old = data[c::SELECT][at];
        data[c::SELECT][at] = 7;
        assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
        data[c::SELECT][at] = old;
        let last = data[OUTPUT].pop().unwrap();
        assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
        data[OUTPUT].push(last);
        data[OUTPUT].push(0);
        assert!(validate_snapshot(g, &data, &before, &cancel).is_err());
    }
    #[test]
    fn original_info_inventory_includes_both_metadata_copies() {
        for address in [cubecl::ir::AddressType::U32, cubecl::ir::AddressType::U64] {
            let mut info = cubecl::InfoBuilder::default();
            for _ in 0..8 {
                info.metadata
                    .register_array(16_777_216, 16_777_216, address);
            }
            for _ in 0..21 {
                info.scalars.push(1u64);
            }
            assert!(info.finish(address).data.len() * 8 <= INFO_BYTES);
        }
        assert_eq!(PREP_DISPATCHES, 18);
        assert_eq!(ATTENTION_DISPATCHES, 5);
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
                        .kernel_name("original_qsa_projection"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                uint,
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn original_layer_six_projection_shapes_validate_checked_unchecked_wgsl() {
        use crate::quant_matvec::{BF16, Q4_K, Q6_K};
        use cubecl::server::ExecutionMode;
        for (dtype, (rows, columns)) in [Q4_K, Q4_K, Q6_K, BF16, BF16, Q4_K]
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
