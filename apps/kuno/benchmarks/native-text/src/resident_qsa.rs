//! Buffer-only resident QSA orchestration for the isolated Rust laboratory.
//! Requires admitted BoundGraph, same PrivateArena client/stream/epoch and
//! immutable public parameters. Uses resident_gdn's shared SequenceKey and
//! ResidentProjector contracts. No core hook, device initialization or logs.
//!
//! Candidate K/V/rawIK is distinct from readonly committed cache prefix. A
//! bounded append writes ONLY its unused suffix, and its position, during
//! enqueue; both remain uncommitted until WHOLE-token acceptance. Commit is a
//! host prefix-length advance, with no new GPU operation. Failure/cancellation
//! poisons the entire sequence; erase/close all arena resources, never retry a
//! layer or preserve an idle prefix. No output/status receipt is success proof.
//!
//! The host selector may read transient block scores/status only. It must use
//! the exact core selector, upload into a preallocated private buffer, clear
//! every host/staging copy on every path, retain no private handles, and finish
//! all work on the same stream. Runtime readback/upload/uniform pools require
//! root's separate erasure audit; this staging contract proves no secure wipe.
//! Projector finite/completion statuses also belong to whole-token acceptance.
//! Fixed history capacity avoids repeated full-cache copies or per-token
//! allocations. Positions/counts remain runtime; no request-derived comptime.

use crate::private_arena::{self, BufferId, BufferLayout, PrivateArena};
use crate::qsa_gpu::{self as qsa, QsaLayout};
pub use crate::resident_gdn::{
    BoundProjection, PublicGdnCache, ResidentProjector, SequenceKey, StatusRegion,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::{AttentionKind, BoundGraph, GraphBindingId, GraphGeometry};
use kuno_text::{Error, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const N: usize = 31;
const CACHE: usize = 0;
const POSITIONS: usize = 1;
const RAWQ: usize = 2;
const PACKED: usize = 3;
const QUERY: usize = 4;
const RAWK: usize = 5;
const KEY: usize = 6;
const VALUE: usize = 7;
const RAWIQ: usize = 8;
const IQ: usize = 9;
const RAWIK: usize = 10;
const TOKEN: usize = 11;
const MEANS: usize = 12;
const POOLED: usize = 13;
const SCORES: usize = 14;
const SELECTED: usize = 15;
const LOGITS: usize = 16;
const PROBS: usize = 17;
const GATED: usize = 18;
const STATUS: usize = 19;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn fault(s: &'static str) -> Error {
    Error::InvalidFormat(s)
}
fn arena_fault(_: private_arena::ArenaError) -> Error {
    fault("resident QSA arena unavailable")
}
fn identity() -> Result<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(fault("resident QSA cancelled"))
    } else {
        Ok(())
    }
}

/// Immutable PUBLIC model vectors; exact already-converted gamma and f32
/// public Rust-generated sin/cos tables. Public cache API never receives private data.
pub struct PublicQsaParameters {
    binding: GraphBindingId,
    layer: usize,
    identity: u64,
    vectors: [Handle; 6], // qgamma256,kgamma256,iqgamma128,ikgamma128,sin128K,cos128K
}
impl PublicQsaParameters {
    pub fn prepare(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        cache: &mut impl PublicGdnCache,
    ) -> Result<Self> {
        StageLayout::from_graph(graph, layer)?;
        let names = [
            format!("blk.{layer}.attn_q_norm.weight"),
            format!("blk.{layer}.attn_k_norm.weight"),
            format!("blk.{layer}.indexer.q_norm.weight"),
            format!("blk.{layer}.indexer.k_norm.weight"),
        ];
        let values = [
            graph.static_vector(&names[0])?,
            graph.static_vector(&names[1])?,
            graph.static_vector(&names[2])?,
            graph.static_vector(&names[3])?,
        ];
        validate_public_vectors(values)?;
        let binding = graph.binding_id();
        let identity = identity()?;
        let mut vectors = Vec::new();
        vectors
            .try_reserve_exact(6)
            .map_err(|_| Error::LimitExceeded("public QSA metadata"))?;
        for (name, values) in names.iter().zip(values) {
            vectors.push(cache.upload_public_vector(binding, name, values)?);
        }
        let (sin_table, cos_table) = qsa::rope_tables();
        vectors.push(cache.upload_public_vector(
            binding,
            "kuno_public_qsa_rope_sin32_theta1e7_positions4096",
            sin_table,
        )?);
        vectors.push(cache.upload_public_vector(
            binding,
            "kuno_public_qsa_rope_cos32_theta1e7_positions4096",
            cos_table,
        )?);
        let vectors: [Handle; 6] = vectors
            .try_into()
            .map_err(|_| fault("public QSA vector count"))?;
        for (h, n) in vectors.iter().zip([
            qsa::HD,
            qsa::HD,
            qsa::ID,
            qsa::ID,
            qsa::ROPE_TABLE_WORDS,
            qsa::ROPE_TABLE_WORDS,
        ]) {
            bounded_view(h, 0, n)?;
        }
        Ok(Self {
            binding,
            layer,
            identity,
            vectors,
        })
    }
}
fn validate_public_vectors(v: [&[f32]; 4]) -> Result<()> {
    if v.iter()
        .zip([qsa::HD, qsa::HD, qsa::ID, qsa::ID])
        .any(|(v, n)| v.len() != n)
    {
        return Err(Error::InvalidBufferLength);
    }
    if v.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
        return Err(fault("invalid public QSA gamma"));
    }
    Ok(())
}

/// Exact core-selected IDs with no public mutator/Debug. Drop clears the full
/// owned u32 capacity, including unused tail. Selector owns score/readback wipe.
pub struct PrivateSelection {
    layout: QsaLayout,
    ids: Vec<u32>,
}
impl PrivateSelection {
    pub fn from_scores(layout: QsaLayout, scores: &[f32], status: &[u32]) -> Result<Self> {
        let ids = qsa::select_blocks(layout, scores, status).map_err(fault)?;
        Ok(Self { layout, ids })
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}
impl Drop for PrivateSelection {
    fn drop(&mut self) {
        // SAFETY: capacity denotes writable allocated u32 storage; writing
        // zeros is valid even in the uninitialized portion beyond Vec::len.
        for i in 0..self.ids.capacity() {
            unsafe { std::ptr::write_volatile(self.ids.as_mut_ptr().add(i), 0) }
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}
pub trait ResidentQsaSelector {
    /// Fence/read only logical score/status prefixes. Return via from_scores;
    /// source bytes/floats/status staging must be cleared even on panic/error.
    fn select_transient(
        &mut self,
        arena: &PrivateArena,
        layout: QsaLayout,
        scores: StatusRegion,
        status: StatusRegion,
        cancel: &AtomicBool,
    ) -> Result<PrivateSelection>;
    /// Upload exactly selected IDs into the first ids.len words of the existing
    /// private target, on same stream; no retained slices/handles/staging copies.
    fn upload_selected(
        &mut self,
        arena: &PrivateArena,
        target: BufferId,
        selection: &PrivateSelection,
        cancel: &AtomicBool,
    ) -> Result<()>;
}

#[derive(Clone, Copy, PartialEq)]
pub struct StageLayout {
    pub input_words: usize,
    pub max_context: usize,
    pub epsilon: f32,
}
impl StageLayout {
    pub fn from_graph(graph: BoundGraph<'_, '_>, layer: usize) -> Result<Self> {
        if graph.layer_kind(layer)? != AttentionKind::Qsa {
            return Err(fault("resident QSA requires QSA layer"));
        }
        Self::from_geometry(graph.geometry())
    }
    fn from_geometry(g: GraphGeometry) -> Result<Self> {
        let s = g.qsa;
        if [
            s.query_heads,
            s.kv_heads,
            s.head_dim,
            s.indexer_heads,
            s.indexer_dim,
            s.block_size,
            s.token_budget,
            s.rotary_dims,
        ] != [
            qsa::QH,
            qsa::KVH,
            qsa::HD,
            qsa::IH,
            qsa::ID,
            qsa::BLOCK,
            qsa::BUDGET,
            64,
        ] || s.rope_theta != qsa::THETA
            || s.epsilon != qsa::EPSILON
            || g.epsilon != qsa::EPSILON
            || g.residual.width == 0
            || g.max_context == 0
            || g.max_context > qsa::MAX_CONTEXT
        {
            return Err(fault("unsupported resident QSA geometry"));
        }
        Ok(Self {
            input_words: g.residual.width,
            max_context: g.max_context,
            epsilon: g.epsilon,
        })
    }
    pub fn allocation_words(self) -> [usize; N] {
        let l = QsaLayout::new(self.max_context).expect("validated public QSA geometry");
        let complete = l.complete.max(1);
        // A whole-block final context may have fewer selected tokens than an
        // EARLIER incomplete tail. Capacity must cover every intermediate step.
        let selected = self.max_context.min(qsa::BUDGET + qsa::BLOCK - 1);
        let logits = qsa::QH * selected;
        [
            self.max_context * qsa::TOKEN,
            self.max_context,
            qsa::QUERY * 2,
            qsa::QUERY * 2,
            qsa::QUERY,
            qsa::KV,
            qsa::KV,
            qsa::KV,
            qsa::INDEX_QUERY,
            qsa::INDEX_QUERY,
            qsa::ID,
            qsa::TOKEN,
            complete * qsa::ID,
            complete * qsa::ID,
            complete,
            selected,
            logits,
            logits,
            qsa::QUERY,
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            complete,
            complete,
            complete,
            logits,
            qsa::QH,
            qsa::QUERY,
            qsa::TOKEN + 1,
        ]
    }
    pub fn owned_bytes(self) -> Result<usize> {
        self.allocation_words()
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(BufferLayout::new(n).map_err(arena_fault)?.owned_bytes)
                    .ok_or(Error::ArithmeticOverflow)
            })
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Ready(u64),
    Pending(u64),
    Poisoned,
}
struct Lifecycle {
    phase: Phase,
    max: usize,
}
impl Lifecycle {
    fn new(max: usize) -> Self {
        Self {
            phase: Phase::Ready(0),
            max,
        }
    }
    fn begin(&mut self, p: u64) -> Result<QsaLayout> {
        if self.phase != Phase::Ready(p) || p >= self.max as u64 {
            self.phase = Phase::Poisoned;
            return Err(fault("resident QSA token order/state"));
        }
        let l = QsaLayout::new(p as usize + 1).map_err(fault)?;
        self.phase = Phase::Pending(p);
        Ok(l)
    }
    fn commit(&mut self, p: u64) -> Result<()> {
        if self.phase != Phase::Pending(p) {
            self.phase = Phase::Poisoned;
            return Err(fault("resident QSA stale commit"));
        }
        self.phase = Phase::Ready(p.checked_add(1).ok_or(Error::ArithmeticOverflow)?);
        Ok(())
    }
}
struct Projections<'f> {
    query: BoundProjection<'f>,
    key: BoundProjection<'f>,
    value: BoundProjection<'f>,
    iq: BoundProjection<'f>,
    ik: BoundProjection<'f>,
    output: BoundProjection<'f>,
}
fn projection<'f>(
    g: BoundGraph<'_, 'f>,
    name: &str,
    inputs: usize,
    outputs: usize,
) -> Result<BoundProjection<'f>> {
    let tensor = g.tensor(name)?;
    let matrix = Matrix::new(tensor)?;
    if matrix.inputs() != inputs || matrix.outputs() != outputs || matrix.experts() != 1 {
        return Err(fault("resident QSA projection geometry"));
    }
    Ok(BoundProjection { tensor, matrix })
}
pub struct PendingQsa {
    stage: u64,
    sequence: SequenceKey,
    position: u64,
    output: BufferId,
    statuses: [StatusRegion; 12],
}
impl PendingQsa {
    /// Same11 QSA stages plus candidate append+position. Skip empty prefixes.
    pub fn statuses(&self) -> &[StatusRegion; 12] {
        &self.statuses
    }
    pub fn output(&self) -> BufferId {
        self.output
    }
    pub fn position(&self) -> u64 {
        self.position
    }
}
/// Read ONLY receipt logical prefixes; allocated capacity/padding intentionally
/// remains zero. Empty pool stages are skipped, not interpreted as failures.
pub fn validate_token_status(layout: QsaLayout, stages: [&[u32]; 12]) -> Result<()> {
    qsa::validate_token_status(layout, std::array::from_fn(|j| stages[j])).map_err(fault)?;
    if stages[11].len() != qsa::TOKEN + 1 || stages[11].iter().any(|&s| s != 1) {
        return Err(fault("QSA append stage failed"));
    }
    Ok(())
}
pub struct ResidentQsa<'f> {
    binding: GraphBindingId,
    layer: usize,
    sequence: SequenceKey,
    identity: u64,
    public_identity: u64,
    layout: StageLayout,
    ids: [BufferId; N],
    projections: Projections<'f>,
    lifecycle: Lifecycle,
}
impl<'f> ResidentQsa<'f> {
    pub fn allocate(
        graph: BoundGraph<'_, 'f>,
        layer: usize,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        public: &PublicQsaParameters,
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        let layout = StageLayout::from_graph(graph, layer)?;
        if public.binding != graph.binding_id() || public.layer != layer {
            return Err(fault("resident QSA public binding mismatch"));
        }
        if arena
            .charged_bytes()
            .checked_add(layout.owned_bytes()?)
            .ok_or(Error::ArithmeticOverflow)?
            > arena.budget_bytes()
        {
            return Err(Error::LimitExceeded("resident QSA arena budget"));
        }
        let prefix = format!("blk.{layer}.");
        let inputs = layout.input_words;
        let projections = Projections {
            query: projection(
                graph,
                &format!("{prefix}attn_q.weight"),
                inputs,
                qsa::QUERY * 2,
            )?,
            key: projection(graph, &format!("{prefix}attn_k.weight"), inputs, qsa::KV)?,
            value: projection(graph, &format!("{prefix}attn_v.weight"), inputs, qsa::KV)?,
            iq: projection(
                graph,
                &format!("{prefix}indexer.q_proj.weight"),
                inputs,
                qsa::INDEX_QUERY,
            )?,
            ik: projection(
                graph,
                &format!("{prefix}indexer.k_proj.weight"),
                inputs,
                qsa::ID,
            )?,
            output: projection(
                graph,
                &format!("{prefix}attn_output.weight"),
                qsa::QUERY,
                inputs,
            )?,
        };
        let mut ids = Vec::new();
        ids.try_reserve_exact(N)
            .map_err(|_| Error::LimitExceeded("resident QSA metadata"))?;
        for n in layout.allocation_words() {
            ids.push(arena.allocate_zeroed(n).map_err(arena_fault)?);
        }
        let ids: [BufferId; N] = ids
            .try_into()
            .map_err(|_| fault("resident QSA allocation count"))?;
        validate_regions(arena, &ids, public)?;
        Ok(Self {
            binding: graph.binding_id(),
            layer,
            sequence,
            identity: identity()?,
            public_identity: public.identity,
            layout,
            ids,
            projections,
            lifecycle: Lifecycle::new(layout.max_context),
        })
    }
    pub fn layout(&self) -> StageLayout {
        self.layout
    }
    pub fn requires_erasure_if_abandoned(&self) -> bool {
        !matches!(self.lifecycle.phase, Phase::Ready(_))
    }
    pub fn poison(&mut self) {
        self.lifecycle.phase = Phase::Poisoned;
    }
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicQsaParameters,
        projector: &mut impl ResidentProjector,
        selector: &mut impl ResidentQsaSelector,
        input: BufferId,
        output: BufferId,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PendingQsa> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.enqueue_checked(
                graph, arena, public, projector, selector, input, output, position, cancel,
            )
        }));
        match result {
            Ok(Ok(p)) => Ok(p),
            Ok(Err(e)) => {
                self.poison();
                Err(e)
            }
            Err(_) => {
                self.poison();
                Err(fault("resident QSA enqueue failed"))
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicQsaParameters,
        projector: &mut impl ResidentProjector,
        selector: &mut impl ResidentQsaSelector,
        input: BufferId,
        output: BufferId,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PendingQsa> {
        cancelled(cancel)?;
        if graph.binding_id() != self.binding
            || public.binding != self.binding
            || public.layer != self.layer
            || public.identity != self.public_identity
        {
            return Err(fault("resident QSA binding changed"));
        }
        graph.verify_unchanged()?;
        if StageLayout::from_graph(graph, self.layer)? != self.layout {
            return Err(fault("resident QSA geometry changed"));
        }
        self.validate_ids(arena, input, output)?;
        let l = self.lifecycle.begin(position)?;
        let p = position as u32;
        let ids = &self.ids;
        let counts = [
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            l.complete,
            l.complete,
            l.complete,
            l.logits_elements,
            qsa::QH,
            qsa::QUERY,
            qsa::TOKEN + 1,
        ];
        let statuses = std::array::from_fn(|j| StatusRegion {
            buffer: ids[STATUS + j],
            words: counts[j],
        });
        // Prove the exact logical prefixes passed to every dispatch, including
        // capacities larger than the current score/selection/logit counts.
        self.validate_launch_lengths(arena, l, counts)?;
        let client = arena.client().map_err(arena_fault)?;
        for s in statuses {
            let span = arena.layout(s.buffer).map_err(arena_fault)?;
            private_arena::erase_full_span::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(span.workgroups_x, span.workgroups_y, 1),
                CubeDim::new_1d(64),
                unsafe {
                    ArrayArg::from_raw_parts(
                        arena.handle(s.buffer).map_err(arena_fault)?.clone(),
                        span.owned_words,
                    )
                },
            );
        }
        for (projection, target) in [
            (self.projections.query, ids[RAWQ]),
            (self.projections.key, ids[RAWK]),
            (self.projections.value, ids[VALUE]),
            (self.projections.iq, ids[RAWIQ]),
            (self.projections.ik, ids[RAWIK]),
        ] {
            cancelled(cancel)?;
            projector.enqueue(projection, arena, input, target, cancel)?;
        }
        cancelled(cancel)?;
        qsa::split_query_gate::launch::<WgpuRuntime>(
            client,
            grid(qsa::QUERY * 2),
            CubeDim::new_1d(64),
            arg(arena, ids[RAWQ], qsa::QUERY * 2)?,
            arg(arena, ids[PACKED], qsa::QUERY * 2)?,
            arg(arena, statuses[0].buffer, counts[0])?,
        );
        // Query view occupies the first6144; raw gate remains second6144.
        self.norm(
            arena,
            public,
            raw_arg(
                bounded_view(
                    arena.handle(ids[PACKED]).map_err(arena_fault)?,
                    0,
                    qsa::QUERY,
                )?,
                qsa::QUERY,
            ),
            ids[QUERY],
            statuses[1],
            0,
            qsa::QH,
            qsa::HD,
            false,
            l,
            p,
        )?;
        self.norm(
            arena,
            public,
            arg(arena, ids[RAWK], qsa::KV)?,
            ids[KEY],
            statuses[2],
            1,
            qsa::KVH,
            qsa::HD,
            false,
            l,
            p,
        )?;
        self.norm(
            arena,
            public,
            arg(arena, ids[RAWIQ], qsa::INDEX_QUERY)?,
            ids[IQ],
            statuses[3],
            2,
            qsa::IH,
            qsa::ID,
            false,
            l,
            p,
        )?;
        cancelled(cancel)?;
        qsa::pack_token::launch::<WgpuRuntime>(
            client,
            grid(qsa::TOKEN),
            CubeDim::new_1d(64),
            arg(arena, ids[KEY], qsa::KV)?,
            arg(arena, ids[VALUE], qsa::KV)?,
            arg(arena, ids[RAWIK], qsa::ID)?,
            arg(arena, ids[TOKEN], qsa::TOKEN)?,
            arg(arena, statuses[4].buffer, qsa::TOKEN)?,
        );
        // Writes only the never-committed suffix; acceptance includes this status.
        append_candidate::launch::<WgpuRuntime>(
            client,
            grid(qsa::TOKEN + 1),
            CubeDim::new_1d(64),
            arg(arena, ids[TOKEN], qsa::TOKEN)?,
            arg(arena, ids[CACHE], self.layout.max_context * qsa::TOKEN)?,
            arg(arena, ids[POSITIONS], self.layout.max_context)?,
            arg(arena, statuses[11].buffer, qsa::TOKEN + 1)?,
            l.previous,
            p,
            self.layout.max_context,
        );
        if l.complete > 0 {
            cancelled(cancel)?;
            qsa::pool_indexer_raw::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(l.complete as u32, 1, 1),
                CubeDim::new_1d(64),
                self.old_arg(arena, l)?,
                arg(arena, ids[TOKEN], qsa::TOKEN)?,
                arg(arena, ids[MEANS], l.complete * qsa::ID)?,
                arg(arena, statuses[5].buffer, l.complete)?,
                l.previous,
                l.complete,
            );
            self.norm(
                arena,
                public,
                arg(arena, ids[MEANS], l.complete * qsa::ID)?,
                ids[POOLED],
                statuses[6],
                3,
                l.complete,
                qsa::ID,
                true,
                l,
                p,
            )?;
            qsa::score_blocks::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(l.complete as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(arena, ids[POOLED], l.complete * qsa::ID)?,
                arg(arena, ids[IQ], qsa::INDEX_QUERY)?,
                arg(arena, ids[SCORES], l.complete)?,
                arg(arena, statuses[7].buffer, l.complete)?,
                l.complete,
            );
        }
        cancelled(cancel)?;
        let selection = if l.complete == 0 {
            PrivateSelection::from_scores(l, &[], &[])?
        } else {
            selector.select_transient(
                arena,
                l,
                StatusRegion {
                    buffer: ids[SCORES],
                    words: l.complete,
                },
                statuses[7],
                cancel,
            )?
        };
        if selection.layout != l {
            return Err(fault("resident QSA stale selection"));
        }
        qsa::validate_selected(l, selection.ids()).map_err(fault)?;
        selector.upload_selected(arena, ids[SELECTED], &selection, cancel)?;
        drop(selection);
        cancelled(cancel)?;
        qsa::attention_logits::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.logits_elements as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(arena, ids[QUERY], qsa::QUERY)?,
            self.old_arg(arena, l)?,
            arg(arena, ids[TOKEN], qsa::TOKEN)?,
            arg(arena, ids[SELECTED], l.selected)?,
            arg(arena, ids[LOGITS], l.logits_elements)?,
            arg(arena, statuses[8].buffer, l.logits_elements)?,
            l.previous,
            l.selected,
        );
        cancelled(cancel)?;
        qsa::attention_softmax::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(qsa::QH as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(arena, ids[LOGITS], l.logits_elements)?,
            arg(arena, ids[PROBS], l.logits_elements)?,
            arg(arena, statuses[9].buffer, qsa::QH)?,
            l.selected,
        );
        cancelled(cancel)?;
        qsa::attention_values_gated::launch::<WgpuRuntime>(
            client,
            grid(qsa::QUERY),
            CubeDim::new_1d(64),
            self.old_arg(arena, l)?,
            arg(arena, ids[TOKEN], qsa::TOKEN)?,
            arg(arena, ids[PROBS], l.logits_elements)?,
            arg(arena, ids[SELECTED], l.selected)?,
            raw_arg(
                bounded_view(
                    arena.handle(ids[PACKED]).map_err(arena_fault)?,
                    qsa::QUERY,
                    qsa::QUERY,
                )?,
                qsa::QUERY,
            ),
            arg(arena, ids[GATED], qsa::QUERY)?,
            arg(arena, statuses[10].buffer, qsa::QUERY)?,
            l.previous,
            l.selected,
        );
        cancelled(cancel)?;
        projector.enqueue(self.projections.output, arena, ids[GATED], output, cancel)?;
        cancelled(cancel)?;
        Ok(PendingQsa {
            stage: self.identity,
            sequence: self.sequence,
            position,
            output,
            statuses,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        arena: &PrivateArena,
        public: &PublicQsaParameters,
        input: ArrayArg<WgpuRuntime>,
        output: BufferId,
        status: StatusRegion,
        gamma: usize,
        groups: usize,
        width: usize,
        block: bool,
        l: QsaLayout,
        p: u32,
    ) -> Result<()> {
        qsa::validate_norm_lengths(
            groups * width,
            width,
            qsa::ROPE_TABLE_WORDS,
            qsa::ROPE_TABLE_WORDS,
            self.layout.max_context,
            groups * width,
            status.words,
            groups,
            width,
            block,
            p,
            self.layout.epsilon,
        )
        .map_err(fault)?;
        let client = arena.client().map_err(arena_fault)?;
        qsa::rms_rope::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(groups as u32, 1, 1),
            CubeDim::new_1d(64),
            input,
            raw_arg(bounded_view(&public.vectors[gamma], 0, width)?, width),
            raw_arg(
                bounded_view(&public.vectors[4], 0, qsa::ROPE_TABLE_WORDS)?,
                qsa::ROPE_TABLE_WORDS,
            ),
            raw_arg(
                bounded_view(&public.vectors[5], 0, qsa::ROPE_TABLE_WORDS)?,
                qsa::ROPE_TABLE_WORDS,
            ),
            arg(arena, self.ids[POSITIONS], self.layout.max_context)?,
            arg(arena, output, groups * width)?,
            arg(arena, status.buffer, status.words)?,
            groups,
            p,
            self.layout.epsilon,
            width,
            true,
            block,
        );
        // l documents the visible prefix even though grouped RMS only accesses
        // first completed-block positions, already bounded by max capacity.
        if block && (groups != l.complete || (groups - 1) * 4 >= l.visible) {
            return Err(fault("resident QSA pool positions"));
        }
        Ok(())
    }
    fn old_arg(&self, arena: &PrivateArena, l: QsaLayout) -> Result<ArrayArg<WgpuRuntime>> {
        // WGSL binds one valid word on the first token but logical Array length
        // is0; cache helper cannot load old until token<previous, hence never.
        Ok(raw_arg(
            bounded_view(
                arena.handle(self.ids[CACHE]).map_err(arena_fault)?,
                0,
                l.old_elements.max(1),
            )?,
            l.old_elements,
        ))
    }
    fn validate_ids(&self, arena: &PrivateArena, input: BufferId, output: BufferId) -> Result<()> {
        if input == output
            || arena.layout(input).map_err(arena_fault)?.logical_words != self.layout.input_words
            || arena.layout(output).map_err(arena_fault)?.logical_words != self.layout.input_words
        {
            return Err(fault("resident QSA input/output geometry or alias"));
        }
        for (id, n) in self.ids.iter().zip(self.layout.allocation_words()) {
            if *id == input
                || *id == output
                || arena.layout(*id).map_err(arena_fault)?.logical_words != n
            {
                return Err(fault("resident QSA stale or aliased allocation"));
            }
        }
        Ok(())
    }

    fn validate_launch_lengths(
        &self,
        arena: &PrivateArena,
        l: QsaLayout,
        counts: [usize; 12],
    ) -> Result<()> {
        let ids = &self.ids;
        let words = |slot: usize, n: usize| prefix_words(arena, ids[slot], n);
        let old = words(CACHE, l.old_elements)?;
        let token = words(TOKEN, qsa::TOKEN)?;
        qsa::validate_projection_lengths(
            words(RAWQ, qsa::QUERY * 2)?,
            words(PACKED, qsa::QUERY * 2)?,
            words(STATUS, counts[0])?,
            words(KEY, qsa::KV)?,
            words(VALUE, qsa::KV)?,
            words(RAWIK, qsa::ID)?,
            token,
            words(STATUS + 4, counts[4])?,
        )
        .map_err(fault)?;
        qsa::validate_pool_score_lengths(
            l,
            old,
            token,
            words(MEANS, l.complete * qsa::ID)?,
            words(STATUS + 5, counts[5])?,
            words(POOLED, l.complete * qsa::ID)?,
            words(IQ, qsa::INDEX_QUERY)?,
            words(SCORES, l.complete)?,
            words(STATUS + 7, counts[7])?,
        )
        .map_err(fault)?;
        qsa::validate_attention_lengths(
            l,
            words(QUERY, qsa::QUERY)?,
            old,
            token,
            words(SELECTED, l.selected)?,
            words(LOGITS, l.logits_elements)?,
            words(STATUS + 8, counts[8])?,
            words(PROBS, l.logits_elements)?,
            words(STATUS + 9, counts[9])?,
            qsa::QUERY,
            words(GATED, qsa::QUERY)?,
            words(STATUS + 10, counts[10])?,
        )
        .map_err(fault)
    }
    /// OWNER precondition: successful whole-token sync, every12 stage status1,
    /// every projector status/completion, all other layers accepted, no cancel/
    /// context failure. Serialized owner only; late failure erases all layers.
    pub fn commit_after_whole_token_acceptance(
        &mut self,
        receipt: PendingQsa,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let result = (|| {
            cancelled(cancel)?;
            arena.client().map_err(arena_fault)?;
            if receipt.stage != self.identity || receipt.sequence != self.sequence {
                return Err(fault("resident QSA foreign receipt"));
            }
            for id in self.ids {
                arena.layout(id).map_err(arena_fault)?;
            }
            self.lifecycle.commit(receipt.position)
        })();
        if result.is_err() {
            self.poison();
        }
        result
    }
}
/// Candidate append/status is included in whole-token acceptance. Committed
/// prefix never changes, even if another token stage fails or cancellation wins.
#[cube(launch)]
// Portable CubeCL finite comparisons; host-only range/finite APIs cannot lower.
#[allow(clippy::eq_op, clippy::manual_range_contains)]
pub fn append_candidate(
    candidate: &Array<f32>,
    cache: &mut Array<f32>,
    positions: &mut Array<u32>,
    status: &mut Array<u32>,
    slot: usize,
    position: u32,
    capacity: usize,
) {
    let i = ABSOLUTE_POS;
    if i < qsa::TOKEN + 1 {
        status[i] = 0;
        if slot < capacity && position <= qsa::MAX_POSITION {
            if i < qsa::TOKEN {
                let x = candidate[i];
                cache[slot * qsa::TOKEN + i] = 0.0;
                if x == x && x <= f32::MAX && x >= -f32::MAX {
                    cache[slot * qsa::TOKEN + i] = x;
                    status[i] = 1;
                }
            } else {
                positions[slot] = position;
                status[i] = 1;
            }
        }
    }
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static(n.div_ceil(64) as u32, 1, 1)
}
fn raw_arg(h: Handle, n: usize) -> ArrayArg<WgpuRuntime> {
    unsafe { ArrayArg::from_raw_parts(h, n) }
}
fn prefix_words(arena: &PrivateArena, id: BufferId, n: usize) -> Result<usize> {
    if n > arena.layout(id).map_err(arena_fault)?.logical_words {
        return Err(Error::InvalidBufferLength);
    }
    Ok(n)
}
fn arg(arena: &PrivateArena, id: BufferId, n: usize) -> Result<ArrayArg<WgpuRuntime>> {
    prefix_words(arena, id, n)?;
    if n == 0 {
        return Err(Error::InvalidBufferLength);
    }
    Ok(raw_arg(
        bounded_view(arena.handle(id).map_err(arena_fault)?, 0, n)?,
        n,
    ))
}
fn bounded_view(h: &Handle, offset: usize, words: usize) -> Result<Handle> {
    let start = h.offset_start.unwrap_or(0);
    let end = h.offset_end.unwrap_or(0);
    let used = h
        .size()
        .checked_sub(start)
        .and_then(|n| n.checked_sub(end))
        .ok_or(Error::InvalidBufferLength)?;
    let relative = (offset as u64)
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let bytes = (words as u64)
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let last = relative
        .checked_add(bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let absolute = start
        .checked_add(relative)
        .ok_or(Error::ArithmeticOverflow)?;
    if words == 0
        || !h.size().is_multiple_of(4)
        || !start.is_multiple_of(4)
        || !end.is_multiple_of(4)
        || last > used
        || absolute % 256 != 0
    {
        return Err(fault("QSA storage bounds/alignment"));
    }
    let view = h.clone().offset_start(relative).offset_end(used - last);
    if view.size_in_used() != bytes {
        return Err(fault("QSA storage view mismatch"));
    }
    Ok(view)
}
fn validate_regions(
    arena: &PrivateArena,
    ids: &[BufferId; N],
    public: &PublicQsaParameters,
) -> Result<()> {
    let client = arena.client().map_err(arena_fault)?;
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(N + 6)
        .map_err(|_| Error::LimitExceeded("QSA region metadata"))?;
    for &id in ids {
        handles.push(arena.handle(id).map_err(arena_fault)?.clone());
    }
    handles.extend(public.vectors.iter().cloned());
    let stream = handles[0].stream;
    if handles.iter().any(|h| h.stream != stream) {
        return Err(fault("QSA cross-stream allocation"));
    }
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(handles.len())
        .map_err(|_| Error::LimitExceeded("QSA region metadata"))?;
    for h in handles {
        resources.push(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| client.get_resource(h)))
                .map_err(|_| fault("QSA resource lookup failed"))?
                .map_err(|_| fault("QSA resource lookup failed"))?,
        );
    }
    for (i, left) in resources.iter().enumerate().take(N) {
        let a = left.resource();
        let end = a
            .offset
            .checked_add(a.size)
            .ok_or(Error::ArithmeticOverflow)?;
        for right in resources.iter().skip(i + 1) {
            let b = right.resource();
            let bend = b
                .offset
                .checked_add(b.size)
                .ok_or(Error::ArithmeticOverflow)?;
            if a.buffer == b.buffer && a.offset < bend && b.offset < end {
                return Err(fault("QSA physically aliased allocation"));
            }
        }
    }
    Ok(())
}

/// Fixed public operator-chain probe. Caller alone initializes/runs a device
/// under its protected unit, with PrivateArena's required pool configuration.
/// This entry never reads model payloads or prompts. It tests admitted FULL QSA
/// heads at small/full contexts; there is no reduced-head QSA specialization.
/// Dense projections are supplied as fixed synthetic results here; the real
/// BoundGraph/ResidentProjector path is separately exposed by ResidentQsa.
#[derive(Clone, Copy, Debug)]
pub struct PublicProbeReport {
    pub visible: usize,
    pub complete_blocks: usize,
    pub selected_tokens: usize,
    pub max_norm_error: f32,
    pub max_score_error: f32,
    pub max_output_error: f32,
    pub charged_bytes: usize,
}
fn op<T>(r: std::result::Result<T, kuno_text::ops::OpsError>) -> Result<T> {
    r.map_err(|_| fault("public QSA oracle failed"))
}
fn raw_key(t: usize) -> Vec<f32> {
    (0..qsa::KV)
        .map(|d| (((d * 3 + t * 7) % 29) as f32 - 14.0) / 32.0)
        .collect()
}
fn raw_value(t: usize) -> Vec<f32> {
    (0..qsa::KV)
        .map(|d| (((d * 5 + t * 11) % 37) as f32 - 18.0) / 16.0)
        .collect()
}
fn raw_index_key(t: usize) -> Vec<f32> {
    let mut r = vec![0.0; qsa::ID];
    // Unique zero-score block0 ensures the513-block cutoff has a wide margin.
    // Other blocks have unequal raw member vectors; pool BEFORE RMS matters.
    if t >= 4 {
        r[0] = 0.5 + (t % 4) as f32 / 4.0;
        r[64] = ((t % 4) as f32 - 1.5) / 8.0;
    }
    r
}
fn normalized(mut x: Vec<f32>, gamma: &[f32], width: usize, p: u32) -> Result<Vec<f32>> {
    x = op(kuno_text::ops::rms_norm(&x, gamma, width, qsa::EPSILON))?;
    op(kuno_text::ops::rope_neox_heads(
        &mut x,
        width,
        64,
        p,
        qsa::THETA,
    ))?;
    Ok(x)
}
struct PublicFixture {
    l: QsaLayout,
    rawq: Vec<f32>,
    rawk: Vec<f32>,
    value: Vec<f32>,
    rawiq: Vec<f32>,
    rawik: Vec<f32>,
    cache: Vec<f32>,
    positions: Vec<u32>,
    gamma: [Vec<f32>; 4],
    query: Vec<f32>,
    key: Vec<f32>,
    iq: Vec<f32>,
    pooled: Vec<f32>,
    scores: Vec<f32>,
    selected: Vec<u32>,
    gated: Vec<f32>,
}
fn public_fixture(visible: usize) -> Result<PublicFixture> {
    let l = QsaLayout::new(visible).map_err(fault)?;
    let gamma = [
        (0..qsa::HD)
            .map(|i| 0.625 + (i % 7) as f32 / 16.0)
            .collect::<Vec<_>>(),
        (0..qsa::HD).map(|i| 0.75 + (i % 9) as f32 / 16.0).collect(),
        (0..qsa::ID).map(|i| 1.0 + (i % 3) as f32 / 8.0).collect(),
        (0..qsa::ID)
            .map(|i| 0.875 + (i % 5) as f32 / 16.0)
            .collect(),
    ];
    let rawq = (0..qsa::QUERY * 2)
        .map(|i| ((i % 37) as f32 - 18.0) / 16.0)
        .collect::<Vec<_>>();
    let query_raw = rawq
        .chunks_exact(qsa::HD * 2)
        .flat_map(|h| h[..qsa::HD].iter().copied())
        .collect::<Vec<_>>();
    let gate = rawq
        .chunks_exact(qsa::HD * 2)
        .flat_map(|h| h[qsa::HD..].iter().copied())
        .collect::<Vec<_>>();
    let mut rawiq = vec![0.0; qsa::INDEX_QUERY];
    rawiq[0] = 1.0;
    rawiq[qsa::ID] = -1.0;
    rawiq[qsa::ID * 2 + 32] = 1.0;
    rawiq[qsa::ID * 3 + 32] = -1.0;
    let p = l.previous as u32;
    let query = normalized(query_raw, &gamma[0], qsa::HD, p)?;
    let iq = normalized(rawiq.clone(), &gamma[2], qsa::ID, p)?;
    let rawk = raw_key(l.previous);
    let value = raw_value(l.previous);
    let rawik = raw_index_key(l.previous);
    let key = normalized(rawk.clone(), &gamma[1], qsa::HD, p)?;
    let mut cache = vec![0.0; visible * qsa::TOKEN];
    let mut allkeys = Vec::new();
    let mut allvalues = Vec::new();
    let mut rawik_all = Vec::new();
    for t in 0..visible {
        let k = normalized(raw_key(t), &gamma[1], qsa::HD, t as u32)?;
        let v = raw_value(t);
        let r = raw_index_key(t);
        if t < l.previous {
            cache[t * qsa::TOKEN..t * qsa::TOKEN + qsa::KV].copy_from_slice(&k);
            cache[t * qsa::TOKEN + qsa::KV..t * qsa::TOKEN + qsa::KV * 2].copy_from_slice(&v);
            cache[t * qsa::TOKEN + qsa::KV * 2..(t + 1) * qsa::TOKEN].copy_from_slice(&r);
        }
        allkeys.extend(k);
        allvalues.extend(v);
        rawik_all.extend(r);
    }
    let positions = (0..visible as u32).collect::<Vec<_>>();
    qsa::validate_positions(l, &positions).map_err(fault)?;
    let shape = kuno_text::ops::QsaShape::qwen38();
    let scores = op(kuno_text::ops::qsa_block_scores(
        &rawik_all, &iq, &gamma[3], &positions, visible, shape,
    ))?;
    let selected = qsa::select_blocks(l, &scores, &vec![1; l.complete]).map_err(fault)?;
    let selected_usize = selected.iter().map(|&i| i as usize).collect::<Vec<_>>();
    let mut gated = op(kuno_text::ops::qsa_attend(
        &query,
        &allkeys,
        &allvalues,
        &selected_usize,
        visible,
        shape,
    ))?;
    for (y, g) in gated.iter_mut().zip(&gate) {
        *y *= kuno_text::ops::sigmoid(*g);
    }
    let mut pooled = Vec::new();
    for b in 0..l.complete {
        let mean = (0..qsa::ID)
            .map(|d| {
                (0..4)
                    .map(|j| rawik_all[(b * 4 + j) * qsa::ID + d] as f64)
                    .sum::<f64>() as f32
                    / 4.0
            })
            .collect::<Vec<_>>();
        pooled.extend(normalized(mean, &gamma[3], qsa::ID, positions[b * 4])?);
    }
    Ok(PublicFixture {
        l,
        rawq,
        rawk,
        value,
        rawiq,
        rawik,
        cache,
        positions,
        gamma,
        query,
        key,
        iq,
        pooled,
        scores,
        selected,
        gated,
    })
}
fn public_words(arena: &PrivateArena, id: BufferId, n: usize) -> Result<Vec<u32>> {
    if n == 0 {
        return Ok(vec![]);
    }
    let h = bounded_view(arena.handle(id).map_err(arena_fault)?, 0, n)?;
    let mut bytes = arena
        .client()
        .map_err(arena_fault)?
        .read_one(h)
        .map_err(|_| fault("public QSA readback failed"))?;
    let result = if bytes.len() != n * 4 {
        Err(Error::InvalidBufferLength)
    } else {
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    for byte in bytes.iter_mut() {
        unsafe { std::ptr::write_volatile(byte, 0) }
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
    result
}
fn public_floats(arena: &PrivateArena, id: BufferId, n: usize) -> Result<Vec<f32>> {
    Ok(public_words(arena, id, n)?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}
fn public_parity(visible: usize, stage: &str, actual: &[f32], expected: &[f32]) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err(Error::InvalidBufferLength);
    }
    let mut error = 0.0_f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(fault("public QSA nonfinite result"));
        }
        error = error.max((a - b).abs());
        if (a - b).abs() > 3e-5 + 1e-4 * b.abs() {
            eprintln!(
                "public_qsa_parity_failure visible={visible} stage={stage} index={index} actual={a} expected={b} absolute_error={} tolerance={}",
                (a - b).abs(),
                3e-5 + 1e-4 * b.abs()
            );
            return Err(fault("public QSA numerical parity"));
        }
    }
    Ok(error)
}
pub fn public_parity_probe(client: &ComputeClient<WgpuRuntime>) -> Result<Vec<PublicProbeReport>> {
    let retired = std::sync::Arc::new(AtomicBool::new(false));
    let mut reports = Vec::new();
    for visible in [1, 4, 5, 2048, 2049, 2052] {
        let f = public_fixture(visible)?;
        let l = f.l;
        let layout = StageLayout {
            input_words: 2560,
            max_context: visible,
            epsilon: qsa::EPSILON,
        };
        let mut arena = PrivateArena::new(client.clone(), 64 * 1024 * 1024, retired.clone())
            .map_err(arena_fault)?;
        let words = layout.allocation_words();
        let mut slots = Vec::new();
        for (i, n) in words.into_iter().enumerate() {
            let id = match i {
                CACHE => arena.upload_f32(&f.cache),
                POSITIONS => {
                    let mut uncommitted = f.positions.clone();
                    uncommitted[l.previous] = qsa::MAX_POSITION;
                    arena.upload_u32(&uncommitted)
                }
                RAWQ => arena.upload_f32(&f.rawq),
                RAWK => arena.upload_f32(&f.rawk),
                VALUE => arena.upload_f32(&f.value),
                RAWIQ => arena.upload_f32(&f.rawiq),
                RAWIK => arena.upload_f32(&f.rawik),
                _ => arena.allocate_zeroed(n),
            }
            .map_err(arena_fault)?;
            slots.push(id);
        }
        let ids: [BufferId; N] = slots
            .try_into()
            .map_err(|_| fault("public QSA slot count"))?;
        let mut gamma_ids = Vec::new();
        for g in &f.gamma {
            gamma_ids.push(arena.upload_f32(g).map_err(arena_fault)?);
        }
        let (public_sin, public_cos) = qsa::rope_tables();
        let sin_table = arena.upload_f32(public_sin).map_err(arena_fault)?;
        let cos_table = arena.upload_f32(public_cos).map_err(arena_fault)?;
        let counts = [
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            l.complete,
            l.complete,
            l.complete,
            l.logits_elements,
            qsa::QH,
            qsa::QUERY,
            qsa::TOKEN + 1,
        ];
        let old = |arena: &PrivateArena| -> Result<ArrayArg<WgpuRuntime>> {
            Ok(raw_arg(
                bounded_view(
                    arena.handle(ids[CACHE]).map_err(arena_fault)?,
                    0,
                    l.old_elements.max(1),
                )?,
                l.old_elements,
            ))
        };
        qsa::split_query_gate::launch::<WgpuRuntime>(
            client,
            grid(qsa::QUERY * 2),
            CubeDim::new_1d(64),
            arg(&arena, ids[RAWQ], qsa::QUERY * 2)?,
            arg(&arena, ids[PACKED], qsa::QUERY * 2)?,
            arg(&arena, ids[STATUS], counts[0])?,
        );
        let norm = |input: ArrayArg<WgpuRuntime>,
                    target: BufferId,
                    s: usize,
                    g: usize,
                    groups: usize,
                    width: usize,
                    block: bool|
         -> Result<()> {
            qsa::rms_rope::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(groups as u32, 1, 1),
                CubeDim::new_1d(64),
                input,
                arg(&arena, gamma_ids[g], width)?,
                arg(&arena, sin_table, qsa::ROPE_TABLE_WORDS)?,
                arg(&arena, cos_table, qsa::ROPE_TABLE_WORDS)?,
                arg(&arena, ids[POSITIONS], visible)?,
                arg(&arena, target, groups * width)?,
                arg(&arena, ids[STATUS + s], groups)?,
                groups,
                l.previous as u32,
                qsa::EPSILON,
                width,
                true,
                block,
            );
            Ok(())
        };
        norm(
            raw_arg(
                bounded_view(
                    arena.handle(ids[PACKED]).map_err(arena_fault)?,
                    0,
                    qsa::QUERY,
                )?,
                qsa::QUERY,
            ),
            ids[QUERY],
            1,
            0,
            qsa::QH,
            qsa::HD,
            false,
        )?;
        norm(
            arg(&arena, ids[RAWK], qsa::KV)?,
            ids[KEY],
            2,
            1,
            qsa::KVH,
            qsa::HD,
            false,
        )?;
        norm(
            arg(&arena, ids[RAWIQ], qsa::INDEX_QUERY)?,
            ids[IQ],
            3,
            2,
            qsa::IH,
            qsa::ID,
            false,
        )?;
        qsa::pack_token::launch::<WgpuRuntime>(
            client,
            grid(qsa::TOKEN),
            CubeDim::new_1d(64),
            arg(&arena, ids[KEY], qsa::KV)?,
            arg(&arena, ids[VALUE], qsa::KV)?,
            arg(&arena, ids[RAWIK], qsa::ID)?,
            arg(&arena, ids[TOKEN], qsa::TOKEN)?,
            arg(&arena, ids[STATUS + 4], qsa::TOKEN)?,
        );
        append_candidate::launch::<WgpuRuntime>(
            client,
            grid(qsa::TOKEN + 1),
            CubeDim::new_1d(64),
            arg(&arena, ids[TOKEN], qsa::TOKEN)?,
            arg(&arena, ids[CACHE], visible * qsa::TOKEN)?,
            arg(&arena, ids[POSITIONS], visible)?,
            arg(&arena, ids[STATUS + 11], qsa::TOKEN + 1)?,
            l.previous,
            l.previous as u32,
            visible,
        );
        if l.complete > 0 {
            qsa::pool_indexer_raw::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(l.complete as u32, 1, 1),
                CubeDim::new_1d(64),
                old(&arena)?,
                arg(&arena, ids[TOKEN], qsa::TOKEN)?,
                arg(&arena, ids[MEANS], l.complete * qsa::ID)?,
                arg(&arena, ids[STATUS + 5], l.complete)?,
                l.previous,
                l.complete,
            );
            norm(
                arg(&arena, ids[MEANS], l.complete * qsa::ID)?,
                ids[POOLED],
                6,
                3,
                l.complete,
                qsa::ID,
                true,
            )?;
            qsa::score_blocks::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(l.complete as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(&arena, ids[POOLED], l.complete * qsa::ID)?,
                arg(&arena, ids[IQ], qsa::INDEX_QUERY)?,
                arg(&arena, ids[SCORES], l.complete)?,
                arg(&arena, ids[STATUS + 7], l.complete)?,
                l.complete,
            );
        }
        let actual_scores = public_floats(&arena, ids[SCORES], l.complete)?;
        let score_status = public_words(&arena, ids[STATUS + 7], l.complete)?;
        let selection = PrivateSelection::from_scores(l, &actual_scores, &score_status)?;
        if selection.ids() != f.selected {
            return Err(fault("public QSA selected blocks differ"));
        }
        // Probe-only one bounded upload/fixture, not the resident owner's
        // preallocated-update callback. Arena erases the upload/full padding.
        let selected = arena.upload_u32(selection.ids()).map_err(arena_fault)?;
        qsa::attention_logits::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.logits_elements as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(&arena, ids[QUERY], qsa::QUERY)?,
            old(&arena)?,
            arg(&arena, ids[TOKEN], qsa::TOKEN)?,
            arg(&arena, selected, l.selected)?,
            arg(&arena, ids[LOGITS], l.logits_elements)?,
            arg(&arena, ids[STATUS + 8], l.logits_elements)?,
            l.previous,
            l.selected,
        );
        qsa::attention_softmax::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(qsa::QH as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(&arena, ids[LOGITS], l.logits_elements)?,
            arg(&arena, ids[PROBS], l.logits_elements)?,
            arg(&arena, ids[STATUS + 9], qsa::QH)?,
            l.selected,
        );
        qsa::attention_values_gated::launch::<WgpuRuntime>(
            client,
            grid(qsa::QUERY),
            CubeDim::new_1d(64),
            old(&arena)?,
            arg(&arena, ids[TOKEN], qsa::TOKEN)?,
            arg(&arena, ids[PROBS], l.logits_elements)?,
            arg(&arena, selected, l.selected)?,
            raw_arg(
                bounded_view(
                    arena.handle(ids[PACKED]).map_err(arena_fault)?,
                    qsa::QUERY,
                    qsa::QUERY,
                )?,
                qsa::QUERY,
            ),
            arg(&arena, ids[GATED], qsa::QUERY)?,
            arg(&arena, ids[STATUS + 10], qsa::QUERY)?,
            l.previous,
            l.selected,
        );
        // Final status reads fence device work; no prior stage success inferred.
        let stages = (0..12)
            .map(|j| public_words(&arena, ids[STATUS + j], counts[j]))
            .collect::<Result<Vec<_>>>()?;
        validate_token_status(l, std::array::from_fn(|j| stages[j].as_slice()))?;
        let max_norm_error = public_parity(
            visible,
            "query",
            &public_floats(&arena, ids[QUERY], qsa::QUERY)?,
            &f.query,
        )?
        .max(public_parity(
            visible,
            "key",
            &public_floats(&arena, ids[KEY], qsa::KV)?,
            &f.key,
        )?)
        .max(public_parity(
            visible,
            "iq",
            &public_floats(&arena, ids[IQ], qsa::INDEX_QUERY)?,
            &f.iq,
        )?)
        .max(public_parity(
            visible,
            "pooled",
            &public_floats(&arena, ids[POOLED], l.complete * qsa::ID)?,
            &f.pooled,
        )?);
        let max_score_error = public_parity(visible, "scores", &actual_scores, &f.scores)?;
        let max_output_error = public_parity(
            visible,
            "gated",
            &public_floats(&arena, ids[GATED], qsa::QUERY)?,
            &f.gated,
        )?;
        let history = public_floats(&arena, ids[CACHE], visible * qsa::TOKEN)?;
        if history[..l.old_elements] != f.cache[..l.old_elements]
            || public_words(&arena, ids[POSITIONS], visible)? != f.positions
        {
            return Err(fault("public QSA committed prefix changed"));
        }
        let mut expected = f.key.clone();
        expected.extend_from_slice(&f.value);
        expected.extend_from_slice(&f.rawik);
        public_parity(
            visible,
            "appended_history",
            &history[l.old_elements..],
            &expected,
        )?;
        reports.push(PublicProbeReport {
            visible,
            complete_blocks: l.complete,
            selected_tokens: l.selected,
            max_norm_error,
            max_score_error,
            max_output_error,
            charged_bytes: arena.charged_bytes(),
        });
        drop(selection);
        arena.close().map_err(arena_fault)?;
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry() -> GraphGeometry {
        GraphGeometry {
            layers: 48,
            residual: kuno_text::ops::GatedResidualShape {
                streams: 4,
                width: 2560,
            },
            low_rank: 320,
            vocabulary: 248320,
            experts: 512,
            experts_used: 10,
            expert_hidden: 640,
            epsilon: 1e-6,
            max_context: 4096,
            generation_eos: 248046,
            gdn: kuno_text::ops::GdnShape::qwen38(),
            gdn_conv_channels: 10240,
            gdn_conv_kernel: 4,
            gdn_conv_dilation: 1,
            qsa: kuno_text::ops::QsaShape::qwen38(),
            ple: kuno_text::model::PleGeometry {
                layer: 1,
                head_width: 160,
                heads: 16,
                table_rows: 320001536,
                predecessor_tokens: 2,
                boundary_token: 248044,
                conv_channels: 10240,
                conv_kernel: 4,
                conv_dilation: 3,
            },
        }
    }
    #[test]
    fn full_plan_is_bounded_and_public_gamma_exact() {
        let l = StageLayout::from_geometry(geometry()).unwrap();
        let w = l.allocation_words();
        assert_eq!(w[CACHE], 4096 * qsa::TOKEN);
        assert_eq!(w[POSITIONS], 4096);
        assert_eq!(w[SELECTED], 2051); // earlier incomplete tails need2051
        assert!(l.owned_bytes().unwrap() < 40 * 1024 * 1024);
        for max in [1, 3, 4, 5, 2049, 2051, 4095] {
            let mut g = geometry();
            g.max_context = max;
            let l = StageLayout::from_geometry(g).unwrap();
            assert!(l.allocation_words().iter().all(|&n| n > 0));
        }
        let mut g = geometry();
        g.qsa.rotary_dims = 128;
        assert!(StageLayout::from_geometry(g).is_err());
        let f = public_fixture(5).unwrap();
        validate_public_vectors(std::array::from_fn(|i| f.gamma[i].as_slice())).unwrap();
        let mut bad = f.gamma.clone();
        bad[0][3] = f32::NAN;
        assert!(validate_public_vectors(std::array::from_fn(|i| bad[i].as_slice())).is_err());
    }
    #[test]
    fn phase_rejects_stale_commit_retry_and_context_exhaustion() {
        let mut life = Lifecycle::new(2);
        assert_eq!(life.begin(0).unwrap().visible, 1);
        life.commit(0).unwrap();
        assert_eq!(life.begin(1).unwrap().visible, 2);
        life.commit(1).unwrap();
        assert!(life.begin(2).is_err());
        let mut life = Lifecycle::new(2);
        life.begin(0).unwrap();
        assert!(life.begin(0).is_err());
        assert!(life.commit(0).is_err());
        let mut life = Lifecycle::new(2);
        life.begin(0).unwrap();
        assert!(life.commit(1).is_err());
        assert!(life.begin(0).is_err());
        let cancel = AtomicBool::new(true);
        assert!(cancelled(&cancel).is_err());
        let l = QsaLayout::new(1).unwrap();
        let n = [
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            0,
            0,
            0,
            l.logits_elements,
            qsa::QH,
            qsa::QUERY,
            qsa::TOKEN + 1,
        ];
        let mut stages = n.map(|n| vec![1; n]);
        validate_token_status(l, std::array::from_fn(|i| stages[i].as_slice())).unwrap();
        stages[11][0] = 0;
        assert!(validate_token_status(l, std::array::from_fn(|i| stages[i].as_slice())).is_err());
    }
    #[test]
    fn fixed_probe_cutoff_has_unique_zero_block_and_exact_tails() {
        for visible in [1, 4, 5, 2048, 2049, 2052] {
            let f = public_fixture(visible).unwrap();
            let l = f.l;
            assert_eq!(f.selected.len(), l.selected);
            assert!(f.gated.iter().all(|x| x.is_finite()));
            if l.complete > 0 {
                assert_eq!(f.scores[0], 0.0);
                assert!(f.scores[1..].iter().all(|&s| s > 1.0));
            }
            if visible == 2052 {
                assert_eq!(f.selected, (4..2052).collect::<Vec<_>>());
            }
            if visible == 2049 {
                assert_eq!(f.selected, (0..2049).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn append_generates_runtime_bounds_status_wgsl() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(uint);
            b.scope.register_type::<usize>(uint);
            b.scope.register_type::<f32>(float);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let x = b.input_array(Type::new(float)).into();
            let c = b.output_array(Type::new(float)).into();
            let p = b.output_array(Type::new(uint)).into();
            let s = b.output_array(Type::new(uint)).into();
            let slot = b.scalar(uint).into();
            let pos = b.scalar(uint).into();
            let cap = b.scalar(uint).into();
            append_candidate::expand(&mut b.scope, x, c, p, s, slot, pos, cap);
            let wgsl = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("qsa_append"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    uint,
                )
                .unwrap()
                .to_string();
            assert!(wgsl.contains("info.scalars_u32[2]") && wgsl.contains("u32(1)"));
            if std::env::var_os("KUNO_QSA_DUMP_STATIC_WGSL").is_some() {
                std::fs::write(format!("/tmp/kuno-qsa-append-{mode:?}.wgsl"), wgsl).unwrap();
            }
        }
    }
}
