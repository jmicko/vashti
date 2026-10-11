//! Unconnected native MAX8 adapter into actual ResidentQsa cache owners.
//! Uses original cache-backed projections/public shared gammas/Rust RoPE.
//! Original CPU selection uses a guarded chunk dependency fence. An explicit
//! public-target alternative keeps exact all-retained selection on the GPU;
//! no diagnostic/acceptance readback or state commit occurs in enqueue.
//! Cursor acceptance below requires a FUTURE whole-model completed fault,
//! dropped leases, checked cleanup/admission, health and cancellation proof.
#![allow(dead_code)] // Whole-model coordinator and physical-device acceptance pending.
use super::*;
use crate::{
    executor::ResidentAdmission,
    qsa_chunk as c,
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_util as u,
};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};
#[path = "qsa_chronological_probe.rs"]
pub(crate) mod chronological_probe;
#[path = "qsa_chronological_select.rs"]
pub(crate) mod chronological_select;
#[allow(unused_imports)] // Parent constructor consumes these fixed reservations.
pub(crate) use chronological_select::{EXTRA_HOST_BYTES_PER_CHUNK, EXTRA_INFO_BYTES_PER_CHUNK};
const S: usize = 29;
const MAX_LAYERS: usize = 64;
const RQ: usize = 0;
const RK: usize = 1;
const RV: usize = 2;
const RIQ: usize = 3;
const RIK: usize = 4;
const PACK: usize = 5;
const Q: usize = 6;
const K: usize = 7;
const IQ: usize = 8;
const CAND: usize = 9;
const MEAN: usize = 10;
const POOL: usize = 11;
const SCORE: usize = 12;
const SELECT: usize = 13;
const LOGIT: usize = 14;
const PROB: usize = 15;
const OUT: usize = 16;
const FLAGS: usize = 17;
const APPEND: usize = 28;
const IO_DISPATCH_BOUND: u64 = 128;
const INFO_BYTES: u64 = 512;
#[derive(Clone, Copy, PartialEq)]
struct Layout {
    stage: StageLayout,
    widths: [usize; S],
    words: [usize; S],
}
impl Layout {
    fn new(stage: StageLayout) -> Result<Self> {
        if !(4..=qsa::MAX_CONTEXT).contains(&stage.max_context) || stage.epsilon != qsa::EPSILON {
            return Err(fault("native QSA batch geometry"));
        }
        let blocks = stage.max_context / qsa::BLOCK;
        let selected = qsa::BUDGET + 3;
        let widths = [
            qsa::QUERY * 2,
            qsa::KV,
            qsa::KV,
            qsa::INDEX_QUERY,
            qsa::ID,
            qsa::QUERY * 2,
            qsa::QUERY,
            qsa::KV,
            qsa::INDEX_QUERY,
            qsa::TOKEN,
            blocks * qsa::ID,
            blocks * qsa::ID,
            blocks,
            selected,
            qsa::QH * selected,
            qsa::QH * selected,
            qsa::QUERY,
            qsa::QUERY * 2,
            qsa::QH,
            qsa::KVH,
            qsa::IH,
            qsa::TOKEN,
            blocks,
            blocks,
            blocks,
            qsa::QH * selected,
            qsa::QH,
            qsa::QUERY,
            qsa::TOKEN + 1,
        ];
        let mut words = [0; S];
        for (i, width) in widths.into_iter().enumerate() {
            words[i] = width
                .checked_mul(MAX_BATCH)
                .ok_or(Error::ArithmeticOverflow)?;
            validate_view(MAX_BATCH, 0, width, width, words[i])?;
            BufferLayout::new(words[i]).map_err(arena_fault)?;
        }
        Ok(Self {
            stage,
            widths,
            words,
        })
    }
    fn requests(self) -> Result<[u64; S]> {
        let mut out = [0; S];
        for (i, n) in self.words.into_iter().enumerate() {
            out[i] = BufferLayout::new(n).map_err(arena_fault)?.owned_bytes as u64;
        }
        Ok(out)
    }
    fn bytes(self) -> Result<u64> {
        self.requests()?.into_iter().try_fold(0u64, |s, n| {
            s.checked_add(n).ok_or(Error::ArithmeticOverflow)
        })
    }
    fn token(self, t: usize, start: usize) -> Result<QsaLayout> {
        QsaLayout::new(start.checked_add(t + 1).ok_or(Error::ArithmeticOverflow)?).map_err(fault)
    }
    fn io_bytes(self, audited: bool) -> Result<u64> {
        // Original main destination is already admitted; include the full owned
        // selected upload source and two complete owning score/status maps.
        let pages = self.requests()?;
        (IO_DISPATCH_BOUND * INFO_BYTES * 2)
            .checked_add(pages[SCORE])
            .and_then(|n| n.checked_add(pages[FLAGS + 7]))
            .and_then(|n| n.checked_add(pages[SELECT]))
            .and_then(|n| n.checked_mul(if audited { 2 } else { 1 }))
            .ok_or(Error::ArithmeticOverflow)
    }
}
fn expected_stages(
    layers: usize,
    mut kind: impl FnMut(usize) -> Result<AttentionKind>,
) -> Result<u64> {
    if layers == 0 || layers > MAX_LAYERS {
        return Err(Error::InvalidBufferLength);
    }
    let mut expected = 0;
    for l in 0..layers {
        if kind(l)? == AttentionKind::Qsa {
            expected |= 1u64 << l;
        }
    }
    if expected == 0 {
        return Err(fault("native QSA batch missing stages"));
    }
    Ok(expected)
}
fn graph_plan(graph: BoundGraph<'_, '_>) -> Result<(Layout, u64)> {
    graph.verify_unchanged()?;
    let expected = expected_stages(graph.geometry().layers, |l| graph.layer_kind(l))?;
    let mut layout = None;
    for l in 0..graph.geometry().layers {
        if expected & (1 << l) != 0 {
            let candidate = Layout::new(StageLayout::from_graph(graph, l)?)?;
            if layout.is_some_and(|old| old != candidate) {
                return Err(fault("native QSA heterogeneous geometry"));
            }
            layout = Some(candidate);
        }
    }
    Ok((
        layout.ok_or(fault("native QSA missing geometry"))?,
        expected,
    ))
}
fn guarded<T>(poison: &AtomicBool, op: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)) {
        Ok(Ok(v)) => Ok(v),
        _ => {
            poison.store(true, Ordering::Release);
            Err(fault("native QSA batch guarded failure"))
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    workspace: u64,
    chunk: u64,
    generation: u64,
    stage: u64,
    sequence: SequenceKey,
    public: u64,
    layer: usize,
    start: u64,
    count: usize,
    input_first: usize,
    input_stride: usize,
    output_first: usize,
    output_stride: usize,
}
impl Zeroize for Stamp {
    fn zeroize(&mut self) {
        self.workspace.zeroize();
        self.chunk.zeroize();
        self.generation.zeroize();
        self.stage.zeroize();
        self.sequence.zeroize();
        self.public.zeroize();
        self.layer.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.input_first.zeroize();
        self.input_stride.zeroize();
        self.output_first.zeroize();
        self.output_stride.zeroize();
    }
}
#[derive(Clone, PartialEq, Eq)]
struct Metadata {
    stamp: Stamp,
    scratch: [BufferId; S],
    owners: [BufferId; 2],
    input: BufferId,
    output: BufferId,
    fault: BufferId,
}
impl Zeroize for Metadata {
    fn zeroize(&mut self) {
        self.stamp.zeroize();
        self.scratch.zeroize();
        self.owners.zeroize();
        self.input.zeroize();
        self.output.zeroize();
        self.fault.zeroize();
    }
}
pub(super) struct StageAnchor(Zeroizing<Metadata>);
pub(crate) struct PendingQsaBatch {
    metadata: Zeroizing<Metadata>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for PendingQsaBatch {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChunkPhase {
    Idle,
    Queuing,
    Accepted,
}
struct Policy {
    phase: ChunkPhase,
    expected: u64,
    issued: Zeroizing<u64>,
    committed: Zeroizing<u64>,
    generations: Zeroizing<[u64; MAX_LAYERS]>,
    stage_ids: Zeroizing<[u64; MAX_LAYERS]>,
    chunk: Zeroizing<u64>,
    generation: Zeroizing<u64>,
    start: Zeroizing<u64>,
    count: Zeroizing<usize>,
}
impl Policy {
    fn new(expected: u64) -> Self {
        Self {
            phase: ChunkPhase::Idle,
            expected,
            issued: Zeroizing::new(0),
            committed: Zeroizing::new(0),
            generations: Zeroizing::new([0; MAX_LAYERS]),
            stage_ids: Zeroizing::new([0; MAX_LAYERS]),
            chunk: Zeroizing::new(0),
            generation: Zeroizing::new(0),
            start: Zeroizing::new(0),
            count: Zeroizing::new(0),
        }
    }
    fn issue(
        &mut self,
        layer: usize,
        stage: u64,
        start: u64,
        count: usize,
        context: usize,
    ) -> Result<(u64, u64)> {
        checked_count(start, count, context)?;
        if layer >= MAX_LAYERS
            || self.expected & (1u64 << layer) == 0
            || self.phase == ChunkPhase::Accepted
            || self.phase == ChunkPhase::Queuing && (*self.start != start || *self.count != count)
            || *self.issued & (1u64 << layer) != 0
            || layer != (self.expected & !*self.issued).trailing_zeros() as usize
            || (self.expected & !*self.issued).trailing_zeros() as usize != layer
        {
            return Err(fault("native QSA batch mixed or duplicate stage"));
        }
        if self.phase == ChunkPhase::Idle {
            *self.chunk = self.chunk.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
            *self.start = start;
            *self.count = count;
            self.phase = ChunkPhase::Queuing;
        }
        *self.generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.generations[layer] = *self.generation;
        self.stage_ids[layer] = stage;
        *self.issued |= 1u64 << layer;
        Ok((*self.chunk, *self.generation))
    }
    fn check(&self, stamp: Stamp) -> Result<()> {
        if stamp.layer >= MAX_LAYERS
            || self.phase == ChunkPhase::Idle
            || stamp.chunk != *self.chunk
            || stamp.start != *self.start
            || stamp.count != *self.count
            || stamp.generation != self.generations[stamp.layer]
            || stamp.stage != self.stage_ids[stamp.layer]
            || *self.issued & (1u64 << stamp.layer) == 0
        {
            return Err(fault("native QSA batch stale stage receipt"));
        }
        Ok(())
    }
    fn accept(&mut self, seen: u64) -> Result<()> {
        if self.phase != ChunkPhase::Queuing
            || seen != self.expected
            || *self.issued != self.expected
        {
            return Err(fault("native QSA batch incomplete preflight"));
        }
        self.phase = ChunkPhase::Accepted;
        Ok(())
    }
    fn commit(&mut self, stamp: Stamp) -> Result<()> {
        self.check(stamp)?;
        if self.phase != ChunkPhase::Accepted || *self.committed & (1u64 << stamp.layer) != 0 {
            return Err(fault("native QSA batch commit without acceptance"));
        }
        *self.committed |= 1u64 << stamp.layer;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if self.phase != ChunkPhase::Accepted || *self.committed != self.expected {
            return Err(fault("native QSA batch partial state commit"));
        }
        self.issued.zeroize();
        self.committed.zeroize();
        self.generations.zeroize();
        self.stage_ids.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.phase = ChunkPhase::Idle;
        Ok(())
    }
}
// Real arena retains each full GPU owner before this receives its opaque ID.
// Fixed guarded ID metadata never grows or releases those owners on failure.
fn allocate_ids(
    words: [usize; N],
    cancel: &AtomicBool,
    mut allocate: impl FnMut(usize) -> Result<BufferId>,
) -> Result<Zeroizing<[BufferId; N]>> {
    cancelled(cancel)?;
    let first = allocate(words[0])?;
    let mut ids = Zeroizing::new([first; N]);
    for i in 1..N {
        cancelled(cancel)?;
        ids[i] = allocate(words[i])?;
    }
    cancelled(cancel)?;
    Ok(ids)
}

fn checked_count(start: u64, count: usize, context: usize) -> Result<u64> {
    if !(1..=MAX_BATCH).contains(&count) {
        return Err(Error::InvalidBufferLength);
    }
    let next = start
        .checked_add(count as u64)
        .ok_or(Error::ArithmeticOverflow)?;
    if next > context as u64 {
        return Err(fault("native QSA batch context extent"));
    }
    Ok(next)
}

/// ONE model-owned scratch bank. Constructor requests join aggregate MAIN
/// admission before allocation; existing cache/POS owners are never duplicated.
pub(crate) struct QsaBatchWorkspace {
    binding: GraphBindingId,
    sequence: SequenceKey,
    identity: u64,
    layout: Layout,
    ids: Zeroizing<[BufferId; S]>,
    fault: Option<BufferId>,
    poison: Arc<AtomicBool>,
    policy: Policy,
    public_grouped_status: crate::status_fold_batch::PublicPolicy,
}
impl QsaBatchWorkspace {
    pub(crate) fn public_grouped_status_folds(&self) -> (bool, u64) {
        self.public_grouped_status.state()
    }
    pub(crate) fn validate_public_grouped_status_selection(
        &self,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.healthy(arena, cancel)?;
        if self.policy.phase != ChunkPhase::Idle
            || *self.policy.issued != 0
            || *self.policy.committed != 0
        {
            return Err(Error::InvalidFormat("target grouped QSA status boundary"));
        }
        self.public_grouped_status.validate_selection()
    }
    pub(crate) fn enable_public_grouped_status_folds(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) {
        self.public_grouped_status.enable();
    }

    pub(crate) fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; S]> {
        graph_plan(graph)?.0.requests()
    }
    /// Conservative component reservation: do not subtract prior queued work
    /// or use MAIN telemetry as runtime IO credit. Whole graph remains64MiB.
    pub(crate) fn runtime_io_reservation(graph: BoundGraph<'_, '_>, audit: bool) -> Result<u64> {
        let (l, mask) = graph_plan(graph)?;
        l.io_bytes(audit)?
            .checked_mul(mask.count_ones() as u64)
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(crate) fn allocate(
        graph: BoundGraph<'_, '_>,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        poison: Arc<AtomicBool>,
    ) -> Result<Self> {
        guarded(&poison, || {
            if poison.load(Ordering::Acquire) {
                return Err(fault("native QSA sequence poisoned"));
            }
            let (layout, expected) = graph_plan(graph)?;
            let total = (arena.charged_bytes() as u64)
                .checked_add(layout.bytes()?)
                .ok_or(Error::ArithmeticOverflow)?;
            if total > arena.budget_bytes() as u64 {
                return Err(Error::LimitExceeded("native QSA complete MAIN pages"));
            }
            let first = arena
                .allocate_zeroed(layout.words[0])
                .map_err(arena_fault)?;
            let mut ids = Zeroizing::new([first; S]);
            for i in 1..S {
                ids[i] = arena
                    .allocate_zeroed(layout.words[i])
                    .map_err(arena_fault)?;
            }
            let out = Self {
                binding: graph.binding_id(),
                sequence,
                identity: identity()?,
                layout,
                ids,
                fault: None,
                poison: poison.clone(),
                policy: Policy::new(expected),
                public_grouped_status: crate::status_fold_batch::PublicPolicy::default(),
            };
            out.validate_ids(arena)?;
            Ok(out)
        })
    }
    fn validate_ids(&self, arena: &PrivateArena) -> Result<()> {
        for (i, &id) in self.ids.iter().enumerate() {
            if self.ids[..i].contains(&id)
                || arena.layout(id).map_err(arena_fault)?
                    != BufferLayout::new(self.layout.words[i]).map_err(arena_fault)?
            {
                return Err(fault("native QSA stale workspace"));
            }
        }
        Ok(())
    }
    fn healthy(&self, arena: &PrivateArena, cancel: &AtomicBool) -> Result<()> {
        cancelled(cancel)?;
        if self.poison.load(Ordering::Acquire) {
            return Err(fault("native QSA sequence poisoned"));
        }
        arena.client().map_err(arena_fault)?;
        self.validate_ids(arena)
    }
    fn check_receipt(
        &self,
        stage: &ResidentQsa<'_>,
        receipt: &PendingQsaBatch,
        arena: &PrivateArena,
    ) -> Result<()> {
        let m = &*receipt.metadata;
        if !receipt.armed
            || !Arc::ptr_eq(&receipt.poison, &self.poison)
            || m.stamp.workspace != self.identity
            || m.stamp.sequence != self.sequence
            || m.stamp.stage != stage.identity
            || m.stamp.layer != stage.layer
            || m.stamp.public != stage.public_identity
            || stage.sequence != self.sequence
            || stage.binding != self.binding
            || stage.layout != self.layout.stage
            || m.scratch != *self.ids
            || Some(m.fault) != self.fault
            || m.owners != [stage.ids[CACHE], stage.ids[POSITIONS]]
            || stage
                .batch_pending
                .as_ref()
                .is_none_or(|anchor| *anchor.0 != *m)
            || stage.lifecycle.phase != Phase::PendingBatch(m.stamp.start, m.stamp.count)
            || stage
                .batch_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
        {
            return Err(fault("native QSA foreign chunk receipt"));
        }
        self.policy.check(m.stamp)?;
        checked_count(m.stamp.start, m.stamp.count, stage.layout.max_context)?;
        for id in m
            .scratch
            .into_iter()
            .chain(m.owners)
            .chain([m.input, m.output, m.fault])
        {
            arena.layout(id).map_err(arena_fault)?;
        }
        Ok(())
    }
    /// FUTURE WHOLE-MODEL OWNER PRECONDITION: every layer and final residual/head
    /// completed the original global fault fence, all map/future leases dropped,
    /// checked runtime cleanup/admission succeeded, device health/cancel accepted.
    /// Preflight every other family before ANY model state swap/cursor advance.
    /// This unconnected crate-private factory performs NO sync/read/audit and
    /// does not manufacture GPU acceptance. Expected QSA mask is graph-derived.
    pub(crate) fn accept_after_whole_model_chunk_acceptance<'a>(
        &'a mut self,
        arena: &PrivateArena,
        pending: &[(&ResidentQsa<'_>, &PendingQsaBatch)],
        cancel: &AtomicBool,
    ) -> Result<QsaBatchAcceptance<'a>> {
        let attempted = (|| {
            self.healthy(arena, cancel)?;
            let mut seen = Zeroizing::new(0u64);
            for &(stage, receipt) in pending {
                self.check_receipt(stage, receipt, arena)?;
                let bit = 1u64 << stage.layer;
                if *seen & bit != 0 {
                    return Err(fault("native QSA duplicate receipt preflight"));
                }
                *seen |= bit;
            }
            self.policy.accept(*seen)?;
            cancelled(cancel)
        })();
        if attempted.is_err() {
            self.poison.store(true, Ordering::Release);
        }
        attempted?;
        Ok(QsaBatchAcceptance {
            workspace: self,
            armed: true,
        })
    }
}
impl Drop for QsaBatchWorkspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
        self.sequence.zeroize();
        self.identity.zeroize();
        if let Some(id) = &mut self.fault {
            id.zeroize();
        }
    }
}
pub(crate) struct QsaBatchAcceptance<'a> {
    workspace: &'a mut QsaBatchWorkspace,
    armed: bool,
}
impl QsaBatchAcceptance<'_> {
    pub(crate) fn finish(mut self, arena: &PrivateArena, cancel: &AtomicBool) -> Result<()> {
        self.workspace.healthy(arena, cancel)?;
        self.workspace.policy.finish()?;
        if let Some(id) = &mut self.workspace.fault {
            id.zeroize();
        }
        self.workspace.fault = None;
        cancelled(cancel)?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for QsaBatchAcceptance<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.workspace.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}
impl Lifecycle {
    fn begin_batch(&mut self, start: u64, count: usize) -> Result<()> {
        if self.phase != Phase::Ready(start) || checked_count(start, count, self.max).is_err() {
            self.phase = Phase::Poisoned;
            return Err(fault("native QSA chunk cursor/order"));
        }
        self.phase = Phase::PendingBatch(start, count);
        Ok(())
    }
    fn commit_batch(&mut self, start: u64, count: usize) -> Result<()> {
        if self.phase != Phase::PendingBatch(start, count) {
            self.phase = Phase::Poisoned;
            return Err(fault("native QSA stale chunk cursor"));
        }
        self.phase = Phase::Ready(checked_count(start, count, self.max)?);
        Ok(())
    }
}
impl Drop for Lifecycle {
    fn drop(&mut self) {
        match &mut self.phase {
            Phase::Ready(p) | Phase::Pending(p) => p.zeroize(),
            Phase::PendingBatch(p, n) => {
                p.zeroize();
                n.zeroize();
            }
            Phase::Poisoned => {}
        }
    }
}
impl Drop for ResidentQsa<'_> {
    fn drop(&mut self) {
        if self.batch_pending.is_some() {
            self.poison();
        }
    }
}
pub(super) fn ensure_scalar(stage: &ResidentQsa<'_>) -> Result<()> {
    if stage.batch_pending.is_some()
        || stage
            .batch_poison
            .as_ref()
            .is_some_and(|p| p.load(Ordering::Acquire))
    {
        return Err(fault("native QSA batch unavailable for scalar"));
    }
    Ok(())
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Clear(usize),
    Positions,
    Project(usize),
    Split,
    Norm(usize),
    Pack,
    Pool,
    PoolNorm,
    Score,
    Selector,
    Logits,
    Softmax,
    Values,
    Output,
    Append,
    Fold(usize, usize),
}
fn queue_stage(
    count: usize,
    start: usize,
    cancel: &AtomicBool,
    mut queue: impl FnMut(Operation) -> Result<()>,
) -> Result<()> {
    checked_count(start as u64, count, qsa::MAX_CONTEXT)?;
    for op in (0..S)
        .map(Operation::Clear)
        .chain([Operation::Positions])
        .chain((0..5).map(Operation::Project))
        .chain([Operation::Split])
        .chain((0..3).map(Operation::Norm))
        .chain([
            Operation::Pack,
            Operation::Pool,
            Operation::PoolNorm,
            Operation::Score,
            Operation::Selector,
            Operation::Logits,
            Operation::Softmax,
            Operation::Values,
            Operation::Output,
            Operation::Append,
        ])
    {
        cancelled(cancel)?;
        queue(op)?;
    }
    for i in [0, 1, 2, 3, 4, 9, 10, 11] {
        cancelled(cancel)?;
        queue(Operation::Fold(i, MAX_BATCH))?;
    }
    for t in 0..count {
        for i in [5, 6, 7, 8] {
            if i == 8 || (start + t + 1) / 4 > 0 {
                cancelled(cancel)?;
                queue(Operation::Fold(i, t))?;
            }
        }
    }
    cancelled(cancel)
}

/// New append kernels preserve authoritative cache/POS prefix [0,start).
/// Positions equal the original contiguous scalar token index, passed runtime.
#[cube(launch)]
fn positions_batch(
    positions: &mut Array<u32>,
    status: &mut Array<u32>,
    start: usize,
    count: usize,
    capacity: usize,
    status_stride: usize,
) {
    let t = ABSOLUTE_POS_X as usize;
    if t < count {
        status[t * status_stride + qsa::TOKEN] = 0;
        let p = start + t;
        if p < capacity && p <= qsa::MAX_POSITION as usize {
            positions[p] = p as u32;
            status[t * status_stride + qsa::TOKEN] = 1;
        }
    }
}
#[cube(launch)]
#[allow(clippy::eq_op, clippy::manual_range_contains)]
fn append_batch(
    candidate: &Array<f32>,
    cache: &mut Array<f32>,
    positions: &Array<u32>,
    status: &mut Array<u32>,
    start: usize,
    count: usize,
    capacity: usize,
    candidate_stride: usize,
    status_stride: usize,
) {
    let t = CUBE_POS_Y as usize;
    if t < count {
        let i = ABSOLUTE_POS_X as usize;
        if i < qsa::TOKEN {
            status[t * status_stride + i] = 0;
            let p = start + t;
            if p < capacity {
                // All storage reads stay nested beneath active capacity checks.
                let position = positions[p];
                let x = candidate[t * candidate_stride + i];
                cache[p * qsa::TOKEN + i] = 0.0;
                if position == p as u32
                    && position <= qsa::MAX_POSITION
                    && x == x
                    && x <= f32::MAX
                    && x >= -f32::MAX
                {
                    cache[p * qsa::TOKEN + i] = x;
                    status[t * status_stride + i] = 1;
                }
            }
        }
    }
}

/// Exact original stable selection. CPU implementations retain their completed
/// owning maps; the admitted all-retained GPU implementation validates complete
/// sources before publication. Neither path constitutes state acceptance.
pub(crate) trait QsaBatchSelector {
    /// Real admission identity/owned-I/O proof before ANY candidate mutation.
    fn validate_owner(&self, arena: &PrivateArena, poison: &Arc<AtomicBool>) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn select_upload(
        &mut self,
        arena: &PrivateArena,
        layout: SelectionLayout,
        scores: BufferId,
        status: BufferId,
        target: BufferId,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()>;
}
#[derive(Clone, Copy)]
pub(crate) struct SelectionLayout {
    start: usize,
    count: usize,
    blocks: usize,
    stride: usize,
    words: usize,
}
impl SelectionLayout {
    fn new(l: Layout, start: usize, count: usize) -> Result<Self> {
        checked_count(start as u64, count, l.stage.max_context)?;
        Ok(Self {
            start,
            count,
            blocks: l.widths[SCORE],
            stride: l.widths[SELECT],
            words: l.words[SELECT],
        })
    }
    fn needs_read(self) -> bool {
        (self.start + self.count) / 4 != 0
    }
    fn select(
        self,
        scores: &[u32],
        flags: &[u32],
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<u32>>> {
        let map_words = BufferLayout::new(self.blocks * MAX_BATCH)
            .map_err(arena_fault)?
            .owned_words;
        if self.needs_read() && (scores.len() != map_words || flags.len() != map_words) {
            return Err(Error::InvalidBufferLength);
        }
        if !self.needs_read() && (!scores.is_empty() || !flags.is_empty()) {
            return Err(Error::InvalidBufferLength);
        }
        let mut out = Zeroizing::new(Vec::new());
        out.try_reserve_exact(self.words)
            .map_err(|_| Error::LimitExceeded("native QSA selection host"))?;
        out.resize(self.words, 0);
        for t in 0..self.count {
            cancelled(cancel)?;
            let l = QsaLayout::new(self.start + t + 1).map_err(fault)?;
            let mut values = Zeroizing::new(Vec::new());
            values
                .try_reserve_exact(l.complete)
                .map_err(|_| Error::LimitExceeded("native QSA scores host"))?;
            for &bits in scores
                .get(t * self.blocks..t * self.blocks + l.complete)
                .unwrap_or(&[])
            {
                values.push(f32::from_bits(bits));
            }
            let status = flags
                .get(t * self.blocks..t * self.blocks + l.complete)
                .unwrap_or(&[]);
            let ids = c::guarded_select(l, &values, status)?;
            out[t * self.stride..][..l.selected].copy_from_slice(&ids);
        }
        cancelled(cancel)?;
        Ok(out)
    }
}
struct Lease(cubecl::bytes::Bytes);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn decode_map(bytes: &[u8], l: BufferLayout) -> Result<Zeroizing<Vec<u32>>> {
    if bytes.len() != l.owned_bytes {
        return Err(Error::InvalidBufferLength);
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(l.owned_words)
        .map_err(|_| Error::LimitExceeded("native QSA owning map"))?;
    for b in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes(
            b.try_into().map_err(|_| Error::InvalidBufferLength)?,
        ));
    }
    Ok(out)
}
fn read_owned(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let l = arena.layout(id).map_err(arena_fault)?;
    let bytes = Lease(
        arena
            .client()
            .map_err(arena_fault)?
            .read_one(arena.handle(id).map_err(arena_fault)?.clone())
            .map_err(|_| fault("native QSA owning dependency map"))?,
    );
    decode_map(&bytes.0, l)
}
pub(crate) struct AdmittedBatchSelector<'a> {
    pub(crate) admission: &'a ResidentAdmission,
}
impl QsaBatchSelector for AdmittedBatchSelector<'_> {
    fn validate_owner(&self, arena: &PrivateArena, poison: &Arc<AtomicBool>) -> Result<()> {
        self.admission
            .check_client(arena.client().map_err(arena_fault)?)?;
        if !Arc::ptr_eq(&self.admission.sequence_poison(), poison) {
            return Err(fault("native QSA foreign model poison"));
        }
        if !self.admission.transient_io() {
            return Err(fault("native QSA requires owned runtime IO"));
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn select_upload(
        &mut self,
        arena: &PrivateArena,
        layout: SelectionLayout,
        scores: BufferId,
        status: BufferId,
        target: BufferId,
        _fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        cancelled(cancel)?;
        // No ordinary write fallback: disabled/unconfigured owned IO fails.
        self.validate_owner(arena, &self.admission.sequence_poison())?;
        let ids = if layout.needs_read() {
            self.admission.readback_intermediate(|| {
                let values = read_owned(arena, scores)?;
                let flags = read_owned(arena, status)?;
                // Both Bytes leases have dropped before this closure returns.
                layout.select(&values, &flags, cancel)
            })?
        } else {
            layout.select(&[], &[], cancel)?
        };
        let l = arena.layout(target).map_err(arena_fault)?;
        if l != BufferLayout::new(layout.words).map_err(arena_fault)? {
            return Err(Error::InvalidBufferLength);
        }
        let mut upload = Zeroizing::new(Vec::new());
        upload
            .try_reserve_exact(l.owned_bytes)
            .map_err(|_| Error::LimitExceeded("native QSA owning upload"))?;
        for &id in ids.iter() {
            upload.extend_from_slice(&id.to_le_bytes());
        }
        upload.resize(l.owned_bytes, 0);
        cancelled(cancel)?;
        // Destination already MAIN-admitted. Source is tracked mapped COPY_SRC
        // under native runtime IO's hard64MiB; no MAIN refund/allocation here.
        arena
            .client()
            .map_err(arena_fault)?
            .write_private(
                arena.handle(target).map_err(arena_fault)?.clone(),
                std::mem::take(&mut *upload),
            )
            .map_err(|_| fault("native QSA owned selection upload"))?;
        cancelled(cancel)
    }
}

/// Construction-fixed public-target selection. The ordinary selector above
/// remains unchanged; contexts outside the conservative original all-retained
/// range execute that same completed CPU read/select/upload path.
pub(crate) struct BatchSelector<'a> {
    original: AdmittedBatchSelector<'a>,
    gpu_chronological: bool,
}
impl<'a> BatchSelector<'a> {
    pub(crate) fn new(admission: &'a ResidentAdmission, gpu_chronological: bool) -> Self {
        Self {
            original: AdmittedBatchSelector { admission },
            gpu_chronological,
        }
    }
}
impl QsaBatchSelector for BatchSelector<'_> {
    fn validate_owner(&self, arena: &PrivateArena, poison: &Arc<AtomicBool>) -> Result<()> {
        self.original.validate_owner(arena, poison)
    }
    #[allow(clippy::too_many_arguments)]
    fn select_upload(
        &mut self,
        arena: &PrivateArena,
        layout: SelectionLayout,
        scores: BufferId,
        status: BufferId,
        target: BufferId,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let geometry = chronological_select::Geometry {
            start: layout.start,
            count: layout.count,
            blocks: layout.blocks,
        };
        if !self.gpu_chronological || !geometry.supported()? {
            return self
                .original
                .select_upload(arena, layout, scores, status, target, fault_id, cancel);
        }
        cancelled(cancel)?;
        self.original
            .validate_owner(arena, &self.original.admission.sequence_poison())?;
        if layout.stride != chronological_select::SELECT_STRIDE
            || layout.words != MAX_BATCH * layout.stride
            || arena.layout(fault_id).map_err(arena_fault)?
                != BufferLayout::new(1).map_err(arena_fault)?
        {
            return Err(Error::InvalidBufferLength);
        }
        let ids = [scores, status, target, fault_id];
        let logical = [
            MAX_BATCH * layout.blocks,
            MAX_BATCH * layout.blocks,
            layout.words,
            1,
        ];
        for (index, (&id, words)) in ids.iter().zip(logical).enumerate() {
            if ids[..index].contains(&id)
                || arena.layout(id).map_err(arena_fault)?
                    != BufferLayout::new(words).map_err(arena_fault)?
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        let owners = ids
            .map(|id| arena.handle(id).map_err(arena_fault).cloned())
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        chronological_select::launch(
            arena.client().map_err(arena_fault)?,
            owners.try_into().map_err(|_| Error::InvalidBufferLength)?,
            geometry,
        )?;
        cancelled(cancel)
    }
}

fn project_input_spans(parent: &BatchSpans, output_width: usize) -> BatchSpans {
    BatchSpans::new(
        parent.count(),
        0,
        parent.input_first(),
        parent.input_stride(),
        0,
        output_width,
    )
}

fn full_public(h: &Handle, words: usize) -> Result<Handle> {
    let page = BufferLayout::new(words).map_err(arena_fault)?.owned_bytes as u64;
    if h.size() != page || h.size_in_used() != words as u64 * 4 || h.offset_start.unwrap_or(0) != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut full = h.clone();
    full.offset_end = None;
    if full.size_in_used() != page {
        return Err(Error::InvalidBufferLength);
    }
    Ok(full)
}
fn validate_page(offset: u64, size: u64, physical: u64, owned: u64) -> Result<()> {
    if offset != 0 || size != owned || physical != owned || owned == 0 {
        return Err(fault("native QSA full page extent"));
    }
    Ok(())
}

/// All immutable scalar owner pages remain pinned too, preventing overlap with
/// retained scalar scratch. Public handles are logical views of full pages.
fn validate_full_regions(
    arena: &PrivateArena,
    workspace: &QsaBatchWorkspace,
    stage: &ResidentQsa<'_>,
    public: &PublicQsaParameters,
    input: BufferId,
    output: BufferId,
    fault_id: BufferId,
) -> Result<()> {
    const PRIVATE: usize = S + super::N + 3;
    const TOTAL: usize = PRIVATE + 6;
    let mut ids = Zeroizing::new([input; PRIVATE]);
    ids[..S].copy_from_slice(&*workspace.ids);
    ids[S..S + super::N].copy_from_slice(&stage.ids);
    ids[S + super::N..].copy_from_slice(&[input, output, fault_id]);
    let mut resources: [Option<_>; TOTAL] = std::array::from_fn(|_| None);
    let mut full_handles = Vec::with_capacity(TOTAL);
    let public_words = [
        qsa::HD,
        qsa::HD,
        qsa::ID,
        qsa::ID,
        qsa::ROPE_TABLE_WORDS,
        qsa::ROPE_TABLE_WORDS,
    ];
    let client = arena.client().map_err(arena_fault)?;
    let stream = arena.handle(ids[0]).map_err(arena_fault)?.stream;
    let mut bytes = Zeroizing::new([0u64; TOTAL]);
    for i in 0..TOTAL {
        let h = if i < PRIVATE {
            if ids[..i].contains(&ids[i]) {
                return Err(fault("native QSA owning alias"));
            }
            let l = arena.layout(ids[i]).map_err(arena_fault)?;
            let expected = if i < S {
                workspace.layout.words[i]
            } else if i < S + super::N {
                stage.layout.allocation_words()[i - S]
            } else {
                l.logical_words
            };
            if l != BufferLayout::new(expected).map_err(arena_fault)? {
                return Err(Error::InvalidBufferLength);
            }
            let h = arena.handle(ids[i]).map_err(arena_fault)?.clone();
            if h.size() != l.owned_bytes as u64
                || h.size_in_used() != h.size()
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
            {
                return Err(Error::InvalidBufferLength);
            }
            h
        } else {
            full_public(&public.vectors[i - PRIVATE], public_words[i - PRIVATE])?
        };
        if h.stream != stream {
            return Err(fault("native QSA foreign stream"));
        }
        bytes[i] = h.size();
        full_handles.push(h);
    }
    let metadata = crate::resident_util::resources(client, full_handles)?;
    if metadata.len() != resources.len() {
        return Err(Error::InvalidBufferLength);
    }
    for (slot, value) in resources.iter_mut().zip(metadata) {
        *slot = Some(value);
    }
    for i in 0..TOTAL {
        let page = resources[i]
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .resource();
        validate_page(page.offset, page.size, page.buffer.size(), bytes[i])?;
        if resources[..i]
            .iter()
            .flatten()
            .any(|r| r.resource().buffer == page.buffer)
        {
            return Err(fault("native QSA full physical backing/alias"));
        }
    }
    Ok(())
}
impl ResidentQsa<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_batch(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicQsaParameters,
        projector: &mut impl BatchedResidentProjector,
        selector: &mut impl QsaBatchSelector,
        workspace: &mut QsaBatchWorkspace,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: u64,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<PendingQsaBatch> {
        let attempted = guarded(&workspace.poison.clone(), || {
            self.enqueue_batch_checked(
                graph, arena, public, projector, selector, workspace, input, output, spans, start,
                fault_id, cancel,
            )
        });
        if attempted.is_err() {
            self.poison();
        }
        attempted
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicQsaParameters,
        projector: &mut impl BatchedResidentProjector,
        selector: &mut impl QsaBatchSelector,
        workspace: &mut QsaBatchWorkspace,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: u64,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<PendingQsaBatch> {
        workspace.healthy(arena, cancel)?;
        graph.verify_unchanged()?;
        let count = spans.count();
        let l = workspace.layout;
        if graph.binding_id() != self.binding
            || workspace.binding != self.binding
            || workspace.sequence != self.sequence
            || l.stage != self.layout
            || public.binding != self.binding
            || public.layer != self.layer
            || public.identity != self.public_identity
            || self.batch_pending.is_some()
            || spans.packed_first() != 0
            || projector.global_fault() != fault_id
            || u::words(arena, fault_id)? != 1
            || workspace.fault.is_some_and(|f| f != fault_id)
            || self
                .batch_poison
                .as_ref()
                .is_some_and(|p| !Arc::ptr_eq(p, &workspace.poison))
        {
            return Err(fault("native QSA batch binding/sequence/fault"));
        }
        checked_count(start, count, self.layout.max_context)?;
        selector.validate_owner(arena, &workspace.poison)?;
        validate_view(
            MAX_BATCH,
            spans.input_first(),
            spans.input_stride(),
            self.layout.input_words,
            u::words(arena, input)?,
        )?;
        validate_view(
            MAX_BATCH,
            spans.output_first(),
            spans.output_stride(),
            self.layout.input_words,
            u::words(arena, output)?,
        )?;
        validate_full_regions(arena, workspace, self, public, input, output, fault_id)?;
        self.lifecycle.begin_batch(start, count)?;
        self.batch_poison = Some(workspace.poison.clone());
        let (chunk, generation) = workspace.policy.issue(
            self.layer,
            self.identity,
            start,
            count,
            self.layout.max_context,
        )?;
        workspace.fault = Some(fault_id);
        let metadata = Zeroizing::new(Metadata {
            stamp: Stamp {
                workspace: workspace.identity,
                chunk,
                generation,
                stage: self.identity,
                sequence: self.sequence,
                public: self.public_identity,
                layer: self.layer,
                start,
                count,
                input_first: spans.input_first(),
                input_stride: spans.input_stride(),
                output_first: spans.output_first(),
                output_stride: spans.output_stride(),
            },
            scratch: *workspace.ids,
            owners: [self.ids[CACHE], self.ids[POSITIONS]],
            input,
            output,
            fault: fault_id,
        });
        self.batch_pending = Some(StageAnchor(Zeroizing::new((*metadata).clone())));
        let ids = &workspace.ids;
        let client = arena.client().map_err(arena_fault)?;
        let dim = CubeDim::new_1d(64);
        let a = |i| u::arg(arena, ids[i]);
        let cache = || u::arg(arena, self.ids[CACHE]);
        let pos = || u::arg(arena, self.ids[POSITIONS]);
        let gr = |x| CubeCount::Static(x as u32, count as u32, 1);
        let gamma = |i: usize| {
            u::raw_arg(
                public.vectors[i].clone(),
                if i < 2 { qsa::HD } else { qsa::ID },
            )
        };
        let sin = || u::raw_arg(public.vectors[4].clone(), qsa::ROPE_TABLE_WORDS);
        let cos = || u::raw_arg(public.vectors[5].clone(), qsa::ROPE_TABLE_WORDS);
        queue_stage(count, start as usize, cancel, |op| {
            match op {
                Operation::Clear(i) => {
                    if u::grouped_scratch_clear_enabled() {
                        if i == 0 {
                            u::zero_many(arena, &**ids)?;
                        }
                    } else {
                        u::zero(arena, ids[i])?;
                    }
                }
                Operation::Positions => positions_batch::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    pos()?,
                    a(APPEND)?,
                    start as usize,
                    count,
                    self.layout.max_context,
                    l.widths[APPEND],
                ),
                Operation::Project(i) => {
                    let p = [
                        self.projections.query,
                        self.projections.key,
                        self.projections.value,
                        self.projections.iq,
                        self.projections.ik,
                    ][i];
                    projector.enqueue_batch(
                        p,
                        arena,
                        input,
                        ids[i],
                        &project_input_spans(spans, l.widths[i]),
                        cancel,
                    )?;
                }
                Operation::Split => c::split_batch::launch::<WgpuRuntime>(
                    client,
                    gr((qsa::QUERY * 2).div_ceil(64)),
                    dim,
                    a(RQ)?,
                    a(PACK)?,
                    a(FLAGS)?,
                    count,
                    0,
                    l.widths[RQ],
                    0,
                    l.widths[PACK],
                    0,
                    l.widths[FLAGS],
                ),
                Operation::Norm(i) => {
                    let (input, output, status, heads, width) = [
                        (PACK, Q, FLAGS + 1, qsa::QH, qsa::HD),
                        (RK, K, FLAGS + 2, qsa::KVH, qsa::HD),
                        (RIQ, IQ, FLAGS + 3, qsa::IH, qsa::ID),
                    ][i];
                    c::norm_batch::launch::<WgpuRuntime>(
                        client,
                        gr(heads),
                        dim,
                        a(input)?,
                        gamma(i)?,
                        sin()?,
                        cos()?,
                        pos()?,
                        a(output)?,
                        a(status)?,
                        count,
                        start as usize,
                        heads,
                        0,
                        l.widths[input],
                        0,
                        0,
                        0,
                        qsa::ROPE_PAIRS,
                        0,
                        qsa::ROPE_PAIRS,
                        0,
                        1,
                        0,
                        l.widths[output],
                        0,
                        l.widths[status],
                        0,
                        width,
                    );
                }
                Operation::Pack => c::pack_batch::launch::<WgpuRuntime>(
                    client,
                    gr(qsa::TOKEN.div_ceil(64)),
                    dim,
                    a(K)?,
                    a(RV)?,
                    a(RIK)?,
                    a(CAND)?,
                    a(FLAGS + 4)?,
                    count,
                    0,
                    l.widths[K],
                    0,
                    l.widths[RV],
                    0,
                    l.widths[RIK],
                    0,
                    l.widths[CAND],
                    0,
                    l.widths[FLAGS + 4],
                ),
                Operation::Pool => c::pool_batch::launch::<WgpuRuntime>(
                    client,
                    gr(l.widths[SCORE]),
                    dim,
                    cache()?,
                    a(CAND)?,
                    a(MEAN)?,
                    a(FLAGS + 5)?,
                    start as usize,
                    count,
                    0,
                    qsa::TOKEN,
                    0,
                    l.widths[CAND],
                    0,
                    l.widths[MEAN],
                    0,
                    l.widths[FLAGS + 5],
                ),
                Operation::PoolNorm => c::norm_batch::launch::<WgpuRuntime>(
                    client,
                    gr(l.widths[SCORE]),
                    dim,
                    a(MEAN)?,
                    gamma(3)?,
                    sin()?,
                    cos()?,
                    pos()?,
                    a(POOL)?,
                    a(FLAGS + 6)?,
                    count,
                    start as usize,
                    l.widths[SCORE],
                    0,
                    l.widths[MEAN],
                    0,
                    0,
                    0,
                    qsa::ROPE_PAIRS,
                    0,
                    qsa::ROPE_PAIRS,
                    0,
                    1,
                    0,
                    l.widths[POOL],
                    0,
                    l.widths[FLAGS + 6],
                    1,
                    qsa::ID,
                ),
                Operation::Score => c::score_batch::launch::<WgpuRuntime>(
                    client,
                    gr(l.widths[SCORE]),
                    dim,
                    a(POOL)?,
                    a(IQ)?,
                    a(SCORE)?,
                    a(FLAGS + 7)?,
                    count,
                    start as usize,
                    0,
                    l.widths[POOL],
                    0,
                    l.widths[IQ],
                    0,
                    l.widths[SCORE],
                    0,
                    l.widths[FLAGS + 7],
                ),
                Operation::Selector => selector.select_upload(
                    arena,
                    SelectionLayout::new(l, start as usize, count)?,
                    ids[SCORE],
                    ids[FLAGS + 7],
                    ids[SELECT],
                    fault_id,
                    cancel,
                )?,
                Operation::Logits => c::logits_batch::launch::<WgpuRuntime>(
                    client,
                    gr(qsa::QH * (qsa::BUDGET + 3)),
                    dim,
                    a(Q)?,
                    cache()?,
                    a(CAND)?,
                    a(SELECT)?,
                    a(LOGIT)?,
                    a(FLAGS + 8)?,
                    start as usize,
                    count,
                    0,
                    l.widths[Q],
                    0,
                    qsa::TOKEN,
                    0,
                    l.widths[CAND],
                    0,
                    l.widths[SELECT],
                    0,
                    l.widths[LOGIT],
                    0,
                    l.widths[FLAGS + 8],
                ),
                Operation::Softmax => c::softmax_batch::launch::<WgpuRuntime>(
                    client,
                    gr(qsa::QH),
                    dim,
                    a(LOGIT)?,
                    a(PROB)?,
                    a(FLAGS + 9)?,
                    count,
                    start as usize,
                    0,
                    l.widths[LOGIT],
                    0,
                    l.widths[PROB],
                    0,
                    l.widths[FLAGS + 9],
                ),
                Operation::Values => c::values_batch::launch::<WgpuRuntime>(
                    client,
                    gr(qsa::QUERY.div_ceil(64)),
                    dim,
                    cache()?,
                    a(CAND)?,
                    a(PROB)?,
                    a(SELECT)?,
                    a(PACK)?,
                    a(OUT)?,
                    a(FLAGS + 10)?,
                    start as usize,
                    count,
                    0,
                    qsa::TOKEN,
                    0,
                    l.widths[CAND],
                    0,
                    l.widths[PROB],
                    0,
                    l.widths[SELECT],
                    0,
                    l.widths[PACK],
                    0,
                    l.widths[OUT],
                    0,
                    l.widths[FLAGS + 10],
                ),
                Operation::Output => projector.enqueue_batch(
                    self.projections.output,
                    arena,
                    ids[OUT],
                    output,
                    &BatchSpans::new(
                        count,
                        0,
                        0,
                        l.widths[OUT],
                        spans.output_first(),
                        spans.output_stride(),
                    ),
                    cancel,
                )?,
                Operation::Append => append_batch::launch::<WgpuRuntime>(
                    client,
                    gr(qsa::TOKEN.div_ceil(64)),
                    dim,
                    a(CAND)?,
                    cache()?,
                    pos()?,
                    a(APPEND)?,
                    start as usize,
                    count,
                    self.layout.max_context,
                    l.widths[CAND],
                    l.widths[APPEND],
                ),
                Operation::Fold(i, t) => {
                    if t == MAX_BATCH
                        && (workspace.public_grouped_status.state().0
                            || crate::status_fold_batch::selected())
                    {
                        if i == 0 {
                            let views = [0, 1, 2, 3, 4, 9, 10, 11].map(|j| {
                                crate::status_fold_batch::View::new(
                                    ids[FLAGS + j],
                                    count,
                                    0,
                                    l.widths[FLAGS + j],
                                    l.widths[FLAGS + j],
                                )
                            });
                            if workspace.public_grouped_status.state().0 {
                                workspace
                                    .public_grouped_status
                                    .launch(arena, &views, fault_id)?;
                            } else {
                                crate::status_fold_batch::launch(arena, &views, fault_id)?;
                            }
                        }
                    } else {
                        let (width, first, n) = if t == MAX_BATCH {
                            (l.widths[FLAGS + i], 0, count)
                        } else {
                            let token = l.token(t, start as usize)?;
                            let width = if i == 8 {
                                token.logits_elements
                            } else {
                                token.complete
                            };
                            (width, t * l.widths[FLAGS + i], 1)
                        };
                        u::status_batch(
                            arena,
                            ids[FLAGS + i],
                            fault_id,
                            n,
                            first,
                            l.widths[FLAGS + i],
                            width,
                        )?;
                    }
                }
            }
            Ok(())
        })?;
        Ok(PendingQsaBatch {
            metadata,
            poison: workspace.poison.clone(),
            armed: true,
        })
    }
    /// This permit preflighted ALL graph-derived QSA owners BEFORE mutation;
    /// future coordinator must also preflight GDN/PLE/other families first.
    pub(crate) fn commit_after_whole_chunk_acceptance(
        &mut self,
        mut receipt: PendingQsaBatch,
        arena: &PrivateArena,
        acceptance: &mut QsaBatchAcceptance<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let attempted = (|| {
            acceptance.workspace.healthy(arena, cancel)?;
            acceptance.workspace.check_receipt(self, &receipt, arena)?;
            let stamp = receipt.metadata.stamp;
            checked_count(stamp.start, stamp.count, self.layout.max_context)?;
            acceptance.workspace.policy.commit(stamp)?;
            self.lifecycle.commit_batch(stamp.start, stamp.count)?;
            self.batch_pending = None;
            receipt.armed = false;
            cancelled(cancel)
        })();
        if attempted.is_err() {
            self.poison();
            acceptance.workspace.poison.store(true, Ordering::Release);
        }
        attempted
    }
}

/// One exact committed raw KV/raw-indexer suffix, never a position-only rewind.
pub(crate) struct LatestQsaCheckpoint {
    metadata: Zeroizing<LatestQsaMetadata>,
    binding: Option<GraphBindingId>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
#[derive(Clone, PartialEq, Eq)]
struct LatestQsaMetadata {
    workspace: u64,
    stage: u64,
    public: u64,
    sequence: SequenceKey,
    layer: usize,
    start: u64,
    count: usize,
    chunk: u64,
    generation: u64,
    owners: [BufferId; 2],
    fault: BufferId,
}
impl Zeroize for LatestQsaMetadata {
    fn zeroize(&mut self) {
        self.workspace.zeroize();
        self.stage.zeroize();
        self.public.zeroize();
        self.sequence.zeroize();
        self.layer.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.chunk.zeroize();
        self.generation.zeroize();
        self.owners.zeroize();
        self.fault.zeroize();
    }
}
fn latest_qsa_order(old: &LatestQsaMetadata, next: &LatestQsaMetadata, stages: u32) -> Result<()> {
    let mut expected = Zeroizing::new(old.clone());
    expected.chunk = old.chunk.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
    expected.generation = old
        .generation
        .checked_add(stages as u64)
        .ok_or(Error::ArithmeticOverflow)?;
    if *expected != *next {
        return Err(fault("native latest QSA checkpoint identity"));
    }
    Ok(())
}
fn latest_qsa_roots(
    stage: &ResidentQsa<'_>,
    arena: &PrivateArena,
    fault_id: BufferId,
) -> Result<()> {
    let ids = [stage.ids[CACHE], stage.ids[POSITIONS], fault_id];
    let words = [
        stage.layout.max_context * qsa::TOKEN,
        stage.layout.max_context,
        1,
    ];
    let mut handles = Vec::with_capacity(3);
    for (i, (id, words)) in ids.into_iter().zip(words).enumerate() {
        if ids[..i].contains(&id) {
            return Err(fault("native latest QSA root alias"));
        }
        let layout = arena.layout(id).map_err(arena_fault)?;
        let handle = arena.handle(id).map_err(arena_fault)?;
        if layout != BufferLayout::new(words).map_err(arena_fault)?
            || handle.offset_start.unwrap_or(0) != 0
            || handle.offset_end.unwrap_or(0) != 0
            || handle.size() != layout.owned_bytes as u64
            || handle.size_in_used() != layout.owned_bytes as u64
            || handles
                .first()
                .is_some_and(|h: &Handle| h.stream != handle.stream)
        {
            return Err(fault("native latest QSA full roots"));
        }
        handles.push(handle.clone());
    }
    let resources = u::resources(arena.client().map_err(arena_fault)?, handles)?;
    if resources.len() != 3 {
        return Err(Error::InvalidBufferLength);
    }
    for (i, r) in resources.iter().enumerate() {
        let page = r.resource();
        let bytes = arena.layout(ids[i]).map_err(arena_fault)?.owned_bytes as u64;
        validate_page(page.offset, page.size, page.buffer.size(), bytes)?;
        if resources[..i]
            .iter()
            .any(|old| old.resource().buffer == page.buffer)
        {
            return Err(fault("native latest QSA backing alias"));
        }
    }
    Ok(())
}
impl QsaBatchWorkspace {
    pub(crate) fn checkpoint_latest(
        &self,
        stage: &ResidentQsa<'_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<LatestQsaCheckpoint> {
        self.checkpoint_latest_with_four_rows(stage, arena, start, count, fault_id, false, cancel)
    }
    #[allow(clippy::too_many_arguments)] // Actual family/owner/window plus immutable four-row scope.
    pub(crate) fn checkpoint_latest_with_four_rows(
        &self,
        stage: &ResidentQsa<'_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        fault_id: BufferId,
        four: bool,
        cancel: &AtomicBool,
    ) -> Result<LatestQsaCheckpoint> {
        self.healthy(arena, cancel)?;
        checked_count(start, count, stage.layout.max_context)?;
        if !(1..=if four { 4 } else { 3 }).contains(&count)
            || self.policy.phase != ChunkPhase::Idle
            || *self.policy.issued != 0
            || *self.policy.committed != 0
            || stage.lifecycle.phase != Phase::Ready(start)
            || stage.batch_pending.is_some()
            || stage.sequence != self.sequence
            || stage.binding != self.binding
            || stage.layout != self.layout.stage
            || stage.layer >= MAX_LAYERS
            || self.policy.expected & (1u64 << stage.layer) == 0
            || stage
                .batch_poison
                .as_ref()
                .is_some_and(|p| !Arc::ptr_eq(p, &self.poison))
        {
            return Err(fault("native latest QSA checkpoint boundary"));
        }
        latest_qsa_roots(stage, arena, fault_id)?;
        self.policy
            .chunk
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.policy
            .generation
            .checked_add(self.policy.expected.count_ones() as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(LatestQsaCheckpoint {
            metadata: Zeroizing::new(LatestQsaMetadata {
                workspace: self.identity,
                stage: stage.identity,
                public: stage.public_identity,
                sequence: stage.sequence,
                layer: stage.layer,
                start,
                count,
                chunk: *self.policy.chunk,
                generation: *self.policy.generation,
                owners: [stage.ids[CACHE], stage.ids[POSITIONS]],
                fault: fault_id,
            }),
            binding: Some(self.binding),
            poison: self.poison.clone(),
            armed: true,
        })
    }
}
impl LatestQsaCheckpoint {
    pub(crate) fn layer(&self) -> usize {
        self.metadata.layer
    }
    pub(crate) fn start(&self) -> usize {
        self.metadata.start as usize
    }
    pub(crate) fn count(&self) -> usize {
        self.metadata.count
    }
    pub(crate) fn binding(&self) -> Option<GraphBindingId> {
        self.binding
    }
    pub(crate) fn validate_completed(
        &self,
        stage: &ResidentQsa<'_>,
        workspace: &QsaBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        workspace.healthy(arena, cancel)?;
        // Original whole-chunk acceptance erases its temporary fault pointer.
        // An idle completed workspace must have no live candidate pointer;
        // the checkpoint retains the exact separately validated physical owner.
        let completed_fault = latest_completed_fault(workspace.fault, self.metadata.fault)?;
        if !self.armed
            || workspace.policy.phase != ChunkPhase::Idle
            || *workspace.policy.issued != 0
            || *workspace.policy.committed != 0
            || self.binding != Some(stage.binding)
            || stage.binding != workspace.binding
            || !Arc::ptr_eq(&self.poison, &workspace.poison)
            || stage
                .batch_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
            || stage.batch_pending.is_some()
            || stage.layout != workspace.layout.stage
            || stage.lifecycle.phase
                != Phase::Ready(checked_count(
                    self.metadata.start,
                    self.metadata.count,
                    stage.layout.max_context,
                )?)
        {
            return Err(fault("native latest QSA completed checkpoint"));
        }
        latest_qsa_order(
            &self.metadata,
            &LatestQsaMetadata {
                workspace: workspace.identity,
                stage: stage.identity,
                public: stage.public_identity,
                sequence: stage.sequence,
                layer: stage.layer,
                start: self.metadata.start,
                count: self.metadata.count,
                chunk: *workspace.policy.chunk,
                generation: *workspace.policy.generation,
                owners: [stage.ids[CACHE], stage.ids[POSITIONS]],
                fault: completed_fault,
            },
            workspace.policy.expected.count_ones(),
        )?;
        latest_qsa_roots(stage, arena, self.metadata.fault)
    }
    pub(crate) fn prepare_restore<'a, 'f>(
        self,
        stage: &'a mut ResidentQsa<'f>,
        workspace: &QsaBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<LatestQsaRestore<'a, 'f>> {
        self.validate_completed(stage, workspace, arena, cancel)?;
        Ok(LatestQsaRestore {
            checkpoint: self,
            stage,
        })
    }
    pub(crate) fn accept_completed(
        mut self,
        stage: &ResidentQsa<'_>,
        workspace: &QsaBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_completed(stage, workspace, arena, cancel)?;
        self.armed = false;
        Ok(())
    }
}
fn latest_completed_fault(active: Option<BufferId>, pinned: BufferId) -> Result<BufferId> {
    if active.is_some() {
        return Err(fault("native latest QSA incomplete candidate fault owner"));
    }
    Ok(pinned)
}
impl Drop for LatestQsaCheckpoint {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
        self.binding = None;
    }
}
pub(crate) struct LatestQsaRestore<'a, 'f> {
    checkpoint: LatestQsaCheckpoint,
    stage: &'a mut ResidentQsa<'f>,
}
pub(crate) struct LatestQsaCleared<'a, 'f> {
    checkpoint: LatestQsaCheckpoint,
    stage: &'a mut ResidentQsa<'f>,
}
pub(crate) const LATEST_QSA_RUNTIME_INFO: u64 = 12 * 2 * 2048;
impl<'a, 'f> LatestQsaRestore<'a, 'f> {
    /// Caller must preflight ALL state families before the first call. Two
    /// runtime INFO records per QSA stage: clear and exact cleared-suffix fold.
    /// No new MAIN owner, full clear of previous prefix or fault reset.
    pub(crate) fn enqueue_clear(
        self,
        arena: &PrivateArena,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<LatestQsaCleared<'a, 'f>> {
        cancelled(cancel)?;
        latest_qsa_roots(self.stage, arena, self.checkpoint.metadata.fault)?;
        let ids = self.checkpoint.metadata.owners;
        if fault_id != self.checkpoint.metadata.fault
            || ids.contains(&fault_id)
            || arena.layout(fault_id).map_err(arena_fault)?.logical_words != 1
        {
            return Err(fault("native latest QSA clear fault owner"));
        }
        let words = self.checkpoint.metadata.count * qsa::TOKEN;
        clear_latest_suffix::launch::<WgpuRuntime>(
            arena.client().map_err(arena_fault)?,
            u::grid(words)?,
            CubeDim::new_1d(64),
            u::arg(arena, ids[0])?,
            u::arg(arena, ids[1])?,
            self.checkpoint.metadata.start as usize,
            self.checkpoint.metadata.count,
        );
        verify_latest_cleared::launch::<WgpuRuntime>(
            arena.client().map_err(arena_fault)?,
            u::grid(words)?,
            CubeDim::new_1d(64),
            u::arg(arena, ids[0])?,
            u::arg(arena, ids[1])?,
            u::arg(arena, fault_id)?,
            self.checkpoint.metadata.start as usize,
            self.checkpoint.metadata.count,
        );
        cancelled(cancel)?;
        Ok(LatestQsaCleared {
            checkpoint: self.checkpoint,
            stage: self.stage,
        })
    }
}
impl LatestQsaCleared<'_, '_> {
    /// External completed original fault/audit proof is required first. All
    /// fallible work precedes this cursor publication; monotonic policy stays.
    pub(crate) fn commit(mut self) {
        self.stage.lifecycle.phase = Phase::Ready(self.checkpoint.metadata.start);
        self.checkpoint.armed = false;
    }
}

/// Fixed-public healthy cursor permit. Full cache, position and scratch pages
/// are erased together by the parent arena; no suffix-only credit is accepted.
pub(crate) struct PublicWarmQsaReset<'a, 'f> {
    stage: &'a mut ResidentQsa<'f>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl QsaBatchWorkspace {
    pub(crate) fn prepare_public_warm_reset<'a, 'f>(
        &self,
        stage: &'a mut ResidentQsa<'f>,
        arena: &PrivateArena,
        fault_id: BufferId,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PublicWarmQsaReset<'a, 'f>> {
        self.healthy(arena, cancel)?;
        if position != 13
            || self.policy.phase != ChunkPhase::Idle
            || *self.policy.chunk == 0
            || *self.policy.issued != 0
            || *self.policy.committed != 0
            || *self.policy.start != 0
            || *self.policy.count != 0
            || self.policy.generations.iter().any(|&v| v != 0)
            || self.policy.stage_ids.iter().any(|&v| v != 0)
            || stage.lifecycle.phase != Phase::Ready(position)
            || stage.batch_pending.is_some()
            || stage.binding != self.binding
            || stage.sequence != self.sequence
            || stage.layout != self.layout.stage
            || stage.layer >= MAX_LAYERS
            || self.policy.expected & (1u64 << stage.layer) == 0
            || stage
                .batch_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
            || self.fault.is_some()
        {
            return Err(fault("native public warm QSA boundary"));
        }
        latest_qsa_roots(stage, arena, fault_id)?;
        Ok(PublicWarmQsaReset {
            stage,
            poison: self.poison.clone(),
            armed: false,
        })
    }
}
impl PublicWarmQsaReset<'_, '_> {
    pub(crate) fn arm(&mut self) {
        self.armed = true;
    }
    /// Requires complete parent erasure/fault/I/O proof, then no fallible work.
    pub(crate) fn commit(
        mut self,
        _proof: &crate::resident_model::batch::public_warm_replay::CompletedReset,
    ) {
        self.stage.lifecycle.phase = Phase::Ready(0);
        self.armed = false;
    }
}
impl Drop for PublicWarmQsaReset<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
    }
}
#[cube(launch)]
fn clear_latest_suffix(
    cache: &mut Array<f32>,
    positions: &mut Array<u32>,
    start: usize,
    count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count * qsa::TOKEN {
        cache[start * qsa::TOKEN + i] = 0.;
    }
    if i < count {
        positions[start + i] = 0;
    }
}
#[cube(launch)]
fn verify_latest_cleared(
    cache: &Array<f32>,
    positions: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    start: usize,
    count: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count * qsa::TOKEN && u32::reinterpret(cache[start * qsa::TOKEN + i]) != 0 {
        fault[0].fetch_or(4);
    }
    if i < count && positions[start + i] != 0 {
        fault[0].fetch_or(4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latest_checkpoint_uses_pinned_fault_after_original_candidate_pointer_erasure() {
        let pinned = BufferId::metadata_fixture(31, 38, 2);
        let foreign = BufferId::metadata_fixture(31, 38, 3);
        // Original acceptance.finish clears this transient association. A
        // completed checkpoint must retain its own exact physical owner.
        let mut candidate = Some(pinned);
        assert!(latest_completed_fault(candidate, pinned).is_err());
        candidate.as_mut().unwrap().zeroize();
        candidate = None;
        assert!(latest_completed_fault(candidate, pinned).unwrap() == pinned);
        assert!(latest_completed_fault(Some(foreign), pinned).is_err());
    }
    fn latest() -> LatestQsaMetadata {
        LatestQsaMetadata {
            workspace: 7,
            stage: 11,
            public: 13,
            sequence: SequenceKey::new().unwrap(),
            layer: 3,
            start: 23,
            count: 3,
            chunk: 29,
            generation: 100,
            owners: std::array::from_fn(|i| BufferId::metadata_fixture(31, 37, i)),
            fault: BufferId::metadata_fixture(31, 37, 2),
        }
    }
    #[test]
    fn latest_qsa_rejects_foreign_roots_stale_epochs_and_more_than_one_chunk() {
        let old = latest();
        let mut completed = old.clone();
        completed.chunk += 1;
        completed.generation += 12;
        latest_qsa_order(&old, &completed, 12).unwrap();
        for field in 0..11 {
            let mut wrong = completed.clone();
            match field {
                0 => wrong.workspace += 1,
                1 => wrong.stage += 1,
                2 => wrong.public += 1,
                3 => wrong.sequence = SequenceKey::new().unwrap(),
                4 => wrong.layer += 1,
                5 => wrong.start += 1,
                6 => wrong.count -= 1,
                7 => wrong.chunk += 1,
                8 => wrong.generation -= 1,
                9 => wrong.owners[0] = BufferId::metadata_fixture(31, 38, 0),
                _ => wrong.fault = BufferId::metadata_fixture(31, 38, 2),
            }
            assert!(latest_qsa_order(&old, &wrong, 12).is_err());
        }
        assert!(latest_qsa_order(&old, &old, 12).is_err());
    }
    #[test]
    fn latest_qsa_suffix_clear_and_strict_zero_fold_compile_without_touching_prefix() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for verify in [false, true] {
                let mut b = KernelBuilder::default();
                let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
                b.scope.register_type::<u32>(u);
                b.scope.register_type::<usize>(u);
                b.scope.register_type::<f32>(f);
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                let start: NativeExpand<usize> = b.scalar(u).into();
                let count: NativeExpand<usize> = b.scalar(u).into();
                if verify {
                    let cache = b.input_array(Type::new(f)).into();
                    let positions = b.input_array(Type::new(u)).into();
                    let fault = b
                        .output_array(Type::new(StorageType::Atomic(ElemType::UInt(
                            UIntKind::U32,
                        ))))
                        .into();
                    verify_latest_cleared::expand(
                        &mut b.scope,
                        cache,
                        positions,
                        fault,
                        start,
                        count,
                    );
                } else {
                    let cache = b.output_array(Type::new(f)).into();
                    let positions = b.output_array(Type::new(u)).into();
                    clear_latest_suffix::expand(&mut b.scope, cache, positions, start, count);
                }
                let source = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("latest_qsa"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        u,
                    )
                    .unwrap()
                    .to_string();
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                let info = wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::empty(),
                )
                .validate(&module)
                .unwrap();
                let words = wgpu::naga::back::spv::write_vec(
                    &module,
                    &info,
                    &wgpu::naga::back::spv::Options::default(),
                    None,
                )
                .unwrap();
                assert!(!words.is_empty());
                assert!(source.contains("scalars_u32[0]"));
                assert!(source.contains("scalars_u32[1]"));
                if verify {
                    assert!(source.contains("atomicOr"));
                } else {
                    assert!(!source.contains("atomicStore"));
                }
            }
        }
        // Runtime offset bounds match all current chunk sizes, including last
        // context rows. No whole-cache scan exceeds portable dispatch limits.
        for count in 1..=3 {
            for start in [0, 1, 23, 4096 - count] {
                let from = start * qsa::TOKEN;
                let end = (start + count) * qsa::TOKEN;
                assert!(from <= end && end <= 4096 * qsa::TOKEN);
                assert!(u::grid(count * qsa::TOKEN).is_ok());
            }
        }
        assert_eq!(LATEST_QSA_RUNTIME_INFO, 49_152);
    }
    fn layout() -> Layout {
        Layout::new(StageLayout {
            input_words: 2560,
            max_context: 4096,
            epsilon: qsa::EPSILON,
        })
        .unwrap()
    }
    fn stamp(layer: usize, stage: u64, chunk: u64, generation: u64, count: usize) -> Stamp {
        Stamp {
            workspace: 77,
            chunk,
            generation,
            stage,
            sequence: SequenceKey::new().unwrap(),
            public: 123,
            layer,
            start: 0,
            count,
            input_first: 7,
            input_stride: 2573,
            output_first: 11,
            output_stride: 2579,
        }
    }
    fn metadata(stamp: Stamp) -> Metadata {
        Metadata {
            stamp,
            scratch: std::array::from_fn(|i| BufferId::metadata_fixture(7, 13, i)),
            owners: std::array::from_fn(|i| BufferId::metadata_fixture(7, 13, S + i)),
            input: BufferId::metadata_fixture(7, 13, S + 2),
            output: BufferId::metadata_fixture(7, 13, S + 3),
            fault: BufferId::metadata_fixture(7, 13, S + 4),
        }
    }
    #[test]
    fn exact_full_pages_one_shared_bank_and_no_cache_duplicates() {
        let l = layout();
        assert_eq!(l.bytes().unwrap(), 17_825_792);
        assert_eq!(
            l.requests().unwrap(),
            [
                524288, 32768, 32768, 32768, 32768, 524288, 262144, 32768, 32768, 65536, 4194304,
                4194304, 32768, 131072, 2097152, 2097152, 262144, 524288, 32768, 32768, 32768,
                65536, 32768, 32768, 32768, 2097152, 32768, 262144, 65536
            ]
        );
        assert_eq!(l.stage.owned_bytes().unwrap(), 36_306_944);
        assert_eq!(785_186_816 + l.bytes().unwrap(), 803_012_608);
        assert_eq!(l.io_bytes(false).unwrap(), 327680);
        assert_eq!(l.io_bytes(true).unwrap(), 655360);
        assert_eq!(l.io_bytes(true).unwrap() * 12, 7_864_320);
        // This component never grants capacity to unrelated pending IO.
        assert!(l.io_bytes(true).unwrap() * 12 < 64 * 1024 * 1024);
        for ctx in [0, 1, 3, 4097] {
            assert!(
                Layout::new(StageLayout {
                    max_context: ctx,
                    ..l.stage
                })
                .is_err()
            );
        }
        for eps in [0., f32::NAN, f32::INFINITY] {
            assert!(
                Layout::new(StageLayout {
                    epsilon: eps,
                    ..l.stage
                })
                .is_err()
            );
        }
    }
    #[test]
    fn expected_qsa_mask_is_graph_derived_and_not_caller_supplied() {
        let mask = expected_stages(48, |i| {
            Ok(if i % 4 == 3 {
                AttentionKind::Qsa
            } else {
                AttentionKind::Gdn
            })
        })
        .unwrap();
        assert_eq!(mask.count_ones(), 12);
        for i in 0..48 {
            assert_eq!(mask & (1 << i) != 0, i % 4 == 3);
        }
        for n in [0, 65] {
            assert!(expected_stages(n, |_| Ok(AttentionKind::Qsa)).is_err());
        }
        assert!(expected_stages(48, |_| Ok(AttentionKind::Gdn)).is_err());
        assert!(
            expected_stages(48, |i| if i == 7 {
                Err(Error::OutOfBounds)
            } else {
                Ok(AttentionKind::Qsa)
            })
            .is_err()
        );
    }
    #[test]
    fn runtime_projection_offsets_preserve_original_packed_base_and_all_columns() {
        for count in [1, 3, 5, 8] {
            for width in [12288, 512, 128] {
                let parent = BatchSpans::new(count, 0, 7, 2573, 11, 2579);
                let p = project_input_spans(&parent, width);
                assert_eq!(p.packed_first(), 0);
                assert_eq!(p.input_first(), 7);
                assert_eq!(p.input_stride(), 2573);
                assert_eq!(p.output_first(), 0);
                assert_eq!(p.output_stride(), width);
                assert_eq!(p.count(), count);
                let full = 7 + 7 * 2573 + 2560;
                validate_view(MAX_BATCH, 7, 2573, 2560, full).unwrap();
                assert!(validate_view(MAX_BATCH, 7, 2573, 2560, full - 1).is_err());
            }
        }
        for count in [0, 9, usize::MAX] {
            assert!(checked_count(0, count, 4096).is_err());
        }
        assert!(checked_count(u64::MAX, 1, 4096).is_err());
        for count in [1, 3, 5, 8] {
            assert_eq!(
                checked_count(4096 - count as u64, count, 4096).unwrap(),
                4096
            );
            assert!(checked_count(4097 - count as u64, count, 4096).is_err());
        }
    }
    #[test]
    fn recorded_generation_survives_reuse_but_all_stage_preflight_precedes_commit() {
        for count in [1, 3, 5, 8] {
            let mut p = Policy::new((1 << 3) | (1 << 7) | (1 << 11));
            let mut stamps = Zeroizing::new(Vec::new());
            for layer in [3, 7, 11] {
                let (c, g) = p.issue(layer, 100 + layer as u64, 0, count, 4096).unwrap();
                stamps.push(stamp(layer, 100 + layer as u64, c, g, count));
                for &s in stamps.iter() {
                    p.check(s).unwrap();
                }
                assert!(p.commit(stamps[0]).is_err());
                if layer != 11 {
                    assert!(p.accept(*p.issued).is_err());
                }
            }
            p.accept(*p.issued).unwrap();
            for &s in stamps.iter() {
                p.commit(s).unwrap();
            }
            assert!(p.commit(stamps[0]).is_err());
            p.finish().unwrap();
            let (c, g) = p.issue(3, 103, 0, count, 4096).unwrap();
            assert!(c > stamps[0].chunk && g > stamps[2].generation);
            assert!(p.check(stamps[0]).is_err());
        }
    }
    #[test]
    fn duplicate_wrong_order_stale_mixed_and_incomplete_ranges_cannot_authorize_commit() {
        let mut p = Policy::new((1 << 3) | (1 << 7));
        assert!(p.issue(7, 7, 0, 3, 4096).is_err());
        let (c, g) = p.issue(3, 3, 0, 3, 4096).unwrap();
        let s = stamp(3, 3, c, g, 3);
        for (layer, start, count) in [(3, 0, 3), (4, 0, 3), (7, 1, 3), (7, 0, 5)] {
            assert!(p.issue(layer, layer as u64, start, count, 4096).is_err());
        }
        for s in [
            Stamp {
                generation: g + 1,
                ..s
            },
            Stamp { stage: 4, ..s },
            Stamp { chunk: c + 1, ..s },
            Stamp { count: 5, ..s },
            Stamp { start: 1, ..s },
            Stamp { layer: 64, ..s },
        ] {
            assert!(p.check(s).is_err());
        }
        assert!(p.accept(1 << 3).is_err());
        assert!(p.commit(s).is_err());
        assert!(p.finish().is_err());
        for generation in [false, true] {
            let mut q = Policy::new(1);
            if generation {
                *q.generation = u64::MAX
            } else {
                *q.chunk = u64::MAX
            };
            assert!(q.issue(0, 1, 0, 1, 4096).is_err());
        }
    }
    #[test]
    fn queue_dependency_is_once_and_all_exact_prefix_status_folds_precede_reuse() {
        let cancel = AtomicBool::new(false);
        for (start, count) in [(0, 1), (0, 3), (3, 5), (2047, 8), (2048, 8), (2052, 8)] {
            let mut events = Vec::new();
            queue_stage(count, start, &cancel, |op| {
                events.push(op);
                Ok(())
            })
            .unwrap();
            assert!(
                events[..S]
                    .iter()
                    .enumerate()
                    .all(|(i, &op)| op == Operation::Clear(i))
            );
            assert_eq!(
                events.iter().filter(|&&x| x == Operation::Selector).count(),
                1
            );
            let dependency = events
                .iter()
                .position(|&x| x == Operation::Selector)
                .unwrap();
            let append = events.iter().position(|&x| x == Operation::Append).unwrap();
            assert!(events.iter().position(|&x| x == Operation::Score).unwrap() < dependency);
            assert!(dependency < events.iter().position(|&x| x == Operation::Logits).unwrap());
            assert!(
                events[..append]
                    .iter()
                    .all(|x| !matches!(x, Operation::Fold(..)))
            );
            let fixed = events
                .iter()
                .filter(|&&op| matches!(op, Operation::Fold(_, MAX_BATCH)))
                .count();
            assert_eq!(fixed, 8);
            let expected = (0..count)
                .map(|t| if (start + t + 1) / 4 > 0 { 4 } else { 1 })
                .sum::<usize>();
            assert_eq!(events.len() - append - 1, 8 + expected);
            assert!(events.len() as u64 + 7 < IO_DISPATCH_BOUND);
        }
    }
    #[test]
    fn errors_panics_and_late_cancel_poison_at_every_real_dispatch_boundary() {
        let mut events = Vec::new();
        queue_stage(8, 2048, &AtomicBool::new(false), |op| {
            events.push(op);
            Ok(())
        })
        .unwrap();
        for failure in 0..events.len() {
            for mode in 0..3 {
                let poison = AtomicBool::new(false);
                let cancel = AtomicBool::new(false);
                let mut n = 0;
                let attempt = guarded(&poison, || {
                    queue_stage(8, 2048, &cancel, |_| {
                        let here = n;
                        n += 1;
                        if here == failure {
                            match mode {
                                0 => return Err(Error::InvalidBufferLength),
                                1 => panic!("fixed public queue failure"),
                                _ => cancel.store(true, Ordering::Release),
                            }
                        }
                        Ok(())
                    })
                });
                assert!(attempt.is_err() && poison.load(Ordering::Acquire));
                assert_eq!(n, failure + 1);
            }
        }
    }
    #[test]
    fn exact_owning_score_maps_and_late_status_failure_match_guarded_core_selection() {
        let l = layout();
        let cancel = AtomicBool::new(false);
        for start in [0, 3, 4, 2047, 2048, 2049, 2051] {
            for count in [1, 3, 5, 8] {
                let s = SelectionLayout::new(l, start, count).unwrap();
                let words = BufferLayout::new(l.words[SCORE]).unwrap().owned_words;
                let mut scores = Zeroizing::new(vec![0; words]);
                let mut flags = Zeroizing::new(vec![0; words]);
                for t in 0..count {
                    let q = QsaLayout::new(start + t + 1).unwrap();
                    for b in 0..q.complete {
                        scores[t * s.blocks + b] = (if b % 7 == 0 {
                            -0.0f32
                        } else {
                            ((b * 13) % 29) as f32
                        })
                        .to_bits();
                        flags[t * s.blocks + b] = 1;
                    }
                }
                let ids = if s.needs_read() {
                    s.select(&scores, &flags, &cancel).unwrap()
                } else {
                    s.select(&[], &[], &cancel).unwrap()
                };
                for t in 0..count {
                    let q = QsaLayout::new(start + t + 1).unwrap();
                    let values: Vec<_> = scores[t * s.blocks..][..q.complete]
                        .iter()
                        .map(|&x| f32::from_bits(x))
                        .collect();
                    let expected =
                        kuno_text::ops::qsa_select_blocks(&values, q.visible, 4, 2048).unwrap();
                    assert!(
                        ids[t * s.stride..][..q.selected]
                            .iter()
                            .map(|&id| id as usize)
                            .eq(expected)
                    );
                    assert!(
                        ids[t * s.stride + q.selected..(t + 1) * s.stride]
                            .iter()
                            .all(|&x| x == 0)
                    );
                }
                assert!(ids[count * s.stride..].iter().all(|&x| x == 0));
                if s.needs_read() {
                    assert!(s.select(&scores[..words - 1], &flags, &cancel).is_err());
                    let mut extra = scores.to_vec();
                    extra.push(0);
                    assert!(s.select(&extra, &flags, &cancel).is_err());
                    let q = QsaLayout::new(start + count).unwrap();
                    flags[(count - 1) * s.blocks + q.complete - 1] = 0;
                    assert!(s.select(&scores, &flags, &cancel).is_err());
                }
                cancel.store(true, Ordering::Release);
                assert!(s.select(&[], &[], &cancel).is_err());
                cancel.store(false, Ordering::Release);
            }
        }
    }
    #[test]
    fn owning_decode_and_original_logical_public_views_reject_bad_extents() {
        let l = BufferLayout::new(8 * 1024).unwrap();
        let mut bytes = vec![0; l.owned_bytes];
        bytes[4..8].copy_from_slice(&0x7fc0_0011u32.to_le_bytes());
        assert_eq!(decode_map(&bytes, l).unwrap()[1], 0x7fc0_0011);
        for len in [0, l.owned_bytes - 4, l.owned_bytes + 4] {
            assert!(decode_map(&vec![0; len], l).is_err());
        }
        let stream = cubecl::stream_id::StreamId::current();
        for words in [qsa::HD, qsa::ID, qsa::ROPE_TABLE_WORDS] {
            let page = BufferLayout::new(words).unwrap().owned_bytes as u64;
            let used = words as u64 * 4;
            let h = Handle::new(stream, page).offset_end(page - used);
            let full = full_public(&h, words).unwrap();
            assert_eq!(h.size_in_used(), used);
            assert_eq!(full.size_in_used(), page);
            assert!(full_public(&h.clone().offset_start(4), words).is_err());
            assert!(
                full_public(
                    &Handle::new(stream, page * 2).offset_end(page * 2 - used),
                    words
                )
                .is_err()
            );
            if page != used {
                assert!(full_public(&Handle::new(stream, page), words).is_err());
            }
        }
        validate_page(0, 32768, 32768, 32768).unwrap();
        for x in [
            (4, 32768, 32768, 32768),
            (0, 32764, 32768, 32768),
            (0, 32768, 65536, 32768),
            (0, 32768, 16384, 32768),
            (0, 0, 0, 0),
        ] {
            assert!(validate_page(x.0, x.1, x.2, x.3).is_err());
        }
    }
    #[test]
    fn pending_drop_poison_and_full_fixed_metadata_wipe_cover_all_owner_ids() {
        let mut m = metadata(stamp(3, 11, 13, 17, 8));
        m.zeroize();
        let zero = BufferId::metadata_fixture(0, 0, 0);
        let mut sequence = SequenceKey::new().unwrap();
        sequence.zeroize();
        assert!(m.stamp.sequence == sequence);
        assert!(
            m.scratch
                .iter()
                .chain(m.owners.iter())
                .chain([&m.input, &m.output, &m.fault])
                .all(|&id| id == zero)
        );
        assert_eq!(
            [
                m.stamp.workspace,
                m.stamp.chunk,
                m.stamp.generation,
                m.stamp.stage,
                m.stamp.public,
                m.stamp.start
            ],
            [0; 6]
        );
        assert_eq!(
            [
                m.stamp.layer,
                m.stamp.count,
                m.stamp.input_first,
                m.stamp.input_stride,
                m.stamp.output_first,
                m.stamp.output_stride
            ],
            [0; 6]
        );
        for armed in [true, false] {
            let poison = Arc::new(AtomicBool::new(false));
            drop(PendingQsaBatch {
                metadata: Zeroizing::new(metadata(stamp(3, 11, 13, 17, 8))),
                poison: poison.clone(),
                armed,
            });
            assert_eq!(poison.load(Ordering::Acquire), armed);
        }
    }
    #[test]
    fn batch_cursor_accepts_exact_count_and_then_original_scalar_next_token() {
        for count in [1, 3, 5, 8] {
            let mut l = Lifecycle::new(4096);
            l.begin_batch(0, count).unwrap();
            assert!(l.commit(0).is_err());
            let mut l = Lifecycle::new(4096);
            l.begin_batch(0, count).unwrap();
            l.commit_batch(0, count).unwrap();
            assert!(l.phase == Phase::Ready(count as u64));
            l.begin(count as u64).unwrap();
            l.commit(count as u64).unwrap();
            assert!(l.phase == Phase::Ready(count as u64 + 1));
            let mut l = Lifecycle::new(4096);
            l.begin(0).unwrap();
            assert!(l.commit_batch(0, count).is_err());
        }
        let mut l = Lifecycle::new(4096);
        assert!(l.begin_batch(4090, 8).is_err());
        assert!(l.phase == Phase::Poisoned);
    }
    #[test]
    fn new_suffix_kernels_compile_checked_unchecked_with_all_request_values_runtime() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for append in [false, true] {
                let mut b = KernelBuilder::default();
                let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
                b.scope.register_type::<u32>(u);
                b.scope.register_type::<usize>(u);
                b.scope.register_type::<f32>(f);
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                let n = if append { 5 } else { 4 };
                let args: Vec<NativeExpand<usize>> = (0..n).map(|_| b.scalar(u).into()).collect();
                if append {
                    let x = b.input_array(Type::new(f)).into();
                    let cache = b.output_array(Type::new(f)).into();
                    let p = b.input_array(Type::new(u)).into();
                    let status = b.output_array(Type::new(u)).into();
                    append_batch::expand(
                        &mut b.scope,
                        x,
                        cache,
                        p,
                        status,
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                        args[3].clone(),
                        args[4].clone(),
                    );
                } else {
                    let p = b.output_array(Type::new(u)).into();
                    let status = b.output_array(Type::new(u)).into();
                    positions_batch::expand(
                        &mut b.scope,
                        p,
                        status,
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                        args[3].clone(),
                    );
                }
                let source = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("qsa_suffix"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        u,
                    )
                    .unwrap()
                    .to_string();
                let m = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::empty(),
                )
                .validate(&m)
                .unwrap();
                for i in 0..n {
                    assert!(source.contains(&format!("scalars_u32[{i}]")), "{source}");
                }
                assert!(source.contains("global_id.x"));
                if append {
                    assert!(source.contains("workgroup_id.y"));
                    let body = source.split("fn qsa_suffix(").nth(1).unwrap();
                    assert!(
                        body.find("if ").unwrap() < body.find("buffer_0_global[").unwrap(),
                        "{source}"
                    );
                }
            }
        }
    }
}
