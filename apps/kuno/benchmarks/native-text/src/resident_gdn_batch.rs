//! Unconnected native MAX8 adapter into the actual ResidentGdn state owner.
//! One admitted workspace is reused in stream order after all status folds.
//! No readback, synchronization, cleanup or state commit occurs in enqueue.
//! The crate-private acceptance seam requires the future WHOLE-model owner to
//! prove its completed fence, lease drop, checked I/O audit and context health.
//! Full scratch clears are deliberately conservative, not an optimization.
#![allow(dead_code)] // Future whole-model coordinator; no default selection.

use super::*;
use crate::{
    gdn_batch_pointwise as point,
    gdn_chunk::{self, ChunkBuffers, ChunkSpans, RecurrencePolicy, StateSpan, TokenSpan},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_util as u,
};
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

#[path = "strata_gdn_replay.rs"]
pub(crate) mod strata_replay;
const N: usize = 16;
const MAX_LAYERS: usize = 64;
const QKV: usize = 0;
const Z: usize = 1;
const ALPHA: usize = 2;
const BETA: usize = 3;
const ACT: usize = 4;
const Q: usize = 5;
const K: usize = 6;
const V: usize = 7;
const CONTROL: usize = 8;
const RECURRENT: usize = 9;
const GATED: usize = 10;
const STATUS: usize = 11;
pub(crate) const INFO_BYTES_PER_STAGE: u64 = 64 * 2 * 512
    + if cfg!(feature = "strata-intel-core") {
        crate::strata_gdn::prompt::COMPATIBILITY_INFO_PER_STAGE
    } else {
        0
    };

#[derive(Clone, Copy, PartialEq)]
struct Layout {
    stage: StageLayout,
    widths: [usize; N],
    words: [usize; N],
}
impl Layout {
    fn new(stage: StageLayout) -> Result<Self> {
        let p = stage.pointwise;
        if p != PointwiseLayout::new(p.shape).map_err(fault)?
            || stage.recurrence != GdnLayout::new(p.shape).map_err(fault)?
            || p.shape.head_dim > gdn_chunk::MAX_HEAD_DIM
            || !stage.epsilon.is_finite()
            || stage.epsilon <= 0.
        {
            return Err(fault("native GDN batch geometry"));
        }
        let widths = [
            p.channels,
            p.value_elements,
            p.shape.value_heads,
            p.shape.value_heads,
            p.channels,
            p.qk_elements,
            p.qk_elements,
            p.value_elements,
            p.control_elements,
            p.value_elements,
            p.value_elements,
            p.channels,
            p.channels,
            p.shape.value_heads,
            p.value_elements,
            p.value_elements,
        ];
        let mut words = [0; N];
        for (i, width) in widths.into_iter().enumerate() {
            words[i] = width
                .checked_mul(MAX_BATCH)
                .ok_or(Error::ArithmeticOverflow)?;
            BufferLayout::new(words[i]).map_err(arena_fault)?;
            validate_view(MAX_BATCH, 0, width, width, words[i])?;
        }
        Ok(Self {
            stage,
            widths,
            words,
        })
    }
    fn requests(self) -> Result<[u64; N]> {
        let mut requests = [0; N];
        for (i, words) in self.words.into_iter().enumerate() {
            requests[i] = BufferLayout::new(words).map_err(arena_fault)?.owned_bytes as u64;
        }
        Ok(requests)
    }
    fn bytes(self) -> Result<u64> {
        self.requests()?.into_iter().try_fold(0u64, |s, n| {
            s.checked_add(n).ok_or(Error::ArithmeticOverflow)
        })
    }
    fn token(self, i: usize) -> TokenSpan {
        TokenSpan {
            base: 0,
            stride: self.widths[i],
            words: self.words[i],
        }
    }
    fn chunk(self, count: usize) -> Result<ChunkSpans> {
        let shape = self.stage.pointwise.shape;
        let state = StateSpan {
            base: 0,
            head_stride: shape
                .head_dim
                .checked_mul(shape.head_dim)
                .ok_or(Error::ArithmeticOverflow)?,
            key_stride: shape.head_dim,
            column_stride: 1,
            words: self.stage.recurrence.state_elements,
        };
        let spans = ChunkSpans {
            count,
            query: self.token(Q),
            key: self.token(K),
            value: self.token(V),
            controls: self.token(CONTROL),
            old: state,
            next: state,
            output: self.token(RECURRENT),
            status: self.token(STATUS + 3),
        };
        spans.validate(self.stage.recurrence)?;
        Ok(spans)
    }
}

fn graph_plan(graph: BoundGraph<'_, '_>) -> Result<(Layout, u64)> {
    graph.verify_unchanged()?;
    let layers = graph.geometry().layers;
    if layers == 0 || layers > MAX_LAYERS {
        return Err(fault("native GDN batch layer bound"));
    }
    let expected = expected_stages(layers, |layer| graph.layer_kind(layer))?;
    let mut layout = None;
    for layer in 0..layers {
        if expected & (1u64 << layer) != 0 {
            let candidate = Layout::new(StageLayout::from_graph(graph, layer)?)?;
            if layout.is_some_and(|l| l != candidate) {
                return Err(fault("native GDN batch heterogeneous geometry"));
            }
            layout = Some(candidate);
        }
    }
    Ok((
        layout.ok_or(fault("native GDN batch missing stages"))?,
        expected,
    ))
}

fn expected_stages(
    layers: usize,
    mut kind: impl FnMut(usize) -> Result<AttentionKind>,
) -> Result<u64> {
    if layers == 0 || layers > MAX_LAYERS {
        return Err(Error::InvalidBufferLength);
    }
    let mut expected = 0;
    for layer in 0..layers {
        if kind(layer)? == AttentionKind::Gdn {
            expected |= 1u64 << layer;
        }
    }
    if expected == 0 {
        return Err(fault("native GDN batch missing stages"));
    }
    Ok(expected)
}

fn guarded<T>(poison: &AtomicBool, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(Ok(result)) => Ok(result),
        _ => {
            poison.store(true, Ordering::Release);
            Err(fault("native GDN batch guarded failure"))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Stamp {
    workspace: u64,
    chunk: u64,
    generation: u64,
    stage: u64,
    sequence: u64,
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
    scratch: [BufferId; N],
    state: [BufferId; 4],
    input: BufferId,
    output: BufferId,
    fault: BufferId,
}
impl Zeroize for Metadata {
    fn zeroize(&mut self) {
        self.stamp.zeroize();
        self.scratch.zeroize();
        self.state.zeroize();
        self.input.zeroize();
        self.output.zeroize();
        self.fault.zeroize();
    }
}
pub(super) struct StageAnchor(Zeroizing<Metadata>);
fn anchor_matches(expected: &Metadata, received: &Metadata) -> bool {
    expected == received
}

/// Opaque enqueue receipt. Drop before a successful commit poisons the whole
/// sequence; fixed-capacity metadata is wiped even for a consumed receipt.
pub(crate) struct PendingGdnBatch {
    metadata: Zeroizing<Metadata>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for PendingGdnBatch {
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
            || (self.expected & !*self.issued).trailing_zeros() as usize != layer
        {
            return Err(fault("native GDN batch mixed or duplicate stage"));
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
            return Err(fault("native GDN batch stale stage receipt"));
        }
        Ok(())
    }
    fn accept(&mut self, seen: u64) -> Result<()> {
        if self.phase != ChunkPhase::Queuing
            || seen != self.expected
            || *self.issued != self.expected
        {
            return Err(fault("native GDN batch incomplete preflight"));
        }
        self.phase = ChunkPhase::Accepted;
        Ok(())
    }
    fn commit(&mut self, stamp: Stamp) -> Result<()> {
        self.check(stamp)?;
        if self.phase != ChunkPhase::Accepted || *self.committed & (1u64 << stamp.layer) != 0 {
            return Err(fault("native GDN batch commit without acceptance"));
        }
        *self.committed |= 1u64 << stamp.layer;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if self.phase != ChunkPhase::Accepted || *self.committed != self.expected {
            return Err(fault("native GDN batch partial state commit"));
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
    fn completed_boundary(&self) -> Result<()> {
        if self.phase != ChunkPhase::Idle
            || *self.chunk == 0
            || *self.issued != 0
            || *self.committed != 0
        {
            return Err(fault(
                "native GDN recurrence policy requires completed chunk",
            ));
        }
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
        return Err(fault("native GDN batch context extent"));
    }
    Ok(next)
}

/// Caller retains the arena and original public caches through completed
/// consumers/acceptance and explicit full-page cleanup. Allocation requests
/// must be appended to the caller's admitted MAIN plan before construction.
/// Construct once at idle sequence initialization, before ordinary inference;
/// the existing arena zero initializer is a checked initialization boundary.
pub(crate) struct GdnBatchWorkspace {
    binding: GraphBindingId,
    sequence: SequenceKey,
    identity: u64,
    layout: Layout,
    ids: Zeroizing<[BufferId; N]>,
    fault: Option<BufferId>,
    poison: Arc<AtomicBool>,
    policy: Policy,
    recurrence_policy: RecurrencePolicy,
    shared_vector: bool,
    pub(crate) shared_vector_calls: u64,
    public_grouped_status: crate::status_fold_batch::PublicPolicy,
}
/// Records a genuine large prompt boundary separately from MAX8 receipts.
pub(crate) struct PromptBoundary<'a> {
    workspace: &'a mut GdnBatchWorkspace,
    next: u64,
}
impl PromptBoundary<'_> {
    pub(crate) fn commit(self) {
        *self.workspace.policy.chunk = self.next;
    }
}
impl GdnBatchWorkspace {
    pub(crate) fn prepare_prompt_boundary<'a>(
        &'a mut self,
        arena: &PrivateArena,
        completed: &crate::resident_model::batch::prompt_full::Completed,
        cancel: &AtomicBool,
    ) -> Result<PromptBoundary<'a>> {
        self.healthy(arena, cancel)?;
        completed.check(self.binding, &self.poison)?;
        if self.policy.phase != ChunkPhase::Idle
            || *self.policy.issued != 0
            || *self.policy.committed != 0
            || self.fault.is_some()
        {
            return Err(Error::InvalidFormat("large prompt idle workspace boundary"));
        }
        let next = self
            .policy
            .chunk
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(PromptBoundary {
            workspace: self,
            next,
        })
    }

    pub(crate) fn public_grouped_status_folds(&self) -> (bool, u64) {
        self.public_grouped_status.state()
    }
    pub(crate) fn validate_public_grouped_status_selection(
        &self,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.healthy(arena, cancel)?;
        self.policy.completed_boundary()?;
        self.public_grouped_status.validate_selection()
    }
    pub(crate) fn enable_public_grouped_status_folds(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) {
        self.public_grouped_status.enable();
    }

    pub(crate) fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; N]> {
        graph_plan(graph)?.0.requests()
    }
    /// GDN portion only; future whole-graph admission must add other operators,
    /// uploads/maps and already-retained runtime spans under the same hard cap.
    pub(crate) fn runtime_info_reservation(
        graph: BoundGraph<'_, '_>,
        audited_duplicate: bool,
    ) -> Result<u64> {
        let stages = graph_plan(graph)?.1.count_ones() as u64;
        stages
            .checked_mul(INFO_BYTES_PER_STAGE)
            .and_then(|n| n.checked_mul(if audited_duplicate { 2 } else { 1 }))
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(crate) fn allocate(
        graph: BoundGraph<'_, '_>,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        poison: Arc<AtomicBool>,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        guarded(&poison, || {
            cancelled(cancel)?;
            if poison.load(Ordering::Acquire) {
                return Err(fault("native GDN sequence poisoned"));
            }
            let (layout, expected) = graph_plan(graph)?;
            let next = (arena.charged_bytes() as u64)
                .checked_add(layout.bytes()?)
                .ok_or(Error::ArithmeticOverflow)?;
            if next > arena.budget_bytes() as u64 {
                return Err(Error::LimitExceeded("native GDN batch MAIN admission"));
            }
            arena.client().map_err(arena_fault)?;
            let ids = allocate_ids(layout.words, cancel, |words| {
                arena.allocate_zeroed(words).map_err(arena_fault)
            })?;
            let workspace = Self {
                binding: graph.binding_id(),
                sequence,
                identity: identity()?,
                layout,
                ids,
                fault: None,
                poison: poison.clone(),
                policy: Policy::new(expected),
                recurrence_policy: RecurrencePolicy::Chunk,
                shared_vector: false,
                shared_vector_calls: 0,
                public_grouped_status: crate::status_fold_batch::PublicPolicy::default(),
            };
            workspace.validate_ids(arena)?;
            cancelled(cancel)?;
            Ok(workspace)
        })
    }
    pub(crate) fn recurrence_policy(&self) -> RecurrencePolicy {
        self.recurrence_policy
    }
    /// Public lab policy only, after the enclosing model proved its completed
    /// boundary. This performs no enqueue, map, fence or allocation and keeps
    /// every MAX8 scratch/state owner and metadata reservation unchanged.
    pub(crate) fn set_lab_recurrence_policy(
        &mut self,
        executor: &crate::executor::GpuExecutor<'_>,
        arena: &PrivateArena,
        policy: RecurrencePolicy,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
        cancel: &AtomicBool,
    ) -> Result<RecurrencePolicy> {
        self.healthy(arena, cancel)?;
        self.policy.completed_boundary()?;
        if !arena.public_owned_zero_audit_enabled() {
            return Err(fault("native GDN recurrence policy requires full audit"));
        }
        executor.check_client(arena.client().map_err(arena_fault)?)?;
        self.recurrence_policy = policy;
        self.shared_vector = executor.shared_gdn && policy == RecurrencePolicy::SingleStreaming;
        Ok(self.recurrence_policy)
    }
    fn healthy(&self, arena: &PrivateArena, cancel: &AtomicBool) -> Result<()> {
        cancelled(cancel)?;
        if self.poison.load(Ordering::Acquire) {
            return Err(fault("native GDN sequence poisoned"));
        }
        arena.client().map_err(arena_fault)?;
        self.validate_ids(arena)
    }
    fn validate_ids(&self, arena: &PrivateArena) -> Result<()> {
        for (i, id) in self.ids.iter().copied().enumerate() {
            if self.ids[..i].contains(&id)
                || arena.layout(id).map_err(arena_fault)?
                    != BufferLayout::new(self.layout.words[i]).map_err(arena_fault)?
            {
                return Err(fault("native GDN stale workspace"));
            }
        }
        Ok(())
    }
    fn check_receipt(
        &self,
        stage: &ResidentGdn<'_>,
        receipt: &PendingGdnBatch,
        arena: &PrivateArena,
    ) -> Result<()> {
        let m = &*receipt.metadata;
        if !receipt.armed
            || !Arc::ptr_eq(&receipt.poison, &self.poison)
            || m.stamp.workspace != self.identity
            || m.stamp.sequence != self.sequence.0
            || m.stamp.stage != stage.identity
            || m.stamp.layer != stage.layer
            || m.stamp.public != stage.public_identity
            || stage.sequence != self.sequence
            || stage.binding != self.binding
            || stage.layout != self.layout.stage
            || m.scratch != *self.ids
            || Some(m.fault) != self.fault
            || m.state != state_ids(stage)
            || stage
                .batch_pending
                .as_ref()
                .is_none_or(|anchor| !anchor_matches(&anchor.0, m))
            || stage.lifecycle.phase != Phase::UncommittedBatch(m.stamp.start, m.stamp.count)
            || stage
                .batch_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
        {
            return Err(fault("native GDN batch foreign receipt"));
        }
        self.policy.check(m.stamp)?;
        checked_count(m.stamp.start, m.stamp.count, stage.layout.max_context)?;
        for id in m
            .state
            .into_iter()
            .chain(m.scratch)
            .chain([m.input, m.output, m.fault])
        {
            arena.layout(id).map_err(arena_fault)?;
        }
        Ok(())
    }
    /// OWNER PRECONDITION: the WHOLE-model chunk (including QSA/PLE/MoE/GR and
    /// final residual/head) completed its exact output/global-fault fence;
    /// every readback lease is dropped, checked runtime I/O audit succeeded,
    /// context health and cancellation are accepted. Caller also preflights
    /// other state owners before ANY model swap. This unconnected seam performs
    /// no read/sync/audit and does not manufacture device-success evidence.
    /// All expected GDN stages are derived internally from the original graph.
    pub(crate) fn accept_after_whole_model_chunk_acceptance<'a>(
        &'a mut self,
        arena: &PrivateArena,
        pending: &[(&ResidentGdn<'_>, &PendingGdnBatch)],
        cancel: &AtomicBool,
    ) -> Result<GdnBatchAcceptance<'a>> {
        let attempted = (|| {
            self.healthy(arena, cancel)?;
            let mut seen = Zeroizing::new(0u64);
            for &(stage, receipt) in pending {
                self.check_receipt(stage, receipt, arena)?;
                let bit = 1u64 << stage.layer;
                if *seen & bit != 0 {
                    return Err(fault("native GDN duplicate receipt preflight"));
                }
                *seen |= bit;
            }
            self.policy.accept(*seen)?;
            cancelled(cancel)?;
            Ok(())
        })();
        if attempted.is_err() {
            self.poison.store(true, Ordering::Release);
        }
        attempted?;
        Ok(GdnBatchAcceptance {
            workspace: self,
            armed: true,
        })
    }
}
impl Drop for GdnBatchWorkspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
        if let Some(id) = &mut self.fault {
            id.zeroize();
        }
        self.sequence.0.zeroize();
        self.identity.zeroize();
    }
}
pub(crate) struct GdnBatchAcceptance<'a> {
    workspace: &'a mut GdnBatchWorkspace,
    armed: bool,
}
impl GdnBatchAcceptance<'_> {
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
impl Drop for GdnBatchAcceptance<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.workspace.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operation {
    Clear(usize),
    Projection(usize),
    Convolution,
    Preparation,
    Controls,
    Recurrence,
    Closing,
    Output,
    Status(usize),
}
// The production dispatcher and fake queue use this exact stream-order producer.
// No completion operation exists here; status folds precede the next reuse.
fn queue_stage(
    cancel: &AtomicBool,
    mut enqueue: impl FnMut(Operation) -> Result<()>,
) -> Result<()> {
    for operation in (0..N)
        .map(Operation::Clear)
        .chain((0..4).map(Operation::Projection))
        .chain([
            Operation::Convolution,
            Operation::Preparation,
            Operation::Controls,
            Operation::Recurrence,
            Operation::Closing,
            Operation::Output,
        ])
        .chain((0..5).map(Operation::Status))
    {
        cancelled(cancel)?;
        enqueue(operation)?;
    }
    cancelled(cancel)
}

fn state_ids(stage: &ResidentGdn<'_>) -> [BufferId; 4] {
    [
        stage.buffers.recurrent[0],
        stage.buffers.recurrent[1],
        stage.buffers.raw_conv[0],
        stage.buffers.raw_conv[1],
    ]
}
impl ResidentGdn<'_> {
    /// The original old/next pairs swap together on accepted commit/restore.
    /// These two static binding arrangements are independent of token position.
    pub(crate) fn native_capture_state_role(&self) -> Result<u8> {
        let recurrent = self.buffers.recurrent[0].original_pair_role(self.buffers.recurrent[1]);
        let convolution = self.buffers.raw_conv[0].original_pair_role(self.buffers.raw_conv[1]);
        match (recurrent, convolution) {
            (Some(a), Some(b)) if a == b => Ok(u8::from(a)),
            _ => Err(fault("native GDN state role alignment")),
        }
    }
}
impl Lifecycle {
    fn begin_batch(&mut self, start: u64, count: usize) -> Result<()> {
        if self.phase != Phase::Ready(start)
            || checked_count(start, count, self.max_context).is_err()
        {
            self.poison();
            return Err(fault("native GDN batch cursor/order"));
        }
        self.phase = Phase::UncommittedBatch(start, count);
        Ok(())
    }
    fn commit_batch(&mut self, start: u64, count: usize) -> Result<()> {
        if self.phase != Phase::UncommittedBatch(start, count) {
            self.poison();
            return Err(fault("native GDN batch stale state commit"));
        }
        self.phase = Phase::Ready(checked_count(start, count, self.max_context)?);
        Ok(())
    }
}
impl Drop for Lifecycle {
    fn drop(&mut self) {
        match &mut self.phase {
            Phase::Ready(position) | Phase::Uncommitted(position) => position.zeroize(),
            Phase::UncommittedBatch(position, count) => {
                position.zeroize();
                count.zeroize();
            }
            Phase::Poisoned => {}
        }
    }
}
impl Drop for ResidentGdn<'_> {
    fn drop(&mut self) {
        if self.batch_pending.is_some() {
            self.poison();
        }
    }
}

impl ResidentGdn<'_> {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_native_gdn_operation(
        &self,
        op: Operation,
        arena: &PrivateArena,
        public: &PublicGdnParameters,
        ids: &[BufferId; N],
        layout: Layout,
        start: usize,
        count: usize,
        fault_id: BufferId,
    ) -> Result<bool> {
        use crate::strata_gdn::{self as native, Kernel};
        let p = layout.stage.pointwise;
        if p.shape.head_dim != 128 {
            return Err(fault("native GDN requires original128-state graph"));
        }
        let h = |id| u::handle(arena, id);
        let info = [
            count as u32,
            start as u32,
            p.channels as u32,
            p.shape.key_heads as u32,
            p.shape.value_heads as u32,
            (public.gamma_words == 128) as u32,
            layout.stage.epsilon.to_bits(),
            0,
        ];
        if crate::strata_prompt::active() {
            let [raw, z, prepared, recurrent] = crate::strata_prompt::gdn_workspace()?;
            match op {
                Operation::Convolution => native::prompt::convolution(
                    arena,
                    info,
                    [
                        raw,
                        public.convolution.clone(),
                        h(self.buffers.raw_conv[0])?,
                        h(self.buffers.raw_conv[1])?,
                        prepared,
                        h(ids[Q])?,
                        h(ids[K])?,
                        h(ids[V])?,
                        h(ids[STATUS])?,
                        h(ids[STATUS + 1])?,
                    ],
                    h(fault_id)?,
                )?,
                Operation::Recurrence => native::prompt::recurrence(
                    arena,
                    info,
                    [
                        prepared,
                        h(ids[CONTROL])?,
                        h(self.buffers.recurrent[0])?,
                        h(self.buffers.recurrent[1])?,
                        recurrent,
                        z,
                        public.gamma.clone(),
                        h(ids[STATUS + 3])?,
                        h(ids[STATUS + 4])?,
                    ],
                    h(fault_id)?,
                )?,
                Operation::Preparation | Operation::Closing => {}
                Operation::Controls => return Err(fault("native prompt AB producer bypassed")),
                _ => return Ok(false),
            }
            return Ok(true);
        }
        match op {
            Operation::Convolution => native::launch(
                arena,
                Kernel::Conv,
                [p.channels / 128, count, 1],
                vec![
                    h(ids[QKV])?,
                    public.convolution.clone(),
                    h(self.buffers.raw_conv[0])?,
                    h(self.buffers.raw_conv[1])?,
                    h(ids[Q])?,
                    h(ids[K])?,
                    h(ids[V])?,
                    h(ids[STATUS])?,
                    h(ids[STATUS + 1])?,
                ],
                info,
            )?,
            Operation::Preparation | Operation::Closing => {}
            Operation::Controls => return Err(fault("native AB producer bypassed")),
            Operation::Recurrence => native::launch(
                arena,
                Kernel::Step,
                [p.shape.value_heads, 1, 1],
                vec![
                    h(ids[Q])?,
                    h(ids[K])?,
                    h(ids[V])?,
                    h(ids[CONTROL])?,
                    h(self.buffers.recurrent[0])?,
                    h(self.buffers.recurrent[1])?,
                    h(ids[Z])?,
                    public.gamma.clone(),
                    h(ids[GATED])?,
                    h(ids[STATUS + 3])?,
                    h(ids[STATUS + 4])?,
                ],
                info,
            )?,
            _ => return Ok(false),
        }
        Ok(true)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_batch(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicGdnParameters,
        projector: &mut impl BatchedResidentProjector,
        workspace: &mut GdnBatchWorkspace,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: u64,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<PendingGdnBatch> {
        let attempted = guarded(&workspace.poison.clone(), || {
            self.enqueue_batch_checked(
                graph, arena, public, projector, workspace, input, output, spans, start, fault_id,
                cancel,
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
        public: &PublicGdnParameters,
        projector: &mut impl BatchedResidentProjector,
        workspace: &mut GdnBatchWorkspace,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: u64,
        fault_id: BufferId,
        cancel: &AtomicBool,
    ) -> Result<PendingGdnBatch> {
        workspace.healthy(arena, cancel)?;
        workspace.recurrence_policy.validate_count(spans.count())?;
        graph.verify_unchanged()?;
        let count = spans.count();
        let layout = workspace.layout;
        if self.binding != graph.binding_id()
            || workspace.binding != self.binding
            || workspace.sequence != self.sequence
            || layout.stage != self.layout
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
            return Err(fault("native GDN batch binding/sequence/fault"));
        }
        checked_count(start, count, self.layout.max_context)?;
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
        layout.chunk(count)?;
        validate_full_regions(arena, workspace, self, public, input, output, fault_id)?;
        let gamma_shared = if public.gamma_words == self.layout.pointwise.shape.head_dim {
            true
        } else if public.gamma_words == self.layout.pointwise.value_elements {
            false
        } else {
            return Err(Error::InvalidBufferLength);
        };
        let convolution = bounded_view(
            &public.convolution,
            0,
            self.layout.pointwise.conv_weight_elements,
        )?;
        let params = bounded_view(&public.controls, 0, self.layout.pointwise.control_elements)?;
        let gamma = bounded_view(&public.gamma, 0, public.gamma_words)?;
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
                sequence: self.sequence.0,
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
            state: state_ids(self),
            input,
            output,
            fault: fault_id,
        });
        self.batch_pending = Some(StageAnchor(Zeroizing::new((*metadata).clone())));
        let ids = &workspace.ids;
        let p = self.layout.pointwise;
        let client = arena.client().map_err(arena_fault)?;
        let dim = CubeDim::new_1d(64);
        let arg = |i: usize| u::arg(arena, ids[i]);
        queue_stage(cancel, |operation| {
            if crate::strata_intel_core::enabled() {
                if operation == Operation::Controls {
                    let request = crate::strata_gdn::Ab {
                        arena,
                        input,
                        controls: ids[CONTROL],
                        status: ids[STATUS + 2],
                        fault: fault_id,
                        params: public.controls.clone(),
                        count,
                        columns: self.layout.input_words,
                        heads: p.shape.value_heads,
                        first: spans.input_first(),
                        stride: spans.input_stride(),
                    };
                    if !projector.enqueue_native_gdn_ab(
                        [self.projections.alpha, self.projections.beta],
                        request,
                        cancel,
                    )? {
                        return Err(fault("native GDN AB owner required"));
                    }
                    return Ok(());
                }
                if operation == Operation::Recurrence
                    && let Some(cache) = &mut self.strata_inputs
                {
                    cache.retain(
                        arena,
                        [ids[QKV], ids[Q], ids[K], ids[V], ids[CONTROL]],
                        if crate::strata_prompt::active() {
                            Some(crate::strata_prompt::gdn_workspace()?[0].clone())
                        } else {
                            None
                        },
                        start,
                        count,
                        chunk,
                    )?;
                }
                if self.enqueue_native_gdn_operation(
                    operation,
                    arena,
                    public,
                    ids,
                    layout,
                    start as usize,
                    count,
                    fault_id,
                )? {
                    return Ok(());
                }
            }
            match operation {
                Operation::Clear(i) => {
                    if u::grouped_scratch_clear_enabled() {
                        if i == 0 {
                            u::zero_many(arena, &**ids)?;
                        }
                    } else {
                        u::zero(arena, ids[i])?;
                    }
                }
                Operation::Projection(i) => {
                    if crate::strata_prompt::active() {
                        if i < 2 {
                            let target = crate::strata_prompt::gdn_workspace()?[i].clone();
                            let projection = [self.projections.qkv, self.projections.z][i];
                            projector.enqueue_prompt_view(
                                projection,
                                arena,
                                u::handle(arena, input)?,
                                target,
                                &BatchSpans::new(
                                    count,
                                    0,
                                    spans.input_first(),
                                    spans.input_stride(),
                                    0,
                                    [p.channels, p.value_elements][i],
                                ),
                                cancel,
                            )?;
                        }
                        return Ok(());
                    }
                    if crate::strata_intel_core::enabled() && (1..=8).contains(&count) {
                        if i == 0 {
                            let jobs = [(self.projections.qkv, QKV), (self.projections.z, Z)].map(
                                |(p, target)| {
                                    (
                                        p,
                                        ids[target],
                                        BatchSpans::new(
                                            count,
                                            0,
                                            spans.input_first(),
                                            spans.input_stride(),
                                            0,
                                            layout.widths[target],
                                        ),
                                    )
                                },
                            );
                            projector.enqueue_bundle(&jobs, arena, input, cancel)?;
                        }
                        return Ok(());
                    }
                    let (projection, target) = [
                        (self.projections.qkv, QKV),
                        (self.projections.z, Z),
                        (self.projections.alpha, ALPHA),
                        (self.projections.beta, BETA),
                    ][i];
                    projector.enqueue_batch(
                        projection,
                        arena,
                        input,
                        ids[target],
                        &BatchSpans::new(
                            count,
                            0,
                            spans.input_first(),
                            spans.input_stride(),
                            0,
                            layout.widths[target],
                        ),
                        cancel,
                    )?;
                }
                Operation::Convolution => point::conv_silu::launch::<WgpuRuntime>(
                    client,
                    u::grid(p.channels)?,
                    dim,
                    arg(QKV)?,
                    raw_f32_arg(convolution.clone(), p.conv_weight_elements),
                    u::arg(arena, self.buffers.raw_conv[0])?,
                    u::arg(arena, self.buffers.raw_conv[1])?,
                    arg(ACT)?,
                    arg(STATUS)?,
                    count,
                    0,
                    p.channels,
                    0,
                    3,
                    0,
                    3,
                    0,
                    p.channels,
                    0,
                    p.channels,
                    p.channels,
                ),
                Operation::Preparation => point::prepare_qkv::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(
                        (2 * p.shape.key_heads + p.shape.value_heads) as u32,
                        count as u32,
                        1,
                    ),
                    dim,
                    arg(ACT)?,
                    arg(Q)?,
                    arg(K)?,
                    arg(V)?,
                    arg(STATUS + 1)?,
                    self.layout.epsilon,
                    count,
                    0,
                    p.channels,
                    0,
                    p.qk_elements,
                    0,
                    p.qk_elements,
                    0,
                    p.value_elements,
                    0,
                    p.channels,
                    p.shape.head_dim,
                    p.shape.key_heads,
                    p.shape.value_heads,
                ),
                Operation::Controls => point::prepare_controls::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(p.shape.value_heads.div_ceil(64) as u32, count as u32, 1),
                    dim,
                    arg(ALPHA)?,
                    arg(BETA)?,
                    raw_f32_arg(params.clone(), p.control_elements),
                    arg(CONTROL)?,
                    arg(STATUS + 2)?,
                    count,
                    0,
                    p.shape.value_heads,
                    0,
                    p.shape.value_heads,
                    0,
                    p.control_elements,
                    0,
                    p.shape.value_heads,
                    p.shape.value_heads,
                ),
                Operation::Recurrence => {
                    if let Some(cache) = &mut self.strata_inputs {
                        cache.retain(
                            arena,
                            [ids[QKV], ids[Q], ids[K], ids[V], ids[CONTROL]],
                            None,
                            start,
                            count,
                            chunk,
                        )?;
                    }
                    if workspace.shared_vector {
                        gdn_chunk::enqueue_shared_count1(
                            arena,
                            p.shape,
                            layout.chunk(count)?,
                            &ChunkBuffers([
                                ids[Q],
                                ids[K],
                                ids[V],
                                ids[CONTROL],
                                self.buffers.recurrent[0],
                                self.buffers.recurrent[1],
                                ids[RECURRENT],
                                ids[STATUS + 3],
                            ]),
                            cancel,
                        )?;
                        workspace.shared_vector_calls = workspace
                            .shared_vector_calls
                            .checked_add(1)
                            .ok_or(Error::ArithmeticOverflow)?;
                    } else {
                        gdn_chunk::enqueue_with_policy(
                            arena,
                            p.shape,
                            layout.chunk(count)?,
                            &ChunkBuffers([
                                ids[Q],
                                ids[K],
                                ids[V],
                                ids[CONTROL],
                                self.buffers.recurrent[0],
                                self.buffers.recurrent[1],
                                ids[RECURRENT],
                                ids[STATUS + 3],
                            ]),
                            cancel,
                            workspace.recurrence_policy,
                        )?
                    }
                }
                Operation::Closing => point::output_gate::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(p.shape.value_heads as u32, count as u32, 1),
                    dim,
                    arg(RECURRENT)?,
                    arg(Z)?,
                    raw_f32_arg(gamma.clone(), public.gamma_words),
                    arg(GATED)?,
                    arg(STATUS + 4)?,
                    self.layout.epsilon,
                    count,
                    0,
                    p.value_elements,
                    0,
                    p.value_elements,
                    0,
                    p.value_elements,
                    0,
                    p.value_elements,
                    p.shape.head_dim,
                    p.shape.value_heads,
                    gamma_shared,
                ),
                Operation::Output => {
                    let out_spans = BatchSpans::new(
                        count,
                        0,
                        0,
                        p.value_elements,
                        spans.output_first(),
                        spans.output_stride(),
                    );
                    if crate::strata_prompt::active() {
                        projector.enqueue_prompt_prepacked(
                            self.projections.output,
                            arena,
                            crate::strata_prompt::activation_stage()?,
                            output,
                            &out_spans,
                            cancel,
                        )?;
                    } else {
                        projector.enqueue_batch(
                            self.projections.output,
                            arena,
                            ids[GATED],
                            output,
                            &out_spans,
                            cancel,
                        )?;
                    }
                }
                Operation::Status(i) => {
                    if workspace.public_grouped_status.state().0
                        || crate::status_fold_batch::selected()
                    {
                        if i == 0 {
                            let views: [crate::status_fold_batch::View; 5] =
                                std::array::from_fn(|j| {
                                    crate::status_fold_batch::View::new(
                                        ids[STATUS + j],
                                        count,
                                        0,
                                        layout.widths[STATUS + j],
                                        layout.widths[STATUS + j],
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
                        u::status_batch(
                            arena,
                            ids[STATUS + i],
                            fault_id,
                            count,
                            0,
                            layout.widths[STATUS + i],
                            layout.widths[STATUS + i],
                        )?;
                    }
                }
            }
            Ok(())
        })?;
        Ok(PendingGdnBatch {
            metadata,
            poison: workspace.poison.clone(),
            armed: true,
        })
    }
    /// All GDN receipts were preflighted by this sealed permit BEFORE any state
    /// swap. Other model owners still require their whole-model preflight.
    pub(crate) fn commit_after_whole_chunk_acceptance(
        &mut self,
        mut receipt: PendingGdnBatch,
        arena: &PrivateArena,
        acceptance: &mut GdnBatchAcceptance<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let attempted = (|| {
            acceptance.workspace.healthy(arena, cancel)?;
            acceptance.workspace.check_receipt(self, &receipt, arena)?;
            let stamp = receipt.metadata.stamp;
            // All fallible receipt/extent checks precede the actual swap.
            checked_count(stamp.start, stamp.count, self.layout.max_context)?;
            acceptance.workspace.policy.commit(stamp)?;
            self.lifecycle.commit_batch(stamp.start, stamp.count)?;
            self.buffers.recurrent.swap(0, 1);
            self.buffers.raw_conv.swap(0, 1);
            self.batch_pending = None;
            receipt.armed = false;
            cancelled(cancel)?;
            Ok(())
        })();
        if attempted.is_err() {
            self.poison();
            acceptance.workspace.poison.store(true, Ordering::Release);
        }
        attempted
    }
}

/// Actual prechunk roots retained by the ordinary old/next state swap. The
/// external exclusive round allows exactly one completed chunk before consume.
pub(crate) struct LatestGdnCheckpoint {
    metadata: Zeroizing<LatestGdnMetadata>,
    binding: Option<GraphBindingId>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
#[derive(Clone, PartialEq, Eq)]
struct LatestGdnMetadata {
    workspace: u64,
    stage: u64,
    public: u64,
    sequence: u64,
    layer: usize,
    start: u64,
    count: usize,
    chunk: u64,
    generation: u64,
    state: [BufferId; 4],
}
impl Zeroize for LatestGdnMetadata {
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
        self.state.zeroize();
    }
}
fn latest_gdn_order(old: &LatestGdnMetadata, next: &LatestGdnMetadata, stages: u32) -> Result<()> {
    let mut expected = Zeroizing::new(old.clone());
    expected.chunk = old.chunk.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
    expected.generation = old
        .generation
        .checked_add(stages as u64)
        .ok_or(Error::ArithmeticOverflow)?;
    expected.state.swap(0, 1);
    expected.state.swap(2, 3);
    if *expected != *next {
        return Err(fault("native latest GDN checkpoint identity"));
    }
    Ok(())
}
fn latest_gdn_roots(stage: &ResidentGdn<'_>, arena: &PrivateArena) -> Result<()> {
    let ids = state_ids(stage);
    let sizes = [stage.layout.recurrence.state_elements; 2]
        .into_iter()
        .chain([stage.layout.pointwise.ring_elements; 2]);
    let mut handles = Vec::with_capacity(4);
    for (i, (id, words)) in ids.into_iter().zip(sizes).enumerate() {
        if ids[..i].contains(&id) {
            return Err(fault("native latest GDN root alias"));
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
            return Err(fault("native latest GDN full roots"));
        }
        handles.push(handle.clone());
    }
    let resources = u::resources(arena.client().map_err(arena_fault)?, handles)?;
    if resources.len() != 4 {
        return Err(Error::InvalidBufferLength);
    }
    for (i, r) in resources.iter().enumerate() {
        let page = r.resource();
        let bytes = arena.layout(ids[i]).map_err(arena_fault)?.owned_bytes as u64;
        validate_page(page.offset, page.size, page.buffer.size(), bytes, bytes)?;
        if resources[..i]
            .iter()
            .any(|old| old.resource().buffer == page.buffer)
        {
            return Err(fault("native latest GDN backing alias"));
        }
    }
    Ok(())
}
impl GdnBatchWorkspace {
    pub(crate) fn checkpoint_latest(
        &self,
        stage: &ResidentGdn<'_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<LatestGdnCheckpoint> {
        self.checkpoint_latest_with_four_rows(stage, arena, start, count, false, cancel)
    }
    pub(crate) fn checkpoint_latest_with_four_rows(
        &self,
        stage: &ResidentGdn<'_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        four: bool,
        cancel: &AtomicBool,
    ) -> Result<LatestGdnCheckpoint> {
        self.healthy(arena, cancel)?;
        checked_count(start, count, stage.layout.max_context)?;
        if count >= 4 && self.recurrence_policy != RecurrencePolicy::FourStreaming
            || !(1..=if four {
                crate::strata_port::verify_capacity()
            } else {
                3
            })
                .contains(&count)
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
            return Err(fault("native latest GDN checkpoint boundary"));
        }
        latest_gdn_roots(stage, arena)?;
        self.policy
            .chunk
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        self.policy
            .generation
            .checked_add(self.policy.expected.count_ones() as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(LatestGdnCheckpoint {
            metadata: Zeroizing::new(LatestGdnMetadata {
                workspace: self.identity,
                stage: stage.identity,
                public: stage.public_identity,
                sequence: stage.sequence.0,
                layer: stage.layer,
                start,
                count,
                chunk: *self.policy.chunk,
                generation: *self.policy.generation,
                state: state_ids(stage),
            }),
            binding: Some(self.binding),
            poison: self.poison.clone(),
            armed: true,
        })
    }
}
impl LatestGdnCheckpoint {
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
        stage: &ResidentGdn<'_>,
        workspace: &GdnBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        workspace.healthy(arena, cancel)?;
        workspace.policy.completed_boundary()?;
        if !self.armed
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
            return Err(fault("native latest GDN completed checkpoint"));
        }
        latest_gdn_order(
            &self.metadata,
            &LatestGdnMetadata {
                workspace: workspace.identity,
                stage: stage.identity,
                public: stage.public_identity,
                sequence: stage.sequence.0,
                layer: stage.layer,
                start: self.metadata.start,
                count: self.metadata.count,
                chunk: *workspace.policy.chunk,
                generation: *workspace.policy.generation,
                state: state_ids(stage),
            },
            workspace.policy.expected.count_ones(),
        )?;
        latest_gdn_roots(stage, arena)
    }
    pub(crate) fn prepare_restore<'a, 'f>(
        self,
        stage: &'a mut ResidentGdn<'f>,
        workspace: &GdnBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<LatestGdnRestore<'a, 'f>> {
        self.validate_completed(stage, workspace, arena, cancel)?;
        Ok(LatestGdnRestore {
            checkpoint: self,
            stage,
        })
    }
    pub(crate) fn accept_completed(
        mut self,
        stage: &ResidentGdn<'_>,
        workspace: &GdnBatchWorkspace,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_completed(stage, workspace, arena, cancel)?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for LatestGdnCheckpoint {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
        self.binding = None;
    }
}
pub(crate) struct LatestGdnRestore<'a, 'f> {
    checkpoint: LatestGdnCheckpoint,
    stage: &'a mut ResidentGdn<'f>,
}
impl LatestGdnRestore<'_, '_> {
    /// Requires every family preflight and completed suffix clear first. There
    /// is no fallible work here and no poison/generation reset.
    pub(crate) fn commit(mut self) {
        self.stage.buffers.recurrent.swap(0, 1);
        self.stage.buffers.raw_conv.swap(0, 1);
        self.stage.lifecycle.phase = Phase::Ready(self.checkpoint.metadata.start);
        self.checkpoint.armed = false;
    }
}

/// Sealed successful-boundary cursor permit for the fixed thirteen-row public
/// warm replay. BOTH state/ring root pairs are erased by the model's complete
/// arena audit before commit; pointer order and monotonic stamps are retained.
pub(crate) struct PublicWarmGdnReset<'a, 'f> {
    stage: &'a mut ResidentGdn<'f>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl GdnBatchWorkspace {
    pub(crate) fn prepare_public_warm_reset<'a, 'f>(
        &self,
        stage: &'a mut ResidentGdn<'f>,
        arena: &PrivateArena,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PublicWarmGdnReset<'a, 'f>> {
        self.healthy(arena, cancel)?;
        self.policy.completed_boundary()?;
        if position != 13
            || *self.policy.start != 0
            || *self.policy.count != 0
            || self.policy.generations.iter().any(|&v| v != 0)
            || self.policy.stage_ids.iter().any(|&v| v != 0)
            || self.fault.is_some()
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
        {
            return Err(fault("native public warm GDN boundary"));
        }
        latest_gdn_roots(stage, arena)?;
        Ok(PublicWarmGdnReset {
            stage,
            poison: self.poison.clone(),
            armed: false,
        })
    }
}
impl PublicWarmGdnReset<'_, '_> {
    pub(crate) fn arm(&mut self) {
        self.armed = true;
    }
    /// All families/complete private pages and runtime I/O must be accepted
    /// first. The model makes the remaining commits infallible as one group.
    pub(crate) fn commit(
        mut self,
        _proof: &crate::resident_model::batch::public_warm_replay::CompletedReset,
    ) {
        self.stage.lifecycle.phase = Phase::Ready(0);
        self.armed = false;
    }
}
impl Drop for PublicWarmGdnReset<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
    }
}

/// Fixed owning handles/resources only; no allocation of private metadata.
/// Original public vectors expose logical tail-trimmed views of precharged
/// full pages. Lookup restores their FULL owning clone, while launch views stay
/// logical and immutable. Every public/private page must equal its actual span.
fn validate_full_regions(
    arena: &PrivateArena,
    workspace: &GdnBatchWorkspace,
    stage: &ResidentGdn<'_>,
    public: &PublicGdnParameters,
    input: BufferId,
    output: BufferId,
    fault_id: BufferId,
) -> Result<()> {
    const PRIVATE: usize = N + 18 + 3;
    const TOTAL: usize = PRIVATE + 3;
    let mut ids = Zeroizing::new([input; PRIVATE]);
    ids[..N].copy_from_slice(&*workspace.ids);
    ids[N..N + 18].copy_from_slice(&stage.buffers.ids());
    ids[N + 18..].copy_from_slice(&[input, output, fault_id]);
    for (i, &id) in ids.iter().enumerate() {
        if ids[..i].contains(&id) {
            return Err(fault("native GDN batch owning alias"));
        }
        let l = arena.layout(id).map_err(arena_fault)?;
        if i < N && l != BufferLayout::new(workspace.layout.words[i]).map_err(arena_fault)?
            || (N..N + 18).contains(&i)
                && l != BufferLayout::new(stage.layout.allocation_words()[i - N])
                    .map_err(arena_fault)?
        {
            return Err(fault("native GDN batch owning geometry"));
        }
    }
    let client = arena.client().map_err(arena_fault)?;
    let stream = arena.handle(ids[0]).map_err(arena_fault)?.stream;
    let mut resources: [Option<_>; TOTAL] = std::array::from_fn(|_| None);
    let mut full_handles = Vec::with_capacity(TOTAL);
    let mut bytes = Zeroizing::new([0u64; TOTAL]);
    for i in 0..TOTAL {
        let full = if i < PRIVATE {
            let h = arena.handle(ids[i]).map_err(arena_fault)?;
            let owned = arena.layout(ids[i]).map_err(arena_fault)?.owned_bytes as u64;
            if h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.size() != owned
                || h.size_in_used() != owned
            {
                return Err(fault("native GDN batch handle/page disagreement"));
            }
            h.clone()
        } else {
            let (h, words) = [
                (
                    &public.convolution,
                    stage.layout.pointwise.conv_weight_elements,
                ),
                (&public.controls, stage.layout.pointwise.control_elements),
                (&public.gamma, public.gamma_words),
            ][i - PRIVATE];
            full_public_view(h, words)?
        };
        if full.stream != stream {
            return Err(fault("native GDN batch cross-stream owner"));
        }
        bytes[i] = full.size();
        full_handles.push(full);
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
        validate_page(
            page.offset,
            page.size,
            page.buffer.size(),
            bytes[i],
            bytes[i],
        )?;
        if resources[..i]
            .iter()
            .flatten()
            .any(|r| r.resource().buffer == page.buffer)
        {
            return Err(fault("native GDN batch full backing/alias"));
        }
    }
    Ok(())
}

fn full_public_view(view: &Handle, words: usize) -> Result<Handle> {
    let page = BufferLayout::new(words).map_err(arena_fault)?.owned_bytes as u64;
    let logical = (words as u64)
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    if view.offset_start.unwrap_or(0) != 0
        || view.size() != page
        || view.size_in_used() != logical
        || view.offset_end.unwrap_or(0)
            != page
                .checked_sub(logical)
                .ok_or(Error::InvalidBufferLength)?
    {
        return Err(fault("native GDN original public logical/full page"));
    }
    let mut full = view.clone();
    full.offset_end = None;
    if full.size_in_used() != page {
        return Err(Error::InvalidBufferLength);
    }
    Ok(full)
}

fn validate_page(offset: u64, size: u64, physical: u64, owned: u64, bound: u64) -> Result<()> {
    if owned == 0 || offset != 0 || size != owned || physical < owned || physical > bound {
        return Err(fault("native GDN batch full physical page"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn latest() -> LatestGdnMetadata {
        LatestGdnMetadata {
            workspace: 7,
            stage: 11,
            public: 13,
            sequence: 17,
            layer: 0,
            start: 23,
            count: 3,
            chunk: 29,
            generation: 100,
            state: std::array::from_fn(|i| BufferId::metadata_fixture(31, 37, i)),
        }
    }
    #[test]
    fn latest_gdn_requires_one_real_swap_of_both_root_pairs_and_monotonic_generation() {
        let old = latest();
        let mut completed = old.clone();
        completed.chunk += 1;
        completed.generation += 36;
        completed.state.swap(0, 1);
        completed.state.swap(2, 3);
        latest_gdn_order(&old, &completed, 36).unwrap();
        let mut wrong = completed.clone();
        wrong.state.swap(2, 3);
        assert!(latest_gdn_order(&old, &wrong, 36).is_err());
        for field in 0..10 {
            let mut wrong = completed.clone();
            match field {
                0 => wrong.workspace += 1,
                1 => wrong.stage += 1,
                2 => wrong.public += 1,
                3 => wrong.sequence += 1,
                4 => wrong.layer += 1,
                5 => wrong.start += 1,
                6 => wrong.count -= 1,
                7 => wrong.chunk += 1,
                8 => wrong.generation -= 1,
                _ => wrong.state[1] = BufferId::metadata_fixture(31, 38, 0),
            }
            assert!(latest_gdn_order(&old, &wrong, 36).is_err());
        }
        assert!(latest_gdn_order(&old, &old, 36).is_err());
        let mut overflow = old.clone();
        overflow.generation = u64::MAX;
        assert!(latest_gdn_order(&overflow, &completed, 36).is_err());
    }
    #[test]
    fn latest_gdn_abandoned_and_consumed_tickets_do_not_reset_poison() {
        for armed in [false, true] {
            let poison = Arc::new(AtomicBool::new(false));
            drop(LatestGdnCheckpoint {
                metadata: Zeroizing::new(latest()),
                binding: None,
                poison: poison.clone(),
                armed,
            });
            assert_eq!(poison.load(Ordering::Acquire), armed);
        }
        let poison = Arc::new(AtomicBool::new(true));
        drop(LatestGdnCheckpoint {
            metadata: Zeroizing::new(latest()),
            binding: None,
            poison: poison.clone(),
            armed: false,
        });
        assert!(poison.load(Ordering::Acquire));
    }
    use kuno_text::model::PleGeometry;
    use kuno_text::ops::{GatedResidualShape, GdnShape, QsaShape};

    fn layout() -> Layout {
        Layout::new(
            StageLayout::from_geometry(GraphGeometry {
                layers: 48,
                residual: GatedResidualShape {
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
                gdn: GdnShape::qwen38(),
                gdn_conv_channels: 10240,
                gdn_conv_kernel: 4,
                gdn_conv_dilation: 1,
                qsa: QsaShape::qwen38(),
                ple: PleGeometry {
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
            })
            .unwrap(),
        )
        .unwrap()
    }
    fn stamp(layer: usize, stage: u64, chunk: u64, generation: u64, count: usize) -> Stamp {
        Stamp {
            workspace: 77,
            chunk,
            generation,
            stage,
            sequence: 23,
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
            state: std::array::from_fn(|i| BufferId::metadata_fixture(7, 13, 16 + i)),
            input: BufferId::metadata_fixture(7, 13, 20),
            output: BufferId::metadata_fixture(7, 13, 21),
            fault: BufferId::metadata_fixture(7, 13, 22),
        }
    }
    #[test]
    fn exact_full_workspace_pages_and_reused_state_budget() {
        let l = layout();
        assert_eq!(
            l.requests().unwrap(),
            [
                524288, 262144, 32768, 32768, 524288, 65536, 65536, 262144, 32768, 262144, 262144,
                524288, 524288, 32768, 262144, 262144
            ]
        );
        assert_eq!(l.bytes().unwrap(), 3_932_160);
        assert_eq!(
            l.stage.allocation_words()[..4]
                .iter()
                .map(|&n| BufferLayout::new(n).unwrap().owned_bytes)
                .sum::<usize>(),
            8_650_752
        );
        assert_eq!(785_186_816 + l.bytes().unwrap(), 789_118_976);
        assert_eq!(
            BufferLayout::new(8 * 2560).unwrap().owned_bytes * 2,
            262_144
        );
        for count in [1, 3, 5, 8] {
            let s = l.chunk(count).unwrap();
            assert_eq!(s.query.words, 16_384);
            assert_eq!(s.old.words, 786_432);
            assert_eq!(s.old.head_stride, 16_384);
            assert_eq!(s.old.key_stride, 128);
            assert_eq!(s.old.column_stride, 1);
        }
    }
    #[test]
    fn expected_stage_mask_is_derived_from_bound_layer_kinds() {
        let mask = expected_stages(48, |i| {
            Ok(if i % 4 == 3 {
                AttentionKind::Qsa
            } else {
                AttentionKind::Gdn
            })
        })
        .unwrap();
        assert_eq!(mask.count_ones(), 36);
        for i in 0..48 {
            assert_eq!(mask & (1 << i) != 0, i % 4 != 3);
        }
        for n in [0, 65] {
            assert!(expected_stages(n, |_| Ok(AttentionKind::Gdn)).is_err());
        }
        assert!(expected_stages(48, |_| Ok(AttentionKind::Qsa)).is_err());
        assert!(
            expected_stages(48, |i| if i == 3 {
                Err(Error::OutOfBounds)
            } else {
                Ok(AttentionKind::Gdn)
            })
            .is_err()
        );
    }
    #[test]
    fn full_eight_extent_is_required_even_for_one_active_token() {
        for count in [1, 3, 5, 8] {
            assert_eq!(
                checked_count(4096 - count as u64, count, 4096).unwrap(),
                4096
            );
            assert!(checked_count(4097 - count as u64, count, 4096).is_err());
            let first = 7;
            let stride = 2573;
            let width = 2560;
            let words = first + 7 * stride + width;
            assert!(validate_view(MAX_BATCH, first, stride, width, words).is_ok());
            assert!(validate_view(MAX_BATCH, first, stride, width, words - 1).is_err());
            assert!(validate_view(MAX_BATCH, first, width - 1, width, words).is_err());
        }
        for count in [0, 9, usize::MAX] {
            assert!(checked_count(0, count, 4096).is_err());
        }
        assert!(checked_count(u64::MAX, 1, 4096).is_err());
    }
    #[test]
    fn prior_stage_generation_survives_later_workspace_reuse_and_no_early_commit() {
        for count in [1, 3, 5, 8] {
            let mut p = Policy::new(0b111);
            let mut stamps = Vec::new();
            for i in 0..3 {
                let (c, g) = p.issue(i, 100 + i as u64, 0, count, 4096).unwrap();
                stamps.push(stamp(i, 100 + i as u64, c, g, count));
                for &s in &stamps {
                    p.check(s).unwrap();
                }
                assert!(p.commit(stamps[0]).is_err());
                assert!(p.accept((1 << (i + 1)) - 1).is_err() || i == 2);
            }
            // The final attempted accept above succeeded only with every stage.
            assert!(p.phase == ChunkPhase::Accepted);
            for &s in &stamps {
                p.commit(s).unwrap();
            }
            assert!(p.commit(stamps[0]).is_err());
            p.finish().unwrap();
            let (c, g) = p.issue(0, 100, 0, count, 4096).unwrap();
            assert!(c > stamps[0].chunk && g > stamps[2].generation);
            assert!(p.check(stamps[0]).is_err());
        }
    }
    #[test]
    fn public_recurrence_policy_requires_a_completed_idle_chunk() {
        let mut p = Policy::new(1);
        assert!(p.completed_boundary().is_err());
        let (chunk, generation) = p.issue(0, 100, 0, 1, 4096).unwrap();
        let s = stamp(0, 100, chunk, generation, 1);
        assert!(p.completed_boundary().is_err());
        p.accept(1).unwrap();
        assert!(p.completed_boundary().is_err());
        p.commit(s).unwrap();
        assert!(p.completed_boundary().is_err());
        p.finish().unwrap();
        p.completed_boundary().unwrap();
    }
    #[test]
    fn duplicate_mixed_missing_and_overflowed_chunk_policy_reject() {
        let mut p = Policy::new(0b101);
        assert!(p.issue(2, 9, 0, 3, 4096).is_err());
        let (c, g) = p.issue(0, 7, 0, 3, 4096).unwrap();
        assert!(p.issue(0, 7, 0, 3, 4096).is_err());
        assert!(p.issue(1, 8, 0, 3, 4096).is_err());
        assert!(p.issue(2, 9, 1, 3, 4096).is_err());
        assert!(p.issue(2, 9, 0, 5, 4096).is_err());
        assert!(p.accept(0b001).is_err());
        let original = stamp(0, 7, c, g, 3);
        for changed in [
            Stamp {
                generation: g + 1,
                ..original
            },
            Stamp {
                stage: 8,
                ..original
            },
            Stamp {
                chunk: c + 1,
                ..original
            },
            Stamp {
                count: 5,
                ..original
            },
            Stamp {
                start: 1,
                ..original
            },
            Stamp {
                layer: 64,
                ..original
            },
        ] {
            assert!(p.check(changed).is_err());
        }
        let mut q = Policy::new(1);
        *q.chunk = u64::MAX;
        assert!(q.issue(0, 1, 0, 1, 4096).is_err());
        let mut q = Policy::new(1);
        *q.generation = u64::MAX;
        assert!(q.issue(0, 1, 0, 1, 4096).is_err());
        assert!(p.finish().is_err());
    }
    #[test]
    fn exact_production_queue_folds_every_status_before_next_clear() {
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        for _ in 0..2 {
            queue_stage(&cancel, |op| {
                events.push(op);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(events.len(), 62);
        for stage in events.chunks_exact(31) {
            assert!(
                stage[..16]
                    .iter()
                    .enumerate()
                    .all(|(i, op)| *op == Operation::Clear(i))
            );
            assert!(
                stage[16..20]
                    .iter()
                    .enumerate()
                    .all(|(i, op)| *op == Operation::Projection(i))
            );
            assert!(
                stage[20..26]
                    == [
                        Operation::Convolution,
                        Operation::Preparation,
                        Operation::Controls,
                        Operation::Recurrence,
                        Operation::Closing,
                        Operation::Output
                    ]
            );
            assert!(
                stage[26..]
                    .iter()
                    .enumerate()
                    .all(|(i, op)| *op == Operation::Status(i))
            );
        }
        // A queue receipt has no sync/map/accept operation to truncate this order.
        assert!(events[30] == Operation::Status(4) && events[31] == Operation::Clear(0));
    }
    #[test]
    fn queue_error_panic_and_cancellation_at_every_boundary_poison_without_replay() {
        for failure in 0..31 {
            for mode in 0..3 {
                let poison = AtomicBool::new(false);
                let cancel = AtomicBool::new(false);
                let mut n = 0;
                let attempted = guarded(&poison, || {
                    queue_stage(&cancel, |_| {
                        let here = n;
                        n += 1;
                        if here == failure {
                            match mode {
                                0 => return Err(Error::InvalidBufferLength),
                                1 => panic!("fixed public queue panic"),
                                _ => cancel.store(true, Ordering::Release),
                            }
                        }
                        Ok(())
                    })
                });
                assert!(attempted.is_err() && poison.load(Ordering::Acquire));
                assert_eq!(n, failure + 1);
            }
        }
        let poison = AtomicBool::new(false);
        assert!(
            guarded(&poison, || -> Result<()> {
                Err(Error::LimitExceeded("fixed allocation failure"))
            })
            .is_err()
        );
        assert!(poison.load(Ordering::Acquire));
    }
    #[test]
    fn receipt_drop_poison_and_complete_metadata_wipe_are_independent_of_payloads() {
        let s = stamp(7, 11, 13, 17, 8);
        let mut m = metadata(s);
        let zero = BufferId::metadata_fixture(0, 0, 0);
        m.zeroize();
        assert!(
            m.scratch
                .iter()
                .chain(m.state.iter())
                .chain([&m.input, &m.output, &m.fault])
                .all(|id| *id == zero)
        );
        assert_eq!(
            [
                m.stamp.workspace,
                m.stamp.chunk,
                m.stamp.generation,
                m.stamp.stage,
                m.stamp.sequence,
                m.stamp.public,
                m.stamp.start
            ],
            [0; 7]
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
            drop(PendingGdnBatch {
                metadata: Zeroizing::new(metadata(s)),
                poison: poison.clone(),
                armed,
            });
            assert_eq!(poison.load(Ordering::Acquire), armed);
        }
    }
    #[test]
    fn actual_anchor_rejects_every_mixed_owner_view_and_epoch() {
        let s = stamp(1, 2, 3, 4, 5);
        let anchor = metadata(s);
        assert!(anchor_matches(&anchor, &metadata(s)));
        let changed = [
            Stamp { workspace: 78, ..s },
            Stamp { chunk: 4, ..s },
            Stamp { generation: 5, ..s },
            Stamp { stage: 3, ..s },
            Stamp { sequence: 24, ..s },
            Stamp { public: 124, ..s },
            Stamp { layer: 2, ..s },
            Stamp { start: 1, ..s },
            Stamp { count: 3, ..s },
            Stamp {
                input_first: 8,
                ..s
            },
            Stamp {
                input_stride: 2574,
                ..s
            },
            Stamp {
                output_first: 12,
                ..s
            },
            Stamp {
                output_stride: 2580,
                ..s
            },
        ];
        for c in changed {
            assert!(!anchor_matches(&anchor, &metadata(c)));
        }
        for i in 0..N + 4 + 3 {
            let mut m = metadata(s);
            let wrong = BufferId::metadata_fixture(7, 14, 0); // same logical owner slot in new epoch.
            match i {
                0..N => m.scratch[i] = wrong,
                N..20 => m.state[i - N] = wrong,
                20 => m.input = wrong,
                21 => m.output = wrong,
                _ => m.fault = wrong,
            }
            assert!(!anchor_matches(&anchor, &m));
            m.zeroize();
        }
    }
    #[test]
    fn partial_allocation_and_cancel_keep_every_already_tracked_full_page() {
        let l = layout();
        for failure in 0..N {
            let poison = AtomicBool::new(false);
            let cancel = AtomicBool::new(false);
            let mut owners = Vec::new();
            let attempt = guarded(&poison, || {
                allocate_ids(l.words, &cancel, |words| {
                    if owners.len() == failure {
                        return Err(Error::LimitExceeded("fixed reserve failure"));
                    }
                    let id = BufferId::metadata_fixture(7, 13, owners.len());
                    owners.push((id, BufferLayout::new(words).unwrap().owned_bytes));
                    Ok(id)
                })
            });
            assert!(attempt.is_err() && poison.load(Ordering::Acquire));
            assert_eq!(owners.len(), failure);
            assert_eq!(
                owners.iter().map(|(_, b)| *b as u64).sum::<u64>(),
                l.requests().unwrap()[..failure].iter().sum::<u64>()
            );
        }
        for after in 0..N {
            let poison = AtomicBool::new(false);
            let cancel = AtomicBool::new(false);
            let mut owners = Vec::new();
            let attempt = guarded(&poison, || {
                allocate_ids(l.words, &cancel, |words| {
                    let id = BufferId::metadata_fixture(7, 13, owners.len());
                    owners.push((id, BufferLayout::new(words).unwrap().owned_bytes));
                    if owners.len() == after + 1 {
                        cancel.store(true, Ordering::Release);
                    }
                    Ok(id)
                })
            });
            assert!(attempt.is_err() && poison.load(Ordering::Acquire));
            assert_eq!(owners.len(), after + 1);
        }
    }
    #[test]
    fn batch_lifecycle_advances_exact_count_and_scalar_receipts_do_not_mix() {
        for count in [1, 3, 5, 8] {
            let mut l = Lifecycle::new(4096);
            l.begin_batch(0, count).unwrap();
            assert!(l.commit(0).is_err()); // scalar commit cannot consume batch count1.
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
    fn physical_page_trims_hidden_padding_and_unowned_extent_reject() {
        validate_page(0, 524288, 524288, 524288, 524288).unwrap();
        // Original public 40 KiB vector may own its charged 64 KiB page.
        validate_page(0, 40960, 65536, 40960, 65536).unwrap();
        for (offset, size, physical, owned, bound) in [
            (4, 524288, 524288, 524288, 524288),
            (0, 524284, 524288, 524288, 524288),
            (0, 524288, 1048576, 524288, 524288),
            (0, 524288, 262144, 524288, 524288),
            (0, 40960, 131072, 40960, 65536),
            (0, 0, 0, 0, 0),
        ] {
            assert!(validate_page(offset, size, physical, owned, bound).is_err());
        }
    }
    #[test]
    fn original_public_logical_tails_restore_full_owning_proof_without_changing_launch_view() {
        let stream = cubecl::stream_id::StreamId::current();
        for words in [40960, 96, 128, 6144] {
            let page = BufferLayout::new(words).unwrap().owned_bytes as u64;
            let used = words as u64 * 4;
            let logical = Handle::new(stream, page).offset_end(page - used);
            let full = full_public_view(&logical, words).unwrap();
            assert_eq!(logical.size_in_used(), used);
            assert_eq!(logical.offset_end, Some(page - used));
            assert_eq!(full.size_in_used(), page);
            assert_eq!(full.offset_end, None);
            assert!(full_public_view(&Handle::new(stream, page), words).is_err());
            assert!(
                full_public_view(
                    &Handle::new(stream, page * 2).offset_end(page * 2 - used),
                    words
                )
                .is_err()
            );
            assert!(full_public_view(&logical.clone().offset_start(4), words).is_err());
            assert!(full_public_view(&logical.clone().offset_end(4), words).is_err());
        }
    }
    #[test]
    fn info_inventory_and_actual_projection_tiles_fit_separate_runtime_cap() {
        // Runtime metadata pessimistically counts two u64 lengths per ArrayArg
        // and an eight-byte slot for every runtime scalar. Signatures changing
        // beyond this pinned inventory fail rather than silently enlarging IO.
        let signatures = [
            (include_str!("gdn_batch_pointwise.rs"), "conv_silu", 6, 11),
            (include_str!("gdn_batch_pointwise.rs"), "prepare_qkv", 5, 12),
            (
                include_str!("gdn_batch_pointwise.rs"),
                "prepare_controls",
                5,
                9,
            ),
            (include_str!("gdn_batch_pointwise.rs"), "output_gate", 5, 10),
            (include_str!("gdn_chunk.rs"), "gdn_chunk", 8, 21),
            (include_str!("gdn_chunk.rs"), "gdn_single_streaming", 8, 21),
            (include_str!("quant_matmul.rs"), "packed_matmul", 3, 6),
            (
                include_str!("resident_util.rs"),
                "verify_status_batch",
                2,
                4,
            ),
        ];
        for (source, name, arrays, scalars) in signatures {
            let start = source.find(&format!("fn {name}(")).unwrap();
            let signature = source[start..].split("\n) {").next().unwrap();
            let actual_arrays = signature.lines().filter(|l| l.contains("Array<")).count();
            let actual_scalars = signature
                .lines()
                .filter(|l| {
                    !l.contains("#[comptime]")
                        && (l.contains(": usize") || l.contains(": f32") || l.contains(": u32"))
                })
                .count();
            assert_eq!((actual_arrays, actual_scalars), (arrays, scalars), "{name}");
            assert!(arrays * 16 + scalars * 8 <= 512);
        }
        let shapes = [
            (2560usize, 10240usize),
            (2560, 6144),
            (2560, 48),
            (2560, 48),
            (6144, 2560),
        ];
        let tiles = shapes
            .into_iter()
            .map(|(input, output)| output.div_ceil(((64 * 1024 * 1024) / (input * 4)).min(16384)))
            .sum::<usize>();
        assert_eq!(tiles, 6); // worst supported F32; packed dtypes use no more tiles.
        assert_eq!(16 + tiles + 5 + 5 + 5, 37); // clear, matrix, finite, math, status.
        assert_eq!(
            INFO_BYTES_PER_STAGE * 36,
            if crate::strata_intel_core::enabled() {
                2_506_752
            } else {
                2_359_296
            }
        );
        assert_eq!(
            INFO_BYTES_PER_STAGE * 36 * 2,
            if crate::strata_intel_core::enabled() {
                5_013_504
            } else {
                4_718_592
            }
        );
        const { assert!(INFO_BYTES_PER_STAGE * 36 * 2 < 64 * 1024 * 1024) };
    }
}
