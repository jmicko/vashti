//! Unconnected native layer1 PLE bridge into the actual CPU history owner.
//! Cached original projections/public vectors only. GPU OLD is a readonly
//! snapshot, NEXT is candidate-only. Enqueue never maps/syncs/audits/commits.
//! Count1 uses the separately accepted GPU-tail topology, not scalar CPU tail.
#![allow(dead_code)] // Future all-family coordinator; no default/model hook.
use super::*;
use crate::{
    ple_batch_gpu as p,
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_gdn::BoundProjection,
    resident_gr::batch::full_private,
};
use cubecl::server::Handle;
use kuno_text::{cpu::Matrix, model::GraphBindingId, ops::ple_transition as core};
use std::sync::atomic::AtomicU64;

const N: usize = 19;
const EMB: usize = 0;
const KEY: usize = 1;
const VALUE: usize = 2;
const KEY_NORM: usize = 3;
const QUERY_NORM: usize = 4;
const GATES: usize = 5;
const GATED: usize = 6;
const NORMALIZED: usize = 7;
const CONV: usize = 8;
const OLD: usize = 9;
const NEXT: usize = 10;
const STATUS: usize = 11;
const CHANNELS: usize = 10240;
const WIDTH: usize = 2560;
const STREAMS: usize = 4;
const HISTORY: usize = 9;
const HOST_CAP: usize = 4 * 1024 * 1024;
pub(crate) const MAIN_BYTES: u64 = 6_258_688;
pub(crate) const RUNTIME_AUDITED_RESERVATION: u64 = 4_587_520;
const INFO_BYTES: u64 = 64 * 2 * 512;
const WIDTHS: [usize; N] = [
    2560, 10240, 2560, 10240, 10240, 4, 10240, 10240, 10240, 92160, 92160, 4, 4, 4, 10240, 4,
    10240, 10240, 10240,
];
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn error() -> Error {
    Error::InvalidFormat("native PLE chunk owner/transition")
}
fn completed_projection_cursor(next: usize, position: usize, pending: bool) -> Result<()> {
    if next == 0 || next != position || pending {
        return Err(error());
    }
    Ok(())
}
fn guarded<T>(poison: &AtomicBool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        _ => {
            poison.store(true, Ordering::Release);
            Err(error())
        }
    }
}
fn identity() -> Result<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)
}
fn words(i: usize) -> usize {
    if [OLD, NEXT, 18].contains(&i) {
        WIDTHS[i]
    } else {
        WIDTHS[i] * MAX_BATCH
    }
}
fn graph_preflight(graph: BoundGraph<'_, '_>) -> Result<()> {
    graph.verify_unchanged()?;
    let g = graph.geometry();
    let p = g.ple;
    if p.layer != 1
        || p.head_width != 160
        || p.heads != 16
        || p.table_rows != 320001536
        || p.predecessor_tokens != 2
        || p.boundary_token != 248044
        || p.conv_channels != CHANNELS
        || p.conv_kernel != 4
        || p.conv_dilation != 3
        || g.residual.streams != STREAMS
        || g.residual.width != WIDTH
        || g.vocabulary != 248320
    {
        return Err(error());
    }
    Ok(())
}
fn requests() -> Result<[u64; N]> {
    let mut result = [0; N];
    for (i, r) in result.iter_mut().enumerate() {
        *r = BufferLayout::new(words(i)).map_err(u::fault)?.owned_bytes as u64;
    }
    Ok(result)
}
fn token_packet(
    count: usize,
    start: usize,
    context: usize,
    tokens: &[u32; MAX_BATCH],
    positions: &[u32; MAX_BATCH],
) -> Result<()> {
    if !(1..=MAX_BATCH).contains(&count)
        || start.checked_add(count).is_none_or(|n| n > context)
        || (0..count).any(|t| positions[t] as usize != start + t || tokens[t] >= 248320)
        || tokens[count..]
            .iter()
            .chain(&positions[count..])
            .any(|&v| v != 0)
    {
        return Err(error());
    }
    Ok(())
}
#[derive(Clone, PartialEq, Eq)]
struct Metadata {
    owner: u64,
    generation: u64,
    sequence: SequenceKey,
    start: usize,
    count: usize,
    ids: [BufferId; N],
    input: BufferId,
    output: BufferId,
    fault: BufferId,
    first: [usize; 2],
    stride: [usize; 2],
    tokens: [u32; MAX_BATCH],
    positions: [u32; MAX_BATCH],
    old_tokens: [i32; 2],
}
impl Zeroize for Metadata {
    fn zeroize(&mut self) {
        self.owner.zeroize();
        self.generation.zeroize();
        self.sequence.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.ids.zeroize();
        self.input.zeroize();
        self.output.zeroize();
        self.fault.zeroize();
        self.first.zeroize();
        self.stride.zeroize();
        self.tokens.zeroize();
        self.positions.zeroize();
        self.old_tokens.zeroize();
    }
}
fn metadata_matches(expected: &Metadata, received: &Metadata) -> bool {
    expected == received
}

pub(super) struct Anchor {
    metadata: Zeroizing<Metadata>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Anchor {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}
pub(crate) struct PendingPleBatch {
    metadata: Zeroizing<Metadata>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for PendingPleBatch {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}
struct Public {
    gamma: [Handle; 3],
    conv: Handle,
}
pub(crate) struct PleBatchWorkspace {
    binding: GraphBindingId,
    sequence: SequenceKey,
    identity: u64,
    generation: u64,
    next: usize,
    ids: Zeroizing<[BufferId; N]>,
    public: Public,
    poison: Arc<AtomicBool>,
    pending: bool,
}
impl Drop for PleBatchWorkspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
        self.sequence.zeroize();
        self.identity.zeroize();
        self.generation.zeroize();
        self.next.zeroize();
    }
}
impl PleBatchWorkspace {
    pub(crate) fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; N]> {
        graph_preflight(graph)?;
        requests()
    }
    /// Caller must include these exact19 pages in actual MainAdmission BEFORE
    /// allocation. Idle constructor boundary ONLY, before queuing any chunk;
    /// allocation initialization is cleanup-bounded, never an inference limit.
    /// This method does not grant graph-wide MAIN or I/O credits.
    pub(crate) fn allocate(
        model: &mut ResidentModel<'_, '_>,
        sequence: SequenceKey,
        poison: Arc<AtomicBool>,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        let result = guarded(&poison, || {
            u::cancel(cancel)?;
            graph_preflight(model.graph)?;
            if !Arc::ptr_eq(&model.admission.sequence_poison(), &poison)
                || !model.admission.transient_io()
                || model.poisoned
                || model.ple_batch_pending.is_some()
                || poison.load(Ordering::Acquire)
                || model
                    .ple_batch_poison
                    .as_ref()
                    .is_some_and(|p| !Arc::ptr_eq(p, &poison))
            {
                return Err(error());
            }
            core::owner_capacity_bytes(&model.ple_conv, &model.ple_history).map_err(u::fault)?;
            // Sealed cache/actual client BEFORE any vector lookup.
            model
                .public
                .validate_native_batch_owner(model.graph, &model.arena)?;
            let public = Public {
                gamma: [
                    model
                        .public
                        .vector(model.graph, "blk.1.ple_norm_key.weight")?,
                    model
                        .public
                        .vector(model.graph, "blk.1.ple_norm_query.weight")?,
                    model
                        .public
                        .vector(model.graph, "blk.1.ple_norm_conv.weight")?,
                ],
                conv: model
                    .public
                    .vector(model.graph, "blk.1.ple_conv1d.weight")?,
            };
            let needed = requests()?.into_iter().try_fold(0u64, |s, n| {
                s.checked_add(n).ok_or(Error::ArithmeticOverflow)
            })?;
            if needed != MAIN_BYTES
                || needed
                    > model
                        .arena
                        .budget_bytes()
                        .checked_sub(model.arena.charged_bytes())
                        .ok_or(Error::ArithmeticOverflow)? as u64
            {
                return Err(error());
            }
            let identity = identity()?;
            let mut ids = Zeroizing::new([model.b.fault; N]);
            for (i, id) in ids.iter_mut().enumerate() {
                u::cancel(cancel)?;
                *id = model.arena.allocate_zeroed(words(i)).map_err(u::fault)?;
            }
            let workspace = Self {
                binding: model.graph.binding_id(),
                sequence,
                identity,
                generation: 0,
                next: model.position,
                ids,
                public,
                poison: poison.clone(),
                pending: false,
            };
            workspace.validate_ids(&model.arena)?;
            model.ple_batch_poison = Some(poison.clone());
            u::cancel(cancel)?;
            Ok(workspace)
        });
        if result.is_err() {
            model.poisoned = true;
        }
        result
    }
    fn validate_ids(&self, arena: &PrivateArena) -> Result<()> {
        for (i, &id) in self.ids.iter().enumerate() {
            if self.ids[..i].contains(&id)
                || arena.layout(id).map_err(u::fault)?
                    != BufferLayout::new(words(i)).map_err(u::fault)?
            {
                return Err(error());
            }
        }
        Ok(())
    }
    fn healthy(&self, model: &ResidentModel<'_, '_>, cancel: &AtomicBool) -> Result<()> {
        u::cancel(cancel)?;
        if model.poisoned
            || self.poison.load(Ordering::Acquire)
            || model.retired.load(Ordering::Acquire)
            || self.binding != model.graph.binding_id()
            || model
                .ple_batch_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
        {
            return Err(error());
        }
        self.validate_ids(&model.arena)
    }
    /// PLE advances this cursor only in the all-family accepted commit. Scalar
    /// continuation leaves it behind and therefore cannot change native policy.
    pub(super) fn validate_completed_projection_boundary(
        &self,
        model: &ResidentModel<'_, '_>,
    ) -> Result<()> {
        self.healthy(model, &AtomicBool::new(false))?;
        completed_projection_cursor(self.next, model.position, self.pending)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: usize,
        tokens: &Zeroizing<[u32; MAX_BATCH]>,
        positions: &Zeroizing<[u32; MAX_BATCH]>,
        cancel: &AtomicBool,
    ) -> Result<PendingPleBatch> {
        let poison = self.poison.clone();
        let result = guarded(&poison, || {
            self.enqueue_checked(
                model, projector, input, output, spans, start, tokens, positions, cancel,
            )
        });
        if result.is_err() {
            model.poisoned = true;
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_checked(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: usize,
        tokens: &Zeroizing<[u32; MAX_BATCH]>,
        positions: &Zeroizing<[u32; MAX_BATCH]>,
        cancel: &AtomicBool,
    ) -> Result<PendingPleBatch> {
        self.healthy(model, cancel)?;
        graph_preflight(model.graph)?;
        let count = spans.count();
        token_packet(count, start, model.context, tokens, positions)?;
        if self.pending
            || model.ple_batch_pending.is_some()
            || start != self.next
            || start != model.position
            || spans.packed_first() != 0
            || projector.global_fault() != model.b.fault
        {
            return Err(error());
        }
        validate_view(
            MAX_BATCH,
            spans.input_first(),
            spans.input_stride(),
            CHANNELS,
            u::words(&model.arena, input)?,
        )?;
        validate_view(
            MAX_BATCH,
            spans.output_first(),
            spans.output_stride(),
            CHANNELS,
            u::words(&model.arena, output)?,
        )?;
        validate_full(model, self, input, output)?;
        for (name, target) in [
            ("blk.1.ple_key.weight", KEY),
            ("blk.1.ple_value.weight", VALUE),
        ] {
            let matrix = Matrix::new(model.graph.tensor(name)?)?;
            if matrix.inputs() != WIDTH || matrix.outputs() != WIDTHS[target] {
                return Err(error());
            }
        }
        let host_owned =
            core::owner_capacity_bytes(&model.ple_conv, &model.ple_history).map_err(u::fault)?;
        host_bound(host_owned)?;
        let old_tokens =
            Zeroizing::new([model.ple_history.tokens()[0], model.ple_history.tokens()[1]]);
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        let metadata = Zeroizing::new(Metadata {
            owner: self.identity,
            generation,
            sequence: self.sequence,
            start,
            count,
            ids: *self.ids,
            input,
            output,
            fault: model.b.fault,
            first: [spans.input_first(), spans.output_first()],
            stride: [spans.input_stride(), spans.output_stride()],
            tokens: **tokens,
            positions: **positions,
            old_tokens: *old_tokens,
        });
        // Bind actual owner BEFORE every fallible source read/write/dispatch.
        model.ple_batch_pending = Some(Anchor {
            metadata: Zeroizing::new((*metadata).clone()),
            poison: self.poison.clone(),
            armed: true,
        });
        self.pending = true;
        self.generation = generation;
        let mut old = source(BufferLayout::new(words(OLD)).map_err(u::fault)?.owned_bytes)?;
        for (bytes, &v) in old.chunks_exact_mut(4).zip(model.ple_conv.history()) {
            bytes.copy_from_slice(&v.to_bits().to_le_bytes());
        }
        let mut embedding = embedding(model.graph, count, tokens, &old_tokens, cancel)?;
        let arena = &model.arena;
        queue(cancel, |operation| {
            match operation {
                Operation::Clear(i) => u::zero(arena, self.ids[i])?,
                Operation::Upload(i) => {
                    let data = if i == 0 {
                        std::mem::take(&mut *old)
                    } else {
                        std::mem::take(&mut *embedding)
                    };
                    let id = if i == 0 { self.ids[OLD] } else { self.ids[EMB] };
                    arena
                        .client()
                        .map_err(u::fault)?
                        .write_private(arena.handle(id).map_err(u::fault)?.clone(), data)
                        .map_err(u::fault)?;
                }
                Operation::Projection(i) => {
                    let (name, target) = [
                        ("blk.1.ple_key.weight", KEY),
                        ("blk.1.ple_value.weight", VALUE),
                    ][i];
                    let tensor = model.graph.tensor(name)?;
                    let matrix = Matrix::new(tensor)?;
                    if matrix.inputs() != WIDTH || matrix.outputs() != WIDTHS[target] {
                        return Err(error());
                    }
                    projector.enqueue_batch(
                        BoundProjection { tensor, matrix },
                        arena,
                        self.ids[EMB],
                        self.ids[target],
                        &BatchSpans::new(count, 0, 0, WIDTH, 0, WIDTHS[target]),
                        cancel,
                    )?;
                    model.numeric_boundary.observe(
                        if i == 0 {
                            super::numeric_boundaries::Boundary::ProjectedKey
                        } else {
                            super::numeric_boundaries::Boundary::ProjectedValue
                        },
                        arena,
                        &model.admission,
                        self.ids[target],
                        model.b.fault,
                        words(target),
                        start,
                        count,
                        cancel,
                    )?;
                }
                Operation::Tail => {
                    tail(arena, &self.ids, &self.public, input, output, spans, cancel)?
                }
                Operation::Status(i) => u::status_batch(
                    arena,
                    self.ids[STATUS + i],
                    model.b.fault,
                    count,
                    0,
                    WIDTHS[STATUS + i],
                    WIDTHS[STATUS + i],
                )?,
                Operation::History => {
                    u::status_prefix(arena, self.ids[18], CHANNELS, model.b.fault)?
                }
                Operation::Finite => u::validate(arena, output, model.b.fault)?,
            }
            Ok(())
        })?;
        Ok(PendingPleBatch {
            metadata,
            poison: self.poison.clone(),
            armed: true,
        })
    }
    fn receipt(
        &self,
        model: &ResidentModel<'_, '_>,
        receipt: &PendingPleBatch,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.healthy(model, cancel)?;
        let m = &*receipt.metadata;
        if !self.pending
            || !receipt.armed
            || !Arc::ptr_eq(&self.poison, &receipt.poison)
            || m.owner != self.identity
            || m.generation != self.generation
            || m.sequence != self.sequence
            || m.ids != *self.ids
            || m.fault != model.b.fault
            || m.start != self.next
            || m.start != model.position
            || model
                .ple_batch_pending
                .as_ref()
                .is_none_or(|a| !metadata_matches(&a.metadata, m))
        {
            return Err(error());
        }
        token_packet(m.count, m.start, model.context, &m.tokens, &m.positions)?;
        validate_full(model, self, m.input, m.output)
    }
    /// EXTERNAL OWNER PRECONDITION: exact WHOLE-model output/global-fault fence
    /// already completed, health/cancellation accepted, all earlier leases
    /// dropped. This performs additional full diagnostics and ACTUAL independent
    /// audited I/O cleanup; caller must preflight every OTHER family before ANY
    /// state swap. No enqueue receipt or MAIN credit substitutes for this proof.
    pub(crate) fn prepare_after_whole_model_fence<'a>(
        &'a mut self,
        model: &'a mut ResidentModel<'_, '_>,
        receipt: PendingPleBatch,
        cancel: &AtomicBool,
    ) -> Result<PleBatchAcceptance<'a>> {
        self.prepare_split_after_whole_model_fence(
            Preparation {
                graph: model.graph,
                arena: &model.arena,
                retired: &model.retired,
                context: model.context,
                position: model.position,
                owner_poisoned: &mut model.poisoned,
                sequence_poison: &model.ple_batch_poison,
                conv: &mut model.ple_conv,
                history: &mut model.ple_history,
                anchor: &mut model.ple_batch_pending,
                fault: model.b.fault,
                scalar_ids: [model.b.ple_emb, model.b.ple_key, model.b.ple_value],
            },
            receipt,
            cancel,
        )
    }
    fn receipt_parts(
        &self,
        c: &Preparation<'_, '_, '_>,
        receipt: &PendingPleBatch,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        c.graph.verify_unchanged()?;
        let m = &*receipt.metadata;
        if *c.owner_poisoned
            || c.retired.load(Ordering::Acquire)
            || self.poison.load(Ordering::Acquire)
            || c.sequence_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
            || self.binding != c.graph.binding_id()
            || !self.pending
            || !receipt.armed
            || !Arc::ptr_eq(&self.poison, &receipt.poison)
            || m.owner != self.identity
            || m.generation != self.generation
            || m.sequence != self.sequence
            || m.ids != *self.ids
            || m.fault != c.fault
            || m.start != self.next
            || m.start != c.position
            || c.anchor
                .as_ref()
                .is_none_or(|a| !metadata_matches(&a.metadata, m))
        {
            return Err(error());
        }
        self.validate_ids(c.arena)?;
        token_packet(m.count, m.start, c.context, &m.tokens, &m.positions)?;
        validate_full_parts(c.arena, self, m.input, m.output, c.fault, c.scalar_ids)
    }
    /// Field-split seam lets the future coordinator retain THIS history permit
    /// while preflighting other families and performing its second completed
    /// fault/cleanup/admission proof. The capability borrows only PLE fields.
    /// Same external WHOLE-model completed-fence prerequisite as above.
    pub(crate) fn prepare_split_after_whole_model_fence<'a>(
        &'a mut self,
        c: Preparation<'a, '_, '_>,
        receipt: PendingPleBatch,
        cancel: &AtomicBool,
    ) -> Result<PleBatchAcceptance<'a>> {
        let poison = self.poison.clone();
        let diagnostics = guarded(&poison, || {
            self.receipt_parts(&c, &receipt, cancel)?;
            let m = &*receipt.metadata;
            let (old, normalized, next) = if crate::next_shared::gr_selected() {
                let existing = core::owner_capacity_bytes(c.conv, c.history).map_err(u::fault)?;
                let [old, normalized, next] = read_full_three(
                    c.arena,
                    [self.ids[OLD], self.ids[NORMALIZED], self.ids[NEXT]],
                    m.count,
                    existing,
                )?;
                (old, normalized, next)
            } else {
                let old = read_full(c.arena, self.ids[OLD], words(OLD), m.count, false)?;
                let normalized = read_full(
                    c.arena,
                    self.ids[NORMALIZED],
                    words(NORMALIZED),
                    m.count,
                    true,
                )?;
                let next = read_full(c.arena, self.ids[NEXT], words(NEXT), m.count, false)?;
                (old, normalized, next)
            };
            // All raw Bytes leases are gone; inference was already completed.
            audit(c.arena, c.retired)?;
            self.receipt_parts(&c, &receipt, cancel)?;
            let existing = core::owner_capacity_bytes(c.conv, c.history).map_err(u::fault)?;
            let maps = old
                .capacity()
                .checked_add(normalized.capacity())
                .and_then(|n| n.checked_add(next.capacity()))
                .and_then(|n| n.checked_mul(4))
                .ok_or(Error::ArithmeticOverflow)?;
            if existing
                .checked_add(maps)
                .and_then(|n| n.checked_add(core::HISTORY_WORDS * 4 + 65536))
                .is_none_or(|n| n > HOST_CAP)
            {
                return Err(error());
            }
            Ok((old, normalized, next))
        });
        let (old, normalized, next) = match diagnostics {
            Ok(v) => v,
            Err(e) => {
                *c.owner_poisoned = true;
                c.retired.store(true, Ordering::Release);
                return Err(e);
            }
        };
        let m = &*receipt.metadata;
        let transition = core::prepare(
            c.conv,
            c.history,
            c.graph.ple_hash(),
            &m.old_tokens,
            &m.tokens[..m.count],
            &old[..words(OLD)],
            &normalized[..words(NORMALIZED)],
            &next[..words(NEXT)],
            HOST_CAP - (old.capacity() + normalized.capacity() + next.capacity()) * 4 - 65536,
            cancel,
        )
        .map_err(u::fault);
        match transition {
            Ok(transition) => Ok(PleBatchAcceptance {
                workspace: self,
                anchor: c.anchor,
                arena: c.arena,
                retired: c.retired,
                transition: Some(transition),
                receipt,
                armed: true,
            }),
            Err(e) => {
                *c.owner_poisoned = true;
                poison.store(true, Ordering::Release);
                Err(e)
            }
        }
    }
}

/// Constructed only inside the actual Model child/coordinator by splitting its
/// private fields. Receipt anchor and full owner proofs reject mixed histories.
/// No mutable arbitrary-state API or public runtime execution is exposed.
pub(crate) struct Preparation<'a, 'm, 'f> {
    pub(super) graph: BoundGraph<'m, 'f>,
    pub(super) arena: &'a PrivateArena,
    pub(super) retired: &'a AtomicBool,
    pub(super) context: usize,
    pub(super) position: usize,
    pub(super) owner_poisoned: &'a mut bool,
    pub(super) sequence_poison: &'a Option<Arc<AtomicBool>>,
    pub(super) conv: &'a mut CausalConvState,
    pub(super) history: &'a mut PleTokenHistory,
    pub(super) anchor: &'a mut Option<Anchor>,
    pub(super) fault: BufferId,
    pub(super) scalar_ids: [BufferId; 3],
}

pub(crate) struct PleBatchAcceptance<'a> {
    workspace: &'a mut PleBatchWorkspace,
    anchor: &'a mut Option<Anchor>,
    arena: &'a PrivateArena,
    retired: &'a AtomicBool,
    transition: Option<core::PleChunkCommit<'a>>,
    receipt: PendingPleBatch,
    armed: bool,
}
impl PleBatchAcceptance<'_> {
    /// All-family preflight BEFORE any state swap. No CPU histories can change
    /// concurrently: the core ticket retains their exclusive mutable borrows.
    pub(crate) fn preflight(&self, cancel: &AtomicBool) -> Result<()> {
        guarded(&self.workspace.poison, || {
            if self.workspace.poison.load(Ordering::Acquire)
                || self.retired.load(Ordering::Acquire)
                || !self.armed
            {
                return Err(error());
            }
            self.workspace.validate_ids(self.arena)?;
            self.transition
                .as_ref()
                .ok_or(error())?
                .preflight(cancel)
                .map_err(u::fault)
        })
    }
    /// Requires ALL family permits preflighted before the first commit. Exact
    /// active count updates this PLE cursor only; model cursor remains the
    /// future whole-model coordinator's responsibility. No fallible swap work.
    pub(crate) fn commit(mut self, cancel: &AtomicBool) -> Result<()> {
        self.preflight(cancel)?;
        if let Some(transition) = self.transition.take() {
            transition.commit();
        }
        self.workspace.next = self.receipt.metadata.start + self.receipt.metadata.count;
        self.workspace.pending = false;
        if let Some(anchor) = self.anchor.as_mut() {
            anchor.armed = false;
        }
        *self.anchor = None;
        self.receipt.armed = false;
        self.armed = false;
        Ok(())
    }
}
impl Drop for PleBatchAcceptance<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.workspace.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
    }
}

fn host_bound(existing: usize) -> Result<()> {
    // Exact simultaneous API-visible plan; charge ACTUAL authoritative capacity.
    let fixed = 368640 + 524288 + 131072 + 3 * 524288 + 524288 + 4096 + 65536;
    if existing.checked_add(fixed).is_none_or(|n| n > HOST_CAP) {
        return Err(error());
    }
    Ok(())
}
fn source(bytes: usize) -> Result<Zeroizing<Vec<u8>>> {
    let mut v = Zeroizing::new(Vec::new());
    v.try_reserve_exact(bytes).map_err(|_| error())?;
    if v.capacity() != bytes {
        return Err(error());
    }
    v.resize(bytes, 0);
    Ok(v)
}
fn embedding(
    graph: BoundGraph<'_, '_>,
    count: usize,
    tokens: &[u32; MAX_BATCH],
    old: &[i32; 2],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<u8>>> {
    let mut rows = Zeroizing::new([[0u32; 16]; MAX_BATCH]);
    let mut previous = Zeroizing::new(*old);
    for t in 0..count {
        u::cancel(cancel)?;
        crate::ple_chunk::rows_into(
            graph.ple_hash().metadata(),
            tokens[t],
            &previous,
            &mut rows[t],
        )
        .map_err(u::fault)?;
        previous[0] = previous[1];
        previous[1] = tokens[t] as i32;
    }
    let table = graph.tensor("per_layer_token_embd.weight")?;
    if table.info().dtype() != kuno_text::TensorType::Q5_0
        || table.info().row_len() != 160
        || table.info().row_bytes() != 110
        || table.info().row_count() != 320001536
    {
        return Err(error());
    }
    let mut upload = source(BufferLayout::new(words(EMB)).map_err(u::fault)?.owned_bytes)?;
    let mut packed = Zeroizing::new([0u8; 110]);
    let mut decoded = Zeroizing::new([0f32; 160]);
    for t in 0..count {
        for head in 0..16 {
            u::cancel(cancel)?;
            table.read_row(u64::from(rows[t][head]), &mut packed[..])?;
            kuno_text::TensorType::Q5_0.decode(&packed[..], &mut decoded[..])?;
            if decoded.iter().any(|x| !x.is_finite()) {
                return Err(error());
            }
            for (d, &v) in decoded.iter().enumerate() {
                let at = (t * WIDTH + head * 160 + d) * 4;
                upload[at..at + 4].copy_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    graph.verify_unchanged()?;
    Ok(upload)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    Clear(usize),
    Upload(usize),
    Projection(usize),
    Tail,
    Status(usize),
    History,
    Finite,
}
fn queue(cancel: &AtomicBool, mut f: impl FnMut(Operation) -> Result<()>) -> Result<()> {
    for op in (0..N)
        .map(Operation::Clear)
        .chain((0..2).map(Operation::Upload))
        .chain((0..2).map(Operation::Projection))
        .chain([Operation::Tail])
        .chain((0..7).map(Operation::Status))
        .chain([Operation::History, Operation::Finite])
    {
        u::cancel(cancel)?;
        f(op)?;
    }
    u::cancel(cancel)
}

#[allow(clippy::too_many_arguments)]
fn tail(
    arena: &PrivateArena,
    ids: &[BufferId; N],
    public: &Public,
    input: BufferId,
    output: BufferId,
    spans: &BatchSpans,
    cancel: &AtomicBool,
) -> Result<()> {
    let count = spans.count();
    let client = arena.client().map_err(u::fault)?;
    let dim = CubeDim::new_1d(64);
    let arg = |i| u::arg(arena, ids[i]);
    let gamma = |i: usize| u::raw_arg(public.gamma[i].clone(), CHANNELS);
    for (id, first, stride, target, status, which) in [
        (ids[KEY], 0, CHANNELS, KEY_NORM, 11, 0),
        (
            input,
            spans.input_first(),
            spans.input_stride(),
            QUERY_NORM,
            12,
            1,
        ),
    ] {
        u::cancel(cancel)?;
        p::rms::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(4, 8, 1),
            dim,
            u::arg(arena, id)?,
            gamma(which)?,
            arg(target)?,
            arg(status)?,
            1e-6,
            count,
            first,
            stride,
            0,
            0,
            CHANNELS,
            0,
            STREAMS,
            STREAMS,
            WIDTH,
        );
    }
    u::cancel(cancel)?;
    p::gate::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(4, 8, 1),
        dim,
        arg(KEY_NORM)?,
        arg(QUERY_NORM)?,
        arg(GATES)?,
        arg(13)?,
        count,
        0,
        CHANNELS,
        0,
        CHANNELS,
        0,
        STREAMS,
        0,
        STREAMS,
        STREAMS,
        WIDTH,
    );
    u::cancel(cancel)?;
    p::gated::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(CHANNELS.div_ceil(64) as u32, 8, 1),
        dim,
        arg(VALUE)?,
        arg(GATES)?,
        arg(GATED)?,
        arg(14)?,
        count,
        0,
        WIDTH,
        0,
        STREAMS,
        0,
        CHANNELS,
        0,
        CHANNELS,
        STREAMS,
        WIDTH,
    );
    u::cancel(cancel)?;
    p::rms::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(4, 8, 1),
        dim,
        arg(GATED)?,
        gamma(2)?,
        arg(NORMALIZED)?,
        arg(15)?,
        1e-6,
        count,
        0,
        CHANNELS,
        0,
        0,
        CHANNELS,
        0,
        STREAMS,
        STREAMS,
        WIDTH,
    );
    u::cancel(cancel)?;
    p::conv_tail::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(CHANNELS.div_ceil(64) as u32, 1, 1),
        dim,
        arg(NORMALIZED)?,
        u::raw_arg(public.conv.clone(), CHANNELS * 4)?,
        arg(OLD)?,
        arg(NEXT)?,
        u::arg(arena, input)?,
        arg(GATED)?,
        arg(CONV)?,
        u::arg(arena, output)?,
        arg(16)?,
        arg(17)?,
        arg(18)?,
        count,
        0,
        CHANNELS,
        0,
        0,
        HISTORY,
        0,
        HISTORY,
        spans.input_first(),
        spans.input_stride(),
        0,
        CHANNELS,
        0,
        CHANNELS,
        spans.output_first(),
        spans.output_stride(),
        0,
        CHANNELS,
        0,
        CHANNELS,
        0,
        CHANNELS,
    );
    u::cancel(cancel)
}

fn full_public(view: &Handle, logical: usize) -> Result<Handle> {
    let page = BufferLayout::new(logical).map_err(u::fault)?.owned_bytes as u64;
    let logical = logical as u64 * 4;
    if view.offset_start.unwrap_or(0) != 0
        || view.size() != page
        || view.size_in_used() != logical
        || view.offset_end.unwrap_or(0) != page - logical
    {
        return Err(error());
    }
    let mut h = view.clone();
    h.offset_end = None;
    Ok(h)
}
fn validate_full(
    model: &ResidentModel<'_, '_>,
    workspace: &PleBatchWorkspace,
    input: BufferId,
    output: BufferId,
) -> Result<()> {
    validate_full_parts(
        &model.arena,
        workspace,
        input,
        output,
        model.b.fault,
        [model.b.ple_emb, model.b.ple_key, model.b.ple_value],
    )
}
fn validate_full_parts(
    arena: &PrivateArena,
    workspace: &PleBatchWorkspace,
    input: BufferId,
    output: BufferId,
    fault: BufferId,
    scalar: [BufferId; 3],
) -> Result<()> {
    if u::words(arena, fault)? != 1 {
        return Err(error());
    }
    const PRIVATE: usize = N + 6;
    const TOTAL: usize = PRIVATE + 4;
    let mut ids = Zeroizing::new([input; PRIVATE]);
    ids[..N].copy_from_slice(&*workspace.ids);
    ids[N..].copy_from_slice(&[input, output, fault, scalar[0], scalar[1], scalar[2]]);
    if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(error());
    }
    let client = arena.client().map_err(u::fault)?;
    let stream = arena.handle(input).map_err(u::fault)?.stream;
    let mut resources: [Option<_>; TOTAL] = std::array::from_fn(|_| None);
    let mut full_handles = Vec::with_capacity(TOTAL);
    let mut sizes = Zeroizing::new([0u64; TOTAL]);
    for i in 0..TOTAL {
        let h = if i < PRIVATE {
            full_private(arena, ids[i], u::words(arena, ids[i])?)?
        } else if i < PRIVATE + 3 {
            full_public(&workspace.public.gamma[i - PRIVATE], CHANNELS)?
        } else {
            full_public(&workspace.public.conv, CHANNELS * 4)?
        };
        if h.stream != stream {
            return Err(error());
        }
        sizes[i] = h.size();
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
        let page = resources[i].as_ref().ok_or(error())?.resource();
        if page.offset != 0
            || page.size != sizes[i]
            || page.buffer.size() != sizes[i]
            || resources[..i]
                .iter()
                .flatten()
                .any(|r| r.resource().buffer == page.buffer)
        {
            return Err(error());
        }
    }
    Ok(())
}
struct Bytes(cubecl::bytes::Bytes);
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn validate_download(bytes: &[u8], logical: usize, count: usize, token_major: bool) -> Result<()> {
    let layout = BufferLayout::new(logical).map_err(u::fault)?;
    if bytes.len() != layout.owned_bytes {
        return Err(error());
    }
    for (i, b) in bytes.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes(b.try_into().map_err(|_| error())?);
        if !f32::from_bits(bits).is_finite()
            || ((i >= logical || (token_major && i >= count * CHANNELS)) && bits != 0)
        {
            return Err(error());
        }
    }
    Ok(())
}
fn read_full(
    arena: &PrivateArena,
    id: BufferId,
    logical: usize,
    count: usize,
    token_major: bool,
) -> Result<Zeroizing<Vec<f32>>> {
    let h = full_private(arena, id, logical)?;
    let bytes = Bytes(
        arena
            .client()
            .map_err(u::fault)?
            .read_one(h)
            .map_err(u::fault)?,
    );
    decode_download(&bytes.0, logical, count, token_major)
}
fn decode_download(
    bytes: &[u8],
    logical: usize,
    count: usize,
    token_major: bool,
) -> Result<Zeroizing<Vec<f32>>> {
    validate_download(bytes, logical, count, token_major)?;
    let n = bytes.len() / 4;
    let mut values = Zeroizing::new(Vec::new());
    values.try_reserve_exact(n).map_err(|_| error())?;
    if values.capacity() != n {
        return Err(error());
    }
    for b in bytes.chunks_exact(4) {
        values.push(f32::from_bits(u32::from_le_bytes(
            b.try_into().map_err(|_| error())?,
        )));
    }
    Ok(values) // raw guarded lease drops BEFORE audit/transition factory.
}
struct ReadBatch(Vec<cubecl::bytes::Bytes>);
impl Drop for ReadBatch {
    fn drop(&mut self) {
        for b in &mut self.0 {
            b.zeroize();
        }
    }
}
fn read_full_three(
    arena: &PrivateArena,
    ids: [BufferId; 3],
    count: usize,
    existing: usize,
) -> Result<[Zeroizing<Vec<f32>>; 3]> {
    let logical = [words(OLD), words(NORMALIZED), words(NEXT)];
    let mut raw_bytes = 0usize;
    let mut handles = Vec::new();
    handles.try_reserve_exact(3).map_err(|_| error())?;
    for (id, n) in ids.into_iter().zip(logical) {
        let h = full_private(arena, id, n)?;
        raw_bytes = raw_bytes
            .checked_add(h.size() as usize)
            .ok_or(Error::ArithmeticOverflow)?;
        handles.push(h);
    }
    // All three existing full maps and their converted Vecs can coexist during
    // conversion. Same4MiB cap, including64KiB original scoped metadata reserve.
    if existing
        .checked_add(raw_bytes.checked_mul(2).ok_or(Error::ArithmeticOverflow)?)
        .and_then(|n| n.checked_add(65536))
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err(error());
    }
    let raw = ReadBatch(
        cubecl::future::block_on(arena.client().map_err(u::fault)?.read_async(handles))
            .map_err(u::fault)?,
    );
    if raw.0.len() != 3 {
        return Err(error());
    }
    let old = decode_download(&raw.0[0], logical[0], count, false)?;
    let normalized = decode_download(&raw.0[1], logical[1], count, true)?;
    let next = decode_download(&raw.0[2], logical[2], count, false)?;
    // Guard wipes every raw map before the caller's original audited cleanup.
    Ok([old, normalized, next])
}
fn audit(arena: &PrivateArena, retired: &AtomicBool) -> Result<()> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(
            arena
                .client()
                .map_err(u::fault)?
                .sanitize_transient_io_audited(),
        )
        .map_err(u::fault)
    }));
    match result {
        Ok(Ok(r)) => {
            let full = r
                .upload_bytes
                .checked_add(r.uniform_bytes)
                .and_then(|n| n.checked_add(r.readback_bytes));
            if full.is_none_or(|n| n == 0 || n != r.inspected_bytes) || r.inspected_buffers == 0 {
                retired.store(true, Ordering::Release);
                return Err(error());
            }
            Ok(())
        }
        _ => {
            retired.store(true, Ordering::Release);
            Err(error())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn projection_switch_requires_an_accepted_matching_idle_native_cursor() {
        use super::completed_projection_cursor;
        for position in [1, 3, 8, 80, 4096] {
            assert!(completed_projection_cursor(position, position, false).is_ok());
            assert!(completed_projection_cursor(position, position, true).is_err());
            assert!(completed_projection_cursor(position, position + 1, false).is_err());
            assert!(completed_projection_cursor(position, position - 1, false).is_err());
        }
        assert!(completed_projection_cursor(0, 0, false).is_err());
        assert!(completed_projection_cursor(0, 1, false).is_err());
    }

    use super::*;
    #[test]
    fn exact_full_pages_and_separate_component_bound_are_pinned() {
        assert_eq!(requests().unwrap().iter().sum::<u64>(), MAIN_BYTES);
        assert_eq!(N, 19);
        assert_eq!(MAIN_BYTES, 6_258_688);
        let info = 64 * 2 * 512;
        let runtime = 131072 + 524288 + 3 * 524288 + info;
        assert_eq!(runtime * 2, RUNTIME_AUDITED_RESERVATION);
        assert!(runtime * 2 < 64 * 1024 * 1024);
        assert!(host_bound(core::HISTORY_WORDS * 4 + 8).is_ok());
        assert!(host_bound(HOST_CAP).is_err());
    }
    #[test]
    fn queue_retains_sources_before_consumers_folds_all_statuses_and_never_completes() {
        let mut ops = Vec::new();
        queue(&AtomicBool::new(false), |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        assert_eq!(&ops[..N], &(0..N).map(Operation::Clear).collect::<Vec<_>>());
        assert_eq!(
            &ops[N..N + 4],
            &[
                Operation::Upload(0),
                Operation::Upload(1),
                Operation::Projection(0),
                Operation::Projection(1)
            ]
        );
        assert_eq!(ops.last(), Some(&Operation::Finite));
        for i in 0..7 {
            assert!(ops.contains(&Operation::Status(i)));
        }
        assert!(ops.contains(&Operation::History));
        let cancel = AtomicBool::new(false);
        let mut n = 0;
        assert!(
            queue(&cancel, |op| {
                n += 1;
                if op == Operation::Tail {
                    cancel.store(true, Ordering::Release);
                }
                Ok(())
            })
            .is_err()
        );
        assert_eq!(n, N + 5);
        // Another enqueue starts only after the previous entire status sequence.
        let first = ops.len();
        queue(&AtomicBool::new(false), |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        assert_eq!(ops[first - 1], Operation::Finite);
        assert_eq!(ops[first], Operation::Clear(0));
    }
    #[test]
    fn exact_runtime_packet_bound_and_overflow_reject_before_private_fill() {
        for count in [1, 3, 5, 8] {
            let mut tokens = [0; 8];
            let mut positions = [0; 8];
            for t in 0..count {
                tokens[t] = (t + 1) as u32;
                positions[t] = (4096 - count + t) as u32;
            }
            assert!(token_packet(count, 4096 - count, 4096, &tokens, &positions).is_ok());
            positions[0] += 1;
            assert!(token_packet(count, 4096 - count, 4096, &tokens, &positions).is_err());
        }
        assert!(token_packet(9, 0, 4096, &[0; 8], &[0; 8]).is_err());
        assert!(token_packet(1, usize::MAX, usize::MAX, &[0; 8], &[0; 8]).is_err());
        assert!(token_packet(1, 0, 4096, &[248320, 0, 0, 0, 0, 0, 0, 0], &[0; 8]).is_err());
    }
    #[test]
    fn full_maps_finite_inactive_and_physical_padding_are_not_silently_accepted() {
        let layout = BufferLayout::new(words(NORMALIZED)).unwrap();
        let mut bytes = vec![0; layout.owned_bytes];
        assert!(validate_download(&bytes, words(NORMALIZED), 1, true).is_ok());
        for (index, bits) in [
            (CHANNELS, (-0f32).to_bits()),
            (layout.owned_words - 1, 1),
            (0, f32::NAN.to_bits()),
        ] {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&bits.to_le_bytes());
            assert!(validate_download(&bytes, words(NORMALIZED), 1, true).is_err());
            bytes[index * 4..index * 4 + 4].fill(0);
        }
        assert!(validate_download(&bytes[..bytes.len() - 4], words(NORMALIZED), 1, true).is_err());
        bytes.extend([0; 4]);
        assert!(validate_download(&bytes, words(NORMALIZED), 1, true).is_err());
    }
    #[test]
    fn info_inventory_counts_actual_tail_runtime_arguments_and_dispatch_window() {
        let source = include_str!("ple_batch_gpu.rs");
        for name in ["rms", "gate", "gated", "conv_tail"] {
            let head = source
                .split(&format!("pub fn {name}("))
                .nth(1)
                .unwrap()
                .split(") {")
                .next()
                .unwrap();
            let arrays = head.matches("Array<").count();
            let runtime = head
                .lines()
                .filter(|line| line.contains(": usize") && !line.contains("#[comptime]"))
                .count()
                + head.lines().filter(|line| line.contains(": f32")).count();
            // U64 metadata two words per ArrayArg and padded8-byte scalars is
            // the pessimistic actual packing. No count/token/data comptime.
            assert!(
                arrays * 16 + runtime * 8 <= 512,
                "public INFO inventory {name}"
            );
            if name == "conv_tail" {
                assert_eq!(arrays, 11);
            }
        }
        // Original F32 PLE key uses2 packed row tiles, value1:3 projections
        // plus2 full finite checks. Pointwise6, status7+history1, outputfinite1.
        let dispatches = N + 5 + 6 + 8 + 1;
        assert_eq!(dispatches, 39);
        assert!(dispatches <= 64);
        assert_eq!(INFO_BYTES, 64 * 2 * 512);
    }
    #[test]
    fn sealed_metadata_rejects_stale_owner_sequence_packet_and_arena_epoch() {
        let id = BufferId::metadata_fixture(4, 8, 2);
        let sequence = SequenceKey::new().unwrap();
        let expected = Zeroizing::new(Metadata {
            owner: 1,
            generation: 2,
            sequence,
            start: 7,
            count: 3,
            ids: [id; N],
            input: id,
            output: id,
            fault: id,
            first: [0; 2],
            stride: [10240; 2],
            tokens: [0; 8],
            positions: [0; 8],
            old_tokens: [-1; 2],
        });
        for change in 0..8 {
            let mut received = Zeroizing::new((*expected).clone());
            match change {
                0 => received.owner += 1,
                1 => received.generation += 1,
                2 => received.sequence = SequenceKey::new().unwrap(),
                3 => received.start += 1,
                4 => received.count += 1,
                5 => received.ids[0] = BufferId::metadata_fixture(4, 9, 2),
                6 => received.tokens[7] = 2,
                _ => received.old_tokens[0] = 17,
            }
            assert!(!metadata_matches(&expected, &received));
        }
        let mut received = Zeroizing::new((*expected).clone());
        received.zeroize();
        assert!(
            received
                .tokens
                .iter()
                .chain(&received.positions)
                .all(|&v| v == 0)
        );
        assert!(received.old_tokens.iter().all(|&v| v == 0));
        assert!(source(usize::MAX).is_err());
    }
    #[test]
    fn error_panic_or_abandoned_receipt_sets_sticky_shared_poison_without_callback() {
        let poison = Arc::new(AtomicBool::new(false));
        assert!(guarded::<()>(&poison, || Err(error())).is_err());
        assert!(poison.load(Ordering::Acquire));
        let poison = Arc::new(AtomicBool::new(false));
        assert!(guarded::<()>(&poison, || panic!("public injected fault")).is_err());
        assert!(poison.load(Ordering::Acquire));
        let id = BufferId::metadata_fixture(4, 8, 2);
        let sequence = SequenceKey::new().unwrap();
        let metadata = Metadata {
            owner: 1,
            generation: 2,
            sequence,
            start: 7,
            count: 3,
            ids: [id; N],
            input: id,
            output: id,
            fault: id,
            first: [0; 2],
            stride: [10240; 2],
            tokens: [0; 8],
            positions: [0; 8],
            old_tokens: [-1; 2],
        };
        let poison = Arc::new(AtomicBool::new(false));
        drop(PendingPleBatch {
            metadata: Zeroizing::new(metadata),
            poison: poison.clone(),
            armed: true,
        });
        assert!(poison.load(Ordering::Acquire));
    }
}
