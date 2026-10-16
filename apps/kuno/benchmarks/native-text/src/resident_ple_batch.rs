//! Native layer1 PLE with original GPU history in the Intel core; legacy
//! paths retain the separately checked CPU convolution-history owner.
//! Invocation preparation owns/stages original OLD/EMB before graph capture;
//! math enqueue consumes that exact prepared transaction without host uploads.
//! GPU OLD is readonly and NEXT candidate-only; completion/commit stay separate.
//! Count1 uses the accepted GPU-tail topology, not scalar CPU tail.
#![allow(dead_code)] // Future all-family coordinator; no default/model hook.
use super::*;
use crate::{
    expert_banks::OriginalPackedProfile,
    ple_batch_gpu as p,
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_gdn::BoundProjection,
    resident_gr::batch::full_private,
};
use cubecl::server::Handle;
use kuno_text::{cpu::Matrix, model::GraphBindingId, ops::ple_transition as core};
use std::sync::atomic::AtomicU64;
#[path = "strata_ple_history.rs"]
mod device_history;
#[path = "resident_ple_prompt.rs"]
pub(crate) mod prompt;
#[path = "strata_prompt_ple.rs"]
pub(crate) mod prompt_source;

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
const HOST_CAP: usize = 4 * 1024 * 1024
    + if cfg!(feature = "strata-intel-core") {
        prompt_source::HOST_BYTES
    } else {
        0
    };
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
// Construction is not an accepted-token commit. Only an untouched original
// workspace can authorize initial source/request preparation; it cannot mint a
// CompletedNativeProjectionBoundary used by runtime policy changes.
fn unstarted_projection_cursor(
    next: usize,
    position: usize,
    generation: u64,
    pending: bool,
) -> Result<()> {
    if next != 0 || position != 0 || generation != 0 || pending {
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
/// Staged original inputs and an armed transaction, not a queued/completed
/// receipt. Dropping this before consumption poisons the same original owner.
pub(crate) struct PreparedPleBatch {
    receipt: PendingPleBatch,
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
    profile: OriginalPackedProfile,
    binding: GraphBindingId,
    sequence: SequenceKey,
    identity: u64,
    generation: u64,
    next: usize,
    ids: Zeroizing<[BufferId; N]>,
    public: Public,
    poison: Arc<AtomicBool>,
    pending: bool,
    public_grouped_reads: bool,
    public_four_rows: bool,
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

pub(crate) struct LatestPleCheckpoint {
    snapshot: Option<ops::ple_checkpoint::Checkpoint>,
    device_snapshot: Option<core::DeviceHistoryTokenCheckpoint>,
    metadata: Zeroizing<LatestPleMetadata>,
    binding: Option<GraphBindingId>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
#[derive(Clone, PartialEq, Eq)]
struct LatestPleMetadata {
    owner: u64,
    generation: u64,
    sequence: SequenceKey,
    start: usize,
    count: usize,
    ids: [BufferId; N],
    tokens: [u32; crate::strata_port::verify_capacity()],
    old_tokens: [i32; 2],
}
impl Zeroize for LatestPleMetadata {
    fn zeroize(&mut self) {
        self.owner.zeroize();
        self.generation.zeroize();
        self.sequence.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.ids.zeroize();
        self.tokens.zeroize();
        self.old_tokens.zeroize();
    }
}
impl PleBatchWorkspace {
    pub(crate) fn native_capture_history_role(&self) -> Result<u8> {
        self.ids[OLD]
            .original_pair_role(self.ids[NEXT])
            .map(u8::from)
            .ok_or(error())
    }
    pub(crate) fn checkpoint_latest(
        &self,
        model: &ResidentModel<'_, '_>,
        tokens: &[u32],
        cancel: &AtomicBool,
    ) -> Result<LatestPleCheckpoint> {
        self.checkpoint_latest_with_four_rows(model, tokens, false, cancel)
    }
    pub(crate) fn checkpoint_latest_with_four_rows(
        &self,
        model: &ResidentModel<'_, '_>,
        tokens: &[u32],
        four: bool,
        cancel: &AtomicBool,
    ) -> Result<LatestPleCheckpoint> {
        self.healthy(model, cancel)?;
        graph_preflight(model.graph)?;
        if !(1..=if four {
            crate::strata_port::verify_capacity()
        } else {
            3
        })
            .contains(&tokens.len())
            || tokens.iter().any(|&t| t >= 248320)
            || self.pending
            || model.ple_batch_pending.is_some()
            || self.next != model.position
            || model
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > model.context)
        {
            return Err(error());
        }
        self.generation
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        let snapshot = if crate::strata_intel_core::enabled() {
            None
        } else {
            Some(
                ops::ple_checkpoint::Checkpoint::capture(
                    &model.ple_conv,
                    &model.ple_history,
                    ops::ple_checkpoint::MAX_OWNED_BYTES,
                )
                .map_err(u::fault)?,
            )
        };
        let device_snapshot = if crate::strata_intel_core::enabled() {
            Some(
                core::DeviceHistoryTokenCheckpoint::capture(&model.ple_history)
                    .map_err(u::fault)?,
            )
        } else {
            None
        };
        let mut packet = Zeroizing::new([0; crate::strata_port::verify_capacity()]);
        packet[..tokens.len()].copy_from_slice(tokens);
        Ok(LatestPleCheckpoint {
            snapshot,
            device_snapshot,
            metadata: Zeroizing::new(LatestPleMetadata {
                owner: self.identity,
                generation: self.generation,
                sequence: self.sequence,
                start: model.position,
                count: tokens.len(),
                ids: *self.ids,
                tokens: *packet,
                old_tokens: [model.ple_history.tokens()[0], model.ple_history.tokens()[1]],
            }),
            binding: Some(self.binding),
            poison: self.poison.clone(),
            armed: true,
        })
    }
}
impl LatestPleCheckpoint {
    pub(crate) fn start(&self) -> usize {
        self.metadata.start
    }
    pub(crate) fn count(&self) -> usize {
        self.metadata.count
    }
    pub(crate) fn binding(&self) -> Option<GraphBindingId> {
        self.binding
    }
    pub(crate) fn owned_bytes(&self) -> Result<usize> {
        let payload = if self.device_snapshot.is_some() && self.snapshot.is_none() {
            std::mem::size_of::<core::DeviceHistoryTokenCheckpoint>()
        } else {
            self.snapshot
                .as_ref()
                .ok_or(error())?
                .owned_bytes()
                .map_err(u::fault)?
        };
        payload
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(Error::ArithmeticOverflow)
    }
    #[allow(clippy::too_many_arguments)]
    fn validate_parts(
        &self,
        workspace: &PleBatchWorkspace,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        position: usize,
        conv: &CausalConvState,
        history: &PleTokenHistory,
        pending: bool,
        poisoned: bool,
        retired: &AtomicBool,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        graph_preflight(graph)?;
        workspace.validate_ids(arena)?;
        let m = &self.metadata;
        let mut expected_ids = m.ids;
        if crate::strata_intel_core::enabled() {
            expected_ids.swap(OLD, NEXT);
        }
        if !self.armed
            || pending
            || workspace.pending
            || poisoned
            || retired.load(Ordering::Acquire)
            || self.poison.load(Ordering::Acquire)
            || !Arc::ptr_eq(&self.poison, &workspace.poison)
            || self.binding != Some(graph.binding_id())
            || workspace.binding != graph.binding_id()
            || workspace.identity != m.owner
            || workspace.sequence != m.sequence
            || *workspace.ids != expected_ids
            || workspace.generation
                != m.generation
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?
            || position
                != m.start
                    .checked_add(m.count)
                    .ok_or(Error::ArithmeticOverflow)?
            || workspace.next != position
        {
            return Err(error());
        }
        let mut next_tokens = Zeroizing::new(m.old_tokens);
        for &token in &m.tokens[..m.count] {
            next_tokens[0] = next_tokens[1];
            next_tokens[1] = token as i32;
        }
        if history.tokens() != *next_tokens {
            return Err(error());
        }
        if crate::strata_intel_core::enabled() {
            self.device_snapshot
                .as_ref()
                .ok_or(error())?
                .preflight(history, graph.ple_hash(), &m.tokens[..m.count], cancel)
                .map_err(u::fault)
        } else {
            self.snapshot
                .as_ref()
                .ok_or(error())?
                .preflight(conv, history, cancel)
                .map_err(u::fault)
        }
    }
    pub(crate) fn validate_completed(
        &self,
        workspace: &PleBatchWorkspace,
        model: &ResidentModel<'_, '_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_parts(
            workspace,
            model.graph,
            &model.arena,
            model.position,
            &model.ple_conv,
            &model.ple_history,
            model.ple_batch_pending.is_some(),
            model.poisoned,
            &model.retired,
            cancel,
        )
    }
    pub(crate) fn prepare_restore<'a>(
        mut self,
        workspace: &'a mut PleBatchWorkspace,
        c: Preparation<'a, '_, '_>,
        cancel: &AtomicBool,
    ) -> Result<LatestPleRestore<'a>> {
        self.validate_parts(
            workspace,
            c.graph,
            c.arena,
            c.position,
            c.conv,
            c.history,
            c.anchor.is_some(),
            *c.owner_poisoned,
            c.retired,
            cancel,
        )?;
        let restore = if crate::strata_intel_core::enabled() {
            device_history::Restore::Device(
                self.device_snapshot
                    .as_ref()
                    .ok_or(error())?
                    .prepare_prefix(
                        c.history,
                        c.graph.ple_hash(),
                        &self.metadata.tokens[..self.metadata.count],
                        0,
                        cancel,
                    )
                    .map_err(u::fault)?,
            )
        } else {
            device_history::Restore::Legacy(
                self.snapshot
                    .take()
                    .ok_or(error())?
                    .prepare_restore(c.conv, c.history, cancel)
                    .map_err(u::fault)?,
            )
        };
        Ok(LatestPleRestore {
            checkpoint: self,
            workspace,
            restore,
            accepted: 0,
        })
    }
    pub(crate) fn prepare_prefix<'a>(
        mut self,
        workspace: &'a mut PleBatchWorkspace,
        c: Preparation<'a, '_, '_>,
        accepted: usize,
        cancel: &AtomicBool,
    ) -> Result<LatestPleRestore<'a>> {
        self.validate_parts(
            workspace,
            c.graph,
            c.arena,
            c.position,
            c.conv,
            c.history,
            c.anchor.is_some(),
            *c.owner_poisoned,
            c.retired,
            cancel,
        )?;
        if !crate::strata_port::enabled() || accepted == 0 || accepted >= self.metadata.count {
            return Err(error());
        }
        let restore = if crate::strata_intel_core::enabled() {
            device_history::Restore::Device(
                self.device_snapshot
                    .as_ref()
                    .ok_or(error())?
                    .prepare_prefix(
                        c.history,
                        c.graph.ple_hash(),
                        &self.metadata.tokens[..self.metadata.count],
                        accepted,
                        cancel,
                    )
                    .map_err(u::fault)?,
            )
        } else {
            device_history::Restore::Legacy(
                self.snapshot
                    .take()
                    .ok_or(error())?
                    .prepare_accepted_prefix(
                        c.conv,
                        c.history,
                        &self.metadata.tokens[..self.metadata.count],
                        accepted,
                        cancel,
                    )
                    .map_err(u::fault)?,
            )
        };
        Ok(LatestPleRestore {
            checkpoint: self,
            workspace,
            restore,
            accepted,
        })
    }
    pub(crate) fn accept_completed(
        mut self,
        workspace: &PleBatchWorkspace,
        model: &ResidentModel<'_, '_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_completed(workspace, model, cancel)?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for LatestPleCheckpoint {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release);
        }
        self.armed.zeroize();
        self.binding = None;
    }
}
pub(crate) struct LatestPleRestore<'a> {
    checkpoint: LatestPleCheckpoint,
    workspace: &'a mut PleBatchWorkspace,
    restore: device_history::Restore<'a>,
    accepted: usize,
}
impl<'a> LatestPleRestore<'a> {
    pub(crate) fn enqueue(
        self,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<device_history::QueuedRestore<'a>> {
        device_history::enqueue_restore(self, arena, fault, cancel)
    }
}

pub(crate) struct PublicWarmPleReset<'a> {
    workspace: &'a mut PleBatchWorkspace,
    conv: &'a mut CausalConvState,
    history: &'a mut PleTokenHistory,
    armed: bool,
}
impl PleBatchWorkspace {
    /// Pins the actual host convolution/history and native cursor. No private
    /// snapshot is copied and no generation, sequence, binding or IDs change.
    pub(crate) fn prepare_public_warm_reset<'a>(
        &'a mut self,
        c: Preparation<'a, '_, '_>,
        cancel: &AtomicBool,
    ) -> Result<PublicWarmPleReset<'a>> {
        u::cancel(cancel)?;
        graph_preflight(c.graph)?;
        self.validate_ids(c.arena)?;
        c.admission
            .check_client(c.arena.client().map_err(u::fault)?)?;
        if c.position != 13
            || c.context != 13
            || self.next != c.position
            || self.generation == 0
            || self.pending
            || c.anchor.is_some()
            || *c.owner_poisoned
            || c.retired.load(Ordering::Acquire)
            || self.poison.load(Ordering::Acquire)
            || self.binding != c.graph.binding_id()
            || c.sequence_poison
                .as_ref()
                .is_none_or(|p| !Arc::ptr_eq(p, &self.poison))
            || c.conv.history().len() != CHANNELS * 9
            || c.conv.history().iter().any(|x| !x.is_finite())
            || c.history.tokens().len() != 2
            || c.history
                .tokens()
                .iter()
                .any(|&t| !(0..248320).contains(&t))
        {
            return Err(error());
        }
        Ok(PublicWarmPleReset {
            workspace: self,
            conv: c.conv,
            history: c.history,
            armed: false,
        })
    }
}
impl PublicWarmPleReset<'_> {
    pub(crate) fn arm(&mut self) {
        self.armed = true;
    }
    /// Complete parent GPU/source/I/O erasure and final cancellation precede
    /// this infallible full-capacity host wipe and cursor-only publication.
    pub(crate) fn commit(
        mut self,
        _proof: &crate::resident_model::batch::public_warm_replay::CompletedReset,
    ) {
        self.conv.reset();
        self.history.reset(); // Original initial sentinel [-1,-1], not [0,0].
        self.workspace.next = 0;
        self.armed = false;
    }
}
impl Drop for PublicWarmPleReset<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.workspace.poison.store(true, Ordering::Release);
        }
    }
}
impl PleBatchWorkspace {
    pub(crate) fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; N]> {
        Self::allocation_requests_with_profile(graph, OriginalPackedProfile::Q4)
    }
    pub(crate) fn allocation_requests_with_profile(
        graph: BoundGraph<'_, '_>,
        profile: OriginalPackedProfile,
    ) -> Result<[u64; N]> {
        graph_preflight(graph)?;
        if profile == OriginalPackedProfile::UnslothQ3 {
            profile.validate_graph(graph)?;
        }
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
        Self::allocate_with_profile(model, sequence, poison, cancel, OriginalPackedProfile::Q4)
    }
    pub(crate) fn allocate_with_profile(
        model: &mut ResidentModel<'_, '_>,
        sequence: SequenceKey,
        poison: Arc<AtomicBool>,
        cancel: &AtomicBool,
        profile: OriginalPackedProfile,
    ) -> Result<Self> {
        let result = guarded(&poison, || {
            u::cancel(cancel)?;
            graph_preflight(model.graph)?;
            if profile == OriginalPackedProfile::UnslothQ3 {
                profile.validate_graph(model.graph)?;
            }
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
                profile,
                binding: model.graph.binding_id(),
                sequence,
                identity,
                generation: 0,
                next: model.position,
                ids,
                public,
                poison: poison.clone(),
                pending: false,
                public_grouped_reads: false,
                public_four_rows: false,
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
    pub(super) fn validate_unstarted_projection_boundary(
        &self,
        model: &ResidentModel<'_, '_>,
    ) -> Result<()> {
        self.healthy(model, &AtomicBool::new(false))?;
        unstarted_projection_cursor(self.next, model.position, self.generation, self.pending)?;
        if self.public_grouped_reads || self.public_four_rows {
            return Err(error());
        }
        Ok(())
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
    /// The native target owner alone selects this immutable public policy at a
    /// completed boundary. No GR arithmetic flag is modified.
    pub(super) fn select_public_four_rows(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<()> {
        if self.public_four_rows || self.pending || self.poison.load(Ordering::Acquire) {
            return Err(error());
        }
        self.public_four_rows = true;
        Ok(())
    }
    pub(super) fn select_public_grouped_reads(
        &mut self,
        position: usize,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<()> {
        completed_projection_cursor(self.next, position, self.pending)?;
        if self.poison.load(Ordering::Acquire) {
            return Err(error());
        }
        self.public_grouped_reads = true;
        Ok(())
    }
    pub(super) fn public_grouped_reads(&self) -> bool {
        self.public_grouped_reads
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
        if projector.global_fault() != model.b.fault {
            return Err(error());
        }
        let prepared = self.prepare_invocation_checked(
            model, input, output, spans, start, tokens, positions, cancel,
        )?;
        self.enqueue_prepared_checked(model, projector, input, output, spans, prepared, cancel)
    }
    /// Bind the exact original history/input owners, fetch ngram rows and stage
    /// both private upload buffers before the target's captured math begins.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_invocation(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: usize,
        tokens: &Zeroizing<[u32; MAX_BATCH]>,
        positions: &Zeroizing<[u32; MAX_BATCH]>,
        cancel: &AtomicBool,
    ) -> Result<PreparedPleBatch> {
        let poison = self.poison.clone();
        let result = guarded(&poison, || {
            self.prepare_invocation_checked(
                model, input, output, spans, start, tokens, positions, cancel,
            )
        });
        if result.is_err() {
            model.poisoned = true;
        }
        result
    }
    /// Consume the staged invocation exactly once. No OLD/EMB upload or source
    /// lookup occurs inside this math-only portion of the captured branch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_prepared(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        prepared: PreparedPleBatch,
        cancel: &AtomicBool,
    ) -> Result<PendingPleBatch> {
        let poison = self.poison.clone();
        let result = guarded(&poison, || {
            self.enqueue_prepared_checked(model, projector, input, output, spans, prepared, cancel)
        });
        if result.is_err() {
            model.poisoned = true;
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_invocation_checked(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        start: usize,
        tokens: &Zeroizing<[u32; MAX_BATCH]>,
        positions: &Zeroizing<[u32; MAX_BATCH]>,
        cancel: &AtomicBool,
    ) -> Result<PreparedPleBatch> {
        self.healthy(model, cancel)?;
        graph_preflight(model.graph)?;
        let count = spans.count();
        token_packet(count, start, model.context, tokens, positions)?;
        if self.pending
            || model.ple_batch_pending.is_some()
            || start != self.next
            || start != model.position
            || spans.packed_first() != 0
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
        let resident_history = crate::strata_intel_core::enabled();
        let mut old = if resident_history {
            Zeroizing::new(Vec::new())
        } else {
            let mut old = source(BufferLayout::new(words(OLD)).map_err(u::fault)?.owned_bytes)?;
            for (bytes, &v) in old.chunks_exact_mut(4).zip(model.ple_conv.history()) {
                bytes.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            old
        };
        let mut embedding = embedding(
            model.graph,
            count,
            start,
            tokens,
            &old_tokens,
            cancel,
            self.profile,
        )?;
        let arena = &model.arena;
        queue_preparation(cancel, resident_history, |operation| {
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
                _ => return Err(error()),
            }
            Ok(())
        })?;
        Ok(PreparedPleBatch {
            receipt: PendingPleBatch {
                metadata,
                poison: self.poison.clone(),
                armed: true,
            },
        })
    }
    pub(crate) fn complete_prepared_replay(
        &mut self,
        model: &ResidentModel<'_, '_>,
        prepared: PreparedPleBatch,
        completed: &crate::resident_model::target_replay::Completed,
        cancel: &AtomicBool,
    ) -> Result<PendingPleBatch> {
        // Receipt Drop remains armed on every failure, poisoning its original
        // sequence. EMB was staged; core OLD is its actual committed GPU owner
        // and legacy OLD was uploaded before the graph submission.
        let receipt = prepared.receipt;
        self.receipt(model, &receipt, cancel)?;
        completed.check(
            model.b.fault,
            receipt.metadata.start as u64,
            receipt.metadata.count,
        )?;
        completed.check_fault_owner(&model.arena, model.b.fault)?;
        if !crate::strata_intel_core::enabled() || crate::strata_prompt::active() {
            return Err(error());
        }
        Ok(receipt)
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_prepared_checked(
        &mut self,
        model: &mut ResidentModel<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        prepared: PreparedPleBatch,
        cancel: &AtomicBool,
    ) -> Result<PendingPleBatch> {
        let receipt = prepared.receipt;
        self.receipt(model, &receipt, cancel)?;
        let metadata = &*receipt.metadata;
        let count = spans.count();
        let start = metadata.start;
        if projector.global_fault() != model.b.fault
            || spans.packed_first() != 0
            || metadata.input != input
            || metadata.output != output
            || metadata.count != count
            || metadata.first != [spans.input_first(), spans.output_first()]
            || metadata.stride != [spans.input_stride(), spans.output_stride()]
        {
            return Err(error());
        }
        let arena = &model.arena;
        queue_compute(cancel, |operation| {
            match operation {
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
                Operation::Clear(_) | Operation::Upload(_) => return Err(error()),
            }
            Ok(())
        })?;
        Ok(receipt)
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
    /// audited I/O cleanup (or an explicit request-owned checkpoint); caller
    /// must preflight every OTHER family before ANY
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
                admission: &model.admission,
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
        if crate::strata_intel_core::enabled() {
            return self.prepare_device_history_after_whole_model_fence(c, receipt, cancel);
        }
        let poison = self.poison.clone();
        let diagnostics = guarded(&poison, || {
            self.receipt_parts(&c, &receipt, cancel)?;
            let m = &*receipt.metadata;
            let public_grouped_read = self.public_grouped_reads;
            if public_grouped_read
                && !(1..=if self.public_four_rows {
                    crate::strata_port::verify_capacity()
                } else {
                    3
                })
                    .contains(&m.count)
            {
                return Err(error());
            }
            let (old, normalized, next) = if public_grouped_read
                || crate::next_shared::gr_selected()
            {
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
            // Keep this independent PLE boundary in the SAME request scope as
            // logits/fault. A direct sanitizer here would erase that pool on
            // every token. Reuse retains every dirty owner/charge, accepts GPU
            // errors and maps now, and performs the full audit at request end.
            if !c.admission.checkpoint_request_io_reuse(cancel)? {
                audit(c.arena, c.retired)?;
            }
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
            Ok((old, normalized, next, public_grouped_read))
        });
        let (old, normalized, next, public_grouped_read) = match diagnostics {
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
                transition: Some(device_history::Commit::Legacy(transition)),
                receipt,
                armed: true,
                public_grouped_read,
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
    pub(super) admission: &'a crate::executor::ResidentAdmission,
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
    transition: Option<device_history::Commit<'a>>,
    receipt: PendingPleBatch,
    armed: bool,
    public_grouped_read: bool,
}
impl PleBatchAcceptance<'_> {
    /// Actual selected read_full_three execution, after full diagnostics and
    /// transition preparation. The coordinator credits it only after commit.
    pub(super) fn public_grouped_read_completed(&self) -> bool {
        self.public_grouped_read
    }
    /// All-family preflight BEFORE any state swap. No CPU histories can change
    /// concurrently: the core ticket retains their exclusive mutable borrows.
    pub(crate) fn preflight(&self, cancel: &AtomicBool) -> Result<()> {
        guarded(&self.workspace.poison, || {
            if self.workspace.poison.load(Ordering::Acquire)
                || self.retired.load(Ordering::Acquire)
                || !self.armed
                || self.public_grouped_read != self.workspace.public_grouped_reads
            {
                return Err(error());
            }
            self.workspace.validate_ids(self.arena)?;
            self.transition.as_ref().ok_or(error())?.preflight(cancel)
        })
    }
    /// Requires ALL family permits preflighted before the first commit. Exact
    /// active count updates this PLE cursor only; model cursor remains the
    /// future whole-model coordinator's responsibility. No fallible swap work.
    pub(crate) fn commit(mut self, cancel: &AtomicBool) -> Result<()> {
        self.preflight(cancel)?;
        if let Some(transition) = self.transition.take() {
            let device = transition.device();
            transition.commit();
            if device {
                self.workspace.ids.swap(OLD, NEXT);
            }
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
    let fixed = 368640
        + 524288
        + 131072
        + 3 * 524288
        + 524288
        + 4096
        + 65536
        + if crate::strata_intel_core::enabled() {
            4 * 128 * 1024 + 65536 + prompt_source::HOST_BYTES
        } else {
            0
        };
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
    start: usize,
    tokens: &[u32; MAX_BATCH],
    old: &[i32; 2],
    cancel: &AtomicBool,
    profile: OriginalPackedProfile,
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
    profile.validate_ple(
        table.info().dtype(),
        table.info().dimensions(),
        table.info().row_bytes(),
    )?;
    if crate::strata_intel_core::enabled() {
        if let Some(ready) = prompt_source::take(graph, &rows[..count], start)? {
            u::cancel(cancel)?;
            return Ok(ready);
        }
        return native_embedding(graph, table, profile, &rows[..count], cancel);
    }
    let mut upload = source(BufferLayout::new(words(EMB)).map_err(u::fault)?.owned_bytes)?;
    let mut packed = Zeroizing::new([0u8; 110]);
    let mut decoded = Zeroizing::new([0f32; 160]);
    for t in 0..count {
        for head in 0..16 {
            u::cancel(cancel)?;
            let row = &mut packed[..profile.ple_row_bytes()];
            table.read_row(u64::from(rows[t][head]), row)?;
            decode_ple_row(profile, table.info().dtype(), row, &mut decoded[..])?;
            for (d, &v) in decoded.iter().enumerate() {
                let at = (t * WIDTH + head * 160 + d) * 4;
                upload[at..at + 4].copy_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    graph.verify_unchanged()?;
    Ok(upload)
}

// Stable read plan contains request-derived row IDs only inside wiping memory.
// One positional read per unique row; destinations retain original token/head
// order regardless of sorted read order or worker completion order.
struct RowReadPlan {
    unique: Zeroizing<Vec<u32>>,
    slots: Zeroizing<Vec<usize>>,
}
fn unique_row_plan(rows: &[[u32; 16]]) -> Result<RowReadPlan> {
    if rows.is_empty()
        || rows.len()
            > if crate::strata_intel_core::enabled() {
                crate::strata_prompt::groups::CAPACITY
            } else {
                MAX_BATCH
            }
    {
        return Err(error());
    }
    let mut ordered = Zeroizing::new(Vec::with_capacity(rows.len() * 16));
    for (t, heads) in rows.iter().enumerate() {
        for (h, &row) in heads.iter().enumerate() {
            if row >= 320001536 {
                return Err(error());
            }
            ordered.push((row, t * 16 + h));
        }
    }
    ordered.sort_unstable();
    let mut unique = Zeroizing::new(Vec::with_capacity(ordered.len()));
    let mut slots = Zeroizing::new(vec![0usize; ordered.len()]);
    for &(row, at) in ordered.iter() {
        if unique.last() != Some(&row) {
            unique.push(row);
        }
        slots[at] = unique.len() - 1;
    }
    Ok(RowReadPlan { unique, slots })
}
fn native_embedding(
    graph: BoundGraph<'_, '_>,
    table: kuno_text::TensorView<'_>,
    profile: OriginalPackedProfile,
    rows: &[[u32; 16]],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<u8>>> {
    let RowReadPlan { unique, slots } = unique_row_plan(rows)?;
    let rb = profile.ple_row_bytes();
    if rb != table.info().row_bytes() as usize {
        return Err(error());
    }
    let mut raw = source(unique.len() * rb)?;
    let reader = table.positional_reader()?;
    graph.verify_unchanged()?;
    // Four finite scoped workers, not per-row jobs. Each has a96 KiB requested
    // stack plus32 KiB runtime allowance, charged with raw/plan storage above.
    // All join before this borrow returns, including cancellation/error paths.
    let per_worker = unique.len().div_ceil(4);
    std::thread::scope(|scope| -> Result<()> {
        let mut jobs = Vec::with_capacity(4);
        let mut failure = None;
        for (worker, bytes) in raw.chunks_mut(per_worker * rb).enumerate() {
            let ids = &unique
                [worker * per_worker..(worker * per_worker + bytes.len() / rb).min(unique.len())];
            let reader = &reader;
            let job = std::thread::Builder::new()
                .stack_size(96 * 1024)
                .spawn_scoped(scope, move || -> Result<()> {
                    for (&row, output) in ids.iter().zip(bytes.chunks_exact_mut(rb)) {
                        u::cancel(cancel)?;
                        reader.read_row(table, u64::from(row), output)?;
                    }
                    u::cancel(cancel)
                });
            match job {
                Ok(job) => jobs.push(job),
                Err(_) => {
                    failure = Some(error());
                    break;
                }
            }
        }
        for job in jobs {
            match job.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if failure.is_none() {
                        failure = Some(e);
                    }
                }
                Err(_) => {
                    if failure.is_none() {
                        failure = Some(error());
                    }
                }
            }
        }
        if let Some(e) = failure {
            Err(e)
        } else {
            Ok(())
        }
    })?;
    graph.verify_unchanged()?;
    u::cancel(cancel)?;
    let output_bytes = if rows.len() > MAX_BATCH {
        rows.len() * WIDTH * 4
    } else {
        BufferLayout::new(words(EMB)).map_err(u::fault)?.owned_bytes
    };
    let mut upload = source(output_bytes)?;
    let mut decoded = Zeroizing::new([0f32; 160]);
    // Scatter by the stable destination plan. Each packed row is still read
    // once, but do not scan every destination for every unique row (quadratic
    // at2048 tokens). Re-decoding a shared row preserves exact original bytes
    // and uses the same single wiping160-float scratch.
    for (destination, &slot) in slots.iter().enumerate() {
        u::cancel(cancel)?;
        decode_ple_row(
            profile,
            table.info().dtype(),
            &raw[slot * rb..(slot + 1) * rb],
            &mut decoded[..],
        )?;
        let at = destination * 160 * 4;
        for (bytes, &v) in upload[at..at + 160 * 4]
            .chunks_exact_mut(4)
            .zip(decoded.iter())
        {
            bytes.copy_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    graph.verify_unchanged()?;
    Ok(upload)
}

fn decode_ple_row(
    profile: OriginalPackedProfile,
    dtype: kuno_text::TensorType,
    packed: &[u8],
    decoded: &mut [f32],
) -> Result<()> {
    // This is selected-row reconstruction for the existing transient embedding
    // upload, not a retained float mirror of the original table or model.
    profile.validate_ple(dtype, &[160, 320001536], packed.len() as u64)?;
    if decoded.len() != 160 {
        return Err(error());
    }
    dtype.decode(packed, decoded)?;
    if decoded.iter().any(|x| !x.is_finite()) {
        return Err(error());
    }
    Ok(())
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
fn queue_preparation(
    cancel: &AtomicBool,
    resident_history: bool,
    mut f: impl FnMut(Operation) -> Result<()>,
) -> Result<()> {
    for op in (0..N)
        .filter(|&i| !resident_history || i != OLD)
        .map(Operation::Clear)
        .chain(
            (0..2)
                .filter(|&i| !resident_history || i != 0)
                .map(Operation::Upload),
        )
    {
        u::cancel(cancel)?;
        f(op)?;
    }
    u::cancel(cancel)
}
fn queue_compute(cancel: &AtomicBool, mut f: impl FnMut(Operation) -> Result<()>) -> Result<()> {
    for op in (0..2)
        .map(Operation::Projection)
        .chain([Operation::Tail])
        .chain((0..7).map(Operation::Status))
        .chain([Operation::History, Operation::Finite])
    {
        u::cancel(cancel)?;
        f(op)?;
    }
    u::cancel(cancel)
}
#[cfg(test)]
fn queue(cancel: &AtomicBool, mut f: impl FnMut(Operation) -> Result<()>) -> Result<()> {
    queue_preparation(cancel, false, &mut f)?;
    queue_compute(cancel, f)
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
    if crate::strata_intel_core::enabled() {
        return native_tail(arena, ids, public, input, output, spans, cancel);
    }
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

#[allow(clippy::too_many_arguments)]
fn native_tail(
    arena: &PrivateArena,
    ids: &[BufferId; N],
    public: &Public,
    input: BufferId,
    output: BufferId,
    spans: &BatchSpans,
    cancel: &AtomicBool,
) -> Result<()> {
    use crate::strata_ple as n;
    let h = |i| u::handle(arena, ids[i]);
    let count = spans.count();
    let info = |ib: usize, is: usize, ob: usize, os: usize, ss: usize, epsilon: f32| {
        [
            count as u32,
            ib as u32,
            is as u32,
            ob as u32,
            os as u32,
            ss as u32,
            0,
            epsilon.to_bits(),
            0,
            0,
            0,
            0,
        ]
    };
    for (source, ib, is, target, status, gamma) in [
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
        n::launch(
            arena,
            n::Kernel::Norm,
            [4, count, 1],
            vec![
                u::handle(arena, source)?,
                public.gamma[gamma].clone(),
                h(target)?,
                h(status)?,
            ],
            info(ib, is, 0, CHANNELS, 4, 1e-6),
        )?;
    }
    u::cancel(cancel)?;
    n::launch(
        arena,
        n::Kernel::Gate,
        [4, count, 1],
        vec![h(KEY_NORM)?, h(QUERY_NORM)?, h(GATES)?, h(13)?],
        info(0, CHANNELS, 0, 4, 4, (WIDTH as f32).sqrt().recip()),
    )?;
    n::launch(
        arena,
        n::Kernel::Broadcast,
        [40, count, 1],
        vec![h(VALUE)?, h(GATES)?, h(GATED)?, h(14)?],
        info(0, 0, 0, 0, 0, 0.),
    )?;
    n::launch(
        arena,
        n::Kernel::Norm,
        [4, count, 1],
        vec![h(GATED)?, public.gamma[2].clone(), h(NORMALIZED)?, h(15)?],
        info(0, CHANNELS, 0, CHANNELS, 4, 1e-6),
    )?;
    u::cancel(cancel)?;
    n::launch(
        arena,
        n::Kernel::Conv,
        [40, count, 1],
        vec![
            h(OLD)?,
            h(NORMALIZED)?,
            public.conv.clone(),
            u::handle(arena, input)?,
            h(GATED)?,
            h(CONV)?,
            u::handle(arena, output)?,
            h(16)?,
            h(17)?,
        ],
        info(
            spans.input_first(),
            spans.input_stride(),
            spans.output_first(),
            spans.output_stride(),
            0,
            0.,
        ),
    )?;
    n::launch(
        arena,
        n::Kernel::History,
        [40, 1, 1],
        vec![h(OLD)?, h(NORMALIZED)?, h(NEXT)?, h(18)?],
        info(0, 0, 0, 0, 0, 0.),
    )?;
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
fn grouped_read_host_bound(existing: usize, raw_bytes: usize) -> Result<()> {
    if existing
        .checked_add(raw_bytes.checked_mul(2).ok_or(Error::ArithmeticOverflow)?)
        .and_then(|n| n.checked_add(65536))
        .is_none_or(|n| n > HOST_CAP)
    {
        return Err(error());
    }
    Ok(())
}
fn decode_three(raw: &ReadBatch, count: usize) -> Result<[Zeroizing<Vec<f32>>; 3]> {
    if raw.0.len() != 3 {
        return Err(error());
    }
    let old = decode_download(&raw.0[0], words(OLD), count, false)?;
    let normalized = decode_download(&raw.0[1], words(NORMALIZED), count, true)?;
    let next = decode_download(&raw.0[2], words(NEXT), count, false)?;
    Ok([old, normalized, next])
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
    // conversion. Same 4MiB cap, including 64KiB original scoped metadata reserve.
    grouped_read_host_bound(existing, raw_bytes)?;
    let raw = ReadBatch(
        cubecl::future::block_on(arena.client().map_err(u::fault)?.read_async(handles))
            .map_err(u::fault)?,
    );
    // Guard wipes every raw map before the caller's original audited cleanup.
    decode_three(&raw, count)
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
    fn native_deduplicated_reads_preserve_every_token_head_destination() {
        let capacity = if crate::strata_intel_core::enabled() {
            crate::strata_prompt::groups::CAPACITY
        } else {
            MAX_BATCH
        };
        assert_eq!(MAX_BATCH, 8); // The speculative verifier has not widened.
        for count in 1..=capacity {
            let rows = (0..count)
                .map(|t| std::array::from_fn(|h| ((t * 7 + h * 3) % 19) as u32))
                .collect::<Vec<_>>();
            let RowReadPlan { unique, slots } = unique_row_plan(&rows).unwrap();
            assert!(unique.windows(2).all(|p| p[0] < p[1]));
            assert_eq!(slots.len(), count * 16);
            for t in 0..count {
                for h in 0..16 {
                    assert_eq!(unique[slots[t * 16 + h]], rows[t][h]);
                }
            }
            assert!(unique.len() <= count * 16);
        }
        let RowReadPlan { unique, slots } =
            unique_row_plan(&[[320001535u32; 16]; MAX_BATCH]).unwrap();
        assert_eq!(&*unique, &[320001535]);
        assert!(slots.iter().all(|&v| v == 0));
        assert!(unique_row_plan(&[]).is_err());
        assert!(unique_row_plan(&vec![[0; 16]; capacity + 1]).is_err());
        assert!(unique_row_plan(&[[320001536; 16]]).is_err());
    }
    #[test]
    fn grouped_ple_raw_plus_decoded_caps_keep_original_four_mib_and_overflow_rejection() {
        let raw: usize = [OLD, NORMALIZED, NEXT]
            .into_iter()
            .map(|i| BufferLayout::new(words(i)).unwrap().owned_bytes)
            .sum();
        assert_eq!(raw, 3 * 524288);
        let maximum_existing = HOST_CAP - 2 * raw - 65536;
        grouped_read_host_bound(core::HISTORY_WORDS * 4 + 8, raw).unwrap();
        grouped_read_host_bound(maximum_existing, raw).unwrap();
        assert!(grouped_read_host_bound(maximum_existing + 1, raw).is_err());
        assert!(grouped_read_host_bound(0, usize::MAX).is_err());
        assert!(grouped_read_host_bound(usize::MAX, raw).is_err());
    }
    #[test]
    fn grouped_ple_actual_three_map_decoder_preserves_all_bits_and_rejects_every_owner_tail() {
        for count in 1..=3 {
            let logical = [words(OLD), words(NORMALIZED), words(NEXT)];
            let make = |part: usize| {
                let mut bytes = vec![0; BufferLayout::new(logical[part]).unwrap().owned_bytes];
                let active = if part == 1 {
                    count * CHANNELS
                } else {
                    logical[part]
                };
                for i in 0..active {
                    let value = if i % 19 == 0 {
                        -0.0f32
                    } else {
                        (part * 101 + i % 107) as f32 / 113.0
                    };
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
                }
                bytes
            };
            let originals: [Vec<u8>; 3] = std::array::from_fn(make);
            let batch = ReadBatch(
                originals
                    .iter()
                    .cloned()
                    .map(cubecl::bytes::Bytes::from_bytes_vec)
                    .collect(),
            );
            let grouped = decode_three(&batch, count).unwrap();
            for part in 0..3 {
                let separate =
                    decode_download(&originals[part], logical[part], count, part == 1).unwrap();
                assert_eq!(grouped[part].len(), 131072);
                assert!(
                    grouped[part]
                        .iter()
                        .zip(separate.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            // Every full owner has exact finite active words and zero unused
            // words, even though count 1/2/3 reads the same 524KiB capacity.
            for part in 0..3 {
                for (index, bits) in [(0, f32::NAN.to_bits()), (131071, 1u32)] {
                    let mut invalid = originals.clone();
                    invalid[part][index * 4..index * 4 + 4].copy_from_slice(&bits.to_le_bytes());
                    let invalid = ReadBatch(
                        invalid
                            .into_iter()
                            .map(cubecl::bytes::Bytes::from_bytes_vec)
                            .collect(),
                    );
                    assert!(decode_three(&invalid, count).is_err());
                }
            }
            let mut inactive = originals.clone();
            inactive[1][count * CHANNELS * 4..count * CHANNELS * 4 + 4]
                .copy_from_slice(&(-0.0f32).to_bits().to_le_bytes());
            assert!(
                decode_three(
                    &ReadBatch(
                        inactive
                            .into_iter()
                            .map(cubecl::bytes::Bytes::from_bytes_vec)
                            .collect()
                    ),
                    count
                )
                .is_err()
            );
        }
        assert!(decode_three(&ReadBatch(Vec::new()), 1).is_err());
    }
    #[test]
    fn native_shared_unstarted_boundary_is_distinct_from_a_committed_cursor() {
        assert!(unstarted_projection_cursor(0, 0, 0, false).is_ok());
        assert!(completed_projection_cursor(0, 0, false).is_err());
        for (next, position, generation, pending) in [
            (1, 0, 0, false),
            (0, 1, 0, false),
            (1, 1, 0, false),
            (0, 0, 1, false),
            (0, 0, 0, true),
            (0, 0, u64::MAX, false),
        ] {
            assert!(unstarted_projection_cursor(next, position, generation, pending).is_err());
        }
        assert!(completed_projection_cursor(1, 1, false).is_ok());
    }

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
    fn q3_ple_decodes_only_the_original_90_byte_row_without_reusing_q5_geometry() {
        let mut storage = [0xa5u8; 110];
        for block in storage[..90].chunks_exact_mut(18) {
            block.fill(0);
            block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let mut decoded = [0.; 160];
        decode_ple_row(
            OriginalPackedProfile::UnslothQ3,
            kuno_text::TensorType::Iq4Nl,
            &storage[..90],
            &mut decoded,
        )
        .unwrap();
        // Code zero is the pinned original nonlinear grid value -127; unit d.
        assert_eq!(decoded, [-127.; 160]);
        assert_eq!(&storage[90..], &[0xa5; 20]);
        for (profile, dtype, length) in [
            (OriginalPackedProfile::Q4, kuno_text::TensorType::Iq4Nl, 90),
            (
                OriginalPackedProfile::UnslothQ3,
                kuno_text::TensorType::Q5_0,
                110,
            ),
            (
                OriginalPackedProfile::UnslothQ3,
                kuno_text::TensorType::Iq4Nl,
                110,
            ),
            (
                OriginalPackedProfile::UnslothQ3,
                kuno_text::TensorType::Iq4Nl,
                89,
            ),
        ] {
            decoded.fill(5.);
            assert!(decode_ple_row(profile, dtype, &storage[..length], &mut decoded).is_err());
            assert_eq!(decoded, [5.; 160]);
        }
        assert!(
            decode_ple_row(
                OriginalPackedProfile::UnslothQ3,
                kuno_text::TensorType::Iq4Nl,
                &storage[..90],
                &mut [0.; 159]
            )
            .is_err()
        );
        let mut nonfinite = storage;
        nonfinite[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(
            decode_ple_row(
                OriginalPackedProfile::UnslothQ3,
                kuno_text::TensorType::Iq4Nl,
                &nonfinite[..90],
                &mut decoded
            )
            .is_err()
        );
    }
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
    fn prepared_ple_phase_has_all_uploads_and_math_phase_has_none() {
        let cancel = AtomicBool::new(false);
        let mut staged = Vec::new();
        queue_preparation(&cancel, false, |op| {
            staged.push(op);
            Ok(())
        })
        .unwrap();
        assert_eq!(staged.len(), N + 2);
        assert!(
            staged
                .iter()
                .all(|op| matches!(op, Operation::Clear(_) | Operation::Upload(_)))
        );
        let mut math = Vec::new();
        queue_compute(&cancel, |op| {
            math.push(op);
            Ok(())
        })
        .unwrap();
        assert!(
            math.iter()
                .all(|op| !matches!(op, Operation::Clear(_) | Operation::Upload(_)))
        );
        assert_eq!(math[0], Operation::Projection(0));
        assert_eq!(math.last(), Some(&Operation::Finite));
        // Cancellation after staging cannot begin a single math consumer.
        cancel.store(true, Ordering::Release);
        let mut dispatched = false;
        assert!(
            queue_compute(&cancel, |_| {
                dispatched = true;
                Ok(())
            })
            .is_err()
        );
        assert!(!dispatched);
    }
    #[test]
    fn resident_gpu_history_preparation_never_clears_or_uploads_its_committed_owner() {
        let cancel = AtomicBool::new(false);
        let mut ops = Vec::new();
        queue_preparation(&cancel, true, |op| {
            ops.push(op);
            Ok(())
        })
        .unwrap();
        assert_eq!(ops.len(), N);
        assert!(!ops.contains(&Operation::Clear(OLD)));
        assert!(!ops.contains(&Operation::Upload(0)));
        assert!(ops.contains(&Operation::Clear(NEXT)));
        assert!(ops.contains(&Operation::Upload(1)));
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
        let staged_poison = Arc::new(AtomicBool::new(false));
        drop(PreparedPleBatch {
            receipt: PendingPleBatch {
                metadata: Zeroizing::new(metadata.clone()),
                poison: staged_poison.clone(),
                armed: true,
            },
        });
        assert!(staged_poison.load(Ordering::Acquire));
        let poison = Arc::new(AtomicBool::new(false));
        drop(PendingPleBatch {
            metadata: Zeroizing::new(metadata),
            poison: poison.clone(),
            armed: true,
        });
        assert!(poison.load(Ordering::Acquire));
    }
}
