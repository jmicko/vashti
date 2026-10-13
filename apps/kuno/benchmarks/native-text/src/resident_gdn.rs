//! Resident GDN layer orchestration for the isolated Rust GPU laboratory.
//!
//! This module enqueues the exact admitted `BoundGraph` GDN path: four dense
//! projections, RAW-history convolution/SiLU, tiled-head Q/K preparation,
//! controls, delta recurrence, RMS/sigmoid gate, and output projection. It
//! downloads no activations or state. Model vectors live in a separate PUBLIC
//! cache; every request-derived allocation belongs to the caller's private
//! sequence arena. The arena must use its required allocator configuration.
//!
//! Enqueue is NOT token acceptance. Before its first mutation the layer becomes
//! uncommitted. The WHOLE-token owner must await completion, accept every kernel
//! status and every projector fault/status, check cancellation/context health,
//! and then call `commit_after_whole_token_acceptance` on ALL participating
//! layers. Only that call swaps the old/next raw histories and recurrent state.
//! On ANY layer/owner failure, poison every participating layer and erase/close
//! the WHOLE sequence arena. There is no per-layer retry or old-state rollback
//! after other layers may have mutated. Abandoning a pending layer also requires
//! whole-sequence cleanup. A late cancellation after one layer's commit still
//! invalidates the whole sequence; commits are serialized by the single owner.
//!
//! Projector contract: use only the admitted TensorView/Matrix supplied here;
//! enqueue on the same client/stream; keep input/output regions disjoint; retain
//! no private host/GPU copies outside the sequence lifecycle; register each
//! projection's completion/finite-result status with WHOLE-token acceptance.
//! In particular, the closing-gate status does NOT validate `ssm_out` results.
//! An enqueue Ok only means commands were queued, never that a GPU succeeded.
//!
//! Physical region checks are performed once at construction for all owned
//! allocations and immutable public vectors. Active arena IDs prove later
//! input/output allocations are distinct from the layer's private scratch.
//! Q/K/V are bounded, 256-byte-aligned READONLY subviews of one prepared array;
//! the recurrence kernel must not write them. Raw handles are immediate enqueue
//! leases and must not escape close/reset. Allocator handle reuse after reset
//! is rejected by the arena's epoch checks. Public caches must upload exactly
//! the public values requested; no readback of model parameters is performed.
//!
//! This is a staging contract, not a runtime privacy assertion. CubeCL/wgpu's
//! upload, readback, uniform/staging pools still need the separate erasure hook
//! under investigation. Host tests/offline shader compilation do not establish
//! device parity, performance, failure recovery, or secure cleanup.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[path = "resident_gdn_batch.rs"]
pub(crate) mod batch;

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::{AttentionKind, BoundGraph, GraphBindingId, GraphGeometry};
use kuno_text::{Error, Result, TensorView};

use crate::gdn_gpu::{self, GdnLayout};
use crate::gdn_pointwise::{self, PointwiseLayout};
use crate::private_arena::{self, BufferId, BufferLayout, PrivateArena};

pub const STORAGE_OFFSET_ALIGNMENT: u64 = 256;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn fault(reason: &'static str) -> Error {
    Error::InvalidFormat(reason)
}
fn identity() -> Result<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(fault("resident GDN cancelled"))
    } else {
        Ok(())
    }
}
fn arena_fault(_: private_arena::ArenaError) -> Error {
    fault("resident GDN arena unavailable")
}

/// Generate once per transient sequence; reuse the key for all of its layers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SequenceKey(u64);
impl zeroize::Zeroize for SequenceKey {
    fn zeroize(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.0);
    }
}
impl SequenceKey {
    pub fn new() -> Result<Self> {
        identity().map(Self)
    }
}

/// Separate PUBLIC-model cache boundary. Returned scalar storage must contain
/// exactly `values`, with at least that many used bytes, on the arena's client
/// stream. Caller owns immutable cache handles until all graph work drains.
/// No request-derived vector may be passed to this public API.
pub trait PublicGdnCache {
    fn upload_public_vector(
        &mut self,
        binding: GraphBindingId,
        name: &str,
        values: &[f32],
    ) -> Result<Handle>;
}

/// Immutable public parameters. Fields are private so an admitted cache entry
/// cannot silently be replaced with another layer/model after construction.
pub struct PublicGdnParameters {
    binding: GraphBindingId,
    layer: usize,
    identity: u64,
    convolution: Handle,
    controls: Handle,
    gamma: Handle,
    gamma_words: usize,
}

impl PublicGdnParameters {
    /// Uses only already-bound static public vectors, never full weight reads.
    pub fn prepare(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        cache: &mut impl PublicGdnCache,
    ) -> Result<Self> {
        let layout = StageLayout::from_graph(graph, layer)?;
        let prefix = format!("blk.{layer}.");
        let conv_name = format!("{prefix}ssm_conv1d.weight");
        let gamma_name = format!("{prefix}ssm_norm.weight");
        let convolution = graph.static_vector(&conv_name)?;
        let dt = graph.static_vector(&format!("{prefix}ssm_dt.bias"))?;
        let a = graph.static_vector(&format!("{prefix}ssm_a"))?;
        let gamma = graph.static_vector(&gamma_name)?;
        validate_public_vectors(layout.pointwise, convolution, dt, a, gamma)?;
        // This is PUBLIC model data, not private request scratch. Stored A is
        // already -exp(A_log); no exponentiation or head permutation here.
        let mut controls = Vec::new();
        controls
            .try_reserve_exact(layout.pointwise.control_elements)
            .map_err(|_| Error::LimitExceeded("public GDN control scratch"))?;
        controls.extend_from_slice(dt);
        controls.extend_from_slice(a);
        let binding = graph.binding_id();
        let identity = identity()?;
        let convolution = cache.upload_public_vector(binding, &conv_name, convolution)?;
        let control_name = format!("{prefix}kuno_public_dt_and_a");
        let controls = cache.upload_public_vector(binding, &control_name, &controls)?;
        let gamma_handle = cache.upload_public_vector(binding, &gamma_name, gamma)?;
        // Validate bounds before Handle's unchecked offset arithmetic.
        bounded_view(&convolution, 0, layout.pointwise.conv_weight_elements)?;
        bounded_view(&controls, 0, layout.pointwise.control_elements)?;
        bounded_view(&gamma_handle, 0, gamma.len())?;
        Ok(Self {
            binding,
            layer,
            identity,
            convolution,
            controls,
            gamma: gamma_handle,
            gamma_words: gamma.len(),
        })
    }
}

fn validate_public_vectors(
    layout: PointwiseLayout,
    convolution: &[f32],
    dt: &[f32],
    a: &[f32],
    gamma: &[f32],
) -> Result<()> {
    let heads = layout.shape.value_heads;
    if convolution.len() != layout.conv_weight_elements
        || dt.len() != heads
        || a.len() != heads
        || (gamma.len() != layout.shape.head_dim && gamma.len() != layout.value_elements)
    {
        return Err(Error::InvalidBufferLength);
    }
    if ![convolution, dt, a, gamma]
        .iter()
        .all(|values| values.iter().all(|v| v.is_finite()))
        || a.iter().any(|v| *v >= 0.0)
    {
        return Err(fault("invalid public GDN parameters"));
    }
    Ok(())
}

/// Both views describe the SAME admitted packed matrix; Matrix has no public
/// tensor accessor, so a packed GPU projector needs the TensorView as well.
#[derive(Clone, Copy)]
pub struct BoundProjection<'file> {
    pub tensor: TensorView<'file>,
    pub matrix: Matrix<'file>,
}

pub trait ResidentProjector {
    fn enqueue(
        &mut self,
        projection: BoundProjection<'_>,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()>;
}

/// Pure geometry and allocation plan; useful before any GPU allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageLayout {
    pub pointwise: PointwiseLayout,
    pub recurrence: GdnLayout,
    pub input_words: usize,
    pub epsilon: f32,
    pub max_context: usize,
}

impl StageLayout {
    pub fn from_graph(graph: BoundGraph<'_, '_>, layer: usize) -> Result<Self> {
        if graph.layer_kind(layer)? != AttentionKind::Gdn {
            return Err(fault("resident GDN requires admitted GDN layer"));
        }
        Self::from_geometry(graph.geometry())
    }
    fn from_geometry(geometry: GraphGeometry) -> Result<Self> {
        let pointwise = PointwiseLayout::new(geometry.gdn).map_err(fault)?;
        let recurrence = GdnLayout::new(geometry.gdn).map_err(fault)?;
        if geometry.gdn_conv_channels != pointwise.channels
            || geometry.gdn_conv_kernel != 4
            || geometry.gdn_conv_dilation != 1
            || !geometry.epsilon.is_finite()
            || geometry.epsilon <= 0.0
            || geometry.residual.width == 0
            || geometry.max_context == 0
        {
            return Err(fault("unsupported resident GDN graph geometry"));
        }
        let layout = Self {
            pointwise,
            recurrence,
            input_words: geometry.residual.width,
            epsilon: geometry.epsilon,
            max_context: geometry.max_context,
        };
        PreparedSlices::new(layout.pointwise)?;
        Ok(layout)
    }
    /// Slots: old/next recurrent, old/next raw ring, qkv,z,alpha,beta,
    /// activation,prepared,controls,recurrence_out,gated,five status arrays.
    pub fn allocation_words(self) -> [usize; 18] {
        let p = self.pointwise;
        [
            self.recurrence.state_elements,
            self.recurrence.state_elements,
            p.ring_elements,
            p.ring_elements,
            p.channels,
            p.value_elements,
            p.shape.value_heads,
            p.shape.value_heads,
            p.channels,
            p.channels,
            p.control_elements,
            p.value_elements,
            p.value_elements,
            p.channels,
            p.channels,
            p.shape.value_heads,
            p.value_elements,
            p.value_elements,
        ]
    }
    /// Charges the entire padded arena spans, including tails.
    pub fn owned_bytes(self) -> Result<usize> {
        self.allocation_words()
            .into_iter()
            .try_fold(0usize, |total, words| {
                let bytes = BufferLayout::new(words).map_err(arena_fault)?.owned_bytes;
                total.checked_add(bytes).ok_or(Error::ArithmeticOverflow)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SliceRegion {
    offset_words: usize,
    words: usize,
}

/// Readonly Q/K/V offsets; every storage-binding start must be aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreparedSlices {
    query: SliceRegion,
    key: SliceRegion,
    value: SliceRegion,
}
impl PreparedSlices {
    fn new(layout: PointwiseLayout) -> Result<Self> {
        if layout != PointwiseLayout::new(layout.shape).map_err(fault)? {
            return Err(fault("modified GDN pointwise layout"));
        }
        let two_qk = layout
            .qk_elements
            .checked_mul(2)
            .ok_or(Error::ArithmeticOverflow)?;
        let slices = Self {
            query: SliceRegion {
                offset_words: 0,
                words: layout.qk_elements,
            },
            key: SliceRegion {
                offset_words: layout.qk_elements,
                words: layout.qk_elements,
            },
            value: SliceRegion {
                offset_words: two_qk,
                words: layout.value_elements,
            },
        };
        for region in [slices.query, slices.key, slices.value] {
            checked_region(
                layout.channels as u64 * 4,
                0,
                0,
                region.offset_words,
                region.words,
            )?;
        }
        Ok(slices)
    }
}

/// Checked relative subrange of a possibly already-offset Handle. The returned
/// start/end trims are additional Handle offsets; both are checked before use.
fn checked_region(
    size: u64,
    current_start: u64,
    current_end: u64,
    offset_words: usize,
    words: usize,
) -> Result<(u64, u64)> {
    if words == 0
        || !size.is_multiple_of(4)
        || !current_start.is_multiple_of(4)
        || !current_end.is_multiple_of(4)
    {
        return Err(Error::InvalidBufferLength);
    }
    let used = size
        .checked_sub(current_start)
        .and_then(|n| n.checked_sub(current_end))
        .ok_or(Error::InvalidBufferLength)?;
    let relative_start = u64::try_from(offset_words)
        .map_err(|_| Error::ArithmeticOverflow)?
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let bytes = u64::try_from(words)
        .map_err(|_| Error::ArithmeticOverflow)?
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let relative_end = relative_start
        .checked_add(bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let absolute_start = current_start
        .checked_add(relative_start)
        .ok_or(Error::ArithmeticOverflow)?;
    if relative_end > used || absolute_start % STORAGE_OFFSET_ALIGNMENT != 0 {
        return Err(fault("GDN storage view bounds/alignment"));
    }
    Ok((relative_start, used - relative_end))
}

fn bounded_view(handle: &Handle, offset_words: usize, words: usize) -> Result<Handle> {
    let (start, end) = checked_region(
        handle.size(),
        handle.offset_start.unwrap_or(0),
        handle.offset_end.unwrap_or(0),
        offset_words,
        words,
    )?;
    let view = handle.clone().offset_start(start).offset_end(end);
    let bytes = u64::try_from(words).map_err(|_| Error::ArithmeticOverflow)? * 4;
    if view.size_in_used() != bytes {
        return Err(fault("GDN storage view mismatch"));
    }
    Ok(view)
}

/// Immediate bounded view for owner/projector calls. Do not retain this clone
/// after dispatch or arena reset/close; logical length is never owned padding.
pub fn logical_handle(arena: &PrivateArena, id: BufferId) -> Result<Handle> {
    let layout = arena.layout(id).map_err(arena_fault)?;
    bounded_view(
        arena.handle(id).map_err(arena_fault)?,
        0,
        layout.logical_words,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Ready(u64),
    Uncommitted(u64),
    UncommittedBatch(u64, usize),
    Poisoned,
}
struct Lifecycle {
    phase: Phase,
    max_context: usize,
}
impl Lifecycle {
    fn new(max_context: usize) -> Self {
        Self {
            phase: Phase::Ready(0),
            max_context,
        }
    }
    fn begin(&mut self, position: u64) -> Result<()> {
        if self.phase != Phase::Ready(position) || position >= self.max_context as u64 {
            self.phase = Phase::Poisoned;
            return Err(fault("resident GDN token order/state"));
        }
        self.phase = Phase::Uncommitted(position);
        Ok(())
    }
    fn commit(&mut self, position: u64) -> Result<()> {
        if self.phase != Phase::Uncommitted(position) {
            self.phase = Phase::Poisoned;
            return Err(fault("resident GDN stale token commit"));
        }
        let next = position.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        self.phase = Phase::Ready(next);
        Ok(())
    }
    fn poison(&mut self) {
        self.phase = Phase::Poisoned;
    }
}

#[derive(Clone, Copy)]
pub struct StatusRegion {
    pub buffer: BufferId,
    pub words: usize,
}

/// Opaque queue receipt; it carries NO claim that device execution succeeded.
pub struct PendingGdn {
    stage: u64,
    sequence: SequenceKey,
    position: u64,
    output: BufferId,
    statuses: [StatusRegion; 5],
}
impl PendingGdn {
    /// Order: raw conv, QKV preparation, controls, recurrence, closing gate.
    /// Owner must also validate projector-owned statuses/completion.
    pub fn statuses(&self) -> &[StatusRegion; 5] {
        &self.statuses
    }
    pub fn output(&self) -> BufferId {
        self.output
    }
    pub fn position(&self) -> u64 {
        self.position
    }
}

struct Buffers {
    recurrent: [BufferId; 2],
    raw_conv: [BufferId; 2],
    qkv: BufferId,
    z: BufferId,
    alpha: BufferId,
    beta: BufferId,
    activation: BufferId,
    prepared: BufferId,
    controls: BufferId,
    recurrent_output: BufferId,
    gated: BufferId,
    statuses: [StatusRegion; 5],
}
impl Buffers {
    fn from_slots(slots: [BufferId; 18], layout: StageLayout) -> Self {
        let p = layout.pointwise;
        Self {
            recurrent: [slots[0], slots[1]],
            raw_conv: [slots[2], slots[3]],
            qkv: slots[4],
            z: slots[5],
            alpha: slots[6],
            beta: slots[7],
            activation: slots[8],
            prepared: slots[9],
            controls: slots[10],
            recurrent_output: slots[11],
            gated: slots[12],
            statuses: [
                StatusRegion {
                    buffer: slots[13],
                    words: p.channels,
                },
                StatusRegion {
                    buffer: slots[14],
                    words: p.channels,
                },
                StatusRegion {
                    buffer: slots[15],
                    words: p.shape.value_heads,
                },
                StatusRegion {
                    buffer: slots[16],
                    words: p.value_elements,
                },
                StatusRegion {
                    buffer: slots[17],
                    words: p.value_elements,
                },
            ],
        }
    }
    fn ids(&self) -> [BufferId; 18] {
        [
            self.recurrent[0],
            self.recurrent[1],
            self.raw_conv[0],
            self.raw_conv[1],
            self.qkv,
            self.z,
            self.alpha,
            self.beta,
            self.activation,
            self.prepared,
            self.controls,
            self.recurrent_output,
            self.gated,
            self.statuses[0].buffer,
            self.statuses[1].buffer,
            self.statuses[2].buffer,
            self.statuses[3].buffer,
            self.statuses[4].buffer,
        ]
    }
}

struct Projections<'file> {
    qkv: BoundProjection<'file>,
    z: BoundProjection<'file>,
    alpha: BoundProjection<'file>,
    beta: BoundProjection<'file>,
    output: BoundProjection<'file>,
}
fn projection<'file>(
    graph: BoundGraph<'_, 'file>,
    name: &str,
    inputs: usize,
    outputs: usize,
) -> Result<BoundProjection<'file>> {
    let tensor = graph.tensor(name)?;
    let matrix = Matrix::new(tensor)?;
    if matrix.inputs() != inputs || matrix.outputs() != outputs || matrix.experts() != 1 {
        return Err(fault("resident GDN projection geometry"));
    }
    Ok(BoundProjection { tensor, matrix })
}

pub struct ResidentGdn<'file> {
    binding: GraphBindingId,
    layer: usize,
    sequence: SequenceKey,
    identity: u64,
    public_identity: u64,
    layout: StageLayout,
    buffers: Buffers,
    projections: Projections<'file>,
    lifecycle: Lifecycle,
    batch_pending: Option<batch::StageAnchor>,
    batch_poison: Option<std::sync::Arc<AtomicBool>>,
    strata_inputs: Option<batch::strata_replay::InputCache>,
}

impl<'file> ResidentGdn<'file> {
    /// All buffers start at zero; both committed/candidate states persist until
    /// whole-sequence arena close. On partial allocation/error, caller must
    /// close the entire arena (this method never drops untracked raw buffers).
    pub fn allocate(
        graph: BoundGraph<'_, 'file>,
        layer: usize,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        public: &PublicGdnParameters,
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        let layout = StageLayout::from_graph(graph, layer)?;
        if public.binding != graph.binding_id() || public.layer != layer {
            return Err(fault("resident GDN public binding mismatch"));
        }
        let total = arena
            .charged_bytes()
            .checked_add(layout.owned_bytes()?)
            .ok_or(Error::ArithmeticOverflow)?;
        if total > arena.budget_bytes() {
            return Err(Error::LimitExceeded("resident GDN arena budget"));
        }
        let prefix = format!("blk.{layer}.");
        let p = layout.pointwise;
        let projections = Projections {
            qkv: projection(
                graph,
                &format!("{prefix}attn_qkv.weight"),
                layout.input_words,
                p.channels,
            )?,
            z: projection(
                graph,
                &format!("{prefix}attn_gate.weight"),
                layout.input_words,
                p.value_elements,
            )?,
            alpha: projection(
                graph,
                &format!("{prefix}ssm_alpha.weight"),
                layout.input_words,
                p.shape.value_heads,
            )?,
            beta: projection(
                graph,
                &format!("{prefix}ssm_beta.weight"),
                layout.input_words,
                p.shape.value_heads,
            )?,
            output: projection(
                graph,
                &format!("{prefix}ssm_out.weight"),
                p.value_elements,
                layout.input_words,
            )?,
        };
        let identity = identity()?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(18)
            .map_err(|_| Error::LimitExceeded("resident GDN metadata"))?;
        for words in layout.allocation_words() {
            slots.push(arena.allocate_zeroed(words).map_err(arena_fault)?);
        }
        let slots: [BufferId; 18] = slots
            .try_into()
            .map_err(|_| fault("resident GDN slot count"))?;
        let buffers = Buffers::from_slots(slots, layout);
        validate_physical_regions(arena, &buffers, public)?;
        let strata_inputs = batch::strata_replay::InputCache::allocate(arena, layout)?;
        Ok(Self {
            binding: graph.binding_id(),
            layer,
            sequence,
            identity,
            public_identity: public.identity,
            layout,
            buffers,
            projections,
            lifecycle: Lifecycle::new(layout.max_context),
            batch_pending: None,
            batch_poison: None,
            strata_inputs,
        })
    }
    pub fn layout(&self) -> StageLayout {
        self.layout
    }
    /// Pending and failed layers cannot start another token. If either is
    /// abandoned, owner must erase the enclosing sequence, not just this layer.
    pub fn requires_erasure_if_abandoned(&self) -> bool {
        !matches!(self.lifecycle.phase, Phase::Ready(_))
            || self
                .batch_poison
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Acquire))
    }
    pub fn poison(&mut self) {
        self.lifecycle.poison();
        if let Some(poison) = &self.batch_poison {
            poison.store(true, Ordering::Release);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicGdnParameters,
        projector: &mut impl ResidentProjector,
        input: BufferId,
        output: BufferId,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PendingGdn> {
        // Poison on ordinary errors AND panics. Error messages deliberately
        // contain no device payload, private data, addresses or token position.
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.enqueue_checked(
                graph, arena, public, projector, input, output, position, cancel,
            )
        }));
        match attempted {
            Ok(Ok(receipt)) => Ok(receipt),
            Ok(Err(error)) => {
                self.poison();
                Err(error)
            }
            Err(_) => {
                self.poison();
                Err(fault("resident GDN enqueue failed"))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        public: &PublicGdnParameters,
        projector: &mut impl ResidentProjector,
        input: BufferId,
        output: BufferId,
        position: u64,
        cancel: &AtomicBool,
    ) -> Result<PendingGdn> {
        cancelled(cancel)?;
        if self
            .batch_poison
            .as_ref()
            .is_some_and(|p| p.load(Ordering::Acquire))
        {
            return Err(fault("resident GDN sequence poisoned"));
        }
        if graph.binding_id() != self.binding
            || public.binding != self.binding
            || public.layer != self.layer
            || public.identity != self.public_identity
        {
            return Err(fault("resident GDN binding changed"));
        }
        graph.verify_unchanged()?;
        if StageLayout::from_graph(graph, self.layer)? != self.layout {
            return Err(fault("resident GDN geometry changed"));
        }
        self.validate_ids(arena, input, output)?;
        let p = self.layout.pointwise;
        let r = self.layout.recurrence;
        let ids = &self.buffers;
        let gamma_shared = gdn_pointwise::validate_output_lengths(
            p,
            p.value_elements,
            p.value_elements,
            public.gamma_words,
            p.value_elements,
            ids.statuses[4].words,
            self.layout.epsilon,
        )
        .map_err(fault)?;
        let conv = bounded_view(&public.convolution, 0, p.conv_weight_elements)?;
        let params = bounded_view(&public.controls, 0, p.control_elements)?;
        let gamma = bounded_view(&public.gamma, 0, public.gamma_words)?;
        let slices = PreparedSlices::new(p)?;
        let prepared = arena.handle(ids.prepared).map_err(arena_fault)?;
        let q = bounded_view(prepared, slices.query.offset_words, slices.query.words)?;
        let k = bounded_view(prepared, slices.key.offset_words, slices.key.words)?;
        let v = bounded_view(prepared, slices.value.offset_words, slices.value.words)?;
        // Mark uncommitted BEFORE resetting statuses or projecting anything.
        self.lifecycle.begin(position)?;
        let client = arena.client().map_err(arena_fault)?;
        for status in ids.statuses {
            let span = arena.layout(status.buffer).map_err(arena_fault)?;
            private_arena::erase_full_span::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(span.workgroups_x, span.workgroups_y, 1),
                CubeDim::new_1d(64),
                // SAFETY: full owned spans, retained by arena until owner drain.
                unsafe {
                    ArrayArg::from_raw_parts(
                        arena.handle(status.buffer).map_err(arena_fault)?.clone(),
                        span.owned_words,
                    )
                },
            );
        }
        for (projection, target) in [
            (self.projections.qkv, ids.qkv),
            (self.projections.z, ids.z),
            (self.projections.alpha, ids.alpha),
            (self.projections.beta, ids.beta),
        ] {
            cancelled(cancel)?;
            projector.enqueue(projection, arena, input, target, cancel)?;
        }
        cancelled(cancel)?;
        gdn_pointwise::conv_silu::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(p.conv_cubes, 1, 1),
            CubeDim::new_1d(64),
            f32_arg(arena, ids.qkv)?,
            raw_f32_arg(conv, p.conv_weight_elements),
            f32_arg(arena, ids.raw_conv[0])?,
            f32_arg(arena, ids.raw_conv[1])?,
            f32_arg(arena, ids.activation)?,
            u32_arg(arena, ids.statuses[0].buffer)?,
            p.channels,
        );
        cancelled(cancel)?;
        gdn_pointwise::prepare_qkv::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(p.qkv_heads as u32, 1, 1),
            CubeDim::new_1d(64),
            f32_arg(arena, ids.activation)?,
            f32_arg(arena, ids.prepared)?,
            u32_arg(arena, ids.statuses[1].buffer)?,
            self.layout.epsilon,
            p.shape.key_heads,
            p.shape.value_heads,
        );
        cancelled(cancel)?;
        gdn_pointwise::prepare_controls::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(p.control_cubes, 1, 1),
            CubeDim::new_1d(64),
            f32_arg(arena, ids.alpha)?,
            f32_arg(arena, ids.beta)?,
            raw_f32_arg(params, p.control_elements),
            f32_arg(arena, ids.controls)?,
            u32_arg(arena, ids.statuses[2].buffer)?,
            p.shape.value_heads,
        );
        cancelled(cancel)?;
        gdn_gpu::gdn_step::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(r.workgroups_x, 1, 1),
            CubeDim::new_1d(64),
            raw_f32_arg(q, r.qk_elements),
            raw_f32_arg(k, r.qk_elements),
            raw_f32_arg(v, r.value_elements),
            f32_arg(arena, ids.controls)?,
            f32_arg(arena, ids.recurrent[0])?,
            f32_arg(arena, ids.recurrent[1])?,
            f32_arg(arena, ids.recurrent_output)?,
            u32_arg(arena, ids.statuses[3].buffer)?,
            r.shape.head_dim,
            r.shape.key_heads,
            r.shape.value_heads,
        );
        cancelled(cancel)?;
        gdn_pointwise::output_gate::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(p.shape.value_heads as u32, 1, 1),
            CubeDim::new_1d(64),
            f32_arg(arena, ids.recurrent_output)?,
            f32_arg(arena, ids.z)?,
            raw_f32_arg(gamma, public.gamma_words),
            f32_arg(arena, ids.gated)?,
            u32_arg(arena, ids.statuses[4].buffer)?,
            self.layout.epsilon,
            p.shape.value_heads,
            gamma_shared,
        );
        cancelled(cancel)?;
        projector.enqueue(self.projections.output, arena, ids.gated, output, cancel)?;
        cancelled(cancel)?;
        Ok(PendingGdn {
            stage: self.identity,
            sequence: self.sequence,
            position,
            output,
            statuses: ids.statuses,
        })
    }

    fn validate_ids(&self, arena: &PrivateArena, input: BufferId, output: BufferId) -> Result<()> {
        if input == output
            || arena.layout(input).map_err(arena_fault)?.logical_words != self.layout.input_words
            || arena.layout(output).map_err(arena_fault)?.logical_words != self.layout.input_words
        {
            return Err(fault("resident GDN input/output geometry or alias"));
        }
        for (id, words) in self
            .buffers
            .ids()
            .into_iter()
            .zip(self.layout.allocation_words())
        {
            if id == input
                || id == output
                || arena.layout(id).map_err(arena_fault)?.logical_words != words
            {
                return Err(fault("resident GDN stale or aliased allocation"));
            }
        }
        // Active IDs in a correctly configured PrivateArena own distinct
        // exclusive allocations. Public handles were checked at construction.
        Ok(())
    }

    /// OWNER PRECONDITION: successful whole-token synchronization, exact statuses
    /// all1, projector finite/completion checks, every other layer accepted,
    /// no cancellation/context fault. This performs no GPU readback or sync.
    /// Call all layer commits under a single serialized owner decision. Any late
    /// error/cancel invalidates and erases the whole sequence, even after a swap.
    pub fn commit_after_whole_token_acceptance(
        &mut self,
        receipt: PendingGdn,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let attempted = (|| {
            cancelled(cancel)?;
            if self
                .batch_poison
                .as_ref()
                .is_some_and(|p| p.load(Ordering::Acquire))
            {
                return Err(fault("resident GDN sequence poisoned"));
            }
            arena.client().map_err(arena_fault)?;
            if receipt.stage != self.identity || receipt.sequence != self.sequence {
                return Err(fault("resident GDN foreign token receipt"));
            }
            for id in self.buffers.ids() {
                arena.layout(id).map_err(arena_fault)?;
            }
            self.lifecycle.commit(receipt.position)?;
            self.buffers.recurrent.swap(0, 1);
            self.buffers.raw_conv.swap(0, 1);
            Ok(())
        })();
        if attempted.is_err() {
            self.poison();
        }
        attempted
    }
}

fn f32_arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    let words = arena.layout(id).map_err(arena_fault)?.logical_words;
    Ok(raw_f32_arg(logical_handle(arena, id)?, words))
}
fn raw_f32_arg(handle: Handle, words: usize) -> ArrayArg<WgpuRuntime> {
    // SAFETY: constructors validated exact bounded scalar f32 storage lengths;
    // callers use read-only inputs/mutable outputs only in disjoint allocations.
    unsafe { ArrayArg::from_raw_parts(handle, words) }
}
fn u32_arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    let words = arena.layout(id).map_err(arena_fault)?.logical_words;
    // SAFETY: status allocations are 32-bit scalar arrays of logical length.
    Ok(unsafe { ArrayArg::from_raw_parts(logical_handle(arena, id)?, words) })
}

/// Constructor-only physical alias proof. Does not read any buffer contents.
/// Keep resources alive during comparison so allocator reuse cannot intervene.
fn validate_physical_regions(
    arena: &PrivateArena,
    buffers: &Buffers,
    public: &PublicGdnParameters,
) -> Result<()> {
    let client = arena.client().map_err(arena_fault)?;
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(21)
        .map_err(|_| Error::LimitExceeded("GDN region metadata"))?;
    for id in buffers.ids() {
        handles.push(arena.handle(id).map_err(arena_fault)?.clone());
    }
    handles.push(public.convolution.clone());
    handles.push(public.controls.clone());
    handles.push(public.gamma.clone());
    let stream = handles[0].stream;
    if handles.iter().any(|handle| handle.stream != stream) {
        return Err(fault("resident GDN cross-stream allocation"));
    }
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(handles.len())
        .map_err(|_| Error::LimitExceeded("GDN region metadata"))?;
    for handle in handles {
        let resource =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| client.get_resource(handle)))
                .map_err(|_| fault("GDN buffer resource lookup failed"))?
                .map_err(|_| fault("GDN buffer resource lookup failed"))?;
        resources.push(resource);
    }
    for (i, left) in resources.iter().enumerate() {
        let left = left.resource();
        let left_end = left
            .offset
            .checked_add(left.size)
            .ok_or(Error::ArithmeticOverflow)?;
        for right in resources.iter().skip(i + 1) {
            let right = right.resource();
            let right_end = right
                .offset
                .checked_add(right.size)
                .ok_or(Error::ArithmeticOverflow)?;
            if left.buffer == right.buffer
                && intervals_overlap(left.offset, left_end, right.offset, right_end)
            {
                return Err(fault("resident GDN physically aliased allocation"));
            }
        }
    }
    Ok(())
}
fn intervals_overlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::model::PleGeometry;
    use kuno_text::ops::{GatedResidualShape, GdnShape, QsaShape};

    fn geometry() -> GraphGeometry {
        // Independent exact geometry fixture; no model payloads or devices.
        GraphGeometry {
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
        }
    }

    #[test]
    fn exact_state_plan_charges_owned_spans_and_separates_every_candidate() {
        let layout = StageLayout::from_geometry(geometry()).unwrap();
        assert_eq!(layout.recurrence.state_elements, 48 * 128 * 128);
        assert_eq!(layout.pointwise.ring_elements, 10240 * 3);
        assert_eq!(layout.allocation_words().len(), 18);
        assert_eq!(
            layout.allocation_words()[0..4],
            [786432, 786432, 30720, 30720]
        );
        // 2 recurrent4MiB + 2 rings128KiB + 5 spans64KiB + 9 spans32KiB.
        assert_eq!(
            layout.owned_bytes().unwrap(),
            2 * 4 * 1024 * 1024 + 2 * 128 * 1024 + 5 * 64 * 1024 + 9 * 32 * 1024
        );
        let mut altered = geometry();
        altered.gdn_conv_dilation = 2;
        assert!(StageLayout::from_geometry(altered).is_err());
        altered = geometry();
        altered.gdn_conv_channels -= 1;
        assert!(StageLayout::from_geometry(altered).is_err());
        altered = geometry();
        altered.epsilon = f32::NAN;
        assert!(StageLayout::from_geometry(altered).is_err());
    }

    #[test]
    fn prepared_views_exact_tiled_head_ranges_and_alignment() {
        let p = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        let s = PreparedSlices::new(p).unwrap();
        assert_eq!(
            s.query,
            SliceRegion {
                offset_words: 0,
                words: 2048
            }
        );
        assert_eq!(
            s.key,
            SliceRegion {
                offset_words: 2048,
                words: 2048
            }
        );
        assert_eq!(
            s.value,
            SliceRegion {
                offset_words: 4096,
                words: 6144
            }
        );
        assert_eq!(
            checked_region(65536, 0, 0, s.key.offset_words, s.key.words).unwrap(),
            (8192, 49152)
        );
        assert_eq!(
            checked_region(65536, 0, 0, s.value.offset_words, s.value.words).unwrap(),
            (16384, 24576)
        );
        let tiny = PointwiseLayout::new(GdnShape::new(128, 1, 3).unwrap()).unwrap();
        assert!(PreparedSlices::new(tiny).is_ok());
        assert!(checked_region(1024, 0, 0, 1, 128).is_err());
        assert!(checked_region(1024, 0, 0, 256, 1).is_err());
        assert!(checked_region(1024, 800, 400, 0, 1).is_err());
        assert!(checked_region(1024, 0, 0, usize::MAX, 1).is_err());
        assert_eq!(checked_region(1024, 256, 256, 64, 64).unwrap(), (256, 0));
        assert!(!intervals_overlap(0, 8192, 8192, 16384));
        assert!(intervals_overlap(0, 8192, 8191, 16384));
    }

    #[test]
    fn public_parameters_require_negative_stored_a_direct_gamma_and_exact_shape() {
        let p = PointwiseLayout::new(GdnShape::new(128, 1, 3).unwrap()).unwrap();
        let conv = vec![0.25; p.conv_weight_elements];
        let dt = vec![0.1; 3];
        let a = vec![-0.5; 3];
        let gamma = vec![1.0; 128];
        assert!(validate_public_vectors(p, &conv, &dt, &a, &gamma).is_ok());
        assert!(validate_public_vectors(p, &conv, &dt, &[0., -1., -1.], &gamma).is_err());
        assert!(validate_public_vectors(p, &conv, &[f32::NAN, 0., 0.], &a, &gamma).is_err());
        assert!(validate_public_vectors(p, &conv, &dt, &a, &gamma[..127]).is_err());
        assert!(validate_public_vectors(p, &conv, &dt, &a, &vec![1.; 384]).is_ok());
    }

    #[test]
    fn state_cannot_advance_before_token_commit_or_retry_after_failure() {
        let mut life = Lifecycle::new(2);
        assert!(life.phase == Phase::Ready(0));
        life.begin(0).unwrap();
        assert!(life.phase == Phase::Uncommitted(0));
        life.commit(0).unwrap();
        assert!(life.phase == Phase::Ready(1));
        life.begin(1).unwrap();
        life.poison();
        assert!(life.commit(1).is_err());
        assert!(life.begin(1).is_err());
        let mut life = Lifecycle::new(2);
        assert!(life.begin(1).is_err()); // skipped token
        assert!(life.phase == Phase::Poisoned);
        let mut life = Lifecycle::new(2);
        life.begin(0).unwrap();
        assert!(life.begin(0).is_err()); // concurrent/retry
        let mut life = Lifecycle::new(2);
        life.begin(0).unwrap();
        life.commit(0).unwrap();
        assert!(life.commit(0).is_err()); // stale/double acceptance
        let mut life = Lifecycle::new(1);
        life.begin(0).unwrap();
        life.commit(0).unwrap();
        assert!(life.begin(1).is_err()); // maximum context reached
    }

    #[test]
    fn sequence_identities_unique_and_cancel_checks_have_no_content() {
        assert!(SequenceKey::new().unwrap() != SequenceKey::new().unwrap());
        let cancel = AtomicBool::new(false);
        assert!(cancelled(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        assert_eq!(
            cancelled(&cancel).unwrap_err().to_string(),
            "invalid GGUF: resident GDN cancelled"
        );
    }
}
