//! Cache-backed native GR MAX8 workspace, not the decoded fixture cache.
//! The caller admits allocation_requests() before allocation and retains the
//! arena/public cache through the whole-model fence and erasure. No readback,
//! sequence commit, device initialization or default selection occurs here.
#![allow(dead_code)] // Whole-model chunk connection and both-card acceptance pending.
use super::PublicVectors;
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_gdn::BoundProjection,
    resident_util as u, residual_batch_gpu as r,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{
    Error, Result,
    cpu::Matrix,
    model::{BoundGraph, GraphBindingId},
};
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

const N: usize = 10;
const MIXED: usize = 0;
const INJECTION: usize = 1;
const NORM: usize = 2;
const DOWN: usize = 3;
const LOW: usize = 4;
const UP: usize = 5;
const SN: usize = 6;
const SL: usize = 7;
const SM: usize = 8;
const SC: usize = 9;

/// Public graph role; neither an arbitrary tensor name nor private input.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Attention(usize),
    Ffn(usize),
    Final,
    FinalAllRows,
    FinalFourRows,
}
impl Kind {
    fn base(self, layers: usize) -> Result<String> {
        match self {
            Self::Attention(l) if l < layers => Ok(format!("blk.{l}.hc_attn")),
            Self::Ffn(l) if l < layers => Ok(format!("blk.{l}.hc_ffn")),
            Self::Final | Self::FinalAllRows | Self::FinalFourRows => Ok("output_hc".into()),
            _ => Err(Error::OutOfBounds),
        }
    }
    fn inject(self) -> bool {
        !matches!(self, Self::Final | Self::FinalAllRows | Self::FinalFourRows)
    }
}

/// Ordinary Final compacts the last original residual column into scratch row0.
/// FinalAllRows retains the bounded first one to three rows; Attention/FFN keep
/// every active column. All roles retain the fixed MAX8 owning allocation.
fn read_columns(kind: Kind, count: usize, width: usize, words: usize) -> Result<(usize, usize)> {
    if (kind == Kind::FinalAllRows && count > crate::resident_model::batch::TARGET_ROWS_MAX)
        || (kind == Kind::FinalFourRows
            && !(4..=crate::strata_port::verify_capacity()).contains(&count))
    {
        return Err(Error::InvalidBufferLength);
    }
    validate_view(count, 0, width, width, words)?;
    let (active, first) = if kind == Kind::Final {
        (
            1,
            (count - 1)
                .checked_mul(width)
                .ok_or(Error::ArithmeticOverflow)?,
        )
    } else {
        (count, 0)
    };
    validate_view(active, first, width, width, words)?;
    Ok((active, first))
}

fn final_head_spans(
    kind: Kind,
    count: usize,
    width: usize,
    vocabulary: usize,
) -> Result<BatchSpans> {
    if !matches!(kind, Kind::Final | Kind::FinalAllRows | Kind::FinalFourRows) || count == 0 {
        return Err(Error::InvalidBufferLength);
    }
    let all_rows = matches!(kind, Kind::FinalAllRows | Kind::FinalFourRows);
    if all_rows
        && count
            > if kind == Kind::FinalFourRows {
                crate::strata_port::verify_capacity()
            } else {
                crate::resident_model::batch::TARGET_ROWS_MAX
            }
    {
        return Err(Error::InvalidBufferLength);
    }
    let first = if all_rows {
        0
    } else {
        (count - 1)
            .checked_mul(width)
            .ok_or(Error::ArithmeticOverflow)?
    };
    Ok(BatchSpans::new(
        if all_rows { count } else { 1 },
        0,
        first,
        width,
        0,
        vocabulary,
    ))
}

/// Logical owning capacity follows the target's immutable constructor flag,
/// independently of the active tail/prefill count and its original arithmetic.
fn final_head_owner_words(
    kind: Kind,
    count: usize,
    vocabulary: usize,
    four_reserved: bool,
) -> Result<usize> {
    if vocabulary == 0 {
        return Err(Error::InvalidBufferLength);
    }
    let rows = match kind {
        Kind::Final if count == 1 && !four_reserved => 1,
        Kind::FinalAllRows
            if (1..=crate::resident_model::batch::TARGET_ROWS_MAX).contains(&count) =>
        {
            if four_reserved {
                crate::strata_port::verify_capacity()
            } else {
                crate::resident_model::batch::TARGET_ROWS_MAX
            }
        }
        Kind::FinalFourRows
            if (4..=crate::strata_port::verify_capacity()).contains(&count) && four_reserved =>
        {
            crate::strata_port::verify_capacity()
        }
        _ => return Err(Error::InvalidBufferLength),
    };
    rows.checked_mul(vocabulary)
        .ok_or(Error::ArithmeticOverflow)
}

#[derive(Clone, Copy)]
struct Layout {
    streams: usize,
    width: usize,
    low: usize,
    widths: [usize; N],
}
impl Layout {
    fn new(streams: usize, width: usize, low: usize) -> Result<Self> {
        let shape = kuno_text::ops::GatedResidualShape::new(streams, width).map_err(u::fault)?;
        let scalar = crate::residual_gpu::ResidualLayout::new(shape, low).map_err(u::fault)?;
        let widths = [
            width,
            streams,
            scalar.residual_elements,
            low,
            low,
            if crate::strata_intel_core::enabled() {
                crate::strata_gr::PARTIAL_WIDTH
            } else {
                scalar.residual_elements
            },
            if crate::strata_intel_core::enabled() {
                2 * streams
            } else {
                streams
            },
            low,
            width,
            scalar.residual_elements,
        ];
        for w in widths {
            let words = w.checked_mul(MAX_BATCH).ok_or(Error::ArithmeticOverflow)?;
            validate_view(MAX_BATCH, 0, w, w, words)?;
            u::grid(w)?;
            BufferLayout::new(words).map_err(u::fault)?;
        }
        Ok(Self {
            streams,
            width,
            low,
            widths,
        })
    }
    fn words(self) -> Result<[usize; N]> {
        let mut out = [0; N];
        for (a, w) in out.iter_mut().zip(self.widths) {
            *a = w.checked_mul(MAX_BATCH).ok_or(Error::ArithmeticOverflow)?;
        }
        Ok(out)
    }
    fn requests(self) -> Result<[u64; N]> {
        let mut out = [0; N];
        for (a, w) in out.iter_mut().zip(self.words()?) {
            *a = BufferLayout::new(w).map_err(u::fault)?.owned_bytes as u64;
        }
        Ok(out)
    }
}

fn distinct<T: PartialEq>(ids: &[T]) -> Result<()> {
    if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
pub(crate) fn full_private(arena: &PrivateArena, id: BufferId, words: usize) -> Result<Handle> {
    let l = arena.layout(id).map_err(u::fault)?;
    let h = arena.handle(id).map_err(u::fault)?.clone();
    if l.logical_words != words
        || l != BufferLayout::new(words).map_err(u::fault)?
        || h.size() != l.owned_bytes as u64
        || h.size_in_used() != h.size()
        || h.offset_start.unwrap_or(0) != 0
        || h.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(h)
}
pub(crate) fn physical(arena: &PrivateArena, owners: &[Handle]) -> Result<()> {
    let stream = owners.first().ok_or(Error::InvalidBufferLength)?.stream;
    if owners.iter().any(|h| h.stream != stream) {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    let resources = u::resources(client, owners.to_vec())?;
    if resources.len() != owners.len() {
        return Err(Error::InvalidBufferLength);
    }
    for (resource, h) in resources.iter().zip(owners) {
        let r = resource.resource();
        validate_full_extent(r.offset, r.size, r.buffer.size(), h.size())?;
    }
    for (i, a) in resources.iter().enumerate() {
        let a = a.resource();
        let ae = a
            .offset
            .checked_add(a.size)
            .ok_or(Error::ArithmeticOverflow)?;
        for b in resources.iter().skip(i + 1) {
            let b = b.resource();
            let be = b
                .offset
                .checked_add(b.size)
                .ok_or(Error::ArithmeticOverflow)?;
            if a.buffer == b.buffer && a.offset < be && b.offset < ae {
                return Err(Error::InvalidBufferLength);
            }
        }
    }
    Ok(())
}
fn validate_full_extent(offset: u64, size: u64, backing: u64, owned: u64) -> Result<()> {
    if owned == 0 || offset != 0 || size != owned || backing != owned {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Available,
    ReadQueued,
    Poisoned,
}
#[derive(Clone, Copy)]
struct PendingWrite {
    source: BufferId,
    block: BufferId,
    output: BufferId,
    count: usize,
}
pub struct Workspace {
    binding: GraphBindingId,
    layout: Layout,
    ids: [BufferId; N],
    phase: Phase,
    pending_write: Option<PendingWrite>,
    public_grouped_status: crate::status_fold_batch::PublicPolicy,
}
impl Workspace {
    /// Identity of the original MIXED owner for transaction preparation. This
    /// neither reads values nor authorizes a GR result; normal read/combine and
    /// whole-model output proofs are still required before state acceptance.
    pub(crate) fn invocation_input_owner(&self) -> Result<BufferId> {
        self.validate_all_consumers_queued()?;
        Ok(self.ids[MIXED])
    }

    pub(crate) fn public_grouped_status_folds(&self) -> (bool, u64) {
        self.public_grouped_status.state()
    }
    pub(crate) fn validate_public_grouped_status_selection(&self) -> Result<()> {
        self.validate_all_consumers_queued()?;
        self.public_grouped_status.validate_selection()
    }
    pub(crate) fn enable_public_grouped_status_folds(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) {
        self.public_grouped_status.enable();
    }

    pub fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; N]> {
        let g = graph.geometry();
        Layout::new(g.residual.streams, g.residual.width, g.low_rank)?.requests()
    }
    pub fn allocate(graph: BoundGraph<'_, '_>, arena: &mut PrivateArena) -> Result<Self> {
        graph.verify_unchanged()?;
        let g = graph.geometry();
        let layout = Layout::new(g.residual.streams, g.residual.width, g.low_rank)?;
        let needed = layout.requests()?.into_iter().try_fold(0u64, |n, b| {
            n.checked_add(b).ok_or(Error::ArithmeticOverflow)
        })?;
        if needed
            > arena
                .budget_bytes()
                .checked_sub(arena.charged_bytes())
                .ok_or(Error::ArithmeticOverflow)? as u64
        {
            return Err(Error::LimitExceeded("GR complete batch pages"));
        }
        // Caller keeps partially allocated arena owners and erases on error.
        let w = layout.words()?;
        let ids = [
            arena.allocate_zeroed(w[0]).map_err(u::fault)?,
            arena.allocate_zeroed(w[1]).map_err(u::fault)?,
            arena.allocate_zeroed(w[2]).map_err(u::fault)?,
            arena.allocate_zeroed(w[3]).map_err(u::fault)?,
            arena.allocate_zeroed(w[4]).map_err(u::fault)?,
            arena.allocate_zeroed(w[5]).map_err(u::fault)?,
            arena.allocate_zeroed(w[6]).map_err(u::fault)?,
            arena.allocate_zeroed(w[7]).map_err(u::fault)?,
            arena.allocate_zeroed(w[8]).map_err(u::fault)?,
            arena.allocate_zeroed(w[9]).map_err(u::fault)?,
        ];
        Ok(Self {
            binding: graph.binding_id(),
            layout,
            ids,
            phase: Phase::Available,
            pending_write: None,
            public_grouped_status: crate::status_fold_batch::PublicPolicy::default(),
        })
    }
    /// Full-model coordinator must also reject abandoned read capabilities.
    /// Available means all consumers/folds QUEUED, not GPU success or credit.
    pub fn validate_all_consumers_queued(&self) -> Result<()> {
        if self.phase != Phase::Available || self.pending_write.is_some() {
            return Err(Error::InvalidFormat("GR batch transaction unfinished"));
        }
        Ok(())
    }
    pub(crate) fn flush_pending(
        &mut self,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let Some(pending) = self.pending_write.take() else {
            return Ok(());
        };
        if self.phase != Phase::Available {
            return Err(Error::InvalidFormat("GR pending write unavailable"));
        }
        self.phase = Phase::Poisoned;
        u::cancel(cancel)?;
        self.prove(
            arena,
            pending.source,
            fault,
            &[
                (pending.block, MAX_BATCH * self.layout.width),
                (
                    pending.output,
                    MAX_BATCH * self.layout.streams * self.layout.width,
                ),
            ],
            None,
        )?;
        crate::strata_gr::write(
            arena,
            pending.source,
            pending.block,
            self.ids[INJECTION],
            pending.output,
            fault,
            pending.count,
        )?;
        u::cancel(cancel)?;
        self.phase = Phase::Available;
        Ok(())
    }
    fn prove(
        &self,
        arena: &PrivateArena,
        residual: BufferId,
        fault: BufferId,
        extra: &[(BufferId, usize)],
        gamma: Option<&Handle>,
    ) -> Result<()> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(N + 2 + extra.len())
            .map_err(|_| Error::LimitExceeded("GR owner metadata"))?;
        ids.extend(self.ids);
        ids.extend([residual, fault]);
        ids.extend(extra.iter().map(|x| x.0));
        distinct(&ids)?;
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(ids.len() + usize::from(gamma.is_some()))
            .map_err(|_| Error::LimitExceeded("GR owner metadata"))?;
        for (id, n) in self.ids.into_iter().zip(self.layout.words()?) {
            handles.push(full_private(arena, id, n)?);
        }
        handles.push(full_private(
            arena,
            residual,
            MAX_BATCH * self.layout.streams * self.layout.width,
        )?);
        handles.push(full_private(arena, fault, 1)?);
        for &(id, n) in extra {
            handles.push(full_private(arena, id, n)?);
        }
        if let Some(gamma) = gamma {
            let logical = (self.layout.streams * self.layout.width * 4) as u64;
            let page = BufferLayout::new(self.layout.streams * self.layout.width)
                .map_err(u::fault)?
                .owned_bytes as u64;
            if gamma.size_in_used() != logical
                || gamma.size() != page
                || gamma.offset_start.unwrap_or(0) != 0
            {
                return Err(Error::InvalidBufferLength);
            }
            let mut full = gamma.clone();
            full.offset_end = None;
            if full.size_in_used() != page {
                return Err(Error::InvalidBufferLength);
            }
            handles.push(full);
        }
        physical(arena, &handles)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn read<'w, 'a>(
        &'w mut self,
        graph: BoundGraph<'_, '_>,
        arena: &'a PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl BatchedResidentProjector,
        kind: Kind,
        residual: BufferId,
        count: usize,
        fault: BufferId,
        cancel: &'a AtomicBool,
    ) -> Result<Prepared<'w, 'a>> {
        // A compact final row must leave every earlier residual materialized
        // for prompt/MTP history. Other GR reads consume the complete write.
        if crate::strata_intel_core::enabled() && kind == Kind::Final {
            self.flush_pending(arena, fault, cancel)?;
        }
        if self.phase != Phase::Available {
            self.phase = Phase::Poisoned;
            return Err(Error::InvalidFormat("GR batch unavailable"));
        }
        self.phase = Phase::Poisoned; // A panic/error cannot authorize scratch reuse.
        u::cancel(cancel)?;
        graph.verify_unchanged()?;
        if self.binding != graph.binding_id() || projector.global_fault() != fault {
            return Err(Error::InvalidBufferLength);
        }
        public.validate_native_batch_owner(graph, arena)?;
        let base = kind.base(graph.geometry().layers)?;
        let l = self.layout;
        let (count, input_first) =
            read_columns(kind, count, l.streams * l.width, u::words(arena, residual)?)?;
        let gamma = public.vector(graph, &format!("{base}_norm.weight"))?;
        self.prove(arena, residual, fault, &[], Some(&gamma))?;
        if crate::strata_intel_core::enabled() {
            if (l.streams, l.width, l.low) != (4, 2560, 320) {
                return Err(Error::InvalidBufferLength);
            }
            let pending = self.pending_write.take();
            if let Some(p) = pending {
                if p.output != residual || p.count != count || input_first != 0 {
                    return Err(Error::InvalidBufferLength);
                }
                self.prove(
                    arena,
                    residual,
                    fault,
                    &[
                        (p.source, MAX_BATCH * l.streams * l.width),
                        (p.block, MAX_BATCH * l.width),
                    ],
                    Some(&gamma),
                )?;
            }
            let bind = |name: String| -> Result<BoundProjection<'_>> {
                let tensor = graph.tensor(&name)?;
                let matrix = Matrix::new(tensor)?;
                Ok(BoundProjection { tensor, matrix })
            };
            let projections = [
                bind(format!("{base}_down.weight"))?,
                bind(format!("{base}_up.weight"))?,
            ];
            let injection = if kind.inject() {
                Some(bind(format!("{base}_inject.weight"))?)
            } else {
                None
            };
            let queued = projector.enqueue_native_gr(
                projections,
                injection,
                crate::strata_gr::Read {
                    arena,
                    residual,
                    previous: pending.map(|p| (p.source, p.block)),
                    gamma,
                    buffers: crate::strata_gr::Buffers {
                        normalized: self.ids[NORM],
                        scales: self.ids[SN],
                        partial: self.ids[UP],
                        down: self.ids[DOWN],
                        low: self.ids[LOW],
                        injection: self.ids[INJECTION],
                        mixed: self.ids[MIXED],
                    },
                    fault,
                    count,
                    first: input_first,
                    epsilon: graph.geometry().epsilon,
                    inject: kind.inject(),
                },
                cancel,
            )?;
            if !queued {
                return Err(Error::InvalidFormat(
                    "connected native GR projector required",
                ));
            }
            u::cancel(cancel)?;
            self.phase = Phase::ReadQueued;
            return Ok(Prepared {
                workspace: self,
                arena,
                residual,
                fault,
                count: Zeroizing::new([count]),
                kind,
                cancel,
                consumed: false,
            });
        }
        let client = arena.client().map_err(u::fault)?;
        // Missing active status stores must not inherit success from a prior
        // layer. Floating scratch is fully overwritten for every active row.
        let clear_ids = Zeroizing::new([self.ids[SN], self.ids[SL], self.ids[SM]]);
        u::zero_many(arena, &*clear_ids)?;
        if crate::next_shared::gr_selected() {
            crate::next_shared::gr_norm(
                client,
                [
                    u::handle(arena, residual)?,
                    gamma,
                    u::handle(arena, self.ids[NORM])?,
                    u::handle(arena, self.ids[SN])?,
                ],
                count,
                l.streams,
                l.width,
                input_first,
                graph.geometry().epsilon,
            )?;
        } else {
            r::grouped_rms_norm::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(l.streams as u32, count as u32, 1),
                CubeDim::new_1d(64),
                u::arg(arena, residual)?,
                u::raw_arg(gamma, l.streams * l.width)?,
                u::arg(arena, self.ids[NORM])?,
                u::arg(arena, self.ids[SN])?,
                graph.geometry().epsilon,
                count,
                input_first,
                l.streams * l.width,
                0,
                l.streams * l.width,
                0,
                l.streams,
                l.streams,
                l.width,
            );
        }
        if !(self.public_grouped_status.state().0 || crate::status_fold_batch::selected()) {
            self.fold(arena, SN, count, fault)?;
        }
        self.project(
            graph,
            arena,
            projector,
            &format!("{base}_down.weight"),
            NORM,
            DOWN,
            count,
            cancel,
        )?;
        r::activate_down::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.low.div_ceil(64) as u32, count as u32, 1),
            CubeDim::new_1d(64),
            u::arg(arena, self.ids[DOWN])?,
            u::arg(arena, self.ids[LOW])?,
            u::arg(arena, self.ids[SL])?,
            count,
            0,
            l.low,
            0,
            l.low,
            0,
            l.low,
            l.streams,
            l.low,
        );
        if !(self.public_grouped_status.state().0 || crate::status_fold_batch::selected()) {
            self.fold(arena, SL, count, fault)?;
        }
        self.project(
            graph,
            arena,
            projector,
            &format!("{base}_up.weight"),
            LOW,
            UP,
            count,
            cancel,
        )?;
        r::mix::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(l.width.div_ceil(64) as u32, count as u32, 1),
            CubeDim::new_1d(64),
            u::arg(arena, self.ids[NORM])?,
            u::arg(arena, self.ids[UP])?,
            u::arg(arena, self.ids[MIXED])?,
            u::arg(arena, self.ids[SM])?,
            count,
            0,
            l.streams * l.width,
            0,
            l.streams * l.width,
            0,
            l.width,
            0,
            l.width,
            l.streams,
            l.width,
        );
        if self.public_grouped_status.state().0 || crate::status_fold_batch::selected() {
            let views = [SN, SL, SM].map(|i| {
                crate::status_fold_batch::View::new(self.ids[i], count, 0, l.widths[i], l.widths[i])
            });
            if self.public_grouped_status.state().0 {
                self.public_grouped_status.launch(arena, &views, fault)?;
            } else {
                crate::status_fold_batch::launch(arena, &views, fault)?;
            }
        } else {
            self.fold(arena, SM, count, fault)?;
        }
        if kind.inject() {
            self.project(
                graph,
                arena,
                projector,
                &format!("{base}_inject.weight"),
                NORM,
                INJECTION,
                count,
                cancel,
            )?;
        }
        u::cancel(cancel)?;
        self.phase = Phase::ReadQueued;
        Ok(Prepared {
            workspace: self,
            arena,
            residual,
            fault,
            count: Zeroizing::new([count]),
            kind,
            cancel,
            consumed: false,
        })
    }
    fn fold(
        &self,
        arena: &PrivateArena,
        status: usize,
        count: usize,
        fault: BufferId,
    ) -> Result<()> {
        let width = self.layout.widths[status];
        u::status_batch(arena, self.ids[status], fault, count, 0, width, width)
    }
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        p: &mut impl BatchedResidentProjector,
        name: &str,
        input: usize,
        output: usize,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let tensor = graph.tensor(name)?;
        let matrix = Matrix::new(tensor)?;
        if matrix.experts() != 1
            || matrix.inputs() != self.layout.widths[input]
            || matrix.outputs() != self.layout.widths[output]
        {
            return Err(Error::InvalidBufferLength);
        }
        let spans = BatchSpans::new(
            count,
            0,
            0,
            self.layout.widths[input],
            0,
            self.layout.widths[output],
        );
        p.enqueue_batch(
            BoundProjection { tensor, matrix },
            arena,
            self.ids[input],
            self.ids[output],
            &spans,
            cancel,
        )
    }
}

/// One final-head capability for the already dead normalization owner. It is
/// minted only after Final mix and its status folds are queued; no clone/copy or
/// public constructor can authorize another workspace/arena/role combination.
pub struct FinalHeadScratch<'a> {
    arena: &'a PrivateArena,
    ids: Zeroizing<[BufferId; 4]>,
}
impl FinalHeadScratch<'_> {
    pub(crate) fn into_planes(
        self,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        fault: BufferId,
    ) -> Result<BufferId> {
        final_head_roles(
            &self.ids,
            std::ptr::eq(self.arena, arena),
            input,
            output,
            fault,
        )
    }
}

/// A distinct all-row capability, minted after FinalAllRows/FinalFourRows mix
/// and its status folds. Constructor capacity is independent of active count;
/// it authorizes the same dead full NORM owner, never a narrowed plane view.
pub struct BatchedFinalHeadScratch<'a> {
    arena: &'a PrivateArena,
    ids: Zeroizing<[BufferId; 4]>,
    kind: Kind,
    count: usize,
    four_reserved: bool,
}
impl BatchedFinalHeadScratch<'_> {
    pub(crate) fn into_planes(
        self,
        arena: &PrivateArena,
        input: BufferId,
        output: BufferId,
        fault: BufferId,
        spans: &BatchSpans,
    ) -> Result<(BufferId, bool)> {
        batched_final_head_roles(self.kind, self.count, self.four_reserved, spans)?;
        let planes = final_head_roles(
            &self.ids,
            std::ptr::eq(self.arena, arena),
            input,
            output,
            fault,
        )?;
        Ok((planes, self.four_reserved))
    }
}
fn batched_final_head_roles(
    kind: Kind,
    count: usize,
    four_reserved: bool,
    spans: &BatchSpans,
) -> Result<()> {
    let expected = final_head_spans(kind, count, 2560, 248320)?;
    if !matches!(kind, Kind::FinalAllRows | Kind::FinalFourRows)
        || (kind == Kind::FinalFourRows
            && (!four_reserved || !(4..=crate::strata_port::verify_capacity()).contains(&count)))
        || spans.count() != expected.count()
        || spans.packed_first() != 0
        || spans.input_first() != expected.input_first()
        || spans.input_stride() != expected.input_stride()
        || spans.output_first() != expected.output_first()
        || spans.output_stride() != expected.output_stride()
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
fn final_head_roles(
    ids: &[BufferId; 4],
    same_arena: bool,
    input: BufferId,
    output: BufferId,
    fault: BufferId,
) -> Result<BufferId> {
    if !same_arena || ids[0] != input || ids[2] != output || ids[3] != fault {
        return Err(Error::InvalidBufferLength);
    }
    distinct(ids)?;
    Ok(ids[1])
}

/// Borrow prevents workspace reuse and mutable arena reset before its consumers
/// are queued. Dropping it without combine/final-mix poisons the workspace.
/// It conveys no state acceptance or completed fence.
pub struct Prepared<'w, 'a> {
    workspace: &'w mut Workspace,
    arena: &'a PrivateArena,
    residual: BufferId,
    fault: BufferId,
    count: Zeroizing<[usize; 1]>,
    kind: Kind,
    cancel: &'a AtomicBool,
    consumed: bool,
}
impl Prepared<'_, '_> {
    pub fn mixed(&self) -> BufferId {
        self.workspace.ids[MIXED]
    }
    pub fn combine(mut self, block: BufferId, output: BufferId) -> Result<()> {
        self.workspace.phase = Phase::Poisoned;
        u::cancel(self.cancel)?;
        if !self.kind.inject() {
            return Err(Error::InvalidFormat("GR final mix has no combine"));
        }
        let l = self.workspace.layout;
        self.workspace.prove(
            self.arena,
            self.residual,
            self.fault,
            &[
                (block, MAX_BATCH * l.width),
                (output, MAX_BATCH * l.streams * l.width),
            ],
            None,
        )?;
        if crate::strata_intel_core::enabled() {
            if self.workspace.pending_write.is_some() {
                return Err(Error::InvalidFormat("GR write already pending"));
            }
            self.workspace.pending_write = Some(PendingWrite {
                source: self.residual,
                block,
                output,
                count: self.count[0],
            });
            self.consumed = true;
            self.workspace.phase = Phase::Available;
            return Ok(());
        }
        u::zero(self.arena, self.workspace.ids[SC])?;
        r::combine::launch::<WgpuRuntime>(
            self.arena.client().map_err(u::fault)?,
            CubeCount::Static(
                (l.streams * l.width).div_ceil(64) as u32,
                self.count[0] as u32,
                1,
            ),
            CubeDim::new_1d(64),
            u::arg(self.arena, self.residual)?,
            u::arg(self.arena, block)?,
            u::arg(self.arena, self.workspace.ids[INJECTION])?,
            u::arg(self.arena, output)?,
            u::arg(self.arena, self.workspace.ids[SC])?,
            self.count[0],
            0,
            l.streams * l.width,
            0,
            l.width,
            0,
            l.streams,
            0,
            l.streams * l.width,
            0,
            l.streams * l.width,
            l.streams,
            l.width,
        );
        self.workspace
            .fold(self.arena, SC, self.count[0], self.fault)?;
        u::cancel(self.cancel)?;
        self.consumed = true;
        self.workspace.phase = Phase::Available;
        Ok(())
    }
    /// Consume the final mixed scratch only after its original output head
    /// has been queued. Never release this capability by returning its ID.
    /// Only the last active prompt column requests logits; earlier chunks
    /// omit the entire final mixer/head and accept their residual fault fence.
    pub fn enqueue_final_head(
        self,
        graph: BoundGraph<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        logits: BufferId,
    ) -> Result<()> {
        self.enqueue_final_head_with_target_capacity(graph, projector, logits, false)
    }
    /// The parent supplies the actual target-output Owners constructor flag.
    /// Counts1..3 retain their original spans but prove the whole reserved owner.
    pub fn enqueue_final_head_with_target_capacity(
        mut self,
        graph: BoundGraph<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        logits: BufferId,
        four_reserved: bool,
    ) -> Result<()> {
        self.workspace.phase = Phase::Poisoned;
        if !matches!(
            self.kind,
            Kind::Final | Kind::FinalAllRows | Kind::FinalFourRows
        ) {
            return Err(Error::InvalidFormat("GR injection consumer missing"));
        }
        u::cancel(self.cancel)?;
        graph.verify_unchanged()?;
        if graph.binding_id() != self.workspace.binding || projector.global_fault() != self.fault {
            return Err(Error::InvalidBufferLength);
        }
        let vocabulary = graph.geometry().vocabulary;
        let owner_words =
            final_head_owner_words(self.kind, self.count[0], vocabulary, four_reserved)?;
        self.workspace.prove(
            self.arena,
            self.residual,
            self.fault,
            &[(logits, owner_words)],
            None,
        )?;
        let tensor = graph.tensor("output.weight")?;
        let matrix = Matrix::new(tensor)?;
        let width = self.workspace.layout.width;
        if matrix.experts() != 1 || matrix.inputs() != width || matrix.outputs() != vocabulary {
            return Err(Error::InvalidBufferLength);
        }
        let all_rows = matches!(self.kind, Kind::FinalAllRows | Kind::FinalFourRows);
        let spans = final_head_spans(self.kind, self.count[0], width, vocabulary)?;
        if all_rows {
            let scratch = BatchedFinalHeadScratch {
                arena: self.arena,
                ids: Zeroizing::new([
                    self.workspace.ids[MIXED],
                    self.workspace.ids[NORM],
                    logits,
                    self.fault,
                ]),
                kind: self.kind,
                count: self.count[0],
                four_reserved,
            };
            // Default projectors retain original arithmetic. The explicit
            // integer variant consumes every prepared row before full erase.
            let used = projector.enqueue_batched_final_head(
                BoundProjection { tensor, matrix },
                self.arena,
                self.workspace.ids[MIXED],
                logits,
                &spans,
                scratch,
                self.cancel,
            )?;
            if used {
                u::zero(self.arena, self.workspace.ids[NORM])?;
            }
            u::cancel(self.cancel)?;
            self.consumed = true;
            self.workspace.phase = Phase::Available;
            return Ok(());
        }
        let scratch = FinalHeadScratch {
            arena: self.arena,
            ids: Zeroizing::new([
                self.workspace.ids[MIXED],
                self.workspace.ids[NORM],
                logits,
                self.fault,
            ]),
        };
        // Final omits injection: after the queued MIXED producer and folds,
        // NORM has no reader left. Same-stream head work must finish reading
        // its prepared planes before the full owner is cleared below.
        let used = projector.enqueue_final_head(
            BoundProjection { tensor, matrix },
            self.arena,
            self.workspace.ids[MIXED],
            logits,
            &spans,
            scratch,
            self.cancel,
        )?;
        if used {
            u::zero(self.arena, self.workspace.ids[NORM])?;
        }
        u::cancel(self.cancel)?;
        self.consumed = true;
        self.workspace.phase = Phase::Available;
        Ok(())
    }
}
impl Drop for Prepared<'_, '_> {
    fn drop(&mut self) {
        if !self.consumed {
            self.workspace.phase = Phase::Poisoned;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reserved_four_head_prefill_and_tails_prove_full_fixed_owner_with_original_spans() {
        let vocabulary = 248320;
        let original = BufferLayout::new(3 * vocabulary).unwrap();
        let reserved = BufferLayout::new(4 * vocabulary).unwrap();
        // Equal physical allocation is insufficient: full_private requires the
        // exact logical constructor extent even for the first count1 prefill.
        assert_eq!(original.owned_bytes, reserved.owned_bytes);
        assert_eq!(reserved.owned_bytes, 4 * 1024 * 1024);
        assert_ne!(original.logical_words, reserved.logical_words);
        for count in 1..=3 {
            assert_eq!(
                final_head_owner_words(Kind::FinalAllRows, count, vocabulary, false).unwrap(),
                original.logical_words
            );
            assert_eq!(
                final_head_owner_words(Kind::FinalAllRows, count, vocabulary, true).unwrap(),
                reserved.logical_words
            );
            let (active, first) =
                read_columns(Kind::FinalAllRows, count, 10240, MAX_BATCH * 10240).unwrap();
            assert_eq!((active, first), (count, 0));
            let spans = final_head_spans(Kind::FinalAllRows, active, 2560, vocabulary).unwrap();
            assert_eq!((spans.count(), spans.input_first()), (count, 0));
            assert_eq!(
                (spans.input_stride(), spans.output_stride()),
                (2560, vocabulary)
            );
            assert_eq!(spans.output_first(), 0);
            for owner in [original.logical_words, reserved.logical_words] {
                assert_eq!(
                    validate_view(count, 0, vocabulary, vocabulary, owner).unwrap(),
                    count * vocabulary
                );
            }
        }
        assert_eq!(
            final_head_owner_words(Kind::Final, 1, vocabulary, false).unwrap(),
            vocabulary
        );
    }
    #[test]
    fn actual_four_head_requires_reserved_owner_and_complete_original_four_spans() {
        let vocabulary = 248320;
        let words = final_head_owner_words(Kind::FinalFourRows, 4, vocabulary, true).unwrap();
        assert_eq!(words, 4 * vocabulary);
        let (active, first) =
            read_columns(Kind::FinalFourRows, 4, 10240, MAX_BATCH * 10240).unwrap();
        assert_eq!((active, first), (4, 0));
        let spans = final_head_spans(Kind::FinalFourRows, active, 2560, vocabulary).unwrap();
        assert_eq!(
            (spans.count(), spans.input_first(), spans.output_first()),
            (4, 0, 0)
        );
        assert_eq!(
            (spans.input_stride(), spans.output_stride()),
            (2560, vocabulary)
        );
        assert_eq!(
            validate_view(4, 0, vocabulary, vocabulary, words).unwrap(),
            words
        );
        assert!(validate_view(4, 0, vocabulary, vocabulary, 3 * vocabulary).is_err());
        assert!(final_head_owner_words(Kind::FinalFourRows, 4, vocabulary, false).is_err());
        assert!(final_head_owner_words(Kind::FinalAllRows, 4, vocabulary, true).is_err());
    }
    #[test]
    fn final_head_owner_capacity_rejects_invalid_roles_counts_and_extents() {
        for four in [false, true] {
            for count in [0, 4, 8, usize::MAX] {
                assert!(final_head_owner_words(Kind::FinalAllRows, count, 248320, four).is_err());
            }
            for count in [0, 1, 2, 3, 5, usize::MAX] {
                assert!(final_head_owner_words(Kind::FinalFourRows, count, 248320, four).is_err());
            }
            for kind in [Kind::Attention(0), Kind::Ffn(0)] {
                assert!(final_head_owner_words(kind, 1, 248320, four).is_err());
            }
            assert!(final_head_owner_words(Kind::FinalAllRows, 1, 0, four).is_err());
            assert!(final_head_owner_words(Kind::FinalAllRows, 1, usize::MAX, four).is_err());
        }
        assert!(final_head_owner_words(Kind::Final, 1, 248320, true).is_err());
        assert!(final_head_owner_words(Kind::Final, 2, 248320, false).is_err());
    }
    #[test]
    fn all_row_final_mixer_and_head_keep_complete_original_row_strides() {
        for count in 1..=3 {
            let (active, first) =
                read_columns(Kind::FinalAllRows, count, 10240, MAX_BATCH * 10240).unwrap();
            assert_eq!((active, first), (count, 0));
            let spans = final_head_spans(Kind::FinalAllRows, active, 2560, 248320).unwrap();
            assert_eq!(spans.count(), count);
            assert_eq!(spans.input_first(), 0);
            assert_eq!(spans.input_stride(), 2560);
            assert_eq!(spans.output_first(), 0);
            assert_eq!(spans.output_stride(), 248320);
            assert_eq!(
                validate_view(count, 0, 248320, 248320, 3 * 248320).unwrap(),
                count * 248320
            );
            // Old Final still compacts its selected source into scratch row0.
            let (last_active, last_first) =
                read_columns(Kind::Final, count, 10240, MAX_BATCH * 10240).unwrap();
            assert_eq!((last_active, last_first), (1, (count - 1) * 10240));
            let last = final_head_spans(Kind::Final, last_active, 2560, 248320).unwrap();
            assert_eq!((last.count(), last.input_first()), (1, 0));
        }
        assert_eq!(Kind::FinalAllRows.base(48).unwrap(), "output_hc");
        assert!(!Kind::FinalAllRows.inject());
        for count in [0, 4, 8, usize::MAX] {
            assert!(read_columns(Kind::FinalAllRows, count, 10240, MAX_BATCH * 10240).is_err());
            assert!(final_head_spans(Kind::FinalAllRows, count, 2560, 248320).is_err());
        }
    }
    #[test]
    fn final_head_scratch_rejects_foreign_stale_reversed_and_aliased_roles() {
        let ids = [0, 1, 2, 3].map(|slot| BufferId::metadata_fixture(7, 4, slot));
        assert!(final_head_roles(&ids, true, ids[0], ids[2], ids[3]).unwrap() == ids[1]);
        assert!(final_head_roles(&ids, false, ids[0], ids[2], ids[3]).is_err());
        for bad in [
            BufferId::metadata_fixture(8, 4, 0),
            BufferId::metadata_fixture(7, 5, 0),
            ids[2],
        ] {
            assert!(final_head_roles(&ids, true, bad, ids[2], ids[3]).is_err());
        }
        for alias in [ids[0], ids[2], ids[3]] {
            let mut bad = ids;
            bad[1] = alias;
            assert!(final_head_roles(&bad, true, ids[0], ids[2], ids[3]).is_err());
        }
        assert!(final_head_roles(&ids, true, ids[0], ids[0], ids[3]).is_err());
        assert!(final_head_roles(&ids, true, ids[0], ids[2], ids[1]).is_err());
    }
    #[test]
    fn q6_scratch_fits_already_admitted_final_norm_and_head_info_envelope() {
        let layout = Layout::new(4, 2560, 320).unwrap();
        let original = layout.requests().unwrap();
        let norm_words = layout.words().unwrap()[NORM];
        let spec = crate::q6_integer_matvec::Spec::new(16384, 2560).unwrap();
        assert_eq!(spec.plane_words(), 1440);
        assert_eq!(original[NORM], 524_288);
        assert!(spec.plane_words() < 4 * 2560);
        assert!(4 * 2560 <= norm_words);
        assert_eq!(
            original.iter().sum::<u64>(),
            1_998_848
                + if crate::strata_intel_core::enabled() {
                    crate::strata_gr::PRIVATE_INCREMENT
                } else {
                    0
                }
        );
        assert_eq!(layout.requests().unwrap(), original);
        let tiles = 248320usize.div_ceil(16384);
        assert_eq!(tiles, 16);
        // Prepare + original sixteen projections + full scratch clear stay
        // below runtime_bytes' unchanged two-per-tile+one bound.
        let actual = 1 + tiles + 1;
        let admitted = 2 * tiles + 1;
        assert!(actual <= admitted);
    }
    #[test]
    fn batched_q6_capability_binds_real_count_and_constructor_capacity() {
        for count in 1..=3 {
            let spans = final_head_spans(Kind::FinalAllRows, count, 2560, 248320).unwrap();
            for reserved in [false, true] {
                batched_final_head_roles(Kind::FinalAllRows, count, reserved, &spans).unwrap();
            }
            assert!(batched_final_head_roles(Kind::FinalFourRows, count, true, &spans).is_err());
            let wrong = BatchSpans::new(count, 0, 1, 2560, 0, 248320);
            assert!(batched_final_head_roles(Kind::FinalAllRows, count, false, &wrong).is_err());
        }
        let four = final_head_spans(Kind::FinalFourRows, 4, 2560, 248320).unwrap();
        batched_final_head_roles(Kind::FinalFourRows, 4, true, &four).unwrap();
        assert!(batched_final_head_roles(Kind::FinalFourRows, 4, false, &four).is_err());
        assert!(batched_final_head_roles(Kind::FinalAllRows, 4, true, &four).is_err());
        assert!(batched_final_head_roles(Kind::Final, 1, false, &four).is_err());
        for wrong in [
            BatchSpans::new(4, 1, 0, 2560, 0, 248320),
            BatchSpans::new(4, 0, 0, 2561, 0, 248320),
            BatchSpans::new(4, 0, 0, 2560, 1, 248320),
            BatchSpans::new(4, 0, 0, 2560, 0, 248321),
        ] {
            assert!(batched_final_head_roles(Kind::FinalFourRows, 4, true, &wrong).is_err());
        }
    }
    #[test]
    fn batched_q6_four_planes_fit_full_original_norm_with_no_extra_charge() {
        let layout = Layout::new(4, 2560, 320).unwrap();
        let requests = layout.requests().unwrap();
        let norm = layout.words().unwrap()[NORM];
        let spec = crate::q6_integer_matvec::Spec::new(16384, 2560).unwrap();
        assert_eq!(norm, 81920);
        assert_eq!(requests[NORM], 524288);
        for count in 1..=4 {
            assert_eq!(count * spec.plane_words(), count * 1440);
            assert!(count * spec.plane_words() < norm);
            let owner = final_head_owner_words(
                if count == 4 {
                    Kind::FinalFourRows
                } else {
                    Kind::FinalAllRows
                },
                count,
                248320,
                count == 4,
            )
            .unwrap();
            assert_eq!(owner, if count == 4 { 993280 } else { 744960 });
        }
        let tiles = 248320usize.div_ceil(spec.rows());
        let actual = 1 + tiles + 1;
        let admitted = 2 * tiles + 1;
        assert_eq!((actual, admitted), (18, 33));
        assert!(actual * 2048 < admitted * 2048);
        assert_eq!(layout.requests().unwrap(), requests);
    }
    #[test]
    fn full_owner_proof_rejects_interior_resource_and_hidden_physical_tail() {
        for owned in [32768, 131072, 524288] {
            assert!(validate_full_extent(0, owned, owned, owned).is_ok());
            assert!(validate_full_extent(4, owned, owned + 4, owned).is_err());
            assert!(validate_full_extent(0, owned, owned * 2, owned).is_err());
            assert!(validate_full_extent(0, owned - 4, owned, owned).is_err());
            assert!(validate_full_extent(0, owned, owned - 4, owned).is_err());
        }
        assert!(validate_full_extent(0, 0, 0, 0).is_err());
    }
    #[test]
    fn actual_max8_pages_are_admitted_once_and_never_logical_payload_only() {
        let l = Layout::new(4, 2560, 320).unwrap();
        assert_eq!(
            l.requests().unwrap(),
            [
                131072,
                32768,
                524288,
                32768,
                32768,
                if crate::strata_intel_core::enabled() {
                    1048576
                } else {
                    524288
                },
                32768,
                32768,
                131072,
                524288
            ]
        );
        assert_eq!(
            l.requests().unwrap().into_iter().sum::<u64>(),
            1_998_848
                + if crate::strata_intel_core::enabled() {
                    crate::strata_gr::PRIVATE_INCREMENT
                } else {
                    0
                }
        );
        for count in [1, 3, 5, 8] {
            for (width, words) in l.widths.into_iter().zip(l.words().unwrap()) {
                assert_eq!(
                    validate_view(count, 0, width, width, words).unwrap(),
                    count * width
                );
                assert!(validate_view(count, 0, width, width, count * width - 1).is_err());
            }
        }
    }
    #[test]
    fn public_roles_and_owner_aliases_fail_before_dispatch() {
        for l in [0, 1, 47] {
            assert!(Kind::Attention(l).base(48).is_ok());
            assert!(Kind::Ffn(l).base(48).is_ok());
        }
        assert!(Kind::Attention(48).base(48).is_err());
        assert!(Kind::Ffn(usize::MAX).base(48).is_err());
        assert!(!Kind::Final.inject());
        let owners = [1, 2, 3, 4];
        assert!(distinct(&owners).is_ok());
        for slot in 1..owners.len() {
            let mut changed = owners;
            changed[slot] = changed[0];
            assert!(distinct(&changed).is_err());
        }
        for (s, w, l) in [
            (0, 2560, 320),
            (4, 0, 320),
            (4, 2560, 0),
            (4, usize::MAX, 320),
        ] {
            assert!(Layout::new(s, w, l).is_err());
        }
    }
    #[test]
    fn final_mixer_selects_last_original_column_before_all_projections() {
        for width in [97, 10240] {
            for count in [1, 3, 5, 8] {
                assert_eq!(
                    read_columns(Kind::Final, count, width, MAX_BATCH * width).unwrap(),
                    (1, (count - 1) * width)
                );
                for kind in [Kind::Attention(0), Kind::Ffn(47)] {
                    assert_eq!(
                        read_columns(kind, count, width, MAX_BATCH * width).unwrap(),
                        (count, 0)
                    );
                }
                assert!(read_columns(Kind::Final, count, width, count * width - 1).is_err());
            }
            for count in [0, 9, usize::MAX] {
                assert!(read_columns(Kind::Final, count, width, MAX_BATCH * width).is_err());
            }
        }
    }
}
