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
}
impl Kind {
    fn base(self, layers: usize) -> Result<String> {
        match self {
            Self::Attention(l) if l < layers => Ok(format!("blk.{l}.hc_attn")),
            Self::Ffn(l) if l < layers => Ok(format!("blk.{l}.hc_ffn")),
            Self::Final => Ok("output_hc".into()),
            _ => Err(Error::OutOfBounds),
        }
    }
    fn inject(self) -> bool {
        self != Self::Final
    }
}

/// Final mixer reads only the last original residual column into scratch row0.
/// Attention/FFN retain every active column. Both source and scratch geometry
/// remain runtime data with fixed MAX8 owning allocation.
fn read_columns(kind: Kind, count: usize, width: usize, words: usize) -> Result<(usize, usize)> {
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
            scalar.residual_elements,
            streams,
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
pub struct Workspace {
    binding: GraphBindingId,
    layout: Layout,
    ids: [BufferId; N],
    phase: Phase,
}
impl Workspace {
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
        })
    }
    /// Full-model coordinator must also reject abandoned read capabilities.
    /// Available means all consumers/folds QUEUED, not GPU success or credit.
    pub fn validate_all_consumers_queued(&self) -> Result<()> {
        if self.phase != Phase::Available {
            return Err(Error::InvalidFormat("GR batch transaction unfinished"));
        }
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
        if !crate::status_fold_batch::selected() {
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
        if !crate::status_fold_batch::selected() {
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
        if crate::status_fold_batch::selected() {
            let views = [SN, SL, SM].map(|i| {
                crate::status_fold_batch::View::new(self.ids[i], count, 0, l.widths[i], l.widths[i])
            });
            crate::status_fold_batch::launch(arena, &views, fault)?;
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
        mut self,
        graph: BoundGraph<'_, '_>,
        projector: &mut impl BatchedResidentProjector,
        logits: BufferId,
    ) -> Result<()> {
        self.workspace.phase = Phase::Poisoned;
        if self.kind != Kind::Final {
            return Err(Error::InvalidFormat("GR injection consumer missing"));
        }
        u::cancel(self.cancel)?;
        graph.verify_unchanged()?;
        if graph.binding_id() != self.workspace.binding || projector.global_fault() != self.fault {
            return Err(Error::InvalidBufferLength);
        }
        let vocabulary = graph.geometry().vocabulary;
        self.workspace.prove(
            self.arena,
            self.residual,
            self.fault,
            &[(logits, vocabulary)],
            None,
        )?;
        let tensor = graph.tensor("output.weight")?;
        let matrix = Matrix::new(tensor)?;
        let width = self.workspace.layout.width;
        if matrix.experts() != 1 || matrix.inputs() != width || matrix.outputs() != vocabulary {
            return Err(Error::InvalidBufferLength);
        }
        let first = (self.count[0] - 1)
            .checked_mul(width)
            .ok_or(Error::ArithmeticOverflow)?;
        let spans = BatchSpans::new(1, 0, first, width, 0, vocabulary);
        projector.enqueue_batch(
            BoundProjection { tensor, matrix },
            self.arena,
            self.workspace.ids[MIXED],
            logits,
            &spans,
            self.cancel,
        )?;
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
                131072, 32768, 524288, 32768, 32768, 524288, 32768, 32768, 131072, 524288
            ]
        );
        assert_eq!(l.requests().unwrap().into_iter().sum::<u64>(), 1_998_848);
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
