//! Native MAX8 shared expert, separate from the routed ten experts.
//! Borrows original cached matrices/public scalar-gate vector. No router,
//! readback, bank/state commit, whole-model acceptance or default selection.
#![allow(dead_code)] // Whole-model chunk connection and physical acceptance pending.
use super::sigmoid;
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, validate_view},
    resident_gdn::BoundProjection,
    resident_gr::{
        PublicVectors,
        batch::{full_private, physical},
    },
    resident_util as u,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{
    Error, Result,
    cpu::Matrix,
    model::{BoundGraph, GraphBindingId},
};
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

pub(crate) const N: usize = if crate::strata_intel_core::enabled() {
    6
} else {
    5
};
pub(crate) const MAIN_BYTES: u64 = 262_144
    + if crate::strata_intel_core::enabled() {
        32_768
    } else {
        0
    };
const GATE: usize = 0;
const UP: usize = 1;
const ACT: usize = 2;
const DOWN: usize = 3;
const SCALE: usize = 4;

#[cube]
#[allow(clippy::manual_range_contains)]
fn finite(value: f32) -> bool {
    value <= 3.4028235e38_f32 && value >= -3.4028235e38_f32
}

#[cube(launch)]
fn activation(
    gate: &Array<f32>,
    up: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    #[comptime] hidden: usize,
) {
    let token = CUBE_POS_Y as usize;
    let i = ABSOLUTE_POS_X as usize;
    // Do not put storage reads into the guard: Cube eagerly evaluates AND.
    if token < count && i < hidden {
        let at = token * hidden + i;
        let g = gate[at];
        let v = up[at];
        // Preserve the original scalar expression and operation order.
        let y = g * sigmoid(g) * v;
        if finite(g) && finite(v) && finite(y) {
            output[at] = y;
        } else {
            output[at] = 0.0;
            fault[0].fetch_or(4);
        }
    }
}

#[cube(launch)]
fn gate_dot(
    input: &Array<f32>,
    weight: &Array<f32>,
    scale: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    // This condition is uniform across the entire workgroup. Inactive groups
    // never read input/weight or enter the shared-memory tree.
    if token < count {
        let mut sum = 0.0_f32;
        for tile in 0..width.div_ceil(64_usize) {
            let i = tile * 64 + lane;
            if i < width {
                sum += input[token * width + i] * weight[i];
            }
        }
        let mut sums = SharedMemory::<f32>::new(64_usize);
        sums[lane] = sum;
        sync_cube();
        let mut stride = 32_usize;
        while stride > 0 {
            if lane < stride {
                sums[lane] += sums[lane + stride];
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            let value = sigmoid(sums[0]);
            if finite(sums[0]) {
                scale[token] = value;
            } else {
                scale[token] = 0.0;
                fault[0].fetch_or(4);
            }
        }
    }
}

#[cube(launch)]
fn add(
    routed: &Array<f32>,
    shared: &Array<f32>,
    scale: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let i = ABSOLUTE_POS_X as usize;
    if token < count && i < width {
        let at = token * width + i;
        let r = routed[at];
        let s = shared[at];
        let g = scale[token];
        let y = r + s * g;
        if finite(r) && finite(s) && finite(g) && finite(y) {
            output[at] = y;
        } else {
            output[at] = 0.0;
            fault[0].fetch_or(4);
        }
    }
}

#[derive(Clone, Copy)]
struct Layout {
    width: usize,
    hidden: usize,
    words: [usize; N],
}
impl Layout {
    fn new(width: usize, hidden: usize) -> Result<Self> {
        let x = width
            .checked_mul(MAX_BATCH)
            .ok_or(Error::ArithmeticOverflow)?;
        let h = hidden
            .checked_mul(MAX_BATCH)
            .ok_or(Error::ArithmeticOverflow)?;
        let mut words = [1; N];
        words[..5].copy_from_slice(&[h, h, h, x, MAX_BATCH]);
        validate_view(MAX_BATCH, 0, width, width, x)?;
        validate_view(MAX_BATCH, 0, hidden, hidden, h)?;
        u::grid(width)?;
        u::grid(hidden)?;
        for w in words {
            BufferLayout::new(w).map_err(u::fault)?;
        }
        Ok(Self {
            width,
            hidden,
            words,
        })
    }
    fn requests(self) -> Result<[u64; N]> {
        let mut out = [0; N];
        for (request, words) in out.iter_mut().zip(self.words) {
            *request = BufferLayout::new(words).map_err(u::fault)?.owned_bytes as u64;
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Available,
    Prepared,
    Poisoned,
}
type PreparedStamp = (Zeroizing<[BufferId; 4]>, Zeroizing<[usize; 2]>);
fn consume_prepared(
    phase: &mut Phase,
    prepared: &mut Option<PreparedStamp>,
) -> Result<PreparedStamp> {
    if *phase != Phase::Prepared {
        return Err(Error::InvalidFormat("shared preparation missing"));
    }
    *phase = Phase::Poisoned;
    prepared.take().ok_or(Error::InvalidBufferLength)
}
pub struct Workspace {
    binding: GraphBindingId,
    layout: Layout,
    ids: [BufferId; N],
    phase: Phase,
    prepared: Option<PreparedStamp>,
    native_prepared: bool,
}
impl Workspace {
    /// The complete captured prepare/finish pairs leave the original scratch
    /// idle. Only a joined native target invocation may use this boundary.
    pub(crate) fn complete_native_replay(
        &self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        completed: &crate::resident_model::target_replay::Completed,
    ) -> Result<()> {
        self.validate_all_consumers_queued()?;
        completed.check_fault_owner(arena, fault)?;
        if !crate::strata_intel_core::enabled()
            || self.binding != graph.binding_id()
            || self.prepared.is_some()
            || self.native_prepared
        {
            return Err(Error::InvalidFormat("shared native replay boundary"));
        }
        for id in self.ids {
            u::words(arena, id)?;
        }
        Ok(())
    }
    pub fn allocation_requests(graph: BoundGraph<'_, '_>) -> Result<[u64; N]> {
        let g = graph.geometry();
        Layout::new(g.residual.width, g.expert_hidden)?.requests()
    }
    pub fn allocate(graph: BoundGraph<'_, '_>, arena: &mut PrivateArena) -> Result<Self> {
        graph.verify_unchanged()?;
        let g = graph.geometry();
        let layout = Layout::new(g.residual.width, g.expert_hidden)?;
        let needed = layout.requests()?.into_iter().try_fold(0u64, |sum, page| {
            sum.checked_add(page).ok_or(Error::ArithmeticOverflow)
        })?;
        if needed
            > arena
                .budget_bytes()
                .checked_sub(arena.charged_bytes())
                .ok_or(Error::ArithmeticOverflow)? as u64
        {
            return Err(Error::LimitExceeded("shared expert complete batch pages"));
        }
        // A failed partial allocation belongs to the caller's poisoned arena.
        let w = layout.words;
        let mut allocated = Vec::with_capacity(N);
        for words in w {
            allocated.push(arena.allocate_zeroed(words).map_err(u::fault)?);
        }
        let ids = allocated
            .try_into()
            .map_err(|_| Error::InvalidBufferLength)?;
        Ok(Self {
            binding: graph.binding_id(),
            layout,
            ids,
            phase: Phase::Available,
            prepared: None,
            native_prepared: false,
        })
    }
    pub fn validate_all_consumers_queued(&self) -> Result<()> {
        if self.phase == Phase::Available {
            Ok(())
        } else {
            Err(Error::InvalidFormat("shared expert batch poisoned"))
        }
    }
    fn prove(
        &self,
        arena: &PrivateArena,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        weight: &Handle,
    ) -> Result<()> {
        let mut ids = [fault; N + 4];
        ids[..N].copy_from_slice(&self.ids);
        ids[N..].copy_from_slice(&[input, routed, output, fault]);
        if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
            return Err(Error::InvalidBufferLength);
        }
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(N + 5)
            .map_err(|_| Error::LimitExceeded("shared expert owner metadata"))?;
        for (id, words) in self.ids.into_iter().zip(self.layout.words) {
            handles.push(full_private(arena, id, words)?);
        }
        let words = self.layout.width * MAX_BATCH;
        for id in [input, routed, output] {
            handles.push(full_private(arena, id, words)?);
        }
        handles.push(full_private(arena, fault, 1)?);
        let logical = (self.layout.width * 4) as u64;
        let page = BufferLayout::new(self.layout.width)
            .map_err(u::fault)?
            .owned_bytes as u64;
        if weight.size_in_used() != logical
            || weight.size() != page
            || weight.offset_start.unwrap_or(0) != 0
        {
            return Err(Error::InvalidBufferLength);
        }
        let mut full = weight.clone();
        full.offset_end = None;
        if full.size_in_used() != page {
            return Err(Error::InvalidBufferLength);
        }
        handles.push(full);
        physical(arena, &handles)
    }
    /// Serial wrapper retains the original shared branch ordering. An explicit
    /// hybrid scheduling experiment can prepare independent GPU work earlier,
    /// then combine ONLY after the routed consumer/CPU upload is queued.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.prepare(
            graph, layer, arena, public, projector, input, routed, output, fault, count, cancel,
        )?;
        self.finish_prepared(
            graph, layer, arena, public, input, routed, output, fault, count, cancel,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.prepare_impl(
            graph, layer, arena, public, projector, input, routed, output, fault, count, cancel,
            false,
        )
    }
    pub(crate) fn native_fault(&self) -> Result<BufferId> {
        if !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidBufferLength);
        }
        self.ids.get(5).copied().ok_or(Error::InvalidBufferLength)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_native(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl BatchedResidentProjector,
        planes: BufferId,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidBufferLength);
        }
        // Include the original packed scratch in the physical alias proof;
        // routed GU/DOWN use their own separately admitted global roots.
        let mut owners = Vec::with_capacity(N + 5);
        for id in self
            .ids
            .into_iter()
            .chain([planes, input, routed, output, fault])
        {
            let mut handle = u::handle(arena, id)?;
            handle.offset_end = None;
            owners.push(handle);
        }
        physical(arena, &owners)?;
        self.prepare_impl(
            graph, layer, arena, public, projector, input, routed, output, fault, count, cancel,
            true,
        )?;
        self.phase = Phase::Poisoned;
        // Both original projections have consumed the same packed input. The
        // fused producer replaces only this independent shared scratch image.
        crate::strata_dense::prepare_swiglu(
            arena.client().map_err(u::fault)?,
            self.layout.hidden,
            u::handle(arena, self.ids[GATE])?,
            u::handle(arena, self.ids[UP])?,
            u::handle(arena, planes)?,
            u::handle(arena, self.native_fault()?)?,
            count,
        )?;
        let tensor = graph.tensor(&format!("blk.{layer}.ffn_down_shexp.weight"))?;
        let matrix = Matrix::new(tensor)?;
        projector.enqueue_native_packed(
            BoundProjection { tensor, matrix },
            arena,
            self.ids[ACT],
            self.ids[DOWN],
            &BatchSpans::new(count, 0, 0, self.layout.hidden, 0, self.layout.width),
            cancel,
        )?;
        let tensor = graph.tensor(&format!("blk.{layer}.ffn_gate_inp_shexp.weight"))?;
        projector.enqueue_native_shared_gate(
            tensor,
            arena,
            input,
            self.ids[SCALE],
            &BatchSpans::new(count, 0, 0, self.layout.width, 0, 1),
            cancel,
        )?;
        crate::strata_shared_native::scale(
            arena.client().map_err(u::fault)?,
            u::handle(arena, self.ids[DOWN])?,
            u::handle(arena, self.ids[SCALE])?,
            u::handle(arena, self.native_fault()?)?,
            count,
        )?;
        u::cancel(cancel)?;
        self.phase = Phase::Prepared;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_impl(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        projector: &mut impl BatchedResidentProjector,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
        native: bool,
    ) -> Result<()> {
        if self.phase != Phase::Available {
            return Err(Error::InvalidFormat("shared expert batch poisoned"));
        }
        self.phase = Phase::Poisoned; // Remains sticky on error, panic or cancel.
        u::cancel(cancel)?;
        graph.verify_unchanged()?;
        if graph.binding_id() != self.binding
            || layer >= graph.geometry().layers
            || projector.global_fault() != if native { self.native_fault()? } else { fault }
        {
            return Err(Error::InvalidBufferLength);
        }
        public.validate_native_batch_owner(graph, arena)?;
        validate_view(
            count,
            0,
            self.layout.width,
            self.layout.width,
            self.layout.width * MAX_BATCH,
        )?;
        let base = format!("blk.{layer}.");
        let mut projections = Vec::new();
        projections
            .try_reserve_exact(3)
            .map_err(|_| Error::LimitExceeded("shared expert projection metadata"))?;
        for (name, i, o) in [
            (
                "ffn_gate_shexp.weight",
                self.layout.width,
                self.layout.hidden,
            ),
            ("ffn_up_shexp.weight", self.layout.width, self.layout.hidden),
            (
                "ffn_down_shexp.weight",
                self.layout.hidden,
                self.layout.width,
            ),
        ] {
            let tensor = graph.tensor(&format!("{base}{name}"))?;
            let matrix = Matrix::new(tensor)?;
            if matrix.experts() != 1 || matrix.inputs() != i || matrix.outputs() != o {
                return Err(Error::InvalidBufferLength);
            }
            projections.push(BoundProjection { tensor, matrix });
        }
        let weight = public.vector(graph, &format!("{base}ffn_gate_inp_shexp.weight"))?;
        self.prove(arena, input, routed, output, fault, &weight)?;
        if native {
            u::zero_many(arena, &[self.native_fault()?])?;
        }
        let spans = BatchSpans::new(count, 0, 0, self.layout.width, 0, self.layout.hidden);
        projector.enqueue_pair(
            [projections[0], projections[1]],
            arena,
            input,
            [self.ids[GATE], self.ids[UP]],
            &spans,
            cancel,
        )?;
        if !native {
            let client = arena.client().map_err(u::fault)?;
            let CubeCount::Static(hidden_grid, _, _) = u::grid(self.layout.hidden)? else {
                return Err(Error::InvalidBufferLength);
            };
            activation::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(hidden_grid, MAX_BATCH as u32, 1),
                CubeDim::new_1d(64),
                u::arg(arena, self.ids[GATE])?,
                u::arg(arena, self.ids[UP])?,
                u::arg(arena, self.ids[ACT])?,
                u::arg(arena, fault)?,
                count,
                self.layout.hidden,
            );
            let spans = BatchSpans::new(count, 0, 0, self.layout.hidden, 0, self.layout.width);
            projector.enqueue_batch(
                projections[2],
                arena,
                self.ids[ACT],
                self.ids[DOWN],
                &spans,
                cancel,
            )?;
            gate_dot::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(1, MAX_BATCH as u32, 1),
                CubeDim::new_1d(64),
                u::arg(arena, input)?,
                u::raw_arg(weight, self.layout.width)?,
                u::arg(arena, self.ids[SCALE])?,
                u::arg(arena, fault)?,
                count,
                self.layout.width,
            );
        }
        u::cancel(cancel)?;
        self.prepared = Some((
            Zeroizing::new([input, routed, output, fault]),
            Zeroizing::new([layer, count]),
        ));
        self.native_prepared = native;
        self.phase = Phase::Prepared;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish_native(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
        combine: impl FnOnce(Handle) -> Result<()>,
    ) -> Result<()> {
        if !self.native_prepared {
            return Err(Error::InvalidFormat("native shared preparation missing"));
        }
        let (ids, coordinates) = consume_prepared(&mut self.phase, &mut self.prepared)?;
        if *ids != [input, routed, output, fault]
            || *coordinates != [layer, count]
            || self.binding != graph.binding_id()
        {
            return Err(Error::InvalidBufferLength);
        }
        u::cancel(cancel)?;
        graph.verify_unchanged()?;
        public.validate_native_batch_owner(graph, arena)?;
        let weight = public.vector(graph, &format!("blk.{layer}.ffn_gate_inp_shexp.weight"))?;
        self.prove(arena, input, routed, output, fault, &weight)?;
        crate::strata_shared_native::fold(
            arena.client().map_err(u::fault)?,
            u::handle(arena, self.native_fault()?)?,
            u::handle(arena, fault)?,
        )?;
        combine(u::handle(arena, self.ids[DOWN])?)?;
        u::cancel(cancel)?;
        self.native_prepared = false;
        self.phase = Phase::Available;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn finish_prepared(
        &mut self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &PrivateArena,
        public: &mut PublicVectors,
        input: BufferId,
        routed: BufferId,
        output: BufferId,
        fault: BufferId,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if self.native_prepared {
            return Err(Error::InvalidFormat("native shared needs joined reduction"));
        }
        let (ids, coordinates) = consume_prepared(&mut self.phase, &mut self.prepared)?;
        if *ids != [input, routed, output, fault]
            || *coordinates != [layer, count]
            || graph.binding_id() != self.binding
        {
            return Err(Error::InvalidBufferLength);
        }
        u::cancel(cancel)?;
        graph.verify_unchanged()?;
        public.validate_native_batch_owner(graph, arena)?;
        let weight = public.vector(graph, &format!("blk.{layer}.ffn_gate_inp_shexp.weight"))?;
        self.prove(arena, input, routed, output, fault, &weight)?;
        let client = arena.client().map_err(u::fault)?;
        let CubeCount::Static(width_grid, _, _) = u::grid(self.layout.width)? else {
            return Err(Error::InvalidBufferLength);
        };
        add::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(width_grid, MAX_BATCH as u32, 1),
            CubeDim::new_1d(64),
            u::arg(arena, routed)?,
            u::arg(arena, self.ids[DOWN])?,
            u::arg(arena, self.ids[SCALE])?,
            u::arg(arena, output)?,
            u::arg(arena, fault)?,
            count,
            self.layout.width,
        );
        u::cancel(cancel)?;
        self.phase = Phase::Available;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_preparation_is_consumed_once_and_missing_stamp_poison_is_sticky() {
        let stamp = || {
            Some((
                Zeroizing::new([BufferId::metadata_fixture(7, 11, 3); 4]),
                Zeroizing::new([47, 1]),
            ))
        };
        for mut phase in [Phase::Available, Phase::Poisoned] {
            let mut prepared = stamp();
            assert!(consume_prepared(&mut phase, &mut prepared).is_err());
        }
        let mut phase = Phase::Prepared;
        let mut prepared = stamp();
        let (ids, coordinates) = consume_prepared(&mut phase, &mut prepared).unwrap();
        assert!(*ids == [BufferId::metadata_fixture(7, 11, 3); 4]);
        assert_eq!(*coordinates, [47, 1]);
        assert!(phase == Phase::Poisoned && prepared.is_none());
        assert!(consume_prepared(&mut phase, &mut prepared).is_err());
        let mut phase = Phase::Prepared;
        assert!(consume_prepared(&mut phase, &mut None).is_err());
        assert!(phase == Phase::Poisoned);
    }
    #[test]
    fn full_pages_cover_all_active_columns_and_reject_growth_before_allocation() {
        let l = Layout::new(2560, 640).unwrap();
        assert_eq!(
            &l.requests().unwrap()[..5],
            &[32768, 32768, 32768, 131072, 32768]
        );
        if N == 6 {
            assert_eq!(l.requests().unwrap().get(5), Some(&32768));
        }
        assert_eq!(l.requests().unwrap().into_iter().sum::<u64>(), MAIN_BYTES);
        for count in [1, 3, 5, 8] {
            for (words, width) in [(l.words[GATE], 640), (l.words[DOWN], 2560)] {
                assert_eq!(
                    validate_view(count, 0, width, width, words).unwrap(),
                    count * width
                );
                assert!(validate_view(count, 0, width, width, count * width - 1).is_err());
            }
        }
        for (width, hidden) in [(0, 640), (2560, 0), (usize::MAX, 640), (2560, usize::MAX)] {
            assert!(Layout::new(width, hidden).is_err());
        }
    }
    #[test]
    fn runtime_count_guard_atomic_faults_and_original_tree_validate_offline() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let a = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for geometry in [(2560, 640), (97, 13)] {
                for variant in 0..3 {
                    let mut b = KernelBuilder::default();
                    b.scope.register_type::<f32>(f);
                    b.scope.register_type::<u32>(u);
                    b.scope.register_type::<usize>(u);
                    let x = b.input_array(Type::new(f)).into();
                    let y = b.input_array(Type::new(f)).into();
                    if variant == 0 {
                        let out = b.output_array(Type::new(f)).into();
                        let fault = b.output_array(Type::new(a)).into();
                        let count = b.scalar(u).into();
                        activation::expand(&mut b.scope, x, y, out, fault, count, geometry.1);
                    } else if variant == 1 {
                        let out = b.output_array(Type::new(f)).into();
                        let fault = b.output_array(Type::new(a)).into();
                        let count = b.scalar(u).into();
                        gate_dot::expand(&mut b.scope, x, y, out, fault, count, geometry.0);
                    } else {
                        let scale = b.input_array(Type::new(f)).into();
                        let out = b.output_array(Type::new(f)).into();
                        let fault = b.output_array(Type::new(a)).into();
                        let count = b.scalar(u).into();
                        add::expand(&mut b.scope, x, y, scale, out, fault, count, geometry.0);
                    }
                    let source = WgslCompiler::default()
                        .compile(
                            b.build(
                                KernelSettings::default()
                                    .cube_dim(CubeDim::new_1d(64))
                                    .kernel_name("native_shared"),
                            ),
                            &WgpuCompilationOptions::default(),
                            mode,
                            u,
                        )
                        .unwrap()
                        .to_string();
                    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                    assert!(source.contains("scalars_u32[0]"), "{source}");
                    assert!(source.contains("workgroup_id.y"), "{source}");
                    assert!(source.contains("atomicOr("), "{source}");
                    if variant == 1 {
                        assert!(source.contains("workgroupBarrier()"), "{source}");
                    }
                    let body = source.split("fn native_shared(").nth(1).unwrap();
                    let read = body
                        .find("buffer_0_global[")
                        .unwrap_or_else(|| panic!("{source}"));
                    let mask = body.find("scalars_u32[0]").unwrap();
                    assert!(
                        read > mask,
                        "inactive storage read before runtime mask: {source}"
                    );
                }
            }
        }
    }
}
