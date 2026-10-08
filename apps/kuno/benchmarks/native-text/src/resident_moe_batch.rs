//! Unconnected cache-backed MAX8 routed family. Original banks remain in the
//! executor cache; all publication requires the future whole-model fence.
#![allow(dead_code)] // Pending full-model coordinator and device acceptance.
use super::*;
use crate::{
    bank_slots::{Plan as BankPlan, StagedSlots},
    private_arena::{BufferId, BufferLayout, PrivateArena},
    quant_matvec::{grouped_moe as k, matmul::BatchSpans},
    resident_gdn::SequenceKey,
    resident_util as u, token_waves,
};
use kuno_text::model::{BoundGraph, GraphBindingId};
use std::sync::atomic::AtomicU64;
use zeroize::Zeroizing;
#[path = "native_count1_selection.rs"]
mod count1;
#[path = "resident_moe_count1.rs"]
mod count1_executor;
#[path = "count1_hybrid_moe.rs"]
mod hybrid_k;
const MAX_LAYERS: usize = 64;
const N: usize = 5;
const MAP: usize = 0;
const INPUT: usize = 1;
const ACT: usize = 2;
const PART: usize = 3;
const PROB: usize = 4;
const OUTPUT: usize = 5;
const STATUS: usize = 6;
const PRIVATE_BYTES: u64 = 1_409_024;
const IO_INCREMENT: u64 = 512 * 1024;
const HOST_PRIVATE: u64 = 2 * 1024 * 1024;
const CPU_WORDS: usize = 10 * 2560;
const OWNER_WORDS: usize = 4 + 80 * 3;
const HYBRID_HOST: usize = 1024 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(1);
fn err(s: &'static str) -> Error {
    Error::InvalidFormat(s)
}
fn next() -> Result<u64> {
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)
}
fn guarded<T>(poison: &AtomicBool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        _ => {
            poison.store(true, Ordering::Release);
            Err(err("native routed chunk failed"))
        }
    }
}
fn expected(layers: usize) -> Result<u64> {
    if layers == 0 || layers > MAX_LAYERS {
        return Err(Error::InvalidBufferLength);
    }
    Ok(if layers == 64 {
        u64::MAX
    } else {
        (1u64 << layers) - 1
    })
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Queuing,
    Accepted,
}
struct Policy {
    expected: u64,
    issued: Zeroizing<u64>,
    committed: Zeroizing<u64>,
    chunk: Zeroizing<u64>,
    start: Zeroizing<u64>,
    count: Zeroizing<usize>,
    phase: Phase,
}
impl Policy {
    fn new(layers: usize) -> Result<Self> {
        Ok(Self {
            expected: expected(layers)?,
            issued: Zeroizing::new(0),
            committed: Zeroizing::new(0),
            chunk: Zeroizing::new(0),
            start: Zeroizing::new(0),
            count: Zeroizing::new(0),
            phase: Phase::Idle,
        })
    }
    fn begin(&mut self, start: u64, count: usize) -> Result<()> {
        if self.phase != Phase::Idle
            || !(1..=8).contains(&count)
            || start.checked_add(count as u64).is_none()
        {
            return Err(err("native routed mixed chunk"));
        }
        *self.chunk = self.chunk.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        *self.start = start;
        *self.count = count;
        self.phase = Phase::Queuing;
        Ok(())
    }
    fn issue(&mut self, layer: usize) -> Result<()> {
        if self.phase != Phase::Queuing
            || layer >= 64
            || layer != (self.expected & !*self.issued).trailing_zeros() as usize
        {
            return Err(err("native routed stage order"));
        }
        *self.issued |= 1u64 << layer;
        Ok(())
    }
    fn preflight(&mut self, seen: u64) -> Result<()> {
        if self.phase != Phase::Queuing || seen != self.expected || *self.issued != self.expected {
            return Err(err("native routed incomplete family"));
        }
        self.phase = Phase::Accepted;
        Ok(())
    }
    fn commit(&mut self, layer: usize) -> Result<()> {
        if layer >= 64 {
            return Err(Error::OutOfBounds);
        }
        let bit = 1u64.checked_shl(layer as u32).ok_or(Error::OutOfBounds)?;
        if self.phase != Phase::Accepted || self.expected & bit == 0 || *self.committed & bit != 0 {
            return Err(err("native routed invalid commit"));
        }
        *self.committed |= bit;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if self.phase != Phase::Accepted || *self.committed != self.expected {
            return Err(err("native routed partial publication"));
        }
        self.issued.zeroize();
        self.committed.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.phase = Phase::Idle;
        Ok(())
    }
    fn completed_boundary(&self) -> Result<()> {
        if self.phase != Phase::Idle
            || *self.chunk == 0
            || *self.issued != 0
            || *self.committed != 0
            || *self.start != 0
            || *self.count != 0
        {
            return Err(err("native routed projection policy boundary"));
        }
        Ok(())
    }
}
#[derive(Clone, PartialEq, Eq)]
struct Stamp {
    workspace: u64,
    chunk: u64,
    start: u64,
    count: usize,
    sequence: SequenceKey,
    fault: BufferId,
    ids: [BufferId; N],
}
impl Zeroize for Stamp {
    fn zeroize(&mut self) {
        self.workspace.zeroize();
        self.chunk.zeroize();
        self.start.zeroize();
        self.count.zeroize();
        self.sequence.zeroize();
        self.fault.zeroize();
        self.ids.zeroize();
    }
}
pub(super) struct Registry {
    binding: GraphBindingId,
    stamp: Zeroizing<Stamp>,
    poison: Arc<AtomicBool>,
}
impl Registry {
    pub(super) fn abort(&self) {
        self.poison.store(true, Ordering::Release)
    }
}
pub(super) fn reject_scalar(e: &GpuExecutor<'_>) -> Result<()> {
    if let Some(r) = &e.batch_moe {
        r.abort();
        e.poison_resident();
        return Err(err("scalar bank access during native chunk"));
    }
    Ok(())
}
pub(crate) struct Session {
    stamp: Zeroizing<Stamp>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Session {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release)
        }
        self.armed.zeroize();
    }
}
struct Layout {
    geometries: Vec<k::Geometry>,
    words: [usize; N],
    capacity: usize,
    tile: usize,
    gate_policy: k::GatePolicy,
}
fn install_q4_geometries(geometries: &mut [k::Geometry]) -> Result<()> {
    if geometries.len() != 48 {
        return Err(err("native routed Q4 layer inventory"));
    }
    // Public metadata only, fixed stack storage. An incompatible late layer
    // cannot leave an earlier layer switched to the candidate policy.
    let mut replacements = [geometries[0]; 48];
    for (next, g) in replacements.iter_mut().zip(geometries.iter()) {
        *next = k::Geometry::new(g.projections(), 1, g.capacity())
            .and_then(k::Geometry::with_q4_block)
            .map_err(err)?;
    }
    geometries.copy_from_slice(&replacements);
    Ok(())
}
impl Layout {
    fn new(graph: BoundGraph<'_, '_>, capacity: usize, tile: usize) -> Result<Self> {
        // A model owner cannot start with the decode-only specialization.
        if !matches!(tile, 4 | 8) {
            return Err(err("native routed initial public tile"));
        }
        graph.verify_unchanged()?;
        let g = graph.geometry();
        if g.layers != 48
            || g.experts != 512
            || g.experts_used != 10
            || g.residual.width != 2560
            || g.expert_hidden != 640
        {
            return Err(err("native routed original geometry"));
        }
        let mut geometries = Vec::new();
        geometries
            .try_reserve_exact(g.layers)
            .map_err(|_| Error::LimitExceeded("native routed metadata"))?;
        for layer in 0..g.layers {
            let w = weights(graph, layer)?;
            let p = [w.gate, w.up, w.down]
                .map(|t| {
                    let d = t.info().dimensions();
                    if d.len() != 3 {
                        return Err(Error::InvalidBufferLength);
                    }
                    let p = k::Projection::new(
                        t.info().dtype().gguf_id(),
                        d[0] as usize,
                        d[1] as usize,
                        d[2] as usize,
                    )
                    .map_err(err)?;
                    if t.info().row_bytes() != p.row_bytes() as u64 {
                        return Err(Error::InvalidBufferLength);
                    }
                    Ok(p)
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let p: [k::Projection; 3] = p.try_into().map_err(|_| Error::InvalidBufferLength)?;
            let geometry = k::Geometry::new(p, tile, capacity).map_err(err)?;
            if geometry.hidden() != g.residual.width
                || geometry.ff() != g.expert_hidden
                || geometry.status_words() != 4800
            {
                return Err(err("native routed layer geometry mismatch"));
            }
            geometries.push(geometry);
        }
        let words = [
            80 * 640,
            80 * 2560,
            k::MAP_WORDS,
            80,
            geometries[0].status_words(),
        ];
        let l = Self {
            geometries,
            words,
            capacity,
            tile,
            gate_policy: k::GatePolicy::Generic,
        };
        if l.requests()?.into_iter().sum::<u64>() != PRIVATE_BYTES {
            return Err(err("native routed exact scratch"));
        }
        Ok(l)
    }
    fn requests(&self) -> Result<[u64; N]> {
        let mut r = [0; N];
        for (i, w) in self.words.into_iter().enumerate() {
            r[i] = BufferLayout::new(w).map_err(u::fault)?.owned_bytes as u64;
        }
        Ok(r)
    }
    fn limits(&self, l: &wgpu::Limits) -> Result<()> {
        if l.max_compute_invocations_per_workgroup < 64
            || l.max_compute_workgroup_size_x < 64
            || l.max_compute_workgroups_per_dimension < 4096
            || l.max_compute_workgroup_size_y < 1
            || l.max_compute_workgroup_size_z < 1
            || l.max_bind_groups < 1
            || l.max_uniform_buffer_binding_size < 512
            || l.max_storage_buffers_per_shader_stage < 7
            || l.max_bindings_per_bind_group < 7
            || l.max_compute_workgroup_storage_size < (2 * 64 * self.tile * 4) as u32
            || l.max_storage_buffer_binding_size < 128 * 1024 * 1024
            || l.max_buffer_size < 128 * 1024 * 1024
            || l.min_storage_buffer_offset_alignment == 0
            || l.min_uniform_buffer_offset_alignment == 0
            || l.min_storage_buffer_offset_alignment > 512
            || l.min_uniform_buffer_offset_alignment > 512
        {
            return Err(Error::LimitExceeded("native routed device limits"));
        }
        Ok(())
    }
}
/// No telemetry/admission/queue callback: also used inside already-admitted
/// initialization. Unsupported source policies reject before scratch mutation.
fn configuration(e: &GpuExecutor<'_>, graph: BoundGraph<'_, '_>) -> Result<()> {
    if e.batch_experts || e.ordered_public_upload || e.word_down || e.batch_moe.is_some() {
        return Err(err("native routed unsupported source policy"));
    }
    e.check_segmented_resident(true)?;
    graph.verify_unchanged()?;
    e.model.verify_file_unchanged()?;
    for layer in 0..graph.geometry().layers {
        let w = weights(graph, layer)?;
        if [w.gate, w.up, w.down]
            .into_iter()
            .any(|t| !t.belongs_to(e.model))
        {
            return Err(err("native routed configuration artifact binding"));
        }
    }
    Ok(())
}

fn weights<'f>(graph: BoundGraph<'_, 'f>, layer: usize) -> Result<RoutedWeights<'f>> {
    let b = format!("blk.{layer}.");
    Ok(RoutedWeights {
        gate: graph.tensor(&format!("{b}ffn_gate_exps.weight"))?,
        up: graph.tensor(&format!("{b}ffn_up_exps.weight"))?,
        down: graph.tensor(&format!("{b}ffn_down_exps.weight"))?,
    })
}
pub(crate) struct Workspace {
    binding: GraphBindingId,
    sequence: SequenceKey,
    identity: u64,
    layout: Layout,
    ids: Zeroizing<[BufferId; N]>,
    admission: ResidentAdmission,
    poison: Arc<AtomicBool>,
    policy: Policy,
    hybrid: Option<HybridOwners>,
}
struct HybridOwners {
    ids: Zeroizing<[BufferId; 2]>,
    operator: Arc<kuno_text::cpu_expert::ExpertOperator>,
    cap: Option<usize>,
    overlap: bool,
}
fn hybrid_worker_geometry(count: usize, pooled: bool) -> bool {
    matches!(count, 4 | 8 | 16) && (count == 16 || pooled)
}
impl Workspace {
    pub(crate) fn count1_hybrid_allocation_requests() -> Result<[u64; 2]> {
        Ok([CPU_WORDS, OWNER_WORDS].map(|n| {
            BufferLayout::new(n)
                .expect("fixed count1 owning layout")
                .owned_bytes as u64
        }))
    }
    pub(crate) fn reserve_count1_hybrid(
        &mut self,
        arena: &mut PrivateArena,
        operator: Arc<kuno_text::cpu_expert::ExpertOperator>,
        overlap: bool,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        // The complete model allocation callback holds MAIN admission. Use
        // its exact borrowed origin proof; healthy() would recursively lock.
        allocation.check(&self.admission)?;
        self.validate_ids(arena)?;
        if self.poison.load(Ordering::Acquire) {
            return Err(err("native routed sequence poisoned"));
        }
        if self.hybrid.is_some()
            || self.policy.phase != Phase::Idle
            || *self.policy.chunk != 0
            || (overlap && !operator.workers_pooled())
            || !hybrid_worker_geometry(operator.worker_count(), operator.workers_pooled())
            || operator.backend()
                != if cfg!(feature = "cpu-fma") {
                    kuno_text::cpu_expert::Backend::Avx2Fma
                } else {
                    kuno_text::cpu_expert::Backend::Avx2
                }
        {
            return Err(err("native count1 explicit constructor reservation"));
        }
        let cpu = arena.allocate_zeroed(CPU_WORDS).map_err(u::fault)?;
        let owner = arena.allocate_zeroed(OWNER_WORDS).map_err(u::fault)?;
        self.hybrid = Some(HybridOwners {
            ids: Zeroizing::new([cpu, owner]),
            operator,
            cap: None,
            overlap,
        });
        self.validate_ids(arena)
    }
    pub(crate) fn cpu_gpu_overlap(&self) -> Option<bool> {
        self.hybrid.as_ref().map(|h| h.overlap)
    }
    pub(crate) fn cpu_workers_pooled(&self) -> Option<bool> {
        self.hybrid.as_ref().map(|h| h.operator.workers_pooled())
    }
    pub(crate) fn cpu_arithmetic_geometry(&self) -> Option<(usize, usize)> {
        self.hybrid
            .as_ref()
            .map(|h| (h.operator.worker_count(), h.operator.q8_residual_passes()))
    }
    pub(crate) fn cpu_mapped_prefetch(&self) -> Option<(bool, u64)> {
        self.hybrid
            .as_ref()
            .map(|h| (h.operator.mapped_prefetch(), h.operator.prefetch_calls()))
    }
    pub(crate) fn cpu_q8_residual(&self) -> Option<bool> {
        self.hybrid.as_ref().map(|h| h.operator.q8_residual())
    }
    pub(crate) fn cpu_q8_activations(&self) -> Option<bool> {
        self.hybrid.as_ref().map(|h| h.operator.q8_activations())
    }
    pub(crate) fn cpu_miss_cap(&self) -> Option<usize> {
        self.hybrid.as_ref().and_then(|h| h.cap)
    }
    pub(crate) fn count1_hybrid_owner_ids(&self) -> Option<[BufferId; 2]> {
        self.hybrid.as_ref().map(|h| *h.ids)
    }
    pub(crate) fn set_lab_count1_cpu_misses(
        &mut self,
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        cap: usize,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<usize> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        executor.check_resident_admission(&self.admission)?;
        executor.check_client(arena.client().map_err(u::fault)?)?;
        graph.verify_unchanged()?;
        if cap > 10
            || ![1, 8].contains(&self.layout.tile)
            || !arena.public_owned_zero_audit_enabled()
            || executor.batch_moe.is_some()
            || executor.banks.len() != 48
            || graph.binding_id() != self.binding
        {
            return Err(err("native count1 completed original owner policy"));
        }
        executor
            .check_original_bank_metadata(48, crate::resident_model::batch::ORIGINAL_BANK_HOST)?;
        for layer in 0..48 {
            let weights = weights(graph, layer)?;
            let bank = executor
                .banks
                .get(weights.gate.info().name())
                .ok_or(err("native count1 requires all original banks"))?;
            bank.check_names(weights)?;
            bank.check_client(arena.client().map_err(u::fault)?)?;
            if bank.slots.capacity() != self.layout.capacity {
                return Err(err("native count1 original bank capacity"));
            }
            let shards = [bank.gate(), bank.up(), bank.down()];
            for shard in 0..shards[0].segments().len() {
                let p = shards
                    .map(|b| b.segments().get(shard))
                    .map(|p| p.ok_or(err("native count1 original shards")))
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
                self.layout.geometries[layer]
                    .validate_shard(
                        p[0].first_slot(),
                        p[0].capacity(),
                        [p[0].words(), p[1].words(), p[2].words()],
                    )
                    .map_err(err)?;
            }
        }
        let h = self
            .hybrid
            .as_mut()
            .ok_or(err("native count1 reservation absent"))?;
        if h.cap.is_some_and(|old| old != cap) {
            return Err(err("native count1 policy needs fresh owner"));
        }
        h.cap = Some(cap);
        Ok(cap)
    }
    pub(crate) fn allocation_requests(
        graph: BoundGraph<'_, '_>,
        capacity: usize,
        tile: usize,
    ) -> Result<[u64; N]> {
        Layout::new(graph, capacity, tile)?.requests()
    }
    pub(crate) fn runtime_io_increment() -> u64 {
        IO_INCREMENT
    }
    /// Public configured policy for the original token-wave planner.
    pub(crate) fn capacity(&self) -> usize {
        self.layout.capacity
    }
    pub(crate) fn projection_tile(&self) -> usize {
        self.layout.tile
    }
    pub(crate) fn gate_policy(&self) -> k::GatePolicy {
        self.layout.gate_policy
    }
    /// The model alone constructs this proof after its completed all-family
    /// cursor boundary. No buffers, ledgers, launches or audits are changed.
    pub(crate) fn set_lab_single_member_decode(
        &mut self,
        executor: &GpuExecutor<'_>,
        arena: &PrivateArena,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<usize> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        executor.check_resident_admission(&self.admission)?;
        executor.check_client(arena.client().map_err(u::fault)?)?;
        if executor.batch_moe.is_some() || !arena.public_owned_zero_audit_enabled() {
            return Err(err("native routed single-member owner policy"));
        }
        // Validate every replacement before mutating any public geometry. The
        // second pass uses only these same immutable checked values, no alloc.
        for g in &self.layout.geometries {
            k::Geometry::new(g.projections(), 1, g.capacity()).map_err(err)?;
        }
        for g in &mut self.layout.geometries {
            *g = k::Geometry::new(g.projections(), 1, g.capacity()).map_err(err)?;
        }
        self.layout.tile = 1;
        self.layout.gate_policy = k::GatePolicy::Generic;
        Ok(self.projection_tile())
    }
    /// The same completed, fully audited model owner; every layer's public
    /// replacement is proved before any policy is changed. No new allocation.
    pub(crate) fn set_lab_single_q4_decode(
        &mut self,
        executor: &GpuExecutor<'_>,
        arena: &PrivateArena,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<usize> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        executor.check_resident_admission(&self.admission)?;
        executor.check_client(arena.client().map_err(u::fault)?)?;
        if executor.batch_moe.is_some() || !arena.public_owned_zero_audit_enabled() {
            return Err(err("native routed Q4 single-member owner policy"));
        }
        install_q4_geometries(&mut self.layout.geometries)?;
        self.layout.tile = 1;
        self.layout.gate_policy = k::GatePolicy::Q4Block;
        Ok(self.projection_tile())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        admission: ResidentAdmission,
        poison: Arc<AtomicBool>,
        limits: &wgpu::Limits,
        tile: usize,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        guarded(&poison.clone(), || {
            u::cancel(cancel)?;
            if poison.load(Ordering::Acquire) {
                return Err(err("native routed sequence poisoned"));
            }
            if !Arc::ptr_eq(&poison, &admission.sequence_poison()) || !admission.transient_io() {
                return Err(err("native routed admission poison owner"));
            }
            configuration(executor, graph)?;
            let capacity = executor.bank_slots;
            if executor
                .resident_admission
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(&actual.ledger, &admission.ledger))
            {
                return Err(err("native routed actual admission binding"));
            }
            let layout = Layout::new(graph, capacity, tile)?;
            layout.limits(limits)?;
            let metadata = StagedSlots::bounded_native_metadata_bytes(capacity)
                .map_err(|_| Error::LimitExceeded("native routed candidate metadata"))?;
            if metadata
                .checked_add(
                    std::mem::size_of::<Pending>()
                        + 6 * std::mem::size_of::<Handle>()
                        + std::mem::size_of::<Last>(),
                )
                .is_none_or(|n| n > 32 * 1024)
                || 48 * 32 * 1024 + 512 * 1024 > HOST_PRIVATE as usize
            {
                return Err(Error::LimitExceeded("native routed complete metadata plan"));
            }

            if arena
                .charged_bytes()
                .checked_add(PRIVATE_BYTES as usize)
                .is_none_or(|n| n > arena.budget_bytes())
            {
                return Err(Error::LimitExceeded("native routed complete private plan"));
            }
            if !std::ptr::eq(
                arena.client().map_err(u::fault)?.info(),
                admission.client.info(),
            ) {
                return Err(err("native routed admission client"));
            }
            let first = arena.allocate_zeroed(layout.words[0]).map_err(u::fault)?;
            let mut ids = Zeroizing::new([first; N]);
            for i in 1..N {
                u::cancel(cancel)?;
                ids[i] = arena.allocate_zeroed(layout.words[i]).map_err(u::fault)?;
            }
            let out = Self {
                binding: graph.binding_id(),
                sequence,
                identity: next()?,
                layout,
                ids,
                admission,
                poison: poison.clone(),
                policy: Policy::new(graph.geometry().layers)?,
                hybrid: None,
            };
            out.validate_ids(arena)?;
            u::cancel(cancel)?;
            Ok(out)
        })
    }
    fn healthy(&self, arena: &PrivateArena) -> Result<()> {
        if self.poison.load(Ordering::Acquire) {
            return Err(err("native routed sequence poisoned"));
        }
        self.admission.active()?;
        self.validate_ids(arena)
    }
    fn validate_ids(&self, arena: &PrivateArena) -> Result<()> {
        if !std::ptr::eq(
            arena.client().map_err(u::fault)?.info(),
            self.admission.client.info(),
        ) {
            return Err(err("native routed foreign arena"));
        }
        for (i, &id) in self.ids.iter().enumerate() {
            if self.ids[..i].contains(&id)
                || arena.layout(id).map_err(u::fault)?
                    != BufferLayout::new(self.layout.words[i]).map_err(u::fault)?
            {
                return Err(err("native routed stale workspace"));
            }
        }
        if let Some(h) = &self.hybrid {
            for (i, &id) in h.ids.iter().enumerate() {
                if self.ids.contains(&id)
                    || h.ids[..i].contains(&id)
                    || arena.layout(id).map_err(u::fault)?
                        != BufferLayout::new([CPU_WORDS, OWNER_WORDS][i]).map_err(u::fault)?
                {
                    return Err(err("native count1 stale owning pages"));
                }
            }
        }
        Ok(())
    }
    fn check(&self, e: &GpuExecutor<'_>, s: &Session, arena: &PrivateArena) -> Result<()> {
        self.healthy(arena)?;
        e.check_resident_admission(&self.admission)?;
        let r = e
            .batch_moe
            .as_ref()
            .ok_or(err("native routed registry missing"))?;
        if !s.armed
            || r.binding != self.binding
            || *r.stamp != *s.stamp
            || s.stamp.workspace != self.identity
            || s.stamp.sequence != self.sequence
            || s.stamp.ids != *self.ids
            || s.stamp.chunk != *self.policy.chunk
            || s.stamp.start != *self.policy.start
            || s.stamp.count != *self.policy.count
            || !Arc::ptr_eq(&s.poison, &self.poison)
            || !Arc::ptr_eq(&r.poison, &self.poison)
            || u::words(arena, s.stamp.fault)? != 1
        {
            return Err(err("native routed mixed session"));
        }
        Ok(())
    }
    fn ids(&self, input: BufferId, output: BufferId) -> Zeroizing<[BufferId; 7]> {
        Zeroizing::new([
            self.ids[2],
            input,
            self.ids[0],
            self.ids[1],
            self.ids[3],
            output,
            self.ids[4],
        ])
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
        self.sequence.zeroize();
        self.identity.zeroize();
    }
}
struct Views {
    ids: Zeroizing<[BufferId; 2]>,
    offsets: Zeroizing<[usize; 4]>,
}
struct Last {
    pending: BankPlan,
    selected: Zeroizing<[usize; 80]>,
    len: Zeroizing<usize>,
}
pub(crate) struct Pending {
    views: Views,
    stamp: Zeroizing<Stamp>,
    layer: Zeroizing<usize>,
    key: String,
    instance: crate::expert_banks::BankInstance,
    pins: Vec<Handle>,
    staged: Option<StagedSlots>,
    last: Option<Last>,
    count1: Option<count1::Proof>,
    waves: Zeroizing<usize>,
    poison: Arc<AtomicBool>,
    armed: bool,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if self.armed {
            self.poison.store(true, Ordering::Release)
        }
        self.armed.zeroize();
    }
}
struct Pin<'a> {
    wave: token_waves::WaveView<'a>,
    pending: &'a BankPlan,
    staged: &'a StagedSlots,
    bank: &'a ExpertBanks,
    selected: Zeroizing<[usize; 80]>,
    pairs: Zeroizing<[(u32, u32); 80]>,
    len: Zeroizing<usize>,
}
impl<'a> Pin<'a> {
    fn new(
        wave: token_waves::WaveView<'a>,
        pending: &'a BankPlan,
        staged: &'a StagedSlots,
        bank: &'a ExpertBanks,
    ) -> Result<Self> {
        let len = wave.protected_experts().len();
        if len == 0 || len > 80 || len > bank.slots.capacity() || pending.slots.len() != len {
            return Err(Error::InvalidBufferLength);
        }
        let mut p = Self {
            wave,
            pending,
            staged,
            bank,
            selected: Zeroizing::new([0; 80]),
            pairs: Zeroizing::new([(0, 0); 80]),
            len: Zeroizing::new(len),
        };
        for (i, &e) in p.wave.protected_experts().iter().enumerate() {
            p.selected[i] = e as usize;
            p.pairs[i] = (e, pending.slots[i] as u32);
        }
        p.validate()?;
        Ok(p)
    }
    fn validate(&self) -> Result<()> {
        self.staged
            .validate_pending(&self.bank.slots, self.pending, &self.selected[..*self.len])
            .map_err(|_| err("native routed canonical pin"))?;
        for (i, &e) in self.wave.protected_experts().iter().enumerate() {
            if self.selected[i] != e as usize || self.pairs[i] != (e, self.pending.slots[i] as u32)
            {
                return Err(err("native routed changed pin"));
            }
        }
        Ok(())
    }
}
impl GpuExecutor<'_> {
    pub(crate) fn routed_moe_batch_allocation_requests(
        &self,
        graph: BoundGraph<'_, '_>,
        tile: usize,
    ) -> Result<[u64; N]> {
        configuration(self, graph)?;
        Workspace::allocation_requests(graph, self.bank_slots, tile)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_routed_moe_chunk(
        &mut self,
        w: &mut Workspace,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        start: u64,
        count: usize,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<Session> {
        let attempted = guarded(&w.poison.clone(), || {
            u::cancel(cancel)?;
            graph.verify_unchanged()?;
            self.model.verify_file_unchanged()?;
            w.healthy(arena)?;
            w.layout.geometries[0].validate_count(count).map_err(err)?;
            if w.cpu_miss_cap().is_some() && count != 1 {
                return Err(err("native count1 public policy count"));
            }
            self.check_resident_admission(&w.admission)?;
            self.check_client(arena.client().map_err(u::fault)?)?;
            if self.batch_moe.is_some()
                || graph.binding_id() != w.binding
                || self.bank_slots != w.layout.capacity
                || self.batch_experts
                || self.ordered_public_upload
                || self.word_down
                || u::words(arena, fault)? != 1
                || start
                    .checked_add(count as u64)
                    .is_none_or(|n| n > graph.geometry().max_context as u64)
            {
                return Err(err("native routed session configuration"));
            }
            self.check_segmented_resident(true)?;
            w.policy.begin(start, count)?;
            let stamp = Zeroizing::new(Stamp {
                workspace: w.identity,
                chunk: *w.policy.chunk,
                start,
                count,
                sequence: w.sequence,
                fault,
                ids: *w.ids,
            });
            self.batch_moe = Some(Registry {
                binding: w.binding,
                stamp: Zeroizing::new((*stamp).clone()),
                poison: w.poison.clone(),
            });
            Ok(Session {
                stamp,
                poison: w.poison.clone(),
                armed: true,
            })
        });
        if attempted.is_err() {
            self.poison_resident()
        }
        attempted
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_routed_moe_batch(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        spans: &BatchSpans,
        routes: &token_waves::Plan,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        let result = guarded(&w.poison.clone(), || {
            self.batch_checked(
                graph, arena, w, s, layer, input, output, spans, routes, cancel,
            )
        });
        if result.is_err() {
            self.poison_resident()
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn batch_checked(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        w: &mut Workspace,
        s: &Session,
        layer: usize,
        input: BufferId,
        output: BufferId,
        views: &BatchSpans,
        routes: &token_waves::Plan,
        cancel: &AtomicBool,
    ) -> Result<Pending> {
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        graph.verify_unchanged()?;
        self.model.verify_file_unchanged()?;
        if graph.binding_id() != w.binding
            || routes.count() != s.stamp.count
            || views.count() != routes.count()
            || views.packed_first() != 0
            || views.input_stride() < 2560
            || views.output_stride() < 2560
            || self.batch_experts
            || self.ordered_public_upload
            || self.word_down
            || w.cpu_miss_cap().is_some()
        {
            return Err(err("native routed chunk binding"));
        }
        let g = *w.layout.geometries.get(layer).ok_or(Error::OutOfBounds)?;
        g.validate_count(routes.count()).map_err(err)?;
        let weights = weights(graph, layer)?;
        if [weights.gate, weights.up, weights.down]
            .into_iter()
            .any(|t| !t.belongs_to(self.model))
        {
            return Err(err("native routed artifact binding"));
        }
        w.policy.issue(layer)?;
        let key = weights.gate.info().name().to_owned();
        if !self.banks.contains_key(&key) {
            let requests = ExpertBanks::allocation_requests(weights, self.bank_slots)?;
            let bank = w.admission.allocate(&requests, || {
                ExpertBanks::new(&self.client, weights, self.bank_slots)
            })?;
            self.bank_reserved = self
                .bank_reserved
                .checked_add(bank.reserved())
                .ok_or(Error::ArithmeticOverflow)?;
            self.banks.insert(key.clone(), bank);
        }
        let bank = self
            .banks
            .get(&key)
            .ok_or(err("native routed actual cache bank"))?;
        bank.check_names(weights)?;
        bank.check_client(arena.client().map_err(u::fault)?)?;
        let ids = w.ids(input, output);
        let spans = k::Spans::new([
            0,
            views.input_first(),
            views.input_stride(),
            0,
            g.ff(),
            0,
            g.hidden(),
            0,
            views.output_first(),
            views.output_stride(),
            0,
        ]);
        let words = (*ids)
            .map(|id| u::words(arena, id))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let words: [usize; 7] = words.try_into().map_err(|_| Error::InvalidBufferLength)?;
        spans.validate(g, words).map_err(err)?;
        physical(arena, bank, g, &ids, s.stamp.fault, w)?;
        for &id in w.ids.iter() {
            u::zero(arena, id)?;
        }
        let mut staged =
            StagedSlots::begin(&bank.slots).map_err(|_| err("native routed staged origin"))?;
        let mut experts = Zeroizing::new([0u32; 80]);
        let mut probabilities = Zeroizing::new([0f32; 80]);
        for t in 0..routes.count() {
            let row = routes.token(t).ok_or(Error::InvalidBufferLength)?;
            if row.len() != 10 {
                return Err(Error::InvalidBufferLength);
            }
            for (r, v) in row.iter().enumerate() {
                experts[t * 10 + r] = v.expert();
                probabilities[t * 10 + r] = v.probability();
            }
        }
        let mut done = 0;
        let mut final_last = None;
        let mut stats = expert_banks::FillStats::default();
        for index in 0..routes.wave_count() {
            u::cancel(cancel)?;
            let wave = routes.wave(index).ok_or(Error::InvalidBufferLength)?;
            if wave.tokens().start != done {
                return Err(err("native routed contiguous waves"));
            }
            let mut selected = Zeroizing::new([0usize; 80]);
            let union = wave.protected_experts();
            if union.is_empty() || union.len() > 80 {
                return Err(Error::InvalidBufferLength);
            }
            for (i, &e) in union.iter().enumerate() {
                selected[i] = e as usize;
            }
            let pending = staged
                .plan(&bank.slots, &selected[..union.len()])
                .map_err(|_| err("native routed real pending plan"))?;
            let pin = Pin::new(wave, &pending, &staged, bank)?;
            for (wi, window) in pending.uploads.chunks(10).enumerate() {
                pin.validate()?;
                u::cancel(cancel)?;
                let requests = fill_requests(
                    [bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64),
                    window.len(),
                    false,
                    self.expert_batch_bytes,
                )?;
                let charged = requests.iter().try_fold(0u64, |n, &b| {
                    n.checked_add(main_admission::page_bytes(b).map_err(admission_error)?)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
                if charged > 40 * 1024 * 1024 {
                    return Err(Error::LimitExceeded("native routed source window"));
                }
                let add = w.admission.allocate(&requests, || {
                    let mut add = expert_banks::FillStats::default();
                    for &(e, slot) in window {
                        pin.validate()?;
                        if w.poison.load(Ordering::Acquire) {
                            return Err(err("native routed aborted source fill"));
                        }
                        for (b, t) in [
                            (bank.gate(), weights.gate),
                            (bank.up(), weights.up),
                            (bank.down(), weights.down),
                        ] {
                            pin.validate()?;
                            let (bytes, read, upload) =
                                expert_banks::fill(&self.client, b, t, e, slot, cancel)?;
                            add.bytes = add
                                .bytes
                                .checked_add(bytes)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.read_ns = add
                                .read_ns
                                .checked_add(read)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.upload_ns = add
                                .upload_ns
                                .checked_add(upload)
                                .ok_or(Error::ArithmeticOverflow)?;
                        }
                        add.experts += 1;
                    }
                    Ok(add)
                })?;
                stats.bytes = stats
                    .bytes
                    .checked_add(add.bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.read_ns = stats
                    .read_ns
                    .checked_add(add.read_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.upload_ns = stats
                    .upload_ns
                    .checked_add(add.upload_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                stats.experts = stats
                    .experts
                    .checked_add(add.experts)
                    .ok_or(Error::ArithmeticOverflow)?;
                if (wi + 1) * 10 < pending.uploads.len() {
                    dependency(&w.admission, arena, s.stamp.fault, cancel)?;
                    pin.validate()?;
                }
            }
            let map = k::GroupMap::new(
                routes.count(),
                pin.wave.tokens().start,
                pin.wave.tokens().len(),
                &experts[..routes.count() * 10],
                &probabilities[..routes.count() * 10],
                &pin.pairs[..*pin.len],
                bank.slots.capacity(),
            )
            .map_err(err)?;
            map.validate(
                &experts[..routes.count() * 10],
                &probabilities[..routes.count() * 10],
                bank.slots.capacity(),
            )
            .map_err(err)?;
            map.validate_geometry(g).map_err(err)?;
            write_words(arena, ids[MAP], map.words(), &w.poison)?;
            let mut probs = Zeroizing::new([0u32; 80]);
            for (i, &p) in map.probabilities().iter().enumerate() {
                probs[i] = p.to_bits()
            }
            write_words(arena, ids[PROB], &*probs, &w.poison)?;
            physical(arena, bank, g, &ids, s.stamp.fault, w)?;
            launch_wave(arena, bank, g, &spans, &ids, map.groups())?;
            launch_finite(arena, g, &spans, &ids)?;
            fold(
                arena,
                g,
                &ids,
                s.stamp.fault,
                routes.count(),
                pin.wave.tokens().start,
                pin.wave.tokens().len(),
                false,
            )?;
            done = pin.wave.tokens().end;
            pin.validate()?;
            drop(map);
            drop(pin);
            if index + 1 < routes.wave_count() {
                dependency(&w.admission, arena, s.stamp.fault, cancel)?;
                staged
                    .advance_after_completed_consumers(
                        &bank.slots,
                        pending,
                        &selected[..union.len()],
                    )
                    .map_err(|_| err("native routed completed shadow wave"))?;
            } else {
                final_last = Some(Last {
                    pending,
                    selected,
                    len: Zeroizing::new(union.len()),
                });
            }
        }
        if done != routes.count()
            || routes.wave_count() == 0
            || routes.wave_count() > 8
            || staged.accepted_waves() + 1 != routes.wave_count()
        {
            return Err(err("native routed incomplete coverage"));
        }
        launch_reduce(arena, g, &spans, &ids)?;
        launch_finite(arena, g, &spans, &ids)?;
        fold(
            arena,
            g,
            &ids,
            s.stamp.fault,
            routes.count(),
            0,
            routes.count(),
            true,
        )?;
        u::cancel(cancel)?;
        w.check(self, s, arena)?;
        staged
            .validate_origin(&bank.slots)
            .map_err(|_| err("native routed source revision changed"))?;
        let pins: Vec<Handle> = [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .flat_map(|b| b.segments().iter().map(|p| p.handle().clone()))
            .collect();
        let last = final_last
            .as_ref()
            .ok_or(err("native routed missing final pin"))?;
        let metadata = staged
            .retained_native_metadata_bytes(&last.pending)
            .map_err(|_| Error::LimitExceeded("native routed retained metadata"))?;
        if metadata
            .checked_add(
                std::mem::size_of::<Pending>()
                    + std::mem::size_of::<Last>()
                    + pins.capacity() * std::mem::size_of::<Handle>(),
            )
            .is_none_or(|n| n > 32 * 1024)
        {
            return Err(Error::LimitExceeded("native routed pending host envelope"));
        }
        let out = Pending {
            views: Views {
                ids: Zeroizing::new([input, output]),
                offsets: Zeroizing::new([
                    views.input_first(),
                    views.input_stride(),
                    views.output_first(),
                    views.output_stride(),
                ]),
            },
            stamp: Zeroizing::new((*s.stamp).clone()),
            layer: Zeroizing::new(layer),
            key,
            instance: bank.instance(),
            pins,
            staged: Some(staged),
            last: final_last,
            count1: None,
            waves: Zeroizing::new(routes.wave_count()),
            poison: w.poison.clone(),
            armed: true,
        };
        self.expert_hits = self
            .expert_hits
            .checked_add((routes.count() * 10) as u64 - stats.experts)
            .ok_or(Error::ArithmeticOverflow)?;
        self.expert_uploads = self
            .expert_uploads
            .checked_add(stats.experts)
            .ok_or(Error::ArithmeticOverflow)?;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add(stats.bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_read_ns = self
            .weight_read_ns
            .checked_add(stats.read_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_upload_ns = self
            .weight_upload_ns
            .checked_add(stats.upload_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        self.moe_calls = self
            .moe_calls
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(out)
    }
}
fn dependency(
    a: &ResidentAdmission,
    arena: &PrivateArena,
    fault: BufferId,
    cancel: &AtomicBool,
) -> Result<()> {
    u::cancel(cancel)?;
    a.readback(|| u::read_fault_fence(arena, fault))?;
    u::cancel(cancel)
}
fn write_words(
    arena: &PrivateArena,
    id: BufferId,
    words: &[u32],
    poison: &AtomicBool,
) -> Result<()> {
    let l = arena.layout(id).map_err(u::fault)?;
    if l.logical_words != words.len() {
        return Err(Error::InvalidBufferLength);
    }
    let mut b = Zeroizing::new(Vec::new());
    b.try_reserve_exact(l.owned_bytes)
        .map_err(|_| Error::LimitExceeded("native routed guarded upload"))?;
    for &v in words {
        b.extend_from_slice(&v.to_le_bytes())
    }
    b.resize(l.owned_bytes, 0);
    arena
        .client()
        .map_err(u::fault)?
        .write_private(
            arena.handle(id).map_err(u::fault)?.clone(),
            std::mem::take(&mut *b),
        )
        .map_err(|_| {
            poison.store(true, Ordering::Release);
            err("native routed owning map upload")
        })
}

fn arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    let l = arena.layout(id).map_err(u::fault)?;
    Ok(unsafe {
        ArrayArg::from_raw_parts(arena.handle(id).map_err(u::fault)?.clone(), l.owned_words)
    })
}

/// Whole original owning pages: private full spans equal physical size; public
/// packed segments may have only their already-precharged exclusive-bin tail.
fn physical(
    arena: &PrivateArena,
    bank: &ExpertBanks,
    g: k::Geometry,
    ids: &[BufferId; 7],
    fault: BufferId,
    w: &Workspace,
) -> Result<()> {
    let client = arena.client().map_err(u::fault)?;
    bank.check_client(client)?;
    let mut all = Zeroizing::new(Vec::with_capacity(10));
    all.extend_from_slice(ids);
    all.push(fault);
    if let Some(h) = &w.hybrid {
        all.extend_from_slice(&*h.ids);
    }
    let banks = [bank.gate(), bank.up(), bank.down()];
    let count = if crate::bank_slots::segmented_capacity(w.layout.capacity) {
        2
    } else {
        1
    };
    if banks.iter().any(|b| b.segments().len() != count)
        || bank.slots.capacity() != w.layout.capacity
        || bank.hidden != 2560
        || bank.ff != 640
        || bank.experts != 512
    {
        return Err(err("native routed bank geometry"));
    }
    let stream = arena.handle(fault).map_err(u::fault)?.stream;
    let public: Vec<_> = banks
        .into_iter()
        .flat_map(|b| b.segments().iter())
        .collect();
    let mut resources = Vec::new();
    resources
        .try_reserve_exact(8 + public.len())
        .map_err(|_| Error::LimitExceeded("native routed physical metadata"))?;
    for (i, &id) in all.iter().enumerate() {
        if all[..i].contains(&id) {
            return Err(err("native routed owning alias"));
        }
        let h = arena.handle(id).map_err(u::fault)?;
        let n = arena.layout(id).map_err(u::fault)?.owned_bytes as u64;
        if h.stream != stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != n
            || h.size_in_used() != n
        {
            return Err(err("native routed private full handle"));
        }
        let mut metadata = u::resources(client, vec![h.clone()])?;
        let r = metadata.pop().ok_or(Error::InvalidBufferLength)?;
        let v = r.resource();
        check_page(v.offset, v.size, v.buffer.size(), n, n)?;
        if resources.iter().any(|b| b == &v.buffer) {
            return Err(err("native routed physical alias"));
        }
        resources.push(v.buffer.clone());
    }
    for segment in &public {
        let h = segment.handle();
        let n = (segment.words() * 4) as u64;
        if h.stream != stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size_in_used() != n
            || h.size() != n
        {
            return Err(err("native routed public full handle"));
        }
        let mut metadata = u::resources(client, vec![h.clone()])?;
        let r = metadata.pop().ok_or(Error::InvalidBufferLength)?;
        let v = r.resource();
        let bound = main_admission::page_bytes(n).map_err(admission_error)?;
        check_page(v.offset, v.size, v.buffer.size(), n, bound)?;
        if resources.iter().any(|b| b == &v.buffer) {
            return Err(err("native routed public physical alias"));
        }
        resources.push(v.buffer.clone());
    }
    // Use actual layer dtypes, not the first layer, at every launch proof.
    for (b, p) in banks.into_iter().zip(g.projections()) {
        if b.expert_bytes() != p.expert_bytes() {
            return Err(err("native routed public expert extent"));
        }
    }
    for shard in 0..count {
        let p = banks.map(|b| &b.segments()[shard]);
        if p.iter()
            .any(|v| v.first_slot() != p[0].first_slot() || v.capacity() != p[0].capacity())
        {
            return Err(err("native routed mismatched shards"));
        }
        g.validate_shard(p[0].first_slot(), p[0].capacity(), p.map(|s| s.words()))
            .map_err(err)?;
    }
    Ok(())
}
fn check_page(offset: u64, size: u64, physical: u64, logical: u64, bound: u64) -> Result<()> {
    if logical == 0 || offset != 0 || size != logical || physical < logical || physical > bound {
        return Err(err("native routed physical extent"));
    }
    Ok(())
}
fn launch_wave(
    arena: &PrivateArena,
    bank: &ExpertBanks,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
    groups: usize,
) -> Result<()> {
    let [gate, up, down] = g.projections();
    let client = arena.client().map_err(u::fault)?;
    for shard in 0..bank.gate().segments().len() {
        let ga = &bank.gate().segments()[shard];
        let u = &bank.up().segments()[shard];
        let d = &bank.down().segments()[shard];
        unsafe {
            k::grouped_gate_up::launch_unchecked::<WgpuRuntime>(
                client,
                g.grid(groups, false).map_err(err)?,
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(ga.handle().clone(), ga.words()),
                ArrayArg::from_raw_parts(u.handle().clone(), u.words()),
                arg(arena, ids[INPUT])?,
                arg(arena, ids[MAP])?,
                arg(arena, ids[ACT])?,
                arg(arena, ids[STATUS])?,
                s.map_first(),
                s.input_first(),
                s.input_stride(),
                s.act_first(),
                s.act_stride(),
                s.status_first(),
                ga.first_slot(),
                ga.capacity(),
                g.hidden(),
                g.ff(),
                gate.dtype(),
                up.dtype(),
                gate.row_bytes(),
                up.row_bytes(),
                g.tile(),
                g.gate_policy() == k::GatePolicy::Q4Block,
            );
            k::grouped_down::launch_unchecked::<WgpuRuntime>(
                client,
                g.grid(groups, true).map_err(err)?,
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(d.handle().clone(), d.words()),
                arg(arena, ids[ACT])?,
                arg(arena, ids[MAP])?,
                arg(arena, ids[PART])?,
                arg(arena, ids[STATUS])?,
                s.map_first(),
                s.act_first(),
                s.act_stride(),
                s.partial_first(),
                s.partial_stride(),
                s.status_first(),
                d.first_slot(),
                d.capacity(),
                g.hidden(),
                g.ff(),
                down.dtype(),
                down.row_bytes(),
                g.tile(),
            );
        }
    }
    Ok(())
}
fn launch_reduce(
    arena: &PrivateArena,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
) -> Result<()> {
    let c = arena.client().map_err(u::fault)?;
    unsafe {
        k::grouped_reduce::launch_unchecked::<WgpuRuntime>(
            c,
            g.reduce_grid(),
            CubeDim::new_1d(64),
            arg(arena, ids[PART])?,
            arg(arena, ids[PROB])?,
            arg(arena, ids[MAP])?,
            arg(arena, ids[STATUS])?,
            arg(arena, ids[OUTPUT])?,
            s.map_first(),
            s.partial_first(),
            s.partial_stride(),
            s.prob_first(),
            s.status_first(),
            s.output_first(),
            s.output_stride(),
            g.hidden(),
        );
    }
    Ok(())
}
fn launch_finite(
    arena: &PrivateArena,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
) -> Result<()> {
    let c = arena.client().map_err(u::fault)?;
    unsafe {
        k::grouped_finite::launch_unchecked::<WgpuRuntime>(
            c,
            g.finite_grid(),
            CubeDim::new_1d(64),
            arg(arena, ids[INPUT])?,
            arg(arena, ids[ACT])?,
            arg(arena, ids[PART])?,
            arg(arena, ids[OUTPUT])?,
            arg(arena, ids[MAP])?,
            arg(arena, ids[STATUS])?,
            s.map_first(),
            s.input_first(),
            s.input_stride(),
            s.act_first(),
            s.act_stride(),
            s.partial_first(),
            s.partial_stride(),
            s.output_first(),
            s.output_stride(),
            s.status_first(),
            g.hidden(),
            g.ff(),
        );
    }
    Ok(())
}

/// Same-model status/guard fold. Counts/ranges/slots stay runtime, and this
/// only ORs the existing global fault: no routed-only success packet exists.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::if_same_then_else
)] // Nested range guards avoid eager invalid-coordinate arithmetic.
fn fold_kernel(
    status: &Array<u32>,
    act: &Array<u32>,
    part: &Array<u32>,
    map: &Array<u32>,
    prob: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    first: usize,
    length: usize,
    complete: u32,
    #[comptime] hidden: usize,
    #[comptime] ff: usize,
) {
    let i = ABSOLUTE_POS;
    let mut bad = false;
    let d = comptime!(hidden.div_ceil(64));
    let f = comptime!(ff.div_ceil(64));
    let a = comptime!(160 + 8 * hidden.div_ceil(64));
    let p = comptime!(160 + 8 * hidden.div_ceil(64) + 80 * ff.div_ceil(64));
    let o =
        comptime!(160 + 8 * hidden.div_ceil(64) + 80 * ff.div_ceil(64) + 80 * hidden.div_ceil(64));
    let end =
        comptime!(160 + 16 * hidden.div_ceil(64) + 80 * ff.div_ceil(64) + 80 * hidden.div_ceil(64));
    if count == 0 || count > 8 || first >= count || length == 0 || length > count {
        bad = true;
    } else if first + length > count {
        bad = true;
    } else {
        if i < status.len() {
            if i < 160 {
                let route = i % 80;
                let token = route / 10;
                if token >= count {
                    if status[i] != 0 {
                        bad = true;
                    }
                } else if token >= first && token < first + length {
                    if status[i] != 1 {
                        bad = true;
                    }
                }
            } else if i < a {
                let token = (i - 160) / d;
                if token >= count {
                    if status[i] != 0 {
                        bad = true;
                    }
                } else {
                    if status[i] != 1 {
                        bad = true;
                    }
                }
            } else if i < p {
                let token = (i - a) / (10 * f);
                if token >= count {
                    if status[i] != 0 {
                        bad = true;
                    }
                } else if token >= first && token < first + length {
                    if status[i] != 1 {
                        bad = true;
                    }
                }
            } else if i < o {
                let token = (i - p) / (10 * d);
                if token >= count {
                    if status[i] != 0 {
                        bad = true;
                    }
                } else if token >= first && token < first + length {
                    if status[i] != 1 {
                        bad = true;
                    }
                }
            } else if i < end {
                let token = (i - o) / d;
                if token >= count {
                    if status[i] != 0 {
                        bad = true;
                    }
                } else if complete != 0 {
                    if status[i] != 1 {
                        bad = true;
                    }
                }
            } else {
                if status[i] != 0 {
                    bad = true;
                }
            }
        }
        if i < act.len() {
            if i >= count * 10 * ff {
                if act[i] != 0 {
                    bad = true;
                }
            }
        }
        if i < part.len() {
            if i >= count * 10 * hidden {
                if part[i] != 0 {
                    bad = true;
                }
            }
        }
        if i < map.len() {
            if i >= k::MAP_WORDS {
                if map[i] != 0 {
                    bad = true;
                }
            }
        }
        if i < prob.len() {
            if i >= count * 10 {
                if prob[i] != 0 {
                    bad = true;
                }
            }
        }
    }
    if bad {
        fault[0].fetch_or(4);
    }
}
#[allow(clippy::too_many_arguments)]
fn fold(
    arena: &PrivateArena,
    g: k::Geometry,
    ids: &[BufferId; 7],
    fault: BufferId,
    count: usize,
    first: usize,
    length: usize,
    complete: bool,
) -> Result<()> {
    if !(1..=8).contains(&count)
        || length == 0
        || first.checked_add(length).is_none_or(|n| n > count)
        || u::words(arena, fault)? != 1
    {
        return Err(Error::InvalidBufferLength);
    }
    let max = [STATUS, ACT, PART, MAP, PROB]
        .into_iter()
        .try_fold(0, |n, i| {
            Ok::<_, Error>(n.max(arena.layout(ids[i]).map_err(u::fault)?.owned_words))
        })?;
    fold_kernel::launch::<WgpuRuntime>(
        arena.client().map_err(u::fault)?,
        u::grid(max)?,
        CubeDim::new_1d(64),
        arg(arena, ids[STATUS])?,
        arg(arena, ids[ACT])?,
        arg(arena, ids[PART])?,
        arg(arena, ids[MAP])?,
        arg(arena, ids[PROB])?,
        arg(arena, fault)?,
        count,
        first,
        length,
        u32::from(complete),
        g.hidden(),
        g.ff(),
    );
    Ok(())
}

impl Workspace {
    fn check_pending(
        &self,
        e: &GpuExecutor<'_>,
        s: &Session,
        arena: &PrivateArena,
        p: &Pending,
    ) -> Result<()> {
        self.check(e, s, arena)?;
        if !p.armed
            || *p.stamp != *s.stamp
            || *p.layer >= self.layout.geometries.len()
            || !Arc::ptr_eq(&p.poison, &self.poison)
            || *self.policy.issued & (1 << *p.layer) == 0
        {
            return Err(err("native routed stale pending receipt"));
        }
        let bank = e
            .banks
            .get(&p.key)
            .ok_or(err("native routed original bank removed"))?;
        if bank.instance() != p.instance
            || p.key != format!("blk.{}.ffn_gate_exps.weight", *p.layer)
        {
            return Err(err("native routed bank instance changed"));
        }
        let handles: Vec<_> = [bank.gate(), bank.up(), bank.down()]
            .into_iter()
            .flat_map(|b| b.segments().iter().map(|p| p.handle()))
            .collect();
        if handles.len() != p.pins.len()
            || handles.iter().zip(&p.pins).any(|(a, b)| {
                a.stream != b.stream
                    || a.size() != b.size()
                    || a.offset_start != b.offset_start
                    || a.offset_end != b.offset_end
            })
        {
            return Err(err("native routed immutable bank pin changed"));
        }
        for (a, b) in handles.iter().zip(&p.pins) {
            let metadata = u::resources(
                arena.client().map_err(u::fault)?,
                vec![(*a).clone(), b.clone()],
            )?;
            if metadata.len() != 2 {
                return Err(Error::InvalidBufferLength);
            }
            let (left, right) = (&metadata[0], &metadata[1]);
            if left.resource().buffer != right.resource().buffer
                || left.resource().offset != right.resource().offset
                || left.resource().size != right.resource().size
            {
                return Err(err("native routed actual pinned bank replaced"));
            }
        }
        let staged = p
            .staged
            .as_ref()
            .ok_or(err("native routed staged ledger removed"))?;
        let last = p
            .last
            .as_ref()
            .ok_or(err("native routed final wave removed"))?;
        if (*last.len == 0 && p.count1.is_none())
            || *last.len > 80
            || *p.waves == 0
            || *p.waves > 8
            || staged.accepted_waves() + 1 != *p.waves
        {
            return Err(err("native routed final wave count"));
        }
        staged
            .validate_pending(&bank.slots, &last.pending, &last.selected[..*last.len])
            .map_err(|_| err("native routed final bank generation"))?;
        if let Some(proof) = &p.count1 {
            if s.stamp.count != 1 || *p.waves != 1 || self.cpu_miss_cap() != Some(proof.cap()) {
                return Err(err("native count1 foreign pending policy"));
            }
            proof.validate_source(&bank.slots, staged, last, true)?;
        } else if self.cpu_miss_cap().is_some() {
            return Err(err("native count1 missing compact proof"));
        }
        let g = self.layout.geometries[*p.layer];
        let ids = self.ids(p.views.ids[0], p.views.ids[1]);
        let v = &p.views.offsets;
        let spans = k::Spans::new([0, v[0], v[1], 0, g.ff(), 0, g.hidden(), 0, v[2], v[3], 0]);
        let mut words = Zeroizing::new([0usize; 7]);
        for (i, &id) in ids.iter().enumerate() {
            words[i] = u::words(arena, id)?;
        }
        spans.validate(g, *words).map_err(err)?;
        physical(arena, bank, g, &ids, s.stamp.fault, self)
    }
    /// OWNER PRECONDITION: the actual whole-model final output/fault fence and
    /// cancellation/context/runtime-I/O acceptance succeeded; every GDN/QSA/PLE,
    /// shared expert, GR and other owner is preflighted BEFORE ANY publication.
    /// This unconnected seam does no read/sync/audit and cannot prove GPU success.
    pub(crate) fn accept_after_whole_model_chunk_acceptance<'a, 'f>(
        &'a mut self,
        e: &'a mut GpuExecutor<'f>,
        s: &'a mut Session,
        arena: &PrivateArena,
        pending: &[&Pending],
        cancel: &AtomicBool,
    ) -> Result<Acceptance<'a, 'f>> {
        let result = guarded(&self.poison.clone(), || {
            u::cancel(cancel)?;
            self.check(e, s, arena)?;
            e.model.verify_file_unchanged()?;
            let mut seen = Zeroizing::new(0u64);
            let mut cpu_misses = Zeroizing::new(0u64);
            for p in pending {
                self.check_pending(e, s, arena, p)?;
                let bit = 1u64 << *p.layer;
                if *seen & bit != 0 {
                    return Err(err("native routed duplicate pending receipt"));
                }
                *seen |= bit;
                // Count only actual joined/source-validated original Pending
                // proofs. No dispatch attempt or requested policy is evidence.
                let count = p.count1.as_ref().map_or(0, |proof| proof.cpu_count());
                *cpu_misses = cpu_misses
                    .checked_add(u64::try_from(count).map_err(|_| Error::ArithmeticOverflow)?)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            self.policy.preflight(*seen)?;
            u::cancel(cancel)?;
            Ok(cpu_misses)
        });
        if result.is_err() {
            e.poison_resident()
        }
        let cpu_misses = result?;
        Ok(Acceptance {
            cpu_misses,
            workspace: self,
            executor: e,
            session: s,
            armed: true,
        })
    }
}
/// Exclusive sealed all-layer permit: original BankSlots remain inaccessible
/// to scalar entry/cache replacement until ALL publications and finish succeed.
pub(crate) struct Acceptance<'a, 'f> {
    cpu_misses: Zeroizing<u64>,
    workspace: &'a mut Workspace,
    executor: &'a mut GpuExecutor<'f>,
    session: &'a mut Session,
    armed: bool,
}
impl Acceptance<'_, '_> {
    pub(crate) fn completed_cpu_misses(&self) -> u64 {
        *self.cpu_misses
    }
    pub(crate) fn commit(
        &mut self,
        mut p: Pending,
        arena: &PrivateArena,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let result = guarded(&self.workspace.poison.clone(), || {
            u::cancel(cancel)?;
            self.workspace
                .check_pending(self.executor, self.session, arena, &p)?;
            // Duplicate publication is rejected before touching any original ledger.
            let bit = 1u64 << *p.layer;
            if self.workspace.policy.phase != Phase::Accepted
                || *self.workspace.policy.committed & bit != 0
            {
                return Err(err("native routed invalid publication"));
            }
            let bank = self
                .executor
                .banks
                .get_mut(&p.key)
                .ok_or(err("native routed original bank missing"))?;
            let mut staged = p
                .staged
                .take()
                .ok_or(err("native routed missing staged ledger"))?;
            let last = p
                .last
                .take()
                .ok_or(err("native routed missing final wave"))?;
            staged
                .advance_after_completed_consumers(
                    &bank.slots,
                    last.pending,
                    &last.selected[..*last.len],
                )
                .map_err(|_| err("native routed final consumer acceptance"))?;
            staged
                .publish_after_whole_model_acceptance(&mut bank.slots, *p.waves)
                .map_err(|_| err("native routed original ledger publication"))?;
            self.workspace.policy.commit(*p.layer)?;
            p.armed = false;
            u::cancel(cancel)
        });
        if result.is_err() {
            self.executor.poison_resident()
        }
        result
    }
    pub(crate) fn finish(mut self, arena: &PrivateArena, cancel: &AtomicBool) -> Result<()> {
        let result = guarded(&self.workspace.poison.clone(), || {
            u::cancel(cancel)?;
            self.workspace.check(self.executor, self.session, arena)?;
            self.workspace.policy.finish()?;
            self.executor
                .batch_moe
                .take()
                .ok_or(err("native routed registry missing at finish"))?;
            self.session.armed = false;
            Ok(())
        });
        if result.is_err() {
            self.executor.poison_resident()
        }
        result?;
        self.armed = false;
        Ok(())
    }
}
impl Drop for Acceptance<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            self.workspace.poison.store(true, Ordering::Release)
        }
        self.armed.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bank_slots::BankSlots;
    #[test]
    fn constructed_hybrid_worker_geometry_admits_only_checked_public_pools() {
        for count in [4, 8, 16] {
            assert!(hybrid_worker_geometry(count, true));
        }
        assert!(hybrid_worker_geometry(16, false));
        for count in [0, 1, 3, 4, 8, 17, usize::MAX] {
            assert!(!hybrid_worker_geometry(count, false));
        }
        for count in [0, 1, 3, 17, usize::MAX] {
            assert!(!hybrid_worker_geometry(count, true));
        }
    }
    fn geometry(capacity: usize) -> k::Geometry {
        k::Geometry::new(
            [
                k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512).unwrap(),
                k::Projection::new(crate::quant_matvec::Q4_K, 2560, 640, 512).unwrap(),
                k::Projection::new(crate::quant_matvec::Q8_0, 640, 2560, 512).unwrap(),
            ],
            8,
            capacity,
        )
        .unwrap()
    }
    fn stamp() -> Stamp {
        Stamp {
            workspace: 7,
            chunk: 11,
            start: 13,
            count: 8,
            sequence: SequenceKey::new().unwrap(),
            fault: BufferId::metadata_fixture(3, 5, 19),
            ids: std::array::from_fn(|i| BufferId::metadata_fixture(3, 5, i)),
        }
    }
    #[test]
    fn q4_replacements_validate_every_layer_before_any_installation() {
        for capacity in [16, 64, 96, 128, 144] {
            let mut layers = [geometry(capacity); 48];
            let original = layers;
            install_q4_geometries(&mut layers).unwrap();
            for (candidate, old) in layers.iter().zip(original) {
                assert_eq!(candidate.tile(), 1);
                assert_eq!(candidate.gate_policy(), k::GatePolicy::Q4Block);
                assert_eq!(candidate.projections(), old.projections());
                assert_eq!(candidate.status_words(), old.status_words());
                assert_eq!(candidate.scratch_bytes(true), old.scratch_bytes(true));
            }
        }
        let mut layers = [geometry(16); 48];
        layers[47] = k::Geometry::new(
            [k::Projection::new(crate::quant_matvec::F32, 256, 256, 512).unwrap(); 3],
            8,
            16,
        )
        .unwrap();
        let before = layers;
        assert!(install_q4_geometries(&mut layers).is_err());
        assert_eq!(layers, before);
        for count in [0, 1, 47] {
            assert!(install_q4_geometries(&mut layers[..count]).is_err());
            assert_eq!(layers, before);
        }
    }
    #[test]
    fn exact_complete_owned_scratch_and_separate_runtime_envelope() {
        let requests = [
            80 * 640,
            80 * 2560,
            k::MAP_WORDS,
            80,
            geometry(64).status_words(),
        ]
        .map(|n| BufferLayout::new(n).unwrap().owned_bytes as u64);
        assert_eq!(requests, [262144, 1048576, 32768, 32768, 32768]);
        assert_eq!(requests.iter().sum::<u64>(), PRIVATE_BYTES);
        assert_eq!(785_186_816 + PRIVATE_BYTES, 786_595_840);
        let infos = 128 * 512 * 2;
        let writes = 2 * 32768;
        let fault = 32768;
        let direct = infos + writes + fault;
        assert_eq!(direct, 229376);
        // Two distinct INFO binding-kind target slabs can each retain <=8KiB
        // unused append-only tail; inspection duplicates must also fit.
        assert!(2 * (direct + 2 * 8192) <= IO_INCREMENT);
        assert_eq!(48 * 32768 + 512 * 1024, HOST_PRIVATE);
        assert_eq!(Workspace::runtime_io_increment(), 512 * 1024);
        for capacity in [10, 16, 64, 72, 96, 128, 144] {
            assert_eq!(geometry(capacity).status_words(), 4800);
            let bound = StagedSlots::bounded_native_metadata_bytes(capacity).unwrap();
            assert!(
                bound
                    + std::mem::size_of::<Pending>()
                    + std::mem::size_of::<Last>()
                    + 6 * std::mem::size_of::<Handle>()
                    < 32768
            );
        }
        for c in [0, 9, 73, 95, 97, usize::MAX] {
            assert!(StagedSlots::bounded_native_metadata_bytes(c).is_err());
        }
    }
    #[test]
    fn entire_family_preflight_precedes_every_original_publication() {
        for count in [1, 3, 5, 8] {
            let mut p = Policy::new(48).unwrap();
            assert!(p.completed_boundary().is_err());
            p.begin(0, count).unwrap();
            assert!(p.completed_boundary().is_err());
            assert!(p.commit(0).is_err());
            for bad in [64, u32::MAX as usize + 1, usize::MAX] {
                assert!(p.commit(bad).is_err());
            }
            for layer in 0..48 {
                p.issue(layer).unwrap();
                assert!(p.preflight((1u64 << (layer + 1)) - 1).is_err() || layer == 47);
            }
            for layer in 0..48 {
                p.commit(layer).unwrap();
                assert!(p.commit(layer).is_err());
                if layer < 47 {
                    assert!(p.finish().is_err());
                }
            }
            p.finish().unwrap();
            p.completed_boundary().unwrap();
            let previous = *p.chunk;
            p.begin(count as u64, count).unwrap();
            assert!(p.completed_boundary().is_err());
            assert!(*p.chunk > previous);
            assert_eq!(*p.issued, 0);
        }
        let mut p = Policy::new(48).unwrap();
        assert!(p.begin(u64::MAX, 1).is_err());
        for count in [0, 9, usize::MAX] {
            assert!(p.begin(0, count).is_err());
        }
        p.begin(0, 8).unwrap();
        assert!(p.begin(0, 8).is_err());
        assert!(p.issue(1).is_err());
        p.issue(0).unwrap();
        assert!(p.issue(0).is_err());
        assert!(p.preflight(u64::MAX).is_err());
        assert!(p.finish().is_err());
    }
    #[test]
    fn actual_last_pending_plan_retains_original_ledger_until_whole_model_swap() {
        for cap in [16, 64, 72, 96, 128, 144] {
            let mut bank = BankSlots::new(cap, 512).unwrap();
            let mut staged = StagedSlots::begin(&bank).unwrap();
            let first = (0..10).collect::<Vec<_>>();
            let pending = staged.plan(&bank, &first).unwrap();
            staged
                .advance_after_completed_consumers(&bank, pending, &first)
                .unwrap();
            let final_selected = (10..20).collect::<Vec<_>>();
            let pending = staged.plan(&bank, &final_selected).unwrap();
            staged.validate_origin(&bank).unwrap();
            staged
                .validate_pending(&bank, &pending, &final_selected)
                .unwrap();
            assert_eq!(staged.accepted_waves(), 1);
            let actual = staged.retained_native_metadata_bytes(&pending).unwrap();
            assert!(actual <= StagedSlots::bounded_native_metadata_bytes(cap).unwrap());
            assert!(
                actual
                    + std::mem::size_of::<Pending>()
                    + std::mem::size_of::<Last>()
                    + 6 * std::mem::size_of::<Handle>()
                    < 32768
            );
            // Neither completed interior waves nor a validated final wave publishes.
            assert_eq!(bank.plan(&first).unwrap().uploads.len(), 10);
            staged
                .advance_after_completed_consumers(&bank, pending, &final_selected)
                .unwrap();
            staged
                .publish_after_whole_model_acceptance(&mut bank, 2)
                .unwrap();
            assert!(bank.plan(&final_selected).unwrap().uploads.is_empty());
        }
        let mut bank = BankSlots::new(16, 512).unwrap();
        let mut staged = StagedSlots::begin(&bank).unwrap();
        let p = staged.plan(&bank, &[1, 2, 3]).unwrap();
        let foreign = BankSlots::new(16, 512).unwrap();
        assert!(staged.validate_pending(&foreign, &p, &[1, 2, 3]).is_err());
        let replacement = bank.plan(&[40]).unwrap();
        bank.commit(replacement).unwrap();
        assert!(staged.validate_origin(&bank).is_err());
    }
    #[test]
    fn native_pending_and_session_drop_are_sticky_even_inside_an_admission_lock() {
        let poison = Arc::new(AtomicBool::new(false));
        let ledger = Mutex::new(MainAdmission::new(32768, 0).unwrap());
        let held = ledger.lock().unwrap();
        let session = Session {
            stamp: Zeroizing::new(stamp()),
            poison: poison.clone(),
            armed: true,
        };
        drop(session);
        assert!(poison.load(Ordering::Acquire));
        assert!(ledger.try_lock().is_err());
        assert!(!held.is_poisoned());
        drop(held);
        let poison = Arc::new(AtomicBool::new(false));
        let pending = Pending {
            views: Views {
                ids: Zeroizing::new([BufferId::metadata_fixture(3, 5, 20); 2]),
                offsets: Zeroizing::new([0; 4]),
            },
            stamp: Zeroizing::new(stamp()),
            layer: Zeroizing::new(2),
            key: "blk.2.ffn_gate_exps.weight".into(),
            instance: crate::expert_banks::fixture_bank_instance(),
            pins: Vec::new(),
            staged: None,
            last: None,
            count1: None,
            waves: Zeroizing::new(0),
            poison: poison.clone(),
            armed: true,
        };
        let held = ledger.lock().unwrap();
        drop(pending);
        assert!(poison.load(Ordering::Acquire));
        assert!(ledger.try_lock().is_err());
        drop(held);
        let poison = Arc::new(AtomicBool::new(false));
        let session = Session {
            stamp: Zeroizing::new(stamp()),
            poison: poison.clone(),
            armed: false,
        };
        drop(session);
        assert!(!poison.load(Ordering::Acquire));
    }
    #[test]
    fn source_error_cancel_and_panic_poison_before_following_retry() {
        for mode in 0..3 {
            let poison = AtomicBool::new(false);
            let result = guarded(&poison, || -> Result<()> {
                match mode {
                    0 => Err(Error::InvalidBufferLength),
                    1 => {
                        let c = AtomicBool::new(true);
                        u::cancel(&c)
                    }
                    _ => panic!("fixed public partial-fill unwind"),
                }
            });
            assert!(result.is_err());
            assert!(poison.load(Ordering::Acquire));
        }
        let mut stamp = stamp();
        stamp.zeroize();
        assert_eq!(
            (stamp.workspace, stamp.chunk, stamp.start, stamp.count),
            (0, 0, 0, 0)
        );
        assert!(
            stamp
                .ids
                .iter()
                .all(|&id| id == BufferId::metadata_fixture(0, 0, 0))
        );
        assert!(check_page(0, 32768, 32768, 32768, 32768).is_ok());
        for tuple in [
            (1, 32768, 32768, 32768, 32768),
            (0, 32767, 32768, 32768, 32768),
            (0, 32768, 65536, 32768, 32768),
            (0, 32768, 32767, 32768, 32768),
        ] {
            assert!(check_page(tuple.0, tuple.1, tuple.2, tuple.3, tuple.4).is_err());
        }
    }
    // Independent host acceptance uses original region definitions rather than
    // emulating the GPU per-word branches; no activation/model bytes are needed.
    fn accept_status(
        g: k::Geometry,
        count: usize,
        first: usize,
        length: usize,
        complete: bool,
        s: &[u32],
    ) -> bool {
        if !(1..=8).contains(&count)
            || length == 0
            || first.checked_add(length).is_none_or(|n| n > count)
            || s.len() < g.status_words()
        {
            return false;
        }
        let d = g.hidden().div_ceil(64);
        let f = g.ff().div_ceil(64);
        let mut offset = 0;
        for (rows, width, kind) in [
            (80, 1, 0),
            (80, 1, 0),
            (8, d, 1),
            (80, f, 2),
            (80, d, 2),
            (8, d, 3),
        ] {
            for row in 0..rows {
                let token = if rows == 80 { row / 10 } else { row };
                let want = if token >= count {
                    Some(0)
                } else if kind == 1
                    || (kind != 3 && token >= first && token < first + length)
                    || (kind == 3 && complete)
                {
                    Some(1)
                } else {
                    None
                };
                if let Some(v) = want
                    && s[offset + row * width..offset + (row + 1) * width]
                        .iter()
                        .any(|&x| x != v)
                {
                    return false;
                }
            }
            offset += rows * width;
        }
        s[offset..].iter().all(|&v| v == 0)
    }
    #[test]
    fn finite_ready_acceptance_requires_every_active_region_and_owned_tail() {
        let g = geometry(64);
        for count in [1, 3, 5, 8] {
            let mut s = vec![0u32; 8192];
            let d = g.hidden().div_ceil(64);
            let f = g.ff().div_ceil(64);
            let mut offset = 0;
            for (rows, width, active) in [
                (80, 1, count * 10),
                (80, 1, count * 10),
                (8, d, count),
                (80, f, count * 10),
                (80, d, count * 10),
                (8, d, count),
            ] {
                s[offset..offset + active * width].fill(1);
                offset += rows * width;
            }
            assert!(accept_status(g, count, 0, count, true, &s));
            for i in 0..s.len() {
                let old = s[i];
                s[i] = if old == 0 { 1 } else { 0 };
                assert!(!accept_status(g, count, 0, count, true, &s), "status {i}");
                s[i] = old;
            }
            if count > 1 {
                let mut partial = s.clone();
                partial[..count * 10].fill(0);
                partial[80..80 + count * 10].fill(0);
                partial[..10].fill(1);
                partial[80..90].fill(1);
                partial[4480..4480 + count * d].fill(0);
                assert!(accept_status(g, count, 0, 1, false, &partial));
                assert!(!accept_status(g, count, 0, count, true, &partial));
            }
        }
    }
    #[test]
    fn fold_offline_wgsl_keeps_all_request_coordinates_runtime_and_guarded() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
            let atom = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(uint);
            b.scope.register_type::<usize>(uint);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let status = b.input_array(Type::new(uint)).into();
            let act = b.input_array(Type::new(uint)).into();
            let part = b.input_array(Type::new(uint)).into();
            let map = b.input_array(Type::new(uint)).into();
            let prob = b.input_array(Type::new(uint)).into();
            let fault = b.output_array(Type::new(atom)).into();
            let count = b.scalar(uint).into();
            let first = b.scalar(uint).into();
            let length = b.scalar(uint).into();
            let complete = b.scalar(uint).into();
            fold_kernel::expand(
                &mut b.scope,
                status,
                act,
                part,
                map,
                prob,
                fault,
                count,
                first,
                length,
                complete,
                2560,
                640,
            );
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("native_routed_fault_fold"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    uint,
                )
                .unwrap()
                .to_string();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert!(source.contains("atomicOr"));
            let storage = module
                .global_variables
                .iter()
                .filter(|(_, v)| matches!(v.space, wgpu::naga::AddressSpace::Storage { .. }))
                .count();
            assert!(storage == 6 || storage == 7);
            assert!(!source.contains("array<u32, 80>"));
        }
    }
}
