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
#[path = "resident_batch_router.rs"]
pub(crate) mod batch_router;
#[path = "native_count1_selection.rs"]
mod count1;
#[path = "resident_moe_count1.rs"]
mod count1_executor;
#[path = "resident_router.rs"]
mod resident_router;
pub(crate) use resident_router::{
    RESIDENT_ROUTER_HOST, RESIDENT_ROUTER_INTEGER_CLEAR_INFO, RESIDENT_ROUTER_MAIN,
    RESIDENT_ROUTER_PREFLIGHT_INFO, RESIDENT_ROUTER_RUNTIME_LIMIT, resident_router_runtime,
};
#[path = "resident_frozen_sources.rs"]
pub(crate) mod frozen_sources;
#[path = "count1_hybrid_moe.rs"]
mod hybrid_k;
#[path = "iq_moe_integer_verify3.rs"]
pub(crate) mod prepared_verify3;
pub(crate) const FROZEN_STAGING_OWNER_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const FROZEN_STAGING_MAIN_BYTES: u64 = 3 * FROZEN_STAGING_OWNER_BYTES;
pub(crate) const FROZEN_STAGING_RUNTIME_BYTES: u64 = 6 * 48 * 2048;
// Construction-only charge for the explicit frozen public GU candidate.
// Whole pages and transient INFO/HOST reservations remain independent.
pub(crate) const FROZEN_GU_INTEGER_SCRATCH_WORDS: usize = 16384;
pub(crate) const FROZEN_GU_INTEGER_MAIN_BYTES: u64 = 65536;
pub(crate) const FROZEN_GU_INTEGER_HOST_BYTES: usize = 65536;
pub(crate) const FROZEN_GU_INTEGER_INFO_BYTES: u64 = 2 * 48 * 2048;
pub(crate) const FROZEN_GU_INTEGER_RUNTIME_BYTES: u64 =
    FROZEN_GU_INTEGER_INFO_BYTES + 2 * 2048 + 4 * FROZEN_GU_INTEGER_MAIN_BYTES;
// Independent construction-only charge for route-indexed integer DOWN.
pub(crate) const FROZEN_DOWN_INTEGER_SCRATCH_WORDS: usize = 16384;
pub(crate) const FROZEN_DOWN_INTEGER_MAIN_BYTES: u64 = 65536;
pub(crate) const FROZEN_DOWN_INTEGER_HOST_BYTES: usize = 65536;
pub(crate) const FROZEN_DOWN_INTEGER_INFO_BYTES: u64 = 2 * 48 * 2048;
pub(crate) const FROZEN_DOWN_INTEGER_RUNTIME_BYTES: u64 =
    FROZEN_DOWN_INTEGER_INFO_BYTES + 2 * 2048 + 4 * FROZEN_DOWN_INTEGER_MAIN_BYTES;
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
const HOST_PRIVATE: u64 = if cfg!(feature = "strata-rust-port") {
    4
} else {
    2
} * 1024
    * 1024;
const METADATA_PER_LAYER: usize = if cfg!(feature = "strata-rust-port") {
    64
} else {
    32
} * 1024;
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
        Ok(Err(cause)) => {
            poison.store(true, Ordering::Release);
            Err(cause)
        }
        Err(_) => {
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
    if e.resident_banks_frozen {
        e.poison_resident();
        return Err(err("resident router forbids scalar bank mutation"));
    }
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
    resident_router: Option<resident_router::RouterOwners>,
    batch_router: Option<batch_router::Owners>,
    prepared_verify3: Option<BufferId>,
    // Constructor-frozen public experiment; never read from ambient settings.
    public_existing_fill: bool,
    public_four_members: bool,
    public_fill_counters: PublicFillCounters,
    frozen_sources: Option<Arc<frozen_sources::Sources>>,
    frozen_source_counters: frozen_sources::Counters,
    frozen_staging: Option<Zeroizing<[BufferId; 3]>>,
    frozen_gu_integer: Option<BufferId>,
    frozen_down_integer: Option<BufferId>,
}

/// Aggregate successfully queued demand-fill diagnostics, not publication or
/// physical-disk counters. Rejected/aborted model transactions are not receipts.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PublicFillCounters {
    pub first_touch_fills: u64,
    pub evictions: u64,
    pub direct_windows: u64,
    pub direct_components: u64,
    pub direct_bytes: u64,
}
impl PublicFillCounters {
    fn add(self, next: Self) -> Result<Self> {
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(Error::ArithmeticOverflow);
        Ok(Self {
            first_touch_fills: add(self.first_touch_fills, next.first_touch_fills)?,
            evictions: add(self.evictions, next.evictions)?,
            direct_windows: add(self.direct_windows, next.direct_windows)?,
            direct_components: add(self.direct_components, next.direct_components)?,
            direct_bytes: add(self.direct_bytes, next.direct_bytes)?,
        })
    }
}

fn public_existing_policy(
    enabled: bool,
    profile: expert_banks::OriginalPackedProfile,
    capacity: usize,
    tile: usize,
    hybrid: bool,
    router: bool,
) -> Result<()> {
    if enabled
        && (profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || capacity != crate::daslab_profile::BANK_SLOTS
            || tile != 8
            || hybrid
            || router)
    {
        return Err(err("public existing expert construction policy"));
    }
    Ok(())
}

fn public_existing_count(enabled: bool, count: usize, waves: usize) -> Result<()> {
    if enabled && (!(1..=3).contains(&count) || waves != 1) {
        return Err(err("public existing expert count policy"));
    }
    Ok(())
}

/// Actual facts from the attached frozen Source, original Workspace owners and
/// selected target/router permissions. A requested four-row flag is insufficient.
#[derive(Clone, Copy)]
struct FrozenFourChunkScope {
    source_four: bool,
    workspace_four: bool,
    target_four_reserved: bool,
    target_four_active: bool,
    target_rows_reserved: bool,
    frozen_router_reserved: bool,
    frozen_router_active: bool,
    router_four_reserved: bool,
    banks_frozen: bool,
    source_identity_matches: bool,
    gu_integer_sources_and_owners: bool,
    down_integer_sources_and_owners: bool,
    original_geometry: bool,
    ordinary_modes_absent: bool,
}
impl FrozenFourChunkScope {
    fn validate(self, count: usize, waves: usize) -> Result<()> {
        if count != 4
            || waves != 1
            || !self.source_four
            || !self.workspace_four
            || !self.target_four_reserved
            || !self.target_four_active
            || !self.target_rows_reserved
            || !self.frozen_router_reserved
            || !self.frozen_router_active
            || !self.router_four_reserved
            || !self.banks_frozen
            || !self.source_identity_matches
            || !self.gu_integer_sources_and_owners
            || !self.down_integer_sources_and_owners
            || !self.original_geometry
            || !self.ordinary_modes_absent
        {
            return Err(err("native actual frozen four-member chunk policy"));
        }
        Ok(())
    }
}

/// Valid canonical single-wave misses read the untouched committed ledger.
/// No staged advance intervenes for the admitted count1..3 /144-slot geometry.
fn classify_public_fills(
    slots: &crate::bank_slots::BankSlots,
    uploads: &[(usize, usize)],
) -> Result<PublicFillCounters> {
    if slots.capacity() != crate::daslab_profile::BANK_SLOTS
        || slots.experts() != 512
        || uploads.len() > 30
    {
        return Err(err("public expert fill counter geometry"));
    }
    let mut counts = PublicFillCounters::default();
    for (i, &(expert, slot)) in uploads.iter().enumerate() {
        if expert >= 512 || uploads[..i].iter().any(|&(e, s)| e == expert || s == slot) {
            return Err(err("public expert fill counter canonical uploads"));
        }
        match slots
            .resident_expert(slot)
            .map_err(|_| err("public expert fill counter slot"))?
        {
            None => counts.first_touch_fills += 1,
            Some(old) if old != expert => counts.evictions += 1,
            Some(_) => return Err(err("public expert fill counter is not a miss")),
        }
    }
    Ok(counts)
}
/// Whole-model public policy validation before owner allocation or mutation.
/// The kernel changes only unused workgroup scratch reductions, never the
/// original packed weights, active arithmetic, route map or acceptance domains.
pub(crate) fn live_member_reduction_policy(
    selected: bool,
    profile: crate::expert_banks::OriginalPackedProfile,
    capacity: usize,
    tile: usize,
    hybrid: bool,
    router: bool,
    prepared_verify3: bool,
) -> Result<()> {
    if selected
        && (profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            || capacity != crate::daslab_profile::BANK_SLOTS
            || tile != 8
            || hybrid
            || router
            || prepared_verify3)
    {
        return Err(err("native live-member original public policy"));
    }
    Ok(())
}
fn live_member_reduction_count(selected: bool, count: usize) -> Result<()> {
    if selected && !(1..=3).contains(&count) {
        return Err(err("native live-member target count"));
    }
    Ok(())
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
    pub(crate) fn live_member_reduction_enabled(&self) -> bool {
        self.layout.geometries[0].live_member_reduction()
    }
    pub(crate) fn public_four_members_policy(&self) -> bool {
        self.public_four_members
    }
    /// The original direct demand-fill path remains bounded to count1..3.
    /// Count4 is a separate no-fill Source path, authenticated before enqueue.
    fn validate_public_existing_chunk_count(
        &self,
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        count: usize,
        waves: usize,
    ) -> Result<()> {
        if !self.public_existing_fill || count != 4 {
            return public_existing_count(self.public_existing_fill, count, waves);
        }
        let source = self
            .frozen_sources
            .as_ref()
            .ok_or(err("native actual frozen four-member source absent"))?;
        FrozenFourChunkScope {
            source_four: source.four_members(),
            workspace_four: self.public_four_members_policy(),
            target_four_reserved: executor.target_four_rows_reserved,
            target_four_active: executor.target_four_rows_active,
            target_rows_reserved: executor.target_rows_reserved,
            frozen_router_reserved: executor.target_frozen_host_router_reserved
                && executor.target_batch_router_reserved
                && self.frozen_host_batch_router_reserved(),
            frozen_router_active: self.frozen_host_batch_router_active(),
            router_four_reserved: self.four_member_router_reserved(),
            banks_frozen: executor.resident_banks_frozen,
            source_identity_matches: executor.target_frozen_host_source_identity
                == Some(source.plan_identity()),
            gu_integer_sources_and_owners: source.gu_integer()
                && executor.target_frozen_gu_integer_reserved
                && self.frozen_gu_integer.is_some(),
            down_integer_sources_and_owners: source.down_integer()
                && executor.target_frozen_down_integer_reserved
                && self.frozen_down_integer.is_some(),
            original_geometry: executor.packed_profile
                == expert_banks::OriginalPackedProfile::DaslabIq3
                && executor.bank_slots == crate::daslab_profile::BANK_SLOTS
                && self.layout.capacity == crate::daslab_profile::BANK_SLOTS
                && self.layout.tile == 8
                && self.layout.geometries.len() == 48
                && graph.geometry().layers == 48,
            ordinary_modes_absent: self.hybrid.is_none()
                && self.resident_router.is_none()
                && self.prepared_verify3.is_none()
                && self.frozen_staging.is_none()
                && !self.live_member_reduction_enabled()
                && executor.adaptive.is_none()
                && executor.secondary.is_none()
                && executor.secondary_frequency.is_none()
                && !executor.resident_router_reserved,
        }
        .validate(count, waves)?;
        executor.check_target_rows_policy()?;
        // Reuse the accepted original bank generations, full512 mapping and
        // active router-table/instance checks; no device query or readback.
        for layer in 0..48 {
            self.check_batch_router_bank(executor, graph, arena, layer)?;
            let original = weights(graph, layer)?;
            let bank = executor
                .banks
                .get(original.gate.info().name())
                .ok_or(err("native actual frozen four-member bank absent"))?;
            let geometry = source.geometry(layer, bank)?;
            if !geometry.four_members() {
                return Err(err("native actual frozen four-member source geometry"));
            }
            geometry.validate_count(count).map_err(err)?;
        }
        Ok(())
    }
    /// Immutable optional four-row permission after accepted original warm
    /// completion, before frozen Source attachment/GPU-router activation.
    pub(crate) fn set_public_four_members_policy(
        &mut self,
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        enabled: bool,
    ) -> Result<()> {
        self.validate_public_warm_boundary(executor, graph, arena)?;
        if self.public_four_members && !enabled {
            return Err(err("native four-member policy is immutable"));
        }
        if !enabled {
            return Ok(());
        }
        if !executor.target_four_rows_reserved
            || !executor.target_rows_reserved
            || !executor.target_frozen_host_router_reserved
            || !executor.target_batch_router_reserved
            || !executor.target_frozen_gu_integer_reserved
            || !executor.target_frozen_down_integer_reserved
            || self.frozen_gu_integer.is_none()
            || self.frozen_down_integer.is_none()
            || !self.frozen_host_batch_router_reserved()
            || !self.four_member_router_reserved()
            || self.batch_router_active()
            || self.frozen_sources.is_some()
            || self.frozen_staging.is_some()
            || self.prepared_verify3.is_some()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || !self.public_existing_fill
            || self.live_member_reduction_enabled()
            || self.layout.geometries.len() != 48
            || self.layout.capacity != crate::daslab_profile::BANK_SLOTS
            || self.layout.tile != 8
            || self.layout.gate_policy != k::GatePolicy::Generic
            || executor.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
            || executor.adaptive.is_some()
            || executor.secondary.is_some()
            || executor.secondary_frequency.is_some()
            || executor.resident_router_reserved
            || executor.target_frozen_host_source_identity.is_some()
        {
            return Err(err("native four-member accepted public policy"));
        }
        // Prove all actual original model geometries BEFORE changing the flag.
        // Source attachment independently reconstructs the same full geometry.
        for &original in &self.layout.geometries {
            original
                .with_frozen_host_sources()
                .and_then(k::Geometry::with_inactive_partition_exit)
                .and_then(k::Geometry::with_four_members)
                .map_err(err)?;
        }
        self.public_four_members = true;
        Ok(())
    }
    pub(crate) fn public_existing_fill_enabled(&self) -> bool {
        self.public_existing_fill
    }
    pub(crate) fn public_fill_counters(&self) -> PublicFillCounters {
        self.public_fill_counters
    }
    pub(crate) fn prepared_verify3_owner_id(&self) -> Option<BufferId> {
        self.prepared_verify3
    }
    pub(crate) fn reserve_prepared_verify3(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        self.validate_ids(arena)?;
        if self.poison.load(Ordering::Acquire)
            || self.prepared_verify3.is_some()
            || self.frozen_gu_integer.is_some()
            || self.frozen_down_integer.is_some()
            || self.live_member_reduction_enabled()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.policy.phase != Phase::Idle
            || *self.policy.chunk != 0
        {
            return Err(err("Verify3 prepared construction reservation"));
        }
        for &geometry in &self.layout.geometries {
            prepared_verify3::validate_geometry(geometry)?;
        }
        let owner = arena
            .allocate_zeroed(prepared_verify3::SCRATCH_WORDS)
            .map_err(u::fault)?;
        self.prepared_verify3 = Some(owner);
        self.validate_ids(arena)
    }
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
            || self.public_existing_fill
            || self.live_member_reduction_enabled()
            || self.prepared_verify3.is_some()
            || self.frozen_gu_integer.is_some()
            || self.frozen_down_integer.is_some()
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
    pub(crate) fn cpu_phase_ns(&self) -> Option<[u64; 4]> {
        self.hybrid.as_ref().and_then(|h| h.operator.phase_ns())
    }
    pub(crate) fn cpu_buffered_weights(&self) -> Option<(bool, u64)> {
        self.hybrid.as_ref().map(|h| {
            (
                h.operator.buffered_weights(),
                h.operator.buffered_weight_reads(),
            )
        })
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
        if self.resident_router_active() {
            return Err(err("resident router policy is frozen"));
        }
        if self.live_member_reduction_enabled() {
            return Err(err("live-member construction policy is frozen"));
        }
        if self.prepared_verify3.is_some() {
            return Err(err("Verify3 prepared construction policy is frozen"));
        }
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
        if self.resident_router_active() {
            return Err(err("resident router policy is frozen"));
        }
        if self.live_member_reduction_enabled() {
            return Err(err("live-member construction policy is frozen"));
        }
        if self.prepared_verify3.is_some() {
            return Err(err("Verify3 prepared construction policy is frozen"));
        }
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
        if self.resident_router_active() {
            return Err(err("resident router policy is frozen"));
        }
        if self.live_member_reduction_enabled() {
            return Err(err("live-member construction policy is frozen"));
        }
        if self.prepared_verify3.is_some() {
            return Err(err("Verify3 prepared construction policy is frozen"));
        }
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
        Self::allocate_with_public_existing(
            executor, graph, sequence, arena, admission, poison, limits, tile, cancel, false,
        )
    }
    /// Parent Config/Plan must capture this opt-in before model construction and
    /// restrict supplied chunks to1..3. The ordinary constructor remains false.
    /// No owners, source-page requests or IO ceilings change with this policy.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate_with_public_existing(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        admission: ResidentAdmission,
        poison: Arc<AtomicBool>,
        limits: &wgpu::Limits,
        tile: usize,
        cancel: &AtomicBool,
        public_existing_fill: bool,
    ) -> Result<Self> {
        Self::allocate_with_policies(
            executor,
            graph,
            sequence,
            arena,
            admission,
            poison,
            limits,
            tile,
            cancel,
            public_existing_fill,
            false,
        )
    }
    /// Both public policies are captured by the model Config/Plan. Live-member
    /// reduction uses the same geometry, owner inventory and INFO dispatches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate_with_policies(
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        sequence: SequenceKey,
        arena: &mut PrivateArena,
        admission: ResidentAdmission,
        poison: Arc<AtomicBool>,
        limits: &wgpu::Limits,
        tile: usize,
        cancel: &AtomicBool,
        public_existing_fill: bool,
        live_member_reduction: bool,
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
            live_member_reduction_policy(
                live_member_reduction,
                executor.packed_profile,
                executor.bank_slots,
                tile,
                false,
                executor.resident_router_reserved,
                false,
            )?;
            if live_member_reduction {
                if !executor.target_rows_reserved || !arena.public_owned_zero_audit_enabled() {
                    return Err(err("live-member public target construction"));
                }
                executor.check_target_rows_policy()?;
            }
            public_existing_policy(
                public_existing_fill,
                executor.packed_profile,
                executor.bank_slots,
                tile,
                false,
                executor.resident_router_reserved,
            )?;
            if public_existing_fill
                && (executor.adaptive.is_some()
                    || executor.secondary.is_some()
                    || executor.secondary_frequency.is_some()
                    || executor.resident_banks_frozen)
            {
                return Err(err("public existing expert demand policy"));
            }
            let capacity = executor.bank_slots;
            if executor
                .resident_admission
                .as_ref()
                .is_none_or(|actual| !Arc::ptr_eq(&actual.ledger, &admission.ledger))
            {
                return Err(err("native routed actual admission binding"));
            }
            let mut layout = Layout::new(graph, capacity, tile)?;
            if live_member_reduction {
                for geometry in &mut layout.geometries {
                    *geometry = geometry.with_live_member_reduction().map_err(err)?;
                }
            }
            layout.limits(limits)?;
            let metadata = StagedSlots::bounded_native_metadata_bytes(capacity)
                .map_err(|_| Error::LimitExceeded("native routed candidate metadata"))?;
            if metadata
                .checked_add(
                    std::mem::size_of::<Pending>()
                        + 6 * std::mem::size_of::<Handle>()
                        + std::mem::size_of::<Last>(),
                )
                .is_none_or(|n| n > METADATA_PER_LAYER)
                || 48 * METADATA_PER_LAYER + 512 * 1024 > HOST_PRIVATE as usize
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
                resident_router: None,
                batch_router: None,
                prepared_verify3: None,
                public_existing_fill,
                public_four_members: false,
                public_fill_counters: PublicFillCounters::default(),
                frozen_sources: None,
                frozen_source_counters: frozen_sources::Counters::default(),
                frozen_staging: None,
                frozen_gu_integer: None,
                frozen_down_integer: None,
            };
            out.validate_ids(arena)?;
            u::cancel(cancel)?;
            Ok(out)
        })
    }
    pub(crate) fn frozen_staging_owner_ids(&self) -> Option<[BufferId; 3]> {
        self.frozen_staging.as_ref().map(|ids| **ids)
    }
    /// Construction-only owners; no source attachment, routes or use authority.
    pub(crate) fn reserve_frozen_host_staging(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        // MAIN is already locked by allocate_model. The construction token
        // proves its origin and active sequence; healthy()/active() would
        // reacquire that lock. Keep the independent arena/owner checks here.
        self.validate_ids(arena)?;
        if self.poison.load(Ordering::Acquire)
            || self.frozen_staging.is_some()
            || self.frozen_gu_integer.is_some()
            || self.frozen_down_integer.is_some()
            || *self.policy.chunk != 0
            || self.frozen_sources.is_some()
            || !self.frozen_host_batch_router_reserved()
            || self.batch_router_active()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.prepared_verify3.is_some()
            || !arena.public_owned_zero_audit_enabled()
            || arena
                .charged_bytes()
                .checked_add(FROZEN_STAGING_MAIN_BYTES as usize)
                .is_none_or(|n| n > arena.budget_bytes())
        {
            return Err(err("frozen host staging construction rejected"));
        }
        let words = (FROZEN_STAGING_OWNER_BYTES / 4) as usize;
        let first = arena.allocate_zeroed(words).map_err(u::fault)?;
        let mut ids = Zeroizing::new([first; 3]);
        for id in &mut ids[1..] {
            *id = arena.allocate_zeroed(words).map_err(u::fault)?;
        }
        self.frozen_staging = Some(ids);
        self.validate_ids(arena)
    }
    pub(crate) fn frozen_gu_integer_owner_id(&self) -> Option<BufferId> {
        self.frozen_gu_integer
    }
    /// Reserve one complete private page using the already-held MAIN origin.
    /// This establishes no source or execution authority.
    pub(crate) fn reserve_frozen_gu_integer(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        self.validate_ids(arena)?;
        if self.poison.load(Ordering::Acquire)
            || self.frozen_gu_integer.is_some()
            || self.frozen_staging.is_some()
            || *self.policy.chunk != 0
            || self.policy.phase != Phase::Idle
            || self.frozen_sources.is_some()
            || !self.frozen_host_batch_router_reserved()
            || self.batch_router_active()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.prepared_verify3.is_some()
            || self.live_member_reduction_enabled()
            || !arena.public_owned_zero_audit_enabled()
            || arena
                .charged_bytes()
                .checked_add(FROZEN_GU_INTEGER_MAIN_BYTES as usize)
                .is_none_or(|n| n > arena.budget_bytes())
        {
            return Err(err("frozen GU integer construction rejected"));
        }
        self.frozen_gu_integer = Some(
            arena
                .allocate_zeroed(FROZEN_GU_INTEGER_SCRATCH_WORDS)
                .map_err(u::fault)?,
        );
        self.validate_ids(arena)
    }
    pub(crate) fn frozen_down_integer_owner_id(&self) -> Option<BufferId> {
        self.frozen_down_integer
    }
    /// Reserve one complete private page using the already-held MAIN origin.
    /// This establishes no source or execution authority.
    pub(crate) fn reserve_frozen_down_integer(
        &mut self,
        arena: &mut PrivateArena,
        allocation: &ModelConstructionAllocation<'_>,
    ) -> Result<()> {
        allocation.check(&self.admission)?;
        self.validate_ids(arena)?;
        if self.poison.load(Ordering::Acquire)
            || self.frozen_down_integer.is_some()
            || self.frozen_staging.is_some()
            || *self.policy.chunk != 0
            || self.policy.phase != Phase::Idle
            || self.frozen_sources.is_some()
            || !self.frozen_host_batch_router_reserved()
            || self.batch_router_active()
            || self.hybrid.is_some()
            || self.resident_router.is_some()
            || self.prepared_verify3.is_some()
            || self.live_member_reduction_enabled()
            || !arena.public_owned_zero_audit_enabled()
            || arena
                .charged_bytes()
                .checked_add(FROZEN_DOWN_INTEGER_MAIN_BYTES as usize)
                .is_none_or(|n| n > arena.budget_bytes())
        {
            return Err(err("frozen DOWN integer construction rejected"));
        }
        self.frozen_down_integer = Some(
            arena
                .allocate_zeroed(FROZEN_DOWN_INTEGER_SCRATCH_WORDS)
                .map_err(u::fault)?,
        );
        self.validate_ids(arena)
    }
    fn healthy(&self, arena: &PrivateArena) -> Result<()> {
        if self.poison.load(Ordering::Acquire) {
            return Err(err("native routed sequence poisoned"));
        }
        self.admission.active()?;
        self.validate_ids(arena)
    }
    /// Read-only completed family gate for the construction-fixed public warm
    /// replay. No registry, resident-route or hybrid permit may survive it.
    pub(crate) fn validate_public_warm_boundary(
        &self,
        executor: &GpuExecutor<'_>,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
    ) -> Result<()> {
        self.healthy(arena)?;
        self.policy.completed_boundary()?;
        executor.check_resident_admission(&self.admission)?;
        executor.check_client(arena.client().map_err(u::fault)?)?;
        graph.verify_unchanged()?;
        if executor.batch_moe.is_some()
            || self.hybrid.is_some()
            || self.resident_router_active()
            || self.layout.tile != 8
            || graph.binding_id() != self.binding
            || !arena.public_owned_zero_audit_enabled()
        {
            return Err(err("native public warm routed boundary"));
        }
        Ok(())
    }
    fn validate_ids(&self, arena: &PrivateArena) -> Result<()> {
        let live = self.live_member_reduction_enabled();
        if self.layout.geometries.iter().any(|g| {
            g.live_member_reduction() != live
                || live && (g.tile() != 8 || g.gate_policy() != k::GatePolicy::Generic)
        }) || live
            && (self.layout.geometries.len() != 48
                || self.layout.tile != 8
                || self.layout.gate_policy != k::GatePolicy::Generic
                || self.hybrid.is_some()
                || self.resident_router.is_some()
                || self.prepared_verify3.is_some())
        {
            return Err(err("native live-member frozen workspace"));
        }
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
        if let Some(ids) = self.batch_router_ids() {
            for (i, &id) in ids.iter().enumerate() {
                if self.ids.contains(&id)
                    || ids[..i].contains(&id)
                    || self.hybrid.is_some()
                    || self.resident_router.is_some()
                    || arena.layout(id).map_err(u::fault)?
                        != BufferLayout::new(
                            [
                                48 * 512,
                                if self.four_member_router_reserved() {
                                    crate::resident_batch_router_gpu::four::AUDIT_WORDS
                                } else {
                                    crate::resident_batch_router_gpu::AUDIT_WORDS
                                },
                            ][i],
                        )
                        .map_err(u::fault)?
                {
                    return Err(err("public batch router complete owner layout"));
                }
            }
        }
        if let Some(ids) = &self.frozen_staging {
            for (i, &id) in ids.iter().enumerate() {
                if self.ids.contains(&id)
                    || ids[..i].contains(&id)
                    || self
                        .batch_router_ids()
                        .is_none_or(|router| router.contains(&id))
                    || self.hybrid.is_some()
                    || self.resident_router.is_some()
                    || self.prepared_verify3.is_some()
                    || arena.layout(id).map_err(u::fault)?
                        != BufferLayout::new((FROZEN_STAGING_OWNER_BYTES / 4) as usize)
                            .map_err(u::fault)?
                {
                    return Err(err("frozen host staging complete owner layout"));
                }
            }
        }
        if let Some(id) = self.frozen_gu_integer
            && (self.ids.contains(&id)
                || self.batch_router_ids().is_none_or(|ids| ids.contains(&id))
                || self.frozen_staging.is_some()
                || self.hybrid.is_some()
                || self.resident_router.is_some()
                || self.prepared_verify3.is_some()
                || self.live_member_reduction_enabled()
                || !arena.public_owned_zero_audit_enabled()
                || arena.layout(id).map_err(u::fault)?
                    != BufferLayout::new(FROZEN_GU_INTEGER_SCRATCH_WORDS).map_err(u::fault)?)
        {
            return Err(err("frozen GU integer complete owner layout"));
        }
        if let Some(id) = self.frozen_down_integer
            && (self.frozen_gu_integer == Some(id)
                || self.ids.contains(&id)
                || self.batch_router_ids().is_none_or(|ids| ids.contains(&id))
                || self.frozen_staging.is_some()
                || self.hybrid.is_some()
                || self.resident_router.is_some()
                || self.prepared_verify3.is_some()
                || self.live_member_reduction_enabled()
                || !arena.public_owned_zero_audit_enabled()
                || arena.layout(id).map_err(u::fault)?
                    != BufferLayout::new(FROZEN_DOWN_INTEGER_SCRATCH_WORDS).map_err(u::fault)?)
        {
            return Err(err("frozen DOWN integer complete owner layout"));
        }
        if let Some(id) = self.prepared_verify3
            && (self.hybrid.is_some()
                || self.resident_router.is_some()
                || self.ids.contains(&id)
                || arena.layout(id).map_err(u::fault)?
                    != BufferLayout::new(prepared_verify3::SCRATCH_WORDS).map_err(u::fault)?)
        {
            return Err(err("Verify3 prepared stale full owner"));
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
        if let Some(ids) = self.resident_router_ids() {
            for (i, &id) in ids.iter().enumerate() {
                if self.ids.contains(&id)
                    || ids[..i].contains(&id)
                    || self.hybrid.as_ref().is_some_and(|h| h.ids.contains(&id))
                    || arena.layout(id).map_err(u::fault)?
                        != BufferLayout::new(
                            [
                                48 * 512,
                                crate::resident_model::batch_io::RESIDENT_AUDIT_WORDS,
                            ][i],
                        )
                        .map_err(u::fault)?
                {
                    return Err(err("resident router stale owning pages"));
                }
            }
        }
        Ok(())
    }
    fn check(&self, e: &GpuExecutor<'_>, s: &Session, arena: &PrivateArena) -> Result<()> {
        self.healthy(arena)?;
        public_existing_policy(
            self.public_existing_fill,
            e.packed_profile,
            e.bank_slots,
            self.layout.tile,
            self.hybrid.is_some(),
            self.resident_router.is_some() || e.resident_router_reserved,
        )?;
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
        self.prepared_verify3.zeroize();
        self.public_existing_fill.zeroize();
        self.public_fill_counters = PublicFillCounters::default();
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
    deferred_router: Option<Zeroizing<u64>>,
    frozen_sources: Option<frozen_sources::Proof>,
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
            live_member_reduction_policy(
                w.live_member_reduction_enabled(),
                self.packed_profile,
                self.bank_slots,
                w.layout.tile,
                w.hybrid.is_some(),
                w.resident_router.is_some() || self.resident_router_reserved,
                w.prepared_verify3.is_some(),
            )?;
            live_member_reduction_count(w.live_member_reduction_enabled(), count)?;
            if w.live_member_reduction_enabled() {
                if !self.target_rows_reserved {
                    return Err(err("live-member target reservation changed"));
                }
                self.check_target_rows_policy()?;
            }
            public_existing_policy(
                w.public_existing_fill,
                self.packed_profile,
                self.bank_slots,
                w.layout.tile,
                w.hybrid.is_some(),
                w.resident_router.is_some() || self.resident_router_reserved,
            )?;
            w.validate_public_existing_chunk_count(self, graph, arena, count, 1)?;
            if w.prepared_verify3.is_some() {
                prepared_verify3::validate_count(
                    count,
                    1,
                    count.checked_mul(10).ok_or(Error::ArithmeticOverflow)?,
                )?;
            }
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
            w.begin_resident_router_token(self, graph, arena, start, count, cancel)?;
            w.begin_batch_router_chunk(arena, start, count)?;
            if w.frozen_host_sources_enabled()
                && (!(1..=if w.public_four_members_policy() { 4 } else { 3 }).contains(&count)
                    || !self.resident_banks_frozen
                    || self.target_frozen_host_source_identity
                        != w.frozen_sources.as_ref().map(|s| s.plan_identity())
                    || w.hybrid.is_some()
                    || w.resident_router.is_some()
                    || w.batch_router.is_some() && !w.frozen_host_batch_router_active()
                    || w.prepared_verify3.is_some())
            {
                return Err(err("frozen source chunk policy"));
            }
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
        if w.resident_router_active() || w.frozen_host_sources_enabled() {
            return Err(err(
                "resident router or frozen sources cannot use demand path",
            ));
        }
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
        live_member_reduction_policy(
            g.live_member_reduction(),
            self.packed_profile,
            self.bank_slots,
            g.tile(),
            w.hybrid.is_some(),
            w.resident_router.is_some() || self.resident_router_reserved,
            w.prepared_verify3.is_some(),
        )?;
        live_member_reduction_count(g.live_member_reduction(), routes.count())?;
        if g.live_member_reduction() {
            if !self.target_rows_reserved {
                return Err(err("live-member target reservation changed"));
            }
            self.check_target_rows_policy()?;
        }
        public_existing_count(w.public_existing_fill, routes.count(), routes.wave_count())?;
        let weights = weights(graph, layer)?;
        if [weights.gate, weights.up, weights.down]
            .into_iter()
            .any(|t| !t.belongs_to(self.model))
        {
            return Err(err("native routed artifact binding"));
        }
        if w.prepared_verify3.is_some() {
            prepared_verify3::validate_geometry(g)?;
            prepared_verify3::validate_count(
                routes.count(),
                routes.wave_count(),
                routes
                    .wave(0)
                    .ok_or(Error::InvalidBufferLength)?
                    .protected_experts()
                    .len(),
            )?;
            if self.packed_profile != crate::expert_banks::OriginalPackedProfile::DaslabIq3
                || self.q3_integer_moe
                || self.adaptive.is_some()
                || self.secondary.is_some()
                || self.secondary_frequency.is_some()
                || self.resident_banks_frozen
            {
                return Err(err("Verify3 prepared immutable source policy"));
            }
        }
        w.policy.issue(layer)?;
        let key = weights.gate.info().name().to_owned();
        if !self.banks.contains_key(&key) {
            let requests = ExpertBanks::allocation_requests_with_profile(
                weights,
                self.bank_slots,
                self.packed_profile,
            )?;
            let bank = w.admission.allocate(&requests, || {
                ExpertBanks::new_with_profile(
                    &self.client,
                    weights,
                    self.bank_slots,
                    self.packed_profile,
                )
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
        if bank.profile() != self.packed_profile {
            return Err(err("native routed original packed profile mismatch"));
        }
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
        u::zero_many(arena, &*w.ids)?;
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
        if let Some(frequency) = &mut self.secondary_frequency {
            frequency.record(
                layer,
                experts[..routes.count() * 10].iter().map(|&id| id as usize),
            )?;
        }
        if let Some(cache) = &mut self.adaptive {
            if cache.ready {
                return Err(err("adaptive prefill after activation"));
            }
            for token in 0..routes.count() {
                cache.record(
                    layer,
                    s.stamp
                        .start
                        .checked_add(token as u64)
                        .and_then(|n| n.checked_add(1))
                        .ok_or(Error::ArithmeticOverflow)?,
                    experts[token * 10..(token + 1) * 10]
                        .iter()
                        .map(|&id| id as usize),
                )?;
            }
        }
        let mut final_last = None;
        let mut stats = expert_banks::FillStats::default();
        let mut fill_counters = PublicFillCounters::default();
        let classify = self.packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3
            && w.layout.capacity == crate::daslab_profile::BANK_SLOTS
            && (1..=3).contains(&routes.count())
            && routes.wave_count() == 1;
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
            let pending = if let Some(cache) = &self.adaptive {
                staged.plan_scored(&bank.slots, &selected[..union.len()], &|id| {
                    cache.score(layer, id)
                })
            } else {
                staged.plan(&bank.slots, &selected[..union.len()])
            }
            .map_err(|_| err("native routed real pending plan"))?;
            let pin = Pin::new(wave, &pending, &staged, bank)?;
            if classify {
                fill_counters =
                    fill_counters.add(classify_public_fills(&bank.slots, &pending.uploads)?)?;
            }
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
                    let mut writes = Vec::new();
                    if w.public_existing_fill {
                        writes
                            .try_reserve_exact(window.len() * 3)
                            .map_err(|_| Error::LimitExceeded("public expert write metadata"))?;
                    }
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
                            let (bytes, read, upload) = if w.public_existing_fill {
                                let (view, data, read) =
                                    expert_banks::prepare_public_existing(b, t, e, slot, cancel)?;
                                let bytes = data.len() as u64;
                                writes.push((view, data));
                                (bytes, read, 0)
                            } else {
                                expert_banks::fill(&self.client, b, t, e, slot, cancel)?
                            };
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
                    if w.public_existing_fill {
                        // Revalidate the unchanged pending identity immediately
                        // before any write; ordinary PUBLIC server.write retains
                        // the original stream ordering and staging policy.
                        pin.validate()?;
                        if w.poison.load(Ordering::Acquire) {
                            return Err(err("native routed aborted source fill"));
                        }
                        let (bytes, upload) = expert_banks::write_public_existing(
                            &self.client,
                            writes,
                            charged,
                            cancel,
                        )?;
                        if bytes != add.bytes {
                            return Err(Error::InvalidBufferLength);
                        }
                        add.upload_ns = upload;
                    }
                    Ok(add)
                })?;
                if w.public_existing_fill {
                    fill_counters = fill_counters.add(PublicFillCounters {
                        direct_windows: 1,
                        direct_components: (window.len() * 3) as u64,
                        direct_bytes: add.bytes,
                        ..PublicFillCounters::default()
                    })?;
                }
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
            if let Some(owner) = w.prepared_verify3 {
                let banks = [bank.gate(), bank.up(), bank.down()];
                let mut shards: Vec<[Handle; 3]> = Vec::with_capacity(2);
                for segment in 0..2 {
                    let triple = banks.map(|b| {
                        b.segments()
                            .get(segment)
                            .map(|s| s.handle().clone())
                            .ok_or(Error::InvalidBufferLength)
                    });
                    let [gate, up, down] = triple;
                    shards.push([gate?, up?, down?]);
                }
                let banks: [[Handle; 3]; 2] =
                    shards.try_into().map_err(|_| Error::InvalidBufferLength)?;
                let owners = ids
                    .each_ref()
                    .map(|&id| arena.handle(id).map_err(u::fault).cloned());
                let mut handles = Vec::with_capacity(7);
                for handle in owners {
                    handles.push(handle?);
                }
                let owners: [Handle; 7] =
                    handles.try_into().map_err(|_| Error::InvalidBufferLength)?;
                prepared_verify3::validate_layout(
                    g,
                    &spans,
                    words,
                    arena.layout(owner).map_err(u::fault)?.owned_words,
                    routes.count(),
                    map.groups(),
                )?;
                u::zero(arena, owner)?;
                prepared_verify3::launch_pair(
                    arena.client().map_err(u::fault)?,
                    g,
                    &spans,
                    banks,
                    owners,
                    arena.handle(owner).map_err(u::fault)?.clone(),
                    arena.handle(s.stamp.fault).map_err(u::fault)?.clone(),
                    routes.count(),
                    map.groups(),
                )?;
            } else {
                launch_wave(arena, bank, g, &spans, &ids, map.groups())?;
            }
            launch_finite_for_count(arena, g, &spans, &ids, routes.count())?;
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
        launch_reduce(arena, g, &spans, &ids, routes.count())?;
        launch_finite_for_count(arena, g, &spans, &ids, routes.count())?;
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
            .is_none_or(|n| n > METADATA_PER_LAYER)
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
            deferred_router: None,
            frozen_sources: None,
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
        w.public_fill_counters = w.public_fill_counters.add(fill_counters)?;
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
    physical_with_packet(arena, bank, g, ids, fault, w, None)
}

/// The deferred mapper also lends the shared original packet. Include it in
/// the same complete physical disjointness proof as the consumed headers,
/// new table/audit owners and six original packed bank pages.
fn physical_with_packet(
    arena: &PrivateArena,
    bank: &ExpertBanks,
    g: k::Geometry,
    ids: &[BufferId; 7],
    fault: BufferId,
    w: &Workspace,
    packet: Option<BufferId>,
) -> Result<()> {
    if packet.is_some() && !w.resident_router_active() && !w.batch_router_active() {
        return Err(err("resident mapper packet requires sealed authorization"));
    }
    let client = arena.client().map_err(u::fault)?;
    bank.check_client(client)?;
    let mut all = Zeroizing::new(Vec::with_capacity(if w.resident_router.is_some() {
        13
    } else {
        10
    }));
    all.extend_from_slice(ids);
    all.push(fault);
    if let Some(h) = &w.hybrid {
        all.extend_from_slice(&*h.ids);
    }
    if let Some(ids) = w.resident_router_ids() {
        all.extend_from_slice(&ids);
    }
    if let Some(ids) = w.batch_router_ids() {
        all.extend_from_slice(&ids);
    }
    if let Some(owner) = w.prepared_verify3 {
        all.push(owner);
    }
    if let Some(packet) = packet {
        all.push(packet);
    }
    if let Some(staging) = w.frozen_staging_owner_ids() {
        all.extend_from_slice(&staging);
    }
    if let Some(owner) = w.frozen_gu_integer_owner_id() {
        all.push(owner);
    }
    if let Some(owner) = w.frozen_down_integer_owner_id() {
        all.push(owner);
    }
    let banks = [bank.gate(), bank.up(), bank.down()];
    let count = original_shard_count(w.layout.capacity, banks.map(|b| b.segments().len()))?;
    if bank.slots.capacity() != w.layout.capacity
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
    let extent_count = all
        .len()
        .checked_add(public.len())
        .ok_or(Error::ArithmeticOverflow)?;
    if extent_count
        > if w.frozen_staging.is_some()
            || w.frozen_gu_integer.is_some()
            || w.frozen_down_integer.is_some()
        {
            20
        } else if w.resident_router.is_some() || w.batch_router.is_some() {
            19
        } else {
            16
        }
    {
        return Err(err("native routed metadata cardinality"));
    }
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(extent_count)
        .map_err(|_| Error::LimitExceeded("native routed physical metadata"))?;
    let mut extents = Zeroizing::new([(0u64, 0u64); 20]);
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
        extents[handles.len()] = (n, n);
        handles.push(h.clone());
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
        let bound = main_admission::page_bytes(n).map_err(admission_error)?;
        extents[handles.len()] = (n, bound);
        handles.push(h.clone());
    }
    // Inspect the same actual full resources once as an ordered retained batch.
    // No resource identity is cached across this original proof boundary.
    let handle_bytes = handles
        .capacity()
        .checked_mul(std::mem::size_of::<Handle>())
        .ok_or(Error::ArithmeticOverflow)?;
    if handle_bytes > 4096 {
        return Err(Error::LimitExceeded("native physical handle metadata"));
    }
    let metadata = u::resources(client, handles)?;
    if metadata.len() != extent_count {
        return Err(Error::InvalidBufferLength);
    }
    let metadata_bytes = metadata
        .capacity()
        .checked_mul(std::mem::size_of_val(&metadata[0]))
        .and_then(|n| n.checked_add(handle_bytes + std::mem::size_of_val(&*extents)))
        .ok_or(Error::ArithmeticOverflow)?;
    if metadata_bytes > 4096 {
        return Err(Error::LimitExceeded("native routed physical metadata"));
    }
    for (i, r) in metadata.iter().enumerate() {
        let v = r.resource();
        let (n, bound) = extents[i];
        check_page(v.offset, v.size, v.buffer.size(), n, bound)?;
        if metadata[..i]
            .iter()
            .any(|p| p.resource().buffer == v.buffer)
        {
            return Err(err("native routed physical alias"));
        }
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
fn original_shard_count(capacity: usize, actual: [usize; 3]) -> Result<usize> {
    if !crate::bank_slots::routed_capacity(capacity) {
        return Err(err("native routed bank capacity"));
    }
    let count = crate::bank_slots::segmented_capacities(capacity).map_or(1, |parts| parts.len());
    if actual != [count; 3] {
        return Err(err("native routed bank shard inventory"));
    }
    Ok(count)
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
        macro_rules! launch_pair {
            ($gate_up:ident, $down:ident) => {
                unsafe {
                    k::$gate_up::launch_unchecked::<WgpuRuntime>(
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
                    k::$down::launch_unchecked::<WgpuRuntime>(
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
            };
        }
        if g.live_member_reduction() {
            launch_pair!(grouped_gate_up_live, grouped_down_live);
        } else {
            launch_pair!(grouped_gate_up, grouped_down);
        }
    }
    Ok(())
}
fn launch_reduce(
    arena: &PrivateArena,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
    count: usize,
) -> Result<()> {
    let grid = g.reduce_grid_for_count(count).map_err(err)?;
    let c = arena.client().map_err(u::fault)?;
    unsafe {
        k::grouped_reduce::launch_unchecked::<WgpuRuntime>(
            c,
            grid,
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
    // Preserve the existing count1/router callers' public-capacity grid.
    launch_finite_for_count(arena, g, s, ids, g.member_capacity())
}
fn launch_finite_for_count(
    arena: &PrivateArena,
    g: k::Geometry,
    s: &k::Spans,
    ids: &[BufferId; 7],
    count: usize,
) -> Result<()> {
    let grid = g.finite_grid_for_count(count).map_err(err)?;
    let c = arena.client().map_err(u::fault)?;
    unsafe {
        k::grouped_finite::launch_unchecked::<WgpuRuntime>(
            c,
            grid,
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
        if p.deferred_router.is_some()
            || !p.armed
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
        if let Some(proof) = &p.frozen_sources {
            let source = self
                .frozen_sources
                .as_ref()
                .ok_or(err("frozen source workspace lost"))?;
            if !Arc::ptr_eq(source, &proof.source)
                || p.staged.is_some()
                || p.last.is_some()
                || p.count1.is_some()
                || *p.waves != 1
                || !(1..=if self.public_four_members_policy() {
                    4
                } else {
                    3
                })
                    .contains(&s.stamp.count)
                || proof.count() != s.stamp.count
                || !e.resident_banks_frozen
                || e.target_frozen_host_source_identity != Some(source.plan_identity())
            {
                return Err(err("frozen source pending policy"));
            }
            proof.validate(*p.layer, bank)?;
            let g = source.geometry(*p.layer, bank)?;
            g.validate_count(s.stamp.count).map_err(err)?;
            let ids = self.ids(p.views.ids[0], p.views.ids[1]);
            let v = &p.views.offsets;
            let spans = k::Spans::new([0, v[0], v[1], 0, g.ff(), 0, g.hidden(), 0, v[2], v[3], 0]);
            let mut words = Zeroizing::new([0usize; 7]);
            for (i, &id) in ids.iter().enumerate() {
                words[i] = u::words(arena, id)?;
            }
            spans.validate(g, *words).map_err(err)?;
            if self.frozen_host_batch_router_active() != proof.router_owners.is_some() {
                return Err(err("frozen source router pending mode"));
            }
            return match &proof.router_owners {
                Some(router) => source.physical_with_router(
                    arena,
                    *p.layer,
                    bank,
                    &ids,
                    s.stamp.fault,
                    Some(&p.pins),
                    router,
                    self.frozen_staging_owner_ids().as_ref(),
                    self.frozen_gu_integer_owner_id(),
                    self.frozen_down_integer_owner_id(),
                ),
                None => source.physical(arena, *p.layer, bank, &ids, s.stamp.fault, Some(&p.pins)),
            };
        }
        if self.frozen_sources.is_some() {
            return Err(err("frozen source pending receipt required"));
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
            if s.stamp.count != 1
                || *p.waves != 1
                || self.cpu_miss_cap() != Some(proof.cap())
                || proof.gpu_limit() != e.gpu_expert_limit
                || proof.empty_slot_fill() != e.q3_empty_slot_fill
            {
                return Err(err("native count1 foreign pending policy"));
            }
            proof.validate_source(&bank.slots, staged, last, true)?;
            proof.validate_secondary(e.secondary.as_ref(), *p.layer, bank)?;
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
                let count = p.count1.as_ref().map_or(0, |proof| proof.cpu_miss_count());
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
            if let Some(proof) = &p.frozen_sources {
                let counters = self.workspace.frozen_source_counters.add(proof, *p.layer)?;
                u::cancel(cancel)?;
                self.workspace.policy.commit(*p.layer)?;
                self.workspace.frozen_source_counters = counters;
                p.armed = false;
                return Ok(());
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
            if let (Some(cache), Some(proof)) = (&mut self.executor.adaptive, &p.count1) {
                if proof.cap() != 10 || proof.gpu_limit() != 10 {
                    return Err(err("adaptive accepted miss policy"));
                }
                cache.record(
                    *p.layer,
                    p.stamp
                        .start
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?,
                    proof.ids().iter().copied(),
                )?;
            }
            if let Some(trace) = &mut self.executor.oracle_trace {
                let proof = p
                    .count1
                    .as_ref()
                    .ok_or(err("public oracle accepted count1 owner"))?;
                if proof.cap() != 10 || proof.gpu_limit() != 10 {
                    return Err(err("public oracle accepted miss policy"));
                }
                trace.record(
                    *p.layer,
                    p.stamp
                        .start
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?,
                    proof.ids(),
                )?;
            }
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
    #[test]
    fn public_existing_policy_is_explicit_daslab144_max8_count1_to3_only() {
        use expert_banks::OriginalPackedProfile as P;
        public_existing_policy(true, P::DaslabIq3, 144, 8, false, false).unwrap();
        for (profile, cap, tile, hybrid, router) in [
            (P::Q4, 144, 8, false, false),
            (P::UnslothQ3, 144, 8, false, false),
            (P::DaslabIq3, 160, 8, false, false),
            (P::DaslabIq3, 144, 1, false, false),
            (P::DaslabIq3, 144, 8, true, false),
            (P::DaslabIq3, 144, 8, false, true),
        ] {
            assert!(public_existing_policy(true, profile, cap, tile, hybrid, router).is_err());
            public_existing_policy(false, profile, cap, tile, hybrid, router).unwrap();
        }
        for count in 1..=3 {
            public_existing_count(true, count, 1).unwrap();
        }
        for (count, waves) in [(0, 1), (4, 1), (8, 1), (3, 0), (3, 2)] {
            assert!(public_existing_count(true, count, waves).is_err());
            public_existing_count(false, count, waves).unwrap();
        }
    }
    #[test]
    fn public_fill_counts_distinguish_first_touch_from_real_protected_victims() {
        let mut slots = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let initial: Vec<_> = (0..72).collect();
        let p = slots.plan(&initial).unwrap();
        assert_eq!(
            classify_public_fills(&slots, &p.uploads[..30])
                .unwrap()
                .first_touch_fills,
            30
        );
        slots.commit(p).unwrap();
        let p = slots.plan(&[0, 71, 400, 401]).unwrap();
        assert_eq!(p.uploads, [(400, 72), (401, 73)]);
        let counts = classify_public_fills(&slots, &p.uploads).unwrap();
        assert_eq!((counts.first_touch_fills, counts.evictions), (2, 0));
        drop(p);
        // Classification neither installs weights nor publishes slot metadata.
        assert_eq!(slots.resident_expert(72).unwrap(), None);
        let all: Vec<_> = (0..144).collect();
        let p = slots.plan(&all).unwrap();
        slots.commit(p).unwrap();
        let p = slots.plan(&[0, 71, 72, 143, 400, 401]).unwrap();
        let counts = classify_public_fills(&slots, &p.uploads).unwrap();
        assert_eq!((counts.first_touch_fills, counts.evictions), (0, 2));
        assert!(
            p.uploads
                .iter()
                .all(|&(_, s)| ![0, 71, 72, 143].contains(&s))
        );
        let hits = slots.plan(&[0, 71, 72, 143]).unwrap();
        assert!(
            classify_public_fills(&slots, &hits.uploads).unwrap() == PublicFillCounters::default()
        );
        for uploads in [
            &[(0, 0)][..],
            &[(400, 144)][..],
            &[(512, 1)][..],
            &[(400, 1), (401, 1)][..],
            &[(400, 1), (400, 2)][..],
        ] {
            assert!(classify_public_fills(&slots, uploads).is_err());
        }
        assert!(classify_public_fills(&slots, &[(400, 1); 31]).is_err());
    }
    #[test]
    fn public_fill_counter_updates_are_checked_and_never_credit_sources() {
        let initial = PublicFillCounters {
            first_touch_fills: 10,
            direct_windows: 1,
            direct_components: 30,
            direct_bytes: 100,
            ..PublicFillCounters::default()
        };
        let next = initial
            .add(PublicFillCounters {
                evictions: 2,
                ..initial
            })
            .unwrap();
        assert_eq!(
            (next.first_touch_fills, next.evictions, next.direct_windows),
            (20, 2, 2)
        );
        assert_eq!((next.direct_components, next.direct_bytes), (60, 200));
        assert!(
            PublicFillCounters {
                direct_bytes: u64::MAX,
                ..initial
            }
            .add(initial)
            .is_err()
        );
        assert!(
            PublicFillCounters {
                evictions: u64::MAX,
                ..initial
            }
            .add(PublicFillCounters {
                evictions: 1,
                ..PublicFillCounters::default()
            })
            .is_err()
        );
    }
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
    #[test]
    fn actual_original_shard_inventory_matches_each_admitted_profile() {
        for (capacity, count) in [(16, 1), (72, 1), (96, 2), (128, 2), (144, 2), (160, 3)] {
            assert_eq!(original_shard_count(capacity, [count; 3]).unwrap(), count);
            for projection in 0..3 {
                let mut short = [count; 3];
                short[projection] -= 1;
                assert!(original_shard_count(capacity, short).is_err());
                let mut extra = [count; 3];
                extra[projection] += 1;
                assert!(original_shard_count(capacity, extra).is_err());
            }
        }
        for capacity in [0, 9, 73, 145, 159, 161, usize::MAX] {
            assert!(original_shard_count(capacity, [3; 3]).is_err());
        }
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
        for capacity in [16, 64, 96, 128, 144, 160] {
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
        for capacity in [10, 16, 64, 72, 96, 128, 144, 160] {
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
        for cap in [16, 64, 72, 96, 128, 144, 160] {
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
            deferred_router: None,
            frozen_sources: None,
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

#[cfg(test)]
mod live_member_reduction_tests {
    use super::*;
    use crate::expert_banks::OriginalPackedProfile;

    #[test]
    fn live_member_policy_is_exact_and_default_preserves_old_profiles() {
        live_member_reduction_policy(
            true,
            OriginalPackedProfile::DaslabIq3,
            144,
            8,
            false,
            false,
            false,
        )
        .unwrap();
        for (profile, capacity, tile, hybrid, router, prepared) in [
            (OriginalPackedProfile::Q4, 144, 8, false, false, false),
            (
                OriginalPackedProfile::UnslothQ3,
                144,
                8,
                false,
                false,
                false,
            ),
            (
                OriginalPackedProfile::DaslabIq3,
                160,
                8,
                false,
                false,
                false,
            ),
            (
                OriginalPackedProfile::DaslabIq3,
                144,
                4,
                false,
                false,
                false,
            ),
            (
                OriginalPackedProfile::DaslabIq3,
                144,
                1,
                false,
                false,
                false,
            ),
            (OriginalPackedProfile::DaslabIq3, 144, 8, true, false, false),
            (OriginalPackedProfile::DaslabIq3, 144, 8, false, true, false),
            (OriginalPackedProfile::DaslabIq3, 144, 8, false, false, true),
        ] {
            assert!(
                live_member_reduction_policy(
                    true, profile, capacity, tile, hybrid, router, prepared
                )
                .is_err()
            );
            live_member_reduction_policy(false, profile, capacity, tile, hybrid, router, prepared)
                .unwrap();
        }
        for count in 1..=3 {
            live_member_reduction_count(true, count).unwrap();
        }
        for count in [0, 4, 8, 9, usize::MAX] {
            assert!(live_member_reduction_count(true, count).is_err());
            live_member_reduction_count(false, count).unwrap();
        }
    }

    #[test]
    fn all_actual_das_geometries_preserve_grids_and_full_original_owner_spans() {
        for layer in 0..48 {
            let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
            let projections = [
                k::Projection::new(gate, 2560, 640, 512).unwrap(),
                k::Projection::new(up, 2560, 640, 512).unwrap(),
                k::Projection::new(down, 640, 2560, 512).unwrap(),
            ];
            let old = k::Geometry::new(projections, 8, 144).unwrap();
            let live = old.with_live_member_reduction().unwrap();
            assert!(!old.live_member_reduction());
            assert!(live.live_member_reduction());
            assert_eq!(old.tile(), live.tile());
            assert_eq!(old.capacity(), live.capacity());
            assert_eq!(old.status_words(), live.status_words());
            assert_eq!(old.projections(), live.projections());
            for count in 1..=3 {
                for groups in [10, count * 10] {
                    for down in [false, true] {
                        let CubeCount::Static(x, y, z) = old.grid(groups, down).unwrap() else {
                            panic!()
                        };
                        let CubeCount::Static(lx, ly, lz) = live.grid(groups, down).unwrap() else {
                            panic!()
                        };
                        assert_eq!((x, y, z), (lx, ly, lz));
                    }
                }
                for (a, b) in [
                    (
                        old.reduce_grid_for_count(count).unwrap(),
                        live.reduce_grid_for_count(count).unwrap(),
                    ),
                    (
                        old.finite_grid_for_count(count).unwrap(),
                        live.finite_grid_for_count(count).unwrap(),
                    ),
                ] {
                    let CubeCount::Static(x, y, z) = a else {
                        panic!()
                    };
                    let CubeCount::Static(lx, ly, lz) = b else {
                        panic!()
                    };
                    assert_eq!((x, y, z), (lx, ly, lz));
                }
            }
            for first in [0, 72] {
                let words = projections.map(|p| p.segment_words(72).unwrap());
                old.validate_shard(first, 72, words).unwrap();
                live.validate_shard(first, 72, words).unwrap();
                let mut short = words;
                short[2] -= 1;
                assert!(live.validate_shard(first, 72, short).is_err());
            }
        }
        assert_eq!(Workspace::runtime_io_increment(), IO_INCREMENT);
        assert_eq!(PRIVATE_BYTES, 1_409_024);
        assert_eq!(HOST_PRIVATE, 2 * 1024 * 1024);
    }
}

#[cfg(test)]
#[path = "resident_moe_four_member_tests.rs"]
mod four_member_tests;
