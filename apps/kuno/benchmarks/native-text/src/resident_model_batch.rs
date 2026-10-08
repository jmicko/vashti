//! Actual layer-major MAX8 laboratory composition. It reuses the original
//! committed state/cache/banks and publishes only after whole-model acceptance.
//! Source-only until parent completes physical whole-model validation. No
//! default/API/catalog change, replay, fallback or private checkpoint export.
use super::*;
use crate::{
    executor::{GpuExecutor, NativeProjectionTile, batch_moe},
    quant_matvec::matmul::{BatchSpans, MAX_BATCH},
    resident_batch_projection::{BatchedResidentProjector, ResidentBatchProjection},
    resident_gdn::{BoundProjection, batch as gdn},
    resident_gr::batch as gr,
    resident_qsa::batch as qsa,
    token_waves,
};
use kuno_text::cpu::Matrix;

const OLD_MAIN: u64 = 785_186_816;
const NEW_MAIN: u64 = 33_062_912;
const TOTAL_MAIN: u64 = OLD_MAIN + NEW_MAIN;
const PUBLIC_MAIN: u64 = 25_165_824;
const HOST_CAP: usize = 12 * 1024 * 1024;
pub(crate) const ORIGINAL_BANK_HOST: usize = 640 * 1024;
const HOST_COMPONENTS: usize = 10_551_296 + ORIGINAL_BANK_HOST;
const META_CAP: usize = 256 * 1024;
const LAYERS: usize = 48;
const D: usize = 2560;
const HC: usize = 4;
const ROUTER: usize = 512;
const RESIDUAL_WORDS: usize = MAX_BATCH * HC * D;
const BLOCK_WORDS: usize = MAX_BATCH * D;
const HYBRID_MAIN: u64 = 163_840;
const HYBRID_HOST: usize = 1024 * 1024;
const HYBRID_RUNTIME: u64 = 1024 * 1024;
// Two extra scratch clears per layer; replacement merge/fold plus the new
// owning packet fold have zero net INFO growth. Reserve128 rather than96.
// Count full INFO source/target/inspection ownership, both bridge upload pages
// with duplicate inspection, packet readback and duplicated slab padding.
const HYBRID_IO_DERIVED: u64 = 128 * 2048 + 4 * HYBRID_MAIN + 2 * 32768 + 4 * 8192;
const _: () = {
    assert!(HYBRID_IO_DERIVED <= HYBRID_RUNTIME);
};

fn checked_cpu_miss_total(current: u64, completed_chunk: u64) -> Result<Zeroizing<u64>> {
    if completed_chunk > (LAYERS * 10) as u64 {
        return Err(error());
    }
    current
        .checked_add(completed_chunk)
        .map(Zeroizing::new)
        .ok_or(Error::ArithmeticOverflow)
}

fn error() -> Error {
    Error::InvalidFormat("native model chunk rejected")
}

/// Not Clone/Copy and constructible only after the model's completed boundary
/// validation below. Public policy setters borrow it or immediately consume it;
/// no caller can construct or retain a copy of this completed-boundary proof.
pub(crate) struct CompletedNativeProjectionBoundary {
    _private: (),
}

/// Public fixed-fixture policy, selected only between completed transactions.
/// Runtime token counts never select a shader specialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum NativeDecodeKernels {
    #[default]
    Chunk,
    SingleMoe,
    SingleGdn,
    SingleAll,
    SingleQ4All,
}
impl NativeDecodeKernels {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "chunk" => Ok(Self::Chunk),
            "single-moe" => Ok(Self::SingleMoe),
            "single-gdn" => Ok(Self::SingleGdn),
            "single-all" => Ok(Self::SingleAll),
            "single-q4-all" => Ok(Self::SingleQ4All),
            _ => Err(error()),
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::SingleMoe => "single-moe",
            Self::SingleGdn => "single-gdn",
            Self::SingleAll => "single-all",
            Self::SingleQ4All => "single-q4-all",
        }
    }
    fn single_moe(self) -> bool {
        matches!(self, Self::SingleMoe | Self::SingleAll | Self::SingleQ4All)
    }
    fn recurrence(self) -> crate::gdn_chunk::RecurrencePolicy {
        if matches!(self, Self::SingleGdn | Self::SingleAll | Self::SingleQ4All) {
            crate::gdn_chunk::RecurrencePolicy::SingleStreaming
        } else {
            crate::gdn_chunk::RecurrencePolicy::Chunk
        }
    }
    fn validate_count(self, count: usize) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&count) || self != Self::Chunk && count != 1 {
            return Err(error());
        }
        Ok(())
    }
}
fn sum(pages: &[u64]) -> Result<u64> {
    pages.iter().try_fold(0_u64, |n, &p| {
        n.checked_add(p).ok_or(Error::ArithmeticOverflow)
    })
}

/// Constructed from the same already-selected public lab client/adapter limits.
/// No device is selected/created here; all component owners recheck that client.
#[derive(Clone)]
pub(crate) struct Config {
    client: ComputeClient<WgpuRuntime>,
    limits: wgpu::Limits,
    tile: usize,
    cpu_operator: Option<Arc<kuno_text::cpu_expert::ExpertOperator>>,
    cpu_gpu_overlap: bool,
}
impl Config {
    pub(crate) fn new(
        client: &ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        tile: usize,
    ) -> Result<Self> {
        validate_limits(limits, tile)?;
        Ok(Self {
            client: client.clone(),
            limits: limits.clone(),
            tile,
            cpu_operator: None,
            cpu_gpu_overlap: false,
        })
    }
    /// Explicit fixed-public laboratory reservation. Ordinary constructors keep
    /// their original owner inventory; reservation alone never places a rank.
    pub(crate) fn reserve_count1_hybrid(self) -> Result<Self> {
        self.reserve_count1_hybrid_workers(false)
    }
    pub(crate) fn reserve_count1_hybrid_workers(self, pooled: bool) -> Result<Self> {
        self.reserve_count1_hybrid_execution(pooled, false)
    }
    pub(crate) fn reserve_count1_hybrid_execution(
        self,
        pooled: bool,
        overlap: bool,
    ) -> Result<Self> {
        self.reserve_count1_hybrid_arithmetic(pooled, overlap, false, false, false, 16)
    }
    pub(crate) fn reserve_count1_hybrid_arithmetic(
        mut self,
        pooled: bool,
        overlap: bool,
        q8: bool,
        residual: bool,
        prefetch: bool,
        threads: usize,
    ) -> Result<Self> {
        use kuno_text::cpu_expert::{Backend, ExpertOperator, WorkerPlan};
        if self.cpu_operator.is_some()
            || self.tile != 8
            || ((overlap || q8) && !pooled)
            || (residual && !q8)
        {
            return Err(error());
        }
        let plan = WorkerPlan::physical(threads)?;
        let backend = if cfg!(feature = "cpu-fma") {
            Backend::Avx2Fma
        } else {
            Backend::Avx2
        };
        let operator = if pooled {
            ExpertOperator::new_pooled(plan, backend)?
        } else {
            ExpertOperator::new(plan, backend)?
        };
        let operator = if prefetch {
            operator.with_mapped_prefetch()
        } else {
            operator
        };
        self.cpu_operator = Some(Arc::new(if residual {
            operator.with_q8_residual()?
        } else if q8 {
            operator.with_q8_activations()?
        } else {
            operator
        }));
        self.cpu_gpu_overlap = overlap;
        Ok(self)
    }
}
fn validate_limits(l: &wgpu::Limits, tile: usize) -> Result<()> {
    if ![4, 8].contains(&tile)
        || l.max_compute_invocations_per_workgroup < 64
        || l.max_compute_workgroup_size_x < 64
        || l.max_compute_workgroups_per_dimension < 16384
        || l.max_compute_workgroup_storage_size < 8192
        || l.max_bind_groups < 1
        || l.max_storage_buffers_per_shader_stage < 11
        || l.max_bindings_per_bind_group < 12
        || l.max_uniform_buffer_binding_size < 512
        || l.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || l.max_buffer_size < 128 * 1024 * 1024
        || l.min_storage_buffer_offset_alignment == 0
        || l.min_uniform_buffer_offset_alignment == 0
        || l.min_storage_buffer_offset_alignment > 512
        || l.min_uniform_buffer_offset_alignment > 512
    {
        return Err(Error::LimitExceeded("native model device inventory"));
    }
    Ok(())
}

pub(super) struct Plan {
    pub(super) requests: Vec<u64>,
    pub(super) config: Config,
    pub(super) runtime: u64,
}
impl Plan {
    pub(super) fn new(
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
        executor: &GpuExecutor<'_>,
        config: Config,
        old: &[u64],
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        if !std::ptr::eq(client.info(), config.client.info())
            || sum(old)? != OLD_MAIN
            || old.len() != 1286
        {
            return Err(error());
        }
        validate_limits(&config.limits, config.tile)?;
        executor.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
        let g = graph.geometry();
        if g.layers != LAYERS
            || g.residual.width != D
            || g.residual.streams != HC
            || g.experts != ROUTER
            || g.experts_used != 10
            || g.max_context != 4096
        {
            return Err(error());
        }
        let mut requests = Vec::new();
        for (pages, expected) in [
            (
                gr::Workspace::allocation_requests(graph)?.to_vec(),
                1_998_848,
            ),
            (
                gdn::GdnBatchWorkspace::allocation_requests(graph)?.to_vec(),
                3_932_160,
            ),
            (
                qsa::QsaBatchWorkspace::allocation_requests(graph)?.to_vec(),
                17_825_792,
            ),
            (
                executor
                    .routed_moe_batch_allocation_requests(graph, config.tile)?
                    .to_vec(),
                1_409_024,
            ),
            (
                ple_batch::PleBatchWorkspace::allocation_requests(graph)?.to_vec(),
                6_258_688,
            ),
            (
                shared_batch::Workspace::allocation_requests(graph)?.to_vec(),
                262_144,
            ),
        ] {
            if sum(&pages)? != expected {
                return Err(error());
            }
            requests.extend(pages);
        }
        for words in [
            RESIDUAL_WORDS,
            RESIDUAL_WORDS,
            BLOCK_WORDS,
            BLOCK_WORDS,
            MAX_BATCH * ROUTER,
            MAX_BATCH * ROUTER + 1,
        ] {
            requests.push(BufferLayout::new(words).map_err(u::fault)?.owned_bytes as u64);
        }
        let hybrid = config.cpu_operator.is_some();
        if hybrid {
            requests.extend(batch_moe::Workspace::count1_hybrid_allocation_requests()?);
        }
        let extra = if hybrid { HYBRID_MAIN } else { 0 };
        if requests.len() != 90 + usize::from(hybrid) * 2
            || sum(&requests)? != NEW_MAIN + extra
            || OLD_MAIN.checked_add(sum(&requests)?) != Some(TOTAL_MAIN + extra)
        {
            return Err(error());
        }
        let head = graph.tensor("output.weight")?;
        let row = head.info().row_bytes();
        if row == 0 {
            return Err(error());
        }
        let tile_rows = (64_u64 * 1024 * 1024 / row).min(16384);
        if tile_rows == 0 {
            return Err(error());
        }
        let head_tiles = (g.vocabulary as u64).div_ceil(tile_rows);
        let runtime = runtime_bytes(head_tiles)?
            .checked_add(if hybrid { HYBRID_RUNTIME } else { 0 })
            .ok_or(Error::ArithmeticOverflow)?;
        if runtime > crate::executor::RESIDENT_RUNTIME_IO_MAX
            || HOST_COMPONENTS + if hybrid { HYBRID_HOST } else { 0 } > HOST_CAP
            || metadata_bytes() > META_CAP
        {
            return Err(Error::LimitExceeded("native model aggregate private plan"));
        }
        // All component/API-visible reservations remain independent of MAIN.
        if gdn::GdnBatchWorkspace::runtime_info_reservation(graph, true)? != 4_718_592
            || qsa::QsaBatchWorkspace::runtime_io_reservation(graph, true)? != 7_864_320
            || batch_moe::Workspace::runtime_io_increment() != 524288
            || ops::GUARDED_ROUTER_CAPACITY_BYTES > 8192
        {
            return Err(error());
        }
        Ok(Self {
            requests,
            config,
            runtime,
        })
    }
    pub(super) fn check_public(&self, graph: BoundGraph<'_, '_>, pages: u64) -> Result<()> {
        // prewarm() independently proves exact planned cache names/counts and
        // seals them; here the actual graph count/full-page SUM must match too.
        if graph.static_vectors().count() != 341 || pages != PUBLIC_MAIN {
            return Err(error());
        }
        let derived = (0..LAYERS).try_fold(0usize, |n, l| {
            Ok::<_, Error>(n + usize::from(graph.layer_kind(l)? == AttentionKind::Gdn))
        })?;
        if 341 + derived + 2 != 379 {
            return Err(error());
        }
        Ok(())
    }
}
fn runtime_bytes(head_tiles: u64) -> Result<u64> {
    // Audited all-stage envelope, counting source+target+inspection separately.
    // Real forced router/interior-MoE fences drain before another source window.
    // Keep the helper's conservative embedding eight-record INFO envelope.
    // These nine actual outer/scalar clears are a separate one-time category.
    const OUTER_CLEAR_INFO: u64 = 9 * 2048;
    const WITHOUT_HEAD: u64 = 61_751_296 + OUTER_CLEAR_INFO;
    let dispatches = head_tiles
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(Error::ArithmeticOverflow)?;
    let head = dispatches
        .checked_mul(2048)
        .and_then(|n| n.checked_add(2 * 1024 * 1024 + 65536))
        .ok_or(Error::ArithmeticOverflow)?;
    WITHOUT_HEAD
        .checked_add(head)
        .ok_or(Error::ArithmeticOverflow)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Purpose {
    Advance,
    LastLogits,
}
struct Outer {
    residual: [BufferId; 2],
    block: [BufferId; 2],
    router: BufferId,
    packet: BufferId,
}
impl Zeroize for Outer {
    fn zeroize(&mut self) {
        self.residual.zeroize();
        self.block.zeroize();
        self.router.zeroize();
        self.packet.zeroize();
    }
}
pub(super) struct Workspace {
    gr: gr::Workspace,
    gdn: gdn::GdnBatchWorkspace,
    qsa: qsa::QsaBatchWorkspace,
    moe: batch_moe::Workspace,
    ple: ple_batch::PleBatchWorkspace,
    shared: shared_batch::Workspace,
    outer: Zeroizing<Outer>,
    poison: Arc<AtomicBool>,
    dense_member_capacity: usize,
}
impl Workspace {
    fn decode_kernels(&self) -> Result<NativeDecodeKernels> {
        let single_moe = match self.moe.projection_tile() {
            1 => true,
            4 | 8 => false,
            _ => return Err(error()),
        };
        let single_gdn =
            self.gdn.recurrence_policy() == crate::gdn_chunk::RecurrencePolicy::SingleStreaming;
        let q4_block =
            self.moe.gate_policy() == crate::quant_matvec::grouped_moe::GatePolicy::Q4Block;
        Ok(
            match (single_moe, single_gdn, q4_block, self.dense_member_capacity) {
                (false, false, false, 8) => NativeDecodeKernels::Chunk,
                (true, false, false, 8) => NativeDecodeKernels::SingleMoe,
                (false, true, false, 8) => NativeDecodeKernels::SingleGdn,
                (true, true, false, 1) => NativeDecodeKernels::SingleAll,
                (true, true, true, 1) => NativeDecodeKernels::SingleQ4All,
                _ => return Err(error()),
            },
        )
    }
    pub(super) fn allocate(
        model: &mut ResidentModel<'_, '_>,
        executor: &GpuExecutor<'_>,
        sequence: SequenceKey,
        config: Config,
        allocation: &crate::executor::ModelConstructionAllocation<'_>,
    ) -> Result<Self> {
        allocation.check(&model.admission)?;
        let graph = model.graph;
        let poison = model.admission.sequence_poison();
        let cancel = AtomicBool::new(false); // Idle zero-state constructor only.
        model
            .admission
            .check_client(model.arena.client().map_err(u::fault)?)?;
        if !model.admission.transient_io()
            || !std::ptr::eq(
                config.client.info(),
                model.arena.client().map_err(u::fault)?.info(),
            )
            || model.public.full_page_bytes() != PUBLIC_MAIN
        {
            return Err(error());
        }
        let gr = gr::Workspace::allocate(graph, &mut model.arena)?;
        let gdn = gdn::GdnBatchWorkspace::allocate(
            graph,
            sequence,
            &mut model.arena,
            poison.clone(),
            &cancel,
        )?;
        let qsa =
            qsa::QsaBatchWorkspace::allocate(graph, sequence, &mut model.arena, poison.clone())?;
        let mut moe = batch_moe::Workspace::allocate(
            executor,
            graph,
            sequence,
            &mut model.arena,
            model.admission.clone(),
            poison.clone(),
            &config.limits,
            config.tile,
            &cancel,
        )?;
        if let Some(operator) = config.cpu_operator {
            moe.reserve_count1_hybrid(
                &mut model.arena,
                operator,
                config.cpu_gpu_overlap,
                allocation,
            )?;
        }
        let shared = shared_batch::Workspace::allocate(graph, &mut model.arena)?;
        let mut alloc = |n| model.arena.allocate_zeroed(n).map_err(u::fault);
        let outer = Zeroizing::new(Outer {
            residual: [alloc(RESIDUAL_WORDS)?, alloc(RESIDUAL_WORDS)?],
            block: [alloc(BLOCK_WORDS)?, alloc(BLOCK_WORDS)?],
            router: alloc(MAX_BATCH * ROUTER)?,
            packet: alloc(MAX_BATCH * ROUTER + 1)?,
        });
        let ple = ple_batch::PleBatchWorkspace::allocate(model, sequence, poison.clone(), &cancel)?;
        let result = Self {
            gr,
            gdn,
            qsa,
            moe,
            ple,
            shared,
            outer,
            poison,
            dense_member_capacity: 8,
        };
        result.prove_outer(model)?;
        Ok(result)
    }
    fn prove_outer(&self, model: &ResidentModel<'_, '_>) -> Result<()> {
        let mut handles = Vec::new(); // Public resource references, no tensor copies.
        for (id, words) in [
            (self.outer.residual[0], RESIDUAL_WORDS),
            (self.outer.residual[1], RESIDUAL_WORDS),
            (self.outer.block[0], BLOCK_WORDS),
            (self.outer.block[1], BLOCK_WORDS),
            (self.outer.router, MAX_BATCH * ROUTER),
            (self.outer.packet, MAX_BATCH * ROUTER + 1),
            (model.b.fault, 1),
            (model.b.logits, model.graph.geometry().vocabulary),
            (model.b.logit_packet, model.graph.geometry().vocabulary + 1),
        ] {
            handles.push(gr::full_private(&model.arena, id, words)?);
        }
        if let Some(ids) = self.moe.count1_hybrid_owner_ids() {
            for (id, words) in ids.into_iter().zip([10 * D, 4 + 80 * 3]) {
                handles.push(gr::full_private(&model.arena, id, words)?);
            }
        }
        gr::physical(&model.arena, &handles)?;
        model
            .admission
            .check_client(model.arena.client().map_err(u::fault)?)?;
        if !Arc::ptr_eq(&self.poison, &model.admission.sequence_poison())
            || self.poison.load(Ordering::Acquire)
        {
            return Err(error());
        }
        Ok(())
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        self.poison.store(true, Ordering::Release);
    }
}

struct Pending {
    gdn: Box<[Option<gdn::PendingGdnBatch>; LAYERS]>,
    qsa: Box<[Option<qsa::PendingQsaBatch>; LAYERS]>,
    moe: Box<[Option<batch_moe::Pending>; LAYERS]>,
    ple: Option<ple_batch::PendingPleBatch>,
}
fn pending_bytes() -> usize {
    std::mem::size_of::<[Option<gdn::PendingGdnBatch>; LAYERS]>()
        + std::mem::size_of::<[Option<qsa::PendingQsaBatch>; LAYERS]>()
        + std::mem::size_of::<[Option<batch_moe::Pending>; LAYERS]>()
        + std::mem::size_of::<Option<ple_batch::PendingPleBatch>>()
}
fn metadata_bytes() -> usize {
    // Actual family anchors retain a subset of their pending metadata. Charge a
    // second complete pending representation for every real GDN/QSA/PLE anchor,
    // plus the controller/session and bounded stack/capability headroom. MoE's
    // retained dynamic candidates and plans have their separate 2 MiB budget.
    pending_bytes()
        + 36 * std::mem::size_of::<gdn::PendingGdnBatch>()
        + 12 * std::mem::size_of::<qsa::PendingQsaBatch>()
        + std::mem::size_of::<ple_batch::PendingPleBatch>()
        + std::mem::size_of::<Workspace>()
        + std::mem::size_of::<batch_moe::Session>()
        + 8192
}
impl Pending {
    fn new() -> Result<Self> {
        if metadata_bytes() > META_CAP {
            return Err(Error::LimitExceeded("native model pending owner capacity"));
        }
        // Allocate zero/None before filling. Each opaque pending owns guarded
        // metadata and sticky-only Drop; no filled inline table is returned.
        Ok(Self {
            gdn: Box::new(std::array::from_fn(|_| None)),
            qsa: Box::new(std::array::from_fn(|_| None)),
            moe: Box::new(std::array::from_fn(|_| None)),
            ple: None,
        })
    }
}
struct Flight {
    poison: Arc<AtomicBool>,
    accepted: bool,
}
impl Drop for Flight {
    fn drop(&mut self) {
        if !self.accepted {
            self.poison.store(true, Ordering::Release);
        }
        self.accepted.zeroize();
    }
}

impl<'m, 'f> ResidentModel<'m, 'f> {
    /// Fixed-public laboratory experiment, called between completed native
    /// chunks. A fresh/scalar owner, pending work or missing full owning audit
    /// cannot change policy. This adds no launch, map, fence or device allocation.
    pub(crate) fn set_lab_native_projection_tile(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tile: usize,
    ) -> Result<usize> {
        if tile != 1 && self.native_decode_kernels()? != NativeDecodeKernels::Chunk {
            return Err(error());
        }
        let tile = NativeProjectionTile::new(tile)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        executor.set_native_projection_tile(tile, boundary)
    }

    fn completed_native_projection_boundary(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<CompletedNativeProjectionBoundary> {
        if self.poisoned
            || self.retired.load(Ordering::Acquire)
            || !self.pending_qsa.is_empty()
            || !self.pending_moe.is_empty()
            || self.ple_batch_pending.is_some()
            || !self.admission.transient_io()
            || !self.arena.public_owned_zero_audit_enabled()
            || !self.io.public_owned_zero_audit_enabled()
        {
            return Err(error());
        }
        self.graph.verify_unchanged()?;
        executor.check_client(self.arena.client().map_err(u::fault)?)?;
        self.admission
            .check_client(self.io.client().map_err(u::fault)?)?;
        executor.check_resident_admission(&self.admission)?;
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.prove_outer(self)?;
        native.gr.validate_all_consumers_queued()?;
        native.shared.validate_all_consumers_queued()?;
        native.ple.validate_completed_projection_boundary(self)?;
        Ok(CompletedNativeProjectionBoundary { _private: () })
    }

    /// Read the actual installed family policies rather than echoing an env
    /// selection. No private metadata or model state is exposed.
    pub(crate) fn native_decode_kernels(&self) -> Result<NativeDecodeKernels> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.decode_kernels()
    }
    pub(crate) fn advise_gpu_sources_not_needed(
        &self,
        executor: &GpuExecutor<'f>,
        pageout: bool,
    ) -> Result<u64> {
        let proof = self.completed_native_projection_boundary(executor)?;
        if self.native_cpu_miss_cap()? != Some(10) {
            return Err(error());
        }
        executor.advise_resident_gpu_sources_not_needed(&proof, pageout)
    }

    #[allow(dead_code)] // Protected receipt exposure is owned separately.
    pub(crate) fn native_dense_member_capacity(&self) -> Result<usize> {
        let native = self.native_batch.as_ref().ok_or(error())?;
        native.decode_kernels()?;
        Ok(native.dense_member_capacity)
    }

    pub(crate) fn native_cpu_miss_cap(&self) -> Result<Option<usize>> {
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_miss_cap())
    }

    pub(crate) fn native_cpu_workers_pooled(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_workers_pooled())
    }

    pub(crate) fn native_cpu_arithmetic_geometry(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<(usize, usize)>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_arithmetic_geometry())
    }
    pub(crate) fn native_cpu_mapped_prefetch(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<(bool, u64)>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_mapped_prefetch())
    }
    pub(crate) fn native_cpu_q8_residual(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_q8_residual())
    }
    pub(crate) fn native_cpu_q8_activations(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_q8_activations())
    }
    pub(crate) fn native_deferred_router_reads(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<(bool, u64)> {
        self.completed_native_projection_boundary(executor)?;
        if self.native_cpu_miss_cap()?.is_none() {
            return Err(error());
        }
        Ok((
            self.admission.deferred_token_io(),
            self.admission.deferred_router_reads(),
        ))
    }

    pub(crate) fn native_cpu_gpu_overlap(
        &self,
        executor: &GpuExecutor<'f>,
    ) -> Result<Option<bool>> {
        self.completed_native_projection_boundary(executor)?;
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .moe
            .cpu_gpu_overlap())
    }

    /// Aggregate only at a healthy completed original-model boundary. The lab
    /// may publish it for the compiled public fixture; private requests may not.
    pub(crate) fn completed_cpu_misses(&self, executor: &GpuExecutor<'f>) -> Result<u64> {
        self.completed_native_projection_boundary(executor)?;
        Ok(*self.completed_cpu_misses)
    }

    pub(crate) fn shared_gdn_calls(&self) -> Result<u64> {
        Ok(self
            .native_batch
            .as_ref()
            .ok_or(error())?
            .gdn
            .shared_vector_calls)
    }

    /// Explicit reserved count1 lab only. The original completed all-family
    /// boundary remains the sole authority; this performs no inference/commit.
    pub(crate) fn set_lab_count1_cpu_misses(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        cap: usize,
        cancel: &AtomicBool,
    ) -> Result<usize> {
        u::cancel(cancel)?;
        if cap > 10 {
            return Err(error());
        }
        let boundary = self.completed_native_projection_boundary(executor)?;
        let attempt =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<usize> {
                let native = self.native_batch.as_mut().ok_or(error())?;
                let observed = native.moe.set_lab_count1_cpu_misses(
                    executor,
                    self.graph,
                    &self.arena,
                    cap,
                    &boundary,
                )?;
                u::cancel(cancel)?;
                Ok(observed)
            }));
        match attempt {
            Ok(Ok(observed)) => Ok(observed),
            _ => {
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }

    pub(crate) fn set_lab_native_decode_kernels(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        policy: NativeDecodeKernels,
        cancel: &AtomicBool,
    ) -> Result<NativeDecodeKernels> {
        u::cancel(cancel)?;
        let boundary = self.completed_native_projection_boundary(executor)?;
        let observed = self.native_decode_kernels()?;
        // Each fixture has one public continuation policy. Repeated same-policy
        // selection is allowed; a fresh owner is required for another sample.
        if observed != NativeDecodeKernels::Chunk && observed != policy
            || policy != NativeDecodeKernels::Chunk && executor.native_projection_tile() != 1
        {
            return Err(error());
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let native = self.native_batch.as_mut().ok_or(error())?;
            native.gdn.set_lab_recurrence_policy(
                executor,
                &self.arena,
                policy.recurrence(),
                &boundary,
                cancel,
            )?;
            if policy == NativeDecodeKernels::SingleQ4All {
                native
                    .moe
                    .set_lab_single_q4_decode(executor, &self.arena, &boundary)?;
            } else if policy.single_moe() {
                native
                    .moe
                    .set_lab_single_member_decode(executor, &self.arena, &boundary)?;
            }
            if matches!(
                policy,
                NativeDecodeKernels::SingleAll | NativeDecodeKernels::SingleQ4All
            ) {
                native.dense_member_capacity =
                    executor.set_lab_native_single_dense_dispatch(&boundary)?;
            }
            if native.dense_member_capacity != executor.native_projection_member_capacity() {
                return Err(error());
            }
            let actual = self.native_decode_kernels()?;
            if actual != policy {
                return Err(error());
            }
            u::cancel(cancel)?;
            Ok(actual)
        }));
        match attempt {
            Ok(Ok(actual)) => Ok(actual),
            _ => {
                // A failed multi-family policy publication cannot leave an
                // owner usable under a partially selected experiment.
                self.admission
                    .sequence_poison()
                    .store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                let _ = self.close();
                let _ = executor.close_transient_resident();
                Err(error())
            }
        }
    }

    pub(crate) fn new_with_native_batch(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut GpuExecutor<'f>,
        config: Config,
        public_zero_audit: bool,
    ) -> Result<Self> {
        Self::construct_native(
            graph,
            client,
            context,
            executor,
            public_zero_audit,
            Some(config),
        )
    }
    pub(crate) fn new_for_lab_fixture(
        graph: BoundGraph<'m, 'f>,
        client: ComputeClient<WgpuRuntime>,
        context: usize,
        executor: &mut GpuExecutor<'f>,
        public_zero_audit: bool,
        config: Option<Config>,
    ) -> Result<Self> {
        match config {
            Some(config) => Self::new_with_native_batch(
                graph,
                client,
                context,
                executor,
                config,
                public_zero_audit,
            ),
            None => Self::new_with_public_owned_zero_audit(
                graph,
                client,
                context,
                executor,
                public_zero_audit,
            ),
        }
    }
    /// Real layer-major enqueue and whole-chunk acceptance. Opaque receipts
    /// cannot publish a token until ALL families and both required fences pass.
    pub(crate) fn prefill_chunk(
        &mut self,
        executor: &mut GpuExecutor<'f>,
        tokens: &[u32],
        purpose: Purpose,
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<f32>>> {
        let mut native = self.native_batch.take().ok_or(error())?;
        let mut flight = Flight {
            poison: native.poison.clone(),
            accepted: false,
        };
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native.run(self, executor, tokens, purpose, cancel)
        }));
        match attempt {
            Ok(Ok(logits)) => {
                flight.accepted = true;
                self.native_batch = Some(native);
                Ok(logits)
            }
            _ => {
                flight.poison.store(true, Ordering::Release);
                self.poisoned = true;
                executor.poison_resident();
                self.native_batch = Some(native);
                let _ = self.close(); // Explicit close outside admission-held guards.
                let _ = executor.close_transient_resident(); // Abort/drain the actual staged bank registry.
                Err(error())
            }
        }
    }
}

type TokenPacket = (
    Zeroizing<[u32; MAX_BATCH]>,
    Zeroizing<[u32; MAX_BATCH]>,
    Zeroizing<usize>,
);
fn packet(start: usize, context: usize, tokens: &[u32]) -> Result<TokenPacket> {
    if !(1..=MAX_BATCH).contains(&tokens.len()) || context == 0 || context > 4096 {
        return Err(error());
    }
    let end = start
        .checked_add(tokens.len())
        .ok_or(Error::ArithmeticOverflow)?;
    if end > context || tokens.iter().any(|&t| t as usize >= 248320) {
        return Err(error());
    }
    let mut ids = Zeroizing::new([0; MAX_BATCH]);
    let mut positions = Zeroizing::new([0; MAX_BATCH]);
    for (t, &id) in tokens.iter().enumerate() {
        ids[t] = id;
        positions[t] = (start + t) as u32;
    }
    Ok((ids, positions, Zeroizing::new(end)))
}

/// Used by real dispatch: a failed/cancelled actual layer cannot queue a later
/// layer. The callback contains the full GR/attention/router/MoE/shared work.
fn drive_layers(cancel: &AtomicBool, mut layer: impl FnMut(usize) -> Result<()>) -> Result<()> {
    for index in 0..LAYERS {
        u::cancel(cancel)?;
        layer(index)?;
    }
    u::cancel(cancel)
}
/// Both actual completed proofs surround PLE's real owning diagnostic maps.
/// No guarded output/PLE permit escapes if a map, audit or second proof fails.
fn completed_proof<O, P>(
    first: impl FnOnce() -> Result<O>,
    ple: impl FnOnce() -> Result<P>,
    second: impl FnOnce() -> Result<()>,
) -> Result<(O, P)> {
    let output = first()?;
    let prepared = ple()?;
    second()?;
    Ok((output, prepared))
}

impl Workspace {
    fn run(
        &mut self,
        m: &mut ResidentModel<'_, '_>,
        e: &mut GpuExecutor<'_>,
        tokens: &[u32],
        purpose: Purpose,
        cancel: &AtomicBool,
    ) -> Result<Zeroizing<Vec<f32>>> {
        self.prove_outer(m)?;
        e.check_client(m.arena.client().map_err(u::fault)?)?;
        e.check_resident_admission(&m.admission)?;
        if self.dense_member_capacity != e.native_projection_member_capacity() {
            return Err(error());
        }
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        // Preflight both single-token families before clears, embeddings or
        // any other layer can enqueue work for this transaction.
        self.decode_kernels()?.validate_count(tokens.len())?;
        if self.moe.cpu_miss_cap().is_some() && tokens.len() != 1 {
            return Err(error());
        }
        if m.poisoned
            || m.retired.load(Ordering::Acquire)
            || !m.pending_qsa.is_empty()
            || !m.pending_moe.is_empty()
            || m.ple_batch_pending.is_some()
            || m.layers.len() != LAYERS
        {
            return Err(error());
        }
        let start = Zeroizing::new(m.position);
        let count = Zeroizing::new(tokens.len());
        let (ids, positions, end) = packet(*start, m.context, tokens)?;
        let mut pending = Pending::new()?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        let clear_ids = Zeroizing::new([
            self.outer.residual[0],
            self.outer.residual[1],
            self.outer.block[0],
            self.outer.block[1],
            self.outer.router,
            self.outer.packet,
            m.b.fault,
            m.b.logits,
            m.b.logit_packet,
        ]);
        u::zero_many(&m.arena, &*clear_ids)?;
        batch_io::enqueue_embedding(
            m.graph,
            &m.arena,
            &m.admission,
            self.outer.residual[0],
            batch_io::InputView {
                first: 0,
                stride: HC * D,
                words: RESIDUAL_WORDS,
            },
            *count,
            *start,
            m.context,
            &ids,
            &positions,
            cancel,
        )?;
        let mut session = e.begin_routed_moe_chunk(
            &mut self.moe,
            m.graph,
            &m.arena,
            *start as u64,
            *count,
            m.b.fault,
            cancel,
        )?;
        let mut current = Zeroizing::new(0usize);
        drive_layers(cancel, |layer| {
            let attention_start = Instant::now();
            if layer == 1 {
                m.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Input,
                    &m.arena,
                    &m.admission,
                    self.outer.residual[*current],
                    m.b.fault,
                    RESIDUAL_WORDS,
                    *start,
                    *count,
                    cancel,
                )?;
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                };
                let spans = BatchSpans::new(*count, 0, 0, HC * D, 0, HC * D);
                pending.ple = Some(self.ple.enqueue(
                    m,
                    &mut projector,
                    self.outer.residual[*current],
                    self.outer.residual[1 - *current],
                    &spans,
                    *start,
                    &ids,
                    &positions,
                    cancel,
                )?);
                m.numeric_boundary.observe(
                    numeric_boundaries::Boundary::Injected,
                    &m.arena,
                    &m.admission,
                    self.outer.residual[1 - *current],
                    m.b.fault,
                    RESIDUAL_WORDS,
                    *start,
                    *count,
                    cancel,
                )?;
                *current = 1 - *current;
            }
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
            };
            let prepared = self.gr.read(
                m.graph,
                &m.arena,
                &mut m.public,
                &mut projector,
                gr::Kind::Attention(layer),
                self.outer.residual[*current],
                *count,
                m.b.fault,
                cancel,
            )?;
            let spans = BatchSpans::new(*count, 0, 0, D, 0, D);
            match &mut m.layers[layer] {
                Layer::Gdn(stage, public) => {
                    pending.gdn[layer] = Some(stage.enqueue_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut projector,
                        &mut self.gdn,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?)
                }
                Layer::Qsa(stage, public) => {
                    let mut selector = qsa::AdmittedBatchSelector {
                        admission: &m.admission,
                    };
                    pending.qsa[layer] = Some(stage.enqueue_batch(
                        m.graph,
                        &m.arena,
                        public,
                        &mut projector,
                        &mut selector,
                        &mut self.qsa,
                        prepared.mixed(),
                        self.outer.block[0],
                        &spans,
                        *start as u64,
                        m.b.fault,
                        cancel,
                    )?);
                }
            }
            prepared.combine(self.outer.block[0], self.outer.residual[1 - *current])?;
            *current = 1 - *current;
            // Public laboratory aggregates only. These wall-clock enqueue
            // intervals include any internal dependency waits; they are not
            // isolated GPU kernel times or additional completion proofs.
            *m.phase_ns
                .entry("native_attention_and_gr_enqueue")
                .or_default() += attention_start.elapsed().as_nanos();
            if m.cancel_after_attention == Some(layer) {
                m.cancel_after_attention = None;
                cancel.store(true, Ordering::Release);
                u::cancel(cancel)?;
            }
            let router_enqueue_start = Instant::now();
            let prepared = self.gr.read(
                m.graph,
                &m.arena,
                &mut m.public,
                &mut projector,
                gr::Kind::Ffn(layer),
                self.outer.residual[*current],
                *count,
                m.b.fault,
                cancel,
            )?;
            u::zero_many(&m.arena, &[self.outer.router, self.outer.packet])?;
            let tensor = m
                .graph
                .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))?;
            let matrix = Matrix::new(tensor)?;
            if matrix.inputs() != D || matrix.outputs() != ROUTER || matrix.experts() != 1 {
                return Err(error());
            }
            projector.enqueue_batch(
                BoundProjection { tensor, matrix },
                &m.arena,
                prepared.mixed(),
                self.outer.router,
                &BatchSpans::new(*count, 0, 0, D, 0, ROUTER),
                cancel,
            )?;
            *m.phase_ns
                .entry("native_ffn_gr_and_router_enqueue")
                .or_default() += router_enqueue_start.elapsed().as_nanos();
            let router_fence_start = Instant::now();
            let hybrid_packet = if self.moe.cpu_miss_cap().is_some() {
                Some(batch_io::read_count1_router_input(
                    m.graph,
                    &m.arena,
                    &m.admission,
                    prepared.mixed(),
                    self.outer.router,
                    self.outer.packet,
                    m.b.fault,
                    batch_io::InputView {
                        first: 0,
                        stride: D,
                        words: BLOCK_WORDS,
                    },
                    batch_io::RouterView {
                        first: 0,
                        stride: ROUTER,
                        words: MAX_BATCH * ROUTER,
                    },
                    *count,
                    cancel,
                    layer,
                    *start as u64,
                    e.gpu_router,
                    e.shared_kernels,
                )?)
            } else {
                None
            };
            let ordinary_logits = if hybrid_packet.is_none() {
                Some(batch_io::read_router(
                    m.graph,
                    &m.arena,
                    &m.admission,
                    self.outer.router,
                    self.outer.packet,
                    m.b.fault,
                    batch_io::RouterView {
                        first: 0,
                        stride: ROUTER,
                        words: MAX_BATCH * ROUTER,
                    },
                    *count,
                    cancel,
                )?)
            } else {
                None
            };
            let logits = match (&hybrid_packet, &ordinary_logits) {
                (Some(packet), None) => packet.logits(),
                (None, Some(logits)) => logits,
                _ => return Err(error()),
            };
            *m.phase_ns.entry("native_router_fence").or_default() +=
                router_fence_start.elapsed().as_nanos();
            let use_gpu_router = e.gpu_router && hybrid_packet.is_some();
            if use_gpu_router {
                if hybrid_packet
                    .as_ref()
                    .and_then(|packet| packet.gpu_selection())
                    .is_none()
                {
                    return Err(error());
                }
                e.gpu_router_calls = e
                    .gpu_router_calls
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            let expert_start = Instant::now();
            let mut routes = token_waves::Input::new(*count, self.moe.capacity(), ROUTER)
                .map_err(|_| error())?;
            for token in 0..*count {
                let selection = if use_gpu_router {
                    None
                } else {
                    Some(ops::router_topk_guarded(logits.row(token)?, 10, true).map_err(u::fault)?)
                };
                let selected = match (&selection, &hybrid_packet) {
                    (Some(selection), _) => selection.as_slice(),
                    (None, Some(packet)) => packet.gpu_selection().ok_or(error())?.as_slice(),
                    _ => return Err(error()),
                };
                for (rank, &(expert, probability)) in selected.iter().enumerate() {
                    routes
                        .set(token, rank, expert, probability)
                        .map_err(|_| error())?;
                }
            }
            let waves = token_waves::Plan::build(&routes, cancel).map_err(|_| error())?;
            let overlap_shared =
                hybrid_packet.is_some() && self.moe.cpu_gpu_overlap() == Some(true);
            if overlap_shared {
                let mut projector = ResidentBatchProjection {
                    executor: e,
                    fault: m.b.fault,
                };
                self.shared.prepare(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
            }
            pending.moe[layer] = Some(if let Some(packet) = &hybrid_packet {
                e.enqueue_routed_moe_count1_hybrid(
                    m.graph,
                    &m.arena,
                    &mut self.moe,
                    &session,
                    layer,
                    prepared.mixed(),
                    self.outer.block[0],
                    &spans,
                    &waves,
                    packet,
                    cancel,
                )?
            } else {
                e.enqueue_routed_moe_batch(
                    m.graph,
                    &m.arena,
                    &mut self.moe,
                    &session,
                    layer,
                    prepared.mixed(),
                    self.outer.block[0],
                    &spans,
                    &waves,
                    cancel,
                )?
            });
            e.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
            *m.phase_ns
                .entry("native_route_and_expert_enqueue")
                .or_default() += expert_start.elapsed().as_nanos();
            let shared_start = Instant::now();
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
            };
            if overlap_shared {
                self.shared.finish_prepared(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
                e.shared_overlap_calls = e
                    .shared_overlap_calls
                    .checked_add(1)
                    .ok_or(Error::ArithmeticOverflow)?;
            } else {
                self.shared.enqueue(
                    m.graph,
                    layer,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    prepared.mixed(),
                    self.outer.block[0],
                    self.outer.block[1],
                    m.b.fault,
                    *count,
                    cancel,
                )?;
            }
            prepared.combine(self.outer.block[1], self.outer.residual[1 - *current])?;
            *current = 1 - *current;
            *m.phase_ns
                .entry("native_shared_and_ffn_combine_enqueue")
                .or_default() += shared_start.elapsed().as_nanos();
            // Fixed-public lab hook: actual routed/shared consumers and HC
            // combine are already queued; no family has published its state.
            m.lifecycle_hooks.late_combine(layer)?;
            Ok(())
        })?;
        let final_head_start = Instant::now();
        if purpose == Purpose::LastLogits {
            let mut projector = ResidentBatchProjection {
                executor: e,
                fault: m.b.fault,
            };
            self.gr
                .read(
                    m.graph,
                    &m.arena,
                    &mut m.public,
                    &mut projector,
                    gr::Kind::Final,
                    self.outer.residual[*current],
                    *count,
                    m.b.fault,
                    cancel,
                )?
                .enqueue_final_head(m.graph, &mut projector, m.b.logits)?;
        }
        *m.phase_ns.entry("native_final_head_enqueue").or_default() +=
            final_head_start.elapsed().as_nanos();
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        let final_proof_start = Instant::now();
        let (output, ple) = completed_proof(
            || {
                Ok(if purpose == Purpose::LastLogits {
                    batch_io::read_last_logits(
                        m.graph,
                        &m.arena,
                        &m.admission,
                        m.b.logits,
                        m.b.logit_packet,
                        m.b.fault,
                        cancel,
                    )?
                } else {
                    batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)?;
                    Zeroizing::new(Vec::new())
                })
            },
            || {
                self.ple.prepare_split_after_whole_model_fence(
                    ple_batch::Preparation {
                        graph: m.graph,
                        arena: &m.arena,
                        retired: &m.retired,
                        context: m.context,
                        position: *start,
                        owner_poisoned: &mut m.poisoned,
                        sequence_poison: &m.ple_batch_poison,
                        conv: &mut m.ple_conv,
                        history: &mut m.ple_history,
                        anchor: &mut m.ple_batch_pending,
                        fault: m.b.fault,
                        scalar_ids: [m.b.ple_emb, m.b.ple_key, m.b.ple_value],
                    },
                    pending.ple.take().ok_or(error())?,
                    cancel,
                )
            },
            || {
                // The real PLE preparation exists and the first whole-model
                // proof completed. Failure here must drop that real ticket.
                m.lifecycle_hooks.before_second_proof()?;
                batch_io::read_fault(&m.arena, &m.admission, m.b.fault, cancel)
            },
        )?;
        *m.phase_ns
            .entry("native_completed_final_proofs")
            .or_default() += final_proof_start.elapsed().as_nanos();
        let family_start = Instant::now();
        let mut ga = {
            let (first_stage, first_receipt) = match (&m.layers[0], pending.gdn[0].as_ref()) {
                (Layer::Gdn(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 36];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Gdn(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.gdn[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(error());
            }
            self.gdn
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        let mut qa = {
            let (first_stage, first_receipt) = match (&m.layers[3], pending.qsa[3].as_ref()) {
                (Layer::Qsa(s, _), Some(r)) => (s.as_ref(), r),
                _ => return Err(error()),
            };
            let mut refs = [(first_stage, first_receipt); 12];
            let mut at = 0;
            for layer in 0..LAYERS {
                if let Layer::Qsa(s, _) = &m.layers[layer] {
                    refs[at] = (s, pending.qsa[layer].as_ref().ok_or(error())?);
                    at += 1;
                }
            }
            if at != refs.len() {
                return Err(error());
            }
            self.qsa
                .accept_after_whole_model_chunk_acceptance(&m.arena, &refs, cancel)?
        };
        let mut ma = {
            let first = pending.moe[0].as_ref().ok_or(error())?;
            let mut refs = [first; LAYERS];
            for (i, r) in refs.iter_mut().enumerate() {
                *r = pending.moe[i].as_ref().ok_or(error())?;
            }
            self.moe.accept_after_whole_model_chunk_acceptance(
                e,
                &mut session,
                &m.arena,
                &refs,
                cancel,
            )?
        };
        ple.preflight(cancel)?;
        self.gr.validate_all_consumers_queued()?;
        self.shared.validate_all_consumers_queued()?;
        m.graph.verify_unchanged()?;
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire)
            || m.retired.load(Ordering::Acquire)
            || m.position != *start
        {
            return Err(error());
        }
        // Prepare the checked counter before ANY family publication. The
        // guarded proposal is discarded on all late failure/cancellation paths.
        let completed_cpu_misses =
            checked_cpu_miss_total(*m.completed_cpu_misses, ma.completed_cpu_misses())?;
        *m.phase_ns.entry("native_all_family_preflight").or_default() +=
            family_start.elapsed().as_nanos();
        let commit_start = Instant::now();
        // Every actual-family permit has now preflighted. Remaining cancellation
        // or fallible commit failure terminally destroys the whole model; it
        // cannot publish output/cursor or resume a partially changed sequence.
        for (layer, owner) in m.layers.iter_mut().enumerate() {
            match owner {
                Layer::Gdn(stage, _) => {
                    stage.commit_after_whole_chunk_acceptance(
                        pending.gdn[layer].take().ok_or(error())?,
                        &m.arena,
                        &mut ga,
                        cancel,
                    )?;
                    // Inject only AFTER the first genuine family commit. The
                    // next real commit must reject and destroy this sequence.
                    m.lifecycle_hooks.after_gdn_commit(layer, cancel);
                }
                Layer::Qsa(stage, _) => stage.commit_after_whole_chunk_acceptance(
                    pending.qsa[layer].take().ok_or(error())?,
                    &m.arena,
                    &mut qa,
                    cancel,
                )?,
            }
        }
        for receipt in pending.moe.iter_mut() {
            ma.commit(receipt.take().ok_or(error())?, &m.arena, cancel)?;
        }
        ga.finish(&m.arena, cancel)?;
        qa.finish(&m.arena, cancel)?;
        ma.finish(&m.arena, cancel)?;
        ple.commit(cancel)?;
        *m.phase_ns.entry("native_all_family_commit").or_default() +=
            commit_start.elapsed().as_nanos();
        e.check_original_bank_metadata(LAYERS, ORIGINAL_BANK_HOST)?;
        u::cancel(cancel)?;
        if self.poison.load(Ordering::Acquire) || m.retired.load(Ordering::Acquire) || m.poisoned {
            return Err(error());
        }
        *m.completed_cpu_misses = *completed_cpu_misses;
        m.position = *end;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_counter_preflight_rejects_overflow_and_impossible_chunk_without_publication() {
        assert_eq!(*checked_cpu_miss_total(7, 480).unwrap(), 487);
        assert_eq!(*checked_cpu_miss_total(u64::MAX, 0).unwrap(), u64::MAX);
        assert!(checked_cpu_miss_total(u64::MAX, 1).is_err());
        assert!(checked_cpu_miss_total(0, 481).is_err());
        // The current value is never borrowed mutably during preflight: a
        // prepared proposal can be abandoned after any family failure.
        let current = Zeroizing::new(7u64);
        drop(checked_cpu_miss_total(*current, 480).unwrap());
        assert_eq!(*current, 7);
    }
    #[test]
    fn public_decode_profiles_are_strict_and_preflight_single_counts() {
        for (label, profile) in [
            ("chunk", NativeDecodeKernels::Chunk),
            ("single-moe", NativeDecodeKernels::SingleMoe),
            ("single-gdn", NativeDecodeKernels::SingleGdn),
            ("single-all", NativeDecodeKernels::SingleAll),
            ("single-q4-all", NativeDecodeKernels::SingleQ4All),
        ] {
            assert_eq!(NativeDecodeKernels::parse(label).unwrap(), profile);
            assert_eq!(profile.label(), label);
            for count in [0, 1, 3, 5, 8, 9, usize::MAX] {
                assert_eq!(
                    profile.validate_count(count).is_ok(),
                    (1..=MAX_BATCH).contains(&count)
                        && (profile == NativeDecodeKernels::Chunk || count == 1)
                );
            }
        }
        for label in [
            "",
            "1",
            "SingleAll",
            " single-all",
            "single-all\n",
            "chunk ",
        ] {
            assert!(NativeDecodeKernels::parse(label).is_err());
        }
        assert_eq!(NativeDecodeKernels::default(), NativeDecodeKernels::Chunk);
    }
    #[test]
    fn aggregate_plan_counts_original_owners_once_and_independent_runtime_head_tiles() {
        assert_eq!(TOTAL_MAIN, 818249728);
        assert_eq!(
            NEW_MAIN + PUBLIC_MAIN + 2 * 67108864 + 41943040 + OLD_MAIN,
            1019576320
        );
        assert_eq!(runtime_bytes(16).unwrap(), 64000000);
        assert_eq!(runtime_bytes(38).unwrap(), 64090112);
        assert!(runtime_bytes(u64::MAX).is_err());
        assert!(runtime_bytes(4096).unwrap() > crate::executor::RESIDENT_RUNTIME_IO_MAX);
        assert!(metadata_bytes() <= META_CAP);
        assert_eq!(HOST_COMPONENTS, 11206656);
        const { assert!(HOST_COMPONENTS <= HOST_CAP) };
    }
    #[test]
    fn explicit_count1_reservation_reuses_packet_and_stays_under_independent_caps() {
        assert_eq!(
            batch_moe::Workspace::count1_hybrid_allocation_requests().unwrap(),
            [131072, 32768]
        );
        assert_eq!(TOTAL_MAIN + HYBRID_MAIN, 818413568);
        assert_eq!(1286 + 90 + 2, 1378);
        assert_eq!(HOST_CAP - (HOST_COMPONENTS + HYBRID_HOST), 327680);
        assert_eq!(runtime_bytes(38).unwrap() + HYBRID_RUNTIME, 65138688);
        assert_eq!(HYBRID_IO_DERIVED, 1015808);
        assert_eq!(HYBRID_RUNTIME - HYBRID_IO_DERIVED, 32768);
        assert!(
            runtime_bytes(38).unwrap() + HYBRID_RUNTIME <= crate::executor::RESIDENT_RUNTIME_IO_MAX
        );
        assert_eq!(
            BufferLayout::new(MAX_BATCH * ROUTER + 1)
                .unwrap()
                .owned_bytes,
            32768
        );
        const { assert!(D + ROUTER <= MAX_BATCH * ROUTER) };
    }
    #[test]
    fn runtime_packet_covers_counts_boundaries_without_future_or_inactive_values() {
        for count in [1, 3, 5, 8] {
            for end in [4, 2048, 2049, 2052, 4096] {
                if count > end {
                    continue;
                }
                let input = vec![7; count];
                let (ids, p, actual) = packet(end - count, 4096, &input).unwrap();
                assert_eq!(*actual, end);
                for t in 0..8 {
                    if t < count {
                        assert_eq!(ids[t], 7);
                        assert_eq!(p[t] as usize, end - count + t);
                    } else {
                        assert_eq!(ids[t], 0);
                        assert_eq!(p[t], 0);
                    }
                }
            }
        }
        for (start, context, ids) in [
            (0, 4096, vec![]),
            (0, 4096, vec![1; 9]),
            (4096, 4096, vec![1]),
            (usize::MAX, 4096, vec![1]),
            (0, 4096, vec![248320]),
            (0, 4097, vec![1]),
        ] {
            assert!(packet(start, context, &ids).is_err());
        }
    }
    #[test]
    fn actual_inflight_guard_abandon_error_unwind_is_sticky_without_lock_callback() {
        for kind in 0..4 {
            let poison = Arc::new(AtomicBool::new(false));
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut flight = Flight {
                    poison: poison.clone(),
                    accepted: false,
                };
                match kind {
                    0 => {
                        flight.accepted = true;
                        Ok(())
                    }
                    1 => Err(error()),
                    2 => Ok(()),
                    _ => panic!("public injected chunk panic"),
                }
            }));
            assert_eq!(poison.load(Ordering::Acquire), kind != 0);
            assert_eq!(attempt.is_err(), kind == 3);
        }
    }
    #[test]
    fn production_layer_driver_is_complete_and_stops_before_next_actual_layer() {
        for failed in [None, Some(0), Some(3), Some(23), Some(47)] {
            let mut queued = Vec::new();
            let result = drive_layers(&AtomicBool::new(false), |layer| {
                queued.push(layer);
                if failed == Some(layer) {
                    Err(error())
                } else {
                    Ok(())
                }
            });
            assert_eq!(
                queued,
                (0..failed.map_or(48, |n| n + 1)).collect::<Vec<_>>()
            );
            assert_eq!(result.is_err(), failed.is_some());
        }
        let cancelled = AtomicBool::new(false);
        let mut queued = Vec::new();
        assert!(
            drive_layers(&cancelled, |layer| {
                queued.push(layer);
                if layer == 3 {
                    cancelled.store(true, Ordering::Release);
                }
                Ok(())
            })
            .is_err()
        );
        assert_eq!(queued, [0, 1, 2, 3]);
    }
    #[test]
    fn production_two_proofs_never_return_output_or_ple_permit_after_any_failure() {
        use std::cell::RefCell;
        struct DropProof<'a>(&'a RefCell<Vec<&'static str>>, &'static str);
        impl Drop for DropProof<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push(self.1);
            }
        }
        for failure in [None, Some(0), Some(1), Some(2)] {
            let events = RefCell::new(Vec::new());
            let result = completed_proof(
                || {
                    events.borrow_mut().push("first model owning fence/audit");
                    if failure == Some(0) {
                        Err(error())
                    } else {
                        Ok(DropProof(&events, "output dropped"))
                    }
                },
                || {
                    events.borrow_mut().push("PLE owning maps/audit/prepare");
                    if failure == Some(1) {
                        Err(error())
                    } else {
                        Ok(DropProof(&events, "PLE permit dropped"))
                    }
                },
                || {
                    events.borrow_mut().push("second model fault/cleanup proof");
                    if failure == Some(2) {
                        Err(error())
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(result.is_err(), failure.is_some());
            match result {
                Ok((output, permit)) => {
                    assert_eq!(events.borrow().len(), 3);
                    events.borrow_mut().push("all family preflight then commit");
                    drop(permit);
                    drop(output);
                }
                Err(_) => assert!(
                    !events
                        .borrow()
                        .contains(&"all family preflight then commit")
                ),
            }
            let observed = events.borrow();
            assert_eq!(observed.first(), Some(&"first model owning fence/audit"));
            if failure == Some(0) {
                assert_eq!(observed.len(), 1);
            }
            if failure == Some(1) {
                assert_eq!(
                    &*observed,
                    &[
                        "first model owning fence/audit",
                        "PLE owning maps/audit/prepare",
                        "output dropped"
                    ]
                );
            }
            if failure == Some(2) {
                assert_eq!(
                    &*observed,
                    &[
                        "first model owning fence/audit",
                        "PLE owning maps/audit/prepare",
                        "second model fault/cleanup proof",
                        "PLE permit dropped",
                        "output dropped"
                    ]
                );
            }
        }
    }
    #[test]
    fn wholeflight_poison_survives_panic_inside_the_actual_second_proof() {
        let poison = Arc::new(AtomicBool::new(false));
        let reached_commit = AtomicBool::new(false);
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _flight = Flight {
                poison: poison.clone(),
                accepted: false,
            };
            completed_proof(
                || Ok(()),
                || Ok(()),
                || panic!("public injected second-fence panic"),
            )
            .unwrap();
            reached_commit.store(true, Ordering::Release);
        }));
        assert!(attempt.is_err());
        assert!(poison.load(Ordering::Acquire));
        assert!(!reached_commit.load(Ordering::Acquire));
    }
    #[test]
    fn actual_limit_inventory_rejects_binding_alignment_and_workspace_shortfalls() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 16384,
            max_storage_buffers_per_shader_stage: 11,
            max_bindings_per_bind_group: 12,
            max_compute_workgroup_storage_size: 8192,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            max_buffer_size: 128 * 1024 * 1024,
            ..Default::default()
        };
        for tile in [4, 8] {
            validate_limits(&limits, tile).unwrap();
        }
        assert!(validate_limits(&limits, 16).is_err());
        for kind in 0..6 {
            let mut malformed = limits.clone();
            match kind {
                0 => malformed.max_compute_workgroups_per_dimension = 4096,
                1 => malformed.max_storage_buffers_per_shader_stage = 10,
                2 => malformed.max_bindings_per_bind_group = 11,
                3 => malformed.max_compute_workgroup_storage_size = 8191,
                4 => malformed.min_storage_buffer_offset_alignment = 1024,
                _ => malformed.min_uniform_buffer_offset_alignment = 0,
            }
            assert!(validate_limits(&malformed, 8).is_err());
        }
    }
}
