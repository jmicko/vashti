//! Variable global HOST complement, using the original nonlocal Vulkan heap and
//! sole CubeCL immutable registry. No fixed 432-page or uniform-bank assumption.
#[cfg(feature="strata-host-expert-blobs")]
#[path="strata_host_blob_cpu_audit.rs"]
mod blob_cpu_audit;
#[path="strata_host_blob_loader.rs"]
mod blob_loader;
#[path = "strata_global_exchange.rs"]
pub(crate) mod exchange;
#[cfg(feature = "strata-parallel-loading")]
#[path = "strata_parallel_reads.rs"]
mod parallel_reads;
#[path = "strata_prompt_shadow.rs"]
mod shadow;
#[path = "strata_prompt_shadow_plan.rs"]
pub(crate) mod shadow_plan;
use super::*;
use crate::strata_global_plan::{Plan as GlobalPlan, Source as GlobalSource, Tier};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::public_external::{
    PublicExternalBudget, PublicExternalHandle, PublicExternalHostSource,
};
use kuno_text::{Error, Result as KResult};
const METADATA_BYTES: usize = 1024 * 1024;
const SOURCE_WINDOW: u64 = 1024 * 1024;
// Global plan, owner maps/aliases and inspection temporaries replace the old
// fixed-bank metadata cap. Bounded startup upload owns at most one original
// chunk plus its backend copy; the HOST reader owns one additional window.
// The optional direct-reader path also charges its eight bounded thread stacks
// and per-worker setup allowance; actual readers join before page publication.
const GLOBAL_METADATA_CAP: u64 =
    32 * 1024 * 1024 + crate::strata_global_sources::PLACEMENT_HOST_BYTES;
const SETUP_WINDOWS: u64 = 2 * crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW
    + SOURCE_WINDOW
    + if cfg!(feature = "strata-parallel-loading") {
        PARALLEL_SETUP_BYTES
    } else {
        0
    };
#[cfg(feature = "strata-parallel-loading")]
const PARALLEL_SETUP_BYTES: u64 = parallel_reads::SETUP_BYTES;
#[cfg(not(feature = "strata-parallel-loading"))]
const PARALLEL_SETUP_BYTES: u64 = 0;
const MAX_ALIGNMENT: u64 = 1024 * 1024;
fn envelope(bytes: u64) -> KResult<u64> {
    bytes
        .checked_add(MAX_ALIGNMENT - 1)
        .map(|n| n & !(MAX_ALIGNMENT - 1))
        .ok_or(Error::ArithmeticOverflow)
}
fn fail() -> Error {
    Error::InvalidFormat("Strata global host complement")
}
struct GlobalPage {
    layer: usize,
    role: usize,
    piece: usize,
    owner: PageOwner,
    authority: Option<PublicExternalHandle>,
}
// LAYERS identifies optional prompt shadow pages. The disjoint next tag
// identifies immutable embedding pages; no expert/loan indexing may alias it.
const EMBEDDING_LAYER: usize = crate::strata_global_plan::LAYERS + 1;
pub(crate) struct Embedding {
    pub(crate) pages: Vec<crate::strata_shared_weights::Page>,
    _readers: Arc<()>,
}
fn embedding_detach_ready(readers: &Arc<()>) -> bool {
    Arc::strong_count(readers) == 1
}
pub(crate) struct Host {
    client: ComputeClient<WgpuRuntime>,
    device: wgpu::Device,
    instance: u64,
    registry_budget: Option<PublicExternalBudget>,
    embedding_readers: Arc<()>,
    shadow: Option<shadow::Shadow>,
    pages: Vec<GlobalPage>,
    closed: bool,
    logical: u64,
    physical: u64,
    cpu_borrow: std::sync::atomic::AtomicBool,
    packed: bool,
}
fn base_page_role(packed:bool,layer:usize,role:usize)->KResult<usize>{
    if layer>=crate::strata_global_plan::LAYERS || role>=3 {return Err(fail());}
    Ok(if packed {crate::strata_global_plan::PACKED_HOST_ROLE}else{role})
}
impl Host {
    fn base_role(&self,layer:usize,role:usize)->KResult<usize>{
        base_page_role(self.packed,layer,role)
    }
    /// Static original-weight ownership only. No mapping, device query, source
    /// reconstruction or private route/usage state enters this diagnostic.
    pub(crate) fn compiled_public_capacity(&self) -> KResult<serde_json::Value> {
        if self.closed
            || self.registry_budget.is_none()
            || self.cpu_borrow.load(std::sync::atomic::Ordering::Acquire)
            || self
                .pages
                .iter()
                .any(|page| page.owner.phase != Phase::Registered || page.authority.is_none())
        {
            return Err(fail());
        }
        Ok(serde_json::json!({
            "registered_buffers":self.pages.len(),
            "registered_logical_bytes":self.logical,
            "allocated_backing_bytes":self.physical,
            "prompt_loan_shadow_prepared":self.shadow.is_some()
        }))
    }
    pub(crate) fn borrow_cpu<'a>(
        &'a self,
        graph: BoundGraph<'_, '_>,
        plan: &GlobalPlan,
        layer: usize,
        cells: &[usize],
    ) -> KResult<CpuLayer<'a>> {
        plan.check_binding(graph)?;
        if self.closed
            || self.instance != plan.instance()
            || self.packed != plan.has_host_blobs()
            || layer >= 48
            || cells.len() > 80
            || self
                .cpu_borrow
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_err()
        {
            return Err(fail());
        }
        let mut guard = CpuLayer {
            host: self,
            layer,
            maps: Vec::new(),
        };
        for &cell in cells {
            for role in 0..3 {
                let GlobalSource::Host { piece, .. } = plan.source(layer, cell, role)? else {
                    return Err(fail());
                };
                let role=self.base_role(layer,role)?;
                if guard.maps.iter().any(|(r, p, _)| *r == role && *p == piece) {
                    continue;
                }
                let page = self
                    .pages
                    .iter()
                    .find(|p| (p.layer, p.role, p.piece) == (layer, role, piece))
                    .ok_or_else(fail)?;
                if page.owner.phase != Phase::Registered || page.authority.is_none() {
                    return Err(fail());
                }
                let backing = page.owner.backing.as_ref().ok_or_else(fail)?;
                let memory = backing.memory.ok_or_else(fail)?;
                // SAFETY: sole scoped CPU map, authenticated original HOST_VISIBLE
                // source; source publication has joined before this readonly borrow.
                let pointer = unsafe {
                    backing
                        .raw
                        .map_memory(memory, 0, backing.bytes, vk::MemoryMapFlags::empty())
                }
                .map_err(|_| fail())?
                .cast::<u8>() as *const u8;
                guard.maps.push((
                    role,
                    piece,
                    ReadOnlyMapped {
                        owner: &page.owner,
                        pointer,
                    },
                ));
                if page.owner.placement.property_flags
                    & vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()
                    == 0
                {
                    // SAFETY: whole actual mapped allocation, no pending GPU writes;
                    // invalidate after the last joined exchange for noncoherent HOST.
                    unsafe {
                        backing.raw.invalidate_mapped_memory_ranges(&[
                            vk::MappedMemoryRange::default()
                                .memory(memory)
                                .offset(0)
                                .size(vk::WHOLE_SIZE),
                        ])
                    }
                    .map_err(|_| fail())?;
                }
            }
        }
        Ok(guard)
    }
    pub(crate) fn reservation(plan: &GlobalPlan) -> KResult<u64> {
        let physical = plan.host_roots()?.iter().try_fold(0u64,|n,p|{
            n.checked_add(envelope(p.logical_bytes)?).ok_or(Error::ArithmeticOverflow)
        })?;
        let embedding =
            crate::strata_shared_weights::tiles(crate::strata_shared_weights::Tensor::Embedding)
                .iter()
                .try_fold(0u64, |n, p| {
                    n.checked_add(envelope(p.owned as u64)?)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
        physical
            .checked_add(embedding)
            .and_then(|n| n.checked_add(METADATA_BYTES as u64))
            .and_then(|n| n.checked_add(GLOBAL_METADATA_CAP))
            .and_then(|n| n.checked_add(SETUP_WINDOWS))
            .filter(|&n| n <= HOST_CAP)
            .ok_or(Error::LimitExceeded("global host reservation"))
    }
    /// # Safety
    /// Original shard hashes have been authenticated. The complete separate RAM
    /// reservation is held; this client/device owns the only consumer stream.
    pub(crate) unsafe fn load(
        plan: &GlobalPlan,
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
        device: &wgpu::Device,
        reserved: u64,
    ) -> KResult<Self> {
        if !crate::strata_intel_core::enabled() || reserved != Self::reservation(plan)? {
            return Err(fail());
        }
        plan.check_binding(graph)?;
        let count = plan.host_roots()?.len()
            .checked_add(
                crate::strata_shared_weights::tiles(
                    crate::strata_shared_weights::Tensor::Embedding,
                )
                .len(),
            )
            .ok_or(Error::ArithmeticOverflow)?;
        if count > cubecl_runtime::public_external::MAX_BUFFERS {
            return Err(fail());
        }
        let mut host = Self {
            client: client.clone(),
            device: device.clone(),
            instance: plan.instance(),
            registry_budget: None,
            embedding_readers: Arc::new(()),
            shadow: None,
            pages: Vec::with_capacity(count),
            closed: false,
            logical: 0,
            physical: 0,
            cpu_borrow: std::sync::atomic::AtomicBool::new(false),
            packed: plan.has_host_blobs(),
        };
        if count == 0 {
            return Ok(host);
        }
        let pool_bytes = reserved
            .checked_sub(METADATA_BYTES as u64 + GLOBAL_METADATA_CAP + SETUP_WINDOWS)
            .ok_or(Error::ArithmeticOverflow)?;
        let registry_budget = PublicExternalBudget {
            backing_bytes: pool_bytes,
            buffers: count,
            metadata_bytes: METADATA_BYTES,
        };
        client
            .configure_public_external_host(registry_budget)
            .map_err(|_| fail())?;
        host.registry_budget = Some(registry_budget);
        let mut charges = HeapCharges::default();
        #[cfg(feature = "strata-parallel-loading")]
        let reader = graph
            .tensor("blk.0.ffn_gate_exps.weight")?
            .positional_reader()?;
        #[cfg(not(feature = "strata-parallel-loading"))]
        let mut window = Zeroizing::new(Vec::new());
        #[cfg(not(feature = "strata-parallel-loading"))]
        {
            window
                .try_reserve_exact(SOURCE_WINDOW as usize)
                .map_err(|_| Error::LimitExceeded("global HOST source window"))?;
            window.resize(SOURCE_WINDOW as usize, 0);
        }
        if plan.has_host_blobs() {
            host.load_blob_pages(plan,graph,&mut charges)?;
        } else {
        for (layer, l) in plan.layers.iter().enumerate() {
            for (role, pieces) in l.roles.iter().enumerate() {
                let name = [
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ][role];
                let tensor = graph.tensor(&format!("blk.{layer}.{name}"))?;
                let shard = tensor.info().shard_index();
                if shard >= SOURCE_SHA256.len() {
                    return Err(fail());
                }
                for (piece, p) in pieces.iter().enumerate() {
                    if p.tier != Tier::Host {
                        continue;
                    }
                    host.allocate_original_page(
                        shadow::PageSpec {
                            layer,
                            role,
                            piece,
                            shard,
                            logical: p.logical_bytes,
                            backing_bound: envelope(p.logical_bytes)?,
                        },
                        &mut charges,
                        |mapped| {
                            #[cfg(feature = "strata-parallel-loading")]
                            {
                                let bytes = usize::try_from(p.expert_bytes)
                                    .map_err(|_| Error::ArithmeticOverflow)?;
                                if bytes == 0 || bytes > SOURCE_WINDOW as usize {
                                    return Err(Error::InvalidBufferLength);
                                }
                                let logical = p
                                    .count
                                    .checked_mul(bytes)
                                    .ok_or(Error::ArithmeticOverflow)?;
                                if logical as u64 != p.logical_bytes {
                                    return Err(Error::InvalidBufferLength);
                                }
                                let target = mapped
                                    .get_mut(..logical)
                                    .ok_or(Error::InvalidBufferLength)?;
                                // All allocation padding is already initialized.
                                // Scoped workers borrow disjoint FINAL spans;
                                // none may outlive this original mapped owner.
                                parallel_reads::fill(target, bytes, |ordinal, target| {
                                    let expert = l.expert_at(
                                        p.first
                                            .checked_add(ordinal)
                                            .ok_or(Error::ArithmeticOverflow)?,
                                    )?;
                                    reader.read_range(
                                        tensor,
                                        (expert as u64)
                                            .checked_mul(p.expert_bytes)
                                            .ok_or(Error::ArithmeticOverflow)?,
                                        target,
                                    )
                                })
                                .map_err(|error| match error {
                                    parallel_reads::Failure::Read(error) => error,
                                    parallel_reads::Failure::Geometry => Error::InvalidBufferLength,
                                    parallel_reads::Failure::Spawn => {
                                        Error::LimitExceeded("global HOST read workers")
                                    }
                                })?;
                                reader.verify_file_unchanged()?;
                            }
                            #[cfg(not(feature = "strata-parallel-loading"))]
                            {
                                for ordinal in 0..p.count {
                                    let expert = l.expert_at(p.first + ordinal)?;
                                    let bytes = usize::try_from(p.expert_bytes)
                                        .map_err(|_| Error::ArithmeticOverflow)?;
                                    let target = window
                                        .get_mut(..bytes)
                                        .ok_or(Error::InvalidBufferLength)?;
                                    tensor.read_range(
                                        (expert as u64)
                                            .checked_mul(p.expert_bytes)
                                            .ok_or(Error::ArithmeticOverflow)?,
                                        target,
                                    )?;
                                    copy_range(
                                        mapped,
                                        p.logical_bytes as usize,
                                        ordinal as u64 * p.expert_bytes,
                                        target,
                                    )
                                    .map_err(Error::InvalidFormat)?;
                                    target.zeroize();
                                }
                            }
                            Ok(())
                        },
                    )?;
                }
            }
        }
        }
        let embedding = graph.tensor(crate::strata_shared_weights::Tensor::Embedding.name())?;
        crate::strata_shared_weights::Weights::validate_graph(graph)?;
        for (piece, tile) in
            crate::strata_shared_weights::tiles(crate::strata_shared_weights::Tensor::Embedding)
                .into_iter()
                .enumerate()
        {
            host.allocate_original_page(
                shadow::PageSpec {
                    layer: EMBEDDING_LAYER,
                    role: 0,
                    piece,
                    shard: embedding.info().shard_index(),
                    logical: tile.owned as u64,
                    backing_bound: envelope(tile.owned as u64)?,
                },
                &mut charges,
                |mapped| {
                    let bytes = mapped
                        .get_mut(..tile.owned)
                        .ok_or(Error::InvalidBufferLength)?;
                    embedding.read_range(
                        (tile.first * crate::strata_shared_weights::Tensor::Embedding.row_bytes())
                            as u64,
                        &mut bytes[..tile.logical],
                    )?;
                    crate::strata_shared_weights::validate_payload(
                        crate::strata_shared_weights::Tensor::Embedding,
                        bytes,
                        tile,
                    )
                },
            )?;
        }
        if host.pages.len() != count
            || host.logical
                != plan
                    .host_registered_bytes()
                    .checked_add(crate::strata_shared_weights::embedding_bytes())
                    .ok_or(Error::ArithmeticOverflow)?
            || host.physical > pool_bytes
        {
            return Err(fail());
        }
        graph.verify_unchanged()?;
        Ok(host)
    }
    /// Extend a separately PRECHARGED loan-shadow reservation. This changes
    /// registry metadata only; backing allocation/initialization/registration
    /// follows under exclusive Sources ownership, never an inferred free credit.
    /// # Safety
    /// Caller holds `reserved` extra backing AND layout metadata through all readers
    /// and final detach, excludes every source/CPU consumer, and uses the owning
    /// protected same-worker stream. This is not a general inference API.
    pub(crate) unsafe fn reserve_loan_shadow(
        &mut self,
        global: &GlobalPlan,
        layout: &shadow_plan::Plan,
        reserved: u64,
    ) -> KResult<()> {
        layout.check(global)?;
        if self.closed
            || self.instance != global.instance()
            || self.cpu_borrow.load(std::sync::atomic::Ordering::Acquire)
            || layout.reservation()? != reserved
        {
            return Err(fail());
        }
        let old = self.registry_budget;
        let next = PublicExternalBudget {
            backing_bytes: old
                .map_or(0, |b| b.backing_bytes)
                .checked_add(layout.backing_bytes())
                .ok_or(Error::ArithmeticOverflow)?,
            buffers: old
                .map_or(0, |b| b.buffers)
                .checked_add(layout.pages().len())
                .ok_or(Error::ArithmeticOverflow)?,
            metadata_bytes: METADATA_BYTES,
        }
        .validate()
        .map_err(Error::InvalidFormat)?;
        if old.is_some() {
            self.client.extend_public_external_host(next)
        } else {
            self.client.configure_public_external_host(next)
        }
        .map_err(|_| fail())?;
        self.registry_budget = Some(next);
        Ok(())
    }
    pub(crate) fn role(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &GlobalPlan,
        layer: usize,
        expert: usize,
        role: usize,
    ) -> KResult<Option<(cubecl_runtime::server::Handle, u64, u64)>> {
        plan.check_binding(graph)?;
        if self.closed || self.instance != plan.instance() || self.packed != plan.has_host_blobs() {
            return Err(fail());
        }
        let GlobalSource::Host {
            piece,
            offset,
            bytes,
        } = plan.source(layer, expert, role)?
        else {
            return Ok(None);
        };
        let role=self.base_role(layer,role)?;
        let p = self
            .pages
            .iter()
            .find(|p| (p.layer, p.role, p.piece) == (layer, role, piece))
            .ok_or_else(fail)?;
        if p.owner.phase != Phase::Registered
            || offset
                .checked_add(bytes)
                .is_none_or(|n| n > p.owner.logical)
        {
            return Err(fail());
        }
        Ok(Some((
            p.authority.as_ref().ok_or_else(fail)?.handle(),
            offset,
            bytes,
        )))
    }
    pub(crate) fn embedding(&self) -> KResult<Embedding> {
        if self.closed {
            return Err(fail());
        }
        let mut pages = Vec::new();
        for (piece, tile) in
            crate::strata_shared_weights::tiles(crate::strata_shared_weights::Tensor::Embedding)
                .into_iter()
                .enumerate()
        {
            let page = self
                .pages
                .iter()
                .find(|p| (p.layer, p.role, p.piece) == (EMBEDDING_LAYER, 0, piece))
                .ok_or_else(fail)?;
            if page.owner.phase != Phase::Registered || page.owner.logical != tile.owned as u64 {
                return Err(fail());
            }
            pages.push(crate::strata_shared_weights::Page {
                handle: page.authority.as_ref().ok_or_else(fail)?.handle(),
                tile,
            });
        }
        Ok(Embedding {
            pages,
            _readers: self.embedding_readers.clone(),
        })
    }
    pub(crate) fn handles(&self) -> KResult<Vec<cubecl_runtime::server::Handle>> {
        self.pages
            .iter()
            .map(|p| {
                p.authority
                    .as_ref()
                    .map(PublicExternalHandle::handle)
                    .ok_or_else(fail)
            })
            .collect()
    }
    pub(crate) fn close(mut self) -> KResult<(u64, usize)> {
        self.close_inner()?;
        Ok((self.physical, self.pages.len()))
    }
    fn close_inner(&mut self) -> KResult<()> {
        if self.closed {
            return Ok(());
        }
        // Shared target/MTP consumers must release the original embedding loan
        // before any registry detach, buffer destruction or HOST backing free.
        if !embedding_detach_ready(&self.embedding_readers) {
            return Err(fail());
        }
        for p in &mut self.pages {
            if p.owner.phase == Phase::Registered {
                let binding = p.owner.binding.as_ref().ok_or_else(fail)?;
                let authority = p.authority.take().ok_or_else(fail)?;
                let receipt = self
                    .client
                    .detach_public_external_host(authority)
                    .map_err(|_| fail())?;
                let resource = receipt.resource();
                validate_detach(
                    p.owner.phase,
                    p.owner.logical,
                    p.owner.backing.as_ref().ok_or_else(fail)?.bytes,
                    receipt.matches_source(binding),
                    DetachShape {
                        offset: resource.offset,
                        size: resource.size,
                        buffer_bytes: receipt.buffer_bytes,
                        backing_bytes: receipt.backing_bytes,
                        joined: receipt.queue_joined,
                        destroyed: receipt.source_destroyed,
                    },
                )
                .map_err(Error::InvalidFormat)?;
                if p.owner.buffer.as_ref() != Some(&resource.buffer) {
                    return Err(fail());
                }
                p.owner.phase = Phase::Detached;
            } else if !matches!(
                p.owner.phase,
                Phase::Sealed | Phase::Detached | Phase::Released
            ) {
                return Err(fail());
            }
        }
        join(&self.device).map_err(Error::InvalidFormat)?;
        for p in &mut self.pages {
            if let Some(b) = &p.owner.backing {
                let mut mapped = b.map().map_err(Error::InvalidFormat)?;
                if !mapped.zero_and_audit() {
                    return Err(fail());
                }
                mapped
                    .cache(p.owner.placement.coherent(), true)
                    .map_err(Error::InvalidFormat)?;
            }
            p.owner.phase = Phase::Released;
        }
        self.closed = true;
        Ok(())
    }
}

pub(crate) struct CpuLayer<'a> {
    host: &'a Host,
    layer: usize,
    maps: Vec<(usize, usize, ReadOnlyMapped<'a>)>,
}
impl CpuLayer<'_> {
    pub(crate) fn role(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &GlobalPlan,
        layer: usize,
        cell: usize,
        role: usize,
    ) -> KResult<&[u8]> {
        plan.check_binding(graph)?;
        if layer != self.layer || self.host.instance != plan.instance() {
            return Err(fail());
        }
        let GlobalSource::Host {
            piece,
            offset,
            bytes,
        } = plan.source(layer, cell, role)?
        else {
            return Err(fail());
        };
        let role=self.host.base_role(layer,role)?;
        let map = self
            .maps
            .iter()
            .find(|(r, p, _)| *r == role && *p == piece)
            .ok_or_else(fail)?;
        map.2.span(offset, bytes).map_err(Error::InvalidFormat)
    }
}
impl Drop for CpuLayer<'_> {
    fn drop(&mut self) {
        self.maps.clear();
        self.host
            .cpu_borrow
            .store(false, std::sync::atomic::Ordering::Release);
    }
}
impl Drop for Host {
    fn drop(&mut self) {
        if self.close_inner().is_err() {
            // A failed or unknown detach never permits freeing potentially live
            // BDA backing, including pages whose authority was consumed on error.
            for p in &mut self.pages {
                if let Some(b) = p.owner.buffer.take() {
                    std::mem::forget(b);
                }
                if let Some(b) = p.owner.backing.take() {
                    std::mem::forget(b);
                }
            }
        }
    }
}

#[cfg(test)]
mod loan_registry_tests {
    use cubecl_common::stream_id::StreamId;
    use cubecl_runtime::public_external::*;
    fn budget(buffers: usize) -> PublicExternalBudget {
        PublicExternalBudget {
            backing_bytes: (buffers * 4096) as u64,
            buffers,
            metadata_bytes: MAX_METADATA_BYTES,
        }
    }
    fn source(id: u32) -> PublicExternalHostSource<u32> {
        // Synthetic immutable public integers; no device/memory is constructed.
        unsafe {
            PublicExternalHostSource::from_initialized_public_host(id, 64, 4096, 2, 0, [1; 32])
        }
        .unwrap()
    }
    #[test]
    fn shadow_budget_extension_keeps_live_and_retired_identity_and_all_counters() {
        let mut pool = PublicExternalPool::new(budget(2), StreamId { value: 84 }).unwrap();
        let retired = pool.register(source(1)).unwrap();
        let retired_binding = retired.handle().binding();
        pool.begin_detach(&retired).unwrap();
        // No real backing or users in this synthetic fixture.
        unsafe { pool.finish_detach(retired) }.unwrap();
        let live = pool.register(source(2)).unwrap();
        let binding = live.handle().binding();
        let before = pool.usage();
        pool.extend_budget(budget(4)).unwrap();
        let after = pool.usage();
        assert_eq!(
            after.registered_backing_bytes,
            before.registered_backing_bytes
        );
        assert_eq!(after.total_registered_bytes, before.total_registered_bytes);
        assert_eq!(after.total_detached_bytes, before.total_detached_bytes);
        assert_eq!(after.registration_count, before.registration_count);
        assert_eq!(pool.resolve(&binding).unwrap(), 2);
        assert!(pool.resolve(&retired_binding).is_err());
        let new = pool.register(source(3)).unwrap();
        assert_eq!(pool.usage().registration_count, 3);
        assert_eq!(pool.resolve(&new.handle().binding()).unwrap(), 3);
        assert_ne!(
            new.handle().binding().memory.allocation_id(),
            retired_binding.memory.allocation_id()
        );
        assert!(pool.resolve(&retired_binding).is_err());
    }
    #[test]
    fn joined_model_rebase_revokes_only_selected_roots_and_preserves_identity() {
        let mut pool = PublicExternalPool::new(budget(3), StreamId { value: 78 }).unwrap();
        let first = pool.register(source(1)).unwrap();
        let second = pool.register(source(2)).unwrap();
        let bindings = [first.handle().binding(), second.handle().binding()];
        let before = pool.usage();
        assert!(
            pool.begin_rebase(vec![bindings[0].clone(), bindings[0].clone()])
                .is_err()
        );
        assert_eq!(pool.usage(), before);
        assert_eq!(pool.resolve(&bindings[0]).unwrap(), 1);
        let token = pool.begin_rebase(vec![bindings[0].clone()]).unwrap();
        assert!(pool.resolve(&bindings[0]).is_err());
        assert_eq!(pool.resolve(&bindings[1]).unwrap(), 2);
        assert!(pool.begin_detach(&first).is_err());
        assert!(pool.extend_budget(budget(4)).is_err());
        assert!(pool.begin_rebase(vec![bindings[1].clone()]).is_err());
        assert!(pool.usage().metadata_owned_bytes > before.metadata_owned_bytes);
        // Synthetic immutable integer values only; no device/memory mutated.
        unsafe { pool.finish_rebase(token) }.unwrap();
        assert_eq!(pool.resolve(&bindings[0]).unwrap(), 1);
        assert_eq!(pool.usage(), before);
        let token = pool
            .begin_rebase(vec![bindings[0].clone(), bindings[1].clone()])
            .unwrap();
        drop(token); // no successful publication authority remains
        assert!(pool.resolve(&bindings[0]).is_err());
        assert!(pool.resolve(&bindings[1]).is_err());
        assert!(pool.begin_detach(&second).is_err());
    }
    #[test]
    fn shadow_extension_rejects_shrinking_closing_and_unquoted_metadata_peaks() {
        let mut pool = PublicExternalPool::new(budget(2), StreamId { value: 83 }).unwrap();
        let live = pool.register(source(4)).unwrap();
        let binding = live.handle().binding();
        let before = pool.usage();
        assert!(pool.extend_budget(budget(1)).is_err());
        assert_eq!(pool.usage(), before);
        assert_eq!(pool.resolve(&binding).unwrap(), 4);
        pool.begin_detach(&live).unwrap();
        let closing = pool.usage();
        assert!(pool.extend_budget(budget(4)).is_err());
        assert_eq!(pool.usage(), closing);
        unsafe { pool.finish_detach(live) }.unwrap();
        pool.extend_budget(budget(4)).unwrap();
        let small = PublicExternalBudget {
            metadata_bytes: before.metadata_owned_bytes + 256,
            ..budget(2)
        };
        let mut pool = PublicExternalPool::new(small, StreamId { value: 82 }).unwrap();
        let live = pool.register(source(5)).unwrap();
        let binding = live.handle().binding();
        let before = pool.usage();
        assert!(
            pool.extend_budget(PublicExternalBudget {
                metadata_bytes: small.metadata_bytes,
                ..budget(MAX_BUFFERS)
            })
            .is_err()
        );
        assert_eq!(pool.usage(), before);
        assert_eq!(pool.resolve(&binding).unwrap(), 5);
    }
}

#[cfg(test)]
mod embedding_owner_tests {
    use super::*;
    #[test]
    fn shared_embedding_loan_must_end_before_owning_host_detach() {
        let owner = Arc::new(());
        assert!(embedding_detach_ready(&owner));
        let first = Embedding {
            pages: Vec::new(),
            _readers: owner.clone(),
        };
        let second = Embedding {
            pages: Vec::new(),
            _readers: owner.clone(),
        };
        assert!(!embedding_detach_ready(&owner));
        drop(first);
        assert!(!embedding_detach_ready(&owner));
        drop(second);
        assert!(embedding_detach_ready(&owner));
        assert_ne!(EMBEDDING_LAYER, crate::strata_global_plan::LAYERS);
        assert!(EMBEDDING_LAYER > 47); // cannot alias expert or optional shadow tags
    }
}

#[cfg(test)]
mod blob_mapping_tests {
 use super::*;
 #[test]
 fn host_blob_cpu_map_key_deduplicates_base_roles_without_shadow_or_embedding_aliases(){
  for layer in 0..48 {for role in 0..3 {
   assert_eq!(base_page_role(false,layer,role).unwrap(),role);
   assert_eq!(base_page_role(true,layer,role).unwrap(),3);
  }}
  for (l,r) in [(48,0),(49,0),(0,3),(usize::MAX,0)]{assert!(base_page_role(true,l,r).is_err());assert!(base_page_role(false,l,r).is_err());}
 }
}
