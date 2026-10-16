//! One owner for the ranked GPU cache, original HOST complement and per-expert
//! physical address tables. No legacy uniform bank may coexist with this owner.
use crate::{
    public_host_expert_mirror::strata_global::Host,
    strata_global_gpu::Cache,
    strata_global_plan::{EXPERTS, LAYERS, Plan, Source},
};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::{Handle, KernelArguments};
use kuno_text::{Error, Result, model::BoundGraph};
use std::collections::HashMap;

#[path = "strata_cache_placement.rs"]
mod placement;
#[path = "strata_prompt_loan.rs"]
mod prompt_loan;
pub(crate) use placement::Placement;
pub(crate) use prompt_loan::BorrowedPrompt;
pub(crate) const PLACEMENT_HOST_BYTES: u64 = placement::HOST_BYTES;
pub(crate) const TABLE_BYTES: u64 = 32768;
pub(crate) const OWNED_TABLE_BYTES: u64 = TABLE_BYTES * LAYERS as u64;
fn reject() -> Error {
    Error::InvalidFormat("Strata global source owner")
}

/// One real completion gate for the source lending transaction. Prefix/tail
/// mismatch, clear/restore errors and panics cannot reopen model consumers.
fn finish_borrowed_roots(
    expected: crate::private_arena::OwnedZeroAudit,
    erase: impl FnOnce() -> Result<crate::private_arena::OwnedZeroAudit>,
    restore: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if expected.buffers != 3 || expected.bytes < 3 * crate::private_arena::MIN_SPAN_BYTES as u64 {
        return Err(reject());
    }
    let erased = std::panic::catch_unwind(std::panic::AssertUnwindSafe(erase));
    if !matches!(erased,Ok(Ok(receipt)) if receipt==expected) {
        return Err(reject());
    }
    let restored = std::panic::catch_unwind(std::panic::AssertUnwindSafe(restore));
    if !matches!(restored, Ok(Ok(()))) {
        return Err(reject());
    }
    Ok(())
}
pub(crate) struct Sources {
    client: ComputeClient<WgpuRuntime>,
    device: wgpu::Device,
    plan: Plan,
    gpu: Option<Cache>,
    host: Option<Host>,
    tables: Vec<Handle>,
    prompt_stage: Option<Handle>,
    reads: Vec<Vec<Handle>>,
    closed: bool,
    closing_failed: bool,
    mutation_pending: bool,
    loan_pending: bool,
    loan_ready: bool,
    placement: Placement,
    pending_placement: Option<Placement>,
    exchange_inventory: Option<
        std::sync::Arc<crate::public_host_expert_mirror::strata_global::exchange::Inventory>,
    >,
    retired: std::sync::Arc<std::sync::atomic::AtomicBool>,
    queue: Option<wgpu::Queue>,
}
impl Sources {
    fn unavailable(&self) -> bool {
        self.closed
            || self.closing_failed
            || self.retired.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn gpu_requests(plan: &Plan) -> Vec<u64> {
        let mut requests = Cache::requests(plan);
        requests.extend([TABLE_BYTES; LAYERS]);
        requests
    }
    /// # Safety
    /// Caller authenticates the original public shards and reserves the complete
    /// HOST and exact GPU budgets before entering exclusive persistent admission.
    /// This source client/device is the sole consumer stream. No legacy banks or
    /// previous public external registry exist on this client.
    pub(crate) unsafe fn load(
        graph: BoundGraph<'_, '_>,
        plan: Plan,
        client: &ComputeClient<WgpuRuntime>,
        device: &wgpu::Device,
        host_reserved: u64,
        retired: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self> {
        if !crate::strata_intel_core::enabled()
            || host_reserved != Host::reservation(&plan)?
            || retired.load(std::sync::atomic::Ordering::Acquire)
            || client.public_external_host_usage().is_ok()
        {
            return Err(reject());
        }
        plan.check_binding(graph)?;
        let mut sources = Self {
            client: client.clone(),
            device: device.clone(),
            plan,
            gpu: None,
            host: None,
            tables: Vec::with_capacity(LAYERS),
            prompt_stage: None,
            reads: Vec::with_capacity(LAYERS),
            closed: false,
            closing_failed: false,
            mutation_pending: false,
            loan_pending: false,
            loan_ready: false,
            placement: Placement::identity(),
            pending_placement: None,
            exchange_inventory: None,
            retired,
            queue: None,
        };
        // Incremental ownership makes every construction failure take the same
        // completed-reader closure path as a fully constructed source set.
        sources.gpu = Some(Cache::load(client, graph, &sources.plan)?);
        sources.host =
            Some(unsafe { Host::load(&sources.plan, graph, client, device, host_reserved)? });
        sources.build_tables(graph, device)?;
        if crate::strata_global_adapt::collect_gpu_route_usage(true) {
            sources.prepare_exchange_inventory(graph)?;
        }
        graph.verify_unchanged()?;
        Ok(sources)
    }
    pub(crate) fn embedding(
        &self,
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Result<crate::public_host_expert_mirror::strata_global::Embedding> {
        self.plan.check_binding(graph)?;
        if !std::ptr::eq(client.info(), self.client.info()) || self.unavailable() {
            return Err(reject());
        }
        self.host.as_ref().ok_or_else(reject)?.embedding()
    }
    pub(crate) fn embedding_table(
        &self,
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Result<Handle> {
        self.plan.check_binding(graph)?;
        if !std::ptr::eq(client.info(), self.client.info())
            || self.unavailable()
            || self.tables.len() != LAYERS
        {
            return Err(reject());
        }
        self.tables.first().cloned().ok_or_else(reject)
    }
    fn build_tables(&mut self, graph: BoundGraph<'_, '_>, device: &wgpu::Device) -> Result<()> {
        let gpu = self.gpu.as_ref().ok_or_else(reject)?;
        let host = self.host.as_ref().ok_or_else(reject)?;
        let excluded = if self.loan_pending {
            Some(host.loan_shadow_indices().ok_or_else(reject)?)
        } else {
            None
        };
        let mut handles = gpu
            .roots()
            .iter()
            .enumerate()
            .filter(|(i, _)| excluded.is_none_or(|indices| !indices.contains(i)))
            .map(|(_, h)| h.clone())
            .collect::<Vec<_>>();
        handles.extend(host.handles()?);
        let resources = crate::resident_util::retained_resources(&self.client, handles.clone())?;
        if resources.len() != handles.len() {
            return Err(reject());
        }
        let dev = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
        let mut roots = HashMap::with_capacity(handles.len());
        let mut physical = std::collections::HashSet::with_capacity(handles.len());
        for (handle, resource) in handles.iter().zip(&resources) {
            let r = resource.resource();
            if handle.stream != handles.first().ok_or_else(reject)?.stream
                || handle.offset_start.unwrap_or(0) != 0
                || handle.offset_end.unwrap_or(0) != 0
                || r.offset != 0
                || r.size != r.buffer.size()
                || r.size != handle.size_in_used()
            {
                return Err(reject());
            }
            let raw = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
            if !physical.insert(unsafe { raw.raw_buffer_handle() }) {
                return Err(reject());
            }
            let address = unsafe {
                dev.raw_device().get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::default().buffer(raw.raw_buffer_handle()),
                )
            };
            if address == 0
                || !address.is_multiple_of(16)
                || roots
                    .insert(
                        handle.clone().binding().memory.allocation_id(),
                        (address, r.size, handle.clone()),
                    )
                    .is_some()
            {
                return Err(reject());
            }
        }
        for layer in 0..LAYERS {
            let mut words = vec![0u32; (TABLE_BYTES / 4) as usize];
            let mut reads = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for role in 0..3 {
                for expert in 0..EXPERTS {
                    let (handle, offset, bytes, tier) =
                        self.effective_role(graph, layer, expert, role)?;
                    let id = handle.binding().memory.allocation_id();
                    let (base, size, root) = roots.get(&id).ok_or_else(reject)?;
                    if bytes == 0
                        || !bytes.is_multiple_of(4)
                        || offset.checked_add(bytes).is_none_or(|n| n > *size)
                    {
                        return Err(reject());
                    }
                    let address = base.checked_add(offset).ok_or(Error::ArithmeticOverflow)?;
                    if !address.is_multiple_of(16) {
                        return Err(reject());
                    }
                    let at = (role * EXPERTS + expert) * 4;
                    words[at..at + 4].copy_from_slice(&[
                        address as u32,
                        (address >> 32) as u32,
                        u32::try_from(bytes / 4).map_err(|_| Error::ArithmeticOverflow)?,
                        tier,
                    ]);
                    if seen.insert(id) {
                        reads.push(root.clone());
                    }
                }
            }
            if layer == 0 {
                let embedding = host.embedding()?;
                let mut pages = Vec::with_capacity(embedding.pages.len());
                for page in &embedding.pages {
                    let id = page.handle.clone().binding().memory.allocation_id();
                    let &(base, size, _) = roots.get(&id).ok_or_else(reject)?;
                    pages.push((base, size));
                }
                crate::strata_shared_embedding::encode(&mut words, &pages)?;
            }
            let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let table = if let Some(table) = self.tables.get(layer) {
                self.client
                    .write_public_owned_batch(vec![(table.clone(), bytes)])
                    .map_err(crate::resident_util::fault)?;
                table.clone()
            } else {
                self.client.create_public_owned(bytes)
            };
            let resource = self
                .client
                .retained_resource_metadata(vec![table.clone()])
                .map_err(crate::resident_util::fault)?;
            if table.size_in_used() != TABLE_BYTES
                || resource.len() != 1
                || resource[0].resource().offset != 0
                || resource[0].resource().buffer.size() != TABLE_BYTES
            {
                return Err(reject());
            }
            if self.tables.len() == layer {
                self.tables.push(table);
            }
            if self.reads.len() == layer {
                self.reads.push(reads);
            } else {
                self.reads[layer] = reads;
            }
        }
        Ok(())
    }
    fn effective_role(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        cell: usize,
        role: usize,
    ) -> Result<(Handle, u64, u64, u32)> {
        let host = self.host.as_ref().ok_or_else(reject)?;
        if self.loan_pending
            && let Some((h, o, b)) = host.loan_shadow_role(graph, &self.plan, layer, cell, role)?
        {
            return Ok((h, o, b, 1));
        }
        let (source, tier) = match self.plan.source(layer, cell, role)? {
            Source::Resident { .. } => (
                self.gpu
                    .as_ref()
                    .ok_or_else(reject)?
                    .resident_role(graph, &self.plan, layer, cell, role)?,
                0,
            ),
            Source::Host { .. } => (host.role(graph, &self.plan, layer, cell, role)?, 1),
        };
        let (h, o, b) = source.ok_or_else(reject)?;
        Ok((h, o, b, tier))
    }
    pub(crate) fn validate_prompt_ready(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        self.validate_execution_attached()?;
        self.plan.check_binding(graph)
    }
    pub(crate) fn prototype(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        role: usize,
    ) -> Result<Handle> {
        self.validate_prompt_ready(graph)?;
        Ok(self.effective_role(graph, layer, 0, role)?.0)
    }
    /// The selected NO_HOST decode needs immutable source cells and the
    /// actual execution queue, not dormant per-role fetch stages/address tables.
    /// Caller uses the same authenticated device/queue/client as source load.
    pub(crate) fn attach_execution_queue(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        if self.unavailable()
            || self.queue.is_some()
            || self.mutation_pending
            || self.loan_pending
            || self.gpu.is_none()
            || self.host.is_none()
            || self.tables.len() != LAYERS
            || self.reads.len() != LAYERS
            || self.reads.iter().any(Vec::is_empty)
        {
            return Err(reject());
        }
        let actual = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
        let original =
            unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
        if actual.raw_device().handle() != original.raw_device().handle() {
            return Err(reject());
        }
        self.queue = Some(queue.clone());
        Ok(())
    }
    pub(crate) fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Attach the real separately priced native-owned prompt raw root. This
    /// retains its actual raw-root dependency independently of direct decode.
    pub(crate) fn attach_prompt_stage(&mut self, root: Handle) -> Result<()> {
        self.validate_attached()?;
        if self.prompt_stage.is_some()
            || root.stream != self.tables.first().ok_or_else(reject)?.stream
            || root.offset_start.unwrap_or(0) != 0
            || root.offset_end.unwrap_or(0) != 0
        {
            return Err(reject());
        }
        // This owner is now the raw DMA ring only. MoE scratch is in the
        // separate phase overlay; it must not fabricate a larger raw parent.
        crate::strata_prompt::allocation::whole_root_bytes(&root)?;
        if root.size_in_used() < crate::strata_prompt::allocation::RAW_RING_BYTES {
            return Err(reject());
        }
        let resources = self
            .client
            .retained_resource_metadata(vec![root.clone()])
            .map_err(crate::resident_util::fault)?;
        if resources.len() != 1
            || resources[0].resource().offset != 0
            || resources[0].resource().size != root.size_in_used()
            || resources[0].resource().buffer.size() != root.size_in_used()
        {
            return Err(reject());
        }
        self.prompt_stage = Some(root);
        Ok(())
    }
    /// Intel prefill's original HOST -> staging transfer. Resident cells remain
    /// in place. The caller exclusively owns this private ring until all queued
    /// consumers finish; copies and kernels use the same checked owning stream.
    /// No request-derived offset, cell ID or source content is logged.
    pub(crate) fn stage_prompt_group(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        cell: usize,
        role_bytes: [usize; 3],
        destination: &Handle,
        first: usize,
    ) -> Result<()> {
        self.validate_prompt_ready(graph)?;
        let original_stage = self.prompt_stage.as_ref().ok_or_else(reject)?;
        if destination.clone().binding().memory.allocation_id()
            != original_stage.clone().binding().memory.allocation_id()
            || destination.stream != original_stage.stream
            || cell >= EXPERTS
            || destination.size_in_used() != original_stage.size_in_used()
            || destination.offset_start.unwrap_or(0) != 0
            || destination.offset_end.unwrap_or(0) != 0
            || !first.is_multiple_of(16)
            || role_bytes.iter().any(|&b| b == 0 || !b.is_multiple_of(16))
        {
            return Err(reject());
        }
        let total = role_bytes.iter().try_fold(0usize, |a, &b| {
            a.checked_add(b).ok_or(Error::ArithmeticOverflow)
        })?;
        if first
            .checked_add(total)
            .is_none_or(|end| end > crate::strata_prompt::allocation::RAW_RING_BYTES as usize)
        {
            return Err(reject());
        }
        let mut copies = Vec::with_capacity(3);
        let mut offset = first as u64;
        for (role, &expected) in role_bytes.iter().enumerate() {
            let (handle, source_first, bytes, tier) =
                self.effective_role(graph, layer, cell, role)?;
            if tier == 1 {
                if bytes != expected as u64 || handle.stream != destination.stream {
                    return Err(reject());
                }
                copies.push(cubecl_runtime::public_external::PublicHostCopy {
                    source: handle.binding(),
                    source_first,
                    destination: destination.clone().binding(),
                    destination_first: offset,
                    bytes,
                });
            }
            offset = offset
                .checked_add(expected as u64)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        if !copies.is_empty() {
            self.client
                .stage_public_host_copies(copies)
                .map_err(crate::resident_util::fault)?;
        }
        Ok(())
    }
    /// Actual attachment authority, independent of the retired frozen-bank
    /// receipts. A half-published exchange cannot authorize another window.
    pub(crate) fn validate_attached(&self) -> Result<()> {
        if self.unavailable()
            || self.mutation_pending
            || self.loan_pending
            || self.gpu.is_none()
            || self.host.is_none()
            || self.queue.is_none()
            || self.tables.len() != LAYERS
            || self.reads.len() != LAYERS
            || self.reads.iter().any(Vec::is_empty)
        {
            return Err(reject());
        }
        Ok(())
    }
    /// Readonly scheduler admission for the active prompt Scope. Decode binds
    /// remain separately forbidden while cache roots hold private scratch.
    pub(crate) fn validate_execution_attached(&self) -> Result<()> {
        if !self.loan_pending {
            return self.validate_attached();
        }
        if self.unavailable()
            || !self.loan_ready
            || !self.mutation_pending
            || !crate::strata_prompt::active()
            || self.prompt_stage.is_none()
            || self.tables.len() != LAYERS
            || self.reads.len() != LAYERS
            || self.gpu.is_none()
            || self.host.is_none()
            || self.queue.is_none()
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn validate_execution_boundary(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        if self.loan_pending {
            self.validate_prompt_ready(graph)
        } else {
            self.validate_graph_ready(graph)
        }
    }
    pub(crate) fn validate_graph_ready(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        self.validate_attached()?;
        self.plan.check_binding(graph)
    }
    pub(crate) fn retained_route_map(
        &self,
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Result<zeroize::Zeroizing<Vec<u16>>> {
        // Full attachment rejects partial publication, loans and retirement.
        self.validate_graph_ready(graph)?;
        if !std::ptr::eq(client.info(), self.client.info()) {
            return Err(reject());
        }
        self.placement.route_map()
    }
    pub(crate) fn borrow_cpu(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        cells: &[usize],
    ) -> Result<crate::public_host_expert_mirror::strata_global::CpuLayer<'_>> {
        self.validate_graph_ready(graph)?;
        // All HOST writes joined before source publication. Queued inference
        // consumers only read these roots, so mapping must not drain GPU work.
        self.host
            .as_ref()
            .ok_or_else(reject)?
            .borrow_cpu(graph, &self.plan, layer, cells)
    }
    /// Bind actual immutable cell addresses and every original indirect read.
    /// Private logical remaps and dense input planes have their own real owners.
    pub(crate) fn bind(&self, layer: usize, args: KernelArguments) -> Result<KernelArguments> {
        if self.unavailable() || self.mutation_pending || self.loan_pending {
            return Err(reject());
        }
        let table = self.tables.get(layer).ok_or_else(reject)?;
        let reads = self.reads.get(layer).ok_or_else(reject)?;
        Ok(args
            .with_buffer(table.clone().binding())
            .with_read_dependencies(reads.iter().cloned().map(Handle::binding).collect()))
    }
    /// Prompt-image dequantization has no fetch-stage pointer table or pointer
    /// reads. Its real raw-slot descriptor is tracked separately. This binder
    /// is only for that closed shader family, whose interface tracks its actual raw-ring owner.
    pub(crate) fn bind_prompt_image(
        &self,
        layer: usize,
        args: KernelArguments,
    ) -> Result<KernelArguments> {
        if self.loan_pending {
            if self.unavailable()
                || !self.loan_ready
                || !self.mutation_pending
                || !crate::strata_prompt::active()
            {
                return Err(reject());
            }
        } else if self.unavailable() || self.mutation_pending {
            return Err(reject());
        }
        if self.prompt_stage.is_none() {
            return Err(reject());
        }
        let table = self.tables.get(layer).ok_or_else(reject)?;
        let reads = self.reads.get(layer).ok_or_else(reject)?;
        Ok(args
            .with_buffer(table.clone().binding())
            .with_read_dependencies(reads.iter().cloned().map(Handle::binding).collect()))
    }
    /// Authenticate static physical cells ONCE on the sole client thread.
    /// No raw source pointer or allocator/admission capability crosses threads.
    fn prepare_exchange_inventory(&mut self, graph: BoundGraph<'_, '_>) -> Result<()> {
        if self.exchange_inventory.is_some() {
            return Ok(());
        }
        use crate::public_host_expert_mirror::strata_global::exchange::{Cell, Inventory};
        let mut cells = Vec::with_capacity(LAYERS * EXPERTS * 3);
        let mut resources = HashMap::new();
        for layer in 0..LAYERS {
            for cell in 0..EXPERTS {
                for role in 0..3 {
                    let (gpu, (handle, offset, bytes)) = match self
                        .gpu
                        .as_ref()
                        .ok_or_else(reject)?
                        .resident_role(graph, &self.plan, layer, cell, role)?
                    {
                        Some(r) => (true, r),
                        None => (
                            false,
                            self.host
                                .as_ref()
                                .ok_or_else(reject)?
                                .role(graph, &self.plan, layer, cell, role)?
                                .ok_or_else(reject)?,
                        ),
                    };
                    let id = handle.clone().binding().memory.allocation_id();
                    if let std::collections::hash_map::Entry::Vacant(entry) = resources.entry(id) {
                        let mut found = self
                            .client
                            .retained_resource_metadata(vec![handle])
                            .map_err(crate::resident_util::fault)?;
                        if found.len() != 1 {
                            return Err(reject());
                        }
                        entry.insert(found.pop().ok_or_else(reject)?.resource().clone());
                    }
                    let full = resources.get(&id).ok_or_else(reject)?;
                    if full.offset != 0
                        || full.size != full.buffer.size()
                        || offset.checked_add(bytes).is_none_or(|n| n > full.size)
                    {
                        return Err(reject());
                    }
                    cells.push(Cell {
                        gpu,
                        resource: cubecl::wgpu::WgpuResource {
                            buffer: full.buffer.clone(),
                            offset,
                            size: bytes,
                        },
                    });
                }
            }
        }
        self.exchange_inventory = Some(std::sync::Arc::new(Inventory::new(
            cells,
            u64::from(self.device.limits().min_storage_buffer_offset_alignment),
        )?));
        Ok(())
    }
    /// Joined original source boundary. Revoke expert readers/pin original
    /// scratch HERE, then delegate only owned numeric/resource capabilities.
    pub(crate) fn submit_exchange(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &crate::private_arena::PrivateArena,
        scratch_id: crate::private_arena::BufferId,
        cache: crate::strata_global_adapt::Cache,
        usage: zeroize::Zeroizing<Vec<f32>>,
    ) -> Result<crate::public_host_expert_mirror::strata_global::exchange::Flight> {
        if self.unavailable()
            || self.mutation_pending
            || self.loan_pending
            || crate::strata_prompt::active()
            || self.pending_placement.is_some()
        {
            return Err(reject());
        }
        self.plan.check_binding(graph)?;
        cache.check_retained_route_map(&self.retained_route_map(graph, &self.client)?)?;
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        self.prepare_exchange_inventory(graph)?;
        let scratch_owner = self.prompt_stage.as_ref().ok_or_else(reject)?;
        let private_root = arena
            .handle(scratch_id)
            .map_err(crate::resident_util::fault)?;
        if private_root.clone().binding().memory.allocation_id()
            != scratch_owner.clone().binding().memory.allocation_id()
            || private_root.size_in_used() != scratch_owner.size_in_used()
            || scratch_owner.size_in_used() < crate::strata_prompt::allocation::RAW_RING_BYTES
        {
            return Err(reject());
        }
        let mut resources = self
            .client
            .retained_resource_metadata(vec![scratch_owner.clone()])
            .map_err(crate::resident_util::fault)?;
        if resources.len() != 1 {
            return Err(reject());
        }
        let scratch = resources.pop().ok_or_else(reject)?.resource().clone();
        if scratch.offset != 0
            || scratch.size != scratch_owner.size_in_used()
            || scratch.buffer.size() != scratch.size
            || !scratch
                .buffer
                .usage()
                .contains(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
        {
            return Err(reject());
        }
        self.mutation_pending = true; // retained until actual control/physical publication
        let mut guard = arena
            .begin_external_root_flight(scratch_id)
            .map_err(crate::resident_util::fault)?;
        let token = match self
            .host
            .as_mut()
            .ok_or_else(reject)
            .and_then(|host| host.begin_adaptive_rebase())
        {
            Ok(token) => token,
            Err(error) => {
                // No worker or scratch command exists yet. Release ONLY this
                // known-idle private root; source mutation stays terminal.
                let _ = guard.joined();
                return Err(error);
            }
        };
        crate::public_host_expert_mirror::strata_global::exchange::Flight::submit(
            &self.device,
            &self.client,
            self.exchange_inventory.as_ref().ok_or_else(reject)?.clone(),
            scratch,
            token,
            guard,
            cache,
            usage,
        )
    }
    /// Actual worker and copy fences have joined; validate the next physical
    /// placement before any private control or source-reader publication.
    pub(crate) fn complete_exchange(
        &mut self,
        flight: crate::public_host_expert_mirror::strata_global::exchange::Flight,
    ) -> Result<(
        crate::strata_global_adapt::Cache,
        crate::strata_global_adapt::Proposal,
    )> {
        if self.unavailable() || !self.mutation_pending || self.pending_placement.is_some() {
            return Err(reject());
        }
        let (cache, proposal) = flight.finish()?;
        self.pending_placement = Some(self.placement.propose(&proposal.swaps)?);
        Ok((cache, proposal))
    }
    /// Join PRIVATE remap/usage writes before admitting another source reader.
    pub(crate) fn finish_publication(&mut self, commit: impl FnOnce() -> Result<()>) -> Result<()> {
        if self.unavailable()
            || self.loan_pending
            || !self.mutation_pending
            || self.pending_placement.is_none()
        {
            return Err(reject());
        }
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        commit()?;
        self.placement = self.pending_placement.take().ok_or_else(reject)?;
        self.mutation_pending = false;
        Ok(())
    }
    /// # Safety
    /// Protected sole worker, all model reservations held; no Scope/ID/replay
    /// may escape. Model consumers are limited to the prompt-only source path.
    pub(crate) unsafe fn with_private_cache_roots<T>(
        &mut self,
        graph: BoundGraph<'_, '_>,
        indices: [usize; 3],
        body: impl FnOnce(
            &crate::private_arena::PrivateArena,
            [crate::private_arena::BufferId; 3],
        ) -> Result<T>,
    ) -> Result<T> {
        if self.host.as_ref().and_then(Host::loan_shadow_indices) != Some(indices) {
            return Err(reject());
        }
        let loan = unsafe { self.begin_prompt_loan(graph)? };
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            body(loan.arena.as_ref().ok_or_else(reject)?, loan.ids)
        }));
        self.finish_prompt_loan(graph, loan)?;
        match attempted {
            Ok(v) => v,
            Err(_) => Err(reject()),
        }
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.closing_failed || self.loan_pending {
            return Err(reject());
        }
        self.closing_failed = true;
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        // New borrows are unavailable after tables/read aliases are released;
        // detach revokes lookups, joins all scheduled/native readers and destroys
        // registered buffers before granting the HOST owner zero/free authority.
        self.tables.clear();
        self.reads.clear();
        self.prompt_stage = None;
        self.exchange_inventory = None;
        if let Some(host) = self.host.take() {
            host.close()?;
        }
        self.gpu = None;
        self.closed = true;
        self.closing_failed = false;
        Ok(())
    }
}
impl Drop for Sources {
    fn drop(&mut self) {
        if self.close().is_err() {
            // Failed queue completion grants no permission to release BDA roots.
            std::mem::forget(std::mem::take(&mut self.tables));
            std::mem::forget(std::mem::take(&mut self.reads));
            if let Some(root) = self.prompt_stage.take() {
                std::mem::forget(root);
            }
            if let Some(inventory) = self.exchange_inventory.take() {
                std::mem::forget(inventory);
            }
            if let Some(gpu) = self.gpu.take() {
                std::mem::forget(gpu);
            }
            if let Some(host) = self.host.take() {
                std::mem::forget(host);
            }
        }
    }
}

#[cfg(test)]
mod native_shared_table_tests {
    use super::*;
    #[test]
    fn native_shared_prefix_quote_includes_all_original_address_tables() {
        for budget in [1024 * 1024, 1024 * 1024 * 1024, 20 * 1024 * 1024 * 1024] {
            let plan = Plan::shadow_test_fixture(budget).unwrap();
            let requests = Sources::gpu_requests(&plan);
            let gpu = Cache::requests(&plan);
            assert_eq!(&requests[..gpu.len()], &gpu);
            assert_eq!(&requests[gpu.len()..], &[TABLE_BYTES; LAYERS]);
            assert_eq!(OWNED_TABLE_BYTES, 48 * 32768);
            assert!(gpu.iter().sum::<u64>() <= budget);
            assert!(requests.iter().sum::<u64>() <= budget + OWNED_TABLE_BYTES);
            // Tables get their own fixed charge; they cannot consume space
            // already promised to the actual ranked expert prefix.
            assert_eq!(
                requests.iter().sum::<u64>() - gpu.iter().sum::<u64>(),
                OWNED_TABLE_BYTES
            );
            if crate::strata_intel_core::enabled() {
                let mut ledger =
                    crate::main_admission::MainAdmission::new(budget + OWNED_TABLE_BYTES, 0)
                        .unwrap();
                let charged = ledger.precharge_persistent(&requests, 256).unwrap();
                assert_eq!(charged, requests.iter().sum::<u64>());
                assert_eq!(ledger.upper_bytes(), charged);
            }
        }
    }
}

#[cfg(test)]
mod borrowed_completion_tests {
    use super::*;
    use crate::private_arena::OwnedZeroAudit;
    use std::cell::RefCell;
    #[test]
    fn only_complete_erasure_then_complete_restoration_can_reopen_sources() {
        let expected = OwnedZeroAudit {
            bytes: 3 * 33024,
            buffers: 3,
        };
        for failure in [
            "none",
            "erase",
            "erase panic",
            "prefix",
            "missing root",
            "restore",
            "restore panic",
        ] {
            let order = RefCell::new(Vec::new());
            let result = finish_borrowed_roots(
                expected,
                || {
                    order.borrow_mut().push("erase");
                    match failure {
                        "erase" => Err(reject()),
                        "erase panic" => panic!("public synthetic borrowed erasure fault"),
                        "prefix" => Ok(OwnedZeroAudit {
                            bytes: expected.bytes - 256,
                            ..expected
                        }),
                        "missing root" => Ok(OwnedZeroAudit {
                            buffers: 2,
                            ..expected
                        }),
                        _ => Ok(expected),
                    }
                },
                || {
                    order.borrow_mut().push("restore");
                    match failure {
                        "restore" => Err(reject()),
                        "restore panic" => panic!("public synthetic restore fault"),
                        _ => Ok(()),
                    }
                },
            );
            assert_eq!(result.is_ok(), failure == "none");
            let expected_order = if matches!(failure, "none" | "restore" | "restore panic") {
                vec!["erase", "restore"]
            } else {
                vec!["erase"]
            };
            assert_eq!(*order.borrow(), expected_order);
        }
        assert!(
            finish_borrowed_roots(
                OwnedZeroAudit::default(),
                || panic!("must not erase"),
                || panic!("must not restore")
            )
            .is_err()
        );
    }
}
