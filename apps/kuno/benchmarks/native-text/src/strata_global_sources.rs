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

pub(crate) const TABLE_BYTES: u64 = 32768;
fn reject() -> Error {
    Error::InvalidFormat("Strata global source owner")
}

pub(crate) struct Sources {
    client: ComputeClient<WgpuRuntime>,
    plan: Plan,
    gpu: Option<Cache>,
    host: Option<Host>,
    tables: Vec<Handle>,
    stage_table: Option<Handle>,
    stages: Option<[Handle; 3]>,
    reads: Vec<Vec<Handle>>,
    closed: bool,
    closing_failed: bool,
    mutation_pending: bool,
    queue: Option<wgpu::Queue>,
}
impl Sources {
    pub(crate) fn gpu_requests(plan: &Plan) -> Vec<u64> {
        let mut requests = Cache::requests(plan);
        requests.extend([TABLE_BYTES; LAYERS + 1]);
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
    ) -> Result<Self> {
        if !crate::strata_intel_core::enabled()
            || host_reserved != Host::reservation(&plan)?
            || client.public_external_host_usage().is_ok()
        {
            return Err(reject());
        }
        plan.check_binding(graph)?;
        let mut sources = Self {
            client: client.clone(),
            plan,
            gpu: None,
            host: None,
            tables: Vec::with_capacity(LAYERS),
            stage_table: Some(client.create_public_owned(vec![0; TABLE_BYTES as usize])),
            stages: None,
            reads: Vec::with_capacity(LAYERS),
            closed: false,
            closing_failed: false,
            mutation_pending: false,
            queue: None,
        };
        // Incremental ownership makes every construction failure take the same
        // completed-reader closure path as a fully constructed source set.
        sources.gpu = Some(Cache::load(client, graph, &sources.plan)?);
        sources.host =
            Some(unsafe { Host::load(&sources.plan, graph, client, device, host_reserved)? });
        sources.build_tables(graph, device)?;
        graph.verify_unchanged()?;
        Ok(sources)
    }
    fn build_tables(&mut self, graph: BoundGraph<'_, '_>, device: &wgpu::Device) -> Result<()> {
        let gpu = self.gpu.as_ref().ok_or_else(reject)?;
        let host = self.host.as_ref().ok_or_else(reject)?;
        let mut handles = gpu.roots().to_vec();
        handles.extend(host.handles()?);
        let resources = self
            .client
            .retained_resource_metadata(handles.clone())
            .map_err(crate::resident_util::fault)?;
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
                    let source = self.plan.source(layer, expert, role)?;
                    let (handle, offset, bytes, tier) = match source {
                        Source::Resident { .. } => {
                            let (h, o, b) = gpu
                                .resident_role(graph, &self.plan, layer, expert, role)?
                                .ok_or_else(reject)?;
                            (h, o, b, 0)
                        }
                        Source::Host { .. } => {
                            let (h, o, b) = host
                                .role(graph, &self.plan, layer, expert, role)?
                                .ok_or_else(reject)?;
                            (h, o, b, 1)
                        }
                    };
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
            let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let table = self.client.create_public_owned(bytes);
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
            self.tables.push(table);
            self.reads.push(reads);
        }
        Ok(())
    }
    pub(crate) fn prototype(
        &self,
        graph: BoundGraph<'_, '_>,
        layer: usize,
        role: usize,
    ) -> Result<Handle> {
        if self.closed || self.closing_failed || self.mutation_pending {
            return Err(reject());
        }
        if let Some((h, _, _)) = self
            .gpu
            .as_ref()
            .ok_or_else(reject)?
            .resident_role(graph, &self.plan, layer, 0, role)?
        {
            Ok(h)
        } else {
            Ok(self
                .host
                .as_ref()
                .ok_or_else(reject)?
                .role(graph, &self.plan, layer, 0, role)?
                .ok_or_else(reject)?
                .0)
        }
    }
    pub(crate) fn attach_stages(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        stages: [Handle; 3],
    ) -> Result<()> {
        if self.closed || self.closing_failed || self.stages.is_some() {
            return Err(reject());
        }
        let resources = self
            .client
            .retained_resource_metadata(stages.to_vec())
            .map_err(crate::resident_util::fault)?;
        let dev = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
        let mut words = vec![0u32; TABLE_BYTES as usize / 4];
        if resources.len() != 3 {
            return Err(reject());
        }
        for (role, r) in resources.iter().enumerate() {
            let r = r.resource();
            if resources[..role]
                .iter()
                .any(|other| other.resource().buffer == r.buffer)
            {
                return Err(reject());
            }
            if r.offset != 0
                || r.size != r.buffer.size()
                || r.size != crate::strata_global_experts::STAGE_BYTES
            {
                return Err(reject());
            }
            let raw = unsafe { r.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(reject)?;
            let address = unsafe {
                dev.raw_device().get_buffer_device_address(
                    &ash::vk::BufferDeviceAddressInfo::default().buffer(raw.raw_buffer_handle()),
                )
            };
            if address == 0 || !address.is_multiple_of(16) {
                return Err(reject());
            }
            words[role * 4..role * 4 + 4].copy_from_slice(&[
                address as u32,
                (address >> 32) as u32,
                (r.size / 4) as u32,
                0,
            ]);
        }
        let bytes = words.into_iter().flat_map(u32::to_le_bytes).collect();
        self.client
            .write_public_owned_batch(vec![(
                self.stage_table.as_ref().ok_or_else(reject)?.clone(),
                bytes,
            )])
            .map_err(crate::resident_util::fault)?;
        self.stages = Some(stages);
        self.queue = Some(queue.clone());
        Ok(())
    }
    pub(crate) fn plan(&self) -> &Plan {
        &self.plan
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
        self.validate_graph_ready(graph)?;
        let original_stage = self.stages.as_ref().ok_or_else(reject)?;
        if destination.clone().binding().memory.allocation_id()
            != original_stage[2].clone().binding().memory.allocation_id()
            || destination.stream != original_stage[2].stream
            || cell >= EXPERTS
            || destination.size_in_used() != crate::strata_global_experts::STAGE_BYTES
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
            .is_none_or(|end| end as u64 > destination.size_in_used())
        {
            return Err(reject());
        }
        if (0..3).all(|role| {
            matches!(
                self.plan.source(layer, cell, role),
                Ok(Source::Resident { .. })
            )
        }) {
            return Ok(());
        }
        let mut copies = Vec::with_capacity(3);
        let mut offset = first as u64;
        for (role, &expected) in role_bytes.iter().enumerate() {
            if matches!(self.plan.source(layer, cell, role)?, Source::Host { .. }) {
                let (handle, source_first, bytes) = self
                    .host
                    .as_ref()
                    .ok_or_else(reject)?
                    .role(graph, &self.plan, layer, cell, role)?
                    .ok_or_else(reject)?;
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
        if self.closed
            || self.closing_failed
            || self.mutation_pending
            || self.gpu.is_none()
            || self.host.is_none()
            || self.queue.is_none()
            || self.stages.is_none()
            || self.stage_table.is_none()
            || self.tables.len() != LAYERS
            || self.reads.len() != LAYERS
            || self.reads.iter().any(Vec::is_empty)
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn validate_graph_ready(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        self.validate_attached()?;
        self.plan.check_binding(graph)
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
    /// Retains the actual immutable roots in native recording/fence tracking.
    /// Adds only static physical-cell/stage address descriptors. Private
    /// logical remaps stay in the family arena and never enter these tables.
    pub(crate) fn bind(&self, layer: usize, args: KernelArguments) -> Result<KernelArguments> {
        if self.closed || self.closing_failed || self.mutation_pending {
            return Err(reject());
        }
        let table = self.tables.get(layer).ok_or_else(reject)?;
        let reads = self.reads.get(layer).ok_or_else(reject)?;
        let stages = self.stages.as_ref().ok_or_else(reject)?;
        Ok(args
            .with_buffer(table.clone().binding())
            .with_buffer(
                self.stage_table
                    .as_ref()
                    .ok_or_else(reject)?
                    .clone()
                    .binding(),
            )
            .with_read_dependencies(
                reads
                    .iter()
                    .chain(stages)
                    .cloned()
                    .map(Handle::binding)
                    .collect(),
            ))
    }
    /// Prompt-image dequantization has no fetch-stage pointer table or pointer
    /// reads. Its real raw-slot descriptor is tracked separately. This binder
    /// is only for that closed shader family, whose interface omits stage_table.
    pub(crate) fn bind_prompt_image(
        &self,
        layer: usize,
        args: KernelArguments,
    ) -> Result<KernelArguments> {
        if self.closed || self.closing_failed || self.mutation_pending || self.stages.is_none() {
            return Err(reject());
        }
        let table = self.tables.get(layer).ok_or_else(reject)?;
        let reads = self.reads.get(layer).ok_or_else(reject)?;
        Ok(args
            .with_buffer(table.clone().binding())
            .with_read_dependencies(reads.iter().cloned().map(Handle::binding).collect()))
    }
    /// Consumer boundaries have completed. Verify exact initial physical cells,
    /// then execute all role exchanges as one joined owning transaction. The
    /// static tables never change; subsequent consumers await PRIVATE remap.
    pub(crate) fn exchange(
        &mut self,
        graph: BoundGraph<'_, '_>,
        swaps: &[crate::strata_global_adapt::Swap],
    ) -> Result<()> {
        if self.closed
            || self.closing_failed
            || self.mutation_pending
            || swaps.len() > crate::strata_global_adapt::MAX_SWAPS
        {
            return Err(reject());
        }
        self.plan.check_binding(graph)?;
        if swaps.is_empty() {
            self.mutation_pending = true;
            return Ok(());
        }
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        let mut pairs = Vec::with_capacity(swaps.len() * 3);
        for swap in swaps {
            for role in 0..3 {
                let (gpu, go, gb) = self
                    .gpu
                    .as_ref()
                    .ok_or_else(reject)?
                    .resident_role(graph, &self.plan, swap.layer, swap.gpu_cell, role)?
                    .ok_or_else(reject)?;
                let (host, ho, hb) = self
                    .host
                    .as_ref()
                    .ok_or_else(reject)?
                    .role(graph, &self.plan, swap.layer, swap.host_cell, role)?
                    .ok_or_else(reject)?;
                if gb != hb {
                    return Err(reject());
                }
                let mut resources = self
                    .client
                    .retained_resource_metadata(vec![gpu, host])
                    .map_err(crate::resident_util::fault)?;
                if resources.len() != 2 {
                    return Err(reject());
                }
                let h = resources.pop().ok_or_else(reject)?;
                let g = resources.pop().ok_or_else(reject)?;
                let range = |r: &cubecl::wgpu::WgpuResource,
                             offset: u64|
                 -> Result<cubecl::wgpu::WgpuResource> {
                    if r.offset != 0
                        || r.size != r.buffer.size()
                        || offset.checked_add(gb).is_none_or(|n| n > r.size)
                    {
                        return Err(reject());
                    }
                    Ok(cubecl::wgpu::WgpuResource {
                        buffer: r.buffer.clone(),
                        offset,
                        size: gb,
                    })
                };
                pairs.push(
                    crate::public_host_expert_mirror::strata_global::exchange::Pair {
                        gpu: range(g.resource(), go)?,
                        host: range(h.resource(), ho)?,
                    },
                );
            }
        }
        // From here until full private publication, unknown/failed completion
        // cannot resume inference. close can still attempt ordinary joined erase.
        self.mutation_pending = true;
        // SAFETY: immutable capabilities above authenticate all owning cells,
        // this exclusive source owner joined every prior CubeCL/BDA reader.
        unsafe {
            self.host
                .as_mut()
                .ok_or_else(reject)?
                .exchange(self.queue.as_ref().ok_or_else(reject)?, &pairs)?;
        }
        Ok(())
    }
    /// Join PRIVATE remap/usage writes before admitting another source reader.
    pub(crate) fn finish_publication(&mut self, commit: impl FnOnce() -> Result<()>) -> Result<()> {
        if self.closed || self.closing_failed {
            return Err(reject());
        }
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        commit()?;
        self.mutation_pending = false;
        Ok(())
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.closing_failed {
            return Err(reject());
        }
        self.closing_failed = true;
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        // New borrows are unavailable after tables/read aliases are released;
        // detach revokes lookups, joins all scheduled/native readers and destroys
        // registered buffers before granting the HOST owner zero/free authority.
        self.tables.clear();
        self.reads.clear();
        self.stages = None;
        self.stage_table = None;
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
            if let Some(stages) = self.stages.take() {
                std::mem::forget(stages);
            }
            if let Some(table) = self.stage_table.take() {
                std::mem::forget(table);
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
