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
    reads: Vec<Vec<Handle>>,
    closed: bool,
    closing_failed: bool,
}
impl Sources {
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
            reads: Vec::with_capacity(LAYERS),
            closed: false,
            closing_failed: false,
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
    pub(crate) fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Retains the actual immutable roots in native recording/fence tracking.
    /// The only descriptor added is the fixed per-expert address table.
    pub(crate) fn bind(&self, layer: usize, args: KernelArguments) -> Result<KernelArguments> {
        if self.closed || self.closing_failed {
            return Err(reject());
        }
        let table = self.tables.get(layer).ok_or_else(reject)?;
        let reads = self.reads.get(layer).ok_or_else(reject)?;
        Ok(args
            .with_buffer(table.clone().binding())
            .with_read_dependencies(reads.iter().cloned().map(Handle::binding).collect()))
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
            if let Some(gpu) = self.gpu.take() {
                std::mem::forget(gpu);
            }
            if let Some(host) = self.host.take() {
                std::mem::forget(host);
            }
        }
    }
}
