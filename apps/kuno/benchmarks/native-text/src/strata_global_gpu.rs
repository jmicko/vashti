//! Owning native-global PUBLIC expert cache. Original GGUF bytes, exact persistent
//! GPU roots and one bounded startup window; no decoded model or request data.
//! Admission/precharge belongs to GpuExecutor; consumers must retain this owner.
use crate::strata_global_plan::{Plan, Source};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{Error, Result, model::BoundGraph};
use std::sync::atomic::AtomicBool;
fn reject() -> Error {
    Error::InvalidFormat("Strata owning global GPU cache")
}
pub(crate) struct Cache {
    client: ComputeClient<WgpuRuntime>,
    roots: Vec<Handle>,
    bytes: u64,
    plan_instance: u64,
}
impl Cache {
    /// Caller has charged every exact root under exclusive persistent admission.
    pub(crate) fn load(
        client: &ComputeClient<WgpuRuntime>,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
    ) -> Result<Self> {
        if !crate::strata_intel_core::enabled() {
            return Err(reject());
        }
        plan.check_binding(graph)?;
        let mut roots = Vec::with_capacity(plan.chunks.len());
        let mut bytes = 0u64;
        for (index, chunk) in plan.chunks.iter().enumerate() {
            if chunk.physical_bytes > client.properties().memory.max_page_size {
                return Err(Error::LimitExceeded(
                    "global cache root exceeds backend binding",
                ));
            }
            let size =
                usize::try_from(chunk.physical_bytes).map_err(|_| Error::ArithmeticOverflow)?;
            // Allocate the true persistent GPU root without a same-sized host
            // Vec. Every byte, including slot/root padding, is initialized by
            // bounded original-model windows before any table/consumer exists.
            let handle = client.empty(size);
            if handle.offset_start.unwrap_or(0) != 0
                || handle.offset_end.unwrap_or(0) != 0
                || handle.size() != chunk.physical_bytes
                || handle.size_in_used() != chunk.physical_bytes
            {
                return Err(reject());
            }
            roots.push(handle.clone());
            let mut first = 0u64;
            while first < chunk.physical_bytes {
                let count = (chunk.physical_bytes - first)
                    .min(crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW)
                    as usize;
                let mut window = Vec::new();
                window
                    .try_reserve_exact(count)
                    .map_err(|_| Error::LimitExceeded("global cache public load window"))?;
                if window.capacity() != count {
                    return Err(reject());
                }
                window.resize(count, 0);
                plan.load_chunk_range(graph, index, first, &mut window)?;
                let view = handle
                    .clone()
                    .offset_start(first)
                    .offset_end(chunk.physical_bytes - first - count as u64);
                client
                    .write_public_owned_batch(vec![(view, window)])
                    .map_err(crate::resident_util::fault)?;
                // Moved source/upload resources drain before the next window;
                // increasing GPU contiguity never enlarges the host envelope.
                cubecl_common::future::block_on(client.sync())
                    .map_err(crate::resident_util::fault)?;
                first += count as u64;
            }
            bytes = bytes
                .checked_add(chunk.physical_bytes)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        if bytes != plan.resident_piece_bytes {
            return Err(reject());
        }
        let resources = client
            .retained_resource_metadata(roots.clone())
            .map_err(|_| reject())?;
        if resources.len() != roots.len() {
            return Err(reject());
        }
        for (index, resource) in resources.iter().enumerate() {
            let r = resource.resource();
            if r.offset != 0
                || r.size != plan.chunks[index].physical_bytes
                || r.buffer.size() != r.size
                || resources[..index]
                    .iter()
                    .any(|other| other.resource().buffer == r.buffer)
            {
                return Err(reject());
            }
        }
        graph.verify_unchanged()?;
        Ok(Self {
            client: client.clone(),
            roots,
            bytes,
            plan_instance: plan.instance(),
        })
    }
    /// Exclusive source transaction only: private exposure has already been
    /// completely erased/inspected. Restore CURRENT logical placement, not the
    /// initial ranked profile. No allocation or private public-upload shortcut.
    pub(crate) fn restore_chunks(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
        indices: [usize; 3],
        placement: &crate::strata_global_sources::Placement,
    ) -> Result<()> {
        plan.check_binding(graph)?;
        if self.plan_instance != plan.instance() {
            return Err(reject());
        }
        for (j, &index) in indices.iter().enumerate() {
            if indices[..j].contains(&index) {
                return Err(reject());
            }
            let root = self.roots.get(index).ok_or_else(reject)?;
            let chunk = plan.chunks.get(index).ok_or_else(reject)?;
            if root.size_in_used() != chunk.physical_bytes
                || root.size() != chunk.physical_bytes
                || root.offset_start.unwrap_or(0) != 0
                || root.offset_end.unwrap_or(0) != 0
            {
                return Err(reject());
            }
            let mut first = 0u64;
            while first < chunk.physical_bytes {
                let n = (chunk.physical_bytes - first)
                    .min(crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW)
                    as usize;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(n)
                    .map_err(|_| Error::LimitExceeded("cache restore public window"))?;
                if bytes.capacity() != n {
                    return Err(reject());
                }
                bytes.resize(n, 0);
                plan.load_chunk_placed(graph, index, first, &mut bytes, placement)?;
                let view = root
                    .clone()
                    .offset_start(first)
                    .offset_end(chunk.physical_bytes - first - n as u64);
                self.client
                    .write_public_owned_batch(vec![(view, bytes)])
                    .map_err(crate::resident_util::fault)?;
                cubecl_common::future::block_on(self.client.sync())
                    .map_err(crate::resident_util::fault)?;
                first += n as u64;
            }
        }
        graph.verify_unchanged()?;
        Ok(())
    }
    pub(crate) fn requests(plan: &Plan) -> Vec<u64> {
        plan.chunks.iter().map(|c| c.physical_bytes).collect()
    }
    pub(crate) fn roots(&self) -> &[Handle] {
        &self.roots
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Actual model-weight capability, bounded to the original expert role.
    /// No invented source root or residency credit for HOST complements.
    pub(crate) fn resident_role(
        &self,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
        layer: usize,
        expert: usize,
        role: usize,
    ) -> Result<Option<(Handle, u64, u64)>> {
        plan.check_binding(graph)?;
        if self.plan_instance != plan.instance() {
            return Err(reject());
        }
        let Source::Resident {
            chunk,
            offset,
            bytes,
        } = plan.source(layer, expert, role)?
        else {
            return Ok(None);
        };
        let root = self.roots.get(chunk).ok_or_else(reject)?;
        if offset
            .checked_add(bytes)
            .is_none_or(|n| n > plan.chunks[chunk].logical_bytes)
            || root.size_in_used() != plan.chunks[chunk].physical_bytes
        {
            return Err(reject());
        }
        Ok(Some((root.clone(), offset, bytes)))
    }
    pub(crate) fn persistent_load(
        client: &ComputeClient<WgpuRuntime>,
        graph: BoundGraph<'_, '_>,
        plan: &Plan,
    ) -> Result<Self> {
        use cubecl_runtime::{
            config::{CubeClRuntimeConfig, RuntimeConfig, memory::PersistentMemory},
            memory_management::MemoryAllocationMode,
        };
        if !matches!(
            CubeClRuntimeConfig::get().memory.persistent_memory,
            PersistentMemory::Enabled
        ) {
            return Err(reject());
        }
        let alignment = client.properties().memory.alignment;
        if alignment == 0
            || !alignment.is_power_of_two()
            || Self::requests(plan)
                .iter()
                .any(|b| !b.is_multiple_of(alignment))
        {
            return Err(reject());
        }
        let retired = AtomicBool::new(false);
        crate::expert_banks::exact_persistent::persistent_scope(
            |mode: MemoryAllocationMode| {
                // SAFETY: owning executor holds exclusive stream admission and the
                // complete persistent precharge while this scope is active.
                unsafe {
                    client.allocation_mode(mode);
                }
                Ok(())
            },
            &retired,
            || Self::load(client, graph, plan),
        )
    }
}
impl Drop for Cache {
    fn drop(&mut self) {
        // Joined model/request completion is required before parent close. The
        // runtime also retains submitted buffers until their completion fence.
        let _ = self.client.flush();
    }
}
