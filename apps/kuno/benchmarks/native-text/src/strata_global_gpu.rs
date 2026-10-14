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
            let mut window = Vec::new();
            window
                .try_reserve_exact(size)
                .map_err(|_| Error::LimitExceeded("global cache public load window"))?;
            window.resize(size, 0);
            plan.load_chunk(graph, index, &mut window)?;
            let handle = client.create_public_owned(window);
            if handle.offset_start.unwrap_or(0) != 0
                || handle.offset_end.unwrap_or(0) != 0
                || handle.size() != chunk.physical_bytes
                || handle.size_in_used() != chunk.physical_bytes
            {
                return Err(reject());
            }
            roots.push(handle);
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
    pub(crate) fn requests(plan: &Plan) -> Vec<u64> {
        plan.chunks.iter().map(|c| c.physical_bytes).collect()
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
