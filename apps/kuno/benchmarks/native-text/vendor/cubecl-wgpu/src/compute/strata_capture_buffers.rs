//! Capture-local exact buffer identity lookup, protected by one real Wgpu
//! destruction lock per phase. Never persisted in a recording or across calls.
use super::{WgpuResource, fault};
use ash::vk;
use cubecl_core::server::ServerError;

pub(super) struct Keys {
    buffers: Vec<wgpu::Buffer>,
    bytes: usize,
    guard_budget: usize,
}
impl Keys {
    pub fn new(resources: &[WgpuResource], budget: usize) -> Result<Self, ServerError> {
        if resources.is_empty() || resources.len() > 65536 {
            return Err(fault("native capture root capacity"));
        }
        if !resources.windows(2).all(|w| w[0].buffer < w[1].buffer) {
            return Err(fault("native capture original root identity order"));
        }
        // Includes core's Arc + HAL pointer vectors, temporary real BufferIds,
        // all headers/rank data and alignment. Rechecked by both actual APIs.
        let guard_budget = resources
            .len()
            .checked_mul(32)
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| fault("native capture guard metadata overflow"))?;
        let minimum = resources
            .len()
            .checked_mul(size_of::<wgpu::Buffer>())
            .and_then(|n| n.checked_add(guard_budget))
            .ok_or_else(|| fault("native capture root metadata overflow"))?;
        if minimum > budget {
            return Err(fault("native capture root metadata capacity"));
        }
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(resources.len())
            .map_err(|_| fault("native capture root allocation"))?;
        let bytes = buffers
            .capacity()
            .checked_mul(size_of::<wgpu::Buffer>())
            .and_then(|n| n.checked_add(guard_budget))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or_else(|| fault("native capture root metadata overflow"))?;
        if bytes > budget {
            return Err(fault("native capture actual root metadata capacity"));
        }
        buffers.extend(resources.iter().map(|r| r.buffer.clone()));
        Ok(Self {
            buffers,
            bytes,
            guard_budget,
        })
    }
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    pub fn acquire(&self) -> Result<Guarded<'_>, ServerError> {
        // SAFETY: All actual original owners and views remain pinned. The
        // returned guard prevents destruction until released. No nested Wgpu
        // lookup/poll/drop/destruction is allowed in the guarded phase.
        let batch = unsafe {
            wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(
                &self.buffers,
                self.guard_budget,
            )
        }
        .ok_or_else(|| fault("native capture Vulkan destruction guard"))?;
        if batch.metadata_bytes().is_none_or(|n| n > self.guard_budget) {
            return Err(fault("native capture actual guard metadata capacity"));
        }
        Ok(Guarded { keys: self, batch })
    }
}
pub(super) struct Guarded<'a> {
    keys: &'a Keys,
    batch: wgpu::wgc::KunoBufferHalBatch<wgpu::hal::api::Vulkan>,
}
impl Guarded<'_> {
    pub fn get(&self, resource: &WgpuResource) -> Result<vk::Buffer, ServerError> {
        let index = self
            .keys
            .buffers
            .binary_search(&resource.buffer)
            .map_err(|_| fault("native capture foreign root identity"))?;
        let raw = self
            .batch
            .get(index)
            .ok_or_else(|| fault("native capture missing actual guarded buffer"))?;
        // SAFETY: This exact original raw object remains guarded for all Vulkan
        // planning/encoding uses, not just the duration of this method.
        Ok(unsafe { raw.raw_buffer_handle() })
    }
}
