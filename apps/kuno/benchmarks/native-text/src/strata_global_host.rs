//! Variable global HOST complement, using the original nonlocal Vulkan heap and
//! sole CubeCL immutable registry. No fixed 432-page or uniform-bank assumption.
#[path = "strata_global_exchange.rs"]
pub(crate) mod exchange;
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
const GLOBAL_METADATA_CAP: u64 =
    32 * 1024 * 1024 + crate::strata_global_sources::PLACEMENT_HOST_BYTES;
const SETUP_WINDOWS: u64 = 2 * crate::strata_global_plan::PUBLIC_UPLOAD_WINDOW + SOURCE_WINDOW;
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
pub(crate) struct Host {
    client: ComputeClient<WgpuRuntime>,
    device: wgpu::Device,
    instance: u64,
    pages: Vec<GlobalPage>,
    closed: bool,
    logical: u64,
    physical: u64,
    exchange_runtime: Option<exchange::Runtime>,
    cpu_borrow: std::sync::atomic::AtomicBool,
}
impl Host {
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
        let physical = plan
            .layers
            .iter()
            .flat_map(|l| l.roles.iter().flatten())
            .filter(|p| p.tier == Tier::Host)
            .try_fold(0u64, |n, p| {
                n.checked_add(envelope(p.logical_bytes)?)
                    .ok_or(Error::ArithmeticOverflow)
            })?;
        physical
            .checked_add(METADATA_BYTES as u64)
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
        let count = plan
            .layers
            .iter()
            .flat_map(|l| l.roles.iter().flatten())
            .filter(|p| p.tier == Tier::Host)
            .count();
        if count > cubecl_runtime::public_external::MAX_BUFFERS {
            return Err(fail());
        }
        let mut host = Self {
            client: client.clone(),
            device: device.clone(),
            instance: plan.instance(),
            pages: Vec::with_capacity(count),
            closed: false,
            logical: 0,
            physical: 0,
            exchange_runtime: None,
            cpu_borrow: std::sync::atomic::AtomicBool::new(false),
        };
        if count == 0 {
            return Ok(host);
        }
        let pool_bytes = reserved
            .checked_sub(METADATA_BYTES as u64 + GLOBAL_METADATA_CAP + SETUP_WINDOWS)
            .ok_or(Error::ArithmeticOverflow)?;
        client
            .configure_public_external_host(PublicExternalBudget {
                backing_bytes: pool_bytes,
                buffers: count,
                metadata_bytes: METADATA_BYTES,
            })
            .map_err(|_| fail())?;
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
        let raw = Arc::new(hal.raw_device().clone());
        let queue_families = hal.kuno_buffer_queue_families().ok_or_else(fail)?;
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(hal.raw_physical_device())
        };
        drop(hal);
        let mut charges = HeapCharges::default();
        let mut window = Zeroizing::new(Vec::new());
        window
            .try_reserve_exact(SOURCE_WINDOW as usize)
            .map_err(|_| Error::LimitExceeded("global HOST source window"))?;
        window.resize(SOURCE_WINDOW as usize, 0);
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
                    let envelope = envelope(p.logical_bytes)?;
                    let descriptor = vk::BufferCreateInfo::default()
                        .size(p.logical_bytes)
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | vk::BufferUsageFlags::TRANSFER_SRC
                                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        )
                        .sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(&queue_families);
                    let buffer =
                        unsafe { raw.create_buffer(&descriptor, None) }.map_err(|_| fail())?;
                    let mut backing = Backing {
                        device: device.clone(),
                        raw: raw.clone(),
                        raw_buffer: Some(buffer),
                        memory: None,
                        bytes: 0,
                    };
                    let requirements =
                        Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) });
                    check_requirements(p.logical_bytes, envelope, requirements)
                        .map_err(Error::InvalidFormat)?;
                    let placement =
                        select_host_type(&properties, requirements.types, requirements.bytes)
                            .map_err(Error::InvalidFormat)?;
                    charges
                        .admit_with_limit(placement, requirements.bytes, pool_bytes, count)
                        .map_err(Error::InvalidFormat)?;
                    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
                    let mut flags = vk::MemoryAllocateFlagsInfo::default()
                        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                    let allocation = vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.bytes)
                        .memory_type_index(placement.memory_type)
                        .push_next(&mut dedicated)
                        .push_next(&mut flags);
                    backing.memory = Some(
                        unsafe { raw.allocate_memory(&allocation, None) }.map_err(|_| fail())?,
                    );
                    backing.bytes = requirements.bytes;
                    unsafe { raw.bind_buffer_memory(buffer, backing.memory.unwrap(), 0) }
                        .map_err(|_| fail())?;
                    if Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) })
                        != requirements
                    {
                        return Err(fail());
                    }
                    let mut map = backing.map().map_err(Error::InvalidFormat)?;
                    map.bytes_mut().zeroize();
                    for ordinal in 0..p.count {
                        let expert = l.expert_at(p.first + ordinal)?;
                        let bytes = usize::try_from(p.expert_bytes)
                            .map_err(|_| Error::ArithmeticOverflow)?;
                        let target = window.get_mut(..bytes).ok_or(Error::InvalidBufferLength)?;
                        tensor.read_range(
                            (expert as u64)
                                .checked_mul(p.expert_bytes)
                                .ok_or(Error::ArithmeticOverflow)?,
                            target,
                        )?;
                        copy_range(
                            map.bytes_mut(),
                            p.logical_bytes as usize,
                            ordinal as u64 * p.expert_bytes,
                            target,
                        )
                        .map_err(Error::InvalidFormat)?;
                        target.zeroize();
                    }
                    map.cache(placement.coherent(), true)
                        .map_err(Error::InvalidFormat)?;
                    drop(map);
                    // Own the backing before HAL import transfers raw-buffer
                    // destruction authority. An import unwind has unknown exposure
                    // and therefore retains the memory instead of freeing it.
                    let owner = PageOwner {
                        backing: Some(backing),
                        buffer: None,
                        placement,
                        allocation_alignment: requirements.alignment,
                        logical: p.logical_bytes,
                        phase: Phase::Sealed,
                        binding: None,
                        shard,
                    };
                    host.pages.push(GlobalPage {
                        layer,
                        role,
                        piece,
                        owner,
                        authority: None,
                    });
                    let page = host.pages.last_mut().unwrap();
                    page.owner.backing.as_mut().unwrap().raw_buffer = None;
                    page.owner.phase = Phase::Exposed;
                    let imported = unsafe {
                        device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                            wgpu::hal::vulkan::Buffer::from_raw(buffer),
                            &wgpu::BufferDescriptor {
                                label: Some("original-global-host-expert-piece"),
                                size: p.logical_bytes,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            },
                        )
                    };
                    page.owner.buffer = Some(imported);
                    page.owner.phase = Phase::Sealed;
                    let resource = WgpuResource {
                        buffer: page.owner.buffer.as_ref().unwrap().clone(),
                        offset: 0,
                        size: p.logical_bytes,
                    };
                    // Retain exposed backing BEFORE any fallible registration.
                    page.owner.phase = Phase::Exposed;
                    let source = unsafe {
                        PublicExternalHostSource::from_initialized_public_host(
                            resource,
                            p.logical_bytes,
                            requirements.bytes,
                            placement.property_flags,
                            placement.heap_flags,
                            SOURCE_SHA256[shard],
                        )
                    }
                    .map_err(Error::InvalidFormat)?;
                    let authority = client
                        .register_public_external_host(source)
                        .map_err(|_| fail())?;
                    page.owner.binding = Some(authority.handle().binding());
                    page.authority = Some(authority);
                    page.owner.phase = Phase::Registered;
                    host.logical = host
                        .logical
                        .checked_add(p.logical_bytes)
                        .ok_or(Error::ArithmeticOverflow)?;
                    host.physical = host
                        .physical
                        .checked_add(requirements.bytes)
                        .ok_or(Error::ArithmeticOverflow)?;
                }
            }
        }
        if host.pages.len() != count
            || host.logical != plan.host_logical_bytes
            || host.physical > pool_bytes
        {
            return Err(fail());
        }
        graph.verify_unchanged()?;
        Ok(host)
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
        if self.closed || self.instance != plan.instance() {
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
