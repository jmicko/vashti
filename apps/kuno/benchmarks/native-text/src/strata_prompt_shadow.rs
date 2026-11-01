//! Original authenticated model bytes for experts displaced by prompt scratch.
//! Persistent HOST pages use the same registry/queues as the normal complement;
//! current physical placement is refreshed only under joined source exclusion.
use super::*;
use crate::strata_global_plan::{EXPERTS, LAYERS};
use crate::strata_global_sources::Placement as CachePlacement;
pub(crate) static HOST_ZERO_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static HOST_FILL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) struct Shadow {
    pub layout: shadow_plan::Plan,
    pub first_page: usize,
}
pub(super) struct PageSpec {
    pub layer: usize,
    pub role: usize,
    pub piece: usize,
    pub shard: usize,
    pub logical: u64,
    pub backing_bound: u64,
}
impl Host {
    pub(super) fn allocate_original_page(
        &mut self,
        spec: PageSpec,
        charges: &mut HeapCharges,
        fill: impl FnOnce(&mut [u8]) -> KResult<()>,
    ) -> KResult<()> {
        self.allocate_page(spec, charges, false, fill)
    }
    /// As `allocate_original_page`, but `fill` promises to write every byte of
    /// `[..spec.logical]` (including its own padding); only the tail beyond the
    /// logical span is zeroed here.
    pub(super) fn allocate_original_page_filled(
        &mut self,
        spec: PageSpec,
        charges: &mut HeapCharges,
        fill: impl FnOnce(&mut [u8]) -> KResult<()>,
    ) -> KResult<()> {
        self.allocate_page(spec, charges, true, fill)
    }
    fn allocate_page(
        &mut self,
        spec: PageSpec,
        charges: &mut HeapCharges,
        fill_covers_logical: bool,
        fill: impl FnOnce(&mut [u8]) -> KResult<()>,
    ) -> KResult<()> {
        let device = self.device.clone();
        let budget = self.registry_budget.ok_or_else(fail)?;
        if self.closed || spec.shard >= SOURCE_SHA256.len() {
            return Err(fail());
        }
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.ok_or_else(fail)?;
        let raw = Arc::new(hal.raw_device().clone());
        let (queue_families, family_count) = hal.kuno_buffer_queue_families().ok_or_else(fail)?;
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(hal.raw_physical_device())
        };
        drop(hal);
        let descriptor = vk::BufferCreateInfo::default()
            .size(spec.logical)
            .usage(exchange::host_vk_usage())
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&queue_families[..family_count]);
        let buffer = unsafe { raw.create_buffer(&descriptor, None) }.map_err(|_| fail())?;
        let mut backing = Backing {
            device: device.clone(),
            raw: raw.clone(),
            raw_buffer: Some(buffer),
            memory: None,
            bytes: 0,
        };
        let requirements =
            Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) });
        check_requirements(spec.logical, spec.backing_bound, requirements)
            .map_err(Error::InvalidFormat)?;
        let placement = select_host_type(&properties, requirements.types, requirements.bytes)
            .map_err(Error::InvalidFormat)?;
        charges
            .admit_with_limit(
                placement,
                requirements.bytes,
                budget.backing_bytes,
                budget.buffers,
            )
            .map_err(Error::InvalidFormat)?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.bytes)
            .memory_type_index(placement.memory_type)
            .push_next(&mut dedicated)
            .push_next(&mut flags);
        backing.memory =
            Some(unsafe { raw.allocate_memory(&allocation, None) }.map_err(|_| fail())?);
        backing.bytes = requirements.bytes;
        unsafe { raw.bind_buffer_memory(buffer, backing.memory.unwrap(), 0) }
            .map_err(|_| fail())?;
        if Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) }) != requirements
        {
            return Err(fail());
        }
        let mut map = backing.map().map_err(Error::InvalidFormat)?;
        // Public model bytes only: a plain memset initializes padding and the
        // fresh driver allocation before fill. Volatile per-byte zeroize is for
        // erasing secrets and is several times slower on mapped host memory.
        let started = std::time::Instant::now();
        {
            let bytes = map.bytes_mut();
            let from = if fill_covers_logical {
                usize::try_from(spec.logical).map_err(|_| Error::ArithmeticOverflow)?.min(bytes.len())
            } else {
                0
            };
            let tail = &mut bytes[from..];
            unsafe { std::ptr::write_bytes(tail.as_mut_ptr(), 0, tail.len()) };
        }
        HOST_ZERO_NS.fetch_add(started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let started = std::time::Instant::now();
        fill(map.bytes_mut())?;
        HOST_FILL_NS.fetch_add(started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
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
            logical: spec.logical,
            phase: Phase::Sealed,
            binding: None,
            shard: spec.shard,
        };
        self.pages.push(GlobalPage {
            layer: spec.layer,
            role: spec.role,
            piece: spec.piece,
            owner,
            authority: None,
        });
        let page = self.pages.last_mut().unwrap();
        page.owner.backing.as_mut().unwrap().raw_buffer = None;
        page.owner.phase = Phase::Exposed;
        let imported = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                wgpu::hal::vulkan::Buffer::from_raw(buffer),
                &wgpu::BufferDescriptor {
                    label: Some("original-global-host-expert-piece"),
                    size: spec.logical,
                    usage: exchange::HOST_USAGE,
                    mapped_at_creation: false,
                },
            )
        };
        page.owner.buffer = Some(imported);
        page.owner.phase = Phase::Sealed;
        let resource = WgpuResource {
            buffer: page.owner.buffer.as_ref().unwrap().clone(),
            offset: 0,
            size: spec.logical,
        };
        // Retain exposed backing BEFORE any fallible registration.
        page.owner.phase = Phase::Exposed;
        let source = unsafe {
            PublicExternalHostSource::from_initialized_public_host(
                resource,
                spec.logical,
                requirements.bytes,
                placement.property_flags,
                placement.heap_flags,
                SOURCE_SHA256[spec.shard],
            )
        }
        .map_err(Error::InvalidFormat)?;
        let authority = self
            .client
            .register_public_external_host(source)
            .map_err(|_| fail())?;
        page.owner.binding = Some(authority.handle().binding());
        page.authority = Some(authority);
        page.owner.phase = Phase::Registered;
        self.logical = self
            .logical
            .checked_add(spec.logical)
            .ok_or(Error::ArithmeticOverflow)?;
        self.physical = self
            .physical
            .checked_add(requirements.bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(())
    }
}

fn role_name(layer: usize, role: usize) -> String {
    format!(
        "blk.{layer}.{}",
        [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight"
        ][role]
    )
}
fn fill_shadow_page(
    graph: BoundGraph<'_, '_>,
    global: &GlobalPlan,
    placement: &CachePlacement,
    layout: &shadow_plan::Plan,
    page: usize,
    mapped: &mut [u8],
    window: &mut [u8],
) -> KResult<()> {
    let p = layout.pages().get(page).ok_or_else(fail)?;
    if mapped.len() < p.logical as usize {
        return Err(fail());
    }
    // Padding and the complete driver allocation are initialized on every use.
    mapped.zeroize();
    for cell in layout.cells() {
        let layer = cell.pair.layer;
        let logical = placement.original_at(layer, cell.pair.expert)?;
        if logical >= EXPERTS {
            return Err(fail());
        }
        for role in 0..3 {
            let (source_page, offset, bytes) = cell.sources[role];
            if source_page != page {
                continue;
            }
            if bytes != global.layers[layer].role_expert_bytes[role] {
                return Err(fail());
            }
            let tensor = graph.tensor(&role_name(layer, role))?;
            if tensor.info().shard_index() != p.shard {
                return Err(fail());
            }
            let target = window
                .get_mut(..usize::try_from(bytes).map_err(|_| Error::ArithmeticOverflow)?)
                .ok_or(Error::InvalidBufferLength)?;
            tensor.read_range(
                (logical as u64)
                    .checked_mul(bytes)
                    .ok_or(Error::ArithmeticOverflow)?,
                target,
            )?;
            copy_range(mapped, p.logical as usize, offset, target).map_err(Error::InvalidFormat)?;
            target.zeroize();
        }
    }
    Ok(())
}
impl Host {
    /// Exact backing/metadata quote before allocating any original-model copy.
    pub(crate) fn loan_shadow_layout(
        graph: BoundGraph<'_, '_>,
        global: &GlobalPlan,
        indices: [usize; 3],
    ) -> KResult<shadow_plan::Plan> {
        global.check_binding(graph)?;
        let mut shards = [[0usize; 3]; LAYERS];
        for (layer, roles) in shards.iter_mut().enumerate() {
            for (role, shard) in roles.iter_mut().enumerate() {
                *shard = graph.tensor(&role_name(layer, role))?.info().shard_index();
            }
        }
        shadow_plan::Plan::new(global, indices, shards)
    }
    /// Initialize persistent original-weight backing, with no read from lent
    /// GPU scratch. The separately quoted pages are retained through final close.
    /// # Safety
    /// Actual model hashes are authenticated; caller holds `reserved` RAM and
    /// excludes every model/CPU/GPU consumer on this protected owning stream.
    pub(crate) unsafe fn create_loan_shadow(
        &mut self,
        graph: BoundGraph<'_, '_>,
        global: &GlobalPlan,
        layout: shadow_plan::Plan,
        placement: &CachePlacement,
        reserved: u64,
    ) -> KResult<()> {
        global.check_binding(graph)?;
        layout.check(global)?;
        if self.shadow.is_some() || self.pages.iter().any(|p| p.layer == LAYERS) {
            return Err(fail());
        }
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        join(&self.device).map_err(Error::InvalidFormat)?;
        unsafe {
            self.reserve_loan_shadow(global, &layout, reserved)?;
        }
        let mut charges = HeapCharges::default();
        let budget = self.registry_budget.ok_or_else(fail)?;
        for p in &self.pages {
            charges
                .admit_with_limit(
                    p.owner.placement,
                    p.owner.backing.as_ref().ok_or_else(fail)?.bytes,
                    budget.backing_bytes,
                    budget.buffers,
                )
                .map_err(Error::InvalidFormat)?;
        }
        self.pages
            .try_reserve_exact(layout.pages().len())
            .map_err(|_| Error::LimitExceeded("shadow owner metadata"))?;
        let first_page = self.pages.len();
        let mut window = Zeroizing::new(vec![0; SOURCE_WINDOW as usize]);
        for (i, p) in layout.pages().iter().enumerate() {
            self.allocate_original_page(
                PageSpec {
                    layer: LAYERS,
                    role: 0,
                    piece: i,
                    shard: p.shard,
                    logical: p.logical,
                    backing_bound: p.backing_bound,
                },
                &mut charges,
                |mapped| {
                    fill_shadow_page(graph, global, placement, &layout, i, mapped, &mut window)
                },
            )?;
        }
        graph.verify_unchanged()?;
        self.shadow = Some(Shadow { layout, first_page });
        Ok(())
    }
    /// Refresh the same registered pages after an adaptive placement change.
    /// This never consumes new lifetime registry slots or allocates a model copy.
    /// # Safety
    /// All native recorded/scheduled consumers and CPU maps are joined/excluded;
    /// caller is the sole protected source owner, and holds the original grant.
    pub(crate) unsafe fn refresh_loan_shadow(
        &mut self,
        graph: BoundGraph<'_, '_>,
        global: &GlobalPlan,
        placement: &CachePlacement,
    ) -> KResult<()> {
        global.check_binding(graph)?;
        if self.closed || self.cpu_borrow.load(std::sync::atomic::Ordering::Acquire) {
            return Err(fail());
        }
        let shadow = self.shadow.as_ref().ok_or_else(fail)?;
        shadow.layout.check(global)?;
        cubecl_common::future::block_on(self.client.sync()).map_err(crate::resident_util::fault)?;
        join(&self.device).map_err(Error::InvalidFormat)?;
        let bindings = self.pages[shadow.first_page..]
            .iter()
            .map(|p| {
                p.authority
                    .as_ref()
                    .map(|a| a.handle().binding())
                    .ok_or_else(fail)
            })
            .collect::<KResult<Vec<_>>>()?;
        // Explicit registry revocation precedes raw mapping. A failed refresh
        // drops only the token; lookups stay revoked until process recovery.
        let token =
            unsafe { self.client.begin_public_external_rebase(bindings) }.map_err(|_| fail())?;
        let mut window = Zeroizing::new(vec![0; SOURCE_WINDOW as usize]);
        for (i, spec) in shadow.layout.pages().iter().enumerate() {
            let page = self.pages.get(shadow.first_page + i).ok_or_else(fail)?;
            if page.layer != LAYERS
                || page.piece != i
                || page.owner.shard != spec.shard
                || page.owner.logical != spec.logical
                || page.owner.phase != Phase::Registered
                || page.authority.is_none()
            {
                return Err(fail());
            }
            let mut map = page
                .owner
                .backing
                .as_ref()
                .ok_or_else(fail)?
                .map()
                .map_err(Error::InvalidFormat)?;
            fill_shadow_page(
                graph,
                global,
                placement,
                &shadow.layout,
                i,
                map.bytes_mut(),
                &mut window,
            )?;
            map.cache(page.owner.placement.coherent(), true)
                .map_err(Error::InvalidFormat)?;
        }
        graph.verify_unchanged()?;
        // Every original-model byte/padding was initialized and flushed; raw
        // mappings are gone, and no inference/private data entered these pages.
        unsafe { self.client.finish_public_external_rebase(token) }.map_err(|_| fail())
    }
    pub(crate) fn loan_shadow_indices(&self) -> Option<[usize; 3]> {
        self.shadow.as_ref().map(|s| s.layout.indices())
    }
    pub(crate) fn loan_shadow_role(
        &self,
        graph: BoundGraph<'_, '_>,
        global: &GlobalPlan,
        layer: usize,
        cell: usize,
        role: usize,
    ) -> KResult<Option<(cubecl_runtime::server::Handle, u64, u64)>> {
        global.check_binding(graph)?;
        if self.closed || self.instance != global.instance() {
            return Err(fail());
        }
        let shadow = self.shadow.as_ref().ok_or_else(fail)?;
        shadow.layout.check(global)?;
        let Some((i, offset, bytes)) = shadow.layout.role(layer, cell, role)? else {
            return Ok(None);
        };
        let p = self.pages.get(shadow.first_page + i).ok_or_else(fail)?;
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
}
