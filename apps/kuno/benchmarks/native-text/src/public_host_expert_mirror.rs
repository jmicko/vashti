//! Original PUBLIC expert complement in explicitly nonlocal Vulkan host memory.
//!
//! Construction materializes only a previously validated, completed frozen
//! Plan. It never chooses routes, loads GPU banks, registers CubeCL resources,
//! evaluates a model, or changes an ordinary/private allocator. Full original
//! artifact hashes and the banks' loaded-byte/same-device proofs are caller
//! prerequisites. File descriptor/stat guards are consistency, not authenticity.
//!
//! Every HAL import owns its VkBuffer but has NO managed memory allocation;
//! separate guards own VkDeviceMemory and keep the actual wgpu Device alive.
//! The entire physical envelope, combined typed metadata and one wiping source
//! window are precharged before any native allocation. Public backing is never
//! credited to GPU MAIN or private I/O. Allocation requirements/heap placement
//! remain runtime checks; this source alone establishes no physical acceptance.
//!
//! Exposed pages must each return a non-forgeable, exact CubeCL detach handoff
//! before CPU zero/free. Failed detach or join grants no release authority.
//! Drop leaks exposed backing/device rather than free potentially live memory.
//! No private erasure, pinning, whole-model performance or production claim.

#[path = "strata_global_host.rs"]
#[allow(dead_code)] // Connected global source construction; routing follows.
pub(crate) mod strata_global;
#[path = "strata_gpu_exchange.rs"]
pub(crate) mod strata_gpu_exchange;

use crate::{
    expert_banks::ExpertBanks,
    public_host_expert_plan::{self as plan, Plan, PlanIdentity, Role},
};
use ash::vk;
use cubecl::wgpu::WgpuResource;
use cubecl_runtime::{
    public_external::{PublicExternalDetachReceipt, is_public_external_binding},
    server::Binding,
};
use kuno_text::{
    GgufFile,
    model::{BoundGraph, GraphBindingId},
};
use std::{mem::size_of, sync::Arc, time::Duration};
use zeroize::{Zeroize, Zeroizing};

type Result<T> = std::result::Result<T, &'static str>;
pub(crate) const PAGES: usize = plan::LAYERS * 3 * 3;
const HOST_CAP: u64 = 64 * 1024 * 1024 * 1024;
const ROLES: [Role; 3] = [Role::Gate, Role::Up, Role::Down];
const SOURCE_SHA256: [[u8; 32]; 2] = [
    [
        0x4c, 0x1e, 0xb2, 0xce, 0xb4, 0x91, 0x5e, 0x11, 0x92, 0xf4, 0xf3, 0x86, 0x02, 0x18, 0x97,
        0xbd, 0xe5, 0x6a, 0x97, 0xf4, 0x0a, 0x0b, 0xb7, 0x8b, 0xb8, 0x64, 0x65, 0xe0, 0xf7, 0xd2,
        0xac, 0xa3,
    ],
    [
        0x31, 0x6b, 0x46, 0xf3, 0xa2, 0xdb, 0xd6, 0x8c, 0x90, 0x0f, 0x43, 0x13, 0x6a, 0xb9, 0x44,
        0x9f, 0x9d, 0xcc, 0x37, 0x25, 0xdf, 0xd8, 0xc7, 0x94, 0x84, 0x7c, 0x20, 0x4b, 0xc1, 0x61,
        0xe1, 0x13,
    ],
];

fn rejected() -> &'static str {
    "public host expert mirror rejected"
}
fn index(layer: usize, role: Role, page: usize) -> Result<usize> {
    let role = match role {
        Role::Gate => 0,
        Role::Up => 1,
        Role::Down => 2,
    };
    if layer >= plan::LAYERS || page >= 3 {
        return Err(rejected());
    }
    Ok(layer * 9 + role * 3 + page)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Placement {
    pub memory_type: u32,
    pub heap: u32,
    pub property_flags: u32,
    pub heap_flags: u32,
    pub heap_bytes: u64,
}
impl Placement {
    fn coherent(self) -> bool {
        self.property_flags & vk::MemoryPropertyFlags::HOST_COHERENT.as_raw() != 0
    }
}
fn select_host_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    allowed: u32,
    bytes: u64,
) -> Result<Placement> {
    if bytes == 0
        || bytes > plan::MAX_BINDING_BYTES
        || properties.memory_type_count > 32
        || properties.memory_heap_count > 16
    {
        return Err(rejected());
    }
    let mut best: Option<(u32, Placement)> = None;
    for i in 0..properties.memory_type_count {
        if allowed & (1 << i) == 0 {
            continue;
        }
        let memory = properties.memory_types[i as usize];
        if memory.heap_index >= properties.memory_heap_count {
            return Err(rejected());
        }
        let heap = properties.memory_heaps[memory.heap_index as usize];
        let flags = memory.property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            || flags.intersects(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::LAZILY_ALLOCATED
                    | vk::MemoryPropertyFlags::PROTECTED,
            )
            || heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            || bytes > heap.size
        {
            continue;
        }
        let score = u32::from(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)) * 2
            + u32::from(flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT));
        let placement = Placement {
            memory_type: i,
            heap: memory.heap_index,
            property_flags: flags.as_raw(),
            heap_flags: heap.flags.as_raw(),
            heap_bytes: heap.size,
        };
        if best.is_none_or(|(prior, _)| score > prior) {
            best = Some((score, placement));
        }
    }
    best.map(|(_, p)| p)
        .ok_or("public host expert nonlocal heap unavailable")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Requirements {
    bytes: u64,
    alignment: u64,
    types: u32,
}
impl From<vk::MemoryRequirements> for Requirements {
    fn from(r: vk::MemoryRequirements) -> Self {
        Self {
            bytes: r.size,
            alignment: r.alignment,
            types: r.memory_type_bits,
        }
    }
}
fn check_requirements(logical: u64, envelope: u64, r: Requirements) -> Result<()> {
    if !(32..=plan::MAX_BINDING_BYTES).contains(&logical)
        || !logical.is_multiple_of(4)
        || envelope < logical
        || envelope > plan::MAX_BINDING_BYTES
        || r.bytes < logical
        || r.bytes > envelope
        || r.types == 0
        || r.alignment == 0
        || !r.alignment.is_power_of_two()
        || !r.bytes.is_multiple_of(r.alignment)
    {
        return Err(rejected());
    }
    usize::try_from(r.bytes).map_err(|_| rejected())?;
    Ok(())
}
fn metadata_bytes(mapping: u64, generations: u64, pages: usize, proofs: usize) -> Result<u64> {
    let pages = pages
        .checked_mul(size_of::<PageOwner>())
        .ok_or(rejected())?;
    let proofs = proofs
        .checked_mul(size_of::<PublicExternalDetachReceipt<WgpuResource>>())
        .ok_or(rejected())?;
    // Each page can issue exactly one retained coordinate/resource witness.
    let witnesses = PAGES
        .checked_mul(size_of::<PageResource>())
        .ok_or(rejected())?;
    mapping
        .checked_add(generations)
        .and_then(|n| n.checked_add(pages as u64))
        .and_then(|n| n.checked_add(proofs as u64))
        .and_then(|n| n.checked_add(witnesses as u64))
        .and_then(|n| n.checked_add(size_of::<Mirror<'_, '_, '_>>() as u64))
        // One scoped layer may map the SAME registered allocations readonly.
        // Fixed headers only; no copied source payload or additional RAM page.
        .and_then(|n| {
            n.checked_add((size_of::<ReadOnlyLayer<'_>>() + size_of::<ReadOnlyMapped<'_>>()) as u64)
        })
        .and_then(|n| n.checked_add((size_of::<ash::Device>() + 2 * size_of::<usize>()) as u64))
        // Source fill/registration/close temporarily retain one full descriptor,
        // backing, receipt and a complete validation bitmap, all public metadata.
        .and_then(|n| n.checked_add((size_of::<PageOwner>() * 2 + PAGES) as u64))
        .filter(|&n| n <= plan::METADATA_HOST_CAP)
        .ok_or(rejected())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ledger {
    pub reserved_host_bytes: u64,
    pub physical_envelope_bytes: u64,
    pub live_backing_bytes: u64,
    pub total_allocated_bytes: u64,
    pub total_freed_bytes: u64,
    pub pages: usize,
    pub metadata_owned_bytes: u64,
    pub source_window_peak_bytes: u64,
    pub logical_source_bytes: u64,
}
#[derive(Clone, Copy)]
struct Reservation {
    physical: u64,
    total: u64,
    metadata: u64,
}
impl Reservation {
    fn new(physical: u64, mapping: u64, generations: u64, declared: u64) -> Result<Self> {
        let metadata = metadata_bytes(mapping, generations, PAGES, PAGES)?;
        let total = physical
            .checked_add(plan::METADATA_HOST_CAP)
            .and_then(|n| n.checked_add(plan::SOURCE_WINDOW_HOST_BYTES))
            .ok_or(rejected())?;
        if physical < plan::LOGICAL_PUBLIC_RAM_BYTES || declared < total || declared > HOST_CAP {
            return Err(rejected());
        }
        Ok(Self {
            physical,
            total,
            metadata,
        })
    }
}
#[derive(Default)]
struct HeapCharges {
    bytes: [u64; 16],
    total: u64,
    count: usize,
}
impl HeapCharges {
    fn admit(&mut self, p: Placement, bytes: u64, cap: u64) -> Result<()> {
        self.admit_with_limit(p, bytes, cap, PAGES)
    }
    fn admit_with_limit(&mut self, p: Placement, bytes: u64, cap: u64, limit: usize) -> Result<()> {
        let heap = self.bytes.get_mut(p.heap as usize).ok_or(rejected())?;
        let next = heap.checked_add(bytes).ok_or(rejected())?;
        let total = self.total.checked_add(bytes).ok_or(rejected())?;
        let count = self.count.checked_add(1).ok_or(rejected())?;
        if next > p.heap_bytes || total > cap || count > limit {
            return Err(rejected());
        }
        *heap = next;
        self.total = total;
        self.count = count;
        Ok(())
    }
}

struct Backing {
    device: wgpu::Device,
    raw: Arc<ash::Device>,
    raw_buffer: Option<vk::Buffer>,
    memory: Option<vk::DeviceMemory>,
    bytes: u64,
}
impl Backing {
    fn map(&self) -> Result<Mapped<'_>> {
        let memory = self.memory.ok_or(rejected())?;
        // SAFETY: owned HOST_VISIBLE allocation, offset0/fullsize; this is only
        // called before publication or after authenticated detach plus join.
        let pointer = unsafe {
            self.raw
                .map_memory(memory, 0, self.bytes, vk::MemoryMapFlags::empty())
        }
        .map_err(|_| "public host expert map failed")?
        .cast::<u8>();
        Ok(Mapped {
            backing: self,
            pointer,
        })
    }
    fn release(&mut self) {
        // SAFETY: caller proved no remaining consumers; raw_buffer, when Some,
        // has never transferred into wgpu. Destroy VkBuffer before its memory.
        unsafe {
            if let Some(b) = self.raw_buffer.take() {
                self.raw.destroy_buffer(b, None);
            }
            if let Some(m) = self.memory.take() {
                self.raw.free_memory(m, None);
            }
        }
    }
}
impl Drop for Backing {
    fn drop(&mut self) {
        self.release();
    }
}
struct Mapped<'a> {
    backing: &'a Backing,
    pointer: *mut u8,
}
impl Mapped<'_> {
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: sole map guard, exact checked allocation; no other reference
        // to the map is retained while a mutable slice is live.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.backing.bytes as usize) }
    }
    fn cache(&self, coherent: bool, flush: bool) -> Result<()> {
        if coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(self.backing.memory.ok_or(rejected())?)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: full allocation is mapped; WHOLE_SIZE handles the final atom.
        let result = unsafe {
            if flush {
                self.backing.raw.flush_mapped_memory_ranges(&[range])
            } else {
                self.backing.raw.invalidate_mapped_memory_ranges(&[range])
            }
        };
        result.map_err(|_| "public host expert cache operation failed")
    }
    fn zero_and_audit(&mut self) -> bool {
        self.bytes_mut().zeroize();
        (0..self.backing.bytes as usize).all(|i| {
            // SAFETY: exclusive full map retained through this volatile audit.
            unsafe { std::ptr::read_volatile(self.pointer.add(i)) == 0 }
        })
    }
}
impl Drop for Mapped<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard exists only after successful map and unmaps once.
        unsafe { self.backing.raw.unmap_memory(self.backing.memory.unwrap()) }
    }
}

fn readonly_extent(logical: u64, backing: u64, first: u64, bytes: u64) -> Result<(usize, usize)> {
    if logical == 0
        || logical > backing
        || backing > plan::MAX_BINDING_BYTES
        || bytes == 0
        || first.checked_add(bytes).is_none_or(|end| end > logical)
    {
        return Err(rejected());
    }
    let first = usize::try_from(first).map_err(|_| rejected())?;
    let bytes = usize::try_from(bytes).map_err(|_| rejected())?;
    Ok((first, bytes))
}
fn require_readonly_phase(phase: Phase) -> Result<()> {
    if phase != Phase::Registered {
        return Err(rejected());
    }
    Ok(())
}
fn same_readonly_binding_shape(a: &Binding, b: &Binding) -> bool {
    // Opaque descriptor identity stays inside the runtime. Both bindings come
    // from sealed producer-owned registration; exact resolved buffer identity
    // is independently checked below, not inferred from this shape comparison.
    is_public_external_binding(a)
        && is_public_external_binding(b)
        && a.stream == b.stream
        && a.size == b.size
        && a.offset_start == b.offset_start
        && a.offset_end == b.offset_end
}
fn readonly_resource_shape(
    logical: u64,
    offset: u64,
    size: u64,
    buffer: u64,
    same_original_buffer: bool,
) -> Result<()> {
    if logical == 0 || offset != 0 || size != logical || buffer != logical || !same_original_buffer
    {
        return Err(rejected());
    }
    Ok(())
}
fn distinct_readonly_memories(memories: [Option<vk::DeviceMemory>; 9]) -> Result<()> {
    for (i, memory) in memories.iter().enumerate() {
        if memory.is_none() || memories[..i].contains(memory) {
            return Err(rejected());
        }
    }
    Ok(())
}

/// Private map of an already flushed immutable registered allocation. Unlike
/// the construction/close map, this has NO mutable slice or cache operation.
struct ReadOnlyMapped<'a> {
    owner: &'a PageOwner,
    pointer: *const u8,
}
impl ReadOnlyMapped<'_> {
    fn span(&self, first: u64, bytes: u64) -> Result<&[u8]> {
        let backing = self.owner.backing.as_ref().ok_or(rejected())?;
        let (first, bytes) = readonly_extent(self.owner.logical, backing.bytes, first, bytes)?;
        // SAFETY: sole scoped readonly map of this complete retained allocation;
        // the checked logical range excludes physical padding. Existing source
        // attachment forbids all CPU/GPU writes until joined detach and close.
        Ok(unsafe { std::slice::from_raw_parts(self.pointer.add(first), bytes) })
    }
}
impl Drop for ReadOnlyMapped<'_> {
    fn drop(&mut self) {
        if let Some(backing) = self.owner.backing.as_ref()
            && let Some(memory) = backing.memory
        {
            // SAFETY: this guard alone owns this CPU map. Unmap neither frees
            // the allocation nor mutates the registered GPU source binding.
            unsafe { backing.raw.unmap_memory(memory) };
        }
    }
}

/// Opaque, at most nine same-backing maps. Attachment owns the only public
/// lifetime path; no source pointer, handle, mutable bytes or padding escapes.
pub(crate) struct ReadOnlyLayer<'a> {
    maps: [Option<ReadOnlyMapped<'a>>; 9],
}
impl ReadOnlyLayer<'_> {
    pub(crate) fn check_resource(
        &self,
        role: Role,
        page: usize,
        binding: &Binding,
        resource: &WgpuResource,
    ) -> Result<()> {
        let owner = self.maps[index(0, role, page)?]
            .as_ref()
            .ok_or(rejected())?
            .owner;
        require_readonly_phase(owner.phase)?;
        readonly_resource_shape(
            owner.logical,
            resource.offset,
            resource.size,
            resource.buffer.size(),
            owner.buffer.as_ref() == Some(&resource.buffer),
        )?;
        if !same_readonly_binding_shape(owner.binding.as_ref().ok_or(rejected())?, binding)
            || !is_public_external_binding(binding)
            || binding.size != owner.logical
            || binding.offset_start.is_some()
            || binding.offset_end.is_some()
        {
            return Err(rejected());
        }
        Ok(())
    }
    pub(crate) fn span(&self, role: Role, page: usize, first: u64, bytes: u64) -> Result<&[u8]> {
        let i = index(0, role, page)?;
        self.maps[i].as_ref().ok_or(rejected())?.span(first, bytes)
    }
    pub(crate) fn placements(&self) -> Result<[Placement; 9]> {
        let mut result = [Placement {
            memory_type: 0,
            heap: 0,
            property_flags: 0,
            heap_flags: 0,
            heap_bytes: 0,
        }; 9];
        for (out, map) in result.iter_mut().zip(&self.maps) {
            *out = map.as_ref().ok_or(rejected())?.owner.placement;
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Sealed,
    Exposed,
    Registered,
    Detached,
    Joined,
    Released,
}
fn can_release(phase: Phase) -> bool {
    matches!(phase, Phase::Sealed | Phase::Joined | Phase::Released)
}
#[derive(Clone, Copy)]
struct DetachShape {
    offset: u64,
    size: u64,
    buffer_bytes: u64,
    backing_bytes: u64,
    joined: bool,
    destroyed: bool,
}
fn validate_detach(
    phase: Phase,
    logical: u64,
    backing: u64,
    exact_binding: bool,
    shape: DetachShape,
) -> Result<()> {
    if phase != Phase::Registered
        || !exact_binding
        || shape.offset != 0
        || shape.size != logical
        || shape.buffer_bytes != logical
        || shape.backing_bytes != backing
        || !shape.joined
        || !shape.destroyed
    {
        return Err(rejected());
    }
    Ok(())
}
struct PageOwner {
    backing: Option<Backing>,
    buffer: Option<wgpu::Buffer>,
    placement: Placement,
    allocation_alignment: u64,
    logical: u64,
    phase: Phase,
    binding: Option<Binding>,
    shard: usize,
}
impl Drop for PageOwner {
    fn drop(&mut self) {
        if !can_release(self.phase) {
            // No detach receipt: even a successful generic device poll would
            // not grant authority over CubeCL's unflushed scheduled consumers.
            if let Some(buffer) = self.buffer.take() {
                std::mem::forget(buffer);
            }
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
        } else if let Some(buffer) = self.buffer.take() {
            buffer.destroy();
        }
    }
}

/// Immutable original-weight resource. No Debug/serialization or writable map.
/// A buffer clone grants no untracked/raw consumer or mutation authority.
pub(crate) struct PageResource {
    plan_identity: PlanIdentity,
    binding: GraphBindingId,
    resource: WgpuResource,
    layer: usize,
    role: Role,
    page: usize,
    backing_bytes: u64,
    allocation_alignment: u64,
    placement: Placement,
    source_sha256: [u8; 32],
}
impl PageResource {
    pub(crate) fn plan_identity(&self) -> PlanIdentity {
        self.plan_identity
    }
    pub(crate) fn binding(&self) -> GraphBindingId {
        self.binding
    }
    /// Require this exact frozen plan, not another mapping with equal geometry.
    /// Its opaque monotonic identity is independent of addresses/lifetimes.
    pub(crate) fn belongs_to(&self, plan: &Plan<'_, '_>) -> bool {
        self.plan_identity == plan.identity() && self.binding == plan.binding()
    }
    /// Coordinates are issued only by the mirror from the validated plan.
    /// Same-shaped gate/up resources must still match their original role.
    pub(crate) fn coordinates(&self) -> (usize, Role, usize) {
        (self.layer, self.role, self.page)
    }
    pub(crate) fn resource(&self) -> &WgpuResource {
        &self.resource
    }
    pub(crate) fn backing_bytes(&self) -> u64 {
        self.backing_bytes
    }
    /// Actual native requirement authenticated before allocation and import.
    /// The plan's rounding is a conservative envelope, not driver alignment.
    pub(crate) fn allocation_alignment(&self) -> u64 {
        self.allocation_alignment
    }
    pub(crate) fn placement(&self) -> Placement {
        self.placement
    }
    pub(crate) fn source_sha256(&self) -> [u8; 32] {
        self.source_sha256
    }
}
pub(crate) struct Mirror<'plan, 'model, 'file> {
    plan: &'plan Plan<'model, 'file>,
    device: wgpu::Device,
    pages: Vec<PageOwner>,
    ledger: Ledger,
    strata_exchange: Option<strata_gpu_exchange::ExchangeRuntime>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Close {
    pub ledger: Ledger,
    pub full_allocation_zero: bool,
    pub checked_detach_pages: usize,
}

fn copy_range(destination: &mut [u8], logical: usize, offset: u64, source: &[u8]) -> Result<()> {
    let offset = usize::try_from(offset).map_err(|_| rejected())?;
    let end = offset.checked_add(source.len()).ok_or(rejected())?;
    if source.is_empty()
        || source.len() > plan::SOURCE_WINDOW_HOST_BYTES as usize
        || logical > destination.len()
        || end > logical
    {
        return Err(rejected());
    }
    destination[offset..end].copy_from_slice(source);
    Ok(())
}
fn join(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|_| "public host expert submitted work join failed")?;
    Ok(())
}

impl<'plan, 'model, 'file> Mirror<'plan, 'model, 'file> {
    /// # Safety
    /// The actual pinned DAS files must have passed full SHA256 authentication;
    /// these exact banks must be completed, physically proved on this actual
    /// device and frozen against GPU fills/alias writes throughout mirror life.
    /// The caller separately reserves `host_reserved_bytes` in its process/RAM
    /// admission before invoking this constructor. Every exported page has one
    /// registry/stream owner; no second registration, untracked/raw consumer,
    /// or alias/mutation of the page/backing is permitted. No private content
    /// enters. Source identities are later checked before every consuming wave.
    pub(crate) unsafe fn new(
        plan: &'plan Plan<'model, 'file>,
        graph: BoundGraph<'model, 'file>,
        model: &'file GgufFile,
        banks: [&ExpertBanks; plan::LAYERS],
        device: &wgpu::Device,
        host_reserved_bytes: u64,
    ) -> Result<Self> {
        plan.check_frozen(graph, model, banks)
            .map_err(|_| rejected())?;
        if model.file_len() != 83_617_662_656
            || model.shard_count() != SOURCE_SHA256.len()
            || device.limits().max_storage_buffer_binding_size < plan::MAX_BINDING_BYTES
            || device.limits().max_buffer_size < plan::MAX_BINDING_BYTES
        {
            return Err(rejected());
        }
        let budget = plan.budget();
        // Whole-pool precharge precedes Vec/window/native allocations. It is a
        // checked reservation, not a claim that the OS prefaulted/pinned RAM.
        let reserved = Reservation::new(
            budget.physical_public_ram_envelope_bytes,
            budget.mapping_metadata_bytes,
            budget.source_generation_metadata_bytes,
            host_reserved_bytes,
        )?;
        let mut pages = Vec::new();
        pages
            .try_reserve_exact(PAGES)
            .map_err(|_| "public host expert bounded metadata allocation")?;
        if pages.capacity() != PAGES {
            return Err(rejected());
        }
        metadata_bytes(
            budget.mapping_metadata_bytes,
            budget.source_generation_metadata_bytes,
            pages.capacity(),
            PAGES,
        )?;
        let mut window = Zeroizing::new(Vec::new());
        window
            .try_reserve_exact(plan::SOURCE_WINDOW_HOST_BYTES as usize)
            .map_err(|_| "public host expert bounded source allocation")?;
        if window.capacity() > plan::SOURCE_WINDOW_HOST_BYTES as usize {
            return Err(rejected());
        }
        window.resize(plan::SOURCE_WINDOW_HOST_BYTES as usize, 0);
        let mut charges = HeapCharges::default();
        // SAFETY: borrowing only actual Vulkan identity/properties, never raw
        // destruction through the HAL guard. Imports use this exact device.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or("public host expert Vulkan required")?;
        let raw = Arc::new(hal.raw_device().clone());
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(hal.raw_physical_device())
        };
        let queue_families = hal.kuno_buffer_queue_families();
        drop(hal);
        for layer in 0..plan::LAYERS {
            for role in ROLES {
                for p in 0..3 {
                    let page = plan.page(layer, role, p).map_err(|_| rejected())?;
                    let mut descriptor = vk::BufferCreateInfo::default()
                        .size(page.logical_bytes())
                        .usage(
                            vk::BufferUsageFlags::STORAGE_BUFFER
                                | if crate::strata_intel_core::enabled() {
                                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                                } else {
                                    vk::BufferUsageFlags::empty()
                                },
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    if let Some((families, count)) = &queue_families {
                        descriptor = descriptor
                            .sharing_mode(vk::SharingMode::CONCURRENT)
                            .queue_family_indices(&families[..*count]);
                    }
                    let buffer = unsafe { raw.create_buffer(&descriptor, None) }
                        .map_err(|_| "public host expert native buffer creation failed")?;
                    let mut backing = Backing {
                        device: device.clone(),
                        raw: raw.clone(),
                        raw_buffer: Some(buffer),
                        memory: None,
                        bytes: 0,
                    };
                    let requirements =
                        Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) });
                    check_requirements(
                        page.logical_bytes(),
                        page.physical_envelope_bytes(),
                        requirements,
                    )?;
                    page.check_actual_allocation(requirements.bytes, requirements.alignment)
                        .map_err(|_| rejected())?;
                    let placement =
                        select_host_type(&properties, requirements.types, requirements.bytes)?;
                    // Physical heap/pool precharge occurs before vkAllocateMemory.
                    charges.admit(placement, requirements.bytes, reserved.physical)?;
                    // Each page owns a single complete allocation at offset0.
                    // Dedicated allocation also satisfies drivers that require
                    // it for this buffer; no aliasing/suballocation is introduced.
                    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
                    let mut flags = vk::MemoryAllocateFlagsInfo::default()
                        .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                    let mut allocation = vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.bytes)
                        .memory_type_index(placement.memory_type)
                        .push_next(&mut dedicated);
                    if crate::strata_intel_core::enabled() {
                        allocation = allocation.push_next(&mut flags);
                    }
                    backing.memory = Some(
                        unsafe { raw.allocate_memory(&allocation, None) }
                            .map_err(|_| "public host expert nonlocal allocation failed")?,
                    );
                    backing.bytes = requirements.bytes;
                    unsafe { raw.bind_buffer_memory(buffer, backing.memory.unwrap(), 0) }
                        .map_err(|_| "public host expert memory binding failed")?;
                    if Requirements::from(unsafe { raw.get_buffer_memory_requirements(buffer) })
                        != requirements
                    {
                        return Err(rejected());
                    }
                    let mut mapped = backing.map()?;
                    mapped.bytes_mut().zeroize();
                    let first = page.first_virtual_slot() - plan::VRAM_SLOTS;
                    let mut shard = None;
                    for ordinal in first..first + page.slots() {
                        let range = plan.range(layer, role, ordinal).map_err(|_| rejected())?;
                        let source = range.tensor();
                        let actual = source.info().shard_index();
                        if actual >= SOURCE_SHA256.len()
                            || shard.is_some_and(|previous| previous != actual)
                            || range.page() != p
                        {
                            return Err(rejected());
                        }
                        shard = Some(actual);
                        let length = usize::try_from(range.bytes()).map_err(|_| rejected())?;
                        let bytes = window.get_mut(..length).ok_or(rejected())?;
                        source
                            .read_range(range.source_offset(), bytes)
                            .map_err(|_| "public host expert original source read failed")?;
                        copy_range(
                            mapped.bytes_mut(),
                            page.logical_bytes() as usize,
                            range.destination_offset(),
                            bytes,
                        )?;
                        bytes.zeroize();
                    }
                    mapped.cache(placement.coherent(), true)?;
                    drop(mapped);
                    let shard = shard.ok_or(rejected())?;
                    // SAFETY: initialized+flushed full allocation, same device, exact
                    // STORAGE-only descriptor. HAL takes VkBuffer, not its memory.
                    backing.raw_buffer = None;
                    let imported = unsafe {
                        device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                            wgpu::hal::vulkan::Buffer::from_raw(buffer),
                            &wgpu::BufferDescriptor {
                                label: Some("original-public-host-expert-page"),
                                size: page.logical_bytes(),
                                usage: wgpu::BufferUsages::STORAGE,
                                mapped_at_creation: false,
                            },
                        )
                    };
                    pages.push(PageOwner {
                        backing: Some(backing),
                        buffer: Some(imported),
                        placement,
                        allocation_alignment: requirements.alignment,
                        logical: page.logical_bytes(),
                        phase: Phase::Sealed,
                        binding: None,
                        shard,
                    });
                }
            }
        }
        plan.check_frozen(graph, model, banks)
            .map_err(|_| rejected())?;
        if charges.count != PAGES || pages.len() != PAGES {
            return Err(rejected());
        }
        let ledger = Ledger {
            reserved_host_bytes: reserved.total,
            physical_envelope_bytes: reserved.physical,
            live_backing_bytes: charges.total,
            total_allocated_bytes: charges.total,
            total_freed_bytes: 0,
            pages: PAGES,
            metadata_owned_bytes: reserved.metadata,
            source_window_peak_bytes: window.capacity() as u64,
            logical_source_bytes: plan::LOGICAL_PUBLIC_RAM_BYTES,
        };
        drop(window);
        Ok(Self {
            plan,
            device: device.clone(),
            pages,
            ledger,
            strata_exchange: None,
        })
    }
    pub(crate) fn ledger(&self) -> Ledger {
        self.ledger
    }
    /// Authenticate a full registered page against its actual retained owner.
    /// This returns no resource clone or native memory identity.
    pub(crate) fn check_readonly_page(
        &self,
        expected_plan: &Plan<'_, '_>,
        layer: usize,
        role: Role,
        page: usize,
        binding: &Binding,
        resource: &WgpuResource,
    ) -> Result<()> {
        if !std::ptr::eq(self.plan, expected_plan)
            || self.plan.identity() != expected_plan.identity()
            || self.plan.binding() != expected_plan.binding()
        {
            return Err(rejected());
        }
        let owner = self
            .pages
            .get(index(layer, role, page)?)
            .ok_or(rejected())?;
        require_readonly_phase(owner.phase)?;
        let expected = self.plan.page(layer, role, page).map_err(|_| rejected())?;
        let backing = owner.backing.as_ref().ok_or(rejected())?;
        expected
            .check_actual_allocation(backing.bytes, owner.allocation_alignment)
            .map_err(|_| rejected())?;
        readonly_resource_shape(
            owner.logical,
            resource.offset,
            resource.size,
            resource.buffer.size(),
            owner.buffer.as_ref() == Some(&resource.buffer),
        )?;
        if !same_readonly_binding_shape(owner.binding.as_ref().ok_or(rejected())?, binding)
            || !is_public_external_binding(binding)
            || binding.size != expected.logical_bytes()
            || binding.offset_start.is_some()
            || binding.offset_end.is_some()
            || owner.logical != expected.logical_bytes()
            || backing.device != self.device
            || backing.memory.is_none()
            || backing.raw_buffer.is_some()
            || owner.placement.property_flags & vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() == 0
            || owner.placement.property_flags
                & (vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::LAZILY_ALLOCATED
                    | vk::MemoryPropertyFlags::PROTECTED)
                    .as_raw()
                != 0
            || owner.placement.heap_flags & vk::MemoryHeapFlags::DEVICE_LOCAL.as_raw() != 0
        {
            return Err(rejected());
        }
        Ok(())
    }
    /// Called only after all nine actual registrations have been authenticated
    /// by the exclusive Attachment borrow. No mutable registered map exists.
    pub(crate) fn borrow_readonly_layer(&self, layer: usize) -> Result<ReadOnlyLayer<'_>> {
        if layer >= plan::LAYERS {
            return Err(rejected());
        }
        // Preflight every owner and pair BEFORE the first vkMapMemory call.
        let mut memories = [None; 9];
        for (i, memory) in memories.iter_mut().enumerate() {
            let owner = self.pages.get(layer * 9 + i).ok_or(rejected())?;
            require_readonly_phase(owner.phase)?;
            let backing = owner.backing.as_ref().ok_or(rejected())?;
            *memory = backing.memory;
        }
        distinct_readonly_memories(memories)?;
        let mut mapped = ReadOnlyLayer {
            maps: std::array::from_fn(|_| None),
        };
        for (i, map) in mapped.maps.iter_mut().enumerate() {
            let owner = &self.pages[layer * 9 + i];
            let backing = owner.backing.as_ref().ok_or(rejected())?;
            // SAFETY: unique complete HOST_VISIBLE allocation, immutable and
            // flushed before publication; Attachment retains its backing and
            // excludes another CPU map/close until this scoped borrow ends.
            let pointer = unsafe {
                backing.raw.map_memory(
                    backing.memory.ok_or(rejected())?,
                    0,
                    backing.bytes,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .map_err(|_| "public host expert readonly map failed")?
            .cast::<u8>() as *const u8;
            *map = Some(ReadOnlyMapped { owner, pointer });
        }
        Ok(mapped)
    }
    /// # Safety
    /// Strata's parent has joined all GPU consumers and excluded every CPU
    /// read borrow. Only original public weight bytes may enter this swap;
    /// all three roles and the logical slot bijection publish together later.
    #[allow(dead_code)] // Reference staging path, no longer selected by the port.
    pub(crate) unsafe fn swap_strata_public_role(
        &mut self,
        layer: usize,
        role: usize,
        page: usize,
        first: usize,
        outgoing: &[u8],
    ) -> Result<zeroize::Zeroizing<Vec<u8>>> {
        if !crate::strata_port::enabled()
            || layer >= 48
            || role >= 3
            || page >= 3
            || outgoing.is_empty()
            || outgoing.len() > 1024 * 1024
            || !first.is_multiple_of(4)
            || !outgoing.len().is_multiple_of(4)
        {
            return Err(rejected());
        }
        let owner = self
            .pages
            .get(layer * 9 + role * 3 + page)
            .ok_or_else(rejected)?;
        require_readonly_phase(owner.phase)?;
        let end = first.checked_add(outgoing.len()).ok_or_else(rejected)?;
        if end > owner.logical as usize {
            return Err(rejected());
        }
        let backing = owner.backing.as_ref().ok_or_else(rejected)?;
        let mut map = backing.map()?;
        map.cache(owner.placement.coherent(), false)?;
        let incoming = zeroize::Zeroizing::new(map.bytes_mut()[first..end].to_vec());
        map.bytes_mut()[first..end].copy_from_slice(outgoing);
        map.cache(owner.placement.coherent(), true)?;
        Ok(incoming)
    }
    pub(crate) fn check_frozen(
        &self,
        graph: BoundGraph<'_, '_>,
        model: &GgufFile,
        banks: [&ExpertBanks; plan::LAYERS],
        device: &wgpu::Device,
    ) -> Result<()> {
        if device != &self.device {
            return Err(rejected());
        }
        self.plan
            .check_frozen(graph, model, banks)
            .map_err(|_| rejected())
    }
    /// Export once; caller registers this complete immutable PUBLIC source only
    /// once on the selected stream, retaining its actual binding for handoff.
    pub(crate) fn page_resource(
        &mut self,
        device: &wgpu::Device,
        layer: usize,
        role: Role,
        p: usize,
    ) -> Result<PageResource> {
        if device != &self.device {
            return Err(rejected());
        }
        let owner = self
            .pages
            .get_mut(index(layer, role, p)?)
            .ok_or(rejected())?;
        if owner.phase != Phase::Sealed {
            return Err(rejected());
        }
        owner.phase = Phase::Exposed;
        Ok(PageResource {
            plan_identity: self.plan.identity(),
            binding: self.plan.binding(),
            resource: WgpuResource {
                buffer: owner.buffer.as_ref().unwrap().clone(),
                offset: 0,
                size: owner.logical,
            },
            layer,
            role,
            page: p,
            backing_bytes: owner.backing.as_ref().unwrap().bytes,
            allocation_alignment: owner.allocation_alignment,
            placement: owner.placement,
            source_sha256: SOURCE_SHA256[owner.shard],
        })
    }
    /// Retain the exact full source identity returned by registration. Resource
    /// identity is independently checked again in the consuming detach receipt.
    pub(crate) fn registered(
        &mut self,
        layer: usize,
        role: Role,
        p: usize,
        binding: Binding,
    ) -> Result<()> {
        let owner = self
            .pages
            .get_mut(index(layer, role, p)?)
            .ok_or(rejected())?;
        if owner.phase != Phase::Exposed
            || !is_public_external_binding(&binding)
            || binding.size != owner.logical
            || binding.offset_start.is_some()
            || binding.offset_end.is_some()
        {
            return Err(rejected());
        }
        owner.binding = Some(binding);
        owner.phase = Phase::Registered;
        Ok(())
    }
    /// All432 exact sources must detach before any CPU map/free. A failed close
    /// returns no accepted receipt; Drop retains any backing lacking safe release.
    pub(crate) fn close(
        mut self,
        receipts: Vec<PublicExternalDetachReceipt<WgpuResource>>,
    ) -> Result<Close> {
        let budget = self.plan.budget();
        let metadata = metadata_bytes(
            budget.mapping_metadata_bytes,
            budget.source_generation_metadata_bytes,
            self.pages.capacity(),
            receipts.capacity(),
        )?;
        self.ledger.metadata_owned_bytes = self.ledger.metadata_owned_bytes.max(metadata);
        if receipts.len() != PAGES || self.pages.len() != PAGES {
            return Err(rejected());
        }
        let mut matched = Zeroizing::new([false; PAGES]);
        for receipt in &receipts {
            let resource = receipt.resource();
            let i = self
                .pages
                .iter()
                .position(|p| p.buffer.as_ref() == Some(&resource.buffer))
                .ok_or(rejected())?;
            let owner = &self.pages[i];
            if matched[i] {
                return Err(rejected());
            }
            validate_detach(
                owner.phase,
                owner.logical,
                owner.backing.as_ref().unwrap().bytes,
                receipt.matches_source(owner.binding.as_ref().ok_or(rejected())?),
                DetachShape {
                    offset: resource.offset,
                    size: resource.size,
                    buffer_bytes: receipt.buffer_bytes,
                    backing_bytes: receipt.backing_bytes,
                    joined: receipt.queue_joined,
                    destroyed: receipt.source_destroyed,
                },
            )?;
            matched[i] = true;
        }
        if matched.iter().any(|&p| !p) {
            return Err(rejected());
        }
        for page in &mut self.pages {
            page.phase = Phase::Detached;
            page.buffer.as_ref().unwrap().destroy();
        }
        // Every detach already joined actual submitted consumers; explicit outer
        // join is still checked. On join failure, retain all physical backing.
        if let Err(error) = join(&self.device) {
            self.leak_all();
            return Err(error);
        }
        for page in &mut self.pages {
            page.phase = Phase::Joined;
        }
        for page in &self.pages {
            let backing = page.backing.as_ref().unwrap();
            let mut mapped = backing.map()?;
            mapped.cache(page.placement.coherent(), false)?;
            if !mapped.zero_and_audit() {
                return Err("public host expert full zero audit failed");
            }
            mapped.cache(page.placement.coherent(), true)?;
        }
        if let Err(error) = join(&self.device) {
            self.leak_all();
            return Err(error);
        }
        for page in &mut self.pages {
            page.buffer.take();
            let mut backing = page.backing.take().unwrap();
            let bytes = backing.bytes;
            backing.release();
            page.phase = Phase::Released;
            self.ledger.live_backing_bytes = self
                .ledger
                .live_backing_bytes
                .checked_sub(bytes)
                .ok_or(rejected())?;
            self.ledger.total_freed_bytes = self
                .ledger
                .total_freed_bytes
                .checked_add(bytes)
                .ok_or(rejected())?;
        }
        if self.ledger.live_backing_bytes != 0
            || self.ledger.total_freed_bytes != self.ledger.total_allocated_bytes
        {
            return Err(rejected());
        }
        Ok(Close {
            ledger: self.ledger,
            full_allocation_zero: true,
            checked_detach_pages: receipts.len(),
        })
    }
    fn leak_all(&mut self) {
        for page in &mut self.pages {
            if let Some(buffer) = page.buffer.take() {
                std::mem::forget(buffer);
            }
            if let Some(backing) = page.backing.take() {
                std::mem::forget(backing);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_original_page_identity_and_full_resource_extent_are_required() {
        readonly_resource_shape(100, 0, 100, 100, true).unwrap();
        for args in [
            (100, 0, 100, 100, false),
            (100, 1, 100, 100, true),
            (100, 0, 99, 100, true),
            (100, 0, 101, 100, true),
            (100, 0, 100, 128, true),
            (100, 0, 100, 99, true),
            (0, 0, 0, 0, true),
        ] {
            assert!(readonly_resource_shape(args.0, args.1, args.2, args.3, args.4).is_err());
        }
    }
    #[test]
    fn readonly_payload_extent_excludes_complete_allocation_padding() {
        assert_eq!(readonly_extent(100, 128, 0, 100).unwrap(), (0, 100));
        assert_eq!(readonly_extent(100, 128, 80, 20).unwrap(), (80, 20));
        for args in [
            (100, 128, 100, 1),
            (100, 128, 0, 101),
            (100, 128, 0, 128),
            (100, 99, 0, 1),
            (0, 128, 0, 1),
            (100, 128, 0, 0),
            (100, 128, u64::MAX, 1),
            (100, u64::MAX, 0, 1),
        ] {
            assert!(readonly_extent(args.0, args.1, args.2, args.3).is_err());
        }
    }
    #[test]
    fn only_registered_pages_allow_a_readonly_cpu_map() {
        for phase in [
            Phase::Sealed,
            Phase::Exposed,
            Phase::Registered,
            Phase::Detached,
            Phase::Joined,
            Phase::Released,
        ] {
            assert_eq!(
                require_readonly_phase(phase).is_ok(),
                phase == Phase::Registered
            );
        }
    }
    #[test]
    fn all_nine_actual_memories_preflight_before_first_map() {
        use ash::vk::Handle as _;
        let good = std::array::from_fn(|i| Some(vk::DeviceMemory::from_raw(i as u64 + 1)));
        distinct_readonly_memories(good).unwrap();
        for i in 0..9 {
            let mut absent = good;
            absent[i] = None;
            assert!(distinct_readonly_memories(absent).is_err());
            if i != 0 {
                let mut alias = good;
                alias[i] = good[0];
                assert!(distinct_readonly_memories(alias).is_err());
            }
        }
    }
    fn inventory() -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 4,
            memory_heap_count: 2,
            ..Default::default()
        };
        p.memory_heaps[0] = vk::MemoryHeap {
            size: HOST_CAP,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        p.memory_heaps[1] = vk::MemoryHeap {
            size: HOST_CAP,
            flags: vk::MemoryHeapFlags::empty(),
        };
        p.memory_types[0] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        p.memory_types[1] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        p.memory_types[2] = vk::MemoryType {
            heap_index: 1,
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        };
        p.memory_types[3] = vk::MemoryType {
            heap_index: 1,
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED,
        };
        p
    }
    #[test]
    fn placement_rejects_bar_and_invalid_inventory_without_device_local_fallback() {
        let mut p = inventory();
        assert_eq!(select_host_type(&p, 15, 4096).unwrap().memory_type, 3);
        assert_eq!(select_host_type(&p, 7, 4096).unwrap().memory_type, 2);
        assert!(select_host_type(&p, 3, 4096).is_err());
        p.memory_types[2].heap_index = 0;
        assert!(select_host_type(&p, 4, 4096).is_err());
        p.memory_types[2].heap_index = 9;
        assert!(select_host_type(&p, 4, 4096).is_err());
        p.memory_type_count = 33;
        assert!(select_host_type(&p, 15, 4096).is_err());
        assert!(select_host_type(&inventory(), 15, plan::MAX_BINDING_BYTES + 4).is_err());
    }
    #[test]
    fn requirement_envelope_alignment_and_changes_fail_closed() {
        let good = Requirements {
            bytes: 65536,
            alignment: 4096,
            types: 4,
        };
        check_requirements(65000, 65536, good).unwrap();
        for r in [
            Requirements {
                bytes: 64000,
                ..good
            },
            Requirements {
                bytes: 131072,
                ..good
            },
            Requirements {
                alignment: 0,
                ..good
            },
            Requirements {
                alignment: 3,
                ..good
            },
            Requirements {
                alignment: 131072,
                ..good
            },
            Requirements { types: 0, ..good },
        ] {
            assert!(check_requirements(65000, 65536, r).is_err());
            assert_ne!(r, good);
        }
        assert!(check_requirements(31, 65536, good).is_err());
        assert!(check_requirements(65001, 65536, good).is_err());
    }
    #[test]
    fn whole_precharge_heap_bounds_and_failed_admit_do_not_grant_credit() {
        let cap = plan::LOGICAL_PUBLIC_RAM_BYTES;
        let total = cap + plan::METADATA_HOST_CAP + plan::SOURCE_WINDOW_HOST_BYTES;
        let reservation = Reservation::new(cap, 128 * 1024, 1536 * 1024, total).unwrap();
        assert_eq!(reservation.total, total);
        assert!(Reservation::new(cap, 128 * 1024, 1536 * 1024, total - 1).is_err());
        assert!(Reservation::new(u64::MAX, 0, 0, HOST_CAP).is_err());
        assert!(metadata_bytes(0, 0, usize::MAX, 0).is_err());
        assert!(metadata_bytes(plan::METADATA_HOST_CAP, 1, 1, 1).is_err());
        let exact = metadata_bytes(0, 0, PAGES, PAGES).unwrap();
        assert_eq!(
            metadata_bytes(0, 0, PAGES, PAGES + 1).unwrap() - exact,
            size_of::<PublicExternalDetachReceipt<WgpuResource>>() as u64,
        );
        let placement = select_host_type(&inventory(), 4, 4096).unwrap();
        let mut charges = HeapCharges::default();
        charges.admit(placement, 4096, 8192).unwrap();
        assert!(charges.admit(placement, 8192, 8192).is_err());
        assert_eq!((charges.total, charges.count), (4096, 1));
        let small = Placement {
            heap_bytes: 4096,
            ..placement
        };
        assert!(charges.admit(small, 4096, 8192).is_err());
        assert_eq!(charges.total, 4096);
    }
    #[test]
    fn direct_local_fill_preserves_other_ranges_and_every_physical_tail() {
        let mut allocation = [0u8; 32];
        copy_range(&mut allocation, 20, 4, &[1, 2, 3, 4]).unwrap();
        copy_range(&mut allocation, 20, 12, &[5, 6, 7, 8]).unwrap();
        assert_eq!(&allocation[4..8], &[1, 2, 3, 4]);
        assert_eq!(&allocation[12..16], &[5, 6, 7, 8]);
        assert!(
            allocation[..4]
                .iter()
                .chain(&allocation[8..12])
                .chain(&allocation[16..])
                .all(|&b| b == 0)
        );
        let original = allocation;
        for offset in [u64::MAX, 19, 31] {
            assert!(copy_range(&mut allocation, 20, offset, &[1, 2, 3, 4]).is_err());
        }
        assert!(copy_range(&mut allocation, 33, 0, &[1]).is_err());
        assert!(copy_range(&mut allocation, 20, 0, &[]).is_err());
        assert_eq!(allocation, original);
    }
    #[test]
    fn release_policy_and_all432_coordinates_are_bounded() {
        assert!(can_release(Phase::Sealed));
        assert!(!can_release(Phase::Detached));
        assert!(can_release(Phase::Joined));
        assert!(can_release(Phase::Released));
        assert!(!can_release(Phase::Exposed));
        assert!(!can_release(Phase::Registered));
        let mut seen = [false; PAGES];
        for l in 0..48 {
            for r in ROLES {
                for p in 0..3 {
                    let i = index(l, r, p).unwrap();
                    assert!(!seen[i]);
                    seen[i] = true;
                }
            }
        }
        assert!(seen.into_iter().all(|v| v));
        assert!(index(48, Role::Gate, 0).is_err());
        assert!(index(0, Role::Gate, 3).is_err());
        for (expected, pin) in crate::daslab_carry_reference::SHARD_SHA256
            .iter()
            .zip(SOURCE_SHA256)
        {
            let actual: String = pin.into_iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(&actual, expected);
        }
    }
    #[test]
    fn detach_requires_original_complete_identity_and_join_before_release() {
        let shape = DetachShape {
            offset: 0,
            size: 65000,
            buffer_bytes: 65000,
            backing_bytes: 65536,
            joined: true,
            destroyed: true,
        };
        validate_detach(Phase::Registered, 65000, 65536, true, shape).unwrap();
        for phase in [
            Phase::Sealed,
            Phase::Exposed,
            Phase::Detached,
            Phase::Joined,
            Phase::Released,
        ] {
            assert!(validate_detach(phase, 65000, 65536, true, shape).is_err());
        }
        assert!(validate_detach(Phase::Registered, 65000, 65536, false, shape).is_err());
        for bad in [
            DetachShape { offset: 4, ..shape },
            DetachShape {
                size: 64996,
                ..shape
            },
            DetachShape {
                buffer_bytes: 65536,
                ..shape
            },
            DetachShape {
                backing_bytes: 65000,
                ..shape
            },
            DetachShape {
                joined: false,
                ..shape
            },
            DetachShape {
                destroyed: false,
                ..shape
            },
        ] {
            assert!(validate_detach(Phase::Registered, 65000, 65536, true, bad).is_err());
        }
        assert!(!can_release(Phase::Detached));
    }
}
