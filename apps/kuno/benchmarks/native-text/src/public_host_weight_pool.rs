//! Small, immutable, compiled-public Vulkan host-weight allocation experiment.
//!
//! This deliberately does not register a CubeCL allocation or accept model /
//! request bytes. Its sole source is the fixed original Q2_0 block fixture.
//! HOST_VISIBLE alone is insufficient: both the memory type and its heap must
//! be non-DEVICE_LOCAL. The imported HAL buffer has no managed backing, avoiding
//! wgpu-hal 29.0.4's raw-managed memory-counter asymmetry. This module owns the
//! VkDeviceMemory separately, destroys the wgpu buffer to prohibit future use,
//! joins submitted work, and only then releases that memory. HAL buffer-memory
//! counters therefore exclude this pool; the explicit ledger is authoritative.
//! No portable pinning, throughput, private erasure, or full-mirror claim.

use ash::vk;
use std::time::Duration;
use zeroize::Zeroize;

type Result<T> = std::result::Result<T, &'static str>;
pub(crate) const BUFFER_BYTES: u64 = 64 * 1024;
pub(crate) const HOST_CAP: u64 = 1024 * 1024;
pub(crate) const ROWS: usize = 4;
pub(crate) const COLUMNS: usize = 64;
pub(crate) const BLOCK_BYTES: usize = 18;
const SCALE_BITS: [u16; ROWS] = [0x3800, 0x3400, 0x3c00, 0x3000];

/// Only these immutable compiled-public sources can enter the allocation path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fixture {
    Aligned,
    Prefix3,
    BadPublicTail,
}
impl Fixture {
    pub(crate) const fn prefix(self) -> usize {
        match self {
            Self::Aligned => 0,
            Self::Prefix3 | Self::BadPublicTail => 3,
        }
    }
    pub(crate) fn byte(self, index: usize) -> u8 {
        if self == Self::BadPublicTail && index == BUFFER_BYTES as usize - 1 {
            return 0x5a;
        }
        let Some(index) = index.checked_sub(self.prefix()) else {
            return 0;
        };
        if index >= ROWS * BLOCK_BYTES {
            return 0;
        }
        let (row, column) = (index / BLOCK_BYTES, index % BLOCK_BYTES);
        match column {
            0 => SCALE_BITS[row] as u8,
            1 => (SCALE_BITS[row] >> 8) as u8,
            _ => ((column - 2) * 73 + row * 41 + 9) as u8,
        }
    }
    pub(crate) fn packed(self) -> [u8; ROWS * BLOCK_BYTES] {
        std::array::from_fn(|i| self.byte(self.prefix() + i))
    }
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

/// Pure selection is tested independently; there is no device-local fallback.
fn select_host_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    allowed: u32,
    bytes: u64,
) -> Result<Placement> {
    if bytes == 0
        || bytes > HOST_CAP
        || properties.memory_type_count > 32
        || properties.memory_heap_count > 16
    {
        return Err("public host weight memory inventory/extent");
    }
    let mut best: Option<(u32, Placement)> = None;
    for index in 0..properties.memory_type_count {
        if allowed & (1 << index) == 0 {
            continue;
        }
        let memory = properties.memory_types[index as usize];
        if memory.heap_index >= properties.memory_heap_count {
            return Err("public host weight memory heap index");
        }
        let heap = properties.memory_heaps[memory.heap_index as usize];
        let flags = memory.property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            || flags.intersects(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::PROTECTED
                    | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            )
            || heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            || heap.size < bytes
        {
            continue;
        }
        let score = u32::from(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)) * 2
            + u32::from(flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT));
        let placement = Placement {
            memory_type: index,
            heap: memory.heap_index,
            property_flags: flags.as_raw(),
            heap_flags: heap.flags.as_raw(),
            heap_bytes: heap.size,
        };
        if best.is_none_or(|(prior, _)| score > prior) {
            best = Some((score, placement));
        }
    }
    best.map(|(_, placement)| placement)
        .ok_or("public host weight nonlocal compatible heap unavailable")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllocationPlan {
    allocation_bytes: u64,
    placement: Placement,
}
fn plan(
    requirements: vk::MemoryRequirements,
    properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<AllocationPlan> {
    if requirements.size < BUFFER_BYTES
        || requirements.size > HOST_CAP
        || requirements.alignment == 0
        || !requirements.alignment.is_power_of_two()
        || !requirements.size.is_multiple_of(requirements.alignment)
    {
        return Err("public host weight physical allocation plan");
    }
    Ok(AllocationPlan {
        allocation_bytes: requirements.size,
        placement: select_host_type(properties, requirements.memory_type_bits, requirements.size)?,
    })
}

#[derive(Default, Debug)]
struct Ledger {
    live: Option<u64>,
    allocated: u64,
    freed: u64,
}
impl Ledger {
    fn allocate(&mut self, bytes: u64) -> Result<()> {
        if self.live.is_some() || self.allocated != 0 || !(BUFFER_BYTES..=HOST_CAP).contains(&bytes)
        {
            return Err("public host weight ledger allocation");
        }
        self.live = Some(bytes);
        self.allocated = bytes;
        Ok(())
    }
    fn free(&mut self, bytes: u64) -> Result<()> {
        if self.live != Some(bytes) || self.freed != 0 {
            return Err("public host weight ledger release");
        }
        self.live = None;
        self.freed = bytes;
        Ok(())
    }
    fn closed(&self) -> bool {
        self.live.is_none() && self.allocated >= BUFFER_BYTES && self.freed == self.allocated
    }
}

/// Unpublished raw allocation guard. A device clone keeps VkDevice valid.
struct Backing {
    device: wgpu::Device,
    raw: ash::Device,
    buffer: Option<vk::Buffer>,
    memory: Option<vk::DeviceMemory>,
    allocation_bytes: u64,
}
impl Backing {
    fn release_unsubmitted(&mut self) {
        // SAFETY: this guard owns both objects and they have not been imported
        // or submitted. Buffer destruction precedes memory release.
        unsafe {
            if let Some(buffer) = self.buffer.take() {
                self.raw.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.memory.take() {
                self.raw.free_memory(memory, None);
            }
        }
    }
    fn map(&self) -> Result<*mut u8> {
        // SAFETY: this allocation is exclusively owned, HOST_VISIBLE, offset0;
        // callers only map before import or after a completed device join.
        unsafe {
            self.raw.map_memory(
                self.memory.ok_or("public host weight missing backing")?,
                0,
                self.allocation_bytes,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map(|pointer| pointer.cast::<u8>())
        .map_err(|_| "public host weight native map failed")
    }
    fn unmap(&self) {
        // SAFETY: every call follows this guard's successful map.
        unsafe { self.raw.unmap_memory(self.memory.unwrap()) }
    }
    fn cache(&self, coherent: bool, flush: bool) -> Result<()> {
        if coherent {
            return Ok(());
        }
        let range = vk::MappedMemoryRange::default()
            .memory(
                self.memory
                    .ok_or("public host weight missing cache backing")?,
            )
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the entire allocation is mapped. WHOLE_SIZE reaches the
        // allocation end, satisfying nonCoherentAtomSize without tail overrun.
        let result = unsafe {
            if flush {
                self.raw.flush_mapped_memory_ranges(&[range])
            } else {
                self.raw.invalidate_mapped_memory_ranges(&[range])
            }
        };
        result.map_err(|_| "public host weight native cache operation failed")
    }
}
impl Drop for Backing {
    fn drop(&mut self) {
        self.release_unsubmitted();
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Close {
    pub allocation_bytes: u64,
    pub source_unchanged: bool,
    pub full_allocation_zero: bool,
    pub ledger_closed: bool,
}

pub(crate) struct PublicHostWeightPool {
    fixture: Fixture,
    placement: Placement,
    backing: Option<Backing>,
    buffer: Option<wgpu::Buffer>,
    ledger: Ledger,
}
impl PublicHostWeightPool {
    pub(crate) fn new(device: &wgpu::Device, fixture: Fixture) -> Result<Self> {
        // SAFETY: borrow backend identity only; no object is destroyed through
        // the guard. Drop it before calling wgpu's import path.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or("public host weight Vulkan device required")?;
        let raw = hal.raw_device().clone();
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_memory_properties(hal.raw_physical_device())
        };
        let queue_families = hal.kuno_buffer_queue_families();
        let mut buffer_info = vk::BufferCreateInfo::default()
            .size(BUFFER_BYTES)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if let Some((families, count)) = &queue_families {
            buffer_info = buffer_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families[..*count]);
        }
        let vk_buffer = unsafe { raw.create_buffer(&buffer_info, None) }
            .map_err(|_| "public host weight native buffer creation failed")?;
        let mut backing = Backing {
            device: device.clone(),
            raw,
            buffer: Some(vk_buffer),
            memory: None,
            allocation_bytes: 0,
        };
        let requirements = unsafe { backing.raw.get_buffer_memory_requirements(vk_buffer) };
        let plan = plan(requirements, &properties)?;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(plan.allocation_bytes)
            .memory_type_index(plan.placement.memory_type);
        backing.memory = Some(
            unsafe { backing.raw.allocate_memory(&allocation, None) }
                .map_err(|_| "public host weight native host allocation failed")?,
        );
        backing.allocation_bytes = plan.allocation_bytes;
        unsafe {
            backing
                .raw
                .bind_buffer_memory(vk_buffer, backing.memory.unwrap(), 0)
        }
        .map_err(|_| "public host weight native memory binding failed")?;
        let pointer = backing.map()?;
        // SAFETY: successful full map; allocation capped at1MiB, offset0.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(pointer, plan.allocation_bytes as usize) };
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = fixture.byte(index);
        }
        let flushed = backing.cache(plan.placement.coherent(), true);
        backing.unmap();
        flushed?;
        drop(hal);
        let mut ledger = Ledger::default();
        ledger.allocate(plan.allocation_bytes)?;
        // SAFETY: initialized complete buffer, same device, exact descriptor.
        // from_raw owns the VkBuffer, but NEVER owns/counters its memory.
        backing.buffer = None;
        let imported = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                wgpu::hal::vulkan::Buffer::from_raw(vk_buffer),
                &wgpu::BufferDescriptor {
                    label: Some("compiled-public-host-weights"),
                    size: BUFFER_BYTES,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                },
            )
        };
        Ok(Self {
            fixture,
            placement: plan.placement,
            backing: Some(backing),
            buffer: Some(imported),
            ledger,
        })
    }
    /// Only the sibling compiled-public probe borrows this source. No resource
    /// accessor is exported by the application or CubeCL runtime.
    pub(crate) fn source(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().unwrap()
    }
    pub(crate) fn placement(&self) -> Placement {
        self.placement
    }
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.ledger.allocated
    }
    fn join(&self) -> Result<()> {
        let backing = self
            .backing
            .as_ref()
            .ok_or("public host weight closed owner")?;
        backing
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .map_err(|_| "public host weight submitted work join failed")?;
        Ok(())
    }
    pub(crate) fn close(mut self) -> Result<Close> {
        // First prohibit future submissions through all safe wgpu clones. Only
        // already-submitted consumers may remain, and the join precedes CPU map.
        self.buffer.take().unwrap().destroy();
        self.join()?;
        let backing = self.backing.as_ref().unwrap();
        let pointer = backing.map()?;
        let invalidated = backing.cache(self.placement.coherent(), false);
        let length = backing.allocation_bytes as usize;
        let unchanged = invalidated.is_ok()
            && (0..length).all(|i| {
                // SAFETY: full mapped allocation, after the completed join.
                (unsafe { std::ptr::read_volatile(pointer.add(i)) }) == self.fixture.byte(i)
            });
        {
            // SAFETY: exclusive full mapped allocation after join. Keep this
            // mutable reference disjoint in time from the raw-pointer audits.
            let bytes = unsafe { std::slice::from_raw_parts_mut(pointer, length) };
            bytes.zeroize();
        }
        let zero = (0..length).all(|i| {
            // SAFETY: full mapped allocation, retained throughout this audit.
            (unsafe { std::ptr::read_volatile(pointer.add(i)) }) == 0
        });
        let flushed = backing.cache(self.placement.coherent(), true);
        backing.unmap();
        invalidated?;
        flushed?;
        if !unchanged || !zero {
            return Err("public host weight immutable/full zero audit failed");
        }
        // Mark every safe wgpu clone unusable BEFORE the final join/free. Vulkan
        // permits freeing memory still bound to an inactive buffer; all submitted
        // commands must be completed and the buffer must not be used afterwards.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkFreeMemory.html
        self.join()?;
        let mut backing = self.backing.take().unwrap();
        let allocation_bytes = backing.allocation_bytes;
        backing.release_unsubmitted();
        self.ledger.free(allocation_bytes)?;
        Ok(Close {
            allocation_bytes,
            source_unchanged: unchanged,
            full_allocation_zero: zero,
            ledger_closed: self.ledger.closed(),
        })
    }
}
impl Drop for PublicHostWeightPool {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            buffer.destroy();
        }
        if self.backing.is_some() && self.join().is_err() {
            // No successful receipt: preserving memory/device lifetime is safer
            // than freeing backing still potentially consumed by a lost GPU.
            // This explicit failure leak is not reported as physical recovery.
            std::mem::forget(self.backing.take().unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn inventory() -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 4,
            memory_heap_count: 2,
            ..Default::default()
        };
        p.memory_heaps[0] = vk::MemoryHeap {
            size: 1 << 30,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
        p.memory_heaps[1] = vk::MemoryHeap {
            size: 1 << 30,
            flags: vk::MemoryHeapFlags::empty(),
        };
        p.memory_types[0] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        p.memory_types[1] = vk::MemoryType {
            heap_index: 0,
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
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
    fn host_selection_rejects_bar_vram_and_respects_requirements_bits() {
        let p = inventory();
        assert_eq!(
            select_host_type(&p, 15, BUFFER_BYTES).unwrap().memory_type,
            3
        );
        assert_eq!(
            select_host_type(&p, 7, BUFFER_BYTES).unwrap().memory_type,
            2
        );
        assert!(select_host_type(&p, 3, BUFFER_BYTES).is_err());
        assert!(select_host_type(&p, 0, BUFFER_BYTES).is_err());
    }
    #[test]
    fn host_selection_checks_actual_heap_and_malformed_inventory() {
        let mut p = inventory();
        p.memory_types[2].heap_index = 0;
        assert!(select_host_type(&p, 4, BUFFER_BYTES).is_err());
        p.memory_types[2].heap_index = 9;
        assert!(select_host_type(&p, 4, BUFFER_BYTES).is_err());
        p.memory_type_count = 33;
        assert!(select_host_type(&p, 15, BUFFER_BYTES).is_err());
        assert!(select_host_type(&inventory(), 15, HOST_CAP + 1).is_err());
    }
    #[test]
    fn physical_plan_checks_rounded_capacity_and_alignment() {
        let p = inventory();
        let mut r = vk::MemoryRequirements {
            size: BUFFER_BYTES,
            alignment: 4096,
            memory_type_bits: 15,
        };
        assert_eq!(plan(r, &p).unwrap().allocation_bytes, BUFFER_BYTES);
        for size in [0, BUFFER_BYTES - 1, HOST_CAP + 4096] {
            r.size = size;
            assert!(plan(r, &p).is_err());
        }
        r.size = BUFFER_BYTES;
        for alignment in [0, 3, BUFFER_BYTES * 2] {
            r.alignment = alignment;
            assert!(plan(r, &p).is_err());
        }
    }
    #[test]
    fn fixed_q2_original_four_codes_and_unaligned_source_extent() {
        let mut values = [0.; ROWS * COLUMNS];
        kuno_text::TensorType::from_gguf_id(42)
            .unwrap()
            .decode(&Fixture::Aligned.packed(), &mut values)
            .unwrap();
        for row in 0..ROWS {
            let scale = [0.5, 0.25, 1., 0.125][row];
            let mut seen = [false; 4];
            for column in 0..COLUMNS {
                let byte = Fixture::Aligned.byte(row * BLOCK_BYTES + 2 + column / 4);
                let code = (byte >> (2 * (column % 4))) & 3;
                seen[code as usize] = true;
                assert_eq!(
                    values[row * COLUMNS + column].to_bits(),
                    (scale * (i32::from(code) - 1) as f32).to_bits()
                );
            }
            assert_eq!(seen, [true; 4]);
        }
        for i in 0..BUFFER_BYTES as usize {
            let expected = if i < 3 {
                0
            } else {
                Fixture::Aligned.byte(i - 3)
            };
            assert_eq!(Fixture::Prefix3.byte(i), expected);
        }
        assert_eq!(Fixture::Prefix3.byte(BUFFER_BYTES as usize), 0);
        assert_eq!(Fixture::BadPublicTail.byte(BUFFER_BYTES as usize - 1), 0x5a);
        assert_eq!(Fixture::BadPublicTail.byte(BUFFER_BYTES as usize), 0);
    }
    #[test]
    fn ledger_is_nonmutating_on_failed_allocate_or_double_release() {
        let mut l = Ledger::default();
        assert!(l.allocate(HOST_CAP + 1).is_err());
        assert_eq!(l.live, None);
        l.allocate(BUFFER_BYTES).unwrap();
        assert!(l.allocate(BUFFER_BYTES).is_err());
        assert!(l.free(BUFFER_BYTES + 1).is_err());
        assert_eq!(l.live, Some(BUFFER_BYTES));
        l.free(BUFFER_BYTES).unwrap();
        assert!(l.closed());
        assert!(l.free(BUFFER_BYTES).is_err());
        assert!(l.allocate(BUFFER_BYTES).is_err());
        assert!(l.closed());
    }
}
