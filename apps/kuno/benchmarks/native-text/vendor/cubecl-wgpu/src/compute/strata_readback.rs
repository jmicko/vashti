//! Original native readback owner: real copy/fence first, then host mapping.
//! No Wgpu primary-queue copy, implicit initialization or map-poll submission.
use cubecl_runtime::transient::PrivateHostAllocation;
use std::{cell::Cell, time::Instant};
use wgpu::hal::Device as _;

pub(super) struct Plan {
    pub copy: u64,
    pub owned: u64,
    logical: usize,
}
impl Plan {
    /// The caller has validated original source offsets/backing using the same
    /// copy_bounds as ordinary private reads. Never narrow the original copy.
    pub fn new(copy: u64, owned: u64, logical: usize) -> Result<Self, &'static str> {
        if copy == 0
            || !copy.is_multiple_of(4)
            || owned < 256
            || !owned.is_multiple_of(256)
            || copy > owned
            || owned > 128 * 1024 * 1024
            || logical as u64 > copy
        {
            return Err("native readback original owned span");
        }
        Ok(Self {
            copy,
            owned,
            logical,
        })
    }
    fn copy_host(
        &self,
        bytes: &[u8],
        deadline: Instant,
        mut now: impl FnMut() -> Instant,
    ) -> Result<PrivateHostAllocation, &'static str> {
        if bytes.len() as u64 != self.owned {
            return Err("native readback mapping full span");
        }
        if now() >= deadline {
            return Err("native readback original map deadline");
        }
        // The logical output uses the same volatile-wiping host controller as
        // ordinary private readbacks; padding never escapes to the consumer.
        let result = PrivateHostAllocation::copied(&bytes[..self.logical])
            .map_err(|_| "native readback host allocation")?;
        if now() >= deadline {
            return Err("native readback original map deadline");
        }
        Ok(result)
    }
}

/// Local synchronous map only. The original Wgpu buffer and HAL guards remain
/// borrowed through unmap. No raw pointer or guard crosses an async boundary.
struct Mapping<'a> {
    device: &'a wgpu::hal::vulkan::Device,
    buffer: &'a wgpu::hal::vulkan::Buffer,
    failed: &'a Cell<bool>,
}
impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            self.device.unmap_buffer(self.buffer);
        }))
        .is_err()
        {
            self.failed.set(true);
        }
    }
}

/// # Safety
/// The caller owns an exclusive original private readback lease, has queued a
/// full-owner native clear followed by this exact original copy, and has joined
/// its ACTUAL final native fence (including every transfer segment). That graph
/// emits TRANSFER_WRITE -> HOST_READ availability before signaling completion.
/// No GPU write, Wgpu map, reuse or erasure may touch the owner during this call.
pub(super) unsafe fn read_joined(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    plan: &Plan,
    deadline: Instant,
) -> Result<PrivateHostAllocation, &'static str> {
    if buffer.size() != plan.owned
        || !buffer
            .usage()
            .contains(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)
        || Instant::now() >= deadline
    {
        return Err("native readback joined owner or deadline");
    }
    let device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("native readback original Vulkan device")?;
    let buffer = unsafe { buffer.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("native readback original Vulkan buffer")?;
    // Wgpu's owned MAP_READ allocation is host-visible. Vulkan HAL uses its
    // original allocator mapping; no primary work or untracked allocation.
    let mapped = unsafe { device.map_buffer(&buffer, 0..plan.owned) }
        .map_err(|_| "native readback host map")?;
    let failed = Cell::new(false);
    let mapping = Mapping {
        device: &device,
        buffer: &buffer,
        failed: &failed,
    };
    if !mapped.is_coherent {
        // Invalidate the COMPLETE actual mapped owner after its real GPU fence,
        // using the original HAL's non-coherent atom alignment implementation.
        unsafe {
            device.invalidate_mapped_ranges(&buffer, std::iter::once(0..plan.owned));
        }
    }
    // SAFETY: the original full owner was initialized by native fill/copy and
    // joined, mapped successfully and stays exclusively leased through unmap.
    let bytes = unsafe { std::slice::from_raw_parts(mapped.ptr.as_ptr(), plan.owned as usize) };
    let result = plan.copy_host(bytes, deadline, Instant::now);
    drop(mapping);
    if failed.get() {
        return Err("native readback host unmap");
    }
    if Instant::now() >= deadline {
        return Err("native readback original map deadline");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_read_plan_preserves_full_copy_and_rejects_unowned_mapping_spans() {
        for copy in [4u64, 128, 256, 260, 512, 4096] {
            let owned = copy.next_multiple_of(256);
            for logical in [0, 1, copy as usize] {
                let p = Plan::new(copy, owned, logical).unwrap();
                assert_eq!(p.copy, copy);
                assert_eq!(p.owned, owned);
                assert!(Plan::new(copy, owned, copy as usize + 1).is_err());
            }
        }
        for (copy, owned, logical) in [
            (0, 256, 0),
            (3, 256, 1),
            (260, 256, 1),
            (4, 255, 4),
            (4, 257, 4),
            (4, 256, 257),
            (4, u64::MAX, 4),
            (4, 128 * 1024 * 1024 + 256, 4),
        ] {
            assert!(Plan::new(copy, owned, logical).is_err());
        }
    }
    #[test]
    fn native_host_copy_uses_only_original_logical_bytes_and_one_deadline() {
        let source: Vec<u8> = (0..512).map(|i| (i * 73) as u8).collect();
        let begin = Instant::now();
        let deadline = begin + std::time::Duration::from_secs(1);
        for logical in [0, 1, 129, 260] {
            let plan = Plan::new(260, 512, logical).unwrap();
            let copy = plan.copy_host(&source, deadline, || begin).unwrap();
            assert_eq!(copy.as_slice(), &source[..logical]);
            assert!(plan.copy_host(&source[..511], deadline, || begin).is_err());
            assert!(plan.copy_host(&source, deadline, || deadline).is_err());
            let mut times = [begin, deadline].into_iter();
            assert!(
                plan.copy_host(&source, deadline, || times.next().unwrap())
                    .is_err()
            );
        }
    }
}
