//! Owning original-weight exchange for variable global cells. Request-derived
//! offsets never enter a log or a public pointer table. Whole joined transaction.
use super::*;
#[path = "strata_adapt_transfer.rs"]
mod flight;
pub(crate) use flight::Flight;

pub(crate) struct Pair {
    pub gpu: WgpuResource,
    pub host: WgpuResource,
}
pub(super) const HOST_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
pub(super) fn host_vk_usage() -> vk::BufferUsageFlags {
    vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    Gpu,
    Host,
    Scratch,
}
#[derive(Clone, Copy, Debug)]
struct Copy {
    pair: usize,
    from: Endpoint,
    to: Endpoint,
    bytes: u64,
}
fn copy_plan(sizes: impl ExactSizeIterator<Item = u64>, capacity: u64) -> KResult<Vec<Copy>> {
    let count = sizes.len();
    if count == 0 || !count.is_multiple_of(3) || count > crate::strata_global_adapt::MAX_SWAPS * 3 {
        return Err(fail());
    }
    let mut plan = Vec::new();
    plan.try_reserve_exact(count * 3).map_err(|_| fail())?;
    for (pair, bytes) in sizes.enumerate() {
        if bytes == 0 || bytes > 1024 * 1024 || bytes > capacity || !bytes.is_multiple_of(4) {
            return Err(fail());
        }
        for (from, to) in [
            (Endpoint::Host, Endpoint::Scratch),
            (Endpoint::Gpu, Endpoint::Host),
            (Endpoint::Scratch, Endpoint::Gpu),
        ] {
            plan.push(Copy {
                pair,
                from,
                to,
                bytes,
            });
        }
    }
    Ok(plan)
}
const _: () = assert!(
    crate::strata_global_adapt::MAX_SWAPS * 3 * 3 * std::mem::size_of::<Copy>() < 128 * 1024
);
impl Host {
    /// # Safety
    /// Same actual owning device/queue. All source consumers have completed,
    /// no CPU map or new consumer overlaps, and Sources authenticated every GPU
    /// cell. All three roles of each proposed pair must be published together.
    pub(crate) unsafe fn submit_exchange(
        &mut self,
        pairs: &[Pair],
        scratch: &WgpuResource,
        guard: crate::strata_external_flight::Flight,
    ) -> KResult<Flight> {
        if self.closed
            || pairs.is_empty()
            || pairs.len() > crate::strata_global_adapt::MAX_SWAPS * 3
        {
            return Err(fail());
        }
        // Original attached dormant prompt raw owner, not a new allocation or
        // a source-provided replacement. Full client joins precede this call.
        let copy_usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        if scratch.offset != 0
            || scratch.size != scratch.buffer.size()
            || scratch.size < crate::strata_prompt::allocation::RAW_RING_BYTES
            || !scratch.buffer.usage().contains(copy_usage)
        {
            return Err(fail());
        }
        let copies = copy_plan(pairs.iter().map(|p| p.gpu.size), scratch.size)?;
        let align = u64::from(self.device.limits().min_storage_buffer_offset_alignment);
        for (i, pair) in pairs.iter().enumerate() {
            if pair.gpu.size != pair.host.size
                || pair.gpu.size == 0
                || pair.gpu.size > 1024 * 1024
                || !pair.gpu.size.is_multiple_of(4)
                || pair.gpu.buffer == pair.host.buffer
                || pair.gpu.buffer == scratch.buffer
                || pair.host.buffer == scratch.buffer
                || !pair.gpu.buffer.usage().contains(copy_usage)
                || !pair.host.buffer.usage().contains(copy_usage)
                || !pair.gpu.offset.is_multiple_of(align)
                || !pair.host.offset.is_multiple_of(align)
                || pair
                    .gpu
                    .offset
                    .checked_add(pair.gpu.size)
                    .is_none_or(|n| n > pair.gpu.buffer.size())
                || pair
                    .host
                    .offset
                    .checked_add(pair.host.size)
                    .is_none_or(|n| n > pair.host.buffer.size())
            {
                return Err(fail());
            }
            let owner = self
                .pages
                .iter()
                .find(|p| p.owner.buffer.as_ref() == Some(&pair.host.buffer))
                .ok_or_else(fail)?;
            if owner.owner.phase != Phase::Registered
                || owner.authority.is_none()
                || pair.host.offset + pair.host.size > owner.owner.logical
            {
                return Err(fail());
            }
            for earlier in &pairs[..i] {
                for a in [&earlier.gpu, &earlier.host] {
                    for b in [&pair.gpu, &pair.host] {
                        if a.buffer == b.buffer
                            && a.offset < b.offset + b.size
                            && b.offset < a.offset + a.size
                        {
                            return Err(fail());
                        }
                    }
                }
            }
        }
        let mut bindings = Vec::new();
        for pair in pairs {
            let owner = self
                .pages
                .iter()
                .find(|p| p.owner.buffer.as_ref() == Some(&pair.host.buffer))
                .ok_or_else(fail)?;
            let binding = owner
                .authority
                .as_ref()
                .ok_or_else(fail)?
                .handle()
                .binding();
            if !bindings.iter().any(|b: &cubecl_runtime::server::Binding| {
                b.memory.allocation_id() == binding.memory.allocation_id()
            }) {
                bindings.push(binding);
            }
        }
        // Swap only original model weights while registry readers are revoked.
        let token =
            unsafe { self.client.begin_public_external_rebase(bindings) }.map_err(|_| fail())?;
        Flight::submit(
            &self.device,
            &self.client,
            pairs,
            scratch,
            &copies,
            token,
            guard,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_transfer_plan_preserves_both_original_bit_patterns_across_all_three_roles_and_reused_scratch()
     {
        for count in [3, 6, 9, crate::strata_global_adapt::MAX_SWAPS * 3] {
            let sizes = (0..count)
                .map(|i| ((i % 17 + 1) * 4) as u64)
                .collect::<Vec<_>>();
            let mut gpu = sizes
                .iter()
                .enumerate()
                .map(|(p, &n)| {
                    (0..n)
                        .map(|i| (p as u8).wrapping_mul(37).wrapping_add(i as u8))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut host = sizes
                .iter()
                .enumerate()
                .map(|(p, &n)| {
                    (0..n)
                        .map(|i| (p as u8).wrapping_mul(13) ^ 0xa5 ^ (i as u8).rotate_left(3))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let expected_gpu = host.clone();
            let expected_host = gpu.clone();
            let mut scratch = vec![0xff; *sizes.iter().max().unwrap() as usize];
            let plan = copy_plan(
                sizes.iter().copied(),
                crate::strata_prompt::allocation::RAW_RING_BYTES,
            )
            .unwrap();
            assert_eq!(plan.len(), count * 3);
            for (i, copy) in plan.iter().enumerate() {
                assert_eq!(copy.pair, i / 3);
                assert_eq!(copy.bytes, sizes[copy.pair]);
                let n = copy.bytes as usize;
                let original = match copy.from {
                    Endpoint::Gpu => &gpu[copy.pair][..n],
                    Endpoint::Host => &host[copy.pair][..n],
                    Endpoint::Scratch => &scratch[..n],
                }
                .to_vec();
                let target = match copy.to {
                    Endpoint::Gpu => &mut gpu[copy.pair][..n],
                    Endpoint::Host => &mut host[copy.pair][..n],
                    Endpoint::Scratch => &mut scratch[..n],
                };
                target.copy_from_slice(&original);
            }
            assert_eq!(gpu, expected_gpu);
            assert_eq!(host, expected_host);
        }
    }
    #[test]
    fn original_copy_plan_rejects_partial_role_sets_bad_spans_and_scratch_overrun_before_submission()
     {
        for n in [0, 1, 2, crate::strata_global_adapt::MAX_SWAPS * 3 + 3] {
            assert!(copy_plan(vec![4; n].into_iter(), 32 * 1024 * 1024).is_err());
        }
        for size in [0, 1, 2, 3, 5, 1024 * 1024 + 4] {
            assert!(copy_plan([size; 3].into_iter(), 32 * 1024 * 1024).is_err());
        }
        assert!(copy_plan([8; 3].into_iter(), 4).is_err());
        assert_eq!(
            copy_plan([1024 * 1024; 3].into_iter(), 32 * 1024 * 1024)
                .unwrap()
                .len(),
            9
        );
    }
    #[test]
    fn actual_host_import_flags_and_hal_copy_to_host_dependency_match_the_transfer_producer() {
        assert_eq!(
            HOST_USAGE,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        );
        assert_eq!(
            host_vk_usage(),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        );
        let (stage, access) =
            wgpu::hal::vulkan::CommandEncoder::kuno_public_host_transfer_read_dependency();
        assert_eq!(stage, vk::PipelineStageFlags::TRANSFER);
        assert_eq!(access, vk::AccessFlags::TRANSFER_WRITE);
        assert!(!stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
    }
}
