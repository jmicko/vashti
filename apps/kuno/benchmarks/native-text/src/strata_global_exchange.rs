//! Owning original-weight exchange for variable global cells. Request-derived
//! offsets never enter a log or a public pointer table. Whole joined transaction.
use super::*;
#[path = "strata_adapt_transfer.rs"]
mod flight;
pub(crate) use flight::Flight;
#[path = "strata_adapt_inventory.rs"]
mod inventory;
pub(crate) use inventory::{Cell, Inventory};
pub(crate) const INVENTORY_HOST_BYTES: usize = inventory::HOST_BYTES;

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
    /// Revoke all original expert HOST pages before a worker can choose its
    /// bounded swap set. Embedding/shadow pages remain separate authorities.
    /// Registry mutation and its token NEVER leave the sole original thread.
    pub(crate) fn begin_adaptive_rebase(
        &mut self,
    ) -> KResult<Option<cubecl_runtime::public_external::PublicExternalRebase>> {
        if self.closed || self.cpu_borrow.load(std::sync::atomic::Ordering::Acquire) {
            return Err(fail());
        }
        let mut bindings = Vec::new();
        for page in self
            .pages
            .iter()
            .filter(|p| p.layer < crate::strata_global_plan::LAYERS)
        {
            if page.owner.phase != Phase::Registered {
                return Err(fail());
            }
            bindings.push(page.authority.as_ref().ok_or_else(fail)?.handle().binding());
        }
        if bindings.is_empty() {
            return Ok(None);
        } // fully resident: no HOST expert source exists
        // SAFETY: sole original source owner has joined all preceding readers;
        // all selected pages are genuine registered expert sources on this client.
        unsafe { self.client.begin_public_external_rebase(bindings) }
            .map(Some)
            .map_err(|_| fail())
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
