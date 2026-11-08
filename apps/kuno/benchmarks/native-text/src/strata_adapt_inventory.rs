//! Original physical expert cells, authenticated on the sole client thread.
//! Worker receives only owned same-device resources and immutable PUBLIC ranges.
use super::*;
use crate::strata_global_plan::{EXPERTS, LAYERS};
pub(crate) const HOST_BYTES: usize = 8 * 1024 * 1024;
pub(crate) struct Cell {
    pub(crate) resource: WgpuResource,
    pub(crate) gpu: bool,
}
pub(crate) struct Inventory {
    cells: Vec<Cell>,
}
impl Inventory {
    pub(crate) fn new(cells: Vec<Cell>, alignment: u64) -> KResult<Self> {
        if !alignment.is_power_of_two()
            || cells.len() != LAYERS * EXPERTS * 3
            || cells.capacity() * std::mem::size_of::<Cell>() > HOST_BYTES
        {
            return Err(fail());
        }
        for c in &cells {
            let r = &c.resource;
            if !r.offset.is_multiple_of(alignment)
                || r.size == 0
                || r.size > 1024 * 1024
                || !r.size.is_multiple_of(4)
                || r.offset
                    .checked_add(r.size)
                    .is_none_or(|n| n > r.buffer.size())
                || !r
                    .buffer
                    .usage()
                    .contains(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST)
            {
                return Err(fail());
            }
        }
        Ok(Self { cells })
    }
    pub(super) fn pairs(
        &self,
        swaps: &[crate::strata_global_adapt::Swap],
        scratch: &WgpuResource,
    ) -> KResult<Vec<Pair>> {
        if swaps.len() > crate::strata_global_adapt::MAX_SWAPS {
            return Err(fail());
        }
        let mut pairs: Vec<Pair> = Vec::with_capacity(swaps.len() * 3);
        for s in swaps {
            if s.layer >= LAYERS || s.gpu_cell >= EXPERTS || s.host_cell >= EXPERTS {
                return Err(fail());
            }
            for role in 0..3 {
                let g = &self.cells[(s.layer * EXPERTS + s.gpu_cell) * 3 + role];
                let h = &self.cells[(s.layer * EXPERTS + s.host_cell) * 3 + role];
                if !g.gpu
                    || h.gpu
                    || g.resource.size != h.resource.size
                    || g.resource.buffer == h.resource.buffer
                    || g.resource.buffer == scratch.buffer
                    || h.resource.buffer == scratch.buffer
                {
                    return Err(fail());
                }
                for earlier in &pairs {
                    for a in [&earlier.gpu, &earlier.host] {
                        for b in [&g.resource, &h.resource] {
                            if a.buffer == b.buffer
                                && a.offset < b.offset + b.size
                                && b.offset < a.offset + a.size
                            {
                                return Err(fail());
                            }
                        }
                    }
                }
                pairs.push(Pair {
                    gpu: g.resource.clone(),
                    host: h.resource.clone(),
                });
            }
        }
        Ok(pairs)
    }
}
