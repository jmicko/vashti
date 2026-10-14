//! Original-model-only HOST fallback layout for complete lent GPU cache roots.
//! No logical routing, prompts or expert-usage history enter this plan.
use crate::strata_global_plan::{EXPERTS, LAYERS, Plan as Global, RankedPair};
use kuno_text::{Error, Result};
const PAGE: u64 = cubecl_runtime::public_external::MAX_BUFFER_BYTES;
const MAX_PAGES: usize = 30;
const HEAP_ALIGN: u64 = 1024 * 1024;
pub(crate) const HOST_METADATA: u64 = 4 * 1024 * 1024;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Page {
    pub root: usize,
    pub shard: usize,
    pub logical: u64,
    pub backing_bound: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell {
    pub pair: RankedPair,
    pub sources: [(usize, u64, u64); 3],
}
pub(crate) struct Plan {
    instance: u64,
    indices: [usize; 3],
    pages: Vec<Page>,
    cells: Vec<Cell>,
    backing: u64,
}
fn reject() -> Error {
    Error::InvalidFormat("prompt original-weight shadow layout")
}
impl Plan {
    pub(crate) fn new(
        global: &Global,
        indices: [usize; 3],
        shards: [[usize; 3]; LAYERS],
    ) -> Result<Self> {
        if shards.iter().flatten().any(|&s| s >= 2) {
            return Err(reject());
        }
        if indices
            .iter()
            .enumerate()
            .any(|(i, n)| indices[..i].contains(n))
        {
            return Err(reject());
        }
        let mut count = 0usize;
        for &i in &indices {
            let c = global.chunks.get(i).ok_or_else(reject)?;
            if c.logical_bytes == 0
                || c.physical_bytes > crate::strata_global_plan::MAX_GPU_CHUNK_BYTES
                || c.logical_bytes > c.physical_bytes
            {
                return Err(reject());
            }
            count = count
                .checked_add(c.slots)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        if count == 0 || count > LAYERS * EXPERTS {
            return Err(reject());
        }
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(count)
            .map_err(|_| Error::LimitExceeded("shadow metadata"))?;
        let mut pages = Vec::<Page>::with_capacity(MAX_PAGES);
        let mut seen = vec![false; LAYERS * EXPERTS];
        for &root in &indices {
            let chunk = &global.chunks[root];
            let end = chunk
                .first_slot
                .checked_add(chunk.slots)
                .ok_or(Error::ArithmeticOverflow)?;
            let mut cursor = 0;
            for n in chunk.first_slot..end {
                let slot = global.slots.get(n).ok_or_else(reject)?;
                let address = global.cells.get(n).ok_or_else(reject)?;
                if slot.pair.layer >= LAYERS
                    || slot.pair.expert >= EXPERTS
                    || address.chunk != root
                    || address.offset != cursor
                    || slot.aligned_bytes > PAGE
                    || !slot.aligned_bytes.is_multiple_of(256)
                    || seen[slot.pair.layer * EXPERTS + slot.pair.expert]
                {
                    return Err(reject());
                }
                seen[slot.pair.layer * EXPERTS + slot.pair.expert] = true;
                let roles = global.layers[slot.pair.layer].role_expert_bytes;
                let role_bytes = roles
                    .iter()
                    .try_fold(0u64, |a, b| a.checked_add(*b))
                    .ok_or(Error::ArithmeticOverflow)?;
                if role_bytes != slot.bytes
                    || roles.iter().any(|n| *n == 0 || !n.is_multiple_of(4))
                    || slot.aligned_bytes < role_bytes
                {
                    return Err(reject());
                }
                // A registered original-weight page belongs to exactly one
                // authenticated shard even if a cache root mixes layers/roles.
                let mut sources = [(0, 0, 0); 3];
                for (role, &bytes) in roles.iter().enumerate() {
                    let shard = shards[slot.pair.layer][role];
                    let page = match pages.iter().rposition(|p| {
                        p.root == root
                            && p.shard == shard
                            && p.logical.checked_add(bytes).is_some_and(|n| n <= PAGE)
                    }) {
                        Some(i) => i,
                        None => {
                            if pages.len() == MAX_PAGES {
                                return Err(Error::LimitExceeded("shadow page inventory"));
                            }
                            pages.push(Page {
                                root,
                                shard,
                                logical: 0,
                                backing_bound: 0,
                            });
                            pages.len() - 1
                        }
                    };
                    let offset = pages[page].logical;
                    sources[role] = (page, offset, bytes);
                    pages[page].logical = (offset + bytes + 255) & !255;
                }
                cells.push(Cell {
                    pair: slot.pair,
                    sources,
                });
                cursor = cursor
                    .checked_add(slot.aligned_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            if cursor != chunk.logical_bytes {
                return Err(reject());
            }
        }
        let mut backing = 0u64;
        for p in &mut pages {
            p.backing_bound = p
                .logical
                .checked_add(HEAP_ALIGN - 1)
                .ok_or(Error::ArithmeticOverflow)?
                & !(HEAP_ALIGN - 1);
            backing = backing
                .checked_add(p.backing_bound)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        let metadata = cells
            .capacity()
            .checked_mul(std::mem::size_of::<Cell>())
            .and_then(|n| n.checked_add(pages.capacity() * std::mem::size_of::<Page>()))
            .and_then(|n| n.checked_add(seen.capacity() + std::mem::size_of::<Self>()))
            .ok_or(Error::ArithmeticOverflow)?;
        if metadata as u64 > HOST_METADATA {
            return Err(Error::LimitExceeded("shadow actual metadata capacity"));
        }
        cells.sort_unstable_by_key(|c| (c.pair.layer, c.pair.expert));
        Ok(Self {
            instance: global.instance(),
            indices,
            pages,
            cells,
            backing,
        })
    }
    pub(crate) fn check(&self, global: &Global) -> Result<()> {
        if self.instance != global.instance() {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn pages(&self) -> &[Page] {
        &self.pages
    }
    pub(crate) fn cells(&self) -> &[Cell] {
        &self.cells
    }
    pub(crate) fn indices(&self) -> [usize; 3] {
        self.indices
    }
    pub(crate) fn backing_bytes(&self) -> u64 {
        self.backing
    }
    pub(crate) fn reservation(&self) -> Result<u64> {
        self.backing
            .checked_add(HOST_METADATA)
            .ok_or(Error::ArithmeticOverflow)
    }
    pub(crate) fn role(
        &self,
        layer: usize,
        cell: usize,
        role: usize,
    ) -> Result<Option<(usize, u64, u64)>> {
        if layer >= LAYERS || cell >= EXPERTS || role >= 3 {
            return Err(reject());
        }
        let Ok(i) = self
            .cells
            .binary_search_by_key(&(layer, cell), |c| (c.pair.layer, c.pair.expert))
        else {
            return Ok(None);
        };
        let (page, offset, bytes) = self.cells[i].sources[role];
        if offset
            .checked_add(bytes)
            .is_none_or(|n| n > self.pages[page].logical)
        {
            return Err(reject());
        }
        Ok(Some((page, offset, bytes)))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_lent_role_has_exact_public_fallback_and_no_unlent_cell_is_invented() {
        let global = Global::shadow_test_fixture(24 * 1024 * 1024 * 1024).unwrap();
        let indices = [
            global.chunks.len() - 4,
            global.chunks.len() - 3,
            global.chunks.len() - 2,
        ];
        // Deliberately split roles/layers across both shards; page identity
        // must never claim only the first source for mixed original bytes.
        let shards = std::array::from_fn(|layer| std::array::from_fn(|role| (layer + role) % 2));
        let plan = Plan::new(&global, indices, shards).unwrap();
        assert!(plan.pages.len() <= MAX_PAGES);
        assert_eq!(
            plan.cells.len(),
            indices
                .iter()
                .map(|i| global.chunks[*i].slots)
                .sum::<usize>()
        );
        let mut ranges = vec![Vec::new(); plan.pages.len()];
        for (layer, l) in global.layers.iter().enumerate() {
            for cell in 0..EXPERTS {
                for role in 0..3 {
                    let source = global.source(layer, cell, role).unwrap();
                    let lent = matches!(source,crate::strata_global_plan::Source::Resident{chunk,..} if indices.contains(&chunk));
                    let fallback = plan.role(layer, cell, role).unwrap();
                    assert_eq!(fallback.is_some(), lent);
                    if let Some((page, first, bytes)) = fallback {
                        assert_eq!(bytes, l.role_expert_bytes[role]);
                        assert_eq!(plan.pages[page].shard, shards[layer][role]);
                        assert!(first + bytes <= plan.pages[page].logical);
                        ranges[page].push((first, first + bytes));
                    }
                }
            }
        }
        for (i, rs) in ranges.iter_mut().enumerate() {
            rs.sort_unstable();
            assert!(rs.windows(2).all(|w| w[0].1 <= w[1].0));
            assert!(plan.pages[i].logical <= PAGE);
            assert!(plan.pages[i].backing_bound >= plan.pages[i].logical);
        }
        assert_eq!(
            plan.backing_bytes(),
            plan.pages.iter().map(|p| p.backing_bound).sum::<u64>()
        );
        assert_eq!(
            plan.reservation().unwrap(),
            plan.backing_bytes() + HOST_METADATA
        );
        assert!(Plan::new(&global, [0, 0, 1], shards).is_err());
        assert!(Plan::new(&global, [0, 1, global.chunks.len()], shards).is_err());
        assert!(
            plan.check(&Global::shadow_test_fixture(24 * 1024 * 1024 * 1024).unwrap())
                .is_err()
        );
    }
}
