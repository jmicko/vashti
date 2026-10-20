//! Exact original-view COPY-queue hazards. Disjoint writes do not require a
//! dependency. Genuine overlaps retain the original whole-buffer barrier and
//! TRANSFER stage/access semantics. No cross-queue plan or owner is changed.
use super::strata_schedule::{Access, Segment};
use std::{collections::HashMap, mem::size_of, ops::Range};

const READ: u32 = 1;
const WRITE: u32 = 2;
const MAX_NODES: usize = 32768;
const MAX_ROOTS: usize = 65536;
const MAX_REGIONS: usize = 262144;

/// COPY epochs already synchronized by genuine incoming peer timeline waits.
/// This derives ONLY from the completed original Plan, before any submission.
/// It grants no submission, completion, residency or allocator authority.
pub(super) struct Visibility {
    acquired_copy: Vec<u64>,
}
impl Visibility {
    pub(super) fn new(segments: &[Segment], budget: usize) -> Result<Self, &'static str> {
        if segments.is_empty() || segments.len() > MAX_NODES + 1 {
            return Err("native transfer visibility segment capacity");
        }
        let wanted = segments
            .len()
            .checked_mul(size_of::<[u64; 3]>() + size_of::<u64>())
            .and_then(|n| n.checked_add(4096))
            .ok_or("native transfer visibility metadata overflow")?;
        if wanted > budget {
            return Err("native transfer visibility metadata capacity");
        }
        let mut ancestors: Vec<[u64; 3]> = Vec::new();
        let mut acquired_copy = Vec::new();
        ancestors
            .try_reserve_exact(segments.len())
            .map_err(|_| "native transfer visibility ancestor allocation")?;
        acquired_copy
            .try_reserve_exact(segments.len())
            .map_err(|_| "native transfer visibility allocation")?;
        let actual = ancestors
            .capacity()
            .checked_mul(size_of::<[u64; 3]>())
            .and_then(|n| n.checked_add(acquired_copy.capacity().checked_mul(size_of::<u64>())?))
            .and_then(|n| n.checked_add(4096))
            .ok_or("native transfer visibility metadata overflow")?;
        if actual > budget {
            return Err("native transfer visibility metadata capacity");
        }
        let mut previous = [[0u64; 3]; 3];
        let mut copy_visible = 0;
        for (index, segment) in segments.iter().enumerate() {
            let own = segment.queue();
            if segment.copy && segment.shared
                || segment.signal != index as u64 + 1
                || segment.waits[own] != 0
                || segment.wait != *segment.waits.iter().max().unwrap()
                || index == 0 && (own != 0 || segment.waits != [0; 3])
            {
                return Err("native transfer visibility original segment identity");
            }
            let mut prefix = previous[own];
            let mut acquired = 0;
            for peer in 0..3 {
                let wait = segment.waits[peer];
                if wait == 0 {
                    continue;
                }
                let before = usize::try_from(wait - 1)
                    .map_err(|_| "native transfer visibility wait index")?;
                if before >= index || segments[before].queue() != peer {
                    return Err("native transfer visibility original peer wait");
                }
                for lane in 0..3 {
                    prefix[lane] = prefix[lane].max(ancestors[before][lane]);
                }
                // Every actual submit wait uses ALL_COMMANDS (original
                // strata_submit::Batch). Only a real incoming peer wait
                // acquires the preceding COPY writes; own-queue order alone
                // is deliberately NOT counted as a memory dependency.
                acquired = acquired.max(ancestors[before][1]);
            }
            if own == 1 {
                copy_visible = copy_visible.max(acquired);
            }
            acquired_copy.push(if own == 1 { copy_visible } else { 0 });
            prefix[own] = segment.signal;
            previous[own] = prefix;
            ancestors.push(prefix);
        }
        Ok(Self { acquired_copy })
    }
    pub(super) fn before(&self, segment: usize) -> Result<u64, &'static str> {
        self.acquired_copy
            .get(segment)
            .copied()
            .ok_or("native transfer visibility segment index")
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, &'static str> {
        self.acquired_copy
            .capacity()
            .checked_mul(size_of::<u64>())
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("native transfer visibility metadata overflow")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Region {
    first: u64,
    end: u64,
    flags: u32,
    epoch: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Dependency {
    pub buffer: u64,
    pub before: u32,
    pub after: u32,
}
impl Dependency {
    pub(super) fn before_flags(&self) -> ash::vk::AccessFlags {
        flags(self.before)
    }
    pub(super) fn after_flags(&self) -> ash::vk::AccessFlags {
        flags(self.after)
    }
}
fn flags(value: u32) -> ash::vk::AccessFlags {
    let mut result = ash::vk::AccessFlags::empty();
    if value & READ != 0 {
        result |= ash::vk::AccessFlags::TRANSFER_READ;
    }
    if value & WRITE != 0 {
        result |= ash::vk::AccessFlags::TRANSFER_WRITE;
    }
    result
}
pub(super) struct Plan {
    nodes: Vec<Range<usize>>,
    dependencies: Vec<Dependency>,
}
impl Plan {
    pub(super) fn dependency_count(&self) -> usize {
        self.dependencies.len()
    }
    pub(super) fn node(&self, index: usize) -> Result<&[Dependency], &'static str> {
        let range = self
            .nodes
            .get(index)
            .ok_or("native transfer barrier node index")?;
        self.dependencies
            .get(range.clone())
            .ok_or("native transfer barrier dependency index")
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, &'static str> {
        self.nodes
            .capacity()
            .checked_mul(size_of::<Range<usize>>())
            .and_then(|n| {
                n.checked_add(
                    self.dependencies
                        .capacity()
                        .checked_mul(size_of::<Dependency>())?,
                )
            })
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("native transfer barrier metadata overflow")
    }
}
pub(super) struct Recorder {
    plan: Plan,
    roots: HashMap<u64, Vec<Region>>,
    region_capacity: usize,
    regions: usize,
    expected: usize,
    copy_capacity: usize,
    copied: usize,
    budget: usize,
}
impl Recorder {
    pub(super) fn new(nodes: usize, copies: usize, budget: usize) -> Result<Self, &'static str> {
        if nodes == 0 || nodes > MAX_NODES || copies > nodes {
            return Err("native transfer barrier node capacity");
        }
        let mut result = Self {
            plan: Plan {
                nodes: Vec::new(),
                dependencies: Vec::new(),
            },
            roots: HashMap::new(),
            region_capacity: 0,
            regions: 0,
            expected: nodes,
            copy_capacity: copies,
            copied: 0,
            budget,
        };
        let wanted = nodes
            .checked_mul(size_of::<Range<usize>>())
            .and_then(|n| {
                n.checked_add(
                    copies
                        .checked_mul(2)?
                        .checked_mul(size_of::<Dependency>())?,
                )
            })
            .and_then(|n| n.checked_add(4096))
            .ok_or("native transfer barrier metadata overflow")?;
        if wanted > budget {
            return Err("native transfer barrier metadata capacity");
        }
        result
            .plan
            .nodes
            .try_reserve_exact(nodes)
            .map_err(|_| "native transfer barrier node allocation")?;
        result
            .plan
            .dependencies
            .try_reserve_exact(copies * 2)
            .map_err(|_| "native transfer barrier dependency allocation")?;
        result.check_budget()?;
        Ok(result)
    }
    fn check_budget(&self) -> Result<(), &'static str> {
        let bytes = self
            .plan
            .metadata_bytes()?
            .checked_add(4096)
            .and_then(|n| n.checked_add(self.region_capacity.checked_mul(size_of::<Region>())?))
            // Conservative actual usable-entry -> hash bucket/control bound.
            .and_then(|n| {
                n.checked_add(
                    self.roots
                        .capacity()
                        .checked_mul(2)?
                        .checked_mul(size_of::<(u64, Vec<Region>)>() + 1)?,
                )
            })
            .ok_or("native transfer barrier metadata overflow")?;
        if bytes > self.budget || self.region_capacity > MAX_REGIONS * 2 {
            return Err("native transfer barrier metadata capacity");
        }
        Ok(())
    }
    #[cfg(test)]
    fn push(&mut self, copy: Option<&[Access]>) -> Result<(), &'static str> {
        self.push_scoped(copy, 1, 0)
    }
    pub(super) fn push_scoped(
        &mut self,
        copy: Option<&[Access]>,
        signal: u64,
        joined: u64,
    ) -> Result<(), &'static str> {
        if signal == 0 || joined >= signal {
            return Err("native transfer barrier visibility epoch");
        }
        if self.plan.nodes.len() >= self.expected {
            return Err("native transfer barrier node capacity");
        }
        let begin = self.plan.dependencies.len();
        if let Some(accesses) = copy {
            if accesses.is_empty() || accesses.len() > 2 || self.copied == self.copy_capacity {
                return Err("native transfer barrier original copy inventory");
            }
            self.copied += 1;
            for access in accesses {
                if access.buffer == 0
                    || access.bytes == 0
                    || access.first.checked_add(access.bytes).is_none()
                {
                    return Err("native transfer barrier original interval");
                }
            }
            // Compute ALL dependencies against preceding nodes before changing
            // state. Aliases within this one command must not invent hazards.
            for (i, access) in accesses.iter().enumerate() {
                if accesses[..i].iter().any(|a| a.buffer == access.buffer) {
                    continue;
                }
                let mut before = 0;
                let mut after = 0;
                let mut hazard = false;
                for current in accesses.iter().filter(|a| a.buffer == access.buffer) {
                    let next = if current.write { WRITE } else { READ };
                    after |= next;
                    if let Some(regions) = self.roots.get(&current.buffer) {
                        let first = regions.partition_point(|r| r.end <= current.first);
                        for old in &regions[first..] {
                            if old.first >= current.first + current.bytes {
                                break;
                            }
                            // Actual ALL_COMMANDS peer semaphore wait already
                            // completes/visibilizes this preceding COPY epoch.
                            // A same-segment write can NEVER satisfy this test.
                            if old.epoch > joined {
                                before |= old.flags;
                                hazard |= (old.flags | next) & WRITE != 0;
                            }
                        }
                    }
                }
                if hazard {
                    self.plan.dependencies.push(Dependency {
                        buffer: access.buffer,
                        before,
                        after,
                    });
                }
                // Canonicalize this node's at-most-two genuine descriptor
                // ranges. Retain gaps and read/write aliases exactly, using
                // original declared source views (never narrower copy bytes).
                let mut ends = [0u64; 4];
                let mut count = 0;
                for a in accesses.iter().filter(|a| a.buffer == access.buffer) {
                    ends[count] = a.first;
                    ends[count + 1] = a.first + a.bytes;
                    count += 2;
                }
                ends[..count].sort_unstable();
                let mut canonical = [Region {
                    first: 0,
                    end: 0,
                    flags: 0,
                    epoch: signal,
                }; 3];
                let mut used = 0;
                for pair in ends[..count].windows(2) {
                    if pair[0] == pair[1] {
                        continue;
                    }
                    let mut flags = 0;
                    for a in accesses.iter().filter(|a| a.buffer == access.buffer) {
                        if a.first <= pair[0] && a.first + a.bytes >= pair[1] {
                            flags |= if a.write { WRITE } else { READ };
                        }
                    }
                    if flags != 0 {
                        canonical[used] = Region {
                            first: pair[0],
                            end: pair[1],
                            flags,
                            epoch: signal,
                        };
                        used += 1;
                    }
                }
                // This buffer's preceding state was queried completely above;
                // other input buffers cannot alias its actual VkBuffer key.
                if !self.roots.contains_key(&access.buffer) && self.roots.len() == MAX_ROOTS {
                    return Err("native transfer barrier root capacity");
                }
                let regions = self.roots.entry(access.buffer).or_default();
                let old_capacity = regions.capacity();
                for region in &canonical[..used] {
                    replace(regions, *region, &mut self.regions)?;
                }
                self.region_capacity = self
                    .region_capacity
                    .checked_sub(old_capacity)
                    .and_then(|n| n.checked_add(regions.capacity()))
                    .ok_or("native transfer barrier region capacity overflow")?;
                self.check_budget()?;
            }
        }
        self.plan.nodes.push(begin..self.plan.dependencies.len());
        Ok(())
    }
    pub(super) fn finish(self) -> Result<Plan, &'static str> {
        if self.plan.nodes.len() != self.expected || self.copied != self.copy_capacity {
            return Err("native transfer barrier incomplete inventory");
        }
        self.check_budget()?;
        Ok(self.plan)
    }
}
fn insert(
    regions: &mut Vec<Region>,
    at: usize,
    value: Region,
    total: &mut usize,
) -> Result<(), &'static str> {
    if *total == MAX_REGIONS {
        return Err("native transfer barrier interval capacity");
    }
    regions.insert(at, value);
    *total += 1;
    Ok(())
}
fn split(regions: &mut Vec<Region>, at: u64, total: &mut usize) -> Result<(), &'static str> {
    let i = regions.partition_point(|r| r.end <= at);
    if i < regions.len() && regions[i].first < at {
        let original = regions[i];
        insert(
            regions,
            i + 1,
            Region {
                first: at,
                ..original
            },
            total,
        )?;
        regions[i].end = at;
    }
    Ok(())
}
fn replace(regions: &mut Vec<Region>, next: Region, total: &mut usize) -> Result<(), &'static str> {
    split(regions, next.first, total)?;
    split(regions, next.end, total)?;
    let mut i = regions.partition_point(|r| r.end <= next.first);
    let mut cursor = next.first;
    while cursor < next.end {
        if i == regions.len() || regions[i].first > cursor {
            let end = regions.get(i).map_or(next.end, |r| next.end.min(r.first));
            insert(
                regions,
                i,
                Region {
                    first: cursor,
                    end,
                    flags: next.flags,
                    epoch: next.epoch,
                },
                total,
            )?;
        } else {
            regions[i].flags = next.flags;
            regions[i].epoch = next.epoch;
        }
        cursor = regions[i].end;
        i += 1;
    }
    let mut keep = 0;
    for read in 0..regions.len() {
        let value = regions[read];
        if keep > 0
            && regions[keep - 1].end == value.first
            && regions[keep - 1].flags == value.flags
            && regions[keep - 1].epoch == value.epoch
        {
            regions[keep - 1].end = value.end;
        } else {
            regions[keep] = value;
            keep += 1;
        }
    }
    *total -= regions.len() - keep;
    regions.truncate(keep);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn a(buffer: u64, first: u64, bytes: u64, write: bool) -> Access {
        Access {
            buffer,
            first,
            bytes,
            write,
        }
    }

    fn segment(queue: usize, signal: u64, waits: [u64; 3]) -> Segment {
        Segment {
            copy: queue == 1,
            shared: queue == 2,
            signal,
            wait: *waits.iter().max().unwrap(),
            waits,
        }
    }
    #[test]
    fn only_real_peer_wait_ancestry_acquires_old_copy_epochs() {
        let segments = [
            segment(0, 1, [0; 3]),
            segment(1, 2, [1, 0, 0]),
            segment(0, 3, [0, 2, 0]),
            segment(2, 4, [3, 0, 0]),
            segment(1, 5, [0, 0, 4]),
            segment(1, 6, [0; 3]),
        ];
        let v = Visibility::new(&segments, 1 << 20).unwrap();
        assert_eq!(v.before(1).unwrap(), 0);
        assert_eq!(v.before(4).unwrap(), 2);
        assert_eq!(v.before(5).unwrap(), 2);
        let access = [a(1, 0, 32, true)];
        let mut r = Recorder::new(4, 4, 1 << 20).unwrap();
        r.push_scoped(Some(&access), 2, v.before(1).unwrap())
            .unwrap();
        r.push_scoped(Some(&access), 5, v.before(4).unwrap())
            .unwrap();
        // Acquiring COPY2 does NOT acquire another COPY5 write, including
        // an earlier command in the SAME actual command-buffer segment.
        r.push_scoped(Some(&access), 5, v.before(4).unwrap())
            .unwrap();
        r.push_scoped(Some(&access), 6, v.before(5).unwrap())
            .unwrap();
        let p = r.finish().unwrap();
        assert!(p.node(0).unwrap().is_empty());
        assert!(p.node(1).unwrap().is_empty());
        assert_eq!(
            p.node(2).unwrap(),
            [Dependency {
                buffer: 1,
                before: WRITE,
                after: WRITE
            }]
        );
        assert_eq!(
            p.node(3).unwrap(),
            [Dependency {
                buffer: 1,
                before: WRITE,
                after: WRITE
            }]
        );
        let without = [
            segment(0, 1, [0; 3]),
            segment(1, 2, [1, 0, 0]),
            segment(0, 3, [0; 3]),
            segment(1, 4, [3, 0, 0]),
        ];
        assert_eq!(
            Visibility::new(&without, 1 << 20)
                .unwrap()
                .before(3)
                .unwrap(),
            0
        );
        for bad in [
            vec![segment(0, 2, [0; 3])],
            vec![segment(0, 1, [0; 3]), segment(1, 2, [3, 0, 0])],
            vec![segment(0, 1, [0; 3]), segment(1, 2, [0, 1, 0])],
            vec![segment(0, 1, [0; 3]), segment(1, 2, [0, 0, 1])],
        ] {
            assert!(Visibility::new(&bad, 1 << 20).is_err());
        }
        assert!(Visibility::new(&segments, 0).is_err());
    }
    #[test]
    fn disjoint_original_role_copies_need_no_barrier_but_real_slot_reuse_does() {
        let mut r = Recorder::new(5, 4, 1 << 20).unwrap();
        r.push(Some(&[a(1, 0, 16, false), a(2, 0, 16, true)]))
            .unwrap();
        r.push(Some(&[a(1, 16, 16, false), a(2, 16, 16, true)]))
            .unwrap();
        r.push(None).unwrap();
        r.push(Some(&[a(1, 32, 16, false), a(2, 32, 16, true)]))
            .unwrap();
        r.push(Some(&[a(1, 0, 16, false), a(2, 0, 16, true)]))
            .unwrap();
        let p = r.finish().unwrap();
        for i in 0..4 {
            assert!(p.node(i).unwrap().is_empty());
        }
        assert_eq!(
            p.node(4).unwrap(),
            [Dependency {
                buffer: 2,
                before: WRITE,
                after: WRITE
            }]
        );
    }
    #[test]
    fn byte_oracle_preserves_old_views_aliases_gaps_and_all_real_read_write_hazards() {
        let mut state = 37u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let count = 8192;
        let mut r = Recorder::new(count, count, 8 << 20).unwrap();
        let mut oracle = [[0u32; 64]; 5];
        for _ in 0..count {
            let mut actual = Vec::new();
            for _ in 0..1 + (next() % 2) {
                let buffer = 1 + next() % 5;
                let first = next() % 64;
                let bytes = 1 + next() % (64 - first);
                let write = next() % 2 == 0;
                actual.push(a(buffer as u64, first as u64, bytes as u64, write));
            }
            let mut expected = Vec::new();
            for (i, x) in actual.iter().enumerate() {
                if actual[..i].iter().any(|old| old.buffer == x.buffer) {
                    continue;
                }
                let mut before = 0;
                let mut after = 0;
                let mut hazard = false;
                for y in actual.iter().filter(|y| y.buffer == x.buffer) {
                    let flags = if y.write { WRITE } else { READ };
                    after |= flags;
                    for old in &oracle[(y.buffer - 1) as usize]
                        [y.first as usize..(y.first + y.bytes) as usize]
                    {
                        before |= old;
                        hazard |= *old != 0 && (*old | flags) & WRITE != 0;
                    }
                }
                if hazard {
                    expected.push(Dependency {
                        buffer: x.buffer,
                        before,
                        after,
                    });
                }
            }
            r.push(Some(&actual)).unwrap();
            assert_eq!(r.plan.node(r.plan.nodes.len() - 1).unwrap(), expected);
            let mut current = [[0u32; 64]; 5];
            for y in &actual {
                for v in &mut current[(y.buffer - 1) as usize]
                    [y.first as usize..(y.first + y.bytes) as usize]
                {
                    *v |= if y.write { WRITE } else { READ };
                }
            }
            for b in 0..5 {
                for i in 0..64 {
                    if current[b][i] != 0 {
                        oracle[b][i] = current[b][i];
                    }
                }
            }
            for b in 0..5 {
                for i in 0..64 {
                    let real = r
                        .roots
                        .get(&(b as u64 + 1))
                        .and_then(|rs| rs.iter().find(|x| x.first <= i && i < x.end))
                        .map_or(0, |x| x.flags);
                    assert_eq!(real, oracle[b][i as usize]);
                }
            }
        }
        assert_eq!(r.finish().unwrap().nodes.len(), count);
    }
    #[test]
    fn rejects_invalid_or_unbounded_inventory_before_encoding() {
        assert!(Recorder::new(0, 0, 1 << 20).is_err());
        assert!(Recorder::new(MAX_NODES + 1, 0, 1 << 20).is_err());
        assert!(Recorder::new(1, 2, 1 << 20).is_err());
        assert!(Recorder::new(1, 1, 0).is_err());
        for bad in [a(0, 0, 1, true), a(1, 0, 0, true), a(1, u64::MAX, 1, true)] {
            let mut r = Recorder::new(1, 1, 1 << 20).unwrap();
            assert!(r.push(Some(&[bad])).is_err());
        }
        assert!(Recorder::new(1, 1, 1 << 20).unwrap().finish().is_err());
        let mut r = Recorder::new(1, 1, 1 << 20).unwrap();
        assert!(r.push(Some(&[])).is_err());
    }
}
