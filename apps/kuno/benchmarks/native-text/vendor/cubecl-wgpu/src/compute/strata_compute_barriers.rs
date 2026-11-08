//! Same-queue hazards of COMPLETE original compute/inline-transfer views.
//! Cross-queue plans/waits/releases remain untouched. No completion authority.
use ash::vk;
use std::{collections::HashMap, mem::size_of, ops::Range};
const MAX_NODES: usize = 32768;
const MAX_ROOTS: usize = 65536;
const MAX_REGIONS: usize = 262144;
const MAX_DEPS: usize = 262144;
fn writes() -> vk::AccessFlags {
    vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE
}
/// A real whole-buffer barrier retires ALL preceding root accesses, so cover
/// every subsequent native access class as well, not merely this node's mask.
pub(super) fn all_accesses() -> vk::AccessFlags {
    vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::UNIFORM_READ
        | vk::AccessFlags::INDIRECT_COMMAND_READ
        | vk::AccessFlags::TRANSFER_READ
        | vk::AccessFlags::TRANSFER_WRITE
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Access {
    pub buffer: u64,
    pub first: u64,
    pub bytes: u64,
    pub flags: vk::AccessFlags,
    pub atomic_or: bool,
}
/// Complete actual access classes from the same immutable original inventory.
pub(super) struct ClassBuilder {
    roots: [HashMap<u64, vk::AccessFlags>; 3],
    expected: usize,
    seen: usize,
    budget: usize,
}
pub(super) struct SealedClasses {
    roots: [HashMap<u64, vk::AccessFlags>; 3],
    nodes: usize,
}
fn class_bytes(roots: &[HashMap<u64, vk::AccessFlags>; 3]) -> Result<usize, &'static str> {
    roots
        .iter()
        .try_fold(4096usize, |n, r| {
            n.checked_add(
                r.capacity()
                    .checked_mul(2)?
                    .checked_mul(size_of::<(u64, vk::AccessFlags)>() + 1)?,
            )
        })
        .ok_or("native compute class metadata overflow")
}
impl ClassBuilder {
    pub fn new(nodes: usize, budget: usize) -> Result<Self, &'static str> {
        if nodes == 0 || nodes > MAX_NODES || budget < 4096 {
            return Err("native compute class inventory capacity");
        }
        Ok(Self {
            roots: Default::default(),
            expected: nodes,
            seen: 0,
            budget,
        })
    }
    pub fn push(&mut self, queue: usize, accesses: Option<&[Access]>) -> Result<(), &'static str> {
        if queue >= 3 || self.seen == self.expected {
            return Err("native compute class original node/queue");
        }
        if let Some(accesses) = accesses {
            if accesses.len() > 4129 {
                return Err("native compute class access capacity");
            }
            for a in accesses {
                if a.buffer == 0
                    || a.bytes == 0
                    || a.first.checked_add(a.bytes).is_none()
                    || a.flags.is_empty()
                    || !all_accesses().contains(a.flags)
                {
                    return Err("native compute class actual view");
                }
                if let Some(flags) = self.roots[queue].get_mut(&a.buffer) {
                    *flags |= a.flags;
                    continue;
                }
                if self.roots[queue].len() == MAX_ROOTS {
                    return Err("native compute class root capacity");
                }
                // Charge old + worst new hash allocation before growth.
                let growth = (self.roots[queue].capacity() + 4)
                    .checked_mul(4 * (size_of::<(u64, vk::AccessFlags)>() + 1))
                    .ok_or("native compute class growth overflow")?;
                if class_bytes(&self.roots)?
                    .checked_add(growth)
                    .is_none_or(|n| n > self.budget)
                {
                    return Err("native compute class metadata capacity");
                }
                self.roots[queue]
                    .try_reserve(1)
                    .map_err(|_| "native compute class allocation")?;
                self.roots[queue].insert(a.buffer, a.flags);
                if class_bytes(&self.roots)? > self.budget {
                    return Err("native compute class actual metadata capacity");
                }
            }
        }
        self.seen += 1;
        Ok(())
    }
    pub fn finish(self) -> Result<SealedClasses, &'static str> {
        if self.seen != self.expected {
            return Err("native compute class incomplete inventory");
        }
        Ok(SealedClasses {
            roots: self.roots,
            nodes: self.expected,
        })
    }
}
impl SealedClasses {
    pub fn metadata_bytes(&self) -> Result<usize, &'static str> {
        class_bytes(&self.roots)
    }
    fn flags(&self, queue: usize, buffer: u64) -> Result<vk::AccessFlags, &'static str> {
        self.roots
            .get(queue)
            .and_then(|r| r.get(&buffer))
            .copied()
            .ok_or("native compute class unknown queue/root")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Region {
    first: u64,
    end: u64,
    flags: vk::AccessFlags,
    atomic_or: bool,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Dependency {
    pub buffer: u64,
    pub before: vk::AccessFlags,
    pub after: vk::AccessFlags,
}
/// Global memory scope is at least as strong as each original full-buffer
/// scope. Pure WARs still return Some(EMPTY,EMPTY), preserving execution order.
pub(super) fn global_masks(
    dependencies: &[Dependency],
) -> Option<(vk::AccessFlags, vk::AccessFlags)> {
    if dependencies.is_empty() {
        return None;
    }
    let mut before = vk::AccessFlags::empty();
    let mut after = vk::AccessFlags::empty();
    for dependency in dependencies {
        if !dependency.after.is_empty() {
            before |= dependency.before;
            after |= dependency.after;
        }
    }
    Some((before, after))
}
struct Root {
    regions: Vec<Region>,
    legacy: vk::AccessFlags,
    future_classes: Option<vk::AccessFlags>,
}

pub(super) struct Plan {
    nodes: Vec<Range<usize>>,
    dependencies: Vec<Dependency>,
    candidates: usize,
    proven_or_views: usize,
    legacy_calls: usize,
    actual_calls: usize,
    narrowed: usize,
    execution_only: usize,
}
impl Plan {
    pub fn node(&self, index: usize) -> Result<&[Dependency], &'static str> {
        self.dependencies
            .get(
                self.nodes
                    .get(index)
                    .ok_or("native compute barrier node index")?
                    .clone(),
            )
            .ok_or("native compute barrier dependency index")
    }
    pub fn count(&self) -> usize {
        self.dependencies.len()
    }
    pub fn candidates(&self) -> usize {
        self.candidates
    }
    pub fn proven_or_views(&self) -> usize {
        self.proven_or_views
    }
    pub fn call_counts(&self) -> [usize; 4] {
        [
            self.legacy_calls,
            self.actual_calls,
            self.narrowed,
            self.execution_only,
        ]
    }
    pub fn metadata_bytes(&self) -> Result<usize, &'static str> {
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
            .ok_or("native compute barrier metadata overflow")
    }
}
pub(super) struct Recorder {
    plan: Plan,
    roots: [HashMap<u64, Root>; 3],
    region_capacity: usize,
    regions: usize,
    expected: usize,
    budget: usize,
    sealed_classes: bool,
}
impl Recorder {
    #[cfg(test)]
    pub fn new(nodes: usize, budget: usize) -> Result<Self, &'static str> {
        Self::broad(nodes, budget)
    }
    pub fn broad(nodes: usize, budget: usize) -> Result<Self, &'static str> {
        Self::build(nodes, budget, None)
    }
    pub fn with_classes(
        nodes: usize,
        budget: usize,
        classes: SealedClasses,
    ) -> Result<Self, &'static str> {
        if classes.nodes != nodes {
            return Err("native compute class original node count");
        }
        Self::build(nodes, budget, Some(classes))
    }
    fn build(
        nodes: usize,
        budget: usize,
        classes: Option<SealedClasses>,
    ) -> Result<Self, &'static str> {
        if nodes == 0 || nodes > MAX_NODES {
            return Err("native compute barrier node capacity");
        }
        let wanted = nodes
            .checked_mul(size_of::<Range<usize>>() + 4 * size_of::<Dependency>())
            .and_then(|n| n.checked_add(4096))
            .ok_or("native compute barrier metadata overflow")?;
        if wanted
            .checked_add(
                classes
                    .as_ref()
                    .map_or(Ok(0), SealedClasses::metadata_bytes)?,
            )
            .is_none_or(|n| n > budget)
        {
            return Err("native compute barrier metadata capacity");
        }
        let mut result = Self {
            plan: Plan {
                nodes: Vec::new(),
                dependencies: Vec::new(),
                candidates: 0,
                proven_or_views: 0,
                legacy_calls: 0,
                actual_calls: 0,
                narrowed: 0,
                execution_only: 0,
            },
            roots: Default::default(),
            region_capacity: 0,
            regions: 0,
            expected: nodes,
            budget,
            sealed_classes: classes.is_some(),
        };
        result
            .plan
            .nodes
            .try_reserve_exact(nodes)
            .map_err(|_| "native compute barrier node allocation")?;
        result
            .plan
            .dependencies
            .try_reserve_exact(nodes * 4)
            .map_err(|_| "native compute barrier dependency allocation")?;
        if let Some(classes) = classes {
            let original_class_bytes = classes.metadata_bytes()?;
            for (queue, entries) in classes.roots.into_iter().enumerate() {
                let next = (entries.len() + 4)
                    .checked_mul(4 * (size_of::<(u64, Root)>() + 1))
                    .and_then(|n| n.checked_add(original_class_bytes))
                    .ok_or("native compute class root metadata overflow")?;
                result.check(next)?;
                result.roots[queue]
                    .try_reserve(entries.len())
                    .map_err(|_| "native compute class root allocation")?;
                result.check(original_class_bytes)?;
                for (buffer, future) in entries {
                    result.roots[queue].insert(
                        buffer,
                        Root {
                            regions: Vec::new(),
                            legacy: vk::AccessFlags::empty(),
                            future_classes: Some(future),
                        },
                    );
                }
            }
        }
        result.check(0)?;
        Ok(result)
    }
    fn check(&self, extra: usize) -> Result<(), &'static str> {
        let hash_bytes = self
            .roots
            .iter()
            .try_fold(0usize, |n, r| {
                n.checked_add(
                    r.capacity()
                        .checked_mul(2)?
                        .checked_mul(size_of::<(u64, Root)>() + 1)?,
                )
            })
            .ok_or("native compute barrier root metadata overflow")?;
        let n = self
            .plan
            .metadata_bytes()?
            .checked_add(hash_bytes)
            .and_then(|n| n.checked_add(self.region_capacity.checked_mul(size_of::<Region>())?))
            .and_then(|n| n.checked_add(extra))
            .and_then(|n| n.checked_add(4096))
            .ok_or("native compute barrier metadata overflow")?;
        if n > self.budget || self.region_capacity > 2 * MAX_REGIONS {
            return Err("native compute barrier metadata capacity");
        }
        Ok(())
    }
    fn dependency(&mut self, dependency: Dependency) -> Result<(), &'static str> {
        if self.plan.dependencies.len() == MAX_DEPS {
            return Err("native compute barrier dependency capacity");
        }
        if self.plan.dependencies.len() == self.plan.dependencies.capacity() {
            let wanted = (self.plan.dependencies.len() + 1)
                .max(self.plan.dependencies.capacity() * 2)
                .min(MAX_DEPS);
            // Charge both old and new allocations at a possible realloc peak.
            self.check(
                wanted
                    .checked_mul(size_of::<Dependency>())
                    .ok_or("native compute barrier metadata overflow")?,
            )?;
            self.plan
                .dependencies
                .try_reserve_exact(wanted - self.plan.dependencies.len())
                .map_err(|_| "native compute barrier dependency allocation")?;
            self.check(0)?;
        }
        self.plan.narrowed += usize::from(
            self.sealed_classes
                && !dependency.after.is_empty()
                && dependency.after != all_accesses(),
        );
        self.plan.execution_only += usize::from(dependency.after.is_empty());
        self.plan.dependencies.push(dependency);
        Ok(())
    }
    /// Input scratch may be sorted in place; every original view is retained.
    /// A None node belongs to the ORIGINAL dedicated COPY queue, unchanged.
    pub fn push(
        &mut self,
        queue: usize,
        accesses: Option<&mut [Access]>,
    ) -> Result<(), &'static str> {
        if queue >= 3 || self.plan.nodes.len() >= self.expected {
            return Err("native compute barrier original node/queue");
        }
        let begin = self.plan.dependencies.len();
        let legacy_begin = self.plan.candidates;
        if let Some(accesses) = accesses {
            if accesses.len() > 4129 {
                return Err("native compute barrier access capacity");
            }
            for a in accesses.iter() {
                if a.buffer == 0
                    || a.bytes == 0
                    || a.first.checked_add(a.bytes).is_none()
                    || a.flags.is_empty()
                    || !all_accesses().contains(a.flags)
                    || a.atomic_or
                        && a.flags != (vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                {
                    return Err("native compute barrier original view");
                }
            }
            self.plan.proven_or_views += accesses.iter().filter(|a| a.atomic_or).count();
            accesses.sort_unstable_by_key(|a| (a.buffer, a.first));
            let mut first = 0;
            while first < accesses.len() {
                let buffer = accesses[first].buffer;
                let end = first + accesses[first..].partition_point(|a| a.buffer == buffer);
                let current = &accesses[first..end];
                let old = self.roots[queue].get(&buffer);
                let current_flags = current
                    .iter()
                    .fold(vk::AccessFlags::empty(), |all, a| all | a.flags);
                let future = old.and_then(|r| r.future_classes);
                if self.sealed_classes && future.is_none_or(|f| !f.contains(current_flags)) {
                    return Err("native compute class unsealed future access");
                }
                let legacy = old.map_or(vk::AccessFlags::empty(), |r| r.legacy);
                let legacy_barrier =
                    !legacy.is_empty() && (legacy | current_flags).intersects(writes());
                self.plan.candidates += usize::from(legacy_barrier);
                let next_legacy = if legacy_barrier {
                    current_flags
                } else {
                    legacy | current_flags
                };
                let old = old.map(|r| &r.regions);
                let mut before = vk::AccessFlags::empty();
                let mut hazard = false;
                if let Some(old) = old {
                    for r in old {
                        before |= r.flags;
                    }
                    for a in current {
                        for r in old
                            .iter()
                            .skip_while(|r| r.end <= a.first)
                            .take_while(|r| r.first < a.first + a.bytes)
                        {
                            hazard |= (r.flags | a.flags).intersects(writes())
                                && !(r.atomic_or && a.atomic_or);
                        }
                    }
                }
                if hazard {
                    // Retire a read-only root with execution ordering alone.
                    // A disjoint outstanding writer anywhere on this full root
                    // still requires real availability/visibility operations.
                    let after = if before.intersects(writes()) {
                        future.unwrap_or_else(all_accesses)
                    } else {
                        vk::AccessFlags::empty()
                    };
                    self.dependency(Dependency {
                        buffer,
                        before,
                        after,
                    })?;
                }
                if !self.roots[queue].contains_key(&buffer) {
                    if self.roots[queue].len() == MAX_ROOTS {
                        return Err("native compute barrier root capacity");
                    }
                    // Precharge a worst double hash growth, plus old buckets.
                    self.check(
                        (self.roots[queue].capacity() + 4)
                            .checked_mul(4 * (size_of::<(u64, Root)>() + 1))
                            .ok_or("native compute barrier root metadata overflow")?,
                    )?;
                    self.roots[queue]
                        .try_reserve(1)
                        .map_err(|_| "native compute barrier root allocation")?;
                }
                let old_len = self.roots[queue]
                    .get(&buffer)
                    .map_or(0, |r| r.regions.len());
                let old_cap = self.roots[queue]
                    .get(&buffer)
                    .map_or(0, |r| r.regions.capacity());
                let wanted = (if hazard {
                    0
                } else {
                    old_len
                        .checked_mul(2)
                        .ok_or("native compute barrier region overflow")?
                })
                .checked_add(
                    current
                        .len()
                        .checked_mul(3)
                        .ok_or("native compute barrier region overflow")?,
                )
                .ok_or("native compute barrier region overflow")?;
                if wanted > MAX_REGIONS
                    || self
                        .regions
                        .checked_sub(old_len)
                        .and_then(|n| n.checked_add(wanted))
                        .is_none_or(|n| n > MAX_REGIONS)
                {
                    return Err("native compute barrier region capacity");
                }
                if wanted > old_cap {
                    self.check(
                        wanted
                            .checked_mul(size_of::<Region>())
                            .ok_or("native compute barrier region overflow")?,
                    )?;
                }
                let root = self.roots[queue].entry(buffer).or_insert_with(|| Root {
                    regions: Vec::new(),
                    legacy: vk::AccessFlags::empty(),
                    future_classes: None,
                });
                root.legacy = next_legacy;
                let regions = &mut root.regions;
                if wanted > old_cap {
                    regions
                        .try_reserve_exact(wanted - regions.len())
                        .map_err(|_| "native compute barrier region allocation")?;
                }
                self.region_capacity = self
                    .region_capacity
                    .checked_sub(old_cap)
                    .and_then(|n| n.checked_add(regions.capacity()))
                    .ok_or("native compute barrier region overflow")?;
                if hazard {
                    self.regions -= regions.len();
                    regions.clear();
                }
                for a in current {
                    cover(
                        regions,
                        Region {
                            first: a.first,
                            end: a.first + a.bytes,
                            flags: a.flags,
                            atomic_or: a.atomic_or,
                        },
                        &mut self.regions,
                    )?;
                }
                self.check(0)?;
                first = end;
            }
        }
        self.plan.legacy_calls += usize::from(self.plan.candidates != legacy_begin);
        self.plan.actual_calls += usize::from(self.plan.dependencies.len() != begin);
        self.plan.nodes.push(begin..self.plan.dependencies.len());
        Ok(())
    }
    pub fn finish(self) -> Result<Plan, &'static str> {
        if self.plan.nodes.len() != self.expected {
            return Err("native compute barrier incomplete inventory");
        }
        self.check(0)?;
        Ok(self.plan)
    }
}
fn insert(
    regions: &mut Vec<Region>,
    index: usize,
    value: Region,
    total: &mut usize,
) -> Result<(), &'static str> {
    if regions.len() == regions.capacity() || *total >= MAX_REGIONS {
        return Err("native compute barrier precharged interval capacity");
    }
    regions.insert(index, value);
    *total += 1;
    Ok(())
}
fn split(regions: &mut Vec<Region>, at: u64, total: &mut usize) -> Result<(), &'static str> {
    let i = regions.partition_point(|r| r.end <= at);
    if i < regions.len() && regions[i].first < at {
        let old = regions[i];
        insert(regions, i + 1, Region { first: at, ..old }, total)?;
        regions[i].end = at;
    }
    Ok(())
}
fn cover(regions: &mut Vec<Region>, new: Region, total: &mut usize) -> Result<(), &'static str> {
    split(regions, new.first, total)?;
    split(regions, new.end, total)?;
    let mut i = regions.partition_point(|r| r.end <= new.first);
    let mut at = new.first;
    while at < new.end {
        if i == regions.len() || regions[i].first > at {
            let end = regions.get(i).map_or(new.end, |r| new.end.min(r.first));
            insert(
                regions,
                i,
                Region {
                    first: at,
                    end,
                    flags: new.flags,
                    atomic_or: new.atomic_or,
                },
                total,
            )?;
        } else {
            regions[i].flags |= new.flags;
            regions[i].atomic_or &= new.atomic_or;
        }
        at = regions[i].end;
        i += 1;
    }
    let mut keep = 0;
    for read in 0..regions.len() {
        let r = regions[read];
        if keep > 0
            && regions[keep - 1].end == r.first
            && regions[keep - 1].flags == r.flags
            && regions[keep - 1].atomic_or == r.atomic_or
        {
            regions[keep - 1].end = r.end;
        } else {
            regions[keep] = r;
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
    fn a(buffer: u64, first: u64, bytes: u64, flags: vk::AccessFlags) -> Access {
        Access {
            buffer,
            first,
            bytes,
            flags,
            atomic_or: false,
        }
    }
    #[test]
    fn real_disjoint_views_aliases_and_indirect_transfer_classes_keep_exact_hazards() {
        let r = vk::AccessFlags::SHADER_READ;
        let w = vk::AccessFlags::SHADER_WRITE;
        let mut p = Recorder::new(7, 1 << 20).unwrap();
        p.push(0, Some(&mut [a(1, 0, 16, w)])).unwrap();
        p.push(0, Some(&mut [a(1, 32, 16, w)])).unwrap();
        p.push(0, Some(&mut [a(1, 16, 16, r)])).unwrap();
        p.push(0, Some(&mut [a(1, 8, 8, r), a(1, 12, 8, w)]))
            .unwrap();
        p.push(
            0,
            Some(&mut [a(1, 0, 8, vk::AccessFlags::INDIRECT_COMMAND_READ)]),
        )
        .unwrap();
        p.push(0, Some(&mut [a(1, 16, 4, vk::AccessFlags::TRANSFER_READ)]))
            .unwrap();
        p.push(0, Some(&mut [a(1, 48, 16, r), a(1, 56, 16, w)]))
            .unwrap();
        let p = p.finish().unwrap();
        assert_eq!(
            (0..7).map(|i| p.node(i).unwrap().len()).collect::<Vec<_>>(),
            [0, 0, 0, 1, 0, 1, 0]
        );
        assert!(all_accesses().contains(vk::AccessFlags::UNIFORM_READ | writes()));
    }
    #[test]
    fn three_queue_masks_match_an_independent_byte_addressed_oracle() {
        const N: usize = 8192;
        let mut p = Recorder::new(N, 16 << 20).unwrap();
        let mut state = [[[0u32; 64]; 4]; 3];
        let mut seed = 0x5a67c912u64;
        let mut legacy = [[0u32; 4]; 3];
        let mut legacy_count = 0usize;
        for index in 0..N {
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let queue = (next() % 3) as usize;
            let n = (next() % 6 + 1) as usize;
            let mut input = Vec::new();
            let mut current = [[0u32; 64]; 4];
            for _ in 0..n {
                let b = (next() % 4) as usize;
                let start = (next() % 60) as usize;
                let len = (next() % 4 + 1) as usize;
                let flag = match next() % 5 {
                    0 => vk::AccessFlags::SHADER_WRITE,
                    1 => vk::AccessFlags::TRANSFER_WRITE,
                    2 => vk::AccessFlags::INDIRECT_COMMAND_READ,
                    3 => vk::AccessFlags::UNIFORM_READ,
                    _ => vk::AccessFlags::SHADER_READ,
                };
                input.push(a(b as u64 + 1, start as u64, len as u64, flag));
                for v in &mut current[b][start..start + len] {
                    *v |= flag.as_raw();
                }
            }
            for b in 0..4 {
                let current_flags = current[b].iter().fold(0u32, |a, b| a | *b);
                if current_flags == 0 {
                    continue;
                }
                let before = legacy[queue][b];
                if before != 0 && (before | current_flags) & writes().as_raw() != 0 {
                    legacy_count += 1;
                    legacy[queue][b] = current_flags;
                } else {
                    legacy[queue][b] |= current_flags;
                }
            }
            p.push(queue, Some(&mut input)).unwrap();
            assert_eq!(p.plan.candidates, legacy_count);
            let deps = p.plan.node(index).unwrap();
            for b in 0..4 {
                let hazard = (0..64).any(|at| {
                    current[b][at] != 0
                        && state[queue][b][at] != 0
                        && (current[b][at] | state[queue][b][at]) & writes().as_raw() != 0
                });
                let actual = deps
                    .iter()
                    .filter(|d| d.buffer == b as u64 + 1)
                    .collect::<Vec<_>>();
                assert_eq!(actual.len(), usize::from(hazard));
                if hazard {
                    let before = state[queue][b].iter().fold(0, |a, b| a | *b);
                    assert_eq!(actual[0].before.as_raw(), before);
                    state[queue][b] = [0; 64];
                }
                for at in 0..64 {
                    state[queue][b][at] |= current[b][at];
                }
            }
        }
        assert_eq!(p.finish().unwrap().nodes.len(), N);
    }
    #[test]
    fn mixed_atomic_and_non_atomic_views_match_a_byte_oracle_on_all_three_queues() {
        let mut p = Recorder::new(2048, 8 << 20).unwrap();
        let mut flags = [[[0u32; 64]; 4]; 3];
        let mut atomic = [[[false; 64]; 4]; 3];
        let mut seed = 0xabc876u64;
        for index in 0..2048 {
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let q = (random() % 3) as usize;
            let mut input = Vec::new();
            let mut current = [[0u32; 64]; 4];
            let mut is_or = [[false; 64]; 4];
            for _ in 0..(random() % 6 + 1) {
                let b = (random() % 4) as usize;
                let first = (random() % 60) as usize;
                let len = (random() % 4 + 1) as usize;
                let mode = random() % 4;
                let f = match mode {
                    0 => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    1 => vk::AccessFlags::SHADER_READ,
                    2 => vk::AccessFlags::TRANSFER_WRITE,
                    _ => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                };
                let ao = mode == 0;
                input.push(Access {
                    buffer: b as u64 + 1,
                    first: first as u64,
                    bytes: len as u64,
                    flags: f,
                    atomic_or: ao,
                });
                for at in first..first + len {
                    is_or[b][at] = if current[b][at] == 0 {
                        ao
                    } else {
                        is_or[b][at] && ao
                    };
                    current[b][at] |= f.as_raw();
                }
            }
            p.push(q, Some(&mut input)).unwrap();
            let actual = p.plan.node(index).unwrap();
            for b in 0..4 {
                let hazard = (0..64).any(|at| {
                    flags[q][b][at] != 0
                        && current[b][at] != 0
                        && (flags[q][b][at] | current[b][at]) & writes().as_raw() != 0
                        && !(atomic[q][b][at] && is_or[b][at])
                });
                let found = actual
                    .iter()
                    .filter(|d| d.buffer == b as u64 + 1)
                    .collect::<Vec<_>>();
                assert_eq!(found.len(), usize::from(hazard));
                if hazard {
                    assert_eq!(
                        found[0].before.as_raw(),
                        flags[q][b].iter().fold(0, |a, b| a | *b)
                    );
                    flags[q][b] = [0; 64];
                    atomic[q][b] = [false; 64];
                }
                for at in 0..64 {
                    if current[b][at] != 0 {
                        atomic[q][b][at] = if flags[q][b][at] == 0 {
                            is_or[b][at]
                        } else {
                            atomic[q][b][at] && is_or[b][at]
                        };
                        flags[q][b][at] |= current[b][at];
                    }
                }
            }
        }
        assert!(p.finish().unwrap().proven_or_views() > 0);
    }
    #[test]
    fn pure_war_uses_only_execution_but_disjoint_pending_writes_cannot_be_retired_without_memory() {
        let mut r = Recorder::new(2, 1 << 20).unwrap();
        r.push(0, Some(&mut [a(1, 0, 8, vk::AccessFlags::SHADER_READ)]))
            .unwrap();
        r.push(0, Some(&mut [a(1, 0, 8, vk::AccessFlags::SHADER_WRITE)]))
            .unwrap();
        let p = r.finish().unwrap();
        assert!(p.node(1).unwrap()[0].after.is_empty());
        assert_eq!(p.call_counts(), [1, 1, 0, 1]);
        let mut r = Recorder::new(2, 1 << 20).unwrap();
        r.push(
            0,
            Some(&mut [
                a(1, 0, 8, vk::AccessFlags::SHADER_READ),
                a(1, 32, 8, vk::AccessFlags::SHADER_WRITE),
            ]),
        )
        .unwrap();
        r.push(0, Some(&mut [a(1, 0, 8, vk::AccessFlags::SHADER_WRITE)]))
            .unwrap();
        let p = r.finish().unwrap();
        assert_eq!(p.node(1).unwrap()[0].after, all_accesses());
    }
    #[test]
    fn real_global_mask_producer_unions_data_dependencies_and_preserves_execution_only_calls() {
        let deps = [
            Dependency {
                buffer: 1,
                before: vk::AccessFlags::SHADER_WRITE,
                after: vk::AccessFlags::SHADER_READ,
            },
            Dependency {
                buffer: 2,
                before: vk::AccessFlags::TRANSFER_WRITE,
                after: vk::AccessFlags::UNIFORM_READ,
            },
            Dependency {
                buffer: 3,
                before: vk::AccessFlags::INDIRECT_COMMAND_READ,
                after: vk::AccessFlags::empty(),
            },
        ];
        assert_eq!(
            global_masks(&deps),
            Some((
                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::UNIFORM_READ
            ))
        );
        assert_eq!(
            global_masks(&deps[2..]),
            Some((vk::AccessFlags::empty(), vk::AccessFlags::empty()))
        );
        assert_eq!(global_masks(&[]), None);
    }
    #[test]
    fn sealed_future_classes_cover_untouched_bytes_without_importing_other_queue_classes() {
        let inputs = vec![
            (0, vec![a(1, 0, 16, vk::AccessFlags::SHADER_WRITE)]),
            (0, vec![a(1, 32, 16, vk::AccessFlags::SHADER_WRITE)]),
            (0, vec![a(1, 8, 8, vk::AccessFlags::SHADER_READ)]),
            (
                0,
                vec![a(1, 32, 16, vk::AccessFlags::INDIRECT_COMMAND_READ)],
            ),
            (2, vec![a(1, 0, 48, vk::AccessFlags::TRANSFER_READ)]),
            (0, vec![a(1, 8, 8, vk::AccessFlags::SHADER_WRITE)]),
        ];
        let mut c = ClassBuilder::new(inputs.len(), 1 << 20).unwrap();
        for (q, views) in &inputs {
            c.push(*q, Some(views)).unwrap();
        }
        let c = c.finish().unwrap();
        let expected = vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::INDIRECT_COMMAND_READ;
        assert_eq!(c.flags(0, 1).unwrap(), expected);
        assert_eq!(c.flags(2, 1).unwrap(), vk::AccessFlags::TRANSFER_READ);
        let mut r = Recorder::with_classes(inputs.len(), 1 << 20, c).unwrap();
        for (q, mut views) in inputs {
            r.push(q, Some(&mut views)).unwrap();
        }
        let p = r.finish().unwrap();
        assert_eq!(
            (0..6).map(|i| p.node(i).unwrap().len()).collect::<Vec<_>>(),
            [0, 0, 1, 0, 0, 1]
        );
        assert_eq!(p.node(2).unwrap()[0].after, expected);
        assert_eq!(p.call_counts(), [3, 2, 1, 1]);
    }
    #[test]
    fn unsealed_future_class_unknown_root_queue_count_and_capacity_cannot_grant_a_mask() {
        assert!(ClassBuilder::new(0, 1 << 20).is_err());
        assert!(ClassBuilder::new(MAX_NODES + 1, 1 << 20).is_err());
        assert!(ClassBuilder::new(1, 4095).is_err());
        assert!(ClassBuilder::new(1, 1 << 20).unwrap().finish().is_err());
        assert!(
            ClassBuilder::new(1, 4096)
                .unwrap()
                .push(0, Some(&[a(1, 0, 4, writes())]))
                .is_err()
        );
        let sealed = || {
            let mut c = ClassBuilder::new(2, 1 << 20).unwrap();
            c.push(0, Some(&[a(1, 0, 4, vk::AccessFlags::SHADER_WRITE)]))
                .unwrap();
            c.push(0, Some(&[a(1, 0, 4, vk::AccessFlags::SHADER_READ)]))
                .unwrap();
            c.finish().unwrap()
        };
        assert!(Recorder::with_classes(3, 1 << 20, sealed()).is_err());
        for (q, view) in [
            (0, a(1, 0, 4, vk::AccessFlags::INDIRECT_COMMAND_READ)),
            (2, a(1, 0, 4, vk::AccessFlags::SHADER_READ)),
            (0, a(2, 0, 4, vk::AccessFlags::SHADER_READ)),
        ] {
            let mut p = Recorder::with_classes(2, 1 << 20, sealed()).unwrap();
            p.push(0, Some(&mut [a(1, 0, 4, vk::AccessFlags::SHADER_WRITE)]))
                .unwrap();
            assert!(p.push(q, Some(&mut [view])).is_err());
            assert!(p.finish().is_err());
        }
    }
    #[test]
    fn sealed_manifest_mixed_aliases_match_the_independent_byte_oracle() {
        let mut seed = 0x987654u64;
        let mut inputs = Vec::new();
        let mut classes = [[vk::AccessFlags::empty(); 4]; 3];
        for _ in 0..2048 {
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let q = (random() % 3) as usize;
            let mut views = Vec::new();
            for _ in 0..(random() % 6 + 1) {
                let b = (random() % 4) as usize;
                let first = random() % 60;
                let bytes = random() % 4 + 1;
                let kind = random() % 6;
                let flags = match kind {
                    0 => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    1 => vk::AccessFlags::SHADER_READ,
                    2 => vk::AccessFlags::TRANSFER_WRITE,
                    3 => vk::AccessFlags::UNIFORM_READ,
                    4 => vk::AccessFlags::INDIRECT_COMMAND_READ,
                    _ => vk::AccessFlags::SHADER_WRITE,
                };
                views.push(Access {
                    buffer: b as u64 + 1,
                    first,
                    bytes,
                    flags,
                    atomic_or: kind == 0,
                });
                classes[q][b] |= flags;
            }
            inputs.push((q, views));
        }
        let mut c = ClassBuilder::new(inputs.len(), 8 << 20).unwrap();
        for (q, v) in &inputs {
            c.push(*q, Some(v)).unwrap();
        }
        let mut p = Recorder::with_classes(inputs.len(), 8 << 20, c.finish().unwrap()).unwrap();
        let mut flags = [[[0u32; 64]; 4]; 3];
        let mut atomic = [[[false; 64]; 4]; 3];
        for (index, (q, mut input)) in inputs.into_iter().enumerate() {
            let mut current = [[0u32; 64]; 4];
            let mut is_or = [[false; 64]; 4];
            for a in &input {
                let b = (a.buffer - 1) as usize;
                for at in a.first as usize..(a.first + a.bytes) as usize {
                    is_or[b][at] = if current[b][at] == 0 {
                        a.atomic_or
                    } else {
                        is_or[b][at] && a.atomic_or
                    };
                    current[b][at] |= a.flags.as_raw();
                }
            }
            p.push(q, Some(&mut input)).unwrap();
            let actual = p.plan.node(index).unwrap();
            for b in 0..4 {
                let hazard = (0..64).any(|at| {
                    flags[q][b][at] != 0
                        && current[b][at] != 0
                        && (flags[q][b][at] | current[b][at]) & writes().as_raw() != 0
                        && !(atomic[q][b][at] && is_or[b][at])
                });
                let found = actual
                    .iter()
                    .filter(|d| d.buffer == b as u64 + 1)
                    .collect::<Vec<_>>();
                assert_eq!(found.len(), usize::from(hazard));
                if hazard {
                    assert_eq!(
                        found[0].before.as_raw(),
                        flags[q][b].iter().fold(0, |a, b| a | *b)
                    );
                    assert_eq!(
                        found[0].after,
                        if found[0].before.intersects(writes()) {
                            classes[q][b]
                        } else {
                            vk::AccessFlags::empty()
                        }
                    );
                    flags[q][b] = [0; 64];
                    atomic[q][b] = [false; 64];
                }
                for at in 0..64 {
                    if current[b][at] != 0 {
                        atomic[q][b][at] = if flags[q][b][at] == 0 {
                            is_or[b][at]
                        } else {
                            atomic[q][b][at] && is_or[b][at]
                        };
                        flags[q][b][at] |= current[b][at];
                    }
                }
            }
        }
        assert!(p.finish().unwrap().call_counts()[1] > 0);
    }
    #[test]
    fn proven_or_never_bypasses_clear_observation_or_a_non_atomic_alias() {
        let rw = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        let atomic = || Access {
            buffer: 1,
            first: 0,
            bytes: 4,
            flags: rw,
            atomic_or: true,
        };
        let mut p = Recorder::new(11, 1 << 20).unwrap();
        p.push(0, Some(&mut [a(1, 0, 4, vk::AccessFlags::TRANSFER_WRITE)]))
            .unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        p.push(0, Some(&mut [a(1, 0, 4, vk::AccessFlags::SHADER_READ)]))
            .unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        p.push(
            0,
            Some(&mut [atomic(), a(1, 0, 4, vk::AccessFlags::SHADER_READ)]),
        )
        .unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        p.push(1, None).unwrap();
        p.push(0, Some(&mut [a(1, 0, 4, vk::AccessFlags::SHADER_WRITE)]))
            .unwrap();
        p.push(0, Some(&mut [atomic()])).unwrap();
        let p = p.finish().unwrap();
        assert_eq!(
            (0..11)
                .map(|n| p.node(n).unwrap().len())
                .collect::<Vec<_>>(),
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]
        );
        assert_eq!(p.candidates(), 9);
        let mut invalid = atomic();
        invalid.flags = vk::AccessFlags::SHADER_READ;
        assert!(
            Recorder::new(1, 1 << 20)
                .unwrap()
                .push(0, Some(&mut [invalid]))
                .is_err()
        );
    }
    #[test]
    fn interval_capacity_failure_cannot_reallocate_or_lose_the_previous_span() {
        let mut regions = Vec::with_capacity(1);
        regions.push(Region {
            first: 0,
            end: 8,
            flags: vk::AccessFlags::SHADER_READ,
            atomic_or: false,
        });
        let pointer = regions.as_ptr();
        let mut total = 1;
        assert!(
            cover(
                &mut regions,
                Region {
                    first: 2,
                    end: 4,
                    flags: writes(),
                    atomic_or: false,
                },
                &mut total
            )
            .is_err()
        );
        assert_eq!(regions.as_ptr(), pointer);
        assert_eq!(regions.capacity(), 1);
        assert_eq!(regions.len(), 1);
        assert_eq!((regions[0].first, regions[0].end, total), (0, 8, 1));
    }
    #[test]
    fn invalid_views_queues_and_real_capacity_fail_before_submission() {
        assert!(Recorder::new(0, 1 << 20).is_err());
        assert!(Recorder::new(32769, 1 << 20).is_err());
        assert!(Recorder::new(4, 1).is_err());
        for view in [
            a(0, 0, 1, writes()),
            a(1, u64::MAX, 2, writes()),
            a(1, 0, 0, writes()),
            a(1, 0, 1, vk::AccessFlags::HOST_WRITE),
            a(1, 0, 1, vk::AccessFlags::empty()),
        ] {
            assert!(
                Recorder::new(1, 1 << 20)
                    .unwrap()
                    .push(0, Some(&mut [view]))
                    .is_err()
            );
        }
        assert!(Recorder::new(1, 1 << 20).unwrap().push(3, None).is_err());
        assert!(Recorder::new(1, 1 << 20).unwrap().finish().is_err());
    }
}
