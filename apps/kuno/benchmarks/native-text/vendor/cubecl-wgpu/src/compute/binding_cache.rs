// Bounded exact binding-object reuse for one exclusive private request.
use std::hash::{Hash, Hasher};

#[cfg(test)]
const ENTRIES: usize = 512;
const CAPACITIES: [usize; 4] = [256, 512, 2048, 4096];
const BUCKETS: usize = 1024;
const MAX_BINDINGS: usize = 32;

#[derive(Debug)]
struct Binding<B> {
    buffer: B,
    offset: u64,
    size: u64,
}
#[derive(Debug)]
struct Entry<P, B, G> {
    pipeline: P,
    bindings: Vec<Binding<B>>,
    group: G,
    hash: u64,
    next: u32,
}

// Drop every live Rust value before overwriting its now-uninitialized capacity.
fn wipe_empty_capacity<T>(values: &mut Vec<T>) {
    values.clear();
    let pointer = values.as_mut_ptr().cast::<u8>();
    let bytes = values.capacity() * std::mem::size_of::<T>();
    for n in 0..bytes {
        // SAFETY: length is zero; only exclusively owned allocation bytes are
        // written, including spare capacity. No invalid T is read or dropped.
        unsafe {
            pointer.add(n).write_volatile(0);
        }
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}
impl<P, B, G> Drop for Entry<P, B, G> {
    fn drop(&mut self) {
        wipe_empty_capacity(&mut self.bindings);
    }
}

#[derive(Debug)]
pub(super) struct BindingCache<P, B, G> {
    entries: Vec<Entry<P, B, G>>,
    buckets: [u32; BUCKETS],
    binding_bytes: usize,
    next_replacement: usize,
}
impl<P, B, G> BindingCache<P, B, G> {
    pub fn reserved_bytes_for_limit(limit: usize) -> Option<usize> {
        if !CAPACITIES.contains(&limit) {
            return None;
        }
        limit
            .checked_mul(std::mem::size_of::<Entry<P, B, G>>())?
            .checked_add(std::mem::size_of::<Self>())
    }
    #[cfg(test)]
    pub fn maximum_key_bytes_for_limit(limit: usize) -> Option<usize> {
        Self::reserved_bytes_for_limit(limit)?;
        limit
            .checked_mul(MAX_BINDINGS)?
            .checked_mul(std::mem::size_of::<Binding<B>>())
    }
    #[cfg(test)]
    pub fn reserved_bytes() -> usize {
        Self::reserved_bytes_for_limit(ENTRIES).unwrap()
    }
    #[cfg(test)]
    pub fn new() -> Option<Self> {
        Self::with_limit(ENTRIES)
    }
    pub fn with_limit(limit: usize) -> Option<Self> {
        if !CAPACITIES.contains(&limit) {
            return None;
        }
        let mut entries = Vec::new();
        entries.try_reserve_exact(limit).ok()?;
        if entries.capacity() != limit {
            return None;
        }
        Some(Self {
            entries,
            buckets: [0; BUCKETS],
            binding_bytes: 0,
            next_replacement: 0,
        })
    }
    pub fn host_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.entries.capacity() * std::mem::size_of::<Entry<P, B, G>>()
            + self.binding_bytes
    }
}
impl<P: Clone + Eq + Hash, B: Clone + Eq + Hash, G: Clone> BindingCache<P, B, G> {
    fn hash<'a>(pipeline: &P, bindings: impl Iterator<Item = (&'a B, u64, u64)> + Clone) -> u64
    where
        B: 'a,
    {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        pipeline.hash(&mut hash);
        bindings.clone().count().hash(&mut hash);
        for (buffer, offset, size) in bindings {
            buffer.hash(&mut hash);
            offset.hash(&mut hash);
            size.hash(&mut hash);
        }
        hash.finish()
    }
    pub fn find<'a>(
        &self,
        pipeline: &P,
        bindings: impl Iterator<Item = (&'a B, u64, u64)> + Clone,
    ) -> Option<G>
    where
        B: 'a,
    {
        let hash = Self::hash(pipeline, bindings.clone());
        let count = bindings.clone().count();
        let mut link = self.buckets[hash as usize & (BUCKETS - 1)];
        for _ in 0..self.entries.len() {
            let entry = self.entries.get(link.checked_sub(1)? as usize)?;
            if entry.hash == hash
                && entry.pipeline == *pipeline
                && entry.bindings.len() == count
                && entry
                    .bindings
                    .iter()
                    .zip(bindings.clone())
                    .all(|(a, (b, o, s))| a.buffer == *b && a.offset == o && a.size == s)
            {
                return Some(entry.group.clone());
            }
            link = entry.next;
        }
        None
    }
    // Validate the complete old bucket before changing a single link. A
    // repeated link cannot terminate at zero, so the existing entry count is
    // also a cycle bound without allocating a private visited-index table.
    fn replacement_link(&self, victim: usize) -> Option<(usize, Option<usize>, u32)> {
        let bucket = self.entries.get(victim)?.hash as usize & (BUCKETS - 1);
        let mut link = self.buckets[bucket];
        let mut previous = None;
        let mut found = None;
        for _ in 0..self.entries.len() {
            if link == 0 {
                return found.map(|(previous, next)| (bucket, previous, next));
            }
            let index = link.checked_sub(1)? as usize;
            let entry = self.entries.get(index)?;
            if entry.hash as usize & (BUCKETS - 1) != bucket {
                return None;
            }
            if index == victim {
                if found.is_some() {
                    return None;
                }
                found = Some((previous, entry.next));
            }
            previous = Some(index);
            link = entry.next;
        }
        (link == 0)
            .then_some(found)
            .flatten()
            .map(|(previous, next)| (bucket, previous, next))
    }
    pub fn insert<'a>(
        &mut self,
        pipeline: &P,
        bindings: impl Iterator<Item = (&'a B, u64, u64)> + Clone,
        group: G,
        max_bytes: usize,
    ) -> bool
    where
        B: 'a,
    {
        let count = bindings.clone().count();
        let full = self.entries.len() == self.entries.capacity();
        // Original small caches remain append-only. Extended laboratory caches
        // replace FIFO entries only after their fixed allocation is complete.
        // A key-byte ceiling reached before a full table still falls back; this
        // first candidate never evicts multiple entries or grants early credit.
        if (full && self.entries.capacity() < 2048) || count == 0 || count > MAX_BINDINGS {
            return false;
        }
        let replacement = if full {
            let victim = self.next_replacement;
            let Some((bucket, previous, next)) = self.replacement_link(victim) else {
                return false;
            };
            Some((victim, bucket, previous, next))
        } else {
            None
        };
        let Some(bytes) = count.checked_mul(std::mem::size_of::<Binding<B>>()) else {
            return false;
        };
        // During replacement both old and new key allocations, the staged
        // Entry, and the retired Entry are charged. No speculative eviction
        // credit is granted before allocation/cloning completes.
        let temporary = if full {
            2 * std::mem::size_of::<Entry<P, B, G>>()
        } else {
            0
        };
        if self
            .host_bytes()
            .checked_add(bytes)
            .and_then(|n| n.checked_add(temporary))
            .is_none_or(|n| n > max_bytes)
        {
            return false;
        }
        let mut owned = Vec::new();
        if owned.try_reserve_exact(count).is_err() {
            return false;
        }
        // Charge actual capacity before cloning any request-derived owners.
        let Some(actual) = owned
            .capacity()
            .checked_mul(std::mem::size_of::<Binding<B>>())
        else {
            return false;
        };
        if self
            .host_bytes()
            .checked_add(actual)
            .and_then(|n| n.checked_add(temporary))
            .is_none_or(|n| n > max_bytes)
        {
            return false;
        }
        let hash = Self::hash(pipeline, bindings.clone());
        // The staged Entry owns/wipes key capacity if any later clone unwinds.
        let mut staged = Entry {
            pipeline: pipeline.clone(),
            bindings: owned,
            group,
            hash,
            next: 0,
        };
        for (buffer, offset, size) in bindings {
            staged.bindings.push(Binding {
                buffer: buffer.clone(),
                offset,
                size,
            });
        }
        let bucket = hash as usize & (BUCKETS - 1);
        if let Some((victim, old_bucket, previous, next)) = replacement {
            let Some(old_bytes) = self.entries[victim]
                .bindings
                .capacity()
                .checked_mul(std::mem::size_of::<Binding<B>>())
            else {
                return false;
            };
            let Some(updated) = self
                .binding_bytes
                .checked_add(actual)
                .and_then(|n| n.checked_sub(old_bytes))
            else {
                return false;
            };
            // All allocation, key cloning, and old-chain validation succeeded.
            // Unlink before reading the new bucket head (they may be the same).
            if let Some(previous) = previous {
                self.entries[previous].next = next;
            } else {
                self.buckets[old_bucket] = next;
            }
            staged.next = self.buckets[bucket];
            let retired = std::mem::replace(&mut self.entries[victim], staged);
            self.buckets[bucket] = victim as u32 + 1;
            self.binding_bytes = updated;
            self.next_replacement = (victim + 1) % self.entries.len();
            // Drops cached references and wipes the full OLD binding capacity;
            // original private pages/INFO pins and their ledger stay untouched.
            drop(retired);
        } else {
            staged.next = self.buckets[bucket];
            self.entries.push(staged);
            self.binding_bytes += actual;
            self.buckets[bucket] = self.entries.len() as u32;
        }
        true
    }
}
impl<P, B, G> Drop for BindingCache<P, B, G> {
    fn drop(&mut self) {
        wipe_empty_capacity(&mut self.entries);
        for value in &mut self.buckets {
            // SAFETY: initialized scalar exclusively owned during Drop.
            unsafe {
                std::ptr::write_volatile(value, 0);
            }
        }
        // SAFETY: initialized scalar exclusively owned during Drop.
        unsafe {
            std::ptr::write_volatile(&mut self.binding_bytes, 0);
            std::ptr::write_volatile(&mut self.next_replacement, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[test]
    fn explicit_capacity_uses_actual_reservation_without_extra_credit() {
        for limit in CAPACITIES {
            let mut cache = BindingCache::<u64, u64, u64>::with_limit(limit).unwrap();
            assert_eq!(cache.entries.capacity(), limit);
            assert_eq!(
                cache.host_bytes(),
                BindingCache::<u64, u64, u64>::reserved_bytes_for_limit(limit).unwrap()
            );
            for n in 0..limit {
                assert!(cache.insert(
                    &1,
                    std::iter::once(&(n as u64)).map(|b| (b, 0, 64)),
                    2,
                    usize::MAX
                ));
            }
            assert_eq!(
                cache.insert(&1, std::iter::once((&9999, 0, 64)), 2, usize::MAX),
                limit >= 2048
            );
        }
        assert!(BindingCache::<u64, u64, u64>::with_limit(0).is_none());
        assert!(BindingCache::<u64, u64, u64>::with_limit(1024).is_none());
    }
    #[test]
    fn four_thousand_entries_reuse_exact_graph_keys_without_growth() {
        let mut cache = BindingCache::<u64, u64, u64>::with_limit(4096).unwrap();
        // More entries than one current whole-model token's 2854 dispatches.
        let budget = 4 * 1024 * 1024;
        for n in 0..4096u64 {
            let buffers = [n * 3, n * 3 + 1, n * 3 + 2];
            assert!(cache.insert(&7, buffers.iter().map(|b| (b, 0, 64)), n, budget));
        }
        let full = cache.host_bytes();
        for n in 0..4096u64 {
            let buffers = [n * 3, n * 3 + 1, n * 3 + 2];
            assert_eq!(cache.find(&7, buffers.iter().map(|b| (b, 0, 64))), Some(n));
            assert_eq!(cache.find(&7, buffers.iter().map(|b| (b, 4, 64))), None);
        }
        assert!(cache.insert(&8, std::iter::once((&0, 0, 64)), 0, budget));
        assert!(cache.host_bytes() <= full);
        assert_eq!(cache.find(&7, [0, 1, 2].iter().map(|b| (b, 0, 64))), None);
        assert_eq!(cache.find(&8, std::iter::once((&0, 0, 64))), Some(0));
    }
    #[test]
    fn exact_owner_pipeline_offset_size_and_order_are_required() {
        let mut cache = BindingCache::<u64, u64, u64>::new().unwrap();
        let bindings = [(11, 4, 64), (12, 8, 128)];
        let iter = || bindings.iter().map(|(b, o, s)| (b, *o, *s));
        assert!(cache.insert(&9, iter(), 100, usize::MAX));
        assert_eq!(cache.find(&9, iter()), Some(100));
        assert_eq!(cache.find(&10, iter()), None);
        for wrong in [
            [(13, 4, 64), (12, 8, 128)],
            [(11, 8, 64), (12, 8, 128)],
            [(11, 4, 68), (12, 8, 128)],
            [(12, 8, 128), (11, 4, 64)],
        ] {
            assert_eq!(
                cache.find(&9, wrong.iter().map(|(b, o, s)| (b, *o, *s))),
                None
            );
        }
        assert_eq!(
            cache.find(&9, bindings[..1].iter().map(|(b, o, s)| (b, *o, *s))),
            None
        );
        assert_eq!(
            cache.find(
                &9,
                [(11, 4, 0), (12, 8, 128)]
                    .iter()
                    .map(|(b, o, s)| (b, *o, *s))
            ),
            None
        );
    }
    #[test]
    fn capacity_falls_back_without_growth_or_eviction() {
        let mut cache = BindingCache::<u64, u64, u64>::new().unwrap();
        let size = cache.host_bytes();
        let b = 7;
        assert!(!cache.insert(&0, std::iter::once((&b, 0, 64)), 1, size));
        assert_eq!(cache.host_bytes(), size);
        for n in 0..ENTRIES as u64 {
            assert!(cache.insert(&n, std::iter::once((&b, 0, 64)), n, usize::MAX));
        }
        let size = cache.host_bytes();
        assert!(!cache.insert(&999, std::iter::once((&b, 0, 64)), 1, usize::MAX));
        assert_eq!(cache.host_bytes(), size);
        assert_eq!(cache.find(&0, std::iter::once((&b, 0, 64))), Some(0));
        assert!(cache.host_bytes() >= BindingCache::<u64, u64, u64>::reserved_bytes());
    }
    #[test]
    fn malformed_collisions_cannot_substitute_or_loop() {
        let mut cache = BindingCache::<u64, u64, u64>::new().unwrap();
        let b = 7;
        assert!(cache.insert(&1, std::iter::once((&b, 0, 64)), 11, usize::MAX));
        let hash = BindingCache::<u64, u64, u64>::hash(&2, std::iter::once((&b, 0, 64)));
        cache.entries[0].hash = hash;
        cache.entries[0].next = 1;
        cache.buckets[hash as usize & (BUCKETS - 1)] = 1;
        assert_eq!(cache.find(&2, std::iter::once((&b, 0, 64))), None);
        cache.buckets[hash as usize & (BUCKETS - 1)] = u32::MAX;
        assert_eq!(cache.find(&2, std::iter::once((&b, 0, 64))), None);
    }
    #[test]
    fn dropping_scope_releases_all_cached_references() {
        #[derive(Clone)]
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut cache = BindingCache::<u64, u64, Tracked>::new().unwrap();
        let b = 1;
        assert!(cache.insert(
            &1,
            std::iter::once((&b, 0, 64)),
            Tracked(dropped.clone()),
            usize::MAX
        ));
        let hit = cache.find(&1, std::iter::once((&b, 0, 64))).unwrap();
        drop(hit);
        drop(cache);
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
    }
    fn extended_with_collision_chain() -> (BindingCache<u64, u64, u64>, [u64; 4]) {
        let bucket = BindingCache::<u64, u64, u64>::hash(&1, std::iter::once((&0, 0, 64))) as usize
            & (BUCKETS - 1);
        let mut keys = [0; 4];
        let mut selected = 0;
        for key in 0..u64::MAX {
            let hash = BindingCache::<u64, u64, u64>::hash(&1, std::iter::once((&key, 0, 64)));
            if hash as usize & (BUCKETS - 1) == bucket {
                keys[selected] = key;
                selected += 1;
                if selected == keys.len() {
                    break;
                }
            }
        }
        let mut cache = BindingCache::with_limit(2048).unwrap();
        for (rank, key) in keys[..3].iter().enumerate() {
            assert!(cache.insert(&1, std::iter::once((key, 0, 64)), rank as u64, usize::MAX));
        }
        for key in 0..u64::MAX {
            let hash = BindingCache::<u64, u64, u64>::hash(&1, std::iter::once((&key, 0, 64)));
            if hash as usize & (BUCKETS - 1) != bucket {
                assert!(cache.insert(&1, std::iter::once((&key, 0, 64)), key + 10, usize::MAX));
                if cache.entries.len() == 2048 {
                    break;
                }
            }
        }
        (cache, keys)
    }

    #[test]
    fn extended_fifo_unlinks_head_middle_tail_and_relinks_same_or_different_bucket() {
        for victim in 0..3 {
            for same_bucket in [false, true] {
                let (mut cache, keys) = extended_with_collision_chain();
                cache.next_replacement = victim;
                let new_key = keys[3];
                let old_bucket = cache.entries[victim].hash as usize & (BUCKETS - 1);
                let new_pipeline = if same_bucket {
                    1
                } else {
                    (2..u64::MAX)
                        .find(|pipeline| {
                            BindingCache::<u64, u64, u64>::hash(
                                pipeline,
                                std::iter::once((&new_key, 0, 64)),
                            ) as usize
                                & (BUCKETS - 1)
                                != old_bucket
                        })
                        .unwrap()
                };
                let before = cache.host_bytes();
                assert!(cache.insert(
                    &new_pipeline,
                    std::iter::once((&new_key, 0, 64)),
                    99,
                    usize::MAX
                ));
                assert_eq!(cache.entries.len(), 2048);
                assert_eq!(cache.entries.capacity(), 2048);
                assert_eq!(cache.host_bytes(), before);
                assert_eq!(cache.next_replacement, victim + 1);
                for (rank, key) in keys[..3].iter().enumerate() {
                    assert_eq!(
                        cache.find(&1, std::iter::once((key, 0, 64))),
                        (rank != victim).then_some(rank as u64)
                    );
                }
                assert_eq!(
                    cache.find(&new_pipeline, std::iter::once((&new_key, 0, 64))),
                    Some(99)
                );
                assert_eq!(
                    cache.find(&new_pipeline, std::iter::once((&new_key, 4, 64))),
                    None
                );
                // All unaffected noncolliding bindings must still be reachable.
                for entry in &cache.entries[3..] {
                    assert_eq!(
                        cache.find(
                            &entry.pipeline,
                            entry.bindings.iter().map(|b| (&b.buffer, b.offset, b.size))
                        ),
                        Some(entry.group)
                    );
                }
            }
        }
    }

    #[test]
    fn extended_fifo_pressure_and_malformed_chains_fail_without_mutation() {
        let (mut cache, keys) = extended_with_collision_chain();
        let old = cache.find(&1, std::iter::once((&keys[0], 0, 64)));
        let entries = cache.entries.len();
        let bytes = cache.host_bytes();
        let buckets = cache.buckets;
        let minimum_peak = bytes
            + std::mem::size_of::<Binding<u64>>()
            + 2 * std::mem::size_of::<Entry<u64, u64, u64>>();
        assert!(!cache.insert(&1, std::iter::once((&keys[3], 0, 64)), 99, minimum_peak - 1));
        assert_eq!(cache.buckets, buckets);
        assert_eq!(cache.entries.len(), entries);
        assert_eq!(cache.host_bytes(), bytes);
        assert_eq!(cache.next_replacement, 0);
        assert_eq!(cache.find(&1, std::iter::once((&keys[0], 0, 64))), old);
        assert!(cache.find(&1, std::iter::once((&keys[3], 0, 64))).is_none());
        let bucket = cache.entries[0].hash as usize & (BUCKETS - 1);
        for corrupt in [u32::MAX, 1] {
            cache.buckets = buckets;
            let old_next = cache.entries[0].next;
            cache.entries[0].next = corrupt;
            let broken = cache.buckets;
            assert!(!cache.insert(&1, std::iter::once((&keys[3], 0, 64)), 99, usize::MAX));
            assert_eq!(cache.buckets, broken);
            assert_eq!(cache.entries[0].next, corrupt);
            assert_eq!(cache.host_bytes(), bytes);
            assert_eq!(cache.next_replacement, 0);
            cache.entries[0].next = old_next;
        }
        cache.buckets[bucket] = u32::MAX;
        let broken = cache.buckets;
        assert!(!cache.insert(&1, std::iter::once((&keys[3], 0, 64)), 99, usize::MAX));
        assert_eq!(cache.buckets, broken);
        assert_eq!(cache.host_bytes(), bytes);
    }

    #[test]
    fn extended_byte_credit_saturation_before_entry_capacity_remains_bounded() {
        let mut cache = BindingCache::<u64, u64, u64>::with_limit(4096).unwrap();
        let empty = cache.host_bytes();
        let key_bytes = MAX_BINDINGS * std::mem::size_of::<Binding<u64>>();
        let budget = empty + 100 * key_bytes;
        for pipeline in 0..100 {
            assert!(cache.insert(
                &pipeline,
                std::iter::repeat_n((&1, 0, 64), MAX_BINDINGS),
                pipeline,
                budget
            ));
        }
        assert_eq!(cache.entries.len(), 100);
        assert_eq!(cache.binding_bytes, 100 * key_bytes);
        assert_eq!(cache.host_bytes(), budget);
        assert_eq!(cache.next_replacement, 0);
        let buckets = cache.buckets;
        assert!(!cache.insert(
            &100,
            std::iter::repeat_n((&1, 0, 64), MAX_BINDINGS),
            100,
            budget
        ));
        assert_eq!(cache.entries.len(), 100);
        assert_eq!(cache.binding_bytes, 100 * key_bytes);
        assert_eq!(cache.buckets, buckets);
        assert_eq!(cache.next_replacement, 0);
        assert_eq!(
            cache.find(&0, std::iter::repeat_n((&1, 0, 64), MAX_BINDINGS)),
            Some(0)
        );
        assert_eq!(
            cache.find(&100, std::iter::repeat_n((&1, 0, 64), MAX_BINDINGS)),
            None
        );
    }
    #[test]
    fn extended_fifo_wraps_and_releases_the_retired_group_without_stale_hits() {
        #[derive(Clone)]
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut cache = BindingCache::<u64, u64, Tracked>::with_limit(2048).unwrap();
        for key in 0..2048 {
            assert!(cache.insert(
                &1,
                std::iter::once((&key, 0, 64)),
                Tracked(dropped.clone()),
                usize::MAX
            ));
        }
        cache.next_replacement = 2047;
        assert!(cache.insert(
            &1,
            std::iter::once((&2048, 0, 64)),
            Tracked(dropped.clone()),
            usize::MAX
        ));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(cache.next_replacement, 0);
        assert!(cache.find(&1, std::iter::once((&2047, 0, 64))).is_none());
        assert!(cache.find(&1, std::iter::once((&2048, 0, 64))).is_some());
        assert_eq!(dropped.load(Ordering::Relaxed), 2); // returned clone dropped
        drop(cache);
        assert_eq!(dropped.load(Ordering::Relaxed), 2050);
    }

    #[test]
    fn replacement_wipes_retired_full_binding_capacity_before_its_allocation_is_freed() {
        use std::sync::atomic::AtomicBool;
        struct Witness {
            pointer: usize,
            bytes: usize,
            result: Arc<AtomicBool>,
        }
        struct Pipeline {
            id: u64,
            witness: Option<Witness>,
        }
        impl Clone for Pipeline {
            fn clone(&self) -> Self {
                Self {
                    id: self.id,
                    witness: None,
                }
            }
        }
        impl PartialEq for Pipeline {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }
        impl Eq for Pipeline {}
        impl Hash for Pipeline {
            fn hash<H: Hasher>(&self, hasher: &mut H) {
                self.id.hash(hasher);
            }
        }
        impl Drop for Pipeline {
            fn drop(&mut self) {
                if let Some(witness) = &self.witness {
                    // Entry::drop already wiped bindings. Rust drops pipeline
                    // BEFORE the following bindings field frees that allocation.
                    let zero = (0..witness.bytes).all(|n| unsafe {
                        (witness.pointer as *const u8).add(n).read_volatile() == 0
                    });
                    witness.result.store(zero, Ordering::Relaxed);
                }
            }
        }
        let result = Arc::new(AtomicBool::new(false));
        let mut cache = BindingCache::<Pipeline, u64, u64>::with_limit(2048).unwrap();
        for key in 0..2048 {
            assert!(cache.insert(
                &Pipeline {
                    id: key,
                    witness: None
                },
                std::iter::once((&key, 0, 64)),
                key,
                usize::MAX
            ));
        }
        let bindings = &cache.entries[0].bindings;
        let witness = Witness {
            pointer: bindings.as_ptr() as usize,
            bytes: bindings.capacity() * std::mem::size_of::<Binding<u64>>(),
            result: result.clone(),
        };
        cache.entries[0].pipeline.witness = Some(witness);
        assert!(cache.insert(
            &Pipeline {
                id: 2048,
                witness: None
            },
            std::iter::once((&2048, 0, 64)),
            2048,
            usize::MAX
        ));
        assert!(result.load(Ordering::Relaxed));
    }
    #[test]
    fn oversized_and_empty_binding_lists_do_not_allocate_cache_keys() {
        let mut cache = BindingCache::<u64, u64, u64>::new().unwrap();
        let b = 1;
        let size = cache.host_bytes();
        assert!(!cache.insert(&1, std::iter::empty(), 1, usize::MAX));
        assert!(!cache.insert(
            &1,
            std::iter::repeat_n((&b, 0, 4), MAX_BINDINGS + 1),
            1,
            usize::MAX
        ));
        assert_eq!(cache.host_bytes(), size);
    }
}
