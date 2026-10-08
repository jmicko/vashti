// Bounded exact binding-object reuse for one exclusive private request.
use std::hash::{Hash, Hasher};

const ENTRIES: usize = 512;
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
}
impl<P, B, G> BindingCache<P, B, G> {
    pub fn reserved_bytes() -> usize {
        std::mem::size_of::<Self>() + ENTRIES * std::mem::size_of::<Entry<P, B, G>>()
    }
    #[cfg(test)]
    pub fn new() -> Option<Self> {
        Self::with_limit(ENTRIES)
    }
    pub fn with_limit(limit: usize) -> Option<Self> {
        if ![256, 512].contains(&limit) {
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
        if self.entries.len() == self.entries.capacity() || count == 0 || count > MAX_BINDINGS {
            return false;
        }
        let Some(bytes) = count.checked_mul(std::mem::size_of::<Binding<B>>()) else {
            return false;
        };
        if self
            .host_bytes()
            .checked_add(bytes)
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
            .is_none_or(|n| n > max_bytes)
        {
            return false;
        }
        let hash = Self::hash(pipeline, bindings.clone());
        for (buffer, offset, size) in bindings {
            owned.push(Binding {
                buffer: buffer.clone(),
                offset,
                size,
            });
        }
        let bucket = hash as usize & (BUCKETS - 1);
        self.entries.push(Entry {
            pipeline: pipeline.clone(),
            bindings: owned,
            group,
            hash,
            next: self.buckets[bucket],
        });
        self.binding_bytes += actual;
        self.buckets[bucket] = self.entries.len() as u32;
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
        for limit in [256, 512] {
            let mut cache = BindingCache::<u64, u64, u64>::with_limit(limit).unwrap();
            assert_eq!(cache.entries.capacity(), limit);
            for n in 0..limit {
                assert!(cache.insert(
                    &1,
                    std::iter::once(&(n as u64)).map(|b| (b, 0, 64)),
                    2,
                    usize::MAX
                ));
            }
            assert!(!cache.insert(&1, std::iter::once((&9999, 0, 64)), 2, usize::MAX));
        }
        assert!(BindingCache::<u64, u64, u64>::with_limit(0).is_none());
        assert!(BindingCache::<u64, u64, u64>::with_limit(1024).is_none());
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
