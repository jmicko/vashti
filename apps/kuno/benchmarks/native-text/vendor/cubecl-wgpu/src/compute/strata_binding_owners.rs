//! Original allocator bindings retained by their actual native recording.
//! Deduplication uses allocation identity only; it grants no access/completion.
use cubecl_runtime::server::Binding;
use std::{
    mem::{align_of, size_of},
    sync::Arc,
};

pub(super) type Bundle = Arc<Vec<Binding>>;

pub(super) struct Owners<T = Binding> {
    bundles: Vec<Arc<Vec<T>>>,
    bytes: usize,
}

impl<T> Owners<T> {
    pub(super) fn collect<'a>(
        bundles: impl Iterator<Item = &'a Arc<Vec<T>>>,
        budget: usize,
    ) -> Result<Self, &'static str>
    where
        T: 'a,
    {
        let mut retained = Vec::new();
        for bundle in bundles {
            // Same bound as the original Graph node inventory. Reserve/charge
            // the real pointer vector even when most entries later deduplicate.
            if retained.len() == 32768 {
                return Err("native binding owner node capacity");
            }
            if bundle.is_empty() {
                return Err("native binding owner empty bundle");
            }
            retained.push(bundle.clone());
        }
        retained.sort_unstable_by_key(|bundle| Arc::as_ptr(bundle) as usize);
        retained.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let mut bytes = retained
            .capacity()
            .checked_mul(size_of::<Arc<Vec<T>>>())
            .ok_or("native binding owner metadata overflow")?;
        for bundle in &retained {
            // ArcInner has two counters followed by the Vec header/data.
            // Include worst alignment padding rather than counting Arc handles
            // alone. Original allocator descriptors and GPU spans keep their
            // existing memory-manager reservations, independently of this Vec.
            let own = bundle
                .capacity()
                .checked_mul(size_of::<T>())
                .and_then(|n| n.checked_add(size_of::<Vec<T>>()))
                .and_then(|n| n.checked_add(2 * size_of::<usize>()))
                .and_then(|n| n.checked_add(align_of::<Vec<T>>() - 1))
                .ok_or("native binding owner metadata overflow")?;
            bytes = bytes
                .checked_add(own)
                .ok_or("native binding owner metadata overflow")?;
        }
        if bytes > budget {
            return Err("native binding owner metadata capacity");
        }
        Ok(Self {
            bundles: retained,
            bytes,
        })
    }

    pub(super) fn metadata_bytes(&self) -> usize {
        self.bytes
    }
}

impl<T> Drop for Owners<T> {
    fn drop(&mut self) {
        // Fields drop normally only when their original Recording does. If
        // completion is unknown, Graph quarantines that whole Recording.
        self.bundles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Probe(Arc<AtomicUsize>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn split_original_recordings_each_keep_the_same_allocator_owner_alive() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(vec![Probe(dropped.clone())]);
        let first = Owners::collect([&source, &source].into_iter(), 65536).unwrap();
        let second = Owners::collect([&source].into_iter(), 65536).unwrap();
        assert_eq!(first.bundles.len(), 1);
        assert!(Arc::ptr_eq(&first.bundles[0], &second.bundles[0]));
        drop(source);
        drop(first);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        drop(second);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn actual_capacity_and_distinct_equal_allocations_are_charged() {
        let mut values = Vec::with_capacity(4096);
        values.push(7u64);
        let first = Arc::new(values);
        let second = Arc::new(vec![7u64]);
        let owners = Owners::collect([&first, &first, &second].into_iter(), 65536).unwrap();
        assert_eq!(owners.bundles.len(), 2);
        assert!(owners.metadata_bytes() >= 4097 * size_of::<u64>());
        assert!(Owners::collect([&first].into_iter(), 4096).is_err());
        assert!(Owners::<u64>::collect([&Arc::new(Vec::new())].into_iter(), 65536).is_err());
    }
}
