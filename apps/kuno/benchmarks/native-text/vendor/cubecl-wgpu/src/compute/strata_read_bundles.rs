//! Batch-local sharing of identical ORDERED original binding-table indices.
//! No same-byte identity, cross-batch source cache or access/owner substitution.
use std::{mem::size_of, sync::Arc};
const MAX_VIEWS: usize = 4096;
const MAX_BUNDLES: usize = 16;
#[derive(Default)]
pub(super) struct Indices(Vec<usize>);
impl std::ops::Deref for Indices {
    type Target = Vec<usize>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Indices {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
type Key = Indices;
impl Indices {
    fn erase(&mut self) {
        self.0.clear();
        for i in 0..self.0.capacity() {
            unsafe { self.0.as_mut_ptr().add(i).write_volatile(0) }
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst)
    }
}
impl Drop for Indices {
    fn drop(&mut self) {
        self.erase()
    }
}
pub(super) struct Cache<T> {
    entries: Vec<(Key, Arc<Vec<T>>)>,
    views: usize,
}
pub(super) fn reservation<T>() -> usize {
    MAX_VIEWS * (2 * size_of::<T>() + 2 * size_of::<usize>())
        + MAX_BUNDLES
            * (size_of::<(Key, Arc<Vec<T>>)>() + size_of::<Vec<T>>() + 2 * size_of::<usize>() + 64)
        + 4096
}
/// EXACT production host-reservation formula, also used by the original
/// actual-INFO budget controls. No cap is enlarged by this helper.
pub(super) fn host_bytes<B, R, K, M, P>(tape: usize, commands: usize) -> Option<usize> {
    let tables = MAX_VIEWS
        * 2
        * (size_of::<B>() + size_of::<R>() + size_of::<K>() + size_of::<usize>() + 32);
    let per_command = 2
        * (size_of::<Option<P>>()
            + size_of::<M>()
            + size_of::<R>()
            + size_of::<Arc<Vec<R>>>()
            + size_of::<Vec<R>>()
            + 2 * size_of::<usize>()
            + 16);
    tape.checked_mul(2)?
        .checked_add(tables)?
        .checked_add(reservation::<R>())?
        .checked_add(commands.checked_mul(per_command)?)?
        .checked_add(32 * 512 + 32768)
}
impl<T> Cache<T> {
    pub fn new() -> Result<Self, &'static str> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(MAX_BUNDLES)
            .map_err(|_| "native source bundle cache allocation")?;
        if entries.capacity() > MAX_BUNDLES {
            return Err("native source bundle cache capacity");
        };
        Ok(Self { entries, views: 0 })
    }
    pub fn resolve(
        &mut self,
        indices: &[usize],
        mut get: impl FnMut(usize) -> Result<T, &'static str>,
    ) -> Result<Arc<Vec<T>>, &'static str> {
        if indices.len() > MAX_VIEWS {
            return Err("native source bundle view capacity");
        }
        if let Some((_, v)) = self.entries.iter().find(|(key, _)| key.0 == indices) {
            return Ok(v.clone());
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(indices.len())
            .map_err(|_| "native source bundle view allocation")?;
        if values.capacity() != indices.len() {
            return Err("native source bundle actual view capacity");
        }
        for &i in indices {
            values.push(get(i)?)
        }
        let result = Arc::new(values);
        if self.entries.len() < MAX_BUNDLES
            && self
                .views
                .checked_add(indices.len())
                .is_some_and(|n| n <= MAX_VIEWS)
        {
            let mut key = Indices(Vec::new());
            key.0
                .try_reserve_exact(indices.len())
                .map_err(|_| "native source bundle key allocation")?;
            if key.0.capacity() != indices.len() {
                return Err("native source bundle actual key capacity");
            }
            key.0.extend_from_slice(indices);
            self.views += indices.len();
            self.entries.push((key, result.clone()));
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    #[test]
    fn full_order_duplicates_and_distinct_original_indices_are_preserved() {
        let calls = Cell::new(0);
        let mut c = Cache::new().unwrap();
        let mut get = |i| {
            calls.set(calls.get() + 1);
            Ok(i % 2)
        };
        let a = c.resolve(&[4, 5, 4], &mut get).unwrap();
        let b = c.resolve(&[4, 5, 4], &mut get).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(&**a, &[0, 1, 0]);
        assert_eq!(calls.get(), 3);
        let other = c.resolve(&[6, 7, 6], &mut get).unwrap();
        assert_eq!(*a, *other);
        assert!(!Arc::ptr_eq(&a, &other));
        let reverse = c.resolve(&[4, 4, 5], get).unwrap();
        assert_eq!(&**reverse, &[0, 0, 1]);
    }
    #[test]
    fn cache_capacity_keeps_uncached_original_views_and_errors_cannot_publish_half_a_bundle() {
        let mut c = Cache::new().unwrap();
        let a = c.resolve(&[1], Ok::<_, &'static str>).unwrap();
        for i in 2..=16 {
            c.resolve(&[i], Ok::<_, &'static str>).unwrap();
        }
        let b = c.resolve(&[17, 18], Ok::<_, &'static str>).unwrap();
        let d = c.resolve(&[17, 18], Ok::<_, &'static str>).unwrap();
        assert_eq!(*b, *d);
        assert!(!Arc::ptr_eq(&b, &d));
        assert_eq!(c.entries.len(), 16);
        assert!(
            c.resolve(&[55, 99], |i| if i == 99 {
                Err("missing actual original binding")
            } else {
                Ok(i)
            })
            .is_err()
        );
        assert!(Arc::ptr_eq(
            &a,
            &c.resolve(&[1], Ok::<_, &'static str>).unwrap()
        ));
        drop(c);
        assert_eq!(&**a, &[1]);
        let mut other = Cache::new().unwrap();
        let fresh = other.resolve(&[1], Ok::<_, &'static str>).unwrap();
        assert!(!Arc::ptr_eq(&a, &fresh));
    }
    #[test]
    fn summed_view_bound_and_scratch_reservation_cover_actual_capacities() {
        let mut c = Cache::new().unwrap();
        let ids = vec![7; 4096];
        let a = c.resolve(&ids, Ok::<_, &'static str>).unwrap();
        assert_eq!(a.len(), 4096);
        assert!(Arc::ptr_eq(
            &a,
            &c.resolve(&ids, Ok::<_, &'static str>).unwrap()
        ));
        c.resolve(&[8], Ok::<_, &'static str>).unwrap();
        assert_eq!(c.entries.len(), 1);
        assert!(c.resolve(&vec![0; 4097], Ok::<_, &'static str>).is_err());
        assert!(reservation::<u64>() < 1024 * 1024);
        let mut k = Indices(Vec::with_capacity(128));
        unsafe { std::ptr::write_bytes(k.0.as_mut_ptr(), 0xff, k.0.capacity()) };
        k.0.extend([3, 4]);
        k.erase();
        assert!(
            unsafe { std::slice::from_raw_parts(k.0.as_ptr(), k.0.capacity()) }
                .iter()
                .all(|x| *x == 0)
        );
    }
}
