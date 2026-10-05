// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
//! Full-capacity wiping for the laboratory's scalar/INFO host metadata.
//! POD capacity can be initialized with zero even beyond logical length. Grow
//! by moving through a guard, wiping the previous allocation BEFORE release;
//! ordinary Vec::reserve/realloc could release a private old allocation intact.
use alloc::vec::Vec;
use bytemuck::Pod;
use core::sync::atomic::{Ordering, compiler_fence};

pub(crate) fn wipe_vec<T: Pod>(values: &mut Vec<T>) {
    let ptr = values.as_mut_ptr();
    for i in 0..values.capacity() {
        // SAFETY: POD zero is valid; entire allocation writable and exclusive.
        unsafe { ptr.add(i).write_volatile(T::zeroed()) };
    }
    compiler_fence(Ordering::SeqCst);
    values.clear();
}
struct Guard<T: Pod>(Vec<T>);
impl<T: Pod> Drop for Guard<T> {
    fn drop(&mut self) {
        wipe_vec(&mut self.0);
    }
}

pub(crate) fn extend<T: Pod>(values: &mut Vec<T>, data: &[T]) {
    let len = values
        .len()
        .checked_add(data.len())
        .expect("private metadata length overflow");
    if len > values.capacity() {
        let mut next = Guard(Vec::new());
        next.0
            .try_reserve_exact(len)
            .expect("private metadata allocation failed");
        next.0.extend_from_slice(values);
        next.0.extend_from_slice(data);
        wipe_vec(values);
        *values = core::mem::take(&mut next.0);
    } else {
        values.extend_from_slice(data);
    }
}
pub(crate) fn push<T: Pod>(values: &mut Vec<T>, value: T) {
    extend(values, &[value]);
}

/// Wipe a Vec's entire raw capacity only AFTER all active non-POD entries have
/// been dropped or moved out. Zeroed inactive bytes never become live values.
pub(crate) fn wipe_empty<T>(values: &mut Vec<T>) {
    assert!(values.is_empty(), "private metadata container still live");
    let ptr = values.as_mut_ptr().cast::<u8>();
    let bytes = values
        .capacity()
        .checked_mul(core::mem::size_of::<T>())
        .expect("private metadata span overflow");
    for i in 0..bytes {
        // SAFETY: len0 means no active T needs representation/drop; writable
        // raw bytes inside exclusively owned allocation, including spare space.
        unsafe { ptr.add(i).write_volatile(0) };
    }
    compiler_fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wipes_all_pod_capacity_and_guarded_growth_preserves_only_live_values() {
        let mut v = Vec::with_capacity(63);
        v.resize(v.capacity(), 0x69u32);
        v.truncate(3);
        wipe_vec(&mut v);
        // SAFETY: wipe initialized every POD capacity slot above.
        assert!(
            unsafe { core::slice::from_raw_parts(v.as_ptr(), v.capacity()) }
                .iter()
                .all(|v| *v == 0)
        );
        extend(&mut v, &[7, 9]);
        let cap = v.capacity();
        extend(&mut v, &vec![13; cap + 1]);
        assert_eq!(&v[..2], &[7, 9]);
        assert!(v[2..].iter().all(|v| *v == 13));
        wipe_vec(&mut v);
    }
}
