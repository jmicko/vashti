// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use alloc::vec::Vec;

use cubecl_ir::StorageType;

use crate::{INFO_ALIGN, ScalarArgType};

/// Stores the data and type for a scalar arg
pub type ScalarValues = Vec<u8>;

#[derive(Default)]
pub struct ScalarBuilder {
    /// Sorted list of scalars, should be faster than `BTreeMap` for this purpose. Benchmark later.
    scalars: Vec<(StorageType, ScalarValues)>,
}

impl ScalarBuilder {
    /// Add a new scalar value to the state.
    pub fn push<T: ScalarArgType>(&mut self, val: T) {
        let val = [val];
        let bytes = T::as_bytes(&val);
        crate::private::extend(self.get_or_insert_mut(T::cube_type()), bytes);
    }

    /// Add a new raw value to the state.
    pub fn push_raw(&mut self, bytes: &[u8], dtype: StorageType) {
        crate::private::extend(self.get_or_insert_mut(dtype), bytes);
    }

    fn get_or_insert_mut(&mut self, ty: StorageType) -> &mut ScalarValues {
        let pos = self.scalars.iter().position(|(k, _)| *k >= ty);

        match pos {
            Some(i) if self.scalars[i].0 == ty => &mut self.scalars[i].1,
            Some(i) => {
                self.scalars.insert(i, (ty, Vec::new()));
                &mut self.scalars[i].1
            }
            None => {
                self.scalars.push((ty, Vec::new()));
                &mut self.scalars.last_mut().unwrap().1
            }
        }
    }

    pub fn len_aligned(&self) -> usize {
        self.scalars
            .iter()
            .map(|(_, v)| v.len().div_ceil(INFO_ALIGN))
            .sum()
    }

    /// Wipe all retained scalar capacities, including abandoned registrations.
    pub fn clear_private(&mut self) {
        for (_, values) in &mut self.scalars {
            crate::private::wipe_vec(values);
        }
        self.scalars.clear();
        crate::private::wipe_empty(&mut self.scalars);
    }

    pub fn finish(&mut self, out: &mut [u64]) {
        struct Clear<'a>(&'a mut ScalarBuilder);
        impl Drop for Clear<'_> {
            fn drop(&mut self) {
                self.0.clear_private();
            }
        }
        let clear = Clear(self);
        let mut out_u8 = bytemuck::cast_slice_mut::<u64, u8>(out);

        for (_, values) in clear.0.scalars.iter_mut().filter(|(_, v)| !v.is_empty()) {
            let len_padded = values.len().next_multiple_of(INFO_ALIGN);

            out_u8[0..values.len()].copy_from_slice(values);
            out_u8 = &mut out_u8[len_padded..];
            crate::private::wipe_vec(values);
        }
    }
}

impl Drop for ScalarBuilder {
    fn drop(&mut self) {
        self.clear_private();
    }
}

#[cfg(test)]
mod private_tests {
    use super::*;
    fn outer_capacity_zero(builder: &ScalarBuilder) -> bool {
        // SAFETY: clear_private/finish initialized every inactive raw byte.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                builder.scalars.as_ptr().cast::<u8>(),
                builder.scalars.capacity() * core::mem::size_of::<(StorageType, ScalarValues)>(),
            )
        };
        bytes.iter().all(|b| *b == 0)
    }
    #[test]
    fn scalar_pack_wipes_tls_capacity_after_success_and_panic_with_retained_builder() {
        let mut b = ScalarBuilder::default();
        b.push(0x23456789u32);
        let mut out = [0u64; 1];
        b.finish(&mut out);
        assert_eq!(out[0], 0x23456789);
        assert!(b.scalars.is_empty());
        assert!(outer_capacity_zero(&b));
        b.push(0x76543210u32);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.finish(&mut []))).is_err()
        );
        assert!(b.scalars.is_empty());
        assert!(outer_capacity_zero(&b));
        b.push(37u32);
        b.finish(&mut out);
        assert_eq!(out[0], 37);
    }
}
