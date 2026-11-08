// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use alloc::{vec, vec::Vec};

use cubecl_ir::{AddressType, StorageType};
use cubecl_runtime::{kernel::ScalarKernelArg, server::MetadataBindingInfo};

use crate::{Metadata, MetadataBuilder, ScalarBuilder};

pub(crate) const INFO_ALIGN: usize = size_of::<u64>();

/// Helper to calculate info struct fields
#[derive(Clone, Debug, Default)]
pub struct Info {
    pub scalars: Vec<SizedInfoField>,
    pub sized_meta: Option<SizedInfoField>,
    pub has_dynamic_meta: bool,
    pub dynamic_meta_offset: usize,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug)]
pub struct SizedInfoField {
    pub ty: StorageType,
    pub size: usize,
    pub offset: usize,
}

impl SizedInfoField {
    pub fn padded_size(&self) -> usize {
        let padding_factor = INFO_ALIGN / self.ty.size();
        self.size.next_multiple_of(padding_factor)
    }
}

impl Info {
    pub fn new(scalars: &[ScalarKernelArg], metadata: Metadata, address_type: StorageType) -> Self {
        let mut scalar_fields = Vec::with_capacity(scalars.len());
        let mut sized_meta = None;

        let mut offset = 0;

        for scalar in scalars {
            scalar_fields.push(SizedInfoField {
                ty: scalar.ty,
                size: scalar.count,
                offset,
            });
            offset += (scalar.ty.size() * scalar.count).next_multiple_of(INFO_ALIGN);
        }

        if metadata.static_len() > 0 {
            let size = metadata.static_len() as usize;
            sized_meta = Some(SizedInfoField {
                ty: address_type,
                size,
                offset,
            });
            offset += (address_type.size() * size).next_multiple_of(INFO_ALIGN);
        }

        Info {
            scalars: scalar_fields,
            sized_meta,
            has_dynamic_meta: metadata.num_extended_meta() > 0,
            dynamic_meta_offset: offset,
            metadata,
        }
    }

    pub fn has_info(&self) -> bool {
        !self.scalars.is_empty() || self.sized_meta.is_some()
    }
}

#[derive(Default)]
pub struct InfoBuilder {
    pub scalars: ScalarBuilder,
    pub metadata: MetadataBuilder,
}

impl InfoBuilder {
    pub fn finish(&mut self, address_type: AddressType) -> MetadataBindingInfo {
        struct FinishGuard<'a> {
            info: &'a mut InfoBuilder,
            out: MetadataBindingInfo,
        }
        impl Drop for FinishGuard<'_> {
            fn drop(&mut self) {
                self.info.clear_private();
                self.out.wipe_private();
                #[cfg(all(test, feature = "std"))]
                PACK_WIPED.with(|cell| cell.set(self.out.data.iter().all(|v| *v == 0)));
            }
        }
        // Guard exists BEFORE size arithmetic/allocation/copies can panic.
        // MetadataBindingInfo Drop independently wipes the packed output Vec.
        let mut guard = FinishGuard {
            info: self,
            out: MetadataBindingInfo::default(),
        };
        let addr_packing = INFO_ALIGN / address_type.size();

        let scalars_size = guard.info.scalars.len_aligned();
        let static_len = guard.info.metadata.static_len(address_type);
        let static_size = static_len.div_ceil(addr_packing);
        let dynamic_len = guard.info.metadata.dynamic_len(address_type);
        let dynamic_size = dynamic_len.div_ceil(addr_packing);

        guard.out.data = vec![
            0;
            scalars_size
                .checked_add(static_size)
                .and_then(|n| n.checked_add(dynamic_size))
                .expect("private metadata size overflow")
        ];
        guard
            .info
            .scalars
            .finish(&mut guard.out.data[..scalars_size]);
        #[cfg(all(test, feature = "std"))]
        if PACK_PANIC.with(|cell| cell.replace(false)) {
            panic!("private metadata synthetic pack failure");
        }
        guard.info.metadata.finish(
            address_type,
            guard.out.data[scalars_size..].split_at_mut(static_size),
        );
        guard.out.dynamic_metadata_offset = scalars_size + static_size;
        core::mem::take(&mut guard.out)
    }
}

impl InfoBuilder {
    /// Wipe all request-derived TLS scalar/metadata capacities before reuse.
    pub fn clear_private(&mut self) {
        self.scalars.clear_private();
        self.metadata.clear_private();
    }
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
    static PACK_PANIC:core::cell::Cell<bool>=const {core::cell::Cell::new(false)};
    static PACK_WIPED:core::cell::Cell<bool>=const {core::cell::Cell::new(false)};
}
#[cfg(all(test, feature = "std"))]
mod private_tests {
    use super::*;
    #[test]
    fn packed_temporary_and_retained_builder_wipe_during_partial_pack_unwind() {
        let mut builder = InfoBuilder::default();
        builder.scalars.push(0x23456789u32);
        builder.metadata.register_array(777, 512, AddressType::U32);
        PACK_PANIC.with(|c| c.set(true));
        PACK_WIPED.with(|c| c.set(false));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || builder.finish(AddressType::U32)
            ))
            .is_err()
        );
        assert_eq!(builder.scalars.len_aligned(), 0);
        assert_eq!(builder.metadata.static_len(AddressType::U32), 0);
        assert_eq!(builder.metadata.static_len(AddressType::U64), 0);
        assert!(PACK_WIPED.with(|c| c.get()));
        builder.scalars.push(37u32);
        builder.metadata.register_array(777, 512, AddressType::U32);
        let mut packed = builder.finish(AddressType::U32);
        assert_eq!(packed.data, [37, 777 | (512u64 << 32)]);
        let debug = format!("{packed:?}");
        assert!(!debug.contains("777"));
        assert!(!debug.contains("2199023256329"));
        packed.wipe_private();
        // SAFETY: wipe_private initialized every capacity word.
        assert!(
            unsafe { core::slice::from_raw_parts(packed.data.as_ptr(), packed.data.capacity()) }
                .iter()
                .all(|v| *v == 0)
        );
    }
}
