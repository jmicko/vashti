// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
//! Laboratory-only private upload ownership and transient-I/O receipt.
//!
//! No file-backed input is accepted by the private client entry point. It moves
//! an owned Vec into this controller without cloning; Drop wipes the ENTIRE
//! allocation capacity, including stale bytes beyond the logical length.
//! Callers retain responsibility for their own earlier copies and for preventing
//! swap/core dumps. Bytes cloning/growing outside this private path is not a
//! supported secure operation. A receipt covers explicit API buffer spans only,
//! never opaque driver allocations, historical public writes, or main storage.

use alloc::{boxed::Box, vec::Vec};
use core::mem::MaybeUninit;
use core::sync::atomic::{Ordering, compiler_fence};
use cubecl_common::bytes::{AllocationController, AllocationProperty, Bytes};

/// Checked sanitizer completion for the explicitly owned transient I/O buffers.
/// This is not a production privacy certificate or device/driver memory proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransientIoReceipt {
    /// Successfully cleaned stream-local epoch.
    pub epoch: u64,
    /// Full owned upload-stage bytes erased before release.
    pub upload_bytes: u64,
    /// Full owned uniform bytes erased before release.
    pub uniform_bytes: u64,
    /// Full owned readback bytes erased before release.
    pub readback_bytes: u64,
    /// API-visible bytes checked as zero, including audit staging, when opted in.
    /// Zero in ordinary sanitize mode; contents are NEVER returned.
    pub inspected_bytes: u64,
    /// Full API-visible buffer spans checked in opted-in device audit mode.
    pub inspected_buffers: u64,
}

/// Heap-only controller that preserves wiping on success/error/cancellation.
/// Exported for the Wgpu readback controller; the Vec is never copied here.
pub struct PrivateHostAllocation {
    data: Vec<u8>,
}
impl PrivateHostAllocation {
    /// Move an existing initialized allocation into private ownership.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }
    /// Allocate initialized logical bytes without a temporary request copy.
    pub fn zeroed(len: usize) -> Result<Self, ()> {
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| ())?;
        data.resize(len, 0);
        Ok(Self { data })
    }
    /// Logical initialized bytes, used only while filling a private readback.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
    /// Logical initialized length.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether the initialized logical region is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    fn wipe(&mut self) {
        let pointer = self.data.as_mut_ptr();
        for i in 0..self.data.capacity() {
            // SAFETY: all bytes of this exclusively owned allocation are
            // writable, including uninitialized spare capacity. Zero is valid.
            unsafe { pointer.add(i).write_volatile(0) };
        }
        compiler_fence(Ordering::SeqCst);
    }
}
impl Drop for PrivateHostAllocation {
    fn drop(&mut self) {
        self.wipe();
    }
}
impl AllocationController for PrivateHostAllocation {
    fn alloc_align(&self) -> usize {
        1
    }
    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }
    unsafe fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: MaybeUninit<u8> has u8 layout; all logical bytes initialized.
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr().cast(), self.data.len()) }
    }
    fn memory(&self) -> &[MaybeUninit<u8>] {
        // SAFETY: same representation and initialized logical byte range.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr().cast(), self.data.len()) }
    }
}

pub(crate) fn upload_bytes(data: Vec<u8>) -> Bytes {
    let len = data.len();
    // SAFETY: the moved Vec initializes exactly len bytes and owns its lifetime.
    unsafe { Bytes::from_controller(Box::new(PrivateHostAllocation::from_vec(data)), len) }
}

/// Shared validation used by the real client and host-only ownership fixtures.
/// Contiguous allocation records explicit zero offsets even for a full span.
/// Accept those and absent offsets equally; reject every nonzero trim before
/// calling any ordinary size helper that could underflow on a forged offset.
/// Only a retained FULL destination is valid.
pub(crate) fn private_write_descriptor(
    handle: crate::server::Handle,
    stream: cubecl_common::stream_id::StreamId,
    len: usize,
) -> Result<crate::server::CopyDescriptor, crate::server::ServerError> {
    if len == 0
        || handle.stream != stream
        || handle.offset_start.unwrap_or(0) != 0
        || handle.offset_end.unwrap_or(0) != 0
        || u64::try_from(len).ok() != Some(handle.size())
    {
        return Err(crate::server::ServerError::Generic {
            reason: "private write requires full owned same-stream span".into(),
            backtrace: cubecl_common::backtrace::BackTrace::capture(),
        });
    }
    Ok(handle.copy_descriptor([len].into(), [1].into(), 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_destination_validation_preserves_original_owner_on_failure() {
        use crate::server::Handle;
        use cubecl_common::stream_id::StreamId;
        let stream = StreamId { value: 23 };
        let owner = Handle::new(stream, 32768);
        let memory = owner.memory.clone();
        for len in [0, 32767, 32769] {
            assert!(private_write_descriptor(owner.clone(), stream, len).is_err());
            assert_eq!(owner.size(), 32768);
            assert!(!memory.is_free());
        }
        assert!(private_write_descriptor(owner.clone(), StreamId { value: 24 }, 32768).is_err());
        let mut malformed = owner.clone();
        malformed.offset_start = Some(u64::MAX);
        assert!(private_write_descriptor(malformed, stream, 32768).is_err());
        let descriptor = private_write_descriptor(owner.clone(), stream, 32768).unwrap();
        assert_eq!(descriptor.shape.as_slice(), &[32768]);
        assert_eq!(&*descriptor.strides, &[1]);
        assert_eq!(descriptor.elem_size, 1);
        assert!(!memory.is_free());
    }
    #[test]
    fn private_destination_accepts_all_full_span_zero_offset_representations() {
        use crate::server::Handle;
        use cubecl_common::stream_id::StreamId;
        let stream = StreamId { value: 23 };
        let owner = Handle::new(stream, 32768);
        for start in [None, Some(0)] {
            for end in [None, Some(0)] {
                let mut full = owner.clone();
                full.offset_start = start;
                full.offset_end = end;
                let descriptor = private_write_descriptor(full, stream, 32768).unwrap();
                assert_eq!(descriptor.handle.offset_start, start);
                assert_eq!(descriptor.handle.offset_end, end);
                assert_eq!(descriptor.handle.size_in_used(), 32768);
                assert!(!owner.memory.is_free());
            }
        }
    }
    #[test]
    fn private_destination_accepts_actual_contiguous_single_full_allocation() {
        use crate::allocator::ContiguousMemoryLayoutPolicy;
        use crate::server::{MemoryLayoutDescriptor, MemoryLayoutPolicy};
        use cubecl_common::stream_id::StreamId;
        let stream = StreamId { value: 23 };
        let policy = ContiguousMemoryLayoutPolicy::new(256);
        // ComputeClient::empty uses this contiguous descriptor/apply path.
        let descriptor = MemoryLayoutDescriptor::contiguous([32768].into(), 1);
        let (owner, layouts) = policy.apply(stream, &[descriptor]);
        let full = layouts.into_iter().next().unwrap().memory;
        assert_eq!(full.offset_start, Some(0));
        assert_eq!(full.offset_end, Some(0));
        assert_eq!(full.size(), 32768);
        let copied = private_write_descriptor(full, stream, 32768).unwrap();
        assert_eq!(copied.handle.size_in_used(), owner.size());
        assert!(!owner.memory.is_free());
    }
    #[test]
    fn private_destination_rejects_actual_padding_and_coallocated_subspans() {
        use crate::allocator::ContiguousMemoryLayoutPolicy;
        use crate::server::{MemoryLayoutDescriptor, MemoryLayoutPolicy};
        use cubecl_common::stream_id::StreamId;
        let stream = StreamId { value: 23 };
        let policy = ContiguousMemoryLayoutPolicy::new(256);
        let padded = MemoryLayoutDescriptor::contiguous([257].into(), 1);
        let (owner, layouts) = policy.apply(stream, &[padded]);
        let memory = owner.memory.clone();
        let subspan = layouts.into_iter().next().unwrap().memory;
        assert_eq!(subspan.offset_start, Some(0));
        assert_eq!(subspan.offset_end, Some(255));
        assert!(private_write_descriptor(subspan.clone(), stream, 257).is_err());
        assert!(private_write_descriptor(subspan, stream, owner.size() as usize).is_err());
        assert!(!memory.is_free());

        let descriptors = [
            MemoryLayoutDescriptor::contiguous([256].into(), 1),
            MemoryLayoutDescriptor::contiguous([256].into(), 1),
        ];
        let (owner, layouts) = policy.apply(stream, &descriptors);
        let memory = owner.memory.clone();
        for subspan in layouts {
            assert!(private_write_descriptor(subspan.memory.clone(), stream, 256).is_err());
            assert!(
                private_write_descriptor(subspan.memory, stream, owner.size() as usize).is_err()
            );
            assert!(!memory.is_free());
        }
    }
    #[test]
    fn private_destination_rejects_positive_and_overflowing_trims_before_size_helpers() {
        use crate::server::Handle;
        use cubecl_common::stream_id::StreamId;
        let stream = StreamId { value: 23 };
        let owner = Handle::new(stream, 32768);
        let memory = owner.memory.clone();
        for (start, end) in [(1, 0), (0, 1), (32769, 0), (0, 32769), (u64::MAX, u64::MAX)] {
            let mut subspan = owner.clone();
            subspan.offset_start = Some(start);
            subspan.offset_end = Some(end);
            assert!(private_write_descriptor(subspan, stream, 32768).is_err());
            assert!(!memory.is_free());
        }
    }
    #[test]
    fn ownership_moves_the_original_pointer_without_clones() {
        let input = Vec::from([13u8, 17, 23]);
        let pointer = input.as_ptr();
        let bytes = upload_bytes(input);
        assert_eq!(bytes.as_ptr(), pointer);
        assert_eq!(&*bytes, &[13, 17, 23]);
    }
    #[test]
    fn wipes_every_capacity_byte_including_old_truncated_content() {
        let mut input = Vec::with_capacity(129);
        input.resize(input.capacity(), 0x95);
        input.truncate(3);
        let mut guard = PrivateHostAllocation::from_vec(input);
        let capacity = guard.data.capacity();
        guard.wipe();
        // SAFETY: wipe initialized the full allocation; inspect BEFORE release.
        let full = unsafe { core::slice::from_raw_parts(guard.data.as_ptr(), capacity) };
        assert!(full.iter().all(|v| *v == 0));
    }
    #[test]
    fn guarded_readback_mutation_preserves_wipe_ownership() {
        let mut guard = PrivateHostAllocation::zeroed(19).unwrap();
        guard.as_mut_slice().fill(0x67);
        let capacity = guard.data.capacity();
        guard.wipe();
        // SAFETY: full capacity initialized by wipe above.
        let full = unsafe { core::slice::from_raw_parts(guard.data.as_ptr(), capacity) };
        assert!(full.iter().all(|v| *v == 0));
    }
}
