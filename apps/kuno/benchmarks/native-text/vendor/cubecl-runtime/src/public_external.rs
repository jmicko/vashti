//! Laboratory-only, immutable PUBLIC host sources outside ordinary allocators.
//!
//! A source's backing remains caller-owned. Registration is not allocation,
//! pinning, erasure, or release of that backing. Only a successful detach receipt
//! permits the caller to release it. This module never accepts private scratch.

use crate::{
    memory_management::{ManagedMemoryHandle, memory_pool::handle::MemoryLocation},
    server::{Binding, Handle},
};
use alloc::vec::Vec;
use core::mem::size_of;
use cubecl_common::stream_id::StreamId;

const EXTERNAL_TAG: u8 = u8::MAX;
/// Checked original public-host weight copy onto the same owning stream.
/// Both bindings retain full owners; offsets select bounded model-weight cells.
pub struct PublicHostCopy {
    /// Full registered immutable HOST owner.
    pub source: Binding,
    /// Original model-cell byte offset in that owner.
    pub source_first: u64,
    /// Full retained same-stream staging owner.
    pub destination: Binding,
    /// Ring-cell byte offset in that owner.
    pub destination_first: u64,
    /// Exact original model-cell bytes to copy.
    pub bytes: u64,
}
/// Check full owning bindings and bounded aligned copy views before dispatch.
pub fn validate_host_copies(
    copies: &[PublicHostCopy],
    stream: StreamId,
) -> Result<(), &'static str> {
    if !(1..=24).contains(&copies.len()) {
        return Err("public host copy inventory");
    }
    for c in copies {
        if !is_public_external_binding(&c.source)
            || is_public_external_binding(&c.destination)
            || c.source.stream != stream
            || c.destination.stream != stream
            || c.source.offset_start.unwrap_or(0) != 0
            || c.source.offset_end.unwrap_or(0) != 0
            || c.destination.offset_start.unwrap_or(0) != 0
            || c.destination.offset_end.unwrap_or(0) != 0
            || c.bytes == 0
            || !c.bytes.is_multiple_of(4)
            || !c.source_first.is_multiple_of(4)
            || !c.destination_first.is_multiple_of(4)
            || c.source_first
                .checked_add(c.bytes)
                .is_none_or(|e| e > c.source.size)
            || c.destination_first
                .checked_add(c.bytes)
                .is_none_or(|e| e > c.destination.size)
        {
            return Err("public host copy exact same-stream extents");
        }
    }
    Ok(())
}
/// Maximum full descriptor span in the initial public-host laboratory.
pub const MAX_BUFFER_BYTES: u64 = 128 * 1024 * 1024;
/// Maximum separately accounted public host backing across one stream.
pub const MAX_BACKING_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Maximum registrations; retired slots are never reused.
pub const MAX_BUFFERS: usize = 1024;
/// Maximum registry metadata reservation, including insertion temporaries.
pub const MAX_METADATA_BYTES: usize = 1024 * 1024;

/// Explicit PUBLIC-only host/metadata budget. Ordinary pools are unaffected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicExternalBudget {
    /// Full Vulkan allocations, not only shader descriptor bytes.
    pub backing_bytes: u64,
    /// Lifetime registrations; detach never grants a recycled slot.
    pub buffers: usize,
    /// Actual retained typed capacity plus bounded insertion temporaries.
    pub metadata_bytes: usize,
}
impl PublicExternalBudget {
    /// Validate the bounded laboratory envelope without allocation or mutation.
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.backing_bytes == 0
            || self.backing_bytes > MAX_BACKING_BYTES
            || self.buffers == 0
            || self.buffers > MAX_BUFFERS
            || self.metadata_bytes == 0
            || self.metadata_bytes > MAX_METADATA_BYTES
        {
            return Err("public external budget rejected");
        }
        Ok(self)
    }
}

/// Owned descriptor for an authenticated immutable PUBLIC host allocation.
/// Construction is unsafe because the backend cannot authenticate source bytes
/// or own/free the caller's external VkDeviceMemory.
pub struct PublicExternalHostSource<R> {
    resource: R,
    bytes: u64,
    backing_bytes: u64,
    source_sha256: [u8; 32],
}
impl<R> PublicExternalHostSource<R> {
    /// Declare one complete external PUBLIC allocation, before registration.
    ///
    /// # Safety
    /// The caller must authenticate `source_sha256` against the actual immutable
    /// public artifact, initialize the COMPLETE buffer/backing including gaps,
    /// and independently verify the supplied actual Vulkan allocation/heap flags.
    /// No prompt, image, activation, private key or other private bytes may enter.
    /// The resource and backing must belong to the selected target device. The
    /// caller must retain backing+device and prohibit CPU/raw GPU mutation until
    /// successful detach. This must be the allocation's SOLE external registry:
    /// no other server, raw/wgpu queue or alias may consume it while registered.
    /// All consumers must use this one owning stream through the checked backend.
    /// On failed detach/channel/queue join it MUST retain them
    /// (including on Drop); a failed operation grants no free/recovery authority.
    pub unsafe fn from_initialized_public_host(
        resource: R,
        bytes: u64,
        backing_bytes: u64,
        memory_property_flags: u32,
        memory_heap_flags: u32,
        source_sha256: [u8; 32],
    ) -> Result<Self, &'static str> {
        // Vulkan DEVICE_LOCAL=1, HOST_VISIBLE=2, LAZY=16, PROTECTED=32;
        // heap DEVICE_LOCAL=1. Flags here are caller attestations, not discovery.
        if bytes < 32
            || bytes > MAX_BUFFER_BYTES
            || !bytes.is_multiple_of(4)
            || backing_bytes < bytes
            || backing_bytes > MAX_BACKING_BYTES
            || memory_property_flags & 2 == 0
            || memory_property_flags & (1 | 16 | 32) != 0
            || memory_heap_flags & 1 != 0
            || source_sha256 == [0; 32]
        {
            return Err("public external immutable host source rejected");
        }
        Ok(Self {
            resource,
            bytes,
            backing_bytes,
            source_sha256,
        })
    }
    /// Borrow only for backend identity/full-span validation before registration.
    pub fn resource(&self) -> &R {
        &self.resource
    }
    /// Full shader buffer descriptor extent, excluding physical allocation tail.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// One consuming detach authority. Ordinary Handle clones do not grant detach.
pub struct PublicExternalHandle {
    handle: Handle,
}
impl PublicExternalHandle {
    /// Read-only kernel binding view. Backends must reject writes and writable
    /// shader bindings for this reserved allocation class.
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }
}

/// Checked handoff to the caller; this does NOT claim its backing was freed.
pub struct PublicExternalDetachReceipt<R> {
    resource: R,
    source_id: crate::memory_management::memory_pool::handle::ManagedMemoryId,
    source_stream: StreamId,
    /// Full registered shader descriptor bytes.
    pub buffer_bytes: u64,
    /// Caller-owned physical backing bytes now safe to release.
    pub backing_bytes: u64,
    /// Every already-submitted consumer completed before handoff.
    pub queue_joined: bool,
    /// Safe Buffer clones cannot submit future consumers.
    pub source_destroyed: bool,
}

impl<R> PublicExternalDetachReceipt<R> {
    /// Borrow the actual destroyed registered resource for exact caller-owned
    /// backing identity comparison. This does not make it usable for submission.
    pub fn resource(&self) -> &R {
        &self.resource
    }
    /// Authenticate a saved exact full registration binding without exposing its
    /// source identity. Private construction prevents fabricated success flags.
    pub fn matches_source(&self, binding: &Binding) -> bool {
        is_public_external_binding(binding)
            && binding.memory.descriptor().id == self.source_id
            && binding.stream == self.source_stream
            && binding.size == self.buffer_bytes
            && binding.offset_start.is_none()
            && binding.offset_end.is_none()
    }
}

impl<R> core::fmt::Debug for PublicExternalDetachReceipt<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PublicExternalDetachReceipt")
            .field("buffer_bytes", &self.buffer_bytes)
            .field("backing_bytes", &self.backing_bytes)
            .field("queue_joined", &self.queue_joined)
            .field("source_destroyed", &self.source_destroyed)
            .finish()
    }
}

/// Content-free pool accounting; no managed MAIN/runtime ownership is involved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PublicExternalUsage {
    /// Actual caller backing still registered or retained after failed detach.
    pub registered_backing_bytes: u64,
    /// Cumulative successful registered backing, never inferred from HAL stats.
    pub total_registered_bytes: u64,
    /// Cumulative safe handoff bytes; the caller still performs the real free.
    pub total_detached_bytes: u64,
    /// Live plus retired slot count; slots are never reused.
    pub registration_count: usize,
    /// Complete registry typed capacity plus bounded insertion temporaries.
    pub metadata_owned_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Active,
    Closing,
}
struct Entry<R> {
    handle: Handle,
    resource: R,
    bytes: u64,
    backing_bytes: u64,
    source_sha256: [u8; 32],
    state: State,
}

/// Separate, bounded, never-reused registration pool. It never allocates a GPU
/// resource or calls ordinary allocation/free/rebinding/cleanup functions.
pub struct PublicExternalPool<R> {
    budget: PublicExternalBudget,
    stream: StreamId,
    entries: Vec<Option<Entry<R>>>,
    usage: PublicExternalUsage,
}
impl<R> core::fmt::Debug for PublicExternalPool<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PublicExternalPool")
            .field("budget", &self.budget)
            .field("usage", &self.usage)
            .finish()
    }
}
impl<R: Clone> PublicExternalPool<R> {
    fn metadata_reservation(capacity: usize, buffers: usize) -> Result<usize, &'static str> {
        size_of::<Self>()
            .checked_add(
                capacity
                    .checked_mul(size_of::<Option<Entry<R>>>())
                    .ok_or("public external metadata overflow")?,
            )
            // Both separately allocated Arc control blocks / managed descriptor
            // are retained by Handle/Binding clones, even after slot retirement.
            // Conservatively charge the full lifetime registration capacity.
            .and_then(|n| {
                n.checked_add(
                    buffers
                        * (size_of::<
                            crate::memory_management::memory_pool::handle::ManagedMemoryDescriptor,
                        >() + 4 * size_of::<usize>()),
                )
            })
            // Bounded external-only kernel ID mask wrappers; driver/compiler
            // caches themselves remain outside this typed registry claim.
            .and_then(|n| n.checked_add(MAX_BUFFERS * 64))
            .and_then(|n| n.checked_add(2 * size_of::<Entry<R>>()))
            .ok_or("public external metadata overflow")
    }
    /// Reserve complete typed metadata once; no ordinary allocation is made.
    pub fn new(budget: PublicExternalBudget, stream: StreamId) -> Result<Self, &'static str> {
        let budget = budget.validate()?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(budget.buffers)
            .map_err(|_| "public external metadata allocation")?;
        let metadata_owned_bytes = Self::metadata_reservation(entries.capacity(), budget.buffers)?;
        if metadata_owned_bytes > budget.metadata_bytes {
            return Err("public external metadata capacity");
        }
        Ok(Self {
            budget,
            stream,
            entries,
            usage: PublicExternalUsage {
                metadata_owned_bytes,
                ..Default::default()
            },
        })
    }
    /// Monotonic extension of an already held PUBLIC backing/metadata grant.
    /// No allocation/free/re-registration of model resources and no recycled
    /// lifetime slots. Build replacement metadata before changing any identity.
    /// Caller holds the larger reservation; the stream owner joins readers.
    pub fn extend_budget(&mut self, budget: PublicExternalBudget) -> Result<(), &'static str> {
        let budget = budget.validate()?;
        if budget.backing_bytes < self.budget.backing_bytes
            || budget.buffers < self.budget.buffers
            || budget.metadata_bytes < self.budget.metadata_bytes
            || self
                .entries
                .iter()
                .flatten()
                .any(|e| e.state != State::Active)
        {
            return Err("public external extension must preserve healthy grants");
        }
        if budget == self.budget {
            return Ok(());
        }
        let old_vector = self
            .entries
            .capacity()
            .checked_mul(size_of::<Option<Entry<R>>>())
            .ok_or("public external extension metadata overflow")?;
        let predicted = Self::metadata_reservation(budget.buffers, budget.buffers)?;
        if predicted
            .checked_add(old_vector)
            .is_none_or(|n| n > budget.metadata_bytes)
        {
            return Err("public external extension peak metadata capacity");
        }
        let mut next = Vec::new();
        next.try_reserve_exact(budget.buffers)
            .map_err(|_| "public external extension metadata allocation")?;
        let actual = Self::metadata_reservation(next.capacity(), budget.buffers)?;
        if actual
            .checked_add(old_vector)
            .is_none_or(|n| n > budget.metadata_bytes)
        {
            return Err("public external extension peak metadata capacity");
        }
        next.append(&mut self.entries);
        self.entries = next;
        self.budget = budget;
        self.usage.metadata_owned_bytes = actual;
        Ok(())
    }
    /// Snapshot only numeric registry ownership, separate from managed memory.
    pub fn usage(&self) -> PublicExternalUsage {
        self.usage
    }
    /// Read registered resources for backend duplicate-buffer validation only.
    pub fn resources(&self) -> impl Iterator<Item = &R> {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|entry| &entry.resource))
    }
    /// Insert only after backend same-device/full-span/read-only checks succeed.
    pub fn register(
        &mut self,
        source: PublicExternalHostSource<R>,
    ) -> Result<PublicExternalHandle, &'static str> {
        let live = self
            .usage
            .registered_backing_bytes
            .checked_add(source.backing_bytes)
            .ok_or("public external backing overflow")?;
        let total = self
            .usage
            .total_registered_bytes
            .checked_add(source.backing_bytes)
            .ok_or("public external cumulative overflow")?;
        if self.entries.len() >= self.budget.buffers || live > self.budget.backing_bytes {
            return Err("public external registration capacity");
        }
        let memory = ManagedMemoryHandle::new_public_external();
        memory.descriptor().update_location(MemoryLocation::new(
            EXTERNAL_TAG,
            0,
            self.entries.len() as u32,
        ));
        let handle = Handle::from_memory(memory, self.stream, source.bytes);
        self.entries.push(Some(Entry {
            handle: handle.clone(),
            resource: source.resource,
            bytes: source.bytes,
            backing_bytes: source.backing_bytes,
            source_sha256: source.source_sha256,
            state: State::Active,
        }));
        self.usage.registered_backing_bytes = live;
        self.usage.total_registered_bytes = total;
        self.usage.registration_count = self.entries.len();
        Ok(PublicExternalHandle { handle })
    }
    fn entry(&self, binding: &Binding) -> Result<(usize, &Entry<R>), &'static str> {
        let descriptor = binding.memory.descriptor();
        let location = descriptor.location();
        if binding.stream != self.stream
            || !is_public_external_binding(binding)
            || location.init != 1
            || location.page != 0
        {
            return Err("public external stream/class identity");
        }
        let index = location.slice as usize;
        let entry = self
            .entries
            .get(index)
            .and_then(Option::as_ref)
            .ok_or("public external retired/unknown source")?;
        if entry.handle.memory.descriptor() != descriptor
            || binding.size != entry.bytes
            || entry.source_sha256 == [0; 32]
        {
            return Err("public external exact source identity");
        }
        let first = binding.offset_start.unwrap_or(0);
        let end = binding.offset_end.unwrap_or(0);
        if first.checked_add(end).is_none_or(|n| n >= entry.bytes)
            || !first.is_multiple_of(4)
            || !end.is_multiple_of(4)
        {
            return Err("public external binding span");
        }
        Ok((index, entry))
    }
    /// Resolve only active immutable sources. Closing/retired Handles are stale.
    pub fn resolve(&self, binding: &Binding) -> Result<R, &'static str> {
        let (_, entry) = self.entry(binding)?;
        if entry.state != State::Active {
            return Err("public external source closing");
        }
        Ok(entry.resource.clone())
    }
    /// Revoke future binding lookups BEFORE queue completion. The caller must
    /// retain this closing entry on failure and never free its backing.
    pub fn begin_detach(&mut self, handle: &PublicExternalHandle) -> Result<R, &'static str> {
        let binding = handle.handle.clone().binding();
        let (index, entry) = self.entry(&binding)?;
        if entry.state != State::Active
            || binding.offset_start.is_some()
            || binding.offset_end.is_some()
        {
            return Err("public external detach authority");
        }
        self.entries[index].as_mut().unwrap().state = State::Closing;
        Ok(self.entries[index].as_ref().unwrap().resource.clone())
    }
    /// Consume ONLY after the backend has destroyed the safe Buffer and joined
    /// every original queue consumer. This method itself performs neither.
    ///
    /// # Safety
    /// The backend must have successfully joined queued consumers, destroyed
    /// every safe clone through the shared Buffer identity, and joined again.
    pub unsafe fn finish_detach(
        &mut self,
        handle: PublicExternalHandle,
    ) -> Result<PublicExternalDetachReceipt<R>, &'static str> {
        let (index, entry) = self.entry(&handle.handle.binding())?;
        if entry.state != State::Closing {
            return Err("public external detach not joined");
        }
        let (buffer_bytes, backing_bytes) = (entry.bytes, entry.backing_bytes);
        let live = self
            .usage
            .registered_backing_bytes
            .checked_sub(backing_bytes)
            .ok_or("public external detach accounting")?;
        let total = self
            .usage
            .total_detached_bytes
            .checked_add(backing_bytes)
            .ok_or("public external detach accounting")?;
        let source_id = entry.handle.memory.descriptor().id;
        let resource = entry.resource.clone();
        self.entries[index] = None;
        self.usage.registered_backing_bytes = live;
        self.usage.total_detached_bytes = total;
        Ok(PublicExternalDetachReceipt {
            resource,
            source_id,
            source_stream: self.stream,
            buffer_bytes,
            backing_bytes,
            queue_joined: true,
            source_destroyed: true,
        })
    }
}

/// Reserved allocation-class check; no ordinary pool can resolve this class.
pub fn is_public_external_binding(binding: &Binding) -> bool {
    binding.memory.is_public_external()
}
/// Reserved class check for operations that must reject source mutation.
pub fn is_public_external_memory(memory: &crate::memory_management::ManagedMemoryBinding) -> bool {
    memory.is_public_external()
}
