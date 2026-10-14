//! Small laboratory owner for transient resident-graph GPU buffers.
//!
//! Every allocation is a complete power-of-two 32-bit span, at least 32 KiB
//! (the required custom pool's minimum bucket). Logical lengths are separate from owned
//! spans; initialization and erasure cover EVERY owned word, including padding.
//! The checked budget charges owned spans, not logical payloads. Public model
//! weights do not belong in this arena.
//!
//! BufferId binds arena identity + reset epoch + slot. It exposes no raw memory
//! ID. A borrowed handle is validated before use. Caller must launch work on
//! this same worker thread/client stream; raw clones are for immediate enqueue
//! only and must never escape the arena lifetime or be used after reset/close.
//! This owner is deliberately !Send/!Sync and not a production execution API.
//! The client MUST have been initialized with required_memory_configuration().
//! Default CubeCL pools may own larger hidden pages than Handle::size(); merely
//! requesting power-of-two spans does not prove erasure of those hidden tails.
//!
//! close/reset drain earlier work, enqueue erasure of complete tracked spans,
//! synchronize erasure, then release handles. Drop attempts the same cleanup.
//! Any initialization/backend/cleanup failure retires the SHARED context flag;
//! the worker must stop using this device context, including public caches.
//! Failure does not prove erasure; context/process recovery remains root-owned.
//! new() retains the original PUBLIC fixture upload/sync policy. new_transient
//! opts into owned runtime I/O and bounded sanitizer cleanup. Its caller must
//! provision enough I/O budget for ALL eraser uniforms after a checkpoint;
//! the arena never falls back to an ordinary upload or unbounded sync.
//! Optional fixed-public zero audit maps each complete owning page only AFTER
//! checked erasure. It drops each guarded lease then audits/erases its runtime
//! readback before any main handle release. A failed audit retires the context;
//! Drop quarantines retained handles until process recovery. Counts cover API
//! pages and padding, never opaque driver allocations or public model caches.
//! Runtime-owned I/O needs separate device audit acceptance. No content or backend error payload
//! is logged. No device initialization occurs here or in host tests.

use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};
use zeroize::Zeroize;

pub const MIN_SPAN_BYTES: usize = 32 * 1024;
pub const MAX_SPAN_BYTES: usize = if cfg!(feature = "strata-intel-core") {
    512 * 1024 * 1024
} else {
    128 * 1024 * 1024
};
const ZERO_AUDIT_WINDOW: usize = 32 * 1024 * 1024;
const LANES: usize = 64;
static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Required allocator configuration at Wgpu init_device, BEFORE constructing
/// a client or any public/private allocations on that context. Requests that
/// exactly equal a bin maximum cannot acquire a larger hidden page: CubeCL
/// exclusive pools allocate max(request, rolling_mean)<=this bin maximum.
/// This constructor cannot inspect an existing client's allocator config.
pub fn required_memory_configuration() -> cubecl::MemoryConfiguration {
    cubecl::MemoryConfiguration::Custom {
        pool_options: (15..=MAX_SPAN_BYTES.ilog2())
            .map(|power| MemoryPoolOptions {
                pool_type: PoolType::ExclusivePages {
                    max_alloc_size: 1u64 << power,
                },
                dealloc_period: Some(1),
            })
            .collect(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaError {
    InvalidLayout,
    BudgetExceeded,
    HostAllocation,
    Closed,
    Retired,
    StaleHandle,
    IdentityExhausted,
    DeviceAllocation,
    Initialization,
    Cleanup,
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self {
            Self::InvalidLayout => "private arena invalid layout",
            Self::BudgetExceeded => "private arena budget exceeded",
            Self::HostAllocation => "private arena host allocation failed",
            Self::Closed => "private arena closed",
            Self::Retired => "private arena context retired",
            Self::StaleHandle => "private arena stale or foreign handle",
            Self::IdentityExhausted => "private arena identity exhausted",
            Self::DeviceAllocation => "private arena device allocation failed",
            Self::Initialization => "private arena initialization failed",
            Self::Cleanup => "private arena cleanup failed",
        };
        f.write_str(reason)
    }
}
impl std::error::Error for ArenaError {}
pub type Result<T> = std::result::Result<T, ArenaError>;

/// Opaque token: private fields prevent forged/cross-arena slot access.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BufferId {
    arena: u64,
    epoch: u64,
    slot: usize,
}

#[cfg(test)]
impl BufferId {
    pub(crate) fn metadata_fixture(arena: u64, epoch: u64, slot: usize) -> Self {
        Self { arena, epoch, slot }
    }
}

// Fixed private transaction metadata can wipe opaque IDs without exposing or
// forging their fields. The zero token is invalid (arena identities start at1).
impl Zeroize for BufferId {
    fn zeroize(&mut self) {
        self.arena.zeroize();
        self.epoch.zeroize();
        self.slot.zeroize();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferLayout {
    pub logical_words: usize,
    pub owned_words: usize,
    pub owned_bytes: usize,
    pub workgroups_x: u32,
    pub workgroups_y: u32,
}

impl BufferLayout {
    /// An attested complete external cache buffer has its exact physical size,
    /// not an allocator-bin rounding. This constructor is never an allocator.
    fn borrowed_complete(owned_bytes: usize) -> Result<Self> {
        if !(MIN_SPAN_BYTES..=MAX_SPAN_BYTES).contains(&owned_bytes)
            || !owned_bytes.is_multiple_of(256)
        {
            return Err(ArenaError::InvalidLayout);
        }
        let owned_words = owned_bytes / 4;
        let groups = owned_words.div_ceil(LANES);
        let x = groups.min(65535);
        Ok(Self {
            logical_words: owned_words,
            owned_words,
            owned_bytes,
            workgroups_x: x as u32,
            workgroups_y: groups.div_ceil(x) as u32,
        })
    }
    pub fn new(logical_words: usize) -> Result<Self> {
        if logical_words == 0 {
            return Err(ArenaError::InvalidLayout);
        }
        let logical_bytes = logical_words
            .checked_mul(4)
            .ok_or(ArenaError::InvalidLayout)?;
        let owned_bytes = logical_bytes
            .max(MIN_SPAN_BYTES)
            .checked_next_power_of_two()
            .ok_or(ArenaError::InvalidLayout)?;
        if owned_bytes > MAX_SPAN_BYTES {
            return Err(ArenaError::InvalidLayout);
        }
        let owned_words = owned_bytes / 4;
        // Both selected caps fit32-bit flattened indexing. The ordinary path
        // keeps portable128MiB bindings; native prompt owners additionally
        // require the explicit larger hardware admission before allocation.
        let groups = owned_words.div_ceil(LANES);
        let x = groups.min(65_535);
        let y = groups.div_ceil(x);
        Ok(Self {
            logical_words,
            owned_words,
            owned_bytes,
            workgroups_x: x as u32,
            workgroups_y: y as u32,
        })
    }
}

/// Pure host bookkeeping, shared by real owner and host tests.
struct Ledger {
    arena: u64,
    epoch: u64,
    budget: usize,
    charged: usize,
    spans: Vec<BufferLayout>,
    closed: bool,
}

impl Ledger {
    fn new(budget: usize) -> Result<Self> {
        if budget < MIN_SPAN_BYTES {
            return Err(ArenaError::BudgetExceeded);
        }
        let arena = NEXT_ARENA_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| ArenaError::IdentityExhausted)?;
        Ok(Self {
            arena,
            epoch: 1,
            budget,
            charged: 0,
            spans: Vec::new(),
            closed: false,
        })
    }
    fn reserve(&mut self, layout: BufferLayout) -> Result<BufferId> {
        if self.closed {
            return Err(ArenaError::Closed);
        }
        if layout != BufferLayout::new(layout.logical_words)? {
            return Err(ArenaError::InvalidLayout);
        }
        self.reserve_checked(layout)
    }
    fn reserve_borrowed(&mut self, layout: BufferLayout) -> Result<BufferId> {
        if self.closed {
            return Err(ArenaError::Closed);
        }
        if layout != BufferLayout::borrowed_complete(layout.owned_bytes)? {
            return Err(ArenaError::InvalidLayout);
        }
        self.reserve_checked(layout)
    }
    fn reserve_checked(&mut self, layout: BufferLayout) -> Result<BufferId> {
        let charged = self
            .charged
            .checked_add(layout.owned_bytes)
            .ok_or(ArenaError::BudgetExceeded)?;
        if charged > self.budget {
            return Err(ArenaError::BudgetExceeded);
        }
        self.spans
            .try_reserve(1)
            .map_err(|_| ArenaError::HostAllocation)?;
        let id = BufferId {
            arena: self.arena,
            epoch: self.epoch,
            slot: self.spans.len(),
        };
        self.spans.push(layout);
        self.charged = charged;
        Ok(id)
    }
    fn lookup(&self, id: BufferId) -> Result<BufferLayout> {
        if self.closed {
            return Err(ArenaError::Closed);
        }
        if id.arena != self.arena || id.epoch != self.epoch {
            return Err(ArenaError::StaleHandle);
        }
        self.spans
            .get(id.slot)
            .copied()
            .ok_or(ArenaError::StaleHandle)
    }
    fn rollback_last(&mut self, id: BufferId) {
        // Allocation failure before a raw handle exists. Only this final
        // unpublished reservation can be rolled back; slots are never reused
        // for published IDs within an epoch.
        if id.arena == self.arena
            && id.epoch == self.epoch
            && id.slot + 1 == self.spans.len()
            && let Some(span) = self.spans.pop()
        {
            self.charged -= span.owned_bytes;
        }
    }
    fn reset_after_erasure(&mut self) -> Result<()> {
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(ArenaError::IdentityExhausted)?;
        self.spans.clear();
        self.charged = 0;
        self.epoch = epoch;
        Ok(())
    }
    fn close_after_erasure(&mut self) {
        self.closed = true;
        self.spans.clear();
        self.charged = 0;
    }
}

#[cube(launch)]
pub fn erase_full_span(words: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < words.len() {
        words[i] = 0;
    }
}

fn sync(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sync())
    })) {
        Ok(Ok(())) => Ok(()),
        _ => Err(ArenaError::Cleanup),
    }
}

struct Upload(Vec<u8>);
impl Upload {
    fn into_vec(mut self) -> Vec<u8> {
        // Transfer the original allocation; empty Upload Drop cannot erase or
        // release the moved bytes. Runtime immediately installs its wipe guard.
        std::mem::take(&mut self.0)
    }
    fn zeroed(layout: BufferLayout) -> Result<Self> {
        let mut data = Vec::new();
        data.try_reserve_exact(layout.owned_bytes)
            .map_err(|_| ArenaError::HostAllocation)?;
        data.resize(layout.owned_bytes, 0);
        Ok(Self(data))
    }
    fn floats(layout: BufferLayout, values: &[f32]) -> Result<Self> {
        if values.len() != layout.logical_words || !values.iter().all(|v| v.is_finite()) {
            return Err(ArenaError::InvalidLayout);
        }
        let mut upload = Self::zeroed(layout)?;
        for (word, value) in upload.0.chunks_exact_mut(4).zip(values) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        Ok(upload)
    }
    fn words(layout: BufferLayout, values: &[u32]) -> Result<Self> {
        if values.len() != layout.logical_words {
            return Err(ArenaError::InvalidLayout);
        }
        let mut upload = Self::zeroed(layout)?;
        for (word, value) in upload.0.chunks_exact_mut(4).zip(values) {
            word.copy_from_slice(&value.to_le_bytes());
        }
        Ok(upload)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IoPolicy {
    PublicFixture,
    OwnedTransient,
}

/// Same cleanup transaction drives the native owner and fault-injected host
/// fixtures. The first checkpoint frees prior staging/uniform budget BEFORE
/// the main erasers allocate metadata. No error skips a known main span or the
/// second checkpoint, and a prior failure cannot become accepted cleanup.
trait TransientCleanup {
    fn checkpoint(&mut self) -> Result<()>;
    fn erase(&mut self, slot: usize) -> Result<()>;
}
fn cleanup_transient(
    backend: &mut impl TransientCleanup,
    buffers: usize,
    previous_failure: bool,
) -> Result<()> {
    let mut failed = previous_failure;
    failed |= backend.checkpoint().is_err();
    for slot in 0..buffers {
        failed |= backend.erase(slot).is_err();
    }
    failed |= backend.checkpoint().is_err();
    if failed {
        Err(ArenaError::Cleanup)
    } else {
        Ok(())
    }
}
impl Drop for Upload {
    fn drop(&mut self) {
        let pointer = self.0.as_mut_ptr();
        for i in 0..self.0.capacity() {
            // SAFETY: each address is inside this exclusively owned allocation;
            // zero is valid for u8, including writable uninitialized capacity.
            unsafe {
                pointer.add(i).write_volatile(0);
            }
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}

/// Borrowed ownership prevents a write/error callback from discarding the
/// destination owner. Only its handle clone and the original guarded Vec move
/// into the backend; failures leave the owning handle slice untouched.
fn write_owned_upload<H: Clone>(
    handles: &[H],
    slot: usize,
    upload: Upload,
    write: impl FnOnce(H, Vec<u8>) -> Result<()>,
) -> Result<()> {
    let handle = handles.get(slot).ok_or(ArenaError::Initialization)?.clone();
    write(handle, upload.into_vec())
}

fn initialize_owned_upload(
    ordered: bool,
    mut checkpoint: impl FnMut() -> Result<()>,
    write: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if !ordered {
        checkpoint()?;
    }
    write()?;
    if !ordered {
        checkpoint()?;
    }
    Ok(())
}

/// Counts of COMPLETE arena-owned pages accepted as zero. No readback bytes
/// escape inspection, and counts advance only after all leases/stages are gone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OwnedZeroAudit {
    pub bytes: u64,
    pub buffers: u64,
}
impl OwnedZeroAudit {
    fn checked_add(self, next: Self) -> Result<Self> {
        Ok(Self {
            bytes: self
                .bytes
                .checked_add(next.bytes)
                .ok_or(ArenaError::Cleanup)?,
            buffers: self
                .buffers
                .checked_add(next.buffers)
                .ok_or(ArenaError::Cleanup)?,
        })
    }
}

struct AuditBytes(cubecl::bytes::Bytes);
impl AsRef<[u8]> for AuditBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl Drop for AuditBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

trait OwnedZeroInspection {
    type Lease: AsRef<[u8]>;
    fn read_range(&mut self, slot: usize, first: usize, bytes: usize) -> Result<Self::Lease>;
    fn cleanup_after_read(&mut self) -> Result<()>;
}

/// Consumes the lease; its complete private host allocation is guarded even
/// when bounds/zero acceptance fails. In particular, logical-prefix zeros do
/// not credit an uninspected tail. No content is returned or formatted.
fn inspect_zero_span(lease: impl AsRef<[u8]>, owned_bytes: usize) -> Result<()> {
    let bytes = lease.as_ref();
    if owned_bytes == 0
        || bytes.len() != owned_bytes
        || !cubecl_runtime::transient::full_span_is_zero(bytes)
    {
        return Err(ArenaError::Cleanup);
    }
    Ok(())
}
fn inspect_owned_zero(
    backend: &mut impl OwnedZeroInspection,
    spans: &[BufferLayout],
) -> Result<OwnedZeroAudit> {
    inspect_owned_zero_window(backend, spans, ZERO_AUDIT_WINDOW)
}
fn inspect_owned_zero_window(
    backend: &mut impl OwnedZeroInspection,
    spans: &[BufferLayout],
    window: usize,
) -> Result<OwnedZeroAudit> {
    if window == 0 || !window.is_multiple_of(4) {
        return Err(ArenaError::Cleanup);
    }
    let mut counts = OwnedZeroAudit::default();
    for (slot, span) in spans.iter().enumerate() {
        if span.owned_bytes == 0 {
            return Err(ArenaError::Cleanup);
        }
        for first in (0..span.owned_bytes).step_by(window) {
            let bytes = window.min(span.owned_bytes - first);
            let zero = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inspect_zero_span(backend.read_range(slot, first, bytes)?, bytes)
            }));
            // Every lease is wiped/dropped before sanitizer and the next
            // bounded window. No prefix can credit an uninspected root tail.
            let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.cleanup_after_read()
            }));
            if !matches!(zero, Ok(Ok(()))) || !matches!(cleanup, Ok(Ok(()))) {
                return Err(ArenaError::Cleanup);
            }
        }
        counts = counts.checked_add(OwnedZeroAudit {
            bytes: span.owned_bytes as u64,
            buffers: 1,
        })?;
    }
    Ok(counts)
}

fn accept_owned_zero_cleanup(
    retired: &AtomicBool,
    previous: OwnedZeroAudit,
    cleanup: impl FnOnce() -> Result<OwnedZeroAudit>,
) -> Result<OwnedZeroAudit> {
    let accepted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup))
        .map_err(|_| ArenaError::Cleanup)
        .and_then(|counts| counts)
        .and_then(|counts| previous.checked_add(counts));
    if retired.load(Ordering::Acquire) || accepted.is_err() {
        retired.store(true, Ordering::Release);
        return Err(ArenaError::Cleanup);
    }
    accepted
}
fn accept_retained_public_erasure(
    retired: &AtomicBool,
    expected: OwnedZeroAudit,
    cleanup: impl FnOnce() -> Result<OwnedZeroAudit>,
) -> Result<OwnedZeroAudit> {
    accept_owned_zero_cleanup(retired, OwnedZeroAudit::default(), || {
        let counts = cleanup()?;
        if counts != expected {
            return Err(ArenaError::Cleanup);
        }
        Ok(counts)
    })
}
fn quarantine_failed_audit_handles<H>(handles: &mut Vec<H>, audit: bool, failed: bool) {
    if audit && failed {
        std::mem::forget(std::mem::take(handles));
    }
}
pub struct PrivateArena {
    client: ComputeClient<WgpuRuntime>,
    ledger: Ledger,
    handles: Vec<Handle>,
    retired: Arc<AtomicBool>,
    io_policy: IoPolicy,
    ordered_uploads: bool,
    public_zero_audit: bool,
    zero_audit_counts: OwnedZeroAudit,
    closed_clean: bool,
    // Registered roots remain charged to their persistent cache owner. Ledger
    // charge here is private coverage, never a new physical allocation credit.
    borrowed_roots: bool,
    // CubeCL default stream depends on the current thread. Pin all arena
    // construction, commands, cleanup and Drop to the same worker thread.
    _same_thread: PhantomData<Rc<()>>,
}

impl PrivateArena {
    /// PRECONDITION: client was initialized with required_memory_configuration;
    /// the worker serializes all commands/cleanup on this construction thread.
    pub fn new(
        client: ComputeClient<WgpuRuntime>,
        budget_bytes: usize,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        if retired.load(Ordering::Acquire) {
            return Err(ArenaError::Retired);
        }
        Ok(Self {
            client,
            ledger: Ledger::new(budget_bytes)?,
            handles: Vec::new(),
            retired,
            io_policy: IoPolicy::PublicFixture,
            ordered_uploads: false,
            public_zero_audit: false,
            zero_audit_counts: OwnedZeroAudit::default(),
            closed_clean: false,
            borrowed_roots: false,
            _same_thread: PhantomData,
        })
    }
    /// Opt-in laboratory ownership on one serialized worker/client stream.
    /// The client must use required_memory_configuration before any allocation.
    /// Configuration failure retires the shared context without a fallback.
    pub fn new_transient(
        client: ComputeClient<WgpuRuntime>,
        budget_bytes: usize,
        io_budget_bytes: u64,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut arena = Self::new(client, budget_bytes, retired)?;
        arena.io_policy = IoPolicy::OwnedTransient;
        let configured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arena.client.configure_transient_io(io_budget_bytes)
        }));
        if !matches!(configured, Ok(Ok(()))) {
            arena.retired.store(true, Ordering::Release);
            arena.ledger.closed = true;
            return Err(ArenaError::Initialization);
        }
        Ok(arena)
    }
    /// PRECONDITION: the root has already configured bounded transient I/O on
    /// this SAME serialized stream. Several request arenas may share it.
    /// Actual private writes/checkpoints reject an unconfigured backend; there
    /// is no ordinary-I/O fallback and failure retires the shared context.
    pub(crate) fn new_transient_on_configured(
        client: ComputeClient<WgpuRuntime>,
        budget_bytes: usize,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut arena = Self::new(client, budget_bytes, retired)?;
        arena.io_policy = IoPolicy::OwnedTransient;
        Ok(arena)
    }
    /// Register real persistent cache roots under an exclusive lending owner.
    /// No allocation/public upload/configuration is performed here. Local ledger
    /// bytes describe private coverage of ALREADY precharged persistent memory.
    ///
    /// # Safety
    /// The caller owns the complete cache roots and their persistent charge,
    /// has joined all prior readers, excludes all ordinary source consumers and
    /// retains that exclusion through completed erasure AND model restoration.
    /// No handle/ID may escape that transaction or survive close. Same configured
    /// serialized transient client/worker, with the real shared retirement flag.
    pub(crate) unsafe fn register_cache_roots_on_configured(
        client: ComputeClient<WgpuRuntime>,
        roots: [Handle; 3],
        retired: Arc<AtomicBool>,
    ) -> Result<(Self, [BufferId; 3])> {
        if !cfg!(feature = "strata-intel-core") {
            return Err(ArenaError::InvalidLayout);
        }
        let mut layouts = Vec::with_capacity(3);
        let mut budget = 0usize;
        for (i, h) in roots.iter().enumerate() {
            let bytes = usize::try_from(h.size_in_used()).map_err(|_| ArenaError::InvalidLayout)?;
            if h.size() != bytes as u64
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.stream != roots[0].stream
                || roots[..i].iter().any(|other| {
                    h.clone().binding().memory.allocation_id()
                        == other.clone().binding().memory.allocation_id()
                })
            {
                return Err(ArenaError::InvalidLayout);
            }
            layouts.push(BufferLayout::borrowed_complete(bytes)?);
            budget = budget
                .checked_add(bytes)
                .ok_or(ArenaError::BudgetExceeded)?;
        }
        // Backend proves these are the actual complete buffers. A logical
        // Handle or a cache subview is insufficient, including hidden tails.
        let resources = client
            .retained_resource_metadata(roots.to_vec())
            .map_err(|_| ArenaError::DeviceAllocation)?;
        if resources.len() != roots.len() {
            return Err(ArenaError::DeviceAllocation);
        }
        for (i, r) in resources.iter().enumerate() {
            let r = r.resource();
            if r.offset != 0
                || r.size != roots[i].size_in_used()
                || r.buffer.size() != r.size
                || resources[..i]
                    .iter()
                    .any(|prior| prior.resource().buffer == r.buffer)
            {
                return Err(ArenaError::DeviceAllocation);
            }
        }
        let mut arena = Self::new_transient_on_configured(client, budget, retired)?;
        arena.borrowed_roots = true;
        arena
            .handles
            .try_reserve_exact(3)
            .map_err(|_| ArenaError::HostAllocation)?;
        let mut ids = Vec::with_capacity(3);
        for (h, layout) in roots.into_iter().zip(layouts) {
            let id = arena.ledger.reserve_borrowed(layout)?;
            arena.handles.push(h);
            ids.push(id);
        }
        // All owning handles are retained before the first destructive write.
        for slot in 0..3 {
            if arena.initialize_transient(slot, None).is_err() {
                arena.fail();
                return Err(ArenaError::Initialization);
            }
        }
        Ok((
            arena,
            ids.try_into().map_err(|_| ArenaError::Initialization)?,
        ))
    }
    /// Source-only arena for same-stream immediate copies. A full-span owned
    /// upload is QUEUED before its ID returns; it is not a completed fence.
    /// Caller retains every source until a completed readback plus sanitizer,
    /// then reset/close erases all main spans. Never sanitize pending inference
    /// merely to upload a source: cleanup timeouts are not inference deadlines.
    pub(crate) fn new_transient_source_on_configured(
        client: ComputeClient<WgpuRuntime>,
        budget_bytes: usize,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut arena = Self::new_transient_on_configured(client, budget_bytes, retired)?;
        arena.ordered_uploads = true;
        Ok(arena)
    }
    /// Explicit fixed-public operator diagnostic. Set before allocations so
    /// constructor-failure Drop also follows this policy. Never applicable to
    /// the ordinary benchmark owner or a production/private request API.
    pub(crate) fn enable_public_owned_zero_audit(&mut self) -> Result<()> {
        self.active()?;
        if self.io_policy != IoPolicy::OwnedTransient || !self.handles.is_empty() {
            self.retired.store(true, Ordering::Release);
            return Err(ArenaError::Initialization);
        }
        self.public_zero_audit = true;
        Ok(())
    }
    /// Private borrowing mandates complete audit without granting PUBLIC fixture
    /// replay authority. Strict cleanup and public-test permission are separate.
    fn strict_zero_audit(&self) -> bool {
        self.public_zero_audit || self.borrowed_roots
    }
    /// Configuration only; successful erasure credit still requires close.
    pub(crate) fn public_owned_zero_audit_enabled(&self) -> bool {
        self.io_policy == IoPolicy::OwnedTransient && self.public_zero_audit
    }
    /// Accepted counts are available only AFTER checked explicit close. A
    /// failed/retired context cannot emit prior successful partial audit credit.
    pub(crate) fn owned_zero_audit_receipt(&self) -> Option<OwnedZeroAudit> {
        (self.strict_zero_audit() && self.closed_clean && !self.retired.load(Ordering::Acquire))
            .then_some(self.zero_audit_counts)
    }
    fn active(&self) -> Result<()> {
        if self.ledger.closed {
            return Err(ArenaError::Closed);
        }
        if self.retired.load(Ordering::Acquire) {
            return Err(ArenaError::Retired);
        }
        Ok(())
    }
    pub fn charged_bytes(&self) -> usize {
        self.ledger.charged
    }
    pub fn budget_bytes(&self) -> usize {
        self.ledger.budget
    }
    pub fn buffer_count(&self) -> usize {
        self.handles.len()
    }
    pub fn layout(&self, id: BufferId) -> Result<BufferLayout> {
        self.active()?;
        self.ledger.lookup(id)
    }
    /// Returns the FULL owned span; ArrayArg logical length must be supplied
    /// separately from layout.logical_words. Never derive logical length from
    /// this Handle's byte size. Borrowed handles cannot survive mutable reset.
    pub fn handle(&self, id: BufferId) -> Result<&Handle> {
        self.layout(id)?;
        self.handles.get(id.slot).ok_or(ArenaError::StaleHandle)
    }
    pub fn client(&self) -> Result<&ComputeClient<WgpuRuntime>> {
        self.active()?;
        Ok(&self.client)
    }
    /// Full-span zero initialization is synchronized before an ID is returned.
    pub fn allocate_zeroed(&mut self, logical_words: usize) -> Result<BufferId> {
        self.allocate(BufferLayout::new(logical_words)?, None)
    }
    /// Caller retains responsibility for clearing its source slice. The
    /// arena-owned padded host upload is cleared over its entire capacity.
    pub fn upload_f32(&mut self, values: &[f32]) -> Result<BufferId> {
        self.active()?;
        let layout = BufferLayout::new(values.len())?;
        let upload = Upload::floats(layout, values)?;
        self.allocate(layout, Some(upload))
    }
    pub fn upload_u32(&mut self, values: &[u32]) -> Result<BufferId> {
        self.active()?;
        let layout = BufferLayout::new(values.len())?;
        let upload = Upload::words(layout, values)?;
        self.allocate(layout, Some(upload))
    }
    fn allocate(&mut self, layout: BufferLayout, upload: Option<Upload>) -> Result<BufferId> {
        self.active()?;
        if self.borrowed_roots {
            return Err(ArenaError::InvalidLayout);
        }
        // Allocate metadata capacity before any device operation. Charge the
        // complete span before allocating; never publish an uninitialized ID.
        self.handles
            .try_reserve(1)
            .map_err(|_| ArenaError::HostAllocation)?;
        let id = self.ledger.reserve(layout)?;
        let allocated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match (self.io_policy, upload.as_ref()) {
                (IoPolicy::PublicFixture, Some(bytes)) => self.client.create_from_slice(&bytes.0),
                // Own a separately allocated FULL main span BEFORE a private
                // write can fail; create_private cannot return its handle on Err.
                _ => self.client.empty(layout.owned_bytes),
            }
        }));
        let handle = match allocated {
            Ok(h) => h,
            Err(_) => {
                self.ledger.rollback_last(id);
                self.fail();
                return Err(ArenaError::DeviceAllocation);
            }
        };
        // Own the handle before any launch/sync that can fail. No private
        // buffer is freed without a tracked best-effort full-span erasure.
        self.handles.push(handle);
        if self.handles[id.slot].size_in_used() != layout.owned_bytes as u64
            || self.handles[id.slot].size() != layout.owned_bytes as u64
        {
            self.fail();
            return Err(ArenaError::DeviceAllocation);
        }
        let initialized = match self.io_policy {
            IoPolicy::PublicFixture if upload.is_some() => sync(&self.client),
            IoPolicy::PublicFixture => self.queue_erase(id.slot).and_then(|()| sync(&self.client)),
            IoPolicy::OwnedTransient => self.initialize_transient(id.slot, upload),
        };
        if initialized.is_err() {
            self.fail();
            return Err(ArenaError::Initialization);
        }
        Ok(id)
    }
    fn checkpoint_transient(&self) -> Result<()> {
        self.checkpoint_transient_with_audit(false)
    }
    fn checkpoint_transient_with_audit(&self, audit: bool) -> Result<()> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::future::block_on(if audit {
                self.client.sanitize_transient_io_audited()
            } else {
                self.client.sanitize_transient_io()
            })
        })) {
            Ok(Ok(_counts)) => Ok(()),
            _ => Err(ArenaError::Cleanup),
        }
    }
    fn initialize_transient(&self, slot: usize, upload: Option<Upload>) -> Result<()> {
        // Destination is already tracked. Ordinary owned initialization frees
        // I/O budget and proves completion. A source-only upload instead relies
        // on same-stream copy order and the owner's next completed readback.
        match upload {
            Some(upload) => initialize_owned_upload(
                self.ordered_uploads,
                || self.checkpoint_transient(),
                || {
                    write_owned_upload(&self.handles, slot, upload, |handle, data| {
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.client.write_private(handle, data)
                        })) {
                            Ok(Ok(())) => Ok(()),
                            _ => Err(ArenaError::Initialization),
                        }
                    })
                },
            ),
            None => {
                self.checkpoint_transient()?;
                self.queue_erase(slot)?;
                self.checkpoint_transient()
            }
        }
    }
    fn queue_erase(&self, slot: usize) -> Result<()> {
        let layout = self
            .ledger
            .spans
            .get(slot)
            .copied()
            .ok_or(ArenaError::Cleanup)?;
        let handle = self.handles.get(slot).ok_or(ArenaError::Cleanup)?;
        if handle.size() != layout.owned_bytes as u64
            || handle.size_in_used() != layout.owned_bytes as u64
        {
            return Err(ArenaError::Cleanup);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            erase_full_span::launch::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
                CubeDim::new_1d(LANES as u32),
                // SAFETY: full owning handle has exactly owned_words*4 bytes;
                // arena retains ownership until all queued commands finish.
                unsafe { ArrayArg::from_raw_parts(handle.clone(), layout.owned_words) },
            );
        })) {
            Ok(()) => Ok(()),
            Err(_) => Err(ArenaError::Cleanup),
        }
    }
    fn erase_and_drain(&self) -> Result<()> {
        if self.io_policy == IoPolicy::OwnedTransient {
            struct Native<'a>(&'a PrivateArena);
            impl TransientCleanup for Native<'_> {
                fn checkpoint(&mut self) -> Result<()> {
                    self.0
                        .checkpoint_transient_with_audit(self.0.strict_zero_audit())
                }
                fn erase(&mut self, slot: usize) -> Result<()> {
                    self.0.queue_erase(slot)
                }
            }
            return cleanup_transient(
                &mut Native(self),
                self.handles.len(),
                self.retired.load(Ordering::Acquire),
            );
        }
        // Even if the first drain fails, attempt erasure of EVERY known buffer,
        // then another sync. A failed first drain still retires the context.
        let mut failed = sync(&self.client).is_err();
        for slot in 0..self.handles.len() {
            failed |= self.queue_erase(slot).is_err();
        }
        failed |= sync(&self.client).is_err();
        if failed {
            Err(ArenaError::Cleanup)
        } else {
            Ok(())
        }
    }
    fn erase_and_verify(&self) -> Result<OwnedZeroAudit> {
        // Explicit cleanup or a reset AFTER an accepted completed token fence.
        // The ordinary inference read path never calls this cleanup-only read.
        self.erase_and_drain()?;
        if !self.strict_zero_audit() {
            return Ok(OwnedZeroAudit::default());
        }
        if self.io_policy != IoPolicy::OwnedTransient
            || self.handles.len() != self.ledger.spans.len()
        {
            return Err(ArenaError::Cleanup);
        }
        struct Native<'a>(&'a PrivateArena);
        impl OwnedZeroInspection for Native<'_> {
            type Lease = AuditBytes;
            fn read_range(
                &mut self,
                slot: usize,
                first: usize,
                bytes: usize,
            ) -> Result<Self::Lease> {
                let handle = self.0.handles.get(slot).ok_or(ArenaError::Cleanup)?;
                let span = self.0.ledger.spans.get(slot).ok_or(ArenaError::Cleanup)?;
                if handle.size() != span.owned_bytes as u64
                    || handle.size_in_used() != span.owned_bytes as u64
                    || bytes == 0
                    || first
                        .checked_add(bytes)
                        .is_none_or(|end| end > span.owned_bytes)
                {
                    return Err(ArenaError::Cleanup);
                }
                self.0
                    .client
                    .read_private_cleanup(
                        handle
                            .clone()
                            .offset_start(first as u64)
                            .offset_end((span.owned_bytes - first - bytes) as u64),
                    )
                    .map(AuditBytes)
                    .map_err(|_| ArenaError::Cleanup)
            }
            fn cleanup_after_read(&mut self) -> Result<()> {
                match cubecl::future::block_on(self.0.client.sanitize_transient_io_audited()) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(ArenaError::Cleanup),
                }
            }
        }
        let counts = inspect_owned_zero(&mut Native(self), &self.ledger.spans)?;
        if counts.bytes != self.ledger.charged as u64 || counts.buffers != self.handles.len() as u64
        {
            return Err(ArenaError::Cleanup);
        }
        Ok(counts)
    }
    fn fail(&mut self) {
        self.retired.store(true, Ordering::Release);
        self.ledger.closed = true;
        // Keep handles/spans if cleanup fails so Drop can retry; never imply
        // that a failed erasure returned clean memory to the pool.
        if self.erase_and_drain().is_ok() && self.io_policy == IoPolicy::PublicFixture {
            self.handles.clear();
            self.ledger.close_after_erasure();
        }
    }
    /// Erase/release all buffers and invalidate every previous ID. The next
    /// allocation uses a new epoch. Reallocation is explicit and budgeted.
    pub fn reset(&mut self) -> Result<()> {
        self.active()?;
        if self.borrowed_roots {
            return Err(ArenaError::InvalidLayout);
        }
        if self.ledger.epoch == u64::MAX {
            self.fail();
            return Err(ArenaError::IdentityExhausted);
        }
        let accepted = accept_owned_zero_cleanup(&self.retired, self.zero_audit_counts, || {
            self.erase_and_verify()
        });
        let counts = match accepted {
            Ok(counts) => counts,
            Err(_) => {
                self.fail();
                return Err(ArenaError::Cleanup);
            }
        };
        self.ledger.reset_after_erasure()?;
        self.zero_audit_counts = counts;
        self.handles.clear();
        Ok(())
    }
    /// One-shot fixed-public model replay boundary only. The caller first
    /// proves a successfully completed healthy sequence, then reinitializes
    /// its family cursors ONLY after this complete owning-page audit. Unlike
    /// reset/close, no handle, epoch, MAIN charge or close receipt is changed.
    /// This is not a recovery operation for a poisoned or retired context.
    pub(crate) fn erase_retained_public_owners(&self) -> Result<OwnedZeroAudit> {
        self.active()?;
        if self.borrowed_roots || !self.public_owned_zero_audit_enabled() {
            return Err(ArenaError::Cleanup);
        }
        let accepted = accept_retained_public_erasure(
            &self.retired,
            OwnedZeroAudit {
                bytes: self.ledger.charged as u64,
                buffers: self.handles.len() as u64,
            },
            || self.erase_and_verify(),
        );
        match accepted {
            Ok(counts) => {
                // Deliberately separate from cumulative reset/close receipts:
                // these pages stay owned and will be inspected again at close.
                Ok(counts)
            }
            _ => {
                // Keep every tracked owner/ledger slot for terminal close. The
                // atomic retirement gate prevents further arena use immediately.
                self.retired.store(true, Ordering::Release);
                Err(ArenaError::Cleanup)
            }
        }
    }
    /// Idempotent after success. On failure the context remains retired and
    /// retained tracked handles allow a later close/Drop best-effort retry.
    pub fn close(&mut self) -> Result<()> {
        if self.closed_clean {
            return Ok(());
        }
        self.ledger.closed = true;
        if self.handles.is_empty() && self.io_policy == IoPolicy::PublicFixture {
            self.ledger.close_after_erasure();
            self.closed_clean = true;
            return Ok(());
        }
        let accepted = accept_owned_zero_cleanup(&self.retired, self.zero_audit_counts, || {
            self.erase_and_verify()
        });
        let counts = match accepted {
            Ok(counts) => counts,
            Err(_) => {
                self.retired.store(true, Ordering::Release);
                return Err(ArenaError::Cleanup);
            }
        };
        self.zero_audit_counts = counts;
        self.handles.clear();
        self.ledger.close_after_erasure();
        self.closed_clean = true;
        Ok(())
    }
}
impl Drop for PrivateArena {
    fn drop(&mut self) {
        let failed = self.close().is_err();
        // Failure is not permission to recycle unaudited pages. Retain the
        // bounded owning handles until process recovery; never issue clean
        // counts from Drop. This diagnostic deliberately quarantines them.
        let audited = self.strict_zero_audit();
        quarantine_failed_audit_handles(&mut self.handles, audited, failed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_complete_layout_has_exact_coverage_without_weakening_owned_bins() {
        for bytes in [
            MIN_SPAN_BYTES + 256,
            MIN_SPAN_BYTES * 2 - 256,
            MAX_SPAN_BYTES - 256,
        ] {
            let layout = BufferLayout::borrowed_complete(bytes).unwrap();
            assert_eq!(layout.owned_bytes, bytes);
            assert_eq!(layout.owned_words * 4, bytes);
            assert_eq!(layout.logical_words, layout.owned_words);
            assert!(
                (layout.workgroups_x as usize) * (layout.workgroups_y as usize) * LANES
                    >= layout.owned_words
            );
            let mut ledger = Ledger::new(bytes).unwrap();
            assert_eq!(
                ledger.reserve(layout).err(),
                Some(ArenaError::InvalidLayout)
            );
            let id = ledger.reserve_borrowed(layout).unwrap();
            assert_eq!(ledger.lookup(id).unwrap(), layout);
            assert_eq!(ledger.charged, bytes);
            assert_eq!(
                ledger.reserve_borrowed(layout).err(),
                Some(ArenaError::BudgetExceeded)
            );
            let mut forged = layout;
            forged.owned_words -= 1;
            assert_eq!(
                Ledger::new(bytes).unwrap().reserve_borrowed(forged).err(),
                Some(ArenaError::InvalidLayout)
            );
        }
        for bytes in [
            0,
            MIN_SPAN_BYTES - 256,
            MIN_SPAN_BYTES + 1,
            MAX_SPAN_BYTES + 256,
        ] {
            assert_eq!(
                BufferLayout::borrowed_complete(bytes),
                Err(ArenaError::InvalidLayout)
            );
        }
    }
    #[test]
    fn borrowed_irregular_root_erasure_audits_the_last_short_window_before_credit() {
        let mut good = fake_zero_audit("none", usize::MAX);
        good.spans = [
            MIN_SPAN_BYTES + 256,
            MIN_SPAN_BYTES + 512,
            MIN_SPAN_BYTES + 768,
        ]
        .map(|n| BufferLayout::borrowed_complete(n).unwrap())
        .to_vec();
        good.budget = 4096;
        let spans = good.spans.clone();
        let proof = inspect_owned_zero_window(&mut good, &spans, 4096).unwrap();
        assert_eq!(
            proof,
            OwnedZeroAudit {
                bytes: spans.iter().map(|s| s.owned_bytes as u64).sum(),
                buffers: 3
            }
        );
        assert_eq!(good.reads.len(), 27); // eight full windows + each irregular tail
        assert_eq!(good.cleanups, 27);
        for failure in ["tail", "tail map"] {
            let mut bad = fake_zero_audit(failure, 2);
            bad.spans = spans.clone();
            bad.budget = 4096;
            assert!(inspect_owned_zero_window(&mut bad, &spans, 4096).is_err());
            assert_eq!(bad.live.get(), 0);
            assert_eq!(bad.reads.len(), 27);
        }
    }
    #[test]
    fn retained_public_erasure_audits_full_padding_without_close_credit() {
        let retired = AtomicBool::new(false);
        let previous_close = OwnedZeroAudit {
            bytes: 17,
            buffers: 9,
        };
        let mut backend = fake_zero_audit("none", usize::MAX);
        let spans = backend.spans.clone();
        let expected = OwnedZeroAudit {
            bytes: spans.iter().map(|s| s.owned_bytes as u64).sum(),
            buffers: spans.len() as u64,
        };
        let accepted = accept_retained_public_erasure(&retired, expected, || {
            inspect_owned_zero(&mut backend, &spans)
        })
        .unwrap();
        assert_eq!(accepted, expected);
        assert_eq!(
            previous_close,
            OwnedZeroAudit {
                bytes: 17,
                buffers: 9
            }
        );
        assert!(!retired.load(Ordering::Acquire));
        assert_eq!(backend.reads, [0, 1, 2]);
        assert_eq!(backend.cleanups, 3);
        assert_eq!(backend.live.get(), 0);
        // The same retained pages are audited afresh, not cumulative close
        // credit that could mislabel a successful reset as released ownership.
        let replay = accept_retained_public_erasure(&retired, expected, || {
            inspect_owned_zero(&mut backend, &spans)
        })
        .unwrap();
        assert_eq!(replay, expected);
        assert_eq!(backend.cleanups, 6);
    }
    #[test]
    fn retained_erasure_retires_on_tail_map_cleanup_and_exact_inventory_failure() {
        for failure in [
            "padding",
            "logical",
            "short",
            "long",
            "map",
            "cleanup",
            "view panic",
            "cleanup panic",
        ] {
            let retired = AtomicBool::new(false);
            let mut backend = fake_zero_audit(failure, 1);
            let spans = backend.spans.clone();
            let expected = OwnedZeroAudit {
                bytes: spans.iter().map(|s| s.owned_bytes as u64).sum(),
                buffers: 3,
            };
            assert!(
                accept_retained_public_erasure(&retired, expected, || inspect_owned_zero(
                    &mut backend,
                    &spans
                ))
                .is_err()
            );
            assert!(retired.load(Ordering::Acquire));
            assert_eq!(backend.live.get(), 0);
        }
        for wrong in [
            OwnedZeroAudit {
                bytes: 4,
                buffers: 3,
            },
            OwnedZeroAudit {
                bytes: 8,
                buffers: 2,
            },
        ] {
            let retired = AtomicBool::new(false);
            assert!(
                accept_retained_public_erasure(
                    &retired,
                    OwnedZeroAudit {
                        bytes: 8,
                        buffers: 3
                    },
                    || Ok(wrong)
                )
                .is_err()
            );
            assert!(retired.load(Ordering::Acquire));
        }
    }
    struct FakeAuditLease {
        bytes: Vec<u8>,
        live: Rc<std::cell::Cell<usize>>,
        panic_on_view: bool,
    }
    impl AsRef<[u8]> for FakeAuditLease {
        fn as_ref(&self) -> &[u8] {
            assert!(!self.panic_on_view, "public synthetic inspection panic");
            &self.bytes
        }
    }
    impl Drop for FakeAuditLease {
        fn drop(&mut self) {
            self.bytes.zeroize();
            self.live.set(self.live.get() - 1);
        }
    }
    struct FakeZeroAudit {
        spans: Vec<BufferLayout>,
        live: Rc<std::cell::Cell<usize>>,
        reads: Vec<usize>,
        cleanups: usize,
        fail_at: usize,
        failure: &'static str,
        budget: usize,
    }
    impl OwnedZeroInspection for FakeZeroAudit {
        type Lease = FakeAuditLease;
        fn read_range(&mut self, slot: usize, first: usize, length: usize) -> Result<Self::Lease> {
            // These fake owned pages are retained throughout every read.
            assert_eq!(self.spans.len(), 3);
            assert_eq!(self.live.get(), 0); // no lease accumulation
            self.reads.push(slot);
            let span = self.spans[slot];
            let tail = first + length == span.owned_bytes;
            if slot == self.fail_at && tail && self.failure == "tail map" {
                return Err(ArenaError::Cleanup);
            }
            if length > self.budget
                || first
                    .checked_add(length)
                    .is_none_or(|end| end > span.owned_bytes)
                || (slot == self.fail_at && self.failure == "map")
            {
                return Err(ArenaError::Cleanup);
            }
            if slot == self.fail_at && self.failure == "read panic" {
                panic!("public synthetic map setup panic");
            }
            let mut bytes = vec![0; length];
            if slot == self.fail_at {
                match self.failure {
                    "padding" => bytes[length - 1] = 1,
                    "tail" if tail => bytes[length - 1] = 1,
                    "logical" => bytes[0] = 1,
                    "short" => {
                        bytes.pop();
                    }
                    "long" => bytes.push(0),
                    _ => (),
                }
            }
            self.live.set(1);
            Ok(FakeAuditLease {
                bytes,
                live: self.live.clone(),
                panic_on_view: slot == self.fail_at && self.failure == "view panic",
            })
        }
        fn cleanup_after_read(&mut self) -> Result<()> {
            assert_eq!(self.live.get(), 0); // host wipe + lease drop BEFORE audit
            self.cleanups += 1;
            if self.reads.last() == Some(&self.fail_at) {
                if self.failure == "cleanup" {
                    return Err(ArenaError::Cleanup);
                }
                if self.failure == "cleanup panic" {
                    panic!("public synthetic cleanup panic");
                }
            }
            Ok(())
        }
    }
    fn fake_zero_audit(failure: &'static str, fail_at: usize) -> FakeZeroAudit {
        FakeZeroAudit {
            spans: [1, 8193, 3]
                .map(|words| BufferLayout::new(words).unwrap())
                .to_vec(),
            live: Rc::new(std::cell::Cell::new(0)),
            reads: Vec::new(),
            cleanups: 0,
            fail_at,
            failure,
            budget: 2 * MIN_SPAN_BYTES,
        }
    }
    #[test]
    fn bounded_audit_windows_cover_complete_roots_and_refuse_late_tail_failure() {
        let mut good = fake_zero_audit("none", usize::MAX);
        good.budget = 4096;
        let spans = good.spans.clone();
        let counts = inspect_owned_zero_window(&mut good, &spans, 4096).unwrap();
        assert_eq!(counts.bytes, 4 * MIN_SPAN_BYTES as u64);
        assert_eq!(counts.buffers, 3);
        assert_eq!(good.reads.len(), 32);
        assert_eq!(good.cleanups, 32);
        assert_eq!(good.live.get(), 0);
        for failure in ["tail", "tail map"] {
            let mut bad = fake_zero_audit(failure, 1);
            bad.budget = 4096;
            let spans = bad.spans.clone();
            assert!(inspect_owned_zero_window(&mut bad, &spans, 4096).is_err());
            assert_eq!(bad.reads.len(), 24);
            assert_eq!(bad.cleanups, 24);
            assert_eq!(bad.live.get(), 0);
            assert_eq!(bad.reads.last(), Some(&1));
        }
    }
    #[test]
    fn full_owned_zero_audit_counts_padding_and_drops_each_lease_before_cleanup() {
        let mut backend = fake_zero_audit("none", usize::MAX);
        let spans = backend.spans.clone();
        let receipt = inspect_owned_zero(&mut backend, &spans).unwrap();
        assert_eq!(
            receipt,
            OwnedZeroAudit {
                bytes: 4 * MIN_SPAN_BYTES as u64,
                buffers: 3
            }
        );
        assert_eq!(backend.reads, [0, 1, 2]);
        assert_eq!(backend.cleanups, 3);
        assert_eq!(backend.live.get(), 0);
    }
    #[test]
    fn zero_audit_never_accepts_partial_padding_map_budget_error_or_panic() {
        for failure in [
            "padding",
            "logical",
            "short",
            "long",
            "map",
            "read panic",
            "view panic",
            "cleanup",
            "cleanup panic",
            "budget",
        ] {
            let mut backend = fake_zero_audit(failure, 1);
            if failure == "budget" {
                backend.budget = MIN_SPAN_BYTES;
            }
            let spans = backend.spans.clone();
            let receipt = inspect_owned_zero(&mut backend, &spans);
            assert_eq!(receipt, Err(ArenaError::Cleanup));
            assert_eq!(backend.reads, [0, 1]);
            assert_eq!(backend.cleanups, 2); // even rejected/missing leases
            assert_eq!(backend.live.get(), 0);
            assert_eq!(backend.spans, spans); // no owned page released on failure
        }
        assert_eq!(
            inspect_zero_span(vec![0; MIN_SPAN_BYTES - 1], MIN_SPAN_BYTES),
            Err(ArenaError::Cleanup)
        );
        assert_eq!(
            inspect_zero_span(Vec::<u8>::new(), 0),
            Err(ArenaError::Cleanup)
        );
    }
    #[test]
    fn failed_zero_cleanup_retires_and_never_promotes_prior_counts_or_retry() {
        for failure in ["zero", "panic", "overflow"] {
            let retired = AtomicBool::new(false);
            let previous = OwnedZeroAudit {
                bytes: if failure == "overflow" {
                    u64::MAX
                } else {
                    32768
                },
                buffers: 1,
            };
            let result = accept_owned_zero_cleanup(&retired, previous, || match failure {
                "zero" => Err(ArenaError::Cleanup),
                "panic" => panic!("public synthetic cleanup panic"),
                _ => Ok(OwnedZeroAudit {
                    bytes: 32768,
                    buffers: 1,
                }),
            });
            assert_eq!(result, Err(ArenaError::Cleanup));
            assert!(retired.load(Ordering::Acquire));
            assert!(
                accept_owned_zero_cleanup(&retired, previous, || Ok(OwnedZeroAudit::default()))
                    .is_err()
            );
        }
    }
    #[test]
    fn audit_counter_overflow_cannot_issue_partial_receipt() {
        let maximum = OwnedZeroAudit {
            bytes: u64::MAX,
            buffers: u64::MAX,
        };
        assert!(
            maximum
                .checked_add(OwnedZeroAudit {
                    bytes: 1,
                    buffers: 1
                })
                .is_err()
        );
        assert_eq!(
            maximum,
            OwnedZeroAudit {
                bytes: u64::MAX,
                buffers: u64::MAX
            }
        );
    }
    #[test]
    fn audit_failure_quarantine_retains_pages_while_ordinary_drop_releases() {
        for (audit, failed) in [(false, false), (false, true), (true, false), (true, true)] {
            let owner = Rc::new(());
            let mut handles = vec![owner.clone()];
            quarantine_failed_audit_handles(&mut handles, audit, failed);
            drop(handles);
            assert_eq!(
                Rc::strong_count(&owner),
                if audit && failed { 2 } else { 1 }
            );
        }
    }
    #[test]
    fn ordered_private_source_write_has_no_cleanup_deadline_or_fence_credit() {
        use std::cell::RefCell;
        for ordered in [false, true] {
            let events = RefCell::new(Vec::new());
            initialize_owned_upload(
                ordered,
                || {
                    events.borrow_mut().push("checkpoint");
                    Ok(())
                },
                || {
                    events.borrow_mut().push("owned full write");
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(
                *events.borrow(),
                if ordered {
                    vec!["owned full write"]
                } else {
                    vec!["checkpoint", "owned full write", "checkpoint"]
                }
            );
        }
        let checkpointed = std::cell::Cell::new(false);
        assert!(
            initialize_owned_upload(
                true,
                || {
                    checkpointed.set(true);
                    Ok(())
                },
                || Err(ArenaError::Initialization),
            )
            .is_err()
        );
        assert!(!checkpointed.get());
    }
    #[test]
    fn moved_upload_keeps_owned_destination_on_write_error_and_unwind() {
        use cubecl::bytes::AllocationController;
        use cubecl_runtime::transient::PrivateHostAllocation;
        let memory = Rc::new(());
        let handles = vec![memory.clone()];
        for panic in [false, true] {
            let upload = Upload::words(BufferLayout::new(2).unwrap(), &[7, 9]).unwrap();
            let pointer = upload.0.as_ptr();
            let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                write_owned_upload(&handles, 0, upload, |handle, data| {
                    assert_eq!(handles.len(), 1); // already retained during write
                    assert!(Rc::ptr_eq(&handle, &memory));
                    assert_eq!(data.as_ptr(), pointer); // original heap moved
                    let guarded = PrivateHostAllocation::from_vec(data);
                    assert_eq!(guarded.memory().as_ptr().cast::<u8>(), pointer);
                    if panic {
                        panic!("synthetic private write panic")
                    }
                    Err(ArenaError::Initialization)
                })
            }));
            assert!(!matches!(attempted, Ok(Ok(()))));
            assert_eq!(handles.len(), 1);
            assert!(Rc::ptr_eq(&handles[0], &memory));
            assert_eq!(Rc::strong_count(&memory), 2);
        }
    }

    #[test]
    fn transient_cleanup_checkpoints_before_uniforms_and_never_releases_on_fault() {
        struct Fake {
            events: Vec<(char, usize)>,
            checkpoints: usize,
            failure: usize,
        }
        impl TransientCleanup for Fake {
            fn checkpoint(&mut self) -> Result<()> {
                self.events.push(('c', self.checkpoints));
                self.checkpoints += 1;
                if self.failure == self.events.len() {
                    Err(ArenaError::Cleanup)
                } else {
                    Ok(())
                }
            }
            fn erase(&mut self, slot: usize) -> Result<()> {
                self.events.push(('e', slot));
                if self.failure == self.events.len() {
                    Err(ArenaError::Cleanup)
                } else {
                    Ok(())
                }
            }
        }
        for failure in 0..=6 {
            let mut ledger = Ledger::new(3 * MIN_SPAN_BYTES).unwrap();
            let ids: Vec<_> = (0..3)
                .map(|_| ledger.reserve(BufferLayout::new(1).unwrap()).unwrap())
                .collect();
            let mut backend = Fake {
                events: Vec::new(),
                checkpoints: 0,
                failure,
            };
            let result = cleanup_transient(&mut backend, 3, failure == 6);
            assert_eq!(
                backend.events,
                [('c', 0), ('e', 0), ('e', 1), ('e', 2), ('c', 1)]
            );
            if result.is_ok() {
                ledger.close_after_erasure();
            }
            assert_eq!(result.is_ok(), failure == 0);
            if failure != 0 {
                assert_eq!(ledger.charged, 3 * MIN_SPAN_BYTES);
                assert!(ids.into_iter().all(|id| ledger.lookup(id).is_ok()));
            } else {
                assert_eq!(ledger.charged, 0);
                assert!(ledger.closed);
            }
        }
    }
    #[test]
    fn required_pool_bins_make_each_owned_span_an_entire_underlying_page() {
        let cubecl::MemoryConfiguration::Custom { pool_options } = required_memory_configuration()
        else {
            panic!("expected explicit private allocator contract");
        };
        assert_eq!(
            pool_options.len(),
            if cfg!(feature = "strata-intel-core") {
                15
            } else {
                13
            }
        );
        for (i, pool) in pool_options.iter().enumerate() {
            assert_eq!(pool.dealloc_period, Some(1));
            let PoolType::ExclusivePages { max_alloc_size } = pool.pool_type else {
                panic!("subslice allocator hides page tails");
            };
            let expected = MIN_SPAN_BYTES << i;
            assert_eq!(max_alloc_size, expected as u64);
            let l = BufferLayout::new(expected / 4).unwrap();
            assert_eq!(l.owned_bytes, expected);
            // Earlier arbitrary public requests in this pool never exceed its
            // maximum, so its rolling mean and cached pages are <=maximum.
            // A private request equal to that maximum always owns the full page.
            for mean in [0.0, expected as f64 / 2.0, expected as f64] {
                assert_eq!((mean as u64).max(l.owned_bytes as u64), max_alloc_size);
            }
        }
        let first = pool_options.first().unwrap();
        let last = pool_options.last().unwrap();
        assert!(
            matches!(first.pool_type,PoolType::ExclusivePages{max_alloc_size} if max_alloc_size==MIN_SPAN_BYTES as u64)
        );
        assert!(
            matches!(last.pool_type,PoolType::ExclusivePages{max_alloc_size} if max_alloc_size==MAX_SPAN_BYTES as u64)
        );
    }
    #[test]
    fn layout_covers_every_padding_word_and_portable_grid() {
        for words in [
            1,
            8191,
            8192,
            8193,
            16384,
            16385,
            786432,
            MAX_SPAN_BYTES / 4,
        ] {
            let l = BufferLayout::new(words).unwrap();
            assert!(l.owned_bytes.is_power_of_two());
            assert!(l.owned_bytes >= MIN_SPAN_BYTES);
            assert!(l.owned_words >= words);
            assert_eq!(l.owned_words * 4, l.owned_bytes);
            assert!(l.workgroups_x <= 65_535);
            let invocations = l.workgroups_x as usize * l.workgroups_y as usize * LANES;
            assert!(invocations >= l.owned_words);
            // Eraser's index<len guard writes all owned words exactly once,
            // including owned_words-logical_words padding, and no words after.
            assert_eq!(
                (0..invocations).filter(|&i| i < l.owned_words).count(),
                l.owned_words
            );
        }
        assert!(BufferLayout::new(0).is_err());
        assert!(BufferLayout::new(usize::MAX).is_err());
        assert!(BufferLayout::new(MAX_SPAN_BYTES / 4 + 1).is_err());
        assert_eq!(BufferLayout::new(8193).unwrap().owned_bytes, 65536);
    }
    #[test]
    fn budget_charges_full_spans_and_rejection_is_transactional() {
        let mut l = Ledger::new(3 * MIN_SPAN_BYTES).unwrap();
        let first = l.reserve(BufferLayout::new(1).unwrap()).unwrap();
        let second = l.reserve(BufferLayout::new(8193).unwrap()).unwrap();
        assert_eq!(l.charged, 3 * MIN_SPAN_BYTES);
        assert!(matches!(
            l.reserve(BufferLayout::new(1).unwrap()),
            Err(ArenaError::BudgetExceeded)
        ));
        assert_eq!((l.spans.len(), l.charged), (2, 3 * MIN_SPAN_BYTES));
        assert!(l.lookup(first).is_ok());
        assert!(l.lookup(second).is_ok());
        l.rollback_last(second);
        assert_eq!((l.spans.len(), l.charged), (1, MIN_SPAN_BYTES));
        assert!(matches!(l.lookup(second), Err(ArenaError::StaleHandle)));
        assert!(matches!(
            Ledger::new(MIN_SPAN_BYTES - 1),
            Err(ArenaError::BudgetExceeded)
        ));
        let mut forged = BufferLayout::new(1).unwrap();
        forged.owned_bytes = 4;
        assert!(l.reserve(forged).is_err());
        assert_eq!(l.charged, MIN_SPAN_BYTES);
    }
    #[test]
    fn ids_bind_arena_epoch_and_close_even_when_slots_repeat() {
        let mut a = Ledger::new(MIN_SPAN_BYTES).unwrap();
        let mut b = Ledger::new(MIN_SPAN_BYTES).unwrap();
        let old = a.reserve(BufferLayout::new(1).unwrap()).unwrap();
        let foreign = b.reserve(BufferLayout::new(1).unwrap()).unwrap();
        assert!(matches!(a.lookup(foreign), Err(ArenaError::StaleHandle)));
        a.reset_after_erasure().unwrap();
        let fresh = a.reserve(BufferLayout::new(1).unwrap()).unwrap();
        assert!(old != fresh);
        assert!(matches!(a.lookup(old), Err(ArenaError::StaleHandle)));
        assert!(a.lookup(fresh).is_ok());
        a.close_after_erasure();
        assert!(matches!(a.lookup(fresh), Err(ArenaError::Closed)));
        assert!(matches!(
            a.reserve(BufferLayout::new(1).unwrap()),
            Err(ArenaError::Closed)
        ));
    }
    #[test]
    fn padded_upload_initializes_every_owned_byte() {
        let l = BufferLayout::new(3).unwrap();
        let f = Upload::floats(l, &[1.0, -2.0, 3.0]).unwrap();
        assert_eq!(&f.0[..4], &1.0_f32.to_le_bytes());
        assert!(f.0[12..].iter().all(|&b| b == 0));
        let w = Upload::words(l, &[7, 0xdeadbeef, u32::MAX]).unwrap();
        assert_eq!(&w.0[4..8], &0xdeadbeef_u32.to_le_bytes());
        assert!(w.0[12..].iter().all(|&b| b == 0));
        assert!(Upload::floats(l, &[1., f32::NAN, 2.]).is_err());
        assert!(Upload::words(l, &[1]).is_err());
    }
    #[test]
    fn reset_epoch_overflow_rejects_without_mutating_ledger() {
        let mut l = Ledger::new(MIN_SPAN_BYTES).unwrap();
        l.epoch = u64::MAX;
        let id = l.reserve(BufferLayout::new(1).unwrap()).unwrap();
        assert_eq!(l.reset_after_erasure(), Err(ArenaError::IdentityExhausted));
        assert!(l.lookup(id).is_ok());
        assert_eq!(l.charged, MIN_SPAN_BYTES);
    }
    #[test]
    fn erasure_kernel_lowers_to_wgsl_offline() {
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, WgpuCompilationOptions};
        let mut b = KernelBuilder::default();
        b.runtime_properties(WgpuRuntime::target_properties());
        AddressType::U32.register(&mut b.scope);
        let u = u32::as_type(&b.scope);
        let output = b.output_array(u).into();
        erase_full_span::expand(&mut b.scope, output);
        let definition = b.build(
            KernelSettings::default()
                .cube_dim(CubeDim::new_1d(64))
                .kernel_name("erase_full_span"),
        );
        assert_eq!(definition.buffers.len(), 1);
        let shader = WgslCompiler::default()
            .compile(
                definition,
                &WgpuCompilationOptions::default(),
                ExecutionMode::Checked,
                AddressType::U32.unsigned_type(),
            )
            .unwrap();
        let source = shader.to_string();
        assert!(source.contains("fn erase_full_span"));
        assert!(!source.contains("enable f16"));
    }
}
