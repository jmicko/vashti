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
//! Runtime-owned upload/staging/readback copies are outside this owner and
//! still require a separate privacy audit. No content or backend error payload
//! is logged. No device initialization occurs here or in host tests.

use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};

pub const MIN_SPAN_BYTES: usize = 32 * 1024;
pub const MAX_SPAN_BYTES: usize = 128 * 1024 * 1024;
const LANES: usize = 64;
static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Required allocator configuration at Wgpu init_device, BEFORE constructing
/// a client or any public/private allocations on that context. Requests that
/// exactly equal a bin maximum cannot acquire a larger hidden page: CubeCL
/// exclusive pools allocate max(request, rolling_mean)<=this bin maximum.
/// This constructor cannot inspect an existing client's allocator config.
pub fn required_memory_configuration() -> cubecl::MemoryConfiguration {
    cubecl::MemoryConfiguration::Custom {
        pool_options: (15..=27)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferLayout {
    pub logical_words: usize,
    pub owned_words: usize,
    pub owned_bytes: usize,
    pub workgroups_x: u32,
    pub workgroups_y: u32,
}

impl BufferLayout {
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
        // MAX_SPAN_BYTES guarantees each buffer and the flattened dispatch
        // position fit 32-bit indexing and portable storage binding limits.
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

pub struct PrivateArena {
    client: ComputeClient<WgpuRuntime>,
    ledger: Ledger,
    handles: Vec<Handle>,
    retired: Arc<AtomicBool>,
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
            _same_thread: PhantomData,
        })
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
        // Allocate metadata capacity before any device operation. Charge the
        // complete span before allocating; never publish an uninitialized ID.
        self.handles
            .try_reserve(1)
            .map_err(|_| ArenaError::HostAllocation)?;
        let id = self.ledger.reserve(layout)?;
        let allocated =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match upload.as_ref() {
                Some(bytes) => self.client.create_from_slice(&bytes.0),
                None => self.client.empty(layout.owned_bytes),
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
        let initialized = if upload.is_some() {
            sync(&self.client)
        } else {
            self.queue_erase(id.slot).and_then(|()| sync(&self.client))
        };
        if initialized.is_err() {
            self.fail();
            return Err(ArenaError::Initialization);
        }
        Ok(id)
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
    fn fail(&mut self) {
        self.retired.store(true, Ordering::Release);
        self.ledger.closed = true;
        // Keep handles/spans if cleanup fails so Drop can retry; never imply
        // that a failed erasure returned clean memory to the pool.
        if self.erase_and_drain().is_ok() {
            self.handles.clear();
            self.ledger.close_after_erasure();
        }
    }
    /// Erase/release all buffers and invalidate every previous ID. The next
    /// allocation uses a new epoch. Reallocation is explicit and budgeted.
    pub fn reset(&mut self) -> Result<()> {
        self.active()?;
        if self.ledger.epoch == u64::MAX {
            self.fail();
            return Err(ArenaError::IdentityExhausted);
        }
        if self.erase_and_drain().is_err() {
            self.fail();
            return Err(ArenaError::Cleanup);
        }
        self.handles.clear();
        self.ledger.reset_after_erasure()
    }
    /// Idempotent after success. On failure the context remains retired and
    /// retained tracked handles allow a later close/Drop best-effort retry.
    pub fn close(&mut self) -> Result<()> {
        self.ledger.closed = true;
        if self.handles.is_empty() {
            self.ledger.close_after_erasure();
            return Ok(());
        }
        if self.erase_and_drain().is_err() {
            self.retired.store(true, Ordering::Release);
            return Err(ArenaError::Cleanup);
        }
        self.handles.clear();
        self.ledger.close_after_erasure();
        Ok(())
    }
}
impl Drop for PrivateArena {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn required_pool_bins_make_each_owned_span_an_entire_underlying_page() {
        let cubecl::MemoryConfiguration::Custom { pool_options } = required_memory_configuration()
        else {
            panic!("expected explicit private allocator contract");
        };
        assert_eq!(pool_options.len(), 13);
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
