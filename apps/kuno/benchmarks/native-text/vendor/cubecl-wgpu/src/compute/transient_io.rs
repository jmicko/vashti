// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
//! Opt-in bounded, explicitly owned transient I/O. No device creation in tests.
//!
//! Private uploads NEVER use Queue::write_buffer[_with]. MAP_WRITE|COPY_SRC
//! mapped_at_creation buffers avoid wgpu-core's implicit upload staging path.
//! Every direct buffer is initialized over its complete API-visible span and
//! retained until checked sanitization. Private uniforms/readbacks bypass the
//! CubeCL pools, so retired/oversized hidden pool pages do not escape tracking.
//! Public main weight buffers and ordinary PUBLIC uploads remain separate.
//!
//! Returned readbacks are moved into wiping heap controllers (logical bytes
//! only), with leases held across futures AND returned Bytes. Future Drop aborts
//! pending mapping via unmap. Sanitization rejects live leases, drains first,
//! remaps/volatile-wipes full upload spans then unmaps to flush writes; clears
//! full uniform/readback spans; awaits completion/errors before releasing.
//! Failure retains buffers and permanently retires this stream. Owner MUST
//! retire the shared context, poison its sequence and perform its main-span
//! cleanup. A Drop attempt is best-effort, never an erasure receipt.
//!
//! API-visible erasure is not proof of opaque driver/physical-allocation padding,
//! CPU swap/coredump erasure or preceding caller copies. No GPU test is claimed.
//! Only serialized native streams are supported; WASM refuses this experiment.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::WgpuResource;
use cubecl_common::backtrace::BackTrace;
use cubecl_common::bytes::{AllocationController, AllocationProperty, Bytes};
use cubecl_runtime::server::ServerError;
use cubecl_runtime::transient::{PrivateHostAllocation, TransientIoReceipt};

// Dedicated buffers do not use the private arena's 32-KiB pool bins.
// Public 256-byte minimum satisfies native mapping/copy/uniform alignment;
// erasure covers the FULL API span, never opaque driver-allocation padding.
const MIN_BYTES: u64 = 256;
const MAX_BYTES: u64 = 128 * 1024 * 1024;
const CLEAN_READBACK_BYTES: u64 = 8 * 1024 * 1024;
const CLEAN_READBACK_COUNT: usize = 64;
static READBACK_REUSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Numeric fixed-public laboratory evidence; never exposes buffer or content data.
pub fn transient_readback_reuses() -> u64 {
    READBACK_REUSES.load(Ordering::Relaxed)
}
fn configured_readback_reuse(raw: Option<&str>, batch: bool) -> Result<bool, ServerError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if batch => Ok(true),
        _ => Err(fault("readback reuse requires explicit batched INFO")),
    }
}
static GPU_UPLOAD_CLEARS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Fixed-public numeric laboratory evidence only.
pub fn transient_gpu_upload_clears() -> u64 {
    GPU_UPLOAD_CLEARS.load(Ordering::Relaxed)
}
fn configured_gpu_upload_clear(
    raw: Option<&str>,
    batch: bool,
    primary: bool,
) -> Result<bool, ServerError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if batch && primary => Ok(true),
        _ => Err(fault(
            "GPU upload clear requires batched INFO and primary mapping",
        )),
    }
}
fn needs_audit_stage(kind: Kind, gpu_upload_clear: bool) -> bool {
    kind == Kind::Uniform || kind == Kind::Upload && !gpu_upload_clear
}
fn allocation_usage(kind: Kind, gpu_upload_clear: bool) -> wgpu::BufferUsages {
    match kind {
        Kind::Upload => {
            wgpu::BufferUsages::MAP_WRITE
                | wgpu::BufferUsages::COPY_SRC
                | if gpu_upload_clear {
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::empty()
                }
        }
        Kind::Uniform => {
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
        }
        Kind::Readback => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    }
}
static INLINE_READ_BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Content-free fixed-public laboratory counter.
pub fn transient_inline_readback_batches() -> u64 {
    INLINE_READ_BATCHES.load(Ordering::Relaxed)
}
fn configured_inline_readback(
    raw: Option<&str>,
    batch: bool,
    inline: bool,
) -> Result<bool, ServerError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if batch && inline => Ok(true),
        _ => Err(fault(
            "inline readback requires batched INFO and inline dispatch",
        )),
    }
}
const POLL_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) type ErrorFuture = Pin<Box<dyn Future<Output = Option<wgpu::Error>> + Send>>;

pub(crate) fn fault(reason: &'static str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Upload,
    Uniform,
    Readback,
}
impl Kind {
    fn slot(self) -> usize {
        match self {
            Self::Upload => 0,
            Self::Uniform => 1,
            Self::Readback => 2,
        }
    }
}

fn span(logical: u64) -> Result<u64, ServerError> {
    if logical == 0 {
        return Err(fault("private I/O empty buffer"));
    }
    let owned = logical
        .max(MIN_BYTES)
        .checked_next_power_of_two()
        .ok_or_else(|| fault("private I/O span overflow"))?;
    if owned > MAX_BYTES {
        return Err(fault("private I/O buffer exceeds bound"));
    }
    Ok(owned)
}

// Public laboratory opt-in. The default keeps the existing per-INFO owners.
// INFO is Storage/read in WGSL and can be Uniform in other representations.
// Use a checked common device alignment/extent without compiler-key changes.
fn configured_info_slab(raw: Option<&str>) -> Result<Option<u64>, ServerError> {
    match raw {
        None | Some("0") => Ok(None),
        Some("4096") => Ok(Some(4096)),
        Some("8192") => Ok(Some(8192)),
        _ => Err(fault("private INFO slab requires 0, 4096 or 8192")),
    }
}
fn configured_info_batch(raw: Option<&str>, slab: Option<u64>) -> Result<bool, ServerError> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if slab.is_some() => Ok(true),
        _ => Err(fault(
            "batched private INFO requires explicit compact slabs",
        )),
    }
}
#[derive(Clone, Copy)]
struct InfoPolicy {
    slab_bytes: u64,
    alignment: u64,
    binding_limit: u64,
}
impl InfoPolicy {
    fn new(
        slab_bytes: u64,
        storage_alignment: u64,
        uniform_alignment: u64,
        storage_limit: u64,
        uniform_limit: u64,
    ) -> Result<Self, ServerError> {
        if !matches!(slab_bytes, 4096 | 8192)
            || !storage_alignment.is_power_of_two()
            || !uniform_alignment.is_power_of_two()
            || storage_alignment < 4
            || uniform_alignment < 4
            || storage_limit == 0
            || uniform_limit == 0
        {
            return Err(fault("private INFO slab device geometry"));
        }
        let alignment = storage_alignment.max(uniform_alignment);
        let binding_limit = storage_limit.min(uniform_limit).min(MAX_BYTES);
        if alignment > slab_bytes || binding_limit < 4 {
            return Err(fault("private INFO slab device alignment/extent"));
        }
        Ok(Self {
            slab_bytes,
            alignment,
            binding_limit,
        })
    }
    fn plan(
        self,
        logical: u64,
        current: Option<(u64, u64)>, // (FULL owner bytes, private append cursor)
    ) -> Result<InfoPlan, ServerError> {
        if logical == 0 || !logical.is_multiple_of(4) || logical > self.binding_limit {
            return Err(fault("private INFO slab logical binding extent"));
        }
        let source_bytes = span(logical)?;
        if logical > self.slab_bytes {
            return Ok(InfoPlan {
                logical,
                source_bytes,
                target_bytes: span(logical)?,
                offset: 0,
                next_cursor: 0,
                existing: false,
                slab: false,
            });
        }
        if let Some((owned, cursor)) = current {
            if owned != self.slab_bytes || cursor > owned {
                return Err(fault("private INFO slab owner/cursor mismatch"));
            }
            let offset = cursor
                .checked_add(self.alignment - 1)
                .map(|n| n / self.alignment * self.alignment)
                .ok_or_else(|| fault("private INFO slab offset overflow"))?;
            let end = offset
                .checked_add(logical)
                .ok_or_else(|| fault("private INFO slab record overflow"))?;
            if end <= owned {
                return Ok(InfoPlan {
                    logical,
                    source_bytes,
                    target_bytes: 0,
                    offset,
                    next_cursor: end,
                    existing: true,
                    slab: true,
                });
            }
        }
        Ok(InfoPlan {
            logical,
            source_bytes,
            target_bytes: self.slab_bytes,
            offset: 0,
            next_cursor: logical,
            existing: false,
            slab: true,
        })
    }
}
struct InfoPlan {
    logical: u64,
    source_bytes: u64,
    target_bytes: u64, // Zero only when reusing an append region in this epoch.
    offset: u64,
    next_cursor: u64,
    existing: bool,
    slab: bool,
}
struct InfoSlab {
    buffer: wgpu::Buffer,
    source: Option<wgpu::Buffer>,
    owner_index: usize,
    cursor: u64,
    epoch: u64,
}
fn info_epoch(owner: u64, current: u64) -> Result<(), ServerError> {
    if owner == 0 || owner != current {
        return Err(fault("private INFO slab epoch mismatch"));
    }
    Ok(())
}
impl Drop for InfoSlab {
    fn drop(&mut self) {
        // Cursor is request-derived host bookkeeping, not a persistent key.
        // SAFETY: exclusively owned fields, with no references surviving Drop.
        unsafe {
            core::ptr::addr_of_mut!(self.owner_index).write_volatile(0);
            core::ptr::addr_of_mut!(self.cursor).write_volatile(0);
            core::ptr::addr_of_mut!(self.epoch).write_volatile(0);
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}

/// Validate full aligned copy bounds against the actual API-visible backing
/// buffer before allocation/command scheduling, including offset overflow.
fn copy_bounds(size: u64, offset: u64, backing: u64, logical: usize) -> Result<u64, ServerError> {
    let copy_len = size
        .checked_add(3)
        .map(|n| n / 4 * 4)
        .ok_or_else(|| fault("private I/O copy overflow"))?;
    if copy_len == 0
        || logical as u64 > copy_len
        || !offset.is_multiple_of(4)
        || offset.checked_add(copy_len).is_none_or(|end| end > backing)
    {
        return Err(fault("private I/O copy bounds/alignment"));
    }
    Ok(copy_len)
}

/// Cleanup-only read must cover exactly the full underlying API-visible
/// page, including any allocator padding. Interior/logical views never certify.
pub(crate) fn cleanup_read_bounds(offset: u64, size: u64, backing: u64) -> Result<(), ServerError> {
    if offset != 0 || size == 0 || size != backing || !size.is_multiple_of(4) {
        return Err(fault("private cleanup read has hidden page tail"));
    }
    Ok(())
}
#[derive(Debug)]
struct Ledger {
    budget: u64,
    charged: u64,
    clean_bytes: u64,
    totals: [u64; 3],
    epoch: u64,
}
impl Ledger {
    fn new(budget: u64) -> Result<Self, ServerError> {
        if budget < MIN_BYTES {
            return Err(fault("private I/O budget below minimum"));
        }
        Ok(Self {
            budget,
            charged: 0,
            clean_bytes: 0,
            totals: [0; 3],
            epoch: 1,
        })
    }
    fn check(&self, bytes: u64) -> Result<(), ServerError> {
        if self
            .charged
            .checked_add(self.clean_bytes)
            .and_then(|n| n.checked_add(bytes))
            .is_none_or(|n| n > self.budget)
        {
            return Err(fault("private I/O budget exhausted; owner must checkpoint"));
        }
        Ok(())
    }
    fn reserve(&mut self, kind: Kind, logical: u64) -> Result<u64, ServerError> {
        let bytes = span(logical)?;
        self.check(bytes)?;
        let total = self.totals[kind.slot()]
            .checked_add(bytes)
            .ok_or_else(|| fault("private I/O counter overflow"))?;
        self.charged += bytes;
        self.totals[kind.slot()] = total;
        Ok(bytes)
    }
    fn retain_clean(&mut self, bytes: u64) -> Result<(), ServerError> {
        self.check(bytes)?;
        self.clean_bytes = self
            .clean_bytes
            .checked_add(bytes)
            .ok_or_else(|| fault("clean readback charge overflow"))?;
        Ok(())
    }
    fn activate_clean_readback(&mut self, bytes: u64) -> Result<(), ServerError> {
        if span(bytes)? != bytes || self.clean_bytes < bytes {
            return Err(fault("clean readback charge mismatch"));
        }
        self.check(0)?;
        let charged = self
            .charged
            .checked_add(bytes)
            .ok_or_else(|| fault("clean readback active charge overflow"))?;
        let total = self.totals[Kind::Readback.slot()]
            .checked_add(bytes)
            .ok_or_else(|| fault("clean readback counter overflow"))?;
        // Move exact already-charged ownership; never credit or double-charge it.
        self.clean_bytes -= bytes;
        self.charged = charged;
        self.totals[Kind::Readback.slot()] = total;
        Ok(())
    }
    /// Opt-in slabs reserve all full Upload/Uniform audit duplicates before
    /// normal mutation. Readback creation must preserve that same headroom.
    /// Audit-stage allocation consumes it with ordinary reserve(), never adds
    /// another recursive audit reservation or tries to checkpoint itself.
    fn check_info_audit_group(
        &self,
        upload: u64,
        uniform: u64,
        readback: u64,
    ) -> Result<(), ServerError> {
        let extra = upload
            .checked_add(uniform)
            .and_then(|n| n.checked_add(readback))
            .ok_or_else(|| fault("private INFO slab allocation group overflow"))?;
        let audit = self.totals[Kind::Upload.slot()]
            .checked_add(self.totals[Kind::Uniform.slot()])
            .and_then(|n| n.checked_add(upload))
            .and_then(|n| n.checked_add(uniform))
            .ok_or_else(|| fault("private INFO slab audit headroom overflow"))?;
        self.check(
            extra
                .checked_add(audit)
                .ok_or_else(|| fault("private INFO slab group overflow"))?,
        )
    }
    fn after_erasure(&mut self) -> Result<TransientIoReceipt, ServerError> {
        let next = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| fault("private I/O epoch exhausted"))?;
        let receipt = TransientIoReceipt {
            epoch: self.epoch,
            upload_bytes: self.totals[0],
            uniform_bytes: self.totals[1],
            readback_bytes: self.totals[2],
            inspected_bytes: 0,
            inspected_buffers: 0,
        };
        self.epoch = next;
        self.charged = 0;
        self.totals = [0; 3];
        Ok(receipt)
    }
}

// Diagnostic codes contain no backend descriptions, values, offsets or shader
// text. First OBSERVED failure wins, including failures in async lease owners.
#[derive(Clone, Copy)]
#[repr(usize)]
enum Failure {
    HeldLease,
    MutationAbandoned,
    LeaseOverflow,
    ReadbackMap,
    ReadbackConversion,
    Unmap,
    PriorStreamError,
    ContextRetired,
    DrainTimeout,
    PollIndex,
    UploadMap,
    UploadWipe,
    DestinationClear,
    QueueSubmit,
    GpuOutOfMemory,
    GpuValidation,
    GpuInternal,
    AuditBudget,
    AuditAllocation,
    AuditMap,
    AuditNonzero,
    AuditBookkeeping,
    Panic,
    EpochOverflow,
    AllocationBudget,
    AllocationMetadata,
    AllocationSpan,
    CleanupRejected,
    UploadMapRejected,
    UploadMapDisconnected,
    UploadMapCallbackTimeout,
    AuditMapRejected,
    AuditMapDisconnected,
    AuditMapCallbackTimeout,
    CleanupReadMapRejected,
    CleanupReadMapDisconnected,
    CleanupReadMapCallbackTimeout,
}
const FAILURE_CODES: [&str; 37] = [
    "kuno_private_io_failure:held_lease",
    "kuno_private_io_failure:mutation_abandoned",
    "kuno_private_io_failure:lease_overflow",
    "kuno_private_io_failure:readback_map",
    "kuno_private_io_failure:readback_conversion",
    "kuno_private_io_failure:unmap",
    "kuno_private_io_failure:prior_stream_error",
    "kuno_private_io_failure:context_retired",
    "kuno_private_io_failure:drain_timeout",
    "kuno_private_io_failure:poll_submission_index",
    "kuno_private_io_failure:upload_map",
    "kuno_private_io_failure:upload_wipe",
    "kuno_private_io_failure:destination_clear",
    "kuno_private_io_failure:queue_submit",
    "kuno_private_io_failure:gpu_out_of_memory",
    "kuno_private_io_failure:gpu_validation",
    "kuno_private_io_failure:gpu_internal",
    "kuno_private_io_failure:audit_budget",
    "kuno_private_io_failure:audit_allocation",
    "kuno_private_io_failure:audit_map",
    "kuno_private_io_failure:audit_nonzero",
    "kuno_private_io_failure:audit_bookkeeping",
    "kuno_private_io_failure:panic",
    "kuno_private_io_failure:epoch_overflow",
    "kuno_private_io_failure:allocation_budget",
    "kuno_private_io_failure:allocation_metadata",
    "kuno_private_io_failure:allocation_span",
    "kuno_private_io_failure:cleanup_rejected",
    "kuno_private_io_failure:upload_map_rejected",
    "kuno_private_io_failure:upload_map_disconnected",
    "kuno_private_io_failure:upload_map_callback_timeout",
    "kuno_private_io_failure:audit_map_rejected",
    "kuno_private_io_failure:audit_map_disconnected",
    "kuno_private_io_failure:audit_map_callback_timeout",
    "kuno_private_io_failure:cleanup_read_map_rejected",
    "kuno_private_io_failure:cleanup_read_map_disconnected",
    "kuno_private_io_failure:cleanup_read_map_callback_timeout",
];
impl Failure {
    fn code(self) -> &'static str {
        FAILURE_CODES[self as usize]
    }
}
fn gpu_failure(error: &wgpu::Error) -> Failure {
    match error {
        wgpu::Error::OutOfMemory { .. } => Failure::GpuOutOfMemory,
        wgpu::Error::Validation { .. } => Failure::GpuValidation,
        wgpu::Error::Internal { .. } => Failure::GpuInternal,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapCallbackFailure {
    Rejected,
    Disconnected,
    Timeout,
}
impl MapCallbackFailure {
    fn upload(self) -> Failure {
        match self {
            Self::Rejected => Failure::UploadMapRejected,
            Self::Disconnected => Failure::UploadMapDisconnected,
            Self::Timeout => Failure::UploadMapCallbackTimeout,
        }
    }
    fn audit(self) -> Failure {
        match self {
            Self::Rejected => Failure::AuditMapRejected,
            Self::Disconnected => Failure::AuditMapDisconnected,
            Self::Timeout => Failure::AuditMapCallbackTimeout,
        }
    }
    fn cleanup_read(self) -> Failure {
        match self {
            Self::Rejected => Failure::CleanupReadMapRejected,
            Self::Disconnected => Failure::CleanupReadMapDisconnected,
            Self::Timeout => Failure::CleanupReadMapCallbackTimeout,
        }
    }
}

fn remaining_map_time(deadline: Instant) -> Option<Duration> {
    remaining_map_time_at(deadline, Instant::now())
}
fn remaining_map_time_at(deadline: Instant, now: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(now)
        .filter(|remaining| !remaining.is_zero())
}

/// A successful poll is not a delivered mapping callback: another native poll
/// can claim the callback under the queue lock and fire it after releasing that
/// lock. Wait for delivery within the SAME deadline used for the native poll.
/// The unit BufferAsyncError has no typed cause to inspect; never format it.
fn await_map_callback(
    receiver: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    deadline: Instant,
) -> Result<(), MapCallbackFailure> {
    let remaining = remaining_map_time(deadline).ok_or(MapCallbackFailure::Timeout)?;
    let result = receiver.recv_timeout(remaining);
    // Reject even a queued success if scheduling resumed after the deadline.
    // Full mapping acceptance, not only the blocking wait, is time-bounded.
    if remaining_map_time(deadline).is_none() {
        return Err(MapCallbackFailure::Timeout);
    }
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(MapCallbackFailure::Rejected),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err(MapCallbackFailure::Disconnected)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(MapCallbackFailure::Timeout),
    }
}
pub(crate) fn held_lease_fault() -> ServerError {
    fault(Failure::HeldLease.code())
}
pub(crate) fn cleanup_panic_fault() -> ServerError {
    fault(Failure::Panic.code())
}

#[derive(Debug, Default)]
struct LeaseTracker {
    count: AtomicUsize,
    retired: AtomicBool,
    first_failure: AtomicUsize,
}
impl LeaseTracker {
    fn record(&self, category: Failure) {
        let _ = self.first_failure.compare_exchange(
            0,
            category as usize + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
    fn reason(&self, fallback: Failure) -> &'static str {
        self.first_failure
            .load(Ordering::Acquire)
            .checked_sub(1)
            .and_then(|i| FAILURE_CODES.get(i))
            .copied()
            .unwrap_or(fallback.code())
    }
    fn retire_with(&self, category: Failure) {
        self.record(category);
        self.retired.store(true, Ordering::Release);
    }
}
struct LeaseTicket(Arc<LeaseTracker>);
/// A synchronous configured-private operation must either complete or retire
/// the ledger. This guard reaches retirement even when the channel catches a
/// backend panic and continues serving the same context.
pub(crate) struct PrivateMutationGuard {
    tracker: Arc<LeaseTracker>,
    accepted: bool,
}
impl PrivateMutationGuard {
    pub(crate) fn accept(mut self) {
        self.accepted = true;
    }
}
impl Drop for PrivateMutationGuard {
    fn drop(&mut self) {
        if !self.accepted {
            self.tracker.retire_with(Failure::MutationAbandoned);
        }
    }
}
impl LeaseTicket {
    fn new(tracker: Arc<LeaseTracker>) -> Result<Self, ServerError> {
        tracker
            .count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| {
                tracker.record(Failure::LeaseOverflow);
                fault(Failure::LeaseOverflow.code())
            })?;
        Ok(Self(tracker))
    }
}
impl Drop for LeaseTicket {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct ReadbackLease {
    buffer: wgpu::Buffer,
    ticket: LeaseTicket,
    unmap: bool,
}
impl ReadbackLease {
    fn unmap(&mut self) {
        self.buffer.unmap();
        self.unmap = false;
    }
    fn retire(&self) {
        self.ticket.0.retire_with(Failure::ReadbackMap);
    }
}
impl Drop for ReadbackLease {
    fn drop(&mut self) {
        if self.unmap
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.buffer.unmap()))
                .is_err()
        {
            self.ticket.0.retire_with(Failure::Unmap);
        }
        // ticket drops AFTER pending-map cancellation/unmap is attempted.
    }
}

struct ReadbackController {
    data: PrivateHostAllocation,
    _lease: ReadbackLease,
}
impl AllocationController for ReadbackController {
    fn alloc_align(&self) -> usize {
        1
    }
    fn property(&self) -> AllocationProperty {
        AllocationProperty::Other
    }
    unsafe fn memory_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        // SAFETY: delegated exclusively owned initialized heap allocation.
        unsafe { self.data.memory_mut() }
    }
    fn memory(&self) -> &[std::mem::MaybeUninit<u8>] {
        self.data.memory()
    }
}

pub(crate) struct PendingReadback {
    pub buffer: wgpu::Buffer,
    lease: ReadbackLease,
    logical: usize,
}
impl PendingReadback {
    pub fn fail(&self) {
        self.lease.retire();
    }
    /// Called only after map_async succeeded. Copy directly into a wiping heap
    /// controller; no ordinary Vec/to_vec request copy exists on this path.
    pub fn into_bytes(mut self) -> Result<Bytes, ServerError> {
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let view = self.buffer.get_mapped_range(..);
            if self.logical > view.len() {
                return Err(fault("private readback logical bounds"));
            }
            let data = PrivateHostAllocation::copied(&view[..self.logical])
                .map_err(|_| fault("private readback host allocation failed"))?;
            drop(view);
            self.lease.unmap();
            Ok(data)
        }));
        match attempted {
            Ok(Ok(data)) => {
                let controller = Box::new(ReadbackController {
                    data,
                    _lease: self.lease,
                });
                // SAFETY: logical bytes are initialized and controller owns them.
                Ok(unsafe { Bytes::from_controller(controller, self.logical) })
            }
            _ => {
                self.lease.ticket.0.record(Failure::ReadbackConversion);
                self.fail();
                Err(fault(
                    self.lease.ticket.0.reason(Failure::ReadbackConversion),
                ))
            }
        }
    }
}

struct Tracked<B = wgpu::Buffer> {
    buffer: B,
    kind: Kind,
    // A completed scope fence may authorize reuse, never ownership release.
    reuse_ready: bool,
    next_free: usize,
    // A cached exact INFO record pins the entire page until the full audit.
    info_pinned: bool,
    // Includes unused remaps retained across checkpoints. This obligation
    // survives selection and partial producers until the actual unmap succeeds.
    upload_mapped: bool,
}
const REUSE_METADATA_MAX: usize = 256 * 1024;
static NEXT_REUSE_NONCE: AtomicU64 = AtomicU64::new(1);
static REQUEST_IO_REUSES: AtomicU64 = AtomicU64::new(0);
/// Bounded numeric laboratory diagnostic; no per-request trace or content.
pub fn transient_request_io_reuses() -> u64 {
    REQUEST_IO_REUSES.load(Ordering::Relaxed)
}
#[derive(Default)]
struct ReuseScope {
    nonce: Option<u64>,
    free: ReuseIndex,
}
impl ReuseScope {
    fn begin(&mut self, empty: bool, live_leases: usize) -> Result<u64, ServerError> {
        if self.nonce.is_some() || !empty || live_leases != 0 {
            return Err(fault("private reuse begin requires empty exclusive owner"));
        }
        let nonce = NEXT_REUSE_NONCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| fault("private reuse nonce exhausted"))?;
        self.nonce = Some(nonce);
        Ok(nonce)
    }
    fn check(&self, nonce: u64) -> Result<(), ServerError> {
        if nonce == 0 || self.nonce != Some(nonce) {
            Err(fault("private reuse owner mismatch"))
        } else {
            Ok(())
        }
    }
    fn finish(&mut self, nonce: u64, empty: bool, live_leases: usize) -> Result<(), ServerError> {
        self.check(nonce)?;
        if !empty || live_leases != 0 {
            return Err(fault("private reuse end still owns dirty buffers"));
        }
        self.nonce = None;
        Ok(())
    }
}
/// Accept-only checkpoints cannot publish readiness: they have no mutable
/// buffer/index/ledger access. Only the caller's completed-error drain runs.
fn accept_only_checkpoint(
    tracker: &LeaseTracker,
    owner_nonce: Option<u64>,
    nonce: u64,
    live_leases: usize,
    pending_info: bool,
    completed: impl FnOnce() -> Result<(), ServerError>,
) -> Result<(), ServerError> {
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if tracker.retired.load(Ordering::Acquire) {
            return Err(fault(tracker.reason(Failure::ContextRetired)));
        }
        if nonce == 0 || owner_nonce != Some(nonce) {
            return Err(fault("private reuse owner mismatch"));
        }
        if live_leases != 0 || pending_info {
            return Err(fault("private reuse fence has outstanding owner"));
        }
        completed()?;
        if tracker.retired.load(Ordering::Acquire) {
            return Err(fault(tracker.reason(Failure::ContextRetired)));
        }
        Ok(())
    }));
    match attempted {
        Ok(Ok(())) => Ok(()),
        _ => {
            tracker.retire_with(Failure::CleanupRejected);
            Err(fault(tracker.reason(Failure::CleanupRejected)))
        }
    }
}
const NO_REUSE_INDEX: usize = usize::MAX;
const REUSE_SIZE_BUCKETS: usize = (MAX_BYTES.ilog2() - MIN_BYTES.ilog2() + 1) as usize;
struct ReuseIndex {
    heads: [[usize; REUSE_SIZE_BUCKETS]; 3],
}
impl Default for ReuseIndex {
    fn default() -> Self {
        Self {
            heads: [[NO_REUSE_INDEX; REUSE_SIZE_BUCKETS]; 3],
        }
    }
}
impl ReuseIndex {
    fn bucket(bytes: u64) -> Result<usize, ServerError> {
        if span(bytes)? != bytes {
            return Err(fault("private reuse index span mismatch"));
        }
        Ok((bytes.ilog2() - MIN_BYTES.ilog2()) as usize)
    }
    fn reset(&mut self) {
        self.heads.fill([NO_REUSE_INDEX; REUSE_SIZE_BUCKETS]);
    }
    fn available(&self, kind: Kind, bytes: u64) -> Result<bool, ServerError> {
        Ok(self.heads[kind.slot()][Self::bucket(bytes)?] != NO_REUSE_INDEX)
    }
    fn rebuild<B>(
        &mut self,
        buffers: &mut [Tracked<B>],
        size: impl Fn(&B) -> u64,
    ) -> Result<(), ServerError> {
        self.reset();
        // Validate the entire cohort before publishing any reusable owner.
        for owner in buffers.iter() {
            Self::bucket(size(&owner.buffer))?;
            if owner.info_pinned && owner.kind != Kind::Uniform {
                return Err(fault("private INFO pin owner mismatch"));
            }
            if owner.kind == Kind::Upload && !owner.upload_mapped {
                return Err(fault("private reuse upload remap incomplete"));
            }
        }
        // Reverse insertion preserves the former first-owner allocation order.
        for (index, owner) in buffers.iter_mut().enumerate().rev() {
            if owner.info_pinned {
                owner.next_free = NO_REUSE_INDEX;
                owner.reuse_ready = false;
                continue;
            }
            let head = &mut self.heads[owner.kind.slot()][Self::bucket(size(&owner.buffer))?];
            owner.next_free = *head;
            owner.reuse_ready = true;
            *head = index;
        }
        Ok(())
    }
    fn take<B>(
        &mut self,
        buffers: &mut [Tracked<B>],
        kind: Kind,
        bytes: u64,
        size: impl Fn(&B) -> u64,
    ) -> Result<Option<usize>, ServerError> {
        let head = &mut self.heads[kind.slot()][Self::bucket(bytes)?];
        let index = *head;
        if index == NO_REUSE_INDEX {
            return Ok(None);
        }
        let owner = buffers
            .get_mut(index)
            .ok_or_else(|| fault("private reuse index owner missing"))?;
        if !owner.reuse_ready
            || owner.info_pinned
            || owner.kind != kind
            || size(&owner.buffer) != bytes
            || kind == Kind::Upload && !owner.upload_mapped
        {
            return Err(fault("private reuse index owner mismatch"));
        }
        *head = owner.next_free;
        owner.next_free = NO_REUSE_INDEX;
        owner.reuse_ready = false;
        Ok(Some(index))
    }
}
struct PendingInfo<B = wgpu::Buffer> {
    source: B,
    target: B,
    source_index: usize,
}
fn reuse_metadata_capacity<T>(
    capacity: usize,
    pending_capacity: usize,
    cache_bytes: usize,
) -> Result<(), ServerError> {
    if capacity
        .checked_mul(std::mem::size_of::<Tracked<T>>())
        .and_then(|n| {
            n.checked_add(pending_capacity.checked_mul(std::mem::size_of::<PendingInfo<T>>())?)
        })
        .and_then(|n| n.checked_add(std::mem::size_of::<ReuseScope>()))
        .and_then(|n| n.checked_add(cache_bytes))
        .is_none_or(|n| n > REUSE_METADATA_MAX)
    {
        return Err(fault("private reuse metadata limit"));
    }
    Ok(())
}
const INFO_KEY_BYTES: usize = 64 * 1024;
const INFO_CACHE_ENTRIES: usize = 512;
const INFO_CACHE_BUCKETS: usize = 512;
const INFO_BUCKET_BYTES: usize = INFO_CACHE_BUCKETS * std::mem::size_of::<u32>();
static INFO_CACHE_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Cumulative exact INFO hits, misses (including uncached saturation), page pins.
/// Numeric only; never exposes hashes, keys, buffer IDs, or request identifiers.
pub fn transient_info_cache_stats() -> (u64, u64, u64) {
    (
        INFO_CACHE_STATS[0].load(Ordering::Relaxed),
        INFO_CACHE_STATS[1].load(Ordering::Relaxed),
        INFO_CACHE_STATS[2].load(Ordering::Relaxed),
    )
}
fn record_info_cache_stat(slot: usize) {
    let _ = INFO_CACHE_STATS[slot].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_add(1))
    });
}
fn info_hash(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
struct InfoEntry<R> {
    hash: u64,
    key_start: usize,
    key_len: usize,
    owner_index: usize,
    // One-based prior entry in this bucket; zero terminates the chain.
    next: u32,
    resource: R,
}
struct InfoCache<R = WgpuResource> {
    keys: PrivateHostAllocation,
    // Request-derived bucket links need the same guarded ownership as keys.
    buckets: PrivateHostAllocation,
    used: usize,
    entries: Vec<InfoEntry<R>>,
}
impl<R> InfoCache<R> {
    fn reserved_bytes() -> usize {
        INFO_KEY_BYTES
            + INFO_BUCKET_BYTES
            + INFO_CACHE_ENTRIES * std::mem::size_of::<InfoEntry<R>>()
            + std::mem::size_of::<Self>()
    }
    fn new() -> Result<Self, ServerError> {
        let mut cache = Self {
            keys: PrivateHostAllocation::zeroed(INFO_KEY_BYTES)
                .map_err(|_| fault("private INFO key allocation"))?,
            buckets: PrivateHostAllocation::zeroed(INFO_BUCKET_BYTES)
                .map_err(|_| fault("private INFO bucket allocation"))?,
            used: 0,
            entries: Vec::new(),
        };
        cache
            .entries
            .try_reserve_exact(INFO_CACHE_ENTRIES)
            .map_err(|_| fault("private INFO entry allocation"))?;
        if cache.keys.capacity() != INFO_KEY_BYTES
            || cache.buckets.capacity() != INFO_BUCKET_BYTES
            || cache.entries.capacity() > INFO_CACHE_ENTRIES
        {
            return Err(fault("private INFO cache actual capacity"));
        }
        // Every future private key/header/link stays inside these fixed owners.
        reuse_metadata_capacity::<wgpu::Buffer>(0, 0, cache.host_bytes())?;
        Ok(cache)
    }
    fn host_bytes(&self) -> usize {
        // Construction bounds all capacities; their sum cannot overflow.
        self.keys.capacity()
            + self.buckets.capacity()
            + self.entries.capacity() * std::mem::size_of::<InfoEntry<R>>()
            + std::mem::size_of::<Self>()
    }
    fn lookup(&self, data: &[u8], hash: u64) -> Option<&InfoEntry<R>> {
        let mut link = self.bucket_head(hash);
        // Even a malformed link cannot escape the allocation or loop forever.
        // Normal chains only visit records in one bucket, newest first.
        for _ in 0..self.entries.len() {
            let index = link.checked_sub(1)? as usize;
            let entry = self.entries.get(index)?;
            if entry.hash == hash
                && entry.key_len == data.len()
                && self.keys.as_slice()[entry.key_start..entry.key_start + entry.key_len] == *data
            {
                return Some(entry);
            }
            link = entry.next;
        }
        None
    }
    fn bucket_offset(hash: u64) -> usize {
        ((hash ^ (hash >> 32)) as usize & (INFO_CACHE_BUCKETS - 1)) * std::mem::size_of::<u32>()
    }
    fn bucket_head(&self, hash: u64) -> u32 {
        let at = Self::bucket_offset(hash);
        u32::from_ne_bytes(self.buckets.as_slice()[at..at + 4].try_into().unwrap())
    }
    fn has_room(&self, data: &[u8]) -> bool {
        !data.is_empty()
            && self.entries.len() < INFO_CACHE_ENTRIES
            && self
                .used
                .checked_add(data.len())
                .is_some_and(|end| end <= INFO_KEY_BYTES)
    }
    fn insert(&mut self, data: &[u8], hash: u64, owner_index: usize, resource: R) -> bool {
        if !self.has_room(data) {
            return false;
        }
        let key_start = self.used;
        let next = self.bucket_head(hash);
        self.keys.as_mut_slice()[key_start..key_start + data.len()].copy_from_slice(data);
        self.used += data.len();
        // Capacity was admitted before any private key existed; never reallocates.
        self.entries.push(InfoEntry {
            hash,
            key_start,
            key_len: data.len(),
            owner_index,
            next,
            resource,
        });
        // Publish only after the entry and key are initialized. Append-only
        // indices stay valid until the entire cache is destroyed at full audit.
        let at = Self::bucket_offset(hash);
        self.buckets.as_mut_slice()[at..at + 4]
            .copy_from_slice(&(self.entries.len() as u32).to_ne_bytes());
        true
    }
    fn clear_private(&mut self) {
        // Release initialized GPU references before overwriting the raw allocation.
        // No live Rust object is zeroed, and spare capacity is only written, not read.
        self.entries.clear();
        // Successful construction requires the key slab's logical span to equal
        // its actual capacity. Constructor failures have never copied private keys;
        // PrivateHostAllocation's Drop still wipes all spare capacity as well.
        let keys = self.keys.as_mut_slice();
        // SAFETY: this initialized exclusive slice remains owned throughout wipe.
        unsafe {
            cubecl_runtime::transient::wipe_mapped_full_span(keys.as_mut_ptr(), keys.len());
        }
        let buckets = self.buckets.as_mut_slice();
        // SAFETY: fixed, initialized, exclusively owned allocation remains live.
        unsafe {
            cubecl_runtime::transient::wipe_mapped_full_span(buckets.as_mut_ptr(), buckets.len());
        }
        let pointer = self.entries.as_mut_ptr().cast::<u8>();
        let bytes = self.entries.capacity() * std::mem::size_of::<InfoEntry<R>>();
        for index in 0..bytes {
            // SAFETY: this Vec exclusively owns all allocation capacity. Length
            // is zero, all initialized values have dropped, and writes cover only
            // its writable allocation, including never-initialized spare bytes.
            unsafe {
                pointer.add(index).write_volatile(0);
            }
        }
        // SAFETY: the request-derived used header belongs exclusively to self.
        unsafe {
            core::ptr::addr_of_mut!(self.used).write_volatile(0);
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}
impl<R> Drop for InfoCache<R> {
    fn drop(&mut self) {
        self.clear_private();
        // PrivateHostAllocation then wipes its complete capacity on Drop.
    }
}
static SCOPED_READ_NS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Scoped-only aggregate host nanoseconds: setup, submission flush, map/callback.
/// Includes rejected attempts, no request identifiers or content; not GPU time.
pub fn transient_scoped_read_ns() -> (u64, u64, u64) {
    (
        SCOPED_READ_NS[0].load(Ordering::Relaxed),
        SCOPED_READ_NS[1].load(Ordering::Relaxed),
        SCOPED_READ_NS[2].load(Ordering::Relaxed),
    )
}
pub(crate) struct ScopedReadTimer {
    started: Option<Instant>,
    phase: usize,
}
impl ScopedReadTimer {
    pub(crate) fn new(enabled: bool, phase: usize) -> Self {
        Self {
            started: enabled.then(Instant::now),
            phase,
        }
    }
    pub(crate) fn finish(&mut self) {
        if let Some(started) = self.started.take() {
            let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            let _ = SCOPED_READ_NS[self.phase].fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |n| Some(n.saturating_add(elapsed)),
            );
        }
    }
}
impl Drop for ScopedReadTimer {
    fn drop(&mut self) {
        self.finish();
    }
}
/// Zero-only owners admitted after complete inspected two-pass sanitization.
/// Only Readback buffers enter: returned read leases prove no external accessor
/// survives the drain. Uniform resources and mapped Uploads are never recycled.
struct CleanReadbacks<B> {
    buffers: Vec<(B, u64)>,
    enabled: bool,
}
impl<B> CleanReadbacks<B> {
    fn new(enabled: bool) -> Result<Self, ServerError> {
        let mut buffers = Vec::new();
        if enabled {
            let bytes = CLEAN_READBACK_COUNT
                .checked_mul(std::mem::size_of::<(B, u64)>())
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .filter(|&n| n <= 4096)
                .ok_or_else(|| fault("clean readback metadata bound"))?;
            let _ = bytes;
            buffers
                .try_reserve_exact(CLEAN_READBACK_COUNT)
                .map_err(|_| fault("clean readback metadata allocation"))?;
            if buffers
                .capacity()
                .checked_mul(std::mem::size_of::<(B, u64)>())
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .is_none_or(|n| n > 4096)
            {
                return Err(fault("clean readback actual metadata bound"));
            }
        }
        Ok(Self { buffers, enabled })
    }
    fn take(&mut self, bytes: u64, ledger: &mut Ledger) -> Result<Option<B>, ServerError> {
        let Some(index) = self.buffers.iter().position(|(_, size)| *size == bytes) else {
            return Ok(None);
        };
        ledger.activate_clean_readback(bytes)?;
        Ok(Some(self.buffers.swap_remove(index).0))
    }
    /// Caller must prove complete accepted inspected erasure, drained scheduling
    /// and zero live mapping/returned-read leases BEFORE transferring an owner.
    fn offer(&mut self, buffer: B, bytes: u64, ledger: &mut Ledger) -> Result<bool, ServerError> {
        if !self.enabled
            || self.buffers.len() >= CLEAN_READBACK_COUNT
            || span(bytes)? != bytes
            || ledger
                .clean_bytes
                .checked_add(bytes)
                .is_none_or(|n| n > CLEAN_READBACK_BYTES)
        {
            return Ok(false); // Fully clean excess owner is dropped, never reallocated.
        }
        ledger.retain_clean(bytes)?;
        self.buffers.push((buffer, bytes)); // Preallocated public pointer envelope.
        Ok(true)
    }
}
pub(crate) struct TransientIo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ledger: Ledger,
    buffers: Vec<Tracked>,
    clean_readbacks: CleanReadbacks<wgpu::Buffer>,
    reuse_scope: ReuseScope,
    audit_retained_capacity: usize,
    info_cache: Option<InfoCache>,
    leases: Arc<LeaseTracker>,
    errors: Vec<ErrorFuture>,
    info_policy: Option<InfoPolicy>,
    info_slab: Option<InfoSlab>,
    info_batch: bool,
    gpu_upload_clear: bool,
    inline_readback: bool,
    pending_info: Vec<PendingInfo>,
}
impl std::fmt::Debug for TransientIo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransientIo")
            .field("owned_bytes", &self.ledger.charged)
            .field("retired", &self.retired())
            .finish_non_exhaustive()
    }
}

/// Local guards are !Send, but popped native futures are Send. Keep guards
/// only across synchronous API calls; never store them inside a ComputeServer.
pub(crate) struct IoScopes(Vec<wgpu::ErrorScopeGuard>);
impl IoScopes {
    pub fn new(device: &wgpu::Device) -> Self {
        // Allocate BEFORE scopes are pushed, and establish the LIFO owner
        // before individual API calls. Vec's default Drop order is unsuitable
        // for nested scopes on early-return/panic paths.
        let mut scopes = Self(Vec::with_capacity(3));
        scopes
            .0
            .push(device.push_error_scope(wgpu::ErrorFilter::OutOfMemory));
        scopes
            .0
            .push(device.push_error_scope(wgpu::ErrorFilter::Internal));
        scopes
            .0
            .push(device.push_error_scope(wgpu::ErrorFilter::Validation));
        scopes
    }
    pub fn finish(mut self, io: &mut TransientIo) {
        while let Some(scope) = self.0.pop() {
            io.errors.push(Box::pin(scope.pop()));
        }
    }
}

fn drop_lifo<T>(values: &mut Vec<T>) {
    while let Some(value) = values.pop() {
        drop(value);
    }
}
impl Drop for IoScopes {
    fn drop(&mut self) {
        drop_lifo(&mut self.0);
    }
}

/// The SAME ordering policy is used by the native eraser and fault-injected
/// host fixture. Keep release/ledger advancement outside this function: false
/// cannot become an erasure receipt, even if some buffers became zero.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct AuditCounts {
    bytes: u64,
    buffers: u64,
}
impl AuditCounts {
    fn add(&mut self, other: Self) -> Result<(), ServerError> {
        self.bytes = self
            .bytes
            .checked_add(other.bytes)
            .ok_or_else(|| fault("private audit counter overflow"))?;
        self.buffers = self
            .buffers
            .checked_add(other.buffers)
            .ok_or_else(|| fault("private audit counter overflow"))?;
        Ok(())
    }
}
trait AuditBackend {
    type Buffer;
    /// Retain/charge the SAME full source and extra owned staging before copies.
    fn prepare_full(
        &mut self,
        buffer: &Self::Buffer,
        kind: Kind,
    ) -> Result<AuditCounts, ServerError>;
    /// Complete all prepared copies and inspect every full readable span.
    fn inspect_prepared(&mut self, buffers: &[Tracked<Self::Buffer>]) -> Result<(), ServerError>;
    fn finish_scopes(&mut self) {}
}
fn accepts_clean_epoch(initial_erasure: bool, inspection: bool, final_erasure: bool) -> bool {
    initial_erasure && inspection && final_erasure
}
fn inspect_all<A: AuditBackend>(
    audit: &mut A,
    buffers: &[Tracked<A::Buffer>],
) -> Result<AuditCounts, ServerError> {
    let result = (|| {
        let mut counts = AuditCounts::default();
        for tracked in buffers {
            counts.add(audit.prepare_full(&tracked.buffer, tracked.kind)?)?;
        }
        audit.inspect_prepared(buffers)?;
        Ok(counts)
    })();
    // Normal errors also retain their typed scopes for the final erase fence.
    // The native scope owner's Drop remains LIFO on an actual panic.
    audit.finish_scopes();
    result
}

/// Public fixed metadata bound. No request bytes or variable-length map owner
/// list is retained here; guards must abort/unmap on every return/unwind.
const AUDIT_MAP_GROUP: usize = 64;
trait AuditMapping {
    type Buffer;
    type Mapping;
    fn prepare(&mut self, buffer: &Self::Buffer) -> Result<Self::Mapping, ServerError>;
    fn request(&mut self, mapping: &mut Self::Mapping) -> Result<(), ServerError>;
    fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError>;
    fn callback(&mut self, mapping: &Self::Mapping, deadline: Instant) -> Result<(), ServerError>;
    fn inspect(&mut self, mapping: &Self::Mapping) -> Result<(), ServerError>;
}
fn inspect_map_group<M: AuditMapping>(
    mapper: &mut M,
    buffers: &[Option<&M::Buffer>; AUDIT_MAP_GROUP],
    count: usize,
) -> Result<(), ServerError> {
    if count == 0 || count > AUDIT_MAP_GROUP {
        return Err(fault("private audit map group extent"));
    }
    // Establish every guard slot before requesting the first map. One deadline
    // covers requests, the checked native poll and ALL callback deliveries.
    let mut mappings: [Option<M::Mapping>; AUDIT_MAP_GROUP] = std::array::from_fn(|_| None);
    let deadline = Instant::now() + POLL_TIMEOUT;
    for i in 0..count {
        mappings[i] =
            Some(mapper.prepare(
                buffers[i].ok_or_else(|| fault("private audit map group missing owner"))?,
            )?);
    }
    for mapping in mappings[..count].iter_mut().flatten() {
        mapper.request(mapping)?;
    }
    mapper.poll_group(deadline)?;
    // No inspected success is possible from a partially accepted callback set.
    for mapping in mappings[..count].iter().flatten() {
        mapper.callback(mapping, deadline)?;
    }
    for mapping in mappings[..count].iter().flatten() {
        mapper.inspect(mapping)?;
    }
    // Guards drop here BEFORE final erasure; errors/panics drop them as well.
    Ok(())
}
#[cfg(test)]
fn inspect_readable_groups<M: AuditMapping>(
    mapper: &mut M,
    originals: &[Tracked<M::Buffer>],
    stages: &[Tracked<M::Buffer>],
) -> Result<(), ServerError> {
    inspect_readable_groups_selected(mapper, originals, stages, false)
}
fn inspect_readable_groups_selected<M: AuditMapping>(
    mapper: &mut M,
    originals: &[Tracked<M::Buffer>],
    stages: &[Tracked<M::Buffer>],
    gpu_upload_clear: bool,
) -> Result<(), ServerError> {
    if stages.iter().any(|stage| stage.kind != Kind::Readback) {
        return Err(fault("private audit stage kind"));
    }
    let mut group = [None; AUDIT_MAP_GROUP];
    let mut count = 0;
    for tracked in originals
        .iter()
        .filter(|tracked| !needs_audit_stage(tracked.kind, gpu_upload_clear))
        .chain(stages)
    {
        group[count] = Some(&tracked.buffer);
        count += 1;
        if count == AUDIT_MAP_GROUP {
            inspect_map_group(mapper, &group, count)?;
            group.fill(None);
            count = 0;
        }
    }
    if count != 0 {
        inspect_map_group(mapper, &group, count)?;
    }
    Ok(())
}

trait EraseBackend {
    type Buffer;
    fn drain(&mut self) -> bool;
    fn erase_uploads_full_spans(&mut self, buffers: &[Tracked<Self::Buffer>]) -> bool;
    fn clear_copy_destination_full_span(&mut self, buffer: &Self::Buffer) -> bool;
    fn finish_and_accept_errors(&mut self) -> bool;
}
fn erase_all<E: EraseBackend>(
    eraser: &mut E,
    buffers: &[Tracked<E::Buffer>],
    previous_error: bool,
) -> bool {
    let mut failed = previous_error;
    failed |= !eraser.drain();
    failed |= !eraser.erase_uploads_full_spans(buffers);
    for tracked in buffers {
        if tracked.kind != Kind::Upload {
            failed |= !eraser.clear_copy_destination_full_span(&tracked.buffer);
        }
    }
    failed |= !eraser.finish_and_accept_errors();
    !failed
}

/// Upload erasure is best effort across EVERY owner, unlike audit acceptance.
/// One failed preparation/map/callback/wipe/unmap cannot skip later owners or
/// destination clears. Fixed group bookkeeping never contains request bytes.
const UPLOAD_MAP_GROUP: usize = 64;
trait UploadMapping {
    type Buffer;
    type Mapping;
    fn prepare(&mut self, buffer: &Self::Buffer) -> Result<Self::Mapping, ServerError>;
    fn request(&mut self, mapping: &mut Self::Mapping) -> Result<(), ServerError>;
    fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError>;
    fn callback(&mut self, mapping: &Self::Mapping, deadline: Instant) -> Result<(), ServerError>;
    fn wipe(&mut self, mapping: &Self::Mapping) -> Result<(), ServerError>;
    fn unmap(&mut self, mapping: &mut Self::Mapping) -> Result<(), ServerError>;
    fn failure(&mut self, category: Failure);
}
fn attempt_upload<M: UploadMapping, T>(
    mapper: &mut M,
    category: Failure,
    operation: impl FnOnce(&mut M) -> Result<T, ServerError>,
) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(mapper))) {
        Ok(Ok(value)) => Some(value),
        _ => {
            // Native methods can record a more precise typed callback/poll
            // category first; the tracker never overwrites that first failure.
            mapper.failure(category);
            None
        }
    }
}
fn erase_upload_map_group<M: UploadMapping>(
    mapper: &mut M,
    buffers: &[Option<&M::Buffer>; UPLOAD_MAP_GROUP],
    count: usize,
) -> bool {
    if count == 0 || count > UPLOAD_MAP_GROUP {
        mapper.failure(Failure::UploadMap);
        return false;
    }
    let deadline = Instant::now() + POLL_TIMEOUT;
    let mut mappings: [Option<M::Mapping>; UPLOAD_MAP_GROUP] = std::array::from_fn(|_| None);
    let mut requested = [false; UPLOAD_MAP_GROUP];
    let mut ready = [false; UPLOAD_MAP_GROUP];
    let mut failed = false;
    // Prepare ALL channels and unmap owners before the first map request. A
    // failed slot does not stop preparing or requesting the other full owners.
    for i in 0..count {
        mappings[i] = attempt_upload(mapper, Failure::UploadMap, |mapper| {
            mapper.prepare(buffers[i].ok_or_else(|| fault("private upload map missing owner"))?)
        });
        failed |= mappings[i].is_none();
    }
    for (i, mapping) in mappings[..count].iter_mut().enumerate() {
        if let Some(mapping) = mapping {
            requested[i] =
                attempt_upload(mapper, Failure::UploadMap, |mapper| mapper.request(mapping))
                    .is_some();
            failed |= !requested[i];
        }
    }
    if mappings[..count].iter().any(Option::is_some) {
        failed |= attempt_upload(mapper, Failure::UploadMap, |mapper| {
            mapper.poll_group(deadline)
        })
        .is_none();
    }
    // The SAME deadline applies independently to every callback. Continue on
    // rejection/disconnection/timeout; never grant a fresh per-owner timeout.
    for (i, mapping) in mappings[..count].iter().enumerate() {
        if requested[i] {
            ready[i] = attempt_upload(mapper, Failure::UploadMap, |mapper| {
                mapper.callback(mapping.as_ref().unwrap(), deadline)
            })
            .is_some();
            failed |= !ready[i];
        }
    }
    for (i, mapping) in mappings[..count].iter().enumerate() {
        if ready[i] {
            failed |= attempt_upload(mapper, Failure::UploadWipe, |mapper| {
                mapper.wipe(mapping.as_ref().unwrap())
            })
            .is_none();
        }
    }
    // Explicitly check EVERY requested/pending unmap before destination clears
    // or the next group. Guard Drop supplies the unwind/failure fallback too.
    for mapping in mappings[..count].iter_mut().flatten() {
        failed |= attempt_upload(mapper, Failure::Unmap, |mapper| mapper.unmap(mapping)).is_none();
    }
    !failed
}
/// Prepare a bounded remap cohort without erasing or releasing any owner.
/// On every partial failure the retained Mapping guards cancel/unmap all
/// attempted maps. Successful callers must transfer each guard's obligation
/// into its original Tracked owner before disarming that guard.
fn map_upload_reuse_group<M: UploadMapping>(
    mapper: &mut M,
    buffers: &[Option<&M::Buffer>; UPLOAD_MAP_GROUP],
    count: usize,
) -> Result<[Option<M::Mapping>; UPLOAD_MAP_GROUP], ServerError> {
    if count == 0 || count > UPLOAD_MAP_GROUP {
        return Err(fault("private reuse map group size"));
    }
    let deadline = Instant::now() + POLL_TIMEOUT;
    let mut mappings = std::array::from_fn(|_| None);
    let mut requested = [false; UPLOAD_MAP_GROUP];
    let mut failed = false;
    for i in 0..count {
        mappings[i] = attempt_upload(mapper, Failure::UploadMap, |m| {
            m.prepare(buffers[i].ok_or_else(|| fault("private reuse map owner missing"))?)
        });
        failed |= mappings[i].is_none();
    }
    for i in 0..count {
        if let Some(mapping) = &mut mappings[i] {
            requested[i] =
                attempt_upload(mapper, Failure::UploadMap, |m| m.request(mapping)).is_some();
            failed |= !requested[i];
        }
    }
    failed |= attempt_upload(mapper, Failure::UploadMap, |m| m.poll_group(deadline)).is_none();
    for i in 0..count {
        if requested[i] {
            failed |= attempt_upload(mapper, Failure::UploadMap, |m| {
                m.callback(mappings[i].as_ref().unwrap(), deadline)
            })
            .is_none();
        }
    }
    if failed {
        Err(fault("private reuse remap rejected"))
    } else {
        Ok(mappings)
    }
}
fn erase_upload_groups<M: UploadMapping>(mapper: &mut M, buffers: &[Tracked<M::Buffer>]) -> bool {
    let mut group = [None; UPLOAD_MAP_GROUP];
    let mut count = 0;
    let mut failed = false;
    for tracked in buffers
        .iter()
        .filter(|tracked| tracked.kind == Kind::Upload)
    {
        group[count] = Some(&tracked.buffer);
        count += 1;
        if count == UPLOAD_MAP_GROUP {
            failed |= !erase_upload_map_group(mapper, &group, count);
            group.fill(None);
            count = 0;
        }
    }
    if count != 0 {
        failed |= !erase_upload_map_group(mapper, &group, count);
    }
    !failed
}

/// Same pre-owned unmap guard as audits; UploadMapping chooses WRITE maps and
/// volatile full-span writes instead of read inspection. Audit policy is unchanged.
struct NativeUploadMapping<'a>(&'a TransientIo);
impl UploadMapping for NativeUploadMapping<'_> {
    type Buffer = wgpu::Buffer;
    type Mapping = AuditMapGuard;
    fn prepare(&mut self, buffer: &wgpu::Buffer) -> Result<AuditMapGuard, ServerError> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        Ok(AuditMapGuard {
            buffer: buffer.clone(),
            retired: self.0.leases.clone(),
            sender: Some(sender),
            receiver,
            requested: false,
        })
    }
    fn request(&mut self, mapping: &mut AuditMapGuard) -> Result<(), ServerError> {
        let sender = mapping
            .sender
            .take()
            .ok_or_else(|| fault("private upload map callback missing"))?;
        mapping.requested = true;
        mapping
            .buffer
            .map_async(wgpu::MapMode::Write, .., move |result| {
                let _ = sender.try_send(result);
            });
        Ok(())
    }
    fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError> {
        if !self.0.poll_map_until(deadline) {
            return Err(fault(self.0.leases.reason(Failure::UploadMap)));
        }
        Ok(())
    }
    fn callback(&mut self, mapping: &AuditMapGuard, deadline: Instant) -> Result<(), ServerError> {
        await_map_callback(&mapping.receiver, deadline).map_err(|category| {
            self.0.leases.record(category.upload());
            fault(self.0.leases.reason(category.upload()))
        })
    }
    fn wipe(&mut self, mapping: &AuditMapGuard) -> Result<(), ServerError> {
        let mut view = mapping.buffer.get_mapped_range_mut(..);
        if view.len() as u64 != mapping.buffer.size() {
            return Err(fault("private upload wipe extent"));
        }
        let pointer = view.slice(..).as_raw_element_ptr().as_ptr();
        // SAFETY: exclusive full WRITE view, exact owned API-visible span;
        // the map guard remains active until checked explicit unmap below.
        unsafe { cubecl_runtime::transient::wipe_mapped_full_span(pointer, view.len()) };
        // The full view drops before checked explicit unmap/guard fallback.
        Ok(())
    }
    fn unmap(&mut self, mapping: &mut AuditMapGuard) -> Result<(), ServerError> {
        if mapping.requested {
            mapping.buffer.unmap();
            mapping.requested = false;
        }
        Ok(())
    }
    fn failure(&mut self, category: Failure) {
        if matches!(category, Failure::Unmap) {
            self.0.leases.retire_with(category);
        } else {
            self.0.leases.record(category);
        }
    }
}

struct NativeEraser<'a> {
    io: &'a mut TransientIo,
    encoder: Option<wgpu::CommandEncoder>,
    scopes: Option<IoScopes>,
}
impl EraseBackend for NativeEraser<'_> {
    type Buffer = wgpu::Buffer;
    fn drain(&mut self) -> bool {
        self.io.poll(None)
    }
    fn erase_uploads_full_spans(&mut self, buffers: &[Tracked<wgpu::Buffer>]) -> bool {
        if self.io.gpu_upload_clear {
            let mut accepted = true;
            for tracked in buffers.iter().filter(|b| b.kind == Kind::Upload) {
                let cleared = tracked
                    .buffer
                    .usage()
                    .contains(wgpu::BufferUsages::COPY_DST)
                    && self.clear_copy_destination_full_span(&tracked.buffer);
                accepted &= cleared;
                if cleared {
                    GPU_UPLOAD_CLEARS.fetch_add(1, Ordering::Relaxed);
                }
            }
            return accepted;
        }
        erase_upload_groups(&mut NativeUploadMapping(&*self.io), buffers)
    }
    fn clear_copy_destination_full_span(&mut self, buffer: &wgpu::Buffer) -> bool {
        let accepted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.encoder.as_mut().unwrap().clear_buffer(buffer, 0, None);
        }))
        .is_ok();
        if !accepted {
            self.io.leases.record(Failure::DestinationClear);
        }
        accepted
    }
    fn finish_and_accept_errors(&mut self) -> bool {
        let mut failed = false;
        let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.io
                .queue
                .submit([self.encoder.take().unwrap().finish()])
        }));
        match submitted {
            Ok(index) => failed |= !self.io.poll(Some(index)),
            Err(_) => {
                self.io.leases.record(Failure::QueueSubmit);
                failed = true
            }
        }
        self.scopes.take().unwrap().finish(self.io);
        for future in self.io.errors.drain(..) {
            if let Some(error) = cubecl_common::future::block_on(future) {
                self.io.leases.record(gpu_failure(&error));
                failed = true;
            }
        }
        !failed
    }
}

/// All verification copies share one submission. Extra stages remain charged
/// and retained in the SAME ledger, including partial setup/error/unwind.
// Only the explicitly selected batched-INFO laboratory packs audit copies.
// Full original sources and full staging padding remain independently owned.
const PACKED_AUDIT_BYTES: u64 = 1024 * 1024;
fn audit_stage_plan(sizes: impl Iterator<Item = u64>) -> Result<Vec<(u64, u64)>, ServerError> {
    let mut plan = Vec::new();
    let mut pending = 0u64;
    for bytes in sizes {
        if bytes < MIN_BYTES || !bytes.is_power_of_two() || bytes > MAX_BYTES {
            return Err(fault("private packed audit source span"));
        }
        if pending != 0
            && pending
                .checked_add(bytes)
                .is_none_or(|n| n > PACKED_AUDIT_BYTES)
        {
            if plan.len() >= 256 {
                return Err(fault("private packed audit metadata limit"));
            }
            plan.try_reserve_exact(1)
                .map_err(|_| fault("private packed audit plan allocation"))?;
            plan.push((pending, span(pending)?));
            pending = 0;
        }
        if bytes >= PACKED_AUDIT_BYTES {
            if plan.len() >= 256 {
                return Err(fault("private packed audit metadata limit"));
            }
            plan.try_reserve_exact(1)
                .map_err(|_| fault("private packed audit plan allocation"))?;
            plan.push((bytes, bytes));
        } else {
            pending = pending
                .checked_add(bytes)
                .ok_or_else(|| fault("private packed audit span overflow"))?;
        }
    }
    if pending != 0 {
        if plan.len() >= 256 {
            return Err(fault("private packed audit metadata limit"));
        }
        plan.try_reserve_exact(1)
            .map_err(|_| fault("private packed audit plan allocation"))?;
        plan.push((pending, span(pending)?));
    }
    // Consumers pop in original source order; no private source trace is kept.
    plan.reverse();
    Ok(plan)
}
struct PackedAuditStage {
    buffer: wgpu::Buffer,
    cursor: u64,
    logical: u64,
}
struct NativeAudit<'a> {
    io: &'a mut TransientIo,
    encoder: Option<wgpu::CommandEncoder>,
    scopes: Option<IoScopes>,
    stage_plan: Option<Vec<(u64, u64)>>,
    current_stage: Option<PackedAuditStage>,
}
impl<'a> NativeAudit<'a> {
    fn new(io: &'a mut TransientIo, originals: &[Tracked]) -> Result<Self, ServerError> {
        let stage_plan = if io.info_batch {
            Some(audit_stage_plan(
                originals
                    .iter()
                    .filter(|b| needs_audit_stage(b.kind, io.gpu_upload_clear))
                    .map(|b| b.buffer.size()),
            )?)
        } else {
            None
        };
        let scopes = IoScopes::new(&io.device);
        let encoder = io
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL private zero audit"),
            });
        Ok(Self {
            io,
            encoder: Some(encoder),
            scopes: Some(scopes),
            stage_plan,
            current_stage: None,
        })
    }
}
/// Own the unmap obligation BEFORE map_async. Pending, rejected and completed
/// maps all unmap on normal return/error/unwind before the final erase pass.
struct AuditMapGuard {
    buffer: wgpu::Buffer,
    retired: Arc<LeaseTracker>,
    sender: Option<std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>>,
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    requested: bool,
}
impl Drop for AuditMapGuard {
    fn drop(&mut self) {
        if self.requested
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.buffer.unmap()))
                .is_err()
        {
            self.retired.retire_with(Failure::Unmap);
        }
    }
}
struct NativeAuditMapping<'a>(&'a TransientIo);
impl AuditMapping for NativeAuditMapping<'_> {
    type Buffer = wgpu::Buffer;
    type Mapping = AuditMapGuard;
    fn prepare(&mut self, buffer: &wgpu::Buffer) -> Result<AuditMapGuard, ServerError> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        Ok(AuditMapGuard {
            buffer: buffer.clone(),
            retired: self.0.leases.clone(),
            sender: Some(sender),
            receiver,
            requested: false,
        })
    }
    fn request(&mut self, mapping: &mut AuditMapGuard) -> Result<(), ServerError> {
        let sender = mapping
            .sender
            .take()
            .ok_or_else(|| fault("private audit map callback missing"))?;
        mapping.requested = true;
        mapping
            .buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.try_send(result);
            });
        Ok(())
    }
    fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError> {
        // Wait(None) drains the submitted verification copies AND all requested
        // maps; no separate per-copy submission poll or inference wait exists.
        if !self.0.poll_map_until(deadline) {
            self.0.leases.record(Failure::AuditMap);
            return Err(fault(self.0.leases.reason(Failure::AuditMap)));
        }
        Ok(())
    }
    fn callback(&mut self, mapping: &AuditMapGuard, deadline: Instant) -> Result<(), ServerError> {
        await_map_callback(&mapping.receiver, deadline).map_err(|category| {
            self.0.leases.record(category.audit());
            fault(self.0.leases.reason(category.audit()))
        })
    }
    fn inspect(&mut self, mapping: &AuditMapGuard) -> Result<(), ServerError> {
        let view = mapping.buffer.get_mapped_range(..);
        if view.len() as u64 != mapping.buffer.size()
            || !cubecl_runtime::transient::full_span_is_zero(&view)
        {
            self.0.leases.record(Failure::AuditNonzero);
            return Err(fault(self.0.leases.reason(Failure::AuditNonzero)));
        }
        // View drops before its group guard. No copy/to_vec/format/log exists.
        Ok(())
    }
}
impl AuditBackend for NativeAudit<'_> {
    type Buffer = wgpu::Buffer;
    fn prepare_full(
        &mut self,
        source: &wgpu::Buffer,
        kind: Kind,
    ) -> Result<AuditCounts, ServerError> {
        let mut counts = AuditCounts {
            bytes: source.size(),
            buffers: 1,
        };
        if kind == Kind::Upload
            && self.io.gpu_upload_clear
            && !source
                .usage()
                .contains(allocation_usage(Kind::Upload, true))
        {
            return Err(fault("primary upload audit usage mismatch"));
        }
        if needs_audit_stage(kind, self.io.gpu_upload_clear) {
            // Charge the same complete MAP_READ span before any copy, and keep
            // its owner in io.buffers even if later preparation/submission fails.
            self.io.ledger.check(span(source.size())?).map_err(|_| {
                self.io.leases.record(Failure::AuditBudget);
                fault(self.io.leases.reason(Failure::AuditBudget))
            })?;
            let packed = self.stage_plan.is_some();
            if self.current_stage.as_ref().is_some_and(|stage| {
                stage
                    .cursor
                    .checked_add(source.size())
                    .is_none_or(|n| n > stage.logical)
            }) {
                if self.current_stage.as_ref().unwrap().cursor
                    != self.current_stage.as_ref().unwrap().logical
                {
                    return Err(fault("private packed audit incomplete group"));
                }
                self.current_stage = None;
            }
            if self.current_stage.is_none() {
                let (logical, bytes) = if let Some(plan) = &mut self.stage_plan {
                    plan.pop()
                        .ok_or_else(|| fault("private packed audit plan exhausted"))?
                } else {
                    (source.size(), source.size())
                };
                if bytes < source.size() {
                    return Err(fault("private packed audit source exceeds stage"));
                }
                let stage = self
                    .io
                    .allocate_audit(Kind::Readback, bytes)
                    .inspect_err(|_| {
                        self.io.leases.record(Failure::AuditAllocation);
                    })?;
                let encoder = self
                    .encoder
                    .as_mut()
                    .ok_or_else(|| fault("private audit encoder missing"))?;
                encoder.clear_buffer(&stage, 0, None);
                counts.add(AuditCounts {
                    bytes: stage.size(),
                    buffers: 1,
                })?;
                self.current_stage = Some(PackedAuditStage {
                    buffer: stage,
                    cursor: 0,
                    logical,
                });
            }
            let stage = self.current_stage.as_mut().unwrap();
            let next = stage
                .cursor
                .checked_add(source.size())
                .filter(|n| *n <= stage.logical)
                .ok_or_else(|| fault("private packed audit copy extent"))?;
            self.encoder
                .as_mut()
                .ok_or_else(|| fault("private audit encoder missing"))?
                .copy_buffer_to_buffer(source, 0, &stage.buffer, stage.cursor, source.size());
            stage.cursor = next;
            if !packed {
                self.current_stage = None;
            }
        }
        Ok(counts)
    }
    fn inspect_prepared(&mut self, originals: &[Tracked<wgpu::Buffer>]) -> Result<(), ServerError> {
        if self.stage_plan.as_ref().is_some_and(|p| !p.is_empty()) {
            return Err(fault("private packed audit unused plan"));
        }
        if self
            .current_stage
            .as_ref()
            .is_some_and(|s| s.cursor != s.logical)
        {
            return Err(fault("private packed audit incomplete final group"));
        }
        self.current_stage = None;
        let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.io
                .queue
                .submit([self.encoder.take().unwrap().finish()])
        }));
        if submitted.is_err() {
            self.io.leases.record(Failure::QueueSubmit);
            return Err(fault(self.io.leases.reason(Failure::QueueSubmit)));
        }
        let mut mapper = NativeAuditMapping(&*self.io);
        inspect_readable_groups_selected(
            &mut mapper,
            originals,
            &self.io.buffers,
            self.io.gpu_upload_clear,
        )
    }
    fn finish_scopes(&mut self) {
        if let Some(scopes) = self.scopes.take() {
            scopes.finish(self.io);
        }
    }
}

impl TransientIo {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, budget: u64) -> Result<Self, ServerError> {
        #[cfg(target_family = "wasm")]
        {
            let _ = (device, queue, budget);
            return Err(fault("private I/O requires native runtime"));
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let raw = std::env::var_os("KUNO_LAB_INFO_SLAB_BYTES");
            let slab_bytes = configured_info_slab(
                raw.as_ref()
                    .map(|v| {
                        v.to_str()
                            .ok_or_else(|| fault("private INFO slab setting encoding"))
                    })
                    .transpose()?,
            )?;
            let raw_batch = std::env::var_os("KUNO_LAB_INFO_BATCH_UPLOAD");
            let info_batch = configured_info_batch(
                raw_batch
                    .as_ref()
                    .map(|v| {
                        v.to_str()
                            .ok_or_else(|| fault("batched private INFO setting encoding"))
                    })
                    .transpose()?,
                slab_bytes,
            )?;
            if info_batch {
                println!("private_info_batch_upload enabled=true");
            }
            let raw_reuse = std::env::var_os("KUNO_LAB_READBACK_REUSE");
            let reuse = configured_readback_reuse(
                raw_reuse
                    .as_ref()
                    .map(|v| {
                        v.to_str()
                            .ok_or_else(|| fault("readback reuse setting encoding"))
                    })
                    .transpose()?,
                info_batch,
            )?;
            let raw_gpu = std::env::var_os("KUNO_LAB_GPU_UPLOAD_CLEAR");
            let gpu_upload_clear = configured_gpu_upload_clear(
                raw_gpu
                    .as_ref()
                    .map(|v| {
                        v.to_str()
                            .ok_or_else(|| fault("GPU upload clear setting encoding"))
                    })
                    .transpose()?,
                info_batch,
                device
                    .features()
                    .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            )?;
            if gpu_upload_clear {
                println!(
                    "private_primary_upload_clear enabled=true two_gpu_passes=true direct_full_cpu_inspection=true"
                );
            }
            let raw_inline = std::env::var_os("KUNO_LAB_INLINE_READBACK");
            let inline_readback = configured_inline_readback(
                raw_inline
                    .as_ref()
                    .map(|v| {
                        v.to_str()
                            .ok_or_else(|| fault("inline readback setting encoding"))
                    })
                    .transpose()?,
                info_batch,
                std::env::var_os("KUNO_LAB_DEVICE_DISPATCH").as_deref()
                    == Some(std::ffi::OsStr::new("inline")),
            )?;
            if inline_readback {
                println!(
                    "private_inline_readback enabled=true checked_callbacks=true held_leases=true"
                );
            }
            let clean_readbacks = CleanReadbacks::new(reuse)?;
            if reuse {
                println!(
                    "private_clean_readback_reuse enabled=true max_buffers=64 max_bytes=8388608"
                );
            }
            let info_policy = slab_bytes
                .map(|bytes| {
                    let limits = device.limits();
                    InfoPolicy::new(
                        bytes,
                        u64::from(limits.min_storage_buffer_offset_alignment),
                        u64::from(limits.min_uniform_buffer_offset_alignment),
                        limits.max_storage_buffer_binding_size,
                        limits.max_uniform_buffer_binding_size,
                    )
                })
                .transpose()?;
            Ok(Self {
                device,
                queue,
                ledger: Ledger::new(budget)?,
                buffers: Vec::new(),
                clean_readbacks,
                reuse_scope: ReuseScope::default(),
                audit_retained_capacity: 0,
                info_cache: None,
                leases: Arc::default(),
                errors: Vec::new(),
                info_policy,
                info_slab: None,
                info_batch,
                gpu_upload_clear,
                inline_readback,
                pending_info: Vec::new(),
            })
        }
    }
    pub fn retired(&self) -> bool {
        self.leases.retired.load(Ordering::Acquire)
    }
    fn unmap_upload(&mut self, index: usize, buffer: &wgpu::Buffer) -> Result<(), ServerError> {
        let owner = self
            .buffers
            .get_mut(index)
            .ok_or_else(|| fault("private upload owner missing"))?;
        if owner.buffer != *buffer || owner.kind != Kind::Upload || !owner.upload_mapped {
            return Err(fault("private upload mapping obligation missing"));
        }
        buffer.unmap();
        owner.upload_mapped = false;
        Ok(())
    }
    fn additional_charge(&self, kind: Kind, bytes: u64) -> Result<u64, ServerError> {
        if bytes == 0
            || (self.reuse_scope.nonce.is_some() && self.reuse_scope.free.available(kind, bytes)?)
        {
            Ok(0)
        } else {
            Ok(bytes)
        }
    }
    pub(crate) fn reuse_active(&self) -> bool {
        self.reuse_scope.nonce.is_some()
    }
    /// Completion and all GPU validation/OOM/internal scope results are
    /// mandatory even when the private bytes stay dirty inside their owner.
    fn drain_accept_errors(&mut self) -> Result<(), ServerError> {
        let mut failed = !self.poll(None);
        for future in self.errors.drain(..) {
            if let Some(error) = cubecl_common::future::block_on(future) {
                self.leases.record(gpu_failure(&error));
                failed = true;
            }
        }
        if failed || self.retired() {
            self.leases.retire_with(Failure::PriorStreamError);
            return Err(fault(self.leases.reason(Failure::PriorStreamError)));
        }
        Ok(())
    }
    pub(crate) fn begin_reuse(&mut self) -> Result<u64, ServerError> {
        self.active()?;
        if !self.info_batch || !self.inline_readback || self.gpu_upload_clear {
            return Err(fault(
                "private reuse requires batched inline standard-upload runtime",
            ));
        }
        // A previously audited zero-only Readback pool may exist. All dirty
        // allocations, pending consumers, maps and old scopes must be absent.
        let empty = self.buffers.is_empty()
            && self.ledger.charged == 0
            && self.pending_info.is_empty()
            && self.info_slab.is_none()
            && self.info_cache.is_none();
        if !empty || self.leases() != 0 || self.reuse_scope.nonce.is_some() {
            return Err(fault("private reuse begin requires empty exclusive owner"));
        }
        reuse_metadata_capacity::<wgpu::Buffer>(
            self.buffers.capacity(),
            self.pending_info.capacity(),
            self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
        )?;
        self.drain_accept_errors()?;
        self.reuse_scope.free.reset();
        self.reuse_scope.begin(empty, self.leases())
    }
    pub(crate) fn check_reuse(&self, nonce: u64) -> Result<(), ServerError> {
        self.reuse_scope.check(nonce)
    }
    pub(crate) fn accept_reuse(&mut self, nonce: u64) -> Result<(), ServerError> {
        let tracker = self.leases.clone();
        accept_only_checkpoint(
            &tracker,
            self.reuse_scope.nonce,
            nonce,
            self.leases(),
            !self.pending_info.is_empty() || self.info_slab.is_some(),
            || self.drain_accept_errors(),
        )
        // No remapping, readiness/index mutation, destination clear or ledger
        // credit. Already-ready unused owners retain their prior permission.
    }
    pub(crate) fn recycle_reuse(&mut self, nonce: u64) -> Result<(), ServerError> {
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.active()?;
            self.check_reuse(nonce)?;
            if self.leases() != 0 || !self.pending_info.is_empty() || self.info_slab.is_some() {
                return Err(fault("private reuse fence has outstanding owner"));
            }
            self.drain_accept_errors()?;
            // Stop old availability until every map and error check succeeds.
            self.reuse_scope.free.reset();
            for owner in &mut self.buffers {
                owner.reuse_ready = false;
            }
            let scopes = IoScopes::new(&self.device);
            let remapped: Result<(), ServerError> = (|| {
                let mut begin = 0;
                while begin < self.buffers.len() {
                    let mut ids = [0usize; UPLOAD_MAP_GROUP];
                    let mut group = [None; UPLOAD_MAP_GROUP];
                    let mut count = 0;
                    while begin < self.buffers.len() && count < UPLOAD_MAP_GROUP {
                        let owner = &self.buffers[begin];
                        if owner.kind == Kind::Upload && !owner.upload_mapped {
                            ids[count] = begin;
                            group[count] = Some(&owner.buffer);
                            count += 1;
                        }
                        begin += 1;
                    }
                    if count == 0 {
                        continue;
                    }
                    let mut maps =
                        map_upload_reuse_group(&mut NativeUploadMapping(self), &group, count)?;
                    for i in 0..count {
                        let map = maps[i]
                            .as_mut()
                            .ok_or_else(|| fault("private reuse map missing"))?;
                        // Install the enduring unmap duty BEFORE disarming the
                        // local guard. The owner remains fully charged/tracked.
                        self.buffers[ids[i]].upload_mapped = true;
                        map.requested = false;
                    }
                }
                // Hoist old shader-read -> copy-destination dependencies out of
                // each later INFO submission into this one completed checkpoint.
                // Completed, lease-free Readbacks likewise leave MAP_READ once,
                // before later copies. Immutable pinned INFO stays shader-readable.
                // Uploads remain MAP_WRITE mapped and must never enter this batch.
                if self
                    .buffers
                    .iter()
                    .any(|owner| owner.kind != Kind::Upload && !owner.info_pinned)
                {
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("CubeCL private reuse copy transitions"),
                            });
                    encoder.transition_resources(
                        self.buffers
                            .iter()
                            .filter(|owner| owner.kind != Kind::Upload && !owner.info_pinned)
                            .map(|owner| wgpu::BufferTransition {
                                buffer: &owner.buffer,
                                state: wgpu::BufferUses::COPY_DST,
                            }),
                        std::iter::empty(),
                    );
                    // Legacy clean readbacks end with a real GPU write after
                    // CPU inspection. Preserve that physical last-writer state
                    // in one cohort too: a tracker transition alone does not
                    // touch coherent, host-cached readback cachelines. All raw
                    // leases are gone. This grants no erasure credit and does
                    // not replace the constructor clear or terminal full audit.
                    for owner in &self.buffers {
                        if owner.kind == Kind::Readback {
                            encoder.clear_buffer(&owner.buffer, 0, None);
                        }
                    }
                    self.queue.submit([encoder.finish()]);
                }
                Ok(())
            })();
            scopes.finish(self);
            remapped?;
            self.drain_accept_errors()?;
            self.reuse_scope
                .free
                .rebuild(&mut self.buffers, wgpu::Buffer::size)?;
            // Deliberately no ledger.after_erasure, zero receipt, clean pool
            // handoff, allocation release or source-content inspection claim.
            Ok(())
        }));
        match attempted {
            Ok(Ok(())) => Ok(()),
            _ => {
                self.leases.retire_with(Failure::CleanupRejected);
                Err(fault(self.leases.reason(Failure::CleanupRejected)))
            }
        }
    }
    pub(crate) fn finish_reuse(&mut self, nonce: u64) -> Result<(), ServerError> {
        self.active()?;
        let empty = self.buffers.is_empty()
            && self.ledger.charged == 0
            && self.pending_info.is_empty()
            && self.info_slab.is_none()
            && self.errors.is_empty()
            && self.info_cache.is_none();
        self.reuse_scope.finish(nonce, empty, self.leases())?;
        self.reuse_scope.free.reset();
        Ok(())
    }
    pub(crate) fn reads_inline(&self) -> bool {
        self.inline_readback
    }
    pub(crate) fn await_inline_read_maps(&self, deadline: Instant) -> Result<(), ServerError> {
        if self.poll_map_until(deadline) {
            Ok(())
        } else {
            Err(fault(self.leases.reason(Failure::ReadbackMap)))
        }
    }
    pub(crate) fn record_inline_read_batch(&self) {
        INLINE_READ_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
    pub(crate) fn batches_info(&self) -> bool {
        self.info_batch
    }
    pub(crate) fn has_pending_info(&self) -> bool {
        !self.pending_info.is_empty()
    }

    /// Each page is write-once and contains only this submission's INFO. Every
    /// full source/target remains in the original ledger through erase/audit.
    /// Returning this encoder before the task encoder proves copy-before-use;
    /// no pending source is remapped or reused after this submission boundary.
    pub(crate) fn prepare_batched_info(
        &mut self,
    ) -> Result<Option<wgpu::CommandBuffer>, ServerError> {
        if self.pending_info.is_empty() {
            return Ok(None);
        }
        self.active()?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL owned batched INFO copies"),
            });
        self.info_slab = None;
        if self
            .pending_info
            .iter()
            .any(|entry| entry.source.size() != entry.target.size())
        {
            self.retire();
            return Err(fault("batched INFO owner span mismatch"));
        }
        for entry in &self.pending_info {
            let owner = self
                .buffers
                .get_mut(entry.source_index)
                .ok_or_else(|| fault("private INFO source owner missing"))?;
            if owner.buffer != entry.source || owner.kind != Kind::Upload || !owner.upload_mapped {
                return Err(fault("private INFO source mapping obligation missing"));
            }
            entry.source.unmap();
            owner.upload_mapped = false;
            encoder.copy_buffer_to_buffer(&entry.source, 0, &entry.target, 0, entry.source.size());
        }
        self.pending_info.clear();
        Ok(Some(encoder.finish()))
    }

    pub(crate) fn batched_uniform(&mut self, data: &[u8]) -> Result<WgpuResource, ServerError> {
        self.active()?;
        if !self.info_batch {
            return Err(fault("batched INFO not selected"));
        }
        let policy = self
            .info_policy
            .ok_or_else(|| fault("batched INFO missing policy"))?;
        let logical = (data.len() as u64)
            .checked_add(3)
            .map(|n| n / 4 * 4)
            .ok_or_else(|| fault("batched INFO length overflow"))?;
        let current = self
            .info_slab
            .as_ref()
            .map(|s| {
                info_epoch(s.epoch, self.ledger.epoch)?;
                if s.source.is_none() {
                    return Err(fault("batched INFO source missing"));
                }
                Ok((s.buffer.size(), s.cursor))
            })
            .transpose()?;
        let plan = policy.plan(logical, current)?;
        if !plan.slab {
            return Err(fault("batched INFO record exceeds fixed page"));
        }
        let cache_hash = if self.reuse_active() {
            if self.info_cache.is_none() {
                // Admit both fixed allocations before creating the optional cache.
                reuse_metadata_capacity::<wgpu::Buffer>(
                    self.buffers.capacity(),
                    self.pending_info.capacity(),
                    InfoCache::<WgpuResource>::reserved_bytes(),
                )?;
                let cache = InfoCache::new()?;
                reuse_metadata_capacity::<wgpu::Buffer>(
                    self.buffers.capacity(),
                    self.pending_info.capacity(),
                    cache.host_bytes(),
                )?;
                self.info_cache = Some(cache);
            }
            let hash = info_hash(data);
            if let Some(entry) = self.info_cache.as_ref().unwrap().lookup(data, hash) {
                let owner = self
                    .buffers
                    .get(entry.owner_index)
                    .ok_or_else(|| fault("private INFO cached owner missing"))?;
                if owner.kind != Kind::Uniform
                    || !owner.info_pinned
                    || owner.buffer != entry.resource.buffer
                    || entry.resource.size != logical
                {
                    return Err(fault("private INFO cached owner mismatch"));
                }
                copy_bounds(
                    logical,
                    entry.resource.offset,
                    owner.buffer.size(),
                    data.len(),
                )?;
                record_info_cache_stat(0);
                // A hit before first submission deliberately leaves pending_info
                // intact: its initial full copy still precedes every consumer.
                return Ok(WgpuResource {
                    buffer: entry.resource.buffer.clone(),
                    offset: entry.resource.offset,
                    size: entry.resource.size,
                });
            }
            record_info_cache_stat(1);
            Some(hash)
        } else {
            None
        };
        let source_charge = if plan.existing { 0 } else { plan.target_bytes };
        self.ledger
            .check_info_audit_group(
                self.additional_charge(Kind::Upload, source_charge)?,
                self.additional_charge(
                    Kind::Uniform,
                    if plan.existing { 0 } else { plan.target_bytes },
                )?,
                0,
            )
            .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        if !plan.existing {
            if self.reuse_active() {
                reuse_metadata_capacity::<wgpu::Buffer>(
                    self.buffers.capacity(),
                    self.pending_info.len() + 1,
                    self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
                )?;
                self.pending_info
                    .try_reserve_exact(1)
                    .map_err(|_| fault("batched INFO pointer allocation"))?;
                reuse_metadata_capacity::<wgpu::Buffer>(
                    self.buffers.capacity(),
                    self.pending_info.capacity(),
                    self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
                )?;
            } else {
                self.pending_info
                    .try_reserve(1)
                    .map_err(|_| fault("batched INFO pointer allocation"))?;
            }
            let (target, target_index) = self.allocate_indexed(Kind::Uniform, plan.target_bytes)?;
            let (source, source_index) = self.allocate_indexed(Kind::Upload, source_charge)?;
            {
                let mut map = source.get_mapped_range_mut(..);
                map.slice(..).fill(0);
            }
            self.pending_info.push(PendingInfo {
                source: source.clone(),
                target: target.clone(),
                source_index,
            });
            self.info_slab = Some(InfoSlab {
                buffer: target,
                source: Some(source),
                owner_index: target_index,
                cursor: 0,
                epoch: self.ledger.epoch,
            });
        }
        let page = self
            .info_slab
            .as_mut()
            .ok_or_else(|| fault("batched INFO current page missing"))?;
        copy_bounds(plan.logical, plan.offset, page.buffer.size(), data.len())?;
        // Reserve append interval before filling; failures cannot authorize a
        // subsequent overwrite. Mapped views never outlive this synchronous call.
        page.cursor = plan.next_cursor;
        {
            let source = page
                .source
                .as_ref()
                .ok_or_else(|| fault("batched INFO current source missing"))?;
            let mut map = source.get_mapped_range_mut(..);
            map.slice(plan.offset as usize..plan.offset as usize + data.len())
                .copy_from_slice(data);
        }
        let resource = WgpuResource {
            buffer: page.buffer.clone(),
            offset: plan.offset,
            size: plan.logical,
        };
        if let Some(hash) = cache_hash {
            let cache = self.info_cache.as_mut().unwrap();
            if cache.has_room(data) {
                let owner = self
                    .buffers
                    .get_mut(page.owner_index)
                    .ok_or_else(|| fault("private INFO cache target missing"))?;
                if owner.kind != Kind::Uniform
                    || owner.buffer != resource.buffer
                    || owner.reuse_ready
                {
                    return Err(fault("private INFO cache target mismatch"));
                }
                // Pin the full target before publishing its first cached range.
                // Its append source exists only until the original first submit.
                let new_pin = !owner.info_pinned;
                owner.info_pinned = true;
                if !cache.insert(
                    data,
                    hash,
                    page.owner_index,
                    WgpuResource {
                        buffer: resource.buffer.clone(),
                        offset: resource.offset,
                        size: resource.size,
                    },
                ) {
                    return Err(fault("private INFO cache admitted capacity changed"));
                }
                if new_pin {
                    record_info_cache_stat(2);
                }
            }
            // Saturation retains all old keys/pins and takes the ordinary path.
        }
        Ok(resource)
    }
    pub fn retire(&self) {
        self.leases.retire_with(Failure::ContextRetired);
    }
    pub fn mutation_guard(&self) -> PrivateMutationGuard {
        PrivateMutationGuard {
            tracker: self.leases.clone(),
            accepted: false,
        }
    }
    pub fn active(&self) -> Result<(), ServerError> {
        if self.retired() {
            Err(fault(self.leases.reason(Failure::ContextRetired)))
        } else {
            Ok(())
        }
    }
    /// Bounded native queue drain. Avoid the ordinary stream sync callback
    /// future, which can remain pending indefinitely after a device wedge.
    pub fn drain_submitted(&self) -> Result<(), ServerError> {
        if self.poll(None) {
            Ok(())
        } else {
            Err(fault(self.leases.reason(Failure::DrainTimeout)))
        }
    }
    /// Synchronous cleanup reads use the same bounded native drain/callback
    /// completion as erasure/audit. Caller starts this deadline before mapping,
    /// using CleanupBoundary's timeout, and retains its ReadbackLease throughout.
    /// Record before caller marks that lease failed, preserving the useful code.
    pub(crate) fn await_cleanup_read_map(
        &self,
        receiver: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        deadline: Instant,
    ) -> Result<(), ServerError> {
        if !self.poll_map_until(deadline) {
            return Err(fault(self.leases.reason(Failure::DrainTimeout)));
        }
        await_map_callback(receiver, deadline).map_err(|category| {
            self.leases.record(category.cleanup_read());
            fault(self.leases.reason(category.cleanup_read()))
        })
    }
    pub fn leases(&self) -> usize {
        self.leases.count.load(Ordering::Acquire)
    }
    fn allocate(&mut self, kind: Kind, logical: u64) -> Result<wgpu::Buffer, ServerError> {
        self.allocate_indexed(kind, logical)
            .map(|(buffer, _)| buffer)
    }
    fn allocate_indexed(
        &mut self,
        kind: Kind,
        logical: u64,
    ) -> Result<(wgpu::Buffer, usize), ServerError> {
        self.active()?;
        if self.reuse_active() {
            let full = span(logical)?;
            if let Some(index) =
                self.reuse_scope
                    .free
                    .take(&mut self.buffers, kind, full, wgpu::Buffer::size)?
            {
                REQUEST_IO_REUSES
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                    .map_err(|_| fault("private reuse count exhausted"))?;
                return Ok((self.buffers[index].buffer.clone(), index));
            }
        }
        if self.info_policy.is_some() {
            let full = span(logical)?;
            let mut group = [0; 3];
            group[kind.slot()] = full;
            self.ledger
                .check_info_audit_group(group[0], group[1], group[2])
                .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        }
        let index = self.buffers.len();
        self.allocate_audit(kind, logical)
            .map(|buffer| (buffer, index))
    }
    fn allocate_audit(&mut self, kind: Kind, logical: u64) -> Result<wgpu::Buffer, ServerError> {
        self.active()?;
        if self.reuse_scope.nonce.is_some() {
            reuse_metadata_capacity::<wgpu::Buffer>(
                self.buffers
                    .len()
                    .checked_add(1)
                    .and_then(|n| n.checked_add(self.audit_retained_capacity))
                    .ok_or_else(|| fault("private reuse metadata overflow"))?,
                self.pending_info.capacity(),
                self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
            )?;
            self.buffers
                .try_reserve_exact(1)
                .map_err(|_| fault("private reuse metadata allocation"))?;
            reuse_metadata_capacity::<wgpu::Buffer>(
                self.buffers
                    .capacity()
                    .checked_add(self.audit_retained_capacity)
                    .ok_or_else(|| fault("private reuse metadata overflow"))?,
                self.pending_info.capacity(),
                self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
            )?;
        } else {
            self.buffers.try_reserve(1).map_err(|_| {
                self.leases.record(Failure::AllocationMetadata);
                fault(self.leases.reason(Failure::AllocationMetadata))
            })?;
        }
        let full = span(logical)?;
        if kind == Kind::Readback
            && let Some(buffer) = self.clean_readbacks.take(full, &mut self.ledger)?
        {
            if buffer.size() != full {
                self.leases.retire_with(Failure::AllocationSpan);
                return Err(fault(self.leases.reason(Failure::AllocationSpan)));
            }
            self.buffers.push(Tracked {
                reuse_ready: false,
                next_free: NO_REUSE_INDEX,
                info_pinned: false,
                upload_mapped: false,
                buffer: buffer.clone(),
                kind,
            });
            READBACK_REUSES.fetch_add(1, Ordering::Relaxed);
            return Ok(buffer);
        }
        let owned = self.ledger.reserve(kind, logical).inspect_err(|_| {
            self.leases.record(Failure::AllocationBudget);
        })?;
        let usage = allocation_usage(kind, self.gpu_upload_clear);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeCL owned private I/O"),
            size: owned,
            usage,
            mapped_at_creation: kind == Kind::Upload,
        });
        // Retain full ownership before any initialization/mapping/copy can fail.
        self.buffers.push(Tracked {
            reuse_ready: false,
            next_free: NO_REUSE_INDEX,
            info_pinned: false,
            upload_mapped: kind == Kind::Upload,
            buffer: buffer.clone(),
            kind,
        });
        if buffer.size() != owned {
            self.leases.retire_with(Failure::AllocationSpan);
            return Err(fault(self.leases.reason(Failure::AllocationSpan)));
        }
        Ok(buffer)
    }
    pub fn upload(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        resource: &WgpuResource,
        data: &[u8],
    ) -> Result<(), ServerError> {
        self.active()?;
        let copy_len = copy_bounds(
            resource.size,
            resource.offset,
            resource.buffer.size(),
            data.len(),
        )?;
        let (stage, stage_index) = self.allocate_indexed(Kind::Upload, copy_len)?;
        {
            let mut view = stage.get_mapped_range_mut(..);
            // Initialize the FULL mapped API-visible span, including padding.
            view.slice(..).fill(0);
            view.slice(..data.len()).copy_from_slice(data);
        }
        self.unmap_upload(stage_index, &stage)?;
        encoder.copy_buffer_to_buffer(&stage, 0, &resource.buffer, resource.offset, copy_len);
        Ok(())
    }
    pub fn uniform(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        data: &[u8],
    ) -> Result<WgpuResource, ServerError> {
        let logical = (data.len() as u64)
            .checked_add(3)
            .map(|n| n / 4 * 4)
            .ok_or_else(|| fault("private uniform overflow"))?;
        if let Some(policy) = self.info_policy {
            return self.uniform_slab(encoder, data, logical, policy);
        }
        let needed = span(logical)?
            .checked_mul(2)
            .ok_or_else(|| fault("private uniform charge overflow"))?;
        self.ledger.check(needed).inspect_err(|_| {
            self.leases.record(Failure::AllocationBudget);
        })?;
        let buffer = self.allocate(Kind::Uniform, logical)?;
        encoder.clear_buffer(&buffer, 0, None);
        let resource = WgpuResource {
            buffer,
            offset: 0,
            size: logical,
        };
        self.upload(encoder, &resource, data)?;
        Ok(resource)
    }
    fn uniform_slab(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        data: &[u8],
        logical: u64,
        policy: InfoPolicy,
    ) -> Result<WgpuResource, ServerError> {
        self.active()?;
        let current = self
            .info_slab
            .as_ref()
            .map(|slab| -> Result<_, ServerError> {
                // Owner is created only by this same stream/device; no API
                // accepts foreign slabs. Every successful epoch drops it.
                info_epoch(slab.epoch, self.ledger.epoch)?;
                Ok((slab.buffer.size(), slab.cursor))
            })
            .transpose()?;
        let plan = policy.plan(logical, current)?;
        self.ledger
            .check_info_audit_group(plan.source_bytes, plan.target_bytes, 0)
            .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        let buffer = if plan.existing {
            self.info_slab
                .as_ref()
                .ok_or_else(|| fault("private INFO slab missing owner"))?
                .buffer
                .clone()
        } else {
            let (buffer, owner_index) = self.allocate_indexed(Kind::Uniform, plan.target_bytes)?;
            encoder.clear_buffer(&buffer, 0, None);
            if plan.slab {
                self.info_slab = Some(InfoSlab {
                    buffer: buffer.clone(),
                    source: None,
                    owner_index,
                    cursor: 0,
                    epoch: self.ledger.epoch,
                });
            }
            buffer
        };
        copy_bounds(plan.logical, plan.offset, buffer.size(), data.len())?;
        if plan.slab {
            // This append region cannot be reused within the epoch, even if
            // a subsequent mapping/copy fails. The mutation owner retires.
            self.info_slab
                .as_mut()
                .ok_or_else(|| fault("private INFO slab missing cursor"))?
                .cursor = plan.next_cursor;
        }
        let resource = WgpuResource {
            buffer,
            offset: plan.offset,
            size: plan.logical,
        };
        self.upload(encoder, &resource, data)?;
        Ok(resource)
    }
    pub fn readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        resource: &WgpuResource,
        logical: usize,
    ) -> Result<PendingReadback, ServerError> {
        let copy_len = copy_bounds(
            resource.size,
            resource.offset,
            resource.buffer.size(),
            logical,
        )?;
        let buffer = self.allocate(Kind::Readback, copy_len)?;
        // No pooled page tails: initialize the full owned direct buffer.
        encoder.clear_buffer(&buffer, 0, None);
        encoder.copy_buffer_to_buffer(&resource.buffer, resource.offset, &buffer, 0, copy_len);
        let ticket = LeaseTicket::new(self.leases.clone())?;
        let lease = ReadbackLease {
            buffer: buffer.clone(),
            ticket,
            unmap: true,
        };
        Ok(PendingReadback {
            buffer,
            lease,
            logical,
        })
    }

    /// Call only AFTER stream scheduling is drained. Refuse live readback leases
    /// without touching buffers; caller may drop leases and try cleanup again.
    pub fn sanitize_after_drain(
        &mut self,
        previous_error: bool,
        inspect: bool,
    ) -> Result<TransientIoReceipt, ServerError> {
        if self.leases() != 0 {
            return Err(fault(Failure::HeldLease.code()));
        }
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.erase_checked(previous_error, inspect || self.reuse_scope.nonce.is_some())
        }));
        match attempted {
            Ok(Ok(receipt)) => Ok(receipt),
            Ok(Err(_)) => {
                self.leases.retire_with(Failure::CleanupRejected);
                Err(fault(self.leases.reason(Failure::CleanupRejected)))
            }
            Err(_) => {
                self.leases.retire_with(Failure::Panic);
                Err(fault(self.leases.reason(Failure::Panic)))
            }
        }
    }
    #[cfg(not(target_family = "wasm"))]
    fn poll(&self, index: Option<wgpu::SubmissionIndex>) -> bool {
        self.poll_with_timeout(index, POLL_TIMEOUT)
    }
    /// Mapping's drain and delivered callback must fit the same 30-second
    /// deadline. Poll only the remaining window; never restart that timeout.
    #[cfg(not(target_family = "wasm"))]
    fn poll_map_until(&self, deadline: Instant) -> bool {
        match remaining_map_time(deadline) {
            Some(timeout) => self.poll_with_timeout(None, timeout),
            None => {
                self.leases.record(Failure::DrainTimeout);
                false
            }
        }
    }
    #[cfg(not(target_family = "wasm"))]
    fn poll_with_timeout(&self, index: Option<wgpu::SubmissionIndex>, timeout: Duration) -> bool {
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: index,
            timeout: Some(timeout),
        }) {
            Ok(_) => true,
            Err(wgpu::PollError::Timeout) => {
                self.leases.record(Failure::DrainTimeout);
                false
            }
            Err(wgpu::PollError::WrongSubmissionIndex(_, _)) => {
                self.leases.record(Failure::PollIndex);
                false
            }
        }
    }
    #[cfg(target_family = "wasm")]
    fn poll(&self, _index: Option<wgpu::SubmissionIndex>) -> bool {
        false
    }
    #[cfg(target_family = "wasm")]
    fn poll_map_until(&self, _deadline: Instant) -> bool {
        false
    }
    fn erase_set(&mut self, buffers: &[Tracked], previous_error: bool) -> bool {
        let scopes = IoScopes::new(&self.device);
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL private I/O erase"),
            });
        if previous_error {
            self.leases.record(Failure::PriorStreamError);
        }
        if self.retired() {
            self.leases.record(Failure::ContextRetired);
        }
        let failed = previous_error || self.retired();
        let mut eraser = NativeEraser {
            io: self,
            encoder: Some(encoder),
            scopes: Some(scopes),
        };
        erase_all(&mut eraser, buffers, failed)
    }
    fn erase_checked(
        &mut self,
        previous_error: bool,
        inspect: bool,
    ) -> Result<TransientIoReceipt, ServerError> {
        // Drop cached references and wipe all key/header capacities before any
        // full sanitizer touches original GPU owners. Those remain charged here.
        self.info_cache = None;
        // Audit invalidates every intrusive index before moving or releasing owners.
        self.reuse_scope.free.reset();
        // Failed submission/preparation can leave append sources mapped. Their
        // complete tracked buffers still belong to the ledger and must become
        // available to the existing remap/wipe eraser on every failure path.
        for tracked in &mut self.buffers {
            tracked.reuse_ready = false;
            tracked.info_pinned = false;
            if tracked.kind == Kind::Upload && tracked.upload_mapped {
                tracked.buffer.unmap();
                tracked.upload_mapped = false;
            }
        }
        self.info_slab = None;
        self.pending_info.clear();
        if self.info_policy.is_some() && inspect {
            // Check ALL duplicate full-span stages before any erase/audit
            // mutation. Never recurse into a checkpoint to obtain space.
            self.ledger
                .check_info_audit_group(0, 0, 0)
                .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        }
        if self.info_batch && inspect {
            // Charge packed stage padding before the FIRST erase or audit copy.
            // No original runtime cap or full-span owner is credited early.
            let plan = audit_stage_plan(
                self.buffers
                    .iter()
                    .filter(|b| needs_audit_stage(b.kind, self.gpu_upload_clear))
                    .map(|b| b.buffer.size()),
            )?;
            let bytes = plan
                .iter()
                .try_fold(0u64, |sum, (_, owned)| sum.checked_add(*owned))
                .ok_or_else(|| fault("private packed audit charge overflow"))?;
            self.ledger
                .check(bytes)
                .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        }
        // Move bookkeeping, never GPU bytes. Restore ALL original and audit
        // stages before handling any panic/error. Failed audit never releases.
        let mut buffers = std::mem::take(&mut self.buffers);
        if inspect {
            // Reserve restoration and simultaneous audit-stage metadata before
            // creating verification copies. Both live vector capacities count.
            let stages = buffers.iter().filter(|b| b.kind != Kind::Readback).count();
            let prepared = (|| {
                if self.reuse_active() {
                    let total = buffers
                        .capacity()
                        .max(
                            buffers
                                .len()
                                .checked_add(stages)
                                .ok_or_else(|| fault("private audit metadata overflow"))?,
                        )
                        .checked_add(stages)
                        .ok_or_else(|| fault("private audit metadata overflow"))?;
                    reuse_metadata_capacity::<wgpu::Buffer>(
                        total,
                        self.pending_info.capacity(),
                        self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
                    )?;
                }
                buffers
                    .try_reserve_exact(stages)
                    .map_err(|_| fault("private audit metadata allocation"))?;
                if self.reuse_active() {
                    let total = buffers
                        .capacity()
                        .checked_add(stages)
                        .ok_or_else(|| fault("private audit metadata overflow"))?;
                    reuse_metadata_capacity::<wgpu::Buffer>(
                        total,
                        self.pending_info.capacity(),
                        self.info_cache.as_ref().map_or(0, InfoCache::host_bytes),
                    )?;
                }
                Ok::<(), ServerError>(())
            })();
            if prepared.is_err() {
                self.leases.record(Failure::AuditBookkeeping);
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.erase_set(&buffers, true)
                }));
                self.buffers = buffers;
                return Err(fault(self.leases.reason(Failure::AuditBookkeeping)));
            }
        }
        self.audit_retained_capacity = buffers.capacity();
        let initial = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.erase_set(&buffers, previous_error)
        }));
        if initial.is_err() {
            self.leases.record(Failure::Panic);
        }
        let initial_ok = matches!(initial, Ok(true));
        let inspected = if inspect && initial_ok {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut audit = NativeAudit::new(self, &buffers)?;
                inspect_all(&mut audit, &buffers)
            }))
            .map_err(|_| {
                self.leases.record(Failure::Panic);
                fault(self.leases.reason(Failure::Panic))
            })
            .and_then(|result| result)
        } else if inspect {
            Err(fault("private audit preceding erasure failed"))
        } else {
            Ok(AuditCounts::default())
        };
        // Audit allocation can fail after some stages were created: retain and
        // erase those stages even on budget exhaustion/panic/nonzero bytes.
        buffers.append(&mut self.buffers);
        self.audit_retained_capacity = 0;
        let final_erased = if inspect {
            let preceding_failure = !initial_ok || inspected.is_err();
            let final_attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.erase_set(&buffers, preceding_failure)
            }));
            if final_attempt.is_err() {
                self.leases.record(Failure::Panic);
            }
            matches!(final_attempt, Ok(true)) && !preceding_failure
        } else {
            initial_ok
        };
        let accepted = accepts_clean_epoch(initial_ok, inspected.is_ok(), final_erased);
        self.buffers = buffers;
        if !accepted {
            self.leases.record(Failure::CleanupRejected);
            return Err(fault(self.leases.reason(Failure::CleanupRejected)));
        }
        let mut receipt = self.ledger.after_erasure().inspect_err(|_| {
            self.leases.record(Failure::EpochOverflow);
        })?;
        let counts = inspected?;
        receipt.inspected_bytes = counts.bytes;
        receipt.inspected_buffers = counts.buffers;
        self.info_slab = None;
        // Only inspected complete erasure can authorize reuse. Every newly dirty
        // cohort still undergoes the same original two passes and full inspection.
        // Zero leases were checked before drain; audit maps are dropped before here.
        if inspect && !previous_error && self.clean_readbacks.enabled {
            for tracked in self.buffers.drain(..) {
                if tracked.kind == Kind::Readback {
                    let bytes = tracked.buffer.size();
                    self.clean_readbacks
                        .offer(tracked.buffer, bytes, &mut self.ledger)?;
                }
            }
        } else {
            self.buffers.clear();
        }
        Ok(receipt)
    }
}
impl Drop for TransientIo {
    fn drop(&mut self) {
        // This standalone owner cannot drain a stream's unsubmitted encoder.
        // WgpuStream's explicit sanitizer is authoritative. Drop attempts only
        // owned I/O cleanup and never emits a success/privacy claim.
        let _ = self.sanitize_after_drain(false, false);
    }
}

#[cfg(test)]
mod cleanup_read_tests {
    use super::cleanup_read_bounds;
    #[test]
    fn only_entire_api_visible_pages_can_be_cleanup_audited() {
        for bytes in [32768, 65536, 32 * 1024 * 1024] {
            assert!(cleanup_read_bounds(0, bytes, bytes).is_ok());
            assert!(cleanup_read_bounds(4, bytes, bytes + 4).is_err());
            assert!(cleanup_read_bounds(0, bytes - 4, bytes).is_err());
            assert!(cleanup_read_bounds(0, bytes, bytes * 2).is_err());
        }
        assert!(cleanup_read_bounds(0, 0, 0).is_err());
        assert!(cleanup_read_bounds(0, 32767, 32767).is_err());
        assert!(cleanup_read_bounds(u64::MAX, 4, 4).is_err());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_audit_plan_preserves_complete_sources_padding_and_original_order() {
        for sizes in [
            vec![],
            vec![256],
            vec![256, 512, 256],
            vec![4096; 511],
            vec![512, 1048576, 256, 256],
            vec![524288, 262144, 524288, 2097152, 256],
        ] {
            let mut plan = audit_stage_plan(sizes.iter().copied()).unwrap();
            let input_bytes: u64 = sizes.iter().sum();
            assert_eq!(plan.iter().map(|s| s.0).sum::<u64>(), input_bytes);
            assert!(
                plan.iter()
                    .all(|(logical, owned)| *owned == span(*logical).unwrap())
            );
            let mut current = None;
            for bytes in sizes {
                if current.is_none_or(|(cursor, limit)| cursor + bytes > limit) {
                    if let Some((cursor, limit)) = current {
                        assert_eq!(cursor, limit);
                    }
                    let (limit, _) = plan.pop().unwrap();
                    current = Some((0, limit));
                }
                let (cursor, limit) = current.as_mut().unwrap();
                assert_eq!(*cursor % wgpu::COPY_BUFFER_ALIGNMENT, 0);
                *cursor += bytes;
                assert!(*cursor <= *limit);
            }
            assert!(plan.is_empty());
            if let Some((cursor, limit)) = current {
                assert_eq!(cursor, limit);
            }
        }
        for n in [0, 1, 257, MAX_BYTES + 1, u64::MAX] {
            assert!(audit_stage_plan([n].into_iter()).is_err());
        }
        assert!(audit_stage_plan(std::iter::repeat_n(PACKED_AUDIT_BYTES, 257)).is_err());
    }
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn batched_info_requires_explicit_slabs_and_complete_source_audit_charge() {
        assert!(!configured_info_batch(None, None).unwrap());
        assert!(!configured_info_batch(Some("0"), Some(4096)).unwrap());
        assert!(configured_info_batch(Some("1"), Some(4096)).unwrap());
        assert!(configured_info_batch(Some("1"), None).is_err());
        for bad in ["", "auto", "true", "01", "1 "] {
            assert!(configured_info_batch(Some(bad), Some(4096)).is_err());
        }
        let policy = InfoPolicy::new(4096, 256, 256, MAX_BYTES, 65536).unwrap();
        let first = policy.plan(16, None).unwrap();
        let mut ledger = Ledger::new(16384).unwrap();
        ledger
            .check_info_audit_group(first.target_bytes, first.target_bytes, 0)
            .unwrap();
        assert!(
            Ledger::new(16383)
                .unwrap()
                .check_info_audit_group(4096, 4096, 0)
                .is_err()
        );
        ledger.reserve(Kind::Uniform, 4096).unwrap();
        ledger.reserve(Kind::Upload, 4096).unwrap();
        let append = policy.plan(64, Some((4096, first.next_cursor))).unwrap();
        assert!(append.existing);
        assert_eq!(append.target_bytes, 0);
        assert!(append.offset >= first.logical);
        ledger.check_info_audit_group(0, 0, 0).unwrap();
        assert!(ledger.check_info_audit_group(4096, 4096, 0).is_err());
        // A flush starts fresh owners, never writes an in-flight earlier page.
        assert!(!policy.plan(16, None).unwrap().existing);
    }

    #[test]
    fn info_slab_opt_in_and_actual_storage_uniform_limits_are_strict() {
        assert_eq!(configured_info_slab(None).unwrap(), None);
        assert_eq!(configured_info_slab(Some("0")).unwrap(), None);
        for size in [4096, 8192] {
            assert_eq!(
                configured_info_slab(Some(&size.to_string())).unwrap(),
                Some(size)
            );
        }
        for raw in ["", "00", "004096", "+4096", "4096 ", "16384", "256"] {
            assert!(configured_info_slab(Some(raw)).is_err());
        }
        let policy = InfoPolicy::new(4096, 512, 256, MAX_BYTES, 65536).unwrap();
        assert_eq!(policy.alignment, 512);
        assert_eq!(policy.binding_limit, 65536);
        for alignment in [0, 1, 2, 3, 12, 8192, u64::MAX] {
            assert!(InfoPolicy::new(4096, alignment, 256, MAX_BYTES, 65536).is_err());
            assert!(InfoPolicy::new(4096, 256, alignment, MAX_BYTES, 65536).is_err());
        }
        assert!(InfoPolicy::new(256, 256, 256, MAX_BYTES, 65536).is_err());
        assert!(InfoPolicy::new(4096, 256, 256, 0, 65536).is_err());
        assert!(InfoPolicy::new(4096, 256, 256, MAX_BYTES, 0).is_err());
    }

    #[test]
    fn append_records_keep_all_prior_values_gaps_and_full_owner_tails() {
        for size in [4096, 8192] {
            let policy = InfoPolicy::new(size, 128, 256, MAX_BYTES, 65536).unwrap();
            let mut owners = Vec::<Vec<u8>>::new();
            let mut regions = Vec::new();
            let mut cursor = None;
            for n in 0..67usize {
                let logical = [4, 20, 68, 260][n % 4];
                let p = policy.plan(logical, cursor).unwrap();
                if !p.existing {
                    assert_eq!(p.target_bytes, size);
                    owners.push(vec![0; size as usize]);
                }
                let owner = owners.len() - 1;
                let start = p.offset as usize;
                let end = start + logical as usize;
                assert!(p.offset.is_multiple_of(policy.alignment));
                assert!(end <= size as usize);
                assert!(
                    regions
                        .iter()
                        .all(|&(o, first, last, _)| o != owner || end <= first || start >= last)
                );
                owners[owner][start..end].fill((n + 1) as u8);
                regions.push((owner, start, end, (n + 1) as u8));
                cursor = Some((size, p.next_cursor));
            }
            assert!(owners.len() > 1);
            for &(owner, first, last, value) in &regions {
                assert!(owners[owner][first..last].iter().all(|&v| v == value));
            }
            for (owner, bytes) in owners.iter().enumerate() {
                for (i, &byte) in bytes.iter().enumerate() {
                    if !regions
                        .iter()
                        .any(|&(o, first, last, _)| o == owner && i >= first && i < last)
                    {
                        assert_eq!(byte, 0);
                    }
                }
            }
            for bytes in &mut owners {
                bytes.fill(0);
            }
            assert!(owners.iter().flatten().all(|&b| b == 0));
        }
    }

    #[test]
    fn info_plan_boundaries_and_oversized_individual_path_never_overwrite() {
        let p = InfoPolicy::new(4096, 256, 256, MAX_BYTES, 65536).unwrap();
        let last = p.plan(256, Some((4096, 3840))).unwrap();
        assert!(last.existing && last.slab);
        assert_eq!(
            (last.offset, last.next_cursor, last.target_bytes),
            (3840, 4096, 0)
        );
        let next = p.plan(4, Some((4096, 4096))).unwrap();
        assert!(!next.existing && next.slab);
        assert_eq!(
            (next.offset, next.next_cursor, next.target_bytes),
            (0, 4, 4096)
        );
        let large = p.plan(4100, Some((4096, 20))).unwrap();
        assert!(!large.existing && !large.slab);
        assert_eq!(
            (
                large.logical,
                large.target_bytes,
                large.source_bytes,
                large.offset
            ),
            (4100, 8192, 8192, 0)
        );
        for n in [0, 1, 3, 65540, u64::MAX] {
            assert!(p.plan(n, None).is_err());
        }
        assert!(p.plan(4, Some((8192, 0))).is_err());
        assert!(p.plan(4, Some((4096, 4097))).is_err());
        assert!(p.plan(4, Some((4096, u64::MAX))).is_err());
        assert!(info_epoch(0, 0).is_err());
        assert!(info_epoch(1, 2).is_err());
        assert!(info_epoch(2, 1).is_err());
        assert!(info_epoch(1, 1).is_ok());
    }

    #[test]
    fn slab_group_preflight_includes_every_future_audit_stage_without_partial_charge() {
        let mut ledger = Ledger::new(16384).unwrap();
        ledger.check_info_audit_group(256, 4096, 0).unwrap();
        ledger.reserve(Kind::Uniform, 4096).unwrap();
        for _ in 0..16 {
            ledger.check_info_audit_group(256, 0, 0).unwrap();
            ledger.reserve(Kind::Upload, 20).unwrap();
        }
        assert_eq!(ledger.charged, 8192);
        assert_eq!(ledger.totals, [4096, 4096, 0]);
        ledger.check_info_audit_group(0, 0, 0).unwrap();
        let before = (ledger.charged, ledger.totals, ledger.epoch);
        assert!(ledger.check_info_audit_group(256, 4096, 0).is_err());
        assert!(ledger.check_info_audit_group(0, 0, 256).is_err());
        assert_eq!((ledger.charged, ledger.totals, ledger.epoch), before);
        for _ in 0..16 {
            ledger.reserve(Kind::Readback, 20).unwrap();
        }
        ledger.reserve(Kind::Readback, 4096).unwrap();
        assert_eq!(ledger.charged, 16384);
        assert!(ledger.reserve(Kind::Readback, 1).is_err());
        let receipt = ledger.after_erasure().unwrap();
        assert_eq!(
            receipt.upload_bytes + receipt.uniform_bytes + receipt.readback_bytes,
            16384
        );
        assert_eq!(ledger.charged, 0);
        assert_eq!(ledger.epoch, 2);
    }

    #[test]
    fn claimed_callback_empty_channel_is_not_a_mapping_rejection() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let (claimed_sender, claimed_receiver) = std::sync::mpsc::sync_channel(1);
        let release_callback = Arc::new(std::sync::Barrier::new(2));
        let callback_release = release_callback.clone();
        let callback = std::thread::spawn(move || {
            // Models another native poll removing the callback under its lock,
            // then being paused before UserClosures::fire outside that lock.
            claimed_sender.send(()).unwrap();
            callback_release.wait();
            sender.try_send(Ok(())).unwrap();
        });
        claimed_receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert!(matches!(
            receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
        let before_wait = Arc::new(std::sync::Barrier::new(2));
        let waiter_ready = before_wait.clone();
        let waiter = std::thread::spawn(move || {
            waiter_ready.wait();
            await_map_callback(&receiver, Instant::now() + Duration::from_secs(5))
        });
        before_wait.wait();
        release_callback.wait();
        assert_eq!(waiter.join().unwrap(), Ok(()));
        callback.join().unwrap();
    }

    #[test]
    fn callback_rejection_and_disconnection_have_distinct_static_categories() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        sender.try_send(Err(wgpu::BufferAsyncError)).unwrap();
        assert_eq!(
            await_map_callback(&receiver, Instant::now() + Duration::from_secs(5)),
            Err(MapCallbackFailure::Rejected)
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        drop(sender);
        assert_eq!(
            await_map_callback(&receiver, Instant::now() + Duration::from_secs(5)),
            Err(MapCallbackFailure::Disconnected)
        );
        for (category, upload, audit) in [
            (
                MapCallbackFailure::Rejected,
                "kuno_private_io_failure:upload_map_rejected",
                "kuno_private_io_failure:audit_map_rejected",
            ),
            (
                MapCallbackFailure::Disconnected,
                "kuno_private_io_failure:upload_map_disconnected",
                "kuno_private_io_failure:audit_map_disconnected",
            ),
            (
                MapCallbackFailure::Timeout,
                "kuno_private_io_failure:upload_map_callback_timeout",
                "kuno_private_io_failure:audit_map_callback_timeout",
            ),
        ] {
            assert_eq!(category.upload().code(), upload);
            assert_eq!(category.audit().code(), audit);
            let tracker = LeaseTracker::default();
            tracker.record(Failure::GpuOutOfMemory);
            tracker.record(category.upload());
            assert_eq!(
                tracker.reason(category.audit()),
                Failure::GpuOutOfMemory.code()
            );
        }
    }

    #[test]
    fn mapping_callback_timeout_never_restarts_deadline_or_accepts_stale_success() {
        let now = Instant::now();
        let deadline = now + POLL_TIMEOUT;
        assert_eq!(remaining_map_time_at(deadline, now), Some(POLL_TIMEOUT));
        // Models a 29-second native drain: callback receives only one second.
        assert_eq!(
            remaining_map_time_at(deadline, now + Duration::from_secs(29)),
            Some(Duration::from_secs(1))
        );
        assert_eq!(remaining_map_time_at(deadline, deadline), None);
        assert_eq!(
            remaining_map_time_at(deadline, deadline + Duration::from_nanos(1)),
            None
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        sender.try_send(Ok(())).unwrap();
        assert_eq!(
            await_map_callback(&receiver, now - Duration::from_secs(1)),
            Err(MapCallbackFailure::Timeout)
        );
        let (_sender, receiver) = std::sync::mpsc::sync_channel(1);
        assert_eq!(
            await_map_callback(&receiver, Instant::now() + Duration::from_millis(2)),
            Err(MapCallbackFailure::Timeout)
        );
    }

    #[test]
    fn cleanup_read_callback_categories_survive_lease_failure_and_mutation_drop() {
        for (category, expected) in [
            (
                MapCallbackFailure::Rejected,
                "kuno_private_io_failure:cleanup_read_map_rejected",
            ),
            (
                MapCallbackFailure::Disconnected,
                "kuno_private_io_failure:cleanup_read_map_disconnected",
            ),
            (
                MapCallbackFailure::Timeout,
                "kuno_private_io_failure:cleanup_read_map_callback_timeout",
            ),
        ] {
            let tracker = Arc::new(LeaseTracker::default());
            // Same order as await_cleanup_read_map then PendingReadback.fail:
            // callback classification precedes generic readback retirement.
            tracker.record(category.cleanup_read());
            tracker.retire_with(Failure::ReadbackMap);
            drop(PrivateMutationGuard {
                tracker: tracker.clone(),
                accepted: false,
            });
            assert_eq!(tracker.reason(Failure::CleanupRejected), expected);
            assert!(tracker.retired.load(Ordering::Acquire));
            let prior = LeaseTracker::default();
            prior.record(Failure::DrainTimeout);
            prior.record(category.cleanup_read());
            assert_eq!(
                prior.reason(Failure::ReadbackMap),
                Failure::DrainTimeout.code()
            );
        }
    }

    #[test]
    fn fixed_first_observed_category_survives_later_cleanup_retirement_and_drop() {
        let tracker = Arc::new(LeaseTracker::default());
        tracker.record(Failure::AuditBudget);
        tracker.record(Failure::GpuOutOfMemory);
        tracker.retire_with(Failure::ContextRetired);
        drop(PrivateMutationGuard {
            tracker: tracker.clone(),
            accepted: false,
        });
        assert_eq!(
            tracker.reason(Failure::CleanupRejected),
            Failure::AuditBudget.code()
        );
        assert!(tracker.retired.load(Ordering::Acquire));
        assert_eq!(
            tracker.first_failure.load(Ordering::Acquire),
            Failure::AuditBudget as usize + 1
        );
        // Held leases reject only this attempt; their fixed code is not sticky.
        let clean = LeaseTracker::default();
        let ServerError::Generic { reason, .. } = held_lease_fault() else {
            panic!("wrong error kind")
        };
        assert_eq!(reason, Failure::HeldLease.code());
        assert_eq!(clean.first_failure.load(Ordering::Acquire), 0);
        assert!(!clean.retired.load(Ordering::Acquire));
    }

    #[test]
    fn typed_scope_errors_never_export_source_description_or_shader_text() {
        let sensitive = "do-not-export-fixture-description-or-private-metadata";
        let errors = [
            wgpu::Error::OutOfMemory {
                source: Box::new(std::io::Error::other(sensitive)),
            },
            wgpu::Error::Validation {
                source: Box::new(std::io::Error::other(sensitive)),
                description: sensitive.into(),
            },
            wgpu::Error::Internal {
                source: Box::new(std::io::Error::other(sensitive)),
                description: sensitive.into(),
            },
        ];
        for (error, expected) in errors.into_iter().zip([
            Failure::GpuOutOfMemory,
            Failure::GpuValidation,
            Failure::GpuInternal,
        ]) {
            let tracker = LeaseTracker::default();
            tracker.record(gpu_failure(&error));
            let code = tracker.reason(Failure::CleanupRejected);
            assert_eq!(code, expected.code());
            assert!(!code.contains(sensitive));
            assert!(FAILURE_CODES.contains(&code));
        }
        assert_eq!(
            FAILURE_CODES.len(),
            Failure::CleanupReadMapCallbackTimeout as usize + 1
        );
        assert!(
            FAILURE_CODES
                .iter()
                .all(|code| code.starts_with("kuno_private_io_failure:"))
        );
        let mut unique = FAILURE_CODES.to_vec();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), FAILURE_CODES.len());
    }

    #[test]
    fn first_category_is_atomic_and_never_reset_by_a_new_guard() {
        let tracker = Arc::new(LeaseTracker::default());
        tracker.record(Failure::DrainTimeout);
        std::thread::scope(|scope| {
            for category in [Failure::Unmap, Failure::Panic, Failure::GpuValidation] {
                let tracker = tracker.clone();
                scope.spawn(move || tracker.retire_with(category));
            }
        });
        assert_eq!(
            tracker.reason(Failure::CleanupRejected),
            Failure::DrainTimeout.code()
        );
        PrivateMutationGuard {
            tracker: tracker.clone(),
            accepted: false,
        }
        .accept();
        assert_eq!(
            tracker.reason(Failure::CleanupRejected),
            Failure::DrainTimeout.code()
        );
        assert!(tracker.retired.load(Ordering::Acquire));
    }

    #[test]
    fn configured_private_mutation_unwind_and_early_error_permanently_retire() {
        for panic in [false, true] {
            let tracker = Arc::new(LeaseTracker::default());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = PrivateMutationGuard {
                    tracker: tracker.clone(),
                    accepted: false,
                };
                if panic {
                    panic!("synthetic private enqueue/read/flush panic")
                }
                Err::<(), _>(fault("synthetic private operation error"))
            }));
            assert!(!matches!(result, Ok(Ok(()))));
            assert!(tracker.retired.load(Ordering::Acquire));
            PrivateMutationGuard {
                tracker: tracker.clone(),
                accepted: false,
            }
            .accept();
            assert!(tracker.retired.load(Ordering::Acquire));
        }
        let tracker = Arc::new(LeaseTracker::default());
        PrivateMutationGuard {
            tracker: tracker.clone(),
            accepted: false,
        }
        .accept();
        assert!(!tracker.retired.load(Ordering::Acquire));
    }

    #[derive(Clone)]
    struct FakeBuffer {
        kind: Kind,
        words: std::rc::Rc<RefCell<Vec<u8>>>,
    }
    struct FakeEraser {
        events: Vec<&'static str>,
        fail: Option<&'static str>,
    }
    impl EraseBackend for FakeEraser {
        type Buffer = FakeBuffer;
        fn drain(&mut self) -> bool {
            self.events.push("drain");
            self.fail != Some("drain")
        }
        fn erase_uploads_full_spans(&mut self, buffers: &[Tracked<FakeBuffer>]) -> bool {
            let mut failed = false;
            for tracked in buffers
                .iter()
                .filter(|tracked| tracked.kind == Kind::Upload)
            {
                assert_eq!(tracked.buffer.kind, Kind::Upload);
                self.events.push("map-write-full-upload");
                if self.fail == Some("upload-map") {
                    failed = true;
                    continue;
                }
                tracked.buffer.words.borrow_mut().fill(0);
                self.events.push("unmap-flush-upload");
            }
            !failed
        }
        fn clear_copy_destination_full_span(&mut self, b: &FakeBuffer) -> bool {
            self.events.push(match b.kind {
                Kind::Uniform => "clear-full-uniform",
                Kind::Readback => "clear-full-readback",
                _ => panic!("upload cannot use COPY_DST"),
            });
            if self.fail == Some("copy") {
                return false;
            }
            b.words.borrow_mut().fill(0);
            true
        }
        fn finish_and_accept_errors(&mut self) -> bool {
            self.events.push("submit-wait-errors");
            self.fail != Some("finish")
        }
    }
    fn fake_buffers() -> Vec<Tracked<FakeBuffer>> {
        [Kind::Upload, Kind::Uniform, Kind::Readback]
            .into_iter()
            .map(|kind| Tracked {
                reuse_ready: false,
                next_free: NO_REUSE_INDEX,
                info_pinned: false,
                upload_mapped: false,
                // Include sentinel-filled tails beyond19 logical bytes. The shared
                // full-span erasure policy must clear those tails before release.
                kind,
                buffer: FakeBuffer {
                    kind,
                    words: std::rc::Rc::new(RefCell::new(vec![0xa7; MIN_BYTES as usize])),
                },
            })
            .collect()
    }
    #[test]
    fn shared_erase_policy_clears_all_owned_tails_before_release_preserving_public_cache() {
        let public_model = vec![0x53; 1024];
        let buffers = fake_buffers();
        let mut backend = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        assert!(erase_all(&mut backend, &buffers, false));
        assert_eq!(
            backend.events,
            [
                "drain",
                "map-write-full-upload",
                "unmap-flush-upload",
                "clear-full-uniform",
                "clear-full-readback",
                "submit-wait-errors"
            ]
        );
        assert!(
            buffers
                .iter()
                .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0))
        );
        assert_eq!(public_model, vec![0x53; 1024]);
        let mut ledger = Ledger::new(3 * MIN_BYTES).unwrap();
        for b in &buffers {
            ledger.reserve(b.kind, 19).unwrap();
        }
        let receipt = ledger.after_erasure().unwrap();
        assert_eq!(
            receipt.upload_bytes + receipt.uniform_bytes + receipt.readback_bytes,
            3 * MIN_BYTES
        );
    }
    #[test]
    fn failed_map_copy_drain_completion_never_accepts_release_and_attempts_other_spans() {
        for fail in ["upload-map", "copy", "drain", "finish"] {
            let buffers = fake_buffers();
            let mut backend = FakeEraser {
                events: Vec::new(),
                fail: Some(fail),
            };
            let mut ledger = Ledger::new(3 * MIN_BYTES).unwrap();
            for b in &buffers {
                ledger.reserve(b.kind, 19).unwrap();
            }
            let accepted = erase_all(&mut backend, &buffers, false);
            assert!(!accepted);
            assert!(backend.events.contains(&"clear-full-uniform"));
            assert!(backend.events.contains(&"clear-full-readback"));
            assert!(backend.events.contains(&"submit-wait-errors"));
            // The owner only calls after_erasure()/release on true. Keep the
            // original epoch and all charged spans on EVERY injected failure.
            assert_eq!(ledger.epoch, 1);
            assert_eq!(ledger.charged, 3 * MIN_BYTES);
        }
        let buffers = fake_buffers();
        let mut backend = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        assert!(!erase_all(&mut backend, &buffers, true)); // preceding token/device fault
    }
    #[test]
    fn full_span_budget_is_checked_before_mutation_and_epoch_after_erase_only() {
        let mut l = Ledger::new(3 * MIN_BYTES).unwrap();
        assert_eq!(l.reserve(Kind::Upload, 1).unwrap(), MIN_BYTES);
        assert_eq!(l.reserve(Kind::Uniform, 17).unwrap(), MIN_BYTES);
        assert_eq!(l.reserve(Kind::Readback, 31).unwrap(), MIN_BYTES);
        let before = (l.charged, l.totals, l.epoch);
        assert!(l.reserve(Kind::Upload, 1).is_err());
        assert_eq!((l.charged, l.totals, l.epoch), before);
        let r = l.after_erasure().unwrap();
        assert_eq!(r.epoch, 1);
        assert_eq!(
            (r.upload_bytes, r.uniform_bytes, r.readback_bytes),
            (MIN_BYTES, MIN_BYTES, MIN_BYTES)
        );
        assert_eq!((l.charged, l.totals, l.epoch), (0, [0; 3], 2));
        assert!(span(0).is_err());
        assert!(span(u64::MAX).is_err());
        assert_eq!(span(MIN_BYTES + 1).unwrap(), 2 * MIN_BYTES);
        assert!(span(MAX_BYTES + 1).is_err());
    }
    #[test]
    fn leases_span_pending_future_returned_controller_and_cancellation() {
        let tracker = Arc::<LeaseTracker>::default();
        let ticket = LeaseTicket::new(tracker.clone()).unwrap();
        assert_eq!(tracker.count.load(Ordering::Acquire), 1);
        let returned_controller = ticket;
        assert_eq!(tracker.count.load(Ordering::Acquire), 1);
        drop(returned_controller);
        assert_eq!(tracker.count.load(Ordering::Acquire), 0);
        let cancelled_future = LeaseTicket::new(tracker.clone()).unwrap();
        drop(cancelled_future);
        assert_eq!(tracker.count.load(Ordering::Acquire), 0);
        tracker.retired.store(true, Ordering::Release);
        assert!(tracker.retired.load(Ordering::Acquire));
    }
    struct FakeAudit<'a> {
        ledger: &'a mut Ledger,
        stages: Vec<Tracked<FakeBuffer>>,
        fail: Option<&'static str>,
        submissions: usize,
    }
    impl AuditBackend for FakeAudit<'_> {
        type Buffer = FakeBuffer;
        fn prepare_full(
            &mut self,
            source: &FakeBuffer,
            kind: Kind,
        ) -> Result<AuditCounts, ServerError> {
            let mut count = AuditCounts {
                bytes: source.words.borrow().len() as u64,
                buffers: 1,
            };
            if kind != Kind::Readback {
                if self.fail == Some("allocation") && !self.stages.is_empty() {
                    return Err(fault("synthetic audit allocation failed"));
                }
                // Same bounded ledger reserve path as the native owned stage.
                let size = self.ledger.reserve(Kind::Readback, count.bytes)?;
                let stage = FakeBuffer {
                    kind: Kind::Readback,
                    words: std::rc::Rc::new(RefCell::new(vec![0; size as usize])),
                };
                stage
                    .words
                    .borrow_mut()
                    .copy_from_slice(&source.words.borrow());
                self.stages.push(Tracked {
                    reuse_ready: false,
                    next_free: NO_REUSE_INDEX,
                    info_pinned: false,
                    upload_mapped: false,
                    buffer: stage,
                    kind: Kind::Readback,
                });
                if self.fail == Some("copy") && self.stages.len() == 2 {
                    return Err(fault("synthetic audit copy failed"));
                }
                if self.fail == Some("panic-copy") && self.stages.len() == 2 {
                    panic!("synthetic audit copy panic");
                }
                count.add(AuditCounts {
                    bytes: size,
                    buffers: 1,
                })?;
            }
            Ok(count)
        }
        fn inspect_prepared(
            &mut self,
            originals: &[Tracked<FakeBuffer>],
        ) -> Result<(), ServerError> {
            self.submissions += 1;
            if self.fail == Some("submit") {
                return Err(fault("synthetic audit submission failed"));
            }
            let failure = match self.fail {
                Some("map") => Some(BatchFailure::Callback(0)),
                Some("nonzero") => Some(BatchFailure::Nonzero(0)),
                _ => None,
            };
            inspect_readable_groups(&mut FakeMapper::new(failure), originals, &self.stages)
        }
    }
    #[test]
    fn audit_counts_full_padding_and_charges_extra_staging_before_release() {
        let mut buffers = fake_buffers();
        let public = vec![0x47; 19];
        let mut ledger = Ledger::new(5 * MIN_BYTES).unwrap();
        for b in &buffers {
            ledger.reserve(b.kind, 1).unwrap();
        }
        let mut eraser = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        let initially_erased = erase_all(&mut eraser, &buffers, false);
        let mut audit = FakeAudit {
            ledger: &mut ledger,
            stages: Vec::new(),
            fail: None,
            submissions: 0,
        };
        let counts = inspect_all(&mut audit, &buffers).unwrap();
        assert_eq!(
            counts,
            AuditCounts {
                bytes: 5 * MIN_BYTES,
                buffers: 5
            }
        );
        buffers.append(&mut audit.stages);
        drop(audit);
        assert_eq!(ledger.charged, 5 * MIN_BYTES);
        assert_eq!(ledger.totals[2], 3 * MIN_BYTES);
        let finally_erased = erase_all(&mut eraser, &buffers, false);
        assert!(accepts_clean_epoch(initially_erased, true, finally_erased));
        assert!(
            buffers
                .iter()
                .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0))
        );
        let receipt = ledger.after_erasure().unwrap();
        assert_eq!(
            receipt.upload_bytes + receipt.uniform_bytes + receipt.readback_bytes,
            counts.bytes
        );
        assert_eq!(public, vec![0x47; 19]);
    }
    #[test]
    fn batched_audit_counts_originals_and_every_duplicate_across_map_groups() {
        let mut buffers: Vec<_> = (0..34).flat_map(|_| fake_buffers()).collect();
        let mut ledger = Ledger::new(170 * MIN_BYTES).unwrap();
        for buffer in &buffers {
            ledger.reserve(buffer.kind, 19).unwrap();
        }
        let mut eraser = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        assert!(erase_all(&mut eraser, &buffers, false));
        let mut audit = FakeAudit {
            ledger: &mut ledger,
            stages: Vec::new(),
            fail: None,
            submissions: 0,
        };
        let counts = inspect_all(&mut audit, &buffers).unwrap();
        assert_eq!(audit.submissions, 1);
        assert_eq!(audit.stages.len(), 68);
        assert_eq!(
            counts,
            AuditCounts {
                bytes: 170 * MIN_BYTES,
                buffers: 170
            }
        );
        buffers.append(&mut audit.stages);
        drop(audit);
        assert!(erase_all(&mut eraser, &buffers, false));
        assert_eq!(ledger.charged, counts.bytes);
        let receipt = ledger.after_erasure().unwrap();
        assert_eq!(
            receipt.upload_bytes + receipt.uniform_bytes + receipt.readback_bytes,
            counts.bytes
        );
    }
    #[test]
    fn failed_audit_budget_map_or_padding_never_accepts_and_erases_new_stages() {
        for fail in [Some("map"), Some("nonzero"), None] {
            let mut buffers = fake_buffers();
            let budget = if fail.is_none() {
                4 * MIN_BYTES
            } else {
                5 * MIN_BYTES
            };
            let mut ledger = Ledger::new(budget).unwrap();
            for b in &buffers {
                ledger.reserve(b.kind, 1).unwrap();
            }
            let mut eraser = FakeEraser {
                events: Vec::new(),
                fail: None,
            };
            let initial = erase_all(&mut eraser, &buffers, false);
            let mut audit = FakeAudit {
                ledger: &mut ledger,
                stages: Vec::new(),
                fail,
                submissions: 0,
            };
            let inspected = inspect_all(&mut audit, &buffers);
            assert!(inspected.is_err());
            buffers.append(&mut audit.stages);
            drop(audit);
            let final_erased = erase_all(&mut eraser, &buffers, true);
            assert!(!accepts_clean_epoch(
                initial,
                inspected.is_ok(),
                final_erased
            ));
            assert!(
                buffers
                    .iter()
                    .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0))
            );
            assert_eq!(ledger.epoch, 1);
            assert!(ledger.charged >= 4 * MIN_BYTES); // NO reset/release receipt
        }
    }

    #[test]
    fn batch_audit_preparation_failures_keep_all_charges_and_attempt_final_erasure() {
        for failure in ["allocation", "copy", "submit", "panic-copy"] {
            let mut buffers = fake_buffers();
            let mut ledger = Ledger::new(5 * MIN_BYTES).unwrap();
            for buffer in &buffers {
                ledger.reserve(buffer.kind, 19).unwrap();
            }
            let mut eraser = FakeEraser {
                events: Vec::new(),
                fail: None,
            };
            assert!(erase_all(&mut eraser, &buffers, false));
            let mut audit = FakeAudit {
                ledger: &mut ledger,
                stages: Vec::new(),
                fail: Some(failure),
                submissions: 0,
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inspect_all(&mut audit, &buffers)
            }));
            assert!(!matches!(result, Ok(Ok(_))));
            assert_eq!(audit.submissions, usize::from(failure == "submit"));
            assert_eq!(
                audit.stages.len(),
                if failure == "allocation" { 1 } else { 2 }
            );
            buffers.append(&mut audit.stages);
            drop(audit);
            // Even if final best-effort erasure clears every byte, the failed
            // transaction can never publish counts, advance epoch or release.
            assert!(!erase_all(&mut eraser, &buffers, true));
            assert!(
                buffers
                    .iter()
                    .all(|buffer| buffer.buffer.words.borrow().iter().all(|b| *b == 0))
            );
            assert_eq!(ledger.epoch, 1);
            assert_eq!(ledger.charged, buffers.len() as u64 * MIN_BYTES);
            assert_eq!(ledger.totals[2], (buffers.len() - 2) as u64 * MIN_BYTES);
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum BatchFailure {
        Prepare(usize),
        PreparePanic(usize),
        Request(usize),
        Poll(usize),
        PollPanic(usize),
        Callback(usize),
        CallbackPanic(usize),
        Disconnect(usize),
        Pending(usize),
        Late(usize),
        Panic(usize),
        Nonzero(usize),
        Wipe(usize),
        WipePanic(usize),
        ShortView(usize),
        Unmap(usize),
        UnmapPanic(usize),
    }
    #[derive(Default)]
    struct BatchState {
        events: Vec<(&'static str, usize)>,
        poll_deadline: Option<Instant>,
        polls: usize,
    }
    struct FakeMapGuard {
        buffer: FakeBuffer,
        id: usize,
        requested: bool,
        sender: Option<std::sync::mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>>,
        receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        state: std::rc::Rc<RefCell<BatchState>>,
        tracker: Arc<LeaseTracker>,
        fail: Option<BatchFailure>,
    }
    impl Drop for FakeMapGuard {
        fn drop(&mut self) {
            if self.requested {
                self.state.borrow_mut().events.push(("unmap", self.id));
                if matches!(self.fail, Some(BatchFailure::Unmap(id)) if id == self.id) {
                    self.tracker.retire_with(Failure::Unmap);
                }
            }
        }
    }
    struct FakeMapper {
        state: std::rc::Rc<RefCell<BatchState>>,
        tracker: Arc<LeaseTracker>,
        next: usize,
        fail: Option<BatchFailure>,
    }
    impl FakeMapper {
        fn new(fail: Option<BatchFailure>) -> Self {
            Self {
                state: Default::default(),
                tracker: Default::default(),
                next: 0,
                fail,
            }
        }
    }
    impl AuditMapping for FakeMapper {
        type Buffer = FakeBuffer;
        type Mapping = FakeMapGuard;
        fn prepare(&mut self, buffer: &FakeBuffer) -> Result<FakeMapGuard, ServerError> {
            let id = self.next;
            self.next += 1;
            self.state.borrow_mut().events.push(("prepare", id));
            if matches!(self.fail, Some(BatchFailure::Prepare(n)) if n == id) {
                return Err(fault("synthetic batch bookkeeping failed"));
            }
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            Ok(FakeMapGuard {
                buffer: buffer.clone(),
                id,
                requested: false,
                sender: Some(sender),
                receiver,
                state: self.state.clone(),
                tracker: self.tracker.clone(),
                fail: self.fail,
            })
        }
        fn request(&mut self, mapping: &mut FakeMapGuard) -> Result<(), ServerError> {
            mapping.requested = true;
            self.state.borrow_mut().events.push(("request", mapping.id));
            if matches!(self.fail, Some(BatchFailure::Panic(id)) if id == mapping.id) {
                panic!("synthetic requested map panic");
            }
            if matches!(self.fail, Some(BatchFailure::Request(id)) if id == mapping.id) {
                return Err(fault("synthetic requested map failed"));
            }
            let sender = mapping.sender.take().unwrap();
            if matches!(self.fail, Some(BatchFailure::Disconnect(id)) if id == mapping.id) {
                drop(sender);
            } else {
                let result = if matches!(self.fail, Some(BatchFailure::Callback(id)) if id == mapping.id)
                {
                    Err(wgpu::BufferAsyncError)
                } else {
                    Ok(())
                };
                sender.try_send(result).unwrap();
            }
            Ok(())
        }
        fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError> {
            let mut state = self.state.borrow_mut();
            let group = state.polls;
            state.polls += 1;
            state.poll_deadline = Some(deadline);
            state.events.push(("poll", group));
            if matches!(self.fail, Some(BatchFailure::Poll(n)) if n == group) {
                self.tracker.record(Failure::DrainTimeout);
                return Err(fault(self.tracker.reason(Failure::AuditMap)));
            }
            Ok(())
        }
        fn callback(
            &mut self,
            mapping: &FakeMapGuard,
            deadline: Instant,
        ) -> Result<(), ServerError> {
            assert_eq!(self.state.borrow().poll_deadline, Some(deadline));
            self.state
                .borrow_mut()
                .events
                .push(("callback", mapping.id));
            // Model delivery after the group's deadline even with a queued
            // success. Existing await_map_callback rejects stale acceptance.
            let deadline = if matches!(self.fail, Some(BatchFailure::Late(id)) if id == mapping.id)
            {
                Instant::now() - Duration::from_nanos(1)
            } else {
                deadline
            };
            await_map_callback(&mapping.receiver, deadline).map_err(|failure| {
                self.tracker.record(failure.audit());
                fault(self.tracker.reason(failure.audit()))
            })
        }
        fn inspect(&mut self, mapping: &FakeMapGuard) -> Result<(), ServerError> {
            self.state.borrow_mut().events.push(("inspect", mapping.id));
            if matches!(self.fail, Some(BatchFailure::Nonzero(id)) if id == mapping.id) {
                *mapping.buffer.words.borrow_mut().last_mut().unwrap() = 0x6d;
            }
            if mapping.buffer.words.borrow().iter().any(|b| *b != 0) {
                self.tracker.record(Failure::AuditNonzero);
                return Err(fault(self.tracker.reason(Failure::AuditNonzero)));
            }
            Ok(())
        }
    }
    /// Inject failures through the SAME best-effort group helper as native
    /// uploads. Reuse only public fake owner/channel bookkeeping from audits.
    struct FakeUploadMapper(FakeMapper);
    impl FakeUploadMapper {
        fn new(fail: Option<BatchFailure>) -> Self {
            Self(FakeMapper::new(fail))
        }
    }
    impl UploadMapping for FakeUploadMapper {
        type Buffer = FakeBuffer;
        type Mapping = FakeMapGuard;
        fn prepare(&mut self, buffer: &FakeBuffer) -> Result<FakeMapGuard, ServerError> {
            let mapping = AuditMapping::prepare(&mut self.0, buffer)?;
            if matches!(self.0.fail, Some(BatchFailure::PreparePanic(id)) if id == mapping.id) {
                panic!("synthetic upload bookkeeping panic");
            }
            Ok(mapping)
        }
        fn request(&mut self, mapping: &mut FakeMapGuard) -> Result<(), ServerError> {
            if matches!(self.0.fail, Some(BatchFailure::Pending(id)) if id == mapping.id) {
                mapping.requested = true;
                self.0
                    .state
                    .borrow_mut()
                    .events
                    .push(("request", mapping.id));
                // Keep the sender alive without delivering: a successful poll
                // alone does not make this callback acceptable.
                Ok(())
            } else {
                AuditMapping::request(&mut self.0, mapping)
            }
        }
        fn poll_group(&mut self, deadline: Instant) -> Result<(), ServerError> {
            let group = self.0.state.borrow().polls;
            let result = AuditMapping::poll_group(&mut self.0, deadline);
            if matches!(self.0.fail, Some(BatchFailure::PollPanic(id)) if id == group) {
                panic!("synthetic upload poll panic");
            }
            result
        }
        fn callback(
            &mut self,
            mapping: &FakeMapGuard,
            deadline: Instant,
        ) -> Result<(), ServerError> {
            assert_eq!(self.0.state.borrow().poll_deadline, Some(deadline));
            self.0
                .state
                .borrow_mut()
                .events
                .push(("callback", mapping.id));
            if matches!(self.0.fail, Some(BatchFailure::CallbackPanic(id)) if id == mapping.id) {
                panic!("synthetic upload callback panic");
            }
            let deadline = match self.0.fail {
                Some(BatchFailure::Late(id)) if id == mapping.id => {
                    Instant::now() - Duration::from_nanos(1)
                }
                Some(BatchFailure::Pending(id)) if id == mapping.id => {
                    Instant::now() + Duration::from_millis(2)
                }
                _ => deadline,
            };
            await_map_callback(&mapping.receiver, deadline).map_err(|category| {
                self.0.tracker.record(category.upload());
                fault(self.0.tracker.reason(category.upload()))
            })
        }
        fn wipe(&mut self, mapping: &FakeMapGuard) -> Result<(), ServerError> {
            self.0.state.borrow_mut().events.push(("wipe", mapping.id));
            if matches!(self.0.fail, Some(BatchFailure::WipePanic(id)) if id == mapping.id) {
                panic!("synthetic full upload wipe panic");
            }
            if matches!(self.0.fail, Some(BatchFailure::Wipe(id) | BatchFailure::ShortView(id)) if id == mapping.id)
            {
                return Err(fault("synthetic full upload wipe extent failed"));
            }
            assert_eq!(mapping.buffer.words.borrow().len(), MIN_BYTES as usize);
            // Fake sentinel tails are private-owner stand-ins, never live data.
            mapping.buffer.words.borrow_mut().fill(0);
            Ok(())
        }
        fn unmap(&mut self, mapping: &mut FakeMapGuard) -> Result<(), ServerError> {
            if mapping.requested {
                self.0.state.borrow_mut().events.push(("unmap", mapping.id));
                if matches!(self.0.fail, Some(BatchFailure::UnmapPanic(id)) if id == mapping.id) {
                    panic!("synthetic upload unmap panic");
                }
                if matches!(self.0.fail, Some(BatchFailure::Unmap(id)) if id == mapping.id) {
                    return Err(fault("synthetic upload unmap failure"));
                }
                mapping.requested = false;
            }
            Ok(())
        }
        fn failure(&mut self, category: Failure) {
            if matches!(category, Failure::Unmap) {
                self.0.tracker.retire_with(category);
            } else {
                self.0.tracker.record(category);
            }
        }
    }
    struct FakeBatchedEraser {
        mapper: FakeUploadMapper,
    }
    impl EraseBackend for FakeBatchedEraser {
        type Buffer = FakeBuffer;
        fn drain(&mut self) -> bool {
            self.mapper.0.state.borrow_mut().events.push(("drain", 0));
            true
        }
        fn erase_uploads_full_spans(&mut self, buffers: &[Tracked<FakeBuffer>]) -> bool {
            erase_upload_groups(&mut self.mapper, buffers)
        }
        fn clear_copy_destination_full_span(&mut self, buffer: &FakeBuffer) -> bool {
            self.mapper.0.state.borrow_mut().events.push((
                match buffer.kind {
                    Kind::Uniform => "clear-full-uniform",
                    Kind::Readback => "clear-full-readback",
                    Kind::Upload => panic!("upload cannot use COPY_DST"),
                },
                0,
            ));
            buffer.words.borrow_mut().fill(0);
            true
        }
        fn finish_and_accept_errors(&mut self) -> bool {
            self.mapper
                .0
                .state
                .borrow_mut()
                .events
                .push(("submit-wait-errors", 0));
            true
        }
    }
    fn fake_uploads(count: usize) -> Vec<Tracked<FakeBuffer>> {
        (0..count)
            .map(|_| Tracked {
                reuse_ready: false,
                next_free: NO_REUSE_INDEX,
                info_pinned: false,
                upload_mapped: false,
                kind: Kind::Upload,
                buffer: FakeBuffer {
                    kind: Kind::Upload,
                    words: std::rc::Rc::new(RefCell::new(vec![0xa7; MIN_BYTES as usize])),
                },
            })
            .collect()
    }
    #[test]
    fn request_reuse_nonce_rejects_held_dirty_concurrent_foreign_and_stale_owners() {
        let mut scope = ReuseScope::default();
        assert!(scope.begin(false, 0).is_err());
        assert!(scope.begin(true, 1).is_err());
        let nonce = scope.begin(true, 0).unwrap();
        assert_ne!(nonce, 0);
        assert!(scope.begin(true, 0).is_err());
        assert!(scope.check(0).is_err());
        assert!(scope.check(nonce + 1).is_err());
        assert!(scope.finish(nonce, false, 0).is_err());
        assert!(scope.finish(nonce, true, 1).is_err());
        scope.check(nonce).unwrap();
        scope.finish(nonce, true, 0).unwrap();
        assert!(scope.check(nonce).is_err());
        let next = scope.begin(true, 0).unwrap();
        assert_ne!(nonce, next);
        let mut other = ReuseScope::default();
        let foreign = other.begin(true, 0).unwrap();
        assert!(scope.check(foreign).is_err());
    }
    #[test]
    fn accept_only_keeps_dirty_charges_and_does_not_publish_reuse() {
        let mut scope = ReuseScope::default();
        let nonce = scope.begin(true, 0).unwrap();
        let mut buffers = fake_buffers();
        buffers[0].upload_mapped = true;
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        scope.free.rebuild(&mut buffers, size).unwrap();
        scope
            .free
            .take(&mut buffers, Kind::Readback, MIN_BYTES, size)
            .unwrap();
        let mut ledger = Ledger::new(5 * MIN_BYTES).unwrap();
        for owner in &buffers {
            ledger.reserve(owner.kind, 19).unwrap();
        }
        let original = (ledger.charged, ledger.totals, ledger.epoch);
        let tracker = LeaseTracker::default();
        let drains = std::cell::Cell::new(0);
        for _ in 0..3 {
            accept_only_checkpoint(&tracker, scope.nonce, nonce, 0, false, || {
                drains.set(drains.get() + 1);
                Ok(())
            })
            .unwrap();
            assert!(scope.free.available(Kind::Upload, MIN_BYTES).unwrap());
            assert!(!scope.free.available(Kind::Readback, MIN_BYTES).unwrap());
            assert_eq!((ledger.charged, ledger.totals, ledger.epoch), original);
            assert!(
                buffers
                    .iter()
                    .all(|b| b.buffer.words.borrow().iter().all(|byte| *byte == 0xa7))
            );
            assert!(buffers[0].upload_mapped);
            assert!(!buffers[2].reuse_ready);
        }
        assert_eq!(drains.get(), 3);
        assert!(!tracker.retired.load(Ordering::Acquire));
        scope.check(nonce).unwrap();
    }
    #[test]
    fn accept_only_rejects_stale_held_pending_and_retired_before_completion() {
        for fault_kind in 0..5 {
            let tracker = LeaseTracker::default();
            if fault_kind == 4 {
                tracker.retire_with(Failure::ContextRetired);
            }
            let completed = std::cell::Cell::new(false);
            let result = accept_only_checkpoint(
                &tracker,
                Some(17),
                if fault_kind == 0 {
                    0
                } else if fault_kind == 1 {
                    18
                } else {
                    17
                },
                usize::from(fault_kind == 2),
                fault_kind == 3,
                || {
                    completed.set(true);
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert!(!completed.get());
            assert!(tracker.retired.load(Ordering::Acquire));
        }
        let tracker = LeaseTracker::default();
        assert!(accept_only_checkpoint(&tracker, None, 17, 0, false, || Ok(())).is_err());
    }
    #[test]
    fn accept_only_completion_error_panic_and_late_retirement_never_accept() {
        for fault_kind in 0..3 {
            let tracker = LeaseTracker::default();
            let result = accept_only_checkpoint(&tracker, Some(17), 17, 0, false, || {
                tracker.record(Failure::GpuValidation);
                match fault_kind {
                    0 => Err(fault("synthetic GPU validation failure")),
                    1 => panic!("synthetic completion panic"),
                    _ => {
                        tracker.retire_with(Failure::ContextRetired);
                        Ok(())
                    }
                }
            });
            assert!(result.is_err());
            assert!(tracker.retired.load(Ordering::Acquire));
            assert_eq!(
                tracker.reason(Failure::CleanupRejected),
                Failure::GpuValidation.code()
            );
        }
    }
    #[test]
    fn request_reuse_keeps_original_dirty_spans_charged_until_full_terminal_audit() {
        let mut buffers = fake_buffers();
        let mut ledger = Ledger::new(5 * MIN_BYTES).unwrap();
        for buffer in &buffers {
            ledger.reserve(buffer.kind, 19).unwrap();
        }
        let original = (ledger.charged, ledger.totals, ledger.epoch);
        // No partial readiness is published until every Upload remap succeeds.
        let mut free = ReuseIndex::default();
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        assert!(free.rebuild(&mut buffers, size).is_err());
        assert!(buffers.iter().all(|b| !b.reuse_ready));
        buffers[0].upload_mapped = true;
        for _ in 0..16 {
            free.rebuild(&mut buffers, size).unwrap();
            for kind in [Kind::Upload, Kind::Uniform, Kind::Readback] {
                assert!(!free.available(kind, MIN_BYTES * 2).unwrap());
                let index = free
                    .take(&mut buffers, kind, MIN_BYTES, size)
                    .unwrap()
                    .unwrap();
                assert!(
                    free.take(&mut buffers, kind, MIN_BYTES, size)
                        .unwrap()
                        .is_none()
                );
                // Production constructors fully overwrite ALL padding before use.
                buffers[index].buffer.words.borrow_mut().fill(0);
                buffers[index].buffer.words.borrow_mut()[0] = 0xb3;
            }
            assert_eq!((ledger.charged, ledger.totals, ledger.epoch), original);
            assert!(ledger.reserve(Kind::Upload, 3 * MIN_BYTES).is_err());
        }
        let mut eraser = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        assert!(erase_all(&mut eraser, &buffers, false));
        let mut audit = FakeAudit {
            ledger: &mut ledger,
            stages: Vec::new(),
            fail: None,
            submissions: 0,
        };
        let counts = inspect_all(&mut audit, &buffers).unwrap();
        buffers.append(&mut audit.stages);
        drop(audit);
        assert!(erase_all(&mut eraser, &buffers, false));
        assert!(
            buffers
                .iter()
                .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0))
        );
        assert_eq!(counts.bytes, 5 * MIN_BYTES);
        assert_eq!(counts.buffers, 5);
        let receipt = ledger.after_erasure().unwrap();
        assert_eq!(
            receipt.upload_bytes + receipt.uniform_bytes + receipt.readback_bytes,
            5 * MIN_BYTES
        );
        assert_eq!(ledger.charged, 0);
        assert_eq!(ledger.epoch, original.2 + 1);
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(2048, 64, 0).is_ok());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(usize::MAX, 0, 0).is_err());
    }
    #[test]
    fn indexed_reuse_preserves_kind_span_order_and_rejects_double_claim() {
        let mut buffers = fake_readable(5);
        buffers[1].kind = Kind::Uniform;
        buffers[2]
            .buffer
            .words
            .borrow_mut()
            .resize((2 * MIN_BYTES) as usize, 0xa7);
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        let mut free = ReuseIndex::default();
        free.rebuild(&mut buffers, size).unwrap();
        for expected in [0, 3, 4] {
            assert_eq!(
                free.take(&mut buffers, Kind::Readback, MIN_BYTES, size)
                    .unwrap(),
                Some(expected)
            );
        }
        assert!(
            free.take(&mut buffers, Kind::Readback, MIN_BYTES, size)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            free.take(&mut buffers, Kind::Readback, 2 * MIN_BYTES, size)
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            free.take(&mut buffers, Kind::Uniform, MIN_BYTES, size)
                .unwrap(),
            Some(1)
        );
        assert!(
            buffers
                .iter()
                .all(|b| !b.reuse_ready && b.next_free == NO_REUSE_INDEX)
        );
        // Claiming/relinking never mutates or releases dirty content.
        assert!(buffers.iter().all(|b| b.buffer.words.borrow()[0] == 0));
        free.rebuild(&mut buffers, size).unwrap();
        free.reset();
        assert!(!free.available(Kind::Readback, MIN_BYTES).unwrap());
        assert!(
            free.take(&mut buffers, Kind::Readback, MIN_BYTES, size)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn indexed_reuse_rejects_corrupt_or_stale_owner_before_consumption() {
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        for fault_kind in 0..5 {
            let mut buffers = fake_readable(1);
            let mut free = ReuseIndex::default();
            free.rebuild(&mut buffers, size).unwrap();
            match fault_kind {
                0 => buffers[0].reuse_ready = false,
                1 => buffers[0].kind = Kind::Uniform,
                2 => buffers[0]
                    .buffer
                    .words
                    .borrow_mut()
                    .resize((2 * MIN_BYTES) as usize, 0),
                3 => free.heads[Kind::Readback.slot()][0] = usize::MAX - 1,
                _ => {
                    buffers.clear();
                }
            }
            assert!(
                free.take(&mut buffers, Kind::Readback, MIN_BYTES, size)
                    .is_err()
            );
        }
        let mut buffers = fake_uploads(1);
        buffers[0].upload_mapped = true;
        let mut free = ReuseIndex::default();
        free.rebuild(&mut buffers, size).unwrap();
        buffers[0].upload_mapped = false;
        assert!(
            free.take(&mut buffers, Kind::Upload, MIN_BYTES, size)
                .is_err()
        );
    }
    #[test]
    fn indexed_reuse_invalid_rebuild_never_publishes_partial_availability() {
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        let mut buffers = fake_readable(2);
        let mut free = ReuseIndex::default();
        free.rebuild(&mut buffers, size).unwrap();
        buffers[1].buffer.words.borrow_mut().truncate(17);
        assert!(free.rebuild(&mut buffers, size).is_err());
        assert!(!free.available(Kind::Readback, MIN_BYTES).unwrap());
        for bytes in [0, 1, MIN_BYTES - 1, MIN_BYTES + 1, MAX_BYTES + 1, u64::MAX] {
            assert!(free.available(Kind::Readback, bytes).is_err());
        }
        assert_eq!(ReuseIndex::bucket(MIN_BYTES).unwrap(), 0);
        assert_eq!(
            ReuseIndex::bucket(MAX_BYTES).unwrap(),
            REUSE_SIZE_BUCKETS - 1
        );
    }
    #[test]
    fn indexed_reuse_metadata_counts_fixed_heads_and_actual_pending_capacity() {
        let fixed = std::mem::size_of::<ReuseScope>();
        let per = std::mem::size_of::<Tracked<wgpu::Buffer>>();
        let max = (REUSE_METADATA_MAX - fixed) / per;
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(max, 0, 0).is_ok());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(max + 1, 0, 0).is_err());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(max, 1, 0).is_err());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(0, usize::MAX, 0).is_err());
        assert!(std::mem::size_of::<ReuseIndex>() <= 512);
    }
    #[test]
    fn info_cache_exact_bytes_and_length_decide_hits_even_with_hash_collision() {
        let mut cache = InfoCache::new().unwrap();
        assert!(cache.insert(&[1, 2, 3], 7, 12, 91u32));
        let hit = cache.lookup(&[1, 2, 3], 7).unwrap();
        assert_eq!((hit.owner_index, hit.resource), (12, 91));
        assert!(cache.lookup(&[1, 2, 4], 7).is_none());
        assert!(cache.lookup(&[1, 2], 7).is_none());
        assert!(cache.lookup(&[1, 2, 3, 0], 7).is_none());
        assert!(cache.lookup(&[1, 2, 3], 8).is_none());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.used, 3);
    }
    #[test]
    fn info_bucket_collisions_retain_every_exact_key_and_reject_absent_keys() {
        let mut cache = InfoCache::new().unwrap();
        let original = cache.host_bytes();
        // Full hash collisions exercise the longest permitted chain, including
        // first/middle/last entries rather than only the newest bucket head.
        for i in 0..INFO_CACHE_ENTRIES {
            assert!(cache.insert(&(i as u32).to_le_bytes(), 7, i, i + 17));
        }
        for i in 0..INFO_CACHE_ENTRIES {
            let hit = cache.lookup(&(i as u32).to_le_bytes(), 7).unwrap();
            assert_eq!((hit.owner_index, hit.resource), (i, i + 17));
        }
        assert!(cache.lookup(&u32::MAX.to_le_bytes(), 7).is_none());
        assert!(cache.lookup(&[0; 3], 7).is_none());
        assert!(!cache.insert(&u32::MAX.to_le_bytes(), 7, 0, 0));
        assert_eq!(cache.host_bytes(), original);

        // Distinct hashes sharing a bucket must still check the entire hash.
        let mut partial = InfoCache::new().unwrap();
        let first = 7;
        let second = first + INFO_CACHE_BUCKETS as u64;
        assert_eq!(
            InfoCache::<u32>::bucket_offset(first),
            InfoCache::<u32>::bucket_offset(second)
        );
        assert!(partial.insert(&[1, 2, 3], first, 3, 41u32));
        assert!(partial.insert(&[1, 2, 3], second, 8, 67u32));
        assert_eq!(partial.lookup(&[1, 2, 3], first).unwrap().resource, 41);
        assert_eq!(partial.lookup(&[1, 2, 3], second).unwrap().resource, 67);
    }
    #[test]
    fn info_bucket_malformed_links_are_bounded_and_never_escape_owned_entries() {
        let mut cache = InfoCache::new().unwrap();
        assert!(cache.insert(&[1], 7, 0, ()));
        assert!(cache.insert(&[2], 7, 1, ()));
        // A synthetic self-cycle cannot turn a miss into an unbounded scan.
        cache.entries[1].next = 2;
        assert!(cache.lookup(&[3], 7).is_none());
        cache.entries[1].next = u32::MAX;
        assert!(cache.lookup(&[1], 7).is_none());
        let at = InfoCache::<()>::bucket_offset(7);
        cache.buckets.as_mut_slice()[at..at + 4].copy_from_slice(&u32::MAX.to_ne_bytes());
        assert!(cache.lookup(&[2], 7).is_none());
    }
    #[test]
    fn info_cache_saturation_falls_back_without_eviction_mutation_or_growth() {
        let mut cache = InfoCache::new().unwrap();
        let original = cache.host_bytes();
        for i in 0..INFO_CACHE_ENTRIES {
            let bytes = (i as u32).to_le_bytes();
            assert!(cache.insert(&bytes, info_hash(&bytes), i, ()));
        }
        assert!(!cache.insert(&[9; 5], 0, 0, ()));
        assert_eq!(cache.host_bytes(), original);
        assert_eq!(cache.entries.len(), INFO_CACHE_ENTRIES);
        for i in 0..INFO_CACHE_ENTRIES {
            let bytes = (i as u32).to_le_bytes();
            assert_eq!(
                cache.lookup(&bytes, info_hash(&bytes)).unwrap().owner_index,
                i
            );
        }
        let mut full = InfoCache::new().unwrap();
        let bytes = [0x73; INFO_KEY_BYTES];
        assert!(full.insert(&bytes, 8, 42, ()));
        assert!(!full.insert(&[1], 9, 99, ()));
        assert!(!full.insert(&[], 9, 99, ()));
        assert_eq!(full.lookup(&bytes, 8).unwrap().owner_index, 42);
        assert_eq!(full.entries.len(), 1);
        assert_eq!(full.used, INFO_KEY_BYTES);
    }
    #[test]
    fn info_cache_cleanup_drops_resources_before_wiping_full_key_header_capacities() {
        struct DropCount(std::rc::Rc<std::cell::Cell<usize>>);
        impl Drop for DropCount {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut cache = InfoCache::new().unwrap();
        assert!(cache.insert(&[0xa7; 19], 0xbeef, 37, DropCount(dropped.clone())));
        cache.keys.as_mut_slice()[19..].fill(0xbd);
        cache.buckets.as_mut_slice().fill(0xe7);
        // Fill never-initialized spare headers so the test proves full capacity,
        // not only the logical entry range. No live object is overwritten.
        let initialized = cache.entries.len() * std::mem::size_of::<InfoEntry<DropCount>>();
        let total = cache.entries.capacity() * std::mem::size_of::<InfoEntry<DropCount>>();
        let pointer = cache.entries.as_mut_ptr().cast::<u8>();
        for i in initialized..total {
            // SAFETY: exclusively owned spare allocation, never read as a value.
            unsafe {
                pointer.add(i).write(0xca);
            }
        }
        cache.clear_private();
        assert_eq!(dropped.get(), 1);
        assert_eq!(cache.used, 0);
        assert!(cache.entries.is_empty());
        assert_eq!(cache.keys.capacity(), INFO_KEY_BYTES);
        assert_eq!(cache.buckets.capacity(), INFO_BUCKET_BYTES);
        assert!(cache.keys.as_slice().iter().all(|byte| *byte == 0));
        assert!(cache.buckets.as_slice().iter().all(|byte| *byte == 0));
        // SAFETY: clear_private initialized every capacity byte while retaining
        // the allocation; no Rust entry remains live and all bytes are readable.
        let headers = unsafe { std::slice::from_raw_parts(pointer, total) };
        assert!(headers.iter().all(|byte| *byte == 0));
        drop(cache);
        assert_eq!(dropped.get(), 1);
    }
    #[test]
    fn info_cache_actual_capacities_share_original_metadata_allowance() {
        let cache = InfoCache::<WgpuResource>::new().unwrap();
        let actual = cache.keys.capacity()
            + cache.buckets.capacity()
            + cache.entries.capacity() * std::mem::size_of::<InfoEntry<WgpuResource>>()
            + std::mem::size_of_val(&cache);
        assert_eq!(cache.host_bytes(), actual);
        assert_eq!(InfoCache::<WgpuResource>::reserved_bytes(), actual);
        assert_eq!(cache.buckets.capacity(), 512 * std::mem::size_of::<u32>());
        assert!(actual > INFO_KEY_BYTES + INFO_BUCKET_BYTES);
        assert!(actual < REUSE_METADATA_MAX);
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(0, 0, actual).is_ok());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(4096, 64, actual).is_err());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(0, 0, usize::MAX).is_err());
        let remaining = REUSE_METADATA_MAX - std::mem::size_of::<ReuseScope>() - actual;
        let max_owners = remaining / std::mem::size_of::<Tracked<wgpu::Buffer>>();
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(max_owners, 0, actual).is_ok());
        assert!(reuse_metadata_capacity::<wgpu::Buffer>(max_owners + 1, 0, actual).is_err());
    }
    #[test]
    fn info_pins_exclude_full_uniform_owner_from_recycling_until_audit_reset() {
        let mut buffers = fake_buffers();
        buffers[0].upload_mapped = true;
        buffers[1].info_pinned = true;
        let size = |b: &FakeBuffer| b.words.borrow().len() as u64;
        let mut free = ReuseIndex::default();
        free.rebuild(&mut buffers, size).unwrap();
        assert!(!free.available(Kind::Uniform, MIN_BYTES).unwrap());
        assert!(!buffers[1].reuse_ready);
        assert_eq!(
            free.take(&mut buffers, Kind::Upload, MIN_BYTES, size)
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            free.take(&mut buffers, Kind::Readback, MIN_BYTES, size)
                .unwrap(),
            Some(2)
        );
        buffers[1].info_pinned = false; // Original audit clears pins before erase.
        free.rebuild(&mut buffers, size).unwrap();
        assert_eq!(
            free.take(&mut buffers, Kind::Uniform, MIN_BYTES, size)
                .unwrap(),
            Some(1)
        );
        buffers[2].info_pinned = true;
        assert!(free.rebuild(&mut buffers, size).is_err());
        assert!(!free.available(Kind::Uniform, MIN_BYTES).unwrap());
    }
    #[test]
    fn reuse_upload_maps_transfer_only_after_all_callbacks_without_wipe_or_unmap() {
        for count in [1, 63, 64] {
            let buffers = fake_uploads(count);
            let group = std::array::from_fn(|i| buffers.get(i).map(|b| &b.buffer));
            let mut mapper = FakeUploadMapper::new(None);
            let mut maps = map_upload_reuse_group(&mut mapper, &group, count).unwrap();
            let state = mapper.0.state.borrow();
            assert_eq!(state.polls, 1);
            assert_eq!(
                state.events.iter().filter(|v| v.0 == "callback").count(),
                count
            );
            assert!(!state.events.iter().any(|v| v.0 == "wipe" || v.0 == "unmap"));
            drop(state);
            assert!(
                buffers
                    .iter()
                    .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0xa7))
            );
            // Stand in for transferring each full unmap obligation to Tracked.
            for map in maps.iter_mut().take(count).flatten() {
                map.requested = false;
            }
            drop(maps);
            assert!(
                !mapper
                    .0
                    .state
                    .borrow()
                    .events
                    .iter()
                    .any(|v| v.0 == "unmap")
            );
        }
        let mut mapper = FakeUploadMapper::new(None);
        assert!(map_upload_reuse_group(&mut mapper, &[None; UPLOAD_MAP_GROUP], 0).is_err());
        assert!(map_upload_reuse_group(&mut mapper, &[None; UPLOAD_MAP_GROUP], 65).is_err());
    }
    #[test]
    fn reuse_partial_maps_keep_dirty_owners_and_unmap_every_attempt_before_failure() {
        for failure in [
            BatchFailure::Prepare(1),
            BatchFailure::PreparePanic(1),
            BatchFailure::Request(1),
            BatchFailure::Panic(1),
            BatchFailure::Poll(0),
            BatchFailure::PollPanic(0),
            BatchFailure::Callback(1),
            BatchFailure::CallbackPanic(1),
            BatchFailure::Disconnect(1),
            BatchFailure::Pending(1),
            BatchFailure::Late(1),
        ] {
            let buffers = fake_uploads(3);
            let group = std::array::from_fn(|i| buffers.get(i).map(|b| &b.buffer));
            let mut mapper = FakeUploadMapper::new(Some(failure));
            assert!(map_upload_reuse_group(&mut mapper, &group, 3).is_err());
            assert!(
                buffers
                    .iter()
                    .all(|b| b.buffer.words.borrow().iter().all(|v| *v == 0xa7))
            );
            let state = mapper.0.state.borrow();
            let requests = state.events.iter().filter(|v| v.0 == "request").count();
            let unmapped = state.events.iter().filter(|v| v.0 == "unmap").count();
            assert_eq!(requests, unmapped, "{failure:?}");
            assert!(!state.events.iter().any(|v| v.0 == "wipe"));
        }
    }
    #[test]
    fn upload_groups_bound_polls_prepare_all_maps_and_unmap_before_later_groups_or_clears() {
        for count in [0, 1, 63, 64, 65, 129] {
            let mut buffers = fake_uploads(count);
            buffers.extend(
                fake_buffers()
                    .into_iter()
                    .filter(|buffer| buffer.kind != Kind::Upload),
            );
            let mut eraser = FakeBatchedEraser {
                mapper: FakeUploadMapper::new(None),
            };
            assert!(erase_all(&mut eraser, &buffers, false));
            let state = eraser.mapper.0.state.borrow();
            assert_eq!(state.polls, count.div_ceil(UPLOAD_MAP_GROUP));
            for name in ["prepare", "request", "callback", "wipe", "unmap"] {
                assert_eq!(
                    state.events.iter().filter(|event| event.0 == name).count(),
                    count
                );
            }
            let position = |event| {
                state
                    .events
                    .iter()
                    .position(|entry| *entry == event)
                    .unwrap()
            };
            for first in (0..count).step_by(UPLOAD_MAP_GROUP) {
                let last = (first + UPLOAD_MAP_GROUP).min(count) - 1;
                assert!(position(("prepare", last)) < position(("request", first)));
                assert!(position(("callback", last)) < position(("wipe", first)));
                assert!(position(("wipe", last)) < position(("unmap", first)));
                if last + 1 < count {
                    assert!(position(("unmap", last)) < position(("prepare", last + 1)));
                }
            }
            if count != 0 {
                assert!(position(("unmap", count - 1)) < position(("clear-full-uniform", 0)));
            }
            assert!(position(("clear-full-readback", 0)) < position(("submit-wait-errors", 0)));
            assert!(
                buffers.iter().all(|buffer| buffer
                    .buffer
                    .words
                    .borrow()
                    .iter()
                    .all(|byte| *byte == 0))
            );
        }
    }
    #[test]
    fn upload_group_policy_attempts_all_later_owners_and_destinations_after_every_error_or_panic() {
        for failure in [
            BatchFailure::Prepare(17),
            BatchFailure::PreparePanic(17),
            BatchFailure::Request(17),
            BatchFailure::Panic(17),
            BatchFailure::Poll(0),
            BatchFailure::PollPanic(0),
            BatchFailure::Callback(17),
            BatchFailure::CallbackPanic(17),
            BatchFailure::Disconnect(17),
            BatchFailure::Pending(17),
            BatchFailure::Late(17),
            BatchFailure::Wipe(17),
            BatchFailure::WipePanic(17),
            BatchFailure::ShortView(17),
            BatchFailure::Unmap(17),
            BatchFailure::UnmapPanic(17),
            BatchFailure::Callback(64),
            BatchFailure::WipePanic(128),
        ] {
            let mut buffers = fake_uploads(129);
            buffers.extend(
                fake_buffers()
                    .into_iter()
                    .filter(|buffer| buffer.kind != Kind::Upload),
            );
            let mut ledger = Ledger::new(buffers.len() as u64 * MIN_BYTES).unwrap();
            for buffer in &buffers {
                ledger.reserve(buffer.kind, 19).unwrap();
            }
            let before = (ledger.epoch, ledger.charged, ledger.totals);
            let mut eraser = FakeBatchedEraser {
                mapper: FakeUploadMapper::new(Some(failure)),
            };
            assert!(!erase_all(&mut eraser, &buffers, false), "{failure:?}");
            let state = eraser.mapper.0.state.borrow();
            assert_eq!(state.polls, 3, "{failure:?}");
            assert_eq!(
                state
                    .events
                    .iter()
                    .filter(|event| event.0 == "prepare")
                    .count(),
                129
            );
            let missing = if matches!(
                failure,
                BatchFailure::Prepare(_) | BatchFailure::PreparePanic(_)
            ) {
                1
            } else {
                0
            };
            assert_eq!(
                state
                    .events
                    .iter()
                    .filter(|event| event.0 == "request")
                    .count(),
                129 - missing
            );
            for id in [18, 63, 65, 127] {
                assert!(state.events.contains(&("wipe", id)), "{failure:?}");
                assert!(state.events.contains(&("unmap", id)), "{failure:?}");
                assert!(
                    buffers[id]
                        .buffer
                        .words
                        .borrow()
                        .iter()
                        .all(|byte| *byte == 0)
                );
            }
            for &(_, id) in state.events.iter().filter(|event| event.0 == "request") {
                assert!(state.events.contains(&("unmap", id)), "{failure:?}");
            }
            let position = |event| {
                state
                    .events
                    .iter()
                    .position(|entry| *entry == event)
                    .unwrap()
            };
            assert!(position(("unmap", 128)) < position(("clear-full-uniform", 0)));
            assert!(state.events.contains(&("clear-full-readback", 0)));
            assert!(state.events.contains(&("submit-wait-errors", 0)));
            assert_eq!((ledger.epoch, ledger.charged, ledger.totals), before);
            if matches!(
                failure,
                BatchFailure::Unmap(_) | BatchFailure::UnmapPanic(_)
            ) {
                assert!(eraser.mapper.0.tracker.retired.load(Ordering::Acquire));
                assert_eq!(
                    eraser.mapper.0.tracker.reason(Failure::CleanupRejected),
                    Failure::Unmap.code()
                );
            }
        }
    }
    #[test]
    fn upload_callback_failures_keep_typed_first_cause_and_reject_later_clean_pass() {
        for (failure, expected) in [
            (BatchFailure::Callback(17), Failure::UploadMapRejected),
            (BatchFailure::Disconnect(17), Failure::UploadMapDisconnected),
            (BatchFailure::Pending(17), Failure::UploadMapCallbackTimeout),
            (BatchFailure::Late(17), Failure::UploadMapCallbackTimeout),
            (BatchFailure::Poll(0), Failure::DrainTimeout),
        ] {
            let mut buffers = fake_uploads(65);
            buffers.extend(
                fake_buffers()
                    .into_iter()
                    .filter(|buffer| buffer.kind != Kind::Upload),
            );
            let mut eraser = FakeBatchedEraser {
                mapper: FakeUploadMapper::new(Some(failure)),
            };
            assert!(!erase_all(&mut eraser, &buffers, false));
            assert_eq!(
                eraser.mapper.0.tracker.reason(Failure::CleanupRejected),
                expected.code()
            );
            eraser.mapper.0.fail = None;
            assert!(!erase_all(&mut eraser, &buffers, true));
            assert_eq!(
                eraser.mapper.0.tracker.reason(Failure::CleanupRejected),
                expected.code()
            );
            assert!(
                buffers.iter().all(|buffer| buffer
                    .buffer
                    .words
                    .borrow()
                    .iter()
                    .all(|byte| *byte == 0))
            );
        }
    }
    #[test]
    fn upload_group_rejects_invalid_or_missing_extents_without_skipping_present_owner() {
        let empty = [None; UPLOAD_MAP_GROUP];
        for count in [0, UPLOAD_MAP_GROUP + 1] {
            let mut mapper = FakeUploadMapper::new(None);
            assert!(!erase_upload_map_group(&mut mapper, &empty, count));
            assert!(mapper.0.state.borrow().events.is_empty());
        }
        let buffers = fake_uploads(1);
        let mut group = [None; UPLOAD_MAP_GROUP];
        group[1] = Some(&buffers[0].buffer);
        let mut mapper = FakeUploadMapper::new(None);
        assert!(!erase_upload_map_group(&mut mapper, &group, 2));
        assert!(mapper.0.state.borrow().events.contains(&("wipe", 0)));
        assert!(mapper.0.state.borrow().events.contains(&("unmap", 0)));
        assert!(
            buffers[0]
                .buffer
                .words
                .borrow()
                .iter()
                .all(|byte| *byte == 0)
        );
    }
    fn fake_readable(count: usize) -> Vec<Tracked<FakeBuffer>> {
        (0..count)
            .map(|_| Tracked {
                reuse_ready: false,
                next_free: NO_REUSE_INDEX,
                info_pinned: false,
                upload_mapped: false,
                kind: Kind::Readback,
                buffer: FakeBuffer {
                    kind: Kind::Readback,
                    words: std::rc::Rc::new(RefCell::new(vec![0; MIN_BYTES as usize])),
                },
            })
            .collect()
    }
    #[test]
    fn actual_group_policy_rejects_missing_or_invalid_owners_before_maps() {
        let empty = [None; AUDIT_MAP_GROUP];
        for count in [0, 1, AUDIT_MAP_GROUP + 1] {
            let mut mapper = FakeMapper::new(None);
            assert!(inspect_map_group(&mut mapper, &empty, count).is_err());
            assert!(mapper.state.borrow().events.is_empty());
        }
        let mut stages = fake_readable(1);
        stages[0].kind = Kind::Upload;
        let mut mapper = FakeMapper::new(None);
        assert!(inspect_readable_groups(&mut mapper, &[], &stages).is_err());
        assert!(mapper.state.borrow().events.is_empty());
    }
    #[test]
    fn primary_upload_inspection_covers_every_original_owner_and_rejects_poison() {
        for failure in [
            None,
            Some(BatchFailure::Nonzero(17)),
            Some(BatchFailure::Callback(17)),
        ] {
            let mut buffers = fake_readable(65);
            for buffer in &mut buffers {
                buffer.kind = Kind::Upload;
            }
            let mut mapper = FakeMapper::new(failure);
            let accepted =
                inspect_readable_groups_selected(&mut mapper, &buffers, &[], true).is_ok();
            assert_eq!(accepted, failure.is_none());
            let events = &mapper.state.borrow().events;
            assert_eq!(
                events.iter().filter(|e| e.0 == "request").count(),
                events.iter().filter(|e| e.0 == "unmap").count()
            );
            if accepted {
                assert_eq!(events.iter().filter(|e| e.0 == "inspect").count(), 65);
            }
        }
    }
    #[test]
    fn actual_group_policy_preallocates_and_accepts_every_callback_before_inspection() {
        for count in [1, 63, 64, 65, 129] {
            let buffers = fake_readable(count);
            let mut mapper = FakeMapper::new(None);
            inspect_readable_groups(&mut mapper, &buffers, &[]).unwrap();
            let state = mapper.state.borrow();
            assert_eq!(state.polls, count.div_ceil(AUDIT_MAP_GROUP));
            for name in ["prepare", "request", "callback", "inspect", "unmap"] {
                assert_eq!(
                    state.events.iter().filter(|event| event.0 == name).count(),
                    count
                );
            }
            for first in (0..count).step_by(AUDIT_MAP_GROUP) {
                let last = (first + AUDIT_MAP_GROUP).min(count) - 1;
                let position = |event| {
                    state
                        .events
                        .iter()
                        .position(|entry| *entry == event)
                        .unwrap()
                };
                assert!(position(("prepare", last)) < position(("request", first)));
                assert!(position(("callback", last)) < position(("inspect", first)));
                assert!(position(("inspect", last)) < position(("unmap", first)));
                if last + 1 < count {
                    assert!(position(("unmap", last)) < position(("prepare", last + 1)));
                }
            }
        }
    }
    #[test]
    fn actual_group_policy_failure_or_panic_unmaps_every_requested_owner() {
        for failure in [
            BatchFailure::Prepare(17),
            BatchFailure::Request(17),
            BatchFailure::Poll(0),
            BatchFailure::Callback(17),
            BatchFailure::Disconnect(17),
            BatchFailure::Late(17),
            BatchFailure::Panic(17),
            BatchFailure::Nonzero(17),
            BatchFailure::Callback(64),
        ] {
            let buffers = fake_readable(65);
            let mut mapper = FakeMapper::new(Some(failure));
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inspect_readable_groups(&mut mapper, &buffers, &[])
            }));
            assert!(!matches!(outcome, Ok(Ok(()))));
            let events = &mapper.state.borrow().events;
            let requested: Vec<_> = events
                .iter()
                .filter(|event| event.0 == "request")
                .map(|event| event.1)
                .collect();
            let unmapped: Vec<_> = events
                .iter()
                .filter(|event| event.0 == "unmap")
                .map(|event| event.1)
                .collect();
            assert_eq!(requested, unmapped, "{failure:?}");
            if matches!(failure, BatchFailure::Prepare(_)) {
                assert!(requested.is_empty());
            }
            if matches!(
                failure,
                BatchFailure::Callback(17) | BatchFailure::Disconnect(_) | BatchFailure::Late(_)
            ) {
                assert!(!events.iter().any(|event| event.0 == "inspect"));
            }
            let mut eraser = FakeEraser {
                events: Vec::new(),
                fail: None,
            };
            assert!(!erase_all(&mut eraser, &buffers, true));
            assert!(
                buffers
                    .iter()
                    .all(|buffer| buffer.buffer.words.borrow().iter().all(|b| *b == 0))
            );
        }
    }
    #[test]
    fn group_unmap_failure_retires_and_cannot_accept_even_after_successful_inspection() {
        let buffers = fake_readable(65);
        let mut mapper = FakeMapper::new(Some(BatchFailure::Unmap(17)));
        inspect_readable_groups(&mut mapper, &buffers, &[]).unwrap();
        assert!(mapper.tracker.retired.load(Ordering::Acquire));
        assert_eq!(
            mapper.tracker.reason(Failure::CleanupRejected),
            Failure::Unmap.code()
        );
        let mut eraser = FakeEraser {
            events: Vec::new(),
            fail: None,
        };
        assert!(!erase_all(
            &mut eraser,
            &buffers,
            mapper.tracker.retired.load(Ordering::Acquire)
        ));
    }
    #[test]
    fn nested_error_scope_raii_uses_lifo_on_early_return_and_panic() {
        use std::rc::Rc;
        struct Scope(u32, Rc<RefCell<Vec<u32>>>);
        impl Drop for Scope {
            fn drop(&mut self) {
                self.1.borrow_mut().push(self.0);
            }
        }
        struct Owner(Vec<Scope>);
        impl Drop for Owner {
            fn drop(&mut self) {
                drop_lifo(&mut self.0);
            }
        }
        for panic in [false, true] {
            let events = Rc::new(RefCell::new(Vec::new()));
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _owner = Owner((0..3).map(|id| Scope(id, events.clone())).collect());
                if panic {
                    panic!("synthetic scope unwind");
                }
            }));
            assert_eq!(*events.borrow(), [2, 1, 0]);
        }
    }
    #[test]
    fn checked_copy_bounds_reject_before_any_device_operation() {
        assert_eq!(copy_bounds(13, 8, 24, 13).unwrap(), 16);
        assert!(copy_bounds(13, 8, 23, 13).is_err());
        assert!(copy_bounds(13, 9, 25, 13).is_err());
        assert!(copy_bounds(13, 8, 24, 17).is_err());
        assert!(copy_bounds(4, u64::MAX - 3, u64::MAX, 4).is_err());
        assert!(copy_bounds(u64::MAX, 0, u64::MAX, 1).is_err());
        assert!(copy_bounds(0, 0, 0, 0).is_err());
    }
    #[test]
    fn counter_epoch_exhaustion_never_releases_or_resets_a_dirty_ledger() {
        let mut l = Ledger::new(MIN_BYTES).unwrap();
        l.reserve(Kind::Upload, 1).unwrap();
        l.epoch = u64::MAX;
        assert!(l.after_erasure().is_err());
        assert_eq!(l.charged, MIN_BYTES);
        assert_eq!(l.totals, [MIN_BYTES, 0, 0]);
    }
}

#[cfg(test)]
mod clean_readback_tests {
    use super::*;
    #[test]
    fn reuse_requires_explicit_batched_info_and_strict_encoding() {
        for raw in [None, Some("0")] {
            assert!(!configured_readback_reuse(raw, false).unwrap());
        }
        assert!(configured_readback_reuse(Some("1"), true).unwrap());
        for raw in [Some("1"), Some("2"), Some(""), Some("true")] {
            assert!(configured_readback_reuse(raw, false).is_err());
        }
    }
    #[test]
    fn clean_owner_charge_moves_exactly_and_remains_inside_original_budget() {
        let mut ledger = Ledger::new(1024).unwrap();
        ledger.reserve(Kind::Readback, 512).unwrap();
        assert_eq!(ledger.after_erasure().unwrap().readback_bytes, 512);
        let mut pool = CleanReadbacks::new(true).unwrap();
        assert!(pool.offer(7u32, 512, &mut ledger).unwrap());
        assert_eq!(ledger.clean_bytes, 512);
        assert_eq!(ledger.charged, 0);
        assert!(ledger.check(513).is_err());
        assert_eq!(pool.take(256, &mut ledger).unwrap(), None);
        assert_eq!(pool.take(512, &mut ledger).unwrap(), Some(7));
        assert_eq!(ledger.clean_bytes, 0);
        assert_eq!(ledger.charged, 512);
        assert_eq!(ledger.totals, [0, 0, 512]);
        ledger.reserve(Kind::Uniform, 512).unwrap();
        assert!(ledger.reserve(Kind::Readback, 256).is_err());
    }
    #[test]
    fn clean_pool_is_bounded_and_unselected_drops_owners() {
        let mut ledger = Ledger::new(32 * 1024 * 1024).unwrap();
        let mut disabled = CleanReadbacks::new(false).unwrap();
        assert!(!disabled.offer(0u32, 256, &mut ledger).unwrap());
        assert_eq!(ledger.clean_bytes, 0);
        let mut pool = CleanReadbacks::new(true).unwrap();
        for i in 0..CLEAN_READBACK_COUNT {
            assert!(
                pool.offer(
                    i,
                    CLEAN_READBACK_BYTES / CLEAN_READBACK_COUNT as u64,
                    &mut ledger
                )
                .unwrap()
            );
        }
        assert_eq!(ledger.clean_bytes, CLEAN_READBACK_BYTES);
        assert!(!pool.offer(99, 256, &mut ledger).unwrap());
        assert_eq!(pool.buffers.len(), CLEAN_READBACK_COUNT);
        assert!(ledger.activate_clean_readback(129).is_err());
        assert_eq!(ledger.clean_bytes, CLEAN_READBACK_BYTES);
    }
    #[test]
    fn rejected_transfer_does_not_mutate_clean_owner_or_ledger() {
        let mut ledger = Ledger::new(512).unwrap();
        let mut pool = CleanReadbacks::new(true).unwrap();
        assert!(pool.offer(7u32, 512, &mut ledger).unwrap());
        ledger.charged = 256; // Deliberately corrupt outside the admitted envelope.
        assert!(pool.take(512, &mut ledger).is_err());
        assert_eq!(pool.buffers, [(7, 512)]);
        assert_eq!(ledger.clean_bytes, 512);
        assert_eq!(ledger.charged, 256);
    }
}

#[cfg(test)]
mod clean_readback_metadata_tests {
    use super::*;
    #[test]
    fn actual_wgpu_pool_metadata_fits_original_four_kib_envelope() {
        let pool = CleanReadbacks::<wgpu::Buffer>::new(true).unwrap();
        assert!(
            pool.buffers.capacity() * std::mem::size_of::<(wgpu::Buffer, u64)>()
                + std::mem::size_of_val(&pool)
                <= 4096
        );
        assert!(CleanReadbacks::<[u8; 4096]>::new(true).is_err());
    }
}

#[cfg(test)]
mod gpu_upload_clear_policy_tests {
    use super::*;
    #[test]
    fn requires_explicit_compatible_capability_and_batched_info() {
        for raw in [None, Some("0")] {
            assert!(!configured_gpu_upload_clear(raw, false, false).unwrap());
        }
        assert!(configured_gpu_upload_clear(Some("1"), true, true).unwrap());
        for (batch, primary) in [(false, false), (false, true), (true, false)] {
            assert!(configured_gpu_upload_clear(Some("1"), batch, primary).is_err());
        }
        for raw in [Some("2"), Some(""), Some("true")] {
            assert!(configured_gpu_upload_clear(raw, true, true).is_err());
        }
    }
    #[test]
    fn only_primary_uploads_change_usage_and_receive_direct_full_inspection() {
        for kind in [Kind::Upload, Kind::Uniform, Kind::Readback] {
            let ordinary = allocation_usage(kind, false);
            let primary = allocation_usage(kind, true);
            if kind == Kind::Upload {
                assert_eq!(
                    primary,
                    ordinary | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
                );
                assert!(needs_audit_stage(kind, false));
                assert!(!needs_audit_stage(kind, true));
            } else {
                assert_eq!(primary, ordinary);
                assert_eq!(
                    needs_audit_stage(kind, false),
                    needs_audit_stage(kind, true)
                );
            }
        }
    }
}

#[cfg(test)]
mod inline_readback_policy_tests {
    use super::*;
    #[test]
    fn inline_wait_requires_strict_explicit_batched_and_inline_dispatch() {
        for raw in [None, Some("0")] {
            assert!(!configured_inline_readback(raw, false, false).unwrap());
        }
        assert!(configured_inline_readback(Some("1"), true, true).unwrap());
        for (batch, inline) in [(false, false), (true, false), (false, true)] {
            assert!(configured_inline_readback(Some("1"), batch, inline).is_err());
        }
        for raw in [Some("2"), Some(""), Some("true")] {
            assert!(configured_inline_readback(raw, true, true).is_err());
        }
    }
}
