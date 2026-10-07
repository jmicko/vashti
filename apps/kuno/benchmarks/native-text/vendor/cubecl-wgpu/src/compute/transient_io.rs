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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
            totals: [0; 3],
            epoch: 1,
        })
    }
    fn check(&self, bytes: u64) -> Result<(), ServerError> {
        if self
            .charged
            .checked_add(bytes)
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
            let mut data = PrivateHostAllocation::zeroed(self.logical)
                .map_err(|_| fault("private readback host allocation failed"))?;
            data.as_mut_slice().copy_from_slice(&view[..self.logical]);
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
}
pub(crate) struct TransientIo {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ledger: Ledger,
    buffers: Vec<Tracked>,
    leases: Arc<LeaseTracker>,
    errors: Vec<ErrorFuture>,
    info_policy: Option<InfoPolicy>,
    info_slab: Option<InfoSlab>,
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
fn inspect_readable_groups<M: AuditMapping>(
    mapper: &mut M,
    originals: &[Tracked<M::Buffer>],
    stages: &[Tracked<M::Buffer>],
) -> Result<(), ServerError> {
    if stages.iter().any(|stage| stage.kind != Kind::Readback) {
        return Err(fault("private audit stage kind"));
    }
    let mut group = [None; AUDIT_MAP_GROUP];
    let mut count = 0;
    for tracked in originals
        .iter()
        .filter(|tracked| tracked.kind == Kind::Readback)
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
    fn erase_upload_full_span(&mut self, buffer: &Self::Buffer) -> bool;
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
    for tracked in buffers {
        if tracked.kind == Kind::Upload {
            failed |= !eraser.erase_upload_full_span(&tracked.buffer);
        }
    }
    for tracked in buffers {
        if tracked.kind != Kind::Upload {
            failed |= !eraser.clear_copy_destination_full_span(&tracked.buffer);
        }
    }
    failed |= !eraser.finish_and_accept_errors();
    !failed
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
    fn erase_upload_full_span(&mut self, buffer: &wgpu::Buffer) -> bool {
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Mapping, polling and callback delivery share one bounded window.
            // No second map attempt or extra full timeout is permitted.
            let deadline = Instant::now() + POLL_TIMEOUT;
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            buffer.map_async(wgpu::MapMode::Write, .., move |r| {
                let _ = sender.try_send(r);
            });
            if !self.io.poll_map_until(deadline) {
                self.io.leases.record(Failure::UploadMap);
                return false;
            }
            if let Err(category) = await_map_callback(&receiver, deadline) {
                self.io.leases.record(category.upload());
                return false;
            }
            {
                let mut view = buffer.get_mapped_range_mut(..);
                let pointer = view.slice(..).as_raw_element_ptr().as_ptr();
                for i in 0..view.len() {
                    // SAFETY: exclusively mapped full WRITE region retained.
                    unsafe { pointer.add(i).write_volatile(0) };
                }
                std::sync::atomic::compiler_fence(Ordering::SeqCst);
            }
            buffer.unmap();
            true
        }));
        if matches!(attempted, Ok(true)) {
            true
        } else {
            if attempted.is_err() {
                self.io.leases.record(Failure::UploadWipe);
            }
            // Always abort failed/pending mapping. Never report success here.
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| buffer.unmap())).is_err() {
                self.io.leases.record(Failure::Unmap);
            }
            false
        }
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
struct NativeAudit<'a> {
    io: &'a mut TransientIo,
    encoder: Option<wgpu::CommandEncoder>,
    scopes: Option<IoScopes>,
}
impl<'a> NativeAudit<'a> {
    fn new(io: &'a mut TransientIo) -> Self {
        let scopes = IoScopes::new(&io.device);
        let encoder = io
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CubeCL private zero audit"),
            });
        Self {
            io,
            encoder: Some(encoder),
            scopes: Some(scopes),
        }
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
        if view.len() as u64 != mapping.buffer.size() || view.iter().any(|byte| *byte != 0) {
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
        if kind != Kind::Readback {
            // Charge the same complete MAP_READ span before any copy, and keep
            // its owner in io.buffers even if later preparation/submission fails.
            self.io.ledger.check(span(source.size())?).map_err(|_| {
                self.io.leases.record(Failure::AuditBudget);
                fault(self.io.leases.reason(Failure::AuditBudget))
            })?;
            let stage = self
                .io
                .allocate_audit(Kind::Readback, source.size())
                .inspect_err(|_| {
                    self.io.leases.record(Failure::AuditAllocation);
                })?;
            let encoder = self
                .encoder
                .as_mut()
                .ok_or_else(|| fault("private audit encoder missing"))?;
            encoder.clear_buffer(&stage, 0, None);
            encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
            // These are exactly the old source+stage inspection counts. Both
            // remain owned through full inspection and final checked erasure.
            counts.add(AuditCounts {
                bytes: stage.size(),
                buffers: 1,
            })?;
        }
        Ok(counts)
    }
    fn inspect_prepared(&mut self, originals: &[Tracked<wgpu::Buffer>]) -> Result<(), ServerError> {
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
        inspect_readable_groups(&mut mapper, originals, &self.io.buffers)
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
                leases: Arc::default(),
                errors: Vec::new(),
                info_policy,
                info_slab: None,
            })
        }
    }
    pub fn retired(&self) -> bool {
        self.leases.retired.load(Ordering::Acquire)
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
        if self.info_policy.is_some() {
            let full = span(logical)?;
            let mut group = [0; 3];
            group[kind.slot()] = full;
            self.ledger
                .check_info_audit_group(group[0], group[1], group[2])
                .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        }
        self.allocate_audit(kind, logical)
    }
    fn allocate_audit(&mut self, kind: Kind, logical: u64) -> Result<wgpu::Buffer, ServerError> {
        self.active()?;
        self.buffers.try_reserve(1).map_err(|_| {
            self.leases.record(Failure::AllocationMetadata);
            fault(self.leases.reason(Failure::AllocationMetadata))
        })?;
        let owned = self.ledger.reserve(kind, logical).inspect_err(|_| {
            self.leases.record(Failure::AllocationBudget);
        })?;
        let usage = match kind {
            Kind::Upload => wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            Kind::Uniform => {
                wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
            }
            Kind::Readback => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeCL owned private I/O"),
            size: owned,
            usage,
            mapped_at_creation: kind == Kind::Upload,
        });
        // Retain full ownership before any initialization/mapping/copy can fail.
        self.buffers.push(Tracked {
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
        let stage = self.allocate(Kind::Upload, copy_len)?;
        {
            let mut view = stage.get_mapped_range_mut(..);
            // Initialize the FULL mapped API-visible span, including padding.
            view.slice(..).fill(0);
            view.slice(..data.len()).copy_from_slice(data);
        }
        stage.unmap();
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
            let buffer = self.allocate(Kind::Uniform, plan.target_bytes)?;
            encoder.clear_buffer(&buffer, 0, None);
            if plan.slab {
                self.info_slab = Some(InfoSlab {
                    buffer: buffer.clone(),
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
            self.erase_checked(previous_error, inspect)
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
        if self.info_policy.is_some() && inspect {
            // Check ALL duplicate full-span stages before any erase/audit
            // mutation. Never recurse into a checkpoint to obtain space.
            self.ledger
                .check_info_audit_group(0, 0, 0)
                .inspect_err(|_| self.leases.record(Failure::AuditBudget))?;
        }
        // Move bookkeeping, never GPU bytes. Restore ALL original and audit
        // stages before handling any panic/error. Failed audit never releases.
        let mut buffers = std::mem::take(&mut self.buffers);
        if inspect {
            // At most one verification stage per original upload/uniform.
            // Pre-reserve pointer bookkeeping BEFORE staging can contain any
            // bytes. Final ownership restoration then cannot allocate/panic.
            let stages = buffers.iter().filter(|b| b.kind != Kind::Readback).count();
            if buffers.try_reserve_exact(stages).is_err() {
                self.leases.record(Failure::AuditBookkeeping);
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.erase_set(&buffers, true)
                }));
                self.buffers = buffers;
                return Err(fault(self.leases.reason(Failure::AuditBookkeeping)));
            }
        }
        let initial = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.erase_set(&buffers, previous_error)
        }));
        if initial.is_err() {
            self.leases.record(Failure::Panic);
        }
        let initial_ok = matches!(initial, Ok(true));
        let inspected = if inspect && initial_ok {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                inspect_all(&mut NativeAudit::new(self), &buffers)
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
        self.info_slab = None; // No clean pool or cross-epoch owner reuse.
        self.buffers.clear(); // ONLY after full erase + optional audit acceptance
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
    use super::*;
    use std::cell::RefCell;

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
        fn erase_upload_full_span(&mut self, b: &FakeBuffer) -> bool {
            assert_eq!(b.kind, Kind::Upload);
            self.events.push("map-write-full-upload");
            if self.fail == Some("upload-map") {
                return false;
            }
            b.words.borrow_mut().fill(0);
            self.events.push("unmap-flush-upload");
            true
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
        Request(usize),
        Poll(usize),
        Callback(usize),
        Disconnect(usize),
        Late(usize),
        Panic(usize),
        Nonzero(usize),
        Unmap(usize),
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
    fn fake_readable(count: usize) -> Vec<Tracked<FakeBuffer>> {
        (0..count)
            .map(|_| Tracked {
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
