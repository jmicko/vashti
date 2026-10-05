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
use std::time::Duration;

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
}
const FAILURE_CODES: [&str; 28] = [
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
    /// Check the FULL source span and any extra owned staging. No bytes escape.
    fn inspect_full(
        &mut self,
        buffer: &Self::Buffer,
        kind: Kind,
    ) -> Result<AuditCounts, ServerError>;
}
fn accepts_clean_epoch(initial_erasure: bool, inspection: bool, final_erasure: bool) -> bool {
    initial_erasure && inspection && final_erasure
}
fn inspect_all<A: AuditBackend>(
    audit: &mut A,
    buffers: &[Tracked<A::Buffer>],
) -> Result<AuditCounts, ServerError> {
    let mut counts = AuditCounts::default();
    for tracked in buffers {
        counts.add(audit.inspect_full(&tracked.buffer, tracked.kind)?)?;
    }
    Ok(counts)
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
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            buffer.map_async(wgpu::MapMode::Write, .., move |r| {
                let _ = sender.try_send(r);
            });
            if !self.io.poll(None) || !matches!(receiver.try_recv(), Ok(Ok(()))) {
                self.io.leases.record(Failure::UploadMap);
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

/// The inspection hook owns all extra staging in the SAME bounded ledger.
/// Public weights are never passed here; it cannot return mapped/request bytes.
struct NativeAudit<'a>(&'a mut TransientIo);
impl NativeAudit<'_> {
    fn inspect_readable(&mut self, buffer: &wgpu::Buffer) -> Result<(), ServerError> {
        struct Unmap<'a> {
            buffer: &'a wgpu::Buffer,
            retired: Arc<LeaseTracker>,
        }
        impl Drop for Unmap<'_> {
            fn drop(&mut self) {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.buffer.unmap()))
                    .is_err()
                {
                    self.retired.retire_with(Failure::Unmap);
                }
            }
        }
        let scopes = IoScopes::new(&self.0.device);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        // Guard established before requesting a map; panic/error/cancellation
        // aborts pending maps, with the tracked buffer retained for erasure.
        let _unmap = Unmap {
            buffer,
            retired: self.0.leases.clone(),
        };
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.try_send(result);
        });
        scopes.finish(self.0);
        if !self.0.poll(None) || !matches!(receiver.try_recv(), Ok(Ok(()))) {
            self.0.leases.record(Failure::AuditMap);
            return Err(fault(self.0.leases.reason(Failure::AuditMap)));
        }
        let view = buffer.get_mapped_range(..);
        if view.len() as u64 != buffer.size() || view.iter().any(|byte| *byte != 0) {
            self.0.leases.record(Failure::AuditNonzero);
            return Err(fault(self.0.leases.reason(Failure::AuditNonzero)));
        }
        // View drops before the unmap guard. No copy/to_vec/format/log exists.
        Ok(())
    }
}
impl AuditBackend for NativeAudit<'_> {
    type Buffer = wgpu::Buffer;
    fn inspect_full(
        &mut self,
        source: &wgpu::Buffer,
        kind: Kind,
    ) -> Result<AuditCounts, ServerError> {
        let mut counts = AuditCounts {
            bytes: source.size(),
            buffers: 1,
        };
        if kind == Kind::Readback {
            self.inspect_readable(source)?;
        } else {
            // MAP_WRITE uploads cannot be read directly. Copy full source
            // spans into an explicitly owned/tracked MAP_READ stage, including
            // padding, and inspect that stage. No opaque queue upload staging.
            let scopes = IoScopes::new(&self.0.device);
            self.0.ledger.check(span(source.size())?).map_err(|_| {
                self.0.leases.record(Failure::AuditBudget);
                fault(self.0.leases.reason(Failure::AuditBudget))
            })?;
            let stage = self
                .0
                .allocate(Kind::Readback, source.size())
                .inspect_err(|_| {
                    self.0.leases.record(Failure::AuditAllocation);
                })?;
            let mut encoder =
                self.0
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CubeCL private zero audit"),
                    });
            encoder.clear_buffer(&stage, 0, None);
            encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
            let submitted = self.0.queue.submit([encoder.finish()]);
            scopes.finish(self.0);
            if !self.0.poll(Some(submitted)) {
                return Err(fault(self.0.leases.reason(Failure::DrainTimeout)));
            }
            self.inspect_readable(&stage)?;
            // stage.span is equal to source.span (both bounded powers of two),
            // and was itself checked in FULL above. It must also be erased.
            counts.add(AuditCounts {
                bytes: stage.size(),
                buffers: 1,
            })?;
        }
        Ok(counts)
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
        Ok(Self {
            device,
            queue,
            ledger: Ledger::new(budget)?,
            buffers: Vec::new(),
            leases: Arc::default(),
            errors: Vec::new(),
        })
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
    pub fn leases(&self) -> usize {
        self.leases.count.load(Ordering::Acquire)
    }
    fn allocate(&mut self, kind: Kind, logical: u64) -> Result<wgpu::Buffer, ServerError> {
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
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: index,
            timeout: Some(POLL_TIMEOUT),
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
                inspect_all(&mut NativeAudit(self), &buffers)
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
        assert_eq!(FAILURE_CODES.len(), Failure::CleanupRejected as usize + 1);
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

    struct FakeBuffer {
        kind: Kind,
        words: RefCell<Vec<u8>>,
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
                    words: RefCell::new(vec![0xa7; MIN_BYTES as usize]),
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
    }
    impl AuditBackend for FakeAudit<'_> {
        type Buffer = FakeBuffer;
        fn inspect_full(
            &mut self,
            source: &FakeBuffer,
            kind: Kind,
        ) -> Result<AuditCounts, ServerError> {
            let mut count = AuditCounts {
                bytes: source.words.borrow().len() as u64,
                buffers: 1,
            };
            if kind == Kind::Readback {
                if source.words.borrow().iter().any(|b| *b != 0) {
                    return Err(fault("synthetic audit nonzero"));
                }
            } else {
                // Same bounded ledger reserve path as the native owned stage.
                let size = self.ledger.reserve(Kind::Readback, count.bytes)?;
                let stage = FakeBuffer {
                    kind: Kind::Readback,
                    words: RefCell::new(vec![0; size as usize]),
                };
                stage
                    .words
                    .borrow_mut()
                    .copy_from_slice(&source.words.borrow());
                self.stages.push(Tracked {
                    buffer: stage,
                    kind: Kind::Readback,
                });
                let stage = &self.stages.last().unwrap().buffer;
                if self.fail == Some("map") {
                    return Err(fault("synthetic audit map failed"));
                }
                if self.fail == Some("nonzero") {
                    *stage.words.borrow_mut().last_mut().unwrap() = 0x6d;
                }
                if stage.words.borrow().iter().any(|b| *b != 0) {
                    return Err(fault("synthetic audit nonzero"));
                }
                count.add(AuditCounts {
                    bytes: size,
                    buffers: 1,
                })?;
            }
            Ok(count)
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
