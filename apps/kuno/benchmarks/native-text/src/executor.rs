//! Operator laboratory only: bounded packed model cache + Rust GPU matvec.
//! No HTTP path, arbitrary prompts, production logging or installation.
use crate::{
    expert_banks::{self, ExpertBanks},
    main_admission::{self, MainAdmission},
    quant_matvec,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::cpu::Matrix;
use kuno_text::model::{MatrixExecutor, RoutedWeights};
use kuno_text::{Error, GgufFile, Result, TensorView};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use zeroize::Zeroize;

#[path = "resident_moe_batch.rs"]
pub(crate) mod batch_moe;

const TILE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const RESIDENT_IO_MAX: u64 = 64 * 1024 * 1024;
/// Separate runtime upload/uniform/readback ownership. Not main-pool telemetry.
pub(crate) const RESIDENT_RUNTIME_IO_MAX: u64 = 64 * 1024 * 1024;

/// Public static-projection specialization, never selected from active count.
/// Existing MAX8 workspaces and grouped MoE retain their original capacities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum NativeProjectionTile {
    One,
    Four,
    #[default]
    Eight,
}
impl NativeProjectionTile {
    pub(crate) fn new(tile: usize) -> Result<Self> {
        match tile {
            1 => Ok(Self::One),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            _ => Err(Error::InvalidFormat("native public projection tile")),
        }
    }
    fn tokens(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

#[allow(dead_code)] // Shared public-only construction and numerical probe seam.
#[path = "public_host_bank_completion.rs"]
mod public_host_bank_completion;
#[allow(dead_code)] // Fixed PUBLIC placement candidate; explicit parent registration.
#[path = "public_strata_bank_profile.rs"]
mod public_strata_bank_profile;
pub(crate) use public_host_bank_completion::{
    BankFillPublication, CompletedPublicHostBankFill, PublicHostBankFillPolicy,
};
#[allow(unused_imports)] // Controller separately consumes the immutable provenance pin.
pub(crate) use public_strata_bank_profile::{
    PUBLIC_STRATA_BANK_PROFILE_BYTES, PUBLIC_STRATA_BANK_PROFILE_SHA256,
};

fn validate_target_frozen_host_policy(
    frozen: bool,
    authenticated_identity: bool,
    rows: bool,
    profile: expert_banks::OriginalPackedProfile,
    capacity: usize,
    batch_router: bool,
    frozen_router_reserved: bool,
) -> Result<()> {
    if frozen_router_reserved
        && (!rows
            || profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || capacity != crate::daslab_profile::BANK_SLOTS
            || !batch_router)
        || authenticated_identity
            && (!frozen
                || !rows
                || profile != expert_banks::OriginalPackedProfile::DaslabIq3
                || capacity != crate::daslab_profile::BANK_SLOTS
                || batch_router != frozen_router_reserved)
        || rows && frozen && !authenticated_identity
    {
        return Err(Error::InvalidFormat(
            "native target frozen source identity policy",
        ));
    }
    Ok(())
}

fn validate_target_verify3_policy(
    selected: bool,
    frozen: Option<bool>,
    target_rows: bool,
    shared_dense: bool,
    members: usize,
    tile: usize,
) -> Result<()> {
    if frozen.is_some_and(|old| old != selected)
        || selected && (!target_rows || !shared_dense || members != 8 || tile != 8)
    {
        return Err(Error::InvalidFormat("native Verify3 construction policy"));
    }
    Ok(())
}

fn validate_target_verify3_fused_policy(
    selected: bool,
    frozen: Option<bool>,
    verify3: bool,
    target_rows: bool,
    members: usize,
    tile: usize,
) -> Result<()> {
    if frozen.is_some_and(|old| old != selected)
        || selected && (!verify3 || !target_rows || members != 8 || tile != 8)
    {
        return Err(Error::InvalidFormat(
            "native Verify3 finite construction policy",
        ));
    }
    Ok(())
}

fn activate_target_iq4_decoder(
    previous: crate::next_shared::Iq4Decoder,
    original_public_scope: bool,
) -> Result<crate::next_shared::Iq4Decoder> {
    if previous != crate::next_shared::Iq4Decoder::Original || !original_public_scope {
        return Err(Error::InvalidFormat(
            "IQ4 decoder immutable public activation",
        ));
    }
    Ok(crate::next_shared::Iq4Decoder::PackedRegisters)
}

fn target_verify3_iq4_spec(
    rows: usize,
    columns: usize,
    dtype: u32,
    decoder: crate::next_shared::Iq4Decoder,
) -> Result<crate::next_shared::MatrixSpec> {
    let original = crate::next_shared::MatrixSpec::new(rows, columns, dtype)?;
    if matches!(dtype, 20 | 23) {
        original.with_iq4_decoder(decoder)
    } else {
        // This changes only the two original IQ4 decoders. Other packed formats
        // retain their ordinary original source and receive no candidate credit.
        Ok(original)
    }
}

fn next_target_iq4_dispatch_count(
    previous: u64,
    actual: crate::next_shared::Iq4Decoder,
) -> Result<u64> {
    if actual == crate::next_shared::Iq4Decoder::PackedRegisters {
        previous.checked_add(1).ok_or(Error::ArithmeticOverflow)
    } else {
        Ok(previous)
    }
}

fn cleanup_failure_category(code: &str) -> Option<&'static str> {
    // Forward fixed infrastructure categories only, never backend descriptions,
    // shader source, metadata values or request content.
    const CODES: [&str; 37] = [
        "held_lease",
        "mutation_abandoned",
        "lease_overflow",
        "readback_map",
        "readback_conversion",
        "unmap",
        "prior_stream_error",
        "context_retired",
        "drain_timeout",
        "poll_submission_index",
        "upload_map",
        "upload_wipe",
        "destination_clear",
        "queue_submit",
        "gpu_out_of_memory",
        "gpu_validation",
        "gpu_internal",
        "audit_budget",
        "audit_allocation",
        "audit_map",
        "audit_nonzero",
        "audit_bookkeeping",
        "panic",
        "epoch_overflow",
        "allocation_budget",
        "allocation_metadata",
        "allocation_span",
        "cleanup_rejected",
        "upload_map_rejected",
        "upload_map_disconnected",
        "upload_map_callback_timeout",
        "audit_map_rejected",
        "audit_map_disconnected",
        "audit_map_callback_timeout",
        "cleanup_read_map_rejected",
        "cleanup_read_map_disconnected",
        "cleanup_read_map_callback_timeout",
    ];
    CODES.into_iter().find(|known| *known == code)
}

pub(crate) fn sanitize_resident_io(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cubecl::future::block_on(client.sanitize_transient_io())
    })) {
        Ok(Ok(_counts)) => Ok(()),
        Ok(Err(cubecl_runtime::server::ServerError::Generic { reason, .. })) => {
            Err(Error::InvalidFormat(
                reason
                    .strip_prefix("kuno_private_io_failure:")
                    .and_then(cleanup_failure_category)
                    .unwrap_or("resident runtime I/O cleanup failed"),
            ))
        }
        _ => Err(Error::InvalidFormat("resident runtime I/O cleanup failed")),
    }
}

fn checked_transient_cleanup(
    retired: &AtomicBool,
    cleanup: impl FnOnce() -> Result<()>,
) -> Result<()> {
    // Retired streams still get one best-effort attempt, never a clean receipt.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(cleanup));
    if !matches!(result, Ok(Ok(()))) || retired.load(Ordering::Acquire) {
        retired.store(true, Ordering::Release);
        return match result {
            Ok(Err(Error::InvalidFormat(code))) if cleanup_failure_category(code).is_some() => {
                Err(Error::InvalidFormat(code))
            }
            _ => Err(Error::InvalidFormat("resident runtime I/O cleanup failed")),
        };
    }
    Ok(())
}

pub(crate) fn checked_resident_io_audit(
    retired: &AtomicBool,
    audit: impl FnOnce() -> Result<TransientIoReceipt>,
) -> Result<()> {
    checked_transient_cleanup(retired, || {
        let receipt = audit()?;
        let total = receipt
            .upload_bytes
            .checked_add(receipt.uniform_bytes)
            .and_then(|n| n.checked_add(receipt.readback_bytes));
        if total.is_none_or(|n| n == 0 || n != receipt.inspected_bytes)
            || receipt.inspected_buffers == 0
        {
            return Err(Error::InvalidFormat(
                "resident runtime I/O audit incomplete",
            ));
        }
        Ok(())
    })
}

/// Same policy for the real admission and fault-injected host fixtures. The
/// completed raw lease is already gone. Audit success alone cannot release an
/// ordered source: cancellation/sequence health must still accept this boundary.
fn audited_source_cleanup<H: Clone>(
    sources: Option<&mut crate::ordered_public::PendingSources<H>>,
    retired: &AtomicBool,
    sequence_poison: &AtomicBool,
    cancel: &AtomicBool,
    audit: impl FnOnce() -> Result<TransientIoReceipt>,
) -> Result<()> {
    let complete = || {
        checked_resident_io_audit(retired, audit)?;
        // Healthy cancellation poisons the logical owner, not the GPU context.
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("GPU execution cancelled"));
        }
        if retired.load(Ordering::Acquire) || sequence_poison.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("resident sequence poisoned"));
        }
        Ok(())
    };
    match sources {
        Some(sources) => sources.after_completed_read(complete),
        None => complete(),
    }
}

/// A bookkeeping/unmap callback panic retains the previous retirement policy.
/// Ordinary cancellation/sequence errors still leave a healthy context usable
/// by a fresh logical owner after its separate erasure/cleanup has completed.
fn checked_audited_boundary(
    retired: &AtomicBool,
    operation: impl FnOnce() -> Result<()>,
) -> Result<()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(Err(error)) => Err(error),
        Ok(Ok(())) if !retired.load(Ordering::Acquire) => Ok(()),
        _ => {
            retired.store(true, Ordering::Release);
            Err(Error::InvalidFormat("resident runtime I/O cleanup failed"))
        }
    }
}

fn checkpoint_needed(intermediate: bool, token_checkpoint: bool, ordered_router: bool) -> bool {
    ordered_router || !intermediate || !token_checkpoint
}
fn deferred_bridge_allowed(
    transient: bool,
    token: bool,
    ordered: bool,
    retired: bool,
    cancelled: bool,
) -> bool {
    transient && token && !ordered && !retired && !cancelled
}
type OrderedSources = Rc<RefCell<crate::ordered_public::PendingSources<Handle>>>;
fn release_ordered_sources(
    client: &ComputeClient<WgpuRuntime>,
    sources: &OrderedSources,
) -> Result<()> {
    sources
        .borrow_mut()
        .after_completed_read(|| sanitize_resident_io(client))?;
    // No admission callback: this also runs while a readback holds its ledger.
    client.memory_cleanup();
    Ok(())
}

fn admission_error(_: main_admission::AdmissionError) -> Error {
    Error::LimitExceeded("resident main-pool admission")
}
/// Source geometry mirrors ExpertBanks::fill_plan's public BatchLayout:
/// whole bundles only, portable 65535*256-word grid, at most ten ranks.
fn fill_requests(parts: [u64; 3], misses: usize, batch: bool, window: usize) -> Result<Vec<u64>> {
    if misses > 10 || parts.iter().any(|&n| n == 0 || !n.is_multiple_of(4)) {
        return Err(Error::InvalidBufferLength);
    }
    if !batch {
        return Ok((0..misses).flat_map(|_| parts).collect());
    }
    expert_banks::configured_batch_bytes(Some(&window.to_string()))?;
    let bundle = parts.into_iter().try_fold(0_u64, |sum, n| {
        sum.checked_add(n).ok_or(Error::ArithmeticOverflow)
    })?;
    let per_chunk = ((window as u64).min(65_535 * 256 * 4) / bundle).min(10);
    if per_chunk == 0 {
        return Err(Error::LimitExceeded("public expert batch bundle"));
    }
    let mut left = misses as u64;
    let mut requests = Vec::new();
    while left != 0 {
        let count = left.min(per_chunk);
        requests.push(count.checked_mul(bundle).ok_or(Error::ArithmeticOverflow)?);
        left -= count;
    }
    Ok(requests)
}

fn guarded_allocation<T>(
    ledger: &mut MainAdmission,
    requests: &[u64],
    mut telemetry: impl FnMut() -> Result<u64>,
    allocate: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let result = (|| {
        ledger
            .observe_before(telemetry()?)
            .map_err(admission_error)?;
        ledger.precharge(requests).map_err(admission_error)?;
        let value = allocate()?;
        ledger
            .observe_after(telemetry()?)
            .map_err(admission_error)?;
        Ok(value)
    })();
    if result.is_err() {
        ledger.poison();
    }
    result
}
fn guarded_readback<T>(
    ledger: &mut MainAdmission,
    read: impl FnOnce() -> Result<T>,
    cleanup: impl FnOnce() -> Result<()>,
    telemetry: impl FnOnce() -> Result<u64>,
) -> Result<T> {
    let result = (|| {
        if ledger.is_poisoned() {
            return Err(Error::InvalidFormat("resident admission poisoned"));
        }
        let value = read()?;
        // Helpers return only owned host values. Every mapped Bytes/future
        // lease must be dropped before sanitizing or granting fence credit.
        cleanup()?;
        ledger
            .completed_fence(telemetry()?)
            .map_err(admission_error)?;
        Ok(value)
    })();
    if result.is_err() {
        ledger.poison();
    }
    result
}

/// Sticky logical failure is checked under the existing admission guard. A
/// pending owner's Drop sets only this atomic; it never reenters the mutex.
fn checked_sequence_guard<T>(
    ledger: &mut MainAdmission,
    poison: &AtomicBool,
    operation: impl FnOnce(&mut MainAdmission) -> Result<T>,
) -> Result<T> {
    if poison.load(Ordering::Acquire) {
        ledger.poison();
        return Err(Error::InvalidFormat("resident sequence poisoned"));
    }
    let value = operation(ledger);
    // A failed operation may itself drop a pending owner and set the sticky
    // flag. Preserve its original diagnostic; it still poisons this ledger.
    if value.is_err() {
        ledger.poison();
        return value;
    }
    if poison.load(Ordering::Acquire) {
        ledger.poison();
        return Err(Error::InvalidFormat("resident sequence poisoned"));
    }
    if value.is_err() {
        ledger.poison();
    }
    value
}

/// Shared host accounting only. It owns no activations/sequence state. The
/// exclusive resident owner serializes all allocations on the same stream;
/// selector callbacks need a clone while the projector borrows its executor.
#[derive(Clone)]
pub(crate) struct ResidentAdmission {
    client: ComputeClient<WgpuRuntime>,
    ledger: Arc<Mutex<MainAdmission>>,
    transient_io: bool,
    token_io_checkpoint: bool,
    ordered_sources: Option<OrderedSources>,
    retired: Arc<AtomicBool>,
    sequence_poison: Arc<AtomicBool>,
    deferred_router_reads: Arc<AtomicU64>,
    request_io_nonce: Arc<AtomicU64>,
    request_io_checkpoints: Arc<AtomicU64>,
    target_verify3_dense: bool,
    target_verify3_fused: bool,
    target_four_rows_reserved: bool,
}
/// Borrowed only inside the existing guarded model allocation callback. This
/// proves that its MAIN lock is already held; constructor checks must not
/// reacquire that lock. Not constructible/cloneable by callers or retained.
pub(crate) struct ModelConstructionAllocation<'a> {
    admission: &'a ResidentAdmission,
}
impl ModelConstructionAllocation<'_> {
    pub(crate) fn check(&self, admission: &ResidentAdmission) -> Result<()> {
        checked_construction_origin(
            &self.admission.ledger,
            &admission.ledger,
            &self.admission.retired,
            &admission.retired,
            &self.admission.sequence_poison,
            &admission.sequence_poison,
        )
    }
}
fn checked_construction_origin<T>(
    expected_ledger: &Arc<Mutex<T>>,
    actual_ledger: &Arc<Mutex<T>>,
    expected_retired: &Arc<AtomicBool>,
    actual_retired: &Arc<AtomicBool>,
    expected_poison: &Arc<AtomicBool>,
    actual_poison: &Arc<AtomicBool>,
) -> Result<()> {
    if !Arc::ptr_eq(expected_ledger, actual_ledger)
        || !Arc::ptr_eq(expected_retired, actual_retired)
        || !Arc::ptr_eq(expected_poison, actual_poison)
        || actual_retired.load(Ordering::Acquire)
        || actual_poison.load(Ordering::Acquire)
    {
        return Err(Error::InvalidFormat(
            "resident model construction origin rejected",
        ));
    }
    Ok(())
}

impl ResidentAdmission {
    /// The model obtains this capability only at a completed public laboratory
    /// boundary. Clones share the nonce; a different model cannot recycle it.
    pub(crate) fn begin_request_io_reuse(&self) -> Result<()> {
        self.begin_request_io_reuse_for(false)
    }
    /// Separate construction-sealed PUBLIC target capability. The backend
    /// rejects its fixed target stream through the ordinary count1 entry.
    pub(crate) fn begin_public_target_request_io_reuse(&self) -> Result<()> {
        self.begin_request_io_reuse_for(true)
    }
    fn begin_request_io_reuse_for(&self, public_target: bool) -> Result<()> {
        self.active()?;
        if !self.transient_io || !self.token_io_checkpoint {
            return Err(Error::InvalidFormat(
                "request I/O requires owned token checkpoints",
            ));
        }
        self.with_ledger(|ledger| {
            checked_sequence_guard(ledger, &self.sequence_poison, |_| {
                if self.request_io_nonce.load(Ordering::Acquire) != 0 {
                    return Err(Error::InvalidFormat("request I/O scope already active"));
                }
                let nonce = if public_target {
                    self.client.begin_public_target_request_io_reuse()
                } else {
                    self.client.begin_transient_io_reuse()
                }
                .map_err(|_| Error::InvalidFormat("request I/O scope rejected"))?;
                if nonce == 0 {
                    self.retired.store(true, Ordering::Release);
                    return Err(Error::InvalidFormat("request I/O scope missing"));
                }
                self.request_io_nonce.store(nonce, Ordering::Release);
                Ok(())
            })
        })
    }
    /// Actual admission capability state only; never expose its nonce.
    pub(crate) fn request_io_reuse_active(&self) -> bool {
        self.request_io_nonce.load(Ordering::Acquire) != 0
    }
    pub(crate) fn request_io_checkpoints(&self) -> u64 {
        self.request_io_checkpoints.load(Ordering::Relaxed)
    }
    /// A family may have completed its own guarded maps outside the admission
    /// callback. Join the same owning checkpoint only when its explicit request
    /// capability exists; ordinary family audits remain unchanged.
    pub(crate) fn checkpoint_request_io_reuse(&self, cancel: &AtomicBool) -> Result<bool> {
        if self.request_io_nonce.load(Ordering::Acquire) == 0 {
            return Ok(false);
        }
        self.readback_audited(cancel, || Ok(()))?;
        Ok(true)
    }
    /// Explicit close remains legal after logical cancellation/poison. Do not
    /// clear the capability unless the original full audit actually succeeded.
    pub(crate) fn end_request_io_reuse(&self) -> Result<()> {
        self.with_ledger(|_| {
            let nonce = self.request_io_nonce.load(Ordering::Acquire);
            if nonce == 0 {
                return Ok(());
            }
            checked_transient_cleanup(&self.retired, || {
                let receipt = self
                    .client
                    .end_transient_io_reuse(nonce)
                    .map_err(|_| Error::InvalidFormat("request I/O closing audit failed"))?;
                let bytes = receipt
                    .upload_bytes
                    .checked_add(receipt.uniform_bytes)
                    .and_then(|n| n.checked_add(receipt.readback_bytes));
                if bytes != Some(receipt.inspected_bytes)
                    || (receipt.inspected_bytes == 0) != (receipt.inspected_buffers == 0)
                {
                    return Err(Error::InvalidFormat("request I/O closing audit incomplete"));
                }
                self.request_io_nonce.store(0, Ordering::Release);
                Ok(())
            })
        })
    }
    fn with_ledger<T>(&self, f: impl FnOnce(&mut MainAdmission) -> Result<T>) -> Result<T> {
        let mut ledger = self
            .ledger
            .lock()
            .map_err(|_| Error::InvalidFormat("resident admission lock poisoned"))?;
        f(&mut ledger)
    }
    fn telemetry(&self) -> Result<u64> {
        self.client
            .memory_usage()
            .map(|m| m.bytes_reserved)
            .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))
    }
    pub(crate) fn poison(&self) {
        match self.ledger.lock() {
            Ok(mut ledger) => ledger.poison(),
            Err(error) => error.into_inner().poison(),
        }
    }
    fn active(&self) -> Result<()> {
        if self.retired.load(Ordering::Acquire) || self.sequence_poison.load(Ordering::Acquire) {
            self.poison();
            return Err(Error::InvalidFormat("resident GPU context retired"));
        }
        self.with_ledger(|ledger| {
            if ledger.is_poisoned() {
                Err(Error::InvalidFormat("resident admission poisoned"))
            } else {
                Ok(())
            }
        })
    }
    pub(crate) fn allocate<T>(&self, requests: &[u64], f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.with_ledger(|ledger| {
            checked_sequence_guard(ledger, &self.sequence_poison, |ledger| {
                guarded_allocation(ledger, requests, || self.telemetry(), f)
            })
        })
    }
    fn allocate_banks<T>(&self, requests: &[u64], f: impl FnOnce() -> Result<T>) -> Result<T> {
        if !crate::strata_port::enabled() {
            return self.allocate(requests, f);
        }
        self.with_ledger(|ledger| {
            checked_sequence_guard(ledger, &self.sequence_poison, |ledger| {
                ledger
                    .observe_before(self.telemetry()?)
                    .map_err(admission_error)?;
                ledger
                    .precharge_persistent(requests, self.client.properties().memory.alignment)
                    .map_err(admission_error)?;
                let value = f()?;
                ledger
                    .observe_after(self.telemetry()?)
                    .map_err(admission_error)?;
                Ok(value)
            })
        })
    }
    pub(crate) fn allocate_model<T>(
        &self,
        requests: &[u64],
        f: impl FnOnce(&ModelConstructionAllocation<'_>) -> Result<T>,
    ) -> Result<T> {
        self.allocate(requests, || {
            f(&ModelConstructionAllocation { admission: self })
        })
    }
    pub(crate) fn readback<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.readback_with_cadence(false, false, f)
    }
    /// Completed map/error acceptance is the ONE admission cleanup callback.
    /// An explicit request capability may retain dirty runtime owners until its
    /// full closing audit. Ordinary callers and ordered MAIN source release
    /// retain the original immediate full audit. No dirty runtime credit is
    /// granted by reuse; telemetry/MAIN fence credit remains last.
    pub(crate) fn readback_audited<T>(
        &self,
        cancel: &AtomicBool,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.transient_io
            || self.retired.load(Ordering::Acquire)
            || cancel.load(Ordering::Acquire)
        {
            self.poison();
            return Err(Error::InvalidFormat("resident audited readback inactive"));
        }
        self.with_ledger(|ledger| {
            checked_sequence_guard(ledger, &self.sequence_poison, |ledger| {
                guarded_readback(
                    ledger,
                    f,
                    || {
                        checked_audited_boundary(&self.retired, || {
                            let nonce = self.request_io_nonce.load(Ordering::Acquire);
                            // No MAIN source is credited by a dirty checkpoint.
                            // Rare public uploads keep their original full audit.
                            let sources_empty = self
                                .ordered_sources
                                .as_ref()
                                .is_none_or(|s| s.borrow().bytes() == 0);
                            if nonce != 0 && sources_empty {
                                self.client.recycle_transient_io(nonce).map_err(|_| {
                                    Error::InvalidFormat("request I/O checkpoint rejected")
                                })?;
                                if cancel.load(Ordering::Acquire)
                                    || self.retired.load(Ordering::Acquire)
                                    || self.sequence_poison.load(Ordering::Acquire)
                                {
                                    return Err(Error::InvalidFormat(
                                        "request I/O checkpoint invalidated",
                                    ));
                                }
                                self.request_io_checkpoints
                                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                                        n.checked_add(1)
                                    })
                                    .map_err(|_| Error::ArithmeticOverflow)?;
                                return Ok(());
                            }
                            let audit = || {
                                cubecl::future::block_on(
                                    self.client.sanitize_transient_io_audited(),
                                )
                                .map_err(|error| match error {
                                    cubecl_runtime::server::ServerError::Generic {
                                        reason, ..
                                    } => Error::InvalidFormat(
                                        reason
                                            .strip_prefix("kuno_private_io_failure:")
                                            .and_then(cleanup_failure_category)
                                            .unwrap_or("resident runtime I/O audit failed"),
                                    ),
                                    _ => Error::InvalidFormat("resident runtime I/O audit failed"),
                                })
                            };
                            if let Some(sources) = &self.ordered_sources {
                                audited_source_cleanup(
                                    Some(&mut *sources.borrow_mut()),
                                    &self.retired,
                                    &self.sequence_poison,
                                    cancel,
                                    audit,
                                )?;
                                self.client.memory_cleanup();
                                Ok(())
                            } else {
                                audited_source_cleanup::<Handle>(
                                    None,
                                    &self.retired,
                                    &self.sequence_poison,
                                    cancel,
                                    audit,
                                )
                            }
                        })
                    },
                    || self.telemetry(),
                )
            })
        })
    }
    /// Explicit token-owned intermediate bridge. All raw runtime buffers stay
    /// charged/tracked until mandatory final logit/fault audit or owning close.
    /// No map/source release, audit success or private erasure is claimed here.
    pub(crate) fn readback_count1_bridge<T>(
        &self,
        cancel: &AtomicBool,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.token_io_checkpoint {
            return self.readback_audited(cancel, f);
        }
        if !deferred_bridge_allowed(
            self.transient_io,
            self.token_io_checkpoint,
            self.ordered_sources.is_some(),
            self.retired.load(Ordering::Acquire),
            cancel.load(Ordering::Acquire),
        ) {
            self.poison();
            return Err(Error::InvalidFormat("deferred count1 bridge inactive"));
        }
        self.readback_intermediate(|| {
            let out = f()?;
            if !deferred_bridge_allowed(
                self.transient_io,
                self.token_io_checkpoint,
                self.ordered_sources.is_some(),
                self.retired.load(Ordering::Acquire),
                cancel.load(Ordering::Acquire),
            ) {
                return Err(Error::InvalidFormat("deferred count1 bridge invalidated"));
            }
            self.deferred_router_reads
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| Error::ArithmeticOverflow)?;
            Ok(out)
        })
    }
    pub(crate) fn deferred_router_reads(&self) -> u64 {
        self.deferred_router_reads.load(Ordering::Relaxed)
    }
    pub(crate) fn deferred_token_io(&self) -> bool {
        self.token_io_checkpoint
    }
    /// Completed intermediate maps still drop their guarded Bytes immediately.
    /// Deferred I/O remains tracked and charged in the separate hard runtime
    /// budget; main-pool fence credit cannot reclaim those runtime spans.
    pub(crate) fn readback_intermediate<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.readback_with_cadence(true, false, f)
    }
    /// The existing router read completes normally before bounded cleanup.
    /// Ordered public sources force this boundary even on cache-hit layers and
    /// under token-only cadence; upload/enqueue never invokes cleanup itself.
    pub(crate) fn readback_router<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.readback_with_cadence(true, self.ordered_sources.is_some(), f)
    }
    fn readback_with_cadence<T>(
        &self,
        intermediate: bool,
        ordered_router: bool,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if self.retired.load(Ordering::Acquire) {
            self.poison();
            return Err(Error::InvalidFormat("resident GPU context retired"));
        }
        self.with_ledger(|ledger| {
            checked_sequence_guard(ledger, &self.sequence_poison, |ledger| {
                guarded_readback(
                    ledger,
                    f,
                    || {
                        if checkpoint_needed(intermediate, self.token_io_checkpoint, ordered_router)
                        {
                            self.checkpoint()
                        } else {
                            Ok(())
                        }
                    },
                    || self.telemetry(),
                )
            })
        })
    }
    #[allow(dead_code)] // Shared native-family dependency; consumers arrive separately.
    pub(crate) fn sequence_poison(&self) -> Arc<AtomicBool> {
        self.sequence_poison.clone()
    }
    /// Identity only: dependency selectors must clean/credit the SAME stream
    /// client that owns their maps. No telemetry or nested admission lock.
    pub(crate) fn check_client(&self, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
        if !std::ptr::eq(self.client.info(), client.info()) {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
    pub(crate) fn transient_io(&self) -> bool {
        self.transient_io
    }
    pub(crate) fn retired_context(&self) -> Arc<AtomicBool> {
        self.retired.clone()
    }
    fn checkpoint(&self) -> Result<()> {
        if !self.transient_io {
            return Ok(());
        }
        checked_transient_cleanup(&self.retired, || {
            if let Some(sources) = &self.ordered_sources {
                release_ordered_sources(&self.client, sources)
            } else {
                sanitize_resident_io(&self.client)
            }
        })
    }
    /// Router table uploads charge the private I/O exclusive-owner ledger.
    /// A completed public activation must leave that ledger empty so the next
    /// request I/O reuse begin is legal without an intervening commit.
    pub(crate) fn checkpoint_public_router_activation(&self) -> Result<()> {
        self.checkpoint()
    }
    /// Explicit abandoned-owner teardown, outside any admission lock. A
    /// cancelled attention step may not have reached its next router read.
    /// Drain/erase ordered upload staging before the optional MAIN zero audit
    /// allocates inspection stages. This grants no MAIN fence credit, resets
    /// no poisoned ledger, and leaves pending sources retained on failure.
    pub(crate) fn cleanup_abandoned_ordered_sources(&self) -> Result<()> {
        if self.ordered_sources.is_none() {
            return Ok(());
        }
        self.checkpoint()
    }
    pub(crate) fn prewarm(
        &self,
        graph: kuno_text::model::BoundGraph<'_, '_>,
        public: &mut crate::resident_gr::PublicVectors,
    ) -> Result<u64> {
        self.with_ledger(|ledger| {
            let result = public.prewarm(graph, ledger);
            if result.is_err() {
                ledger.poison();
            }
            result
        })
    }
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    tensor: String,
    expert: usize,
    first: usize,
}
struct Cached {
    handle: Handle,
    words: usize,
    bytes: usize,
    last: u64,
}

/// The lifetime of this executor is one fixed synthetic experiment. Its cache
/// keys/ranks must never be persisted or reused as private request history.
pub struct GpuExecutor<'a> {
    pub(crate) packed_profile: expert_banks::OriginalPackedProfile,
    pub(crate) q3_empty_slot_fill: bool,
    pub(crate) q3_skip_empty_segments: bool,
    pub(crate) q3_subgroup_moe: bool,
    pub(crate) q3_integer_moe: bool,
    #[allow(dead_code)] // Inert until parent wires the separate original-MoE leaf.
    pub(crate) q3_shared_original_moe: bool,
    pub(crate) q6_integer_head: bool,
    pub(crate) q6_integer_verify_head: bool,
    pub(crate) q6_integer_verify_head_prepares: u64,
    pub(crate) q6_integer_verify_head_tiles: u64,
    pub(crate) q6_integer_verify_head_members: u64,
    pub(crate) resident_router_reserved: bool,
    pub(crate) target_rows_reserved: bool,
    pub(crate) target_four_rows_reserved: bool,
    pub(crate) target_four_rows_active: bool,
    pub(crate) target_verify4_dense_calls: u64,
    pub(crate) target_verify4_fused_calls: u64,
    pub(crate) target_batch_router_reserved: bool,
    pub(crate) target_frozen_host_router_reserved: bool,
    pub(crate) target_frozen_host_staging_reserved: bool,
    pub(crate) target_frozen_gu_integer_reserved: bool,
    pub(crate) target_frozen_down_integer_reserved: bool,
    pub(crate) target_frozen_host_source_identity:
        Option<crate::public_host_expert_plan::PlanIdentity>,
    pub(crate) target_verify3_dense: bool,
    pub(crate) target_verify3_dense_calls: u64,
    pub(crate) target_verify3_fused: bool,
    pub(crate) target_verify3_fused_calls: u64,
    target_iq4_decoder: crate::next_shared::Iq4Decoder,
    target_iq4_decoder_calls: u64,
    pub(crate) resident_banks_frozen: bool,
    pub(crate) resident_router_audits: u64,
    pub q6_integer_head_prepares: u64,
    pub q6_integer_head_tiles: u64,
    pub(crate) q3_expert_oracle: bool,
    pub(crate) q3_router_reference: Option<std::sync::Arc<crate::q3_router_reference::Reference>>,
    pub(crate) oracle_trace: Option<crate::oracle_trace::Trace>,
    pub(crate) adaptive: Option<crate::adaptive_experts::Cache>,
    pub(crate) secondary: Option<crate::secondary_experts::Secondary>,
    pub(crate) secondary_frequency: Option<crate::secondary_experts::Frequency>,
    pub shared_overlap_calls: u64,
    pub gpu_router: bool,
    pub fused_experts: bool,
    pub integer_experts: bool,
    pub gpu_expert_limit: usize,
    pub cpu_resident_expert_calls: u64,
    pub fused_expert_calls: u64,
    pub gpu_router_calls: u64,
    pub fast_dense: bool,
    pub fast_dense_calls: u64,
    pub fast_dense_fault: bool,
    pub fast_dense_fault_calls: u64,
    pub shared_kernels: bool,
    pub shared_dense_calls: u64,
    pub shared_gdn: bool,
    model: &'a GgufFile,
    pub block_kernel: bool,
    pub block_large_only: bool,
    pub block_moe: bool,
    pub fused_matrix_fault: bool,
    pub word_down: bool,
    pub batch_experts: bool,
    pub expert_batch_bytes: usize,
    pub expert_scatter: expert_banks::ScatterMode,
    pub gpu_gdn: bool,
    pub token_io_checkpoint: bool,
    pub ordered_public_upload: bool,
    ordered_sources: Option<OrderedSources>,
    pub small_cpu: crate::small_cpu::SmallCpu<'a>,
    pub small_cpu_enabled: bool,
    pub small_cpu_ns: u128,
    pub small_cpu_calls: u64,
    client: ComputeClient<WgpuRuntime>,
    cache: BTreeMap<Key, Cached>,
    cache_bytes: usize,
    budget: usize,
    tick: u64,
    banks: BTreeMap<String, ExpertBanks>,
    pub(crate) strata_global_sources: Option<std::sync::Arc<crate::strata_global_sources::Sources>>,
    batch_moe: Option<batch_moe::Registry>,
    native_projection_tile: NativeProjectionTile,
    native_projection_members: usize,
    bank_reserved: usize,
    bank_slots: usize,
    reserved_limit: usize,
    resident_admission: Option<ResidentAdmission>,
    transient_resident: Option<Arc<AtomicBool>>,
    pub by_tensor: BTreeMap<String, u128>,
    pub gdn_ns: u128,
    pub gdn_calls: u64,
    pub moe_ns: u128,
    pub moe_calls: u64,
    pub expert_hits: u64,
    pub expert_uploads: u64,
    pub uploaded_bytes: u64,
    pub cache_hits: u64,
    pub calls: u64,
    pub matrix_ns: u128,
    pub weight_read_ns: u128,
    pub weight_upload_ns: u128,
    pub gpu_wait_ns: u128,
}
#[allow(clippy::too_many_arguments)]
fn validate_q6_final_head(
    profile: expert_banks::OriginalPackedProfile,
    admitted: bool,
    member_capacity: usize,
    tile: usize,
    name: &str,
    dtype: u32,
    columns: usize,
    rows: usize,
    experts: usize,
    spans: &quant_matvec::matmul::BatchSpans,
) -> Result<()> {
    if !profile.is_low_bit()
        || !admitted
        || member_capacity != 1
        || tile != 1
        || name != "output.weight"
        || dtype != 14
        || columns != 2560
        || rows != 248320
        || experts != 1
        || spans.count() != 1
        || spans.packed_first() != 0
        || spans.input_first() != 0
        || spans.input_stride() != 2560
        || spans.output_first() != 0
        || spans.output_stride() != 248320
    {
        return Err(Error::InvalidFormat(
            "Q6 integer fixed-public final head contract",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_q6_verify_head(
    profile: expert_banks::OriginalPackedProfile,
    admitted: bool,
    member_capacity: usize,
    tile: usize,
    four_reserved: bool,
    four_active: bool,
    name: &str,
    dtype: u32,
    columns: usize,
    rows: usize,
    experts: usize,
    spans: &quant_matvec::matmul::BatchSpans,
) -> Result<()> {
    if profile != expert_banks::OriginalPackedProfile::DaslabIq3
        || !admitted
        || member_capacity != 8
        || tile != 8
        || four_active && !four_reserved
        || !(1..=if four_active {
            crate::strata_port::verify_capacity()
        } else {
            3
        })
            .contains(&spans.count())
        || name != "output.weight"
        || dtype != 14
        || columns != 2560
        || rows != 248320
        || experts != 1
        || spans.packed_first() != 0
        || spans.input_first() != 0
        || spans.input_stride() != 2560
        || spans.output_first() != 0
        || spans.output_stride() != 248320
    {
        return Err(Error::InvalidFormat(
            "Q6 integer fixed-public verifier head contract",
        ));
    }
    Ok(())
}
fn activate_q6_verify_head(previous: bool, scope: bool) -> Result<bool> {
    if previous || !scope {
        return Err(Error::InvalidFormat(
            "Q6 verifier head immutable public selection",
        ));
    }
    Ok(true)
}
fn validate_q6_verify_counters(
    selected: bool,
    prepares: u64,
    tiles: u64,
    members: u64,
) -> Result<()> {
    if !selected && (prepares != 0 || tiles != 0 || members != 0)
        || tiles != prepares.checked_mul(16).ok_or(Error::ArithmeticOverflow)?
        || members < prepares
        || members > prepares.checked_mul(4).ok_or(Error::ArithmeticOverflow)?
    {
        return Err(Error::InvalidFormat(
            "Q6 verifier head actual completed counters",
        ));
    }
    Ok(())
}

fn oracle_fixture_profile_matches(
    fixture: crate::oracle_trace::Fixture,
    profile: expert_banks::OriginalPackedProfile,
) -> bool {
    matches!(
        (fixture, profile),
        (
            crate::oracle_trace::Fixture::Q4Rainwater,
            expert_banks::OriginalPackedProfile::Q4
        ) | (
            crate::oracle_trace::Fixture::Q3Prefix23,
            expert_banks::OriginalPackedProfile::UnslothQ3
        ) | (
            crate::oracle_trace::Fixture::DaslabPrefix13,
            expert_banks::OriginalPackedProfile::DaslabIq3
        ) | (
            crate::oracle_trace::Fixture::Daslab32,
            expert_banks::OriginalPackedProfile::DaslabIq3
        )
    )
}
impl<'a> GpuExecutor<'a> {
    #[allow(dead_code)] // Original descriptor for the additive public mirror controller.
    pub(crate) fn public_host_model(&self) -> &'a GgufFile {
        self.model
    }

    pub fn new(
        model: &'a GgufFile,
        client: ComputeClient<WgpuRuntime>,
        budget: usize,
        bank_slots: usize,
        reserved_limit: usize,
    ) -> Result<Self> {
        Self::new_with_profile(
            model,
            client,
            budget,
            bank_slots,
            reserved_limit,
            expert_banks::OriginalPackedProfile::Q4,
        )
    }
    pub(crate) fn new_with_profile(
        model: &'a GgufFile,
        client: ComputeClient<WgpuRuntime>,
        budget: usize,
        bank_slots: usize,
        reserved_limit: usize,
        packed_profile: expert_banks::OriginalPackedProfile,
    ) -> Result<Self> {
        packed_profile.validate_capacity(bank_slots)?;
        let explicit_static = match std::env::var("KUNO_LAB_STATIC_CACHE_BYTES") {
            Ok(raw) => Some(expert_banks::configured_static_cache(Some(&raw), budget)?),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err(Error::InvalidFormat("laboratory static cache environment")),
        };
        let block = std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1");
        let batch = std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1");
        let word = std::env::var("KUNO_LAB_WORD_DOWN").as_deref() == Ok("1");
        if packed_profile == expert_banks::OriginalPackedProfile::Q4 {
            expert_banks::validate_segmented_configuration(
                bank_slots,
                explicit_static,
                budget,
                block,
                batch,
                word,
                true,
            )?;
        } else {
            expert_banks::validate_segmented_configuration_with_profile(
                bank_slots,
                explicit_static,
                budget,
                block,
                batch,
                word,
                true,
                packed_profile,
            )?;
        }
        expert_banks::validate_ordered_segmented(bank_slots)?;

        let expert_batch_bytes = match std::env::var("KUNO_LAB_EXPERT_BATCH_BYTES") {
            Ok(raw) => expert_banks::configured_batch_bytes(Some(&raw))?,
            Err(std::env::VarError::NotPresent) => expert_banks::configured_batch_bytes(None)?,
            Err(_) => {
                return Err(Error::InvalidFormat(
                    "KUNO_LAB_EXPERT_BATCH_BYTES must be integer 4..64 MiB",
                ));
            }
        };
        let expert_scatter = match std::env::var("KUNO_LAB_EXPERT_SCATTER") {
            Ok(raw) => expert_banks::configured_scatter(Some(&raw))?,
            Err(std::env::VarError::NotPresent) => expert_banks::configured_scatter(None)?,
            Err(_) => {
                return Err(Error::InvalidFormat(
                    "KUNO_LAB_EXPERT_SCATTER must be 0, 1 or 2",
                ));
            }
        };
        if !(TILE_BYTES..=16 * 1024 * 1024 * 1024).contains(&budget) {
            return Err(Error::LimitExceeded("laboratory GPU cache"));
        }
        let ordered_raw = match std::env::var("KUNO_LAB_ORDERED_PUBLIC_UPLOAD") {
            Ok(raw) => Some(raw),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err(Error::InvalidFormat("ordered public upload setting")),
        };
        let ordered_public_upload = crate::ordered_public::configured(
            ordered_raw.as_deref(),
            std::env::var("KUNO_LAB_TRANSIENT_IO").as_deref() == Ok("1"),
            std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
        )?;
        Ok(Self {
            packed_profile,
            q3_empty_slot_fill: false,
            q3_skip_empty_segments: false,
            q3_subgroup_moe: false,
            q3_integer_moe: false,
            q3_shared_original_moe: false,
            q6_integer_head: false,
            q6_integer_verify_head: false,
            q6_integer_verify_head_prepares: 0,
            q6_integer_verify_head_tiles: 0,
            q6_integer_verify_head_members: 0,
            resident_router_reserved: false,
            target_rows_reserved: false,
            target_four_rows_reserved: false,
            target_four_rows_active: false,
            target_verify4_dense_calls: 0,
            target_verify4_fused_calls: 0,
            target_batch_router_reserved: false,
            target_frozen_host_router_reserved: false,
            target_frozen_host_staging_reserved: false,
            target_frozen_gu_integer_reserved: false,
            target_frozen_down_integer_reserved: false,
            target_frozen_host_source_identity: None,
            target_verify3_dense: false,
            target_verify3_dense_calls: 0,
            target_verify3_fused: false,
            target_verify3_fused_calls: 0,
            target_iq4_decoder: crate::next_shared::Iq4Decoder::Original,
            target_iq4_decoder_calls: 0,
            resident_banks_frozen: false,
            resident_router_audits: 0,
            q6_integer_head_prepares: 0,
            q6_integer_head_tiles: 0,
            q3_expert_oracle: false,
            q3_router_reference: None,
            oracle_trace: None,
            adaptive: if std::env::var("KUNO_LAB_ADAPTIVE_EXPERTS").as_deref() == Ok("1") {
                Some(crate::adaptive_experts::Cache::new()?)
            } else {
                None
            },
            secondary: None,
            secondary_frequency: if std::env::var("KUNO_LAB_SECONDARY_EXPERTS").as_deref()
                == Ok("1")
            {
                Some(crate::secondary_experts::Frequency::new()?)
            } else {
                None
            },
            ordered_public_upload,
            ordered_sources: ordered_public_upload
                .then(|| Rc::new(RefCell::new(crate::ordered_public::PendingSources::new()))),
            small_cpu: crate::small_cpu::SmallCpu::new(model, 64 * 1024 * 1024)?,
            small_cpu_enabled: std::env::var("KUNO_LAB_SMALL_CPU").as_deref() == Ok("1"),
            small_cpu_ns: 0,
            small_cpu_calls: 0,
            block_kernel: std::env::var("KUNO_LAB_BLOCK_KERNEL").as_deref() != Ok("0"),
            block_large_only: std::env::var("KUNO_LAB_BLOCK_KERNEL").as_deref() != Ok("1"),
            block_moe: std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1"),
            fused_matrix_fault: std::env::var("KUNO_LAB_FUSED_MATRIX_FAULT").as_deref() == Ok("1"),
            word_down: std::env::var("KUNO_LAB_WORD_DOWN").as_deref() == Ok("1"),
            // Candidate batching stays opt-in until paired measurements win.
            batch_experts: std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
            expert_batch_bytes,
            expert_scatter,
            token_io_checkpoint: std::env::var("KUNO_LAB_IO_TOKEN_CHECKPOINT").as_deref()
                == Ok("1"),
            gpu_gdn: std::env::var("KUNO_LAB_CPU_GDN").as_deref() != Ok("1"),
            model,
            client,
            cache: BTreeMap::new(),
            cache_bytes: 0,
            budget,
            tick: 0,
            banks: BTreeMap::new(),
            strata_global_sources: None,
            batch_moe: None,
            shared_overlap_calls: 0,
            gpu_router: false,
            fused_experts: false,
            integer_experts: false,
            gpu_expert_limit: 10,
            cpu_resident_expert_calls: 0,
            fused_expert_calls: 0,
            gpu_router_calls: 0,
            fast_dense: false,
            fast_dense_calls: 0,
            fast_dense_fault: false,
            fast_dense_fault_calls: 0,
            shared_kernels: false,
            shared_dense_calls: 0,
            shared_gdn: false,
            native_projection_tile: NativeProjectionTile::default(),
            native_projection_members: quant_matvec::matmul::MAX_BATCH,
            bank_reserved: 0,
            bank_slots,
            reserved_limit,
            resident_admission: None,
            transient_resident: None,
            by_tensor: BTreeMap::new(),
            gdn_ns: 0,
            gdn_calls: 0,
            moe_ns: 0,
            moe_calls: 0,
            expert_hits: 0,
            expert_uploads: 0,
            uploaded_bytes: 0,
            cache_hits: 0,
            calls: 0,
            matrix_ns: 0,
            weight_read_ns: 0,
            weight_upload_ns: 0,
            gpu_wait_ns: 0,
        })
    }
    pub(crate) fn check_client(&self, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
        if !std::ptr::eq(self.client.info(), client.info()) {
            return Err(Error::InvalidFormat("resident GPU context mismatch"));
        }
        Ok(())
    }
    /// Adopt configuration performed ONCE by the fixed-stream lab runner,
    /// before any uniforms/readbacks. The SAME retirement flag belongs to the
    /// physical stream across fresh logical owners. No configuration/fallback.
    pub(crate) fn enable_transient_resident(&mut self, retired: Arc<AtomicBool>) -> Result<()> {
        if let Some(previous) = &self.transient_resident {
            previous.store(true, Ordering::Release);
            retired.store(true, Ordering::Release);
            self.poison_resident();
            return Err(Error::InvalidFormat("resident I/O adoption after use"));
        }
        self.transient_resident = Some(retired.clone());
        if self.resident_admission.is_some() || !self.cache.is_empty() || !self.banks.is_empty() {
            retired.store(true, Ordering::Release);
            self.poison_resident();
            return Err(Error::InvalidFormat("resident I/O adoption after use"));
        }
        if checked_transient_cleanup(&retired, || sanitize_resident_io(&self.client)).is_err() {
            return Err(Error::InvalidFormat(
                "resident I/O configuration unavailable",
            ));
        }
        Ok(())
    }
    /// Checked fixed-fixture shutdown. No ordinary sync on owned-runtime I/O;
    /// failure keeps the context retired and the admission ledger poisoned.
    pub(crate) fn close_transient_resident(&mut self) -> Result<()> {
        if let Some(mut trace) = self.oracle_trace.take() {
            trace.erase();
            println!("public_oracle_trace_closed=true");
        }
        if let Some(secondary) = &mut self.secondary {
            secondary.close()?;
        }
        if let Some(mut cache) = self.adaptive.take() {
            cache.erase();
            println!("public_adaptive_closed=true host_reservation=196608");
        }
        self.secondary = None;
        self.secondary_frequency = None;
        let retired = self
            .transient_resident
            .clone()
            .ok_or(Error::InvalidFormat("resident I/O ownership unavailable"))?;
        self.poison_resident();
        if let Some(r) = &self.batch_moe {
            r.abort();
        }
        checked_transient_cleanup(&retired, || {
            if let Some(sources) = &self.ordered_sources {
                release_ordered_sources(&self.client, sources)
            } else {
                sanitize_resident_io(&self.client)
            }
        })?;
        if let Some(sources) = &mut self.strata_global_sources {
            std::sync::Arc::get_mut(sources)
                .ok_or(Error::InvalidFormat("global source readers still live"))?
                .close()?;
        }
        self.strata_global_sources = None;
        self.cache.clear();
        self.cache_bytes = 0;
        self.banks.clear();
        self.resident_banks_frozen = false;
        self.target_frozen_host_source_identity = None;
        self.target_frozen_host_router_reserved = false;
        self.target_frozen_host_staging_reserved = false;
        self.target_frozen_gu_integer_reserved = false;
        self.target_frozen_down_integer_reserved = false;
        self.target_iq4_decoder = crate::next_shared::Iq4Decoder::Original;
        self.target_iq4_decoder_calls = 0;
        self.q6_integer_verify_head = false;
        self.q6_integer_verify_head_prepares = 0;
        self.q6_integer_verify_head_tiles = 0;
        self.q6_integer_verify_head_members = 0;
        self.batch_moe = None;
        self.bank_reserved = 0;
        // All previous commands completed before public cache handles release.
        let cleaned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.client.memory_cleanup();
        }));
        if cleaned.is_err() {
            retired.store(true, Ordering::Release);
            return Err(Error::InvalidFormat("resident public cache cleanup failed"));
        }
        checked_transient_cleanup(&retired, || sanitize_resident_io(&self.client))
    }
    pub(crate) fn ordered_source_peak_bytes(&self) -> u64 {
        self.ordered_sources
            .as_ref()
            .map_or(0, |sources| sources.borrow().peak())
    }
    pub(crate) fn ordered_source_pending_bytes(&self) -> u64 {
        self.ordered_sources
            .as_ref()
            .map_or(0, |sources| sources.borrow().bytes())
    }
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes + self.bank_reserved
    }
    pub(crate) fn check_lab_oracle_fixture(
        &self,
        fixture: crate::oracle_trace::Fixture,
    ) -> Result<()> {
        if !oracle_fixture_profile_matches(fixture, self.packed_profile) {
            return Err(Error::InvalidFormat(
                "public oracle fixture profile mismatch",
            ));
        }
        Ok(())
    }
    /// The model issues the sole completed-boundary proof. This diagnostic has
    /// no adaptive/secondary policy and never admits an arbitrary route trace.
    pub(crate) fn check_lab_oracle_policy(
        &self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<()> {
        if self.resident_banks_frozen
            || self.oracle_trace.is_some()
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.secondary_frequency.is_some()
            || self.q3_expert_oracle
            || self.q3_router_reference.is_some()
            || self.bank_slots != crate::daslab_profile::BANK_SLOTS
            || self.banks.len() != 48
            || self.gpu_expert_limit != 10
            || self.batch_moe.is_some()
            || self.transient_resident.is_none()
        {
            return Err(Error::InvalidFormat("public oracle original bank policy"));
        }
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("public oracle resident admission"))?
            .active()?;
        for bank in self.banks.values() {
            bank.check_client(&self.client)?;
            if bank.slots.capacity() != crate::daslab_profile::BANK_SLOTS
                || bank.slots.experts() != 512
                || bank.profile() != self.packed_profile
            {
                return Err(Error::InvalidFormat("public oracle original bank geometry"));
            }
        }
        self.model.verify_file_unchanged()
    }

    pub(crate) fn check_lab_oracle_preloaded(
        &self,
        boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
        trace: &crate::oracle_trace::Trace,
    ) -> Result<()> {
        self.check_lab_oracle_policy(boundary)?;
        trace.validate_ready_replay()?;
        self.check_lab_oracle_fixture(trace.fixture())?;
        for layer in 0..48 {
            let bank = self
                .banks
                .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
                .ok_or(Error::InvalidFormat("public oracle preloaded bank"))?;
            let union = trace.union(layer)?;
            for &expert in union.iter() {
                if bank
                    .slots
                    .resident_slot(expert)
                    .map_err(|_| Error::InvalidFormat("public oracle preloaded slot"))?
                    .is_none()
                {
                    return Err(Error::InvalidFormat("public oracle exact preload missing"));
                }
            }
        }
        Ok(())
    }

    /// Exact PUBLIC original weights only. Every layer retains one protected
    /// complete-union plan while bounded original copy windows finish. There is
    /// no ledger publication before the last audited owning completion.
    pub(crate) fn preload_lab_oracle(
        &mut self,
        boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
        trace: &crate::oracle_trace::Trace,
        arena: &crate::private_arena::PrivateArena,
        fault: crate::private_arena::BufferId,
        cancel: &AtomicBool,
    ) -> Result<expert_banks::FillStats> {
        self.check_lab_oracle_policy(boundary)?;
        trace.validate_ready_replay()?;
        self.check_lab_oracle_fixture(trace.fixture())?;
        self.check_client(arena.client().map_err(crate::resident_util::fault)?)?;
        let admission = self
            .resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("public oracle preload admission"))?
            .clone();
        let mut total = expert_banks::FillStats::default();
        for layer in 0..48 {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("GPU execution cancelled"));
            }
            let selected = trace.union(layer)?;
            let bank = self
                .banks
                .get_mut(&format!("blk.{layer}.ffn_gate_exps.weight"))
                .ok_or(Error::InvalidFormat("public oracle preload original bank"))?;
            let plan = bank
                .slots
                .plan(&selected)
                .map_err(|_| Error::InvalidFormat("public oracle complete union plan"))?;
            bank.slots
                .validate_pending(&plan, &selected)
                .map_err(|_| Error::InvalidFormat("public oracle union pending proof"))?;
            let private = trace
                .owned_bytes()
                .checked_add(
                    selected
                        .capacity()
                        .checked_mul(std::mem::size_of::<usize>())
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .and_then(|n| n.checked_add(plan.owned_metadata_bytes().ok()?))
                .and_then(|n| n.checked_add(4096))
                .ok_or(Error::ArithmeticOverflow)?;
            if private > 192 * 1024 {
                return Err(Error::LimitExceeded("public oracle preload private host"));
            }
            bank.check_client(&self.client)?;
            let [gate, up, down] = bank.source_names().map(|name| {
                self.model
                    .tensor(name)
                    .ok_or(Error::InvalidFormat("public oracle original tensor"))
            });
            let weights = RoutedWeights {
                gate: gate?,
                up: up?,
                down: down?,
            };
            bank.check_names(weights)?;
            for window in plan.uploads.chunks(10) {
                if cancel.load(Ordering::Acquire) {
                    return Err(Error::InvalidFormat("GPU execution cancelled"));
                }
                bank.slots
                    .validate_pending(&plan, &selected)
                    .map_err(|_| Error::InvalidFormat("public oracle window protected proof"))?;
                let requests = fill_requests(
                    [bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64),
                    window.len(),
                    false,
                    self.expert_batch_bytes,
                )?;
                let charged = requests.iter().try_fold(0u64, |sum, &bytes| {
                    sum.checked_add(main_admission::page_bytes(bytes).map_err(admission_error)?)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
                if charged > 40 * 1024 * 1024 {
                    return Err(Error::LimitExceeded("public oracle original source window"));
                }
                let add = admission.allocate(&requests, || {
                    let mut add = expert_banks::FillStats::default();
                    for &(expert, slot) in window {
                        for (packed, tensor) in [
                            (bank.gate(), weights.gate),
                            (bank.up(), weights.up),
                            (bank.down(), weights.down),
                        ] {
                            let (bytes, read, upload) = expert_banks::fill(
                                &self.client,
                                packed,
                                tensor,
                                expert,
                                slot,
                                cancel,
                            )?;
                            add.bytes = add
                                .bytes
                                .checked_add(bytes)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.read_ns = add
                                .read_ns
                                .checked_add(read)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.upload_ns = add
                                .upload_ns
                                .checked_add(upload)
                                .ok_or(Error::ArithmeticOverflow)?;
                        }
                        add.experts += 1;
                    }
                    Ok(add)
                })?;
                crate::resident_model::batch_io::read_fault(arena, &admission, fault, cancel)?;
                if cancel.load(Ordering::Acquire) {
                    return Err(Error::InvalidFormat("GPU execution cancelled"));
                }
                total.bytes = total
                    .bytes
                    .checked_add(add.bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                total.read_ns = total
                    .read_ns
                    .checked_add(add.read_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                total.upload_ns = total
                    .upload_ns
                    .checked_add(add.upload_ns)
                    .ok_or(Error::ArithmeticOverflow)?;
                total.experts = total
                    .experts
                    .checked_add(add.experts)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            bank.slots
                .validate_pending(&plan, &selected)
                .map_err(|_| Error::InvalidFormat("public oracle completed union proof"))?;
            bank.slots
                .commit(plan)
                .map_err(|_| Error::InvalidFormat("public oracle completed union publication"))?;
        }
        self.model.verify_file_unchanged()?;
        self.expert_uploads = self
            .expert_uploads
            .checked_add(total.experts)
            .ok_or(Error::ArithmeticOverflow)?;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add(total.bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_read_ns = self
            .weight_read_ns
            .checked_add(total.read_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_upload_ns = self
            .weight_upload_ns
            .checked_add(total.upload_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(total)
    }
    /// Aggregate only; never expose route-derived expert identities.
    pub(crate) fn q3_occupied_slots(&self) -> Result<usize> {
        if !self.packed_profile.is_low_bit()
            || self.bank_slots != crate::daslab_profile::BANK_SLOTS
            || self.banks.len() != 48
        {
            return Err(Error::InvalidFormat("Q3 original slot receipt geometry"));
        }
        let mut occupied = 0;
        for bank in self.banks.values() {
            if bank.profile() != self.packed_profile
                || bank.slots.capacity() != crate::daslab_profile::BANK_SLOTS
                || bank.slots.experts() != 512
            {
                return Err(Error::InvalidFormat("Q3 original slot receipt owner"));
            }
            for slot in 0..bank.slots.capacity() {
                occupied += usize::from(
                    bank.slots
                        .resident_expert(slot)
                        .map_err(|_| Error::InvalidFormat("Q3 original slot receipt"))?
                        .is_some(),
                );
            }
        }
        Ok(occupied)
    }
    pub(crate) fn enable_adaptive_after_prefill(&mut self) -> Result<()> {
        if std::env::var("KUNO_LAB_ADAPTIVE_EXPERTS").as_deref() != Ok("1") {
            return Ok(());
        }
        if self.adaptive.as_ref().is_none_or(|c| c.ready)
            || self.secondary.is_some()
            || self.bank_slots != crate::daslab_profile::BANK_SLOTS
            || self.banks.len() != 48
            || self.gpu_expert_limit != 10
            || self.batch_moe.is_some()
        {
            return Err(Error::InvalidFormat("adaptive completed prefill policy"));
        }
        self.adaptive
            .as_mut()
            .ok_or(Error::InvalidFormat("adaptive prefill owner"))?
            .ready = true;
        Ok(())
    }
    pub(crate) fn promote_adaptive(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedAdaptiveBoundary,
        arena: &crate::private_arena::PrivateArena,
        fault: crate::private_arena::BufferId,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let Some(cache) = &mut self.adaptive else {
            return Ok(());
        };
        if !cache.ready {
            return Ok(());
        }
        let direct_upload = std::env::var("KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD").as_deref() == Ok("1");
        let wall_start = Instant::now();
        let policy_start = Instant::now();
        let admission = self
            .resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("adaptive resident admission"))?;
        let mut pending = Vec::with_capacity(crate::adaptive_experts::BUDGET);
        let mut requests = Vec::with_capacity(crate::adaptive_experts::BUDGET * 3);
        let start = cache.first_layer();
        let mut next = (start + 1) % 48;
        let mut private_bytes = cache.owned_bytes() + 4096;
        for offset in 0..48 {
            let layer = (start + offset) % 48;
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            let bank = self
                .banks
                .get_mut(&key)
                .ok_or(Error::InvalidFormat("adaptive original bank"))?;
            let Some((expert, victim)) = cache.candidate(layer, &bank.slots)? else {
                continue;
            };
            let selected = zeroize::Zeroizing::new([expert]);
            let plan = bank
                .slots
                .plan_in_slots(&*selected, &[victim])
                .map_err(|_| Error::InvalidFormat("adaptive scored victim plan"))?;
            bank.slots
                .validate_pending(&plan, &*selected)
                .map_err(|_| Error::InvalidFormat("adaptive original plan proof"))?;
            if plan.uploads.len() != 1 {
                return Err(Error::InvalidFormat("adaptive single original upload"));
            }
            private_bytes = private_bytes
                .checked_add(
                    plan.owned_metadata_bytes()
                        .map_err(|_| Error::ArithmeticOverflow)?,
                )
                .ok_or(Error::ArithmeticOverflow)?;
            if private_bytes > crate::adaptive_experts::HOST {
                return Err(Error::LimitExceeded("adaptive live private host"));
            }
            requests.extend([bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64));
            pending.push((layer, plan));
            next = (layer + 1) % 48;
            if pending.len() == crate::adaptive_experts::BUDGET {
                break;
            }
        }
        cache.stats.policy_ns += policy_start.elapsed().as_nanos();
        if !pending.is_empty() {
            let charged = requests.iter().try_fold(0u64, |n, &b| {
                n.checked_add(main_admission::page_bytes(b).map_err(admission_error)?)
                    .ok_or(Error::ArithmeticOverflow)
            })?;
            if charged > 40 * 1024 * 1024 {
                return Err(Error::LimitExceeded("adaptive original source"));
            }
            cache.stats.private_peak = cache.stats.private_peak.max(private_bytes);
            cache.stats.source_peak = cache.stats.source_peak.max(charged);
            let fill_start = Instant::now();
            let add = admission.allocate(&requests, || {
                let mut add = expert_banks::FillStats::default();
                let mut writes =
                    Vec::with_capacity(if direct_upload { pending.len() * 3 } else { 0 });
                for (layer, plan) in &pending {
                    let bank = self
                        .banks
                        .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
                        .ok_or(Error::InvalidFormat("adaptive original fill bank"))?;
                    bank.check_client(&self.client)?;
                    let [(expert, slot)] = plan.uploads.as_slice() else {
                        return Err(Error::InvalidFormat("adaptive original fill proof"));
                    };
                    bank.slots
                        .validate_pending(plan, &[*expert])
                        .map_err(|_| Error::InvalidFormat("adaptive fill ledger proof"))?;
                    for (packed, name) in [bank.gate(), bank.up(), bank.down()]
                        .into_iter()
                        .zip(bank.source_names())
                    {
                        let tensor = self
                            .model
                            .tensor(name)
                            .ok_or(Error::InvalidFormat("adaptive original tensor"))?;
                        if direct_upload {
                            let (view, data, read) = expert_banks::prepare_public_existing(
                                packed, tensor, *expert, *slot, cancel,
                            )?;
                            add.bytes += data.len() as u64;
                            add.read_ns += read;
                            writes.push((view, data));
                        } else {
                            let (bytes, read, upload) = expert_banks::fill(
                                &self.client,
                                packed,
                                tensor,
                                *expert,
                                *slot,
                                cancel,
                            )?;
                            add.bytes += bytes;
                            add.read_ns += read;
                            add.upload_ns += upload;
                        }
                    }
                }
                if direct_upload {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(Error::InvalidFormat("GPU execution cancelled"));
                    }
                    let upload = Instant::now();
                    self.client
                        .write_public_owned_batch(writes)
                        .map_err(|_| Error::InvalidFormat("direct original expert upload"))?;
                    add.upload_ns += upload.elapsed().as_nanos();
                }
                Ok(add)
            })?;
            cache.stats.fill_ns += fill_start.elapsed().as_nanos();
            // ONE owning completion follows all admitted fills. No ledger is
            // committed while any original bank copy can still be in flight.
            let fence = Instant::now();
            crate::resident_model::batch_io::read_fault(arena, admission, fault, cancel)?;
            cache.stats.fence_ns += fence.elapsed().as_nanos();
            cache.stats.completed_batches += 1;
            if direct_upload {
                cache.stats.direct_batches += 1;
                cache.stats.direct_components += (pending.len() * 3) as u64;
                cache.stats.direct_bytes += add.bytes;
            }
            cache.stats.bytes += add.bytes;
            cache.stats.read_ns += add.read_ns;
            cache.stats.upload_ns += add.upload_ns;
            let publication_start = Instant::now();
            for (layer, plan) in pending {
                self.banks
                    .get_mut(&format!("blk.{layer}.ffn_gate_exps.weight"))
                    .ok_or(Error::InvalidFormat("adaptive completed original bank"))?
                    .slots
                    .commit(plan)
                    .map_err(|_| Error::InvalidFormat("adaptive completed publication"))?;
                cache.stats.promotions += 1;
            }
            cache.stats.publication_ns += publication_start.elapsed().as_nanos();
        }
        cache.stats.wall_ns += wall_start.elapsed().as_nanos();
        cache.advance(next)
    }
    pub(crate) fn enable_secondary_after_prefill(&mut self) -> Result<()> {
        let Some(frequency) = self.secondary_frequency.as_ref() else {
            return Ok(());
        };
        if self.secondary.is_some()
            || self.bank_slots != crate::daslab_profile::BANK_SLOTS
            || self.banks.len() != 48
            || !self.shared_kernels
            || self.gpu_expert_limit != 10
            || self.batch_moe.is_some()
        {
            return Err(Error::InvalidFormat("secondary completed prefill policy"));
        }
        let model = kuno_text::model::Qwen4Exp::bind(self.model)?;
        let client = cubecl::future::block_on(crate::secondary_experts::new_client())?;
        self.secondary = Some(crate::secondary_experts::Secondary::build(
            client,
            self.model,
            model.bound_graph(),
            |layer| {
                self.banks
                    .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
                    .ok_or(Error::InvalidFormat("secondary original primary bank"))
            },
            frequency,
            &AtomicBool::new(false),
        )?);
        self.secondary_frequency = None;
        Ok(())
    }
    /// Completed fixed-public cap10 boundary only: CPU misses never consume
    /// static GPU-resident spans. Fully adaptive banks can return to CPU and
    /// are excluded; only immutable dense sources retain advice in that mode. This
    /// hints the OS to reclaim duplicate original file-cache pages, never GPU
    /// owners, tensors, private state or arbitrary host caches. No freed-byte
    /// claim: the returned total counts successfully advised source bytes.
    pub(crate) fn advise_resident_gpu_sources_not_needed(
        &self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
        pageout: bool,
    ) -> Result<u64> {
        if !self.model.weights_mapped() || self.resident_admission.is_none() {
            return Err(Error::InvalidFormat(
                "GPU source cache hint requires resident mapped owner",
            ));
        }
        self.model.verify_file_unchanged()?;
        let mut bytes = 0u64;
        for (key, cached) in &self.cache {
            let tensor = self
                .model
                .tensor(&key.tensor)
                .ok_or(Error::InvalidFormat("GPU cached source tensor absent"))?;
            let rows = *tensor
                .info()
                .dimensions()
                .get(1)
                .ok_or(Error::InvalidBufferLength)?;
            let offset = zeroize::Zeroizing::new(
                (key.expert as u64)
                    .checked_mul(rows)
                    .and_then(|n| n.checked_add(key.first as u64))
                    .and_then(|n| n.checked_mul(tensor.info().row_bytes()))
                    .ok_or(Error::ArithmeticOverflow)?,
            );
            // Cached words include at most3 bytes of binding alignment, never
            // another tensor/expert. Advise only original complete rows.
            let row_bytes = tensor.info().row_bytes();
            if row_bytes < 4 {
                return Err(Error::InvalidBufferLength);
            }
            let length = (cached.words as u64 * 4) / row_bytes * row_bytes;
            bytes = bytes
                .checked_add(if pageout {
                    self.model
                        .advise_gpu_source_pageout(tensor, *offset, length)?
                } else {
                    self.model
                        .advise_gpu_source_not_needed(tensor, *offset, length)?
                })
                .ok_or(Error::ArithmeticOverflow)?;
        }
        for bank in self.banks.values().filter(|_| self.gpu_expert_limit == 10) {
            // Every expert bank slot can return to CPU in adaptive mode.
            if self.adaptive.is_some() {
                continue;
            }
            for slot in 0..bank.slots.capacity() {
                let expert = zeroize::Zeroizing::new(
                    bank.slots
                        .resident_expert(slot)
                        .map_err(|_| Error::InvalidFormat("GPU source committed residency"))?,
                );
                if let Some(expert) = *expert {
                    for (name, packed) in
                        bank.source_names()
                            .into_iter()
                            .zip([bank.gate(), bank.up(), bank.down()])
                    {
                        let tensor = self
                            .model
                            .tensor(name)
                            .ok_or(Error::InvalidFormat("GPU bank source tensor absent"))?;
                        let length = packed.expert_bytes() as u64;
                        let offset = zeroize::Zeroizing::new(
                            (expert as u64)
                                .checked_mul(length)
                                .ok_or(Error::ArithmeticOverflow)?,
                        );
                        bytes = bytes
                            .checked_add(if pageout {
                                self.model
                                    .advise_gpu_source_pageout(tensor, *offset, length)?
                            } else {
                                self.model
                                    .advise_gpu_source_not_needed(tensor, *offset, length)?
                            })
                            .ok_or(Error::ArithmeticOverflow)?;
                    }
                }
            }
        }
        self.model.verify_file_unchanged()?;
        Ok(bytes)
    }
    pub fn reserved_bytes(&self) -> Result<u64> {
        self.client
            .memory_usage()
            .map(|m| m.bytes_reserved)
            .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))
    }
    fn check_segmented_resident(&self, resident: bool) -> Result<()> {
        if self.strata_global_sources.is_some() {
            return Err(Error::InvalidFormat(
                "legacy banks conflict with global sources",
            ));
        }
        // The constructor pins explicit budget configuration; no environment
        // re-read is needed after creation. Public mutable experiment flags
        // still need revalidation before every allocation/dispatch seam.
        expert_banks::validate_segmented_configuration_with_profile(
            self.bank_slots,
            Some(self.budget),
            self.budget,
            self.block_moe,
            self.batch_experts,
            self.word_down,
            resident,
            self.packed_profile,
        )?;
        self.packed_profile.validate_shortcuts(
            false,
            self.fused_experts,
            self.integer_experts,
            self.word_down,
            self.block_moe,
        )?;
        if self.packed_profile.is_low_bit()
            && (self.secondary.is_some() || self.secondary_frequency.is_some())
        {
            return Err(Error::InvalidFormat(
                "secondary experts require original Q4 profile",
            ));
        }
        expert_banks::check_ordered_segmented(self.bank_slots, self.ordered_public_upload)
    }
    pub(crate) fn resident_bank_capacity(&self) -> usize {
        self.bank_slots
    }
    pub(crate) fn resident_runtime_io_limit(&self) -> u64 {
        if self.target_rows_reserved {
            crate::resident_model::batch::TARGET_ROWS_RUNTIME_LIMIT
        } else if self.resident_router_reserved {
            batch_moe::RESIDENT_ROUTER_RUNTIME_LIMIT
        } else {
            RESIDENT_RUNTIME_IO_MAX
        }
    }
    /// One immutable activation after the original public prefill or consumed
    /// CPU13 reset has authenticated its actual frozen source attachment.
    /// No scalar source, private owner, capacity or runtime cadence is changed.
    pub(crate) fn activate_public_target_iq4_decoder(&mut self) -> Result<()> {
        self.check_target_rows_policy()?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat(
                "IQ4 decoder actual model admission absent",
            ))?
            .active()?;
        self.target_iq4_decoder = activate_target_iq4_decoder(
            self.target_iq4_decoder,
            self.target_rows_reserved
                && self.target_verify3_dense
                && self.target_verify3_fused
                && self.token_io_checkpoint
                && self.resident_banks_frozen
                && self.target_frozen_host_source_identity.is_some()
                && self.packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3
                && self.bank_slots == crate::daslab_profile::BANK_SLOTS
                && self
                    .client
                    .public_target_request_io_configured()
                    .map_err(|_| {
                        Error::InvalidFormat("IQ4 decoder original target runtime selection")
                    })?,
        )?;
        Ok(())
    }
    /// Count actual successful candidate launches only. Callers publish this
    /// scalar receipt only after the original whole-model acceptance boundary.
    pub(crate) fn public_target_iq4_decoder(
        &self,
    ) -> Result<(crate::next_shared::Iq4Decoder, u64)> {
        self.check_target_rows_policy()?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat(
                "IQ4 decoder actual model admission absent",
            ))?
            .active()?;
        Ok((self.target_iq4_decoder, self.target_iq4_decoder_calls))
    }
    /// Source authority of the connected core, never a fabricated frozen-bank
    /// identity. Construction before source attachment is not an active scope.
    pub(crate) fn global_target_scope(&self) -> Result<bool> {
        let Some(sources) = &self.strata_global_sources else {
            return Ok(false);
        };
        if !crate::strata_intel_core::enabled()
            || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || !self.banks.is_empty()
            || self.bank_reserved != 0
            || self.resident_banks_frozen
            || self.target_frozen_host_source_identity.is_some()
            || self.batch_moe.is_some()
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.cache.keys().any(|k| k.tensor.contains("_exps.weight"))
        {
            return Err(Error::InvalidFormat("global target execution authority"));
        }
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("global target admission absent"))?
            .active()?;
        sources.validate_attached()?;
        Ok(true)
    }
    fn q6_verify_head_scope(&self) -> bool {
        self.target_rows_reserved
            && self.target_verify3_dense
            && self.target_verify3_fused
            && self.token_io_checkpoint
            && ((self.resident_banks_frozen
                && self.target_frozen_host_source_identity.is_some()
                && self.target_frozen_gu_integer_reserved
                && self.target_frozen_down_integer_reserved)
                || self.global_target_scope().unwrap_or(false))
            && self.packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3
            && self.bank_slots == crate::daslab_profile::BANK_SLOTS
            && self.native_projection_member_capacity() == 8
            && self.native_projection_tile() == 8
            && !self.q6_integer_head
            && self.target_iq4_decoder == crate::next_shared::Iq4Decoder::Original
    }
    /// Called only by the completed model boundary wrapper, after original
    /// prefill and actual source attachment; selection never allocates owners.
    pub(crate) fn activate_public_q6_integer_verify_head(&mut self) -> Result<()> {
        self.check_target_rows_policy()?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .active()?;
        validate_q6_verify_counters(
            self.q6_integer_verify_head,
            self.q6_integer_verify_head_prepares,
            self.q6_integer_verify_head_tiles,
            self.q6_integer_verify_head_members,
        )?;
        self.q6_integer_verify_head = activate_q6_verify_head(
            self.q6_integer_verify_head,
            self.q6_verify_head_scope()
                && self
                    .client
                    .public_target_request_io_configured()
                    .map_err(|_| Error::InvalidFormat("Q6 verifier target runtime selection"))?,
        )?;
        Ok(())
    }
    /// Native core arithmetic is fixed before the first prompt input. Source
    /// attachment and the model's real completed boundary precede this call.
    pub(crate) fn activate_native_core_q6_head(&mut self) -> Result<()> {
        self.check_target_rows_policy()?;
        if !self.global_target_scope()?
            || !self.q6_verify_head_scope()
            || !self
                .client
                .public_target_request_io_configured()
                .map_err(|_| Error::InvalidFormat("native core head runtime"))?
        {
            return Err(Error::InvalidFormat("native core head source authority"));
        }
        validate_q6_verify_counters(
            self.q6_integer_verify_head,
            self.q6_integer_verify_head_prepares,
            self.q6_integer_verify_head_tiles,
            self.q6_integer_verify_head_members,
        )?;
        self.q6_integer_verify_head = activate_q6_verify_head(self.q6_integer_verify_head, true)?;
        Ok(())
    }
    pub(crate) fn public_q6_integer_verify_head(&self) -> Result<(bool, u64, u64, u64)> {
        self.check_target_rows_policy()?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .active()?;
        validate_q6_verify_counters(
            self.q6_integer_verify_head,
            self.q6_integer_verify_head_prepares,
            self.q6_integer_verify_head_tiles,
            self.q6_integer_verify_head_members,
        )?;
        Ok((
            self.q6_integer_verify_head,
            self.q6_integer_verify_head_prepares,
            self.q6_integer_verify_head_tiles,
            self.q6_integer_verify_head_members,
        ))
    }
    /// Construction-only capability; this never activates a four-row chunk.
    #[allow(dead_code)] // Construction-only until the isolated four-row parent is integrated.
    pub(crate) fn reserve_public_four_rows(&mut self) -> Result<()> {
        if self.target_four_rows_reserved
            || self.target_four_rows_active
            || self.resident_admission.is_some()
            || !self.target_rows_reserved
            || !self.target_verify3_dense
            || !self.target_verify3_fused
            || !self.token_io_checkpoint
            || !self.target_frozen_host_router_reserved
            || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || self.bank_slots != crate::daslab_profile::BANK_SLOTS
            || self.native_projection_member_capacity() != 8
            || self.native_projection_tile() != 8
            || self.target_frozen_host_staging_reserved
            || self.target_iq4_decoder != crate::next_shared::Iq4Decoder::Original
        {
            return Err(Error::InvalidFormat("public four-row construction policy"));
        }
        self.target_four_rows_reserved = true;
        Ok(())
    }
    pub(crate) fn check_target_rows_policy(&self) -> Result<()> {
        let global = self.global_target_scope()?;
        if self.q6_integer_verify_head && !self.q6_verify_head_scope()
            || !self.q6_integer_verify_head
                && (self.q6_integer_verify_head_prepares != 0
                    || self.q6_integer_verify_head_tiles != 0
                    || self.q6_integer_verify_head_members != 0)
        {
            return Err(Error::InvalidFormat("Q6 verifier head actual source scope"));
        }
        if self
            .resident_admission
            .as_ref()
            .is_some_and(|a| a.target_four_rows_reserved != self.target_four_rows_reserved)
            || self.target_four_rows_active
                && (!self.target_four_rows_reserved
                    || self.resident_admission.is_none()
                    || !(global
                        || self.resident_banks_frozen
                            && self.target_frozen_host_source_identity.is_some()))
            || !self.target_four_rows_active
                && (self.target_verify4_dense_calls != 0 || self.target_verify4_fused_calls != 0)
        {
            return Err(Error::InvalidFormat("public four-row immutable admission"));
        }
        if self.target_iq4_decoder == crate::next_shared::Iq4Decoder::PackedRegisters
            && (!self.target_rows_reserved
                || !self.target_verify3_dense
                || !self.target_verify3_fused
                || !self.token_io_checkpoint
                || !self.resident_banks_frozen
                || self.target_frozen_host_source_identity.is_none()
                || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
                || self.bank_slots != crate::daslab_profile::BANK_SLOTS)
            || self.target_iq4_decoder == crate::next_shared::Iq4Decoder::Original
                && self.target_iq4_decoder_calls != 0
        {
            return Err(Error::InvalidFormat(
                "IQ4 decoder actual original target scope",
            ));
        }
        if self.target_frozen_host_staging_reserved
            && (!self.target_rows_reserved
                || !self.target_batch_router_reserved
                || !self.target_frozen_host_router_reserved)
        {
            return Err(Error::InvalidFormat("packed staging construction scope"));
        }
        if self.target_frozen_gu_integer_reserved
            && (!self.target_rows_reserved
                || !self.target_batch_router_reserved
                || !self.target_frozen_host_router_reserved
                || self.target_frozen_host_staging_reserved)
        {
            return Err(Error::InvalidFormat("frozen GU integer construction scope"));
        }
        if self.target_frozen_down_integer_reserved
            && (!self.target_rows_reserved
                || !self.target_batch_router_reserved
                || !self.target_frozen_host_router_reserved
                || self.target_frozen_host_staging_reserved)
        {
            return Err(Error::InvalidFormat(
                "frozen DOWN integer construction scope",
            ));
        }
        validate_target_frozen_host_policy(
            self.resident_banks_frozen,
            self.target_frozen_host_source_identity.is_some(),
            self.target_rows_reserved,
            self.packed_profile,
            self.bank_slots,
            self.target_batch_router_reserved,
            self.target_frozen_host_router_reserved,
        )?;
        validate_target_verify3_policy(
            self.target_verify3_dense,
            self.resident_admission
                .as_ref()
                .map(|a| a.target_verify3_dense),
            self.target_rows_reserved,
            self.fast_dense && self.fast_dense_fault && self.shared_kernels,
            self.native_projection_member_capacity(),
            self.native_projection_tile(),
        )?;
        validate_target_verify3_fused_policy(
            self.target_verify3_fused,
            self.resident_admission
                .as_ref()
                .map(|a| a.target_verify3_fused),
            self.target_verify3_dense,
            self.target_rows_reserved,
            self.native_projection_member_capacity(),
            self.native_projection_tile(),
        )?;
        if self.target_rows_reserved
            && (self.resident_router_reserved
                || !self.packed_profile.is_low_bit()
                || self.bank_slots != crate::daslab_profile::BANK_SLOTS
                || self.transient_resident.is_none()
                || !self.model.weights_mapped()
                || self.adaptive.is_some()
                || self.secondary.is_some()
                || self.secondary_frequency.is_some()
                || self.ordered_public_upload
                || self.batch_experts
                || self.word_down
                || self.fused_experts
                || self.integer_experts
                || self.q3_empty_slot_fill
                || self.q3_skip_empty_segments
                || self.q3_subgroup_moe
                || self.q3_integer_moe
                || self.q3_shared_original_moe
                || self.q6_integer_head
                || (self.resident_banks_frozen
                    && self.target_frozen_host_source_identity.is_none())
                || self.resident_router_audits != 0
                || self.oracle_trace.is_some()
                || self.q3_expert_oracle
                || self.q3_router_reference.is_some()
                || self.gpu_router
                || self.native_projection_member_capacity() != 8
                || self.native_projection_tile() != 8)
        {
            return Err(Error::InvalidFormat(
                "native target rows explicit construction policy",
            ));
        }
        Ok(())
    }
    /// Before ANY owner allocation, admit all static full pages and every
    /// simultaneously queued source page. Capacity plan is distinct from the
    /// actual baseline ledger; this laboratory executor binds only one owner.
    pub(crate) fn begin_resident(
        &mut self,
        graph: kuno_text::model::BoundGraph<'_, '_>,
        private_owned: u64,
        public_pages: u64,
    ) -> Result<ResidentAdmission> {
        self.check_segmented_resident(true)?;
        self.check_target_rows_policy()?;
        let target_private = if self.target_batch_router_reserved {
            if !self.target_rows_reserved
                || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
                || self.bank_slots != crate::daslab_profile::BANK_SLOTS
                || self.resident_banks_frozen
            {
                return Err(Error::InvalidFormat(
                    "public batch router construction policy",
                ));
            }
            target_rows_private_extent_with_down_math(
                private_owned,
                true,
                self.target_frozen_host_router_reserved,
                self.target_frozen_host_staging_reserved,
                self.target_frozen_gu_integer_reserved,
                self.target_frozen_down_integer_reserved,
            )
        } else {
            target_rows_private_extent_with_down_math(
                private_owned,
                false,
                false,
                self.target_frozen_host_staging_reserved,
                self.target_frozen_gu_integer_reserved,
                self.target_frozen_down_integer_reserved,
            )
        };
        if self.target_rows_reserved && !target_private {
            return Err(Error::InvalidFormat(
                "native target rows explicit construction policy",
            ));
        }
        if self.resident_router_reserved
            && (!self.packed_profile.is_low_bit()
                || self.bank_slots != crate::daslab_profile::BANK_SLOTS
                || self.transient_resident.is_none()
                || !self.model.weights_mapped()
                || self.adaptive.is_some()
                || self.secondary.is_some()
                || self.ordered_public_upload)
        {
            return Err(Error::InvalidFormat(
                "resident router explicit construction policy",
            ));
        }
        if self.resident_admission.is_some()
            || !self.cache.is_empty()
            || !self.banks.is_empty()
            || self.target_iq4_decoder != crate::next_shared::Iq4Decoder::Original
            || self.target_iq4_decoder_calls != 0
            || self.q6_integer_verify_head
            || self.q6_integer_verify_head_prepares != 0
            || self.q6_integer_verify_head_tiles != 0
            || self.q6_integer_verify_head_members != 0
        {
            return Err(Error::InvalidFormat("resident executor already used"));
        }
        if self.token_io_checkpoint && self.transient_resident.is_none() {
            return Err(Error::InvalidFormat(
                "token checkpoint requires owned runtime I/O",
            ));
        }
        if self
            .transient_resident
            .as_ref()
            .is_some_and(|retired| retired.load(Ordering::Acquire))
        {
            return Err(Error::InvalidFormat("resident GPU context retired"));
        }
        if self.ordered_public_upload {
            if self.transient_resident.is_none() || self.batch_experts {
                return Err(Error::InvalidFormat(
                    "ordered public upload policy mismatch",
                ));
            }
            crate::ordered_public::validate_geometry(graph.geometry())?;
            if crate::ordered_public::WINDOW_RUNTIME > RESIDENT_RUNTIME_IO_MAX {
                return Err(Error::LimitExceeded("ordered public runtime window"));
            }
        }
        let banks = expert_banks::preflight_residency_with_profile(
            graph,
            self.bank_slots,
            self.budget,
            self.reserved_limit,
            self.packed_profile,
        )? as u64;
        let mut source_pages = 0_u64;
        for layer in 0..graph.geometry().layers {
            let base = format!("blk.{layer}.");
            let mut parts = [0_u64; 3];
            for (i, suffix) in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]
            .into_iter()
            .enumerate()
            {
                let t = graph.tensor(&format!("{base}{suffix}"))?;
                parts[i] = t
                    .info()
                    .row_bytes()
                    .checked_mul(t.info().dimensions()[1])
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            if self.ordered_public_upload {
                crate::ordered_public::validate_parts(parts)?;
            }
            let requests = fill_requests(parts, 10, self.batch_experts, self.expert_batch_bytes)?;
            let pages = requests.into_iter().try_fold(0_u64, |sum, n| {
                sum.checked_add(main_admission::page_bytes(n).map_err(admission_error)?)
                    .ok_or(Error::ArithmeticOverflow)
            })?;
            source_pages = source_pages.max(pages);
        }
        let baseline = self.reserved_bytes()?;
        main_admission::check_capacity(
            self.reserved_limit as u64,
            &[
                self.budget as u64,
                banks,
                private_owned,
                RESIDENT_IO_MAX,
                if self.transient_resident.is_some() {
                    self.resident_runtime_io_limit()
                } else {
                    0
                },
                public_pages,
                source_pages,
                baseline,
            ],
        )
        .map_err(admission_error)?;
        let admission = ResidentAdmission {
            target_verify3_dense: self.target_verify3_dense,
            target_verify3_fused: self.target_verify3_fused,
            target_four_rows_reserved: self.target_four_rows_reserved,
            deferred_router_reads: Arc::new(AtomicU64::new(0)),
            request_io_nonce: Arc::new(AtomicU64::new(0)),
            request_io_checkpoints: Arc::new(AtomicU64::new(0)),
            sequence_poison: Arc::new(AtomicBool::new(false)),
            client: self.client.clone(),
            transient_io: self.transient_resident.is_some(),
            token_io_checkpoint: self.token_io_checkpoint,
            ordered_sources: self.ordered_sources.clone(),
            retired: self
                .transient_resident
                .clone()
                .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            ledger: Arc::new(Mutex::new(
                MainAdmission::new(self.reserved_limit as u64, baseline)
                    .map_err(admission_error)?,
            )),
        };
        self.resident_admission = Some(admission.clone());
        // A fresh logical owner must never inherit a previous decode experiment.
        self.native_projection_tile = NativeProjectionTile::default();
        self.native_projection_members = quant_matvec::matmul::MAX_BATCH;
        Ok(admission)
    }
    pub(crate) fn check_resident_admission(&self, owner: &ResidentAdmission) -> Result<()> {
        match &self.resident_admission {
            Some(admission) if Arc::ptr_eq(&admission.ledger, &owner.ledger) => admission.active(),
            _ => {
                owner.poison();
                self.poison_resident();
                Err(Error::InvalidFormat("resident admission owner mismatch"))
            }
        }
    }
    pub(crate) fn native_projection_tile(&self) -> usize {
        self.native_projection_tile.tokens()
    }
    pub(crate) fn native_projection_member_capacity(&self) -> usize {
        self.native_projection_members
    }
    /// The model alone constructs this completed, full-audit native boundary.
    /// No private count selects policy, and no owner or fault scan is narrowed.
    pub(crate) fn set_lab_native_single_dense_dispatch(
        &mut self,
        _boundary: &crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<usize> {
        if self.native_projection_tile() != 1
            || self.batch_moe.is_some()
            || self.transient_resident.is_none()
        {
            return Err(Error::InvalidFormat("native single projection boundary"));
        }
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("native projection admission"))?
            .active()?;
        self.native_projection_members = 1;
        Ok(self.native_projection_member_capacity())
    }
    /// Only a completed original native model owner can construct this proof.
    /// No queued work, buffers, runtime ledgers or cache entries are changed.
    pub(crate) fn set_native_projection_tile(
        &mut self,
        tile: NativeProjectionTile,
        _boundary: crate::resident_model::batch::CompletedNativeProjectionBoundary,
    ) -> Result<usize> {
        if self.batch_moe.is_some() || self.transient_resident.is_none() {
            return Err(Error::InvalidFormat("native projection policy boundary"));
        }
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("native projection admission"))?
            .active()?;
        self.native_projection_tile = tile;
        Ok(self.native_projection_tile())
    }
    pub(crate) fn admission_upper_bytes(&self) -> Result<Option<u64>> {
        self.resident_admission
            .as_ref()
            .map(|a| a.with_ledger(|ledger| Ok(ledger.upper_bytes())))
            .transpose()
    }
    pub(crate) fn poison_resident(&self) {
        if let Some(admission) = &self.resident_admission {
            admission.poison();
        }
    }
    fn nonresident_only(&self) -> Result<()> {
        if self.resident_admission.is_some() {
            self.poison_resident();
            return Err(Error::InvalidFormat(
                "CPU activation helper on resident executor",
            ));
        }
        Ok(())
    }
    /// Keep committed bank ledgers separate from retained native candidate
    /// plans. Validate actual capacities as well as the public 48-layer bound.
    pub(crate) fn check_original_bank_metadata(&self, layers: usize, cap: usize) -> Result<()> {
        let per_bank =
            crate::bank_slots::BankSlots::bounded_original_metadata_bytes(self.bank_slots, 512)
                .map_err(|_| Error::InvalidFormat("original bank host plan"))?;
        if layers != 48
            || self.banks.len() > layers
            || per_bank
                .checked_mul(layers)
                .ok_or(Error::ArithmeticOverflow)?
                > cap
        {
            return Err(Error::LimitExceeded("original bank host plan"));
        }
        let actual = self.banks.values().try_fold(0usize, |sum, bank| {
            let bytes = bank
                .slots
                .original_metadata_bytes()
                .map_err(|_| Error::ArithmeticOverflow)?;
            if bank.slots.capacity() != self.bank_slots
                || bank.slots.experts() != 512
                || bytes > per_bank
            {
                return Err(Error::LimitExceeded("original bank host capacity"));
            }
            sum.checked_add(bytes).ok_or(Error::ArithmeticOverflow)
        })?;
        if actual > cap {
            return Err(Error::LimitExceeded("original bank host capacity"));
        }
        Ok(())
    }

    /// # Safety
    /// The original shard hashes and sole device identity are authenticated;
    /// the caller reserves the complete separate HOST budget before allocation.
    /// No request execution or alternate reader stream overlaps construction.
    #[allow(dead_code)] // Coordinator starts from this owner in the next connection.
    pub(crate) unsafe fn prepare_strata_global_sources(
        &mut self,
        graph: kuno_text::model::BoundGraph<'_, '_>,
        plan: crate::strata_global_plan::Plan,
        device: &wgpu::Device,
        host_reserved: u64,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || !self.banks.is_empty()
            || self.bank_reserved != 0
            || self.strata_global_sources.is_some()
            || self.batch_moe.is_some()
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.cache.keys().any(|k| k.tensor.contains("_exps.weight"))
        {
            return Err(Error::InvalidFormat("global expert loader selection"));
        }
        graph.verify_unchanged()?;
        for name in graph.tensor_names() {
            if !graph.tensor(name)?.belongs_to(self.model) {
                return Err(Error::InvalidFormat("global expert model binding"));
            }
        }
        let admission = self
            .resident_admission
            .as_ref()
            .ok_or(Error::InvalidFormat("global expert admission required"))?;
        let requests = crate::strata_global_sources::Sources::gpu_requests(&plan);
        let retired = AtomicBool::new(false);
        let result = admission.allocate_banks(&requests, || {
            crate::expert_banks::exact_persistent::persistent_scope(
                |mode| {
                    unsafe {
                        self.client.allocation_mode(mode);
                    }
                    Ok(())
                },
                &retired,
                || unsafe {
                    crate::strata_global_sources::Sources::load(
                        graph,
                        plan,
                        &self.client,
                        device,
                        host_reserved,
                    )
                },
            )
        });
        match result {
            Ok(sources) => {
                self.strata_global_sources = Some(std::sync::Arc::new(sources));
                Ok(())
            }
            Err(error) => {
                self.poison_resident();
                Err(error)
            }
        }
    }
    pub(crate) fn strata_gr_weights(
        &mut self,
        tensors: [TensorView<'_>; 2],
        injection: Option<TensorView<'_>>,
    ) -> Result<[Handle; 3]> {
        if !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidFormat("native GR selection"));
        }
        self.model.verify_file_unchanged()?;
        self.check_target_rows_policy()?;
        for (t, (inputs, outputs)) in tensors.iter().zip([(10240, 320), (320, 10240)]) {
            let m = Matrix::new(*t)?;
            if !t.belongs_to(self.model)
                || t.info().dtype().gguf_id() != 30
                || m.experts() != 1
                || (m.inputs(), m.outputs()) != (inputs, outputs)
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        if let Some(t) = injection {
            let m = Matrix::new(t)?;
            if !t.belongs_to(self.model)
                || t.info().dtype().gguf_id() != 30
                || m.experts() != 1
                || (m.inputs(), m.outputs()) != (10240, 4)
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        // Same complete original BF16 cache rows as the former projections;
        // no duplicate public roots, conversion or per-decode weight copies.
        let down = self.weight(tensors[0], 0, 0, 320)?.0;
        let up = self.weight(tensors[1], 0, 0, 10240)?.0;
        let inject = if let Some(t) = injection {
            self.weight(t, 0, 0, 4)?.0
        } else {
            down.clone()
        };
        Ok([down, up, inject])
    }
    fn weight(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        first: usize,
        rows: usize,
    ) -> Result<(Handle, usize)> {
        let result = self.weight_inner(tensor, expert, first, rows);
        if result.is_err() {
            self.poison_resident();
        }
        result
    }
    fn weight_inner(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        first: usize,
        rows: usize,
    ) -> Result<(Handle, usize)> {
        if self.strata_global_sources.is_some() && tensor.info().name().contains("_exps.weight") {
            return Err(Error::InvalidFormat(
                "legacy expert copy conflicts with global source",
            ));
        }
        if let Some(admission) = &self.resident_admission {
            admission.active()?;
        }
        self.tick = self.tick.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        let key = Key {
            tensor: tensor.info().name().into(),
            expert,
            first,
        };
        if let Some(cached) = self.cache.get_mut(&key) {
            cached.last = self.tick;
            self.cache_hits += 1;
            return Ok((cached.handle.clone(), cached.words));
        }
        let row_bytes = tensor.info().row_bytes() as usize;
        let bytes = rows
            .checked_mul(row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let padded = bytes.checked_add(3).ok_or(Error::ArithmeticOverflow)? / 4 * 4;
        let exact = crate::strata_port::exact_static_cache()
            && self.packed_profile == expert_banks::OriginalPackedProfile::DaslabIq3;
        let alignment = self.client.properties().memory.alignment as usize;
        if exact && (alignment == 0 || !alignment.is_power_of_two() || alignment > 4096) {
            return Err(Error::InvalidFormat("exact public static alignment"));
        }
        let reserved = if exact {
            padded
                .checked_add(alignment - 1)
                .ok_or(Error::ArithmeticOverflow)?
                / alignment
                * alignment
        } else {
            padded.max(32 * 1024).next_power_of_two()
        };
        while self.cache_bytes + reserved > self.budget {
            let old = self
                .cache
                .iter()
                .min_by_key(|(_, c)| c.last)
                .map(|(k, _)| k.clone())
                .ok_or(Error::LimitExceeded("GPU cache capacity"))?;
            let removed = self
                .cache
                .remove(&old)
                .ok_or(Error::InvalidFormat("GPU cache invariant"))?;
            self.cache_bytes -= removed.bytes;
        }
        self.client.memory_cleanup();
        if self.resident_admission.is_none()
            && self.reserved_bytes()?
                > (self.budget + self.bank_reserved + 512 * 1024 * 1024).min(self.reserved_limit)
                    as u64
        {
            self.cache.clear();
            self.cache_bytes = 0;
            self.client.memory_cleanup();
            if self.reserved_bytes()?
                > (self.budget + self.bank_reserved + 512 * 1024 * 1024).min(self.reserved_limit)
                    as u64
            {
                return Err(Error::LimitExceeded("GPU allocator reserved memory"));
            }
        }
        let mut packed = vec![0u8; padded];
        let output_rows = tensor.info().dimensions()[1] as usize;
        let offset = expert
            .checked_mul(output_rows)
            .and_then(|n| n.checked_add(first))
            .and_then(|n| n.checked_mul(row_bytes))
            .ok_or(Error::ArithmeticOverflow)?;
        let started = Instant::now();
        tensor.read_range(offset as u64, &mut packed[..bytes])?;
        self.weight_read_ns += started.elapsed().as_nanos();
        // GPU storage uses original little-endian byte packing; only final word
        // padding is added. No host f32 mirror/requantized model copy exists.
        let started = Instant::now();
        let upload = || {
            if exact {
                use cubecl_runtime::memory_management::MemoryAllocationMode;
                // Same exclusive stream admission and unwind-safe allocation
                // guard as exact expert banks. Only ORIGINAL PUBLIC weights.
                let retired = AtomicBool::new(false);
                expert_banks::exact_persistent::persistent_scope(
                    |mode: MemoryAllocationMode| {
                        // SAFETY: laboratory owns this stream's allocation scope.
                        unsafe {
                            self.client.allocation_mode(mode);
                        }
                        Ok(())
                    },
                    &retired,
                    || Ok(self.client.create_public_owned(packed)),
                )
            } else {
                Ok(self.client.create_public_owned(packed))
            }
        };
        let handle = if let Some(admission) = &self.resident_admission {
            admission.allocate(
                &[if exact {
                    reserved as u64
                } else {
                    padded as u64
                }],
                upload,
            )?
        } else {
            upload()?
        };
        if exact {
            let resource = self
                .client
                .get_resource(handle.clone())
                .map_err(|_| Error::InvalidFormat("exact static root"))?;
            let root = resource.resource();
            if root.offset != 0
                || root.size != padded as u64
                || root.buffer.size() < padded as u64
                || root.buffer.size() > reserved as u64
            {
                return Err(Error::InvalidFormat("exact public static physical extent"));
            }
        }
        self.weight_upload_ns += started.elapsed().as_nanos();
        self.uploaded_bytes += padded as u64;
        let words = padded / 4;
        self.cache.insert(
            key,
            Cached {
                handle: handle.clone(),
                words,
                bytes: reserved,
                last: self.tick,
            },
        );
        self.cache_bytes += reserved;
        Ok((handle, words))
    }

    fn fill_expert_misses(
        &mut self,
        bank: &ExpertBanks,
        weights: RoutedWeights<'_>,
        misses: &[(usize, usize)],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if let Some(admission) = self.resident_admission.clone() {
            admission.active()?;
            if misses.is_empty() {
                return Ok(());
            }
            // A previous fill must have crossed the NEXT completed router.
            // No cleanup/reentry while allocation holds the admission mutex.
            if let Some(sources) = &self.ordered_sources
                && sources.borrow().bytes() != 0
            {
                admission.poison();
                return Err(Error::InvalidFormat("ordered public source fence missing"));
            }
            let parts = [bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64);
            let requests = fill_requests(
                parts,
                misses.len(),
                self.batch_experts,
                self.expert_batch_bytes,
            )?;
            // The post-check occurs before resident.enqueue/slot-plan acceptance.
            admission.allocate(&requests, || {
                self.fill_expert_misses_inner(bank, weights, misses, cancel)
            })
        } else {
            if self.ordered_public_upload {
                return Err(Error::InvalidFormat(
                    "ordered public fill requires resident admission",
                ));
            }
            self.fill_expert_misses_inner(bank, weights, misses, cancel)
        }
    }

    fn fill_expert_misses_inner(
        &mut self,
        bank: &ExpertBanks,
        weights: RoutedWeights<'_>,
        misses: &[(usize, usize)],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if self.batch_experts {
            let stats = expert_banks::fill_plan(
                &self.client,
                bank,
                weights,
                misses,
                cancel,
                self.expert_batch_bytes,
                self.expert_scatter,
            )?;
            self.uploaded_bytes += stats.bytes;
            self.weight_read_ns += stats.read_ns;
            self.weight_upload_ns += stats.upload_ns;
            self.expert_uploads += stats.experts;
        } else {
            // Fixed-fixture paired comparison retains the original copy path.
            for &(expert, slot) in misses {
                for (target, tensor) in [
                    (bank.gate(), weights.gate),
                    (bank.up(), weights.up),
                    (bank.down(), weights.down),
                ] {
                    let (bytes, read, upload) = if let Some(sources) = &self.ordered_sources {
                        expert_banks::fill_ordered(
                            &self.client,
                            target,
                            tensor,
                            expert,
                            slot,
                            cancel,
                            &mut sources.borrow_mut(),
                        )?
                    } else {
                        expert_banks::fill(&self.client, target, tensor, expert, slot, cancel)?
                    };
                    self.uploaded_bytes += bytes;
                    self.weight_read_ns += read;
                    self.weight_upload_ns += upload;
                }
                self.expert_uploads += 1;
            }
        }
        Ok(())
    }

    /// Enqueue all ten routed experts without activation readbacks. The
    /// whole-token owner must validate the returned resident statuses and all
    /// separate shared-expert/projector statuses before reuse. On any later
    /// device/numerical/cancellation failure, discard this executor/context:
    /// public slot metadata below describes ORDERED uploads, not accepted GPU
    /// completion, and must not survive an uncertain partially executed queue.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Standalone hook until full resident graph integration.
    pub fn enqueue_routed_moe(
        &mut self,
        graph: kuno_text::model::BoundGraph<'_, '_>,
        arena: &crate::private_arena::PrivateArena,
        resident: &mut crate::resident_moe::ResidentMoe,
        input: crate::private_arena::BufferId,
        output: crate::private_arena::BufferId,
        selected: &[(usize, f32)],
        cancel: &AtomicBool,
    ) -> Result<crate::resident_moe::PendingMoe> {
        batch_moe::reject_scalar(self)?;
        let result =
            self.enqueue_routed_moe_inner(graph, arena, resident, input, output, selected, cancel);
        if result.is_err() {
            resident.poison();
            self.poison_resident();
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn enqueue_routed_moe_inner(
        &mut self,
        graph: kuno_text::model::BoundGraph<'_, '_>,
        arena: &crate::private_arena::PrivateArena,
        resident: &mut crate::resident_moe::ResidentMoe,
        input: crate::private_arena::BufferId,
        output: crate::private_arena::BufferId,
        selected: &[(usize, f32)],
        cancel: &AtomicBool,
    ) -> Result<crate::resident_moe::PendingMoe> {
        self.check_segmented_resident(self.resident_admission.is_some())?;
        if let Some(admission) = &self.resident_admission {
            admission.active()?;
        }
        let weights = resident.weights(graph)?;
        if [weights.gate, weights.up, weights.down]
            .iter()
            .any(|t| !t.belongs_to(self.model))
            || !std::ptr::eq(
                self.client.info(),
                arena
                    .client()
                    .map_err(|_| Error::InvalidFormat("resident arena unavailable"))?
                    .info(),
            )
            || selected.len() != 10
            || selected
                .iter()
                .any(|&(e, p)| e >= 512 || !p.is_finite() || p < 0.)
        {
            resident.poison();
            return Err(Error::InvalidFormat("resident MoE artifact/input binding"));
        }
        self.model.verify_file_unchanged()?;
        let key = weights.gate.info().name().to_owned();
        let mut bank = if let Some(bank) = self.banks.remove(&key) {
            bank
        } else {
            let size = ExpertBanks::allocation_with_profile(
                weights,
                self.bank_slots,
                self.packed_profile,
            )?;
            let bank = if let Some(admission) = &self.resident_admission {
                let requests = ExpertBanks::allocation_requests_with_profile(
                    weights,
                    self.bank_slots,
                    self.packed_profile,
                )?;
                admission.allocate_banks(&requests, || {
                    ExpertBanks::new_with_profile(
                        &self.client,
                        weights,
                        self.bank_slots,
                        self.packed_profile,
                    )
                })?
            } else {
                if self
                    .bank_reserved
                    .checked_add(size)
                    .and_then(|n| n.checked_add(self.budget + 512 * 1024 * 1024))
                    .is_none_or(|n| n > self.reserved_limit)
                {
                    resident.poison();
                    return Err(Error::LimitExceeded("resident expert bank residency"));
                }
                ExpertBanks::new_with_profile(
                    &self.client,
                    weights,
                    self.bank_slots,
                    self.packed_profile,
                )?
            };
            self.bank_reserved += bank.reserved();
            bank
        };
        let result = (|| {
            bank.check_names(weights)?;
            let ids = zeroize::Zeroizing::new(selected.iter().map(|&(e, _)| e).collect::<Vec<_>>());
            let plan = bank
                .slots
                .plan(&ids)
                .map_err(|_| Error::InvalidFormat("resident expert slot plan"))?;
            self.expert_hits += (ids.len() - plan.uploads.len()) as u64;
            self.fill_expert_misses(&bank, weights, &plan.uploads, cancel)?;
            let probabilities =
                zeroize::Zeroizing::new(selected.iter().map(|&(_, p)| p).collect::<Vec<_>>());
            let pending = resident.enqueue(
                graph,
                arena,
                &bank,
                input,
                output,
                &plan.slots,
                &probabilities,
                cancel,
                self.block_moe,
                self.word_down,
            )?;
            // Same-stream commands retain old handles through completion. The
            // next token cannot reuse this layer until whole-token acceptance.
            bank.slots
                .commit(plan)
                .map_err(|_| Error::InvalidFormat("resident expert slot commit"))?;
            if self.resident_admission.is_none()
                && self.reserved_bytes()? > self.reserved_limit as u64
            {
                return Err(Error::LimitExceeded(
                    "resident GPU allocator reserved memory",
                ));
            }
            Ok(pending)
        })();
        match result {
            Ok(pending) => {
                self.banks.insert(key, bank);
                self.moe_calls += 1;
                Ok(pending)
            }
            Err(error) => {
                resident.poison();
                self.poison_resident();
                self.bank_reserved -= bank.reserved();
                drop(bank);
                Err(error)
            }
        }
    }

    /// Enqueue a validated projection between resident laboratory allocations.
    /// No private activation is uploaded or downloaded here.
    ///
    /// # Safety
    /// x/y must be distinct, live allocations on THIS client's stream, with
    /// at least input_len/output_len f32s. The caller owns their full-span
    /// erasure on success, cancellation, panic and every partially queued error.
    /// Results require numerical validation before sequence state is committed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_matrix(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        x: &Handle,
        input_len: usize,
        y: &Handle,
        output_len: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        // SAFETY: forwarded original caller-owned span contract.
        unsafe {
            self.enqueue_matrix_inner(tensor, expert, x, input_len, y, output_len, cancel, None)
        }
    }

    /// Opt-in finite fusion only for the unchanged cooperative path. Returns
    /// false for every fallback, which still needs the original separate fence.
    /// # Safety
    /// In addition to enqueue_matrix's contract, fault is a separate one-word
    /// same-stream live owner, retained through whole-token completion/erasure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_matrix_with_finite_fault(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        x: &Handle,
        input_len: usize,
        y: &Handle,
        output_len: usize,
        fault: &Handle,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        let fused = self.fused_matrix_fault
            && !(self.block_kernel && (!self.block_large_only || output_len >= 16_384));
        if fault.size_in_used() != 4 || fault.stream != x.stream || fault.stream != y.stream {
            return Err(Error::InvalidBufferLength);
        }
        // SAFETY: caller proves all three arena spans distinct before entry.
        unsafe {
            self.enqueue_matrix_inner(
                tensor,
                expert,
                x,
                input_len,
                y,
                output_len,
                cancel,
                fused.then_some(fault),
            )?;
        }
        Ok(fused)
    }

    /// Core-only attention stage borrows authentic private GU scratch before
    /// that layer's prepare; the ordinary projection path remains separate.
    /// This queues work only; the caller still accepts every finite/device
    /// fence and commits all causal candidates together, or poisons/erases.
    /// # Safety
    /// x/y are distinct live same-client/arena owners covering their logical
    /// words through completed execution and full-span erasure. Spans are a
    /// guarded runtime packet, never a shader or persistent cache key.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue_strata_dense(
        &mut self,
        tensor: TensorView<'_>,
        input: &Handle,
        planes: &Handle,
        output: &Handle,
        fault: &Handle,
        spans: &quant_matvec::matmul::BatchSpans,
        cancel: &AtomicBool,
        prepare: bool,
    ) -> Result<()> {
        let matrix = Matrix::new(tensor)?;
        let dtype = tensor.info().dtype().gguf_id();
        if !tensor.belongs_to(self.model)
            || matrix.experts() != 1
            || spans.packed_first() != 0
            || !crate::strata_dense::supports_count(
                dtype,
                matrix.inputs(),
                planes.size_in_used(),
                spans.count(),
            )
        {
            return Err(Error::InvalidBufferLength);
        }
        self.model.verify_file_unchanged()?;
        self.check_target_rows_policy()?;
        if prepare {
            crate::strata_dense::prepare(
                &self.client,
                matrix.inputs(),
                input.clone(),
                planes.clone(),
                fault.clone(),
                spans,
            )?;
        }
        let row_bytes = tensor.info().row_bytes() as usize;
        if row_bytes == 0 || row_bytes > TILE_BYTES {
            return Err(Error::InvalidBufferLength);
        }
        let per_tile = (TILE_BYTES / row_bytes).min(16384);
        for first in (0..matrix.outputs()).step_by(per_tile) {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("resident batch cancelled"));
            }
            let rows = per_tile.min(matrix.outputs() - first);
            let (weights, _) = self.weight(tensor, 0, first, rows)?;
            crate::strata_dense::project(
                &self.client,
                dtype,
                rows,
                matrix.inputs(),
                weights,
                planes.clone(),
                output.clone(),
                fault.clone(),
                spans.output_first() + first,
                spans.output_stride(),
                spans.count(),
            )?;
        }
        Ok(())
    }
    /// x/y/fault remain distinct live arena roots through completion and erase.
    /// # Safety
    /// All logical views must cover the checked original matrix intervals.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_matrix_batch(
        &mut self,
        tensor: TensorView<'_>,
        x: &Handle,
        input_words: usize,
        y: &Handle,
        output_words: usize,
        fault: &Handle,
        spans: &quant_matvec::matmul::BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        // Reject before weights/cache mutation or any launch. The public model
        // policy is fixed between completed chunks, never chosen from count.
        self.check_target_rows_policy()?;
        if self.target_verify3_dense
            && (!(1..=if self.target_four_rows_active {
                crate::strata_port::verify_capacity()
            } else {
                3
            })
                .contains(&spans.count())
                || self.resident_admission.is_none())
        {
            return Err(Error::InvalidFormat("native Verify3 projection count"));
        }
        if self.native_projection_members == 1 && spans.count() != 1 {
            return Err(Error::InvalidFormat("native single projection count"));
        }
        if !tensor.belongs_to(self.model) || spans.packed_first() != 0 {
            return Err(Error::InvalidFormat(
                "resident batch artifact/packed binding",
            ));
        }
        self.model.verify_file_unchanged()?;
        let m = Matrix::new(tensor)?;
        if m.experts() != 1 {
            return Err(Error::InvalidFormat(
                "resident batch requires static projection",
            ));
        }
        crate::resident_batch_projection::validate_view(
            spans.count(),
            spans.input_first(),
            spans.input_stride(),
            m.inputs(),
            input_words,
        )?;
        crate::resident_batch_projection::validate_view(
            spans.count(),
            spans.output_first(),
            spans.output_stride(),
            m.outputs(),
            output_words,
        )?;
        if x.stream != y.stream
            || fault.stream != x.stream
            || fault.size_in_used() != 4
            || x.size_in_used() < (input_words as u64) * 4
            || y.size_in_used() < (output_words as u64) * 4
            || self.target_verify3_fused && y.size_in_used() != (output_words as u64) * 4
        {
            return Err(Error::InvalidBufferLength);
        }
        let fused = self.target_verify3_fused
            || self.fast_dense_fault
                && self.fast_dense
                && self.native_projection_members == 1
                && self.native_projection_tile() == 1;
        let row_bytes =
            usize::try_from(tensor.info().row_bytes()).map_err(|_| Error::ArithmeticOverflow)?;
        if row_bytes == 0 || row_bytes > TILE_BYTES {
            return Err(Error::LimitExceeded("resident batch packed row"));
        }
        let per_tile = (TILE_BYTES / row_bytes).min(16_384);
        for first in (0..m.outputs()).step_by(per_tile) {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("resident batch cancelled"));
            }
            let rows = per_tile.min(m.outputs() - first);
            let mut g = quant_matvec::matmul::MatrixGeometry::new(
                tensor.info().dtype().gguf_id(),
                rows,
                m.inputs(),
                self.native_projection_tile(),
            )
            .map_err(|_| Error::InvalidFormat("resident batch matrix geometry"))?;
            if self.native_projection_members == 1 {
                g = g
                    .with_single_member()
                    .map_err(|_| Error::InvalidFormat("resident single matrix geometry"))?;
            }
            let packet = quant_matvec::matmul::BatchSpans::new(
                spans.count(),
                0,
                spans.input_first(),
                spans.input_stride(),
                spans
                    .output_first()
                    .checked_add(first)
                    .ok_or(Error::ArithmeticOverflow)?,
                spans.output_stride(),
            );
            // Prove all runtime extents against the predicted packed view BEFORE
            // public-cache allocation; repeat against the actual returned words.
            g.validate_spans(
                &packet,
                g.total_bytes().div_ceil(4),
                input_words,
                output_words,
            )
            .map_err(|_| Error::InvalidBufferLength)?;
            let verify4 = if self.target_four_rows_active && spans.count() >= 4 {
                let spec = crate::next_shared::MatrixSpec::new(g.rows(), g.columns(), g.dtype())?;
                crate::next_shared::validate_verify4_spans(
                    spec,
                    &packet,
                    g.total_bytes().div_ceil(4) as u64 * 4,
                    input_words,
                    output_words,
                )?;
                Some(spec)
            } else {
                None
            };
            let verify3 = if self.target_verify3_dense && verify4.is_none() && spans.count() <= 3 {
                let spec = target_verify3_iq4_spec(
                    g.rows(),
                    g.columns(),
                    g.dtype(),
                    self.target_iq4_decoder,
                )?;
                crate::next_shared::validate_verify3_spans(
                    spec,
                    &packet,
                    g.total_bytes().div_ceil(4) as u64 * 4,
                    input_words,
                    output_words,
                )?;
                Some(spec)
            } else {
                None
            };
            let (weights, words) = self.weight(tensor, 0, first, rows)?;
            if self.resident_admission.is_none()
                && self.reserved_bytes()? > self.reserved_limit as u64
            {
                return Err(Error::LimitExceeded("GPU allocator reserved memory"));
            }
            g.validate_spans(&packet, words, input_words, output_words)
                .map_err(|_| Error::InvalidBufferLength)?;
            // SAFETY: same original packed row proof as scalar enqueue; caller
            // owns disjoint x/y, and the exact runtime spans passed validation.
            unsafe {
                if let Some(spec) = verify4 {
                    let dense = self
                        .target_verify4_dense_calls
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                    let fused = self
                        .target_verify4_fused_calls
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                    crate::next_shared::matrix_verify4_finite(
                        &self.client,
                        spec,
                        weights,
                        x.clone(),
                        y.clone(),
                        fault.clone(),
                        input_words,
                        output_words,
                        &packet,
                        m.outputs(),
                        spans.output_first(),
                        first + rows == m.outputs(),
                    )?;
                    self.target_verify4_dense_calls = dense;
                    self.target_verify4_fused_calls = fused;
                } else if let Some(spec) = verify3 {
                    let iq4_calls = next_target_iq4_dispatch_count(
                        self.target_iq4_decoder_calls,
                        spec.iq4_decoder(),
                    )?;
                    let calls = self
                        .target_verify3_dense_calls
                        .checked_add(1)
                        .ok_or(Error::ArithmeticOverflow)?;
                    if self.target_verify3_fused {
                        let fused_calls = self
                            .target_verify3_fused_calls
                            .checked_add(1)
                            .ok_or(Error::ArithmeticOverflow)?;
                        crate::next_shared::matrix_verify3_finite(
                            &self.client,
                            spec,
                            weights,
                            x.clone(),
                            y.clone(),
                            fault.clone(),
                            input_words,
                            output_words,
                            &packet,
                            m.outputs(),
                            spans.output_first(),
                            first + rows == m.outputs(),
                        )?;
                        self.target_verify3_fused_calls = fused_calls;
                    } else {
                        crate::next_shared::matrix_verify3(
                            &self.client,
                            spec,
                            weights,
                            x.clone(),
                            y.clone(),
                            fault.clone(),
                            input_words,
                            output_words,
                            &packet,
                        )?;
                    }
                    self.target_verify3_dense_calls = calls;
                    self.target_iq4_decoder_calls = iq4_calls;
                    self.shared_dense_calls += 1;
                    self.fast_dense_calls += 1;
                } else if self.fast_dense
                    && self.native_projection_members == 1
                    && g.tile_tokens() == 1
                {
                    if !packet.packed_first().is_multiple_of(4) {
                        return Err(Error::InvalidBufferLength);
                    }
                    if self.shared_kernels
                        && crate::next_shared::MatrixSpec::supports_dtype(g.dtype())
                    {
                        if packet.packed_first() != 0 {
                            return Err(Error::InvalidFormat("shared count1 projection contract"));
                        }
                        if fused {
                            crate::next_shared::matrix_fault(
                                &self.client,
                                crate::next_shared::MatrixSpec::new(
                                    g.rows(),
                                    g.columns(),
                                    g.dtype(),
                                )?,
                                weights,
                                x.clone(),
                                y.clone(),
                                input_words,
                                output_words,
                                packet.input_first(),
                                packet.output_first(),
                                fault.clone(),
                                m.outputs(),
                                spans.output_first(),
                                first + rows == m.outputs(),
                            )?;
                            self.fast_dense_fault_calls += 1;
                        } else {
                            crate::next_shared::matrix(
                                &self.client,
                                crate::next_shared::MatrixSpec::new(
                                    g.rows(),
                                    g.columns(),
                                    g.dtype(),
                                )?,
                                weights,
                                x.clone(),
                                y.clone(),
                                input_words,
                                output_words,
                                packet.input_first(),
                                packet.output_first(),
                            )?;
                        }
                        self.shared_dense_calls += 1;
                    } else if fused {
                        let tail = first + rows == m.outputs();
                        quant_matvec::fast::packed_single_fault::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            quant_matvec::fast::fault_grid(g.rows(), output_words, tail),
                            quant_matvec::fast::cube(),
                            ArrayArg::from_raw_parts(weights, words),
                            ArrayArg::from_raw_parts(x.clone(), input_words),
                            ArrayArg::from_raw_parts(y.clone(), output_words),
                            ArrayArg::from_raw_parts(fault.clone(), 1),
                            packet.count(),
                            packet.packed_first(),
                            packet.input_first(),
                            packet.output_first(),
                            spans.output_first(),
                            g.rows(),
                            g.columns(),
                            g.dtype(),
                            g.row_bytes(),
                            m.outputs(),
                            tail,
                        );
                        self.fast_dense_fault_calls += 1;
                    } else {
                        quant_matvec::fast::packed_single::launch_unchecked::<WgpuRuntime>(
                            &self.client,
                            quant_matvec::fast::grid(g.rows()),
                            quant_matvec::fast::cube(),
                            ArrayArg::from_raw_parts(weights, words),
                            ArrayArg::from_raw_parts(x.clone(), input_words),
                            ArrayArg::from_raw_parts(y.clone(), output_words),
                            packet.count(),
                            packet.packed_first(),
                            packet.input_first(),
                            packet.output_first(),
                            g.rows(),
                            g.columns(),
                            g.dtype(),
                            g.row_bytes(),
                        );
                    }
                    self.fast_dense_calls += 1;
                } else {
                    quant_matvec::matmul::packed_matmul::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        g.grid(),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(weights, words),
                        ArrayArg::from_raw_parts(x.clone(), input_words),
                        ArrayArg::from_raw_parts(y.clone(), output_words),
                        packet.count(),
                        packet.packed_first(),
                        packet.input_first(),
                        packet.input_stride(),
                        packet.output_first(),
                        packet.output_stride(),
                        g.rows(),
                        g.columns(),
                        g.dtype(),
                        g.tile_tokens(),
                    );
                }
            }
            self.calls += 1;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("resident batch cancelled"));
        }
        Ok(fused)
    }

    /// Fixed-public admitted low-bit Final head only. The sealed GR caller
    /// lends its dead full NORM page; no owner, cache format/key or allocation
    /// changes. This explicitly changes activation/reduction arithmetic.
    /// # Safety
    /// x/planes/y/fault are distinct complete same-stream private arena owners,
    /// frozen through all queued tiles and the original whole-model proof.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_q6_integer_final_head(
        &mut self,
        tensor: TensorView<'_>,
        x: &Handle,
        planes: &Handle,
        y: &Handle,
        fault: &Handle,
        spans: &quant_matvec::matmul::BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        use crate::q6_integer_matvec::{self as q6, PrepareSpans, ProjectSpans, Spec};
        if !self.q6_integer_head || !tensor.belongs_to(self.model) {
            return Err(Error::InvalidFormat("Q6 integer final head selection"));
        }
        self.model.verify_file_unchanged()?;
        let matrix = Matrix::new(tensor)?;
        validate_q6_final_head(
            self.packed_profile,
            self.resident_admission.is_some() && self.transient_resident.is_some(),
            self.native_projection_members,
            self.native_projection_tile(),
            tensor.info().name(),
            tensor.info().dtype().gguf_id(),
            matrix.inputs(),
            matrix.outputs(),
            matrix.experts(),
            spans,
        )?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .active()?;
        let mut extents = [0usize; 4];
        for (h, extent) in [x, planes, y, fault].into_iter().zip(&mut extents) {
            if h.stream != x.stream
                || h.size() != h.size_in_used()
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || !h.size().is_multiple_of(4)
                || h.size() / 4 > u32::MAX as u64
            {
                return Err(Error::InvalidBufferLength);
            }
            *extent = h.size() as usize / 4;
        }
        let prepare_spec = Spec::new(16384, 2560)?;
        let prepare_spans = PrepareSpans {
            input_first: spans.input_first(),
            planes_first: 0,
            fault_first: 0,
        };
        prepare_spans.validate(prepare_spec, [extents[0], extents[1], extents[3]])?;
        // Exact unchanged cache partition: fifteen16384-row tiles and2560
        // final rows. Prove every predicted source/output before preparation.
        for first in (0..248320).step_by(16384) {
            let rows = 16384.min(248320 - first);
            let spec = Spec::new(rows, 2560)?;
            ProjectSpans {
                packed_first: 0,
                planes_first: 0,
                output_first: first,
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: 248320,
            }
            .validate(
                spec,
                spec.weight_bytes(),
                [extents[1], extents[2], extents[3]],
            )?;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("Q6 integer final head cancelled"));
        }
        let prepares = self
            .q6_integer_head_prepares
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        q6::prepare(
            &self.client,
            prepare_spec,
            x.clone(),
            planes.clone(),
            fault.clone(),
            prepare_spans,
        )?;
        self.q6_integer_head_prepares = prepares;
        for first in (0..248320).step_by(16384) {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("Q6 integer final head cancelled"));
            }
            let rows = 16384.min(248320 - first);
            let spec = Spec::new(rows, 2560)?;
            let packet = ProjectSpans {
                packed_first: 0,
                planes_first: 0,
                output_first: first,
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: 248320,
            };
            let (weights, words) = self.weight(tensor, 0, first, rows)?;
            packet.validate(spec, words * 4, [extents[1], extents[2], extents[3]])?;
            let tiles = self
                .q6_integer_head_tiles
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            let calls = self.calls.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
            q6::project(
                &self.client,
                spec,
                weights,
                planes.clone(),
                y.clone(),
                fault.clone(),
                packet,
                first + rows == 248320,
            )?;
            self.q6_integer_head_tiles = tiles;
            self.calls = calls;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("Q6 integer final head cancelled"));
        }
        Ok(())
    }

    /// Fixed-public all-row verifier head only, using the unchanged sixteen tiles. The sealed GR caller
    /// lends its dead full NORM page; no owner, cache format/key or allocation
    /// changes. This explicitly changes activation/reduction arithmetic.
    /// # Safety
    /// x/planes/y/fault are distinct complete same-stream private arena owners,
    /// frozen through all queued tiles and the original whole-model proof.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_q6_integer_verify_head(
        &mut self,
        tensor: TensorView<'_>,
        x: &Handle,
        planes: &Handle,
        y: &Handle,
        fault: &Handle,
        spans: &quant_matvec::matmul::BatchSpans,
        cancel: &AtomicBool,
    ) -> Result<()> {
        use crate::q6_integer_matvec::{
            Spec,
            batch::{self as q6, PrepareSpans, ProjectSpans},
        };
        self.check_target_rows_policy()?;
        if !self.q6_integer_verify_head || !tensor.belongs_to(self.model) {
            return Err(Error::InvalidFormat("Q6 integer verifier head selection"));
        }
        self.model.verify_file_unchanged()?;
        let matrix = Matrix::new(tensor)?;
        validate_q6_verify_head(
            self.packed_profile,
            self.resident_admission.is_some() && self.transient_resident.is_some(),
            self.native_projection_members,
            self.native_projection_tile(),
            self.target_four_rows_reserved,
            self.target_four_rows_active,
            tensor.info().name(),
            tensor.info().dtype().gguf_id(),
            matrix.inputs(),
            matrix.outputs(),
            matrix.experts(),
            spans,
        )?;
        self.resident_admission
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .active()?;
        let mut extents = [0usize; 4];
        for (h, extent) in [x, planes, y, fault].into_iter().zip(&mut extents) {
            if h.stream != x.stream
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.size() != h.size_in_used()
                || !h.size().is_multiple_of(4)
                || h.size() / 4 > u32::MAX as u64
            {
                return Err(Error::InvalidBufferLength);
            }
            *extent = h.size() as usize / 4;
        }
        if crate::strata_intel_core::enabled() {
            if !self.global_target_scope()? {
                return Err(Error::InvalidBufferLength);
            }
            let prepares = self
                .q6_integer_verify_head_prepares
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            let tiles = self
                .q6_integer_verify_head_tiles
                .checked_add(248320_usize.div_ceil(16384) as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            let members = self
                .q6_integer_verify_head_members
                .checked_add(spans.count() as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            self.enqueue_strata_dense(tensor, x, planes, y, fault, spans, cancel, true)?;
            self.q6_integer_verify_head_prepares = prepares;
            self.q6_integer_verify_head_tiles = tiles;
            self.q6_integer_verify_head_members = members;
            return Ok(());
        }
        let prepare_spec = Spec::new(16384, 2560)?;
        let prepare_spans = PrepareSpans {
            count: spans.count(),
            input_first: spans.input_first(),
            input_stride: spans.input_stride(),
            planes_first: 0,
            planes_stride: prepare_spec.plane_words(),
            fault_first: 0,
        };
        prepare_spans.validate(prepare_spec, [extents[0], extents[1], extents[3]])?;
        // Exact unchanged cache partition: fifteen16384-row tiles and2560
        // final rows. Prove every predicted source/output before preparation.
        for first in (0..248320).step_by(16384) {
            let rows = 16384.min(248320 - first);
            let spec = Spec::new(rows, 2560)?;
            ProjectSpans {
                count: spans.count(),
                packed_first: 0,
                planes_first: 0,
                planes_stride: prepare_spec.plane_words(),
                output_first: first,
                output_stride: spans.output_stride(),
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: 248320,
            }
            .validate(
                spec,
                spec.weight_bytes(),
                [extents[1], extents[2], extents[3]],
            )?;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("Q6 integer verifier head cancelled"));
        }
        let members = self
            .q6_integer_verify_head_members
            .checked_add(spans.count() as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        let prepares = self
            .q6_integer_verify_head_prepares
            .checked_add(1)
            .ok_or(Error::ArithmeticOverflow)?;
        q6::prepare(
            &self.client,
            prepare_spec,
            x.clone(),
            planes.clone(),
            fault.clone(),
            prepare_spans,
        )?;
        self.q6_integer_verify_head_prepares = prepares;
        for first in (0..248320).step_by(16384) {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("Q6 integer verifier head cancelled"));
            }
            let rows = 16384.min(248320 - first);
            let spec = Spec::new(rows, 2560)?;
            let packet = ProjectSpans {
                count: spans.count(),
                packed_first: 0,
                planes_first: 0,
                planes_stride: prepare_spec.plane_words(),
                output_first: first,
                output_stride: spans.output_stride(),
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: 248320,
            };
            let (weights, words) = self.weight(tensor, 0, first, rows)?;
            packet.validate(spec, words * 4, [extents[1], extents[2], extents[3]])?;
            let tiles = self
                .q6_integer_verify_head_tiles
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            let calls = self.calls.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
            q6::project(
                &self.client,
                spec,
                weights,
                planes.clone(),
                y.clone(),
                fault.clone(),
                packet,
                first + rows == 248320,
            )?;
            self.q6_integer_verify_head_tiles = tiles;
            self.calls = calls;
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("Q6 integer verifier head cancelled"));
        }
        self.q6_integer_verify_head_members = members;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_matrix_inner(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        x: &Handle,
        input_len: usize,
        y: &Handle,
        output_len: usize,
        cancel: &AtomicBool,
        fault: Option<&Handle>,
    ) -> Result<()> {
        if !tensor.belongs_to(self.model) {
            return Err(Error::InvalidFormat("GPU artifact binding"));
        }
        self.model.verify_file_unchanged()?;
        let matrix = Matrix::new(tensor)?;
        if input_len != matrix.inputs()
            || output_len != matrix.outputs()
            || expert >= matrix.experts()
            || x.size_in_used() < (input_len * 4) as u64
            || y.size_in_used() < (output_len * 4) as u64
        {
            return Err(Error::InvalidBufferLength);
        }
        let row_bytes = tensor.info().row_bytes() as usize;
        if row_bytes > TILE_BYTES {
            return Err(Error::LimitExceeded("GPU row"));
        }
        let per_tile = (TILE_BYTES / row_bytes).min(16_384);
        for first in (0..matrix.outputs()).step_by(per_tile) {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::InvalidFormat("GPU execution cancelled"));
            }
            let rows = per_tile.min(matrix.outputs() - first);
            let dtype = tensor.info().dtype().gguf_id();
            quant_matvec::validate_layout(
                dtype,
                rows,
                matrix.inputs(),
                rows * row_bytes,
                input_len,
                rows,
            )
            .map_err(|_| Error::InvalidFormat("GPU packed layout"))?;
            let (weights, words) = self.weight(tensor, expert, first, rows)?;
            if self.resident_admission.is_none()
                && self.reserved_bytes()? > self.reserved_limit as u64
            {
                return Err(Error::LimitExceeded("GPU allocator reserved memory"));
            }
            // SAFETY: validated packed row geometry, exact buffer lengths,
            // output_first+rows<=matrix.outputs(), and 64-lane row grid.
            unsafe {
                if self.block_kernel && (!self.block_large_only || matrix.outputs() >= 16_384) {
                    quant_matvec::block::block_matvec::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(weights, words),
                        ArrayArg::from_raw_parts(x.clone(), input_len),
                        ArrayArg::from_raw_parts(y.clone(), matrix.outputs()),
                        rows,
                        matrix.inputs(),
                        dtype,
                        first,
                    );
                } else if let Some(fault) = fault {
                    quant_matvec::cooperative::cooperative_matvec_with_fault::launch_unchecked::<
                        WgpuRuntime,
                    >(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(weights, words),
                        ArrayArg::from_raw_parts(x.clone(), input_len),
                        ArrayArg::from_raw_parts(y.clone(), matrix.outputs()),
                        ArrayArg::from_raw_parts(fault.clone(), 1),
                        rows,
                        matrix.inputs(),
                        dtype,
                        first,
                    );
                } else {
                    quant_matvec::cooperative::cooperative_matvec::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(weights, words),
                        ArrayArg::from_raw_parts(x.clone(), input_len),
                        ArrayArg::from_raw_parts(y.clone(), matrix.outputs()),
                        rows,
                        matrix.inputs(),
                        dtype,
                        first,
                    );
                }
            }
            self.calls += 1;
        }
        Ok(())
    }
}

impl MatrixExecutor for GpuExecutor<'_> {
    fn delta_rule(
        &mut self,
        state: &mut kuno_text::ops::GdnState,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        self.nonresident_only()?;
        let start = Instant::now();
        if !self.gpu_gdn {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::InvalidFormat("graph cancelled"));
            }
            return state
                .step(query, key, value, log_decay, beta)
                .map_err(|_| Error::InvalidFormat("CPU GDN step"));
        }
        let out = crate::gdn_executor::run(
            &self.client,
            state,
            query,
            key,
            value,
            log_decay,
            beta,
            cancel,
        )?;
        self.gdn_ns += start.elapsed().as_nanos();
        self.gdn_calls += 1;
        Ok(out)
    }

    fn routed_moe(
        &mut self,
        weights: RoutedWeights<'_>,
        input: &[f32],
        selected: &[(usize, f32)],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        batch_moe::reject_scalar(self)?;
        self.check_segmented_resident(false)?;
        self.nonresident_only()?;
        let started = Instant::now();
        if [weights.gate, weights.up, weights.down]
            .iter()
            .any(|t| !t.belongs_to(self.model))
        {
            return Err(Error::InvalidFormat("GPU artifact binding"));
        }
        self.model.verify_file_unchanged()?;
        let key = weights.gate.info().name().to_owned();
        let mut bank = if let Some(bank) = self.banks.remove(&key) {
            bank
        } else {
            let size = ExpertBanks::allocation_with_profile(
                weights,
                self.bank_slots,
                self.packed_profile,
            )?;
            if self
                .bank_reserved
                .checked_add(size)
                .and_then(|n| n.checked_add(self.budget + 512 * 1024 * 1024))
                .is_none_or(|n| n > self.reserved_limit)
            {
                return Err(Error::LimitExceeded("expert bank residency"));
            }
            let bank = ExpertBanks::new_with_profile(
                &self.client,
                weights,
                self.bank_slots,
                self.packed_profile,
            )?;
            self.bank_reserved += bank.reserved();
            bank
        };
        let result = (|| {
            bank.check(weights, input, selected)?;
            let ids: Vec<_> = selected.iter().map(|&(e, _)| e).collect();
            let plan = bank
                .slots
                .plan(&ids)
                .map_err(|_| Error::InvalidFormat("expert slot plan"))?;
            self.expert_hits += (ids.len() - plan.uploads.len()) as u64;
            self.fill_expert_misses(&bank, weights, &plan.uploads, cancel)?;
            let probabilities: Vec<_> = selected.iter().map(|&(_, p)| p).collect();
            let mut output = expert_banks::execute(
                &self.client,
                &bank,
                weights,
                input,
                &plan.slots,
                &probabilities,
                cancel,
                self.block_moe,
                self.word_down,
            )?;
            let validate = (|| {
                bank.slots
                    .commit(plan)
                    .map_err(|_| Error::InvalidFormat("expert slot commit"))?;
                if self.reserved_bytes()? > self.reserved_limit as u64 {
                    return Err(Error::LimitExceeded("GPU allocator reserved memory"));
                }
                Ok(())
            })();
            if let Err(e) = validate {
                output.zeroize();
                return Err(e);
            }
            Ok(output)
        })();
        match result {
            Ok(output) => {
                self.banks.insert(key, bank);
                self.moe_calls += 1;
                self.moe_ns += started.elapsed().as_nanos();
                Ok(Some(output))
            }
            Err(error) => {
                // A partial upload invalidates old contents; never reuse stale metadata.
                self.bank_reserved -= bank.reserved();
                drop(bank);
                self.client.memory_cleanup();
                Err(error)
            }
        }
    }

    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        self.nonresident_only()?;
        let started = Instant::now();
        if !tensor.belongs_to(self.model) {
            return Err(Error::InvalidFormat("GPU artifact binding"));
        }
        self.model.verify_file_unchanged()?;
        let matrix = Matrix::new(tensor)?;
        if input.len() != matrix.inputs()
            || expert >= matrix.experts()
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidBufferLength);
        }
        if self.small_cpu_enabled
            && let Some(result) = self.small_cpu.run(tensor, expert, input, cancel)?
        {
            let ns = started.elapsed().as_nanos();
            self.small_cpu_ns += ns;
            self.small_cpu_calls += 1;
            self.matrix_ns += ns;
            *self
                .by_tensor
                .entry(tensor.info().name().into())
                .or_default() += ns;
            return Ok(result);
        }
        let x = self.client.create_from_slice(f32::as_bytes(input));
        let y = self.client.empty(matrix.outputs() * 4);
        let execute = (|| {
            if self.reserved_bytes()? > self.reserved_limit as u64 {
                return Err(Error::LimitExceeded("GPU allocator reserved memory"));
            }
            // SAFETY: distinct x/y were allocated here with these exact f32
            // lengths; this method owns erasure on every returned result.
            unsafe {
                self.enqueue_matrix(
                    tensor,
                    expert,
                    &x,
                    input.len(),
                    &y,
                    matrix.outputs(),
                    cancel,
                )?;
            }
            let waited = Instant::now();
            let read = self.client.read_one(y.clone());
            self.gpu_wait_ns += waited.elapsed().as_nanos();
            let mut bytes = read.map_err(|_| Error::InvalidFormat("GPU output read"))?;
            let values = f32::from_bytes(&bytes);
            let out = if values.len() != matrix.outputs() || values.iter().any(|v| !v.is_finite()) {
                Err(Error::InvalidFormat("GPU output values"))
            } else if cancel.load(Ordering::Relaxed) {
                Err(Error::InvalidFormat("GPU execution cancelled"))
            } else {
                Ok(values.to_vec())
            };
            bytes.zeroize();
            out
        })();
        // Queue erasure on ordinary success/error before returning allocations.
        expert_banks::clear(&self.client, &[(x, input.len()), (y, matrix.outputs())]);
        let result = execute?;
        let elapsed = started.elapsed().as_nanos();
        self.matrix_ns += elapsed;
        *self
            .by_tensor
            .entry(tensor.info().name().into())
            .or_default() += elapsed;
        Ok(result)
    }
}
impl Drop for GpuExecutor<'_> {
    fn drop(&mut self) {
        if self.transient_resident.is_some() {
            // Drain and erase runtime ownership before releasing public cache
            // handles. Sanitizer failure never becomes a clean-context claim.
            let _ = self.close_transient_resident();
            return;
        }
        self.cache.clear();
        self.cache_bytes = 0;
        self.banks.clear();
        self.resident_banks_frozen = false;
        self.target_frozen_host_source_identity = None;
        self.target_frozen_host_router_reserved = false;
        self.target_frozen_host_staging_reserved = false;
        self.target_frozen_gu_integer_reserved = false;
        self.target_frozen_down_integer_reserved = false;
        self.target_iq4_decoder = crate::next_shared::Iq4Decoder::Original;
        self.target_iq4_decoder_calls = 0;
        self.q6_integer_verify_head = false;
        self.q6_integer_verify_head_prepares = 0;
        self.q6_integer_verify_head_tiles = 0;
        self.q6_integer_verify_head_members = 0;
        self.bank_reserved = 0;
        self.client.memory_cleanup();
        let _ = cubecl::future::block_on(self.client.sync());
    }
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn fixed_oracle_fixtures_cannot_cross_original_quant_profiles() {
        use crate::expert_banks::OriginalPackedProfile as P;
        use crate::oracle_trace::Fixture as F;
        for fixture in [
            F::Q4Rainwater,
            F::Q3Prefix23,
            F::DaslabPrefix13,
            F::Daslab32,
        ] {
            for profile in [P::Q4, P::UnslothQ3, P::DaslabIq3] {
                assert_eq!(
                    super::oracle_fixture_profile_matches(fixture, profile),
                    matches!(
                        (fixture, profile),
                        (F::Q4Rainwater, P::Q4)
                            | (F::Q3Prefix23, P::UnslothQ3)
                            | (F::DaslabPrefix13, P::DaslabIq3)
                            | (F::Daslab32, P::DaslabIq3)
                    )
                );
            }
        }
    }
    #[test]
    fn native_projection_policy_is_public_bounded_and_default_eight() {
        use super::NativeProjectionTile;
        assert_eq!(NativeProjectionTile::default().tokens(), 8);
        for tile in [1, 4, 8] {
            assert_eq!(NativeProjectionTile::new(tile).unwrap().tokens(), tile);
        }
        for tile in [0, 2, 3, 7, 16, usize::MAX] {
            assert!(NativeProjectionTile::new(tile).is_err());
        }
    }

    #[test]
    fn cleanup_categories_discard_backend_descriptions_and_private_values() {
        use super::cleanup_failure_category;
        assert_eq!(
            cleanup_failure_category("gpu_validation"),
            Some("gpu_validation")
        );
        assert_eq!(
            cleanup_failure_category("audit_budget"),
            Some("audit_budget")
        );
        for unknown in [
            "",
            "gpu_validation:fixture value",
            "fixture private text",
            "shader source",
        ] {
            assert!(cleanup_failure_category(unknown).is_none());
        }
    }

    #[test]
    fn first_fixed_cleanup_category_survives_checked_retirement() {
        use super::{Error, checked_transient_cleanup};
        use std::sync::atomic::{AtomicBool, Ordering};
        let retired = AtomicBool::new(false);
        assert!(matches!(
            checked_transient_cleanup(&retired, || Err(Error::InvalidFormat("gpu_validation"))),
            Err(Error::InvalidFormat("gpu_validation"))
        ));
        assert!(retired.load(Ordering::Acquire));
        assert!(matches!(
            checked_transient_cleanup(&retired, || Err(Error::InvalidFormat(
                "fixture private text"
            ))),
            Err(Error::InvalidFormat("resident runtime I/O cleanup failed"))
        ));
    }

    use super::*;
    #[test]
    fn only_owned_live_unordered_token_epochs_can_defer_count1_bridges() {
        for transient in [false, true] {
            for token in [false, true] {
                for ordered in [false, true] {
                    for retired in [false, true] {
                        for cancelled in [false, true] {
                            assert_eq!(
                                deferred_bridge_allowed(
                                    transient, token, ordered, retired, cancelled
                                ),
                                [transient, token, ordered, retired, cancelled]
                                    == [true, true, false, false, false]
                            );
                            if deferred_bridge_allowed(
                                transient, token, ordered, retired, cancelled,
                            ) {
                                assert!(!checkpoint_needed(true, token, ordered));
                                assert!(checkpoint_needed(false, token, ordered));
                            }
                        }
                    }
                }
            }
        }
    }
    use std::cell::{Cell, RefCell};
    const PAGE: u64 = main_admission::MIN_PAGE_BYTES;
    fn failure<T>() -> Result<T> {
        Err(Error::InvalidFormat("fixed public failure fixture"))
    }

    #[test]
    fn token_cadence_keeps_cleanup_mandatory_at_final_fence() {
        assert!(checkpoint_needed(true, false, false));
        assert!(checkpoint_needed(false, false, false));
        assert!(!checkpoint_needed(true, true, false));
        assert!(checkpoint_needed(false, true, false));
        assert!(checkpoint_needed(true, true, true)); // Cache-hit next router too.
    }

    #[test]
    fn deferred_fence_drops_host_lease_and_final_failure_poisons_main() {
        struct Lease<'a>(&'a Cell<bool>);
        impl Drop for Lease<'_> {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let leased = Cell::new(false);
        let mut main = MainAdmission::new(8 * PAGE, PAGE).unwrap();
        main.precharge(&[PAGE]).unwrap();
        let out = guarded_readback(
            &mut main,
            || {
                leased.set(true);
                let _lease = Lease(&leased);
                Ok(zeroize::Zeroizing::new(vec![1_u32]))
            },
            || {
                assert!(!leased.get());
                assert!(!checkpoint_needed(true, true, false));
                Ok(())
            },
            || Ok(PAGE),
        )
        .unwrap();
        assert_eq!(*out, vec![1]);
        // Final acceptance still runs the mandatory checkpoint first. Failure
        // poisons main admission before any new fence credit.
        assert!(guarded_readback(&mut main, || Ok(()), failure, || Ok(PAGE)).is_err());
        assert!(main.is_poisoned());
    }

    #[test]
    fn read_lease_drops_before_sanitizer_and_fence_credit() {
        struct Lease<'a>(&'a Cell<bool>);
        impl Drop for Lease<'_> {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let leased = Cell::new(false);
        let events = RefCell::new(Vec::new());
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        ledger.precharge(&[PAGE]).unwrap();
        let owned = guarded_readback(
            &mut ledger,
            || {
                leased.set(true);
                let _bytes = Lease(&leased);
                events.borrow_mut().push("read lease");
                Ok(zeroize::Zeroizing::new(vec![1.0_f32]))
            },
            || {
                assert!(!leased.get());
                events.borrow_mut().push("sanitize");
                Ok(())
            },
            || {
                events.borrow_mut().push("telemetry");
                Ok(PAGE)
            },
        )
        .unwrap();
        assert_eq!(*owned, vec![1.0_f32]);
        assert_eq!(*events.borrow(), ["read lease", "sanitize", "telemetry"]);
        assert_eq!(ledger.upper_bytes(), PAGE);
    }

    fn completed_audit_receipt() -> TransientIoReceipt {
        TransientIoReceipt {
            epoch: 7,
            upload_bytes: 32768,
            uniform_bytes: 512,
            readback_bytes: 32768,
            inspected_bytes: 66048,
            inspected_buffers: 4,
        }
    }
    #[test]
    fn audited_readback_runs_one_full_cleanup_before_source_release_and_fence_credit() {
        struct Lease<'a>(&'a Cell<bool>, &'a RefCell<Vec<&'static str>>);
        impl Drop for Lease<'_> {
            fn drop(&mut self) {
                self.0.set(false);
                self.1.borrow_mut().push("lease drop");
            }
        }
        for ordered in [false, true] {
            let leased = Cell::new(false);
            let audits = Cell::new(0);
            let events = RefCell::new(Vec::new());
            let retired = AtomicBool::new(false);
            let poison = AtomicBool::new(false);
            let cancel = AtomicBool::new(false);
            let sources = RefCell::new(crate::ordered_public::PendingSources::new());
            sources
                .borrow_mut()
                .enqueue(vec![0; PAGE as usize], Ok, |_, _| Ok(()))
                .unwrap();
            let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
            ledger.precharge(&[PAGE]).unwrap();
            let value = guarded_readback(
                &mut ledger,
                || {
                    leased.set(true);
                    let _lease = Lease(&leased, &events);
                    events.borrow_mut().push("read/decode");
                    Ok(zeroize::Zeroizing::new(vec![17_u32]))
                },
                || {
                    checked_audited_boundary(&retired, || {
                        let audit = || {
                            assert!(!leased.get());
                            events.borrow_mut().push("full audit");
                            audits.set(audits.get() + 1);
                            Ok(completed_audit_receipt())
                        };
                        if ordered {
                            audited_source_cleanup(
                                Some(&mut *sources.borrow_mut()),
                                &retired,
                                &poison,
                                &cancel,
                                audit,
                            )?;
                            events.borrow_mut().push("sources released");
                        } else {
                            audited_source_cleanup::<usize>(
                                None, &retired, &poison, &cancel, audit,
                            )?;
                        }
                        Ok(())
                    })
                },
                || {
                    assert_eq!(sources.borrow().bytes(), if ordered { 0 } else { PAGE });
                    events.borrow_mut().push("telemetry");
                    Ok(PAGE)
                },
            )
            .unwrap();
            assert_eq!(*value, vec![17]);
            assert_eq!(audits.get(), 1); // no second non-audited/empty checkpoint
            let mut expected = vec!["read/decode", "lease drop", "full audit"];
            if ordered {
                expected.push("sources released");
            }
            expected.push("telemetry");
            assert_eq!(*events.borrow(), expected);
            assert_eq!(ledger.upper_bytes(), PAGE);
            assert!(!retired.load(Ordering::Acquire));
            // This unconsumed fake source belongs to the separately modeled
            // ordinary path; explicit teardown avoids abandoned-owner leakage.
            sources
                .borrow_mut()
                .after_completed_read(|| Ok(()))
                .unwrap();
        }
    }

    #[test]
    fn audited_readback_failures_never_credit_and_health_failure_retains_ordered_sources() {
        for failure in 0..8 {
            let retired = AtomicBool::new(false);
            let poison = AtomicBool::new(false);
            let cancel = AtomicBool::new(false);
            let audits = Cell::new(0);
            let sampled = Cell::new(false);
            let sources = RefCell::new(crate::ordered_public::PendingSources::new());
            sources
                .borrow_mut()
                .enqueue(vec![0; PAGE as usize], Ok, |_, _| Ok(()))
                .unwrap();
            let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
            ledger.precharge(&[PAGE]).unwrap();
            let result = guarded_readback(
                &mut ledger,
                || {
                    if failure == 0 {
                        Err(Error::InvalidBufferLength)
                    } else {
                        Ok(17)
                    }
                },
                || {
                    checked_audited_boundary(&retired, || {
                        audited_source_cleanup(
                            Some(&mut *sources.borrow_mut()),
                            &retired,
                            &poison,
                            &cancel,
                            || {
                                audits.set(audits.get() + 1);
                                let mut receipt = completed_audit_receipt();
                                match failure {
                                    1 => return Err(Error::InvalidBufferLength),
                                    2 => panic!("public injected audited cleanup panic"),
                                    3 => receipt.inspected_bytes -= 1,
                                    4 => cancel.store(true, Ordering::Release),
                                    5 => poison.store(true, Ordering::Release),
                                    6 => retired.store(true, Ordering::Release),
                                    _ => {}
                                }
                                Ok(receipt)
                            },
                        )
                    })
                },
                || {
                    sampled.set(true);
                    Err(Error::InvalidBufferLength) // failure7: no fence credit
                },
            );
            assert!(result.is_err());
            assert_eq!(audits.get(), usize::from(failure != 0));
            assert_eq!(sampled.get(), failure == 7);
            assert_eq!(
                sources.borrow().bytes(),
                if failure == 7 { 0 } else { PAGE }
            );
            assert_eq!(
                retired.load(Ordering::Acquire),
                matches!(failure, 1 | 2 | 3 | 6)
            );
            assert!(ledger.is_poisoned());
            assert_eq!(ledger.upper_bytes(), 2 * PAGE);
            // Failed logical owners never retry; only explicit source teardown
            // is modeled here. Cancellation/poison alone keeps the GPU healthy.
            sources
                .borrow_mut()
                .after_completed_read(|| Ok(()))
                .unwrap();
        }
    }

    #[test]
    fn audited_boundary_preserves_typed_failure_and_retires_bookkeeping_panic() {
        let retired = AtomicBool::new(false);
        let result = checked_audited_boundary(&retired, || {
            checked_resident_io_audit(&retired, || Err(Error::InvalidFormat("audit_nonzero")))
        });
        assert!(matches!(result, Err(Error::InvalidFormat("audit_nonzero"))));
        assert!(retired.load(Ordering::Acquire));

        let retired = AtomicBool::new(false);
        assert!(matches!(
            checked_audited_boundary(&retired, || Err(Error::InvalidFormat(
                "GPU execution cancelled"
            ))),
            Err(Error::InvalidFormat("GPU execution cancelled"))
        ));
        assert!(!retired.load(Ordering::Acquire));

        let retired = AtomicBool::new(false);
        assert!(checked_audited_boundary(&retired, || panic!("public bookkeeping panic")).is_err());
        assert!(retired.load(Ordering::Acquire));

        let retired = AtomicBool::new(true);
        assert!(checked_audited_boundary(&retired, || Ok(())).is_err());
        assert!(retired.load(Ordering::Acquire));
    }

    #[test]
    fn cleanup_failure_never_samples_or_credits_a_completed_read() {
        let sampled = Cell::new(false);
        let retired = AtomicBool::new(false);
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        ledger.precharge(&[PAGE]).unwrap();
        let result = guarded_readback(
            &mut ledger,
            || Ok(zeroize::Zeroizing::new(vec![1.0_f32])),
            || checked_transient_cleanup(&retired, failure),
            || {
                sampled.set(true);
                Ok(PAGE)
            },
        );
        assert!(result.is_err());
        assert!(!sampled.get());
        assert!(retired.load(Ordering::Acquire));
        assert!(ledger.is_poisoned());
        assert_eq!(ledger.upper_bytes(), 2 * PAGE);
    }

    #[test]
    fn ordered_source_cleanup_failure_retains_owner_before_main_credit() {
        let mut sources = crate::ordered_public::PendingSources::new();
        sources
            .enqueue(vec![0; PAGE as usize], Ok, |_, _| Ok(()))
            .unwrap();
        let sampled = Cell::new(false);
        let retired = AtomicBool::new(false);
        let mut main = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        main.precharge(&[PAGE]).unwrap();
        assert!(
            guarded_readback(
                &mut main,
                || Ok(()),
                || checked_transient_cleanup(&retired, || sources.after_completed_read(failure)),
                || {
                    sampled.set(true);
                    Ok(PAGE)
                },
            )
            .is_err()
        );
        assert_eq!(sources.bytes(), PAGE);
        assert!(!sampled.get());
        assert!(retired.load(Ordering::Acquire));
        assert!(main.is_poisoned());
        assert_eq!(main.upper_bytes(), 2 * PAGE);
        // Only explicit teardown may try again. No ledger reuse/clean receipt.
        sources.after_completed_read(|| Ok(())).unwrap();
    }

    #[test]
    fn cleanup_panic_retires_and_later_best_effort_cannot_claim_clean() {
        let stream = Arc::new(AtomicBool::new(false));
        let retired = stream.clone();
        assert!(checked_transient_cleanup(&retired, || panic!("public fault fixture")).is_err());
        assert!(retired.load(Ordering::Acquire));
        let fresh_owner = stream.clone();
        let retried = Cell::new(false);
        assert!(
            checked_transient_cleanup(&fresh_owner, || {
                retried.set(true);
                Ok(())
            })
            .is_err()
        );
        assert!(retried.get());
        assert!(retired.load(Ordering::Acquire));
    }

    #[test]
    fn sequence_failure_rejects_before_mutation_and_keeps_partial_source_debt() {
        let poison = AtomicBool::new(true);
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        let entered = Cell::new(false);
        let result = checked_sequence_guard(&mut ledger, &poison, |_| {
            entered.set(true);
            Ok(())
        });
        assert!(result.is_err());
        assert!(!entered.get());
        assert!(ledger.is_poisoned());
        assert_eq!(ledger.upper_bytes(), PAGE);

        let healthy = AtomicBool::new(false);
        let mut partial = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        let result: Result<()> = checked_sequence_guard(&mut partial, &healthy, |ledger| {
            ledger.precharge(&[PAGE]).map_err(admission_error)?;
            failure()
        });
        assert!(result.is_err());
        assert!(partial.is_poisoned());
        assert_eq!(partial.upper_bytes(), 2 * PAGE);
        assert!(partial.completed_fence(0).is_err());
    }

    #[test]
    fn sticky_drop_inside_held_admission_guard_does_not_reenter_mutex() {
        struct StickyDrop(Arc<AtomicBool>);
        impl Drop for StickyDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let poison = Arc::new(AtomicBool::new(false));
        let physical_retired = AtomicBool::new(false);
        let ledger = Mutex::new(MainAdmission::new(4 * PAGE, PAGE).unwrap());
        let mut held = ledger.lock().unwrap();
        let pending = StickyDrop(poison.clone());
        let result = checked_sequence_guard(&mut held, &poison, |_| {
            assert!(ledger.try_lock().is_err());
            drop(pending);
            Ok(())
        });
        assert!(result.is_err());
        assert!(held.is_poisoned());
        assert!(poison.load(Ordering::Acquire));
        assert!(!physical_retired.load(Ordering::Acquire));
    }

    #[test]
    fn healthy_sequence_guard_accepts_but_unwound_owner_blocks_later_reuse() {
        struct StickyDrop(Arc<AtomicBool>);
        impl Drop for StickyDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let getter: fn(&ResidentAdmission) -> Arc<AtomicBool> = ResidentAdmission::sequence_poison;
        let _ = getter;
        let poison = Arc::new(AtomicBool::new(false));
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        assert_eq!(
            checked_sequence_guard(&mut ledger, &poison, |_| Ok(7)).unwrap(),
            7
        );
        assert!(!ledger.is_poisoned());
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _pending = StickyDrop(poison.clone());
            let _: Result<()> = checked_sequence_guard(&mut ledger, &poison, |_| {
                panic!("public logical-owner unwind fixture")
            });
        }));
        assert!(unwind.is_err());
        assert!(poison.load(Ordering::Acquire));
        assert!(checked_sequence_guard(&mut ledger, &poison, |_| Ok(())).is_err());
        assert!(ledger.is_poisoned());
    }

    #[test]
    fn allocation_unwind_poisoned_mutex_blocks_a_following_cached_operation() {
        // ResidentAdmission holds this same standard mutex through allocation.
        // An allocation panic cannot reach the next active/cache-hit guard.
        let ledger = Mutex::new(MainAdmission::new(4 * PAGE, PAGE).unwrap());
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut held = ledger.lock().unwrap();
            let _: Result<()> = guarded_allocation(
                &mut held,
                &[PAGE],
                || Ok(PAGE),
                || panic!("public partial allocation fixture"),
            );
        }));
        assert!(attempted.is_err());
        assert!(ledger.lock().is_err());
    }

    #[test]
    fn allocation_precharges_before_create_and_checks_before_publish() {
        let events = RefCell::new(Vec::new());
        let calls = Cell::new(0);
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        let value = guarded_allocation(
            &mut ledger,
            &[4],
            || {
                let call = calls.get();
                calls.set(call + 1);
                events
                    .borrow_mut()
                    .push(if call == 0 { "before" } else { "after" });
                Ok(if call == 0 { PAGE } else { 2 * PAGE })
            },
            || {
                events.borrow_mut().push("create");
                Ok(17)
            },
        )
        .unwrap();
        events.borrow_mut().push("publish");
        assert_eq!(value, 17);
        assert_eq!(*events.borrow(), ["before", "create", "after", "publish"]);
        assert_eq!(ledger.upper_bytes(), 2 * PAGE);
        assert!(!ledger.is_poisoned());
    }

    #[test]
    fn six_segment_pages_are_precharged_together_and_partial_creation_poisons_owner() {
        const MIB: u64 = 1024 * 1024;
        // Actual gate/up Q4_K and down Q8_0 original 64+32 logical extents.
        let requests = [
            921600 * 64,
            921600 * 32,
            921600 * 64,
            921600 * 32,
            1740800 * 64,
            1740800 * 32,
        ];
        let charge = requests
            .iter()
            .map(|&n| main_admission::page_bytes(n).unwrap())
            .sum::<u64>();
        assert_eq!(charge, 384 * MIB);
        let created = Cell::new(0);
        let mut short = MainAdmission::new(PAGE + charge - 1, PAGE).unwrap();
        assert!(
            guarded_allocation(
                &mut short,
                &requests,
                || Ok(PAGE),
                || {
                    created.set(6);
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(created.get(), 0);
        assert!(short.is_poisoned());
        let mut exact = MainAdmission::new(PAGE + charge, PAGE).unwrap();
        let result = guarded_allocation(
            &mut exact,
            &requests,
            || Ok(PAGE),
            || {
                // Failure after the first physical shard: the enclosing ledger
                // remains poisoned, retains debt and cannot publish/accept retry.
                created.set(1);
                failure::<()>()
            },
        );
        assert!(result.is_err());
        assert_eq!(created.get(), 1);
        assert_eq!(exact.upper_bytes(), PAGE + charge);
        assert!(exact.is_poisoned());
        assert!(exact.completed_fence(0).is_err());
        assert!(guarded_allocation(&mut exact, &requests, || Ok(0), || Ok(())).is_err());
    }

    #[test]
    fn failed_precharge_or_pretelemetry_never_allocates_and_poison_is_sticky() {
        for bad_telemetry in [false, true] {
            let allocated = Cell::new(false);
            let mut ledger = MainAdmission::new(PAGE, PAGE).unwrap();
            let result = guarded_allocation(
                &mut ledger,
                &[4],
                || if bad_telemetry { failure() } else { Ok(PAGE) },
                || {
                    allocated.set(true);
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert!(!allocated.get());
            assert!(ledger.is_poisoned());
            assert!(ledger.completed_fence(0).is_err());
            assert_eq!(ledger.upper_bytes(), PAGE);
        }
    }

    #[test]
    fn failed_create_or_postguard_cannot_publish_an_allocation() {
        for failure_at in ["create", "telemetry", "growth"] {
            let calls = Cell::new(0);
            let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
            let result = guarded_allocation(
                &mut ledger,
                &[4],
                || {
                    let call = calls.get();
                    calls.set(call + 1);
                    if call == 0 {
                        Ok(PAGE)
                    } else if failure_at == "telemetry" {
                        failure()
                    } else {
                        Ok(3 * PAGE)
                    }
                },
                || {
                    if failure_at == "create" {
                        failure()
                    } else {
                        Ok(17)
                    }
                },
            );
            assert!(result.is_err());
            assert!(ledger.is_poisoned());
            assert_eq!(ledger.upper_bytes(), 2 * PAGE);
            assert_eq!(calls.get(), if failure_at == "create" { 1 } else { 2 });
        }
    }

    #[test]
    fn only_completed_readback_can_reclaim_and_reclamation_precedes_commit() {
        let events = RefCell::new(Vec::new());
        let mut ledger = MainAdmission::new(8 * PAGE, PAGE).unwrap();
        ledger.precharge(&[PAGE, PAGE, PAGE]).unwrap();
        ledger.observe_after(2 * PAGE).unwrap();
        // A submission/telemetry observation never credits dropped handles.
        assert_eq!(ledger.upper_bytes(), 4 * PAGE);
        let packet = guarded_readback(
            &mut ledger,
            || {
                events.borrow_mut().push("completed read");
                Ok(31)
            },
            || Ok(()),
            || {
                events.borrow_mut().push("physical sample");
                Ok(2 * PAGE)
            },
        )
        .unwrap();
        assert_eq!(ledger.upper_bytes(), 2 * PAGE);
        events.borrow_mut().push("commit");
        assert_eq!(packet, 31);
        assert_eq!(
            *events.borrow(),
            ["completed read", "physical sample", "commit"]
        );
        // Two pages remain physically alive, so a read does not refund them.
        assert_eq!(ledger.upper_bytes(), 2 * PAGE);
    }

    #[test]
    fn failed_fence_or_telemetry_never_reclaims_or_accepts() {
        for read_failure in [true, false] {
            let sampled = Cell::new(false);
            let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
            ledger.precharge(&[PAGE]).unwrap();
            let result = guarded_readback(
                &mut ledger,
                || if read_failure { failure() } else { Ok(31) },
                || Ok(()),
                || {
                    sampled.set(true);
                    failure()
                },
            );
            assert!(result.is_err());
            assert_eq!(sampled.get(), !read_failure);
            assert_eq!(ledger.upper_bytes(), 2 * PAGE);
            assert!(ledger.is_poisoned());
        }
        let read = Cell::new(false);
        let mut ledger = MainAdmission::new(4 * PAGE, PAGE).unwrap();
        ledger.poison();
        assert!(
            guarded_readback(
                &mut ledger,
                || {
                    read.set(true);
                    Ok(31)
                },
                || Ok(()),
                || Ok(0)
            )
            .is_err()
        );
        assert!(!read.get());
    }

    #[test]
    fn exact_legacy_and_batched_fill_pages_count_every_source() {
        for down in [1_228_800_u64, 1_740_800] {
            let parts = [921_600, 921_600, down];
            for misses in 0..=10 {
                let legacy = fill_requests(parts, misses, false, 8 * 1024 * 1024).unwrap();
                assert_eq!(legacy.len(), 3 * misses);
                let legacy_pages = legacy
                    .iter()
                    .map(|&n| main_admission::page_bytes(n).unwrap())
                    .sum::<u64>();
                assert_eq!(legacy_pages, misses as u64 * 4 * 1024 * 1024);
                for window_mib in [4_usize, 8, 16, 64] {
                    let window = window_mib * 1024 * 1024;
                    let requests = fill_requests(parts, misses, true, window).unwrap();
                    let bundle = parts.iter().sum::<u64>();
                    let count = ((window as u64).min(65_535 * 256 * 4) / bundle).min(10);
                    assert_eq!(requests.len(), (misses as u64).div_ceil(count) as usize);
                    assert_eq!(requests.iter().sum::<u64>(), misses as u64 * bundle);
                    for &n in &requests {
                        assert!(n <= window as u64 && n <= 65_535 * 256 * 4);
                    }
                    let mut ledger = MainAdmission::new(128 * 1024 * 1024, 0).unwrap();
                    assert_eq!(
                        ledger.precharge(&requests).unwrap(),
                        requests
                            .iter()
                            .map(|&n| main_admission::page_bytes(n).unwrap())
                            .sum::<u64>()
                    );
                }
            }
        }
        // A ten-rank miss plan needs five independently rounded 8 MiB sources.
        assert_eq!(
            fill_requests([921_600, 921_600, 1_740_800], 10, true, 8 * 1024 * 1024)
                .unwrap()
                .len(),
            5
        );
        assert!(fill_requests([4, 4, 4], 11, false, 8 * 1024 * 1024).is_err());
        assert!(fill_requests([0, 4, 4], 1, false, 8 * 1024 * 1024).is_err());
        assert!(fill_requests([u64::MAX - 3, 4, 4], 1, true, 8 * 1024 * 1024).is_err());
        assert!(fill_requests([4, 4, 4], 1, true, 3 * 1024 * 1024).is_err());
    }
}

#[cfg(test)]
mod target_verify3_tests {
    use super::*;

    #[test]
    fn iq4_decoder_activation_is_one_way_and_requires_actual_original_public_scope() {
        use crate::next_shared::Iq4Decoder as D;
        assert_eq!(
            activate_target_iq4_decoder(D::Original, true).unwrap(),
            D::PackedRegisters
        );
        assert!(activate_target_iq4_decoder(D::Original, false).is_err());
        for scope in [false, true] {
            assert!(activate_target_iq4_decoder(D::PackedRegisters, scope).is_err());
        }
    }

    #[test]
    fn actual_verify3_spec_selects_only_original_iq4_types_with_distinct_immutable_keys() {
        use crate::next_shared::{Iq4Decoder as D, MatrixSpec};
        for dtype in [0, 1, 6, 8, 12, 13, 14, 20, 23, 30] {
            let original = MatrixSpec::new(7, 256, dtype).unwrap();
            assert_eq!(
                target_verify3_iq4_spec(7, 256, dtype, D::Original).unwrap(),
                original
            );
            let candidate = target_verify3_iq4_spec(7, 256, dtype, D::PackedRegisters).unwrap();
            if matches!(dtype, 20 | 23) {
                assert_ne!(candidate, original);
                assert_eq!(candidate.iq4_decoder(), D::PackedRegisters);
                let hash = |spec: MatrixSpec| {
                    use std::hash::{Hash, Hasher};
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    spec.hash(&mut hash);
                    hash.finish()
                };
                assert_ne!(hash(original), hash(candidate));
            } else {
                assert_eq!(candidate, original);
                assert_eq!(candidate.iq4_decoder(), D::Original);
            }
        }
        assert!(target_verify3_iq4_spec(7, 257, 20, D::PackedRegisters).is_err());
        assert!(target_verify3_iq4_spec(7, 256, 999, D::PackedRegisters).is_err());
    }

    #[test]
    fn iq4_counter_credits_only_actual_candidate_dispatch_and_rejects_overflow() {
        use crate::next_shared::Iq4Decoder as D;
        assert_eq!(next_target_iq4_dispatch_count(0, D::Original).unwrap(), 0);
        assert_eq!(next_target_iq4_dispatch_count(7, D::Original).unwrap(), 7);
        assert_eq!(
            next_target_iq4_dispatch_count(0, D::PackedRegisters).unwrap(),
            1
        );
        assert_eq!(
            next_target_iq4_dispatch_count(7, D::PackedRegisters).unwrap(),
            8
        );
        assert_eq!(
            next_target_iq4_dispatch_count(u64::MAX, D::Original).unwrap(),
            u64::MAX
        );
        assert!(next_target_iq4_dispatch_count(u64::MAX, D::PackedRegisters).is_err());
    }

    #[test]
    fn verify3_is_fixed_at_construction_and_requires_original_target_batch_policy() {
        for frozen in [None, Some(true)] {
            validate_target_verify3_policy(true, frozen, true, true, 8, 8).unwrap();
        }
        for (selected, frozen, rows, shared, members, tile) in [
            (true, Some(false), true, true, 8, 8),
            (false, Some(true), true, true, 8, 8),
            (true, None, false, true, 8, 8),
            (true, None, true, false, 8, 8),
            (true, None, true, true, 1, 8),
            (true, None, true, true, 8, 1),
            (true, None, true, true, 8, 4),
        ] {
            assert!(
                validate_target_verify3_policy(selected, frozen, rows, shared, members, tile)
                    .is_err()
            );
        }
        // Off preserves every original tile/member/target/control choice.
        for rows in [false, true] {
            for shared in [false, true] {
                for (members, tile) in [(1, 1), (8, 1), (8, 4), (8, 8)] {
                    for frozen in [None, Some(false)] {
                        validate_target_verify3_policy(false, frozen, rows, shared, members, tile)
                            .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn complete_verify3_finite_mode_is_separately_frozen_before_mutation() {
        for frozen in [None, Some(true)] {
            validate_target_verify3_fused_policy(true, frozen, true, true, 8, 8).unwrap();
        }
        for (selected, frozen, verify3, rows, members, tile) in [
            (true, Some(false), true, true, 8, 8),
            (false, Some(true), true, true, 8, 8),
            (true, None, false, true, 8, 8),
            (true, None, true, false, 8, 8),
            (true, None, true, true, 1, 8),
            (true, None, true, true, 8, 1),
            (true, None, true, true, 8, 4),
        ] {
            assert!(
                validate_target_verify3_fused_policy(
                    selected, frozen, verify3, rows, members, tile
                )
                .is_err()
            );
        }
        for verify3 in [false, true] {
            for rows in [false, true] {
                for (members, tile) in [(1, 1), (8, 1), (8, 4), (8, 8)] {
                    for frozen in [None, Some(false)] {
                        validate_target_verify3_fused_policy(
                            false, frozen, verify3, rows, members, tile,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod q6_final_head_tests {
    use super::*;
    use quant_matvec::matmul::BatchSpans;

    #[test]
    fn only_admitted_low_bit_count1_original_head_selects_integer_arithmetic() {
        let spans = BatchSpans::new(1, 0, 0, 2560, 0, 248320);
        let check = |profile,
                     admitted,
                     members,
                     tile,
                     name,
                     dtype,
                     columns,
                     rows,
                     experts,
                     packet: &BatchSpans| {
            validate_q6_final_head(
                profile, admitted, members, tile, name, dtype, columns, rows, experts, packet,
            )
        };
        for profile in [
            expert_banks::OriginalPackedProfile::UnslothQ3,
            expert_banks::OriginalPackedProfile::DaslabIq3,
        ] {
            check(
                profile,
                true,
                1,
                1,
                "output.weight",
                14,
                2560,
                248320,
                1,
                &spans,
            )
            .unwrap();
        }
        let profile = expert_banks::OriginalPackedProfile::DaslabIq3;
        for (p, admitted, members, tile, name, dtype, columns, rows, experts) in [
            (
                expert_banks::OriginalPackedProfile::Q4,
                true,
                1,
                1,
                "output.weight",
                14,
                2560,
                248320,
                1,
            ),
            (profile, false, 1, 1, "output.weight", 14, 2560, 248320, 1),
            (profile, true, 8, 1, "output.weight", 14, 2560, 248320, 1),
            (profile, true, 1, 8, "output.weight", 14, 2560, 248320, 1),
            (
                profile,
                true,
                1,
                1,
                "blk.0.attn_output.weight",
                14,
                2560,
                248320,
                1,
            ),
            (profile, true, 1, 1, "output.weight", 8, 2560, 248320, 1),
            (profile, true, 1, 1, "output.weight", 14, 256, 248320, 1),
            (profile, true, 1, 1, "output.weight", 14, 2560, 16384, 1),
            (profile, true, 1, 1, "output.weight", 14, 2560, 248320, 512),
        ] {
            assert!(
                check(
                    p, admitted, members, tile, name, dtype, columns, rows, experts, &spans
                )
                .is_err()
            );
        }
        for packet in [
            BatchSpans::new(2, 0, 0, 2560, 0, 248320),
            BatchSpans::new(1, 4, 0, 2560, 0, 248320),
            BatchSpans::new(1, 0, 4, 2560, 0, 248320),
            BatchSpans::new(1, 0, 0, 2564, 0, 248320),
            BatchSpans::new(1, 0, 0, 2560, 4, 248320),
            BatchSpans::new(1, 0, 0, 2560, 0, 248324),
        ] {
            assert!(
                check(
                    profile,
                    true,
                    1,
                    1,
                    "output.weight",
                    14,
                    2560,
                    248320,
                    1,
                    &packet
                )
                .is_err()
            );
        }
    }
    #[test]
    fn original_head_tiles_cover_every_row_once_with_one_plane_preparation() {
        use crate::q6_integer_matvec::{PrepareSpans, ProjectSpans, Spec};
        let mut expected_first = 0;
        let mut tiles = 0;
        let full_words = [32768usize, 131072, 262144, 8192];
        let prepare = PrepareSpans {
            input_first: 0,
            planes_first: 0,
            fault_first: 0,
        };
        prepare
            .validate(
                Spec::new(16384, 2560).unwrap(),
                [full_words[0], full_words[1], full_words[3]],
            )
            .unwrap();
        let mut packed_total = 0;
        let mut page_total = 0;
        for first in (0..248320).step_by(16384) {
            assert_eq!(first, expected_first);
            let rows = 16384.min(248320 - first);
            let spec = Spec::new(rows, 2560).unwrap();
            let packet = ProjectSpans {
                packed_first: 0,
                planes_first: 0,
                output_first: first,
                fault_first: 0,
                matrix_first: 0,
                matrix_rows: 248320,
            };
            packet
                .validate(
                    spec,
                    spec.weight_bytes(),
                    [full_words[1], full_words[2], full_words[3]],
                )
                .unwrap();
            assert!(
                packet
                    .validate(
                        spec,
                        spec.weight_bytes() - 4,
                        [full_words[1], full_words[2], full_words[3]]
                    )
                    .is_err()
            );
            expected_first += rows;
            packed_total += spec.weight_bytes();
            page_total += main_admission::page_bytes(spec.weight_bytes() as u64).unwrap();
            tiles += 1;
        }
        assert_eq!(expected_first, 248320);
        assert_eq!(tiles, 16);
        assert_eq!(packed_total, 521_472_000);
        assert_eq!(page_total, 1_015_021_568);
    }
}

#[cfg(test)]
mod construction_lock_tests {
    use super::*;
    #[test]
    fn checked_constructor_origin_completes_while_main_lock_is_held() {
        let ledger = Arc::new(Mutex::new(()));
        let retired = Arc::new(AtomicBool::new(false));
        let poison = Arc::new(AtomicBool::new(false));
        let held = ledger.lock().unwrap();
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            let (ledger_ref, retired_ref, poison_ref) = (&ledger, &retired, &poison);
            scope.spawn(move || {
                tx.send(
                    checked_construction_origin(
                        ledger_ref,
                        ledger_ref,
                        retired_ref,
                        retired_ref,
                        poison_ref,
                        poison_ref,
                    )
                    .is_ok(),
                )
                .unwrap();
            });
            let completed = rx.recv_timeout(std::time::Duration::from_secs(1));
            drop(held); // Even a regressed callback can exit before assertion.
            assert!(completed.unwrap());
        });
    }
    #[test]
    fn construction_proof_rejects_foreign_ledger_stream_sequence_and_poisoned_flags() {
        let ledger = Arc::new(Mutex::new(()));
        let foreign_ledger = Arc::new(Mutex::new(()));
        let retired = Arc::new(AtomicBool::new(false));
        let poison = Arc::new(AtomicBool::new(false));
        let foreign = Arc::new(AtomicBool::new(false));
        assert!(
            checked_construction_origin(
                &ledger,
                &foreign_ledger,
                &retired,
                &retired,
                &poison,
                &poison
            )
            .is_err()
        );
        assert!(
            checked_construction_origin(&ledger, &ledger, &retired, &foreign, &poison, &poison)
                .is_err()
        );
        assert!(
            checked_construction_origin(&ledger, &ledger, &retired, &retired, &poison, &foreign)
                .is_err()
        );
        retired.store(true, Ordering::Release);
        assert!(
            checked_construction_origin(&ledger, &ledger, &retired, &retired, &poison, &poison)
                .is_err()
        );
        retired.store(false, Ordering::Release);
        poison.store(true, Ordering::Release);
        assert!(
            checked_construction_origin(&ledger, &ledger, &retired, &retired, &poison, &poison)
                .is_err()
        );
    }
}

#[cfg(test)]
fn target_rows_private_extent(
    private: u64,
    batch: bool,
    frozen: bool,
    stage: bool,
    gu_integer: bool,
) -> bool {
    target_rows_private_extent_with_down_math(private, batch, frozen, stage, gu_integer, false)
}
fn target_rows_private_extent_with_down_math(
    private: u64,
    batch: bool,
    frozen: bool,
    stage: bool,
    gu_integer: bool,
    down_integer: bool,
) -> bool {
    if (stage || gu_integer || down_integer) && (!batch || !frozen)
        || stage && (gu_integer || down_integer)
    {
        return false;
    }
    if batch {
        private
            == crate::resident_model::batch::TARGET_ROWS_PRIVATE_MAIN
                + batch_moe::batch_router::MAIN_BYTES
                + if stage {
                    batch_moe::FROZEN_STAGING_MAIN_BYTES
                } else {
                    0
                }
                + if gu_integer {
                    batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES
                } else {
                    0
                }
                + if down_integer {
                    batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES
                } else {
                    0
                }
    } else {
        [
            crate::resident_model::batch::TARGET_ROWS_PRIVATE_MAIN,
            crate::resident_model::batch::TARGET_ROWS_PREPARED_VERIFY3_MAIN,
        ]
        .contains(&private)
    }
}

#[cfg(test)]
mod frozen_host_policy_tests {
    #[test]
    fn frozen_down_integer_constructor_requires_each_exact_independent_scratch_charge() {
        let base = crate::resident_model::batch::TARGET_ROWS_PRIVATE_MAIN
            + batch_moe::batch_router::MAIN_BYTES;
        for gu in [false, true] {
            for down in [false, true] {
                let exact = base
                    + u64::from(gu) * batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES
                    + u64::from(down) * batch_moe::FROZEN_DOWN_INTEGER_MAIN_BYTES;
                assert!(target_rows_private_extent_with_down_math(
                    exact, true, true, false, gu, down
                ));
                for n in [exact - 1, exact + 1, exact + 65536] {
                    assert!(!target_rows_private_extent_with_down_math(
                        n, true, true, false, gu, down
                    ));
                }
                if down {
                    assert!(!target_rows_private_extent_with_down_math(
                        exact, false, true, false, gu, down
                    ));
                    assert!(!target_rows_private_extent_with_down_math(
                        exact, true, false, false, gu, down
                    ));
                    assert!(!target_rows_private_extent_with_down_math(
                        exact, true, true, true, gu, down
                    ));
                    assert!(!target_rows_private_extent_with_down_math(
                        exact, true, true, false, gu, false
                    ));
                }
            }
        }
    }

    use super::*;
    #[test]
    fn packed_staging_accepts_only_exact_precharged_frozen_router_extent() {
        let plain = crate::resident_model::batch::TARGET_ROWS_PRIVATE_MAIN;
        let routed = plain + batch_moe::batch_router::MAIN_BYTES;
        let staged = routed + batch_moe::FROZEN_STAGING_MAIN_BYTES;
        assert!(target_rows_private_extent(
            plain, false, false, false, false
        ));
        assert!(target_rows_private_extent(
            routed, true, false, false, false
        ));
        assert!(target_rows_private_extent(routed, true, true, false, false));
        assert!(target_rows_private_extent(staged, true, true, true, false));
        for bytes in [plain, routed, staged - 1, staged + 1] {
            assert!(!target_rows_private_extent(bytes, true, true, true, false));
        }
        assert!(!target_rows_private_extent(
            staged, true, true, false, false
        ));
        for (batch, frozen) in [(false, false), (false, true), (true, false)] {
            assert!(!target_rows_private_extent(
                staged, batch, frozen, true, false
            ));
        }
    }
    #[test]
    fn frozen_gu_integer_accepts_only_exact_precharged_frozen_router_extent() {
        let routed = crate::resident_model::batch::TARGET_ROWS_PRIVATE_MAIN
            + batch_moe::batch_router::MAIN_BYTES;
        let exact = routed + batch_moe::FROZEN_GU_INTEGER_MAIN_BYTES;
        assert!(target_rows_private_extent(exact, true, true, false, true));
        for bytes in [routed, exact - 1, exact + 1] {
            assert!(!target_rows_private_extent(bytes, true, true, false, true));
        }
        assert!(!target_rows_private_extent(exact, true, true, false, false));
        for (batch, frozen, stage) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert!(!target_rows_private_extent(
                exact, batch, frozen, stage, true
            ));
        }
    }
    #[test]
    fn target_freeze_requires_matching_authenticated_source_domain() {
        let p = expert_banks::OriginalPackedProfile::DaslabIq3;
        validate_target_frozen_host_policy(true, true, true, p, 144, false, false).unwrap();
        validate_target_frozen_host_policy(false, false, true, p, 144, false, false).unwrap();
        validate_target_frozen_host_policy(true, false, false, p, 144, false, false).unwrap();
        for other in [
            expert_banks::OriginalPackedProfile::Q4,
            expert_banks::OriginalPackedProfile::UnslothQ3,
        ] {
            assert!(
                validate_target_frozen_host_policy(true, true, true, other, 144, false, false)
                    .is_err()
            );
        }

        validate_target_frozen_host_policy(true, true, true, p, 144, true, true).unwrap();
        validate_target_frozen_host_policy(false, false, true, p, 144, true, true).unwrap();
        assert!(validate_target_frozen_host_policy(true, true, true, p, 144, false, true).is_err());
        assert!(
            validate_target_frozen_host_policy(false, false, false, p, 144, true, true).is_err()
        );
        assert!(validate_target_frozen_host_policy(false, false, true, p, 96, true, true).is_err());
        for (frozen, identity, rows, capacity, router) in [
            (true, false, true, 144, false),
            (false, true, true, 144, false),
            (true, true, false, 144, false),
            (true, true, true, 96, false),
            (true, true, true, 144, true),
        ] {
            assert!(
                validate_target_frozen_host_policy(
                    frozen, identity, rows, p, capacity, router, false
                )
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod q6_verify_head_tests {
    use super::*;
    use quant_matvec::matmul::BatchSpans;

    #[test]
    fn q6_verifier_requires_actual_max8_policy_not_active_row_capacity() {
        let profile = expert_banks::OriginalPackedProfile::DaslabIq3;
        for count in 1..=4 {
            let spans = BatchSpans::new(count, 0, 0, 2560, 0, 248320);
            for four_reserved in [false, true] {
                for four_active in [false, true] {
                    let expected = (!four_active || four_reserved) && (count <= 3 || four_active);
                    let result = validate_q6_verify_head(
                        profile,
                        true,
                        8,
                        8,
                        four_reserved,
                        four_active,
                        "output.weight",
                        14,
                        2560,
                        248320,
                        1,
                        &spans,
                    );
                    assert_eq!(result.is_ok(), expected);
                }
            }
            for (members, tile) in [(1, 1), (3, 3), (4, 4), (4, 8), (8, 4)] {
                assert!(
                    validate_q6_verify_head(
                        profile,
                        true,
                        members,
                        tile,
                        true,
                        true,
                        "output.weight",
                        14,
                        2560,
                        248320,
                        1,
                        &spans
                    )
                    .is_err()
                );
            }
        }
    }
    #[test]
    fn q6_verifier_rejects_wrong_original_artifact_and_views() {
        use expert_banks::OriginalPackedProfile as P;
        let spans = BatchSpans::new(3, 0, 0, 2560, 0, 248320);
        for (profile, admitted, name, dtype, columns, rows, experts) in [
            (P::Q4, true, "output.weight", 14, 2560, 248320, 1),
            (P::UnslothQ3, true, "output.weight", 14, 2560, 248320, 1),
            (P::DaslabIq3, false, "output.weight", 14, 2560, 248320, 1),
            (P::DaslabIq3, true, "token_embd.weight", 14, 2560, 248320, 1),
            (P::DaslabIq3, true, "output.weight", 12, 2560, 248320, 1),
            (P::DaslabIq3, true, "output.weight", 14, 256, 248320, 1),
            (P::DaslabIq3, true, "output.weight", 14, 2560, 16384, 1),
            (P::DaslabIq3, true, "output.weight", 14, 2560, 248320, 512),
        ] {
            assert!(
                validate_q6_verify_head(
                    profile, admitted, 8, 8, true, true, name, dtype, columns, rows, experts,
                    &spans
                )
                .is_err()
            );
        }
        for wrong in [
            BatchSpans::new(0, 0, 0, 2560, 0, 248320),
            BatchSpans::new(5, 0, 0, 2560, 0, 248320),
            BatchSpans::new(3, 1, 0, 2560, 0, 248320),
            BatchSpans::new(3, 0, 1, 2560, 0, 248320),
            BatchSpans::new(3, 0, 0, 2561, 0, 248320),
            BatchSpans::new(3, 0, 0, 2560, 1, 248320),
            BatchSpans::new(3, 0, 0, 2560, 0, 248321),
        ] {
            assert!(
                validate_q6_verify_head(
                    P::DaslabIq3,
                    true,
                    8,
                    8,
                    true,
                    true,
                    "output.weight",
                    14,
                    2560,
                    248320,
                    1,
                    &wrong
                )
                .is_err()
            );
        }
    }
    #[test]
    fn q6_verifier_immutable_activation_and_actual_completed_counters() {
        assert!(activate_q6_verify_head(false, true).unwrap());
        for args in [(false, false), (true, false), (true, true)] {
            assert!(activate_q6_verify_head(args.0, args.1).is_err());
        }
        validate_q6_verify_counters(false, 0, 0, 0).unwrap();
        validate_q6_verify_counters(true, 0, 0, 0).unwrap();
        for members in 3..=12 {
            validate_q6_verify_counters(true, 3, 48, members).unwrap();
        }
        for (selected, prepares, tiles, members) in [
            (false, 1, 16, 1),
            (true, 1, 0, 0),
            (true, 1, 15, 1),
            (true, 1, 17, 1),
            (true, 3, 48, 2),
            (true, 3, 48, 13),
            (true, u64::MAX, 0, 0),
        ] {
            assert!(validate_q6_verify_counters(selected, prepares, tiles, members).is_err());
        }
    }
    #[test]
    fn q6_verifier_preflights_all_original_tiles_for_every_real_member() {
        use crate::q6_integer_matvec::{
            Spec,
            batch::{PrepareSpans, ProjectSpans},
        };
        let prepare_spec = Spec::new(16384, 2560).unwrap();
        for count in 1..=4 {
            let output_words = if count == 4 { 993280 } else { 744960 };
            let prepare = PrepareSpans {
                count,
                input_first: 0,
                input_stride: 2560,
                planes_first: 0,
                planes_stride: 1440,
                fault_first: 0,
            };
            prepare
                .validate(prepare_spec, [20480, 81920, 8192])
                .unwrap();
            let mut total_rows = 0;
            let mut total_bytes = 0;
            let mut tiles = 0;
            for first in (0..248320).step_by(16384) {
                let rows = 16384.min(248320 - first);
                let spec = Spec::new(rows, 2560).unwrap();
                let packet = ProjectSpans {
                    count,
                    packed_first: 0,
                    planes_first: 0,
                    planes_stride: 1440,
                    output_first: first,
                    output_stride: 248320,
                    fault_first: 0,
                    matrix_first: 0,
                    matrix_rows: 248320,
                };
                packet
                    .validate(spec, spec.weight_bytes(), [81920, output_words, 8192])
                    .unwrap();
                assert!(
                    packet
                        .validate(spec, spec.weight_bytes() - 1, [81920, output_words, 8192])
                        .is_err()
                );
                for member in 0..count {
                    assert!(member * 248320 + first + rows <= output_words);
                }
                total_rows += rows;
                total_bytes += spec.weight_bytes();
                tiles += 1;
            }
            assert_eq!((tiles, total_rows, total_bytes), (16, 248320, 521472000));
            assert!(
                prepare
                    .validate(prepare_spec, [count * 2560 - 1, 81920, 8192])
                    .is_err()
            );
            assert!(
                prepare
                    .validate(prepare_spec, [20480, count * 1440 - 1, 8192])
                    .is_err()
            );
        }
    }
}
