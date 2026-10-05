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
use kuno_text::cpu::Matrix;
use kuno_text::model::{MatrixExecutor, RoutedWeights};
use kuno_text::{Error, GgufFile, Result, TensorView};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use zeroize::Zeroize;

const TILE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const RESIDENT_IO_MAX: u64 = 64 * 1024 * 1024;
/// Separate runtime upload/uniform/readback ownership. Not main-pool telemetry.
pub(crate) const RESIDENT_RUNTIME_IO_MAX: u64 = 64 * 1024 * 1024;

fn cleanup_failure_category(code: &str) -> Option<&'static str> {
    // Forward fixed infrastructure categories only, never backend descriptions,
    // shader source, metadata values or request content.
    const CODES: [&str; 28] = [
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

fn checkpoint_needed(intermediate: bool, token_checkpoint: bool, ordered_router: bool) -> bool {
    ordered_router || !intermediate || !token_checkpoint
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
}
impl ResidentAdmission {
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
        if self.retired.load(Ordering::Acquire) {
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
        self.with_ledger(|ledger| guarded_allocation(ledger, requests, || self.telemetry(), f))
    }
    pub(crate) fn readback<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.readback_with_cadence(false, false, f)
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
            guarded_readback(
                ledger,
                f,
                || {
                    if checkpoint_needed(intermediate, self.token_io_checkpoint, ordered_router) {
                        self.checkpoint()
                    } else {
                        Ok(())
                    }
                },
                || self.telemetry(),
            )
        })
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
    model: &'a GgufFile,
    pub block_kernel: bool,
    pub block_large_only: bool,
    pub block_moe: bool,
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
impl<'a> GpuExecutor<'a> {
    pub fn new(
        model: &'a GgufFile,
        client: ComputeClient<WgpuRuntime>,
        budget: usize,
        bank_slots: usize,
        reserved_limit: usize,
    ) -> Result<Self> {
        expert_banks::validate_capacity(bank_slots)?;
        let explicit_static = match std::env::var("KUNO_LAB_STATIC_CACHE_BYTES") {
            Ok(raw) => Some(expert_banks::configured_static_cache(Some(&raw), budget)?),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err(Error::InvalidFormat("laboratory static cache environment")),
        };
        expert_banks::validate_segmented_configuration(
            bank_slots,
            explicit_static,
            budget,
            std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1"),
            std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
            std::env::var("KUNO_LAB_WORD_DOWN").as_deref() == Ok("1"),
            true,
        )?;
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
        let retired = self
            .transient_resident
            .clone()
            .ok_or(Error::InvalidFormat("resident I/O ownership unavailable"))?;
        self.poison_resident();
        checked_transient_cleanup(&retired, || {
            if let Some(sources) = &self.ordered_sources {
                release_ordered_sources(&self.client, sources)
            } else {
                sanitize_resident_io(&self.client)
            }
        })?;
        self.cache.clear();
        self.cache_bytes = 0;
        self.banks.clear();
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
    pub fn reserved_bytes(&self) -> Result<u64> {
        self.client
            .memory_usage()
            .map(|m| m.bytes_reserved)
            .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))
    }
    fn check_segmented_resident(&self, resident: bool) -> Result<()> {
        // The constructor pins explicit budget configuration; no environment
        // re-read is needed after creation. Public mutable experiment flags
        // still need revalidation before every allocation/dispatch seam.
        expert_banks::validate_segmented_configuration(
            self.bank_slots,
            Some(self.budget),
            self.budget,
            self.block_moe,
            self.batch_experts,
            self.word_down,
            resident,
        )?;
        expert_banks::check_ordered_segmented(self.bank_slots, self.ordered_public_upload)
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
        if self.resident_admission.is_some() || !self.cache.is_empty() || !self.banks.is_empty() {
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
        let banks = expert_banks::preflight_residency(
            graph,
            self.bank_slots,
            self.budget,
            self.reserved_limit,
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
                    RESIDENT_RUNTIME_IO_MAX
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
        let reserved = padded.max(32 * 1024).next_power_of_two();
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
        let handle = if let Some(admission) = &self.resident_admission {
            admission.allocate(&[padded as u64], || {
                Ok(self.client.create_from_slice(&packed))
            })?
        } else {
            self.client.create_from_slice(&packed)
        };
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
            let size = ExpertBanks::allocation(weights, self.bank_slots)?;
            let bank = if let Some(admission) = &self.resident_admission {
                let requests = ExpertBanks::allocation_requests(weights, self.bank_slots)?;
                admission.allocate(&requests, || {
                    ExpertBanks::new(&self.client, weights, self.bank_slots)
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
                ExpertBanks::new(&self.client, weights, self.bank_slots)?
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
            let size = ExpertBanks::allocation(weights, self.bank_slots)?;
            if self
                .bank_reserved
                .checked_add(size)
                .and_then(|n| n.checked_add(self.budget + 512 * 1024 * 1024))
                .is_none_or(|n| n > self.reserved_limit)
            {
                return Err(Error::LimitExceeded("expert bank residency"));
            }
            let bank = ExpertBanks::new(&self.client, weights, self.bank_slots)?;
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
        self.bank_reserved = 0;
        self.client.memory_cleanup();
        let _ = cubecl::future::block_on(self.client.sync());
    }
}

#[cfg(test)]
mod admission_tests {
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
