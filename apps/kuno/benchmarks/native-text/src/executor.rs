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
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use zeroize::Zeroize;

const TILE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const RESIDENT_IO_MAX: u64 = 64 * 1024 * 1024;

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
    telemetry: impl FnOnce() -> Result<u64>,
) -> Result<T> {
    let result = (|| {
        if ledger.is_poisoned() {
            return Err(Error::InvalidFormat("resident admission poisoned"));
        }
        let value = read()?;
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
        self.with_ledger(|ledger| guarded_readback(ledger, f, || self.telemetry()))
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
    pub batch_experts: bool,
    pub expert_batch_bytes: usize,
    pub expert_scatter: expert_banks::ScatterMode,
    pub gpu_gdn: bool,
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
        Ok(Self {
            small_cpu: crate::small_cpu::SmallCpu::new(model, 64 * 1024 * 1024)?,
            small_cpu_enabled: std::env::var("KUNO_LAB_SMALL_CPU").as_deref() == Ok("1"),
            small_cpu_ns: 0,
            small_cpu_calls: 0,
            block_kernel: std::env::var("KUNO_LAB_BLOCK_KERNEL").as_deref() != Ok("0"),
            block_large_only: std::env::var("KUNO_LAB_BLOCK_KERNEL").as_deref() != Ok("1"),
            block_moe: std::env::var("KUNO_LAB_BLOCK_MOE").as_deref() == Ok("1"),
            // Candidate batching stays opt-in until paired measurements win.
            batch_experts: std::env::var("KUNO_LAB_BATCH_EXPERTS").as_deref() == Ok("1"),
            expert_batch_bytes,
            expert_scatter,
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
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes + self.bank_reserved
    }
    pub fn reserved_bytes(&self) -> Result<u64> {
        self.client
            .memory_usage()
            .map(|m| m.bytes_reserved)
            .map_err(|_| Error::InvalidFormat("GPU memory telemetry"))
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
        if self.resident_admission.is_some() || !self.cache.is_empty() || !self.banks.is_empty() {
            return Err(Error::InvalidFormat("resident executor already used"));
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
                public_pages,
                source_pages,
                baseline,
            ],
        )
        .map_err(admission_error)?;
        let admission = ResidentAdmission {
            client: self.client.clone(),
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
                    let (bytes, read, upload) =
                        expert_banks::fill(&self.client, target, tensor, expert, slot, cancel)?;
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
                let requests = [weights.gate, weights.up, weights.down]
                    .map(|t| {
                        t.info()
                            .row_bytes()
                            .checked_mul(t.info().dimensions()[1])
                            .and_then(|n| n.checked_mul(self.bank_slots as u64))
                            .ok_or(Error::ArithmeticOverflow)
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?;
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
    use super::*;
    use std::cell::{Cell, RefCell};
    const PAGE: u64 = main_admission::MIN_PAGE_BYTES;
    fn failure<T>() -> Result<T> {
        Err(Error::InvalidFormat("fixed public failure fixture"))
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
