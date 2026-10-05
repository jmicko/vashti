//! Operator laboratory only: bounded packed model cache + Rust GPU matvec.
//! No HTTP path, arbitrary prompts, production logging or installation.
use crate::{
    expert_banks::{self, ExpertBanks},
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
use std::time::Instant;
use zeroize::Zeroize;

const TILE_BYTES: usize = 64 * 1024 * 1024;
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
    fn weight(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        first: usize,
        rows: usize,
    ) -> Result<(Handle, usize)> {
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
        if self.reserved_bytes()?
            > (self.budget + self.bank_reserved + 512 * 1024 * 1024).min(self.reserved_limit) as u64
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
        let handle = self.client.create_from_slice(&packed);
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
            if self
                .bank_reserved
                .checked_add(size)
                .and_then(|n| n.checked_add(self.budget + 512 * 1024 * 1024))
                .is_none_or(|n| n > self.reserved_limit)
            {
                resident.poison();
                return Err(Error::LimitExceeded("resident expert bank residency"));
            }
            let bank = ExpertBanks::new(&self.client, weights, self.bank_slots)?;
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
            for &(expert, slot) in &plan.uploads {
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
            if self.reserved_bytes()? > self.reserved_limit as u64 {
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
            if self.reserved_bytes()? > self.reserved_limit as u64 {
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
            for &(expert, slot) in &plan.uploads {
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
