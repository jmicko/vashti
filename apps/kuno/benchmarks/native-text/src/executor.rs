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
}

#[cube(launch)]
fn clear_scratch(input: &mut Array<f32>, output: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        input[i] = 0.;
    }
    if i < output.len() {
        output[i] = 0.;
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
                    (&bank.gate, weights.gate),
                    (&bank.up, weights.up),
                    (&bank.down, weights.down),
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
        let mut result = vec![0.; matrix.outputs()];
        let execute = (|| {
            let row_bytes = tensor.info().row_bytes() as usize;
            if row_bytes > TILE_BYTES {
                return Err(Error::LimitExceeded("GPU row"));
            }
            // Keep cooperative row grids within the portable Vulkan limits.
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
                    input.len(),
                    rows,
                )
                .map_err(|_| Error::InvalidFormat("GPU packed layout"))?;
                let (weights, words) = self.weight(tensor, expert, first, rows)?;
                let x = self.client.create_from_slice(f32::as_bytes(input));
                let y = self.client.empty(rows * 4);
                if self.reserved_bytes()? > self.reserved_limit as u64 {
                    expert_banks::clear(&self.client, &[(x, input.len()), (y, rows)]);
                    return Err(Error::LimitExceeded("GPU allocator reserved memory"));
                }
                // SAFETY: host geometry/read/upload validation proves the
                // handles hold exactly these typed logical element counts.
                unsafe {
                    quant_matvec::cooperative::cooperative_matvec::launch_unchecked::<WgpuRuntime>(
                        &self.client,
                        CubeCount::Static(rows as u32, 1, 1),
                        CubeDim::new_1d(64),
                        ArrayArg::from_raw_parts(weights, words),
                        ArrayArg::from_raw_parts(x.clone(), input.len()),
                        ArrayArg::from_raw_parts(y.clone(), rows),
                        rows,
                        matrix.inputs(),
                        dtype,
                    );
                }
                let waited = Instant::now();
                let read = self.client.read_one(y.clone());
                self.gpu_wait_ns += waited.elapsed().as_nanos();
                // SAFETY: same validated x/y handles; clear initialized logical
                // scratch before releasing them to the runtime allocator.
                clear_scratch::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static((rows.max(input.len()) as u32).div_ceil(64), 1, 1),
                    CubeDim::new_1d(64),
                    unsafe { ArrayArg::from_raw_parts(x, input.len()) },
                    unsafe { ArrayArg::from_raw_parts(y, rows) },
                );
                let mut bytes = read.map_err(|_| Error::InvalidFormat("GPU output read"))?;
                let values = f32::from_bytes(&bytes);
                if values.len() != rows || values.iter().any(|v| !v.is_finite()) {
                    bytes.zeroize();
                    return Err(Error::InvalidFormat("GPU output values"));
                }
                result[first..first + rows].copy_from_slice(values);
                bytes.zeroize();
                self.calls += 1;
            }
            Ok(())
        })();
        if let Err(error) = execute {
            result.zeroize();
            return Err(error);
        }
        if cancel.load(Ordering::Relaxed) {
            result.zeroize();
            return Err(Error::InvalidFormat("GPU execution cancelled"));
        }
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
