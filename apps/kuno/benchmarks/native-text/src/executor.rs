//! Operator laboratory only: bounded packed model cache + Rust GPU matvec.
//! No HTTP path, arbitrary prompts, production logging or installation.
use crate::quant_matvec;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::MatrixExecutor;
use kuno_text::{Error, Result, TensorView};
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
pub struct GpuExecutor {
    client: ComputeClient<WgpuRuntime>,
    cache: BTreeMap<Key, Cached>,
    cache_bytes: usize,
    budget: usize,
    tick: u64,
    pub uploaded_bytes: u64,
    pub cache_hits: u64,
    pub calls: u64,
    pub matrix_ns: u128,
    pub weight_read_ns: u128,
    pub weight_upload_ns: u128,
    pub gpu_wait_ns: u128,
}
impl GpuExecutor {
    pub fn new(client: ComputeClient<WgpuRuntime>, budget: usize) -> Result<Self> {
        if !(TILE_BYTES..=16 * 1024 * 1024 * 1024).contains(&budget) {
            return Err(Error::LimitExceeded("laboratory GPU cache"));
        }
        Ok(Self {
            client,
            cache: BTreeMap::new(),
            cache_bytes: 0,
            budget,
            tick: 0,
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
        self.cache_bytes
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
        while self.cache_bytes + padded > self.budget {
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
        if self.reserved_bytes()? > (self.budget + 512 * 1024 * 1024) as u64 {
            self.cache.clear();
            self.cache_bytes = 0;
            self.client.memory_cleanup();
            if self.reserved_bytes()? > (self.budget + 512 * 1024 * 1024) as u64 {
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
                bytes: padded,
                last: self.tick,
            },
        );
        self.cache_bytes += padded;
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

impl MatrixExecutor for GpuExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let started = Instant::now();
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
            let per_tile = TILE_BYTES / row_bytes;
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
                // SAFETY: host geometry/read/upload validation proves the
                // handles hold exactly these typed logical element counts.
                quant_matvec::packed_matvec::launch::<WgpuRuntime>(
                    &self.client,
                    CubeCount::Static((rows as u32).div_ceil(64), 1, 1),
                    CubeDim::new_1d(64),
                    unsafe { ArrayArg::from_raw_parts(weights, words) },
                    unsafe { ArrayArg::from_raw_parts(x.clone(), input.len()) },
                    unsafe { ArrayArg::from_raw_parts(y.clone(), rows) },
                    rows,
                    matrix.inputs(),
                    dtype,
                );
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
                let bytes = read.map_err(|_| Error::InvalidFormat("GPU output read"))?;
                let values = f32::from_bytes(&bytes);
                if values.len() != rows || values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::InvalidFormat("GPU output values"));
                }
                result[first..first + rows].copy_from_slice(values);
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
        self.matrix_ns += started.elapsed().as_nanos();
        Ok(result)
    }
}
impl Drop for GpuExecutor {
    fn drop(&mut self) {
        self.cache.clear();
        self.cache_bytes = 0;
        self.client.memory_cleanup();
        let _ = cubecl::future::block_on(self.client.sync());
    }
}
