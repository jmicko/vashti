//! Bounded parallel CPU rows for the native Rust graph.
//!
//! The executor retains only a dedicated thread pool. Packed weight tiles and
//! decoded rows are request-local; no expert, input, output or routing cache
//! survives a call. Each row uses the same serial f32 accumulation as Scalar.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;
use zeroize::{Zeroize, Zeroizing};

use crate::cpu::Matrix;
use crate::model::MatrixExecutor;
use crate::{Error, Result, TensorView};

const TILE_READ_BYTES: usize = 512 * 1024;
const TILE_ROWS: usize = 32;

pub struct ThreadedExecutor {
    pool: rayon::ThreadPool,
}

impl ThreadedExecutor {
    pub fn new(threads: usize) -> Result<Self> {
        if !(1..=32).contains(&threads) {
            return Err(Error::LimitExceeded("threaded matrix thread count"));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("kuno-text-cpu-{index}"))
            .build()
            .map_err(|_| Error::LimitExceeded("threaded matrix thread pool"))?;
        Ok(Self { pool })
    }

    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Run one selected expert or ordinary matrix. On every error, the complete
    /// caller output slice is zeroed after all dispatched tasks have returned.
    pub fn apply(
        &self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        output: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        let result = self.compute(tensor, expert, input, output, cancel);
        if result.is_err() {
            output.zeroize();
        }
        result
    }

    fn compute(
        &self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        output: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        let matrix = Matrix::new(tensor)?;
        if input.len() != matrix.inputs()
            || output.len() != matrix.outputs()
            || expert >= matrix.experts()
        {
            return Err(Error::InvalidBufferLength);
        }
        if !input.iter().all(|value| value.is_finite()) {
            return Err(Error::InvalidFormat("non-finite matrix input"));
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("matrix execution cancelled"));
        }
        let row_bytes =
            usize::try_from(tensor.info().row_bytes()).map_err(|_| Error::ArithmeticOverflow)?;
        if row_bytes == 0 || row_bytes > TILE_READ_BYTES {
            return Err(Error::LimitExceeded("matrix row scratch"));
        }
        let rows_per_tile = TILE_ROWS.min(TILE_READ_BYTES / row_bytes);
        let tile_bytes = rows_per_tile
            .checked_mul(row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        let expert_start = expert
            .checked_mul(matrix.outputs())
            .ok_or(Error::ArithmeticOverflow)?;
        let failed = AtomicBool::new(false);
        let first_error = Mutex::new(None);
        // install/for_each waits for every in-flight closure before returning;
        // the caller's borrowed request slices never enter persistent pool state.
        self.pool.install(|| {
            output
                .par_chunks_mut(rows_per_tile)
                .enumerate()
                .for_each_init(
                    || Scratch::new(tile_bytes, matrix.inputs()),
                    |scratch, (tile, outputs)| {
                        let result = (|| {
                            if failed.load(Ordering::Relaxed) {
                                return Ok(());
                            }
                            if cancel.load(Ordering::Relaxed) {
                                return Err(Error::InvalidFormat("matrix execution cancelled"));
                            }
                            let scratch = scratch.as_mut().map_err(|_| {
                                Error::LimitExceeded("threaded matrix scratch allocation")
                            })?;
                            let first = tile
                                .checked_mul(rows_per_tile)
                                .ok_or(Error::ArithmeticOverflow)?;
                            let start = expert_start
                                .checked_add(first)
                                .and_then(|row| row.checked_mul(row_bytes))
                                .ok_or(Error::ArithmeticOverflow)?;
                            let count = outputs
                                .len()
                                .checked_mul(row_bytes)
                                .ok_or(Error::ArithmeticOverflow)?;
                            tensor.read_range(
                                u64::try_from(start).map_err(|_| Error::ArithmeticOverflow)?,
                                &mut scratch.packed[..count],
                            )?;
                            for (row, target) in outputs.iter_mut().enumerate() {
                                if failed.load(Ordering::Relaxed) {
                                    return Ok(());
                                }
                                if cancel.load(Ordering::Relaxed) {
                                    return Err(Error::InvalidFormat("matrix execution cancelled"));
                                }
                                tensor.info().dtype().decode(
                                    &scratch.packed[row * row_bytes..(row + 1) * row_bytes],
                                    &mut scratch.decoded,
                                )?;
                                if !scratch.decoded.iter().all(|value| value.is_finite()) {
                                    return Err(Error::InvalidFormat("non-finite matrix weights"));
                                }
                                let sum = scratch
                                    .decoded
                                    .iter()
                                    .zip(input)
                                    .map(|(weight, value)| weight * value)
                                    .sum::<f32>();
                                if !sum.is_finite() {
                                    return Err(Error::InvalidFormat("non-finite matrix output"));
                                }
                                *target = sum;
                            }
                            Ok(())
                        })();
                        if let Err(error) = result {
                            let mut first = first_error
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            if first.is_none() {
                                *first = Some(error);
                            }
                            failed.store(true, Ordering::Relaxed);
                        }
                    },
                )
        });
        if let Some(error) = first_error
            .into_inner()
            .map_err(|_| Error::InvalidFormat("threaded matrix error lock"))?
        {
            return Err(error);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("matrix execution cancelled"));
        }
        Ok(())
    }
}

impl MatrixExecutor for ThreadedExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let matrix = Matrix::new(tensor)?;
        let mut output = Zeroizing::new(Vec::new());
        output
            .try_reserve_exact(matrix.outputs())
            .map_err(|_| Error::LimitExceeded("threaded matrix output allocation"))?;
        output.resize(matrix.outputs(), 0.);
        self.apply(tensor, expert, input, &mut output, cancel)?;
        Ok(std::mem::take(&mut *output))
    }
}

struct Scratch {
    packed: Zeroizing<Vec<u8>>,
    decoded: Zeroizing<Vec<f32>>,
}

impl Scratch {
    fn new(bytes: usize, features: usize) -> Result<Self> {
        let mut packed = Zeroizing::new(Vec::new());
        packed
            .try_reserve_exact(bytes)
            .map_err(|_| Error::LimitExceeded("threaded matrix scratch allocation"))?;
        packed.resize(bytes, 0);
        let mut decoded = Zeroizing::new(Vec::new());
        decoded
            .try_reserve_exact(features)
            .map_err(|_| Error::LimitExceeded("threaded matrix scratch allocation"))?;
        decoded.resize(features, 0.);
        Ok(Self { packed, decoded })
    }
}
