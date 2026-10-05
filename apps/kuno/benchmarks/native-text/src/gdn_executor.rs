//! Stateless experimental WGPU executor for the Rust GDN recurrence.
//!
//! CPU `GdnState` owns the only authoritative recurrent state. Each call uploads
//! it, downloads a candidate, validates all columns, queues erasure of every GPU
//! allocation, synchronizes that erasure, then commits. Cancellation/errors
//! preserve the old CPU state. Scratch host buffers use non-elidable clearing.
//! There is no model data, device initialization, state cache or logging here.
//! Runtime-owned upload/readback staging remains CubeCL's responsibility; this
//! experiment alone does not establish production worker privacy acceptance.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::ops::{GdnShape, GdnState};
use kuno_text::{Error, Result};

use super::gdn_gpu;

fn fault(reason: &'static str) -> Error {
    Error::InvalidFormat(reason)
}

/// Only initialized bytes are read/written. Volatile stores and the fence
/// prevent dead-store removal when a buffer is freed immediately afterwards.
fn wipe_bytes(bytes: &mut [u8]) {
    for byte in bytes {
        // SAFETY: byte is a live mutable reference into an owned allocation.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

fn wipe_floats(values: &mut Vec<f32>) {
    let pointer = values.as_mut_ptr();
    for index in 0..values.capacity() {
        // SAFETY: within this owned Vec's writable allocation; zero is a valid
        // f32 representation, including previously initialized spare capacity.
        unsafe { pointer.add(index).write_volatile(0.0) };
    }
    std::sync::atomic::compiler_fence(Ordering::SeqCst);
}

struct SecretFloats(Vec<f32>);
impl SecretFloats {
    fn into_output(mut self) -> Vec<f32> {
        std::mem::take(&mut self.0)
    }
}
impl Drop for SecretFloats {
    fn drop(&mut self) {
        wipe_floats(&mut self.0);
    }
}

struct SecretUpload(Vec<u8>);
impl SecretUpload {
    fn from_floats(values: &[f32]) -> Result<Self> {
        let bytes = values
            .len()
            .checked_mul(4)
            .ok_or(Error::ArithmeticOverflow)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(bytes)
            .map_err(|_| fault("GDN host upload allocation failed"))?;
        for value in values {
            result.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Self(result))
    }
}
impl Drop for SecretUpload {
    fn drop(&mut self) {
        // Initialize spare capacity before clearing it; avoids reading it.
        self.0.resize(self.0.capacity(), 0);
        wipe_bytes(&mut self.0);
    }
}

struct SecretReadbacks(Vec<Bytes>);
impl Drop for SecretReadbacks {
    fn drop(&mut self) {
        for bytes in &mut self.0 {
            wipe_bytes(bytes);
        }
    }
}

struct SecretWords(Vec<u32>);
impl Drop for SecretWords {
    fn drop(&mut self) {
        let pointer = self.0.as_mut_ptr();
        for index in 0..self.0.capacity() {
            // SAFETY: within this owned Vec allocation; zero is valid for u32.
            unsafe { pointer.add(index).write_volatile(0) };
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
}

fn decode_floats(bytes: &[u8], count: usize) -> Result<SecretFloats> {
    if bytes.len() != count.checked_mul(4).ok_or(Error::ArithmeticOverflow)? {
        return Err(fault("GDN readback byte length mismatch"));
    }
    let mut values = SecretFloats(Vec::new());
    values
        .0
        .try_reserve_exact(count)
        .map_err(|_| fault("GDN host readback allocation failed"))?;
    for chunk in bytes.chunks_exact(4) {
        values
            .0
            .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

#[cube(launch)]
fn erase_words(words: &mut Array<u32>) {
    let index = ABSOLUTE_POS;
    if index < words.len() {
        words[index] = 0;
    }
}

struct GpuScratch<'a> {
    client: &'a ComputeClient<WgpuRuntime>,
    handles: Vec<(Handle, usize)>,
}
impl<'a> GpuScratch<'a> {
    fn new(client: &'a ComputeClient<WgpuRuntime>) -> Self {
        // Allocate tracking capacity before the first device allocation.
        Self {
            client,
            handles: Vec::with_capacity(8),
        }
    }
    fn upload(&mut self, values: &[f32]) -> Result<Handle> {
        let upload = SecretUpload::from_floats(values)?;
        let handle = self.client.create_from_slice(&upload.0);
        self.handles.push((handle.clone(), values.len()));
        Ok(handle)
    }
    fn empty(&mut self, words: usize) -> Result<Handle> {
        let bytes = words.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
        let handle = self.client.empty(bytes);
        self.handles.push((handle.clone(), words));
        Ok(handle)
    }
    /// Queue erasure before any handle is released, then wait for completion.
    /// An adapter/device failure may prevent erasure: propagate a static error
    /// and preserve CPU state rather than claim successful cleanup/commit.
    fn erase(&mut self) -> Result<()> {
        let mut failed = false;
        for (handle, words) in &self.handles {
            // Erasure treats f32 and u32 storage as 32-bit words. Zero bits are
            // the correct erasure for both; these allocations never contain f16.
            let queued = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let groups = words.div_ceil(64);
                let x = groups.min(65_535);
                let y = groups.div_ceil(x);
                erase_words::launch::<WgpuRuntime>(
                    self.client,
                    CubeCount::Static(x as u32, y as u32, 1),
                    CubeDim::new_1d(64),
                    // SAFETY: a tracked allocation is exactly words*4 bytes;
                    // the handle remains owned until all queued erasure finishes.
                    unsafe { ArrayArg::from_raw_parts(handle.clone(), *words) },
                );
            }));
            failed |= queued.is_err();
        }
        let synchronized = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::future::block_on(self.client.sync())
        }));
        if failed || !matches!(synchronized, Ok(Ok(()))) {
            return Err(fault("GDN GPU scratch cleanup failed"));
        }
        Ok(())
    }
}

/// Observes cancellation before upload, after download and before CPU commit.
/// A cancellation set after the final observation belongs to the caller's
/// request lifecycle. It must reject late results/reset the enclosing sequence.
#[allow(clippy::too_many_arguments)]
pub fn run(
    client: &ComputeClient<WgpuRuntime>,
    state: &mut GdnState,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    cancel: &AtomicBool,
) -> Result<Vec<f32>> {
    if cancel.load(Ordering::Relaxed) {
        return Err(fault("GDN GPU execution cancelled"));
    }
    let shape = state.shape();
    let layout =
        gdn_gpu::validate_inputs(shape, query, key, value, state.values(), log_decay, beta)
            .map_err(fault)?;
    let controls = SecretFloats(gdn_gpu::pack_controls(shape, log_decay, beta).map_err(fault)?);
    let mut scratch = GpuScratch::new(client);
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q = scratch.upload(query)?;
        let k = scratch.upload(key)?;
        let v = scratch.upload(value)?;
        let control = scratch.upload(&controls.0)?;
        let old = scratch.upload(state.values())?;
        let next = scratch.empty(layout.state_elements)?;
        let out = scratch.empty(layout.value_elements)?;
        let status = scratch.empty(layout.value_elements)?;
        if cancel.load(Ordering::Relaxed) {
            return Err(fault("GDN GPU execution cancelled"));
        }
        gdn_gpu::gdn_step::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(layout.workgroups_x, 1, 1),
            CubeDim::new_1d(gdn_gpu::LANES as u32),
            // SAFETY: input validation and tracked allocations establish exact
            // scalar types/lengths. Old/next use distinct non-aliasing allocations.
            unsafe { ArrayArg::from_raw_parts(q, layout.qk_elements) },
            unsafe { ArrayArg::from_raw_parts(k, layout.qk_elements) },
            unsafe { ArrayArg::from_raw_parts(v, layout.value_elements) },
            unsafe { ArrayArg::from_raw_parts(control, layout.control_elements) },
            unsafe { ArrayArg::from_raw_parts(old, layout.state_elements) },
            unsafe { ArrayArg::from_raw_parts(next.clone(), layout.state_elements) },
            unsafe { ArrayArg::from_raw_parts(out.clone(), layout.value_elements) },
            unsafe { ArrayArg::from_raw_parts(status.clone(), layout.value_elements) },
            shape.head_dim,
            shape.key_heads,
            shape.value_heads,
        );
        let readback = SecretReadbacks(
            cubecl::future::block_on(client.read_async(vec![next, out, status]))
                .map_err(|_| fault("GDN GPU readback failed"))?,
        );
        if readback.0.len() != 3 {
            return Err(fault("GDN GPU readback count mismatch"));
        }
        let next_state = decode_floats(&readback.0[0], layout.state_elements)?;
        let output = decode_floats(&readback.0[1], layout.value_elements)?;
        let statuses = &readback.0[2];
        if statuses.len() != layout.value_elements * 4 {
            return Err(fault("GDN GPU status length mismatch"));
        }
        // Status is success/failure metadata; clear its host copy as well.
        let mut status_words = SecretWords(Vec::new());
        status_words
            .0
            .try_reserve_exact(layout.value_elements)
            .map_err(|_| fault("GDN host status allocation failed"))?;
        for chunk in statuses.chunks_exact(4) {
            status_words
                .0
                .push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        gdn_gpu::validate_download(layout, &next_state.0, &output.0, &status_words.0)
            .map_err(fault)?;
        if cancel.load(Ordering::Relaxed) {
            return Err(fault("GDN GPU execution cancelled"));
        }
        Ok((next_state, output))
    }));
    // This runs for success, Result errors and caught dispatch/readback panics.
    // Candidate guards remain alive until cleanup finishes, including failure.
    let cleanup = scratch.erase();
    let candidate = attempted.map_err(|_| fault("GDN GPU backend failed"))?;
    cleanup?;
    let (next_state, output) = candidate?;
    if cancel.load(Ordering::Relaxed) {
        return Err(fault("GDN GPU execution cancelled"));
    }
    state
        .set_values(&next_state.0)
        .map_err(|_| fault("GDN GPU state commit failed"))?;
    Ok(output.into_output())
}

#[derive(Clone, Copy, Debug)]
pub struct ParityReport {
    pub shape: GdnShape,
    pub steps: usize,
    pub max_state_abs_error: f32,
    pub max_output_abs_error: f32,
    pub elapsed: Duration,
}

fn compare(actual: &[f32], expected: &[f32], max_error: &mut f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(fault("GDN synthetic parity length mismatch"));
    }
    for (&a, &e) in actual.iter().zip(expected) {
        let error = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || error > 3e-5 + 3e-4 * e.abs() {
            return Err(fault("GDN synthetic parity failed"));
        }
        *max_error = max_error.max(error);
    }
    Ok(())
}

/// Explicitly invoked by root after selecting a device. Uses only fixed public
/// synthetic tensors; seven sequential steps compare BOTH state and outputs
/// against the independent higher-precision CPU implementation.
pub fn parity_probe(client: &ComputeClient<WgpuRuntime>) -> Result<Vec<ParityReport>> {
    use kuno_text::ops::l2_norm_heads;
    let mut reports = Vec::new();
    for shape in [
        GdnShape::new(2, 2, 4).map_err(|_| fault("GDN synthetic shape"))?,
        GdnShape::new(5, 2, 6).map_err(|_| fault("GDN synthetic shape"))?,
        GdnShape::qwen38(),
    ] {
        let layout = gdn_gpu::GdnLayout::new(shape).map_err(fault)?;
        let mut actual = GdnState::new(shape).map_err(|_| fault("GDN synthetic state"))?;
        let mut oracle = GdnState::new(shape).map_err(|_| fault("GDN synthetic state"))?;
        let initial = SecretFloats(
            (0..layout.state_elements)
                .map(|i| ((i % 23) as f32 - 11.) / 256.)
                .collect(),
        );
        actual
            .set_values(&initial.0)
            .map_err(|_| fault("GDN synthetic initial state"))?;
        oracle
            .set_values(&initial.0)
            .map_err(|_| fault("GDN synthetic initial state"))?;
        let mut report = ParityReport {
            shape,
            steps: 7,
            max_state_abs_error: 0.,
            max_output_abs_error: 0.,
            elapsed: Duration::ZERO,
        };
        let started = Instant::now();
        for step in 0..report.steps {
            let raw_q = SecretFloats(
                (0..layout.qk_elements)
                    .map(|i| (((i + step * 3) % 19) as f32 - 9.) / 16.)
                    .collect(),
            );
            let raw_k = SecretFloats(
                (0..layout.qk_elements)
                    .map(|i| (((i * 3 + step * 5) % 17) as f32 - 8.) / 16.)
                    .collect(),
            );
            let mut q = SecretFloats(
                l2_norm_heads(&raw_q.0, shape.head_dim, 1e-6)
                    .map_err(|_| fault("GDN synthetic query norm"))?,
            );
            q.0.iter_mut()
                .for_each(|v| *v /= (shape.head_dim as f32).sqrt());
            let k = SecretFloats(
                l2_norm_heads(&raw_k.0, shape.head_dim, 1e-6)
                    .map_err(|_| fault("GDN synthetic key norm"))?,
            );
            let value = SecretFloats(
                (0..layout.value_elements)
                    .map(|i| (((i + step * 7) % 31) as f32 - 15.) / 32.)
                    .collect(),
            );
            let log_decay = SecretFloats(
                (0..shape.value_heads)
                    .map(|h| -(((h + step) % 7 + 1) as f32) / 16.)
                    .collect(),
            );
            let beta = SecretFloats(
                (0..shape.value_heads)
                    .map(|h| ((h + step) % 9) as f32 / 8.)
                    .collect(),
            );
            let expected = SecretFloats(
                oracle
                    .step(&q.0, &k.0, &value.0, &log_decay.0, &beta.0)
                    .map_err(|_| fault("GDN synthetic CPU oracle"))?,
            );
            let output = SecretFloats(run(
                client,
                &mut actual,
                &q.0,
                &k.0,
                &value.0,
                &log_decay.0,
                &beta.0,
                &AtomicBool::new(false),
            )?);
            compare(&output.0, &expected.0, &mut report.max_output_abs_error)?;
            compare(
                actual.values(),
                oracle.values(),
                &mut report.max_state_abs_error,
            )?;
        }
        report.elapsed = started.elapsed();
        reports.push(report);
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_wire_roundtrip_and_length_validation() {
        let input = [0.0, -0.0, 1.25, -3.5, f32::MAX];
        let bytes = SecretUpload::from_floats(&input).unwrap();
        let decoded = decode_floats(&bytes.0, input.len()).unwrap();
        for (&a, &b) in decoded.0.iter().zip(&input) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(decode_floats(&bytes.0[..bytes.0.len() - 1], input.len()).is_err());
    }

    #[test]
    fn parity_comparison_rejects_nonfinite_and_excessive_error() {
        let mut max_error = 0.0;
        assert!(compare(&[1.00001], &[1.], &mut max_error).is_ok());
        assert!(max_error > 0.);
        assert!(compare(&[f32::NAN], &[1.], &mut max_error).is_err());
        assert!(compare(&[1.], &[f32::NAN], &mut max_error).is_err());
        assert!(compare(&[1.01], &[1.], &mut max_error).is_err());
        assert!(compare(&[], &[1.], &mut max_error).is_err());
    }
}
