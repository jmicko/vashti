//! Checked scalar reference operators for the borrowed `qwen4exp` GGUF graph.
//!
//! Dense/quantized projections, model admission and execution are owned by the
//! graph. These operators use GGUF's already-folded norm multipliers and tiled
//! GDN value-head order. They are correctness oracles, not optimized kernels.
//! Sums use f64 where useful, then checked f32 conversion; GPU acceptance must
//! use numerical tolerances rather than assume identical reduction rounding.
//!
//! Formula references (independently translated into Rust): Transformers
//! `0a896aa41bba78f92338db21cc468fe043888657`, `modeling_qwen4_exp.py`
//! (Apache-2.0); llama.cpp `806eee9841de5f2c20f9d43914117f157d2baacc`,
//! `conversion/qwen.py`, `src/models/qwen4exp.cpp` (MIT). In particular, HF's
//! grouped GDN heads become tiled GGUF heads; QSA selects complete blocks plus
//! the exact incomplete tail. No upstream engine or native kernel is called.
//!
//! Persistent operator state is cleared on reset/drop. Returned intermediate
//! vectors remain request content: the caller must protect/clear them and must
//! never write real-request checkpoints or values to diagnostics.

use std::fmt;

#[path = "ops_router_guarded.rs"]
mod router_guarded;
pub use router_guarded::{
    GUARDED_ROUTER_CAPACITY_BYTES, GUARDED_ROUTER_MAX_EXPERTS, GUARDED_ROUTER_MAX_TOP_K,
    GuardedRouterSelection, router_topk_guarded,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpsError {
    InvalidShape(&'static str),
    DimensionMismatch(&'static str),
    InvalidParameter(&'static str),
    NonFinite(&'static str),
    CapacityExceeded(&'static str),
    AllocationFailed(&'static str),
}

impl fmt::Display for OpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, context) = match self {
            Self::InvalidShape(c) => ("invalid shape", c),
            Self::DimensionMismatch(c) => ("dimension mismatch", c),
            Self::InvalidParameter(c) => ("invalid parameter", c),
            Self::NonFinite(c) => ("non-finite arithmetic", c),
            Self::CapacityExceeded(c) => ("capacity exceeded", c),
            Self::AllocationFailed(c) => ("allocation failed", c),
        };
        write!(f, "{context}: {kind}")
    }
}

impl std::error::Error for OpsError {}

fn finite(values: &[f32], context: &'static str) -> Result<(), OpsError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(OpsError::NonFinite(context))
    }
}

fn checked_f32(value: f64, context: &'static str) -> Result<f32, OpsError> {
    let output = value as f32;
    if value.is_finite() && output.is_finite() {
        Ok(output)
    } else {
        Err(OpsError::NonFinite(context))
    }
}

fn product(a: usize, b: usize, context: &'static str) -> Result<usize, OpsError> {
    a.checked_mul(b).ok_or(OpsError::CapacityExceeded(context))
}

fn zeros(length: usize, context: &'static str) -> Result<Vec<f32>, OpsError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| OpsError::AllocationFailed(context))?;
    result.resize(length, 0.0);
    Ok(result)
}

fn checked_copy(input: &[f32], context: &'static str) -> Result<Vec<f32>, OpsError> {
    let mut result = zeros(input.len(), context)?;
    result.copy_from_slice(input);
    Ok(result)
}

#[allow(unsafe_code)]
fn clear_f32_capacity(values: &mut Vec<f32>) {
    let pointer = values.as_mut_ptr();
    for i in 0..values.capacity() {
        // SAFETY: every address is within this Vec's writable allocation. f32
        // has no invalid zero representation. Volatile writes also clear old
        // initialized values beyond len without reading uninitialized slots.
        unsafe { pointer.add(i).write_volatile(0.0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[allow(unsafe_code)]
fn clear_i32_capacity(values: &mut Vec<i32>) {
    let pointer = values.as_mut_ptr();
    for i in 0..values.capacity() {
        // SAFETY: within the owned Vec allocation; zero is valid for i32.
        unsafe { pointer.add(i).write_volatile(0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

pub fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// Grouped RMS normalization using direct, already-folded GGUF gamma.
/// Gamma may be shared by every group or have one multiplier per input value.
pub fn rms_norm(
    input: &[f32],
    gamma: &[f32],
    group_width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, OpsError> {
    if group_width == 0 || input.is_empty() || !input.len().is_multiple_of(group_width) {
        return Err(OpsError::InvalidShape("rms_norm"));
    }
    if gamma.len() != group_width && gamma.len() != input.len() {
        return Err(OpsError::DimensionMismatch("rms_norm gamma"));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OpsError::InvalidParameter("rms_norm epsilon"));
    }
    finite(input, "rms_norm input")?;
    finite(gamma, "rms_norm gamma")?;
    let mut result = zeros(input.len(), "rms_norm")?;
    for (group, values) in input.chunks_exact(group_width).enumerate() {
        let mean = values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / group_width as f64;
        let inverse = 1.0 / (mean + f64::from(epsilon)).sqrt();
        for (i, &value) in values.iter().enumerate() {
            let flat = group * group_width + i;
            let weight = gamma[if gamma.len() == group_width { i } else { flat }];
            result[flat] = checked_f32(
                f64::from(value) * inverse * f64::from(weight),
                "rms_norm result",
            )?;
        }
    }
    Ok(result)
}

/// GDN normalizes by the sum of squares, unlike RMS's mean of squares.
pub fn l2_norm_heads(input: &[f32], width: usize, epsilon: f32) -> Result<Vec<f32>, OpsError> {
    if width == 0 || input.is_empty() || !input.len().is_multiple_of(width) {
        return Err(OpsError::InvalidShape("l2_norm_heads"));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OpsError::InvalidParameter("l2_norm_heads epsilon"));
    }
    finite(input, "l2_norm_heads input")?;
    let mut output = zeros(input.len(), "l2_norm_heads")?;
    for (group, head) in input.chunks_exact(width).enumerate() {
        let sum = head.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let inv = 1.0 / (sum + f64::from(epsilon)).sqrt();
        for (i, &v) in head.iter().enumerate() {
            output[group * width + i] = checked_f32(f64::from(v) * inv, "l2_norm_heads result")?;
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedResidualShape {
    pub streams: usize,
    pub width: usize,
}

impl GatedResidualShape {
    pub fn new(streams: usize, width: usize) -> Result<Self, OpsError> {
        if streams == 0 || width == 0 {
            return Err(OpsError::InvalidShape("gated residual"));
        }
        product(streams, width, "gated residual")?;
        Ok(Self { streams, width })
    }

    pub fn activate_down(&self, projected_down: &[f32]) -> Result<Vec<f32>, OpsError> {
        Self::new(self.streams, self.width)?;
        if projected_down.is_empty() {
            return Err(OpsError::InvalidShape("gated residual low rank"));
        }
        finite(projected_down, "gated residual down")?;
        projected_down
            .iter()
            .map(|&v| {
                checked_f32(
                    f64::from(silu(v / self.streams as f32)),
                    "gated residual down",
                )
            })
            .collect()
    }

    pub fn mix(&self, normalized: &[f32], projected_up: &[f32]) -> Result<Vec<f32>, OpsError> {
        let expected = product(self.streams, self.width, "gated residual mix")?;
        Self::new(self.streams, self.width)?;
        if normalized.len() != expected || projected_up.len() != expected {
            return Err(OpsError::DimensionMismatch("gated residual mix"));
        }
        finite(normalized, "gated residual normalized")?;
        finite(projected_up, "gated residual up")?;
        let mut mixed = zeros(self.width, "gated residual mix")?;
        for (d, output) in mixed.iter_mut().enumerate() {
            let sum = (0..self.streams)
                .map(|c| {
                    let i = c * self.width + d;
                    f64::from(normalized[i]) * f64::from(sigmoid(projected_up[i]))
                })
                .sum::<f64>();
            *output = checked_f32(sum / self.streams as f64, "gated residual mix result")?;
        }
        Ok(mixed)
    }

    pub fn combine(
        &self,
        original_residual: &[f32],
        block_output: &[f32],
        projected_injection: &[f32],
    ) -> Result<Vec<f32>, OpsError> {
        Self::new(self.streams, self.width)?;
        if original_residual.len() != product(self.streams, self.width, "gated residual combine")?
            || block_output.len() != self.width
            || projected_injection.len() != self.streams
        {
            return Err(OpsError::DimensionMismatch("gated residual combine"));
        }
        finite(original_residual, "gated residual original")?;
        finite(block_output, "gated residual block")?;
        finite(projected_injection, "gated residual injection")?;
        let mut result = zeros(original_residual.len(), "gated residual combine")?;
        for (c, &injection) in projected_injection.iter().enumerate() {
            let weight = 2.0 * sigmoid(injection / self.streams as f32);
            for (d, &value) in block_output.iter().enumerate() {
                let i = c * self.width + d;
                result[i] = checked_f32(
                    f64::from(original_residual[i]) + f64::from(value) * f64::from(weight),
                    "gated residual combine result",
                )?;
            }
        }
        Ok(result)
    }
}

/// Channel-major causal convolution history, oldest sample first.
/// Input history is raw for GDN and normalized gated-value history for PLE.
pub struct CausalConvState {
    channels: usize,
    kernel: usize,
    dilation: usize,
    history: Vec<f32>,
}

impl CausalConvState {
    pub fn new(channels: usize, kernel: usize, dilation: usize) -> Result<Self, OpsError> {
        Self::new_bounded(channels, kernel, dilation, 128 * 1024 * 1024)
    }

    pub fn new_bounded(
        channels: usize,
        kernel: usize,
        dilation: usize,
        byte_budget: usize,
    ) -> Result<Self, OpsError> {
        if channels == 0 || kernel == 0 || dilation == 0 {
            return Err(OpsError::InvalidShape("causal convolution"));
        }
        let taps = product(kernel - 1, dilation, "causal convolution history")?;
        let elements = product(channels, taps, "causal convolution history")?;
        if product(elements, 4, "causal convolution bytes")? > byte_budget {
            return Err(OpsError::CapacityExceeded("causal convolution byte budget"));
        }
        Ok(Self {
            channels,
            kernel,
            dilation,
            history: zeros(elements, "causal convolution")?,
        })
    }

    pub fn history(&self) -> &[f32] {
        &self.history
    }

    pub fn reset(&mut self) {
        clear_f32_capacity(&mut self.history);
    }

    fn preview(&self, input: &[f32], weights: &[f32]) -> Result<Vec<f32>, OpsError> {
        if input.len() != self.channels
            || weights.len() != product(self.channels, self.kernel, "causal convolution weights")?
        {
            return Err(OpsError::DimensionMismatch("causal convolution"));
        }
        finite(input, "causal convolution input")?;
        finite(weights, "causal convolution kernel")?;
        let past = (self.kernel - 1) * self.dilation;
        let mut result = zeros(self.channels, "causal convolution output")?;
        for (c, output) in result.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            for tap in 0..self.kernel {
                let value = if tap + 1 == self.kernel {
                    input[c]
                } else {
                    self.history[c * past + tap * self.dilation]
                };
                sum += f64::from(value) * f64::from(weights[c * self.kernel + tap]);
            }
            *output = checked_f32(sum, "causal convolution output")?;
        }
        Ok(result)
    }

    fn advance(&mut self, input: &[f32]) {
        let past = (self.kernel - 1) * self.dilation;
        if past == 0 {
            return;
        }
        for (c, &value) in input.iter().enumerate() {
            let history = &mut self.history[c * past..(c + 1) * past];
            history.copy_within(1.., 0);
            history[past - 1] = value;
        }
    }

    /// Commit history only after every output passed finite/shape checks.
    pub fn step(&mut self, input: &[f32], weights: &[f32]) -> Result<Vec<f32>, OpsError> {
        let output = self.preview(input, weights)?;
        self.advance(input);
        Ok(output)
    }
}

impl Drop for CausalConvState {
    fn drop(&mut self) {
        self.reset();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnShape {
    pub head_dim: usize,
    pub key_heads: usize,
    pub value_heads: usize,
}

impl GdnShape {
    pub fn new(head_dim: usize, key_heads: usize, value_heads: usize) -> Result<Self, OpsError> {
        if head_dim == 0
            || key_heads == 0
            || value_heads == 0
            || !value_heads.is_multiple_of(key_heads)
        {
            return Err(OpsError::InvalidShape("gdn"));
        }
        product(
            product(value_heads, head_dim, "gdn state")?,
            head_dim,
            "gdn state",
        )?;
        Ok(Self {
            head_dim,
            key_heads,
            value_heads,
        })
    }

    pub fn qwen38() -> Self {
        Self {
            head_dim: 128,
            key_heads: 16,
            value_heads: 48,
        }
    }
}

/// Logical state layout is `[value_head][key_dim][value_dim]`, last dim fastest.
pub struct GdnState {
    shape: GdnShape,
    values: Vec<f32>,
}

impl GdnState {
    pub fn new(shape: GdnShape) -> Result<Self, OpsError> {
        Self::new_bounded(shape, 128 * 1024 * 1024)
    }

    pub fn new_bounded(shape: GdnShape, byte_budget: usize) -> Result<Self, OpsError> {
        GdnShape::new(shape.head_dim, shape.key_heads, shape.value_heads)?;
        let elements = product(
            product(shape.value_heads, shape.head_dim, "gdn state")?,
            shape.head_dim,
            "gdn state",
        )?;
        // step uses a second state image for transactional finite checking.
        if product(elements, 8, "gdn state bytes")? > byte_budget {
            return Err(OpsError::CapacityExceeded("gdn byte budget"));
        }
        Ok(Self {
            shape,
            values: zeros(elements, "gdn state")?,
        })
    }

    pub fn shape(&self) -> GdnShape {
        self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn set_values(&mut self, values: &[f32]) -> Result<(), OpsError> {
        if values.len() != self.values.len() {
            return Err(OpsError::DimensionMismatch("gdn initial state"));
        }
        finite(values, "gdn initial state")?;
        self.values.copy_from_slice(values);
        Ok(())
    }

    pub fn reset(&mut self) {
        clear_f32_capacity(&mut self.values);
    }

    /// Q/K are already L2 normalized; Q also includes `1/sqrt(head_dim)`.
    /// Decay arguments are nonpositive LOG decays. Beta is sigmoid output.
    pub fn step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, OpsError> {
        let s = self.shape;
        let qk_elements = product(s.key_heads, s.head_dim, "gdn qk")?;
        let v_elements = product(s.value_heads, s.head_dim, "gdn value")?;
        if query.len() != qk_elements
            || key.len() != qk_elements
            || value.len() != v_elements
            || log_decay.len() != s.value_heads
            || beta.len() != s.value_heads
        {
            return Err(OpsError::DimensionMismatch("gdn step"));
        }
        finite(query, "gdn query")?;
        finite(key, "gdn key")?;
        finite(value, "gdn value")?;
        finite(log_decay, "gdn decay")?;
        finite(beta, "gdn beta")?;
        if log_decay.iter().any(|&g| g > 0.0) || beta.iter().any(|&b| !(0.0..=1.0).contains(&b)) {
            return Err(OpsError::InvalidParameter("gdn decay or beta"));
        }
        // Allocate the output before copying private state, so an output
        // allocation failure cannot abandon an uncleared state copy.
        let mut output = zeros(v_elements, "gdn output")?;
        let mut next = checked_copy(&self.values, "gdn next state")?;
        let compute = (|| {
            for h in 0..s.value_heads {
                let kh = h % s.key_heads;
                let decay = log_decay[h].exp();
                for j in 0..s.head_dim {
                    let mut predicted = 0.0_f64;
                    for i in 0..s.head_dim {
                        let index = (h * s.head_dim + i) * s.head_dim + j;
                        next[index] *= decay;
                        predicted += f64::from(next[index]) * f64::from(key[kh * s.head_dim + i]);
                    }
                    let delta = checked_f32(
                        (f64::from(value[h * s.head_dim + j]) - predicted) * f64::from(beta[h]),
                        "gdn delta",
                    )?;
                    let mut readout = 0.0_f64;
                    for i in 0..s.head_dim {
                        let index = (h * s.head_dim + i) * s.head_dim + j;
                        next[index] = checked_f32(
                            f64::from(next[index])
                                + f64::from(key[kh * s.head_dim + i]) * f64::from(delta),
                            "gdn updated state",
                        )?;
                        readout += f64::from(next[index]) * f64::from(query[kh * s.head_dim + i]);
                    }
                    output[h * s.head_dim + j] = checked_f32(readout, "gdn readout")?;
                }
            }
            Ok(())
        })();
        if let Err(error) = compute {
            clear_f32_capacity(&mut next);
            clear_f32_capacity(&mut output);
            return Err(error);
        }
        std::mem::swap(&mut self.values, &mut next);
        clear_f32_capacity(&mut next);
        Ok(output)
    }
}

impl Drop for GdnState {
    fn drop(&mut self) {
        self.reset();
    }
}

/// Stored `ssm_a` is already `-exp(A_log)`, not the HF logarithm.
pub fn gdn_log_decay(
    alpha: &[f32],
    dt_bias: &[f32],
    stored_a: &[f32],
) -> Result<Vec<f32>, OpsError> {
    if alpha.is_empty() || alpha.len() != dt_bias.len() || alpha.len() != stored_a.len() {
        return Err(OpsError::DimensionMismatch("gdn log decay"));
    }
    finite(alpha, "gdn alpha")?;
    finite(dt_bias, "gdn dt bias")?;
    finite(stored_a, "gdn stored a")?;
    if stored_a.iter().any(|&v| v >= 0.0) {
        return Err(OpsError::InvalidParameter("gdn stored a sign"));
    }
    alpha
        .iter()
        .zip(dt_bias)
        .zip(stored_a)
        .map(|((&a, &dt), &weight)| {
            let biased = checked_f32(f64::from(a) + f64::from(dt), "gdn alpha bias")?;
            checked_f32(
                f64::from(softplus(biased)) * f64::from(weight),
                "gdn log decay result",
            )
        })
        .collect()
}

pub fn gdn_output_gate(
    output: &[f32],
    gate: &[f32],
    gamma: &[f32],
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<f32>, OpsError> {
    if output.len() != gate.len() {
        return Err(OpsError::DimensionMismatch("gdn output gate"));
    }
    finite(gate, "gdn output gate")?;
    let mut normalized = rms_norm(output, gamma, head_dim, epsilon)?;
    for (result, &z) in normalized.iter_mut().zip(gate) {
        *result *= sigmoid(z);
    }
    Ok(normalized)
}

#[derive(Clone, Debug)]
pub struct PleHashConfig {
    multipliers: Vec<u64>,
    vocab_sizes: Vec<u64>,
    offsets: Vec<u64>,
    heads_per_ngram: usize,
    boundary_token: i32,
    token_vocab_size: u32,
}

/// Borrowed original public PLE integer metadata. No allocation, conversion,
/// token history, selected rows, cache keys, or request data are returned.
#[derive(Clone, Copy)]
pub struct PleHashMetadata<'a> {
    pub multipliers: &'a [u64],
    pub vocab_sizes: &'a [u64],
    pub offsets: &'a [u64],
    pub heads_per_ngram: usize,
    pub boundary_token: i32,
    pub token_vocab_size: u32,
}

impl PleHashConfig {
    pub fn new(
        multipliers: Vec<u64>,
        vocab_sizes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: usize,
        boundary_token: i32,
        token_vocab_size: u32,
        table_rows: u64,
    ) -> Result<Self, OpsError> {
        if !(2..=16).contains(&multipliers.len())
            || heads_per_ngram == 0
            || token_vocab_size == 0
            || boundary_token < 0
            || boundary_token as u32 >= token_vocab_size
        {
            return Err(OpsError::InvalidShape("ple hash"));
        }
        let heads = product(multipliers.len() - 1, heads_per_ngram, "ple hash heads")?;
        if heads != vocab_sizes.len()
            || heads != offsets.len()
            || table_rows > u64::from(u32::MAX) + 1
        {
            return Err(OpsError::DimensionMismatch("ple hash metadata"));
        }
        if multipliers.contains(&0) || vocab_sizes.contains(&0) {
            return Err(OpsError::InvalidParameter("ple hash constants"));
        }
        let mut previous_end = 0;
        for (&offset, &vocab) in offsets.iter().zip(&vocab_sizes) {
            let end = offset
                .checked_add(vocab)
                .ok_or(OpsError::CapacityExceeded("ple table rows"))?;
            if offset < previous_end || end > table_rows {
                return Err(OpsError::InvalidParameter("ple table row ranges"));
            }
            previous_end = end;
        }
        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            heads_per_ngram,
            boundary_token,
            token_vocab_size,
        })
    }

    pub fn qwen38() -> Self {
        Self::new(
            vec![23703573157769, 20109073645365, 8052911324071],
            vec![
                20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
            ],
            vec![
                0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                300001275,
            ],
            8,
            248044,
            248320,
            320001536,
        )
        .expect("fixed Qwen3.8 PLE constants are valid")
    }

    pub fn history_len(&self) -> usize {
        self.multipliers.len() - 1
    }
    /// Preserve exact model metadata for an allocation-free guarded caller.
    /// The existing scalar rows/history API and arithmetic remain unchanged.
    pub fn metadata(&self) -> PleHashMetadata<'_> {
        PleHashMetadata {
            multipliers: &self.multipliers,
            vocab_sizes: &self.vocab_sizes,
            offsets: &self.offsets,
            heads_per_ngram: self.heads_per_ngram,
            boundary_token: self.boundary_token,
            token_vocab_size: self.token_vocab_size,
        }
    }

    /// Previous tokens are oldest-first. The current token's own EOS does not
    /// reset its predecessors; a predecessor EOS/missing value cuts older ones.
    pub fn rows(&self, current: i32, history: &PleTokenHistory) -> Result<Vec<u32>, OpsError> {
        if current < 0 || current as u32 >= self.token_vocab_size {
            return Err(OpsError::InvalidParameter("ple current token"));
        }
        if history.tokens.len() != self.history_len() {
            return Err(OpsError::DimensionMismatch("ple token history"));
        }
        if history
            .tokens
            .iter()
            .any(|&v| v >= 0 && v as u32 >= self.token_vocab_size)
        {
            return Err(OpsError::InvalidParameter("ple predecessor token"));
        }
        let mut context = [0_u64; 16];
        context[0] = current as u64;
        let mut cut = false;
        for (distance, output) in context
            .iter_mut()
            .enumerate()
            .take(self.multipliers.len())
            .skip(1)
        {
            let previous = history.tokens[self.history_len() - distance];
            cut |= previous < 0 || previous == self.boundary_token;
            *output = if cut {
                self.boundary_token as u64
            } else {
                previous as u64
            };
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.vocab_sizes.len())
            .map_err(|_| OpsError::AllocationFailed("ple rows"))?;
        let mut mixed = context[0].wrapping_mul(self.multipliers[0]);
        for (distance, (&token, &mult)) in context.iter().zip(&self.multipliers).enumerate().skip(1)
        {
            mixed ^= token.wrapping_mul(mult);
            let base = (distance - 1) * self.heads_per_ngram;
            for h in base..base + self.heads_per_ngram {
                rows.push((mixed % self.vocab_sizes[h] + self.offsets[h]) as u32);
            }
        }
        Ok(rows)
    }

    pub fn rows_and_advance(
        &self,
        current: i32,
        history: &mut PleTokenHistory,
    ) -> Result<Vec<u32>, OpsError> {
        let rows = self.rows(current, history)?;
        history.tokens.copy_within(1.., 0);
        let last = history.tokens.len() - 1;
        history.tokens[last] = current;
        Ok(rows)
    }
}

pub struct PleTokenHistory {
    tokens: Vec<i32>,
}

impl PleTokenHistory {
    pub fn new(config: &PleHashConfig) -> Self {
        Self {
            tokens: vec![-1; config.history_len()],
        }
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub fn reset(&mut self) {
        clear_i32_capacity(&mut self.tokens);
        self.tokens.fill(-1);
    }
}

impl Drop for PleTokenHistory {
    fn drop(&mut self) {
        clear_i32_capacity(&mut self.tokens);
    }
}

pub struct PleOutput {
    pub residual: Vec<f32>,
    pub gates: Vec<f32>,
    pub normalized: Vec<f32>,
}

#[path = "ops_ple_transition.rs"]
pub mod ple_transition;

/// PLE projections are performed by the graph. This operator receives projected
/// key/value and stores only normalized gated values in its dilated convolution.
#[allow(clippy::too_many_arguments)]
pub fn ple_inject_projected(
    key: &[f32],
    value: &[f32],
    residual: &[f32],
    key_gamma: &[f32],
    query_gamma: &[f32],
    conv_gamma: &[f32],
    conv_weights: &[f32],
    conv_state: &mut CausalConvState,
    shape: GatedResidualShape,
    epsilon: f32,
) -> Result<PleOutput, OpsError> {
    let len = product(shape.streams, shape.width, "ple residual")?;
    GatedResidualShape::new(shape.streams, shape.width)?;
    if key.len() != len
        || residual.len() != len
        || value.len() != shape.width
        || conv_state.channels != len
    {
        return Err(OpsError::DimensionMismatch("ple projections"));
    }
    finite(value, "ple value")?;
    let key_norm = rms_norm(key, key_gamma, shape.width, epsilon)?;
    let query_norm = rms_norm(residual, query_gamma, shape.width, epsilon)?;
    let mut gates = zeros(shape.streams, "ple gates")?;
    let mut gated = zeros(len, "ple gated value")?;
    for (c, gate) in gates.iter_mut().enumerate() {
        let dot = (0..shape.width)
            .map(|d| {
                f64::from(key_norm[c * shape.width + d])
                    * f64::from(query_norm[c * shape.width + d])
            })
            .sum::<f64>()
            / (shape.width as f64).sqrt();
        let signed_root = if dot == 0.0 {
            0.0
        } else {
            dot.signum() * dot.abs().max(1e-6).sqrt()
        };
        *gate = sigmoid(checked_f32(signed_root, "ple signed root")?);
        for (d, &v) in value.iter().enumerate() {
            gated[c * shape.width + d] =
                checked_f32(f64::from(*gate) * f64::from(v), "ple gated value")?;
        }
    }
    let normalized = rms_norm(&gated, conv_gamma, shape.width, epsilon)?;
    let raw_conv = conv_state.preview(&normalized, conv_weights)?;
    let mut output = zeros(len, "ple output")?;
    for i in 0..len {
        output[i] = checked_f32(
            f64::from(residual[i]) + f64::from(gated[i]) + f64::from(silu(raw_conv[i])),
            "ple residual output",
        )?;
    }
    conv_state.advance(&normalized);
    Ok(PleOutput {
        residual: output,
        gates,
        normalized,
    })
}

/// Text-only NeoX RoPE: first rotary_dims, pairs `(i,i+rotary_dims/2)`.
/// Three-axis MRoPE reduces to this when every axis has the same text position.
pub fn rope_neox_heads(
    input: &mut [f32],
    head_dim: usize,
    rotary_dims: usize,
    position: u32,
    theta: f32,
) -> Result<(), OpsError> {
    if head_dim == 0
        || input.is_empty()
        || !input.len().is_multiple_of(head_dim)
        || rotary_dims > head_dim
        || !rotary_dims.is_multiple_of(2)
    {
        return Err(OpsError::InvalidShape("rope"));
    }
    if !theta.is_finite() || theta <= 1.0 || position > (1 << 24) {
        return Err(OpsError::InvalidParameter("rope position or theta"));
    }
    finite(input, "rope input")?;
    let half = rotary_dims / 2;
    for head in input.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let angle = position as f32 * theta.powf(-(i as f32) / half as f32);
            let (sin, cos) = angle.sin_cos();
            let x = head[i];
            let y = head[i + half];
            head[i] = checked_f32(
                f64::from(x) * f64::from(cos) - f64::from(y) * f64::from(sin),
                "rope result",
            )?;
            head[i + half] = checked_f32(
                f64::from(x) * f64::from(sin) + f64::from(y) * f64::from(cos),
                "rope result",
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct QsaShape {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub indexer_heads: usize,
    pub indexer_dim: usize,
    pub block_size: usize,
    pub token_budget: usize,
    pub rotary_dims: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}

impl QsaShape {
    pub fn qwen38() -> Self {
        Self {
            query_heads: 24,
            kv_heads: 2,
            head_dim: 256,
            indexer_heads: 4,
            indexer_dim: 128,
            block_size: 4,
            token_budget: 2048,
            rotary_dims: 64,
            rope_theta: 10_000_000.0,
            epsilon: 1e-6,
        }
    }

    pub fn validate(&self) -> Result<(), OpsError> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.indexer_heads == 0
            || self.indexer_dim == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.block_size == 0
            || self.token_budget == 0
            || !self.token_budget.is_multiple_of(self.block_size)
            || !self.rotary_dims.is_multiple_of(2)
            || self.rotary_dims > self.indexer_dim
            || self.rotary_dims > self.head_dim
        {
            return Err(OpsError::InvalidShape("qsa"));
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(OpsError::InvalidParameter("qsa"));
        }
        product(self.query_heads, self.head_dim, "qsa query")?;
        product(self.kv_heads, self.head_dim, "qsa kv")?;
        product(self.indexer_heads, self.indexer_dim, "qsa indexer")?;
        Ok(())
    }
}

/// Score only completed blocks within this query's causal visible prefix.
/// Query indexer heads arrive already RMS-normalized and RoPE-rotated.
pub fn qsa_block_scores(
    raw_indexer_keys: &[f32],
    normalized_query: &[f32],
    key_gamma: &[f32],
    positions: &[u32],
    visible_tokens: usize,
    shape: QsaShape,
) -> Result<Vec<f32>, OpsError> {
    shape.validate()?;
    if visible_tokens == 0
        || visible_tokens > positions.len()
        || raw_indexer_keys.len() != product(positions.len(), shape.indexer_dim, "qsa raw keys")?
        || normalized_query.len()
            != product(shape.indexer_heads, shape.indexer_dim, "qsa indexer query")?
        || key_gamma.len() != shape.indexer_dim
    {
        return Err(OpsError::DimensionMismatch("qsa block scores"));
    }
    if positions[..visible_tokens].windows(2).any(|w| w[0] >= w[1]) {
        return Err(OpsError::InvalidParameter("qsa position order"));
    }
    finite(
        &raw_indexer_keys[..visible_tokens * shape.indexer_dim],
        "qsa raw indexer keys",
    )?;
    finite(normalized_query, "qsa indexer query")?;
    let complete = visible_tokens / shape.block_size;
    let mut scores = zeros(complete, "qsa block scores")?;
    let mut mean = zeros(shape.indexer_dim, "qsa pooled key")?;
    for (b, score) in scores.iter_mut().enumerate() {
        for (d, output) in mean.iter_mut().enumerate() {
            let sum = (0..shape.block_size)
                .map(|member| {
                    f64::from(
                        raw_indexer_keys[((b * shape.block_size + member) * shape.indexer_dim) + d],
                    )
                })
                .sum::<f64>();
            *output = checked_f32(sum / shape.block_size as f64, "qsa pooled mean")?;
        }
        let mut pooled = rms_norm(&mean, key_gamma, shape.indexer_dim, shape.epsilon)?;
        rope_neox_heads(
            &mut pooled,
            shape.indexer_dim,
            shape.rotary_dims,
            positions[b * shape.block_size],
            shape.rope_theta,
        )?;
        let mut sum = 0.0_f64;
        for query in normalized_query.chunks_exact(shape.indexer_dim) {
            let dot = query
                .iter()
                .zip(&pooled)
                .map(|(&q, &k)| f64::from(q) * f64::from(k))
                .sum::<f64>();
            sum += dot.max(0.0);
        }
        *score = checked_f32(sum / (shape.indexer_dim as f64).sqrt(), "qsa block score")?;
    }
    Ok(scores)
}

/// Complete-block top-k plus *exact* incomplete tail; sorted unique token IDs.
pub fn qsa_select_blocks(
    block_scores: &[f32],
    visible_tokens: usize,
    block_size: usize,
    token_budget: usize,
) -> Result<Vec<usize>, OpsError> {
    if block_size == 0 || token_budget == 0 || !token_budget.is_multiple_of(block_size) {
        return Err(OpsError::InvalidShape("qsa selection"));
    }
    if visible_tokens == 0 || block_scores.len() != visible_tokens / block_size {
        return Err(OpsError::DimensionMismatch("qsa selection scores"));
    }
    finite(block_scores, "qsa selection scores")?;
    if block_scores.iter().any(|&score| score < 0.0) {
        return Err(OpsError::InvalidParameter("qsa selection score sign"));
    }
    let mut blocks: Vec<usize> = (0..block_scores.len()).collect();
    blocks.sort_unstable_by(|&a, &b| {
        if block_scores[a] == block_scores[b] {
            // IEEE signed zeros are a mathematical tie, too.
            a.cmp(&b)
        } else {
            block_scores[b].total_cmp(&block_scores[a])
        }
    });
    blocks.truncate(token_budget / block_size);
    blocks.sort_unstable();
    let len = product(blocks.len(), block_size, "qsa selected tokens")?
        .checked_add(visible_tokens % block_size)
        .ok_or(OpsError::CapacityExceeded("qsa selected tokens"))?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(len)
        .map_err(|_| OpsError::AllocationFailed("qsa selection"))?;
    for b in blocks {
        ids.extend(b * block_size..(b + 1) * block_size);
    }
    ids.extend(visible_tokens / block_size * block_size..visible_tokens);
    Ok(ids)
}

pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, OpsError> {
    if logits.is_empty() {
        return Err(OpsError::InvalidShape("softmax"));
    }
    finite(logits, "softmax logits")?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = zeros(logits.len(), "softmax")?;
    let mut sum = 0.0_f64;
    for (&logit, result) in logits.iter().zip(&mut probabilities) {
        *result = (f64::from(logit) - f64::from(maximum)).exp() as f32;
        sum += f64::from(*result);
    }
    for probability in &mut probabilities {
        *probability = checked_f32(f64::from(*probability) / sum, "softmax probability")?;
    }
    Ok(probabilities)
}

pub fn router_topk(
    logits: &[f32],
    k: usize,
    normalize: bool,
) -> Result<Vec<(usize, f32)>, OpsError> {
    if k == 0 || k > logits.len() {
        return Err(OpsError::InvalidShape("router top k"));
    }
    let probabilities = softmax(logits)?;
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_unstable_by(|&a, &b| {
        probabilities[b]
            .total_cmp(&probabilities[a])
            .then(a.cmp(&b))
    });
    ids.truncate(k);
    let denominator = if normalize {
        ids.iter()
            .map(|&i| f64::from(probabilities[i]))
            .sum::<f64>()
            .max(2_f64.powi(-14))
    } else {
        1.0
    };
    Ok(ids
        .into_iter()
        .map(|i| (i, (f64::from(probabilities[i]) / denominator) as f32))
        .collect())
}

/// GQA over selected causal IDs. Q/K are already normed/rotated; V is raw.
/// Caches are token-major `[token][kv_head][head_dim]`. Query h uses KV h/G.
pub fn qsa_attend(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[usize],
    visible_tokens: usize,
    shape: QsaShape,
) -> Result<Vec<f32>, OpsError> {
    shape.validate()?;
    let kv_width = product(shape.kv_heads, shape.head_dim, "qsa kv width")?;
    let query_width = product(shape.query_heads, shape.head_dim, "qsa query width")?;
    if query.len() != query_width
        || keys.len() != values.len()
        || !keys.len().is_multiple_of(kv_width)
        || visible_tokens == 0
        || visible_tokens > keys.len() / kv_width
    {
        return Err(OpsError::DimensionMismatch("qsa attention"));
    }
    if selected.iter().any(|&id| id >= visible_tokens) || selected.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(OpsError::InvalidParameter("qsa selected causal ids"));
    }
    finite(query, "qsa attention query")?;
    finite(&keys[..visible_tokens * kv_width], "qsa attention keys")?;
    finite(&values[..visible_tokens * kv_width], "qsa attention values")?;
    let mut output = zeros(query_width, "qsa attention output")?;
    if selected.is_empty() {
        return Ok(output);
    }
    let mut logits = zeros(selected.len(), "qsa attention logits")?;
    let group = shape.query_heads / shape.kv_heads;
    for h in 0..shape.query_heads {
        let kh = h / group;
        for (n, &id) in selected.iter().enumerate() {
            let dot = (0..shape.head_dim)
                .map(|d| {
                    f64::from(query[h * shape.head_dim + d])
                        * f64::from(keys[id * kv_width + kh * shape.head_dim + d])
                })
                .sum::<f64>();
            logits[n] = checked_f32(dot / (shape.head_dim as f64).sqrt(), "qsa attention logit")?;
        }
        let weights = softmax(&logits)?;
        for d in 0..shape.head_dim {
            let sum = selected
                .iter()
                .zip(&weights)
                .map(|(&id, &weight)| {
                    f64::from(weight) * f64::from(values[id * kv_width + kh * shape.head_dim + d])
                })
                .sum::<f64>();
            output[h * shape.head_dim + d] = checked_f32(sum, "qsa attention output")?;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 2e-5 * e.abs().max(1.0), "{a} != {e}");
        }
    }

    #[test]
    fn grouped_norm_uses_direct_gamma_and_per_stream_mean() {
        let out = rms_norm(&[3.0, 4.0, 0.0, 5.0], &[2.0, 1.0, 1.0, 3.0], 2, 1e-6).unwrap();
        let inv = 1.0 / (12.5_f32 + 1e-6).sqrt();
        near(&out, &[6.0 * inv, 4.0 * inv, 0.0, 15.0 * inv]);
        near(
            &rms_norm(&[3.0, 4.0, 0.0, 5.0], &[1.0, 1.0], 2, 1e-6).unwrap(),
            &[3.0 * inv, 4.0 * inv, 0.0, 5.0 * inv],
        );
    }

    #[test]
    fn norm_rejects_bad_shapes_nonfinite_and_epsilon() {
        assert!(rms_norm(&[1.0], &[1.0], 0, 1e-6).is_err());
        assert!(rms_norm(&[1.0], &[1.0], 1, 0.0).is_err());
        assert!(rms_norm(&[f32::NAN], &[1.0], 1, 1e-6).is_err());
        assert!(rms_norm(&[1.0], &[1.0, 1.0], 1, 1e-6).is_err());
    }

    #[test]
    fn l2_uses_sum_not_mean() {
        near(&l2_norm_heads(&[3.0, 4.0], 2, 1e-6).unwrap(), &[0.6, 0.8]);
    }

    #[test]
    fn gr_zero_gates_mix_and_inject_original_residual() {
        let shape = GatedResidualShape::new(2, 2).unwrap();
        near(
            &shape.mix(&[1.0, 2.0, 3.0, 6.0], &[0.0; 4]).unwrap(),
            &[1.0, 2.0],
        );
        near(
            &shape
                .combine(&[10.0, 20.0, 30.0, 40.0], &[2.0, 3.0], &[0.0; 2])
                .unwrap(),
            &[12.0, 23.0, 32.0, 43.0],
        );
    }

    #[test]
    fn gr_division_is_inside_silu_and_injection_sigmoid() {
        let shape = GatedResidualShape::new(4, 1).unwrap();
        near(&shape.activate_down(&[8.0]).unwrap(), &[silu(2.0)]);
        assert!((shape.activate_down(&[8.0]).unwrap()[0] - silu(8.0) / 4.0).abs() > 0.1);
        let out = shape.combine(&[0.0; 4], &[1.0], &[4.0; 4]).unwrap();
        near(&out, &[2.0 * sigmoid(1.0); 4]);
    }

    #[test]
    fn conv_orders_taps_and_keeps_raw_history() {
        let mut state = CausalConvState::new(1, 4, 1).unwrap();
        for v in 1..=3 {
            state.step(&[v as f32], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        near(&state.step(&[4.0], &[1.0, 2.0, 3.0, 4.0]).unwrap(), &[30.0]);
        assert_eq!(state.history(), &[2.0, 3.0, 4.0]);
        state.reset();
        assert_eq!(state.history(), &[0.0; 3]);
    }

    #[test]
    fn dilated_conv_reads_lags_nine_six_three_zero() {
        let mut state = CausalConvState::new(1, 4, 3).unwrap();
        for v in 1..=9 {
            state.step(&[v as f32], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        near(
            &state.step(&[10.0], &[1.0, 2.0, 3.0, 4.0]).unwrap(),
            &[70.0],
        );
    }

    #[test]
    fn conv_errors_do_not_advance_history_and_budget_is_checked() {
        let mut state = CausalConvState::new(1, 2, 1).unwrap();
        state.step(&[3.0], &[1.0, 1.0]).unwrap();
        assert!(state.step(&[f32::NAN], &[1.0, 1.0]).is_err());
        assert_eq!(state.history(), &[3.0]);
        assert!(CausalConvState::new_bounded(10240, 4, 3, 100).is_err());
        assert!(CausalConvState::new(usize::MAX, 4, 3).is_err());
    }

    #[test]
    fn gdn_decays_before_correction_and_reads_updated_state() {
        let mut state = GdnState::new(GdnShape::new(2, 1, 1).unwrap()).unwrap();
        state.set_values(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let scale = 1.0 / 2.0_f32.sqrt();
        near(
            &state
                .step(
                    &[scale, 0.0],
                    &[1.0, 0.0],
                    &[1.0, 2.0],
                    &[0.25_f32.ln()],
                    &[0.5],
                )
                .unwrap(),
            &[0.625 * scale, 1.25 * scale],
        );
        near(state.values(), &[0.625, 1.25, 0.75, 1.0]);
    }

    #[test]
    fn gdn_pairs_tiled_heads_by_modulo() {
        let mut state = GdnState::new(GdnShape::new(1, 2, 4).unwrap()).unwrap();
        near(
            &state
                .step(&[1.0, 2.0], &[1.0, 3.0], &[1.0; 4], &[0.0; 4], &[1.0; 4])
                .unwrap(),
            &[1.0, 6.0, 1.0, 6.0],
        );
    }

    #[test]
    fn gdn_carried_state_matches_independent_two_step_oracle() {
        let shape = GdnShape::new(2, 1, 1).unwrap();
        let mut continuous = GdnState::new(shape).unwrap();
        continuous.set_values(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let scale = 1.0 / 2.0_f32.sqrt();
        continuous
            .step(
                &[scale, 0.0],
                &[1.0, 0.0],
                &[1.0, 2.0],
                &[0.25_f32.ln()],
                &[0.5],
            )
            .unwrap();
        let mut chunked = GdnState::new(shape).unwrap();
        chunked.set_values(&[0.625, 1.25, 0.75, 1.0]).unwrap();
        // Independent second-step arithmetic: decay top row to .15625,.3125;
        // beta corrections .921875,.34375 give 1.078125,.65625.
        for state in [&mut continuous, &mut chunked] {
            let output = state
                .step(
                    &[scale, 0.0],
                    &[1.0, 0.0],
                    &[2.0, 1.0],
                    &[0.25_f32.ln()],
                    &[0.5],
                )
                .unwrap();
            near(&output, &[1.078125 * scale, 0.65625 * scale]);
            near(state.values(), &[1.078125, 0.65625, 0.1875, 0.25]);
        }
        near(continuous.values(), chunked.values());
        continuous.reset();
        assert!(continuous.values().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gdn_rejects_growth_invalid_beta_and_overflow_without_commit() {
        let mut state = GdnState::new(GdnShape::new(1, 1, 1).unwrap()).unwrap();
        state.set_values(&[2.0]).unwrap();
        assert!(state.step(&[1.0], &[1.0], &[1.0], &[0.1], &[0.5]).is_err());
        assert!(state.step(&[1.0], &[1.0], &[1.0], &[-0.1], &[2.0]).is_err());
        assert!(
            state
                .step(&[f32::MAX], &[f32::MAX], &[f32::MAX], &[0.0], &[1.0])
                .is_err()
        );
        assert_eq!(state.values(), &[2.0]);
        assert!(GdnState::new_bounded(GdnShape::qwen38(), 1).is_err());
    }

    #[test]
    fn gdn_stored_a_is_already_negative_exp_and_output_gate_is_sigmoid() {
        near(
            &gdn_log_decay(&[0.0], &[0.0], &[-2.0]).unwrap(),
            &[-2.0 * 2.0_f32.ln()],
        );
        assert!(gdn_log_decay(&[0.0], &[0.0], &[2.0]).is_err());
        near(
            &gdn_output_gate(&[2.0], &[0.0], &[1.0], 1, 1e-6).unwrap(),
            &[0.5],
        );
    }

    #[test]
    fn ple_borrowed_metadata_is_original_without_allocation_or_conversion() {
        let cfg = PleHashConfig::qwen38();
        let metadata = cfg.metadata();
        assert_eq!(metadata.multipliers.as_ptr(), cfg.multipliers.as_ptr());
        assert_eq!(metadata.vocab_sizes.as_ptr(), cfg.vocab_sizes.as_ptr());
        assert_eq!(metadata.offsets.as_ptr(), cfg.offsets.as_ptr());
        assert_eq!(metadata.multipliers.len(), 3);
        assert_eq!(metadata.vocab_sizes.len(), 16);
        assert_eq!(metadata.heads_per_ngram, 8);
        assert_eq!(metadata.boundary_token, 248044);
        assert_eq!(metadata.token_vocab_size, 248320);
        let history = PleTokenHistory::new(&cfg);
        let before = cfg.rows(42, &history).unwrap();
        let _borrow = cfg.metadata();
        assert_eq!(cfg.rows(42, &history).unwrap(), before);
        assert_eq!(history.tokens(), &[-1, -1]);
    }

    #[test]
    fn ple_actual_rows_are_exact_and_predecessors_are_oldest_first() {
        let cfg = PleHashConfig::qwen38();
        let mut history = PleTokenHistory::new(&cfg);
        cfg.rows_and_advance(3, &mut history).unwrap();
        cfg.rows_and_advance(7, &mut history).unwrap();
        assert_eq!(
            cfg.rows(42, &history).unwrap(),
            vec![
                14120497, 39300209, 51890715, 73518158, 80628040, 109664886, 133220230, 151294224,
                163952504, 198468448, 205404373, 233797536, 251057218, 268317062, 280737099,
                302837232
            ]
        );
        assert_eq!(history.tokens(), &[3, 7]);
        history.reset();
        assert_eq!(history.tokens(), &[-1, -1]);
    }

    #[test]
    fn ple_eos_missing_zero_and_current_eos_are_distinct() {
        let cfg = PleHashConfig::qwen38();
        let initial = PleTokenHistory::new(&cfg);
        let missing = cfg.rows(42, &initial).unwrap();
        let mut history = PleTokenHistory::new(&cfg);
        cfg.rows_and_advance(3, &mut history).unwrap();
        cfg.rows_and_advance(248044, &mut history).unwrap();
        assert_eq!(cfg.rows(42, &history).unwrap(), missing);
        history.reset();
        cfg.rows_and_advance(3, &mut history).unwrap();
        cfg.rows_and_advance(7, &mut history).unwrap();
        assert_ne!(cfg.rows(248044, &history).unwrap(), missing);
        assert_eq!(cfg.rows(248044, &history).unwrap()[0], 10204458);
        history.reset();
        cfg.rows_and_advance(0, &mut history).unwrap();
        cfg.rows_and_advance(7, &mut history).unwrap();
        assert_eq!(cfg.rows(42, &history).unwrap()[8], 160331325);
        assert!(cfg.rows(-1, &history).is_err());
        assert!(cfg.rows(248320, &history).is_err());
    }

    #[test]
    fn ple_u64_products_wrap_and_xor_precedes_modulo() {
        let cfg = PleHashConfig::new(
            vec![u64::MAX, u64::MAX - 2],
            vec![97],
            vec![0],
            1,
            9,
            10,
            97,
        )
        .unwrap();
        let mut history = PleTokenHistory::new(&cfg);
        cfg.rows_and_advance(3, &mut history).unwrap();
        let expected = 7_u64.wrapping_mul(u64::MAX) ^ 3_u64.wrapping_mul(u64::MAX - 2);
        assert_eq!(cfg.rows(7, &history).unwrap(), vec![(expected % 97) as u32]);
    }

    #[test]
    fn ple_zero_dot_gate_and_normalized_conv_history() {
        let shape = GatedResidualShape::new(2, 2).unwrap();
        let mut state = CausalConvState::new(4, 4, 3).unwrap();
        let output = ple_inject_projected(
            &[0.0; 4],
            &[6.0, 8.0],
            &[0.0; 4],
            &[1.0; 4],
            &[1.0; 4],
            &[1.0; 4],
            &[0.0; 16],
            &mut state,
            shape,
            1e-6,
        )
        .unwrap();
        assert_eq!(output.gates, vec![0.5; 2]);
        near(&output.residual, &[3.0, 4.0, 3.0, 4.0]);
        let inv = 1.0 / (12.5_f32 + 1e-6).sqrt();
        near(
            &output.normalized,
            &[3.0 * inv, 4.0 * inv, 3.0 * inv, 4.0 * inv],
        );
        for c in 0..4 {
            near(&[state.history()[c * 9 + 8]], &[output.normalized[c]]);
        }
    }

    fn toy_qsa() -> QsaShape {
        QsaShape {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 2,
            indexer_heads: 1,
            indexer_dim: 2,
            block_size: 2,
            token_budget: 2,
            rotary_dims: 0,
            rope_theta: 10_000.0,
            epsilon: 1e-6,
        }
    }

    #[test]
    fn qsa_blocks_keep_exact_tail_and_causal_pooling() {
        let keys = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 100.0, 0.0];
        let positions = [0, 1, 2, 3, 4];
        let scores =
            qsa_block_scores(&keys, &[1.0, 0.0], &[1.0, 1.0], &positions, 5, toy_qsa()).unwrap();
        assert!(scores[0] > scores[1]);
        assert_eq!(qsa_select_blocks(&scores, 5, 2, 2).unwrap(), vec![0, 1, 4]);
        let prefix =
            qsa_block_scores(&keys, &[1.0, 0.0], &[1.0, 1.0], &positions, 3, toy_qsa()).unwrap();
        assert_eq!(prefix.len(), 1);
        assert_eq!(qsa_select_blocks(&prefix, 3, 2, 2).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn qsa_budget_boundaries_never_select_partial_complete_blocks() {
        for visible in 2051..=2056 {
            let scores: Vec<f32> = (0..visible / 4).map(|b| b as f32).collect();
            let ids = qsa_select_blocks(&scores, visible, 4, 2048).unwrap();
            assert_eq!(ids.len(), 2048 + visible % 4);
            assert!(ids.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(
                ids[ids.len() - visible % 4..],
                (visible / 4 * 4..visible).collect::<Vec<_>>()
            );
            for chunk in ids[..2048].chunks_exact(4) {
                assert_eq!(chunk, &[chunk[0], chunk[0] + 1, chunk[0] + 2, chunk[0] + 3]);
                assert_eq!(chunk[0] % 4, 0);
            }
        }
    }

    #[test]
    fn qsa_scores_rectify_each_head_then_sum_and_ties_are_deterministic() {
        let mut shape = toy_qsa();
        shape.indexer_heads = 4;
        let scores = qsa_block_scores(
            &[1.0, 0.0, 1.0, 0.0],
            &[2.0, 0.0, -3.0, 0.0, 1.0, 0.0, -4.0, 0.0],
            &[1.0, 1.0],
            &[0, 1],
            2,
            shape,
        )
        .unwrap();
        near(&scores, &[3.0]);
        assert_eq!(
            qsa_select_blocks(&[1.0, 1.0, 1.0], 6, 2, 2).unwrap(),
            vec![0, 1]
        );
        assert_eq!(
            qsa_select_blocks(&[-0.0, 0.0], 4, 2, 2).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn qsa_pools_raw_keys_before_norm_and_rotates_at_first_position() {
        let raw = [3.0, 4.0, 0.0, 2.0];
        let gamma = [1.0, 1.0];
        let positions = [3, 4];
        let mut shape = toy_qsa();
        let score = qsa_block_scores(&raw, &[0.0, 1.0], &gamma, &positions, 2, shape).unwrap();
        let inv = 1.0 / (5.625_f32 + 1e-6).sqrt();
        near(&score, &[3.0 * inv / 2.0_f32.sqrt()]);
        shape.rotary_dims = 2;
        let rotated = qsa_block_scores(&raw, &[-1.0, 0.0], &gamma, &positions, 2, shape).unwrap();
        let first_rotated = (1.5 * 3.0_f32.cos() - 3.0 * 3.0_f32.sin()) * inv;
        near(&rotated, &[(-first_rotated).max(0.0) / 2.0_f32.sqrt()]);
        assert!(rotated[0] > 0.0);
    }

    #[test]
    fn qsa_gqa_pairs_by_division_and_rejects_future_ids() {
        let shape = toy_qsa();
        let out = qsa_attend(
            &[0.0; 8],
            &[0.0; 8],
            &[1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
            &[0, 1],
            2,
            shape,
        )
        .unwrap();
        near(&out, &[2.0, 3.0, 2.0, 3.0, 20.0, 30.0, 20.0, 30.0]);
        assert!(qsa_attend(&[0.0; 8], &[0.0; 8], &[0.0; 8], &[1], 1, shape).is_err());
        assert!(qsa_attend(&[0.0; 8], &[0.0; 8], &[0.0; 8], &[1, 0], 2, shape).is_err());
    }

    #[test]
    fn rope_neox_half_pairing_partial_dims_and_zero_position() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 50.0, 60.0];
        let mut zero = original.clone();
        rope_neox_heads(&mut zero, 6, 4, 0, 10_000.0).unwrap();
        assert_eq!(zero, original);
        let mut rotated = original.clone();
        rope_neox_heads(&mut rotated, 6, 4, 1, 10_000.0).unwrap();
        near(
            &rotated,
            &[
                1.0 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
                2.0 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin(),
                1.0 * 1.0_f32.sin() + 3.0 * 1.0_f32.cos(),
                2.0 * 0.01_f32.sin() + 4.0 * 0.01_f32.cos(),
                50.0,
                60.0,
            ],
        );
    }

    #[test]
    fn router_global_softmax_topk_and_renormalization() {
        let routed = router_topk(&[0.0; 512], 10, true).unwrap();
        assert_eq!(
            routed.iter().map(|&(i, _)| i).collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
        near(
            &routed.iter().map(|&(_, w)| w).collect::<Vec<_>>(),
            &[0.1; 10],
        );
        let raw = router_topk(&[0.0, 1.0, 2.0], 2, false).unwrap();
        assert_eq!(raw[0].0, 2);
        assert!(raw.iter().map(|&(_, p)| p).sum::<f32>() < 1.0);
        let normalized = router_topk(&[0.0, 1.0, 2.0], 2, true).unwrap();
        near(&[normalized.iter().map(|&(_, p)| p).sum::<f32>()], &[1.0]);
    }

    #[test]
    fn activations_are_stable_on_large_finite_values() {
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(softplus(1000.0), 1000.0);
        assert_eq!(softplus(-1000.0), 0.0);
        near(&softmax(&[f32::MAX, f32::MAX]).unwrap(), &[0.5, 0.5]);
        assert!(softmax(&[f32::INFINITY]).is_err());
    }
}
