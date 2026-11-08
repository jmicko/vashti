//! Stateless Rust/CubeCL GDN preprocessing and closing-gate prototypes.
//!
//! Oracle: kuno-text ops.rs/model.rs at
//! f8b2d2d5442f65cc772503074e1ab6670d3d70ce. No graph hook or hidden state.
//! A separate sequence owner supplies all buffers, keeps committed/candidate
//! convolution rings and recurrent states DISTINCT, and poisons the WHOLE
//! token on any status/validation/cancellation/device failure. Commit neither
//! candidate until preprocessing, recurrence, closing gate and token-wide
//! status checks succeed. Clear all transient candidates on failure/drop.
//! These kernels never commit state, retain handles, log data, or run devices.
//! The owner must reset the entire status arena to zero for each token attempt
//! and await successful device completion; stale success bits are not evidence
//! that a cancelled/failed/mis-sized dispatch completed. Launch-length helpers
//! below support resident buffers without downloading private activations.
//! Every mutable output region must be disjoint from every input/output region
//! in the same dispatch, not just the old/next rings; the kernels initialize
//! outputs before all input loads have completed. Validate real handle regions.
//!
//! All arrays use scalar f32/u32, CubeDim=(64,1,1). Conv uses ceil(channels/64)
//! cubes; QKV uses 2*key_heads+value_heads cubes; controls uses ceil(VH/64);
//! closing gate uses value_heads cubes. Status is one u32 per logical output,
//! initialized to 0 and set to 1 only for a finite valid result. Inspect every
//! status, including recurrence's, before accepting the token; partial success
//! must never make partially updated history reusable.
//!
//! Ring format matches the core's channel-major oldest-first history: three
//! RAW QKV samples per channel. Conv weights are [channel][oldest..current].
//! This explicit next-ring shift needs no cursor and is mathematically the
//! same three-slot ring transition. QKV output remains flat [Q,K,V]; views
//! start at 0, qk_elements, 2*qk_elements respectively. Q alone receives a
//! separate f32 division by sqrt(128), after its L2 normalization.
//! Controls are flat [all log_decay, all sigmoid(beta_logits)]; model params
//! are [all dt_bias, all stored_negative_A]. Stored A is already -exp(A_log).
//! Closing gamma is already folded, shared128 or per-value; gate is sigmoid.
//!
//! CPU convolution/norm reductions use f64; this portable kernel uses f32
//! lane/tree reductions, so numerical/logit acceptance needs tolerances.
//! Intermediate f32 overflow deliberately reports failure, even where the
//! higher-precision oracle could cancel it. Exp/log/subnormal/FMA behavior can
//! differ by device. CubeCL 0.10's WGSL Log1p is log(1+x), so softplus uses an
//! explicit compensated log1p instead of that inaccurate small-tail lowering.
//! Host/offline tests below cannot establish GPU parity or performance.

use cubecl::prelude::*;
use kuno_text::ops::GdnShape;

pub const LANES: usize = 64;
pub const HEAD_DIM: usize = 128;
pub const EPSILON: f32 = 1e-6;
pub const ORACLE_SHA: &str = "f8b2d2d5442f65cc772503074e1ab6670d3d70ce";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointwiseLayout {
    pub shape: GdnShape,
    pub channels: usize,
    pub qk_elements: usize,
    pub value_elements: usize,
    pub ring_elements: usize,
    pub conv_weight_elements: usize,
    pub control_elements: usize,
    pub qkv_heads: usize,
    pub conv_cubes: u32,
    pub control_cubes: u32,
}

impl PointwiseLayout {
    /// Fixed 128-wide Qwen operator family; smaller head counts support tests.
    /// Recompute this layout before trusting its public fields or array sizes.
    pub fn new(shape: GdnShape) -> Result<Self, &'static str> {
        GdnShape::new(shape.head_dim, shape.key_heads, shape.value_heads)
            .map_err(|_| "invalid GDN pointwise shape")?;
        if shape.head_dim != HEAD_DIM || shape.key_heads > 16 || shape.value_heads > 48 {
            return Err("pointwise prototype requires width128 and bounded Qwen head counts");
        }
        let qkv_heads = shape
            .key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(shape.value_heads))
            .ok_or("GDN head arithmetic overflow")?;
        let channels = qkv_heads
            .checked_mul(HEAD_DIM)
            .ok_or("GDN channel arithmetic overflow")?;
        let qk_elements = shape.key_heads * HEAD_DIM;
        let value_elements = shape.value_heads * HEAD_DIM;
        let ring_elements = channels
            .checked_mul(3)
            .ok_or("GDN ring arithmetic overflow")?;
        let conv_weight_elements = channels
            .checked_mul(4)
            .ok_or("GDN conv arithmetic overflow")?;
        let control_elements = shape
            .value_heads
            .checked_mul(2)
            .ok_or("GDN control arithmetic overflow")?;
        if conv_weight_elements > u32::MAX as usize / 4 || qkv_heads > 65_535 {
            return Err("GDN pointwise exceeds portable byte/grid bounds");
        }
        Ok(Self {
            shape,
            channels,
            qk_elements,
            value_elements,
            ring_elements,
            conv_weight_elements,
            control_elements,
            qkv_heads,
            conv_cubes: channels.div_ceil(LANES) as u32,
            control_cubes: shape.value_heads.div_ceil(LANES) as u32,
        })
    }
    fn check(self) -> Result<(), &'static str> {
        if self != Self::new(self.shape)? {
            return Err("GDN pointwise layout was modified");
        }
        Ok(())
    }
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}
fn epsilon_valid(epsilon: f32) -> bool {
    epsilon.is_finite() && epsilon > 0.0
}

/// Buffer identities/regions must additionally prove old/next ring nonaliasing.
#[allow(clippy::too_many_arguments)]
pub fn validate_conv_lengths(
    layout: PointwiseLayout,
    input: usize,
    weights: usize,
    old_ring: usize,
    next_ring: usize,
    activation: usize,
    status: usize,
) -> Result<(), &'static str> {
    layout.check()?;
    if input != layout.channels
        || weights != layout.conv_weight_elements
        || old_ring != layout.ring_elements
        || next_ring != layout.ring_elements
        || activation != layout.channels
        || status != layout.channels
    {
        return Err("GDN conv buffer length mismatch");
    }
    Ok(())
}

pub fn validate_qkv_lengths(
    layout: PointwiseLayout,
    activation: usize,
    prepared: usize,
    status: usize,
    epsilon: f32,
) -> Result<(), &'static str> {
    layout.check()?;
    if activation != layout.channels
        || prepared != layout.channels
        || status != layout.channels
        || !epsilon_valid(epsilon)
    {
        return Err("GDN QKV buffer length/epsilon mismatch");
    }
    Ok(())
}

pub fn validate_control_lengths(
    layout: PointwiseLayout,
    alpha: usize,
    beta_logits: usize,
    params: usize,
    controls: usize,
    status: usize,
) -> Result<(), &'static str> {
    layout.check()?;
    let heads = layout.shape.value_heads;
    if alpha != heads
        || beta_logits != heads
        || params != layout.control_elements
        || controls != layout.control_elements
        || status != heads
    {
        return Err("GDN control buffer length mismatch");
    }
    Ok(())
}

/// Returns the shared_gamma compile-time argument for the validated gamma size.
#[allow(clippy::too_many_arguments)]
pub fn validate_output_lengths(
    layout: PointwiseLayout,
    output: usize,
    z: usize,
    gamma: usize,
    gated: usize,
    status: usize,
    epsilon: f32,
) -> Result<bool, &'static str> {
    layout.check()?;
    if output != layout.value_elements
        || z != layout.value_elements
        || gated != layout.value_elements
        || status != layout.value_elements
        || (gamma != HEAD_DIM && gamma != layout.value_elements)
        || !epsilon_valid(epsilon)
    {
        return Err("GDN closing buffer length/epsilon mismatch");
    }
    Ok(gamma == HEAD_DIM)
}

pub fn validate_conv_inputs(
    layout: PointwiseLayout,
    input: &[f32],
    weights: &[f32],
    old_ring: &[f32],
) -> Result<(), &'static str> {
    layout.check()?;
    if input.len() != layout.channels
        || weights.len() != layout.conv_weight_elements
        || old_ring.len() != layout.ring_elements
    {
        return Err("GDN conv input length mismatch");
    }
    if !finite(input) || !finite(weights) || !finite(old_ring) {
        return Err("GDN conv non-finite input");
    }
    Ok(())
}

pub fn validate_qkv_inputs(
    layout: PointwiseLayout,
    activation: &[f32],
    epsilon: f32,
) -> Result<(), &'static str> {
    layout.check()?;
    if activation.len() != layout.channels || !epsilon_valid(epsilon) {
        return Err("GDN QKV length/epsilon mismatch");
    }
    if !finite(activation) {
        return Err("GDN QKV non-finite input");
    }
    Ok(())
}

pub fn validate_control_inputs(
    layout: PointwiseLayout,
    alpha: &[f32],
    beta_logits: &[f32],
    dt_and_a: &[f32],
) -> Result<(), &'static str> {
    layout.check()?;
    let h = layout.shape.value_heads;
    if alpha.len() != h || beta_logits.len() != h || dt_and_a.len() != layout.control_elements {
        return Err("GDN control input length mismatch");
    }
    if !finite(alpha)
        || !finite(beta_logits)
        || !finite(dt_and_a)
        || dt_and_a[h..].iter().any(|&a| a >= 0.0)
    {
        return Err("GDN invalid control input/A sign");
    }
    Ok(())
}

pub fn validate_output_inputs(
    layout: PointwiseLayout,
    output: &[f32],
    z: &[f32],
    gamma: &[f32],
    epsilon: f32,
) -> Result<(), &'static str> {
    layout.check()?;
    if output.len() != layout.value_elements
        || z.len() != layout.value_elements
        || (gamma.len() != HEAD_DIM && gamma.len() != layout.value_elements)
        || !epsilon_valid(epsilon)
    {
        return Err("GDN closing input length/epsilon mismatch");
    }
    if !finite(output) || !finite(z) || !finite(gamma) {
        return Err("GDN closing non-finite input");
    }
    Ok(())
}

fn status_valid(status: &[u32], expected: usize) -> bool {
    status.len() == expected && status.iter().all(|&s| s == 1)
}

/// This is the token-wide acceptance gate, not a state commit. Any Err must
/// poison the owning token/sequence and forbid reuse of all candidate states.
pub fn validate_token_status(
    layout: PointwiseLayout,
    conv: &[u32],
    qkv: &[u32],
    controls: &[u32],
    recurrence: &[u32],
    closing: &[u32],
) -> Result<(), &'static str> {
    layout.check()?;
    if !status_valid(conv, layout.channels)
        || !status_valid(qkv, layout.channels)
        || !status_valid(controls, layout.shape.value_heads)
        || !status_valid(recurrence, layout.value_elements)
        || !status_valid(closing, layout.value_elements)
    {
        return Err("GDN whole token failed status validation");
    }
    Ok(())
}

/// Optional offline/readback acceptance; resident execution uses device status.
pub fn validate_candidates(
    layout: PointwiseLayout,
    next_ring: &[f32],
    prepared: &[f32],
    controls: &[f32],
    gated: &[f32],
) -> Result<(), &'static str> {
    layout.check()?;
    if next_ring.len() != layout.ring_elements
        || prepared.len() != layout.channels
        || controls.len() != layout.control_elements
        || gated.len() != layout.value_elements
    {
        return Err("GDN candidate length mismatch");
    }
    if !finite(next_ring)
        || !finite(prepared)
        || !finite(controls)
        || !finite(gated)
        || controls[..layout.shape.value_heads]
            .iter()
            .any(|&g| g > 0.0)
        || controls[layout.shape.value_heads..]
            .iter()
            .any(|&b| !(0.0..=1.0).contains(&b))
    {
        return Err("GDN non-finite or invalid candidate");
    }
    Ok(())
}

#[allow(clippy::manual_range_contains)]
#[cube]
fn shader_finite(value: f32) -> bool {
    value <= 3.4028235e38_f32 && value >= -3.4028235e38_f32
}

#[cube]
fn shader_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

#[cube]
fn shader_softplus(value: f32) -> f32 {
    let tail = (-value.abs()).exp();
    let rounded = 1.0 + tail;
    let correction = if rounded == 1.0 {
        tail
    } else {
        rounded.ln() * (tail / (rounded - 1.0))
    };
    value.max(0.0) + correction
}

/// Six bindings; old_ring is immutable and next_ring MUST be a distinct buffer.
#[cube(launch, launch_unchecked)]
pub fn conv_silu(
    input: &Array<f32>,
    weights: &Array<f32>,
    old_ring: &Array<f32>,
    next_ring: &mut Array<f32>,
    activation: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] channels: usize,
) {
    let c = ABSOLUTE_POS;
    if c < channels {
        status[c] = 0;
        activation[c] = 0.0;
        #[unroll]
        for tap in 0..3_usize {
            next_ring[c * 3 + tap] = 0.0;
        }
        let current = input[c];
        let mut valid = shader_finite(current);
        let mut sum = 0.0_f32;
        #[unroll]
        for tap in 0..4_usize {
            let value = if tap == 3 {
                current
            } else {
                old_ring[c * 3 + tap]
            };
            let weight = weights[c * 4 + tap];
            valid = valid && shader_finite(value) && shader_finite(weight);
            sum += value * weight;
        }
        let result = sum * shader_sigmoid(sum);
        if valid && shader_finite(sum) && shader_finite(result) {
            next_ring[c * 3] = old_ring[c * 3 + 1];
            next_ring[c * 3 + 1] = old_ring[c * 3 + 2];
            next_ring[c * 3 + 2] = current;
            activation[c] = result;
            status[c] = 1;
        }
    }
}

/// Flat prepared [Q,K,V], preserving already-permuted GGUF head order.
/// One cube per 128-wide head; three bindings. All barriers are unconditional.
#[cube(launch, launch_unchecked)]
pub fn prepare_qkv(
    activation: &Array<f32>,
    prepared: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let head = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let total_heads = key_heads * 2 + value_heads;
    let mut squares = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    let mut x0 = 0.0_f32;
    let mut x1 = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut valid = true;
    if head < total_heads {
        x0 = activation[head * HEAD_DIM + lane];
        x1 = activation[head * HEAD_DIM + lane + LANES];
        valid = shader_finite(x0) && shader_finite(x1) && shader_finite(epsilon) && epsilon > 0.0;
        if head < key_heads * 2 {
            sum = x0 * x0 + x1 * x1;
        }
        valid = valid && shader_finite(sum);
    }
    squares[lane] = sum;
    // CubeCL 0.10 WGSL loses conditional-literal assignment expressions.
    // Explicit stores preserve the successful lane in generated device code.
    let mut lane_flag = 0_u32;
    if valid {
        lane_flag = 1_u32;
    }
    flags[lane] = lane_flag;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            squares[lane] = squares[lane] + squares[lane + stride];
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if head < total_heads {
        let base = head * HEAD_DIM + lane;
        prepared[base] = 0.0;
        prepared[base + LANES] = 0.0;
        status[base] = 0;
        status[base + LANES] = 0;
        let mut y0 = x0;
        let mut y1 = x1;
        let mut acceptable = flags[0] == 1;
        if head < key_heads * 2 {
            let denominator = squares[0] + epsilon;
            let inverse = 1.0 / denominator.sqrt();
            acceptable = acceptable
                && shader_finite(squares[0])
                && shader_finite(denominator)
                && denominator > 0.0
                && shader_finite(inverse);
            y0 = x0 * inverse;
            y1 = x1 * inverse;
            if head < key_heads {
                y0 /= (HEAD_DIM as f32).sqrt();
                y1 /= (HEAD_DIM as f32).sqrt();
            }
        }
        if acceptable && shader_finite(y0) && shader_finite(y1) {
            prepared[base] = y0;
            prepared[base + LANES] = y1;
            status[base] = 1;
            status[base + LANES] = 1;
        }
    }
}

/// Five bindings; params=[dt_bias,A], controls=[log_decay,sigmoid(beta)].
#[allow(clippy::manual_range_contains)] // Range methods do not lower in Cube DSL.
#[cube(launch, launch_unchecked)]
pub fn prepare_controls(
    alpha: &Array<f32>,
    beta_logits: &Array<f32>,
    params: &Array<f32>,
    controls: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] value_heads: usize,
) {
    let h = ABSOLUTE_POS;
    if h < value_heads {
        status[h] = 0;
        controls[h] = 0.0;
        controls[value_heads + h] = 0.0;
        let a = alpha[h];
        let b = beta_logits[h];
        let dt = params[h];
        let stored_a = params[value_heads + h];
        let biased = a + dt;
        let log_decay = shader_softplus(biased) * stored_a;
        let beta = shader_sigmoid(b);
        if shader_finite(a)
            && shader_finite(b)
            && shader_finite(dt)
            && shader_finite(stored_a)
            && stored_a < 0.0
            && shader_finite(biased)
            && shader_finite(log_decay)
            && log_decay <= 0.0
            && shader_finite(beta)
            && beta >= 0.0
            && beta <= 1.0
        {
            controls[h] = log_decay;
            controls[value_heads + h] = beta;
            status[h] = 1;
        }
    }
}

/// Five bindings; shared_gamma=true uses gamma128, false uses per-value gamma.
#[cube(launch, launch_unchecked)]
pub fn output_gate(
    output: &Array<f32>,
    z: &Array<f32>,
    gamma: &Array<f32>,
    gated: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    #[comptime] value_heads: usize,
    #[comptime] shared_gamma: bool,
) {
    let head = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut squares = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    let mut x0 = 0.0_f32;
    let mut x1 = 0.0_f32;
    let mut z0 = 0.0_f32;
    let mut z1 = 0.0_f32;
    let mut g0 = 0.0_f32;
    let mut g1 = 0.0_f32;
    let mut valid = true;
    if head < value_heads {
        let base = head * HEAD_DIM + lane;
        x0 = output[base];
        x1 = output[base + LANES];
        z0 = z[base];
        z1 = z[base + LANES];
        if comptime!(shared_gamma) {
            g0 = gamma[lane];
            g1 = gamma[lane + LANES];
        } else {
            g0 = gamma[base];
            g1 = gamma[base + LANES];
        }
        valid = shader_finite(x0)
            && shader_finite(x1)
            && shader_finite(z0)
            && shader_finite(z1)
            && shader_finite(g0)
            && shader_finite(g1)
            && shader_finite(epsilon)
            && epsilon > 0.0;
    }
    let sum = x0 * x0 + x1 * x1;
    squares[lane] = sum;
    let mut lane_flag = 0_u32;
    if valid && shader_finite(sum) {
        lane_flag = 1_u32;
    }
    flags[lane] = lane_flag;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            squares[lane] = squares[lane] + squares[lane + stride];
            flags[lane] = flags[lane] & flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if head < value_heads {
        let base = head * HEAD_DIM + lane;
        gated[base] = 0.0;
        gated[base + LANES] = 0.0;
        status[base] = 0;
        status[base + LANES] = 0;
        let denominator = squares[0] / HEAD_DIM as f32 + epsilon;
        let inverse = 1.0 / denominator.sqrt();
        let normalized0 = x0 * inverse * g0;
        let normalized1 = x1 * inverse * g1;
        let y0 = normalized0 * shader_sigmoid(z0);
        let y1 = normalized1 * shader_sigmoid(z1);
        if flags[0] == 1
            && shader_finite(squares[0])
            && shader_finite(denominator)
            && denominator > 0.0
            && shader_finite(inverse)
            && shader_finite(normalized0)
            && shader_finite(normalized1)
            && shader_finite(y0)
            && shader_finite(y1)
        {
            gated[base] = y0;
            gated[base + LANES] = y1;
            status[base] = 1;
            status[base + LANES] = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::ops;

    fn close(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && e.is_finite() && (a - e).abs() <= absolute + relative * e.abs(),
                "index={i} actual={a} expected={e}"
            );
        }
    }
    fn tree_squares(head: &[f32]) -> f32 {
        let mut sums = [0.0_f32; LANES];
        for lane in 0..LANES {
            sums[lane] = head[lane] * head[lane] + head[lane + LANES] * head[lane + LANES];
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                sums[lane] += sums[lane + stride];
            }
            stride /= 2;
        }
        sums[0]
    }
    fn emulate_conv(
        layout: PointwiseLayout,
        input: &[f32],
        weights: &[f32],
        old: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let mut next = vec![0.0; layout.ring_elements];
        let mut out = vec![0.0; layout.channels];
        let mut status = vec![0; layout.channels];
        for c in 0..layout.channels {
            let mut sum = 0.0_f32;
            let mut valid = input[c].is_finite();
            for tap in 0..4 {
                let value = if tap == 3 { input[c] } else { old[c * 3 + tap] };
                valid &= value.is_finite() && weights[c * 4 + tap].is_finite();
                sum += value * weights[c * 4 + tap];
            }
            let result = sum * shader_sigmoid(sum);
            if valid && sum.is_finite() && result.is_finite() {
                next[c * 3..c * 3 + 3].copy_from_slice(&[old[c * 3 + 1], old[c * 3 + 2], input[c]]);
                out[c] = result;
                status[c] = 1;
            }
        }
        (next, out, status)
    }
    fn emulate_qkv(layout: PointwiseLayout, input: &[f32]) -> Vec<f32> {
        let mut out = input.to_vec();
        for (h, head) in out
            .chunks_exact_mut(HEAD_DIM)
            .enumerate()
            .take(layout.shape.key_heads * 2)
        {
            let inv = 1.0 / (tree_squares(head) + EPSILON).sqrt();
            for value in head {
                *value *= inv;
                if h < layout.shape.key_heads {
                    *value /= (HEAD_DIM as f32).sqrt();
                }
            }
        }
        out
    }
    fn emulate_output(input: &[f32], z: &[f32], gamma: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; input.len()];
        for (h, head) in input.chunks_exact(HEAD_DIM).enumerate() {
            let inverse = 1.0 / (tree_squares(head) / HEAD_DIM as f32 + EPSILON).sqrt();
            for j in 0..HEAD_DIM {
                let index = h * HEAD_DIM + j;
                let g = gamma[if gamma.len() == HEAD_DIM { j } else { index }];
                let normalized = head[j] * inverse * g;
                out[index] = normalized * shader_sigmoid(z[index]);
            }
        }
        out
    }

    #[test]
    fn full_qwen_layout_and_whole_token_status_are_bounded() {
        let l = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        assert_eq!(
            (l.channels, l.qk_elements, l.value_elements),
            (10240, 2048, 6144)
        );
        assert_eq!(
            (l.ring_elements, l.conv_weight_elements, l.control_elements),
            (30720, 40960, 96)
        );
        assert_eq!((l.conv_cubes, l.qkv_heads, l.control_cubes), (160, 80, 1));
        assert!(validate_conv_lengths(l, 10240, 40960, 30720, 30720, 10240, 10240).is_ok());
        assert!(validate_conv_lengths(l, 10240, 40960, 30720, 30719, 10240, 10240).is_err());
        assert!(validate_qkv_lengths(l, 10240, 10240, 10240, EPSILON).is_ok());
        assert!(validate_qkv_lengths(l, 10240, 10240, 10239, EPSILON).is_err());
        assert!(validate_control_lengths(l, 48, 48, 96, 96, 48).is_ok());
        assert!(validate_control_lengths(l, 48, 48, 95, 96, 48).is_err());
        assert_eq!(
            validate_output_lengths(l, 6144, 6144, 128, 6144, 6144, EPSILON),
            Ok(true)
        );
        assert_eq!(
            validate_output_lengths(l, 6144, 6144, 6144, 6144, 6144, EPSILON),
            Ok(false)
        );
        assert!(validate_output_lengths(l, 6144, 6144, 127, 6144, 6144, EPSILON).is_err());
        let conv = vec![1; l.channels];
        let qkv = conv.clone();
        let controls = vec![1; 48];
        let recurrence = vec![1; l.value_elements];
        let closing = recurrence.clone();
        assert!(validate_token_status(l, &conv, &qkv, &controls, &recurrence, &closing).is_ok());
        for stage in 0..5 {
            let mut statuses = [
                conv.clone(),
                qkv.clone(),
                controls.clone(),
                recurrence.clone(),
                closing.clone(),
            ];
            let last = statuses[stage].len() - 1;
            statuses[stage][last] = 0;
            assert!(
                validate_token_status(
                    l,
                    &statuses[0],
                    &statuses[1],
                    &statuses[2],
                    &statuses[3],
                    &statuses[4]
                )
                .is_err()
            );
        }
        let mut forged = l;
        forged.channels -= 1;
        assert!(
            validate_token_status(forged, &conv, &qkv, &controls, &recurrence, &closing).is_err()
        );
        for shape in [
            GdnShape {
                head_dim: 64,
                key_heads: 1,
                value_heads: 1,
            },
            GdnShape {
                head_dim: 128,
                key_heads: 0,
                value_heads: 1,
            },
            GdnShape {
                head_dim: 128,
                key_heads: 2,
                value_heads: 3,
            },
        ] {
            assert!(PointwiseLayout::new(shape).is_err());
        }
    }

    #[test]
    fn seven_raw_conv_ring_steps_match_independent_causal_conv_then_silu() {
        let l = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        let mut oracle = ops::CausalConvState::new(l.channels, 4, 1).unwrap();
        let weights: Vec<f32> = (0..l.conv_weight_elements)
            .map(|i| ((i % 13) as f32 - 6.0) / 32.0)
            .collect();
        let mut old = vec![0.0; l.ring_elements];
        for token in 0..7 {
            let input: Vec<f32> = (0..l.channels)
                .map(|i| ((i * 7 + token * 11) % 29) as f32 / 16.0 - 0.875)
                .collect();
            validate_conv_inputs(l, &input, &weights, &old).unwrap();
            let preserved = old.clone();
            let (next, actual, status) = emulate_conv(l, &input, &weights, &old);
            let expected: Vec<f32> = oracle
                .step(&input, &weights)
                .unwrap()
                .into_iter()
                .map(ops::silu)
                .collect();
            close(&actual, &expected, 0.0, 0.0); // These dyadic four-tap sums are exact.
            assert_eq!(next, oracle.history());
            assert_eq!(old, preserved);
            assert!(status.iter().all(|&v| v == 1));
            old = next;
        }
    }

    #[test]
    fn qk_l2_scaling_and_value_order_match_independent_reference() {
        let l = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        let mut input: Vec<f32> = (0..l.channels)
            .map(|i| ((i % 37) as f32 - 18.0) / 32.0)
            .collect();
        input[..HEAD_DIM].fill(0.0);
        input[HEAD_DIM..HEAD_DIM * 2].fill(1.0);
        input[HEAD_DIM * 2..HEAD_DIM * 3].fill(0.0);
        input[HEAD_DIM * 2 + 127] = 1e-3;
        let actual = emulate_qkv(l, &input);
        let mut expected = ops::l2_norm_heads(&input[..l.qk_elements], HEAD_DIM, EPSILON).unwrap();
        expected
            .iter_mut()
            .for_each(|v| *v /= (HEAD_DIM as f32).sqrt());
        expected.extend(
            ops::l2_norm_heads(&input[l.qk_elements..2 * l.qk_elements], HEAD_DIM, EPSILON)
                .unwrap(),
        );
        expected.extend_from_slice(&input[2 * l.qk_elements..]);
        close(&actual, &expected, 2e-7, 2e-6);
        assert_eq!(&actual[2 * l.qk_elements..], &input[2 * l.qk_elements..]);
        assert!(actual[HEAD_DIM * 2 + 127] > 0.06); // EPS is inside sqrt; query scaling is separate.
    }

    #[test]
    fn compensated_softplus_controls_keep_small_tails_and_negative_stored_a() {
        let l = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        let alpha: Vec<f32> = (0..48)
            .map(|i| [-100.0, -20.0, -8.0, -1.0, 0.0, 1.0, 20.0, 100.0][i % 8])
            .collect();
        let dt: Vec<f32> = (0..48).map(|i| (i % 3) as f32 * 0.25).collect();
        let a: Vec<f32> = (0..48)
            .map(|i| {
                if i % 8 == 1 {
                    -1e8
                } else {
                    -(i + 1) as f32 / 8.0
                }
            })
            .collect();
        let beta: Vec<f32> = (0..48)
            .map(|i| [-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0][i % 7])
            .collect();
        let params: Vec<f32> = dt.iter().chain(&a).copied().collect();
        validate_control_inputs(l, &alpha, &beta, &params).unwrap();
        let actual: Vec<f32> = alpha
            .iter()
            .zip(&dt)
            .zip(&a)
            .map(|((&x, &d), &weight)| shader_softplus(x + d) * weight)
            .collect();
        let expected = ops::gdn_log_decay(&alpha, &dt, &a).unwrap();
        close(&actual, &expected, 2e-7, 3e-6);
        assert!(shader_softplus(-20.0) > 2e-9 && actual[1] < -0.2);
        for x in [
            -100.0_f32, -20.0, -17.0, -8.0, -1.0, -0.0, 0.0, 1.0, 8.0, 17.0, 20.0, 100.0,
        ] {
            close(&[shader_softplus(x)], &[ops::softplus(x)], 1e-44, 3e-6);
            assert_eq!(shader_sigmoid(x).to_bits(), ops::sigmoid(x).to_bits());
        }
        let mut invalid = params.clone();
        invalid[48] = 0.0;
        assert!(validate_control_inputs(l, &alpha, &beta, &invalid).is_err());
    }

    #[test]
    fn closing_rms_folded_gamma_and_sigmoid_match_independent_reference() {
        let l = PointwiseLayout::new(GdnShape::qwen38()).unwrap();
        let mut out: Vec<f32> = (0..l.value_elements)
            .map(|i| ((i % 31) as f32 - 15.0) / 16.0 * (1.0 + (i / 128 % 7) as f32))
            .collect();
        out[..128].fill(0.0);
        out[128..256].fill(1e-3);
        let z: Vec<f32> = (0..l.value_elements)
            .map(|i| [-80.0, -2.0, -0.0, 0.0, 0.5, 2.0, 80.0][i % 7])
            .collect();
        for gamma in [
            (0..128)
                .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
                .collect::<Vec<_>>(),
            (0..l.value_elements)
                .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                .collect(),
        ] {
            validate_output_inputs(l, &out, &z, &gamma, EPSILON).unwrap();
            let actual = emulate_output(&out, &z, &gamma);
            let expected = ops::gdn_output_gate(&out, &z, &gamma, HEAD_DIM, EPSILON).unwrap();
            close(&actual, &expected, 3e-7, 3e-6);
        }
    }

    #[test]
    fn rejects_wrong_lengths_nonfinite_and_overflow_without_mutating_old_ring() {
        let l = PointwiseLayout::new(GdnShape::new(128, 1, 1).unwrap()).unwrap();
        let mut input = vec![1.0; l.channels];
        let weights = vec![1.0; l.conv_weight_elements];
        let old = vec![0.0; l.ring_elements];
        let preserved = old.clone();
        assert!(validate_conv_inputs(l, &input[..input.len() - 1], &weights, &old).is_err());
        input[3] = f32::NAN;
        assert!(validate_conv_inputs(l, &input, &weights, &old).is_err());
        let (next, _, status) = emulate_conv(l, &input, &weights, &old);
        assert_eq!(status[3], 0);
        assert_eq!(&next[9..12], &[0.0; 3]);
        assert_eq!(old, preserved);
        input.fill(f32::MAX);
        let enormous = vec![2.0; l.conv_weight_elements];
        let (_, out, status) = emulate_conv(l, &input, &enormous, &old);
        assert!(status.iter().all(|&s| s == 0));
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(old, preserved);
        assert!(validate_qkv_inputs(l, &vec![0.0; l.channels], 0.0).is_err());
        assert!(
            validate_output_inputs(
                l,
                &vec![0.0; l.value_elements],
                &vec![0.0; l.value_elements],
                &[1.0; 127],
                EPSILON
            )
            .is_err()
        );
    }

    #[test]
    fn all_kernels_generate_checked_unchecked_wgsl_without_a_device() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for variant in 0..5 {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let mut b = KernelBuilder::default();
                b.scope.register_type::<u32>(uint);
                b.scope.register_type::<usize>(uint);
                b.scope.register_type::<f32>(float);
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                if variant == 0 {
                    let x = b.input_array(Type::new(float)).into();
                    let w = b.input_array(Type::new(float)).into();
                    let old = b.input_array(Type::new(float)).into();
                    let next = b.output_array(Type::new(float)).into();
                    let out = b.output_array(Type::new(float)).into();
                    let status = b.output_array(Type::new(uint)).into();
                    conv_silu::expand(&mut b.scope, x, w, old, next, out, status, 10240);
                } else if variant == 1 {
                    let x = b.input_array(Type::new(float)).into();
                    let out = b.output_array(Type::new(float)).into();
                    let status = b.output_array(Type::new(uint)).into();
                    let eps = b.scalar(float).into();
                    prepare_qkv::expand(&mut b.scope, x, out, status, eps, 16, 48);
                } else if variant == 2 {
                    let alpha = b.input_array(Type::new(float)).into();
                    let beta = b.input_array(Type::new(float)).into();
                    let params = b.input_array(Type::new(float)).into();
                    let out = b.output_array(Type::new(float)).into();
                    let status = b.output_array(Type::new(uint)).into();
                    prepare_controls::expand(&mut b.scope, alpha, beta, params, out, status, 48);
                } else {
                    let x = b.input_array(Type::new(float)).into();
                    let z = b.input_array(Type::new(float)).into();
                    let gamma = b.input_array(Type::new(float)).into();
                    let out = b.output_array(Type::new(float)).into();
                    let status = b.output_array(Type::new(uint)).into();
                    let eps = b.scalar(float).into();
                    output_gate::expand(
                        &mut b.scope,
                        x,
                        z,
                        gamma,
                        out,
                        status,
                        eps,
                        48,
                        variant == 3,
                    );
                }
                let settings = KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("gdn_pointwise");
                let source = WgslCompiler::default()
                    .compile(
                        b.build(settings),
                        &WgpuCompilationOptions::default(),
                        mode,
                        uint,
                    )
                    .unwrap()
                    .to_string();
                assert!(source.contains("@compute"));
                if variant == 1 || variant >= 3 {
                    // This is a compiler contract regression: an earlier
                    // conditional-literal assignment produced empty branches
                    // and unconditional zero flags despite correct host math.
                    let prefix = source.split("workgroupBarrier();").next().unwrap();
                    assert!(
                        prefix.contains("= u32(1);"),
                        "WGSL lost the successful lane assignment"
                    );
                    assert!(prefix.contains("shared_memory_1["));
                }
            }
        }
    }
}
