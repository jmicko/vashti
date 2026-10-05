//! Experimental Rust/CubeCL f32 GDN recurrence, with CPU-owned state.
//!
//! The logical GGUF state is `[value_head][key_dim][value_dim]`. One lane owns
//! `(value_head, value_dim)` and adjacent lanes access adjacent final dimensions.
//! Value head h reads Q/K head `h % key_heads`: the GGUF converter already
//! changed HF's grouped heads to this tiled order. Do not permute weights again.
//! Q/K must already be L2 normalized, with Q additionally scaled by 1/sqrt(S).
//! This module performs no projection, convolution, output RMS or sigmoid gate.
//!
//! Formula provenance: independent Rust translation of Transformers revision
//! `0a896aa41bba78f92338db21cc468fe043888657`, modeling_qwen4_exp.py:404-452
//! (Apache-2.0), plus the GGUF head permutation in llama.cpp revision
//! `806eee9841de5f2c20f9d43914117f157d2baacc`, conversion/qwen.py:460-630
//! (MIT). `kuno_text::ops::GdnState` is the independent higher-precision oracle.
//!
//! Call validate_inputs BEFORE allocating/uploading buffers. Launch scalar
//! ArrayArgs in the signature order, using CubeDim=(64,1,1) and
//! CubeCount=(layout.workgroups_x,1,1), then S/key_heads/value_heads. Eight
//! storage bindings are used; controls contains `[all log_decay, all beta]`.
//! Old/new state handles MUST be distinct. Never commit new state until
//! validate_download succeeds. The kernel zeroes a failed column and reports
//! status=0, but the host rejects the entire step, preserving its old state.
//! No backend state or request-content diagnostics live in this operator.
//! Host/device buffer erasure and cancellation remain the calling worker's
//! responsibility. Device execution/parity has not been tested by host tests.

use cubecl::prelude::*;
use kuno_text::ops::GdnShape;

pub const LANES: usize = 64;
const STATE_PAIR_BUDGET: usize = 128 * 1024 * 1024;
const MAX_WORKGROUPS_X: usize = 65_535;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnLayout {
    pub shape: GdnShape,
    pub qk_elements: usize,
    pub value_elements: usize,
    pub state_elements: usize,
    pub control_elements: usize,
    pub workgroups_x: u32,
}

impl GdnLayout {
    pub fn new(shape: GdnShape) -> Result<Self, &'static str> {
        GdnShape::new(shape.head_dim, shape.key_heads, shape.value_heads)
            .map_err(|_| "invalid GDN geometry")?;
        let qk_elements = shape
            .key_heads
            .checked_mul(shape.head_dim)
            .ok_or("GDN Q/K length overflow")?;
        let value_elements = shape
            .value_heads
            .checked_mul(shape.head_dim)
            .ok_or("GDN value length overflow")?;
        let state_elements = value_elements
            .checked_mul(shape.head_dim)
            .ok_or("GDN state length overflow")?;
        let control_elements = shape
            .value_heads
            .checked_mul(2)
            .ok_or("GDN control length overflow")?;
        if state_elements
            .checked_mul(8)
            .ok_or("GDN state byte length overflow")?
            > STATE_PAIR_BUDGET
        {
            return Err("GDN state pair exceeds prototype budget");
        }
        for length in [
            qk_elements,
            value_elements,
            state_elements,
            control_elements,
        ] {
            if length > u32::MAX as usize {
                return Err("GDN buffer exceeds 32-bit shader indexing");
            }
        }
        let groups = value_elements.div_ceil(LANES);
        if groups > MAX_WORKGROUPS_X {
            return Err("GDN launch grid exceeds portable prototype limit");
        }
        let workgroups_x = u32::try_from(groups).map_err(|_| "GDN launch grid overflow")?;
        Ok(Self {
            shape,
            qk_elements,
            value_elements,
            state_elements,
            control_elements,
            workgroups_x,
        })
    }
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

pub fn validate_inputs(
    shape: GdnShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    old_state: &[f32],
    log_decay: &[f32],
    beta: &[f32],
) -> Result<GdnLayout, &'static str> {
    let layout = GdnLayout::new(shape)?;
    if query.len() != layout.qk_elements
        || key.len() != layout.qk_elements
        || value.len() != layout.value_elements
        || old_state.len() != layout.state_elements
        || log_decay.len() != shape.value_heads
        || beta.len() != shape.value_heads
    {
        return Err("GDN input length mismatch");
    }
    if ![query, key, value, old_state, log_decay, beta]
        .iter()
        .all(|values| finite(values))
    {
        return Err("GDN input is non-finite");
    }
    if log_decay.iter().any(|&g| g > 0.0) || beta.iter().any(|&b| !(0.0..=1.0).contains(&b)) {
        return Err("GDN decay or beta outside supported range");
    }
    Ok(layout)
}

pub fn pack_controls(
    shape: GdnShape,
    log_decay: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>, &'static str> {
    let layout = GdnLayout::new(shape)?;
    if log_decay.len() != shape.value_heads
        || beta.len() != shape.value_heads
        || !finite(log_decay)
        || !finite(beta)
        || log_decay.iter().any(|&g| g > 0.0)
        || beta.iter().any(|&b| !(0.0..=1.0).contains(&b))
    {
        return Err("GDN invalid controls");
    }
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(layout.control_elements)
        .map_err(|_| "GDN control allocation failed")?;
    packed.extend_from_slice(log_decay);
    packed.extend_from_slice(beta);
    Ok(packed)
}

/// Every lane must report exactly 1. No partial next state may be committed.
pub fn validate_download(
    layout: GdnLayout,
    next_state: &[f32],
    output: &[f32],
    status: &[u32],
) -> Result<(), &'static str> {
    // Recompute public layout fields rather than trust manually constructed data.
    if layout != GdnLayout::new(layout.shape)?
        || next_state.len() != layout.state_elements
        || output.len() != layout.value_elements
        || status.len() != layout.value_elements
    {
        return Err("GDN download length or geometry mismatch");
    }
    if status.iter().any(|&value| value != 1) || !finite(next_state) || !finite(output) {
        return Err("GDN device step failed finite validation");
    }
    Ok(())
}

// Rust range methods do not lower through this CubeCL shader DSL.
#[allow(clippy::manual_range_contains)]
#[cube]
fn shader_finite(value: f32) -> bool {
    // WGSL has no portable isnan/isinf intrinsic. Both comparisons reject NaN,
    // and bounds reject infinities, without requiring bitcasting or shader f64.
    value <= 3.4028235e38_f32 && value >= -3.4028235e38_f32
}

/// All input/output arrays are scalar and have the validated logical lengths.
/// Old/new state allocations must not alias. No cube barrier is required:
/// each lane reads immutable old state and writes one unique next-state column.
#[cube(launch)]
pub fn gdn_step(
    query: &Array<f32>,
    key: &Array<f32>,
    value: &Array<f32>,
    controls: &Array<f32>,
    old_state: &Array<f32>,
    next_state: &mut Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let lane = ABSOLUTE_POS;
    if lane < value_heads * head_dim {
        let h = lane / head_dim;
        let j = lane % head_dim;
        let kh = h % key_heads;
        let qk_start = kh * head_dim;
        let state_start = h * head_dim * head_dim + j;
        status[lane] = 0;
        output[lane] = 0.0_f32;
        let decay = controls[h].exp();
        let beta = controls[value_heads + h];
        let mut predicted = 0.0_f32;
        // Initialize this column even if the arithmetic will subsequently fail.
        for i in 0..head_dim {
            let index = state_start + i * head_dim;
            next_state[index] = 0.0_f32;
            let decayed = old_state[index] * decay;
            predicted += decayed * key[qk_start + i];
        }
        let delta = (value[lane] - predicted) * beta;
        if shader_finite(predicted) && shader_finite(delta) {
            let mut readout = 0.0_f32;
            let mut valid = true;
            for i in 0..head_dim {
                let index = state_start + i * head_dim;
                let decayed = old_state[index] * decay;
                let updated = decayed + key[qk_start + i] * delta;
                valid = valid && shader_finite(updated);
                next_state[index] = updated;
                readout += updated * query[qk_start + i];
            }
            if valid && shader_finite(readout) {
                output[lane] = readout;
                status[lane] = 1;
            } else {
                for i in 0..head_dim {
                    next_state[state_start + i * head_dim] = 0.0_f32;
                }
            }
        }
    }
}

/// Pure-host emulation of the shader's f32 reduction order. This is useful for
/// separating device codegen bugs from expected scalar-oracle rounding. Device
/// FMA and exp rounding may still differ. Compare both state and output against
/// the independent higher-precision GdnState; do not accept self-parity alone.
#[cfg(test)]
pub struct GdnOutput {
    pub state: Vec<f32>,
    pub output: Vec<f32>,
}

#[cfg(test)]
pub fn lane_reference(
    shape: GdnShape,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    old_state: &[f32],
    log_decay: &[f32],
    beta: &[f32],
) -> Result<GdnOutput, &'static str> {
    let layout = validate_inputs(shape, query, key, value, old_state, log_decay, beta)?;
    let mut state = vec![0.0_f32; layout.state_elements];
    let mut output = vec![0.0_f32; layout.value_elements];
    for lane in 0..layout.value_elements {
        let h = lane / shape.head_dim;
        let j = lane % shape.head_dim;
        let kh = h % shape.key_heads;
        let qk_start = kh * shape.head_dim;
        let state_start = h * shape.head_dim * shape.head_dim + j;
        let decay = log_decay[h].exp();
        let mut predicted = 0.0_f32;
        for i in 0..shape.head_dim {
            let index = state_start + i * shape.head_dim;
            let decayed = old_state[index] * decay;
            predicted += decayed * key[qk_start + i];
        }
        let delta = (value[lane] - predicted) * beta[h];
        if !predicted.is_finite() || !delta.is_finite() {
            return Err("GDN f32 recurrence overflow");
        }
        let mut readout = 0.0_f32;
        for i in 0..shape.head_dim {
            let index = state_start + i * shape.head_dim;
            let decayed = old_state[index] * decay;
            let updated = decayed + key[qk_start + i] * delta;
            if !updated.is_finite() {
                return Err("GDN f32 updated state overflow");
            }
            state[index] = updated;
            readout += updated * query[qk_start + i];
        }
        if !readout.is_finite() {
            return Err("GDN f32 readout overflow");
        }
        output[lane] = readout;
    }
    Ok(GdnOutput { state, output })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::ops::{GdnState, l2_norm_heads};

    fn near(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert!(
                a.is_finite() && (a - e).abs() <= 3e-5 + 3e-4 * e.abs(),
                "synthetic GDN mismatch"
            );
        }
    }

    #[test]
    fn independently_calculated_decay_delta_and_updated_readout() {
        let shape = GdnShape::new(2, 1, 2).unwrap();
        let initial = [1., 2., 3., 4., -1., 0.5, 2., -2.];
        let result = lane_reference(
            shape,
            &[0.25, 0.5],
            &[0.5, -0.25],
            &[1., -2., 3., 4.],
            &initial,
            &[-std::f32::consts::LN_2; 2],
            &[0.75, 0.25],
        )
        .unwrap();
        // These dyadic numbers were derived by hand, independently of both
        // implementations. Readout must read the updated, already-decayed state.
        assert_eq!(result.output, [0.875, 1.25, 0.375, -0.4375]);
        assert_eq!(
            result.state,
            [
                0.921875, 0.25, 1.2890625, 2.375, -0.0625, 0.703125, 0.78125, -1.2265625
            ]
        );
    }

    #[test]
    fn gguf_tiled_head_association_is_modulo() {
        let shape = GdnShape::new(1, 2, 4).unwrap();
        let result = lane_reference(
            shape,
            &[1., 0.5],
            &[1., 0.5],
            &[1., 2., 3., 4.],
            &[0.; 4],
            &[0.; 4],
            &[1.; 4],
        )
        .unwrap();
        assert_eq!(result.state, [1., 1., 3., 2.]);
        assert_eq!(result.output, [1., 0.5, 3., 1.]);
    }

    #[test]
    fn multistep_f32_lanes_match_independent_f64_oracle() {
        for shape in [
            GdnShape::new(2, 2, 4).unwrap(),
            GdnShape::new(5, 2, 6).unwrap(),
            GdnShape::qwen38(),
        ] {
            let layout = GdnLayout::new(shape).unwrap();
            let mut oracle = GdnState::new(shape).unwrap();
            let mut state: Vec<f32> = (0..layout.state_elements)
                .map(|i| ((i % 23) as f32 - 11.) / 256.)
                .collect();
            oracle.set_values(&state).unwrap();
            for step in 0..7 {
                let raw_q: Vec<f32> = (0..layout.qk_elements)
                    .map(|i| (((i + step * 3) % 19) as f32 - 9.) / 16.)
                    .collect();
                let raw_k: Vec<f32> = (0..layout.qk_elements)
                    .map(|i| (((i * 3 + step * 5) % 17) as f32 - 8.) / 16.)
                    .collect();
                let mut q = l2_norm_heads(&raw_q, shape.head_dim, 1e-6).unwrap();
                q.iter_mut()
                    .for_each(|value| *value /= (shape.head_dim as f32).sqrt());
                let k = l2_norm_heads(&raw_k, shape.head_dim, 1e-6).unwrap();
                let value: Vec<f32> = (0..layout.value_elements)
                    .map(|i| (((i + step * 7) % 31) as f32 - 15.) / 32.)
                    .collect();
                let log_decay: Vec<f32> = (0..shape.value_heads)
                    .map(|h| -(((h + step) % 7 + 1) as f32) / 16.)
                    .collect();
                let beta: Vec<f32> = (0..shape.value_heads)
                    .map(|h| ((h + step) % 9) as f32 / 8.)
                    .collect();
                let expected = oracle.step(&q, &k, &value, &log_decay, &beta).unwrap();
                let actual =
                    lane_reference(shape, &q, &k, &value, &state, &log_decay, &beta).unwrap();
                near(&actual.output, &expected);
                near(&actual.state, oracle.values());
                state = actual.state;
            }
        }
    }

    #[test]
    fn shape_controls_and_input_validation_reject_before_launch() {
        let shape = GdnShape::new(2, 1, 2).unwrap();
        let layout = validate_inputs(
            shape, &[1.; 2], &[1.; 2], &[1.; 4], &[0.; 8], &[-1.; 2], &[0.5; 2],
        )
        .unwrap();
        assert_eq!(layout.workgroups_x, 1);
        assert_eq!(
            pack_controls(shape, &[-1., -2.], &[0., 1.]).unwrap(),
            [-1., -2., 0., 1.]
        );
        assert!(
            validate_inputs(
                shape, &[1.; 1], &[1.; 2], &[1.; 4], &[0.; 8], &[-1.; 2], &[0.5; 2]
            )
            .is_err()
        );
        assert!(
            validate_inputs(
                shape,
                &[f32::NAN; 2],
                &[1.; 2],
                &[1.; 4],
                &[0.; 8],
                &[-1.; 2],
                &[0.5; 2]
            )
            .is_err()
        );
        assert!(
            validate_inputs(
                shape, &[1.; 2], &[1.; 2], &[1.; 4], &[0.; 8], &[0.1; 2], &[0.5; 2]
            )
            .is_err()
        );
        assert!(pack_controls(shape, &[-1.; 2], &[1.1; 2]).is_err());
        assert!(
            GdnLayout::new(GdnShape {
                head_dim: 2,
                key_heads: 2,
                value_heads: 3
            })
            .is_err()
        );
        assert!(
            GdnLayout::new(GdnShape {
                head_dim: usize::MAX,
                key_heads: 1,
                value_heads: 1
            })
            .is_err()
        );
    }

    #[test]
    fn f32_overflow_rejects_whole_step_without_mutating_old_state() {
        let shape = GdnShape::new(2, 1, 1).unwrap();
        let old_state = [f32::MAX; 4];
        assert!(
            lane_reference(
                shape,
                &[1.; 2],
                &[1.; 2],
                &[0.; 2],
                &old_state,
                &[0.],
                &[1.]
            )
            .is_err()
        );
        assert_eq!(old_state, [f32::MAX; 4]);
    }

    #[test]
    fn download_acceptance_requires_all_lanes_finite_and_successful() {
        let mut layout = GdnLayout::new(GdnShape::new(2, 1, 1).unwrap()).unwrap();
        assert!(validate_download(layout, &[0.; 4], &[0.; 2], &[1; 2]).is_ok());
        assert!(validate_download(layout, &[0.; 4], &[0.; 2], &[1, 0]).is_err());
        assert!(validate_download(layout, &[0.; 4], &[f32::INFINITY, 0.], &[1; 2]).is_err());
        assert!(validate_download(layout, &[f32::NAN; 4], &[0.; 2], &[1; 2]).is_err());
        assert!(validate_download(layout, &[0.; 3], &[0.; 2], &[1; 2]).is_err());
        layout.state_elements = 3;
        assert!(validate_download(layout, &[0.; 3], &[0.; 2], &[1; 2]).is_err());
    }

    #[test]
    fn lowers_to_wgsl_offline_without_device_initialization() {
        use cubecl::wgpu::{WgpuRuntime, WgslCompiler};
        use cubecl::{Compiler, WgpuCompilationOptions};

        for shape in [GdnShape::new(2, 1, 2).unwrap(), GdnShape::qwen38()] {
            let mut builder = KernelBuilder::default();
            builder.runtime_properties(WgpuRuntime::target_properties());
            AddressType::U32.register(&mut builder.scope);
            let floats = f32::as_type(&builder.scope);
            let integers = u32::as_type(&builder.scope);
            let query = builder.input_array(floats).into();
            let key = builder.input_array(floats).into();
            let value = builder.input_array(floats).into();
            let controls = builder.input_array(floats).into();
            let old_state = builder.input_array(floats).into();
            let next_state = builder.output_array(floats).into();
            let output = builder.output_array(floats).into();
            let status = builder.output_array(integers).into();
            gdn_step::expand(
                &mut builder.scope,
                query,
                key,
                value,
                controls,
                old_state,
                next_state,
                output,
                status,
                shape.head_dim,
                shape.key_heads,
                shape.value_heads,
            );
            let definition = builder.build(
                KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(LANES as u32))
                    .kernel_name("gdn_step"),
            );
            assert_eq!(definition.buffers.len(), 8);
            let shader = WgslCompiler::default()
                .compile(
                    definition,
                    &WgpuCompilationOptions::default(),
                    ExecutionMode::Checked,
                    AddressType::U32.unsigned_type(),
                )
                .unwrap();
            let source = shader.to_string();
            assert!(source.contains("fn gdn_step"));
            assert!(!source.contains("enable f16"));
            assert!(!source.contains("f64"));
        }
    }
}
