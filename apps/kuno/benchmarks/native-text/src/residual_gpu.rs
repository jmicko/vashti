//! Laboratory resident-vector kernels for the exact borrowed GGUF residual graph.
//!
//! Independent Rust translation of Transformers 0a896aa41bba78f92338db21cc468fe043888657
//! `modeling_qwen4_exp.py` and the folded-GGUF graph recorded in
//! `kuno_text::ops::GatedResidualShape`. No upstream engine is called.
//!
//! Launch scalar ArrayArgs, CubeDim=(64,1,1). RMS uses one cube per stream;
//! other kernels use ceil(output_length/64) cubes. All inputs and outputs must
//! have the validated logical lengths and MUST be distinct allocations (also
//! preserves ORIGINAL residual across normalization). Stored gamma is direct.
//! RMS status has HC words; other status lengths match their output. Every
//! status word must be 1 before accepting a graph token; invalid numerical
//! inputs/results produce output=0,status=0. Caller owns buffers/erasure and
//! device acceptance. Host tests do not execute a GPU.

use cubecl::prelude::*;
use kuno_text::ops::GatedResidualShape;

pub const LANES: usize = 64;
const MAX_GROUPS: usize = 65_535;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualLayout {
    pub shape: GatedResidualShape,
    pub low_rank: usize,
    pub residual_elements: usize,
    pub norm_groups: u32,
    pub down_groups: u32,
    pub mix_groups: u32,
    pub combine_groups: u32,
}

impl ResidualLayout {
    pub fn new(shape: GatedResidualShape, low_rank: usize) -> Result<Self, &'static str> {
        GatedResidualShape::new(shape.streams, shape.width)
            .map_err(|_| "invalid residual shape")?;
        if low_rank == 0 {
            return Err("residual low rank is zero");
        }
        let residual_elements = shape
            .streams
            .checked_mul(shape.width)
            .ok_or("residual length overflow")?;
        let groups = |n: usize| -> Result<u32, &'static str> {
            let g = n.div_ceil(LANES);
            if n > u32::MAX as usize || g > MAX_GROUPS {
                return Err("residual portable grid limit");
            }
            Ok(g as u32)
        };
        if shape.streams > MAX_GROUPS {
            return Err("residual norm grid limit");
        }
        Ok(Self {
            shape,
            low_rank,
            residual_elements,
            norm_groups: shape.streams as u32,
            down_groups: groups(low_rank)?,
            mix_groups: groups(shape.width)?,
            combine_groups: groups(residual_elements)?,
        })
    }
    pub fn validate_norm(
        self,
        input_len: usize,
        gamma_len: usize,
        epsilon: f32,
    ) -> Result<(), &'static str> {
        self.check()?;
        if input_len != self.residual_elements || gamma_len != self.residual_elements {
            return Err("residual norm length mismatch");
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("residual norm epsilon invalid");
        }
        Ok(())
    }
    pub fn check(self) -> Result<(), &'static str> {
        if self != Self::new(self.shape, self.low_rank)? {
            return Err("residual layout inconsistent");
        }
        Ok(())
    }
}

pub fn validate_download(
    output: &[f32],
    status: &[u32],
    expected: usize,
    statuses: usize,
) -> Result<(), &'static str> {
    if output.len() != expected || status.len() != statuses {
        return Err("residual download length mismatch");
    }
    if !output.iter().all(|v| v.is_finite()) || status.iter().any(|&v| v != 1) {
        return Err("residual device numerical failure");
    }
    Ok(())
}

#[allow(clippy::manual_range_contains)] // Cube DSL does not lower Range::contains.
#[cube]
fn finite(x: f32) -> bool {
    x <= 3.4028235e38_f32 && x >= -3.4028235e38_f32
}

#[cube]
#[allow(unused_assignments)]
fn logistic(x: f32) -> f32 {
    let mut result = 0.0_f32;
    if x >= 0.0 {
        result = 1.0 / (1.0 + (-x).exp());
    } else {
        let e = x.exp();
        result = e / (1.0 + e);
    }
    result
}

/// Per-stream RMS: x/sqrt(mean(x*x)+eps)*gamma. Max scaling is algebraically
/// equivalent, avoids intermediate x*x overflow, and requires no shader f64.
/// All barriers are unconditional and loops have cube-uniform bounds.
#[allow(clippy::collapsible_if)] // Keep lane bounds outside shared-memory loads in Cube DSL.
#[cube(launch)]
pub fn grouped_rms_norm(
    input: &Array<f32>,
    gamma: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let group = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut maximum = 1.0_f32;
    let mut good = 1.0_f32;
    let tiles = width.div_ceil(LANES);
    if group < streams {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if d < width {
                let i = group * width + d;
                let x = input[i];
                let w = gamma[i];
                if finite(x) && finite(w) {
                    if x.abs() > maximum {
                        maximum = x.abs();
                    }
                } else {
                    good = 0.0;
                }
            }
        }
    }
    let mut scales = SharedMemory::<f32>::new(LANES);
    let mut goods = SharedMemory::<f32>::new(LANES);
    scales[lane] = maximum;
    goods[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            if scales[lane + stride] > scales[lane] {
                scales[lane] = scales[lane + stride];
            }
            if goods[lane + stride] < goods[lane] {
                goods[lane] = goods[lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    let scale = scales[0];
    let mut sum = 0.0_f32;
    if group < streams && goods[0] == 1.0 {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if d < width {
                let x = input[group * width + d] / scale;
                sum += x * x;
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    sums[lane] = sum;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let denominator = (sums[0] / width as f32 + (epsilon / scale) / scale).sqrt();
    let mut output_good = 1.0_f32;
    if group < streams {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if d < width {
                let i = group * width + d;
                let y = ((input[i] / scale) / denominator) * gamma[i];
                if goods[0] == 1.0 && finite(epsilon) && epsilon > 0.0 && finite(y) {
                    output[i] = y;
                } else {
                    output[i] = 0.0;
                    output_good = 0.0;
                }
            }
        }
    }
    // Preserve the immutable input-validity reduction until every lane has
    // finished reading it. Reusing goods here creates a shared-memory race
    // when a faster lane overwrites goods[0] while another emits its tile.
    let mut output_goods = SharedMemory::<f32>::new(LANES);
    output_goods[lane] = output_good;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            if output_goods[lane + stride] < output_goods[lane] {
                output_goods[lane] = output_goods[lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && group < streams {
        status[group] = output_goods[0] as u32;
    }
}

/// Down is LR elements, not LR*HC. Scaling is INSIDE SiLU.
#[cube(launch)]
pub fn activate_down(
    input: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] streams: usize,
    #[comptime] low_rank: usize,
) {
    let i = ABSOLUTE_POS;
    if i < low_rank {
        let x = input[i] / streams as f32;
        let y = x * logistic(x);
        if finite(input[i]) && finite(y) {
            output[i] = y;
            status[i] = 1;
        } else {
            output[i] = 0.0;
            status[i] = 0;
        }
    }
}

/// Mix[d] = mean_c(norm[c,d]*sigmoid(up[c,d])). No up scaling.
#[cube(launch)]
pub fn mix(
    normalized: &Array<f32>,
    up: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let d = ABSOLUTE_POS;
    if d < width {
        let mut sum = 0.0_f32;
        let mut good = true;
        for c in 0..streams {
            let i = c * width + d;
            good = good && finite(normalized[i]) && finite(up[i]);
            sum += normalized[i] * logistic(up[i]);
        }
        let y = sum / streams as f32;
        if good && finite(y) {
            output[d] = y;
            status[d] = 1;
        } else {
            output[d] = 0.0;
            status[d] = 0;
        }
    }
}

/// Combine ORIGINAL residual + block[d]*2*sigmoid(injection[c]/HC).
#[cube(launch)]
pub fn combine(
    residual: &Array<f32>,
    block: &Array<f32>,
    injection: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < streams * width {
        let c = i / width;
        let d = i % width;
        let weight = 2.0 * logistic(injection[c] / streams as f32);
        let y = residual[i] + block[d] * weight;
        if finite(residual[i]) && finite(block[d]) && finite(injection[c]) && finite(y) {
            output[i] = y;
            status[i] = 1;
        } else {
            output[i] = 0.0;
            status[i] = 0;
        }
    }
}

/// Pure-host reproduction of the RMS f32 lane/reduction schedule.
pub fn reference_rms(
    input: &[f32],
    gamma: &[f32],
    width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, &'static str> {
    if width == 0
        || input.is_empty()
        || !input.len().is_multiple_of(width)
        || input.len() != gamma.len()
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !input.iter().chain(gamma).all(|x| x.is_finite())
    {
        return Err("invalid reference RMS input");
    }
    let mut output = vec![0.0; input.len()];
    for (g, values) in input.chunks_exact(width).enumerate() {
        let mut scales = [1.0_f32; LANES];
        for (i, x) in values.iter().enumerate() {
            scales[i % LANES] = scales[i % LANES].max(x.abs());
        }
        let mut stride = 32;
        while stride > 0 {
            for i in 0..stride {
                scales[i] = scales[i].max(scales[i + stride]);
            }
            stride /= 2;
        }
        let scale = scales[0];
        let mut sums = [0.0_f32; LANES];
        for (i, x) in values.iter().enumerate() {
            let v = x / scale;
            sums[i % LANES] += v * v;
        }
        stride = 32;
        while stride > 0 {
            for i in 0..stride {
                sums[i] += sums[i + stride];
            }
            stride /= 2;
        }
        let denom = (sums[0] / width as f32 + (epsilon / scale) / scale).sqrt();
        for (d, x) in values.iter().enumerate() {
            let i = g * width + d;
            output[i] = (x / scale) / denom * gamma[i];
        }
    }
    if !output.iter().all(|x| x.is_finite()) {
        return Err("reference RMS result non-finite");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::ops;
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            assert!(
                (a - b).abs() <= 2e-5 + 2e-4 * b.abs(),
                "element {i}: {a} != {b}"
            );
        }
    }
    #[test]
    fn scaled_rms_matches_f64_oracle_full_partial_and_actual_groups() {
        for hc in [1, 2, 4] {
            for width in [1, 5, 63, 64, 65, 128, 2560] {
                let x: Vec<f32> = (0..hc * width)
                    .map(|i| ((i % 43) as f32 - 21.0) / 8.0)
                    .collect();
                let gamma: Vec<f32> = (0..hc * width)
                    .map(|i| 0.25 + (i % 17) as f32 / 16.0)
                    .collect();
                close(
                    &reference_rms(&x, &gamma, width, 1e-6).unwrap(),
                    &ops::rms_norm(&x, &gamma, width, 1e-6).unwrap(),
                );
            }
        }
        for x in [[f32::MAX, -f32::MAX], [0.0, 0.0], [1e-30, -1e-30]] {
            close(
                &reference_rms(&x, &[1.0, 2.0], 2, 1e-6).unwrap(),
                &ops::rms_norm(&x, &[1.0, 2.0], 2, 1e-6).unwrap(),
            );
        }
    }
    #[test]
    fn independent_gating_fixtures_distinguish_scale_and_original_residual() {
        let shape = GatedResidualShape::new(2, 2).unwrap();
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let down = [-8.0, -2.0, 0.0, 2.0, 8.0];
        let expected: Vec<f32> = down
            .iter()
            .map(|&x| {
                let v = f64::from(x) / 2.0;
                (v * sigmoid(v)) as f32
            })
            .collect();
        close(&shape.activate_down(&down).unwrap(), &expected);
        let norm = [1., 2., 3., 4.];
        let up = [-2., 1., 0.5, -1.];
        let expected: Vec<f32> = (0..2)
            .map(|d| {
                ((f64::from(norm[d]) * sigmoid(f64::from(up[d]))
                    + f64::from(norm[2 + d]) * sigmoid(f64::from(up[2 + d])))
                    / 2.0) as f32
            })
            .collect();
        close(&shape.mix(&norm, &up).unwrap(), &expected);
        let original = [10., 20., 30., 40.];
        let block = [2., -3.];
        let injection = [-4., 2.];
        let expected: Vec<f32> = (0..4)
            .map(|i| {
                (f64::from(original[i])
                    + f64::from(block[i % 2]) * 2.0 * sigmoid(f64::from(injection[i / 2]) / 2.0))
                    as f32
            })
            .collect();
        close(
            &shape.combine(&original, &block, &injection).unwrap(),
            &expected,
        );
    }
    #[test]
    fn layout_and_download_reject_bad_shape_and_numerics() {
        let l = ResidualLayout::new(GatedResidualShape::new(4, 2560).unwrap(), 320).unwrap();
        assert_eq!(
            (
                l.residual_elements,
                l.norm_groups,
                l.down_groups,
                l.mix_groups,
                l.combine_groups
            ),
            (10240, 4, 5, 40, 160)
        );
        assert!(l.validate_norm(10240, 10240, 1e-6).is_ok());
        assert!(l.validate_norm(10240, 2560, 1e-6).is_err());
        assert!(l.validate_norm(10240, 10240, 0.0).is_err());
        assert!(
            ResidualLayout::new(
                GatedResidualShape {
                    streams: usize::MAX,
                    width: 2
                },
                320
            )
            .is_err()
        );
        assert!(
            ResidualLayout::new(
                GatedResidualShape {
                    streams: 1,
                    width: 1
                },
                0
            )
            .is_err()
        );
        assert!(validate_download(&[0.; 2], &[1; 2], 2, 2).is_ok());
        assert!(validate_download(&[f32::NAN, 0.], &[1; 2], 2, 2).is_err());
        assert!(validate_download(&[0.; 2], &[1, 0], 2, 2).is_err());
        assert!(reference_rms(&[f32::INFINITY], &[1.], 1, 1e-6).is_err());
    }
    #[test]
    fn all_four_kernels_lower_to_wgsl_offline() {
        use cubecl::wgpu::{WgpuRuntime, WgslCompiler};
        use cubecl::{Compiler, WgpuCompilationOptions};
        for (hc, width, lr) in [(2, 5, 3), (4, 2560, 320)] {
            for which in 0..4 {
                let mut b = KernelBuilder::default();
                b.runtime_properties(WgpuRuntime::target_properties());
                AddressType::U32.register(&mut b.scope);
                let f = f32::as_type(&b.scope);
                let u = u32::as_type(&b.scope);
                let a = b.input_array(f).into();
                match which {
                    0 => {
                        let gamma = b.input_array(f).into();
                        let out = b.output_array(f).into();
                        let s = b.output_array(u).into();
                        let eps = f32::__expand_new(&mut b.scope, 1e-6);
                        grouped_rms_norm::expand(&mut b.scope, a, gamma, out, s, eps, hc, width);
                    }
                    1 => {
                        let out = b.output_array(f).into();
                        let s = b.output_array(u).into();
                        activate_down::expand(&mut b.scope, a, out, s, hc, lr);
                    }
                    2 => {
                        let up = b.input_array(f).into();
                        let out = b.output_array(f).into();
                        let s = b.output_array(u).into();
                        mix::expand(&mut b.scope, a, up, out, s, hc, width);
                    }
                    _ => {
                        let block = b.input_array(f).into();
                        let inject = b.input_array(f).into();
                        let out = b.output_array(f).into();
                        let s = b.output_array(u).into();
                        combine::expand(&mut b.scope, a, block, inject, out, s, hc, width);
                    }
                }
                let name = ["grouped_rms_norm", "activate_down", "mix", "combine"][which];
                let def = b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name(name),
                );
                let shader = WgslCompiler::default()
                    .compile(
                        def,
                        &WgpuCompilationOptions::default(),
                        ExecutionMode::Checked,
                        AddressType::U32.unsigned_type(),
                    )
                    .unwrap();
                let source = shader.to_string();
                assert!(source.contains(&format!("fn {name}")));
                assert!(!source.contains("enable f16"));
                assert!(!source.contains("f64"));
            }
        }
    }
}
