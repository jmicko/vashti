//! Optional MAX_BATCH8 token-major exact-schedule GR pointwise kernels.
//! Translation of residual_gpu.rs at root9649c864 + current admitted source.
//! Model shape alone is comptime; actual count/offset/stride remain runtime.
//! Full bindings remain owned until completed acceptance/erasure. Inactive
//! columns and logical/power-of-two padding receive no stores.
use crate::quant_matvec::matmul::MAX_BATCH;
use cubecl::prelude::*;

pub const LANES: usize = 64;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub first: usize,
    pub stride: usize,
    pub width: usize,
    pub words: usize,
}
impl Span {
    pub fn fixture(width: usize) -> Result<Self, &'static str> {
        if width == 0 {
            return Err("empty batch span");
        }
        let stride = width.checked_add(3).ok_or("batch span overflow")?;
        let words = MAX_BATCH
            .checked_mul(stride)
            .and_then(|n| n.checked_add(12))
            .ok_or("batch span overflow")?;
        let span = Self {
            first: 5,
            stride,
            width,
            words,
        };
        span.validate(MAX_BATCH, words)?;
        Ok(span)
    }
    /// Scalar reference kernels bind logical subviews. First and each token
    /// stride are 256-byte aligned; the public-mode caller checks the actual
    /// device requirement divides 256 before any fixture/model allocations.
    pub fn aligned_fixture(width: usize) -> Result<Self, &'static str> {
        if width == 0 {
            return Err("empty scalar span");
        }
        let stride = width
            .checked_add(3)
            .and_then(|n| n.checked_add(63))
            .map(|n| n / 64 * 64)
            .ok_or("scalar span overflow")?;
        let words = MAX_BATCH
            .checked_mul(stride)
            .and_then(|n| n.checked_add(128))
            .ok_or("scalar span overflow")?;
        let span = Self {
            first: 64,
            stride,
            width,
            words,
        };
        span.validate(MAX_BATCH, words)?;
        Ok(span)
    }
    pub fn validate(self, count: usize, bound: usize) -> Result<(), &'static str> {
        if !(1..=MAX_BATCH).contains(&count)
            || self.width == 0
            || self.width.div_ceil(LANES) > 65_535
            || self.stride < self.width
            || self.words > bound
        {
            return Err("invalid batch span");
        }
        let end = (count - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.first))
            .and_then(|n| n.checked_add(self.width))
            .ok_or("batch span overflow")?;
        if end > self.words
            || [self.first, self.stride, self.words, bound, end]
                .iter()
                .any(|&n| n > u32::MAX as usize)
        {
            return Err("batch span exceeds bound");
        }
        Ok(())
    }
    pub fn index(self, token: usize, element: usize) -> usize {
        self.first + token * self.stride + element
    }
    pub fn grid(self) -> CubeCount {
        CubeCount::Static(self.width.div_ceil(LANES) as u32, MAX_BATCH as u32, 1)
    }
}
#[allow(clippy::manual_range_contains)] // Cube DSL does not lower Range::contains.
#[cube]
fn finite(x: f32) -> bool {
    x <= 3.4028235e38_f32 && x >= -3.4028235e38_f32
}

#[cube]
#[allow(unused_assignments)] // Cube DSL needs a mutable initialized value for both branches.
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
    count: usize,
    input_first: usize,
    input_stride: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
    status_stride: usize,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let group = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut maximum = 1.0_f32;
    let mut good = 1.0_f32;
    let tiles = width.div_ceil(LANES);
    if token < count && group < streams {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if token < count && d < width {
                let i = group * width + d;
                let x = input[input_first + token * input_stride + i];
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
    if token < count && group < streams && goods[0] == 1.0 {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if token < count && d < width {
                let x = input[input_first + token * input_stride + group * width + d] / scale;
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
    if token < count && group < streams {
        for tile in 0..tiles {
            let d = tile * LANES + lane;
            if token < count && d < width {
                let i = group * width + d;
                let y = ((input[input_first + token * input_stride + i] / scale) / denominator)
                    * gamma[i];
                if goods[0] == 1.0 && finite(epsilon) && epsilon > 0.0 && finite(y) {
                    output[output_first + token * output_stride + i] = y;
                } else {
                    output[output_first + token * output_stride + i] = 0.0;
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
    if lane == 0 && token < count && group < streams {
        status[status_first + token * status_stride + group] = output_goods[0] as u32;
    }
}

/// Down is LR elements, not LR*HC. Scaling is INSIDE SiLU.
#[cube(launch)]
pub fn activate_down(
    input: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    input_first: usize,
    input_stride: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
    status_stride: usize,
    #[comptime] streams: usize,
    #[comptime] low_rank: usize,
) {
    let token = CUBE_POS_Y as usize;
    let i = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if token < count && i < low_rank {
        let x = input[input_first + token * input_stride + i] / streams as f32;
        let y = x * logistic(x);
        if finite(input[input_first + token * input_stride + i]) && finite(y) {
            output[output_first + token * output_stride + i] = y;
            status[status_first + token * status_stride + i] = 1;
        } else {
            output[output_first + token * output_stride + i] = 0.0;
            status[status_first + token * status_stride + i] = 0;
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
    count: usize,
    normalized_first: usize,
    normalized_stride: usize,
    up_first: usize,
    up_stride: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
    status_stride: usize,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let d = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if token < count && d < width {
        let mut sum = 0.0_f32;
        let mut good = true;
        for c in 0..streams {
            let i = c * width + d;
            good = good
                && finite(normalized[normalized_first + token * normalized_stride + i])
                && finite(up[up_first + token * up_stride + i]);
            sum += normalized[normalized_first + token * normalized_stride + i]
                * logistic(up[up_first + token * up_stride + i]);
        }
        let y = sum / streams as f32;
        if good && finite(y) {
            output[output_first + token * output_stride + d] = y;
            status[status_first + token * status_stride + d] = 1;
        } else {
            output[output_first + token * output_stride + d] = 0.0;
            status[status_first + token * status_stride + d] = 0;
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
    count: usize,
    residual_first: usize,
    residual_stride: usize,
    block_first: usize,
    block_stride: usize,
    injection_first: usize,
    injection_stride: usize,
    output_first: usize,
    output_stride: usize,
    status_first: usize,
    status_stride: usize,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let i = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if token < count && i < streams * width {
        let c = i / width;
        let d = i % width;
        let weight = 2.0
            * logistic(injection[injection_first + token * injection_stride + c] / streams as f32);
        let y = residual[residual_first + token * residual_stride + i]
            + block[block_first + token * block_stride + d] * weight;
        if finite(residual[residual_first + token * residual_stride + i])
            && finite(block[block_first + token * block_stride + d])
            && finite(injection[injection_first + token * injection_stride + c])
            && finite(y)
        {
            output[output_first + token * output_stride + i] = y;
            status[status_first + token * status_stride + i] = 1;
        } else {
            output[output_first + token * output_stride + i] = 0.0;
            status[status_first + token * status_stride + i] = 0;
        }
    }
}

#[cube(launch)]
pub fn copy_span(
    source: &Array<f32>,
    target: &mut Array<f32>,
    count: usize,
    source_first: usize,
    source_stride: usize,
    target_first: usize,
    target_stride: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let d = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if token < count && d < width {
        target[target_first + token * target_stride + d] =
            source[source_first + token * source_stride + d];
    }
}
#[cube(launch)]
pub fn finite_span(
    input: &Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    input_first: usize,
    input_stride: usize,
    status_first: usize,
    status_stride: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let d = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if token < count && d < width {
        let x = input[input_first + token * input_stride + d];
        let mut flag = 1_u32;
        if !finite(x) {
            flag = 0_u32;
        }
        status[status_first + token * status_stride + d] = flag;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residual_gpu;
    use kuno_text::ops::{self, GatedResidualShape};
    #[test]
    fn complete_batch_spans_reject_zero_overflow_overlap_and_partial_capacity() {
        for width in [1, 5, 65, 2560, 10240] {
            let s = Span::fixture(width).unwrap();
            for count in [1, 3, 5, 8] {
                s.validate(count, s.words).unwrap();
                assert!(s.index(count - 1, width - 1) < s.words);
            }
            assert!(s.validate(0, s.words).is_err());
            assert!(s.validate(9, s.words).is_err());
            assert!(s.validate(8, s.words - 1).is_err());
            assert!(
                Span {
                    stride: width - 1,
                    ..s
                }
                .validate(1, s.words)
                .is_err()
            );
            assert!(
                Span {
                    first: usize::MAX,
                    ..s
                }
                .validate(8, s.words)
                .is_err()
            );
        }
        for width in [0, usize::MAX, u32::MAX as usize] {
            assert!(Span::fixture(width).is_err());
            assert!(Span::aligned_fixture(width).is_err());
        }
    }
    fn logistic(x: f32) -> f32 {
        if x >= 0. {
            1. / (1. + (-x).exp())
        } else {
            let e = x.exp();
            e / (1. + e)
        }
    }
    fn close(a: f32, b: f32) {
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() <= 2e-5 + 2e-4 * b.abs(), "{a} {b}");
    }
    #[test]
    fn token_major_schedule_matches_core_and_keeps_every_inactive_gap() {
        for width in [5, 65, 2560] {
            let shape = GatedResidualShape::new(4, width).unwrap();
            let r = 4 * width;
            let s = Span::fixture(r).unwrap();
            let gamma: Vec<_> = (0..r).map(|i| 0.3 + (i % 19) as f32 / 20.).collect();
            for count in [1, 3, 5, 8] {
                let mut out = vec![-12345.; s.words + 17];
                for token in 0..count {
                    let x: Vec<_> = (0..r)
                        .map(|i| ((i * 3 + token * 11) % 29) as f32 / 16. - 0.7)
                        .collect();
                    let norm = residual_gpu::reference_rms(&x, &gamma, width, 1e-6).unwrap();
                    let core = ops::rms_norm(&x, &gamma, width, 1e-6).unwrap();
                    for (&a, &b) in norm.iter().zip(&core) {
                        close(a, b);
                    }
                    let up: Vec<_> = (0..r)
                        .map(|i| ((i + token) % 17) as f32 / 2. - 4.)
                        .collect();
                    let mixed = shape.mix(&norm, &up).unwrap();
                    let block: Vec<_> = (0..width)
                        .map(|d| ((d + token) % 11) as f32 / 8. - 0.5)
                        .collect();
                    let injection = [-80., -1., 0., 80.];
                    let combined = shape.combine(&x, &block, &injection).unwrap();
                    for d in 0..width {
                        let mut sum = 0_f32;
                        for c in 0..4 {
                            sum += norm[c * width + d] * logistic(up[c * width + d]);
                        }
                        close(sum / 4., mixed[d]);
                    }
                    for i in 0..r {
                        let y =
                            x[i] + block[i % width] * (2. * logistic(injection[i / width] / 4.));
                        close(y, combined[i]);
                        out[s.index(token, i)] = y;
                    }
                    let down = [-f32::MAX, -80., -4., 0., 4., 80.];
                    let low = shape.activate_down(&down).unwrap();
                    for (&x, &y) in down.iter().zip(&low) {
                        let v = x / 4.;
                        close(v * logistic(v), y);
                    }
                }
                for (i, &v) in out.iter().enumerate() {
                    let relative = i.saturating_sub(s.first);
                    let active =
                        i >= s.first && relative / s.stride < count && relative % s.stride < r;
                    if !active {
                        assert_eq!(v, -12345.);
                    }
                }
            }
        }
    }
    fn shader(
        which: usize,
        hc: usize,
        width: usize,
        lr: usize,
        mode: cubecl::server::ExecutionMode,
    ) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let mut k = KernelBuilder::default();
        k.scope.register_type::<u32>(uint);
        k.scope.register_type::<usize>(uint);
        k.scope.register_type::<f32>(float);
        k.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        k.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        let x = k.input_array(Type::new(float)).into();
        let z = k.output_array(Type::new(float)).into();
        let flags = k.output_array(Type::new(uint)).into();
        let n = match which {
            0 | 1 => 7,
            2 => 9,
            3 => 11,
            _ => 5,
        };
        let a: Vec<NativeExpand<usize>> = (0..n).map(|_| k.scalar(uint).into()).collect();
        match which {
            0 => {
                let gamma = k.input_array(Type::new(float)).into();
                let eps = k.scalar(float).into();
                grouped_rms_norm::expand(
                    &mut k.scope,
                    x,
                    gamma,
                    z,
                    flags,
                    eps,
                    a[0].clone(),
                    a[1].clone(),
                    a[2].clone(),
                    a[3].clone(),
                    a[4].clone(),
                    a[5].clone(),
                    a[6].clone(),
                    hc,
                    width,
                );
            }
            1 => activate_down::expand(
                &mut k.scope,
                x,
                z,
                flags,
                a[0].clone(),
                a[1].clone(),
                a[2].clone(),
                a[3].clone(),
                a[4].clone(),
                a[5].clone(),
                a[6].clone(),
                hc,
                lr,
            ),
            2 => {
                let up = k.input_array(Type::new(float)).into();
                mix::expand(
                    &mut k.scope,
                    x,
                    up,
                    z,
                    flags,
                    a[0].clone(),
                    a[1].clone(),
                    a[2].clone(),
                    a[3].clone(),
                    a[4].clone(),
                    a[5].clone(),
                    a[6].clone(),
                    a[7].clone(),
                    a[8].clone(),
                    hc,
                    width,
                );
            }
            3 => {
                let block = k.input_array(Type::new(float)).into();
                let inject = k.input_array(Type::new(float)).into();
                combine::expand(
                    &mut k.scope,
                    x,
                    block,
                    inject,
                    z,
                    flags,
                    a[0].clone(),
                    a[1].clone(),
                    a[2].clone(),
                    a[3].clone(),
                    a[4].clone(),
                    a[5].clone(),
                    a[6].clone(),
                    a[7].clone(),
                    a[8].clone(),
                    a[9].clone(),
                    a[10].clone(),
                    hc,
                    width,
                );
            }
            4 => copy_span::expand(
                &mut k.scope,
                x,
                z,
                a[0].clone(),
                a[1].clone(),
                a[2].clone(),
                a[3].clone(),
                a[4].clone(),
                width,
            ),
            _ => finite_span::expand(
                &mut k.scope,
                x,
                flags,
                a[0].clone(),
                a[1].clone(),
                a[2].clone(),
                a[3].clone(),
                a[4].clone(),
                width,
            ),
        }
        WgslCompiler::default()
            .compile(
                k.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("batch_gr"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                AddressType::U32.unsigned_type(),
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn all_batch_kernels_validate_offline_and_actual_count_stays_runtime() {
        use cubecl::server::ExecutionMode;
        for (hc, width, lr) in [(4, 65, 7), (4, 2560, 320)] {
            for which in 0..6 {
                for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                    let s = shader(which, hc, width, lr, mode);
                    let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                    assert!(s.contains("scalars_u32[0]"));
                    assert!(
                        !s.contains("enable f16") && !s.contains("subgroup") && !s.contains("fma(")
                    );
                    if which == 0 {
                        assert!(s.contains("workgroupBarrier"));
                    }
                    if which == 5 {
                        // Follow the actual initialized mutable flag through its
                        // failure assignment and final storage write. Checking
                        // @compute or unrelated 0/1 literals misses lost stores.
                        let flag = s
                            .lines()
                            .find_map(|line| {
                                line.trim()
                                    .strip_suffix(" = u32(1);")
                                    .filter(|name| name.starts_with("l_mut_"))
                            })
                            .expect("finite success flag must survive lowering");
                        assert!(s.contains(&format!("{flag} = u32(0);")), "{s}");
                        assert!(s.contains(&format!("] = {flag};")), "{s}");
                    }
                }
            }
        }
    }
}
