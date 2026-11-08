//! Optional fixedMAX8 GDN pointwise chunk operators. No commit or graph hook.
//! Exact f32 arithmetic/reduction schedule translated from gdn_pointwise.rs;
//! core ops.rs supplies an independent f64 oracle. Public S/K/H specialize;
//! actual count, all token/history bases and strides are runtime scalars.
//! Old RAW history is immutable; candidate history is returned separately.
use cubecl::prelude::*;
use kuno_text::ops::GdnShape;
pub const MAX_BATCH: usize = 8;
const LANES: usize = 64;
pub const EPSILON: f32 = 1e-6;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: GdnShape,
    pub channels: usize,
    pub qk: usize,
    pub values: usize,
}
impl Layout {
    pub fn new(shape: GdnShape) -> Result<Self, &'static str> {
        GdnShape::new(shape.head_dim, shape.key_heads, shape.value_heads)
            .map_err(|_| "pointwise shape")?;
        if shape.head_dim > 128 || shape.key_heads > 16 || shape.value_heads > 48 {
            return Err("pointwise shape bound");
        }
        Ok(Self {
            shape,
            channels: (2 * shape.key_heads + shape.value_heads) * shape.head_dim,
            qk: shape.key_heads * shape.head_dim,
            values: shape.value_heads * shape.head_dim,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub base: usize,
    pub stride: usize,
    pub width: usize,
    pub words: usize,
}
impl Span {
    pub fn new(width: usize, base: usize, gap: usize) -> Result<Self, &'static str> {
        let stride = width.checked_add(gap).ok_or("span overflow")?;
        let words = MAX_BATCH
            .checked_mul(stride)
            .and_then(|n| n.checked_add(base))
            .and_then(|n| n.checked_add(13))
            .ok_or("span overflow")?;
        let s = Self {
            base,
            stride,
            width,
            words,
        };
        s.validate(MAX_BATCH, words)?;
        Ok(s)
    }
    pub fn validate(self, count: usize, bound: usize) -> Result<(), &'static str> {
        if !(1..=MAX_BATCH).contains(&count)
            || self.width == 0
            || self.stride < self.width
            || self.words > bound
            || bound > u32::MAX as usize
        {
            return Err("span bound");
        }
        let end = (MAX_BATCH - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.base))
            .and_then(|n| n.checked_add(self.width))
            .ok_or("span overflow")?;
        if end > self.words {
            return Err("span full capacity");
        }
        Ok(())
    }
    pub fn at(self, t: usize) -> usize {
        self.base + t * self.stride
    }
    pub fn active(self, count: usize, i: usize) -> bool {
        i >= self.base
            && (i - self.base) / self.stride < count
            && (i - self.base) % self.stride < self.width
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct History {
    pub base: usize,
    pub stride: usize,
    pub channels: usize,
    pub words: usize,
}
impl History {
    pub fn new(channels: usize, base: usize, stride: usize) -> Result<Self, &'static str> {
        let words = channels
            .checked_mul(stride)
            .and_then(|n| n.checked_add(base))
            .and_then(|n| n.checked_add(13))
            .ok_or("history overflow")?;
        let h = Self {
            base,
            stride,
            channels,
            words,
        };
        h.validate(words)?;
        Ok(h)
    }
    pub fn validate(self, bound: usize) -> Result<(), &'static str> {
        if self.channels == 0
            || self.stride < 3
            || self.words > bound
            || [self.base, self.stride, self.channels, self.words, bound]
                .into_iter()
                .any(|n| n > u32::MAX as usize)
        {
            return Err("history bound");
        }
        let end = (self.channels - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(self.base))
            .and_then(|n| n.checked_add(3))
            .ok_or("history overflow")?;
        if end > self.words {
            return Err("history capacity");
        }
        Ok(())
    }
    pub fn at(self, c: usize) -> usize {
        self.base + c * self.stride
    }
    pub fn active(self, i: usize) -> bool {
        i >= self.base
            && (i - self.base) / self.stride < self.channels
            && (i - self.base) % self.stride < 3
    }
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

/// One lane owns one raw channel and advances its three oldest-first taps.
/// A bad later token invalidates earlier outputs for this channel too.
#[cube(launch)]
pub fn conv_silu(
    input: &Array<f32>,
    weights: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    out: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    input_base: usize,
    input_stride: usize,
    old_base: usize,
    old_stride: usize,
    next_base: usize,
    next_stride: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] channels: usize,
) {
    let c = ABSOLUTE_POS;
    if c < channels {
        let mut history = Array::<f32>::new(3_usize);
        let mut good = true;
        for i in 0..3_usize {
            history[i] = old[old_base + c * old_stride + i];
            good = good && shader_finite(history[i]);
        }
        for t in 0..count {
            status[status_base + t * status_stride + c] = 0;
            out[out_base + t * out_stride + c] = 0.0;
            let current = input[input_base + t * input_stride + c];
            good = good && shader_finite(current);
            let mut sum = 0.0_f32;
            #[unroll]
            for tap in 0..4_usize {
                let mut value = current;
                if tap < 3 {
                    value = history[tap];
                }
                let w = weights[c * 4 + tap];
                good = good && shader_finite(w);
                sum += value * w;
            }
            let y = sum * shader_sigmoid(sum);
            good = good && shader_finite(sum) && shader_finite(y);
            if good {
                history[0_usize] = history[1_usize];
                history[1_usize] = history[2_usize];
                history[2_usize] = current;
                out[out_base + t * out_stride + c] = y;
                status[status_base + t * status_stride + c] = 1;
            }
        }
        for i in 0..3_usize {
            if good {
                next[next_base + c * next_stride + i] = history[i];
            } else {
                next[next_base + c * next_stride + i] = 0.0;
            }
        }
        if !good {
            for t in 0..count {
                out[out_base + t * out_stride + c] = 0.0;
                status[status_base + t * status_stride + c] = 0;
            }
        }
    }
}

/// Separate Q/K/V token spans, exactly h%K-ready GGUF order. Barriers uniform.
#[cube(launch)]
pub fn prepare_qkv(
    activation: &Array<f32>,
    query: &mut Array<f32>,
    key: &mut Array<f32>,
    value: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    count: usize,
    in_base: usize,
    in_stride: usize,
    q_base: usize,
    q_stride: usize,
    k_base: usize,
    k_stride: usize,
    v_base: usize,
    v_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] key_heads: usize,
    #[comptime] value_heads: usize,
) {
    let head = CUBE_POS_X as usize;
    let token = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let total = 2 * key_heads + value_heads;
    let mut x0 = 0.0_f32;
    let mut x1 = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut valid = true;
    if token < count && head < total {
        if lane < head_dim {
            x0 = activation[in_base + token * in_stride + head * head_dim + lane];
        }
        if lane + LANES < head_dim {
            x1 = activation[in_base + token * in_stride + head * head_dim + lane + LANES];
        }
        valid = shader_finite(x0) && shader_finite(x1) && shader_finite(epsilon) && epsilon > 0.0;
        if head < 2 * key_heads {
            sum = x0 * x0 + x1 * x1;
        }
        valid = valid && shader_finite(sum);
    }
    let mut squares = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    squares[lane] = sum;
    let mut flag = 0_u32;
    if valid {
        flag = 1_u32;
    }
    flags[lane] = flag;
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
    if token < count && head < total {
        let mut y0 = x0;
        let mut y1 = x1;
        let mut ok = flags[0] == 1;
        if head < 2 * key_heads {
            let den = squares[0] + epsilon;
            let inv = 1.0 / den.sqrt();
            ok = ok
                && shader_finite(squares[0])
                && shader_finite(den)
                && den > 0.0
                && shader_finite(inv);
            y0 = x0 * inv;
            y1 = x1 * inv;
            if head < key_heads {
                y0 /= (head_dim as f32).sqrt();
                y1 /= (head_dim as f32).sqrt();
            }
        }
        ok = ok && shader_finite(y0) && shader_finite(y1);
        #[unroll]
        for half in 0..2_usize {
            let d = lane + half * LANES;
            if d < head_dim {
                let mut y = y0;
                if half == 1 {
                    y = y1;
                }
                let mut emitted = 0.0_f32;
                let mut emitted_flag = 0_u32;
                if ok {
                    emitted = y;
                    emitted_flag = 1_u32;
                }
                if head < key_heads {
                    query[q_base + token * q_stride + head * head_dim + d] = emitted;
                } else if head < 2 * key_heads {
                    key[k_base + token * k_stride + (head - key_heads) * head_dim + d] = emitted;
                } else {
                    value[v_base + token * v_stride + (head - 2 * key_heads) * head_dim + d] =
                        emitted;
                }
                status[status_base + token * status_stride + head * head_dim + d] = emitted_flag;
            }
        }
    }
}

/// Stored A is already negative-exp; exactly softplus(alpha+dt)*A.
// CubeCL does not lower RangeInclusive::contains; retain explicit scalar comparisons.
#[allow(clippy::manual_range_contains)]
#[cube(launch)]
pub fn prepare_controls(
    alpha: &Array<f32>,
    beta: &Array<f32>,
    params: &Array<f32>,
    out: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    a_base: usize,
    a_stride: usize,
    b_base: usize,
    b_stride: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] heads: usize,
) {
    let t = CUBE_POS_Y as usize;
    let h = CUBE_POS_X as usize * LANES + UNIT_POS_X as usize;
    if t < count && h < heads {
        out[out_base + t * out_stride + h] = 0.0;
        out[out_base + t * out_stride + heads + h] = 0.0;
        status[status_base + t * status_stride + h] = 0;
        let a = alpha[a_base + t * a_stride + h];
        let b = beta[b_base + t * b_stride + h];
        let dt = params[h];
        let stored = params[heads + h];
        let biased = a + dt;
        let decay = shader_softplus(biased) * stored;
        let probability = shader_sigmoid(b);
        if shader_finite(a)
            && shader_finite(b)
            && shader_finite(dt)
            && shader_finite(stored)
            && stored < 0.0
            && shader_finite(biased)
            && shader_finite(decay)
            && decay <= 0.0
            && shader_finite(probability)
            && probability >= 0.0
            && probability <= 1.0
        {
            out[out_base + t * out_stride + h] = decay;
            out[out_base + t * out_stride + heads + h] = probability;
            status[status_base + t * status_stride + h] = 1;
        }
    }
}

/// Per-value head RMS and sigmoid(z), with public shared/per-value gamma.
#[cube(launch)]
pub fn output_gate(
    input: &Array<f32>,
    z: &Array<f32>,
    gamma: &Array<f32>,
    out: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    count: usize,
    in_base: usize,
    in_stride: usize,
    z_base: usize,
    z_stride: usize,
    out_base: usize,
    out_stride: usize,
    status_base: usize,
    status_stride: usize,
    #[comptime] head_dim: usize,
    #[comptime] heads: usize,
    #[comptime] shared_gamma: bool,
) {
    let head = CUBE_POS_X as usize;
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let mut x0 = 0.0_f32;
    let mut x1 = 0.0_f32;
    let mut z0 = 0.0_f32;
    let mut z1 = 0.0_f32;
    let mut g0 = 0.0_f32;
    let mut g1 = 0.0_f32;
    let mut valid = true;
    if t < count && head < heads {
        if lane < head_dim {
            let d = head * head_dim + lane;
            x0 = input[in_base + t * in_stride + d];
            z0 = z[z_base + t * z_stride + d];
            if comptime!(shared_gamma) {
                g0 = gamma[lane];
            } else {
                g0 = gamma[d];
            }
        }
        if lane + LANES < head_dim {
            let d = head * head_dim + lane + LANES;
            x1 = input[in_base + t * in_stride + d];
            z1 = z[z_base + t * z_stride + d];
            if comptime!(shared_gamma) {
                g1 = gamma[lane + LANES];
            } else {
                g1 = gamma[d];
            }
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
    let mut squares = SharedMemory::<f32>::new(LANES);
    let mut flags = SharedMemory::<u32>::new(LANES);
    squares[lane] = sum;
    let mut flag = 0_u32;
    if valid && shader_finite(sum) {
        flag = 1_u32;
    }
    flags[lane] = flag;
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
    if t < count && head < heads {
        let denominator = squares[0] / head_dim as f32 + epsilon;
        let inverse = 1.0 / denominator.sqrt();
        let n0 = x0 * inverse * g0;
        let n1 = x1 * inverse * g1;
        let y0 = n0 * shader_sigmoid(z0);
        let y1 = n1 * shader_sigmoid(z1);
        let ok = flags[0] == 1
            && shader_finite(squares[0])
            && shader_finite(denominator)
            && denominator > 0.0
            && shader_finite(inverse)
            && shader_finite(n0)
            && shader_finite(n1)
            && shader_finite(y0)
            && shader_finite(y1);
        #[unroll]
        for half in 0..2_usize {
            let d = lane + half * LANES;
            if d < head_dim {
                let i = head * head_dim + d;
                out[out_base + t * out_stride + i] = 0.0;
                status[status_base + t * status_stride + i] = 0;
                let mut y = y0;
                if half == 1 {
                    y = y1;
                }
                if ok {
                    out[out_base + t * out_stride + i] = y;
                    status[status_base + t * status_stride + i] = 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shader(
        which: usize,
        s: usize,
        k: usize,
        h: usize,
        shared: bool,
        mode: cubecl::server::ExecutionMode,
    ) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<f32>(f);
        b.scope.register_type::<u32>(u);
        b.scope.register_type::<usize>(u);
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        b.scope
            .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
        let x = b.input_array(Type::new(f)).into();
        let out = b.output_array(Type::new(f)).into();
        let flag = b.output_array(Type::new(u)).into();
        let n = if which < 2 { 11 } else { 9 };
        let a: Vec<NativeExpand<usize>> = (0..n).map(|_| b.scalar(u).into()).collect();
        match which {
            0 => {
                let weights = b.input_array(Type::new(f)).into();
                let old = b.input_array(Type::new(f)).into();
                let next = b.output_array(Type::new(f)).into();
                conv_silu::expand(
                    &mut b.scope,
                    x,
                    weights,
                    old,
                    next,
                    out,
                    flag,
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
                    (2 * k + h) * s,
                );
            }
            1 => {
                let key = b.output_array(Type::new(f)).into();
                let value = b.output_array(Type::new(f)).into();
                let eps = b.scalar(f).into();
                prepare_qkv::expand(
                    &mut b.scope,
                    x,
                    out,
                    key,
                    value,
                    flag,
                    eps,
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
                    s,
                    k,
                    h,
                );
            }
            2 => {
                let beta = b.input_array(Type::new(f)).into();
                let params = b.input_array(Type::new(f)).into();
                prepare_controls::expand(
                    &mut b.scope,
                    x,
                    beta,
                    params,
                    out,
                    flag,
                    a[0].clone(),
                    a[1].clone(),
                    a[2].clone(),
                    a[3].clone(),
                    a[4].clone(),
                    a[5].clone(),
                    a[6].clone(),
                    a[7].clone(),
                    a[8].clone(),
                    h,
                );
            }
            _ => {
                let z = b.input_array(Type::new(f)).into();
                let gamma = b.input_array(Type::new(f)).into();
                let eps = b.scalar(f).into();
                output_gate::expand(
                    &mut b.scope,
                    x,
                    z,
                    gamma,
                    out,
                    flag,
                    eps,
                    a[0].clone(),
                    a[1].clone(),
                    a[2].clone(),
                    a[3].clone(),
                    a[4].clone(),
                    a[5].clone(),
                    a[6].clone(),
                    a[7].clone(),
                    a[8].clone(),
                    s,
                    h,
                    shared,
                );
            }
        }
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("batch_pointwise"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                AddressType::U32.unsigned_type(),
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn all_pointwise_shaders_validate_bounds_tails_barriers_and_runtime_scalars_offline() {
        use cubecl::server::ExecutionMode;
        for (s, k, h) in [
            (1, 1, 1),
            (7, 2, 6),
            (127, 1, 3),
            (128, 1, 3),
            (128, 16, 48),
        ] {
            for which in 0..4 {
                for shared in [false, true] {
                    for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                        let source = shader(which, s, k, h, shared, mode);
                        let module =
                            wgpu::naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
                                panic!("which={which} shape={s}/{k}/{h}: {e}\n{source}")
                            });
                        wgpu::naga::valid::Validator::new(
                            wgpu::naga::valid::ValidationFlags::all(),
                            wgpu::naga::valid::Capabilities::empty(),
                        )
                        .validate(&module)
                        .unwrap_or_else(|e| {
                            panic!("which={which} shape={s}/{k}/{h}: {e}\n{source}")
                        });
                        assert!(source.contains("info.scalars_u32[0]"));
                        assert!(
                            !source.contains("subgroup")
                                && !source.contains("enable f16")
                                && !source.contains("fma(")
                        );
                        if which == 1 || which == 3 {
                            assert!(source.contains("workgroupBarrier"));
                            let flag = source
                                .lines()
                                .find_map(|line| {
                                    line.trim()
                                        .strip_suffix(" = u32(0);")
                                        .filter(|name| name.starts_with("l_mut_"))
                                })
                                .expect("initialized flag must survive");
                            assert!(source.contains(&format!("{flag} = u32(1);")), "{source}");
                            assert!(source.contains(&format!("] = {flag};")), "{source}");
                        }
                    }
                }
            }
        }
    }
}
