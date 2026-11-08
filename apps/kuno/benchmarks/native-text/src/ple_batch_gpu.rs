//! Candidate-only PLE MAX8 pointwise tail. No graph hook or state commit.
//! RMS schedule is the accepted Rust residual64-lane tree; PLE's independent
//! scalar oracle remains core ops::ple_inject_projected (f64 reductions).
//! Only public stream/width geometry specializes. Every private base/stride,
//! actual count and epsilon are runtime; token planes never flatten into X.
use cubecl::prelude::*;
pub const LANES: usize = 64;

#[allow(clippy::manual_range_contains)]
#[cube]
fn finite(x: f32) -> bool {
    x <= 3.4028235e38_f32 && x >= -3.4028235e38_f32
}
#[cube]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[allow(clippy::collapsible_if)]
#[cube(launch)]
pub fn rms(
    input: &Array<f32>,
    gamma: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    epsilon: f32,
    count: usize,
    ib: usize,
    is_: usize,
    gb: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
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
            if d < width {
                let i = group * width + d;
                let x = input[ib + token * is_ + i];
                let w = gamma[gb + i];
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
            if d < width {
                let x = input[ib + token * is_ + group * width + d] / scale;
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
            if d < width {
                let i = group * width + d;
                let y = ((input[ib + token * is_ + i] / scale) / denominator) * gamma[gb + i];
                if goods[0] == 1.0 && finite(epsilon) && epsilon > 0.0 && finite(y) {
                    output[ob + token * os + i] = y;
                } else {
                    output[ob + token * os + i] = 0.0;
                    output_good = 0.0;
                }
            }
        }
    }
    // Separate output flags preserve the immutable input reduction while other
    // lanes emit their tiles; do not reuse shared goods before all reads end.
    let mut out_goods = SharedMemory::<f32>::new(LANES);
    out_goods[lane] = output_good;
    sync_cube();
    stride = 32;
    while stride > 0 {
        if lane < stride {
            if out_goods[lane + stride] < out_goods[lane] {
                out_goods[lane] = out_goods[lane + stride];
            }
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && token < count && group < streams {
        status[sb + token * ss + group] = out_goods[0] as u32;
    }
}

/// Original signed-root dot gate. sign(0)=0, therefore sigmoid(0)=0.5.
#[cube(launch)]
pub fn gate(
    key: &Array<f32>,
    query: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    kb: usize,
    ks: usize,
    qb: usize,
    qs: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let stream = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut sum = 0.0_f32;
    let mut good = 1_u32;
    if token < count && stream < streams {
        for tile in 0..width.div_ceil(LANES) {
            let d = tile * LANES + lane;
            if d < width {
                let i = stream * width + d;
                let x = key[kb + token * ks + i];
                let y = query[qb + token * qs + i];
                let product = x * y;
                if finite(x) && finite(y) && finite(product) {
                    sum += product;
                } else {
                    good = 0;
                }
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(LANES);
    let mut goods = SharedMemory::<u32>::new(LANES);
    sums[lane] = sum;
    goods[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
            goods[lane] = goods[lane] & goods[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && token < count && stream < streams {
        let score = sums[0] / (width as f32).sqrt();
        let mut root = 0.0_f32;
        if score > 0.0 {
            root = score.abs().max(1e-6).sqrt();
        }
        if score < 0.0 {
            root = -score.abs().max(1e-6).sqrt();
        }
        let y = sigmoid(root);
        let mut flag = 0_u32;
        if goods[0] == 1 && finite(score) && finite(y) {
            output[ob + token * os + stream] = y;
            flag = 1;
        } else {
            output[ob + token * os + stream] = 0.0;
        }
        status[sb + token * ss + stream] = flag;
    }
}

#[cube(launch)]
pub fn gated(
    value: &Array<f32>,
    gates: &Array<f32>,
    output: &mut Array<f32>,
    status: &mut Array<u32>,
    count: usize,
    vb: usize,
    vs: usize,
    gb: usize,
    gs: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
    #[comptime] streams: usize,
    #[comptime] width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let i = ABSOLUTE_POS_X as usize;
    if token < count && i < streams * width {
        let v = value[vb + token * vs + i % width];
        let g = gates[gb + token * gs + i / width];
        let y = g * v;
        let mut flag = 0_u32;
        if finite(v) && finite(g) && finite(y) {
            output[ob + token * os + i] = y;
            flag = 1;
        } else {
            output[ob + token * os + i] = 0.0;
        }
        status[sb + token * ss + i] = flag;
    }
}

/// Each X lane owns one channel and walks active tokens in causal order. Tap
/// order9/6/3/0 and (R+gated)+SiLU(conv) are explicit. OLD never receives a store.
/// Candidate stores the last9 NORMALIZED values, not embeddings/raw gated rows.
#[cube(launch)]
pub fn conv_tail(
    normalized: &Array<f32>,
    weights: &Array<f32>,
    old: &Array<f32>,
    next: &mut Array<f32>,
    residual: &Array<f32>,
    gated_value: &Array<f32>,
    conv: &mut Array<f32>,
    output: &mut Array<f32>,
    conv_status: &mut Array<u32>,
    out_status: &mut Array<u32>,
    history_status: &mut Array<u32>,
    count: usize,
    nb: usize,
    ns: usize,
    wb: usize,
    hb: usize,
    hs: usize,
    cb: usize,
    cs: usize,
    rb: usize,
    rs: usize,
    gb: usize,
    gs: usize,
    vb: usize,
    vs: usize,
    ob: usize,
    os: usize,
    sb: usize,
    ss: usize,
    tb: usize,
    ts: usize,
    fb: usize,
    #[comptime] channels: usize,
) {
    let c = ABSOLUTE_POS_X as usize;
    if c < channels {
        let mut history_good = 1_u32;
        for j in 0..9 {
            if !finite(old[hb + c * hs + j]) {
                history_good = 0;
            }
        }
        for tap in 0..4 {
            if !finite(weights[wb + c * 4 + tap]) {
                history_good = 0;
            }
        }
        for token in 0..count {
            let mut sum = 0.0_f32;
            let mut good = history_good;
            for tap in 0..4 {
                let lag = (3 - tap) * 3;
                let x = if token >= lag {
                    normalized[nb + (token - lag) * ns + c]
                } else {
                    old[hb + c * hs + 9 + token - lag]
                };
                let w = weights[wb + c * 4 + tap];
                let product = x * w;
                if finite(x) && finite(product) {
                    sum += product;
                } else {
                    good = 0;
                }
            }
            let y = sum * sigmoid(sum);
            let r = residual[rb + token * rs + c];
            let g = gated_value[gb + token * gs + c];
            let z = (r + g) + y;
            let mut cv = 0.0_f32;
            let mut out = 0.0_f32;
            let mut cflag = 0_u32;
            let mut oflag = 0_u32;
            if good == 1 && finite(sum) && finite(y) {
                cv = y;
                cflag = 1;
            }
            if cflag == 1 && finite(r) && finite(g) && finite(z) {
                out = z;
                oflag = 1;
            }
            conv[vb + token * vs + c] = cv;
            output[ob + token * os + c] = out;
            conv_status[sb + token * ss + c] = cflag;
            out_status[tb + token * ts + c] = oflag;
        }
        for j in 0..9 {
            let x = if j + count < 9 {
                old[hb + c * hs + j + count]
            } else {
                normalized[nb + (j + count - 9) * ns + c]
            };
            let mut y = 0.0_f32;
            if finite(x) {
                y = x;
            } else {
                history_good = 0;
            }
            next[cb + c * cs + j] = y;
        }
        history_status[fb + c] = history_good;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shader(
        which: usize,
        streams: usize,
        width: usize,
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
        let y = k.input_array(Type::new(float)).into();
        let z = k.output_array(Type::new(float)).into();
        let flags = k.output_array(Type::new(uint)).into();
        let n = match which {
            0 => 8,
            1 | 2 => 9,
            _ => 21,
        };
        let a: Vec<NativeExpand<usize>> = (0..n).map(|_| k.scalar(uint).into()).collect();
        match which {
            0 => {
                let eps = k.scalar(float).into();
                rms::expand(
                    &mut k.scope,
                    x,
                    y,
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
                    a[7].clone(),
                    streams,
                    width,
                );
            }
            1 => gate::expand(
                &mut k.scope,
                x,
                y,
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
                streams,
                width,
            ),
            2 => gated::expand(
                &mut k.scope,
                x,
                y,
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
                streams,
                width,
            ),
            _ => {
                let old = k.input_array(Type::new(float)).into();
                let next = k.output_array(Type::new(float)).into();
                let residual = k.input_array(Type::new(float)).into();
                let gv = k.input_array(Type::new(float)).into();
                let output = k.output_array(Type::new(float)).into();
                let out_status = k.output_array(Type::new(uint)).into();
                let history_status = k.output_array(Type::new(uint)).into();
                conv_tail::expand(
                    &mut k.scope,
                    x,
                    y,
                    old,
                    next,
                    residual,
                    gv,
                    z,
                    output,
                    flags,
                    out_status,
                    history_status,
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
                    a[11].clone(),
                    a[12].clone(),
                    a[13].clone(),
                    a[14].clone(),
                    a[15].clone(),
                    a[16].clone(),
                    a[17].clone(),
                    a[18].clone(),
                    a[19].clone(),
                    a[20].clone(),
                    streams * width,
                );
            }
        }
        WgslCompiler::default()
            .compile(
                k.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("ple_max8"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                AddressType::U32.unsigned_type(),
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn all_ple_kernels_validate_without_fma_private_specialization_or_flattened_planes() {
        use cubecl::server::ExecutionMode;
        for (streams, width) in [(3, 97), (4, 2560)] {
            for which in 0..4 {
                for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                    let s = shader(which, streams, width, mode);
                    let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                    assert!(s.contains("scalars_u32[0]"));
                    assert!(
                        !s.contains("fma(") && !s.contains("enable f16") && !s.contains("subgroup")
                    );
                    if which <= 1 {
                        assert!(s.contains("workgroupBarrier"));
                    }
                    if which == 2 {
                        assert!(s.contains("global_id.x"));
                        assert!(s.contains("workgroup_id.y"));
                        assert!(!s.contains("global_id.y *"));
                    }
                }
            }
        }
    }
    #[test]
    fn complete_dispatch_inventory_fits_runtime_info_stage_bound() {
        let source = include_str!("ple_batch_gpu.rs");
        for (name, arrays, scalars) in [
            ("rms", 4, 9),
            ("gate", 4, 9),
            ("gated", 4, 9),
            ("conv_tail", 11, 21),
        ] {
            let sig = source
                .split(&format!("pub fn {name}("))
                .nth(1)
                .unwrap()
                .split("#[comptime]")
                .next()
                .unwrap();
            assert_eq!(sig.matches("Array<").count(), arrays);
            assert_eq!(
                sig.matches(": usize").count() + sig.matches(": f32").count(),
                scalars
            );
            let mut info = cubecl::InfoBuilder::default();
            let address = cubecl::ir::AddressType::U64;
            for _ in 0..arrays {
                info.metadata.register_array(131072, 131072, address);
            }
            for _ in 0..scalars {
                info.scalars.push(1u64);
            }
            assert!(info.finish(address).data.len() * 8 <= 512);
        }
        let source = include_str!("ple_chunk.rs");
        let tail = source
            .split("pub fn enqueue_tail(")
            .nth(1)
            .unwrap()
            .split("#[derive(")
            .next()
            .unwrap();
        assert_eq!(tail.matches("p::rms::launch").count(), 2);
        assert_eq!(tail.matches("p::gate::launch").count(), 1);
        assert_eq!(tail.matches("p::gated::launch").count(), 1);
        assert_eq!(tail.matches("p::conv_tail::launch").count(), 1);
        assert_eq!(crate::ple_chunk::MAX_BATCH, 8);
    }
}
