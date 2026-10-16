//! MIT Strata fb58e0 native PLE projections/postops/history, Rust-owned graph.
//! Existing authentic private/public owners remain the only data roots.
use crate::{private_arena::PrivateArena, resident_util as u};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{ComputeClient, CubeCount, CubeDim},
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Norm,
    Gate,
    Broadcast,
    Conv,
    History,
    Projection,
    PromptNorm,
    PromptNormInPlace,
    PromptGate,
    PromptBroadcast,
    PromptConv,
    PromptHistory,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Norm | Self::PromptNorm | Self::PromptNormInPlace => 1024,
            Self::Gate | Self::PromptGate => 512,
            _ => 256,
        }
    }
    fn source(self) -> String {
        let all = include_str!("strata_ple.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut s = section("COMMON").to_owned();
        s.push_str(section(match self {
            Self::Norm | Self::PromptNorm | Self::PromptNormInPlace => "NORM",
            Self::Gate | Self::PromptGate => "GATE",
            Self::Broadcast | Self::PromptBroadcast => "BROADCAST",
            Self::Conv | Self::PromptConv => "CONV",
            Self::History | Self::PromptHistory => "HISTORY",
            Self::Projection => "PROJECTION",
        }));
        let edits: &[(&str, &str)] = match self {
            Self::PromptNorm | Self::PromptNormInPlace => &[(
                "if sw!=32u || t>=p.count || h>=4u {return;}",
                "if sw!=32u{if tid==0u&&t<p.count&&h<4u{status[t*p.ss+h]=0u;}return;}if t>=p.count||h>=4u{return;}",
            )],
            Self::PromptGate => &[(
                "if sw!=32u || h>=4u || t>=p.count {return;}",
                "if sw!=32u{if tid==0u&&t<p.count&&h<4u{status[t*p.ss+h]=0u;}return;}if h>=4u||t>=p.count{return;}",
            )],
            Self::PromptBroadcast => &[
                (
                    "@binding(3) var<storage,read_write> status:array<u32>;",
                    "@binding(3) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[t*10240u+c]=select(0u,1u,finite(y));",
                    "if !finite(y){atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::PromptHistory => &[
                (
                    "@binding(3) var<storage,read_write> status:array<u32>;",
                    "@binding(3) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[c]=select(0u,1u,ok);",
                    "if !ok{atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::PromptConv => &[
                (
                    "@group(0) @binding(5) var<storage,read_write> conv:array<f32>;\n",
                    "",
                ),
                (
                    "@binding(6) var<storage,read_write> result:array<f32>;",
                    "@binding(5) var<storage,read_write> result:array<f32>;",
                ),
                (
                    "@binding(7) var<storage,read_write> conv_status:array<u32>;",
                    "@binding(6) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "@group(0) @binding(8) var<storage,read_write> result_status:array<u32>;\n",
                    "",
                ),
                ("@binding(9) var<uniform>", "@binding(7) var<uniform>"),
                (
                    "conv[t*10240u+c]=select(0.0,activation,ok&&finite(activation));",
                    "",
                ),
                (
                    "conv_status[t*10240u+c]=select(0u,1u,ok&&finite(activation));result_status[t*10240u+c]=select(0u,1u,ok&&finite(y));",
                    "if !ok||!finite(activation)||!finite(y){atomicOr(&fault[0],1u);}",
                ),
            ],
            _ => &[],
        };
        for &(old, new) in edits {
            assert!(s.contains(old), "prompt PLE source drift");
            s = s.replace(old, new);
        }
        if self == Self::PromptNormInPlace {
            // The original key is both source and destination. Use ONE writable
            // descriptor; do not bind a read-only alias of that same interval.
            for (old, new) in [
                (
                    "@binding(0) var<storage,read> input:array<f32>;",
                    "@binding(0) var<storage,read_write> values:array<f32>;",
                ),
                (
                    "@group(0) @binding(2) var<storage,read_write> output:array<f32>;\n",
                    "",
                ),
                (
                    "@binding(3) var<storage,read_write> status:",
                    "@binding(2) var<storage,read_write> status:",
                ),
                ("@binding(4) var<uniform>", "@binding(3) var<uniform>"),
                ("input[base+col]", "values[base+col]"),
                ("output[p.ob+t*p.os+h*2560u+col]", "values[base+col]"),
            ] {
                assert!(s.contains(old), "in-place PLE source drift");
                s = s.replace(old, new);
            }
        }
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Norm => "StrataPleNativeNorm",
            Self::Gate => "StrataPleNativeGate",
            Self::Broadcast => "StrataPleNativeBroadcast",
            Self::Conv => "StrataPleNativeConv",
            Self::History => "StrataPleNativeHistory",
            Self::Projection => "StrataNativeBf16Multi",
            Self::PromptNorm => "StrataPromptPleNorm",
            Self::PromptNormInPlace => "StrataPromptPleNormInPlace",
            Self::PromptGate => "StrataPromptPleGate",
            Self::PromptBroadcast => "StrataPromptPleBroadcast",
            Self::PromptConv => "StrataPromptPleConv",
            Self::PromptHistory => "StrataPromptPleHistory",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(self.lanes()))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 12],
) -> Result<()> {
    launch_client(
        arena.client().map_err(u::fault)?,
        kernel,
        grid,
        handles,
        info,
    )
}
pub(crate) fn launch_client(
    client: &ComputeClient<WgpuRuntime>,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 12],
) -> Result<()> {
    launch_impl(client, kernel, grid, handles, info, false)
}
pub(crate) fn launch_commit_history(
    arena: &PrivateArena,
    handles: Vec<Handle>,
    info: [u32; 12],
) -> Result<()> {
    launch_impl(
        arena.client().map_err(u::fault)?,
        Kernel::PromptHistory,
        [40, 1, 1],
        handles,
        info,
        true,
    )
}
fn launch_impl(
    client: &ComputeClient<WgpuRuntime>,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 12],
    commit: bool,
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || !(1..=if matches!(
            kernel,
            Kernel::PromptNorm
                | Kernel::PromptNormInPlace
                | Kernel::PromptGate
                | Kernel::PromptBroadcast
                | Kernel::PromptConv
                | Kernel::PromptHistory
        ) && crate::strata_prompt::active()
        {
            crate::strata_prompt::groups::CAPACITY
        } else {
            8
        })
            .contains(&(info[0] as usize))
        || kernel == Kernel::PromptNormInPlace
            && (info[1] != info[3] || info[2] != 10240 || info[4] != 10240)
        || grid.contains(&0)
        || grid.iter().any(|&g| g > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || client.properties().hardware.max_units_per_cube < kernel.lanes()
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    let mut metadata = MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    );
    if commit {
        metadata = metadata
            .with_native_parameter(0, cubecl_runtime::server::NativeStrataValue::AcceptedCount);
    }
    args = args.with_info(metadata);
    // SAFETY: actual connected owner validates source roles, distinct complete
    // pages, live intervals and retained lifetime before enqueue/publication.
    unsafe {
        client.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Norm,
        Kernel::Gate,
        Kernel::Broadcast,
        Kernel::Conv,
        Kernel::History,
        Kernel::Projection,
        Kernel::PromptNorm,
        Kernel::PromptNormInPlace,
        Kernel::PromptGate,
        Kernel::PromptBroadcast,
        Kernel::PromptConv,
        Kernel::PromptHistory,
    ]
    .into_iter()
    .map(|k| {
        k.compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use kuno_text::ops::{self, CausalConvState, GatedResidualShape};
    fn xor32(mut x: [f32; 32]) -> f32 {
        for d in [16, 8, 4, 2, 1] {
            let old = x;
            for l in 0..32 {
                x[l] += old[l ^ d];
            }
        }
        x[0]
    }
    fn norm(input: &[f32], gamma: &[f32]) -> Vec<f32> {
        let mut out = vec![0.; 10240];
        for h in 0..4 {
            let mut partial = [0f32; 1024];
            for (tid, p) in partial.iter_mut().enumerate() {
                for col in (tid..2560).step_by(1024) {
                    let x = input[h * 2560 + col];
                    *p += x * x;
                }
            }
            let warps =
                std::array::from_fn(|sg| xor32(partial[sg * 32..sg * 32 + 32].try_into().unwrap()));
            let scale = 1. / (xor32(warps) / 2560. + 1e-6).sqrt();
            for c in 0..2560 {
                out[h * 2560 + c] = (scale * input[h * 2560 + c]) * gamma[h * 2560 + c];
            }
        }
        out
    }
    fn gate(key: &[f32], query: &[f32]) -> [f32; 4] {
        std::array::from_fn(|h| {
            let mut lanes = [0f32; 512];
            for (tid, x) in lanes.iter_mut().enumerate() {
                for j in 0..8 {
                    let col = tid + j * 512;
                    let p = if col < 2560 {
                        key[h * 2560 + col] * query[h * 2560 + col]
                    } else {
                        0.
                    };
                    *x += p;
                }
            }
            let warps = std::array::from_fn(|sg| {
                if sg < 16 {
                    xor32(lanes[sg * 32..sg * 32 + 32].try_into().unwrap())
                } else {
                    0.
                }
            });
            let s = (1. / 2560f32.sqrt()).mul_add(xor32(warps), 0.);
            let root = if s == 0. {
                0.
            } else {
                s.signum() * s.abs().max(1e-6).sqrt()
            };
            1. / (1. + (-root).exp())
        })
    }
    #[test]
    fn native_postops_and_every_history_prefix_match_independent_scalar_oracle() {
        const D: usize = 10240;
        let gamma = (0..D)
            .map(|i| 1. + (i % 19) as f32 / 113.)
            .collect::<Vec<_>>();
        let weights = (0..D * 4)
            .map(|i| half::f16::from_f32(((i * 17 % 41) as f32 - 20.) / 61.).to_f32())
            .collect::<Vec<_>>();
        let shape = GatedResidualShape {
            width: 2560,
            streams: 4,
        };
        for count in 1..=8 {
            let mut state = CausalConvState::new(D, 4, 3).unwrap();
            let key = (0..D)
                .map(|i| ((i * 29 % 127) as f32 - 63.) / 43.)
                .collect::<Vec<_>>();
            let hidden = (0..D)
                .map(|i| ((i * 31 % 131) as f32 - 65.) / 51.)
                .collect::<Vec<_>>();
            let value = (0..2560)
                .map(|i| ((i * 13 % 67) as f32 - 33.) / 47.)
                .collect::<Vec<_>>();
            for _ in 0..9 {
                ops::ple_inject_projected(
                    &key, &value, &hidden, &gamma, &gamma, &gamma, &weights, &mut state, shape,
                    1e-6,
                )
                .unwrap();
            }
            let old = state.history().to_vec();
            let mut all = Vec::new();
            for t in 0..count {
                let mut k = key.clone();
                let mut h = hidden.clone();
                let mut v = value.clone();
                for x in &mut k {
                    *x += t as f32 / 17.;
                }
                for x in &mut h {
                    *x -= t as f32 / 23.;
                }
                for x in &mut v {
                    *x += t as f32 / 31.;
                }
                let kn = norm(&k, &gamma);
                let qn = norm(&h, &gamma);
                let gates = gate(&kn, &qn);
                let gated = (0..D)
                    .map(|c| v[c % 2560] * gates[c / 2560])
                    .collect::<Vec<_>>();
                let normalized = norm(&gated, &gamma);
                all.extend_from_slice(&normalized);
                let reference = ops::ple_inject_projected(
                    &k, &v, &h, &gamma, &gamma, &gamma, &weights, &mut state, shape, 1e-6,
                )
                .unwrap();
                for c in 0..D {
                    let mut sum = 0.;
                    for tap in 0..4 {
                        let source = t as isize - 9 + 3 * tap as isize;
                        let x = if source < 0 {
                            old[c * 9 + (9 + source) as usize]
                        } else {
                            all[source as usize * D + c]
                        };
                        let term = x * weights[c * 4 + tap];
                        if tap == 0 {
                            sum = term;
                        } else {
                            sum += term;
                        }
                    }
                    let y = (h[c] + gated[c]) + sum / (1. + (-sum).exp());
                    assert!(
                        (y - reference.residual[c]).abs() < 7e-5 * (1. + y.abs()),
                        "count={count} t={t} c={c}"
                    );
                }
            }
            for prefix in 0..=count {
                let mut expected = old.clone();
                for t in 0..prefix {
                    for c in 0..D {
                        expected[c * 9..c * 9 + 9].copy_within(1..9, 0);
                        expected[c * 9 + 8] = all[t * D + c];
                    }
                }
                for c in 0..D {
                    for r in 0..9 {
                        let source = prefix as isize - 9 + r as isize;
                        let actual = if source >= 0 {
                            all[source as usize * D + c]
                        } else {
                            old[c * 9 + prefix + r]
                        };
                        assert_eq!(actual.to_bits(), expected[c * 9 + r].to_bits());
                    }
                }
            }
        }
        assert_eq!(gate(&[0.; D], &[1.; D]), [0.5; 4]);
    }
    #[test]
    fn native_bf16_multi_pair_order_covers_actual_columns_and_all_rhs() {
        let weights = (0..2560)
            .map(|i| half::bf16::from_f32(((i * 17 % 73) as f32 - 36.) / 57.).to_f32())
            .collect::<Vec<_>>();
        for count in 1..=8 {
            let input = (0..count * 2560)
                .map(|i| ((i * 23 % 101) as f32 - 50.) / 67.)
                .collect::<Vec<_>>();
            for t in 0..count {
                let mut lanes = [0f32; 256];
                let mut visits = [0u8; 2560];
                for (tid, sum) in lanes.iter_mut().enumerate() {
                    for pair in (tid..1280).step_by(256) {
                        for col in [pair * 2, pair * 2 + 1] {
                            *sum = weights[col].mul_add(input[t * 2560 + col], *sum);
                            visits[col] += 1;
                        }
                    }
                }
                let warps = std::array::from_fn(|sg| {
                    if sg < 8 {
                        xor32(lanes[sg * 32..sg * 32 + 32].try_into().unwrap())
                    } else {
                        0.
                    }
                });
                let actual = xor32(warps);
                let expected = weights
                    .iter()
                    .zip(&input[t * 2560..(t + 1) * 2560])
                    .map(|(&w, &x)| w as f64 * x as f64)
                    .sum::<f64>();
                assert_eq!(visits, [1; 2560]);
                assert!((actual as f64 - expected).abs() < 1e-5 * (1. + expected.abs()));
            }
        }
    }

    #[test]
    fn native_router_full_softmax_and_winning_lane_sum_match_independent_oracle() {
        for case in 0..7 {
            let logits = std::array::from_fn::<_, 512, _>(|i| {
                if case == 0 {
                    0.
                } else {
                    ((i * 37 + case * 13) % 149) as f32 / 17. - 4.
                }
            });
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut values = logits.map(|x| (x - maximum).exp());
            let sums = std::array::from_fn(|lane| {
                (0..16)
                    .map(|j| values[lane + j * 32])
                    .fold(0f32, |a, b| a + b)
            });
            let inverse = 1. / xor32(sums);
            for x in &mut values {
                *x *= inverse;
            }
            let mut ids = (0..512).collect::<Vec<_>>();
            ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            ids.truncate(10);
            let mut winning = [0f32; 32];
            for &id in &ids {
                winning[id % 32] += values[id];
            }
            let inverse = 1. / xor32(winning).max(0.00006103515625);
            let weights = ids
                .iter()
                .map(|&id| values[id] * inverse)
                .collect::<Vec<_>>();
            let mut expected = (0..512).collect::<Vec<_>>();
            expected.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            expected.truncate(10);
            assert_eq!(ids, expected);
            let sum = expected
                .iter()
                .map(|&id| ((logits[id] - maximum) as f64).exp())
                .sum::<f64>();
            for (w, &id) in weights.iter().zip(&expected) {
                let target = ((logits[id] - maximum) as f64).exp() / sum;
                assert!((*w as f64 - target).abs() < 1e-6);
            }
            assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
            if case == 0 {
                assert_eq!(ids, (0..10).collect::<Vec<_>>());
            }
        }
    }
}

#[cfg(test)]
mod inplace_prompt_tests {
    use super::*;
    #[test]
    fn actual_inplace_key_norm_has_one_writable_value_binding_and_original_reduction_shape() {
        let kernel = Kernel::PromptNormInPlace;
        assert_eq!(kernel.lanes(), 1024);
        let source = kernel.source();
        let m = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        let mut bindings = m
            .global_variables
            .iter()
            .filter_map(|(_, v)| {
                v.binding
                    .as_ref()
                    .map(|b| (b.binding, v.name.as_deref().unwrap(), v.space))
            })
            .collect::<Vec<_>>();
        bindings.sort_by_key(|v| v.0);
        assert_eq!(
            bindings.iter().map(|v| (v.0, v.1)).collect::<Vec<_>>(),
            vec![(0, "values"), (1, "gamma"), (2, "status"), (3, "p")]
        );
        assert_eq!(
            bindings[0].2,
            wgpu::naga::AddressSpace::Storage {
                access: wgpu::naga::StorageAccess::LOAD | wgpu::naga::StorageAccess::STORE
            }
        );
        assert_eq!(m.entry_points[0].workgroup_size, [1024, 1, 1]);
        // The ordinary out-of-place query/conv normalization remains a distinct
        // interface; no same-buffer duplicate binding is accepted for the key.
        let out = Kernel::PromptNorm.source();
        let m = wgpu::naga::front::wgsl::parse_str(&out).unwrap();
        assert_eq!(
            m.global_variables
                .iter()
                .filter(|(_, v)| v.binding.is_some())
                .count(),
            5
        );
    }
}
