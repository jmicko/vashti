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
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Norm => 1024,
            Self::Gate => 512,
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
            Self::Norm => "NORM",
            Self::Gate => "GATE",
            Self::Broadcast => "BROADCAST",
            Self::Conv => "CONV",
            Self::History => "HISTORY",
            Self::Projection => "PROJECTION",
        }));
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
    if !crate::strata_intel_core::enabled()
        || !(1..=8).contains(&info[0])
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
    args = args.with_info(MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    ));
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
