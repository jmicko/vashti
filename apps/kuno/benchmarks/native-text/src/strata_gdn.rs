//! Connected MIT Strata fb58e0 Intel verify_kernels.dp.cpp GDN family.
//! Native conv/L2 and four-row-group FMA recurrence/norm share the same MAX8
//! producers and accepted-prefix replay. Rust owns all private state.
use crate::{private_arena::PrivateArena, resident_util as u};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{CubeCount, CubeDim},
    wgpu::AutoCompiler,
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
    Conv,
    Controls,
    Step,
    Replay,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Conv => 128,
            Self::Controls => 64,
            _ => 512,
        }
    }
    fn source(self) -> String {
        let all = include_str!("strata_gdn.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut source = section("COMMON").to_owned();
        source.push_str(section(match self {
            Self::Conv => "CONV",
            Self::Controls => "CONTROLS",
            Self::Step | Self::Replay => "STEP",
        }));
        if matches!(self, Self::Replay) {
            source = source.replace(
                "const STATE_ONLY:bool=false;",
                "const STATE_ONLY:bool=true;",
            );
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Conv => "StrataGdnConvL2",
            Self::Controls => "StrataGdnControls",
            Self::Step => "StrataGdnStepNorm",
            Self::Replay => "StrataGdnAcceptedPrefixReplay",
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
/// All roots are validated by the parent as distinct, full, admitted pages.
/// INFO is runtime geometry; no prompt or result is copied to host memory.
pub(crate) fn launch(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || grid.contains(&0)
        || grid.iter().any(|&x| x > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || !(1..=8).contains(&info[0])
        || !(1..=16).contains(&info[3])
        || !(1..=48).contains(&info[4])
        || info[2] != 128 * (2 * info[3] + info[4])
        || !info[4].is_multiple_of(info[3])
    {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    let hw = &client.properties().hardware;
    if kernel.lanes() > hw.max_units_per_cube || kernel.lanes() > hw.max_cube_dim.0 {
        return Err(Error::InvalidFormat("native GDN device workgroup capacity"));
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
    // SAFETY: authenticated GDN parent validates fixed widths and complete private
    // owners before enqueue; kernels additionally bound runtime active intervals.
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
    [Kernel::Conv, Kernel::Controls, Kernel::Step, Kernel::Replay]
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
    #[test]
    fn native_four_group_fma_math_and_every_accepted_prefix_match_independent_state_oracle() {
        let initial = std::array::from_fn::<_, 128, _>(|i| ((i * 13 % 17) as f32 - 8.) / 53.);
        let run = |count: usize| {
            let mut state = initial;
            let mut oracle = initial.map(f64::from);
            let mut out = Vec::new();
            for t in 0..count {
                let key = std::array::from_fn::<_, 128, _>(|i| {
                    (((i * 7 + t * 19) % 31) as f32 - 15.) / 97.
                });
                let query = std::array::from_fn::<_, 128, _>(|i| {
                    (((i * 11 + t * 3) % 23) as f32 - 11.) / 79.
                });
                let decay = (-0.07 * (t + 1) as f32).exp();
                let beta = 0.2;
                let value = t as f32 / 11. - 0.4;
                let mut partial = [0f32; 4];
                for rg in 0..4 {
                    for r in 0..32 {
                        let i = rg * 32 + r;
                        partial[rg] = state[i].mul_add(key[i], partial[rg]);
                    }
                }
                let predicted = ((partial[0] + partial[1]) + partial[2]) + partial[3];
                let delta = (value - decay * predicted) * beta;
                partial.fill(0.);
                for rg in 0..4 {
                    for r in 0..32 {
                        let i = rg * 32 + r;
                        state[i] = decay.mul_add(state[i], key[i] * delta);
                        partial[rg] = state[i].mul_add(query[i], partial[rg]);
                    }
                }
                let output =
                    (((partial[0] + partial[1]) + partial[2]) + partial[3]) * (1. / 128f32.sqrt());
                out.push(output);
                let prediction = oracle
                    .iter()
                    .zip(key)
                    .map(|(&s, k)| s * k as f64)
                    .sum::<f64>()
                    * decay as f64;
                let delta64 = (value as f64 - prediction) * beta as f64;
                for i in 0..128 {
                    oracle[i] = decay as f64 * oracle[i] + key[i] as f64 * delta64;
                    assert!((state[i] as f64 - oracle[i]).abs() < 2e-7);
                }
                let expected = oracle
                    .iter()
                    .zip(query)
                    .map(|(&s, q)| s * q as f64)
                    .sum::<f64>()
                    / 128f64.sqrt();
                assert!((output as f64 - expected).abs() < 2e-7);
            }
            (state, out)
        };
        for count in 1..=8 {
            let (_, whole) = run(count);
            for accepted in 1..=count {
                let (_, prefix) = run(accepted);
                assert_eq!(prefix, whole[..accepted]);
            }
        }
    }
}
