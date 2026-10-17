//! Connected MIT Strata fb58e0 Intel verify_kernels.dp.cpp GDN family.
//! Native conv/L2 and four-row-group FMA recurrence/norm share the same MAX8
//! producers and accepted-prefix replay. Rust owns all private state.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
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
#[path = "strata_prompt_gdn.rs"]
pub(crate) mod prompt;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Ab,
    AbJoin,
    Conv,
    ConvVerify,
    ConvCommit,
    Step,
    Verify,
    Replay,
    CommitFault,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Ab | Self::AbJoin | Self::CommitFault => 64,
            Self::Conv | Self::ConvVerify | Self::Replay => 128,
            Self::ConvCommit => 256,
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
            Self::Ab => "AB",
            Self::AbJoin => "AB_JOIN",
            Self::Conv | Self::ConvVerify => "CONV",
            Self::ConvCommit => "HISTORY",
            Self::Step | Self::Verify => "STEP",
            Self::Replay => "REPLAY",
            Self::CommitFault => "COMMIT_FAULT",
        }));
        if self == Self::Verify {
            source = source.replace(
                "const STORE_STATE:bool=true;",
                "const STORE_STATE:bool=false;",
            );
        }
        if self == Self::ConvVerify {
            source = source.replace(
                "const STORE_HISTORY:bool=true;",
                "const STORE_HISTORY:bool=false;",
            );
        }
        source
    }
}
/// Native default commit: four independent 32-column tiles per value head.
pub(crate) fn replay_grid(heads: usize) -> Result<[usize; 3]> {
    if !(1..=48).contains(&heads) {
        return Err(Error::InvalidBufferLength);
    }
    Ok([heads, 4, 1])
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Ab => "StrataGdnNativeAbMulti",
            Self::AbJoin => "StrataGdnNativeAbJoin",
            Self::Conv => "StrataGdnConvL2",
            Self::ConvVerify => "StrataGdnVerifyConvL2",
            Self::ConvCommit => "StrataGdnAcceptedHistory",
            Self::Step => "StrataGdnStepNorm",
            Self::Verify => "StrataGdnVerifyOutputs",
            Self::Replay => "StrataGdnAcceptedPrefixReplay",
            Self::CommitFault => "StrataGdnCommitFault",
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
    launch_impl(arena, kernel, grid, handles, info, false)
}
pub(crate) fn launch_commit(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    if !matches!(
        kernel,
        Kernel::ConvCommit | Kernel::Replay | Kernel::CommitFault
    ) {
        return Err(Error::InvalidBufferLength);
    }
    launch_impl(arena, kernel, grid, handles, info, true)
}
fn launch_impl(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
    commit: bool,
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || grid.contains(&0)
        || grid.iter().any(|&x| x > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || !(1..=8).contains(&info[0])
        || !(1..=48).contains(&info[4])
        || kernel == Kernel::Ab
        || kernel == Kernel::Verify && info[0] < 2
        || kernel == Kernel::ConvVerify && info[0] < 2
        || kernel == Kernel::ConvCommit && grid != [(info[2] as usize).div_ceil(256), 1, 1]
        || kernel == Kernel::Replay && grid != replay_grid(info[4] as usize)?
        || kernel != Kernel::AbJoin
            && (!(1..=16).contains(&info[3])
                || info[2] != 128 * (2 * info[3] + info[4])
                || !info[4].is_multiple_of(info[3]))
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
    let mut metadata = MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    );
    // AB's closing status packet uses zero in this word. Conv/step/replay
    // packets carry the actual position even where one entry does not read it.
    if kernel != Kernel::AbJoin {
        metadata = metadata.with_native_parameter(
            1,
            cubecl_runtime::server::NativeStrataValue::Position { offset: 0 },
        );
    }
    if commit {
        metadata = metadata
            .with_native_parameter(0, cubecl_runtime::server::NativeStrataValue::AcceptedCount);
    }
    args = args.with_info(metadata);
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
    [
        Kernel::Ab,
        Kernel::AbJoin,
        Kernel::Conv,
        Kernel::ConvVerify,
        Kernel::ConvCommit,
        Kernel::Step,
        Kernel::Verify,
        Kernel::Replay,
        Kernel::CommitFault,
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
pub(crate) struct Ab<'a> {
    pub arena: &'a PrivateArena,
    pub input: BufferId,
    pub controls: BufferId,
    pub status: BufferId,
    pub fault: BufferId,
    pub params: Handle,
    pub count: usize,
    pub columns: usize,
    pub heads: usize,
    pub first: usize,
    pub stride: usize,
}
pub(crate) fn ab(request: Ab<'_>, weights: [Handle; 2]) -> Result<()> {
    let Ab {
        arena,
        input,
        controls,
        status,
        fault,
        params,
        count,
        columns,
        heads,
        first,
        stride,
    } = request;
    let ids = [input, controls, status, fault];
    if !crate::strata_intel_core::enabled()
        || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
        || columns != 2560
        || !(1..=48).contains(&heads)
        || u::words(arena, fault)? != 1
        || u::words(arena, controls)? != 8 * heads * 2
        || u::words(arena, status)? != 8 * heads
    {
        return Err(Error::InvalidBufferLength);
    }
    crate::resident_batch_projection::validate_view(
        count,
        first,
        stride,
        columns,
        u::words(arena, input)?,
    )?;
    let mut handles = weights.to_vec();
    handles.push(u::handle(arena, input)?);
    handles.push(params);
    handles.push(u::handle(arena, controls)?);
    handles.push(u::handle(arena, status)?);
    handles.push(u::handle(arena, fault)?);
    if handles.iter().any(|h| h.stream != handles[0].stream)
        || weights
            .iter()
            .any(|h| h.size_in_used() < ((columns * heads * 2) as u64))
        || handles[3].size_in_used() < ((heads * 2 * 4) as u64)
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(vec![
        count as u64 | ((first as u64) << 32),
        stride as u64 | ((columns as u64) << 32),
        heads as u64,
        0,
    ]));
    // SAFETY: actual admitted distinct private roots, original resident BF16
    // matrices and validated MAX8 input intervals remain owned through completion.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(Kernel::Ab),
            CubeCount::Static(heads as u32, 1, 1),
            args,
        );
    }
    launch(
        arena,
        Kernel::AbJoin,
        [(count * heads).div_ceil(64), 1, 1],
        vec![u::handle(arena, status)?, u::handle(arena, fault)?],
        [count as u32, 0, 0, 0, heads as u32, 1, 0, 0],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn history_only_commit_matches_streamed_raw_history_for_every_prefix() {
        // Distinct raw versus activated values catch accidental use of hbuf.
        for channels in [128usize, 256, 640, 8192] {
            let old = (0..channels * 3)
                .map(|i| i as f32 + 1000.)
                .collect::<Vec<_>>();
            let raw = (0..8 * channels)
                .map(|i| -(i as f32) - 0.5)
                .collect::<Vec<_>>();
            for accepted in 1..=8 {
                let mut streamed = old.clone();
                for t in 0..accepted {
                    for c in 0..channels {
                        streamed[c * 3] = streamed[c * 3 + 1];
                        streamed[c * 3 + 1] = streamed[c * 3 + 2];
                        streamed[c * 3 + 2] = raw[t * channels + c];
                    }
                }
                let mut candidate = vec![f32::NAN; old.len()];
                let mut visits = vec![0u8; channels];
                for group in 0..channels.div_ceil(256) {
                    for lane in 0..256 {
                        let c = group * 256 + lane;
                        if c >= channels {
                            continue;
                        }
                        visits[c] += 1;
                        for j in 0..3 {
                            let at = accepted + j;
                            candidate[c * 3 + j] = if at < 3 {
                                old[c * 3 + at]
                            } else {
                                raw[(at - 3) * channels + c]
                            };
                        }
                    }
                }
                assert_eq!(candidate, streamed);
                assert!(visits.iter().all(|&n| n == 1));
            }
        }
    }

    #[test]
    fn default_commit_tiles_cover_every_native_state_cell_once() {
        assert!(super::replay_grid(0).is_err());
        assert!(super::replay_grid(49).is_err());
        assert_eq!(super::Kernel::Replay.lanes(), 128);
        for heads in 1..=48 {
            let [gx, gy, gz] = super::replay_grid(heads).unwrap();
            assert_eq!((gy, gz), (4, 1));
            let mut visits = vec![0u8; 128 * heads * 128];
            for head in 0..gx {
                for tile in 0..gy {
                    for tid in 0..128 {
                        let col = tile * 32 + tid % 32;
                        let rg = tid / 32;
                        for r in 0..32 {
                            visits[((rg * 32 + r) * heads + head) * 128 + col] += 1;
                        }
                    }
                }
            }
            assert!(visits.iter().all(|&v| v == 1));
        }
    }

    #[test]
    fn tiled_commit_all_prefixes_match_independent_full_matrix_recurrence() {
        // Public synthetic fixtures only: multiple value heads sharing each
        // key head exercise both column tiling and row/value-head state order.
        let heads = 4;
        let key_heads = 2;
        let initial = (0..128 * heads * 128)
            .map(|i| ((i * 13 % 43) as f32 - 21.) / 193.)
            .collect::<Vec<_>>();
        let keys = (0..8 * key_heads * 128)
            .map(|i| ((i * 7 % 29) as f32 - 14.) / 197.)
            .collect::<Vec<_>>();
        let values = (0..8 * heads * 128)
            .map(|i| ((i * 19 % 53) as f32 - 26.) / 89.)
            .collect::<Vec<_>>();
        for accepted in 1..=8 {
            let mut candidate = vec![f32::NAN; initial.len()];
            let mut oracle = initial.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
            for t in 0..accepted {
                for head in 0..heads {
                    let decay = (-0.03 * (t + head + 1) as f32).exp();
                    let beta = 0.17 + head as f32 / 31.;
                    for col in 0..128 {
                        let prediction = (0..128)
                            .map(|row| {
                                oracle[(row * heads + head) * 128 + col]
                                    * f64::from(
                                        keys[t * key_heads * 128 + (head % key_heads) * 128 + row],
                                    )
                            })
                            .sum::<f64>();
                        let delta = (f64::from(values[t * heads * 128 + head * 128 + col])
                            - f64::from(decay) * prediction)
                            * f64::from(beta);
                        for row in 0..128 {
                            let at = (row * heads + head) * 128 + col;
                            oracle[at] = f64::from(decay) * oracle[at]
                                + f64::from(
                                    keys[t * key_heads * 128 + (head % key_heads) * 128 + row],
                                ) * delta;
                        }
                    }
                }
            }
            for head in 0..heads {
                for tile in 0..4 {
                    for local_col in 0..32 {
                        let col = tile * 32 + local_col;
                        let mut state = std::array::from_fn::<_, 128, _>(|row| {
                            initial[(row * heads + head) * 128 + col]
                        });
                        for t in 0..accepted {
                            let key = &keys[t * key_heads * 128 + (head % key_heads) * 128..];
                            let decay = (-0.03 * (t + head + 1) as f32).exp();
                            let beta = 0.17 + head as f32 / 31.;
                            let mut partial = [0f32; 4];
                            for rg in 0..4 {
                                for r in 0..32 {
                                    let row = rg * 32 + r;
                                    partial[rg] = state[row].mul_add(key[row], partial[rg]);
                                }
                            }
                            let prediction = ((partial[0] + partial[1]) + partial[2]) + partial[3];
                            let delta = (values[t * heads * 128 + head * 128 + col]
                                - decay * prediction)
                                * beta;
                            for row in 0..128 {
                                state[row] = decay.mul_add(state[row], key[row] * delta);
                            }
                        }
                        for row in 0..128 {
                            candidate[(row * heads + head) * 128 + col] = state[row];
                        }
                    }
                }
            }
            for (&got, expected) in candidate.iter().zip(oracle) {
                assert!(got.is_finite() && (f64::from(got) - expected).abs() < 2e-7);
            }
        }
    }

    #[test]
    fn packed_bf16_ab_window_matches_independent_f64_dot_and_keeps_each_rhs() {
        let columns = 2560usize;
        let weights = (0..columns)
            .map(|i| half::bf16::from_f32((i as i32 * 19 % 37 - 18) as f32 / 91.))
            .collect::<Vec<_>>();
        let inputs = (0..8)
            .map(|t| {
                (0..columns)
                    .map(|i| ((i * 13 + t * 17) % 41) as f32 / 137. - 0.1)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for count in [1, 3, 5, 8] {
            let mut acc = [[0.0f32; 8]; 32];
            let mut visits = vec![0u8; columns];
            for (lane, row) in acc.iter_mut().enumerate() {
                for pack in (lane..columns / 8).step_by(32) {
                    for pair in 0..4 {
                        let d = pack * 8 + pair * 2;
                        let bits =
                            weights[d].to_bits() as u32 | ((weights[d + 1].to_bits() as u32) << 16);
                        let a = f32::from_bits(bits << 16);
                        let b = f32::from_bits(bits & 0xffff0000);
                        for t in 0..count {
                            row[t] = a.mul_add(inputs[t][d], row[t]);
                            row[t] = b.mul_add(inputs[t][d + 1], row[t]);
                        }
                        visits[d] += 1;
                        visits[d + 1] += 1;
                    }
                }
            }
            assert!(visits.iter().all(|&n| n == 1));
            for distance in [16, 8, 4, 2, 1] {
                let prior = acc;
                for lane in 0..32 {
                    for t in 0..count {
                        acc[lane][t] = prior[lane][t] + prior[lane ^ distance][t];
                    }
                }
            }
            for t in 0..count {
                let oracle = weights
                    .iter()
                    .zip(&inputs[t])
                    .map(|(w, &x)| w.to_f32() as f64 * x as f64)
                    .sum::<f64>();
                assert!((acc[0][t] as f64 - oracle).abs() < 3e-6);
                assert!(acc.iter().all(|r| r[t].to_bits() == acc[0][t].to_bits()));
            }
            assert!(acc.iter().all(|r| r[count..].iter().all(|&v| v == 0.)));
        }
    }
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
