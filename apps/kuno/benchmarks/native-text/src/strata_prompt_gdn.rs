//! MIT Strata fb58e0 Intel prefill/kernels.dp.cpp's connected prompt GDN.
//! Contiguous token-major H, causal conv/L2, CB32 pipelined recurrence and
//! native FP16 norm output are selected together. Existing double-buffered
//! head-major state and split replay inputs are explicit owner substitutions.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Conv,
    History,
    L2,
    Split,
    Recurrence,
    Norm,
}
impl Op {
    fn source(self) -> String {
        let all = include_str!("strata_prompt_gdn.wgsl");
        let part = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        format!(
            "{}\n{}",
            part("COMMON"),
            part(match self {
                Self::Conv => "CONV",
                Self::History => "HISTORY",
                Self::L2 => "L2",
                Self::Split => "SPLIT",
                Self::Recurrence => "RECURRENCE",
                Self::Norm => "NORM",
            })
        )
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Conv => "StrataPromptGdnCausalConv",
            Self::History => "StrataPromptGdnHistory",
            Self::L2 => "StrataPromptGdnL2",
            Self::Split => "StrataPromptGdnReplayViews",
            Self::Recurrence => "StrataPromptGdnPipelinedColumns",
            Self::Norm => "StrataPromptGdnF16Norm",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(128))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Op {
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
            debug_info: None,
            repr: None,
            source: self.source(),
            cube_dim: CubeDim::new_1d(128),
        })
    }
}
fn launch(
    arena: &PrivateArena,
    op: Op,
    grid: [usize; 3],
    handles: Vec<Handle>,
    p: [u32; 8],
) -> Result<()> {
    if !crate::strata_prompt::active()
        || !(1..=4096).contains(&p[0])
        || p[3] == 0
        || p[3] > 16
        || p[4] == 0
        || p[4] > 48
        || !p[4].is_multiple_of(p[3])
        || p[2] != 128 * (2 * p[3] + p[4])
        || p[5] > 1
        || p[7] != 0
        || !f32::from_bits(p[6]).is_finite()
        || f32::from_bits(p[6]) <= 0.
        || grid.contains(&0)
        || grid.iter().any(|&v| v > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        p.chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    // SAFETY: actual parent geometry/owner bounds are checked below. The
    // private H, state, replay, status and half-image consumers share one graph.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
fn span(h: &Handle, bytes: usize) -> Result<()> {
    if h.size_in_used() < bytes as u64 {
        Err(Error::InvalidBufferLength)
    } else {
        Ok(())
    }
}
/// Raw,w,old,next,H,Q,K,V,conv status,prepared status: original real owners.
pub(crate) fn convolution(arena: &PrivateArena, p: [u32; 8], h: [Handle; 10]) -> Result<()> {
    let t = p[0] as usize;
    let c = p[2] as usize;
    let k = p[3] as usize * 128;
    let v = p[4] as usize * 128;
    for (id, n) in [
        (0, t * c),
        (1, c * 4),
        (2, c * 3),
        (3, c * 3),
        (4, t * c),
        (5, t * k),
        (6, t * k),
        (7, t * v),
        (8, t * c),
        (9, t * c),
    ] {
        span(&h[id], n * 4)?;
    }
    launch(
        arena,
        Op::Conv,
        [c / 128, t.div_ceil(64), 1],
        vec![
            h[0].clone(),
            h[1].clone(),
            h[2].clone(),
            h[4].clone(),
            h[8].clone(),
            h[9].clone(),
        ],
        p,
    )?;
    launch(
        arena,
        Op::History,
        [c / 128, 1, 1],
        vec![h[0].clone(), h[2].clone(), h[3].clone()],
        p,
    )?;
    launch(
        arena,
        Op::L2,
        [p[3] as usize * 2, t, 1],
        vec![h[4].clone(), h[9].clone()],
        p,
    )?;
    // Preserve existing accepted-prefix replay consumers. Native prefill only
    // keeps H; these additional split writes are an explicit compatibility gap.
    launch(
        arena,
        Op::Split,
        [(t * c).div_ceil(128), 1, 1],
        vec![h[4].clone(), h[5].clone(), h[6].clone(), h[7].clone()],
        p,
    )
}
/// H,controls,old,next,raw readout,Z,gamma,recurrence status,closing status.
pub(crate) fn recurrence(arena: &PrivateArena, p: [u32; 8], h: [Handle; 9]) -> Result<()> {
    let t = p[0] as usize;
    let c = p[2] as usize;
    let heads = p[4] as usize;
    let v = heads * 128;
    for (id, n) in [
        (0, t * c),
        (1, t * heads * 2),
        (2, heads * 16384),
        (3, heads * 16384),
        (4, t * v),
        (5, t * v),
        (6, if p[5] != 0 { 128 } else { v }),
        (7, t * v),
        (8, t * v),
    ] {
        span(&h[id], n * 4)?;
    }
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, t * v * 2)?;
    launch(
        arena,
        Op::Recurrence,
        [heads * 4, 1, 1],
        vec![
            h[0].clone(),
            h[1].clone(),
            h[2].clone(),
            h[3].clone(),
            h[4].clone(),
            h[7].clone(),
        ],
        p,
    )?;
    launch(
        arena,
        Op::Norm,
        [heads, t, 1],
        vec![
            h[4].clone(),
            h[5].clone(),
            h[6].clone(),
            packed,
            h[7].clone(),
            h[8].clone(),
        ],
        p,
    )
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Op::Conv,
        Op::History,
        Op::L2,
        Op::Split,
        Op::Recurrence,
        Op::Norm,
    ]
    .into_iter()
    .map(|op| {
        op.compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
    })
    .collect()
}
