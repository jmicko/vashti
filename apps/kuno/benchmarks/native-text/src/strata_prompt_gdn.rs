//! MIT Strata fb58e0 Intel prefill/kernels.dp.cpp's connected prompt GDN.
//! Contiguous token-major H, causal conv/L2, CB32 pipelined recurrence and
//! native FP16 norm output are selected together. Prompt, decode and repair
//! share native row/value-head state. Split verifier replay remains separate.
use super::*;
pub(crate) const COMPATIBILITY_INFO_PER_STAGE: u64 = 2 * 2048;
pub(crate) const INFO_BYTES: u64 = 36 * COMPATIBILITY_INFO_PER_STAGE;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Conv,
    History,
    L2,
    Split,
    Recurrence,
    Norm,
    Retain,
    ConvFault,
    L2Fault,
    RecurrenceFault,
    NormFault,
    Compatibility,
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
        let mut source = format!(
            "{}\n{}",
            part("COMMON"),
            part(match self {
                Self::Conv | Self::ConvFault => "CONV",
                Self::History => "HISTORY",
                Self::L2 | Self::L2Fault => "L2",
                Self::Split => "SPLIT",
                Self::Recurrence | Self::RecurrenceFault => "RECURRENCE",
                Self::Norm | Self::NormFault => "NORM",
                Self::Retain => "RETAIN",
                Self::Compatibility => "COMPATIBILITY",
            })
        );
        // Transform only the closing stores/bindings. The native prompt
        // arithmetic and state schedule remain the same connected source.
        let edits: &[(&str, &str)] = match self {
            Self::ConvFault => &[
                (
                    "@binding(4) var<storage,read_write> status:array<u32>;",
                    "@binding(4) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "@group(0) @binding(5) var<storage,read_write> prepared:array<u32>;\n",
                    "",
                ),
                ("@binding(6) var<uniform>", "@binding(5) var<uniform>"),
                (
                    "status[t*p.channels+c]=select(0u,1u,good);prepared[t*p.channels+c]=select(0u,1u,good);",
                    "if !good{atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::L2Fault => &[
                (
                    "@binding(1) var<storage,read_write> prepared:array<u32>;",
                    "@binding(1) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "prepared[at]=0u;return;",
                    "if col==0u{atomicOr(&fault[0],1u);}return;",
                ),
                ("finite(y)&&prepared[at]==1u", "finite(x)&&finite(y)"),
                (
                    "prepared[at]=select(0u,1u,good);",
                    "if !good{atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::RecurrenceFault => &[
                (
                    "@binding(5) var<storage,read_write> status:array<u32>;",
                    "@binding(5) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[at]=select(0u,1u,good);",
                    "if !good{atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::NormFault => &[
                (
                    "@binding(4) var<storage,read> status:array<u32>;",
                    "@binding(4) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "@group(0) @binding(5) var<storage,read_write> closing:array<u32>;\n",
                    "",
                ),
                ("@binding(6) var<uniform>", "@binding(5) var<uniform>"),
                (
                    "closing[at]=0u;return;",
                    "if col==0u{atomicOr(&fault[0],1u);}return;",
                ),
                (
                    "closing[at]=select(0u,1u,finite(y)&&finite(ga)&&finite(zz)&&status[at]==1u);",
                    "if !finite(y)||!finite(ga)||!finite(zz){atomicOr(&fault[0],1u);}",
                ),
                (
                    "closing[at-col+col*2u]=0u;closing[at-col+col*2u+1u]=0u;",
                    "atomicOr(&fault[0],1u);",
                ),
            ],
            _ => &[],
        };
        for &(old, new) in edits {
            assert!(source.contains(old), "prompt GDN closing source drift");
            source = source.replace(old, new);
        }
        source
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
            Self::Retain => "StrataPromptGdnRetainRawPrefix",
            Self::ConvFault => "StrataPromptFullConv",
            Self::L2Fault => "StrataPromptFullL2",
            Self::RecurrenceFault => "StrataPromptFullRecurrence",
            Self::NormFault => "StrataPromptFullNorm",
            Self::Compatibility => "StrataPromptLegacyClosingViews",
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
        || (op != Op::Retain
            && (p[3] == 0
                || p[3] > 16
                || p[4] == 0
                || p[4] > 48
                || !p[4].is_multiple_of(p[3])
                || p[2] != 128 * (2 * p[3] + p[4])
                || p[5] > 1))
        || if op == Op::Compatibility {
            p[7] == 0 || p[7] > 10240
        } else {
            p[7] != 0
        }
        || !f32::from_bits(p[6]).is_finite()
        || f32::from_bits(p[6]) <= 0.
        || grid.contains(&0)
        || grid.iter().any(|&v| v > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
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
/// Retain the actual carved raw projection for the existing MAX8 replay owner.
/// Inactive rows are zeroed; full prompt state acceptance will not need this
/// compatibility copy. Both resources remain in the sole owning native graph.
pub(crate) fn retain(
    arena: &PrivateArena,
    count: usize,
    channels: usize,
    source: Handle,
    target: Handle,
) -> Result<()> {
    if !(1..=8).contains(&count)
        || channels == 0
        || channels > 10240
        || target.size_in_used() != (8 * channels * 4) as u64
        || crate::strata_prompt::storage::overlaps(&source, &target)?
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    crate::strata_prompt::validate_output(&source)?;
    span(&source, count * channels * 4)?;
    launch(
        arena,
        Op::Retain,
        [(8 * channels).div_ceil(128), 1, 1],
        vec![source, target],
        [
            count as u32,
            0,
            channels as u32,
            0,
            0,
            0,
            1e-6f32.to_bits(),
            0,
        ],
    )
}
/// Raw,w,old,next,H,Q,K,V,conv status,prepared status: original real owners.
pub(crate) fn convolution(
    arena: &PrivateArena,
    p: [u32; 8],
    h: [Handle; 10],
    fault: Handle,
) -> Result<()> {
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
    convolution_full(
        arena,
        p,
        [
            h[0].clone(),
            h[1].clone(),
            h[2].clone(),
            h[3].clone(),
            h[4].clone(),
            fault.clone(),
        ],
    )?;
    compatibility(arena, p, c, [fault, h[8].clone(), h[9].clone()])?;
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
pub(crate) fn recurrence(
    arena: &PrivateArena,
    p: [u32; 8],
    h: [Handle; 9],
    fault: Handle,
) -> Result<()> {
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
    recurrence_full(
        arena,
        p,
        [
            h[0].clone(),
            h[1].clone(),
            h[2].clone(),
            h[3].clone(),
            h[4].clone(),
            h[5].clone(),
            h[6].clone(),
            fault.clone(),
        ],
    )?;
    compatibility(arena, p, v, [fault, h[7].clone(), h[8].clone()])
}
fn compatibility(
    arena: &PrivateArena,
    mut p: [u32; 8],
    width: usize,
    h: [Handle; 3],
) -> Result<()> {
    if !(1..=8).contains(&p[0]) {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    span(&h[0], 4)?;
    span(&h[1], p[0] as usize * width * 4)?;
    span(&h[2], p[0] as usize * width * 4)?;
    p[7] = width as u32;
    launch(
        arena,
        Op::Compatibility,
        [(p[0] as usize * width).div_ceil(128), 1, 1],
        h.into(),
        p,
    )
}
/// Whole-chunk entry: raw, convolution weights, old/next history, H, fault.
/// Produces no split replay rows or per-element status maps. Actual state
/// remains pending until the outer model's completed fault fence accepts it.
pub(crate) fn convolution_full(arena: &PrivateArena, p: [u32; 8], h: [Handle; 6]) -> Result<()> {
    let t = p[0] as usize;
    let c = p[2] as usize;
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&t)
        || crate::strata_prompt::storage::overlaps(&h[2], &h[3])?
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    for (handle, n) in h.iter().zip([t * c, c * 4, c * 3, c * 3, t * c, 1]) {
        span(handle, n * 4)?;
    }
    crate::strata_prompt::validate_output(&h[0])?;
    crate::strata_prompt::validate_output(&h[4])?;
    launch(
        arena,
        Op::ConvFault,
        [c / 128, t.div_ceil(64), 1],
        vec![
            h[0].clone(),
            h[1].clone(),
            h[2].clone(),
            h[4].clone(),
            h[5].clone(),
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
        Op::L2Fault,
        [p[3] as usize * 2, t, 1],
        vec![h[4].clone(), h[5].clone()],
        p,
    )
}
/// H, controls, old/next recurrence, readout, Z, gamma, sticky model fault.
/// Count is prompt capacity, independent of MAX8 verifier/replay geometry.
pub(crate) fn recurrence_full(arena: &PrivateArena, p: [u32; 8], h: [Handle; 8]) -> Result<()> {
    let t = p[0] as usize;
    let c = p[2] as usize;
    let heads = p[4] as usize;
    let v = heads * 128;
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&t)
        || crate::strata_prompt::storage::overlaps(&h[2], &h[3])?
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    for (handle, n) in h.iter().zip([
        t * c,
        t * heads * 2,
        heads * 16384,
        heads * 16384,
        t * v,
        t * v,
        if p[5] != 0 { 128 } else { v },
        1,
    ]) {
        span(handle, n * 4)?;
    }
    for i in [0, 4, 5] {
        crate::strata_prompt::validate_output(&h[i])?;
    }
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, t * v * 2)?;
    launch(
        arena,
        Op::RecurrenceFault,
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
        Op::NormFault,
        [heads, t, 1],
        vec![
            h[4].clone(),
            h[5].clone(),
            h[6].clone(),
            packed,
            h[7].clone(),
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
        Op::Retain,
        Op::ConvFault,
        Op::L2Fault,
        Op::RecurrenceFault,
        Op::NormFault,
        Op::Compatibility,
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
