//! MIT Strata fb58e0 Intel prefill QSA producers and FP16 consumer.
//! Separate RMS and RoPE match prefill/kernels.dp.cpp, rather than reusing
//! decode's fused norm. The existing default Intel attention remains selected;
//! the opt-in XMX kernel and native chunk/queue owners are separate audit rows.
use super::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Op {
    Split,
    Norm,
    Rope,
    Gate,
}
impl Op {
    fn lanes(self) -> u32 {
        if self == Self::Norm { 256 } else { 64 }
    }
    fn source(self) -> String {
        let all = include_str!("strata_prompt_qsa.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        format!(
            "{}\n{}",
            section("COMMON"),
            section(match self {
                Self::Split => "SPLIT",
                Self::Norm => "NORM",
                Self::Rope => "ROPE",
                Self::Gate => "GATE",
            })
        )
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Split => "StrataPromptQsaSplit",
            Self::Norm => "StrataPromptQsaRmsRows",
            Self::Rope => "StrataPromptQsaRope",
            Self::Gate => "StrataPromptQsaF16Gate",
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
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
fn span(handle: &Handle, words: usize) -> Result<()> {
    if handle.size_in_used() < words.checked_mul(4).ok_or(Error::ArithmeticOverflow)? as u64 {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
fn launch(
    arena: &PrivateArena,
    op: Op,
    grid: [usize; 3],
    handles: Vec<Handle>,
    p: [u32; 12],
) -> Result<()> {
    if !crate::strata_prompt::active()
        || !(1..=4096).contains(&p[0])
        || p[1].checked_add(p[0]).is_none_or(|end| end > p[2])
        || p[2] > 4096
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
    // SAFETY: caller checks the real parent owner extents and runtime geometry.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn split(arena: &PrivateArena, p: [u32; 12], h: [Handle; 3]) -> Result<()> {
    for handle in &h {
        span(handle, p[0] as usize * 12288)?;
    }
    launch(
        arena,
        Op::Split,
        [12288 / 64, p[0] as usize, 1],
        h.into(),
        p,
    )
}
/// Input,gamma,positions,output,status. Prior block rows are retained, not read
/// from the current chunk's uninitialized mean owner. Norm and RoPE share the
/// exact closing status so a failed norm cannot be rehabilitated by rotation.
pub(crate) fn norm_rope(arena: &PrivateArena, p: [u32; 12], h: [Handle; 5]) -> Result<()> {
    let count = p[0] as usize;
    let heads = p[6] as usize;
    let width = p[7] as usize;
    if !matches!(width, 128 | 256)
        || heads == 0
        || heads > 1024
        || p[8] > 1
        || (p[8] != 0 && heads != p[2] as usize / 4)
        || (p[3] as usize) < heads * width
        || (p[4] as usize) < heads * width
        || (p[5] as usize) < heads
        || !f32::from_bits(p[9]).is_finite()
        || f32::from_bits(p[9]) <= 0.
        || !f32::from_bits(p[10]).is_finite()
        || f32::from_bits(p[10]) <= 0.
        || p[11] != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    span(&h[0], count * p[3] as usize)?;
    span(&h[1], width)?;
    span(&h[2], p[2] as usize)?;
    span(&h[3], count * p[4] as usize)?;
    span(&h[4], count * p[5] as usize)?;
    launch(
        arena,
        Op::Norm,
        [heads, count, 1],
        vec![h[0].clone(), h[1].clone(), h[3].clone(), h[4].clone()],
        p,
    )?;
    launch(
        arena,
        Op::Rope,
        [heads, count, 1],
        vec![h[2].clone(), h[3].clone(), h[4].clone()],
        p,
    )
}
/// Original interleaved Qfull, attention, full closing status. Half output goes
/// straight to the authenticated activation stage consumed by attn_output.
pub(crate) fn gate(arena: &PrivateArena, p: [u32; 12], h: [Handle; 3]) -> Result<()> {
    span(&h[0], p[0] as usize * 12288)?;
    span(&h[1], p[0] as usize * 6144)?;
    if p[5] < 6144 {
        return Err(Error::InvalidBufferLength);
    }
    span(&h[2], p[0] as usize * p[5] as usize)?;
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, p[0] as usize * 3072)?;
    let packed_id = packed.clone().binding().memory.allocation_id();
    if h.iter().any(|v| {
        v.stream != packed.stream || v.clone().binding().memory.allocation_id() == packed_id
    }) {
        return Err(Error::InvalidBufferLength);
    }
    launch(
        arena,
        Op::Gate,
        [3072 / 64, p[0] as usize, 1],
        vec![h[0].clone(), h[1].clone(), packed, h[2].clone()],
        p,
    )
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Op::Split, Op::Norm, Op::Rope, Op::Gate]
        .into_iter()
        .map(|k| {
            k.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
        })
        .collect::<std::result::Result<_, _>>()
        .unwrap()
}
