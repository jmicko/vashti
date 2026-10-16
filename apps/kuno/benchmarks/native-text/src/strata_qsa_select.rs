//! MIT Strata fb58e0 qsa_select.dp.cpp::block_topk_kernel and the indexer's
//! initial dead key. Four weighted radix passes, stable cell-order ties and
//! chronological output stay on the actual GPU. No score readback/upload.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Dead,
    Validate,
    Select,
}
impl Op {
    fn lanes(self) -> u32 {
        if self == Self::Select { 1024 } else { 256 }
    }
    fn source(self) -> String {
        let all = include_str!("strata_qsa_select.wgsl");
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
                Self::Dead => "DEAD",
                Self::Validate => "VALIDATE",
                Self::Select => "SELECT",
            })
        )
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Dead => "StrataQsaInitialDeadKey",
            Self::Validate => "StrataQsaScoreClosingFault",
            Self::Select => "StrataQsaWeightedRadixCells",
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
fn span(h: &Handle, words: usize) -> Result<()> {
    if h.size_in_used() < words.checked_mul(4).ok_or(Error::ArithmeticOverflow)? as u64 {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
fn launch(
    arena: &PrivateArena,
    op: Op,
    grid: usize,
    owners: [Handle; 4],
    p: [u32; 8],
) -> Result<()> {
    if grid == 0 || grid > 65535 || owners.iter().any(|h| h.stream != owners[0].stream) {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    if client.properties().hardware.max_units_per_cube < op.lanes()
        || (op == Op::Select && client.properties().hardware.max_shared_memory_size < 32912)
    {
        return Err(Error::LimitExceeded("native QSA radix workgroup limits"));
    }
    let mut args = KernelArguments::new();
    for h in owners {
        args = args.with_buffer(h.binding());
    }
    let mut metadata = MetadataBindingInfo::custom(
        p.chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    );
    if op != Op::Dead {
        metadata = metadata.with_native_parameter(
            1,
            cubecl_runtime::server::NativeStrataValue::Position { offset: 0 },
        );
    }
    args = args.with_info(metadata);
    // SAFETY: same-client private parent owners and exact active spans have
    // been checked before this entry. Selection's barriers are query-uniform.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid as u32, 1, 1),
            args,
        );
    }
    Ok(())
}
/// Candidate (rounded raw IK), gamma, authoritative pooled/dead, sticky fault.
/// Dead occupies the extra128 words after all completed-block slots and is
/// immutable once cell zero is accepted, including speculative suffix repair.
pub(crate) fn initialize(arena: &PrivateArena, owners: [Handle; 4], capacity: usize) -> Result<()> {
    if !(4..=4096).contains(&capacity) || !capacity.is_multiple_of(4) {
        return Err(Error::InvalidBufferLength);
    }
    let dead = (capacity / 4) * 128;
    for (h, n) in owners.iter().zip([super::CACHE_WORDS, 128, dead + 128, 1]) {
        span(h, n)?;
    }
    launch(
        arena,
        Op::Dead,
        1,
        owners,
        [0, 0, 0, 0, 0, 0, dead as u32, 0],
    )
}
/// Scores, score flags, selected cells and actual model fault. Width is
/// min(visible,2051), including partial blocks; future rows never participate.
pub(crate) fn select(
    arena: &PrivateArena,
    owners: [Handle; 4],
    start: usize,
    count: usize,
    blocks: usize,
    stride: usize,
) -> Result<()> {
    let maximum = if crate::strata_prompt::active() {
        crate::strata_prompt::groups::CAPACITY
    } else {
        8
    };
    if !crate::strata_intel_core::enabled()
        || !(1..=maximum).contains(&count)
        || !(1..=1024).contains(&blocks)
        || stride != 2051
        || start
            .checked_add(count)
            .is_none_or(|end| end > 4096 || end > blocks * 4)
    {
        return Err(Error::InvalidBufferLength);
    }
    for (h, n) in owners
        .iter()
        .zip([count * blocks, count * blocks, count * stride, 1])
    {
        span(h, n)?;
    }
    let p = [
        count as u32,
        start as u32,
        blocks as u32,
        stride as u32,
        blocks as u32,
        blocks as u32,
        0,
        0,
    ];
    launch(arena, Op::Validate, count, owners.clone(), p)?;
    launch(arena, Op::Select, count, owners, p)
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Op::Dead, Op::Validate, Op::Select]
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
