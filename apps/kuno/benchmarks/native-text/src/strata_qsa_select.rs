//! MIT Strata fb58e0 qsa_select.dp.cpp::block_topk_kernel and the indexer's
//! initial dead key. Four weighted radix passes, stable cell-order ties and
//! chronological output stay on the actual GPU. No score readback/upload.
use super::*;

#[cfg(feature = "strata-intel-core")]
#[path = "strata_qsa_select_numeric_probe.rs"]
mod numeric;
#[cfg(feature = "strata-intel-core")]
pub(crate) use numeric::cases as public_numeric;

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
/// Raw IK (rounded here through FP16), gamma, pooled/dead, sticky fault.
/// Dead occupies the extra128 words after all completed-block slots and is
/// immutable once cell zero is accepted, including speculative suffix repair.
pub(crate) fn initialize(arena: &PrivateArena, owners: [Handle; 4], capacity: usize) -> Result<()> {
    if !(4..=crate::qsa_gpu::MAX_CONTEXT).contains(&capacity) || !capacity.is_multiple_of(4) {
        return Err(Error::InvalidBufferLength);
    }
    let dead = (capacity / 4) * 128;
    for (h, n) in owners.iter().zip([128, 128, dead + 128, 1]) {
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
    first: usize,
) -> Result<()> {
    select_checked(arena, owners, start, count, blocks, stride, first, blocks)
}

/// Match selected SYCL prefill's last-query n_bid+1 grid reach. Keep every
/// backing row at its original full capacity; this bounds work, not ownership.
/// The extra exact-four tail has weight zero and is never read by selection.
pub(super) fn prompt_active_blocks(start: usize, count: usize, blocks: usize) -> Result<usize> {
    if count == 0 || !(1..=crate::qsa_gpu::MAX_CONTEXT / 4).contains(&blocks) {
        return Err(Error::InvalidBufferLength);
    }
    let end = start.checked_add(count).ok_or(Error::ArithmeticOverflow)?;
    if end > blocks * 4 {
        return Err(Error::InvalidBufferLength);
    }
    Ok((end / 4 + 1).min(blocks))
}

/// Whole-prompt scores can omit all CTAs beyond the batch's last visible block.
/// Ordinary decode/short-read select() retains the full bound because its
/// native graph Position is rebound on replay independently of this parameter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_active_prefix(
    arena: &PrivateArena,
    owners: [Handle; 4],
    start: usize,
    count: usize,
    blocks: usize,
    stride: usize,
    first: usize,
    active_blocks: usize,
) -> Result<()> {
    if !crate::strata_prompt::active()
        || active_blocks != prompt_active_blocks(start, count, blocks)?
    {
        return Err(Error::InvalidBufferLength);
    }
    select_checked(
        arena,
        owners,
        start,
        count,
        blocks,
        stride,
        first,
        active_blocks,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_checked(
    arena: &PrivateArena,
    owners: [Handle; 4],
    start: usize,
    count: usize,
    blocks: usize,
    stride: usize,
    first: usize,
    active_blocks: usize,
) -> Result<()> {
    let maximum = if crate::strata_prompt::active() {
        crate::strata_prompt::groups::CAPACITY
    } else {
        8
    };
    if !crate::strata_intel_core::enabled()
        || !(1..=maximum).contains(&count)
        || !(1..=crate::qsa_gpu::MAX_CONTEXT / 4).contains(&blocks)
        || !(1..=blocks).contains(&active_blocks)
        || stride != 2051
        || first.checked_add(count).is_none_or(|end| end > maximum)
        || first > start
        || (!crate::strata_prompt::active() && first != 0)
        || start
            .checked_add(count)
            .is_none_or(|end| end > crate::qsa_gpu::MAX_CONTEXT || end > blocks * 4)
    {
        return Err(Error::InvalidBufferLength);
    }
    for (h, n) in owners
        .iter()
        .zip([count * blocks, count * blocks, (first + count) * stride, 1])
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
        active_blocks as u32,
        first as u32,
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

#[cfg(test)]
mod active_prefix_tests {
    use super::*;

    #[test]
    fn canonical_batch_reach_covers_every_causal_cell_at_full_owner_stride() {
        for capacity in [4usize, 512, 4096, crate::qsa_gpu::MAX_CONTEXT] {
            let blocks = capacity / 4;
            for count in [1usize, 3, 4, 7, 8, 9, 255, 256]
                .into_iter()
                .filter(|&n| n <= capacity)
            {
                for start in [0, (capacity - count) / 2, capacity - count] {
                    let active = prompt_active_blocks(start, count, blocks).unwrap();
                    // Upstream includes the exact-four zero-weight tail, then
                    // clamps to the original context owner's block capacity.
                    assert_eq!(active, ((start + count) / 4 + 1).min(blocks));
                    for t in 0..count {
                        let n = start + t + 1;
                        let visible = n.div_ceil(4);
                        assert!(visible <= active && active <= blocks);
                        for b in 0..=n / 4 {
                            let weight = if b < n / 4 { 4 } else { n % 4 };
                            if weight != 0 {
                                assert!(b < active);
                                assert!(t * blocks + b < count * blocks);
                            }
                        }
                    }
                }
            }
        }
        assert!(prompt_active_blocks(0, 0, 8192).is_err());
        assert!(prompt_active_blocks(0, 1, 0).is_err());
        assert!(prompt_active_blocks(0, 1, 8193).is_err());
        assert!(prompt_active_blocks(32767, 2, 8192).is_err());
        assert!(prompt_active_blocks(usize::MAX, 1, 8192).is_err());
    }

    #[test]
    fn first_prompt_chunk_omits_empty_ctas_without_repricing_owning_strides() {
        let full = 8192usize.div_ceil(8) * 16;
        let bounded: usize = (0..4096)
            .step_by(256)
            .map(|start| prompt_active_blocks(start, 256, 8192).unwrap().div_ceil(8))
            .sum();
        assert_eq!(bounded, 1104);
        assert_eq!(full, 16384);
        assert!(bounded * 100 < full * 7);
        assert_eq!(prompt_active_blocks(32512, 256, 8192).unwrap(), 8192);
    }

    #[test]
    fn actual_selection_sources_ignore_uncomputed_padding_and_initialize_output_tails() {
        let validate = Op::Validate.source();
        let select = Op::Select.source();
        for source in [&validate, &select] {
            assert!(source.contains("visible_blocks>p.dead_offset"));
            assert!(source.contains("p.dead_offset>p.blocks"));
        }
        // Validation visits only actual causal blocks. Radix keys never read
        // the upstream extra zero-weight tail or omitted/future score slots.
        assert!(validate.contains("b<visible_blocks"));
        assert!(select.contains("b<b1&&weight(b,nbid,n)>0u"));
        assert!(select.contains("if w==0u{continue;}"));
        assert!(select.contains("scores[t*p.score_stride+b]"));
        assert!(validate.contains("status[t*p.status_stride+b]"));
        let initialize = select.find("j=width+tid;j<p.stride;j+=1024u").unwrap();
        let sentinel = select.find("=p.blocks*4u;").unwrap();
        let first_write = select.find("if n<=width").unwrap();
        assert!(initialize < sentinel && sentinel < first_write);
        assert!(select.contains("selected[(t+p.reserved)*p.stride+j]"));
    }
}
