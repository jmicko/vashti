//! MIT Strata fb58e0 Intel prefill QSA producers and FP16 consumer.
//! Separate RMS and RoPE match prefill/kernels.dp.cpp, rather than reusing
//! decode's fused norm. The existing default Intel attention remains selected;
//! the opt-in XMX kernel and native chunk/queue owners are separate audit rows.
use super::*;
#[path = "strata_prompt_qsa_full.rs"]
mod full;
pub(crate) use full::{compatibility, run};
// Original Prefill::Impl uses256 score/selection queries and32 attention queries.
// Full selections persist across batches; score/closing pages are reused.
pub(crate) const SELECT_BATCH: usize = 256;
pub(crate) const INFO_BYTES: u64 = 12
    * (2 * crate::strata_prompt::groups::CAPACITY.div_ceil(32) as u64
        + 3 * crate::strata_prompt::groups::CAPACITY.div_ceil(SELECT_BATCH) as u64
        + 18)
    * 2048;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Op {
    Split,
    Norm,
    Rope,
    Gate,
    PoolMean,
    PoolNorm,
    PoolRope,
    PoolFlags,
    SplitFault,
    GateFault,
    Positions,
    JoinFlags,
    Compatibility,
}
impl Op {
    fn lanes(self) -> u32 {
        if matches!(self, Self::Norm | Self::PoolNorm) {
            256
        } else if matches!(self, Self::PoolMean | Self::Gate | Self::GateFault) {
            128
        } else {
            64
        }
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
        let mut result = format!(
            "{}\n{}",
            section("COMMON"),
            section(match self {
                Self::Split | Self::SplitFault => "SPLIT",
                Self::Norm => "NORM",
                Self::Rope => "ROPE",
                Self::Gate | Self::GateFault => "GATE",
                Self::PoolMean => "POOL_MEAN",
                Self::PoolNorm => "NORM",
                Self::PoolRope => "ROPE",
                Self::PoolFlags => "POOL_FLAGS",
                Self::Positions => "POSITIONS",
                Self::JoinFlags => "JOIN_FLAGS",
                Self::Compatibility => "COMPATIBILITY",
            })
        );
        if matches!(self, Self::PoolNorm | Self::PoolRope) {
            result = result.replace("(p.start+t+1u)/4u", "(p.start+p.count)/4u");
        }
        if self == Self::PoolNorm {
            result = result.replace("var ok=1u;", "var ok=status[head];");
        }
        let edits: &[(&str, &str)] = match self {
            Self::SplitFault => &[
                (
                    "@binding(2) var<storage,read_write> status:array<u32>;",
                    "@binding(2) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[t*12288u+i]=select(0u,1u,good);",
                    "if !good{atomicOr(&fault[0],1u);}",
                ),
            ],
            Self::GateFault => &[
                (
                    "@binding(3) var<storage,read_write> status:array<u32>;",
                    "@binding(3) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[t*p.status_stride+i]=select(0u,1u,oka);status[t*p.status_stride+i+1u]=select(0u,1u,okb);",
                    "if !oka||!okb{atomicOr(&fault[0],1u);}",
                ),
            ],
            _ => &[],
        };
        for &(old, new) in edits {
            assert!(result.contains(old), "prompt QSA producer source drift");
            result = result.replace(old, new);
        }
        result
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Split => "StrataPromptQsaSplit",
            Self::Norm => "StrataPromptQsaRmsRows",
            Self::Rope => "StrataPromptQsaRope",
            Self::Gate => "StrataPromptQsaF16Gate",
            Self::PoolMean => "StrataPromptSharedIndexMean",
            Self::PoolNorm => "StrataPromptSharedIndexNorm",
            Self::PoolRope => "StrataPromptSharedIndexRope",
            Self::PoolFlags => "StrataPromptSharedIndexClosingFlags",
            Self::SplitFault => "StrataPromptFullQsaSplit",
            Self::GateFault => "StrataPromptFullQsaGate",
            Self::Positions => "StrataPromptQsaPositions",
            Self::JoinFlags => "StrataPromptCompactHeadFault",
            Self::Compatibility => "StrataPromptQsaLegacyClosingViews",
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
        return Err(crate::strata_prompt::invalid_buffer_length());
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
        || p[2] > crate::qsa_gpu::MAX_CONTEXT as u32
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
    span(&h[0], p[0] as usize * 12288)?;
    span(&h[1], p[0] as usize * 6144)?;
    span(&h[2], p[0] as usize * 12288)?;
    launch(
        arena,
        Op::Split,
        [12288 / 64, p[0] as usize, 1],
        h.into(),
        p,
    )
}
/// Completed blocks are shared by every query which can see them. Mean,
/// normalization and RoPE run once per newly completed block, not once per row.
/// Legacy closing views remain explicit until the new prompt receipt replaces
/// them. Pooled state is private and cannot publish before the model fence.
pub(crate) fn pool(arena: &PrivateArena, mut p: [u32; 12], h: [Handle; 2]) -> Result<()> {
    let q = crate::strata_prompt::qsa_workspace()?;
    span(&h[0], INDEX_TAIL_WORDS)?;
    span(&h[1], p[0] as usize * 128)?;
    span(&q[12], (p[2] as usize / 4) * 128)?;
    span(&q[13], p[2] as usize / 4)?;
    p[3] = (p[2] / 4) * 128;
    launch(
        arena,
        Op::PoolMean,
        [p[2] as usize / 4, 1, 1],
        vec![h[0].clone(), h[1].clone(), q[12].clone(), q[13].clone()],
        p,
    )
}
/// Gamma,positions,authoritative pooled,legacy mean flags,legacy norm flags.
pub(crate) fn pool_norm(arena: &PrivateArena, mut p: [u32; 12], h: [Handle; 5]) -> Result<()> {
    let q = crate::strata_prompt::qsa_workspace()?;
    let blocks = p[2] as usize / 4;
    span(&h[0], 128)?;
    span(&h[1], p[2] as usize)?;
    span(&h[2], blocks * 128)?;
    span(&h[3], p[0] as usize * blocks)?;
    span(&h[4], p[0] as usize * blocks)?;
    p[3] = (blocks * 128) as u32;
    p[4] = p[3];
    p[5] = blocks as u32;
    p[6] = blocks as u32;
    p[7] = 128;
    p[8] = 1;
    launch(
        arena,
        Op::PoolNorm,
        [blocks, 1, 1],
        vec![q[12].clone(), h[0].clone(), h[2].clone(), q[13].clone()],
        p,
    )?;
    launch(
        arena,
        Op::PoolRope,
        [blocks, 1, 1],
        vec![h[1].clone(), h[2].clone(), q[13].clone()],
        p,
    )?;
    launch(
        arena,
        Op::PoolFlags,
        [blocks.div_ceil(64), p[0] as usize, 1],
        vec![q[13].clone(), h[3].clone(), h[4].clone()],
        p,
    )
}
/// Native default attention schedule: a whole chunk's Q/selection/output, with
/// only32 queries' partials/statistics live at once. Merge each tile before
/// overwriting its scratch. `start` and candidate row zero keep their original
/// meaning across tiles, so causal prefix and in-chunk KV reads remain exact.
/// The selection and flags are parent-owned until full prompt receipts land.
pub(crate) fn attention(
    arena: &PrivateArena,
    mut p: [u32; 12],
    h: [Handle; 5], // cache, selected, attention flags, merge flags, candidate
    merge_stride: usize,
) -> Result<()> {
    let count = p[0] as usize;
    let q = crate::strata_prompt::qsa_workspace()?;
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
        || p[11] != 0
        || (p[5] as usize) < 24 * cells_for_capacity(p[2] as usize)
        || merge_stride < 24
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    span(&q[5], count * 6144)?;
    span(&q[9], count * 6144)?;
    span(&q[10], 32 * PARTIAL_WORDS)?;
    span(&q[11], 32 * STATS_WORDS)?;
    span(&h[0], INDEX_TAIL_WORDS)?;
    span(&h[1], count * 2051)?;
    span(&h[2], count * p[5] as usize)?;
    span(&h[3], count * merge_stride)?;
    span(&h[4], count * CACHE_WORDS)?;
    let attention_stride = p[5];
    for first in (0..count).step_by(32) {
        let rows = (count - first).min(32);
        p[0] = rows as u32;
        p[11] = first as u32;
        p[5] = attention_stride;
        super::launch(
            arena,
            Kernel::PromptAttention,
            [CHUNKS, 2, rows],
            vec![
                q[5].clone(),
                h[0].clone(),
                h[4].clone(),
                h[1].clone(),
                q[10].clone(),
                q[11].clone(),
                h[2].clone(),
            ],
            p,
        )?;
        p[5] = merge_stride as u32;
        super::launch(
            arena,
            Kernel::PromptMerge,
            [24, rows, 1],
            vec![q[10].clone(), q[11].clone(), q[9].clone(), h[3].clone()],
            p,
        )?;
    }
    Ok(())
}
fn cells_for_capacity(capacity: usize) -> usize {
    capacity.min(2051)
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
        return Err(crate::strata_prompt::invalid_buffer_length());
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
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    span(&h[2], p[0] as usize * p[5] as usize)?;
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, p[0] as usize * 3072)?;
    let packed_id = packed.clone().binding().memory.allocation_id();
    if h.iter().any(|v| {
        v.stream != packed.stream || v.clone().binding().memory.allocation_id() == packed_id
    }) {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    launch(
        arena,
        Op::Gate,
        [3072 / 128, p[0] as usize, 1],
        vec![h[0].clone(), h[1].clone(), packed, h[2].clone()],
        p,
    )
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Op::Split,
        Op::Norm,
        Op::Rope,
        Op::Gate,
        Op::PoolMean,
        Op::PoolNorm,
        Op::PoolRope,
        Op::PoolFlags,
        Op::SplitFault,
        Op::GateFault,
        Op::Positions,
        Op::JoinFlags,
        Op::Compatibility,
    ]
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
