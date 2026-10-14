//! MIT Strata Intel prompt GEMM activation/weight ownership and consumers.
//! Rust/Vulkan tile GEMM substitutes for oneMKL, with original BF16 or decoded
//! FP16 operands and F32 accumulation. Existing admitted stages are borrowed.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
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
use std::cell::RefCell;
#[path = "strata_prompt_experts.rs"]
pub(crate) mod experts;
#[path = "strata_prompt_groups.rs"]
pub(crate) mod groups;
#[path = "strata_prompt_storage.rs"]
pub(crate) mod storage;
thread_local! {static STAGES:RefCell<Option<storage::Storage>>=const{RefCell::new(None)};}
pub(crate) struct Scope;
impl Scope {
    pub(crate) fn new(arena: &PrivateArena, ids: [BufferId; 3]) -> Result<Self> {
        if !crate::strata_intel_core::enabled()
            || ids[0] == ids[1]
            || ids[0] == ids[2]
            || ids[1] == ids[2]
        {
            return Err(Error::InvalidBufferLength);
        }
        let handles = [
            u::handle(arena, ids[0])?,
            u::handle(arena, ids[1])?,
            u::handle(arena, ids[2])?,
        ];
        if handles.iter().any(|h| {
            h.size_in_used() != crate::strata_global_experts::STAGE_BYTES
                || h.stream != handles[0].stream
        }) {
            return Err(Error::InvalidBufferLength);
        }
        STAGES.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_some() {
                return Err(Error::InvalidBufferLength);
            }
            *s = Some(storage::Storage::new(handles)?);
            Ok(Self)
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        STAGES.with(|s| *s.borrow_mut() = None);
    }
}
pub(crate) fn active() -> bool {
    STAGES.with(|s| s.borrow().is_some())
}
fn stages() -> Result<[Handle; 3]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::working)
            .ok_or(Error::InvalidBufferLength)
    })
}
fn gr_workspace() -> Result<[Handle; 4]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::gr)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn outer_workspace() -> Result<[Handle; 13]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::outer)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn ple_workspace() -> Result<[Handle; 12]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::ple)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn gdn_workspace() -> Result<[Handle; 4]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::gdn)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn gdn_controls() -> Result<[Handle; 3]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::gdn_controls)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn qsa_workspace() -> Result<[Handle; 20]> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .map(storage::Storage::qsa)
            .ok_or(Error::InvalidBufferLength)
    })
}
pub(crate) fn validate_output(handle: &Handle) -> Result<()> {
    STAGES.with(|s| {
        if s.borrow().as_ref().is_some_and(|v| v.output(handle)) {
            Ok(())
        } else {
            Err(Error::InvalidBufferLength)
        }
    })
}
pub(crate) fn activation_stage() -> Result<Handle> {
    Ok(stages()?[0].clone())
}
/// Exact request-private device copy, including copies into another admitted
/// arena on the same native client. No host activation map is created.
pub(crate) fn copy_handles(
    client: &ComputeClient<WgpuRuntime>,
    input: Handle,
    output: Handle,
    first: usize,
    words: usize,
) -> Result<()> {
    if words == 0
        || words > groups::CAPACITY * 10240
        || first
            .checked_add(words)
            .is_none_or(|n| n as u64 * 4 > input.size_in_used())
        || words as u64 * 4 > output.size_in_used()
        || storage::overlaps(&input, &output)?
        || first > u32::MAX as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut p = [0; 16];
    p[0] = 1;
    p[1] = words as u32;
    p[4] = first as u32;
    launch(
        client,
        Kernel::Copy,
        [words.div_ceil(256), 1, 1],
        vec![input, output],
        p,
    )
}
pub(crate) fn head_flags(
    client: &ComputeClient<WgpuRuntime>,
    flags: Handle,
    fault: Handle,
    count: usize,
    heads: usize,
) -> Result<()> {
    if !(1..=groups::CAPACITY).contains(&count)
        || heads == 0
        || heads > 1024
        || flags.size_in_used() < (count * heads * 4) as u64
        || fault.size_in_used() != 4
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut p = [0; 16];
    p[0] = count as u32;
    p[1] = heads as u32;
    launch(
        client,
        Kernel::Flags,
        [(count * heads).div_ceil(256), 1, 1],
        vec![flags, fault],
        p,
    )
}
pub(crate) fn embedding(
    graph: kuno_text::model::BoundGraph<'_, '_>,
    client: &ComputeClient<WgpuRuntime>,
    tokens: &[u32],
    target: Handle,
    residual: Handle,
    fault: Handle,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    if !(1..=groups::CAPACITY).contains(&tokens.len())
        || tokens.iter().any(|&t| t >= 248320)
        || target.size_in_used() < (tokens.len() * 2560 * 4) as u64
        || residual.size_in_used() < (tokens.len() * 10240 * 4) as u64
        || fault.size_in_used() != 4
    {
        return Err(Error::InvalidBufferLength);
    }
    validate_output(&target)?;
    validate_output(&residual)?;
    graph.verify_unchanged()?;
    let tensor = graph.tensor("token_embd.weight")?;
    if tensor.info().row_len() != 2560 || tensor.info().row_count() != 248320 {
        return Err(Error::InvalidBufferLength);
    }
    let rb = tensor.info().row_bytes() as usize;
    if rb == 0 || rb > 10240 {
        return Err(Error::InvalidBufferLength);
    }
    let mut packed = zeroize::Zeroizing::new(vec![0u8; rb]);
    let mut decoded = zeroize::Zeroizing::new([0f32; 2560]);
    let mut upload = zeroize::Zeroizing::new(vec![0u8; tokens.len() * 10240]);
    for (t, &token) in tokens.iter().enumerate() {
        u::cancel(cancel)?;
        tensor.read_row(token as u64, &mut packed)?;
        tensor.info().dtype().decode(&packed, &mut decoded[..])?;
        for (i, &v) in decoded.iter().enumerate() {
            upload[(t * 2560 + i) * 4..(t * 2560 + i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    graph.verify_unchanged()?;
    let bytes = upload.len() as u64;
    let view = target.clone().offset_end(target.size_in_used() - bytes);
    client
        .write_private(view, std::mem::take(&mut *upload))
        .map_err(u::fault)?;
    let mut p = [0; 16];
    p[0] = tokens.len() as u32;
    launch(
        client,
        Kernel::Embedding,
        [(tokens.len() * 10240).div_ceil(256), 1, 1],
        vec![target, residual, fault],
        p,
    )
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Pack,
    Dequant(u32),
    Gemm,
    GrNorm,
    GrLow,
    GrMix,
    AbControls,
    GrWrite,
    Copy,
    Embedding,
    Flags,
    SharedPair,
    SharedMerge,
}
impl Kernel {
    fn lanes(self) -> u32 {
        if self == Self::Gemm { 64 } else { 256 }
    }
    fn source(self) -> String {
        let all = include_str!("strata_prompt.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut source = section("COMMON").to_owned();
        if let Self::Dequant(dtype) = self {
            let bytes = crate::quant_matvec::block_layout(dtype).unwrap().1;
            source.push_str(&format!("const DTYPE:u32={dtype}u;const GROUP_VALUES:u32={}u;const BLOCK_BYTES:u32={bytes}u;\n",if dtype==14 {16}else{32}));
            source.push_str(
                include_str!("strata_dense.wgsl")
                    .split_once("@compute")
                    .unwrap()
                    .0,
            );
        }
        source.push_str(section(match self {
            Self::Pack => "PACK",
            Self::Dequant(_) => "DEQUANT",
            Self::Gemm => "GEMM",
            Self::GrNorm => "GR_NORM",
            Self::GrLow => "GR_LOW",
            Self::GrMix => "GR_MIX",
            Self::AbControls => "AB_CONTROLS",
            Self::GrWrite => "GR_WRITE",
            Self::Copy => "COPY",
            Self::Embedding => "EMBEDDING",
            Self::Flags => "FLAGS",
            Self::SharedPair => "SHARED_PAIR",
            Self::SharedMerge => "SHARED_MERGE",
        }));
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Pack => "StrataPromptActivationImage",
            Self::Dequant(_) => "StrataPromptOriginalWeightImage",
            Self::Gemm => "StrataPromptTileGemm",
            Self::GrNorm => "StrataPromptGrNormalize",
            Self::GrLow => "StrataPromptGrLowActivation",
            Self::GrMix => "StrataPromptGrMix",
            Self::AbControls => "StrataPromptAbControls",
            Self::GrWrite => "StrataPromptResidualBoundaryWrite",
            Self::Copy => "StrataPromptDeviceCopy",
            Self::Embedding => "StrataPromptEmbeddingBroadcast",
            Self::Flags => "StrataPromptCompactClosingFault",
            Self::SharedPair => "StrataPromptSharedSwiGluFp16",
            Self::SharedMerge => "StrataPromptSharedRoutedCombine",
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
    client: &ComputeClient<WgpuRuntime>,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 16],
) -> Result<()> {
    if !active()
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
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    // SAFETY: connected original-source caller validates live disjoint spans,
    // authentic stage roots and serial reuse until its completed model fence.
    unsafe {
        client.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn supports(dtype: u32, columns: usize) -> bool {
    matches!(dtype, 8 | 12 | 13 | 14 | 20 | 23 | 30)
        && columns > 0
        && columns <= 16384
        && columns.is_multiple_of(32)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn project(
    client: &ComputeClient<WgpuRuntime>,
    dtype: u32,
    columns: usize,
    rows: usize,
    row_bytes: usize,
    weight: Handle,
    input: Handle,
    output: Handle,
    fault: Handle,
    count: usize,
    first: usize,
    stride: usize,
    out_first: usize,
    out_stride: usize,
    saturate: bool,
    prepacked: bool,
) -> Result<()> {
    let stage = stages()?;
    if !supports(dtype, columns)
        || stage[..2]
            .iter()
            .any(|h| storage::overlaps(h, &output).unwrap_or(true))
        || rows == 0
        || rows > 16384
        || count == 0
        || count > 4096
        || stride < columns
        || out_stride < rows
        || fault.size_in_used() != 4
        || prepacked
            && (dtype == 30
                || first != 0
                || stride != columns
                || input.stream != stage[0].stream
                || input.clone().binding().memory.allocation_id()
                    != stage[0].clone().binding().memory.allocation_id()
                || input.size_in_used() != stage[0].size_in_used())
        || prepacked
            && (input.offset_start.unwrap_or(0) != stage[0].offset_start.unwrap_or(0)
                || input.offset_end.unwrap_or(0) != stage[0].offset_end.unwrap_or(0))
        || count
            .checked_mul(columns * 2)
            .is_none_or(|n| n as u64 > stage[0].size_in_used())
        || (dtype != 30 && (rows * columns * 2) as u64 > stage[1].size_in_used())
        || weight.size_in_used() < (rows * row_bytes) as u64
        || !prepacked
            && first
                .checked_add((count - 1) * stride)
                .and_then(|n| n.checked_add(columns))
                .is_none_or(|n| n as u64 * 4 > input.size_in_used())
        || out_first
            .checked_add((count - 1) * out_stride)
            .and_then(|n| n.checked_add(rows))
            .is_none_or(|n| n as u64 * 4 > output.size_in_used())
    {
        return Err(Error::InvalidBufferLength);
    }
    let info = [
        count as u32,
        columns as u32,
        rows as u32,
        row_bytes as u32,
        first as u32,
        stride as u32,
        out_first as u32,
        out_stride as u32,
        (dtype == 30) as u32,
        0,
        0,
        0,
        saturate as u32,
        0,
        0,
        0,
    ];
    if !prepacked {
        launch(
            client,
            Kernel::Pack,
            [(count * columns / 2).div_ceil(256), 1, 1],
            vec![input, stage[0].clone(), fault.clone()],
            info,
        )?;
    }
    let weights = if dtype == 30 {
        weight
    } else {
        launch(
            client,
            Kernel::Dequant(dtype),
            [(rows * columns / 2).div_ceil(256), 1, 1],
            vec![weight, stage[1].clone(), fault.clone()],
            info,
        )?;
        stage[1].clone()
    };
    launch(
        client,
        Kernel::Gemm,
        [rows.div_ceil(8), count.div_ceil(8), 1],
        vec![stage[0].clone(), weights, output, fault],
        info,
    )
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Pack,
        Kernel::Dequant(8),
        Kernel::Dequant(12),
        Kernel::Dequant(13),
        Kernel::Dequant(14),
        Kernel::Dequant(20),
        Kernel::Dequant(23),
        Kernel::Gemm,
        Kernel::GrNorm,
        Kernel::GrLow,
        Kernel::GrMix,
        Kernel::AbControls,
        Kernel::GrWrite,
        Kernel::Copy,
        Kernel::Embedding,
        Kernel::Flags,
        Kernel::SharedPair,
        Kernel::SharedMerge,
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

pub(crate) fn gr(request: crate::strata_gr::Read<'_>, weights: [Handle; 3]) -> Result<()> {
    let q = request;
    let a = q.arena;
    let b = q.buffers;
    let h = |id| u::handle(a, id);
    let ids = [
        q.residual,
        b.normalized,
        b.down,
        b.low,
        b.partial,
        b.injection,
        b.mixed,
        q.fault,
    ];
    if !(1..=8).contains(&q.count)
        || q.first
            .checked_add(q.count * 10240)
            .is_none_or(|n| n > u::words(a, q.residual).unwrap_or(0))
        || u::words(a, q.fault)? != 1
        || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id))
        || q.gamma.size_in_used() != 40960
        || weights[0].size_in_used() != 6553600
        || weights[1].size_in_used() != 6553600
        || q.inject && weights[2].size_in_used() != 81920
    {
        return Err(Error::InvalidBufferLength);
    }
    for (id, words) in [
        (b.normalized, 81920),
        (b.down, 2560),
        (b.low, 2560),
        (b.partial, 207360),
        (b.injection, 32),
        (b.mixed, 20480),
    ] {
        if u::words(a, id)? != words {
            return Err(Error::InvalidBufferLength);
        }
    }
    let previous = q
        .previous
        .map(|(source, block)| -> Result<(Handle, Handle)> { Ok((h(source)?, h(block)?)) })
        .transpose()?;
    gr_handles(
        GrHandles {
            client: a.client().map_err(u::fault)?,
            residual: h(q.residual)?,
            previous,
            gamma: q.gamma,
            injection: h(b.injection)?,
            mixed: h(b.mixed)?,
            fault: h(q.fault)?,
            count: q.count,
            first: q.first,
            epsilon: q.epsilon,
            inject: q.inject,
        },
        weights,
    )
}

pub(crate) struct GrHandles<'a> {
    pub client: &'a ComputeClient<WgpuRuntime>,
    pub residual: Handle,
    pub previous: Option<(Handle, Handle)>,
    pub gamma: Handle,
    pub injection: Handle,
    pub mixed: Handle,
    pub fault: Handle,
    pub count: usize,
    pub first: usize,
    pub epsilon: f32,
    pub inject: bool,
}
/// Prompt injection is already activated by GR_LOW. Boundary materialization
/// must apply it once, unlike decode's fused writer which receives raw logits.
pub(crate) fn write_handles(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 5],
    count: usize,
) -> Result<()> {
    if !(1..=groups::CAPACITY).contains(&count) {
        return Err(Error::InvalidBufferLength);
    }
    for (v, n) in h
        .iter()
        .zip([count * 10240, count * 2560, count * 4, count * 10240, 1])
    {
        if v.size_in_used() < (n * 4) as u64 {
            return Err(Error::InvalidBufferLength);
        }
    }
    if h[..3]
        .iter()
        .chain(h[4..].iter())
        .any(|v| storage::overlaps(v, &h[3]).unwrap_or(true))
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut p = [0; 16];
    p[0] = count as u32;
    launch(
        client,
        Kernel::GrWrite,
        [(count * 10240).div_ceil(256), 1, 1],
        h.into(),
        p,
    )
}
/// Full prompt GR entry point: owns no allocation and does not use MAX8 buffer
/// geometry. Its four mutable intermediates are authenticated carved views.
pub(crate) fn gr_handles(q: GrHandles<'_>, weights: [Handle; 3]) -> Result<()> {
    let [normalized, down, low, up] = gr_workspace()?;
    let extent = |h: &Handle, n: usize| {
        n.checked_mul(4)
            .is_some_and(|bytes| h.size_in_used() >= bytes as u64)
    };
    if !(1..=groups::CAPACITY).contains(&q.count)
        || q.first
            .checked_add(q.count * 10240)
            .is_none_or(|n| n > groups::CAPACITY * 10240 || !extent(&q.residual, n))
        || !extent(&q.mixed, q.count * 2560)
        || !extent(&q.injection, q.count * 4)
        || q.fault.size_in_used() != 4
        || q.gamma.size_in_used() != 40960
        || weights[0].size_in_used() != 6553600
        || weights[1].size_in_used() != 6553600
        || q.inject && weights[2].size_in_used() != 81920
        || !q.epsilon.is_finite()
        || q.epsilon <= 0.
    {
        return Err(Error::InvalidBufferLength);
    }
    let external = [&q.residual, &q.injection, &q.mixed, &q.fault, &q.gamma];
    for h in [&normalized, &down, &low, &up] {
        if external
            .iter()
            .any(|v| storage::overlaps(h, v).unwrap_or(true))
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    if let Some((source, block)) = &q.previous
        && (!extent(source, q.first + q.count * 10240)
            || !extent(block, q.count * 2560)
            || [&normalized, &down, &low, &up].iter().any(|h| {
                storage::overlaps(h, source).unwrap_or(true)
                    || storage::overlaps(h, block).unwrap_or(true)
            }))
    {
        return Err(Error::InvalidBufferLength);
    }
    let info = [
        q.count as u32,
        0,
        0,
        0,
        q.first as u32,
        0,
        0,
        0,
        0,
        q.previous.is_some() as u32,
        q.inject as u32,
        q.epsilon.to_bits(),
        0,
        0,
        0,
        0,
    ];
    // The no-apply branch reads neither dummy. Both still cover full possible
    // source intervals and remain disjoint from the writable norm/residual.
    let (previous, block) = q.previous.unwrap_or((up.clone(), q.mixed.clone()));
    launch(
        q.client,
        Kernel::GrNorm,
        [4, q.count, 1],
        vec![
            previous,
            block,
            q.injection.clone(),
            q.gamma,
            q.residual.clone(),
            normalized.clone(),
            q.fault.clone(),
        ],
        info,
    )?;
    let client = q.client;
    project(
        client,
        30,
        10240,
        320,
        20480,
        weights[0].clone(),
        normalized.clone(),
        down.clone(),
        q.fault.clone(),
        q.count,
        0,
        10240,
        0,
        320,
        false,
        false,
    )?;
    if q.inject {
        project(
            client,
            30,
            10240,
            4,
            20480,
            weights[2].clone(),
            normalized.clone(),
            q.injection.clone(),
            q.fault.clone(),
            q.count,
            0,
            10240,
            0,
            4,
            false,
            false,
        )?;
    }
    launch(
        client,
        Kernel::GrLow,
        [(q.count * 320).div_ceil(256), 1, 1],
        vec![
            down.clone(),
            low.clone(),
            q.injection.clone(),
            q.fault.clone(),
        ],
        info,
    )?;
    project(
        client,
        30,
        320,
        10240,
        640,
        weights[1].clone(),
        low.clone(),
        up.clone(),
        q.fault.clone(),
        q.count,
        0,
        320,
        0,
        10240,
        false,
        false,
    )?;
    launch(
        client,
        Kernel::GrMix,
        [(q.count * 2560).div_ceil(256), 1, 1],
        vec![
            normalized.clone(),
            up.clone(),
            q.mixed.clone(),
            q.fault.clone(),
        ],
        info,
    )
}
pub(crate) fn ab(q: crate::strata_gdn::Ab<'_>, weights: [Handle; 2]) -> Result<()> {
    if q.columns != 2560
        || !(1..=8).contains(&q.count)
        || !(1..=48).contains(&q.heads)
        || u::words(q.arena, q.controls)? != 8 * q.heads * 2
        || u::words(q.arena, q.status)? != 8 * q.heads
        || q.params.size_in_used() < (q.heads * 8) as u64
        || u::words(q.arena, q.fault)? != 1
    {
        return Err(Error::InvalidBufferLength);
    }
    crate::resident_batch_projection::validate_view(
        q.count,
        q.first,
        q.stride,
        2560,
        u::words(q.arena, q.input)?,
    )?;
    ab_handles(
        AbHandles {
            client: q.arena.client().map_err(u::fault)?,
            input: u::handle(q.arena, q.input)?,
            params: q.params,
            controls: u::handle(q.arena, q.controls)?,
            status: u::handle(q.arena, q.status)?,
            fault: u::handle(q.arena, q.fault)?,
            count: q.count,
            heads: q.heads,
            first: q.first,
            stride: q.stride,
        },
        weights,
    )
}
pub(crate) struct AbHandles<'a> {
    pub client: &'a ComputeClient<WgpuRuntime>,
    pub input: Handle,
    pub params: Handle,
    pub controls: Handle,
    pub status: Handle,
    pub fault: Handle,
    pub count: usize,
    pub heads: usize,
    pub first: usize,
    pub stride: usize,
}
/// Full prompt AB producer: its raw projections use the authenticated carved
/// region. The supplied controls/status can be full prompt owners or the
/// existing small verifier owners, without changing the arithmetic or MAX8.
pub(crate) fn ab_handles(q: AbHandles<'_>, weights: [Handle; 2]) -> Result<()> {
    let raw = gdn_controls()?[0].clone();
    let extent = |h: &Handle, n: usize| {
        n.checked_mul(4)
            .is_some_and(|bytes| h.size_in_used() >= bytes as u64)
    };
    if !(1..=groups::CAPACITY).contains(&q.count)
        || !(1..=48).contains(&q.heads)
        || q.stride < 2560
        || q.stride
            .checked_mul(q.count - 1)
            .and_then(|n| q.first.checked_add(n))
            .and_then(|n| n.checked_add(2560))
            .is_none_or(|n| !extent(&q.input, n))
        || !extent(&q.params, q.heads * 2)
        || !extent(&q.controls, q.count * q.heads * 2)
        || !extent(&q.status, q.count * q.heads)
        || !extent(&raw, q.count * q.heads * 2)
        || q.fault.size_in_used() != 4
        || [&q.input, &q.params, &q.controls, &q.status, &q.fault]
            .iter()
            .any(|h| storage::overlaps(&raw, h).unwrap_or(true))
    {
        return Err(Error::InvalidBufferLength);
    }
    let outputs = [&q.controls, &q.status, &q.fault];
    for (i, h) in outputs.iter().enumerate() {
        if [&q.input, &q.params]
            .iter()
            .chain(outputs[..i].iter())
            .any(|other| storage::overlaps(h, other).unwrap_or(true))
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let client = q.client;
    let fault = q.fault;
    for (i, w) in weights.into_iter().enumerate() {
        project(
            client,
            30,
            2560,
            q.heads,
            5120,
            w,
            q.input.clone(),
            raw.clone(),
            fault.clone(),
            q.count,
            q.first,
            q.stride,
            i * q.heads,
            q.heads * 2,
            false,
            false,
        )?;
    }
    let info = [
        q.count as u32,
        0,
        q.heads as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    launch(
        client,
        Kernel::AbControls,
        [(q.count * q.heads).div_ceil(256), 1, 1],
        vec![raw, q.params, q.controls, q.status, fault],
        info,
    )
}

#[cfg(test)]
mod tests {
    fn h(b: &[u8], at: usize) -> f32 {
        half::f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32()
    }
    fn value(dtype: u32, b: &[u8], col: usize) -> f32 {
        let table = [
            -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89.,
            113.,
        ];
        if dtype == 8 {
            return h(b, 0) * (b[2 + col] as i8 as f32);
        }
        if dtype == 20 {
            let q = (b[2 + col % 16] >> if col >= 16 { 4 } else { 0 }) & 15;
            return h(b, 0) * table[q as usize];
        }
        if dtype == 23 {
            let sub = col / 32;
            let lo = (b[4 + sub / 2] >> ((sub % 2) * 4)) & 15;
            let hi = (u16::from_le_bytes([b[2], b[3]]) >> (sub * 2)) & 3;
            let q = (b[8 + sub * 16 + col % 16] >> if col % 32 >= 16 { 4 } else { 0 }) & 15;
            return (h(b, 0) * ((lo as i32 | ((hi as i32) << 4)) - 32) as f32) * table[q as usize];
        }
        if dtype == 14 {
            let sub = col / 16;
            let half = sub / 8;
            let quarter = sub % 8 / 2;
            let off = sub % 2 * 16 + col % 16;
            let lo = b[half * 64 + (quarter % 2) * 32 + off];
            let hi = b[128 + half * 32 + off];
            let code = ((lo >> if quarter >= 2 { 4 } else { 0 }) & 15)
                | (((hi >> (quarter * 2)) & 3) << 4);
            return (h(b, 208) * (b[192 + sub] as i8 as f32)) * (code as i32 - 32) as f32;
        }
        let group = col / 32;
        let i = group % 4;
        let scale = |minimum: bool| {
            let at = if minimum { 8 } else { 4 };
            let mut v = b[at + i] & 63;
            if group >= 4 {
                let c = b[12 + i];
                let hi = b[at + i] >> 6;
                v = (if minimum { c >> 4 } else { c & 15 }) | (hi << 4);
            }
            v as f32
        };
        let byte = b[if dtype == 13 { 48 } else { 16 } + (group / 2) * 32 + col % 32];
        let mut q = (byte >> if group % 2 == 1 { 4 } else { 0 }) & 15;
        if dtype == 13 {
            q |= ((b[16 + col % 32] >> group) & 1) << 4;
        }
        (h(b, 0) * scale(false)) * q as f32 - h(b, 2) * scale(true)
    }
    #[test]
    fn prompt_original_weight_images_match_independent_canonical_decoders() {
        for (kind, dtype) in [
            (kuno_text::TensorType::Q8_0, 8),
            (kuno_text::TensorType::Q4K, 12),
            (kuno_text::TensorType::Q5K, 13),
            (kuno_text::TensorType::Q6K, 14),
            (kuno_text::TensorType::Iq4Nl, 20),
            (kuno_text::TensorType::Iq4Xs, 23),
        ] {
            for salt in 0..11 {
                let mut b = (0..kind.block_bytes() as usize)
                    .map(|i| (i * 37 + salt * 19) as u8)
                    .collect::<Vec<_>>();
                if dtype == 14 {
                    for x in &mut b[192..208] {
                        *x = (*x % 17) as i8 as u8;
                    }
                    b[208..210]
                        .copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                } else {
                    b[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
                    if matches!(dtype, 12 | 13) {
                        b[2..4].copy_from_slice(
                            &half::f16::from_f32(0.015625).to_bits().to_le_bytes(),
                        );
                    }
                }
                let mut out = vec![0.; if matches!(dtype, 8 | 20) { 32 } else { 256 }];
                kind.decode(&b, &mut out).unwrap();
                for (col, &expected) in out.iter().enumerate() {
                    assert_eq!(
                        half::f16::from_f32(value(dtype, &b, col)).to_bits(),
                        half::f16::from_f32(expected).to_bits(),
                        "dtype={dtype} col={col}"
                    );
                }
            }
        }
    }
    #[test]
    fn prompt_operand_rounding_and_edge_tiles_match_independent_matrix_products() {
        for bfloat in [false, true] {
            for count in [1usize, 3, 8, 17] {
                for rows in [1usize, 7, 8, 19] {
                    let columns = 64;
                    let round = |x| {
                        if bfloat {
                            half::bf16::from_f32(x).to_f32()
                        } else {
                            half::f16::from_f32(x).to_f32()
                        }
                    };
                    let x = (0..count * columns)
                        .map(|i| round(((i * 17 % 101) as f32 - 50.) / 31.))
                        .collect::<Vec<_>>();
                    let w = (0..rows * columns)
                        .map(|i| round(((i * 19 % 97) as f32 - 48.) / 37.))
                        .collect::<Vec<_>>();
                    let mut hits = vec![0u8; rows * count];
                    for gt in 0..count.div_ceil(8) {
                        for gr in 0..rows.div_ceil(8) {
                            for tid in 0..64 {
                                let t = gt * 8 + tid / 8;
                                let row = gr * 8 + tid % 8;
                                if t >= count || row >= rows {
                                    continue;
                                }
                                let mut sum = 0f32;
                                for k in (0..columns).step_by(32) {
                                    for j in 0..32 {
                                        sum = x[t * columns + k + j]
                                            .mul_add(w[row * columns + k + j], sum);
                                    }
                                }
                                let expected = x[t * columns..(t + 1) * columns]
                                    .iter()
                                    .zip(&w[row * columns..(row + 1) * columns])
                                    .map(|(&a, &b)| a as f64 * b as f64)
                                    .sum::<f64>();
                                assert!(
                                    (sum as f64 - expected).abs() < 1e-5 * (1. + expected.abs())
                                );
                                hits[t * rows + row] += 1;
                            }
                        }
                    }
                    assert!(hits.iter().all(|&n| n == 1));
                }
            }
        }
        for x in [0., 1., -1., 1.00390625, -1.00390625, 0.000000001, 127.5] {
            let bits = f32::to_bits(x);
            let actual = ((bits.wrapping_add(32767).wrapping_add((bits >> 16) & 1)) >> 16) as u16;
            assert_eq!(actual, half::bf16::from_f32(x).to_bits());
        }
    }
}
