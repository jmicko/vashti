//! Connected Rust/Vulkan translation of MIT Strata's Intel fused GR pipeline.
//! Resident BF16 owners, original norm weights and private scratch are retained
//! by the parent through graph completion. No private allocation or readback.
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

pub(crate) const PARTIAL_WIDTH: usize = 80 * 324;
pub(crate) const PRIVATE_INCREMENT: u64 = 524288;
#[derive(Clone, Copy)]
pub(crate) struct Buffers {
    pub normalized: BufferId,
    pub scales: BufferId,
    pub partial: BufferId,
    pub down: BufferId,
    pub low: BufferId,
    pub injection: BufferId,
    pub mixed: BufferId,
}
pub(crate) struct Read<'a> {
    pub arena: &'a PrivateArena,
    pub residual: BufferId,
    pub previous: Option<(BufferId, BufferId)>,
    pub gamma: Handle,
    pub buffers: Buffers,
    pub fault: BufferId,
    pub count: usize,
    pub first: usize,
    pub epsilon: f32,
    pub inject: bool,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Kernel {
    Norm,
    Slices,
    Reduce,
    Up,
    Write,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Reduce => 128,
            Self::Write => 64,
            _ => 256,
        }
    }
    fn source(self) -> String {
        let all = include_str!("strata_gr.wgsl");
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
            Self::Norm => "NORM",
            Self::Slices => "SLICES",
            Self::Reduce => "REDUCE",
            Self::Up => "UP",
            Self::Write => "WRITE",
        }));
        if matches!(self, Self::Slices | Self::Up) {
            source.push_str(section("DOT"));
        }
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Norm => "StrataGrNormPendingWrite",
            Self::Slices => "StrataGrSlicedDownInject",
            Self::Reduce => "StrataGrFixedSliceReduce",
            Self::Up => "StrataGrUpMixCommit",
            Self::Write => "StrataGrBoundaryWrite",
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
fn enqueue(
    client: &ComputeClient<WgpuRuntime>,
    kernel: Kernel,
    grid: u32,
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    if grid == 0 || handles.is_empty() || handles.iter().any(|h| h.stream != handles[0].stream) {
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
    // SAFETY: parent retains authentic arena/private and resident public roots;
    // all fixed geometry/runtime intervals are validated before this sequence.
    unsafe {
        client.launch_unchecked(Box::new(kernel), CubeCount::Static(grid, 1, 1), args);
    }
    Ok(())
}
pub(crate) fn read(request: Read<'_>, weights: [Handle; 3]) -> Result<()> {
    let Read {
        arena,
        residual,
        previous,
        gamma,
        buffers: b,
        fault,
        count,
        first,
        epsilon,
        inject,
    } = request;
    if !crate::strata_intel_core::enabled()
        || !(1..=8).contains(&count)
        || !first.is_multiple_of(10240)
        || !epsilon.is_finite()
        || epsilon <= 0.
    {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    let (source, block) = previous.unwrap_or((residual, b.mixed));
    let expected = [
        (source, 81920),
        (residual, 81920),
        (block, 20480),
        (b.normalized, 81920),
        (b.scales, 64),
        (b.partial, 207360),
        (b.down, 2560),
        (b.low, 2560),
        (b.injection, 32),
        (b.mixed, 20480),
        (fault, 1),
    ];
    for (id, n) in expected {
        if u::words(arena, id)? != n {
            return Err(Error::InvalidBufferLength);
        }
    }
    if first.checked_add(count * 10240).is_none_or(|n| n > 81920)
        || gamma.size_in_used() != 40960
        || weights[0].size_in_used() != 6553600
        || weights[1].size_in_used() != 6553600
        || inject && weights[2].size_in_used() != 81920
    {
        return Err(Error::InvalidBufferLength);
    }
    let h = |id| u::handle(arena, id);
    let info = [
        count as u32,
        first as u32,
        previous.is_some() as u32,
        inject as u32,
        epsilon.to_bits(),
        0,
        0,
        0,
    ];
    enqueue(
        client,
        Kernel::Norm,
        (count * 4) as u32,
        vec![
            h(source)?,
            h(block)?,
            h(b.injection)?,
            gamma,
            h(b.normalized)?,
            h(b.scales)?,
            h(fault)?,
        ],
        info,
    )?;
    enqueue(
        client,
        Kernel::Slices,
        80,
        vec![
            h(b.normalized)?,
            weights[0].clone(),
            weights[2].clone(),
            h(b.partial)?,
        ],
        info,
    )?;
    enqueue(
        client,
        Kernel::Reduce,
        (count * 324).div_ceil(128) as u32,
        vec![h(b.partial)?, h(b.down)?, h(b.injection)?, h(fault)?],
        info,
    )?;
    // No same-node RO/RW alias when apply=false: source is not read in that
    // branch, so use the already-read-only normalized owner as its dummy.
    let up_source = if previous.is_some() {
        source
    } else {
        b.normalized
    };
    let up_block = if previous.is_some() {
        block
    } else {
        b.normalized
    };
    enqueue(
        client,
        Kernel::Up,
        160,
        vec![
            weights[1].clone(),
            h(b.down)?,
            h(b.normalized)?,
            h(b.scales)?,
            h(up_source)?,
            h(up_block)?,
            h(residual)?,
            h(b.low)?,
            h(b.mixed)?,
            h(fault)?,
        ],
        info,
    )?;
    Ok(())
}
pub(crate) fn write(
    arena: &PrivateArena,
    source: BufferId,
    block: BufferId,
    injection: BufferId,
    output: BufferId,
    fault: BufferId,
    count: usize,
) -> Result<()> {
    if !crate::strata_intel_core::enabled() || !(1..=8).contains(&count) || source == output {
        return Err(Error::InvalidBufferLength);
    }
    for (id, n) in [
        (source, 81920),
        (block, 20480),
        (injection, 32),
        (output, 81920),
        (fault, 1),
    ] {
        if u::words(arena, id)? != n {
            return Err(Error::InvalidBufferLength);
        }
    }
    if crate::strata_prompt::active() {
        return crate::strata_prompt::write_handles(
            arena.client().map_err(u::fault)?,
            [
                u::handle(arena, source)?,
                u::handle(arena, block)?,
                u::handle(arena, injection)?,
                u::handle(arena, output)?,
                u::handle(arena, fault)?,
            ],
            count,
        );
    }
    enqueue(
        arena.client().map_err(u::fault)?,
        Kernel::Write,
        (count * 10240).div_ceil(64) as u32,
        vec![
            u::handle(arena, source)?,
            u::handle(arena, block)?,
            u::handle(arena, injection)?,
            u::handle(arena, output)?,
            u::handle(arena, fault)?,
        ],
        [count as u32, 0, 1, 1, 0, 0, 0, 0],
    )
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Norm,
        Kernel::Slices,
        Kernel::Reduce,
        Kernel::Up,
        Kernel::Write,
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
    #[test]
    fn native_low_activation_keeps_division_in_actual_up_shader() {
        // Fixed public rounded denominator: division and multiplying by its
        // rounded reciprocal differ by one F32 ULP even before exp differences.
        let x = f32::from_bits(0x3e4c_cccd);
        let denominator = f32::from_bits(0x3fe8_cc2c);
        assert_eq!((x / denominator).to_bits(), 0x3de1_3646);
        assert_eq!((x * (1.0 / denominator)).to_bits(), 0x3de1_3645);
        let source = super::Kernel::Up.source();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        use wgpu::naga as n;
        let entry = &module.entry_points[0];
        assert_eq!(entry.workgroup_size, [256, 1, 1]);
        let expression = |name: &str| {
            *entry
                .function
                .named_expressions
                .iter()
                .find(|(_, candidate)| candidate.as_str() == name)
                .unwrap()
                .0
        };
        let n::Expression::Binary {
            op: n::BinaryOperator::Divide,
            left,
            ..
        } = entry.function.expressions[expression("v")]
        else {
            panic!("native GR activation must divide by sigmoid denominator");
        };
        assert_eq!(left, expression("x"));
        let words = n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options {
                lang_version: (1, 3),
                zero_initialize_workgroup_memory:
                    n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                ..Default::default()
            },
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        assert_eq!(words[0], 0x0723_0203);
        let prompt = crate::strata_prompt::sources()
            .into_iter()
            .find(|shader| shader.source.contains("let a=down[i*2u]/4.0"))
            .unwrap();
        let module = n::front::wgsl::parse_str(&prompt.source).unwrap();
        n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let entry = &module.entry_points[0];
        let handle = *entry
            .function
            .named_expressions
            .iter()
            .find(|(_, name)| name.as_str() == "v")
            .unwrap()
            .0;
        let n::Expression::Binary {
            op: n::BinaryOperator::Multiply,
            right,
            ..
        } = entry.function.expressions[handle]
        else {
            panic!("prompt GR injection must scale its rounded sigmoid");
        };
        assert!(matches!(
            entry.function.expressions[right],
            n::Expression::Binary {
                op: n::BinaryOperator::Divide,
                ..
            }
        ));
    }
}
