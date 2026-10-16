//! Native retained-commit form of the original QSA suffix clear/zero fold.
//! Geometry stays fixed at proposed T; original accepted n/position are INFO.
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
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo, NativeStrataValue},
};
use kuno_text::{Error, Result};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Clear,
    Fold,
}
impl Kernel {
    fn source(self) -> String {
        let text = include_str!("strata_prefix_qsa.wgsl");
        let section = |name: &str| {
            text.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut source = section("COMMON").to_owned();
        source.push_str(section(match self {
            Self::Clear => "CLEAR",
            Self::Fold => "FOLD",
        }));
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Clear => "StrataQsaCommitClear",
            Self::Fold => "StrataQsaCommitFold",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(64))
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
            cube_dim: CubeDim::new_1d(64),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(
    arena: &PrivateArena,
    kernel: Kernel,
    handles: Vec<Handle>,
    proposed: usize,
    start: usize,
    accepted: usize,
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || !(2..=8).contains(&proposed)
        || !(1..=proposed).contains(&accepted)
        || start
            .checked_add(proposed)
            .is_none_or(|end| end > crate::qsa_gpu::MAX_CONTEXT)
        || handles.len() != 4
        || handles.iter().any(|h| h.stream != handles[0].stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let stride = crate::strata_qsa::cache_words();
    let info = [
        proposed as u32,
        start as u32,
        accepted as u32,
        stride as u32,
        0,
        0,
        0,
        0,
    ];
    let metadata = MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    )
    .with_native_parameter(1, NativeStrataValue::Position { offset: 0 })
    .with_native_parameter(2, NativeStrataValue::AcceptedCount);
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(metadata);
    let words = (proposed * stride).max(proposed.div_ceil(4) * 128);
    // SAFETY: caller checks original pending roots and physical inventories;
    // fixed proposed-span dispatches guard every current accepted suffix.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(words.div_ceil(64) as u32, 1, 1),
            args,
        );
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Kernel::Clear, Kernel::Fold]
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
