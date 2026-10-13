//! GPU-owned real argmax boundary for the isolated Strata Rust engine.
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::{
    prelude::*,
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
#[derive(Clone)]
struct Packet;
impl KernelMetadata for Packet {
    fn name(&self) -> &'static str {
        "StrataGpuGreedyCarryPacket"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(128))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Packet {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: include_str!("strata-packet.wgsl").into(),
            repr: None,
            cube_dim: CubeDim::new_1d(128),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 4],
    count: usize,
    start: usize,
) {
    let mut args = KernelArguments::new();
    for owner in owners {
        args = args.with_buffer(owner.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(vec![
        (count as u64) | ((start as u64) << 32),
        0,
    ]));
    // Parent authenticates all complete disjoint private owners and coordinates;
    // no reduced span acquires free/erasure authority over a backing owner.
    unsafe {
        client.launch_unchecked(Box::new(Packet), CubeCount::Static(4, 1, 1), args);
    }
}
