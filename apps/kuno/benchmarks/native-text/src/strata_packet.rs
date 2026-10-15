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
struct Packet {
    selected: bool,
}
impl KernelMetadata for Packet {
    fn name(&self) -> &'static str {
        if self.selected {
            "StrataGpuSelectedCarryPacket"
        } else {
            "StrataGpuGreedyCarryPacket"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.selected)
            .cube_dim(CubeDim::new_1d(128))
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
            source: if self.selected {
                selected_shader()
            } else {
                shader()
            },
            repr: None,
            cube_dim: CubeDim::new_1d(128),
            debug_info: None,
        })
    }
}
pub(crate) fn launch_selected(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 4],
    count: usize,
    start: usize,
) {
    launch_impl(client, owners, count, start, true)
}
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 4],
    count: usize,
    start: usize,
) {
    launch_impl(client, owners, count, start, false)
}
fn launch_impl(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 4],
    count: usize,
    start: usize,
    selected: bool,
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
        client.launch_unchecked(
            Box::new(Packet { selected }),
            CubeCount::Static(crate::strata_port::verify_capacity() as u32, 1, 1),
            args,
        );
    }
}

pub(crate) fn selected_shader() -> String {
    configure(include_str!("strata-selected-packet.wgsl"))
}
pub(crate) fn shader() -> String {
    configure(include_str!("strata-packet.wgsl"))
}
fn configure(source: &str) -> String {
    source
        .replace(
            "VERIFY_ROWS",
            &crate::strata_port::verify_capacity().to_string(),
        )
        .replace(
            "PACKET_OWNER_WORDS",
            &(crate::resident_model::batch::target_rows::OWNER_BYTES / 4).to_string(),
        )
        .replace(
            "CARRY_FIRST",
            &(8 + crate::strata_port::verify_capacity()).to_string(),
        )
}
