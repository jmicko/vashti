//! MIT Strata fb58e0 shared sigmoid-scale; original fault joins on MAIN.
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Kernel {
    Scale,
    Fold,
}
impl Kernel {
    fn source(self) -> String {
        match self {
  Self::Scale=>r#"
struct Params {count:u32,width:u32,r0:u32,r1:u32}
@group(0) @binding(0) var<storage,read_write> output:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> gate:array<f32>;
@group(0) @binding(2) var<storage,read_write> fault:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 if arrayLength(&fault)!=1u {return;}
 if p.count<1u || p.count>8u || p.width!=2560u || arrayLength(&output)!=5120u || arrayLength(&gate)!=8u {
  if all(id==vec3<u32>(0u)) {atomicOr(&fault[0],8u);}return;
 }
 if id.x>=640u || id.y>=p.count || id.z!=0u {return;}
 let at=id.y*640u+id.x;let raw=gate[id.y];let scale=1.0/(1.0+exp(-raw));
 let x=output[at];let y=x*scale;
 if abs(raw)<=bitcast<f32>(0x7f7fffffu) && all(abs(x)<=vec4<f32>(bitcast<f32>(0x7f7fffffu))) && all(abs(y)<=vec4<f32>(bitcast<f32>(0x7f7fffffu))) {output[at]=y;}
 else {output[at]=vec4<f32>(0.0);atomicOr(&fault[0],4u);}
}
"#.into(),
  Self::Fold=>r#"
@group(0) @binding(0) var<storage,read> shared_fault:array<u32>;
@group(0) @binding(1) var<storage,read_write> fault:array<atomic<u32>>;
@compute @workgroup_size(1)
fn main() {if arrayLength(&shared_fault)==1u && arrayLength(&fault)==1u {atomicOr(&fault[0],shared_fault[0]);}}
"#.into()
 }
    }
    fn dim(self) -> CubeDim {
        CubeDim::new_1d(if self == Self::Scale { 128 } else { 1 })
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Scale => "StrataNativeSharedScale",
            Self::Fold => "StrataNativeSharedFaultJoin",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self).cube_dim(self.dim())
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
            debug_info: None,
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: self.dim(),
        })
    }
}
pub(crate) fn scale(
    client: &ComputeClient<WgpuRuntime>,
    output: Handle,
    gate: Handle,
    fault: Handle,
    count: usize,
) -> Result<()> {
    if !(1..=8).contains(&count)
        || output.size_in_used() != 8 * 2560 * 4
        || gate.size_in_used() != 32
        || fault.size_in_used() != 4
    {
        return Err(Error::InvalidBufferLength);
    }
    let args = KernelArguments::new()
        .with_buffer(output.binding())
        .with_buffer(gate.binding())
        .with_buffer(fault.binding())
        .with_info(MetadataBindingInfo::custom(vec![
            count as u64 | (2560u64 << 32),
            0,
        ]));
    // SAFETY: original checked complete private owners; exact native rows.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Scale),
            CubeCount::Static(5, count as u32, 1),
            args,
        );
    }
    Ok(())
}
pub(crate) fn fold(
    client: &ComputeClient<WgpuRuntime>,
    shared: Handle,
    fault: Handle,
) -> Result<()> {
    if shared.size_in_used() != 4 || fault.size_in_used() != 4 {
        return Err(Error::InvalidBufferLength);
    }
    let args = KernelArguments::new()
        .with_buffer(shared.binding())
        .with_buffer(fault.binding());
    // SAFETY: owning target scope joins these original distinct fault roots.
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Fold), CubeCount::Static(1, 1, 1), args);
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Kernel::Scale, Kernel::Fold]
        .into_iter()
        .map(|k| {
            k.compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                k.address_type(),
            )
            .unwrap()
        })
        .collect()
}

pub(crate) struct QueueScope<'a> {
    client: &'a ComputeClient<WgpuRuntime>,
    active: bool,
}
impl<'a> QueueScope<'a> {
    pub(crate) fn begin(client: &'a ComputeClient<WgpuRuntime>) -> Result<Self> {
        client
            .native_strata_shared_boundary(
                cubecl_runtime::server::NativeStrataSharedBoundary::Begin,
            )
            .map_err(crate::resident_util::fault)?;
        Ok(Self {
            client,
            active: true,
        })
    }
    pub(crate) fn end(mut self) -> Result<()> {
        let result = self
            .client
            .native_strata_shared_boundary(cubecl_runtime::server::NativeStrataSharedBoundary::End)
            .map_err(crate::resident_util::fault);
        self.active = false;
        result
    }
}
impl Drop for QueueScope<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.client.native_strata_shared_boundary(
                cubecl_runtime::server::NativeStrataSharedBoundary::End,
            );
        }
    }
}
pub(crate) fn release_main(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    client
        .native_strata_shared_boundary(
            cubecl_runtime::server::NativeStrataSharedBoundary::ReleaseMain,
        )
        .map_err(crate::resident_util::fault)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_scale_and_fault_join_preserve_actual_read_write_owners_and_shader_abi() {
        use wgpu::naga as n;
        for kernel in [Kernel::Scale, Kernel::Fold] {
            let source = kernel.source();
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(
                module.entry_points[0].workgroup_size,
                [kernel.dim().x, 1, 1]
            );
            let uses = info.get_entry_point(0);
            let mut access = std::collections::BTreeMap::new();
            for (id, g) in module.global_variables.iter() {
                if let Some(binding) = &g.binding {
                    access.insert(
                        binding.binding,
                        (
                            uses[id].contains(n::valid::GlobalUse::READ),
                            uses[id].contains(n::valid::GlobalUse::WRITE),
                        ),
                    );
                }
            }
            if kernel == Kernel::Fold {
                assert_eq!(access.get(&0), Some(&(true, false))); // Shared writer's actual result, read by MAIN.
                assert!(access[&1].1);
                assert_eq!(access.len(), 2);
            } else {
                assert_eq!(access[&0], (true, true));
                assert_eq!(access[&1], (true, false));
                assert!(access[&2].1);
                assert_eq!(access.len(), 4);
            }
            n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options::default(),
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
    }
}
