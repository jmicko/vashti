//! Private model-weight staging at an authenticated target layer boundary.
//! Model roots remain original; transient selected copies live in PrivateArena
//! and are erased with it. No allocation, route readback or disk access at decode.
use crate::private_arena::{BufferId, PrivateArena};
use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::{
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
pub(crate) const OWNER_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const OWNER_WORDS: usize = OWNER_BYTES as usize / 4;
pub(crate) const MAIN_BYTES: u64 = 3 * OWNER_BYTES;
thread_local! {static STAGES:RefCell<Option<[Handle;3]>>=const {RefCell::new(None)};}
fn reject() -> Error {
    Error::InvalidFormat("native Strata stage owner")
}
pub(crate) struct Scope;
impl Scope {
    pub(crate) fn new(arena: &PrivateArena, ids: Option<[BufferId; 3]>) -> Result<Option<Self>> {
        if !crate::strata_intel_core::enabled() {
            return Ok(None);
        }
        let ids = ids.ok_or_else(reject)?;
        if ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2] {
            return Err(reject());
        }
        let handles = [
            arena
                .handle(ids[0])
                .map_err(crate::resident_util::fault)?
                .clone(),
            arena
                .handle(ids[1])
                .map_err(crate::resident_util::fault)?
                .clone(),
            arena
                .handle(ids[2])
                .map_err(crate::resident_util::fault)?
                .clone(),
        ];
        if handles.iter().any(|h| {
            h.size_in_used() != OWNER_BYTES
                || h.size() != OWNER_BYTES
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
                || h.stream != handles[0].stream
        }) {
            return Err(reject());
        }
        STAGES.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_some() {
                return Err(reject());
            }
            *s = Some(handles);
            Ok(Some(Self))
        })
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        STAGES.with(|s| *s.borrow_mut() = None);
    }
}
pub(crate) fn handles() -> Result<[Handle; 3]> {
    STAGES.with(|s| s.borrow().clone().ok_or_else(reject))
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Kernel;
impl Kernel {
    fn source() -> String {
        format!(
            "// STRATA_PHYSICAL_EXPERTS\nstruct Params {{first:u32,count:u32,vectors:u32,role:u32}}\n@group(0) @binding(0) var<storage,read> gate0:array<vec4<u32>>;\n@group(0) @binding(1) var<storage,read> gate1:array<vec4<u32>>;\n@group(0) @binding(2) var<storage,read> gate2:array<vec4<u32>>;\n@group(0) @binding(3) var<storage,read> gate3:array<vec4<u32>>;\n@group(0) @binding(4) var<storage,read> gate4:array<vec4<u32>>;\n@group(0) @binding(5) var<storage,read> map:array<u32>;\n@group(0) @binding(6) var<storage,read_write> stage:array<vec4<u32>>;\n@group(0) @binding(7) var<storage,read> addresses:array<vec4<u32>>;\n@group(0) @binding(8) var<uniform> p:Params;\nvar<private> strata_gate_address:vec2<u32>;\n{}",
            include_str!("strata_stage.wgsl")
        )
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "StrataFetchOriginalHostExperts"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
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
            source: Self::source(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
        })
    }
}
static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) fn calls() -> u64 {
    CALLS.load(std::sync::atomic::Ordering::Relaxed)
}
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    g: crate::quant_matvec::grouped_moe::Geometry,
    pages: &[Handle],
    map: &Handle,
    first: usize,
    count: usize,
) -> Result<()> {
    let stages = handles()?;
    if pages.len() != 15
        || !(1..=4).contains(&count)
        || first
            .checked_add(804)
            .is_none_or(|n| n as u64 * 4 > map.size_in_used())
    {
        return Err(reject());
    }
    for (role, projection) in g.projections().into_iter().enumerate() {
        let bytes = projection.expert_bytes();
        if !bytes.is_multiple_of(16) || bytes * 40 > OWNER_BYTES as usize {
            return Err(reject());
        }
        let mut args = KernelArguments::new();
        for (shard, slots) in crate::daslab_profile::SOURCE_PARTS.into_iter().enumerate() {
            let source = &pages[role * 5 + shard];
            if source.size_in_used() != bytes as u64 * slots as u64 {
                return Err(reject());
            }
            args = args.with_buffer(source.clone().binding());
        }
        args = args
            .with_buffer(map.clone().binding())
            .with_buffer(stages[role].clone().binding());
        args = crate::strata_intel_core::bind_addresses(client, pages, args)?;
        let info = [first as u32, count as u32, (bytes / 16) as u32, role as u32];
        args = args.with_info(MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        ));
        // SAFETY: original full role pages and the authentic PrivateArena stage
        // roots cover every checked selected copy. Graph orders it before consumers.
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel),
                CubeCount::Static(
                    (bytes / 16).div_ceil(256) as u32,
                    if count == 4 { 40 } else { 30 },
                    1,
                ),
                args,
            );
        }
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    vec![
        Kernel
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap(),
    ]
}
