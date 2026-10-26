//! One immutable routed-X publication per authentic expert layer.
use super::*;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Gather;
impl KernelMetadata for Gather {
    fn name(&self) -> &'static str {
        "StrataExpertLayerGather"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Gather {
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
            source: include_str!("strata_prompt_layer_gather.wgsl").into(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}
#[cfg(feature = "strata-large-packed-tiles")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct PackedGather;
#[cfg(feature = "strata-large-packed-tiles")]
impl KernelMetadata for PackedGather {
    fn name(&self) -> &'static str {
        "StrataExpertLayerPackedGather"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
#[cfg(feature = "strata-large-packed-tiles")]
impl CubeTask<AutoCompiler> for PackedGather {
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
            source: include_str!("strata_prompt_layer_packed_gather.wgsl").into(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}
pub(crate) fn x() -> Result<Handle> {
    STAGES.with(|s| {
        s.borrow()
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .layer_x()
    })
}
pub(crate) fn enqueue(
    client: &ComputeClient<WgpuRuntime>,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
    map: &[u32],
) -> Result<()> {
    let (group_count, padded) = groups::validate_layer(map, count)?;
    let mixed = mixed_input_image(&owners[1], count, 0, 2560, false);
    let source = mixed.clone().unwrap_or_else(|| owners[1].clone());
    let x = x()?;
    let root = stages()?[0].clone();
    let mut h = root.clone();
    h.offset_start = Some(root.offset_start.unwrap_or(0) + 20 * 1024 * 1024);
    h.offset_end = Some(root.size() - h.offset_start.unwrap() - 5 * 1024 * 1024);
    if x.size_in_used() != groups::LAYER_X_BYTES || h.size_in_used() != 5 * 1024 * 1024 {
        return Err(Error::InvalidBufferLength);
    }
    let words = [
        count as u32,
        group_count as u32,
        u32::from(mixed.is_some()),
        padded as u32,
        u32::from(cfg!(feature = "strata-large-packed-tiles")),
        0,
        0,
        0,
    ];
    #[cfg(feature = "strata-large-packed-tiles")]
    launch_packed(
        client,
        [owners[0].clone(), source, x, h, fault.clone()],
        &words,
    )?;
    #[cfg(not(feature = "strata-large-packed-tiles"))]
    launch(
        client,
        [owners[0].clone(), source, x, h, fault.clone()],
        &words,
    )?;
    for (key, n) in [
        ("expert_layer_gather_launches", 1),
        ("expert_layer_gather_rows", (count * 10) as u64),
        ("expert_layer_gather_padded_rows", padded as u64),
        ("expert_layer_x_owner_bytes", groups::LAYER_X_BYTES),
    ] {
        crate::strata_prefill_profile::count(key, n);
    }
    Ok(())
}

pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    handles: [Handle; 5],
    words: &[u32; 8],
) -> Result<()> {
    launch_inner(client, handles, words, false)
}
#[cfg(feature = "strata-large-packed-tiles")]
pub(crate) fn launch_packed(
    client: &ComputeClient<WgpuRuntime>,
    handles: [Handle; 5],
    words: &[u32; 8],
) -> Result<()> {
    launch_inner(client, handles, words, true)
}
fn launch_inner(
    client: &ComputeClient<WgpuRuntime>,
    handles: [Handle; 5],
    words: &[u32; 8],
    packed: bool,
) -> Result<()> {
    if !(1..=4096).contains(&words[0])
        || !(1..=512).contains(&words[1])
        || words[2] > 1
        || words[3] < words[0] * 10
        || words[3] > groups::LAYER_PADDED_ROWS as u32
        || words[3] % 32 != 0
        || words[4] != u32::from(packed)
        || words[5..].iter().any(|&v| v != 0)
        || (packed && !cfg!(feature = "strata-large-packed-tiles"))
    {
        return Err(Error::InvalidBufferLength);
    }
    let args = KernelArguments::new()
        .with_buffers(handles.into_iter().map(Handle::binding).collect())
        .with_info(MetadataBindingInfo::custom(
            words
                .chunks_exact(2)
                .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
                .collect(),
        ));
    let task: Box<dyn CubeTask<AutoCompiler>> = {
        #[cfg(feature = "strata-large-packed-tiles")]
        {
            if packed {
                Box::new(PackedGather)
            } else {
                Box::new(Gather)
            }
        }
        #[cfg(not(feature = "strata-large-packed-tiles"))]
        {
            Box::new(Gather)
        }
    };
    unsafe {
        client.launch_unchecked(task, CubeCount::Static(5, words[3], 1), args);
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    let sources = vec![
        Gather
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap(),
    ];
    #[cfg(feature = "strata-large-packed-tiles")]
    let sources = sources
        .into_iter()
        .chain([PackedGather
            .compile(
                &mut AutoCompiler::Wgsl(Default::default()),
                &Default::default(),
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap()])
        .collect();
    sources
}
#[path = "strata_prompt_layer_gather_probe.rs"]
mod probe;
#[path = "strata_layer_probe_util.rs"]
pub(crate) mod probe_util;
pub(crate) use probe::cases as public_producers;
#[cfg(feature = "strata-large-packed-tiles")]
#[path = "strata_prompt_layer_packed_gather_probe.rs"]
mod packed_probe;
#[cfg(feature = "strata-large-packed-tiles")]
pub(crate) use packed_probe::cases as public_packed_producers;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_gather_wgsl_has_bounded_two_dimensional_dispatch() {
        let source = &sources()[0].source;
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(groups::LAYER_X_BYTES, 277 * 1024 * 1024 + 512 * 1024);
        assert!(groups::LAYER_PADDED_ROWS <= 65535);
        assert!(source.contains("let row=id.y;let i=row*1280u+id.x;"));
    }
    #[cfg(feature = "strata-large-packed-tiles")]
    #[test]
    fn packed_layer_gather_actual_wgsl_keeps_original_owner_shape_and_distinct_format() {
        let all = sources();
        assert_eq!(all.len(), 2);
        let source = &all[1].source;
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert!(source.contains("p.spare.x!=1u"));
        assert!(source.contains("if members>=256u"));
        assert!(source.contains("arrayLength(&packed)!=72744960u"));
        assert_eq!(PackedGather.name(), "StrataExpertLayerPackedGather");
        assert_eq!(Gather.name(), "StrataExpertLayerGather");
    }
    #[test]
    fn layer_gather_explicit_physical_charge_preserves_expert_budget() {
        assert_eq!(
            allocation::ROOT_SIZES,
            [756 * 1024 * 1024, 780 * 1024 * 1024, 32 * 1024 * 1024]
        );
        assert_eq!(layout::layer_extra_bytes(), 138 * 1024 * 1024);
        assert_eq!(
            crate::target_lab_build::NATIVE_SHARED_MAIN_BYTES - (31u64 << 30),
            layout::layer_extra_bytes()
        );
        assert_eq!(
            allocation::MAIN_BYTES - (1430 * 1024 * 1024),
            layout::layer_extra_bytes()
        );
        assert!(
            crate::target_lab_build::NATIVE_SHARED_MAIN_BYTES + 512 * 1024 * 1024 < 34242297856
        );
    }
}
