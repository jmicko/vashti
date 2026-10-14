//! PUBLIC-only external-buffer registration checks and readonly lowering.
use super::storage::WgpuResource;
use crate::{AutoCompiler, AutoRepresentation};
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::{
    kernel::CompiledKernel,
    public_external::{PublicExternalHostSource, is_public_external_binding},
    server::KernelArguments,
};

pub(super) fn fault(reason: &'static str) -> cubecl_runtime::server::ServerError {
    cubecl_runtime::server::ServerError::Generic {
        reason: reason.into(),
        backtrace: cubecl_common::backtrace::BackTrace::capture(),
    }
}

/// One immutable-source pool per server; otherwise one Buffer could acquire
/// consumers in two pools and a single-stream detach could not join both.
pub(super) fn configure_stream(existing: Option<StreamId>) -> Result<(), &'static str> {
    if existing.is_some() {
        return Err("public external owner stream already fixed");
    }
    Ok(())
}

/// External consumers stay on the owning stream so detach joins every queued use.
pub(super) fn validate_stream(
    args: &KernelArguments,
    stream: StreamId,
) -> Result<(), &'static str> {
    if args
        .buffers
        .iter()
        .chain(&args.read_dependencies)
        .any(|b| is_public_external_binding(b) && b.stream != stream)
    {
        return Err("public external cross-stream consumer");
    }
    Ok(())
}

pub(super) fn external_mask(args: &KernelArguments) -> Result<u32, &'static str> {
    if args.buffers.len() > 32 && args.buffers.iter().any(is_public_external_binding) {
        return Err("public external binding inventory exceeded");
    }
    Ok(args
        .buffers
        .iter()
        .enumerate()
        .fold(0, |mask, (index, binding)| {
            if is_public_external_binding(binding) {
                mask | (1 << index)
            } else {
                mask
            }
        }))
}

pub(super) fn check_source(
    device: &wgpu::Device,
    source: &PublicExternalHostSource<WgpuResource>,
) -> Result<(), &'static str> {
    let resource = source.resource();
    let usage = resource.buffer.usage();
    if resource.offset != 0
        || resource.size != source.bytes()
        || resource.buffer.size() != source.bytes()
        || (usage != wgpu::BufferUsages::STORAGE
            && usage != (wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC))
        || source.bytes() > device.limits().max_storage_buffer_binding_size
    {
        return Err("public external exact immutable descriptor");
    }
    // Native identity must actually validate on this device. A VkBuffer handle
    // alone does not encode VkDevice identity; wgpu's bind validation does.
    if unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_none()
        || unsafe { resource.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.is_none()
    {
        return Err("public external Vulkan device/buffer required");
    }
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("public-external-readonly-identity"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: resource.buffer.as_entire_binding(),
        }],
    });
    let invalid = cubecl_common::future::block_on(error_scope.pop()).is_some();
    drop((group, layout));
    if invalid {
        return Err("public external same-device readonly validation");
    }
    Ok(())
}

pub(super) fn lower(
    compiled: &mut CompiledKernel<AutoCompiler>,
    mask: u32,
) -> Result<(), &'static str> {
    if mask == 0 {
        return Ok(());
    }
    match compiled.repr.as_mut() {
        Some(AutoRepresentation::Wgsl(shader)) => {
            shader.set_public_external_readonly_mask(mask)?;
            compiled.source = shader.to_string();
        }
        None => {} // Handwritten WGSL must already declare LOAD-only sources.
        #[allow(unreachable_patterns)]
        _ => return Err("public external only audited WGSL lowering supported"),
    }
    check_readonly_source(&compiled.source, mask)
}

fn non_atomic_type(
    module: &wgpu::naga::Module,
    ty: wgpu::naga::Handle<wgpu::naga::Type>,
    depth: usize,
) -> bool {
    if depth >= 64 {
        return false;
    }
    match &module.types[ty].inner {
        wgpu::naga::TypeInner::Atomic(_) => false,
        wgpu::naga::TypeInner::Array { base, .. } => non_atomic_type(module, *base, depth + 1),
        wgpu::naga::TypeInner::Struct { members, .. } => members
            .iter()
            .all(|m| non_atomic_type(module, m.ty, depth + 1)),
        _ => true,
    }
}

pub(super) fn check_readonly_source(source: &str, mask: u32) -> Result<(), &'static str> {
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .map_err(|_| "public external WGSL reflection parse")?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|_| "public external WGSL reflection validation")?;
    let mut found = 0u32;
    for (_, variable) in module.global_variables.iter() {
        let Some(binding) = variable.binding.as_ref() else {
            continue;
        };
        if binding.group != 0 || binding.binding >= 32 || mask & (1 << binding.binding) == 0 {
            continue;
        }
        if !non_atomic_type(&module, variable.ty, 0)
            || variable.space
                != (wgpu::naga::AddressSpace::Storage {
                    access: wgpu::naga::StorageAccess::LOAD,
                })
        {
            return Err("public external writable/atomic source binding");
        }
        found |= 1 << binding.binding;
    }
    if found != mask {
        return Err("public external reflected binding missing");
    }
    Ok(())
}
