//! Actual device admission controls; no dispatch, buffer or request is invented.
//! Compare original ordinary admission with sealed preparation using the SAME
//! real module/layout/device/entry. Original operator controls exercise actual
//! selected native execution, owners, parameters, faults and complete erasure.
use kuno_text::{Error, Result};
use serde_json::{Value, json};
use std::borrow::Cow;

const SOURCE: &str = "@group(0) @binding(0) var<storage, read> source:array<u32>; @compute @workgroup_size(1) fn main() { if arrayLength(&source)>0u { if source[0]>0u { return; } } }";
fn reject() -> Error {
    Error::InvalidFormat("native prepared actual admission control")
}
fn module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Public native prepared admission control"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
fn layout(device: &wgpu::Device, ty: Option<wgpu::BufferBindingType>) -> wgpu::PipelineLayout {
    let entries: Vec<_> = ty
        .into_iter()
        .map(|ty| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bindings = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Public native prepared actual bindings"),
        entries: &entries,
    });
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Public native prepared actual layout"),
        bind_group_layouts: &[Some(&bindings)],
        immediate_size: 0,
    })
}
fn descriptor<'a>(
    module: &'a wgpu::ShaderModule,
    layout: Option<&'a wgpu::PipelineLayout>,
    entry: &'a str,
) -> wgpu::ComputePipelineDescriptor<'a> {
    wgpu::ComputePipelineDescriptor {
        label: Some("Public native prepared admission control"),
        layout,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    }
}
fn compare(
    device: &wgpu::Device,
    desc: &wgpu::ComputePipelineDescriptor<'_>,
    expected: bool,
    name: &'static str,
) -> Result<Value> {
    let prepared = device.kuno_prepare_native_compute_pipeline(desc);
    if prepared.is_ok() != expected {
        return Err(reject());
    }
    if let Ok(token) = &prepared {
        if !token.is_for_device(device) || !token.is_for_module(desc.module) {
            return Err(reject());
        }
    }
    // This original executable exists only in the fixed-public comparison,
    // never to prime or time the whole-model speed fixture.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let ordinary = device.create_compute_pipeline(desc);
    let original_ok = cubecl::future::block_on(scope.pop()).is_none();
    drop(ordinary);
    if original_ok != expected {
        return Err(reject());
    }
    Ok(
        json!({"case":name,"expected_admitted":expected,"original_matches":true,
        "prepared_matches":true,"no_dispatch_or_private_data":true}),
    )
}
pub(crate) fn cases(device: &wgpu::Device) -> Result<Vec<Value>> {
    let good = module(device, SOURCE);
    let other = module(device, SOURCE);
    let storage = layout(
        device,
        Some(wgpu::BufferBindingType::Storage { read_only: true }),
    );
    let uniform = layout(device, Some(wgpu::BufferBindingType::Uniform));
    let missing = layout(device, None);
    let mut cases = vec![
        compare(
            device,
            &descriptor(&good, Some(&storage), "main"),
            true,
            "explicit_original_layout",
        )?,
        compare(
            device,
            &descriptor(&good, None, "main"),
            true,
            "sealed_original_automatic_layout",
        )?,
        compare(
            device,
            &descriptor(&good, Some(&storage), "absent"),
            false,
            "missing_original_entry",
        )?,
        compare(
            device,
            &descriptor(&good, Some(&uniform), "main"),
            false,
            "wrong_original_binding_kind",
        )?,
        compare(
            device,
            &descriptor(&good, Some(&missing), "main"),
            false,
            "missing_original_binding",
        )?,
    ];
    let token = device
        .kuno_prepare_native_compute_pipeline(&descriptor(&good, Some(&storage), "main"))
        .map_err(|_| reject())?;
    if token.is_for_module(&other) {
        return Err(reject());
    }
    cases.push(
        json!({"case":"distinct_actual_module_is_not_same_bytes_authority",
        "rejected_other_module":true,"no_dispatch_or_private_data":true}),
    );
    let too_wide = device
        .limits()
        .max_compute_workgroup_size_x
        .checked_add(1)
        .ok_or_else(reject)?;
    let wide_source = SOURCE.replace("workgroup_size(1)", &format!("workgroup_size({too_wide})"));
    let wide = module(device, &wide_source);
    cases.push(compare(
        device,
        &descriptor(&wide, Some(&storage), "main"),
        false,
        "actual_device_workgroup_limit",
    )?);

    let mut options = descriptor(&good, Some(&storage), "main");
    options.compilation_options.zero_initialize_workgroup_memory = true;
    if device
        .kuno_prepare_native_compute_pipeline(&options)
        .is_ok()
    {
        return Err(reject());
    }
    cases.push(json!({"case":"unsupported_native_zero_initialization",
        "rejected":true,"ordinary_semantics_not_substituted":true}));
    options.compilation_options.zero_initialize_workgroup_memory = false;
    options.compilation_options.constants = &[("not_native_override", 1.)];
    if device
        .kuno_prepare_native_compute_pipeline(&options)
        .is_ok()
    {
        return Err(reject());
    }
    cases.push(json!({"case":"unsupported_native_pipeline_constants",
        "rejected":true,"ordinary_semantics_not_substituted":true}));

    let raw_source = "@compute @workgroup_size(1) fn main() {}";
    let parsed = wgpu::naga::front::wgsl::parse_str(raw_source).map_err(|_| reject())?;
    let info = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&parsed)
    .map_err(|_| reject())?;
    let words = wgpu::naga::back::spv::write_vec(
        &parsed,
        &info,
        &wgpu::naga::back::spv::Options::default(),
        None,
    )
    .map_err(|_| reject())?;
    // SAFETY: the complete compiled-public module was parsed, validated and
    // lowered above; no physical addresses, bindings or private data exist.
    let raw = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("Public native prepared passthrough control"),
            spirv: Some(Cow::Owned(words)),
            ..Default::default()
        })
    };
    cases.push(compare(
        device,
        &descriptor(&raw, None, "main"),
        false,
        "original_passthrough_cannot_derive_layout",
    )?);
    cases.push(compare(
        device,
        &descriptor(&raw, Some(&missing), "main"),
        true,
        "original_passthrough_explicit_layout",
    )?);
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::SOURCE;
    #[test]
    fn actual_public_admission_fixture_has_one_genuine_readonly_storage_binding() {
        let module = wgpu::naga::front::wgsl::parse_str(SOURCE).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].workgroup_size, [1, 1, 1]);
        let globals: Vec<_> = module.global_variables.iter().collect();
        assert_eq!(globals.len(), 1);
        let (handle, variable) = globals[0];
        assert_eq!(
            variable.binding.unwrap(),
            wgpu::naga::ResourceBinding {
                group: 0,
                binding: 0
            }
        );
        assert!(matches!(
            variable.space,
            wgpu::naga::AddressSpace::Storage {
                access: wgpu::naga::StorageAccess::LOAD
            }
        ));
        assert!(info.get_entry_point(0)[handle].contains(wgpu::naga::valid::GlobalUse::READ));
    }
    #[test]
    fn oversized_workgroup_fixture_is_valid_ir_before_actual_device_limit_admission() {
        let source = SOURCE.replace("workgroup_size(1)", "workgroup_size(1025)");
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [1025, 1, 1]);
    }
}
