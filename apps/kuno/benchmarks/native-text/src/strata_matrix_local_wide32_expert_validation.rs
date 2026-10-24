use super::*;
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in definitions() {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(k) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition,
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("global wide32 expert actual SPIRV".into());
            };
            if k.shared_size != 8192 {
                return Err("global wide32 expert only bounded8KiB publication scratch".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("global wide32 expert actual SG16".into());
            }
            if let Ok(dir) = std::env::var("KUNO_PUBLIC_LOCAL_PREFETCH_MODULE_DIR") {
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("local-expert-{name}-{mask}.spv")),
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            crate::strata_matrix::native_wide::local_wide32::validation::validate_program(
                &words, true,
            )?;
            #[cfg(test)]
            crate::strata_matrix::native_wide::local_wide32::validation::validate_negatives(
                &words, true,
            )?;
            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("global wide32 expert validator input")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("global wide32 expert SPIRV validation".into());
            }
            println!(
                "strata_local_wide32_expert_shader_accepted name={name} layout={mask} shared_bytes=8192 operand_shared_bytes=0 workgroup=256 subgroup=16 global_operand_loads=15 products=12 original_publication_and_finite_consumers=true"
            );
        }
    }
    validate_guard_sources()?;
    // Original unchanged fixture validation is retained in the old variant.
    Ok(())
}

pub(super) fn validate_guard_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        let mut b = KernelBuilder::default();
        cubecl::ir::AddressType::U32.register(&mut b.scope);
        let map = b.input_array(u32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let c = b.scalar(u32::as_type(&b.scope).storage_type());
        let g = b.scalar(u32::as_type(&b.scope).storage_type());
        route_guard::expand(&mut b.scope, map.into(), fault.into(), c.into(), g.into());
        assert!(b.scope.pop_errors().is_empty());
        let def = b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(256)));
        let mut o = WgpuCompilationOptions {
            supports_vulkan: true,
            ..Default::default()
        };
        o.vulkan.max_spirv_version = (1, 6);
        o.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
        o.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
        o.vulkan.supports_explicit_smem = mask & 4 != 0;
        let AutoRepresentation::SpirV(k) = Compiler::compile(
            &mut AutoCompiler::SpirV(Default::default()),
            def,
            &o,
            ExecutionMode::Unchecked,
            cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
        )?
        else {
            return Err("wide32 guard SPIRV".into());
        };
        if k.shared_size != 0 {
            return Err("wide32 guard zero scratch".into());
        }
        let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
        let (mut at, mut size, mut matrices, mut atomics) = (5, false, 0, 0);
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if n == 0 || at + n > words.len() {
                return Err("wide32 guard opcode extent".into());
            }
            let op = words[at] & 65535;
            if op == 16 && n == 6 && words[at + 2] == 17 {
                size = words[at + 3..at + 6] == [256, 1, 1];
            }
            if matches!(op, 4456..=4459) {
                matrices += 1;
            }
            if (227..=242).contains(&op) {
                atomics += 1;
            }
            at += n;
        }
        if !size || matrices != 0 || atomics == 0 {
            return Err("wide32 genuine guard shader".into());
        }
        let mut cmd = std::process::Command::new("spirv-val");
        cmd.args(["--target-env", "vulkan1.3"]);
        if mask & 1 != 0 {
            cmd.arg("--uniform-buffer-standard-layout");
        }
        let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
        child.stdin.take().unwrap().write_all(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        if !child.wait()?.success() {
            return Err("wide32 guard validated SPIRV".into());
        }
        println!(
            "strata_local_wide32_guard_shader_accepted layout={mask} shared_bytes=0 workgroup=256 no_matrix=true"
        );
    }
    Ok(())
}
