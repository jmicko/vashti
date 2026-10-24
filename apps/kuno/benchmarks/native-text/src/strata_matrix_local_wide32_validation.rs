use super::*;
pub(in crate::strata_matrix) fn validate_program(
    w: &[u32],
    expert: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    crate::strata_matrix::local_validation::validate_program(w, expert, None)
}
#[cfg(test)]
pub(in crate::strata_matrix) fn validate_negatives(
    w: &[u32],
    expert: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    crate::strata_matrix::local_validation::negatives(w, expert, None)
}
pub(in crate::strata_matrix) fn validate_sources()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in [
            ("fp16", local_wide32_definition::<f16>(false)),
            ("bf16", local_wide32_definition::<bf16>(false)),
            ("gr_bf16", local_wide32_definition::<bf16>(true)),
        ] {
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
                return Err("global wide32 actual SPIRV".into());
            };
            if k.shared_size != 0 {
                return Err("global wide32 actual zero shared bytes".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("global wide32 actual SG16".into());
            }
            if let Ok(dir) = std::env::var("KUNO_PUBLIC_LOCAL_PREFETCH_MODULE_DIR") {
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("local-dense-{name}-{mask}.spv")),
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            validate_program(&words, false)?;
            #[cfg(test)]
            validate_negatives(&words, false)?;

            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("global wide32 validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("global wide32 SPIRV validation".into());
            }
            println!(
                "strata_local_wide32_shader_accepted dtype={name} layout_mask={mask} shared_bytes=0 workgroup=256 subgroup=16 m=128 n=128 k=16 products=8 global_operand_loads=6 alternating_two_a=true global_stores=8 whole_workgroup_handoffs=0"
            );
        }
    }
    Ok(())
}
