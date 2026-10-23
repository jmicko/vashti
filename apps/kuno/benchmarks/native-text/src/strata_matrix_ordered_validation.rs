//! Separate compiler proof for actual ordered-output producers.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{
    Compiler, WgpuCompilationOptions,
    ir::{ElemType, StorageType, UIntKind},
};
use cubecl_runtime::kernel::KernelDefinition;
use std::io::Write;
pub(super) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = native::ordered::definitions();
    #[cfg(feature = "strata-native-expert-wide")]
    out.extend(native_expert_wide::ordered::definitions());
    #[cfg(all(feature = "strata-native-expert-wide", feature = "strata-deep-matrix"))]
    out.extend(native_expert_wide::ordered::deep::definitions());
    #[cfg(feature = "strata-expert-cohorts")]
    out.extend(cohorts::ordered::definitions());
    out
}
fn shared_bytes(name: &str, explicit: bool) -> Option<usize> {
    if name == "ordered_expert_down" {
        Some(1280)
    } else if name.contains("macro64_deep") || name.contains("wide64") {
        Some(if explicit { 65536 } else { 81920 })
    } else if name.contains("cohort") {
        Some(if name.contains("cohort8") { 2048 } else { 4096 })
    } else if name.contains("wide32") {
        Some(49152)
    } else if name.contains("tail32") {
        native::compiled_shared_bytes("expert_macro", explicit)
    } else if name.contains("tail8") {
        native::compiled_shared_bytes("staged_narrow8", explicit)
    } else if name.contains("tail16") {
        native::compiled_shared_bytes("staged_narrow16", explicit)
    } else {
        native::compiled_shared_bytes(name.strip_prefix("ordered_")?, explicit)
    }
}
pub(super) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
        for (name, definition) in definitions() {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition,
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )?
            else {
                return Err("ordered expert compiler representation".into());
            };
            if let Some(expected) = shared_bytes(name, mask & 4 != 0) {
                if kernel.shared_size != expected {
                    return Err(format!(
                        "ordered expert shared envelope {name} mask{mask}: {} != {expected}",
                        kernel.shared_size
                    )
                    .into());
                }
            } else {
                return Err("ordered expert missing shared envelope".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("ordered expert genuine subgroup16".into());
            }
            let mut at = 5;
            let mut store = false;
            let mut product = false;
            let mut fault = false;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
                    return Err("ordered expert instruction extent".into());
                }
                match words[at] & 65535 {
                    4458 => store = true,
                    4459 => product = true,
                    241 => fault = true,
                    _ => {}
                }
                at += n;
            }
            if !store || !product || !fault {
                return Err("ordered expert product/store/checked-fault absent".into());
            }
            #[cfg(feature = "strata-wide-register-stage")]
            if name.contains("wide32") {
                wide_pipeline::validate_program(&words)?;
            }
            let mut command = std::process::Command::new("spirv-val");
            command.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                command.arg("--uniform-buffer-standard-layout");
            }
            let mut child = command
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            child
                .stdin
                .take()
                .ok_or("ordered expert validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("ordered expert SPIRV validation failed".into());
            }
            println!(
                "strata_expert_ordered_shader_accepted name={name} subgroup=16 layout_mask={mask}"
            );
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "strata_matrix_ordered_tests.rs"]
mod tests;
