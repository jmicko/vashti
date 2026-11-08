//! Additive genuine compiler controls; never substitute for original layouts.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{Compiler, WgpuCompilationOptions};
use std::io::Write;

fn shared_bytes(name: &str, explicit: bool) -> Option<usize> {
    if name.contains("cohort") || name.contains("global_narrow") {
        Some(2048)
    } else if name.contains("macro64") {
        Some(if explicit { 32768 } else { 36864 })
    } else if name.contains("macro") || name.contains("tail32") {
        Some(if explicit { 24576 } else { 26624 })
    } else if name.contains("narrow8") || name.contains("tail8") {
        Some(if explicit { 18432 } else { 20480 })
    } else if name.contains("narrow16") || name.contains("tail16") {
        Some(if explicit { 20480 } else { 22528 })
    } else {
        None
    }
}
pub(crate) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
        let definitions = super::definitions();
        #[cfg(feature = "strata-expert-cohorts")]
        let definitions = definitions
            .into_iter()
            .chain(crate::strata_matrix::cohorts::small_direct::definitions());
        for (name, definition) in definitions {
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
                StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("small direct actual native compiler representation".into());
            };
            let expected =
                shared_bytes(name, mask & 4 != 0).ok_or("small direct missing real SLM quote")?;
            if kernel.shared_size != expected {
                return Err(format!(
                    "small direct actual shared layout {name} mask{mask}: {} != {expected}",
                    kernel.shared_size
                )
                .into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("small direct actual subgroup16".into());
            }
            let mut at = 5;
            let mut product = false;
            let mut fault = false;
            let mut direct_store = false;
            let mut checked_store = false;
            let mut pointer_types = std::collections::HashMap::new();
            let mut pointer_values = std::collections::HashMap::new();
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if n == 0 || at + n > words.len() {
                    return Err("small direct SPIRV extent".into());
                }
                let op = words[at] & 65535;
                if op == 32 && n == 4 {
                    pointer_types.insert(words[at + 1], words[at + 2]);
                }
                if matches!(op, 55 | 59 | 65 | 66 | 67 | 70 | 83 | 169 | 245) && n >= 3 {
                    if let Some(&class) = pointer_types.get(&words[at + 1]) {
                        pointer_values.insert(words[at + 2], class);
                    }
                }
                if op == 4458 && n >= 2 {
                    match pointer_values.get(&words[at + 1]) {
                        Some(12) | Some(5349) => direct_store = true,
                        Some(4) => checked_store = true,
                        _ => {}
                    }
                }
                if op == 4459 {
                    product = true;
                }
                if op == 241 {
                    fault = true;
                }
                at += n;
            }
            if !product || !fault || !direct_store || !checked_store {
                return Err(format!("small direct missing actual math/global/checked/fault: {name} p{product} f{fault} g{direct_store} s{checked_store}").into());
            }
            let mut command = std::process::Command::new("spirv-val");
            command.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                command.arg("--uniform-buffer-standard-layout");
            }
            let mut validator = command.arg("-")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            validator
                .stdin
                .take()
                .ok_or("small direct validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !validator.wait()?.success() {
                return Err("small direct SPIRV validation".into());
            }
            println!(
                "strata_small_direct_shader_accepted name={name} layout={mask} shared_bytes={} direct_global_store=true checked_scratch_store=true subgroup=16",
                kernel.shared_size
            );
        }
    }
    Ok(())
}
