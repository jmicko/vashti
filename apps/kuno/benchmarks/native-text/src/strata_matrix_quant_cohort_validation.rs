//! Genuine additional offline compiler proofs, independent of all old layouts.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{Compiler, WgpuCompilationOptions};
use std::io::Write;
pub(crate) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
        for (name, definition) in definitions() {
            let tile = if name.ends_with("tile8") {
                8
            } else if name.ends_with("tile16") {
                16
            } else {
                32
            };
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
                return Err("quant cohort actual SPIRV compiler".into());
            };
            let expected = tile * 512 + 8192 + 2048;
            if kernel.shared_size != expected {
                return Err(format!(
                    "quant cohort real SLM {name} mask{mask}: {} != {expected}",
                    kernel.shared_size
                )
                .into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("quant cohort real SG16".into());
            }
            let mut at = 5;
            let mut products = 0;
            let mut fault = false;
            let mut subgroup_barrier = false;
            let mut actual_wg64 = false;
            let mut half_types = std::collections::HashSet::new();
            let mut float_narrow = false;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                let op = words[at] & 65535;
                if n == 0 || at + n > words.len() {
                    return Err("quant cohort SPIRV extent".into());
                }
                if op == 16 && n == 6 && words[at + 2] == 17 {
                    actual_wg64 = words[at + 3..at + 6] == [64, 1, 1];
                }
                if op == 22 && n == 3 && words[at + 2] == 16 {
                    half_types.insert(words[at + 1]);
                }
                if op == 23 && n == 4 && half_types.contains(&words[at + 2]) {
                    half_types.insert(words[at + 1]);
                }
                if op == 115 && n >= 4 && half_types.contains(&words[at + 1]) {
                    float_narrow = true;
                }
                products += usize::from(op == 4459);
                fault |= op == 241;
                if op == 224 {
                    subgroup_barrier = true;
                }
                at += n;
            }
            if products != 2 * (tile / 8)
                || !fault
                || !subgroup_barrier
                || !actual_wg64
                || float_narrow
            {
                return Err(format!("quant cohort real product/fault/barrier/exact-bit-half {name}: p{products} f{fault} b{subgroup_barrier} narrowing{float_narrow}").into());
            }
            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut validator = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            validator
                .stdin
                .take()
                .ok_or("quant cohort validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !validator.wait()?.success() {
                return Err("quant cohort SPIRV validator".into());
            }
            println!(
                "strata_quant_cohort_shader_accepted name={name} layout={mask} shared_bytes={} subgroup=16 exact_bit_half=true",
                kernel.shared_size
            );
        }
    }
    Ok(())
}
