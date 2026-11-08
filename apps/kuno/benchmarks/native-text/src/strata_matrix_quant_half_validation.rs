//! Same genuine native conversion/CMMA compiler path for the public edge proof.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{Compiler, WgpuCompilationOptions};
use std::io::Write;

pub(crate) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
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
            super::half_probe::definition(),
            &options,
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )?
        else {
            return Err("quant half edge actual SPIRV compiler".into());
        };
        // Four SGs own A8x16 plus B16x16 FP16 operands. The observable
        // accumulator is stored to its genuine global witness owner.
        if kernel.shared_size != 3072 || kernel.bindings.len() != 3 {
            return Err(format!(
                "quant half edge actual operand shared/bindings layout{mask}: shared={} bindings={}",
                kernel.shared_size,
                kernel.bindings.len()
            )
            .into());
        }
        let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
        let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
            .ok_or("quant half edge actual subgroup shared fence")?;
        let words = wgpu::hal::vulkan::kuno_quant_half_pack(&words)?
            .ok_or("quant half edge actual native conversion lowering")?;
        let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?
            .ok_or("quant half edge actual cooperative FP16 preservation")?;
        if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
            return Err("quant half edge actual SG16".into());
        }
        let mut glsl_sets = std::collections::HashSet::new();
        let mut pack = 0;
        let mut unpack = 0;
        let mut products = 0;
        let mut barriers = 0;
        let mut preserve16 = false;
        let mut actual_wg64 = false;
        let mut entry = None;
        let mut at = 5;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            let op = words[at] & 65535;
            if n == 0 || at + n > words.len() {
                return Err("quant half edge SPIRV extent".into());
            }
            if op == 15 && n >= 4 {
                if entry.replace(words[at + 2]).is_some() || words[at + 1] != 5 {
                    return Err("quant half edge single actual compute entry".into());
                }
            }
            if op == 16 && n == 6 && words[at + 2] == 17 {
                actual_wg64 = words[at + 3..at + 6] == [64, 1, 1];
            }
            if op == 16 && n == 4 && Some(words[at + 1]) == entry {
                preserve16 |= words[at + 2..at + 4] == [4459, 16];
            }
            if op == 11 && n >= 3 {
                let name = words[at + 2..at + n]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .take_while(|v| *v != 0)
                    .collect::<Vec<_>>();
                if name == b"GLSL.std.450" {
                    glsl_sets.insert(words[at + 1]);
                }
            }
            if op == 12 && n == 6 && glsl_sets.contains(&words[at + 3]) {
                pack += usize::from(words[at + 4] == 58);
                unpack += usize::from(words[at + 4] == 62);
            }
            if op == 115 {
                return Err("quant half edge unlowered FConvert".into());
            }
            products += usize::from(op == 4459);
            barriers += usize::from(op == 224);
            at += n;
        }
        if !actual_wg64 || !preserve16 || products != 1 || barriers < 2 || pack == 0 || unpack == 0
        {
            return Err(format!(
                "quant half edge actual native instructions layout{mask}: wg{actual_wg64} preserve{preserve16} products{products} barriers{barriers} pack{pack} unpack{unpack}"
            )
            .into());
        }
        let mut command = std::process::Command::new("spirv-val");
        command.args(["--target-env", "vulkan1.3"]);
        if mask & 1 != 0 {
            command.arg("--uniform-buffer-standard-layout");
        }
        let mut validator = command
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        validator
            .stdin
            .take()
            .ok_or("quant half edge validator stdin")?
            .write_all(
                &words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        if !validator.wait()?.success() {
            return Err("quant half edge actual SPIRV validator".into());
        }
        println!(
            "strata_quant_half_edge_shader_accepted layout={mask} data_bindings=3 workgroup=64 subgroup=16 pack_half_2x16={pack} unpack_half_2x16={unpack} raw_half_narrowing=0"
        );
    }
    Ok(())
}
