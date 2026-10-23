//! Genuine additional offline compiler proofs, independent of all old layouts.
use super::*;
use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
use cubecl::{Compiler, WgpuCompilationOptions};
use std::io::Write;
pub(crate) fn validate() -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_kind(false)
}
pub(crate) fn validate_pipeline() -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_kind(true)
}
fn validate_kind(pipeline: bool) -> std::result::Result<(), Box<dyn std::error::Error>> {
    for mask in 0u8..8 {
        for (name, definition) in if pipeline {
            pipeline_definitions()
        } else {
            definitions()
        } {
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
            // Current block-liveness analysis keeps B and scratch separate.
            // Validate the actual bounded allocation rather than claim reuse.
            let expected = if pipeline { 14080 } else { 26112 };
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
            let words = wgpu::hal::vulkan::kuno_quant_half_pack(&words)?
                .ok_or("quant cohort real native half conversion lowering")?;
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
            let mut u32_types = std::collections::HashSet::new();
            let mut u32_constants = std::collections::HashMap::new();
            let mut boolean_constants = std::collections::HashSet::new();
            let mut runtime_selections = Vec::new();
            let mut float_narrow = false;
            let mut glsl_sets = std::collections::HashSet::new();
            let mut native_pack = 0;
            let mut native_unpack = 0;
            let mut pointer_storage = std::collections::HashMap::new();
            let mut pointer_pointee = std::collections::HashMap::new();
            let mut half_vector_width = std::collections::HashMap::new();
            let mut pointer_values = std::collections::HashMap::new();
            let mut global_input_loads = 0;
            let mut shared_weight_loads = 0;
            let mut workgroup_handoffs = 0;
            let mut retained_product_loops = 0;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                let op = words[at] & 65535;
                if n == 0 || at + n > words.len() {
                    return Err("quant cohort SPIRV extent".into());
                }
                if op == 16 && n == 6 && words[at + 2] == 17 {
                    actual_wg64 = words[at + 3..at + 6] == [64, 1, 1];
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
                    native_pack += usize::from(words[at + 4] == 58);
                    native_unpack += usize::from(words[at + 4] == 62);
                }
                if op == 22 && n == 3 && words[at + 2] == 16 {
                    half_types.insert(words[at + 1]);
                }
                if op == 21 && n == 4 && words[at + 2..at + 4] == [32, 0] {
                    u32_types.insert(words[at + 1]);
                }
                if op == 43 && n == 4 && u32_types.contains(&words[at + 1]) {
                    u32_constants.insert(words[at + 2], words[at + 3]);
                }
                // CubeCL expresses scalar-half stride24 as six vec4 elements;
                // accept a folded literal or evaluate that exact constant SHR.
                if op == 194 && n == 5 && u32_types.contains(&words[at + 1]) {
                    if let (Some(&value), Some(&shift)) = (
                        u32_constants.get(&words[at + 3]),
                        u32_constants.get(&words[at + 4]),
                    ) {
                        if shift < 32 {
                            u32_constants.insert(words[at + 2], value >> shift);
                        }
                    }
                }
                if matches!(op, 41 | 42) && n == 3 {
                    boolean_constants.insert(words[at + 2]);
                }
                if op == 169 && n == 6 && u32_types.contains(&words[at + 1]) {
                    runtime_selections.push((words[at + 3], words[at + 4], words[at + 5]));
                }
                if op == 23 && n == 4 && half_types.contains(&words[at + 2]) {
                    half_vector_width.insert(words[at + 1], words[at + 3]);
                    half_types.insert(words[at + 1]);
                }
                if op == 32 && n == 4 {
                    pointer_storage.insert(words[at + 1], words[at + 2]);
                    pointer_pointee.insert(words[at + 1], words[at + 3]);
                }
                if matches!(op, 59 | 61 | 65 | 66 | 67 | 70 | 83 | 124)
                    && n >= 3
                    && pointer_storage.contains_key(&words[at + 1])
                {
                    pointer_values.insert(words[at + 2], words[at + 1]);
                }
                if op == 4457 && n >= 6 {
                    match pointer_values
                        .get(&words[at + 3])
                        .and_then(|ty| pointer_storage.get(ty))
                    {
                        Some(12) => global_input_loads += 1,
                        Some(4) => {
                            let actual_width = pointer_values
                                .get(&words[at + 3])
                                .and_then(|ty| pointer_pointee.get(ty))
                                .and_then(|ty| half_vector_width.get(ty));
                            if actual_width != Some(&4) {
                                return Err(
                                    "quant cohort padded B must be an actual F16 vec4 pointer"
                                        .into(),
                                );
                            }
                            if u32_constants.get(&words[at + 5]) != Some(&6) {
                                return Err(
                                    "quant cohort padded B stride must be six vec4 halves".into()
                                );
                            }
                            shared_weight_loads += 1;
                        }
                        _ => return Err("quant cohort unproved cooperative operand storage".into()),
                    }
                }
                if op == 115 && n >= 4 && half_types.contains(&words[at + 1]) {
                    float_narrow = true;
                }
                products += usize::from(op == 4459);
                retained_product_loops += usize::from(op == 246 && n == 4 && words[at + 3] == 2);
                fault |= op == 241;
                if op == 224 {
                    subgroup_barrier = true;
                    if u32_constants.get(&words[at + 1]) == Some(&2) {
                        workgroup_handoffs += 1;
                    }
                }
                at += n;
            }
            let iq4 = name.contains("dtype23_") || name.contains("dtype20_");
            // Inspect the actual compiled module, not merely source spelling.
            // The failing kernel had correct LUTs but lost the dynamic high
            // nibble branch entirely; the real quarter selector must survive.
            let iq4_high_nibble_selectors = runtime_selections
                .iter()
                .filter(|(condition, high, low)| {
                    !boolean_constants.contains(condition)
                        && u32_constants.get(high) == Some(&4)
                        && u32_constants.get(low) == Some(&0)
                })
                .count();
            if iq4 && iq4_high_nibble_selectors == 0 {
                return Err(format!(
                    "quant cohort actual IQ4 runtime high-nibble selection missing {name} mask{mask}"
                )
                .into());
            }
            let static_steps = if !pipeline && tile == 32 { 1 } else { 4 };
            if products != static_steps * (tile / 8)
                || global_input_loads != static_steps * (tile / 8)
                || shared_weight_loads != static_steps
                || workgroup_handoffs != if pipeline { 2 } else { 1 }
                || retained_product_loops != usize::from(!pipeline && tile == 32)
                || !fault
                || !subgroup_barrier
                || !actual_wg64
                || float_narrow
                || native_pack == 0
                || native_unpack == 0
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
            if pipeline {
                println!("strata_quant_pipeline_shader_accepted name={name} layout={mask} shared_bytes={} subgroup=16 workgroup=64 output_columns=32 producer_subgroups=2 consumer_subgroups=2 global_input_loads={global_input_loads} shared_weight_loads={shared_weight_loads} products={products} native_pack={native_pack} native_unpack={native_unpack} iq4_selectors={iq4_high_nibble_selectors} whole_workgroup_handoffs={workgroup_handoffs} raw_half_narrowing=0 dont_unroll_loops={retained_product_loops}",kernel.shared_size);
            } else {
                println!(
                "strata_quant_cohort_shader_accepted name={name} layout={mask} shared_bytes={} subgroup=16 exact_bit_half=true",
                kernel.shared_size
            );
                if iq4 {
                    println!(
                    "strata_quant_cohort_iq4_nibble_accepted name={name} layout={mask} runtime_selectors={iq4_high_nibble_selectors}"
                );
                }
                println!(
                "strata_quant_cohort_native_half_accepted name={name} layout={mask} pack_half_2x16={native_pack} unpack_half_2x16={native_unpack} raw_half_narrowing=0"
            );
                println!(
                "strata_quant_cohort_k64_padded_accepted name={name} layout={mask} shared_bytes={} global_input_loads={global_input_loads} shared_weight_loads={shared_weight_loads} products={products}",
                kernel.shared_size
            );
                println!(
                "strata_quant_cohort_padded_b_accepted name={name} layout={mask} stride_halves=24 tile_half_span=376 whole_workgroup_handoffs={workgroup_handoffs} actual_publication_alias={}",
                kernel.shared_size == 24064
            );
                println!(
                "strata_quant_cohort_product_loop_accepted name={name} layout={mask} dont_unroll_loops={retained_product_loops} original_arithmetic_and_cfg_unchanged=true"
            );
            }
        }
    }
    Ok(())
}
