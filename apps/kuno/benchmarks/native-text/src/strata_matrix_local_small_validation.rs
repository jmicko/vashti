use super::*;
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    validate_sources_mode(false)
}
pub(crate) fn validate_sources_mode(
    carry: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8u8 {
        let defs = definitions_mode(carry)
            .into_iter()
            .map(|(n, d)| {
                (
                    n,
                    d,
                    false,
                    if n.contains("macro64") { 64usize } else { 32 },
                )
            })
            .chain(
                crate::strata_matrix::cohorts::local_small::definitions_mode(carry)
                    .into_iter()
                    .map(|(n, d)| (n, d, true, 16usize)),
            );
        for (name, def, cohort, tile) in defs {
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
                return Err("small full walk actual SPIRV".into());
            };
            // Final exact per-layout resource populations are frozen after these real
            // modules compile; never assume larger scratch than the preceding kernels.
            if k.shared_size as usize != tile * 128 {
                return Err(
                    format!("balanced actual publication scratch {}", k.shared_size).into(),
                );
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("small full walk authentic SG16".into());
            }
            #[cfg(test)]
            if let Some(dir) = std::env::var_os(if carry {
                "KUNO_PUBLIC_SMALL_LOCAL_CARRY_MODULE_DIR"
            } else {
                "KUNO_PUBLIC_LOCAL_PREFETCH_MODULE_DIR"
            }) {
                let path = std::path::PathBuf::from(dir);
                // Fixed compiler definitions only: no request, activation,
                // weight contents or GPU data is present in these modules.
                let file = path.join(format!("{name}-layout{mask}.spv"));
                let bytes = words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>();
                std::fs::write(file, bytes)?;
            }
            let shape = Some((
                tile,
                cohort,
                if name.ends_with("_down") { 640 } else { 2560 },
            ));
            if carry {
                #[cfg(feature = "strata-small-local-carry")]
                {
                    crate::strata_matrix::small_local_carry_validation::validate_program(
                        &words, true, shape,
                    )?;
                    #[cfg(test)]
                    crate::strata_matrix::small_local_carry_validation::negatives(
                        &words, true, shape,
                    )?;
                }
                #[cfg(not(feature = "strata-small-local-carry"))]
                return Err("small local carry feature disabled".into());
            } else {
                crate::strata_matrix::local_validation::validate_program(
                    &words,
                    true,
                    Some((
                        tile,
                        cohort,
                        if name.ends_with("_down") { 640 } else { 2560 },
                    )),
                )?;
                #[cfg(test)]
                crate::strata_matrix::local_validation::negatives(
                    &words,
                    true,
                    Some((
                        tile,
                        cohort,
                        if name.ends_with("_down") { 640 } else { 2560 },
                    )),
                )?;
            }
            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("small full walk validator input")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("small full walk validated actual module".into());
            }
            println!(
                "{}_shader_accepted name={name} layout={mask} shared_bytes={} workgroup={} subgroup=16 weights_first=true checked_original_halves=true",
                if carry {
                    "strata_small_local_carry"
                } else {
                    "strata_local_small"
                },
                k.shared_size,
                tile * 4
            );
        }
    }
    Ok(())
}
