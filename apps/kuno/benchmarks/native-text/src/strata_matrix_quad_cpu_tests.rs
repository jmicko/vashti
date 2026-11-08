use super::*;
#[test]
fn quad_regular_grid_covers_every_original_output_once_without_extra_scratch() {
    for members in 32usize..256 {
        for rows in [1280usize, 2560] {
            let mut seen = vec![0u8; members * rows];
            for y in 0..members.div_ceil(16) {
                for x in 0..rows / 64 {
                    for subgroup in 0..2 {
                        for cell in 0..4 {
                            for member in 0..16 {
                                for row in 0..8 {
                                    let t = y * 16 + member;
                                    let r = x * 64 + subgroup * 32 + cell * 8 + row;
                                    if t < members {
                                        seen[t * rows + r] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().all(|v| *v == 1));
        }
    }
    let old = include_str!("strata_matrix_dual_regular_cohort.rs");
    let new = include_str!("strata_matrix_quad_regular_cohort.rs");
    let guards = |s: &str| {
        s[s.find("    let slot =").unwrap()..s.find("    crate::strata_matrix::").unwrap()]
            .to_owned()
    };
    assert_eq!(guards(new), guards(old));
    let p = include_str!("strata_matrix_quad_products.rs");
    assert!(p.contains("SharedMemory::<f32>::new(256usize)"));
    for cell in ["c0", "c1", "c2", "c3"] {
        assert!(p.contains(&format!("let {cell} = cmma::Matrix")));
    }
}

#[test]
fn quad_regular_all16_actual_modules_prove_full_banks_and_original_c_order() {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{
        Compiler, WgpuCompilationOptions,
        ir::{ElemType, StorageType, UIntKind},
    };
    for mask in 0u8..8 {
        for (name, definition) in cohorts::quad_regular::definitions() {
            let mut compiler = AutoCompiler::SpirV(Default::default());
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let compiled = Compiler::compile(
                &mut compiler,
                definition,
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap();
            let AutoRepresentation::SpirV(kernel) = compiled else {
                panic!("actual native SPIR-V required")
            };
            assert_eq!(kernel.shared_size, 1024);
            let words =
                wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module).unwrap();
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)
                .unwrap()
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
                .unwrap()
                .unwrap_or(words);
            assert_eq!(
                wgpu::hal::vulkan::kuno_cooperative_subgroup(&words),
                Some(16)
            );
            let shape = Some((
                16usize,
                true,
                if name.ends_with("_down") { 640 } else { 2560 },
            ));
            quad_validation::validate_program(&words, true, shape).unwrap();
            quad_validation::negatives(&words, true, shape).unwrap();
        }
    }
}
