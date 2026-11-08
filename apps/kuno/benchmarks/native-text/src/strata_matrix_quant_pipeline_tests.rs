use super::*;
use cubecl::{Compiler, WgpuCompilationOptions};

#[test]
fn pipeline_roles_publish_every_n32_row_and_keep_common_double_bank_handoffs() {
    for rows in [1280usize, 2560] {
        let mut coverage = vec![false; rows];
        for block in 0..rows / 32 {
            for consumer in 2..4usize {
                for row in 0..16 {
                    let address = block * 32 + (consumer - 2) * 16 + row;
                    assert!(!coverage[address]);
                    coverage[address] = true;
                }
            }
        }
        assert!(coverage.into_iter().all(|v| v));
    }
    for columns in [640usize, 2560] {
        let mut bank = [Some(0usize), None];
        let mut current = 0;
        let mut barriers = [vec![0usize], vec![0usize], vec![0usize], vec![0usize]];
        let mut products = [Vec::new(), Vec::new()];
        for (iteration, k) in (0..columns).step_by(32).enumerate() {
            assert_eq!(bank[current], Some(k));
            let next = 1 - current;
            if k + 32 < columns {
                bank[next] = Some(k + 32);
            }
            // Writing next cannot overwrite the panel still consumed now.
            assert_eq!(bank[current], Some(k));
            for consumer in &mut products {
                for step in 0..2 {
                    consumer.push(k + step * 16);
                }
            }
            for trace in &mut barriers {
                trace.push(iteration + 1);
            }
            current = next;
        }
        assert!(barriers.windows(2).all(|w| w[0] == w[1]));
        for product in products {
            assert_eq!(product, (0..columns).step_by(16).collect::<Vec<_>>());
        }
    }
    // Independent exact data/padding ownership for all four K32 banks.
    for sg in 0..2usize {
        for bank in 0..2usize {
            let first_bank = (sg * 2 + bank) * 188;
            let mut coverage = std::collections::BTreeSet::new();
            for round in 0..4usize {
                for lane in 0..16usize {
                    let row = round * 4 + lane / 4;
                    let col = lane % 4 * 8;
                    let first = first_bank + col / 16 * 94 + row * 6 + (col % 16) / 4;
                    assert!(coverage.insert(first));
                    assert!(coverage.insert(first + 1));
                    if col % 16 == 0 && row < 15 {
                        assert!(coverage.insert(first + 4));
                        assert!(coverage.insert(first + 5));
                    }
                }
            }
            assert_eq!(coverage, (first_bank..first_bank + 188).collect());
        }
    }
    assert_eq!(2 * 2 * 188 * 8 + 4 * 128 * 4, 8064);
    let source = include_str!("strata_matrix_quant_pipeline_kernel.rs");
    assert_eq!(source.matches("sync_cube()").count(), 2);
    assert!(!source.contains("b.free()"));
    let steady = source.split_once("while k < columns").unwrap().1;
    assert!(steady.find("k + 32").unwrap() < steady.find("cmma::fill").unwrap());
    assert!(steady.find("cmma::fill").unwrap() < steady.find("} else {").unwrap());
}

#[test]
fn pipeline_actual_modules_keep_exact_half_sg16_wg64_and_original_entry_boundaries() {
    let options = WgpuCompilationOptions {
        supports_vulkan: true,
        ..Default::default()
    };
    for dtype in [18u32, 21, 22, 23, 20, 42] {
        for tile in [8usize, 16, 32] {
            let mut options = options.clone();
            options.vulkan.max_spirv_version = (1, 6);
            let cubecl::wgpu::AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut cubecl::wgpu::AutoCompiler::SpirV(Default::default()),
                definition_kind(matches!(dtype, 20 | 42), tile, dtype, true),
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap() else {
                panic!("actual pipeline SPIRV")
            };
            assert_eq!(kernel.shared_size, 8064);
            let original =
                wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module).unwrap();
            let native = wgpu::hal::vulkan::kuno_quant_half_pack(&original)
                .unwrap()
                .unwrap();
            assert_eq!(
                wgpu::hal::vulkan::kuno_cooperative_subgroup(&native),
                Some(16)
            );
            assert_eq!(
                wgpu::hal::vulkan::kuno_quant_half_pack(&native)
                    .unwrap()
                    .unwrap(),
                native
            );
            let mut wrong = original.clone();
            let mut at = 5;
            let mut products = 0;
            let mut retained_loops = 0;
            while at < native.len() {
                let n = (native[at] >> 16) as usize;
                let op = native[at] & 65535;
                products += usize::from(op == 4459);
                retained_loops += usize::from(op == 246 && n == 4 && native[at + 3] == 2);
                at += n;
            }
            assert_eq!(products, (if tile == 32 { 1 } else { 2 }) * (tile / 8));
            assert_eq!(retained_loops, usize::from(tile == 32));
            at = 5;
            while at < wrong.len() {
                let n = (wrong[at] >> 16) as usize;
                if wrong[at] & 65535 == 16 && n == 6 && wrong[at + 2] == 17 {
                    wrong[at + 3] = 128;
                    break;
                }
                at += n;
            }
            assert!(wgpu::hal::vulkan::kuno_quant_half_pack(&wrong).is_err());
        }
    }
}
