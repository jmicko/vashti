use super::*;

#[test]
fn deep_macro_authentic_geometry_requires_real_explicit_layout_and_full_64_kib() {
    let bindings = [(80 * 1024 * 1024, 256); 2];
    for count in 0..=4096 {
        for (rows, cols) in [(1280, 2560), (2560, 640)] {
            for (shared, explicit) in [
                (49152, false),
                (65535, true),
                (65536, false),
                (65536, true),
                (81920, true),
            ] {
                assert_eq!(
                    geometry(count, rows, cols, 4, shared, bindings, explicit),
                    (64..256).contains(&count) && shared >= 65536 && explicit
                );
            }
        }
    }
    for (rows, cols, first) in [
        (1279, 2560, 4),
        (1280, 2559, 4),
        (2560, 641, 4),
        (1280, 2560, 1),
    ] {
        assert!(!geometry(129, rows, cols, first, 65536, bindings, true));
    }
    for bad in [(0, 256), (7, 256), (8, 2)] {
        for role in 0..2 {
            let mut b = bindings;
            b[role] = bad;
            assert!(!geometry(129, 1280, 2560, 4, 65536, b, true));
        }
    }
    assert_eq!(OPERAND_BYTES, 2 * (64 + 64) * 128 * 2);
    assert_eq!(LOGICAL_BYTES, OPERAND_BYTES + 64 * 64 * 4);
}

#[test]
fn deep_macro_stage_jobs_mask_only_original_m_tail_and_cover_every_actual_vector() {
    for count in [64usize, 65, 127, 128, 129, 255] {
        for cols in [640usize, 2560] {
            assert_eq!(cols % 128, 0);
            for base in (0..count).step_by(64) {
                for k in (0..cols).step_by(128) {
                    for bank in 0..2 {
                        let mut a = vec![None; 4096];
                        let mut b = vec![None; 4096];
                        let mut owners = vec![0u8; 2048];
                        for tid in 0..128 {
                            for round in 0..8 {
                                let job = tid + 128 * round;
                                let row = job / 16;
                                let col = k + (job % 16) * 8;
                                for vector in 0..2 {
                                    let at = bank * 2048 + job * 2 + vector;
                                    owners[job * 2 + vector] += 1;
                                    if base + row < count {
                                        let source = ((base + row) * cols + col + vector * 4) / 4;
                                        assert!(source * 4 + 4 <= count * cols);
                                        a[at] = Some(source);
                                    }
                                    b[at] = Some((row * cols + col + vector * 4) / 4);
                                }
                            }
                        }
                        assert!(owners.iter().all(|&n| n == 1));
                        for sg in 0..8 {
                            for step in 0..8 {
                                for tile in 0..2 {
                                    let af = bank * 2048 + (sg / 2) * 512 + step * 4 + tile * 256;
                                    let bf = bank * 2048 + (sg % 2) * 1024 + step * 4 + tile * 512;
                                    for row in 0..8 {
                                        for vector in 0..4 {
                                            let token = base + (sg / 2) * 16 + tile * 8 + row;
                                            assert_eq!(
                                                a[af + row * 32 + vector],
                                                if token < count {
                                                    Some(
                                                        (token * cols + k + step * 16 + vector * 4)
                                                            / 4,
                                                    )
                                                } else {
                                                    None
                                                }
                                            );
                                        }
                                    }
                                    for row in 0..16 {
                                        for vector in 0..4 {
                                            let feature = (sg % 2) * 32 + tile * 16 + row;
                                            assert_eq!(
                                                b[bf + row * 32 + vector],
                                                Some(
                                                    (feature * cols + k + step * 16 + vector * 4)
                                                        / 4
                                                )
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn deep_macro_preserves_original_entry_guards_output_owners_and_chronological_k16() {
    let original = include_str!("strata_matrix_macro64.rs");
    let deep = include_str!("strata_matrix_macro64_deep.rs");
    let original = original
        .split_once("#[cube(launch_unchecked)]\n")
        .unwrap()
        .1
        .split_once("\nfn definition_expert_macro64")
        .unwrap()
        .0;
    let entry = deep
        .split_once("#[cube(launch_unchecked)]\n")
        .unwrap()
        .1
        .split_once("\npub(super) fn definition")
        .unwrap()
        .0;
    assert_eq!(entry, original);
    for cols in [640, 2560] {
        let old: Vec<_> = (0..cols)
            .step_by(64)
            .flat_map(|k| (0..4).map(move |s| k + s * 16))
            .collect();
        let new: Vec<_> = (0..cols)
            .step_by(128)
            .flat_map(|k| (0..8).map(move |s| k + s * 16))
            .collect();
        assert_eq!(old, new);
    }
    let initial = deep.find("    sync_cube();\n    let mut k").unwrap();
    let handoff = deep[initial..]
        .find("        sync_cube();\n        current = next;")
        .unwrap()
        + initial;
    let free_a = deep.find("        a.free();").unwrap();
    let free_b = deep.find("        b.free();").unwrap();
    let publication = deep
        .find("    let mut out = SharedMemory::<f32>::new(4096usize);")
        .unwrap();
    assert!(initial < handoff && handoff < free_a && free_a < free_b && free_b < publication);
    assert!(!deep[free_b..publication].contains("cmma::load"));
}

#[test]
fn deep_macro_actual_native_compiler_retains_old_family_and_all_sixteen_new_layouts() {
    let name = core::any::type_name::<expert_macro64::ExpertMacro64<WgpuRuntime>>();
    assert!(name.contains("::deep_macro64::expert_macro64::ExpertMacro64"));
    for role in [3, 5] {
        assert!(
            cubecl_runtime::server::native_expert_batch::native_strata_expert_operator(
                name, false, role
            )
        );
        assert!(
            !cubecl_runtime::server::native_expert_batch::native_strata_expert_operator(
                name, true, role
            )
        );
    }
    for down in [false, true] {
        for mask in 0..8 {
            compile(down, mask).unwrap();
        }
    }
}
