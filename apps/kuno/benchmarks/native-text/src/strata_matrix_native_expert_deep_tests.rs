use super::*;

#[test]
fn deep_expert_physical_capability_composes_original_complete_shape_and_workgroup_guards() {
    assert_eq!(
        (THREADS, SUBGROUPS, M_FRAGMENTS, SUBGROUP_ROWS),
        (512, 32, 2, 16)
    );
    assert_eq!(OPERAND_BYTES, 2 * (128 + 128) * 64 * 2);
    assert_eq!(LOGICAL_BYTES, OPERAND_BYTES + 32 * 128 * 4);
    for count in 0..=4096 {
        for (columns, rows, down) in [(2560, 1280, false), (640, 2560, true)] {
            for (shared, explicit) in [
                (49152, false),
                (65535, true),
                (65536, false),
                (65536, true),
                (81920, true),
            ] {
                let old = super::super::selected(
                    count,
                    columns,
                    rows,
                    0,
                    down,
                    shared,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                );
                assert_eq!(
                    old && geometry(shared, explicit),
                    (MIN_GROUP..=4096).contains(&count) && shared >= 65536 && explicit
                );
            }
        }
    }
    for (units, dim, pmin, pmax) in [
        (511, (512, 1, 1), 16, 32),
        (512, (511, 1, 1), 16, 32),
        (512, (512, 1, 1), 17, 32),
        (512, (512, 1, 1), 8, 15),
    ] {
        assert!(!super::super::selected(
            256, 2560, 1280, 0, false, 65536, units, dim, pmin, pmax
        ));
    }
}

#[test]
fn deep_expert_k16_panel_stage_and_both_prefetch_rounds_have_exact_original_vector_owners() {
    for count in [256usize, 257, 383, 4096] {
        let prefix = count / 128 * 128;
        for (cols, rows) in [(2560usize, 1280usize), (640, 2560)] {
            assert_eq!(cols % 64, 0);
            for token_base in [0, prefix - 128] {
                for row_base in [0, rows - 128] {
                    for k in [0, cols - 64] {
                        for first in [0usize, 4, 128] {
                            for bank in 0..2 {
                                let mut a = vec![None; 4096];
                                let mut b = vec![None; 4096];
                                let mut owners = vec![0u8; 2048];
                                for tid in 0..512 {
                                    for round in 0..2 {
                                        let job = tid + round * 512;
                                        let row = job / 8;
                                        let col = (job % 8) * 8;
                                        for vector in 0..2 {
                                            // Independent logical-panel coordinates.
                                            let at = bank * 2048
                                                + (col / 16) * 512
                                                + row * 4
                                                + (col % 16) / 4
                                                + vector;
                                            owners[at - bank * 2048] += 1;
                                            let aa =
                                                ((token_base + row) * cols + k + col + vector * 4)
                                                    / 4;
                                            let bb = (first
                                                + (row_base + row) * cols
                                                + k
                                                + col
                                                + vector * 4)
                                                / 4;
                                            assert!(aa * 4 + 4 <= count * cols);
                                            assert!(
                                                bb * 4 >= first
                                                    && bb * 4 + 4 <= first + rows * cols
                                            );
                                            a[at] = Some(aa);
                                            b[at] = Some(bb);
                                            // Actual subgroup/lane spelling used by
                                            // register prefetch covers the SAME jobs.
                                            let prefetch_job =
                                                (tid / 16) * 16 + tid % 16 + round * 512;
                                            let target = bank * 2048
                                                + (prefetch_job % 8 / 2) * 512
                                                + (prefetch_job / 8) * 4
                                                + (prefetch_job % 2) * 2
                                                + vector;
                                            assert_eq!(target, at);
                                        }
                                    }
                                }
                                assert!(owners.iter().all(|&n| n == 1));
                                for sg in 0..32 {
                                    for step in 0..4 {
                                        for tile in 0..2 {
                                            let af = bank * 2048
                                                + step * 512
                                                + (sg / 4) * 64
                                                + tile * 32;
                                            let bf = bank * 2048
                                                + step * 512
                                                + (sg % 4) * 128
                                                + tile * 64;
                                            for row in 0..8 {
                                                for vector in 0..4 {
                                                    let token =
                                                        token_base + (sg / 4) * 16 + tile * 8 + row;
                                                    assert_eq!(
                                                        a[af + row * 4 + vector],
                                                        Some(
                                                            (token * cols
                                                                + k
                                                                + step * 16
                                                                + vector * 4)
                                                                / 4
                                                        )
                                                    );
                                                }
                                            }
                                            for row in 0..16 {
                                                for vector in 0..4 {
                                                    let feature =
                                                        row_base + (sg % 4) * 32 + tile * 16 + row;
                                                    assert_eq!(
                                                        b[bf + row * 4 + vector],
                                                        Some(
                                                            (first
                                                                + feature * cols
                                                                + k
                                                                + step * 16
                                                                + vector * 4)
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
    }
}

#[test]
fn deep_expert_retains_original_header_publication_helpers_tail_and_k16_order() {
    let old = include_str!("strata_matrix_native_expert_wide.rs");
    let new = include_str!("strata_matrix_native_expert_deep.rs");
    fn guard(s: &str) -> Vec<&str> {
        s.split_once("    if !header_valid(map, count, group)")
            .unwrap()
            .1
            .split_once("    // Two typed")
            .unwrap()
            .0
            .split_whitespace()
            .collect()
    }
    let voted_header = "plane_all(header_valid(map, count, group))";
    assert_eq!(new.matches(voted_header).count(), 1);
    assert!(new.find(voted_header).unwrap() < new.find("let mut a =").unwrap());
    // Only normalize the vote over an already uniform, immutable header. All
    // original rejection terms and their failure/termination body stay exact.
    let normalized = new.replace(voted_header, "header_valid(map, count, group)");
    assert_eq!(guard(old), guard(&normalized));
    assert!(new.contains("for stage_index in 0usize..columns / 64usize"));
    assert!(new.contains("let k = stage_index * 64usize;"));
    assert!(!new.contains("while k < columns"));
    fn publication(s: &str) -> Vec<&str> {
        s.split_once("    if comptime!(direct_gu) {")
            .unwrap()
            .1
            .split_once("\n}\n\n#[cube")
            .unwrap()
            .0
            .split_whitespace()
            .collect()
    }
    assert_eq!(publication(old), publication(new));
    assert!(!new.contains("fn publish_routed") && !new.contains("fn publish_direct_gu"));
    assert!(!new.contains("fn tail(")); // retain the actual ORIGINAL tail below prefix
    for cols in [640, 2560] {
        let original: Vec<_> = (0..cols)
            .step_by(32)
            .flat_map(|k| (0..2).map(move |s| k + s * 16))
            .collect();
        let deep: Vec<_> = (0..cols)
            .step_by(64)
            .flat_map(|k| (0..4).map(move |s| k + s * 16))
            .collect();
        assert_eq!(original, deep);
    }
    let free = new.find("a.free();").unwrap();
    let handoff = new[..free].rfind("sync_cube();").unwrap();
    let scratch = new[free..].find("let mut scratch").unwrap() + free;
    assert!(handoff < free && free < scratch);
    assert!(!new[free..scratch].contains("cmma::load"));
}

#[test]
fn deep_expert_actual_three_role_entries_compile_all_twenty_four_precise_layouts() {
    let names = [
        core::any::type_name::<expert_wide_gu::ExpertWideGu<WgpuRuntime>>(),
        core::any::type_name::<expert_wide_gu_direct::ExpertWideGuDirect<WgpuRuntime>>(),
        core::any::type_name::<expert_wide_down::ExpertWideDown<WgpuRuntime>>(),
    ];
    for (name, exact) in names.into_iter().zip([
        "::native_expert_wide::deep::expert_wide_gu::ExpertWideGu",
        "::native_expert_wide::deep::expert_wide_gu_direct::ExpertWideGuDirect",
        "::native_expert_wide::deep::expert_wide_down::ExpertWideDown",
    ]) {
        assert!(name.contains(exact));
    }
    for (down, direct) in [(false, false), (false, true), (true, false)] {
        for mask in 0..8 {
            compile(down, direct, mask).unwrap();
        }
    }
}
