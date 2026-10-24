//! Actual producer/plan/classifier controls; metadata controls are not GPU proof.
use super::*;
fn fixture(counts: &[u32]) -> [u32; WORDS] {
    let mut w = [0; WORDS];
    w[0] = 4096;
    w[1] = 2560;
    w[2] = 1280;
    w[4] = 7;
    w[6] = 100;
    w[7] = 100;
    w[8] = 300;
    w[9] = counts.len() as u32;
    w[10] = *counts.iter().max().unwrap();
    let mut prefix = 0;
    for (i, &n) in counts.iter().enumerate() {
        w[16 + i * 4..20 + i * 4].copy_from_slice(&[7 + i as u32, n, prefix, i as u32]);
        prefix += n.min(4096).div_ceil(16) * 16;
    }
    w
}
#[test]
fn regular_cohort_info_proves_exact_existing_images_packed_rows_and_strict_member_class() {
    for n in 32..256u32 {
        for len in 1..=8 {
            let w = fixture(&vec![n; len]);
            let rows = validate_words(&w).unwrap();
            assert_eq!(rows, len * n.div_ceil(16) as usize * 16);
            assert!(rows <= 2048);
            assert!(rows as u64 * 2560 * 2 <= X_BYTES);
            assert!(rows as u64 * 640 * 2 <= H_BYTES);
            assert_eq!(IMAGE_BYTES * 8, 75 * 1024 * 1024);
            assert!(IMAGE_BYTES * 8 <= storage::WORKING_BYTES);
            assert!(crate::strata_prompt::experts::cohorts::validate_words(&w).is_err());
        }
    }
    for n in [0, 1, 8, 16, 31, 256, 4096, u32::MAX] {
        assert!(validate_words(&fixture(&[n])).is_err());
    }
    let mut old = fixture(&[16]);
    old[10] = 16;
    assert!(crate::strata_prompt::experts::cohorts::validate_words(&old).is_ok());
}
#[test]
fn regular_cohort_info_rejects_changed_prefix_spare_slot_and_unknown_common_metadata() {
    let counts = [32, 33, 63, 64, 65, 127, 128, 255];
    let w = fixture(&counts);
    validate_words(&w).unwrap();
    for (i, v) in [
        (0, 0),
        (0, 4097),
        (0, 31),
        (4, 8),
        (6, 101),
        (7, 101),
        (8, 200),
        (9, 0),
        (9, 7),
        (9, 9),
        (10, 254),
        (12, 2),
        (13, 1),
        (14, 1),
        (15, 1),
    ] {
        let mut bad = w;
        bad[i] = v;
        assert!(validate_words(&bad).is_err(), "field {i}");
    }
    for i in 0..8 {
        for (field, value) in [(0, 999), (1, 31), (1, 256), (2, w[18 + i * 4] + 16), (3, 9)] {
            let mut bad = w;
            bad[16 + i * 4 + field] = value;
            assert!(validate_words(&bad).is_err(), "record {i} field {field}");
        }
    }
    for i in 1..8 {
        let mut bad = w;
        bad[19 + i * 4] = 0;
        assert!(validate_words(&bad).is_err());
    }
    let mut spare = fixture(&[32, 64]);
    spare[24] = 1;
    assert!(validate_words(&spare).is_err());
}
#[test]
fn regular_cohort_all_eight_actual_wgsl_producers_validate_with_distinct_direct8_identity() {
    let sources = sources();
    assert_eq!(sources.len(), 8);
    for k in sources {
        let module = wgpu::naga::front::wgsl::parse_str(&k.source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert!(k.source.contains("v.y<32u || v.y>=256u"));
        assert!(k.source.contains("v.z%16u"));
    }
    assert_eq!(
        RegularOp::Image {
            dtype: q::IQ3_S,
            down: false
        }
        .name(),
        "StrataExpertRegularCohortGuDirect8Image"
    );
    assert_eq!(
        RegularOp::Image {
            dtype: q::IQ4_NL,
            down: true
        }
        .name(),
        "StrataExpertRegularCohortDownDirect8Image"
    );
}
#[test]
fn regular_cohort_real_matrix_layouts_keep_original_resources_and_carried_reaching_definitions() {
    // Executes the real complete compiler/validator including all new16 layouts,
    // original granular product SSA and actual1024B publication allocation.
    crate::strata_matrix::validate_sources().unwrap();
}
#[test]
fn regular_cohort_two_bank_plan_keeps_order_unique_slots_original_budget_and_bounded_grid_waste() {
    use crate::strata_prompt::experts::raw_banks::Plan;
    let values = [
        1, 8, 31, 32, 33, 48, 63, 64, 65, 127, 128, 255, 256, 257, 1024, 4096,
    ];
    for len in [1, 7, 8, 9, 32, 127, 512] {
        for pattern in 0..12 {
            let members = (0..len)
                .map(|i| values[(i * 7 + pattern) % values.len()])
                .collect::<Vec<_>>();
            let host = (0..len).map(|i| (i + pattern) % 3 != 0).collect::<Vec<_>>();
            let mut p = Plan::new_regular(&members, &host).unwrap();
            let mut group = 0;
            let mut seen = vec![0; len];
            while group < len {
                while p.issue_next().unwrap().is_some() {}
                let unit = p.current(group).unwrap();
                let mut used = [false; 8];
                for i in unit.first..unit.end {
                    seen[i] += 1;
                    match p.slot(unit, i).unwrap() {
                        Some(s) => {
                            assert!(host[i] && !used[s]);
                            used[s] = true;
                        }
                        None => assert!(!host[i]),
                    }
                }
                if unit.regular {
                    assert!(!unit.legacy && unit.end - unit.first <= 8);
                    let blocks = members[unit.first..unit.end]
                        .iter()
                        .map(|n| n.div_ceil(16))
                        .collect::<Vec<_>>();
                    assert!(
                        members[unit.first..unit.end]
                            .iter()
                            .all(|n| (32..256).contains(n))
                    );
                    assert!(
                        blocks.iter().max().unwrap() * (blocks.len() as u32)
                            <= 2 * blocks.iter().sum::<u32>()
                    );
                    assert!(blocks.iter().sum::<u32>() * 16 <= 2048);
                    p.consumed(unit).unwrap();
                } else if unit.legacy {
                    assert!(members[unit.first..unit.end].iter().all(|n| *n >= 256));
                    assert!(unit.end - unit.first <= 4);
                    for i in unit.first..unit.end {
                        p.complete_group(unit, i).unwrap();
                    }
                } else {
                    assert!(members[unit.first..unit.end].iter().all(|n| *n < 32));
                    p.consumed(unit).unwrap();
                }
                group = unit.end;
            }
            let (m, r, peak, _) = p.finish().unwrap();
            assert_eq!(m, host.iter().filter(|v| **v).count());
            assert_eq!(m + r, len);
            assert!(peak <= 16);
            assert!(seen.iter().all(|v| *v == 1));
        }
    }
}
#[test]
fn regular_cohort_bank_retirement_requires_the_actual_complete_unit_and_rejects_forged_kinds() {
    use crate::strata_prompt::experts::raw_banks::Plan;
    let mut p = Plan::new_regular(&[33; 24], &[true; 24]).unwrap();
    let a = p.issue_next().unwrap().unwrap();
    let b = p.issue_next().unwrap().unwrap();
    assert!(a.regular && b.regular);
    assert!(p.issue_next().unwrap().is_none());
    assert!(p.complete_group(a, a.first).is_err());
    assert!(p.consumed(b).is_err());
    let mut bad = a;
    bad.regular = false;
    assert!(p.consumed(bad).is_err());
    p.consumed(a).unwrap();
    assert!(p.slots(a).is_err());
    let c = p.issue_next().unwrap().unwrap();
    assert_eq!(c.bank, a.bank);
    p.consumed(b).unwrap();
    p.consumed(c).unwrap();
    assert_eq!(p.finish().unwrap(), (24, 0, 16, 3));
    let mut old = Plan::new(&[33; 8], &[true; 8]).unwrap();
    let u = old.issue_next().unwrap().unwrap();
    assert!(u.legacy && !u.regular && u.end - u.first == 4);
}
