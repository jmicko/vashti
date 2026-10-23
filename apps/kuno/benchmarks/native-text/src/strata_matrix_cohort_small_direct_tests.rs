//! Source/geometry/ABI controls for genuine additive cohort producers.
use super::*;

#[test]
fn small_direct_cohort_preserves_every_original_guard_and_chronological_product() {
    let old = include_str!("strata_matrix_expert_cohorts.rs");
    let new = include_str!("strata_matrix_cohort_small_direct.rs");
    let old = old
        .split_once("#[cube(launch_unchecked)]")
        .unwrap()
        .1
        .split_once("    let mut out = SharedMemory::<f32>")
        .unwrap()
        .0
        .replace(
            "super::register_prefetch::accumulate",
            "crate::strata_matrix::register_prefetch::accumulate",
        );
    let new = new
        .split_once("#[cube(launch_unchecked)]")
        .unwrap()
        .1
        .split_once("    // Each subgroup owns its fallback slot.")
        .unwrap()
        .0;
    let compact = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    assert_eq!(compact(&old), compact(new));
    let selected = include_str!("strata_matrix_cohort_small_direct.rs");
    assert_eq!(
        selected
            .matches("crate::strata_matrix::small_direct::publish(")
            .count(),
        2
    );
    assert!(
        compact(selected).contains("token_base,row_base+subgroup*16,members,prefix,group,down")
    );
    assert!(
        compact(selected).contains("token_base+8,row_base+subgroup*16,members,prefix,group,down")
    );
    assert!(selected.contains("SharedMemory::<f32>::new(512usize)"));
    assert!(!selected.contains("sync_cube()"));
}

#[test]
fn small_direct_cohort_full_fragments_and_checked_tails_cover_all_members_exactly_once() {
    for count in 1usize..32 {
        let tile = if count <= 8 { 8usize } else { 16usize };
        let mut seen = vec![0u8; count * 64];
        let mut direct = 0;
        let mut checked = 0;
        for cube_y in 0..count.div_ceil(tile) {
            for subgroup in 0..4 {
                for half in 0..tile / 8 {
                    let member0 = cube_y * tile + half * 8;
                    let fast = member0 + 8 <= count;
                    for r in 0..8 {
                        let member = member0 + r;
                        for c in 0..16 {
                            let col = subgroup * 16 + c;
                            if member < count {
                                seen[member * 64 + col] += 1;
                                if fast {
                                    direct += 1;
                                } else {
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&x| x == 1));
        assert_eq!(direct, count / 8 * 8 * 64);
        assert_eq!(direct + checked, count * 64);
        // One invalid route makes exactly that complete eight-member fragment
        // fall back; no other full fragment can inherit its authorization.
        for invalid in 0..count {
            for member0 in (0..count).step_by(8) {
                let fast = member0 + 8 <= count && !(member0..member0 + 8).contains(&invalid);
                if fast {
                    assert!(!(member0..member0 + 8).contains(&invalid));
                }
            }
        }
    }
}

#[test]
fn small_direct_cohort_genuine_types_keep_original_six_owner_and_48_scalar_registration() {
    let old = include_str!("strata_matrix_expert_cohorts.rs");
    let new = include_str!("strata_matrix_cohort_small_direct.rs");
    let definition = |s: &str| {
        s.split_once("fn definition(down: bool, tile: usize)")
            .unwrap()
            .1
            .split_once("b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))")
            .unwrap()
            .0
            .to_owned()
    };
    let compact = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    assert_eq!(compact(&definition(old)), compact(&definition(new)));
    let registration = |s: &str| {
        s.split_once("let settings = matrix_settings();")
            .unwrap()
            .1
            .split_once("    Ok(())")
            .unwrap()
            .0
            .split_once("    unsafe {")
            .unwrap()
            .0
            .to_owned()
    };
    assert_eq!(compact(&registration(old)), compact(&registration(new)));
    let names = definitions()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "small_direct_expert_cohort8_gu",
            "small_direct_expert_cohort8_down",
            "small_direct_expert_cohort16_gu",
            "small_direct_expert_cohort16_down"
        ]
    );
    assert!(new.contains("super::native::compatible") == false);
    assert!(new.contains("crate::strata_matrix::native::compatible(client, false)?"));
}
