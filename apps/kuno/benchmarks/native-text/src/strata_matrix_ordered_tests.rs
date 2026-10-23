use super::*;
fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}("))
        .or_else(|| source.find(&format!("fn {name}<"))).expect(name);
    let open = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (offset, byte) in source[open..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            return &source[start..open + offset + 1];
        }
    }
    panic!("function extent {name}")
}
fn tokens(s: &str) -> String {
    // rustfmt adds a syntactic trailing call comma when the qualified deep
    // helper wraps. No expression/operator/argument is normalized away.
    s.split_whitespace().collect::<String>().replace(",)", ")")
}
fn restore_rank(s: &str) -> String {
    tokens(s).replace(
        "(usize::cast_from(map[1028+group])+member)*2560",
        "route*2560",
    )
}
#[test]
fn every_original_product_guard_and_finite_publication_stays_available() {
    let original = include_str!("strata_matrix.rs");
    let ordered = include_str!("strata_matrix_ordered.rs");
    for name in ["expert", "expert_macro", "expert_narrow"] {
        assert_eq!(
            tokens(function(original, name)),
            restore_rank(function(ordered, name)),
            "{name}"
        );
    }
    #[cfg(feature = "strata-expert-macro64")]
    assert_eq!(
        tokens(function(
            include_str!("strata_matrix_macro64.rs"),
            "expert_macro64"
        )),
        restore_rank(function(ordered, "expert_macro64"))
    );
    #[cfg(feature = "strata-deep-matrix")]
    {
        let new = function(ordered, "expert_macro64_deep")
            .replace("fn expert_macro64_deep(", "fn expert_macro64(")
            .replace("deep_macro64::macro64_product", "macro64_product");
        assert_eq!(
            tokens(function(
                include_str!("strata_matrix_macro64_deep.rs"),
                "expert_macro64"
            )),
            restore_rank(&new)
        );
    }
    #[cfg(feature = "strata-expert-cohorts")]
    assert_eq!(
        tokens(function(
            include_str!("strata_matrix_expert_cohorts.rs"),
            "expert_cohort"
        )),
        restore_rank(function(
            include_str!("strata_matrix_ordered_cohorts.rs"),
            "expert_cohort"
        ))
    );
}
#[test]
fn wide_k32_k64_arithmetic_and_original_tail_are_token_identical() {
    let old = include_str!("strata_matrix_native_expert_wide.rs");
    let new = include_str!("strata_matrix_ordered_wide.rs");
    for name in ["wide", "tail", "publish_direct_gu", "publication_barrier"] {
        assert_eq!(
            tokens(function(old, name)),
            tokens(function(new, name)),
            "{name}"
        );
    }
    assert_eq!(
        tokens(function(old, "store_routed")),
        restore_rank(function(new, "store_routed"))
    );
    assert_eq!(
        tokens(function(old, "publish_routed")),
        tokens(function(new, "publish_checked")).replace("fnpublish_checked(", "fnpublish_routed(")
    );
    #[cfg(feature = "strata-deep-matrix")]
    for name in ["wide", "stage", "selected", "geometry", "launch_prefix"] {
        assert_eq!(
            tokens(function(
                include_str!("strata_matrix_native_expert_deep.rs"),
                name
            )),
            tokens(function(
                include_str!("strata_matrix_ordered_deep.rs"),
                name
            )),
            "deep {name}"
        );
    }
}
#[test]
fn ordered_destinations_are_a_complete_capacity_preserving_permutation() {
    use crate::strata_prompt::groups as abi;
    let count = 4096;
    let mut raw = vec![0u32; abi::LOGICAL + count * 10];
    raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    for route in 0..count * 10 {
        let expert = (route * 17 + (route % 10) * 13) % 512;
        raw[abi::LOGICAL + route] = expert as u32;
        raw[1540 + route] = ((expert * 13 + 17) % 512) as u32;
    }
    let map = abi::group(&raw, count).unwrap();
    let groups = abi::validate(&map, count).unwrap();
    let mut seen = vec![false; count * 10];
    let mut inverse = vec![usize::MAX; count * 10];
    for group in 0..groups {
        let prefix = map[1028 + group] as usize;
        for member in 0..map[516 + group] as usize {
            let ordinal = prefix + member;
            let route = map[1540 + ordinal] as usize;
            assert!(ordinal < count * 10 && route < count * 10);
            assert!(!seen[ordinal]);
            seen[ordinal] = true;
            assert_eq!(inverse[route], usize::MAX);
            inverse[route] = ordinal;
            assert!((ordinal * 2560 + 2559) * 4 < abi::PART_BYTES as usize);
        }
    }
    assert!(seen.into_iter().all(|v| v));
    assert!(inverse.into_iter().all(|v| v < count * 10));
}
#[test]
fn collective_global_store_and_checked_bad_route_fallback_cover_exact_members() {
    for count in [128usize, 129, 255, 256, 257, 287, 383] {
        let prefix = count / 128 * 128;
        let mut seen = vec![false; prefix * 2560];
        let mut status = vec![0usize; prefix];
        for mblock in (0..prefix).step_by(128) {
            for nblock in (0..2560).step_by(128) {
                for sg in 0..32 {
                    for tile_m in 0..2 {
                        for tile_n in 0..2 {
                            let member0 = mblock + (sg / 4) * 16 + tile_m * 8;
                            let column = nblock + (sg % 4) * 32 + tile_n * 16;
                            for row in 0..8 {
                                for col in 0..16 {
                                    let at = (member0 + row) * 2560 + column + col;
                                    assert!(at < seen.len());
                                    assert!(!seen[at]);
                                    seen[at] = true;
                                }
                            }
                            if column == 0 {
                                for lane in 0..8 {
                                    status[member0 + lane] += 1;
                                }
                            }
                            for bad in [0, prefix - 1] {
                                let all_good = !(member0..member0 + 8).contains(&bad);
                                for lane in 0..8 {
                                    assert_eq!(all_good, (member0 + lane) / 8 != bad / 8);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|v| v));
        assert!(status.into_iter().all(|v| v == 1));
    }
}
#[test]
fn additional_definitions_have_genuine_distinct_types_and_do_not_replace_old_population() {
    let old = native::definitions();
    assert!(old.iter().all(|(name, _)| !name.starts_with("ordered_")));
    let extra = definitions();
    let expected =
        4 + if cfg!(feature = "strata-narrow-global") {
            2
        } else {
            0
        } + if cfg!(feature = "strata-expert-macro64") {
            1
        } else {
            0
        } + if cfg!(feature = "strata-deep-matrix") {
            1
        } else {
            0
        } + if cfg!(feature = "strata-native-expert-wide") {
            4
        } else {
            0
        } + if cfg!(all(
            feature = "strata-native-expert-wide",
            feature = "strata-deep-matrix"
        )) {
            1
        } else {
            0
        } + if cfg!(feature = "strata-expert-cohorts") {
            2
        } else {
            0
        };
    assert_eq!(extra.len(), expected);
    assert!(
        extra
            .iter()
            .all(|(n, _)| n.starts_with("ordered_") && n.ends_with("down"))
    );
    let old = core::any::type_name::<native::expert::Expert<WgpuRuntime>>();
    let new = core::any::type_name::<native::ordered::expert::Expert<WgpuRuntime>>();
    assert_ne!(old, new);
    assert!(new.contains("::native::ordered::expert::Expert<"));
    let source = include_str!("strata_matrix_ordered_wide.rs");
    let publish = function(source, "publish_routed");
    assert!(publish.contains("plane_all(route_good)"));
    assert!(
        publish.contains("status[comptime!(abi::ROUTES)+route]=2")
            || tokens(publish).contains("status[comptime!(abi::ROUTES)+route]=2")
    );
    assert!(publish.contains("publish_checked("));
    assert!(cfg!(feature = "strata-native-subgroup-publish"));
}
