use super::*;
fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let p = source
        .find(&format!("fn {name}("))
        .or_else(|| source.find(&format!("fn {name}<")))
        .unwrap();
    let first = source[p..].find('{').unwrap() + p;
    let mut depth = 1;
    let mut end = first + 1;
    while depth > 0 {
        depth +=
            i32::from(source.as_bytes()[end] == b'{') - i32::from(source.as_bytes()[end] == b'}');
        end += 1;
    }
    &source[first + 1..end - 1]
}
fn normalized(s: &str) -> String {
    s.split_whitespace()
        .collect::<String>()
        .replace("crate::strata_matrix::native::macro_stage", "macro_stage")
        .replace(
            "crate::strata_matrix::native::macro64_stage",
            "macro64_stage",
        )
        .replace("crate::strata_matrix::native::narrow_stage", "narrow_stage")
        .replace(
            "crate::strata_matrix::register_prefetch::accumulate",
            "super::register_prefetch::accumulate",
        )
}
#[test]
fn small_direct_products_keep_original_operand_stages_chronological_math_and_handoff() {
    let new = include_str!("strata_matrix_small_direct_products.rs");
    for (old, name) in [
        (include_str!("strata_matrix.rs"), "macro_product"),
        (include_str!("strata_matrix_macro64.rs"), "macro64_product"),
        (include_str!("strata_matrix.rs"), "narrow_product"),
    ] {
        let old = body(old, name)
            .split_once("let mut out = SharedMemory")
            .unwrap()
            .0;
        let new = body(new, name).split_once("let mut scratch").unwrap().0;
        assert_eq!(
            normalized(old),
            normalized(new),
            "{name} complete original math/staging prefix"
        );
    }
}
#[test]
fn small_direct_fragment_mapping_covers_original_members_columns_and_complete8_cutoff() {
    for count in [
        1usize, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 257, 287, 288,
    ] {
        for tile in [8usize, 16, 32, 64] {
            let mut seen = vec![0u8; count * 64];
            let subgroups = if tile == 64 { 8 } else { 4 };
            for y in 0..count.div_ceil(tile) {
                for sg in 0..subgroups {
                    let m0 = y * tile + if tile >= 32 { sg / 2 * 16 } else { 0 };
                    let n0 = if tile >= 32 { sg % 2 * 32 } else { sg * 16 };
                    for tm in 0..if tile == 8 { 1 } else { 2 } {
                        for tn in 0..if tile >= 32 { 2 } else { 1 } {
                            let member0 = m0 + tm * 8;
                            let row0 = n0 + tn * 16;
                            let complete = member0 + 8 <= count;
                            for m in 0..8 {
                                for n in 0..16 {
                                    let member = member0 + m;
                                    if member < count {
                                        seen[member * 64 + row0 + n] += 1;
                                    }
                                    if complete {
                                        assert!(member < count);
                                    }
                                }
                            }
                            for bad in [0usize, count - 1] {
                                let good = complete && (member0..member0 + 8).all(|m| m != bad);
                                assert_eq!(good, complete && bad / 8 != member0 / 8);
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "tile{tile} count{count}");
        }
    }
}
#[test]
fn small_direct_publisher_keeps_real_collective_pending_status_and_checked_owner_rules() {
    let source = include_str!("strata_matrix_small_direct.rs");
    let p = body(source, "publish");
    assert!(p.contains("if complete && lane < 8"));
    assert!(p.contains("plane_all(route_good)"));
    assert!(p.contains("row0 + 16 <= rows"));
    assert!(p.contains("member < count && row < rows"));
    assert!(p.contains("(gu_prefix + member0) * 1280 + row0"));
    assert!(p.contains("(prefix + member0) * 2560 + row0"));
    assert!(p.contains("ROUTES) + route] = 2"));
    assert!(p.contains("if finite && status[route] == 1"));
    assert_eq!(p.matches("sync_plane()").count(), 2);
    assert!(!p.contains("sync_cube()"));
    let entries = include_str!("strata_matrix_small_direct_entries.rs");
    assert!(entries.contains("member_base != count / 128 * 128"));
    assert!(entries.contains("piece_members != count - member_base"));
    assert!(entries.contains("tail_header_valid(map, count, group)"));
    let activation = include_str!("strata_prompt_experts.rs");
    assert!(activation.contains("Op::ActivateCheckGu"));
    assert!(include_str!("strata_prompt_ordered_shared_reduce.wgsl").contains("down==2u"));
}
