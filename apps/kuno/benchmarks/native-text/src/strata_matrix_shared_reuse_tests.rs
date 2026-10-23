use super::*;
use native_expert_wide::shared_reuse as reuse;
fn section<'a>(s: &'a str, a: &str, b: &str) -> &'a str {
    s.split_once(a).unwrap().1.split_once(b).unwrap().0
}
fn tokens(s: &str) -> String {
    s.split_whitespace().collect::<String>().replace(",)", ")")
}
#[test]
fn real32k_explicit_admission_keeps_every_original_shape_and_device_condition() {
    assert_eq!((reuse::OPERAND_BYTES, reuse::LOGICAL_BYTES), (32768, 49152));
    for count in 0..=4097 {
        for down in [false, true] {
            for shared in [32767, 32768, 49152, 65536] {
                for explicit in [false, true] {
                    let (k, n) = if down { (640, 2560) } else { (2560, 1280) };
                    let expected = (if cfg!(feature = "strata-native-expert-medium") {
                        128
                    } else {
                        256
                    }..=4096)
                        .contains(&count)
                        && shared >= 32768
                        && explicit;
                    assert_eq!(
                        reuse::geometry(
                            count,
                            k,
                            n,
                            128,
                            down,
                            shared,
                            512,
                            (512, 1, 1),
                            16,
                            32,
                            explicit
                        ),
                        expected
                    );
                    assert!(!reuse::geometry(
                        count,
                        k,
                        n,
                        129,
                        down,
                        shared,
                        512,
                        (512, 1, 1),
                        16,
                        32,
                        explicit
                    ));
                    assert!(!reuse::geometry(
                        count,
                        k,
                        n,
                        128,
                        down,
                        shared,
                        511,
                        (512, 1, 1),
                        16,
                        32,
                        explicit
                    ));
                    assert!(!reuse::geometry(
                        count,
                        k,
                        n,
                        128,
                        down,
                        shared,
                        512,
                        (511, 1, 1),
                        16,
                        32,
                        explicit
                    ));
                    assert!(!reuse::geometry(
                        count,
                        k,
                        n,
                        128,
                        down,
                        shared,
                        512,
                        (512, 1, 1),
                        17,
                        32,
                        explicit
                    ));
                }
            }
        }
    }
}
#[test]
fn every_original_k32_operand_and_chronological_k16_product_stays_identical() {
    let old = include_str!("strata_matrix_native_expert_wide.rs");
    let new = include_str!("strata_matrix_wide_shared_reuse.rs");
    let old = section(
        old,
        "// REGISTER_STAGE_LOAD_BEGIN",
        "// REGISTER_STAGE_STORE_END",
    );
    let new = section(
        new,
        "// REGISTER_STAGE_LOAD_BEGIN",
        "// REGISTER_STAGE_STORE_END",
    );
    assert_eq!(tokens(old), tokens(new));
    for columns in [640usize, 2560] {
        let mut old = Vec::new();
        let mut k = 0;
        let mut current = 0;
        while k < columns {
            for step in 0..2 {
                old.push((k + step * 16, current));
            }
            current = 1 - current;
            k += 32;
        }
        let mut new = Vec::new();
        let mut current = 0;
        for stage_index in 0..columns / 32 {
            let k = stage_index * 32;
            for step in 0..2 {
                new.push((k + step * 16, current));
            }
            current = 1 - current;
        }
        assert_eq!(old, new);
        assert_eq!(new.len(), columns / 16);
    }
}
#[test]
fn final_workgroup_handoff_dominates_frees_and_every_original_publication_path() {
    let s = include_str!("strata_matrix_wide_shared_reuse.rs");
    let free = s.find("a.free();").unwrap();
    let handoff = s[..free].rfind("sync_cube();").unwrap();
    let scratch = s[free..].find("let mut scratch").unwrap() + free;
    assert!(handoff < free && free < scratch);
    assert!(!s[free..].contains("&a.to_slice()") && !s[free..].contains("&b.to_slice()"));
    assert!(!s[..free].contains("let mut scratch ="));
    assert!(s.contains("if !plane_all(header_valid(map, count, group))"));
    assert!(s.contains("for stage_index in 0usize..columns / 32usize"));
    assert!(s.contains("super::ordered::publish_routed("));
    assert!(s.contains("super::publish_routed("));
    assert!(s.contains("publish_direct_gu("));
    assert!(s.contains("super::ordered::expert_tail_down::launch_unchecked"));
    assert!(s.contains("super::expert_tail_down::launch_unchecked"));
    assert!(s.contains("super::expert_tail_gu::launch_unchecked"));
    let host = s.split_once("pub(crate) fn try_expert_mode(").unwrap().1;
    assert!(host.find("let tail_args").unwrap() < host.find("\n    unsafe {").unwrap());
    assert!(cfg!(feature = "strata-native-subgroup-publish"));
    let original = include_str!("strata_matrix_native_expert_wide.rs");
    assert!(original.contains("pub(super) const SHARED_BYTES: usize = 49152;"));
    let begin = original.find("fn wide(").unwrap();
    let end = original[begin..].find("fn tail(").unwrap() + begin;
    assert!(!original[begin..end].contains(".free();"));
}
#[test]
fn new_actual_entry_types_are_distinct_and_original_public_modes_stay_unchanged() {
    let names = [
        core::any::type_name::<
            native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<WgpuRuntime>,
        >(),
        core::any::type_name::<
            native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect<
                WgpuRuntime,
            >,
        >(),
        core::any::type_name::<
            native_expert_wide::shared_reuse::expert_wide_down::ExpertWideDown<WgpuRuntime>,
        >(),
        core::any::type_name::<
            native_expert_wide::ordered::shared_reuse::expert_wide_down::ExpertWideDown<
                WgpuRuntime,
            >,
        >(),
    ];
    for (name, suffix) in names.into_iter().zip([
        "::native_expert_wide::shared_reuse::expert_wide_gu::ExpertWideGu<",
        "::native_expert_wide::shared_reuse::expert_wide_gu_direct::ExpertWideGuDirect<",
        "::native_expert_wide::shared_reuse::expert_wide_down::ExpertWideDown<",
        "::native_expert_wide::ordered::shared_reuse::expert_wide_down::ExpertWideDown<",
    ]) {
        assert!(name.contains(suffix), "{name}");
    }
    assert_eq!(definitions().len(), 4);
    assert!(
        native_expert_wide::definitions()
            .iter()
            .all(|(name, _)| !name.contains("shared_reuse"))
    );
    assert!(
        native_expert_wide::ordered::definitions()
            .iter()
            .all(|(name, _)| !name.contains("shared_reuse"))
    );
    let main = include_str!("strata_matrix.rs");
    assert_eq!(
        main.matches("&& native_expert_wide::shared_reuse::try_expert_mode")
            .count(),
        2
    );
}
