use super::*;
#[test]
fn local_prefetch_real_tiles_cover_each_output_once_and_preserve_initialized_source_extents() {
    for (members, tile, n) in [
        (128usize, 128usize, 128usize),
        (32, 32, 128),
        (33, 32, 128),
        (63, 32, 128),
        (64, 64, 128),
        (65, 64, 128),
        (127, 64, 128),
        (128, 64, 128),
        (129, 64, 128),
        (255, 64, 128),
        (9, 16, 128),
        (15, 16, 128),
        (16, 16, 128),
        (17, 16, 128),
        (31, 16, 128),
    ] {
        let pair = tile == 128;
        let per_sg = if pair { 32 } else { 16 };
        let threads = if pair { 256 } else { tile * 4 };
        let padded =
            members.div_ceil(if tile == 16 { 16 } else { 32 }) * if tile == 16 { 16 } else { 32 };
        let mut seen = vec![0u8; members * n];
        for y in 0..members.div_ceil(tile) {
            for sg in 0..threads / 16 {
                let token = y * tile + (sg / 4) * per_sg;
                if token >= members {
                    continue;
                }
                assert!(token + per_sg <= padded);
                for t in token..(token + per_sg).min(members) {
                    for row in sg % 4 * 32..sg % 4 * 32 + 32 {
                        seen[t * n + row] += 1;
                    }
                }
                for cols in [64usize, 320, 640, 2560, 10240] {
                    for k in (0..cols).step_by(16) {
                        assert!((token + per_sg - 1) * cols + k + 15 < padded * cols);
                        assert!((sg % 4 * 32 + 31) * cols + k + 15 < n * cols);
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|v| v == 1));
    }
}
#[test]
fn local_prefetch_actual_all_layouts_prove_current_operands_alternation_and_publication() {
    native_wide::local_wide32::validate_local_wide32_sources().unwrap();
    native_expert_wide::local_wide32::validate_sources().unwrap();
    local_small::validate_sources().unwrap();
}
#[test]
fn local_prefetch_actual_generated_roles_keep_matrix_and_expert_family_attribution() {
    for (i, name) in local_wide32_profile_names().into_iter().enumerate() {
        assert_eq!(
            crate::strata_prefill_profile::local_prefetch_test_classes(name).0,
            1
        );
        if i < 2 {
            assert_eq!(
                crate::strata_prefill_profile::local_prefetch_test_classes(name).1,
                i
            );
        }
    }
    for (i, name) in local_small::profile_names().into_iter().enumerate() {
        assert_eq!(
            crate::strata_prefill_profile::local_prefetch_test_classes(name).0,
            1
        );
        assert_eq!(
            crate::strata_prefill_profile::local_prefetch_test_classes(name).1,
            if i == 4 { 6 } else { 4 }
        );
    }
}
