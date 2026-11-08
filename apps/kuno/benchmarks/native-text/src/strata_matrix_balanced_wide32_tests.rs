use super::*;
#[test]
fn balanced_wide32_subgroups_cover_every_original_cell_once_without_overread() {
    for rows in [128usize, 256, 1280, 2560] {
        let mut cells = vec![0u8; 128 * rows];
        for x in 0..rows / 128 {
            for sg in 0..32 {
                for w in 0..32 {
                    for t in 0..16 {
                        let token = sg / 4 * 16 + t;
                        let row = x * 128 + sg % 4 * 32 + w;
                        cells[token * rows + row] += 1;
                    }
                }
            }
        }
        assert!(cells.iter().all(|&n| n == 1));
    }
    for columns in [64usize, 320, 640, 2560, 10240] {
        for sg in 0..32 {
            let token = sg / 4 * 16;
            let row = sg % 4 * 32;
            let mut bank0 = 0usize;
            let mut observed = Vec::new();
            for k in (0..columns).step_by(32) {
                assert_eq!(bank0, k);
                for offset in [bank0, k + 16] {
                    assert!((token + 15) * columns + offset + 15 < 128 * columns);
                    assert!((row + 31) * columns + offset + 15 < 128 * columns);
                    observed.push(offset);
                }
                if k + 32 < columns {
                    bank0 = k + 32;
                }
            }
            assert_eq!(observed, (0..columns).step_by(16).collect::<Vec<_>>());
        }
    }
}
#[test]
fn balanced_wide32_requires_actual512_thread_capability_and_original_prefix_extents() {
    for count in [128usize, 256, 512, 4096] {
        assert!(balanced_wide32_geometry(
            count,
            2560,
            128,
            128,
            49152,
            512,
            (512, 1, 1),
            16,
            32
        ));
        assert!(!balanced_wide32_geometry(
            count,
            2560,
            128,
            128,
            49152,
            256,
            (512, 1, 1),
            16,
            32
        ));
        assert!(!balanced_wide32_geometry(
            count,
            2560,
            128,
            128,
            49152,
            512,
            (256, 1, 1),
            16,
            32
        ));
    }
    for count in [0usize, 127, 129, 257] {
        assert!(!balanced_wide32_geometry(
            count,
            2560,
            128,
            128,
            49152,
            512,
            (512, 1, 1),
            16,
            32
        ));
    }
    assert!(gr_down_shape(true, 10240, 320, 0, 0, 320));
}
#[test]
fn balanced_wide32_actual_dense_and_expert_modules_validate_real_carried_operands_and_consumers() {
    validate_balanced_wide32_sources().unwrap();
    crate::strata_matrix::native_expert_wide::balanced_wide32::validate_sources().unwrap();
}
