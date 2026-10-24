use super::*;
#[test]
fn n64_full_tiles_cover_original_logical_outputs_and_authentic_gr320_without_padding_reads() {
    for rows in [128usize, 256, 320, 1280, 2560] {
        let mut seen = vec![0u8; 128 * rows];
        for rb in (0..rows).step_by(64) {
            for sg in 0..32 {
                for tm in 0..2 {
                    for m in 0..8 {
                        for n in 0..16 {
                            let t = sg / 4 * 16 + tm * 8 + m;
                            let row = rb + sg % 4 * 16 + n;
                            assert!(row < rows);
                            seen[t * rows + row] += 1;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
        for cols in [64usize, 128, 320, 10240] {
            for sg in 0..32 {
                for k in [0, cols - 16] {
                    for tm in 0..2 {
                        for m in 0..8 {
                            let t = sg / 4 * 16 + tm * 8 + m;
                            let at = (t * cols + k) / 4;
                            assert!(at + 3 < 128 * cols / 4);
                        }
                    }
                    for rb in (0..rows).step_by(64) {
                        let row = rb + sg % 4 * 16 + 15;
                        let at = (128 + row * cols + k) / 4;
                        assert!(at + 3 < (128 + rows * cols) / 4);
                    }
                }
            }
        }
    }
}
#[test]
fn n64_actual_two_bank_walk_preserves_k16_chronology_original_scope_and_full_gr_credit() {
    for cols in (32usize..=10240).step_by(32) {
        let mut seen = Vec::new();
        let mut bank0 = 0;
        for k in (0..cols).step_by(32) {
            assert_eq!(bank0, k);
            let bank1 = k + 16;
            seen.push(bank0);
            if k + 32 < cols {
                bank0 = k + 32;
                assert!(bank0 + 15 < cols);
            }
            seen.push(bank1);
            assert!(bank1 + 15 < cols);
        }
        assert_eq!(seen, (0..cols).step_by(16).collect::<Vec<_>>());
    }
    for n in [128usize, 256, 512, 4096] {
        assert!(global_n64_geometry(
            n,
            10240,
            384,
            4,
            49152,
            512,
            (512, 1, 1),
            16,
            32
        ));
    }
    for n in [0, 127, 129, 148, 257, 513] {
        assert!(!global_n64_geometry(
            n,
            10240,
            384,
            4,
            49152,
            512,
            (512, 1, 1),
            16,
            32
        ));
    }
    assert!(!global_n64_geometry(
        128,
        10240,
        384,
        4,
        49151,
        512,
        (512, 1, 1),
        16,
        32
    ));
    assert!(!global_n64_geometry(
        128,
        10240,
        384,
        4,
        49152,
        511,
        (512, 1, 1),
        16,
        32
    ));
    assert!(!global_n64_geometry(
        128,
        10240,
        384,
        4,
        49152,
        512,
        (511, 1, 1),
        16,
        32
    ));
    assert!(!global_n64_geometry(
        128,
        10240,
        384,
        4,
        49152,
        512,
        (512, 1, 1),
        32,
        32
    ));
    assert!(!global_n64_geometry(
        128,
        10240,
        320,
        4,
        49152,
        512,
        (512, 1, 1),
        16,
        32
    ));
    assert!(gr_down_shape(true, 10240, 320, 0, 0, 320));
    assert!(!gr_down_shape(false, 10240, 320, 0, 0, 320));
    for n in [128, 256, 512, 513, 1024] {
        assert_eq!(full_gr_credit(true, n, false), n >= 512);
        assert!(!full_gr_credit(true, n, true));
        assert!(!full_gr_credit(false, n, false));
    }
    for n in [148, 257, 513] {
        let (a, b) = prefix_partition(n).unwrap();
        assert_eq!(a + b, n);
        assert_eq!(b, n % 128);
    }
}
#[test]
fn n64_actual_dense_modules_keep_original_bindings_two_banks_zero_slm_and_native_negatives() {
    validate_global_n64_sources().unwrap();
}
