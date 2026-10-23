use super::*;
#[test]
fn global_register_source_fragments_and_output_cover_original_tile_without_padding_reads() {
    for gr in [false, true] {
        for row_base in if gr { vec![0, 128, 256] } else { vec![0] } {
            let mut output = std::collections::BTreeSet::new();
            for sg in 0..32 {
                for tm in 0..2 {
                    for tn in 0..2 {
                        for m in 0..8 {
                            for n in 0..16 {
                                let token = sg / 4 * 16 + tm * 8 + m;
                                let row = row_base + sg % 4 * 32 + tn * 16 + n;
                                if !gr || row < 320 {
                                    assert!(output.insert((token, row)));
                                }
                            }
                        }
                    }
                }
            }
            let expected = (0..128)
                .flat_map(|t| {
                    (row_base..(row_base + 128).min(if gr { 320 } else { 128 }))
                        .map(move |r| (t, r))
                })
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(output, expected);
            for cols in [64usize, 128, 320, 10240] {
                for sg in 0..32 {
                    for tm in 0..2 {
                        for row in 0..8 {
                            for k in (0..cols).step_by(16) {
                                let a = ((sg / 4 * 16 + tm * 8 + row) * cols + k) / 4;
                                assert!(a + 3 < 128 * cols / 4);
                            }
                        }
                    }
                    for tn in 0..2 {
                        let r = row_base + sg % 4 * 32 + tn * 16;
                        if !gr || r < 320 {
                            assert!(r + 15 < if gr { 320 } else { 128 });
                            let b = ((r + 15) * cols + cols - 16) / 4;
                            assert!(b + 3 < (if gr { 320 } else { 128 }) * cols / 4);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn global_register_two_banks_keep_k16_chronology_final_bounds_and_original_capability_scope() {
    for cols in (32usize..=10240).step_by(32) {
        let (mut bank0, mut seen) = (0, Vec::new());
        for k in (0..cols).step_by(32) {
            let bank1 = k + 16;
            assert_eq!(bank0, k);
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
    let valid = |count, cols, rows, first, shared, units, dim, pmin, pmax| {
        global_register_geometry(count, cols, rows, first, shared, units, dim, pmin, pmax)
    };
    for n in [128, 256, 384, 512, 4096] {
        assert!(valid(n, 10240, 384, 4, 49152, 512, (512, 1, 1), 16, 32));
    }
    for n in [0, 127, 129, 148, 257, 513] {
        assert!(!valid(n, 10240, 384, 4, 49152, 512, (512, 1, 1), 16, 32));
    }
    assert!(!valid(128, 10240, 384, 4, 49151, 512, (512, 1, 1), 16, 32));
    assert!(!valid(128, 10240, 384, 4, 49152, 511, (512, 1, 1), 16, 32));
    assert!(!valid(128, 10240, 384, 4, 49152, 512, (511, 1, 1), 16, 32));
    assert!(!valid(128, 10239, 384, 4, 49152, 512, (512, 1, 1), 16, 32));
    assert!(!valid(128, 10240, 320, 4, 49152, 512, (512, 1, 1), 16, 32));
    assert!(!valid(128, 10240, 384, 1, 49152, 512, (512, 1, 1), 16, 32));
    assert!(!valid(128, 10240, 384, 4, 49152, 512, (512, 1, 1), 32, 32));
    for count in [148, 257, 513] {
        let (prefix, suffix) = prefix_partition(count).unwrap();
        assert_eq!(prefix + suffix, count);
        assert_eq!(suffix, count % 128);
    }
}
#[test]
fn global_register_actual_modules_keep_zero_slm_both_global_banks_and_original_bindings() {
    validate_global_register_sources().unwrap();
}
