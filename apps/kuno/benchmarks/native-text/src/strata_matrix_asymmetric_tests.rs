use super::*;
#[test]
fn asymmetric_stage_operands_and_register_jobs_cover_two_banks_without_alias() {
    for bank in 0..2usize {
        let (mut a, mut b) = (
            std::collections::BTreeMap::new(),
            std::collections::BTreeMap::new(),
        );
        for round in 0..2usize {
            for lane in 0..512usize {
                let job = lane + round * 512;
                let row = job / 4;
                let col = job % 4 * 8;
                for v in 0..2usize {
                    let da = bank * 2048 + col / 16 * 1024 + row * 4 + (col % 16) / 4 + v;
                    assert!(a.insert(da, (row, col + v * 4)).is_none());
                    if round == 0 {
                        let db = bank * 1024 + col / 16 * 512 + row * 4 + (col % 16) / 4 + v;
                        assert!(b.insert(db, (row, col + v * 4)).is_none());
                    }
                }
            }
        }
        assert_eq!(
            a.keys().copied().collect::<Vec<_>>(),
            (bank * 2048..(bank + 1) * 2048).collect::<Vec<_>>()
        );
        assert_eq!(
            b.keys().copied().collect::<Vec<_>>(),
            (bank * 1024..(bank + 1) * 1024).collect::<Vec<_>>()
        );
        for lane in 0..512usize {
            let da = bank * 2048 + (lane % 4 / 2) * 1024 + (lane / 4) * 4 + (lane % 2) * 2;
            let db = bank * 1024 + (lane % 4 / 2) * 512 + (lane / 4) * 4 + (lane % 2) * 2;
            for v in 0..2 {
                assert_eq!(a[&(da + v)], (lane / 4, lane % 4 * 8 + v * 4));
                assert_eq!(a[&(da + 512 + v)], (lane / 4 + 128, lane % 4 * 8 + v * 4));
                assert_eq!(b[&(db + v)], (lane / 4, lane % 4 * 8 + v * 4));
            }
        }
    }
    assert_eq!((4096 + 2048) * 8, 49152);
}
#[test]
fn asymmetric_output_chronology_final_prefetch_and_hardware_guards_are_exact() {
    for gr in [false, true] {
        for row_base in if gr { vec![0, 128, 256] } else { vec![0] } {
            let mut coverage = std::collections::BTreeSet::new();
            for sg in 0..32 {
                for tm in 0..4 {
                    for tn in 0..2 {
                        for m in 0..8 {
                            for n in 0..16 {
                                let token = sg / 4 * 32 + tm * 8 + m;
                                let row = row_base + sg % 4 * 32 + tn * 16 + n;
                                if !gr || row < 320 {
                                    assert!(coverage.insert((token, row)));
                                }
                            }
                        }
                    }
                }
            }
            let expected = (0..256)
                .flat_map(|t| {
                    (row_base..(row_base + 128).min(if gr { 320 } else { 128 }))
                        .map(move |r| (t, r))
                })
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(coverage, expected);
        }
    }
    for cols in [32usize, 64, 320, 2560, 10240] {
        assert_eq!(
            (0..cols)
                .step_by(32)
                .flat_map(|k| [k, k + 16])
                .collect::<Vec<_>>(),
            (0..cols).step_by(16).collect::<Vec<_>>()
        );
        for k in (0..cols).step_by(32) {
            if k + 32 < cols {
                for lane in 0..512 {
                    let col = k + 32 + lane % 4 * 8;
                    assert!(col + 7 < cols);
                }
            }
        }
    }
    let valid = |count, cols, rows, first, slm, units, dim, pmin, pmax| {
        asymmetric_geometry(count, cols, rows, first, slm, units, dim, pmin, pmax)
    };
    assert!(valid(256, 32, 128, 4, 49152, 512, (512, 1, 1), 16, 32));
    for count in [0, 128, 255, 257, 384] {
        assert!(!valid(count, 2560, 128, 0, 49152, 512, (512, 1, 1), 16, 32));
    }
    assert!(!valid(256, 2560, 128, 0, 49151, 512, (512, 1, 1), 16, 32));
    assert!(!valid(256, 2560, 128, 0, 49152, 511, (512, 1, 1), 16, 32));
    assert!(!valid(256, 2560, 128, 0, 49152, 512, (511, 1, 1), 16, 32));
    assert!(!valid(256, 2560, 128, 0, 49152, 512, (512, 1, 1), 32, 32));
    assert!(!valid(256, 2559, 128, 0, 49152, 512, (512, 1, 1), 16, 32));
    assert!(!valid(256, 2560, 127, 0, 49152, 512, (512, 1, 1), 16, 32));
    assert!(!valid(256, 2560, 128, 1, 49152, 512, (512, 1, 1), 16, 32));
}
#[test]
fn asymmetric_actual_modules_keep_48k_geometry_chronology_and_register_pipeline() {
    validate_asymmetric_sources().unwrap();
}
