use super::*;
#[test]
fn n64_actual_expert_modules_keep_original_publication_and_no_operand_slm() {
    validate_sources().unwrap();
}
#[test]
fn n64_original_expert_scope_and_physical_columns_keep_same_logical_prefix_tail_and_owners() {
    for count in [256, 257, 287, 288, 383, 384, 4096] {
        let p = plan(count).unwrap();
        assert_eq!(p.prefix + p.tail, count);
        assert_eq!(p.prefix % 128, 0);
        for down in [false, true] {
            let (k, n) = if down { (640, 2560) } else { (2560, 1280) };
            assert!(super::super::shared_reuse::geometry(
                count,
                k,
                n,
                128,
                down,
                49152,
                512,
                (512, 1, 1),
                16,
                32,
                true
            ));
            assert!(!super::super::shared_reuse::geometry(
                count,
                k,
                n,
                1,
                down,
                49152,
                512,
                (512, 1, 1),
                16,
                32,
                true
            ));
            let mut cells = vec![0u8; p.prefix * n];
            for y in 0..p.prefix / 128 {
                for x in 0..n / 64 {
                    for sg in 0..32 {
                        for tm in 0..2 {
                            for m in 0..8 {
                                for col in 0..16 {
                                    let t = y * 128 + sg / 4 * 16 + tm * 8 + m;
                                    let row = x * 64 + sg % 4 * 16 + col;
                                    assert!(t < p.prefix && row < n);
                                    cells[t * n + row] += 1;
                                }
                            }
                        }
                    }
                }
            }
            assert!(cells.iter().all(|&v| v == 1));
            assert_eq!(
                (n / 64) * (p.prefix / 128),
                2 * (n / 128) * (p.prefix / 128)
            );
        }
    }
}
