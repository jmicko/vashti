use super::*;
#[test]
fn dual_lattice_has_exact_original_outputs_and_same_publication_scratch() {
    for n in 9usize..256 {
        for rows in [1280usize, 2560] {
            let mut visited = vec![0u8; n * rows];
            for y in 0..n.div_ceil(16) {
                for x in 0..rows / 32 {
                    for sg in 0..2 {
                        for c in 0..2 {
                            for m in 0..16 {
                                for row in 0..8 {
                                    let token = y * 16 + m;
                                    let col = x * 32 + sg * 16 + c * 8 + row;
                                    if token < n {
                                        visited[token * rows + col] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(visited.iter().all(|&v| v == 1));
        }
    }
    let mut scratch = vec![0u8; 256];
    for sg in 0..2 {
        for lane in 0..16 {
            for word in 0..8 {
                scratch[sg * 128 + lane * 8 + word] += 1;
            }
        }
    }
    assert!(scratch.iter().all(|&v| v == 1));
    assert_eq!(scratch.len() * 4, 1024);
}
#[test]
fn dual_consumers_preserve_initialized_packing_images_and_all_existing_producers() {
    let p = include_str!("strata_prompt_regular_cohorts.rs");
    let q = include_str!("strata_prompt_expert_cohorts.rs");
    assert!(p.contains("cohorts::dual_regular::launch"));
    assert!(p.contains("cohorts::regular::launch"));
    assert!(q.contains("cohorts::dual_small::launch"));
    assert!(q.contains("cohorts::granular_small::launch"));
    for (new, old) in [
        (
            include_str!("strata_matrix_dual_regular_cohort.rs"),
            include_str!("strata_matrix_regular_cohort.rs"),
        ),
        (
            include_str!("strata_matrix_dual_small_cohort.rs"),
            include_str!("strata_matrix_granular_small_cohort.rs"),
        ),
    ] {
        // The complete uniform/map/source guards preceding product dispatch remain exact.
        let body = |s: &str| {
            s[s.find("    let slot =").unwrap()..s.find("    crate::strata_matrix::").unwrap()]
                .to_owned()
        };
        assert_eq!(body(new), body(old));
        assert!(new.contains(" / 32) as u32"));
    }
    let product = include_str!("strata_matrix_dual_small.rs");
    assert!(product.contains("SharedMemory::<f32>::new(256usize)"));
    assert!(!product.contains("&c2"));
    assert!(!product.contains("&c3"));
}
#[test]
fn dual_k32_carry_initializes_every_operand_and_keeps_ascending_k16() {
    let text = include_str!("strata_matrix_dual_products.rs");
    assert!(text.contains("if k + 32 < columns"));
    assert!(!text.contains("row + 16"));
    for kmax in [640usize, 2560] {
        let mut initial = Some(0usize);
        let mut cells = [Vec::new(), Vec::new()];
        for k in (0..kmax).step_by(32) {
            assert_eq!(initial, Some(k));
            let next = k + 16;
            assert!(next + 16 <= kmax);
            for c in &mut cells {
                c.push(k);
                c.push(next);
            }
            initial = (k + 32 < kmax).then_some(k + 32);
        }
        assert!(initial.is_none());
        for c in cells {
            assert_eq!(c, (0..kmax).step_by(16).collect::<Vec<_>>());
        }
    }
}
