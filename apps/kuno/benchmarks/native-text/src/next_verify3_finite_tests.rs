use super::*;
use crate::quant_matvec::matmul::BatchSpans;

fn active(index: usize, first: usize, stride: usize, rows: usize, count: usize) -> bool {
    index >= first && (index - first) / stride < count && (index - first) % stride < rows
}

#[test]
fn final_scan_visits_every_logical_complement_once_and_no_active_tile() {
    // Enumerate the ACTUAL WGSL group/local/strided indexing independently of
    // the interval formula: producer sets contain all original matrix tiles.
    for count in 1..=3 {
        for (matrix_rows, first, gap) in [(9usize, 0usize, 0usize), (17, 7, 11), (248320, 3, 9)] {
            let stride = matrix_rows + gap;
            let words = first + 8 * stride + 13;
            let mut producers = vec![false; words];
            for token in 0..count {
                for row in 0..matrix_rows {
                    producers[first + token * stride + row] = true;
                }
            }
            for tile_rows in [1usize, 7, 2560, 16384] {
                let tile_rows = tile_rows.min(matrix_rows);
                let spec = MatrixSpec::new(tile_rows, 256, 14).unwrap();
                let groups = verify3_finite_grid(spec, words, true);
                assert!((1..=4096).contains(&groups));
                assert_eq!(
                    verify3_finite_grid(spec, words, false),
                    tile_rows.div_ceil(4)
                );
                let mut scanned = vec![0u8; words];
                for group in 0..groups {
                    for local in 0..128 {
                        for i in (group * 128 + local..words).step_by(128 * groups) {
                            if !active(i, first, stride, matrix_rows, count) {
                                scanned[i] += 1;
                            }
                        }
                    }
                }
                for (i, &visits) in scanned.iter().enumerate() {
                    assert_eq!(visits, u8::from(!producers[i]));
                }
            }
        }
    }
}

#[test]
fn active_self_checks_plus_complement_equal_full_finite_or_with_sticky_prior() {
    for count in 1..=3 {
        let (rows, first, stride, words) = (17usize, 7usize, 29usize, 7 + 8 * 29 + 13);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in 0..words {
                let mut values = vec![0.5f32; words];
                values[index] = bad;
                let mut full = 4u32;
                let mut fused = 4u32;
                for &value in &values {
                    if !value.is_finite() {
                        full |= 1;
                    }
                }
                // All original tiles self-check their own stores, not the
                // prior contents of any active destination. Final complement
                // never reads an active row produced by a different tile.
                for token in 0..count {
                    for tile in (0..rows).step_by(5) {
                        for row in tile..(tile + 5).min(rows) {
                            if !values[first + token * stride + row].is_finite() {
                                fused |= 1;
                            }
                        }
                    }
                }
                for (i, &value) in values.iter().enumerate() {
                    if !active(i, first, stride, rows, count) && !value.is_finite() {
                        fused |= 1;
                    }
                }
                assert_eq!(fused, full);
                assert_eq!(fused, 5);
            }
        }
    }
}

#[test]
fn full_interval_requires_exact_last_tile_and_complete_count_geometry() {
    let spec = MatrixSpec::new(7, 256, 14).unwrap();
    let bytes = (spec.rows * spec.row_bytes).div_ceil(4) as u64 * 4;
    let full = BatchSpans::new(1, 0, 4, 268, 7, 17);
    assert!(
        validate_verify3_finite_spans(
            spec,
            &full,
            bytes,
            4 + 8 * 268,
            VERIFY3_FINITE_WORD_LIMIT + 1,
            7,
            7,
            true
        )
        .is_err()
    );
    for count in 1..=3 {
        let spans = BatchSpans::new(count, 0, 4, 268, 17, 29);
        for (rows, first, tail) in [(17, 7, true), (18, 7, false)] {
            validate_verify3_finite_spans(
                spec,
                &spans,
                bytes,
                4 + 8 * 268,
                7 + 8 * 29,
                rows,
                first,
                tail,
            )
            .unwrap();
        }
        for (rows, first, tail) in [
            (17, 7, false),
            (18, 7, true),
            (16, 7, true),
            (0, 7, false),
            (30, 7, false),
            (17, 18, true),
            (usize::MAX, 7, true),
            (17, usize::MAX, true),
        ] {
            assert!(
                validate_verify3_finite_spans(
                    spec,
                    &spans,
                    bytes,
                    4 + 8 * 268,
                    7 + 8 * 29,
                    rows,
                    first,
                    tail
                )
                .is_err()
            );
        }
        assert!(
            validate_verify3_finite_spans(
                spec,
                &spans,
                bytes,
                4 + 8 * 268,
                7 + (count - 1) * 29 + 16,
                17,
                7,
                true
            )
            .is_err()
        );
    }
}

#[test]
fn finite_shader_keeps_original_math_and_runtime_intervals_and_valid_spirv() {
    for dtype in [0, 1, 30, 6, 8, 12, 13, 14, 20, 23] {
        let spec = MatrixSpec::new(7, 256, dtype).unwrap();
        for tail in [false, true] {
            let kernel = Kernel::MatrixVerify3Finite(spec, tail);
            let source = kernel.source();
            // Exact inherited producer body, with only guarded finite coverage
            // outside it. No projection/reduction expression is rewritten.
            let original = spec.source_verify3();
            let start = "    let row=group.x*(128u/width)+sg;";
            let end = "    // VERIFY3_LOGICAL_COMPLEMENT_SCAN";
            assert_eq!(
                source
                    .split_once(start)
                    .unwrap()
                    .1
                    .split_once(end)
                    .unwrap()
                    .0,
                original
                    .split_once(start)
                    .unwrap()
                    .1
                    .split_once(end)
                    .unwrap()
                    .0
            );
            assert!(source.contains("p.output_words==arrayLength(&output)"));
            assert!(source.contains("p.output_stride>=p.matrix_rows"));
            assert_eq!(source.contains("i+=128u*dispatched.x"), tail);
            if tail {
                assert!(source.contains(
                    "relative/p.output_stride<p.count && relative%p.output_stride<p.matrix_rows"
                ));
            }
            assert!(!source.contains("fma("));
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::SUBGROUP
                    | wgpu::naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
            )
            .validate(&module)
            .unwrap();
            let words = wgpu::naga::back::spv::write_vec(
                &module,
                &info,
                &wgpu::naga::back::spv::Options::default(),
                None,
            )
            .unwrap();
            assert!(!words.is_empty());
            assert_eq!(module.global_variables.len(), 5);
            assert_ne!(kernel.id(), Kernel::MatrixVerify3(spec).id());
            assert_ne!(kernel.id(), Kernel::MatrixVerify3Finite(spec, !tail).id());
        }
    }
}

#[test]
fn finite_info_remains_one_owned_record_but_requires_full_uniform_extent() {
    let mut limits = wgpu::Limits {
        max_uniform_buffer_binding_size: 32,
        ..Default::default()
    };
    validate_verify3_limits(&limits, true, 16, 32).unwrap();
    assert!(validate_verify3_finite_limits(&limits, true, 16, 32).is_err());
    limits.max_uniform_buffer_binding_size = 48;
    validate_verify3_finite_limits(&limits, true, 16, 32).unwrap();
    // Backend INFO records are separately rounded to2048, no new private owner.
    assert_eq!(48usize.div_ceil(2048), 32usize.div_ceil(2048));
}
