const IQ_KINDS: [TensorType; 3] = [TensorType::Iq3Xxs, TensorType::Iq4Nl, TensorType::Iq4Xs];

fn oracle_records(kind: TensorType) -> &'static [u8] {
    match kind {
        TensorType::Iq3Xxs => include_bytes!("../tests/fixtures/iq-b11232/iq3_xxs.bin"),
        TensorType::Iq4Nl => include_bytes!("../tests/fixtures/iq-b11232/iq4_nl.bin"),
        TensorType::Iq4Xs => include_bytes!("../tests/fixtures/iq-b11232/iq4_xs.bin"),
        _ => unreachable!(),
    }
}

#[test]
fn iq_simd_decodes_all8032_independent_c_values_bit_exactly() {
    if !avx2_available() {
        return;
    }
    let mut values = 0;
    let cancel = AtomicBool::new(false);
    for kind in IQ_KINDS {
        let bytes = kind.block_bytes() as usize;
        let width = kind.block_elements() as usize;
        let records = oracle_records(kind);
        let stride = bytes + width * 4;
        assert!(records.len().is_multiple_of(stride));
        for record in records.chunks_exact(stride) {
            let packed = &record[..bytes];
            let expected: Vec<_> = record[bytes..]
                .chunks_exact(4)
                .map(|bits| f32::from_le_bytes(bits.try_into().unwrap()))
                .collect();
            let mut decoded = vec![f32::from_bits(0x7fc0_1234); width];
            avx::test_iq_decode(kind, packed, &mut decoded).unwrap();
            same_bits(&decoded, &expected);
            // Check the actual direct-dot consumer too, including signed zeros,
            // both subnormal boundaries and the largest finite binary16 scale.
            let x = input(width);
            let reference = reference_dot(&expected, &x);
            for backend in [Backend::Avx2, Backend::Avx2Fma] {
                if backend == Backend::Avx2Fma && !fma_available() {
                    continue;
                }
                let mut scratch = Scratch::new(width).unwrap();
                scratch.block.fill(f32::from_bits(0x7fc0_4567));
                let got = dot_row(backend, kind, packed, &x, &mut scratch, &cancel).unwrap();
                assert_eq!(got.to_bits(), reference.to_bits());
                assert_eq!(scratch.decoded.capacity(), 0);
                assert!(scratch.block.iter().all(|v| v.to_bits() == 0x7fc0_4567));
            }
            values += width;
        }
    }
    assert_eq!(values, 8032);
}

#[test]
fn iq_simd_bad_lengths_and_scales_leave_the_complete_output_unchanged() {
    if !avx2_available() {
        return;
    }
    for kind in IQ_KINDS {
        let (packed, _) = block(kind, 0);
        let width = kind.block_elements() as usize;
        let sentinel = f32::from_bits(0x7fc0_1234);
        for packed_len in 0..=packed.len() + 1 {
            if packed_len == packed.len() {
                continue;
            }
            let mut malformed = packed.clone();
            malformed.resize(packed_len, 91);
            let mut out = vec![sentinel; width];
            assert!(matches!(
                avx::test_iq_decode(kind, &malformed, &mut out),
                Err(Error::InvalidBufferLength)
            ));
            assert!(out.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        }
        for output_len in [0, width - 1, width + 1] {
            let mut out = vec![sentinel; output_len];
            assert!(matches!(
                avx::test_iq_decode(kind, &packed, &mut out),
                Err(Error::InvalidBufferLength)
            ));
            assert!(out.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        }
        for half_bits in [0x7c00u16, 0xfc00, 0x7e01] {
            let mut malformed = packed.clone();
            malformed[..2].copy_from_slice(&half_bits.to_le_bytes());
            let mut out = vec![sentinel; width];
            assert!(avx::test_iq_decode(kind, &malformed, &mut out).is_err());
            assert!(out.iter().all(|v| v.to_bits() == sentinel.to_bits()));
            for backend in [Backend::Avx2, Backend::Avx2Fma] {
                if backend == Backend::Avx2Fma && !fma_available() {
                    continue;
                }
                let mut scratch = Scratch::new(width).unwrap();
                scratch.block.fill(sentinel);
                assert!(
                    dot_row(
                        backend,
                        kind,
                        &malformed,
                        &input(width),
                        &mut scratch,
                        &AtomicBool::new(false),
                    )
                    .is_err()
                );
                assert!(
                    scratch
                        .block
                        .iter()
                        .all(|v| v.to_bits() == sentinel.to_bits())
                );
                assert_eq!(scratch.decoded.capacity(), 0);
            }
        }
    }
}

#[test]
fn iq_native_shaped_matrix_rows_match_c_oracle_and_old_block_path() {
    if !avx2_available() {
        return;
    }
    let cancel = AtomicBool::new(false);
    // Original Ornith hidden/FF widths2048/512, Flash-Next2560/640. IQ3/XS
    // original256-value geometry is used only where those widths admit it.
    for (kind, width) in [
        (TensorType::Iq3Xxs, 2048),
        (TensorType::Iq3Xxs, 2560),
        (TensorType::Iq4Xs, 2048),
        (TensorType::Iq4Xs, 2560),
        (TensorType::Iq4Nl, 512),
        (TensorType::Iq4Nl, 640),
    ] {
        let x = input(width);
        let mut old_scratch = Scratch::new(width).unwrap();
        let mut scratch = Scratch::new(width).unwrap();
        for row in 0..17 {
            let (packed, weights) = encoded_row(kind, width, row * 13 + 7);
            let expected = reference_dot(&weights, &x);
            let old = avx::test_decoded_dot(kind, &packed, &x, &mut old_scratch, &cancel).unwrap();
            assert_eq!(old.to_bits(), expected.to_bits());
            for backend in [Backend::Avx2, Backend::Avx2Fma] {
                if backend == Backend::Avx2Fma && !fma_available() {
                    continue;
                }
                let got = dot_row(backend, kind, &packed, &x, &mut scratch, &cancel).unwrap();
                assert_eq!(got.to_bits(), expected.to_bits());
                assert_eq!(scratch.decoded.capacity(), 0);
            }
        }
    }
}

#[test]
fn iq_native_shaped_complete_experts_preserve_c_oracle_ranks_and_capacity() {
    if !avx2_available() {
        return;
    }
    let cancel = AtomicBool::new(false);
    for (hidden, ff, gate_up) in [
        (2048, 512, TensorType::Iq3Xxs),
        (2560, 640, TensorType::Iq3Xxs),
        (2560, 640, TensorType::Iq4Xs),
    ] {
        let f = fixture([gate_up, gate_up, TensorType::Iq4Nl], hidden, ff, 2);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let x = input(hidden);
        let expected = expected_experts(&f, &[1, 0], &x);
        for backend in [Backend::Portable, Backend::Avx2, Backend::Avx2Fma] {
            if backend == Backend::Avx2Fma && !fma_available() {
                continue;
            }
            let op = ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), backend).unwrap();
            let got = op.execute(&weights, &[1, 0], &x, &cancel).unwrap();
            same_bits(got.active(), &expected);
            assert!(got.full()[2 * hidden..].iter().all(|v| v.to_bits() == 0));
            assert!(got.owned_bytes() <= MAX_PRIVATE_HOST_BYTES);
        }
    }
}

#[test]
#[ignore = "fixed-public IQ packed CPU-only comparison; not whole-model throughput"]
fn iq_packed_cpu_public_timing_probe() {
    assert!(avx2_available(), "explicit probe requires AVX2");
    use std::{hint::black_box, time::Instant};
    const TRIALS: usize = 7;
    const REPEATS: usize = 4;
    let cancel = AtomicBool::new(false);
    for (kind, columns, rows) in [
        (TensorType::Iq3Xxs, 2560, 640),
        (TensorType::Iq4Xs, 2560, 640),
        (TensorType::Iq4Nl, 640, 2560),
    ] {
        let mut matrix = Vec::new();
        for row in 0..rows {
            matrix.extend(encoded_row(kind, columns, row * 13 + 7).0);
        }
        let row_bytes = kind.row_bytes(columns as u64).unwrap() as usize;
        let x = input(columns);
        let mut scratch = Scratch::new(columns).unwrap();
        let mut candidate = |old: bool| {
            let mut checksum = 0u32;
            for _ in 0..REPEATS {
                for packed in matrix.chunks_exact(row_bytes) {
                    let got = if old {
                        avx::test_decoded_dot(kind, black_box(packed), &x, &mut scratch, &cancel)
                    } else {
                        dot_row(
                            Backend::Avx2,
                            kind,
                            black_box(packed),
                            &x,
                            &mut scratch,
                            &cancel,
                        )
                    }
                    .unwrap();
                    checksum = checksum.wrapping_add(black_box(got).to_bits());
                }
            }
            black_box(checksum)
        };
        assert_eq!(candidate(true), candidate(false));
        let mut old_ns = [0u128; TRIALS];
        let mut new_ns = [0u128; TRIALS];
        for trial in 0..TRIALS {
            let mut hashes = [0u32; 2];
            // Alternate the order to avoid a persistent first-run bias.
            for old in [trial % 2 == 0, trial % 2 != 0] {
                let start = Instant::now();
                hashes[usize::from(old)] = candidate(old);
                let elapsed = start.elapsed().as_nanos();
                if old {
                    old_ns[trial] = elapsed;
                } else {
                    new_ns[trial] = elapsed;
                }
            }
            assert_eq!(hashes[0], hashes[1]);
        }
        println!(
            "fixed_public_iq_cpu ggml_id={} columns={} rows={} packed_bytes={} repeats={} old_ns={:?} new_ns={:?}",
            kind.gguf_id(),
            columns,
            rows,
            matrix.len(),
            REPEATS,
            old_ns,
            new_ns
        );
        old_ns.sort_unstable();
        new_ns.sort_unstable();
        println!(
            "fixed_public_iq_cpu ggml_id={} old_median_ns={} new_median_ns={} speedup={:.4}",
            kind.gguf_id(),
            old_ns[TRIALS / 2],
            new_ns[TRIALS / 2],
            old_ns[TRIALS / 2] as f64 / new_ns[TRIALS / 2] as f64
        );
    }
}
