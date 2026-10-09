const DASLAB_KINDS: [TensorType; 4] = [
    TensorType::Iq2S,
    TensorType::Iq3S,
    TensorType::Q5K,
    TensorType::Q2_0,
];

fn oracle_records(kind: TensorType) -> &'static [u8] {
    match kind {
        TensorType::Iq2S => include_bytes!("../tests/fixtures/daslab-b11232/iq2_s.bin"),
        TensorType::Iq3S => include_bytes!("../tests/fixtures/daslab-b11232/iq3_s.bin"),
        TensorType::Q5K => include_bytes!("../tests/fixtures/daslab-b11232/q5_K.bin"),
        TensorType::Q2_0 => include_bytes!("../tests/fixtures/daslab-b11232/q2_0.bin"),
        _ => unreachable!(),
    }
}

#[test]
fn daslab_simd_decodes_all20032_independent_c_values_bit_exactly() {
    if !avx2_available() {
        return;
    }
    let mut values = 0;
    let cancel = AtomicBool::new(false);
    for kind in DASLAB_KINDS {
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
            avx::test_daslab_decode(kind, packed, &mut decoded).unwrap();
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
    assert_eq!(values, 20032);
}

#[test]
fn daslab_simd_bad_lengths_and_scales_leave_the_complete_output_unchanged() {
    if !avx2_available() {
        return;
    }
    for kind in DASLAB_KINDS {
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
                avx::test_daslab_decode(kind, &malformed, &mut out),
                Err(Error::InvalidBufferLength)
            ));
            assert!(out.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        }
        for output_len in [0, width - 1, width + 1] {
            let mut out = vec![sentinel; output_len];
            assert!(matches!(
                avx::test_daslab_decode(kind, &packed, &mut out),
                Err(Error::InvalidBufferLength)
            ));
            assert!(out.iter().all(|v| v.to_bits() == sentinel.to_bits()));
        }
        let offsets: &[usize] = if kind == TensorType::Q5K {
            &[0, 2]
        } else {
            &[0]
        };
        for &offset in offsets {
            for half_bits in [0x7c00u16, 0xfc00, 0x7e01] {
                let mut malformed = packed.clone();
                malformed[offset..offset + 2].copy_from_slice(&half_bits.to_le_bytes());
                let mut out = vec![sentinel; width];
                assert!(avx::test_daslab_decode(kind, &malformed, &mut out).is_err());
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
}

#[test]
fn daslab_native_shaped_matrix_rows_match_c_oracle_and_old_block_path() {
    if !avx2_available() {
        return;
    }
    let cancel = AtomicBool::new(false);
    // Actual hidden2560 admits256-value formats; actual down width640
    // admits Q2_0's64-value blocks. Other lengths check independent rows.
    for (kind, width) in [
        (TensorType::Iq2S, 2048),
        (TensorType::Iq2S, 2560),
        (TensorType::Iq3S, 2048),
        (TensorType::Iq3S, 2560),
        (TensorType::Q5K, 2048),
        (TensorType::Q5K, 2560),
        (TensorType::Q2_0, 512),
        (TensorType::Q2_0, 640),
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
fn daslab_native_shaped_complete_experts_preserve_c_oracle_ranks_and_capacity() {
    if !avx2_available() {
        return;
    }
    let cancel = AtomicBool::new(false);
    for (hidden, ff, gate_up) in [
        (2048, 512, TensorType::Iq2S),
        (2560, 640, TensorType::Iq3S),
        (2560, 640, TensorType::Q5K),
    ] {
        let f = fixture([gate_up, gate_up, TensorType::Q2_0], hidden, ff, 2);
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
fn new_daslab_formats_never_implicitly_select_existing_q8_activation_math() {
    let cancel = AtomicBool::new(false);
    for kind in DASLAB_KINDS {
        let width = kind.block_elements() as usize;
        let mut activation = quantized::Input::empty();
        activation.fill(&vec![0.25; width], &cancel).unwrap();
        assert!(!quantized::supported(kind));
        assert!(matches!(
            quantized::dot(
                kind,
                &vec![0; kind.block_bytes() as usize],
                &activation,
                &cancel
            ),
            Err(Error::InvalidFormat("CPU quantized format unsupported"))
        ));
    }
}
