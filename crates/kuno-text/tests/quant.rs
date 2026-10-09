use kuno_text::quant::f16_to_f32;
use kuno_text::{Error, TensorType};

#[test]
fn binary16_matches_independent_ieee_equations_for_all_bit_patterns() {
    for bits in 0..=u16::MAX {
        let sign = if bits & 0x8000 != 0 { -1.0_f64 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 31);
        let fraction = f64::from(bits & 1023);
        let expected = match exponent {
            0 => sign * fraction * 2.0_f64.powi(-24),
            31 if fraction == 0.0 => sign * f64::INFINITY,
            31 => f64::NAN,
            _ => sign * (1.0 + fraction / 1024.0) * 2.0_f64.powi(exponent - 15),
        } as f32;
        let actual = f16_to_f32(bits);
        if expected.is_nan() {
            assert!(actual.is_nan(), "bits={bits:04x}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "bits={bits:04x}");
        }
    }
}

#[test]
fn plain_float_types_preserve_special_values_and_little_endian_layout() {
    let f32_values = [1.25_f32, -0.0, f32::INFINITY, f32::NEG_INFINITY];
    let packed: Vec<_> = f32_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut decoded = [0.0; 4];
    TensorType::F32.decode(&packed, &mut decoded).unwrap();
    assert_eq!(decoded.map(f32::to_bits), f32_values.map(f32::to_bits));

    let packed: Vec<_> = [0x3f80_u16, 0xbf80, 0x8000, 0x7f80]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    TensorType::Bf16.decode(&packed, &mut decoded).unwrap();
    assert_eq!(
        decoded.map(f32::to_bits),
        [1.0_f32, -1.0, -0.0, f32::INFINITY].map(f32::to_bits)
    );

    let packed: Vec<_> = [0x3c00_u16, 0xbc00, 0x0001, 0x8000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    TensorType::F16.decode(&packed, &mut decoded).unwrap();
    assert_eq!(
        decoded.map(f32::to_bits),
        [1.0_f32, -1.0, 2.0_f32.powi(-24), -0.0].map(f32::to_bits)
    );
}

fn pack_q5(values: &[i8; 32], scale: u16) -> [u8; 22] {
    let mut block = [0_u8; 22];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    let mut high = 0_u32;
    for (index, &value) in values.iter().enumerate() {
        let value = (i16::from(value) + 16) as u8;
        assert!(value < 32);
        high |= u32::from(value >> 4) << index;
        block[6 + index % 16] |= (value & 15) << (4 * (index / 16));
    }
    block[2..6].copy_from_slice(&high.to_le_bytes());
    block
}

#[test]
fn q5_0_respects_high_bit_word_nibble_halves_and_negative_offsets() {
    let values = std::array::from_fn(|index| index as i8 - 16);
    let packed = pack_q5(&values, 0x3c00);
    assert_eq!(&packed[2..6], &0xffff_0000_u32.to_le_bytes());
    let mut decoded = [0.0; 32];
    TensorType::Q5_0.decode(&packed, &mut decoded).unwrap();
    assert_eq!(decoded, values.map(f32::from));
}

#[test]
fn q5_0_ple_row_is_five_complete_blocks_and_110_bytes() {
    assert_eq!(TensorType::Q5_0.row_bytes(160).unwrap(), 110);
    let mut packed = Vec::new();
    let mut expected = Vec::new();
    for block in 0..5 {
        let values = std::array::from_fn(|index| ((index * 13 + block * 7) % 32) as i8 - 16);
        packed.extend(pack_q5(&values, 0x3800));
        expected.extend(values.map(|value| f32::from(value) * 0.5));
    }
    let mut decoded = [0.0; 160];
    TensorType::Q5_0.decode(&packed, &mut decoded).unwrap();
    assert_eq!(decoded.as_slice(), expected);
}

#[test]
fn q8_0_sign_extends_int8_before_scaling() {
    let mut packed = vec![0x00, 0x34]; // binary16 0.25
    let values: [i8; 32] = std::array::from_fn(|index| (index as i16 * 8 - 128) as i8);
    packed.extend(values.map(|value| value as u8));
    let mut decoded = [0.0; 32];
    TensorType::Q8_0.decode(&packed, &mut decoded).unwrap();
    assert_eq!(decoded, values.map(|value| f32::from(value) * 0.25));
}

#[test]
fn q4_k_all_eight_scale_min_fields_and_nibble_halves_round_trip() {
    let scales = [1_u8, 9, 17, 33, 49, 63, 35, 20];
    let minima = [2_u8, 8, 18, 34, 50, 62, 36, 21];
    let mut packed = [0_u8; 144];
    packed[..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // d=0.5
    packed[2..4].copy_from_slice(&0x3400_u16.to_le_bytes()); // dmin=0.25
    for group in 0..8 {
        if group < 4 {
            packed[4 + group] |= scales[group];
            packed[8 + group] |= minima[group];
        } else {
            packed[8 + group] = (scales[group] & 15) | ((minima[group] & 15) << 4);
            packed[group] |= (scales[group] >> 4) << 6;
            packed[4 + group] |= (minima[group] >> 4) << 6;
        }
    }
    let quants: [u8; 256] = std::array::from_fn(|index| ((index * 7 + index / 32) % 16) as u8);
    for (index, &quant) in quants.iter().enumerate() {
        let group = index / 32;
        packed[16 + (group / 2) * 32 + index % 32] |= quant << ((group % 2) * 4);
    }
    let mut decoded = [0.0; 256];
    TensorType::Q4K.decode(&packed, &mut decoded).unwrap();
    for index in 0..256 {
        let expected = 0.5 * f32::from(scales[index / 32]) * f32::from(quants[index])
            - 0.25 * f32::from(minima[index / 32]);
        assert_eq!(decoded[index], expected, "index={index}");
    }
}

#[test]
fn q6_k_all_quarters_signed_scales_and_high_bits_round_trip() {
    let mut packed = [0_u8; 210];
    packed[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
    let scales: [i8; 16] = std::array::from_fn(|index| index as i8 - 8);
    for (index, &scale) in scales.iter().enumerate() {
        packed[192 + index] = scale as u8;
    }
    let values: [i8; 256] =
        std::array::from_fn(|index| ((index * 13 + index / 16) % 64) as i8 - 32);
    for (index, &value) in values.iter().enumerate() {
        let raw = (i16::from(value) + 32) as u8;
        let half = index / 128;
        let quarter = (index % 128) / 32;
        let lane = index % 32;
        packed[half * 64 + (quarter % 2) * 32 + lane] |= (raw & 15) << ((quarter / 2) * 4);
        packed[128 + half * 32 + lane] |= (raw >> 4) << (quarter * 2);
    }
    let mut decoded = [0.0; 256];
    TensorType::Q6K.decode(&packed, &mut decoded).unwrap();
    for index in 0..256 {
        assert_eq!(
            decoded[index],
            0.5 * f32::from(scales[index / 16]) * f32::from(values[index]),
            "index={index}"
        );
    }
}

#[test]
fn packing_sizes_and_ids_match_actual_artifact_formats() {
    for (kind, id, elements, bytes) in [
        (TensorType::F32, 0, 1, 4),
        (TensorType::F16, 1, 1, 2),
        (TensorType::Q5_0, 6, 32, 22),
        (TensorType::Q8_0, 8, 32, 34),
        (TensorType::Q4K, 12, 256, 144),
        (TensorType::Q6K, 14, 256, 210),
        (TensorType::Iq3Xxs, 18, 256, 98),
        (TensorType::Iq4Nl, 20, 32, 18),
        (TensorType::Iq4Xs, 23, 256, 136),
        (TensorType::Bf16, 30, 1, 2),
    ] {
        assert_eq!(kind.gguf_id(), id);
        assert_eq!(TensorType::from_gguf_id(id).unwrap(), kind);
        assert_eq!(kind.block_elements(), elements);
        assert_eq!(kind.block_bytes(), bytes);
        assert_eq!(kind.row_bytes(elements * 5).unwrap(), bytes * 5);
    }
    assert!(matches!(
        TensorType::from_gguf_id(999),
        Err(Error::UnsupportedTensorType(999))
    ));
    assert!(matches!(
        TensorType::F32.row_bytes(u64::MAX),
        Err(Error::ArithmeticOverflow)
    ));
}

#[test]
fn malformed_lengths_fail_before_mutating_the_output() {
    let mut output = [1234.0; 32];
    assert!(matches!(
        TensorType::Q5_0.decode(&[0; 21], &mut output),
        Err(Error::InvalidBufferLength)
    ));
    assert_eq!(output, [1234.0; 32]);
    assert!(TensorType::Q4K.decode(&[0; 144], &mut output).is_err());
    assert_eq!(output, [1234.0; 32]);
    assert!(TensorType::Q5_0.row_bytes(160 + 1).is_err());
    assert!(TensorType::F32.decode(&[], &mut []).is_err());
}

// These public bytes and F32 bit patterns were generated by compiling the
// SHA256-pinned upstream C dequantizers, independently of Kuno's decoder.
// See fixtures/iq-b11232/README.md for complete codebook/field coverage.
const IQ_ORACLES: [(TensorType, &[u8]); 3] = [
    (
        TensorType::Iq3Xxs,
        include_bytes!("fixtures/iq-b11232/iq3_xxs.bin"),
    ),
    (
        TensorType::Iq4Nl,
        include_bytes!("fixtures/iq-b11232/iq4_nl.bin"),
    ),
    (
        TensorType::Iq4Xs,
        include_bytes!("fixtures/iq-b11232/iq4_xs.bin"),
    ),
];

#[test]
fn iq_scalar_decode_matches_pinned_independent_c_bits_for_every_fixture_lane() {
    for (kind, fixture) in IQ_ORACLES {
        let bytes = kind.block_bytes() as usize;
        let elements = kind.block_elements() as usize;
        let record_bytes = bytes + elements * 4;
        assert!(fixture.len().is_multiple_of(record_bytes));
        let mut packed = Vec::new();
        let mut expected = Vec::new();
        for (record, block) in fixture.chunks_exact(record_bytes).enumerate() {
            let mut actual = vec![f32::NAN; elements];
            kind.decode(&block[..bytes], &mut actual).unwrap();
            for (lane, (actual, expected)) in actual
                .iter()
                .zip(block[bytes..].chunks_exact(4))
                .enumerate()
            {
                let bits = u32::from_le_bytes(expected.try_into().unwrap());
                assert_eq!(
                    actual.to_bits(),
                    bits,
                    "{kind:?}, record={record}, lane={lane}"
                );
            }
            packed.extend_from_slice(&block[..bytes]);
            expected.extend_from_slice(&block[bytes..]);
        }
        // Whole rows must also preserve block boundaries and output ordering.
        let mut actual = vec![f32::NAN; expected.len() / 4];
        kind.decode(&packed, &mut actual).unwrap();
        for (actual, expected) in actual.iter().zip(expected.chunks_exact(4)) {
            assert_eq!(
                actual.to_bits(),
                u32::from_le_bytes(expected.try_into().unwrap())
            );
        }
    }
}

#[test]
fn iq_malformed_rows_fail_before_any_output_write() {
    for (kind, _) in IQ_ORACLES {
        let elements = kind.block_elements() as usize;
        let bytes = kind.block_bytes() as usize;
        for packed_length in [0, bytes - 1, bytes + 1, bytes * 2] {
            let mut output = vec![1234.5; elements];
            assert!(matches!(
                kind.decode(&vec![0; packed_length], &mut output),
                Err(Error::InvalidBufferLength)
            ));
            assert_eq!(output, vec![1234.5; elements]);
        }
        for output_length in [0, 1, elements - 1, elements + 1] {
            let mut output = vec![1234.5; output_length];
            assert!(matches!(
                kind.decode(&vec![0; bytes], &mut output),
                Err(Error::InvalidFormat(_))
            ));
            assert_eq!(output, vec![1234.5; output_length]);
        }
        // Quantized geometry at the largest whole row remains exact without
        // allocating a payload or expanding weights into caller storage.
        let whole = u64::MAX - (u64::MAX % kind.block_elements());
        assert_eq!(
            kind.row_bytes(whole).unwrap(),
            whole / kind.block_elements() * kind.block_bytes()
        );
    }
}
