//! Independent host oracle for the explicit packed-IQ/two-Q8-plane operator.
//! Scalar original byte packing is checked against separately compiled pinned
//! GGML C values; integer sums and f64 products do not reuse GPU shader math.
//! Activation codes/scales are complete erased host owners, never persisted.
use kuno_text::{Error, Result, TensorType};
use zeroize::{Zeroize, Zeroizing};

struct Block {
    first: [i8; 32],
    second: [i8; 32],
    scale: f32,
    residual_scale: f32,
}
impl Zeroize for Block {
    fn zeroize(&mut self) {
        self.first.zeroize();
        self.second.zeroize();
        self.scale.zeroize();
        self.residual_scale.zeroize();
    }
}
pub(crate) struct Planes {
    blocks: Zeroizing<Vec<Block>>,
}
fn rejected() -> Error {
    Error::InvalidFormat("fixed packed-IQ integer oracle")
}
fn code(x: f32, scale: f32) -> i8 {
    if scale == 0. {
        0
    } else {
        (x / scale).clamp(-127., 127.).round_ties_even() as i8
    }
}
impl Planes {
    pub(crate) fn new(input: &[f32]) -> Result<Self> {
        if input.is_empty()
            || input.len() > 2560
            || !input.len().is_multiple_of(32)
            || input.iter().any(|v| !v.is_finite() || v.abs() == f32::MAX)
        {
            return Err(rejected());
        }
        let count = input.len() / 32;
        let mut blocks = Zeroizing::new(Vec::with_capacity(count));
        if blocks.capacity() != count {
            return Err(rejected());
        }
        for values in input.chunks_exact(32) {
            let mut block = Zeroizing::new(Block {
                first: [0; 32],
                second: [0; 32],
                scale: 0.,
                residual_scale: 0.,
            });
            let maximum = values.iter().map(|v| v.abs()).fold(0f32, f32::max);
            block.scale = maximum / 127.;
            let mut residual = Zeroizing::new([0f32; 32]);
            for i in 0..32 {
                block.first[i] = code(values[i], block.scale);
                residual[i] = values[i] - f32::from(block.first[i]) * block.scale;
                if !residual[i].is_finite() {
                    return Err(rejected());
                }
            }
            block.residual_scale = residual.iter().map(|v| v.abs()).fold(0f32, f32::max) / 127.;
            for i in 0..32 {
                block.second[i] = code(residual[i], block.residual_scale);
            }
            // Copy only into the already allocated erased owner; the temporary
            // block and all residual cells are wiped on every exit above/below.
            blocks.push(Block {
                first: block.first,
                second: block.second,
                scale: block.scale,
                residual_scale: block.residual_scale,
            });
        }
        Ok(Self { blocks })
    }
    pub(crate) fn columns(&self) -> usize {
        self.blocks.len() * 32
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Block>()
    }
    /// Independent diagnostic of the two-plane representation, not a bound
    /// on later weight projections or nonlinear whole-model amplification.
    pub(crate) fn relative_l2_error(&self, input: &[f32]) -> Result<f64> {
        if input.len() != self.columns() || input.iter().any(|v| !v.is_finite()) {
            return Err(rejected());
        }
        let mut error = 0f64;
        let mut reference = 0f64;
        for (i, &x) in input.iter().enumerate() {
            let block = &self.blocks[i / 32];
            let lane = i % 32;
            let reconstructed = f64::from(block.scale) * f64::from(block.first[lane])
                + f64::from(block.residual_scale) * f64::from(block.second[lane]);
            error += (f64::from(x) - reconstructed).powi(2);
            reference += f64::from(x).powi(2);
        }
        Ok(if reference == 0. {
            0.
        } else {
            (error / reference).sqrt()
        })
    }
}
fn half(bytes: &[u8], offset: usize) -> f32 {
    kuno_text::quant::f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}
/// Original natural-order32 integer weights plus their one F32 factor. The
/// full row and dtype are validated BEFORE reading or writing any scalar code.
pub(crate) fn unpack32(
    dtype: TensorType,
    bytes: &[u8],
    columns: usize,
    index: usize,
) -> Result<([i8; 32], f32)> {
    if !matches!(
        dtype,
        TensorType::Iq3Xxs
            | TensorType::Iq3S
            | TensorType::Iq4Nl
            | TensorType::Iq4Xs
            | TensorType::Q8_0
            | TensorType::Q2_0
    ) || columns == 0
        || columns > 2560
        || !columns.is_multiple_of(32)
        || index >= columns / 32
        || dtype.row_bytes(columns as u64)? as usize != bytes.len()
    {
        return Err(rejected());
    }
    let mut codes = [0i8; 32];
    let factor = match dtype {
        TensorType::Iq3Xxs => {
            let base = index / 8 * 98;
            let group = index % 8;
            let word = u32::from_le_bytes(
                bytes[base + 66 + group * 4..base + 70 + group * 4]
                    .try_into()
                    .unwrap(),
            );
            for octet in 0..4 {
                let sign_code = (word >> (octet * 7)) & 127;
                let signs = sign_code | ((sign_code.count_ones() & 1) << 7);
                for quartet in 0..2 {
                    let grid = kuno_text::quant::IQ3_XXS_GRID
                        [usize::from(bytes[base + 2 + group * 8 + octet * 2 + quartet])]
                    .to_le_bytes();
                    for (lane, value) in grid.into_iter().enumerate() {
                        let at = quartet * 4 + lane;
                        codes[octet * 8 + at] = if signs & (1 << at) == 0 {
                            value as i8
                        } else {
                            -(value as i8)
                        };
                    }
                }
            }
            half(bytes, base) * (0.5 + (word >> 28) as f32) * 0.5
        }
        TensorType::Iq3S => {
            let base = index / 8 * 110;
            let group = index % 8;
            let high = usize::from(bytes[base + 66 + group]);
            for quartet in 0..8 {
                let table = usize::from(bytes[base + 2 + group * 8 + quartet])
                    | (((high >> quartet) & 1) << 8);
                let grid = kuno_text::quant::IQ3_S_GRID[table].to_le_bytes();
                let signs = bytes[base + 74 + group * 4 + quartet / 2] >> ((quartet % 2) * 4);
                for (lane, magnitude) in grid.into_iter().enumerate() {
                    codes[quartet * 4 + lane] = if signs & (1 << lane) == 0 {
                        magnitude as i8
                    } else {
                        -(magnitude as i8)
                    };
                }
            }
            let scale = (bytes[base + 106 + group / 2] >> ((group % 2) * 4)) & 15;
            half(bytes, base) * f32::from(1 + 2 * scale)
        }
        TensorType::Iq4Nl | TensorType::Iq4Xs => {
            let (base, offset, factor) = if dtype == TensorType::Iq4Nl {
                let base = index * 18;
                (base, 2, half(bytes, base))
            } else {
                let base = index / 8 * 136;
                let group = index % 8;
                let low = (bytes[base + 4 + group / 2] >> (group % 2 * 4)) & 15;
                let high =
                    (u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]) >> (group * 2)) & 3;
                (
                    base,
                    8 + group * 16,
                    half(bytes, base) * f32::from(i16::from(low | (high as u8) << 4) - 32),
                )
            };
            for lane in 0..16 {
                let byte = bytes[base + offset + lane];
                codes[lane] = kuno_text::quant::IQ4_NL_VALUES[usize::from(byte & 15)];
                codes[lane + 16] = kuno_text::quant::IQ4_NL_VALUES[usize::from(byte >> 4)];
            }
            factor
        }
        TensorType::Q8_0 => {
            let base = index * 34;
            for lane in 0..32 {
                codes[lane] = bytes[base + 2 + lane] as i8;
            }
            half(bytes, base)
        }
        TensorType::Q2_0 => {
            let base = index / 2 * 18;
            let first = 2 + (index % 2) * 8;
            for (lane, code) in codes.iter_mut().enumerate() {
                *code = ((bytes[base + first + lane / 4] >> (2 * (lane % 4))) & 3) as i8 - 1;
            }
            half(bytes, base)
        }
        _ => return Err(rejected()),
    };
    if !factor.is_finite() {
        return Err(rejected());
    }
    Ok((codes, factor))
}
/// IQ2_S requires a distinct original subscale per16 weights. This entry never
/// flattens the two adjacent factors into one32-element activation block.
pub(crate) fn unpack_iq2_s16(
    bytes: &[u8],
    columns: usize,
    index: usize,
) -> Result<([i8; 16], f32)> {
    if columns == 0
        || columns > 2560
        || !columns.is_multiple_of(256)
        || index >= columns / 16
        || TensorType::Iq2S.row_bytes(columns as u64)? as usize != bytes.len()
    {
        return Err(rejected());
    }
    let base = index / 16 * 82;
    let group = (index % 16) / 2;
    let side = index % 2;
    let mut codes = [0i8; 16];
    for octet in 0..2 {
        let position = side * 2 + octet;
        let low = usize::from(bytes[base + 2 + group * 4 + position]);
        let high = (usize::from(bytes[base + 66 + group]) << (8 - 2 * position)) & 0x300;
        let grid = kuno_text::quant::IQ2_S_GRID[low | high].to_le_bytes();
        let signs = bytes[base + 34 + group * 4 + position];
        for (lane, magnitude) in grid.into_iter().enumerate() {
            codes[octet * 8 + lane] = if signs & (1 << lane) == 0 {
                magnitude as i8
            } else {
                -(magnitude as i8)
            };
        }
    }
    let scale = (bytes[base + 74 + group] >> (side * 4)) & 15;
    let factor = half(bytes, base) * (0.5 + f32::from(scale)) * 0.25;
    if !factor.is_finite() {
        return Err(rejected());
    }
    Ok((codes, factor))
}
pub(crate) fn project(dtype: TensorType, wire: &[u8], planes: &Planes) -> Result<f32> {
    let mut sum = 0f64;
    for (index, plane) in planes.blocks.iter().enumerate() {
        if dtype == TensorType::Iq2S {
            for side in 0..2 {
                let (weights, factor) = unpack_iq2_s16(wire, planes.columns(), index * 2 + side)?;
                let mut first = 0i64;
                let mut second = 0i64;
                for (i, &weight) in weights.iter().enumerate() {
                    first += i64::from(weight) * i64::from(plane.first[side * 16 + i]);
                    second += i64::from(weight) * i64::from(plane.second[side * 16 + i]);
                }
                if first.abs() > 16 * 128 * 127 || second.abs() > 16 * 128 * 127 {
                    return Err(rejected());
                }
                sum += f64::from(factor)
                    * (f64::from(plane.scale) * first as f64
                        + f64::from(plane.residual_scale) * second as f64);
            }
            continue;
        }
        let (weights, factor) = unpack32(dtype, wire, planes.columns(), index)?;
        let mut first = 0i64;
        let mut second = 0i64;
        for (i, &weight) in weights.iter().enumerate() {
            first += i64::from(weight) * i64::from(plane.first[i]);
            second += i64::from(weight) * i64::from(plane.second[i]);
        }
        if first.abs() > 32 * 128 * 127 || second.abs() > 32 * 128 * 127 {
            return Err(rejected());
        }
        sum += f64::from(factor)
            * (f64::from(plane.scale) * first as f64
                + f64::from(plane.residual_scale) * second as f64);
    }
    let result = sum as f32;
    if !result.is_finite() {
        return Err(rejected());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_matvec::iq_fixtures;
    #[test]
    fn all8032_c_values_match_finite_factors_or_are_explicitly_rejected() {
        let mut count = 0;
        for dtype in [TensorType::Iq3Xxs, TensorType::Iq4Nl, TensorType::Iq4Xs] {
            let (records, _) = iq_fixtures::records(dtype.gguf_id());
            let block = dtype.block_elements() as usize;
            let bytes = dtype.block_bytes() as usize;
            for record in records.chunks_exact(bytes + block * 4) {
                let values = record[bytes..]
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                for index in 0..block / 32 {
                    match unpack32(dtype, &record[..bytes], block, index) {
                        Ok((codes, scale)) => {
                            for (lane, &q) in codes.iter().enumerate() {
                                assert_eq!(
                                    (scale * f32::from(q)).to_bits(),
                                    values[index * 32 + lane].to_bits()
                                );
                            }
                        }
                        Err(_) => assert!(
                            values[index * 32..index * 32 + 32]
                                .iter()
                                .all(|v| !v.is_finite())
                        ),
                    }
                    count += 32;
                }
            }
        }
        assert_eq!(count, 8032);
    }
    #[test]
    fn all15680_das_c_values_match_original_half_scales_or_explicit_rejection() {
        let mut count = 0;
        for dtype in [TensorType::Iq2S, TensorType::Iq3S, TensorType::Q2_0] {
            let (records, _) = iq_fixtures::records(dtype.gguf_id());
            let block = dtype.block_elements() as usize;
            let bytes = dtype.block_bytes() as usize;
            let unit = if dtype == TensorType::Iq2S { 16 } else { 32 };
            for record in records.chunks_exact(bytes + block * 4) {
                let values = record[bytes..]
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>();
                for index in 0..block / unit {
                    let decoded = if dtype == TensorType::Iq2S {
                        unpack_iq2_s16(&record[..bytes], block, index)
                            .map(|(codes, scale)| (codes.to_vec(), scale))
                    } else {
                        unpack32(dtype, &record[..bytes], block, index)
                            .map(|(codes, scale)| (codes.to_vec(), scale))
                    };
                    match decoded {
                        Ok((codes, scale)) => {
                            for (lane, &q) in codes.iter().enumerate() {
                                assert_eq!(
                                    (scale * f32::from(q)).to_bits(),
                                    values[index * unit + lane].to_bits()
                                );
                            }
                        }
                        Err(_) => assert!(
                            values[index * unit..index * unit + unit]
                                .iter()
                                .all(|v| !v.is_finite())
                        ),
                    }
                    count += unit;
                }
            }
        }
        assert_eq!(count, 15680);
    }
    #[test]
    fn bytewise_iq3_negation_has_no_cross_byte_carry_for_all_grids_and_signs() {
        let grids = kuno_text::quant::IQ3_XXS_GRID
            .into_iter()
            .chain(kuno_text::quant::IQ3_S_GRID)
            .chain(
                kuno_text::quant::IQ2_S_GRID
                    .into_iter()
                    .flat_map(|word| [word as u32, (word >> 32) as u32]),
            );
        for grid in grids {
            for signs in 0..16u32 {
                let mask = ((signs & 1) * 255)
                    | ((((signs >> 1) & 1) * 255) << 8)
                    | ((((signs >> 2) & 1) * 255) << 16)
                    | ((((signs >> 3) & 1) * 255) << 24);
                let actual = ((grid ^ mask) + (mask & 0x01010101)).to_le_bytes();
                for (i, magnitude) in grid.to_le_bytes().into_iter().enumerate() {
                    assert!((1..=127).contains(&magnitude));
                    let expected = if signs & (1 << i) == 0 {
                        magnitude as i8
                    } else {
                        -(magnitude as i8)
                    };
                    assert_eq!(actual[i] as i8, expected);
                }
            }
        }
    }
    #[test]
    fn q8_planes_have_even_ties_residual_improvement_and_bounded_complete_owner() {
        let mut input = [0f32; 32];
        input[..9].copy_from_slice(&[127., -127., -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5]);
        let planes = Planes::new(&input).unwrap();
        assert_eq!(
            &planes.blocks[0].first[..9],
            &[127, -127, -2, -2, 0, 0, 2, 2, 126]
        );
        assert_eq!(planes.owned_bytes(), 72);
        for (i, &x) in input.iter().enumerate() {
            let b = &planes.blocks[0];
            let one = f32::from(b.first[i]) * b.scale;
            let two = one + f32::from(b.second[i]) * b.residual_scale;
            assert!((x - two).abs() <= (x - one).abs());
        }
        let zeros = Planes::new(&[0.; 2560]).unwrap();
        assert_eq!(zeros.owned_bytes(), 5760);
        assert!(zeros.blocks.iter().all(|b| b.scale == 0.
            && b.residual_scale == 0.
            && b.first == [0; 32]
            && b.second == [0; 32]));
        assert!(Planes::new(&[0.; 31]).is_err());
        assert!(Planes::new(&[0.; 2592]).is_err());
        input[31] = f32::NAN;
        assert!(Planes::new(&input).is_err());
        assert!(Planes::new(&[f32::MAX; 32]).is_err());
        assert!(Planes::new(&[-f32::MAX; 32]).is_err());
    }
    #[test]
    fn two_plane_normal_range_relative_l2_respects_independent_residual_bound() {
        // Exact-rounding bound per32: first residual <=amax/(2*127),
        // second residual <=amax/(4*127^2), hence relativeL2 <=sqrt(32)
        // /(4*127^2). The explicit1e-4 allowance includes F32 round points.
        let exact_bound = 32f64.sqrt() / (4. * 127f64.powi(2));
        assert!(exact_bound < 1e-4);
        for magnitude in [1e-30f32, 1e-10, 1., 1e10, 1e30] {
            let input = Zeroizing::new(
                (0..2560)
                    .map(|i| ((i * 19 % 43) as f32 - 21.) / 21. * magnitude)
                    .collect::<Vec<_>>(),
            );
            let planes = Planes::new(&input).unwrap();
            assert!(planes.relative_l2_error(&input).unwrap() < 1e-4);
            assert!(planes.relative_l2_error(&input[..32]).is_err());
        }
        assert_eq!(
            Planes::new(&[0.; 32])
                .unwrap()
                .relative_l2_error(&[0.; 32])
                .unwrap(),
            0.
        );
        // Underflow is an explicit zero-code policy, outside the normal-range
        // analytic bound; do not hide this loss behind a norm floor.
        let tiny = [f32::from_bits(1); 32];
        assert_eq!(
            Planes::new(&tiny)
                .unwrap()
                .relative_l2_error(&tiny)
                .unwrap(),
            1.
        );
    }
    #[test]
    fn packed_integer_projection_matches_independent_c_weights_and_dequantized_planes() {
        for (dtype, columns) in [
            (TensorType::Iq3Xxs, 2560usize),
            (TensorType::Iq4Nl, 640),
            (TensorType::Iq4Xs, 2560),
            (TensorType::Iq2S, 2560),
            (TensorType::Iq3S, 2560),
            (TensorType::Q2_0, 640),
        ] {
            let wire = iq_fixtures::packed(dtype.gguf_id(), 4, columns, 1);
            let weights = iq_fixtures::oracle(dtype.gguf_id(), 4, columns, 1);
            let input = Zeroizing::new(
                (0..columns)
                    .map(|i| ((i * 17 % 37) as f32 - 18.) / 37.)
                    .collect::<Vec<_>>(),
            );
            let planes = Planes::new(&input).unwrap();
            let row = dtype.row_bytes(columns as u64).unwrap() as usize;
            for r in 0..4 {
                let got = project(dtype, &wire[r * row..(r + 1) * row], &planes).unwrap();
                let expected = weights[r * columns..(r + 1) * columns]
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        let b = &planes.blocks[i / 32];
                        let lane = i % 32;
                        f64::from(w)
                            * (f64::from(b.scale) * f64::from(b.first[lane])
                                + f64::from(b.residual_scale) * f64::from(b.second[lane]))
                    })
                    .sum::<f64>() as f32;
                assert!((got - expected).abs() <= 1e-5 + 1e-6 * expected.abs());
            }
        }
        let planes = Planes::new(&[127.; 32]).unwrap();
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        q8[2..].fill(128);
        assert_eq!(
            project(TensorType::Q8_0, &q8, &planes).unwrap(),
            -128. * 127. * 32.
        );
        q8[2..].fill(127);
        assert_eq!(
            project(TensorType::Q8_0, &q8, &planes).unwrap(),
            127. * 127. * 32.
        );
    }
    #[test]
    fn every_short_malformed_original_record_rejects_before_decode() {
        for dtype in [
            TensorType::Iq3Xxs,
            TensorType::Iq4Nl,
            TensorType::Iq4Xs,
            TensorType::Q8_0,
            TensorType::Iq3S,
            TensorType::Q2_0,
        ] {
            let columns = dtype.block_elements() as usize;
            let bytes = dtype.row_bytes(columns as u64).unwrap() as usize;
            let record = vec![0; bytes];
            for length in 0..bytes {
                assert!(unpack32(dtype, &record[..length], columns, 0).is_err());
            }
            assert!(unpack32(dtype, &record, columns, columns / 32).is_err());
            assert!(unpack32(dtype, &record, usize::MAX, 0).is_err());
        }
        let row = vec![0u8; 82];
        for length in 0..82 {
            assert!(unpack_iq2_s16(&row[..length], 256, 0).is_err());
        }
        assert!(unpack_iq2_s16(&row, 256, 16).is_err());
        assert!(unpack_iq2_s16(&row, 128, 0).is_err());
        assert!(unpack_iq2_s16(&row, usize::MAX, 0).is_err());
    }
}
