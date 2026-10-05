//! Scalar packed-weight matvec written in Rust/CubeCL 0.10.0.
//!
//! Reference: `crates/kuno-text/src/quant.rs`, whose GGML packing provenance is
//! ggml revision `4bf5f6000653b7881d00963cd6ddb665ccd62a8d` (MIT).
//! This decodes weights to f32 and accumulates in column order. It does not
//! implement GGML's additional quantization of the activation vector.
//!
//! Launch `packed_matvec::launch::<R>` with Array<u32> / Array<f32> /
//! Array<f32>, then rows, columns, and the compile-time GGUF dtype ID.
//! Validate the layout first. Rows are tightly byte-packed: do NOT pad each
//! 22/34/210-byte block or each row. Pad only the entire upload to four bytes.
//! F16 conversion needs integer bit operations and f32 bitcast, not shader f16.
//! No GPU has been run by this module's host tests. Device parity still needs
//! independent `kuno-text` decoding, especially rounding/FMA and subnormals.

use cubecl::prelude::*;

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLayout {
    pub row_bytes: usize,
    pub total_bytes: usize,
    pub padded_words: usize,
}

pub const fn block_layout(dtype: u32) -> Option<(usize, usize)> {
    match dtype {
        F32 => Some((1, 4)),
        F16 | BF16 => Some((1, 2)),
        Q5_0 => Some((32, 22)),
        Q8_0 => Some((32, 34)),
        Q4_K => Some((256, 144)),
        Q6_K => Some((256, 210)),
        _ => None,
    }
}

/// Host validation, before any upload/launch. This prototype uses 32-bit shader
/// indexing; larger tensors must be sliced into independently validated views.
pub fn validate_layout(
    dtype: u32,
    rows: usize,
    columns: usize,
    packed_bytes: usize,
    input_len: usize,
    output_len: usize,
) -> Result<PackedLayout, &'static str> {
    let (elements, bytes) = block_layout(dtype).ok_or("unsupported GGUF dtype")?;
    if rows == 0 || columns == 0 || !columns.is_multiple_of(elements) {
        return Err("rows must be nonzero and columns must contain whole blocks");
    }
    let row_bytes = (columns / elements)
        .checked_mul(bytes)
        .ok_or("row byte size overflow")?;
    let total_bytes = rows
        .checked_mul(row_bytes)
        .ok_or("tensor byte size overflow")?;
    if rows > u32::MAX as usize || columns > u32::MAX as usize || total_bytes > u32::MAX as usize {
        return Err("tensor exceeds prototype's 32-bit shader indexing");
    }
    if packed_bytes != total_bytes || input_len < columns || output_len < rows {
        return Err("packed/input/output length mismatch");
    }
    Ok(PackedLayout {
        row_bytes,
        total_bytes,
        padded_words: total_bytes.div_ceil(4),
    })
}

/// Numeric little-endian words. The last word's unused bytes are zero; no row
/// padding is inserted. Upload their little-endian byte encoding, not native
/// POD bytes on a big-endian host (see `words_to_le_upload`).
pub fn pack_le_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|chunk| {
            let mut word = [0_u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect()
}

/// WGPU storage-buffer wire encoding. On this little-endian Linux host this is
/// identical to u32 POD bytes; explicit encoding also works on other hosts.
pub fn words_to_le_upload(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[cube]
fn byte_at(words: &Array<u32>, offset: usize) -> u32 {
    (words[offset / 4] >> (((offset % 4) * 8) as u32)) & 255
}

#[cube]
fn u16_at(words: &Array<u32>, offset: usize) -> u32 {
    byte_at(words, offset) | (byte_at(words, offset + 1) << 8)
}

#[cube]
fn u32_at(words: &Array<u32>, offset: usize) -> u32 {
    byte_at(words, offset)
        | (byte_at(words, offset + 1) << 8)
        | (byte_at(words, offset + 2) << 16)
        | (byte_at(words, offset + 3) << 24)
}

/// Integer IEEE-754 conversion preserves zero signs, half subnormals and NaN
/// payload bits before the backend's f32 arithmetic is applied.
#[cube]
fn binary16(bits: u32) -> f32 {
    let sign = (bits & 32768) << 16;
    let exponent = (bits >> 10) & 31;
    let mut mantissa = bits & 1023;
    let mut result = sign;
    if exponent == 0 {
        if mantissa != 0 {
            let mut adjusted = 113_u32;
            while (mantissa & 1024) == 0 {
                mantissa <<= 1;
                adjusted -= 1;
            }
            result = sign | (adjusted << 23) | ((mantissa & 1023) << 13);
        }
    } else if exponent == 31 {
        result = sign | 2139095040_u32 | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    f32::from_bits(result)
}

#[cube]
fn signed_byte(value: u32) -> i32 {
    let mut result = value as i32;
    if value >= 128 {
        result -= 256;
    }
    result
}

#[cube]
fn packed_value(
    words: &Array<u32>,
    row_offset: usize,
    column: usize,
    #[comptime] dtype: u32,
) -> f32 {
    let mut value = 0.0_f32;
    if comptime!(dtype == F32) {
        value = f32::from_bits(u32_at(words, row_offset + column * 4));
    } else if comptime!(dtype == F16) {
        value = binary16(u16_at(words, row_offset + column * 2));
    } else if comptime!(dtype == BF16) {
        value = f32::from_bits(u16_at(words, row_offset + column * 2) << 16);
    } else if comptime!(dtype == Q5_0) {
        let base = row_offset + (column / 32) * 22;
        let index = column % 32;
        let low = byte_at(words, base + 6 + index % 16);
        let mut code = low & 15;
        if index >= 16 {
            code = low >> 4;
        }
        let high = (u32_at(words, base + 2) >> (index as u32)) & 1;
        let quant = (code | (high << 4)) as i32 - 16;
        value = binary16(u16_at(words, base)) * quant as f32;
    } else if comptime!(dtype == Q8_0) {
        let base = row_offset + (column / 32) * 34;
        let quant = signed_byte(byte_at(words, base + 2 + column % 32));
        value = binary16(u16_at(words, base)) * quant as f32;
    } else if comptime!(dtype == Q4_K) {
        let base = row_offset + (column / 256) * 144;
        let index = column % 256;
        let group = index / 32;
        let subscale = if group < 4 {
            byte_at(words, base + 4 + group) & 63
        } else {
            (byte_at(words, base + 8 + group) & 15) | ((byte_at(words, base + group) >> 6) << 4)
        };
        let subminimum = if group < 4 {
            byte_at(words, base + 8 + group) & 63
        } else {
            (byte_at(words, base + 8 + group) >> 4) | ((byte_at(words, base + 4 + group) >> 6) << 4)
        };
        let packed = byte_at(words, base + 16 + (index / 64) * 32 + index % 32);
        let mut quant = packed & 15;
        if index % 64 >= 32 {
            quant = packed >> 4;
        }
        let factor = binary16(u16_at(words, base)) * subscale as f32;
        let bias = binary16(u16_at(words, base + 2)) * subminimum as f32;
        value = factor * quant as f32 - bias;
    } else if comptime!(dtype == Q6_K) {
        let base = row_offset + (column / 256) * 210;
        let index = column % 256;
        let half = index / 128;
        let quarter = (index % 128) / 32;
        let lane = index % 32;
        let low = byte_at(words, base + half * 64 + lane + (quarter % 2) * 32);
        let high = byte_at(words, base + 128 + half * 32 + lane);
        let code = ((low >> (((quarter / 2) * 4) as u32)) & 15)
            | (((high >> ((quarter * 2) as u32)) & 3) << 4);
        let quant = code as i32 - 32;
        let subscale = signed_byte(byte_at(
            words,
            base + 192 + half * 8 + quarter * 2 + lane / 16,
        ));
        value = binary16(u16_at(words, base + 208)) * subscale as f32 * quant as f32;
    }
    value
}

/// One thread per row, f32 input/output, compile-time GGUF dtype specialization.
/// Host precondition: `validate_layout` succeeds, and ArrayArg lengths describe
/// padded_words, input_len and output_len respectively. Scalar ArrayArg only.
#[cube(launch)]
pub fn packed_matvec(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    rows: usize,
    columns: usize,
    #[comptime] dtype: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut row_bytes = columns * 4;
        if comptime!(dtype == F16 || dtype == BF16) {
            row_bytes = columns * 2;
        } else if comptime!(dtype == Q5_0) {
            row_bytes = (columns / 32) * 22;
        } else if comptime!(dtype == Q8_0) {
            row_bytes = (columns / 32) * 34;
        } else if comptime!(dtype == Q4_K) {
            row_bytes = (columns / 256) * 144;
        } else if comptime!(dtype == Q6_K) {
            row_bytes = (columns / 256) * 210;
        }
        let mut sum = 0.0_f32;
        for column in 0..columns {
            let weight = packed_value(packed, row * row_bytes, column, dtype);
            sum += weight * input[column];
        }
        output[row] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tightly_packed_rows_and_end_padding() {
        for (dtype, columns, expected_row) in [
            (F32, 5, 20),
            (F16, 5, 10),
            (BF16, 5, 10),
            (Q5_0, 32, 22),
            (Q8_0, 32, 34),
            (Q4_K, 256, 144),
            (Q6_K, 256, 210),
        ] {
            let layout = validate_layout(dtype, 3, columns, expected_row * 3, columns, 3).unwrap();
            assert_eq!(layout.row_bytes, expected_row);
            assert_eq!(layout.padded_words, (expected_row * 3).div_ceil(4));
            let bytes: Vec<u8> = (0..layout.total_bytes)
                .map(|i| (i * 37 + 11) as u8)
                .collect();
            let words = pack_le_words(&bytes);
            let upload = words_to_le_upload(&words);
            assert_eq!(&upload[..bytes.len()], bytes.as_slice());
            for (offset, expected) in bytes.iter().enumerate() {
                assert_eq!(
                    ((words[offset / 4] >> ((offset % 4) * 8)) & 255) as u8,
                    *expected
                );
            }
            for offset in layout.total_bytes..words.len() * 4 {
                assert_eq!((words[offset / 4] >> ((offset % 4) * 8)) & 255, 0);
            }
        }
    }

    #[test]
    fn rejects_partial_blocks_mismatch_and_overflow() {
        for dtype in [Q5_0, Q8_0, Q4_K, Q6_K] {
            let (n, b) = block_layout(dtype).unwrap();
            assert!(validate_layout(dtype, 1, n - 1, b, n, 1).is_err());
            assert!(validate_layout(dtype, 1, n, b - 1, n, 1).is_err());
            assert!(validate_layout(dtype, 1, n, b, n - 1, 1).is_err());
            assert!(validate_layout(dtype, 1, n, b, n, 0).is_err());
        }
        assert!(validate_layout(99, 1, 1, 4, 1, 1).is_err());
        assert!(validate_layout(F32, 0, 1, 0, 1, 0).is_err());
        assert!(validate_layout(F32, 1, 0, 0, 0, 1).is_err());
        assert!(validate_layout(F32, usize::MAX, 2, 0, 2, 1).is_err());
        assert!(validate_layout(F32, 1, usize::MAX, 0, 1, 1).is_err());
        assert!(validate_layout(F32, 1, u32::MAX as usize, 0, 1, 1).is_err());
    }

    // #[cube] scalar helpers also retain their ordinary Rust function bodies;
    // these tests validate all 65,536 half encodings without a GPU or payload.
    #[test]
    fn binary16_matches_half_reference_exhaustively() {
        for bits in 0..=u16::MAX {
            // Independent arithmetic oracle instead of the bit-normalization
            // algorithm used by the kernel. Every finite half fits exactly f32.
            let sign = if bits & 0x8000 != 0 {
                -1.0_f32
            } else {
                1.0_f32
            };
            let exponent = i32::from((bits >> 10) & 31);
            let mantissa = f32::from(bits & 1023);
            let expected = if exponent == 0 {
                sign * mantissa * 2.0_f32.powi(-24)
            } else if exponent == 31 {
                if mantissa == 0.0 {
                    sign * f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                sign * (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent - 15)
            };
            let actual = binary16(u32::from(bits));
            if expected.is_nan() {
                assert!(actual.is_nan());
                assert_eq!(
                    actual.to_bits(),
                    (u32::from(bits & 0x8000) << 16) | 0x7f80_0000 | (u32::from(bits & 1023) << 13)
                );
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "half bits {bits:04x}");
            }
        }
    }

    #[test]
    fn signed_byte_has_exact_twos_complement_range() {
        for byte in 0..=u8::MAX {
            assert_eq!(signed_byte(u32::from(byte)), i32::from(byte as i8));
        }
    }
}

#[path = "cooperative_matvec.rs"]
pub mod cooperative;

#[path = "moe_gpu.rs"]
pub mod moe;
