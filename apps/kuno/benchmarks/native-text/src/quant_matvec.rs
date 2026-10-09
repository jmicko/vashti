//! Scalar packed-weight matvec written in Rust/CubeCL 0.10.0.
//!
//! Reference: `crates/kuno-text/src/quant.rs`, whose GGML packing provenance is
//! ggml revision `4bf5f6000653b7881d00963cd6ddb665ccd62a8d` (MIT) for the
//! initial formats and pinned llama.cpp b11232 for IQ3/IQ4 codebooks/geometry.
//! This decodes weights to f32 and accumulates in column order. It does not
//! implement GGML's additional quantization of the activation vector.
//!
//! Launch `packed_matvec::launch::<R>` with Array<u32> / Array<f32> /
//! Array<f32>, then rows, columns, and the compile-time GGUF dtype ID.
//! Validate the layout first. Rows are tightly byte-packed: do NOT pad each
//! 18/22/34/98/210-byte block or each row. Pad only the entire upload to four bytes.
//! F16 conversion needs integer bit operations and f32 bitcast, not shader f16.
//! No GPU has been run by this module's host tests. Device parity still needs
//! independent `kuno-text` decoding, especially rounding/FMA and subnormals.

use cubecl::prelude::*;

#[path = "quant_iq_fixtures.rs"]
pub(crate) mod iq_fixtures;

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;
pub const IQ3_XXS: u32 = 18;
pub const IQ4_NL: u32 = 20;
pub const IQ4_XS: u32 = 23;
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
        IQ3_XXS => Some((256, 98)),
        IQ4_NL => Some((32, 18)),
        IQ4_XS => Some((256, 136)),
        _ => None,
    }
}

/// Original tightly packed shader row stride. The host's block_layout and
/// validated columns must agree; no per-row padding or conversion is allowed.
#[cube]
pub(crate) fn packed_row_bytes(columns: usize, #[comptime] dtype: u32) -> usize {
    let mut bytes = columns * 4;
    if comptime!(dtype == F16 || dtype == BF16) {
        bytes = columns * 2;
    } else if comptime!(dtype == Q5_0) {
        bytes = (columns / 32) * 22;
    } else if comptime!(dtype == Q8_0) {
        bytes = (columns / 32) * 34;
    } else if comptime!(dtype == Q4_K) {
        bytes = (columns / 256) * 144;
    } else if comptime!(dtype == Q6_K) {
        bytes = (columns / 256) * 210;
    } else if comptime!(dtype == IQ3_XXS) {
        bytes = (columns / 256) * 98;
    } else if comptime!(dtype == IQ4_NL) {
        bytes = (columns / 32) * 18;
    } else if comptime!(dtype == IQ4_XS) {
        bytes = (columns / 256) * 136;
    }
    bytes
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

/// Public codebooks are compiled as shader-global constant arrays. There is no
/// new storage binding, per-request lookup upload or expanded weight payload.
#[cube]
fn iq3_grid_word(index: usize) -> u32 {
    let grid = Array::<u32>::from_data(comptime!(kuno_text::quant::IQ3_XXS_GRID.to_vec()));
    grid[index]
}

#[cube]
fn iq4_value(index: usize) -> f32 {
    let values = Array::<i32>::from_data(comptime!(
        kuno_text::quant::IQ4_NL_VALUES
            .iter()
            .copied()
            .map(i32::from)
            .collect::<Vec<_>>()
    ));
    values[index] as f32
}

/// F32 callers must use a four-byte-aligned row_offset. Validated tightly
/// packed F32 rows are columns*4 bytes; whole-row tile/expert views preserve
/// alignment. Quantized rows keep their original byte addressing below.
#[cube]
pub(crate) fn packed_value(
    words: &Array<u32>,
    row_offset: usize,
    column: usize,
    #[comptime] dtype: u32,
) -> f32 {
    let mut value = 0.0_f32;
    if comptime!(dtype == F32) {
        // Original little-endian u32 word, exact bitcast; no weight arithmetic.
        value = f32::from_bits(words[row_offset / 4 + column]);
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
    } else if comptime!(dtype == IQ3_XXS) {
        let base = row_offset + (column / 256) * 98;
        let within = column % 256;
        let group = within / 32;
        let octet = (within % 32) / 8;
        let lane = within % 8;
        let word = u32_at(words, base + 66 + group * 4);
        let scale = binary16(u16_at(words, base)) * (0.5_f32 + (word >> 28) as f32) * 0.5;
        let sign_code = (word >> ((octet * 7) as u32)) & 127;
        // Reconstruct the eighth sign from parity, matching GGML's complete
        // 128-entry ksigns_iq2xs table without another shader constant array.
        let mut parity = sign_code ^ (sign_code >> 4);
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        let signs = sign_code | ((parity & 1) << 7);
        let grid_index = byte_at(words, base + 2 + group * 8 + octet * 2 + lane / 4);
        let grid = iq3_grid_word(grid_index as usize);
        let magnitude = (grid >> (((lane % 4) * 8) as u32)) & 255;
        let mut sign = 1.0_f32;
        if ((signs >> (lane as u32)) & 1) != 0 {
            sign = -1.0;
        }
        value = scale * magnitude as f32 * sign;
    } else if comptime!(dtype == IQ4_NL) {
        let base = row_offset + (column / 32) * 18;
        let lane = column % 32;
        let codes = byte_at(words, base + 2 + lane % 16);
        let mut code = codes & 15;
        if lane >= 16 {
            code = codes >> 4;
        }
        value = binary16(u16_at(words, base)) * iq4_value(code as usize);
    } else if comptime!(dtype == IQ4_XS) {
        let base = row_offset + (column / 256) * 136;
        let within = column % 256;
        let group = within / 32;
        let lane = within % 32;
        let low_scale = (byte_at(words, base + 4 + group / 2) >> ((group % 2 * 4) as u32)) & 15;
        let high_scale = (u16_at(words, base + 2) >> ((group * 2) as u32)) & 3;
        let subscale = (low_scale | (high_scale << 4)) as i32 - 32;
        let scale = binary16(u16_at(words, base)) * subscale as f32;
        let codes = byte_at(words, base + 8 + group * 16 + lane % 16);
        let mut code = codes & 15;
        if lane >= 16 {
            code = codes >> 4;
        }
        value = scale * iq4_value(code as usize);
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
        let row_bytes = packed_row_bytes(columns, dtype);
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
            (IQ3_XXS, 256, 98),
            (IQ4_NL, 32, 18),
            (IQ4_XS, 256, 136),
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
        for dtype in [Q5_0, Q8_0, Q4_K, Q6_K, IQ3_XXS, IQ4_NL, IQ4_XS] {
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

    #[test]
    fn original_iq_packing_is_admitted_without_row_padding_or_weight_expansion() {
        for kind in [
            kuno_text::TensorType::Iq3Xxs,
            kuno_text::TensorType::Iq4Nl,
            kuno_text::TensorType::Iq4Xs,
        ] {
            let columns = kind.block_elements() as usize;
            let bytes = kind.block_bytes() as usize;
            assert_eq!(block_layout(kind.gguf_id()), Some((columns, bytes)));
            assert_eq!(
                validate_layout(kind.gguf_id(), 1, columns, bytes, columns, 1),
                Ok(PackedLayout {
                    row_bytes: bytes,
                    total_bytes: bytes,
                    padded_words: bytes.div_ceil(4),
                })
            );
            for tile in [1, 4, 8] {
                let geometry =
                    matmul::MatrixGeometry::new(kind.gguf_id(), 3, columns, tile).unwrap();
                assert_eq!(geometry.row_bytes(), bytes);
                assert_eq!(geometry.total_bytes(), bytes * 3);
            }
        }
    }

    #[test]
    fn shader_row_strides_match_original_gguf_geometry_including_unaligned_rows() {
        for dtype in [
            F32, F16, BF16, Q4_K, Q5_0, Q6_K, Q8_0, IQ3_XXS, IQ4_NL, IQ4_XS,
        ] {
            let kind = kuno_text::TensorType::from_gguf_id(dtype).unwrap();
            for blocks in [1, 2, 5, 10] {
                let columns = kind.block_elements() as usize * blocks;
                assert_eq!(
                    packed_row_bytes(columns, dtype),
                    kind.row_bytes(columns as u64).unwrap() as usize
                );
            }
        }
        assert_eq!(packed_row_bytes(160, IQ4_NL), 90);
        assert_eq!(packed_row_bytes(2560, IQ3_XXS), 980);
        assert_eq!(packed_row_bytes(2560, IQ4_XS), 1360);
    }

    #[test]
    fn aligned_f32_original_words_preserve_special_values_and_independent_decode_bits() {
        let special = [
            0_u32,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0xffc0_1234,
            0x7f80_0001,
            0xff80_0001,
        ];
        for columns in [1, 3, 5, 63, 64, 65, 320, 2560, 10240] {
            let rows = 3;
            let bits: Vec<u32> = (0..rows * columns)
                .map(|i| special[i % special.len()])
                .collect();
            let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
            let layout = validate_layout(F32, rows, columns, bytes.len(), columns, rows).unwrap();
            let words = pack_le_words(&bytes);
            let mut independent = vec![0_f32; bits.len()];
            kuno_text::TensorType::F32
                .decode(&bytes, &mut independent)
                .unwrap();
            for row in 0..rows {
                let row_offset = row * layout.row_bytes;
                assert!(row_offset.is_multiple_of(4));
                for column in 0..columns {
                    let index = row_offset / 4 + column;
                    assert_eq!(index, row * columns + column);
                    // Shader performs this original-word bitcast; its actual
                    // IR/load shape is independently checked in the WGSL test.
                    assert_eq!(
                        f32::from_bits(words[index]).to_bits(),
                        independent[index].to_bits()
                    );
                    assert_eq!(independent[index].to_bits(), bits[index]);
                }
            }
        }
    }
    #[test]
    fn f32_geometry_rejects_non_word_lengths_and_proves_whole_row_offsets() {
        for columns in [1, 5, 48, 320, 2560, 10240] {
            let rows = 3;
            let size = rows * columns * 4;
            for invalid in [size - 3, size - 2, size - 1, size + 1, size + 2, size + 3] {
                assert!(validate_layout(F32, rows, columns, invalid, columns, rows).is_err());
            }
            let layout = validate_layout(F32, rows, columns, size, columns, rows).unwrap();
            assert!(layout.row_bytes.is_multiple_of(4));
            assert_eq!(layout.padded_words, rows * columns);
            // A view beginning at any whole row/expert remains word-aligned.
            for first in [0, 1, 2, 17, 511 * rows] {
                assert!((first * layout.row_bytes).is_multiple_of(4));
            }
        }
    }
    #[test]
    fn f32_wgsl_uses_one_original_word_bitcast_in_both_execution_modes() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let mut b = KernelBuilder::default();
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            b.scope.register_type::<f32>(f);
            let packed = b.input_array(Type::new(u)).into();
            let input = b.input_array(Type::new(f)).into();
            let output = b.output_array(Type::new(f)).into();
            let rows = b.scalar(u).into();
            let columns = b.scalar(u).into();
            packed_matvec::expand(&mut b.scope, packed, input, output, rows, columns, F32);
            let code = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("f32_original_word_probe"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    f,
                )
                .unwrap()
                .to_string();
            assert!(code.contains("bitcast<f32>"));
            assert_eq!(code.matches("buffer_0_global[").count(), 1, "{code}");
            assert!(
                !code.contains("u32(255)") && !code.contains(" >> ") && !code.contains(" << "),
                "{code}"
            );
        }
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

#[path = "moe_word_down.rs"]
pub mod moe_word_down;

#[path = "cooperative_matvec.rs"]
pub mod cooperative;

#[path = "moe_gpu.rs"]
pub mod moe;

#[path = "block_matvec.rs"]
pub mod block;

#[path = "fast_matvec.rs"]
pub mod fast;
#[path = "quant_matmul.rs"]
pub mod matmul;
#[path = "moe_block.rs"]
pub mod moe_block;
#[path = "segmented_moe.rs"]
pub mod segmented_moe;

// Opt-in fixed routed operator owner; ordinary graph selection is unchanged.
#[path = "grouped_moe.rs"]
pub mod grouped_moe;
