//! Optional block-specialized Q4_K/Q6_K matvec, as a child of `quant_matvec`.
//!
//! Use `block_matvec::launch::<R>` (or `launch_unchecked`) with the parent's
//! packed/input/output arguments, rows, columns, and compile-time dtype.
//! CubeDim=(64,1,1), CubeCount=(rows,1,1); validate each row view with the
//! parent's `validate_layout` and respect the adapter's grid/binding limits.
//! Other supported dtypes retain the generic cooperative decoder.
//!
//! Q4_K loads its two half scales and three scale words once per 256-value
//! block/lane. Q6_K loads one half scale per block/lane and reuses each low/high
//! byte pair for its two quarters. Original bytes and f32 activation math stay
//! intact. Lane accumulation visits exactly the same columns, in the same
//! order, as the reference 64-lane cooperative kernel; the reduction tree is
//! also unchanged. Backend contraction still needs device numerical acceptance.
//! Only the complete upload receives u32 padding; never pad blocks or rows.
//! Q6_K block/row bases can have byte alignment 2, which is handled explicitly.
//!
//! Host tests emulate the field loads and lane schedule on public synthetic
//! packed bytes and compare with independent `kuno_text::TensorType::decode`.
//! They do not launch a GPU or prove device performance.

use super::{BF16, F16, Q4_K, Q5_0, Q6_K, Q8_0, binary16, packed_value, signed_byte};
use cubecl::prelude::*;

pub const LANES: usize = 64;

#[cube]
fn word_byte(word: u32, byte: usize) -> u32 {
    (word >> ((byte * 8) as u32)) & 255
}

/// The requested halfword must start at an even byte offset. It cannot cross
/// a word boundary, including the last Q6_K scale in a two-byte-padded upload.
#[cube]
fn even_half(words: &Array<u32>, offset: usize) -> u32 {
    (words[offset / 4] >> (((offset & 2) * 8) as u32)) & 65535
}

#[cube]
fn packed_byte(words: &Array<u32>, offset: usize) -> u32 {
    word_byte(words[offset / 4], offset & 3)
}

#[cube]
fn q4_scale(scales0: u32, scales2: u32, group: usize) -> u32 {
    if group < 4 {
        word_byte(scales0, group) & 63
    } else {
        (word_byte(scales2, group - 4) & 15) | ((word_byte(scales0, group - 4) >> 6) << 4)
    }
}

#[cube]
fn q4_minimum(scales1: u32, scales2: u32, group: usize) -> u32 {
    if group < 4 {
        word_byte(scales1, group) & 63
    } else {
        (word_byte(scales2, group - 4) >> 4) | ((word_byte(scales1, group - 4) >> 6) << 4)
    }
}

#[cube]
fn q6_quant(low: u32, high: u32, quarter: usize) -> i32 {
    let code = ((low >> (((quarter / 2) * 4) as u32)) & 15)
        | (((high >> ((quarter * 2) as u32)) & 3) << 4);
    code as i32 - 32
}

/// All barriers are cube-uniform, including out-of-range cubes. Unchecked
/// safety additionally requires actual nonaliasing buffer lengths, whole
/// quant blocks, 32-bit-safe geometry, and exactly 64 x-lanes.
#[cube(launch, launch_unchecked)]
pub fn block_matvec(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    rows: usize,
    columns: usize,
    #[comptime] dtype: u32,
    output_first: usize,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut partial = 0.0_f32;
    if row < rows {
        if comptime!(dtype == Q4_K) {
            let blocks = columns / 256;
            let row_word = row * blocks * 36;
            let quarter_lane = lane % 32;
            let odd_group = lane / 32;
            for block in 0..blocks {
                let base_word = row_word + block * 36;
                let header = packed[base_word];
                let d = binary16(header & 65535);
                let dmin = binary16(header >> 16);
                let scales0 = packed[base_word + 1];
                let scales1 = packed[base_word + 2];
                let scales2 = packed[base_word + 3];
                #[unroll]
                for pair in 0..4_usize {
                    let group = pair * 2 + odd_group;
                    let quant_word = packed[base_word + 4 + pair * 8 + quarter_lane / 4];
                    let quant_byte = word_byte(quant_word, quarter_lane & 3);
                    let quant = (quant_byte >> ((odd_group * 4) as u32)) & 15;
                    let factor = d * q4_scale(scales0, scales2, group) as f32;
                    let bias = dmin * q4_minimum(scales1, scales2, group) as f32;
                    let weight = factor * quant as f32 - bias;
                    let column = block * 256 + pair * 64 + lane;
                    partial += weight * input[column];
                }
            }
        } else if comptime!(dtype == Q6_K) {
            let blocks = columns / 256;
            let row_offset = row * blocks * 210;
            let quarter_lane = lane % 32;
            let first_quarter = lane / 32;
            for block in 0..blocks {
                let base = row_offset + block * 210;
                let d = binary16(even_half(packed, base + 208));
                #[unroll]
                for half in 0..2_usize {
                    let low = packed_byte(packed, base + half * 64 + lane);
                    let high = packed_byte(packed, base + 128 + half * 32 + quarter_lane);
                    let scale_offset =
                        base + 192 + half * 8 + first_quarter * 2 + quarter_lane / 16;
                    let scale_first = signed_byte(packed_byte(packed, scale_offset));
                    let scale_second = signed_byte(packed_byte(packed, scale_offset + 4));
                    let weight_first =
                        d * scale_first as f32 * q6_quant(low, high, first_quarter) as f32;
                    let weight_second =
                        d * scale_second as f32 * q6_quant(low, high, first_quarter + 2) as f32;
                    let column = block * 256 + half * 128 + lane;
                    partial += weight_first * input[column];
                    partial += weight_second * input[column + 64];
                }
            }
        } else {
            let mut row_bytes = columns * 4;
            if comptime!(dtype == F16 || dtype == BF16) {
                row_bytes = columns * 2;
            } else if comptime!(dtype == Q5_0) {
                row_bytes = (columns / 32) * 22;
            } else if comptime!(dtype == Q8_0) {
                row_bytes = (columns / 32) * 34;
            }
            let row_offset = row * row_bytes;
            for tile in 0..columns / LANES {
                let column = tile * LANES + lane;
                partial += packed_value(packed, row_offset, column, dtype) * input[column];
            }
            if lane < columns % LANES {
                let column = (columns / LANES) * LANES + lane;
                partial += packed_value(packed, row_offset, column, dtype) * input[column];
            }
        }
    }

    let mut sums = SharedMemory::<f32>::new(LANES);
    sums[lane] = partial;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        output[output_first + row] = sums[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::TensorType;

    #[test]
    fn wgsl_compiles_without_creating_a_device() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for dtype in [Q4_K, Q6_K, super::super::F32, F16, BF16, Q5_0, Q8_0] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let mut builder = KernelBuilder::default();
                builder.scope.register_type::<u32>(uint);
                builder.scope.register_type::<usize>(uint);
                builder.scope.register_type::<f32>(float);
                builder
                    .scope
                    .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
                builder
                    .scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                let packed = builder.input_array(Type::new(uint)).into();
                let input = builder.input_array(Type::new(float)).into();
                let output = builder.output_array(Type::new(float)).into();
                let rows = builder.scalar(uint).into();
                let columns = builder.scalar(uint).into();
                let output_first = builder.scalar(uint).into();
                block_matvec::expand(
                    &mut builder.scope,
                    packed,
                    input,
                    output,
                    rows,
                    columns,
                    dtype,
                    output_first,
                );
                let settings = KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("block_matvec");
                let shader = WgslCompiler::default()
                    .compile(
                        builder.build(settings),
                        &WgpuCompilationOptions::default(),
                        mode,
                        uint,
                    )
                    .unwrap();
                let source = shader.to_string();
                assert!(source.contains("@compute"));
                assert!(source.contains("workgroupBarrier"));
            }
        }
    }

    // Slice-backed CPU field-load emulator; no CubeCL client/device is created.
    fn byte(words: &[u32], offset: usize) -> u32 {
        word_byte(words[offset / 4], offset & 3)
    }

    fn emulated_row(words: &[u32], row: usize, columns: usize, dtype: u32) -> Vec<f32> {
        let blocks = columns / 256;
        let mut decoded = vec![0.0; columns];
        for block in 0..blocks {
            if dtype == Q4_K {
                let w = (row * blocks + block) * 36;
                let d = binary16(words[w] & 65535);
                let dmin = binary16(words[w] >> 16);
                let scales = [words[w + 1], words[w + 2], words[w + 3]];
                for lane in 0..LANES {
                    for pair in 0..4 {
                        let group = pair * 2 + lane / 32;
                        let q = word_byte(words[w + 4 + pair * 8 + (lane % 32) / 4], lane & 3);
                        let q = (q >> ((lane / 32) * 4)) & 15;
                        let factor = d * q4_scale(scales[0], scales[2], group) as f32;
                        let bias = dmin * q4_minimum(scales[1], scales[2], group) as f32;
                        decoded[block * 256 + pair * 64 + lane] = factor * q as f32 - bias;
                    }
                }
            } else {
                let base = (row * blocks + block) * 210;
                let offset = base + 208;
                let d = binary16((words[offset / 4] >> ((offset & 2) * 8)) & 65535);
                for lane in 0..LANES {
                    for half in 0..2 {
                        let low = byte(words, base + half * 64 + lane);
                        let high = byte(words, base + 128 + half * 32 + lane % 32);
                        let s = base + 192 + half * 8 + (lane / 32) * 2 + (lane % 32) / 16;
                        let column = block * 256 + half * 128 + lane;
                        decoded[column] = d
                            * signed_byte(byte(words, s)) as f32
                            * q6_quant(low, high, lane / 32) as f32;
                        decoded[column + 64] = d
                            * signed_byte(byte(words, s + 4)) as f32
                            * q6_quant(low, high, lane / 32 + 2) as f32;
                    }
                }
            }
        }
        decoded
    }

    fn tree_dot(decoded: &[f32], input: &[f32]) -> f32 {
        let mut sums = [0.0_f32; LANES];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for column in (lane..decoded.len()).step_by(LANES) {
                *sum += decoded[column] * input[column];
            }
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                sums[lane] += sums[lane + stride];
            }
            stride /= 2;
        }
        sums[0]
    }

    fn synthetic_block(dtype: u32, seed: usize, half_bits: u16) -> Vec<u8> {
        if dtype == Q4_K {
            let mut packed = vec![0_u8; 144];
            packed[..2].copy_from_slice(&half_bits.to_le_bytes());
            packed[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
            for group in 0..8 {
                let scale = ((group * 19 + seed * 7) % 64) as u8;
                let minimum = ((group * 29 + seed * 11) % 64) as u8;
                if group < 4 {
                    packed[4 + group] |= scale;
                    packed[8 + group] |= minimum;
                } else {
                    packed[8 + group] = (scale & 15) | ((minimum & 15) << 4);
                    packed[group] |= (scale >> 4) << 6;
                    packed[4 + group] |= (minimum >> 4) << 6;
                }
            }
            for column in 0..256 {
                let group = column / 32;
                let quant = ((column * 7 + group + seed * 3) % 16) as u8;
                packed[16 + (group / 2) * 32 + column % 32] |= quant << ((group % 2) * 4);
            }
            packed
        } else {
            let mut packed = vec![0_u8; 210];
            packed[208..].copy_from_slice(&half_bits.to_le_bytes());
            // Include both int8 extremes, zero, and opposite signs at lane 16.
            for group in 0..16 {
                packed[192 + group] = ((group * 47 + seed * 19) % 256) as u8;
            }
            for column in 0..256 {
                let raw = ((column * 13 + column / 16 + seed * 5) % 64) as u8;
                let half = column / 128;
                let quarter = (column % 128) / 32;
                let lane = column % 32;
                packed[half * 64 + (quarter % 2) * 32 + lane] |= (raw & 15) << ((quarter / 2) * 4);
                packed[128 + half * 32 + lane] |= (raw >> 4) << (quarter * 2);
            }
            packed
        }
    }

    #[test]
    fn block_fields_match_independent_gguf_decode_and_lane_schedule() {
        for (dtype, kind) in [(Q4_K, TensorType::Q4K), (Q6_K, TensorType::Q6K)] {
            for columns in [256, 512, 768, 2560] {
                let (rows, blocks) = (3, columns / 256);
                let bytes_per_block = if dtype == Q4_K { 144 } else { 210 };
                let packed: Vec<u8> = (0..rows * blocks)
                    .flat_map(|n| synthetic_block(dtype, n + 1, [0x3800, 0xb400, 0x0001][n % 3]))
                    .collect();
                let input: Vec<f32> = (0..columns)
                    .map(|n| ((n % 31) as f32 - 15.0) / 16.0)
                    .collect();
                let layout = super::super::validate_layout(
                    dtype,
                    rows,
                    columns,
                    packed.len(),
                    columns,
                    rows,
                )
                .unwrap();
                let words = super::super::pack_le_words(&packed);
                assert_eq!(words.len(), layout.padded_words);
                for row in 0..rows {
                    let mut expected = vec![0.0; columns];
                    let row_bytes = blocks * bytes_per_block;
                    kind.decode(
                        &packed[row * row_bytes..(row + 1) * row_bytes],
                        &mut expected,
                    )
                    .unwrap();
                    let actual = emulated_row(&words, row, columns, dtype);
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "dtype={dtype} row={row} columns={columns}"
                    );
                    assert_eq!(
                        tree_dot(&actual, &input).to_bits(),
                        tree_dot(&expected, &input).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn every_column_contributes_once_and_q6_final_halfword_is_in_bounds() {
        for (dtype, kind) in [(Q4_K, TensorType::Q4K), (Q6_K, TensorType::Q6K)] {
            let rows = 3;
            let packed: Vec<u8> = (0..rows)
                .flat_map(|n| synthetic_block(dtype, n + 1, 0x3800))
                .collect();
            let words = super::super::pack_le_words(&packed);
            let row_bytes = if dtype == Q4_K { 144 } else { 210 };
            let mut expected = vec![0.0; 256];
            kind.decode(
                &packed[(rows - 1) * row_bytes..rows * row_bytes],
                &mut expected,
            )
            .unwrap();
            let actual = emulated_row(&words, rows - 1, 256, dtype);
            for column in 0..256 {
                let mut input = [0.0_f32; 256];
                input[column] = 1.0;
                assert_eq!(
                    tree_dot(&actual, &input),
                    expected[column],
                    "dtype={dtype} column={column}"
                );
            }
            if dtype == Q6_K {
                assert_eq!(packed.len() % 4, 2);
                assert_eq!((packed.len() - 2) / 4, words.len() - 1);
            }
        }
    }
}
