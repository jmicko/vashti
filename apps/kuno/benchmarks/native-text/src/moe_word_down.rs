//! Optional original-byte Q5_0/Q8_0 MoE down projection.
//!
//! Sixteen physical threads own the existing 64 virtual-lane accumulators.
//! Each virtual lane visits r, r+64, ... and the original 64-slot sum tree is
//! unchanged. No activation quantization, scale factoring, dot or explicit FMA.
//! GGML packing follows kuno-text::quant (MIT provenance recorded there).
//! Only the whole bank is u32-padded; 22/34-byte blocks and rows stay tight.

use super::{Q5_0, Q8_0, binary16, byte_at, signed_byte};
use cubecl::prelude::*;

pub const PHYSICAL_LANES: usize = 16;
const VIRTUAL_LANES: usize = 64;

pub fn validate(dtype: u32, columns: usize) -> Result<(), &'static str> {
    if !matches!(dtype, Q5_0 | Q8_0) || columns == 0 || !columns.is_multiple_of(32) {
        return Err("word down requires Q5_0/Q8_0 whole 32-value blocks");
    }
    Ok(())
}

/// Every supported header is two-byte aligned; never crosses a u32.
#[cube]
fn half_bits(words: &Array<u32>, offset: usize) -> u32 {
    (words[offset / 4] >> (((offset & 2) * 8) as u32)) & 65535
}

/// Four bytes known to belong to an admitted block. Branch avoids shift32 and
/// avoids reading a second word at the aligned end of the whole bank.
#[cube]
fn word_even(words: &Array<u32>, offset: usize) -> u32 {
    let first = offset / 4;
    let mut value = words[first];
    if offset & 2 != 0 {
        value = (value >> 16) | (words[first + 1] << 16);
    }
    value
}

/// CubeDim(16,1,1), CubeCount(selected*hidden,1,1), same four bindings and
/// runtime dimensions as moe::moe_down. Host retains the existing bank/slot,
/// disjoint-buffer and 32-bit bounds proofs before unchecked dispatch.
#[cube(launch, launch_unchecked)]
pub fn moe_down(
    down_bank: &Array<u32>,
    activations: &Array<f32>,
    slots: &Array<u32>,
    partials: &mut Array<f32>,
    selected: usize,
    hidden: usize,
    feed_forward: usize,
    #[comptime] down_dtype: u32,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    if task < selected * hidden {
        let rank = task / hidden;
        let feature = task % hidden;
        let row = (slots[rank] as usize) * hidden + feature;
        let activation_offset = rank * feed_forward;
        let tiles = feed_forward.div_ceil(VIRTUAL_LANES);
        if comptime!(down_dtype == Q5_0) {
            let row_offset = row * (feed_forward / 32) * 22;
            for tile in 0..tiles {
                #[unroll]
                for pair in 0..2_usize {
                    let column = tile * VIRTUAL_LANES + pair * 32 + lane;
                    if column < feed_forward {
                        let base = row_offset + (tile * 2 + pair) * 22;
                        let d = binary16(half_bits(down_bank, base));
                        let high = word_even(down_bank, base + 2);
                        let low = byte_at(down_bank, base + 6 + lane);
                        let code0 = (low & 15) | (((high >> (lane as u32)) & 1) << 4);
                        let code1 = (low >> 4) | (((high >> ((lane + 16) as u32)) & 1) << 4);
                        let weight0 = d * (code0 as i32 - 16) as f32;
                        let weight1 = d * (code1 as i32 - 16) as f32;
                        let term0 = weight0 * activations[activation_offset + column];
                        let term1 = weight1 * activations[activation_offset + column + 16];
                        if pair == 0 {
                            sum0 += term0;
                            sum1 += term1;
                        } else {
                            sum2 += term0;
                            sum3 += term1;
                        }
                    }
                }
            }
        } else if comptime!(down_dtype == Q8_0) {
            let row_offset = row * (feed_forward / 32) * 34;
            for tile in 0..tiles {
                let column = tile * VIRTUAL_LANES + lane * 4;
                if column < feed_forward {
                    let base = row_offset + (column / 32) * 34;
                    let d = binary16(half_bits(down_bank, base));
                    let quants = word_even(down_bank, base + 2 + column % 32);
                    let weight0 = d * signed_byte(quants & 255) as f32;
                    let weight1 = d * signed_byte((quants >> 8) & 255) as f32;
                    let weight2 = d * signed_byte((quants >> 16) & 255) as f32;
                    let weight3 = d * signed_byte(quants >> 24) as f32;
                    sum0 += weight0 * activations[activation_offset + column];
                    sum1 += weight1 * activations[activation_offset + column + 1];
                    sum2 += weight2 * activations[activation_offset + column + 2];
                    sum3 += weight3 * activations[activation_offset + column + 3];
                }
            }
        }
    }
    let mut sums = SharedMemory::<f32>::new(VIRTUAL_LANES);
    if comptime!(down_dtype == Q5_0) {
        sums[lane] = sum0;
        sums[lane + 16] = sum1;
        sums[lane + 32] = sum2;
        sums[lane + 48] = sum3;
    } else {
        sums[lane * 4] = sum0;
        sums[lane * 4 + 1] = sum1;
        sums[lane * 4 + 2] = sum2;
        sums[lane * 4 + 3] = sum3;
    }
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            sums[lane] = sums[lane] + sums[lane + stride];
        }
        if stride > PHYSICAL_LANES {
            sums[lane + 16] = sums[lane + 16] + sums[lane + 48];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && task < selected * hidden {
        partials[task] = sums[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::TensorType;

    fn half(bits: u16) -> f32 {
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 31;
        let fraction = bits & 1023;
        if exponent == 0 {
            let value = f32::from(fraction) * 2.0_f32.powi(-24);
            if sign != 0 { -value } else { value }
        } else {
            f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (u32::from(fraction) << 13))
        }
    }
    fn read_half(words: &[u32], offset: usize) -> f32 {
        half(((words[offset / 4] >> ((offset & 2) * 8)) & 65535) as u16)
    }
    fn read_word(words: &[u32], offset: usize) -> u32 {
        let value = words[offset / 4];
        if offset & 2 == 0 {
            value
        } else {
            (value >> 16) | (words[offset / 4 + 1] << 16)
        }
    }
    fn read_byte(words: &[u32], offset: usize) -> u32 {
        (words[offset / 4] >> ((offset % 4) * 8)) & 255
    }
    fn fixture(dtype: u32, blocks: usize) -> Vec<u8> {
        let scales = [0x0001_u16, 0x03ff, 0x0400, 0x1800, 0x3000, 0x8000, 0xb000];
        let bytes = if dtype == Q5_0 { 22 } else { 34 };
        (0..blocks)
            .flat_map(|block| {
                let mut out = vec![0_u8; bytes];
                out[..2].copy_from_slice(&scales[block % scales.len()].to_le_bytes());
                for (index, byte) in out[2..].iter_mut().enumerate() {
                    *byte = (block.wrapping_mul(71).wrapping_add(index * 53 + 19) & 255) as u8;
                }
                out
            })
            .collect()
    }
    fn emulate(
        words: &[u32],
        row: usize,
        columns: usize,
        dtype: u32,
        input: &[f32],
    ) -> (Vec<f32>, f32) {
        let row_offset = row * (columns / 32) * if dtype == Q5_0 { 22 } else { 34 };
        let mut decoded = vec![0.0; columns];
        let mut sums = [0.0_f32; VIRTUAL_LANES];
        for lane in 0..PHYSICAL_LANES {
            for tile in 0..columns.div_ceil(VIRTUAL_LANES) {
                if dtype == Q5_0 {
                    for pair in 0..2 {
                        let column = tile * VIRTUAL_LANES + pair * 32 + lane;
                        if column >= columns {
                            continue;
                        }
                        let base = row_offset + (tile * 2 + pair) * 22;
                        let d = read_half(words, base);
                        let high = read_word(words, base + 2);
                        let low = read_byte(words, base + 6 + lane);
                        for j in 0..2 {
                            let code =
                                ((low >> (j * 4)) & 15) | (((high >> (lane + j * 16)) & 1) << 4);
                            let weight = d * (code as i32 - 16) as f32;
                            let c = column + j * 16;
                            decoded[c] = weight;
                            sums[c % VIRTUAL_LANES] += weight * input[c];
                        }
                    }
                } else {
                    let column = tile * VIRTUAL_LANES + lane * 4;
                    if column >= columns {
                        continue;
                    }
                    let base = row_offset + (column / 32) * 34;
                    let d = read_half(words, base);
                    let word = read_word(words, base + 2 + column % 32);
                    for j in 0..4 {
                        let quant = ((word >> (j * 8)) as u8) as i8;
                        let weight = d * f32::from(quant);
                        let c = column + j;
                        decoded[c] = weight;
                        sums[c % VIRTUAL_LANES] += weight * input[c];
                    }
                }
            }
        }
        let mut stride = 32;
        while stride > 0 {
            // Emulate actual 16-thread writes, including both stride32 ranges.
            for lane in 0..PHYSICAL_LANES {
                if lane < stride {
                    sums[lane] += sums[lane + stride];
                }
                if stride > PHYSICAL_LANES {
                    sums[lane + 16] += sums[lane + 48];
                }
            }
            stride /= 2;
        }
        (decoded, sums[0])
    }
    fn reference_tree(weights: &[f32], input: &[f32]) -> f32 {
        let mut sums = [0.0_f32; VIRTUAL_LANES];
        for (column, (&weight, &value)) in weights.iter().zip(input).enumerate() {
            sums[column % VIRTUAL_LANES] += weight * value;
        }
        let mut stride = 32;
        while stride != 0 {
            for lane in 0..stride {
                sums[lane] += sums[lane + stride];
            }
            stride /= 2;
        }
        sums[0]
    }

    #[test]
    fn geometry_accepts_actual_down_and_rejects_other_formats() {
        for dtype in [Q5_0, Q8_0] {
            for columns in [32, 64, 96, 640] {
                assert_eq!(validate(dtype, columns), Ok(()));
            }
            for columns in [0, 1, 31, 33, 639] {
                assert!(validate(dtype, columns).is_err());
            }
        }
        for dtype in [
            super::super::F32,
            super::super::Q4_K,
            super::super::Q6_K,
            999,
        ] {
            assert!(validate(dtype, 640).is_err());
        }
    }

    #[test]
    fn all_ten_slots_and_every_original_column_match_independent_decode_and_tree() {
        let slots = [9_usize, 1, 10, 3, 0, 11, 8, 2, 7, 5];
        let hidden = 3;
        for (dtype, kind, block_bytes) in
            [(Q5_0, TensorType::Q5_0, 22), (Q8_0, TensorType::Q8_0, 34)]
        {
            for columns in [32, 64, 96, 128, 640] {
                let row_bytes = columns / 32 * block_bytes;
                let packed = fixture(dtype, 12 * hidden * columns / 32);
                let words = super::super::pack_le_words(&packed);
                let mut parts = vec![0.0; slots.len() * hidden];
                let mut expected_parts = parts.clone();
                for (rank, &slot) in slots.iter().enumerate() {
                    let input: Vec<f32> = (0..columns)
                        .map(|i| ((i * 7 + rank * 3) % 29) as f32 / 32.0 - 0.4375)
                        .collect();
                    for feature in 0..hidden {
                        let row = slot * hidden + feature;
                        let mut reference = vec![0.0; columns];
                        kind.decode(
                            &packed[row * row_bytes..(row + 1) * row_bytes],
                            &mut reference,
                        )
                        .unwrap();
                        let (decoded, value) = emulate(&words, row, columns, dtype, &input);
                        assert_eq!(
                            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        let expected = reference_tree(&reference, &input);
                        assert_eq!(
                            value.to_bits(),
                            expected.to_bits(),
                            "dtype={dtype} columns={columns} slot={slot} feature={feature}"
                        );
                        let precise: f64 = reference
                            .iter()
                            .zip(&input)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum();
                        assert!((f64::from(value) - precise).abs() <= 1e-5 + precise.abs() * 3e-6);
                        parts[rank * hidden + feature] = value;
                        expected_parts[rank * hidden + feature] = expected;
                    }
                }
                let probabilities: Vec<f32> = (1..=10).map(|n| n as f32 / 55.0).collect();
                assert_eq!(
                    super::super::moe::reference_weighted_sum(&parts, &probabilities, hidden)
                        .unwrap(),
                    super::super::moe::reference_weighted_sum(
                        &expected_parts,
                        &probabilities,
                        hidden
                    )
                    .unwrap()
                );
                // Final row exercises every end word without sentinel padding.
                let row = 12 * hidden - 1;
                let mut reference = vec![0.0; columns];
                kind.decode(&packed[row * row_bytes..], &mut reference)
                    .unwrap();
                for column in 0..columns {
                    let mut input = vec![0.0; columns];
                    input[column] = 1.0;
                    assert_eq!(
                        emulate(&words, row, columns, dtype, &input).1.to_bits(),
                        reference_tree(&reference, &input).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn original_cancellation_tree_is_not_replaced_with_subgroup_or_left_fold() {
        let mut values = [0.0; VIRTUAL_LANES];
        values[0] = 16_777_216.0;
        values[1] = 1.0;
        values[32] = -16_777_216.0;
        assert_eq!(reference_tree(&values, &[1.0; VIRTUAL_LANES]), 1.0);
        assert_eq!(values.into_iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn physical_threads_publish_every_virtual_lane_once() {
        for dtype in [Q5_0, Q8_0] {
            let mut owners = [0_u32; VIRTUAL_LANES];
            for lane in 0..PHYSICAL_LANES {
                for j in 0..4 {
                    let index = if dtype == Q5_0 {
                        lane + j * 16
                    } else {
                        lane * 4 + j
                    };
                    owners[index] += 1;
                }
            }
            assert_eq!(owners, [1; VIRTUAL_LANES]);
        }
    }

    #[test]
    fn last_unaligned_original_block_reads_only_the_final_padded_word() {
        for (dtype, kind) in [(Q5_0, TensorType::Q5_0), (Q8_0, TensorType::Q8_0)] {
            let packed = fixture(dtype, 1);
            assert_eq!(packed.len() % 4, 2);
            let words = super::super::pack_le_words(&packed);
            assert_eq!(words.len(), packed.len().div_ceil(4));
            assert_eq!(words.last().unwrap() >> 16, 0);
            let mut reference = [0.0; 32];
            kind.decode(&packed, &mut reference).unwrap();
            for column in 0..32 {
                let mut input = [0.0; 32];
                input[column] = 1.0;
                assert_eq!(
                    emulate(&words, 0, 32, dtype, &input).1.to_bits(),
                    reference_tree(&reference, &input).to_bits()
                );
            }
        }
    }

    #[test]
    fn offline_wgsl_has_sixteen_threads_original_shared_tree_and_no_dot_or_fma() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for dtype in [Q5_0, Q8_0] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
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
                let bank = builder.input_array(Type::new(uint)).into();
                let activation = builder.input_array(Type::new(float)).into();
                let slots = builder.input_array(Type::new(uint)).into();
                let output = builder.output_array(Type::new(float)).into();
                let selected = builder.scalar(uint).into();
                let hidden = builder.scalar(uint).into();
                let ff = builder.scalar(uint).into();
                moe_down::expand(
                    &mut builder.scope,
                    bank,
                    activation,
                    slots,
                    output,
                    selected,
                    hidden,
                    ff,
                    dtype,
                );
                let definition = builder.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(PHYSICAL_LANES as u32))
                        .kernel_name("moe_word_down"),
                );
                let shader = WgslCompiler::default()
                    .compile(definition, &WgpuCompilationOptions::default(), mode, uint)
                    .expect("valid portable word down shader")
                    .to_string();
                let module = wgpu::naga::front::wgsl::parse_str(&shader)
                    .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&shader)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::empty(),
                )
                .validate(&module)
                .expect(
                    "portable checked/unchecked shader validates without subgroup/f16 capabilities",
                );
                assert!(shader.contains("@workgroup_size(16, 1, 1)"), "{shader}");
                assert!(shader.contains("array<f32, 64>"), "{shader}");
                assert!(shader.contains("workgroupBarrier()"), "{shader}");
                for forbidden in ["fma(", "dot(", "subgroup", "enable f16", "atomic"] {
                    assert!(!shader.contains(forbidden), "{forbidden}: {shader}");
                }
            }
        }
    }
}
