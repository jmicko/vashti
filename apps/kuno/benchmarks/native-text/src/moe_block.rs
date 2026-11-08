//! Optional Q4_K block-specialized selected-expert gate/up kernel.
//!
//! Include as a child of `quant_matvec`. `moe_gate_up::launch[_unchecked]`
//! has exactly the `moe_gpu::moe_gate_up` arguments: five scalar arrays,
//! selected, hidden, feed_forward, compile-time gate_dtype and up_dtype.
//! CubeDim=(64,1,1), CubeCount=(selected*feed_forward,1,1).
//! Banks remain tightly packed [slot][feature][hidden]. Only the entire bank
//! is padded; original Q4_K blocks occupy 144 bytes per 256 input values.
//!
//! Both Q4_K banks use a hoisted header and three scale words per block/lane.
//! Other dtype pairs retain the original generic decoder. Every selected rank
//! and input column contributes; routing, probabilities, down/reduce and the
//! shared expert stay with the existing graph. The 64-lane partial sums visit
//! columns in the same order as `moe_gpu`, and retain its reduction tree and
//! sign-stable SiLU. Device rounding/contraction and performance need root's
//! explicit acceptance; host tests and WGSL generation do not measure either.
//!
//! Q4_K field reconstruction below is a narrow specialization of the packing
//! documented by `kuno-text/src/quant.rs` (GGML packing provenance: revision
//! 4bf5f6000653b7881d00963cd6ddb665ccd62a8d, MIT), as in `block_matvec.rs`.
//! Unchecked launch requires validated bank capacity/slot IDs, whole blocks,
//! actual nonaliasing buffer lengths, 32-bit-safe arithmetic, stable selected
//! slots through gate/up and down, and the stated 64-lane grid.

use super::{BF16, F16, Q4_K, Q5_0, Q6_K, Q8_0, binary16, packed_value};
use cubecl::prelude::*;

pub const LANES: usize = 64;

#[cube]
fn word_byte(word: u32, byte: usize) -> u32 {
    (word >> ((byte * 8) as u32)) & 255
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
fn row_bytes(columns: usize, #[comptime] dtype: u32) -> usize {
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
    }
    bytes
}

/// Same expression/order as `kuno_text::ops::silu` and `moe_gpu::moe_silu`.
#[cube]
fn moe_silu(value: f32) -> f32 {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    value * sigmoid
}

#[cube(launch, launch_unchecked)]
pub fn moe_gate_up(
    gate_bank: &Array<u32>,
    up_bank: &Array<u32>,
    input: &Array<f32>,
    slots: &Array<u32>,
    activations: &mut Array<f32>,
    selected: usize,
    hidden: usize,
    feed_forward: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut gate = 0.0_f32;
    let mut up = 0.0_f32;
    if task < selected * feed_forward {
        let rank = task / feed_forward;
        let feature = task % feed_forward;
        let bank_row = (slots[rank] as usize) * feed_forward + feature;
        if comptime!(gate_dtype == Q4_K && up_dtype == Q4_K) {
            let blocks = hidden / 256;
            let row_word = bank_row * blocks * 36;
            let quarter_lane = lane % 32;
            let odd_group = lane / 32;
            for block in 0..blocks {
                let base = row_word + block * 36;
                let gate_header = gate_bank[base];
                let up_header = up_bank[base];
                let gate_d = binary16(gate_header & 65535);
                let gate_dmin = binary16(gate_header >> 16);
                let up_d = binary16(up_header & 65535);
                let up_dmin = binary16(up_header >> 16);
                let gate_scales0 = gate_bank[base + 1];
                let gate_scales1 = gate_bank[base + 2];
                let gate_scales2 = gate_bank[base + 3];
                let up_scales0 = up_bank[base + 1];
                let up_scales1 = up_bank[base + 2];
                let up_scales2 = up_bank[base + 3];
                #[unroll]
                for pair in 0..4_usize {
                    let group = pair * 2 + odd_group;
                    let quant_offset = base + 4 + pair * 8 + quarter_lane / 4;
                    let gate_byte = word_byte(gate_bank[quant_offset], quarter_lane & 3);
                    let up_byte = word_byte(up_bank[quant_offset], quarter_lane & 3);
                    let gate_quant = (gate_byte >> ((odd_group * 4) as u32)) & 15;
                    let up_quant = (up_byte >> ((odd_group * 4) as u32)) & 15;
                    let gate_factor = gate_d * q4_scale(gate_scales0, gate_scales2, group) as f32;
                    let gate_bias =
                        gate_dmin * q4_minimum(gate_scales1, gate_scales2, group) as f32;
                    let up_factor = up_d * q4_scale(up_scales0, up_scales2, group) as f32;
                    let up_bias = up_dmin * q4_minimum(up_scales1, up_scales2, group) as f32;
                    let gate_weight = gate_factor * gate_quant as f32 - gate_bias;
                    let up_weight = up_factor * up_quant as f32 - up_bias;
                    let column = block * 256 + pair * 64 + lane;
                    let value = input[column];
                    gate += gate_weight * value;
                    up += up_weight * value;
                }
            }
        } else {
            let gate_offset = bank_row * row_bytes(hidden, gate_dtype);
            let up_offset = bank_row * row_bytes(hidden, up_dtype);
            for tile in 0..hidden / LANES {
                let column = tile * LANES + lane;
                let value = input[column];
                gate += packed_value(gate_bank, gate_offset, column, gate_dtype) * value;
                up += packed_value(up_bank, up_offset, column, up_dtype) * value;
            }
            if lane < hidden % LANES {
                let column = (hidden / LANES) * LANES + lane;
                let value = input[column];
                gate += packed_value(gate_bank, gate_offset, column, gate_dtype) * value;
                up += packed_value(up_bank, up_offset, column, up_dtype) * value;
            }
        }
    }

    let mut gates = SharedMemory::<f32>::new(LANES);
    let mut ups = SharedMemory::<f32>::new(LANES);
    gates[lane] = gate;
    ups[lane] = up;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            gates[lane] = gates[lane] + gates[lane + stride];
            ups[lane] = ups[lane] + ups[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && task < selected * feed_forward {
        activations[task] = moe_silu(gates[0]) * ups[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::TensorType;

    #[test]
    fn offline_wgsl_supports_specialized_and_mixed_dtype_pairs() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        let pairs = [
            (Q4_K, Q4_K),
            (super::super::F32, super::super::F32),
            (F16, F16),
            (BF16, BF16),
            (Q5_0, Q5_0),
            (Q6_K, Q6_K),
            (Q8_0, Q8_0),
            (Q4_K, Q8_0),
            (Q8_0, Q4_K),
            (Q4_K, super::super::F32),
            (super::super::F32, Q4_K),
        ];
        for (gate_dtype, up_dtype) in pairs {
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
                let gate = builder.input_array(Type::new(uint)).into();
                let up = builder.input_array(Type::new(uint)).into();
                let input = builder.input_array(Type::new(float)).into();
                let slots = builder.input_array(Type::new(uint)).into();
                let output = builder.output_array(Type::new(float)).into();
                let selected = builder.scalar(uint).into();
                let hidden = builder.scalar(uint).into();
                let ff = builder.scalar(uint).into();
                moe_gate_up::expand(
                    &mut builder.scope,
                    gate,
                    up,
                    input,
                    slots,
                    output,
                    selected,
                    hidden,
                    ff,
                    gate_dtype,
                    up_dtype,
                );
                let settings = KernelSettings::default()
                    .cube_dim(CubeDim::new_1d(64))
                    .kernel_name("moe_gate_up");
                let source = WgslCompiler::default()
                    .compile(
                        builder.build(settings),
                        &WgpuCompilationOptions::default(),
                        mode,
                        uint,
                    )
                    .unwrap()
                    .to_string();
                assert!(source.contains("@compute") && source.contains("workgroupBarrier"));
            }
        }
    }

    fn synthetic_q4(seed: usize, scale_bits: u16) -> Vec<u8> {
        let mut packed = vec![0_u8; 144];
        packed[..2].copy_from_slice(&scale_bits.to_le_bytes());
        packed[2..4].copy_from_slice(&0x2400_u16.to_le_bytes());
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
    }

    // CPU emulation of the specialized word loads; no runtime/device access.
    fn emulate_row(words: &[u32], bank_row: usize, hidden: usize) -> Vec<f32> {
        let blocks = hidden / 256;
        let mut decoded = vec![0.0_f32; hidden];
        for block in 0..blocks {
            let base = (bank_row * blocks + block) * 36;
            let d = binary16(words[base] & 65535);
            let dmin = binary16(words[base] >> 16);
            let s = [words[base + 1], words[base + 2], words[base + 3]];
            for lane in 0..LANES {
                for pair in 0..4 {
                    let group = pair * 2 + lane / 32;
                    let quant = word_byte(words[base + 4 + pair * 8 + (lane % 32) / 4], lane & 3);
                    let quant = (quant >> ((lane / 32) * 4)) & 15;
                    let factor = d * q4_scale(s[0], s[2], group) as f32;
                    let bias = dmin * q4_minimum(s[1], s[2], group) as f32;
                    decoded[block * 256 + pair * 64 + lane] = factor * quant as f32 - bias;
                }
            }
        }
        decoded
    }

    fn tree_dot(row: &[f32], input: &[f32]) -> f32 {
        let mut sums = [0.0_f32; LANES];
        for (lane, sum) in sums.iter_mut().enumerate() {
            for column in (lane..row.len()).step_by(LANES) {
                *sum += row[column] * input[column];
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

    #[test]
    fn all_ten_nontrivial_slots_match_independent_decoded_f64_projection_and_silu() {
        let (capacity, ff) = (12, 3);
        let slots = [9_usize, 1, 10, 3, 0, 11, 8, 2, 7, 5];
        for hidden in [256, 512] {
            let blocks = hidden / 256;
            let gate: Vec<u8> = (0..capacity * ff * blocks)
                .flat_map(|n| synthetic_q4(n + 1, if n % 2 == 0 { 0x2800 } else { 0xa800 }))
                .collect();
            let up: Vec<u8> = (0..capacity * ff * blocks)
                .flat_map(|n| synthetic_q4(n + 137, 0x2800))
                .collect();
            let gate_words = super::super::pack_le_words(&gate);
            let up_words = super::super::pack_le_words(&up);
            let input: Vec<f32> = (0..hidden)
                .map(|n| ((n % 17) as f32 - 8.0) / 16.0)
                .collect();
            let row_bytes = blocks * 144;
            let mut activations = vec![0.0; slots.len() * ff];
            for (rank, &slot) in slots.iter().enumerate() {
                for feature in 0..ff {
                    let row = slot * ff + feature;
                    let mut gate_reference = vec![0.0; hidden];
                    let mut up_reference = vec![0.0; hidden];
                    TensorType::Q4K
                        .decode(
                            &gate[row * row_bytes..(row + 1) * row_bytes],
                            &mut gate_reference,
                        )
                        .unwrap();
                    TensorType::Q4K
                        .decode(
                            &up[row * row_bytes..(row + 1) * row_bytes],
                            &mut up_reference,
                        )
                        .unwrap();
                    let gate_decoded = emulate_row(&gate_words, row, hidden);
                    let up_decoded = emulate_row(&up_words, row, hidden);
                    assert_eq!(
                        gate_decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        gate_reference
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        up_decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        up_reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    let gate_expected = gate_reference
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| f64::from(*w) * f64::from(*x))
                        .sum::<f64>() as f32;
                    let up_expected = up_reference
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| f64::from(*w) * f64::from(*x))
                        .sum::<f64>() as f32;
                    let g = tree_dot(&gate_decoded, &input);
                    let u = tree_dot(&up_decoded, &input);
                    assert_eq!(g.to_bits(), gate_expected.to_bits());
                    assert_eq!(u.to_bits(), up_expected.to_bits());
                    activations[rank * ff + feature] = moe_silu(g) * u;
                    let expected = kuno_text::ops::silu(gate_expected) * up_expected;
                    assert_eq!(
                        activations[rank * ff + feature].to_bits(),
                        expected.to_bits(),
                        "hidden={hidden} rank={rank} slot={slot} feature={feature}"
                    );
                }
            }
            assert!(activations.iter().all(|v| v.is_finite()));
            assert!(activations.windows(ff).any(|pair| pair[0] != pair[1]));
        }
    }

    #[test]
    fn every_q4_column_and_bank_last_word_are_covered() {
        let packed: Vec<u8> = (0..6).flat_map(|n| synthetic_q4(n + 1, 0x2800)).collect();
        let words = super::super::pack_le_words(&packed);
        let mut expected = vec![0.0; 512];
        TensorType::Q4K
            .decode(&packed[4 * 144..], &mut expected)
            .unwrap();
        let actual = emulate_row(&words, 2, 512);
        for column in 0..512 {
            let mut input = vec![0.0; 512];
            input[column] = 1.0;
            assert_eq!(
                tree_dot(&actual, &input),
                expected[column],
                "column={column}"
            );
        }
        assert_eq!((2 * 2 + 1) * 36 + 35, words.len() - 1);
    }

    #[test]
    fn sign_stable_silu_retains_reference_zero_and_extreme_values() {
        for value in [
            -100.0_f32, -80.0, -8.0, -1.0, -0.0, 0.0, 1.0, 8.0, 80.0, 100.0,
        ] {
            assert_eq!(
                moe_silu(value).to_bits(),
                kuno_text::ops::silu(value).to_bits()
            );
            assert!(moe_silu(value).is_finite());
        }
    }
}
