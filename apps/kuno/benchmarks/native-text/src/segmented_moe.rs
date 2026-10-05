//! Opt-in two-shard original-weight MoE. Shard geometry is public, but shard
//! first/count, global slots and ranks remain runtime scalar/buffer values.
//! Inactive ranks participate in every barrier and never overwrite outputs.
//! Arithmetic is copied from moe_block (gate/up) and moe_gpu (down): the
//! original per-lane column order, 64-lane tree and global rank are unchanged.
//! Root must retain all shards and one global plan until whole-token acceptance.
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
    first_slot: usize,
    slot_count: usize,
    #[comptime] gate_dtype: u32,
    #[comptime] up_dtype: u32,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut active = false;
    let mut gate = 0.0_f32;
    let mut up = 0.0_f32;
    if task < selected * feed_forward {
        let rank = task / feed_forward;
        let feature = task % feed_forward;
        let slot = slots[rank] as usize;
        if slot >= first_slot {
            active = slot - first_slot < slot_count;
        }
        if active {
            let bank_row = (slot - first_slot) * feed_forward + feature;
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
                        let gate_factor =
                            gate_d * q4_scale(gate_scales0, gate_scales2, group) as f32;
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
    if lane == 0 && task < selected * feed_forward && active {
        activations[task] = moe_silu(gates[0]) * ups[0];
    }
}

#[cube(launch, launch_unchecked)]
pub fn moe_down(
    down_bank: &Array<u32>,
    activations: &Array<f32>,
    slots: &Array<u32>,
    partials: &mut Array<f32>,
    selected: usize,
    hidden: usize,
    feed_forward: usize,
    first_slot: usize,
    slot_count: usize,
    #[comptime] down_dtype: u32,
) {
    let task = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut active = false;
    let mut partial = 0.0_f32;
    if task < selected * hidden {
        let rank = task / hidden;
        let feature = task % hidden;
        let slot = slots[rank] as usize;
        if slot >= first_slot {
            active = slot - first_slot < slot_count;
        }
        if active {
            let bank_row = (slot - first_slot) * hidden + feature;
            let row_offset = bank_row * row_bytes(feed_forward, down_dtype);
            let activation_offset = rank * feed_forward;
            let full_tiles = feed_forward / LANES;
            for tile in 0..full_tiles {
                let column = tile * LANES + lane;
                partial += packed_value(down_bank, row_offset, column, down_dtype)
                    * activations[activation_offset + column];
            }
            if lane < feed_forward % LANES {
                let column = full_tiles * LANES + lane;
                partial += packed_value(down_bank, row_offset, column, down_dtype)
                    * activations[activation_offset + column];
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
    if lane == 0 && task < selected * hidden && active {
        partials[task] = sums[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::TensorType;
    #[test]
    fn offline_shards_compile_checked_and_unchecked_with_runtime_choices() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        for down_dtype in [Q5_0, Q8_0] {
            for down in [false, true] {
                for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                    let mut b = KernelBuilder::default();
                    b.scope.register_type::<u32>(u);
                    b.scope.register_type::<usize>(u);
                    b.scope.register_type::<f32>(f);
                    b.scope
                        .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
                    b.scope
                        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                    let weight = b.input_array(Type::new(u)).into();
                    let input = b.input_array(Type::new(f)).into();
                    let slots = b.input_array(Type::new(u)).into();
                    let output = b.output_array(Type::new(f)).into();
                    let selected = b.scalar(u).into();
                    let hidden = b.scalar(u).into();
                    let ff = b.scalar(u).into();
                    let first = b.scalar(u).into();
                    let count = b.scalar(u).into();
                    if down {
                        moe_down::expand(
                            &mut b.scope,
                            weight,
                            input,
                            slots,
                            output,
                            selected,
                            hidden,
                            ff,
                            first,
                            count,
                            down_dtype,
                        );
                    } else {
                        let up = b.input_array(Type::new(u)).into();
                        moe_gate_up::expand(
                            &mut b.scope,
                            weight,
                            up,
                            input,
                            slots,
                            output,
                            selected,
                            hidden,
                            ff,
                            first,
                            count,
                            Q4_K,
                            Q4_K,
                        );
                    }
                    let src = WgslCompiler::default()
                        .compile(
                            b.build(
                                KernelSettings::default()
                                    .cube_dim(CubeDim::new_1d(64))
                                    .kernel_name("segmented_moe"),
                            ),
                            &WgpuCompilationOptions::default(),
                            mode,
                            u,
                        )
                        .unwrap()
                        .to_string();
                    assert!(src.contains("@compute") && src.contains("workgroupBarrier"));
                    assert!(src.contains("scalars_u32"));
                    // The public dtype is the only specialization. All five
                    // geometry/shard scalars are runtime inputs, with slots
                    // kept in a storage binding and no literal fixture IDs.
                    for n in 0..5 {
                        assert!(src.contains(&format!("scalars_u32[{n}]")));
                    }
                    assert!(!src.contains("scalars_u32[5]"));
                }
            }
        }
    }
    fn dot64(weights: &[f32], input: &[f32]) -> f32 {
        let mut lanes = [0.0f32; 64];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for column in (lane..input.len()).step_by(64) {
                *sum += weights[column] * input[column];
            }
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
            stride /= 2;
        }
        lanes[0]
    }
    fn q4(seed: usize) -> Vec<u8> {
        let mut v = vec![0u8; 144];
        v[..2].copy_from_slice(&0x2800u16.to_le_bytes());
        v[2..4].copy_from_slice(&0x2000u16.to_le_bytes());
        for (n, x) in v[4..].iter_mut().enumerate() {
            *x = (n * 17 + seed * 23) as u8;
        }
        v
    }
    fn down_row(dtype: TensorType, seed: usize, ff: usize) -> Vec<u8> {
        let bytes = if dtype == TensorType::Q5_0 { 22 } else { 34 };
        let mut v = vec![0u8; ff / 32 * bytes];
        for (block, chunk) in v.chunks_mut(bytes).enumerate() {
            chunk[..2].copy_from_slice(&0x2400u16.to_le_bytes());
            for (n, x) in chunk[2..].iter_mut().enumerate() {
                *x = (seed * 29 + block * 13 + n * 7) as u8;
            }
        }
        v
    }
    #[test]
    fn shard_global_rank_addressing_preserves_all_ten_original_numeric_paths() {
        let hidden = 256;
        let ff = 32;
        let slots = [95, 0, 63, 64, 1, 94, 31, 65, 62, 32];
        let probability = [0.12, 0.13, 0.07, 0.1, 0.11, 0.08, 0.09, 0.06, 0.14, 0.1];
        let input: Vec<_> = (0..hidden).map(|n| (n % 17) as f32 / 16.0 - 0.5).collect();
        for dtype in [TensorType::Q5_0, TensorType::Q8_0] {
            let gate: Vec<u8> = (0..96 * ff).flat_map(q4).collect();
            let up: Vec<u8> = (0..96 * ff).flat_map(|n| q4(n + 3)).collect();
            let down: Vec<u8> = (0..96 * hidden)
                .flat_map(|n| down_row(dtype, n, ff))
                .collect();
            let drow = if dtype == TensorType::Q5_0 { 22 } else { 34 };
            let mut gate_f = vec![0.0; 96 * ff * hidden];
            let mut up_f = gate_f.clone();
            let mut down_f = vec![0.0; 96 * hidden * ff];
            TensorType::Q4K.decode(&gate, &mut gate_f).unwrap();
            TensorType::Q4K.decode(&up, &mut up_f).unwrap();
            dtype.decode(&down, &mut down_f).unwrap();
            let original = super::super::moe::reference_grouped_moe(
                &gate_f,
                &up_f,
                &down_f,
                &input,
                &slots.map(|s| s as u32),
                &probability,
                ff,
            )
            .unwrap();
            let mut act = vec![f32::NAN; 10 * ff];
            let mut partial = vec![f32::NAN; 10 * hidden];
            let mut visits = vec![0; 10 * ff];
            let mut dvisits = vec![0; 10 * hidden];
            for (first, count) in [(0, 64), (64, 32)] {
                let gb = &gate[first * ff * 144..(first + count) * ff * 144];
                let ub = &up[first * ff * 144..(first + count) * ff * 144];
                let db = &down[first * hidden * drow..(first + count) * hidden * drow];
                for (rank, &slot) in slots.iter().enumerate() {
                    if slot < first || slot - first >= count {
                        continue;
                    }
                    let local = slot - first;
                    for feature in 0..ff {
                        let row = (local * ff + feature) * 144;
                        let mut g = vec![0.0; hidden];
                        let mut u = vec![0.0; hidden];
                        TensorType::Q4K.decode(&gb[row..row + 144], &mut g).unwrap();
                        TensorType::Q4K.decode(&ub[row..row + 144], &mut u).unwrap();
                        act[rank * ff + feature] = moe_silu(dot64(&g, &input)) * dot64(&u, &input);
                        visits[rank * ff + feature] += 1;
                    }
                    for feature in 0..hidden {
                        let row = (local * hidden + feature) * drow;
                        let mut d = vec![0.0; ff];
                        dtype.decode(&db[row..row + drow], &mut d).unwrap();
                        partial[rank * hidden + feature] =
                            dot64(&d, &act[rank * ff..(rank + 1) * ff]);
                        dvisits[rank * hidden + feature] += 1;
                    }
                }
            }
            assert!(visits.iter().chain(&dvisits).all(|&n| n == 1));
            let output =
                super::super::moe::reference_weighted_sum(&partial, &probability, hidden).unwrap();
            for (actual, expected) in [
                (&act, &original.activations),
                (&partial, &original.partials),
                (&output, &original.output),
            ] {
                assert_eq!(
                    actual.iter().map(|n| n.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|n| n.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
