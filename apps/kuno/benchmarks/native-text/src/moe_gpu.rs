//! Three Rust/CubeCL kernels for a full selected-expert MoE, as a child module
//! of `quant_matvec`. Packed banks are tightly stored [slot][output][input].
//! Slots and probabilities stay in the host's selected-expert order; this
//! module neither reroutes, prunes, renormalizes nor adds the shared expert.
//! All packed types use the parent's already validated `packed_value` decoder.
//! The matvec phases reassociate f32 sums into a 64-lane tree, requiring the
//! same numerical/logit acceptance checks as the cooperative dense prototype.
//!
//! gate/up: five arrays (gate, up, input, slots, activations); grid K*FF x 64.
//! down: four arrays (down, activations, slots, partials); grid K*D x 64.
//! reduce: three arrays (partials, probabilities, output); ceil(D/64) x 64.
//! Each array uses scalar elements. Only the full packed bank is u32-padded;
//! rows/slots must not acquire extra padding. Root owns paging and lifetimes.

use super::{BF16, F16, Q4_K, Q5_0, Q6_K, Q8_0, packed_value};
use cubecl::prelude::*;

pub const LANES: usize = 64;
pub const MAX_SELECTED: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeShape {
    pub selected: usize,
    pub hidden: usize,
    pub feed_forward: usize,
    pub gate_up_cubes: u32,
    pub down_cubes: u32,
    pub reduce_cubes: u32,
    pub activation_elements: usize,
    pub partial_elements: usize,
}

impl MoeShape {
    /// Check dimensions/grid sizes before dispatch. The caller additionally
    /// validates each packed bank's format, row geometry, byte bounds and slot
    /// IDs, and keeps all selected slots stable through both matvec phases.
    pub fn new(
        selected: usize,
        hidden: usize,
        feed_forward: usize,
        grid_limit: u32,
    ) -> Result<Self, &'static str> {
        if selected == 0 || selected > MAX_SELECTED || hidden == 0 || feed_forward == 0 {
            return Err("MoE needs 1..10 selected experts and nonzero dimensions");
        }
        let activation_elements = selected
            .checked_mul(feed_forward)
            .ok_or("activation shape overflow")?;
        let partial_elements = selected
            .checked_mul(hidden)
            .ok_or("partial shape overflow")?;
        let max = (u32::MAX as usize) / 4;
        if activation_elements > max || partial_elements > max {
            return Err("MoE intermediate exceeds prototype's 32-bit byte bounds");
        }
        let gate_up_cubes =
            u32::try_from(activation_elements).map_err(|_| "gate/up grid overflow")?;
        let down_cubes = u32::try_from(partial_elements).map_err(|_| "down grid overflow")?;
        let reduce_cubes =
            u32::try_from(hidden.div_ceil(LANES)).map_err(|_| "reduce grid overflow")?;
        if grid_limit == 0
            || gate_up_cubes > grid_limit
            || down_cubes > grid_limit
            || reduce_cubes > grid_limit
        {
            return Err("MoE grid exceeds adapter workgroup limit");
        }
        Ok(Self {
            selected,
            hidden,
            feed_forward,
            gate_up_cubes,
            down_cubes,
            reduce_cubes,
            activation_elements,
            partial_elements,
        })
    }
}

#[cube]
fn packed_row_bytes(columns: usize, #[comptime] dtype: u32) -> usize {
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

/// Same sign-stable sigmoid expression as kuno-text::ops::silu.
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

/// CubeDim(64,1,1), CubeCount(K*FF,1,1). Output is [selected rank][FF].
/// Runtime scalars: selected, hidden, feed_forward; then two compile-time IDs.
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
        let gate_offset = bank_row * packed_row_bytes(hidden, gate_dtype);
        let up_offset = bank_row * packed_row_bytes(hidden, up_dtype);
        let full_tiles = hidden / LANES;
        for tile in 0..full_tiles {
            let column = tile * LANES + lane;
            let value = input[column];
            gate += packed_value(gate_bank, gate_offset, column, gate_dtype) * value;
            up += packed_value(up_bank, up_offset, column, up_dtype) * value;
        }
        if lane < hidden % LANES {
            let column = full_tiles * LANES + lane;
            let value = input[column];
            gate += packed_value(gate_bank, gate_offset, column, gate_dtype) * value;
            up += packed_value(up_bank, up_offset, column, up_dtype) * value;
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

/// CubeDim(64,1,1), CubeCount(K*D,1,1). Output is [selected rank][D].
/// Runtime scalars: selected, hidden, feed_forward; then compile-time down ID.
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
    let mut partial = 0.0_f32;
    if task < selected * hidden {
        let rank = task / hidden;
        let feature = task % hidden;
        let bank_row = (slots[rank] as usize) * hidden + feature;
        let row_offset = bank_row * packed_row_bytes(feed_forward, down_dtype);
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
    if lane == 0 && task < selected * hidden {
        partials[task] = sums[0];
    }
}

/// CubeDim(64,1,1), CubeCount(ceil(D/64),1,1). Preserve the host's probability
/// order explicitly: no tree, atomics, rerouting, pruning or renormalization.
#[cube(launch, launch_unchecked)]
pub fn moe_reduce(
    partials: &Array<f32>,
    probabilities: &Array<f32>,
    output: &mut Array<f32>,
    selected: usize,
    hidden: usize,
) {
    let feature = ABSOLUTE_POS;
    if feature < hidden {
        let mut sum = 0.0_f32;
        for rank in 0..selected {
            sum += probabilities[rank] * partials[rank * hidden + feature];
        }
        output[feature] = sum;
    }
}

#[cfg(test)]
#[derive(Debug)]
pub struct MoeReference {
    pub activations: Vec<f32>,
    pub partials: Vec<f32>,
    pub output: Vec<f32>,
}

#[cfg(test)]
fn validate_probabilities(probabilities: &[f32]) -> Result<(), &'static str> {
    if probabilities.is_empty()
        || probabilities.len() > MAX_SELECTED
        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err("probabilities must be finite nonnegative weights for 1..10 experts");
    }
    let sum: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
    if (sum - 1.0).abs() > 1e-5 {
        return Err("selected probabilities must already be normalized");
    }
    Ok(())
}

#[cfg(test)]
fn reference_dot(weights: &[f32], input: &[f32]) -> f32 {
    let mut sums = [0.0_f32; LANES];
    for (column, (&weight, &value)) in weights.iter().zip(input).enumerate() {
        sums[column % LANES] += weight * value;
    }
    let mut stride = LANES / 2;
    while stride > 0 {
        for lane in 0..stride {
            sums[lane] += sums[lane + stride];
        }
        stride /= 2;
    }
    sums[0]
}

/// Host oracle for the final rank-ordered f32 accumulation.
#[cfg(test)]
pub fn reference_weighted_sum(
    partials: &[f32],
    probabilities: &[f32],
    hidden: usize,
) -> Result<Vec<f32>, &'static str> {
    validate_probabilities(probabilities)?;
    if hidden == 0 || probabilities.len().checked_mul(hidden) != Some(partials.len()) {
        return Err("partial matrix shape mismatch");
    }
    let mut output = vec![0.0_f32; hidden];
    for feature in 0..hidden {
        for rank in 0..probabilities.len() {
            output[feature] += probabilities[rank] * partials[rank * hidden + feature];
        }
    }
    Ok(output)
}

/// Small-fixture host oracle on already independently decoded dense banks.
/// Root's device acceptance should also compare higher-precision intermediates
/// and selected-token behavior, not only this matching reduction schedule.
#[cfg(test)]
pub fn reference_grouped_moe(
    gate_bank: &[f32],
    up_bank: &[f32],
    down_bank: &[f32],
    input: &[f32],
    slots: &[u32],
    probabilities: &[f32],
    feed_forward: usize,
) -> Result<MoeReference, &'static str> {
    validate_probabilities(probabilities)?;
    if slots.len() != probabilities.len() {
        return Err("slot and probability count mismatch");
    }
    let shape = MoeShape::new(slots.len(), input.len(), feed_forward, u32::MAX)?;
    let per_slot = shape
        .hidden
        .checked_mul(feed_forward)
        .ok_or("bank shape overflow")?;
    if gate_bank.is_empty()
        || !gate_bank.len().is_multiple_of(per_slot)
        || up_bank.len() != gate_bank.len()
        || down_bank.len() != gate_bank.len()
    {
        return Err("dense fixture banks must contain equally sized whole slots");
    }
    let slot_count = gate_bank.len() / per_slot;
    if slots.iter().any(|&slot| slot as usize >= slot_count) {
        return Err("selected slot outside bank");
    }
    let mut activations = vec![0.0; shape.activation_elements];
    let mut partials = vec![0.0; shape.partial_elements];
    for (rank, &slot) in slots.iter().enumerate() {
        for feature in 0..feed_forward {
            let offset = (slot as usize * feed_forward + feature) * shape.hidden;
            let gate = reference_dot(&gate_bank[offset..offset + shape.hidden], input);
            let up = reference_dot(&up_bank[offset..offset + shape.hidden], input);
            activations[rank * feed_forward + feature] = moe_silu(gate) * up;
        }
        let activation = &activations[rank * feed_forward..(rank + 1) * feed_forward];
        for feature in 0..shape.hidden {
            let offset = (slot as usize * shape.hidden + feature) * feed_forward;
            partials[rank * shape.hidden + feature] =
                reference_dot(&down_bank[offset..offset + feed_forward], activation);
        }
    }
    let output = reference_weighted_sum(&partials, probabilities, shape.hidden)?;
    Ok(MoeReference {
        activations,
        partials,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_graph_grid_and_intermediate_sizes() {
        let shape = MoeShape::new(10, 2560, 640, 65535).unwrap();
        assert_eq!(shape.gate_up_cubes, 6400);
        assert_eq!(shape.down_cubes, 25600);
        assert_eq!(shape.reduce_cubes, 40);
        assert_eq!(shape.activation_elements * 4, 25600);
        assert_eq!(shape.partial_elements * 4, 102400);
        assert!(MoeShape::new(10, 2560, 640, 25600).is_ok());
        assert!(MoeShape::new(10, 2560, 640, 25599).is_err());
        for (k, d, f) in [
            (0, 4, 3),
            (11, 4, 3),
            (1, 0, 3),
            (1, 4, 0),
            (10, usize::MAX, 3),
            (10, 4, usize::MAX),
        ] {
            assert!(MoeShape::new(k, d, f, u32::MAX).is_err());
        }
    }

    #[test]
    fn final_sum_preserves_probability_rank_order() {
        // Weighted first-column terms are [2^24,1,-2^24]. Left-fold gives
        // zero; cancellation-first reassociation would give one.
        let partials = [33554432.0, 1.0, 4.0, 4.0, 8.0, 1.0, -67108864.0, -4.0, 8.0];
        let actual = reference_weighted_sum(&partials, &[0.5, 0.25, 0.25], 3).unwrap();
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0.0_f32, 1.5, 4.25]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn nontrivial_slot_remapping_matches_independent_f64_moe() {
        let (bank_slots, hidden, ff) = (5_usize, 4_usize, 3_usize);
        let mut gate = vec![0.0; bank_slots * ff * hidden];
        let mut up = gate.clone();
        let mut down = gate.clone();
        for slot in 0..bank_slots {
            for feature in 0..ff {
                for column in 0..hidden {
                    let offset = (slot * ff + feature) * hidden + column;
                    gate[offset] = (slot + feature + column + 1) as f32 / 16.0;
                    up[offset] = ((slot * 11 + feature * 3 + column) as f32 - 17.0) / 16.0;
                }
            }
            for feature in 0..hidden {
                for column in 0..ff {
                    down[(slot * hidden + feature) * ff + column] =
                        ((slot * 13 + feature * 5 + column) as f32 - 25.0) / 32.0;
                }
            }
        }
        let input = [0.5_f32, -1.0, 0.25, 2.0];
        let slots = [2_u32, 0, 4];
        let probabilities = [0.5_f32, 0.3, 0.2];
        let actual =
            reference_grouped_moe(&gate, &up, &down, &input, &slots, &probabilities, ff).unwrap();
        let mut expected = [0.0_f64; 4];
        for (rank, &slot) in slots.iter().enumerate() {
            let mut activation = [0.0_f64; 3];
            for (feature, a) in activation.iter_mut().enumerate() {
                let offset = (slot as usize * ff + feature) * hidden;
                let g: f64 = (0..hidden)
                    .map(|c| f64::from(gate[offset + c]) * f64::from(input[c]))
                    .sum();
                let u: f64 = (0..hidden)
                    .map(|c| f64::from(up[offset + c]) * f64::from(input[c]))
                    .sum();
                *a = (g / (1.0 + (-g).exp())) * u;
                assert!((f64::from(actual.activations[rank * ff + feature]) - *a).abs() < 1e-6);
            }
            for (feature, e) in expected.iter_mut().enumerate() {
                let offset = (slot as usize * hidden + feature) * ff;
                let value: f64 = (0..ff)
                    .map(|c| f64::from(down[offset + c]) * activation[c])
                    .sum();
                assert!((f64::from(actual.partials[rank * hidden + feature]) - value).abs() < 2e-6);
                *e += f64::from(probabilities[rank]) * value;
            }
        }
        for (feature, e) in expected.iter().enumerate() {
            assert!((f64::from(actual.output[feature]) - e).abs() < 2e-6);
        }
        let wrong_rank_slots =
            reference_grouped_moe(&gate, &up, &down, &input, &[0, 1, 2], &probabilities, ff)
                .unwrap();
        assert!(
            actual
                .output
                .iter()
                .zip(&wrong_rank_slots.output)
                .any(|(a, b)| (a - b).abs() > 0.05)
        );
    }

    #[test]
    fn all_ten_experts_contribute_and_stable_silu_handles_signs() {
        let slots: Vec<u32> = (0..10).rev().collect();
        let actual = reference_grouped_moe(
            &[1.0; 10],
            &[1.0; 10],
            &(1..=10).map(|n| n as f32).collect::<Vec<_>>(),
            &[1.0],
            &slots,
            &[0.1; 10],
            1,
        )
        .unwrap();
        assert!(actual.activations.iter().all(|v| *v > 0.0));
        let expected = 5.5_f64 / (1.0 + (-1.0_f64).exp());
        assert!((f64::from(actual.output[0]) - expected).abs() < 1e-6);
        for value in [-80.0_f32, -8.0, -1.0, 0.0, 1.0, 8.0, 80.0] {
            let expected = f64::from(value) / (1.0 + (-f64::from(value)).exp());
            let tolerance = expected.abs() * 2e-6 + 1e-36;
            assert!((f64::from(moe_silu(value)) - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn mixed_gate_signs_and_zero_follow_resident_slot_order() {
        let gate_pairs = [
            [(0, 0.0), (1, -1.0), (2, 1.0)],
            [(3, 1.0), (0, -1.0), (1, 1.0)],
            [(2, -1.0), (3, -1.0), (0, 1.0)],
        ];
        let up_pairs = [
            [(3, -2.0), (0, 3.0), (1, -2.0)],
            [(2, 2.0), (3, -1.0), (0, 5.0)],
            [(1, -3.0), (2, 1.0), (3, -2.0)],
        ];
        let mut gate = [0.0_f32; 36];
        let mut up = [0.0_f32; 36];
        for slot in 0..3 {
            for feature in 0..3 {
                let (column, value) = gate_pairs[slot][feature];
                gate[(slot * 3 + feature) * 4 + column] = value;
                let (column, value) = up_pairs[slot][feature];
                up[(slot * 3 + feature) * 4 + column] = value;
            }
        }
        let actual = reference_grouped_moe(
            &gate,
            &up,
            &[1.0; 36],
            &[1.0, -2.0, 3.0, -4.0],
            &[2, 0, 1],
            &[0.5, 0.25, 0.25],
            3,
        )
        .unwrap();
        let expected_gates = [-3.0_f64, 4.0, 1.0, 0.0, 2.0, 3.0, -4.0, -1.0, -2.0];
        let expected_ups = [6.0_f64, 3.0, 8.0, 8.0, 3.0, 4.0, 6.0, 4.0, 5.0];
        for feature in 0..9 {
            let gate = expected_gates[feature];
            let expected = gate / (1.0 + (-gate).exp()) * expected_ups[feature];
            assert!((f64::from(actual.activations[feature]) - expected).abs() < 2e-6);
        }
        assert_eq!(actual.activations[3].to_bits(), 0.0_f32.to_bits());
        assert!(actual.activations[0] < 0.0 && actual.activations[1] > 0.0);
    }

    #[test]
    fn rejects_invalid_dense_fixture_and_probability_contracts() {
        for probabilities in [
            &[][..],
            &[0.2][..],
            &[-1.0, 2.0][..],
            &[f32::NAN][..],
            &[f32::INFINITY][..],
        ] {
            assert!(reference_weighted_sum(&[1.0], probabilities, 1).is_err());
        }
        assert!(reference_weighted_sum(&[1.0], &[1.0], 2).is_err());
        assert!(reference_grouped_moe(&[1.0], &[1.0], &[1.0], &[1.0], &[1], &[1.0], 1).is_err());
        assert!(reference_grouped_moe(&[1.0], &[1.0], &[1.0], &[1.0], &[], &[1.0], 1).is_err());
        assert!(reference_grouped_moe(&[1.0], &[1.0], &[], &[1.0], &[0], &[1.0], 1).is_err());
    }
}
