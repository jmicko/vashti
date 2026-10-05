//! Cooperative packed-weight matvec; include as a child of `quant_matvec`.
//!
//! Launch `cooperative_matvec::launch::<R>` with the same arguments as the
//! parent's scalar kernel, but CubeDim::new_1d(64) and CubeCount(rows, 1, 1).
//! Each cube owns one row and adjacent lanes decode adjacent columns. No
//! subgroup size, subgroup features, shader f16, or backend-specific source is
//! required. The parent owns packed decoding and host layout validation.
//!
//! This changes f32 summation order, not stored weights or model operators.
//! Acceptance must use numerical/logit tolerances; serial bit identity is not
//! promised. GPU performance and numerical parity remain root's validation.
//! Respect the adapter's maximum workgroups per dimension. A larger matrix
//! (e.g. the vocabulary head) needs validated row views, not an oversized grid.

use super::{BF16, F16, Q4_K, Q5_0, Q6_K, Q8_0, packed_value};
use cubecl::prelude::*;

pub const LANES: usize = 64;

/// All cube barriers are unconditional. Out-of-range cubes still initialize
/// and reduce zero partials, then avoid output access. The inner full-tile loop
/// has a cube-uniform bound; only the final partial tile has lane predication.
///
/// Preconditions: parent's `validate_layout` succeeds; scalar ArrayArg values;
/// CubeDim=(64,1,1); CubeCount=(rows,1,1), within the adapter's grid limit.
#[cube(launch, launch_unchecked)]
pub fn cooperative_matvec(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    rows: usize,
    columns: usize,
    #[comptime] dtype: u32,
) {
    let row = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut partial = 0.0_f32;
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
        let row_offset = row * row_bytes;
        let full_tiles = columns / LANES;
        for tile in 0..full_tiles {
            let column = tile * LANES + lane;
            partial += packed_value(packed, row_offset, column, dtype) * input[column];
        }
        if lane < columns % LANES {
            let column = full_tiles * LANES + lane;
            partial += packed_value(packed, row_offset, column, dtype) * input[column];
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
        output[row] = sums[0];
    }
}

/// Pure-host oracle for this reduction schedule on already decoded weights.
/// Root should additionally compare the device result with independent
/// `TensorType::decode` and a higher-precision/scalar dot, not only this oracle.
pub fn reference_tree_dot(weights: &[f32], input: &[f32]) -> Result<f32, &'static str> {
    if weights.is_empty() || weights.len() != input.len() {
        return Err("decoded weights and input must have equal nonzero lengths");
    }
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
    Ok(sums[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_partial_tiles_match_independent_exact_dot() {
        // Dyadic products and bounded totals are exactly representable in f32,
        // so the independently accumulated f64 result must match bit-for-bit.
        for columns in [1, 5, 32, 63, 64, 65, 137, 256, 257, 640, 2560, 10240] {
            let weights: Vec<f32> = (0..columns)
                .map(|i| ((i % 29) as f32 - 14.0) / 32.0)
                .collect();
            let input: Vec<f32> = (0..columns)
                .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                .collect();
            let expected = weights
                .iter()
                .zip(&input)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32;
            assert_eq!(
                reference_tree_dot(&weights, &input).unwrap().to_bits(),
                expected.to_bits(),
                "columns={columns}"
            );
        }
    }

    #[test]
    fn every_lane_and_strided_tile_contributes_once() {
        let input = vec![1.0_f32; 129];
        for active in 0..129 {
            let mut weights = vec![0.0_f32; 129];
            weights[active] = 1.0;
            assert_eq!(
                reference_tree_dot(&weights, &input).unwrap(),
                1.0,
                "active column {active}"
            );
        }
        assert_eq!(reference_tree_dot(&vec![1.0; 129], &input).unwrap(), 129.0);
    }

    #[test]
    fn exposes_expected_f32_reassociation() {
        // Serial cancellation preserves the late 1; lane 0 loses it before
        // reducing with lane 1. This documents why serial bit equality is not
        // an acceptable universal assertion for a parallel f32 reduction.
        let mut weights = vec![0.0_f32; 128];
        weights[0] = 100_000_000.0;
        weights[1] = -100_000_000.0;
        weights[64] = 1.0;
        let input = vec![1.0_f32; 128];
        let serial = weights.iter().zip(&input).map(|(w, x)| w * x).sum::<f32>();
        assert_eq!(serial, 1.0);
        assert_eq!(reference_tree_dot(&weights, &input).unwrap(), 0.0);
    }

    #[test]
    fn rejects_empty_and_mismatched_vectors() {
        assert!(reference_tree_dot(&[], &[]).is_err());
        assert!(reference_tree_dot(&[1.0], &[]).is_err());
        assert!(reference_tree_dot(&[], &[1.0]).is_err());
    }
}
