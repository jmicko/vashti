//! The same genuine N128/four-C owner with selectable temporal operand carry.
use super::*;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(in crate::strata_matrix) fn compute(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    c0: &cmma::Matrix<f32>,
    c1: &cmma::Matrix<f32>,
    c2: &cmma::Matrix<f32>,
    c3: &cmma::Matrix<f32>,
    token: usize,
    row: usize,
    columns: usize,
    weight_first: usize,
    #[comptime] carry: bool,
) {
    if comptime!(carry) {
        // The retained four-C core loads bank0 before the loop, bank1 before
        // current products, and bank0 K+32 only when another iteration exists.
        // Both genuine roles have K divisible by32; no speculative final read.
        crate::strata_matrix::native_wide::balanced_wide32::compute::<f16>(
            input,
            weight,
            c0,
            c1,
            c2,
            c3,
            token,
            row,
            columns,
            weight_first,
        );
    } else {
        crate::strata_matrix::local_products::compute::<f16>(
            input,
            weight,
            c0,
            c0,
            c1,
            c1,
            c2,
            c2,
            c3,
            c3,
            token,
            row,
            columns,
            weight_first,
            false,
        );
    }
}
