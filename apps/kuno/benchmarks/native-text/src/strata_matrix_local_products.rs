//! Shared local operand prefetch, chronological K16, no carried A/B values.
use super::*;
#[cube]
#[allow(clippy::too_many_arguments)]
fn row<F: Float>(
    a: &cmma::Matrix<F>,
    b0: &cmma::Matrix<F>,
    b1: &cmma::Matrix<F>,
    c0: &cmma::Matrix<f32>,
    c1: &cmma::Matrix<f32>,
    #[comptime] pair: bool,
) {
    cmma::execute::<F, F, f32, f32>(a, b0, c0, c0);
    if comptime!(pair) {
        cmma::execute::<F, F, f32, f32>(a, b1, c1, c1);
    }
}
#[cube]
#[allow(clippy::too_many_arguments)]
pub(in crate::strata_matrix) fn compute<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    c00: &cmma::Matrix<f32>,
    c01: &cmma::Matrix<f32>,
    c10: &cmma::Matrix<f32>,
    c11: &cmma::Matrix<f32>,
    c20: &cmma::Matrix<f32>,
    c21: &cmma::Matrix<f32>,
    c30: &cmma::Matrix<f32>,
    c31: &cmma::Matrix<f32>,
    token: usize,
    out_row: usize,
    columns: usize,
    weight_first: usize,
    #[comptime] pair: bool,
) {
    // SAFETY: every genuine fragment is assigned before each use; B1 is unused
    // in the single-member-fragment specialization and is removed at compile time.
    let a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let b0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let b1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let mut k = 0usize;
    while k < columns {
        let bf = (token * columns + k) / 4;
        cmma::load(
            &b0,
            &input.to_slice().slice(bf, input.len()),
            u32::cast_from(columns),
        );
        if comptime!(pair) {
            cmma::load(
                &b1,
                &input.to_slice().slice(bf + 16 * columns / 4, input.len()),
                u32::cast_from(columns),
            );
        }
        cmma::load(
            &a0,
            &weight
                .to_slice()
                .slice((weight_first + out_row * columns + k) / 4, weight.len()),
            u32::cast_from(columns),
        );
        cmma::load(
            &a1,
            &weight.to_slice().slice(
                (weight_first + (out_row + 8) * columns + k) / 4,
                weight.len(),
            ),
            u32::cast_from(columns),
        );
        row::<F>(&a0, &b0, &b1, c00, c01, pair);
        cmma::load(
            &a0,
            &weight.to_slice().slice(
                (weight_first + (out_row + 16) * columns + k) / 4,
                weight.len(),
            ),
            u32::cast_from(columns),
        );
        row::<F>(&a1, &b0, &b1, c10, c11, pair);
        cmma::load(
            &a1,
            &weight.to_slice().slice(
                (weight_first + (out_row + 24) * columns + k) / 4,
                weight.len(),
            ),
            u32::cast_from(columns),
        );
        row::<F>(&a0, &b0, &b1, c20, c21, pair);
        row::<F>(&a1, &b0, &b1, c30, c31, pair);
        k += 16;
    }
}
