//! Interleaved partial bank carry: four A, two B and four C fragments.
use super::*;
use half::f16;
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_a(
    weight: &Array<Vector<f16, Const<4>>>,
    a0: &cmma::Matrix<f16>,
    a1: &cmma::Matrix<f16>,
    row: usize,
    k: usize,
    columns: usize,
    first: usize,
) {
    cmma::load(
        a0,
        &weight
            .to_slice()
            .slice((first + row * columns + k) / 4, weight.len()),
        u32::cast_from(columns),
    );
    cmma::load(
        a1,
        &weight
            .to_slice()
            .slice((first + (row + 8) * columns + k) / 4, weight.len()),
        u32::cast_from(columns),
    );
}
#[cube]
fn load_b(
    input: &Array<Vector<f16, Const<4>>>,
    b: &cmma::Matrix<f16>,
    token: usize,
    k: usize,
    columns: usize,
) {
    cmma::load(
        b,
        &input
            .to_slice()
            .slice((token * columns + k) / 4, input.len()),
        u32::cast_from(columns),
    );
}
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
    first: usize,
) {
    let a00 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a01 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a10 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a11 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let b0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let b1 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };

    load_b(input, &b0, token, 0usize, columns);
    load_a(weight, &a00, &a01, row, 0usize, columns, first);
    let mut k = 0usize;
    while k < columns {
        load_b(input, &b1, token, k + 16, columns);
        load_a(weight, &a10, &a11, row, k + 16, columns, first);
        cmma::execute::<f16, f16, f32, f32>(&a00, &b0, c0, c0);
        cmma::execute::<f16, f16, f32, f32>(&a01, &b0, c1, c1);
        load_a(weight, &a00, &a01, row + 16, k, columns, first);
        cmma::execute::<f16, f16, f32, f32>(&a10, &b1, c0, c0);
        cmma::execute::<f16, f16, f32, f32>(&a11, &b1, c1, c1);
        load_a(weight, &a10, &a11, row + 16, k + 16, columns, first);
        cmma::execute::<f16, f16, f32, f32>(&a00, &b0, c2, c2);
        cmma::execute::<f16, f16, f32, f32>(&a01, &b0, c3, c3);
        if k + 32 < columns {
            load_b(input, &b0, token, k + 32, columns);
            load_a(weight, &a00, &a01, row, k + 32, columns, first);
        }
        cmma::execute::<f16, f16, f32, f32>(&a10, &b1, c2, c2);
        cmma::execute::<f16, f16, f32, f32>(&a11, &b1, c3, c3);
        k += 32;
    }
}
