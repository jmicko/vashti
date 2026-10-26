//! WG-shared packed K16 operands. Same eight C chains/K16 order.
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
pub(in crate::strata_matrix) fn compute(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    scratch: &mut SharedMemory<f32>,
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
) {
    let pair = comptime!(true);
    // SAFETY: every genuine fragment is assigned before each use; B1 is unused
    // in the single-member-fragment specialization and is removed at compile time.
    let a0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a1 = unsafe {
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
    let sg = usize::cast_from(PLANE_POS);
    let t = usize::cast_from(UNIT_POS);
    let first_input = token - (sg / 4) * 32;
    let first_weight = out_row - (sg % 4) * 32;
    let r = t / 4;
    let q = (t % 4) * 4;
    // Each thread owns four UNIQUE vec4 FP16 reads. Resolve all packed
    // row/parent bases once; each K16 advances B16 by64 and B8 by32vec4.
    let mut wi0 = (weight_first + super::packed_operands::weight_half(first_weight + r, q, columns)) / 4;
    let mut wi1 = (weight_first + super::packed_operands::weight_half(first_weight + 64 + r, q, columns)) / 4;
    let mut xi0 = super::packed_operands::input_half(first_input + r, q, columns) / 4;
    let mut xi1 = super::packed_operands::input_half(first_input + 64 + r, q, columns) / 4;
    let bf = 1024 + (sg / 4) * 256;
    let bf1 = bf + 128;
    let af0 = (sg % 4) * 256;
    let af1 = af0 + 64;
    let af2 = af0 + 128;
    let af3 = af0 + 192;
    let mut k = 0usize;
    while k < columns {
        // Equal-width OpBitcast only; packed f32 bits undergo no arithmetic.
        let w0 = Vector::<f32, Const<2>>::reinterpret(weight[wi0]);
        let w1 = Vector::<f32, Const<2>>::reinterpret(weight[wi1]);
        let x0 = Vector::<f32, Const<2>>::reinterpret(input[xi0]);
        let x1 = Vector::<f32, Const<2>>::reinterpret(input[xi1]);
        scratch[t * 2] = w0[0];
        scratch[t * 2 + 1] = w0[1];
        scratch[512 + t * 2] = w1[0];
        scratch[512 + t * 2 + 1] = w1[1];
        scratch[1024 + t * 2] = x0[0];
        scratch[1024 + t * 2 + 1] = x0[1];
        scratch[1536 + t * 2] = x1[0];
        scratch[1536 + t * 2 + 1] = x1[1];
        sync_cube();
        cmma::load(
            &b0,
            &scratch.to_slice().slice(bf, 2048usize),
            8u32,
        );
        if comptime!(pair) {
            cmma::load(
                &b1,
                &scratch.to_slice().slice(bf1, 2048usize),
                8u32,
            );
        }
        cmma::load(
            &a0,
            &scratch.to_slice().slice(af0, 2048usize),
            8u32,
        );
        cmma::load(
            &a1,
            &scratch.to_slice().slice(af1, 2048usize),
            8u32,
        );
        row::<f16>(&a0, &b0, &b1, c00, c01, pair);
        cmma::load(
            &a0,
            &scratch.to_slice().slice(af2, 2048usize),
            8u32,
        );
        row::<f16>(&a1, &b0, &b1, c10, c11, pair);
        cmma::load(
            &a1,
            &scratch.to_slice().slice(af3, 2048usize),
            8u32,
        );
        row::<f16>(&a0, &b0, &b1, c20, c21, pair);
        row::<f16>(&a1, &b0, &b1, c30, c31, pair);
        // All matrix users are finished before any panel overwrite.
        sync_cube();
        xi0 += 64;
        xi1 += 64;
        wi0 += 32;
        wi1 += 32;
        k += 16;
    }
}
