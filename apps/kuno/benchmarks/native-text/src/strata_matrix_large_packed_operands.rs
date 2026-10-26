//! LARGE-only size-preserving tiled operands. Same eight C chains/K16 order.
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
) {
    let pair = comptime!(true);
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
        let bf = input_half(token,k,columns) / 4;
        cmma::load(
            &b0,
            &input.to_slice().slice(bf, input.len()),
            16u32,
        );
        if comptime!(pair) {
            cmma::load(
                &b1,
                &input.to_slice().slice(input_half(token + 16,k,columns) / 4, input.len()),
                16u32,
            );
        }
        cmma::load(
            &a0,
            &weight
                .to_slice()
                .slice((weight_first + weight_half(out_row,k,columns)) / 4, weight.len()),
            16u32,
        );
        cmma::load(
            &a1,
            &weight.to_slice().slice(
                (weight_first + weight_half(out_row + 8,k,columns)) / 4,
                weight.len(),
            ),
            16u32,
        );
        row::<F>(&a0, &b0, &b1, c00, c01, pair);
        cmma::load(
            &a0,
            &weight.to_slice().slice(
                (weight_first + weight_half(out_row + 16,k,columns)) / 4,
                weight.len(),
            ),
            16u32,
        );
        row::<F>(&a1, &b0, &b1, c10, c11, pair);
        cmma::load(
            &a1,
            &weight.to_slice().slice(
                (weight_first + weight_half(out_row + 24,k,columns)) / 4,
                weight.len(),
            ),
            16u32,
        );
        row::<F>(&a0, &b0, &b1, c20, c21, pair);
        row::<F>(&a1, &b0, &b1, c30, c31, pair);
        k += 16;
    }
}

/// Parent/group offsets are outside this bijection. Pad32 prefixes are aligned
/// to both B=16 and B=8, so input global member addressing preserves each owner.
#[cube]
pub(super) fn input_half(row:usize,k:usize,columns:usize)->usize {
    row/16*(16*columns)+(k/16)*256+(row%16)*16+k%16
}
#[cube]
pub(super) fn weight_half(row:usize,k:usize,columns:usize)->usize {
    row/8*(8*columns)+(k/16)*128+(row%8)*16+k%16
}

// The original checked route fallback has two A fragments and one B fragment.
// Both row blocks are addressed independently: adjacent 8-row weight blocks are
// separated by8*K rather than an8*16 row-stride inside a single tile.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn load<F:Float>(
    input:&Array<Vector<F,Const<4>>>, weight:&Array<Vector<F,Const<4>>>,
    a0:&cmma::Matrix<F>, a1:&cmma::Matrix<F>, b:&cmma::Matrix<F>,
    token:usize,row:usize,k:usize,columns:usize,weight_first:usize,
) {
    let af=(weight_first+weight_half(row,k,columns))/4;
    let af1=(weight_first+weight_half(row+8,k,columns))/4;
    let bf=input_half(token,k,columns)/4;
    cmma::load(a0,&weight.to_slice().slice(af,weight.len()),16u32);
    cmma::load(a1,&weight.to_slice().slice(af1,weight.len()),16u32);
    cmma::load(b,&input.to_slice().slice(bf,input.len()),16u32);
}
