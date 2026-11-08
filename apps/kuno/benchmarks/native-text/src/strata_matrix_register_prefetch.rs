//! Two independent K16 operand sets, original chronological accumulations.
//! No new owner, layout, SLM, precision, output or publication operation.
use cubecl::prelude::*;

#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn accumulate<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    columns: usize,
    input_first: usize,
    weight_first: usize,
    a0: &cmma::Matrix<F>,
    a1: &cmma::Matrix<F>,
    bm: &cmma::Matrix<F>,
    c0: &cmma::Matrix<f32>,
    c1: &cmma::Matrix<f32>,
    #[comptime] tile: usize,
) {
    let next_a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let next_a1 = if comptime!(tile == 16) {
        unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    } else {
        next_a0
    };
    let next_b = unsafe {
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
        let af = input_first + k / 4;
        let bf = weight_first + k / 4;
        cmma::load(
            a0,
            &input.to_slice().slice(af, input.len()),
            u32::cast_from(columns),
        );
        if comptime!(tile == 16) {
            cmma::load(
                a1,
                &input.to_slice().slice(af + 8 * columns / 4, input.len()),
                u32::cast_from(columns),
            );
        }
        cmma::load(
            bm,
            &weight.to_slice().slice(bf, weight.len()),
            u32::cast_from(columns),
        );
        // Original host guards admit only complete K16 columns. The final
        // single block remains valid; never speculatively read a missing one.
        if k + 16 < columns {
            cmma::load(
                &next_a0,
                &input.to_slice().slice(af + 4, input.len()),
                u32::cast_from(columns),
            );
            if comptime!(tile == 16) {
                cmma::load(
                    &next_a1,
                    &input
                        .to_slice()
                        .slice(af + 4 + 8 * columns / 4, input.len()),
                    u32::cast_from(columns),
                );
            }
            cmma::load(
                &next_b,
                &weight.to_slice().slice(bf + 4, weight.len()),
                u32::cast_from(columns),
            );
        }
        cmma::execute::<F, F, f32, f32>(a0, bm, c0, c0);
        if comptime!(tile == 16) {
            cmma::execute::<F, F, f32, f32>(a1, bm, c1, c1);
        }
        if k + 16 < columns {
            cmma::execute::<F, F, f32, f32>(&next_a0, &next_b, c0, c0);
            if comptime!(tile == 16) {
                cmma::execute::<F, F, f32, f32>(&next_a1, &next_b, c1, c1);
            }
        }
        k += 32;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn guarded_two_block_walk_preserves_chronological_products_and_every_operand_extent() {
        for columns in (16..=2560usize).step_by(16) {
            for tile in [8, 16] {
                let original = (0..columns)
                    .step_by(16)
                    .flat_map(|k| (0..tile / 8).map(move |m| (k, m)))
                    .collect::<Vec<_>>();
                let mut actual = Vec::new();
                for k in (0..columns).step_by(32) {
                    for kk in [k, k + 16].into_iter().filter(|&v| v < columns) {
                        for m in 0..tile / 8 {
                            actual.push((kk, m));
                        }
                        for row in 0..8 {
                            assert!((row * columns + kk + 15) / 4 < tile * columns / 4);
                            assert!((15 * columns + kk + 15) / 4 < 16 * columns / 4);
                        }
                    }
                }
                assert_eq!(actual, original);
            }
        }
    }
}
