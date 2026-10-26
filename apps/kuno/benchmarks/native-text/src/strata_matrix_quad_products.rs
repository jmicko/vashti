// Pure-global four-C cells: 16 members x32 output rows per SG16.
// Original real grid, padded producers, independent8-half publication and ABI.
use super::*;
use half::f16;
const PADDED_GLOBAL: bool = true;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn product(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    group: usize,
    input_member_base: usize,
    gu_prefix: usize,
    #[comptime] down: bool,
    #[comptime] tile: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let token = usize::cast_from(CUBE_POS_Y) * comptime!(tile) + (sg / 2) * 16;
    let row = usize::cast_from(CUBE_POS_X) * 64 + (sg % 2) * 32;
    if token >= count {
        terminate!();
    }
    let prefix = usize::cast_from(map[1028 + group]);
    let max_route = usize::cast_from(map[0]) * 10;
    let mut good = false;
    if token + 16 <= count {
        good = usize::cast_from(map[1540 + prefix + token + lane]) < max_route;
    }
    let all_good = plane_all(good);
    let c0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    compute(
        input,
        weight,
        &c0,
        &c1,
        &c2,
        &c3,
        input_member_base + token,
        row,
        columns,
        weight_first,
    );
    if all_good {
        let mut first = (gu_prefix + token) * rows + row;
        if comptime!(down) {
            first = (prefix + token) * rows + row;
        }
        cmma::store(
            &mut output.slice_mut(first + 0, output.len()),
            &c0,
            u32::cast_from(rows),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 8, output.len()),
            &c1,
            u32::cast_from(rows),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 16, output.len()),
            &c2,
            u32::cast_from(rows),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 24, output.len()),
            &c3,
            u32::cast_from(rows),
            cmma::MatrixLayout::ColMajor,
        );
        if comptime!(down) && row == 0 {
            let route = usize::cast_from(map[1540 + prefix + token + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
        }
    } else {
        let mut scratch = SharedMemory::<f32>::new(256usize);
        crate::strata_matrix::global_small::publish(
            &c0,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token,
            row + 0,
            count,
            gu_prefix,
            group,
            down,
        );
        crate::strata_matrix::global_small::publish(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token,
            row + 8,
            count,
            gu_prefix,
            group,
            down,
        );
        crate::strata_matrix::global_small::publish(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token,
            row + 16,
            count,
            gu_prefix,
            group,
            down,
        );
        crate::strata_matrix::global_small::publish(
            &c3,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token,
            row + 24,
            count,
            gu_prefix,
            group,
            down,
        );
    }
}

// Guarded K32 carry: eight A, two B and four C fragments.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_a(
    weight: &Array<Vector<f16, Const<4>>>,
    a0: &cmma::Matrix<f16>,
    a1: &cmma::Matrix<f16>,
    a2: &cmma::Matrix<f16>,
    a3: &cmma::Matrix<f16>,
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
    cmma::load(
        a2,
        &weight
            .to_slice()
            .slice((first + (row + 16) * columns + k) / 4, weight.len()),
        u32::cast_from(columns),
    );
    cmma::load(
        a3,
        &weight
            .to_slice()
            .slice((first + (row + 24) * columns + k) / 4, weight.len()),
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
    let a02 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a03 = unsafe {
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
    let a12 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a13 = unsafe {
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
    load_a(weight, &a00, &a01, &a02, &a03, row, 0usize, columns, first);
    let mut k = 0usize;
    while k < columns {
        load_b(input, &b1, token, k + 16, columns);
        load_a(weight, &a10, &a11, &a12, &a13, row, k + 16, columns, first);
        cmma::execute::<f16, f16, f32, f32>(&a00, &b0, c0, c0);
        cmma::execute::<f16, f16, f32, f32>(&a01, &b0, c1, c1);
        cmma::execute::<f16, f16, f32, f32>(&a02, &b0, c2, c2);
        cmma::execute::<f16, f16, f32, f32>(&a03, &b0, c3, c3);
        if k + 32 < columns {
            load_b(input, &b0, token, k + 32, columns);
            load_a(weight, &a00, &a01, &a02, &a03, row, k + 32, columns, first);
        }
        cmma::execute::<f16, f16, f32, f32>(&a10, &b1, c0, c0);
        cmma::execute::<f16, f16, f32, f32>(&a11, &b1, c1, c1);
        cmma::execute::<f16, f16, f32, f32>(&a12, &b1, c2, c2);
        cmma::execute::<f16, f16, f32, f32>(&a13, &b1, c3, c3);
        k += 32;
    }
}
