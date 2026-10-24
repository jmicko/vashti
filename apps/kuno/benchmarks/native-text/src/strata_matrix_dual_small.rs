// Pure-global two-C cells: 16 members x16 output rows per SG16.
// Original real grid, padded producers, independent8-half publication and ABI.
use super::*;
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
    let row = usize::cast_from(CUBE_POS_X) * 32 + (sg % 2) * 16;
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
    crate::strata_matrix::dual_products::compute(
        input,
        weight,
        &c0,
        &c1,
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
    }
}
