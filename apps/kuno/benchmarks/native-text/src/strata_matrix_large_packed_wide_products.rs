//! Distinct packed operands; original eight-C prefix and checked publication.
use super::*;
// The transposed C covers16 members by8 columns. Route failure semantics
// still belong to each original8-member half, independently of the other half.
#[cube]
#[allow(clippy::too_many_arguments)]
fn publish(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    member0: usize,
    row0: usize,
    count: usize,
    gu_prefix: usize,
    group: usize,
    #[comptime] down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let prefix = usize::cast_from(map[1028 + group]);
    let rows = comptime!(if down { 2560usize } else { 1280usize });
    let complete = member0 + 16 <= count && row0 + 8 <= rows;
    let mut good = false;
    if complete {
        let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
        good = route < usize::cast_from(map[0]) * 10;
    }
    let low_good = plane_all(lane >= 8 || good);
    let high_good = plane_all(lane < 8 || good);
    if low_good && high_good {
        let mut first = (gu_prefix + member0) * 1280 + row0;
        let mut stride = 1280u32;
        if comptime!(down) {
            first = (prefix + member0) * 2560 + row0;
            stride = 2560u32;
        }
        cmma::store(
            &mut output.slice_mut(first, output.len()),
            acc,
            stride,
            cmma::MatrixLayout::ColMajor,
        );
        if comptime!(down) && row0 == 0 {
            let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
        }
    } else {
        cmma::store(
            &mut scratch.slice_mut(sg * 128, sg * 128 + 128),
            acc,
            8,
            cmma::MatrixLayout::ColMajor,
        );
        sync_plane();
        #[unroll]
        for round in 0..8usize {
            let i = lane + round * 16;
            let member = member0 + i / 8;
            let row = row0 + i % 8;
            if member < count && row < rows {
                let route = usize::cast_from(map[1540 + prefix + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let value = scratch[sg * 128 + i];
                    let mut first = (gu_prefix + member) * 1280;
                    if comptime!(down) {
                        first = (prefix + member) * 2560;
                    }
                    let raw_half = (i / 8 < 8 && low_good) || (i / 8 >= 8 && high_good);
                    if raw_half {
                        output[first + row] = value;
                        if comptime!(down) && row == 0 {
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
                        }
                    } else {
                        let finite = value.abs() <= f32::MAX;
                        if finite {
                            output[first + row] = value;
                        } else {
                            output[first + row] = 0.0;
                            fault[0].fetch_or(128);
                        }
                        if comptime!(down) && row == 0 {
                            let mut valid = 0u32;
                            if finite && status[route] == 1 {
                                valid = 1;
                            }
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = valid;
                        }
                    }
                } else {
                    fault[0].fetch_or(128);
                }
            }
        }
        sync_plane();
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn wide(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    packed_prefix: u32,
    input_prefix: u32,
    tile_index: u32,
    piece_members: u32,
    #[comptime] down: bool,
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let group = usize::cast_from(group);
    let weight_first = usize::cast_from(weight_first);
    let packed_prefix = usize::cast_from(packed_prefix);
    let input_prefix=usize::cast_from(input_prefix);
    if !plane_all(crate::strata_matrix::native_expert_wide::header_valid(map, count, group))
        || packed_prefix > comptime!(crate::strata_prompt::groups::CAPACITY) || ((count+31)/32)*32 > comptime!(crate::strata_prompt::groups::CAPACITY)-packed_prefix
        || usize::cast_from(piece_members) != count / 128 * 128
        || columns != comptime!(if down { 640 } else { 2560 })
        || rows != comptime!(if down { 2560 } else { 1280 })
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Route ownership and the fallback mapping below are one coupled change.
    // Unlike a global sticky-fault read, these independent map reads preserve
    // the backend's original unused-result AtomicOr-only fault classification.
    let raw_good = if comptime!(cfg!(feature = "strata-wide32-local-routes")) {
        let sg = usize::cast_from(PLANE_POS);
        let lane = usize::cast_from(UNIT_POS_PLANE);
        let token = usize::cast_from(tile_index) * 128 + (sg / 4) * 32;
        let prefix = usize::cast_from(map[1028 + group]);
        let total = usize::cast_from(map[0]) * 10;
        let low = usize::cast_from(map[1540 + prefix + token + lane]);
        let high = usize::cast_from(map[1540 + prefix + token + 16 + lane]);
        low < total && high < total
    } else {
        fault[0].load() == 0
    };
    if plane_all(raw_good) {
        let c00 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c01 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c10 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c11 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c20 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c21 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c30 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let c31 = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let sg = usize::cast_from(PLANE_POS);
        let token = usize::cast_from(tile_index) * 128 + (sg / 4) * 32;
        let row = usize::cast_from(CUBE_POS_X) * 128 + (sg % 4) * 32;
        super::packed_operands::compute::<f16>(
            input,
            weight,
            &c00,
            &c01,
            &c10,
            &c11,
            &c20,
            &c21,
            &c30,
            &c31,
            token + input_prefix,
            row,
            columns,
            weight_first,
        );
        let out_stride = rows;
        let prefix = usize::cast_from(map[1028 + group]);
        let mut first = (packed_prefix + token) * rows + row;
        if comptime!(down) {
            first = (prefix + token) * rows + row;
        }
        cmma::store(
            &mut output.slice_mut(first + 0 * out_stride + 0, output.len()),
            &c00,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 16 * out_stride + 0, output.len()),
            &c01,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 0 * out_stride + 8, output.len()),
            &c10,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 16 * out_stride + 8, output.len()),
            &c11,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 0 * out_stride + 16, output.len()),
            &c20,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 16 * out_stride + 16, output.len()),
            &c21,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 0 * out_stride + 24, output.len()),
            &c30,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        cmma::store(
            &mut output.slice_mut(first + 16 * out_stride + 24, output.len()),
            &c31,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::ColMajor,
        );
        // All C fragments are dead before these original pending-status writes.
        // The local-route variant validates precisely these32 routes itself.
        // The preceding variant uses the original ordered Gather/guard fault.
        if comptime!(down) && row == 0 {
            let lane = usize::cast_from(UNIT_POS_PLANE);
            let r0 = usize::cast_from(map[1540 + prefix + token + lane]);
            let r1 = usize::cast_from(map[1540 + prefix + token + 16 + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + r0] = 2;
            status[comptime!(crate::strata_prompt::groups::ROUTES) + r1] = 2;
        }
    } else {
        // Original8-member publication granularity, no writing invalid rows.
        // Keep these serial subtiles rolled: only two C fragments live here.
        for tile_m in 0..2usize {
            for tile_n in 0..2usize {
                // SAFETY: this genuine matrix is loaded before its first product.
                let a00 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::A,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                    )
                };
                // SAFETY: this genuine matrix is loaded before its first product.
                let a01 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::A,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                    )
                };
                // SAFETY: this genuine matrix is loaded before its first product.
                let b0 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::B,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::ColMajor,
                    )
                };
                // SAFETY: this genuine matrix is loaded before its first product.
                let a10 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::A,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                    )
                };
                // SAFETY: this genuine matrix is loaded before its first product.
                let a11 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::A,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::RowMajor,
                    )
                };
                // SAFETY: this genuine matrix is loaded before its first product.
                let b1 = unsafe {
                    cmma::Matrix::<f16>::uninitialized(
                        cmma::MatrixIdent::B,
                        8usize,
                        16usize,
                        16usize,
                        cmma::MatrixLayout::ColMajor,
                    )
                };
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
                let sg = usize::cast_from(PLANE_POS);
                let token = if comptime!(cfg!(feature = "strata-wide32-local-routes")) {
                    usize::cast_from(tile_index) * 128 + (sg / 4) * 32 + tile_m * 16
                } else {
                    usize::cast_from(tile_index) * 128 + tile_m * 64 + (sg / 4) * 16
                };
                let row = if comptime!(cfg!(feature = "strata-wide32-local-routes")) {
                    usize::cast_from(CUBE_POS_X) * 128 + (sg % 4) * 32 + tile_n * 16
                } else {
                    usize::cast_from(CUBE_POS_X) * 128 + tile_n * 64 + (sg % 4) * 16
                };
                super::packed_operands::load::<f16>(
                    input,
                    weight,
                    &a00,
                    &a01,
                    &b0,
                    token + input_prefix,
                    row,
                    0usize,
                    columns,
                    weight_first,
                );
                let mut k = 0usize;
                while k < columns {
                    super::packed_operands::load::<f16>(
                        input,
                        weight,
                        &a10,
                        &a11,
                        &b1,
                        token + input_prefix,
                        row,
                        k + 16,
                        columns,
                        weight_first,
                    );
                    crate::strata_matrix::native_wide::global_transpose::products::<f16>(
                        &a00, &a01, &b0, &c0, &c1,
                    );
                    if k + 32 < columns {
                        super::packed_operands::load::<f16>(
                            input,
                            weight,
                            &a00,
                            &a01,
                            &b0,
                            token + input_prefix,
                            row,
                            k + 32,
                            columns,
                            weight_first,
                        );
                    }
                    crate::strata_matrix::native_wide::global_transpose::products::<f16>(
                        &a10, &a11, &b1, &c0, &c1,
                    );
                    k += 32;
                }
                let mut scratch = SharedMemory::<f32>::new(2048usize);
                publish(
                    &c0,
                    &mut scratch,
                    output,
                    map,
                    status,
                    fault,
                    token,
                    row,
                    count,
                    packed_prefix,
                    group,
                    down,
                );
                publish(
                    &c1,
                    &mut scratch,
                    output,
                    map,
                    status,
                    fault,
                    token,
                    row + 8,
                    count,
                    packed_prefix,
                    group,
                    down,
                );
            }
        }
    }
}
