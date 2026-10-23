//! Additive ordered PART CMMA producers; original rank entries are untouched.
#[cfg(feature="strata-wide-shared-reuse")]
#[path="strata_matrix_ordered_shared_reuse.rs"]
pub(super) mod shared_reuse;
use super::*;
#[cfg(feature = "strata-deep-matrix")]
#[path = "strata_matrix_ordered_deep.rs"]
pub(crate) mod deep;

#[cube]
#[allow(clippy::too_many_arguments)]
fn store_routed(
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    member: usize,
    row: usize,
    value: f32,
    group: usize,
    #[comptime] down: bool,
) {
    let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
    if route < usize::cast_from(map[0]) * 10 {
        let mut base = member * 1280;
        if comptime!(down) {
            base = (usize::cast_from(map[1028 + group]) + member) * 2560;
        }
        let finite = value.abs() <= f32::MAX;
        if finite {
            output[base + row] = value;
        } else {
            output[base + row] = 0.0;
            fault[0].fetch_or(128);
        }
        if comptime!(down) && row == 0 {
            let mut good = 0u32;
            if finite && status[route] == 1 {
                good = 1;
            }
            status[comptime!(abi::ROUTES) + route] = good;
        }
    } else {
        fault[0].fetch_or(128);
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn publication_barrier() {
    if comptime!(cfg!(feature = "strata-native-subgroup-publish")) {
        // Both real Vulkan compiler paths add AcquireRelease|WorkgroupMemory.
        // Only this subgroup owns and consumes its128-F32 publication slot.
        sync_plane();
    } else {
        sync_cube();
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn publish_checked(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    token_base: usize,
    row_base: usize,
    group: usize,
    #[comptime] down: bool,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
) {
    let subgroup = usize::cast_from(PLANE_POS);
    cmma::store(
        &mut scratch.slice_mut(subgroup * 128, subgroup * 128 + 128),
        acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    publication_barrier();
    #[unroll]
    for round in 0..8usize {
        let i = usize::cast_from(UNIT_POS_PLANE) + round * 16;
        let member = token_base + (subgroup / 4) * 16 + comptime!(tile_m * 8) + i / 16;
        let row = row_base + (subgroup % 4) * 32 + comptime!(tile_n * 16) + i % 16;
        store_routed(
            output,
            map,
            status,
            fault,
            member,
            row,
            scratch[subgroup * 128 + i],
            group,
            down,
        );
    }
    publication_barrier();
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn publish_direct_gu(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    token_base: usize,
    row_base: usize,
    group: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let member0 = token_base + (sg / 4) * 16 + comptime!(tile_m * 8);
    // All lanes take the same branch, including the cooperative store/fences.
    // A bad route must leave its old GU bytes untouched, just as standalone GU.
    let mut route_good = true;
    if lane < 8 {
        let route =
            usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member0 + lane]);
        route_good = route < usize::cast_from(map[0]) * 10;
    }
    if plane_all(route_good) {
        let first = member0 * 1280 + row_base + (sg % 4) * 32 + comptime!(tile_n * 16);
        cmma::store(
            &mut output.slice_mut(first, output.len()),
            acc,
            1280,
            cmma::MatrixLayout::RowMajor,
        );
    } else {
        publish_routed(
            acc, scratch, output, map, status, fault, token_base, row_base, group, false, tile_m,
            tile_n,
        );
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn wide(
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
    member_base: u32,
    piece_members: u32,
    #[comptime] down: bool,
    #[comptime] direct_gu: bool,
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let weight_first = usize::cast_from(weight_first);
    let group = usize::cast_from(group);
    let member_base = usize::cast_from(member_base);
    let piece_members = usize::cast_from(piece_members);
    if !header_valid(map, count, group)
        || member_base != 0
        || piece_members != count / 128 * 128
        || columns != comptime!(if down { 640 } else { 2560 })
        || rows != comptime!(if down { 2560 } else { 1280 })
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Two typed stages per operand:2*128*32*2 bytes each. Scratch remains a
    // separate real8/16KiB allocation; admission never assumes lifetime aliasing.
    let mut a = SharedMemory::<Vector<f16, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<f16, Const<4>>>::new(2048usize);
    let mut scratch = SharedMemory::<f32>::new(comptime!(SUBGROUPS * 128));
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a1 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a2 = if comptime!(M_FRAGMENTS == 4) {
        unsafe {
            cmma::Matrix::<f16>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    } else {
        a0
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a3 = if comptime!(M_FRAGMENTS == 4) {
        unsafe {
            cmma::Matrix::<f16>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    } else {
        a1
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
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
    let c4 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c0
    };
    let c5 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c1
    };
    let c6 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c2
    };
    let c7 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c3
    };
    let subgroup = usize::cast_from(PLANE_POS);
    let token_base = usize::cast_from(CUBE_POS_Y) * 128;
    let row_base = usize::cast_from(CUBE_POS_X) * 128;
    stage::<f16>(
        input,
        weight,
        &mut a,
        &mut b,
        columns,
        weight_first,
        token_base,
        row_base,
        0usize,
        0usize,
    );
    sync_cube();
    let mut k = 0usize;
    let mut current = 0usize;
    while k < columns {
        let next = 1usize - current;
        // REGISTER_STAGE_LOAD_BEGIN: real global values stay live across
        // the current original matrix products; next SLM writes occur later.
        let mut next_a_lo = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_a_hi = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_lo = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_hi = Vector::<f16, Const<4>>::new(f16::new(0.0));
        if comptime!(cfg!(feature = "strata-wide-register-stage") && STAGE_ROUNDS == 1) {
            if k + 32 < columns {
                let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let row = job / 4;
                let off = k + 32 + (job % 4) * 8;
                let source_a = ((token_base + row) * columns + off) / 4;
                let source_b = (weight_first + (row_base + row) * columns + off) / 4;
                next_a_lo = input[source_a];
                next_a_hi = input[source_a + 1];
                next_b_lo = weight[source_b];
                next_b_hi = weight[source_b + 1];
            }
        } else {
            // Same accepted immediate global-load→SLM-store software-prefetch ordering.
            // This expresses overlap opportunity, not guaranteed asynchronous execution.
            if k + 32 < columns {
                stage::<f16>(
                    input,
                    weight,
                    &mut a,
                    &mut b,
                    columns,
                    weight_first,
                    token_base,
                    row_base,
                    k + 32,
                    next,
                );
            }
        }
        // REGISTER_STAGE_LOAD_END
        #[unroll]
        for step in 0..2usize {
            let af = current * 1024 + (subgroup / 4) * comptime!(SUBGROUP_ROWS * 8) + step * 4;
            let bf = current * 1024 + (subgroup % 4) * 256 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(af + 64, a.len()), 32);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::load(&a2, &a.to_slice().slice(af + 128, a.len()), 32);
                cmma::load(&a3, &a.to_slice().slice(af + 192, a.len()), 32);
            }
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 32);
            cmma::load(&b1, &b.to_slice().slice(bf + 128, b.len()), 32);
            cmma::execute::<f16, f16, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<f16, f16, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<f16, f16, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<f16, f16, f32, f32>(&a1, &b1, &c3, &c3);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::execute::<f16, f16, f32, f32>(&a2, &b0, &c4, &c4);
                cmma::execute::<f16, f16, f32, f32>(&a2, &b1, &c5, &c5);
                cmma::execute::<f16, f16, f32, f32>(&a3, &b0, &c6, &c6);
                cmma::execute::<f16, f16, f32, f32>(&a3, &b1, &c7, &c7);
            }
        }
        // REGISTER_STAGE_STORE_BEGIN: only after all chronological current
        // CMMA products, publish the prefetched values into the disjoint bank.
        if comptime!(cfg!(feature = "strata-wide-register-stage") && STAGE_ROUNDS == 1) {
            if k + 32 < columns {
                let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let row = job / 4;
                let dest = next * 1024 + job * 2;
                a[dest] = next_a_lo;
                a[dest + 1] = next_a_hi;
                b[dest] = next_b_lo;
                b[dest + 1] = next_b_hi;
            }
        }
        // REGISTER_STAGE_STORE_END
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 32;
    }
    if comptime!(direct_gu) {
        publish_direct_gu(
            &c0,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            0usize,
            0usize,
        );
        publish_direct_gu(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            0usize,
            1usize,
        );
        publish_direct_gu(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            1usize,
            0usize,
        );
        publish_direct_gu(
            &c3,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            1usize,
            1usize,
        );
    } else {
        publish_routed(
            &c0,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            0usize,
            0usize,
        );
        publish_routed(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            0usize,
            1usize,
        );
        publish_routed(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            1usize,
            0usize,
        );
        publish_routed(
            &c3,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            1usize,
            1usize,
        );
        if comptime!(M_FRAGMENTS == 4) {
            publish_routed(
                &c4,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                2usize,
                0usize,
            );
            publish_routed(
                &c5,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                2usize,
                1usize,
            );
            publish_routed(
                &c6,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                3usize,
                0usize,
            );
            publish_routed(
                &c7,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                3usize,
                1usize,
            );
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn tail(
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
    member_base: u32,
    piece_members: u32,
    #[comptime] down: bool,
    #[comptime] tile: usize,
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let weight_first = usize::cast_from(weight_first);
    let group = usize::cast_from(group);
    let member_base = usize::cast_from(member_base);
    let piece_members = usize::cast_from(piece_members);
    if !header_valid(map, count, group)
        || member_base != count / 128 * 128
        || piece_members == 0
        || piece_members != count - member_base
        || columns != comptime!(if down { 640 } else { 2560 })
        || rows != comptime!(if down { 2560 } else { 1280 })
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    let values = if comptime!(tile == 32) {
        crate::strata_matrix::native::macro_product::<f16>(
            input,
            weight,
            columns,
            rows,
            piece_members,
            weight_first,
        )
    } else {
        crate::strata_matrix::native::narrow_product::<f16>(
            input,
            weight,
            columns,
            rows,
            piece_members,
            weight_first,
            tile,
            false,
        )
    };
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < comptime!(if tile == 32 { 512 } else { tile * 16 }) {
        let local_member = if comptime!(tile == 32) {
            usize::cast_from(CUBE_POS_Y) * 32 + (subgroup / 2) * 16 + (i / 256) * 8 + (i % 128) / 16
        } else {
            usize::cast_from(CUBE_POS_Y) * tile + i / 16
        };
        let row = if comptime!(tile == 32) {
            usize::cast_from(CUBE_POS_X) * 64 + (subgroup % 2) * 32 + ((i / 128) % 2) * 16 + i % 16
        } else {
            usize::cast_from(CUBE_POS_X) * 64 + subgroup * 16 + i % 16
        };
        if local_member < piece_members && row < rows {
            store_routed(
                output,
                map,
                status,
                fault,
                member_base + local_member,
                row,
                values[subgroup * comptime!(if tile == 32 { 512 } else { tile * 16 }) + i],
                group,
                down,
            );
        }
        i += 16;
    }
}

#[cube(launch_unchecked)]
pub fn expert_wide_gu(
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
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        false,
        false,
    );
}

#[cube(launch_unchecked)]
pub fn expert_wide_gu_direct(
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
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        false,
        true,
    );
}

#[cube(launch_unchecked)]
pub fn expert_wide_down(
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
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        true,
        false,
    );
}

#[cube(launch_unchecked)]
pub fn expert_tail_gu(
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
    member_base: u32,
    piece_members: u32,
    #[comptime] tile: usize,
) {
    tail(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        false,
        tile,
    );
}

#[cube(launch_unchecked)]
pub fn expert_tail_down(
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
    member_base: u32,
    piece_members: u32,
    #[comptime] tile: usize,
) {
    tail(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        true,
        tile,
    );
}

pub(crate) fn try_expert(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> Result<bool> {
    if !down {
        return Err(Error::InvalidBufferLength);
    }
    let direct = cfg!(feature = "strata-native-direct-gu")
        && !down
        && client
            .native_strata_expert_capture_open()
            .map_err(crate::resident_util::fault)?;
    try_expert_mode(
        client,
        h,
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
        direct,
    )
}

pub(crate) fn try_expert_mode(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
    direct: bool,
) -> Result<bool> {
    if !down {
        return Err(Error::InvalidBufferLength);
    }
    let hw = &client.properties().hardware;
    if !selected(
        count,
        columns,
        rows,
        weight_first,
        down,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
    ) || !vector_views(&h)
    {
        return Ok(false);
    }
    let kind = ElemType::Float(FloatKind::F16);
    if !client
        .properties()
        .features
        .matmul
        .cmma
        .contains(&MmaConfig {
            a_type: kind.into(),
            b_type: kind.into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 8,
            n: 16,
            k: 16,
        })
    {
        return Ok(false);
    }
    let p = check_bounds(&h, count, columns, rows, weight_first, group, down)?;
    let grids = [
        (rows / BLOCK, p.prefix / BLOCK),
        (
            rows / 64,
            if p.tail == 0 {
                0
            } else {
                p.tail.div_ceil(p.tail_tile)
            },
        ),
    ];
    if hw.max_cube_count.2 < 1
        || grids
            .iter()
            .any(|&(x, y)| x > hw.max_cube_count.0 as usize || y > hw.max_cube_count.1 as usize)
    {
        return Err(Error::LimitExceeded("native wide expert grid"));
    }
    let tail_input = if p.tail == 0 {
        None
    } else {
        Some(tail_view(&h[0], count, columns, p)?)
    };
    // Every fallible argument/view conversion happens before either dispatch.
    let vector =
        |v: Handle| unsafe { ArrayArg::from_raw_parts(v.clone(), v.size_in_used() as usize / 8) };
    let raw = |v: Handle| crate::resident_util::raw_arg(v.clone(), v.size_in_used() as usize / 4);
    let prefix_args = (
        vector(h[0].clone()),
        vector(h[1].clone()),
        raw(h[2].clone())?,
        raw(h[3].clone())?,
        raw(h[4].clone())?,
        raw(h[5].clone())?,
    );
    let tail_args = tail_input
        .map(|input| {
            Ok::<_, Error>((
                vector(input),
                vector(h[1].clone()),
                raw(h[2].clone())?,
                raw(h[3].clone())?,
                raw(h[4].clone())?,
                raw(h[5].clone())?,
            ))
        })
        .transpose()?;
    let (columns, rows, count, weight_first, group) = (
        columns as u32,
        rows as u32,
        count as u32,
        weight_first as u32,
        group as u32,
    );
    let (ia, wa, oa, ma, sa, fa) = prefix_args;
    #[cfg(feature = "strata-deep-matrix")]
    let deep_selected = deep::selected(client);
    #[cfg(not(feature = "strata-deep-matrix"))]
    let deep_selected = false;
    // SAFETY: authentic original six owners, fixed admitted shape/CMMA, complete
    // full-tile prefix, real native32 geometry, preflighted original tail below.
    unsafe {
        if deep_selected {
            #[cfg(feature = "strata-deep-matrix")]
            {
                crate::strata_prefill_profile::count("expert_wide_deep_products", 1);
                deep::launch_prefix(
                    client,
                    [ia, wa, oa, ma, sa, fa],
                    grids[0],
                    [
                        columns,
                        rows,
                        count,
                        weight_first,
                        group,
                        0,
                        p.prefix as u32,
                    ],
                    down,
                    direct,
                );
            }
        } else if down {
            expert_wide_down::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grids[0].0 as u32, grids[0].1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                0,
                p.prefix as u32,
            );
        } else if direct {
            expert_wide_gu_direct::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grids[0].0 as u32, grids[0].1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                0,
                p.prefix as u32,
            );
        } else {
            expert_wide_gu::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grids[0].0 as u32, grids[0].1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                0,
                p.prefix as u32,
            );
        }
        if let Some((ia, wa, oa, ma, sa, fa)) = tail_args {
            if down {
                expert_tail_down::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(grids[1].0 as u32, grids[1].1 as u32, 1),
                    CubeDim::new_1d(64),
                    ia,
                    wa,
                    oa,
                    ma,
                    sa,
                    fa,
                    columns,
                    rows,
                    count,
                    weight_first,
                    group,
                    p.prefix as u32,
                    p.tail as u32,
                    p.tail_tile,
                );
            } else {
                expert_tail_gu::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(grids[1].0 as u32, grids[1].1 as u32, 1),
                    CubeDim::new_1d(64),
                    ia,
                    wa,
                    oa,
                    ma,
                    sa,
                    fa,
                    columns,
                    rows,
                    count,
                    weight_first,
                    group,
                    p.prefix as u32,
                    p.tail as u32,
                    p.tail_tile,
                );
            }
        }
    }
    Ok(true)
}

fn definition(down: bool, tile: usize) -> KernelDefinition {
    definition_impl(down, tile, false)
}

fn definition_impl(down: bool, tile: usize, direct_gu: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: [cubecl::ir::ManagedVariable; 7] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    if tile == 0 {
        if direct_gu {
            expert_wide_gu_direct::expand(
                &mut b.scope,
                input.into(),
                weight.into(),
                output.into(),
                map.into(),
                status.into(),
                fault.into(),
                a[0].clone().into(),
                a[1].clone().into(),
                a[2].clone().into(),
                a[3].clone().into(),
                a[4].clone().into(),
                a[5].clone().into(),
                a[6].clone().into(),
            );
        } else if down {
            expert_wide_down::expand(
                &mut b.scope,
                input.into(),
                weight.into(),
                output.into(),
                map.into(),
                status.into(),
                fault.into(),
                a[0].clone().into(),
                a[1].clone().into(),
                a[2].clone().into(),
                a[3].clone().into(),
                a[4].clone().into(),
                a[5].clone().into(),
                a[6].clone().into(),
            );
        } else {
            expert_wide_gu::expand(
                &mut b.scope,
                input.into(),
                weight.into(),
                output.into(),
                map.into(),
                status.into(),
                fault.into(),
                a[0].clone().into(),
                a[1].clone().into(),
                a[2].clone().into(),
                a[3].clone().into(),
                a[4].clone().into(),
                a[5].clone().into(),
                a[6].clone().into(),
            );
        }
    } else {
        if down {
            expert_tail_down::expand(
                &mut b.scope,
                input.into(),
                weight.into(),
                output.into(),
                map.into(),
                status.into(),
                fault.into(),
                a[0].clone().into(),
                a[1].clone().into(),
                a[2].clone().into(),
                a[3].clone().into(),
                a[4].clone().into(),
                a[5].clone().into(),
                a[6].clone().into(),
                tile,
            );
        } else {
            expert_tail_gu::expand(
                &mut b.scope,
                input.into(),
                weight.into(),
                output.into(),
                map.into(),
                status.into(),
                fault.into(),
                a[0].clone().into(),
                a[1].clone().into(),
                a[2].clone().into(),
                a[3].clone().into(),
                a[4].clone().into(),
                a[5].clone().into(),
                a[6].clone().into(),
                tile,
            );
        }
    }
    let errors = b.scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");
    b.build(
        KernelSettings::default().cube_dim(CubeDim::new_1d(if tile == 0 { THREADS } else { 64 })),
    )
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn publish_routed(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    token_base: usize,
    row_base: usize,
    group: usize,
    #[comptime] down: bool,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
) {
    if comptime!(down) {
        let subgroup = usize::cast_from(PLANE_POS);
        let lane = usize::cast_from(UNIT_POS_PLANE);
        let member0 = token_base + (subgroup / 4) * 16 + comptime!(tile_m * 8);
        let prefix = usize::cast_from(map[1028 + group]);
        let mut route_good = true;
        if lane < 8 {
            let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
            route_good = route < usize::cast_from(map[0]) * 10;
        }
        // Genuine collective control: every lane participates, and only this
        // subgroup owns its fallback scratch. Feature requires subgroup fences.
        if plane_all(route_good) {
            let column = row_base + (subgroup % 4) * 32 + comptime!(tile_n * 16);
            let first = (prefix + member0) * 2560 + column;
            cmma::store(
                &mut output.slice_mut(first, output.len()),
                acc,
                2560,
                cmma::MatrixLayout::RowMajor,
            );
            if column == 0 && lane < 8 {
                let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
                // Raw CMMA result is PENDING. The matching consumer checks
                // activation, column0 and every component before publication.
                status[comptime!(abi::ROUTES) + route] = 2;
            }
        } else {
            publish_checked(
                acc, scratch, output, map, status, fault, token_base, row_base, group, true,
                tile_m, tile_n,
            );
        }
    } else {
        publish_checked(
            acc, scratch, output, map, status, fault, token_base, row_base, group, false, tile_m,
            tile_n,
        );
    }
}
pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    [
        ("ordered_expert_wide32_down", 0),
        ("ordered_expert_tail8_down", 8),
        ("ordered_expert_tail16_down", 16),
        ("ordered_expert_tail32_down", 32),
    ]
    .into_iter()
    .map(|(name, tile)| (name, definition(true, tile)))
    .collect()
}
