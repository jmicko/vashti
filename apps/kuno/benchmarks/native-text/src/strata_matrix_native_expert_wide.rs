//! Opt-in native32 routed expert prefix plus authentic original-product tail.
//! Original FP16 operands, GU/PART/map/status/fault owners and chronological K16.
use crate::strata_prompt::groups as abi;
use cubecl::{
    ir::{ElemType, FloatKind, features::MmaConfig},
    prelude::*,
    server::Handle,
    wgpu::WgpuRuntime,
};
use half::f16;
use kuno_text::{Error, Result};
const BLOCK: usize = 128;
const THREADS: u32 = 512;
const SUBGROUPS: usize = 32;
const SUBGROUP_ROWS: usize = 16;
const M_FRAGMENTS: usize = 2;
const STAGE_ROUNDS: usize = 1;
const STAGE_THREADS: usize = 512;
pub(super) const SHARED_BYTES: usize = 49152;
// Invalid-route fragments retain the exact original checked publication.
pub(super) const DIRECT_GU_SHARED_BYTES: usize = SHARED_BYTES;

#[cube]
fn header_valid(map: &Array<u32>, count: usize, group: usize) -> bool {
    let total = usize::cast_from(map[0]);
    let mut valid = false;
    if total >= 1
        && total <= comptime!(abi::ROUTES / 10)
        && group < 512
        && group < usize::cast_from(map[3])
    {
        let offset = usize::cast_from(map[1028 + group]);
        valid = map[1] == 0
            && usize::cast_from(map[2]) == total
            && usize::cast_from(map[3]) <= 512
            && usize::cast_from(map[4 + group]) < 512
            && usize::cast_from(map[516 + group]) == count
            && count >= 256
            && count <= total
            && offset <= total * 10
            && count <= total * 10 - offset;
    }
    valid
}

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
            base = route * 2560;
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
fn stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    columns: usize,
    weight_first: usize,
    token_base: usize,
    row_base: usize,
    k: usize,
    buffer: usize,
) {
    // Full128x128x32 stages only. Two genuine vec4s per job preserve original
    // FP16/BF16 bits; every one of the1024 vectors per operand is written once.
    #[unroll]
    for round in 0..comptime!(STAGE_ROUNDS) {
        let job = usize::cast_from(UNIT_POS) + round * comptime!(STAGE_THREADS);
        let offset = k + (job % 4) * 8;
        let row = job / 4;
        let source_a = ((token_base + row) * columns + offset) / 4;
        let source_b = (weight_first + (row_base + row) * columns + offset) / 4;
        a[buffer * 1024 + job * 2] = input[source_a];
        a[buffer * 1024 + job * 2 + 1] = input[source_a + 1];
        b[buffer * 1024 + job * 2] = weight[source_b];
        b[buffer * 1024 + job * 2 + 1] = weight[source_b + 1];
    }
}

#[cube]
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
fn publish_routed(
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

// Literal role is authenticated by the actual generated entry identity.
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

// Literal role is authenticated by the actual generated entry identity.
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

// Literal role is authenticated by the actual generated entry identity.
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

// Literal role is authenticated by the actual generated entry identity.
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

// Literal role is authenticated by the actual generated entry identity.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    prefix: usize,
    tail: usize,
    tail_tile: usize,
}
fn plan(count: usize) -> Option<Plan> {
    if !(256..=abi::ROUTE_CAPACITY).contains(&count) {
        return None;
    }
    let prefix = count / BLOCK * BLOCK;
    let tail = count - prefix;
    let tail_tile = if tail == 0 {
        0
    } else if tail <= 8 {
        8
    } else if tail < 32 {
        16
    } else {
        32
    };
    Some(Plan {
        prefix,
        tail,
        tail_tile,
    })
}
#[allow(clippy::too_many_arguments)]
fn selected(
    count: usize,
    columns: usize,
    rows: usize,
    first: usize,
    down: bool,
    shared: usize,
    units: u32,
    dim: (u32, u32, u32),
    pmin: u32,
    pmax: u32,
) -> bool {
    plan(count).is_some()
        && columns == if down { 640 } else { 2560 }
        && rows == if down { 2560 } else { 1280 }
        && first.is_multiple_of(4)
        && shared >= SHARED_BYTES
        && units >= THREADS
        && dim.0 >= THREADS
        && dim.1 >= 1
        && dim.2 >= 1
        && pmin <= 16
        && pmax >= 16
}
fn vector_views(h: &[Handle; 6]) -> bool {
    h[..2].iter().all(|v| {
        v.size_in_used() != 0
            && v.size_in_used().is_multiple_of(8)
            && v.offset_start.unwrap_or(0).is_multiple_of(8)
    })
}
fn bytes(elements: usize, size: usize) -> Result<u64> {
    u64::try_from(
        elements
            .checked_mul(size)
            .ok_or(Error::ArithmeticOverflow)?,
    )
    .map_err(|_| Error::ArithmeticOverflow)
}
fn tail_view(input: &Handle, count: usize, columns: usize, p: Plan) -> Result<Handle> {
    if p.tail == 0 || Some(p) != plan(count) {
        return Err(Error::InvalidBufferLength);
    }
    let start = bytes(
        p.prefix
            .checked_mul(columns)
            .ok_or(Error::ArithmeticOverflow)?,
        2,
    )?;
    let original = bytes(
        count
            .checked_mul(columns)
            .ok_or(Error::ArithmeticOverflow)?,
        2,
    )?;
    let spare = input
        .size_in_used()
        .checked_sub(original)
        .ok_or(Error::InvalidBufferLength)?;
    let view = input.clone().offset_start(start).offset_end(spare);
    if view.size_in_used()
        != bytes(
            p.tail
                .checked_mul(columns)
                .ok_or(Error::ArithmeticOverflow)?,
            2,
        )?
        || !view.offset_start.unwrap_or(0).is_multiple_of(8)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(view)
}
#[allow(clippy::too_many_arguments)]
fn check_bounds(
    h: &[Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    first: usize,
    group: usize,
    down: bool,
) -> Result<Plan> {
    let p = plan(count).ok_or(Error::InvalidBufferLength)?;
    let input = count
        .checked_mul(columns)
        .ok_or(Error::ArithmeticOverflow)?;
    let weight = rows
        .checked_mul(columns)
        .and_then(|n| first.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    if columns != if down { 640 } else { 2560 }
        || rows != if down { 2560 } else { 1280 }
        || group >= 512
        || [
            count, columns, rows, first, group, p.prefix, p.tail, input, weight,
        ]
        .iter()
        .any(|&v| u32::try_from(v).is_err())
        || h.iter().any(|v| v.stream != h[0].stream)
        || !vector_views(h)
        || bytes(input, 2)? > h[0].size_in_used()
        || bytes(weight, 2)? > h[1].size_in_used()
        || h[2].size_in_used() != if down { abi::PART_BYTES } else { abi::GU_BYTES }
        || h[3].size_in_used() != bytes(abi::MAP_WORDS, 4)?
        || h[4].size_in_used() != bytes(abi::STATUS_WORDS, 4)?
        || h[5].size_in_used() != 4
        || h[2..].iter().any(|v| {
            !v.size_in_used().is_multiple_of(4) || !v.offset_start.unwrap_or(0).is_multiple_of(4)
        })
    {
        return Err(Error::InvalidBufferLength);
    }
    if p.tail != 0 {
        tail_view(&h[0], count, columns, p)?;
    }
    Ok(p)
}

/// False means no command was emitted. Once selected, both pieces and every
/// owner/grid/view are preflighted BEFORE the first genuine wide launch.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_expert(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> Result<bool> {
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
#[allow(clippy::too_many_arguments)]
pub(super) fn try_expert_mode(
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
    // SAFETY: authentic original six owners, fixed admitted shape/CMMA, complete
    // full-tile prefix, real native32 geometry, preflighted original tail below.
    unsafe {
        if down {
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
pub(super) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = vec![];
    for down in [false, true] {
        for tile in [0, 8, 16, 32] {
            let name = match (down, tile) {
                (false, 0) => "expert_wide32_gu",
                (true, 0) => "expert_wide32_down",
                (false, 8) => "expert_tail8_gu",
                (true, 8) => "expert_tail8_down",
                (false, 16) => "expert_tail16_gu",
                (true, 16) => "expert_tail16_down",
                (false, 32) => "expert_tail32_gu",
                (true, 32) => "expert_tail32_down",
                _ => unreachable!(),
            };
            out.push((name, definition(down, tile)));
        }
    }
    if cfg!(feature = "strata-native-direct-gu") {
        out.push(("expert_direct_gu", definition_impl(false, 0, true)));
    }
    out
}

#[cfg(test)]
pub(super) fn entry_names() -> [&'static str; 4] {
    [
        core::any::type_name::<expert_wide_gu::ExpertWideGu<WgpuRuntime>>(),
        core::any::type_name::<expert_wide_down::ExpertWideDown<WgpuRuntime>>(),
        core::any::type_name::<expert_tail_gu::ExpertTailGu<WgpuRuntime>>(),
        core::any::type_name::<expert_tail_down::ExpertTailDown<WgpuRuntime>>(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_direct_gu_has_global_stores_and_original_invalid_route_fallback_fences() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        for mask in 0..8u8 {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(k) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition_impl(false, 0, true),
                &options,
                cubecl::server::ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap() else {
                panic!("actual direct GU SPIRV")
            };
            assert_eq!(k.shared_size, DIRECT_GU_SHARED_BYTES, "layout{mask}");
            assert!(matches!(
                k.bindings[2],
                cubecl_runtime::kernel::Visibility::ReadWrite
            ));
            let w = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module).unwrap();
            super::super::native_wide::tests::assert_publication_scope_contract(&w);
            // Follow actual access chains back to the real descriptor variable.
            // An SLM cooperative store with the same opcode is insufficient.
            let mut roots = std::collections::HashMap::new();
            let mut storage = std::collections::HashMap::new();
            let mut bindings = std::collections::HashMap::new();
            let mut global_stores = 0;
            let mut scratch_stores = 0;
            let mut pos = 5;
            while pos < w.len() {
                let n = (w[pos] >> 16) as usize;
                let a = &w[pos + 1..pos + n];
                match w[pos] & 65535 {
                    59 => {
                        roots.insert(a[1], a[1]);
                        storage.insert(a[1], a[2]);
                    }
                    71 if a.len() == 3 && a[1] == 33 => {
                        bindings.insert(a[0], a[2]);
                    }
                    65 | 66 | 67 | 70 => {
                        let root = *roots.get(&a[2]).expect("actual pointer root");
                        roots.insert(a[1], root);
                    }
                    83 if roots.contains_key(&a[2]) => {
                        roots.insert(a[1], roots[&a[2]]);
                    }
                    4458 => {
                        let root = roots[&a[0]];
                        if storage[&root] == 12 {
                            global_stores += 1;
                            assert_eq!(bindings[&root], 2, "GU output descriptor");
                        } else {
                            scratch_stores += 1;
                            assert_eq!(storage[&root], 4, "original Workgroup fallback");
                        }
                    }
                    _ => {}
                }
                pos += n;
            }
            assert_eq!(global_stores, 4);
            assert_eq!(scratch_stores, 4);
            let mut stores = 0;
            let mut barriers = 0;
            let mut products = 0;
            let mut fault = false;
            let mut at = 5;
            while at < w.len() {
                let n = (w[at] >> 16) as usize;
                match w[at] & 65535 {
                    4458 => stores += 1,
                    4459 => products += 1,
                    224 => barriers += 1,
                    241 => fault = true,
                    _ => {}
                };
                at += n;
            }
            assert_eq!(stores, 8);
            assert_eq!(products, 8);
            assert_eq!(barriers, 10);
            assert!(fault);
        }
    }
    #[test]
    fn selection_and_complete_prefix_tail_plan_preserve_all_original_members() {
        for count in 1..=abi::ROUTE_CAPACITY {
            match plan(count) {
                None => assert!(count < 256),
                Some(p) => {
                    assert!(p.prefix >= 256 && p.prefix.is_multiple_of(128));
                    assert_eq!(p.prefix + p.tail, count);
                    assert!(p.tail < 128);
                    assert_eq!(
                        p.tail_tile,
                        if p.tail == 0 {
                            0
                        } else if p.tail <= 8 {
                            8
                        } else if p.tail < 32 {
                            16
                        } else {
                            32
                        }
                    );
                }
            }
        }
        for down in [false, true] {
            let (columns, rows) = if down { (640, 2560) } else { (2560, 1280) };
            let yes =
                |count, k, n, w, s, u, d, lo, hi| selected(count, k, n, w, down, s, u, d, lo, hi);
            assert!(yes(
                257,
                columns,
                rows,
                128,
                SHARED_BYTES,
                512,
                (512, 1, 1),
                16,
                32
            ));
            for (count, k, n, w, s, u, d, lo, hi) in [
                (
                    255,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    4097,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns - 1,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows - 1,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    127,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES - 1,
                    512,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    511,
                    (512, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (511, 1, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 0, 1),
                    16,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    32,
                    32,
                ),
                (
                    257,
                    columns,
                    rows,
                    128,
                    SHARED_BYTES,
                    512,
                    (512, 1, 1),
                    8,
                    8,
                ),
            ] {
                assert!(!yes(count, k, n, w, s, u, d, lo, hi));
            }
        }
    }
    #[test]
    fn authentic_tail_subview_retains_offsets_identity_stream_and_full_six_owners() {
        let owner = |n| Handle::new(cubecl_common::stream_id::StreamId::current(), n);
        for down in [false, true] {
            for count in [256, 257, 264, 265, 287, 288, 383, 384, 4095, 4096] {
                let (columns, rows) = if down { (640, 2560) } else { (2560, 1280) };
                let h = [
                    owner(count as u64 * columns as u64 * 2 + 32)
                        .offset_start(8)
                        .offset_end(8),
                    owner((128 + rows * columns) as u64 * 2),
                    owner(if down { abi::PART_BYTES } else { abi::GU_BYTES }),
                    owner(abi::MAP_WORDS as u64 * 4),
                    owner(abi::STATUS_WORDS as u64 * 4),
                    owner(4),
                ];
                let p = check_bounds(&h, count, columns, rows, 128, 10, down).unwrap();
                if p.tail != 0 {
                    let v = tail_view(&h[0], count, columns, p).unwrap();
                    assert_eq!(
                        v.clone().binding().memory.allocation_id(),
                        h[0].clone().binding().memory.allocation_id()
                    );
                    assert_eq!(v.size(), h[0].size());
                    assert_eq!(v.stream, h[0].stream);
                    assert_eq!(v.offset_start, Some(8 + (p.prefix * columns * 2) as u64));
                    assert_eq!(v.offset_end, Some(24));
                    assert_eq!(v.size_in_used(), (p.tail * columns * 2) as u64);
                    assert!(
                        tail_view(
                            &h[0],
                            count,
                            columns,
                            Plan {
                                prefix: p.prefix - 128,
                                ..p
                            }
                        )
                        .is_err()
                    );
                }
                for index in 0..6 {
                    let mut bad = h.clone();
                    bad[index] = owner(h[index].size_in_used() - 4);
                    assert!(
                        check_bounds(&bad, count, columns, rows, 128, 10, down).is_err(),
                        "owner{index}"
                    );
                }
                let mut bad = h.clone();
                bad[1] = Handle::new(
                    cubecl_common::stream_id::StreamId {
                        value: h[0].stream.value ^ 1,
                    },
                    h[1].size(),
                );
                assert!(check_bounds(&bad, count, columns, rows, 128, 10, down).is_err());
            }
        }
    }
    #[test]
    fn genuine_native32_publication_and_original_tails_write_each_member_feature_once() {
        for count in [256, 257, 264, 265, 287, 288, 383, 384] {
            for rows in [1280, 2560] {
                let p = plan(count).unwrap();
                let mut writes = vec![0u8; count * rows];
                let mut feature0 = vec![0u8; count];
                let mut put = |member: usize, row: usize| {
                    assert!(member < count && row < rows);
                    writes[member * rows + row] += 1;
                    if row == 0 {
                        feature0[member] += 1;
                    }
                };
                for gy in 0..p.prefix / 128 {
                    for gx in 0..rows / 128 {
                        for sg in 0..32 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    for lane in 0..16 {
                                        for round in 0..8 {
                                            let i = lane + round * 16;
                                            put(
                                                gy * 128 + sg / 4 * 16 + m * 8 + i / 16,
                                                gx * 128 + sg % 4 * 32 + n * 16 + i % 16,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if p.tail != 0 {
                    for gy in 0..p.tail.div_ceil(p.tail_tile) {
                        for gx in 0..rows / 64 {
                            for sg in 0..4 {
                                for lane in 0..16 {
                                    let mut i = lane;
                                    let cells = if p.tail_tile == 32 {
                                        512
                                    } else {
                                        p.tail_tile * 16
                                    };
                                    while i < cells {
                                        let member = if p.tail_tile == 32 {
                                            gy * 32 + sg / 2 * 16 + i / 256 * 8 + i % 128 / 16
                                        } else {
                                            gy * p.tail_tile + i / 16
                                        };
                                        let row = if p.tail_tile == 32 {
                                            gx * 64 + sg % 2 * 32 + i / 128 % 2 * 16 + i % 16
                                        } else {
                                            gx * 64 + sg * 16 + i % 16
                                        };
                                        if member < p.tail {
                                            put(p.prefix + member, row);
                                        }
                                        i += 16;
                                    }
                                }
                            }
                        }
                    }
                }
                assert!(writes.iter().all(|&n| n == 1));
                assert!(feature0.iter().all(|&n| n == 1));
            }
        }
        for k in [640, 2560] {
            let order = (0..k)
                .step_by(32)
                .flat_map(|base| [base, base + 16])
                .collect::<Vec<_>>();
            assert_eq!(order, (0..k).step_by(16).collect::<Vec<_>>());
        }
    }
    #[test]
    fn role_bound_generated_entries_have_seven_u32_scalars_and_original_packed_lengths() {
        use cubecl::ir::AddressType;
        let names = entry_names();
        for (name, suffix) in names.iter().zip([
            "::expert_wide_gu::ExpertWideGu<",
            "::expert_wide_down::ExpertWideDown<",
            "::expert_tail_gu::ExpertTailGu<",
            "::expert_tail_down::ExpertTailDown<",
        ]) {
            assert!(name.contains(suffix), "{name}");
        }
        for (_, d) in definitions() {
            assert_eq!(d.buffers.len(), 6);
            assert_eq!(d.scalars.len(), 1);
            assert_eq!(d.scalars[0].count, 7);
            assert_eq!(
                d.scalars[0].ty,
                StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32))
            );
        }
        let scalars = [2560u32, 1280, 257, 128, 10, 256, 1];
        let buffer_lens = [640u64, 820000, 5242880, 65536, 131072, 1];
        // Production vector() passes ArrayArg.length=used_bytes/8 for each
        // vec4<f16> input/weight. KernelLauncher::process_array independently
        // divides that logical length by vector_size=4; its physical buffer
        // length is used_bytes/Type.size=used_bytes/8. The four scalar arrays
        // retain equal physical/logical lengths. This checks exact InfoBuilder
        // packing, supplemented by the actual KernelLauncher registration seam
        // in strata_expert_bulk_budget_tests (not a fabricated GPU dispatch).
        let logical_lens = [
            buffer_lens[0] / 4,
            buffer_lens[1] / 4,
            buffer_lens[2],
            buffer_lens[3],
            buffer_lens[4],
            buffer_lens[5],
        ];
        assert_ne!(&buffer_lens[..2], &logical_lens[..2]);
        let mut builder = cubecl::InfoBuilder::default();
        for v in scalars {
            builder.scalars.push(v);
        }
        for (buffer_len, logical_len) in buffer_lens.into_iter().zip(logical_lens) {
            builder
                .metadata
                .register_array(buffer_len, logical_len, AddressType::U32);
        }
        let packed = builder.finish(AddressType::U32);
        let words = packed
            .data
            .iter()
            .flat_map(|&v| [v as u32, (v >> 32) as u32])
            .collect::<Vec<_>>();
        assert_eq!(&words[..7], &scalars);
        assert_eq!(words[7], 0);
        assert_eq!(&words[8..14], &buffer_lens.map(|n| n as u32));
        assert_eq!(&words[14..20], &logical_lens.map(|n| n as u32));
        assert_eq!(packed.dynamic_metadata_offset, 10);
        assert_eq!(words.len(), 20);
    }
    #[test]
    fn actual_wide_and_original_tail_modules_keep_ieee_layout_geometry_and_faults() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        for mask in 0..8 {
            for (name, d) in definitions() {
                let mut options = WgpuCompilationOptions {
                    supports_vulkan: true,
                    ..Default::default()
                };
                options.vulkan.max_spirv_version = (1, 6);
                options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
                options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
                options.vulkan.supports_explicit_smem = mask & 4 != 0;
                let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                    &mut AutoCompiler::SpirV(Default::default()),
                    d,
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap() else {
                    panic!("actual SPIRV required")
                };
                let direct = name == "expert_direct_gu";
                let wide = direct || name.starts_with("expert_wide32_");
                let family = if wide {
                    ""
                } else if name.contains("tail8") {
                    "expert_narrow8"
                } else if name.contains("tail16") {
                    "expert_narrow16"
                } else {
                    "expert_macro"
                };
                let shared = if direct {
                    DIRECT_GU_SHARED_BYTES
                } else if wide {
                    SHARED_BYTES
                } else {
                    super::super::native::compiled_shared_bytes(family, mask & 4 != 0).unwrap()
                };
                assert_eq!(kernel.shared_size, shared, "{name}/{mask}");
                let words =
                    wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)
                        .unwrap();
                if wide {
                    super::super::native_wide::tests::assert_publication_scope_contract(&words);
                } else {
                    assert!(
                        wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)
                            .unwrap()
                            .is_none()
                    );
                }
                let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)
                    .unwrap()
                    .unwrap_or(words);
                let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
                    .unwrap()
                    .unwrap_or(words);
                assert_eq!(
                    wgpu::hal::vulkan::kuno_cooperative_subgroup(&words),
                    Some(16)
                );
                let mut at = 5;
                let mut local = None;
                let mut products = 0;
                let mut stores = 0;
                let mut fault = false;
                while at < words.len() {
                    let n = (words[at] >> 16) as usize;
                    let op = words[at] & 65535;
                    let a = &words[at + 1..at + n];
                    if op == 16 && a.get(1) == Some(&17) {
                        local = Some(a[2..5].to_vec());
                    }
                    products += usize::from(op == 4459);
                    stores += usize::from(op == 4458);
                    fault |= op == 241;
                    assert!(op != 71 || a.get(1) != Some(&40), "relaxed FPFastMathMode");
                    at += n;
                }
                assert_eq!(local, Some(vec![if wide { 512 } else { 64 }, 1, 1]));
                assert!(products > 0 && stores > 0 && fault);
                if wide {
                    assert_eq!(products, 8);
                    assert_eq!(stores, if direct { 8 } else { 4 });
                }
                let mut command = std::process::Command::new("spirv-val");
                command.args(["--target-env", "vulkan1.3"]);
                if mask & 1 != 0 {
                    command.arg("--uniform-buffer-standard-layout");
                }
                let mut child = command
                    .arg("-")
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .unwrap();
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(
                        &words
                            .iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                assert!(child.wait().unwrap().success(), "{name}/{mask}");
                println!(
                    "native_routed_expert_cpu name={name} mask={mask} shared_bytes={shared} products={products} stores={stores}"
                );
            }
        }
    }
}
