//! Genuine K32 operand/publication lifetime reuse; original48KiB entries stay.
use super::*;
pub(crate) const OPERAND_BYTES: usize = 32768;
pub(crate) const LOGICAL_BYTES: usize = 49152;
#[allow(clippy::too_many_arguments)]
pub(crate) fn geometry(
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
    explicit: bool,
) -> bool {
    explicit
        && shared >= OPERAND_BYTES
        && plan(count).is_some()
        && columns == if down { 640 } else { 2560 }
        && rows == if down { 2560 } else { 1280 }
        && first.is_multiple_of(4)
        && units >= THREADS
        && dim.0 >= THREADS
        && dim.1 >= 1
        && dim.2 >= 1
        && pmin <= 16
        && pmax >= 16
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn publish_layout(
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
    #[comptime] ordered_output: bool,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
) {
    if comptime!(ordered_output) {
        super::ordered::publish_routed(
            acc, scratch, output, map, status, fault, token_base, row_base, group, true, tile_m,
            tile_n,
        );
    } else {
        super::publish_routed(
            acc, scratch, output, map, status, fault, token_base, row_base, group, down, tile_m,
            tile_n,
        );
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
    member_base: u32,
    piece_members: u32,
    #[comptime] down: bool,
    #[comptime] direct_gu: bool,
    #[comptime] ordered_output: bool,
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let weight_first = usize::cast_from(weight_first);
    let group = usize::cast_from(group);
    let member_base = usize::cast_from(member_base);
    let piece_members = usize::cast_from(piece_members);
    if !plane_all(header_valid(map, count, group))
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
    // Two original K32 operand stages occupy16KiB each. Their lifetimes end
    // after the final whole-workgroup handoff, before publication scratch.
    let mut a = SharedMemory::<Vector<f16, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<f16, Const<4>>>::new(2048usize);
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
    let mut current = 0usize;
    // Runtime scalar bounds are uniform. This retains the original GU80 /
    // DOWN20 K32 stages and two chronological K16 steps in each stage.
    for stage_index in 0usize..columns / 32usize {
        let k = stage_index * 32usize;
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
    }
    // SAFETY: the final original full-WG handoff ends every cooperative
    // operand read; no pointer to either array is used below. Only the actual
    // explicit-layout compiler may alias the subsequent scratch with A/B.
    unsafe {
        a.free();
        b.free();
    }
    let mut scratch = SharedMemory::<f32>::new(comptime!(SUBGROUPS * 128));
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
        publish_layout(
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
            ordered_output,
            0usize,
            0usize,
        );
        publish_layout(
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
            ordered_output,
            0usize,
            1usize,
        );
        publish_layout(
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
            ordered_output,
            1usize,
            0usize,
        );
        publish_layout(
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
            ordered_output,
            1usize,
            1usize,
        );
        if comptime!(M_FRAGMENTS == 4) {
            publish_layout(
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
                ordered_output,
                2usize,
                0usize,
            );
            publish_layout(
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
                ordered_output,
                2usize,
                1usize,
            );
            publish_layout(
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
                ordered_output,
                3usize,
                0usize,
            );
            publish_layout(
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
                ordered_output,
                3usize,
                1usize,
            );
        }
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
        false,
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
        false,
    );
}

#[allow(clippy::too_many_arguments)]
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
    ordered_output: bool,
) -> Result<bool> {
    try_expert_mode_impl(client,h,count,columns,rows,weight_first,group,down,direct,ordered_output,false)
}

#[cfg(feature="strata-small-direct-publish")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_expert_mode_small_tail(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
    direct: bool,
    ordered_output: bool,
) -> Result<bool> {
    if down && !ordered_output {return Err(Error::InvalidBufferLength);}
    try_expert_mode_impl(client,h,count,columns,rows,weight_first,group,down,direct,ordered_output,true)
}

#[allow(clippy::too_many_arguments)]
fn try_expert_mode_impl(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
    direct: bool,
    ordered_output: bool,
    small_tail: bool,
) -> Result<bool> {
    if (ordered_output && !down) || (direct && down) {
        return Err(Error::InvalidBufferLength);
    }
    crate::strata_matrix::native::compatible(client, false)?;
    let hw = &client.properties().hardware;
    if !geometry(
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
        client.native_explicit_shared_layout(),
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
        if ordered_output {
            super::ordered::shared_reuse::expert_wide_down::launch_unchecked::<WgpuRuntime>(
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
        // A genuine selected prefix has now been queued. Ordinary private
        // workers keep no counters; only the existing public collector sees it.
        crate::strata_prefill_profile::count(
            if down {
                "expert_shared_reuse_down_prefixes"
            } else {
                "expert_shared_reuse_gu_prefixes"
            },
            1,
        );
        if let Some((ia, wa, oa, ma, sa, fa)) = tail_args {
            if small_tail {
                #[cfg(feature="strata-small-direct-publish")]
                crate::strata_matrix::small_direct::launch_tail(client,[ia,wa,oa,ma,sa,fa],grids[1],
                    [columns,rows,count,weight_first,group,p.prefix as u32,p.tail as u32],down,p.tail_tile);
            } else {
            if ordered_output {
                super::ordered::expert_tail_down::launch_unchecked::<WgpuRuntime>(
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
            } else if down {
                super::expert_tail_down::launch_unchecked::<WgpuRuntime>(
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
                super::expert_tail_gu::launch_unchecked::<WgpuRuntime>(
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
    }
    Ok(true)
}

fn definition(down: bool, direct_gu: bool) -> KernelDefinition {
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
    {
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
    }
    let errors = b.scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    vec![
        ("shared_reuse_gu", definition(false, false)),
        ("shared_reuse_direct_gu", definition(false, true)),
        ("shared_reuse_rank_down", definition(true, false)),
    ]
}
