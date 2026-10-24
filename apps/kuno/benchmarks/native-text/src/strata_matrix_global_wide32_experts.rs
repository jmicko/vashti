//! Complete original-owner global expert products and image-cohort selection.
use super::*;

#[cube(launch_unchecked)]
fn route_guard(map:&Array<u32>,fault:&mut Array<Atomic<u32>>,count:u32,group:u32){
    let count=usize::cast_from(count);let group=usize::cast_from(group);
    if !header_valid(map,count,group){if UNIT_POS==0{fault[0].fetch_or(128);}terminate!();}
    let member=usize::cast_from(ABSOLUTE_POS_X);
    if member<count/128*128 {
        let prefix=usize::cast_from(map[1028+group]);
        if usize::cast_from(map[1540+prefix+member])>=usize::cast_from(map[0])*10{fault[0].fetch_or(128);}
    }
}


// The transposed C covers16 members by8 columns. Route failure semantics
// still belong to each original8-member half, independently of the other half.
#[cube]
#[allow(clippy::too_many_arguments)]
fn publish(
    acc: &cmma::Matrix<f32>, scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>, map: &Array<u32>, status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>, member0: usize, row0: usize,
    count: usize, gu_prefix: usize, group: usize, #[comptime] down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let lane = usize::cast_from(UNIT_POS_PLANE);
    let prefix = usize::cast_from(map[1028 + group]);
    let rows = comptime!(if down {2560usize} else {1280usize});
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
        if comptime!(down) {first = (prefix + member0) * 2560 + row0; stride = 2560u32;}
        cmma::store(&mut output.slice_mut(first, output.len()), acc, stride, cmma::MatrixLayout::ColMajor);
        if comptime!(down) && row0 == 0 {
            let route = usize::cast_from(map[1540 + prefix + member0 + lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
        }
    } else {
        cmma::store(&mut scratch.slice_mut(sg * 128, sg * 128 + 128), acc, 8, cmma::MatrixLayout::ColMajor);
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
                    if comptime!(down) {first = (prefix + member) * 2560;}
                    let raw_half = (i / 8 < 8 && low_good) || (i / 8 >= 8 && high_good);
                    if raw_half {
                        output[first + row] = value;
                        if comptime!(down) && row == 0 {
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = 2;
                        }
                    } else {
                        let finite = value.abs() <= f32::MAX;
                        if finite {output[first + row] = value;} else {output[first + row] = 0.0; fault[0].fetch_or(128);}
                        if comptime!(down) && row == 0 {
                            let mut valid = 0u32;
                            if finite && status[route] == 1 {valid = 1;}
                            status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = valid;
                        }
                    }
                } else {fault[0].fetch_or(128);}
            }
        }
        sync_plane();
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
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let group = usize::cast_from(group);
    let weight_first = usize::cast_from(weight_first);
    if !plane_all(header_valid(map, count, group))
        || member_base != 0
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
        let token = usize::cast_from(CUBE_POS_Y) * 128 + (sg / 4) * 32;
        let prefix = usize::cast_from(map[1028 + group]);
        let total = usize::cast_from(map[0]) * 10;
        let low = usize::cast_from(map[1540 + prefix + token + lane]);
        let high = usize::cast_from(map[1540 + prefix + token + 16 + lane]);
        low < total && high < total
    } else {
        fault[0].load() == 0
    };
    if plane_all(raw_good) {
    let c00=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c01=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c10=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c11=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c20=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c21=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c30=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
    let c31=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
        let sg=usize::cast_from(PLANE_POS);
        let token=usize::cast_from(CUBE_POS_Y)*128+(sg/4)*32;
        let row=usize::cast_from(CUBE_POS_X)*128+(sg%4)*32;
        crate::strata_matrix::native_wide::global_wide32::compute::<f16>(input,weight,&c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,token,row,columns,weight_first);
        let out_stride=rows;
        let prefix=usize::cast_from(map[1028+group]);
        let mut first=token*rows+row;
        if comptime!(down){first=(prefix+token)*rows+row;}
    cmma::store(&mut output.slice_mut(first+0*out_stride+0,output.len()),&c00,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+0,output.len()),&c01,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+8,output.len()),&c10,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+8,output.len()),&c11,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+16,output.len()),&c20,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+16,output.len()),&c21,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+24,output.len()),&c30,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+24,output.len()),&c31,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
        // All C fragments are dead before these original pending-status writes.
        // The local-route variant validates precisely these32 routes itself.
        // The preceding variant uses the original ordered Gather/guard fault.
        if comptime!(down) && row==0 {
            let lane=usize::cast_from(UNIT_POS_PLANE);
            let r0=usize::cast_from(map[1540+prefix+token+lane]);
            let r1=usize::cast_from(map[1540+prefix+token+16+lane]);
            status[comptime!(crate::strata_prompt::groups::ROUTES)+r0]=2;
            status[comptime!(crate::strata_prompt::groups::ROUTES)+r1]=2;
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
        usize::cast_from(CUBE_POS_Y) * 128 + (sg / 4) * 32 + tile_m * 16
    } else {
        usize::cast_from(CUBE_POS_Y) * 128 + tile_m * 64 + (sg / 4) * 16
    };
    let row = if comptime!(cfg!(feature = "strata-wide32-local-routes")) {
        usize::cast_from(CUBE_POS_X) * 128 + (sg % 4) * 32 + tile_n * 16
    } else {
        usize::cast_from(CUBE_POS_X) * 128 + tile_n * 64 + (sg % 4) * 16
    };
    crate::strata_matrix::native_wide::global_transpose::load::<f16>(
        input,
        weight,
        &a00,
        &a01,
        &b0,
        token,
        row,
        0usize,
        columns,
        weight_first,
    );
    let mut k = 0usize;
    while k < columns {
        crate::strata_matrix::native_wide::global_transpose::load::<f16>(
            input,
            weight,
            &a10,
            &a11,
            &b1,
            token,
            row,
            k + 16,
            columns,
            weight_first,
        );
        crate::strata_matrix::native_wide::global_transpose::products::<f16>(&a00, &a01, &b0, &c0, &c1);
        if k + 32 < columns {
            crate::strata_matrix::native_wide::global_transpose::load::<f16>(
                input,
                weight,
                &a00,
                &a01,
                &b0,
                token,
                row,
                k + 32,
                columns,
                weight_first,
            );
        }
        crate::strata_matrix::native_wide::global_transpose::products::<f16>(&a10, &a11, &b1, &c0, &c1);
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
        0usize,
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
        0usize,
        group,
        down,
    );
            }
        }
    }
}

#[cube(launch_unchecked)]
pub fn expert_global_wide32_gu(
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
    );
}
#[cube(launch_unchecked)]
pub fn expert_global_wide32_down(
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
) -> Result<bool> {
    crate::strata_matrix::native::compatible(client, false)?;
    let hw = &client.properties().hardware;
    if !super::shared_reuse::geometry(
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
        (rows / 128, p.prefix / BLOCK),
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
    let captured=client.native_strata_expert_capture_open().map_err(crate::resident_util::fault)?;
    if !captured {
        unsafe {route_guard::launch_unchecked::<WgpuRuntime>(client,CubeCount::Static((p.prefix.div_ceil(256))as u32,1,1),CubeDim::new_1d(256),raw(h[3].clone())?,raw(h[5].clone())?,count,group);}
    }
    crate::strata_prefill_profile::count(if captured{"expert_global_wide32_gather_guard_prefixes"}else{"expert_global_wide32_standalone_guard_prefixes"},1);
    let (ia, wa, oa, ma, sa, fa) = prefix_args;
    // SAFETY: authentic original six owners, fixed admitted shape/CMMA, complete
    // full-tile prefix, real native32 geometry, preflighted original tail below.
    unsafe {
        if down {
            expert_global_wide32_down::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grids[0].0 as u32, grids[0].1 as u32, 1),
                CubeDim::new_1d(256),
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
            expert_global_wide32_gu::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grids[0].0 as u32, grids[0].1 as u32, 1),
                CubeDim::new_1d(256),
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
                "expert_global_wide32_down_prefixes"
            } else {
                "expert_global_wide32_gu_prefixes"
            },
            1,
        );
        crate::strata_prefill_profile::count(
            "expert_global_wide32_workgroups",
            (rows as u64 / 128) * (u64::from(p.prefix as u32) / 128),
        );
        if let Some((ia, wa, oa, ma, sa, fa)) = tail_args {
            crate::strata_matrix::small_direct::launch_tail(
                client,
                [ia, wa, oa, ma, sa, fa],
                grids[1],
                [
                    columns,
                    rows,
                    count,
                    weight_first,
                    group,
                    p.prefix as u32,
                    p.tail as u32,
                ],
                down,
                p.tail_tile,
            );
        }
    }
    Ok(true)
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    [false, true]
        .into_iter()
        .map(|down| {
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
            if down {
                expert_global_wide32_down::expand(
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
                expert_global_wide32_gu::expand(
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
            assert!(b.scope.pop_errors().is_empty());
            (
                if down {
                    "global_wide32_expert_down"
                } else {
                    "global_wide32_expert_gu"
                },
                b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(256))),
            )
        })
        .collect()
}

#[path="strata_matrix_global_wide32_expert_validation.rs"]
mod validation;
pub(crate) use validation::validate_sources;
#[cfg(test)]
#[path = "strata_matrix_global_wide32_expert_tests.rs"]
mod tests;
