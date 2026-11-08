//! Complete original-owner global expert products and image-cohort selection.
use super::*;
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
    let token_base = usize::cast_from(CUBE_POS_Y) * 128;
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    let token = token_base + (sg / 4) * 16;
    let row = row_base + (sg % 4) * 16;
    crate::strata_matrix::native_wide::global_n64::load::<f16>(
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
        crate::strata_matrix::native_wide::global_n64::load::<f16>(
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
        crate::strata_matrix::native_wide::global_n64::products::<f16>(&a00, &a01, &b0, &c0, &c1);
        if k + 32 < columns {
            crate::strata_matrix::native_wide::global_n64::load::<f16>(
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
        crate::strata_matrix::native_wide::global_n64::products::<f16>(&a10, &a11, &b1, &c0, &c1);
        k += 32;
    }
    let mut scratch = SharedMemory::<f32>::new(4096usize);
    crate::strata_matrix::small_direct::publish(
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
    crate::strata_matrix::small_direct::publish(
        &c1,
        &mut scratch,
        output,
        map,
        status,
        fault,
        token + 8,
        row,
        count,
        0usize,
        group,
        down,
    );
}
#[cube(launch_unchecked)]
pub fn expert_global_n64_gu(
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
pub fn expert_global_n64_down(
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
        (rows / 64, p.prefix / BLOCK),
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
            expert_global_n64_down::launch_unchecked::<WgpuRuntime>(
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
            expert_global_n64_gu::launch_unchecked::<WgpuRuntime>(
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
                "expert_global_n64_down_prefixes"
            } else {
                "expert_global_n64_gu_prefixes"
            },
            1,
        );
        crate::strata_prefill_profile::count(
            "expert_global_n64_workgroups",
            (rows as u64 / 64) * (u64::from(p.prefix as u32) / 128),
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
                expert_global_n64_down::expand(
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
                expert_global_n64_gu::expand(
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
                    "global_n64_expert_down"
                } else {
                    "global_n64_expert_gu"
                },
                b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(512))),
            )
        })
        .collect()
}

include!("strata_matrix_global_n64_expert_validation.rs");
#[cfg(test)]
#[path = "strata_matrix_global_n64_expert_tests.rs"]
mod tests;
