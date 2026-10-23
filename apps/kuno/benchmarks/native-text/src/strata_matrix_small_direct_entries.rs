// Genuine small direct entries; original entrypoints remain unchanged.

#[allow(clippy::too_many_arguments)]
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
    compatible(client, false)?;
    let [input, weight, output, map, status, fault] = h;
    if (1..32).contains(&count) && cfg!(feature = "strata-narrow-global") {
        let tile = if count <= 8 { 8 } else { 16 };
        let padded = count.div_ceil(tile) * tile;
        if rows != if down { 2560 } else { 1280 }
            || columns != if down { 640 } else { 2560 }
            || input.size_in_used() < (padded * columns * 2) as u64
            || weight.size_in_used() < ((weight_first + rows * columns) * 2) as u64
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let use_macro = macro_compatible(client, &input, &weight, count, rows, columns, weight_first);
    let macro64_selected = macro64_geometry(
        count,
        rows,
        columns,
        weight_first,
        client.properties().hardware.max_shared_memory_size,
        [&input, &weight].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0))),
    );
    let narrow = narrow_compatible(client, &input, &weight, count, rows, columns, weight_first);
    if !use_macro && narrow.is_none() {
        return Ok(false);
    }
    let ia = macro_half_arg(input)?;
    let wa = macro_half_arg(weight)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
    let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    unsafe {
        if use_macro && macro64_selected {
            crate::strata_prefill_profile::count("expert_small_direct_macro64_products", 1);
            crate::strata_prefill_profile::count("expert_macro64_products", 1);
            expert_macro64::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(64) as u32, 1),
                CubeDim::new_1d(128),
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
                down,
            );
        } else if use_macro {
            crate::strata_prefill_profile::count("expert_small_direct_macro_products", 1);
            expert_macro::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(32) as u32, 1),
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
                down,
            );
        } else if let Some(tile) = narrow {
            crate::strata_prefill_profile::count("expert_small_direct_narrow_products", 1);
            expert_narrow::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(tile) as u32, 1),
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
                down,
                tile,
                cfg!(feature = "strata-narrow-global"),
            );
        }
    }
    Ok(true)
}

pub(crate) unsafe fn launch_tail(
    client: &ComputeClient<WgpuRuntime>,
    args: [ArrayArg<WgpuRuntime>; 6],
    grid: (usize, usize),
    words: [u32; 7],
    down: bool,
    tile: usize,
) {
    let [ia, wa, oa, ma, sa, fa] = args;
    let [
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
    ] = words;
    unsafe {
        if down {
            expert_tail_down::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
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
                member_base,
                piece_members,
                tile,
            );
        } else {
            expert_tail_gu::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
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
                member_base,
                piece_members,
                tile,
            );
        }
    }
    crate::strata_prefill_profile::count(
        if down {
            "expert_small_direct_tail_down_products"
        } else {
            "expert_small_direct_tail_gu_products"
        },
        1,
    );
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn expert_macro(
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
    #[comptime] down: bool,
) {
    crate::strata_matrix::small_direct::products::macro_product::<f16>(
        input,
        weight,
        columns,
        rows,
        count,
        weight_first,
        output,
        map,
        status,
        fault,
        group,
        0,
        count,
        down,
    );
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn expert_macro64(
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
    #[comptime] down: bool,
) {
    // Workgroup-uniform guards precede every map-derived index and barrier.
    let mut header_good = false;
    if group < 512 && map.len() == 65536 {
        let total = usize::cast_from(map[0]);
        if total >= 1 && total <= 4096 && group < usize::cast_from(map[3]) {
            let prefix = usize::cast_from(map[1028 + group]);
            header_good = map[1] == 0
                && usize::cast_from(map[2]) == total
                && map[3] <= 512
                && map[4 + group] < 512
                && usize::cast_from(map[516 + group]) == count
                && count >= 64
                && count < 256
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good
        || columns != comptime!(if down { 640 } else { 2560 })
        || rows != comptime!(if down { 2560 } else { 1280 })
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if comptime!(cfg!(feature = "strata-macro64-static-geometry")) {
        // Header/role validation above still fires first. A surplus X tile
        // originally performed no output stores; close it before fixed loads.
        if usize::cast_from(CUBE_POS_X) >= rows / 64 {
            terminate!();
        }
    }
    crate::strata_matrix::small_direct::products::macro64_product::<f16>(
        input,
        weight,
        columns,
        rows,
        count,
        weight_first,
        output,
        map,
        status,
        fault,
        group,
        0,
        count,
        down,
    );
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn expert_narrow(
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
    #[comptime] down: bool,
    #[comptime] tile: usize,
    #[comptime] direct: bool,
) {
    crate::strata_matrix::small_direct::products::narrow_product::<f16>(
        input,
        weight,
        columns,
        rows,
        count,
        weight_first,
        tile,
        direct,
        output,
        map,
        status,
        fault,
        group,
        0,
        count,
        down,
    );
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
    if !crate::strata_matrix::small_direct::tail_header_valid(map, count, group)
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
    if comptime!(tile == 32) {
        crate::strata_matrix::small_direct::products::macro_product::<f16>(
            input,
            weight,
            columns,
            rows,
            piece_members,
            weight_first,
            output,
            map,
            status,
            fault,
            group,
            member_base,
            count,
            down,
        );
    } else {
        crate::strata_matrix::small_direct::products::narrow_product::<f16>(
            input,
            weight,
            columns,
            rows,
            piece_members,
            weight_first,
            tile,
            false,
            output,
            map,
            status,
            fault,
            group,
            member_base,
            count,
            down,
        );
    }
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
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
    crate::strata_matrix::small_direct::tail(
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
#[allow(clippy::too_many_arguments)]
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
    crate::strata_matrix::small_direct::tail(
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
#[cube]
fn tail_header_valid(map: &Array<u32>, count: usize, group: usize) -> bool {
    let total = usize::cast_from(map[0]);
    let mut valid = false;
    if total >= 1
        && total <= comptime!(crate::strata_prompt::groups::ROUTES / 10)
        && group < 512
        && group < usize::cast_from(map[3])
    {
        let offset = usize::cast_from(map[1028 + group]);
        valid = map[1] == 0
            && usize::cast_from(map[2]) == total
            && usize::cast_from(map[3]) <= 512
            && usize::cast_from(map[4 + group]) < 512
            && usize::cast_from(map[516 + group]) == count
            && count
                >= comptime!(if cfg!(feature = "strata-native-expert-medium") {
                    128usize
                } else {
                    256usize
                })
            && count <= total
            && offset <= total * 10
            && count <= total * 10 - offset;
    }
    valid
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = vec![];
    for (name, family, down, tile, direct) in [
        ("small_direct_expert_macro_gu", 0, false, 0, false),
        ("small_direct_expert_macro_down", 0, true, 0, false),
        ("small_direct_expert_macro64_gu", 1, false, 0, false),
        ("small_direct_expert_macro64_down", 1, true, 0, false),
        ("small_direct_expert_narrow8_gu", 2, false, 8, false),
        ("small_direct_expert_narrow8_down", 2, true, 8, false),
        ("small_direct_expert_narrow16_gu", 2, false, 16, false),
        ("small_direct_expert_narrow16_down", 2, true, 16, false),
        ("small_direct_expert_global_narrow8_gu", 2, false, 8, true),
        ("small_direct_expert_global_narrow8_down", 2, true, 8, true),
        ("small_direct_expert_global_narrow16_gu", 2, false, 16, true),
        (
            "small_direct_expert_global_narrow16_down",
            2,
            true,
            16,
            true,
        ),
        ("small_direct_expert_tail8_gu", 3, false, 8, false),
        ("small_direct_expert_tail8_down", 3, true, 8, false),
        ("small_direct_expert_tail16_gu", 3, false, 16, false),
        ("small_direct_expert_tail16_down", 3, true, 16, false),
        ("small_direct_expert_tail32_gu", 3, false, 32, false),
        ("small_direct_expert_tail32_down", 3, true, 32, false),
    ] {
        out.push((name, definition(family, down, tile, direct)));
    }
    out
}
fn definition(family: usize, down: bool, tile: usize, direct: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let t = Vector::<f16, Const<4>>::as_type(&b.scope);
    let input = b.input_array(t);
    let weight = b.input_array(t);
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: Vec<cubecl::ir::ManagedVariable> = (0..if family == 3 { 7 } else { 5 })
        .map(|_| b.scalar(u32::as_type(&b.scope).storage_type()))
        .collect();
    match family {
        0 => expert_macro::expand(
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
            down,
        ),
        1 => expert_macro64::expand(
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
            down,
        ),
        2 => expert_narrow::expand(
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
            down,
            tile,
            direct,
        ),
        _ => {
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
                )
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
                )
            }
        }
    }
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(if family == 1 { 128 } else { 64 })))
}
