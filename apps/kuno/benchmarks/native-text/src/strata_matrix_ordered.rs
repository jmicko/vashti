// Additive expert-ordered PART producers. Original entries remain rank-ordered.
use super::*;

#[cube(launch_unchecked)]
pub fn expert(
    input: &Array<f16>,
    weight: &Array<f16>,
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
    let values = product::<f16>(input, weight, columns, rows, count, weight_first);
    let mut i = usize::cast_from(UNIT_POS);
    while i < 128 {
        let member = usize::cast_from(CUBE_POS_Y) * 8 + i / 16;
        let row = usize::cast_from(CUBE_POS_X) * 16 + i % 16;
        if member < count && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = member * 1280;
                if down {
                    base = (usize::cast_from(map[1028 + group]) + member) * 2560;
                }
                let v = values[i];
                let valid = v.abs() <= f32::MAX;
                if valid {
                    output[base + row] = v;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if down && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    // Same full-capacity plane as prompt activation and
                    // the rank-ordered reducer, including routes >20479.
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}

#[cube(launch_unchecked)]
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
    let values = macro_product::<f16>(input, weight, columns, rows, count, weight_first);
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < 512 {
        let member = usize::cast_from(CUBE_POS_Y) * 32
            + (subgroup / 2) * 16
            + (i / 256) * 8
            + (i % 128) / 16;
        let row =
            usize::cast_from(CUBE_POS_X) * 64 + (subgroup % 2) * 32 + ((i / 128) % 2) * 16 + i % 16;
        if member < count && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = member * 1280;
                if down {
                    base = (usize::cast_from(map[1028 + group]) + member) * 2560;
                }
                let v = values[subgroup * 512 + i];
                let valid = v.abs() <= f32::MAX;
                if valid {
                    output[base + row] = v;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if down && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    // Same full-capacity plane as prompt activation and
                    // the rank-ordered reducer, including routes >20479.
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}

#[cube(launch_unchecked)]
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
    let values = narrow_product::<f16>(
        input,
        weight,
        columns,
        rows,
        count,
        weight_first,
        tile,
        direct,
    );
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < comptime!(tile * 16) {
        let member = usize::cast_from(CUBE_POS_Y) * tile + i / 16;
        let row = usize::cast_from(CUBE_POS_X) * 64 + subgroup * 16 + i % 16;
        if member < count && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = member * 1280;
                if down {
                    base = (usize::cast_from(map[1028 + group]) + member) * 2560;
                }
                let v = values[subgroup * comptime!(tile * 16) + i];
                let valid = v.abs() <= f32::MAX;
                if valid {
                    output[base + row] = v;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if down && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    // Same full-capacity plane as prompt activation and
                    // the rank-ordered reducer, including routes >20479.
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}

fn definition_expert(down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let t = f16::as_type(&b.scope);
    let input = b.input_array(t);
    let weight = b.input_array(t);
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let args: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
        args[4].clone().into(),
        down,
    );
    b.build(settings())
}

fn definition_expert_macro(down: bool) -> KernelDefinition {
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
    let args: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_macro::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
        args[4].clone().into(),
        down,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
}

fn definition_expert_narrow_at(down: bool, tile: usize, direct: bool) -> KernelDefinition {
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
    let args: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_narrow::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
        args[4].clone().into(),
        down,
        tile,
        direct,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
}

#[cfg(feature = "strata-expert-macro64")]
#[cube(launch_unchecked)]
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
    let values = macro64_product::<f16>(input, weight, columns, rows, count, weight_first, down);
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < 512 {
        let member = usize::cast_from(CUBE_POS_Y) * 64
            + (subgroup / 2) * 16
            + (i / 256) * 8
            + (i % 128) / 16;
        let row =
            usize::cast_from(CUBE_POS_X) * 64 + (subgroup % 2) * 32 + ((i / 128) % 2) * 16 + i % 16;
        if member < count && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = member * 1280;
                if down {
                    base = (usize::cast_from(map[1028 + group]) + member) * 2560;
                }
                let v = values[subgroup * 512 + i];
                let valid = v.abs() <= f32::MAX;
                if valid {
                    output[base + row] = v;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if down && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    // Same full-capacity plane as prompt activation and
                    // the rank-ordered reducer, including routes >20479.
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}

#[cfg(feature = "strata-expert-macro64")]
fn definition_expert_macro64(down: bool) -> KernelDefinition {
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
    let args: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_macro64::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
        args[4].clone().into(),
        down,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(128)))
}

#[cfg(feature = "strata-deep-matrix")]
#[cube(launch_unchecked)]
pub fn expert_macro64_deep(
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
    let values = deep_macro64::macro64_product::<f16>(
        input,
        weight,
        columns,
        rows,
        count,
        weight_first,
        down,
    );
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let mut i = usize::cast_from(UNIT_POS) % 16;
    while i < 512 {
        let member = usize::cast_from(CUBE_POS_Y) * 64
            + (subgroup / 2) * 16
            + (i / 256) * 8
            + (i % 128) / 16;
        let row =
            usize::cast_from(CUBE_POS_X) * 64 + (subgroup % 2) * 32 + ((i / 128) % 2) * 16 + i % 16;
        if member < count && row < rows {
            let route = usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
            if route < usize::cast_from(map[0]) * 10 {
                let mut base = member * 1280;
                if down {
                    base = (usize::cast_from(map[1028 + group]) + member) * 2560;
                }
                let v = values[subgroup * 512 + i];
                let valid = v.abs() <= f32::MAX;
                if valid {
                    output[base + row] = v;
                } else {
                    output[base + row] = 0.0;
                    fault[0].fetch_or(128);
                }
                if down && row == 0 {
                    let mut good = 0u32;
                    if valid && status[route] == 1 {
                        good = 1;
                    }
                    // Same full-capacity plane as prompt activation and
                    // the rank-ordered reducer, including routes >20479.
                    status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                }
            } else {
                fault[0].fetch_or(128);
            }
        }
        i += 16;
    }
}

#[cfg(feature = "strata-deep-matrix")]
pub(super) fn definition_expert_macro64_deep(down: bool) -> KernelDefinition {
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
    let args: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    expert_macro64_deep::expand(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        map.into(),
        status.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
        args[4].clone().into(),
        down,
    );
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(128)))
}

pub(crate) fn run_expert(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> Result<()> {
    if !down {
        return Err(Error::InvalidBufferLength);
    }
    compatible(client, false)?;
    #[cfg(feature = "strata-native-expert-wide")]
    if super::super::native_expert_wide::ordered::try_expert(
        client,
        h.clone(),
        count,
        columns,
        rows,
        weight_first,
        group,
        down,
    )? {
        return Ok(());
    }

    let [input, weight, output, map, status, fault] = h;
    #[cfg(feature = "strata-narrow-global")]
    if (1..32).contains(&count) {
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
    #[cfg(feature = "strata-expert-macro64")]
    let macro64_selected = macro64_geometry(
        count,
        rows,
        columns,
        weight_first,
        client.properties().hardware.max_shared_memory_size,
        [&input, &weight].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0))),
    );
    #[cfg(feature = "strata-deep-matrix")]
    let deep_macro64_selected = use_macro
        && macro64_selected
        && deep_macro64::selected(
            client,
            count,
            rows,
            columns,
            weight_first,
            [&input, &weight].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0))),
        );
    let narrow = narrow_compatible(client, &input, &weight, count, rows, columns, weight_first);
    let ia = if use_macro || narrow.is_some() {
        macro_half_arg(input)?
    } else {
        half_arg(input)?
    };
    let wa = if use_macro || narrow.is_some() {
        macro_half_arg(weight)?
    } else {
        half_arg(weight)?
    };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
    let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    // SAFETY: the actual prompt grouping/row map and all physical packed
    // images were checked by the original-source expert caller.
    unsafe {
        #[cfg(feature = "strata-expert-macro64")]
        if use_macro && macro64_selected {
            crate::strata_prefill_profile::count("expert_macro64_products", 1);
            #[cfg(feature = "strata-deep-matrix")]
            if deep_macro64_selected {
                crate::strata_prefill_profile::count("expert_macro64_deep_products", 1);
                expert_macro64_deep::launch_unchecked::<WgpuRuntime>(
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
                return Ok(());
            }
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
            return Ok(());
        }
        if use_macro {
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
        } else {
            expert::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(16) as u32, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
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
        }
    }
    Ok(())
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = vec![
        ("ordered_expert_down", definition_expert(true)),
        ("ordered_expert_macro_down", definition_expert_macro(true)),
        (
            "ordered_expert_narrow8_down",
            definition_expert_narrow_at(true, 8, false),
        ),
        (
            "ordered_expert_narrow16_down",
            definition_expert_narrow_at(true, 16, false),
        ),
    ];
    #[cfg(feature = "strata-narrow-global")]
    out.extend([
        (
            "ordered_expert_global_narrow8_down",
            definition_expert_narrow_at(true, 8, true),
        ),
        (
            "ordered_expert_global_narrow16_down",
            definition_expert_narrow_at(true, 16, true),
        ),
    ]);
    #[cfg(feature = "strata-expert-macro64")]
    out.push((
        "ordered_expert_macro64_down",
        definition_expert_macro64(true),
    ));
    #[cfg(feature = "strata-deep-matrix")]
    out.push((
        "ordered_expert_macro64_deep_down",
        definition_expert_macro64_deep(true),
    ));
    out
}
