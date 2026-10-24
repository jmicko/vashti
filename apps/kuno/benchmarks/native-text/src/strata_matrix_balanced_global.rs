//! Pure-global two-C cells: 16 members x16 output rows per SG16.
//! Original real grid, padded producers, independent8-half publication and ABI.
use super::*;
const PADDED_GLOBAL:bool=true;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn product(
 input:&Array<Vector<f16,Const<4>>>,weight:&Array<Vector<f16,Const<4>>>,
 output:&mut Array<f32>,map:&Array<u32>,status:&mut Array<u32>,fault:&mut Array<Atomic<u32>>,
 columns:usize,rows:usize,count:usize,weight_first:usize,group:usize,
 input_member_base:usize,gu_prefix:usize,#[comptime] down:bool,#[comptime] tile:usize,
){
 let sg=usize::cast_from(PLANE_POS);let lane=usize::cast_from(UNIT_POS_PLANE);
 let token=usize::cast_from(CUBE_POS_Y)*comptime!(tile)+(sg/4)*16;
 let row=usize::cast_from(CUBE_POS_X)*64+(sg%4)*16;
 if token>=count {terminate!();}
 let prefix=usize::cast_from(map[1028+group]);let max_route=usize::cast_from(map[0])*10;
 let mut good=false;
 if token+16<=count {good=usize::cast_from(map[1540+prefix+token+lane])<max_route;}
 let all_good=plane_all(good);
 let a=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::A,8usize,16usize,16usize,cmma::MatrixLayout::RowMajor)};
 let b=unsafe{cmma::Matrix::<f16>::uninitialized(cmma::MatrixIdent::B,8usize,16usize,16usize,cmma::MatrixLayout::ColMajor)};
 let c0=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
 let c1=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,8usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0);
 let mut k=0usize;
 while k<columns {
  cmma::load(&b,&input.to_slice().slice(((input_member_base+token)*columns+k)/4,input.len()),u32::cast_from(columns));
  cmma::load(&a,&weight.to_slice().slice((weight_first+row*columns+k)/4,weight.len()),u32::cast_from(columns));
  cmma::execute::<f16,f16,f32,f32>(&a,&b,&c0,&c0);
  cmma::load(&a,&weight.to_slice().slice((weight_first+(row+8)*columns+k)/4,weight.len()),u32::cast_from(columns));
  cmma::execute::<f16,f16,f32,f32>(&a,&b,&c1,&c1);
  k+=16;
 }
 if all_good {
  let mut first=(gu_prefix+token)*rows+row;
  if comptime!(down){first=(prefix+token)*rows+row;}
  cmma::store(&mut output.slice_mut(first,output.len()),&c0,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  cmma::store(&mut output.slice_mut(first+8,output.len()),&c1,u32::cast_from(rows),cmma::MatrixLayout::ColMajor);
  if comptime!(down)&&row==0 {let route=usize::cast_from(map[1540+prefix+token+lane]);status[comptime!(crate::strata_prompt::groups::ROUTES)+route]=2;}
 }else{
  let mut scratch=SharedMemory::<f32>::new(comptime!(tile*32));
  crate::strata_matrix::global_small::publish(&c0,&mut scratch,output,map,status,fault,token,row,count,gu_prefix,group,down);
  crate::strata_matrix::global_small::publish(&c1,&mut scratch,output,map,status,fault,token,row+8,count,gu_prefix,group,down);
 }
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro32_gu(
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
                && count >= 32
                && count < 64
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good || columns != 2560 || rows != 1280 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X != 128u32 || CUBE_DIM_Y != 1u32 || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if usize::cast_from(CUBE_POS_X) >= rows / 64 || usize::cast_from(CUBE_POS_Y) * 32 >= count {
        terminate!();
    }
    product(
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
        0usize,
        0usize,
        false,
        32usize,
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro32_down(
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
                && count >= 32
                && count < 64
                && count <= total
                && prefix <= total * 10
                && count <= total * 10 - prefix;
        }
    }
    if !header_good || columns != 640 || rows != 2560 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X != 128u32 || CUBE_DIM_Y != 1u32 || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if usize::cast_from(CUBE_POS_X) >= rows / 64 || usize::cast_from(CUBE_POS_Y) * 32 >= count {
        terminate!();
    }
    product(
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
        0usize,
        0usize,
        true,
        32usize,
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro64_gu(
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
    if !header_good || columns != 2560 || rows != 1280 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X != 256u32 || CUBE_DIM_Y != 1u32 || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if usize::cast_from(CUBE_POS_X) >= rows / 64 || usize::cast_from(CUBE_POS_Y) * 64 >= count {
        terminate!();
    }
    product(
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
        0usize,
        0usize,
        false,
        64usize,
    );
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_global_macro64_down(
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
    if !header_good || columns != 640 || rows != 2560 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if CUBE_DIM_X != 256u32 || CUBE_DIM_Y != 1u32 || CUBE_DIM_Z != 1u32 {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    if usize::cast_from(CUBE_POS_X) >= rows / 64 || usize::cast_from(CUBE_POS_Y) * 64 >= count {
        terminate!();
    }
    product(
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
        0usize,
        0usize,
        true,
        64usize,
    );
}

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
    if !(32..256).contains(&count) {
        return Ok(false);
    }
    compatible(client, false)?;
    if group >= 512
        || u32::try_from(weight_first).is_err()
        || columns != if down { 640 } else { 2560 }
        || rows != if down { 2560 } else { 1280 }
        || !macro_compatible(client, &h[0], &h[1], count, rows, columns, weight_first)
        || h.iter().any(|v| v.stream != h[0].stream)
        || ((if PADDED_GLOBAL {count.div_ceil(32)*32} else {count}) * columns * 2) as u64 > h[0].size_in_used()
        || weight_first
            .checked_add(rows * columns)
            .is_none_or(|n| (n as u64) * 2 > h[1].size_in_used())
        || h[2].size_in_used()
            < (if down {
                crate::strata_prompt::groups::PART_BYTES
            } else {
                (count * 1280 * 4) as u64
            })
        || h[3].size_in_used() != (crate::strata_prompt::groups::MAP_WORDS * 4) as u64
        || h[4].size_in_used() != (crate::strata_prompt::groups::STATUS_WORDS * 4) as u64
        || h[5].size_in_used() != 4
    {
        return Err(Error::InvalidBufferLength);
    }
    let tile = if count < 64 { 32usize } else { 64usize };
    let hw = &client.properties().hardware;
    if hw.max_units_per_cube<(tile*4)as u32 || hw.max_cube_dim.0<(tile*4)as u32 || hw.max_shared_memory_size<tile*128 {return Ok(false);}
    if rows / 64 > hw.max_cube_count.0 as usize
        || count.div_ceil(tile) > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("global small grid"));
    }
    let [i, w, o, m, s, f] = h;
    let ia = macro_half_arg(i)?;
    let wa = macro_half_arg(w)?;
    let raw = |v: Handle| crate::resident_util::raw_arg(v.clone(), v.size_in_used() as usize / 4);
    let oa = raw(o)?;
    let ma = raw(m)?;
    let sa = raw(s)?;
    let fa = raw(f)?;
    let grid = CubeCount::Static((rows / 64) as u32, count.div_ceil(tile) as u32, 1);
    let dim = CubeDim::new_1d(if tile == 64 { 256 } else { 128 });
    unsafe {
        match (tile, down) {
            (32, false) => expert_global_macro32_gu::launch_unchecked::<WgpuRuntime>(
                client,
                grid,
                dim,
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
            ),
            (32, true) => expert_global_macro32_down::launch_unchecked::<WgpuRuntime>(
                client,
                grid,
                dim,
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
            ),
            (64, false) => expert_global_macro64_gu::launch_unchecked::<WgpuRuntime>(
                client,
                grid,
                dim,
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
            ),
            (64, true) => expert_global_macro64_down::launch_unchecked::<WgpuRuntime>(
                client,
                grid,
                dim,
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
            ),
            _ => unreachable!(),
        }
    }
    if PADDED_GLOBAL {crate::strata_prefill_profile::count(if tile==64 {"expert_balanced_global_macro64_products"}else{"expert_balanced_global_macro32_products"},1);}
    crate::strata_prefill_profile::count(
        if tile == 64 {
            "expert_full_walk_macro64_products"
        } else {
            "expert_full_walk_macro32_products"
        },
        1,
    );
    if tile == 64 {
        crate::strata_prefill_profile::count("expert_macro64_products", 1);
    }
    crate::strata_prefill_profile::count(
        "expert_full_walk_macro_workgroups",
        (rows / 64 * count.div_ceil(tile)) as u64,
    );
    Ok(true)
}

pub(crate) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = Vec::new();
    for tile in [32usize, 64] {
        for down in [false, true] {
            let mut b = KernelBuilder::default();
            cubecl::ir::AddressType::U32.register(&mut b.scope);
            let t = Vector::<f16, Const<4>>::as_type(&b.scope);
            let i = b.input_array(t);
            let w = b.input_array(t);
            let o = b.output_array(f32::as_type(&b.scope));
            let m = b.input_array(u32::as_type(&b.scope));
            let s = b.output_array(u32::as_type(&b.scope));
            let f = b.output_array(
                cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
            );
            let a: [cubecl::ir::ManagedVariable; 5] =
                std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
            match (tile, down) {
                (32, false) => expert_global_macro32_gu::expand(
                    &mut b.scope,
                    i.into(),
                    w.into(),
                    o.into(),
                    m.into(),
                    s.into(),
                    f.into(),
                    a[0].clone().into(),
                    a[1].clone().into(),
                    a[2].clone().into(),
                    a[3].clone().into(),
                    a[4].clone().into(),
                ),
                (32, true) => expert_global_macro32_down::expand(
                    &mut b.scope,
                    i.into(),
                    w.into(),
                    o.into(),
                    m.into(),
                    s.into(),
                    f.into(),
                    a[0].clone().into(),
                    a[1].clone().into(),
                    a[2].clone().into(),
                    a[3].clone().into(),
                    a[4].clone().into(),
                ),
                (64, false) => expert_global_macro64_gu::expand(
                    &mut b.scope,
                    i.into(),
                    w.into(),
                    o.into(),
                    m.into(),
                    s.into(),
                    f.into(),
                    a[0].clone().into(),
                    a[1].clone().into(),
                    a[2].clone().into(),
                    a[3].clone().into(),
                    a[4].clone().into(),
                ),
                (64, true) => expert_global_macro64_down::expand(
                    &mut b.scope,
                    i.into(),
                    w.into(),
                    o.into(),
                    m.into(),
                    s.into(),
                    f.into(),
                    a[0].clone().into(),
                    a[1].clone().into(),
                    a[2].clone().into(),
                    a[3].clone().into(),
                    a[4].clone().into(),
                ),
                _ => unreachable!(),
            }
            let name = match (tile, down) {
                (32, false) => if PADDED_GLOBAL {"balanced_global_macro32_gu"}else{"small_full_walk_macro32_gu"},
                (32, true) => if PADDED_GLOBAL {"balanced_global_macro32_down"}else{"small_full_walk_macro32_down"},
                (64, false) => if PADDED_GLOBAL {"balanced_global_macro64_gu"}else{"small_full_walk_macro64_gu"},
                _ => if PADDED_GLOBAL {"balanced_global_macro64_down"}else{"small_full_walk_macro64_down"},
            };
            assert!(b.scope.pop_errors().is_empty());
            out.push((
                name,
                b.build(
                    KernelSettings::default().cube_dim(CubeDim::new_1d(if tile == 64 {
                        256
                    } else {
                        128
                    })),
                ),
            ));
        }
    }
    out
}

mod validation {include!("strata_matrix_balanced_global_validation.rs");}
pub(crate) use validation::validate_sources;
#[cfg(test)]mod tests {
 use super::*;
 #[test]fn actual_balanced_modules_keep_two_products_exact_geometry_distinct_A_and_publication(){validate_sources().unwrap();}
 #[test]fn real_subgroups_cover_each_output_cell_once_and_only_use_authentic_padding(){
  for tile in [16usize,32,64]{for count in 1usize..=255 {
   if tile==16 && count>=32 || tile==32 && !(32..64).contains(&count) || tile==64 && !(64..256).contains(&count){continue;}
   let padded=count.div_ceil(if tile==16 {16}else{32})*if tile==16 {16}else{32};
   let mut cells=vec![0u8;count*64];
   for block in 0..count.div_ceil(tile){for sg in 0..tile/4 {
    let token=block*tile+(sg/4)*16;let row=sg%4*16;if token>=count{continue;}
    assert!(token+16<=padded);
    for m in token..(token+16).min(count){for n in row..row+16 {cells[m*64+n]+=1;}}
   }}assert!(cells.into_iter().all(|n|n==1));
  }}
 }
}
pub(crate) fn profile_names()->[&'static str;5]{[
 core::any::type_name::<expert_global_macro32_gu::ExpertGlobalMacro32Gu<WgpuRuntime>>(),
 core::any::type_name::<expert_global_macro32_down::ExpertGlobalMacro32Down<WgpuRuntime>>(),
 core::any::type_name::<expert_global_macro64_gu::ExpertGlobalMacro64Gu<WgpuRuntime>>(),
 core::any::type_name::<expert_global_macro64_down::ExpertGlobalMacro64Down<WgpuRuntime>>(),
 core::any::type_name::<crate::strata_matrix::cohorts::balanced_global::expert_balanced_cohort::ExpertBalancedCohort<WgpuRuntime>>()
]}
mod probe{include!("strata_matrix_balanced_global_probe.rs");}
pub(crate) use probe::cases as public_numeric;
