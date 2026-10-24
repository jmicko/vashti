//! Larger genuine subgroup tiles, original chronological K16 arithmetic.
use super::*;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(in crate::strata_matrix) fn compute<F:Float>(input:&Array<Vector<F,Const<4>>>,weight:&Array<Vector<F,Const<4>>>,
c00:&cmma::Matrix<f32>,c01:&cmma::Matrix<f32>,c10:&cmma::Matrix<f32>,c11:&cmma::Matrix<f32>,c20:&cmma::Matrix<f32>,c21:&cmma::Matrix<f32>,c30:&cmma::Matrix<f32>,c31:&cmma::Matrix<f32>,token:usize,row:usize,columns:usize,weight_first:usize){
    // SAFETY: both input fragments and the reused weight fragment load before use.
    let a=unsafe{cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A,8usize,16usize,16usize,cmma::MatrixLayout::RowMajor)};
    let b0=unsafe{cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B,8usize,16usize,16usize,cmma::MatrixLayout::ColMajor)};
    let b1=unsafe{cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B,8usize,16usize,16usize,cmma::MatrixLayout::ColMajor)};
    let mut k=0usize;
    while k<columns{
        let bf=(token*columns+k)/4;
        cmma::load(&b0,&input.to_slice().slice(bf,input.len()),u32::cast_from(columns));
        cmma::load(&b1,&input.to_slice().slice(bf+16*columns/4,input.len()),u32::cast_from(columns));
        let af0=(weight_first+(row+0)*columns+k)/4;
        cmma::load(&a,&weight.to_slice().slice(af0,weight.len()),u32::cast_from(columns));
        cmma::execute::<F,F,f32,f32>(&a,&b0,c00,c00);
        cmma::execute::<F,F,f32,f32>(&a,&b1,c01,c01);
        let af1=(weight_first+(row+8)*columns+k)/4;
        cmma::load(&a,&weight.to_slice().slice(af1,weight.len()),u32::cast_from(columns));
        cmma::execute::<F,F,f32,f32>(&a,&b0,c10,c10);
        cmma::execute::<F,F,f32,f32>(&a,&b1,c11,c11);
        let af2=(weight_first+(row+16)*columns+k)/4;
        cmma::load(&a,&weight.to_slice().slice(af2,weight.len()),u32::cast_from(columns));
        cmma::execute::<F,F,f32,f32>(&a,&b0,c20,c20);
        cmma::execute::<F,F,f32,f32>(&a,&b1,c21,c21);
        let af3=(weight_first+(row+24)*columns+k)/4;
        cmma::load(&a,&weight.to_slice().slice(af3,weight.len()),u32::cast_from(columns));
        cmma::execute::<F,F,f32,f32>(&a,&b0,c30,c30);
        cmma::execute::<F,F,f32,f32>(&a,&b1,c31,c31);
        k+=16;
    }
}
#[cube(launch_unchecked)]
pub(in crate::strata_matrix) fn dense_global_wide32_store<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] gr_down: bool,
) {
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
    compute::<F>(input,weight,&c00,&c01,&c10,&c11,&c20,&c21,&c30,&c31,token,row,columns,weight_first);
    let first=out_first+token*out_stride+row;
    cmma::store(&mut output.slice_mut(first+0*out_stride+0,output.len()),&c00,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+0,output.len()),&c01,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+8,output.len()),&c10,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+8,output.len()),&c11,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+16,output.len()),&c20,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+16,output.len()),&c21,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+0*out_stride+24,output.len()),&c30,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
    cmma::store(&mut output.slice_mut(first+16*out_stride+24,output.len()),&c31,u32::cast_from(out_stride),cmma::MatrixLayout::ColMajor);
}
#[allow(clippy::too_many_arguments)]
fn run_global_wide32_store(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
    gr_down: bool,
    finish_grid: (u32, u32),
) -> Result<()> {
    let [input, weight, output, fault] = h;
    // Original try_dense has already checked all original capabilities, owners,
    // logical bounds and grids. Actual same-view WRITE->READ/WRITE dependencies
    // order the unsanitized producer and finite consumer without a host fence.
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 8) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 8) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    unsafe {
        if bfloat {
            dense_global_wide32_store::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(256),
                ia,
                wa,
                oa,
                columns,
                weight_first,
                out_first,
                out_stride,
                gr_down,
            );
        } else {
            dense_global_wide32_store::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static((rows / 128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(256),
                ia,
                wa,
                oa,
                columns,
                weight_first,
                out_first,
                out_stride,
                false,
            );
        }
        dense_finite::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(finish_grid.0, finish_grid.1, 1),
            CubeDim::new_1d(256),
            crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?,
            crate::resident_util::raw_arg(fault, 1)?,
            count,
            rows,
            out_first,
            out_stride,
        );
    }
    crate::strata_prefill_profile::count("dense_direct_store_products", 1);
    crate::strata_prefill_profile::count("dense_direct_store_finite_consumers", 1);
    crate::strata_prefill_profile::count("dense_global_wide32_products", 1);
    Ok(())
}

fn global_wide32_definition<F: Float>(gr_down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_global_wide32_store::expand::<F>(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        a[0].clone().into(),
        a[1].clone().into(),
        a[2].clone().into(),
        a[3].clone().into(),
        gr_down,
    );
    assert!(b.scope.pop_errors().is_empty());
    b.build(
        KernelSettings::default()
            .cube_dim(CubeDim::new_1d(256))
            .kernel_name("dense_global_wide32_store_f_"),
    )
}

#[allow(clippy::too_many_arguments)]
fn global_wide32_geometry(
    count: usize,
    columns: usize,
    rows: usize,
    first: usize,
    shared: usize,
    units: u32,
    dim: (u32, u32, u32),
    pmin: u32,
    pmax: u32,
) -> bool {
    selected_min(
        count, columns, rows, first, shared, units, dim, pmin, pmax, 128,
    ) && count.is_multiple_of(128)
        && shared >= 49152
        && units >= 256
        && dim.0 >= 256
}
#[allow(clippy::too_many_arguments)]
pub(in crate::strata_matrix) fn try_global_wide32_dense(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    try_global_wide32_impl(
        client,
        h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
        bfloat,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_global_wide32_impl(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
    prefix: bool,
) -> Result<bool> {
    let gr_down = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    if gr_down || !rows.is_multiple_of(128) {return Ok(false);}
    let hw = &client.properties().hardware;
    if !global_wide32_geometry(
        count,
        columns,
        if gr_down {
            rows.div_ceil(128) * 128
        } else {
            rows
        },
        weight_first,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
    ) || !vector_views(&h)
    {
        return Ok(false);
    }
    let kind = ElemType::Float(if bfloat {
        FloatKind::BF16
    } else {
        FloatKind::F16
    });
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
    check_bounds(
        &h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
    )?;
    let Some(grid) = direct_selection(&h[2], count, rows, out_first, out_stride, hw.max_cube_count)
    else {
        return Ok(false);
    };
    if rows.div_ceil(128) > hw.max_cube_count.0 as usize
        || count / 128 > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("global_wide32 dense grid"));
    }
    run_global_wide32_store(
        client,
        h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
        bfloat,
        gr_down,
        grid,
    )?;
    // Match the existing full-wide GR credit; shorter original prefix paths
    // intentionally did not claim this population.
    if full_gr_credit(gr_down, count, prefix) {
        crate::strata_prefill_profile::count("gr_wide_tail_products", 1);
    }
    Ok(true)
}

fn full_gr_credit(gr_down: bool, count: usize, prefix: bool) -> bool {
    gr_down && count >= 512 && !prefix
}

pub(crate) fn global_wide32_available(client: &ComputeClient<WgpuRuntime>) -> bool {
    let hw = &client.properties().hardware;
    global_wide32_geometry(
        512,
        2560,
        128,
        0,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
    ) && [FloatKind::F16, FloatKind::BF16].into_iter().all(|f| {
        let t = ElemType::Float(f);
        client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: t.into(),
                b_type: t.into(),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 8,
                n: 16,
                k: 16,
            })
    })
}
// Real complete-M128 prefix plus original exact remaining-member producer.
#[allow(clippy::too_many_arguments)]
pub(in crate::strata_matrix) fn try_global_wide32_split(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    let Some((prefix, suffix)) = prefix_partition(count) else {
        return Ok(false);
    };
    let gr = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    let hw = &client.properties().hardware;
    if !vector_views(&h)
        || !selected_min(
            prefix,
            columns,
            if gr {
                rows.div_ceil(BLOCK) * BLOCK
            } else {
                rows
            },
            weight_first,
            hw.max_shared_memory_size,
            hw.max_units_per_cube,
            hw.max_cube_dim,
            hw.plane_size_min,
            hw.plane_size_max,
            128,
        )
    {
        return Ok(false);
    }
    // Validate the entire real call before enqueuing its first prefix dispatch.
    // This does not advertise a larger input count or synthesize private tokens.
    check_bounds(
        &h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
    )?;
    let shifted = prefix_input_tail(&h[0], prefix, columns)?;
    let tail_first = out_first
        .checked_add(
            prefix
                .checked_mul(out_stride)
                .ok_or(Error::ArithmeticOverflow)?,
        )
        .ok_or(Error::ArithmeticOverflow)?;
    if suffix > 0 {
        check_bounds(
            &[shifted.clone(), h[1].clone(), h[2].clone(), h[3].clone()],
            suffix,
            columns,
            rows,
            weight_first,
            tail_first,
            out_stride,
        )?;
    }
    if !try_global_wide32_impl(
        client,
        h.clone(),
        prefix,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
        bfloat,
        true,
    )? {
        return Ok(false);
    }
    if suffix > 0 {
        crate::strata_matrix::native::run_dense_original(
            client,
            [shifted, h[1].clone(), h[2].clone(), h[3].clone()],
            suffix,
            columns,
            rows,
            weight_first,
            tail_first,
            out_stride,
            bfloat,
        )?;
    }
    crate::strata_prefill_profile::count("dense_prefix_products", 1);
    crate::strata_prefill_profile::count("dense_prefix_tokens", prefix as u64);
    crate::strata_prefill_profile::count("dense_suffix_tokens", suffix as u64);
    Ok(true)
}

#[path="strata_matrix_global_wide32_validation.rs"]
pub(in crate::strata_matrix) mod validation;
pub(in crate::strata_matrix) use validation::validate_sources as validate_global_wide32_sources;
#[cfg(test)]
#[path = "strata_matrix_global_wide32_tests.rs"]
mod tests;

#[cfg(test)]
pub(in crate::strata_matrix) fn profile_name() -> &'static str {
    core::any::type_name::<dense_global_wide32_store::DenseGlobalWide32Store<f16, WgpuRuntime>>()
}
