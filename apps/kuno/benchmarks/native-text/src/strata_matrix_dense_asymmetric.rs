// Complete asymmetric producer, chronological readers and direct publisher.
#[cube]
#[allow(clippy::too_many_arguments)]
fn asymmetric_stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    columns: usize,
    weight_first: usize,
    token_base: usize,
    row_base: usize,
    k: usize,
    bank: usize,
    #[comptime] gr_down: bool,
) {
    #[unroll]
    for round in 0..2usize {
        let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + round * 512;
        let row = job / 4;
        let col = (job % 4) * 8;
        let sa = ((token_base + row) * columns + k + col) / 4;
        let da = bank * 2048 + (col / 16) * 1024 + row * 4 + (col % 16) / 4;
        a[da] = input[sa];
        a[da + 1] = input[sa + 1];
        if round == 0 {
            let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
            let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
            if !comptime!(gr_down) || row_base + row < 320 {
                let sb = (weight_first + (row_base + row) * columns + k + col) / 4;
                lo = weight[sb];
                hi = weight[sb + 1];
            }
            let db = bank * 1024 + (col / 16) * 512 + row * 4 + (col % 16) / 4;
            b[db] = lo;
            b[db + 1] = hi;
        }
    }
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn asymmetric_publish_global(
    acc: &cmma::Matrix<f32>,
    output: &mut Array<f32>,
    token_base: usize,
    row_base: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
    #[comptime] gr_down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let token = token_base + (sg / 4) * 32 + comptime!(tile_m * 8);
    let row = row_base + (sg % 4) * 32 + comptime!(tile_n * 16);
    if !comptime!(gr_down) || row < 320 {
        let first = out_first + token * out_stride + row;
        cmma::store(
            &mut output.slice_mut(first, output.len()),
            acc,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::RowMajor,
        );
    }
}
#[cube(launch_unchecked)]
fn dense_asymmetric_store<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] gr_down: bool,
) {
    // Same source operands and chronological K16 products, M256/N128/K32 stage. No publication
    // scratch allocation: cooperative stores target the actual output owner.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(4096usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a2 = if comptime!(true) {
        unsafe {
            cmma::Matrix::<F>::uninitialized(
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
    let a3 = if comptime!(true) {
        unsafe {
            cmma::Matrix::<F>::uninitialized(
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
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
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
    let c4 = if comptime!(true) {
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
    let c5 = if comptime!(true) {
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
    let c6 = if comptime!(true) {
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
    let c7 = if comptime!(true) {
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
    let token_base = usize::cast_from(CUBE_POS_Y) * 256;
    let row_base = usize::cast_from(CUBE_POS_X) * 128;
    asymmetric_stage::<F>(
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
        gr_down,
    );
    sync_cube();
    let mut k = 0usize;
    let mut current = 0usize;
    while k < columns {
        let next = 1usize - current;
        // ASYMMETRIC_REGISTER_LOAD_BEGIN: two A jobs and one B job/lane.
        let mut next_a0_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a0_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a1_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a1_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b0_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b0_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        if k + 32 < columns {
            let job0 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
            let row0 = job0 / 4;
            let off = k + 32 + (job0 % 4) * 8;
            let sa0 = ((token_base + row0) * columns + off) / 4;
            next_a0_lo = input[sa0];
            next_a0_hi = input[sa0 + 1];
            let sa1 = ((token_base + row0 + 128) * columns + off) / 4;
            next_a1_lo = input[sa1];
            next_a1_hi = input[sa1 + 1];
            if !comptime!(gr_down) || row_base + row0 < 320 {
                let sb0 = (weight_first + (row_base + row0) * columns + off) / 4;
                next_b0_lo = weight[sb0];
                next_b0_hi = weight[sb0 + 1];
            }
        }
        // ASYMMETRIC_REGISTER_LOAD_END
        #[unroll]
        for step in 0..2usize {
            let af = current * 2048 + step * 1024 + (subgroup / 4) * 128;
            let bf = current * 1024 + step * 512 + (subgroup % 4) * 128;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 16);
            cmma::load(&a1, &a.to_slice().slice(af + 32, a.len()), 16);
            if comptime!(true) {
                cmma::load(&a2, &a.to_slice().slice(af + 64, a.len()), 16);
                cmma::load(&a3, &a.to_slice().slice(af + 96, a.len()), 16);
            }
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 16);
            cmma::load(&b1, &b.to_slice().slice(bf + 64, b.len()), 16);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
            if comptime!(true) {
                cmma::execute::<F, F, f32, f32>(&a2, &b0, &c4, &c4);
                cmma::execute::<F, F, f32, f32>(&a2, &b1, &c5, &c5);
                cmma::execute::<F, F, f32, f32>(&a3, &b0, &c6, &c6);
                cmma::execute::<F, F, f32, f32>(&a3, &b1, &c7, &c7);
            }
        }
        // ASYMMETRIC_REGISTER_STORE_BEGIN: complete current products first.
        if k + 32 < columns {
            let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
            let da0 = next * 2048 + ((job % 4) / 2) * 1024 + (job / 4) * 4 + (job % 2) * 2;
            let da1 = da0 + 512;
            let db = next * 1024 + ((job % 4) / 2) * 512 + (job / 4) * 4 + (job % 2) * 2;
            a[da0] = next_a0_lo;
            a[da0 + 1] = next_a0_hi;
            a[da1] = next_a1_lo;
            a[da1 + 1] = next_a1_hi;
            b[db] = next_b0_lo;
            b[db + 1] = next_b0_hi;
        }
        // ASYMMETRIC_REGISTER_STORE_END
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 32;
    }
    asymmetric_publish_global(
        &c0, output, token_base, row_base, out_first, out_stride, 0usize, 0usize, gr_down,
    );
    asymmetric_publish_global(
        &c1, output, token_base, row_base, out_first, out_stride, 0usize, 1usize, gr_down,
    );
    asymmetric_publish_global(
        &c2, output, token_base, row_base, out_first, out_stride, 1usize, 0usize, gr_down,
    );
    asymmetric_publish_global(
        &c3, output, token_base, row_base, out_first, out_stride, 1usize, 1usize, gr_down,
    );
    if comptime!(true) {
        asymmetric_publish_global(
            &c4, output, token_base, row_base, out_first, out_stride, 2usize, 0usize, gr_down,
        );
        asymmetric_publish_global(
            &c5, output, token_base, row_base, out_first, out_stride, 2usize, 1usize, gr_down,
        );
        asymmetric_publish_global(
            &c6, output, token_base, row_base, out_first, out_stride, 3usize, 0usize, gr_down,
        );
        asymmetric_publish_global(
            &c7, output, token_base, row_base, out_first, out_stride, 3usize, 1usize, gr_down,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_asymmetric_store(
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
            dense_asymmetric_store::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(128) as u32, (count / 256) as u32, 1),
                CubeDim::new_1d(512),
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
            dense_asymmetric_store::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static((rows / 128) as u32, (count / 256) as u32, 1),
                CubeDim::new_1d(512),
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
    crate::strata_prefill_profile::count("dense_asymmetric_products", 1);
    Ok(())
}

fn asymmetric_definition<F: Float>(gr_down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_asymmetric_store::expand::<F>(
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
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(512)))
}

#[allow(clippy::too_many_arguments)]
fn asymmetric_geometry(
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
        count, columns, rows, first, shared, units, dim, pmin, pmax, 256,
    ) && count.is_multiple_of(256)
        && shared >= 49152
        && units >= 512
        && dim.0 >= 512
}
#[allow(clippy::too_many_arguments)]
pub(super) fn try_asymmetric_dense(
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
    let gr_down = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    let hw = &client.properties().hardware;
    if !asymmetric_geometry(
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
        || count / 256 > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("asymmetric dense grid"));
    }
    run_asymmetric_store(
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
    if gr_down && count >= 512 {
        crate::strata_prefill_profile::count("gr_wide_tail_products", 1);
    }
    Ok(true)
}

pub(crate) fn asymmetric_available(client: &ComputeClient<WgpuRuntime>) -> bool {
    let hw = &client.properties().hardware;
    asymmetric_geometry(
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
pub(super) fn validate_asymmetric_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write;
    for mask in 0..8u8 {
        for (name, definition) in [
            ("fp16", asymmetric_definition::<f16>(false)),
            ("bf16", asymmetric_definition::<bf16>(false)),
            ("gr_bf16", asymmetric_definition::<bf16>(true)),
        ] {
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
                definition,
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("asymmetric actual SPIRV".into());
            };
            if k.shared_size != 49152 {
                return Err(format!(
                    "asymmetric actual shared {name} mask{mask}: {}",
                    k.shared_size
                )
                .into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("asymmetric actual SG16".into());
            }
            super::wide_pipeline::validate_asymmetric_program(&words)?;
            let (mut parents, mut classes, mut constants) =
                (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
            let (mut loads, mut stores) = (Vec::new(), Vec::new());
            let (mut products, mut atomics, mut barriers) = (0, 0, 0);
            let mut wg512 = false;
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                let op = words[at] & 65535;
                if n == 0 || at + n > words.len() {
                    return Err("asymmetric opcode extent".into());
                }
                match op {
                    16 if n == 6 && words[at + 2] == 17 => {
                        wg512 = words[at + 3..at + 6] == [512, 1, 1]
                    }
                    43 if n == 4 => {
                        constants.insert(words[at + 2], words[at + 3]);
                    }
                    194 if n == 5 => {
                        if let (Some(&v), Some(&shift)) =
                            (constants.get(&words[at + 3]), constants.get(&words[at + 4]))
                        {
                            if shift < 32 {
                                constants.insert(words[at + 2], v >> shift);
                            }
                        }
                    }
                    59 => {
                        classes.insert(words[at + 2], words[at + 3]);
                    }
                    65 | 66 | 67 | 83 | 124 => {
                        parents.insert(words[at + 2], words[at + 3]);
                    }
                    4457 if n >= 6 => loads.push((words[at + 3], words[at + 5])),
                    4458 => stores.push(words[at + 1]),
                    4459 => products += 1,
                    241 => atomics += 1,
                    224 if constants.get(&words[at + 1]) == Some(&2) => barriers += 1,
                    _ => {}
                }
                at += n;
            }
            if !wg512
                || products != 16
                || loads.len() != 12
                || stores.len() != 8
                || atomics != 0
                || barriers != 2
            {
                return Err(format!("asymmetric actual WG/products/loads/stores/atomics/barriers {name} {wg512}/{products}/{}/{}/{atomics}/{barriers}",loads.len(),stores.len()).into());
            }
            let root = |mut v: u32| -> std::result::Result<u32, Box<dyn std::error::Error>> {
                let mut seen = BTreeSet::new();
                while let Some(&p) = parents.get(&v) {
                    if !seen.insert(v) {
                        return Err("asymmetric pointer cycle".into());
                    }
                    v = p;
                }
                classes
                    .get(&v)
                    .copied()
                    .ok_or_else(|| "asymmetric unproved storage".into())
            };
            for (ptr, stride) in loads {
                if root(ptr)? != 4 || constants.get(&stride) != Some(&4) {
                    return Err("asymmetric genuine shared stride16 cooperative load".into());
                }
            }
            for ptr in stores {
                if root(ptr)? != 12 {
                    return Err("asymmetric genuine global publication".into());
                }
            }
            let mut cmd = std::process::Command::new("spirv-val");
            cmd.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                cmd.arg("--uniform-buffer-standard-layout");
            }
            let mut child = cmd.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("asymmetric validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("asymmetric SPIRV validation".into());
            }
            println!("strata_asymmetric_dense_shader_accepted dtype={name} layout_mask={mask} shared_bytes=49152 workgroup=512 subgroup=16 m=256 n=128 k=32 products=16 shared_loads=12 global_stores=8 whole_workgroup_handoffs=2");
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "strata_matrix_asymmetric_tests.rs"]
mod asymmetric_tests;
