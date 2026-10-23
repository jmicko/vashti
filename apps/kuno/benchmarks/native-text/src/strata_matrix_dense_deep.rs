// Full M128/N128/K64 operands, double buffered. The direct producer
// has no publication scratch: exactly64KiB with or without explicit layout.
#[cube]
#[allow(clippy::too_many_arguments)]
fn deep_stage<F: Float>(
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
    #[comptime] _gr_down: bool,
) {
    #[unroll]
    for round in 0..2usize {
        let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + round * 512;
        let offset = k + (job % 8) * 8;
        let row = job / 8;
        let sa = ((token_base + row) * columns + offset) / 4;
        let sb = (weight_first + (row_base + row) * columns + offset) / 4;
        a[buffer * 2048 + job * 2] = input[sa];
        a[buffer * 2048 + job * 2 + 1] = input[sa + 1];
        b[buffer * 2048 + job * 2] = weight[sb];
        b[buffer * 2048 + job * 2 + 1] = weight[sb + 1];
    }
}

#[cube(launch_unchecked)]
fn dense_deep_store<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] gr_down: bool,
) {
    // Same source operands and chronological K16 products, K64 stage. No publication
    // scratch allocation: cooperative stores target the actual output owner.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(4096usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(4096usize);
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
    let a2 = if comptime!(M_FRAGMENTS == 4) {
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
    let a3 = if comptime!(M_FRAGMENTS == 4) {
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
    deep_stage::<F>(
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
        // DEEP_REGISTER_LOAD_BEGIN: two genuine vec4 pairs per lane,
        // kept in registers across the current four chronological K16 steps.
        let mut next_a0_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a0_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a1_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a1_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b0_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b0_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b1_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b1_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        if k + 64 < columns {
            let job0 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + 0;
            let row0 = job0 / 8;
            let off0 = k + 64 + (job0 % 8) * 8;
            let sa0 = ((token_base + row0) * columns + off0) / 4;
            let sb0 = (weight_first + (row_base + row0) * columns + off0) / 4;
            next_a0_lo = input[sa0];
            next_a0_hi = input[sa0 + 1];
            next_b0_lo = weight[sb0];
            next_b0_hi = weight[sb0 + 1];
            let job1 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + 512;
            let row1 = job1 / 8;
            let off1 = k + 64 + (job1 % 8) * 8;
            let sa1 = ((token_base + row1) * columns + off1) / 4;
            let sb1 = (weight_first + (row_base + row1) * columns + off1) / 4;
            next_a1_lo = input[sa1];
            next_a1_hi = input[sa1 + 1];
            next_b1_lo = weight[sb1];
            next_b1_hi = weight[sb1 + 1];
        }
        // DEEP_REGISTER_LOAD_END
        #[unroll]
        for step in 0..4usize {
            let af = current * 2048 + (subgroup / 4) * comptime!(SUBGROUP_ROWS * 16) + step * 4;
            let bf = current * 2048 + (subgroup % 4) * 512 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 64);
            cmma::load(&a1, &a.to_slice().slice(af + 128, a.len()), 64);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::load(&a2, &a.to_slice().slice(af + 256, a.len()), 64);
                cmma::load(&a3, &a.to_slice().slice(af + 384, a.len()), 64);
            }
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 64);
            cmma::load(&b1, &b.to_slice().slice(bf + 256, b.len()), 64);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::execute::<F, F, f32, f32>(&a2, &b0, &c4, &c4);
                cmma::execute::<F, F, f32, f32>(&a2, &b1, &c5, &c5);
                cmma::execute::<F, F, f32, f32>(&a3, &b0, &c6, &c6);
                cmma::execute::<F, F, f32, f32>(&a3, &b1, &c7, &c7);
            }
        }
        // DEEP_REGISTER_STORE_BEGIN: only after all four chronological
        // K16 steps; every destination is in the disjoint next bank.
        if k + 64 < columns {
            let job0 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + 0;
            let dest0 = next * 2048 + job0 * 2;
            a[dest0] = next_a0_lo;
            a[dest0 + 1] = next_a0_hi;
            b[dest0] = next_b0_lo;
            b[dest0 + 1] = next_b0_hi;
            let job1 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE) + 512;
            let dest1 = next * 2048 + job1 * 2;
            a[dest1] = next_a1_lo;
            a[dest1 + 1] = next_a1_hi;
            b[dest1] = next_b1_lo;
            b[dest1 + 1] = next_b1_hi;
        }
        // DEEP_REGISTER_STORE_END
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 64;
    }
    publish_global(
        &c0, output, token_base, row_base, out_first, out_stride, 0usize, 0usize, gr_down,
    );
    publish_global(
        &c1, output, token_base, row_base, out_first, out_stride, 0usize, 1usize, gr_down,
    );
    publish_global(
        &c2, output, token_base, row_base, out_first, out_stride, 1usize, 0usize, gr_down,
    );
    publish_global(
        &c3, output, token_base, row_base, out_first, out_stride, 1usize, 1usize, gr_down,
    );
    if comptime!(M_FRAGMENTS == 4) {
        publish_global(
            &c4, output, token_base, row_base, out_first, out_stride, 2usize, 0usize, gr_down,
        );
        publish_global(
            &c5, output, token_base, row_base, out_first, out_stride, 2usize, 1usize, gr_down,
        );
        publish_global(
            &c6, output, token_base, row_base, out_first, out_stride, 3usize, 0usize, gr_down,
        );
        publish_global(
            &c7, output, token_base, row_base, out_first, out_stride, 3usize, 1usize, gr_down,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_deep_store(
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
            dense_deep_store::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(THREADS),
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
            dense_deep_store::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static((rows / 128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(THREADS),
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
    crate::strata_prefill_profile::count("dense_deep_k64_products", 1);
    Ok(())
}

fn deep_definition<F: Float>(gr_down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_deep_store::expand::<F>(
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
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}

fn deep_geometry(
    count: usize,
    columns: usize,
    rows: usize,
    first: usize,
    shared: usize,
    units: u32,
    dim: (u32, u32, u32),
    pmin: u32,
    pmax: u32,
    minimum: usize,
) -> bool {
    selected_min(
        count, columns, rows, first, shared, units, dim, pmin, pmax, minimum,
    ) && columns.is_multiple_of(64)
        && shared >= 65536
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_deep_dense(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
    minimum: usize,
) -> Result<bool> {
    let hw = &client.properties().hardware;
    // Complete N128 tiles only. Accepted genuine GR320+injection fusion and
    // every original masked/fallback shape remain available without change.
    if !deep_geometry(
        count,
        columns,
        rows,
        weight_first,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
        minimum,
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
    if rows / 128 > hw.max_cube_count.0 as usize
        || count / 128 > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("deep dense grid"));
    }
    run_deep_store(
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
        grid,
    )?;
    Ok(true)
}

// Real complete-M128 prefix plus original exact remaining-member producer.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_deep_dense_split(
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
    if !rows.is_multiple_of(128) || !columns.is_multiple_of(64) {
        return Ok(false);
    }
    let gr = false;
    let hw = &client.properties().hardware;
    if !vector_views(&h)
        || !deep_geometry(
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
    // Same physical hardware/CMMA/owners/grid checks, real count128 or more.
    // The original general selector retains its original minimum512 policy.
    if !try_deep_dense(
        client,
        h.clone(),
        prefix,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
        bfloat,
        128,
    )? {
        return Ok(false);
    }
    if suffix > 0 {
        super::native::run_dense_original(
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

pub(super) fn validate_deep_dense_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8 {
        for (name, def) in [
            ("fp16", deep_definition::<f16>(false)),
            ("bf16", deep_definition::<bf16>(false)),
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
                def,
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("deep dense compiler representation".into());
            };
            if k.shared_size != 65536 {
                return Err("deep dense actual64KiB envelope".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&k.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
                return Err("deep dense actualSG16".into());
            }
            super::wide_pipeline::validate_program_expected(
                &words,
                16,
                "strata_deep_register_stage_accepted",
            )?;
            let mut parents = std::collections::BTreeMap::new();
            let mut classes = std::collections::BTreeMap::new();
            let mut stores = Vec::new();
            let (mut products, mut atomics) = (0, 0);
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if n == 0 || at + n > words.len() {
                    return Err("deep dense opcode extent".into());
                }
                match words[at] & 65535 {
                    59 => {
                        classes.insert(words[at + 2], words[at + 3]);
                    }
                    65 | 66 | 67 | 83 | 124 => {
                        parents.insert(words[at + 2], words[at + 3]);
                    }
                    4458 => stores.push(words[at + 1]),
                    4459 => products += 1,
                    241 => atomics += 1,
                    _ => {}
                }
                at += n;
            }
            if products != 16 || stores.len() != 4 || atomics != 0 {
                return Err("deep dense exact orderedproducts/stores/noatomics".into());
            }
            for mut pointer in stores {
                let mut seen = std::collections::BTreeSet::new();
                while let Some(&parent) = parents.get(&pointer) {
                    if !seen.insert(pointer) {
                        return Err("deep dense pointer cycle".into());
                    }
                    pointer = parent;
                }
                if classes.get(&pointer) != Some(&12) {
                    return Err("deep dense genuine global publication".into());
                }
            }
            let mut val = std::process::Command::new("spirv-val");
            val.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                val.arg("--uniform-buffer-standard-layout");
            }
            let mut child = val.arg("-").stdin(std::process::Stdio::piped()).spawn()?;
            child
                .stdin
                .take()
                .ok_or("deep dense validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("deep dense SPIRV validation".into());
            }
            println!(
                "strata_deep_dense_shader_accepted dtype={name} layout_mask={mask} shared_bytes=65536"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod deep_dense_tests {
    use super::*;
    #[test]
    fn full_deep_stage_vector_sources_and_destinations_cover_each_actual_operand_once() {
        for bank in 0..2usize {
            let mut positions = vec![0u8; 2048];
            for lane in 0..512usize {
                for round in 0..2usize {
                    let job = lane + round * 512;
                    let row = job / 8;
                    let offset = (job % 8) * 8;
                    assert!(row < 128 && offset + 7 < 64);
                    assert_eq!(row * 16 + offset / 4, job * 2);
                    for half in 0..2 {
                        positions[job * 2 + half] += 1;
                        assert!(bank * 2048 + job * 2 + half < 4096);
                    }
                }
            }
            assert!(positions.iter().all(|&n| n == 1));
        }
        // Every logical128x128 output is covered once by the32 real SG16
        // groups and their four actual8x16 accumulators, without padding.
        let mut outputs = vec![0u8; 128 * 128];
        for sg in 0..32 {
            for m in 0..2 {
                for n in 0..2 {
                    for i in 0..8 {
                        for j in 0..16 {
                            let row = (sg / 4) * 16 + m * 8 + i;
                            let col = (sg % 4) * 32 + n * 16 + j;
                            outputs[row * 128 + col] += 1;
                        }
                    }
                }
            }
        }
        assert!(outputs.iter().all(|&n| n == 1));
    }
    #[test]
    fn deep_reduction_preserves_every_chronological_k16_step_without_final_prefetch() {
        for columns in (64..=10240usize).step_by(64) {
            let chronological = (0..columns)
                .step_by(64)
                .flat_map(|k| (0..4).map(move |s| k + s * 16))
                .collect::<Vec<_>>();
            assert_eq!(chronological, (0..columns).step_by(16).collect::<Vec<_>>());
            let mut stages = vec![0u8; columns / 64];
            stages[0] = 1;
            for k in (0..columns).step_by(64) {
                if k + 64 < columns {
                    stages[(k + 64) / 64] += 1;
                    assert!(k + 64 + 63 < columns);
                }
            }
            assert!(stages.iter().all(|&n| n == 1));
        }
    }
    #[test]
    fn deep_selection_requires_real_full_tiles_and64k_without_replacing_original_fallbacks() {
        assert!(deep_geometry(
            512,
            2560,
            128,
            0,
            65536,
            512,
            (512, 1, 1),
            16,
            32,
            512
        ));
        assert!(deep_geometry(
            128,
            2560,
            128,
            0,
            65536,
            512,
            (512, 1, 1),
            16,
            32,
            128
        ));
        for (count, k, n, shared) in [
            (512, 2560, 128, 65535),
            (512, 2592, 128, 65536),
            (512, 2560, 320, 65536),
            (148, 2560, 128, 65536),
            (128, 2560, 128, 65536),
        ] {
            assert!(!deep_geometry(
                count,
                k,
                n,
                0,
                shared,
                512,
                (512, 1, 1),
                16,
                32,
                512
            ));
        }
    }
    #[test]
    fn actual_deep_dense_modules_retain64k_and_ordered_products_and_real_global_stores() {
        validate_deep_dense_sources().unwrap();
    }
}
