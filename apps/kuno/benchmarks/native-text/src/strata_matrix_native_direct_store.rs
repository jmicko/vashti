// Coupled direct dense publication and finite sanitizer; included in native_wide.
#[cube]
#[allow(clippy::too_many_arguments)]
fn direct_stage<F: Float>(
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
    #[comptime] gr_down: bool,
) {
    // Full128x128x32 stages only. Two genuine vec4s per job preserve original
    // FP16/BF16 bits; every one of the1024 vectors per operand is written once.
    #[unroll]
    for round in 0..comptime!(STAGE_ROUNDS) {
        // Genuine complete-SG16 lane mapping retains all original stage jobs.
        let job = usize::cast_from(PLANE_POS) * 16
            + usize::cast_from(UNIT_POS_PLANE)
            + round * comptime!(STAGE_THREADS);
        let offset = k + (job % 4) * 8;
        let row = job / 4;
        let source_a = ((token_base + row) * columns + offset) / 4;
        let source_b = (weight_first + (row_base + row) * columns + offset) / 4;
        a[buffer * 1024 + job * 2] = input[source_a];
        a[buffer * 1024 + job * 2 + 1] = input[source_a + 1];
        if comptime!(gr_down) {
            // The sole extra shape is the authentic320-row GR down image.
            // Do not read the missing last64 rows, even from a larger owner.
            let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
            let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
            if row_base + row < 320 {
                lo = weight[source_b];
                hi = weight[source_b + 1];
            }
            b[buffer * 1024 + job * 2] = lo;
            b[buffer * 1024 + job * 2 + 1] = hi;
        } else {
            b[buffer * 1024 + job * 2] = weight[source_b];
            b[buffer * 1024 + job * 2 + 1] = weight[source_b + 1];
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn publish_global(
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
    let token = token_base + (sg / 4) * comptime!(SUBGROUP_ROWS) + comptime!(tile_m * 8);
    let row = row_base + (sg % 4) * 32 + comptime!(tile_n * 16);
    // Every branch is uniform within its cooperative subgroup. The authentic
    // GR tail is64 features, so each16-feature fragment is wholly valid or absent.
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
fn dense_wide_store<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] gr_down: bool,
) {
    // Same authentic operand stages and chronological products. No publication
    // scratch allocation: cooperative stores target the actual output owner.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
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
    direct_stage::<F>(
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
        // Same accepted immediate global-load→SLM-store software-prefetch ordering.
        // This expresses overlap opportunity, not guaranteed asynchronous execution.
        if k + 32 < columns {
            direct_stage::<F>(
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
                gr_down,
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
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 32;
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

#[cube(launch_unchecked)]
fn dense_finite(
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    rows: usize,
    out_first: usize,
    out_stride: usize,
) {
    let group = usize::cast_from(CUBE_POS_X)
        + usize::cast_from(CUBE_POS_Y) * usize::cast_from(CUBE_COUNT_X);
    let i = group * 256 + usize::cast_from(UNIT_POS);
    if i < count * rows {
        let at = out_first + (i / rows) * out_stride + i % rows;
        let value = output[at];
        if value.abs() <= f32::MAX {
        } else {
            output[at] = 0.0;
            fault[0].fetch_or(1);
        }
    }
}
fn direct_selection(
    output: &Handle,
    count: usize,
    rows: usize,
    out_first: usize,
    out_stride: usize,
    max_grid: (u32, u32, u32),
) -> Option<(u32, u32)> {
    if !output.offset_start.unwrap_or(0).is_multiple_of(16)
        || !out_first.is_multiple_of(4)
        || !out_stride.is_multiple_of(4)
    {
        return None;
    }
    let groups = count.checked_mul(rows)?.div_ceil(256);
    if groups == 0 || max_grid.0 == 0 || max_grid.2 < 1 {
        return None;
    }
    let y = groups.div_ceil(max_grid.0 as usize);
    let x = groups.div_ceil(y);
    (y <= max_grid.1 as usize).then_some((x as u32, y as u32))
}
#[allow(clippy::too_many_arguments)]
fn run_direct_store(
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
            dense_wide_store::launch_unchecked::<bf16, WgpuRuntime>(
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
            dense_wide_store::launch_unchecked::<f16, WgpuRuntime>(
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
    Ok(())
}
fn direct_definition<F: Float>(gr_down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_wide_store::expand::<F>(
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
fn finite_definition() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let output = b.output_array(f32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_finite::expand(
        &mut b.scope,
        output.into(),
        fault.into(),
        a[0].clone().into(),
        a[1].clone().into(),
        a[2].clone().into(),
        a[3].clone().into(),
    );
    assert!(b.scope.pop_errors().is_empty());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(256)))
}
pub(super) fn validate_direct_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8 {
        for (name, def, producer) in [
            ("dense_store_fp16", direct_definition::<f16>(false), true),
            ("dense_store_bf16", direct_definition::<bf16>(false), true),
            ("dense_store_gr_bf16", direct_definition::<bf16>(true), true),
            ("dense_store_finite", finite_definition(), false),
        ] {
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
                def,
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("direct matrix compiler representation".into());
            };
            if kernel.shared_size != if producer { 32768 } else { 0 } {
                return Err("direct dense actual shared envelope".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            let sg = wgpu::hal::vulkan::kuno_cooperative_subgroup(&words);
            if (producer && sg != Some(16)) || (!producer && sg.is_some()) {
                return Err("direct composite subgroup identity".into());
            }
            let mut parents = std::collections::BTreeMap::new();
            let mut classes = std::collections::BTreeMap::new();
            let mut pointers = Vec::new();
            let (mut multiply, mut atomic) = (0, 0);
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if n == 0 || at + n > words.len() {
                    return Err("direct opcode extent".into());
                }
                match words[at] & 65535 {
                    59 => {
                        classes.insert(words[at + 2], words[at + 3]);
                    }
                    65 | 66 | 67 | 83 | 124 => {
                        parents.insert(words[at + 2], words[at + 3]);
                    }
                    4458 => pointers.push(words[at + 1]),
                    4459 => multiply += 1,
                    241 => atomic += 1,
                    _ => {}
                }
                at += n;
            }
            if producer {
                if multiply != 4 * M_FRAGMENTS || pointers.len() != 2 * M_FRAGMENTS || atomic != 0 {
                    return Err("direct matrix original products or publication count".into());
                }
                for mut p in pointers {
                    let mut seen = std::collections::BTreeSet::new();
                    while let Some(&parent) = parents.get(&p) {
                        if !seen.insert(p) {
                            return Err("direct pointer cycle".into());
                        }
                        p = parent;
                    }
                    if classes.get(&p) != Some(&12) {
                        return Err(
                            "direct cooperative store must target genuine StorageBuffer".into()
                        );
                    }
                }
            } else if multiply != 0 || atomic != 1 {
                return Err("complementary finite writer must retain actual sticky fault".into());
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
                .ok_or("direct validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("direct composite actual SPIRV invalid".into());
            }
            println!(
                "strata_dense_store_shader_accepted name={name} producer={producer} layout_mask={mask}"
            );
        }
    }
    Ok(())
}
/// Compiled-public independent comparator; retains the original shared writer.
#[allow(clippy::too_many_arguments)]
pub(super) fn shared_reference(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<()> {
    check_bounds(
        &h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
    )?;
    let gr = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    if !selected(
        count,
        columns,
        if gr { 384 } else { rows },
        weight_first,
        SHARED_BYTES,
        THREADS,
        (THREADS, 1, 1),
        16,
        32,
    ) {
        return Err(Error::InvalidBufferLength);
    }
    let [input, weight, output, fault] = h;
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 8) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 8) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    unsafe {
        if bfloat {
            dense_wide::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                fa,
                columns,
                weight_first,
                out_first,
                out_stride,
                gr,
            );
        } else {
            dense_wide::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static((rows / 128) as u32, (count / 128) as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                fa,
                columns,
                weight_first,
                out_first,
                out_stride,
                false,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod direct_tests {
    use super::*;
    #[test]
    fn all_global_fragments_and_balanced_sanitizer_grids_cover_only_logical_outputs() {
        // The genuine SG/lane staging map covers each original local-ID job
        // exactly once; no assumption about physical local-ID subgroup order.
        let mut jobs = vec![0u8; STAGE_ROUNDS * STAGE_THREADS];
        for round in 0..STAGE_ROUNDS {
            for sg in 0..SUBGROUPS {
                for lane in 0..16 {
                    jobs[sg * 16 + lane + round * STAGE_THREADS] += 1;
                }
            }
        }
        assert!(jobs.iter().all(|&n| n == 1));
        for (count, rows, first, stride) in [
            (512usize, 128usize, 0usize, 128usize),
            (512, 128, 4, 132),
            (640, 320, 0, 320),
            (512, 10240, 0, 10240),
        ] {
            let len = first + (count - 1) * stride + rows;
            let mut seen = vec![0u8; len + 64];
            for gy in 0..count / 128 {
                for gx in 0..rows.div_ceil(128) {
                    for sg in 0..SUBGROUPS {
                        for m in 0..M_FRAGMENTS {
                            for n in 0..2 {
                                let t0 = gy * 128 + sg / 4 * SUBGROUP_ROWS + m * 8;
                                let r0 = gx * 128 + sg % 4 * 32 + n * 16;
                                if r0 < rows {
                                    assert!(r0 + 16 <= rows);
                                    for t in t0..t0 + 8 {
                                        for r in r0..r0 + 16 {
                                            seen[first + t * stride + r] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (i, &n) in seen.iter().enumerate() {
                let rel = i.saturating_sub(first);
                let logical = i >= first && rel / stride < count && rel % stride < rows;
                assert_eq!(n, u8::from(logical));
            }
            let output = Handle::new(
                cubecl_common::stream_id::StreamId::current(),
                ((len + 64) * 4) as u64,
            );
            let (x, y) =
                direct_selection(&output, count, rows, first, stride, (65535, 65535, 1)).unwrap();
            let groups = (count * rows).div_ceil(256);
            assert!(x as usize * y as usize >= groups);
            assert!((x as usize * y as usize) - groups < y as usize);
            for g in 0..x as usize * y as usize {
                let gx = g % x as usize;
                let gy = g / x as usize;
                assert_eq!(gx + gy * x as usize, g);
                if g < groups {
                    let i = g * 256;
                    assert!(i < count * rows);
                    let at = first + (i / rows) * stride + i % rows;
                    assert!(seen[at] == 1);
                }
            }
        }
        let h = Handle::new(cubecl_common::stream_id::StreamId::current(), 1024 * 1024);
        assert!(direct_selection(&h, 4096, 10240, 0, 10240, (65535, 65535, 1)).is_some());
        assert!(direct_selection(&h, 512, 128, 1, 128, (65535, 65535, 1)).is_none());
        assert!(direct_selection(&h, 512, 128, 0, 131, (65535, 65535, 1)).is_none());
        assert!(
            direct_selection(
                &h.clone().offset_start(4),
                512,
                128,
                0,
                128,
                (65535, 65535, 1)
            )
            .is_none()
        );
        assert!(direct_selection(&h, 4096, 10240, 0, 10240, (1, 1, 1)).is_none());
    }
    #[test]
    fn whole_original_operand_and_accumulator_prefix_remains_token_identical() {
        let old = include_str!("strata_matrix_native_wide.rs");
        let new = include_str!("strata_matrix_native_direct_store.rs");
        let old = &old[old.find("fn dense_wide<F: Float>").unwrap()..];
        let new = &new[new.find("fn dense_wide_store<F: Float>").unwrap()..];
        let old = &old[old.find("    let mut a =").unwrap()..old.find("    publish(\n").unwrap()];
        let new =
            &new[new.find("    let mut a =").unwrap()..new.find("    publish_global(\n").unwrap()];
        let original = old.replace(
            "    let mut scratch = SharedMemory::<f32>::new(comptime!(SUBGROUPS * 128));\n",
            "",
        );
        assert_eq!(
            original.split_whitespace().collect::<Vec<_>>(),
            new.replace("direct_stage::<F>(", "stage::<F>(")
                .split_whitespace()
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn actual_composite_modules_keep_original_products_global_stores_and_complementary_fault_writer()
     {
        validate_direct_sources().unwrap();
    }
}
