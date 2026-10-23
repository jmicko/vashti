// Coupled pitch40 operand staging and matching cooperative readers.
// Included within native_wide; original compact producer and finite consumer retained.
#[cube]
#[allow(clippy::too_many_arguments)]
fn slm40_stage<F: Float>(
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
        a[buffer * 1280 + row * 10 + (job % 4) * 2] = input[source_a];
        a[buffer * 1280 + row * 10 + (job % 4) * 2 + 1] = input[source_a + 1];
        // Every padded word is initialized by exactly one real staging job.
        if job % 4 == 0 {
            a[buffer * 1280 + row * 10 + 8] = Vector::<F, Const<4>>::new(F::new(0.0));
            a[buffer * 1280 + row * 10 + 9] = Vector::<F, Const<4>>::new(F::new(0.0));
            b[buffer * 1280 + row * 10 + 8] = Vector::<F, Const<4>>::new(F::new(0.0));
            b[buffer * 1280 + row * 10 + 9] = Vector::<F, Const<4>>::new(F::new(0.0));
        }
        if comptime!(gr_down) {
            // The sole extra shape is the authentic320-row GR down image.
            // Do not read the missing last64 rows, even from a larger owner.
            let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
            let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
            if row_base + row < 320 {
                lo = weight[source_b];
                hi = weight[source_b + 1];
            }
            b[buffer * 1280 + row * 10 + (job % 4) * 2] = lo;
            b[buffer * 1280 + row * 10 + (job % 4) * 2 + 1] = hi;
        } else {
            b[buffer * 1280 + row * 10 + (job % 4) * 2] = weight[source_b];
            b[buffer * 1280 + row * 10 + (job % 4) * 2 + 1] = weight[source_b + 1];
        }
    }
}

#[cube(launch_unchecked)]
fn dense_slm40_store<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] gr_down: bool,
) {
    // Same source operands and chronological products, physical pitch40. No publication
    // scratch allocation: cooperative stores target the actual output owner.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2560usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2560usize);
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
    slm40_stage::<F>(
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
            slm40_stage::<F>(
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
            let af = current * 1280 + (subgroup / 4) * comptime!(SUBGROUP_ROWS * 10) + step * 4;
            let bf = current * 1280 + (subgroup % 4) * 320 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 40);
            cmma::load(&a1, &a.to_slice().slice(af + 80, a.len()), 40);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::load(&a2, &a.to_slice().slice(af + 160, a.len()), 40);
                cmma::load(&a3, &a.to_slice().slice(af + 240, a.len()), 40);
            }
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 40);
            cmma::load(&b1, &b.to_slice().slice(bf + 160, b.len()), 40);
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

#[allow(clippy::too_many_arguments)]
fn run_slm40_store(
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
            dense_slm40_store::launch_unchecked::<bf16, WgpuRuntime>(
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
            dense_slm40_store::launch_unchecked::<f16, WgpuRuntime>(
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
    crate::strata_prefill_profile::count("dense_slm40_products", 1);
    Ok(())
}
fn slm40_definition<F: Float>(gr_down: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let a: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_slm40_store::expand::<F>(
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
pub(super) fn validate_slm40_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8 {
        for (name, def, producer) in [
            ("dense_slm40_fp16", slm40_definition::<f16>(false), true),
            ("dense_slm40_bf16", slm40_definition::<bf16>(false), true),
            ("dense_slm40_gr_bf16", slm40_definition::<bf16>(true), true),
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
            if kernel.shared_size != if producer { 40960 } else { 0 } {
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
                "strata_dense_slm40_shader_accepted name={name} producer={producer} layout_mask={mask}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod slm40_tests {
    use super::*;
    #[test]
    fn both_real_buffers_and_each_sg_fragment_preserve_original_operands() {
        for gr in [false, true] {
            let mut seen = vec![0u8; 10240];
            let mut source = vec![None; 10240];
            let mut input_source = vec![None; 10240];
            for buffer in 0..2 {
                for rowbase in [0, 256] {
                    seen.fill(0);
                    source.fill(None);
                    input_source.fill(None);
                    for sg in 0..SUBGROUPS {
                        for lane in 0..16 {
                            for round in 0..STAGE_ROUNDS {
                                let job = sg * 16 + lane + round * STAGE_THREADS;
                                let row = job / 4;
                                let col = (job % 4) * 8;
                                for j in 0..8 {
                                    let at = buffer * 5120 + row * 40 + col + j;
                                    seen[at] += 1;
                                    input_source[at] = Some(row * 32 + col + j);
                                    source[at] = Some(if !gr || rowbase + row < 320 {
                                        row * 32 + col + j
                                    } else {
                                        usize::MAX
                                    });
                                }
                                if job % 4 == 0 {
                                    for j in 32..40 {
                                        seen[buffer * 5120 + row * 40 + j] += 1;
                                    }
                                }
                            }
                        }
                    }
                    for row in 0..128 {
                        for col in 0..40 {
                            let at = buffer * 5120 + row * 40 + col;
                            assert_eq!(seen[at], 1);
                            if col < 32 {
                                assert_eq!(input_source[at], Some(row * 32 + col));
                                assert_eq!(
                                    source[at],
                                    Some(if !gr || rowbase + row < 320 {
                                        row * 32 + col
                                    } else {
                                        usize::MAX
                                    })
                                );
                            } else {
                                assert_eq!(source[at], None);
                                assert_eq!(input_source[at], None);
                            }
                        }
                    }
                    for sg in 0..SUBGROUPS {
                        for step in 0..2 {
                            for m in 0..M_FRAGMENTS {
                                for arow in 0..8 {
                                    for col in 0..16 {
                                        let row = sg / 4 * SUBGROUP_ROWS + m * 8 + arow;
                                        let k = step * 16 + col;
                                        assert_eq!(
                                            input_source[buffer * 5120 + row * 40 + k],
                                            Some(row * 32 + k)
                                        );
                                    }
                                }
                            }
                            for n in 0..2 {
                                for brow in 0..16 {
                                    for col in 0..16 {
                                        let row = sg % 4 * 32 + n * 16 + brow;
                                        let k = step * 16 + col;
                                        assert_eq!(
                                            source[buffer * 5120 + row * 40 + k],
                                            Some(if !gr || rowbase + row < 320 {
                                                row * 32 + k
                                            } else {
                                                usize::MAX
                                            })
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(2 * 2 * 128 * 40 * 2, 40960);
        assert!(40960 <= SHARED_BYTES);
    }
    #[test]
    fn chronological_products_prefetch_and_publication_remain_original() {
        let old = include_str!("strata_matrix_native_direct_store.rs");
        let new = include_str!("strata_matrix_dense_slm40.rs");
        let old = &old[old.find("fn dense_wide_store<F: Float>").unwrap()
            ..old.find("fn dense_finite(").unwrap()];
        let new = &new[new.find("fn dense_slm40_store<F: Float>").unwrap()
            ..new.find("fn run_slm40_store(").unwrap()];
        let products = |s: &str| {
            s.lines()
                .filter(|l| l.contains("cmma::execute::<"))
                .map(str::trim)
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(products(old), products(new));
        for s in [old, new] {
            assert!(s.contains("for step in 0..2usize"));
            assert!(s.contains("if k + 32 < columns"));
            assert!(s.contains("k += 32;"));
            assert_eq!(s.matches("sync_cube();").count(), 2);
            assert_eq!(s.matches("publish_global(").count(), 8);
        }
    }
    #[test]
    fn actual_padded_modules_keep_real_envelope_global_stores_and_original_products() {
        validate_slm40_sources().unwrap();
    }
}
