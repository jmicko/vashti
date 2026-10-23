// Deep64x64 expert macro: original K16 products in double K128 operand stages.
use super::*;
pub(crate) const OPERAND_BYTES: usize = 65536;
pub(crate) const LOGICAL_BYTES: usize = OPERAND_BYTES + 16384;

#[allow(clippy::too_many_arguments)]
pub(crate) fn geometry(
    count: usize,
    rows: usize,
    columns: usize,
    weight_first: usize,
    shared: usize,
    bindings: [(u64, u64); 2],
    explicit: bool,
) -> bool {
    explicit
        && shared >= OPERAND_BYTES
        && super::macro64_geometry(count, rows, columns, weight_first, shared, bindings)
}

pub(crate) fn selected(
    client: &ComputeClient<WgpuRuntime>,
    count: usize,
    rows: usize,
    columns: usize,
    weight_first: usize,
    bindings: [(u64, u64); 2],
) -> bool {
    // Resolve the actual client/compiler capability ONLY after the original
    // shape, source alignment and the full new64KiB physical envelope fit.
    let shared = client.properties().hardware.max_shared_memory_size;
    shared >= OPERAND_BYTES
        && super::macro64_geometry(count, rows, columns, weight_first, shared, bindings)
        && geometry(
            count,
            rows,
            columns,
            weight_first,
            shared,
            bindings,
            client.native_explicit_shared_layout(),
        )
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn macro64_stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    token_base: usize,
    row_base: usize,
    k: usize,
    buffer: usize,
) {
    if comptime!(cfg!(feature = "strata-macro64-static-geometry")) {
        // The authenticated entry has complete64-wide N/128-wide K axes and
        // rejects out-of-grid X before any stage. Preserve the M tail.
        if token_base + 64 <= count {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let token = token_base + job / 16;
                let col = k + (job % 16) * 8;
                let source = (token * columns + col) / 4;
                a[target] = input[source];
                a[target + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let token = token_base + job / 16;
                let col = k + (job % 16) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count {
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[target] = lo;
                a[target + 1] = hi;
            }
        }
        #[unroll]
        for round in 0..8usize {
            let job = usize::cast_from(UNIT_POS) + round * 128;
            let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
            let row = row_base + job / 16;
            let col = k + (job % 16) * 8;
            let source = (weight_first + row * columns + col) / 4;
            b[target] = weight[source];
            b[target + 1] = weight[source + 1];
        }
    } else {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + 64 <= count && k + 128 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let token = token_base + job / 16;
                let col = k + (job % 16) * 8;
                let source = (token * columns + col) / 4;
                a[target] = input[source];
                a[target + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let token = token_base + job / 16;
                let col = k + (job % 16) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[target] = lo;
                a[target + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 128 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let row = row_base + job / 16;
                let col = k + (job % 16) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[target] = weight[source];
                b[target + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let target = buffer * 2048 + (job % 16 / 2) * 256 + (job / 16) * 4 + (job % 2) * 2;
                let row = row_base + job / 16;
                let col = k + (job % 16) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if row < rows && col < columns {
                    let source = (weight_first + row * columns + col) / 4;
                    lo = weight[source];
                    hi = weight[source + 1];
                }
                b[target] = lo;
                b[target + 1] = hi;
            }
        }
    }
}

#[cube]
pub(super) fn macro64_product<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    #[comptime] down: bool,
) -> SharedMemory<f32> {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(4096usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(4096usize);
    // SAFETY: both operands are loaded before the corresponding products.
    let a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: both operands are loaded before the corresponding products.
    let a1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: both operands are loaded before the corresponding products.
    let b0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    // SAFETY: both operands are loaded before the corresponding products.
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
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let token_base = usize::cast_from(CUBE_POS_Y) * 64;
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    let stage_columns = if comptime!(cfg!(feature = "strata-macro64-static-geometry")) {
        usize::cast_from(comptime!(if down { 640usize } else { 2560usize }))
    } else {
        columns
    };
    let stage_rows = if comptime!(cfg!(feature = "strata-macro64-static-geometry")) {
        usize::cast_from(comptime!(if down { 2560usize } else { 1280usize }))
    } else {
        rows
    };
    // Buffer0 is fully initialized, including every masked tail, before
    // any subgroup reads it. Each lane copies original half
    // values through eight pairs of genuine vec4 loads/stores without float conversion.
    macro64_stage::<F>(
        input,
        weight,
        &mut a,
        &mut b,
        stage_columns,
        stage_rows,
        count,
        weight_first,
        token_base,
        row_base,
        0,
        0,
    );
    sync_cube();
    let mut k = 0usize;
    let mut current = 0usize;
    while k < columns {
        let next = 1usize - current;
        // Software prefetch: independent next-stage global loads/stores
        // are issued before current-stage MMA. This expresses overlap
        // opportunity, not a guaranteed asynchronous DMA operation.
        if k + 128 < columns {
            macro64_stage::<F>(
                input,
                weight,
                &mut a,
                &mut b,
                stage_columns,
                stage_rows,
                count,
                weight_first,
                token_base,
                row_base,
                k + 128,
                next,
            );
        }
        // Expose all eight ordered K16 groups together. This keeps the
        // same four accumulators and exact reduction order, but avoids
        // a runtime loop around the native matrix instructions.
        #[unroll]
        for step in 0..8usize {
            let inner = step * 16;
            if k + inner < columns {
                let af = current * 2048 + step * 256 + (subgroup / 2) * 64;
                let bf = current * 2048 + step * 256 + (subgroup % 2) * 128;
                // Offsets index real vec4 storage; cmma's stride remains
                // the16 scalar elements in each physical K16-panel row. The compiler
                // divides by the authentic vector width for the pointer.
                cmma::load(&a0, &a.to_slice().slice(af, a.len()), 16);
                cmma::load(&a1, &a.to_slice().slice(af + 32, a.len()), 16);
                cmma::load(&b0, &b.to_slice().slice(bf, b.len()), 16);
                cmma::load(&b1, &b.to_slice().slice(bf + 64, b.len()), 16);
                cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
                cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
                cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
                cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
            }
        }
        // All old readers finish before either stage is reused. Every
        // next-stage write is visible before it becomes current. Loop
        // conditions and the barrier are uniform across the workgroup.
        sync_cube();
        current = next;
        k += 128;
    }
    // Every lane has completed the final K128 handoff barrier, and no
    // operand slice is used beyond this uniform loop exit. Matrix fragments
    // retain their values in registers. End the authentic SLM lifetimes
    // before allocating output; the compiler may reuse physical offsets
    // only on devices supporting explicit workgroup memory layout.
    // SAFETY: uniform control flow, no surviving operand-pointer uses.
    unsafe {
        a.free();
        b.free();
    }
    let mut out = SharedMemory::<f32>::new(4096usize);
    cmma::store(
        &mut out.slice_mut(subgroup * 512, subgroup * 512 + 128),
        &c0,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        &mut out.slice_mut(subgroup * 512 + 128, subgroup * 512 + 256),
        &c1,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        &mut out.slice_mut(subgroup * 512 + 256, subgroup * 512 + 384),
        &c2,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    cmma::store(
        &mut out.slice_mut(subgroup * 512 + 384, subgroup * 512 + 512),
        &c3,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    out
}
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
                    base = route * 2560;
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
pub(super) fn definition(down: bool) -> KernelDefinition {
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

fn compile(down: bool, mask: u8) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, WgpuCompilationOptions};
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
        definition(down),
        &options,
        ExecutionMode::Unchecked,
        cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
    )?
    else {
        return Err("deep macro compiler representation".into());
    };
    if kernel.shared_size
        != if mask & 4 != 0 {
            OPERAND_BYTES
        } else {
            LOGICAL_BYTES
        }
    {
        return Err("deep macro actual physical/logical shared layout".into());
    }
    let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
    let words =
        wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?.unwrap_or(words);
    let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
    if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
        return Err("deep macro original subgroup".into());
    }
    let (mut at, mut products, mut loads, mut stores, mut barriers, mut faults, mut local) =
        (5, 0, 0, 0, 0, 0, false);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err("deep macro instruction extent".into());
        }
        let i = &words[at..at + n];
        match i[0] & 65535 {
            4459 => products += 1,
            4457 => loads += 1,
            4458 => stores += 1,
            224 => barriers += 1,
            241 => faults += 1,
            16 if i.len() == 6 && i[2] == 17 => {
                local = i[3..] == [128, 1, 1];
            }
            _ => {}
        }
        at += n;
    }
    if products != 32 || loads != 32 || stores != 4 || barriers != 3 || faults == 0 || !local {
        return Err("deep macro actual chronological products/owners/publication/geometry".into());
    }
    Ok(words)
}

pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    for mask in 0..8 {
        for down in [false, true] {
            let words = compile(down, mask)?;
            let mut validator = std::process::Command::new("spirv-val");
            validator.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                validator.arg("--uniform-buffer-standard-layout");
            }
            let mut command = validator
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            command
                .stdin
                .take()
                .ok_or("deep macro validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !command.wait()?.success() {
                return Err("deep macro SPIRV validation".into());
            }
            println!("strata_deep_macro_shader_accepted down={down} layout_mask={mask}");
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "strata_matrix_macro64_deep_tests.rs"]
mod tests;
