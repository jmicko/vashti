// Genuine64x64 expert macro tile:8 subgroups, orderedK16, original routed stores.
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
        // The authenticated entry has fixed complete64-wide N/K axes and
        // rejects out-of-grid X before any stage. Preserve the M tail.
        if token_base + 64 <= count {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (token * columns + col) / 4;
                a[buffer * 1024 + job * 2] = input[source];
                a[buffer * 1024 + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count {
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * 1024 + job * 2] = lo;
                a[buffer * 1024 + job * 2 + 1] = hi;
            }
        }
        #[unroll]
        for round in 0..4usize {
            let job = usize::cast_from(UNIT_POS) + round * 128;
            let row = row_base + job / 8;
            let col = k + (job % 8) * 8;
            let source = (weight_first + row * columns + col) / 4;
            b[buffer * 1024 + job * 2] = weight[source];
            b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
        }
    } else {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + 64 <= count && k + 64 <= columns {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (token * columns + col) / 4;
                a[buffer * 1024 + job * 2] = input[source];
                a[buffer * 1024 + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * 1024 + job * 2] = lo;
                a[buffer * 1024 + job * 2 + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 64 <= columns {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[buffer * 1024 + job * 2] = weight[source];
                b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 128;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if row < rows && col < columns {
                    let source = (weight_first + row * columns + col) / 4;
                    lo = weight[source];
                    hi = weight[source + 1];
                }
                b[buffer * 1024 + job * 2] = lo;
                b[buffer * 1024 + job * 2 + 1] = hi;
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
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
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
    // any subgroup reads it. Each lane copies eight consecutive half
    // values as two genuine vec4 loads/stores without float conversion.
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
        if k + 64 < columns {
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
                k + 64,
                next,
            );
        }
        // Expose all four ordered K16 groups together. This keeps the
        // same four accumulators and exact reduction order, but avoids
        // a runtime loop around the native matrix instructions.
        #[unroll]
        for step in 0..4usize {
            let inner = step * 16;
            if k + inner < columns {
                let af = current * 1024 + (subgroup / 2) * 256 + inner / 4;
                let bf = current * 1024 + (subgroup % 2) * 512 + inner / 4;
                // Offsets index real vec4 storage; cmma's stride remains
                // the64 scalar elements in each staged row. The compiler
                // divides by the authentic vector width for the pointer.
                cmma::load(&a0, &a.to_slice().slice(af, a.len()), 64);
                cmma::load(&a1, &a.to_slice().slice(af + 128, a.len()), 64);
                cmma::load(&b0, &b.to_slice().slice(bf, b.len()), 64);
                cmma::load(&b1, &b.to_slice().slice(bf + 256, b.len()), 64);
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
        k += 64;
    }
    // Every lane has completed the final K64 handoff barrier, and no
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

#[cfg(test)]
mod macro64_tests {
    use super::*;
    #[test]
    fn authenticated_fixed_axes_cover_every_full_weight_vector_and_reject_surplus_x() {
        for (cols, rows) in [(2560usize, 1280usize), (640, 2560)] {
            assert_eq!(cols % 64, 0);
            assert_eq!(rows % 64, 0);
            for x in 0..rows / 64 + 2 {
                for k in (0..cols).step_by(64) {
                    let admitted = x < rows / 64;
                    assert_eq!(admitted, x * 64 + 64 <= rows && k + 64 <= cols);
                    if !admitted {
                        continue;
                    }
                    for first in [0usize, 4, 128] {
                        for tid in 0..128 {
                            for round in 0..4 {
                                for vector in 0..2 {
                                    let job = tid + 128 * round;
                                    let row = x * 64 + job / 8;
                                    let col = k + (job % 8) * 8 + vector * 4;
                                    let original = (first + row * cols + col) / 4;
                                    assert!(
                                        original * 4 >= first
                                            && original * 4 + 4 <= first + rows * cols
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        let text = include_str!("strata_matrix_macro64.rs");
        let entry = text.find("    if !header_good").unwrap();
        let clip = text
            .find("if usize::cast_from(CUBE_POS_X) >= rows / 64")
            .unwrap();
        let product = text.find("    let values = macro64_product").unwrap();
        assert!(entry < clip && clip < product);
        assert!(text[entry..clip].contains("columns != comptime!(if down { 640 } else { 2560 })"));
        assert!(text[entry..clip].contains("rows != comptime!(if down { 2560 } else { 1280 })"));
    }
    #[test]
    fn fixed_k_tiles_preserve_every_masked_input_member_and_original_prefetch_boundary() {
        for cols in [640usize, 2560] {
            for k in (0..cols).step_by(64) {
                assert!(k + 64 <= cols);
                if k + 64 < cols {
                    assert!(k + 128 <= cols);
                }
                for count in 64usize..256 {
                    for base in (0..count).step_by(64) {
                        assert_eq!(base + 64 <= count && k + 64 <= cols, base + 64 <= count);
                        for job in 0..512 {
                            for vector in 0..2 {
                                let token = base + job / 8;
                                let col = k + (job % 8) * 8 + vector * 4;
                                assert_eq!(token < count && col < cols, token < count);
                                if token < count {
                                    let at = (token * cols + col) / 4;
                                    assert!(at * 4 + 4 <= count * cols);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn genuine_shape_binding_and_full_shared_memory_admission() {
        let bindings = [(80 * 1024 * 1024, 256); 2];
        for count in [64, 65, 127, 128, 129, 255] {
            for (rows, cols) in [(1280, 2560), (2560, 640)] {
                assert!(macro64_geometry(count, rows, cols, 4, 49152, bindings));
                assert!(!macro64_geometry(count, rows, cols, 4, 49151, bindings));
            }
        }
        for count in [0, 1, 32, 63, 256, 257, 4096] {
            assert!(!macro64_geometry(count, 1280, 2560, 4, 65536, bindings));
        }
        for (rows, cols, first) in [
            (1279, 2560, 4),
            (1280, 2559, 4),
            (2560, 641, 4),
            (1280, 2560, 1),
        ] {
            assert!(!macro64_geometry(129, rows, cols, first, 65536, bindings));
        }
        for bad in [(0, 256), (7, 256), (8, 2)] {
            for i in 0..2 {
                let mut b = bindings;
                b[i] = bad;
                assert!(!macro64_geometry(129, 1280, 2560, 4, 65536, b));
            }
        }
        assert_eq!(
            compiled_shared_bytes("expert_macro64_gu", false),
            Some(49152)
        );
        assert_eq!(
            compiled_shared_bytes("expert_macro64_down", true),
            Some(32768)
        );
        assert_eq!(compiled_shared_bytes("expert_macro_gu", false), Some(32768));
        assert_eq!(compiled_shared_bytes("expert_macro_gu", true), Some(24576));
    }
    #[test]
    fn all_real_stage_jobs_and_masked_tail_reads_have_exact_unique_owners() {
        for count in [64usize, 65, 127, 128, 129, 255] {
            for cols in [640usize, 2560] {
                for token_base in (0..count).step_by(64) {
                    for k in (0..cols).step_by(64) {
                        let mut a = vec![None; 2048];
                        let mut b = vec![None; 2048];
                        for bank in 0..2 {
                            let mut writes = vec![0; 1024];
                            for tid in 0..128 {
                                for round in 0..4 {
                                    let job = tid + round * 128;
                                    let row = job / 8;
                                    let col = (job % 8) * 8;
                                    for vector in 0..2 {
                                        let slot = job * 2 + vector;
                                        writes[slot] += 1;
                                        a[bank * 1024 + slot] = if token_base + row < count {
                                            Some(
                                                ((token_base + row) * cols + k + col + vector * 4)
                                                    / 4,
                                            )
                                        } else {
                                            None
                                        };
                                        b[bank * 1024 + slot] =
                                            Some((row * cols + k + col + vector * 4) / 4);
                                    }
                                }
                            }
                            assert!(writes.iter().all(|&n| n == 1));
                            for sg in 0..8 {
                                for step in 0..4 {
                                    for tile in 0..2 {
                                        let af =
                                            bank * 1024 + (sg / 2) * 256 + step * 4 + tile * 128;
                                        for row in 0..8 {
                                            for v in 0..4 {
                                                let token =
                                                    token_base + (sg / 2) * 16 + tile * 8 + row;
                                                assert_eq!(
                                                    a[af + row * 16 + v],
                                                    if token < count {
                                                        Some(
                                                            (token * cols + k + step * 16 + v * 4)
                                                                / 4,
                                                        )
                                                    } else {
                                                        None
                                                    }
                                                );
                                            }
                                        }
                                        let bf =
                                            bank * 1024 + (sg % 2) * 512 + step * 4 + tile * 256;
                                        for row in 0..16 {
                                            for v in 0..4 {
                                                let feature = (sg % 2) * 32 + tile * 16 + row;
                                                assert_eq!(
                                                    b[bf + row * 16 + v],
                                                    Some(
                                                        (feature * cols + k + step * 16 + v * 4)
                                                            / 4
                                                    )
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn actual_output_lattice_and_chronological_k16_match_the_original_products() {
        let mut owners = vec![0; 64 * 64];
        for sg in 0..8 {
            for lane in 0..16 {
                for i in (lane..512).step_by(16) {
                    let m = (sg / 2) * 16 + (i / 256) * 8 + (i % 128) / 16;
                    let n = (sg % 2) * 32 + ((i / 128) % 2) * 16 + i % 16;
                    assert!(m < 64 && n < 64);
                    owners[m * 64 + n] += 1;
                }
            }
        }
        assert!(owners.iter().all(|&n| n == 1));
        for cols in [640, 2560] {
            let order: Vec<_> = (0..cols)
                .step_by(64)
                .flat_map(|k| (0..4).map(move |step| k + step * 16))
                .collect();
            assert_eq!(order, (0..cols).step_by(16).collect::<Vec<_>>());
        }
    }
    #[test]
    fn actual_native_compiler_keeps_128_threads_16_ordered_products_and_full_layouts() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        let name = core::any::type_name::<expert_macro64::ExpertMacro64<WgpuRuntime>>();
        for role in [3, 5] {
            assert!(
                cubecl_runtime::server::native_expert_batch::native_strata_expert_operator(
                    name, false, role
                )
            );
            assert!(
                !cubecl_runtime::server::native_expert_batch::native_strata_expert_operator(
                    "probe::expert_macro64::ExpertMacro64Fake<Runtime>",
                    false,
                    role
                )
            );
        }
        for down in [false, true] {
            for mask in 0..8 {
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
                    definition_expert_macro64(down),
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap() else {
                    panic!("actual SPIRV")
                };
                assert_eq!(k.shared_size, if mask & 4 != 0 { 32768 } else { 49152 });
                let w = &k.assembled_module;
                let mut at = 5;
                let mut products = 0;
                let mut barriers = 0;
                let mut local_size = false;
                while at < w.len() {
                    let n = (w[at] >> 16) as usize;
                    assert!(n > 0 && at + n <= w.len());
                    let i = &w[at..at + n];
                    match i[0] & 65535 {
                        4459 => products += 1,
                        224 => barriers += 1,
                        16 if i.len() == 6 && i[2] == 17 => {
                            assert_eq!(i[3..], [128, 1, 1]);
                            local_size = true;
                        }
                        _ => {}
                    }
                    at += n;
                }
                assert!(local_size);
                assert_eq!(products, 16);
                assert_eq!(barriers, 3);
            }
        }
    }
}
