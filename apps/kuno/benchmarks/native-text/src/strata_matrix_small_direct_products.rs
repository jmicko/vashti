//! Original arithmetic bodies with only final publication replaced.
use super::*;
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn macro_product<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    group: usize,
    member_base: usize,
    total_count: usize,
    #[comptime] down: bool,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
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
    let token_base = usize::cast_from(CUBE_POS_Y) * 32;
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    // Buffer0 is fully initialized, including every masked tail, before
    // any subgroup reads it. Each lane copies eight consecutive half
    // values as two genuine vec4 loads/stores without float conversion.
    crate::strata_matrix::native::macro_stage::<F>(
        input,
        weight,
        &mut a,
        &mut b,
        columns,
        rows,
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
            crate::strata_matrix::native::macro_stage::<F>(
                input,
                weight,
                &mut a,
                &mut b,
                columns,
                rows,
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
                let af = current * 512 + (subgroup / 2) * 256 + inner / 4;
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
    let mut scratch = SharedMemory::<f32>::new(4usize * 128);
    let member0 = member_base + token_base + (subgroup / 2) * 16;
    let row0 = row_base + (subgroup % 2) * 32;
    crate::strata_matrix::small_direct::publish(
        &c0,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0,
        row0,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c1,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0,
        row0 + 16,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c2,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0 + 8,
        row0,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c3,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0 + 8,
        row0 + 16,
        total_count,
        0,
        group,
        down,
    );
}
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn macro64_product<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,

    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    group: usize,
    member_base: usize,
    total_count: usize,
    #[comptime] down: bool,
) {
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
    crate::strata_matrix::native::macro64_stage::<F>(
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
            crate::strata_matrix::native::macro64_stage::<F>(
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
    let mut scratch = SharedMemory::<f32>::new(8usize * 128);
    let member0 = member_base + token_base + (subgroup / 2) * 16;
    let row0 = row_base + (subgroup % 2) * 32;
    crate::strata_matrix::small_direct::publish(
        &c0,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0,
        row0,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c1,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0,
        row0 + 16,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c2,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0 + 8,
        row0,
        total_count,
        0,
        group,
        down,
    );
    crate::strata_matrix::small_direct::publish(
        &c3,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0 + 8,
        row0 + 16,
        total_count,
        0,
        group,
        down,
    );
}
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn narrow_product<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    columns: usize,
    rows: usize,
    count: usize,
    weight_first: usize,
    #[comptime] tile: usize,
    #[comptime] direct: bool,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    group: usize,
    member_base: usize,
    total_count: usize,
    #[comptime] down: bool,
) {
    // SAFETY: the complete current operand stage is initialized before
    // every product, including zero-masked token/feature/reduction tails.
    let a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let a1 = if comptime!(tile == 16) {
        // SAFETY: only the16-row variant loads and uses this fragment.
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
    let bm = unsafe {
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
    let c1 = if comptime!(tile == 16) {
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
    let subgroup = usize::cast_from(UNIT_POS) / 16;
    let token_base = usize::cast_from(CUBE_POS_Y) * tile;
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    if comptime!(direct) {
        // Authentic Gather AND activation zero every inactive row in the
        // complete8/16-member tile. Global vector views retain full owners;
        // both fixed expert feature axes are complete64-wide tiles.
        if comptime!(cfg!(feature = "strata-expert-register-prefetch")) {
            crate::strata_matrix::register_prefetch::accumulate::<F>(
                input,
                weight,
                columns,
                token_base * columns / 4,
                (weight_first + (row_base + subgroup * 16) * columns) / 4,
                &a0,
                &a1,
                &bm,
                &c0,
                &c1,
                tile,
            );
        } else {
            let mut k = 0usize;
            while k < columns {
                let af = (token_base * columns + k) / 4;
                let bf = (weight_first + (row_base + subgroup * 16) * columns + k) / 4;
                cmma::load(
                    &a0,
                    &input.to_slice().slice(af, input.len()),
                    u32::cast_from(columns),
                );
                cmma::load(
                    &bm,
                    &weight.to_slice().slice(bf, weight.len()),
                    u32::cast_from(columns),
                );
                cmma::execute::<F, F, f32, f32>(&a0, &bm, &c0, &c0);
                if comptime!(tile == 16) {
                    let af1 = af + 8 * columns / 4;
                    cmma::load(
                        &a1,
                        &input.to_slice().slice(af1, input.len()),
                        u32::cast_from(columns),
                    );
                    cmma::execute::<F, F, f32, f32>(&a1, &bm, &c1, &c1);
                }
                k += 16;
            }
        }
    } else {
        let mut a = SharedMemory::<Vector<F, Const<4>>>::new(comptime!(tile * 32));
        let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
        crate::strata_matrix::native::narrow_stage::<F>(
            input,
            weight,
            &mut a,
            &mut b,
            columns,
            rows,
            count,
            weight_first,
            token_base,
            row_base,
            0,
            0,
            tile,
        );
        sync_cube();
        let mut k = 0usize;
        let mut current = 0usize;
        while k < columns {
            let next = 1usize - current;
            if k + 64 < columns {
                crate::strata_matrix::native::narrow_stage::<F>(
                    input,
                    weight,
                    &mut a,
                    &mut b,
                    columns,
                    rows,
                    count,
                    weight_first,
                    token_base,
                    row_base,
                    k + 64,
                    next,
                    tile,
                );
            }
            #[unroll]
            for step in 0..4usize {
                let inner = step * 16;
                if k + inner < columns {
                    let af = current * comptime!(tile * 16) + inner / 4;
                    let bf = current * 1024 + subgroup * 256 + inner / 4;
                    cmma::load(&a0, &a.to_slice().slice(af, a.len()), 64);
                    cmma::load(&bm, &b.to_slice().slice(bf, b.len()), 64);
                    cmma::execute::<F, F, f32, f32>(&a0, &bm, &c0, &c0);
                    if comptime!(tile == 16) {
                        cmma::load(&a1, &a.to_slice().slice(af + 128, a.len()), 64);
                        cmma::execute::<F, F, f32, f32>(&a1, &bm, &c1, &c1);
                    }
                }
            }
            // A single uniform handoff perK64 replaces the scalar partial
            // path's two workgroup barriers perK16. Old readers finish before
            // stage reuse; every next-stage store is visible before the switch.
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
    }
    let mut scratch = SharedMemory::<f32>::new(4usize * 128);
    let member0 = member_base + token_base;
    let row0 = row_base + subgroup * 16;
    crate::strata_matrix::small_direct::publish(
        &c0,
        &mut scratch,
        output,
        map,
        status,
        fault,
        member0,
        row0,
        total_count,
        0,
        group,
        down,
    );
    if comptime!(tile == 16) {
        crate::strata_matrix::small_direct::publish(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            member0 + 8,
            row0,
            total_count,
            0,
            group,
            down,
        );
    }
}
