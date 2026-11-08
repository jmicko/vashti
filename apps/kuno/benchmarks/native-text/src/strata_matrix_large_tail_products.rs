//! Original residual-tail arithmetic; packed input and GU bases are explicit.
use super::*;
#[cube]
fn macro_stage<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        a: &mut SharedMemory<Vector<F, Const<4>>>,
        b: &mut SharedMemory<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        input_prefix: usize,
        token_base: usize,
        row_base: usize,
        k: usize,
        buffer: usize,
    ) {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + 32 <= count && k + 64 <= columns {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = ((input_prefix + token) * columns + col) / 4;
                a[buffer * 512 + job * 2] = input[source];
                a[buffer * 512 + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = ((input_prefix + token) * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * 512 + job * 2] = lo;
                a[buffer * 512 + job * 2 + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 64 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[buffer * 1024 + job * 2] = weight[source];
                b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
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
#[cube]
fn narrow_stage<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        a: &mut SharedMemory<Vector<F, Const<4>>>,
        b: &mut SharedMemory<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        input_prefix: usize,
        token_base: usize,
        row_base: usize,
        k: usize,
        buffer: usize,
        #[comptime] tile: usize,
    ) {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + tile <= count && k + 64 <= columns {
            #[unroll]
            for round in 0..comptime!(tile / 8) {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = ((input_prefix + token) * columns + col) / 4;
                a[buffer * comptime!(tile * 16) + job * 2] = input[source];
                a[buffer * comptime!(tile * 16) + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..comptime!(tile / 8) {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = ((input_prefix + token) * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * comptime!(tile * 16) + job * 2] = lo;
                a[buffer * comptime!(tile * 16) + job * 2 + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 64 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[buffer * 1024 + job * 2] = weight[source];
                b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
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
    packed_prefix: usize,
    input_prefix: usize,
    tile_index: usize,
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
    let token_base = tile_index * 32;
    let row_base = usize::cast_from(CUBE_POS_X) * 64;
    // Buffer0 is fully initialized, including every masked tail, before
    // any subgroup reads it. Each lane copies eight consecutive half
    // values as two genuine vec4 loads/stores without float conversion.
    macro_stage::<F>(
        input,
        weight,
        &mut a,
        &mut b,
        columns,
        rows,
        count,
        weight_first,
        input_prefix + member_base,
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
            macro_stage::<F>(
                input,
                weight,
                &mut a,
                &mut b,
                columns,
                rows,
                count,
                weight_first,
                input_prefix + member_base,
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
        packed_prefix,
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
        packed_prefix,
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
        packed_prefix,
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
        packed_prefix,
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
    packed_prefix: usize,
    input_prefix: usize,
    tile_index: usize,
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
    let token_base = tile_index * tile;
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
                (input_prefix + member_base + token_base) * columns / 4,
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
                let af = ((input_prefix + member_base + token_base) * columns + k) / 4;
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
        narrow_stage::<F>(
            input,
            weight,
            &mut a,
            &mut b,
            columns,
            rows,
            count,
            weight_first,
            input_prefix + member_base,
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
                narrow_stage::<F>(
                    input,
                    weight,
                    &mut a,
                    &mut b,
                    columns,
                    rows,
                    count,
                    weight_first,
                    input_prefix + member_base,
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
        packed_prefix,
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
            packed_prefix,
            group,
            down,
        );
    }
}
