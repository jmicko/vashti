// Combined genuine HC down320 and injection4, preserving original wide K16 products.
// Included within native_wide. All buffers are original distinct admitted owners.
const HC_DI_SHARED_BYTES: usize = 40960 + 4096;

#[cube]
#[allow(clippy::too_many_arguments)]
fn hc_di_publish(
    acc: &cmma::Matrix<f32>,
    output: &mut Array<f32>,
    injection: &mut Array<f32>,
    scratch: &mut SharedMemory<f32>,
    token_base: usize,
    row_base: usize,
    out_first: usize,
    out_stride: usize,
    inject_first: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
    #[comptime] gr_down: bool,
) {
    let sg = usize::cast_from(PLANE_POS);
    let token = token_base + (sg / 4) * comptime!(SUBGROUP_ROWS) + comptime!(tile_m * 8);
    let row = row_base + (sg % 4) * 32 + comptime!(tile_n * 16);
    if row < 320 {
        let first = out_first + token * out_stride + row;
        cmma::store(
            &mut output.slice_mut(first, output.len()),
            acc,
            u32::cast_from(out_stride),
            cmma::MatrixLayout::RowMajor,
        );
    } else if comptime!(tile_n == 0) {
        if row == 320 {
            // ONLY sg%4==2 and tile_n==0 can enter. Eight disjoint128-F32
            // slots are owned by sg/4. Both barriers are true SG16 barriers;
            // no divergent workgroup barrier or partial16-column global store.
            let slot = (sg / 4) * 128;
            cmma::store(
                &mut scratch.slice_mut(slot, slot + 128),
                acc,
                16,
                cmma::MatrixLayout::RowMajor,
            );
            sync_plane();
            #[unroll]
            for round in 0..8usize {
                let i = usize::cast_from(UNIT_POS_PLANE) + round * 16;
                if i % 16 < 4 {
                    injection[inject_first + (token + i / 16) * 4 + i % 16] = scratch[slot + i];
                }
            }
            sync_plane();
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn hc_di_stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    inject_weight: &Array<Vector<F, Const<4>>>,
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
            // Two original owners form324 real columns. The remaining60
            // columns are local zero padding, never an expanded weight view.
            let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
            let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
            if row_base + row < 320 {
                lo = weight[source_b];
                hi = weight[source_b + 1];
            } else if row_base + row < 324 {
                let source_i = ((row_base + row - 320) * columns + offset) / 4;
                lo = inject_weight[source_i];
                hi = inject_weight[source_i + 1];
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
fn hc_down_inject<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    inject_weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    injection: &mut Array<f32>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    inject_first: usize,
    #[comptime] gr_down: bool,
) {
    // Same source operands and chronological products, physical pitch40.
    // Down stores directly; only the genuine4-column injection uses4KiB SLM.
    let mut injection_scratch = SharedMemory::<f32>::new(1024usize);
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
    hc_di_stage::<F>(
        input,
        weight,
        inject_weight,
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
        // REGISTER_STAGE_LOAD_BEGIN: real global values stay live across
        // the current original matrix products; next SLM writes occur later.
        let mut next_a_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_a_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b_lo = Vector::<F, Const<4>>::new(F::new(0.0));
        let mut next_b_hi = Vector::<F, Const<4>>::new(F::new(0.0));
        if comptime!(cfg!(feature = "strata-wide-register-stage") && STAGE_ROUNDS == 1) {
            if k + 32 < columns {
                let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let row = job / 4;
                let off = k + 32 + (job % 4) * 8;
                let source_a = ((token_base + row) * columns + off) / 4;
                let source_b = (weight_first + (row_base + row) * columns + off) / 4;
                next_a_lo = input[source_a];
                next_a_hi = input[source_a + 1];
                if !comptime!(gr_down) || row_base + row < 320 {
                    next_b_lo = weight[source_b];
                    next_b_hi = weight[source_b + 1];
                } else if row_base + row < 324 {
                    let source_i = ((row_base + row - 320) * columns + off) / 4;
                    next_b_lo = inject_weight[source_i];
                    next_b_hi = inject_weight[source_i + 1];
                }
            }
        } else {
            // Same accepted immediate global-load→SLM-store software-prefetch ordering.
            // This expresses overlap opportunity, not guaranteed asynchronous execution.
            if k + 32 < columns {
                hc_di_stage::<F>(
                    input,
                    weight,
                    inject_weight,
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
        }
        // REGISTER_STAGE_LOAD_END
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
        // REGISTER_STAGE_STORE_BEGIN: only after all chronological current
        // CMMA products, publish the prefetched values into the disjoint bank.
        if comptime!(cfg!(feature = "strata-wide-register-stage") && STAGE_ROUNDS == 1) {
            if k + 32 < columns {
                let job = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let row = job / 4;
                let dest = next * 1280 + row * 10 + (job % 4) * 2;
                a[dest] = next_a_lo;
                a[dest + 1] = next_a_hi;
                b[dest] = next_b_lo;
                b[dest + 1] = next_b_hi;
                if job % 4 == 0 {
                    a[next * 1280 + row * 10 + 8] = Vector::<F, Const<4>>::new(F::new(0.0));
                    a[next * 1280 + row * 10 + 9] = Vector::<F, Const<4>>::new(F::new(0.0));
                    b[next * 1280 + row * 10 + 8] = Vector::<F, Const<4>>::new(F::new(0.0));
                    b[next * 1280 + row * 10 + 9] = Vector::<F, Const<4>>::new(F::new(0.0));
                }
            }
        }
        // REGISTER_STAGE_STORE_END
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 32;
    }
    hc_di_publish(
        &c0,
        output,
        injection,
        &mut injection_scratch,
        token_base,
        row_base,
        out_first,
        out_stride,
        inject_first,
        0usize,
        0usize,
        gr_down,
    );
    hc_di_publish(
        &c1,
        output,
        injection,
        &mut injection_scratch,
        token_base,
        row_base,
        out_first,
        out_stride,
        inject_first,
        0usize,
        1usize,
        gr_down,
    );
    hc_di_publish(
        &c2,
        output,
        injection,
        &mut injection_scratch,
        token_base,
        row_base,
        out_first,
        out_stride,
        inject_first,
        1usize,
        0usize,
        gr_down,
    );
    hc_di_publish(
        &c3,
        output,
        injection,
        &mut injection_scratch,
        token_base,
        row_base,
        out_first,
        out_stride,
        inject_first,
        1usize,
        1usize,
        gr_down,
    );
    if comptime!(M_FRAGMENTS == 4) {
        hc_di_publish(
            &c4,
            output,
            injection,
            &mut injection_scratch,
            token_base,
            row_base,
            out_first,
            out_stride,
            inject_first,
            2usize,
            0usize,
            gr_down,
        );
        hc_di_publish(
            &c5,
            output,
            injection,
            &mut injection_scratch,
            token_base,
            row_base,
            out_first,
            out_stride,
            inject_first,
            2usize,
            1usize,
            gr_down,
        );
        hc_di_publish(
            &c6,
            output,
            injection,
            &mut injection_scratch,
            token_base,
            row_base,
            out_first,
            out_stride,
            inject_first,
            3usize,
            0usize,
            gr_down,
        );
        hc_di_publish(
            &c7,
            output,
            injection,
            &mut injection_scratch,
            token_base,
            row_base,
            out_first,
            out_stride,
            inject_first,
            3usize,
            1usize,
            gr_down,
        );
    }
}

#[cube(launch_unchecked)]
fn hc_di_finite(
    down: &mut Array<f32>,
    injection: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    down_first: usize,
    inject_first: usize,
) {
    let group = usize::cast_from(CUBE_POS_X)
        + usize::cast_from(CUBE_POS_Y) * usize::cast_from(CUBE_COUNT_X);
    let i = group * 256 + usize::cast_from(UNIT_POS);
    if i < count * 324 {
        let t = i / 324;
        let row = i % 324;
        if row < 320 {
            let at = down_first + t * 320 + row;
            let value = down[at];
            if value.abs() <= f32::MAX {
            } else {
                down[at] = 0.0;
                fault[0].fetch_or(1);
            }
        } else {
            let at = inject_first + t * 4 + row - 320;
            let value = injection[at];
            if value.abs() <= f32::MAX {
            } else {
                injection[at] = 0.0;
                fault[0].fetch_or(1);
            }
        }
    }
}

/// Complete authenticated raw-owner validation shared by production and public
/// comparator. This does not create or claim a prompt Scope. Production checks
/// that separately before calling this function. Validate ALL suffix arguments
/// before the first prefix enqueue, including the actual old tiny workspace.
fn hc_di_bounds(
    h: &[Handle; 6],
    workspace: &Handle,
    count: usize,
    down_first: usize,
    inject_first: usize,
) -> Result<()> {
    use crate::strata_prompt::storage::overlaps;
    if !(128..=crate::strata_prompt::groups::CAPACITY).contains(&count)
        || h.iter().any(|v| v.stream != h[0].stream)
        || workspace.stream != h[0].stream
        || h[1].size_in_used() != 320 * 10240 * 2
        || h[2].size_in_used() != 4 * 10240 * 2
        || h[5].size_in_used() != 4
        || !down_first.is_multiple_of(4)
        || !inject_first.is_multiple_of(4)
        || h[..3].iter().any(|v| {
            !v.offset_start.unwrap_or(0).is_multiple_of(8) || !v.size_in_used().is_multiple_of(8)
        })
        || h[3..].iter().any(|v| {
            !v.offset_start.unwrap_or(0).is_multiple_of(4) || !v.size_in_used().is_multiple_of(4)
        })
        || !h[3].offset_start.unwrap_or(0).is_multiple_of(16)
        || !workspace.offset_start.unwrap_or(0).is_multiple_of(4)
        || !workspace.size_in_used().is_multiple_of(4)
        || workspace.size_in_used() < 16 * 10240 * 2
    {
        return Err(Error::InvalidBufferLength);
    }
    for (words, bytes, owner) in [
        (count.checked_mul(10240), 2usize, &h[0]),
        (
            count
                .checked_mul(320)
                .and_then(|n| n.checked_add(down_first)),
            4,
            &h[3],
        ),
        (
            count
                .checked_mul(4)
                .and_then(|n| n.checked_add(inject_first)),
            4,
            &h[4],
        ),
    ] {
        let end = words
            .and_then(|n| n.checked_mul(bytes))
            .ok_or(Error::ArithmeticOverflow)?;
        if u32::try_from(end).is_err() || end as u64 > owner.size_in_used() {
            return Err(Error::InvalidBufferLength);
        }
    }
    // Read-only sources may share a parent, but neither output nor the fault
    // may overlap any other live descriptor. The old suffix's mutable pad
    // workspace is also genuinely disjoint from every live descriptor.
    for i in 3..6 {
        for j in 0..i {
            if overlaps(&h[i], &h[j])? {
                return Err(Error::InvalidBufferLength);
            }
        }
    }
    for v in h {
        if overlaps(workspace, v)? {
            return Err(Error::InvalidBufferLength);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_hc_down_inject(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    workspace: Handle,
    count: usize,
    down_first: usize,
    inject_first: usize,
) -> Result<bool> {
    if count < 128 {
        return Ok(false);
    }
    let prefix = count / 128 * 128;
    let suffix = count - prefix;
    let hw = &client.properties().hardware;
    if !selected_min(
        prefix,
        10240,
        384,
        0,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
        128,
    ) || hw.max_shared_memory_size < HC_DI_SHARED_BYTES
        || THREADS != 512
        || M_FRAGMENTS != 2
        || !cfg!(feature = "strata-native-subgroup-publish")
    {
        return Ok(false);
    }
    super::native::compatible(client, true)?;
    hc_di_bounds(&h, &workspace, count, down_first, inject_first)?;
    let [input, down_weight, inject_weight, down, injection, fault] = h;
    let finish = direct_selection(&down, prefix, 324, 0, 324, hw.max_cube_count)
        .ok_or(Error::LimitExceeded("HC combined finite grid"))?;
    if hw.max_cube_count.0 < 3
        || prefix / 128 > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 == 0
    {
        return Err(Error::LimitExceeded("HC combined matrix grid"));
    }
    let tail_input = prefix_input_tail(&input, prefix, 10240)?;
    let tail_down = down_first
        .checked_add(prefix * 320)
        .ok_or(Error::ArithmeticOverflow)?;
    let tail_inject = inject_first
        .checked_add(prefix * 4)
        .ok_or(Error::ArithmeticOverflow)?;
    let vector =
        |v: Handle| unsafe { ArrayArg::from_raw_parts(v.clone(), v.size_in_used() as usize / 8) };
    // SAFETY: all real separate source/output owners and complete M128/K32,
    // SG16/512-thread, descriptor, SLM and grids were validated before enqueue.
    unsafe {
        hc_down_inject::launch_unchecked::<bf16, WgpuRuntime>(
            client,
            CubeCount::Static(3, (prefix / 128) as u32, 1),
            CubeDim::new_1d(THREADS),
            vector(input),
            vector(down_weight.clone()),
            vector(inject_weight.clone()),
            crate::resident_util::raw_arg(down.clone(), down.size_in_used() as usize / 4)?,
            crate::resident_util::raw_arg(
                injection.clone(),
                injection.size_in_used() as usize / 4,
            )?,
            10240,
            0,
            down_first,
            320,
            inject_first,
            true,
        );
        hc_di_finite::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(finish.0, finish.1, 1),
            CubeDim::new_1d(256),
            crate::resident_util::raw_arg(down.clone(), down.size_in_used() as usize / 4)?,
            crate::resident_util::raw_arg(
                injection.clone(),
                injection.size_in_used() as usize / 4,
            )?,
            crate::resident_util::raw_arg(fault.clone(), 1)?,
            prefix,
            down_first,
            inject_first,
        );
    }
    if suffix != 0 {
        // Authentic subview of the validated full normalized owner. Calling
        // packed_gr_projection here would wrongly claim the shifted view is
        // the full STAGES image; use the original checked raw primitives.
        super::native::run_dense_original(
            client,
            [tail_input.clone(), down_weight, down, fault.clone()],
            suffix,
            10240,
            320,
            0,
            tail_down,
            320,
            true,
        )?;
        super::native::run_tiny(
            client,
            [tail_input, inject_weight, workspace, injection, fault],
            suffix,
            10240,
            4,
            tail_inject,
            4,
            true,
        )?;
    }
    crate::strata_prefill_profile::count("hc_fused_down_products", 1);
    crate::strata_prefill_profile::count("hc_fused_injection_products", 1);
    if count >= 512 && suffix == 0 {
        crate::strata_prefill_profile::count("hc_fused_full_down_products", 1);
    }
    crate::strata_prefill_profile::count("hc_fused_prefix_tokens", prefix as u64);
    crate::strata_prefill_profile::count("hc_fused_suffix_tokens", suffix as u64);
    Ok(true)
}

fn hc_di_definition(producer: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    if producer {
        let input = b.input_array(Vector::<bf16, Const<4>>::as_type(&b.scope));
        let down_weight = b.input_array(Vector::<bf16, Const<4>>::as_type(&b.scope));
        let inject_weight = b.input_array(Vector::<bf16, Const<4>>::as_type(&b.scope));
        let down = b.output_array(f32::as_type(&b.scope));
        let injection = b.output_array(f32::as_type(&b.scope));
        let p: [cubecl::ir::ManagedVariable; 5] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        hc_down_inject::expand::<bf16>(
            &mut b.scope,
            input.into(),
            down_weight.into(),
            inject_weight.into(),
            down.into(),
            injection.into(),
            p[0].clone().into(),
            p[1].clone().into(),
            p[2].clone().into(),
            p[3].clone().into(),
            p[4].clone().into(),
            true,
        );
    } else {
        let down = b.output_array(f32::as_type(&b.scope));
        let injection = b.output_array(f32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let p: [cubecl::ir::ManagedVariable; 3] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        hc_di_finite::expand(
            &mut b.scope,
            down.into(),
            injection.into(),
            fault.into(),
            p[0].clone().into(),
            p[1].clone().into(),
            p[2].clone().into(),
        );
    }
    assert!(b.scope.pop_errors().is_empty());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(if producer { 512 } else { 256 })))
}

pub(super) fn validate_hc_down_inject_sources()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8 {
        for producer in [true, false] {
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
                hc_di_definition(producer),
                &options,
                ExecutionMode::Unchecked,
                cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("HC actual compiler representation".into());
            };
            if kernel.shared_size != if producer { HC_DI_SHARED_BYTES } else { 0 } {
                return Err("HC actual shared envelope".into());
            }
            let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
            let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?
                .unwrap_or(words);
            let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
            if producer {
                super::wide_pipeline::validate_program(&words)?;
            }
            let subgroup = wgpu::hal::vulkan::kuno_cooperative_subgroup(&words);
            if (producer && subgroup != Some(16)) || (!producer && subgroup.is_some()) {
                return Err("HC actual cooperative subgroup".into());
            }
            let mut classes = std::collections::BTreeMap::new();
            let mut parents = std::collections::BTreeMap::new();
            let (mut multiply, mut atomic, mut roots) = (0, 0, Vec::new());
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if n == 0 || at + n > words.len() {
                    return Err("HC SPIRV opcode extent".into());
                }
                match words[at] & 65535 {
                    59 => {
                        classes.insert(words[at + 2], words[at + 3]);
                    }
                    65 | 66 | 67 | 83 | 124 => {
                        parents.insert(words[at + 2], words[at + 3]);
                    }
                    4458 => roots.push(words[at + 1]),
                    4459 => multiply += 1,
                    241 => atomic += 1,
                    _ => {}
                }
                at += n;
            }
            if producer {
                if multiply != 8 || atomic != 0 || roots.len() != 6 {
                    return Err("HC original matrix products/publication/fault separation".into());
                }
                let (mut global, mut shared) = (0, 0);
                for mut p in roots {
                    let mut seen = std::collections::BTreeSet::new();
                    while let Some(&parent) = parents.get(&p) {
                        if !seen.insert(p) {
                            return Err("HC pointer cycle".into());
                        }
                        p = parent;
                    }
                    match classes.get(&p) {
                        Some(12) => global += 1,
                        Some(4) => shared += 1,
                        _ => return Err("HC cooperative output storage class".into()),
                    }
                }
                if global != 4 || shared != 2 {
                    return Err("HC direct down and real SG injection scratch publication".into());
                }
            } else if multiply != 0 || atomic != 2 {
                return Err("HC combined original finite/sticky fault branches".into());
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
                .ok_or("HC SPIRV validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("HC combined actual SPIRV invalid".into());
            }
            println!(
                "strata_hc_down_inject_shader_accepted producer={producer} layout_mask={mask}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod hc_di_tests {
    #[test]
    fn whole_staging_map_only_reads_two_authentic_weight_owners() {
        let mut a = vec![0u8; 128 * 32];
        let mut down = vec![0u8; 320 * 32];
        let mut injection = vec![0u8; 4 * 32];
        let mut padding = 0;
        for tile in 0..3 {
            for job in 0..512 {
                for off in 0..8 {
                    let row = job / 4;
                    let k = (job % 4) * 8 + off;
                    a[row * 32 + k] += 1;
                    let n = tile * 128 + row;
                    if n < 320 {
                        down[n * 32 + k] += 1;
                    } else if n < 324 {
                        injection[(n - 320) * 32 + k] += 1;
                    } else {
                        padding += 1;
                    }
                }
            }
        }
        assert!(a.iter().all(|&n| n == 3));
        assert!(down.iter().all(|&n| n == 1));
        assert!(injection.iter().all(|&n| n == 1));
        assert_eq!(padding, 60 * 32);
    }
    #[test]
    fn partial_injection_fragments_publish_once_and_scratch_slots_are_disjoint() {
        let mut down = vec![0u8; 128 * 320];
        let mut injection = vec![0u8; 128 * 4];
        for tile in 0..3 {
            for sg in 0..32 {
                for tm in 0..2 {
                    for tn in 0..2 {
                        let token = (sg / 4) * 16 + tm * 8;
                        let row = tile * 128 + (sg % 4) * 32 + tn * 16;
                        for i in 0..128 {
                            if row < 320 {
                                down[(token + i / 16) * 320 + row + i % 16] += 1;
                            } else if row == 320 && i % 16 < 4 {
                                assert_eq!(sg % 4, 2);
                                assert_eq!(tn, 0);
                                assert!((sg / 4) * 128 + i < 1024);
                                injection[(token + i / 16) * 4 + i % 16] += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(down.iter().chain(&injection).all(|&n| n == 1));
        assert_eq!(40960 + 8 * 128 * 4, super::HC_DI_SHARED_BYTES);
        assert!(super::HC_DI_SHARED_BYTES <= super::SHARED_BYTES);
        for count in 128..=4096 {
            assert_eq!(count / 128 * 128 + count % 128, count);
        }
    }
    #[test]
    fn whole_call_checks_reject_source_output_workspace_alias_and_short_owners() {
        use super::*;
        use cubecl_common::stream_id::StreamId;
        let own = |n| Handle::new(StreamId::current(), n);
        let count = 148usize;
        let h = [
            own((count * 10240 * 2) as u64),
            own(320 * 10240 * 2),
            own(4 * 10240 * 2),
            own((count * 320 * 4) as u64),
            own((count * 4 * 4) as u64),
            own(4),
        ];
        let workspace = own(16 * 10240 * 2);
        assert!(hc_di_bounds(&h, &workspace, count, 0, 0).is_ok());
        for i in 0..6 {
            let mut short = h.clone();
            short[i] = own(h[i].size_in_used() - 1);
            assert!(hc_di_bounds(&short, &workspace, count, 0, 0).is_err());
        }
        let mut alias = h.clone();
        alias[4] = h[3].clone();
        assert!(hc_di_bounds(&alias, &workspace, count, 0, 0).is_err());
        assert!(hc_di_bounds(&h, &h[0], count, 0, 0).is_err());
        assert!(hc_di_bounds(&h, &workspace, count, 4, 0).is_err());
        assert!(hc_di_bounds(&h, &workspace, count, 0, 4).is_err());
        assert!(hc_di_bounds(&h, &workspace, 127, 0, 0).is_err());
        assert!(hc_di_bounds(&h, &workspace, 4097, 0, 0).is_err());
    }
    #[test]
    fn actual_compiler_retains_owners_products_and_sg_scratch_contract() {
        super::validate_hc_down_inject_sources().unwrap();
    }
}
